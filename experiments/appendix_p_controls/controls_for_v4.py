# =============================================================================
# Drop-in cells for v4-results. Paste after the setup cells (1-14) have run.
# Uses only names defined in the notebook: train, make_task, Store, cached,
# set_seed, S, SEEDS_MAIN, IS_VISUAL, MODALITY, CFG, DEVICE, csv, F, nn, np, pd,
# math, copy, torch. Written against the uploaded code but NOT executed here,
# so run PRESET="smoke" first.
# =============================================================================


# ---- CELL A: shared helpers -------------------------------------------------
def _rng_fork():
    devs = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    return torch.random.fork_rng(devices=devs)


@torch.no_grad()
def fixed_batches(task, load, n=16, seed=12345):
    """Same trials for every evaluation, without disturbing the training RNG."""
    with _rng_fork():
        torch.manual_seed(seed)
        return [task.generate(CFG.batch_size, load) for _ in range(n)]


@torch.no_grad()
def route_share(m, batches):
    m.eval(); sh, sc = [], []
    for b in batches:
        p = m(b["x"])["parts"]
        sh.append(p["hidden"].std(-1)); sc.append(p["context"].std(-1))
    m.train()
    h, c = torch.cat(sh).mean().item(), torch.cat(sc).mean().item()
    return c / (h + c + 1e-9)


@torch.no_grad()
def paired_metrics(m, batches):
    """Clean and ablated accuracy on the SAME trials (paired), with both
    resample and mean ablation, plus single-route accuracy (the other route's
    logits simply dropped: this is attention deletion without retraining)."""
    m.eval()
    chance = 1.0 / m.w_h.out_features

    def acc(iv=None, perm_seed=None):
        c = t = 0
        with _rng_fork():
            for i, b in enumerate(batches):
                if perm_seed is not None:
                    torch.manual_seed(perm_seed + i)
                c += (m(b["x"], intervene=iv)["logits"].argmax(-1) == b["y"]).sum().item()
                t += b["y"].numel()
        return c / t

    a = acc()
    den = max(a - chance, 1e-6)
    D = lambda x: float(np.clip((a - x) / den, 0, 1))
    out = {"acc": a}
    for mode in ("shuffle", "mean"):
        out[f"Dh_{mode}"] = D(acc({"hidden": (mode, 1.0)}, perm_seed=777))
        out[f"Dc_{mode}"] = D(acc({"context": (mode, 1.0)}, perm_seed=777))
    ch = cc = t = 0
    for b in batches:
        p = m(b["x"])["parts"]
        ch += (p["hidden"].argmax(-1) == b["y"]).sum().item()
        cc += (p["context"].argmax(-1) == b["y"]).sum().item()
        t += b["y"].numel()
    out["acc_recurrent_only"], out["acc_retrieval_only"] = ch / t, cc / t
    m.train()
    out["share"] = route_share(m, batches)
    return out


@torch.no_grad()
def probe_hT(m, task, load, n_fit=40, n_test=10, seed=999, ridge=1e-2):
    """Ridge least-squares probe for the target from h_T. Weaker than a logistic
    probe but deterministic and identical across arms; swap in your existing
    probe if you prefer consistency with Table 10."""
    def feats(bs):
        X, Y = [], []
        for b in bs:
            X.append(m(b["x"], return_internals=True)["h_T"]); Y.append(b["y"])
        return torch.cat(X), torch.cat(Y)
    m.eval()
    Xf, Yf = feats(fixed_batches(task, load, n_fit, seed))
    Xt, Yt = feats(fixed_batches(task, load, n_test, seed + 1))
    m.train()
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
    one = lambda X: torch.cat([(X - mu) / sd, torch.ones(len(X), 1, device=X.device)], 1)
    Xf, Xt = one(Xf), one(Xt)
    Yoh = F.one_hot(Yf, m.w_h.out_features).float()
    W = torch.linalg.solve(Xf.T @ Xf + ridge * torch.eye(Xf.shape[1], device=Xf.device),
                           Xf.T @ Yoh)
    return float(((Xt @ W).argmax(-1) == Yt).float().mean())


# ---- CELL B: readout-only vs core refit vs late price ----------------------
# Question a reviewer will ask: after the free model solves the task through
# retrieval, is "steering" just retraining the recurrent READOUT on information
# h_T already carries?
#
#   late_price         everything trainable, CE + lam*||lc||   (the late arm; it starts
#                      a fresh AdamW at the switch like the other arms, so its
#                      numbers can differ slightly from Table 5)
#   readout_only@a     w_c scaled by a and frozen; only proj_h, w_h train; CE only
#   core_refit@a       w_c scaled by a and frozen; emb, rnn, proj_h, w_h train; CE only
#
# a = "match": retrieval share matched to late_price's final share (your post-hoc
#               attenuation, but with retraining allowed)
# a = 0      : retrieval deleted, then refit
#
# Reading it:
#   readout_only reaches late_price accuracy  -> the cheap story holds
#   readout_only fails, core_refit succeeds   -> the core had to reorganise
#   core_refit@0 matches late_price           -> "why price instead of delete +
#       fine-tune?" becomes a question you must answer (controllability and
#       reversibility still stand; the scaffold framing weakens)

RC_LOAD = {"abstract": 8, "mnist": 4, "cifar10": 2}[MODALITY]
RC_LAM = {"abstract": 0.03, "mnist": 0.0997, "cifar10": 0.0194}[MODALITY]  # four-arm prices
RC_PRE, RC_POST = 2000, S(10000, 3000, 300)     # same switch and post-switch budget as four-arm
RC_SEEDS = list(range(8)) if IS_VISUAL else SEEDS_MAIN
ALL = ("emb.", "rnn.", "proj_h.", "w_h.", "q.", "k.", "v.", "proj_c.", "w_c.")
READOUT = ("proj_h.", "w_h.")
CORE = ("emb.", "rnn.", "proj_h.", "w_h.")


def refit(m, task, load, steps, trainable, lam=0.0, seed=0, batches=None, log_every=1000):
    set_seed(10_000 + seed)
    for n, p in m.named_parameters():
        p.requires_grad_(n.startswith(trainable))
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=CFG.lr, weight_decay=CFG.weight_decay)
    trace = []
    for step in range(1, steps + 1):
        m.train()
        b = task.generate(CFG.batch_size, load)
        o = m(b["x"])
        loss = F.cross_entropy(o["logits"], b["y"])
        if lam > 0:
            loss = loss + lam * o["parts"]["context"].norm(dim=-1).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(params, CFG.grad_clip); opt.step()
        if step % log_every == 0 or step == steps:
            trace.append({"step": step, **paired_metrics(m, batches)})
    for p in m.parameters():
        p.requires_grad_(True)
    return m, trace


@torch.no_grad()
def match_alpha(m, w_c0, target, batches, lo=1e-6, hi=1.0, iters=30):
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        m.w_c.weight.copy_(mid * w_c0)
        if route_share(m, batches) > target:
            hi = mid
        else:
            lo = mid
    a = math.sqrt(lo * hi); m.w_c.weight.copy_(a * w_c0)
    return a


task = make_task("selective_recall")
EVB = fixed_batches(task, RC_LOAD)
st = Store("readout_control")
ARMS_RC = ["late_price", "readout_only@match", "core_refit@match",
           "readout_only@0", "core_refit@0"]

for sd in RC_SEEDS:
    specs = {arm: {"block": "readout_control", "arm": arm, "load": RC_LOAD, "lam": RC_LAM,
                   "seed": sd, "pre": RC_PRE, "post": RC_POST} for arm in ARMS_RC}
    if all(st.has(s) for s in specs.values()):
        continue
    base, _ = train(task, RC_LOAD, lam=0.0, seed=sd, steps=RC_PRE, schedule="constant")
    base_metrics = paired_metrics(base, EVB)
    base_probe = probe_hT(base, task, RC_LOAD)

    def _late(sd=sd):
        m, tr = refit(copy.deepcopy(base), task, RC_LOAD, RC_POST, ALL, lam=RC_LAM,
                      seed=sd, batches=EVB)
        return {**tr[-1], "probe_hT": probe_hT(m, task, RC_LOAD), "alpha": 1.0,
                "base": base_metrics, "base_probe_hT": base_probe, "trace": tr}

    late = cached(st, specs["late_price"], _late, label=f"late_price seed {sd}")

    for arm in ARMS_RC[1:]:
        kind, a_spec = arm.split("@")

        def _go(kind=kind, a_spec=a_spec, sd=sd):
            m = copy.deepcopy(base)
            w_c0 = m.w_c.weight.detach().clone()
            if a_spec == "0":
                alpha = 0.0
                with torch.no_grad():
                    m.w_c.weight.zero_()
            else:
                alpha = match_alpha(m, w_c0, late["share"], EVB)
            at_start = paired_metrics(m, EVB)          # post-hoc, before any refit
            m, tr = refit(m, task, RC_LOAD, RC_POST,
                          READOUT if kind == "readout_only" else CORE,
                          seed=sd, batches=EVB)
            return {**tr[-1], "probe_hT": probe_hT(m, task, RC_LOAD), "alpha": alpha,
                    "at_start": at_start, "base": base_metrics,
                    "base_probe_hT": base_probe, "trace": tr}

        cached(st, specs[arm], _go, label=f"{arm:<20} seed {sd}")

RC = st.frame()
RC.to_csv(csv("readout_control"), index=False)
cols = ["acc", "acc_recurrent_only", "Dh_shuffle", "Dc_shuffle", "Dh_mean", "Dc_mean",
        "share", "probe_hT"]
print(f"\nreadout control [{MODALITY}] load {RC_LOAD}, lam {RC_LAM}, "
      f"{RC_POST} refit steps, n={RC.seed.nunique()} seeds (mean ± sd):\n")
summ = RC.groupby("arm")[cols].agg(["mean", "std"])
print(summ.reindex(ARMS_RC).round(3).to_string())
print(f"\nfree model at the switch: probe_hT {RC.base_probe_hT.mean():.3f} "
      f"(chance {1 / (10 if IS_VISUAL else CFG.n_symbols):.3f})")


# ---- CELL C: late-switch negative controls ---------------------------------
# The late arm is the true post-convergence test. Tables 4/5 need a late-arm
# run at the load where the boundary says inversion should FAIL. Four-arm runs
# took ~1 min each in cell 33, so this block is minutes.
LC_CONDS = {"abstract": [(8, 0.03), (12, 0.03), (16, 0.03)],
            "mnist": [(4, 0.0997), (8, 0.0997), (4, 0.0498), (8, 0.0498)],
            "cifar10": [(2, 0.0194), (3, 0.0194), (2, 0.0097), (3, 0.0097)]}[MODALITY]
LC_SW, LC_ST = 2000, S(12000, 4000, 400)
st = Store("late_control")
for (L, lam), sd in itertools.product(LC_CONDS, RC_SEEDS):
    spec = {"block": "late_control", "load": L, "lam": lam, "seed": sd,
            "switch": LC_SW, "steps": LC_ST}

    def _go(L=L, lam=lam, sd=sd):
        tk = make_task("selective_recall")
        m, tr = train(tk, L, lam=0.0, lam_after=lam, switch=LC_SW, seed=sd, steps=LC_ST,
                      schedule="constant", log_every=500)
        pre = tr[tr.step <= LC_SW].iloc[-1]
        return {"acc_at_switch": float(pre.acc_clean),
                "Dh_at_switch": float(pre.hidden_dep), "Dc_at_switch": float(pre.attn_dep),
                "share_at_switch": float(pre.context_share),
                **paired_metrics(m, fixed_batches(tk, L)),
                "trace": tr.to_dict("records")}

    cached(st, spec, _go, label=f"late L={L} lam={lam} seed {sd}")

LC = st.frame()
LC.to_csv(csv("late_control"), index=False)
LC["inverted_end"] = LC.Dh_shuffle > LC.Dc_shuffle
LC["inverted_end_mean_abl"] = LC.Dh_mean > LC.Dc_mean
print(f"\nlate-switch controls [{MODALITY}] (paired end-of-run evaluation):")
print(LC.groupby(["lam", "load"]).agg(
    n=("seed", "size"), acc_at_switch=("acc_at_switch", "mean"),
    Dc_at_switch=("Dc_at_switch", "mean"), acc=("acc", "mean"),
    Dh=("Dh_shuffle", "mean"), Dc=("Dc_shuffle", "mean"),
    inverted=("inverted_end", "sum"), inverted_mean_abl=("inverted_end_mean_abl", "sum"),
    rec_only_acc=("acc_recurrent_only", "mean")).round(3).to_string())


# ---- CELL D: recount existing inversions by end state (no compute) ---------
# The current count marks a seed "inverted" if Dh > Dc at ANY logged checkpoint
# after convergence. With unpaired 1,024-trial evaluations the per-checkpoint
# SE of Dh - Dc is ~0.025, so near-tied seeds can cross on noise alone.
def recount(traces, keys, conv=None, last_k=3):
    rows = []
    for k, g in traces.groupby(keys):
        s0 = g.step.min()
        if conv is not None:
            curve = g.groupby("step").acc_clean.mean()
            hit = curve.index[curve >= conv]
            s0 = hit.min() if len(hit) else np.inf
        for sd, gs in g[g.step >= s0].groupby("seed"):
            gs = gs.sort_values("step"); gap = (gs.hidden_dep - gs.attn_dep).values
            rows.append({**dict(zip(keys, k if isinstance(k, tuple) else (k,))),
                         "seed": sd, "ever": bool((gap > 0).any()),
                         "final": bool(gap[-1] > 0),
                         "last_k": bool(gap[-last_k:].mean() > 0)})
    R = pd.DataFrame(rows)
    return R.groupby(keys).agg(n=("seed", "size"), ever=("ever", "sum"),
                               final=("final", "sum"), last_k=("last_k", "sum"))


# symbolic used 0.99 and MNIST 0.95; CIFAR used 0.99 x plateau (~0.955), approximated here
_conv = {"abstract": 0.99, "mnist": 0.95, "cifar10": 0.95}[MODALITY]
print("\ninversion counts: ever-crossed vs final checkpoint vs mean of last 3")
print(recount(pd.read_csv(csv("p8_inversion").replace(".csv", "_traces.csv")),
              ["lam", "load"], conv=_conv).to_string())
_sw = pd.read_csv(csv("p9_switch_traces"))
print(recount(_sw[_sw.step > 2000], ["arm", "load"]).to_string())
