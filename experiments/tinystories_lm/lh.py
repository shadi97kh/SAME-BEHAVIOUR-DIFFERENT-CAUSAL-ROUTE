"""Route handover in a from-scratch recurrent-attention language model on TinyStories.

Core library for the natural-language extension (Appendix R). One training job per process:

    python lh.py free    --ctx 64 --dh 512 --seed 0   # free hybrid (lambda = 0), trained to plateau
    python lh.py reconly --ctx 64 --dh 512 --seed 0   # recurrent-only screen model
    python lh.py arm     --ctx 64 --dh 512 --seed 0 --name late0.1 --lam 0.1   # continue a free checkpoint
    python lh.py bench   --ctx 64 --dh 512            # 30-step timing benchmark

Output directory: $LH_OUT (default ./runs_fast); optional config overrides in $LH_OUT/cfg.json.
run_experiments.py schedules every job reported in the paper; analyze.py rebuilds all tables and
the figure from the logs. The computational code is unchanged from the version used for the paper."""
import argparse, os, sys, json, math, time, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

OUT = os.environ.get("LH_OUT", "runs_fast")
DATA, LOGD, CKD = (os.path.join(OUT, p) for p in ("data", "logs", "ckpt"))
CFG = dict(
    vocab_keep=8192,                  # keep the most frequent GPT-2 tokens, rest -> UNK (6x cheaper output layer)
    d=512, n_trunk=4, d_ff=2048, n_heads=8, d_c=512,
    tokens_per_step=8192,             # batch = tokens_per_step // ctx  (same tokens/step at every ctx)
    lr=3e-4, warmup=500, clip=1.0,
    eval_every=1000, eval_tokens=160_000, ckpt_every=1000,
    min_free_steps=6000, max_free_steps=20000, plateau_window=3, plateau_tol=0.015,
    price_steps=6000,
    repeat_k=32, repeat_trials=512,
)
dev = "cuda"


def cond(ctx, dh=None):
    dh = int(dh or CFG["d"])
    return f"c{ctx}" + ("" if dh == CFG["d"] else f"h{dh}")


def load_cfg():
    p = os.path.join(OUT, "cfg.json")
    if os.path.exists(p): CFG.update(json.load(open(p)))
    for d in (DATA, LOGD, CKD): os.makedirs(d, exist_ok=True)
    return CFG


def amp_dtype():
    # T4 (sm75) has no fast bf16 -> use fp16 + GradScaler there; bf16 on A100/H100
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ----------------------------------------------------------------------------- data
def prepare(n_stories=400_000, raw_cache=None, num_proc=4):
    """Tokenise once (optionally from a cached GPT-2 token array), remap the vocabulary, build the bigram predictor."""
    load_cfg()
    mp = os.path.join(DATA, "meta.json")
    if os.path.exists(mp):
        m = json.load(open(mp)); print({k: v for k, v in m.items() if k != "keep_gpt2_ids"}); return m
    if raw_cache and os.path.exists(raw_cache):
        raw = np.load(raw_cache); print("reusing", raw_cache)
    else:
        from datasets import load_dataset
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2"); eos = tok.eos_token_id
        ds = load_dataset("roneneldan/TinyStories", split="train").select(range(n_stories))
        ds = ds.map(lambda b: {"ids": [t + [eos] for t in tok(b["text"])["input_ids"]]},
                    batched=True, remove_columns=ds.column_names, num_proc=num_proc)
        raw = np.concatenate([np.asarray(x, dtype=np.uint16) for x in ds["ids"]])
    eos, K = 50256, CFG["vocab_keep"]
    counts = np.bincount(raw, minlength=50257)
    keep = np.argsort(-counts, kind="stable")[:K - 1]
    if eos not in keep: keep = np.concatenate([keep[:-1], [eos]])
    unk = K - 1
    lut = np.full(50257, unk, dtype=np.uint16); lut[keep] = np.arange(len(keep), dtype=np.uint16)
    tokens = lut[raw]
    cov = float(counts[keep].sum() / counts.sum())
    n_val = int(0.02 * len(tokens)); train = tokens[:-n_val]
    V = K; bc = np.zeros(V * V, dtype=np.int64); ch = 20_000_000
    for i in range(0, len(train) - 1, ch):
        a = train[i:i + ch + 1].astype(np.int64)
        bc += np.bincount(a[:-1] * V + a[1:], minlength=V * V)
    big = bc.reshape(V, V).argmax(1).astype(np.int32); del bc
    np.save(os.path.join(DATA, "tokens.npy"), tokens); np.save(os.path.join(DATA, "bigram.npy"), big)
    m = dict(V=V, unk=unk, eos=int(lut[eos]), coverage=cov, n_tokens=int(len(tokens)), n_val=n_val,
             keep_gpt2_ids=[int(t) for t in keep])
    json.dump(m, open(mp, "w"))
    print(f"V={V}  token coverage={cov:.4f}  tokens={len(tokens)/1e6:.1f}M")
    if cov < 0.995: print("WARNING: coverage < 99.5% -> set vocab_keep=16384 and rerun prepare (delete data/)")
    return m


class Data:
    def __init__(s, ctx):
        m = json.load(open(os.path.join(DATA, "meta.json")))
        s.V, s.unk, s.ctx = m["V"], m["unk"], ctx
        tok = np.load(os.path.join(DATA, "tokens.npy"), mmap_mode="r")
        s.train, s.val = tok[:-m["n_val"]], tok[-m["n_val"]:]
        s.big = np.load(os.path.join(DATA, "bigram.npy"))
        s.B = CFG["tokens_per_step"] // ctx
        rng = np.random.default_rng(1234)
        nb = max(1, CFG["eval_tokens"] // (s.B * ctx))
        s.VAL = [s.batch(s.val, rng) for _ in range(nb)]
        s.MASK, s.HARD = zip(*[s.masks(x, y) for x, y in s.VAL])
        k, N = CFG["repeat_k"], CFG["repeat_trials"]
        r = np.random.default_rng(99).integers(50, 3000, size=(N, k))
        seq = torch.from_numpy(np.concatenate([r, r], 1).astype(np.int64)).to(dev)
        s.REP_X, s.REP_Y = seq[:, :-1], seq[:, 1:]
        s.REP_M = torch.zeros_like(s.REP_X, dtype=torch.bool); s.REP_M[:, k:] = True   # second copy: next token is copyable

    def batch(s, arr, rng, B=None):
        B = B or s.B; T = s.ctx
        ix = rng.integers(0, len(arr) - T - 1, size=B)
        x = torch.from_numpy(np.stack([np.asarray(arr[i:i + T + 1]) for i in ix]).astype(np.int64)).to(dev)
        return x[:, :-1], x[:, 1:]

    def masks(s, x, y):
        # copy position: x_t seen earlier and its most recent continuation == y_t (UNK excluded)
        # hard copy: additionally the bigram predictor gets y_t wrong
        xn, yn = x.cpu().numpy(), y.cpu().numpy()
        m = np.zeros_like(xn, dtype=bool)
        for b in range(xn.shape[0]):
            seen = {}
            for t in range(xn.shape[1]):
                cur, nxt = xn[b, t], yn[b, t]
                if cur != s.unk and nxt != s.unk and seen.get(cur, -1) == nxt: m[b, t] = True
                seen[cur] = nxt
        hard = m & (s.big[xn] != yn)
        return torch.from_numpy(m).to(dev), torch.from_numpy(hard).to(dev)


# ----------------------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(s, d, dff):
        super().__init__(); s.ln = nn.LayerNorm(d); s.fc1 = nn.Linear(d, dff); s.fc2 = nn.Linear(dff, d)
    def forward(s, x): return x + s.fc2(F.gelu(s.fc1(s.ln(x))))


class Hybrid(nn.Module):
    """logits_t = W f(h_t) + W g(c_t); c_t = Attn(q=h_t, K/V = h_{<t}) with a learned sink (dual-route readout of Sec. 3)."""
    def __init__(s, V, retrieval=True, dh=None):
        super().__init__(); c = CFG; d = c["d"]; dh = int(dh or d); s.dh = dh
        s.emb = nn.Embedding(V, d); nn.init.normal_(s.emb.weight, std=0.02)
        s.trunk = nn.ModuleList([Block(d, c["d_ff"]) for _ in range(c["n_trunk"])])
        s.ln_in = nn.LayerNorm(d); s.gru = nn.GRU(d, dh, batch_first=True)
        s.f = nn.Sequential(nn.LayerNorm(dh), nn.Linear(dh, d), nn.GELU())
        s.retrieval = retrieval
        if retrieval:
            s.nh, s.d_c = c["n_heads"], c["d_c"]
            s.q, s.k, s.v = (nn.Linear(dh, s.d_c, bias=False) for _ in range(3))
            s.sink_k = nn.Parameter(torch.zeros(1, 1, s.d_c)); s.sink_v = nn.Parameter(torch.zeros(1, 1, s.d_c))
            s.o = nn.Linear(s.d_c, s.d_c, bias=False)
            s.g = nn.Sequential(nn.LayerNorm(s.d_c), nn.Linear(s.d_c, d), nn.GELU())
        s.out = nn.Linear(d, V, bias=False); s.out.weight = s.emb.weight

    def states(s, x):
        h = s.emb(x)
        for b in s.trunk: h = b(h)
        with torch.autocast("cuda", enabled=False):
            h, _ = s.gru(s.ln_in(h).float())
        return h

    def context(s, h):
        B, T, _ = h.shape
        q = s.q(h); k = s.k(h); v = s.v(h)
        k = torch.cat([s.sink_k.expand(B, 1, -1).to(k.dtype), k], 1)
        v = torch.cat([s.sink_v.expand(B, 1, -1).to(v.dtype), v], 1)
        q, k, v = (t.view(B, -1, s.nh, s.d_c // s.nh).transpose(1, 2) for t in (q, k, v))
        mask = torch.ones(T, T + 1, dtype=torch.bool, device=h.device).tril(0)   # col0 = sink, col j = state j-1 < t
        att = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return s.o(att.transpose(1, 2).reshape(B, T, s.d_c))

    def forward(s, x, perm_h=None, perm_c=None, beta=1.0, retrieval_on=True):
        h = s.states(x)
        c = s.context(h) if (s.retrieval and retrieval_on) else None    # clean context before any permutation
        lh = s.out(s.f(h if perm_h is None else h[perm_h]))
        if c is None: return lh, None
        return lh, beta * s.out(s.g(c if perm_c is None else c[perm_c]))


def make(V, retrieval=True, dh=None):
    return Hybrid(V, retrieval, dh).to(dev)


def loss_fn(lh, lc, y, lam, V):
    logits = lh if lc is None else lh + lc
    ce = F.cross_entropy(logits.reshape(-1, V).float(), y.reshape(-1))
    price = torch.zeros((), device=y.device) if lc is None else lc.float().norm(dim=-1).mean() / math.sqrt(V)
    return ce + lam * price, ce, price


# ----------------------------------------------------------------------------- metrics
@torch.no_grad()
def evaluate(model, D, perm=None, retrieval_on=True, beta=1.0):
    was = model.training; model.eval()
    g = torch.Generator(device=dev).manual_seed(0)
    tot = n = cc = cn = hc = hn = 0; sh, sc = [], []
    for (x, y), m, hm in zip(D.VAL, D.MASK, D.HARD):
        B = x.shape[0]
        ph = torch.randperm(B, generator=g, device=dev) if perm == "h" else None
        pc = torch.randperm(B, generator=g, device=dev) if perm == "c" else None
        with torch.autocast("cuda", dtype=amp_dtype()):
            lh, lc = model(x, ph, pc, beta, retrieval_on)
        logits = (lh if lc is None else lh + lc).float()
        tot += F.cross_entropy(logits.reshape(-1, D.V), y.reshape(-1), reduction="sum").item(); n += y.numel()
        pred = logits.argmax(-1)
        cc += (pred[m] == y[m]).sum().item(); cn += m.sum().item()
        hc += (pred[hm] == y[hm]).sum().item(); hn += hm.sum().item()
        if lc is not None and perm is None:
            sh.append(lh.float().std(-1).mean().item()); sc.append(lc.float().std(-1).mean().item())
    model.train(was)
    out = dict(loss=tot / n, ppl=math.exp(tot / n), copy_acc=cc / max(cn, 1), hard_acc=hc / max(hn, 1))
    if sh: out["share_c"] = float(np.mean(sc) / (np.mean(sh) + np.mean(sc)))
    return out


@torch.no_grad()
def repeat_acc(model, D, retrieval_on=True):
    was = model.training; model.eval(); c = n = 0
    for i in range(0, D.REP_X.shape[0], 128):
        x, y, m = D.REP_X[i:i + 128], D.REP_Y[i:i + 128], D.REP_M[i:i + 128]
        with torch.autocast("cuda", dtype=amp_dtype()):
            lh, lc = model(x, retrieval_on=retrieval_on)
        pred = (lh if lc is None else lh + lc).float().argmax(-1)
        c += (pred[m] == y[m]).sum().item(); n += m.sum().item()
    model.train(was); return c / max(n, 1)


def route_report(model, D):
    cl = evaluate(model, D); a = cl["copy_acc"]
    D_ = lambda ar: float(np.clip((a - ar) / max(a, 1e-6), 0, 1))
    rh = evaluate(model, D, perm="h")
    rep = dict(ppl=cl["ppl"], loss=cl["loss"], copy_acc=a, hard_acc=cl["hard_acc"], share_c=cl.get("share_c", float("nan")),
               D_h=D_(rh["copy_acc"]), dloss_h=rh["loss"] - cl["loss"], rep_acc=repeat_acc(model, D))
    if model.retrieval:
        rc = evaluate(model, D, perm="c"); de = evaluate(model, D, retrieval_on=False)
        rep.update(D_c=D_(rc["copy_acc"]), dloss_c=rc["loss"] - cl["loss"],
                   del_ppl=de["ppl"], del_ppl_ratio=de["ppl"] / cl["ppl"], del_copy=de["copy_acc"],
                   del_cost=a - de["copy_acc"], del_hard_cost=cl["hard_acc"] - de["hard_acc"],
                   del_rep_acc=repeat_acc(model, D, retrieval_on=False))
    return rep


# ----------------------------------------------------------------------------- training with resumable checkpoints
def _ck(tag): return os.path.join(CKD, f"{tag}.pt")
def _part(tag): return os.path.join(CKD, f"{tag}.part.pt")
def _log(tag): return os.path.join(LOGD, f"{tag}.jsonl")


def _save(path, model, opt, scaler, meta):
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(), meta=meta), path + ".tmp")
    os.replace(path + ".tmp", path)


def _load(path): return torch.load(path, map_location=dev, weights_only=False)


def _new_opt(model): return torch.optim.AdamW(model.parameters(), lr=CFG["lr"], betas=(0.9, 0.95), weight_decay=0.0)
def _new_scaler(): return torch.amp.GradScaler("cuda", enabled=amp_dtype() == torch.float16)


def _truncate_log(tag, step):
    p = _log(tag)
    if not os.path.exists(p): return
    keep = [l for l in open(p) if l.strip() and json.loads(l)["step"] < step]
    open(p, "w").writelines(keep)


def train(model, D, tag, lam, seed, start, n_steps, opt=None, scaler=None, plateau=False):
    opt = opt or _new_opt(model); scaler = scaler or _new_scaler()
    rng = np.random.default_rng(seed * 100_003 + start)
    step, hist, end = start, [], start + n_steps
    if os.path.exists(_part(tag)):
        C = _load(_part(tag)); model.load_state_dict(C["model"]); opt.load_state_dict(C["opt"]); scaler.load_state_dict(C["scaler"])
        step, hist = C["meta"]["step"], C["meta"]["hist"]; rng.bit_generator.state = C["meta"]["rng"]
        print(f"[{tag}] resuming at step {step}", flush=True)
    else:
        open(_log(tag), "w").close()      # fresh run -> fresh log (no mixing with aborted runs)
    _truncate_log(tag, step)
    logf = open(_log(tag), "a"); t0 = time.time(); model.train(); plateaued = False
    while step < end:
        for gp in opt.param_groups: gp["lr"] = CFG["lr"] * min(1.0, (step + 1) / CFG["warmup"])
        x, y = D.batch(D.train, rng)
        with torch.autocast("cuda", dtype=amp_dtype()):
            lh, lc = model(x)
        loss, ce, price = loss_fn(lh, lc, y, lam, D.V)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["clip"]); scaler.step(opt); scaler.update()
        if step % CFG["eval_every"] == 0 or step == end - 1:
            rep = route_report(model, D)
            rep.update(step=step, tag=tag, lam=lam, train_ce=ce.item(), price=price.item(), sec=time.time() - t0)
            logf.write(json.dumps(rep) + "\n"); logf.flush(); hist.append(rep["ppl"])
            print(f"[{tag}] {step:6d} ppl {rep['ppl']:.3f} copy {rep['copy_acc']:.3f} hard {rep['hard_acc']:.3f} "
                  f"share_c {rep['share_c']:.3f} D_c {rep.get('D_c', float('nan')):.2f} "
                  f"del_ppl_ratio {rep.get('del_ppl_ratio', float('nan')):.3f} rep {rep['rep_acc']:.3f}", flush=True)
            w = CFG["plateau_window"]
            if plateau and step >= CFG["min_free_steps"] and len(hist) > w and \
                    (hist[-1 - w] - hist[-1]) / hist[-1 - w] < CFG["plateau_tol"]:
                plateaued = True; step += 1; break
        step += 1
        if step % CFG["ckpt_every"] == 0 and step < end:
            _save(_part(tag), model, opt, scaler, dict(step=step, hist=hist, rng=rng.bit_generator.state))
    logf.close()
    _save(_ck(tag), model, opt, scaler, dict(step=step, hist=hist, plateaued=plateaued))
    if os.path.exists(_part(tag)): os.remove(_part(tag))
    print(f"[{tag}] done at step {step} (plateau reached: {plateaued})", flush=True)


# ----------------------------------------------------------------------------- jobs
def job_free(ctx, seed, retrieval, dh=None):
    tag = f"{cond(ctx, dh)}_{'free' if retrieval else 'reconly'}_s{seed}"
    if os.path.exists(_ck(tag)): print("exists", tag); return
    set_seed(seed); D = Data(ctx); m = make(D.V, retrieval, dh)
    print(f"[{tag}] {sum(p.numel() for p in m.parameters())/1e6:.2f}M params, batch {D.B}", flush=True)
    train(m, D, tag, 0.0, seed, 0, CFG["max_free_steps"], plateau=True)


def job_arm(ctx, seed, name, lam, dh=None):
    tag = f"{cond(ctx, dh)}_{name}_s{seed}"
    if os.path.exists(_ck(tag)): print("exists", tag); return
    set_seed(seed); D = Data(ctx); m = make(D.V, True, dh)
    C = _load(_ck(f"{cond(ctx, dh)}_free_s{seed}")); m.load_state_dict(C["model"])
    opt = _new_opt(m); opt.load_state_dict(C["opt"]); sc = _new_scaler(); sc.load_state_dict(C["scaler"])
    train(m, D, tag, lam, seed, C["meta"]["step"], CFG["price_steps"], opt, sc)   # same start step & data stream for every arm


def bench(ctx, n=30, dh=None):
    D = Data(ctx); m = make(D.V, True, dh); opt = _new_opt(m); sc = _new_scaler(); rng = np.random.default_rng(0)
    def step():
        x, y = D.batch(D.train, rng)
        with torch.autocast("cuda", dtype=amp_dtype()):
            lh, lc = m(x)
        loss, _, _ = loss_fn(lh, lc, y, 0.1, D.V)
        opt.zero_grad(); sc.scale(loss).backward(); sc.step(opt); sc.update()
    for _ in range(5): step()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): step()
    torch.cuda.synchronize(); ms = (time.time() - t) / n * 1000
    t = time.time(); route_report(m, D); ev = time.time() - t
    r = dict(ctx=ctx, dh=m.dh, cond=cond(ctx, dh), params_M=sum(p.numel() for p in m.parameters()) / 1e6, batch=D.B, ms_per_step=ms, eval_sec=ev, amp=str(amp_dtype()), gpu=torch.cuda.get_device_name(0),
             mem_gb=torch.cuda.max_memory_allocated() / 1e9)
    print(r, flush=True); open(os.path.join(OUT, "bench.jsonl"), "a").write(json.dumps(r) + "\n")


# ----------------------------------------------------------------------------- reporting helpers (interactive; analyze.py is the CPU-only equivalent)
def last(tag):
    p = _log(tag)
    if not os.path.exists(p): return None
    L = [l for l in open(p) if l.strip()]
    return json.loads(L[-1]) if L else None


def progress():
    for f in sorted(os.listdir(LOGD)):
        r = last(f[:-6])
        if r: print(f"{f[:-6]:28s} step {r['step']:6d} ppl {r['ppl']:.3f} copy {r['copy_acc']:.3f} "
                    f"del_ppl_ratio {r.get('del_ppl_ratio', float('nan')):.3f} D_c {r.get('D_c', float('nan')):.2f}")


def _norm(c):
    return (int(c), int(CFG["d"])) if isinstance(c, (int, np.integer)) else (int(c[0]), int(c[1] or CFG["d"]))


def report_A(conds, seeds):
    """conds: list of ctx ints or (ctx, d_h) tuples. Returns (table, {(ctx, d_h): POSITIVE/NEGATIVE/UNUSABLE})."""
    import pandas as pd
    rows = []
    for ctx, dh in map(_norm, conds):
        for s in seeds:
            h, r = last(f"{cond(ctx, dh)}_free_s{s}"), last(f"{cond(ctx, dh)}_reconly_s{s}")
            if not (h and r): continue
            rows.append(dict(ctx=ctx, d_h=dh, seed=s, hyb_step=h["step"], rec_step=r["step"], hyb_ppl=h["ppl"], rec_ppl=r["ppl"],
                             hyb_copy=h["copy_acc"], rec_copy=r["copy_acc"], hyb_hard=h["hard_acc"], rec_hard=r["hard_acc"],
                             del_cost=h["del_cost"], del_ppl_ratio=h["del_ppl_ratio"],
                             viab_ppl=h["ppl"] / r["ppl"], viab_copy=r["copy_acc"] / h["copy_acc"],
                             viab_hard=r["hard_acc"] / max(h["hard_acc"], 1e-6)))
    df = pd.DataFrame(rows); roles = {}
    if df.empty: print("no finished free/reconly pairs yet"); return df, roles
    print(df.round(3).to_string(index=False)); print()
    for (ctx, dh), g in df.groupby(["ctx", "d_h"]):
        lb = g.del_cost.mean() > 0.10 and g.del_ppl_ratio.mean() > 1.2          # deletion-based gate (resampling D_h is structurally high in an LM)
        vi = g.viab_ppl.mean() >= 0.95 and g.viab_copy.mean() >= 0.95
        k = (int(ctx), int(dh)); roles[k] = "POSITIVE" if (lb and vi) else ("NEGATIVE" if lb else "UNUSABLE")
        print(f"ctx {ctx:4d}  GRU width {dh:5d}: load-bearing={'YES' if lb else 'NO'}  viable={'YES' if vi else 'NO'}"
              f"  (viab_ppl {g.viab_ppl.mean():.3f}, viab_copy {g.viab_copy.mean():.3f})  ->  {roles[k]}")
    return df, roles


def report_B(ctx, seeds, lams, posthoc=True, dh=None):
    import pandas as pd
    arms = [("never", 0.0)] + [(f"late{l:g}", l) for l in lams]; rows = []
    D = Data(ctx) if posthoc else None
    for s in seeds:
        nv = last(f"{cond(ctx, dh)}_never_s{s}")
        if nv is None: continue
        for name, lam in arms:
            e = last(f"{cond(ctx, dh)}_{name}_s{s}")
            if e is None: continue
            ok = e["del_ppl_ratio"] < 1.05 and e["del_cost"] < 0.01 and e["ppl"] <= 1.03 * nv["ppl"]
            rows.append(dict(arm=name, lam=lam, seed=s, ppl=e["ppl"], copy=e["copy_acc"], hard=e["hard_acc"], share_c=e["share_c"],
                             D_c=e["D_c"], del_cost=e["del_cost"], del_hard_cost=e["del_hard_cost"],
                             del_ppl_ratio=e["del_ppl_ratio"], rep=e["rep_acc"], del_rep=e["del_rep_acc"], handover=ok))
            if posthoc and name != "never":
                m = make(D.V, True, dh); m.load_state_dict(_load(_ck(f"{cond(ctx, dh)}_never_s{s}"))["model"])
                s1 = evaluate(m, D)["share_c"]; t = e["share_c"]
                beta = (t / (1 - t)) * ((1 - s1) / s1)          # exact: share is monotone in beta (retrieval std scales linearly)
                p = evaluate(m, D, beta=beta)
                rows.append(dict(arm=f"never+posthoc(match {name})", lam=lam, seed=s, ppl=p["ppl"], copy=p["copy_acc"],
                                 hard=p["hard_acc"], share_c=p["share_c"], handover=False))
                del m; torch.cuda.empty_cache()
    res = pd.DataFrame(rows)
    agg = res.groupby("arm", sort=False).agg(**{k: (k, "mean") for k in
          ["ppl", "copy", "hard", "share_c", "D_c", "del_cost", "del_hard_cost", "del_ppl_ratio", "rep", "del_rep"] if k in res},
          handover=("handover", "sum"), n=("seed", "count"))
    print(agg.round(3).to_string()); print()
    print("HANDOVER per seed = del_ppl_ratio < 1.05 AND del_cost < 0.01 AND ppl <= 1.03 x never (same seed)")
    res.to_csv(os.path.join(OUT, f"phaseB_{cond(ctx, dh)}.csv"), index=False)
    return res, agg


def plot_arms(ctx, seeds, lams, path=None, dh=None):
    import matplotlib.pyplot as plt
    arms = ["never"] + [f"late{l:g}" for l in lams]
    keys = [("del_ppl_ratio", "ppl ratio when retrieval deleted"), ("ppl", "val ppl"), ("D_c", "D_c"), ("hard_acc", "hard-copy acc")]
    fig, ax = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.2))
    for i, (k, lab) in enumerate(keys):
        for j, a in enumerate(arms):
            for s in seeds:
                p = _log(f"{cond(ctx, dh)}_{a}_s{s}")
                if not os.path.exists(p): continue
                L = [json.loads(l) for l in open(p) if l.strip()]
                ax[i].plot([r["step"] for r in L], [r[k] for r in L], color=f"C{j}", alpha=.8, label=a if s == seeds[0] else None)
        ax[i].set_title(f"{cond(ctx, dh)}: {lab}"); ax[i].set_xlabel("step")
    ax[0].axhline(1.05, ls=":", c="gray"); ax[0].legend(); plt.tight_layout()
    plt.savefig(path or os.path.join(OUT, f"arms_{cond(ctx, dh)}.png"), dpi=130); plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=["free", "reconly", "arm", "bench"])
    ap.add_argument("--ctx", type=int, required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam", type=float, default=0.0); ap.add_argument("--name", default=None)
    ap.add_argument("--dh", type=int, default=0)             # recurrent (GRU) width; 0 = same as d
    a = ap.parse_args(); load_cfg(); torch.backends.cuda.matmul.allow_tf32 = True
    dh = a.dh or None
    if a.job == "free": job_free(a.ctx, a.seed, True, dh)
    elif a.job == "reconly": job_free(a.ctx, a.seed, False, dh)
    elif a.job == "arm": job_arm(a.ctx, a.seed, a.name, a.lam, dh)
    else: bench(a.ctx, dh=dh)
