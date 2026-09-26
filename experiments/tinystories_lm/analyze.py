"""Rebuild every TinyStories table, number and figure in Appendix S from the run logs (CPU only).

    python analyze.py --out results           # the released logs from the paper runs
    python analyze.py --out runs --check      # your own re-run, compared against the paper numbers

Writes <out>/analysis/: screen.csv, steering_per_seed.csv, steering_summary.csv, routes.csv,
extra_price.csv, convergence.csv, tables.tex, summary.json, tinystories_handover.{pdf,png}.
"""
import argparse, json, math, os
import numpy as np
import pandas as pd

D_MODEL = 512
CONDS = [(64, 512), (32, 512), (64, 1536), (32, 1536)]   # ordered by recurrent viability
LAM = "late0.1"
EXTRA = ((64, 512), "late0.3")
HERE = os.path.dirname(os.path.abspath(__file__))


def cond(ctx, dh):
    return f"c{ctx}" + ("" if dh == D_MODEL else f"h{dh}")


def label(ctx, dh):
    return f"{ctx}/{dh}"


class Logs:
    def __init__(self, out):
        self.out = out

    def lines(self, tag):
        p = os.path.join(self.out, "logs", f"{tag}.jsonl")
        if not os.path.exists(p):
            return None
        return [json.loads(l) for l in open(p) if l.strip()]

    def last(self, tag):
        L = self.lines(tag)
        return L[-1] if L else None

    def posthoc(self, tag, seed):
        p = os.path.join(self.out, "posthoc", f"{tag}_{LAM}_s{seed}.json")
        return json.load(open(p)) if os.path.exists(p) else None


def handover(nv, lt):
    return dict(
        ppl_cost=lt["ppl"] / nv["ppl"] - 1,
        r_never=nv["del_ppl_ratio"], r_late=lt["del_ppl_ratio"],
        h=1 - math.log(lt["del_ppl_ratio"]) / math.log(nv["del_ppl_ratio"]),
        h_copy=1 - lt["del_cost"] / nv["del_cost"],
        h_hard=1 - lt["del_hard_cost"] / nv["del_hard_cost"],
        deletable=bool(lt["del_ppl_ratio"] < 1.05 and lt["del_cost"] < 0.01 and lt["ppl"] <= 1.03 * nv["ppl"]),
    )


def build(out):
    L = Logs(out)
    # ---------------------------------------------------------------- viability screen (seed 0)
    screen = []
    for ctx, dh in CONDS:
        f, r = L.last(f"{cond(ctx, dh)}_free_s0"), L.last(f"{cond(ctx, dh)}_reconly_s0")
        if f is None or r is None:
            continue
        v = dict(ctx=ctx, d_h=dh, hyb_step=f["step"], rec_step=r["step"],
                 hyb_ppl=f["ppl"], rec_ppl=r["ppl"], hyb_copy=f["copy_acc"], rec_copy=r["copy_acc"],
                 hyb_hard=f["hard_acc"], rec_hard=r["hard_acc"],
                 viab_ppl=f["ppl"] / r["ppl"], viab_copy=r["copy_acc"] / f["copy_acc"], viab_hard=r["hard_acc"] / f["hard_acc"],
                 free_del_ratio=f["del_ppl_ratio"], free_del_cost=f["del_cost"])
        v["load_bearing"] = bool(v["free_del_cost"] > 0.10 and v["free_del_ratio"] > 1.2)
        v["viable"] = bool(v["viab_ppl"] >= 0.95 and v["viab_copy"] >= 0.95)
        screen.append(v)
    screen = pd.DataFrame(screen)

    # ---------------------------------------------------------------- steering (never vs late, lambda = 0.1)
    rows = []
    for ctx, dh in CONDS:
        tag = cond(ctx, dh)
        rec = L.last(f"{tag}_reconly_s0")
        for s in range(10):
            nv, lt = L.last(f"{tag}_never_s{s}"), L.last(f"{tag}_{LAM}_s{s}")
            if nv is None or lt is None:
                continue
            row = dict(ctx=ctx, d_h=dh, seed=s, never_ppl=nv["ppl"], late_ppl=lt["ppl"],
                       never_copy=nv["copy_acc"], late_copy=lt["copy_acc"], never_hard=nv["hard_acc"], late_hard=lt["hard_acc"],
                       share_never=nv["share_c"], share_late=lt["share_c"], Dc_never=nv["D_c"], Dc_late=lt["D_c"],
                       Dh_never=nv["D_h"], Dh_late=lt["D_h"],
                       del_ppl_never=nv["ppl"] * nv["del_ppl_ratio"], del_ppl_late=lt["ppl"] * lt["del_ppl_ratio"],
                       del_hard_never=nv["del_hard_cost"], del_hard_late=lt["del_hard_cost"],
                       rec_only_ppl=rec["ppl"] if rec else np.nan, **handover(nv, lt))
            ph = L.posthoc(tag, s)
            if ph:
                row.update(posthoc_ppl=ph["ppl"], posthoc_copy=ph["copy_acc"], posthoc_hard=ph["hard_acc"],
                           posthoc_share=ph["share_c"], posthoc_cost=ph["ppl"] / nv["ppl"] - 1,
                           posthoc_vs_late=ph["ppl"] / lt["ppl"] - 1)
            rows.append(row)
    per_seed = pd.DataFrame(rows)

    num = [c for c in per_seed.columns if c not in ("ctx", "d_h", "seed", "deletable")]
    g = per_seed.groupby(["ctx", "d_h"], sort=False)
    summary = g[num].mean().add_suffix("_mean").join(g[num].std().add_suffix("_sd"))
    summary["n_seeds"] = g.size()
    summary["n_deletable"] = g["deletable"].sum()
    summary = summary.reset_index()

    # ---------------------------------------------------------------- extra price, convergence
    extra = []
    (ctx, dh), name = EXTRA
    nv, lt = L.last(f"{cond(ctx, dh)}_never_s0"), L.last(f"{cond(ctx, dh)}_{name}_s0")
    if nv and lt:
        extra.append(dict(ctx=ctx, d_h=dh, arm=name, seed=0, **handover(nv, lt),
                          del_ppl_late=lt["ppl"] * lt["del_ppl_ratio"]))
    extra = pd.DataFrame(extra)

    conv = []
    for ctx, dh in CONDS:
        for job in ("free", "reconly"):
            for s in range(10):
                Ls = L.lines(f"{cond(ctx, dh)}_{job}_s{s}")
                if not Ls:
                    continue
                pp = {r["step"]: r["ppl"] for r in Ls}; last = Ls[-1]["step"]
                ref = max(k for k in pp if k <= last - 3000)
                conv.append(dict(ctx=ctx, d_h=dh, job=job, seed=s, final_step=last,
                                 stopped_by="plateau rule" if last % 1000 == 0 and last < 19999 else "20k cap",
                                 ppl_change_last_3k=(pp[ref] - pp[last]) / pp[ref], hours=Ls[-1]["sec"] / 3600))
    conv = pd.DataFrame(conv)
    return screen, per_seed, summary, extra, conv


# -------------------------------------------------------------------- LaTeX
def fmt_ms(m, s, n, d=3):
    return f"${m:.{d}f} \\pm {s:.{d}f}$" if n > 1 else f"{m:.{d}f}"


def latex(screen, per_seed, summary):
    T = []
    T.append(r"""\begin{table}[h]\centering\small
\caption{\textbf{TinyStories viability screen} (seed 0). Viability is recurrent-only relative to the hybrid.}
\label{tab:ts_screen}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lcccccccc}\toprule
& \multicolumn{2}{c}{perplexity} & \multicolumn{2}{c}{copy acc.} & \multicolumn{3}{c}{viability} & free hybrid \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-8}
ctx / $d_h$ & hybrid & rec.\ only & hybrid & rec.\ only & ppl & copy & hard & deletion $r$ \\ \midrule""")
    for _, r in screen.iterrows():
        T.append(f"{int(r.ctx)} / {int(r.d_h)} & {r.hyb_ppl:.3f} & {r.rec_ppl:.3f} & {r.hyb_copy:.3f} & {r.rec_copy:.3f} & "
                 f"{r.viab_ppl:.3f} & {r.viab_copy:.3f} & {r.viab_hard:.3f} & {r.free_del_ratio:.2f} \\\\")
    T.append(r"\bottomrule\end{tabular}}\end{table}" + "\n")

    T.append(r"""\begin{table}[h]\centering\small
\caption{\textbf{Late pricing on TinyStories} ($\lambda = 0.1$, 6k steps). Mean $\pm$ standard deviation where
several seeds are available. Post-hoc: never arm's retrieval logits scaled to the late arm's share without training; its perplexity cost is relative to the late arm.}
\label{tab:ts_steer}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lccccccc}\toprule
ctx / $d_h$ & seeds & ppl cost & post-hoc ppl cost & $r$: never $\to$ late & $h$ & $h_{\text{copy}}$ & $h_{\text{hard}}$ \\ \midrule""")
    for _, r in summary.iterrows():
        n = int(r.n_seeds)
        ph = f"$+{100 * r.posthoc_vs_late_mean:.1f}\\%$" if "posthoc_vs_late_mean" in r and not np.isnan(r.posthoc_vs_late_mean) else "--"
        T.append(f"{int(r.ctx)} / {int(r.d_h)} & {n} & $+{100 * r.ppl_cost_mean:.1f}\\%$ & {ph} & "
                 f"${r.r_never_mean:.2f} \\to {r.r_late_mean:.2f}$ & {fmt_ms(r.h_mean, r.h_sd, n)} & "
                 f"{fmt_ms(r.h_copy_mean, r.h_copy_sd, n)} & {fmt_ms(r.h_hard_mean, r.h_hard_sd, n)} \\\\")
    T.append(r"\bottomrule\end{tabular}}\end{table}" + "\n")

    T.append(r"""\begin{table}[h]\centering\small
\caption{\textbf{Per-seed TinyStories steering endpoints.}}
\label{tab:ts_perseed}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lcccccccc}\toprule
& & \multicolumn{3}{c}{perplexity} & & & & \\ \cmidrule(lr){3-5}
ctx / $d_h$ & seed & never & late & post-hoc & $r$: never $\to$ late & $h$ & $h_{\text{copy}}$ & $h_{\text{hard}}$ \\ \midrule""")
    prev = None
    for _, r in per_seed.iterrows():
        if prev is not None and (r.ctx, r.d_h) != prev:
            T.append(r"\midrule")
        prev = (r.ctx, r.d_h)
        f = lambda x: f"${x:.3f}$" if x < 0 else f"{x:.3f}"
        ph = f"{r.posthoc_ppl:.3f}" if "posthoc_ppl" in r and not np.isnan(r.posthoc_ppl) else "--"
        T.append(f"{int(r.ctx)} / {int(r.d_h)} & {int(r.seed)} & {r.never_ppl:.3f} & {r.late_ppl:.3f} & {ph} & "
                 f"${r.r_never:.3f} \\to {r.r_late:.3f}$ & {f(r.h)} & {f(r.h_copy)} & {f(r.h_hard)} \\\\")
    T.append(r"\bottomrule\end{tabular}}\end{table}" + "\n")

    T.append(r"""\begin{table}[h]\centering\small
\caption{\textbf{Route measurements before and after pricing} (means over available seeds).}
\label{tab:ts_routes}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lcccccc}\toprule
ctx / $d_h$ & share: never $\to$ late & $D_c$: never $\to$ late & copy: late / post-hoc & hard: late / post-hoc & deleted ppl: never $\to$ late & rec.\ only ppl \\ \midrule""")
    for _, r in summary.iterrows():
        T.append(f"{int(r.ctx)} / {int(r.d_h)} & ${r.share_never_mean:.3f} \\to {r.share_late_mean:.3f}$ & "
                 f"${r.Dc_never_mean:.3f} \\to {r.Dc_late_mean:.3f}$ & {r.late_copy_mean:.3f} / {r.posthoc_copy_mean:.3f} & "
                 f"{r.late_hard_mean:.3f} / {r.posthoc_hard_mean:.3f} & ${r.del_ppl_never_mean:.2f} \\to {r.del_ppl_late_mean:.2f}$ & "
                 f"{r.rec_only_ppl_mean:.2f} \\\\")
    T.append(r"\bottomrule\end{tabular}}\end{table}")
    return "\n".join(T)


# -------------------------------------------------------------------- figure
def figure(screen, per_seed, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "legend.fontsize": 6.5,
                         "axes.spines.top": False, "axes.spines.right": False})
    order = [(c, d) for c, d in CONDS if ((screen.ctx == c) & (screen.d_h == d)).any()]
    x = np.arange(len(order)); labs = [label(*o) for o in order]
    BLUE, ORANGE, GREEN, GREY = "#1f77b4", "#ff7f0e", "#2ca02c", "#7f7f7f"
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.2))
    a = ax[0]
    S = screen.set_index(["ctx", "d_h"]).loc[order]
    for k, (col, c, mk, nm) in enumerate([("viab_ppl", BLUE, "o", "perplexity"), ("viab_copy", ORANGE, "s", "copy"),
                                          ("viab_hard", GREEN, "^", "hard copy")]):
        a.plot(x + (k - 1) * 0.12, S[col], mk, color=c, ms=4.5, label=nm)
    a.axhline(0.95, ls="--", lw=0.8, c=GREY); a.text(-0.35, 0.953, "0.95 screen", fontsize=6, color=GREY, va="bottom")
    a.set_ylim(0.8, 1.02); a.set_ylabel("recurrent-only / hybrid"); a.set_title("(a) recurrent viability")
    a.legend(loc="upper center", ncol=3, frameon=False, handletextpad=0.2, columnspacing=0.8, bbox_to_anchor=(0.56, 1.03))

    def bars(axis, cols, df):
        for k, (col, colr, nm) in enumerate(cols):
            if col not in df: continue
            for i, o in enumerate(order):
                v = df[(df.ctx == o[0]) & (df.d_h == o[1])][col].dropna().values
                if not len(v): continue
                xi = i + (k - 0.5) * 0.28
                axis.bar(xi, v.mean(), width=0.26, color=colr, alpha=0.35 if len(v) == 1 else 0.8, label=nm if i == 0 else None)
                axis.plot(np.full(len(v), xi), v, "o", color="k", ms=2.2)

    b = ax[1]
    bars(b, [("h", BLUE, "perplexity $h$"), ("h_hard", ORANGE, "hard copy $h_{hard}$")], per_seed)
    b.axhline(1.0, ls=":", lw=0.8, c=GREY); b.text(len(order) - 0.55, 0.97, "full handover", fontsize=6, color=GREY, ha="right", va="top")
    b.axhline(0, lw=0.5, c="k"); b.set_ylim(-0.1, 1.05); b.set_ylabel("handover fraction")
    b.set_title(r"(b) transfer at $\lambda=0.1$"); b.legend(loc="center left", frameon=False, bbox_to_anchor=(0, 0.72))

    c = ax[2]
    pc = per_seed.assign(ppl_cost_pct=100 * per_seed.ppl_cost)
    if "posthoc_cost" in per_seed: pc["posthoc_pct"] = 100 * per_seed.posthoc_cost      # both relative to the never arm
    bars(c, [("ppl_cost_pct", BLUE, "late pricing (learned)"), ("posthoc_pct", GREY, "post-hoc scaling")], pc)
    c.set_ylim(0, 48); c.set_ylabel("perplexity increase vs. never arm (%)"); c.set_title("(c) same retrieval share")
    c.legend(loc="upper right", frameon=False, bbox_to_anchor=(1.03, 1.02))
    for axis in ax:
        axis.set_xticks(x); axis.set_xticklabels(labs, fontsize=6.3); axis.set_xlabel("context / GRU width $d_h$")
    plt.tight_layout(w_pad=1.2)
    for ext in ("pdf", "png"):
        plt.savefig(f"{path}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------- check against the paper
def check(screen, summary, per_seed, extra, expected_path):
    E = json.load(open(expected_path)); tol = E["tolerance"]; bad = 0
    def cmp(name, got, want, t):
        nonlocal bad
        ok = abs(got - want) <= t; bad += not ok
        print(f"  {'ok  ' if ok else 'DIFF'} {name:38s} got {got:8.3f}  paper {want:8.3f}  (tol {t})")
    print("\nComparison with the numbers reported in the paper:")
    for key, v in E["screen"].items():
        ctx, dh = map(int, key.split("/")); r = screen[(screen.ctx == ctx) & (screen.d_h == dh)]
        if r.empty: print(f"  MISSING screen {key}"); bad += 1; continue
        for m in ("viab_ppl", "viab_copy", "viab_hard"):
            cmp(f"screen {key} {m}", float(r[m].iloc[0]), v[m], tol["viability"])
    for key, v in E["steering"].items():
        ctx, dh = map(int, key.split("/")); r = summary[(summary.ctx == ctx) & (summary.d_h == dh)]
        if r.empty: print(f"  MISSING steering {key}"); bad += 1; continue
        r = r.iloc[0]
        cmp(f"steer {key} h (n={int(r.n_seeds)})", r.h_mean, v["h"], tol["h"])
        cmp(f"steer {key} ppl cost", r.ppl_cost_mean, v["ppl_cost"], tol["ppl_cost"])
        cmp(f"steer {key} r late", r.r_late_mean, v["r_late"], tol["r"])
        if "posthoc_vs_late" in v and "posthoc_vs_late_mean" in r and not np.isnan(r.posthoc_vs_late_mean):
            cmp(f"steer {key} post-hoc ppl cost", r.posthoc_vs_late_mean, v["posthoc_vs_late"], tol["posthoc"])
    nd = int(per_seed.deletable.sum())
    print(f"  {'ok  ' if nd == E['n_deletable'] else 'DIFF'} deletable late arms: {nd} (paper {E['n_deletable']})"); bad += nd != E["n_deletable"]
    print("\nAll checks passed." if bad == 0 else f"\n{bad} value(s) outside tolerance (small differences are expected on other GPUs/software).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results", help="directory containing logs/ (and posthoc/)")
    ap.add_argument("--check", action="store_true", help="compare against expected/paper_numbers.json")
    a = ap.parse_args()
    screen, per_seed, summary, extra, conv = build(a.out)
    dst = os.path.join(a.out, "analysis"); os.makedirs(dst, exist_ok=True)
    screen.to_csv(f"{dst}/screen.csv", index=False); per_seed.to_csv(f"{dst}/steering_per_seed.csv", index=False)
    summary.to_csv(f"{dst}/steering_summary.csv", index=False); extra.to_csv(f"{dst}/extra_price.csv", index=False)
    conv.to_csv(f"{dst}/convergence.csv", index=False)
    open(f"{dst}/tables.tex", "w").write(latex(screen, per_seed, summary))
    figure(screen, per_seed, f"{dst}/tinystories_handover")

    pd.set_option("display.width", 200)
    print("=== Viability screen (seed 0)")
    print(screen[["ctx", "d_h", "hyb_ppl", "rec_ppl", "viab_ppl", "viab_copy", "viab_hard", "free_del_ratio", "load_bearing", "viable"]].round(3).to_string(index=False))
    print("\n=== Late pricing, lambda = 0.1 (per seed)")
    cols = ["ctx", "d_h", "seed", "ppl_cost", "r_never", "r_late", "h", "h_copy", "h_hard", "deletable"] + \
           (["posthoc_vs_late"] if "posthoc_vs_late" in per_seed else [])
    print(per_seed[cols].round(3).to_string(index=False))
    print("\n=== Summary (mean over seeds)")
    print(summary[["ctx", "d_h", "n_seeds", "ppl_cost_mean", "r_late_mean", "h_mean", "h_sd", "h_hard_mean", "n_deletable"]].round(3).to_string(index=False))
    if len(extra):
        print("\n=== Price strength (lambda = 0.3, seed 0)"); print(extra.round(3).to_string(index=False))
    logs = [f for f in os.listdir(os.path.join(a.out, "logs")) if f.endswith(".jsonl")]
    gpu_h = sum(Logs(a.out).last(f[:-6])["sec"] for f in logs) / 3600
    print(f"\n=== Compute: {gpu_h:.1f} GPU-hours over {len(logs)} runs ({conv.hours.sum():.1f} h in free / recurrent-only runs)")
    json.dump(dict(n_late_arms=int(len(per_seed)), n_deletable=int(per_seed.deletable.sum()),
                   h_by_condition={label(int(r.ctx), int(r.d_h)): [round(r.h_mean, 4), round(0 if np.isnan(r.h_sd) else r.h_sd, 4), int(r.n_seeds)]
                                   for _, r in summary.iterrows()}),
              open(f"{dst}/summary.json", "w"), indent=1)
    print(f"\nWrote tables, CSVs and figure to {dst}/")
    if a.check:
        check(screen, summary, per_seed, extra, os.path.join(HERE, "expected", "paper_numbers.json"))


if __name__ == "__main__":
    main()
