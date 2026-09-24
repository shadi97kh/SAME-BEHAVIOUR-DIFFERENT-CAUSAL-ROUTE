"""Run every TinyStories experiment reported in Appendix R.

Jobs run one per GPU as separate processes. Finished jobs are skipped and interrupted jobs resume
from their last 1,000-step checkpoint, so the script can simply be re-run after a disconnect.

    python run_experiments.py --stage all            # everything (~18 T4 GPU-hours)
    python run_experiments.py --stage screen         # one stage at a time
    python run_experiments.py --list                 # show the job plan without running it

Stages (each depends on the previous one):
    prepare  tokenise TinyStories and build the vocabulary / bigram tables (CPU, ~10 min)
    screen   free hybrid + recurrent-only model, seed 0, four (ctx, d_h) conditions
    steer    never / late (lambda = 0.1) arms from each seed-0 free checkpoint, plus lambda = 0.3 at (64, 512)
    seeds    seeds 1-2 at the lowest- and highest-viability conditions: free runs, then never / late arms
    posthoc  matched post-hoc attenuation control for every late (lambda = 0.1) arm
"""
import argparse, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))

# (context length, GRU width d_h) conditions reported in the paper, in order of recurrent viability
CONDS = [(64, 512), (32, 512), (64, 1536), (32, 1536)]
SEED_CONDS = [(64, 512), (32, 1536)]      # three-seed conditions
LAM = 0.1                                 # pre-selected price
EXTRA_PRICE = ((64, 512), 0.3)            # single-seed price-strength check


def _c(ctx, dh, seed):
    return ["--ctx", str(ctx), "--dh", str(dh), "--seed", str(seed)]


def arm(ctx, dh, seed, name, lam):
    return ["lh.py", "arm", *_c(ctx, dh, seed), "--name", name, "--lam", str(lam)]


def plan():
    """Ordered stages; each stage is a list of waves, and the jobs within a wave are independent."""
    screen = [[["lh.py", job, *_c(ctx, dh, 0)] for ctx, dh in CONDS for job in ("free", "reconly")]]
    steer = [[a for ctx, dh in CONDS for a in (arm(ctx, dh, 0, "never", 0), arm(ctx, dh, 0, f"late{LAM:g}", LAM))]
             + [arm(*EXTRA_PRICE[0], 0, f"late{EXTRA_PRICE[1]:g}", EXTRA_PRICE[1])]]
    seeds = [[["lh.py", "free", *_c(ctx, dh, s)] for ctx, dh in SEED_CONDS for s in (1, 2)],
             [a for ctx, dh in SEED_CONDS for s in (1, 2)
              for a in (arm(ctx, dh, s, "never", 0), arm(ctx, dh, s, f"late{LAM:g}", LAM))]]
    posthoc_jobs = [["posthoc.py", *_c(ctx, dh, 0), "--name", f"late{LAM:g}"] for ctx, dh in CONDS] + \
                   [["posthoc.py", *_c(ctx, dh, s), "--name", f"late{LAM:g}"] for ctx, dh in SEED_CONDS for s in (1, 2)]
    return {"screen": screen, "steer": steer, "seeds": seeds, "posthoc": [posthoc_jobs]}


def n_gpus():
    import torch
    n = torch.cuda.device_count()
    if n == 0:
        raise SystemExit("No CUDA GPU found. Training requires a GPU (the paper used Tesla T4s).")
    return n


def run_wave(jobs, out, ngpu, poll=20):
    os.makedirs(os.path.join(out, "stdout"), exist_ok=True)
    queue, running = list(jobs), {}
    while queue or running:
        for g in range(ngpu):
            if g not in running and queue:
                args = queue.pop(0)
                name = "_".join([os.path.splitext(args[0])[0]] + [x.lstrip("-") for x in args[1:]])
                log = open(os.path.join(out, "stdout", f"{name}.txt"), "a")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), LH_OUT=out)
                p = subprocess.Popen([sys.executable, os.path.join(HERE, args[0]), *args[1:]],
                                     env=env, stdout=log, stderr=subprocess.STDOUT, cwd=HERE)
                running[g] = (p, name, time.time())
                print(f"[gpu{g}] start {name}", flush=True)
        for g, (p, name, t0) in list(running.items()):
            if p.poll() is not None:
                status = "done" if p.returncode == 0 else f"FAILED (exit {p.returncode}); see {out}/stdout/{name}.txt"
                print(f"[gpu{g}] {status} {name} [{(time.time() - t0) / 3600:.2f} h]", flush=True)
                if p.returncode != 0:
                    raise SystemExit(1)
                del running[g]
        time.sleep(poll)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", choices=["all", "prepare", "screen", "steer", "seeds", "posthoc"])
    ap.add_argument("--out", default="runs", help="output directory (logs, checkpoints, data)")
    ap.add_argument("--raw-cache", default=None, help="optional cached GPT-2 token array (.npy) to skip tokenisation")
    ap.add_argument("--list", action="store_true", help="print the job plan and exit")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.environ["LH_OUT"] = out

    stages = ["prepare", "screen", "steer", "seeds", "posthoc"] if a.stage == "all" else [a.stage]
    P = plan()
    if a.list:
        for st in stages:
            if st == "prepare": print("prepare: tokenise + vocabulary + bigram (CPU)"); continue
            for i, wave in enumerate(P[st]):
                print(f"{st} / wave {i + 1}: {len(wave)} jobs")
                for j in wave: print("   python", " ".join(j))
        return

    for st in stages:
        print(f"\n===== stage: {st} =====", flush=True)
        if st == "prepare":
            sys.path.insert(0, HERE)
            import lh
            lh.load_cfg(); meta = lh.prepare(raw_cache=a.raw_cache)
            # sanity check against the tokenised corpus used in the paper
            ok = meta["V"] == 8192 and abs(meta["coverage"] - 0.9979) < 5e-4 and abs(meta["n_tokens"] / 89.9e6 - 1) < 2e-3
            print("data check:", "matches the paper corpus (V=8192, coverage 0.9979, 89.9M tokens)" if ok else
                  f"WARNING: differs from the paper corpus (V={meta['V']}, coverage {meta['coverage']:.4f}, "
                  f"{meta['n_tokens'] / 1e6:.1f}M tokens); the TinyStories dataset revision may have changed")
            continue
        ngpu = n_gpus()
        for wave in P[st]:
            run_wave(wave, out, ngpu)
    print("\nAll requested stages finished. Next: python analyze.py --out", a.out)


if __name__ == "__main__":
    main()
