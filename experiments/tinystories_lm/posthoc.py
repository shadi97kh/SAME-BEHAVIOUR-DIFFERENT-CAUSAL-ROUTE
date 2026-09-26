"""Matched post-hoc attenuation control (Appendix S).

Scales the never arm's retrieval logits by a constant beta, without any training, so that its
retrieval contribution share equals the late arm's share, then evaluates the scaled model.
Writes $LH_OUT/posthoc/<cond>_<name>_s<seed>.json.

    python posthoc.py --ctx 64 --dh 512 --seed 0 --name late0.1
"""
import argparse, json, os
import lh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--dh", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="late0.1", help="late arm whose retrieval share is matched")
    a = ap.parse_args()
    lh.load_cfg()
    dh = a.dh or None
    tag = lh.cond(a.ctx, dh)
    out_dir = os.path.join(lh.OUT, "posthoc"); os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{tag}_{a.name}_s{a.seed}.json")
    if os.path.exists(out):
        print("exists", out); return
    late = lh.last(f"{tag}_{a.name}_s{a.seed}")
    if late is None:
        raise SystemExit(f"missing log for {tag}_{a.name}_s{a.seed}")
    D = lh.Data(a.ctx)
    m = lh.make(D.V, True, dh)
    m.load_state_dict(lh._load(lh._ck(f"{tag}_never_s{a.seed}"))["model"])
    s1 = lh.evaluate(m, D)["share_c"]                      # never arm's own retrieval share (beta = 1)
    t = late["share_c"]                                    # target: the late arm's share
    beta = (t / (1 - t)) * ((1 - s1) / s1)                 # exact, since retrieval logit std scales linearly in beta
    p = lh.evaluate(m, D, beta=beta)
    rec = dict(cond=tag, seed=a.seed, matched_arm=a.name, beta=beta, target_share=t, **p)
    json.dump(rec, open(out, "w"), indent=1)
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
