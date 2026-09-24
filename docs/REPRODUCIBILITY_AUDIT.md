# Reproducibility audit

An audit of the released code against the submitted manuscript. It records
what matches, what does not, and what cannot be checked. Read it before relying on any single
number, and before making this repository public.

## What can and cannot be verified here

The notebooks are released **without saved outputs**. Every reported number is therefore
reproducible only by re-running the code; nothing in this repository proves a number was
produced by the released version. The one exception is `experiments/tinystories_lm/`, which
ships the evaluation logs of all 29 runs and rebuilds its tables and figure on CPU
(`python analyze.py --out results --check`).

## Verified in this release

- `experiments/tinystories_lm/`: `python analyze.py --out results --check` was run against the
  committed logs and all 20 checks pass (viability screen, transfer fraction *h*, perplexity
  cost, post-hoc control, and 0 deletable late arms). This is the only experiment whose reported
  numbers can be re-derived without a GPU.

## Open mismatches between paper and code

These are unresolved. Each needs either a code update or a correction to the manuscript.

1. **Fusion appendix (App G).** The paper reports load 8, λ=0.25, 16k constant-LR steps,
   5 seeds. `experiments/appendix_g_fusion/prefusion_check.ipynb` sets `LAM = 0.10` and
   `STEPS = 2500`. Load, seeds and the three fusion rules match. As released, the notebook
   cannot produce the reported table.
2. **Falcon-H1 `D_SSM` (App E).** The paper normalises by in-context chance `1/L`. The
   notebook's `dependence()` divides by `1/len(VAL_IDS)`; `chance_ctx = 1/L` is computed but
   unused. The published column comes from a post-hoc calculation not present here.
3. **Inversion counting (§4.4, App C).** The paper reports crossings at the final checkpoint,
   and unchanged when averaged over the last three. Both visual notebooks count a seed as
   inverted if `D_h > D_c` at *any* post-convergence checkpoint.
4. **Mamba-2 screen (App F).** The paper reports four prompt seeds × 64 prompts and a
   16-fact condition. The notebook has a single `SEED = 0` and `SCREEN_LOADS = [1, 2, 4, 8]`.
   With fast kernels installed it also prefers a 370M checkpoint over the reported 130M.
5. **MNIST details (App C).** The boundary sweep runs 5 seeds; the paper says 3. No code
   produces the load-6 recurrent-only value (0.894) quoted in the appendix.
6. **CRAM row (App P).** The table caption says five seeds; `cram_reimplementation.ipynb`
   runs `SEEDS = [0, 1, 2]`.
7. **GPT-2 IOI (App O).** The paper says the logit difference is centred; the code uses a
   raw difference scaled by `|base|`.

## Resolved during the audit

- **Large-scale parameter count.** The released model has **19,773,952** parameters
  (recounted from `large_scale_boundary_clean.ipynb`: embeddings 52,224; trunk 16,781,312;
  norm 512; GRU 1,575,936; recurrent readout 288,768; retrieval route 1,075,200). Deleting
  retrieval gives 18.70M, a 5.4% reduction. An earlier draft said 30.17M → 28.59M, whose
  1.58M difference equals the GRU's parameter count rather than retrieval's. The manuscript
  now reports 19.77M, which also matches the title of the supplied
  large-scale trajectory figure.
- **Sequence lengths.** The final selective-recall task produces `2 + L + 2·min(4,L)` tokens:
  18 at load 8 and 42 at load 32. The 266- and 74-token figures in the earlier draft came
  from a removed pair-task version, and the latency numbers derived from them have been
  dropped; the paper now makes a state-memory claim only.
- **Scaffold budget.** `anneal()` runs 2500 unpriced steps, then 1250-step stages over
  λ ∈ {0.05, 0.1, 0.2, 0.3, 0.5, 0.8}, stopping once retrieval share < 0.15, so its total is
  3750–10000 steps against 7500 for the direct run. The paper no longer claims matched budgets.
- **Accounting.** The `free_long` and `deflation` blocks are not in the 23 accounted blocks
  and are excluded from the 14.4 GPU-hour total; the appendix now says so. Run counts in the
  seed table are records in each stored block and include superseded runs (`width` 315 vs 245
  in the current grid; `ood_pairs`/`ood_eval` 50/90 vs 45).
- **Optimiser defaults.** The `neg_optcost` and `neg_learnability` probes construct AdamW
  without arguments, so they use weight decay 0.01 and no gradient clipping, unlike the rest
  of the suite. The appendix now states this.

## Known gaps

- No code for the delayed-attention, weight-decay and L1 baselines
  (`results/appendix_p/table22_reported.csv` preserves their reported values).
- `experiments/appendix_p_controls/controls_for_v4.py` was never executed as part of the
  released pipeline.
- The large-scale route-viability screen runs a single seed; the paper does not say otherwise,
  but it is worth knowing.
- Model checkpoints are intentionally not committed. Pretrained-model notebooks download
  their upstream weights and regenerate local checkpoints.

## Before publishing

- Publish from an account that does not identify the authors. The git history of this
  repository currently carries a real name and email, so export a fresh tree rather than
  pushing this history.
- Re-check notebook outputs, Drive paths and remotes after any re-run; outputs can reintroduce
  local paths and account names.
