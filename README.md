# Same Behaviour, Different Causal Route — ICLR 2027 code release

Clean GitHub-ready release for the paper **“Same Behaviour, Different Causal Route: Post-Convergence Steering in Recurrent–Attention Hybrids.”**

The release separates paper experiments from exploratory pilots, removes notebook backup/debug clutter, encodes the final MNIST/CIFAR eight-seed runs directly, and cleans the large-scale notebook so the final load-dependent switch protocol is run once rather than through repeated correction cells.

## Repository map

```text
experiments/
  core/                 symbolic main suite
  visual/               final MNIST + CIFAR-10 replications
  large_scale/          cleaned final large-scale boundary protocol
  appendix_d_mqar/      fused MQAR
  appendix_e_falcon_h1/ pretrained Falcon-H1 measurement
  appendix_g_fusion/    non-additive / pre-fusion check
  appendix_n_attribution/
  appendix_o_gpt2_ioi/
  appendix_p_controls/
  appendix_q_worm_vsr/
  pilots/               follow-up / non-paper experiments
results/                 raw symbolic data + selected reported/result tables
figures/                 paper-useful figures supplied with the runs
```

See `docs/PAPER_CODE_MAP.md` for the paper-to-code map and `docs/REPRODUCIBILITY_AUDIT.md` before making the repository public.

## Quick start

- Core symbolic suite: `experiments/core/symbolic_main.ipynb`
- MNIST final eight-seed replication: `experiments/visual/mnist_replication_8seed.ipynb`
- CIFAR-10 final eight-seed replication: `experiments/visual/cifar10_replication_8seed.ipynb`
- Large-scale test: `experiments/large_scale/large_scale_boundary_clean.ipynb`

Use smoke/short presets first where the notebook provides them. The raw symbolic JSONL/CSV stores and supplied figures are included so tables/figures can be regenerated without retraining every run.

## Missing code

The only paper-reported implementations not present in the supplied source are:

- delayed attention baseline
- weight-decay baseline
- L1 retrieval-penalty baseline

Their reported Table 22 values are preserved in `results/appendix_p/table22_reported.csv`. All other paper components are represented by supplied code or by the cleaned/reorganized versions of that supplied code; see `docs/PAPER_CODE_MAP.md` and the audit notes for provenance/version details.

## Large files

Model checkpoints are intentionally excluded. The supplied 75MB Mamba checkpoint archive is not committed; the pretrained-model notebooks download/load their upstream model and regenerate local checkpoints.

## Double-blind review

Before sending a repository URL to ICLR, publish from an anonymous/non-identifying account and inspect Git history, repository ownership, issue history, notebook outputs, and external Drive/Hugging Face paths for identifying information.
