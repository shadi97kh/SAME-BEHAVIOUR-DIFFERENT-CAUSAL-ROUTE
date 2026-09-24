# Appendix F — pretrained Mamba-2 boundary and resampling diagnostic

- `mamba2_stage1_added_attention.ipynb`: frozen-backbone viability screen and the added-attention
  diagnostic, where a necessary route registers near-zero resampling dependence because its
  contribution is nearly input-invariant.
- `mamba2_steering_stage2.ipynb`: follow-up steering stage and the common-mode decomposition.

Needs a GPU with `mamba-ssm` and `causal-conv1d` installed, and downloads
`state-spaces/mamba-130m-hf` (the reported model) plus WikiText-2.

Two differences from the paper are recorded in `docs/REPRODUCIBILITY_AUDIT.md`: the screen here
runs a single prompt seed over loads 1–8, while the paper reports four prompt seeds and a
16-fact condition.
