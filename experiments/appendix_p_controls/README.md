# Appendix P — steering baselines and controls

- `cram_reimplementation.ipynb`: the CRAM-style adaptation and its fidelity check. Runs three
  seeds (`SEEDS = [0, 1, 2]`); the paper's table caption says five.
- `controls_for_v4.py`: additional readout-refit and late-switch controls. This script was not
  executed as part of the released pipeline.
- Availability/dropout and direct retrieval pricing are in `experiments/core/symbolic_main.ipynb`.
- The eight-seed matched post-hoc attenuation control is in
  `experiments/visual/mnist_replication_8seed.ipynb`.

**Missing code:** delayed attention, weight decay, and the L1 retrieval penalty. Their reported
values are preserved, with provenance, in `results/appendix_p/table22_reported.csv`.
