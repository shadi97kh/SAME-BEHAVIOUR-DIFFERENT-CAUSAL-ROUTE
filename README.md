# Same Behaviour, Different Causal Route

Code for **"Same Behaviour, Different Causal Route: Post-Convergence Steering in
Recurrent–Attention Hybrids"** (ICLR 2027 submission, anonymous).

After a recurrent–attention hybrid has solved a task through its attention retrieval route, we
price that route and ask whether causal reliance can move to recurrence without changing
behaviour. It can, but only when the receiving route is independently capable of solving the
task; after a completed handover the retrieval route can be deleted at essentially no cost in
accuracy. This repository contains the experiments behind every reported result.

## Layout

```text
experiments/
  core/                 symbolic suite: setup, metrics, pricing, and most main-text results
  visual/               MNIST and CIFAR-10 eight-seed replications
  large_scale/          19.77M-parameter boundary test and route deletion
  appendix_d_mqar/      fused MQAR hybrid
  appendix_e_falcon_h1/ pretrained Falcon-H1 route measurement
  appendix_f_mamba2/    pretrained Mamba-2 boundary and resampling diagnostic
  appendix_g_fusion/    non-additive / pre-fusion readouts
  appendix_n_attribution/  attribution-proxy crossing offsets
  appendix_o_gpt2_ioi/  pretrained GPT-2 IOI measurement check
  appendix_p_controls/  steering baselines and CRAM-style fidelity check
  appendix_q_worm_vsr/  pretrained WorM VSR route-availability diagnostic
  majority/             Majority task: a structurally distinct handover task
  tinystories_lm/       TinyStories language-model extension (self-contained, with logs)
  pilots/               exploratory and superseded notebooks; no paper result depends on these
docs/                   paper-to-code map and reproducibility audit
figures/                figures supplied with the runs
results/                reported values preserved for experiments without code
```

`docs/PAPER_CODE_MAP.md` maps every paper section, table and figure to the file that produces
it. Read `docs/REPRODUCIBILITY_AUDIT.md` before relying on any individual number: it lists what
has been verified, several open mismatches between the code and the manuscript, and what cannot
be checked from this release.

## Getting started

```bash
pip install -r requirements.txt
```

The fastest way to see a real result is the TinyStories extension, which ships its logs and
rebuilds every number and its figure on CPU in under a minute:

```bash
cd experiments/tinystories_lm
python analyze.py --out results --check
```

For the training experiments, open a notebook and run it top to bottom. Each one selects its
scale with a `PRESET` variable at the top; start with `PRESET = "smoke"` to check the pipeline
before committing to `"full"`, which reproduces the paper numbers. Most notebooks also have a
`RUN` dictionary that switches individual experiment blocks on and off.

Entry points:

| Experiment | File |
| --- | --- |
| Symbolic suite (most main-text results) | `experiments/core/symbolic_main.ipynb` |
| MNIST / CIFAR-10 replications | `experiments/visual/*_replication_8seed.ipynb` |
| Large-scale boundary test and deletion | `experiments/large_scale/large_scale_boundary_clean.ipynb` |
| Majority task | `experiments/majority/majority_handover.ipynb` |
| TinyStories language model | `experiments/tinystories_lm/` |

## Results, storage and compute

Tasks are generated online from fixed seeds; no dataset is bundled. Notebooks write
append-only JSONL/CSV stores and rebuild their tables and figures from them, so an interrupted
run resumes instead of retraining. Results are written to `results/` by default; set
`RESULTS_DIR` to put them elsewhere.

**The notebooks are released without saved outputs**, so nothing here proves a number came from
this version of the code — re-running is the only check. The exception is
`experiments/tinystories_lm/`, which includes the evaluation logs of all 29 runs.

A GPU is needed in practice: the symbolic suite is 14.4 GPU-hours on a T4, and the TinyStories
experiments are about 17.6. Smoke presets run in minutes. These notebooks download public
models and datasets on first use: MNIST and CIFAR-10 (torchvision), WikiText-2 and TinyStories,
GPT-2, Falcon-H1, Mamba-2, and the WorM pretrained checkpoints. Model checkpoints are not
committed.

## Known gaps

Three baselines in the steering comparison — delayed attention, weight decay, and the L1
retrieval penalty — have no implementation here. Their reported values are preserved with
provenance in `results/appendix_p/table22_reported.csv`. The audit lists the remaining open
mismatches.

## License

MIT, see `LICENSE`.
