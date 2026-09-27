# Paper → code map

Every result in the paper, and the file that produces it. Appendix letters follow the
submitted manuscript. Notebooks are released **without outputs**: they
store their own results as append-only JSONL/CSV under their working directory and rebuild
every table and figure from that store.

| Paper element | File | Notes |
| --- | --- | --- |
| §3 setup, metrics `D_h`/`D_c`, share `S_c`, route pricing | `experiments/core/symbolic_main.ipynb` | model, task, price and measurement definitions used by every symbolic result |
| §4.1 four-arm late switch (symbolic) | `experiments/core/symbolic_main.ipynb` (`switch` block) | λ=0.03, switch at step 2000, constant LR, 12k steps, loads 8 and 12 |
| §4.2 + App B large-scale boundary test, deletion | `experiments/large_scale/large_scale_boundary_clean.ipynb` | see the parameter-count caveat in `REPRODUCIBILITY_AUDIT.md` |
| §4.3 matched post-hoc attenuation (MNIST, 8 seeds) | `experiments/visual/mnist_replication_8seed.ipynb` | α matched by bisection to the maintained-price reference |
| §4.4 availability / retrieval dropout | `experiments/core/symbolic_main.ipynb` (`availability`) | no penalty term in the objective |
| §4.5 long-horizon inversion | `experiments/visual/*_8seed.ipynb`, `experiments/core/symbolic_main.ipynb` (`inversion`) | 16k steps, constant LR, 8 seeds (visual) |
| §5.1–5.2 viability screen, load–cost boundary | `experiments/core/symbolic_main.ipynb` (`free`, `free_long`, `boundary`), `experiments/visual/*` | |
| §5.3 storage-width control | `experiments/core/symbolic_main.ipynb` (`width`) | `d_h` and `d_c` swept 32–160 at load 8 |
| §5.4 optimisation scaffold | `experiments/core/symbolic_main.ipynb` (`anneal`, `ood_pairs`) | budgets are not matched; stated in App I |
| §5.5 fusion rules + App G | `experiments/appendix_g_fusion/prefusion_check.ipynb` | **mismatch**: notebook uses λ=0.10 / 2500 steps, paper reports λ=0.25 / 16k |
| §6.1 silencing | `experiments/core/symbolic_main.ipynb` (`silencing`) | dead zone = pre-GELU activation < −2 |
| §6.2 reversibility + App K | `experiments/core/symbolic_main.ipynb` (`reversibility`, `reversibility_tail`) | |
| §6.3 swap errors + App M | `experiments/core/symbolic_main.ipynb` (`swap`) | 128-symbol alphabet |
| §6.3 OOD + App I | `experiments/core/symbolic_main.ipynb` (`ood_pairs`, `ood_eval`) | |
| App A accounting | `experiments/core/symbolic_main.ipynb` (final block) | |
| App C cross-representation replication | `experiments/visual/mnist_replication_8seed.ipynb`, `experiments/visual/cifar10_replication_8seed.ipynb` | |
| App D fused MQAR | `experiments/appendix_d_mqar/hybrid_mqar.ipynb` | `graded_necessity.ipynb` is an extra analysis, not reported |
| App E pretrained Falcon-H1 | `experiments/appendix_e_falcon_h1/falcon_h1_measurement.ipynb` | set `PRESET="full"` for the 0.5B model; see audit for the `D_SSM` note |
| App F pretrained Mamba-2 | `experiments/appendix_f_mamba2/` | needs GPU `mamba-ssm` / `causal-conv1d` builds |
| App H additional boundary results | `experiments/core/symbolic_main.ipynb` (`lamstar`, `costforms`) | |
| App J generality (cores, schedule) | `experiments/core/symbolic_main.ipynb` (`cores`, `schedule`, `equilibrium`, `replication`) | |
| App L predictors | `experiments/core/symbolic_main.ipynb` (`neg_*` blocks) | these probes use AdamW defaults (weight decay 0.01, no clipping) |
| App N attribution-proxy offsets | `experiments/appendix_n_attribution/offset_across_families.ipynb` | `attribution_comparison.ipynb` uses a different grid and is not the table source |
| App O pretrained GPT-2 IOI | `experiments/appendix_o_gpt2_ioi/gpt2_ioi_dissociation.ipynb` | set `MODEL_NAME` for Small / Medium |
| App P baselines, CRAM fidelity | `experiments/appendix_p_controls/`, `results/appendix_p/table22_reported.csv` | three baselines have **no code**; see below |
| App Q pretrained WorM VSR | `experiments/appendix_q_worm_vsr/` | `worm_vsr_diagnostic.ipynb` imports setup from `worm_vsr_setup.ipynb` |
| App R Majority (structurally distinct task) | `experiments/majority/majority_handover.ipynb` | |
| App S TinyStories language-model extension | `experiments/tinystories_lm/` | ships its logs; `python analyze.py --out results --check` rebuilds every reported number on CPU |

## Missing code

No implementation is included for three baseline rows: **delayed attention**, **weight decay**
and **L1 retrieval penalty**. Their reported values are preserved, with provenance, in
`results/appendix_p/table22_reported.csv`.

## Exploratory work

`experiments/pilots/` holds follow-up and superseded notebooks that no paper result depends
on. They are kept so the provenance of the released experiments is visible, and are not
maintained.
