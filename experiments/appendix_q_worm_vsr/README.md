# Appendix Q — pretrained WorM VSR route availability

- `worm_vsr_diagnostic.ipynb` is the paper's diagnostic: a retrieval route is available over the
  frozen pretrained representation but is never load-bearing, so resample-defined route ordering
  becomes a free parameter while deletion stays free.
- `worm_vsr_setup.ipynb` holds the official WorM setup that the diagnostic imports.

Both download the official WorM repository and pretrained checkpoints. Result discovery prefers
repo-local files; set `RESULTS_DIR` (or `WORM_RESULT_DIR`) to relocate outputs. The scientific
logic is unchanged from the supplied version.

The human reference data that accompany the benchmark were never accessed and are not included.
