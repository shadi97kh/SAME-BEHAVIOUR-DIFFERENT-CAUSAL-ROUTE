# Majority — a structurally distinct handover task

`majority_handover.ipynb` reproduces the Majority experiment: a sequence of `L` symbols with a
unique mode, three delay tokens, then a readout token whose target is the modal symbol. No
single position is informative, so the task cannot be solved by a one-position pointer, which
makes it structurally different from selective recall.

The model, training loop, price, resampling dependence and switch mechanics are taken verbatim
from `experiments/core/symbolic_main.ipynb`.

Set `PRESET` at the top: `"smoke"` (about 5 minutes) to check the pipeline, `"full"` for the
reported numbers. `RESUME = True` skips work already stored on disk.
