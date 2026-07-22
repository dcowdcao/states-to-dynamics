# Linear probe

Standalone reproduction of the OthelloGPT board-state linear probe. `train_linear_probe.py`
produces `activations/linear_probe.pt` ([512,8,8,3], ~0.97 board-reconstruction accuracy),
the ground-truth reference the geometry studies compare SAE features against. It confirms
layer-5 activations contain the board state, so the bottleneck is SAE feature design, not
information loss. Large `.pt` under `activations/` is gitignored.
