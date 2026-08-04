# Linear probe

Standalone reproduction of the OthelloGPT board-state linear probe. `train_linear_probe.py`
produces `activations/linear_probe.pt` ([512,8,8,3], ~0.97 board-reconstruction accuracy),
the ground-truth reference the geometry studies compare SAE features against. It confirms
layer-5 activations contain the board state, so the bottleneck is SAE feature design, not
information loss. Large `.pt` under `activations/` is gitignored.

The committed `activations/board_states.pt` has **53 mislabeled rows in 10,000**: parity was
taken from move index rather than engine replay, and forfeits desync the two. Derive parity
from replay in anything new (`docs/claims_status.md`).
