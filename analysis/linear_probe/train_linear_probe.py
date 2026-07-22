"""Train the gold-standard linear board-state probe on OthelloGPT layer-5 activations.

Extracts resid_post from the Li et al. OthelloGPT (Baidicoot/Othello-GPT-Transformer-Lens —
the model Karvonen's SAEs were trained on) across ~200 ARENA games, sweeps all 8 layers,
trains the full probe on the best layer. Result: linear_probe.pt of shape [512, 8, 8, 3]
at ~98.96% eval accuracy — reused as ground truth across the geometry studies.

Outputs to activations/ (gitignored .pt) + figures/. Imports ARENA part53_othellogpt utils
from reference/. Skeleton — port from the original analysis/linear_probe/.
"""

# def extract_layer_activations(model, games): ...
# def sweep_layers(...): ...
# def train_probe(acts, board_states): ...
# def main(): ...
