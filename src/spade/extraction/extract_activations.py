"""Pipeline stage 1 — extract SAE training/eval data.

Stream OthelloGPT games, capture layer-5 resid_post activations, and compute the
ground-truth labels every downstream metric needs: board states (mine/yours/blank per
square), legal moves, predicted moves, raw game tokens. Save train/test splits as .pt +
a metadata.json.

Entrypoint: `python -m spade.extraction.extract_activations --output_dir data/activations
             --device auto`
"""

# def collect_games(n_games): ...                       # stream from the dataset
# def tokens_to_board_positions(tokens): ...            # ground-truth board per move
# def extract_activations_batch(model, games): ...      # NNsight capture at HOOK_POINT
# def main(): ...   # argparse: --output_dir, --n_games, --device; writes {train,test}.pt + metadata.json
