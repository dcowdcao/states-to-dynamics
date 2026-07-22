"""SAE loading + standardization wrapper + board-reconstruction metric functions.

Shared hub used by both the main eval path (run_eval.py) and the legality study. Loads a
checkpoint back into the right class from its `trainer_class` string — dispatch order
matters: match CosineSpaDE before SpaDE, BatchTopK before TopK (substring collisions).
`StandardizedDictionary` re-applies the training-time center+scale at eval.

Implements Karvonen's train/test board-probe protocol: learn feature→board-cell
assignments on a train split, score coverage + reconstruction F1 on the test split (middle
4 squares masked).
"""

# def load_karvonen_sae(checkpoint_dir): ...        # dispatch by trainer_class; lazy circuits/spade imports
# def load_standardization_params(checkpoint_dir): ...
# class StandardizedDictionary: ...                 # wraps a Dictionary; center+scale on encode
# def compute_f1_per_feature(acts, board_states): ...   # -> f1_TFRRC.pt tensor
# def learn_feature_assignments(train_f1): ...
# def compute_coverage_from_assignments(...): ...
# def compute_board_reconstruction_from_assignments(...): ...
