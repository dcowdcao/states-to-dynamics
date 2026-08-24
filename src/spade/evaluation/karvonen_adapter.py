"""
Karvonen evaluation adapter
============================
Provides SAE loading for Karvonen trainSAE() checkpoint format, a
StandardizedDictionary wrapper for evaluation-time normalization, and
board-state evaluation functions (F1 per feature, coverage, board
reconstruction) from Karvonen et al. (2024).

Key data structure:
  f1_TFRRC : Tensor [T, F, 8, 8, C]
    T = number of binarization thresholds
    F = number of SAE features (dictionary_size)
    8x8 = board squares
    C = board-state classes (typically 3: opponent / empty / mine)
"""

import os
import json
import torch
import torch.nn as nn

from spade.common.paths import add_karvonen_to_path

add_karvonen_to_path()


# ---------------------------------------------------------------------------
# SAE loading from Karvonen trainSAE() checkpoint format
# ---------------------------------------------------------------------------

def load_karvonen_sae(checkpoint_dir, device="cpu"):
    """Load SAE from Karvonen trainSAE() checkpoint format.

    Expects:
        checkpoint_dir/ae.pt       -- state dict
        checkpoint_dir/config.json -- {"trainer": {...}, "buffer": {...}}

    Returns (ae, trainer_config) where ae is a Dictionary subclass.
    """
    config_path = os.path.join(checkpoint_dir, "config.json")
    ae_path = os.path.join(checkpoint_dir, "ae.pt")

    with open(config_path) as f:
        config = json.load(f)

    trainer_config = config.get("trainer", config)

    # Karvonen's 2024-05-23 checkpoints predate these two keys (e.g. standard/trainer25-v2),
    # and both are recoverable exactly rather than guessed: the buffer records the submodule
    # width, and the encoder's own weight carries (dict_size, activation_dim). Only reached
    # when the key is absent, which until now raised.
    shape = None
    if "activation_dim" not in trainer_config or "dict_size" not in trainer_config:
        weights = torch.load(ae_path, map_location="cpu", weights_only=True)
        key = next((k for k in ("encoder.weight", "W_enc") if k in weights), None)
        assert key, f"{ae_path}: no encoder weight to read the dimensions off"
        shape = tuple(weights[key].shape)
    activation_dim = trainer_config.get("activation_dim") or (
        config.get("buffer", {}).get("d_submodule") or shape[1])
    dict_size = trainer_config.get("dict_size") or shape[0]
    if "trainer_class" in trainer_config:
        trainer_class = trainer_config["trainer_class"]
    elif "wandb_name" in trainer_config:
        trainer_class = trainer_config["wandb_name"]
    else:
        raise KeyError(
            f"{config_path} has neither 'trainer_class' nor 'wandb_name'; "
            f"cannot determine which Dictionary subclass to instantiate."
        )

    # Instantiate the right dictionary class based on trainer type.
    # Order matters — "TopK" is a substring of "BatchTopK", and "SpaDE" is a
    # substring of "CosineSpaDE", so the more-specific branches must come first.
    if "BatchTopK" in trainer_class:
        from circuits.dictionary_learning.trainers.batch_top_k import BatchTopKSAE
        ae = BatchTopKSAE(activation_dim, dict_size, trainer_config["k"])
    elif "TopK" in trainer_class:
        from circuits.dictionary_learning.trainers.top_k import AutoEncoderTopK
        ae = AutoEncoderTopK(activation_dim, dict_size, trainer_config["k"])
    elif "Gated" in trainer_class:
        from circuits.dictionary_learning.dictionary import GatedAutoEncoder
        ae = GatedAutoEncoder(activation_dim, dict_size)
    elif "CosineSpaDE" in trainer_class:
        from spade.saes.spade_cosine import CosineSpaDEAutoEncoder
        ae = CosineSpaDEAutoEncoder(activation_dim, dict_size)
    elif "SpaDE" in trainer_class:
        from spade.saes.spade import SpaDEAutoEncoder
        ae = SpaDEAutoEncoder(activation_dim, dict_size)
    elif "Standard" in trainer_class or "PAnneal" in trainer_class:
        from circuits.dictionary_learning.dictionary import AutoEncoder
        ae = AutoEncoder(activation_dim, dict_size)
    else:
        raise ValueError(
            f"Unrecognized trainer_class {trainer_class!r} in {config_path}; "
            f"refusing to guess the Dictionary subclass."
        )

    state_dict = torch.load(ae_path, map_location=device, weights_only=True)
    # `k` and `threshold` are constructor-set buffers, not learned weights: k comes from
    # config.json and threshold defaults to -1, which `encode` ignores unless asked for it.
    # Karvonen's 2024-07-10 top_k checkpoints predate both being saved. Only these two may be
    # absent -- anything else missing is a real weight and still raises.
    DERIVED = {"k", "threshold"}
    missing, unexpected = ae.load_state_dict(state_dict, strict=False)
    assert not unexpected, f"{ae_path}: unexpected keys {unexpected}"
    assert set(missing) <= DERIVED, f"{ae_path}: missing weights {sorted(set(missing) - DERIVED)}"
    ae.to(device)
    ae.eval()

    return ae, trainer_config


def load_standardization_params(checkpoint_dir, device="cpu"):
    """Load standardization parameters saved by the training entrypoints.

    Searches for standardization_params.json in the checkpoint dir and its
    parent directory (the top-level save_dir where trainSAE saves all trainers).

    Returns dict with keys: "mean" (list), "scale" (float), "d_model" (int).
    """
    checkpoint_dir = os.path.normpath(checkpoint_dir)
    for search_dir in [checkpoint_dir, os.path.dirname(checkpoint_dir)]:
        path = os.path.join(search_dir, "standardization_params.json")
        if os.path.exists(path):
            with open(path) as f:
                params = json.load(f)
            return params

    raise FileNotFoundError(
        f"standardization_params.json not found in {checkpoint_dir} "
        f"or its parent directory"
    )


# ---------------------------------------------------------------------------
# StandardizedDictionary wrapper
# ---------------------------------------------------------------------------

class StandardizedDictionary(nn.Module):
    """Wraps a Karvonen Dictionary with center + scale standardization.

    During training, activations were transformed as:
        act_std = scale * (act - mean)

    This wrapper applies standardization before encode and inverse-transforms
    after decode, so the wrapped dictionary operates in raw activation space.

    Handles both 2D [batch, d_model] and 3D [batch, seq_len, d_model] inputs.
    """

    def __init__(self, dictionary, mean, scale):
        super().__init__()
        self.dictionary = dictionary
        self.register_buffer("mean", mean)  # [d_model]
        self.scale = scale  # scalar
        # Expose interface expected by Karvonen's loss_recovered()
        self.activation_dim = dictionary.activation_dim
        self.dict_size = dictionary.dict_size

    def _standardize(self, x):
        return self.scale * (x - self.mean)

    def _unstandardize(self, x_std):
        return x_std / self.scale + self.mean

    def forward(self, x, output_features=False):
        shape = x.shape
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])

        x_std = self._standardize(x)

        if output_features:
            x_hat_std, f = self.dictionary(x_std, output_features=True)
            x_hat = self._unstandardize(x_hat_std)
            return x_hat.view(shape), f
        else:
            x_hat_std = self.dictionary(x_std)
            x_hat = self._unstandardize(x_hat_std)
            return x_hat.view(shape)

    def encode(self, x):
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])
        return self.dictionary.encode(self._standardize(x))

    def decode(self, f):
        return self._unstandardize(self.dictionary.decode(f))


# ---------------------------------------------------------------------------
# Middle-square mask
# In standard Othello, the 4 center squares (d4, d5, e4, e5) are pre-occupied
# at game start and never change. Karvonen et al. exclude them from all metrics
# because they carry no learnable information.
# ---------------------------------------------------------------------------

_MIDDLE_SQUARES = {(3, 3), (3, 4), (4, 3), (4, 4)}  # 0-indexed board coords


def _non_middle_mask_RC():
    """Return a boolean tensor [8, 8] that is True for the 60 evaluable squares."""
    mask = torch.ones(8, 8, dtype=torch.bool)
    for r, c in _MIDDLE_SQUARES:
        mask[r, c] = False
    return mask


# ---------------------------------------------------------------------------
# F1 computation
# ---------------------------------------------------------------------------

def compute_f1_per_feature(feature_acts, board_states, thresholds_T,
                           batch_size=128, sample_chunk=8192,
                           compute_device=None):
    """Compute F1 for each (threshold, feature, row, col, class).

    For each SAE feature, binarize its activation at several threshold
    fractions of its max value.  Then treat "feature active" as a binary
    classifier for each (square, class) label and compute F1.

    Parameters
    ----------
    feature_acts : Tensor [N, F]  — may be on CPU (large tensors)
    board_states : Tensor [N, 8, 8, C]  — may be on CPU
    thresholds_T : list[float]
    batch_size : int — features processed together (controls GPU memory)
    sample_chunk : int — samples processed together
    compute_device : torch.device — device for computation (default: cuda if available)

    Returns
    -------
    f1_TFRRC : Tensor [T, F, 8, 8, C]  — on CPU
    """
    N, F = feature_acts.shape
    C = board_states.shape[-1]
    T = len(thresholds_T)

    if compute_device is None:
        compute_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    max_act = feature_acts.max(dim=0).values  # [F], on CPU

    rc_mask = _non_middle_mask_RC().to(compute_device)

    from tqdm import tqdm
    f1_TFRRC = torch.zeros(T, F, 8, 8, C)  # CPU
    n_feat_batches = (F + batch_size - 1) // batch_size

    for f_start in tqdm(range(0, F, batch_size), total=n_feat_batches, desc="F1 features"):
        f_end = min(f_start + batch_size, F)
        f_batch = f_end - f_start

        feat_slice = feature_acts[:, f_start:f_end]  # [N, f_batch], CPU
        max_slice = max_act[f_start:f_end]

        for t_idx, frac in enumerate(thresholds_T):
            thresh_vals = frac * max_slice
            pred = (feat_slice > thresh_vals.unsqueeze(0)).float()  # [N, f_batch], CPU

            tp = torch.zeros(f_batch, 8, 8, C, device=compute_device)
            fp = torch.zeros(f_batch, 8, 8, C, device=compute_device)
            fn = torch.zeros(f_batch, 8, 8, C, device=compute_device)

            for s_start in range(0, N, sample_chunk):
                s_end = min(s_start + sample_chunk, N)
                pred_c = pred[s_start:s_end].to(compute_device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                truth_c = board_states[s_start:s_end].to(compute_device).unsqueeze(1).float()
                tp += (pred_c * truth_c).sum(dim=0)
                fp += (pred_c * (1 - truth_c)).sum(dim=0)
                fn += ((1 - pred_c) * truth_c).sum(dim=0)

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            f1 = f1 * rc_mask.unsqueeze(0).unsqueeze(-1)
            f1_TFRRC[t_idx, f_start:f_end] = f1.cpu()

    return f1_TFRRC


# ---------------------------------------------------------------------------
# Train/test split protocol (Karvonen-matched)
# ---------------------------------------------------------------------------

def learn_feature_assignments(f1_TFRRC_train, thresholds_T):
    """Learn feature-to-label assignments from training-data F1 scores.

    For each feature, finds the (threshold, square, class) that gives the
    highest F1 on training data.

    Also identifies the best (threshold, feature) for each (square, class)
    to be used for coverage on test data.

    Returns dict with:
        best_t_per_feat : [F] int — best threshold index per feature
        best_r_per_feat : [F] int — best row per feature
        best_c_per_feat : [F] int — best col per feature
        best_cls_per_feat : [F] int — best class per feature
        best_thresh_frac : [F] float — actual threshold fraction per feature
        best_tf_per_label : (T_idx[R,C,cls], F_idx[R,C,cls]) — best (t,f) per label for coverage
    """
    T, F, _, _, C = f1_TFRRC_train.shape
    device = f1_TFRRC_train.device

    # Per-feature best assignment
    f1_flat = f1_TFRRC_train.reshape(T, F, -1)  # [T, F, 8*8*C]
    best_over_t_vals, best_over_t_idx = f1_flat.max(dim=0)  # [F, 8*8*C]
    best_over_pos_vals, best_over_pos_idx = best_over_t_vals.max(dim=1)  # [F]

    feat_range = torch.arange(F, device=device)
    pos_flat = best_over_pos_idx  # [F]
    best_t_per_feat = best_over_t_idx[feat_range, pos_flat]  # [F]

    best_cls_per_feat = pos_flat % C
    rc_idx = pos_flat // C
    best_r_per_feat = rc_idx // 8
    best_c_per_feat = rc_idx % 8

    thresholds_tensor = torch.tensor(thresholds_T, device=device)
    best_thresh_frac = thresholds_tensor[best_t_per_feat]

    # Per-label best (threshold, feature) for coverage
    # For each (r, c, cls), find the (t, f) with the highest F1 on train
    best_tf_vals = f1_TFRRC_train.reshape(T * F, 8, 8, C)
    best_tf_flat_idx = best_tf_vals.argmax(dim=0)  # [8, 8, C]
    best_tf_t_idx = best_tf_flat_idx // F  # [8, 8, C]
    best_tf_f_idx = best_tf_flat_idx % F   # [8, 8, C]

    return {
        'best_t_per_feat': best_t_per_feat,
        'best_r_per_feat': best_r_per_feat,
        'best_c_per_feat': best_c_per_feat,
        'best_cls_per_feat': best_cls_per_feat,
        'best_thresh_frac': best_thresh_frac,
        'best_tf_t_idx': best_tf_t_idx,
        'best_tf_f_idx': best_tf_f_idx,
    }


def compute_coverage_from_assignments(f1_TFRRC_test, assignments):
    """Compute coverage on test data using train-derived assignments.

    For each (square, class), looks up the best (threshold, feature) identified
    on training data, then reads that specific F1 value from the test F1 tensor.

    Returns (coverage, best_f1_RRC).
    """
    device = f1_TFRRC_test.device
    rc_mask = _non_middle_mask_RC().to(device)

    best_tf_t_idx = assignments['best_tf_t_idx'].to(device)  # [8, 8, C]
    best_tf_f_idx = assignments['best_tf_f_idx'].to(device)  # [8, 8, C]

    C = f1_TFRRC_test.shape[-1]
    best_f1_RRC = torch.zeros(8, 8, C, device=device)

    for r in range(8):
        for c in range(8):
            if not rc_mask[r, c]:
                continue
            for cls in range(C):
                t_idx = best_tf_t_idx[r, c, cls].item()
                f_idx = best_tf_f_idx[r, c, cls].item()
                best_f1_RRC[r, c, cls] = f1_TFRRC_test[t_idx, f_idx, r, c, cls]

    mask = rc_mask.unsqueeze(-1).expand_as(best_f1_RRC)
    coverage = best_f1_RRC[mask].mean().item()
    return coverage, best_f1_RRC


def compute_board_reconstruction_from_assignments(
    feature_acts_test, board_states_test, assignments, thresholds_T
):
    """Board reconstruction on test data using train-derived feature assignments.

    Uses the feature-to-label assignments learned on training data to
    reconstruct boards from test activations, then computes F1.
    """
    F = feature_acts_test.shape[1]
    N = feature_acts_test.shape[0]
    C = board_states_test.shape[-1]
    device = feature_acts_test.device

    max_act = feature_acts_test.max(dim=0).values  # [F]
    rc_mask = _non_middle_mask_RC().to(device)

    best_r_per_feat = assignments['best_r_per_feat'].to(device)
    best_c_per_feat = assignments['best_c_per_feat'].to(device)
    best_cls_per_feat = assignments['best_cls_per_feat'].to(device)
    best_thresh_frac = assignments['best_thresh_frac'].to(device)

    thresh_vals = best_thresh_frac * max_act
    active = (feature_acts_test > thresh_vals.unsqueeze(0)).float()

    flat_label = best_r_per_feat * (8 * C) + best_c_per_feat * C + best_cls_per_feat
    recon_flat = torch.zeros(N, 8 * 8 * C, device=device)
    flat_label_exp = flat_label.unsqueeze(0).expand(N, -1)
    recon_flat.scatter_add_(1, flat_label_exp, active)
    recon = (recon_flat.reshape(N, 8, 8, C) > 0).float()

    truth = board_states_test.float()
    mask = rc_mask.unsqueeze(0).unsqueeze(-1).expand_as(truth)

    tp = ((recon * truth) * mask).sum(dim=0)
    fp = ((recon * (1 - truth)) * mask).sum(dim=0)
    fn = (((1 - recon) * truth) * mask).sum(dim=0)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    mask_rc = rc_mask.unsqueeze(-1).expand(8, 8, C)
    macro_precision = precision[mask_rc].mean().item()
    macro_recall = recall[mask_rc].mean().item()
    macro_f1 = f1[mask_rc].mean().item()

    return {
        'precision': macro_precision,
        'recall': macro_recall,
        'f1': macro_f1,
        'f1_per_square': f1.cpu(),
        'precision_per_square': precision.cpu(),
        'recall_per_square': recall.cpu(),
    }
