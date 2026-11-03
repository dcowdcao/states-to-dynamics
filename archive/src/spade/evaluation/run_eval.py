"""
Run Karvonen-style evaluation on SAEs trained by the spade.training entrypoints.

Computes five metrics for each SAE:
  1. Loss Recovered (Fidelity) — fraction of LM loss preserved by SAE
     reconstruction, computed by running the model 3x (original, reconstructed,
     zeroed) via Karvonen's loss_recovered() function.
  2. L0 — mean active features per sample (sparsity).
  3. Variance Explained — R^2 of SAE reconstruction vs raw activations.
  4. Coverage — mean best-F1 across 192 board-state labels.
  5. Reconstruction F1 — macro F1 when composing features to predict boards.

Usage:
    python -m spade.evaluation.run_eval --checkpoint_dir checkpoints/main/trainer_0 \\
        --test_data_dir data/activations --output_dir results/main

Outputs:
    results/main/<run_name>/metrics.json    — all metrics
    results/main/<run_name>/coverage.json   — per-position best-F1
    results/main/<run_name>/reconstruction_f1.json
    results/main/<run_name>/f1_TFRRC.pt     — raw F1 tensor
    results/main/all_results.csv            — one-row append (idempotent)
"""

import argparse
import csv
import json
import os
import time

import torch
from tqdm import tqdm

from spade.common.device import select_device
from spade.common.net import force_ipv4
from spade.common.paths import add_karvonen_to_path, repo_root

force_ipv4()
add_karvonen_to_path()

from spade.evaluation.karvonen_adapter import (
    load_karvonen_sae,
    load_standardization_params,
    StandardizedDictionary,
    compute_f1_per_feature,
    learn_feature_assignments,
    compute_coverage_from_assignments,
    compute_board_reconstruction_from_assignments,
)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_l0(feature_acts):
    """Mean number of active features per sample (Karvonen: f != 0)."""
    return (feature_acts != 0).float().sum(dim=-1).mean().item()


def compute_variance_explained(original, reconstructed):
    """Fraction of variance in original explained by reconstructed (R^2)."""
    ss_res = ((original - reconstructed) ** 2).sum().item()
    ss_tot = ((original - original.mean(dim=0, keepdim=True)) ** 2).sum().item()
    if ss_tot == 0:
        return 1.0
    return 1.0 - ss_res / ss_tot


def compute_frac_recovered(wrapped_ae, device, n_batches=10, batch_size=32):
    """Compute Loss Recovered (Fidelity) for OthelloGPT.

    Implements Karvonen's loss_recovered logic directly for OthelloGPT,
    which lacks a tokenizer and cannot use the upstream loss_recovered()
    function (it probes with model.trace("_"), requiring a tokenizer).

    Runs OthelloGPT 3 times per batch via nnsight model.trace():
      1. Original activations → loss_orig
      2. SAE-reconstructed activations → loss_recon
      3. Zeroed activations → loss_zero

    Then: frac_recovered = (loss_recon - loss_zero) / (loss_orig - loss_zero)

    Parameters
    ----------
    wrapped_ae : StandardizedDictionary
        SAE wrapped with standardization (handles raw activations).
    device : str
    n_batches : int — number of batches to average over
    batch_size : int — games per batch

    Returns
    -------
    frac_recovered : float
    """
    from circuits.utils import (
        othello_hf_dataset_to_generator,
        get_model,
        get_submodule,
        SubmoduleType,
    )

    model_name = "Baidicoot/Othello-GPT-Transformer-Lens"
    dataset_name = "taufeeque/othellogpt"
    context_length = 59
    layer = 5

    print("  Loading OthelloGPT for loss_recovered computation...")
    data_gen = othello_hf_dataset_to_generator(
        dataset_name, context_length=context_length, split="train", streaming=True
    )
    model = get_model(model_name, device)
    submodule = get_submodule(model_name, layer, model, SubmoduleType.resid_post)

    total_frac = 0.0
    total_orig = 0.0
    total_recon = 0.0
    total_zero = 0.0

    for i in tqdm(range(n_batches), desc="Loss recovered"):
        # Collect a batch of game token sequences
        batch = []
        for _ in range(batch_size):
            tokens = next(data_gen)
            if not isinstance(tokens, torch.Tensor):
                tokens = torch.tensor(tokens, dtype=torch.long)
            batch.append(tokens)
        text = torch.stack(batch).to(device)

        # 1. Unmodified logits
        with model.trace(text):
            logits_original = model.output.save()
        logits_original = logits_original.value

        # 2. Extract activations, reconstruct with SAE, then intervene
        with model.trace(text):
            x = submodule.output.save()

        x_hat = wrapped_ae(x).to(logits_original.dtype)

        with model.trace(text):
            submodule.output[:] = x_hat
            logits_reconstructed = model.output.save()
        logits_reconstructed = logits_reconstructed.value

        # 3. Zero ablation
        with model.trace(text):
            x_zero = submodule.output
            submodule.output[:] = torch.zeros_like(x_zero)
            logits_zero = model.output.save()
        logits_zero = logits_zero.value

        # Handle HookedTransformer output format
        try:
            logits_original = logits_original.logits
            logits_reconstructed = logits_reconstructed.logits
            logits_zero = logits_zero.logits
        except AttributeError:
            pass

        # Compute cross-entropy losses (next-token prediction)
        losses = []
        for logits in [logits_original, logits_reconstructed, logits_zero]:
            loss = torch.nn.CrossEntropyLoss()(
                logits[:, :-1, :].reshape(-1, logits.shape[-1]),
                text[:, 1:].reshape(-1),
            )
            losses.append(loss)

        loss_orig, loss_recon, loss_zero = losses
        frac = (loss_recon - loss_zero) / (loss_orig - loss_zero)
        total_frac += frac.item()
        total_orig += loss_orig.item()
        total_recon += loss_recon.item()
        total_zero += loss_zero.item()

    avg_frac = total_frac / n_batches
    print(f"  loss_orig={total_orig / n_batches:.4f}, "
          f"loss_recon={total_recon / n_batches:.4f}, "
          f"loss_zero={total_zero / n_batches:.4f}")
    print(f"  frac_recovered = {avg_frac:.4f}")

    return avg_frac


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Run Karvonen evaluation on a trained SAE')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory containing ae.pt and config.json '
                             '(output of trainSAE)')
    parser.add_argument('--test_data_dir', type=str, default='data/activations',
                        help='Directory with test_activations.pt and '
                             'test_board_states.pt')
    parser.add_argument('--output_dir', type=str, default='results/main',
                        help='Base output directory')
    parser.add_argument('--n_samples', type=int, default=0,
                        help='Number of test samples (0 = all)')
    parser.add_argument('--n_fidelity_batches', type=int, default=10,
                        help='Number of batches for loss_recovered averaging')
    parser.add_argument('--fidelity_batch_size', type=int, default=32,
                        help='Games per batch for loss_recovered')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Feature-batch size for F1 computation')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--skip_fidelity', action='store_true',
                        help='Skip loss_recovered computation (faster)')
    args = parser.parse_args()

    # Resolve relative paths
    for attr in ('checkpoint_dir', 'test_data_dir', 'output_dir'):
        val = getattr(args, attr)
        if val and not os.path.isabs(val):
            setattr(args, attr, os.path.join(str(repo_root()), val))

    device = select_device(args.device)
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load SAE and standardization params
    # ------------------------------------------------------------------
    print(f"\nLoading SAE from {args.checkpoint_dir} ...")
    ae, trainer_config = load_karvonen_sae(args.checkpoint_dir, device=device)

    activation_dim = trainer_config["activation_dim"]
    dict_size = trainer_config["dict_size"]
    sae_type = trainer_config.get("trainer_class",
                                   trainer_config.get("wandb_name", "unknown"))
    wandb_name = trainer_config.get("wandb_name", sae_type)
    print(f"  type={sae_type}, dict_size={dict_size}, "
          f"activation_dim={activation_dim}")

    # Load standardization params and wrap SAE
    std_params = load_standardization_params(args.checkpoint_dir, device)
    mean = torch.tensor(std_params["mean"], device=device, dtype=torch.float32)
    scale = std_params["scale"]
    wrapped_ae = StandardizedDictionary(ae, mean, scale).to(device)
    print(f"  Standardization: scale={scale:.6f}, mean_norm={mean.norm():.4f}")

    # Build a descriptive run name: e.g. "trainer_0_PAnnealTrainer_d4096"
    base_name = os.path.basename(os.path.normpath(args.checkpoint_dir))
    run_name = f"{base_name}_{sae_type}_d{dict_size}"
    out_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 2. Loss Recovered (Fidelity) — requires running the LM
    # ------------------------------------------------------------------
    frac_recovered = None
    if not args.skip_fidelity:
        print("\n=== Computing Loss Recovered (Fidelity) ===")
        frac_recovered = compute_frac_recovered(
            wrapped_ae, device,
            n_batches=args.n_fidelity_batches,
            batch_size=args.fidelity_batch_size,
        )
    else:
        print("\nSkipping loss_recovered (--skip_fidelity)")

    # ------------------------------------------------------------------
    # 3. Load train + test data for board-state analysis
    #    (Karvonen protocol: learn feature assignments on train, evaluate on test)
    # ------------------------------------------------------------------
    print(f"\nLoading data from {args.test_data_dir} ...")

    # Load all data to CPU first — feature_acts tensors are too large for GPU
    # (e.g., 5.3M samples × 8192 features = ~174GB at float32)
    train_acts = torch.load(os.path.join(args.test_data_dir, 'train_activations.pt'),
                            map_location='cpu', weights_only=True)
    train_bs = torch.load(os.path.join(args.test_data_dir, 'train_board_states.pt'),
                          map_location='cpu', weights_only=True)
    print(f"  Train: N={train_acts.shape[0]}, d_model={train_acts.shape[1]}")

    # Test data — used for final metric computation
    test_acts = torch.load(os.path.join(args.test_data_dir, 'test_activations.pt'),
                           map_location='cpu', weights_only=True)
    test_bs = torch.load(os.path.join(args.test_data_dir, 'test_board_states.pt'),
                         map_location='cpu', weights_only=True)

    # Optional blank-mask board states (Karvonen protocol)
    train_bm_path = os.path.join(args.test_data_dir,
                                  'train_board_states_blank_mask.pt')
    test_bm_path = os.path.join(args.test_data_dir,
                                 'test_board_states_blank_mask.pt')
    train_bs_bm = None
    test_bs_bm = None
    if os.path.exists(train_bm_path) and os.path.exists(test_bm_path):
        train_bs_bm = torch.load(train_bm_path, map_location='cpu',
                                  weights_only=True)
        test_bs_bm = torch.load(test_bm_path, map_location='cpu',
                                 weights_only=True)
        print(f"  Loaded blank-mask board states (train + test)")

    if args.n_samples > 0:
        train_acts = train_acts[:args.n_samples]
        train_bs = train_bs[:args.n_samples]
        test_acts = test_acts[:args.n_samples]
        test_bs = test_bs[:args.n_samples]
        if train_bs_bm is not None:
            train_bs_bm = train_bs_bm[:args.n_samples]
            test_bs_bm = test_bs_bm[:args.n_samples]

    N = test_acts.shape[0]
    C = test_bs.shape[-1]
    print(f"  Test: N={N}, d_model={test_acts.shape[1]}, classes={C}")

    # ------------------------------------------------------------------
    # 4. Encode train + test activations through wrapped SAE
    # ------------------------------------------------------------------
    encode_batch = 256  # safe for all architectures including SpaDE

    print("\nEncoding train activations (for feature assignment learning) ...")
    tic = time.perf_counter()
    N_train = train_acts.shape[0]
    n_train_batches = (N_train + encode_batch - 1) // encode_batch
    with torch.no_grad():
        train_feat_chunks = []
        for i in tqdm(range(0, N_train, encode_batch), total=n_train_batches, desc="Encode train"):
            x = train_acts[i:i + encode_batch].to(device)
            f = wrapped_ae.encode(x)
            train_feat_chunks.append(f.cpu())
        train_feature_acts = torch.cat(train_feat_chunks, dim=0)  # CPU
    del train_acts
    toc = time.perf_counter()
    print(f"  Encoded {N_train} train samples in {toc - tic:.1f}s  (stored on CPU)")

    print("Encoding test activations ...")
    tic = time.perf_counter()
    n_test_batches = (N + encode_batch - 1) // encode_batch
    with torch.no_grad():
        feat_chunks, recon_chunks = [], []
        for i in tqdm(range(0, N, encode_batch), total=n_test_batches, desc="Encode test"):
            x = test_acts[i:i + encode_batch].to(device)
            f = wrapped_ae.encode(x)
            r = wrapped_ae.decode(f)
            feat_chunks.append(f.cpu())
            recon_chunks.append(r.cpu())
        feature_acts = torch.cat(feat_chunks, dim=0)    # CPU
        reconstructed = torch.cat(recon_chunks, dim=0)   # CPU
    toc = time.perf_counter()
    print(f"  Encoded {N} test samples in {toc - tic:.1f}s  (stored on CPU)")

    # ------------------------------------------------------------------
    # 5. Board-state metrics (Karvonen train/test protocol)
    #    Step 1: Compute F1 on TRAIN data → learn feature assignments
    #    Step 2: Compute F1 on TEST data → evaluate using train assignments
    # ------------------------------------------------------------------
    thresholds_T = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    print(f"\nStep 1: Computing F1 on TRAIN data ({len(thresholds_T)} thresholds) ...")
    tic = time.perf_counter()
    f1_train_TFRRC = compute_f1_per_feature(
        train_feature_acts, train_bs, thresholds_T, batch_size=args.batch_size)
    toc = time.perf_counter()
    print(f"  Done in {toc - tic:.1f}s")

    # Learn feature-to-label assignments from training data
    assignments = learn_feature_assignments(f1_train_TFRRC, thresholds_T)

    # NOTE: train_feature_acts needed for blank-mask evaluation below;
    # free train_bs and f1_train_TFRRC now, free train_feature_acts after blank-mask block.
    del train_bs, f1_train_TFRRC

    print(f"\nStep 2: Computing F1 on TEST data ...")
    tic = time.perf_counter()
    f1_TFRRC = compute_f1_per_feature(
        feature_acts, test_bs, thresholds_T, batch_size=args.batch_size)
    toc = time.perf_counter()
    print(f"  Done in {toc - tic:.1f}s")

    # Coverage and reconstruction using train-derived assignments on test data
    coverage, best_f1_RRC = compute_coverage_from_assignments(
        f1_TFRRC, assignments)
    print(f"\nCoverage = {coverage:.4f}")

    print("\nComputing board reconstruction (train assignments → test data) ...")
    recon_results = compute_board_reconstruction_from_assignments(
        feature_acts, test_bs, assignments, thresholds_T)
    print(f"  Reconstruction F1 = {recon_results['f1']:.4f}")

    # Blank-mask evaluation (same train/test protocol)
    bm_coverage = None
    bm_recon_results = None
    if train_bs_bm is not None and test_bs_bm is not None:
        print("\n--- Blank-mask evaluation (train/test protocol) ---")
        f1_train_bm = compute_f1_per_feature(
            train_feature_acts, train_bs_bm, thresholds_T,
            batch_size=args.batch_size)
        bm_assignments = learn_feature_assignments(f1_train_bm, thresholds_T)

        f1_bm = compute_f1_per_feature(
            feature_acts, test_bs_bm, thresholds_T, batch_size=args.batch_size)
        bm_coverage, bm_best_f1_RRC = compute_coverage_from_assignments(
            f1_bm, bm_assignments)
        bm_recon_results = compute_board_reconstruction_from_assignments(
            feature_acts, test_bs_bm, bm_assignments, thresholds_T)
        print(f"  Blank-mask coverage = {bm_coverage:.4f}, "
              f"F1 = {bm_recon_results['f1']:.4f}")

    # Free remaining train data
    del train_feature_acts
    if train_bs_bm is not None:
        del train_bs_bm

    # ------------------------------------------------------------------
    # 6. L0 and variance explained
    # ------------------------------------------------------------------
    l0 = compute_l0(feature_acts)
    var_explained = compute_variance_explained(test_acts, reconstructed)
    print(f"\nL0 = {l0:.2f}")
    print(f"Variance explained = {var_explained:.4f}")
    if frac_recovered is not None:
        print(f"Loss Recovered (Fidelity) = {frac_recovered:.4f}")

    # ------------------------------------------------------------------
    # 7. Save results
    # ------------------------------------------------------------------
    print(f"\nSaving results to {out_dir}/ ...")

    coverage_data = {
        'coverage': coverage,
        'best_f1_per_position': best_f1_RRC.tolist(),
    }
    with open(os.path.join(out_dir, 'coverage.json'), 'w') as f:
        json.dump(coverage_data, f, indent=2)

    recon_save = {
        'precision': recon_results['precision'],
        'recall': recon_results['recall'],
        'f1': recon_results['f1'],
        'f1_per_square': recon_results['f1_per_square'].tolist(),
        'precision_per_square': recon_results['precision_per_square'].tolist(),
        'recall_per_square': recon_results['recall_per_square'].tolist(),
    }
    with open(os.path.join(out_dir, 'reconstruction_f1.json'), 'w') as f:
        json.dump(recon_save, f, indent=2)

    metrics = {
        'sae_name': run_name,
        'sae_type': sae_type,
        'wandb_name': wandb_name,
        'activation_dim': activation_dim,
        'dict_size': dict_size,
        'n_samples': N,
        'frac_recovered': frac_recovered,
        'coverage': coverage,
        'reconstruction_f1': recon_results['f1'],
        'reconstruction_precision': recon_results['precision'],
        'reconstruction_recall': recon_results['recall'],
        'l0': l0,
        'variance_explained': var_explained,
        'thresholds': thresholds_T,
    }
    if bm_coverage is not None:
        metrics['coverage_blank_mask'] = bm_coverage
        metrics['reconstruction_f1_blank_mask'] = bm_recon_results['f1']

    for key in ('gamma_reg', 'lr', 'lambda_init', 'k', 'seed',
                'initial_sparsity_penalty', 'l1_penalty',
                'sparsity_function', 'p_start', 'p_end'):
        if key in trainer_config:
            metrics[key] = trainer_config[key]

    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    torch.save(f1_TFRRC.cpu(), os.path.join(out_dir, 'f1_TFRRC.pt'))

    if bm_coverage is not None:
        bm_data = {
            'coverage': bm_coverage,
            'best_f1_per_position': bm_best_f1_RRC.cpu().tolist(),
        }
        with open(os.path.join(out_dir, 'coverage_blank_mask.json'), 'w') as f:
            json.dump(bm_data, f, indent=2)
        torch.save(f1_bm.cpu(), os.path.join(out_dir, 'f1_TFRRC_blank_mask.pt'))

    # ------------------------------------------------------------------
    # 8. Append to all_results.csv (idempotent)
    # ------------------------------------------------------------------
    csv_path = os.path.join(args.output_dir, 'all_results.csv')
    csv_fields = [
        'sae_name', 'sae_type', 'wandb_name',
        'activation_dim', 'dict_size',
        'frac_recovered', 'coverage', 'reconstruction_f1',
        'reconstruction_precision', 'reconstruction_recall',
        'coverage_blank_mask', 'reconstruction_f1_blank_mask',
        'l0', 'variance_explained',
        'seed', 'n_samples',
    ]
    row = {k: metrics.get(k, '') for k in csv_fields}

    import fcntl
    lock_path = csv_path + '.lock'
    with open(lock_path, 'w') as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        existing_rows = []
        if os.path.exists(csv_path):
            with open(csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                existing_rows = [r for r in reader
                                 if r.get('sae_name') != run_name]
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields,
                                    lineterminator='\n')
            writer.writeheader()
            writer.writerows(existing_rows)
            writer.writerow(row)
            f.flush()

    print(f"  Appended to {csv_path}")
    print("\nDone!")


if __name__ == '__main__':
    main()
