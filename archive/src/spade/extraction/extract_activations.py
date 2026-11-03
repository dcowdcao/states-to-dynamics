"""Step 1 of the OthelloGPT SAE pipeline: extract training data.

Research context:
    We train Sparse Autoencoders (SAEs) on OthelloGPT's residual stream to find
    interpretable features that correspond to board-state concepts (piece positions,
    legal moves, etc.). OthelloGPT is an 8-layer transformer (512-dim residual stream)
    trained to predict legal Othello moves from move sequences alone — it receives no
    explicit board representation, yet must internally represent the board to play
    correctly. SAE features should recover these implicit representations.

What this script does:
    1. Streams Othello game sequences from HuggingFace
    2. Runs each game through OthelloGPT, caching layer-5 residual stream activations
    3. Computes ground-truth board states (mine/yours encoding, 3 channels) and valid
       moves using Karvonen's Othello engine
    4. Saves train/test splits of activations + labels for downstream SAE training
       and linear probe evaluation

Key conventions:
    - HuggingFace tokens use indices 1–60 (60 playable squares); Karvonen's engine
      uses 0–63 board positions (all 64 squares). tokens_to_board_positions() bridges this.
    - Board states use "mine/yours" perspective (channels flip each turn):
      channel 0 = opponent, channel 1 = empty, channel 2 = current player
    - We extract turns 4–56 (configurable), skipping the fixed opening and
      deterministic endgame, to focus on strategically interesting positions.

Usage: python -m spade.extraction.extract_activations
"""

import torch
import json
import os
import argparse
import time

import numpy as np
from tqdm import tqdm

from spade.common.config import HF_REVISIONS, provenance
from spade.common.device import select_device
from spade.common.net import force_ipv4
from spade.common.paths import add_karvonen_to_path, repo_root

force_ipv4()
add_karvonen_to_path()

from circuits.othello_utils import (
    games_batch_to_state_stack_BLRRC,
    games_batch_to_state_stack_mine_yours_BLRRC,
    games_batch_to_state_stack_mine_yours_blank_mask_BLRRC,
    games_batch_to_valid_moves_BLRRC,
)
from circuits.othello_engine_utils import to_string

# Karvonen reproduction protocol constants
_KARVONEN_DATASET = "adamkarvonen/othello_45MB_games"
_KARVONEN_TURN_START = 0
_KARVONEN_TURN_END = 59


def tokens_to_board_positions(games_tokens):
    """Convert dataset tokens (1-60) to board positions (0-63) for Karvonen utils.

    The HuggingFace dataset uses token indices 1-60 (60 valid board positions),
    but OthelloBoardState.umpire() expects board positions 0-63 (skipping center 4).
    """
    return [to_string(game) for game in games_tokens]


def collect_games(dataset_name, n_games, seed=42):
    """Stream games from HuggingFace and collect as list of token lists.

    If n_games=0, load all available games from the dataset.
    """
    from datasets import load_dataset

    if n_games == 0:
        print(f"Loading ALL available games from {dataset_name}...")
        dataset = load_dataset(dataset_name, split="train", revision=HF_REVISIONS[dataset_name])
        games = [example["tokens"] for example in tqdm(dataset, desc="Collecting games")]
        print(f"Collected {len(games)} games, each with {len(games[0])} moves")
        return games

    print(f"Loading {n_games} games from {dataset_name}...")
    dataset = load_dataset(dataset_name, split="train", streaming=True,
                           revision=HF_REVISIONS[dataset_name])
    # A reservoir over the corpus's leading rows, not a sample of it — docs/game_corpus.md
    dataset = dataset.shuffle(seed=seed, buffer_size=10000)

    games = []
    pbar = tqdm(total=n_games, desc="Collecting games")
    for example in dataset:
        if len(games) >= n_games:
            break
        games.append(example["tokens"])
        pbar.update(1)
    pbar.close()

    print(f"Collected {len(games)} games, each with {len(games[0])} moves")
    return games


def extract_activations_batch(model, tokens_batch, hook_point, device):
    """Extract residual stream activations for a batch of games.

    Args:
        model: HookedTransformer
        tokens_batch: [batch, seq_len] int tensor
        hook_point: str, e.g. "blocks.5.hook_resid_post"
        device: str

    Returns:
        activations: [batch, seq_len, d_model] float tensor (on CPU)
        logits: [batch, seq_len, vocab_size] float tensor (on CPU)
    """
    tokens_batch = tokens_batch.to(device)
    logits, cache = model.run_with_cache(tokens_batch, names_filter=hook_point)
    activations = cache[hook_point].cpu()  # move to CPU immediately
    logits = logits.cpu()
    return activations, logits


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract OthelloGPT activations and board states for evaluation')
    parser.add_argument('--model_name', type=str,
                        default="Baidicoot/Othello-GPT-Transformer-Lens")
    parser.add_argument('--dataset_name', type=str,
                        default="adamkarvonen/othello_45MB_games")
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=8)
    parser.add_argument('--layer', type=int, default=5)
    parser.add_argument('--hook_point', type=str,
                        default="blocks.5.hook_resid_post")
    parser.add_argument('--n_games', type=int, default=100000,
                        help='Number of games to extract (0 = all)')
    parser.add_argument('--turn_start', type=int, default=0)
    parser.add_argument('--turn_end', type=int, default=59)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--train_split', type=float, default=0.9)
    parser.add_argument('--output_dir', type=str, default='data/activations')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--karvonen_repro', action='store_true',
                        help='Use Karvonen exact protocol: '
                             'adamkarvonen/othello_45MB_games, '
                             'full 59-token context [0,59)')
    args = parser.parse_args()

    # --karvonen_repro: override dataset and turn window
    if args.karvonen_repro:
        print("=== KARVONEN REPRO MODE ===")
        print(f"  Overriding dataset: {args.dataset_name} -> {_KARVONEN_DATASET}")
        print(f"  Overriding turn window: [{args.turn_start},{args.turn_end}) -> "
              f"[{_KARVONEN_TURN_START},{_KARVONEN_TURN_END})")
        args.dataset_name = _KARVONEN_DATASET
        args.turn_start = _KARVONEN_TURN_START
        args.turn_end = _KARVONEN_TURN_END

    # Resolve relative paths against the experiment directory
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(str(repo_root()), args.output_dir)

    device = select_device(args.device)
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect games from HuggingFace
    all_games = collect_games(args.dataset_name, args.n_games, seed=args.seed)

    # Load OthelloGPT
    from transformer_lens import HookedTransformer
    torch.set_grad_enabled(False)  # inference only — save memory

    print(f"\nLoading model: {args.model_name}")
    model = HookedTransformer.from_pretrained(
        args.model_name, revision=HF_REVISIONS[args.model_name], device=device)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Config: n_layers={model.cfg.n_layers}, d_model={model.cfg.d_model}")

    # Process in batches
    n_turns = args.turn_end - args.turn_start  # number of turns to keep per game
    batch_size = args.batch_size

    # Accumulate per-batch results; will concatenate after loop
    all_activations = []       # residual stream vectors for SAE training
    all_board_states = []      # ground-truth board state labels for evaluation
    all_board_states_blank_mask = []  # blank-mask variant (empty channel zeroed)
    all_valid_moves = []       # ground-truth legal moves for evaluation
    all_predicted_moves = []   # model's top-1 predictions (for error analysis)
    all_game_tokens = []       # raw game tokens (for reproducibility / downstream use)

    n_batches = (len(all_games) + batch_size - 1) // batch_size
    print(f"\nExtracting activations: {len(all_games)} games in {n_batches} batches of {batch_size}")
    print(f"Turn range: [{args.turn_start}, {args.turn_end}), {n_turns} turns per game")

    tic = time.perf_counter()

    for batch_idx in tqdm(range(n_batches), desc="Extracting"):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(all_games))
        batch_games = all_games[start:end]
        actual_batch = len(batch_games)

        # Convert to tensor for model input
        tokens_full = torch.tensor(batch_games, dtype=torch.long)  # [B, 60]
        all_game_tokens.append(tokens_full)

        # OthelloGPT has n_ctx=59: it sees moves 0–58 and predicts moves 1–59.
        # The 60th token exists in the dataset but exceeds the model's context window.
        tokens_model = tokens_full[:, :59]  # [B, 59]

        # Extract activations
        # Activation at position i = residual stream after processing token i
        activations, logits = extract_activations_batch(
            model, tokens_model, args.hook_point, device
        )
        # activations shape: [B, 59, 512]
        # logits shape: [B, 59, vocab_size]

        # Slice to turn range and flatten batch+turn dims into a single sample dim.
        # Each sample = one (game, turn) pair's residual stream activation vector.
        act_slice = activations[:, args.turn_start:args.turn_end, :]  # [B, n_turns, 512]
        all_activations.append(act_slice.reshape(-1, args.d_model))  # [B*n_turns, 512]

        # Predicted next moves (argmax of logits)
        pred_moves = logits[:, args.turn_start:args.turn_end, :].argmax(dim=-1)  # [B, n_turns]
        all_predicted_moves.append(pred_moves.reshape(-1))  # [B*n_turns]

        # Convert dataset tokens (1-60) to board positions (0-63) for Karvonen utils
        batch_board_positions = tokens_to_board_positions(batch_games)

        # Compute ground-truth board states in mine/yours encoding.
        # "mine/yours" flips perspective each turn so the SAE sees a consistent
        # representation regardless of which color is playing.
        board_states_BLRRC = games_batch_to_state_stack_mine_yours_BLRRC(batch_board_positions)
        # board_states_BLRRC shape: [B, 60, 8, 8, 3] — state[i] = board after move i
        bs_slice = board_states_BLRRC[:, args.turn_start:args.turn_end, :, :, :]
        all_board_states.append(bs_slice.reshape(-1, 8, 8, 3))

        # Blank-mask board states: same as mine_yours but with empty channel zeroed.
        # Karvonen et al. evaluate against this variant; it tests whether SAE features
        # can identify piece ownership without relying on the "empty" signal.
        blank_mask_BLRRC = games_batch_to_state_stack_mine_yours_blank_mask_BLRRC(batch_board_positions)
        bm_slice = blank_mask_BLRRC[:, args.turn_start:args.turn_end, :, :, :]
        all_board_states_blank_mask.append(bm_slice.reshape(-1, 8, 8, 3))

        # Valid moves
        valid_moves_BLRRC = games_batch_to_valid_moves_BLRRC(batch_board_positions)
        # valid_moves_BLRRC shape: [B, 60, 8, 8, 1] — valid_moves[i] = legal moves after move i
        vm_slice = valid_moves_BLRRC[:, args.turn_start:args.turn_end, :, :, :]
        all_valid_moves.append(vm_slice.reshape(-1, 8, 8, 1))

    toc = time.perf_counter()
    print(f"\nExtraction complete in {toc - tic:.1f}s")

    # Concatenate all batches
    print("Concatenating...")
    all_activations = torch.cat(all_activations, dim=0)  # [N, 512]
    all_board_states = torch.cat(all_board_states, dim=0)  # [N, 8, 8, 3]
    all_board_states_blank_mask = torch.cat(all_board_states_blank_mask, dim=0)  # [N, 8, 8, 3]
    all_valid_moves = torch.cat(all_valid_moves, dim=0)  # [N, 8, 8, 1]
    all_predicted_moves = torch.cat(all_predicted_moves, dim=0)  # [N]
    all_game_tokens = torch.cat(all_game_tokens, dim=0)  # [n_games, 60]

    N = all_activations.shape[0]
    print(f"Total activations: N={N}")
    print(f"  all_activations: {tuple(all_activations.shape)}")
    print(f"  all_board_states: {tuple(all_board_states.shape)}")
    print(f"  all_board_states_blank_mask: {tuple(all_board_states_blank_mask.shape)}")
    print(f"  all_valid_moves: {tuple(all_valid_moves.shape)}")
    print(f"  all_predicted_moves: {tuple(all_predicted_moves.shape)}")

    # Train/test split — split by game (not by activation) to prevent data leakage
    # from the same game appearing in both sets.
    n_games_total = len(all_games)
    n_games_train = int(n_games_total * args.train_split)
    n_train = n_games_train * n_turns  # each game contributes n_turns activation samples
    n_test = N - n_train

    print(f"\nSplit: {n_games_train} train games ({n_train} activations), "
          f"{n_games_total - n_games_train} test games ({n_test} activations)")

    # Split data
    train_acts = all_activations[:n_train]
    test_acts = all_activations[n_train:]
    train_bs = all_board_states[:n_train]
    test_bs = all_board_states[n_train:]
    train_bs_bm = all_board_states_blank_mask[:n_train]
    test_bs_bm = all_board_states_blank_mask[n_train:]
    train_vm = all_valid_moves[:n_train]
    test_vm = all_valid_moves[n_train:]
    train_pm = all_predicted_moves[:n_train]
    test_pm = all_predicted_moves[n_train:]
    train_tokens = all_game_tokens[:n_games_train]
    test_tokens = all_game_tokens[n_games_train:]

    # Save to disk
    print("\nSaving to disk...")
    save_pairs = [
        ("train_activations.pt", train_acts),
        ("test_activations.pt", test_acts),
        ("train_board_states.pt", train_bs),
        ("test_board_states.pt", test_bs),
        ("train_board_states_blank_mask.pt", train_bs_bm),
        ("test_board_states_blank_mask.pt", test_bs_bm),
        ("train_valid_moves.pt", train_vm),
        ("test_valid_moves.pt", test_vm),
        ("train_predicted_moves.pt", train_pm),
        ("test_predicted_moves.pt", test_pm),
        ("train_game_tokens.pt", train_tokens),
        ("test_game_tokens.pt", test_tokens),
    ]

    for fname, tensor in save_pairs:
        fpath = os.path.join(args.output_dir, fname)
        torch.save(tensor, fpath)
        size_mb = os.path.getsize(fpath) / 1e6
        print(f"  {fname}: {tuple(tensor.shape)}, {size_mb:.1f} MB")

    # Save metadata
    metadata = {
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "layer": args.layer,
        "hook_point": args.hook_point,
        "n_games": n_games_total,
        "n_games_train": n_games_train,
        "n_games_test": n_games_total - n_games_train,
        "turn_start": args.turn_start,
        "turn_end": args.turn_end,
        "n_turns_per_game": n_turns,
        "n_train": n_train,
        "n_test": n_test,
        "d_model": args.d_model,
        "activation_shape": list(train_acts.shape),
        "board_state_encoding": {
            "channels": ["opponent/white", "empty", "mine/black"],
            "note": "mine_yours perspective, flipped each turn"
        },
        "board_state_blank_mask_encoding": {
            "channels": ["opponent/white", "zero (masked)", "mine/black"],
            "note": "same as mine_yours but empty channel zeroed out (Karvonen blank_mask)"
        },
        "karvonen_repro": args.karvonen_repro,
        "seed": args.seed,
        **provenance(),
    }
    meta_path = os.path.join(args.output_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  metadata.json saved")

    # Quick sanity checks — catch extraction bugs early before expensive SAE training
    print("\n=== Sanity Checks ===")
    print(f"  Activations finite: {torch.isfinite(train_acts).all()}")
    print(f"  Activations mean: {train_acts.mean():.4f}, std: {train_acts.std():.4f}")
    # Each cell must be exactly one of {opponent, empty, mine} — one-hot across channel dim
    print(f"  Board states one-hot sum check (should be 1 everywhere): "
          f"{(train_bs.sum(dim=-1) == 1).all()}")
    # Typical Othello positions have ~5-15 legal moves; extreme values suggest a bug
    print(f"  Valid moves per position (mean): {train_vm.sum(dim=(1,2,3)).float().mean():.1f}")
    print(f"  Predicted moves range: [{train_pm.min()}, {train_pm.max()}]")

    print("\nDone!")
