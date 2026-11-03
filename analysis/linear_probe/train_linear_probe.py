"""Historical: the ARENA-corpus probe fit, superseded by `spade.probes.train`.

Kept because it also writes the ARENA activation cache the bracket-flip, rules and
geometry code still reads. Its probe is not the project's board probe any more --
`spade.probes.load.board(5)` is -- and its 200/500-epoch budgets have no convergence
criterion, which is what `spade.probes.train` exists to fix. See README.md.

Extracts resid_post from the Li et al. OthelloGPT (the model Karvonen's SAEs were trained
on) across 200 ARENA games, sweeps the embedding space and all 8 layers with quick probes
over several initialisation seeds, then trains the full probe on the best layer.
`linear_probe.pt` of shape [512, 8, 8, 3] was the ground truth the geometry studies used
before the port.

Labels are mover-relative ({empty, theirs, mine}), and the mover is the player who actually
moves next (`spade.othello.boards.next_mover`), never move-index parity: Othello skips a
player with no legal move, so parity goes out of phase with the true mover for the rest of
that game. ARENA's construction (solutions.py:401-408) uses parity and mislabels 53 of these
10,000 rows, in 3 of the 200 games.

Writes activations/ (gitignored) and the tracked sweep table results/layer_sweep.csv,
which make_figures.py draws.

Usage: python analysis/linear_probe/train_linear_probe.py
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import einops
import numpy as np
import pandas as pd
import torch as t
import torch.nn.functional as F
from transformer_lens import HookedTransformer

from spade.common.config import CTX_LEN, D_MODEL, HF_REVISIONS, LAYER, MODEL_NAME
from spade.common.device import select_device
from spade.common.net import force_ipv4
from spade.common.paths import add_karvonen_to_path, linear_probe_activations, repo_root
from spade.othello.boards import next_mover

force_ipv4()   # before the HookedTransformer download; the cluster does not route IPv6
add_karvonen_to_path()

from circuits.othello_engine_utils import OthelloBoardState, to_string  # noqa: E402

t.set_grad_enabled(False)

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
# Verbatim from ARENA_3.0 (github.com/callummcdougall/ARENA_3.0). Vocab tokens 1..60;
# the engine's 0..63 squares are to_string() of them, so only this encoding is vendored.
ARENA_GAMES = (repo_root() / "reference/research_materials/ARENA/chapter1_transformer_interp"
               / "exercises/part53_othellogpt/board_seqs_id_small.npy")

NUM_GAMES = 200
BATCH_SIZE = 50
TURN_START, TURN_END = 5, 55
N_TRAIN = 8000
LR = 1e-2
SWEEP_EPOCHS, PROBE_EPOCHS = 200, 500
SWEEP_SEEDS = (42, 43, 44, 45, 46)   # probe init only; games and split are fixed
PROBE_SEED = 42
EMBED = "embed"   # resid_pre at block 0: the input space the layer sweep is anchored at


def replay(squares: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
    """Board state after each move and the colour that moves next, both from the engine.

    Returns `states` [games, moves, 8, 8] in absolute colours (1 black, -1 white, 0 empty)
    and `to_move` [games, moves], the colour that plays the *following* move -- the nominal
    mover unless they have to be skipped, decided from the board and not from the record.
    """
    n_games, n_moves = squares.shape
    states = t.zeros((n_games, n_moves, 8, 8), dtype=t.int32)
    to_move = t.zeros((n_games, n_moves), dtype=t.int32)
    for n in range(n_games):
        board = OthelloBoardState()
        for i in range(n_moves):
            board.umpire(int(squares[n, i]))
            states[n, i] = t.from_numpy(board.state)
            to_move[n, i] = next_mover(board.state, board.next_hand_color)
    return states, to_move


def check_mover(states: t.Tensor, to_move: t.Tensor, squares: t.Tensor) -> None:
    """The mover named at position i must be the one whose disc move i+1 puts on the board.

    `to_move` is decided from the board alone; the colour of the disc placed by the next move
    is read off the state array, which no mover code path touches. Move-index parity fails
    this check at every skip, which is what makes a desync loud instead of silent.
    """
    placed = states.flatten(2).gather(2, squares[..., None]).squeeze(-1)
    assert (to_move[:, :-1] == placed[:, 1:]).all(), \
        "mover disagrees with the colour of the disc the next move places"


def index_parity(n_moves: int) -> t.Tensor:
    """ARENA's mover: -1 (white) after an even move, +1 (black) after an odd one."""
    return (-1 + 2 * (t.arange(n_moves) % 2))[None, :]


@dataclass(frozen=True)
class ParityAudit:
    """What ARENA's move-index mover gets wrong, on the rows this probe keeps.

    Rows are `game * n_turns + (turn - TURN_START)` and the split is positional, so a
    desyncing game below `first_eval_game` moves the probe only through its fit, while one
    at or above it moves the reported eval accuracy itself.
    """
    slots: int
    rows: int
    games: list[int]
    eval_games: list[int]
    first_eval_game: int

    def report(self, n_slots: int, n_rows: int) -> str:
        eval_note = (f"IN THE EVAL SPLIT: games {self.eval_games}"
                     if self.eval_games else
                     f"none at or above game {self.first_eval_game}, so the eval split is "
                     f"unaffected and only the fit changes")
        return (f"move-index parity names the wrong mover at {self.slots}/{n_slots} game "
                f"slots; inside turns [{TURN_START}, {TURN_END}) it mislabels {self.rows}/"
                f"{n_rows} rows in games {self.games} -- {eval_note}")


def audit_parity(to_move: t.Tensor, labels: t.Tensor, parity_labels: t.Tensor) -> ParityAudit:
    window = slice(TURN_START, TURN_END)
    wrong = (labels != parity_labels)[:, window].flatten(2).any(-1)
    games = wrong.any(1).nonzero().flatten().tolist()
    first_eval_game = N_TRAIN // (TURN_END - TURN_START)
    return ParityAudit(
        slots=int((to_move != index_parity(to_move.shape[1])).sum()),
        rows=int(wrong.sum()),
        games=games,
        eval_games=[game for game in games if game >= first_eval_game],
        first_eval_game=first_eval_game)


def relative_labels(states: t.Tensor, to_move: t.Tensor) -> t.Tensor:
    """Absolute colours -> {0 empty, 1 theirs, 2 mine} for the player to move."""
    relative = states * to_move[..., None, None]
    labels = t.zeros_like(relative, dtype=t.int64)
    labels[relative == 1] = 2
    labels[relative == -1] = 1
    return labels


def residual_activations(model: HookedTransformer, games: t.Tensor,
                         device: str) -> tuple[t.Tensor, t.Tensor]:
    """resid_post at every layer (layer-major) and resid_pre at block 0, over
    positions [TURN_START, TURN_END)."""
    hooks = [f"blocks.{layer}.hook_resid_post" for layer in range(model.cfg.n_layers)]
    hooks.append("blocks.0.hook_resid_pre")
    collected = {hook: [] for hook in hooks}
    for start in range(0, len(games), BATCH_SIZE):
        batch = games[start:start + BATCH_SIZE, :CTX_LEN].to(device)
        _, cache = model.run_with_cache(batch, names_filter=lambda name: name in collected)
        for hook in hooks:
            collected[hook].append(cache[hook][:, TURN_START:TURN_END, :].cpu())
        del cache
        print(f"  activations for games {start}-{start + len(batch) - 1}")
    flat = {hook: t.cat(acts).reshape(-1, D_MODEL) for hook, acts in collected.items()}
    return t.stack([flat[hook] for hook in hooks[:-1]]), flat[hooks[-1]]


def fit_probe(acts: t.Tensor, labels: t.Tensor, epochs: int, seed: int) -> t.Tensor:
    t.manual_seed(seed)
    probe = (t.randn(D_MODEL, 8, 8, 3, device=acts.device) * 0.01).requires_grad_(True)
    optimizer = t.optim.Adam([probe], lr=LR)
    with t.enable_grad():
        for _ in range(epochs):
            logits = einops.einsum(acts, probe, "n d, d r c cls -> n r c cls")
            F.cross_entropy(logits.reshape(-1, 3), labels.reshape(-1)).backward()
            optimizer.step()
            optimizer.zero_grad()
    return probe.detach()


def accuracy(acts: t.Tensor, probe: t.Tensor, labels: t.Tensor) -> float:
    logits = einops.einsum(acts, probe, "n d, d r c cls -> n r c cls")
    return (logits.argmax(-1) == labels).float().mean().item()


def majority_accuracy(labels: t.Tensor) -> float:
    """Empirical null: each square's most frequent training class, scored on the eval rows."""
    counts = t.stack([(labels[:N_TRAIN] == cls).sum(0) for cls in range(3)])
    return (labels[N_TRAIN:] == counts.argmax(0)).float().mean().item()


def sweep_spaces(spaces: dict[str, t.Tensor], labels: t.Tensor, device: str) -> pd.DataFrame:
    """Quick-probe eval accuracy per space per init seed, plus the majority-class null."""
    train, evaluate = labels[:N_TRAIN].to(device), labels[N_TRAIN:].to(device)
    rows = [{"space": "majority", "seed": None, "eval_acc": majority_accuracy(labels)}]
    for space, acts in spaces.items():
        acts_train, acts_eval = acts[:N_TRAIN].to(device), acts[N_TRAIN:].to(device)
        for seed in SWEEP_SEEDS:
            probe = fit_probe(acts_train, train, SWEEP_EPOCHS, seed)
            rows.append({"space": space, "seed": seed,
                         "eval_acc": accuracy(acts_eval, probe, evaluate)})
            print(f"  {space} seed {seed}: eval_acc={rows[-1]['eval_acc']:.4f}")
        del acts_train, acts_eval
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = select_device(args.device)
    print(f"device: {device}")
    activations_dir = linear_probe_activations()
    activations_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    games_id = t.from_numpy(np.load(ARENA_GAMES)).long()[:NUM_GAMES]
    games_square = games_id.clone().apply_(to_string)

    states, to_move = replay(games_square)
    check_mover(states, to_move, games_square)
    labels = relative_labels(states, to_move)
    parity_labels = relative_labels(states, index_parity(states.shape[1]).expand_as(to_move))
    n_turns = TURN_END - TURN_START
    audit = audit_parity(to_move, labels, parity_labels)
    print(audit.report(to_move.numel(), NUM_GAMES * n_turns))

    model = HookedTransformer.from_pretrained(
        MODEL_NAME, revision=HF_REVISIONS[MODEL_NAME], device=device)
    print(f"{MODEL_NAME}: n_layers={model.cfg.n_layers}, d_model={model.cfg.d_model}, "
          f"normalization={model.cfg.normalization_type}")
    assert model.cfg.d_model == D_MODEL, "spade.common.config disagrees with the checkpoint"

    layers, embed = residual_activations(model, games_id, device)
    board_states = labels[:, TURN_START:TURN_END].reshape(-1, 8, 8)
    move_indices = t.arange(TURN_START, TURN_END).repeat(NUM_GAMES)
    # Row = game * n_turns + (turn - TURN_START), shared by every artifact here and paired by
    # row index with no key downstream (bracket_flip), so the layout is a contract:
    # check it through the documented formula rather than through the reshape that built it.
    assert board_states.shape == (NUM_GAMES * n_turns, 8, 8)
    assert layers.shape == (model.cfg.n_layers, NUM_GAMES * n_turns, D_MODEL)
    row_game = t.arange(NUM_GAMES).repeat_interleave(n_turns)
    assert (board_states == labels[row_game, move_indices]).all(), \
        "row is not game * n_turns + (turn - TURN_START)"
    print(f"N={layers.shape[1]} positions, {len(layers)} layers")

    t.save(layers, activations_dir / "all_layer_activations.pt")
    t.save(embed, activations_dir / "embedding_activations.pt")
    t.save(board_states, activations_dir / "board_states.pt")
    t.save(move_indices, activations_dir / "move_indices.pt")
    (activations_dir / "metadata.json").write_text(json.dumps({
        "n_games": NUM_GAMES,
        "turn_range": [TURN_START, TURN_END],
        "n_activations": layers.shape[1],
        "d_model": D_MODEL,
        "n_layers": model.cfg.n_layers,
        "model_source": MODEL_NAME,
        "normalization_type": model.cfg.normalization_type,
        "label_encoding": {"0": "empty", "1": "theirs", "2": "mine"},
        "row_order": "game * n_turns + (turn - turn_range[0])",
        "mover_source": "spade.othello.boards.next_mover (engine replay, skip-aware)",
        "parity_desync_slots": audit.slots,
        "parity_mislabeled_rows": audit.rows,
        "parity_desync_games": audit.games,
    }, indent=2))

    print("=== sweep ===")
    spaces = {EMBED: embed, **{str(layer): layers[layer] for layer in range(len(layers))}}
    sweep = sweep_spaces(spaces, board_states, device)
    sweep.to_csv(RESULTS_DIR / "layer_sweep.csv", index=False)
    per_space = sweep.groupby("space").eval_acc.agg(["mean", "std"])
    by_layer = per_space.loc[[str(layer) for layer in range(len(layers))]]
    primary_layer = int(by_layer["mean"].idxmax())
    print(per_space.to_string())
    print(f"best layer: {primary_layer}; Karvonen trains SAEs on layer {LAYER} "
          f"({by_layer['mean'].loc[str(LAYER)]:.4f})")

    print(f"=== full probe, layer {primary_layer} ===")
    acts_train = layers[primary_layer][:N_TRAIN].to(device)
    acts_eval = layers[primary_layer][N_TRAIN:].to(device)
    train, evaluate = board_states[:N_TRAIN].to(device), board_states[N_TRAIN:].to(device)
    probe = fit_probe(acts_train, train, PROBE_EPOCHS, PROBE_SEED)
    eval_acc = accuracy(acts_eval, probe, evaluate)
    print(f"train_acc={accuracy(acts_train, probe, train):.4f} eval_acc={eval_acc:.4f}")

    t.save(probe.cpu(), activations_dir / "linear_probe.pt")
    t.save(layers[primary_layer].clone(), activations_dir / "primary_layer_activations.pt")
    probe_metadata = json.dumps({
        "primary_layer": primary_layer,
        "layer_accuracies": by_layer["mean"].tolist(),
        "layer_accuracies_sd": by_layer["std"].tolist(),
        "embed_accuracy": per_space["mean"].loc[EMBED],
        "majority_accuracy": per_space["mean"].loc["majority"],
        "sweep_seeds": list(SWEEP_SEEDS),
        "sweep_epochs": SWEEP_EPOCHS,
        "probe_seed": PROBE_SEED,
        "probe_eval_acc": eval_acc,
        "n_train": N_TRAIN,
        "n_eval": layers.shape[1] - N_TRAIN,
        "n_epochs": PROBE_EPOCHS,
        "probe_shape": list(probe.shape),
    }, indent=2)
    # Same text twice: downstream reads it beside the weights, which are gitignored, and the
    # figure and the README need it tracked.
    (activations_dir / "probe_metadata.json").write_text(probe_metadata)
    (RESULTS_DIR / "probe_metadata.json").write_text(probe_metadata)
    print(f"probe saved to {activations_dir / 'linear_probe.pt'}")


if __name__ == "__main__":
    main()
