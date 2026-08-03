"""Step 1 of the straightness study: build the four game classes.

Every game is a fixed CTX_LEN(=59)-token sequence, no pass tokens:
- legal: first 59 moves of streamed synthetic games (the model's training
  distribution).
- rollout: OthelloGPT temperature-1.0 samples (pass token masked), opened with
  a uniform-random legal first move. first_illegal records the first position
  whose move the engine rejects (-1 if the whole rollout is legal); board
  tracking stops there since the board is undefined after.
- violation: legal-class prefix to t0 (round-robin over [10, 40]), one injected
  illegal move from the empty-square taxonomy categories placed with NO flips
  and the turn passed, then uniform-random legal continuation. src_game points
  at the source legal game for matched-pair analyses.
- random_tokens: uniform tokens without repetition -- maximal OOD.
- championship: real human tournament games from the WTHOR database
  (ffothello.org), engine-validated -- legal rules, shifted move policy.

Usage: python -m spade.straightness.build_games [--n_per_class 2000] [--seed 42]
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from random import Random

import torch

from spade.common.config import CTX_LEN, MODEL_NAME
from spade.common.device import select_device
from spade.common.study import study_dir
from spade.othello.games import stream_games
from spade.othello.taxonomy import classify_all

from circuits.othello_engine_utils import OthelloBoardState, stoi, to_string

CLASSES = ["legal", "rollout", "violation", "random_tokens", "championship"]
EMPTY_CATEGORIES = [3, 4, 5, 6]  # illegal placements on empty squares (well-defined)
T0_MIN, T0_MAX = 10, 40
ROLLOUT_BATCH = 250
WTHOR_YEARS = range(2014, 2024)
WTHOR_URL = "https://www.ffothello.org/wthor/base_zip/WTH_{year}.ZIP"


def collect_legal(games, n: int, skips: Counter) -> list[list[int]]:
    out = []
    while len(out) < n:
        seq = next(games)["tokens"][:CTX_LEN]
        if len(seq) < CTX_LEN:
            skips["short_game"] += 1
        elif 0 in seq:
            skips["pass_token"] += 1
        else:
            out.append(seq)
    return out


def engine_legal(board: OthelloBoardState, square: int) -> bool:
    r, c = square // 8, square % 8
    if board.state[r, c] != 0:
        return False
    kind = board.tentative_move(square)
    if kind == 1:
        return True
    if kind == 0:
        return False
    # flips only for the opponent: engine-legal iff the mover must forfeit
    return all(board.tentative_move(m) != 1 for m in range(64))


def first_illegal_pos(seq: list[int]) -> int:
    board = OthelloBoardState()
    for pos, tok in enumerate(seq):
        sq = to_string(tok)
        if not engine_legal(board, sq):
            return pos
        board.umpire(sq)
    return -1


def sample_rollouts(model, device: str, n: int, seed: int) -> tuple[list[list[int]], list[int]]:
    g = torch.Generator().manual_seed(seed)
    board = OthelloBoardState()
    openings = torch.tensor([stoi[sq] for sq in board.get_valid_moves()])
    seqs = []
    for start in range(0, n, ROLLOUT_BATCH):
        b = min(ROLLOUT_BATCH, n - start)
        tokens = torch.zeros(b, CTX_LEN, dtype=torch.long)
        tokens[:, 0] = openings[torch.randint(len(openings), (b,), generator=g)]
        for pos in range(1, CTX_LEN):
            logits = model(tokens[:, :pos].to(device))[:, -1, :].cpu()
            logits[:, 0] = -torch.inf
            tokens[:, pos] = torch.multinomial(logits.softmax(-1), 1, generator=g).squeeze(1)
        seqs.extend(tokens.tolist())
        print(f"rollouts sampled {start + b}/{n}")
    return seqs, [first_illegal_pos(s) for s in seqs]


def collect_championship(n: int, cache_dir: Path, rng: Random, skips: Counter) -> list[list[int]]:
    """WTB records: 16-byte file header, then 68-byte games with 60 move bytes
    at offset 8; move byte b = 10*(column+1) + (row+1), zero-padded when the
    game ends early."""
    import requests
    from io import BytesIO
    from zipfile import ZipFile
    seqs, seen = [], set()
    for year in WTHOR_YEARS:
        path = cache_dir / f"WTH_{year}.wtb"
        if not path.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
            r = requests.get(WTHOR_URL.format(year=year), timeout=60)
            r.raise_for_status()
            path.write_bytes(ZipFile(BytesIO(r.content)).read(f"WTH_{year}.wtb"))
        data = path.read_bytes()
        for i in range(int.from_bytes(data[4:8], "little")):
            moves = data[16 + 68 * i + 8 : 16 + 68 * i + 68]
            squares = [(b // 10 - 1) * 8 + (b % 10 - 1) for b in moves if b != 0]
            if len(squares) < CTX_LEN:
                skips["championship_short"] += 1
                continue
            seq = [stoi[sq] for sq in squares[:CTX_LEN]]
            if tuple(seq) in seen:
                skips["championship_dup"] += 1
                continue
            if first_illegal_pos(seq) != -1:
                skips["championship_invalid"] += 1
                continue
            seen.add(tuple(seq))
            seqs.append(seq)
        print(f"wthor {year}: {len(seqs)} valid games so far")
    assert len(seqs) >= n, f"only {len(seqs)}/{n} championship games collected"
    rng.shuffle(seqs)
    return seqs[:n]


def place_no_flip(board: OthelloBoardState, square: int) -> None:
    r, c = square // 8, square % 8
    assert board.state[r, c] == 0
    board.state[r, c] = board.next_hand_color
    board.next_hand_color *= -1
    board.history.append(square)


def try_violation(prefix: list[int], t0: int, rng: Random):
    """Returns (tokens, category) or None (forfeit at t0 / dead continuation)."""
    board = OthelloBoardState()
    board.update(to_string(prefix))
    if not any(board.tentative_move(m) == 1 for m in board.get_valid_moves()):
        return None
    cats = classify_all(board)
    by_cat = {c: [i + 1 for i, ci in enumerate(cats) if ci == c] for c in EMPTY_CATEGORIES}
    available = [c for c in EMPTY_CATEGORIES if by_cat[c]]
    if not available:
        return None
    category = rng.choice(available)
    injected = rng.choice(by_cat[category])
    place_no_flip(board, to_string(injected))
    seq = prefix + [injected]
    while len(seq) < CTX_LEN:
        valids = board.get_valid_moves()
        if not valids:
            return None
        sq = rng.choice(valids)
        board.umpire(sq)
        seq.append(stoi[sq])
    return seq, category


def build_violations(legal: list[list[int]], n: int, rng: Random, skips: Counter):
    rows = []  # (tokens, t0, category, src_game)
    for j, src in enumerate(legal):
        if len(rows) >= n:
            break
        t0s = [T0_MIN + (j % (T0_MAX - T0_MIN + 1))]
        t0s += rng.sample([t for t in range(T0_MIN, T0_MAX + 1) if t != t0s[0]],
                          T0_MAX - T0_MIN)
        for t0 in t0s:
            result = try_violation(src[:t0], t0, rng)
            if result is not None:
                rows.append((result[0], t0, result[1], j))
                break
            skips["violation_retry"] += 1
        else:
            skips["violation_source_exhausted"] += 1
    assert len(rows) == n, f"only {len(rows)}/{n} violations built"
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_class", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_name", default="adamkarvonen/othello_45MB_games")
    args = parser.parse_args()

    from transformer_lens import HookedTransformer

    n, rng, skips = args.n_per_class, Random(args.seed), Counter()
    device = select_device()
    torch.set_grad_enabled(False)

    legal = collect_legal(stream_games(args.dataset_name, args.seed), n, skips)
    for seq in legal[:50]:
        assert first_illegal_pos(seq) == -1, "legal-class game fails engine replay"

    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device).eval()
    rollouts, first_illegal = sample_rollouts(model, device, n, args.seed)
    violations = build_violations(legal, n, rng, skips)
    randoms = [rng.sample(range(1, 61), CTX_LEN) for _ in range(n)]
    championship = collect_championship(
        n, study_dir("straightness") / "wthor", rng, skips)

    tokens = legal + rollouts + [v[0] for v in violations] + randoms + championship
    pad = [-1] * n
    data = {
        "tokens": torch.tensor(tokens, dtype=torch.int16),
        "cls": torch.tensor(sum(([c] * n for c in range(len(CLASSES))), []), dtype=torch.int8),
        "t0": torch.tensor(pad * 2 + [v[1] for v in violations] + pad * 2),
        "category": torch.tensor(pad * 2 + [v[2] for v in violations] + pad * 2, dtype=torch.int8),
        "first_illegal": torch.tensor(pad + first_illegal + pad * 3),
        "src_game": torch.tensor(pad * 2 + [v[3] for v in violations] + pad * 2),
    }
    assert data["tokens"].shape == (len(CLASSES) * n, CTX_LEN) and (data["tokens"] > 0).all()

    out_dir = study_dir("straightness") / "games"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_dir / "games.pt")

    n_legal_rollouts = sum(f == -1 for f in first_illegal)
    manifest = {
        "seed": args.seed,
        "dataset": args.dataset_name,
        "wthor_years": [WTHOR_YEARS.start, WTHOR_YEARS.stop - 1],
        "n_per_class": n,
        "classes": CLASSES,
        "rollout_fully_legal": n_legal_rollouts,
        "violation_category_counts": dict(Counter(v[2] for v in violations)),
        "skips": dict(skips),
        "note": "violation = no-flip placement at t0 + turn passed + random legal "
                "continuation; rollout legality stops being tracked at first_illegal",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(CLASSES) * n} games to {out_dir}")
    print(f"rollouts fully legal: {n_legal_rollouts}/{n}")
    print(json.dumps({"skips": dict(skips)}, indent=2))


if __name__ == "__main__":
    main()
