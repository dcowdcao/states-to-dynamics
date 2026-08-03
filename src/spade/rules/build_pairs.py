"""Step 1 of the rules study: build matched legal/illegal move pairs.

For each sampled legal game prefix m1..m(t-1), emit sequences prefix+move for:
2 distinct legal moves (the second is the legal-legal control pair, the
token-swap null) and 1 illegal move per available taxonomy category. All
sequences sharing a prefix differ only in the final token, so activation
differences at position t isolate the injected move.

t is assigned round-robin over [5, 57] to accepted prefixes (uniform t by
construction, shared across conditions); skipped games retry the same t.

Usage: python -m spade.rules.build_pairs [--n_prefixes 4000] [--seed 42]
"""

import argparse
import json
from collections import Counter
from random import Random

import torch

from spade.common.net import force_ipv4
from spade.common.paths import add_karvonen_to_path

force_ipv4()
add_karvonen_to_path()

from spade.common.study import study_dir
from spade.othello.games import T_MAX, T_MIN, classify_prefix, stream_games
from spade.othello.taxonomy import CATEGORIES, LEGAL

from circuits.othello_engine_utils import stoi, stoi_indices

MAX_LEN = T_MAX + 1

# classify_all() index i must correspond to vocab token i+1
assert all(stoi[sq] == i + 1 for i, sq in enumerate(stoi_indices))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_prefixes", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_name", default="adamkarvonen/othello_45MB_games")
    args = parser.parse_args()

    import datasets as hf_datasets
    rng = Random(args.seed)
    games = stream_games(args.dataset_name, args.seed)
    skips = Counter()

    rows = {k: [] for k in ["tokens", "length", "t", "category", "prefix_id",
                            "injected_token", "mover_parity", "n_legal",
                            "is_control_legal"]}
    square_cats_rows = []
    t_hists = {name: Counter() for name in CATEGORIES}

    j = 0
    while j < args.n_prefixes:
        game = next(games)["tokens"]
        t = T_MIN + (j % (T_MAX - T_MIN + 1))
        result = classify_prefix(game, t, skips)
        if result is None:
            continue
        board, cats = result

        by_cat = {c: [i + 1 for i, ci in enumerate(cats) if ci == c]
            for c in range(len(CATEGORIES))}  # token = vocab index + 1
        legal_tokens = by_cat[LEGAL]
        if len(legal_tokens) < 2:
            skips["single_legal"] += 1
            continue

        parity = board.get_next_hand_color()  # engine-derived: forfeits flip it

        picks = rng.sample(legal_tokens, 2)
        injected = [(picks[0], LEGAL, 0), (picks[1], LEGAL, 1)]
        injected += [(rng.choice(by_cat[c]), c, 0)
            for c in range(1, len(CATEGORIES)) if by_cat[c]]

        for token, cat, is_control in injected:
            seq = game[:t] + [token]
            assert all(v > 0 for v in seq)
            rows["tokens"].append(seq + [0] * (MAX_LEN - len(seq)))
            rows["length"].append(t + 1)
            rows["t"].append(t)
            rows["category"].append(cat)
            rows["prefix_id"].append(j)
            rows["injected_token"].append(token)
            rows["mover_parity"].append(parity)
            rows["n_legal"].append(len(legal_tokens))
            rows["is_control_legal"].append(is_control)
            if not is_control:
                t_hists[CATEGORIES[cat]][t] += 1
        square_cats_rows.append(cats)
        j += 1
        if j % 500 == 0:
            print(f"{j}/{args.n_prefixes} prefixes, {len(rows['t'])} items")

    items = {k: torch.tensor(v, dtype=torch.int16 if k == "tokens" else torch.long)
        for k, v in rows.items()}
    items["square_categories"] = torch.tensor(square_cats_rows, dtype=torch.int8)
    for i in range(len(items["t"])):
        L = items["length"][i]
        assert (items["tokens"][i, :L] != 0).all() and (items["tokens"][i, L:] == 0).all()

    out_dir = study_dir("rules") / "pairs"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(items, out_dir / "items.pt")

    counts = Counter(CATEGORIES[c] for c, ctrl
        in zip(rows["category"], rows["is_control_legal"]) if not ctrl)
    manifest = {
        "seed": args.seed,
        "dataset": args.dataset_name,
        "datasets_version": hf_datasets.__version__,
        "n_prefixes": args.n_prefixes,
        "n_items": len(rows["t"]),
        "category_counts": dict(counts),
        "category_t_histograms": {k: dict(v) for k, v in t_hists.items()},
        "skips": dict(skips),
        "categories": CATEGORIES,
        "note": "t assigned to accepted prefixes; skips retry the same t, so "
                "late-t prefixes are conditioned on pass-free non-forfeit games",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(rows['t'])} items to {out_dir}")
    print(json.dumps({"category_counts": dict(counts), "skips": dict(skips)}, indent=2))


if __name__ == "__main__":
    main()
