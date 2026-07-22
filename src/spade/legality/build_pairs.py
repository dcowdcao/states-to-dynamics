"""Legality step 1 — build matched legal/illegal move pairs sharing a game prefix.

For each prefix, emit a token-swap null (legal) plus one illegal candidate per taxonomy
category, so downstream contrasts hold the prefix fixed. Writes to the netscratch study dir.

Entrypoint: `python -m spade.legality.build_pairs`
"""

# def stream_games(...): ...
# def classify_prefix(prefix): ...
# def main(): ...   # -> pairs.pt in legality_dir()
