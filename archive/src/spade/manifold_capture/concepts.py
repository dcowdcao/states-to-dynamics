"""Concept-manifold definitions over the two labeled corpora.

Corpus "games": natural positions from build_corpus/extract_acts (t, n_flipped,
n_legal, disc_diff, mover-relative board). Corpus "rules": the rules study's
injected-move activations, whose 7-way taxonomy is the rule-violation concept.
Every position carries every concept, so a concept manifold is the corpus
labeled by that concept's value; the t confound is handled by the per-t-centered
variant rather than t-band restriction (t itself is a concept, raw-only).
"""

from dataclasses import dataclass

import torch

from spade.common.study import study_dir
from spade.spaces import center_by_t as _center_by_t

SUBSAMPLE_SEED = 11
SQUARES = {"sq_D2": (2, 3), "sq_D0": (0, 3), "sq_A0": (0, 0)}  # center-adjacent/edge/corner
SQUARE_CLASSES = ("empty", "theirs", "mine")
RULE_CLASSES = ("legal", "occupied_own", "occupied_opp", "empty_detached",
                "empty_own_only", "empty_no_bracket", "opponent_only")


@dataclass(frozen=True)
class Concept:
    name: str
    kind: str      # "continuous" | "categorical"
    corpus: str    # "games" | "rules"
    class_names: tuple[str, ...] = ()

    @property
    def variants(self) -> tuple[str, ...]:
        return ("raw",) if self.name == "t" else ("raw", "centered")


def concepts() -> list[Concept]:
    cont = [Concept(n, "continuous", "games")
            for n in ["t", "n_flipped", "n_legal", "disc_diff"]]
    squares = [Concept(n, "categorical", "games", SQUARE_CLASSES) for n in SQUARES]
    return cont + squares + [Concept("rule_category", "categorical", "rules", RULE_CLASSES)]


def load_corpus(corpus: str) -> dict[str, torch.Tensor]:
    """acts [N, 512] fp32 raw activations + per-row label tensors (always has "t")."""
    if corpus == "games":
        data = torch.load(study_dir("manifold") / "corpus" / "corpus.pt", weights_only=True)
        data["acts"] = torch.load(study_dir("manifold") / "acts" / "acts.pt",
                                  weights_only=True)
        data["legal"] = torch.ones(len(data["t"]), dtype=torch.bool)
    else:
        items = torch.load(study_dir("rules") / "pairs" / "items.pt", weights_only=True)
        data = {"acts": torch.load(study_dir("rules") / "acts" / "acts_final.pt",
                                   weights_only=True),
                "t": items["t"], "rule_category": items["category"],
                "legal": items["category"] == 0}
    assert len(data["acts"]) == len(data["t"])
    return data


def values(concept: Concept, corpus_data: dict) -> torch.Tensor:
    """Per-row concept value: float for continuous, class index for categorical."""
    if concept.name in SQUARES:
        r, c = SQUARES[concept.name]
        return corpus_data["board"][:, r, c].long()
    v = corpus_data[concept.name]
    return v.long() if concept.kind == "categorical" else v.float()


def subsample(n: int, cap: int, seed: int = SUBSAMPLE_SEED) -> torch.Tensor:
    """Fixed seeded row subsample, sorted — shared across spaces and analyses."""
    if n <= cap:
        return torch.arange(n)
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(n, generator=g)[:cap].sort().values


def center_by_t(feats: torch.Tensor, t: torch.Tensor,
                legal: torch.Tensor | None = None) -> torch.Tensor:
    """Per-t-bin centering; reference rows are the legal set (rules corpus) or
    everything (games corpus, where all rows are legal)."""
    if legal is None:
        legal = torch.ones(len(t), dtype=torch.bool)
    return _center_by_t(feats, t, legal)
