"""The game corpus as token arrays, read off its parquet at the revision every study streamed."""

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

DATASET = "adamkarvonen/othello_45MB_games"
REVISION = "caafaff9fd273cd1783e146d039e262f97c9f5f8"
PLIES = 60


def tokens(split: str) -> np.ndarray:
    """[n_games, 60] int8 vocab tokens of one split, in stored row order."""
    path = hf_hub_download(DATASET, f"data/{split}-00000-of-00001.parquet",
                           repo_type="dataset", revision=REVISION)
    column = pq.read_table(path).column("tokens").combine_chunks()
    lengths = np.diff(column.offsets.to_numpy())
    assert (lengths == PLIES).all(), f"{split}: game lengths {set(lengths.tolist())}"
    return column.values.to_numpy().astype(np.int8).reshape(-1, PLIES)
