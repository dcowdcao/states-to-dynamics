"""Put the paper's data release under $SPADE_DATA_ROOT and check every file's sha256.

    SPADE_RELEASE_DATASET=<hf user>/<dataset> python scripts/fetch_release_data/fetch.py
    SPADE_RELEASE_DATA_DIR=<a staged copy>   python scripts/fetch_release_data/fetch.py

The files are manifests/data_release.tsv's rows, at the same relative paths under the data
root. They come from the Hugging Face dataset SPADE_RELEASE_DATASET or, when it is set, from
the local directory SPADE_RELEASE_DATA_DIR (scripts/stage_release_data/ builds one). A file
already in place with the right sha256 is kept; one with another sha256 stops the run.
"""

import csv, hashlib, os, shutil, sys
from pathlib import Path
from typing import NoReturn

from huggingface_hub import snapshot_download

from spade.common.paths import repo_root
from spade.common.study import data_root

MANIFEST = repo_root() / "manifests" / "data_release.tsv"
# The anonymous dataset this release is published as; empty until it exists.
SPADE_RELEASE_DATASET = os.environ.get("SPADE_RELEASE_DATASET", "")


class FetchError(Exception):
    pass


def die(msg: str) -> NoReturn:
    raise FetchError(msg)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source() -> Path:
    if "SPADE_RELEASE_DATA_DIR" in os.environ:
        return Path(os.environ["SPADE_RELEASE_DATA_DIR"])
    if not SPADE_RELEASE_DATASET:
        die("no source for the data release: set SPADE_RELEASE_DATASET to its Hugging Face "
            "dataset id, or SPADE_RELEASE_DATA_DIR to a local copy")
    return Path(snapshot_download(SPADE_RELEASE_DATASET, repo_type="dataset"))


def fetch(src: Path, rows: list[dict]) -> None:
    dest_root = data_root()
    for row in rows:
        dest = dest_root / row["path"]
        if dest.exists():
            if sha256(dest) != row["sha256"]:
                die(f"{dest} exists with another sha256; move it aside and re-run")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        partial = dest.with_name(dest.name + ".partial")
        shutil.copyfile(src / row["path"], partial)
        if sha256(partial) != row["sha256"]:
            die(f"{src / row['path']}: sha256 differs from {MANIFEST.name}")
        partial.rename(dest)


def main() -> int:
    with open(MANIFEST) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    try:
        src = source()
        missing = [r["path"] for r in rows if not (src / r["path"]).is_file()]
        if missing:
            die(f"{len(missing)} release files not in {src}, e.g. {missing[0]}")
        fetch(src, rows)
    except FetchError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    total = sum(int(r["bytes"]) for r in rows)
    print(f"{len(rows)} files, {total / 1e6:.1f} MB, sha256-checked in {data_root()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
