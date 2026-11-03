"""Copy the data release out of $SPADE_DATA_ROOT into OUT_DIR, laid out as the data root is.

    python scripts/release/stage_data.py OUT_DIR

The released paths are the `path` column of manifests/data_release.tsv. Each is copied with
symlinks dereferenced, and two kinds of file are changed on the way:
- every file that is not a tensor archive (.pt, .npz) is text, and loses the data root's
  absolute prefix, so the eval tables and manifests name paths relative to any data root;
- spade_straightness/games/games.pt keeps only its leading legal block, rows 0-1999 of the HF
  corpus: the rows after it include WTHOR championship games, which the release does not re-host.
Then the staged tree is scanned with identity_scan.py's patterns (text, the pickles inside .pt,
string arrays inside .npz). Any hit prints as TSV and exits 1; a clean tree rewrites the
manifest's bytes and sha256 columns, so a changed input shows in its git diff. OUT_DIR is then
uploaded as is; scripts/fetch_release_data/ downloads it.
"""

import argparse, csv, hashlib, re, shutil, sys, zipfile
from pathlib import Path
from typing import NoReturn

import numpy as np
import torch

import identity_scan
from spade.common.paths import repo_root
from spade.common.study import data_root

MANIFEST = repo_root() / "manifests" / "data_release.tsv"
COLUMNS = ["path", "bytes", "sha256"]
TENSOR_ARCHIVES = (".pt", ".npz")
GAMES = "spade_straightness/games/games.pt"
LEGAL = 0  # spade.straightness.build_games.CLASSES.index("legal")
PICKLE_TEXT = re.compile(rb"[\x20-\x7e]{4,}")


class StageError(Exception):
    pass


def die(msg: str) -> NoReturn:
    raise StageError(msg)


def legal_block(src: Path, dest: Path) -> None:
    games = torch.load(src, weights_only=True)
    n = int((games["cls"] == LEGAL).sum())
    if not (games["cls"][:n] == LEGAL).all():
        die(f"{src}: the legal games are not its leading block")
    torch.save({k: v[:n].clone() for k, v in games.items()}, dest)


def copy(rel: str, dest: Path) -> None:
    src = data_root() / rel
    if rel == GAMES:
        legal_block(src, dest)
    elif rel.endswith(TENSOR_ARCHIVES):
        shutil.copyfile(src, dest)
    else:
        dest.write_text(src.read_text().replace(f"{data_root()}/", ""))


def texts(path: Path) -> dict[str, str]:
    """What identity_scan reads in one staged file, keyed by where it sits in the file."""
    if path.suffix == ".npz":
        with np.load(path) as z:
            return {f"{k}:": "\n".join(map(str, z[k].ravel())) for k in z.files if z[k].dtype.kind in "US"}
    if path.suffix == ".pt":
        with zipfile.ZipFile(path) as z:
            return {f"{m}:": "\n".join([m, *(r.decode() for r in PICKLE_TEXT.findall(z.read(m)))])
                    for m in z.namelist() if m.endswith(".pkl")}
    return {"": path.read_text()}


def scan(out: Path, paths: list[str]) -> list[identity_scan.Hit]:
    patterns = identity_scan.load_patterns()
    return [hit for rel in paths
        for where, text in {"filename:": rel, **texts(out / rel)}.items()
        for hit in identity_scan.scan_text(rel, text, patterns, where)]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stage(out: Path) -> list[dict]:
    if out.exists() and any(out.iterdir()):
        die(f"{out}: not empty")
    with open(MANIFEST) as f:
        paths = [row["path"] for row in csv.DictReader(f, delimiter="\t")]
    missing = [p for p in paths if not (data_root() / p).is_file()]
    if missing:
        die(f"not in {data_root()}: {', '.join(missing)}")
    for rel in paths:
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        copy(rel, out / rel)
    hits = scan(out, paths)
    if hits:
        identity_scan.print_hits(hits)
        die(f"{len(hits)} identity hits in {len({h.path for h in hits})} staged files; "
            f"{MANIFEST.name} is left as it was")
    return [{"path": p, "bytes": (out / p).stat().st_size, "sha256": sha256(out / p)} for p in paths]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path, help="empty or absent directory to stage into")
    out = ap.parse_args().out_dir.resolve()
    try:
        rows = stage(out)
    except (StageError, identity_scan.ScanError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    total = sum(r["bytes"] for r in rows)
    print(f"staged {len(rows)} files, {total / 1e6:.1f} MB, into {out}; rewrote {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
