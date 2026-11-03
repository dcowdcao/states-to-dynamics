"""Check figures/ and results/ against manifests/figures.tsv and manifests/results.tsv.

    python scripts/verify_outputs/verify.py            # after make figures
    python scripts/verify_outputs/verify.py --update   # after a deliberate change to an output

Every file is compared by sha256. A PDF is hashed without its /CreationDate, the one field a
re-render changes. Prints one line per mismatched, missing or unlisted file and exits 1 if any.
"""

import argparse, csv, hashlib, re, sys
from pathlib import Path

from spade.common.paths import repo_root

TREES = {"figures": "manifests/figures.tsv", "results": "manifests/results.tsv"}
CREATION_DATE = re.compile(rb"/CreationDate \([^)]*\)")


def digest(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix == ".pdf":
        data = CREATION_DATE.sub(b"", data)
    return hashlib.sha256(data).hexdigest()


def current(tree: str) -> dict[str, str]:
    root = repo_root()
    return {str(p.relative_to(root)): digest(p) for p in sorted((root / tree).rglob("*"))
        if p.is_file()}


def recorded(manifest: str) -> dict[str, str]:
    with open(repo_root() / manifest) as f:
        return {row["path"]: row["sha256"] for row in csv.DictReader(f, delimiter="\t")}


def write(manifest: str, digests: dict[str, str]) -> None:
    with open(repo_root() / manifest, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["path", "sha256"])
        w.writerows(digests.items())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--update", action="store_true", help="rewrite the manifests from the files")
    update = ap.parse_args().update
    bad = 0
    for tree, manifest in TREES.items():
        now = current(tree)
        if update:
            write(manifest, now)
            print(f"wrote {manifest}: {len(now)} files")
            continue
        want = recorded(manifest)
        for path in sorted(want.keys() | now.keys()):
            if path not in now:
                print(f"missing   {path}")
            elif path not in want:
                print(f"unlisted  {path}")
            elif now[path] != want[path]:
                print(f"differs   {path}")
            else:
                continue
            bad += 1
        print(f"{tree}/: {len(want) - sum(now.get(p) != h for p, h in want.items())} of {len(want)} match")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
