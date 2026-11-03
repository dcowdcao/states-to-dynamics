"""Build the anonymous public repo from one ref of this (private) repo.

    python scripts/release/export_public.py REF OUT_DIR

Takes the tracked tree at REF (git archive, never the working tree), drops the paths
in export_exclude.txt and scans the rest with identity_scan.py. Any hit, or a dropped
licence notice that still covers exported files, prints as TSV on stdout and exits 1,
leaving OUT_DIR without .git or manifest. A clean tree gets MANIFEST.sha256 and one
commit by an anonymous author at a fixed date, so the same tree gives the same commit.
Run it from inside the repo; OUT_DIR must be empty or absent.
"""

import argparse, fnmatch, hashlib, os, subprocess, sys, tarfile
from pathlib import Path, PurePosixPath
from typing import NoReturn

import identity_scan

EXCLUDE_FILE = Path(__file__).with_name("export_exclude.txt")
MANIFEST = "MANIFEST.sha256"
LICENCE_PREFIXES = ("LICENSE", "LICENCE", "COPYING")
ANONYMOUS = {
    "GIT_AUTHOR_NAME": "Anonymous",
    "GIT_AUTHOR_EMAIL": "anonymous@example.com",
    "GIT_AUTHOR_DATE": "2026-01-01T00:00:00+0000",
    "GIT_COMMITTER_NAME": "Anonymous",
    "GIT_COMMITTER_EMAIL": "anonymous@example.com",
    "GIT_COMMITTER_DATE": "2026-01-01T00:00:00+0000",
    # No user or system git config reaches the export: no hooks, signing or templates.
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
}
COMMIT_MESSAGE = "Initial commit"


class ExportError(Exception):
    pass


def die(msg: str) -> NoReturn:
    raise ExportError(msg)


def load_excludes(path: Path = EXCLUDE_FILE) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines()
        if line.strip()
        and not line.startswith("#")]


def is_excluded(path: str, pattern: str) -> bool:
    return path.startswith(pattern) if pattern.endswith("/") else fnmatch.fnmatchcase(path, pattern)


def resolve(ref: str) -> str:
    out = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
        capture_output=True, text=True)
    if out.returncode:
        die(f"{ref}: not a commit in this repo")
    return out.stdout.strip()


def extract(commit: str, out: Path, excludes: list[str]) -> tuple[list[str], list[str]]:
    """Write the kept files of `commit` under `out`; return (kept, dropped) repo paths."""
    archive = subprocess.Popen(["git", "archive", "--format=tar", commit], stdout=subprocess.PIPE)
    kept, dropped = [], []
    with tarfile.open(fileobj=archive.stdout, mode="r|") as tar:
        for m in tar:
            if m.isdir():
                continue
            if not m.isfile():
                die(f"{m.name}: not a regular file (symlink or submodule); decide how to export it")
            if any(is_excluded(m.name, p) for p in excludes):
                dropped.append(m.name)
                continue
            kept.append(m.name)
            dest = out / m.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(tar.extractfile(m).read())
            dest.chmod(0o755 if m.mode & 0o111 else 0o644)
    if archive.wait():
        die(f"git archive {commit} failed")
    return kept, dropped


def orphaned_notices(kept: list[str], dropped: list[str]) -> list[str]:
    """Dropped licence files whose directory still has exported files."""
    notices = [d for d in dropped if PurePosixPath(d).name.upper().startswith(LICENCE_PREFIXES)]
    return [n for n in notices
        if any(k.startswith(n[:n.rfind("/") + 1]) for k in kept)]


def write_manifest(out: Path, kept: list[str]) -> None:
    lines = [f"{hashlib.sha256((out / p).read_bytes()).hexdigest()}  {p}\n" for p in sorted(kept)]
    (out / MANIFEST).write_text("".join(lines))


def commit_anonymously(out: Path) -> str:
    env = {**os.environ, **ANONYMOUS}
    def git(*args):
        return subprocess.run(["git", "-C", str(out), *args], env=env, check=True,
            capture_output=True, text=True).stdout.strip()
    git("init", "-q", "-b", "main")
    git("add", "--all", "--force")  # the exported .gitignore must not drop tracked files
    git("commit", "-q", "-m", COMMIT_MESSAGE)
    return git("rev-parse", "HEAD")


def export(ref: str, out: Path) -> None:
    if out.exists() and any(out.iterdir()):
        die(f"{out}: not empty")
    commit = resolve(ref)
    excludes = load_excludes()
    out.mkdir(parents=True, exist_ok=True)
    kept, dropped = extract(commit, out, excludes)
    unused = [p for p in excludes if not any(is_excluded(d, p) for d in dropped)]
    if unused:
        print(f"note: exclude patterns that matched nothing at {ref}: {', '.join(unused)}", file=sys.stderr)

    hits = identity_scan.scan(out, kept, identity_scan.load_patterns())
    orphans = orphaned_notices(kept, dropped)
    size = sum((out / p).stat().st_size for p in kept)
    print(f"{ref} ({commit[:12]}): kept {len(kept)} files ({size / 1e6:.1f} MB), dropped {len(dropped)}",
        file=sys.stderr)
    if hits:
        identity_scan.print_hits(hits)
    if hits or orphans:
        die(f"{len(hits)} identity hits in {len({h.path for h in hits})} files"
            + (f"; dropped licence notices that still cover exported files: {', '.join(orphans)}" if orphans else "")
            + f". {out} is left unfinished (no .git, no {MANIFEST}).")

    write_manifest(out, kept)
    print(f"committed {commit_anonymously(out)} in {out}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref", help="git ref to export, e.g. new-architecture")
    ap.add_argument("out_dir", type=Path, help="empty or absent directory for the public repo")
    args = ap.parse_args()
    try:
        export(args.ref, args.out_dir.resolve())
    except (ExportError, identity_scan.ScanError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
