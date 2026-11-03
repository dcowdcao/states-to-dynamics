"""Scan a file tree for strings that identify the authors.

    python scripts/release/identity_scan.py DIR

Prints one TSV row per hit (path, where, category, match, context) and exits 1 if
there is any. Text files are scanned line by line; PNG text chunks, PDF metadata and
inflated PDF streams, and images embedded in notebooks are scanned too. A PDF that is
not a matplotlib render, and any other binary, is itself a hit: nothing ships unread.
Patterns come from identity_patterns.txt beside this file.
"""

import argparse, base64, bisect, json, re, struct, sys, zlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

PATTERNS_FILE = Path(__file__).with_name("identity_patterns.txt")
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
PDF_MAGIC = b"%PDF-"
PDF_STREAM = re.compile(rb"(?<!end)stream\r?\n(.*?)\r?\nendstream", re.S)
PDF_PRODUCER = re.compile(r"/Producer \(([^)]*)\)")
PIPELINE_PDF_PRODUCER = "Matplotlib pdf backend"
PRINTABLE_RUN = re.compile(rb"[\x20-\x7e]{8,}")
CONTEXT_CHARS = 60
TSV_HEADER = "path\twhere\tcategory\tmatch\tcontext"


@dataclass(frozen=True)
class Pattern:
    category: str
    regex: re.Pattern


@dataclass(frozen=True)
class Hit:
    path: str
    where: str
    category: str
    match: str
    context: str

    def tsv(self) -> str:
        return "\t".join(" ".join(f.split()) for f in (self.path, self.where, self.category, self.match, self.context))


class ScanError(Exception):
    pass


def load_patterns(path: Path = PATTERNS_FILE) -> list[Pattern]:
    patterns = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        category, sep, regex = line.partition("\t")
        if not sep or not regex:
            raise ScanError(f"{path}:{n}: expected `category<TAB>regex`")
        patterns.append(Pattern(category, re.compile(regex)))
    return patterns


def scan_text(path: str, text: str, patterns: list[Pattern], where: str = "") -> Iterator[Hit]:
    """`where` prefixes the 1-based line number of each hit."""
    starts = [0] + [m.end() for m in re.finditer("\n", text)]
    for p in patterns:
        for m in p.regex.finditer(text):
            i = bisect.bisect_right(starts, m.start()) - 1
            end = text.find("\n", m.end())
            line = text[starts[i]:len(text) if end < 0 else end]
            col = m.start() - starts[i]
            context = line[max(0, col - CONTEXT_CHARS):col + len(m.group()) + CONTEXT_CHARS]
            yield Hit(path, f"{where}{i + 1}", p.category, m.group(), context)


def _printable(data: bytes) -> str:
    return "\n".join(m.group().decode("ascii") for m in PRINTABLE_RUN.finditer(data))


def _png_text_chunks(data: bytes) -> Iterator[tuple[str, str]]:
    pos = len(PNG_MAGIC)
    while pos < len(data):
        length, kind = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        key, _, rest = body.partition(b"\0")
        match kind:
            case b"tEXt":
                yield key.decode("latin-1"), rest.decode("latin-1")
            case b"zTXt":
                yield key.decode("latin-1"), zlib.decompress(rest[1:]).decode("latin-1")
            case b"iTXt":
                _lang, _, rest2 = rest[2:].partition(b"\0")
                _translated, _, text = rest2.partition(b"\0")
                yield key.decode("latin-1"), (zlib.decompress(text) if rest[0] else text).decode("utf-8")
            case b"eXIf":
                yield "eXIf", _printable(body)


def _pdf_texts(data: bytes) -> Iterator[tuple[str, str]]:
    """The object structure without stream bodies, then every inflated non-image stream."""
    yield "pdf:", _printable(PDF_STREAM.sub(b"stream endstream", data))
    for k, m in enumerate(PDF_STREAM.finditer(data)):
        if b"/Image" in data[data.rfind(b"obj", 0, m.start()):m.start()]:
            continue  # pixels, whose printable runs are noise
        try:
            inflated = zlib.decompress(m.group(1))
        except zlib.error:  # not FlateDecode, e.g. a DCT image
            continue
        yield f"pdf stream {k}:", _printable(inflated)


def _scan_png(path: str, data: bytes, patterns: list[Pattern], where: str = "") -> Iterator[Hit]:
    for key, text in _png_text_chunks(data):
        yield from scan_text(path, f"{key}: {text}", patterns, f"{where}png {key}:")


def _scan_pdf(path: str, data: bytes, patterns: list[Pattern]) -> Iterator[Hit]:
    texts = list(_pdf_texts(data))
    producers = [m.group(1) for _, t in texts for m in PDF_PRODUCER.finditer(t)]
    if not any(p.startswith(PIPELINE_PDF_PRODUCER) for p in producers):
        yield Hit(path, "pdf /Producer", "non_pipeline_pdf", ", ".join(producers) or "none", "")
    for where, text in texts:
        yield from scan_text(path, text, patterns, where)


def _scan_notebook_images(path: str, text: str, patterns: list[Pattern]) -> Iterator[Hit]:
    for i, cell in enumerate(json.loads(text)["cells"]):
        for out in cell.get("outputs", []):  # markdown cells and stream outputs carry no image
            png = out.get("data", {}).get("image/png")
            if png:
                yield from _scan_png(path, base64.b64decode(png), patterns, f"cell {i} ")


def scan_file(root: Path, path: str, patterns: list[Pattern]) -> Iterator[Hit]:
    yield from scan_text(path, path, patterns, "filename:")
    data = (root / path).read_bytes()
    if data.startswith(PNG_MAGIC):
        yield from _scan_png(path, data, patterns)
        return
    if data.startswith(PDF_MAGIC):
        yield from _scan_pdf(path, data, patterns)
        return
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = None
    if text is None or "\0" in text:
        yield Hit(path, "file", "unscanned_binary", f"{len(data)} bytes, starts {data[:8]!r}", "")
        return
    yield from scan_text(path, text, patterns)
    if path.endswith(".ipynb"):
        yield from _scan_notebook_images(path, text, patterns)


def scan(root: Path, paths: list[str], patterns: list[Pattern]) -> list[Hit]:
    return [h for p in sorted(paths) for h in scan_file(root, p, patterns)]


def print_hits(hits: list[Hit]) -> None:
    print(TSV_HEADER)
    for h in hits:
        print(h.tsv())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="directory to scan (its .git is skipped)")
    root = ap.parse_args().root
    paths = [str(p.relative_to(root)) for p in root.rglob("*")
        if p.is_file()
        and ".git" not in p.relative_to(root).parts]
    try:
        hits = scan(root, paths, load_patterns())
    except ScanError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print_hits(hits)
    print(f"{len(hits)} hits in {len({h.path for h in hits})} of {len(paths)} files", file=sys.stderr)
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
