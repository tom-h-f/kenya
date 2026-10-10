#!/usr/bin/env python3
"""Fail on forbidden wording in user-facing strings.

Usage: lint_language.py [--allowlist FILE] [--include-user-content] PATH...

PATH is a file or directory. `.json` files are walked string by string. Every
other text file (tsx, ts, md, ...) is scanned line by line. In bundle JSON the
fields that carry other people's words (post text, bios, handles, tags, urls)
are skipped unless --include-user-content is set, because the exporter does not
author them. Fixtures are fully synthetic, so the tests lint them with the flag.

Allowlist (JSON list): {"file": glob, "phrase": exact text, "reason": why}.
A hit is suppressed only when it sits inside an occurrence of `phrase` in a
file whose path matches `file`. There is no way to allow a bare word.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
from pathlib import Path

FORBIDDEN = [
    ("bot", r"\bbots?\b|\bbotnets?\b"),
    ("troll", r"\btroll(?:s|ing|ed)?\b|\btroll[- ]farms?\b"),
    ("inauthentic", r"\binauthentic\w*"),
    ("fake", r"\bfake\b|\bfakes\b|\bnetwork of fake\b"),
    ("sockpuppet", r"\bsock[- ]?pupp\w+"),
    ("astroturf", r"\bastroturf\w*"),
    ("coordinated", r"\bco-?ordinated\b"),
    ("paid", r"\bpaid (?:accounts?|posters?|actors?|operatives?)\b"),
    ("operative", r"\boperatives?\b"),
    ("spam", r"\bspam(?:mers?|bots?)?\b"),
    ("disinformation actor", r"\bdisinformation (?:network|campaign|account|actor)s?\b"),
]
FORBIDDEN_RX = [(name, re.compile(rx, re.I)) for name, rx in FORBIDDEN]

USER_CONTENT_KEYS = {
    "text", "bio", "display_name", "handle", "author_handle", "tag", "tags", "url",
    "urls", "domain", "terms",
}
TEXT_SUFFIXES = {".ts", ".tsx", ".js", ".jsx", ".mjs", ".md", ".mdx", ".html", ".txt", ".css"}
SKIP_DIRS = {"node_modules", ".next", ".git", "dist", ".open-next"}


def load_allowlist(path: Path | None) -> list[dict]:
    if path is None:
        return []
    items = json.loads(path.read_text())
    for it in items:
        if not {"file", "phrase", "reason"} <= set(it) or not it["reason"].strip():
            raise SystemExit(f"allowlist entry needs file, phrase and a reason: {it}")
    return items


def allowed_spans(text: str, rel: str, allowlist: list[dict]):
    spans = []
    for it in allowlist:
        if fnmatch.fnmatch(rel, it["file"]):
            for m in re.finditer(re.escape(it["phrase"]), text, re.I):
                spans.append((m.start(), m.end()))
    return spans


def scan_text(text: str, rel: str, where: str, allowlist, out: list[str]):
    spans = allowed_spans(text, rel, allowlist)
    for name, rx in FORBIDDEN_RX:
        for m in rx.finditer(text):
            if any(a <= m.start() and m.end() <= b for a, b in spans):
                continue
            snippet = text[max(0, m.start() - 30): m.end() + 30].replace("\n", " ")
            out.append(f"{rel}: {where}: forbidden wording {name!r}: ...{snippet}...")


def walk_json(node, rel, path, allowlist, include_user, out):
    if isinstance(node, str):
        scan_text(node, rel, path, allowlist, out)
    elif isinstance(node, dict):
        for k, v in node.items():
            if not include_user and k in USER_CONTENT_KEYS:
                continue
            scan_text(k.replace("_", " "), rel, f"{path}/{k} (key)", allowlist, out)
            walk_json(v, rel, f"{path}/{k}", allowlist, include_user, out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk_json(v, rel, f"{path}/{i}", allowlist, include_user, out)


def lint_file(path: Path, rel: str, allowlist, include_user) -> list[str]:
    out: list[str] = []
    if path.suffix == ".json":
        walk_json(json.loads(path.read_text()), rel, "", allowlist, include_user, out)
    elif path.suffix in TEXT_SUFFIXES:
        text = path.read_text()
        for i, line in enumerate(text.splitlines(), 1):
            scan_text(line, rel, f"line {i}", allowlist, out)
    return out


def iter_files(paths):
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and not (set(f.relative_to(p).parts) & SKIP_DIRS):
                    yield f.relative_to(p).as_posix(), f
        elif p.is_file():
            yield p.as_posix(), p


def lint_paths(paths, allowlist=None, include_user_content=False) -> list[str]:
    allowlist = allowlist or []
    problems: list[str] = []
    for rel, f in iter_files(paths):
        problems += lint_file(f, rel, allowlist, include_user_content)
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--allowlist", type=Path)
    ap.add_argument("--include-user-content", action="store_true")
    args = ap.parse_args(argv)
    problems = lint_paths(args.paths, load_allowlist(args.allowlist), args.include_user_content)
    for p in problems:
        print(p)
    print(f"{len(problems)} problem(s)" if problems else "ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
