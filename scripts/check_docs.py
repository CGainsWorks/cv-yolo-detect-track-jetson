#!/usr/bin/env python
"""Every number in the documentation traces to a published result.

    python scripts/check_docs.py

Two checks, and the second is the one that catches drift.

**Generated blocks match.**  ``README.md`` and ``docs/results.md`` embed tables
between ``<!-- BEGIN generated:NAME -->`` and ``<!-- END generated:NAME -->``.
This re-reads the file the generator wrote into ``docs/tables/`` and requires
the embedded copy to match it exactly.  A stale table is a failure, not a
warning.

**Free numbers are accounted for.**  Every decimal number in the prose is
looked up in the values of every JSON file under ``results/published/``.  A
number that is not there and not in ``docs/allowed_numbers.txt`` fails the
check.  The allowlist exists because a document legitimately contains numbers
that are not measurements, and every entry in it carries the reason it is
there.

Run with ``--fix-blocks`` to rewrite the generated blocks in place instead of
failing on them.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from edgetrack.paths import write_text_lf

BLOCK = re.compile(
    r"<!-- BEGIN generated:(?P<name>[\w.-]+) -->\n(?P<body>.*?)<!-- END generated:(?P=name) -->",
    re.DOTALL,
)
NUMBER = re.compile(r"(?<![\w.])(\d+\.\d+)(?![\w.])")
CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE = re.compile(r"`[^`\n]*`")
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


def collect_published(root: Path) -> set[str]:
    """Every numeric leaf in every published JSON, in several renderings.

    A document may quote 0.8321 as 0.832, 0.83 or 83.2%, and all three should
    count as the same measurement.  Rendering the source number every way a
    human would write it is cheaper and less brittle than parsing the document
    for intent.
    """
    values: set[float] = set()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for item in node.values():
                walk(item)
        elif isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, bool):
            return
        elif isinstance(node, (int, float)):
            values.add(float(node))

    for path in sorted(root.rglob("*.json")):
        walk(json.loads(path.read_text(encoding="utf-8")))

    rendered: set[str] = set()
    for value in values:
        for places in range(0, 6):
            rendered.add(f"{value:.{places}f}")
            rendered.add(f"{value * 100:.{places}f}")
            rendered.add(f"{abs(value):.{places}f}")
        rendered.add(repr(value))
        rendered.add(str(value))
    return rendered


def strip_uncheckable(text: str) -> str:
    """Remove code, inline code and comments before hunting for numbers.

    Numbers inside a command line or a file name are not claims about results,
    and requiring them to appear in a results file would make the check
    unusable rather than strict.
    """
    text = HTML_COMMENT.sub(" ", text)
    text = CODE_FENCE.sub(" ", text)
    return INLINE_CODE.sub(" ", text)


def check_blocks(path: Path, table_dir: Path, fix: bool) -> list[str]:
    problems: list[str] = []
    text = path.read_text(encoding="utf-8")
    replaced = text
    for match in BLOCK.finditer(text):
        name = match.group("name")
        source = table_dir / f"{name}.md"
        if not source.exists():
            problems.append(f"{path}: block {name} has no generator output at {source}")
            continue
        expected = source.read_text(encoding="utf-8").strip() + "\n"
        actual = match.group("body")
        if actual.strip() + "\n" != expected:
            if fix:
                replaced = replaced.replace(
                    match.group(0),
                    f"<!-- BEGIN generated:{name} -->\n{expected}<!-- END generated:{name} -->",
                )
            else:
                problems.append(
                    f"{path}: block {name} is stale; run "
                    "`python scripts/make_results_table.py` then "
                    "`python scripts/check_docs.py --fix-blocks`"
                )
    if fix and replaced != text:
        write_text_lf(path, replaced)
    return problems


def check_numbers(path: Path, published: set[str], allowed: set[str]) -> list[str]:
    problems = []
    for number in sorted(set(NUMBER.findall(strip_uncheckable(path.read_text(encoding="utf-8"))))):
        if number in allowed or number in published:
            continue
        problems.append(
            f"{path}: {number} is not in results/published/ and not in "
            "docs/allowed_numbers.txt"
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--published", default="results/published", type=Path)
    parser.add_argument("--table-dir", default="docs/tables", type=Path)
    parser.add_argument("--allowed", default="docs/allowed_numbers.txt", type=Path)
    parser.add_argument("--docs", nargs="*", default=None)
    parser.add_argument("--fix-blocks", action="store_true")
    parser.add_argument("--require-results", action="store_true",
                        help="fail if results/published is empty rather than skipping")
    args = parser.parse_args(argv)

    documents = (
        [Path(p) for p in args.docs]
        if args.docs
        else [Path("README.md"), *sorted(Path("docs").glob("*.md"))]
    )

    if not args.published.exists() or not any(args.published.rglob("*.json")):
        message = f"no published results under {args.published}"
        if args.require_results:
            print(message, file=sys.stderr)
            return 1
        print(f"{message}; skipping the number check")
        return 0

    published = collect_published(args.published)
    allowed: set[str] = set()
    if args.allowed.exists():
        for line in args.allowed.read_text(encoding="utf-8").splitlines():
            token = line.split("#", 1)[0].strip()
            if token:
                allowed.add(token)

    problems: list[str] = []
    for document in documents:
        if not document.exists():
            continue
        problems.extend(check_blocks(document, args.table_dir, args.fix_blocks))
        problems.extend(check_numbers(document, published, allowed))

    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        print(f"\n{len(problems)} problems", file=sys.stderr)
        return 1
    print(f"checked {len(documents)} documents against {len(published)} published values")
    return 0


if __name__ == "__main__":
    sys.exit(main())
