"""Exercise the pinned parser stress test at its actual user-visible node limits."""

import hashlib
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest

import forkprobe
from verification_budget import pack_ms

TEST = "tests/test_parser.py::TestParser::test_max_nodes"
SOURCE_SHA = "2114cab8c35f04743a069627a6ad480d2175cf396b8d41deae3df58ece0f4ab4"
# A size-dependent edit may distinguish two inputs containing different numbers
# of repeated columns. Keep the original in that case, even if the probe passes.
SIZE_SENSITIVE = re.compile(
    r"\b(?:max_nodes|_node_count|setrecursionlimit|recursionlimit|getrecursionlimit|"
    r"memory_limit|MAX_NODES|MAX_TOKENS)\b|\b(?:\d[\d_]{2,}|0[xX][\da-fA-F_]+)\b"
)
MEASURE = r"(?:\b(?:len|sizeof)\([^)]*\)|\b(?:_tokens_size|_index|size|length|count)\b)"
COMPARATOR = r"(?:<=|>=|==|!=|<|>)"
LIMIT_AFTER_SIZE = re.compile(MEASURE + r"\s*" + COMPARATOR + r"\s*(.*)")
LIMIT_BEFORE_SIZE = re.compile(
    r"\b(?:\d[\d_]*|limit|threshold|[A-Z][A-Z_]+)\s*"
    + COMPARATOR
    + r"\s*(?:self\.)?"
    + MEASURE
)
SMALL_LIMIT = re.compile(r"[0-4]\b(?:[):,]|$|\s+(?:and|or)\b)")


def eligible(changes):
    """Keep the full test for edits with visible input-size or node-budget dependencies."""
    if not changes:
        return False
    for change in changes:
        path = change["old"]
        structural = "/dev/null" in (path, change["new"])
        if structural:
            path = change["new"] if path == "/dev/null" else path
            if not path.startswith(("sqlglot/parsers/", "sqlglot/generators/", "sqlglot/dialects/")):
                return False
        if (
            (not structural and path != change["new"])
            or not path.startswith("sqlglot/")
            or not path.endswith(".py")
        ):
            return False
        for hunk in change["hunks"]:
            for line in hunk["lines"]:
                code = line[1:].strip()
                if line[:1] not in ("+", "-") or code.startswith("#"):
                    continue
                if SIZE_SENSITIVE.search(code) or LIMIT_BEFORE_SIZE.search(code):
                    return False
                match = LIMIT_AFTER_SIZE.search(code)
                if match and not SMALL_LIMIT.match(match[1]):
                    return False
    return True


def parser_trial():
    """Run the pinned assertions and exercise both actual large allocation boundaries."""
    test_hash = hashlib.sha256(Path("tests/test_parser.py").read_bytes()).hexdigest()
    if test_hash != SOURCE_SHA:
        return False
    from sqlglot import Parser
    from tests import test_parser

    original_parse = test_parser.parse_one
    original_validate = Parser.validate_expression
    equality_chain = "x" + "=x" * 100
    large_query = "SELECT " + ",".join([equality_chain] * 10000)
    compact_query = "SELECT " + ",".join([equality_chain] * 32)
    remaining = {1_000_000: 3200, 50_000: 1000}
    visited = []

    def parse(sql, *args, **kwargs):
        if sql != large_query:
            return original_parse(sql, *args, **kwargs)
        limit = kwargs.get("max_nodes")
        if args or set(kwargs) != {"max_nodes"} or limit not in remaining:
            raise ValueError("Unrecognized stress call")
        seen = False

        def validate(parser, expression, *args, **kwargs):
            nonlocal seen
            if not seen:
                if parser.max_nodes != limit or parser._node_count != 0:
                    raise ValueError("Unexpected initial parser state")
                # Each repeated equality column accounts for exactly 100 counted
                # expressions on the pinned parser. At this point we stand in
                # for the omitted prefix; the patched parser enforces the actual
                # public limit and produces its own exception and error message.
                parser._node_count = limit - remaining[limit]
                seen = True
                visited.append(limit)
            return original_validate(parser, expression, *args, **kwargs)

        Parser.validate_expression = validate
        try:
            return original_parse(compact_query, **kwargs)
        finally:
            Parser.validate_expression = original_validate

    test_parser.parse_one = parse
    try:
        outcome = unittest.TestResult()
        test_parser.TestParser("test_max_nodes").run(outcome)
        return (
            outcome.wasSuccessful()
            and not outcome.skipped
            and visited == [1_000_000, 50_000]
        )
    finally:
        test_parser.parse_one = original_parse


def child(args):
    root, marker = args
    os.chdir(root)
    sys.path.insert(0, str(root))
    if parser_trial():
        marker.touch()


def refine(sel, changes, selected, checkout, remaining):
    ident = sel.by_name.get(TEST)
    if (
        ident not in selected
        or ident in sel.confirmed_failures
        or remaining < 1.0
        or not eligible(changes)
    ):
        return selected
    durations = getattr(sel, "scaled_durations", None)
    if durations is None:
        durations = sel.scaled_durations = {
            t["number"]: round(t["seconds"] * 1000) for t in sel.tests
        }
    before = pack_ms(durations[n] for n in selected)
    after = pack_ms(durations[n] for n in selected if n != ident)
    if before - after < 900:
        return selected
    with tempfile.TemporaryDirectory(prefix="parser-boundary-") as tmp:
        root = Path(tmp) / "src"
        marker = Path(tmp) / "pass"
        checkout(root, changes, sel)
        forkprobe.run((root, marker), {}, child, min(remaining, 2.8))
        if marker.exists():
            return selected - {ident}
    return selected
