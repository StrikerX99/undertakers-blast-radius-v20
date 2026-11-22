"""Find tests reading transitive aliases of token sets constructed by the base parser."""

import ast
import json
from functools import lru_cache
from pathlib import Path

BASE = "sqlglot/parser.py"
HERE = Path(__file__).resolve().parent


def assigned(stmt):
    if isinstance(stmt, ast.AnnAssign):
        return [stmt.target]
    return stmt.targets if isinstance(stmt, ast.Assign) else ()


def name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def build(root, output):
    aliases, uses = {}, {}
    for src in sorted((root / "sqlglot").rglob("*.py")):
        path = src.relative_to(root).as_posix()
        tree = ast.parse(src.read_bytes())
        for stmt in ast.walk(tree):
            targets = {name(x) for x in assigned(stmt)} - {None}
            if targets:
                value = stmt.value
                if value:
                    for node in ast.walk(value):
                        read = name(node)
                        if read and read.isupper():
                            aliases.setdefault(read, set()).update(
                                k for k in targets if k.isupper()
                            )
            read = name(stmt)
            if read and read.isupper() and isinstance(stmt.ctx, ast.Load):
                uses.setdefault(read, {}).setdefault(path, set()).add(stmt.lineno)
    output.write_text(
        json.dumps(
            {
                "aliases": {key: sorted(vals) for key, vals in aliases.items()},
                "uses": {
                    key: {p: sorted(ls) for p, ls in files.items()}
                    for key, files in uses.items()
                },
            },
            separators=(",", ":"),
        )
    )


@lru_cache(maxsize=1)
def table():
    return json.loads((HERE / "grammar-uses.json").read_bytes())


def extend(sel, path, original, changed_lines, candidates):
    if path != BASE:
        return
    cls = next(
        n for n in original.body if isinstance(n, ast.ClassDef) and n.name == "Parser"
    )
    touched = set()
    for stmt in cls.body:
        if not any(stmt.lineno <= line <= stmt.end_lineno for line in changed_lines):
            continue
        if not isinstance(stmt, (ast.Assign, ast.AnnAssign)) or not stmt.value:
            continue
        # Shared grammar sets are built at import. A later lookup can read a
        # precomputed alias without executing the original construction again.
        if not isinstance(stmt.value, (ast.Set, ast.BinOp, ast.Name, ast.Attribute)):
            continue
        touched.update(
            n for target in assigned(stmt) if (n := name(target)) and n.isupper()
        )
    if not touched:
        return
    info = table()
    pending, done = list(touched), set()
    lines = {}
    while pending:
        key = pending.pop()
        if key in done:
            continue
        done.add(key)
        pending.extend(info["aliases"].get(key, ()))
        for p, ns in info["uses"].get(key, {}).items():
            lines.setdefault(p, set()).update(ns)
    for p, ns in lines.items():
        sel.load_shard(p)
        candidates.update(sel.line_tests(p, ns))


if __name__ == "__main__":
    import sys

    build(Path(sys.argv[1]), Path(sys.argv[2]))
