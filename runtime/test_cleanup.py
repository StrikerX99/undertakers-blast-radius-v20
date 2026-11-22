"""Track removed Python bindings used by cleanup code in the pinned tests."""

import ast
from functools import lru_cache
import json
from pathlib import Path
import sys

INDEX = Path(__file__).with_name("cleanup-reads.json")


def build(root, target):
    reads = set()
    for path in (root / "tests").rglob("*.py"):
        tree = ast.parse(path.read_bytes())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try) or not node.finalbody:
                continue
            for statement in node.finalbody:
                nodes = list(ast.walk(statement))
                # Record cleanup which changes an attribute on a class/module or
                # other shared object. A local variable being discarded is harmless.
                if not any(
                    isinstance(n, (ast.Assign, ast.Delete, ast.AugAssign, ast.Call))
                    for n in nodes
                ):
                    continue
                reads.update(
                    n.attr
                    for n in nodes
                    if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load)
                )
                reads.update(
                    n.id
                    for n in nodes
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                )
    target.write_text(json.dumps(sorted(reads), separators=(",", ":")))


@lru_cache(maxsize=1)
def names():
    return frozenset(json.loads(INDEX.read_bytes()))


def bound(tree):
    here = set()
    for stmt in tree.body:
        if isinstance(stmt, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            here.add(stmt.name)
            if isinstance(stmt, ast.ClassDef):
                here.update(bound(stmt))
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            nodes = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            here.update(
                x.id
                for node in nodes
                for x in ast.walk(node)
                if isinstance(x, ast.Name)
            )
    return here


def changed(sel, changes, apply):
    if not changes:
        return False
    stamp = changes[0]
    if "_removed_cleanup_input" in stamp:
        return stamp["_removed_cleanup_input"]
    import re

    readers = names()
    for c in changes:
        p = c["old"]
        if p != c["new"] or not p.endswith(".py"):
            continue
        deleted = "\n".join(
            l[1:] for h in c["hunks"] for l in h["lines"] if l.startswith("-")
        )
        added = "\n".join(
            l[1:] for h in c["hunks"] for l in h["lines"] if l.startswith("+")
        )
        names_removed = set(re.findall(r"\b[A-Za-z_]\w*\b", deleted)) & readers
        if not names_removed:
            continue
        old, tree = sel.source(p)
        new = ast.parse(apply(old, c)[0])
        if (bound(tree) - bound(new)) & names_removed:
            stamp["_removed_cleanup_input"] = True
            return True
    stamp["_removed_cleanup_input"] = False
    return False


if __name__ == "__main__":
    build(Path(sys.argv[1]), Path(sys.argv[2]))
