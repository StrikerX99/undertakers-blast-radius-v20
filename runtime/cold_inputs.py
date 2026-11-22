"""Reject expensive candidates with independent evidence that the changed files are unused."""

import ast
import importlib
import json
import os
import sys
import tempfile
import zlib
from functools import lru_cache
from pathlib import Path

import forkprobe
from source_fast import paired_tree


@lru_cache(maxsize=1)
def measurements():
    return {
        name: frozenset(paths)
        for name, paths in json.loads(
            zlib.decompress(Path(__file__).with_name("cold.dat").read_bytes())
        ).items()
    }


def pure_value(node):
    allowed = (
        ast.Constant,
        ast.Name,
        ast.Load,
        ast.Attribute,
        ast.Dict,
        ast.List,
        ast.Set,
        ast.Tuple,
        ast.BinOp,
        ast.BitOr,
        ast.BitAnd,
        ast.BitXor,
        ast.Sub,
        ast.Starred,
        ast.UnaryOp,
        ast.USub,
        ast.UAdd,
    )
    return node is not None and all(isinstance(n, allowed) for n in ast.walk(node))


def limited_scope(a, b, equal):
    if equal(a, b):
        return True
    if type(a) is not type(b):
        return False
    if isinstance(a, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        if any(
            not equal(getattr(a, k), getattr(b, k)) for k in a._fields if k != "body"
        ):
            return False
        if isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # A deferred calculation may change, but process-wide state,
            # environment, module discovery and the callable kind must stay local.
            forbidden = (
                ast.Import,
                ast.ImportFrom,
                ast.Global,
                ast.Nonlocal,
                ast.Yield,
                ast.YieldFrom,
                ast.Await,
            )
            dynamic = {
                "sys",
                "os",
                "open",
                "exec",
                "eval",
                "globals",
                "setattr",
                "delattr",
                "__import__",
                "inspect",
            }
            return all(
                not isinstance(n, forbidden)
                and not (isinstance(n, ast.Name) and n.id in dynamic)
                for fn in (a, b)
                for n in ast.walk(fn)
            )
        if len(a.body) != len(b.body):
            return False
        return all(limited_scope(x, y, equal) for x, y in zip(a.body, b.body))
    if isinstance(a, (ast.Assign, ast.AnnAssign)):
        return (
            pure_value(a.value)
            and pure_value(b.value)
            and all(
                equal(getattr(a, k), getattr(b, k)) for k in a._fields if k != "value"
            )
        )
    return False


def load_modules(payload):
    root, names, marker = payload
    os.chdir(root)
    sys.path.insert(0, str(root))
    for m in names:
        importlib.import_module(m)
    marker.touch()


def refine(sel, chs, selected, apply, equal, checkout, cost, remaining):
    paths = {c["old"] for c in chs}
    if not selected or remaining < 1 or not all(c["old"] == c["new"] for c in chs):
        return selected
    # This proof covers leaf dialect implementation modules and test data.
    if not all(
        p.startswith(
            (
                "sqlglot/parsers/",
                "sqlglot/generators/",
                "sqlglot/dialects/",
                "sqlglot/tokenizers/",
                "sqlglot/typing/",
                "tests/fixtures/",
            )
        )
        for p in paths
    ):
        return selected
    if any(p.endswith(("/__init__.py", "/dialect.py", "/base.py")) for p in paths):
        return selected
    possible = {
        sel.by_name[n]
        for n, used in measurements().items()
        if n in sel.by_name and sel.by_name[n] in selected and not (paths & used)
    }
    if not possible or cost(selected) - cost(selected - possible) < 300:
        return selected
    modules = []
    if not hasattr(sel, "suite_body_readers"):
        sel.suite_body_readers = {}
    for c in chs:
        path = c["old"]
        raw, old = sel.source(path)
        updated = apply(raw, c)[0]
        if path in sel.files:
            possible.difference_update(sel.sets[sel.files[path]])
        if not path.endswith(".py"):
            continue

        after = paired_tree(c, old, raw, updated)
        if not limited_scope(old, after, equal):
            return selected
        if path not in sel.suite_body_readers:
            imports = set(sel.mapping.get("import_lines", {}).get(path, ()))
            sel.suite_body_readers[path] = sel.line_tests(
                path, {int(n) for n in sel.lines.get(path, {})} - imports
            )
        possible.difference_update(sel.suite_body_readers[path])
        name = path[:-3].replace("/", ".")
        modules.append(name)
    if not possible or cost(selected) - cost(selected - possible) < 300:
        return selected
    if modules:
        with tempfile.TemporaryDirectory(prefix="br-cold-") as tmp:
            root, marker = Path(tmp) / "repo", Path(tmp) / "ok"
            checkout(root, chs, sel)
            if (
                not forkprobe.run(
                    (root, modules, marker),
                    {"PYTHONPATH": str(root)},
                    load_modules,
                    min(2.5, remaining),
                )
                or not marker.exists()
            ):
                return selected
    return selected - possible
