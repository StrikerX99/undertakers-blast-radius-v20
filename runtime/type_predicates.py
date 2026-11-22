"""Prove redundant exact-type guards using the semantics of unshadowed Python builtins."""

import ast
import builtins
import functools
import os
from pathlib import Path
import sys
import tempfile

import forkprobe

TYPES = {
    t.__name__: t
    for t in (
        bool,
        int,
        float,
        complex,
        str,
        bytes,
        bytearray,
        tuple,
        list,
        set,
        frozenset,
        dict,
        range,
        slice,
        object,
    )
}


def types_in(expr):
    parts = expr.elts if isinstance(expr, ast.Tuple) else [expr]
    if not all(isinstance(x, ast.Name) and x.id in TYPES for x in parts):
        return ()
    return tuple(x.id for x in parts)


def exact(expr):
    if (
        not isinstance(expr, ast.Compare)
        or len(expr.ops) != 1
        or not isinstance(expr.ops[0], ast.Is)
    ):
        return None
    call, cls = expr.left, expr.comparators[0]
    if isinstance(call, ast.Name):
        call, cls = cls, call
    if (
        not isinstance(call, ast.Call)
        or not isinstance(call.func, ast.Name)
        or call.func.id != "type"
        or call.keywords
        or len(call.args) != 1
        or not isinstance(call.args[0], ast.Name)
        or not isinstance(cls, ast.Name)
        or cls.id not in TYPES
    ):
        return None
    return call.args[0].id, cls.id


def accepts(expr):
    if (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Name)
        and expr.func.id == "isinstance"
        and len(expr.args) == 2
        and not expr.keywords
        and isinstance(expr.args[0], ast.Name)
    ):
        return expr.args[0].id, types_in(expr.args[1])
    return None


class Simplify(ast.NodeTransformer):
    def __init__(self):
        self.bindings = set()
        self.subjects = set()

    def visit_BoolOp(self, node):
        node = self.generic_visit(node)
        if not isinstance(node.op, ast.Or) or len(node.values) != 2:
            return node
        # Require exact-type comparison first. Its evaluation neither dispatches
        # to the object nor invokes a custom instance check.
        a, b = node.values
        narrow, broad = exact(a), accepts(b)
        if not narrow or not broad or narrow[0] != broad[0]:
            return node
        if not any(issubclass(TYPES[narrow[1]], TYPES[t]) for t in broad[1]):
            return node
        self.subjects.add(narrow[0])
        self.bindings.update(("isinstance", "type", narrow[1], *broad[1]))
        return b


def checked_namespace(payload):
    src, requirements, marker = payload
    import importlib

    os.chdir(src)
    sys.path.insert(0, str(src))
    for module, required in requirements:
        m = importlib.import_module(module)
        if type(m).__name__ != "module":
            return
        for n in required:
            if vars(m).get(n, getattr(builtins, n)) is not getattr(builtins, n):
                return
    marker.touch()


def residual(sel, changes, apply, equal, checkout, remaining):
    if not changes or sel.unknown or remaining < 0.6:
        return None
    evidence = []
    readers = set()
    for c in changes:
        p = c["old"]
        if p != c["new"] or not p.startswith("sqlglot/") or not p.endswith(".py"):
            return None
        edit = " ".join(
            line[1:]
            for h in c["hunks"]
            for line in h["lines"]
            if line.startswith(("+", "-"))
        )
        if "isinstance" not in edit or "type(" not in edit:
            return None
        raw, old = sel.source(p)
        new_raw = apply(raw, c)[0]
        new = ast.parse(new_raw)
        # No moved statements or line-number changes anywhere in this narrow shortcut.
        if len(raw.splitlines()) != len(new_raw.splitlines()) or len(old.body) != len(
            new.body
        ):
            return None
        needed = set()

        def compare_declarations(left, right):
            if len(left) != len(right):
                return False
            for a, b in zip(left, right):
                if equal(a, b):
                    continue
                if isinstance(a, ast.ClassDef) and isinstance(b, ast.ClassDef):
                    if not all(
                        equal(getattr(a, f), getattr(b, f))
                        for f in a._fields
                        if f != "body"
                    ) or not compare_declarations(a.body, b.body):
                        return False
                    continue
                if type(a) is not ast.FunctionDef or type(b) is not type(a):
                    return False
                if any(
                    not isinstance(d, ast.Name)
                    or d.id not in {"classmethod", "staticmethod", "property"}
                    for d in a.decorator_list
                ):
                    return False
                if not all(
                    equal(getattr(a, f), getattr(b, f))
                    for f in a._fields
                    if f != "body"
                ):
                    return False
                if any(
                    isinstance(n, (ast.ClassDef, ast.Lambda, ast.Global, ast.Nonlocal))
                    or isinstance(n, ast.FunctionDef)
                    and n is not a
                    for n in ast.walk(a)
                ):
                    return False
                import copy

                filt = Simplify()
                aa = filt.visit(copy.deepcopy(a))
                bb = filt.visit(copy.deepcopy(b))
                if not filt.bindings or not equal(aa, bb):
                    return False
                args = {n.arg for n in ast.walk(a.args) if isinstance(n, ast.arg)}
                if not filt.subjects <= args or args & filt.bindings:
                    return False
                for n in ast.walk(a):
                    if (
                        isinstance(n, ast.Name)
                        and isinstance(n.ctx, (ast.Store, ast.Del))
                        and n.id in filt.bindings
                    ):
                        return False
                    if isinstance(n, (ast.Import, ast.ImportFrom)) and any(
                        (o.asname or o.name.split(".")[0]) in filt.bindings
                        for o in n.names
                    ):
                        return False
                needed.update(filt.bindings)
            return True

        if not compare_declarations(old.body, new.body) or not needed:
            return None
        # Dynamic rebinding can change the predicates even after a successful import.
        if any(
            isinstance(n, (ast.Global, ast.Nonlocal)) and set(n.names) & needed
            for n in ast.walk(old)
        ):
            return None
        evidence.append(
            (p[:-3].replace("/", ".").removesuffix(".__init__"), tuple(needed))
        )
        if p in sel.files:
            readers.update(sel.sets[sel.files[p]])
    with tempfile.TemporaryDirectory(prefix="type-guard-") as name:
        root = Path(name) / "src"
        mark = Path(name) / "ok"
        checkout(root, changes, sel)
        forkprobe.run(
            (root, evidence, mark),
            {"PYTHONPATH": str(root)},
            checked_namespace,
            min(1.5, remaining),
        )
        return readers if mark.is_file() else None
