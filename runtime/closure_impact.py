"""Locate old executable source which must be reached before an edited function can diverge."""

import ast
from functools import lru_cache
from types import CodeType

from source_fast import analysis_tree

FOREIGN_STATE = (ast.Yield, ast.YieldFrom, ast.Await, ast.Global, ast.Nonlocal)
STRUCTURED = (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.TryStar)
FUNCTION = ast.FunctionDef


@lru_cache(maxsize=16)
def compiled_scopes(source, path):
    module = compile(source, path, "exec", dont_inherit=True)
    pending = [module]
    found = {}
    while pending:
        scope = pending.pop()
        key = scope.co_qualname
        info = (
            scope.co_flags,
            scope.co_cellvars,
            scope.co_freevars,
            frozenset(scope.co_varnames),
        )
        found[key] = info if key not in found else None
        pending.extend(
            const for const in scope.co_consts if isinstance(const, CodeType)
        )
    return found


def analyze(change, sel, apply, equal):
    path = change["old"]
    if (
        path != change["new"]
        or not path.startswith("sqlglot/")
        or not path.endswith(".py")
    ):
        return None
    raw, old = sel.source(path)
    updated = apply(raw, change)[0]
    new = analysis_tree(change, old, raw, updated)
    old_code = compiled_scopes(raw, path)
    new_code = compiled_scopes(updated, path)
    points = set()
    narrower_import = False
    scopes = set()
    stack = []

    def mark(stmt):
        points.update(range(stmt.lineno, stmt.end_lineno + 1))
        scopes.add(stack[-1])

    def body_only(a, b):
        return type(a) is type(b) and all(
            equal(getattr(a, k), getattr(b, k)) for k in a._fields if k != "body"
        )

    def function(a, b, parent):
        name = parent + a.name
        before, after = old_code.get(name), new_code.get(name)
        if (
            not body_only(a, b)
            or not before
            or not after
            or before[:3] != after[:3]
            or ast.get_docstring(a, clean=False) != ast.get_docstring(b, clean=False)
            or any(
                not isinstance(d, ast.Name)
                or d.id not in {"property", "staticmethod", "classmethod"}
                for d in a.decorator_list
            )
        ):
            return False
        # The caller can observe a different generator/coroutine or a captured variable
        # without executing its body. Leave those functions to the normal selector.
        if any(isinstance(n, FOREIGN_STATE) for fn in (a, b) for n in ast.walk(fn)):
            return False
        stack.append(name)
        if not block(a.body, b.body, name + ".<locals>.", a):
            return False
        if before[-1] != after[-1]:
            # Rebinding a name changes compile-time name resolution throughout the function.
            # Reaching any executable part of this function is conservatively sufficient.
            for stmt in a.body:
                mark(stmt)
        stack.pop()
        return True

    def block(left, right, parent, owner):
        nonlocal narrower_import
        if len(left) != len(right):
            # A new unconditional statement may run on a path which skipped the
            # next old statement. Include every executed line of the old owner.
            if isinstance(owner, FUNCTION):
                for item in owner.body:
                    mark(item)
            else:
                mark(owner)
            return True
        for a, b in zip(left, right):
            if equal(a, b):
                continue
            if isinstance(a, FUNCTION) and isinstance(b, FUNCTION):
                if not function(a, b, parent):
                    return False
            elif isinstance(a, ast.ClassDef) or isinstance(b, (ast.ClassDef, FUNCTION)):
                return False
            elif isinstance(a, STRUCTURED) and type(a) is type(b):
                fields = [
                    s for s in a._fields if s not in ("body", "orelse", "finalbody")
                ]
                if any(not equal(getattr(a, k), getattr(b, k)) for k in fields):
                    mark(a)
                else:
                    for field in ("body", "orelse", "finalbody"):
                        if not block(
                            getattr(a, field, []), getattr(b, field, []), parent, a
                        ):
                            return False
            else:
                if isinstance(a, ast.ImportFrom) and isinstance(b, ast.ImportFrom):
                    old_names = {(alias.name, alias.asname) for alias in a.names}
                    new_names = {(alias.name, alias.asname) for alias in b.names}
                    if (
                        a.module == b.module
                        and a.level == b.level
                        and new_names < old_names
                        and all(k != "*" for k, _ in old_names)
                    ):
                        narrower_import = True
                mark(a)
        return True

    def declarations(left, right, parent):
        if len(left) != len(right):
            return False
        for a, b in zip(left, right):
            if equal(a, b):
                continue
            if isinstance(a, FUNCTION) and isinstance(b, FUNCTION):
                if not function(a, b, parent):
                    return False
            elif isinstance(a, ast.ClassDef) and body_only(a, b):
                if not declarations(a.body, b.body, parent + a.name + "."):
                    return False
            else:
                return False
        return True

    if not declarations(old.body, new.body, "") or not points:
        return None
    return frozenset(points), frozenset(scopes), narrower_import


def records(sel, changes, apply, equal):
    result = {}
    for change in changes:
        if "_closure_impact" not in change:
            try:
                change["_closure_impact"] = analyze(change, sel, apply, equal)
            except (SyntaxError, ValueError, KeyError, TypeError, RecursionError):
                change["_closure_impact"] = None
        record = change["_closure_impact"]
        if record is None or change["old"] in result:
            return None
        result[change["old"]] = record
    return result or None


def spans(sel, changes, apply, equal):
    rows = records(sel, changes, apply, equal)
    return {p: r[0] for p, r in rows.items()} if rows else None


def refine_import(sel, changes, selected, apply, equal):
    """A changed local from-import executes only when its containing code runs."""
    if len(selected) < 100 or not changes or sel.unknown:
        return selected
    if not any(
        line.startswith(("+", "-")) and line[1:].lstrip().startswith("from ")
        for item in changes
        for hunk in item["hunks"]
        for line in hunk["lines"]
    ):
        return selected
    rows = records(sel, changes, apply, equal)
    if not rows or not any(row[2] for row in rows.values()):
        return selected
    from assertion_replay import index

    methods, footprints, _, _ = index()
    tested = set()
    bits = 0
    touched = {}
    for path, (lines, scopes, _) in rows.items():
        sel.load_shard(path)
        if set(sel.mapping.get("import_lines", {}).get(path, ())).intersection(lines):
            return selected
        tested.update(sel.line_tests(path, lines))
        if path in sel.files:
            tested.update(sel.sets[sel.files[path]])
        # A nested function may be created while the class loads but only its
        # later invocation can reach a local import inside the closure.
        for name in scopes:
            bits |= footprints.get((path, name), 0)
        touched[path] = lines
    if bits & 1:
        return selected
    for number in selected - tested:
        name = sel.by_number[number]["id"]
        if name not in methods:
            tested.add(number)
            continue
        start, durations, _ = methods[name]
        if (bits >> start) & ((1 << len(durations)) - 1):
            tested.add(number)
    return (selected & tested) | sel.confirmed_failures
