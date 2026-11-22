"""Discharge TPC-DS when every effective parser edit starts at an unvisited statement."""

import ast
from functools import lru_cache
import json
from pathlib import Path

import cold_parser_lines
import executor_stages
import generator_contexts
import import_cone
import parser_evidence
import symbol_impact

CORE = "sqlglot/parser.py"
TARGET = "tests/test_executor.py::TestExecutor::test_execute_tpcds"


@lru_cache(maxsize=1)
def unused_modules():
    return json.loads(Path(__file__).with_name("executor-unused-modules.json").read_bytes())


def differing_sites(left, right, equal, sites):
    if equal(left, right):
        return True
    if type(left) != type(right):
        return False
    if isinstance(left, (ast.Expr, ast.Return, ast.Raise)):
        sites.update(range(left.lineno, left.end_lineno + 1))
        return not any(isinstance(n, (ast.Yield, ast.YieldFrom, ast.NamedExpr, ast.Await))
                       for item in (left, right) for n in ast.walk(item))
    if isinstance(left, list):
        return len(left) == len(right) and all(
            differing_sites(a, b, equal, sites) for a, b in zip(left, right)
        )
    if isinstance(left, (ast.If, ast.While, ast.For, ast.With, ast.Try, ast.ExceptHandler)):
        for key, old in ast.iter_fields(left):
            new = getattr(right, key)
            if key in ("body", "orelse", "finalbody", "handlers"):
                if not differing_sites(old, new, equal, sites):
                    return False
            elif not equal(old, new):
                return False
        return True
    return False


def annotation(node):
    return (
        node is None or isinstance(node, ast.Name)
        or (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "exp")
    )


def helper(fn):
    if not isinstance(fn, ast.FunctionDef) or not fn.name.startswith("_") or fn.name.startswith("__"):
        return False
    args = fn.args
    if (len(args.args) != 1 or args.args[0].arg != "self" or args.posonlyargs or args.kwonlyargs
            or args.vararg or args.kwarg or args.defaults or not import_cone.local_function(fn)):
        return False
    # Added methods only forward to an existing expression constructor.
    # Their definitions and annotation expressions cannot execute user code.
    if not annotation(fn.returns) or not annotation(args.args[0].annotation):
        return False
    if len(fn.body) != 1 or not isinstance(fn.body[0], ast.Return):
        return False
    value = fn.body[0].value
    return (isinstance(value, ast.Call) and not value.args and not value.keywords
            and isinstance(value.func, ast.Attribute) and isinstance(value.func.value, ast.Name)
            and value.func.value.id == "exp")


@lru_cache(maxsize=1)
def original_scopes(raw):
    return parser_evidence._compiled_scopes(raw, CORE)


def dormant(sel, change, apply, equal):
    raw, before = sel.source(CORE)
    updated = apply(raw, change)[0]
    if change.get("_compiled_source") != updated:
        return False
    after = ast.parse(updated)
    if len(before.body) != len(after.body):
        return False
    sites, added, changed = set(), {}, set()
    for left, right in zip(before.body, after.body):
        if equal(left, right):
            continue
        if not (isinstance(left, ast.ClassDef) and left.name == "Parser"
                and generator_contexts.header(left, right, equal)):
            return False
        names = {node.name for node in left.body if isinstance(node, ast.FunctionDef)}
        kept = []
        for node in right.body:
            if isinstance(node, ast.FunctionDef) and node.name not in names:
                if not helper(node) or node.name in added or node.name in raw:
                    return False
                if symbol_impact.index().get(node.name):
                    return False
                added[node.name] = node
            else:
                kept.append(node)
        if len(kept) != len(left.body):
            return False
        for old, new in zip(left.body, kept):
            if equal(old, new):
                continue
            if not (isinstance(old, ast.FunctionDef) and not old.name.startswith("__")
                    and generator_contexts.header(old, new, equal)
                    and ast.get_docstring(old, clean=False) == ast.get_docstring(new, clean=False)):
                return False
            if not differing_sites(old.body, new.body, equal, sites):
                return False
            changed.add(old.name)
    if not sites:
        return False
    # An extracted private expression may only be called directly by a changed
    # method. Postponed annotation evaluation and deferred method construction
    # have no side effects on this pinned base class.
    if added:
        allowed = set()
        calls = set()
        for fn in kept:
            if not isinstance(fn, ast.FunctionDef) or fn.name not in changed:
                continue
            for n in ast.walk(fn):
                if (isinstance(n, ast.Call) and not n.args and not n.keywords
                    and isinstance(n.func, ast.Attribute) and n.func.attr in added
                    and isinstance(n.func.value, ast.Name) and n.func.value.id == "self"):
                    allowed.add(n.func)
                    calls.add(n.func.attr)
        references = {n for n in ast.walk(after) if isinstance(n, ast.Attribute) and n.attr in added}
        if calls != set(added) or allowed != references:
            return False
    # Refuse added dormant methods which could override an existing declaration
    # used by another native phase even if there was no explicit access by name.
    stages = executor_stages.index()
    for name in added:
        if any(scope.rsplit(".", 1)[-1] == name
               for functions in stages["files"].values() for scope in functions):
            return False
    old_scope = original_scopes(raw)
    new_scope = parser_evidence._compiled_scopes(updated, CORE)
    if any(old_scope.get(n) != new_scope.get(n) for n in changed):
        return False
    spans = {CORE: sites}
    return (cold_parser_lines.untouched(sel.mapping, spans, "test_execute_tpcds")
            and executor_stages.query(spans, sel.mapping) == {})


def refine(sel, changes, selected, apply, equal):
    number = sel.by_name.get(TARGET)
    if number not in selected or number in sel.confirmed_failures or not changes:
        return selected
    if not any(c["old"] == CORE for c in changes):
        return selected
    core = []
    for change in changes:
        path = change["old"]
        if path != change["new"]:
            return selected
        if path == CORE:
            core.append(change)
        elif (path not in unused_modules() or unused_modules()[path] != sel.mapping["source_hashes"].get(path)):
            return selected
    if len(core) != 1 or not dormant(sel, core[0], apply, equal):
        return selected
    return selected - {number}
