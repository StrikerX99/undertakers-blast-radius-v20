"""Bind generator table edits to the receivers and expressions observed at dispatch."""

import ast
import hashlib
import json
import marshal
import os
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from types import FunctionType

import forkprobe
import generator_contexts
import import_cone
import reflective_use
import registry_cache
import shared_tables
import structural_dependencies
import symbol_impact
from verification_budget import pack_ms


def callback(node):
    """Only call two known pure factories at class creation. Other handlers stay deferred."""
    if isinstance(node, ast.Name):
        return True
    if isinstance(node, ast.Attribute):
        return isinstance(node.value, ast.Name) and node.value.id == "transforms"
    if isinstance(node, ast.Lambda):
        return import_cone.safe_callback(node, set())
    if not isinstance(node, ast.Call) or node.keywords or len(node.args) != 1:
        return False
    fn = node.func
    arg = node.args[0]
    if isinstance(fn, ast.Name) and fn.id == "rename_func":
        return isinstance(arg, ast.Constant) and isinstance(arg.value, str)
    return (
        isinstance(fn, ast.Attribute)
        and fn.attr == "preprocess"
        and isinstance(fn.value, ast.Name)
        and fn.value.id == "transforms"
        and isinstance(arg, (ast.List, ast.Tuple))
        and all(
            isinstance(x, (ast.Name, ast.Attribute)) and callback(x) for x in arg.elts
        )
    )


def factory_bindings(tree):
    sources = {
        "rename_func": "sqlglot.dialects.dialect",
        "transforms": "sqlglot",
        "exp": "sqlglot",
    }
    bound = set()
    for stmt in tree.body:
        if isinstance(stmt, ast.ImportFrom) and not stmt.level:
            bound.update(
                a.name
                for a in stmt.names
                if not a.asname and sources.get(a.name) == stmt.module
            )
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.discard(node.id)
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            bound.discard(node.name)
        elif isinstance(node, ast.alias) and node.asname:
            bound.discard(node.asname)
    return bound


def entries(node):
    rows = {}
    order = []
    for key, val in zip(node.keys, node.values):
        if (
            isinstance(key, ast.Attribute)
            and isinstance(key.value, ast.Name)
            and key.value.id == "exp"
        ):
            tag = key.attr
        else:
            tag = "!" + ast.dump(val if key is None else key)
        while tag in rows:
            tag = "!" + tag
        rows[tag] = val
        order.append(tag)
    return rows, order


def differences(left, right, eq, bound):
    if not (
        isinstance(left, ast.Assign)
        and isinstance(right, ast.Assign)
        and eq(left.targets, right.targets)
        and len(left.targets) == 1
        and isinstance(left.targets[0], ast.Name)
        and left.targets[0].id == "TRANSFORMS"
        and isinstance(left.value, ast.Dict)
        and isinstance(right.value, ast.Dict)
    ):
        return None
    a, x = entries(left.value)
    b, y = entries(right.value)
    both = a.keys() & b.keys()
    if [k for k in x if k in both] != [k for k in y if k in both]:
        return None
    delta = {k for k in a.keys() | b.keys() if not eq(a.get(k), b.get(k))}
    if not delta or any(
        k.startswith("!")
        or any(not callback(v) for v in (a.get(k), b.get(k)) if v is not None)
        for k in delta
    ):
        return None
    for k in delta:
        for value in (a.get(k), b.get(k)):
            if value is None:
                continue
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and "rename_func" not in bound
            ):
                return None
            if (
                any(
                    isinstance(v, ast.Name) and v.id == "transforms"
                    for v in ast.walk(value)
                )
                and "transforms" not in bound
            ):
                return None
    return delta if "exp" in bound else None


def inspect_edit(sel, change, apply, eq):
    path = change["old"]
    if (
        path != change["new"]
        or not path.startswith("sqlglot/generators/")
        or not path.endswith(".py")
    ):
        return None
    original, a = sel.source(path)
    text = apply(original, change)[0]
    if change.get("_compiled_source") != text:
        return None
    b = ast.parse(text)
    bound = factory_bindings(a)

    inherited_names = {
        v.name for v in a.body if isinstance(v, (ast.FunctionDef, ast.ClassDef))
    }
    extra = {
        v.name
        for v in b.body
        if isinstance(v, ast.FunctionDef) and v.name not in inherited_names
    }
    additions = [
        v for v in b.body if isinstance(v, ast.FunctionDef) and v.name in extra
    ]
    if additions and not any(
        isinstance(n, ast.ImportFrom)
        and n.module == "__future__"
        and any(t.name == "annotations" for t in n.names)
        for n in a.body
    ):
        return None
    if len(additions) != len(extra) or any(
        not v.name.startswith("_")
        or v.name.startswith("__")
        or not import_cone.local_function(v)
        for v in additions
    ):
        return None
    if any(
        isinstance(v, ast.alias)
        and (v.name == "*" or (v.asname or v.name) in extra)
        or isinstance(v, ast.Name)
        and v.id in extra
        or isinstance(v, ast.Constant)
        and isinstance(v.value, str)
        and v.value in extra
        for v in ast.walk(a)
    ):
        return None
    rest = [v for v in b.body if v not in additions]
    if len(a.body) != len(rest):
        return None
    changes = []
    for cls, newcls in zip(a.body, rest):
        if eq(cls, newcls):
            continue
        if not (
            isinstance(cls, ast.ClassDef)
            and isinstance(newcls, ast.ClassDef)
            and cls.name.endswith("Generator")
            and generator_contexts.header(cls, newcls, eq)
            and len(cls.body) == len(newcls.body)
        ):
            return None
        for idx, (left, right) in enumerate(zip(cls.body, newcls.body)):
            if eq(left, right):
                continue
            keys = differences(left, right, eq, bound)
            if not keys:
                return None
            # A new class variable can derive data at import time from the edited table.
            for after in cls.body[idx + 1 :]:
                early = (
                    [after.args, *after.decorator_list]
                    if isinstance(after, ast.FunctionDef)
                    else [after]
                )
                if any(
                    isinstance(t, (ast.Name, ast.Attribute))
                    and getattr(t, "attr", getattr(t, "id", None)) == "TRANSFORMS"
                    for e in early
                    for t in ast.walk(e)
                ):
                    return None
            changes.extend((cls.name, k) for k in sorted(keys))
    return (path, changes, extra) if changes else None


def state(root, requested):
    os.chdir(root)
    sys.path.insert(0, str(root))
    import sqlglot
    from sqlglot import Generator, exp
    from sqlglot.generator import _build_dispatch

    if Path(sqlglot.__file__).resolve() != (root / "sqlglot/__init__.py").resolve():
        raise ValueError("project origin")
    complete = requested is None
    result = {}
    memo = {}

    def signature(fn):
        if fn is None:
            return "missing"
        if fn not in memo:
            try:
                if not isinstance(fn, FunctionType):
                    raise ValueError
                data = shared_tables.encode(fn)
                memo[fn] = hashlib.sha256(
                    json.dumps(data, separators=(",", ":")).encode()
                ).digest()
            except (ValueError, TypeError, RecursionError):
                memo[fn] = None
        return memo[fn]

    for cls in registry_cache.concrete_receivers(root):
        if not issubclass(cls, Generator):
            continue
        owner = cls.__module__ + ":" + cls.__qualname__
        table = _build_dispatch(cls)
        if complete:
            pairs = sorted(
                {(e.__name__, e) for e in table.keys() | cls.TRANSFORMS.keys()},
                key=lambda x: x[0],
            )
        else:
            pairs = [(k, getattr(exp, k)) for k in requested]
        row = {}
        for k, expr in pairs:
            if k in row or not isinstance(expr, type):
                raise ValueError("ambiguous expression")
            row[k] = [
                expr in cls.TRANSFORMS,
                signature(cls.TRANSFORMS.get(expr)),
                signature(table.get(expr)),
            ]
        result[owner] = row
    return result


def probe(payload):
    root, keys, out = payload
    out.write_bytes(marshal.dumps(state(root, keys)))


@lru_cache(maxsize=1)
def baseline():
    return marshal.loads(
        Path(__file__).with_name("generator-cells-state.bin").read_bytes()
    )


def refine(sel, changes, selected, apply, eq, checkout, remaining):
    if remaining < 2 or len(changes) != 1 or not selected:
        return selected
    proof = inspect_edit(sel, changes[0], apply, eq)
    if not proof:
        return selected
    path, cells, helpers = proof
    groups = generator_contexts.data_dispatch()
    if (
        not groups["sources"].get(path)
        or groups["sources"].get(path) != sel.mapping["source_hashes"].get(path)
        or groups["commit"] != sel.mapping["commit"]
    ):
        return selected
    keys = {key for owner, key in cells}
    if keys & set(groups["structural"]["dialect_init_keys"]):
        return selected
    # All sources observed through reflection, file access and custom test classes
    # remain possible, even when no ordinary dialect dispatches this expression.
    spec = generator_contexts.data("parser-cells.json")
    refs = reflective_use.indices(sel, spec, keys | helpers | {"TRANSFORMS"})
    users = set(sel.confirmed_failures) | sel.unknown
    users.update(
        sel.by_name[spec["names"][i]] for i in refs if spec["names"][i] in sel.by_name
    )
    external_receivers = import_cone.receiver_users(sel, [])
    if external_receivers is None:
        return selected
    users.update(external_receivers)
    if path in sel.files:
        users.update(sel.sets[sel.files[path]])
    subscribers = symbol_impact.index()
    for name in helpers:
        for rp, line, scope, kind in subscribers.get(name, ()):
            if not scope or kind.startswith("from:"):
                return selected
            if rp.startswith("tests/"):
                users.update(t["number"] for t in sel.tests if t["file"] == rp)
            else:
                sel.load_shard(rp)
                users.update(sel.line_tests(rp, [line]))
    module = path[:-3].replace("/", ".")
    protected = set(users)
    for cls, key in cells:
        users.update(
            structural_dependencies.table_observers(
                sel, groups, path, module + ":" + cls, {key}
            )
        )
    optimistic = users & selected
    if optimistic == selected:
        return selected

    def cost(picks):
        return pack_ms(round(1000 * sel.by_number[n]["seconds"]) for n in picks)

    if cost(selected) - cost(optimistic) < 400 and len(selected) - len(optimistic) < 70:
        return selected
    old = baseline()
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src"
        out = Path(tmp) / "state"
        checkout(src, changes, sel)
        forkprobe.run((src, sorted(keys), out), {}, probe, min(remaining, 2.8))
        if not out.exists():
            return selected
        new = marshal.loads(out.read_bytes())
    if old.keys() != new.keys():
        return selected
    delta = set()
    for owner, rows in new.items():
        for key, current in rows.items():
            prior = old[owner].get(key, [False, "missing", "missing"])
            if current != prior or None in current or None in prior:
                delta.add((owner,key))
                users.update(
                    structural_dependencies.table_observers(
                        sel, groups, path, owner, {key}
                    )
                )
    if not helpers:
        sel.validator_keys = (delta, protected)
    surviving = (users & selected) | sel.confirmed_failures
    if surviving != selected:
        sel.registry_candidates = frozenset(surviving)
    return surviving


if __name__ == "__main__":
    root, output = map(Path, sys.argv[1:])
    output.write_bytes(marshal.dumps(state(root, None)))
