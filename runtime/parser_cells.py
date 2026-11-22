"""Compare effective token sets and consult observations of readers of changed cells."""

import ast
import reflective_use
import importlib
import json
import os
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

from source_fast import paired_tree
import forkprobe
import module_footprints


@lru_cache(maxsize=1)
def data():
    return json.loads(Path(__file__).with_name("parser-cells.json").read_bytes())


def token_expr(node, fields):
    if isinstance(node, (ast.Set, ast.Tuple, ast.List)):
        return all(token_expr(n, fields) for n in node.elts)
    if isinstance(node, ast.BinOp):
        return (
            isinstance(node.op, (ast.BitOr, ast.BitAnd, ast.BitXor, ast.Sub))
            and token_expr(node.left, fields)
            and token_expr(node.right, fields)
        )
    if isinstance(node, ast.Name):
        return node.id in fields
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == "TokenType":
            return True
        v = node.value
        return (
            node.attr in fields
            and isinstance(v, ast.Attribute)
            and v.attr == "Parser"
            and isinstance(v.value, ast.Name)
            and v.value.id == "parser"
        )
    return False


def plan(sel, changes, apply, equal):
    spec = data()
    fields = set(spec["fields"])
    touched = []
    for change in changes:
        path = change["old"]
        if path != change["new"] or path not in spec["permitted"]:
            return None
        name, allowed = spec["permitted"][path]
        source, root = sel.source(path)
        updated = apply(source, change)[0]

        new = paired_tree(change, root, source, updated)
        if len(root.body) != len(new.body):
            return None
        for a, b in zip(root.body, new.body):
            if equal(a, b):
                continue
            if not (
                isinstance(a, ast.ClassDef)
                and isinstance(b, ast.ClassDef)
                and a.name == name
                and module_footprints._same_header(a, b, equal)
            ):
                return None

            def split(node):
                defs, rest = {}, []
                for s in node.body:
                    if (
                        isinstance(s, ast.Assign)
                        and len(s.targets) == 1
                        and isinstance(s.targets[0], ast.Name)
                        and s.targets[0].id in allowed
                        and token_expr(s.value, fields)
                    ):
                        key = s.targets[0].id
                        if key in defs:
                            raise ValueError("Reassigned parser field")
                        defs[key] = s.value
                    else:
                        rest.append(s)
                return defs, rest

            left, x = split(a)
            right, y = split(b)
            if not equal(x, y):
                return None
            changed = {
                k
                for k in left.keys() | right.keys()
                if not equal(left.get(k), right.get(k))
            }
            if not changed:
                return None
            # Remaining initializers can propagate a change only through other modeled token sets.
            for stmt in y:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                reads = {
                    n.id
                    for n in ast.walk(stmt)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                }
                if reads & changed:
                    return None
            touched.append((path, name))
    return touched or None


def snapshot(paths, fields):
    from sqlglot.dialects import Dialect, Dialects
    from sqlglot.parser import Parser
    from sqlglot.tokens import TokenType

    for name in Dialects:
        _ = Dialect.get_or_raise(name).parser_class
    owners = [
        getattr(importlib.import_module(p[:-3].replace("/", ".")), n) for p, n in paths
    ]
    classes, todo = set(), [Parser]
    while todo:
        cls = todo.pop()
        if cls in classes:
            continue
        classes.add(cls)
        todo.extend(cls.__subclasses__())
    result = {}
    for cls in classes:
        if not any(issubclass(cls, c) for c in owners):
            continue
        item = {}
        for field in fields:
            v = getattr(cls, field, None)
            if type(v) in (set, frozenset) and all(type(n) is TokenType for n in v):
                item[field] = sorted(n.name for n in v)
        result[cls.__module__ + "." + cls.__qualname__] = item
    return result


def probe(payload):
    root, touched, fields, out = payload
    os.chdir(root)
    sys.path.insert(0, str(root))
    out.write_text(json.dumps(snapshot(touched, fields)))


def verify(sel, changes, selected, apply, equal, make_checkout, remaining):
    if (
        remaining < 1
        or not selected
        or not all(c["old"].startswith("sqlglot/parsers/") for c in changes)
    ):
        return selected
    touched = plan(sel, changes, apply, equal)
    if not touched:
        return selected
    spec = data()
    with tempfile.TemporaryDirectory(prefix="br-cells-") as d:
        root, out = Path(d) / "repo", Path(d) / "state"
        make_checkout(root, changes, sel)
        env = dict(os.environ, PYTHONPATH=str(root))
        if (
            not forkprobe.run(
                (root, touched, spec["fields"], out), env, probe, min(3, remaining)
            )
            or not out.exists()
        ):
            return selected
        new = json.loads(out.read_bytes())
    expected = set().union(*(spec["descendants"][p] for p, n in touched))
    if set(new) != expected:
        return selected
    observations = spec["readers"]
    indices = set(reflective_use.indices(sel, spec, set(spec["fields"])))
    altered_fields = set()
    changed_receivers = set()
    for name, current in new.items():
        before = spec["values"][name]
        if current.keys() != before.keys():
            return selected
        for k, previous in before.items():
            delta = set(previous) ^ set(current[k])
            if not delta:
                continue
            altered_fields.add(k)
            changed_receivers.add((name, k))
            observed = observations.get(name, {}).get(k, {})
            for key in delta | {"*"}:
                indices.update(spec["groups"][observed[key]] if key in observed else ())
    # Opaque receivers (such as test-created subclasses) retain their readers.
    for field in altered_fields:
        indices.update(spec["opaque"].get(field, ()))
        indices.update(spec["direct"].get(field, ()))
    if -1 in indices:
        return selected
    sel.fixture_parser_fields = changed_receivers
    required = {
        sel.by_name[n] for i in indices if (n := spec["names"][i]) in sel.by_name
    }
    for path, _ in touched:
        if path in sel.files:
            required.update(sel.sets[sel.files[path]])
    return (selected & required) | sel.unknown
