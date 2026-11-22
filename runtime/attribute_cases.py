"""Use cold receiver observations and warm line coverage to verify unused settings."""

import ast
import reflective_use
import json
from functools import lru_cache
from pathlib import Path

import field_cases
import parser_cells
import rule_replay
import symbol_impact
from source_fast import paired_tree

HERE = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def observed():
    return json.loads((HERE / "attribute-reads.json").read_bytes())


def changed(sel, change, apply, equal, allowed):
    path = change["old"]
    if path == field_cases.DIALECT:
        fields = field_cases.changed_fields(
            sel, change, apply, equal, rule_replay.ledger()
        )
        return ("sqlglot.dialects.dialect.Dialect", fields) if fields else None
    if not path.startswith("sqlglot/dialects/") or path.endswith("/__init__.py"):
        return None
    raw, old = sel.source(path)
    updated = apply(raw, change)[0]
    new = paired_tree(change, old, raw, updated)
    if len(old.body) != len(new.body):
        return None
    pairs = [(a, b) for a, b in zip(old.body, new.body) if not equal(a, b)]
    if len(pairs) != 1:
        return None
    a, b = pairs[0]
    if (
        not isinstance(a, ast.ClassDef)
        or type(a) is not type(b)
        or a.decorator_list
        or a.keywords
        or not a.bases
        or len(a.body) != len(b.body)
        or any(
            not equal(getattr(a, k), getattr(b, k)) for k in a._fields if k != "body"
        )
    ):
        return None
    keys = set()
    for x, y in zip(a.body, b.body):
        if equal(x, y):
            continue
        if (
            not isinstance(x, (ast.Assign, ast.AnnAssign))
            or type(x) is not type(y)
            or any(
                not equal(getattr(x, k), getattr(y, k))
                for k in x._fields
                if k != "value"
            )
            or not isinstance(x.value, ast.Constant)
            or not isinstance(y.value, ast.Constant)
            or type(x.value.value) is not type(y.value.value)
            or type(x.value.value) not in (bool, str, float, int)
        ):
            return None
        targets = x.targets if isinstance(x, ast.Assign) else [x.target]
        if (
            len(targets) != 1
            or not isinstance(targets[0], ast.Name)
            or targets[0].id not in allowed
        ):
            return None
        keys.add(targets[0].id)
    return (path[:-3].replace("/", ".") + "." + a.name, keys) if keys else None


def warm_readers(sel, fields):
    readers = set()
    points = {}
    for name in fields:
        for path, line, scope, kind in symbol_impact.index().get(name, ()):
            if path.startswith("tests/"):
                readers.update(t["number"] for t in sel.tests if t["file"] == path)
            elif kind != "attr" or not scope:
                return None
            else:
                points.setdefault(path, set()).add(line)
    for path, lines in points.items():
        sel.load_shard(path)
        readers.update(sel.line_tests(path, lines))
    return readers


def refine(sel, changes, selected, apply, equal):
    if not selected or sel.unknown or len(changes) != 1:
        return selected
    c = changes[0]
    path = c["old"]
    if (
        path != c["new"]
        or not path.startswith("sqlglot/dialects/")
        or not path.endswith(".py")
    ):
        return selected
    data = observed()
    if data["pin"] != sel.mapping["commit"]:
        return selected
    if not any(sel.by_name.get(name) in selected for name in data["cold"]):
        return selected
    edit = changed(sel, c, apply, equal, data["fields"])
    if not edit:
        return selected
    owner, fields = edit
    if owner not in data["owners"]:
        return selected
    warm = warm_readers(sel, fields)
    if warm is None:
        return selected
    protected = set(sel.confirmed_failures)
    if path in sel.files:
        protected.update(sel.sets[sel.files[path]])
    ext = sel.dispatch_map()
    protected.update(ext.external | ext.external_by_source.get(path, set()))
    spec = parser_cells.data()
    protected.update(
        sel.by_name[n]
        for i in reflective_use.indices(sel, spec, fields)
        if (n := spec["names"][i]) in sel.by_name
    )
    missing = selected - protected - warm
    return selected - {
        n
        for name, reads in data["cold"].items()
        if (n := sel.by_name.get(name)) in missing
        and fields.isdisjoint(reads.get(owner, ()))
        and fields.isdisjoint(reads.get("*", ()))
    }
