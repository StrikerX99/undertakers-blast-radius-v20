"""Locate edits to literal configuration cells and keep their observed consumers."""

import ast
import reflective_use
import importlib
import json
import os
import sys
import tempfile
import zlib
from functools import lru_cache
from pathlib import Path

from source_fast import paired_tree
import forkprobe


@lru_cache(maxsize=1)
def catalog():
    return json.loads(
        zlib.decompress(Path(__file__).with_name("settings.dat").read_bytes())
    )


def declarations(root):
    return {
        (c.name, t.id): s.value
        for c in root.body
        if isinstance(c, ast.ClassDef)
        for s in c.body
        if isinstance(s, (ast.Assign, ast.AnnAssign))
        for t in ([s.target] if isinstance(s, ast.AnnAssign) else s.targets)
        if isinstance(t, ast.Name) and t.id.isupper()
    }


def key_name(expr):
    if isinstance(expr, ast.Constant) and type(expr.value) in (
        str,
        int,
        bool,
        type(None),
    ):
        return json.dumps(expr.value, ensure_ascii=True)
    return None


def difference(old, new, equal):
    if (
        isinstance(old, ast.Constant)
        and isinstance(new, ast.Constant)
        and type(old.value) is type(new.value)
        and type(old.value) in (str, bool, int, float)
    ):
        return {"*"}
    if (
        not isinstance(old, ast.Dict)
        or not isinstance(new, ast.Dict)
        or not equal(old.keys, new.keys)
    ):
        return None
    keys = set()
    for k, a, b in zip(old.keys, old.values, new.values):
        if equal(a, b):
            continue
        if (
            not isinstance(a, ast.Constant)
            or not isinstance(b, ast.Constant)
            or type(a.value) is not type(b.value)
            or type(a.value) not in (str, int, bool, float, type(None))
        ):
            return None
        # Resolve qualified enum/class keys against the pinned source in the child.
        key = key_name(k)
        if key is None:
            if (
                not isinstance(k, ast.Attribute)
                or not isinstance(k.value, ast.Name)
                or k.value.id not in {"exp", "TokenType"}
            ):
                return None
            key = "@" + k.value.id + "." + k.attr
        keys.add(key)
    return keys or None


def changes_for(sel, chs, apply, equal):
    table = catalog()
    result = []
    for change in chs:
        path = change["old"]
        if path != change["new"] or path not in table["paths"]:
            return None
        raw, old = sel.source(path)
        updated = apply(raw, change)[0]
        new = paired_tree(change, old, raw, updated)
        if len(old.body) != len(new.body):
            return None
        for left, right in zip(old.body, new.body):
            if equal(left, right):
                continue
            if (
                not isinstance(left, ast.ClassDef)
                or not isinstance(right, ast.ClassDef)
                or len(left.body) != len(right.body)
                or not all(
                    equal(getattr(left, k), getattr(right, k))
                    for k in left._fields
                    if k != "body"
                )
            ):
                return None
            cls = left.name
            for a, b in zip(left.body, right.body):
                if equal(a, b):
                    continue
                if (
                    not isinstance(a, (ast.Assign, ast.AnnAssign))
                    or type(a) is not type(b)
                    or not all(
                        equal(getattr(a, k), getattr(b, k))
                        for k in a._fields
                        if k != "value"
                    )
                ):
                    return None
                targets = a.targets if isinstance(a, ast.Assign) else [a.target]
                if len(targets) != 1 or not isinstance(targets[0], ast.Name):
                    return None
                field = targets[0].id
                desc = table["paths"][path].get(cls, {}).get(field)
                if not desc:
                    return None
                cells = difference(a.value, b.value, equal)
                if not cells:
                    return None
                result.append((path, cls, field, sorted(cells), desc))
    return result or None


def import_check(payload):
    root, rows, dest = payload
    os.chdir(root)
    sys.path.insert(0, str(root))
    from sqlglot.dialects import Dialect

    result = []
    for path, cls, field, keys, desc in rows:
        stem = path.rsplit("/", 1)[-1][:-3]
        Dialect.get_or_raise(stem)
        module = importlib.import_module(path[:-3].replace("/", "."))
        target = getattr(module, cls)
        current = getattr(target, field)
        if type(current) not in (int, str, bool, float, dict):
            raise ValueError("Unmodeled effective field")
        actual = []
        for key in keys:
            if key.startswith("@"):
                nm, attr = key[1:].split(".")
                token = getattr(getattr(module, nm), attr)
                if isinstance(token, type):
                    key = token.__module__ + "." + token.__qualname__
                else:
                    key = (
                        type(token).__module__
                        + "."
                        + type(token).__qualname__
                        + "."
                        + token.name
                    )
            actual.append(key)
        result.append(actual)
    dest.write_text(json.dumps(result))


def verify(sel, changes, selected, apply, equal, checkout, remaining):
    if (
        remaining < 1
        or not selected
        or not all(
            c["old"].startswith(
                ("sqlglot/dialects/", "sqlglot/generators/", "sqlglot/parsers/")
            )
            for c in changes
        )
    ):
        return selected
    # Eliminate non-declarative hunks without loading the evidence index.
    for ch in changes:
        edits = [
            line[1:].lstrip()
            for h in ch["hunks"]
            for line in h["lines"]
            if line[0] in "+-" and line[1:].strip()
        ]
        if not edits or not any(
            e[:1].isupper() or e.startswith(('"', "'", "exp.", "TokenType."))
            for e in edits
        ):
            return selected
    table = catalog()
    rows = changes_for(sel, changes, apply, equal)
    if not rows:
        return selected
    with tempfile.TemporaryDirectory(prefix="br-setting-") as d:
        root, out = Path(d) / "repo", Path(d) / "pass"
        checkout(root, changes, sel)
        if (
            not forkprobe.run(
                (root, rows, out),
                {"PYTHONPATH": str(root)},
                import_check,
                min(3, remaining),
            )
            or not out.exists()
        ):
            return selected
        resolved = json.loads(out.read_bytes())
    keep = set(reflective_use.indices(sel, table, {row[2] for row in rows}))
    for (path, owner, field, _, desc), keys in zip(rows, resolved, strict=True):
        keep.update(table["opaque"].get(field, ()))
        keep.update(table["direct"].get(field, ()))
        for name in desc:
            readers = table["readers"].get(name, {}).get(field, {})
            groups = (
                readers.values()
                if "*" in keys
                else [readers[k] for k in set(keys) | {"*"} if k in readers]
            )
            for group in groups:
                keep.update(table["groups"][group])
        if -1 in keep:
            return selected
    if len(changes) == 1:
        changes[0]["_verified_fixture_settings"] = rows
    numbers = {sel.by_name[n] for i in keep if (n := table["names"][i]) in sel.by_name}
    for path, _, _, _, _ in rows:
        if path in sel.files:
            numbers.update(sel.sets[sel.files[path]])
    return selected & numbers | sel.unknown
