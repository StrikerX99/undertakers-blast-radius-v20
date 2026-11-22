"""Resolve bulk edits to a generator's transform map by comparing its actual entries."""

import ast
import reflective_use
import hashlib
import importlib
import json
import marshal
import os
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import forkprobe
from shared_tables import encode
from source_fast import paired_tree
from structural_dependencies import observations, table_observers


def expression(node):
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "exp"
    )


def base_table(node):
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "TRANSFORMS"
        and (
            isinstance(node.value, ast.Name)
            or isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
        )
    )


def pure_expansion(node):
    if base_table(node):
        return True
    if not isinstance(node, ast.DictComp) or len(node.generators) != 1:
        return False
    comp = node.generators[0]
    if (
        comp.is_async
        or not isinstance(comp.target, ast.Tuple)
        or len(comp.target.elts) != 2
    ):
        return False
    key, value = comp.target.elts
    if not all(isinstance(n, ast.Name) for n in (key, value, node.key, node.value)) or (
        key.id,
        value.id,
    ) != (node.key.id, node.value.id):
        return False
    source = comp.iter
    if not (
        isinstance(source, ast.Call)
        and not source.args
        and not source.keywords
        and isinstance(source.func, ast.Attribute)
        and source.func.attr == "items"
        and base_table(source.func.value)
    ):
        return False
    for check in comp.ifs:
        if not (
            isinstance(check, ast.Compare)
            and len(check.ops) == 1
            and isinstance(check.ops[0], (ast.In, ast.NotIn))
            and isinstance(check.left, ast.Name)
            and check.left.id == key.id
            and isinstance(check.comparators[0], (ast.Tuple, ast.Set, ast.List))
            and all(expression(n) for n in check.comparators[0].elts)
        ):
            return False
    return True


def expansions_only(a, b, equal):
    if not isinstance(a, ast.Dict) or not isinstance(b, ast.Dict):
        return False
    # Changed expansions must be side-effect-free. Fixed entries, including any
    # constructor calls, must appear in the same order relative to the expansions.
    if len(a.keys) != len(b.keys):
        return False
    any_change = False
    for k1, v1, k2, v2 in zip(a.keys, a.values, b.keys, b.values):
        if not equal(k1, k2):
            return False
        if equal(v1, v2):
            continue
        if k1 is not None or not (pure_expansion(v1) and pure_expansion(v2)):
            return False
        any_change = True
    return any_change


def owner(sel, change, apply, equal):
    path = change["old"]
    if (
        path != change["new"]
        or not path.startswith("sqlglot/generators/")
        or not path.endswith(".py")
    ):
        return None
    raw, old = sel.source(path)
    final = apply(raw, change)[0]
    new = paired_tree(change, old, raw, final)
    if len(old.body) != len(new.body):
        return None
    classes = []
    for a, b in zip(old.body, new.body):
        if equal(a, b):
            continue
        if not (
            isinstance(a, ast.ClassDef)
            and isinstance(b, ast.ClassDef)
            and all(
                equal(getattr(a, k), getattr(b, k)) for k in a._fields if k != "body"
            )
            and len(a.body) == len(b.body)
        ):
            return None
        changed = []
        for x, y in zip(a.body, b.body):
            if equal(x, y):
                continue
            if not (
                type(x) is type(y)
                and isinstance(x, (ast.AnnAssign, ast.Assign))
                and all(
                    equal(getattr(x, k), getattr(y, k))
                    for k in x._fields
                    if k != "value"
                )
            ):
                return None
            fields = x.targets if isinstance(x, ast.Assign) else [x.target]
            if not (
                len(fields) == 1
                and isinstance(fields[0], ast.Name)
                and fields[0].id == "TRANSFORMS"
                and expansions_only(x.value, y.value, equal)
            ):
                return None
            changed.append(x)
        if len(changed) != 1:
            return None
        # A later class initializer could derive separate state from the table.
        for stmt in a.body[a.body.index(changed[0]) + 1 :]:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                early = [stmt.args, *stmt.decorator_list]
            else:
                early = [stmt]
            if any(
                isinstance(n, (ast.Name, ast.Attribute))
                and getattr(n, "id", getattr(n, "attr", "")) == "TRANSFORMS"
                for e in early
                for n in ast.walk(e)
            ):
                return None
        classes.append(a.name)
    return [(path, k) for k in classes] or None


def digest(root, owners):
    os.chdir(root)
    sys.path.insert(0, str(root))
    import sqlglot
    from sqlglot import Generator

    if Path(sqlglot.__file__).resolve() != (root / "sqlglot/__init__.py").resolve():
        raise ValueError("Unexpected source origin")
    signatures = {}
    result = {}
    for path, clsname in owners:
        mod = importlib.import_module(path[:-3].replace("/", "."))
        cls = getattr(mod, clsname)
        if not isinstance(cls, type) or not issubclass(cls, Generator):
            raise ValueError(clsname)
        row = {}
        for expr, fn in cls.TRANSFORMS.items():
            if not isinstance(expr, type) or expr.__name__ in row:
                raise ValueError("Ambiguous dispatch key")
            if fn not in signatures:
                try:
                    key = encode(fn)
                    signatures[fn] = hashlib.sha256(
                        json.dumps(key, separators=(",", ":")).encode()
                    ).digest()
                except (TypeError, ValueError, RecursionError):
                    signatures[fn] = None
            row[expr.__name__] = signatures[fn]
        result[mod.__name__ + ":" + cls.__qualname__] = row
    return result


def build(root, out):
    owners = []
    for path in sorted((root / "sqlglot/generators").glob("*.py")):
        for cls in ast.parse(path.read_bytes()).body:
            if isinstance(cls, ast.ClassDef) and any(
                isinstance(n, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id == "TRANSFORMS" for t in n.targets
                )
                for n in cls.body
            ):
                owners.append((str(path.relative_to(root)), cls.name))
    out.write_bytes(marshal.dumps(digest(root, owners)))


def probe(payload):
    root, owners, out = payload
    out.write_bytes(marshal.dumps(digest(root, owners)))


@lru_cache(maxsize=1)
def baseline():
    return marshal.loads(Path(__file__).with_name("dispatch-state.bin").read_bytes())


def refine(sel, changes, selected, apply, equal, checkout, remaining):
    if remaining < 1 or len(selected) < 70 or len(changes) != 1:
        return selected
    path = changes[0]["old"]
    if (
        not path.startswith("sqlglot/generators/")
        or "TRANSFORMS" not in sel.source(path)[0]
    ):
        return selected
    owners = owner(sel, changes[0], apply, equal)
    if not owners:
        return selected
    with tempfile.TemporaryDirectory(prefix="br-dispatch-") as tmp:
        temp = Path(tmp)
        root = temp / "repo"
        out = temp / "delta"
        checkout(root, changes, sel)
        forkprobe.run(
            (root, owners, out), {"PYTHONPATH": str(root)}, probe, min(4, remaining)
        )
        if not out.exists():
            return selected
        patched = marshal.loads(out.read_bytes())
    data = observations("generator-dispatch-reads.json")
    readers = set(sel.unknown)
    from parser_cells import data as parser_data

    direct = parser_data()
    readers.update(
        sel.by_name[t]
        for i in reflective_use.indices(sel, direct, {"TRANSFORMS"})
        if (t := direct["names"][i]) in sel.by_name
    )
    if path in sel.files:
        readers.update(sel.sets[sel.files[path]])
    for name, row in patched.items():
        if name not in baseline() or name not in data["rows"]:
            return selected
        previous = baseline()[name]
        keys = {
            k
            for k in previous.keys() | row.keys()
            if k not in previous
            or k not in row
            or previous[k] is None
            or row[k] is None
            or previous[k] != row[k]
        }
        if keys:
            readers.update(table_observers(sel, data, path, name, keys))
    return selected & readers


if __name__ == "__main__":
    build(Path(sys.argv[1]), Path(sys.argv[2]))
