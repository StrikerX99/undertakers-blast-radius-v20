"""Resolve deferred edits using cold class state and pristine method receivers."""

import ast
import importlib
import structural_dependencies
import json
import os
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import forkprobe
import generator_contexts
import reflective_use
import symbol_impact

PREFIXES = ("sqlglot/parsers/", "sqlglot/generators/", "sqlglot/dialects/")


@lru_cache(maxsize=1)
def evidence():
    return json.loads(Path(__file__).with_name("receiver-calls.json").read_bytes())


def receiver_users(sel, classes):
    data = evidence()
    if data["pin"] != sel.mapping["commit"]:
        return None
    groups = {data["custom"], *(data["classes"].get(c, 0) for c in classes)}
    indices = {i for group in groups for i in data["sets"][group]}
    if -1 in indices:
        return None
    if not hasattr(sel, "cone_unmapped"):
        sel.cone_unmapped = {
            sel.by_name[k] for k in sel.by_name.keys() - set(data["tests"])
        }
    return {
        sel.by_name[data["tests"][i]]
        for i in indices
        if data["tests"][i] in sel.by_name
    } | sel.cone_unmapped


BUILTINS = {
    "str",
    "int",
    "bool",
    "float",
    "tuple",
    "list",
    "dict",
    "set",
    "len",
    "max",
    "min",
    "sum",
    "zip",
    "any",
    "all",
    "isinstance",
    "super",
    "enumerate",
    "reversed",
    "sorted",
    "range",
    "ValueError",
    "TypeError",
    "RuntimeError",
    "AssertionError",
    "NotImplementedError",
}


def local_function(fn):
    if (
        fn.decorator_list
        or fn.type_params
        or any(
            n is not None and not isinstance(n, ast.Constant)
            for n in [*fn.args.defaults, *fn.args.kw_defaults]
        )
    ):
        return False
    locals_ = {v.arg for v in ast.walk(fn.args) if isinstance(v, ast.arg)}
    locals_.update(
        v.id
        for v in ast.walk(fn)
        if isinstance(v, ast.Name) and isinstance(v.ctx, ast.Store)
    )
    body = ast.Module(body=fn.body, type_ignores=[])
    fresh = set()
    for node in ast.walk(body):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        call = node.value.func
        if (
            isinstance(call, ast.Attribute)
            and isinstance(call.value, ast.Name)
            and call.value.id == "exp"
            or isinstance(call, ast.Attribute)
            and isinstance(call.value, ast.Call)
            and isinstance(call.value.func, ast.Name)
            and call.value.func.id == "super"
            and call.attr.startswith("_parse_")
        ):
            fresh.update(k.id for k in node.targets if isinstance(k, ast.Name))
    for n in ast.walk(body):
        if (
            isinstance(n, ast.Name)
            and isinstance(n.ctx, ast.Load)
            and n.id not in locals_ | BUILTINS | {"exp"}
        ):
            return False
        if isinstance(
            n,
            (
                ast.Global,
                ast.Nonlocal,
                ast.Import,
                ast.ImportFrom,
                ast.ClassDef,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.Yield,
                ast.Await,
                ast.With,
                ast.AsyncWith,
            ),
        ):
            return False
        if isinstance(n, ast.Attribute) and (
            n.attr.startswith("__") or isinstance(n.ctx, (ast.Store, ast.Del))
        ):
            return False
        if isinstance(n, ast.Subscript) and isinstance(n.ctx, (ast.Store, ast.Del)):
            return False
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        if isinstance(f, ast.Name) and f.id in BUILTINS:
            continue
        if not isinstance(f, ast.Attribute):
            return False
        base = f.value
        if (
            isinstance(base, ast.Name)
            and base.id == "exp"
            and (
                f.attr[0].isupper()
                or f.attr
                in {
                    "var",
                    "to_identifier",
                    "alias_",
                    "column",
                    "cast",
                    "convert",
                    "select",
                    "and_",
                    "or_",
                }
            )
        ):
            continue
        if (
            isinstance(base, ast.Attribute)
            and isinstance(base.value, ast.Name)
            and base.value.id == "exp"
            and base.attr == "Literal"
            and f.attr in {"number", "string"}
        ):
            continue
        if (
            isinstance(base, ast.Call)
            and isinstance(base.func, ast.Name)
            and base.func.id == "super"
            and f.attr == fn.name
        ):
            continue
        if (
            isinstance(base, ast.Name)
            and base.id == "self"
            and (
                f.attr.startswith(("_parse", "_match"))
                or f.attr
                in {
                    "sql",
                    "func",
                    "binary",
                    "unary",
                    "sep",
                    "seg",
                    "expressions",
                    "expression",
                    "_advance",
                    "_retreat",
                    "validate_expression",
                    "raise_error",
                    "unsupported",
                    "normalize_func",
                    "function_fallback_sql",
                }
            )
        ):
            continue
        if isinstance(base, ast.Name) and base.id in fresh and f.attr == "set":
            continue
        while isinstance(base, (ast.Attribute, ast.Subscript)):
            base = base.value
        if (
            isinstance(base, ast.Name)
            and base.id in locals_ - {"self", "cls"}
            and f.attr
            in {
                "get",
                "copy",
                "to_py",
                "sql_name",
                "upper",
                "lower",
                "strip",
                "replace",
                "startswith",
                "endswith",
                "join",
                "expressions",
                "text",
                "eq",
                "is_type",
                "is_int",
            }
        ):
            continue
        return False
    return True


def safe_callback(n, helpers):
    if isinstance(n, ast.Name):
        return n.id in helpers
    if (
        isinstance(n, ast.Lambda)
        and not n.args.defaults
        and not any(n.args.kw_defaults)
    ):
        fn = ast.FunctionDef(
            name="_callback",
            args=n.args,
            body=[ast.Return(value=n.body)],
            decorator_list=[],
            type_params=[],
        )
        return local_function(fn)
    return False


def table_delta(a, b, helpers, equal):
    if not (
        isinstance(a, ast.Assign)
        and isinstance(b, ast.Assign)
        and equal(a.targets, b.targets)
        and len(a.targets) == 1
        and isinstance(a.targets[0], ast.Name)
        and a.targets[0].id == "TRANSFORMS"
        and isinstance(a.value, ast.Dict)
        and isinstance(b.value, ast.Dict)
    ):
        return None

    def entries(node):
        row = {}
        order = []
        for i, (k, v) in enumerate(zip(node.keys, node.values)):
            key = (
                k.attr
                if isinstance(k, ast.Attribute)
                and isinstance(k.value, ast.Name)
                and k.value.id == "exp"
                else None
            )
            if key is None:
                key = "!" + ast.dump(k if k is not None else v)
            if key in row:
                duplicate = "!duplicate." + key
                while duplicate in row:
                    duplicate = "!" + duplicate
                key = duplicate
            row[key] = v
            order.append(key)
        return row, order

    left, lo = entries(a.value)
    right, ro = entries(b.value)
    if left is None or right is None:
        return None
    common = left.keys() & right.keys()
    if [k for k in lo if k in common] != [k for k in ro if k in common]:
        return None
    changed = {
        k for k in left.keys() | right.keys() if not equal(left.get(k), right.get(k))
    }
    if not changed or any(
        k.startswith("!")
        or any(
            not safe_callback(v, helpers)
            for v in (left.get(k), right.get(k))
            if v is not None
        )
        for k in changed
    ):
        return None
    return changed


def edits(path, before, after, equal):
    if not any(
        isinstance(v, ast.ImportFrom)
        and v.module == "__future__"
        and any(i.name == "annotations" for i in v.names)
        for v in before.body
    ):
        return None

    def named(body, kind):
        group = {v.name: v for v in body if isinstance(v, kind)}
        if len(group) != sum(isinstance(v, kind) for v in body):
            raise ValueError("ambiguous declaration")
        return group

    def delta(a, b):
        return {k for k in a.keys() | b.keys() if not equal(a.get(k), b.get(k))}

    fa, fb = named(before.body, ast.FunctionDef), named(after.body, ast.FunctionDef)
    helpers = delta(fa, fb)
    if any(
        not k.startswith("_")
        or k.startswith("__")
        or any(not local_function(n) for n in (fa.get(k), fb.get(k)) if n is not None)
        for k in helpers
    ):
        return None
    ca, cb = named(before.body, ast.ClassDef), named(after.body, ast.ClassDef)
    changed = delta(ca, cb)
    if ca.keys() != cb.keys() or not changed:
        return None
    # Unchanged names keep their original position relative to other declarations
    # and executable module statements. Only newly added helpers can disappear
    # from this ordering check.
    if helpers & fa.keys():
        return None

    def outline(body, classes, methods, inserted):
        return [
            ("class", n.name)
            if isinstance(n, ast.ClassDef) and n.name in classes
            else ("method", n.name)
            if isinstance(n, ast.FunctionDef) and n.name in methods
            else ast.dump(n)
            for n in body
            if not (isinstance(n, ast.FunctionDef) and n.name in inserted)
        ]

    if outline(before.body, changed, set(), set()) != outline(
        after.body, changed, set(), helpers
    ):
        return None
    result = []
    for owner in sorted(changed):
        a, b = ca[owner], cb[owner]
        if not generator_contexts.header(a, b, equal):
            return None
        mm, nn = named(a.body, ast.FunctionDef), named(b.body, ast.FunctionDef)
        methods = delta(mm, nn)
        # Class creation executes its body. Refuse local names that can hide an
        # assignment, import, nested class, or dynamically accessed class namespace.
        for cls in (a, b):
            for stmt in cls.body:
                if isinstance(stmt, ast.FunctionDef):
                    continue
                for n in ast.walk(stmt):
                    if (
                        isinstance(n, ast.Name)
                        and isinstance(n.ctx, (ast.Store, ast.Del))
                        and n.id in methods
                        or isinstance(n, ast.alias)
                        and (n.asname or n.name) in methods
                        or isinstance(n, ast.ClassDef)
                        and n.name in methods
                        or isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Name)
                        and n.func.id in {"globals", "locals", "vars", "exec", "eval"}
                    ):
                        return None
        for k in methods:
            old, new = mm.get(k), nn.get(k)
            if (
                not k.islower()
                or k.startswith("__")
                or any(not local_function(v) for v in (old, new) if v is not None)
            ):
                return None
            result.append((path, owner, "method", k))

        def rest(body):
            return [
                n
                for n in body
                if not isinstance(n, ast.FunctionDef) or n.name not in methods
            ]

        left, right = rest(a.body), rest(b.body)
        if len(left) != len(right):
            return None

        def ordering(body):
            return [
                ("method", n.name) if isinstance(n, ast.FunctionDef) else ("statement",)
                for n in body
                if not (
                    isinstance(n, ast.FunctionDef) and n.name in mm.keys() ^ nn.keys()
                )
            ]

        if ordering(a.body) != ordering(b.body):
            return None
        for x, y in zip(left, right):
            if equal(x, y):
                continue
            keys = table_delta(x, y, helpers, equal)
            if not keys:
                return None
            result.extend((path, owner, "table", k) for k in sorted(keys))
    # These module helpers must be new names. A modified shared function or a
    # colliding assignment can affect unrelated clients outside class dispatch.
    if helpers:
        if helpers & fa.keys() or not any(kind == "table" for _, _, kind, _ in result):
            return None
        if any(
            isinstance(n, ast.Name)
            and isinstance(n.ctx, (ast.Del, ast.Store))
            and n.id in helpers
            or isinstance(n, ast.alias)
            and (n.name == "*" or (n.asname or n.name) in helpers)
            for n in ast.walk(before)
        ):
            return None
    return result, helpers


def snapshot(payload):
    import inspect

    absent = object()
    root, requests, out = payload
    os.chdir(root)
    sys.path.insert(0, str(root))
    from registry_cache import concrete_receivers
    from sqlglot import exp
    from sqlglot.generator import _build_dispatch
    import sqlglot

    if Path(sqlglot.__file__).resolve() != (root / "sqlglot/__init__.py").resolve():
        return
    types = concrete_receivers(root)
    result = []
    for path, owner, kind, attr in requests:
        host = getattr(importlib.import_module(path[:-3].replace("/", ".")), owner)
        if not isinstance(host, type) or not any(issubclass(c, host) for c in types):
            return
        rows = {}
        if kind == "table":
            target = getattr(exp, attr)
            # Follow materialized tables too, including copies outside host.__subclasses__().
            for cls in {
                c
                for c in types
                if hasattr(c, "TRANSFORMS")
                and any(t.__name__ == "Generator" for t in c.__mro__)
            }:
                fun = _build_dispatch(cls).get(target)
                rows[cls.__module__ + "." + cls.__qualname__] = (
                    [fun.__module__, fun.__qualname__]
                    if callable(fun) and hasattr(fun, "__qualname__")
                    else None
                )
        else:
            for cls in types:
                if not issubclass(cls, host):
                    continue
                raw = inspect.getattr_static(cls, attr, absent)
                if raw is not absent and not inspect.isfunction(raw):
                    return
                value = getattr(cls, attr, None)
                if (raw is absent and value is not None) or (
                    raw is not absent and value is not raw
                ):
                    return
                rows[cls.__module__ + "." + cls.__qualname__] = (
                    [value.__module__, value.__qualname__]
                    if callable(value) and hasattr(value, "__qualname__")
                    else None
                )
        result.append(rows)
    out.write_text(json.dumps(result))


def read_state(root, changes):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        forkprobe.run((root, changes, out), {}, snapshot, 2.8)
        return json.loads(out.read_bytes()) if out.exists() else None


@lru_cache(maxsize=32)
def baseline(root, changes):
    try:
        from registry_cache import lookup

        cached = lookup(root, changes)
    except (OSError, ValueError, TypeError):
        cached = None
    return read_state(root, changes) if cached is None else cached


def refine(sel, changes, selected, apply, equal, checkout, root, remaining):
    if (
        len(selected) * 2 < len(sel.all)
        or remaining < 6
        or not changes
        or len(changes) > 3
    ):
        return selected
    requests = []
    helpers = set()
    users = set(sel.confirmed_failures)
    for c in changes:
        path = c["old"]
        if (
            path != c["new"]
            or not path.startswith(PREFIXES)
            or path.endswith(("/__init__.py", "/dialect.py"))
        ):
            return selected
        txt, tree = sel.source(path)
        dest = apply(txt, c)[0]
        if c.get("_compiled_source") != dest:
            return selected
        changeset = edits(path, tree, ast.parse(dest), equal)
        if not changeset:
            return selected
        items, funcs = changeset
        requests.extend(items)
        helpers.update(funcs)
        if path in sel.files:
            users.update(sel.sets[sel.files[path]])
        ext = sel.dispatch_map()
        users.update(ext.external | ext.external_by_source.get(path, set()))
    if not requests:
        return selected
    requests = tuple(requests)
    old = baseline(root, requests)
    if not old:
        return selected
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "src"
        checkout(dest, changes, sel)
        new = read_state(dest, requests)
    if not new:
        return selected
    # Whole-registry, explicit source inspection, and dynamically named test-side readers.
    spec = generator_contexts.data("parser-cells.json")
    indices = reflective_use.indices(sel, spec, {r[3] for r in requests} | helpers)
    users.update(
        sel.by_name[spec["names"][i]]
        for i in indices
        if spec["names"][i] in sel.by_name
    )
    direct = symbol_impact.index()
    for helper in helpers:
        for reader, line, scope, kind in direct.get(helper, ()):
            # Package-level helpers and instance methods have distinct namespaces.
            if kind == "attr" and scope and reader.startswith(PREFIXES):
                _, tree = sel.source(reader)
                fn = next(
                    (
                        n
                        for cls in ast.walk(tree)
                        if isinstance(cls, ast.ClassDef)
                        for n in cls.body
                        if isinstance(n, ast.FunctionDef) and n.lineno == scope
                    ),
                    None,
                )
                if fn is None or not fn.args.args:
                    return selected
                receiver = fn.args.args[0].arg
                if receiver not in {"self", "cls"} or any(
                    isinstance(v, ast.Name)
                    and isinstance(v.ctx, (ast.Store, ast.Del))
                    and v.id == receiver
                    for v in ast.walk(fn)
                ):
                    return selected
                hits = [
                    v
                    for v in ast.walk(fn)
                    if isinstance(v, ast.Attribute)
                    and v.attr == helper
                    and v.lineno == line
                ]
                if hits and all(
                    isinstance(v.value, ast.Name) and v.value.id == receiver
                    for v in hits
                ):
                    continue
            return selected
    receiver_classes = set()
    for (path, owner, kind, name), a, b in zip(requests, old, new, strict=True):
        if a.keys() != b.keys():
            return selected
        module = path[:-3].replace("/", ".")
        if kind == "table":
            owners = {
                c
                for c in a
                if a[c] != b[c] or any(x and x[0] == module for x in (a[c], b[c]))
            }
            # Materialized callback tables also have direct, non-rendering readers.
            rows = generator_contexts.data_dispatch()
            for dotted in owners:
                cls = dotted.rsplit(".", 1)
                users.update(
                    structural_dependencies.table_observers(
                        sel, rows, path, ":".join(cls), {name}
                    )
                )
        else:
            target = [module, owner + "." + name]
            owners = {c for c in a if a[c] != b[c] or target in (a[c], b[c])}
        # A class cohort also covers calls that failed in signature binding and lookups
        # that previously resolved to no implementation. Unknown concrete test subclasses
        # are represented explicitly by receiver_users.
        receiver_classes.update(owners)
        more = receiver_users(sel, owners)
        if more is None:
            return selected
        users.update(more)
        for rp, line, scope, access in direct.get(name, ()):
            if rp.startswith("tests/") and access in {"attr", "string", "import"}:
                users.update(t["number"] for t in sel.tests if t["file"] == rp)
    proposed = (users & selected) | sel.confirmed_failures
    if proposed != selected:
        sel.cone_candidates = frozenset(proposed)
        sel.cone_owners = frozenset(receiver_classes)
    return proposed
