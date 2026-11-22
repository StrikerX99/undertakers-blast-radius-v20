"""Resolve class-method edits against SQLGlot's actual inherited callables."""

import ast
from source_fast import analysis_tree
import copy
import inspect
import json
import os
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import forkprobe
import general_dialects
import generator_contexts
import inherited_methods
import module_footprints
import structural_dependencies
import symbol_impact


def ordinary(fn):
    if not isinstance(fn, ast.FunctionDef) or fn.name.startswith("__"):
        return False
    if any(
        isinstance(n, (ast.Attribute, ast.Subscript))
        and isinstance(n.ctx, (ast.Del, ast.Store))
        for n in ast.walk(fn)
    ):
        return False

    parser = fn.name.startswith("_parse_")
    for parent in ast.walk(fn):
        for child in ast.iter_child_nodes(parent):
            if not isinstance(child, ast.Name) or child.id != "self":
                continue
            if not isinstance(parent, ast.Attribute) or parent.value is not child:
                return False
            name = parent.attr
            if parser:
                if not (
                    name.startswith(("_parse_", "_match"))
                    or name
                    in {
                        "_advance",
                        "_retreat",
                        "_prev",
                        "_curr",
                        "_next",
                        "expression",
                        "raise_error",
                        "validate_expression",
                    }
                    or name.endswith("_TOKENS")
                    and name.isupper()
                ):
                    return False
            elif name not in {
                "sql",
                "func",
                "sep",
                "seg",
                "expressions",
                "binary",
                "unary",
                "unsupported",
                "normalize_func",
                "function_fallback_sql",
            }:
                return False

    class Builders(ast.NodeTransformer):
        def visit_Call(self, call):
            self.generic_visit(call)
            f = call.func
            if (
                isinstance(f, ast.Attribute)
                and isinstance(f.value, ast.Name)
                and f.value.id == "exp"
                and f.attr
                in {
                    "cast",
                    "case",
                    "var",
                    "column",
                    "convert",
                    "alias_",
                    "select",
                    "and_",
                    "or_",
                    "to_identifier",
                }
            ):
                call.func = ast.Name(id="tuple", ctx=ast.Load())
            elif (
                isinstance(f, ast.Attribute)
                and isinstance(f.value, ast.Call)
                and isinstance(f.value.func, ast.Name)
                and f.value.func.id == "super"
                and not f.value.args
                and not f.value.keywords
                and f.attr == fn.name
            ):
                call.func = ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()),
                    attr=f.attr,
                    ctx=ast.Load(),
                )
            elif (
                isinstance(f, ast.Attribute)
                and isinstance(f.value, ast.Call)
                and f.attr in {"eq", "when", "else_", "copy"}
            ):
                call.func = ast.Name(id="tuple", ctx=ast.Load())
                call.args.insert(0, f.value)
            return call

    return general_dialects._ordinary(Builders().visit(copy.deepcopy(fn)))


def edits(sel, changes, apply, equal):
    result = []
    for change in changes:
        path = change["old"]
        if (
            path != change["new"]
            or not (
                path == "sqlglot/parser.py"
                or path.startswith(("sqlglot/parsers/", "sqlglot/generators/"))
            )
            or not path.endswith(".py")
        ):
            return None
        raw, a = sel.source(path)
        b = analysis_tree(change, a, raw, apply(raw, change)[0])
        family = "Generator" if "/generators/" in path else "Parser"
        if not any(
            isinstance(n, ast.ImportFrom)
            and n.module == "__future__"
            and any(i.name == "annotations" for i in n.names)
            for n in a.body
        ) or len(a.body) != len(b.body):
            return None
        for x, y in zip(a.body, b.body):
            if equal(x, y):
                continue
            if not (
                isinstance(x, ast.ClassDef)
                and x.name.endswith(family)
                and isinstance(y, ast.ClassDef)
                and generator_contexts.header(x, y, equal)
            ):
                return None

            def split(cls):
                return {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}

            left, right = split(x), split(y)
            changed = {
                k
                for k in left.keys() | right.keys()
                if not equal(left.get(k), right.get(k))
            }
            if not changed or not equal(
                [
                    n
                    for n in x.body
                    if not isinstance(n, ast.FunctionDef) or n.name not in changed
                ],
                [
                    n
                    for n in y.body
                    if not isinstance(n, ast.FunctionDef) or n.name not in changed
                ],
            ):
                return None
            for k in sorted(changed):
                if any(
                    sum(
                        isinstance(n, ast.FunctionDef) and n.name == k for n in cls.body
                    )
                    > 1
                    for cls in (x, y)
                ):
                    return None
                if not (
                    k.endswith("_sql") and not k.startswith("_")
                    if family == "Generator"
                    else k.startswith("_parse_")
                ):
                    return None
                f, g = left.get(k), right.get(k)
                if any(not ordinary(n) for n in (f, g) if n is not None):
                    return None
                if f and g:
                    # Existing direct callers must enter the same body under either signature.
                    if not equal(f.args, g.args):
                        if (
                            f.args.posonlyargs
                            or g.args.posonlyargs
                            or not all(
                                equal(getattr(f.args, v), getattr(g.args, v))
                                for v in (
                                    "kwonlyargs",
                                    "kw_defaults",
                                    "vararg",
                                    "kwarg",
                                )
                            )
                            or len(g.args.args) < len(f.args.args)
                            or not equal(f.args.args, g.args.args[: len(f.args.args)])
                            or len(g.args.args) - len(g.args.defaults)
                            != len(f.args.args) - len(f.args.defaults)
                        ):
                            return None
                result.append((path, x.name, k, family))
    return result or None


def snapshot(payload):
    root, changes, output = payload
    os.chdir(root)
    sys.path.insert(0, str(root))
    import sqlglot

    if Path(sqlglot.__file__).resolve() != (root / "sqlglot/__init__.py").resolve():
        raise ValueError("unexpected SQLGlot source")
    from sqlglot import Dialect, exp
    from sqlglot.dialects import Dialects
    from sqlglot.generator import _build_dispatch
    import importlib

    result = {}

    def location(fn):
        if fn is None:
            return None
        if not inspect.isfunction(fn):
            raise ValueError("opaque method")
        file = Path(inspect.getsourcefile(fn)).resolve().relative_to(root).as_posix()
        try:
            inspect.signature(fn).bind(object())
            accepts_empty = True
        except TypeError:
            accepts_empty = False
        return [file, fn.__qualname__, fn.__code__.co_firstlineno, accepts_empty]

    for path, owner, name, family in changes:
        top = getattr(importlib.import_module(path[:-3].replace("/", ".")), owner)
        classes = {
            getattr(
                Dialect.get_or_raise(d),
                "generator_class" if family == "Generator" else "parser_class",
            )
            for d in Dialects
        }
        classes.add(top)
        for cls in classes:
            if not issubclass(cls, top):
                continue
            binding = location(getattr(cls, name, None))
            route = None
            expr = None
            if family == "Generator":
                expr = exp.EXPR_CLASSES.get(name[:-4])
                route = location(_build_dispatch(cls).get(expr)) if expr else None
                if expr is None:
                    raise ValueError("unknown expression")
            result[
                repr((path, owner, name, cls.__module__ + ":" + cls.__qualname__))
            ] = [binding, route, expr.__name__ if expr else None]
    output.write_text(json.dumps(result))


def inspect_root(root, changes):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "state"
        forkprobe.run((root, changes, out), {}, snapshot, 3.5)
        return json.loads(out.read_bytes()) if out.exists() else None


@lru_cache(maxsize=20)
def pristine(root, changes):
    return inspect_root(root, changes)


def refine(sel, changes, selected, apply, equal, checkout, root, remaining):
    if (
        len(selected) * 2 < len(sel.all)
        or len(changes) > 3
        or sel.unknown
        or remaining < 8
    ):
        return selected
    proposals = edits(sel, changes, apply, equal)
    if not proposals:
        return selected
    # Avoid searching any module twice for a class or an eager reference.
    external = sel.dispatch_map()
    users = set(external.external)
    meta = module_footprints._catalog()
    seen = set(meta["registry_test_files"]) | set(
        inherited_methods.catalog()["reflective_test_files"]
    )
    users.update(t["number"] for t in sel.tests if t["file"] in seen)
    for p, lines in meta["registry_read_lines"].items():
        sel.load_shard(p)
        users.update(sel.line_tests(p, lines))
    for path, owner, name, family in proposals:
        users.update(external.external_by_source.get(path, ()))
        if path in sel.files:
            users.update(sel.sets[sel.files[path]])
        for p, line, scope, kind in symbol_impact.index().get(name, ()):
            if kind == "attr" and symbol_impact.direct_calls_only(
                sel, p, line, name, scope
            ):
                continue
            if p.startswith("tests/"):
                users.update(t["number"] for t in sel.tests if t["file"] == p)
                continue
            if not scope:
                return selected
            sel.load_shard(p)
            users.update(sel.line_tests(p, {line}))
    old = pristine(root, tuple(proposals))
    if not old:
        return selected
    with tempfile.TemporaryDirectory() as tmp:
        changed = Path(tmp) / "src"
        checkout(changed, changes, sel)
        new = inspect_root(changed, proposals)
    if not new or new.keys() != old.keys():
        return selected
    spans = {}
    for key, (prior, old_route, expr) in old.items():
        p, owner, name, cls = ast.literal_eval(key)
        now, new_route, newexpr = new[key]
        if expr != newexpr:
            return selected
        # Even unchanged qualified names can refer to a changed method body.
        affected = any(
            (binding and binding[0] == p and binding[1] == owner + "." + name)
            for binding in (prior, now)
        )
        if not affected and prior == now and old_route == new_route:
            continue
        if expr:
            data = generator_contexts.data_dispatch()
            users.update(
                structural_dependencies.table_observers(sel, data, p, cls, {expr})
            )
        if prior:
            path, qualname, line, accepts_empty = prior
            # A call that raised during argument binding never reached the old body.
            if not accepts_empty and now and now[-1]:
                return selected
            at = path, qualname, line
            if at not in spans:
                _, tree = sel.source(path)
                f = next(
                    (
                        n
                        for n in ast.walk(tree)
                        if isinstance(n, ast.FunctionDef)
                        and n.lineno == line
                        and n.name == qualname.split(".")[-1]
                    ),
                    None,
                )
                if not f:
                    return selected
                span = range(f.body[0].lineno, f.end_lineno + 1)
                if set(span) & set(sel.mapping.get("import_lines", {}).get(path, ())):
                    return selected
                spans[at] = sel.line_tests(path, span)
            users.update(spans[at])
        elif not expr:
            return selected
        elif not prior:
            # New attribute can turn previously missing direct calls into successes.
            for rp, rl, scope, kind in symbol_impact.index().get(name, ()):
                if kind == "attr":
                    if not scope:
                        return selected
                    sel.load_shard(rp)
                    users.update(sel.line_tests(rp, {rl}))
    return (selected & users) | (selected & sel.confirmed_failures)
