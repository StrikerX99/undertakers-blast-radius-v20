"""Resolve conservative whole-suite fallbacks through statically named readers.

The index is built from pristine source. A pure declaration that cannot be
followed through module initialization leaves the original selection in place.
"""

import ast
import reflective_use
import marshal
import sys
import tempfile
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import forkprobe
from source_fast import paired_tree
import os
import importlib
import inspect
import types
from case_verify import changed_lines

STDLIB = {
    "itertools",
    "collections",
    "collections.abc",
    "typing",
    "functools",
    "operator",
    "math",
    "enum",
    "types",
}
FOLDERS = ("sqlglot/optimizer/", "sqlglot/generators/", "sqlglot/parsers/")
CONTROL = (ast.Yield, ast.YieldFrom, ast.Await, ast.Global, ast.Nonlocal)
FUN = (ast.FunctionDef, ast.AsyncFunctionDef)


def build(root, output):
    refs = defaultdict(set)
    for directory in ("sqlglot", "tests"):
        for file in sorted((root / directory).rglob("*.py")):
            path = file.relative_to(root).as_posix()

            def visit(n, scope=0):
                # These two SQLGlot declarations have a closed set of consumers.
                # Their literal SQL names never name Python module attributes.
                if isinstance(n, (ast.Assign, ast.AnnAssign)):
                    targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                    if len(targets) == 1 and isinstance(targets[0], ast.Name):
                        var = targets[0].id
                        value = n.value
                        if (
                            var == "arg_types"
                            and isinstance(value, ast.Dict)
                            and all(
                                isinstance(k, ast.Constant) and isinstance(k.value, str)
                                for k in value.keys
                            )
                            and all(
                                isinstance(v, ast.Constant)
                                and isinstance(v.value, bool)
                                for v in value.values
                            )
                        ) or (
                            var == "RESERVED_KEYWORDS"
                            and isinstance(value, (ast.List, ast.Tuple, ast.Set))
                            and all(
                                isinstance(v, ast.Constant) and isinstance(v.value, str)
                                for v in value.elts
                            )
                        ):
                            return
                if isinstance(n, ast.Attribute):
                    refs[n.attr].add((path, n.lineno, scope, "attr"))
                elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                    refs[n.id].add((path, n.lineno, scope, "name"))
                elif (
                    isinstance(n, ast.Constant)
                    and type(n.value) is str
                    and n.value.isidentifier()
                ):
                    refs[n.value].add((path, n.lineno, scope, "string"))
                elif isinstance(n, ast.ImportFrom):
                    for alias in n.names:
                        refs[alias.name].add(
                            (path, n.lineno, scope, "from:" + str(n.module))
                        )
                if isinstance(n, FUN):
                    for child in [
                        *n.decorator_list,
                        n.args,
                        *([] if n.returns is None else [n.returns]),
                    ]:
                        visit(child, scope)
                    for stmt in n.body:
                        visit(stmt, n.lineno)
                else:
                    for child in ast.iter_child_nodes(n):
                        visit(child, scope)

            visit(ast.parse(file.read_bytes()))
    output.write_bytes(marshal.dumps({n: sorted(v) for n, v in refs.items()}))


@lru_cache(maxsize=1)
def index():
    return marshal.loads(Path(__file__).with_name("use-index.bin").read_bytes())


def simple_binding(node):
    if (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    ):
        if isinstance(node.value, ast.Constant) and type(node.value.value) in (
            str,
            int,
            float,
            bool,
            type(None),
        ):
            return node.targets[0].id
    return None


def declaration(n):
    if isinstance(n, (*FUN, ast.ClassDef)):
        return ("def", n.name)
    if key := simple_binding(n):
        return ("literal", key)
    if isinstance(n, ast.ImportFrom) and n.level == 0 and n.module in STDLIB:
        return ("import", n.module, tuple((v.name, v.asname) for v in n.names))
    return ("fixed", ast.dump(n, include_attributes=False))


def removed_import_bindings(sel, before, after, path):
    """Recognize the removal of plain exports from a still-executed from-import."""
    if not (
        isinstance(before, ast.ImportFrom)
        and isinstance(after, ast.ImportFrom)
        and before.level == after.level == 0
        and before.module == after.module
        and before.module
        and before.module.startswith("sqlglot.")
    ):
        return None
    old = [(a.name, a.asname) for a in before.names]
    keep = [(a.name, a.asname) for a in after.names]
    if (
        not keep
        or len(old) == len(keep)
        or len(set(old)) != len(old)
        or any(a == "*" for a, _ in old)
        or [x for x in old if x in keep] != keep
    ):
        return None
    leaf = path[:-3].replace("/", ".")

    def relative_from(kind):
        return kind.startswith("from:") and leaf.endswith("." + kind[5:])

    if any(
        kind == "from:" + leaf or relative_from(kind)
        for _, _, _, kind in index().get("*", ())
    ):
        return None
    try:
        _, provider = sel.source(before.module.replace(".", "/") + ".py")
    except (OSError, KeyError, ValueError):
        return None
    definitions = {n.name for n in provider.body if isinstance(n, (*FUN, ast.ClassDef))}
    removed = [pair for pair in old if pair not in keep]
    if any(
        relative_from(kind)
        for name, alias in removed
        for _, _, _, kind in index().get(alias or name, ())
    ):
        return None
    if any(name not in definitions for name, _ in removed):
        return None
    return [alias or name for name, alias in removed]


def analyze(sel, changes, apply, equal):
    if not changes or any(
        c["old"] != c["new"]
        or not (c["old"].startswith(FOLDERS) or c["old"] == "sqlglot/generator.py")
        or not c["old"].endswith(".py")
        for c in changes
    ):
        return None
    spans = defaultdict(set)
    tokens = []
    overrides = []
    names = []
    modules = []

    def mark(p, n):
        start = (
            min(v.lineno for v in n.body) if isinstance(n, FUN) and n.body else n.lineno
        )
        if isinstance(n, FUN) and (start == n.lineno or n.decorator_list):
            # A single-line body shares its line with an eager definition.
            start = n.lineno
        spans[p].update(range(start, n.end_lineno + 1))

    for change in changes:
        path = change["old"]
        raw, old = sel.source(path)
        updated = apply(raw, change)[0]
        if change.get("_compiled_source") is not updated:
            compile(updated, path, "exec", dont_inherit=True)
        new = paired_tree(change, old, raw, updated)
        modules.append(path[:-3].replace("/", "."))

        def diff(a, b, owner=""):
            da = {declaration(n): n for n in a}
            db = {declaration(n): n for n in b}
            if len(da) != len(a) or len(db) != len(b):
                return False
            common = da.keys() & db.keys()
            if [k for k in da if k in common] != [k for k in db if k in common]:
                return False
            for k in da.keys() | db.keys():
                x, y = da.get(k), db.get(k)
                if x is not None and y is not None and equal(x, y):
                    continue
                n = x if x is not None else y
                if k[0] == "import" and not owner and (x is None or y is None):
                    if any(t.name == "*" for t in n.names):
                        return False
                    names.extend((path, t.asname or t.name) for t in n.names)
                elif k[0] == "literal" and owner and k[1].isupper():
                    tokens.append(k[1])
                elif (
                    isinstance(x, ast.ClassDef)
                    and isinstance(y, ast.ClassDef)
                    and not owner
                ):
                    if not all(
                        equal(getattr(x, f), getattr(y, f))
                        for f in x._fields
                        if f != "body"
                    ) or not diff(x.body, y.body, x.name):
                        return False
                elif isinstance(n, FUN):
                    if x is None or y is None:
                        if any(
                            v is not None and not isinstance(v, ast.Constant)
                            for v in [*n.args.defaults, *n.args.kw_defaults]
                        ):
                            return False
                        # A public method can shadow a method on its parent.
                        if (
                            not n.name.startswith("_")
                            or n.name.startswith("__")
                            or n.decorator_list
                        ):
                            return False
                        if owner:
                            # A direct call dispatches to the new MRO only where the
                            # baseline method was used. Also retain every reader of
                            # the name, including code that stores a bound method.
                            if (
                                y is not None
                                or type(n) is not ast.FunctionDef
                                or any(
                                    isinstance(q, (ast.Yield, ast.YieldFrom, ast.Await))
                                    for q in ast.walk(n)
                                )
                                or not path.startswith("sqlglot/parsers/")
                                or len(changes) != 1
                                or not owner.endswith("Parser")
                                or len(n.args.args) != 1
                                or n.args.args[0].arg != "self"
                                or n.args.posonlyargs
                                or n.args.kwonlyargs
                                or n.args.vararg
                                or n.args.kwarg
                                or n.type_params
                                or not x.body
                                or x.body[0].lineno == x.lineno
                            ):
                                return False
                            # The pinned parser hierarchy consists of ordinary
                            # classes. Avoid custom hooks or decorated declarations.
                            class_node = next(
                                t
                                for t in old.body
                                if isinstance(t, ast.ClassDef) and t.name == owner
                            )
                            if (
                                class_node.decorator_list
                                or class_node.keywords
                                or any(
                                    isinstance(t, ast.FunctionDef)
                                    and t.name
                                    in {
                                        "__getattribute__",
                                        "__getattr__",
                                        "__init_subclass__",
                                    }
                                    for t in class_node.body
                                )
                            ):
                                return False
                            overrides.append((path, owner, n.name))
                        names.append((path, n.name))
                        if x:
                            mark(path, x)
                    else:
                        if (
                            x.decorator_list
                            or y.decorator_list
                            or type(x) is not type(y)
                        ):
                            return False
                        if not equal(x.returns, y.returns) or not equal(
                            x.type_params, y.type_params
                        ):
                            return False
                        if not equal(x.args, y.args):
                            if not equal(x.args.defaults, y.args.defaults) or not equal(
                                x.args.kw_defaults, y.args.kw_defaults
                            ):
                                return False
                            if owner or not n.name.startswith("_"):
                                return False
                            names.append((path, n.name))
                        if any(
                            isinstance(q, CONTROL) for f in (x, y) for q in ast.walk(f)
                        ):
                            return False
                        local = changed_lines(
                            ast.Module(body=[x], type_ignores=[]),
                            ast.Module(body=[y], type_ignores=[]),
                            equal,
                        )
                        if local is None:
                            mark(path, x)
                        else:
                            start = min(n.lineno for n in x.body)
                            spans[path].update(line for line in local if line >= start)
                else:
                    return False
            return True

        right = list(new.body)
        if len(changes) == 1 and len(old.body) == len(right):
            for i, (a, b) in enumerate(zip(old.body, right)):
                if isinstance(a, ast.ImportFrom) and not equal(a, b):
                    removed = removed_import_bindings(sel, a, b, path)
                    if removed:
                        names.extend((path, name) for name in removed)
                        right[i] = a
        if not diff(old.body, right):
            return None
    if not tokens and not names:
        return None
    tests = set()

    def reader(path, line, scope):
        if path.startswith("tests/"):
            tests.update(t["number"] for t in sel.tests if t["file"] == path)
            return True
        if not scope:
            return False
        spans[path].add(line)
        return True

    # An unresolved eager consumer can propagate values to other tables.
    for field in set(tokens):
        for p, line, scope, kind in index().get(field, ()):
            if not reader(p, line, scope):
                return None
    method_names = {(p, name) for p, _, name in overrides}
    for file, name in set(names):
        module = file[:-3].replace("/", ".")
        for p, line, scope, kind in index().get(name, ()):
            if (
                p == file
                and kind in ("name", "string")
                or kind == "from:" + module
                or kind == "attr"
                or kind == "string"
            ):
                if (
                    (file, name) in method_names
                    and kind == "attr"
                    and p.startswith("sqlglot/")
                    and direct_calls_only(sel, p, line, name, scope)
                ):
                    continue
                if not reader(p, line, scope):
                    return None
    return dict(spans), tests, modules, overrides


def direct_calls_only(sel, path, line, name, scope):
    from source_fast import ancestors_at

    _, tree = sel.source(path)
    nodes = ancestors_at(tree, line)
    reads = {
        n
        for n in nodes
        if isinstance(n, ast.Attribute) and n.attr == name and n.lineno == line
    }
    callees = {
        n.func
        for n in nodes
        if isinstance(n, ast.Call) and not n.args and not n.keywords
    }
    # If the receiver dispatches to the deleted ordinary method, its body has
    # its own footprint. Other uses of the bound method keep the caller.
    if not scope:
        # A class dispatch dictionary stores this function object. Only the
        # lambda body runs the method, on the receiver provided by the caller.
        tables = {
            statement.value
            for cls in nodes
            if isinstance(cls, ast.ClassDef)
            for statement in cls.body
            if isinstance(statement, (ast.Assign, ast.AnnAssign))
            and isinstance(statement.value, ast.Dict)
            and (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                or isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
            )
        }
        deferred = {
            func.body.func
            for table in tables
            for func in table.values
            if isinstance(func, ast.Lambda)
            and isinstance(func.body, ast.Call)
            and len(func.args.args) == 1
            and not func.args.posonlyargs
            and not func.args.kwonlyargs
            and not func.args.vararg
            and not func.args.kwarg
            and not func.args.defaults
            and isinstance(func.body.func, ast.Attribute)
            and isinstance(func.body.func.value, ast.Name)
            and func.body.func.value.id == func.args.args[0].arg
        }
        callees.intersection_update(deferred)
    return bool(reads) and reads <= callees


def load_checked(payload):
    root, modules, overrides, marker = payload
    os.chdir(root)
    sys.path.insert(0, str(root))
    loaded = {m: importlib.import_module(m) for m in modules}
    if overrides:
        from sqlglot.parser import Parser

        for path, owner, name in overrides:
            cls = getattr(loaded[path[:-3].replace("/", ".")], owner)
            if (
                not issubclass(cls, Parser)
                or type(cls) is not type
                or name in vars(cls)
            ):
                return
            if any(
                hook in vars(ancestor)
                for ancestor in cls.__mro__
                if ancestor is not object
                for hook in ("__init_subclass__", "__getattribute__", "__getattr__")
            ):
                return
            inherited = inspect.getattr_static(cls, name)
            if (
                type(inherited) is not types.FunctionType
                or inspect.isgeneratorfunction(inherited)
                or inspect.iscoroutinefunction(inherited)
            ):
                return
            pars = list(
                inspect.signature(inherited, follow_wrapped=False).parameters.values()
            )
            if (
                len(pars) != 1
                or pars[0].kind
                not in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
                or pars[0].default is not inspect.Parameter.empty
            ):
                return
    marker.touch()


def refine(sel, changes, original, apply, equal, checkout, remaining):
    if remaining < 2 or len(original) < max(50, len(sel.all) // 3):
        return original, None
    result = analyze(sel, changes, apply, equal)
    if result is None:
        return original, None
    spans, tests, modules, overrides = result
    protected = set(tests)
    tests.update(sel.unknown)
    import_lines = sel.mapping.get("import_lines", {})
    for p, lines in spans.items():
        sel.load_shard(p)
        if lines.intersection(import_lines.get(p, ())):
            return original, None
        tests.update(sel.line_tests(p, lines))
        if p in sel.files:
            protected.update(sel.sets[sel.files[p]])
    for c in changes:
        p = c["old"]
        if p in sel.files:
            protected.update(sel.sets[sel.files[p]])
    # Standalone Python processes that the ordinary line tracer could not label.
    external = sel.dispatch_map()
    protected.update(external.external)
    for c in changes:
        protected.update(external.external_by_source.get(c["old"], ()))
    from parser_cells import data

    ref = data()
    protected.update(
        sel.by_name[n]
        for i in reflective_use.indices(sel, ref)
        if (n := ref["names"][i]) in sel.by_name
    )
    with tempfile.TemporaryDirectory(prefix="br-symbol-") as d:
        temp = Path(d)
        repo = temp / "repo"
        marker = temp / "ok"
        checkout(repo, changes, sel)
        if (
            not forkprobe.run(
                (repo, modules, overrides, marker),
                {"PYTHONPATH": str(repo)},
                load_checked,
                min(3, remaining),
            )
            or not marker.exists()
        ):
            return original, None
    return original & (tests | protected), (spans, protected)


if __name__ == "__main__":
    build(Path(sys.argv[1]), Path(sys.argv[2]))
