"""Check whether an edit can reach the executor's generators and reference SQL."""

import ast
from source_fast import analysis_tree
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import tempfile

import forkprobe
from verification_budget import pack_ms

TEST = "tests/test_executor.py::TestExecutor::test_execute_tpcds"


@lru_cache(maxsize=1)
def evidence():
    return json.loads(Path(__file__).with_name("generator-uses.json").read_bytes())


def passive_header(fn):
    return (
        isinstance(fn, ast.FunctionDef)
        and not fn.decorator_list
        and not fn.type_params
        and not any(fn.args.defaults)
        and not any(fn.args.kw_defaults)
    )


def header(a, b, eq):
    return type(a) is type(b) and all(
        eq(getattr(a, k), getattr(b, k)) for k in a._fields if k != "body"
    )


def literal_returns(stmts):
    return bool(stmts) and all(
        isinstance(s, ast.Return)
        and isinstance(s.value, ast.Constant)
        and type(s.value.value) in (str, int, float, bool, type(None))
        for s in stmts
    )


def lexical(fn, names):
    import symtable

    root = symtable.symtable(ast.unparse(fn), "<scope>", "exec").get_children()[0]
    found = {}
    for name in names:
        try:
            sym = root.lookup(name)
        except KeyError:
            continue
        found[name] = (
            sym.is_local(),
            sym.is_global(),
            sym.is_nonlocal(),
            sym.is_free(),
            sym.is_parameter(),
        )
    return found


def prefix_names(statements):
    """Outer names read by the prefix, respecting generator-expression bindings."""
    names = set()

    def walk(n, bound):
        if isinstance(
            n, (ast.FunctionDef, ast.ClassDef, ast.Lambda, ast.AsyncFunctionDef)
        ):
            # A nested callable can inspect its enclosing cell/bindings.
            names.add("locals")
            return
        if isinstance(n, (ast.GeneratorExp, ast.ListComp, ast.SetComp, ast.DictComp)):
            local = set(bound)
            for gen in n.generators:
                walk(gen.iter, local)
                local.update(
                    t.id
                    for t in ast.walk(gen.target)
                    if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store)
                )
                for expr in gen.ifs:
                    walk(expr, local)
            for expr in [n.key, n.value] if isinstance(n, ast.DictComp) else [n.elt]:
                walk(expr, local)
            return
        if (
            isinstance(n, ast.Name)
            and isinstance(n.ctx, ast.Load)
            and n.id not in bound
        ):
            names.add(n.id)
        for child in ast.iter_child_nodes(n):
            walk(child, bound)

    for s in statements:
        walk(s, set())
    return names


def body(start, left, right, scope, eq, old_regions, new_regions):
    if (
        not header(left, right, eq)
        or left.decorator_list
        or ast.get_docstring(left, clean=False) != ast.get_docstring(right, clean=False)
    ):
        return False
    if any(
        isinstance(n, (ast.Yield, ast.YieldFrom, ast.Global, ast.Nonlocal, ast.Await))
        for f in (left, right)
        for n in ast.walk(f)
    ):
        return False
    i = 0
    while i < min(len(left.body), len(right.body)) and eq(left.body[i], right.body[i]):
        i += 1
    pre = left.body[:i]
    names = prefix_names(pre)
    if names & {
        "locals",
        "globals",
        "vars",
        "eval",
        "exec",
        "super",
        "__class__",
        "inspect",
        "sys",
    }:
        return False

    def uses_classcell(fn):
        return any(
            isinstance(n, ast.Name) and n.id in {"__class__", "super"}
            for n in ast.walk(fn)
        )

    if uses_classcell(left) != uses_classcell(right) or lexical(left, names) != lexical(
        right, names
    ):
        return False
    if literal_returns(left.body[i:]) and literal_returns(right.body[i:]):
        return True
    for f, regions in ((left, old_regions), (right, new_regions)):
        suffix = f.body[i:]
        if suffix:
            regions.append((scope, suffix[0].lineno, f.end_lineno))
    return True


def passive_callback(n):
    if isinstance(n, ast.Name):
        return True
    if isinstance(n, ast.Attribute):
        return isinstance(n.value, ast.Name) and n.value.id in {"transforms", "exp"}
    if isinstance(n, ast.Lambda):
        return not n.args.defaults and not any(n.args.kw_defaults)
    return (
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "transforms"
        and n.func.attr == "preprocess"
        and not n.keywords
        and len(n.args) == 1
        and isinstance(n.args[0], (ast.List, ast.Tuple))
        and all(
            isinstance(x, (ast.Name, ast.Attribute)) and passive_callback(x)
            for x in n.args[0].elts
        )
    )


def table(a, b, eq):
    if not (
        isinstance(a, ast.Assign)
        and isinstance(b, ast.Assign)
        and eq(a.targets, b.targets)
        and len(a.targets) == 1
        and isinstance(a.targets[0], ast.Name)
        and a.targets[0].id == "TRANSFORMS"
        and isinstance(a.value, ast.Dict)
        and isinstance(b.value, ast.Dict)
        and eq(a.value.keys, b.value.keys)
    ):
        return None
    changed = []
    for k, x, y in zip(a.value.keys, a.value.values, b.value.values):
        if eq(x, y):
            continue
        if not (isinstance(x, ast.Call) and isinstance(y, ast.Call)):
            return None
        if not (
            isinstance(k, ast.Attribute)
            and isinstance(k.value, ast.Name)
            and k.value.id == "exp"
            and passive_callback(x)
            and passive_callback(y)
        ):
            return None
        changed.append((k.attr, (x.lineno, x.end_lineno), (y.lineno, y.end_lineno)))
    return changed or None


def plan(path, old, new, eq):
    if not any(
        isinstance(n, ast.ImportFrom)
        and n.module == "__future__"
        and any(a.name == "annotations" for a in n.names)
        for n in old.body
    ):
        return None
    old_regions = []
    new_regions = []
    tables = []

    def descend(a, b, owner=""):
        old_names = {
            n.name
            for n in a
            if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }
        old_signatures = (
            {
                n.name
                for n in ast.walk(old)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if not owner
            else set()
        )
        j = 0
        for new_stmt in b:
            n = new_stmt
            if isinstance(n, ast.FunctionDef) and n.name not in old_names:
                if not passive_header(n) or n.name.startswith("__"):
                    return False
                if owner and not n.name.endswith("_sql"):
                    return False
                if not owner and (
                    not n.name.startswith("_") or n.name in old_signatures
                ):
                    return False
                if any(
                    isinstance(t, ast.Name)
                    and t.id == n.name
                    or isinstance(t, ast.Constant)
                    and t.value == n.name
                    for t in ast.walk(old)
                ):
                    return False
                # Reject a new binding that masks an earlier assignment/import.
                if any(
                    isinstance(t, ast.Name)
                    and isinstance(t.ctx, (ast.Store, ast.Del))
                    and t.id == n.name
                    or isinstance(t, ast.alias)
                    and (t.asname or t.name) == n.name
                    for t in ast.walk(old)
                ):
                    return False
                new_regions.append(
                    (
                        (owner + "." if owner else "") + n.name,
                        n.body[0].lineno,
                        n.end_lineno,
                    )
                )
                continue
            if j >= len(a):
                return False
            prev = a[j]
            j += 1
            if eq(prev, n):
                continue
            if (
                not owner
                and isinstance(prev, ast.ClassDef)
                and isinstance(n, ast.ClassDef)
                and header(prev, n, eq)
                and n.name.endswith("Generator")
            ):
                if not descend(prev.body, n.body, n.name):
                    return False
            elif isinstance(prev, ast.FunctionDef) and isinstance(n, ast.FunctionDef):
                if not body(
                    path,
                    prev,
                    n,
                    (owner + "." if owner else "") + n.name,
                    eq,
                    old_regions,
                    new_regions,
                ):
                    return False
            elif owner:
                keys = table(prev, n, eq)
                if not keys:
                    return False
                tables.extend((owner, k, before, after) for k, before, after in keys)
            else:
                return False
        return j == len(a)

    if not descend(old.body, new.body):
        return None
    return old_regions, new_regions, tables


def probe(payload):
    root, destinations, regions, candidates, marker, cpu = payload
    signal.setitimer(signal.ITIMER_PROF, cpu)
    os.chdir(root)
    sys.path.insert(0, str(root))
    import importlib

    checked = {path: {} for path in destinations}
    for path in destinations:
        for scope, lo, hi in regions.get(path, ()):
            checked[path].setdefault(scope, []).append((lo, hi))
    observers = {}
    filenames = {}

    def source_for(code):
        filename = code.co_filename
        if filename not in filenames:
            filenames[filename] = next(
                (p for p in checked if filename.endswith("/" + p)), None
            )
        return filenames[filename]

    monitor = sys.monitoring
    tool = monitor.COVERAGE_ID
    violated = False

    def entry(code, offset):
        nonlocal violated
        if code in observers:
            context = sys._getframe(1).f_locals
            if any(
                all(context.get(k) is v for k, v in vals.items())
                for vals in observers[code]
            ):
                violated = True
                raise ValueError("Changed transform used")
            return
        if code.co_qualname in checked.get(source_for(code), {}):
            monitor.set_local_events(tool, code, monitor.events.LINE)
        return monitor.DISABLE

    def line(code, no):
        nonlocal violated
        if any(
            lo <= no <= hi for lo, hi in checked[source_for(code)][code.co_qualname]
        ):
            violated = True
            raise ValueError("Unverified effects in changed function")
        return monitor.DISABLE

    def produced(code, offset, value):
        if code.co_qualname != "preprocess" or not code.co_filename.endswith(
            "/sqlglot/transforms.py"
        ):
            return monitor.DISABLE
        outer = sys._getframe(2)
        source = source_for(outer.f_code)
        ranges = candidates.get(source, ())
        if any(lo <= outer.f_lineno <= hi for _, _, lo, hi in ranges):
            if type(value) is not type(probe):
                raise ValueError("Unsupported handler factory")
            observers.setdefault(value.__code__, []).append(
                {
                    n: c.cell_contents
                    for n, c in zip(value.__code__.co_freevars, value.__closure__ or ())
                }
            )
            monitor.restart_events()

    monitor.use_tool_id(tool, "generator_reference")
    monitor.register_callback(tool, monitor.events.PY_START, entry)
    monitor.register_callback(tool, monitor.events.LINE, line)
    events = monitor.events.PY_START
    if any(candidates.values()):
        monitor.register_callback(tool, monitor.events.PY_RETURN, produced)
        events |= monitor.events.PY_RETURN
    monitor.set_events(tool, events)
    # Watch the parsed module before any SQLGlot import: a changed callable can
    # also be invoked by a pristine class initializer or lazy registration.
    from tests.helpers import load_sql_fixture_pairs
    from sqlglot import transpile, exp

    for path in destinations:
        module = importlib.import_module(path[:-3].replace("/", "."))
        if Path(module.__file__).resolve() != (root / path).resolve():
            return
        for cls, expr, _, _ in candidates.get(path, ()):
            obj = getattr(module, cls)
            if type(obj) is not type:
                return
            fn = obj.TRANSFORMS[getattr(exp, expr)]
            # The only admitted eager factory is the unchanged preprocess().
            # Its returned closure must have already been observed at that site.
            captures = observers.get(fn.__code__, ())
            my_capture = {
                n: c.cell_contents
                for n, c in zip(fn.__code__.co_freevars, fn.__closure__ or ())
            }
            if not any(
                set(c) == set(my_capture)
                and all(c[k] is v for k, v in my_capture.items())
                for c in captures
            ):
                return
    data = evidence()
    checked_queries = 0
    for i, (meta, sql, _) in enumerate(
        load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql")
    ):
        record = data["outputs"].get(str(i))
        if bool(meta.get("execute")) != (record is not None):
            return
        if record is not None:
            if hashlib.sha256(sql.encode()).hexdigest() != record[0]:
                return
            actual = transpile(sql, write="duckdb")[0]
            if hashlib.sha256(actual.encode()).hexdigest() != record[1]:
                return
            checked_queries += 1
    if (
        not violated
        and i + 1 == data["queries"]
        and checked_queries == len(data["outputs"])
    ):
        marker.touch()


def refine(sel, changes, selected, apply, equal, checkout, remaining, pristine_root):
    number = sel.by_name.get(TEST)
    if (
        number not in selected
        or number in sel.confirmed_failures
        or sel.unknown
        or not changes
        or remaining < 0.2
    ):
        return selected
    if any(
        c["old"] != c["new"] or not c["old"].startswith("sqlglot/generators/")
        for c in changes
    ):
        return selected
    d = evidence()
    if d["pin"] != sel.mapping.get("commit"):
        return selected
    old_regions = {}
    new_regions = {}
    tables = {}
    active = []
    for change in changes:
        path = change["old"]
        if (
            d["hashes"].get(path) is None
            or d["hashes"][path] != sel.mapping["source_hashes"].get(path)
            or path in sel.files
            and number in sel.sets[sel.files[path]]
        ):
            return selected
        raw, old = sel.source(path)
        new_text = apply(raw, change)[0]
        if change.get("_compiled_source") != new_text:
            return selected
        # Only this stage is covered by the native SQL-output and effect checks.
        if d["access"].get(path) != ["duck"]:
            return selected
        plan_data = plan(path, old, analysis_tree(change, old, raw, new_text), equal)
        if plan_data is None:
            return selected
        active.append(path)
        old_regions[path], new_regions[path], tables[path] = plan_data
    # Keep patch interactions across multiple independently classified modules unresolved.
    if len(changes) != 1:
        return selected
    loads = {n: round(1000 * sel.by_number[n]["seconds"]) for n in selected}
    saving = (
        pack_ms(loads.values()) - pack_ms(v for n, v in loads.items() if n != number)
    ) / 1000
    if saving < 2.2 or remaining < 2.2:
        return selected
    from time import monotonic

    deadline = monotonic() + min(remaining, 0.8 * saving, 7.0)
    with tempfile.TemporaryDirectory(prefix="br-generator-reference-") as directory:
        tmp = Path(directory)
        patched = tmp / "src"
        checkout(patched, changes, sel)
        for i, (root, areas) in enumerate(
            ((pristine_root, old_regions), (patched, new_regions))
        ):
            live = deadline - monotonic()
            if live < 0.15:
                return selected
            signal_file = tmp / str(i)
            factory_sites = {
                path: [(cls, k, *ends[i]) for cls, k, *ends in items]
                for path, items in tables.items()
            }
            forkprobe.run(
                (root, active, areas, factory_sites, signal_file, live),
                {"PYTHONPATH": str(root)},
                probe,
                live,
            )
            if not signal_file.exists():
                return selected
    return selected - {number}
