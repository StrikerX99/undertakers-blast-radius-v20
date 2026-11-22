"""Compare generated Python at its compilation site when a pure generator method changes."""

import ast
from functools import lru_cache
import hashlib
import json
import marshal
import os
from pathlib import Path
import pickle
import sys
import tempfile
import zlib

TARGET = "tests/test_executor.py::TestExecutor::test_execute_tpcds"
PIN = "eb11aec4be8cb2122afe713a4816662c3f0fb2fa"
HERE = Path(__file__).parent


def semantic_key(text):
    # Parse the expression exactly as PythonExecutor.generate does. Compilation
    # rejects context or syntax errors which AST construction alone can accept.
    compile(text, text, "eval", optimize=2)
    tree = ast.parse(text, mode="eval")
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).digest()


def duck_key(conn, sql):
    # DuckDB's native parser includes source positions, which differ under indentation.
    def normalized(v):
        if isinstance(v, dict):
            return {
                k: normalized(w)
                for k, w in v.items()
                if k not in ("query_location", "query_location_length")
            }
        if isinstance(v, list):
            return [normalized(w) for w in v]
        return v

    data = json.loads(conn.execute("SELECT json_serialize_sql(?)", [sql]).fetchone()[0])
    if data.get("error") or len(data.get("statements", ())) != 1:
        raise ValueError("Expected one DuckDB SELECT")
    raw = json.dumps(normalized(data), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).digest()


def build(root, dest):
    os.chdir(root)
    sys.path.insert(0, str(root))
    from sqlglot import parse_one
    import sqlglot.executor as engine
    from sqlglot.generator import Generator
    from sqlglot.executor.python import PythonExecutor
    from tests.helpers import load_sql_fixture_pairs
    from query_case import source_environment, table_digest

    cfg = json.loads((HERE / "executor-queries.json").read_bytes())
    scope = source_environment(root, cfg["driver_sha"])
    native_opt, native_render = engine.optimize, Generator.generate
    compiler, logs, raw_only, outside = {}, set(), set(), set()
    driver_path = root / "sqlglot/executor/__init__.py"
    driver_ast = ast.parse(driver_path.read_bytes())
    log_sites = {
        v.lineno
        for v in ast.walk(driver_ast)
        if isinstance(v, ast.Call)
        and isinstance(v.func, ast.Attribute)
        and v.func.attr == "debug"
        and isinstance(v.func.value, ast.Name)
        and v.func.value.id == "logger"
        and len(v.args) == 2
        and isinstance(v.args[1], ast.Call)
        and isinstance(v.args[1].func, ast.Attribute)
        and v.args[1].func.attr == "sql"
    }
    if not log_sites:
        raise ValueError("Unknown log call")
    mon = sys.monitoring
    tool = next(i for i in range(6) if mon.get_tool(i) is None)
    mon.use_tool_id(tool, "compile-origin")
    prefix = str(root) + "/"

    def observe(code, offset):
        name = code.co_filename
        if name.startswith(prefix + "sqlglot/"):
            outside.add((name[len(prefix) :], code.co_qualname, code.co_firstlineno))
        return mon.DISABLE

    mon.register_callback(tool, mon.events.PY_START, observe)

    def monitor(on):
        mon.set_events(tool, mon.events.PY_START if on else 0)

    active = nested = False

    def optimized(*args, **kwargs):
        nonlocal active
        out = native_opt(*args, **kwargs)
        active = True
        monitor(True)
        return out

    def render(gen, expr, copy=True):
        nonlocal nested
        if not active or nested:
            return native_render(gen, expr, copy=copy)
        nested = True
        monitor(False)
        try:
            out = native_render(gen, expr, copy=copy)
            key = hashlib.sha256(out.encode()).digest()
            caller = sys._getframe(1)
            frame = caller
            while frame and frame.f_code is not engine.execute.__code__:
                frame = frame.f_back
            if frame is not None and frame.f_lineno in log_sites:
                logs.add(key)
            elif caller.f_code is PythonExecutor.generate.__code__:
                compiler[key] = semantic_key(out)
            else:
                raw_only.add(key)
            return out
        finally:
            nested = False
            monitor(True)

    engine.optimize, Generator.generate = optimized, render
    try:
        for i, (meta, sql, _) in enumerate(
            load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql")
        ):
            if not meta.get("execute"):
                continue
            actual = scope["mp_execute"](
                pickle.loads(pickle.dumps(parse_one(sql))), meta
            )
            active = False
            monitor(False)
            if table_digest(actual) != cfg["queries"][str(i)]["result"]:
                raise ValueError("Unexpected native worker result")
    finally:
        monitor(False)
        mon.free_tool_id(tool)
        engine.optimize, Generator.generate = native_opt, native_render
    for key in raw_only:
        compiler.pop(key, None)
    logs.difference_update(compiler.keys() | raw_only)
    import duckdb
    from sqlglot import transpile

    duck = {}
    conn = duckdb.connect()
    try:
        for i, (meta, sql, _) in enumerate(
            load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql")
        ):
            if meta.get("execute"):
                native = transpile(sql, write="duckdb")[0]
                # All fixtures were executed successfully against these same pinned tables.
                duck[i] = (
                    hashlib.sha256(native.encode()).digest(),
                    duck_key(conn, native),
                )
    finally:
        conn.close()
    result = PIN, compiler, sorted(logs), sorted(outside), duck
    dest.write_bytes(zlib.compress(marshal.dumps(result), 9))
    print(
        f"{len(compiler)} compiler inputs; {len(logs)} log inputs; {len(outside)} unguarded functions",
        flush=True,
    )


@lru_cache(maxsize=1)
def index():
    pin, compiler, logs, outside, duck = marshal.loads(
        zlib.decompress((HERE / "compiled-inputs.bin").read_bytes())
    )
    if pin != PIN:
        raise ValueError("Wrong pin")
    return compiler, set(logs), set(outside), duck


def replay(backend=True, duck_indices=()):
    import importlib
    from backend_sql import DATA

    compiler, logs, _, duck = index()
    if duck_indices:
        from sqlglot import transpile
        from tests.helpers import load_sql_fixture_pairs

        conn = None
        try:
            fixture = load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql")
            for i, (meta, sql, _) in enumerate(fixture):
                if i not in duck_indices or not meta.get("execute"):
                    continue
                text = transpile(sql, write="duckdb")[0]
                raw, parsed = duck[i]
                if hashlib.sha256(text.encode()).digest() == raw:
                    continue
                if conn is None:
                    import duckdb

                    conn = duckdb.connect()
                if duck_key(conn, text) != parsed:
                    return False
        finally:
            if conn is not None:
                conn.close()
    if not backend:
        return True
    version, streams, blobs = marshal.loads(zlib.decompress(DATA.read_bytes()))
    if version != 2:
        return False

    @lru_cache(maxsize=12)
    def tree(key):
        return pickle.loads(blobs[key])

    for _, module, name, options, events in streams:
        owner = importlib.import_module(module)
        for part in name.split("."):
            owner = getattr(owner, part)
        receiver = owner().generator(**options)
        for key, path, expected in events:
            expr = tree(key)
            for field, pos in path:
                expr = expr.args[field]
                if pos is not None:
                    expr = expr[pos]
            text = receiver.generate(expr, copy=True)
            if hashlib.sha256(text.encode()).digest() != expected:
                if expected in logs:
                    continue
                if expected not in compiler or semantic_key(text) != compiler[expected]:
                    return False
    return True


def check_child(request):
    root, marker, backend, indices = request
    import native_tests

    watch = native_tests.watch_reference(root)
    if replay(backend, indices) and not watch.missing:
        marker.touch()


def template_pieces(expr):
    # The literal parts may change. Dynamic expressions must be evaluated in the
    # same order with the same arguments, conversion and format spec.
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return ()
    if isinstance(expr, ast.JoinedStr):
        return tuple(
            ast.dump(e) for e in expr.values if not isinstance(e, ast.Constant)
        )
    return None


def same_effects(a, b, equal):
    if len(a.body) != len(b.body) or not a.body:
        return False
    if all(equal(x, y) for x, y in zip(a.body[:-1], b.body[:-1])):
        left, right = a.body[-1], b.body[-1]
        if isinstance(left, ast.Return) and isinstance(right, ast.Return):
            old, new = template_pieces(left.value), template_pieces(right.value)
            if old is not None and old == new:
                return True
    if not pure(a) or not pure(b):
        return False

    # Positional leaf substitutions preserve the calls and settings the
    # original method reads. Never admit a new property, global or iterator.
    def skeleton(node):
        if isinstance(node, ast.Constant):
            return ("constant", type(node.value).__name__)
        if isinstance(node, (ast.cmpop,)):
            return ("comparison",)
        if isinstance(node, ast.AST):
            return (
                type(node).__name__,
                tuple(skeleton(getattr(node, k)) for k in node._fields),
            )
        if isinstance(node, list):
            return tuple(map(skeleton, node))
        return node

    return skeleton(a.body) == skeleton(b.body)


def pure(method):
    """Admit scalar computations on arguments and stored settings, without writes or escaping calls."""
    args = {
        p.arg
        for p in (*method.args.args, *method.args.posonlyargs, *method.args.kwonlyargs)
    }
    allowed = (
        ast.Return,
        ast.Constant,
        ast.Load,
        ast.Store,
        ast.Name,
        ast.Attribute,
        ast.BinOp,
        ast.UnaryOp,
        ast.Compare,
        ast.BoolOp,
        ast.GeneratorExp,
        ast.comprehension,
        ast.Call,
        ast.operator,
        ast.unaryop,
        ast.cmpop,
        ast.boolop,
        ast.IfExp,
    )
    locals_ = args | {
        n.target.id
        for n in ast.walk(method)
        if isinstance(n, ast.comprehension) and isinstance(n.target, ast.Name)
    }
    builtins = {"len", "sum", "min", "max", "all", "any", "str", "int", "bool", "abs"}
    for stmt in method.body:
        for n in ast.walk(stmt):
            if not isinstance(n, allowed):
                return False
            if isinstance(n, ast.Name) and n.id not in locals_ | builtins:
                return False
            if isinstance(n, ast.Attribute) and (
                not isinstance(n.value, ast.Name)
                or n.value.id not in args
                or n.attr.startswith("__")
            ):
                return False
            if isinstance(n, ast.Call) and (
                not isinstance(n.func, ast.Name)
                or n.func.id
                not in {
                    "len",
                    "sum",
                    "min",
                    "max",
                    "all",
                    "any",
                    "str",
                    "int",
                    "bool",
                    "abs",
                }
            ):
                return False
            if isinstance(n, ast.comprehension) and (
                n.is_async or not isinstance(n.target, ast.Name)
            ):
                return False
    return True


def eligible(sel, changes, apply, equal):
    import executor_stages
    from source_fast import paired_tree

    _, _, outside, _ = index()
    indices = set()
    backend = False
    for change in changes:
        p = change["old"]
        if p != change["new"] or (
            p != "sqlglot/generator.py" and not p.startswith("sqlglot/generators/")
        ):
            return False
        raw, old = sel.source(p)
        new = apply(raw, change)[0]
        after = paired_tree(change, old, raw, new)
        seen = []

        def traverse(alist, blist, scope):
            nonlocal backend
            if len(alist) != len(blist):
                return False
            for a, b in zip(alist, blist):
                if equal(a, b):
                    continue
                if type(a) is not type(b) or not isinstance(
                    a, (ast.ClassDef, ast.FunctionDef)
                ):
                    return False
                if not all(
                    equal(getattr(a, k), getattr(b, k))
                    for k in a._fields
                    if k != "body"
                ):
                    return False
                name = ".".join((*scope, a.name))
                if isinstance(a, ast.ClassDef):
                    if not traverse(a.body, b.body, (*scope, a.name)):
                        return False
                elif not same_effects(a, b, equal) or a.decorator_list:
                    return False
                else:
                    if (p, name, a.lineno) in outside:
                        return False
                    spans = set(range(a.body[0].lineno, a.end_lineno + 1))
                    stages = executor_stages.query({p: spans}, sel.mapping, {p: name})
                    if stages is None or set(stages) - {"worker_after", "duck"}:
                        return False
                    backend |= bool(stages.get("worker_after"))
                    indices.update(stages.get("duck", ()))
                    seen.append(name)
            return True

        if not traverse(old.body, after.body, ()) or not seen:
            return False
    return backend, indices


def refine(sel, changes, selected, apply, equal, checkout, remaining):
    number = sel.by_name.get(TARGET)
    if (
        number is None
        or number not in selected
        or number in sel.confirmed_failures
        or sel.unknown
        or remaining < 2
    ):
        return selected
    if not changes or any(
        c["old"] != "sqlglot/generator.py"
        and not c["old"].startswith("sqlglot/generators/")
        for c in changes
    ):
        return selected
    from verification_budget import catalog_price

    price = getattr(sel, "verification_cost", None) or catalog_price(sel.by_number)
    possible = price(frozenset(selected)) - price(frozenset(selected - {number}))
    if possible < 1800:
        return selected
    proof = eligible(sel, changes, apply, equal)
    if not proof:
        return selected
    backend, indices = proof
    import forkprobe

    with tempfile.TemporaryDirectory(prefix="native-compile-") as tmp:
        root, marker = Path(tmp) / "src", Path(tmp) / "pass"
        checkout(root, changes, sel)
        forkprobe.run(
            (root, marker, backend, indices),
            {"PYTHONPATH": str(root), "SKIP_INTEGRATION": "0"},
            check_child,
            min(remaining, max(1.5, min(5.0, possible / 1300))),
        )
        return selected - {number} if marker.is_file() else selected


if __name__ == "__main__":
    build(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
