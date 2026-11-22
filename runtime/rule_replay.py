"""Replay eligible optimizer stages of the real TPC-DS test from complete native inputs."""

import ast
import copyreg
import hashlib
import inspect
import io
import os
import pickle
import sqlite3
import sys
import tempfile
import zlib
from functools import lru_cache
from pathlib import Path

import forkprobe
import profile_store
from case_profile import EXEC
from checkpoint_graph import expression_digest

HERE = Path(__file__).resolve().parent
TEST = EXEC + "test_execute_tpcds"
DIALECT = "sqlglot/dialects/dialect.py"


class StagePickler(pickle.Pickler):
    def reducer_override(self, obj):
        # Native Expr.__reduce__ is a transport format; intermediate trees also
        # need their parent pointers, sharing, internal caches and metadata.
        if not isinstance(obj, self.expr):
            return NotImplemented
        slots = {}
        for cls in type(obj).__mro__:
            for key in cls.__dict__.get("__slots__", ()):
                if key not in {"__weakref__", "__dict__"} and hasattr(obj, key):
                    slots[key] = getattr(obj, key)
        return copyreg.__newobj__, (type(obj),), (getattr(obj, "__dict__", None), slots)


def pack(obj):
    from sqlglot import exp

    stream = io.BytesIO()
    p = StagePickler(stream, protocol=5)
    p.expr = exp.Expr
    p.dump(obj)
    return stream.getvalue()


def options_sig(options):
    return hashlib.sha256(pack(options)).digest()


def build(root, out):
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.setrecursionlimit(12000)
    from sqlglot import parse_one
    from sqlglot.optimizer.optimizer import RULES, optimize
    from tests.helpers import load_sql_fixture_pairs
    from query_case import source_environment
    import sqlglot.executor as engine
    from case_profile import digest

    db = sqlite3.connect(out)
    db.executescript(
        "CREATE TABLE stage(id INTEGER PRIMARY KEY,q INTEGER,mod TEXT,fn TEXT,arg BLOB,outcome TEXT,opts BLOB,usable INTEGER); CREATE TABLE reached(src TEXT,line INTEGER,id INTEGER,PRIMARY KEY(src,line,id)) WITHOUT ROWID; CREATE TABLE query(q INTEGER PRIMARY KEY,usable INTEGER);"
    )
    import rule_modules

    rule_modules.build(root, db, RULES)
    baseline = profile_store.get("cases")["cases"]
    expected = {
        r["key"][1]: r["graph"]
        for r in baseline
        if r["kind"] == "optimize" and r["key"][0] == "tpcds"
    }
    queries = list(load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))
    driver = digest((root / "tests/test_executor.py").read_bytes())
    context = source_environment(root, driver)
    context["_tables"] = {
        k: context["Table"](columns=cols, rows=[])
        for k, cols in context["_schema"].items()
    }
    mon = sys.monitoring
    tool = mon.COVERAGE_ID
    mon.use_tool_id(tool, "rule_inputs")
    lines = set()
    files = {}

    def hit(code, line):
        name = code.co_filename
        if name not in files:
            prefix = str(root) + "/"
            files[name] = (
                name[len(prefix) :]
                if name.startswith(prefix + "sqlglot/optimizer/")
                else None
            )
        if files[name]:
            lines.add((files[name], line))
        return mon.DISABLE

    mon.register_callback(tool, mon.events.LINE, hit)

    def wrapper(fn):
        def capture(expression, **kwargs):
            nonlocal lines
            blob = pack((expression, kwargs))
            lines = set()
            mon.restart_events()
            mon.set_events(tool, mon.events.LINE)
            try:
                result = fn(expression, **kwargs)
            finally:
                mon.set_events(tool, 0)
            try:
                sig = expression_digest(result)
                opts = options_sig(kwargs)
                duplicate, kw = pickle.loads(blob)
                identical = (
                    expression_digest(fn(duplicate, **kw)) == sig
                    and options_sig(kw) == opts
                )
            except (ValueError, RuntimeError, TypeError, RecursionError):
                identical = False
                sig = ""
                opts = b""
            n = db.execute(
                "INSERT INTO stage(q,mod,fn,arg,outcome,opts,usable) VALUES(?,?,?,?,?,?,?)",
                (
                    q,
                    fn.__module__,
                    fn.__name__,
                    zlib.compress(blob, 1) if identical else b"",
                    sig,
                    opts,
                    int(identical),
                ),
            ).lastrowid
            db.executemany(
                "INSERT INTO reached VALUES(?,?,?)", [(p, l, n) for p, l in lines]
            )
            return result

        capture.__signature__ = inspect.signature(fn)
        return capture

    rules = tuple(wrapper(f) for f in RULES)

    class Finished(Exception):
        pass

    def observe(*args, **kwargs):
        nonlocal final
        result = optimize(*args, **kwargs, rules=rules)
        final = expression_digest(result)
        raise Finished

    engine.optimize = observe
    for q, graph in sorted(expected.items()):
        final = None
        meta, sql, _ = queries[q]
        try:
            context["mp_execute"](pickle.loads(pickle.dumps(parse_one(sql))), meta)
        except Finished:
            pass
        db.execute("INSERT INTO query VALUES(?,?)", (q, int(final == graph)))
    mon.free_tool_id(tool)
    db.commit()
    print(
        "Optimizer stage inputs",
        db.execute("SELECT SUM(usable),count(*) FROM stage").fetchone(),
        "matched queries",
        db.execute("SELECT SUM(usable),count(*) FROM query").fetchone(),
        flush=True,
    )
    db.close()


@lru_cache(maxsize=1)
def ledger():
    return sqlite3.connect(
        (HERE / "rule-inputs.sqlite").as_uri() + "?mode=ro&immutable=1", uri=True
    )


def external_write(fn):
    # A replayed stage can only compare its return value and argument graph.
    # Reject a function that can write into module state, including mutations
    # accessed through a global container or an imported class.
    locals_ = {arg.arg for arg in ast.walk(fn.args) if isinstance(arg, ast.arg)}
    locals_.update(
        n.id
        for n in ast.walk(fn)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    )

    def root(expr):
        while isinstance(expr, (ast.Subscript, ast.Attribute)):
            expr = expr.value
        return expr.id if isinstance(expr, ast.Name) else None

    mutators = {
        "clear",
        "pop",
        "popitem",
        "update",
        "setdefault",
        "add",
        "append",
        "extend",
        "remove",
        "discard",
        "insert",
        "reverse",
        "sort",
        "set",
        "replace",
        "write",
        "write_text",
        "write_bytes",
    }
    for n in ast.walk(fn):
        if isinstance(n, (ast.Subscript, ast.Attribute)) and isinstance(
            n.ctx, (ast.Store, ast.Del)
        ):
            if root(n) not in locals_:
                return True
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            if n.func.attr in mutators and root(n.func.value) not in locals_:
                return True
    return False


def added_calls(before, after):
    # A new arbitrary helper could mutate global state in another module. Limit
    # new call targets to pinned read-only query helpers or Python primitives.
    def calls(fn):
        return {ast.unparse(n.func) for n in ast.walk(fn) if isinstance(n, ast.Call)}

    primitives = {
        "all",
        "any",
        "bool",
        "isinstance",
        "issubclass",
        "len",
        "max",
        "min",
        "next",
        "reversed",
        "sorted",
        "str",
        "sum",
        "tuple",
        "type",
        "zip",
    }
    searches = {"find_in_scope", "projection_has_aggregate"}
    return calls(after) - calls(before) - primitives - searches


def changed_sites(sel, chs, apply, equal):
    from case_verify import edit_spans
    from source_fast import paired_tree

    if any(c["old"] != c["new"] for c in chs):
        return None
    if not all(
        c["old"].startswith("sqlglot/optimizer/") and c["old"].endswith(".py")
        for c in chs
    ):
        return None
    spans = edit_spans(sel, chs, apply, equal)
    if not spans:
        return None
    forbidden = {
        "globals",
        "locals",
        "setattr",
        "delattr",
        "exec",
        "eval",
        "open",
        "__import__",
        "sys",
        "os",
        "random",
        "time",
        "gc",
        "inspect",
        "threading",
        "setrecursionlimit",
        "environ",
        "__dict__",
        "_hash",
        "cache_clear",
    }
    for c in chs:
        path = c["old"]
        raw, old = sel.source(path)
        updated = apply(raw, c)[0]

        new = paired_tree(c, old, raw, updated)

        def bodies(a, b):
            for x, y in zip(a, b):
                if equal(x, y):
                    continue
                if isinstance(x, ast.ClassDef):
                    if not bodies(x.body, y.body):
                        return False
                    continue
                if (
                    not isinstance(x, ast.FunctionDef)
                    or x.decorator_list
                    or not x.body
                    or not y.body
                    or added_calls(x, y)
                    or external_write(x)
                    or external_write(y)
                ):
                    return False
                for f in (x, y):
                    for n in ast.walk(f):
                        if isinstance(
                            n,
                            (
                                ast.Global,
                                ast.Nonlocal,
                                ast.Yield,
                                ast.YieldFrom,
                                ast.Import,
                                ast.ImportFrom,
                                ast.Await,
                            ),
                        ) or (
                            isinstance(n, (ast.Name, ast.Attribute))
                            and getattr(n, "id", getattr(n, "attr", "")) in forbidden
                        ):
                            return False
                # Defining a function evaluates its header once. Only its body differs.
                spans[path].difference_update(range(x.lineno, x.body[0].lineno))
            return True

        if not bodies(old.body, new.body):
            return None
    return spans


def probe(payload):
    root, jobs, out, budget = payload
    import importlib, signal

    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.setrecursionlimit(12000)
    signal.setitimer(signal.ITIMER_PROF, budget)
    db = sqlite3.connect(
        (HERE / "rule-inputs.sqlite").as_uri() + "?mode=ro&immutable=1", uri=True
    )
    from tests.helpers import load_sql_fixture_pairs
    from case_profile import digest
    from query_case import source_environment, table_digest
    from sqlglot import parse_one

    rows = profile_store.get("cases")["cases"]
    tables = {
        r["key"][1]: r["table"]
        for r in rows
        if r["kind"] == "worker" and r["key"][0] == "tpcds" and "table" in r
    }
    scope = None
    sqls = None
    by_phase = {
        (r["kind"], r["key"][1]): r
        for r in rows
        if isinstance(r["key"], list) and r["key"][0] == "tpcds"
    }
    for q, ids in sorted(jobs.items(), key=lambda it: len(it[1])):
        different = -1 in ids
        if -2 in ids or -3 in ids:
            from sqlglot import transpile
            from case_profile import expression_key, sql_key

            if sqls is None:
                sqls = list(load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))
            meta, sql, _ = sqls[q]
            if (
                -2 in ids
                and sql_key(transpile(sql, write="duckdb")[0])
                != by_phase["duck", q]["statement"]
            ):
                return
            if (
                -3 in ids
                and expression_key(parse_one(sql)) != by_phase["parse", q]["serialized"]
            ):
                different = True
        for n in sorted(i for i in ids if i >= 0):
            if different:
                break
            mod, fn, blob, sig, opts, usable = db.execute(
                "SELECT mod,fn,arg,outcome,opts,usable FROM stage WHERE id=?", (n,)
            ).fetchone()
            if not usable:
                different = True
                break
            arg, kw = pickle.loads(zlib.decompress(blob))
            fun = getattr(importlib.import_module(mod), fn)
            result = fun(arg, **kw)
            if expression_digest(result) != sig or options_sig(kw) != opts:
                different = True
                break
        if different:
            if scope is None:
                scope = source_environment(
                    root, digest((root / "tests/test_executor.py").read_bytes())
                )
                sqls = list(load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))
            meta, sql, _ = sqls[q]
            table = scope["mp_execute"](
                pickle.loads(pickle.dumps(parse_one(sql))), meta
            )
            if q in tables:
                if table is None or table_digest(table) != tables[q]:
                    return
            elif table is not None:
                return
    out.touch()


def refine(sel, chs, selected, apply, equal, checkout, remaining):
    # A native optimizer call already produced a different final graph for this patch.
    # An equality-only replay cannot certify this test. The actual result may still pass.
    if chs and chs[0].get("_optimizer_graph_changed"):
        return selected
    number = sel.by_name.get(TEST)
    if (
        remaining < 2
        or number not in selected
        or sel.unknown
        or not chs
        or sys.flags.hash_randomization
        or number in sel.confirmed_failures
    ):
        return selected
    for ch in chs:
        if ch["old"] in sel.files and number in sel.sets[sel.files[ch["old"]]]:
            return selected
    if any(
        c["old"] != DIALECT and not c["old"].startswith("sqlglot/optimizer/")
        for c in chs
    ):
        return selected
    from verification_budget import pack_ms

    durations = {n: round(1000 * sel.by_number[n]["seconds"]) for n in selected}
    saving = pack_ms(durations.values()) - pack_ms(
        ms for n, ms in durations.items() if n != number
    )
    if saving < 1400:
        return selected
    db = ledger()
    spans = None
    if len(chs) == 1 and chs[0]["old"] == DIALECT:
        import field_cases

        jobs = field_cases.plan(sel, chs[0], apply, equal, db)
        if jobs is None:
            return selected
    else:
        spans = changed_sites(sel, chs, apply, equal)
        if not spans:
            import rule_modules

            jobs = rule_modules.broad(sel, chs, apply, equal, db)
            if jobs is None:
                return selected
    if spans:
        from case_verify import phase_mask

        bits = phase_mask(spans)
        cases = profile_store.get("cases")["cases"]
        required = set()
        while bits:
            bit = bits & -bits
            bits ^= bit
            r = cases[bit.bit_length() - 1]
            k = r["kind"]
            if k in {"shared", "executor_setup"}:
                return selected
            if k in {"worker", "parse", "duck", "optimize"} and r["key"][0] == "tpcds":
                if k != "optimize":
                    return selected
                required.add(r["key"][1])
        jobs = {}
        for p, lines in spans.items():
            for line in lines:
                for n, q in db.execute(
                    "SELECT r.id,s.q FROM reached r JOIN stage s ON s.id=r.id WHERE r.src=? AND r.line=?",
                    (p, line),
                ):
                    jobs.setdefault(q, set()).add(n)
        if set(jobs) != required:
            return selected
        if any(
            not db.execute("SELECT usable FROM query WHERE q=?", (q,)).fetchone()[0]
            for q in jobs
        ):
            return selected
    with tempfile.TemporaryDirectory(prefix="br-rule-") as d:
        tmp = Path(d)
        root = tmp / "repo"
        out = tmp / "complete"
        checkout(root, chs, sel)
        budget = min(7.0, 0.70 * saving / 1000, remaining - 0.5)
        if budget < 0.5:
            return selected
        forkprobe.run(
            (root, jobs, out, budget),
            {"PYTHONPATH": str(root)},
            probe,
            min(remaining, 2 * budget + 1),
        )
        return selected - {number} if out.exists() else selected


if __name__ == "__main__":
    build(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
