"""Verify expensive native tests at the granularity of their independent inputs."""

import ast
import json
import os
import pickle
import signal
import sys
import unittest
import fixture_cases
import profile_store
from functools import lru_cache

from case_profile import EXEC, MEMORY, PIN, STRESS, SUITES, expression_key, sql_key

PHASE = {"parse", "worker", "duck", "optimized"}


@lru_cache(maxsize=1)
def index():
    data = profile_store.get("cases")
    if data["pin"] != PIN:
        raise ValueError("Wrong case map revision")
    return data


def changed_lines(old, new, equal):
    """Locate existing statements the old execution must reach before it can diverge."""
    lines = set()
    simple = (
        ast.Return,
        ast.Assert,
        ast.Expr,
        ast.Assign,
        ast.AnnAssign,
        ast.AugAssign,
        ast.Raise,
    )

    def mark(node):
        lines.update(range(node.lineno, node.end_lineno + 1))

    def same_shell(a, b):
        return type(a) is type(b) and all(
            equal(getattr(a, key), getattr(b, key))
            for key in a._fields
            if key != "body"
        )

    def statements(left, right, owner):
        if len(left) != len(right):
            mark(owner)
            return
        for a, b in zip(left, right):
            if equal(a, b):
                continue
            if (
                isinstance(a, ast.If)
                and isinstance(b, ast.If)
                and equal(a.test, b.test)
            ):
                statements(a.body, b.body, a)
                statements(a.orelse, b.orelse, a)
            elif (
                isinstance(a, ast.If)
                and isinstance(b, ast.If)
                and equal(a.body, b.body)
                and equal(a.orelse, b.orelse)
            ):
                lines.add(a.lineno)
                mark(a.test)
            elif (
                isinstance(a, ast.FunctionDef)
                and same_shell(a, b)
                and not a.decorator_list
                and bindings(a) == bindings(b)
                and {
                    n.id
                    for n in ast.walk(a)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                }
                == {
                    n.id
                    for n in ast.walk(b)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                }
            ):
                # Building the enclosing parser does not run this closure.
                # Keep the outer span if a change could alter its captured names.
                statements(a.body, b.body, a)
            elif isinstance(a, simple) and type(a) is type(b):
                mark(a)
            else:
                mark(owner)

    unsafe = (
        ast.Yield,
        ast.YieldFrom,
        ast.Await,
        ast.Global,
        ast.Nonlocal,
        ast.Import,
        ast.ImportFrom,
    )

    def bindings(fn):
        return {
            (type(n).__name__, n.id)
            for n in ast.walk(fn)
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))
        }

    def visit(before, after):
        if len(before) != len(after):
            return False
        for a, b in zip(before, after):
            if equal(a, b):
                continue
            if isinstance(a, ast.ClassDef) and same_shell(a, b):
                if not visit(a.body, b.body):
                    return False
            elif isinstance(a, ast.FunctionDef) and same_shell(a, b):
                for fn in (a, b):
                    if any(isinstance(n, unsafe) for n in ast.walk(fn)):
                        return False
                if bindings(a) != bindings(b):
                    mark(a)
                else:
                    statements(a.body, b.body, a)
            else:
                return False
        return True

    return lines if visit(old.body, new.body) else None


def edit_spans(selector, changes, apply, equal):
    from source_fast import paired_tree

    result = {}
    for item in changes:
        path = item["old"]
        if (
            path != item["new"]
            or not path.startswith("sqlglot/")
            or not path.endswith(".py")
        ):
            return None
        raw, tree = selector.source(path)
        updated, _, _ = apply(raw, item)

        new = paired_tree(item, tree, raw, updated)
        edits = changed_lines(tree, new, equal)
        if edits is None:
            return None
        result[path] = edits
    return result


def phase_mask(spans):
    bits = 0
    for path, edits in spans.items():
        mapping = profile_store.get("cases", path)
        for line in edits:
            bits |= mapping.get(line, 0)
    return bits


def jobs_for(selector, selected, changes, apply, equal, override=None):
    names = [
        t
        for t in (STRESS, MEMORY, *(EXEC + s for s in SUITES))
        if selector.by_name.get(t) in selected
    ]
    has_fixtures = any(
        selector.by_number[n]["id"].startswith(fixture_cases.PREFIX) for n in selected
    )
    if not names and not has_fixtures:
        return []
    code = [c for c in changes if c["old"].startswith("sqlglot/")]
    files = [c["old"] for c in changes if c not in code]
    if files and code:
        if any(
            not p.startswith("tests/fixtures/") or p not in selector.files
            for p in files
        ):
            return []
        readers = set().union(*(selector.sets[selector.files[p]] for p in files))
        names = [
            n
            for n in names
            if selector.by_name[n] not in readers
            and not (
                n.startswith(EXEC)
                and any(p.startswith("tests/fixtures/optimizer/tpc-") for p in files)
            )
        ]
    spans = (
        override
        if override is not None
        else (edit_spans(selector, code, apply, equal) if code else None)
    )
    fixture_jobs = fixture_cases.jobs(
        selector,
        selected,
        changes,
        spans if not files or not code else None,
        apply,
        equal,
    )
    if spans is None or not names:
        return fixture_jobs
    mask = phase_mask(spans)
    rows = [row for i, row in enumerate(index()["cases"]) if (mask >> i) & 1]
    kinds = {r["kind"] for r in rows}
    if "shared" in kinds:
        return fixture_jobs
    tasks = list(fixture_jobs)
    for name in names:
        if name == STRESS:
            active = [r for r in rows if r["kind"] == "stress"]
        elif name == MEMORY:
            active = [r for r in rows if r["kind"] == "memory"]
        else:
            if "executor_setup" in kinds:
                continue
            family = "tpcds" if name.endswith("tpcds") else "tpch"
            phases = (
                {"duck", "optimized"}
                if name.endswith("optimized_tpch")
                else {"parse", "worker", "duck"}
            )
            active = [r for r in rows if r["kind"] in phases and r["key"][0] == family]
            if not name.endswith("optimized_tpch"):
                all_workers = {
                    tuple(r["key"]): r
                    for r in index()["cases"]
                    if r["kind"] == "worker"
                }
                active = [
                    r
                    for r in active
                    if r["kind"] != "duck" or "table" in all_workers[tuple(r["key"])]
                ]
                full = {tuple(r["key"]) for r in active if r["kind"] == "worker"}
                optimizer_only = all(p.startswith("sqlglot/optimizer/") for p in spans)
                for opt in rows:
                    if (
                        opt["kind"] != "optimize"
                        or opt["key"][0] != family
                        or tuple(opt["key"]) in full
                    ):
                        continue
                    active.append(
                        opt if optimizer_only else all_workers[tuple(opt["key"])]
                    )
        tasks.append(
            (
                name,
                active,
                sum(
                    (2 * r["cpu_ms"] + 4) if r["kind"] == "optimize" else r["cpu_ms"]
                    for r in active
                ),
            )
        )
    return tasks


def check_stress(rows):
    from tests import test_parser as mod
    from sqlglot import exp
    from sqlglot.errors import ParseError

    active = {r["key"] for r in rows}
    all_rows = [r for r in index()["cases"] if r["kind"] == "stress"]
    original = mod.parse_one
    pos = 0

    def checked(sql, **kwargs):
        nonlocal pos
        rec = all_rows[pos]
        pos += 1
        if (
            rec["key"] != pos - 1
            or sql_key(sql) != rec["sql"]
            or kwargs != rec["options"]
        ):
            raise ValueError("Unexpected parser input")
        if rec["key"] in active:
            return original(sql, **kwargs)
        if rec.get("error") == "ParseError":
            raise ParseError(rec["message"])
        if rec.get("result") == "Select":
            return exp.Select()
        raise ValueError("Unsupported parser assertion")

    mod.parse_one = checked
    try:
        res = unittest.TestSuite([mod.TestParser("test_max_nodes")]).run(
            unittest.TestResult()
        )
        return pos == len(all_rows) and res.wasSuccessful() and not res.skipped
    finally:
        mod.parse_one = original


def check_memory(rows):
    from tests import test_generator as mod

    dialects = mod.Dialects
    active = {r["key"] for r in rows}
    mod.Dialects = [d for d in dialects if d.value in active]
    try:
        if len(mod.Dialects) != len(active):
            return False
        res = unittest.TestSuite([mod.TestGenerator("test_generator_memory")]).run(
            unittest.TestResult()
        )
        return res.wasSuccessful() and not res.skipped
    finally:
        mod.Dialects = dialects


def check_queries(root, name, rows, publish=None, native_prefix=False):
    from query_case import source_environment, table_digest
    from sqlglot import parse_one, transpile
    from sqlglot.optimizer import optimize
    from tests.helpers import TPCDS_SCHEMA, TPCH_SCHEMA
    from checkpoint_graph import expression_digest
    from tests.helpers import load_sql_fixture_pairs

    family = "tpcds" if name.endswith("tpcds") else "tpch"
    folder = "tpc-ds" if family == "tpcds" else "tpc-h"
    sqls = list(load_sql_fixture_pairs(f"optimizer/{folder}/{folder}.sql"))
    cache = {}
    ctx = None
    references = {
        r["key"][1]: r
        for r in index()["cases"]
        if r["kind"] == "worker" and r["key"][0] == family
    }

    def receipt(i, phase):
        if family == "tpcds" and publish is not None:
            print(f"{i} {phase}", file=publish, flush=True)

    def native_context():
        nonlocal ctx
        if ctx is None:
            import hashlib
            driver=root / "tests/test_executor.py"
            ctx=source_environment(root, hashlib.sha256(driver.read_bytes()).hexdigest(), family)
        return ctx

    def matches_prefix(expr, meta, expected):
        # Capture the exact schema, tables and options supplied by the native driver.
        import sqlglot.executor as engine
        original = engine.optimize
        scope = native_context()
        class Reached(BaseException):
            pass
        def endpoint(*args, **kwargs):
            value = original(*args, **kwargs)
            raise Reached(expression_digest(value) == expected)
        engine.optimize = endpoint
        try:
            scope["mp_execute"](expr,meta)
        except Reached as result:
            return bool(result.args[0])
        finally:
            engine.optimize = original
        return False

    def table_matches(i, meta, expr):
        nonlocal ctx
        if ctx is None:
            import hashlib

            ctx = source_environment(
                root,
                hashlib.sha256(
                    (root / "tests/test_executor.py").read_bytes()
                ).hexdigest(),
                family,
            )
        result = ctx["mp_execute"](expr, meta)
        matched = result is not None and table_digest(result) == references[i].get("table")
        if matched:
            receipt(i,"worker")
        return matched

    # Run cheaper assertions first. Reuse parsed queries across phases.
    for row in sorted(rows, key=lambda r: r["cpu_ms"]):
        _, i = row["key"]
        meta, sql, optimized = sqls[i]
        kind = row["kind"]
        if kind in ("duck", "optimized"):
            generated = transpile(
                optimized if kind == "optimized" else sql, write="duckdb"
            )[0]
            if sql_key(generated) != row["statement"]:
                return False
            receipt(i,kind)
        else:
            if i not in cache:
                obj = parse_one(sql)
                raw = pickle.dumps(obj, protocol=4)
                cache[i] = (sql_key(sql), expression_key(obj), pickle.loads(raw))
            key, signature, expr = cache[i]
            if kind == "parse":
                if key != row["sql"]:
                    return False
                if signature == row["serialized"]:
                    receipt(i,"parse")
                if row["execute"] and signature != row["serialized"]:
                    if not table_matches(i, meta, expr):
                        return False
            elif kind == "optimize":
                if family == "tpcds" and native_prefix:
                    if matches_prefix(expr, meta, row["graph"]):
                        receipt(i,"optimize")
                        continue
                    if not table_matches(i,meta,expr):
                        return False
                    continue
                graph = optimize(
                    expr,
                    schema=TPCDS_SCHEMA if family == "tpcds" else TPCH_SCHEMA,
                    leave_tables_isolated=True,
                    dialect=None,
                )
                if expression_digest(graph) != row["graph"] and not table_matches(
                    i, meta, expr
                ):
                    return False
            elif kind == "worker":
                if not meta.get("execute") and "table" not in row:
                    # The immutable upstream worker immediately returns None.
                    continue
                if ctx is None:
                    from case_profile import digest

                    driver = root / "tests/test_executor.py"
                    ctx = source_environment(root, digest(driver.read_bytes()), family)
                result = ctx["mp_execute"](expr, meta)
                if result is None or table_digest(result) != row.get("table"):
                    return False
                receipt(i,"worker")
    return True


def invalid_fixture_input(inputs):
    """Check fresh parser inputs without warming state in the full verifier."""
    try:
        pid = os.fork()
    except OSError:
        return False
    if pid == 0:
        try:
            signal.setitimer(signal.ITIMER_PROF, 0.75)
            from sqlglot import parse_one
            from sqlglot.errors import ParseError, TokenError

            try:
                for dialect, sql in inputs:
                    parse_one(sql, read=dialect)
            except (ParseError, TokenError):
                os._exit(2)
        except BaseException:
            pass
        os._exit(0)
    _, status = os.waitpid(pid, 0)
    return os.WIFEXITED(status) and os.WEXITSTATUS(status) == 2


def worker(payload):
    root, tasks, out, budget, costs, *options = payload
    signal.setitimer(signal.ITIMER_PROF, budget)
    os.chdir(root)
    sys.path.insert(0, str(root))
    with open(out, "w", buffering=1, encoding="utf-8") as f:
        from verification_budget import pack_ms

        left = dict(costs)
        for i, (name, rows, _) in enumerate(tasks):
            if i:
                rest = {n for n, _, _ in tasks[i:]}
                saving = pack_ms(left.values()) - pack_ms(
                    ms for n, ms in left.items() if n not in rest
                )
                if saving < max(80, 0.70 * sum(ms for _, _, ms in tasks[i:])):
                    break
            try:
                if name.startswith(fixture_cases.PREFIX):
                    # Every check_file work item first calls parse_one(sql, read=dialect).
                    # A syntax error makes a full local replay unnecessary. Retain the
                    # selected method; only its complete successful run can clear it.
                    inputs = rows[0].get("new_query_inputs", ()) if rows else ()
                    if inputs and invalid_fixture_input(inputs):
                        continue
                    fixtures = fixture_cases.FixtureTests(root)
                    try:
                        passed = fixtures.passed(name, rows)
                    finally:
                        cls = fixtures.mod.TestOptimizer
                        cls.tearDownClass()
                        cls.doClassCleanups()
                        cls.conn.close()
                    passed = passed and not getattr(cls, "tearDown_exceptions", [])
                elif name == STRESS:
                    passed = check_stress(rows)
                elif name == MEMORY:
                    passed = check_memory(rows)
                else:
                    with open(str(out)+".queries","a",encoding="ascii",buffering=1) as progress:
                        passed = check_queries(root,name,rows,progress,bool(options and options[0]))
                if passed:
                    print(json.dumps(name), file=f)
                    left.pop(name, None)
            except Exception:
                continue
