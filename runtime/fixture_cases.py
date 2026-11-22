"""Trace and replay the independent fixture rows consumed by TestOptimizer.check_file."""

import ast
import json
import os
import pickle
import sys
import time
import unittest
import zlib
from concurrent.futures import Future
from functools import lru_cache
from pathlib import Path

import profile_store
from case_profile import Observer, PIN

PREFIX = "tests/test_optimizer.py::TestOptimizer::"


class LocalPool:
    # Preserve the serialization boundary of the public spawned worker.
    def __init__(self, **kwargs):
        self.seen = None

    def submit(self, fn, *args, **kwargs):
        future = Future()
        if self.seen:
            self.seen(future)
        try:
            args, kwargs = pickle.loads(pickle.dumps((args, kwargs), protocol=4))
            obj = fn(*args, **kwargs)
            future.set_result(pickle.loads(pickle.dumps(obj, protocol=4)))
        except Exception as exc:
            future.set_exception(exc)
        return future

    def shutdown(self, *args, **kwargs):
        pass


def wrappers(root):
    mod = ast.parse((root / "tests/test_optimizer.py").read_bytes())
    cls = next(
        c for c in mod.body if isinstance(c, ast.ClassDef) and c.name == "TestOptimizer"
    )
    result = {}
    for n in cls.body:
        if not isinstance(n, ast.FunctionDef) or not n.name.startswith("test_"):
            continue
        files = []
        opaque = False
        for node in ast.walk(n):
            if not isinstance(node, ast.Call):
                continue
            symbolic = ast.unparse(node.func)
            if symbolic == "self.check_file":
                if (
                    not node.args
                    or not isinstance(node.args[0], ast.Constant)
                    or any(k.arg == "only" for k in node.keywords)
                ):
                    opaque = True
                else:
                    files.append("optimizer/" + node.args[0].value + ".sql")
            if symbolic == "load_sql_fixture_pairs" and n.name in (
                "test_annotate_funcs",
                "test_annotate_types",
            ):
                if node.args and isinstance(node.args[0], ast.Constant):
                    files.append(node.args[0].value)
            if symbolic in (
                "self.pool.submit",
                "sys.setrecursionlimit",
                "os.environ",
                "time.perf_counter",
            ):
                opaque = True
        if not opaque:
            result[n.name] = sorted(set(files))
    return result


def build(root, out):
    os.chdir(root)
    sys.path.insert(0, str(root))
    from tests import test_optimizer as mod
    from case_profile import digest

    obs = Observer(root)
    mon = sys.monitoring
    tool = mon.COVERAGE_ID
    mon.use_tool_id(tool, "optimizer_fixture_cases")
    mon.register_callback(tool, mon.events.LINE, obs.line)
    obs.case("init", "shared")
    mon.set_events(tool, mon.events.LINE)
    from sqlglot.dialects import Dialect, Dialects

    for name in Dialects:
        dialect = Dialect.get_or_raise(name)
        _ = dialect.parser_class, dialect.generator_class
    mod.ProcessPoolExecutor = LocalPool
    mod.TestOptimizer.setUpClass()
    loader, completion = mod.load_sql_fixture_pairs, mod.as_completed
    try:
        methods = wrappers(root)
        for method, files in list(methods.items()):
            outside = obs.case("outside", method)
            outer = obs.bit
            start_method = time.process_time()
            moments = []
            spans = {}
            holder = {}

            def fixture(filename):
                if filename not in files:
                    yield from loader(filename)
                    return
                for i, triple in enumerate(loader(filename)):
                    row = obs.case("fixture", [method, filename, i])
                    moments.append(row)
                    bit = obs.bit
                    holder["current"] = (row, bit)
                    t = time.process_time()
                    yield triple
                    row["cpu_ms"] = max(1, round(1000 * (time.process_time() - t)))
                    obs.resume(outer)

            def at_submit(future):
                spans[future] = holder["current"]

            def complete(fs):
                for f in fs:
                    row, bit = spans[f]
                    obs.resume(bit)
                    t = time.process_time()
                    yield f
                    row["cpu_ms"] += max(1, round(1000 * (time.process_time() - t)))
                    obs.resume(outer)

            mod.load_sql_fixture_pairs = fixture
            mod.as_completed = complete
            mod.TestOptimizer.pool.seen = at_submit
            res = mod.TestOptimizer(method).run(unittest.TestResult())
            outside["cpu_ms"] = max(
                1,
                round(1000 * (time.process_time() - start_method))
                - sum(r["cpu_ms"] for r in moments),
            )
            if not res.wasSuccessful() or res.skipped:
                del methods[method]
    finally:
        mod.TestOptimizer.doClassCleanups()
        mod.load_sql_fixture_pairs, mod.as_completed = loader, completion
        mon.set_events(tool, 0)
        mon.free_tool_id(tool)
    table = dict(
        pin=PIN,
        methods=methods,
        cases=obs.cases,
        driver=digest((root / "tests/test_optimizer.py").read_bytes()),
        lines={
            p: {str(n): format(bit, "x") for n, bit in rows.items()}
            for p, rows in obs.rows.items()
        },
    )
    out.write_bytes(zlib.compress(json.dumps(table, separators=(",", ":")).encode(), 9))
    print(
        len(table["methods"]),
        "methods,",
        len(table["cases"]),
        "contexts,",
        out.stat().st_size,
        "bytes",
    )


@lru_cache(maxsize=1)
def profile():
    raw = profile_store.get("fixtures")
    if raw["pin"] != PIN:
        raise ValueError("Wrong fixture profile")
    return raw


def jobs(sel, selected, changes, spans, apply, equal):
    methods = [
        name[len(PREFIX) :]
        for n in selected
        if (name := sel.by_number[n]["id"]).startswith(PREFIX)
    ]
    if not methods:
        return []
    tab = profile()
    methods = set(methods) & tab["methods"].keys()
    if not methods:
        return []
    # When only fixture text changes, the immutable loader gives the exact
    # changed input triples at runtime. For production edits use native coverage.
    files = {c["old"] for c in changes}
    if all(f.startswith("tests/fixtures/") for f in files):
        return [
            (PREFIX + m, [{"kind": "delta", "key": sorted(files)}], 120)
            for m in methods
            if files <= {"tests/fixtures/" + f for f in tab["methods"][m]}
        ]
    if spans is None:
        import fixture_reads

        return fixture_reads.jobs(sel, methods, changes, apply, equal)
    bits = 0
    for p, lines in spans.items():
        table = profile_store.get("fixtures", p)
        for line in lines:
            bits |= table.get(line, 0)
    rows = {m: [] for m in methods}
    while bits:
        bit = bits & -bits
        item = tab["cases"][bit.bit_length() - 1]
        bits -= bit
        if item["kind"] == "init":
            return []
        if item["kind"] == "outside":
            if item["key"] in rows:
                rows[item["key"]].append(item)
        elif item["key"][0] in methods:
            rows[item["key"][0]].append(item)
    return [
        (PREFIX + m, r, sum(row["cpu_ms"] for row in r))
        for m, r in sorted(rows.items())
    ]


class FixtureTests:
    def __init__(self, root):
        from optimizer_session import load

        mod = load(root)
        from case_profile import digest

        if (
            digest((root / "tests/test_optimizer.py").read_bytes())
            != profile()["driver"]
        ):
            raise ValueError("Changed test driver")
        self.root = root
        self.mod = mod
        self.loader = mod.load_sql_fixture_pairs
        mod.ProcessPoolExecutor = LocalPool
        mod.TestOptimizer.setUpClass()

    def passed(self, name, rows):
        from tests import helpers

        method = name[len(PREFIX) :]
        files = profile()["methods"][method]

        complete = any(r["kind"] == "outside" for r in rows)

        def filtered(filename):
            if complete or filename not in files:
                return self.loader(filename)
            patched = list(self.loader(filename))
            if rows[0]["kind"] == "delta":
                if "tests/fixtures/" + filename not in rows[0]["key"]:
                    return []
                from collections import Counter

                original = helpers.FIXTURES_DIR
                try:
                    helpers.FIXTURES_DIR = str(
                        Path(os.environ.get("SQLGLOT_ROOT", "/opt/sqlglot"))
                        / "tests/fixtures"
                    )
                    prior = Counter(
                        json.dumps(r, sort_keys=True) for r in self.loader(filename)
                    )
                finally:
                    helpers.FIXTURES_DIR = original
                changed = []
                for r in patched:
                    key = json.dumps(r, sort_keys=True)
                    if prior[key]:
                        prior[key] -= 1
                    else:
                        changed.append(r)
                return changed
            ids = {
                r["key"][2]
                for r in rows
                if r["kind"] == "fixture" and r["key"][1] == filename
            }
            if any(i >= len(patched) for i in ids):
                raise ValueError("Changed fixture sequence")
            return [r for i, r in enumerate(patched) if i in ids]

        self.mod.load_sql_fixture_pairs = filtered
        try:
            result = self.mod.TestOptimizer(method).run(unittest.TestResult())
            return result.wasSuccessful() and not result.skipped
        finally:
            self.mod.load_sql_fixture_pairs = self.loader


if __name__ == "__main__":
    build(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
