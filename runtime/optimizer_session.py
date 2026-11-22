"""Cost and execute optimizer fixture subsets against a native, lazily seeded DuckDB."""

import ast
import hashlib
import importlib
import importlib.util
import marshal
import os
import sqlite3
import sys
import time
import unittest
from functools import lru_cache
from pathlib import Path

import fixture_cases

DB = Path(__file__).with_name("optimizer-tests.sqlite")
PREFIX = fixture_cases.PREFIX


@lru_cache(maxsize=1)
def profile():
    with sqlite3.connect(DB.as_uri() + "?mode=ro&immutable=1", uri=True) as cx:
        driver, magic, seed, code = cx.execute("SELECT * FROM driver").fetchone()
        if magic != importlib.util.MAGIC_NUMBER:
            raise ValueError("Unknown optimizer driver bytecode version")
        return (
            driver,
            seed,
            marshal.loads(code),
            dict(cx.execute("SELECT name,ms FROM costs")),
            {(m, p, i): us for m, p, i, us in cx.execute("SELECT * FROM rows")},
        )


class Connection:
    def __init__(self, seed):
        self.expected = seed
        self.pending = None
        self.connection = None

    def live(self):
        if self.connection is None:
            self.connection = importlib.import_module("duckdb").connect()
            if self.pending is not None:
                self.connection.execute(self.pending)
        return self.connection

    def execute(self, sql, *args, **kw):
        if (
            self.pending is None
            and self.connection is None
            and sql == self.expected
            and not args
            and not kw
        ):
            self.pending = sql
            return self
        return self.live().execute(sql, *args, **kw)

    def __enter__(self):
        return self.live().__enter__()

    def __exit__(self, *exc):
        return self.live().__exit__(*exc)

    def close(self):
        if self.connection is not None:
            self.connection.close()

    def __getattr__(self, name):
        return getattr(self.live(), name)


class Backend:
    def __init__(self, seed):
        self.seed = seed

    def __getattr__(self, name):
        return getattr(importlib.import_module("duckdb"), name)

    def connect(self, *args, **kwargs):
        if args or kwargs:
            return importlib.import_module("duckdb").connect(*args, **kwargs)
        return Connection(self.seed)


def pandas_equal(*args, **kw):
    return importlib.import_module("pandas.testing").assert_frame_equal(*args, **kw)


def compile_driver(path):
    raw = path.read_bytes()
    tree = ast.parse(raw)
    imports = [
        n
        for n in tree.body
        if isinstance(n, (ast.Import, ast.ImportFrom))
        and ast.unparse(n)
        in ("import duckdb", "from pandas.testing import assert_frame_equal")
    ]
    if len(imports) != 2:
        raise ValueError("Unexpected optimizer imports")
    tree.body = [n for n in tree.body if n not in imports]
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "TestOptimizer"
    )
    setup = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "setUpClass"
    )
    (seed,) = [
        n.args[0].value
        for n in ast.walk(setup)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "execute"
        and len(n.args) == 1
        and isinstance(n.args[0], ast.Constant)
    ]
    return (
        hashlib.sha256(raw).hexdigest(),
        seed,
        compile(tree, str(path), "exec", dont_inherit=True),
    )


def load(root, driver=None):
    path = root / "tests/test_optimizer.py"
    name = "tests.test_optimizer"
    if name in sys.modules:
        return sys.modules[name]
    digest, seed, code = driver or profile()[:3]
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("Changed optimizer driver")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    mod.__dict__.update(duckdb=Backend(seed), assert_frame_equal=pandas_equal)
    parent = importlib.import_module("tests")
    sys.modules[name] = mod
    setattr(parent, "test_optimizer", mod)
    exec(code, mod.__dict__)
    return mod


def build(root, out):
    os.chdir(root)
    sys.path.insert(0, str(root))
    driver = compile_driver(root / "tests/test_optimizer.py")
    if driver[0] != fixture_cases.profile()["driver"]:
        raise ValueError("Unrecognized driver")
    # The only deferred statement is the pristine literal seed. It must really succeed.
    db = importlib.import_module("duckdb").connect()
    try:
        db.execute(driver[1])
    finally:
        db.close()
    mod = load(root, driver)
    mod.ProcessPoolExecutor = fixture_cases.LocalPool
    methods = fixture_cases.profile()["methods"]
    costs, rows = {}, {}
    loader, as_completed = mod.load_sql_fixture_pairs, mod.as_completed
    cls = mod.TestOptimizer
    cls.setUpClass()
    try:
        for method in sorted(methods):
            current = None
            futures = {}
            observed = {}

            def fixture(file):
                nonlocal current
                for i, item in enumerate(loader(file)):
                    key = (method, file, i)
                    current = key
                    start = time.process_time_ns()
                    yield item
                    observed[key] = (
                        observed.get(key, 0) + (time.process_time_ns() - start) // 1000
                    )
                    current = None

            def submitted(f):
                futures[f] = current

            def completed(items):
                for f in items:
                    key = futures[f]
                    start = time.process_time_ns()
                    yield f
                    if key:
                        observed[key] = (
                            observed.get(key, 0)
                            + (time.process_time_ns() - start) // 1000
                        )

            mod.load_sql_fixture_pairs, mod.as_completed = fixture, completed
            cls.pool.seen = submitted
            t = time.process_time()
            result = cls(method).run(unittest.TestResult())
            if result.wasSuccessful() and not result.skipped:
                costs[method] = max(1, round(1000 * (time.process_time() - t)))
                rows.update(observed)
    finally:
        mod.load_sql_fixture_pairs, mod.as_completed = loader, as_completed
        cls.tearDownClass()
        cls.doClassCleanups()
    with sqlite3.connect(out) as cx:
        cx.executescript(
            "CREATE TABLE driver(hash TEXT,magic BLOB,seed TEXT,code BLOB);CREATE TABLE costs(name TEXT PRIMARY KEY,ms INTEGER);CREATE TABLE rows(m TEXT,f TEXT,i INTEGER,us INTEGER);"
        )
        cx.execute(
            "INSERT INTO driver VALUES(?,?,?,?)",
            (
                driver[0],
                importlib.util.MAGIC_NUMBER,
                driver[1],
                marshal.dumps(driver[2]),
            ),
        )
        cx.executemany("INSERT INTO costs VALUES(?,?)", costs.items())
        cx.executemany(
            "INSERT INTO rows VALUES(?,?,?,?)", [(*key, us) for key, us in rows.items()]
        )
    print(
        "Timed",
        len(costs),
        "native optimizer methods and",
        len(rows),
        "fixture cases",
        flush=True,
    )


def tasks(sel, selected, jobs, changes):
    if not changes or not all(
        c["old"] == c["new"] and c["old"].startswith("sqlglot/") for c in changes
    ):
        return jobs
    names = [
        sel.by_number[n]["id"]
        for n in selected
        if sel.by_number[n]["id"].startswith(PREFIX)
    ]
    if not names:
        return jobs
    costs, timings = profile()[3:]
    plans = {name: (rows, ms) for name, rows, ms in jobs}
    for name in names:
        method = name[len(PREFIX) :]
        if method not in costs:
            continue
        full = costs[method]
        if name in plans:
            rows, estimate = plans[name]
            if not rows:
                continue
            if all(r["kind"] == "fixture" and tuple(r["key"]) in timings for r in rows):
                estimate = 10 + sum(timings[tuple(r["key"])] for r in rows) // 1000
                plans[name] = (rows, estimate)
            if full >= estimate:
                continue
        plans[name] = ([{"kind": "outside", "key": method}], full)
    return [(name, rows, ms) for name, (rows, ms) in plans.items()]


if __name__ == "__main__":
    build(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
