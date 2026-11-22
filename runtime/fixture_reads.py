"""Record which optimizer fixture rows consult a particular dialect or parser setting."""

import os
import sqlite3
import sys
import unittest
from pathlib import Path

import fixture_cases
import parser_cells


def build(root, output):
    os.chdir(root)
    sys.path.insert(0, str(root))
    from sqlglot.dialects import Dialect, Dialects
    from sqlglot.parser import Parser
    from tests import test_optimizer as mod

    db = sqlite3.connect(output)
    db.executescript(
        "CREATE TABLE fixture_method (method TEXT PRIMARY KEY);"
        "CREATE TABLE fixture_read (owner TEXT,field TEXT,method TEXT,idx INTEGER,"
        "PRIMARY KEY(owner,field,method,idx)) WITHOUT ROWID;"
    )
    methods = fixture_cases.profile()["methods"]
    cases = fixture_cases.profile()["cases"]
    ids = {
        (r["key"][0], r["key"][1], r["key"][2]): i
        for i, r in enumerate(cases)
        if r["kind"] == "fixture"
    }
    outside = {r["key"]: i for i, r in enumerate(cases) if r["kind"] == "outside"}
    for dialect in Dialects:
        dc = Dialect.get_or_raise(dialect)
        _ = dc.parser_class, dc.generator_class, dc.tokenizer_class
    classes, todo = set(), [Parser]
    while todo:
        cls = todo.pop()
        if cls not in classes:
            classes.add(cls)
            todo.extend(cls.__subclasses__())

    # A descriptor returns the original object itself, preserving type, identity,
    # membership, iteration, and aliasing of the reader's set.
    hits = set()
    point = ("", -1)

    class Watch:
        def __init__(self, field, val, parser):
            self.field, self.val, self.parser = field, val, parser

        def __get__(self, instance, owner):
            who = (
                owner.__module__ + "." + owner.__qualname__
                if self.parser
                else "Dialect"
            )
            hits.add((who, self.field, *point))
            return self.val

    fields = [r[0] for r in db.execute("SELECT name FROM fields")]
    originals = [(Dialect, k, vars(Dialect)[k], False) for k in fields]
    for cls in classes:
        for field in parser_cells.data()["fields"]:
            value = vars(cls).get(field)
            if type(value) in (set, frozenset):
                originals.append((cls, field, value, True))

    # Record reads on ordinary generator classes, preserving the scalar value
    # seen by patched SQL generation. Inherited reads report the actual owner.
    import dialect_cells
    import importlib
    captured = []
    for path, class_fields in dialect_cells.catalog()["paths"].items():
        if not path.startswith("sqlglot/generators/"):
            continue
        module = importlib.import_module(path[:-3].replace("/", "."))
        for class_name, possible in class_fields.items():
            cls = getattr(module, class_name)
            for field in possible:
                val = vars(cls).get(field)
                if type(val) in (bool, int, float, str):
                    originals.append((cls, field, val, True))
                    captured.append((path, class_name, field))
    db.execute("CREATE TABLE fixture_generator(path TEXT, owner TEXT, field TEXT, PRIMARY KEY(path,owner,field)) WITHOUT ROWID")
    db.executemany("INSERT INTO fixture_generator VALUES(?,?,?)", captured)

    loader, completion, pool = (
        mod.load_sql_fixture_pairs,
        mod.as_completed,
        mod.ProcessPoolExecutor,
    )
    valid = []
    try:
        for cls, field, value, parser in originals:
            setattr(cls, field, Watch(field, value, parser))
        mod.ProcessPoolExecutor = fixture_cases.LocalPool
        mod.TestOptimizer.setUpClass()
        for method, filelist in methods.items():
            if not filelist:
                continue
            point = (method, outside[method])
            positions, futures, observed = {}, {}, set()

            def fixture(file):
                nonlocal point
                if file not in filelist:
                    yield from loader(file)
                    return
                for i, value in enumerate(loader(file)):
                    item = ids[method, file, i]
                    saved, point = point, (method, item)
                    positions["current"] = point
                    observed.add(item)
                    yield value
                    point = saved

            def submitted(future):
                futures[future] = positions["current"]

            def complete(items):
                nonlocal point
                for f in items:
                    saved, point = point, futures[f]
                    yield f
                    point = saved

            mod.TestOptimizer.pool.seen = submitted
            mod.load_sql_fixture_pairs, mod.as_completed = fixture, complete
            result = mod.TestOptimizer(method).run(unittest.TestResult())
            expected = {v for (m, _, _), v in ids.items() if m == method}
            if result.wasSuccessful() and not result.skipped and expected == observed:
                valid.append(method)
        point = ("", -1)
        mod.TestOptimizer.tearDownClass()
        mod.TestOptimizer.doClassCleanups()
    finally:
        mod.load_sql_fixture_pairs, mod.as_completed, mod.ProcessPoolExecutor = (
            loader,
            completion,
            pool,
        )
        for cls, k, val, _ in originals:
            setattr(cls, k, val)
    db.executemany("INSERT INTO fixture_method VALUES(?)", [(m,) for m in valid])
    db.executemany("INSERT INTO fixture_read VALUES(?,?,?,?)", sorted(hits))
    db.commit()
    db.close()
    print(
        "Rehearsed optimizer fixture readers",
        len(valid),
        "methods",
        len(hits),
        "observations",
        flush=True,
    )


def jobs(sel, methods, changes, apply, equal):
    from field_cases import changed_fields
    from rule_replay import DIALECT, ledger

    db = ledger()
    keys = getattr(sel, "fixture_parser_fields", None)
    opaque = set()
    if len(changes) == 1 and changes[0]["old"] == changes[0]["new"] == DIALECT:
        fields = changed_fields(sel, changes[0], apply, equal, db)
        if not fields:
            return []
        keys = {("Dialect", f) for f in fields}
    elif len(changes) == 1 and changes[0].get("_verified_fixture_settings"):
        import dialect_cells
        spec = dialect_cells.catalog()
        indices = set(spec["reflective"])
        keys = set()
        for path, owner, field, cells, descendants in changes[0]["_verified_fixture_settings"]:
            if cells != ["*"] or not db.execute(
                "SELECT 1 FROM fixture_generator WHERE path=? AND owner=? AND field=?",
                (path, owner, field),
            ).fetchone():
                return []
            keys.update((desc, field) for desc in descendants)
            indices.update(spec["direct"].get(field, ()))
            indices.update(spec["opaque"].get(field, ()))
        if not keys or -1 in indices:
            return []
        opaque.update(spec["names"][i] for i in indices)
    elif keys is not None and all(
        c["old"] == c["new"] and c["old"].startswith("sqlglot/parsers/")
        for c in changes
    ):
        spec = parser_cells.data()
        indices = set(spec["reflective"])
        for _, field in keys:
            indices.update(spec["opaque"].get(field, ()))
            indices.update(spec["direct"].get(field, ()))
        if -1 in indices:
            return []
        opaque.update(spec["names"][i] for i in indices)
    else:
        return []

    supported = {r[0] for r in db.execute("SELECT method FROM fixture_method")}
    schedule = {
        m: set() for m in methods & supported if fixture_cases.PREFIX + m not in opaque
    }
    for key in keys:
        for method, idx in db.execute(
            "SELECT method,idx FROM fixture_read WHERE owner=? AND field=?", key
        ):
            if not method:
                return []
            if method in schedule:
                schedule[method].add(idx)
    cases = fixture_cases.profile()["cases"]
    result = []
    for method, indices in sorted(schedule.items()):
        rows = [cases[i] for i in sorted(indices)]
        cost = sum(r["cpu_ms"] for r in rows)
        if any(r["kind"] == "outside" for r in rows):
            cost = (
                sum(
                    r["cpu_ms"]
                    for r in cases
                    if r["kind"] == "fixture" and r["key"][0] == method
                )
                + cost
            )
        result.append((fixture_cases.PREFIX + method, rows, cost))
    return result


if __name__ == "__main__":
    root, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    if len(sys.argv) == 4 and sys.argv[3] == "time-setup":
        import time

        os.chdir(root)
        sys.path.insert(0, str(root))
        start = time.process_time()
        group = fixture_cases.FixtureTests(root)
        elapsed = round(1000 * (time.process_time() - start))
        group.mod.TestOptimizer.tearDownClass()
        group.mod.TestOptimizer.doClassCleanups()
        with sqlite3.connect(out) as db:
            db.execute("CREATE TABLE fixture_setup (ms INTEGER)")
            db.execute("INSERT INTO fixture_setup VALUES(?)", (elapsed,))
        print("Cold optimizer fixture setup", elapsed, "CPU ms", flush=True)
    else:
        build(root, out)
