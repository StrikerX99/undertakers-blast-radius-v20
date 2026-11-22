"""Trace reads of immutable base-dialect settings through the real TPC-DS test."""

import ast
import inspect
import os
import pickle
import sqlite3
import sys
from pathlib import Path

DIALECT = "sqlglot/dialects/dialect.py"


def eligible_fields(root):
    tree = ast.parse((root / DIALECT).read_bytes())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Dialect"
    )
    fields = {
        t.id
        for st in cls.body
        if isinstance(st, (ast.Assign, ast.AnnAssign))
        and isinstance(st.value, ast.Constant)
        and type(st.value.value) in (bool, str, int, float)
        for t in (st.targets if isinstance(st, ast.Assign) else [st.target])
        if isinstance(t, ast.Name) and t.id.isupper()
    }
    reject = set()
    for file in (root / "sqlglot").rglob("*.py"):

        def visit(n, eager=True):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                reject.add(n.id)
            if (
                isinstance(n, ast.Constant)
                and type(n.value) is str
                and n.value in fields
            ):
                reject.add(n.value)
            if isinstance(n, ast.Attribute) and (
                eager or not isinstance(n.ctx, ast.Load)
            ):
                reject.add(n.attr)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for x in [*n.decorator_list, n.args, n.returns]:
                    if x:
                        visit(x, eager)
                for x in n.body:
                    visit(x, False)
            else:
                for x in ast.iter_child_nodes(n):
                    visit(x, eager)

        visit(ast.parse(file.read_bytes()))
    return fields - reject


def build(root, out):
    os.chdir(root)
    sys.path.insert(0, str(root))
    import sqlglot.executor as engine
    from sqlglot import Dialect, parse_one
    from sqlglot.dialects import Dialects
    from sqlglot.optimizer.optimizer import RULES
    from tests import test_executor as mod
    from case_profile import expression_key, sql_key
    from query_case import table_digest
    import profile_store

    rows = profile_store.get("cases")["cases"]
    baseline = {
        (r["kind"], r["key"][1]): r
        for r in rows
        if isinstance(r["key"], list) and r["key"][0] == "tpcds"
    }
    fields = eligible_fields(root)
    # Dialect class assembly happens before a worker test starts. Static
    # eligibility above excludes properties consumed during assembly.
    for d in Dialects:
        klass = Dialect.get_or_raise(d)
        _ = klass.generator_class, klass.parser_class, klass.tokenizer_class
    hits = set()
    point = (-1, "setup", "")

    class Watch:
        def __init__(self, key, value):
            self.key, self.value = key, value

        def __get__(self, obj, owner):
            hits.add((self.key, *point))
            return self.value

    originals = {k: vars(Dialect)[k] for k in fields if k in vars(Dialect)}

    def wrapped(fn):
        def call(e, **kwargs):
            nonlocal point
            before = point
            point = (before[0], "rule", fn.__name__)
            try:
                return fn(e, **kwargs)
            finally:
                point = before

        call.__signature__ = inspect.signature(fn)
        return call

    rules = tuple(wrapped(fn) for fn in RULES)
    saved_opt = engine.optimize

    def optimize(*args, **kwargs):
        return saved_opt(*args, rules=rules, **kwargs)

    original_transpile = mod.transpile

    def generated(*args, **kwargs):
        nonlocal point
        before = point
        point = (before[0], "duck", "")
        try:
            val = original_transpile(*args, **kwargs)
            if before[0] >= 0:
                assert sql_key(val[0]) == baseline["duck", before[0]]["statement"]
            return val
        finally:
            point = before

    observed = set()
    try:
        for k, v in originals.items():
            setattr(Dialect, k, Watch(k, v))
        mod.transpile = generated
        engine.optimize = optimize
        tc = mod.TestExecutor
        tc.setUpClass()
        case = tc("test_execute_tpcds")
        try:
            mod.initializer(mod.TPCDS_SCHEMA, tc.tpcds_tables)
            for q, (meta, sql, _) in enumerate(tc.tpcds_sqls):
                point = (q, "parse", "")
                expr = parse_one(sql)
                assert expression_key(expr) == baseline["parse", q]["serialized"]
                arg = pickle.loads(pickle.dumps(expr, protocol=4))
                point = (q, "worker", "")
                result = pickle.loads(
                    pickle.dumps(mod.mp_execute(arg, meta), protocol=4)
                )
                expected = baseline["worker", q]
                assert (result is None) == ("table" not in expected)
                if result is not None:
                    assert table_digest(result) == expected["table"]
                    point = (q, "assert", "")
                    case.subtestHelper(q, result, tpch=False)
                observed.add(q)
            assert observed == {i for kind, i in baseline if kind == "parse"}
        finally:
            point = (-1, "setup", "")
            tc.tearDownClass()
            tc.doClassCleanups()
    finally:
        mod.transpile = original_transpile
        engine.optimize = saved_opt
        for k, v in originals.items():
            setattr(Dialect, k, v)
    db = sqlite3.connect(out)
    db.executescript(
        "CREATE TABLE fields (name TEXT PRIMARY KEY);CREATE TABLE field_use(field TEXT,q INTEGER,phase TEXT,fn TEXT,PRIMARY KEY(field,q,phase,fn)) WITHOUT ROWID;"
    )
    db.executemany("INSERT INTO fields VALUES(?)", [(k,) for k in originals])
    db.executemany("INSERT INTO field_use VALUES(?,?,?,?)", sorted(hits))
    db.commit()
    db.close()
    print(
        "Rehearsed TPC-DS base fields",
        len(originals),
        "observations",
        len(hits),
        flush=True,
    )


def changed_fields(sel, c, apply, equal, db):
    if c.get("_fixture_fields"):
        return c["_fixture_fields"]
    raw, tree = sel.source(DIALECT)
    updated = apply(raw, c)[0]

    from source_fast import paired_tree

    new = paired_tree(c, tree, raw, updated)
    if len(tree.body) != len(new.body):
        return None
    diff = [(a, b) for a, b in zip(tree.body, new.body) if not equal(a, b)]
    if len(diff) != 1:
        return None
    a, b = diff[0]
    if not (
        isinstance(a, ast.ClassDef)
        and isinstance(b, ast.ClassDef)
        and a.name == "Dialect"
        and len(a.body) == len(b.body)
        and all(equal(getattr(a, k), getattr(b, k)) for k in a._fields if k != "body")
    ):
        return None
    fields = set()
    for x, y in zip(a.body, b.body):
        if equal(x, y):
            continue
        if not (
            isinstance(x, (ast.Assign, ast.AnnAssign))
            and type(x) is type(y)
            and all(
                equal(getattr(x, k), getattr(y, k)) for k in x._fields if k != "value"
            )
        ):
            return None
        ts = x.targets if isinstance(x, ast.Assign) else [x.target]
        if not (
            len(ts) == 1
            and isinstance(ts[0], ast.Name)
            and isinstance(x.value, ast.Constant)
            and isinstance(y.value, ast.Constant)
            and type(x.value.value) is type(y.value.value)
        ):
            return None
        name = ts[0].id
        if not db.execute("SELECT 1 FROM fields WHERE name=?", (name,)).fetchone():
            return None
        fields.add(name)
    c["_fixture_fields"] = fields
    return fields


def plan(sel, c, apply, equal, db):
    fields = changed_fields(sel, c, apply, equal, db)
    if fields is None:
        return None
    jobs = {}
    for name in fields:
        for q, phase, fn in db.execute(
            "SELECT q,phase,fn FROM field_use WHERE field=?", (name,)
        ):
            if phase in ("setup", "assert"):
                return None
            ids = jobs.setdefault(q, set())
            if phase == "rule":
                row = db.execute(
                    "SELECT id FROM stage WHERE q=? AND fn=?", (q, fn)
                ).fetchone()
                if not row:
                    return None
                ids.add(row[0])
            else:
                ids.add({"worker": -1, "duck": -2, "parse": -3}[phase])
    return jobs


if __name__ == "__main__":
    build(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
