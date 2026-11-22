"""Verify executor tests at the actual SQL, transport and optimizer boundaries."""

import ast
import functools
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import signal
import sys
import tempfile
import time

from case_profile import EXEC, sql_key
import case_verify
import executor_stages
import forkprobe
from source_fast import paired_tree

CASES = (
    EXEC + "test_execute_tpcds",
    EXEC + "test_execute_tpch",
    EXEC + "test_optimized_tpch",
)
BACK = {"shared", "setup", "transport", "worker_before", "worker_after", "assert"}
DRIVER = "tests/test_executor.py"


@functools.lru_cache(maxsize=1)
def reference():
    rows = case_verify.index()["cases"]
    return {
        (r["kind"], *r["key"]): r
        for r in rows
        if isinstance(r["key"], list) and r["key"][0] in {"tpcds", "tpch"}
    }


def same_shell(a, b, equal):
    return type(a) is type(b) and all(
        equal(getattr(a, k), getattr(b, k)) for k in a._fields if k != "body"
    )


def private_frontend(path, a, b, equal):
    """Changes confined to an existing parser/tokenizer implementation class.

    A token enum or module export can be observed outside the SQL parsing entry points.
    Leave those changes to the ordinary schedule.
    """
    allowed = path in (
        "sqlglot/parser.py",
        "sqlglot/tokens.py",
        "sqlglot/tokenizer_core.py",
    ) or path.startswith("sqlglot/parsers/")
    if not allowed or len(a.body) != len(b.body):
        return False
    for x, y in zip(a.body, b.body):
        if equal(x, y):
            continue
        if not (
            isinstance(x, ast.ClassDef)
            and same_shell(x, y, equal)
            and (x.name.endswith(("Parser", "Tokenizer")) or x.name == "TokenizerCore")
        ):
            return False
    return True


def dialect_class_changes(path, before, after, equal):
    """Allow ordinary class data and method definitions, with unchanged module bindings."""
    if not (
        path == "sqlglot/generator.py"
        or path.startswith(
            ("sqlglot/parsers/", "sqlglot/generators/", "sqlglot/dialects/")
        )
    ):
        return False

    def declarations(module):
        return [
            n for n in module.body if not isinstance(n, (ast.Import, ast.ImportFrom))
        ]

    def imports(module):
        return {
            ast.dump(n)
            for n in module.body
            if isinstance(n, (ast.Import, ast.ImportFrom))
        }

    # Added imports of project names are executed in the actual patched process.
    # Rebinding or removing a baseline import can change unrelated module state.
    old_imports = imports(before)
    if not old_imports <= imports(after) or any(
        not n.module or not n.module.startswith(("sqlglot", "typing")) or n.level
        for n in after.body
        if isinstance(n, ast.ImportFrom) and ast.dump(n) not in old_imports
    ):
        return False
    left, right = declarations(before), declarations(after)
    if len(left) != len(right):
        return False

    def class_body(a, b):
        if not isinstance(a, ast.ClassDef) or not same_shell(a, b, equal):
            return False
        for node in [*a.body, *b.body]:
            if not isinstance(
                node,
                (
                    ast.ClassDef,
                    ast.FunctionDef,
                    ast.Assign,
                    ast.AnnAssign,
                    ast.Pass,
                    ast.Expr,
                ),
            ):
                return False
            if isinstance(node, ast.Expr) and not (
                isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                return False
            if isinstance(node, ast.Assign) and any(
                not isinstance(t, ast.Name) for t in node.targets
            ):
                return False
            if isinstance(node, ast.AnnAssign) and not isinstance(
                node.target, ast.Name
            ):
                return False
        protected = {
            "generate",
            "generator",
            "__init__",
            "__new__",
            "__getattribute__",
            "__getattr__",
            "__init_subclass__",
            "__setattr__",
        }

        def protected_methods(node):
            return {
                v.name: v
                for v in node.body
                if isinstance(v, ast.FunctionDef) and v.name in protected
            }

        x, y = protected_methods(a), protected_methods(b)
        return x.keys() == y.keys() and all(equal(v, y[k]) for k, v in x.items())

    return all(equal(a, b) or class_body(a, b) for a, b in zip(left, right))


def token_declarations(a, b, equal):
    from itertools import zip_longest

    for x, y in zip_longest(a.body, b.body):
        if equal(x, y):
            continue
        if (
            not isinstance(x, ast.ClassDef)
            or not isinstance(y, ast.ClassDef)
            or not same_shell(x, y, equal)
        ):
            return False
        if x.name == "TokenType":

            def enum_values(c):
                return [
                    v
                    for v in c.body
                    if not (
                        isinstance(v, ast.Assign)
                        and len(v.targets) == 1
                        and isinstance(v.targets[0], ast.Name)
                        and isinstance(v.value, ast.Call)
                        and isinstance(v.value.func, ast.Name)
                        and v.value.func.id == "auto"
                        and not v.value.args
                        and not v.value.keywords
                    )
                ]

            if not equal(enum_values(x), enum_values(y)):
                return False
        elif x.name not in ("Tokenizer", "TokenizerCore", "_TokenizerBase"):
            return False
    return True


def unvisited_optimizer_edit(sel, change, apply, before):
    """An exact replacement in an unentered branch may join a multi-file query replay."""
    path = change["old"]
    if not path.startswith("sqlglot/optimizer/"):
        return False
    raw, _ = sel.source(path)
    updated, removed, added = apply(raw, change)
    if len(removed) != 1 or removed != added or raw.count("\n") != updated.count("\n"):
        return False
    (line,) = removed
    # Limit this test to a changed executable statement below the header.
    # Source, compiler metadata, declarations and import-time expressions have separate consumers.
    def ordinary_body(node):
        return any(
            isinstance(fn, ast.FunctionDef) and not fn.decorator_list
            and fn.body and fn.body[0].lineno <= line <= fn.end_lineno
            and any(
                isinstance(stmt, (ast.Assign, ast.Expr, ast.Return, ast.Raise, ast.AugAssign))
                and stmt.lineno == line == stmt.end_lineno
                and not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant))
                for stmt in ast.walk(fn)
            )
            for fn in ast.walk(node)
        )
    parsed = paired_tree(change, before, raw, updated)
    if not ordinary_body(before) or not ordinary_body(parsed):
        return False
    import code_reach
    spans = code_reach.spans(sel, [change], apply)
    return bool(spans and spans.get(path) == removed and executor_stages.query(spans, sel.mapping) == {})


def constraints(sel, changes, apply, equal):
    """Identify work which can affect the expensive consumer even with an identical graph."""
    all_opt = False
    check_backend = False
    touched = set()
    native = set()
    visit = set()
    # Only consume the immutable table/driver footprints when the patch changes production code.
    for c in changes:
        path = c["old"]
        if (
            path != c["new"]
            or not path.startswith("sqlglot/")
            or not path.endswith(".py")
        ):
            return None
        raw, before = sel.source(path)
        new = apply(raw, c)[0]
        after = paired_tree(c, before, raw, new)
        if equal(before, after):
            # The native test and its dependencies can themselves read source text.
            if raw != new:
                return None
            continue
        changed = "\n".join(
            line[1:]
            for h in c["hunks"]
            for line in h["lines"]
            if line[:1] in ("+", "-")
        )
        # Avoid process-wide side effects, hook installation, runtime inspection and I/O.
        if re.search(
            r"\b(?:setattr|delattr|settrace|setprofile|environ|monkeypatch|open|globals|nonlocal|global|tracemalloc|sleep|__dict__|__code__|sys|signal|random)\b",
            changed,
        ):
            return None
        if len(changes) > 1 and unvisited_optimizer_edit(sel, c, apply, before):
            continue
        lines = case_verify.changed_lines(before, after, equal)
        if lines is not None:
            reached = executor_stages.query({path: lines}, sel.mapping)
            if reached is not None and not {
                "shared",
                "setup",
                "transport",
            }.intersection(reached):
                for phase in ("worker_before", "worker_after", "assert"):
                    native.update(reached.get(phase, ()))
                for affected in reached.values():
                    visit.update(affected)
                touched.update(reached.get("optimize", ()))
                continue
        if path in (
            "sqlglot/tokens.py",
            "sqlglot/tokenizer_core.py",
        ) and token_declarations(before, after, equal):
            all_opt = True
            check_backend = True
            continue
        if private_frontend(path, before, after, equal):
            # Parsing and transpilation occur in the parent and in optimization.
            # All their input SQLs are replayed, even for class-table changes.
            all_opt = True
            continue
        if dialect_class_changes(path, before, after, equal):
            if path.startswith("sqlglot/dialects/"):
                # Dialects implement optimizer and planner settings as well as SQL generation.
                # Only use this extension for dialects absent from post-optimizer execution.
                phases = executor_stages.query({path: None}, sel.mapping)
                if phases is not None and set(phases) & {
                    "worker_after",
                    "assert",
                    "setup",
                    "transport",
                    "worker_before",
                }:
                    return None
            all_opt = True
            check_backend |= path == "sqlglot/generator.py" or path.startswith(
                ("sqlglot/generators/", "sqlglot/dialects/")
            )
            continue
        if path.startswith("sqlglot/expressions/"):
            import expression_schema

            nr = sel.by_name.get(CASES[0])
            if nr is not None and not expression_schema.verify(
                sel, [c], {nr}, apply, equal
            ):
                all_opt = True
                check_backend = True
                continue
        return None
        # Only the TPC-DS worker has per-query native downstream coverage.
    return all_opt, touched, check_backend, native, visit


def verify_rows(root, family, i, queries, scope, raw_sql):
    """Run the real worker and its pinned DataFrame comparison for a changed query."""
    from sqlglot import parse_one
    from query_case import table_digest
    from tests import test_executor as mod

    meta, sql, _ = queries[i]
    expression = pickle.loads(pickle.dumps(parse_one(sql), protocol=4))
    result = scope["mp_execute"](expression, meta)
    if result is None:
        raise ValueError("Missing worker result")
    table = pickle.loads(pickle.dumps(result, protocol=4))
    key = (family, i)
    expected = reference()["worker", *key].get("table")
    if (
        sql_key(raw_sql) == reference()["duck", *key]["statement"]
        and table_digest(table) == expected
    ):
        return
    from executor_shapes import wrong_shape
    if wrong_shape(family, i, raw_sql, table):
        raise ValueError("The patched worker's result has the wrong DataFrame shape")
    # Different representations or generated SQL can still satisfy the actual assertion.
    import duckdb

    conn = duckdb.connect()
    folder = "tpc-ds" if family == "tpcds" else "tpc-h"
    try:
        for name, columns in scope["_schema"].items():
            csv_file = str(
                root / "tests" / "fixtures" / "optimizer" / folder / (name + ".csv.gz")
            )
            conn.execute(
                f"CREATE VIEW {name} AS SELECT * FROM READ_CSV('{csv_file}', delim='|', header=True, columns={columns})"
            )
        ctx = mod.TestExecutor(
            "test_execute_tpcds" if family == "tpcds" else "test_execute_tpch"
        )
        ctx.cache = {}
        setattr(ctx, family + "_sqls", queries)
        setattr(ctx, family + "_conn", conn)
        ctx.subtestHelper(i, table, tpch=family == "tpch")
    finally:
        conn.close()


def run(args):
    (
        root,
        planned,
        optimize_ids,
        check_backend,
        force_native,
        visit,
        questionable,
        ledger,
        omitted,
        marker,
        cpu,
    ) = args
    os.chdir(root)
    sys.path.insert(0, str(root))
    start_cpu = time.process_time()
    signal.setitimer(signal.ITIMER_PROF, cpu)
    from method_probe import MissingReferencePackage

    monitor = MissingReferencePackage(
        set(json.loads(Path(__file__).with_name("reference-imports.json").read_bytes()))
    )
    sys.meta_path.append(monitor)
    from sqlglot import parse_one, transpile
    from tests.helpers import load_sql_fixture_pairs
    from checkpoint_graph import expression_digest
    from query_case import source_environment
    import sqlglot.executor as engine

    rows = reference()
    driver = json.loads(Path(__file__).with_name("executor-queries.json").read_bytes())[
        "driver_sha"
    ]
    if hashlib.sha256((root / DRIVER).read_bytes()).hexdigest() != driver:
        return
    native_opt = engine.optimize

    class Matched(BaseException):
        pass

    class ChangedGraph(BaseException):
        pass

    def compare(*aa, **kw):
        if expression_digest(native_opt(*aa, **kw)) == expected:
            raise Matched
        raise ChangedGraph

    with marker.open("w", buffering=1) as out:
        for suite in planned:
            ds = suite == CASES[0]
            family = "tpcds" if ds else "tpch"
            directory = "tpc-ds" if ds else "tpc-h"
            data = list(
                load_sql_fixture_pairs(f"optimizer/{directory}/{directory}.sql")
            )
            if len(data) != sum(k[0] == "parse" and k[1] == family for k in rows):
                continue
            try:
                scope = source_environment(root, driver, family)
                scope["_tables"] = {
                    k: scope["Table"](columns=v, rows=[])
                    for k, v in scope["_schema"].items()
                }
                needed = set(force_native) if ds else set()
                if ds and check_backend:
                    import backend_sql

                    needed.update(backend_sql.replay())
                true_tables = None
                sql_text = {}
                engine.optimize = compare
                indices = list(range(len(data)))
                if ds and questionable in indices:
                    indices.remove(questionable)
                    indices.insert(0, questionable)
                if ds and visit is not None:
                    indices = [i for i in indices if i in visit or i in needed]
                extended = False
                for position,i in enumerate(indices):
                    # Spend a final small installment only after most native cases
                    # have completed and the original CPU slice is nearly exhausted.
                    elapsed = time.process_time()-start_cpu
                    if (not extended and cpu >= 3 and 4*position >= 3*len(indices)
                            and elapsed >= 0.67*cpu):
                        extension=min(2.25,0.35*cpu)
                        signal.setitimer(signal.ITIMER_PROF,max(0.001,cpu+extension-elapsed))
                        extended=True
                    meta, sql, optimized = data[i]
                    if ds and visit is not None and i not in visit and i not in needed:
                        continue
                    key = (family, i)
                    p = rows["parse", *key]
                    if (
                        sql_key(sql) != p["sql"]
                        or bool(meta.get("execute")) != p["execute"]
                    ):
                        raise ValueError("Unrecognized fixture")
                    worker_ok = ds and (i,"worker") in ledger
                    duck_ok = ds and (i,"duck") in ledger
                    if worker_ok:
                        needed.discard(i)
                    if ds and not worker_ok:
                        obj = parse_one(sql)
                        raw = pickle.dumps(obj, protocol=4)
                        expr = pickle.loads(raw)
                        if (
                            p["execute"]
                            and hashlib.sha256(raw).hexdigest() != p["serialized"]
                        ):
                            needed.add(i)
                        if (
                            i not in needed
                            and p["execute"]
                            and (i,"optimize") not in ledger
                            and (optimize_ids is None or i in optimize_ids)
                        ):
                            opt = rows.get(("optimize", *key))
                            if not opt:
                                raise ValueError("No native optimizer checkpoint")
                            expected = opt["graph"]
                            try:
                                scope["mp_execute"](expr, meta)
                            except Matched:
                                pass
                            except ChangedGraph:
                                needed.add(i)
                            else:
                                raise ValueError(
                                    "Worker did not reach expected checkpoint"
                                )
                    w = rows["worker", *key]
                    if ("table" in w or not ds) and (not duck_ok or i in needed):
                        sql_text[i] = transpile(sql, write="duckdb")[0]
                        if sql_key(sql_text[i]) != rows["duck", *key]["statement"]:
                            if ds:
                                needed.add(i)
                            else:
                                raise ValueError("Different reference SQL")
                    if (
                        not ds
                        and sql_key(transpile(optimized, write="duckdb")[0])
                        != rows["optimized", *key]["statement"]
                    ):
                        raise ValueError("Different optimized reference SQL")
                    if ds and i in needed and meta.get("execute"):
                        engine.optimize = native_opt
                        try:
                            if true_tables is None:
                                true_tables = source_environment(root, driver, family)
                            verify_rows(root, family, i, data, true_tables, sql_text[i])
                        finally:
                            engine.optimize = compare
                engine.optimize = native_opt
                if not monitor.missing and not any(x in sys.modules for x in omitted):
                    print(suite, file=out)
            except Exception:
                continue
            finally:
                engine.optimize = native_opt


def refine(sel, changes, selected, apply, equal, checkout, remaining):
    # The native post-optimizer footprint is specific to the TPC-DS worker.
    # Optimized TPC-H only transpiles both source strings and compares query results.
    if remaining < 1.2 or sel.unknown:
        return selected
    names = [
        n
        for n in (CASES[0], CASES[2])
        if sel.by_name.get(n) in selected - sel.confirmed_failures
    ]
    if not names or any(
        c["old"] != c["new"]
        or not c["old"].startswith("sqlglot/")
        or not c["old"].endswith(".py")
        for c in changes
    ):
        return selected
    from verification_budget import catalog_price

    price = getattr(sel, "verification_cost", None)
    if price is None:
        price = sel.verification_cost = catalog_price(sel.by_number)
    ids = frozenset(selected)
    saving = price(ids) - price(ids.difference(sel.by_name[n] for n in names))
    if saving < 1300:
        return selected
    from executor_dormant import unused_modules

    unused = unused_modules()
    ignored = {
        c["old"]
        for c in changes
        if unused.get(c["old"]) is not None
        and unused[c["old"]] == sel.mapping["source_hashes"].get(c["old"])
        and sel.by_name.get(CASES[0])
        not in (sel.sets[sel.files[c["old"]]] if c["old"] in sel.files else ())
    }
    if len(ignored) == len(changes) and CASES[0] in names:
        # Every input file was absent throughout the pristine native worker and
        # its parent. Existing files retain their names. Validate patched syntax
        # and permit ordinary class-local edits only.
        for c in changes:
            path = c["old"]
            raw, before = sel.source(path)
            new = apply(raw, c)[0]
            after = paired_tree(c, before, raw, new)
            if not dialect_class_changes(path, before, after, equal):
                break
        else:
            return set(selected) - {sel.by_name[CASES[0]]}
    active = [c for c in changes if c["old"] not in ignored]
    guard = constraints(sel, changes, apply, equal)
    if guard is not None:
        ignored = set()
    else:
        guard = constraints(sel, active or changes, apply, equal)
    if guard is None:
        return selected
    full, indices, check_backend, force_native, visit = guard
    if force_native:
        # These whole native workers have to complete before a pass can be emitted.
        # A probe that is unlikely to fit the current slice retains the candidate.
        complete = changes[0].get("_native_query_progress", ())
        must_run = {i for i in force_native if (i,"worker") not in complete}
        expected = sum(row.get("cpu_ms", 0) for row in case_verify.index()["cases"]
                       if row["kind"]=="worker" and row["key"][0]=="tpcds" and row["key"][1] in must_run)
        if expected > 1.6*min(7500,remaining*1000):
            return selected
    omitted = tuple(f[:-3].replace("/", ".") for f in ignored)
    # The reference-SQL test has additional setup paths. Restrict it to parser-only edits.
    if CASES[2] in names and not all(
        c["old"] == "sqlglot/parser.py" or c["old"].startswith("sqlglot/parsers/")
        for c in changes
    ):
        names.remove(CASES[2])
    if not names:
        return selected
    with tempfile.TemporaryDirectory(prefix="query-frontier-") as tmp:
        root = Path(tmp) / "src"
        marker = Path(tmp) / "complete"
        checkout(root, changes, sel)
        cpu = min(7.5, 0.69 * saving / 1000, remaining - 0.2)
        forkprobe.run(
            (
                root,
                names,
                None if full else indices,
                check_backend,
                force_native,
                None if full else visit,
                changes[0].get("_different_table"),
                changes[0].get("_native_query_progress", ()),
                omitted,
                marker,
                cpu,
            ),
            {
                "PYTHONPATH": str(root),
                "SKIP_INTEGRATION": "0",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
            run,
            min(sel.verify_wall_deadline - time.monotonic(), 1.5 + 1.85 * cpu),
        )
        done = marker.read_text().splitlines() if marker.is_file() else []
        return set(selected) - {sel.by_name[name] for name in done if name in names}
