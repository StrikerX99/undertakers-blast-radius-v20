"""Collect SQLGlot line footprints at independently checkable native test boundaries.

Run once on the unchanged pinned repository. Each bit is one public test input or
a shared initialization phase. The table contains observed inputs and results
from passing tests, independent of any proposed change.
"""

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
import unittest
import zlib
from collections import defaultdict
from pathlib import Path

PIN = "eb11aec4be8cb2122afe713a4816662c3f0fb2fa"
STRESS = "tests/test_parser.py::TestParser::test_max_nodes"
MEMORY = "tests/test_generator.py::TestGenerator::test_generator_memory"
EXEC = "tests/test_executor.py::TestExecutor::"
SUITES = ("test_execute_tpcds", "test_execute_tpch", "test_optimized_tpch")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sql_key(sql):
    return digest(sql.encode("utf-8"))


def expression_key(expr):
    # This is the serialized object passed to the upstream spawned worker.
    return digest(pickle.dumps(expr, protocol=4))


class Observer:
    def __init__(self, root):
        self.prefix = str(root) + os.sep
        self.rows = defaultdict(dict)
        self.cases = []
        self.files = {}
        self.bit = 0

    def line(self, code, line):
        file = code.co_filename
        if file not in self.files:
            name = file.removeprefix(self.prefix)
            self.files[file] = name if name.startswith("sqlglot/") else None
        name = self.files[file]
        if name is not None:
            rows = self.rows[name]
            rows[line] = rows.get(line, 0) | self.bit
        return sys.monitoring.DISABLE

    def case(self, kind, key):
        item = {"kind": kind, "key": key}
        self.cases.append(item)
        self.bit = 1 << (len(self.cases) - 1)
        sys.monitoring.restart_events()
        return item

    def resume(self, bit):
        self.bit = bit
        sys.monitoring.restart_events()

    def measure(self, kind, key, fn):
        previous = self.bit
        item = self.case(kind, key)
        start = time.process_time()
        try:
            return item, fn()
        finally:
            item["cpu_ms"] = max(1, round(1000 * (time.process_time() - start)))
            self.resume(previous)


def run_baseline(root):
    from query_case import table_digest

    obs = Observer(root)
    mon = sys.monitoring
    tool_id = mon.COVERAGE_ID
    mon.use_tool_id(tool_id, "baseline_cases")
    mon.register_callback(tool_id, mon.events.LINE, obs.line)
    obs.case("shared", "imports")
    mon.set_events(tool_id, mon.events.LINE)
    from tests import (
        test_executor as executor,
        test_parser as parser,
        test_generator as generator,
    )
    import sqlglot.executor as engine
    from checkpoint_graph import expression_digest
    from sqlglot import transpile

    # Include registration and lazy module imports in the shared footprint.
    from sqlglot.dialects import Dialect, Dialects

    for name in Dialects:
        d = Dialect.get_or_raise(name)
        _ = d.parser_class, d.tokenizer_class, d.generator_class
    parser_fn = parser.parse_one
    call_index = 0

    def parser_input(sql, *args, **kwargs):
        nonlocal call_index
        idx = call_index
        call_index += 1
        prev = obs.bit
        row = obs.case("stress", idx)
        row.update(sql=sql_key(sql), options=kwargs)
        start = time.process_time()
        try:
            result = parser_fn(sql, *args, **kwargs)
            row["result"] = type(result).__name__
            return result
        except Exception as exc:
            row.update(error=type(exc).__name__, message=str(exc))
            raise
        finally:
            row["cpu_ms"] = max(1, round(1000 * (time.process_time() - start)))
            obs.resume(prev)

    parser.parse_one = parser_input
    case = unittest.TestSuite([parser.TestParser("test_max_nodes")])
    result = case.run(unittest.TestResult())
    if not result.wasSuccessful() or result.skipped or call_index != 6:
        raise RuntimeError(
            ("parser baseline", result.errors, result.failures, call_index)
        )
    parser.parse_one = parser_fn

    # The memory test has a separate native assertion for each dialect.
    for dialect in Dialects:
        generator.Dialects = [dialect]
        _, result = obs.measure(
            "memory",
            dialect.value,
            lambda: unittest.TestSuite(
                [generator.TestGenerator("test_generator_memory")]
            ).run(unittest.TestResult()),
        )
        if not result.wasSuccessful() or result.skipped:
            raise RuntimeError(
                ("generator baseline", dialect.value, result.errors, result.failures)
            )
    generator.Dialects = Dialects

    tc = executor.TestExecutor
    obs.case("executor_setup", "shared")
    tc.setUpClass()
    obs.case("validation", "baseline assertions")
    try:
        for family in ("tpcds", "tpch"):
            case = tc("test_execute_" + family)
            fixture = getattr(tc, family + "_sqls")
            sch = executor.TPCDS_SCHEMA if family == "tpcds" else executor.TPCH_SCHEMA
            executor.initializer(sch, getattr(tc, family + "_tables"))
            for i, (meta, sql, optimized) in enumerate(fixture):
                key = [family, i]

                def parse_and_send():
                    expr = parser_fn(sql)
                    raw = pickle.dumps(expr, protocol=4)
                    return digest(raw), pickle.loads(raw)

                p, (fingerprint, expression) = obs.measure("parse", key, parse_and_send)
                p.update(
                    sql=sql_key(sql),
                    serialized=fingerprint,
                    execute=bool(meta.get("execute")),
                )
                # Includes transport of the result. Record independent assertion inputs.
                native_opt = engine.optimize

                def optimize(*args, **kwargs):
                    row, expr = obs.measure(
                        "optimize", key, lambda: native_opt(*args, **kwargs)
                    )
                    row["graph"] = expression_digest(expr)
                    return expr

                engine.optimize = optimize
                try:
                    w, table = obs.measure(
                        "worker",
                        key,
                        lambda: pickle.loads(
                            pickle.dumps(
                                executor.mp_execute(expression, meta), protocol=4
                            )
                        ),
                    )
                finally:
                    engine.optimize = native_opt
                if table is not None:
                    w["table"] = table_digest(table)
                if table is not None or family == "tpch":
                    d, statement = obs.measure(
                        "duck", key, lambda: transpile(sql, write="duckdb")[0]
                    )
                    d["statement"] = sql_key(statement)
                if table is not None:
                    case.subtestHelper(i, table, tpch=(family == "tpch"))
                if family == "tpch":
                    d, statement = obs.measure(
                        "optimized",
                        key,
                        lambda: transpile(optimized, write="duckdb")[0],
                    )
                    d["statement"] = sql_key(statement)
        r = unittest.TestSuite([tc("test_optimized_tpch")]).run(unittest.TestResult())
        if not r.wasSuccessful() or r.skipped:
            raise RuntimeError(("tpch baseline", r.failures, r.errors))
    finally:
        tc.tearDownClass()
    mon.set_events(tool_id, 0)
    mon.free_tool_id(tool_id)
    rows = {
        p: {str(l): format(bits, "x") for l, bits in lines.items()}
        for p, lines in obs.rows.items()
    }
    return {"pin": PIN, "cases": obs.cases, "lines": rows}


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("source", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    root, output = args.source.resolve(), args.output.resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    data = run_baseline(root)
    output.write_bytes(
        zlib.compress(json.dumps(data, separators=(",", ":")).encode(), 9)
    )
    print(
        len(data["cases"]),
        "inputs and phases,",
        len(data["lines"]),
        "source files,",
        output.stat().st_size,
        "bytes",
    )


if __name__ == "__main__":
    main()
