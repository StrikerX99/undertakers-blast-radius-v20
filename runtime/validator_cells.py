"""Replay generator-key changes at independent literal validation calls inside unittest methods."""

import ast
import functools
import gc
import hashlib
import importlib
import json
import marshal
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
import zlib

import assertion_replay
import forkprobe

DATA = Path(__file__).with_name("validator-cells.bin")
FORMAT = "ordered-validator-cells-v1"


def literal(node, method):
    probe = ast.FunctionDef(
        name=method.name,
        args=method.args,
        body=[node],
        decorator_list=[],
        type_params=[],
    )
    return assertion_replay.independent(probe)


def build(root, output):
    os.chdir(root)
    sys.path.insert(0, str(root))
    from sqlglot.dialects.dialect import Dialect, Dialects
    from sqlglot.generator import Generator
    from tests.dialects.test_dialect import Validator

    owners = set()
    for dialect in Dialects:
        cls = Dialect.get_or_raise(dialect.value).generator_class
        owners.add(cls)
        cls()  # All normal receivers and their dispatch caches now exist.
    code_groups = {}
    methods = {}
    for file in sorted((root / "tests/dialects").glob("test_*.py")):
        rel = str(file.relative_to(root))
        mod = importlib.import_module(rel[:-3].replace("/", "."))
        for node in ast.parse(file.read_bytes()).body:
            cls = getattr(mod, getattr(node, "name", ""), None)
            if (
                not isinstance(node, ast.ClassDef)
                or not isinstance(cls, type)
                or not issubclass(cls, Validator)
            ):
                continue
            if any(
                getattr(cls, k) is not getattr(Validator, k)
                for k in assertion_replay.HELPERS
            ):
                continue
            if any(
                getattr(cls, k) is not getattr(unittest.TestCase, k)
                for k in ("setUp", "tearDown", "run", "__init__")
            ):
                continue
            for method in node.body:
                if (
                    not isinstance(method, ast.FunctionDef)
                    or method.decorator_list
                    or not method.name.startswith("test_")
                ):
                    continue
                ordinary = [
                    i for i, stmt in enumerate(method.body) if literal(stmt, method)
                ]
                if not ordinary:
                    continue
                # A mock, stateful custom validator or another process can observe a
                # registry without executing a line in this interpreter.
                hazards = {
                    "setattr",
                    "delattr",
                    "patch",
                    "patches",
                    "mock",
                    "subprocess",
                    "get_context",
                    "fork",
                    "Process",
                    "Pool",
                    "getsource",
                    "__dict__",
                    "__code__",
                    "skipTest",
                    "settrace",
                    "setprofile",
                    "setUp",
                    "tearDown",
                    "meta_path",
                }
                if any(
                    isinstance(
                        n, (ast.Global, ast.Nonlocal, ast.Import, ast.ImportFrom)
                    )
                    or isinstance(n, (ast.Name, ast.Attribute))
                    and getattr(n, "attr", getattr(n, "id", "")) in hazards
                    or isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign))
                    and any(
                        isinstance(z, ast.Attribute)
                        and isinstance(z.ctx, ast.Store)
                        and isinstance(z.value, ast.Name)
                        and z.value.id in ("self", "cls")
                        for z in ast.walk(n)
                    )
                    for n in ast.walk(method)
                ):
                    continue
                name = "::".join((rel, cls.__name__, method.name))
                code = getattr(cls, method.name).__code__
                code_groups[code] = True
                methods[name] = cls, method, ordinary
    # All executable reads of the raw TRANSFORMS table outside the main dispatch
    # use the dynamic receiver/key below or conservatively mark the active statement.
    locations = {}
    for file in (root / "sqlglot").rglob("*.py"):
        for node in ast.walk(ast.parse(file.read_bytes())):
            if isinstance(node, ast.Attribute) and node.attr == "TRANSFORMS":
                locations[(str(file), node.lineno)] = "opaque"
    main = Path(Generator.sql.__code__.co_filename)
    base = main.read_text().splitlines()
    for n, line in enumerate(base, 1):
        if "handler = self._dispatch.get(expression.__class__)" in line:
            locations[(str(main), n)] = "expr"
        if "self.TRANSFORMS.get(expression.__class__)" in line:
            locations[(str(main), n)] = "expr"
    for file in (
        (root / "sqlglot/transforms.py"),
        (root / "sqlglot/generators/postgres.py"),
    ):
        for n, line in enumerate(file.read_text().splitlines(), 1):
            if "self.TRANSFORMS.get(type(expression))" in line:
                locations[(str(file), n)] = "expr"
            if "exp.IntDiv in self.TRANSFORMS" in line:
                locations[(str(file), n)] = "IntDiv"
    assert sum(v == "expr" for v in locations.values()) >= 3
    tag = {cls: cls.__module__ + ":" + cls.__qualname__ for cls in owners}
    active_code = None
    active_index = -1
    slots = {}
    sets = {}
    current = None
    monitors = sys.monitoring
    tool = monitors.COVERAGE_ID

    def line(code, row):
        nonlocal active_index, current
        if code is active_code:
            position = slots.get(row, -1)
            if position >= 0 and position != active_index:
                for cache in caches:
                    cache.cache_clear()
            active_index = position
            current = sets[active_index]
            return
        addr = locations.get((code.co_filename, row))
        if addr is None:
            if code not in code_groups:
                return monitors.DISABLE
            return
        if current is None:
            return
        if addr == "opaque":
            current.add(("*", "*"))
            return
        values = sys._getframe(1).f_locals
        cls = tag.get(type(values.get("self")))
        if not cls:
            current.add(("*", "*"))
            return
        if addr == "expr":
            expr = values.get("expression")
            if expr is None:
                current.add(("*", "*"))
                return
            key = type(expr).__name__
        else:
            key = addr
        current.add((cls, key))

    def leave(code, offset, value):
        nonlocal current, active_index
        if code is active_code:
            current = sets[-1]
            active_index = -1
        elif code not in code_groups:
            return monitors.DISABLE

    records = {}
    readers = {}
    next_bit = 0
    caches = [
        c
        for c in gc.get_objects()
        if type(c) is functools._lru_cache_wrapper
        and getattr(c, "__module__", "").startswith("sqlglot")
    ]
    monitors.use_tool_id(tool, "per-validator transformation cells")
    monitors.register_callback(tool, monitors.events.LINE, line)
    monitors.register_callback(tool, monitors.events.PY_RETURN, leave)
    monitors.set_events(tool, monitors.events.LINE | monitors.events.PY_RETURN)
    try:
        for name, (cls, method, ordinary) in methods.items():
            instance = cls(method.name)
            slots = {
                n: i
                for i, stmt in enumerate(method.body)
                for n in range(stmt.lineno, stmt.end_lineno + 1)
            }
            sets = {i: set() for i in range(-1, len(method.body))}
            current = sets[-1]
            active_code = getattr(cls, method.name).__code__
            for cache in caches:
                cache.cache_clear()
            result = unittest.TestResult()
            instance.run(result)
            current = None
            active_code = None
            if not result.wasSuccessful() or result.skipped:
                continue
            # The extra bit is for setup, teardown and all unclassified source.
            # A hit there leaves the complete method eligible.
            count = len(method.body)
            start = next_bit
            next_bit += count + 1
            for cell in sets[-1]:
                readers[cell] = readers.get(cell, 0) | (1 << (start + count))
            for i in range(count):
                for cell in sets[i]:
                    readers[cell] = readers.get(cell, 0) | (1 << (start + i))
            records[name] = (start, tuple(ordinary), len(method.body))
    finally:
        monitors.set_events(tool, 0)
        monitors.register_callback(tool, monitors.events.LINE, None)
        monitors.register_callback(tool, monitors.events.PY_RETURN, None)
        monitors.free_tool_id(tool)
    output.write_bytes(zlib.compress(marshal.dumps((FORMAT, records, readers)), 6))
    print(
        "methods",
        len(records),
        "literal calls",
        sum(len(row[1]) for row in records.values()),
        "receiver keys",
        len(readers),
        flush=True,
    )


@functools.lru_cache(maxsize=1)
def index():
    version, records, readers = marshal.loads(zlib.decompress(DATA.read_bytes()))
    if version != FORMAT:
        raise ValueError("Incompatible validator cell index")
    return records, readers


def refine(sel, changes, selected, checkout, remaining):
    info = getattr(sel, "validator_keys", None)
    if not info or remaining < 0.85:
        return selected
    owner_keys, protected = info
    records, readers = index()
    mask = readers.get(("*", "*"), 0)
    for cell in owner_keys:
        mask |= readers.get(cell, 0)
    rows = []
    passing = set()
    for number in selected - protected - sel.confirmed_failures:
        record = sel.by_number[number]
        name = record["id"]
        if name not in records:
            continue
        start, ordinary, count = records[name]
        if (mask >> (start + count)) & 1:
            continue
        part = (mask >> start) & ((1 << count) - 1)
        ordinary_set = set(ordinary)
        # If a changed expression reaches arbitrary Python in this test, preserve
        # that statement and its entire suffix, including state-dependent helpers.
        stop = next(
            (i for i in range(count) if i not in ordinary_set and (part >> i) & 1),
            count,
        )
        safe = {i for i in ordinary if i < stop and not ((part >> i) & 1)}
        if not safe:
            continue
        active = tuple(i for i in range(count) if i not in safe)
        if not active:
            passing.add(number)
        else:
            rows.append((number, *name.split("::"), active))
    selected = set(selected) - passing
    if not rows:
        return selected
    rows.sort(
        key=lambda r: (
            len(r[4]) / max(1, round(sel.by_number[r[0]]["seconds"] * 1000)),
            r[0],
        )
    )
    from verification_budget import catalog_price

    price = getattr(sel, "verification_cost", None) or catalog_price(sel.by_number)
    gain = price(frozenset(selected)) - price(
        frozenset(selected - {r[0] for r in rows})
    )
    if gain < 370:
        return selected
    with tempfile.TemporaryDirectory(prefix="transform-assertions-") as temp:
        root = Path(temp) / "repo"
        marker = Path(temp) / "result"
        checkout(root, changes, sel)
        cpu = min(3.6, remaining - 0.2, gain * 0.00069)
        forkprobe.run(
            (
                root,
                rows,
                marker,
                cpu,
                [c["old"][:-3].replace("/", ".") for c in changes],
            ),
            {"PYTHONPATH": str(root), "SKIP_INTEGRATION": "0"},
            assertion_replay.run_subset,
            min(remaining, 1.5 + 2 * cpu),
        )
        return selected - (
            {int(x) for x in marker.read_text().splitlines() if x.isdecimal()}
            if marker.is_file()
            else set()
        )


if __name__ == "__main__":
    build(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
