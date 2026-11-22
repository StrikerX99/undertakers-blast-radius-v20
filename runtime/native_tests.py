"""Measure and verify complete unittest methods. Incomplete groups are retained."""

import ast
import importlib
import itertools
import json
import os
import signal
import sys
import time
import unittest
from pathlib import Path

from method_probe import MissingReferencePackage

EXCLUDED_TEST_FILES = {
    "tests/test_executor.py",
    "tests/test_optimizer.py",
    "tests/test_benchmarks.py",
    "tests/test_cli.py",
    "tests/test_mypyc.py",
}


class Results(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.success = set()
        self.incomplete = set()
        self.cpu = {}
        self.fixture_error = False

    def startTest(self, test):
        self.start = time.process_time()
        super().startTest(test)

    def stopTest(self, test):
        self.cpu[test.id()] = max(1, round((time.process_time() - self.start) * 1000))
        super().stopTest(test)

    def addSuccess(self, test):
        self.success.add(test.id())
        super().addSuccess(test)

    def addSkip(self, test, reason):
        self.incomplete.add(getattr(test, "test_case", test).id())
        super().addSkip(test, reason)

    def addError(self, test, error):
        if not isinstance(test, unittest.TestCase):
            self.fixture_error = True
        super().addError(test, error)


def run_groups(groups, callback, guard):
    for filename, class_name, items in groups:
        if guard.missing:
            break
        start = time.process_time()
        try:
            module = importlib.import_module(filename[:-3].replace("/", "."))
            cls = getattr(module, class_name)
            suite = unittest.TestSuite(cls(name) for name in items)
            if not issubclass(cls, unittest.TestCase):
                continue
        except Exception:
            continue
        result = Results()
        suite.run(result)
        duration = round(1000 * (time.process_time() - start))
        if result.fixture_error or guard.missing:
            continue
        passed = result.success - result.incomplete
        completed = {
            name: result.cpu[cls(name).id()]
            for name in items
            if cls(name).id() in passed
        }
        callback(
            filename, class_name, completed, max(0, duration - sum(result.cpu.values()))
        )


def watch_reference(root):
    os.chdir(root)
    sys.path.insert(0, str(root))
    guard = MissingReferencePackage(
        set(json.loads(Path(__file__).with_name("reference-imports.json").read_bytes()))
    )
    sys.meta_path.append(guard)
    return guard


def worker(payload):
    root, groups, dest, cpu_budget = payload
    signal.setitimer(signal.ITIMER_PROF, cpu_budget)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    guard = watch_reference(root)
    with open(dest, "w", buffering=1, encoding="utf-8") as out:

        def publish(path, cls, methods, overhead):
            print(json.dumps([path, cls, list(methods)]), file=out)

        run_groups(groups, publish, guard)


def baseline(root, catalog, output):
    guard = watch_reference(root)
    tests = []
    for t in catalog["tests"]:
        parts = t["id"].split("::")
        if len(parts) != 3 or parts[0] in EXCLUDED_TEST_FILES or t["seconds"] > 3:
            continue
        tests.append(parts)
    records, setup = {}, {}
    risky = {
        "tracemalloc",
        "getrusage",
        "subprocess",
        "get_context",
        "Popen",
        "current_process",
        "getrefcount",
        "stack",
        "getframe",
        "get_objects",
        "get_traced_memory",
        "recursionlimit",
        "getsizeof",
        "environ",
        "chdir",
        "sys",
        "timeit",
        "pytest",
        "SKIP_INTEGRATION",
        "os",
        "resource",
    }

    def accepted(filename, cls, names):
        tree = ast.parse((root / filename).read_bytes())
        declarations = [
            c for c in tree.body if isinstance(c, ast.ClassDef) and c.name == cls
        ]
        if len(declarations) != 1:
            return []
        methods = {
            n.name: n for n in declarations[0].body if isinstance(n, ast.FunctionDef)
        }

        def symbols(fn):
            return {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)} | {
                n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)
            }

        return [m for m in names if m in methods and not (symbols(methods[m]) & risky)]

    for (filename, cls), group in itertools.groupby(
        sorted(tests), key=lambda parts: tuple(parts[:2])
    ):
        names = accepted(filename, cls, [p[2] for p in group])
        if not names:
            continue

        def collect(path, klass, methods, overhead):
            records.update({f"{path}::{klass}::{m}": v for m, v in methods.items()})
            setup[f"{path}::{klass}"] = overhead

        # Optional imports vary by file. Keep complete data for other files.
        guard.missing.clear()
        run_groups([(filename, cls, names)], collect, guard)
    output.write_text(
        json.dumps({"methods": records, "setup": setup}, separators=(",", ":"))
    )
    print(len(records), "baseline-passing methods")


if __name__ == "__main__":
    root, catalog, dest = (Path(s).resolve() for s in sys.argv[1:])
    baseline(root, json.loads(catalog.read_bytes()), dest)
