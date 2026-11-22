"""Verify a whole small set of edited test methods when its last worker is expensive."""

import ast
import os
import sys
import tempfile
from pathlib import Path

import forkprobe
from verification_budget import baseline_costs, pack_ms


def only_assertions(left, right, equal):
    """Permit literal changes and local checks in existing test method bodies."""
    if equal(left, right):
        return True
    if type(left) is not type(right):
        return False
    if isinstance(left, (ast.Module, ast.ClassDef)):
        if len(left.body) != len(right.body) or any(
            not equal(getattr(left, k), getattr(right, k))
            for k in left._fields
            if k != "body"
        ):
            return False
        return all(only_assertions(a, b, equal) for a, b in zip(left.body, right.body))
    if not isinstance(left, ast.FunctionDef) or not left.name.startswith("test_"):
        return False
    if any(
        not equal(getattr(left, k), getattr(right, k))
        for k in left._fields
        if k != "body"
    ):
        return False

    def assertion(stmt):
        if isinstance(stmt, ast.Assert):
            return True
        return (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Attribute)
            and isinstance(stmt.value.func.value, ast.Name)
            and stmt.value.func.value.id == "self"
            and stmt.value.func.attr
            in {
                "assertTrue",
                "assertFalse",
                "assertEqual",
                "assertNotEqual",
                "assertIs",
                "assertIsNot",
                "assertIsNone",
                "assertIsNotNone",
                "assertIn",
                "assertNotIn",
                "assertGreater",
                "assertLess",
            }
        )

    def literal_check(stmt):
        if not assertion(stmt):
            return False
        children = [stmt.test] if isinstance(stmt, ast.Assert) else stmt.value.args
        if isinstance(stmt, ast.Assert) and stmt.msg is not None:
            children = [*children, stmt.msg]
        if isinstance(stmt, ast.Expr) and stmt.value.keywords:
            return False
        return all(isinstance(v, ast.Constant) for v in children)

    old = [st for st in left.body if not literal_check(st)]
    new = [st for st in right.body if not literal_check(st)]
    if len(old) != len(new):
        return False
    for a, b in zip(old, new):
        if equal(a, b):
            continue
        # Keep the actual input expression unchanged. A new expected literal
        # may change the assertion outcome, but cannot call another helper.
        if not assertion(a) or not assertion(b):
            return False
        if not (
            isinstance(a, ast.Expr)
            and isinstance(b, ast.Expr)
            and equal(a.value.func, b.value.func)
            and len(a.value.args) == 2
            and len(b.value.args) == 2
            and not a.value.keywords
            and not b.value.keywords
            and equal(a.value.args[0], b.value.args[0])
            and isinstance(a.value.args[1], ast.Constant)
            and isinstance(b.value.args[1], ast.Constant)
            and type(a.value.args[1].value) is type(b.value.args[1].value)
        ):
            return False
    return True


def run(payload):
    # Run pytest's actual collection, subtests, and teardown. Report only once the
    # entire requested session passes, so a timeout or late error keeps every test.
    root, nodes, output, budget = payload
    import signal

    signal.setitimer(signal.ITIMER_PROF, budget)
    os.chdir(root)
    sys.path.insert(0, str(root))
    from method_probe import MissingReferencePackage
    import json

    guard = MissingReferencePackage(
        set(json.loads(Path(__file__).with_name("reference-imports.json").read_bytes()))
    )
    sys.meta_path.append(guard)
    import pytest

    passed = set()
    incomplete = set()
    collected = set()

    class Observe:
        def pytest_collection_finish(self, session):
            collected.update(item.nodeid for item in session.items)

        def pytest_runtest_logreport(self, report):
            if not report.passed or getattr(report, "wasxfail", False):
                incomplete.add(report.nodeid)
            elif report.when == "call":
                passed.add(report.nodeid)

    status = pytest.main(
        ["-x", "-q", "--tb=no", "-p", "no:cacheprovider", *nodes], plugins=[Observe()]
    )
    if (
        status == 0
        and not guard.missing
        and not incomplete
        and set(nodes) == collected == passed
    ):
        output.touch()


def refine(sel, changes, selected, checkout, remaining, apply, equal):
    if (
        not selected
        or not sel.direct_tests
        or not selected <= sel.direct_tests
        or sel.confirmed_failures
        or sel.unknown
        or sel.import_modules
        or not all(
            c["old"] == c["new"]
            and c["old"].startswith("tests/")
            and c["old"].endswith(".py")
            for c in changes
        )
    ):
        return selected
    tests = [sel.by_number[n] for n in sorted(selected)]
    recorded = [round(t["seconds"] * 1000) for t in tests]
    # The first, inexpensive unittest pass already covers this range.
    if sum(recorded) <= 750 or remaining < 1.2:
        return selected
    costs = baseline_costs()["methods"]
    if any(t["id"] not in costs for t in tests):
        return selected
    forecast = 425 + sum(costs[t["id"]] for t in tests)
    saving = max(2000, pack_ms(recorded))
    if forecast > min(2200, 0.52 * saving):
        return selected
    from source_fast import paired_tree

    for change in changes:
        text, old = sel.source(change["old"])
        updated = apply(text, change)[0]
        if not only_assertions(old, paired_tree(change, old, text, updated), equal):
            return selected
    budget = min(3.2, 0.72 * saving / 1000, 0.5 + 1.65 * forecast / 1000)
    with tempfile.TemporaryDirectory(prefix="br-edited-test-") as d:
        repo = Path(d) / "repo"
        result = Path(d) / "passed"
        checkout(repo, changes, sel)
        env = {
            "PYTHONPATH": str(repo),
            "PYTHON_CPU_COUNT": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "SKIP_INTEGRATION": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        forkprobe.run(
            (repo, [t["id"] for t in tests], result, budget),
            env,
            run,
            min(remaining, 1 + 2 * budget),
        )
        if result.exists():
            return set()
    return selected
