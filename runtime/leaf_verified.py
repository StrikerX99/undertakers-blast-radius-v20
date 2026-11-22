"""Check TPC-DS on an actual patch adding a simple, previously unknown expression class."""

import ast
from functools import lru_cache
from pathlib import Path
import re
import tempfile

TARGET = "tests/test_executor.py::TestExecutor::test_execute_tpcds"
PIN = "eb11aec4be8cb2122afe713a4816662c3f0fb2fa"


@lru_cache(maxsize=1)
def namespace_index():
    import json

    return json.loads(Path(__file__).with_name("leaf-namespace.json").read_bytes())


def literal_class(stmt):
    if not isinstance(stmt, ast.ClassDef) or not re.fullmatch(
        r"[A-Z][A-Za-z0-9]+", stmt.name
    ):
        return False
    if stmt.decorator_list or stmt.keywords or stmt.type_params:
        return False
    if not all(isinstance(base, ast.Name) for base in stmt.bases):
        return False
    if [base.id for base in stmt.bases] != ["Expression", "Func"] or len(
        stmt.body
    ) != 1:
        return False
    field = stmt.body[0]
    if isinstance(field, ast.Pass):
        return True
    if not (
        isinstance(field, ast.Assign)
        and len(field.targets) == 1
        and isinstance(field.targets[0], ast.Name)
        and field.targets[0].id == "arg_types"
        and isinstance(field.value, ast.Dict)
    ):
        return False
    return all(
        isinstance(key, ast.Constant)
        and type(key.value) is str
        and key.value.isidentifier()
        and not key.value.startswith("_")
        for key in field.value.keys
    ) and all(
        isinstance(value, ast.Constant) and type(value.value) is bool
        for value in field.value.values
    )


def added(sel, changes, apply, equal):
    if len(changes) != 1:
        return None
    change = changes[0]
    path = change["old"]
    if (
        not path
        or path != change["new"]
        or not re.fullmatch(r"sqlglot/expressions/[a-z][a-z_]+\.py", path)
    ):
        return None
    if not any(
        line.startswith("+class ") for hunk in change["hunks"] for line in hunk["lines"]
    ):
        return None
    data = namespace_index()
    module = path[:-3].replace("/", ".")
    if (
        module not in data["exported"]
        or data["pin"] != PIN
        or sel.mapping.get("commit") != PIN
        or any(
            sel.mapping["source_hashes"].get(k) != v
            for k, v in data["registry_files"].items()
        )
    ):
        return None
    raw, before = sel.source(path)
    after = ast.parse(apply(raw, change)[0])
    original = iter(before.body)
    prior = next(original, None)
    name = None
    for stmt in after.body:
        if prior is not None and equal(prior, stmt):
            prior = next(original, None)
        elif name is None and literal_class(stmt):
            name = stmt.name
        else:
            return None
    if prior is not None or not name or name in data["forbidden"]:
        return None
    return module, name


def exercise(job):
    import importlib
    import os
    import signal
    import sys
    from registry_snapshot import registry_digests

    root, module, name, marker, allowance = job
    os.chdir(root)
    sys.path.insert(0, str(root))
    signal.setitimer(signal.ITIMER_PROF, allowance)
    from sqlglot import exp
    from sqlglot.helper import camel_to_snake_case

    data = namespace_index()
    cls = vars(importlib.import_module(module)).get(name)
    if not isinstance(cls, type):
        return
    key = camel_to_snake_case(name)
    if (
        cls.__bases__ != (exp.Expression, exp.Func)
        or cls.sql_names() != [key]
        or exp.FUNCTION_BY_NAME.get(key) is not cls
        or exp.EXPR_CLASSES.get(name.lower()) is not cls
        or [item for item in exp.ALL_FUNCTIONS if key in item.sql_names()] != [cls]
        or any(
            vars(importlib.import_module(p)).get(name) is not cls
            for p in data["exported"][module]
        )
        or registry_digests(exp, cls) != data["registries"]
    ):
        return

    # Verify the entire pinned executor workload even when native keys suggest the
    # class is unused. Optimizer enumeration and indirect registry reads are covered
    # by comparing every active optimized graph. Changed candidates use the real
    # query assertion. Other catalog methods keep their prior scheduling decisions.
    import query_frontier

    query_frontier.run(
        (
            root,
            [TARGET],
            None,
            True,
            set(),
            None,
            None,
            set(),
            (),
            marker,
            max(0.2, allowance - 0.65),
        )
    )


def refine(sel, changes, selected, apply, equal, checkout, remaining):
    number = sel.by_name.get(TARGET)
    if number not in selected or number in sel.confirmed_failures or remaining < 6:
        return selected
    declaration = added(sel, changes, apply, equal)
    if not declaration:
        return selected
    from forkprobe import run
    from verification_budget import catalog_price

    price = getattr(sel, "verification_cost", None)
    if price is None:
        price = sel.verification_cost = catalog_price(sel.by_number)
    saving = (price(frozenset(selected)) - price(frozenset(selected - {number}))) / 1000
    if saving < 5:
        return selected
    allowance = min(9.0, 0.9 * saving, remaining - 1)
    with tempfile.TemporaryDirectory(prefix="func-check-") as tmp:
        root = Path(tmp) / "src"
        marker = Path(tmp) / "passed"
        checkout(root, changes, sel)
        run(
            (root, *declaration, marker, allowance),
            {
                "PYTHONPATH": str(root),
                "SKIP_INTEGRATION": "0",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            },
            exercise,
            min(remaining, 2 * allowance + 1),
        )
        if marker.is_file() and TARGET in marker.read_text().splitlines():
            return selected - {number}
    return selected
