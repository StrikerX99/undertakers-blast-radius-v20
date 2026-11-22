"""Verify TPC-DS at the optimizer boundary for edits to its called function bodies."""

import ast
import os
import pickle
import signal
import sys
import tempfile
import types
from pathlib import Path

TEST = "tests/test_executor.py::TestExecutor::test_execute_tpcds"


def header(node):
    return [getattr(node, key) for key in node._fields if key != "body"]


def closure_parameters(args):
    return not (
        args.defaults
        or any(args.kw_defaults)
        or any(n.annotation for n in ast.walk(args) if isinstance(n, ast.arg))
    )


def constraints(before, after, equal):
    import rule_replay
    from rule_modules import FORBIDDEN

    def captured(tree):
        result = {}

        def visit(code):
            k = (code.co_qualname, code.co_firstlineno)
            result[k] = (code.co_cellvars, code.co_freevars)
            for x in code.co_consts:
                if isinstance(x, types.CodeType):
                    visit(x)

        visit(compile(tree, "<optimizer-boundary>", "exec", dont_inherit=True))
        return result

    closure_captures = None
    lines, guard = set(), set()
    barred = (
        ast.Global,
        ast.Nonlocal,
        ast.Yield,
        ast.YieldFrom,
        ast.Await,
        ast.Import,
        ast.ImportFrom,
    )

    def scope(a, b, nested=False):
        nonlocal closure_captures
        if not isinstance(a, ast.FunctionDef) or not isinstance(b, ast.FunctionDef):
            return False
        for fn in (a, b):
            if any(
                not isinstance(d, ast.Name)
                or d.id not in {"staticmethod", "classmethod"}
                for d in fn.decorator_list
            ):
                return False
        if not equal(header(a), header(b)):
            if not (
                nested
                and a.name == b.name
                and not a.decorator_list
                and not b.decorator_list
                and not a.returns
                and not b.returns
                and closure_parameters(a.args)
                and closure_parameters(b.args)
                and not a.type_params
                and not b.type_params
            ):
                return False
        if len(a.body) == len(b.body):
            changed = [(x, y) for x, y in zip(a.body, b.body) if not equal(x, y)]
            if changed and all(
                isinstance(x, ast.FunctionDef)
                and isinstance(y, ast.FunctionDef)
                and x.name == y.name
                and not x.decorator_list
                and not y.decorator_list
                and closure_parameters(x.args)
                and closure_parameters(y.args)
                and not x.returns
                and not y.returns
                and not x.type_params
                and not y.type_params
                for x, y in changed
            ):
                if closure_captures is None:
                    closure_captures = (captured(before), captured(after))
                ac, bc = closure_captures

                def value(table, fn):
                    matches = [
                        v
                        for (name, line), v in table.items()
                        if line == fn.lineno and name.split(".")[-1] == fn.name
                    ]
                    return matches[0] if len(matches) == 1 else None

                for x, y in [(a, b), *changed]:
                    av, bv = value(ac, x), value(bc, y)
                    if av is None or av != bv:
                        return False
                # Creating a nested function with simple arguments evaluates no header
                # expression. Restrict the footprint to calls into the modified body.
                return all(scope(x, y, True) for x, y in changed)
        for n in (a, b):
            if rule_replay.external_write(n) or any(
                isinstance(k, barred)
                or isinstance(k, (ast.Name, ast.Attribute))
                and getattr(k, "id", getattr(k, "attr", None)) in FORBIDDEN
                for k in ast.walk(n)
            ):
                return False
        extra = rule_replay.added_calls(a, b)
        if extra:
            # The sole exception to the known invocation set is a diagnostic while
            # constructing a raised exception. No observed run may enter that line.
            armed = set()
            for e in ast.walk(b):
                if isinstance(e, ast.Raise):
                    armed.update(id(c) for c in ast.walk(e) if isinstance(c, ast.Call))
            for e in ast.walk(b):
                if not isinstance(e, ast.Call) or ast.unparse(e.func) not in extra:
                    continue
                if id(e) not in armed:
                    return False
                guard.update(range(e.lineno, e.end_lineno + 1))
        lines.update(range(a.body[0].lineno, a.end_lineno + 1))
        return True

    def outer(a, b):
        if len(a) != len(b):
            return False
        for x, y in zip(a, b):
            if equal(x, y):
                continue
            if (
                isinstance(x, ast.ClassDef)
                and isinstance(y, ast.ClassDef)
                and equal(header(x), header(y))
            ):
                if not outer(x.body, y.body):
                    return False
            elif not (scope(x, y)):
                return False
        return True

    return (lines, guard) if outer(before.body, after.body) else None


def evidence(sel, changes, apply, equal):
    from case_verify import phase_mask, index
    from rule_replay import ledger
    from source_fast import paired_tree

    db = ledger()
    spans = {}
    forbidden = {}
    for ch in changes:
        path = ch["old"]
        if (
            path != ch["new"]
            or not path.startswith("sqlglot/optimizer/")
            or not path.endswith(".py")
        ):
            return
        src, tree = sel.source(path)
        changed = paired_tree(ch, tree, src, apply(src, ch)[0])
        result = constraints(tree, changed, equal)
        if not result:
            return
        spans[path], forbidden[path] = result
    bits = phase_mask(spans)
    cases = index()["cases"]
    required = set()
    while bits:
        low = bits & -bits
        bits ^= low
        row = cases[low.bit_length() - 1]
        kind = row["kind"]
        if kind in ("shared", "executor_setup", "validation"):
            return
        if isinstance(row["key"], list) and row["key"][0] == "tpcds":
            if kind != "optimize":
                return
            required.add(row["key"][1])
    from executor_stages import query as observed_calls

    actual = observed_calls(spans, sel.mapping)
    if (
        actual is None
        or set(actual) - {"optimize"}
        or actual.get("optimize", set()) != required
    ):
        return
    stages = {}
    for path, lines in spans.items():
        for line in lines:
            for i, q in db.execute(
                "SELECT id,q FROM reached JOIN stage USING(id) WHERE src=? AND line=?",
                (path, line),
            ):
                stages.setdefault(q, set()).add(i)
    if required != set(stages) or any(
        not db.execute("SELECT usable FROM query WHERE q=?", (q,)).fetchone()[0]
        for q in required
    ):
        return
    return stages, forbidden


def worker(job):
    import rule_replay

    root, jobs, banned, out, budget = job
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.setrecursionlimit(12000)
    signal.signal(signal.SIGPROF, signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_PROF, budget)
    paths = {str(root / p): s for p, s in banned.items() if s}
    mon = sys.monitoring
    idx = mon.COVERAGE_ID
    interrupted = False
    if paths:
        mon.use_tool_id(idx, "exception guards")

        def visit(code, line):
            nonlocal interrupted
            if line in paths.get(code.co_filename, ()):
                interrupted = True
                raise ValueError("Unobserved diagnostic invoked")
            return mon.DISABLE

        mon.register_callback(idx, mon.events.LINE, visit)
        mon.set_events(idx, mon.events.LINE)
    if sum(map(len, jobs.values())) <= 6:
        result = out.with_name("rule-complete")
        rule_replay.probe((root, jobs, result, budget))
        if result.exists() and not interrupted:
            out.touch()
        return
    from case_verify import index
    from checkpoint_graph import expression_digest
    from sqlglot import parse_one
    from sqlglot.optimizer import optimize
    from tests.helpers import TPCDS_SCHEMA, load_sql_fixture_pairs

    queries = list(load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))
    expected = {
        r["key"][1]: r["graph"]
        for r in index()["cases"]
        if r["kind"] == "optimize" and r["key"][0] == "tpcds"
    }
    for i in sorted(jobs):
        expr = pickle.loads(pickle.dumps(parse_one(queries[i][1]), protocol=4))
        actual = optimize(
            expr, schema=TPCDS_SCHEMA, leave_tables_isolated=True, dialect=None
        )
        if expression_digest(actual) != expected[i]:
            return
    if not interrupted:
        out.touch()


def refine(sel, changes, selected, apply, equal, checkout, remaining):
    # A native optimizer call already produced a different final graph for this patch.
    # An equality-only replay cannot certify this test. The actual result may still pass.
    if changes and changes[0].get("_optimizer_graph_changed"):
        return selected
    from verification_budget import pack_ms
    import forkprobe

    n = sel.by_name.get(TEST)
    if (
        n not in selected
        or sel.unknown
        or n in sel.confirmed_failures
        or not changes
        or remaining < 2
        or any(
            c["old"] in sel.files and n in sel.sets[sel.files[c["old"]]]
            for c in changes
        )
        or any(not c["old"].startswith("sqlglot/optimizer/") for c in changes)
    ):
        return selected
    # Match the runner's hash environment for every portable optimizer fingerprint.
    if sys.flags.hash_randomization:
        return selected
    result = evidence(sel, changes, apply, equal)
    rule_inputs = result is None
    if rule_inputs:
        import operator_replay

        diagnostic = operator_replay.cases(sel, changes, apply, equal)
        if not diagnostic:
            return selected
        jobs, bindings = diagnostic
        guard = {}
    else:
        jobs, guard = result
    if not jobs:
        return selected - {n}
    loads = {i: round(1000 * sel.by_number[i]["seconds"]) for i in selected}
    gain = (
        pack_ms(loads.values()) - pack_ms(v for i, v in loads.items() if i != n)
    ) / 1000
    estimate = 0.55 + min(0.12 * sum(map(len, jobs.values())), 0.1 * len(jobs))
    if gain < max(1, estimate):
        return selected
    budget = min(9, 0.85 * gain, remaining - 0.5)
    with tempfile.TemporaryDirectory(prefix="optimizer-boundary-") as tmp:
        path = Path(tmp)
        root = path / "src"
        out = path / "equal"
        checkout(root, changes, sel)
        if rule_inputs:
            forkprobe.run(
                (root, jobs, bindings, out, budget),
                {"PYTHONPATH": str(root)},
                operator_replay.check,
                min(remaining, 2 * budget + 1),
            )
            return selected - {n} if out.is_file() else selected
        forkprobe.run(
            (root, jobs, guard, out, budget),
            {
                "PYTHONPATH": str(root),
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
            worker,
            min(remaining, 2 * budget + 1),
        )
        return selected - {n} if out.exists() else selected
