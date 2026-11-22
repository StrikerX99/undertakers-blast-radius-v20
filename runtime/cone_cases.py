"""Run the original generator memory assertions for dialects touched by an admitted class edit."""

import json
import signal
import sys
import tempfile
from pathlib import Path

import case_verify
import forkprobe
import profile_store
from case_profile import MEMORY, PIN
from verification_budget import pack_ms


def dialect_classes():
    from sqlglot import Dialect
    from sqlglot.dialects import Dialects

    result = {}
    for value in Dialects:
        dialect = type(Dialect.get_or_raise(value))
        records = []
        for cls in (dialect, dialect.parser_class, dialect.generator_class):
            records.append([c.__module__ + "." + c.__qualname__ for c in cls.__mro__])
        result[value.value] = records
    return result


def build(root, output):
    sys.path.insert(0, str(root))
    output.write_text(json.dumps({"pin": PIN, "classes": dialect_classes()}))


def inspect_cases(payload):
    root, owners, files, baseline, projected, out, budget = payload
    signal.setitimer(signal.ITIMER_PROF, budget)
    from native_tests import watch_reference

    guard = watch_reference(root)
    from sqlglot import __file__ as origin

    if Path(origin).resolve() != (root / "sqlglot/__init__.py").resolve():
        return
    saved = json.loads(Path(__file__).with_name("cone-memory-mro.json").read_bytes())
    current = dialect_classes()
    if (
        saved["pin"] != PIN
        or current != saved["classes"]
        or current.keys() != baseline.keys()
    ):
        return
    active = set(projected)
    found = set()
    for dialect, records in current.items():
        for lineage in records:
            for dotted in lineage:
                if dotted in owners:
                    found.add(dotted)
                    active.add(dialect)
                if dotted.rpartition(".")[0] in files:
                    active.add(dialect)
    if not owners <= found or not active:
        return
    if 0.25 + 1.25 * sum(baseline[d] for d in active) / 1000 > budget:
        return
    if (
        not guard.missing
        and case_verify.check_memory([{"key": d} for d in active])
        and not guard.missing
    ):
        out.touch()


def refine(sel, changes, selected, checkout, remaining):
    owners = getattr(sel, "cone_owners", None)
    number = sel.by_name.get(MEMORY)
    if (
        not owners
        or not getattr(sel, "cone_candidates", None)
        or number not in selected
        or number in sel.confirmed_failures
        or sel.unknown
        or remaining < 1
        or sel.mapping["commit"] != PIN
    ):
        return selected
    if not all(
        c["old"] == c["new"]
        and c["old"].startswith(("sqlglot/parsers/", "sqlglot/generators/"))
        for c in changes
    ):
        return selected
    durations = {n: round(1000 * sel.by_number[n]["seconds"]) for n in selected}
    before = max(2000, pack_ms(durations.values()))
    tail = [d for n, d in durations.items() if n != number]
    after = max(2000, pack_ms(tail)) if tail else 0
    saving = (before - after) / 1000
    if saving < 1.5:
        return selected
    budget = min(0.68 * saving, 3.2, remaining * 0.7)
    # The independently observed per-dialect passing run records its source reads.
    # Any case that previously ran code in an edited file remains in the real run.
    masks = 0
    for change in changes:
        for bits in profile_store.get("cases", change["old"]).values():
            masks |= bits
    rows = case_verify.index()["cases"]
    times = {row["key"]: row["cpu_ms"] for row in rows if row["kind"] == "memory"}
    projected = {
        row["key"]
        for i, row in enumerate(rows)
        if row["kind"] == "memory" and masks & (1 << i)
    }
    if not times or 0.25 + 1.25 * sum(times[d] for d in projected) / 1000 > budget:
        return selected
    modules = {c["old"][:-3].replace("/", ".") for c in changes}
    with tempfile.TemporaryDirectory(prefix="br-memory-cases-") as d:
        root, out = Path(d) / "repo", Path(d) / "passed"
        checkout(root, changes, sel)
        forkprobe.run(
            (root, owners, modules, times, projected, out, budget),
            {
                "PYTHONPATH": str(root),
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
            inspect_cases,
            min(remaining, 0.4 + 2 * budget),
        )
        return selected - {number} if out.exists() else selected


if __name__ == "__main__":
    build(Path(sys.argv[1]).resolve(), Path(sys.argv[2]))
