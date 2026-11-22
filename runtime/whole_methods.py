"""Verify complete methods when their measured CPU buys down the busiest worker."""

from collections import defaultdict
from functools import lru_cache
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time

import forkprobe
from verification_budget import baseline_costs, catalog_price


@lru_cache(maxsize=1)
def costs():
    return json.loads(Path(__file__).with_name("tail-costs.json").read_bytes())


def verify(args):
    root, names, marker, limit = args
    os.chdir(root)
    sys.path.insert(0, str(root))
    signal.setitimer(signal.ITIMER_PROF, limit)
    from method_probe import MissingReferencePackage

    guard = MissingReferencePackage(
        set(json.loads(Path(__file__).with_name("reference-imports.json").read_bytes()))
    )
    sys.meta_path.append(guard)
    import pytest

    incomplete = set()
    collected = set()
    passed = set()
    torn_down = set()
    invalid_fixture = False

    class Outcomes:
        def pytest_collection_finish(self, session):
            collected.update(x.nodeid for x in session.items)

        def pytest_runtest_logreport(self, report):
            nonlocal invalid_fixture
            if report.when in ("setup", "teardown") and not report.passed:
                invalid_fixture = True
            if not report.passed or getattr(report, "wasxfail", False):
                incomplete.add(report.nodeid)
            elif report.when == "call":
                passed.add(report.nodeid)
            elif report.when == "teardown":
                torn_down.add(report.nodeid)

    result = pytest.main(
        ["-q", "--tb=no", "-p", "no:cacheprovider", *names], plugins=[Outcomes()]
    )
    if (
        result in (0, 1)
        and collected == set(names)
        and not invalid_fixture
        and not guard.missing
    ):
        marker.write_text(json.dumps(sorted((passed & torn_down) - incomplete)))


def refine(sel, changes, selected, checkout, remaining):
    if remaining < 1 or not selected or sel.unknown:
        return selected
    if not all(
        c["old"] == c["new"]
        and c["old"].startswith(
            ("sqlglot/", "tests/fixtures/", "tests/test_", "tests/dialects/test_")
        )
        for c in changes
    ):
        return selected
    if any(c["old"].endswith("conftest.py") for c in changes):
        return selected
    price = getattr(sel, "verification_cost", None)
    if price is None:
        price = sel.verification_cost = catalog_price(sel.by_number)
    ids = frozenset(selected)
    before = price(ids)
    confirmed = sel.confirmed_failures
    if before - price(frozenset(confirmed)) < 850:
        return selected
    # After the earlier checks, a tiny residual plan often already consists of
    # actual failures. A new interpreter has too little recoverable work there.
    if len(ids - confirmed) < 12 and not any(
        sel.by_number[n]["seconds"] >= 5 for n in ids - confirmed
    ):
        return selected
    if not any(c["old"].startswith("sqlglot/") for c in changes):
        return selected
    endpoint = costs()
    other = baseline_costs()["methods"]
    attempted = getattr(sel, "native_attempted", ())
    small = len(ids) <= 42
    groups = defaultdict(list)
    durations = {}
    records = {}
    for n in ids - confirmed:
        t = sel.by_number[n]
        name = t["id"]
        if name in endpoint:
            estimated = endpoint[name]
        elif small and n not in attempted and name in other:
            estimated = other[name]
        else:
            continue
        if 0 < estimated <= 3100:
            records[n] = estimated
            durations[n] = round(t["seconds"] * 1000)
            groups[name.rsplit("::", 1)[0]].append(n)
    if not records:
        return selected

    def plan_score(jobset):
        return before - price(ids.difference(jobset))

    if plan_score(records.keys()) < 850:
        return selected
    # Consider individual bottlenecks, class groups and whole tails. A ranking
    # by recorded duration alone can miss multiple tests masking one another.
    portfolios = set()

    def prefixes(order):
        group = []
        for j, n in enumerate(order, 1):
            group.append(n)
            if (
                j < 12
                or j in (16, 24, 32, 48, 64)
                or j == len(order)
                or durations[n] >= 500
            ):
                portfolios.add(frozenset(group))

    for members in groups.values():
        prefixes(
            sorted(
                members,
                key=lambda n: (records[n] / max(1, durations[n]), -durations[n], n),
            )
        )
    prefixes(
        sorted(
            records, key=lambda n: (records[n] / max(1, durations[n]), -durations[n], n)
        )
    )
    prefixes(sorted(records, key=lambda n: (-durations[n], n)))
    for n in records:
        if durations[n] >= 200:
            portfolios.add(frozenset((n,)))
    best = 0
    chosen = ()
    clocks = 0
    for subset in portfolios:
        runtime = sum(records[n] for n in subset)
        if runtime > 4300:
            continue
        cl = {sel.by_number[n]["id"].rsplit("::", 1)[0] for n in subset}
        # Pytest startup and imports are paid once per patch. Class setup adds a
        # much smaller charge when several methods share the same fixture.
        spend = runtime + 220 + 70 * len(cl)
        saving = plan_score(subset)
        gain = 0.88 * saving - 1.3 * spend
        rank = (round(gain), -spend, tuple(sorted(subset)))
        if gain > 420 and (not chosen or rank > best):
            best = rank
            chosen = subset
            clocks = min(6.0, 0.6 + 1.7 * spend / 1000, 0.85 * saving / 1000)
    if not chosen:
        return selected
    with tempfile.TemporaryDirectory(prefix="whole-tests-") as tmp:
        root = Path(tmp) / "src"
        marker = Path(tmp) / "finished"
        checkout(root, changes, sel)
        names = sorted(sel.by_number[n]["id"] for n in chosen)
        env = dict(
            PYTHONPATH=str(root),
            SKIP_INTEGRATION="0",
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
            PYTHON_CPU_COUNT="1",
        )
        forkprobe.run(
            (root, names, marker, clocks),
            env,
            verify,
            min(sel.verify_wall_deadline - time.monotonic(), 1.2 + 1.7 * clocks),
        )
        passes = json.loads(marker.read_bytes()) if marker.exists() else []
        return set(selected) - {sel.by_name[n] for n in passes if n in names}
