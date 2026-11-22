"""Buy patch verification only when removing passing tests can lower worker cost."""

from heapq import heapreplace
import json
import os
import tempfile
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import attribute_cases
import dispatch_delta
import symbol_impact
import cold_inputs
import case_verify
import forkprobe
import native_tests


@lru_cache(maxsize=1)
def baseline_costs():
    return json.loads(Path(__file__).with_name("native-costs.json").read_bytes())


def pack_ms(durations):
    jobs = sorted((n for n in durations if n), reverse=True)
    if not jobs:
        return 0
    total = sum(jobs)
    best = total + 2000
    for k in range(2, min(len(jobs), 8) + 1):
        if max(jobs[0], (total + k - 1) // k) + 2000 * k >= best:
            continue
        loads = [0] * k
        for length in jobs:
            heapreplace(loads, loads[0] + length)
        best = min(best, max(loads) + 2000 * k)
    return best


def catalog_price(by_number):
    """Price many candidate subsets against the same catalog without repacking the long tail."""
    ms = {number: round(1000 * record["seconds"]) for number, record in by_number.items()}
    catalog = frozenset(ms)
    ordered = sorted(((duration, number) for number, duration in ms.items() if duration), reverse=True)
    full_load = sum(ms.values())
    broad = max(len(ms) // 2, len(ms) - len(ordered) + 8)

    @lru_cache(maxsize=384)
    def price(numbers):
        if not numbers:
            return 0
        if len(numbers) > broad:
            # More than eight selected jobs have positive duration. The longest
            # job is often the optimal makespan on these broad selections.
            load = full_load - sum(map(ms.__getitem__, catalog - numbers))
            longest = next(duration for duration, number in ordered if number in numbers)
            lower, k = min(
                (max(longest, (load + k - 1) // k) + 2000 * k, k)
                for k in range(1, 9)
            )
            if lower == longest + 2000 * k:
                bins = [0] * k
                for duration, number in ordered:
                    if number not in numbers:
                        continue
                    if load + (k - 1) * duration < k * (longest + 1):
                        return lower
                    next_load = bins[0] + duration
                    if next_load > longest:
                        break
                    heapreplace(bins, next_load)
                else:
                    return lower
        return max(2000, pack_ms(map(ms.__getitem__, numbers)))

    return price


def native_plan(sel, ids, cost, confirmed):
    table = baseline_costs()
    groups = defaultdict(list)
    for n in ids - confirmed:
        t = sel.by_number[n]
        name = t["id"]
        if name in table["methods"]:
            groups[name.rsplit("::", 1)[0]].append(
                (n, table["methods"][name], round(1000 * t["seconds"]))
            )
    if not groups:
        return None

    # Prefixes include partial classes, whole classes and groups of cheap classes.
    # The score includes the actual 2 s worker start cost and scheduling changes.
    sets = []
    for cls, jobs in groups.items():
        jobs.sort(key=lambda r: (r[1] / max(1, r[2]), -r[2], r[0]))
        overhead = max(12, min(750, table["setup"].get(cls, 35)))
        total = sum(ms for _, ms, _ in jobs) + overhead
        weight = sum(s for _, _, s in jobs)
        sets.append((total / max(weight, 1), cls, jobs, overhead))
    optional = []
    for _, cls, jobs, overhead in sorted(sets):
        for i, (n, ms, sec) in enumerate(jobs):
            optional.append((n, ms + (overhead if not i else 0), sec))
    nr = len(optional)
    checkpoints = {1, 2, 4, 8, 16, 32, nr}
    k = 64
    while k < nr:
        checkpoints.add(k)
        k = int(k * 1.5)
    for i, (_, _, sec) in enumerate(optional):
        if sec >= 350:
            checkpoints.add(i + 1)
    probed, spent = set(), 95
    before = cost(ids)
    best, best_gain = None, 0
    for i, (n, ms, sec) in enumerate(optional, 1):
        probed.add(n)
        spent += ms
        if spent > 5100:
            break
        if i not in checkpoints:
            continue
        saving = before - cost(ids - probed)
        gain = 0.78 * saving - 1.4 * spent
        if gain > max(220, best_gain):
            best_gain = gain
            best = (
                probed.copy(),
                max(350, min(5400, int(1.8 * spent + 120), int(0.8 * saving))),
            )
    if best is None:
        return None
    chosen, budget = best
    todo = defaultdict(list)
    for n, _, _ in optional:
        if n in chosen:
            file, cls, method = sel.by_number[n]["id"].split("::")
            todo[file, cls].append(method)
    return [(file, cls, methods) for (file, cls), methods in todo.items()], budget


def refine(
    sel,
    changes,
    selected,
    apply,
    equal,
    checkout,
    remaining,
    *,
    use_cases=True,
    use_native=True,
):
    if not selected or remaining < 1 or sel.unknown or not changes:
        return selected
    if not all(
        c["old"] == c["new"] and c["old"].startswith(("sqlglot/", "tests/fixtures/"))
        for c in changes
    ):
        return selected
    # Reconstruct every hunk before allowing a native pass to narrow anything.
    for change in changes:
        raw, _ = sel.source(change["old"])
        apply(raw, change)
    stop = time.monotonic() + remaining

    cached_cost = getattr(sel, "verification_cost", None)
    if cached_cost is None:
        cached_cost = sel.verification_cost = catalog_price(sel.by_number)

    def cost(numbers):
        return cached_cost(frozenset(numbers))

    selected = attribute_cases.refine(sel, changes, selected, apply, equal)
    confirmed = sel.confirmed_failures
    selected = set(selected) | confirmed
    if confirmed and cost(selected) - cost(confirmed) < 350:
        return selected
    fixture_only = all(c["old"].startswith("tests/fixtures/") for c in changes)
    small_fixture = fixture_only and cost(selected) < 3800
    if small_fixture and not (
        use_cases
        and len(changes) == 1
        and sel.views
        and changes[0]["old"] in sel.views.inputs_changed
        and not confirmed
    ):
        return selected
    span_info = None
    if use_cases:
        selected = dispatch_delta.refine(
            sel, changes, selected, apply, equal, checkout, stop - time.monotonic()
        )
        selected, span_info = symbol_impact.refine(
            sel, changes, selected, apply, equal, checkout, stop - time.monotonic()
        )
        selected = cold_inputs.refine(
            sel, changes, selected, apply, equal, checkout, cost, remaining
        )
    tasks = (
        case_verify.jobs_for(
            sel, selected - span_info[1], changes, apply, equal, span_info[0]
        )
        if use_cases and span_info
        else (
            case_verify.jobs_for(sel, selected, changes, apply, equal)
            if use_cases
            else []
        )
    )
    if use_cases:
        from optimizer_session import tasks as native_optimizer_tasks

        tasks = native_optimizer_tasks(sel, selected, tasks, changes)
    if small_fixture:
        # Verify the whole method when its passing result could eliminate
        # the last worker. Use a pristine native rehearsal to price the check.
        import optimizer_session

        prefix = case_verify.fixture_cases.PREFIX
        names = [sel.by_number[n]["id"] for n in sorted(selected)]
        if all(name.startswith(prefix) for name in names):
            costs = optimizer_session.profile()[3]
            if any(name[len(prefix) :] not in costs for name in names):
                return selected
            path = changes[0]["old"]
            fixture = path[len("tests/fixtures/"):]
            readers = case_verify.fixture_cases.profile()["methods"]
            tasks = []
            for name in names:
                method = name[len(prefix):]
                row = {"kind": "outside", "key": name}
                if fixture in readers.get(method, ()):
                    row["new_query_inputs"] = sel.views.new_query_inputs.get(path, ())
                tasks.append((name, [row], costs[method]))
        elif not use_native or any(
            name not in baseline_costs()["methods"] for name in names
        ):
            return selected
    source_readers = set()
    for change in changes:
        path = change["old"]
        if path.startswith("sqlglot/") and path in sel.files:
            source_readers.update(sel.sets[sel.files[path]])
    tasks = [t for t in tasks if sel.by_name[t[0]] not in source_readers | confirmed]
    pending = []
    for name, rows, cpu in tasks:
        if not rows:
            selected.discard(sel.by_name[name])
        else:
            pending.append((name, rows, cpu))
    env = dict(
        os.environ,
        PYTHON_CPU_COUNT="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
        SKIP_INTEGRATION="0",
    )
    with tempfile.TemporaryDirectory(prefix="br-verify-") as d:
        tmp = Path(d)
        root = tmp / "repo"

        def prepare():
            if not root.exists():
                checkout(root, changes, sel)
            env["PYTHONPATH"] = str(root)

        # Optimize the *combined* saving. Two long tests can mask one another.
        winners, best = [], 0
        # Evaluate single jobs and prefixes by measured work per recorded second.
        # This also finds jointly useful checks when two long tests mask each other.
        ranked = sorted(
            pending,
            key=lambda t: (
                (t[2] + 20) / max(1, sel.by_number[sel.by_name[t[0]]]["seconds"] * 1000)
            ),
        )
        subsets = [[t] for t in pending] + [
            ranked[:i] for i in range(2, len(ranked) + 1)
        ]

        def startup_ms(subset):
            if any(
                name.startswith(case_verify.fixture_cases.PREFIX)
                for name, _, _ in subset
            ):
                from rule_replay import ledger

                return max(
                    150, ledger().execute("SELECT ms FROM fixture_setup").fetchone()[0]
                )
            return 150

        for subset in subsets:
            expected = startup_ms(subset) + sum(t[2] for t in subset)
            saving = cost(selected) - cost(
                selected - {sel.by_name[t[0]] for t in subset}
            )
            value = 0.78 * saving - 1.25 * expected
            if small_fixture and len(subset) == len(selected):
                value = 0.85 * saving - expected
            if all(
                name.startswith(case_verify.fixture_cases.PREFIX)
                and 0 < cpu <= 350
                and all(row["kind"] == "fixture" for row in rows)
                for name, rows, cpu in subset
            ):
                value = saving - 1.05 * expected
            if expected <= 7500 and value > max(best, 300):
                winners, best = subset, value
        if winners:
            winners.sort(
                key=lambda t: (
                    -(cost(selected) - cost(selected - {sel.by_name[t[0]]}))
                    / (100 + t[2])
                )
            )
            expense = (
                max(0.55, 0.25 + 1.50 * startup_ms(winners) / 1000)
                + 1.65 * sum(t[2] for t in winners) / 1000
            )
            prepare()
            out = tmp / "cases"
            forkprobe.run(
                (
                    root,
                    winners,
                    out,
                    expense,
                    {
                        sel.by_number[n]["id"]: round(
                            1000 * sel.by_number[n]["seconds"]
                        )
                        for n in selected
                    },
                    all(c["old"] == c["new"] and c["old"].startswith("sqlglot/optimizer/") for c in changes),
                ),
                env,
                case_verify.worker,
                min(stop - time.monotonic(), 3 + 2.1 * expense),
            )
            progress = Path(str(out)+".queries")
            if progress.is_file():
                ledger = changes[0].setdefault("_native_query_progress",set())
                for line in progress.read_text().splitlines():
                    fields=line.split()
                    if len(fields)==2 and fields[0].isdigit() and fields[1] in ("parse","optimize","worker","duck"):
                        ledger.add((int(fields[0]),fields[1]))
            if out.exists():
                for line in out.read_text().splitlines():
                    selected.discard(sel.by_name[json.loads(line)])
        if use_native and time.monotonic() < stop and selected:
            plan = native_plan(sel, selected | confirmed, cost, confirmed)
            if plan:
                groups, ms = plan
                sel.native_attempted.update(sel.by_name[f"{f}::{c}::{m}"] for f,c,names in groups for m in names)
                prepare()
                out = tmp / "methods"
                forkprobe.run(
                    (root, groups, out, ms / 1000),
                    env,
                    native_tests.worker,
                    min(stop - time.monotonic(), 1.4 + 2 * ms / 1000),
                )
                if out.exists():
                    for line in out.read_text().splitlines():
                        file, cls, methods = json.loads(line)
                        selected.difference_update(
                            sel.by_name[f"{file}::{cls}::{m}"] for m in methods
                        )
    return selected
