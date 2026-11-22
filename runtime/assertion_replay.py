"""Index independent SQL validator assertions; run the affected original assertions on the patch."""

import ast
import importlib
import marshal
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest
import zlib
from collections import defaultdict
from functools import lru_cache

TARGET = Path(__file__).with_name("assertion-coverage.bin")
HELPERS = {"validate_identity", "validate_all", "validate_transpile"}
MAGIC = "validator-calls-v3"


def independent(method):
    if method.decorator_list or not method.body or not method.name.startswith("test_"):
        return False
    for stmt in method.body:
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            return False
        call = stmt.value
        if not (
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "self"
            and call.func.attr in HELPERS
        ):
            return False
        if any(kw.arg is None for kw in call.keywords):
            return False
        try:
            for val in [*call.args, *(kw.value for kw in call.keywords)]:
                ast.literal_eval(val)
        except (ValueError, TypeError):
            return False
    return True


def eligible_methods(root):
    from verification_budget import baseline_costs

    ordinary = baseline_costs()["methods"]
    paths = {name.split("::")[0] for name in ordinary}
    paths.update(
        p.relative_to(root).as_posix()
        for p in (root / "tests/dialects").glob("test_*.py")
    )
    for rel in sorted(paths):
        f = root / rel
        try:
            mod = importlib.import_module(rel[:-3].replace("/", "."))
        except (ImportError, OSError):
            continue
        if hasattr(mod, "setUpModule") or hasattr(mod, "tearDownModule"):
            continue
        for owner in ast.parse(f.read_bytes()).body:
            if not isinstance(owner, ast.ClassDef):
                continue
            cls = getattr(mod, owner.name, None)
            from tests.dialects.test_dialect import Validator

            if not isinstance(cls, type) or not issubclass(cls, unittest.TestCase):
                continue
            if any(
                name in c.__dict__
                for c in cls.__mro__
                if c not in (object, unittest.TestCase)
                for name in (
                    "setUp",
                    "tearDown",
                    "setUpClass",
                    "tearDownClass",
                    "run",
                    "__init__",
                )
            ):
                continue
            validator = issubclass(cls, Validator) and all(
                getattr(cls, name) is getattr(Validator, name) for name in HELPERS
            )
            for node in owner.body:
                if not isinstance(node, ast.FunctionDef) or not node.name.startswith(
                    "test_"
                ):
                    continue
                split = validator and independent(node)
                name = "::".join((rel, cls.__name__, node.name))
                if not split and (
                    name not in ordinary or ordinary[name] > 800 or node.decorator_list
                ):
                    continue
                # Do not infer absence of a dependency for introspection, mocks,
                # cross-process tests, global state changes or direct source access.
                forbidden = {
                    "inspect",
                    "getattr",
                    "setattr",
                    "delattr",
                    "open",
                    "patch",
                    "mock",
                    "mro",
                    "globals",
                    "vars",
                    "settrace",
                    "getsource",
                    "__dict__",
                    "__code__",
                    "__class__",
                    "__subclasses__",
                    "modules",
                    "environ",
                    "__file__",
                    "get_context",
                    "subprocess",
                    "Process",
                    "getrefcount",
                    "skipTest",
                    "import_module",
                    "all_classes",
                }
                if not split and any(
                    isinstance(
                        z, (ast.Global, ast.Nonlocal, ast.Import, ast.ImportFrom)
                    )
                    or isinstance(z, ast.Name)
                    and z.id in forbidden
                    or isinstance(z, ast.Attribute)
                    and z.attr in forbidden
                    for z in ast.walk(node)
                ):
                    continue
                yield rel, cls, node, split


def dependent_modules(root):
    """Static imports between dialect implementations, including copied class tables."""
    edges = defaultdict(set)
    specialized = ("sqlglot/dialects/", "sqlglot/generators/", "sqlglot/parsers/")

    def imports(node):
        if isinstance(node, ast.If) and (
            isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
            or isinstance(node.test, ast.Attribute)
            and node.test.attr == "TYPE_CHECKING"
        ):
            return
        if isinstance(node, ast.Import):
            yield from (n.name for n in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            yield node.module
            yield from (node.module + "." + n.name for n in node.names)
        for child in ast.iter_child_nodes(node):
            yield from imports(child)

    for f in (root / "sqlglot").rglob("*.py"):
        path = f.relative_to(root).as_posix()
        for mod in imports(ast.parse(f.read_bytes())):
            rel = mod.replace(".", "/") + ".py"
            if rel.startswith(specialized) and rel != path:
                edges[rel].add(path)
    result = {}
    for f in (root / "sqlglot").rglob("*.py"):
        path = f.relative_to(root).as_posix()
        if not path.startswith(specialized) or f.name in ("__init__.py", "dialect.py"):
            continue
        seen = {path}
        todo = [path]
        while todo:
            for other in edges[todo.pop()] - seen:
                seen.add(other)
                todo.append(other)
        if all(
            k.startswith(specialized)
            and not k.endswith(("/__init__.py", "/dialect.py"))
            for k in seen
        ):
            result[path] = tuple(sorted(seen))
    return result


def build(root, dest):
    os.chdir(root)
    sys.path.insert(0, str(root))
    watch = sys.monitoring
    tool = watch.COVERAGE_ID
    watch.use_tool_id(tool, "independent assertions")
    bit = 1
    footprint = defaultdict(int)
    prefix = str(root) + "/sqlglot/"

    def enter(code, offset):
        filename = code.co_filename
        if filename.startswith(prefix):
            key = (filename[len(str(root)) + 1 :], code.co_qualname)
            footprint[key] |= bit
        return watch.DISABLE

    watch.register_callback(tool, watch.events.PY_START, enter)
    watch.set_events(tool, watch.events.PY_START)
    try:
        methods = list(eligible_methods(root))
        from sqlglot.dialects.dialect import Dialect, Dialects

        for dialect in Dialects:
            Dialect.get_or_raise(dialect.value)
        descriptor = Dialect.__dict__["get_or_raise"]
        original = descriptor.__func__

        @classmethod
        def note_dialect(cls, *args, **kwargs):
            instance = original(cls, *args, **kwargs)
            for owner in (
                type(instance),
                instance.parser_class,
                instance.generator_class,
                instance.tokenizer_class,
            ):
                for parent in owner.__mro__:
                    if parent.__module__.startswith("sqlglot."):
                        path = parent.__module__.replace(".", "/") + ".py"
                        footprint[path, "@dialect"] |= bit
            return instance

        Dialect.get_or_raise = note_dialect
        import gc
        import functools

        caches = [
            obj
            for obj in gc.get_objects()
            if type(obj) is functools._lru_cache_wrapper
            and getattr(obj, "__module__", "").startswith("sqlglot")
        ]

        def cold():
            for obj in caches:
                obj.cache_clear()

        data = {}
        next_index = 1
        for rel, cls, method, split in methods:
            ident = "::".join((rel, cls.__name__, method.name))
            times = []
            instance = cls(method.name)
            # Execute the real unittest method. Its helper calls mark independent
            # statement boundaries, including each helper's validation subtests.
            if not split:
                cold()
                bit = 1 << next_index
                watch.restart_events()
                outcome = unittest.TestResult()
                start = time.process_time()
                instance.run(outcome)
                ms = max(1, round(1000 * (time.process_time() - start)))
                if outcome.wasSuccessful() and not outcome.skipped:
                    data[ident] = (next_index, (ms,), False)
                next_index += 1
                bit = 1
                continue
            saved = {name: getattr(instance, name) for name in HELPERS}
            count = len(method.body)

            def wrap(fn):
                def checked(*args, **kwargs):
                    nonlocal bit
                    cold()
                    bit = 1 << (next_index + len(times))
                    watch.restart_events()
                    start = time.process_time()
                    try:
                        return fn(*args, **kwargs)
                    finally:
                        times.append(
                            max(1, round((time.process_time() - start) * 1000))
                        )
                        bit = 1

                return checked

            for name, body in saved.items():
                setattr(instance, name, wrap(body))
            outcome = unittest.TestResult()
            instance.run(outcome)
            # An unexpected nested validation call is ineligible for slicing.
            if outcome.wasSuccessful() and not outcome.skipped and len(times) == count:
                data[ident] = (next_index, tuple(times), True)
            next_index += max(len(times), count, 1)
            bit = 1
        Dialect.get_or_raise = descriptor
        for (path, qualname), bits in list(footprint.items()):
            while ".<locals>." in qualname:
                qualname = qualname.rsplit(".<locals>.", 1)[0]
                footprint[path, qualname] |= bits
        modules = defaultdict(int)
        for (path, _), bits in footprint.items():
            modules[path] |= bits & ~1
        packed = (MAGIC, data, dict(footprint), dict(modules), dependent_modules(root))
        dest.write_bytes(zlib.compress(marshal.dumps(packed), 6))
        print(
            f"Indexed {len(data)} methods, {sum(len(t) for _, t, sliced in data.values() if sliced)} independent assertions",
            flush=True,
        )
    finally:
        watch.set_events(tool, 0)
        watch.free_tool_id(tool)


@lru_cache(maxsize=1)
def index():
    version, methods, footprints, modules, reverse = marshal.loads(
        zlib.decompress(TARGET.read_bytes())
    )
    if version != MAGIC:
        raise ValueError("Unsupported assertion index")
    return methods, footprints, modules, reverse


def changed_functions(before, after, equal):
    """Existing ordinary definitions only. Changes to declarations or imports use full methods."""
    out = set()

    def visit(old, new, scope):
        if len(old) != len(new):
            return False
        for a, b in zip(old, new):
            if equal(a, b):
                continue
            if type(a) is not type(b) or not isinstance(
                a, (ast.ClassDef, ast.FunctionDef)
            ):
                return False
            if any(
                not equal(getattr(a, k), getattr(b, k))
                for k in a._fields
                if k != "body"
            ):
                return False
            name = ".".join((*scope, a.name))
            if isinstance(a, ast.ClassDef):
                if not visit(a.body, b.body, (*scope, a.name)):
                    return False
            else:
                # Entering a generator's body and calling it have different semantics.
                # Docstrings can be inspected by code which never enters this function.
                if ast.get_docstring(a, clean=False) != ast.get_docstring(
                    b, clean=False
                ):
                    return False
                if any(
                    isinstance(
                        v,
                        (ast.Yield, ast.YieldFrom, ast.Await, ast.Global, ast.Nonlocal),
                    )
                    for fn in (a, b)
                    for v in ast.walk(fn)
                ):
                    return False
                if a.name.startswith("__") and a.name.endswith("__"):
                    return False
                out.add(name)
        return True

    return out if visit(before.body, after.body, ()) else None


def assertion_mask(sel, changes, apply, equal, footprints, modules, reverse):
    from source_fast import paired_tree

    mask = 0
    class_data = False
    for c in changes:
        src = c["old"]
        if src != c["new"] or not src.startswith("sqlglot/") or not src.endswith(".py"):
            return None
        import re

        if any(
            re.search(
                r"\b(?:globals|setattr|delattr|environ|sys|sleep|settrace|setprofile|tracemalloc|mock|open|register|__dict__|__code__|__subclasses__)\b",
                line[1:],
            )
            for h in c["hunks"]
            for line in h["lines"]
            if line[:1] in ("+", "-")
        ):
            return None
        raw, before = sel.source(src)
        new = apply(raw, c)[0]
        after = paired_tree(c, before, raw, new)
        changed = changed_functions(before, after, equal)
        if changed is None:
            from module_footprints import broad_class

            if src not in reverse or not broad_class(sel, c, before, after, equal):
                return None
            class_data = True
            for path in reverse[src]:
                mask |= modules.get(path, 0)
            continue
        for name in changed:
            mask |= footprints.get((src, name), 0)
    return None if mask & 1 else (mask, class_data)


def run_subset(args):
    root, rows, target, cpu, imports = args
    import native_tests

    signal.setitimer(signal.ITIMER_PROF, cpu)
    guard = native_tests.watch_reference(root)
    cache = {}
    for module in imports:
        importlib.import_module(module)
    if guard.missing:
        return
    with target.open("w", buffering=1) as out:
        for number, filename, owner, name, indices in rows:
            if guard.missing:
                return
            try:
                mod = importlib.import_module(filename[:-3].replace("/", "."))
                cls = getattr(mod, owner)
                if indices == ():
                    if guard.missing:
                        return
                    print(number, file=out)
                    continue
                instance = cls(name)
                if indices is not None:
                    if filename not in cache:
                        cache[filename] = {
                            (c.name, m.name): m
                            for c in ast.parse((root / filename).read_bytes()).body
                            if isinstance(c, ast.ClassDef)
                            for m in c.body
                            if isinstance(m, ast.FunctionDef)
                        }
                    import copy
                    import types

                    method = copy.copy(cache[filename][owner, name])
                    method.body = [method.body[i] for i in indices]
                    scope = {}
                    exec(
                        compile(ast.Module([method], []), str(root / filename), "exec"),
                        mod.__dict__,
                        scope,
                    )
                    setattr(instance, name, types.MethodType(scope[name], instance))
                outcome = unittest.TestResult()
                instance.run(outcome)
                if (
                    outcome.wasSuccessful()
                    and not outcome.skipped
                    and not guard.missing
                ):
                    print(number, file=out)
            except Exception:
                continue


def refine(sel, changes, selected, apply, equal, checkout, remaining):
    if not selected or remaining < 0.8 or sel.unknown:
        return selected
    if not all(
        c["old"] == c["new"]
        and c["old"].startswith("sqlglot/")
        and c["old"].endswith(".py")
        for c in changes
    ):
        return selected
    methods, footprints, modules, reverse = index()
    possible = [
        (t, methods[name])
        for t in selected - sel.confirmed_failures
        if (name := sel.by_number[t]["id"]) in methods
    ]
    if not possible:
        return selected
    from verification_budget import catalog_price

    price = getattr(sel, "verification_cost", None)
    if price is None:
        price = sel.verification_cost = catalog_price(sel.by_number)
    selected = frozenset(selected)
    potential = price(selected) - price(selected.difference(t for t, _ in possible))
    if potential < 400:
        return set(selected)
    mask = assertion_mask(sel, changes, apply, equal, footprints, modules, reverse)
    if mask is None:
        return set(selected)
    mask, class_data = mask
    jobs = []
    for t, (start, times, sliced) in possible:
        if any(
            c["old"] in sel.files and t in sel.sets[sel.files[c["old"]]]
            for c in changes
        ):
            continue
        if class_data and not sliced:
            continue
        active = (mask >> start) & ((1 << len(times)) - 1)
        if active == (1 << len(times)) - 1:
            continue
        indices = tuple(i for i in range(len(times)) if (active >> i) & 1)
        ms = sum(times[i] for i in indices)
        number = sel.by_number[t]
        jobs.append((ms / max(1, round(number["seconds"] * 1000)), ms, t, indices))
    if not jobs:
        return set(selected)
    rows, chosen = [], []
    drop = set()
    spent = 120
    best = 180
    ordered = sorted(jobs)
    for i, (_, ms, t, indices) in enumerate(ordered, 1):
        spent += ms
        if spent > 3900:
            break
        rows.append((t, *sel.by_number[t]["id"].split("::"), indices))
        drop.add(t)
        if i < len(ordered) and (
            (ms == 0 and ordered[i][1] == 0)
            or (ms > 0 and i > 8 and i % 12 and sel.by_number[t]["seconds"] < 0.25)
        ):
            continue
        gain = price(selected) - price(selected.difference(drop)) - spent
        if gain > best:
            best = gain
            chosen = rows.copy()
            budget = min(4.5, 0.45 + 1.55 * spent / 1000)
    if not chosen:
        return set(selected)
    import forkprobe

    with tempfile.TemporaryDirectory(prefix="assertions-") as tmp:
        root = Path(tmp) / "src"
        target = Path(tmp) / "complete"
        checkout(root, changes, sel)
        imports = [c["old"][:-3].replace("/", ".") for c in changes]
        forkprobe.run(
            (root, chosen, target, budget, imports),
            {"PYTHONPATH": str(root), "SKIP_INTEGRATION": "0"},
            run_subset,
            min(sel.verify_wall_deadline - time.monotonic(), 1.0 + 1.7 * budget),
        )
        passed = (
            {int(x) for x in target.read_text().splitlines() if x.isdecimal()}
            if target.exists()
            else set()
        )
    return set(selected) - passed


if __name__ == "__main__":
    build(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
