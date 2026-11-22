"""Connect expressions constructed while importing SQLGlot to tests that subsequently use them."""

import ast
from collections import defaultdict
from functools import lru_cache
import importlib
import json
import marshal
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zlib

INDEX = Path(__file__).with_name("derived-values.bin")


def boot(root):
    os.chdir(root)
    sys.path.insert(0, str(root))
    from sqlglot.dialects.dialect import Dialect, Dialects

    for name in Dialects:
        Dialect.get_or_raise(name.value)


def inventory(root):
    boot(root)
    from sqlglot import exp
    from checkpoint_graph import expression_digest

    groups = defaultdict(list)
    hashes = {}
    for name, mod in tuple(sys.modules.items()):
        if not name.startswith("sqlglot.") or not getattr(mod, "__file__", None):
            continue
        stack = [("", vars(mod))]
        while stack:
            scope, ns = stack.pop()
            for attr, obj in ns.items():
                prefix = attr if not scope else scope + "." + attr
                if (
                    isinstance(obj, type)
                    and obj.__module__ == name
                    and obj.__qualname__ == prefix
                ):
                    stack.append((prefix, vars(obj)))
                if isinstance(obj, exp.Expr) and attr.isupper():
                    key = (name, scope, attr)
                    groups[id(obj)].append(key)
                    hashes[key] = expression_digest(obj)
    values = sorted(hashes)
    idx = {key: i for i, key in enumerate(values)}
    aliases = {
        k: tuple(idx[item] for item in group)
        for group in groups.values()
        for k in group
    }
    return values, [hashes[k] for k in values], aliases


def collect_calls(root, sites):
    active = {}
    sources = defaultdict(lambda: defaultdict(set))
    prefix = str(root) + "/"
    known = {}

    def entered(frame, event, arg):
        address = id(frame)
        if event == "return":
            active.pop(address, None)
            return
        if event != "call":
            return
        parent = frame.f_back
        inherited = active.get(id(parent), ()) if parent else ()
        if parent is not None:
            extra = sites.get(
                (parent.f_code.co_filename, parent.f_lineno, parent.f_code.co_name)
            )
            if extra:
                inherited = tuple(set(inherited) | set(extra))
        if not inherited:
            return
        active[address] = inherited
        code = frame.f_code
        if code not in known:
            filename = code.co_filename
            known[code] = (
                (
                    filename[len(prefix) :],
                    code.co_firstlineno,
                    max(
                        [code.co_firstlineno]
                        + [n for _, _, n in code.co_lines() if n is not None]
                    ),
                )
                if filename.startswith(prefix + "sqlglot/")
                else None
            )
        record = known[code]
        if record is not None:
            path, a, b = record
            sources[path][a, b].update(inherited)

    sys.setprofile(entered)
    try:
        boot(root)
    finally:
        sys.setprofile(None)
    return {
        p: [(a, b, tuple(sorted(ids))) for (a, b), ids in sorted(data.items())]
        for p, data in sources.items()
    }


def build(root, dest):
    values, hashes, aliases = inventory(root)
    attributes = defaultdict(set)
    for i, (_, _, attr) in enumerate(values):
        attributes[attr].add(i)
    sites = {}
    direct = defaultdict(set)
    reads = defaultdict(set)
    cover = json.loads(Path(__file__).with_name("coverage-map.json").read_bytes())
    _, methods, footprints, *_ = marshal.loads(
        zlib.decompress(Path(__file__).with_name("assertion-coverage.bin").read_bytes())
    )

    def add_read(path, scope, stmt, numbers):
        if path.startswith("tests/"):
            name = path + "::" + scope.replace(".", "::")
            if name in cover["tests"]:
                direct_ids = {name}
            else:
                direct_ids = {s for s in cover["tests"] if s.startswith(path + "::")}
        else:
            direct_ids = set()
            table = cover["lines"].get(path, {})
            for line in range(stmt.lineno, stmt.end_lineno + 1):
                idx = table.get(str(line))
                if idx is not None:
                    direct_ids.update(cover["tests"][j] for j in cover["sets"][idx])
            bits = footprints.get((path, scope), 0)
            if bits:
                direct_ids.update(
                    n
                    for n, (start, clock, _) in methods.items()
                    if ((bits >> start) & ((1 << len(clock)) - 1))
                )
        for i in numbers:
            reads[i].update(direct_ids)

    for folder in ("sqlglot", "tests"):
        for f in sorted((root / folder).rglob("*.py")):
            path = f.relative_to(root).as_posix()
            mod = path[:-3].replace("/", ".").removesuffix(".__init__")

            def walk(stmts, scope="", holder="<module>"):
                for stmt in stmts:
                    if isinstance(
                        stmt, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                    ):
                        walk(
                            stmt.body,
                            scope + "." + stmt.name if scope else stmt.name,
                            stmt.name,
                        )
                        continue
                    if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                        for target in (
                            stmt.targets
                            if isinstance(stmt, ast.Assign)
                            else [stmt.target]
                        ):
                            if (
                                isinstance(target, ast.Name)
                                and (mod, scope, target.id) in aliases
                            ):
                                group = aliases[mod, scope, target.id]
                                for l in range(stmt.lineno, stmt.end_lineno + 1):
                                    sites[str(f), l, holder] = group
                                    direct[path, l].update(group)
                    wanted = set()
                    for v in ast.walk(stmt):
                        if isinstance(v, ast.Attribute) and isinstance(v.ctx, ast.Load):
                            wanted.update(attributes.get(v.attr, ()))
                        if isinstance(v, ast.Name) and isinstance(v.ctx, ast.Load):
                            wanted.update(attributes.get(v.id, ()))
                        if isinstance(v, ast.Constant) and type(v.value) is str:
                            wanted.update(attributes.get(v.value, ()))
                    if wanted:
                        add_read(path, scope, stmt, wanted)

            walk(ast.parse(f.read_bytes()).body)
    with tempfile.TemporaryDirectory() as tmp:
        context = Path(tmp) / "context"
        context.write_bytes(marshal.dumps(sites))
        subprocess.run(
            [sys.executable, __file__, "trace", str(root), str(context)],
            check=True,
            timeout=60,
        )
        sources = marshal.loads(context.read_bytes())
    store = (
        1,
        values,
        hashes,
        sources,
        {k: tuple(sorted(v)) for k, v in direct.items()},
        {i: tuple(sorted(v)) for i, v in reads.items()},
    )
    dest.write_bytes(zlib.compress(marshal.dumps(store), 7))
    print(
        "Indexed",
        len(values),
        "derived bindings across",
        len(sources),
        "producer files",
        flush=True,
    )


@lru_cache(maxsize=1)
def index():
    version, *parts = marshal.loads(zlib.decompress(INDEX.read_bytes()))
    if version != 1:
        raise ValueError("Unsupported derived values map")
    return parts


def compare(args):
    root, fields, marker = args
    boot(root)
    from checkpoint_graph import expression_digest

    changed = []
    for idx, (module, owner, name), digest in fields:
        try:
            item = importlib.import_module(module)
            for part in owner.split(".") if owner else ():
                item = vars(item)[part]
            if expression_digest(vars(item)[name]) == digest:
                continue
        except (KeyError, TypeError, ValueError, AttributeError, RecursionError):
            pass
        changed.append(idx)
    marker.write_text(json.dumps(changed))


def consumers(sel, changes, selected, apply, checkout, remaining):
    if not changes or not all(
        c["old"] == c["new"]
        and c["old"].startswith("sqlglot/")
        and c["old"].endswith(".py")
        for c in changes
    ):
        return set()
    fields, hashes, producers, definitions, readers = index()
    suspect = set()
    for c in changes:
        path = c["old"]
        if path not in producers and not any(k[0] == path for k in definitions):
            continue
        raw, _ = sel.source(path)
        _, old, new = apply(raw, c)
        changed = old | new
        for l in changed:
            suspect.update(definitions.get((path, l), ()))
        for begin, end, idx in producers.get(path, ()):
            if any(begin <= l <= end for l in changed):
                suspect.update(idx)
    missing = {
        i: {sel.by_name[n] for n in readers.get(i, ()) if n in sel.by_name} - selected
        for i in suspect
    }
    missing = {i: ids for i, ids in missing.items() if ids}
    if not missing:
        return set()
    worst = set().union(*missing.values())
    from verification_budget import catalog_price

    price = getattr(sel, "verification_cost", None)
    if price is None:
        price = sel.verification_cost = catalog_price(sel.by_number)
    if (
        remaining < 0.65
        or price(frozenset(selected | worst)) - price(frozenset(selected)) < 180
    ):
        return worst
    import forkprobe

    with tempfile.TemporaryDirectory(prefix="import-values-") as tmp:
        root = Path(tmp) / "src"
        marker = Path(tmp) / "changed"
        try:
            checkout(root, changes, sel)
        except (ValueError, OSError):
            return worst
        forkprobe.run(
            (root, [(i, fields[i], hashes[i]) for i in sorted(missing)], marker),
            {"PYTHONPATH": str(root)},
            compare,
            min(1.65, remaining),
        )
        if not marker.is_file():
            return worst
        return set().union(*(missing[i] for i in json.loads(marker.read_bytes())))


if __name__ == "__main__":
    if sys.argv[1] == "trace":
        dst = Path(sys.argv[3])
        data = collect_calls(Path(sys.argv[2]), marshal.loads(dst.read_bytes()))
        dst.write_bytes(marshal.dumps(data))
    else:
        build(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
