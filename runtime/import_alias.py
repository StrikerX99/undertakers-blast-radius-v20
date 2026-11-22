"""Recognize private function aliases added to or removed from existing unconditional from-imports."""

from functools import lru_cache
from pathlib import Path
import ast
import json


@lru_cache(maxsize=1)
def data():
    return json.loads(Path(__file__).with_name("alias-readers.json").read_bytes())


def path_module(p):
    return p[:-3].replace("/", ".")


def private(name):
    return (
        name.startswith("_")
        and not name.startswith("__")
        and name.isascii()
        and name.isidentifier()
    )


def changes_to_aliases(old, new, equal):
    if len(old.body) != len(new.body):
        return None
    edits = []
    for a, b in zip(old.body, new.body):
        if equal(a, b):
            continue
        if not (
            isinstance(a, ast.ImportFrom)
            and isinstance(b, ast.ImportFrom)
            and a.module
            and a.module == b.module
            and a.level == b.level == 0
        ):
            return None
        before = [(n.name, n.asname) for n in a.names]
        after = [(n.name, n.asname) for n in b.names]
        if any(n == "*" for n, _ in before + after):
            return None
        common = set(before) & set(after)
        if (
            not common
            or [x for x in before if x in common] != [x for x in after if x in common]
            or len(before) != len(set(before))
            or len(after) != len(set(after))
        ):
            return None
        for pair in set(before) ^ set(after):
            src, alias = pair
            if not alias or not private(alias):
                return None
            edits.append((a.module, src, alias))
    return edits or None


def observers(sel, changes, apply, equal):
    """Return required observer IDs or None when this proof does not apply."""
    if len(changes) != 1:
        return None
    c = changes[0]
    p = c["old"]
    if (
        not p
        or p != c["new"]
        or not p.startswith(("sqlglot/generators/", "sqlglot/parsers/"))
        or not p.endswith(".py")
    ):
        return None
    if all("import " not in line for h in c["hunks"] for line in h["lines"]):
        return None
    maps = data()
    mod = path_module(p)
    if (
        maps["eligible"].get(mod) != sel.mapping["source_hashes"].get(p)
        or mod not in maps["eligible"]
    ):
        return None
    raw, old = sel.source(p)
    new_text = apply(raw, c)[0]
    edits = changes_to_aliases(old, ast.parse(new_text), equal)
    if not edits:
        return None
    scan = maps["tokens"]
    required = set(maps["unobserved"])
    obs = maps["readers"].get(mod, {})
    for name in ("__dict__", "@source"):
        required.update(obs.get(name, ()))
    alias_names = [e[2] for e in edits]
    if len(set(alias_names)) != len(alias_names):
        return None
    for provider, name, alias in edits:
        # Pinned provider has been imported natively, is cycle-free relative
        # to these leaf modules, and exports this plain function directly.
        source = maps["providers"].get(provider, {})
        if name not in source.get("functions", ()) or sel.mapping["source_hashes"].get(
            source.get("path")
        ) != source.get("sha256"):
            return None
        import re

        token = rb"(?<!\w)" + re.escape(alias.encode()) + rb"(?!\w)"
        old_count = len(re.findall(token, raw.encode()))
        new_count = len(re.findall(token, new_text.encode()))
        if sorted((old_count, new_count)) != [0, 1]:
            return None
        # Includes private identifiers and strings in the pinned source, tests and fixtures.
        if scan.get(alias, 0) != old_count:
            return None
        required.update(obs.get(alias, ()))
    if "@startup" in required:
        return None
    return {sel.by_name[n] for n in required if n in sel.by_name} | set(sel.unknown)
