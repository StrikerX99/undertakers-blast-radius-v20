"""Compose pinned structural access sets; a patch must qualify in full."""

import ast
from pathlib import Path
from functools import lru_cache
import json
from source_fast import analysis_tree
import expression_schema, registry_keys, long_reach, closure_impact


@lru_cache(maxsize=1)
def rows():
    f = Path(__file__).with_name("schema-contexts.json")
    return json.loads(f.read_bytes()) if f.is_file() else {}


@lru_cache(maxsize=1)
def binding_reads():
    return json.loads(Path(__file__).with_name("binding-reads.json").read_bytes())


def schema(sel, c, apply, equal):
    src = c["old"]
    result = expression_schema.verify(sel, [c], set(sel.all), apply, equal)
    if result != sel.all:
        return result
    raw, a = sel.source(src)
    b = analysis_tree(c, a, raw, apply(raw, c)[0])
    if (
        len(a.body) != len(b.body)
        or src not in rows().get("classes", {})
        or src in sel.files
        or rows()["sources"].get(src) != sel.mapping["source_hashes"].get(src)
        or rows()["pin"] != sel.mapping["commit"]
    ):
        return None
    ext = sel.dispatch_map()
    readers = ext.external | ext.external_by_source.get(src, set())
    for old, new in zip(a.body, b.body):
        if equal(old, new):
            continue
        left, right = expression_schema.fields(old), expression_schema.fields(new)
        if (
            not left
            or not right
            or old.name != new.name
            or not equal(old.bases, new.bases)
            or left[0] != right[0]
        ):
            return None
        if left[1] == right[1]:
            continue
        row = rows()["classes"][src].get(old.name)
        if row is None:
            return None
        if left[1] is None or right[1] is None:
            return None
        prior, after = dict(left[1]), dict(right[1])
        if any((key in prior) != (key in after) for key in row["members"]):
            return None
        if row["required"] and [(k, v) for k, v in prior.items() if v] != [
            (k, v) for k, v in after.items() if v
        ]:
            return None
        readers.update(sel.by_name[n] for n in row["readers"] if n in sel.by_name)
    return readers


def function(sel, c, apply, equal):
    src = c["old"]
    ed = closure_impact.spans(sel, [c], apply, equal)
    if not ed:
        return None
    lines = ed[src]
    sel.load_shard(src)
    if set(sel.mapping.get("import_lines", {}).get(src, ())).intersection(lines):
        return None
    extern = sel.dispatch_map()
    # Retain every external subprocess not proven unaffected by its complete cold and warm traces.
    users = (
        sel.line_tests(src, lines)
        | extern.external
        | extern.external_by_source.get(src, set())
    )
    if src in sel.files:
        users.update(sel.sets[sel.files[src]])
    _, names, _ = long_reach.index()
    cold = {sel.by_name[n] for n in names if n in sel.by_name}
    unused = {
        sel.by_name[n] for n in long_reach.unused(sel.mapping, ed) if n in sel.by_name
    }
    return (users | cold) - unused


def registry(sel, c, apply, equal, all_changes, checkout, root, remaining):
    import import_alias

    p = c["old"]
    raw, a = sel.source(p)
    edited = apply(raw, c)[0]
    b = analysis_tree(c, a, raw, edited)
    old_imports = {
        ast.dump(n): n for n in a.body if isinstance(n, (ast.Import, ast.ImportFrom))
    }
    new_imports = {
        ast.dump(n) for n in b.body if isinstance(n, (ast.Import, ast.ImportFrom))
    }
    removed = old_imports.keys() - new_imports
    if len(removed) > 1 or new_imports - old_imports.keys():
        return None
    guard = set()
    if removed:
        node = old_imports[next(iter(removed))]
        if (
            not isinstance(node, ast.ImportFrom)
            or node.level
            or not node.module
            or len(node.names) != 1
        ):
            return None
        name = node.names[0].name
        alias = node.names[0].asname or name
        mod = p[:-3].replace("/", ".")
        if alias == "*" or getattr(node, "lineno", None) is None:
            return None
        data = import_alias.data()
        provider = data["providers"].get(node.module, {})
        if (
            name not in provider.get("functions", ())
            or provider["sha256"] != sel.mapping["source_hashes"].get(provider["path"])
            or data["eligible"].get(mod) != sel.mapping["source_hashes"].get(p)
        ):
            return None
        # Unchanged deferred callbacks can start raising NameError after an import
        # is removed. Preserve all readers of their keys and of the module binding.
        ofn = [
            n
            for n in ast.walk(a)
            if isinstance(n, ast.Name) and n.id == alias and isinstance(n.ctx, ast.Load)
        ]
        if not ofn:
            return None
        keys = {}
        covered = set()
        for owner in a.body:
            if not isinstance(owner, ast.ClassDef):
                continue
            for statement in owner.body:
                if registry_keys.binding(statement) != "FUNCTIONS" or not isinstance(
                    statement.value, ast.Dict
                ):
                    continue
                for k, v in zip(statement.value.keys, statement.value.values):
                    if (
                        not isinstance(k, ast.Constant)
                        or type(k.value) is not str
                        or not isinstance(v, ast.Lambda)
                    ):
                        continue
                    sites = {
                        n
                        for n in ast.walk(v.body)
                        if isinstance(n, ast.Name) and n.id == alias
                    }
                    if sites:
                        covered.update(sites)
                        keys[k.value] = mod + ":" + owner.name
        if set(ofn) != covered:
            return None
        info = registry_keys.profile()
        prog = registry_keys.baseline(
            root, "FUNCTIONS", tuple(sorted(keys)), tuple(info["rows"])
        )
        if not prog or any(n not in prog for n in keys.values()):
            return None
        owners = {}
        for n, row in info["rows"].items():
            if n not in prog:
                if not n.startswith("sqlglot."):
                    guard.update(
                        sel.by_name[info["tests"][i]]
                        for i in info["sets"][info["activity"][n]]
                        if info["tests"][i] in sel.by_name
                    )
                continue
            hit = {
                k
                for k, owner in keys.items()
                if prog[n][k] in (prog[owner][k], registry_keys.AMBIGUOUS)
            }
            if not hit:
                continue
            owners[n] = hit
            for k in hit:
                if k in row:
                    guard.update(
                        sel.by_name[info["tests"][i]]
                        for i in info["sets"][row[k]]
                        if info["tests"][i] in sel.by_name
                    )
        extra = registry_keys.direct_users(sel, "FUNCTIONS", owners, set(keys))
        if extra is None:
            return None
        guard.update(extra)
        lookup = binding_reads()
        readers = lookup["mod"].get(mod, {})
        rawset = set(data["unobserved"]) | set(lookup["missing"])
        for key in (alias, "__dict__", "@source"):
            rawset.update(readers.get(key, ()))
        if "@startup" in rawset:
            return None
        guard.update(sel.by_name[n] for n in rawset if n in sel.by_name)
    kept = ast.Module(
        body=[n for n in a.body if n is not node] if removed else a.body,
        type_ignores=[],
    )
    query = registry_keys.edits(p, kept, b, equal)
    if not query or len(query) != 1:
        return None
    # Re-use registry read logic on a projection used only for shape recognition.
    # The semantic snapshot always imports the complete, unmodified composite patch.
    filtered_raw = "".join(
        line
        for i, line in enumerate(raw.splitlines(keepends=True), 1)
        if not removed or not node.lineno <= i <= node.end_lineno
    )

    class View:
        def __getattr__(self, k):
            return getattr(sel, k)

        def source(self, path):
            if path == p:
                return filtered_raw, kept
            return sel.source(path)

    record = dict(c)
    record["_compiled_source"] = edited
    record["_paired_tree"] = (kept, edited, b)
    full_checkout = lambda dst, changes, view: checkout(dst, all_changes, sel)
    ids = registry_keys.refine(
        View(),
        [record],
        set(sel.all),
        lambda text, item: (edited, set(), set()),
        equal,
        full_checkout,
        root,
        remaining,
    )
    return ids | guard


def refine(sel, changes, selected, apply, equal, checkout, root, remaining):
    if (
        len(changes) < 2
        or len(changes) > 5
        or len(selected) < 150
        or remaining < 5
        or sel.unknown
    ):
        return selected
    if not any(c["old"].startswith("sqlglot/expressions/") for c in changes):
        return selected
    union = set(sel.confirmed_failures)
    for c in changes:
        p = c["old"]
        if p != c["new"] or not p.endswith(".py"):
            return selected
        if p.startswith("sqlglot/expressions/"):
            r = schema(sel, c, apply, equal)
        elif p.startswith(("sqlglot/parsers/", "sqlglot/generators/")):
            r = function(sel, c, apply, equal)
            if r is None and p.startswith("sqlglot/parsers/"):
                r = registry(sel, c, apply, equal, changes, checkout, root, remaining)
        else:
            r = None
        if r is None or r == sel.all:
            return selected
        union.update(r)
    if not selected - union:
        return selected
    import selector as br

    oldmods = sel.import_modules
    try:
        sel.import_modules = {c["old"][:-3].replace("/", ".") for c in changes}
        if not br.verify_imports(sel, changes):
            return selected
    finally:
        sel.import_modules = oldmods
    return selected & union
