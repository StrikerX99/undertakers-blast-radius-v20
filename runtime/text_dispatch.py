"""Relate a changed string-keyed parser callback to the tests that consulted its key."""

import ast
import json
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import parser_cells
import reflective_use
from source_fast import paired_tree

HERE = Path(__file__).parent
PIN = "eb11aec4be8cb2122afe713a4816662c3f0fb2fa"


def declaration(node):
    if isinstance(node, ast.AnnAssign):
        target = node.target
    elif isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
    else:
        return None
    return target.id if isinstance(target, ast.Name) else None


def name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def build(root, specfile, outfile):
    spec = json.loads(specfile.read_bytes())
    tables = defaultdict(set)
    all_parser_types = {"Parser", *(pair[0] for pair in spec["permitted"].values())}
    trees = {
        f.relative_to(root).as_posix(): ast.parse(f.read_bytes())
        for f in sorted((root / "sqlglot").rglob("*.py"))
    }
    for module, tree in trees.items():
        if module.startswith("sqlglot/parsers/"):
            all_parser_types.update(
                stmt.name
                for stmt in tree.body
                if isinstance(stmt, ast.ClassDef) and stmt.name.endswith("Parser")
            )
    for path, (owner, _) in spec["permitted"].items():
        # A subclass can copy and modify the parent table at import. Initially
        # admit leaf classes only, and reject references to them from other class bodies.
        if len(spec["descendants"][path]) != 1:
            continue
        cls = next(
            (
                c
                for c in trees[path].body
                if isinstance(c, ast.ClassDef) and c.name == owner
            ),
            None,
        )
        if cls is None:
            continue
        for stmt in cls.body:
            field = declaration(stmt)
            value = getattr(stmt, "value", None)
            if (
                field
                and isinstance(value, ast.Dict)
                and all(
                    k is None or isinstance(k, ast.Constant) and type(k.value) is str
                    for k in value.keys
                )
            ):
                tables[field].add((path, owner))
    ineligible = defaultdict(set)
    observed_fields = set()
    for path, tree in trees.items():
        parents = {ch: n for n in ast.walk(tree) for ch in ast.iter_child_nodes(n)}
        for cell in ast.walk(tree):
            if (
                isinstance(cell, ast.Constant)
                and type(cell.value) is str
                and cell.value in tables
            ):
                ineligible[cell.value].update(tables[cell.value])
            if (
                isinstance(cell, ast.Name)
                and isinstance(cell.ctx, ast.Load)
                and cell.id in tables
            ):
                ineligible[cell.id].update(tables[cell.id])
            if not isinstance(cell, ast.Attribute) or cell.attr not in tables:
                continue
            field = cell.attr
            if (
                isinstance(cell.value, ast.Name)
                and cell.value.id == "self"
                and isinstance(cell.ctx, ast.Load)
            ):
                parent = parents[cell]
                is_match = (
                    isinstance(parent, ast.Call)
                    and isinstance(parent.func, ast.Attribute)
                    and parent.func.attr == "_match_texts"
                    and isinstance(parent.func.value, ast.Name)
                    and parent.func.value.id == "self"
                    and parent.args
                    and cell is parent.args[0]
                )
                if is_match:
                    observed_fields.add(field)
                # Every exact membership and subscript is observed by the collector.
                # Other uses may inspect the entire mapping. The trace marks their receivers opaque.
                if path == "sqlglot/parser.py" or path.startswith("sqlglot/parsers/"):
                    continue
            receiver = name(cell.value)
            if receiver in all_parser_types and isinstance(cell.ctx, ast.Load):
                for source, owner in tables[field]:
                    if receiver == owner:
                        ineligible[field].add((source, owner))
                continue
            ineligible[field].update(tables[field])
    output = defaultdict(list)
    for field in sorted(observed_fields):
        for path, owner in sorted(tables[field] - ineligible[field]):
            output[path].append(field)
    readers = defaultdict(set)
    for path in {s.split("::")[0] for s in spec["names"]}:
        raw = (root / path).read_bytes()
        for field in observed_fields:
            if field.encode() in raw:
                readers[field].add(path)
    outfile.write_text(
        json.dumps(
            {
                "pin": spec["pin"],
                "eligible": output,
                "reader_files": {k: sorted(v) for k, v in readers.items()},
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def unchanged_header(a, b, equal):
    return type(a) is type(b) and all(
        equal(getattr(a, k), getattr(b, k)) for k in a._fields if k != "body"
    )


def one_change(a, b, equal):
    if not unchanged_header(a, b, equal) or len(a.body) != len(b.body):
        return None
    changes = [(x, y) for x, y in zip(a.body, b.body) if not equal(x, y)]
    return changes[0] if len(changes) == 1 else None


def pure_callback(node):
    # A new plain lambda with one positional argument changes no import-time state.
    # Limit it to constructing a SQLGlot expression from constants in this parser.
    if not isinstance(node, ast.Lambda):
        return False
    a = node.args
    if (
        len(a.args) != 1
        or a.posonlyargs
        or a.vararg
        or a.kwonlyargs
        or a.kwarg
        or a.defaults
        or a.kw_defaults
    ):
        return False
    body = node.body
    if not (
        isinstance(body, ast.Call)
        and isinstance(body.func, ast.Attribute)
        and body.func.attr == "expression"
        and isinstance(body.func.value, ast.Name)
        and body.func.value.id == a.args[0].arg
        and len(body.args) == 1
        and not body.keywords
    ):
        return False
    make = body.args[0]
    return (
        isinstance(make, ast.Call)
        and isinstance(make.func, ast.Attribute)
        and isinstance(make.func.value, ast.Name)
        and make.func.value.id == "exp"
        and make.func.attr[:1].isupper()
        and make.func.attr.isidentifier()
        and not make.args
        and all(
            k.arg is not None and isinstance(k.value, ast.Constant)
            for k in make.keywords
        )
    )


def edit(before, after, clsname, allowed, equal):
    change = one_change(before, after, equal)
    if not change:
        return None
    a, b = change
    if not isinstance(a, ast.ClassDef) or a.name != clsname:
        return None
    change = one_change(a, b, equal)
    if not change:
        return None
    x, y = change
    f = declaration(x)
    if (
        f not in allowed
        or type(x) is not type(y)
        or any(
            not equal(getattr(x, k), getattr(y, k)) for k in x._fields if k != "value"
        )
    ):
        return None
    if not isinstance(x.value, ast.Dict) or not isinstance(y.value, ast.Dict):
        return None

    def cells(d):
        pairs = {}
        previous = []
        for k, val in zip(d.keys, d.values):
            if k is None:
                previous.append((None, val))
            elif (
                isinstance(k, ast.Constant)
                and type(k.value) is str
                and k.value not in pairs
            ):
                pairs[k.value] = val
                previous.append((k.value, val))
            else:
                return None, None
        return pairs, previous

    old, oldseq = cells(x.value)
    new, newseq = cells(y.value)
    if old is None or new is None:
        return None
    touched = {k for k in old.keys() | new.keys() if not equal(old.get(k), new.get(k))}
    if not touched or any(
        not pure_callback(d[k]) for d in (old, new) for k in touched if k in d
    ):
        return None

    def stable(seq):
        return [(k, v) for k, v in seq if k not in touched]

    p, q = stable(oldseq), stable(newseq)
    if len(p) != len(q) or any(
        k != j or not equal(v, w) for (k, v), (j, w) in zip(p, q)
    ):
        return None
    return f, touched


@lru_cache(maxsize=1)
def records():
    return (
        json.loads((HERE / "text-dispatch.json").read_bytes()),
        json.loads((HERE / "parser-word-reads.json").read_bytes()),
    )


def refine(sel, changes, selected, apply, equal):
    if not selected or sel.unknown or len(changes) != 1:
        return selected
    ch = changes[0]
    path = ch["old"]
    if path != ch["new"] or not path.startswith("sqlglot/parsers/"):
        return selected
    if not any(
        "lambda" in ln
        for h in ch["hunks"]
        for ln in h["lines"]
        if ln.startswith(("+", "-"))
    ):
        return selected
    static, reads = records()
    spec = parser_cells.data()
    if (
        PIN != sel.mapping["commit"]
        or static["pin"] != PIN
        or reads["pin"] != PIN
        or spec["pin"] != PIN
        or path not in static["eligible"]
        or spec["names"] != reads["tests"]
    ):
        return selected
    original, old = sel.source(path)
    updated = apply(original, ch)[0]
    parsed = edit(
        old,
        paired_tree(ch, old, original, updated),
        spec["permitted"][path][0],
        static["eligible"][path],
        equal,
    )
    if parsed is None:
        return selected
    field, keys = parsed
    if field not in reads["fields"]:
        return selected
    allowed = set(reads["opaque"])
    allowed.update(reflective_use.indices(sel, spec, {field}))
    by_receiver = reads["fields"][field]
    for receiver in [*spec["descendants"][path], "?"]:
        table = by_receiver.get(receiver, {})
        for key in keys:
            allowed.update(table.get(key, ()))
        allowed.update(reads.get("opaque_fields", {}).get(field, {}).get(receiver, ()))
    if -1 in allowed:
        return selected
    test_files = set(static["reader_files"].get(field, []))
    required = {
        sel.by_name[n] for i in allowed if (n := spec["names"][i]) in sel.by_name
    }
    required.update(t["number"] for t in sel.tests if t["file"] in test_files)
    if path in sel.files:
        required.update(sel.sets[sel.files[path]])
    return selected & required


if __name__ == "__main__":
    build(*(Path(x) for x in sys.argv[1:]))
