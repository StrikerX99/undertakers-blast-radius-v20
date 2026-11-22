"""Follow literal edits to parser token sets through their effective values and uses."""

import ast
import reflective_use
import json
import os
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import forkprobe

SOURCE = "sqlglot/parser.py"


@lru_cache(maxsize=1)
def profile():
    return json.loads(Path(__file__).with_name("grammar-cells.json").read_bytes())


def classes():
    from sqlglot.parser import Parser
    from sqlglot.dialects import Dialect, Dialects

    for dialect in Dialects:
        _ = Dialect.get_or_raise(dialect).parser_class
    todo, seen = [Parser], set()
    while todo:
        cls = todo.pop()
        if cls in seen:
            continue
        seen.add(cls)
        todo.extend(cls.__subclasses__())
    return seen


def snapshot(fields):
    from sqlglot.tokens import TokenType

    values = {}
    for cls in classes():
        name = cls.__module__ + "." + cls.__qualname__
        slots = {}
        for k in fields:
            s = getattr(cls, k, None)
            if s is not None:
                if type(s) not in (set, frozenset) or any(
                    type(t) is not TokenType for t in s
                ):
                    slots[k] = None
                else:
                    slots[k] = sorted(t.name for t in s)
        values[name] = slots
    return values


def probe(payload):
    root, fields, out = payload
    os.chdir(root)
    sys.path.insert(0, str(root))
    out.write_text(json.dumps(snapshot(fields)))


def declarations(sel, ch, apply, equal, name):
    before, old = sel.source(ch["old"])
    changed, _, _ = apply(before, ch)
    from source_fast import paired_tree

    new = paired_tree(ch, old, before, changed)
    if len(old.body) != len(new.body):
        return None
    pairs = [(a, b) for a, b in zip(old.body, new.body) if not equal(a, b)]
    if len(pairs) != 1:
        return None
    a, b = pairs[0]
    if (
        not isinstance(a, ast.ClassDef)
        or not isinstance(b, ast.ClassDef)
        or a.name != name
        or len(a.body) != len(b.body)
        or any(
            not equal(getattr(a, k), getattr(b, k)) for k in a._fields if k != "body"
        )
    ):
        return None
    fields = set()
    for x, y in zip(a.body, b.body):
        if equal(x, y):
            continue
        if (
            not isinstance(x, (ast.Assign, ast.AnnAssign))
            or type(x) is not type(y)
            or any(
                not equal(getattr(x, k), getattr(y, k))
                for k in x._fields
                if k != "value"
            )
        ):
            return None
        t = (
            x.target
            if isinstance(x, ast.AnnAssign)
            else x.targets[0]
            if len(x.targets) == 1
            else None
        )
        if not isinstance(t, ast.Name):
            return None

        def literal_edit(left, right):
            if equal(left, right):
                return True
            if isinstance(left, ast.BinOp) and isinstance(right, ast.BinOp):
                return (
                    type(left.op) is type(right.op)
                    and isinstance(
                        left.op, (ast.BitOr, ast.BitAnd, ast.BitXor, ast.Sub)
                    )
                    and literal_edit(left.left, right.left)
                    and literal_edit(left.right, right.right)
                )
            if isinstance(left, ast.Set) and isinstance(right, ast.Set):

                def other_entries(node):
                    return [
                        v
                        for v in node.elts
                        if not (
                            isinstance(v, ast.Attribute)
                            and isinstance(v.value, ast.Name)
                            and v.value.id == "TokenType"
                        )
                    ]

                # Keep all pre-existing starred expressions and their evaluation order.
                return equal(other_entries(left), other_entries(right))
            return False

        if not literal_edit(x.value, y.value):
            return None
        fields.add(t.id)
    return fields


def refine(sel, changes, selected, apply, equal, checkout, remaining):
    if (
        len(changes) != 1
        or changes[0]["old"] != changes[0]["new"]
        or remaining < 1.0
        or sel.unknown
        or not selected
    ):
        return selected
    path = changes[0]["old"]
    if path == SOURCE:
        name, owners = "Parser", None
    elif path.startswith("sqlglot/parsers/"):
        import parser_cells

        registry = parser_cells.data()
        if path not in registry["permitted"]:
            return selected
        name = registry["permitted"][path][0]
        owners = set(registry["descendants"][path])
    else:
        return selected
    if not any(
        line.startswith(("+", "-")) and "TokenType." in line
        for hunk in changes[0]["hunks"]
        for line in hunk["lines"]
    ):
        return selected
    fields = declarations(sel, changes[0], apply, equal, name)
    if not fields:
        return selected
    d = profile()
    if d["pin"] != sel.mapping["commit"] or not fields <= d["closures"].keys():
        return selected
    active = set().union(*(d["closures"][f] for f in fields))
    with tempfile.TemporaryDirectory(prefix="br-grammar-") as tmp:
        root, out = Path(tmp) / "repo", Path(tmp) / "effective"
        checkout(root, changes, sel)
        forkprobe.run(
            (root, sorted(active), out),
            {"PYTHONPATH": str(root)},
            probe,
            min(3.5, remaining),
        )
        if not out.exists():
            return selected
        after = json.loads(out.read_bytes())
    if after.keys() != d["values"].keys():
        return selected
    masks = d["groups"]
    import parser_cells
    spec = parser_cells.data()
    broad = int(masks[d["protected"]], 16)
    if d["tests"] != spec["names"]:
        return selected
    original = sum(1 << i for i in spec["reflective"])
    revised = sum(1 << i for i in reflective_use.indices(sel, spec, active))
    bits = (broad & ~original) | revised
    for owner, row in after.items():
        before = d["values"][owner]
        if set(row) != before.keys() & active:
            return selected
        if owners is not None and owner not in owners:
            # A sibling parser may construct its settings from the edited class.
            # Keep the incoming plan if a changed value escaped the traced family.
            if any(
                v != (d["sets"][before[f]] if before[f] is not None else None)
                for f, v in row.items()
            ):
                return selected
            continue
        for field, v in row.items():
            idx = before[field]
            if idx is None or v is None:
                if idx is not None or v is not None:
                    return selected
                continue
            old = d["sets"][idx]
            for k in [owner, "?"]:
                slot = d["opaque"].get(k, {}).get(field)
                if slot is not None:
                    bits |= int(masks[slot], 16)
            if v == old:
                continue
            if not old or not v:
                return selected
            for token in set(v) ^ set(old):
                slot = d["membership"].get(str(idx), {}).get(token)
                if slot is not None:
                    bits |= int(masks[slot], 16)
    required = set(sel.confirmed_failures)
    if path in sel.files:
        required.update(sel.sets[sel.files[path]])
    external = sel.dispatch_map()
    required.update(external.external)
    required.update(external.external_by_source.get(path, ()))
    names = d["tests"]
    while bits:
        bit = bits & -bits
        ident = names[bit.bit_length() - 1]
        if ident in sel.by_name:
            required.add(sel.by_name[ident])
        bits ^= bit
    # The base line map can miss inherited readers. Include observed token
    # consumers even when the earlier approximation did not select them.
    return required if owners is None else selected & required
