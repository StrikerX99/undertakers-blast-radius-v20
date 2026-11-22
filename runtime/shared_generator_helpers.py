"""Compose first-use dependencies for a shared generator helper and a dialect callback."""

import ast
import copy
import marshal
import tempfile
from pathlib import Path

import dispatch_helpers
import generator_cells
import generator_contexts
import import_cone
import reflective_use
import structural_dependencies
import symbol_impact

SHARED_FILE = "sqlglot/dialects/dialect.py"


def generator_class(node):
    return (
        isinstance(node, ast.ClassDef)
        and node.name.endswith("Generator")
        and not node.decorator_list
        and not node.keywords
    )


def transform_assignment(node):
    return (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "TRANSFORMS"
    )


def eager_table_read(statements):
    for stmt in statements:
        early = (
            [stmt.args, *stmt.decorator_list]
            if isinstance(stmt, ast.FunctionDef)
            else [stmt]
        )
        if any(
            isinstance(n, (ast.Name, ast.Attribute))
            and getattr(n, "id", getattr(n, "attr", "")) == "TRANSFORMS"
            for expr in early
            for n in ast.walk(expr)
        ):
            return True
    return False


def scalar(node, names):
    """Only compare plain values. No calls, indexing, attributes or assignments."""
    if isinstance(node, ast.Constant):
        return type(node.value) in (int, str, bool, type(None))
    if isinstance(node, ast.Name):
        return node.id in names
    if isinstance(node, ast.IfExp):
        return all(scalar(x, names) for x in (node.test, node.body, node.orelse))
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        return isinstance(node.ops[0], (ast.Eq, ast.NotEq, ast.Is, ast.IsNot)) and all(
            scalar(x, names) for x in (node.left, *node.comparators)
        )
    return False


def local_value_edit(left, right, equal, names):
    """Keep every operation and its evaluation order except a scalar argument."""
    if equal(left, right):
        return True
    if scalar(left, names) and scalar(right, names):
        return True
    if not isinstance(left, ast.Call) or not isinstance(right, ast.Call):
        return False
    return (
        equal(left.func, right.func)
        and len(left.args) == len(right.args)
        and [k.arg for k in left.keywords] == [k.arg for k in right.keywords]
        and all(
            local_value_edit(a, b, equal, names) for a, b in zip(left.args, right.args)
        )
        and all(
            local_value_edit(a.value, b.value, equal, names)
            for a, b in zip(left.keywords, right.keywords)
        )
    )


def unchanged_prefix(left, right, equal):
    if not left.body or not right.body or not equal(left.body[:-1], right.body[:-1]):
        return False
    a, b = left.body[-1], right.body[-1]
    # A primitive annotated parameter or its simple local string fallback may
    # dispatch to existing generator behavior. Do not permit Expr truth/equality
    # hooks or any new mutation as a consequence of comparing an operand.
    names = {
        arg.arg
        for arg in (*left.args.args, *left.args.kwonlyargs)
        if isinstance(arg.annotation, ast.Name)
        and arg.annotation.id in ("str", "int", "bool")
    }
    for arg in (*left.args.args, *left.args.kwonlyargs):
        annotation = arg.annotation
        if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
            if any(
                isinstance(x, ast.Name) and x.id in ("str", "int", "bool")
                for x in (annotation.left, annotation.right)
            ) and any(
                isinstance(x, ast.Constant) and x.value is None
                for x in (annotation.left, annotation.right)
            ):
                names.add(arg.arg)
    if not (
        isinstance(a, ast.Return)
        and isinstance(b, ast.Return)
        and isinstance(a.value, ast.Call)
    ):
        return False
    target = a.value.func
    if not (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
        and target.attr in ("sql", "binary", "func")
    ):
        return False
    return local_value_edit(a.value, b.value, equal, names)


def deferred_calls(sel, name):
    """Find every use of the original binding as a named cell, lambda or local call.

    Calls inside ordinary methods are also covered by the full original helper
    body footprint. Explicit call-site readers cover entry and binding errors.
    """
    grouped = {}
    for path, line, scope, kind in symbol_impact.index().get(name, ()):
        if not path.startswith("sqlglot/generators/") or kind not in (
            "name",
            "from:sqlglot.dialects.dialect",
        ):
            return None
        grouped.setdefault(path, []).append((line, scope, kind))
    if not grouped:
        return None
    slots = []
    readers = set()
    for path, refs in grouped.items():
        _, tree = sel.source(path)
        parents = {
            child: node
            for node in ast.walk(tree)
            for child in ast.iter_child_nodes(node)
        }
        loads = {
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load)
        }
        imported = [
            n
            for n in tree.body
            if isinstance(n, ast.ImportFrom)
            and not n.level
            and n.module == "sqlglot.dialects.dialect"
            and any(a.asname is None and a.name == name for a in n.names)
        ]
        if len(imported) != 1 or any(
            isinstance(n, (ast.ClassDef, ast.FunctionDef))
            and n.name == name
            or isinstance(n, ast.Name)
            and n.id == name
            and not isinstance(n.ctx, ast.Load)
            or isinstance(n, ast.alias)
            and (n.asname == name or n.name == "*")
            for n in ast.walk(tree)
        ):
            return None
        resolved = set()
        for owner in (n for n in tree.body if generator_class(n)):
            for at, stmt in enumerate(owner.body):
                if not transform_assignment(stmt):
                    continue
                cells = dispatch_helpers.literal_mapping(stmt.value)
                if cells is None:
                    continue
                for key, value in zip(cells.keys, cells.values):
                    direct = {value} & loads
                    if (
                        isinstance(value, ast.Lambda)
                        and not value.args.defaults
                        and not any(value.args.kw_defaults)
                    ):
                        direct.update(
                            n
                            for n in ast.walk(value.body)
                            if n in loads
                            and isinstance(parents[n], ast.Call)
                            and parents[n].func is n
                        )
                    if not direct:
                        continue
                    if not (
                        isinstance(key, ast.Attribute)
                        and isinstance(key.value, ast.Name)
                        and key.value.id == "exp"
                    ) or eager_table_read(owner.body[at + 1 :]):
                        return None
                    slots.append((path, owner.name, key.attr))
                    resolved.update(direct)
        for node in loads - resolved:
            call = parents[node]
            if not isinstance(call, ast.Call) or call.func is not node:
                return None
            current = call
            while current in parents:
                parent = parents[current]
                if isinstance(parent, ast.FunctionDef) and current in parent.body:
                    break
                if isinstance(parent, (ast.Lambda, ast.FunctionDef, ast.ClassDef)):
                    return None
                current = parent
            else:
                return None
            if call.lineno in sel.mapping.get("import_lines", {}).get(path, ()):
                return None
            readers.update(sel.line_tests(path, [call.lineno]))
        lines = {(n.lineno, "name") for n in loads} | {
            (imported[0].lineno, "from:sqlglot.dialects.dialect")
        }
        if {(ln, k) for ln, scope, k in refs} != lines:
            return None
    return slots, readers


def projections(sel, change, apply, equal):
    old, before = sel.source(SHARED_FILE)
    text = apply(old, change)[0]
    if text != change.get("_compiled_source"):
        return None
    after = ast.parse(text)
    if len(before.body) != len(after.body):
        return None
    slots = []
    body_readers = set()
    helper_names = set()
    for left, right in zip(before.body, after.body):
        if equal(left, right):
            continue
        if not (
            isinstance(left, ast.FunctionDef)
            and isinstance(right, ast.FunctionDef)
            and generator_contexts.header(left, right, equal)
            and left.name.endswith("_sql")
            and isinstance(left.args.args[0].annotation, ast.Name)
            and left.args.args[0].annotation.id == "Generator"
            and ast.get_docstring(left, clean=False)
            == ast.get_docstring(right, clean=False)
        ):
            return None
        if not all(
            isinstance(d, ast.Call)
            and isinstance(d.func, ast.Name)
            and d.func.id == "unsupported_args"
            and not d.keywords
            and all(
                isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                for arg in d.args
            )
            for d in left.decorator_list
        ):
            return None
        special = unchanged_prefix(left, right, equal)
        if not special:
            for fn in (left, right):
                if any(
                    isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr
                    in {
                        "set",
                        "pop",
                        "append",
                        "remove",
                        "update",
                        "extend",
                        "clear",
                        "__setitem__",
                    }
                    for n in ast.walk(fn)
                ):
                    return None
                plain = copy.copy(fn)
                plain.decorator_list = []
                if not import_cone.local_function(plain):
                    return None
        body_lines = range(left.body[0].lineno, left.end_lineno + 1)
        if not set(body_lines).isdisjoint(
            sel.mapping.get("import_lines", {}).get(SHARED_FILE, ())
        ):
            return None
        body_readers.update(sel.line_tests(SHARED_FILE, body_lines))
        proof = deferred_calls(sel, left.name)
        if proof is None:
            return None
        nested, readers = proof
        slots.extend(nested)
        body_readers.update(readers)
        helper_names.add(left.name)
    return (slots, body_readers, helper_names) if helper_names else None


def frame(node, keys):
    """Mask only the selected callback values in a fixed, passive dictionary comprehension."""
    copied = copy.deepcopy(node)
    literal = dispatch_helpers.literal_mapping(copied)
    literal.values = [
        ast.Constant(None) if isinstance(k, ast.Attribute) and k.attr in keys else value
        for k, value in zip(literal.keys, literal.values)
    ]
    return copied


def same_deferred_call(a, b, equal, helper_names):
    """Argument-only changes in a deferred callback cannot change class creation."""
    if not (
        isinstance(a, ast.Lambda)
        and isinstance(b, ast.Lambda)
        and equal(a.args, b.args)
        and not a.args.defaults
        and not any(a.args.kw_defaults)
        and isinstance(a.body, ast.Call)
        and isinstance(b.body, ast.Call)
    ):
        return False
    target = a.body.func
    if not (isinstance(target, ast.Name) and target.id in helper_names):
        return False
    return local_value_edit(a.body, b.body, equal, set())


def changed_keys(a, b, equal, bound, helper_names):
    keys = generator_cells.differences(a, b, equal, bound)
    if keys:
        return keys
    if "exp" not in bound:
        return None
    left, old_order = generator_cells.entries(a.value)
    right, new_order = generator_cells.entries(b.value)
    if old_order != new_order:
        return None
    delta = {k for k in old_order if not equal(left[k], right[k])}
    if not delta or any(
        k.startswith("!")
        or not same_deferred_call(left[k], right[k], equal, helper_names)
        for k in delta
    ):
        return None
    return delta


def table_changes(sel, change, apply, equal, helper_names):
    path = change["old"]
    if (
        path != change["new"]
        or not path.startswith("sqlglot/generators/")
        or not path.endswith(".py")
    ):
        return None
    raw, old = sel.source(path)
    text = apply(raw, change)[0]
    if change.get("_compiled_source") != text:
        return None
    new = ast.parse(text)
    if len(old.body) != len(new.body):
        return None
    found = []
    for left, right in zip(old.body, new.body):
        if equal(left, right):
            continue
        if not (
            generator_class(left)
            and generator_class(right)
            and generator_contexts.header(left, right, equal)
            and len(left.body) == len(right.body)
        ):
            return None
        for idx, (a, b) in enumerate(zip(left.body, right.body)):
            if equal(a, b):
                continue
            if not (
                transform_assignment(a)
                and transform_assignment(b)
                and equal(a.targets, b.targets)
            ):
                return None
            if eager_table_read(left.body[idx + 1 :]):
                return None
            x = dispatch_helpers.literal_mapping(a.value)
            y = dispatch_helpers.literal_mapping(b.value)
            if x is None or y is None:
                return None
            p, q = copy.copy(a), copy.copy(b)
            p.value = x
            q.value = y
            keys = changed_keys(
                p, q, equal, generator_cells.factory_bindings(old), helper_names
            )
            if not keys or not equal(frame(a.value, keys), frame(b.value, keys)):
                return None
            found.extend((path, left.name, k) for k in keys)
    return found or None


def refine(sel, changes, selected, apply, equal, checkout, remaining):
    if not changes or len(changes) > 4 or remaining < 4 or not selected:
        return selected
    shared = next((c for c in changes if c["old"] == c["new"] == SHARED_FILE), None)
    others = [c for c in changes if c is not shared]
    if not all(c["old"].startswith("sqlglot/generators/") for c in others):
        return selected
    if not shared and (len(others) < 2 or len(selected) < 120):
        return selected
    slots = []
    body_readers = set()
    helper_names = set()
    if shared:
        helper = projections(sel, shared, apply, equal)
        if not helper:
            return selected
        slots, body_readers, helper_names = helper
    for other in others:
        edits = table_changes(sel, other, apply, equal, helper_names)
        if not edits:
            return selected
        slots.extend(edits)
    if not slots:
        return selected
    profile = generator_contexts.data_dispatch()
    keys = {key for _, _, key in slots}
    if profile["commit"] != sel.mapping["commit"] or keys & set(
        profile["structural"]["dialect_init_keys"]
    ):
        return selected
    if any(
        not profile["sources"].get(p)
        or profile["sources"][p] != sel.mapping["source_hashes"].get(p)
        for p, _, _ in slots
    ):
        return selected
    custom = import_cone.receiver_users(sel, [])
    if custom is None:
        return selected
    keep = set(sel.confirmed_failures) | sel.unknown | custom | body_readers
    reflect = generator_contexts.data("parser-cells.json")
    keep.update(
        sel.by_name[reflect["names"][i]]
        for i in reflective_use.indices(sel, reflect)
        if reflect["names"][i] in sel.by_name
    )
    paths = {SHARED_FILE} | {p for p, _, _ in slots}
    ext = sel.dispatch_map()
    keep.update(ext.external)
    for path in paths:
        sel.load_shard(path)
        if path in sel.files:
            keep.update(sel.sets[sel.files[path]])
        keep.update(ext.external_by_source.get(path, ()))
    for path, owner, key in slots:
        keep.update(
            structural_dependencies.table_observers(
                sel, profile, path, path[:-3].replace("/", ".") + ":" + owner, {key}
            )
        )
    if selected <= keep:
        return selected
    old = generator_cells.baseline()
    with tempfile.TemporaryDirectory(prefix="generator-shared-") as tmp:
        root = Path(tmp) / "src"
        out = Path(tmp) / "state"
        checkout(root, changes, sel)
        generator_cells.forkprobe.run(
            (root, sorted(keys), out), {}, generator_cells.probe, min(3, remaining)
        )
        if not out.exists():
            return selected
        fresh = marshal.loads(out.read_bytes())
    if fresh.keys() != old.keys():
        return selected
    required = {
        (path[:-3].replace("/", ".") + ":" + owner, k) for path, owner, k in slots
    }
    if any(owner not in fresh or k not in fresh[owner] for owner, k in required):
        return selected
    for owner, values in fresh.items():
        for key, current in values.items():
            prior = old[owner].get(key, [False, "missing", "missing"])
            if current != prior or None in prior or None in current:
                # Nonlocal copied or regenerated registry entries use every original
                # source's cold-reader allowance as well as the effective owner.
                for path in {p for p, _, name in slots if name == key}:
                    keep.update(
                        structural_dependencies.table_observers(
                            sel, profile, path, owner, {key}
                        )
                    )
    proposed = (keep & selected) | sel.confirmed_failures
    if proposed != selected:
        sel.registry_candidates = frozenset(proposed)
    return proposed
