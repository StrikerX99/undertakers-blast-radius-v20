"""Keep actual consumers of a guarded test-side introspection operation."""

import ast
import json
import sys
from functools import lru_cache
from pathlib import Path
from collections import defaultdict


# These are the original baseline scanners' sensitive attribute and call families.
SENSITIVE_ATTRIBUTES = {
    "_build_dispatch",
    "_DISPATCH_CACHE",
    "_dispatch",
    "__dict__",
    "__subclasses__",
    "__code__",
    "__func__",
    "_classes",
    "classes",
}
SENSITIVE_CALLS = {
    "getattr",
    "hasattr",
    "vars",
    "dir",
    "getmembers",
    "getattr_static",
    "getsource",
    "getsourcefile",
    "signature",
    "getfullargspec",
}
REGISTRY_TEXT = (
    "._classes",
    ".classes",
    "__subclasses__",
    "vars(Dialect)",
    "dir(Dialect)",
)


def operations(node):
    if isinstance(node, ast.Attribute):
        return node.attr in SENSITIVE_ATTRIBUTES
    if isinstance(node, ast.Call):
        fn = node.func
        return getattr(fn, "id", getattr(fn, "attr", "")) in SENSITIVE_CALLS
    # Includes embedded Python such as a test that spawns a fresh interpreter.
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and any(key in node.value for key in REGISTRY_TEXT)
    )


def fixed_method_names(node, ancestors, method):
    """Resolve a direct getattr on the string slot of one literal for-loop."""
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) == 2
        and not node.keywords
        and isinstance(node.args[1], ast.Name)
    ):
        return None
    name = node.args[1].id
    stores = [
        n
        for n in ast.walk(method)
        if isinstance(n, ast.Name)
        and n.id == name
        and isinstance(n.ctx, (ast.Store, ast.Del))
    ]
    if len(stores) != 1:
        return None
    for at in reversed(range(len(ancestors))):
        loop = ancestors[at]
        if not isinstance(loop, ast.For) or not isinstance(
            loop.iter, (ast.Tuple, ast.List)
        ):
            continue
        if (
            not isinstance(loop.target, (ast.Tuple, ast.List))
            or stores[0] not in loop.target.elts
        ):
            continue
        child = ancestors[at + 1] if at + 1 < len(ancestors) else node
        if child not in loop.body:
            continue
        index = loop.target.elts.index(stores[0])
        found = set()
        for row in loop.iter.elts:
            if not isinstance(row, (ast.Tuple, ast.List)) or len(row.elts) != len(
                loop.target.elts
            ):
                return None
            entry = row.elts[index]
            if not isinstance(entry, ast.Constant) or not isinstance(entry.value, str):
                return None
            if (
                not entry.value.isidentifier()
                or entry.value.startswith("_")
                or not entry.value.islower()
            ):
                return None
            found.add(entry.value)
        return found or None
    return None


def sites(tree, by_method):
    """Return owner, line span and optional finite lookup keys; opaque scopes retain the file."""
    hits = []

    def visit(node, parents, method):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            direct = (
                len(parents) == 2
                and isinstance(parents[0], ast.Module)
                and isinstance(parents[1], ast.ClassDef)
                and node in parents[1].body
                and node.name in by_method
            )
            # Decorators, defaults, annotations and type parameters can be evaluated
            # while importing/collecting the entire module.
            head = [n for n in ast.iter_child_nodes(node) if n not in node.body]
            for child in head:
                visit(child, (*parents, node), None)
            for child in node.body:
                visit(child, (*parents, node), node if direct else None)
            return
        if isinstance(node, (ast.ClassDef, ast.Lambda)):
            method = None
        if operations(node):
            hits.append(
                (
                    method.name if method else None,
                    node.lineno,
                    node.end_lineno,
                    fixed_method_names(node, parents, method) if method else None,
                )
            )
        for child in ast.iter_child_nodes(node):
            visit(child, (*parents, node), method)

    visit(tree, (), None)
    return hits


def build(root, spec_path, destination):
    spec = json.loads(spec_path.read_bytes())
    by_file = defaultdict(set)
    for i in spec["reflective"]:
        file, _, fn = spec["names"][i].split("::")
        by_file[file].add(fn)
    result = {}
    for file, names in by_file.items():
        tree = ast.parse((root / file).read_bytes())
        found = sites(tree, names)
        if found and all(fn is not None for fn, *_ in found):
            result[file] = [
                [fn, start, end, sorted(keys) if keys else None]
                for fn, start, end, keys in found
            ]
    destination.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")))


@lru_cache(maxsize=1)
def catalog():
    return json.loads(Path(__file__).with_name("reflective-sites.json").read_bytes())


def indices(sel, spec, fields=()):
    """Shrink only the original static reflection allowance. Ordinary readers stay intact."""
    broad = frozenset(spec["reflective"])
    names = spec["names"]
    if not hasattr(sel, "_reflective_site_cache"):
        sel._reflective_site_cache = {}
    key = (frozenset(fields), broad, tuple(names))
    if key in sel._reflective_site_cache:
        return sel._reflective_site_cache[key]
    by_file = defaultdict(list)
    for i in broad:
        by_file[names[i].split("::")[0]].append(i)
    pos = {name: i for i, name in enumerate(names)}
    result = set()
    for path, group in by_file.items():
        local = catalog().get(path)
        if local is None:
            result.update(group)
            continue
        methods = defaultdict(set)
        for i in group:
            methods[names[i].rsplit("::", 1)[-1]].add(i)
        sel.load_shard(path)
        for owner, first, last, keys in local:
            if fields and keys and key[0].isdisjoint(keys):
                continue
            result.update(methods[owner])
            for n in sel.line_tests(path, range(first, last + 1)):
                index = pos.get(sel.by_number[n]["id"])
                if index is not None:
                    result.add(index)
    sel._reflective_site_cache[key] = result
    return result


if __name__ == "__main__":
    build(*(Path(arg) for arg in sys.argv[1:]))
