"""Build or verify the registry of class objects exported by the pinned expressions modules."""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
from pathlib import Path
import re

PIN = "eb11aec4be8cb2122afe713a4816662c3f0fb2fa"


def hash_values(values):
    data = json.dumps(values, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(data).hexdigest()


def class_label(cls):
    return cls.__module__ + "." + cls.__qualname__


def registry_digests(exp, exclude):
    """Check original registry order as well as the identity of all existing class mappings."""
    functions = [
        (key, class_label(cls))
        for key, cls in exp.FUNCTION_BY_NAME.items()
        if cls is not exclude
    ]
    classes = [
        (key, class_label(cls))
        for key, cls in exp.EXPR_CLASSES.items()
        if cls is not exclude
    ]
    inventory = [class_label(cls) for cls in exp.ALL_FUNCTIONS if cls is not exclude]
    return [hash_values(values) for values in (functions, classes, inventory)]


def build(project, target):
    sources = {}
    parents = {}
    reserved = set()
    for area in ("sqlglot", "tests"):
        for path in (project / area).rglob("*.py"):
            text = path.read_bytes()
            reserved.update(
                name.decode("ascii")
                for name in re.findall(rb"(?<![\w])[A-Z][A-Za-z0-9]+(?![\w])", text)
            )
            name = path.relative_to(project).as_posix()
            module = name[:-3].replace("/", ".").removesuffix(".__init__")
            sources[module] = path
            tree = ast.parse(text)
            for stmt in ast.walk(tree):
                if isinstance(stmt, ast.ImportFrom) and any(
                    alias.name == "*" for alias in stmt.names
                ):
                    if stmt.level:
                        base = (
                            module
                            if path.name == "__init__.py"
                            else module.rpartition(".")[0]
                        )
                        try:
                            origin = importlib.util.resolve_name(
                                "." * stmt.level + (stmt.module or ""), base
                            )
                        except ImportError:
                            continue
                    else:
                        origin = stmt.module
                    parents.setdefault(origin, set()).add(module)
    from sqlglot import exp
    from sqlglot.dialects.dialect import Dialect, Dialects

    for dialect in Dialects:
        Dialect.get_or_raise(dialect.value)
    locations = {}
    for p in (project / "sqlglot/expressions").glob("*.py"):
        if p.name in ("__init__.py", "core.py"):
            continue
        module = "sqlglot.expressions." + p.stem
        names = {module}
        todo = [module]
        while todo:
            for reader in parents.get(todo.pop(), ()):
                if reader not in names:
                    names.add(reader)
                    todo.append(reader)
        # Limit admission to whole-module star reexports within SQLGlot.
        if "sqlglot.expressions" not in names or any(
            not n.startswith("sqlglot.") for n in names
        ):
            continue
        for name in names:
            obj = importlib.import_module(name)
            if any(k in vars(obj) for k in ("__all__", "__getattr__", "__dir__")):
                break
            reserved.update(vars(obj).keys())
        else:
            locations[module] = sorted(names)
    return dict(
        pin=PIN,
        exported=locations,
        forbidden=sorted(reserved),
        registries=registry_digests(exp, None),
        registry_files={
            str(sources[m].relative_to(project)): hashlib.sha256(
                sources[m].read_bytes()
            ).hexdigest()
            for m in (
                "sqlglot.expressions.core",
                "sqlglot.expressions",
                "sqlglot.generator",
                "sqlglot.parser",
                "sqlglot.expressions.builders",
                "sqlglot.helper",
                "sqlglot.generators.python",
                "sqlglot.typing",
            )
        },
    )


if __name__ == "__main__":
    import sys

    project = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(project))
    target = Path(sys.argv[2])
    result = build(project, target)
    target.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")))
    print(
        "Pinned",
        len(result["exported"]),
        "expression module namespaces,",
        len(result["forbidden"]),
        "existing names",
    )
