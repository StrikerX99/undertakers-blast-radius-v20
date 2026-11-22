"""Record pristine callable identities at image build time on the pinned interpreter."""

import inspect
import marshal
import sys
import types
from functools import lru_cache
from pathlib import Path

FILE = Path(__file__).with_name("pristine-dispatch.bin")


def dotted(cls):
    return cls.__module__ + "." + cls.__qualname__


def identity(value):
    if callable(value) and hasattr(value, "__qualname__"):
        return [value.__module__, value.__qualname__]
    return None


def concrete_receivers(root):
    """Resolve enum dialects and the concrete classes actually exercised by the test catalog."""
    import importlib
    import json

    from sqlglot import Dialect
    from sqlglot.dialects import Dialects
    from sqlglot.generator import Generator
    from sqlglot.parser import Parser

    classes = set()
    for tag in Dialects:
        dialect = type(Dialect.get_or_raise(tag))
        classes.update((dialect, dialect.parser_class, dialect.generator_class))

    observed = json.loads(Path(__file__).with_name("receiver-calls.json").read_bytes())[
        "classes"
    ]
    for name in observed:
        if not name.startswith("sqlglot."):
            continue
        parts = name.split(".")
        for n in range(len(parts) - 1, 0, -1):
            at = root.joinpath(*parts[:n])
            if at.with_suffix(".py").is_file() or (at / "__init__.py").is_file():
                obj = importlib.import_module(".".join(parts[:n]))
                for part in parts[n:]:
                    obj = vars(obj)[part]
                if not isinstance(obj, type) or dotted(obj) != name:
                    raise ValueError("Unresolved traced receiver " + name)
                if issubclass(obj, (Dialect, Parser, Generator)):
                    classes.add(obj)
                break
        else:
            raise ValueError("Missing receiver module " + name)
    return classes


def build(root):
    sys.path.insert(0, str(root))
    from sqlglot import exp
    from sqlglot.generator import _build_dispatch
    import sqlglot

    assert Path(sqlglot.__file__).resolve() == (root / "sqlglot/__init__.py").resolve()
    classes = concrete_receivers(root)
    methods = {}
    parents = {}
    tables = {}
    aliases = {k: dotted(v) for k, v in vars(exp).items() if isinstance(v, type)}
    for cls in classes:
        name = dotted(cls)
        parents[name] = {dotted(p) for p in cls.__mro__}
        attributes = set().union(*(vars(base) for base in cls.__mro__))
        # Invoke no arbitrary descriptor while building the cache.
        known = {}
        for key in attributes:
            val = inspect.getattr_static(cls, key)
            if isinstance(val, types.FunctionType):
                known[key] = identity(getattr(cls, key))
        meta = type(cls)
        ordinary = meta.__getattribute__ is type.__getattribute__ and not hasattr(
            meta, "__getattr__"
        )
        meta_names = set().union(*(vars(base) for base in meta.__mro__))
        methods[name] = known, attributes | meta_names, ordinary
        if hasattr(cls, "TRANSFORMS") and any(
            p.__name__ == "Generator" for p in cls.__mro__
        ):
            dispatch = _build_dispatch(cls)
            tables[name] = {
                dotted(k): identity(v)
                for k, v in dispatch.items()
                if isinstance(k, type)
            }
    FILE.write_bytes(marshal.dumps((parents, methods, tables, aliases)))


@lru_cache(maxsize=1)
def read():
    return marshal.loads(FILE.read_bytes())


def lookup(root, requests):
    if Path(root) != Path("/opt/sqlglot"):
        return None
    parents, methods, tables, aliases = read()
    result = []
    for path, owner, kind, member in requests:
        host = path[:-3].replace("/", ".") + "." + owner
        receivers = [name for name, bases in parents.items() if host in bases]
        if not receivers:
            return None
        if kind == "table":
            if member not in aliases:
                return None
            result.append(
                {name: values.get(aliases[member]) for name, values in tables.items()}
            )
        elif kind == "method":
            rows = {}
            for cls in receivers:
                values, names, ordinary = methods[cls]
                if member in values and ordinary:
                    rows[cls] = values[member]
                elif member not in names and ordinary:
                    rows[cls] = None
                else:
                    return None
            result.append(rows)
        else:
            return None
    return result


if __name__ == "__main__":
    build(Path(sys.argv[1]))
