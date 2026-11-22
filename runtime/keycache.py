"""Serialize the pinned registry once, retaining exact callback encodings and tokenizer guards."""

import importlib
import json
import marshal
import os
import sys
from functools import lru_cache
from pathlib import Path

CACHE = Path(__file__).with_name("parser-registry.bin")


def build(root):
    import shared_tables

    os.chdir(root)
    sys.path.insert(0, str(root))
    from sqlglot import Parser, Tokenizer

    profiles = json.loads(Path(__file__).with_name("registry-reads.json").read_bytes())
    for package in ("dialects", "parsers"):
        for file in sorted((root / "sqlglot" / package).glob("*.py")):
            if file.stem != "__init__":
                importlib.import_module(f"sqlglot.{package}.{file.stem}")
    for cls in profiles["rows"]:
        module = cls.partition(":")[0]
        if module.startswith("sqlglot."):
            importlib.import_module(module)

    values = [None]
    memo = {}
    fields = {}
    for field, base in (("KEYWORDS", Tokenizer), ("FUNCTIONS", Parser)):
        classes = [base]
        seen = set()
        rows = {}
        for cls in classes:
            if cls in seen:
                continue
            seen.add(cls)
            classes.extend(cls.__subclasses__())
            cells = getattr(cls, field)
            if type(cells) is not dict:
                rows = None
                break
            indices = {}
            for key, value in cells.items():
                if type(key) is not str:
                    continue
                ident = id(value)
                if ident not in memo:
                    try:
                        encoded = shared_tables.encode(value)
                        index = len(values)
                        values.append(encoded)
                    except (ValueError, TypeError, RecursionError):
                        index = -1
                    memo[ident] = index
                indices[key] = memo[ident]
            guard = (
                (cls.HINT_START, tuple(cls.SINGLE_TOKENS))
                if field == "KEYWORDS"
                else None
            )
            rows.setdefault(cls.__module__ + ":" + cls.__qualname__, []).append(
                (guard, indices)
            )
        fields[field] = rows
    CACHE.write_bytes(marshal.dumps((sorted(profiles["rows"]), fields, values)))


@lru_cache(maxsize=1)
def read():
    return marshal.loads(CACHE.read_bytes())


def original(root, field, keys, classes):
    if Path(root) != Path("/opt/sqlglot"):
        return None
    expected, fields, values = read()
    if expected != sorted(classes):
        return None
    shape = fields.get(field)
    if shape is None:
        return None
    answer = {}
    from registry_keys import AMBIGUOUS

    for cls, variants in shape.items():
        row = {}
        for guard, cells in variants:
            for key in keys:
                if guard is not None and (
                    key == guard[0] or any(token in key for token in guard[1])
                ):
                    return None
                index = cells.get(key, 0)
                if index < 0:
                    return None
                value = [key in cells, values[index]]
                if key not in row:
                    row[key] = value
                elif row[key] != value:
                    row[key] = AMBIGUOUS
        # Multiple runtime types can share the same generated qualified name.
        # Their enumeration can vary with weak-reference collection. Keep readers
        # of every requested key on that shared label in the potential impact set.
        if len(variants) > 1:
            row = {key: AMBIGUOUS for key in row}
        answer[cls] = row
    return answer


if __name__ == "__main__":
    build(Path(sys.argv[1]))
