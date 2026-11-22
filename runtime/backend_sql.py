"""Pristine SQL generation inputs used after optimization by the native TPC-DS worker."""

import hashlib
import importlib
import json
import marshal
import os
from pathlib import Path
import pickle
import sys
import zlib

DATA = Path(__file__).with_name("backend-sql.bin")


def write_map(root, dest):
    os.chdir(root)
    sys.path.insert(0, str(root))
    from sqlglot import parse_one
    import sqlglot.executor as engine
    from sqlglot.dialects.dialect import Dialect
    from sqlglot.generator import Generator
    from tests.helpers import load_sql_fixture_pairs
    from query_case import source_environment, table_digest

    cfg = json.loads(Path(__file__).with_name("executor-queries.json").read_bytes())
    scope = source_environment(root, cfg["driver_sha"])
    optimize = engine.optimize
    factory = Dialect.generator
    render = Generator.generate
    active = False
    nesting = 0
    serials = {}
    streams = []
    payloads = {}

    def boundary(*aa, **kw):
        nonlocal active
        value = optimize(*aa, **kw)
        active = True
        return value

    def instrument_factory(dialect, **options):
        gen = factory(dialect, **options)
        if active:
            cls = type(dialect)
            if type(gen).generate is not record:
                raise ValueError("Unexpected native generator entry")
            if vars(dialect).get("settings") or any(
                type(v) not in (str, int, bool, type(None)) for v in options.values()
            ):
                raise ValueError("Unrecognized native generator settings")
            serials[id(gen)] = (gen, len(streams))
            streams.append((i, cls.__module__, cls.__qualname__, options, []))
        return gen

    def record(gen, expression, copy=True):
        nonlocal nesting
        if not active or nesting:
            return render(gen, expression, copy=copy)
        entry = serials.get(id(gen))
        if entry is None or entry[0] is not gen or copy is not True:
            raise ValueError("Generator receiver or mutation mode unknown")
        # generate(copy=True) starts with an independent copy of this expression.
        # Retain the parent's context as well so a patch can inspect the input before that copy.
        anchor = expression
        path = []
        while anchor.parent is not None:
            path.append((anchor.arg_key, anchor.index))
            anchor = anchor.parent
        raw = pickle.dumps(anchor, protocol=4)
        key = hashlib.sha256(raw).digest()
        payloads.setdefault(key, raw)
        nesting += 1
        try:
            sql = render(gen, expression, copy=copy)
        finally:
            nesting -= 1
        streams[entry[1]][4].append(
            (key, tuple(reversed(path)), hashlib.sha256(sql.encode()).digest())
        )
        return sql

    Dialect.generator = instrument_factory
    Generator.generate = record
    engine.optimize = boundary
    try:
        for i, (meta, sql, _) in enumerate(
            load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql")
        ):
            active = False
            if not meta.get("execute"):
                continue
            expected = cfg["queries"][str(i)]
            table = scope["mp_execute"](
                pickle.loads(pickle.dumps(parse_one(sql))), meta
            )
            active = False
            if table_digest(table) != expected["result"]:
                raise ValueError("Native result differs from pinned record")
            serials.clear()
    finally:
        Dialect.generator = factory
        Generator.generate = render
        engine.optimize = optimize
    record_data = (2, streams, payloads)
    if replay(record_data):
        raise ValueError("Baseline backend transcript did not reproduce")
    dest.write_bytes(zlib.compress(marshal.dumps(record_data), 5))
    print(
        "Verified",
        len(streams),
        "backend generator instances and",
        sum(len(s[-1]) for s in streams),
        "SQL calls",
        flush=True,
    )


def replay(record_data=None):
    from functools import lru_cache

    if record_data is None:
        record_data = marshal.loads(zlib.decompress(DATA.read_bytes()))
    version, streams, blobs = record_data
    if version != 2:
        raise ValueError("Unsupported backend input index")

    @lru_cache(maxsize=12)
    def tree(key):
        return pickle.loads(blobs[key])

    different = set()
    for i, module, name, opts, events in streams:
        if i in different:
            continue
        cls = importlib.import_module(module)
        for part in name.split("."):
            cls = getattr(cls, part)
        instance = cls().generator(**opts)
        for key, path, expected in events:
            expr = tree(key)
            for attr, index in path:
                expr = expr.args[attr]
                if index is not None:
                    expr = expr[index]
            sql = instance.generate(expr, copy=True)
            if hashlib.sha256(sql.encode()).digest() != expected:
                different.add(i)
                break
    return different


if __name__ == "__main__":
    write_map(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
