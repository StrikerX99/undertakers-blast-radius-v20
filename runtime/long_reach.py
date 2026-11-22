"""Use a complete native method's cold process-tree coverage to price actual patch spans."""

from array import array
from functools import lru_cache
import json
import struct
import sys
import zlib
from pathlib import Path


@lru_cache(maxsize=1)
def index():
    raw = zlib.decompress(Path(__file__).with_name("long-reach.bin").read_bytes())
    header_size = int.from_bytes(raw[:4], "little")
    paths, tests = json.loads(raw[4 : 4 + header_size])
    masks = array("I")
    masks.frombytes(raw[4 + header_size :])
    if sys.byteorder != "little":
        masks.byteswap()
    return paths, tests, masks


def unused(mapping, spans):
    if not spans:
        return set()
    paths, tests, masks = index()
    active = 0
    for path, lines in spans.items():
        record = paths.get(path)
        if (
            not record
            or record[2] != mapping.get("source_hashes", {}).get(path)
            or not lines
        ):
            return set()
        offset, count = record[:2]
        if min(lines) < 1 or max(lines) >= count:
            return set()
        for line in lines:
            active |= masks[offset + line]
    return {name for i, name in enumerate(tests) if not active & (1 << i)}


def refine(sel, selected, spans):
    if not spans or not selected:
        return selected
    return selected - {
        sel.by_name[name]
        for name in unused(sel.mapping, spans)
        if name in sel.by_name and sel.by_name[name] not in sel.confirmed_failures
    }


def build(source, catalog, target, warm=None):
    src = Path(source)
    paths = json.loads((src / "index.json").read_bytes())
    rows = json.loads(Path(catalog).read_bytes())["tests"]
    names = {r["number"]: r["id"] for r in rows}
    keys = sorted(int(f.stem) for f in src.glob("*.ok"))
    size = max(offset + count for offset, count, _ in paths.values())
    if not 0 < len(keys) <= 32:
        raise ValueError("A map needs 1–32 independently graded methods.")
    masks = array("I", [0]) * size
    for i, key in enumerate(keys):
        bits = (src / f"{key}.bits").read_bytes()
        receipt = json.loads((src / f"{key}.ok").read_bytes())
        if len(bits) != size or receipt[2] != sum(bits) or not sum(bits):
            raise ValueError(f"Incomplete execution record for {key}")
        for line, value in enumerate(bits):
            if value:
                masks[line] |= 1 << i
    if warm is not None:
        baseline = json.loads(Path(warm).read_bytes())["coverage-map"]
        test_index = {name: i for i, name in enumerate(baseline["tests"])}
        watch = {
            test_index[names[n]]: 1 << i
            for i, n in enumerate(keys)
            if names[n] in test_index
        }
        warm_masks = [
            sum(watch[n] for n in group if n in watch) for group in baseline["sets"]
        ]
        for path, entries in baseline["lines"].items():
            file = paths.get(path)
            if not file:
                continue
            if baseline["source_hashes"].get(path) != file[2]:
                raise ValueError(f"Warm trace has different source for {path}")
            for line, group in entries.items():
                masks[file[0] + int(line)] |= warm_masks[group]
    if sys.byteorder != "little":
        masks.byteswap()
    header = json.dumps(
        [paths, [names[n] for n in keys]], separators=(",", ":")
    ).encode()
    dest = struct.pack("<I", len(header)) + header + masks.tobytes()
    Path(target).write_bytes(zlib.compress(dest, 9))


if __name__ == "__main__":
    build(*sys.argv[1:])


def exclude(sel, changes, selected, apply, equal):
    """Prove cold methods untouched and cache only exclusions supported by the full patch."""
    if not changes or not selected:
        return set()
    already = changes[0].get("_cold_unused")
    if already is not None:
        return already & selected - sel.confirmed_failures
    from closure_impact import spans as ast_spans
    from code_reach import spans as bytecode_spans

    paths, tests, _ = index()
    possible = {
        sel.by_name[n] for n in tests if n in sel.by_name
    } & selected - sel.confirmed_failures
    if not possible:
        return set()
    quick = {}
    for c in changes:
        path = c["old"]
        if not path or path != c["new"] or path not in paths or path in quick:
            return set()
        raw, _ = sel.source(path)
        _, old, inserted = apply(raw, c)
        quick[path] = old | inserted
    candidates = {
        sel.by_name[n] for n in unused(sel.mapping, quick) if n in sel.by_name
    }
    if not candidates & possible:
        return set()
    names = unused(sel.mapping, ast_spans(sel, changes, apply, equal))
    unexplained = (possible & candidates) - {
        sel.by_name[n] for n in names if n in sel.by_name
    }
    if unexplained:
        names.update(unused(sel.mapping, bytecode_spans(sel, changes, apply)))
    proved = {sel.by_name[n] for n in names if n in sel.by_name}
    # Caching a subset proved by the AST guard remains correct if the selection grows.
    changes[0]["_cold_unused"] = proved
    return proved & possible
