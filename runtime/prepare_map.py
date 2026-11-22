"""Partition one trusted baseline JSON map for cheap per-file access in the image."""

import hashlib
import json
import marshal
import sys
from pathlib import Path


def shard_name(source):
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def prepare(mapping_path, target):
    mapping = json.loads(mapping_path.read_bytes())
    target.mkdir(exist_ok=True)
    fields = ("lines", "comparisons", "regions", "import_lines")
    tables = {field: mapping.pop(field, {}) for field in fields}
    paths = set().union(*(table.keys() for table in tables.values()))
    # Store sets used by individual sources inside their own shard. The
    # lightweight master index still refers to the original numbered sets.
    used = set(mapping["files"].values())
    sets = mapping.pop("sets")
    for path in paths:
        data = {name: rows[path] for name, rows in tables.items() if path in rows}
        needed = set(data.get("lines", {}).values()) | set(
            data.get("regions", {}).values()
        )
        for record in data.get("comparisons", {}).values():
            needed.add(record["unknown"])
            needed.update(row[1] for row in record["variants"])
        data["sets"] = {i: sets[i] for i in needed}
        (target / shard_name(path)).write_bytes(marshal.dumps(data))
    mapping["sets"] = {i: sets[i] for i in used}
    # Timings from the offline collection are not needed by the running entry.
    mapping.pop("durations", None)
    mapping["paths"] = paths
    (target / "index").write_bytes(marshal.dumps(mapping))


if __name__ == "__main__":
    source, output = map(Path, sys.argv[1:])
    data = json.loads(source.read_bytes())

    # A hex bit denotes a catalog position. Expand before sharding by source file.
    def ids(row):
        if isinstance(row, list):
            return row
        mask = int(row, 16)
        return [i for i in range(mask.bit_length()) if mask & (1 << i)]

    coverage = data["coverage-map"]
    coverage["sets"] = [ids(row) for row in coverage["sets"]]
    for name, profile in data.items():
        (source.parent / (name + ".json")).write_text(
            json.dumps(profile, separators=(",", ":"))
        )
    prepare(source.parent / "coverage-map.json", output)
    import zlib

    observations = json.loads((source.parent / "observations.json").read_bytes())
    for name, dataset in observations.items():
        if "masks" in dataset:

            def hex_mask(value):
                if isinstance(value, str):
                    return value
                return format(
                    sum(((1 << length) - 1) << start for start, length in value), "x"
                )

            masks = [hex_mask(group) for group in dataset.pop("masks")]
            for rows in dataset["lines"].values():
                for line, position in rows.items():
                    rows[line] = masks[position]
        packed = json.dumps(dataset, separators=(",", ":")).encode()
        (source.parent / (name + ".dat")).write_bytes(zlib.compress(packed, 1))
