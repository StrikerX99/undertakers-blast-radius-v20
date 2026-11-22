"""Query cold, spawned consumer footprints for the pinned native executor test."""

import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def index():
    path = Path(__file__).with_name("executor-stages.json")
    data = json.loads(path.read_bytes())
    data["sets"] = [int(mask, 16) for mask in data["sets"]]
    return data


def query(spans, mapping, scopes=None):
    """Return affected native input indices per stage; None means no usable evidence.

    spans maps pristine repo path to old source lines. scopes optionally maps that
    path to a compiler-qualified function name, for callers that can establish
    that creating/calling/importing that function's enclosing scope is unaffected.
    A None value means every line in the file. shared, setup and transport
    spans are returned as the sentinel -1 and must retain the test unless replayed.
    """
    d = index()
    if not spans or mapping.get("commit") != d["pin"]:
        return None
    bits = 0
    for path, lines in spans.items():
        file = d["files"].get(path)
        if file is None:
            # Multiple user patches may add new absolute imports. A path missing
            # from the pristine index is insufficient evidence for such changes.
            return None
        if d["sources"][path] != mapping.get("source_hashes", {}).get(path):
            return None
        scope = (scopes or {}).get(path)
        if scope is not None and scope not in file:
            return None
        for name, entries in file.items():
            if scope is not None and scope != name:
                continue
            for _, rows in entries:
                for line, key in rows:
                    if lines is None or line in lines:
                        bits |= d["sets"][key]
    result = {}
    count = d["count"]
    for k, stage in enumerate(d["phases"]):
        if k < 3:
            if (bits >> k) & 1:
                result[stage] = {-1}
            continue
        active = (bits >> (3 + (k - 3) * count)) & ((1 << count) - 1)
        if active:
            result[stage] = {i for i in range(count) if (active >> i) & 1}
    return result
