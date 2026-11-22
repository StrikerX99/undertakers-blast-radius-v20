"""Cold source line coverage of the native executor tests, including spawned workers."""

import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def index():
    file = Path(__file__).with_name("cold-parser-lines.json")
    return json.loads(file.read_bytes())


def untouched(mapping, spans, suite):
    data = index()
    if not spans or mapping.get("commit") != data["pin"] or suite not in data["rows"]:
        return False
    for path, lines in spans.items():
        digest = data["sources"].get(path)
        if (
            not lines
            or not digest
            or digest != mapping.get("source_hashes", {}).get(path)
        ):
            return False
        if not set(lines).isdisjoint(data["rows"][suite].get(path, ())):
            return False
    return True
