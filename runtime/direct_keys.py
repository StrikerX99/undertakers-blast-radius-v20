"""Refine indirect generator-table reads with the keys observed at each native lookup."""

from functools import lru_cache
import json
from pathlib import Path


@lru_cache(maxsize=1)
def index():
    return json.loads(Path(__file__).with_name("direct-keys.json").read_bytes())


def filter_readers(sel, path, line, keys, readers):
    data = index()
    if sel.mapping["source_hashes"].get(path) != data["sources"].get(path):
        return readers
    site = data["sites"].get(path + ":" + str(line))
    if site is None or "*" in site and set(keys) & set(site["*"]):
        return readers
    # Use the proof only for native, single-process dialect validator methods.
    # Other readers, including subprocess consumers, keep the original treatment.
    eligible = set(data["eligible"])
    observed = set(site.get("@unknown", ()))
    for key in keys:
        observed.update(site.get(key, ()))
    names = data["names"]
    keep = {names[i] for i in observed}
    return {
        n
        for n in readers
        if (name := sel.by_number[n]["id"]) not in eligible or name in keep
    }
