"""Which catalog methods observed the old existence or absence of a SQL fixture?"""

import hashlib
import json
from functools import lru_cache
from pathlib import Path, PurePosixPath


@lru_cache(maxsize=1)
def observations():
    return json.loads(Path(__file__).with_name("fixture-presence.json").read_bytes())


def choose(sel, changes, apply, root):
    if not changes or sel.unknown:
        return None
    if any((c["old"] == "/dev/null") == (c["new"] == "/dev/null") for c in changes):
        return None
    data = observations()
    if data["pin"] != sel.mapping["commit"]:
        return None
    readers = set()
    for change in changes:
        added = change["old"] == "/dev/null"
        name = change["new"] if added else change["old"]
        path = PurePosixPath(name)
        if (
            not name.startswith("tests/fixtures/")
            or path.suffix != ".sql"
            or path.as_posix() != name
            or ".." in path.parts
            or str(path.parent) not in data["directories"]
        ):
            return None
        f = root / name
        if f.is_symlink() or not f.parent.is_dir() or f.exists() == added:
            return None
        # Each path is present only on one side. Verify the supplied hunk is a
        # real add/deletion against the pin before using its dependency set.
        if added:
            apply("", change)
        else:
            raw = f.read_bytes()
            if hashlib.sha256(raw).hexdigest() != data["present"].get(name):
                return None
            if apply(raw.decode("utf8"), change)[0]:
                return None
        observations_for_path = set(data["probes"].get(name, ()))
        observations_for_path.update(data["enumerations"].get(str(path.parent), ()))
        if "bootstrap" in observations_for_path:
            return None
        # A fixture opened during setUpClass or setUpModule can fail every
        # method the fixture serves, even when only the first method opened it.
        for record in observations_for_path:
            phase, node = record.split(":", 1)
            if phase == "call":
                if node not in sel.by_name:
                    return None
                readers.add(sel.by_name[node])
            elif phase in ("setup", "teardown"):
                module = node.split("::", 1)[0]
                readers.update(t["number"] for t in sel.tests if t["file"] == module)
            else:
                return None
        if not added:
            # Retain the independent file readers from the original pinned suite.
            if name in sel.files:
                readers.update(sel.sets[sel.files[name]])
            if name not in data["probes"]:
                return None
    return readers
