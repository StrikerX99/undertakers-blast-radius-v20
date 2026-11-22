"""Build and query the source-indexed store for native test input footprints."""

import json
import marshal
import sqlite3
import zlib
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build(directory):
    file = directory / "case-index.sqlite"
    with sqlite3.connect(file) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS footprints (suite TEXT, source TEXT, data BLOB, PRIMARY KEY(suite,source)) WITHOUT ROWID"
        )
        for suite in ("cases", "fixtures"):
            obj = json.loads(
                zlib.decompress((directory / (suite + ".dat")).read_bytes())
            )
            rows = obj.pop("lines")
            db.execute(
                "INSERT OR REPLACE INTO footprints VALUES(?,?,?)",
                (suite, "", marshal.dumps(obj)),
            )
            for path, data in rows.items():
                bits = {int(line): int(value, 16) for line, value in data.items()}
                db.execute(
                    "INSERT OR REPLACE INTO footprints VALUES(?,?,?)",
                    (suite, path, marshal.dumps(bits)),
                )


@lru_cache(maxsize=1)
def connection():
    return sqlite3.connect(
        (HERE / "case-index.sqlite").as_uri() + "?mode=ro&immutable=1", uri=True
    )


@lru_cache(maxsize=120)
def get(suite, source=""):
    row = (
        connection()
        .execute(
            "SELECT data FROM footprints WHERE suite=? AND source=?", (suite, source)
        )
        .fetchone()
    )
    return marshal.loads(row[0]) if row else {}


if __name__ == "__main__":
    build(HERE)
