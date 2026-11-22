"""Record pinned DataFrame shapes at the real executor test's comparison boundary."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def index():
    return json.loads(Path(__file__).with_name("executor-shapes.json").read_bytes())


def wrong_shape(family, i, generated_sql, table):
    if table is None:
        return False
    row = index().get(f"{family}/{i}")
    if row is None or row[0] != hashlib.sha256(generated_sql.encode()).hexdigest():
        return False
    # The real test constructs a DataFrame from these rows/columns and calls
    # assert_frame_equal. Unequal shapes cannot compare equal. This check
    # only retains tests; it never drops one based on a matching shape.
    return (len(table.rows), len(table.columns)) != (row[1], row[2])


def build(root, destination):
    import os
    import sys

    os.chdir(root)
    sys.path.insert(0, str(root))
    from sqlglot import transpile
    from tests.test_executor import TestExecutor

    TestExecutor.setUpClass()
    try:
        case = TestExecutor("test_execute_tpcds")
        result = {}
        for family in ("tpcds", "tpch"):
            for i, (metadata, sql, _) in enumerate(getattr(case, f"{family}_sqls")):
                if family == "tpcds" and not metadata.get("execute"):
                    continue
                generated = transpile(sql, write="duckdb")[0]
                frame = case.cached_execute(sql, tpch=family == "tpch")
                result[f"{family}/{i}"] = [
                    hashlib.sha256(generated.encode()).hexdigest(),
                    *frame.shape,
                ]
        destination.write_text(json.dumps(result, separators=(",", ":")))
    finally:
        TestExecutor.tearDownClass()


if __name__ == "__main__":
    import sys

    build(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
