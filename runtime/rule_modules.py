"""Find optimizer modules whose only SQLGlot consumer is the immutable rules tuple."""

import ast
from source_fast import analysis_tree

DRIVER = "sqlglot/optimizer/optimizer.py"
FORBIDDEN = {
    "globals",
    "locals",
    "getattr",
    "setattr",
    "delattr",
    "eval",
    "exec",
    "open",
    "sys",
    "os",
    "random",
    "time",
    "inspect",
    "gc",
    "threading",
    "builtins",
    "cache_clear",
    "__dict__",
    "__class__",
    "__import__",
    "__globals__",
    "_hash",
}


def build(root, db, rules):
    stages = {
        fn.__module__: fn.__name__
        for fn in rules
        if fn.__module__.endswith("." + fn.__name__)
    }
    allowed = set(stages)
    for p in [
        *(root / "sqlglot").rglob("*.py"),
        root / "tests/test_executor.py",
        root / "tests/helpers.py",
    ]:
        full = p.relative_to(root).as_posix()
        if full == DRIVER:
            continue
        for node in ast.walk(ast.parse(p.read_bytes())):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                found = [mod] + [(mod + "." + a.name).lstrip(".") for a in node.names]
            elif isinstance(node, ast.Import):
                found = [a.name for a in node.names]
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                found = [node.value]
            else:
                continue
            for name in found:
                for target in list(allowed):
                    if (
                        name == target
                        or name == target.rsplit(".", 1)[-1]
                        and isinstance(node, ast.ImportFrom)
                        and node.level
                    ) and full != target.replace(".", "/") + ".py":
                        allowed.remove(target)
    db.execute("CREATE TABLE isolated_rule(mod TEXT PRIMARY KEY,fn TEXT)")
    db.executemany(
        "INSERT INTO isolated_rule VALUES(?,?)",
        [(m, stages[m]) for m in sorted(allowed)],
    )


def broad(sel, chs, apply, equal, db):
    from rule_replay import added_calls, external_write

    if len(chs) != 1:
        return None
    ch = chs[0]
    p = ch["old"]
    if (
        p != ch["new"]
        or not p.startswith("sqlglot/optimizer/")
        or not p.endswith(".py")
    ):
        return None
    mod = p[:-3].replace("/", ".")
    owner = db.execute("SELECT fn FROM isolated_rule WHERE mod=?", (mod,)).fetchone()
    if not owner:
        return None
    source, before = sel.source(p)
    after = analysis_tree(ch, before, source, apply(source, ch)[0])

    def split(tree):
        fns, imports, others = {}, [], []
        for n in tree.body:
            if isinstance(n, ast.FunctionDef):
                if n.name in fns:
                    return None
                fns[n.name] = n
            elif isinstance(n, ast.ImportFrom) and n.module in {
                "itertools",
                "collections",
                "typing",
            }:
                imports.append(ast.dump(n))
            else:
                others.append(n)
        return fns, set(imports), others

    l, r = split(before), split(after)
    if not l or not r or not equal(l[2], r[2]) or not r[1].issubset(l[1]):
        return None
    if owner[0] not in l[0] or owner[0] not in r[0]:
        return None

    def header(fn):
        return [getattr(fn, k) for k in fn._fields if k != "body"]

    if not equal(header(l[0][owner[0]]), header(r[0][owner[0]])):
        return None
    for name in l[0].keys() | r[0].keys():
        x, y = l[0].get(name), r[0].get(name)
        if equal(x, y):
            continue
        if not x or not y:
            if not name.startswith("_") or y is not None:
                return None
        for f in (x, y):
            if f is None:
                continue
            if f.decorator_list or external_write(f):
                return None
            if any(
                not isinstance(n, ast.Constant)
                for n in [*f.args.defaults, *(a for a in f.args.kw_defaults if a)]
            ):
                return None
            if any(
                isinstance(
                    n, (ast.Global, ast.Nonlocal, ast.Import, ast.ImportFrom, ast.Await)
                )
                or isinstance(n, (ast.Name, ast.Attribute))
                and getattr(n, "id", getattr(n, "attr", None)) in FORBIDDEN
                for n in ast.walk(f)
            ):
                return None
        if x and y:
            extra = added_calls(x, y)
            # These operations only return or mutate nodes of the captured graph.
            if any(
                n != "exp.alias_" and n.rsplit(".", 1)[-1] not in {"set", "replace"}
                for n in extra
            ):
                return None
    allrows = db.execute(
        "SELECT q,id FROM stage WHERE mod=? AND fn=? ORDER BY q", (mod, owner[0])
    ).fetchall()
    if (
        len(allrows)
        != db.execute("SELECT count(*) FROM query WHERE usable=1").fetchone()[0]
    ):
        return None
    return {q: {i} for q, i in allrows}
