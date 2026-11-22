"""Select native readers of changed tokenizer keywords and parser function registry keys."""

import ast
import json
import os
import re
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import forkprobe
import shared_tables
from source_fast import paired_tree

AMBIGUOUS = "multiple effective values"
WORD = re.compile(r"[A-Z][A-Z0-9_]{0,95}\Z", re.ASCII)


@lru_cache(maxsize=1)
def profile():
    return json.loads(Path(__file__).with_name("registry-reads.json").read_bytes())


def binding(n):
    if isinstance(n, ast.Assign) and len(n.targets) == 1:
        lhs = n.targets[0]
    elif isinstance(n, ast.AnnAssign):
        lhs = n.target
    else:
        return None
    return lhs.id if isinstance(lhs, ast.Name) else None


def key(n):
    return (
        n.value
        if isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and WORD.fullmatch(n.value)
        else None
    )


def passive(n, field):
    if field == "KEYWORDS":
        return (
            isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name)
            and n.value.id == "TokenType"
            and n.attr.isupper()
        )
    if isinstance(n, ast.Attribute):
        return (
            n.attr == "from_arg_list"
            and isinstance(n.value, ast.Attribute)
            and isinstance(n.value.value, ast.Name)
            and n.value.value.id == "exp"
            and n.value.attr[:1].isupper()
        )
    if not isinstance(n, ast.Lambda) or n.args.defaults or any(n.args.kw_defaults):
        return False
    params = {a.arg for a in ast.walk(n.args) if isinstance(a, ast.arg)}
    for x in ast.walk(n.body):
        if isinstance(
            x, (ast.NamedExpr, ast.Lambda, ast.ListComp, ast.GeneratorExp, ast.Yield)
        ):
            return False
        if isinstance(x, ast.Attribute) and x.attr.startswith("_"):
            return False
        if isinstance(x, ast.Name) and x.id not in params | {
            "exp",
            "seq_get",
            "bool",
            "len",
        }:
            return False
        if isinstance(x, ast.Call):
            f = x.func
            if isinstance(f, ast.Name) and f.id in {"seq_get", "bool", "len"}:
                continue
            if (
                isinstance(f, ast.Attribute)
                and isinstance(f.value, ast.Name)
                and f.value.id == "exp"
                and f.attr[:1].isupper()
            ):
                continue
            return False
    return True


def entries(expr, field):
    if not isinstance(expr, ast.Dict):
        return None
    current = {}
    other = []
    for k, v in zip(expr.keys, expr.values):
        if key(k) and passive(v, field):
            ks = [key(k)]
            value = v
        elif (
            k is None
            and field == "FUNCTIONS"
            and isinstance(v, ast.Call)
            and not v.keywords
            and isinstance(v.func, ast.Attribute)
            and isinstance(v.func.value, ast.Name)
            and (v.func.value.id, v.func.attr) == ("dict", "fromkeys")
            and len(v.args) == 2
            and isinstance(v.args[0], (ast.Tuple, ast.List))
            and all(key(z) for z in v.args[0].elts)
            and passive(v.args[1], field)
        ):
            ks = [key(z) for z in v.args[0].elts]
            value = v.args[1]
        else:
            other.append((k, v))
            continue
        for name in ks:
            if name in current:
                return None
            current[name] = (len(other), ast.dump(value))
    return current, other


def safe_bindings(tree):
    # These are the only names inserted dictionary expressions can evaluate.
    # SQLGlot's referenced primitives are unchanged imports or ordinary builtins.
    wanted = {
        "TokenType": {
            ("sqlglot.tokens", "TokenType"),
            ("sqlglot.tokenizer_core", "TokenType"),
        },
        "exp": {("sqlglot", "exp")},
        "seq_get": {("sqlglot.helper", "seq_get")},
        "dict": set(),
        "len": set(),
        "bool": set(),
    }

    def scope(stmts):
        for n in stmts:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if n.name in wanted:
                    return False
                continue
            if isinstance(n, ast.ClassDef):
                if n.name in wanted or not scope(n.body):
                    return False
                continue
            for t in ast.walk(n):
                if isinstance(t, (ast.Import, ast.ImportFrom)):
                    for alias in t.names:
                        name = alias.asname or alias.name.split(".")[0]
                        if alias.name == "*":
                            return False
                        if name in wanted and (
                            not isinstance(t, ast.ImportFrom)
                            or t.level
                            or (t.module, alias.name) not in wanted[name]
                        ):
                            return False
                if (
                    isinstance(t, ast.Name)
                    and isinstance(t.ctx, (ast.Store, ast.Del))
                    and t.id in wanted
                ):
                    return False
        return True

    return scope(tree.body)


def edits(path, a, b, equal):
    """Only allow substitutions in one class dictionary with passive new expressions."""
    requests = []

    def walk(x, y, nesting=()):
        if len(x) != len(y):
            return False
        for u, v in zip(x, y):
            if equal(u, v):
                continue
            if (
                isinstance(u, ast.ClassDef)
                and isinstance(v, ast.ClassDef)
                and all(
                    equal(getattr(u, k), getattr(v, k))
                    for k in u._fields
                    if k != "body"
                )
            ):
                if not walk(u.body, v.body, (*nesting, u.name)):
                    return False
                continue
            field = binding(u)
            if not (
                field in ("KEYWORDS", "FUNCTIONS")
                and type(u) is type(v)
                and nesting
                and (
                    field == "KEYWORDS"
                    and path.startswith("sqlglot/dialects/")
                    and len(nesting) == 2
                    and nesting[-1] == "Tokenizer"
                    or field == "FUNCTIONS"
                    and path.startswith("sqlglot/parsers/")
                    and len(nesting) == 1
                    and nesting[0].endswith("Parser")
                )
                and all(
                    equal(getattr(u, k), getattr(v, k))
                    for k in u._fields
                    if k != "value"
                )
            ):
                return False
            left, right = entries(u.value, field), entries(v.value, field)
            if (
                not left
                or not right
                or len(left[1]) != len(right[1])
                or any(
                    not equal(i, m) or not equal(j, n)
                    for (i, j), (m, n) in zip(left[1], right[1])
                )
            ):
                return False
            # Include keys moved to another position: a preceding or following spread may shadow them.
            touched = {
                k
                for k in left[0].keys() | right[0].keys()
                if left[0].get(k) != right[0].get(k)
            }
            if not touched:
                return False
            requests.append((field, sorted(touched)))
        return True

    return requests if walk(a.body, b.body) and requests else None


def snapshot(payload):
    root, field, keys, classes, out = payload
    os.chdir(root)
    sys.path.insert(0, str(root))
    import importlib
    import sqlglot
    from sqlglot import Parser, Tokenizer

    if Path(sqlglot.__file__).resolve() != (root / "sqlglot/__init__.py").resolve():
        return
    for name in sorted((root / "sqlglot/dialects").glob("*.py")):
        if name.stem != "__init__":
            importlib.import_module("sqlglot.dialects." + name.stem)
    for name in sorted((root / "sqlglot/parsers").glob("*.py")):
        if name.stem != "__init__":
            importlib.import_module("sqlglot.parsers." + name.stem)
    for cls in classes:
        module = cls.split(":", 1)[0]
        if module.startswith("sqlglot."):
            importlib.import_module(module)
    base = Tokenizer if field == "KEYWORDS" else Parser
    stack = [base]
    seen = set()
    rows = {}
    for c in stack:
        if c in seen:
            continue
        seen.add(c)
        stack.extend(c.__subclasses__())
        label = c.__module__ + ":" + c.__qualname__
        d = getattr(c, field)
        if type(d) is not dict:
            return
        result = {}
        for k in keys:
            if field == "KEYWORDS" and (
                k == c.HINT_START or any(tok in k for tok in c.SINGLE_TOKENS)
            ):
                # Prefix, multi-character and hint behavior can change the token before
                # any original dictionary read. Keep the incoming selection.
                return
            val = d.get(k)
            result[k] = [k in d, shared_tables.encode(val)]
        prior = rows.get(label)
        if prior is not None:
            result = {
                k: result[k] if result[k] == prior[k] else AMBIGUOUS for k in keys
            }
        rows[label] = result
    out.write_text(json.dumps(rows, sort_keys=True, separators=(",", ":")))


def inspect(root, field, keys, classes):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "state"
        if (
            forkprobe.run((root, field, keys, classes, out), {}, snapshot, 2.0)
            and out.is_file()
        ):
            return json.loads(out.read_bytes())
    return None


@lru_cache(maxsize=32)
def baseline(root, field, keys, classes):
    try:
        from keycache import original

        cached = original(root, field, keys, classes)
    except (OSError, ValueError, TypeError):
        cached = None
    return inspect(root, field, keys, classes) if cached is None else cached


def direct_users(sel, field, owners, keys):
    # Consumers outside normal parsing/tokenization need independent line coverage.
    # Most production attribute reads are class-body copies; their resulting classes
    # were inspected above. Non-copy consumers on the pinned tree are retained here.
    users = set()
    if field == "FUNCTIONS":
        paths = {
            "sqlglot/anonymize.py",
            "sqlglot/expressions/builders.py",
            "sqlglot/parsers/prql.py",
            "sqlglot/generators/snowflake.py",
        }
        for path in paths:
            if path.endswith("snowflake.py") and "ARRAY_GENERATE_RANGE" not in keys:
                continue
            if path.endswith("prql.py") and not any("prql" in n for n in owners):
                continue
            _, root = sel.source(path)
            lines = {
                n.lineno
                for n in ast.walk(root)
                if isinstance(n, ast.Attribute) and n.attr == field
            }
            if set(sel.mapping["import_lines"].get(path, ())) & lines:
                return None
            sel.load_shard(path)
            users.update(sel.line_tests(path, lines))
        # The test suite defines custom parser subclasses by copying the Snowflake
        # registry. They can reach any key independently of the normal dialect names.
        if any("snowflake" in n for n in owners):
            for t in sel.tests:
                if t["file"] in (
                    "tests/test_generator.py",
                    "tests/dialects/test_dialect.py",
                ):
                    users.add(t["number"])
    return users


def refine(sel, changes, selected, apply, equal, checkout, root, remaining):
    if not selected or sel.unknown or len(changes) != 1 or remaining < 5:
        return selected
    c = changes[0]
    p = c["old"]
    if (
        p != c["new"]
        or not p.startswith(("sqlglot/parsers/", "sqlglot/dialects/"))
        or p in ("sqlglot/dialects/dialect.py", "sqlglot/parsers/base.py")
    ):
        return selected
    raw, a = sel.source(p)
    updated = apply(raw, c)[0]
    if c.get("_compiled_source") != updated:
        return selected
    req = edits(p, a, paired_tree(c, a, raw, updated), equal)
    if not req or len(req) != 1:
        return selected
    if not safe_bindings(a):
        return selected
    field, ts = req[0]
    keys = tuple(ts)
    info = profile()
    if info["pin"] != sel.mapping["commit"]:
        return selected
    cls = tuple(info["rows"])
    old = baseline(root, field, keys, cls)
    if old is None:
        return selected
    with tempfile.TemporaryDirectory() as temp:
        tmp = Path(temp) / "src"
        checkout(tmp, changes, sel)
        new = inspect(tmp, field, keys, cls)
    if not new or new.keys() != old.keys():
        return selected
    changed = {
        c: {
            k
            for k in keys
            if old[c][k] != new[c][k] or AMBIGUOUS in (old[c][k], new[c][k])
        }
        for c in new
    }
    owner_keys = {n: vals for n, vals in changed.items() if vals}
    if any(cls not in info["rows"] for cls in owner_keys):
        return selected
    changed_keys = set().union(*owner_keys.values()) if owner_keys else set()
    guards = direct_users(sel, field, owner_keys, changed_keys)
    if guards is None:
        return selected
    users = guards | set(sel.confirmed_failures)
    ext = sel.dispatch_map()
    users.update(ext.external | ext.external_by_source.get(p, set()))
    if p in sel.files:
        users.update(sel.sets[sel.files[p]])
    for receiver, row in info["rows"].items():
        if receiver in new:
            wanted = owner_keys.get(receiver, ())
            for k in wanted:
                if k in row:
                    users.update(
                        sel.by_name[info["tests"][i]]
                        for i in info["sets"][row[k]]
                        if info["tests"][i] in sel.by_name
                    )
        elif not receiver.startswith("sqlglot.") and changed_keys:
            # Test-defined classes can supply their own tokenizer punctuation, hint
            # prefix and dictionary access patterns. Retain every observed use.
            for i in info["sets"][info["activity"][receiver]]:
                name = info["tests"][i]
                if name in sel.by_name:
                    users.add(sel.by_name[name])
    result = selected & users
    if result != selected:
        sel.registry_candidates = frozenset(result)
    return result
