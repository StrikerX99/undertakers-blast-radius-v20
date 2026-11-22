"""Verify complete optimizer rule invocations on their pristine inputs and native outputs."""

import ast

from source_fast import analysis_tree

BUILTINS = {"isinstance", "len", "all", "any", "bool", "enumerate", "zip", "reversed"}
HELPERS = {
    "always_true",
    "always_false",
    "is_zero",
    "is_false",
    "is_null",
    "is_complement",
}
CALLS = {
    "get",
    "set",
    "pop",
    "replace",
    "copy",
    "eq",
    "is_type",
    "append",
    "extend",
    "unnest",
}
DATA = {
    "args",
    "this",
    "expression",
    "expressions",
    "parent",
    "index",
    "type",
    "is_int",
    "is_number",
    "is_string",
    "is_star",
    "is_leaf",
    "name",
}
CONSTRUCTORS = {"paren", "null", "true", "false", "boolean", "column", "var"}
BANNED = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
    ast.Match,
    ast.Import,
    ast.ImportFrom,
    ast.With,
    ast.AsyncWith,
    ast.While,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
    ast.Nonlocal,
    ast.Global,
    ast.Try,
    ast.TryStar,
    ast.Raise,
    ast.Delete,
    ast.AsyncFor,
)


def wrapped(decorator, tree):
    if not isinstance(decorator, ast.Name):
        return False
    if decorator.id in {"classmethod", "staticmethod"}:
        return True
    defs = [
        x
        for x in tree.body
        if isinstance(x, ast.FunctionDef) and x.name == decorator.id
    ]
    if len(defs) != 1:
        return False
    outer = defs[0]
    if outer.decorator_list or len(outer.args.args) != 1 or len(outer.body) != 2:
        return False
    inner, ret = outer.body
    return (
        isinstance(inner, ast.FunctionDef)
        and isinstance(ret, ast.Return)
        and isinstance(ret.value, ast.Name)
        and ret.value.id == inner.name
        and not inner.args.defaults
        and not any(inner.args.kw_defaults)
        and len(inner.decorator_list) == 1
        and ast.unparse(inner.decorator_list[0]) == f"wraps({outer.args.args[0].arg})"
    )


def local_transform(fn, tree, path):
    """Expression inputs, fresh local containers and pinned expression helpers only."""
    args = [a.arg for a in fn.args.posonlyargs + fn.args.args]
    if args and args[0] in {"self", "cls"}:
        args = args[1:]
    if (
        fn.name.startswith("__")
        or len(args) != 1
        or fn.args.vararg
        or fn.args.kwarg
        or fn.args.kwonlyargs
    ):
        return False
    if any(not wrapped(d, tree) for d in fn.decorator_list):
        return False
    nodes = [node for statement in fn.body for node in ast.walk(statement)]
    owners = {child: n for n in nodes for child in ast.iter_child_nodes(n)}
    locals_ = {
        n.id for n in nodes if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    }
    helpers = HELPERS if path == "sqlglot/optimizer/simplify.py" else set()
    reserved = BUILTINS | HELPERS | {"exp", "self", "cls"}
    local_lists = {
        n.id for n in nodes if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    }
    for n in nodes:
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            parent = owners.get(n)
            if not (
                isinstance(parent, ast.Assign) and isinstance(parent.value, ast.List)
            ):
                local_lists.discard(n.id)
    # SQLGlot can intern or share type objects. They may only be queried.
    type_names = set()
    for n in nodes:
        if isinstance(n, ast.Attribute) and n.attr == "type":
            parent = owners[n]
            if not (
                isinstance(parent, ast.Assign)
                and len(parent.targets) == 1
                and isinstance(parent.targets[0], ast.Name)
                and parent.value is n
            ):
                return False
            type_names.add(parent.targets[0].id)
    for n in nodes:
        if not (
            isinstance(n, ast.Name)
            and isinstance(n.ctx, ast.Load)
            and n.id in type_names
        ):
            continue
        parent = owners[n]
        if (
            isinstance(parent, ast.Compare)
            and len(parent.ops) == 1
            and isinstance(parent.ops[0], (ast.Is, ast.IsNot))
            and isinstance(parent.comparators[0], ast.Constant)
            and parent.comparators[0].value is None
        ):
            continue
        if isinstance(parent, ast.Attribute) and parent.attr == "is_type":
            continue
        if (
            isinstance(parent, ast.Attribute)
            and parent.attr == "expressions"
            and isinstance(owners.get(parent), (ast.BoolOp, ast.If))
        ):
            continue
        if (
            isinstance(parent, ast.Call)
            and isinstance(parent.func, ast.Attribute)
            and parent.func.attr == "is_type"
            and n in parent.args
        ):
            continue
        return False
    if locals_ & reserved:
        return False
    for n in nodes:
        parent = owners.get(n)
        if (
            isinstance(n, BANNED)
            or isinstance(n, (ast.Attribute, ast.Subscript))
            and not isinstance(n.ctx, ast.Load)
        ):
            return False
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            if n.id in locals_ or n.id in args:
                continue
            if n.id == "exp" and isinstance(parent, ast.Attribute):
                continue
            if (
                n.id in BUILTINS | helpers
                and isinstance(parent, ast.Call)
                and parent.func is n
            ):
                continue
            return False
        if isinstance(n, ast.Attribute):
            if isinstance(n.value, ast.Name) and n.value.id == "exp":
                if n.attr == "DType":
                    if not (
                        isinstance(parent, ast.Attribute) and parent.attr.isupper()
                    ):
                        return False
                elif (
                    isinstance(parent, ast.Call)
                    and parent.func is n
                    and n.attr in CONSTRUCTORS
                ):
                    pass
                elif not (
                    n.attr[:1].isupper()
                    and isinstance(parent, ast.Call)
                    and isinstance(parent.func, ast.Name)
                    and parent.func.id == "isinstance"
                    and len(parent.args) == 2
                    and parent.args[1] is n
                ):
                    return False
            elif ast.unparse(n.value) == "exp.DType":
                if not (
                    n.attr.isupper()
                    and isinstance(parent, ast.Call)
                    and isinstance(parent.func, ast.Attribute)
                    and parent.func.attr == "is_type"
                    and n in parent.args
                ):
                    return False
            elif isinstance(parent, ast.Call) and parent.func is n:
                if n.attr in {"append", "extend"} and not (
                    isinstance(n.value, ast.Name) and n.value.id in local_lists
                ):
                    return False
                if n.attr not in CALLS:
                    return False
            elif n.attr not in DATA:
                return False
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id not in BUILTINS | helpers
        ):
            return False
        if isinstance(n, ast.Call) and not isinstance(
            n.func, (ast.Name, ast.Attribute)
        ):
            return False
    return True


def changed_methods(old, new, equal, path):
    found = []

    def scope(a, b, owner=""):
        if len(a) != len(b):
            return False
        for left, right in zip(a, b):
            if equal(left, right):
                continue
            if type(left) is not type(right) or not isinstance(
                left, (ast.ClassDef, ast.FunctionDef)
            ):
                return False
            if any(
                not equal(getattr(left, f), getattr(right, f))
                for f in left._fields
                if f != "body"
            ):
                return False
            if isinstance(left, ast.ClassDef):
                if owner or not scope(left.body, right.body, left.name + "."):
                    return False
            else:
                if (
                    ast.get_docstring(left, clean=False)
                    != ast.get_docstring(right, clean=False)
                    or not local_transform(left, old, path)
                    or not local_transform(right, old, path)
                ):
                    return False
                found.append((owner + left.name, left))
        return True

    return found if scope(old.body, new.body) else []


def cases(sel, changes, apply, equal):
    import executor_stages
    from rule_replay import ledger

    if len(changes) != 1:
        return None
    change = changes[0]
    path = change["old"]
    if (
        path != change["new"]
        or not path.startswith("sqlglot/optimizer/")
        or not path.endswith(".py")
    ):
        return None
    if executor_stages.index()["sources"].get(path) != sel.mapping["source_hashes"].get(
        path
    ):
        return None
    raw, old = sel.source(path)
    fresh = analysis_tree(change, old, raw, apply(raw, change)[0])
    if not any(
        isinstance(n, ast.ImportFrom)
        and n.module == "__future__"
        and any(a.name == "annotations" for a in n.names)
        for n in old.body
    ):
        return None
    methods = changed_methods(old, fresh, equal, path)
    if not methods:
        return None
    jobs = {}
    db = ledger()
    bindings = []
    for name, method in methods:
        arg = (method.args.posonlyargs + method.args.args)[-1].arg
        bindings.append((path, name, arg))
        spans = set(range(method.body[0].lineno, method.end_lineno + 1))
        phases = executor_stages.query({path: spans}, sel.mapping, {path: name})
        if not phases or set(phases) != {"optimize"}:
            return None
        reached = {}
        for stage, q, usable in db.execute(
            "SELECT DISTINCT id,q,usable FROM reached JOIN stage USING(id) WHERE src=? AND line BETWEEN ? AND ?",
            (path, min(spans), max(spans)),
        ):
            if not usable:
                return None
            reached.setdefault(q, set()).add(stage)
        if set(reached) != phases["optimize"]:
            return None
        for q, stages in reached.items():
            jobs.setdefault(q, set()).update(stages)
    return jobs, bindings


def check(payload):
    """Accept only complete native rule calls with identical outputs and options."""
    import importlib
    import inspect
    import os
    import pickle
    import signal
    import sqlite3
    import sys
    import zlib
    from pathlib import Path

    root, jobs, bindings, out, budget = payload
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.setrecursionlimit(12000)
    signal.setitimer(signal.ITIMER_PROF, budget)
    from checkpoint_graph import expression_digest
    from rule_replay import options_sig
    from sqlglot import exp

    arguments = {}
    for path, qualname, parameter in bindings:
        obj = importlib.import_module(path[:-3].replace("/", "."))
        for part in qualname.split("."):
            obj = getattr(obj, part)
        code = inspect.unwrap(obj).__code__
        if (
            code.co_qualname != qualname
            or Path(code.co_filename).resolve() != (root / path).resolve()
        ):
            return
        arguments[code] = parameter

    seen = set()
    unsuitable = False

    def entered(code, offset):
        nonlocal unsuitable
        frame = sys._getframe(1)
        if code is not frame.f_code or not isinstance(
            frame.f_locals[arguments[code]], exp.Expr
        ):
            unsuitable = True
            raise ValueError("Unexpected operand in optimizer method")
        seen.add(code)

    mon = sys.monitoring
    idx = mon.COVERAGE_ID
    mon.use_tool_id(idx, "native optimizer operand types")
    mon.register_callback(idx, mon.events.PY_START, entered)
    for code in arguments:
        mon.set_local_events(idx, code, mon.events.PY_START)

    dbfile = Path(__file__).with_name("rule-inputs.sqlite")
    with sqlite3.connect(dbfile.as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        for stage in sorted({i for stages in jobs.values() for i in stages}):
            mod, name, blob, expected, options, usable = db.execute(
                "SELECT mod,fn,arg,outcome,opts,usable FROM stage WHERE id=?", (stage,)
            ).fetchone()
            if not usable:
                return
            expr, kw = pickle.loads(zlib.decompress(blob))
            func = getattr(importlib.import_module(mod), name)
            actual = func(expr, **kw)
            if expression_digest(actual) != expected or options_sig(kw) != options:
                return
    if not unsuitable and set(arguments) == seen:
        out.touch()
