"""Use cold native test inputs to qualify removed guards and local builder rewrites."""

import ast
import json
from functools import lru_cache
from pathlib import Path
from source_fast import paired_tree

CORE = "sqlglot/expressions/core.py"


@lru_cache(maxsize=1)
def profile():
    return json.loads(Path(__file__).with_name("native-boundaries.json").read_bytes())


def header(a, b, equal):
    return type(a) is type(b) and all(
        equal(getattr(a, k), getattr(b, k)) for k in a._fields if k != "body"
    )


def scalar(n):
    return (
        isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id.isupper()
        and isinstance(n.value, ast.Constant)
        and type(n.value.value) in (bool, int, str, type(None))
    )


def guard_key(stmt):
    if not isinstance(stmt, ast.If) or stmt.orelse:
        return False
    head = stmt.test
    while isinstance(head, ast.BoolOp) and isinstance(head.op, ast.And):
        head = head.values[0]
    if isinstance(head, ast.NamedExpr):
        head = head.value
    return (
        isinstance(head, ast.Call)
        and isinstance(head.func, ast.Attribute)
        and head.func.attr == "get"
        and isinstance(head.func.value, ast.Attribute)
        and head.func.value.attr == "args"
        and isinstance(head.func.value.value, ast.Name)
        and len(head.args) == 1
        and isinstance(head.args[0], ast.Constant)
        and type(head.args[0].value) is str
        and not head.keywords
    )


def fn_guards(a, b, equal):
    if (
        not header(a, b, equal)
        or a.decorator_list
        or any(
            isinstance(
                n,
                (
                    ast.Global,
                    ast.Nonlocal,
                    ast.Await,
                    ast.Yield,
                    ast.YieldFrom,
                    ast.ClassDef,
                ),
            )
            for fn in (a, b)
            for n in ast.walk(fn)
        )
    ):
        return None
    guards = []
    i = j = 0
    while i < len(a.body):
        left = a.body[i]
        if j < len(b.body) and equal(left, b.body[j]):
            i += 1
            j += 1
            continue
        if not guard_key(left):
            return None
        if (
            j < len(b.body)
            and isinstance(b.body[j], ast.If)
            and equal(left.test, b.body[j].test)
            and not b.body[j].orelse
        ):
            j += 1
        guards.append(left)
        i += 1
    if j != len(b.body):
        return None

    def locals_(fn):
        return {
            n.id
            for n in ast.walk(fn)
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))
        } | {arg.arg for arg in ast.walk(fn.args) if isinstance(arg, ast.arg)}

    changed = locals_(a) ^ locals_(b)
    leftout = [s for s in a.body if s not in guards]
    if any(
        isinstance(n, ast.Name) and n.id in changed
        for stmt in leftout
        for n in ast.walk(stmt)
    ):
        return None
    if any(
        isinstance(
            n,
            (
                ast.Import,
                ast.ImportFrom,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.Delete,
            ),
        )
        for g in guards
        for n in ast.walk(g)
    ):
        return None
    # Removing a local binding must not change how a surviving guard resolves its receiver.
    for g in guards:
        if any(
            isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in changed
            for n in ast.walk(g.test)
        ):
            return None
    return guards


def gen_change(sel, chs, apply, equal):
    fields = set()
    spans = {}
    sites = set()
    for ch in chs:
        p = ch["old"]
        if not (p == "sqlglot/generator.py" or p.startswith("sqlglot/generators/")):
            return None
        raw, a = sel.source(p)
        updated = apply(raw, ch)[0]
        b = paired_tree(ch, a, raw, updated)
        if len(a.body) != len(b.body):
            return None
        spans[p] = set()
        for x, y in zip(a.body, b.body):
            if equal(x, y):
                continue
            if not isinstance(x, ast.ClassDef) or not header(x, y, equal):
                return None
            i = j = 0
            while i < len(x.body):
                n = x.body[i]
                m = y.body[j] if j < len(y.body) else None
                if equal(n, m):
                    i += 1
                    j += 1
                    continue
                if scalar(n) and (
                    m is None or not scalar(m) or not equal(n.targets, m.targets)
                ):
                    fields.add(n.targets[0].id)
                    i += 1
                    continue
                if not isinstance(n, ast.FunctionDef) or not isinstance(
                    m, ast.FunctionDef
                ):
                    return None
                guards = fn_guards(n, m, equal)
                if not guards:
                    return None
                for node in guards:
                    spans[p].update(range(node.lineno, node.end_lineno + 1))
                    sites.add(f"{p}@{node.lineno}")
                i += 1
                j += 1
            if j != len(y.body):
                return None
    if not sites or not sites.issubset(profile()["guards"]):
        return None
    import symbol_impact

    for f in fields:
        for path, line, scope, kind in symbol_impact.index().get(f, ()):
            if line not in spans.get(path, ()):
                return None
    return sites


def literal(v):
    return ("constant", type(v).__name__, v)


class Unknown(Exception):
    pass


class Prefix:
    def __init__(self, types, env, helpers):
        self.types = types
        self.env = env
        self.helpers = helpers
        self.trace = []

    def op(self, *op):
        self.trace.append(op)
        return ("result", len(self.trace))

    def known_type(self, v):
        if v[0] == "param":
            return self.types[v[1]]
        if v[0] == "constant":
            return type(v[2]).__module__ + "." + type(v[2]).__qualname__
        raise Unknown()

    def truth(self, v):
        if v[0] == "constant":
            return bool(v[2])
        raise Unknown()

    def expr(self, n):
        if isinstance(n, ast.Constant):
            return literal(n.value)
        if isinstance(n, ast.Name):
            if n.id in self.env:
                return self.env[n.id]
            if n.id not in self.helpers["globals"]:
                raise Unknown()
            return ("global", n.id)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            value = self.expr(n.operand)
            return (
                literal(not self.truth(value))
                if value[0] == "constant"
                else self.op("not", value)
            )
        if isinstance(n, ast.IfExp):
            return self.expr(n.body if self.truth(self.expr(n.test)) else n.orelse)
        if (
            isinstance(n, ast.Compare)
            and len(n.ops) == 1
            and isinstance(n.ops[0], (ast.Is, ast.IsNot))
            and isinstance(n.comparators[0], ast.Constant)
            and n.comparators[0].value is None
        ):
            none = self.known_type(self.expr(n.left)) == "builtins.NoneType"
            return literal(none if isinstance(n.ops[0], ast.Is) else not none)
        if isinstance(n, ast.Attribute):
            return self.op("attr", self.expr(n.value), n.attr)
        if isinstance(n, ast.Call):
            if (
                isinstance(n.func, ast.Name)
                and n.func.id == "isinstance"
                and "isinstance" not in self.env
                and len(n.args) == 2
                and isinstance(n.args[1], ast.Name)
                and not n.keywords
            ):
                typ = self.known_type(self.expr(n.args[0]))
                cls = n.args[1].id
                if cls in self.env:
                    raise Unknown()
                parents = profile()["parents"].get(typ)
                if not parents:
                    raise Unknown()
                tgt = (
                    "builtins." + cls
                    if cls in {"str", "int", "float", "bool", "list", "tuple", "dict"}
                    else CORE[:-3].replace("/", ".") + "." + cls
                )
                if (
                    not any(tgt in group for group in profile()["parents"].values())
                    and cls not in self.helpers["classes"]
                ):
                    raise Unknown()
                return literal(tgt in parents)
            fn = self.expr(n.func)
            args = [self.expr(v) for v in n.args]
            if any(k.arg is None for k in n.keywords):
                raise Unknown()
            kwargs = {k.arg: self.expr(k.value) for k in n.keywords}
            if fn == ("global", "to_identifier"):
                h = self.helpers["to_identifier"]
                params = [a.arg for a in h.args.args]
                if len(args) > len(params) or set(kwargs) - set(params[len(args) :]):
                    raise Unknown()
                saved = self.env
                self.env = {
                    p: self.expr(d)
                    for p, d in zip(params[-len(h.args.defaults) :], h.args.defaults)
                }
                self.env.update(zip(params, args))
                self.env.update(kwargs)
                try:
                    value = self.body(h.body)
                    if value is None:
                        raise Unknown()
                    return value
                finally:
                    self.env = saved
            return self.op("call", fn, tuple(args), tuple(kwargs.items()))
        raise Unknown()

    def body(self, stmts):
        for n in stmts:
            if (
                isinstance(n, ast.Expr)
                and isinstance(n.value, ast.Constant)
                and type(n.value.value) is str
            ):
                continue
            if isinstance(n, ast.If):
                v = self.body(n.body if self.truth(self.expr(n.test)) else n.orelse)
                if v is not None:
                    return v
            elif (
                isinstance(n, ast.Assign)
                and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name)
            ):
                self.env[n.targets[0].id] = self.expr(n.value)
            elif isinstance(n, ast.Return):
                return self.expr(n.value)
            else:
                raise Unknown()
        return None


def annotations_only(a, b, equal):
    """Only deferred annotations may vary. Defaults and decorators remain significant."""
    if (
        not isinstance(a, ast.FunctionDef)
        or not isinstance(b, ast.FunctionDef)
        or a.name != b.name
    ):
        return False
    if not equal(a.decorator_list, b.decorator_list) or not equal(
        a.type_params, b.type_params
    ):
        return False

    # A decorator can consume annotations while constructing a function. These
    # two pinned decorators retain the function or construct an ordinary property.
    if any(ast.unparse(d) not in {"property", "t.overload"} for d in a.decorator_list):
        return False
    if a.type_comment != b.type_comment or a.type_params or b.type_params:
        return False

    def argspec(n):
        args = n.args
        slots = (
            args.posonlyargs,
            args.args,
            args.kwonlyargs,
            [args.vararg] if args.vararg else [],
            [args.kwarg] if args.kwarg else [],
        )
        return [[(x.arg, x.type_comment) for x in part] for part in slots] + [
            args.defaults,
            args.kw_defaults,
        ]

    return equal(argspec(a), argspec(b))


def builder(sel, ch, apply, equal):
    raw, a = sel.source(CORE)
    updated = apply(raw, ch)[0]
    b = paired_tree(ch, a, raw, updated)
    if len(a.body) != len(b.body) or "from __future__ import annotations" not in raw:
        return None
    edits = []
    helpers = {"classes": {f.name for f in a.body if isinstance(f, ast.ClassDef)}}
    for x in a.body:
        if isinstance(x, ast.FunctionDef) and x.name == "to_identifier":
            helpers[x.name] = x

    stable = {"isinstance", "str", "int", "bool", "float", "tuple", "dict", "list"}
    for st in a.body:
        if isinstance(st, (ast.ClassDef, ast.FunctionDef)):
            stable.add(st.name)
        if isinstance(st, (ast.Assign, ast.AnnAssign)):
            for target in st.targets if isinstance(st, ast.Assign) else [st.target]:
                if isinstance(target, ast.Name):
                    stable.add(target.id)
    helpers["globals"] = stable
    # Names used for type discrimination must resolve to their baseline builtin
    # or to a concrete class definition in this immutable module.

    def inspect(left, right, inside=False):
        for x, y in zip(left, right):
            if equal(x, y):
                continue
            if (
                isinstance(x, ast.ClassDef)
                and header(x, y, equal)
                and len(x.body) == len(y.body)
            ):
                if not inspect(x.body, y.body, True):
                    return False
                continue
            if not annotations_only(x, y, equal):
                return False
            if equal(x.body, y.body):
                continue
            if (
                inside
                or x.decorator_list
                or x.name in helpers
                or x.args.posonlyargs
                or x.args.vararg
                or x.args.kwarg
            ):
                return False
            if CORE + ":" + x.name not in profile()["builders"]:
                return False
            for f in (x, y):
                if any(
                    isinstance(
                        n,
                        (ast.Global, ast.Nonlocal, ast.Yield, ast.YieldFrom, ast.Await),
                    )
                    or isinstance(n, ast.Name)
                    and n.id
                    in {
                        "locals",
                        "globals",
                        "eval",
                        "exec",
                        "vars",
                        "inspect",
                        "sys",
                        "__class__",
                    }
                    for n in ast.walk(f)
                ):
                    return False
            # Parsed prefixes can discard assignments to a local. Neither side
            # may then resolve that local by looking up a module global instead.
            stores = {
                v.id
                for f in (x, y)
                for v in ast.walk(f)
                if isinstance(v, ast.Name) and isinstance(v.ctx, (ast.Store, ast.Del))
            }
            helpers["globals"].difference_update(stores)
            args = [p.arg for p in x.args.args + x.args.kwonlyargs]
            old = list(x.body)
            new = list(y.body)
            tail = []
            while old and new and equal(old[-1], new[-1]):
                tail.append(old.pop())
                new.pop()
            if not tail or len(edits):
                return False
            live = {
                n.id
                for st in tail
                for n in ast.walk(st)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
            }
            edits.append((x.name, args, old, new, live))
        return True

    if not inspect(a.body, b.body) or len(edits) != 1:
        return None
    fn, args, old, new, live = edits[0]

    def ok(shape):
        types = dict(zip(args, shape.split(",")))
        if len(types) != len(args):
            return False
        env = {
            k: literal({"None": None, "False": False, "True": True}[t])
            if t in ("True", "False", "None")
            else ("param", k)
            for k, t in types.items()
        }
        states = []
        try:
            for body in (old, new):
                p = Prefix(types, env.copy(), helpers)
                if p.body(body) is not None:
                    return False
                states.append((p.trace, {k: p.env[k] for k in live if k in p.env}))
        except (Unknown, KeyError):
            return False
        return states[0] == states[1]

    return CORE + ":" + fn, lru_cache(maxsize=None)(ok)


def refine(sel, chs, selected, apply, equal, checkout, remaining):
    if not chs or remaining < 1 or not selected:
        return selected
    if any(
        c["old"] not in (CORE, "sqlglot/generator.py")
        and not c["old"].startswith("sqlglot/generators/")
        for c in chs
    ):
        return selected
    if any(
        c["old"] != c["new"]
        or c["old"] not in profile()["sources"]
        or sel.mapping["source_hashes"].get(c["old"]) != profile()["sources"][c["old"]]
        for c in chs
    ):
        return selected
    if not (len(chs) == 1 and chs[0]["old"] == CORE) and not any(
        "if " in line or ".args.get(" in line
        for c in chs
        for h in c["hunks"]
        for line in h["lines"]
        if line.startswith(("-", "+"))
    ):
        return selected
    cold = profile()["cases"]
    eligible = {
        name
        for name in cold
        if sel.by_name.get(name) in selected
        and sel.by_name.get(name) not in sel.confirmed_failures
    }
    for c in chs:
        protected = sel.sets[sel.files[c["old"]]] if c["old"] in sel.files else set()
        eligible = {name for name in eligible if sel.by_name[name] not in protected}
    if not eligible:
        return selected
    if len(chs) == 1 and chs[0]["old"] == CORE:
        replay = builder(sel, chs[0], apply, equal)
        if replay is None:
            return selected
        key, ok = replay
        passing = {
            n for n in eligible if all(ok(shape) for shape in cold[n][1].get(key, ()))
        }
    else:
        sites = gen_change(sel, chs, apply, equal)
        if not sites or not sites.issubset(profile()["guards"]):
            return selected
        passing = {n for n in eligible if not sites.intersection(cold[n][0])}
    if not passing:
        return selected
    import tempfile
    import forkprobe
    from cold_inputs import load_modules

    with tempfile.TemporaryDirectory(prefix="br-boundary-") as tmp:
        root = Path(tmp) / "repo"
        out = Path(tmp) / "ok"
        checkout(root, chs, sel)
        mods = [c["old"][:-3].replace("/", ".") for c in chs]
        if (
            not forkprobe.run(
                (root, mods, out),
                {"PYTHONPATH": str(root)},
                load_modules,
                min(3.0, remaining),
            )
            or not out.is_file()
        ):
            return selected
    return set(selected) - {sel.by_name[n] for n in passing}
