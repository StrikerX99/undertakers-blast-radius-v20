"""Locate the first changed Python operations reachable from matching baseline/patched control flow."""

from collections import deque
import dis
from functools import lru_cache
import marshal
import types


@lru_cache(maxsize=16)
def compile_base(path, raw):
    return compile(raw, path, "exec", dont_inherit=True)


def constant(value):
    if isinstance(value, types.CodeType):
        return (
            "code",
            value.co_name,
            value.co_qualname,
            value.co_flags,
            value.co_posonlyargcount,
            value.co_argcount,
            value.co_kwonlyargcount,
            value.co_freevars,
            value.co_cellvars,
            value.co_varnames[
                : value.co_argcount
                + value.co_kwonlyargcount
                + bool(value.co_flags & 4)
                + bool(value.co_flags & 8)
            ],
            value.co_consts[0]
            if value.co_consts and isinstance(value.co_consts[0], str)
            else None,
        )
    return ("value", marshal.dumps(value))


def operations(code):
    steps = list(dis.get_instructions(code))
    by_offset = {s.offset: i for i, s in enumerate(steps)}
    regions = {}
    for e in dis.Bytecode(code).exception_entries:
        for i, s in enumerate(steps):
            if e.start <= s.offset < e.end:
                regions.setdefault(i, (e.depth, e.lasti, by_offset[e.target]))
    return steps, by_offset, regions


def different_lines(old, new):
    changed = set()
    handled = set()

    def diff(a, b):
        pair = id(a), id(b)
        if pair in handled:
            return
        handled.add(pair)
        entire = lambda: changed.update(l for _, _, l in a.co_lines() if l is not None)
        if (a.co_name, a.co_qualname, a.co_flags, a.co_freevars, a.co_cellvars) != (
            b.co_name,
            b.co_qualname,
            b.co_flags,
            b.co_freevars,
            b.co_cellvars,
        ):
            entire()
            return
        # Layout differences may affect introspection and local/global resolution.
        # Their consequences inside this frame conservatively cover all its lines.
        if a.co_varnames != b.co_varnames:
            entire()
            return
        if (
            a.co_code == b.co_code
            and a.co_names == b.co_names
            and a.co_exceptiontable == b.co_exceptiontable
            and len(a.co_consts) == len(b.co_consts)
            and all(
                constant(x) == constant(y) for x, y in zip(a.co_consts, b.co_consts)
            )
        ):
            for x, y in zip(a.co_consts, b.co_consts):
                if isinstance(x, types.CodeType):
                    diff(x, y)
            return
        left, lo, le = operations(a)
        right, ro, re = operations(b)
        todo = [(0, 0)]
        seen = set()
        while todo:
            i, j = todo.pop()
            if (i, j) in seen:
                continue
            seen.add((i, j))
            if len(seen) > 18000 or i >= len(left) or j >= len(right):
                entire()
                return
            x, y = left[i], right[j]

            def mark():
                if x.line_number is None:
                    entire()
                else:
                    changed.add(x.line_number)

            if x.opcode != y.opcode:
                mark()
                continue
            eh, fh = le.get(i), re.get(j)
            if (eh is None) != (fh is None) or eh and eh[:2] != fh[:2]:
                mark()
                continue
            if eh:
                todo.append((eh[2], fh[2]))
            if x.jump_target is not None:
                todo.append((lo[x.jump_target], ro[y.jump_target]))
                if x.opname.startswith(("JUMP_FORWARD", "JUMP_BACKWARD")):
                    continue
            elif x.opcode in dis.hasconst:
                ox, oy = a.co_consts[x.arg], b.co_consts[y.arg]
                if constant(ox) != constant(oy):
                    mark()
                    continue
                if isinstance(ox, types.CodeType):
                    diff(ox, oy)
            elif (
                x.opcode in dis.hasname
                or x.opcode in dis.haslocal
                or x.opcode in dis.hasfree
            ):
                if x.argrepr != y.argrepr or x.argval != y.argval:
                    mark()
                    continue
            elif x.arg != y.arg:
                mark()
                continue
            if x.opname not in {
                "RETURN_VALUE",
                "RETURN_CONST",
                "RAISE_VARARGS",
                "RERAISE",
                "INTERPRETER_EXIT",
            }:
                todo.append((i + 1, j + 1))

    diff(old, new)
    return frozenset(changed)


def spans(sel, changes, apply):
    if not changes:
        return None
    out = {}
    for c in changes:
        p = c["old"]
        if (
            not p
            or p != c["new"]
            or not p.startswith("sqlglot/")
            or not p.endswith(".py")
            or p in out
        ):
            return None
        if "_code_reach" not in c:
            raw, _ = sel.source(p)
            new = apply(raw, c)[0]
            oldcode = compile_base(p, raw)
            updated = compile(new, p, "exec", dont_inherit=True)
            c["_code_reach"] = different_lines(oldcode, updated)
        if c["_code_reach"]:
            out[p] = c["_code_reach"]
    return out or None
