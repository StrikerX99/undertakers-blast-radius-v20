"""Resolve whole, deferred guard deletions against original statement entries."""

import ast

import case_verify
from source_fast import ancestors_at, paired_tree


def validation_guard(node):
    if not isinstance(node, ast.If) or node.orelse or len(node.body) != 1:
        return False
    condition = node.test
    if not (
        isinstance(condition, ast.Call)
        and isinstance(condition.func, ast.Attribute)
        and condition.func.attr == "get"
        and isinstance(condition.func.value, ast.Attribute)
        and condition.func.value.attr == "args"
        and isinstance(condition.func.value.value, ast.Name)
        and len(condition.args) == 1
        and isinstance(condition.args[0], ast.Constant)
        and type(condition.args[0].value) is str
        and not condition.keywords
    ):
        return False
    stmt = node.body[0]
    if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
        return False
    call = stmt.value
    return (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "raise_error"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "self"
        and all(isinstance(v, (ast.Constant, ast.Name)) for v in call.args)
        and all(
            k.arg is not None and isinstance(k.value, (ast.Constant, ast.Name))
            for k in call.keywords
        )
    )


def guard_entries(sel, change, apply, equal):
    path = change["old"]
    if (
        path != change["new"]
        or path != "sqlglot/parser.py"
        and not path.startswith("sqlglot/parsers/")
    ):
        return None
    raw, tree = sel.source(path)
    updated, removed, inserted = apply(raw, change)
    if inserted or not removed or change.get("_compiled_source") != updated:
        return None
    original = raw.splitlines()
    executable = {
        i
        for i in removed
        if original[i - 1].strip() and not original[i - 1].lstrip().startswith("#")
    }
    if not executable:
        return None
    entries = set()
    while executable:
        line = min(executable)
        ancestors = ancestors_at(tree, line)
        guard = next(
            (
                n
                for n in ancestors
                if isinstance(n, ast.If) and n.lineno == line and validation_guard(n)
            ),
            None,
        )
        if guard is None:
            return None
        span = set(range(line, guard.end_lineno + 1))
        if not span <= removed:
            return None
        functions = [
            n
            for n in ancestors
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
        ]
        classes = [n for n in ancestors if isinstance(n, ast.ClassDef)]
        if (
            len(functions) != 1
            or not isinstance(functions[0], ast.FunctionDef)
            or functions[0].decorator_list
            or len(classes) != 1
            or not classes[0].name.endswith("Parser")
            or not functions[0].args.args
            or functions[0].args.args[0].arg != "self"
        ):
            return None
        start = original[line - 1].encode()[: guard.col_offset].strip()
        end = original[guard.end_lineno - 1].encode()[guard.end_col_offset :].strip()
        if start or (end and not end.startswith(b"#")):
            return None
        entries.update(span)
        executable -= span
    # A removed line that looks like a comment may belong to a string literal.
    canonical = "".join(
        line for i, line in enumerate(raw.splitlines(True), 1) if i not in entries
    )
    if canonical != updated and not equal(
        ast.parse(canonical), paired_tree(change, tree, raw, updated)
    ):
        return None
    return entries


def refine(sel, changes, selected, apply, equal):
    if not selected or sel.unknown or not changes:
        return selected
    if any(
        line.startswith("+")
        for change in changes
        for h in change["hunks"]
        for line in h["lines"]
    ):
        return selected
    spans = {}
    readers = set(sel.confirmed_failures)
    for change in changes:
        path = change["old"]
        if (
            path in spans
            or path != change["new"]
            or not path.startswith(("sqlglot/parser.py", "sqlglot/parsers/"))
        ):
            return selected
        lines = guard_entries(sel, change, apply, equal)
        if not lines or lines.intersection(
            sel.mapping.get("import_lines", {}).get(path, ())
        ):
            return selected
        spans[path] = lines
        if path in sel.files:
            readers.update(sel.sets[sel.files[path]])
    # The exhaustive coordinator/worker trace spans both benchmark suites,
    # reference SQL emission, native worker transport and setup.
    mask = case_verify.phase_mask(spans)
    observed = [
        r for i, r in enumerate(case_verify.index()["cases"]) if mask & (1 << i)
    ]
    if any(r["kind"] in ("shared", "executor_setup", "validation") for r in observed):
        return selected
    from cold_parser_lines import untouched

    retained = set(selected)
    for suite in case_verify.SUITES:
        number = sel.by_name.get(case_verify.EXEC + suite)
        family = "tpcds" if suite.endswith("tpcds") else "tpch"
        if (
            number not in readers
            and untouched(sel.mapping, spans, suite)
            and not any(
                isinstance(r["key"], list) and r["key"][0] == family for r in observed
            )
        ):
            retained.discard(number)
    return retained
