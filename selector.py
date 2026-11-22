"""Coverage-directed test selection for the pinned SQLGlot baseline.

The map contains baseline dependencies only. Patches are analyzed afresh; sample
answer labels are never read. Set BR_DATA, SQLGLOT_ROOT and BR_MAP for local use.
"""

from __future__ import annotations

import ast
import dialect_cells
import parser_cells
import native_boundaries
import rule_replay
import grammar_readers
import grammar_cells
import text_dispatch
import direct_outcomes
import verification_budget
import guard_inputs
import test_dependencies
import expression_schema
import module_footprints
import composite_dialects
import inherited_methods
import general_dialects
import shared_tables
import optimizer_replay
import parser_evidence
import structural_dependencies
import flag_inputs
import parser_lookahead
import dispatch_helpers
import generator_contexts
import metadata_inputs
import simplifier_suites
import callable_kind
import input_evidence
import arg_lookups
import literal_impact
import token_impact
import dormant_settings
import logic_guard
import set_relations
import code_index
import query_impact
import reference_probe
import optimizer_checkpoints
import exception_impact
from gated_reads import GatedReads
from builder_replay import equivalent_tests, independent_of_companions
from method_probe import direct_methods
from scalar_coverage import ScalarCoverage
from time_dependencies import TimeDependencies
from fixture_views import FixtureViews
from source_fast import paired_tree, ancestors_at, FUNCTIONS, local_function
import json
import marshal
from prepare_map import shard_name
import hashlib
from workers import fill_spare_capacity, schedule
import os
from pathlib import Path, PurePosixPath
import re
import sys
import time
import runtime_v2
import subprocess
import tempfile
from value_profile import compare, snapshot
from insertions import inserted_before
from dispatch_coverage import DispatchCoverage
from import_scope import initializer_may_change

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("BR_DATA", "/data"))
ROOT = Path(os.environ.get("SQLGLOT_ROOT", "/opt/sqlglot"))
MAP = Path(os.environ.get("BR_MAP", str(HERE / "coverage-map.json")))
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def structurally_equal(left, right):
    if left is right:
        return True
    if type(left) is not type(right):
        return False
    if isinstance(left, ast.AST):
        for name in left._fields:
            if not structurally_equal(
                getattr(left, name, None), getattr(right, name, None)
            ):
                return False
        return True
    if isinstance(left, list):
        if len(left) != len(right):
            return False
        for a, b in zip(left, right):
            if not structurally_equal(a, b):
                return False
        return True
    return left == right


def parse_patch(text: str) -> list[dict]:
    changes = []
    current = None
    hunk = None
    for line in text.splitlines(keepends=True):
        if line.startswith("diff --git "):
            current = {"old": None, "new": None, "hunks": []}
            changes.append(current)
            hunk = None
        elif (
            hunk is None
            and current is not None
            and line.startswith(
                (
                    "old mode ",
                    "new mode ",
                    "rename from ",
                    "rename to ",
                    "copy from ",
                    "copy to ",
                    "GIT binary patch",
                )
            )
        ):
            raise ValueError("Non-content change")
        elif line.startswith("--- ") and hunk is None:
            path = line[4:].strip().split("\t")[0]
            current["old"] = path[2:] if path.startswith("a/") else path
        elif line.startswith("+++ ") and hunk is None:
            path = line[4:].strip().split("\t")[0]
            current["new"] = path[2:] if path.startswith("b/") else path
        elif match := HUNK.match(line):
            hunk = {
                "start": int(match[1]),
                "count": int(match[2] or 1),
                "new_count": int(match[4] or 1),
                "lines": [],
            }
            current["hunks"].append(hunk)
        elif hunk is not None and line[:1] in (" ", "+", "-"):
            hunk["lines"].append(line)
        elif hunk is not None and line.startswith("\\ No newline"):
            hunk["lines"][-1] = hunk["lines"][-1].rstrip("\n")
    if not changes or any(c["old"] is None or c["new"] is None for c in changes):
        raise ValueError("Unsupported diff")
    return changes


def apply_change(original: str, change: dict) -> tuple[str, set[int], set[int]]:
    """Apply exactly matched hunks and return affected baseline coordinates.

    Carry the displacement of the preceding hunk, as Git does. In repeated
    context accept an exact match at that predicted location. When it misses,
    accept a unique exact location or an unambiguous nearest exact location.
    """
    previous = change.get("_applied")
    if previous and previous[0] is original:
        return previous[1]
    source = original.splitlines(keepends=True)
    output = []
    removed = set()
    inserted = set()
    cursor = 0
    displacement = 0
    for hunk in change["hunks"]:
        lines = hunk["lines"]
        old_lines = [line[1:] for line in lines if line[0] in " -"]
        if (
            "new_count" in hunk
            and sum(line[0] in " +" for line in lines) != hunk["new_count"]
        ):
            raise ValueError("Hunk new-side length disagrees with its header")
        if len(old_lines) != hunk["count"]:
            raise ValueError("Hunk old-side length disagrees with its header")
        advertised = hunk["start"] - 1 if old_lines else hunk["start"]
        expected = advertised + displacement
        start = expected
        if (
            expected != advertised
            or start < cursor
            or start > len(source)
            or source[start : start + len(old_lines)] != old_lines
        ):
            if not old_lines:
                raise ValueError("Insertion offset outside the pinned source")
            matches = [
                i
                for i in range(cursor, len(source) - len(old_lines) + 1)
                if source[i] == old_lines[0]
                and source[i : i + len(old_lines)] == old_lines
            ]
            if not matches:
                raise ValueError("Hunk does not match pinned source")
            distance = min(abs(i - expected) for i in matches)
            nearest = [i for i in matches if abs(i - expected) == distance]
            absolute_distance = min(abs(i - advertised) for i in matches)
            absolute = [i for i in matches if abs(i - advertised) == absolute_distance]
            if len(nearest) != 1 or nearest != absolute:
                raise ValueError("Ambiguous hunk placement in pinned source")
            start = nearest[0]
        displacement = start - advertised
        output.extend(source[cursor:start])
        cursor = start
        for line in lines:
            if line[0] == "+":
                inserted.add(max(1, cursor))
                output.append(line[1:])
            else:
                cursor += 1
                if line[0] == "-":
                    removed.add(cursor)
                else:
                    output.append(line[1:])
    output.extend(source[cursor:])
    result = "".join(output), removed, inserted
    change["_applied"] = (original, result)
    return result


class MappedSets:
    def __init__(self, mapping, by_name):
        self.indices = mapping["sets"]
        self.numbers = [by_name.get(name) for name in mapping["tests"]]
        self.cache = {}

    def __getitem__(self, key):
        if key not in self.cache:
            numbers = self.numbers
            self.cache[key] = {
                numbers[index]
                for index in self.indices[key]
                if numbers[index] is not None
            }
        return self.cache[key]


class Selector:
    def __init__(self, tests, mapping):
        if mapping.get(
            "commit"
        ) != "eb11aec4be8cb2122afe713a4816662c3f0fb2fa" or mapping.get(
            "baseline_failures"
        ):
            raise ValueError("Invalid baseline map")
        self.tests = tests
        self.all = {t["number"] for t in tests}
        self.by_number = {t["number"]: t for t in tests}
        self.by_name = {t["id"]: t["number"] for t in tests}
        known_names = set(mapping["tests"])
        self.unknown = {t["number"] for t in tests if t["id"] not in known_names}
        # Only a small fraction of the map is used in any batch. Translate
        # test identifiers on demand rather than duplicating its entire index.
        self.sets = MappedSets(mapping, self.by_name)
        self.lines = mapping["lines"]
        self.files = mapping["files"]
        self.source_cache = {}
        self.code_hashes = code_index.load(HERE / "baseline-code.bin")
        self.file_cache = {}
        self.mapping = mapping
        self.dispatch = None
        self.gated = None
        self.views = None
        self.scalars = None
        self.time_deps = None
        self.shards = mapping.get("paths", set())
        self.shard_directory = HERE / "map-index"

    def load_shard(self, path):
        if path not in self.shards:
            return
        blob = marshal.loads((self.shard_directory / shard_name(path)).read_bytes())
        self.sets.indices.update(blob.pop("sets"))
        for field, rows in blob.items():
            table = self.mapping.setdefault(field, {})
            if field == "regions":
                old = table.get(path, {})
                table[path] = {**rows, **old}
            else:
                table[path] = rows
        self.shards.remove(path)

    def source(self, path):
        self.load_shard(path)
        if path not in self.source_cache:
            p = PurePosixPath(path)
            if p.is_absolute() or ".." in p.parts:
                raise ValueError("Unsafe patch path")
            raw = (ROOT / path).read_bytes()
            expected = self.mapping.get("source_hashes", {}).get(path)
            if expected and hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError("Source does not match dependency map")
            text = raw.decode("utf-8")
            tree = ast.parse(text) if path.endswith(".py") else None
            self.source_cache[path] = (text, tree)
        return self.source_cache[path]

    def line_tests(self, path, lines):
        rows = self.lines.get(path, {})
        result = set()
        for line in lines:
            key = rows.get(str(line))
            if key is not None:
                result.update(self.sets[key])
        return result

    def file_tests(self, path):
        if path not in self.file_cache:
            result = set()
            for key in self.lines.get(path, {}).values():
                result.update(self.sets[key])
            if path in self.files:
                result.update(self.sets[self.files[path]])
            result.update(t["number"] for t in self.tests if t["file"] == path)
            self.file_cache[path] = result
        return self.file_cache[path]

    def dispatch_map(self):
        if self.dispatch is None:
            data = json.loads((HERE / "dispatch.json").read_bytes())
            self.dispatch = DispatchCoverage(data, self.by_name)
        return self.dispatch

    def profiled_changes(self, path, old, new, in_function=False):
        try:
            return self._profiled_changes(path, old, new, in_function)
        except (ValueError, TypeError):
            if (
                in_function
                and path.startswith("sqlglot/")
                and isinstance(old, ast.expr)
                and isinstance(new, ast.expr)
            ):
                if self.gated is None:
                    data = json.loads((HERE / "gates.json").read_bytes())
                    self.gated = GatedReads(
                        data, self.mapping, self.by_name, self.dispatch_map()
                    )
                readers = self.gated.readers(path, old, new)
                if readers is not None:
                    return readers
            raise

    def _profiled_changes(self, path, old, new, in_function=False):
        """Refine coverage using values observed on the unchanged baseline.

        Only replay side-effect-free comparisons with unchanged operands (or
        literal operands). Unknown values keep their tests. Other edits use the
        ordinary dependency map; there is no patch or edit-kind answer table.
        """
        if structurally_equal(old, new):
            return set()
        if not in_function and isinstance(old, (ast.Assign, ast.AnnAssign)):
            if self.scalars is None:
                data = json.loads((HERE / "scalars.json").read_bytes())
                self.scalars = ScalarCoverage(data, self.by_name, self.dispatch_map())
            readers = self.scalars.changed(self, path, old, new, structurally_equal)
            if readers is not None:
                return readers
        if in_function and isinstance(old, (ast.If, ast.While, ast.IfExp)):
            guarded = logic_guard.analyze(self, path, old, new, structurally_equal)
            if guarded is not None:
                return guarded
        if (
            isinstance(old, ast.Compare)
            and isinstance(new, ast.Compare)
            and len(old.ops) == len(new.ops) == 1
            and in_function
        ):
            precise = literal_impact.affected(self, path, old, new, structurally_equal)
            if precise is not None:
                return precise
            key = f"{old.lineno}:{old.col_offset}:{old.end_lineno}:{old.end_col_offset}"
            record = self.mapping.get("comparisons", {}).get(path, {}).get(key)
            if record is None:
                raise ValueError("No operand observations")
            replacements = []
            for a, b in [
                (old.left, new.left),
                (old.comparators[0], new.comparators[0]),
            ]:
                if ast.dump(a) == ast.dump(b):
                    replacements.append(None)
                else:
                    try:
                        ast.literal_eval(a)
                        value = snapshot(ast.literal_eval(b))
                    except (ValueError, TypeError, SyntaxError):
                        raise ValueError("Operand expression changed")
                    if value[0] in ("unknown", "identity"):
                        raise ValueError("Unsupported operand")
                    replacements.append(value)
            affected = set(self.sets[record["unknown"]])
            for values, index in record["variants"]:
                left, right, identical, before = values
                if replacements[0] is not None:
                    left = replacements[0]
                if replacements[1] is not None:
                    right = replacements[1]
                op = type(new.ops[0]).__name__
                if any(v is not None for v in replacements) and op in ("Is", "IsNot"):
                    affected.update(self.sets[index])
                    continue
                try:
                    after = compare(op, left, right, identical)
                except (ValueError, TypeError, KeyError):
                    affected.update(self.sets[index])
                    continue
                if before != after:
                    affected.update(self.sets[index])
            if not any(v is not None for v in replacements):
                return set_relations.refine(
                    self,
                    path,
                    key,
                    type(old.ops[0]).__name__,
                    type(new.ops[0]).__name__,
                    affected,
                )
            return affected
        if isinstance(old, ast.Dict) and isinstance(new, ast.Dict) and not in_function:
            if self.time_deps is None:
                data = json.loads((HERE / "time-dependencies.json").read_bytes())
                self.time_deps = TimeDependencies(
                    data, self.by_name, self.dispatch_map()
                )
            derived_readers = self.time_deps.readers_for(
                path, old, new, structurally_equal
            )
            if derived_readers is not None:
                raw = self.line_tests(
                    path,
                    [
                        k.lineno
                        for k, a, b in zip(old.keys, old.values, new.values)
                        if k is not None and not structurally_equal(a, b)
                    ],
                )
                return derived_readers | raw
            dispatch_readers = self.dispatch_map().changed(
                path, old, new, structurally_equal
            )
            if dispatch_readers is not None:
                return dispatch_readers
            if len(old.keys) != len(new.keys) or any(
                a is None or b is None or ast.dump(a) != ast.dump(b)
                for a, b in zip(old.keys, new.keys)
            ):
                raise ValueError("Dictionary shape changed")
            affected = set()
            for key, a, b in zip(old.keys, old.values, new.values):
                if ast.dump(a) == ast.dump(b):
                    continue
                # Literal data edits have no execution side effects of their own.
                ast.literal_eval(a)
                ast.literal_eval(b)
                region = (
                    f"{key.lineno}:{key.col_offset}:{a.end_lineno}:{a.end_col_offset}"
                )
                index = self.mapping.get("regions", {}).get(path, {}).get(region)
                if index is None:
                    raise ValueError("No entry dependency observations")
                affected.update(self.sets[index])
            return affected
        if type(old) is not type(new):
            raise ValueError("Different expression structure")
        affected = set()
        in_function = in_function or isinstance(
            old, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
        )
        for field in old._fields:
            a, b = getattr(old, field, None), getattr(new, field, None)
            if isinstance(a, ast.AST) and isinstance(b, ast.AST):
                affected.update(self.profiled_changes(path, a, b, in_function))
            elif isinstance(a, list) and isinstance(b, list):
                if len(a) != len(b):
                    raise ValueError("Statement count changed")
                for x, y in zip(a, b):
                    if isinstance(x, ast.AST) and isinstance(y, ast.AST):
                        affected.update(self.profiled_changes(path, x, y, in_function))
                    elif type(x) is not type(y) or x != y:
                        raise ValueError("Unsupported edit")
            elif type(a) is not type(b) or a != b:
                raise ValueError("Unsupported edit")
        return affected

    def select(self, patch) -> set[int]:
        chosen = set(self.unknown)
        self.import_modules = set()
        changes = parse_patch(patch) if isinstance(patch, str) else patch
        fixtures = [
            c
            for c in changes
            if c["old"] == c["new"] and c["old"].startswith("tests/fixtures/")
        ]
        sources = [c for c in changes if c not in fixtures]
        only_test_sources = all(
            c["old"] == c["new"]
            and c["old"].startswith("tests/")
            and PurePosixPath(c["old"]).name.startswith("test_")
            and c["old"].endswith(".py")
            for c in sources
        )
        self.direct_tests = set() if only_test_sources else None
        self.mixed_fixture_methods = bool(fixtures and sources and only_test_sources)
        # Finish checking modified method ownership before comparing fixture views.
        for change in [*sources, *fixtures]:
            path = change["old"]
            if path == "/dev/null" or change["new"] != path:
                return set(self.all)
            original, tree = self.source(path)
            updated, removed, inserted = apply_change(original, change)
            if original == updated:
                continue
            if tree is None:
                # Non-Python files are data dependencies, including fixture comments.
                if input_evidence.unread_data(self, changes, path, original):
                    continue
                readers = self.file_tests(path) or self.all
                if (
                    (
                        not sources
                        or (
                            self.mixed_fixture_methods and self.direct_tests is not None
                        )
                    )
                    and path.endswith(".sql")
                    and path in self.files
                ):
                    if self.views is None:
                        self.views = FixtureViews(
                            HERE / "fixture-views.json",
                            ROOT,
                            self.mapping,
                            self.by_name,
                        )
                    readers = readers - self.views.unchanged_readers(
                        path, original, updated
                    )
                chosen.update(readers)
                continue
            old_lines = original.splitlines(keepends=True)
            new_lines = updated.splitlines(keepends=True)
            try:
                code = compile(updated.encode("utf-8"), path, "exec", dont_inherit=True)
                change["_compiled_source"] = updated
                baseline = self.code_hashes.get(path)
                if baseline is not None and code_index.fingerprint(code) == baseline:
                    if path in self.files:
                        chosen.update(self.sets[self.files[path]])
                    continue
                pair = local_function(tree, old_lines, new_lines, removed | inserted)
                if pair is None:
                    old_fragment, new_tree = (
                        tree,
                        paired_tree(change, tree, original, updated),
                    )
                else:
                    old_fragment, new_tree = pair
            except (SyntaxError, ValueError, TypeError):
                # A failed import can prevent an entire test module from collecting.
                if path.startswith("tests/test_") or path.startswith(
                    "tests/dialects/test_"
                ):
                    chosen.update(
                        test_dependencies.affected_by_failed_test_import(self, path)
                    )
                else:
                    chosen.update(self.all)
                continue
            if self.direct_tests is not None:
                direct = direct_methods(
                    tree, old_fragment, new_tree, structurally_equal, path, self.by_name
                )
                if direct is None or (self.mixed_fixture_methods and not direct):
                    self.direct_tests = None
                else:
                    self.direct_tests.update(direct)
                    if self.mixed_fixture_methods:
                        chosen.update(direct)
            if structurally_equal(old_fragment, new_tree):
                # Ignore actual Python comments/formatting, not comment-looking strings.
                if path in self.files:
                    chosen.update(self.sets[self.files[path]])
                continue
            grammar_readers.extend(self, path, tree, removed | inserted, chosen)
            imported = set(self.mapping.get("import_lines", {}).get(path, []))
            if path.startswith(("sqlglot/", "tests/")) and initializer_may_change(
                old_fragment, new_tree, removed | inserted, imported
            ):
                module = path[:-3].replace("/", ".")
                if module.endswith(".__init__"):
                    module = module[:-9]
                self.import_modules.add(module)
                if path.startswith("tests/"):
                    self.import_modules.update(
                        test_dependencies.imported_test_modules(self, path)
                    )
            if not removed and len(inserted) == 1 and path.startswith("sqlglot/"):
                first = next(iter(inserted)) + 1
                whole_tree = new_tree if pair is None else ast.parse(updated)
                lines = inserted_before(
                    tree,
                    whole_tree,
                    first,
                    len(new_lines) - len(old_lines),
                    structurally_equal,
                )
                if lines is not None:
                    readers = self.line_tests(path, lines)
                    # An absent trace point can also indicate optimized bytecode.
                    if readers:
                        chosen.update(readers)
                        if path in self.files:
                            chosen.update(self.sets[self.files[path]])
                        continue
            try:
                refined = self.profiled_changes(path, old_fragment, new_tree)
            except (ValueError, TypeError, SyntaxError, RecursionError):
                refined = None
            if refined is not None:
                chosen.update(refined)
                if path in self.files:
                    chosen.update(self.sets[self.files[path]])
                continue
            changed_lines = removed | inserted
            candidates = self.line_tests(path, changed_lines)
            # Multiline expressions may only execute their first physical line.
            # Expand to the innermost statement, but keep individual dictionary
            # entries precise when their own data dependency was observed.
            branches = {line: ancestors_at(tree, line) for line in changed_lines}
            for line in changed_lines:
                nodes = [
                    n
                    for n in branches[line]
                    if isinstance(n, ast.stmt)
                    and n.lineno <= line <= getattr(n, "end_lineno", n.lineno)
                ]
                if not nodes:
                    candidates.update(self.file_tests(path) or self.all)
                    continue
                stmt = min(nodes, key=lambda n: n.end_lineno - n.lineno)
                if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.ClassDef)):
                    candidates.update(self.all)
                elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    candidates.update(
                        self.line_tests(path, range(stmt.lineno, stmt.end_lineno + 1))
                    )
                elif isinstance(stmt, (ast.If, ast.While, ast.Assert)):
                    candidates.update(
                        self.line_tests(
                            path, range(stmt.test.lineno, stmt.test.end_lineno + 1)
                        )
                    )
                elif not self.line_tests(path, [line]):
                    candidates.update(
                        self.line_tests(path, range(stmt.lineno, stmt.end_lineno + 1))
                    )
                # Insertions can introduce an unconditional effect in a previously
                # untaken branch: include the enclosing executable block.
                if line in inserted and not removed:
                    functions = [
                        n
                        for n in branches[line]
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and n.lineno <= line <= n.end_lineno
                    ]
                    if functions:
                        fn = min(functions, key=lambda n: n.end_lineno - n.lineno)
                        candidates.update(
                            self.line_tests(path, range(fn.lineno, fn.end_lineno + 1))
                        )
                    else:
                        candidates.update(self.file_tests(path) or self.all)
            if not candidates:
                inside = all(
                    any(
                        isinstance(n, FUNCTIONS) and n.lineno < line
                        for n in branches[line]
                    )
                    for line in changed_lines
                )
                imported = set(self.mapping.get("import_lines", {}).get(path, []))
                if changed_lines & imported:
                    # For literal class/module data, broaden to every observed
                    # user of the module. The separate import check catches
                    # initialization failures. Executable import-time edits
                    # retain the full suite.
                    literal_data = True
                    for line in changed_lines:
                        statements = [
                            n
                            for n in branches[line]
                            if isinstance(n, (ast.Assign, ast.AnnAssign))
                            and n.lineno <= line <= n.end_lineno
                        ]
                        if not statements:
                            literal_data = False
                            break
                        try:
                            ast.literal_eval(
                                min(
                                    statements, key=lambda n: n.end_lineno - n.lineno
                                ).value
                            )
                        except (ValueError, TypeError):
                            literal_data = False
                            break
                    candidates = (
                        (self.file_tests(path) or set(self.all))
                        if literal_data
                        else set(self.all)
                    )
                elif not inside or not removed:
                    candidates = self.file_tests(path) or set(self.all)
            if candidates:
                equivalent = equivalent_tests(
                    path,
                    old_fragment,
                    new_tree,
                    ROOT,
                    self,
                    candidates,
                    structurally_equal,
                )
                if equivalent:
                    companions = [edit for edit in changes if edit is not change]
                    candidates.difference_update(
                        independent_of_companions(
                            self,
                            companions,
                            equivalent,
                            apply_change,
                            structurally_equal,
                        )
                    )
            if path in self.files:
                candidates.update(self.sets[self.files[path]])
            chosen.update(candidates)
        return chosen


def patched_checkout(destination, changes, selector):
    """Share baseline files read-only; materialize only patched paths in /tmp."""
    destination.mkdir()
    for child in ROOT.iterdir():
        if child.name != ".git":
            (destination / child.name).symlink_to(
                child, target_is_directory=child.is_dir()
            )
    for change in changes:
        old, new = change["old"], change["new"]
        if old != new and "/dev/null" not in (old, new):
            raise ValueError("Unsupported rename in overlay")
        name = new if old == "/dev/null" else old
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts or ".git" in relative.parts:
            raise ValueError("Invalid patch path")
        original = "" if old == "/dev/null" else selector.source(old)[0]
        updated, _, _ = apply_change(original, change)
        parent = destination
        for part in relative.parts[:-1]:
            parent /= part
            if parent.is_symlink():
                target = parent.resolve()
                if not target.is_dir() or not target.is_relative_to(ROOT):
                    raise ValueError("Unexpected directory in source overlay")
                parent.unlink()
                parent.mkdir()
                for child in target.iterdir():
                    (parent / child.name).symlink_to(child, target_is_directory=child.is_dir())
            elif not parent.exists() and old == "/dev/null":
                parent.mkdir()
        output = destination / relative
        if old == "/dev/null" and (output.exists() or output.is_symlink()):
            raise ValueError("New file already exists")
        if output.is_symlink():
            output.unlink()
        if new == "/dev/null":
            if updated:
                raise ValueError("Deletion did not remove the complete file")
        else:
            output.write_text(updated)



def guard_child(payload):
    import importlib

    os.chdir(payload["root"])
    sys.path.insert(0, str(payload["root"]))
    for name in payload["modules"]:
        importlib.import_module(name)
    payload["complete"].touch()


def verify_imports(selector, changes):
    if not selector.import_modules:
        return True
    with tempfile.TemporaryDirectory(prefix="br-import-") as temporary:
        root = Path(temporary) / "repo"
        patched_checkout(root, changes, selector)
        env = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1")
        import forkprobe

        complete = Path(temporary) / "ok"
        payload = dict(
            root=root, modules=sorted(selector.import_modules), complete=complete
        )
        if forkprobe.run(payload, env, guard_child, 3):
            return complete.exists()
        code = "import importlib,sys; [importlib.import_module(name) for name in sys.argv[1:]]"
        try:
            result = subprocess.run(
                [sys.executable, "-c", code, *sorted(selector.import_modules)],
                cwd=root,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False


def attach_value_reads(mapping):
    path = HERE / "config-reads.json"
    if not path.exists():
        return
    records = json.loads(path.read_bytes())
    if records["commit"] != mapping["commit"]:
        return
    names = {name: i for i, name in enumerate(mapping["tests"])}
    for source, sites in records["value_reads"].items():
        rows = mapping["regions"].setdefault(source, {})
        for span, readers in sites.items():
            # Distinct from general line coverage: these are baseline reads
            # of an entry's value, for changes that preserve its key.
            if isinstance(mapping["sets"], dict):
                slot = -1 - sum(1 for key in mapping["sets"] if key < 0)
                mapping["sets"][slot] = [names[reader] for reader in readers]
            else:
                slot = len(mapping["sets"])
                mapping["sets"].append([names[reader] for reader in readers])
            rows[span] = slot


def main():
    started = time.monotonic()
    tests = json.loads((DATA / "tests.json").read_text())["tests"]
    try:
        index = HERE / "map-index" / "index"
        if "BR_MAP" not in os.environ and index.is_file():
            mapping = marshal.loads(index.read_bytes())
            for field in ("lines", "regions", "comparisons", "import_lines"):
                mapping[field] = {}
        else:
            mapping = json.loads(MAP.read_bytes())
        attach_value_reads(mapping)
        selector = Selector(tests, mapping)
    except Exception as exc:
        print(f"Map unavailable; conservative fallback: {exc}", file=sys.stderr)
        selector = None
    probe_seconds = float(os.environ.get("BR_PROBE_SECONDS", "1.0"))
    patches = sorted((DATA / "commits").glob("*.patch"))
    # Scale within the published 180 s practice and 270 s final allowances.
    scale = max(1.0, min(1.5, len(patches) / 30))
    deadline = started + 135 * scale
    static_deadline = started + 158 * scale
    import resource
    def cpu_seconds():
        child = resource.getrusage(resource.RUSAGE_CHILDREN)
        return time.process_time() + child.ru_utime + child.ru_stime
    # Spend a bounded amount on ambiguous patches. Verified worker savings buy
    # additional observation time for later patches in the same batch.
    verification_credit = 6.0 + 0.03 * min(len(patches),45)
    durations = {t["number"]: round(t["seconds"] * 1000) for t in tests}
    zero_cost = {n for n, duration in durations.items() if duration == 0}
    for patch in patches:
        if selector:
            selector.validator_keys=None
            selector.confirmed_failures = set()
            selector.native_attempted = set()
            selector.verify_wall_deadline = deadline
            selector.cone_candidates = frozenset()
            selector.cone_owners = frozenset()
            selector.registry_candidates = frozenset()
            selector.fixture_parser_fields = None
        changes = []
        selected = {t["number"] for t in tests}
        if selector and time.monotonic() < static_deadline:
            try:
                changes = parse_patch(patch.read_text())
                precise=None
                import fixture_presence
                if any("/dev/null" in (c["old"],c["new"]) for c in changes):
                    precise = fixture_presence.choose(selector,changes,apply_change,ROOT)
                if precise is None:
                    import type_predicates
                    precise=type_predicates.residual(selector,changes,apply_change,structurally_equal,patched_checkout,deadline-time.monotonic())
                if precise is None:
                    import import_alias
                    precise=import_alias.observers(selector,changes,apply_change,structurally_equal)
                if precise is not None:
                    direct, _ = schedule([selector.by_number[n] for n in precise])
                    print(json.dumps({"commit":patch.stem,"workers":direct},separators=(",",":")))
                    continue
                candidates = selector.select(changes)
                candidates = module_footprints.refine(
                    selector, changes, candidates, apply_change, structurally_equal
                )
                candidates = inherited_methods.refine(
                    selector, changes, candidates, apply_change, structurally_equal
                )
                candidates = simplifier_suites.verify(
                    selector, changes, candidates, apply_change, structurally_equal
                )
                candidates = expression_schema.verify(
                    selector, changes, candidates, apply_change, structurally_equal
                )
                candidates = flag_inputs.verify(
                    selector, changes, candidates, apply_change, structurally_equal
                )
                candidates = parser_lookahead.verify(
                    selector, changes, candidates, apply_change, structurally_equal
                )
                candidates = text_dispatch.refine(
                    selector, changes, candidates, apply_change, structurally_equal
                )
                candidates = parser_evidence.verify(
                    selector, changes, candidates, apply_change, structurally_equal
                )
                candidates = guard_inputs.verify(
                    selector, changes, candidates, apply_change, structurally_equal
                )
                candidates = input_evidence.equivalent_guard(
                    selector, changes, candidates, apply_change
                )
                candidates = token_impact.verify(
                    selector, changes, candidates, apply_change
                )
                candidates = arg_lookups.verify(
                    selector, changes, candidates, apply_change, structurally_equal
                )
                candidates = exception_impact.refine(
                    selector, changes, candidates, apply_change, structurally_equal
                )
                candidates = dormant_settings.verify(
                    selector, changes, candidates, apply_change, structurally_equal
                )
                if probe_seconds > 0 and candidates and time.monotonic() < deadline:
                    try:
                        candidates = query_impact.verify(
                            selector,
                            changes,
                            candidates,
                            apply_change,
                            structurally_equal,
                            patched_checkout,
                            deadline - time.monotonic(),
                        )
                    except (OSError, ValueError, TypeError, KeyError, RecursionError):
                        # An optional replay never invalidates the existing schedule.
                        pass
                    try:
                        candidates = reference_probe.verify(
                            selector,
                            changes,
                            candidates,
                            apply_change,
                            structurally_equal,
                            patched_checkout,
                            deadline - time.monotonic(),
                        )
                    except (OSError, ValueError, TypeError, KeyError, RecursionError):
                        pass
                    try:
                        candidates = optimizer_checkpoints.verify(
                            selector,
                            changes,
                            candidates,
                            apply_change,
                            structurally_equal,
                            patched_checkout,
                            deadline - time.monotonic(),
                        )
                    except (OSError, ValueError, TypeError, KeyError, RecursionError):
                        pass
                    try:
                        candidates = optimizer_replay.verify(
                            selector,
                            changes,
                            candidates,
                            apply_change,
                            structurally_equal,
                            patched_checkout,
                            deadline - time.monotonic(),
                        )
                    except (OSError, ValueError, TypeError, KeyError, RecursionError):
                        pass
                augmented = structural_dependencies.strengthen(
                    selector, changes, candidates, apply_change, structurally_equal
                )
                augmented = composite_dialects.refine(
                    selector, changes, augmented, apply_change, structurally_equal
                )
                if time.monotonic() + 7 < deadline:
                    proposed = general_dialects.refine(
                        selector, changes, augmented, apply_change, structurally_equal
                    )
                    proposed = dispatch_helpers.refine(
                        selector, changes, proposed, apply_change, structurally_equal
                    )
                    augmented = shared_tables.verified(
                        selector,
                        changes,
                        augmented,
                        proposed,
                        patched_checkout,
                        ROOT,
                        deadline - time.monotonic(),
                    )
                augmented = generator_contexts.refine(
                    selector, changes, augmented, apply_change, structurally_equal
                )
                augmented = metadata_inputs.verify(
                    selector, changes, augmented, apply_change, structurally_equal
                )
                import generator_cells
                try:
                    augmented = generator_cells.refine(selector, changes, augmented,
                        apply_change, structurally_equal, patched_checkout, deadline-time.monotonic())
                except (KeyError,ValueError,RuntimeError,TypeError,OSError):
                    pass
                if time.monotonic() + 5 < deadline:
                    import registry_keys
                    try:
                        augmented = registry_keys.refine(selector, changes, augmented, apply_change, structurally_equal, patched_checkout, ROOT, deadline-time.monotonic())
                    except Exception:
                        pass
                if (
                    1 <= len(changes) <= 4
                    and (len(changes) > 1 or changes[0]["old"] == "sqlglot/dialects/dialect.py")
                    and all(
                        c["old"] == "sqlglot/dialects/dialect.py"
                        or c["old"].startswith("sqlglot/generators/")
                        for c in changes
                    )
                ):
                    try:
                        import shared_generator_helpers
                        augmented = shared_generator_helpers.refine(
                            selector, changes, augmented, apply_change, structurally_equal,
                            patched_checkout, deadline - time.monotonic(),
                        )
                    except (AttributeError, IndexError, KeyError, OSError, RuntimeError, TypeError, ValueError):
                        pass
                try:
                    import long_reach, test_cleanup
                    if not test_cleanup.changed(selector, changes, apply_change):
                        before_reach = frozenset(augmented)
                        excluded = long_reach.exclude(selector, changes, augmented, apply_change, structurally_equal)
                        if excluded:
                            augmented = augmented - excluded
                            price = getattr(selector, "verification_cost", None)
                            if price is None:
                                price = selector.verification_cost = verification_budget.catalog_price(selector.by_number)
                            gained = max(0,price(before_reach)-price(frozenset(augmented)))
                            verification_credit = min(12.0,verification_credit + min(2.5,0.30*gained/1000))
                except (ValueError, OSError, TypeError, KeyError, RecursionError, SyntaxError):
                    pass
                extra_readers = augmented - candidates
                candidates = augmented
                probing = (
                    probe_seconds > 0
                    and time.monotonic() < deadline
                    and runtime_v2.eligible(selector, changes, candidates)
                )
                if probing:
                    try:
                        selected = runtime_v2.verify(
                            selector,
                            changes,
                            candidates,
                            probe_seconds,
                            patched_checkout,
                            remaining_wall=deadline - time.monotonic(),
                        )
                    except Exception as exc:
                        print(
                            f"{patch.stem}: verification fallback: {exc}",
                            file=sys.stderr,
                        )
                        selected = (
                            set(selector.all) if selector.import_modules else candidates
                        )
                elif (
                    candidates == selector.all
                    and not input_evidence.can_narrow_failed_import(
                        selector, changes, deadline - time.monotonic()
                    )
                ):
                    # Both success and failure would emit every test.
                    selected = candidates
                elif verify_imports(selector, changes):
                    selected = candidates
                else:
                    print(
                        f"{patch.stem}: import failed; checking independent tests",
                        file=sys.stderr,
                    )
                    selected = (
                        input_evidence.unaffected_by_import_failure(
                            selector,
                            changes,
                            selected,
                            apply_change,
                            structurally_equal,
                            patched_checkout,
                            deadline - time.monotonic(),
                        )
                        | extra_readers
                    )
                selected = callable_kind.strengthen(
                    selector, changes, selected, apply_change
                )
            except Exception as exc:
                selected = set(selector.all)
                print(f"{patch.stem}: conservative fallback: {exc}", file=sys.stderr)
        if (
            selector
            and selected - selector.confirmed_failures
            and time.monotonic() < deadline
            and changes
        ):
            try:
                import import_cone
                try:
                    selected = import_cone.refine(
                        selector, changes, selected, apply_change, structurally_equal,
                        patched_checkout, ROOT, deadline - time.monotonic(),
                    )
                except Exception as exc:
                    print(f"{patch.stem}: import cone retained candidates: {exc}", file=sys.stderr)
                import method_flow
                try:
                    selected = method_flow.refine(
                        selector, changes, selected, apply_change, structurally_equal,
                        patched_checkout, ROOT, deadline - time.monotonic(),
                    )
                except Exception:
                    pass
                import deleted_guards
                selected = deleted_guards.refine(selector, changes, selected, apply_change, structurally_equal)
                selected = grammar_cells.refine(
                    selector,
                    changes,
                    selected,
                    apply_change,
                    structurally_equal,
                    patched_checkout,
                    deadline - time.monotonic(),
                )
                selected = parser_cells.verify(
                    selector,
                    changes,
                    selected,
                    apply_change,
                    structurally_equal,
                    patched_checkout,
                    deadline - time.monotonic(),
                )
                selected = dialect_cells.verify(
                    selector,
                    changes,
                    selected,
                    apply_change,
                    structurally_equal,
                    patched_checkout,
                    deadline - time.monotonic(),
                )
                selected = native_boundaries.refine(
                    selector,
                    changes,
                    selected,
                    apply_change,
                    structurally_equal,
                    patched_checkout,
                    deadline - time.monotonic(),
                )
                selected = rule_replay.refine(
                    selector,
                    changes,
                    selected,
                    apply_change,
                    structurally_equal,
                    patched_checkout,
                    min(12, deadline - time.monotonic()),
                )
                import optimizer_boundary
                try:
                    selected = optimizer_boundary.refine(selector, changes, selected, apply_change, structurally_equal, patched_checkout, min(15, deadline - time.monotonic()))
                except Exception:
                    pass
                import generator_reference
                try:
                    selected = generator_reference.refine(
                        selector, changes, selected, apply_change, structurally_equal,
                        patched_checkout, deadline - time.monotonic(), ROOT,
                    )
                except Exception:
                    pass
                selected = verification_budget.refine(
                    selector,
                    changes,
                    selected,
                    apply_change,
                    structurally_equal,
                    patched_checkout,
                    min(14, deadline - time.monotonic()),
                    use_cases=os.environ.get("BR_CASES", "1") != "0",
                    use_native=os.environ.get("BR_NATIVE", "1") != "0",
                )
            except Exception as exc:
                print(
                    f"{patch.stem}: retained selection after optional check: {exc}",
                    file=sys.stderr,
                )
        if selector and changes and time.monotonic() < deadline:
            try:
                selected = direct_outcomes.refine(
                    selector, changes, selected, patched_checkout, deadline - time.monotonic(),
                    apply_change, structurally_equal
                )
            except (OSError, ValueError, RuntimeError):
                pass
        if selector and changes and getattr(selector, "cone_candidates", None):
            try:
                import cone_cases
                selected = cone_cases.refine(
                    selector, changes, selected, patched_checkout, deadline - time.monotonic()
                )
            except Exception:
                pass
        if selector and changes and selected and time.monotonic() < deadline:
            if any(c["old"] == "sqlglot/parser.py" for c in changes):
                try:
                    import executor_dormant
                    selected = executor_dormant.refine(
                        selector, changes, selected, apply_change, structurally_equal
                    )
                except (OSError, ValueError, TypeError, KeyError, RuntimeError, RecursionError) as exc:
                    print(f"{patch.stem}: retained executor test: {exc}", file=sys.stderr)
            try:
                import scaled_parser
                selected = scaled_parser.refine(
                    selector, changes, selected, patched_checkout, deadline - time.monotonic()
                )
            except (OSError, ValueError, TypeError, KeyError, RuntimeError, RecursionError) as exc:
                print(f"{patch.stem}: retained parser stress test: {exc}", file=sys.stderr)
        if selector and changes and selected and time.monotonic() < deadline:
            import compile_guard
            try:
                selected = compile_guard.refine(selector,changes,selected,apply_change,structurally_equal,patched_checkout,deadline-time.monotonic())
            except (ValueError, OSError, TypeError, KeyError, RuntimeError, RecursionError, SyntaxError):
                pass
        if selector and getattr(selector,"validator_keys",None) and selected and time.monotonic()<deadline:
            import validator_cells
            try:
                selected=validator_cells.refine(selector,changes,selected,patched_checkout,deadline-time.monotonic())
            except (OSError,ValueError,RuntimeError,TypeError,KeyError,RecursionError):
                pass
        if selected and selector and len(changes) > 1:
            import composite_readers
            try:
                selected = composite_readers.refine(
                    selector, changes, selected, apply_change, structurally_equal,
                    patched_checkout, ROOT, deadline - time.monotonic(),
                )
            except (AttributeError, IndexError, KeyError, OSError, RecursionError, RuntimeError,
                    SyntaxError, TypeError, ValueError):
                pass
        prior_selection = selected
        if selector and changes and selected and time.monotonic() < deadline:
            import query_frontier, assertion_replay, whole_methods
            try:
                selected = module_footprints.refine(selector, changes, selected, apply_change, structurally_equal, broader=True)
                selected = set(selected) | selector.confirmed_failures
            except (OSError, ValueError, TypeError, KeyError, RecursionError):
                selected = prior_selection
            if selected != prior_selection:
                price = getattr(selector, "verification_cost", None)
                if price is None:
                    price = selector.verification_cost = verification_budget.catalog_price(selector.by_number)
                verification_credit = min(15.0, verification_credit + 0.75 * max(0, price(frozenset(prior_selection)) - price(frozenset(selected))) / 1000)
            for step in (query_frontier, assertion_replay, whole_methods):
                allowance = min(verification_credit, deadline - time.monotonic())
                if allowance < 1:
                    break
                observed = selected
                tick = cpu_seconds()
                try:
                    if step is whole_methods:
                        selected = step.refine(selector, changes, selected, patched_checkout, allowance)
                    else:
                        selected = step.refine(selector, changes, selected, apply_change, structurally_equal, patched_checkout, allowance)
                except Exception as exc:
                    print(f"{patch.stem}: retained tests after {step.__name__}: {exc}", file=sys.stderr)
                verification_credit -= max(0.0, cpu_seconds() - tick)
                if observed != selected:
                    price = getattr(selector,"verification_cost",None)
                    if price is None:
                        price = selector.verification_cost = verification_budget.catalog_price(selector.by_number)
                    savings = max(0, price(frozenset(observed)) - price(frozenset(selected)))
                    verification_credit = min(15.0, verification_credit + 0.75*savings/1000)
        if selector and changes and selected:
            import leaf_verified
            try:
                selected = leaf_verified.refine(
                    selector, changes, selected, apply_change, structurally_equal,
                    patched_checkout, deadline - time.monotonic(),
                )
            except (OSError, ValueError, TypeError, KeyError, SyntaxError, RecursionError):
                pass
        protected = set()
        if selector and changes:
            import derived_values
            try:
                protected = derived_values.consumers(selector, changes, selected, apply_change, patched_checkout, deadline-time.monotonic())
            except (ValueError, OSError, TypeError, KeyError, RecursionError) as exc:
                print(f"{patch.stem}: import value check unavailable: {exc}",file=sys.stderr)
            selected = set(selected) | protected
        if selector:
            selected.update(selector.confirmed_failures)
        workers, cost = schedule([t for t in tests if t["number"] in selected])
        if selected != prior_selection:
            allowed = set(prior_selection) | selector.confirmed_failures | protected
            original_workers, original_cost = schedule([selector.by_number[n] for n in allowed])
            if original_cost < cost:
                workers, cost, selected = original_workers, original_cost, allowed
        plausible = set()
        if len(workers) > 1 and selector and len(selected) < len(tests):
            for change in changes:
                path = change["old"]
                if path in selector.source_cache:
                    plausible.update(selector.file_tests(path))
        extras = fill_spare_capacity(workers, selected, plausible, durations, zero_cost)
        print(
            json.dumps(
                {"commit": patch.stem, "workers": workers}, separators=(",", ":")
            )
        )
        print(
            f"{patch.stem}: tests={len(selected) + extras}, capacity_additions={extras}, workers={len(workers)}, simulated={cost / 1000:.3f}s",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
