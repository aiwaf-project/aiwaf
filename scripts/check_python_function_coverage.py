#!/usr/bin/env python3
"""Fail when a production Python callable was never entered by the test suite."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path

from coverage import CoverageData


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "py" / "aiwaf"


@dataclass(frozen=True)
class Function:
    path: Path
    name: str
    line: int
    body_lines: frozenset[int]


class FunctionVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.scope: list[str] = []
        self.functions: list[Function] = []

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qualified_name = ".".join((*self.scope, node.name))
        body_lines: set[int] = set()
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(
            body[0].value, ast.Constant
        ) and isinstance(body[0].value.value, str):
            body = body[1:]
        for statement in body:
            body_lines.update(range(statement.lineno, statement.end_lineno + 1))
        self.functions.append(
            Function(self.path, qualified_name, node.lineno, frozenset(body_lines))
        )
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()


def _source_functions() -> list[Function]:
    functions: list[Function] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        if (
            path.name.startswith("test_")
            or "example" in path.name.lower()
            or path.name.endswith("_EXAMPLE.py")
        ):
            continue
        visitor = FunctionVisitor(path)
        visitor.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        functions.extend(visitor.functions)
    return functions


def _coverage_by_path(data: CoverageData) -> dict[Path, set[int]]:
    measured: dict[Path, set[int]] = {}
    for filename in data.measured_files():
        path = Path(filename).resolve()
        measured[path] = set(data.lines(filename) or ())
    return measured


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", default=str(ROOT / ".coverage"))
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Print uncovered callables without returning a failure status.",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=0.0,
        metavar="PERCENT",
        help="Fail when entered-callable coverage is below this percentage.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Known uncovered callables; any newly uncovered callable fails.",
    )
    parser.add_argument(
        "--write-baseline",
        type=Path,
        help="Write the current uncovered callable list and exit.",
    )
    return parser.parse_args()


def _key(function: Function) -> str:
    return f"{function.path.relative_to(ROOT).as_posix()}::{function.name}"


def _read_baseline(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def main() -> int:
    args = _parse_args()
    data = CoverageData(basename=args.data_file)
    data.read()
    measured = _coverage_by_path(data)
    functions = _source_functions()
    uncovered = [
        function
        for function in functions
        if not function.body_lines.intersection(measured.get(function.path.resolve(), set()))
    ]
    uncovered_keys = {_key(function) for function in uncovered}

    if args.write_baseline:
        output = args.write_baseline
        if not output.is_absolute():
            output = ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Known uncovered Python callables.",
            "# Remove an entry when its test is added; new entries fail CI.",
            *sorted(uncovered_keys),
            "",
        ]
        output.write_text("\n".join(lines), encoding="utf-8")
        print(f"Wrote {len(uncovered_keys)} entries to {output.relative_to(ROOT)}")
        return 0

    covered_count = len(functions) - len(uncovered)
    percentage = covered_count / len(functions) * 100
    print(
        f"Python function coverage: {covered_count}/{len(functions)} callables entered "
        f"({percentage:.1f}%)"
    )
    for function in uncovered:
        relative = function.path.relative_to(ROOT)
        print(f"UNCOVERED {relative}:{function.line} {function.name}")

    if percentage < args.fail_under:
        print(
            f"Function coverage {percentage:.1f}% is below the "
            f"required {args.fail_under:.1f}%"
        )
        return 1
    if args.baseline:
        baseline_path = args.baseline
        if not baseline_path.is_absolute():
            baseline_path = ROOT / baseline_path
        baseline = _read_baseline(baseline_path)
        new_uncovered = uncovered_keys - baseline
        newly_covered = baseline - uncovered_keys
        if new_uncovered:
            for key in sorted(new_uncovered):
                print(f"NEW UNCOVERED {key}")
            return 1
        if newly_covered:
            for key in sorted(newly_covered):
                print(f"REMOVE COVERED BASELINE ENTRY {key}")
    return 0 if args.report_only or not uncovered else 1


if __name__ == "__main__":
    raise SystemExit(main())
