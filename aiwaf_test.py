#!/usr/bin/env python3
"""Run every AIWAF Python and JavaScript test suite from one command."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parent
TESTS = ROOT / "tests"


@dataclass(frozen=True)
class TestTask:
    name: str
    command: tuple[str, ...]


def _python_tasks(verbose: bool) -> list[TestTask]:
    root_tests = tuple(
        str(path.relative_to(ROOT)) for path in sorted(TESTS.glob("test_*.py"))
    )
    pytest_targets = root_tests + (
        "tests/core",
        "tests/flask",
        "tests/fast",
        "tests/integration",
    )

    return [
        TestTask(
            "Python / Django",
            (
                sys.executable,
                "manage.py",
                "test",
                "tests.django",
                "--verbosity",
                "2" if verbose else "1",
            ),
        ),
        TestTask(
            "Python / core, Flask, FastAPI, integration",
            (
                sys.executable,
                "-m",
                "pytest",
                *pytest_targets,
                "-vv" if verbose else "-q",
            ),
        ),
    ]


def _javascript_task(coverage: bool = False) -> TestTask:
    npm = shutil.which("npm")
    if npm is None:
        npm = "npm"
    script = "test:coverage" if coverage else "test"
    return TestTask("JavaScript / npm", (npm, "--prefix", "js", "run", script))


def _rust_tasks(verbose: bool) -> list[TestTask]:
    django_modules = (
        "tests.django.test_installed_rust_package_guard_django",
        "tests.django.test_rust_backend_chunked_api",
        "tests.django.test_rust_backend_integration",
        "tests.django.test_rust_backend_toggle",
        "tests.django.test_rust_backend_validate_headers",
        "tests.django.test_rust_feature_extraction",
    )
    pytest_targets = (
        "tests/core/test_training_features.py",
        "tests/fast/test_python_rust_contract.py",
        "tests/core/test_rust_backend.py",
        "tests/fast/test_rust_middleware_toggle.py",
        "tests/fast/test_trainer_rust_batch.py",
        "tests/flask/test_rust_backend.py",
        "tests/flask/test_rust_middleware_toggle.py",
        "tests/flask/test_trainer_rust_batch.py",
    )
    return [
        TestTask(
            "Python / Rust extension availability",
            (
                sys.executable,
                "-c",
                "import aiwaf_rust; print('aiwaf_rust:', aiwaf_rust.__file__)",
            ),
        ),
        TestTask(
            "Python / Django Rust adapter",
            (
                sys.executable,
                "manage.py",
                "test",
                *django_modules,
                "--verbosity",
                "2" if verbose else "1",
            ),
        ),
        TestTask(
            "Python / Flask and FastAPI Rust adapters",
            (
                sys.executable,
                "-m",
                "pytest",
                *pytest_targets,
                "-vv" if verbose else "-q",
            ),
        ),
    ]


def _wasm_task() -> TestTask:
    npm = shutil.which("npm") or "npm"
    return TestTask("JavaScript / real and mocked WASM adapters", (npm, "--prefix", "js", "run", "test:wasm"))


def _layout_task() -> TestTask:
    return TestTask(
        "Monorepo / source-to-test module layout",
        (
            sys.executable,
            "scripts/check_test_layout.py",
            "--baseline",
            "tests/test_layout_baseline.txt",
        ),
    )


def _run_task(task: TestTask) -> tuple[bool, float]:
    print(f"\n{'=' * 72}", flush=True)
    print(f"Running {task.name}", flush=True)
    print(" ".join(task.command), flush=True)
    print("=" * 72, flush=True)

    started = time.monotonic()
    try:
        result = subprocess.run(task.command, cwd=ROOT, check=False)
        passed = result.returncode == 0
    except OSError as exc:
        print(f"Unable to start {task.name}: {exc}", file=sys.stderr, flush=True)
        passed = False
    return passed, time.monotonic() - started


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete AIWAF monorepo test suite."
    )
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--python-only", action="store_true", help="Run only Python test suites."
    )
    scope.add_argument(
        "--js-only", action="store_true", help="Run only the JavaScript test suite."
    )
    scope.add_argument(
        "--rust-only", action="store_true", help="Run only the real Rust adapter suites."
    )
    scope.add_argument(
        "--wasm-only", action="store_true", help="Run only the real and mocked WASM suites."
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failing test group.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Use verbose Python test output."
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Collect Python and JavaScript coverage and print function coverage.",
    )
    parser.add_argument(
        "--strict-function-coverage",
        action="store_true",
        help="Fail if any production Python callable is not entered; implies --coverage.",
    )
    return parser.parse_args(argv)


def _with_python_coverage(task: TestTask) -> TestTask:
    if (
        not task.command
        or task.command[0] != sys.executable
        or task.name == "Python / Rust extension availability"
    ):
        return task
    return TestTask(
        task.name,
        (
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--append",
            "--branch",
            "--source=aiwaf",
            *task.command[1:],
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    coverage_enabled = args.coverage or args.strict_function_coverage
    tasks: list[TestTask] = []
    tasks.append(_layout_task())
    if args.rust_only:
        tasks.extend(_rust_tasks(args.verbose))
    elif args.wasm_only:
        tasks.append(_wasm_task())
    elif args.python_only:
        tasks.extend(_python_tasks(args.verbose))
        tasks.extend(_rust_tasks(args.verbose))
    elif args.js_only:
        tasks.append(_javascript_task(coverage_enabled))
        tasks.append(_wasm_task())
    else:
        tasks.extend(_python_tasks(args.verbose))
        tasks.extend(_rust_tasks(args.verbose))
        tasks.append(_javascript_task(coverage_enabled))
        tasks.append(_wasm_task())

    python_selected = not args.js_only and not args.wasm_only
    if coverage_enabled and python_selected:
        subprocess.run(
            (sys.executable, "-m", "coverage", "erase"), cwd=ROOT, check=True
        )
        tasks = [_with_python_coverage(task) for task in tasks]

    results: list[tuple[str, bool, float]] = []
    for task in tasks:
        passed, duration = _run_task(task)
        results.append((task.name, passed, duration))
        if not passed and args.fail_fast:
            break

    if coverage_enabled and python_selected and all(passed for _, passed, _ in results):
        coverage_report = TestTask(
            "Python / line and branch coverage",
            (sys.executable, "-m", "coverage", "report", "--skip-empty", "--fail-under=62"),
        )
        passed, duration = _run_task(coverage_report)
        results.append((coverage_report.name, passed, duration))

        audit_command = [
            sys.executable,
            "scripts/check_python_function_coverage.py",
        ]
        if not args.strict_function_coverage:
            audit_command.extend(
                (
                    "--report-only",
                    "--fail-under",
                    "75",
                    "--baseline",
                    "tests/function_coverage_baseline.txt",
                )
            )
        function_audit = TestTask("Python / callable coverage audit", tuple(audit_command))
        passed, duration = _run_task(function_audit)
        results.append((function_audit.name, passed, duration))

    print(f"\n{'=' * 72}")
    print("AIWAF monorepo test summary")
    print("=" * 72)
    for name, passed, duration in results:
        status = "PASS" if passed else "FAIL"
        print(f"{status:4}  {duration:7.2f}s  {name}")

    passed_count = sum(passed for _, passed, _ in results)
    print(f"\n{passed_count}/{len(results)} test groups passed")
    return 0 if passed_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
