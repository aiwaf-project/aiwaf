#!/usr/bin/env python3
"""Enforce canonical test-module names for Python and JavaScript sources."""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PY_SOURCE = ROOT / "py" / "aiwaf"
PY_TESTS = ROOT / "tests"
JS_SOURCE = ROOT / "js" / "lib"
JS_TESTS = ROOT / "js" / "test"


def _has_functions(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.walk(tree))


def _python_test_path(source: Path) -> Path:
    relative = source.relative_to(PY_SOURCE)
    parts = relative.parts
    filename = f"test_{source.stem}.py"
    if len(parts) == 1:
        return PY_TESTS / filename
    adapter = parts[0]
    if adapter == "core":
        return PY_TESTS / "core" / filename
    if adapter in {"django", "fast", "flask"}:
        if adapter == "django" and len(parts) > 2:
            return PY_TESTS / adapter / Path(*parts[1:-1]) / filename
        return PY_TESTS / adapter / filename
    return PY_TESTS / Path(*parts[:-1]) / filename


def _normalize_js_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _missing_python() -> set[str]:
    missing: set[str] = set()
    for source in sorted(PY_SOURCE.rglob("*.py")):
        if (
            source.name == "__init__.py"
            or source.name.startswith("test_")
            or "example" in source.name.lower()
            or not _has_functions(source)
        ):
            continue
        expected = _python_test_path(source)
        if not expected.is_file():
            missing.add(
                f"{source.relative_to(ROOT).as_posix()} -> "
                f"{expected.relative_to(ROOT).as_posix()}"
            )
    return missing


def _missing_javascript() -> set[str]:
    existing = {
        _normalize_js_name(path.name.removesuffix(".test.js"))
        for path in JS_TESTS.glob("*.test.js")
    }
    missing: set[str] = set()
    for source in sorted(JS_SOURCE.glob("*.js")):
        if _normalize_js_name(source.stem) not in existing:
            expected = JS_TESTS / f"{source.stem}.test.js"
            missing.add(
                f"{source.relative_to(ROOT).as_posix()} -> "
                f"{expected.relative_to(ROOT).as_posix()}"
            )
    return missing


def _read_baseline(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--write-baseline", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    missing = _missing_python() | _missing_javascript()

    if args.write_baseline:
        path = args.write_baseline
        if not path.is_absolute():
            path = ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(
                (
                    "# Source modules that still need canonically named test modules.",
                    "# New entries fail CI; remove entries as test modules are normalized.",
                    *sorted(missing),
                    "",
                )
            ),
            encoding="utf-8",
        )
        print(f"Wrote {len(missing)} entries to {path.relative_to(ROOT)}")
        return 0

    if not args.baseline:
        print(f"Test layout: {len(missing)} source modules lack canonical test modules")
        for entry in sorted(missing):
            print(f"MISSING {entry}")
        return 1 if missing else 0
    baseline_path = args.baseline
    if not baseline_path.is_absolute():
        baseline_path = ROOT / baseline_path
    baseline = _read_baseline(baseline_path)
    new_missing = missing - baseline
    resolved = baseline - missing
    print(
        f"Test layout: {len(missing)} known missing canonical modules, "
        f"{len(new_missing)} new"
    )
    for entry in sorted(new_missing):
        print(f"NEW MISSING TEST MODULE {entry}")
    for entry in sorted(resolved):
        print(f"REMOVE RESOLVED BASELINE ENTRY {entry}")
    return 1 if new_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
