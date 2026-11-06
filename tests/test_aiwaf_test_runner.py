"""The test runner builds aiwaf-rust from this repository before Python tests."""
import sys

import pytest

import aiwaf_test


@pytest.mark.parametrize("argv, expected", [
    ([], True),
    (["--python-only"], True),
    (["--rust-only"], True),
    (["--js-only"], False),
    (["--wasm-only"], False),
    (["--no-rust-build"], False),
    (["--python-only", "--no-rust-build"], False),
])
def test_rust_is_built_only_when_python_tests_need_it(argv, expected):
    assert aiwaf_test._needs_rust_build(aiwaf_test._parse_args(argv)) is expected


def test_missing_cargo_stops_before_any_tests_run(monkeypatch, capsys):
    monkeypatch.setattr(aiwaf_test.shutil, "which", lambda name: None)
    ran = []
    monkeypatch.setattr(aiwaf_test, "_run_task", lambda task: ran.append(task) or (True, 0.0))

    assert aiwaf_test.main(["--python-only"]) == 1

    assert ran == []
    assert "cargo" in capsys.readouterr().err


def test_system_python_is_rejected_with_guidance(monkeypatch):
    monkeypatch.setattr(aiwaf_test.shutil, "which", lambda name: "cargo")
    monkeypatch.setattr(sys, "base_prefix", sys.prefix)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)

    assert "virtual environment" in aiwaf_test._rust_build_problem()


def test_conda_environments_can_build(monkeypatch):
    monkeypatch.setattr(aiwaf_test.shutil, "which", lambda name: "cargo")
    monkeypatch.setattr(sys, "base_prefix", sys.prefix)
    monkeypatch.setenv("CONDA_PREFIX", sys.prefix)

    assert aiwaf_test._rust_build_problem() is None


def test_failed_build_stops_before_any_tests_run(monkeypatch):
    monkeypatch.setattr(aiwaf_test, "_rust_build_problem", lambda: None)
    ran = []

    def fail_build(task):
        ran.append(task.name)
        return False, 0.0

    monkeypatch.setattr(aiwaf_test, "_run_task", fail_build)

    assert aiwaf_test.main(["--python-only"]) == 1
    assert len(ran) == 1
    assert ran[0].startswith("Rust /")


def test_build_runs_before_tests_and_verifies_the_source(monkeypatch):
    monkeypatch.setattr(aiwaf_test, "_rust_build_problem", lambda: None)
    monkeypatch.setattr(aiwaf_test.importlib.util, "find_spec", lambda name: object())
    ran = []
    monkeypatch.setattr(aiwaf_test, "_run_task", lambda task: ran.append(task) or (True, 0.0))

    assert aiwaf_test.main(["--rust-only"]) == 0

    assert ran[0].name == "Rust / stage package files"
    assert ran[1].name == "Rust / build aiwaf-rust from source"
    assert ran[1].command[1:5] == ("-m", "maturin", "develop", "--release")
    assert ran[2].name == "Rust / check aiwaf-rust came from this repository"
    assert all(not task.name.startswith("Rust /") for task in ran[3:])
    assert len(ran) > 3


def test_missing_maturin_is_installed_first(monkeypatch):
    monkeypatch.setattr(aiwaf_test.importlib.util, "find_spec", lambda name: None)
    tasks = aiwaf_test._rust_build_tasks()
    assert tasks[0].command[-1] == aiwaf_test.MATURIN_REQUIREMENT
    assert tasks[2].name == "Rust / build aiwaf-rust from source"


def test_no_rust_build_skips_the_build(monkeypatch):
    ran = []
    monkeypatch.setattr(aiwaf_test, "_run_task", lambda task: ran.append(task) or (True, 0.0))

    assert aiwaf_test.main(["--rust-only", "--no-rust-build"]) == 0

    assert all(not task.name.startswith("Rust /") for task in ran)
