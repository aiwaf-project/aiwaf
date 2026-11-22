"""Java's RceCore.java is generated from aiwaf-rust's rce.rs and must match it."""
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sync_rce_rules.py"
spec = importlib.util.spec_from_file_location("sync_rce_rules", SCRIPT)
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)


def test_java_rules_match_the_rust_rules():
    assert sync.main(["--check"]) == 0, "run python scripts/sync_rce_rules.py"


def test_every_rust_rule_reaches_java():
    parts = sync.read_rust(sync.RUST.read_text(encoding="utf-8"))
    java = sync.JAVA.read_text(encoding="utf-8")
    assert len(parts["rules"]) >= 9
    for name, _pattern in parts["rules"]:
        assert f'Map.entry("{name}"' in java
    assert ("dollar", "$") in parts["entities"] and ("bsol", "\\") in parts["entities"]


def test_a_stale_copy_is_reported(tmp_path, monkeypatch, capsys):
    stale = tmp_path / "RceCore.java"
    stale.write_text("// old\n", encoding="utf-8")
    monkeypatch.setattr(sync, "JAVA", stale)
    assert sync.main(["--check"]) == 1
    assert "out of date" in capsys.readouterr().err
    assert sync.main([]) == 0
    assert sync.main(["--check"]) == 0
