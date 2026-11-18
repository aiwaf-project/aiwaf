"""The OWASP Benchmark scorer maps findings to test cases and scores them per category."""
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_benchmark.py"
spec = importlib.util.spec_from_file_location("audit_benchmark", SCRIPT)
audit_benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_benchmark)


def test_scores_found_and_false_alarms_per_category(tmp_path, capsys):
    expected = tmp_path / "expectedresults-1.0.csv"
    expected.write_text(
        "# test name, category, real vulnerability, cwe\n"
        "BenchmarkTest00001,sqli,true,89\nBenchmarkTest00002,sqli,false,89\n"
        "BenchmarkTest00003,xss,true,79\nBenchmarkTest00004,hash,true,328\n")
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"findings": [
        {"file": "src/BenchmarkTest00001.java", "sink": "sql"},
        {"file": "src/Helper.java", "entry_file": "src/BenchmarkTest00002.java", "sink": "sql"},
        {"file": "src/BenchmarkTest00003.java", "sink": "path"},
    ]}))
    assert audit_benchmark.main(["--source", str(tmp_path), "--report", str(report), "--json"]) == 0
    table = json.loads(capsys.readouterr().out)
    assert table["sqli"] == {"tp": 1, "fn": 0, "fp": 1, "tn": 0, "tpr": 1.0, "fpr": 1.0, "score": 0.0}
    # A finding of another kind doesn't count, and categories that aren't injection are left out.
    assert table["xss"]["fn"] == 1
    assert "hash" not in table
    assert audit_benchmark.main(["--source", str(tmp_path), "--report", str(report)]) == 0
    assert "sqli" in capsys.readouterr().out
