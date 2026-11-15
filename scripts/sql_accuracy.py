#!/usr/bin/env python3
"""Report how the SQL tokenizer does on the shared labelled cases.

Usage: python scripts/sql_accuracy.py

Exits 1 unless it catches every attack and flags no ordinary value.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))

from aiwaf.core.sql_injection import rust  # noqa: E402

classify = rust().policy_sql_tokenizer

CASES = ROOT / "tests" / "fixtures" / "sql_injection_cases.json"


def main(argv=None) -> int:
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    missed = [value for value in cases["attacks"] if not classify(value)]
    flagged = [value for value in cases["ordinary"] if classify(value)]
    attacks, ordinary = len(cases["attacks"]), len(cases["ordinary"])
    print(f"tokenizer: caught {attacks - len(missed)} of {attacks} attacks, flagged {len(flagged)} of {ordinary} ordinary values")
    for value in missed:
        print(f"  missed: {value!r}")
    for value in flagged:
        print(f"  flagged: {value!r}")
    return 1 if missed or flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
