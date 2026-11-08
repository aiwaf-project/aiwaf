#!/usr/bin/env python3
"""Seed each fuzz target's corpus from the shared case files, so fuzzing starts from real inputs."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
CORPUS = Path(__file__).resolve().parent / "corpus"

values = []
cases = json.loads((FIXTURES / "sql_injection_cases.json").read_text(encoding="utf-8"))
values += cases["attacks"] + cases["ordinary"]
values += json.loads((FIXTURES / "ordinary_text.json").read_text(encoding="utf-8"))["values"]
values += [case["value"] for case in json.loads((FIXTURES / "route_risk_cases.json").read_text(encoding="utf-8"))["rules"]]

for target in ("sql_tokenizer", "request_walker", "route_checks"):
    folder = CORPUS / target
    folder.mkdir(parents=True, exist_ok=True)
    for value in values:
        data = value.encode("utf-8")
        if target == "request_walker":
            data = b"\x80" + b"q=" + data + json.dumps({"field": value}).encode("utf-8")
        elif target == "route_checks":
            data = "/items/<id>\x01/items/7\x01q=1\x01".encode("utf-8") + data
        (folder / hashlib.sha1(data).hexdigest()).write_bytes(data)
print(f"Seeded {len(values)} inputs for each target in {CORPUS}")
