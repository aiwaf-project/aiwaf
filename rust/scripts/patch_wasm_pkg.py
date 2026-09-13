#!/usr/bin/env python3
"""Patch wasm-pack pkg/package.json and copy README.md for npm."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


KEYWORDS = [
    "wasm",
    "waf",
    "security",
    "ai",
    "anomaly-detection",
    "isolation-forest",
    "rust",
    "webassembly",
    "typescript",
]


def main(argv: list[str]) -> int:
    pkg_dir = Path("crates/aiwaf_wasm/pkg")
    src_readme = Path("crates/aiwaf_wasm/README.md")
    pkg_json = pkg_dir / "package.json"

    if not pkg_json.exists():
        print(f"missing {pkg_json}", file=sys.stderr)
        return 1

    data = json.loads(pkg_json.read_text(encoding="utf-8"))
    data["keywords"] = KEYWORDS
    data.setdefault("license", "MIT")
    data.setdefault("repository", {
        "type": "git",
        "url": "https://github.com/aiwaf/aiwaf-rust",
    })
    data.setdefault("homepage", "https://github.com/aiwaf/aiwaf-rust")
    data["readme"] = "README.md"

    pkg_json.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    if src_readme.exists():
        shutil.copyfile(src_readme, pkg_dir / "README.md")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
