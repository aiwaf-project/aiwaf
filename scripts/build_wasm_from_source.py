#!/usr/bin/env python3
"""Build aiwaf-wasm from this repository and install it for the Node.js package.

Node.js needs aiwaf-wasm for its request checks, so tests must run against the WASM built
from this checkout rather than the copy npm installed from the registry. This builds the
bundler package wasm-pack publishes, stages the shared package files, and replaces
js/node_modules/aiwaf-wasm with it.

Usage: python scripts/build_wasm_from_source.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CRATE = ROOT / "rust" / "crates" / "aiwaf_wasm"
PACKAGE = CRATE / "pkg"
INSTALLED = ROOT / "js" / "node_modules" / "aiwaf-wasm"
NEEDED = ("sql_inspect", "route_rule", "sql_tokenizer", "RouteRiskChecker")


def main() -> int:
    wasm_pack = shutil.which("wasm-pack")
    if wasm_pack is None:
        print("wasm-pack is required to build aiwaf-wasm. Install it with: cargo install wasm-pack --locked", file=sys.stderr)
        return 1
    if not (ROOT / "js" / "node_modules").is_dir():
        print("Install the JavaScript dependencies first with: npm ci --prefix js", file=sys.stderr)
        return 1
    subprocess.run([wasm_pack, "build", "--release", "--target", "bundler"], cwd=CRATE, check=True)
    subprocess.run([sys.executable, "scripts/patch_wasm_pkg.py"], cwd=ROOT / "rust", check=True)
    exports = (PACKAGE / "aiwaf_wasm.js").read_text(encoding="utf-8")
    missing = [name for name in NEEDED if name not in exports]
    if missing:
        print(f"The aiwaf-wasm build is missing {', '.join(missing)}", file=sys.stderr)
        return 1
    if INSTALLED.exists():
        shutil.rmtree(INSTALLED)
    shutil.copytree(PACKAGE, INSTALLED, ignore=shutil.ignore_patterns(".gitignore"))
    print(f"Installed aiwaf-wasm from {PACKAGE.relative_to(ROOT)} into {INSTALLED.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
