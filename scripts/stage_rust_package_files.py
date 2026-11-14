"""Copy the shared README and license files that the aiwaf-rust build needs into rust/.

The copies are ignored by Git. The release workflow stages the same files before building wheels.
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUST = ROOT / "rust"


def main() -> int:
    for name in ("README.md", "LICENSE", "NOTICE"):
        shutil.copyfile(ROOT / name, RUST / name)
    shutil.copytree(ROOT / "licenses", RUST / "licenses", dirs_exist_ok=True)
    print(f"Staged package files in {RUST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
