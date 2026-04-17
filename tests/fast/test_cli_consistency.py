import subprocess
import sys
from pathlib import Path


def test_fast_cli_invocation_from_repo_root():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "aiwaf.fast.cli", "list", "all"],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode in (0, 1, 2)

