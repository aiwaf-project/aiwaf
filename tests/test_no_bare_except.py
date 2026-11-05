"""Guard against bare ``except:`` clauses anywhere in the Python package.

Bare ``except:`` also catches ``KeyboardInterrupt`` and ``SystemExit``, so a
shutdown or Ctrl+C can be swallowed. The project has no linter that would flag
new ones, so this test parses every module under ``py/aiwaf`` with ``ast`` and
fails on any bare handler.
"""

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "py" / "aiwaf"


def _bare_except_locations():
    offenders = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                relative = path.relative_to(PACKAGE_ROOT.parent).as_posix()
                offenders.append(f"{relative}:{node.lineno}")
    return offenders


def test_no_bare_except_clauses():
    offenders = _bare_except_locations()
    assert offenders == [], (
        "bare `except:` clauses found; use `except Exception:` instead "
        "(bare except also catches KeyboardInterrupt and SystemExit): "
        + ", ".join(offenders)
    )
