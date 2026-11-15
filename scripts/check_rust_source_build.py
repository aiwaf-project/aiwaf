"""Fail unless the installed aiwaf-rust package was built from this repository's rust/ folder."""
import json
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname


def installed_source() -> Path | None:
    """Return the local folder aiwaf-rust was installed from, or None for a registry install."""
    metadata = distribution("aiwaf-rust").read_text("direct_url.json")
    if not metadata:
        return None
    url = urlparse(json.loads(metadata).get("url", ""))
    if url.scheme != "file":
        return None
    return Path(url2pathname(unquote(url.path))).resolve()


def main() -> int:
    expected = (Path(__file__).resolve().parents[1] / "rust").resolve()
    try:
        source = installed_source()
    except PackageNotFoundError:
        print("aiwaf-rust is not installed. Build it with: cd rust && maturin develop --release --locked")
        return 1
    if source != expected:
        print(f"aiwaf-rust was installed from {source or 'a package registry'}, expected {expected}.")
        print("Build it with: cd rust && maturin develop --release --locked")
        return 1
    print(f"aiwaf-rust built from {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
