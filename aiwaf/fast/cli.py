#!/usr/bin/env python3
"""AIWAF FastAPI CLI."""

from __future__ import annotations

import argparse
import importlib
import sys

from aiwaf.flask.cli import main as _flask_cli_main


def _load_fastapi_app(app_path):
    module_path, _, obj = app_path.partition(":")
    if not module_path or not obj:
        raise ValueError("Use --app module:app or module:create_app")
    module = importlib.import_module(module_path)
    target = getattr(module, obj, None)
    if target is None:
        raise ValueError(f"App not found: {app_path}")
    if callable(target) and not hasattr(target, "routes"):
        return target()
    return target


def _init(argv) -> None:
    parser = argparse.ArgumentParser(description="Generate AIWAF FastAPI path manifest")
    parser.add_argument("--app", required=True, help="FastAPI app import path (module:app or module:create_app)")
    parser.add_argument("--output", default=".aiwaf/paths.json", help="Output manifest path")
    args = parser.parse_args(argv)

    from aiwaf.fast.path_manifest import generate_fastapi_manifest

    app = _load_fastapi_app(args.app)
    manifest = generate_fastapi_manifest(app, args.output)
    print(f"Generated {args.output}")
    print(f"Framework: {manifest['framework']}")
    print(f"Routes: {len(manifest.get('routes', {}))}")
    print(f"Context hash: {manifest['context_hash']}")


def main() -> None:
    argv = list(sys.argv[1:])
    if argv and argv[0] == "init":
        _init(argv[1:])
        return
    _flask_cli_main()


if __name__ == "__main__":
    main()
