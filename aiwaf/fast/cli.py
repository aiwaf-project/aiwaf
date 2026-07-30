#!/usr/bin/env python3
"""AIWAF FastAPI CLI."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from aiwaf.flask.cli import main as _flask_cli_main


def _load_fastapi_app(app_path):
    module_path, _, obj = app_path.partition(":")
    if not module_path or not obj:
        raise ValueError("Use --app module:app or module:create_app")
    project_root = str(Path.cwd())
    while project_root in sys.path:
        sys.path.remove(project_root)
    sys.path.insert(0, project_root)
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


def _migrate_blacklist(argv) -> None:
    parser = argparse.ArgumentParser(description="Upgrade the configured FastAPI blacklist backend")
    parser.add_argument("--app", help="FastAPI app import path; loads its AIWAF storage configuration")
    parser.add_argument("--backend", choices=["csv", "file", "db", "memory"], default="csv")
    parser.add_argument("--storage-path", help="Storage file when --app is not supplied")
    args = parser.parse_args(argv)

    if args.app:
        _load_fastapi_app(args.app)
    else:
        from aiwaf.core.runtime_storage import initialize_storage

        kwargs = {}
        if args.storage_path:
            kwargs["file_path"] = args.storage_path
        initialize_storage(args.backend, **kwargs)

    from aiwaf.core.blacklist_migration import migrate_runtime_storage
    from aiwaf.core.runtime_storage import get_storage

    total, changed = migrate_runtime_storage(get_storage())
    print(f"FastAPI blacklist upgraded: {changed}/{total} legacy entries updated")


def main() -> None:
    argv = list(sys.argv[1:])
    if argv and argv[0] == "init":
        _init(argv[1:])
        return
    if len(argv) >= 2 and argv[:2] == ["blacklist", "migrate"]:
        _migrate_blacklist(argv[2:])
        return
    _flask_cli_main()


if __name__ == "__main__":
    main()
