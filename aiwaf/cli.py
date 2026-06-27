from __future__ import annotations

import sys
from typing import List, Optional


_FRAMEWORK_MODULES = {
    "django": "django",
    "flask": "flask",
    "fastapi": "fastapi",
}


def _normalize_framework(framework: Optional[str]) -> Optional[str]:
    if framework == "fast":
        return "fastapi"
    return framework


def _installed_frameworks() -> List[str]:
    import importlib.util

    return [
        framework
        for framework, module_name in _FRAMEWORK_MODULES.items()
        if importlib.util.find_spec(module_name) is not None
    ]


def _infer_framework_from_app(app) -> Optional[str]:
    if hasattr(app, "url_map"):
        return "flask"
    if hasattr(app, "routes"):
        return "fastapi"
    return None


def _detect_framework(explicit_framework: Optional[str], app=None) -> str:
    framework = _normalize_framework(explicit_framework)
    if framework:
        return framework

    if app is not None:
        framework = _infer_framework_from_app(app)
        if framework:
            return framework

    installed = _installed_frameworks()
    if len(installed) == 1:
        return installed[0]
    if not installed:
        raise SystemExit(
            "Could not detect a supported framework. Install Django, Flask, or FastAPI, "
            "or pass --framework."
        )
    raise SystemExit(
        "Multiple supported frameworks are installed: "
        + ", ".join(installed)
        + ". Pass --framework or --app so AIWAF can choose the correct adapter."
    )


def _handle_init(argv) -> None:
    import argparse
    import importlib

    parser = argparse.ArgumentParser(description="Generate .aiwaf/paths.json")
    parser.add_argument("--framework", choices=["flask", "fastapi", "fast", "django"])
    parser.add_argument("--app", help="App import path for Flask/FastAPI (module:app or module:create_app)")
    parser.add_argument("--output", default=".aiwaf/paths.json", help="Output manifest path")
    args = parser.parse_args(argv)

    framework = _normalize_framework(args.framework)
    app = None
    if args.app:
        module_path, _, obj = args.app.partition(":")
        if not module_path or not obj:
            parser.error("--app must be module:app or module:create_app")
        module = importlib.import_module(module_path)
        target = getattr(module, obj, None)
        if target is None:
            raise SystemExit(f"App not found: {args.app}")
        inferred_framework = framework or _infer_framework_from_app(target)
        if inferred_framework == "flask":
            app = target() if callable(target) and not hasattr(target, "url_map") else target
        elif inferred_framework == "fastapi":
            app = target() if callable(target) and not hasattr(target, "routes") else target
        else:
            app = target() if callable(target) else target
        framework = framework or _infer_framework_from_app(app)

    framework = _detect_framework(framework, app)
    if framework == "django":
        from aiwaf.django.path_manifest import generate_django_manifest

        manifest = generate_django_manifest(args.output)
    else:
        if app is None:
            parser.error("--app is required for Flask/FastAPI")
        if framework == "flask":
            from aiwaf.flask.path_manifest import generate_flask_manifest

            manifest = generate_flask_manifest(app, args.output)
        else:
            from aiwaf.fast.path_manifest import generate_fastapi_manifest

            manifest = generate_fastapi_manifest(app, args.output)

    print(f"Generated {args.output}")
    print(f"Framework: {manifest['framework']}")
    print(f"Routes: {len(manifest.get('routes', {}))}")
    print(f"Context hash: {manifest['context_hash']}")


def aiwaf_detect() -> None:
    try:
        from aiwaf.django.trainer import train
    except Exception as exc:
        sys.stderr.write(
            "aiwaf-detect requires Django integration. Install with:\n"
            "  pip install aiwaf[django]\n"
        )
        raise SystemExit(1) from exc

    train()


def main() -> None:
    """Top-level AIWAF CLI entrypoint.

    Supports both styles:
    - `aiwaf <command> ...` (unified default command set)
    - `aiwaf <framework> <command> ...` where framework is flask|fast|django
    """
    frameworks = {"django", "flask", "fast"}
    argv = list(sys.argv[1:])

    if argv and argv[0] == "init":
        _handle_init(argv[1:])
        return

    if argv and argv[0] in frameworks:
        framework = argv[0]
        framework_args = argv[1:]
    else:
        # Unified shorthand: route bare commands through the shared Fast/Flask CLI.
        framework = "fast"
        framework_args = argv

    if framework == "flask":
        from aiwaf.flask.cli import main as flask_main

        sys.argv = ["aiwaf flask"] + list(framework_args)
        flask_main()
        return
    if framework == "fast":
        from aiwaf.fast.cli import main as fast_main

        sys.argv = ["aiwaf fast"] + list(framework_args)
        fast_main()
        return
    if framework == "django":
        try:
            from django.core.management import execute_from_command_line
        except Exception as exc:
            sys.stderr.write(
                "aiwaf django requires Django integration. Install with:\n"
                "  pip install aiwaf[django]\n"
            )
            raise SystemExit(1) from exc

        if not framework_args:
            sys.stderr.write(
                "Usage: aiwaf django <management-command> [args]\n"
                "Example: aiwaf django aiwaf_list --all\n"
            )
            raise SystemExit(2)

        execute_from_command_line(["aiwaf django"] + list(framework_args))
        return


if __name__ == "__main__":
    main()
