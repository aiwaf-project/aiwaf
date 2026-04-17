#!/usr/bin/env python3
"""
Django management script for AIWAF testing
"""

import os
import sys

if __name__ == "__main__":
    # Add the project root to Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    
    # Set Django settings
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tests.django.test_settings")
    
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    
    # Django's default test discovery can pick up unrelated directories and create
    # confusing import attempts (e.g. `aiwaf.tests`, `aiwaf.aiwaf`) when run from
    # the repo root with no explicit labels. Default to running the Django test
    # package (`tests/django`) in that case.
    if len(sys.argv) >= 2 and sys.argv[1] == "test":
        has_label = any(arg and not arg.startswith("-") for arg in sys.argv[2:])
        if not has_label:
            sys.argv.append("tests.django")

    execute_from_command_line(sys.argv)
