"""Django umbrella command for AIWAF workflows."""

from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "AIWAF management workflows"

    def add_arguments(self, parser):
        subparsers = parser.add_subparsers(dest="subcommand")
        init_parser = subparsers.add_parser("init", help="Generate .aiwaf/paths.json from Django URLConf")
        init_parser.add_argument("--output", default=".aiwaf/paths.json", help="Output manifest path")

    def handle(self, *args, **options):
        if options.get("subcommand") != "init":
            raise CommandError("Usage: python manage.py aiwaf init [--output .aiwaf/paths.json]")

        from aiwaf.django.path_manifest import generate_django_manifest

        output = options.get("output") or ".aiwaf/paths.json"
        manifest = generate_django_manifest(output)
        self.stdout.write(self.style.SUCCESS(f"Generated {output}"))
        self.stdout.write(f"Framework: {manifest['framework']}")
        self.stdout.write(f"Routes: {len(manifest.get('routes', {}))}")
        self.stdout.write(f"Context hash: {manifest['context_hash']}")
