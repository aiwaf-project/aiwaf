from unittest import TestCase

from aiwaf.django.management.commands.aiwaf_pathshell import Command


class TestAiwafPathShellCommand(TestCase):
    def test_contract(self):
        self.assertIn("path", Command().help.lower())
