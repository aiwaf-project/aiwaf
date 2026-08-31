from unittest import TestCase

from aiwaf.django.management.commands.add_pathexemption import Command


class TestAddPathExemptionCommand(TestCase):
    def test_contract(self):
        self.assertIn("path", Command().help.lower())
