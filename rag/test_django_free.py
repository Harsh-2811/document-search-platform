"""Guard test for the package's one architectural rule.

`rag/` must import cleanly with no Django involvement. Checking that from
inside a Django test run would prove nothing — Django is already in
`sys.modules` by then — so the probe runs in a fresh interpreter.

Run standalone:  python -m unittest rag.test_django_free
"""

import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_PROBE = """
import importlib
import pkgutil
import sys

import rag

for module in pkgutil.walk_packages(rag.__path__, prefix="rag."):
    if module.name.startswith("rag.test"):
        continue
    importlib.import_module(module.name)

leaked = sorted(n for n in sys.modules if n == "django" or n.startswith("django."))
print(",".join(leaked))
"""


class DjangoFreeTests(unittest.TestCase):
    def test_rag_imports_without_django(self):
        result = subprocess.run(
            [sys.executable, "-c", _PROBE],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"importing rag/ submodules failed:\n{result.stderr}",
        )
        leaked = [name for name in result.stdout.strip().split(",") if name]
        self.assertEqual(
            leaked,
            [],
            msg=(
                "rag/ pulled in Django: "
                + ", ".join(leaked)
                + " — move the Django-dependent code into the documents app."
            ),
        )


if __name__ == "__main__":
    unittest.main()
