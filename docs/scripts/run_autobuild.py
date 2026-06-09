#!/usr/bin/env python3
"""Wrapper to run sphinx-autobuild with a graceful message if missing."""

import importlib
import subprocess
import sys

try:
    importlib.import_module("sphinx_autobuild")
except Exception:
    print(
        "sphinx-autobuild not installed. Install with: pip install sphinx-autobuild",
        file=sys.stderr,
    )
    sys.exit(1)

subprocess.check_call(
    ["sphinx-autobuild", "src", "build", "--port", "8000", "--open-browser"]
)
