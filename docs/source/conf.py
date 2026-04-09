"""Sphinx configuration for ZedProfiler docs."""

from pathlib import Path
from runpy import run_path

project = "ZedProfiler"
author = "Way Lab"

ROOT = Path(__file__).resolve().parents[2]
VERSION_FILE = ROOT / "zedprofiler" / "_version.py"
release = run_path(str(VERSION_FILE))["__version__"]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "furo"
html_static_path = ["_static"]
