import importlib.util
from pathlib import Path

import pytest

# Load the module relative to this test file
_spec = importlib.util.spec_from_file_location(
    "update_coverage_badge",
    Path(__file__).parent.parent / "scripts" / "update_coverage_badge.py",
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_replaces_existing_badge(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    original = (
        "# Project\n\n"
        "[![Coverage](https://img.shields.io/badge/coverage-42%25-red)](#quality-gates)\n\n"
        "Some text\n"
    )
    readme.write_text(original, encoding="utf-8")

    changed = mod.update_readme_badge(readme, 85)
    assert changed is True

    updated = readme.read_text(encoding="utf-8")
    assert "coverage-85%25-green" in updated
    assert updated.count("[![Coverage]") == 1
    assert "coverage-42%25-red" not in updated


def test_inserts_badge_when_missing(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    original = "# Project\nDescription line\n"
    readme.write_text(original, encoding="utf-8")

    changed = mod.update_readme_badge(readme, 60)
    assert changed is True

    updated = readme.read_text(encoding="utf-8")
    lines = updated.splitlines()
    # After insertion: title at 0, blank line at 1, badge at 2
    assert lines[0] == "# Project"
    assert lines[1] == ""
    assert lines[2].startswith("[![Coverage]")
    assert "coverage-60%25-yellow" in lines[2]


def test_preserves_trailing_newline_on_insert(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    original = "# Project\nDescription line\n"  # ends with newline
    readme.write_text(original, encoding="utf-8")

    changed = mod.update_readme_badge(readme, 70)
    assert changed is True

    updated = readme.read_text(encoding="utf-8")
    assert updated.endswith("\n")


def test_no_change_returns_false_when_badge_already_exact(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    # percent 75 -> color "yellowgreen" (per thresholds)
    badge = "[![Coverage](https://img.shields.io/badge/coverage-75%25-yellowgreen)](#quality-gates)"
    original = f"# Project\n\n{badge}\n\nMore\n"
    readme.write_text(original, encoding="utf-8")

    changed = mod.update_readme_badge(readme, 75)
    assert changed is False
    assert readme.read_text(encoding="utf-8") == original


def test_empty_readme_raises_value_error(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        mod.update_readme_badge(readme, 50)
