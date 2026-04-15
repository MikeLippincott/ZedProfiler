"""Update the README coverage badge from a coverage.py XML report."""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

BADGE_PATTERN = re.compile(
    r"\[!\[Coverage\]\(https://img\.shields\.io/badge/coverage-[^)]+\)\]\(#quality-gates\)"
)
BRIGHTGREEN_THRESHOLD = 90
GREEN_THRESHOLD = 80
YELLOWGREEN_THRESHOLD = 70
YELLOW_THRESHOLD = 60
ORANGE_THRESHOLD = 50


def choose_color(percent: int) -> str:
    """Pick a simple badge color based on rounded percentage coverage."""
    if percent >= BRIGHTGREEN_THRESHOLD:
        return "brightgreen"
    if percent >= GREEN_THRESHOLD:
        return "green"
    if percent >= YELLOWGREEN_THRESHOLD:
        return "yellowgreen"
    if percent >= YELLOW_THRESHOLD:
        return "yellow"
    if percent >= ORANGE_THRESHOLD:
        return "orange"
    return "red"


def read_percent_from_coverage_xml(coverage_file: Path) -> int:
    """Parse overall line-rate from coverage XML and return rounded percent."""
    root = ET.parse(coverage_file).getroot()
    line_rate = root.attrib.get("line-rate")
    if line_rate is None:
        msg = f"Missing 'line-rate' attribute in {coverage_file}"
        raise ValueError(msg)

    return round(float(line_rate) * 100)


def update_readme_badge(readme_file: Path, percent: int) -> bool:
    """Update badge in README and return True when file content changed."""
    color = choose_color(percent)
    badge = (
        f"[![Coverage](https://img.shields.io/badge/coverage-{percent}%25-{color})]"
        "(#quality-gates)"
    )

    original = readme_file.read_text(encoding="utf-8")

    if BADGE_PATTERN.search(original):
        updated = BADGE_PATTERN.sub(badge, original, count=1)
    else:
        lines = original.splitlines()
        if not lines:
            msg = f"README file is empty: {readme_file}"
            raise ValueError(msg)
        lines.insert(1, "")
        lines.insert(2, badge)
        updated = "\n".join(lines)
        if original.endswith("\n"):
            updated += "\n"

    if updated == original:
        return False

    readme_file.write_text(updated, encoding="utf-8")
    return True


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coverage-file",
        default="coverage.xml",
        type=Path,
        help="Path to coverage.py XML report (default: coverage.xml)",
    )
    parser.add_argument(
        "--readme",
        default="README.md",
        type=Path,
        help="Path to README file (default: README.md)",
    )
    return parser.parse_args()


def main() -> int:
    """Update README badge and print a small status message."""
    args = parse_args()
    percent = read_percent_from_coverage_xml(args.coverage_file)
    changed = update_readme_badge(args.readme, percent)

    if changed:
        print(f"Updated coverage badge to {percent}% in {args.readme}")
    else:
        print(f"Coverage badge already up to date at {percent}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
