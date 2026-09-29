"""Shared helper for locating the approved VIX/VVIX CSV files.

Every script expects its approved CSV file in this repository's local
data/ folder (or one of a few alternative local paths it lists near the
top). There is no remote or cloud fallback of any kind: place the files
yourself, see data/README.md for the accepted schemas and official sources.
"""

from pathlib import Path

REPO_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def resolve_results_directory() -> Path:
    """Return this repository's local results/ directory."""
    return Path(__file__).resolve().parents[2] / "results"


def locate_approved_csv(local_paths: list[Path], name: str) -> Path:
    for path in local_paths:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Approved {name} CSV not found. Place it in one of the local candidate "
        "paths listed in this script, see data/README.md for accepted schemas "
        "and sources."
    )
