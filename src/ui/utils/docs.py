"""
Documentation loading helpers (pure, Streamlit-free).

Reads the algorithm reference docs under ``docs/algorithms/`` so they can be
rendered inside the app as an in-app Help & Docs page.  Mirrors the
``response_definitions.py`` pattern: no Streamlit dependency so the helpers
are unit-testable directly.
"""

from pathlib import Path
from typing import Dict, List, Tuple

_DOCS_DIR = Path(__file__).resolve().parents[3] / "docs" / "algorithms"


def docs_algorithms_dir() -> Path:
    """
    Absolute path to the algorithm documentation directory.

    Resolved relative to this module's own location (4 levels up from
    ``src/ui/utils/``), so it works both in the source tree and inside the
    packaged desktop build (where ``docs/`` sits next to ``src/``).
    """
    return _DOCS_DIR


def list_algo_docs() -> List[Tuple[str, str]]:
    """
    List available algorithm docs as ``(slug, title)`` pairs.

    Slugs are the markdown filenames without extension; titles are the
    first-level heading of each file (falling back to the slug when the file
    has no ``# Heading``).
    """
    if not _DOCS_DIR.is_dir():
        return []

    entries = []
    for path in sorted(_DOCS_DIR.glob("*.md")):
        entries.append((path.stem, _title_for(path)))

    return entries


def _title_for(path: Path) -> str:
    """Extract the ``# Title`` heading from a doc file."""
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                return line[2:].strip()
    except OSError:
        pass
    return path.stem


def read_algo_doc(slug: str) -> str:
    """
    Return the markdown contents of an algorithm doc.

    Raises
    ------
    KeyError
        If ``slug`` does not correspond to a known doc file.
    """
    docs_dir = _DOCS_DIR
    if not docs_dir.is_dir():
        raise KeyError(
            f"Unknown documentation topic '{slug}' (docs directory not found)"
        )

    path = docs_dir / f"{slug}.md"
    if not path.is_file():
        raise KeyError(f"Unknown documentation topic '{slug}'")

    return path.read_text(encoding="utf-8")


def get_docs_index() -> Dict[str, str]:
    """
    Map of doc slug → markdown content for all algorithm docs.

    Convenience for tests and non-UI callers that want the whole set at once.
    """
    return {slug: read_algo_doc(slug) for slug, _ in list_algo_docs()}