"""
Tests for the in-app Help & Docs loader (src/ui/utils/docs.py).

The helper is Streamlit-free so it can be exercised without a running app.
"""

import pytest

from src.ui.utils.docs import (
    docs_algorithms_dir,
    list_algo_docs,
    read_algo_doc,
    get_docs_index,
)

EXPECTED_SLUGS = {
    "aliasing",
    "anova_analysis",
    "fractional_factorial",
    "full_factorial",
    "latin_hypercube",
    "optimal_design",
    "optimization",
    "response_surface",
    "split_plot",
}


class TestDocsDir:
    def test_docs_dir_resolves_to_docs_algorithms(self):
        path = docs_algorithms_dir()
        assert path.name == "algorithms"
        assert path.parent.name == "docs"
        assert (path / "response_surface.md").is_file()


class TestListDocs:
    def test_lists_all_known_slugs(self):
        slugs = {slug for slug, _ in list_algo_docs()}
        assert slugs == EXPECTED_SLUGS

    def test_every_entry_has_a_nonempty_title(self):
        for slug, title in list_algo_docs():
            assert title, f"doc {slug} has no # Heading"
            assert title.startswith("# ") is False


class TestReadDoc:
    def test_reads_content_for_every_slug(self):
        for slug in EXPECTED_SLUGS:
            content = read_algo_doc(slug)
            assert len(content.strip()) > 0, f"doc {slug} is empty"
            assert content.lstrip().startswith("#"), f"doc {slug} lacks a heading"

    def test_unknown_slug_raises_key_error(self):
        with pytest.raises(KeyError):
            read_algo_doc("does_not_exist")

    def test_get_docs_index_covers_all_slugs(self):
        index = get_docs_index()
        assert set(index.keys()) == EXPECTED_SLUGS