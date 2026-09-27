"""Tests for project load navigation.

Loading a project used to have two independent defects in
``src.ui.components.sidebar.add_project_load``:

1. Nothing ever called ``st.switch_page``. ``load_project_file`` only set
   ``st.session_state['current_step']`` and printed a string, so the
   "Navigating to data import..." message was aspirational and the app stayed
   on the main screen.
2. The uploader kept a fixed ``key="project_uploader"`` and was followed by an
   unconditional ``st.rerun()``. Streamlit keeps an uploaded file in session
   state across reruns, so the load re-entered on every rerun and the success
   message repeated forever.

``load_project_file`` is exercised through a real Streamlit run rather than
called directly: outside a script run ``st.session_state`` is shared process
state, so one test's restored project leaks into the next and the destination
branch is whichever ran last.
"""

import json
import os
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

PAGES = Path(__file__).parents[1] / "src/ui/pages"

_HARNESS = """
import os
import streamlit as st
from src.ui.utils.state_management import load_project_file

with open(os.environ["LOADFIX_PROJECT"], encoding="utf-8") as fh:
    content = fh.read()
st.session_state["_destination"] = load_project_file(content)
"""


def _factor(name="Temp", factor_type="continuous", levels=None):
    return {
        "name": name,
        "type": factor_type,
        "changeability": "easy",
        "levels": levels if levels is not None else [150.0, 200.0],
        "units": "C",
    }


def _project(factors=None, **extra):
    """Minimal but structurally faithful .doeproject payload."""
    payload = {
        "version": "0.3.0",
        "factors": [_factor()] if factors is None else factors,
        "design_type": None,
        "design_config": {},
        "design_metadata": {},
    }
    payload.update(extra)
    return json.dumps(payload)


_DESIGN_ROWS = [{"RunOrder": 1, "Temp": 150.0}, {"RunOrder": 2, "Temp": 200.0}]


@pytest.fixture
def load_project(tmp_path, monkeypatch):
    """Run ``load_project_file`` in a real script run and report what happened."""

    def _load(content):
        project = tmp_path / "p.doeproject"
        project.write_text(content, encoding="utf-8")
        monkeypatch.setenv("LOADFIX_PROJECT", str(project))
        app = AppTest.from_string(_HARNESS, default_timeout=30).run()
        assert not app.exception, [e.value for e in app.exception]
        return app

    return _load


class TestLoadDestination:
    """``load_project_file`` reports where the project is ready to go.

    The returned step is the one *after* the deepest step whose data was
    restored, so a project carrying a design but no results lands on Import
    Results rather than on Preview Design, where that design already sits.
    """

    def test_design_without_responses_goes_to_import_results(self, load_project):
        app = load_project(_project(design=_DESIGN_ROWS))
        assert app.session_state["_destination"] == 5

    def test_design_with_responses_goes_to_analyze(self, load_project):
        app = load_project(
            _project(
                design=_DESIGN_ROWS,
                responses={"Yield": [80.0, 85.0]},
                response_names=["Yield"],
            )
        )
        assert app.session_state["_destination"] == 6

    def test_design_type_only_goes_to_preview(self, load_project):
        app = load_project(_project(design_type="full_factorial"))
        assert app.session_state["_destination"] == 4

    def test_factors_only_goes_to_select_model(self, load_project):
        """Factors only means model selection is still outstanding.

        The destination is the step after the restored data, so factors-only
        lands on Select Model (2). It used to be 3, which jumped past model
        selection to Choose Design.
        """
        app = load_project(_project())
        assert app.session_state["_destination"] == 2

    def test_current_step_records_the_data_not_the_destination(self, load_project):
        """The highlight follows the restored data, the navigation does not.

        This is the pair that used to contradict each other: the message said
        "Navigating to data import" while current_step was set to 4, which is
        Preview Design.
        """
        app = load_project(_project(design=_DESIGN_ROWS))
        assert app.session_state["current_step"] == 4
        assert app.session_state["_destination"] == 5

    def test_invalid_json_returns_none(self, load_project):
        app = load_project("{not json")
        assert app.session_state["_destination"] is None
        assert any("Invalid project file" in e.value for e in app.error)

    def test_no_valid_factors_returns_none(self, load_project):
        app = load_project(_project(factors=[]))
        assert app.session_state["_destination"] is None
        assert any("No valid factors" in e.value for e in app.error)

    def test_message_names_the_destination_step(self, load_project):
        app = load_project(_project(design=_DESIGN_ROWS))
        assert any("Import Results" in e.value for e in app.success)

    def test_restores_factors_and_design(self, load_project):
        app = load_project(_project(design=_DESIGN_ROWS))
        assert [f.name for f in app.session_state["factors"]] == ["Temp"]
        assert len(app.session_state["design"]) == 2


class TestSidebarNavigation:
    """The real sidebar navigates once and does not loop."""

    @pytest.fixture
    def harness(self, tmp_path, monkeypatch):
        """Render the real ``add_project_load`` with navigation recorded.

        ``st.switch_page`` raises immediately, which is awkward to assert on
        inside AppTest, so it is replaced with a recorder. The element tree is
        never reached after it, exactly as in the real app.
        """
        script = tmp_path / "harness.py"
        script.write_text(
            "import streamlit as st\n"
            "from src.ui.components.sidebar import add_project_load\n"
            "add_project_load()\n"
            "st.text('harness body')\n",
            encoding="utf-8",
        )
        calls = []
        monkeypatch.setattr(
            st, "switch_page", lambda page, **kw: calls.append(page)
        )
        app = AppTest.from_file(str(script), default_timeout=30)
        # The element tree is only populated after a run, and the first run
        # renders the counter-0 uploader.
        app.run()
        app._loadfix_calls = calls
        return app

    def _upload(self, app, content, key="project_uploader_0"):
        app.file_uploader(key=key).upload(
            "p.doeproject", content.encode("utf-8"), "application/json"
        )
        return app.run()

    def test_load_navigates_to_the_destination_page(self, harness):
        self._upload(harness, _project(design=_DESIGN_ROWS))
        assert harness._loadfix_calls == ["pages/5_import_results.py"]

    def test_load_bumps_the_uploader_counter(self, harness):
        self._upload(harness, _project(design=_DESIGN_ROWS))
        assert harness.session_state["_project_load_counter"] == 1

    def test_consumed_file_is_unreachable_on_later_reruns(self, harness):
        """The regression test for the loop.

        The uploader keeps its file across reruns, so the load re-entered on
        every one. Bumping the counter in the key makes the file unreachable:
        the next run renders a different, empty widget.
        """
        self._upload(harness, _project(design=_DESIGN_ROWS))
        assert "factors" in harness.session_state

        del harness.session_state["factors"]
        harness.run()
        assert "factors" not in harness.session_state, (
            "project re-loaded on a later rerun: the consumed upload is still "
            "reachable"
        )

    def test_navigation_happens_once_across_many_reruns(self, harness):
        self._upload(harness, _project(design=_DESIGN_ROWS))
        for _ in range(3):
            harness.run()
        assert len(harness._loadfix_calls) == 1

    def test_second_load_uses_the_next_key_and_navigates_again(self, harness):
        """A second project is still loadable, via a fresh key."""
        self._upload(harness, _project(design=_DESIGN_ROWS))
        assert harness.session_state["_project_load_counter"] == 1
        # The counter-1 uploader only exists once a run has rendered it.
        harness.run()
        self._upload(
            harness,
            _project(design_type="full_factorial"),
            key="project_uploader_1",
        )
        assert harness._loadfix_calls == [
            "pages/5_import_results.py",
            "pages/4_preview_design.py",
        ]
        assert harness.session_state["_project_load_counter"] == 2

    def test_failed_load_does_not_navigate_or_loop(self, harness):
        """A bad file must not navigate, and must not leave the counter bumped."""
        harness.file_uploader(key="project_uploader_0").upload(
            "bad.doeproject", b"{not json", "application/json"
        )
        harness.run()
        assert harness._loadfix_calls == []
        assert "_project_load_counter" not in harness.session_state
        harness.run()
        assert harness._loadfix_calls == []


class TestSharedStepTable:
    def test_names_and_pages_line_up_and_files_exist(self):
        from src.ui.utils.state_management import STEP_NAMES, STEP_PAGES

        assert len(STEP_NAMES) == len(STEP_PAGES)
        for page in STEP_PAGES:
            assert (PAGES.parent / page).is_file(), f"missing page: {page}"

    @pytest.mark.parametrize(
        "step,expected_name,expected_page",
        [
            (1, "Define Factors", "pages/1_define_factors.py"),
            (4, "Preview Design", "pages/4_preview_design.py"),
            (5, "Import Results", "pages/5_import_results.py"),
            (8, "Optimize", "pages/8_optimize.py"),
        ],
    )
    def test_accessors(self, step, expected_name, expected_page):
        from src.ui.utils.state_management import step_name, step_page

        assert step_name(step) == expected_name
        assert step_page(step) == expected_page

    def test_out_of_range_step_name_does_not_raise(self):
        from src.ui.utils.state_management import step_name

        assert step_name(0)
        assert step_name(99)


class TestRoutingTargetsResolve:
    """Every literal ``switch_page("pages/...")`` must name a real file.

    Two did not: ``7_augmentation.py`` pointed at a non-existent
    ``pages/4_import_results.py`` and ``8_optimize.py`` at
    ``pages/6_augmentation.py``, both of which raise
    ``StreamlitPageNotFoundError`` when the button is pressed.
    """

    def test_all_switch_page_targets_exist(self):
        import re

        pattern = re.compile(r'switch_page\(\s*[\'"](pages/[^\'"]+)[\'"]')
        checked = 0
        for source in sorted(PAGES.glob("*.py")):
            for target in pattern.findall(source.read_text(encoding="utf-8")):
                checked += 1
                assert (
                    PAGES.parent / target
                ).is_file(), f"{source.name} points at missing {target}"
        # Guard against the pattern silently matching nothing.
        assert checked >= 15, f"only {checked} switch_page targets found"
