"""Website smoke test: the entry point and every page render; the cereal preset returns a result."""

import pytest
from streamlit.testing.v1 import AppTest

from wetopt import config

PAGES = ["home_page", "optimiser_page", "regulations_page", "method_page"]


def _render(page: str) -> None:
    from wetopt import site

    getattr(site, page)()


def test_entrypoint_renders_home():
    at = AppTest.from_file(str(config.ROOT / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "wetopt"


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_function(_render, args=(page,), default_timeout=60).run()
    assert not at.exception, page


def test_optimiser_default_answers():
    at = AppTest.from_function(_render, args=("optimiser_page",), default_timeout=120).run()
    assert not at.exception and len(at.sidebar.header) == 5
    assert any("Data-quality exclusions: 1" in e.label for e in at.expander)
    at.sidebar.button[0].click().run()  # Find recipe
    assert not at.exception
    outcome = at.session_state["run"]["o"]
    assert outcome.result is not None and outcome.result.k == 1
    assert any("water alone" in el.value for el in at.success)
