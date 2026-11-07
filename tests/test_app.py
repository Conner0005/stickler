"""App smoke test: imports, and a headless run of the cereal preset returns a result."""

from streamlit.testing.v1 import AppTest

from wetopt import config


def test_app_imports_and_cereal_headless():
    import wetopt.app  # noqa: F401  (import has no side effects: UI runs under __main__)

    at = AppTest.from_file(str(config.ROOT / "src" / "wetopt" / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception  # opens on wheat + azoxystrobin + EU + Standard and runs it
    at.sidebar.button[0].click().run()  # Find recipe
    assert not at.exception
    outcome = at.session_state["run"]["o"]
    assert outcome.result is not None and outcome.result.k == 1
    assert any("water alone" in m.value for m in at.markdown)
