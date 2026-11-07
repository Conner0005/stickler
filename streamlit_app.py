"""Streamlit Community Cloud entry point: `streamlit run streamlit_app.py`."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("WETOPT_ROOT", str(ROOT))  # data/, examples/, docs/ live next to this file

try:
    import wetopt  # noqa: F401  (installed from requirements.txt via '.')
except ImportError:  # fallback if the package install failed: use the source tree
    sys.path.insert(0, str(ROOT / "src"))

from wetopt.site import run  # noqa: E402

run()
