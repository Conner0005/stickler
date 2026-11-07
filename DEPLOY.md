# Deploy to Streamlit Community Cloud

The website is the multipage app in `src/wetopt/site.py` (Home, Optimiser, Regulations, Method),
started by `streamlit_app.py` at the repo root.

## What the deploy uses
- `streamlit_app.py`: the entry point. It sets `WETOPT_ROOT` to the repo root, so `data/`,
  `examples/` and `docs/` are found even when wetopt is installed into site-packages. If
  `import wetopt` fails, it falls back to putting `src/` on `sys.path`.
- `requirements.txt`: pinned versions of everything imported, plus `.` on the last line, which
  installs wetopt from `pyproject.toml`.
- `.streamlit/config.toml`: a light theme with one accent colour (#2a78d6).
- `data/library.parquet` (40 candidates; CC BY-NC-SA 4.0, derived from SurfPro, see
  `data/LICENSE-library.md`) and `data/regulations.yaml`. Both are committed. The raw SurfPro
  files are not committed or needed.

**Dependency file:** Community Cloud uses the first dependency file it finds, in this order:
`uv.lock` > Pipfile > environment.yml > `requirements.txt` > `pyproject.toml`. This repo has a
`uv.lock`, so Cloud will use it and ignore `requirements.txt`. The lock pins the same versions,
and the `src/` fallback in `streamlit_app.py` covers the case where wetopt itself is not
installed. To deploy with `requirements.txt` as intended, delete `uv.lock` on the branch you
deploy (`git rm uv.lock`) and keep it on your development branch.

## Steps
1. Create a **public** GitHub repository and push this repo to it:
   ```
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```
   The library data is CC BY-NC-SA 4.0 (non-commercial, share-alike), which allows a public
   repo with attribution.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app** → *Deploy a public app from GitHub*.
4. Choose the repository and branch `main`, and set the **main file path** to `streamlit_app.py`.
5. Open **Advanced settings** and set **Python 3.12** (3.11 also works).
6. Click **Deploy** and watch the build log.

## If the build fails
- **`pip install .` fails** (the last line of `requirements.txt`): delete that line and redeploy.
  `streamlit_app.py` then imports wetopt from `src/` via `sys.path`.
- **rdkit has no wheel** for the chosen Python: switch to Python 3.12 under Advanced settings.

## Free-tier limits
- **Memory:** about 1 GB per app. The Optimiser keeps memory low by screening in 50k-blend
  chunks and running Monte Carlo in 500-blend chunks. By default it runs Monte Carlo only on the
  top candidates (fast mode). On the worst case tested (k ≤ 3, infeasible spec), live array
  memory peaked at about 75 MB and process RSS at about 350 MB fast and 430 MB exhaustive.
- **Run time:** each run is capped at 60 s and stops cleanly with a note. The cereal preset
  runs in under 1 s. "Exhaustive certificate check" can hit the cap on large searches.
- **Caching:** the library, surfactant arrays, regulatory screen, data-quality table and chart
  curves are cached with `st.cache_data`.

## Local check before pushing
```
uv run pytest -q                               # includes AppTest smoke tests for every page
uv run streamlit run streamlit_app.py          # http://localhost:8501
```
