# NextTrack — Prototype

Stateless next-track music recommendation. Design-phase prototype: one public
function `recommend(recent_track_ids, k=10)` proving stateless session-vector
inference is feasible. Throwaway code — production lands in Topic 6.

## Setup (uv, Python 3.11)

```bash
cd code
uv venv
uv sync
```

## Verify

```bash
uv run python -c "import nextrack; print('ok')"
```

## Run

```bash
uv run train      # fit ALS, save artifacts (from Fri 26 Jun)
uv run explore    # open 01-data-explore.ipynb
uv run demo       # open 02-demo.ipynb (the video demo)
```

## Layout

- `prototypes/nextrack/` — five flat modules (data, train, infer, explain) + seeds
- `prototypes/notebooks/` — exploration + demo notebooks
- `prototypes/data/`, `prototypes/artifacts/` — gitignored, regenerable

## Not in scope (deliberate, see spec)

No HTTP/FastAPI, no Redis/Postgres, no Docker, no hyperparameter tuning,
no metadata-graph re-ranker, no tests. Those are Topic 6/7.
