# NextTrack

Stateless next-track music recommendation API (CM3070 final project; template:
CM3035 Advanced Web Development, Project Idea 2). The client sends recent track
IDs with each request — no accounts, no stored history — and gets ranked
recommendations back with tag-based `why` explanations.

## Setup (uv, Python 3.11)

```bash
cd code
uv venv
uv sync
```

## Run

```bash
uv run train      # fit ALS on the LFM-2b slice, save artifacts/
uv run pytest     # test suite (24 tests; API tests skip if artifacts absent)
uv run evaluate   # offline evaluation: leave-last-out, Recall/NDCG/MRR@10
uv run api        # serve the REST API on http://127.0.0.1:8000 (docs at /docs)
uv run demo       # open 02-demo.ipynb (notebook demo)
uv run explore    # open 01-data-explore.ipynb
```

## API

`POST /recommend`

```bash
curl -s http://127.0.0.1:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"recent_track_ids": ["41454059", "7741243", "26878430"], "k": 5}'
```

Request: `{recent_track_ids: [string, ...], k: int (1-50, default 10)}`.
Response: ranked list of `{track_id, artist, title, score, why, shared_tags}`.
Unknown-only seeds return `422` (cold-start is out of scope by design).

`GET /search?q=pink+floyd` — name → track_id lookup for building sessions.
`GET /health` — liveness + catalogue stats.

Measured on the dev laptop: p95 latency 2.0 ms over 100 requests (threshold
from the preliminary report: ≤ 100 ms).

## Evaluation snapshot (leave-last-out, n = 10,068 sessions)

| Model | Recall@10 | NDCG@10 | MRR@10 |
| --- | --- | --- | --- |
| Popularity baseline | 0.0016 | 0.0006 | 0.0003 |
| ALS session-vector | 0.0488 | 0.0275 | 0.0209 |

## Layout

- `prototypes/nextrack/` — flat modules: `data`, `train`, `infer`, `explain`,
  `baselines`, `evaluate`, `api` (+ `seeds`)
- `prototypes/tests/` — pytest suite
- `prototypes/notebooks/` — exploration + demo notebooks
- `prototypes/figures/` — evaluation figures (report-ready)
- `prototypes/data/`, `prototypes/artifacts/` — gitignored, regenerable;
  data provenance and integrity checks in `prototypes/data/INTEGRITY.md`

## Data note

Trained on the LFM-2b 2020 Subset (Schedl et al., 2022), sourced via the
Internet Archive after the original distribution was withdrawn; the dataset
cannot be redistributed here. Spotify is used for playback embedding only —
no Spotify data enters the model (their recommendation/audio-features
endpoints are unavailable to new apps and training on Spotify content is
prohibited by their developer policy).
