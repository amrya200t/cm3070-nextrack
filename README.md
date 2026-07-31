# NextTrack

Stateless next-track music recommendation API (CM3070 final project; template:
CM3035 Advanced Web Development, Project Idea 2). The client sends recent track
IDs with each request — no accounts, no stored history — and gets ranked
recommendations back with tag-based `why` explanations.

## Quickstart (Docker)

```bash
cd code
docker compose up        # builds the image (model artifacts baked in) and
                         # serves API + demo page on http://127.0.0.1:8000
```

## Setup for development (uv, Python 3.11)

```bash
cd code
uv venv
uv sync
```

## Run

```bash
uv run train           # fit ALS on the LFM-2b slice, save artifacts/
uv run pytest          # test suite (65 tests; API tests skip if artifacts absent)
uv run evaluate        # quick evaluation: leave-last-out, Recall/NDCG/MRR@10
uv run evaluate-full   # full protocol: 70/10/20 split, 5 systems, bootstrap CIs
uv run enrich          # backfill genre tags for untagged tracks (Wikidata, resumable)
uv run api             # serve API + demo page on http://127.0.0.1:8000
uv run api-dev         # same, with auto-reload on code changes
uv run demo            # open 02-demo.ipynb (notebook demo)
uv run explore         # open 01-data-explore.ipynb
```

The demo page is served at `/app` (the root redirects there); interactive API
docs at `/docs`. CI runs the test suite on every push (GitHub Actions).

## Spotify (optional — in-place playback)

The demo's in-place player resolves each track to Spotify via the Search API.
It needs client credentials in `code/.env.local` (gitignored):

```ini
SPOTIFY_CLIENT_ID=...
SPOTIFY_CLIENT_SECRET=...
```

`uv run api` loads `.env.local` automatically. Without the credentials — or if
Spotify is unreachable — the player falls back to open-in-Spotify search links;
nothing breaks. Only `SPOTIFY_CLIENT_ID` / `SPOTIFY_CLIENT_SECRET` are used
(metadata lookup only — no Spotify data enters the recommendation model).

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

## Evaluation snapshot (70/10/20 chronological split, n = 9,873 test sessions)

| Model | Recall@10 | NDCG@10 | MRR@10 |
| --- | --- | --- | --- |
| Popularity baseline | 0.0011 | 0.0004 | 0.0002 |
| Content-only (tags) | 0.0256 | 0.0139 | 0.0103 |
| Item-kNN | 0.0368 | 0.0206 | 0.0156 |
| ALS session-vector | 0.0444 | 0.0242 | 0.0180 |
| ALS + hybrid re-ranker | 0.0426 | 0.0235 | 0.0176 |

Bootstrap 95% CIs, per-popularity-bucket analysis, and the validation-locked
re-ranker weights are in `prototypes/artifacts/eval_full.json`
(`uv run evaluate-full` regenerates everything).

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
