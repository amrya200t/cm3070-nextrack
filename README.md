# NextTrack

Stateless next-track music recommendation API (CM3070 final project; template:
CM3035 Advanced Web Development, Project Idea 2). The client sends recent track
IDs with each request — no accounts, no stored history — and gets ranked
recommendations back with tag-based `why` explanations.

**Live demo:** https://nexttrack.amrezzat.com (full-scale model: 948,740 tracks
trained on 30.4M LFM-2b listening events).

## Quickstart (Docker)

```bash
cd code
docker compose up        # builds the image (model artifacts baked in) and
                         # serves API + demo page on http://127.0.0.1:8000
```

The image bakes whichever artifact set `ARTIFACTS_DIR` points at: the default
1M prototype artifacts (~25 MB), or the full-scale model after `uv run python
-m nextrack.full_scale train`:

```bash
docker build --build-arg ARTIFACTS_DIR=prototypes/artifacts_full -t nexttrack:v2 .
```

## Setup for development (uv, Python 3.11)

```bash
cd code
uv venv
uv sync
```

## Run

```bash
uv run train           # fit ALS on the 1M LFM-2b slice, save artifacts/
uv run pytest          # test suite (67 tests; API tests skip if artifacts absent)
uv run evaluate        # quick evaluation: leave-last-out, Recall/NDCG/MRR@10
uv run evaluate-full   # full protocol: 70/10/20 split, 5 systems, bootstrap CIs
uv run enrich          # backfill genre tags for untagged tracks (Wikidata, resumable)
uv run api             # serve API + demo page on http://127.0.0.1:8000
uv run api-dev         # same, with auto-reload on code changes
uv run demo            # open 02-demo.ipynb (notebook demo)
uv run explore         # open 01-data-explore.ipynb

# full-scale (30.4M events) pipeline — writes to artifacts_full/, never
# touching the pinned prototype artifacts:
uv run --no-sync python -m nextrack.scale_experiment    # 1M/5M/20M/full sweep
uv run --no-sync python -m nextrack.full_scale train    # final model + overlay
uv run --no-sync python -m nextrack.full_scale eval     # full protocol at 30.4M
```

Serve the full-scale model locally with `NEXTRACK_ARTIFACTS`:

```bash
NEXTRACK_ARTIFACTS=$PWD/prototypes/artifacts_full uv run api
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

Measured p95 latency: 2.0 ms at the 1M scale; at full scale (948,740 tracks)
pure CF ~46 ms and the re-ranked default ~100 ms on the dev laptop (threshold
from the preliminary report: ≤ 100 ms). Search runs on a startup-built index
(normalise-once + C-level substring prune + inverted word index for the typo
fallback): 36–470 ms per query at full catalogue scale, with edition-variant
dedup and playcount-aware ranking (originals above covers).

## Evaluation snapshot (70/10/20 chronological split)

Full-scale model (30.4M events, n = 14,851 test sessions):

| Model | Recall@10 | NDCG@10 | MRR@10 |
| --- | --- | --- | --- |
| Popularity baseline | 0.0022 | 0.0009 | 0.0005 |
| Content-only (tags) | 0.0302 | 0.0160 | 0.0117 |
| Item-kNN | 0.0813 | 0.0483 | 0.0380 |
| ALS session-vector | 0.0958 | 0.0543 | 0.0415 |
| ALS + hybrid re-ranker | 0.0895 | 0.0520 | 0.0403 |

1M prototype slice (n = 9,873): ALS 0.0444 → the scaling sweep
(`prototypes/figures/scaling.png`) shows recall tracking catalogue coverage
almost proportionally and saturating at ~20M events. Bootstrap 95% CIs,
popularity-bucket analysis, and validation-locked re-ranker weights are in
`prototypes/artifacts/eval_full.json` (1M) and `eval_full_30m.json` (full).

Tag coverage for `why` explanations uses a four-tier provenance ladder (exact
Last.fm tags > edition-sibling copy > artist-level Last.fm top-tags > Wikidata
genres): 77.9% of the full catalogue, 87.2% weighted by plays.

## Layout

- `prototypes/nextrack/` — flat modules: `data`, `train`, `infer`, `explain`,
  `baselines`, `evaluate`, `evaluate_full`, `rerank`, `search`, `enrich`,
  `scale_experiment`, `full_scale`, `spotify`, `api` (+ `seeds`)
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
