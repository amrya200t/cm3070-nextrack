"""FastAPI service: the stateless recommendation API the template promises.

Thin HTTP layer over the existing engine — all recommendation logic stays in
infer.py/explain.py; this module only translates HTTP <-> recommend(). The
statelessness contract is visible in the schema itself: the client sends
recent_track_ids on every request, and no user identity exists anywhere.

Run with `uv run api` (serves on http://127.0.0.1:8000, docs at /docs).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from nextrack import recommend
from nextrack.infer import _load_artifacts
from nextrack.search import get_index, rank_tracks
from nextrack.spotify import get_track_id as spotify_track_id

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load artifacts once at startup (not per request): the 36k x 64 factor
    # matrix and id maps are process-lifetime state, exactly like infer.py's
    # module cache. A cold first request would otherwise pay the load cost.
    _, _, maps = _load_artifacts()
    app.state.maps = maps
    # Build the search index at startup too: normalising the full catalogue is
    # a seconds-scale one-off that must not land on the first keystroke.
    get_index(maps["track_names"], maps.get("track_plays"))
    yield


app = FastAPI(
    title="NextTrack",
    version="0.1.0",
    summary="Stateless music recommendation API",
    description=(
        "Send recent track IDs, get ranked next-track recommendations with "
        "tag-based explanations. No accounts, no stored history: the recent "
        "tracks in the request are the only personalisation signal."
    ),
    lifespan=lifespan,
)

# CORS: only needed when the demo page is hosted on a different origin from
# the API (the default same-origin /app mount below needs none). "*" is fine
# here — the API is stateless and public by design: no auth, no cookies.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("NEXTRACK_CORS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class RecommendParams(BaseModel):
    rerank: bool = Field(
        True,
        description=(
            "Apply the hybrid metadata re-ranker (tag-overlap boost + artist "
            "diversity penalty). False returns the pure CF ranking."
        ),
    )


class RecommendRequest(BaseModel):
    recent_track_ids: list[str] = Field(
        ...,
        min_length=1,
        description="Track IDs of the recent listening session (1+ required).",
        examples=[["41454059", "7741243", "26878430"]],
    )
    k: int = Field(10, ge=1, le=50, description="Number of recommendations.")
    params: RecommendParams = Field(default_factory=RecommendParams)


class Recommendation(BaseModel):
    track_id: str
    artist: str
    title: str
    score: float
    why: str
    shared_tags: list[str]


class TrackHit(BaseModel):
    track_id: str
    artist: str
    title: str


@app.post("/recommend", response_model=list[Recommendation])
def post_recommend(req: RecommendRequest) -> list[dict]:
    """Ranked next-track recommendations for the supplied session."""
    try:
        return recommend(req.recent_track_ids, k=req.k, rerank=req.params.rerank)
    except ValueError:
        # None of the seeds exist in the trained catalogue. Cold-start is out
        # of scope by design (prelim 3.6 predicted failure modes), so this is
        # a client error, not a server one.
        raise HTTPException(
            status_code=422,
            detail=(
                "None of the recent_track_ids are in the trained catalogue. "
                "Cold-start tracks are out of scope; use /search to find "
                "valid catalogue track IDs."
            ),
        )


@app.get("/search", response_model=list[TrackHit])
def search(
    q: str = Query(..., min_length=2, description="Artist and/or title; multi-word, typo-tolerant."),
    limit: int = Query(20, ge=1, le=100),
) -> list[dict]:
    """Ranked name -> track_id lookup so clients can build sessions from real titles."""
    return rank_tracks(
        q, app.state.maps["track_names"], limit,
        plays=app.state.maps.get("track_plays"),
    )


@app.get("/spotify")
def spotify(artist: str = Query(...), title: str = Query(...)) -> dict:
    """Resolve a track to a Spotify ID for the in-place player; null if unknown."""
    return {"spotify_id": spotify_track_id(artist, title)}


@app.get("/health")
def health() -> dict:
    """Liveness + catalogue stats (also the report's 'it runs' evidence)."""
    maps = app.state.maps
    return {
        "status": "ok",
        "catalogue_tracks": len(maps["track_id_to_index"]),
        "tagged_tracks": len(maps["tags"]),
    }


# Serve the demo page at /app (same origin as the API -> engine.js goes live
# automatically, no CORS involved). Root redirects there for a friendly URL.
if WEB_DIR.is_dir():
    app.mount("/app", StaticFiles(directory=WEB_DIR, html=True), name="app")

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/app/")


def _load_env_local() -> None:
    """Load KEY=VALUE lines from code/.env.local into os.environ if present.

    A [project.scripts] entry point runs inside the process, so `uv run api`
    can't pass --env-file to itself; we load the file here before uvicorn starts.
    Only fills vars that are not already set (real env wins). No dependency.
    """
    if os.getenv("SPOTIFY_CLIENT_ID"):
        return
    env_path = Path(__file__).resolve().parent.parent.parent / ".env.local"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def main() -> None:
    import uvicorn

    _load_env_local()
    # Defaults favour the reproducible local demo; env vars cover Docker later.
    uvicorn.run(
        "nextrack.api:app",
        host=os.getenv("NEXTRACK_HOST", "127.0.0.1"),
        port=int(os.getenv("NEXTRACK_PORT", "8000")),
    )


def dev() -> None:
    """`uv run api-dev`: auto-reload on code changes (nodemon-style).

    watchfiles restarts the server whenever a file under nextrack/ changes.
    Dev-only: each restart re-runs the lifespan, so the ~2s artifact load
    happens on every save — fine locally, wrong for serving.
    """
    import uvicorn

    _load_env_local()
    uvicorn.run(
        "nextrack.api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=[str(Path(__file__).resolve().parent)],
    )
