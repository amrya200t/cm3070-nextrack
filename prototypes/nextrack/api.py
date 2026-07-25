"""FastAPI service: the stateless recommendation API the template promises.

Thin HTTP layer over the existing engine — all recommendation logic stays in
infer.py/explain.py; this module only translates HTTP <-> recommend(). The
statelessness contract is visible in the schema itself: the client sends
recent_track_ids on every request, and no user identity exists anywhere.

Run with `uv run api` (serves on http://127.0.0.1:8000, docs at /docs).
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from nextrack import recommend
from nextrack.infer import _load_artifacts


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load artifacts once at startup (not per request): the 36k x 64 factor
    # matrix and id maps are process-lifetime state, exactly like infer.py's
    # module cache. A cold first request would otherwise pay the load cost.
    _, _, maps = _load_artifacts()
    app.state.maps = maps
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
    q: str = Query(..., min_length=2, description="Case-insensitive substring of artist or title."),
    limit: int = Query(20, ge=1, le=100),
) -> list[dict]:
    """Name -> track_id lookup so clients can build sessions from real titles."""
    needle = q.lower()
    hits: list[dict] = []
    for tid, (artist, title) in app.state.maps["track_names"].items():
        if needle in artist.lower() or needle in title.lower():
            hits.append({"track_id": tid, "artist": artist, "title": title})
            if len(hits) >= limit:
                break
    return hits


@app.get("/health")
def health() -> dict:
    """Liveness + catalogue stats (also the report's 'it runs' evidence)."""
    maps = app.state.maps
    return {
        "status": "ok",
        "catalogue_tracks": len(maps["track_id_to_index"]),
        "tagged_tracks": len(maps["tags"]),
    }


def main() -> None:
    import os

    import uvicorn

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
    from pathlib import Path

    import uvicorn

    uvicorn.run(
        "nextrack.api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=[str(Path(__file__).resolve().parent)],
    )
