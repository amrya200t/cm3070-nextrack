"""API-layer tests (Phase A6). Require trained artifacts; skipped when absent
(e.g. fresh CI checkout) so the pure-function suite stays runnable anywhere."""

import pytest

from nextrack.train import FACTORS_PATH, IDMAP_PATH

pytestmark = pytest.mark.skipif(
    not (FACTORS_PATH.exists() and IDMAP_PATH.exists()),
    reason="trained artifacts not present (run `uv run train` first)",
)

SEEDS = ["41454059", "7741243", "26878430"]  # Pink Floyd, verified in catalogue


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from nextrack.api import app

    with TestClient(app) as c:  # context manager runs the lifespan (artifact load)
        yield c


def test_recommend_happy_path(client):
    r = client.post("/recommend", json={"recent_track_ids": SEEDS, "k": 5})
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 5
    first = body[0]
    assert set(first) == {"track_id", "artist", "title", "score", "why", "shared_tags"}
    assert first["track_id"] not in SEEDS  # seeds never recommended back
    scores = [rec["score"] for rec in body]
    assert scores == sorted(scores, reverse=True)


def test_recommend_unknown_seeds_is_422(client):
    r = client.post("/recommend", json={"recent_track_ids": ["nope-1", "nope-2"]})
    assert r.status_code == 422
    assert "cold-start" in r.json()["detail"].lower()


def test_recommend_k_bounds_rejected(client):
    r = client.post("/recommend", json={"recent_track_ids": SEEDS, "k": 0})
    assert r.status_code == 422
    r = client.post("/recommend", json={"recent_track_ids": SEEDS, "k": 51})
    assert r.status_code == 422


def test_recommend_empty_session_rejected(client):
    r = client.post("/recommend", json={"recent_track_ids": []})
    assert r.status_code == 422


def test_search_finds_catalogue_tracks(client):
    r = client.get("/search", params={"q": "pink floyd", "limit": 5})
    assert r.status_code == 200
    hits = r.json()
    assert hits and all("pink floyd" in h["artist"].lower() for h in hits)


def test_health_reports_catalogue(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["catalogue_tracks"] > 30_000
