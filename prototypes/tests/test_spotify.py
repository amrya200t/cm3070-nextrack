"""Unit tests for the Spotify lookup (spotify.py). No real network."""

import httpx

import nextrack.spotify as sp


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _token_resp():
    return _Resp({"access_token": "tok", "expires_in": 3600})


def _search_resp():
    return _Resp({"tracks": {"items": [{"id": "TRACK123"}]}})


def test_returns_none_without_creds(monkeypatch):
    monkeypatch.delenv("SPOTIFY_CLIENT_ID", raising=False)
    monkeypatch.delenv("SPOTIFY_CLIENT_SECRET", raising=False)
    sp.get_track_id.cache_clear()
    sp._reset_token()
    assert sp.get_track_id("Pink Floyd", "Money") is None


def test_returns_id_and_caches(monkeypatch):
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "cid")
    monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "secret")
    sp.get_track_id.cache_clear()
    sp._reset_token()

    calls = {"POST": 0, "GET": 0}

    def fake_request(method, url, **kw):
        calls[method] += 1
        return _token_resp() if method == "POST" else _search_resp()

    monkeypatch.setattr(sp.httpx, "request", fake_request)

    assert sp.get_track_id("Pink Floyd", "Money") == "TRACK123"
    # second call for the SAME track is lru-cached -> no extra HTTP at all
    assert sp.get_track_id("Pink Floyd", "Money") == "TRACK123"
    assert calls["GET"] == 1
    # token fetched once, reused
    assert calls["POST"] == 1


def test_returns_none_on_network_error(monkeypatch):
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "cid")
    monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "secret")
    sp.get_track_id.cache_clear()
    sp._reset_token()

    def boom(method, url, **kw):
        raise httpx.ConnectTimeout("network down")

    monkeypatch.setattr(sp.httpx, "request", boom)
    assert sp.get_track_id("Pink Floyd", "Money") is None


def test_retries_once_on_connect_timeout(monkeypatch):
    # First attempt times out, retry succeeds -> the lookup still returns the ID.
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "cid")
    monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "secret")
    sp.get_track_id.cache_clear()
    sp._reset_token()

    attempts = {"n": 0}

    def flaky(method, url, **kw):
        attempts["n"] += 1
        if method == "POST":
            return _token_resp()
        # first GET times out, second succeeds
        if attempts["n"] == 2:
            raise httpx.ConnectTimeout("transient")
        return _search_resp()

    monkeypatch.setattr(sp.httpx, "request", flaky)
    assert sp.get_track_id("Pink Floyd", "Money") == "TRACK123"
