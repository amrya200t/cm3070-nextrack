"""Resolve (artist, title) -> Spotify track ID for the in-place player.

Client-credentials flow (server-to-server, no user login): needs only
SPOTIFY_CLIENT_ID + SPOTIFY_CLIENT_SECRET. Degrades to None on missing creds,
network error, or no match — the caller then shows an open-in-Spotify link.

RAIL: metadata lookup only. No Spotify recommendation/audio-feature data, and
nothing here enters the recommendation model. Caching track IDs is permitted;
bulk content-metadata storage is not, so we cache only the resolved IDs.
"""

from __future__ import annotations

import functools
import os
import time

import httpx

_TOKEN_URL = "https://accounts.spotify.com/api/token"
_SEARCH_URL = "https://api.spotify.com/v1/search"

# api.spotify.com intermittently hangs connections from some networks: when a
# connection gets through it responds in well under a second, but a stuck one
# would otherwise block for the whole timeout. So we use a tight timeout and one
# quick retry — a hung attempt fails fast and the retry usually succeeds, rather
# than the caller waiting ~8s for a single hung request before falling back.
_TIMEOUT = 2.5
_RETRIES = 1


def _request(method: str, url: str, **kwargs) -> httpx.Response:
    """httpx request with a tight timeout and one retry on connect/timeout.

    Retries only transient connection failures (ConnectError/ConnectTimeout/
    ReadTimeout); a real HTTP error (e.g. 401) raises immediately via the
    caller's raise_for_status. Raises httpx.HTTPError if all attempts fail.
    """
    last_exc: httpx.HTTPError | None = None
    for _ in range(_RETRIES + 1):
        try:
            return httpx.request(method, url, timeout=_TIMEOUT, **kwargs)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
            last_exc = exc
    raise last_exc  # type: ignore[misc]

# module-level bearer-token cache: (token, expires_at monotonic)
_token: str | None = None
_token_expires: float = 0.0


def _reset_token() -> None:
    """Test hook: forget the cached bearer token."""
    global _token, _token_expires
    _token = None
    _token_expires = 0.0


def _get_token() -> str | None:
    """Fetch/reuse a client-credentials bearer token, or None if unavailable."""
    global _token, _token_expires
    if _token and time.monotonic() < _token_expires:
        return _token
    cid = os.getenv("SPOTIFY_CLIENT_ID")
    secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not cid or not secret:
        return None
    try:
        resp = _request(
            "POST",
            _TOKEN_URL,
            data={"grant_type": "client_credentials"},
            auth=(cid, secret),
        )
        resp.raise_for_status()
        payload = resp.json()
    except httpx.HTTPError:
        return None
    _token = payload["access_token"]
    # refresh 60s early to avoid edge expiry
    _token_expires = time.monotonic() + payload.get("expires_in", 3600) - 60
    return _token


@functools.lru_cache(maxsize=2048)
def get_track_id(artist: str, title: str) -> str | None:
    """Resolve (artist, title) -> Spotify track ID, or None if not found."""
    token = _get_token()
    if not token:
        return None
    try:
        resp = _request(
            "GET",
            _SEARCH_URL,
            params={"q": f"{artist} {title}", "type": "track", "limit": 1},
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        items = resp.json().get("tracks", {}).get("items", [])
    except httpx.HTTPError:
        return None
    return items[0]["id"] if items else None
