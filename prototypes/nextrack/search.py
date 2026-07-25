"""Ranked, fuzzy-tolerant track search over the in-memory catalogue.

Hand-rolled tiered token scoring (exact > prefix > word-boundary > substring),
matched against the concatenated "artist title" haystack so multi-word queries
that span both fields ("pink floyd money") work. A rapidfuzz typo fallback runs
only when too few tiered hits are found, so the common case pays no fuzzy cost.
"""

from __future__ import annotations

import unicodedata

from rapidfuzz import fuzz

EXACT, PREFIX, WORD, SUBSTR = 100, 80, 60, 40
ARTIST_WEIGHT = 1.2


def _normalize(s: str) -> str:
    """Casefold + strip diacritics so 'Björk' and 'bjork' compare equal."""
    decomposed = unicodedata.normalize("NFKD", s)
    without_marks = "".join(c for c in decomposed if not unicodedata.combining(c))
    return without_marks.casefold()


def _token_tier(token: str, haystack: str) -> int:
    """Best-tier score for one normalised token against a normalised haystack."""
    if token == haystack:
        return EXACT
    if haystack.startswith(token):
        return PREFIX
    # word-boundary: token starts one of the haystack's words
    if any(word.startswith(token) for word in haystack.split()):
        return WORD
    if token in haystack:
        return SUBSTR
    # space-insensitive: a run-together query ("dualipa") should match a name
    # whose words were typed without the space ("dua lipa"). Scored at SUBSTR so
    # genuine word matches always rank above it.
    if " " in haystack and token in haystack.replace(" ", ""):
        return SUBSTR
    return 0


def _score(query_tokens: list[str], artist: str, title: str) -> float | None:
    """Sum of per-token best tiers, or None if any token matches nothing.

    Each token is scored against the artist and the title separately; the token
    takes its better tier, and an artist-side win is weighted up (artists are the
    stronger disambiguator). AND semantics: one unmatched token drops the row.
    """
    total = 0.0
    for tok in query_tokens:
        a = _token_tier(tok, artist) * ARTIST_WEIGHT
        t = _token_tier(tok, title)
        best = max(a, t)
        if best == 0:
            return None
        total += best
    return total


def rank_tracks(query: str, names: dict, limit: int) -> list[dict]:
    """query + {track_id: (artist, title)} -> ranked [{track_id, artist, title}]."""
    tokens = [_normalize(t) for t in query.split() if t.strip()]
    if not tokens:
        return []

    scored = []
    for tid, (artist, title) in names.items():
        na, nt = _normalize(artist), _normalize(title)
        s = _score(tokens, na, nt)
        if s is not None:
            # tie-break: higher score, then shorter combined name, then id
            scored.append((s, -(len(na) + len(nt)), tid, artist, title))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    results = [
        {"track_id": tid, "artist": artist, "title": title}
        for _, _, tid, artist, title in scored[:limit]
    ]
    if len(results) >= limit:
        return results

    # Fuzzy fallback — only when the tiered pass was thin (typos). Match each
    # query token against the haystack's words with a per-token ratio, and keep
    # AND semantics: EVERY token must fuzzy-match some word (so "…xyzzy" still
    # yields nothing). This is why we compare token-vs-word, not the whole query
    # vs the whole name — the latter dilutes a good single-token typo match.
    have = {r["track_id"] for r in results}
    FUZZY_MIN = 80  # per-token similarity floor; below tier scores by construction
    fuzz_scored = []
    for tid, (artist, title) in names.items():
        if tid in have:
            continue
        words = (_normalize(artist) + " " + _normalize(title)).split()
        total = 0.0
        ok = True
        for tok in tokens:
            best = max((fuzz.ratio(tok, w) for w in words), default=0.0)
            if best < FUZZY_MIN:
                ok = False
                break
            total += best
        if ok:
            fuzz_scored.append((total, tid, artist, title))

    fuzz_scored.sort(key=lambda x: x[0], reverse=True)
    for _total, tid, artist, title in fuzz_scored:
        results.append({"track_id": tid, "artist": artist, "title": title})
        if len(results) >= limit:
            break
    return results
