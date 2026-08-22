"""Ranked, fuzzy-tolerant track search over the in-memory catalogue.

Hand-rolled tiered token scoring (exact > prefix > word-boundary > substring),
matched against the concatenated "artist title" haystack so multi-word queries
that span both fields ("pink floyd money") work. A rapidfuzz typo fallback runs
only when too few tiered hits are found, so the common case pays no fuzzy cost.
"""

from __future__ import annotations

import unicodedata

import numpy as np
from rapidfuzz import fuzz, process

from nextrack.spotify import _clean_title

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


class SearchIndex:
    """Precomputed search structures for one catalogue.

    Normalising 948k artist/title pairs per keystroke costs seconds; doing it
    once at build time and pruning with C-speed substring tests before the
    Python tier-scoring keeps a query under ~100ms at full catalogue scale.
    The fuzzy fallback runs against an inverted unique-word index via
    rapidfuzz's optimised extract instead of a full catalogue scan.
    """

    def __init__(self, names: dict, plays: dict | None = None):
        self.rows: list[tuple[str, str, str, str, str]] = []  # tid, artist, title, na, nt
        self.hay: list[str] = []  # normalised "artist title"
        self.plays: list[int] = []  # total playcount per row (0 when unknown)
        self.dedup_key: list[tuple[str, str]] = []  # (artist, edition-stripped title)
        hay_nospace: list[str] = []
        word_rows: dict[str, list[int]] = {}
        plays = plays or {}
        for i, (tid, (artist, title)) in enumerate(names.items()):
            na, nt = _normalize(artist), _normalize(title)
            h = na + " " + nt
            self.rows.append((tid, artist, title, na, nt))
            self.hay.append(h)
            self.plays.append(int(plays.get(tid, 0)))
            self.dedup_key.append((na, _normalize(_clean_title(title))))
            hay_nospace.append(h.replace(" ", ""))
            for w in set(h.split()):
                word_rows.setdefault(w, []).append(i)
        self.word_rows = word_rows
        self.words = list(word_rows)
        # Megastring prune structures: one newline-joined blob per variant lets
        # the candidate scan run inside C's str.find instead of a 948k-row
        # Python loop; match positions map back to rows via searchsorted over
        # the cumulative offsets. The newline separator prevents cross-row
        # substring matches.
        self.blob = "\n".join(self.hay)
        self.blob_ns = "\n".join(hay_nospace)
        self.offsets = np.cumsum([0] + [len(h) + 1 for h in self.hay])
        self.offsets_ns = np.cumsum([0] + [len(h) + 1 for h in hay_nospace])

    def _rows_containing(self, token: str, blob: str, offsets: np.ndarray) -> set[int]:
        positions = []
        start = blob.find(token)
        while start >= 0:
            positions.append(start)
            start = blob.find(token, start + 1)
        if not positions:
            return set()
        rows = np.searchsorted(offsets, np.asarray(positions), side="right") - 1
        return set(map(int, np.unique(rows)))

    def search(self, query: str, limit: int) -> list[dict]:
        tokens = [_normalize(t) for t in query.split() if t.strip()]
        if not tokens:
            return []

        # Cheap prune: every token must appear as a substring of the (possibly
        # space-collapsed) haystack. Longest token first (rarest, smallest row
        # set); each further token intersects, so the set only shrinks.
        # Scoring is Python-speed, so the prune must keep the candidate set
        # small even for degenerate queries ("a", "the"). Tokens under 3 chars
        # skip the megastring (a 1-char token matches millions of positions)
        # and use word-boundary candidates from the inverted index instead —
        # which is also what the tier scoring would rank on top anyway.
        SCORE_CAP = 20_000

        def rows_for(t: str) -> set[int]:
            if len(t) >= 3:
                return self._rows_containing(t, self.blob, self.offsets) | \
                    self._rows_containing(t, self.blob_ns, self.offsets_ns)
            rows: set[int] = set(self.word_rows.get(t, ()))
            for w in self.words:
                if w.startswith(t) and w != t:
                    rows.update(self.word_rows[w])
                    if len(rows) > SCORE_CAP:
                        break
            return rows

        cand_set: set[int] | None = None
        for t in sorted(tokens, key=len, reverse=True):
            rows = rows_for(t)
            cand_set = rows if cand_set is None else (cand_set & rows)
            if not cand_set:
                break
        candidates = sorted(cand_set or ())[:SCORE_CAP]

        scored = []
        for i in candidates:
            tid, artist, title, na, nt = self.rows[i]
            s = _score(tokens, na, nt)
            if s is not None:
                # tie-breaks: higher tier score, then most-played (so originals
                # rank above obscure covers of the same title), then shorter
                # combined name (base editions above remaster variants).
                scored.append(
                    (s, self.plays[i], -(len(na) + len(nt)), i, tid, artist, title)
                )

        scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        # Collapse edition variants: one result per (artist, base title), the
        # best-ranked (most played) variant representing the group.
        results = []
        seen_keys: set[tuple[str, str]] = set()
        for _s, _p, _l, i, tid, artist, title in scored:
            key = self.dedup_key[i]
            if key in seen_keys:
                continue
            seen_keys.add(key)
            results.append({"track_id": tid, "artist": artist, "title": title})
            if len(results) >= limit:
                break
        if len(results) >= limit:
            return results

        # Fuzzy fallback — only when the tiered pass was thin (typos). AND
        # semantics preserved: every token must fuzzy-match some word of a row.
        # Instead of scanning the catalogue, match each token against the
        # UNIQUE word list (rapidfuzz C implementation), then intersect the
        # rows behind the matched words.
        FUZZY_MIN = 80  # per-token similarity floor
        token_rows: list[set[int]] = []
        token_word_score: list[dict[str, float]] = []
        for tok in tokens:
            matches = process.extract(
                tok, self.words, scorer=fuzz.ratio, score_cutoff=FUZZY_MIN, limit=50
            )
            rows: set[int] = set()
            wscore: dict[str, float] = {}
            for word, score, _idx in matches:
                wscore[word] = max(wscore.get(word, 0.0), score)
                rows.update(self.word_rows[word])
            token_rows.append(rows)
            token_word_score.append(wscore)
        if not token_rows or not all(token_rows):
            return results

        shared = set.intersection(*token_rows)
        have = {r["track_id"] for r in results}
        fuzz_scored = []
        for i in shared:
            tid, artist, title, na, nt = self.rows[i]
            if tid in have:
                continue
            words = set(self.hay[i].split())
            total = 0.0
            for wscore in token_word_score:
                total += max((s for w, s in wscore.items() if w in words), default=0.0)
            fuzz_scored.append((total, tid, artist, title))

        fuzz_scored.sort(key=lambda x: x[0], reverse=True)
        for _total, tid, artist, title in fuzz_scored:
            results.append({"track_id": tid, "artist": artist, "title": title})
            if len(results) >= limit:
                break
        return results


# One cached index per catalogue object: the API loads the id maps once per
# process, so in practice this builds exactly once (at first search or at
# startup warm-up), keyed by the names dict's identity.
_INDEX: SearchIndex | None = None
_INDEX_FOR: int | None = None


def get_index(names: dict, plays: dict | None = None) -> SearchIndex:
    global _INDEX, _INDEX_FOR
    if _INDEX is None or _INDEX_FOR != id(names):
        _INDEX = SearchIndex(names, plays)
        _INDEX_FOR = id(names)
    return _INDEX


def rank_tracks(
    query: str, names: dict, limit: int, plays: dict | None = None
) -> list[dict]:
    """query + {track_id: (artist, title)} -> ranked [{track_id, artist, title}]."""
    return get_index(names, plays).search(query, limit)
