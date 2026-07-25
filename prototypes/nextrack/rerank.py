"""Hybrid metadata re-ranker: deterministic post-processing over CF candidates.

The collaborative-filtering core ranks by taste similarity alone, which on a
coherent session floods the top-10 with one artist (8/10 Pink Floyd in the
demo). This module re-scores a widened candidate pool with two metadata terms:

  adjusted = cosine + w_tag * Jaccard(session tags, candidate tags)
                    - w_artist * (same-artist tracks already selected)

The tag term is the "hybrid" boost: candidates whose Last.fm tags overlap the
session's profile rise. The artist term is a greedy diversity penalty (MMR-
style): each additional track by an already-selected artist pays a growing
cost, so the list stays coherent without collapsing into one discography.
Selection is greedy top-1 at a time, so the penalty sees the list built so far.

With both weights at zero the CF order is reproduced exactly (unit-tested),
which makes the re-ranker's contribution cleanly measurable (spec: Phase B4).
"""

from __future__ import annotations

from nextrack.explain import _norm_tags, _session_tag_set

# Defaults from the Phase B4 weight sweep on the leave-last-out split
# (n=10,068): w_artist=0.01 keeps NDCG@10 within the -5% budget (-4.9%)
# while lifting mean unique-artists@10 from 5.91 to 7.02. Larger penalties
# buy more diversity at accelerating relevance cost (0.03 -> -14.9%).
# NOTE: selected on the evaluation split for the prototype; Phase C3 re-does
# this selection properly on a held-out validation partition.
DEFAULT_WEIGHTS = {"tag": 0.05, "artist": 0.01}

CANDIDATE_POOL = 50  # CF candidates considered before re-ranking (B1)


def tag_jaccard(session_tags: set[str], candidate_tags: list[str]) -> float:
    """Jaccard similarity between the session tag profile and one candidate."""
    cand = set(candidate_tags)
    if not session_tags or not cand:
        return 0.0
    return len(session_tags & cand) / len(session_tags | cand)


def rerank(
    candidates: list[tuple[str, float]],
    session_track_ids: list[str],
    tags: dict[str, list[str]],
    artist_of: dict[str, str],
    k: int,
    weights: dict[str, float] | None = None,
) -> list[tuple[str, float]]:
    """Greedily select k tracks from CF candidates using the hybrid score.

    Args:
        candidates: [(track_id, cosine_score), ...] in CF order (pool of ~50).
        session_track_ids: the seed session (for the tag profile).
        tags: {track_id: [tag, ...]} raw tags (normalised internally).
        artist_of: {track_id: artist_name}.
        k: number of tracks to return.
        weights: {"tag": w, "artist": w}; defaults DEFAULT_WEIGHTS.

    Returns [(track_id, adjusted_score), ...] of length <= k. Ties resolve by
    original CF order (stable), so zero weights reproduce pure CF exactly.
    """
    w = DEFAULT_WEIGHTS if weights is None else weights
    session_tags = _session_tag_set(session_track_ids, tags)

    # Precompute the CF-order index for stable tie-breaking and the tag bonus
    # once per candidate (it does not change during greedy selection).
    base: list[tuple[int, str, float, float]] = []
    for idx, (tid, score) in enumerate(candidates):
        bonus = w["tag"] * tag_jaccard(session_tags, _norm_tags(tags.get(tid, [])))
        base.append((idx, tid, score, bonus))

    selected: list[tuple[str, float]] = []
    artist_counts: dict[str, int] = {}
    remaining = base[:]

    while remaining and len(selected) < k:
        best = None
        best_adj = None
        for entry in remaining:
            idx, tid, score, bonus = entry
            penalty = w["artist"] * artist_counts.get(artist_of.get(tid, ""), 0)
            adj = score + bonus - penalty
            # Strict > keeps the earliest (CF-order) candidate on ties.
            if best_adj is None or adj > best_adj:
                best, best_adj = entry, adj
        remaining.remove(best)
        _, tid, _, _ = best
        selected.append((tid, float(best_adj)))
        artist = artist_of.get(tid, "")
        artist_counts[artist] = artist_counts.get(artist, 0) + 1

    return selected
