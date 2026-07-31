"""Inference: build session vector, cosine-nearest top-K, assemble results.

`recommend` is the single public entry point (re-exported from __init__). It is
stateless: the caller passes recent track IDs each request, no user identity is
stored. Artifacts (factors + id maps + names + tags) are loaded once and cached.

Session aggregation = mean of the recent tracks' latent factors (spec §1.4).
That `build_session_vector` step is the core modelling contribution; see its
docstring for the rationale to defend on camera (spec §2.5).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from nextrack.explain import tag_overlap_explain
from nextrack.train import FACTORS_PATH, IDMAP_PATH

# Module-level cache so repeated recommend() calls don't reload or re-normalise.
_FACTORS: np.ndarray | None = None
_UNIT_FACTORS: np.ndarray | None = None
_MAPS: dict | None = None


def _load_artifacts() -> tuple[np.ndarray, np.ndarray, dict]:
    """Lazy-load and cache the trained factors, their unit-normalised form, and id maps.

    Unit-normalising is done once here (not per recommend() call): the matrix is
    fixed for the life of the process, so caching the 36k x 64 normalised array
    keeps each request to a single matrix-vector product.
    """
    global _FACTORS, _UNIT_FACTORS, _MAPS
    if _FACTORS is None or _MAPS is None:
        _FACTORS = np.load(FACTORS_PATH)
        norms = np.linalg.norm(_FACTORS, axis=1)
        norms[norms == 0] = 1e-9
        _UNIT_FACTORS = _FACTORS / norms[:, None]
        with open(IDMAP_PATH, "rb") as fh:
            _MAPS = pickle.load(fh)
        # Fold in the Wikidata genre overlay (Phase F1) so untagged tracks get a
        # `why` explanation. Last.fm tags always win where present; the overlay
        # only fills tracks that have none. No-op if the harvest hasn't run.
        _MAPS["tags"] = _apply_enrichment_overlay(_MAPS["tags"])
    return _FACTORS, _UNIT_FACTORS, _MAPS


def _apply_enrichment_overlay(tags: dict) -> dict:
    """Merge the Wikidata genre overlay into tags if the overlay file exists."""
    import json

    from nextrack.enrich import OVERLAY_PATH, merged_tags

    if not OVERLAY_PATH.is_file():
        return tags
    overlay = json.loads(OVERLAY_PATH.read_text(encoding="utf-8"))
    return merged_tags(tags, overlay)


def build_session_vector(
    recent_track_ids: list[str],
    track_factors: np.ndarray,
    track_id_to_index: dict[str, int],
) -> np.ndarray:
    """Aggregate the recent tracks into a single session vector (the contribution).

    We represent a listening *session* as the **mean** of its tracks' latent
    factor vectors. Rationale:

      * The latent space is additive/linear for similarity purposes (ALS scores
        are dot products), so the centroid of the session tracks is the point
        that maximises average cosine similarity to all of them — a natural
        "what is this session about" summary.
      * Mean (not sum) keeps the vector scale independent of session length, so a
        3-track and a 7-track session live on the same scale and the downstream
        cosine ranking is comparable.
      * It is the simplest defensible baseline. Weighted/recency-decayed or
        sequence-model (RNN/attention) aggregation are deferred (spec Part 4);
        mean is the control they must beat.

    Track IDs not present in the trained catalogue are skipped. If none of the
    inputs are known, raises ValueError (cold-start is out of scope, spec Part 4).
    """
    indices = [
        track_id_to_index[tid]
        for tid in recent_track_ids
        if tid in track_id_to_index
    ]
    if not indices:
        raise ValueError(
            "None of the recent_track_ids are in the trained catalogue "
            "(cold-start is out of prototype scope)."
        )
    return track_factors[indices].mean(axis=0)


def cosine_nearest(
    session_vector: np.ndarray,
    unit_factors: np.ndarray,
    k: int,
    exclude_indices: set[int],
) -> list[tuple[int, float]]:
    """Cosine top-k over the (pre-normalised) track factors, excluding the inputs.

    We use cosine, not the raw dot product. ALS factor *magnitude* encodes
    popularity/confidence; cosine discards it, so ranking is driven by taste
    similarity rather than popularity — the right choice for "next track in this
    taste cluster" (spec §1.4). `unit_factors` are unit-normalised once at load
    time (see _load_artifacts), so this is a single matrix-vector product.

    Returns [(track_index, score), ...] sorted by descending cosine similarity.
    """
    sv_norm = np.linalg.norm(session_vector)
    if sv_norm == 0:
        sv_norm = 1e-9
    unit_session = session_vector / sv_norm

    scores = unit_factors @ unit_session

    # Mask out the input tracks so we never recommend them back.
    for idx in exclude_indices:
        scores[idx] = -np.inf

    # Top-k via argpartition (cheap), then sort just those k.
    top_unsorted = np.argpartition(-scores, range(min(k, len(scores))))[:k]
    top = top_unsorted[np.argsort(-scores[top_unsorted])]
    return [(int(i), float(scores[i])) for i in top]


def recommend(
    recent_track_ids: list[str],
    k: int = 10,
    rerank: bool = True,
    rerank_weights: dict[str, float] | None = None,
) -> list[dict]:
    """Stateless next-track recommendation.

    Args:
        recent_track_ids: track IDs from the catalogue (no user identifier).
        k: number of recommendations to return.
        rerank: apply the hybrid metadata re-ranker (tag boost + artist
            diversity penalty) over a widened CF candidate pool. False gives
            the pure collaborative-filtering ranking (the Phase-0 behaviour).
        rerank_weights: optional weight overrides, see rerank.DEFAULT_WEIGHTS.

    Returns:
        List of k dicts: {track_id, artist, title, score, why, shared_tags}.
    """
    from nextrack.rerank import CANDIDATE_POOL, rerank as rerank_fn

    factors, unit_factors, maps = _load_artifacts()
    track_id_to_index = maps["track_id_to_index"]
    index_to_track_id = maps["index_to_track_id"]
    track_names = maps["track_names"]
    tags = maps["tags"]

    # Build the session vector from the RAW factors (averaging unit vectors would
    # distort the centroid); rank against the unit factors for cosine similarity.
    session_vector = build_session_vector(
        recent_track_ids, factors, track_id_to_index
    )

    exclude = {
        track_id_to_index[tid]
        for tid in recent_track_ids
        if tid in track_id_to_index
    }
    # Widen the pool when re-ranking so metadata can promote candidates that
    # pure cosine leaves just outside the top-k.
    pool = max(CANDIDATE_POOL, k) if rerank else k
    ranked = cosine_nearest(session_vector, unit_factors, pool, exclude)

    if rerank:
        candidates = [(index_to_track_id[idx], score) for idx, score in ranked]
        artist_of = {tid: track_names.get(tid, ("", ""))[0] for tid, _ in candidates}
        reranked = rerank_fn(
            candidates, recent_track_ids, tags, artist_of, k, rerank_weights
        )
        id_scores = reranked
    else:
        id_scores = [(index_to_track_id[idx], score) for idx, score in ranked[:k]]

    results: list[dict] = []
    for tid, score in id_scores:
        artist, title = track_names.get(tid, ("", ""))
        why, shared = tag_overlap_explain(recent_track_ids, tid, tags)
        results.append(
            {
                "track_id": tid,
                "artist": artist,
                "title": title,
                "score": float(score),
                "why": why,
                "shared_tags": shared,
            }
        )
    return results
