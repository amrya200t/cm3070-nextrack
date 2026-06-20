"""Explanation: Jaccard tag-overlap between session and candidate tracks.

Uses the top-20 tags per track; reports the top-3 shared tags in a plain-English
`why` string. If a track has no tags (the LFM-2b tag join covers ~69% of the
trained catalogue), returns an empty explanation honestly rather than
fabricating one. Spec §1.4.
"""

from __future__ import annotations

TOP_N_TAGS = 20
TOP_SHARED = 3


def _norm_tags(raw: list[str]) -> list[str]:
    """Lower-case + de-duplicate (order-preserving) the top-N tags of one track.

    LFM-2b tags are free-text, so the same concept appears in several forms:
    casing ('Progressive rock' / 'progressive rock') and a 'genre:' prefix
    ('genre: progressive rock' / 'progressive rock'). We lower-case and strip a
    leading 'genre:' so these collapse to one tag, then de-dupe. This unifies
    duplicate concepts (it does NOT drop tags or impose a controlled vocabulary).
    Order is preserved (LFM-2b emits tags count-descending), so the most salient
    shared tags still surface first.
    """
    seen: set[str] = set()
    out: list[str] = []
    for t in raw[:TOP_N_TAGS]:
        norm = t.lower().removeprefix("genre:").strip()
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def _session_tag_set(
    session_track_ids: list[str],
    tags: dict[str, list[str]],
) -> set[str]:
    """Union of the normalised top-20 tags across all session tracks."""
    session_tags: set[str] = set()
    for tid in session_track_ids:
        session_tags.update(_norm_tags(tags.get(tid, [])))
    return session_tags


def tag_overlap_explain(
    session_track_ids: list[str],
    candidate_track_id: str,
    tags: dict[str, list[str]],
) -> tuple[str, list[str]]:
    """Return (why_string, top_3_shared_tags) via tag overlap.

    Tags are lower-cased and de-duplicated on both sides before matching, so the
    overlap is case-insensitive and each shared concept appears once. Shared tags
    are ranked by the candidate's own (count-descending) order.
    """
    session_tags = _session_tag_set(session_track_ids, tags)
    candidate_tags = _norm_tags(tags.get(candidate_track_id, []))

    shared = [t for t in candidate_tags if t in session_tags][:TOP_SHARED]

    if not shared:
        return "", []

    if len(shared) == 1:
        why = f"Because you listened to {shared[0]}"
    else:
        why = "Because you listened to " + ", ".join(shared[:-1]) + f" and {shared[-1]}"
    return why, shared
