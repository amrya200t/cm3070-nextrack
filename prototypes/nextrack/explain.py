"""Explanation: Jaccard tag-overlap between session and candidate tracks.

Uses the top-20 tags per track; reports the top-3 shared tags in a plain-English
`why` string. If a track has no tags (the LFM-2b tag join covers ~half the
catalogue), returns an empty explanation honestly rather than fabricating one.
Spec §1.4.
"""

from __future__ import annotations

TOP_N_TAGS = 20
TOP_SHARED = 3


def _session_tag_set(
    session_track_ids: list[str],
    tags: dict[str, list[str]],
) -> set[str]:
    """Union of the top-20 tags across all session tracks."""
    session_tags: set[str] = set()
    for tid in session_track_ids:
        session_tags.update(tags.get(tid, [])[:TOP_N_TAGS])
    return session_tags


def tag_overlap_explain(
    session_track_ids: list[str],
    candidate_track_id: str,
    tags: dict[str, list[str]],
) -> tuple[str, list[str]]:
    """Return (why_string, top_3_shared_tags) via tag overlap.

    Shared tags are ranked by the candidate's own tag order (which is
    count-descending in LFM-2b), so the most salient shared tags surface first.
    """
    session_tags = _session_tag_set(session_track_ids, tags)
    candidate_tags = tags.get(candidate_track_id, [])[:TOP_N_TAGS]

    shared = [t for t in candidate_tags if t in session_tags][:TOP_SHARED]

    if not shared:
        return "", []

    if len(shared) == 1:
        why = f"Because you listened to {shared[0]}"
    else:
        why = "Because you listened to " + ", ".join(shared[:-1]) + f" and {shared[-1]}"
    return why, shared
