"""Baselines the personalised recommender must beat.

Popularity: recommend the globally most-played tracks, ignoring the session
entirely (except to never recommend a seed back). This is the standard control
in recommender evaluation — any model that cannot beat it is not personalising.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def popularity_ranking(plays: pd.DataFrame) -> list[str]:
    """Global track ranking by total playcount, most-played first.

    `plays` is the filtered [user_id, track_id, playcount] frame from
    data.load_events (the TRAINING portion only — computing popularity on data
    that includes the evaluation targets would leak them into the ranking).
    """
    totals = plays.groupby("track_id")["playcount"].sum()
    return totals.sort_values(ascending=False).index.tolist()


def recommend_popular(
    ranking: list[str],
    exclude: set[str],
    k: int,
) -> list[str]:
    """Top-k most popular tracks, skipping the session's own tracks."""
    out: list[str] = []
    for tid in ranking:
        if tid not in exclude:
            out.append(tid)
            if len(out) == k:
                break
    return out
