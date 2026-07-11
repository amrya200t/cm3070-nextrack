"""Offline evaluation harness: leave-last-out split, NDCG@K / Recall@K / MRR@K.

Protocol (the "large-scale hypothesis test" of the testing strategy):
  1. Stream the same raw events as training, but KEEP timestamps.
  2. For each user, hold out the chronologically LAST distinct track as the
     target; the 5 distinct tracks before it form the evaluation session.
  3. Refit ALS on the training portion only (the held-out (user, target)
     interactions are removed), so the model never sees the answers.
  4. Rank candidates for each session with (a) the popularity baseline and
     (b) the ALS session-vector recommender, and score the target's rank.

Metrics use the single-target convention: Recall@K is the hit rate, NDCG@K is
1/log2(rank+1), MRR@K is 1/rank (0 if the target is outside the top K).

Run with `uv run evaluate`. Results land in artifacts/eval_results.json.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from nextrack.baselines import popularity_ranking, recommend_popular
from nextrack.data import (
    EVENTS_PATH,
    MIN_EVENTS_PER_USER,
    MIN_PLAYS_PER_TRACK,
    TARGET_EVENTS,
    build_csr,
)
from nextrack.train import train_als

K = 10
SESSION_LEN = 5
RESULTS_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "eval_results.json"


# ---------------------------------------------------------------- metrics ----

def rank_of_target(target: str, recommended: list[str]) -> int | None:
    """1-based rank of the target in the list, or None if absent."""
    try:
        return recommended.index(target) + 1
    except ValueError:
        return None


def score_rank(rank: int | None, k: int = K) -> dict[str, float]:
    """Single-target metrics for one evaluation case.

    With exactly one relevant item, Recall@K collapses to the hit rate and
    ideal-DCG is 1, so NDCG@K = 1/log2(rank+1).
    """
    if rank is None or rank > k:
        return {"recall": 0.0, "ndcg": 0.0, "mrr": 0.0}
    return {
        "recall": 1.0,
        "ndcg": 1.0 / np.log2(rank + 1),
        "mrr": 1.0 / rank,
    }


# ------------------------------------------------------------------ split ----

def load_events_with_time(
    path=EVENTS_PATH, max_events: int = TARGET_EVENTS
) -> pd.DataFrame:
    """Same event stream as data.load_events, but keeping the timestamp column."""
    events = pd.read_csv(
        path,
        sep="\t",
        nrows=max_events,
        usecols=[0, 1, 3],
        names=["user_id", "track_id", "timestamp"],
        header=0,
        dtype={"user_id": str, "track_id": str, "timestamp": str},
    )
    return events


def leave_last_out(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (train_plays, eval_cases).

    train_plays: [user_id, track_id, playcount] with the held-out (user, target)
    pairs fully removed, then the same catalogue filters as training.
    eval_cases: [user_id, target, session] with session = the SESSION_LEN
    distinct tracks played most recently before the target.
    """
    events = events.sort_values(["user_id", "timestamp"], kind="stable")

    cases = []
    for user_id, group in events.groupby("user_id", sort=False):
        tracks = group["track_id"].tolist()
        distinct = list(dict.fromkeys(tracks))  # order-preserving
        if len(distinct) < SESSION_LEN + 1:
            continue
        target = distinct[-1]
        session = distinct[-(SESSION_LEN + 1) : -1]
        cases.append((user_id, target, session))
    eval_cases = pd.DataFrame(cases, columns=["user_id", "target", "session"])

    # Remove held-out (user, target) interactions from the training events.
    held = set(zip(eval_cases["user_id"], eval_cases["target"]))
    keep = ~pd.Series(list(zip(events["user_id"], events["track_id"]))).isin(held).to_numpy()
    train_events = events[keep]

    plays = (
        train_events.groupby(["user_id", "track_id"], sort=False)
        .size()
        .reset_index(name="playcount")
    )
    track_totals = plays.groupby("track_id")["playcount"].transform("sum")
    plays = plays[track_totals >= MIN_PLAYS_PER_TRACK]
    user_counts = plays.groupby("user_id")["track_id"].transform("count")
    plays = plays[user_counts >= MIN_EVENTS_PER_USER]
    return plays.reset_index(drop=True), eval_cases


# ----------------------------------------------------------------- runner ----

def evaluate_popularity(train_plays, eval_cases) -> list[dict]:
    ranking = popularity_ranking(train_plays)
    scores = []
    for _, row in eval_cases.iterrows():
        recs = recommend_popular(ranking, exclude=set(row["session"]), k=K)
        scores.append(score_rank(rank_of_target(row["target"], recs)))
    return scores


def evaluate_als(train_plays, eval_cases) -> list[dict]:
    matrix, tid_to_idx, idx_to_tid = build_csr(train_plays)
    model = train_als(matrix)
    factors = model.item_factors
    norms = np.linalg.norm(factors, axis=1)
    norms[norms == 0] = 1e-9
    unit = factors / norms[:, None]

    scores = []
    for _, row in eval_cases.iterrows():
        idxs = [tid_to_idx[t] for t in row["session"] if t in tid_to_idx]
        target_idx = tid_to_idx.get(row["target"])
        if not idxs or target_idx is None:
            # Session or target fell out of the training catalogue: count as a
            # miss rather than silently dropping the case (dropping would bias
            # the comparison toward ALS on the easy, well-covered users).
            scores.append(score_rank(None))
            continue
        sv = factors[idxs].mean(axis=0)
        sv = sv / (np.linalg.norm(sv) or 1e-9)
        sims = unit @ sv
        sims[idxs] = -np.inf  # never recommend a session track back
        top = np.argpartition(-sims, K)[:K]
        top = top[np.argsort(-sims[top])]
        recs = [idx_to_tid[i] for i in top]
        scores.append(score_rank(rank_of_target(row["target"], recs)))
    return scores


def _mean(scores: list[dict]) -> dict[str, float]:
    return {
        m: float(np.mean([s[m] for s in scores])) for m in ("recall", "ndcg", "mrr")
    }


def main() -> None:
    t0 = time.time()
    print("Loading events with timestamps ...")
    events = load_events_with_time()
    print(f"  {len(events):,} events, {events['user_id'].nunique():,} users")

    print("Leave-last-out split ...")
    train_plays, eval_cases = leave_last_out(events)
    print(f"  train pairs: {len(train_plays):,}   eval cases: {len(eval_cases):,}")

    print("Evaluating popularity baseline ...")
    pop = _mean(evaluate_popularity(train_plays, eval_cases))
    print(f"  {pop}")

    print("Training + evaluating ALS session-vector model ...")
    als = _mean(evaluate_als(train_plays, eval_cases))
    print(f"  {als}")

    results = {
        "k": K,
        "session_len": SESSION_LEN,
        "n_eval_cases": int(len(eval_cases)),
        "popularity": pop,
        "als_session_vector": als,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {RESULTS_PATH}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
