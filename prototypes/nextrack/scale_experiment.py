"""Data scale-up experiment (Phase F2): how do catalogue coverage and ranking
quality move as training data grows from the 1M prototype slice to the full
2020 Subset?

Motivation: the full evaluation (Ch5 §5.5) showed the dominant failure mode is
COVERAGE — four-fifths of test targets were absent from the 1M-slice catalogue
entirely, while within-catalogue ranking was essentially flat across
popularity levels. If that diagnosis is right, adding data should grow the
catalogue and lift recall roughly in proportion to coverage, without any
change to the model. This script measures exactly that.

Protocol per scale (1M / 5M / 20M / full events):
  * leave-last-out split with the same semantics as evaluate.leave_last_out
    (target = user's last first-appearing distinct track; session = previous
    SESSION_LEN distinct tracks; held (user, target) pairs removed from
    training before the catalogue filters) — reimplemented here with a
    vectorised anti-join so the full-scale run stays in memory.
  * one seeded ALS fit (same hyperparameters as training: train.train_als),
    popularity baseline + pure-CF arm scored via evaluate.score_pools.
  * recorded: raw events, surviving pairs/users/tracks, eval cases, share of
    targets inside the training catalogue (the coverage number), Recall/NDCG/
    MRR@10 for both arms, and wall-clock times.

Run with `uv run --no-sync python -m nextrack.scale_experiment` (the full step
is the slow one; the whole sweep is expected to take well under an hour).
Results -> artifacts/scaling.json and figures/scaling.png.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from nextrack.data import (
    EVENTS_PATH,
    MIN_EVENTS_PER_USER,
    MIN_PLAYS_PER_TRACK,
)
from nextrack.evaluate import (
    SESSION_LEN,
    build_candidate_pools,
    evaluate_popularity,
    score_pools,
)

RESULTS_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "scaling.json"
FIGURE_PATH = Path(__file__).resolve().parent.parent / "figures" / "scaling.png"

SCALES: list[int | None] = [1_000_000, 5_000_000, 20_000_000, None]  # None = full file


def load_events_int(max_events: int | None) -> pd.DataFrame:
    """Events with timestamps, ids as int64 (memory: ~24 bytes/row vs ~150 as str).

    ISO timestamps sort correctly as strings, but we parse to int64 seconds so
    the full-scale sort and groupby stay cheap.
    """
    frames = []
    reader = pd.read_csv(
        EVENTS_PATH,
        sep="\t",
        usecols=[0, 1, 3],
        names=["user_id", "track_id", "timestamp"],
        header=0,
        dtype={"user_id": np.int64, "track_id": np.int64, "timestamp": str},
        nrows=max_events,
        chunksize=5_000_000,
    )
    for chunk in reader:
        chunk["timestamp"] = (
            pd.to_datetime(chunk["timestamp"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
            .astype("int64")
        )
        frames.append(chunk)
    return pd.concat(frames, ignore_index=True)


def split_leave_last_out(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same semantics as evaluate.leave_last_out, vectorised where it matters."""
    events = events.sort_values(["user_id", "timestamp"], kind="stable")

    cases = []
    for user_id, group in events.groupby("user_id", sort=False):
        distinct = list(dict.fromkeys(group["track_id"].tolist()))
        if len(distinct) < SESSION_LEN + 1:
            continue
        cases.append((user_id, distinct[-1], distinct[-(SESSION_LEN + 1) : -1]))
    eval_cases = pd.DataFrame(cases, columns=["user_id", "target", "session"])

    # Vectorised removal of held (user, target) pairs (anti-join, no 50M zip list).
    held = eval_cases[["user_id", "target"]].rename(columns={"target": "track_id"})
    merged = events.merge(held, on=["user_id", "track_id"], how="left", indicator=True)
    train_events = merged[merged["_merge"] == "left_only"].drop(columns="_merge")

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


def mean_metrics(scores: list[dict]) -> dict[str, float]:
    return {m: float(np.mean([s[m] for s in scores])) for m in ("recall", "ndcg", "mrr")}


def run_scale(max_events: int | None) -> dict:
    label = "full" if max_events is None else f"{max_events // 1_000_000}M"
    t0 = time.time()
    print(f"\n=== scale {label}: loading events ...", flush=True)
    events = load_events_int(max_events)
    n_raw = len(events)
    print(f"  {n_raw:,} events loaded ({time.time() - t0:.0f}s); splitting ...", flush=True)

    train_plays, eval_cases = split_leave_last_out(events)
    del events
    catalogue = set(train_plays["track_id"].unique())
    in_cat = eval_cases["target"].isin(catalogue)
    t_split = time.time()
    print(
        f"  pairs {len(train_plays):,} · users {train_plays['user_id'].nunique():,} · "
        f"tracks {len(catalogue):,} · cases {len(eval_cases):,} · "
        f"target coverage {in_cat.mean():.1%} ({time.time() - t0:.0f}s)",
        flush=True,
    )

    pop_scores = evaluate_popularity(train_plays, eval_cases)
    pools = build_candidate_pools(train_plays, eval_cases)  # one seeded ALS fit
    als_scores, _ = score_pools(pools, rerank=False)
    t_done = time.time()

    result = {
        "scale": label,
        "raw_events": int(n_raw),
        "pairs": int(len(train_plays)),
        "users": int(train_plays["user_id"].nunique()),
        "catalogue_tracks": int(len(catalogue)),
        "eval_cases": int(len(eval_cases)),
        "target_coverage": round(float(in_cat.mean()), 4),
        "popularity": mean_metrics(pop_scores),
        "als": mean_metrics(als_scores),
        "split_seconds": round(t_split - t0, 1),
        "train_eval_seconds": round(t_done - t_split, 1),
    }
    a = result["als"]
    print(
        f"  ALS recall={a['recall']:.4f} ndcg={a['ndcg']:.4f} mrr={a['mrr']:.4f} "
        f"({t_done - t0:.0f}s total)",
        flush=True,
    )
    return result


def make_figure(results: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [r["raw_events"] for r in results]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(x, [r["catalogue_tracks"] for r in results], "o-", color="#1f77b4", label="Catalogue tracks")
    ax1.set_xscale("log")
    ax1.set_xlabel("Training events")
    ax1.set_ylabel("Catalogue tracks", color="#1f77b4")
    ax1b = ax1.twinx()
    ax1b.plot(x, [r["target_coverage"] * 100 for r in results], "s--", color="#d62728", label="Target coverage")
    ax1b.set_ylabel("Eval targets in catalogue (%)", color="#d62728")
    ax1.set_title("Catalogue and coverage vs training data")

    ax2.plot(x, [r["als"]["recall"] for r in results], "o-", label="ALS Recall@10")
    ax2.plot(x, [r["popularity"]["recall"] for r in results], "s--", label="Popularity Recall@10")
    ax2.set_xscale("log")
    ax2.set_xlabel("Training events")
    ax2.set_ylabel("Recall@10")
    ax2.set_title("Ranking quality vs training data")
    ax2.legend()

    fig.suptitle("NextTrack data scale-up (leave-last-out, seeded ALS, identical hyperparameters)")
    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150)
    print(f"figure saved: {FIGURE_PATH}", flush=True)


def main() -> None:
    results = []
    for scale in SCALES:
        results.append(run_scale(scale))
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(results, indent=2))  # checkpoint each scale
    make_figure(results)
    print(f"\nSaved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
