"""Full-scale (30.4M events) training + evaluation for the final report (F2).

Everything writes to SEPARATE paths so the pinned 1M artifacts and results the
draft report cites stay untouched:
  * serving artifacts -> prototypes/artifacts_full/   (v2 image content)
  * protocol results  -> prototypes/artifacts/eval_full_30m.json

`python -m nextrack.full_scale train` : load full events -> fit ALS (locked
hyperparameters, seeded) -> save artifacts -> re-apply the existing Wikidata
overlay cache to the enlarged catalogue (no new harvesting: only the 5,110
already-cached artists are matched against the new untagged tracks).

`python -m nextrack.full_scale eval` : the same 70/10/20 five-system protocol
as evaluate_full, at full scale. Track/user ids are kept as int64 in the event
frame for memory (30M rows of Python strings would need several GB); the name
and tag dictionaries are re-keyed to int at the boundary so every lookup stays
type-consistent. Re-ranker weights are re-selected on the validation partition
at this scale, per the protocol, and locked before the test partition is
scored.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from nextrack import data, enrich
from nextrack import train as trainmod
from nextrack.evaluate import K
from nextrack.evaluate_full import (
    NDCG_BUDGET,
    WEIGHT_GRID,
    Stage,
    bootstrap_ci,
    bucket_recall,
    make_cases,
    mean_metrics,
    split_701020,
    training_plays,
)
from nextrack.scale_experiment import load_events_int

ARTIFACTS_FULL = Path(__file__).resolve().parent.parent / "artifacts_full"
EVAL_RESULTS_PATH = (
    Path(__file__).resolve().parent.parent / "artifacts" / "eval_full_30m.json"
)


def train_full() -> None:
    t0 = time.time()
    print("Loading + filtering FULL listening events (chunked) ...", flush=True)
    plays = data.load_events(max_events=None)
    print(f"  filtered playcount rows: {len(plays):,} ({time.time()-t0:.0f}s)", flush=True)

    matrix, tid_to_idx, idx_to_tid = data.build_csr(plays)
    print(f"  matrix: {matrix.shape}, nnz {matrix.nnz:,}", flush=True)

    track_names = data.load_track_names()
    tags = data.load_tags(track_names)
    print(f"  names {len(track_names):,} · tagged tracks {len(tags):,}", flush=True)

    model = trainmod.train_als(matrix)
    track_plays = plays.groupby("track_id")["playcount"].sum().to_dict()
    trainmod.save_artifacts(
        model, idx_to_tid, tid_to_idx, track_names, tags,
        artifacts_dir=ARTIFACTS_FULL, track_plays=track_plays,
    )
    print(f"artifacts saved to {ARTIFACTS_FULL}", flush=True)

    # Re-apply the existing Wikidata cache to the enlarged catalogue (no network).
    in_model = set(tid_to_idx)
    names_subset = {t: track_names[t] for t in in_model if t in track_names}
    tags_subset = {t: tags[t] for t in in_model if t in tags}
    cache = enrich.load_cache()
    labels = (
        json.loads(enrich.GENRE_LABELS_PATH.read_text(encoding="utf-8"))
        if enrich.GENRE_LABELS_PATH.exists()
        else {}
    )
    overlay = enrich.build_overlay(names_subset, tags_subset, cache, labels)
    (ARTIFACTS_FULL / "enrichment_tags.json").write_text(
        json.dumps(overlay, ensure_ascii=False), encoding="utf-8"
    )
    n = len(in_model)
    before = sum(1 for t in in_model if tags_subset.get(t))
    after = before + sum(1 for t in overlay if t in in_model)
    print(
        f"tag coverage at full scale: {before/n:.1%} -> {after/n:.1%} "
        f"(+{len(overlay):,} tracks from cached Wikidata artists)",
        flush=True,
    )
    print(f"done in {time.time()-t0:.0f}s", flush=True)


def eval_full_scale() -> None:
    t0 = time.time()
    print("Loading FULL events (int ids) ...", flush=True)
    events = load_events_int(None)
    train_ev, val_ev, test_ev = split_701020(events)
    print(
        f"  events: train {len(train_ev):,} / val {len(val_ev):,} / test {len(test_ev):,} "
        f"({time.time()-t0:.0f}s)",
        flush=True,
    )

    names = data.load_track_names()
    artist_of = {int(tid): a for tid, (a, _t) in names.items()}
    tags = {int(tid): v for tid, v in data.load_tags(names).items()}
    Stage._artist_of = artist_of

    val_cases = make_cases(train_ev, val_ev)
    print(f"Validation stage: {len(val_cases):,} cases", flush=True)
    val_stage = Stage(training_plays(train_ev, val_cases), val_cases, tags)
    pools = val_stage.build_pools()
    base_scores, base_div = val_stage.score_pools_arm(pools, None)
    base_ndcg = mean_metrics(base_scores)["ndcg"]
    print(f"  pure CF: ndcg={base_ndcg:.5f} div={base_div:.2f}", flush=True)

    best = None
    for w in WEIGHT_GRID:
        weights = {"tag": 0.05, "artist": w}
        rr_scores, d = val_stage.score_pools_arm(pools, weights)
        rr_ndcg = mean_metrics(rr_scores)["ndcg"]
        ok = rr_ndcg >= NDCG_BUDGET * base_ndcg
        print(
            f"  w_artist={w:<7} ndcg={rr_ndcg:.5f} ({rr_ndcg/base_ndcg-1:+.1%}) "
            f"div={d:.2f} {'PASS' if ok else 'fail'}",
            flush=True,
        )
        if ok and (best is None or d > best[1]):
            best = (w, d)
    locked_w = {"tag": 0.05, "artist": best[0] if best else 0.005}
    print(f"LOCKED weights on validation at full scale: {locked_w}", flush=True)

    trainval_ev = pd.concat([train_ev, val_ev])
    del events, train_ev, val_ev
    test_cases = make_cases(trainval_ev, test_ev)
    print(f"Test stage: {len(test_cases):,} cases", flush=True)
    stage = Stage(training_plays(trainval_ev, test_cases), test_cases, tags)
    out, target_pop, div = stage.run(locked_w)

    results = {
        "protocol": "70/10/20 chronological per user; validation-locked weights; FULL 30.4M events",
        "k": K,
        "n_test_cases": int(len(test_cases)),
        "locked_weights": locked_w,
        "systems": {},
        "buckets_recall_at_10": {s: bucket_recall(out[s], target_pop) for s in out},
        "diversity_unique_artists_at_10": {
            s: round(float(np.mean(div[s])), 2) for s in div
        },
        "runtime_seconds": round(time.time() - t0, 1),
    }
    for s, scores in out.items():
        m = mean_metrics(scores)
        results["systems"][s] = {
            **m,
            "ci95": {metric: bootstrap_ci(scores, metric) for metric in ("recall", "ndcg", "mrr")},
        }
        print(
            f"  {s:<13} recall={m['recall']:.4f} ndcg={m['ndcg']:.4f} mrr={m['mrr']:.4f}",
            flush=True,
        )

    EVAL_RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {EVAL_RESULTS_PATH}  ({results['runtime_seconds']} s)", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    if mode == "train":
        train_full()
    elif mode == "eval":
        eval_full_scale()
    else:
        raise SystemExit(f"unknown mode {mode!r}: use 'train' or 'eval'")
