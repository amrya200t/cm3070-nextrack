"""Enrichment impact measurement (Phase F1 close-out): does the Wikidata
overlay actually help, or is it only a coverage statistic?

The overlay lifts tag coverage 68.8% -> 86.4%, but coverage is an input, not
an outcome. This runner measures the outcome: the two TAG-DEPENDENT systems
(content-only baseline, ALS + re-ranker) scored on the SAME test cases, same
ALS fit, same candidate pools — once with the baseline Last.fm tags and once
with `enrich.merged_tags` (Last.fm + Wikidata overlay for untagged tracks).
Popularity, item-kNN and pure ALS never read tags, so they cannot move and
are not re-run.

Protocol notes:
  * Split, case construction and held-pair removal are imported from
    evaluate_full — byte-identical protocol, no drift.
  * Re-ranker weights are NOT re-selected: the validation-locked weights are
    read from artifacts/eval_full.json. Re-tuning on the enriched tags would
    conflate "better tags" with "better weights".
  * One ALS fit on train+validation, one pool build; both tag variants are
    scored from the same pools, so the delta is attributable to tags alone.

Run with `uv run evaluate-enrich` (needs artifacts/enrichment_tags.json from
`uv run enrich` and artifacts/eval_full.json from `uv run evaluate-full`).
Results -> artifacts/eval_enrich.json.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from nextrack.baselines import ContentOnly
from nextrack.data import load_tags, load_track_names
from nextrack.enrich import OVERLAY_PATH, merged_tags
from nextrack.evaluate import K, load_events_with_time, rank_of_target, score_rank
from nextrack.evaluate_full import (
    RESULTS_PATH as FULL_RESULTS_PATH,
    Stage,
    bootstrap_ci,
    make_cases,
    mean_metrics,
    split_701020,
    training_plays,
)
from nextrack.rerank import rerank as rerank_fn

RESULTS_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "eval_enrich.json"


def score_content(stage: Stage, tags: dict) -> list[dict]:
    """Content-only arm over the stage's cases with an arbitrary tag dict."""
    content = ContentOnly(stage.track_ids, tags, stage.playcounts)
    scores = []
    for _, row in stage.cases.iterrows():
        session, target = row["session"], row["target"]
        idxs = [stage.tid_to_idx[t] for t in session if t in stage.tid_to_idx]
        if not idxs or target not in stage.tid_to_idx:
            scores.append(score_rank(None))
            continue
        recs = stage._topk_from_scores(content.score_session(session, tags), idxs, K)
        scores.append(score_rank(rank_of_target(target, recs)))
    return scores


def score_rerank(stage: Stage, pools: list, tags: dict, weights: dict):
    """ALS + re-ranker arm from prebuilt pools with an arbitrary tag dict."""
    artist_of = stage._artist_of
    scores, divs = [], []
    for case in pools:
        if case is None:
            scores.append(score_rank(None))
            continue
        session, target, pool = case
        recs = [tid for tid, _ in rerank_fn(pool, session, tags, artist_of, K, weights)]
        divs.append(len({artist_of.get(t, t) for t in recs}))
        scores.append(score_rank(rank_of_target(target, recs)))
    return scores, (float(np.mean(divs)) if divs else 0.0)


def summarize(scores: list[dict]) -> dict:
    return {
        **mean_metrics(scores),
        "ci95": {m: bootstrap_ci(scores, m) for m in ("recall", "ndcg", "mrr")},
    }


def main() -> None:
    t0 = time.time()
    if not OVERLAY_PATH.exists():
        raise SystemExit(f"missing {OVERLAY_PATH} — run `uv run enrich` first")
    if not FULL_RESULTS_PATH.exists():
        raise SystemExit(f"missing {FULL_RESULTS_PATH} — run `uv run evaluate-full` first")

    locked_w = json.loads(FULL_RESULTS_PATH.read_text())["locked_weights"]
    overlay = json.loads(OVERLAY_PATH.read_text(encoding="utf-8"))
    print(f"locked weights from eval_full: {locked_w}; overlay tracks: {len(overlay):,}")

    print("Loading events ...")
    events = load_events_with_time()
    train_ev, val_ev, test_ev = split_701020(events)

    names = load_track_names()
    artist_of = {tid: a for tid, (a, _t) in names.items()}
    base_tags = load_tags(names)
    rich_tags = merged_tags(base_tags, overlay)
    Stage._artist_of = artist_of

    trainval_ev = pd.concat([train_ev, val_ev])
    test_cases = make_cases(trainval_ev, test_ev)
    print(f"Test stage: {len(test_cases):,} cases (one ALS fit, shared pools)")
    stage = Stage(training_plays(trainval_ev, test_cases), test_cases, base_tags)
    pools = stage.build_pools()

    in_cat = [tid for tid in stage.track_ids]
    cov = lambda t: sum(1 for tid in in_cat if t.get(tid)) / len(in_cat)  # noqa: E731
    print(f"catalogue tag coverage: baseline {cov(base_tags):.1%} -> enriched {cov(rich_tags):.1%}")

    results = {
        "protocol": "same test stage as eval_full; tag-dependent arms only",
        "locked_weights": locked_w,
        "n_test_cases": int(len(test_cases)),
        "coverage": {"baseline": round(cov(base_tags), 4), "enriched": round(cov(rich_tags), 4)},
        "systems": {},
    }

    for label, tags in (("baseline", base_tags), ("enriched", rich_tags)):
        c_scores = score_content(stage, tags)
        r_scores, r_div = score_rerank(stage, pools, tags, locked_w)
        results["systems"][f"content_only_{label}"] = summarize(c_scores)
        results["systems"][f"als_rerank_{label}"] = {
            **summarize(r_scores),
            "diversity_unique_artists_at_10": round(r_div, 2),
        }
        for s in (f"content_only_{label}", f"als_rerank_{label}"):
            m = results["systems"][s]
            print(f"  {s:<22} recall={m['recall']:.4f} ndcg={m['ndcg']:.4f} mrr={m['mrr']:.4f}")

    for arm in ("content_only", "als_rerank"):
        b, e = results["systems"][f"{arm}_baseline"], results["systems"][f"{arm}_enriched"]
        results[f"{arm}_ndcg_delta_pct"] = round((e["ndcg"] / b["ndcg"] - 1) * 100, 2) if b["ndcg"] else None

    results["runtime_seconds"] = round(time.time() - t0, 1)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {RESULTS_PATH}  ({results['runtime_seconds']} s)")


if __name__ == "__main__":
    main()
