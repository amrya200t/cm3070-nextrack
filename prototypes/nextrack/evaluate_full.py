"""Full evaluation protocol (Phase C): the prelim §3.6 commitments, delivered.

Protocol
  1. 70/10/20 chronological split PER USER: earliest 70% of a user's events
     train, next 10% validation, last 20% test. Time only moves forward — no
     future event ever informs a past prediction.
  2. VALIDATION stage: ALS fit on train; the re-ranker weights are selected on
     the validation cases (max diversity subject to NDCG >= 95% of pure CF)
     and LOCKED before the test partition is touched.
  3. TEST stage: ALS refit on train+validation; five systems scored on the
     test cases: popularity, content-only, item-kNN, ALS, ALS + re-ranker
     (locked weights). As in the leave-last-out harness, each case's
     (user, target) pair is removed from the training playcounts so a hit is
     prediction, not recall of a seen interaction.
  4. Bootstrap 95% confidence intervals (1,000 resamples over cases) on every
     metric of every system.
  5. Popularity-bucket analysis: Recall@10 by target-popularity tercile
     (head/mid/tail of the training playcount distribution) — the long-tail
     failure mode predicted in prelim §3.6, measured.

Run with `uv run evaluate-full` (several minutes; two ALS fits plus five
systems x ~10k cases). Results -> artifacts/eval_full.json.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from nextrack.baselines import ContentOnly, ItemKNN, popularity_ranking, recommend_popular
from nextrack.data import (
    MIN_EVENTS_PER_USER,
    MIN_PLAYS_PER_TRACK,
    build_csr,
    load_tags,
    load_track_names,
)
from nextrack.evaluate import (
    K,
    SESSION_LEN,
    load_events_with_time,
    rank_of_target,
    score_rank,
)
from nextrack.rerank import CANDIDATE_POOL, rerank as rerank_fn
from nextrack.seeds import SEED
from nextrack.train import train_als

RESULTS_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "eval_full.json"

WEIGHT_GRID = [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02]  # w_artist; w_tag fixed at 0.05
NDCG_BUDGET = 0.95  # reranked NDCG must stay >= 95% of pure CF (the -5% rule)
BOOTSTRAP_N = 1_000


# ------------------------------------------------------------------- split ----

def split_701020(events: pd.DataFrame):
    """Chronological 70/10/20 per user -> (train_ev, val_ev, test_ev)."""
    events = events.sort_values(["user_id", "timestamp"], kind="stable")
    pos = events.groupby("user_id", sort=False).cumcount()
    n = events.groupby("user_id", sort=False)["track_id"].transform("size")
    frac = (pos + 1) / n
    return (
        events[frac <= 0.70],
        events[(frac > 0.70) & (frac <= 0.80)],
        events[frac > 0.80],
    )


def make_cases(history_ev: pd.DataFrame, segment_ev: pd.DataFrame) -> pd.DataFrame:
    """One case per user: target = user's FIRST distinct track in the segment
    that is not already in their session; session = last SESSION_LEN distinct
    tracks of their history. Users with too little history are skipped."""
    hist = {
        uid: list(dict.fromkeys(g["track_id"]))
        for uid, g in history_ev.groupby("user_id", sort=False)
    }
    cases = []
    for uid, g in segment_ev.groupby("user_id", sort=False):
        h = hist.get(uid, [])
        if len(h) < SESSION_LEN:
            continue
        session = h[-SESSION_LEN:]
        target = next((t for t in dict.fromkeys(g["track_id"]) if t not in session), None)
        if target is None:
            continue
        cases.append((uid, target, session))
    return pd.DataFrame(cases, columns=["user_id", "target", "session"])


def training_plays(train_ev: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    """Playcounts from the training events with each case's (user, target)
    pair removed, then the standard catalogue filters (mirrors evaluate.py)."""
    held = set(zip(cases["user_id"], cases["target"]))
    keep = ~pd.Series(
        list(zip(train_ev["user_id"], train_ev["track_id"]))
    ).isin(held).to_numpy()
    ev = train_ev[keep]
    plays = (
        ev.groupby(["user_id", "track_id"], sort=False)
        .size()
        .reset_index(name="playcount")
    )
    track_totals = plays.groupby("track_id")["playcount"].transform("sum")
    plays = plays[track_totals >= MIN_PLAYS_PER_TRACK]
    user_counts = plays.groupby("user_id")["track_id"].transform("count")
    plays = plays[user_counts >= MIN_EVENTS_PER_USER]
    return plays.reset_index(drop=True)


# ------------------------------------------------------------------ systems ---

class Stage:
    """Everything the five systems need, built once per partition."""

    def __init__(self, train_plays: pd.DataFrame, cases: pd.DataFrame, tags: dict):
        self.cases = cases
        self.tags = tags
        self.matrix, self.tid_to_idx, self.idx_to_tid = build_csr(train_plays)
        self.track_ids = [self.idx_to_tid[i] for i in range(len(self.idx_to_tid))]
        self.playcounts = np.asarray(self.matrix.sum(axis=0)).ravel()
        self.pop_ranking = popularity_ranking(train_plays)
        self.knn = ItemKNN(self.matrix)
        self.content = ContentOnly(self.track_ids, tags, self.playcounts)

        model = train_als(self.matrix)
        factors = model.item_factors
        norms = np.linalg.norm(factors, axis=1)
        norms[norms == 0] = 1e-9
        self.factors = factors
        self.unit = factors / norms[:, None]

    def _topk_from_scores(self, scores: np.ndarray, exclude: list[int], k: int) -> list[str]:
        scores = scores.copy()
        scores[exclude] = -np.inf
        top = np.argpartition(-scores, k)[:k]
        top = top[np.argsort(-scores[top])]
        return [self.idx_to_tid[i] for i in top]

    def als_pool(self, session_idxs: list[int]) -> list[tuple[str, float]]:
        sv = self.factors[session_idxs].mean(axis=0)
        sv = sv / (np.linalg.norm(sv) or 1e-9)
        sims = self.unit @ sv
        sims[session_idxs] = -np.inf
        top = np.argpartition(-sims, CANDIDATE_POOL)[:CANDIDATE_POOL]
        top = top[np.argsort(-sims[top])]
        return [(self.idx_to_tid[i], float(sims[i])) for i in top]

    def build_pools(self) -> list[tuple | None]:
        """ALS candidate pools for every case, computed once — the weight sweep
        re-scores only the (cheap) greedy re-ranking, never the pools."""
        pools = []
        for _, row in self.cases.iterrows():
            session, target = row["session"], row["target"]
            idxs = [self.tid_to_idx[t] for t in session if t in self.tid_to_idx]
            if not idxs or target not in self.tid_to_idx:
                pools.append(None)
                continue
            pools.append((session, target, self.als_pool(idxs)))
        return pools

    def score_pools_arm(self, pools: list, weights: dict | None):
        """Score one ALS arm (pure CF when weights is None) from prebuilt pools."""
        artist_of = self._artist_of
        scores, divs = [], []
        for case in pools:
            if case is None:
                scores.append(score_rank(None))
                continue
            session, target, pool = case
            if weights is None:
                recs = [tid for tid, _ in pool[:K]]
            else:
                recs = [tid for tid, _ in rerank_fn(pool, session, self.tags, artist_of, K, weights)]
            divs.append(len({artist_of.get(t, t) for t in recs}))
            scores.append(score_rank(rank_of_target(target, recs)))
        return scores, (float(np.mean(divs)) if divs else 0.0)

    def run(self, weights: dict | None):
        """Score all five systems over the cases in one pass.

        Returns {system: list of per-case metric dicts} plus per-case target
        popularity (for buckets) and per-case unique-artist counts for the two
        ALS arms (diversity)."""
        out = {s: [] for s in ("popularity", "content_only", "item_knn", "als", "als_rerank")}
        target_pop: list[float] = []
        div = {"als": [], "als_rerank": []}
        artist_of = self._artist_of

        for _, row in self.cases.iterrows():
            session, target = row["session"], row["target"]
            idxs = [self.tid_to_idx[t] for t in session if t in self.tid_to_idx]
            known_target = target in self.tid_to_idx
            if not idxs or not known_target:
                for s in out:
                    out[s].append(score_rank(None))
                target_pop.append(0.0)
                continue
            target_pop.append(float(self.playcounts[self.tid_to_idx[target]]))
            exclude = idxs

            recs = recommend_popular(self.pop_ranking, set(session), K)
            out["popularity"].append(score_rank(rank_of_target(target, recs)))

            recs = self._topk_from_scores(self.content.score_session(session, self.tags), exclude, K)
            out["content_only"].append(score_rank(rank_of_target(target, recs)))

            recs = self._topk_from_scores(self.knn.score_session(idxs), exclude, K)
            out["item_knn"].append(score_rank(rank_of_target(target, recs)))

            pool = self.als_pool(idxs)
            als_recs = [tid for tid, _ in pool[:K]]
            out["als"].append(score_rank(rank_of_target(target, als_recs)))
            div["als"].append(len({artist_of.get(t, t) for t in als_recs}))

            rr_recs = [tid for tid, _ in rerank_fn(pool, session, self.tags, artist_of, K, weights)]
            out["als_rerank"].append(score_rank(rank_of_target(target, rr_recs)))
            div["als_rerank"].append(len({artist_of.get(t, t) for t in rr_recs}))

        return out, np.array(target_pop), div


# ------------------------------------------------------------- aggregation ---

def mean_metrics(scores: list[dict]) -> dict:
    return {m: float(np.mean([s[m] for s in scores])) for m in ("recall", "ndcg", "mrr")}


def bootstrap_ci(scores: list[dict], metric: str, n: int = BOOTSTRAP_N) -> tuple[float, float]:
    """Percentile bootstrap 95% CI over cases (seeded, vectorised)."""
    vals = np.array([s[metric] for s in scores])
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(vals), size=(n, len(vals)))
    means = vals[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def bucket_recall(scores: list[dict], target_pop: np.ndarray) -> dict:
    """Recall@10 by target-popularity tercile of the scored cases."""
    vals = np.array([s["recall"] for s in scores])
    scored = target_pop > 0
    if not scored.any():
        return {}
    lo_t, hi_t = np.percentile(target_pop[scored], [33.3, 66.7])
    out = {}
    for name, mask in (
        ("tail", scored & (target_pop <= lo_t)),
        ("mid", scored & (target_pop > lo_t) & (target_pop <= hi_t)),
        ("head", scored & (target_pop > hi_t)),
    ):
        out[name] = {"recall": float(vals[mask].mean()) if mask.any() else 0.0,
                     "n": int(mask.sum())}
    return out


# ------------------------------------------------------------------- runner ---

def main() -> None:
    t0 = time.time()
    print("Loading events ...")
    events = load_events_with_time()
    train_ev, val_ev, test_ev = split_701020(events)
    print(f"  events: train {len(train_ev):,} / val {len(val_ev):,} / test {len(test_ev):,}")

    print("Loading names + tags ...")
    names = load_track_names()
    artist_of = {tid: a for tid, (a, _t) in names.items()}
    tags = load_tags(names)
    Stage._artist_of = artist_of  # shared, read-only

    # ---- VALIDATION: select re-ranker weights, then lock -------------------
    val_cases = make_cases(train_ev, val_ev)
    print(f"Validation stage: {len(val_cases):,} cases")
    val_stage = Stage(training_plays(train_ev, val_cases), val_cases, tags)
    pools = val_stage.build_pools()
    base_scores, base_div = val_stage.score_pools_arm(pools, None)
    base_ndcg = mean_metrics(base_scores)["ndcg"]
    print(f"  pure CF: ndcg={base_ndcg:.5f} div={base_div:.2f}")

    best = None
    for w in WEIGHT_GRID:
        weights = {"tag": 0.05, "artist": w}
        rr_scores, d = val_stage.score_pools_arm(pools, weights)
        rr_ndcg = mean_metrics(rr_scores)["ndcg"]
        ok = rr_ndcg >= NDCG_BUDGET * base_ndcg
        print(f"  w_artist={w:<7} ndcg={rr_ndcg:.5f} ({rr_ndcg/base_ndcg-1:+.1%}) "
              f"div={d:.2f} {'PASS' if ok else 'fail'}")
        if ok and (best is None or d > best[1]):
            best = (w, d)
    locked_w = {"tag": 0.05, "artist": best[0] if best else 0.005}
    print(f"LOCKED re-ranker weights on validation: {locked_w}")

    # ---- TEST: five systems, locked weights --------------------------------
    trainval_ev = pd.concat([train_ev, val_ev])
    test_cases = make_cases(trainval_ev, test_ev)
    print(f"Test stage: {len(test_cases):,} cases")
    stage = Stage(training_plays(trainval_ev, test_cases), test_cases, tags)
    out, target_pop, div = stage.run(locked_w)

    results = {
        "protocol": "70/10/20 chronological per user; validation-locked weights",
        "k": K,
        "session_len": SESSION_LEN,
        "n_test_cases": int(len(test_cases)),
        "locked_weights": locked_w,
        "systems": {},
        "buckets_recall_at_10": {s: bucket_recall(out[s], target_pop) for s in out},
        "diversity_unique_artists_at_10": {s: round(float(np.mean(div[s])), 2) for s in div},
        "runtime_seconds": round(time.time() - t0, 1),
    }
    for s, scores in out.items():
        m = mean_metrics(scores)
        results["systems"][s] = {
            **m,
            "ci95": {metric: bootstrap_ci(scores, metric) for metric in ("recall", "ndcg", "mrr")},
        }
        print(f"  {s:<13} recall={m['recall']:.4f} ndcg={m['ndcg']:.4f} mrr={m['mrr']:.4f}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {RESULTS_PATH}  ({results['runtime_seconds']} s)")


if __name__ == "__main__":
    main()
