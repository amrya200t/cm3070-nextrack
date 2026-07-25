"""Unit tests for the full evaluation protocol (Phase C7): new baselines,
split invariants, bootstrap sanity. All pure — no data files, no training."""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from nextrack.baselines import ContentOnly, ItemKNN
from nextrack.evaluate_full import bootstrap_ci, make_cases, split_701020


# ------------------------------------------------------------------ ItemKNN --

def test_itemknn_scores_colistened_track_highest():
    # Users 0+1 both play tracks 0 and 1 together; track 2 is played by an
    # unrelated user. A session containing track 0 must score track 1 above 2.
    matrix = csr_matrix(np.array([
        [3.0, 2.0, 0.0],
        [1.0, 4.0, 0.0],
        [0.0, 0.0, 5.0],
    ]))
    knn = ItemKNN(matrix)
    scores = knn.score_session([0])
    assert scores[1] > scores[2]
    # And the session track itself has the highest self-similarity (cos=1).
    assert scores[0] == pytest.approx(1.0, abs=1e-5)


# --------------------------------------------------------------- ContentOnly --

def test_content_only_ranks_by_tag_jaccard():
    tags = {
        "s":  ["shoegaze", "dream pop"],
        "c1": ["shoegaze", "dream pop"],   # perfect overlap
        "c2": ["shoegaze", "metal"],       # half overlap
        "c3": ["metal"],                   # none
    }
    co = ContentOnly(["c1", "c2", "c3"], tags)
    scores = co.score_session(["s"], tags)
    assert scores[0] > scores[1] > scores[2]
    assert scores[0] == pytest.approx(1.0, abs=1e-5)


def test_content_only_popularity_breaks_ties():
    tags = {"s": ["rock"], "c1": ["rock"], "c2": ["rock"]}
    co = ContentOnly(["c1", "c2"], tags, playcounts=np.array([1.0, 100.0]))
    scores = co.score_session(["s"], tags)
    assert scores[1] > scores[0]              # more popular wins the tie...
    assert scores[1] - scores[0] < 1e-3       # ...by an epsilon only


# -------------------------------------------------------------------- split --

def _events(uid, tracks):
    return pd.DataFrame({
        "user_id": [uid] * len(tracks),
        "track_id": tracks,
        "timestamp": [f"2020-01-{i+1:02d}" for i in range(len(tracks))],
    })


def test_split_701020_is_chronological_per_user():
    ev = pd.concat([_events("u1", [f"t{i}" for i in range(10)]),
                    _events("u2", [f"x{i}" for i in range(20)])])
    train, val, test = split_701020(ev)
    for uid in ("u1", "u2"):
        tr = train[train.user_id == uid]["timestamp"]
        va = val[val.user_id == uid]["timestamp"]
        te = test[test.user_id == uid]["timestamp"]
        assert tr.max() < va.min() <= va.max() < te.min()
    # 70/10/20 proportions hold exactly for u2's 20 events: 14 / 2 / 4.
    assert [len(x[x.user_id == "u2"]) for x in (train, val, test)] == [14, 2, 4]


def test_make_cases_target_is_first_new_distinct_track():
    hist = _events("u1", ["a", "b", "c", "d", "e", "f"])
    seg = _events("u1", ["f", "g", "h"])  # 'f' is in the session -> skip to 'g'
    cases = make_cases(hist, seg)
    assert len(cases) == 1
    case = cases.iloc[0]
    assert case["session"] == ["b", "c", "d", "e", "f"]  # last 5 distinct
    assert case["target"] == "g"


def test_make_cases_skips_thin_history():
    hist = _events("u1", ["a", "b"])  # < SESSION_LEN distinct tracks
    seg = _events("u1", ["c"])
    assert len(make_cases(hist, seg)) == 0


# ---------------------------------------------------------------- bootstrap --

def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    scores = [{"recall": v} for v in ([1.0] * 30 + [0.0] * 70)]
    lo, hi = bootstrap_ci(scores, "recall", n=500)
    assert lo < 0.30 < hi
    assert (lo, hi) == bootstrap_ci(scores, "recall", n=500)  # seeded
