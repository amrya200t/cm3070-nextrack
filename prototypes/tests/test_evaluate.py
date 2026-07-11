"""Unit tests for the evaluation metrics and split logic (evaluate.py)."""

import pandas as pd
import pytest

from nextrack.baselines import popularity_ranking, recommend_popular
from nextrack.evaluate import leave_last_out, rank_of_target, score_rank


def test_rank_of_target():
    assert rank_of_target("b", ["a", "b", "c"]) == 2
    assert rank_of_target("z", ["a", "b", "c"]) is None


def test_score_rank_at_1_is_perfect():
    s = score_rank(1)
    assert s == {"recall": 1.0, "ndcg": 1.0, "mrr": 1.0}


def test_score_rank_outside_k_is_zero():
    assert score_rank(11, k=10) == {"recall": 0.0, "ndcg": 0.0, "mrr": 0.0}
    assert score_rank(None) == {"recall": 0.0, "ndcg": 0.0, "mrr": 0.0}


def test_score_rank_decays_with_rank():
    assert score_rank(2)["ndcg"] == pytest.approx(0.6309, abs=1e-3)  # 1/log2(3)
    assert score_rank(4)["mrr"] == pytest.approx(0.25)


def test_popularity_baseline_excludes_session():
    plays = pd.DataFrame(
        {
            "user_id": ["u"] * 3,
            "track_id": ["hit", "mid", "rare"],
            "playcount": [100, 10, 1],
        }
    )
    ranking = popularity_ranking(plays)
    assert ranking == ["hit", "mid", "rare"]
    assert recommend_popular(ranking, exclude={"hit"}, k=2) == ["mid", "rare"]


def test_leave_last_out_holds_out_final_track():
    # One user, 6 distinct tracks in time order: target must be t5,
    # session the previous five, and t5 removed from the training pairs.
    events = pd.DataFrame(
        {
            "user_id": ["u1"] * 6,
            "track_id": [f"t{i}" for i in range(6)],
            "timestamp": [f"2020-01-0{i+1}" for i in range(6)],
        }
    )
    train_plays, cases = leave_last_out(events)
    assert len(cases) == 1
    case = cases.iloc[0]
    assert case["target"] == "t5"
    assert case["session"] == ["t0", "t1", "t2", "t3", "t4"]
    assert "t5" not in set(train_plays["track_id"])
