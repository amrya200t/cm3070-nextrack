"""Unit tests for the session-vector + cosine ranking core (infer.py).

These cover the pure functions only; recommend() needs trained artifacts and is
exercised by the notebook demo and the evaluation harness instead.
"""

import numpy as np
import pytest

from nextrack.infer import build_session_vector, cosine_nearest


def _factors():
    # 4 tracks in a tiny 2-D latent space, chosen so geometry is obvious:
    # t0 and t1 point the same way, t2 is orthogonal, t3 is opposite.
    return np.array(
        [
            [1.0, 0.0],   # t0
            [1.0, 0.0],   # t1
            [0.0, 1.0],   # t2
            [-1.0, 0.0],  # t3
        ]
    )


ID_TO_INDEX = {"t0": 0, "t1": 1, "t2": 2, "t3": 3}


def test_session_vector_is_centroid():
    # Mean of t0 [1,0] and t2 [0,1] must be their centroid [0.5, 0.5].
    sv = build_session_vector(["t0", "t2"], _factors(), ID_TO_INDEX)
    assert np.allclose(sv, [0.5, 0.5])


def test_session_vector_skips_unknown_ids():
    # Unknown IDs are dropped, not zero-filled: the vector must equal t0 alone.
    sv = build_session_vector(["t0", "nope"], _factors(), ID_TO_INDEX)
    assert np.allclose(sv, [1.0, 0.0])


def test_session_vector_all_unknown_raises():
    with pytest.raises(ValueError):
        build_session_vector(["x", "y"], _factors(), ID_TO_INDEX)


def test_cosine_nearest_ranks_by_similarity_and_excludes_inputs():
    factors = _factors()
    unit = factors / np.linalg.norm(factors, axis=1)[:, None]
    session = np.array([1.0, 0.0])

    ranked = cosine_nearest(session, unit, k=3, exclude_indices={0})

    indices = [i for i, _ in ranked]
    assert 0 not in indices            # input track never recommended back
    assert indices[0] == 1             # identical direction ranks first
    assert indices[-1] == 3            # opposite direction ranks last
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)


def test_cosine_scores_are_cosine_not_dot():
    # A track with a huge-magnitude factor must NOT outrank a perfectly aligned
    # one: cosine discards magnitude (that is the design decision under test).
    factors = np.array([[1.0, 0.0], [0.0, 100.0]])
    unit = factors / np.linalg.norm(factors, axis=1)[:, None]
    ranked = cosine_nearest(np.array([1.0, 0.0]), unit, k=2, exclude_indices=set())
    assert ranked[0][0] == 0
    assert ranked[0][1] == pytest.approx(1.0)
