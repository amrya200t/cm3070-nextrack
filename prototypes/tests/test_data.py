"""Unit tests for the CSR construction (data.py). Pure — no LFM-2b files needed."""

import pandas as pd

from nextrack.data import build_csr


def _plays():
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u2"],
            "track_id": ["a", "b", "a", "b", "c"],
            "playcount": [3, 1, 2, 5, 7],
        }
    )


def test_csr_shape_and_values():
    matrix, tid_to_idx, _ = build_csr(_plays())
    assert matrix.shape == (2, 3)  # 2 users x 3 tracks
    # u2 played track c 7 times; the cell must carry the playcount.
    assert matrix[1, tid_to_idx["c"]] == 7.0
    assert matrix[0, tid_to_idx["a"]] == 3.0


def test_id_maps_are_inverse_of_each_other():
    _, tid_to_idx, idx_to_tid = build_csr(_plays())
    assert len(tid_to_idx) == len(idx_to_tid) == 3
    for tid, idx in tid_to_idx.items():
        assert idx_to_tid[idx] == tid
