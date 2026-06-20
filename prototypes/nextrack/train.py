"""Model trainer: fit implicit ALS on the user x track CSR, save artifacts.

Locked hyperparams (spec §1.4): factors=64, iterations=15, regularization=0.01,
confidence alpha=40, seed from seeds.SEED.

OPENBLAS_NUM_THREADS=1 is set before numpy/implicit import: implicit runs its
own parallelism and a competing OpenBLAS threadpool causes severe slowdowns
(warned by implicit at fit time).
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pickle
from pathlib import Path

import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

from nextrack.seeds import SEED

FACTORS = 64
ITERATIONS = 15
REGULARIZATION = 0.01
ALPHA = 40

_ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
FACTORS_PATH = _ARTIFACTS / "track_factors.npy"
IDMAP_PATH = _ARTIFACTS / "id_map.pkl"


def train_als(matrix: csr_matrix) -> AlternatingLeastSquares:
    """Fit implicit ALS on the user x track matrix with the locked hyperparams.

    Confidence weighting `1 + alpha*playcount` is applied by scaling the matrix
    (Hu et al. 2008): implicit treats the values as confidence, so multiplying
    playcounts by ALPHA gives the standard `C = 1 + alpha*r` once implicit adds
    its implicit +1 baseline.
    """
    model = AlternatingLeastSquares(
        factors=FACTORS,
        iterations=ITERATIONS,
        regularization=REGULARIZATION,
        random_state=SEED,
    )
    model.fit((matrix * ALPHA).tocsr(), show_progress=True)
    return model


def save_artifacts(
    model: AlternatingLeastSquares,
    index_to_track_id: dict,
    track_id_to_index: dict,
    track_names: dict,
    tags: dict,
    artifacts_dir: Path | str = _ARTIFACTS,
) -> None:
    """Persist track factors + everything inference needs.

    Saves track_factors.npy (the item factors) and id_map.pkl (the index<->id
    maps plus names + tags), so infer.py can run without re-reading the raw data.
    """
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    np.save(artifacts_dir / "track_factors.npy", model.item_factors)

    # Subset names + tags to only the tracks actually in the model. The full
    # catalogue is ~4M names / ~1M tag lists; the model holds ~36k tracks, so
    # storing everything bloats id_map.pkl to ~400MB for no benefit.
    in_model = set(index_to_track_id.values())
    names_subset = {tid: track_names[tid] for tid in in_model if tid in track_names}
    tags_subset = {tid: tags[tid] for tid in in_model if tid in tags}

    with open(artifacts_dir / "id_map.pkl", "wb") as fh:
        pickle.dump(
            {
                "index_to_track_id": index_to_track_id,
                "track_id_to_index": track_id_to_index,
                "track_names": names_subset,
                "tags": tags_subset,
            },
            fh,
        )
