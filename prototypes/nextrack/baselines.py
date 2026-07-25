"""Baselines the personalised recommender must beat.

Three tiers of opponent, weakest to strongest:

  * Popularity — the globally most-played tracks, session ignored (except to
    never recommend a seed back). The standard sanity floor: a model that
    cannot beat it is not personalising at all.
  * ContentOnly — rank by tag-profile overlap with the session, collaborative
    signal ignored. Isolates how much metadata alone can do (prelim 3.6: the
    baseline that separates the CF contribution from the tag contribution).
  * ItemKNN — cosine similarity over track co-occurrence columns. The classic
    neighbourhood method and the honest opponent promised in the Topic 6
    write-up: it uses the same interaction data as ALS, so beating it argues
    for factorisation specifically, not just for using the session.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from nextrack.explain import _norm_tags


def popularity_ranking(plays: pd.DataFrame) -> list[str]:
    """Global track ranking by total playcount, most-played first.

    `plays` is the filtered [user_id, track_id, playcount] frame from
    data.load_events (the TRAINING portion only — computing popularity on data
    that includes the evaluation targets would leak them into the ranking).
    """
    totals = plays.groupby("track_id")["playcount"].sum()
    return totals.sort_values(ascending=False).index.tolist()


def recommend_popular(
    ranking: list[str],
    exclude: set[str],
    k: int,
) -> list[str]:
    """Top-k most popular tracks, skipping the session's own tracks."""
    out: list[str] = []
    for tid in ranking:
        if tid not in exclude:
            out.append(tid)
            if len(out) == k:
                break
    return out


class ItemKNN:
    """Item-item cosine over the user x track playcount matrix.

    Columns are L2-normalised once at fit time, so `Xn.T @ Xn` is the full
    track-track cosine matrix — never materialised. A session is scored in two
    sparse mat-vecs: v = sum of the session's normalised columns (a user-space
    vector), then scores = Xn.T @ v = sum over session tracks of their cosine
    similarity to every candidate. No training, no hyperparameters: the whole
    method is this normalisation, which is why it is the honest baseline.
    """

    def __init__(self, matrix: csr_matrix) -> None:
        X = matrix.tocsc().astype(np.float32)
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=0)).ravel())
        norms[norms == 0] = 1e-9
        self._Xn = (X.multiply(1.0 / norms)).tocsc()
        self._XnT = self._Xn.T.tocsr()

    def score_session(self, track_indices: list[int]) -> np.ndarray:
        v = np.asarray(self._Xn[:, track_indices].sum(axis=1)).ravel()
        return self._XnT @ v


class ContentOnly:
    """Tag-profile Jaccard ranking; the collaborative signal is ignored.

    Fit builds a binary track x tag matrix over the normalised tag vocabulary.
    Scoring one session: profile = union of the session tracks' tags; then for
    every candidate, Jaccard = |A∩B| / (|A| + |B| - |A∩B|) — computed for the
    whole catalogue in one sparse mat-vec (the intersection term), since
    out-of-vocabulary profile tags can never intersect anyway.

    Jaccard produces heavy ties (many tracks share identical tag sets), so a
    popularity epsilon (1e-6 x normalised playcount) breaks them
    deterministically — documented, and far below any real score difference.
    """

    def __init__(
        self,
        track_ids: list[str],
        tags: dict[str, list[str]],
        playcounts: np.ndarray | None = None,
    ) -> None:
        vocab: dict[str, int] = {}
        rows: list[int] = []
        cols: list[int] = []
        sizes = np.zeros(len(track_ids), dtype=np.float32)
        for i, tid in enumerate(track_ids):
            tag_set = set(_norm_tags(tags.get(tid, [])))
            sizes[i] = len(tag_set)
            for t in tag_set:
                cols.append(vocab.setdefault(t, len(vocab)))
                rows.append(i)
        self._T = csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(len(track_ids), max(len(vocab), 1)),
        )
        self._sizes = sizes
        self._vocab = vocab
        if playcounts is None:
            self._tiebreak = np.zeros(len(track_ids), dtype=np.float32)
        else:
            self._tiebreak = 1e-6 * playcounts / (playcounts.max() or 1.0)

    def score_session(
        self, session_track_ids: list[str], tags: dict[str, list[str]]
    ) -> np.ndarray:
        profile: set[str] = set()
        for tid in session_track_ids:
            profile.update(_norm_tags(tags.get(tid, [])))
        p = np.zeros(self._T.shape[1], dtype=np.float32)
        in_vocab = [self._vocab[t] for t in profile if t in self._vocab]
        p[in_vocab] = 1.0
        inter = self._T @ p
        union = self._sizes + len(profile) - inter
        union[union == 0] = 1.0
        return inter / union + self._tiebreak
