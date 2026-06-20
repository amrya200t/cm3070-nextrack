"""Data loader: LFM-2b 2020 Subset -> playcount DataFrame + user x track CSR + tag dict.

Interactions come from the 2020 Subset `listening_events` (user_id, track_id,
timestamp). For the implicit-ALS baseline we aggregate to user x track *playcount*
(the confidence weight `1 + alpha*playcount` in train.py needs counts, not raw
events). Real artist/title names come from the subset `tracks.tsv`. Tags for the
`why` field come from the Full-version `tags.json`, joined by (artist, track) name.

See prototype-implementation-spec.md §1.2, §1.5.
"""

from __future__ import annotations

import bz2
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# Resolve data paths relative to this file (prototypes/nextrack/data.py -> prototypes/data).
_DATA = Path(__file__).resolve().parent.parent / "data"
EVENTS_PATH = _DATA / "2020 Subset" / "listening_events.tsv.bz2"
TRACKS_PATH = _DATA / "2020 Subset" / "tracks.tsv.bz2"
TAGS_PATH = _DATA / "Full version" / "tags.json.bz2"

# Filter thresholds (spec §1.5). These are the levers if the catalogue is too
# thin / recommendations too within-artist — do NOT tune for the prelim (rail
# §2.1; tuning belongs in the final report's evaluation chapter):
#   - raise TARGET_EVENTS (1M -> 2-3M): more events -> more tracks survive filters
#   - lower MIN_PLAYS_PER_TRACK (5 -> 3): more long-tail tracks in the catalogue
#   - (FACTORS lives in train.py: 64 -> 128 for richer representations)
MIN_PLAYS_PER_TRACK = 5
MIN_EVENTS_PER_USER = 10
TARGET_EVENTS = 1_000_000


def load_events(
    path: Path | str = EVENTS_PATH,
    max_events: int = TARGET_EVENTS,
) -> pd.DataFrame:
    """Stream the bz2 listening-events file and return a filtered playcount frame.

    Reads up to `max_events` raw events (the file is ~300MB compressed; we cap to
    keep the prototype in memory and fast), then applies the spec filters:
      - drop tracks with < MIN_PLAYS_PER_TRACK plays
      - drop users with < MIN_EVENTS_PER_USER events

    Returns a DataFrame with columns [user_id, track_id, playcount].
    """
    rows: list[tuple[str, str]] = []
    with bz2.open(path, "rt", encoding="utf-8") as fh:
        fh.readline()  # header: user_id, track_id, album_id, timestamp
        for i, line in enumerate(fh):
            if i >= max_events:
                break
            parts = line.split("\t")
            if len(parts) >= 2:
                rows.append((parts[0], parts[1]))

    events = pd.DataFrame(rows, columns=["user_id", "track_id"])

    # Aggregate raw events -> per (user, track) playcount.
    plays = (
        events.groupby(["user_id", "track_id"], sort=False)
        .size()
        .reset_index(name="playcount")
    )

    # Filter: tracks with enough total plays, then users with enough events.
    track_plays = plays.groupby("track_id")["playcount"].transform("sum")
    plays = plays[track_plays >= MIN_PLAYS_PER_TRACK]
    user_events = plays.groupby("user_id")["track_id"].transform("count")
    plays = plays[user_events >= MIN_EVENTS_PER_USER]

    return plays.reset_index(drop=True)


def build_csr(plays: pd.DataFrame) -> tuple[csr_matrix, dict, dict]:
    """Build a user x track CSR playcount matrix from the filtered frame.

    Returns (matrix, track_id_to_index, index_to_track_id). The matrix rows are
    users, columns are tracks; values are playcounts. Inference operates on the
    track (column) factors, so the id maps are keyed on track_id.
    """
    user_codes, _ = pd.factorize(plays["user_id"], sort=False)
    track_ids = plays["track_id"].to_numpy()
    track_index, unique_tracks = pd.factorize(track_ids, sort=False)

    track_id_to_index = {tid: idx for idx, tid in enumerate(unique_tracks)}
    index_to_track_id = {idx: tid for tid, idx in track_id_to_index.items()}

    matrix = csr_matrix(
        (plays["playcount"].to_numpy(dtype=np.float32), (user_codes, track_index)),
        shape=(user_codes.max() + 1, len(unique_tracks)),
    )
    return matrix, track_id_to_index, index_to_track_id


def load_track_names(path: Path | str = TRACKS_PATH) -> dict[str, tuple[str, str]]:
    """Load the subset catalogue -> {track_id: (artist, track)}.

    Used to (a) attach real names to recommendations and (b) bridge track_id ->
    (artist, track) name for the tag join.
    """
    names: dict[str, tuple[str, str]] = {}
    with bz2.open(path, "rt", encoding="utf-8") as fh:
        fh.readline()  # header: track_id, artist, track
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                names[parts[0]] = (parts[1], parts[2])
    return names


def load_tags(
    track_names: dict[str, tuple[str, str]],
    path: Path | str = TAGS_PATH,
) -> dict[str, list[str]]:
    """Load Last.fm tags and key them by track_id (only for tracks we have).

    The tag file is keyed by (artist, track) name; we invert `track_names` to map
    those back to track_id, so explain.py can look up tags by the same track_id
    the recommender returns. Tags are stored top-first (the JSON orders them by
    count). Returns {track_id: [tag, ...]}.
    """
    name_to_track_id = {name: tid for tid, name in track_names.items()}
    tags_by_track: dict[str, list[str]] = {}
    with bz2.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (obj["_id"]["artist"], obj["_id"]["track"])
            tid = name_to_track_id.get(key)
            if tid is None:
                continue
            # JSON preserves insertion order; tags are emitted count-descending.
            tags_by_track[tid] = list(obj.get("tags", {}).keys())
    return tags_by_track
