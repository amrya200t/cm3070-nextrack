"""Wikidata artist-genre enrichment (Phase F1): backfill tags for untagged tracks.

31% of the catalogue (11,193 tracks) has no Last.fm tags, which blanks the
`why` explanation, the content-only baseline, and the re-ranker's tag term for
those tracks. This module backfills at the ARTIST level from Wikidata: resolve
each artist name -> Wikidata item -> P136 genre labels, then give every
untagged track its artist's genres as tags (provenance-marked overlay file —
the original Last.fm tags are never touched or mixed).

Scope decisions (why artist-level, why Wikidata):
  * Per-track MusicBrainz lookups would need ~36k requests at 1 req/s (~10 h);
    5,110 artist lookups at a polite ~4 req/s take ~25 min. Artist genres are
    coarser than track tags, but for a track with NOTHING they are strictly
    better than nothing.
  * A candidate item is accepted only if it carries BOTH P136 genres AND a
    P434 MusicBrainz artist ID. Genre-presence alone is not enough (the Pink
    Floyd FILM has film genres); the MusicBrainz link is the musical-identity
    check — fitting, given MusicBrainz is one of the template's named sources.
  * The harvest is resumable: every artist's result (including misses) is
    appended to a JSONL cache and never re-queried on rerun.

Run `uv run enrich` (harvest + overlay + coverage report). Network: Wikidata
API only, stdlib urllib, descriptive User-Agent, throttled.
"""

from __future__ import annotations

import gzip
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

_DATA = Path(__file__).resolve().parent.parent / "data" / "enrichment"
CACHE_PATH = _DATA / "wikidata_artists.jsonl"
GENRE_LABELS_PATH = _DATA / "wikidata_genre_labels.json"
# Follows the same override as train._ARTIFACTS so the overlay ships alongside
# whichever artifact set is being served.
OVERLAY_PATH = Path(
    os.environ.get("NEXTRACK_ARTIFACTS")
    or Path(__file__).resolve().parent.parent / "artifacts"
) / "enrichment_tags.json"

API = "https://www.wikidata.org/w/api.php"
USER_AGENT = "NextTrack-CM3070-student-project/0.1 (https://github.com/amrya200t/cm3070-nextrack)"
THROTTLE_S = 0.15
BATCH = 50


def _get(params: dict) -> dict:
    params = {**params, "format": "json"}
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"}
    )
    # 10s not 30: on a flaky route, hung requests dominate throughput — failing
    # fast and letting the resumable cache retry later is strictly better.
    with urllib.request.urlopen(req, timeout=10) as resp:
        raw = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
    return json.loads(raw)


def search_artist(name: str) -> list[str]:
    """Artist name -> up to 3 candidate Wikidata QIDs (label search)."""
    data = _get({
        "action": "wbsearchentities", "search": name, "language": "en",
        "type": "item", "limit": 3,
    })
    return [hit["id"] for hit in data.get("search", [])]


def fetch_entities(qids: list[str]) -> dict:
    """Batched claims fetch for up to BATCH QIDs."""
    data = _get({
        "action": "wbgetentities", "ids": "|".join(qids), "props": "claims",
    })
    return data.get("entities", {})


def _claim_qids(entity: dict, prop: str) -> list[str]:
    out = []
    for claim in entity.get("claims", {}).get(prop, []):
        try:
            out.append(claim["mainsnak"]["datavalue"]["value"]["id"])
        except (KeyError, TypeError):
            continue
    return out


def resolve_artist(name: str) -> dict:
    """One artist -> {'artist', 'qid', 'genre_qids'} (empty lists on miss)."""
    candidates = search_artist(name)
    if candidates:
        time.sleep(THROTTLE_S)
        entities = fetch_entities(candidates)  # one batched call for all 3
        for qid in candidates:
            entity = entities.get(qid, {})
            genres = _claim_qids(entity, "P136")
            if genres and "P434" in entity.get("claims", {}):
                return {"artist": name, "qid": qid, "genre_qids": genres}
    return {"artist": name, "qid": None, "genre_qids": []}


def load_cache(path: Path = CACHE_PATH) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    cache[row["artist"]] = row
    return cache


def harvest(artists: list[str], cache_path: Path = CACHE_PATH) -> dict[str, dict]:
    """Resolve all artists, appending to the resumable JSONL cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = load_cache(cache_path)
    todo = [a for a in artists if a not in cache]
    print(f"harvest: {len(cache):,} cached, {len(todo):,} to fetch")
    with open(cache_path, "a", encoding="utf-8") as fh:
        for i, artist in enumerate(todo, 1):
            try:
                row = resolve_artist(artist)
            except Exception as e:  # network hiccup: skip, rerun resumes it
                print(f"  ! {artist!r}: {e}")
                time.sleep(2)
                continue
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            cache[artist] = row
            if i % 200 == 0:
                hits = sum(1 for r in cache.values() if r["qid"])
                print(f"  {i:,}/{len(todo):,}  (hit rate {hits/len(cache):.0%})")
            time.sleep(THROTTLE_S)
    return cache


def genre_labels(qids: set[str], path: Path = GENRE_LABELS_PATH) -> dict[str, str]:
    """Genre QID -> English label, batched + cached."""
    labels: dict[str, str] = {}
    if path.exists():
        labels = json.loads(path.read_text(encoding="utf-8"))
    todo = sorted(q for q in qids if q not in labels)
    for i in range(0, len(todo), BATCH):
        chunk = todo[i : i + BATCH]
        data = _get({
            "action": "wbgetentities", "ids": "|".join(chunk),
            "props": "labels", "languages": "en",
        })
        for qid, ent in data.get("entities", {}).items():
            labels[qid] = ent.get("labels", {}).get("en", {}).get("value", "")
        time.sleep(THROTTLE_S)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(labels, ensure_ascii=False, indent=0), encoding="utf-8")
    return labels


def build_overlay(
    track_names: dict[str, tuple[str, str]],
    tags: dict[str, list[str]],
    cache: dict[str, dict],
    labels: dict[str, str],
) -> dict[str, list[str]]:
    """Untagged track_id -> artist's Wikidata genre labels. Pure (unit-tested).

    Only tracks with NO Last.fm tags get an overlay entry — Last.fm data always
    wins where it exists (finer-grained and listening-derived)."""
    overlay: dict[str, list[str]] = {}
    for tid, (artist, _title) in track_names.items():
        if tags.get(tid):
            continue
        row = cache.get(artist)
        if not row or not row["genre_qids"]:
            continue
        # Wikidata labels say "rock music"/"electronic music" where Last.fm says
        # "rock"/"electronic"; strip the suffix so overlay tags actually match
        # session tags in the Jaccard overlap and read naturally in `why`.
        genre_names = [labels.get(q, "") for q in row["genre_qids"]]
        genre_names = [g.removesuffix(" music").strip() for g in genre_names if g]
        genre_names = [g for g in genre_names if g]
        if genre_names:
            overlay[tid] = genre_names
    return overlay


def build_fallback_overlay(
    track_names: dict[str, tuple[str, str]],
    tags: dict[str, list[str]],
    clean_title,
    normalize,
    top_n: int = 10,
) -> dict[str, list[str]]:
    """Two internal fallback tiers for untagged tracks, from Last.fm data alone.

    Tier 1 (edition siblings): an untagged "Karma Police - Remastered" inherits
    the tags of the same artist's tagged "Karma Police" — the tag file is keyed
    by exact name, so edition suffixes break the join for no musical reason.

    Tier 2 (artist fallback): a track whose artist has other tagged tracks gets
    the artist's most common tags (top_n by frequency across their catalogue) —
    the same artist-level-is-better-than-nothing argument as the Wikidata
    overlay, but sourced from richer local data and requiring no network.

    Pure function (unit-tested); returns {track_id: [tag, ...]} for untagged
    tracks only. Callers merge with merged_tags (Last.fm exact tags still win).
    """
    sibling_key: dict[tuple[str, str], str] = {}
    artist_tag_counts: dict[str, dict[str, int]] = {}
    for tid, (artist, title) in track_names.items():
        if not tags.get(tid):
            continue
        na = normalize(artist)
        sibling_key.setdefault((na, normalize(clean_title(title))), tid)
        counts = artist_tag_counts.setdefault(na, {})
        for t in tags[tid]:
            counts[t] = counts.get(t, 0) + 1

    artist_top: dict[str, list[str]] = {
        na: [t for t, _c in sorted(counts.items(), key=lambda kv: -kv[1])[:top_n]]
        for na, counts in artist_tag_counts.items()
    }

    overlay: dict[str, list[str]] = {}
    for tid, (artist, title) in track_names.items():
        if tags.get(tid):
            continue
        na = normalize(artist)
        sib = sibling_key.get((na, normalize(clean_title(title))))
        if sib is not None:
            overlay[tid] = list(tags[sib])
        elif na in artist_top:
            overlay[tid] = artist_top[na]
    return overlay


def merged_tags(tags: dict[str, list[str]], overlay: dict[str, list[str]]) -> dict[str, list[str]]:
    """Last.fm tags + Wikidata overlay for tracks that have none."""
    out = dict(tags)
    for tid, genres in overlay.items():
        if not out.get(tid):
            out[tid] = genres
    return out


def main() -> None:
    import pickle

    from nextrack.train import IDMAP_PATH

    maps = pickle.load(open(IDMAP_PATH, "rb"))
    track_names, tags = maps["track_names"], maps["tags"]
    catalogue = set(maps["track_id_to_index"])

    untagged_artists = sorted({
        track_names[tid][0] for tid in catalogue
        if tid in track_names and not tags.get(tid)
    })
    cache = harvest(untagged_artists)
    all_genre_qids = {q for r in cache.values() for q in r["genre_qids"]}
    labels = genre_labels(all_genre_qids)
    overlay = build_overlay(track_names, tags, cache, labels)

    OVERLAY_PATH.parent.mkdir(parents=True, exist_ok=True)
    OVERLAY_PATH.write_text(json.dumps(overlay, ensure_ascii=False), encoding="utf-8")

    tagged_before = sum(1 for tid in catalogue if tags.get(tid))
    tagged_after = sum(1 for tid in catalogue if tags.get(tid) or overlay.get(tid))
    print(f"tag coverage: {tagged_before/len(catalogue):.1%} -> {tagged_after/len(catalogue):.1%} "
          f"(+{tagged_after-tagged_before:,} tracks via Wikidata)")
    print(f"overlay saved: {OVERLAY_PATH}")


if __name__ == "__main__":
    main()
