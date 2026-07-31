"""Unit tests for the enrichment overlay logic (Phase F1). Pure — no network."""

import json

from nextrack.enrich import build_overlay, load_cache, merged_tags


NAMES = {
    "t1": ("Nirvana", "Lithium"),
    "t2": ("Nirvana", "Come as You Are"),
    "t3": ("Obscure Act", "Untitled"),
    "t4": ("Pink Floyd", "Time"),
}
TAGS = {"t4": ["progressive rock"]}  # t1-t3 untagged
CACHE = {
    "Nirvana": {"artist": "Nirvana", "qid": "Q11649", "genre_qids": ["Q11365", "Q9610"]},
    "Obscure Act": {"artist": "Obscure Act", "qid": None, "genre_qids": []},
}
LABELS = {"Q11365": "grunge", "Q9610": "alternative rock"}


def test_overlay_fills_only_untagged_tracks_with_resolved_artists():
    overlay = build_overlay(NAMES, TAGS, CACHE, LABELS)
    assert overlay == {
        "t1": ["grunge", "alternative rock"],
        "t2": ["grunge", "alternative rock"],
    }
    assert "t4" not in overlay  # has Last.fm tags -> untouched
    assert "t3" not in overlay  # artist unresolved -> honestly absent


def test_overlay_strips_wikidata_music_suffix():
    # "rock music" (Wikidata style) must become "rock" (Last.fm style) so the
    # overlap matching and `why` phrasing stay consistent.
    cache = {"Nirvana": {"artist": "Nirvana", "qid": "Q11649", "genre_qids": ["Q11399"]}}
    overlay = build_overlay(NAMES, TAGS, cache, {"Q11399": "rock music"})
    assert overlay["t1"] == ["rock"]


def test_overlay_skips_unlabelled_genre_qids():
    cache = {"Nirvana": {"artist": "Nirvana", "qid": "Q11649", "genre_qids": ["Q404"]}}
    assert build_overlay(NAMES, TAGS, cache, {}) == {}


def test_merged_tags_lastfm_always_wins():
    merged = merged_tags(TAGS, {"t1": ["grunge"], "t4": ["wrong genre"]})
    assert merged["t1"] == ["grunge"]
    assert merged["t4"] == ["progressive rock"]  # overlay never overwrites


def test_cache_resume_roundtrip(tmp_path):
    p = tmp_path / "cache.jsonl"
    rows = [CACHE["Nirvana"], CACHE["Obscure Act"]]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    cache = load_cache(p)
    assert set(cache) == {"Nirvana", "Obscure Act"}
    assert cache["Nirvana"]["genre_qids"] == ["Q11365", "Q9610"]


# ---- overlay wiring into inference (Phase F1 integration) -------------------
def test_apply_overlay_merges_when_file_present(tmp_path, monkeypatch):
    import nextrack.infer as infer
    import nextrack.enrich as enrich

    overlay_file = tmp_path / "enrichment_tags.json"
    overlay_file.write_text(json.dumps({"t1": ["grunge"]}), encoding="utf-8")
    monkeypatch.setattr(enrich, "OVERLAY_PATH", overlay_file)

    merged = infer._apply_enrichment_overlay({"t4": ["progressive rock"]})
    assert merged["t1"] == ["grunge"]          # untagged track enriched
    assert merged["t4"] == ["progressive rock"] # existing tags untouched


def test_apply_overlay_noop_when_file_absent(tmp_path, monkeypatch):
    import nextrack.infer as infer
    import nextrack.enrich as enrich

    monkeypatch.setattr(enrich, "OVERLAY_PATH", tmp_path / "does_not_exist.json")
    tags = {"t4": ["progressive rock"]}
    assert infer._apply_enrichment_overlay(tags) == tags
