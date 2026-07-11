"""Unit tests for the tag-overlap explanation (explain.py)."""

from nextrack.explain import tag_overlap_explain


TAGS = {
    "seed1": ["Progressive rock", "psychedelic", "classic rock"],
    "seed2": ["genre: Psychedelic", "ambient"],
    "cand_overlap": ["psychedelic", "classic rock", "electronic"],
    "cand_disjoint": ["hip hop", "rap"],
    "cand_untagged": [],
}


def test_shared_tags_surface_in_why_string():
    why, shared = tag_overlap_explain(["seed1"], "cand_overlap", TAGS)
    assert shared == ["psychedelic", "classic rock"]
    assert why == "Because you listened to psychedelic and classic rock"


def test_normalisation_unifies_case_and_genre_prefix():
    # seed2's "genre: Psychedelic" must match the candidate's "psychedelic".
    _, shared = tag_overlap_explain(["seed2"], "cand_overlap", TAGS)
    assert "psychedelic" in shared


def test_no_overlap_returns_empty_not_fabricated():
    why, shared = tag_overlap_explain(["seed1"], "cand_disjoint", TAGS)
    assert why == "" and shared == []


def test_untagged_candidate_returns_empty():
    # ~31% of the catalogue has no tags; the explanation must stay honest.
    why, shared = tag_overlap_explain(["seed1"], "cand_untagged", TAGS)
    assert why == "" and shared == []


def test_single_shared_tag_grammar():
    tags = {"s": ["ambient"], "c": ["ambient", "idm"]}
    why, shared = tag_overlap_explain(["s"], "c", tags)
    assert shared == ["ambient"]
    assert why == "Because you listened to ambient"
