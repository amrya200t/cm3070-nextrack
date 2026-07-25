"""Unit tests for the hybrid re-ranker (Phase B3)."""

from nextrack.rerank import DEFAULT_WEIGHTS, rerank, tag_jaccard


CANDIDATES = [
    ("a1", 0.90),  # artist X
    ("a2", 0.89),  # artist X
    ("a3", 0.88),  # artist X
    ("a4", 0.87),  # artist X
    ("b1", 0.86),  # artist Y
    ("c1", 0.85),  # artist Z
]
ARTISTS = {"a1": "X", "a2": "X", "a3": "X", "a4": "X", "b1": "Y", "c1": "Z"}
NO_TAGS: dict[str, list[str]] = {}


def test_zero_weights_reproduce_cf_order():
    out = rerank(
        CANDIDATES, ["s"], NO_TAGS, ARTISTS, k=6,
        weights={"tag": 0.0, "artist": 0.0},
    )
    assert [tid for tid, _ in out] == ["a1", "a2", "a3", "a4", "b1", "c1"]
    # And scores are untouched cosine scores.
    assert [round(s, 2) for _, s in out] == [0.90, 0.89, 0.88, 0.87, 0.86, 0.85]


def test_artist_run_gets_demoted():
    # With the artist penalty on, the 4th artist-X track (a4, base 0.87) pays
    # 3 * 0.03 = 0.09 -> adjusted 0.78, so b1 (0.86) and c1 (0.85) must both
    # outrank it.
    out = rerank(
        CANDIDATES, ["s"], NO_TAGS, ARTISTS, k=6,
        weights={"tag": 0.0, "artist": 0.03},
    )
    order = [tid for tid, _ in out]
    assert order.index("b1") < order.index("a4")
    assert order.index("c1") < order.index("a4")


def test_tag_overlap_promotes_candidate():
    # d1 and d2 tie on cosine; only d2 shares the session's tags, so with the
    # tag weight on it must rank first.
    cands = [("d1", 0.80), ("d2", 0.80)]
    artists = {"d1": "P", "d2": "Q"}
    tags = {"s": ["shoegaze", "dream pop"], "d2": ["shoegaze", "dream pop"], "d1": ["metal"]}
    out = rerank(cands, ["s"], tags, artists, k=2, weights={"tag": 0.05, "artist": 0.0})
    assert [tid for tid, _ in out][0] == "d2"


def test_jaccard_bounds_and_empty_honesty():
    assert tag_jaccard({"rock"}, ["rock"]) == 1.0
    assert tag_jaccard({"rock"}, ["jazz"]) == 0.0
    assert tag_jaccard(set(), ["rock"]) == 0.0  # untagged session -> no bonus
    assert tag_jaccard({"rock"}, []) == 0.0     # untagged candidate -> no bonus


def test_defaults_are_the_documented_ones():
    # The report quotes these (from the B4 sweep); silent change = test failure.
    assert DEFAULT_WEIGHTS == {"tag": 0.05, "artist": 0.01}
