"""Unit tests for the ranked search (search.py)."""

from nextrack.search import _normalize, _token_tier, rank_tracks


# ------------------------------------------------------------- normalisation --
def test_normalize_casefolds_and_strips_diacritics():
    assert _normalize("Björk") == "bjork"
    assert _normalize("Motörhead") == "motorhead"
    assert _normalize("Sigur Rós") == "sigur ros"
    assert _normalize("Beyoncé") == "beyonce"
    # casefold handles the German sharp s
    assert _normalize("Straße") == "strasse"


# ---------------------------------------------------------------- tier score --
def test_token_tier_exact_beats_prefix_beats_substring():
    assert _token_tier("money", "money") == 100        # exact
    assert _token_tier("mon", "money") == 80           # starts-with
    assert _token_tier("dark", "the dark side") == 60  # word-boundary
    assert _token_tier("ide", "darkside") == 40        # substring
    assert _token_tier("zzz", "money") == 0            # no match


def test_token_tier_space_insensitive():
    # run-together query matches across word boundaries: 'dualipa' ~ 'dua lipa'
    assert _token_tier("dualipa", "dua lipa") == 40    # squashed substring
    assert _token_tier("pinkfloyd", "pink floyd") == 40
    # a real word match still outranks the squashed one: 'lipa' hits the second
    # word (WORD=60), which beats the squashed-substring tier (40)
    assert _token_tier("lipa", "dua lipa") == 60
    # and a prefix of the whole string is stronger still (PREFIX=80)
    assert _token_tier("dua", "dua lipa") == 80


# ----------------------------------------------------------------- ranking ----
NAMES = {
    "1": ("Pink Floyd", "Money"),
    "2": ("Pink Floyd", "Time"),
    "3": ("Radiohead", "Karma Police"),
    "4": ("Bjork", "Joga"),          # ascii spelling in catalogue
    "5": ("Pink Martini", "Sympathique"),
}


def _ids(results):
    return [r["track_id"] for r in results]


def test_multiword_cross_field_query():
    # "pink floyd money" spans artist + title; must return Money first.
    res = rank_tracks("pink floyd money", NAMES, limit=5)
    assert res[0]["track_id"] == "1"


def test_multiword_partial_artist_plus_title():
    res = rank_tracks("floyd time", NAMES, limit=5)
    assert res[0]["track_id"] == "2"


def test_all_tokens_must_match():
    # xyzzy matches nothing -> no results, even though pink+floyd match.
    assert rank_tracks("pink floyd xyzzy", NAMES, limit=5) == []


def test_word_order_insensitive():
    res = rank_tracks("money floyd", NAMES, limit=5)
    assert res[0]["track_id"] == "1"


def test_single_token_still_works():
    res = rank_tracks("radiohead", NAMES, limit=5)
    assert res[0]["track_id"] == "3"


def test_diacritic_insensitive_query():
    # query has the accent, catalogue has ascii -> still matches
    res = rank_tracks("björk", NAMES, limit=5)
    assert res[0]["track_id"] == "4"


def test_run_together_artist_name():
    # "pinkfloyd" (no space) should still find Pink Floyd tracks.
    res = rank_tracks("pinkfloyd", NAMES, limit=5)
    assert res and res[0]["artist"] == "Pink Floyd"


def test_limit_is_respected():
    res = rank_tracks("pink", NAMES, limit=1)
    assert len(res) == 1


# ------------------------------------------------------------ fuzzy fallback --
def test_fuzzy_fallback_catches_typo():
    # 'radiohed' has no substring match; fuzzy fallback should still find it.
    res = rank_tracks("radiohed", NAMES, limit=5)
    assert res and res[0]["track_id"] == "3"


def test_exact_still_beats_fuzzy():
    # an exact-ish hit must never be displaced by a fuzzy one.
    res = rank_tracks("money", NAMES, limit=5)
    assert res[0]["track_id"] == "1"
