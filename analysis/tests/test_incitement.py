"""Unit tests for the kma.incitement lexicon (no R2, no models)."""

from kma import incitement as inc


def test_scan_text_hits_core_terms():
    hits, cats = inc.scan_text("Hawa ni madoadoa lazima waondoke")
    assert "madoadoa" in hits
    assert "waondoke" in hits
    assert set(cats) == {"dehumanisation", "expulsion"}


def test_scan_text_spacing_and_case_variants():
    hits, _ = inc.scan_text("MADOA DOA everywhere")
    assert "madoadoa" in hits
    hits, _ = inc.scan_text("Watajua  hawajui kabisa")
    assert "watajua_hawajui" in hits


def test_scan_text_word_boundaries():
    hits, _ = inc.scan_text("the mendel experiment on snyoka")
    assert hits == []


def test_scan_text_41_framing():
    for s in ["it is 41 vs 1 again", "41 against 1", "41 versus 1"]:
        hits, cats = inc.scan_text(s)
        assert "41_vs_1" in hits and cats == ["othering"]


def test_scan_text_clean_political_speech():
    hits, cats = inc.scan_text(
        "The government must lower the cost of living before 2027."
    )
    assert hits == [] and cats == []


def test_lexicon_entries_have_metadata():
    for category, terms in inc.LEXICON.items():
        for term, entry in terms.items():
            assert entry["pattern"], (category, term)
            assert entry["fp_risk"] in {"low", "medium", "high"}
            assert entry["notes"]


def test_hypotheses_include_contrast_class():
    assert "political_criticism" in inc.HYPOTHESES
