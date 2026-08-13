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


def test_backfill_opens_its_own_connection(monkeypatch):
    """`backfill` calls `connect()` itself, so the name must be imported at
    module scope - not only under TYPE_CHECKING.

    A `--limit` smoke run does NOT cover this: that path takes a connection from
    its caller and never reaches `connect()`. The full drain does, and died on
    Modal with `NameError: name 'connect' is not defined` minutes after the
    smoke test had passed.
    """
    from kma import incitement

    opened = []
    monkeypatch.setattr(incitement, "connect", lambda: opened.append(1) or "con")
    monkeypatch.setattr(incitement, "score_new", lambda *a, **k: 0)

    assert incitement.backfill() == 0
    assert opened == [1], "backfill must open exactly one connection"


def test_backfill_drains_until_a_pass_returns_nothing(monkeypatch):
    from kma import incitement

    remaining = [500, 500, 120, 0]
    monkeypatch.setattr(incitement, "connect", lambda: "con")
    monkeypatch.setattr(
        incitement, "score_new", lambda *a, **k: remaining.pop(0)
    )

    assert incitement.backfill() == 1120
    assert remaining == []
