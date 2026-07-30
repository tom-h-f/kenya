"""Unit tests for hate-seeking query construction (pure, no network, no R2)."""

from __future__ import annotations

import pytest

from kenya_monitor import hate_seek as hs
from kenya_monitor.collectors.x import MAX_QUERY_CHARS, build_query
from kenya_monitor.config import FP_RISKS, HateTerm, HateTermSet, load_hate_terms

# The 16 entries of kma.incitement.LEXICON. Duplicated as a literal because the
# collector deliberately carries no `kma` dependency; this test is the contract
# that keeps the two registers in sync.
LEXICON_TERMS = {
    "madoadoa", "kwekwe", "sangari", "madimoni", "mende", "nyoka", "fumigate",
    "rudi_kwao", "fukuza", "waondoke",
    "watajua_hawajui", "kimeumana",
    "watu_wa_kule", "wabara", "41_vs_1", "uthamaki",
}


@pytest.fixture(scope="module")
def terms() -> HateTermSet:
    return load_hate_terms()


def test_register_mirrors_the_analysis_lexicon(terms):
    assert {t.term for t in terms.terms} == LEXICON_TERMS


def test_every_term_is_well_formed(terms):
    for t in terms.terms:
        assert t.fp_risk in FP_RISKS, t.term
        assert t.query.strip(), t.term
        assert t.category, t.term
        assert t.notes, f"{t.term} needs a note - these terms are triage seeds, not verdicts"


def test_anchor_tiers_scale_with_false_positive_risk(terms):
    low = hs.HateQuery("madoadoa", "madoadoa", "low", hs.LEXICON)
    med = hs.HateQuery("kwekwe", "kwekwe", "medium", hs.LEXICON)
    high = hs.HateQuery("mende", "mende", "high", hs.LEXICON)
    assert hs.anchors_for(low, terms.anchors) is None
    assert hs.anchors_for(med, terms.anchors) == terms.anchors["core"]
    assert hs.anchors_for(high, terms.anchors) == terms.anchors["wide"]
    assert len(terms.anchors["wide"]) > len(terms.anchors["core"])


def test_high_risk_term_is_scoped_low_risk_term_is_not(terms):
    high = hs.render(
        hs.HateQuery("mende", "mende", "high", hs.LEXICON),
        terms.anchors, since="2026-07-01", until="2026-07-02",
    )
    assert high.startswith("mende (Kenya OR ")
    assert "since:2026-07-01 until:2026-07-02" in high

    low = hs.render(
        hs.HateQuery("madoadoa", "madoadoa", "low", hs.LEXICON), terms.anchors
    )
    assert low == "madoadoa include:nativeretweets"


def test_no_forbidden_operators_are_ever_emitted(terms):
    """`lang:` is useless here (~0% of Swahili posts are tagged sw) and would
    silently drop most of the corpus; `geocode:`/`from:` are unvalidated for
    this use. Account expansion is the targeting pass's job, not search's."""
    for q in hs.select_terms(terms, max_terms=99, max_targets=99):
        rendered = hs.render(q, terms.anchors, since="2026-07-01", until="2026-07-02")
        for banned in ("lang:", "geocode:", "near:", "from:", "to:"):
            assert banned not in rendered, f"{q.term} emitted {banned}"


def test_every_rendered_query_fits_the_platform_limit(terms):
    for q in hs.select_terms(terms, max_terms=99, max_targets=99):
        rendered = hs.render(q, terms.anchors, since="2026-07-01", until="2026-07-02")
        assert len(rendered) <= MAX_QUERY_CHARS, q.term


def test_or_alternatives_are_grouped_before_filters_are_appended():
    """X binds AND tighter than OR, so `a OR b (anchors)` would mean
    `a OR (b AND anchors)` and the leading alternatives would escape every
    filter - exactly the high-false-positive terms that most need scoping."""
    q = build_query("fukuza OR wafukuzwe", anchors=["Kenya", "Ruto"], since="2026-07-01")
    assert q.startswith("(fukuza OR wafukuzwe) (Kenya OR Ruto)")
    assert not q.startswith("fukuza OR")


def test_or_grouping_is_a_noop_for_simple_and_pregrouped_keywords():
    assert build_query("Wantam", anchors=["Kenya"]) == "Wantam (Kenya)"
    assert build_query("(a OR b)", anchors=["Kenya"]) == "(a OR b) (Kenya)"


def test_every_multi_alternative_register_term_is_grouped(terms):
    for q in hs.select_terms(terms, max_terms=99, max_targets=99):
        rendered = hs.render(q, terms.anchors, since="2026-07-01", until="2026-07-02")
        head = rendered.split(" since:")[0]
        if " OR " in q.query and not q.query.startswith("("):
            assert head.startswith("("), f"{q.term} leaks alternatives past its filters"


def test_exclusions_render_as_negations():
    assert build_query("mende", exclude=["dawa"]) == "mende -dawa"


def test_build_query_rejects_an_overlong_query():
    with pytest.raises(ValueError, match="over the"):
        build_query("x", anchors=["anchor"] * 200)


def test_build_query_is_backwards_compatible():
    assert build_query("Wantam") == "Wantam"
    assert (
        build_query("IEBC", min_faves=5, since="2026-07-01", until="2026-07-02",
                    include_retweets=True)
        == "IEBC min_faves:5 since:2026-07-01 until:2026-07-02 include:nativeretweets"
    )


def test_ethnonyms_are_never_searched_bare(terms):
    """A bare ethnonym returns ordinary discussion and biases the corpus toward
    ethnicised discourse. Only the menace conjunction is ever emitted."""
    queries = hs.target_queries(terms)
    assert queries, "expected ethnonym targets in the register"
    for q in queries:
        assert q.fp_risk == "high"
        assert q.source == hs.TARGETS_OF_HATE
        rendered = hs.render(q, terms.anchors)
        ethnonym = q.term.split(":", 1)[1]
        assert "(" in q.query and " OR " in q.query, f"{ethnonym} searched bare"
        assert any(m.strip('"') in rendered for m in terms.menace)


def test_targeted_sources_write_quarantined_partitions(terms):
    """Every hate-seeking partition must be one the analysis side excludes from
    prevalence (kma.db.TARGETED_TYPES)."""
    targeted = {
        "hate_search", "hate_target_search", "hate_timeline",
        "hate_replies", "hate_hydrated", "cib_timeline",
    }
    for source, partition in hs.PARTITIONS.items():
        assert partition in targeted, source
    assert hs.PARTITIONS[hs.TARGETS_OF_HATE] == "hate_target_search"
    assert hs.PARTITIONS[hs.LEXICON] == "hate_search"


def test_selection_respects_the_total_budget(terms):
    picked = hs.select_terms(terms, max_terms=8, max_targets=3, max_mined=5)
    assert len(picked) == 8
    assert sum(q.source == hs.TARGETS_OF_HATE for q in picked) == 3


def test_mined_terms_take_reserved_slots_and_are_always_high_risk(terms):
    mined = [{"term": "newcoded1"}, {"term": "newcoded2"}]
    picked = hs.select_terms(terms, mined=mined, max_terms=8, max_targets=2, max_mined=2)
    mined_picked = [q for q in picked if q.source == hs.MINED]
    assert {q.term for q in mined_picked} == {"newcoded1", "newcoded2"}
    assert all(q.fp_risk == "high" for q in mined_picked)
    assert all(hs.anchors_for(q, terms.anchors) == terms.anchors["wide"] for q in mined_picked)


def test_rotation_covers_the_whole_register(terms):
    """Least-recently-searched first, so a small per-pass cap still sweeps the
    full register instead of re-running the same terms forever."""
    state: dict[str, str] = {}
    seen: set[str] = set()
    for _ in range(8):
        picked = hs.select_terms(terms, state=state, max_terms=8, max_targets=3)
        seen |= {q.term for q in picked}
        state = hs.mark_searched(picked, state)
    expected = {t.term for t in terms.terms} | {q.term for q in hs.target_queries(terms)}
    assert seen == expected


def test_rotation_prefers_never_searched_terms(terms):
    all_lex = hs.lexicon_queries(terms)
    state = hs.mark_searched(all_lex[:14], {})
    picked = hs.select_terms(terms, state=state, max_terms=2, max_targets=0)
    assert {q.term for q in picked} == {t.term for t in all_lex[14:]}


def test_state_round_trips(tmp_path):
    path = tmp_path / "hate_seek.json"
    assert hs.load_state(path) == {}
    state = hs.mark_searched([hs.HateQuery("mende", "mende", "high", hs.LEXICON)])
    hs.save_state(state, path)
    assert hs.load_state(path) == state


def test_loader_rejects_an_invalid_fp_risk(tmp_path):
    bad = tmp_path / "hate_terms.yaml"
    bad.write_text(
        "anchors:\n  core: [Kenya]\n  wide: [Kenya, Nairobi]\n"
        "terms:\n  - {term: x, query: x, fp_risk: extreme}\n"
    )
    with pytest.raises(ValueError, match="fp_risk"):
        load_hate_terms(bad)


def test_loader_requires_both_anchor_tiers(tmp_path):
    bad = tmp_path / "hate_terms.yaml"
    bad.write_text("anchors:\n  core: [Kenya]\nterms: []\n")
    with pytest.raises(ValueError, match="wide"):
        load_hate_terms(bad)


def test_selection_is_empty_without_a_budget(terms):
    assert hs.select_terms(terms, max_terms=0) == []


def test_zero_target_slots_yields_only_lexicon(terms):
    picked = hs.select_terms(terms, max_terms=4, max_targets=0, max_mined=0)
    assert len(picked) == 4
    assert all(q.source == hs.LEXICON for q in picked)


def test_hate_term_set_lookup(terms):
    assert terms.by_name("mende").fp_risk == "high"
    assert terms.by_name("nope") is None
    assert isinstance(terms.terms[0], HateTerm)


def test_days_option_widens_the_swept_window():
    """`--days` must reach back far enough to fill a collection gap; the default
    only covers the last two days, which cannot recover an outage."""
    from kenya_monitor.collectors.x import recent_windows

    assert len(recent_windows(2)) == 2
    assert len(recent_windows(14)) == 14
    # windows are daily and ordered oldest-first, so the first one is the
    # furthest back - that is what determines gap coverage.
    oldest_default, _ = recent_windows(2)[0]
    oldest_wide, _ = recent_windows(14)[0]
    assert oldest_wide < oldest_default
