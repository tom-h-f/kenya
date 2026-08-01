"""Unit tests for collection-scope filtering in kma.db (no R2 required).

`posts_source` is monkeypatched to a local temp table so the same SQL that runs
against the R2 globs can be exercised offline.
"""

import duckdb
import pytest

from kma import db

# `author_handle` matters: the timeline-leak correction reclassifies
# `type=timeline` posts whose author is not a curated target. WilliamsRuto is in
# targets.yaml, so these rows are curated unless a test says otherwise.
ROWS = [
    # (platform_post_id, type, collected_at) - one post per row unless re-collected
    ("baseline_only", "search", "2026-07-01 00:00:00"),
    ("timeline_only", "timeline", "2026-07-01 00:00:00"),
    ("targeted_only", "hate_search", "2026-07-05 00:00:00"),
    ("ethno_only", "hate_target_search", "2026-07-05 00:00:00"),
    # The case the whole design exists for: found by baseline search first, then
    # re-collected by a targeted pass. The latest row says hate_search.
    ("recollected", "search", "2026-07-01 00:00:00"),
    ("recollected", "hate_search", "2026-07-06 00:00:00"),
]


@pytest.fixture
def con(monkeypatch):
    c = duckdb.connect()
    c.execute(
        "CREATE TABLE _posts (platform VARCHAR, platform_post_id VARCHAR, "
        "type VARCHAR, collected_at TIMESTAMP, author_handle VARCHAR)"
    )
    c.executemany(
        "INSERT INTO _posts VALUES ('x', ?, ?, ?::TIMESTAMP, 'WilliamsRuto')",
        [list(r) for r in ROWS],
    )

    def fake_source(platform: str = "*", type: str = "*") -> str:
        if type == "*":
            return "_posts"
        return f"(SELECT * FROM _posts WHERE type = '{type}')"

    monkeypatch.setattr(db, "posts_source", fake_source)
    return c


def _ids(rel) -> set[str]:
    return {r[0] for r in rel.project("platform_post_id").fetchall()}


def test_recollected_post_stays_in_baseline(con):
    """The core guarantee: a post re-collected by a targeted pass must not leave
    the baseline denominator just because its latest row says hate_search."""
    assert "recollected" in _ids(db.latest_posts(con, scope="baseline"))
    assert "recollected" not in _ids(db.latest_posts(con, scope="targeted"))


def test_baseline_and_targeted_partition_the_corpus(con):
    baseline = _ids(db.latest_posts(con, scope="baseline"))
    targeted = _ids(db.latest_posts(con, scope="targeted"))
    every = _ids(db.latest_posts(con, scope="all"))
    assert baseline == {"baseline_only", "timeline_only", "recollected"}
    assert targeted == {"targeted_only", "ethno_only"}
    assert baseline | targeted == every
    assert baseline & targeted == set()


def test_scope_all_matches_unscoped_sql(con):
    """scope="all" must reproduce today's numbers exactly - no join, no filter."""
    unscoped = con.sql(
        """
        SELECT * FROM _posts
        QUALIFY row_number() OVER (
            PARTITION BY platform, platform_post_id ORDER BY collected_at DESC
        ) = 1
        """
    )
    assert _ids(db.latest_posts(con, scope="all")) == _ids(unscoped)
    assert len(db.latest_posts(con, scope="all").fetchall()) == len(unscoped.fetchall())


def test_ethnonym_partition_is_targeted(con):
    """hate_target_search biases the corpus toward ethnicised discourse, so it
    must never reach a baseline prevalence measurement."""
    assert "hate_target_search" in db.TARGETED_TYPES
    assert "ethno_only" not in _ids(db.latest_posts(con, scope="baseline"))


def test_unknown_type_raises_rather_than_defaulting(con):
    con.execute("INSERT INTO _posts VALUES ('x', 'mystery', 'future_pass', now(), 'WilliamsRuto')")
    for scope in ("baseline", "targeted"):
        with pytest.raises(duckdb.InvalidInputException, match="unknown post type"):
            db.latest_posts(con, scope=scope).fetchall()


def test_first_seen_exposes_provenance_columns(con):
    row = (
        db.latest_posts(con, scope="baseline")
        .filter("platform_post_id = 'recollected'")
        .project("first_type, first_collected_at")
        .fetchone()
    )
    assert row[0] == "search"
    assert str(row[1]).startswith("2026-07-01")


def test_type_filter_composes_with_scope(con):
    """Narrowing to one partition must not change which scope a post belongs to:
    first-seen is computed over every type regardless of the `type` filter."""
    got = db.latest_posts(con, type="hate_search", scope="baseline")
    assert _ids(got) == {"recollected"}


def test_notebook_query_shape_composes(con):
    """The hatespeech notebook builds its own SQL rather than calling
    latest_posts: CTE + scope predicate + QUALIFY, then an INNER JOIN onto
    hatespeech scores. Verify that exact shape executes and scopes correctly."""
    con.execute("CREATE TABLE _hate (platform_post_id VARCHAR, label VARCHAR)")
    con.executemany(
        "INSERT INTO _hate VALUES (?, ?)",
        [
            ["baseline_only", "neither"],
            ["timeline_only", "neither"],
            ["recollected", "hate"],
            ["targeted_only", "hate"],
            ["ethno_only", "hate"],
        ],
    )
    got = con.sql(
        f"""
        WITH {db.first_seen_types_cte("x", "fs")},
        p AS (
            SELECT p0.platform_post_id, fs.first_type
            FROM {db.posts_source("x")} p0
            JOIN fs USING (platform, platform_post_id)
            WHERE {db.scope_predicate("baseline", "fs.first_type")}
            QUALIFY row_number() OVER (
                PARTITION BY platform, platform_post_id ORDER BY collected_at DESC
            ) = 1
        )
        SELECT p.platform_post_id, p.first_type, h.label
        FROM p JOIN _hate h USING (platform_post_id)
        """
    ).fetchall()
    by_id = {r[0]: (r[1], r[2]) for r in got}
    assert set(by_id) == {"baseline_only", "timeline_only", "recollected"}
    assert by_id["recollected"] == ("search", "hate")


def test_baseline_rate_is_not_deflated_by_targeted_recollection(con):
    """The failure this guards against: re-collecting toxic posts under a
    targeted partition must not shrink the baseline denominator over time."""
    before = len(_ids(db.latest_posts(con, scope="baseline")))
    con.execute(
        "INSERT INTO _posts VALUES "
        "('x', 'baseline_only', 'hate_search', '2026-07-09'::TIMESTAMP, 'WilliamsRuto'), "
        "('x', 'timeline_only', 'hate_timeline', '2026-07-09'::TIMESTAMP, 'WilliamsRuto')"
    )
    assert len(_ids(db.latest_posts(con, scope="baseline"))) == before


def test_scope_predicate_rejects_unknown_scope():
    for bad in ("Baseline", "targetted", ""):
        with pytest.raises(ValueError, match="unknown scope"):
            db.scope_predicate(bad)
    with pytest.raises(ValueError, match="unknown scope"):
        db.latest_posts(None, scope="bogus")


def test_scope_type_registries_are_disjoint_and_complete():
    assert set(db.BASELINE_TYPES) & set(db.TARGETED_TYPES) == set()
    assert set(db.KNOWN_TYPES) == set(db.BASELINE_TYPES) | set(db.TARGETED_TYPES)
    assert db.scope_predicate("all") == "TRUE"


# --- timeline leak correction ------------------------------------------------


CURATED = {"williamsruto", "rigathi"}


def test_promoted_account_timelines_are_reclassified_as_targeted(con, monkeypatch):
    """Until 2026-08-01 the accounts pass wrote curated targets AND
    coordination-promoted accounts to `type=timeline`, a baseline partition.
    Promoted accounts are selected for looking coordinated, so their timelines
    are targeted collection. Measured leak: 41,595 posts, 6.7% of baseline."""
    monkeypatch.setattr(db, "curated_handles", lambda *a, **k: CURATED)
    con.execute(
        "INSERT INTO _posts VALUES ('x', 'leaked', 'timeline', now(), 'somePromotedAcct')"
    )
    baseline = _ids(db.latest_posts(con, scope="baseline"))
    targeted = _ids(db.latest_posts(con, scope="targeted"))
    assert "leaked" not in baseline, "promoted timeline must leave the denominator"
    assert "leaked" in targeted, "and must land in targeted, not vanish"
    assert "timeline_only" in baseline, "curated targets stay baseline"


def test_reclassification_keeps_scopes_complementary(con, monkeypatch):
    """Reclassify, never drop: every post must still land in exactly one scope."""
    monkeypatch.setattr(db, "curated_handles", lambda *a, **k: CURATED)
    con.execute("INSERT INTO _posts VALUES ('x', 'leaked', 'timeline', now(), 'promoted1')")
    baseline = _ids(db.latest_posts(con, scope="baseline"))
    targeted = _ids(db.latest_posts(con, scope="targeted"))
    every = _ids(db.latest_posts(con, scope="all"))
    assert baseline | targeted == every
    assert baseline & targeted == set()


def test_correction_is_a_noop_without_the_curated_list(con, monkeypatch):
    """If targets.yaml is unreachable the correction disables rather than
    silently reclassifying everything as targeted."""
    monkeypatch.setattr(db, "curated_handles", lambda *a, **k: set())
    con.execute("INSERT INTO _posts VALUES ('x', 'leaked', 'timeline', now(), 'anyone')")
    assert "leaked" in _ids(db.latest_posts(con, scope="baseline"))
    assert db.effective_type_expr(curated=set()) == "first_type"


def test_leak_corrected_reports_whether_the_rule_applied(monkeypatch):
    """Publish this alongside any rate: uncorrected carries a known 6.7% bias."""
    monkeypatch.setattr(db, "curated_handles", lambda *a, **k: CURATED)
    assert db.leak_corrected() is True
    monkeypatch.setattr(db, "curated_handles", lambda *a, **k: set())
    assert db.leak_corrected() is False
