"""Unit tests for kma.dashboard (no R2, no models).

The producer's contract is that the payload is public by construction and that
a missing source reads as "not available" rather than as a number. Both are
asserted against a built payload, not trusted from the docstring.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from kma import dashboard as d
from kma import figures

NOW = datetime(2026, 10, 11, 4, 0, tzinfo=timezone.utc)
TODAY = NOW.date()


def _key(prefix: str, dt: str, n: int = 0, **parts) -> dict:
    mid = "/".join(f"{k}={v}" for k, v in parts.items())
    return {"Key": f"{prefix}/platform=x/{mid + '/' if mid else ''}dt={dt}/run={n}.parquet",
            "Size": 1000}


def _posts(days: list[str], types=("search", "replies", "hate_search")) -> pd.DataFrame:
    return d.parse_listing(_key("posts", dt, i, type=t)
                           for dt in days for i, t in enumerate(types))


def _rows(listing: pd.DataFrame, per_object: int = 10):
    def fn(keys: list[str]) -> dict[str, int]:
        return {k: per_object for k in keys}
    return fn


def _runs(computed_at: str = "2026-10-11T03:50:00+00:00", **over) -> pd.DataFrame:
    row = {
        "run_id": "20261011T033000Z", "window_id": "daily-2026-10-10-30d",
        "window_end": "2026-10-10", "window_days": 30, "computed_at": computed_at,
        "text_threshold": 0.85, "text_overlap": 0.1, "min_entities": 2, "min_kenya": 0.5,
        "networks_s": 12.3, "fuse_centrality_s": 4.5, "relevance_s": 8.0,
        "communities_s": 2.2,
        # Account-level and count columns the real table carries and the
        # payload must not.
        "listed_accounts": 355, "listed_communities": 17, "accounts_in_graph": 588,
        "communities": 40, "edges_in_graph": 9999,
    }
    return pd.DataFrame([{**row, **over}])


def _sources(days: int = 40, runs=None, rows=10) -> d.Sources:
    all_days = [date.fromordinal(TODAY.toordinal() - i).isoformat() for i in range(1, days + 1)]
    listings = {
        "posts/platform=x/": _posts(all_days),
        "embeddings/platform=x/": d.parse_listing(
            _key("embeddings", dt, 0, model="m") for dt in all_days[:30]),
        f"relevance/platform=x/model={d.relevance.MODEL}/": d.parse_listing(
            _key("relevance", dt, 0, model=d.relevance.MODEL) for dt in all_days[1:20]),
        "hatespeech/platform=x/": d.parse_listing(_key("hatespeech", dt) for dt in all_days),
        "incitement/platform=x/": d.parse_listing([]),
    }
    return d.Sources(
        list_prefix=lambda p: listings[p],
        object_rows=lambda keys: {k: rows for k in keys},
        daily_runs=lambda: runs,
    )


@pytest.fixture
def recorded():
    return figures.load_recorded()


@pytest.fixture
def payload(recorded):
    return d.build_summary(_sources(runs=_runs()), recorded, now=NOW, code_version="abc1234")


# --- public safety ----------------------------------------------------------


def test_the_built_payload_is_public_safe(payload):
    assert d.public_violations(payload) == []
    d.assert_public(payload)


def test_payload_is_json_serialisable_without_coercion(payload):
    assert json.loads(json.dumps(payload))["schema_version"] == d.SCHEMA_VERSION


@pytest.mark.parametrize("bad", [
    {"cards": {"x": {"user_id": "1"}}},
    {"note": "see @someaccount for more"},
    {"note": "https://x.com/someaccount/status/1"},
    {"id": "1234567890123456789"},
    {"id": 1234567890123456789},
    {"rows": [{"handle": "a"}]},
    {"author_id": 5},
])
def test_the_guard_rejects_identifying_values(bad):
    assert d.public_violations(bad)
    with pytest.raises(ValueError, match="not public-safe"):
        d.assert_public(bad)


def test_the_guard_passes_run_ids_and_dates():
    assert d.public_violations({"run_id": "20261011T033000Z", "dt": "2026-10-10", "n": 2469983}) == []


def test_no_account_level_column_of_the_daily_run_reaches_the_payload(payload):
    pipeline = json.dumps(payload["cards"]["pipeline"])
    for leaked in ("355", "588", "9999", "listed_accounts", "accounts_in_graph",
                   "listed_communities", "edges_in_graph"):
        assert leaked not in pipeline, leaked
    assert "user_id" not in json.dumps(payload)


def test_the_daily_listing_is_reported_only_as_non_empty(payload):
    card = payload["cards"]["pipeline"]
    assert card["listing_nonempty"] is True
    assert "listed_accounts" not in card
    assert payload["cards"]["detector"]["live_listing"]["published"] is False


def test_publish_refuses_the_raw_data_bucket(payload):
    from kma import db

    with pytest.raises(ValueError, match="raw data bucket"):
        d.publish(payload, bucket=db.BUCKET, client=object())


def test_publish_refuses_an_unsafe_payload():
    with pytest.raises(ValueError, match="not public-safe"):
        d.publish({"generated_at": NOW.isoformat(), "x": "@someone"},
                  bucket="kenya-monitor-dashboard", client=object())


class _Client:
    def __init__(self):
        self.puts = []

    def put_object(self, **kw):
        self.puts.append(kw)


def test_publish_writes_the_blob_and_a_dated_archive(payload):
    client = _Client()
    keys = d.publish(payload, bucket="kenya-monitor-dashboard", client=client)

    assert keys[0] == d.PUBLIC_KEY
    assert re.fullmatch(r"public/archive/dt=2026-10-11/run=20261011T040000Z\.json", keys[1])
    assert {p["Bucket"] for p in client.puts} == {"kenya-monitor-dashboard"}
    assert client.puts[0]["Metadata"] == {"generated_at": NOW.isoformat()}
    assert json.loads(client.puts[0]["Body"])["schema_version"] == d.SCHEMA_VERSION


def test_no_archive_flag_writes_one_object(payload):
    client = _Client()
    d.publish(payload, bucket="kenya-monitor-dashboard", client=client, archive=False)

    assert len(client.puts) == 1


# --- what must be withheld --------------------------------------------------


def test_withheld_names_the_unpublishable_items(payload):
    items = " ".join(w["item"].lower() for w in payload["withheld"])
    for needle in ("prevalence", "account", "coordinated", "toxicity", "corroboration"):
        assert needle in items


def test_no_toxicity_or_prevalence_or_coordination_figure_is_emitted(payload):
    keys = set()

    def walk(v):
        if isinstance(v, dict):
            keys.update(v)
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)

    walk({k: v for k, v in payload["cards"].items()})
    for banned in ("p_hate", "toxicity", "prevalence", "kenya_share", "n_clusters",
                   "clusters", "corroborated", "n_accounts", "edges_total"):
        assert banned not in keys, banned


def test_the_payload_has_no_v1_sections(payload):
    assert not ({"coordination", "toxicity", "network", "share_of_voice", "targeting"}
                & set(payload))


def test_every_ok_card_states_its_basis_and_source(payload):
    for card in payload["cards"].values():
        assert card["status"] in {"ok", "not_available"}
        assert card["source"]
        if card["status"] == "ok":
            assert card["basis"].strip(), card["id"]


# --- missing sources --------------------------------------------------------


def test_no_daily_run_yet_reads_as_not_available_not_zero(recorded):
    out = d.build_summary(_sources(runs=None), recorded, now=NOW)

    card = out["cards"]["pipeline"]
    assert card["status"] == "not_available"
    assert card["reason"] == d.REASON_NO_DAILY
    assert not ({"computed_at", "age_hours", "run_id", "stale"} & set(card))
    assert out["cards"]["detector"]["live_listing"]["status"] == "not_available"
    assert out["cards"]["detector"]["live_listing"]["listing_nonempty"] is None
    assert out["cards"]["collection"]["status"] == "ok"


def test_an_unreadable_source_publishes_a_fixed_reason_not_the_exception(recorded):
    src = _sources(runs=_runs())

    def boom():
        raise OSError("r2://secret-bucket/key=abc could not be opened")

    src.daily_runs = boom
    out = d.build_summary(src, recorded, now=NOW)

    card = out["cards"]["pipeline"]
    assert card["status"] == "not_available"
    assert card["reason"] == d.REASON_UNREADABLE
    assert "secret-bucket" not in json.dumps(out)


def test_it_refuses_to_publish_when_no_live_source_can_be_read(recorded):
    def boom(*_a, **_k):
        raise OSError("down")

    src = d.Sources(list_prefix=boom, object_rows=boom, daily_runs=boom)

    with pytest.raises(RuntimeError, match="not publishing"):
        d.build_summary(src, recorded, now=NOW)


def test_an_empty_posts_prefix_is_not_available(recorded):
    src = _sources(runs=_runs())
    inner = src.list_prefix
    src.list_prefix = lambda p: d.parse_listing([]) if p.startswith("posts") else inner(p)
    out = d.build_summary(src, recorded, now=NOW)

    assert out["cards"]["collection"]["status"] == "not_available"
    assert out["cards"]["enrichment"]["status"] == "not_available"


# --- pipeline card ----------------------------------------------------------


def test_pipeline_freshness_and_stale_flag():
    fresh = d.pipeline_card(_runs("2026-10-11T03:50:00+00:00"), NOW)
    old = d.pipeline_card(_runs("2026-10-09T03:50:00+00:00"), NOW)

    assert fresh["age_hours"] == pytest.approx(0.2, abs=0.05)
    assert fresh["stale"] is False
    assert old["stale"] is True
    assert old["age_hours"] == pytest.approx(48.2, abs=0.1)


def test_pipeline_uses_the_newest_run_when_several_exist():
    runs = pd.concat([_runs("2026-10-08T03:50:00+00:00", run_id="old"),
                      _runs("2026-10-10T03:50:00+00:00", run_id="new")])
    card = d.pipeline_card(runs, NOW)

    assert card["run_id"] == "new"
    assert card["runs_persisted"] == 2


def test_missing_run_days_are_counted_after_the_first_run_only():
    runs = pd.concat([_runs("2026-10-07T03:50:00+00:00"), _runs("2026-10-10T03:50:00+00:00")])
    card = d.pipeline_card(runs, NOW)

    assert card["missing_run_days"] == ["2026-10-08", "2026-10-09"]
    assert card["run_days"] == ["2026-10-07", "2026-10-10"]


def test_a_single_run_has_no_missing_days():
    card = d.pipeline_card(_runs("2026-10-10T03:50:00+00:00"), NOW)

    assert card["missing_run_days"] == []


def test_a_null_timing_stays_null():
    card = d.pipeline_card(_runs(relevance_s=None), NOW)

    assert card["stage_seconds"]["relevance_s"] is None


# --- collection card --------------------------------------------------------


def test_collection_sums_rows_per_day_and_splits_by_scope():
    posts = _posts(["2026-10-09", "2026-10-10"], types=("search", "replies", "hate_search", "control"))
    card = d.collection_card(posts, _rows(posts, 5), TODAY, days=2)

    last = card["series"][-1]
    assert last["dt"] == "2026-10-10"
    assert last["rows"] == 20
    assert last["objects"] == 4
    assert last["by_scope"] == {"baseline": 10, "targeted": 5, "control": 5, "other": 0}


def test_collection_excludes_the_open_day():
    posts = _posts(["2026-10-10", "2026-10-11"])
    card = d.collection_card(posts, _rows(posts), TODAY, days=3)

    assert [r["dt"] for r in card["series"]] == ["2026-10-08", "2026-10-09", "2026-10-10"]


def test_a_day_with_no_partition_is_a_gap_with_zero_rows():
    posts = _posts(["2026-10-07", "2026-10-09", "2026-10-10"])
    card = d.collection_card(posts, _rows(posts), TODAY, days=4)

    by_day = {r["dt"]: r for r in card["series"]}
    assert by_day["2026-10-08"]["gap"] is True
    assert by_day["2026-10-08"]["rows"] == 0
    assert by_day["2026-10-09"]["gap"] is False
    assert card["gap_days"] == ["2026-10-08"]
    assert card["gap_days_total"] == 1


def test_unknown_post_types_are_other_not_dropped():
    posts = _posts(["2026-10-10"], types=("brand_new_type",))
    card = d.collection_card(posts, _rows(posts, 7), TODAY, days=1)

    assert card["series"][0]["by_scope"]["other"] == 7
    assert card["series"][0]["rows"] == 7


def test_collection_refuses_a_partial_footer_answer():
    posts = _posts(["2026-10-10"])

    with pytest.raises(ValueError, match="every posts key"):
        d.collection_card(posts, lambda keys: {keys[0]: 1}, TODAY, days=1)


def test_collection_basis_says_rows_not_posts(payload):
    basis = payload["cards"]["collection"]["basis"]

    assert "not distinct posts" in basis
    assert "gap day" in basis


# --- enrichment card --------------------------------------------------------


def test_enrichment_lag_is_measured_against_the_newest_posts_partition(payload):
    stages = {s["stage"]: s for s in payload["cards"]["enrichment"]["stages"]}

    assert stages["posts"]["lag_days"] == 0
    assert stages["relevance"]["latest_dt"] == "2026-10-09"
    assert stages["relevance"]["lag_days"] == 1
    assert stages["embeddings"]["rows_last_7d"] == 70


def test_a_stage_with_no_partitions_has_null_figures_not_zero(payload):
    stage = {s["stage"]: s for s in payload["cards"]["enrichment"]["stages"]}["incitement"]

    assert stage["latest_dt"] is None
    assert stage["lag_days"] is None
    assert stage["rows_last_7d"] is None


# --- recorded cards ---------------------------------------------------------


def test_benchmark_card_separates_paper_from_reference_code(payload):
    card = payload["cards"]["benchmark"]
    iran = next(r for r in card["rows"] if r["dataset"] == "iran")

    assert iran["paper"] == 60.83
    assert iran["reference_code"] == 71.31
    assert iran["ours"] == 71.43
    assert abs(iran["delta"]) > 10
    assert card["datasets"] == 6
    assert card["within_0_2"] == 5
    assert card["metric"] == "Macro-F1"


def test_benchmark_headline_is_computed_from_rows_not_transcribed(recorded):
    block = json.loads(json.dumps(recorded["benchmark_reproduction"], default=str))
    block["rows"][0]["ours"] = block["rows"][0]["published"] + 3.0
    card = d.benchmark_card(block)

    assert card["within_0_2"] == 4


def test_relevance_gate_card_carries_the_label_count(payload):
    card = payload["cards"]["relevance_gate"]

    assert card["labelled_posts"] == 88
    assert {r["method"] for r in card["rows"]} == {"keyword gate", "learned classifier"}


def test_detector_card_is_framed_as_output_not_prevalence(payload):
    card = payload["cards"]["detector"]

    assert card["influence_operations_found"] == 0
    assert "not a rate" in card["basis"]
    assert card["snapshot"]
    assert all("accounts" not in r for r in card["rows"])


def test_negative_results_carry_a_figure_a_basis_and_a_source(payload):
    items = payload["cards"]["negative_results"]["items"]

    assert {i["id"] for i in items} == {
        "self_amplification", "text_similarity_threshold", "composition_drift"}
    for item in items:
        assert item["figure"] and item["basis"] and item["source"], item["id"]


def test_provenance_travels_with_every_recorded_card(payload):
    for name in ("benchmark", "relevance_gate", "detector", "negative_results"):
        assert payload["cards"][name]["measured"], name
