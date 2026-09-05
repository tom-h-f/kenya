"""The A2 checks, run against synthetic CSVs in both the passing and the
failing shape.

Nothing here downloads the archive: the point of the script is that it can be
pointed at a small sample first, and a fixture IS a small sample. The passing
shape mirrors what was measured on the real mirror - stable hashed ids, text on
every row, URLs common and hashtags rare - and the failing shape is the one that
would kill the plan: a fresh hash per row, no text, no entities.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from kma import ioa_verify

COLUMNS = [
    "tweetid",
    "userid",
    "account_creation_date",
    "account_language",
    "tweet_language",
    "tweet_text",
    "tweet_time",
    "is_retweet",
    "retweet_userid",
    "retweet_tweetid",
    "hashtags",
    "urls",
]


def _write(path: Path, rows: list[dict]) -> str:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in COLUMNS})
    return str(path)


def _row(**kwargs) -> dict:
    row = {
        "tweetid": "1",
        "userid": "hashedaaa",
        "account_creation_date": "2013-04-01",
        "account_language": "ar",
        "tweet_language": "ar",
        "tweet_text": "a post about the election with several words",
        "tweet_time": "2018-06-15 09:00",
        "is_retweet": "false",
        "retweet_userid": "",
        "retweet_tweetid": "",
        "hashtags": "['WorldCup2018', 'WorldCup', 'Portugal']",
        "urls": "['https://example.org/a']",
    }
    row.update(kwargs)
    return row


@pytest.fixture
def healthy(tmp_path) -> str:
    """Two campaigns in contiguous blocks, with an embedded header between them,
    and stable pseudonyms carrying many tweets each."""
    rows = []
    for block, (language, prefix) in enumerate([("ar", "hashedA"), ("ru", "hashedR")]):
        for user in range(3):
            for tweet in range(20):
                rows.append(
                    _row(
                        tweetid=f"{block}{user}{tweet:02d}",
                        userid=f"{prefix}{user}",
                        account_language=language,
                        account_creation_date=f"201{block + 3}-04-01",
                        hashtags="['a', 'b', 'c']" if tweet % 4 == 0 else "[]",
                        urls="['https://example.org/a']" if tweet % 2 == 0 else "",
                        retweet_tweetid="900" if tweet % 5 == 0 else "",
                    )
                )
        if block == 0:
            rows.append({c: c for c in COLUMNS})
    # One public (non-anonymised) account, as in the real file's 2.5%.
    rows += [
        _row(tweetid=f"pub{i}", userid="12345678", account_language="ru") for i in range(4)
    ]
    return _write(tmp_path / "healthy.csv", rows)


@pytest.fixture
def broken(tmp_path) -> str:
    """The shape that would end the plan: a per-row hash, no text, no entities."""
    rows = [
        _row(
            tweetid=str(i),
            userid=f"hash{i:04d}",
            tweet_text="",
            hashtags="[]",
            urls="",
            account_language="ar" if i % 2 else "ru",
        )
        for i in range(40)
    ]
    return _write(tmp_path / "broken.csv", rows)


def _by_name(checks) -> dict:
    return {c.name: c for c in checks}


def test_stable_pseudonyms_pass_and_are_measured(healthy):
    check = _by_name(ioa_verify.verify(healthy))["id_stability"]
    assert check.passed is True
    assert check.detail["hashed"]["accounts"] == 6
    assert check.detail["hashed"]["mean_tweets"] == 20.0
    assert check.detail["hashed"]["singleton_share"] == 0.0
    # The public/anonymised split is reported separately, as in the real file.
    assert check.detail["public"]["accounts"] == 1


def test_per_row_hashes_fail_the_id_check(broken):
    """One row per id is the signature of per-row hashing, and it means no
    similarity network can be built at all."""
    check = _by_name(ioa_verify.verify(broken))["id_stability"]
    assert check.passed is False
    assert check.detail["hashed"]["singleton_share"] == 1.0
    assert "no similarity network" in check.note


def test_text_survival_is_measured_and_split_by_anonymisation(healthy):
    check = _by_name(ioa_verify.verify(healthy))["text_survival"]
    assert check.passed is True
    assert check.detail["text_share"] == 1.0
    assert check.detail["hashed_text_share"] == 1.0
    assert check.detail["mean_words"] == pytest.approx(8.0)


def test_stripped_text_fails(broken):
    check = _by_name(ioa_verify.verify(broken))["text_survival"]
    assert check.passed is False
    assert "text-similarity trace is unavailable" in check.note


def test_entity_survival_separates_any_hashtags_from_three(healthy):
    check = _by_name(ioa_verify.verify(healthy))["entity_survival"]
    assert check.passed is True
    assert check.detail["url_share"] == pytest.approx(0.5, abs=0.05)
    # `hashtag_sequence` needs three tags, so a bare hashtag share overstates it.
    assert check.detail["hashtag_share"] == pytest.approx(0.25, abs=0.05)
    assert check.detail["three_hashtag_share"] == pytest.approx(
        check.detail["hashtag_share"], abs=0.01
    )
    assert check.detail["retweet_share"] > 0


def test_missing_entities_fail(broken):
    check = _by_name(ioa_verify.verify(broken))["entity_survival"]
    assert check.passed is False
    assert "cannot be built" in check.note


def test_campaign_attribution_reports_evidence_and_never_a_labelling(healthy):
    check = _by_name(ioa_verify.verify(healthy))["campaign_attribution"]
    assert check.passed is None, "evidence, not a verdict"
    assert check.detail["embedded_header_rows"] == [61]
    assert check.detail["account_languages"] == 2
    assert check.detail["account_language_runs"] == 2, "contiguous blocks"
    assert check.detail["userid_runs"] == check.detail["userids"]
    assert "blocked" in check.note
    assert not any("country" in str(v).lower() for v in check.detail.values())


def test_campaign_attribution_detects_interleaving(broken):
    """Runs far above distinct values means file order does not segment it."""
    check = _by_name(ioa_verify.verify(broken))["campaign_attribution"]
    assert check.detail["account_language_runs"] > check.detail["account_languages"] * 2
    assert "interleaved" in check.note
    assert check.detail["embedded_header_rows"] == []


def test_limit_reads_only_the_head_of_the_file(healthy):
    """The script has to work on a sample before it works on 113 GB."""
    full = _by_name(ioa_verify.verify(healthy))["text_survival"]
    head = _by_name(ioa_verify.verify(healthy, limit=10))["text_survival"]
    assert head.detail["rows"] == 10 < full.detail["rows"]


def test_report_names_the_failures(healthy, broken):
    assert "no check failed" in ioa_verify.report(ioa_verify.verify(healthy))
    failing = ioa_verify.report(ioa_verify.verify(broken))
    assert "[FAIL] id_stability" in failing
    assert "3 check(s) failed" in failing
    assert "[EVIDENCE] campaign_attribution" in failing


def test_main_exits_nonzero_on_a_failed_check(healthy, broken, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["ioa_verify", healthy])
    assert ioa_verify.main() == 0
    monkeypatch.setattr("sys.argv", ["ioa_verify", broken])
    assert ioa_verify.main() == 1
    assert "FAIL" in capsys.readouterr().out
