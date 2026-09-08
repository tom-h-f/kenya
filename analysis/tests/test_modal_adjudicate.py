"""The adjudication app's contract, without Modal or an API key.

The app body is a Modal function, so these test the decisions it encodes rather
than the wrapper: that a missing key fails loudly, that the unit is a real
parameter, and that provenance travels with a verdict.
"""

from __future__ import annotations

import pathlib

import pytest

APP = pathlib.Path(__file__).resolve().parents[1] / "modal_adjudicate.py"


@pytest.fixture(scope="module")
def source() -> str:
    return APP.read_text()


def test_both_secrets_are_referenced_unconditionally(source):
    """Conditional or file-based secrets break with a local/remote
    dependency-count mismatch, measured on modal_backfill.py."""
    assert 'modal.Secret.from_name("kenya-r2")' in source
    assert 'modal.Secret.from_name("anthropic")' in source
    assert "if os.getenv" not in source.split("SECRETS = [")[1].split("]")[0]


def test_a_missing_key_fails_loudly_and_names_the_fix(source):
    """The tf1 predecessor was dormant for two weeks because a 401 read as a
    no-op. A missing key must raise, and say how to fix it."""
    assert "raise RuntimeError(" in source
    assert "modal secret create anthropic" in source
    assert "--dry-run" in source


def test_the_unit_is_a_parameter_defaulting_to_campaign(source):
    """A3: judge the campaign, score the account. Both misses in the 2026-08-15
    adjudication were engagement-bait clusters, unjudgeable in isolation."""
    assert 'unit: str = "campaign"' in source
    assert 'unit == "campaign"' in source
    assert 'unit == "cluster"' in source


def test_an_unknown_unit_is_rejected_rather_than_defaulted(source):
    assert "unit must be campaign or cluster" in source


def test_provenance_is_written_as_columns(source):
    """A verdict without its model and unit cannot be audited or superseded."""
    assert 'frame["adjudicator"] = chosen' in source
    assert 'frame["unit"] = unit' in source


def test_dry_run_never_reaches_the_model(source):
    body = source.split("if dry_run:")[1].split("chosen =")[0]
    assert "judge_all" not in body


def test_the_model_is_overridable_not_hard_coded(source):
    assert "chosen = model or adjudicate.MODEL" in source


def test_it_does_not_write_to_the_v1_coordination_prefix_directly(source):
    """v1 wrote clusters to coordination/, v2 writes accounts to coord2/.
    Verdicts go through persist_verdicts rather than an ad-hoc COPY."""
    assert "coordination.persist_verdicts(" in source
    assert "COPY" not in source
