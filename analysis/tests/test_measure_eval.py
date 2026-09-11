"""The gate evaluator has to be trustworthy before the gate's number is."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kma import measure_eval


def _labelled(rows):
    return pd.DataFrame(rows, columns=["bucket", "stratum_share", "label"])


def test_score_is_perfect_when_the_gate_is():
    df = _labelled(
        [("kenya", 0.3, "kenya")] * 10 + [("offdomain", 0.7, "offdomain")] * 10
    )
    got = measure_eval.score(df)
    assert got["precision"] == 1.0
    assert got["recall"] == 1.0


def test_score_weights_strata_back_to_corpus_rates():
    """Equal sample sizes must not imply equal corpus sizes: a rare stratum
    sampled heavily should not dominate the estimate."""
    df = _labelled(
        [("kenya", 0.02, "kenya")] * 10
        + [("offdomain", 0.98, "kenya")] * 10  # gate missed all of these
    )
    got = measure_eval.score(df)
    # Recall must reflect that the misses live in the 98% stratum.
    assert got["recall"] < 0.05


def test_unclear_rows_are_excluded_and_reported():
    df = _labelled(
        [("kenya", 0.5, "kenya")] * 5 + [("kenya", 0.5, "unclear")] * 3
    )
    got = measure_eval.score(df)
    assert got["unclear"] == 3
    assert got["labelled"] == 5


def test_unlabelled_rows_are_counted_not_guessed():
    df = _labelled([("kenya", 0.5, "kenya")] * 4 + [("kenya", 0.5, "")] * 6)
    got = measure_eval.score(df)
    assert got["unlabelled"] == 6
    assert got["labelled"] == 4


def test_score_refuses_an_unlabelled_sheet():
    df = _labelled([("kenya", 0.5, "")] * 5)
    with pytest.raises(ValueError, match="fill in the"):
        measure_eval.score(df)


def test_wilson_interval_stays_inside_the_unit_range():
    """A normal interval on 10/10 runs above 1.0 and reads as precision over
    100%, which is why this is Wilson."""
    low, high = measure_eval._wilson(10, 10)
    assert 0.0 <= low <= 1.0 and 0.0 <= high <= 1.0
    assert high == 1.0


def test_wilson_handles_an_empty_stratum():
    low, high = measure_eval._wilson(0, 0)
    assert np.isnan(low) and np.isnan(high)


def test_sample_is_blind_and_carries_stratum_weights():
    """The labeller must not see the gate's guess, but the scorer needs it."""
    cols = ["post_id", "user_id", "text", "label", "bucket", "stratum_share"]
    frame = pd.DataFrame(columns=cols)
    assert list(frame.columns).index("label") < list(frame.columns).index("bucket")


def _model_labelled(per_bucket=40):
    rows = []
    for bucket, share in (("kenya", 0.1), ("offdomain", 0.2), ("ambiguous", 0.7)):
        label = "offdomain" if bucket == "offdomain" else "kenya"
        rows += [(f"{bucket}{k}", f"text {k}", label, bucket, share) for k in range(per_bucket)]
    return pd.DataFrame(rows, columns=["post_id", "text", "label", "bucket", "stratum_share"])


def _copy_model_labels(sheet, sample):
    return sheet.assign(label=sample.set_index("post_id").loc[sheet["post_id"], "label"].to_numpy())


def test_human_sheet_hides_both_the_gate_and_the_model():
    sheet = measure_eval.human_subset(_model_labelled(), n=100)
    assert list(sheet.columns) == ["post_id", "text", "label"]
    assert (sheet["label"] == "").all()


def test_human_sheet_is_stratified_to_the_requested_size():
    sample = _model_labelled()
    sheet = measure_eval.human_subset(sample, n=100)
    assert len(sheet) == 100
    per_bucket = sample.set_index("post_id").loc[sheet["post_id"], "bucket"].value_counts()
    assert sorted(per_bucket.tolist()) == [33, 33, 34]


def test_agreement_is_perfect_when_the_human_matches_the_model():
    sample = _model_labelled()
    sheet = _copy_model_labels(measure_eval.human_subset(sample, n=60), sample)
    got = measure_eval.agreement(sheet, sample)
    assert got["agreement"] == 1.0
    assert got["kappa"] == 1.0
    assert got["gate_vs_human"] == got["gate_vs_model"]


def test_agreement_compares_only_the_posts_the_human_labelled():
    """The model's labels on the other posts must not leak into either score."""
    sample = _model_labelled()
    sheet = _copy_model_labels(measure_eval.human_subset(sample, n=30), sample)
    got = measure_eval.agreement(sheet, sample)
    assert got["posts"] == 30
    assert got["gate_vs_model"]["labelled"] == 30


def test_agreement_rejects_a_typo_rather_than_dropping_it():
    sample = _model_labelled()
    sheet = measure_eval.human_subset(sample, n=30).assign(label="kenyaa")
    with pytest.raises(ValueError, match="kenyaa"):
        measure_eval.agreement(sheet, sample)


def test_confusion_is_keyed_by_the_human_label_first():
    """A pandas crosstab's plain to_dict() keys by column - the model's label -
    which silently transposes the matrix the key's name promises."""
    sample = _model_labelled(per_bucket=10)
    sheet = _copy_model_labels(measure_eval.human_subset(sample, n=30), sample)
    first = sheet["post_id"].iloc[0]
    model_label = sample.set_index("post_id").loc[first, "label"]
    sheet.loc[sheet["post_id"] == first, "label"] = "unclear"
    confusion = measure_eval.agreement(sheet, sample)["confusion_human_by_model"]
    assert confusion["unclear"][model_label] == 1
