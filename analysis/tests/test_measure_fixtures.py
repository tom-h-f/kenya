"""CSV-backed regression: tests/data/measure_fixtures.csv."""

from pathlib import Path

import pandas as pd

from kma import measure
from kma.incitement import scan_text

FIXTURES = Path(__file__).parent / "fixtures" / "measure_fixtures.csv"


def test_measure_fixtures_csv_matches_helpers():
    df = pd.read_csv(FIXTURES)
    assert len(df) >= 8
    for _, row in df.iterrows():
        text = row["text"]
        hits, _ = scan_text(text)
        assert measure.domain_bucket(text) == row["domain"]
        dehum = None if pd.isna(row["dehum"]) else float(row["dehum"])
        viol = None if pd.isna(row["violence"]) else float(row["violence"])
        oth = None if pd.isna(row["othering"]) else float(row["othering"])
        pol = None if pd.isna(row["pol"]) else float(row["pol"])
        coded = measure.coded_suspect(hits, dehum, viol, oth, pol)
        assert bool(coded) is bool(row["coded_suspect"]), row["case"]
        label = row["label"]
        hate_flag = bool(row["hate_flag"])
        explicit = ((label != "neither") or hate_flag) and row["domain"] != "offdomain"
        assert bool(explicit) is bool(row["explicit_toxic"]), row["case"]
