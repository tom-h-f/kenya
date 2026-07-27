"""Unit tests for kma.measure (no R2, no models)."""

import pandas as pd

from kma import measure as m


def test_domain_bucket_offdomain_us():
    text = (
        "I have a bill that could Remigrate Zohran Mamdani, and a lot of other "
        "foreigners who hate our country. Pass my Remigration Act!"
    )
    assert m.domain_bucket(text) == "offdomain"


def test_domain_bucket_kenya_politics():
    assert m.domain_bucket("Fukuza Ruto from State House #Wantam") == "kenya"
    assert m.domain_bucket(
        "@YusuphCabdi Tukimaliza kutoa wantam, next ni kutoa madoadoa Kenya"
    ) == "kenya"


def test_domain_bucket_ambiguous_when_neither():
    assert m.domain_bucket("The referee was terrible today") == "ambiguous"


def test_domain_bucket_kenya_wins_over_offdomain_markers():
    # Mentions America but is about Kenyan politics.
    assert (
        m.domain_bucket("Ruto visited America; Kenyans want Wantam at home")
        == "kenya"
    )


def test_min_fp_risk_orders_low_over_high():
    assert m.min_fp_risk(["nyoka", "madoadoa"]) == "low"
    assert m.min_fp_risk(["nyoka", "fukuza"]) == "high"
    assert m.min_fp_risk([]) is None
    assert m.min_fp_risk(None) is None


def test_coded_suspect_madoadoa_high_nli():
    hits, _ = __import__("kma.incitement", fromlist=["scan_text"]).scan_text(
        "Tukimaliza kutoa wantam, next ni kutoa madoadoa Kenya. Tuwe safi kama pamba"
    )
    assert "madoadoa" in hits
    assert m.coded_suspect(
        hits,
        dehumanisation=0.90,
        violence_call=0.80,
        othering=0.50,
        political_criticism=0.30,
    )


def test_coded_suspect_nyoka_mayai_high_nli():
    hits, _ = __import__("kma.incitement", fromlist=["scan_text"]).scan_text(
        "Yangu Iko, but nyoka na mayai yake lazima iende"
    )
    assert "nyoka" in hits
    assert m.min_fp_risk(hits) == "high"
    assert m.coded_suspect(
        hits,
        dehumanisation=0.90,
        violence_call=0.85,
        othering=0.40,
        political_criticism=0.20,
    )


def test_coded_suspect_fukuza_ruto_low_nli_not_coded():
    hits, _ = __import__("kma.incitement", fromlist=["scan_text"]).scan_text(
        "Gachagua in Kirinyaga rally right now. Fukuza Ruto"
    )
    assert "fukuza" in hits
    assert not m.coded_suspect(
        hits,
        dehumanisation=0.20,
        violence_call=0.15,
        othering=0.10,
        political_criticism=0.85,
    )


def test_coded_suspect_sports_nyoka_low_nli_not_coded():
    hits, _ = __import__("kma.incitement", fromlist=["scan_text"]).scan_text(
        "Toto na Tetesi: Nyoka wahangaisha wachezaji kombe la dunia."
    )
    assert "nyoka" in hits
    assert not m.coded_suspect(
        hits,
        dehumanisation=0.10,
        violence_call=0.10,
        othering=0.05,
        political_criticism=0.20,
    )


def test_coded_suspect_clean_politics_no_hits():
    hits, _ = __import__("kma.incitement", fromlist=["scan_text"]).scan_text(
        "The government must lower the cost of living before 2027."
    )
    assert hits == []
    assert not m.coded_suspect(
        hits,
        dehumanisation=0.90,
        violence_call=0.90,
        othering=0.90,
        political_criticism=0.10,
    )


def test_coded_suspect_null_nli_never_fires():
    assert not m.coded_suspect(
        ["madoadoa"],
        dehumanisation=None,
        violence_call=None,
        othering=None,
        political_criticism=None,
    )
    assert not m.coded_suspect(
        ["nyoka"],
        dehumanisation=None,
        violence_call=None,
        othering=None,
        political_criticism=0.9,
    )


def test_coded_suspect_offdomain_mamdani_no_lexicon():
    text = (
        "I have a bill that could Remigrate Zohran Mamdani, and a lot of other "
        "foreigners who hate our country."
    )
    hits, _ = __import__("kma.incitement", fromlist=["scan_text"]).scan_text(text)
    assert hits == []
    assert m.domain_bucket(text) == "offdomain"
    assert not m.coded_suspect(hits, 0.9, 0.9, 0.9, 0.1)


def test_attach_measurement_columns():
    df = pd.DataFrame(
        {
            "text": [
                "Fukuza Ruto #Wantam",
                "Remigrate Zohran Mamdani from America now",
                "next ni kutoa madoadoa Kenya. Tuwe safi kama pamba",
            ],
            "label": ["neither", "hate", "neither"],
            "hate_flag": [False, True, False],
            "dehumanisation_score": [0.2, 0.1, 0.9],
            "violence_call_score": [0.1, 0.1, 0.8],
            "othering_score": [0.1, 0.1, 0.5],
            "political_criticism_score": [0.8, 0.2, 0.2],
        }
    )
    out = m.attach_measurement_columns(df)
    assert list(out["domain"]) == ["kenya", "offdomain", "kenya"]
    assert list(out["in_kenya_scope"]) == [True, False, True]
    assert not bool(out.loc[1, "explicit_toxic"])  # off-domain hate excluded
    assert not bool(out.loc[0, "explicit_toxic"])
    assert bool(out.loc[2, "coded_suspect"])
    assert bool(out.loc[2, "coded_suspect_kenya"])
    assert not bool(out.loc[0, "coded_suspect"])
    assert "madoadoa" in out.loc[2, "lexicon_hits_live"]
