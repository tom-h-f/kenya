"""The text a relevance model sees must be the same in training and in serving."""

from __future__ import annotations

import json

import pytest

from kma import coord2, measure, relevance, relevance_prep

HANDLES = frozenset({"ruto_tutam2027", "naomikibandi"})


def test_strip_removes_every_mention_like_the_text_trace_does():
    for text in ("@a @b hello", "RT @x_y: ni mbaya", "no mentions", "a@b.com is not one"):
        assert relevance_prep.prepare(text, "strip").split() == coord2._MENTION.sub(" ", text).split()
        assert relevance.MENTION.sub(" ", text).split() == relevance_prep.prepare(text, "strip").split()


def test_the_kenya_terms_are_the_gates_terms():
    assert relevance_prep.KENYA_TERMS.pattern == measure._KENYA_RE.pattern
    assert relevance_prep.KENYA_TERMS.flags == measure._KENYA_RE.flags


def test_known_keeps_only_listed_handles():
    got = relevance_prep.prepare("@Ruto_tutam2027 @SomeForcePolice link", "known", HANDLES)
    assert "@Ruto_tutam2027" in got
    assert "SomeForcePolice" not in got


def test_known_lex_also_keeps_a_handle_that_names_kenya():
    text = "@KenyaPower_Care @WMPolice @Raila_Fan2027 hi"
    assert relevance_prep.prepare(text, "known", HANDLES).split() == ["hi"]
    got = relevance_prep.prepare(text, "known_lex", HANDLES)
    assert "@KenyaPower_Care" in got and "@Raila_Fan2027" in got
    assert "WMPolice" not in got


def test_the_lexicon_matches_whole_words_in_a_handle_not_substrings():
    assert not relevance_prep.handle_is_kenyan("Rutopia", frozenset(), "known_lex")
    assert relevance_prep.handle_is_kenyan("ruto_2027", frozenset(), "known_lex")


def test_an_unknown_variant_is_an_error_not_a_silent_strip():
    with pytest.raises(ValueError):
        relevance_prep.prepare("x", "keep_all")


def test_a_model_directory_records_its_own_preparation(tmp_path):
    assert relevance_prep.load_for_model(tmp_path) == ("strip", frozenset())
    (tmp_path / relevance_prep.PREP_FILE).write_text(json.dumps({"variant": "known_lex"}))
    (tmp_path / relevance_prep.HANDLES_FILE).write_text("A_B\n\nc_d\n")
    assert relevance_prep.load_for_model(tmp_path) == ("known_lex", frozenset({"a_b", "c_d"}))
