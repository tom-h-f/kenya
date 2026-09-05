"""Unit tests for kma.adjudicate (no API calls).

The judgement itself is validated in the investigation against the labelled set
from R4 - 6 confirmed information operations against corroborated pods. What is
under test here is the harness: that the prompt carries the evidence in the
order that matters, that replies are parsed robustly, and that one failure does
not take a triage list with it.
"""

from __future__ import annotations

import json

import pytest

from kma import adjudicate as adj

PACKET = {
    "cluster_id": 7,
    "size": 12,
    "shared_objects": [
        {"object_author": "big_account", "object_text": "Good morning X Family",
         "n_members": 9},
    ],
    "amplification_targets": [
        {"target_handle": "member_one", "acts": 300, "target_is_member": True},
        {"target_handle": "an_outsider", "acts": 50, "target_is_member": False},
    ],
    "representative_posts": [
        {"author_handle": "member_one", "text": "IEBC cannot be trusted"},
    ],
    "provenance": {
        "accounts_profiled": 12, "creation_span_days": 4000,
        "distinct_profile_images": 12, "share_empty_bio": 0.0,
        "share_default_image": 0.0,
    },
    "kenya": {"kenya_share": 0.02, "posts_classified": 500},
    "scores": {"inauthenticity_index": 0.7, "self_amplification": 0.4},
}


def test_the_prompt_leads_with_the_joint_co_action():
    """What they amplified TOGETHER is what the detector fired on. Individual
    posts are context, and a reader who sees them first over-reads them."""
    text = adj.render(PACKET)

    assert text.index("JOINTLY AMPLIFIED") < text.index("MEMBERS POST THEMSELVES")
    assert "Good morning X Family" in text


def test_targets_are_labelled_inside_or_outside_the_cluster():
    """Amplifying your own members and amplifying a politician are different
    claims; the label is what stops a reader conflating them."""
    text = adj.render(PACKET)

    assert "@member_one: 300 acts (cluster member)" in text
    assert "@an_outsider: 50 acts (outside the cluster)" in text


def test_provenance_reaches_the_reader():
    """12 distinct images across 12 accounts is the fact that argues hardest
    against a provisioned network, and no score in the packet carries it."""
    text = adj.render(PACKET)

    assert "distinct profile images: 12" in text
    assert "creation spread: 4000 days" in text


def test_the_keyword_gate_is_offered_as_a_proxy_not_an_answer():
    """It has documented blind spots. Presenting it as ground truth would make
    the reader inherit them, which is most of why a reader is here."""
    text = adj.render(PACKET) + adj.INSTRUCTIONS

    assert "0.02" in text
    assert "proxy" in text.lower()


def test_scores_are_framed_as_describing_coordination_not_purpose():
    """self_amplification was measured against confirmed operations and does
    NOT indicate intent. Handing it over unframed invites exactly the inference
    the ground truth refuted."""
    text = adj.render(PACKET)

    assert "not its purpose" in text


def test_the_system_prompt_sets_the_bar_for_the_worst_error():
    sys = adj.SYSTEM

    assert "deception" in sys.lower()
    assert "unclear" in sys
    assert "engagement" in sys.lower()


def test_the_taxonomy_is_not_binary():
    """'Pod or operation' is the framing that produced a refuted discriminator.
    Most coordinated clusters here are neither."""
    assert len(adj.CLUSTER_TYPES) > 2
    assert "engagement_pod" in adj.CLUSTER_TYPES
    assert "influence_operation" in adj.CLUSTER_TYPES
    assert "unclear" in adj.CLUSTER_TYPES


def test_a_fenced_reply_parses():
    reply = '```json\n{"cluster_type": "engagement_pod", "confidence": "high"}\n```'

    got = adj._parse(reply, cluster_id=3)

    assert got["cluster_type"] == "engagement_pod"
    assert got["cluster_id"] == 3


def test_a_reply_wrapped_in_prose_parses():
    reply = 'Here is my judgement:\n{"cluster_type": "commercial_spam"}\nHope that helps.'

    assert adj._parse(reply, 1)["cluster_type"] == "commercial_spam"


def test_an_unknown_label_becomes_unclear_but_is_not_lost():
    """A label outside the taxonomy is a prompt problem the caller must see,
    not a row to drop silently."""
    got = adj._parse('{"cluster_type": "bot_farm"}', 1)

    assert got["cluster_type"] == "unclear"
    assert got["cluster_type_raw"] == "bot_farm"


def test_an_unparseable_reply_raises():
    with pytest.raises(ValueError):
        adj._parse("I could not make a determination.", 1)


def test_one_failure_does_not_lose_the_rest_of_the_list(monkeypatch):
    """A triage list with a hole in it understates how much there is to look
    at, which is the failure mode that matters here."""
    calls = {"n": 0}

    def flaky(packet, client=None, model=None):
        calls["n"] += 1
        if packet["cluster_id"] == 2:
            raise RuntimeError("rate limited")
        return {"cluster_id": packet["cluster_id"], "cluster_type": "engagement_pod"}

    monkeypatch.setattr(adj, "judge", flaky)

    out = adj.judge_all([{"cluster_id": i} for i in (1, 2, 3)])

    assert [o["cluster_id"] for o in out] == [1, 2, 3]
    assert out[1]["cluster_type"] == "unclear"
    assert "rate limited" in out[1]["rationale"]


def test_prompts_need_no_client_or_key():
    """The path when no API key is configured - the packets are still usable by
    a human reader, which is the same layer three."""
    got = adj.prompts([PACKET])

    assert len(got) == 1 and got[0]["cluster_id"] == 7
    assert "JOINTLY AMPLIFIED" in got[0]["prompt"]


def test_judge_uses_an_injected_client(monkeypatch):
    class _Block:
        type = "text"
        text = json.dumps({"cluster_type": "fandom_or_interest", "confidence": "medium"})

    class _Client:
        class messages:
            @staticmethod
            def create(**kw):
                assert kw["system"] == adj.SYSTEM
                return type("R", (), {"content": [_Block()]})

    got = adj.judge(PACKET, client=_Client())

    assert got["cluster_type"] == "fandom_or_interest"
    assert got["cluster_id"] == 7
