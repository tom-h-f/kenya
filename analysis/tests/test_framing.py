"""Unit tests for kma.framing (no R2 required)."""

import pandas as pd

from kma import framing as fr


def _stories():
    return pd.DataFrame(
        {
            "story_id": [0, 0, 1],
            "stable_story_id": ["aa", "aa", "bb"],
            "platform_post_id": ["p1", "p2", "p3"],
            "text": [
                "iebc rigged election results claim",
                "iebc election results disputed again",
                "weather sunny nairobi today morning",
            ],
            "created_at": pd.to_datetime(
                ["2026-07-08", "2026-07-09", "2026-07-08"], utc=True
            ),
        }
    )


def test_story_topics_maps_by_overlap():
    stories = _stories()
    topics_df = pd.DataFrame(
        {
            "platform_post_id": ["p1", "p2", "p3", "p9"],
            "topic": [3, 3, 7, 3],
            "text": ["a", "b", "c", "d"],
        }
    )
    summary = pd.DataFrame(
        {"topic": [3, 7], "terms": ["iebc, rigged, election", "weather, sunny"]}
    )
    out = fr.story_topics(stories, topics_df, summary)
    assert set(out["story_id"]) == {0, 1}
    row0 = out[out["story_id"] == 0].iloc[0]
    assert row0["topic_id"] == 3
    assert row0["overlap_n"] == 2
    assert "iebc" in row0["topic_terms"]


def test_story_topics_handles_noise_only():
    stories = _stories().query("story_id == 0")
    topics_df = pd.DataFrame(
        {"platform_post_id": ["p1", "p2"], "topic": [-1, -1]}
    )
    out = fr.story_topics(stories, topics_df)
    assert len(out) == 1
    assert out.iloc[0]["topic_id"] == -1


def test_story_topics_empty_inputs():
    assert fr.story_topics(pd.DataFrame(), pd.DataFrame()).empty


def test_sentiment_timeline_buckets():
    stories = _stories().query("story_id == 0")
    labels = pd.DataFrame(
        {
            "platform_post_id": ["p1", "p2"],
            "sentiment": ["negative", "negative"],
        }
    )
    out = fr.sentiment_timeline(stories, labels)
    assert not out.empty
    assert out["mean_sentiment"].iloc[0] == -1.0
    assert out["neg_share"].iloc[0] == 1.0


def test_sentiment_timeline_empty_neighborhood():
    stories = _stories()
    labels = pd.DataFrame(
        {"platform_post_id": ["other"], "sentiment": ["positive"]}
    )
    out = fr.sentiment_timeline(stories, labels)
    assert out.empty


def test_story_framing_bundle():
    stories = _stories()
    topics_df = pd.DataFrame(
        {"platform_post_id": ["p1", "p2", "p3"], "topic": [1, 1, 2]}
    )
    labels = pd.DataFrame(
        {"platform_post_id": ["p1", "p2"], "sentiment": ["neutral", "positive"]}
    )
    bundle = fr.story_framing(stories, topics_df, labels=labels)
    assert "topics" in bundle and "keywords" in bundle and "sentiment_timeline" in bundle
    assert 0 in bundle["keywords"]
    assert isinstance(bundle["keywords"][0], list)
