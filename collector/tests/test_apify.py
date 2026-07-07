from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from kenya_monitor.collectors.apify import ApifyXCollector, parse_twitter_date


def test_parse_twitter_date():
    # Test standard twitter format
    dt = parse_twitter_date("Fri Nov 24 17:49:36 +0000 2023")
    assert dt == datetime(2023, 11, 24, 17, 49, 36, tzinfo=timezone.utc)

    # Test ISO format
    dt = parse_twitter_date("2026-02-17T15:19:19.000Z")
    assert dt == datetime(2026, 2, 17, 15, 19, 19, tzinfo=timezone.utc)

    # Test invalid string defaults to now
    dt = parse_twitter_date("not-a-date")
    assert isinstance(dt, datetime)
    assert dt.tzinfo == timezone.utc


@pytest.mark.asyncio
async def test_apify_collector_search():
    mock_run = {"defaultDatasetId": "dataset_123"}
    mock_items = [
        {
            "id": "111",
            "text": "Hello #KenyaDecides2027",
            "createdAt": "Fri Nov 24 17:49:36 +0000 2023",
            "url": "https://twitter.com/user/status/111",
            "likeCount": 10,
            "author": {
                "id": "user_abc",
                "userName": "testuser",
                "name": "Test User",
            }
        }
    ]

    async def mock_iterate_items(*args, **kwargs):
        for item in mock_items:
            yield item

    # Patch ApifyClientAsync
    with patch("kenya_monitor.collectors.apify.ApifyClientAsync") as mock_client_cls:
        mock_client = MagicMock()
        mock_actor = MagicMock()
        mock_actor.call = AsyncMock(return_value=mock_run)
        mock_client.actor = MagicMock(return_value=mock_actor)

        mock_dataset = MagicMock()
        mock_dataset.iterate_items = mock_iterate_items
        mock_client.dataset = MagicMock(return_value=mock_dataset)

        mock_client_cls.return_value = mock_client

        collector = ApifyXCollector(token="test_token")
        
        # Test search
        # Since the mock item is from 2023 and cutoff is 14 days ago,
        # we will temporarily patch datetime to allow this item to pass,
        # or mock_items created_at can be set to now.
        # Let's set the mock item createdAt to now so it passes the cutoff.
        now_str = datetime.now(timezone.utc).strftime("%a %b %d %H:%M:%S +0000 %Y")
        mock_items[0]["createdAt"] = now_str

        results = []
        async for post in collector.search("keyword", limit=10):
            results.append(post)

        assert len(results) == 1
        post = results[0]
        assert post.platform_post_id == "111"
        assert post.text == "Hello #KenyaDecides2027"
        assert post.author_id == "user_abc"
        assert post.author_handle == "testuser"
        assert post.hashtags == ["KenyaDecides2027"]

        # Check collected authors
        authors = collector.collected_authors()
        assert len(authors) == 1
        assert authors[0].platform_user_id == "user_abc"
        assert authors[0].handle == "testuser"


@pytest.mark.asyncio
async def test_apify_collector_timeline():
    mock_run = {"defaultDatasetId": "dataset_123"}
    now_str = datetime.now(timezone.utc).strftime("%a %b %d %H:%M:%S +0000 %Y")
    mock_items = [
        {
            "id": "222",
            "text": "Timeline tweet",
            "createdAt": now_str,
            "url": "https://twitter.com/user/status/222",
            "author": {
                "id": "user_xyz",
                "userName": "timeline_user",
            }
        }
    ]

    async def mock_iterate_items(*args, **kwargs):
        for item in mock_items:
            yield item

    with patch("kenya_monitor.collectors.apify.ApifyClientAsync") as mock_client_cls:
        mock_client = MagicMock()
        mock_actor = MagicMock()
        mock_actor.call = AsyncMock(return_value=mock_run)
        mock_client.actor = MagicMock(return_value=mock_actor)

        mock_dataset = MagicMock()
        mock_dataset.iterate_items = mock_iterate_items
        mock_client.dataset = MagicMock(return_value=mock_dataset)

        mock_client_cls.return_value = mock_client

        collector = ApifyXCollector(token="test_token")
        
        results = []
        async for post in collector.timeline("@timeline_user", limit=5):
            results.append(post)

        assert len(results) == 1
        assert results[0].platform_post_id == "222"
        assert results[0].author_handle == "timeline_user"


@pytest.mark.asyncio
async def test_apify_collector_refresh_metrics():
    mock_run = {"defaultDatasetId": "dataset_123"}
    mock_items = [
        {
            "id": "111",
            "likeCount": 15,
            "retweetCount": 2,
            "replyCount": 1,
            "quoteCount": 3,
            "viewCount": 100,
        }
    ]

    async def mock_iterate_items(*args, **kwargs):
        for item in mock_items:
            yield item

    with patch("kenya_monitor.collectors.apify.ApifyClientAsync") as mock_client_cls:
        mock_client = MagicMock()
        mock_actor = MagicMock()
        mock_actor.call = AsyncMock(return_value=mock_run)
        mock_client.actor = MagicMock(return_value=mock_actor)

        mock_dataset = MagicMock()
        mock_dataset.iterate_items = mock_iterate_items
        mock_client.dataset = MagicMock(return_value=mock_dataset)

        mock_client_cls.return_value = mock_client

        collector = ApifyXCollector(token="test_token")
        
        results = []
        async for metric in collector.refresh_metrics(["111"]):
            results.append(metric)

        assert len(results) == 1
        assert results[0].platform_post_id == "111"
        assert results[0].like_count == 15
        assert results[0].repost_count == 2
        assert results[0].reply_count == 1
        assert results[0].quote_count == 3
        assert results[0].view_count == 100
