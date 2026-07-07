from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone

from apify_client import ApifyClientAsync

from kenya_monitor.collectors.base import Author, Collector, MetricSnapshot, Post


def parse_twitter_date(dt_str: str | None) -> datetime:
    if not dt_str:
        return datetime.now(timezone.utc)
    # Try parsing common formats
    for fmt in ("%a %b %d %H:%M:%S %z %Y", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            dt = datetime.strptime(dt_str, fmt)
            return dt.astimezone(timezone.utc)
        except ValueError:
            pass
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(dt_str).astimezone(timezone.utc)
    except Exception:
        pass
    return datetime.now(timezone.utc)


class ApifyXCollector(Collector):
    platform = "x"

    def __init__(self, token: str, actor_id: str = "xquik/x-tweet-scraper"):
        self.client = ApifyClientAsync(token)
        self.actor_id = actor_id
        self._authors: dict[str, Author] = {}

    def collected_authors(self) -> list[Author]:
        """Return authors seen since the last drain (latest snapshot per user), and clear."""
        authors = list(self._authors.values())
        self._authors = {}
        return authors

    async def _run_actor_and_get_items(self, run_input: dict) -> list[dict]:
        log = logging.getLogger("kenya_monitor")
        log.info("Running Apify actor %s with input: %r", self.actor_id, run_input)
        try:
            run = await self.client.actor(self.actor_id).call(run_input=run_input)
            if not run:
                log.warning("Apify actor run returned None")
                return []

            dataset_id = None
            if hasattr(run, "default_dataset_id"):
                dataset_id = run.default_dataset_id
            elif isinstance(run, dict):
                dataset_id = run.get("defaultDatasetId") or run.get("default_dataset_id")

            if not dataset_id:
                log.warning("Apify actor run succeeded but defaultDatasetId is missing")
                return []

            items: list[dict] = []
            async_iterator = self.client.dataset(dataset_id).iterate_items()
            async for item in async_iterator:
                # Some runs write a runReport or diagnostics row. Filter those out.
                if isinstance(item, dict) and item.get("resultType") == "diagnostic":
                    log.info("Skipping diagnostic row: %r", item)
                    continue
                items.append(item)
            log.info("Apify actor run completed. Retrieved %d items.", len(items))
            return items
        except Exception as e:
            log.error("Apify actor run failed: %s", e)
            return []

    async def search(
        self,
        keyword: str,
        limit: int,
        since: str | None = None,
        until: str | None = None,
        min_faves: int | None = None,
    ) -> AsyncIterator[Post]:
        parts = [keyword]
        if min_faves:
            parts.append(f"min_faves:{min_faves}")
        if since:
            # Handle standard YYYY-MM-DD or xquik format (since field supports both)
            parts.append(f"since:{since}")
        if until:
            parts.append(f"until:{until}")
        query = " ".join(parts)

        # For xquik searchTerms is supported, but also twitterContent can be used.
        # We'll use searchTerms since it's the batch search input for both xquik and apidojo.
        run_input = {
            "searchTerms": [query],
            "maxItems": limit,
            "queryType": "Latest",
        }

        items = await self._run_actor_and_get_items(run_input)
        cutoff = datetime.now(timezone.utc) - timedelta(days=14)

        for item in items:
            post = self._to_post(item)
            if post.created_at >= cutoff:
                post.source_query = keyword
                yield post

    async def timeline(self, account: str, limit: int) -> AsyncIterator[Post]:
        handle = account.lstrip("@")
        if not handle:
            return

        run_input = {
            "twitterHandles": [handle],
            "maxItems": limit,
        }

        items = await self._run_actor_and_get_items(run_input)
        cutoff = datetime.now(timezone.utc) - timedelta(days=14)

        for item in items:
            post = self._to_post(item)
            if post.created_at >= cutoff:
                yield post

    async def refresh_metrics(self, post_ids: list[str]) -> AsyncIterator[MetricSnapshot]:
        if not post_ids:
            return

        # xquik has a dedicated tweetIds input for looking up tweets by ID
        # which is extremely fast and has no minimum requirement.
        run_input = {
            "tweetIds": post_ids,
            "maxItems": len(post_ids),
        }

        items = await self._run_actor_and_get_items(run_input)
        for item in items:
            pid = str(item.get("id"))
            if not pid:
                continue
            yield MetricSnapshot(
                platform=self.platform,
                platform_post_id=pid,
                like_count=item.get("likeCount", 0),
                reply_count=item.get("replyCount", 0),
                repost_count=item.get("retweetCount", 0),
                quote_count=item.get("quoteCount", 0),
                view_count=item.get("viewCount", 0),
            )

    def _to_author(self, author_data: dict) -> Author:
        created_at_str = author_data.get("createdAt") or author_data.get("created_at")
        created_at = None
        if created_at_str:
            try:
                created_at = parse_twitter_date(created_at_str)
            except Exception:
                pass

        return Author(
            platform=self.platform,
            platform_user_id=str(author_data.get("id", "")),
            handle=author_data.get("userName") or author_data.get("username") or "",
            display_name=author_data.get("name") or author_data.get("display_name") or author_data.get("displayName") or "",
            bio=author_data.get("description") or author_data.get("bio") or "",
            location=author_data.get("location") or "",
            followers_count=author_data.get("followers") or author_data.get("followersCount") or author_data.get("followers_count") or 0,
            following_count=author_data.get("following") or author_data.get("friendsCount") or author_data.get("followingCount") or author_data.get("following_count") or 0,
            tweet_count=author_data.get("statusesCount") or author_data.get("tweetsCount") or author_data.get("tweetCount") or author_data.get("statuses_count") or 0,
            listed_count=author_data.get("listedCount") or author_data.get("listed_count") or 0,
            verified=bool(author_data.get("verified") or author_data.get("isVerified") or False),
            blue=bool(author_data.get("isBlueVerified") or author_data.get("blue") or False),
            created_at=created_at,
            profile_image_url=author_data.get("profilePicture") or author_data.get("profileImageUrl") or author_data.get("profile_image_url") or "",
        )

    def _to_post(self, item: dict) -> Post:
        author_data = item.get("author", {})
        author_id = str(author_data.get("id", ""))
        if author_id:
            self._authors[author_id] = self._to_author(author_data)

        text = item.get("text", "")
        created_at = parse_twitter_date(item.get("createdAt"))

        # Parse media
        media_urls: list[str] = []
        for key in ("media", "images", "videos", "mediaUrls", "media_urls", "imageUrls", "videoUrls"):
            val = item.get(key)
            if isinstance(val, list):
                for m in val:
                    if isinstance(m, str):
                        media_urls.append(m)
                    elif isinstance(m, dict):
                        for url_key in ("url", "thumbnailUrl", "thumbnail_url", "native_url", "nativeUrl"):
                            u = m.get(url_key)
                            if u and isinstance(u, str):
                                media_urls.append(u)
        media_urls = list(dict.fromkeys(media_urls))

        # Parse entities (Xquik format)
        entities = item.get("entities") or {}
        hashtags = []
        cashtags = []
        mentions = []
        urls = []

        if isinstance(entities, dict):
            # Parse hashtags
            for h in entities.get("hashtags") or []:
                if isinstance(h, dict) and h.get("text"):
                    hashtags.append(h["text"])
                elif isinstance(h, str):
                    hashtags.append(h)

            # Parse cashtags
            for c in entities.get("symbols") or entities.get("cashtags") or []:
                if isinstance(c, dict) and c.get("text"):
                    cashtags.append(c["text"])
                elif isinstance(c, str):
                    cashtags.append(c)

            # Parse mentions
            for m in entities.get("user_mentions") or entities.get("mentions") or []:
                if isinstance(m, dict) and m.get("screen_name"):
                    mentions.append(m["screen_name"])
                elif isinstance(m, dict) and m.get("username"):
                    mentions.append(m["username"])
                elif isinstance(m, str):
                    mentions.append(m)

            # Parse urls
            for u in entities.get("urls") or []:
                if isinstance(u, dict) and u.get("expanded_url"):
                    urls.append(u["expanded_url"])
                elif isinstance(u, dict) and u.get("url"):
                    urls.append(u["url"])
                elif isinstance(u, str):
                    urls.append(u)

        # Fallbacks for apidojo format or plain-text regexes
        if not hashtags:
            raw_hashtags = item.get("hashtags") or []
            if isinstance(raw_hashtags, list):
                hashtags = [h for h in raw_hashtags if isinstance(h, str)]
            if not hashtags and text:
                hashtags = re.findall(r"#(\w+)", text)

        if not cashtags:
            raw_cashtags = item.get("cashtags") or []
            if isinstance(raw_cashtags, list):
                cashtags = [c for c in raw_cashtags if isinstance(c, str)]
            if not cashtags and text:
                cashtags = re.findall(r"\$(\w+)", text)

        if not mentions:
            raw_mentions = item.get("mentions") or []
            if isinstance(raw_mentions, list):
                mentions = [m for m in raw_mentions if isinstance(m, str)]
            if not mentions and text:
                mentions = re.findall(r"@(\w+)", text)

        if not urls:
            raw_urls = item.get("urls") or []
            if isinstance(raw_urls, list):
                urls = [u for u in raw_urls if isinstance(u, str)]
            if not urls and text:
                urls = re.findall(r"https?://[^\s]+", text)

        is_repost = bool(item.get("isRetweet") or item.get("is_retweet") or item.get("retweetedTweet") or item.get("retweetedStatus") or item.get("retweeted_status"))
        repost_of_id = None
        if is_repost:
            for key in ("retweetedTweet", "retweetedStatus", "retweeted_status"):
                if item.get(key):
                    repost_of_id = str(item[key].get("id"))
                    break

        quoted_post_id = None
        if item.get("quoteId"):
            quoted_post_id = str(item["quoteId"])
        elif item.get("quotedTweet"):
            quoted_post_id = str(item["quotedTweet"].get("id"))
        elif item.get("quoted_tweet"):
            quoted_post_id = str(item["quoted_tweet"].get("id"))

        place = item.get("place")
        place_name = None
        if isinstance(place, dict):
            place_name = place.get("fullName") or place.get("name")
        elif isinstance(place, str):
            place_name = place

        coordinates = item.get("coordinates") or item.get("geo")
        lat, lon = None, None
        if isinstance(coordinates, dict):
            if coordinates.get("latitude") is not None:
                lat = coordinates.get("latitude")
                lon = coordinates.get("longitude")
            elif isinstance(coordinates.get("coordinates"), list) and len(coordinates["coordinates"]) == 2:
                # GeoJSON coordinates format: [longitude, latitude]
                lat = coordinates["coordinates"][1]
                lon = coordinates["coordinates"][0]

        in_reply_to_user_id = None
        in_reply_to_user = item.get("inReplyToUser") or item.get("in_reply_to_user")
        if isinstance(in_reply_to_user, dict):
            in_reply_to_user_id = str(in_reply_to_user.get("id"))
        elif item.get("inReplyToUserId"):
            in_reply_to_user_id = str(item["inReplyToUserId"])
        elif item.get("in_reply_to_user_id"):
            in_reply_to_user_id = str(item["in_reply_to_user_id"])

        return Post(
            platform=self.platform,
            platform_post_id=str(item.get("id", "")),
            author_id=author_id,
            author_handle=author_data.get("userName") or author_data.get("username") or "",
            text=text,
            created_at=created_at,
            url=item.get("url") or item.get("twitterUrl") or item.get("tweetUrl") or "",
            lang=item.get("lang"),
            in_reply_to_id=str(item["inReplyToTweetId"]) if item.get("inReplyToTweetId") else (str(item["in_reply_to_status_id"]) if item.get("in_reply_to_status_id") else None),
            is_repost=is_repost,
            repost_of_id=repost_of_id,
            like_count=item.get("likeCount") or item.get("like_count") or 0,
            reply_count=item.get("replyCount") or item.get("reply_count") or 0,
            repost_count=item.get("retweetCount") or item.get("retweet_count") or 0,
            quote_count=item.get("quoteCount") or item.get("quote_count") or 0,
            view_count=item.get("viewCount") or item.get("view_count") or 0,
            hashtags=hashtags,
            cashtags=cashtags,
            mentions=mentions,
            urls=urls,
            quoted_post_id=quoted_post_id,
            is_quote=bool(item.get("isQuote") or item.get("isQuoteStatus") or item.get("is_quote_status") or False),
            conversation_id=str(item["conversationId"]) if item.get("conversationId") else (str(item["conversation_id"]) if item.get("conversation_id") else None),
            in_reply_to_user_id=in_reply_to_user_id,
            source_label=item.get("source") or item.get("sourceLabel") or item.get("source_label"),
            place_name=place_name,
            lat=lat,
            lon=lon,
            has_media=bool(media_urls),
            media_count=len(media_urls),
            media_urls=media_urls,
            collected_at=datetime.now(timezone.utc),
        )
