from __future__ import annotations

import os
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

from twscrape import API
from twscrape.models import Tweet

from kenya_monitor.collectors.base import (
    Author,
    Collector,
    Engagement,
    FollowEdge,
    MetricSnapshot,
    Post,
)
from kenya_monitor.accounts import sync_accounts  # re-export for callers
from kenya_monitor.config import APP_ROOT
from kenya_monitor.pacing import human_pause

DEFAULT_DB_PATH = Path(os.getenv("TWS_ACCOUNTS_DB", APP_ROOT / "state" / "accounts.db"))

# Ignore anything older than this. Bounds the search windows and guards timelines.
MAX_AGE_DAYS = 14

Window = tuple[str, str]  # (since_date, until_date) as YYYY-MM-DD, UTC


def _cutoff() -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=MAX_AGE_DAYS)


def recent_windows(recent_days: int, now: datetime | None = None) -> list[Window]:
    """Daily windows covering the last `recent_days` calendar days (incl. today)."""
    today = (now or datetime.now(timezone.utc)).date()
    out: list[Window] = []
    for d in range(recent_days - 1, -1, -1):
        since = today - timedelta(days=d)
        out.append((since.isoformat(), (since + timedelta(days=1)).isoformat()))
    return out


def backfill_windows(
    recent_days: int, chunk_days: int, now: datetime | None = None
) -> list[Window]:
    """`chunk_days`-wide windows covering days `recent_days`..MAX_AGE_DAYS ago."""
    today = (now or datetime.now(timezone.utc)).date()
    floor = today - timedelta(days=MAX_AGE_DAYS)
    out: list[Window] = []
    d = recent_days
    while d < MAX_AGE_DAYS:
        until = today - timedelta(days=d)
        since = max(until - timedelta(days=chunk_days), floor)
        out.append((since.isoformat(), until.isoformat()))
        d += chunk_days
    return out


def full_window(now: datetime | None = None) -> list[Window]:
    """A single window spanning the whole retention period (for one-off collection)."""
    today = (now or datetime.now(timezone.utc)).date()
    since = today - timedelta(days=MAX_AGE_DAYS)
    return [(since.isoformat(), (today + timedelta(days=1)).isoformat())]


def build_api(db_path: Path = DEFAULT_DB_PATH) -> API:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return API(str(db_path))


def build_query(
    keyword: str,
    min_faves: int | None = None,
    since: str | None = None,
    until: str | None = None,
    include_retweets: bool = False,
) -> str:
    """X search query string. `include:nativeretweets` matters: without it the
    co-retweet channel only ever sees retweets from target timelines."""
    parts = [keyword]
    if min_faves:
        parts.append(f"min_faves:{min_faves}")
    if since:
        parts.append(f"since:{since}")
    if until:
        parts.append(f"until:{until}")
    if include_retweets:
        parts.append("include:nativeretweets")
    return " ".join(parts)


def _s(value) -> str | None:
    return str(value) if value is not None else None


def _mentions(tw: Tweet) -> list[str]:
    return [m.username for m in (tw.mentionedUsers or []) if m.username]


def _urls(tw: Tweet) -> list[str]:
    return [link.url for link in (tw.links or []) if link.url]


def _media_urls(media) -> list[str]:
    if media is None:
        return []
    out: list[str] = []
    for photo in getattr(media, "photos", None) or []:
        if photo.url:
            out.append(photo.url)
    for clip in (getattr(media, "videos", None) or []) + (getattr(media, "animated", None) or []):
        thumb = getattr(clip, "thumbnailUrl", None)
        if thumb:
            out.append(thumb)
    return out


class XCollector(Collector):
    platform = "x"

    def __init__(self, api: API):
        self.api = api
        self._authors: dict[str, Author] = {}

    def collected_authors(self) -> list[Author]:
        """Return authors seen since the last drain (latest snapshot per user), and clear."""
        authors = list(self._authors.values())
        self._authors = {}
        return authors

    async def search(
        self,
        keyword: str,
        limit: int,
        since: str | None = None,
        until: str | None = None,
        min_faves: int | None = None,
        product: str | None = None,
        include_retweets: bool = False,
    ) -> AsyncIterator[Post]:
        query = build_query(keyword, min_faves, since, until, include_retweets)
        kv = {"product": product} if product else None
        cutoff = _cutoff()
        async for tw in self.api.search(query, limit=limit, kv=kv):
            if tw.date.astimezone(timezone.utc) >= cutoff:
                post = self._to_post(tw)
                post.source_query = keyword
                yield post

    async def timeline(self, account: str, limit: int) -> AsyncIterator[Post]:
        user = await self.api.user_by_login(account)
        if user is None:
            return
        cutoff = _cutoff()
        async for tw in self.api.user_tweets(user.id, limit=limit):
            if tw.date.astimezone(timezone.utc) >= cutoff:
                yield self._to_post(tw)

    async def retweeters(self, post_id: str, limit: int) -> AsyncIterator[Engagement]:
        async for u in self.api.retweeters(int(post_id), limit=limit):
            self._authors[str(u.id)] = self._to_author(u)
            yield Engagement(
                platform=self.platform,
                platform_post_id=post_id,
                platform_user_id=str(u.id),
                kind="retweet",
            )

    async def replies(self, post_id: str, limit: int) -> AsyncIterator[Post]:
        cutoff = _cutoff()
        async for tw in self.api.tweet_replies(int(post_id), limit=limit):
            if tw.date.astimezone(timezone.utc) >= cutoff:
                yield self._to_post(tw)

    async def hydrate(self, post_ids: list[str]) -> AsyncIterator[Post]:
        """Fetch referenced posts by id; no age cutoff - an old original is
        still the object its retweets point at."""
        for i, pid in enumerate(post_ids):
            if i:
                await human_pause()
            tw = await self.api.tweet_details(int(pid))
            if tw is not None:
                yield self._to_post(tw)

    async def follows(self, handle: str, limit: int) -> AsyncIterator[FollowEdge]:
        user = await self.api.user_by_login(handle)
        if user is None:
            return
        uid = str(user.id)
        async for u in self.api.followers(user.id, limit=limit):
            self._authors[str(u.id)] = self._to_author(u)
            yield FollowEdge(platform=self.platform, follower_id=str(u.id), followed_id=uid)
        await human_pause()
        async for u in self.api.following(user.id, limit=limit):
            self._authors[str(u.id)] = self._to_author(u)
            yield FollowEdge(platform=self.platform, follower_id=uid, followed_id=str(u.id))

    async def refresh_metrics(self, post_ids: list[str]) -> AsyncIterator[MetricSnapshot]:
        for i, pid in enumerate(post_ids):
            if i:
                await human_pause()
            tw = await self.api.tweet_details(int(pid))
            if tw is None:
                continue
            yield MetricSnapshot(
                platform=self.platform,
                platform_post_id=str(tw.id),
                like_count=tw.likeCount or 0,
                reply_count=tw.replyCount or 0,
                repost_count=tw.retweetCount or 0,
                quote_count=tw.quoteCount or 0,
                view_count=tw.viewCount or 0,
            )

    def _to_author(self, u) -> Author:
        return Author(
            platform=self.platform,
            platform_user_id=str(u.id),
            handle=u.username,
            display_name=u.displayname or "",
            bio=u.rawDescription or "",
            location=u.location or "",
            followers_count=u.followersCount or 0,
            following_count=u.friendsCount or 0,
            tweet_count=u.statusesCount or 0,
            listed_count=u.listedCount or 0,
            verified=bool(u.verified),
            blue=bool(u.blue),
            created_at=u.created.astimezone(timezone.utc) if u.created else None,
            profile_image_url=u.profileImageUrl or "",
        )

    def _to_post(self, tw: Tweet) -> Post:
        self._authors[str(tw.user.id)] = self._to_author(tw.user)
        media_urls = _media_urls(tw.media)
        return Post(
            platform=self.platform,
            platform_post_id=str(tw.id),
            author_id=str(tw.user.id),
            author_handle=tw.user.username,
            text=tw.rawContent,
            created_at=tw.date.astimezone(timezone.utc),
            url=tw.url,
            lang=tw.lang or None,
            in_reply_to_id=_s(tw.inReplyToTweetId),
            is_repost=tw.retweetedTweet is not None,
            repost_of_id=_s(tw.retweetedTweet.id) if tw.retweetedTweet else None,
            like_count=tw.likeCount or 0,
            reply_count=tw.replyCount or 0,
            repost_count=tw.retweetCount or 0,
            quote_count=tw.quoteCount or 0,
            view_count=tw.viewCount or 0,
            hashtags=list(tw.hashtags or []),
            cashtags=list(tw.cashtags or []),
            mentions=_mentions(tw),
            urls=_urls(tw),
            quoted_post_id=_s(tw.quotedTweet.id) if tw.quotedTweet else None,
            is_quote=bool(tw.isQuoteStatus),
            conversation_id=_s(tw.conversationId),
            in_reply_to_user_id=_s(tw.inReplyToUser.id) if tw.inReplyToUser else None,
            source_label=tw.sourceLabel or None,
            place_name=(tw.place.fullName or tw.place.name) if tw.place else None,
            lat=tw.coordinates.latitude if tw.coordinates else None,
            lon=tw.coordinates.longitude if tw.coordinates else None,
            has_media=bool(media_urls),
            media_count=len(media_urls),
            media_urls=media_urls,
            collected_at=datetime.now(timezone.utc),
        )
