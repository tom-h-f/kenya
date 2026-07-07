from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Post:
    platform: str
    platform_post_id: str
    author_id: str
    author_handle: str
    text: str
    created_at: datetime
    url: str
    source_query: str | None = None  # keyword that matched this post (search only)
    lang: str | None = None
    in_reply_to_id: str | None = None
    is_repost: bool = False
    repost_of_id: str | None = None
    like_count: int = 0
    reply_count: int = 0
    repost_count: int = 0
    quote_count: int = 0
    view_count: int = 0
    hashtags: list[str] = field(default_factory=list)
    cashtags: list[str] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    quoted_post_id: str | None = None
    is_quote: bool = False
    conversation_id: str | None = None
    in_reply_to_user_id: str | None = None
    source_label: str | None = None
    place_name: str | None = None
    lat: float | None = None
    lon: float | None = None
    has_media: bool = False
    media_count: int = 0
    media_urls: list[str] = field(default_factory=list)
    collected_at: datetime = field(default_factory=_now)

    def as_row(self) -> dict:
        return asdict(self)


@dataclass
class MetricSnapshot:
    platform: str
    platform_post_id: str
    like_count: int = 0
    reply_count: int = 0
    repost_count: int = 0
    quote_count: int = 0
    view_count: int = 0
    collected_at: datetime = field(default_factory=_now)

    def as_row(self) -> dict:
        return asdict(self)


@dataclass
class Author:
    platform: str
    platform_user_id: str
    handle: str
    display_name: str = ""
    bio: str = ""
    location: str = ""
    followers_count: int = 0
    following_count: int = 0
    tweet_count: int = 0
    listed_count: int = 0
    verified: bool = False
    blue: bool = False
    created_at: datetime | None = None
    profile_image_url: str = ""
    collected_at: datetime = field(default_factory=_now)

    def as_row(self) -> dict:
        return asdict(self)


class Collector(ABC):
    """Platform-agnostic contract. Every platform implements the same three methods."""

    platform: str

    def collected_authors(self) -> list[Author]:
        """Return authors seen since the last drain, and clear."""
        return []

    @abstractmethod
    async def search(
        self,
        keyword: str,
        limit: int,
        since: str | None = None,
        until: str | None = None,
        min_faves: int | None = None,
    ) -> AsyncIterator[Post]:
        """Yield posts matching a keyword/hashtag query."""
        raise NotImplementedError
        yield  # pragma: no cover  (marks this an async generator)

    @abstractmethod
    async def timeline(self, account: str, limit: int) -> AsyncIterator[Post]:
        """Yield recent posts from a specific account."""
        raise NotImplementedError
        yield  # pragma: no cover

    @abstractmethod
    async def refresh_metrics(self, post_ids: list[str]) -> AsyncIterator[MetricSnapshot]:
        """Yield fresh engagement snapshots for already-collected posts."""
        raise NotImplementedError
        yield  # pragma: no cover
