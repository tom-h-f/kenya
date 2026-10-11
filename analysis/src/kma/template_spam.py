"""Template-with-variation detector for a community's own posts.

A coupon-spam community survives the text trace because every post carries the
same frame and only filler words, emoji and links change (`kma.coord2` word
overlap cannot see that: the filler is a third of the characters). This module
asks a different question of a set of posts: how much of each post is wording
that several DIFFERENT accounts also wrote?

    stats = template_spam.template_stats(posts)       # posts: user_id, text
    label = template_spam.classify(stats)             # None | "template" | "template_promo"

The label is advisory. It does not remove a community or an account; it is a
flag for a reader, and the thresholds were set on a measurement (see
`analysis/investigations/2026-10-11-text-edge-floor/`).

Normalisation undoes the variation a spam generator adds: compatibility
characters (small-caps and superscript letters fold to ASCII), emoji and
zero-width characters (dropped or turned into a word break), leetspeak digits
inside a word, urls and mentions.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

SHINGLE = 3
MIN_TOKENS = 4
MIN_AUTHORS = 3
FRAME_SHARE = 0.5
MIN_POSTS = 4
TEMPLATE_RATE = 0.6
PROMO_RATE = 0.5
CAP_POSTS = 5000

_URL = re.compile(r"https?://\S+|www\.\S+", re.UNICODE)
_MENTION = re.compile(r"@\w+", re.UNICODE)
_LEET = str.maketrans("013457", "oieast")
_PROMO = re.compile(
    r"\b(coupons?|promo|promos|promocodes?|vouchers?|discounts?|cashback|referral|"
    r"giveaways?|kuponi|msimbo|punguzo)\b"
)


def normalise(text: object) -> str:
    """Lowercased words of a post with emoji, zero-width characters, urls,
    mentions, punctuation and in-word leetspeak digits removed."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", _MENTION.sub(" ", _URL.sub(" ", text)))
    kept = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat in ("Cf", "Mn", "Me"):
            continue
        kept.append(" " if cat[0] in "SP" or cat in ("Zs", "Zl", "Zp", "Cc") else ch)
    words = []
    for word in "".join(kept).lower().split():
        if re.search(r"[^\W\d_]", word) and re.search(r"\d", word):
            word = word.translate(_LEET)
        words.append(word)
    return " ".join(words)


def _shingles(tokens: list[str]) -> set[tuple[str, ...]]:
    return {tuple(tokens[k : k + SHINGLE]) for k in range(len(tokens) - SHINGLE + 1)}


@dataclass(frozen=True)
class TemplateStats:
    posts: int
    authors: int
    eligible: int
    template_rate: float
    promo_rate: float
    link_rate: float

    def as_dict(self) -> dict:
        return asdict(self)


def frame_shares(texts: list[str], authors: list[str]) -> np.ndarray:
    """Per post, the share of its word shingles that at least `MIN_AUTHORS`
    distinct accounts' posts contain. NaN for posts under `MIN_TOKENS` words."""
    tokens = [normalise(t).split() for t in texts]
    owners: dict[tuple[str, ...], set[str]] = {}
    shingled: list[set[tuple[str, ...]] | None] = []
    for toks, author in zip(tokens, authors):
        if len(toks) < MIN_TOKENS:
            shingled.append(None)
            continue
        sh = _shingles(toks)
        shingled.append(sh)
        for s in sh:
            owners.setdefault(s, set()).add(author)
    out = np.full(len(texts), np.nan)
    for k, sh in enumerate(shingled):
        if sh:
            out[k] = sum(len(owners[s]) >= MIN_AUTHORS for s in sh) / len(sh)
    return out


def template_stats(
    posts: pd.DataFrame, *, text: str = "text", author: str = "user_id", seed: int = 0,
    cap: int = CAP_POSTS,
) -> TemplateStats:
    """Template signals over one community's own posts (reposts excluded by the caller)."""
    posts = posts.dropna(subset=[text])
    if len(posts) > cap:
        posts = posts.sample(cap, random_state=seed)
    texts = posts[text].astype(str).tolist()
    authors = posts[author].astype(str).tolist()
    share = frame_shares(texts, authors)
    eligible = ~np.isnan(share)
    templated = eligible & (np.nan_to_num(share) >= FRAME_SHARE)
    n_el = int(eligible.sum())
    promo = np.array([bool(_PROMO.search(normalise(t))) for t in texts])
    links = np.array([bool(_URL.search(t)) for t in texts])
    n_t = int(templated.sum())
    return TemplateStats(
        posts=len(texts),
        authors=len(set(authors)),
        eligible=n_el,
        template_rate=round(n_t / n_el, 4) if n_el else 0.0,
        promo_rate=round(float(promo[templated].mean()), 4) if n_t else 0.0,
        link_rate=round(float(links[templated].mean()), 4) if n_t else 0.0,
    )


def classify(stats: TemplateStats) -> str | None:
    """`template_promo` when most posts share a frame and the frame is
    promotional; `template` when only the frame is shared; None otherwise."""
    if stats.eligible < MIN_POSTS or stats.template_rate < TEMPLATE_RATE:
        return None
    return "template_promo" if stats.promo_rate >= PROMO_RATE else "template"
