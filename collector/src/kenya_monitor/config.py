from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

APP_ROOT = Path(__file__).resolve().parents[2]  # collector/
MONOREPO_ROOT = APP_ROOT.parent

# R2 creds live in the shared root .env; also honour a real environment / local .env.
load_dotenv(MONOREPO_ROOT / ".env")
load_dotenv()

DEFAULT_TARGETS_PATH = APP_ROOT / "config" / "targets.yaml"
DEFAULT_ACCOUNTS_PATH = APP_ROOT / "config" / "accounts.yaml"
DEFAULT_HATE_TERMS_PATH = APP_ROOT / "config" / "hate_terms.yaml"

# Stratified search sampling (env-overridable). See the plan / README.
SEARCH_MIN_FAVES = int(os.getenv("SEARCH_MIN_FAVES", "0"))  # engagement floor per query (0 = none)
SEARCH_RECENT_DAYS = int(os.getenv("SEARCH_RECENT_DAYS", "2"))  # daily windows swept every run
SEARCH_BACKFILL_WINDOW_DAYS = int(os.getenv("SEARCH_BACKFILL_WINDOW_DAYS", "2"))  # older-window width
SEARCH_WINDOW_LIMIT = int(os.getenv("SEARCH_WINDOW_LIMIT", "20"))  # max posts per keyword per window

# CIB collection (docs/collection/cib-collection.md). Retweet-inclusive Latest
# search fixes the co-retweet blind spot; snowball turns hot objects into census
# slices; adaptive promotion closes the analysis -> collection loop.
SEARCH_PRODUCT = os.getenv("SEARCH_PRODUCT", "Latest")  # "Latest" beats relevance-ranking bias
SEARCH_INCLUDE_RETWEETS = os.getenv("SEARCH_INCLUDE_RETWEETS", "1") not in ("0", "false", "")

# Retweeter census is the dominant source of both new accounts and the
# co-amplification traces cluster detection consumes. Measured 2026-08-01: 179
# censused objects yielded 56,755 accounts, but only 31 of 8,293 toxic objects
# had ever been censused (0.4%). Raised hard to work through that backlog; the
# 12h per-object TTL means the cost falls once coverage catches up.
SNOWBALL_TOP_RETWEETED = int(os.getenv("SNOWBALL_TOP_RETWEETED", "250"))
SNOWBALL_TOP_CONVERSATIONS = int(os.getenv("SNOWBALL_TOP_CONVERSATIONS", "60"))
SNOWBALL_RETWEETERS_LIMIT = int(os.getenv("SNOWBALL_RETWEETERS_LIMIT", "300"))
SNOWBALL_REPLIES_LIMIT = int(os.getenv("SNOWBALL_REPLIES_LIMIT", "150"))
SNOWBALL_HYDRATE_LIMIT = int(os.getenv("SNOWBALL_HYDRATE_LIMIT", "50"))
SNOWBALL_LOOKBACK_DAYS = int(os.getenv("SNOWBALL_LOOKBACK_DAYS", "2"))
SNOWBALL_REFRESH_HOURS = int(os.getenv("SNOWBALL_REFRESH_HOURS", "12"))  # per-object TTL
# Objects per write. A single end-of-pass write meant a restart mid-census threw
# away every API call made since it started - ~40 minutes of pool budget at 250
# objects/pass.
SNOWBALL_FLUSH_EVERY = int(os.getenv("SNOWBALL_FLUSH_EVERY", "25"))

DYNAMIC_MAX_KEYWORDS = int(os.getenv("DYNAMIC_MAX_KEYWORDS", "10"))
DYNAMIC_MAX_ACCOUNTS = int(os.getenv("DYNAMIC_MAX_ACCOUNTS", "60"))
DYNAMIC_EXPIRY_DAYS = int(os.getenv("DYNAMIC_EXPIRY_DAYS", "7"))
DYNAMIC_HASHTAG_MIN_COUNT = int(os.getenv("DYNAMIC_HASHTAG_MIN_COUNT", "20"))  # last-24h floor
DYNAMIC_HASHTAG_RATIO = float(os.getenv("DYNAMIC_HASHTAG_RATIO", "5.0"))  # vs prior-7d daily avg
# Min story_suspicion_index for a flagged story's terms to be promoted (Phase 4).
STORY_FLAG_MIN_INDEX = float(os.getenv("STORY_FLAG_MIN_INDEX", "0.6"))
# Min supporting channels for a coordination cluster to drive targeting.
# Measured on the live corpus (2026-07-28): of 906 clusters / 4,732 accounts,
# 894 clusters (4,674 accounts) rest on a single channel and 12 clusters (58
# accounts) are corroborated across co_retweet AND co_reply. Promoting at
# n_channels >= 1 would hand nearly the whole active author base to the
# expansion passes, which is not a signal. Cross-channel corroboration is the
# strongest evidence available short of ground truth (docs/analysis/phase-3).
CLUSTER_MIN_CHANNELS = int(os.getenv("CLUSTER_MIN_CHANNELS", "1"))

# Hate-seeking collection (docs/collection/hate-seeking.md). Runs as its own
# cycle step with its own concurrency, never merged into the baseline keyword
# pool: sharing COLLECT_CONCURRENCY would starve baseline coverage, and sharing
# the `search` partition would destroy the sampling-bias separation that keeps
# prevalence measurable (kma.db.BASELINE_TYPES).
HATE_SEEK_ENABLED = os.getenv("HATE_SEEK_ENABLED", "0") not in ("0", "false", "")
HATE_SEEK_MAX_TERMS = int(os.getenv("HATE_SEEK_MAX_TERMS", "8"))  # per pass, rotated
HATE_TARGET_MAX_TERMS = int(os.getenv("HATE_TARGET_MAX_TERMS", "3"))  # ethnonym slots
HATE_MINE_MAX_TERMS = int(os.getenv("HATE_MINE_MAX_TERMS", "5"))  # mined-term slots
HATE_SEEK_WINDOW_LIMIT = int(os.getenv("HATE_SEEK_WINDOW_LIMIT", "15"))
HATE_SEEK_RECENT_DAYS = int(os.getenv("HATE_SEEK_RECENT_DAYS", "2"))
HATE_SEEK_EVERY_N_CYCLES = int(os.getenv("HATE_SEEK_EVERY_N_CYCLES", "3"))
HATE_SEEK_CONCURRENCY = int(os.getenv("HATE_SEEK_CONCURRENCY", "1"))
HATE_TERMS_PATH = Path(os.getenv("HATE_TERMS_PATH", DEFAULT_HATE_TERMS_PATH))
# Mined terms are machine-derived candidates for searches about ethnic hate.
# Promoting one automatically risks both false accusation and corpus bias, so
# the default is write-and-log; a human moves terms into hate_terms.yaml.
HATE_MINE_AUTOPROMOTE = os.getenv("HATE_MINE_AUTOPROMOTE", "0") not in ("0", "false", "")
HATE_SEED_MAX_ACCOUNTS = int(os.getenv("HATE_SEED_MAX_ACCOUNTS", "20"))  # seeds per expand pass
FOLLOW_CRAWL_TOP_HATE = int(os.getenv("FOLLOW_CRAWL_TOP_HATE", "10"))
# Expansion frontier. Without a ledger the top-N seeds are re-expanded every
# pass and the frontier never advances outward.
HATE_EXPAND_REFRESH_DAYS = int(os.getenv("HATE_EXPAND_REFRESH_DAYS", "3"))
HATE_EXPAND_MAX_ATTEMPTS = int(os.getenv("HATE_EXPAND_MAX_ATTEMPTS", "3"))
HATE_EXPAND_KEEP_DAYS = int(os.getenv("HATE_EXPAND_KEEP_DAYS", "30"))
# Rank this many times the batch size so the frontier has due candidates to pick.
FRONTIER_OVERSAMPLE = int(os.getenv("FRONTIER_OVERSAMPLE", "5"))

BURST_ZSCORE = float(os.getenv("BURST_ZSCORE", "3.0"))
BURST_MIN_POSTS = int(os.getenv("BURST_MIN_POSTS", "100"))  # hourly floor before a burst counts

FOLLOW_FETCH_LIMIT = int(os.getenv("FOLLOW_FETCH_LIMIT", "500"))  # edges per direction per account
FOLLOW_MAX_ACCOUNTS = int(os.getenv("FOLLOW_MAX_ACCOUNTS", "30"))  # accounts per pass
FOLLOW_CRAWL_REFRESH_DAYS = int(os.getenv("FOLLOW_CRAWL_REFRESH_DAYS", "30"))
FOLLOW_CRAWL_MAX_PER_RUN = int(os.getenv("FOLLOW_CRAWL_MAX_PER_RUN", "50"))

# Account pool / throughput (scale with pool size; see kenya_monitor.accounts).
TWS_ACCOUNT_ORDER = os.getenv("TWS_ACCOUNT_ORDER", "COALESCE(last_used, '1970-01-01') ASC")
COLLECT_CONCURRENCY = int(os.getenv("COLLECT_CONCURRENCY", "3"))  # parallel keyword workers
ACCOUNT_SYNC_HOURS = float(os.getenv("ACCOUNT_SYNC_HOURS", "6"))
# Continuous mode: cycles run back to back; the real throttle is per-account
# pacing + twscrape rate-limit rotation. The cooldown is just organic jitter.
CYCLE_COOLDOWN_MIN_S = int(os.getenv("CYCLE_COOLDOWN_MIN_S", "60"))
CYCLE_COOLDOWN_MAX_S = int(os.getenv("CYCLE_COOLDOWN_MAX_S", "300"))
FOLLOW_CRAWL_TOP_SUSPICIOUS = int(os.getenv("FOLLOW_CRAWL_TOP_SUSPICIOUS", "10"))
POSTS_MIN_GAP_HOURS = float(os.getenv("POSTS_MIN_GAP_HOURS", "3"))
POSTS_MAX_GAP_HOURS = float(os.getenv("POSTS_MAX_GAP_HOURS", "5"))
METRICS_MAX_POSTS_FLOOR = int(os.getenv("METRICS_MAX_POSTS_FLOOR", "200"))
METRICS_MAX_POSTS_PER_ACCOUNT = int(os.getenv("METRICS_MAX_POSTS_PER_ACCOUNT", "8"))

STATE_DIR = APP_ROOT / "state"
DYNAMIC_TARGETS_PATH = Path(os.getenv("DYNAMIC_TARGETS_PATH", STATE_DIR / "dynamic_targets.json"))
SNOWBALL_STATE_PATH = Path(os.getenv("SNOWBALL_STATE_PATH", STATE_DIR / "snowball.json"))
FOLLOW_CRAWL_STATE_PATH = Path(os.getenv("FOLLOW_CRAWL_STATE_PATH", STATE_DIR / "follow_crawl.json"))
HATE_SEEK_STATE_PATH = Path(os.getenv("HATE_SEEK_STATE_PATH", STATE_DIR / "hate_seek.json"))
MINED_TERMS_PATH = Path(os.getenv("MINED_TERMS_PATH", STATE_DIR / "mined_terms.json"))
HATE_EXPAND_STATE_PATH = Path(os.getenv("HATE_EXPAND_STATE_PATH", STATE_DIR / "hate_expand.json"))


@dataclass(frozen=True)
class R2Config:
    account_id: str
    access_key_id: str
    secret_access_key: str
    bucket: str

    @property
    def endpoint(self) -> str:
        return f"https://{self.account_id}.r2.cloudflarestorage.com"

    @classmethod
    def from_env(cls) -> R2Config:
        missing = [
            name
            for name in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")
            if not os.getenv(name)
        ]
        if missing:
            raise RuntimeError(f"missing R2 env vars: {', '.join(missing)} (see .env.example)")
        return cls(
            account_id=os.environ["R2_ACCOUNT_ID"],
            access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            bucket=os.environ["R2_BUCKET"],
        )


@dataclass(frozen=True)
class PlatformTargets:
    accounts: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class XAccount:
    username: str
    password: str = ""
    email: str = ""
    email_password: str = ""
    cookies: str = ""
    proxy: str = ""


def load_accounts(path: Path = DEFAULT_ACCOUNTS_PATH) -> list[XAccount]:
    if not path.exists():
        raise RuntimeError(
            f"{path} not found - copy config/accounts.example.yaml to config/accounts.yaml"
        )
    raw = yaml.safe_load(path.read_text()) or {}
    accounts = []
    for entry in raw.get("accounts") or []:
        username = (entry.get("username") or "").strip()
        if not username:
            continue
        accounts.append(
            XAccount(
                username=username,
                password=entry.get("password") or "",
                email=entry.get("email") or "",
                email_password=entry.get("email_password") or "",
                cookies=entry.get("cookies") or "",
                proxy=entry.get("proxy") or "",
            )
        )
    return accounts


FP_RISKS = ("low", "medium", "high")


@dataclass(frozen=True)
class HateTerm:
    """One coded-incitement search term. `query` is an X search fragment (it may
    contain OR groups and quoted phrases); `fp_risk` decides how much Kenya
    anchoring it gets at render time, never whether it is a verdict."""

    term: str
    query: str
    fp_risk: str
    category: str = ""
    notes: str = ""


@dataclass(frozen=True)
class HateTermSet:
    terms: list[HateTerm] = field(default_factory=list)
    anchors: dict[str, list[str]] = field(default_factory=dict)
    menace: list[str] = field(default_factory=list)
    ethnonyms: list[str] = field(default_factory=list)

    def by_name(self, name: str) -> HateTerm | None:
        return next((t for t in self.terms if t.term == name), None)


def load_hate_terms(path: Path = DEFAULT_HATE_TERMS_PATH) -> HateTermSet:
    """The coded-incitement search register (config/hate_terms.yaml).

    Mirrors `kma.incitement.LEXICON` as data - the collector carries no `kma`
    dependency. Terms are triage seeds documented in NCIC/PeaceTech advisories,
    not verdicts; most have innocent everyday senses, which is what `fp_risk`
    is for."""
    if not path.exists():
        raise RuntimeError(f"{path} not found - hate-seeking collection needs a term register")
    raw = yaml.safe_load(path.read_text()) or {}
    terms = []
    for entry in raw.get("terms") or []:
        name = (entry.get("term") or "").strip()
        query = (entry.get("query") or "").strip()
        risk = (entry.get("fp_risk") or "").strip()
        if not name or not query:
            continue
        if risk not in FP_RISKS:
            raise ValueError(f"hate term {name!r} has fp_risk {risk!r}, expected one of {FP_RISKS}")
        terms.append(
            HateTerm(
                term=name,
                query=query,
                fp_risk=risk,
                category=(entry.get("category") or "").strip(),
                notes=(entry.get("notes") or "").strip(),
            )
        )
    anchors = {k: list(v or []) for k, v in (raw.get("anchors") or {}).items()}
    for tier in ("core", "wide"):
        if not anchors.get(tier):
            raise ValueError(f"hate_terms.yaml is missing the {tier!r} anchor set")
    return HateTermSet(
        terms=terms,
        anchors=anchors,
        menace=list(raw.get("menace") or []),
        ethnonyms=list(raw.get("ethnonyms") or []),
    )


def load_targets(path: Path = DEFAULT_TARGETS_PATH) -> dict[str, PlatformTargets]:
    raw = yaml.safe_load(path.read_text()) or {}
    return {
        platform: PlatformTargets(
            accounts=list(cfg.get("accounts") or []),
            keywords=list(cfg.get("keywords") or []),
        )
        for platform, cfg in raw.items()
    }
