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

SNOWBALL_TOP_RETWEETED = int(os.getenv("SNOWBALL_TOP_RETWEETED", "15"))
SNOWBALL_TOP_CONVERSATIONS = int(os.getenv("SNOWBALL_TOP_CONVERSATIONS", "10"))
SNOWBALL_RETWEETERS_LIMIT = int(os.getenv("SNOWBALL_RETWEETERS_LIMIT", "300"))
SNOWBALL_REPLIES_LIMIT = int(os.getenv("SNOWBALL_REPLIES_LIMIT", "150"))
SNOWBALL_HYDRATE_LIMIT = int(os.getenv("SNOWBALL_HYDRATE_LIMIT", "50"))
SNOWBALL_LOOKBACK_DAYS = int(os.getenv("SNOWBALL_LOOKBACK_DAYS", "2"))
SNOWBALL_REFRESH_HOURS = int(os.getenv("SNOWBALL_REFRESH_HOURS", "12"))  # per-object TTL

DYNAMIC_MAX_KEYWORDS = int(os.getenv("DYNAMIC_MAX_KEYWORDS", "10"))
DYNAMIC_MAX_ACCOUNTS = int(os.getenv("DYNAMIC_MAX_ACCOUNTS", "20"))
DYNAMIC_EXPIRY_DAYS = int(os.getenv("DYNAMIC_EXPIRY_DAYS", "7"))
DYNAMIC_HASHTAG_MIN_COUNT = int(os.getenv("DYNAMIC_HASHTAG_MIN_COUNT", "20"))  # last-24h floor
DYNAMIC_HASHTAG_RATIO = float(os.getenv("DYNAMIC_HASHTAG_RATIO", "5.0"))  # vs prior-7d daily avg

BURST_ZSCORE = float(os.getenv("BURST_ZSCORE", "3.0"))
BURST_MIN_POSTS = int(os.getenv("BURST_MIN_POSTS", "100"))  # hourly floor before a burst counts

FOLLOW_FETCH_LIMIT = int(os.getenv("FOLLOW_FETCH_LIMIT", "500"))  # edges per direction per account
FOLLOW_MAX_ACCOUNTS = int(os.getenv("FOLLOW_MAX_ACCOUNTS", "30"))  # accounts per pass

# Account pool / throughput (scale with pool size; see kenya_monitor.accounts).
TWS_ACCOUNT_ORDER = os.getenv("TWS_ACCOUNT_ORDER", "COALESCE(last_used, '1970-01-01') ASC")
COLLECT_CONCURRENCY = int(os.getenv("COLLECT_CONCURRENCY", "3"))  # parallel keyword workers
REQUEST_DELAY_SCALE = float(os.getenv("REQUEST_DELAY_SCALE", "0"))  # 0 = auto-scale with pool
ACCOUNT_SYNC_HOURS = float(os.getenv("ACCOUNT_SYNC_HOURS", "6"))
POSTS_MIN_GAP_HOURS = float(os.getenv("POSTS_MIN_GAP_HOURS", "3"))
POSTS_MAX_GAP_HOURS = float(os.getenv("POSTS_MAX_GAP_HOURS", "5"))
METRICS_MAX_POSTS_FLOOR = int(os.getenv("METRICS_MAX_POSTS_FLOOR", "200"))
METRICS_MAX_POSTS_PER_ACCOUNT = int(os.getenv("METRICS_MAX_POSTS_PER_ACCOUNT", "8"))

STATE_DIR = APP_ROOT / "state"
DYNAMIC_TARGETS_PATH = Path(os.getenv("DYNAMIC_TARGETS_PATH", STATE_DIR / "dynamic_targets.json"))
SNOWBALL_STATE_PATH = Path(os.getenv("SNOWBALL_STATE_PATH", STATE_DIR / "snowball.json"))


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


def load_targets(path: Path = DEFAULT_TARGETS_PATH) -> dict[str, PlatformTargets]:
    raw = yaml.safe_load(path.read_text()) or {}
    return {
        platform: PlatformTargets(
            accounts=list(cfg.get("accounts") or []),
            keywords=list(cfg.get("keywords") or []),
        )
        for platform, cfg in raw.items()
    }
