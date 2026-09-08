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
# Matched to the retweeted arm 2026-08-06, when the conversation arm was banded
# to the hub cap. At 60 unbanded it drew the busiest threads on the platform,
# most of which the analysis-side hub cap then deleted; banded, it draws from
# 14,608 in-band conversations and 60 slots per pass covers 0.4% of them. The
# retweeted arm covers ~9% of its band per pass, and that difference is what the
# two channels' pairable counts look like (27,758 vs 3,949).
SNOWBALL_TOP_CONVERSATIONS = int(os.getenv("SNOWBALL_TOP_CONVERSATIONS", "250"))
SNOWBALL_RETWEETERS_LIMIT = int(os.getenv("SNOWBALL_RETWEETERS_LIMIT", "300"))
SNOWBALL_REPLIES_LIMIT = int(os.getenv("SNOWBALL_REPLIES_LIMIT", "150"))
# One API call per id (`tweet_details`), so this is the expensive arm per unit.
# Raised from 50 because censused-but-unhydrated objects are now prioritised
# ahead of most-engaged refs, and there was a backlog of 2,805 of them on
# 2026-08-02 whose retweeter traces are unusable until their original is
# fetched. At 150 that drains in ~19 passes while the census adds ~250/pass.
SNOWBALL_HYDRATE_LIMIT = int(os.getenv("SNOWBALL_HYDRATE_LIMIT", "150"))
SNOWBALL_LOOKBACK_DAYS = int(os.getenv("SNOWBALL_LOOKBACK_DAYS", "2"))
SNOWBALL_REFRESH_HOURS = int(os.getenv("SNOWBALL_REFRESH_HOURS", "12"))  # per-object TTL
# Objects per write. A single end-of-pass write meant a restart mid-census threw
# away every API call made since it started - ~40 minutes of pool budget at 250
# objects/pass.
SNOWBALL_FLUSH_EVERY = int(os.getenv("SNOWBALL_FLUSH_EVERY", "25"))

# Census only objects inside a repost-count BAND. Measured 2026-08-01: 420 of
# 422 censused objects (99.5%) had >100 amplifiers, and the coordination hub cap
# discards exactly those - "pairs sharing only a mega-viral tweet are organic".
# Of 128,556 amplifier accounts collected, only 11,393 (8.9%) survived the cap
# and could ever become cluster members.
#
# The economics invert once you look: a hub costs ~3 paginated requests for ~300
# accounts that are all discarded; a mid-band object costs 1 request for ~27
# accounts that all count. Banding is both cheaper and strictly more useful.
#
# SNOWBALL_BAND_MAX must track the hub cap on the analysis side
# (kma.coordination.HUB_CAP_MAX), or the census silently fills with objects the
# projection will throw away.
SNOWBALL_BAND_MIN = int(os.getenv("SNOWBALL_BAND_MIN", "3"))
SNOWBALL_BAND_MAX = int(os.getenv("HUB_CAP_MAX", "100"))

# Warn when this share of fetched objects comes back above the band max. Those
# objects are discarded by the analysis hub cap, so the requests spent on them
# buy nothing. Pre-banding the rate was 99.5%; post-banding it measures 6.4%.
CENSUS_OVER_BAND_WARN = float(os.getenv("CENSUS_OVER_BAND_WARN", "0.15"))

# Backfill of missing retweet parents (kenya_monitor.parent_backfill).
#
# Measured on snapshot 2026-09-05-promotion-off: 228,272 distinct retweeted
# objects, 61,052 held, 167,220 missing. One `tweet_details` request per id with
# no batch lookup on the GraphQL path, so the whole backlog is days of pool
# budget and every pass has to be bounded.
#
# 500 per pass because the candidate query is a corpus-wide scan paid once per
# invocation. The comparable census-pool scan measured 215-464s against live R2,
# and 500 ids is the same order of work as SNOWBALL_TOP_RETWEETED's 250-object
# pass (~40 minutes), so the scan stays a minority of the pass rather than its
# dominant cost.
PARENT_BACKFILL_LIMIT = int(os.getenv("PARENT_BACKFILL_LIMIT", "500"))
# Ids per write, for the same reason as SNOWBALL_FLUSH_EVERY: a single
# end-of-pass write means a restart or rate-limit abort discards every API call
# made since it started, and pool budget is the expensive resource here.
PARENT_BACKFILL_FLUSH_EVERY = int(os.getenv("PARENT_BACKFILL_FLUSH_EVERY", "50"))
# Retries for ids whose REQUEST failed (rate limit, proxy, transport). An id
# that came back absent is never retried - see parent_backfill.is_retryable.
PARENT_BACKFILL_MAX_ATTEMPTS = int(os.getenv("PARENT_BACKFILL_MAX_ATTEMPTS", "3"))
# 0 means the whole corpus, which is the default because the backlog is
# historical: the point of the pass is to reach parents of retweets collected
# weeks ago. Set it to bound the amplifier scan if pi0 cannot carry the full
# one; the held-ids stage is never windowed either way.
PARENT_BACKFILL_LOOKBACK_DAYS = int(os.getenv("PARENT_BACKFILL_LOOKBACK_DAYS", "0")) or None
# How long a successful id stays in the ledger. Measured at 167,220 entries: a
# 19 MB file and 260 MB peak RSS for the dict alone, inside a 1 GB container
# that already gives DuckDB 600 MB. A fetched id has a post row, so the
# candidate query's anti-join owns it once R2's listing catches up and the
# ledger entry is only insurance for that window. Terminal failures are kept
# forever regardless - nothing else records them.
PARENT_BACKFILL_OK_RETAIN_HOURS = int(os.getenv("PARENT_BACKFILL_OK_RETAIN_HOURS", "168"))

# Timelines for accounts the retweeter census discovered.
#
# The census finds accounts we know ONLY as retweeter ids - no posts at all.
# Measured 2026-08-02: 168,356 such accounts against 73,876 with posts. They can
# never enter the `co_reply` channel (which needs a collected post carrying
# `in_reply_to_id`), so of the 49 accounts in the validated co_reply layer, 0
# appeared anywhere in co_retweet and corroboration was structurally impossible.
#
# Selection is RANDOM among census-discovered accounts, deliberately not by
# cluster membership or any coordination score. Promoting on the metric would
# give exactly the suspected accounts more posts, more traces and more edges,
# manufacturing the corroboration it was meant to measure - and the
# `cib_timeline` quarantine does not help, because coordination traces are built
# from all post types. Random selection is exogenous to the outcome, so it
# creates the population overlap without conditioning on the answer.
CENSUS_TIMELINE_ACCOUNTS = int(os.getenv("CENSUS_TIMELINE_ACCOUNTS", "20"))
CENSUS_TIMELINE_LIMIT = int(os.getenv("CENSUS_TIMELINE_LIMIT", "30"))  # posts per account
# Accounts with no tweets never gain a post row, so without this they would be
# re-picked forever.
CENSUS_TIMELINE_RETRY_DAYS = int(os.getenv("CENSUS_TIMELINE_RETRY_DAYS", "30"))
# The candidate query scans the engagements and authors globs over the network:
# 215s at 800MB/2 threads, and slower on pi0. Far too slow per cycle to pick 20
# handles, so a pool is drawn occasionally and spent across many cycles.
CENSUS_TIMELINE_POOL_SIZE = int(os.getenv("CENSUS_TIMELINE_POOL_SIZE", "500"))
CENSUS_TIMELINE_POOL_HOURS = int(os.getenv("CENSUS_TIMELINE_POOL_HOURS", "12"))

# Deep timelines for the accounts v2 surfaced (kenya_monitor.deep_timelines).
#
# Measured on snapshot 2026-09-05-promotion-off: 331,138 authors, 251,171
# (75.9%) with exactly one post, mean 3.2 posts per author, 79,967 clearing
# v2's 2-post/2-entity activity floor and only 8,047 with 20 or more posts.
# The corpus is wide and one post deep, which is the single fact behind most of
# v2's problems on it.
#
# 200 posts per account, NOT the ~3,200 the timeline endpoint can reach. At
# ~20 posts per page that is ~10 requests, and the marginal value of page 40 on
# one account is far below page 1 on another while the pool is the binding
# constraint. It is also the saturation boundary of the targeting strata: an
# account already holding 200 posts can at best double its history from a pass
# of this depth, so it ranks last.
DEEP_TIMELINE_DEPTH = int(os.getenv("DEEP_TIMELINE_DEPTH", "200"))
# Accounts per pass. Bounded on REQUESTS rather than accounts, to stay the same
# order of pool spend as PARENT_BACKFILL_LIMIT's 500 one-request ids: 50
# accounts x ~10 pages is ~500 requests.
DEEP_TIMELINE_LIMIT = int(os.getenv("DEEP_TIMELINE_LIMIT", "50"))
# Accounts per write. Lower than PARENT_BACKFILL_FLUSH_EVERY's 50 because the
# unit is ~10 requests rather than 1: 5 accounts is ~50 requests of pool budget
# at risk from a rate-limit abort, and ~1,000 post rows per parquet.
DEEP_TIMELINE_FLUSH_EVERY = int(os.getenv("DEEP_TIMELINE_FLUSH_EVERY", "5"))
# Retries for accounts whose REQUEST failed (rate limit, proxy, transport). An
# account that resolved but returned no posts is never retried - see
# deep_timelines.is_due.
DEEP_TIMELINE_MAX_ATTEMPTS = int(os.getenv("DEEP_TIMELINE_MAX_ATTEMPTS", "3"))
# How long before a deepened account is due again. 30 days, matching
# FOLLOW_CRAWL_REFRESH_DAYS and CENSUS_TIMELINE_RETRY_DAYS, because a refresh at
# a fixed depth only adds whatever the account posted since the last pass: at a
# shorter TTL most of the 200 posts fetched are ones already held, and 30 days
# is over twice the 14-day horizon baseline search can reach anyway.
DEEP_TIMELINE_REFRESH_DAYS = int(os.getenv("DEEP_TIMELINE_REFRESH_DAYS", "30"))
# 0 means the whole corpus, and it is the default for the same reason as
# PARENT_BACKFILL_LOOKBACK_DAYS: the held-post counts that decide the targeting
# strata are a statement about the account's whole history in our corpus, and a
# windowed count calls a long-held account thin and deepens it again.
DEEP_TIMELINE_LOOKBACK_DAYS = int(os.getenv("DEEP_TIMELINE_LOOKBACK_DAYS", "0")) or None
# Target-pool size for the suspicion FALLBACK ranking, used only before a v2
# pass has ever been persisted. 500 matches the size of a persisted v2 run
# (`coord2_run.run(top=500)`), so the fallback offers the same triage budget
# rather than an unbounded one.
DEEP_TIMELINE_FALLBACK_TARGETS = int(os.getenv("DEEP_TIMELINE_FALLBACK_TARGETS", "500"))
# Boundary between the "thin" and "has real history" targeting strata.
#
# 20 because that is where this corpus's own depth distribution breaks: 8,047 of
# 331,138 authors hold 20 or more posts, so the boundary sits at the 97.6th
# percentile of author depth. It exists because the more obvious boundary does
# not discriminate. v2's activity floor is applied BEFORE centrality, so every
# account in a persisted `kind=scores` run already holds 2+ posts and the
# below-the-floor stratum is empty by construction for that target set - without
# this boundary the strata collapse to a plain centrality ranking, which is the
# thing they exist to avoid.
DEEP_TIMELINE_THIN_POSTS = int(os.getenv("DEEP_TIMELINE_THIN_POSTS", "20"))

# DuckDB sizes its budget from the host, not the container cgroup, so on pi0 it
# plans against ~4GB while `mem_limit: 1g` kills it long before that. Bounded so
# a large scan spills to disk instead of taking the collector down.
COLLECTOR_MEMORY_LIMIT = os.getenv("COLLECTOR_MEMORY_LIMIT", "600MB")
COLLECTOR_THREADS = int(os.getenv("COLLECTOR_THREADS", "2"))

# How far back the suspicion ranking looks. It reads posts and authors, both of
# which grow without bound, so it is windowed rather than corpus-wide: the
# ranking picks crawl seeds, and an account with nothing recent is not a seed.
# Measured 2026-09-01 on a 47-day corpus: 2,165,609 post rows over 35 dt
# partitions, of which 1,466,911 fall in 30 days. Partition pruning is what
# makes the window cheap - the pruned count returns in 7.8s against 554s
# unpruned - so this bounds cost as the corpus ages rather than shrinking it now.
SUSPICION_LOOKBACK_DAYS = int(os.getenv("SUSPICION_LOOKBACK_DAYS", "30"))
# How long a materialised suspicion table may be reused. Both seed paths build
# the same table, and a cycle calls both: measured on pi0 2026-09-02, the build
# is 1,508s of the hate path's 3,738s, spent recomputing what the suspicion call
# minutes earlier already had. Well under the ~170min cycle, so a rebuild still
# happens once per cycle.
SUSPICION_CACHE_MINUTES = int(os.getenv("SUSPICION_CACHE_MINUTES", "60"))

DYNAMIC_MAX_KEYWORDS = int(os.getenv("DYNAMIC_MAX_KEYWORDS", "10"))
DYNAMIC_MAX_ACCOUNTS = int(os.getenv("DYNAMIC_MAX_ACCOUNTS", "60"))
DYNAMIC_EXPIRY_DAYS = int(os.getenv("DYNAMIC_EXPIRY_DAYS", "7"))
DYNAMIC_HASHTAG_MIN_COUNT = int(os.getenv("DYNAMIC_HASHTAG_MIN_COUNT", "20"))  # last-24h floor
DYNAMIC_HASHTAG_RATIO = float(os.getenv("DYNAMIC_HASHTAG_RATIO", "5.0"))  # vs prior-7d daily avg
# Min story_suspicion_index for a flagged story's terms to be promoted (Phase 4).
STORY_FLAG_MIN_INDEX = float(os.getenv("STORY_FLAG_MIN_INDEX", "0.6"))
# Min supporting channels for a coordination cluster to drive targeting.
# Promoting at n_channels >= 1 hands nearly the whole active author base to the
# expansion passes, which is not a signal: on 2026-07-28, 894 of 906 clusters
# rested on a single channel.
#
# This was lowered to 1 on 2026-08-01 because corroboration was then structurally
# 0 - the co_retweet and co_reply layers observed disjoint account populations,
# so a floor of 2 promoted nobody and cluster targeting was inert. That premise
# died with the census conversation-band fix: measured 2026-08-13, 446 bridge
# accounts and 485 pairs validated in both channels.
# Promotion of coordination-cluster members to timeline targets. Off during the
# v2 methodology rebuild (docs/plans/2026-09-05-v2-methodology.md §11): v2
# predicts accounts rather than clusters, so this path has no input until
# promotion is re-pointed at v2 centrality ranks.
CLUSTER_PROMOTION_ENABLED = os.getenv("CLUSTER_PROMOTION_ENABLED", "1") not in ("0", "false", "False")
CLUSTER_MIN_CHANNELS = int(os.getenv("CLUSTER_MIN_CHANNELS", "2"))
# Min share of a cluster's scored posts that must reference Kenya before it can
# promote accounts. Corroboration alone is the WRONG gate on its own: measured
# 2026-08-12, corroborated clusters are 8.3% Kenya-referencing against 51.5% for
# single-channel ones, because reciprocal engagement pods are the most abundant
# coordination on the platform and co_reply detects them best. Without this,
# raising the channel floor targets Ugandan, Nigerian and US pods harder.
CLUSTER_MIN_KENYA_SHARE = float(os.getenv("CLUSTER_MIN_KENYA_SHARE", "0.15"))
# Same gate for promoted keywords. A burst detector that measures only volume
# and acceleration promotes whatever is globally trending: #bbnaija,
# #thirstyformore and #citizenweekend all cleared it, and their posts landed in
# the BASELINE search partition.
KEYWORD_MIN_KENYA_SHARE = float(os.getenv("KEYWORD_MIN_KENYA_SHARE", "0.10"))

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
# Give up on an account after this many consecutive failed/unresolvable attempts.
# Without it a handle that can never resolve is re-tried every single run.
FOLLOW_CRAWL_MAX_ATTEMPTS = int(os.getenv("FOLLOW_CRAWL_MAX_ATTEMPTS", "3"))

# Account pool / throughput (scale with pool size; see kenya_monitor.accounts).
TWS_ACCOUNT_ORDER = os.getenv("TWS_ACCOUNT_ORDER", "COALESCE(last_used, '1970-01-01') ASC")
COLLECT_CONCURRENCY = int(os.getenv("COLLECT_CONCURRENCY", "3"))  # parallel keyword workers
ACCOUNT_SYNC_HOURS = float(os.getenv("ACCOUNT_SYNC_HOURS", "6"))
# Continuous mode: cycles run back to back; the real throttle is per-account
# pacing + twscrape rate-limit rotation. The cooldown is just organic jitter.
CYCLE_COOLDOWN_MIN_S = int(os.getenv("CYCLE_COOLDOWN_MIN_S", "60"))
CYCLE_COOLDOWN_MAX_S = int(os.getenv("CYCLE_COOLDOWN_MAX_S", "300"))
FOLLOW_CRAWL_TOP_SUSPICIOUS = int(os.getenv("FOLLOW_CRAWL_TOP_SUSPICIOUS", "10"))
# POSTS_MIN_GAP_HOURS / POSTS_MAX_GAP_HOURS removed 2026-08-13 along with
# `accounts.posts_gap_hours`. They described a wall-clock inter-pass gap that
# nothing ever called: `monitor run` cycles back to back, throttled only by
# per-account pacing and twscrape's rate-limit rotation. Both were documented as
# live in docs/collection/README.md, so an operator tuning them changed nothing.
METRICS_MAX_POSTS_FLOOR = int(os.getenv("METRICS_MAX_POSTS_FLOOR", "200"))
METRICS_MAX_POSTS_PER_ACCOUNT = int(os.getenv("METRICS_MAX_POSTS_PER_ACCOUNT", "8"))

STATE_DIR = APP_ROOT / "state"
# Where DuckDB spills when a scan exceeds COLLECTOR_MEMORY_LIMIT. It otherwise
# defaults to `.tmp` relative to the process cwd, which in the container is
# inside the image layer - so spill files are invisible to the state volume's
# sizing and are lost on redeploy. STATE_DIR is the mounted volume.
COLLECTOR_TEMP_DIR = Path(os.getenv("COLLECTOR_TEMP_DIR", STATE_DIR / "duckdb-tmp"))
DYNAMIC_TARGETS_PATH = Path(os.getenv("DYNAMIC_TARGETS_PATH", STATE_DIR / "dynamic_targets.json"))
SNOWBALL_STATE_PATH = Path(os.getenv("SNOWBALL_STATE_PATH", STATE_DIR / "snowball.json"))
CENSUS_TIMELINE_STATE_PATH = Path(
    os.getenv("CENSUS_TIMELINE_STATE_PATH", STATE_DIR / "census_timelines.json")
)
FOLLOW_CRAWL_STATE_PATH = Path(os.getenv("FOLLOW_CRAWL_STATE_PATH", STATE_DIR / "follow_crawl.json"))
HATE_SEEK_STATE_PATH = Path(os.getenv("HATE_SEEK_STATE_PATH", STATE_DIR / "hate_seek.json"))
PARENT_BACKFILL_STATE_PATH = Path(
    os.getenv("PARENT_BACKFILL_STATE_PATH", STATE_DIR / "parent_backfill.json")
)
DEEP_TIMELINE_STATE_PATH = Path(
    os.getenv("DEEP_TIMELINE_STATE_PATH", STATE_DIR / "deep_timeline.json")
)
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
