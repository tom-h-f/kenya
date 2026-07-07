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
