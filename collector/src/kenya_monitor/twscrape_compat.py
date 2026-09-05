"""Route SearchTimeline through POST.

X started answering GET /i/api/graphql/<id>/SearchTimeline with an empty 404 on
2026-08-28; the same query id succeeds as a POST with the params in a JSON body.
twscrape (0.20.1) still issues every GraphQL query as GET, so search returns
nothing and each 404 locks the account out of the queue for 15 minutes, which
looks exactly like a quiet dataset. Every other operation still answers GET.

Drop this module once twscrape ships the POST switch upstream.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.parse import urlparse

from twscrape import queue_client as qc

POST_ONLY_OPS = frozenset({"SearchTimeline"})

_orig_req = qc.Ctx.req


def _decode(params: dict[str, Any]) -> dict[str, Any]:
    return {k: json.loads(v) if isinstance(v, str) else v for k, v in params.items()}


async def _req(self, method: str, url: str, params: dict[str, Any] | None = None):
    if method != "GET" or params is None or url.rsplit("/", 1)[-1] not in POST_ONLY_OPS:
        return await _orig_req(self, method, url, params=params)

    body = _decode(params)
    body["queryId"] = url.rsplit("/", 2)[-2]
    path = urlparse(url).path or "/"

    tries = 0
    while tries < 3:
        gen = await qc.XClIdGenStore.get(
            self.acc.username,
            proxy=self.proxy,
            cookies=self.acc.cookies,
            fresh=tries > 0,
        )
        hdr = {"x-client-transaction-id": gen.calc("POST", path)}
        rep = await self.clt.request("POST", url, json=body, headers=hdr)
        if rep.status_code != 404:
            return rep

        tries += 1
        await asyncio.sleep(1)

    raise qc.AbortReqError(f"SearchTimeline POST kept returning 404: {url}")


def install() -> None:
    qc.Ctx.req = _req
