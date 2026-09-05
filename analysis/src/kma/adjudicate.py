"""Turn a cluster dossier into a structured triage judgement.

Layer three of the pipeline. Layers one and two - detect coordination, filter
to Kenyan relevance - are code. This one is a reader, because intent is not in
the co-action graph: the framework `coordination.py` implements is
intent-agnostic by construction, and the statistical discriminator built to
replace a reader was measured against confirmed operations and refuted (see
analysis/investigations/2026-08-15-io-ground-truth).

    from kma import dossier, adjudicate
    packets = dossier.build(con, members, cluster_ids=corroborated)
    verdicts = adjudicate.judge_all(packets)          # needs ANTHROPIC_API_KEY
    prompts  = adjudicate.prompts(packets)            # without one

TRIAGE, NEVER A VERDICT. The output ranks clusters for human attention and
records why. It does not establish that any account is inauthentic, and nothing
here may be published as a finding about a named account - the same discipline
`dashboard.py` enforces structurally, enforced here by convention because the
input deliberately carries handles and post text.

The taxonomy is deliberately not binary. "Pod or operation" was the question
that produced a refuted discriminator; most coordinated clusters on this corpus
are neither, and forcing them into two boxes throws away the distinction that
actually matters to a reader.
"""

from __future__ import annotations

import json
import logging
import os
import textwrap

log = logging.getLogger("kma")

MODEL = os.getenv("ADJUDICATE_MODEL", "claude-sonnet-5")
MAX_TOKENS = 1600

CLUSTER_TYPES = (
    "engagement_pod",       # reciprocal amplification for reach, no shared claim
    "political_campaign",   # openly organised political messaging
    "influence_operation",  # covert, coordinated, deceptive about its origin
    "commercial_spam",      # promotion, giveaways, affiliate pushing
    "fandom_or_interest",   # sport, music, celebrity
    "news_amplification",   # outlets and their regular resharers
    "unclear",              # the evidence does not support a call
)

SYSTEM = """\
You are triaging clusters of social media accounts that a statistical detector \
has found to act together more than chance explains. The detection is sound; \
what it cannot tell you is WHY they co-act, which is your job.

Judge only from the evidence given. The most diagnostic field is usually what \
the accounts JOINTLY amplified - that is what the detector actually fired on. \
What individual members post separately is context, not evidence of \
coordination.

Be conservative and be willing to say unclear. A wrong "influence_operation" \
on a group of real people is a far worse error than a missed one, because this \
feeds a public dashboard. Reciprocal engagement farming is common, legal and \
not deceptive about its origin; do not call it an influence operation.

An influence operation requires evidence of DECEPTION about origin or intent - \
coordinated accounts pushing a political message while concealing that they \
are organised. Coordination alone is not enough. Neither is a political topic.\
"""

INSTRUCTIONS = f"""\
Return ONLY a JSON object, no prose around it, with these keys:

  cluster_type      one of: {", ".join(CLUSTER_TYPES)}
  kenya_relevant    true if this is about Kenyan politics or civic life
  confidence        low | medium | high
  rationale         <= 40 words, citing the specific evidence you used
  what_would_change_this   <= 25 words: the evidence that would flip your call

`kenya_relevant` is your own reading of the content. The kenya_share figure in \
the packet is a keyword proxy with known blind spots - it misses posts that \
discuss Kenya without the anchor vocabulary - so treat it as one input, not as \
the answer.\
"""


def _fmt_posts(rows: list[dict], limit: int, text_key: str, who_key: str) -> str:
    if not rows:
        return "    (none collected)"
    out = []
    for r in rows[:limit]:
        who = r.get(who_key) or "unknown"
        text = textwrap.shorten(str(r.get(text_key) or ""), 220)
        extra = f" [{r['n_members']} members]" if "n_members" in r else ""
        out.append(f"    -{extra} @{who}: {text}")
    return "\n".join(out)


def render(packet: dict, max_items: int = 6) -> str:
    """The dossier as the text a reader actually sees.

    Ordered by diagnostic value rather than by how the packet is built: what
    they jointly amplified first, because that is the co-action the detector
    fired on; the scores last, because they describe the coordination and this
    question is about its purpose."""
    p = packet
    lines = [
        f"CLUSTER {p.get('cluster_id')} - {p.get('size')} accounts acting together",
        "",
        "WHAT THEY JOINTLY AMPLIFIED (the co-action the detector found):",
        _fmt_posts(p.get("shared_objects", []), max_items, "object_text", "object_author"),
        "",
        "WHOSE CONTENT THEY PUSH:",
    ]
    targets = p.get("amplification_targets", [])
    if targets:
        for t in targets[:max_items]:
            inside = "cluster member" if t.get("target_is_member") else "outside the cluster"
            lines.append(f"    - @{t.get('target_handle')}: {t.get('acts')} acts ({inside})")
    else:
        lines.append("    (none collected)")
    lines += [
        "",
        "WHAT MEMBERS POST THEMSELVES (context, not evidence of coordination):",
        _fmt_posts(p.get("representative_posts", []), max_items, "text", "author_handle"),
    ]
    prov = p.get("provenance") or {}
    if prov:
        lines += [
            "",
            "HOW THE ACCOUNTS WERE PROVISIONED:",
            f"    accounts profiled: {prov.get('accounts_profiled')}",
            f"    account creation spread: {prov.get('creation_span_days')} days",
            f"    distinct profile images: {prov.get('distinct_profile_images')}"
            f" (out of {prov.get('accounts_profiled')} accounts)",
            f"    share with empty bio: {prov.get('share_empty_bio')}",
            f"    share with default avatar: {prov.get('share_default_image')}",
        ]
    kenya = p.get("kenya") or {}
    if kenya:
        lines += [
            "",
            f"KEYWORD KENYA SHARE: {kenya.get('kenya_share')} "
            f"over {kenya.get('posts_classified')} classified posts (proxy, see above)",
        ]
    scores = p.get("scores") or {}
    if scores:
        lines += ["", "COORDINATION SCORES (describe the coordination, not its purpose):"]
        lines += [f"    {k}: {v}" for k, v in scores.items()]
    return "\n".join(lines)


def prompt(packet: dict) -> str:
    return f"{render(packet)}\n\n{INSTRUCTIONS}"


def prompts(packets: list[dict]) -> list[dict]:
    """Prompts without calling anything - the path when no key is configured."""
    return [{"cluster_id": p.get("cluster_id"), "prompt": prompt(p)} for p in packets]


def _parse(text: str, cluster_id) -> dict:
    """Pull the JSON object out of a reply, tolerating fences and stray prose."""
    body = text.strip()
    if "```" in body:
        body = body.split("```")[1]
        body = body[4:] if body.startswith("json") else body
    start, end = body.find("{"), body.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in reply for cluster {cluster_id}")
    out = json.loads(body[start : end + 1])
    if out.get("cluster_type") not in CLUSTER_TYPES:
        # Kept rather than dropped: an unexpected label is a prompt problem the
        # caller should see, not a row to silently lose.
        out["cluster_type_raw"] = out.get("cluster_type")
        out["cluster_type"] = "unclear"
    out["cluster_id"] = cluster_id
    return out


def judge(packet: dict, client=None, model: str = MODEL) -> dict:
    """One cluster. Requires `anthropic` and ANTHROPIC_API_KEY."""
    if client is None:
        import anthropic

        client = anthropic.Anthropic()
    reply = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt(packet)}],
    )
    return _parse("".join(b.text for b in reply.content if b.type == "text"),
                  packet.get("cluster_id"))


def judge_all(packets: list[dict], client=None, model: str = MODEL) -> list[dict]:
    """Every cluster, one failure never costing the rest.

    A packet that fails adjudication comes back as `unclear` with the error
    attached, because a triage list with a hole in it silently understates how
    much there is to look at."""
    out = []
    for p in packets:
        try:
            out.append(judge(p, client=client, model=model))
        except Exception as exc:
            log.exception("adjudication failed for cluster %s", p.get("cluster_id"))
            out.append({
                "cluster_id": p.get("cluster_id"),
                "cluster_type": "unclear",
                "kenya_relevant": None,
                "confidence": "low",
                "rationale": f"adjudication failed: {exc}",
            })
    return out
