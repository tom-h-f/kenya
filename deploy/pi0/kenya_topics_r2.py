#!/usr/bin/env python3
"""R2 helper for the topics job, run on pi0 (stdlib only, SigV4 by hand).

    kenya_topics_r2.py manifest RUN_ID [--chunks N] [--hours H]   JSON on stdout
    kenya_topics_r2.py verify RUN_ID                              exit 0 if the run's marker is complete and its chunks exist

`manifest` lists the embeddings/ and topics/ objects and signs short-lived GET
urls for them, plus PUT urls for this run's chunk and marker keys, so mike-wsl
needs no R2 credentials. Credentials come from the collector's .env on pi0 and
never reach stdout.
"""
import datetime
import hashlib
import hmac
import json
import os
import sys
import urllib.parse
import urllib.request
from xml.etree import ElementTree as ET

ENV_FILE = os.environ.get("KMA_ENV_FILE", "/home/tom/kenya-monitor-2027/.env")
MODEL = "paraphrase-multilingual-mpnet-base-v2"
PLATFORM = "x"
NAMES = ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ACCOUNT_ID", "R2_BUCKET")


def load_env():
    out = {}
    for line in open(ENV_FILE):
        k, _, v = line.strip().partition("=")
        if k in NAMES:
            out[k] = v.strip().strip('"').strip("'")
    missing = [n for n in NAMES[:3] if not out.get(n)]
    if missing:
        sys.exit(f"missing {missing} in {ENV_FILE}")
    out.setdefault("R2_BUCKET", "kenya-monitor-2027")
    return out


E = load_env()
HOST = f"{E['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
BUCKET = E["R2_BUCKET"]
Q = lambda s: urllib.parse.quote(s, safe="-_.~")  # noqa: E731


def _hmac(key, msg):
    return hmac.new(key, msg.encode(), hashlib.sha256).digest()


def _signing_key(day):
    k = _hmac(("AWS4" + E["R2_SECRET_ACCESS_KEY"]).encode(), day)
    return _hmac(_hmac(_hmac(k, "auto"), "s3"), "aws4_request")


def _canonical_query(q):
    return "&".join(f"{Q(k)}={Q(v)}" for k, v in sorted(q.items()))


def _sign(method, path, query, amz, headers_extra=None):
    day = amz[:8]
    scope = f"{day}/auto/s3/aws4_request"
    headers = {"host": HOST, "x-amz-content-sha256": "UNSIGNED-PAYLOAD", "x-amz-date": amz}
    signed = ";".join(sorted(headers))
    canon_headers = "".join(f"{k}:{headers[k]}\n" for k in sorted(headers))
    cr = "\n".join([method, urllib.parse.quote(path, safe="/-_.~"), _canonical_query(query),
                    canon_headers, signed, "UNSIGNED-PAYLOAD"])
    sts = "\n".join(["AWS4-HMAC-SHA256", amz, scope, hashlib.sha256(cr.encode()).hexdigest()])
    sig = hmac.new(_signing_key(day), sts.encode(), hashlib.sha256).hexdigest()
    headers["authorization"] = (
        f"AWS4-HMAC-SHA256 Credential={E['R2_ACCESS_KEY_ID']}/{scope}, "
        f"SignedHeaders={signed}, Signature={sig}"
    )
    return headers


def _now():
    return datetime.datetime.now(datetime.timezone.utc)


def presign(method, key, seconds):
    amz = _now().strftime("%Y%m%dT%H%M%SZ")
    scope = f"{amz[:8]}/auto/s3/aws4_request"
    q = {"X-Amz-Algorithm": "AWS4-HMAC-SHA256", "X-Amz-Credential": f"{E['R2_ACCESS_KEY_ID']}/{scope}",
         "X-Amz-Date": amz, "X-Amz-Expires": str(seconds), "X-Amz-SignedHeaders": "host"}
    path = f"/{BUCKET}/{key}"
    cr = "\n".join([method, urllib.parse.quote(path, safe="/-_.~"), _canonical_query(q),
                    f"host:{HOST}\n", "host", "UNSIGNED-PAYLOAD"])
    sts = "\n".join(["AWS4-HMAC-SHA256", amz, scope, hashlib.sha256(cr.encode()).hexdigest()])
    sig = hmac.new(_signing_key(amz[:8]), sts.encode(), hashlib.sha256).hexdigest()
    return f"https://{HOST}{urllib.parse.quote(path, safe='/-_.~')}?{_canonical_query(q)}&X-Amz-Signature={sig}"


def list_prefix(prefix):
    out, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if token:
            q["continuation-token"] = token
        h = _sign("GET", f"/{BUCKET}", q, _now().strftime("%Y%m%dT%H%M%SZ"))
        req = urllib.request.Request(
            f"https://{HOST}/{BUCKET}?{_canonical_query(q)}",
            headers={k: v for k, v in h.items() if k != "host"},
        )
        root = ET.fromstring(urllib.request.urlopen(req, timeout=60).read())
        ns = {"s": root.tag.split("}")[0].strip("{")}
        for c in root.findall("s:Contents", ns):
            out.append((c.find("s:Key", ns).text, int(c.find("s:Size", ns).text)))
        if root.find("s:IsTruncated", ns).text == "true":
            token = root.find("s:NextContinuationToken", ns).text
        else:
            return out


def emb_prefix():
    return f"embeddings/platform={PLATFORM}/model={MODEL}/"


def top_prefix():
    return f"topics/platform={PLATFORM}/model={MODEL}/"


def run_keys(run_id, chunks):
    day = f"{run_id[0:4]}-{run_id[4:6]}-{run_id[6:8]}"
    keys = [f"{top_prefix()}dt={day}/run={run_id}-c{i:04d}.parquet" for i in range(chunks)]
    keys.append(f"topics/_assign/platform={PLATFORM}/model={MODEL}/run={run_id}.json")
    return keys


def manifest(run_id, chunks, hours):
    seconds = int(hours * 3600)
    objects = {}
    for prefix in (emb_prefix(), top_prefix()):
        for key, size in list_prefix(prefix):
            objects[key] = {"size": size, "url": presign("GET", key, seconds)}
    puts = {k: {"put": presign("PUT", k, seconds), "get": presign("GET", k, seconds)}
            for k in run_keys(run_id, chunks)}
    return {"version": 1, "run_id": run_id, "objects": objects, "puts": puts,
            "expires_at": (_now() + datetime.timedelta(seconds=seconds)).isoformat()}


def verify(run_id):
    """The marker must say complete and every chunk it names must exist at the
    recorded size in the bucket listing."""
    mkey = f"topics/_assign/platform={PLATFORM}/model={MODEL}/run={run_id}.json"
    marker = json.load(urllib.request.urlopen(presign("GET", mkey, 300), timeout=60))
    listed = dict(list_prefix(top_prefix()))
    problems = []
    if marker.get("status") != "complete":
        problems.append(f"marker status {marker.get('status')!r}")
    if marker.get("readback_rows") != marker.get("assigned") or marker.get("readback_distinct") != marker.get("assigned"):
        problems.append("readback counts do not match assigned")
    for c in marker.get("chunks", []):
        if listed.get(c["key"]) != c["bytes"]:
            problems.append(f"{c['key']}: listing {listed.get(c['key'])} vs marker {c['bytes']}")
    result = {"ok": not problems, "problems": problems, "assigned": marker.get("assigned"),
              "pending_at_start": marker.get("pending_at_start"), "chunks": len(marker.get("chunks", [])),
              "peak_vram_gib": marker.get("peak_vram_gib"), "elapsed_s": marker.get("elapsed_s")}
    print(json.dumps(result))
    return 0 if result["ok"] else 1


def main():
    cmd, run_id = sys.argv[1], sys.argv[2]
    args = dict(zip(sys.argv[3::2], sys.argv[4::2]))
    if cmd == "manifest":
        json.dump(manifest(run_id, int(args.get("--chunks", 120)), float(args.get("--hours", 6))), sys.stdout)
        return 0
    if cmd == "verify":
        return verify(run_id)
    sys.exit(f"unknown command {cmd}")


if __name__ == "__main__":
    sys.exit(main())
