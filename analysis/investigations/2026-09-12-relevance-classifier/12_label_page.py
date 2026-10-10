"""Build the blind labelling page: 100 posts, English translations pre-done.

    cd analysis && uv run python investigations/2026-09-12-relevance-classifier/12_label_page.py

Same job as the B2 page: label posts `kenya` / `offdomain` / `unclear` without
seeing what any model thinks. Most of this corpus is Swahili and Sheng, so each
post carries an English translation done up front - headless claude-opus-5, in
chunks, cached in `out/translations.json` so a rerun costs nothing.

Writes `out/label.html`, a single file with the posts embedded. It autosaves to
the browser's local storage and downloads `human_check_filled.csv`, which
`10_human_check.py --score` reads directly.

Translation only. The page shows nothing about the model's call, and the
`p_kenya` column stays in `human_check_key.csv`, which the page never reads.
"""

import argparse
import html
import json
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
CHUNK = 25
MODEL = "claude-opus-5"

SYSTEM = """You translate social media posts into English. Most are Kenyan English, Swahili or \
Sheng, some are other languages, some are already English.

Translate the meaning, not word for word, and keep it short. Leave @handles, #hashtags, URLs and \
names as they are. If a post is already English, return it unchanged. Never comment on the \
content, never say what it is about, never add notes - only the translation."""


def translate(chunk: pd.DataFrame, workdir: str) -> list[str]:
    body = "\n\n".join(f"POST {k}\n{str(r.text).strip()}" for k, r in enumerate(chunk.itertuples()))
    prompt = (f"{body}\n\nReturn ONLY a JSON array, one object per post, in the order given: "
              '{"post": <n>, "english": "<the translation>"}')
    cmd = ["claude", "-p", "--model", MODEL, "--system-prompt", SYSTEM, "--tools", "",
           "--strict-mcp-config", "--mcp-config", '{"mcpServers": {}}',
           "--settings", '{"disableAllHooks": true}', "--setting-sources", "local",
           "--no-session-persistence", "--output-format", "json"]
    proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=workdir, timeout=900)
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip()[-300:])
    out = json.loads(proc.stdout)
    result = next(m for m in (out if isinstance(out, list) else [out]) if m.get("type") == "result")["result"]
    rows = json.loads(result[result.find("[") : result.rfind("]") + 1])
    by_index = {int(r["post"]): str(r["english"]) for r in rows}
    if len(by_index) != len(chunk):
        raise RuntimeError(f"{len(by_index)} translations for {len(chunk)} posts")
    return [by_index[k] for k in range(len(chunk))]


def translations(sheet: pd.DataFrame, cache: Path) -> dict[str, str]:
    done = json.loads(cache.read_text()) if cache.exists() else {}
    todo = sheet[~sheet["post_id"].astype(str).isin(done)]
    if not todo.empty:
        with tempfile.TemporaryDirectory() as workdir:
            for start in range(0, len(todo), CHUNK):
                part = todo.iloc[start : start + CHUNK]
                for post_id, english in zip(part["post_id"].astype(str), translate(part, workdir)):
                    done[post_id] = english
                cache.write_text(json.dumps(done, indent=1))
                print(f"  translated {min(start + CHUNK, len(todo))}/{len(todo)}", flush=True)
    return done


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kenya relevance: blind pass</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
  :root {
    --ink: #1b1a17; --dim: #6b6760; --line: #ddd8cf; --bg: #f7f5f1; --card: #fffefb;
    --kenya: #1f6f4a; --off: #8a4b2a; --unclear: #5a5f6b; --accent: #1f6f4a;
  }
  @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
    --ink: #eceae5; --dim: #a09a90; --line: #37342f; --bg: #16151300; --bg: #161513; --card: #1f1e1b;
    --kenya: #6fc79a; --off: #d59366; --unclear: #9aa1ad; --accent: #6fc79a;
  } }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink);
    font-family: "IBM Plex Sans", system-ui, sans-serif; padding: 24px 16px 96px; }
  main { max-width: 720px; margin: 0 auto; }
  h1 { font-family: "Source Serif 4", Georgia, serif; font-size: 1.5rem; margin: 0 0 4px;
    text-wrap: balance; }
  .sub { color: var(--dim); font-size: .875rem; margin: 0 0 20px; }
  .bar { height: 4px; background: var(--line); border-radius: 2px; overflow: hidden; margin-bottom: 20px; }
  .bar > i { display: block; height: 100%; background: var(--accent); width: 0; transition: width .2s; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
    padding: 20px; margin-bottom: 16px; }
  .meta { display: flex; justify-content: space-between; color: var(--dim);
    font-size: .75rem; letter-spacing: .06em; text-transform: uppercase; margin-bottom: 12px; }
  .post { font-family: "Source Serif 4", Georgia, serif; font-size: 1.125rem; line-height: 1.5;
    white-space: pre-wrap; overflow-wrap: anywhere; }
  .en { margin-top: 14px; padding-top: 14px; border-top: 1px dashed var(--line);
    color: var(--dim); font-size: .95rem; line-height: 1.5; white-space: pre-wrap; overflow-wrap: anywhere; }
  .en b { color: var(--dim); font-weight: 600; font-size: .7rem; letter-spacing: .08em;
    text-transform: uppercase; display: block; margin-bottom: 4px; }
  .row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 18px; }
  button { font: inherit; font-weight: 500; padding: 10px 16px; border-radius: 8px;
    border: 1px solid var(--line); background: transparent; color: var(--ink); cursor: pointer; }
  button:hover { border-color: var(--ink); }
  button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  button kbd { color: var(--dim); font: inherit; font-size: .8em; margin-left: 6px; }
  .k { border-color: var(--kenya); color: var(--kenya); }
  .o { border-color: var(--off); color: var(--off); }
  .u { border-color: var(--unclear); color: var(--unclear); }
  .done { border-color: var(--accent); background: color-mix(in srgb, var(--accent) 14%, transparent); }
  footer { position: fixed; left: 0; right: 0; bottom: 0; background: var(--card);
    border-top: 1px solid var(--line); padding: 10px 16px; }
  footer div { max-width: 720px; margin: 0 auto; display: flex; gap: 12px; align-items: center;
    justify-content: space-between; font-size: .875rem; color: var(--dim); }
  .ghost { background: transparent; }
  @media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style></head>
<body><main>
  <h1>Kenya relevance: blind pass</h1>
  <p class="sub">Is this post about Kenyan politics, society or the election? Includes posts that
  never say a Kenyan word but are unmistakably about it in context. <b>offdomain</b> is somewhere
  else, <b>unclear</b> is when you cannot tell without more context. Nothing here shows what any
  model said. Keys: 1 Kenya, 2 off-domain, 3 unclear.</p>
  <div class="bar"><i id="fill"></i></div>
  <div id="list"></div>
</main>
<footer><div>
  <span id="count">0 of 0 labelled</span>
  <span>
    <button class="ghost" onclick="download()">Download CSV</button>
    <button class="ghost" onclick="copyJson()">Copy JSON</button>
  </span>
</div></footer>
<script>
const POSTS = __DATA__;
const KEY = "kenya-relevance-blind-pass";
const labels = JSON.parse(localStorage.getItem(KEY) || "{}");
const list = document.getElementById("list");

function render() {
  list.innerHTML = "";
  POSTS.forEach((p, i) => {
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `<div class="meta"><span>${i + 1} of ${POSTS.length}</span><span>${labels[p.id] || ""}</span></div>
      <div class="post"></div>
      <div class="en"><b>English</b><span></span></div>
      <div class="row">
        <button class="k" data-v="kenya">Kenya<kbd>1</kbd></button>
        <button class="o" data-v="offdomain">Off-domain<kbd>2</kbd></button>
        <button class="u" data-v="unclear">Unclear<kbd>3</kbd></button>
      </div>`;
    card.querySelector(".post").textContent = p.text;
    card.querySelector(".en span").textContent = p.en;
    card.querySelectorAll("button").forEach(b => {
      if (labels[p.id] === b.dataset.v) b.classList.add("done");
      b.onclick = () => { labels[p.id] = b.dataset.v; save(); render(); };
    });
    list.appendChild(card);
  });
  const n = POSTS.filter(p => labels[p.id]).length;
  document.getElementById("count").textContent = `${n} of ${POSTS.length} labelled`;
  document.getElementById("fill").style.width = (100 * n / POSTS.length) + "%";
}
function save() { localStorage.setItem(KEY, JSON.stringify(labels)); }
function rows() {
  return "post_id,label\\n" + POSTS.filter(p => labels[p.id])
    .map(p => `${p.id},${labels[p.id]}`).join("\\n") + "\\n";
}
function download() {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([rows()], {type: "text/csv"}));
  a.download = "human_check_filled.csv";
  a.click();
}
function copyJson() { navigator.clipboard.writeText(JSON.stringify(labels, null, 1)); }
document.addEventListener("keydown", e => {
  const map = {"1": "kenya", "2": "offdomain", "3": "unclear"};
  if (!map[e.key]) return;
  const next = POSTS.find(p => !labels[p.id]);
  if (!next) return;
  labels[next.id] = map[e.key];
  save(); render();
  const card = list.children[POSTS.indexOf(next) + 1];
  if (card) card.scrollIntoView({behavior: "smooth", block: "center"});
});
render();
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sheet", type=Path, default=HERE / "out" / "human_check_sheet.csv")
    args = ap.parse_args()

    sheet = pd.read_csv(args.sheet)
    sheet["post_id"] = sheet["post_id"].astype(str)
    english = translations(sheet, HERE / "out" / "translations.json")

    data = [{"id": r.post_id, "text": str(r.text), "en": english.get(r.post_id, "")}
            for r in sheet.itertuples()]
    page = HERE / "out" / "label.html"
    page.write_text(PAGE.replace("__DATA__", json.dumps(data, ensure_ascii=False)))
    print(f"{len(data)} posts -> {page}")
    print(f"  open {page}")


if __name__ == "__main__":
    main()
