"""Contract tests: fixture bundle, validator, language lint, mutations."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from jsonschema import Draft202012Validator

HERE = Path(__file__).resolve().parent
CONTRACT = HERE.parents[2] / "docs" / "plans" / "explorer-contract"
FIXTURES = CONTRACT / "fixtures"
sys.path.insert(0, str(CONTRACT))
sys.path.insert(0, str(HERE))

import lint_language as lint  # noqa: E402
import make_fixtures  # noqa: E402
import validate_bundle as vb  # noqa: E402

LATEST = "20261010T031500Z"
PREV = "20261009T031500Z"
ALLOWLIST = lint.load_allowlist(CONTRACT / "lint-allowlist.json")


def base(root: Path) -> Path:
    return root / "explorer" / "v1" / "runs" / LATEST


def read(p: Path):
    return json.loads(p.read_text())


def write(p: Path, doc):
    p.write_text(json.dumps(doc, separators=(",", ":")) + "\n")


def refresh_manifest(root: Path):
    """Recompute bytes and sha256 so a mutation is judged on its own merits."""
    import hashlib

    mpath = root / "explorer" / "v1" / "manifest.json"
    m = read(mpath)
    for sub, meta in m["files"].items():
        p = base(root) / sub
        if p.is_file():
            data = p.read_bytes()
            meta["bytes"], meta["sha256"] = len(data), hashlib.sha256(data).hexdigest()
    write(mpath, m)


@pytest.fixture
def bundle(tmp_path):
    dst = tmp_path / "bundle"
    shutil.copytree(FIXTURES, dst)
    return dst


def problems(root):
    return vb.validate_bundle(root)


def has(probs, text):
    return any(text in p for p in probs)


class TestFixtures:
    def test_schema_is_valid(self):
        Draft202012Validator.check_schema(vb.load_schema())

    def test_fixtures_validate(self):
        assert problems(FIXTURES) == []

    def test_fixtures_are_current(self, tmp_path):
        out = tmp_path / "regen"
        make_fixtures.main([str(out)])
        a = sorted(p.relative_to(out) for p in out.rglob("*") if p.is_file())
        b = sorted(p.relative_to(FIXTURES) for p in FIXTURES.rglob("*") if p.is_file())
        assert a == b
        for rel in a:
            assert (out / rel).read_bytes() == (FIXTURES / rel).read_bytes(), rel

    def test_every_file_kind_is_present_or_declared_unavailable(self):
        schema = vb.load_schema()
        patterns = vb.compile_file_patterns(schema)
        seen = set()
        for p in (FIXTURES / "explorer" / "v1" / "runs" / LATEST).rglob("*"):
            if p.is_file():
                key, _ = vb.classify(p.relative_to(FIXTURES).as_posix(), patterns)
                seen.add(key)
        manifest = read(FIXTURES / "explorer" / "v1" / "manifest.json")
        unavailable = {u["view"] for u in manifest["unavailable"]}
        expected = set(schema["$defs"]["Files"]["properties"]) - {"manifest", "vendor"}
        assert expected - seen == unavailable

    def test_two_runs_and_changes_populated(self):
        manifest = read(FIXTURES / "explorer" / "v1" / "manifest.json")
        assert [r["run_id"] for r in manifest["runs"]] == [LATEST, PREV]
        ch = read(base(FIXTURES) / "time" / "changes.json")
        assert ch["previous_run"] == PREV
        assert {m["status"] for m in ch["matches"]} == {"continuing", "new", "vanished", "split", "merged"}

    def test_community_shape(self):
        rows = read(base(FIXTURES) / "communities.json")["communities"]
        kept = [r for r in rows if r["kept"]]
        dropped = [r for r in rows if not r["kept"]]
        assert [r["size"] for r in kept] == [60, 30, 12]
        assert len(dropped) == 1
        assert not (base(FIXTURES) / "communities" / f"{dropped[0]['cid']}.json").exists()

    def test_every_trace_has_evidence(self):
        traces = set()
        for p in (base(FIXTURES) / "evidence").glob("c*.json"):
            for e in read(p)["edges"].values():
                traces.update(i["trace"] for i in e["items"])
        assert traces == {"co_retweet", "co_url", "hashtag_sequence", "fast_retweet", "text_similarity"}

    def test_a_co_retweet_original_is_missing_from_posts_map(self):
        missing = 0
        for p in (base(FIXTURES) / "evidence").glob("c*.json"):
            ev = read(p)
            for e in ev["edges"].values():
                missing += sum(1 for i in e["items"] if i["trace"] == "co_retweet" and i["post"] not in ev["posts"])
        assert missing >= 1

    def test_parquet_slices_about_two_thousand_rows(self):
        n = pq.ParquetFile(base(FIXTURES) / "slices" / "posts.parquet").metadata.num_rows
        assert 1500 <= n <= 2500

    def test_nothing_real(self):
        """Handles and hosts are synthetic, ids come from the reserved ranges."""
        for p in FIXTURES.rglob("*.json"):
            text = p.read_text()
            for h in re.findall(r'"(?:handle|author_handle)":"([^"]*)"', text):
                assert re.fullmatch(r"user_\d{3}|origin_\d{2}", h), (p, h)
            for url in re.findall(r'https?://([^/"]+)', text):
                assert url.endswith(".example"), (p, url)
            for i in re.findall(r'"(?:id|author_id)":"(\d+)"', text):
                assert i.startswith("90") or i.startswith("91") or i.startswith("92"), (p, i)

    def test_no_forbidden_keys_anywhere(self):
        for p in FIXTURES.rglob("*.json"):
            for path, key in vb.walk_keys(read(p)):
                assert not vb.FORBIDDEN_KEY.search(key), (p, path)

    def test_language_lint_clean(self):
        assert lint.lint_paths([FIXTURES, CONTRACT / "copy"], ALLOWLIST, include_user_content=True) == []

    def test_cli_exit_codes(self, bundle):
        ok = subprocess.run([sys.executable, str(CONTRACT / "validate_bundle.py"), str(bundle)], capture_output=True)
        assert ok.returncode == 0
        shard = next((base(bundle) / "accounts").glob("*.json"))
        doc = read(shard)
        doc.pop("stamp")
        write(shard, doc)
        bad = subprocess.run([sys.executable, str(CONTRACT / "validate_bundle.py"), str(bundle)], capture_output=True)
        assert bad.returncode == 1

    def test_types_in_sync_with_schema(self, tmp_path):
        if not (CONTRACT / "node_modules" / "json-schema-to-typescript").is_dir() or not shutil.which("bun"):
            pytest.skip("run `bun install` in the contract dir to check types.ts")
        work = tmp_path / "c"
        shutil.copytree(CONTRACT, work, ignore=shutil.ignore_patterns("fixtures", "node_modules"))
        (work / "node_modules").symlink_to(CONTRACT / "node_modules")
        subprocess.run(["bun", "gen-types.mjs"], cwd=work, check=True, capture_output=True)
        assert (work / "types.ts").read_text() == (CONTRACT / "types.ts").read_text()


class TestMutations:
    """Each mutation must fail the right check."""

    def test_flag_field_on_account_card(self, bundle):
        shard = next(p for p in (base(bundle) / "accounts").glob("*.json") if read(p)["accounts"])
        doc = read(shard)
        card = next(iter(doc["accounts"].values()))
        card["hate_flag"] = True
        write(shard, doc)
        refresh_manifest(bundle)
        probs = problems(bundle)
        assert has(probs, "forbidden field name 'hate_flag'")
        assert has(probs, "schema")

    @pytest.mark.parametrize("name", ["incitement_score", "toxicity", "deleted", "suspended", "account_gone", "banned", "prevalence"])
    def test_forbidden_field_names(self, bundle, name):
        p = base(bundle) / "communities.json"
        doc = read(p)
        doc["communities"][0][name] = 1
        write(p, doc)
        refresh_manifest(bundle)
        assert has(problems(bundle), f"forbidden field name {name!r}")

    def test_parquet_flag_column(self, bundle):
        p = base(bundle) / "slices" / "accounts.parquet"
        t = pq.read_table(p).append_column("hate_score", pa.array([0.1] * pq.ParquetFile(p).metadata.num_rows))
        pq.write_table(t, p)
        refresh_manifest(bundle)
        assert has(problems(bundle), "forbidden column 'hate_score'")

    def test_missing_stamp(self, bundle):
        p = base(bundle) / "overview.json"
        doc = read(p)
        del doc["stamp"]
        write(p, doc)
        refresh_manifest(bundle)
        assert has(problems(bundle), "overview.json: schema")

    def test_stamp_missing_params(self, bundle):
        p = base(bundle) / "links" / "index.json"
        doc = read(p)
        del doc["stamp"]["params"]
        write(p, doc)
        refresh_manifest(bundle)
        assert has(problems(bundle), "links/index.json: schema")

    def test_stamp_run_differs_from_path(self, bundle):
        p = base(bundle) / "overview.json"
        doc = read(p)
        doc["stamp"]["run_id"] = PREV
        write(p, doc)
        refresh_manifest(bundle)
        assert has(problems(bundle), "stamp.run_id")

    def test_numeric_account_id(self, bundle):
        p = base(bundle) / "search" / "accounts.json"
        doc = read(p)
        doc["id"][0] = int(doc["id"][0])
        write(p, doc)
        refresh_manifest(bundle)
        assert has(problems(bundle), "search/accounts.json: schema")

    def test_bad_date_time(self, bundle):
        p = base(bundle) / "overview.json"
        doc = read(p)
        doc["stamp"]["computed_at"] = "yesterday"
        write(p, doc)
        refresh_manifest(bundle)
        assert has(problems(bundle), "overview.json: schema")

    def test_non_neutral_label(self, bundle):
        p = base(bundle) / "communities.json"
        doc = read(p)
        doc["communities"][0]["label"] = "The Budget Crowd"
        write(p, doc)
        refresh_manifest(bundle)
        assert has(problems(bundle), "not neutral")

    def test_dropped_community_with_members_file(self, bundle):
        shutil.copy(base(bundle) / "communities" / "0.json", base(bundle) / "communities" / "3.json")
        refresh_manifest(bundle)
        assert has(problems(bundle), "which is not kept")

    def test_dropped_community_carrying_hashtags(self, bundle):
        p = base(bundle) / "communities.json"
        doc = read(p)
        doc["communities"][-1]["top_hashtags"] = [{"tag": "x", "posts": 1, "accounts": 1}]
        write(p, doc)
        refresh_manifest(bundle)
        assert has(problems(bundle), "beyond counts")

    def test_search_lists_dropped_community(self, bundle):
        p = base(bundle) / "search" / "accounts.json"
        doc = read(p)
        doc["cid"][0] = 3
        write(p, doc)
        refresh_manifest(bundle)
        assert has(problems(bundle), "dropped community")

    def test_thin_edge_in_main_graph(self, bundle):
        p = base(bundle) / "graph" / "c0.json"
        doc = read(p)
        doc["edges"]["mask"][0], doc["edges"]["shared"][0] = 1, 1
        write(p, doc)
        refresh_manifest(bundle)
        assert has(problems(bundle), "belongs in thin")

    def test_edge_index_out_of_range(self, bundle):
        p = base(bundle) / "graph" / "c1.json"
        doc = read(p)
        doc["edges"]["s"][0] = 10_000
        write(p, doc)
        refresh_manifest(bundle)
        assert has(problems(bundle), "points outside nodes")

    def test_evidence_key_outside_graph(self, bundle):
        p = base(bundle) / "evidence" / "c2.json"
        doc = read(p)
        doc["edges"]["e99999"] = {"total": 1, "items": [{"trace": "co_url", "url": "https://news.example/a/1"}]}
        write(p, doc)
        refresh_manifest(bundle)
        assert has(problems(bundle), "outside graph")

    def test_account_in_wrong_shard(self, bundle):
        shards = sorted((base(bundle) / "accounts").glob("*.json"), key=lambda p: int(p.stem))
        full = next(p for p in shards if read(p)["accounts"])
        empty = next(p for p in shards if not read(p)["accounts"])
        doc = read(full)
        moved = doc["accounts"].popitem()
        write(full, doc)
        other = read(empty)
        other["accounts"][moved[0]] = moved[1]
        write(empty, other)
        refresh_manifest(bundle)
        assert has(problems(bundle), "wrong shard")

    def test_manifest_hash_mismatch(self, bundle):
        p = base(bundle) / "links" / "index.json"
        p.write_text(p.read_text() + " ")
        assert has(problems(bundle), "bytes")

    def test_unlisted_file(self, bundle):
        shutil.copy(base(bundle) / "links" / "index.json", base(bundle) / "hashtags" / "s99.json")
        probs = problems(bundle)
        assert has(probs, "s99.json")

    def test_unknown_path(self, bundle):
        (base(bundle) / "raw.txt").write_text("x")
        assert has(problems(bundle), "matches no pattern")

    def test_unavailable_view_with_file_present(self, bundle):
        p = base(bundle) / "time" / "coverage.json"
        shutil.copy(base(bundle).parent / PREV / "time" / "coverage.json", p)
        refresh_manifest(bundle)
        mpath = bundle / "explorer" / "v1" / "manifest.json"
        m = read(mpath)
        m["files"]["time/coverage.json"] = {"bytes": p.stat().st_size, "sha256": "0" * 64}
        write(mpath, m)
        refresh_manifest(bundle)
        doc = read(p)
        doc["stamp"]["run_id"] = LATEST
        write(p, doc)
        refresh_manifest(bundle)
        assert has(problems(bundle), "marked unavailable")

    def test_unavailable_without_reason(self, bundle):
        mpath = bundle / "explorer" / "v1" / "manifest.json"
        m = read(mpath)
        m["unavailable"][0]["reason"] = " "
        write(mpath, m)
        assert has(problems(bundle), "empty reason")

    def test_denylisted_id_is_found(self, bundle):
        deny = {"9000000000000000001"}
        assert any("denylisted" in p for p in vb.validate_bundle(bundle, deny))

    def test_unknown_run_directory(self, bundle):
        shutil.copytree(base(bundle), base(bundle).parent / "20250101T000000Z")
        assert has(problems(bundle), "not listed in the manifest")

    def test_flags_only_aggregate(self, bundle):
        p = base(bundle) / "topics" / "0.json"
        doc = read(p)
        doc["flags"]["by_account"] = {"9000000000000000001": True}
        write(p, doc)
        refresh_manifest(bundle)
        assert has(problems(bundle), "topics/0.json: schema")

    def test_topic_flagged_exceeds_scored(self, bundle):
        p = base(bundle) / "topics" / "0.json"
        doc = read(p)
        doc["flags"]["flagged_posts"] = doc["flags"]["scored_posts"] + 1
        write(p, doc)
        refresh_manifest(bundle)
        assert has(problems(bundle), "exceed scored_posts")


class TestLanguageLint:
    @pytest.mark.parametrize("word", [
        "bot", "bots", "Bot", "botnet", "troll", "trolls", "troll farm", "inauthentic", "inauthenticity",
        "fake", "network of fake accounts", "sockpuppet", "sock puppets", "astroturf", "astroturfing",
        "coordinated", "co-ordinated", "paid accounts", "operatives", "spam", "disinformation network",
    ])
    def test_forbidden_word_fails(self, tmp_path, word):
        f = tmp_path / "copy.json"
        write(f, {"label": f"These accounts are {word} online"})
        assert lint.lint_paths([f]) != []

    @pytest.mark.parametrize("phrase", [
        "shares retweets with", "posted near-identical text to", "used the same hashtag sequence as",
        "shared the same link as", "co-active with", "abbot", "both", "robotics", "trolley", "coordinates",
    ])
    def test_allowed_wording_passes(self, tmp_path, phrase):
        f = tmp_path / "copy.json"
        write(f, {"label": f"User A {phrase} user B"})
        assert lint.lint_paths([f]) == []

    def test_keys_are_linted(self, tmp_path):
        f = tmp_path / "x.json"
        write(f, {"bot_score": 1})
        assert lint.lint_paths([f]) != []

    def test_user_content_skipped_unless_asked(self, tmp_path):
        f = tmp_path / "x.json"
        write(f, {"text": "my bot is broken", "bio": "troll", "handle": "bot_fan", "label": "Community 01"})
        assert lint.lint_paths([f]) == []
        assert lint.lint_paths([f], include_user_content=True) != []

    def test_tsx_literal_fails_line_by_line(self, tmp_path):
        f = tmp_path / "Card.tsx"
        f.write_text('export const A = () => <p>Inauthentic accounts</p>;\nconst ok = "shares retweets with";\n')
        out = lint.lint_paths([f])
        assert len(out) == 1 and "line 1" in out[0]

    def test_node_modules_are_skipped(self, tmp_path):
        d = tmp_path / "node_modules" / "pkg"
        d.mkdir(parents=True)
        (d / "a.md").write_text("bot")
        assert lint.lint_paths([tmp_path]) == []

    def test_allowlist_applies_only_to_matching_file(self, tmp_path):
        al = [{"file": "*methodology*", "phrase": "not evidence that any account is inauthentic", "reason": "negative result"}]
        meth = tmp_path / "methodology" / "page.md"
        meth.parent.mkdir()
        meth.write_text("Co-activity is not evidence that any account is inauthentic.\n")
        other = tmp_path / "accounts" / "page.md"
        other.parent.mkdir()
        other.write_text("Co-activity is not evidence that any account is inauthentic.\n")
        assert lint.lint_paths([tmp_path], al) != []
        assert {p.split(":")[0] for p in lint.lint_paths([tmp_path], al)} == {"accounts/page.md"}

    def test_allowlist_does_not_cover_other_words_on_the_line(self, tmp_path):
        al = [{"file": "*", "phrase": "not evidence that any account is inauthentic", "reason": "negative result"}]
        f = tmp_path / "m.md"
        f.write_text("This is not evidence that any account is inauthentic, but user_001 is a bot.\n")
        out = lint.lint_paths([f], al)
        assert len(out) == 1 and "'bot'" in out[0]

    def test_allowlist_entry_needs_reason(self, tmp_path):
        f = tmp_path / "al.json"
        write(f, [{"file": "*", "phrase": "bot"}])
        with pytest.raises(SystemExit):
            lint.load_allowlist(f)

    def test_site_copy_needs_the_allowlist(self):
        assert lint.lint_paths([CONTRACT / "copy"]) != []
        assert lint.lint_paths([CONTRACT / "copy"], ALLOWLIST) == []

    def test_mutated_fixture_verdict_word_fails(self, bundle):
        p = base(bundle) / "graph" / "all.json"
        doc = read(p)
        doc["edge_rule"] = "Edges between accounts that look like a bot network"
        write(p, doc)
        assert lint.lint_paths([bundle]) != []

    def test_mutated_fixture_post_text_fails_with_user_content(self, bundle):
        p = base(bundle) / "evidence" / "c0.json"
        doc = read(p)
        next(iter(doc["posts"].values()))["text"] = "a troll posted this"
        write(p, doc)
        assert lint.lint_paths([bundle]) == []
        assert lint.lint_paths([bundle], include_user_content=True) != []

    def test_cli_exit_code(self, tmp_path):
        f = tmp_path / "x.md"
        f.write_text("fake\n")
        r = subprocess.run([sys.executable, str(CONTRACT / "lint_language.py"), str(f)], capture_output=True)
        assert r.returncode == 1
        f.write_text("fine\n")
        r = subprocess.run([sys.executable, str(CONTRACT / "lint_language.py"), str(f)], capture_output=True)
        assert r.returncode == 0


class TestCorrectionsConfig:
    def test_placeholder_is_marked_and_commitment_is_72h(self):
        c = read(CONTRACT / "copy" / "site-copy.json")["corrections"]
        assert c["removal_hours"] == 72
        assert c["contact_is_placeholder"] is True
        assert "CONFIGURE" in c["contact"] and "@" not in c["contact"]
