import { expect, test } from "bun:test";
import Ajv2020 from "ajv/dist/2020";
import addFormats from "ajv-formats";
import { readdirSync, readFileSync, statSync } from "fs";
import { join } from "path";
import type { Manifest, Overview, CommunityIndex } from "./types";

const here = import.meta.dir;
const schema = JSON.parse(readFileSync(join(here, "schema.json"), "utf8"));
const ajv = new Ajv2020({ strict: false, allErrors: false });
addFormats(ajv);
ajv.addSchema({ ...schema, oneOf: undefined, type: undefined });

const KIND_TO_DEF: Record<string, string> = {
  manifest: "Manifest", overview: "Overview", community_index: "CommunityIndex",
  community_detail: "CommunityDetail", graph_meta: "GraphMeta", graph: "Graph", evidence: "Evidence",
  account_shard: "AccountShard", search_accounts: "SearchAccounts", hashtag_index: "HashtagIndex",
  hashtag_shard: "HashtagShard", link_index: "LinkIndex", topic_index: "TopicIndex",
  topic_detail: "TopicDetail", time_collection: "TimeCollection", time_pipeline: "TimePipeline",
  time_coverage: "TimeCoverage", changes: "Changes",
};

function* walk(dir: string): Generator<string> {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) yield* walk(p);
    else if (p.endsWith(".json")) yield p;
  }
}

const root = join(here, "fixtures");
const files = [...walk(root)];

test("fixture bundle is non-empty", () => {
  expect(files.length).toBeGreaterThan(100);
});

test("every fixture JSON validates against schema.json", () => {
  const failures: string[] = [];
  for (const f of files) {
    const doc = JSON.parse(readFileSync(f, "utf8"));
    const def = KIND_TO_DEF[doc.kind];
    if (!def) { failures.push(`${f}: unknown kind ${doc.kind}`); continue; }
    const validate = ajv.compile({ $ref: `${schema.$id}#/$defs/${def}` });
    if (!validate(doc)) failures.push(`${f}: ${JSON.stringify(validate.errors?.[0])}`);
  }
  expect(failures).toEqual([]);
});

test("a file without stamp fails", () => {
  const doc = JSON.parse(readFileSync(join(root, "explorer/v1/runs/20261010T031500Z/overview.json"), "utf8"));
  delete doc.stamp;
  const validate = ajv.compile({ $ref: `${schema.$id}#/$defs/Overview` });
  expect(validate(doc)).toBe(false);
});

test("generated types fit the fixtures", () => {
  const m: Manifest = JSON.parse(readFileSync(join(root, "explorer/v1/manifest.json"), "utf8"));
  expect(m.latest_run).toBe(m.runs[0].run_id);
  const ov: Overview = JSON.parse(readFileSync(join(root, `explorer/v1/runs/${m.latest_run}/overview.json`), "utf8"));
  expect(ov.stamp.run_id).toBe(m.latest_run);
  const ci: CommunityIndex = JSON.parse(readFileSync(join(root, `explorer/v1/runs/${m.latest_run}/communities.json`), "utf8"));
  expect(ci.communities.filter((c) => c.kept).length).toBe(ov.surfaced.communities_kept);
});
