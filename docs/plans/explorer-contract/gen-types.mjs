// bun add json-schema-to-typescript@15.0.4 (anywhere), then: bun gen-types.mjs
import { compile } from "json-schema-to-typescript";
import fs from "fs";
const here=new URL(".", import.meta.url).pathname; const p=here+"schema.json";
const s=JSON.parse(fs.readFileSync(p));
const defs=s.$defs; const out=[];
for (const [name,def] of Object.entries(defs)) {
  if (name==="Files") continue;
  const ts = await compile({...def,$defs:defs,title:name}, name, {bannerComment:"",additionalProperties:false,unreachableDefinitions:false});
  out.push(ts);
}
const seen=new Set(); const lines=[];
for (const block of out.join("\n").split(/\n(?=export )/)) { const m=block.match(/^export (?:interface|type) (\w+)/); if(!m||seen.has(m[1])) continue; seen.add(m[1]); lines.push(block); }
fs.writeFileSync(here+"types.ts","// Generated from schema.json by json-schema-to-typescript. Do not edit; edit schema.json and regenerate (see README.md).\n\n"+lines.join("\n")+"\n");
console.log(seen.size,"types");
