# Round-3 hard-miss report (Opus-validated labels)

Generated 2026-07-24. Labels treated as human-validated.

## Challenge consensus errors (all 3 seeds)
- n=27 (of ~33-39 per seed)
- hate misses among them: 4
- none of these IDs are in train (challenge-held)

### Hate misses (model under-calls)
- `2073655294744559987` true=hate pred=offensive conf=medium
  - flags=<Flags(allows_duplicate_labels=True)>
  - rationale: 'the Abagusii hv become the epitome of state sponsored goons' collectively brands an ethnic community as goons.
  - text: @FredMatiangi_1 it is clear u can't run the nation the Abagusii hv become the epitome of state sponsored goons within the united opposition call ua house to order.
- `2074275953778266310` true=hate pred=neither conf=medium
  - flags=<Flags(allows_duplicate_labels=True)>
  - rationale: 'The doors to takeover stay open' after 'the DC attack' frames Muslims as an invasive threat, stirring contempt against a religious group.
  - text: Nashville Rep. Zulfat Suara slams scrutiny of Muslim loyalty and calls for rejection of Sen. Tommy Tuberville’s bill to ban Islamic immigration She labels it “Islamophobic” and says we must not “punis
- `2074538379568366006` true=hate pred=offensive conf=low
  - flags=<Flags(allows_duplicate_labels=True)>
  - rationale: 'shida ya Hawa watu wetu wa RiverLake... power kidogo, they grow egos' applies a collective ego stereotype to a regional-ethnic community.
  - text: @KenyanSays Kiburi imemea. Hii ndio shida ya Hawa watu wetu wa RiverLake. 😂😂😂😂 power kidogo, they grow egos
- `2075604774406521087` true=hate pred=neither conf=medium
  - flags=<Flags(allows_duplicate_labels=True)>
  - rationale: 'wanawake sio watu' (women are not people) dehumanises women collectively, a protected group by gender.
  - text: First frame ni bwana analia bibi yake ameshikwa. Second ni bibi anakana l bwanake. Majamaa wanawake sio watu

## Label cleanup decision
No automatic label flips. All challenge misses sit outside train;
cleanup = keep Opus labels, add more like them via hard-flag oversample
(`train2026_opus-v4-hardflags.parquet`, n=82).

## 14 known coded posts
Opus labels on all 14; pack at `out/spotcheck_coded14_opus.parquet`.
Score on Modal next.