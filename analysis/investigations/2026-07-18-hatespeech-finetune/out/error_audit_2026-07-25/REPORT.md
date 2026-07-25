# Error audit sample 2026-07-25

Live lexicon hits: **100** / joined to hatespeech.

## A. Lexicon hit ∩ model miss candidates

Full `neither` + no hate_flag: **81**

| term | n | mean p_hate | max eng |
|---|---:|---:|---:|
| 41_vs_1 | 5 | 0.004 | 555 |
| fukuza | 24 | 0.004 | 1929 |
| fumigate | 1 | 0.001 | 92 |
| kimeumana | 14 | 0.009 | 37 |
| kwekwe | 4 | 0.004 | 2253 |
| madimoni | 1 | 0.002 | 21 |
| madoadoa | 3 | 0.003 | 469 |
| mende | 1 | 0.004 | 3 |
| nyoka | 7 | 0.015 | 5 |
| uthamaki | 2 | 0.040 | 36 |
| waondoke | 1 | 0.008 | 49 |
| watajua_hawajui | 18 | 0.002 | 47 |

Stratified review sample: `error_audit_2026-07-25/A_lexicon_model_miss_candidates.csv` (n=39)

### Preview (low fp_risk, lowest p_hate)

- `41_vs_1` p_hate=0.001 eng=1: @BoboWamboiKuria Hii nayo mkijaribu 41  vs 1 will be a reality ni hayo kwa sasa
- `watajua_hawajui` p_hate=0.001 eng=5: NA Speaker Moses Wetangula has issued a 48-hour demand to Standard Group over a front-page story titled Broad-based family, calling it false
- `watajua_hawajui` p_hate=0.002 eng=47: RT @ItsNayombe: Publicly admitting how they were bribing Ol Kalou Voters, but IEBC are silent this.  Western things have shifted watajua haw
- `watajua_hawajui` p_hate=0.002 eng=46: RT @ItsNayombe: Publicly admitting how they were bribing Ol Kalou Voters, but IEBC are silent this.  Western things have shifted watajua haw
- `watajua_hawajui` p_hate=0.002 eng=44: RT @ItsNayombe: Publicly admitting how they were bribing Ol Kalou Voters, but IEBC are silent this.  Western things have shifted watajua haw
- `watajua_hawajui` p_hate=0.002 eng=42: RT @ItsNayombe: Publicly admitting how they were bribing Ol Kalou Voters, but IEBC are silent this.  Western things have shifted watajua haw
- `watajua_hawajui` p_hate=0.002 eng=42: RT @ItsNayombe: Publicly admitting how they were bribing Ol Kalou Voters, but IEBC are silent this.  Western things have shifted watajua haw
- `watajua_hawajui` p_hate=0.002 eng=42: RT @ItsNayombe: Publicly admitting how they were bribing Ol Kalou Voters, but IEBC are silent this.  Western things have shifted watajua haw
- `watajua_hawajui` p_hate=0.002 eng=42: RT @ItsNayombe: Publicly admitting how they were bribing Ol Kalou Voters, but IEBC are silent this.  Western things have shifted watajua haw
- `watajua_hawajui` p_hate=0.002 eng=31: RT @ItsNayombe: Publicly admitting how they were bribing Ol Kalou Voters, but IEBC are silent this.  Western things have shifted watajua haw
- `watajua_hawajui` p_hate=0.002 eng=31: RT @ItsNayombe: Publicly admitting how they were bribing Ol Kalou Voters, but IEBC are silent this.  Western things have shifted watajua haw
- `watajua_hawajui` p_hate=0.002 eng=31: RT @ItsNayombe: Publicly admitting how they were bribing Ol Kalou Voters, but IEBC are silent this.  Western things have shifted watajua haw

## B. hate_flag by domain bucket (engagement)

| bucket | n | mean eng | p50 eng |
|---|---:|---:|---:|
| ambiguous | 255 | 161 | 2 |
| kenya | 351 | 98 | 3 |
| offdomain | 35 | 2048 | 3 |

Review sample: `error_audit_2026-07-25/B_hateflag_engagement_buckets.csv` (n=60)

### Top off-domain hate_flag by engagement

- eng=59774 label=offensive p_hate=0.322: I have a bill that could Remigrate Zohran Mamdani, and a lot of other foreigners who hate our country.  Pass my Remigration Act! https://t.c
- eng=10837 label=hate p_hate=0.723: SOCIALISM: This will be the youngest member of Congress from Colorado. She has no understanding of economics, history, much less human natur
- eng=343 label=hate p_hate=0.835: @RepOgles We should never allow Muslims to be leaders in this country. They hate America and have an agenda to erase the freedoms and privil
- eng=191 label=hate p_hate=0.673: SOCIALISM: African-born socialists seek to destroy America and remake into an Islamic state free from Christians and Jews.  https://t.co/NZr
- eng=186 label=hate p_hate=0.804: @realMaalouf It's an insult to every American alive and dead for these people to even breath our air. Get the Muslims Out!!
- eng=138 label=hate p_hate=0.813: @realMaalouf OUR COUNTRY IS BEING TAKEN OVER BY COMMUNISTS, MUSLIMS THAT ARE WISHING DEATH TO AMERICA. THEY'RE BEING ELECTED INTO VERY IMPOR
- eng=78 label=hate p_hate=0.846: @RepOgles Who cares about passing laws anymore? What law allowed tens of millions of illegals into my country? Just arrest and deport mamdan
- eng=25 label=hate p_hate=0.472: @JasonSimmo86712 100%. Muslims do not do this in Muslim countries.  It is a sign of their domination of and disrespect for America.
- eng=24 label=hate p_hate=0.679: @realMaalouf Deport these America-hating barbarians.  Islam is incompatible with western values and liberty.  ENOUGH with this!
- eng=22 label=hate p_hate=0.738: @AntiTrumpCanada lol. Canada has been cucked by commies and filthy Indians.   Trump is the least of your problems.

### Top Kenya hate_flag by engagement

- eng=4179 label=hate p_hate=0.386: It's going to take at least three generations for other tribes to outshine Murimaa babes. Those demons are the most beautiful set of women w
- eng=1251 label=hate p_hate=0.675: 98% of Nigerians in Kenyans are swindlers, Rapists & online fraudsters. Their passports expired long time ago, they are in Kenya illegally. 
- eng=1149 label=neither p_hate=0.325: Alfred Keter told Ruto,“If you ever become president,you’ll leave Kenyans hating the Kalenjin and lumping every one of them together with yo
- eng=1132 label=hate p_hate=0.768: Murkomen. Cheruiyot and Maandago are all at war with each other  Uskie Farouq is collecting sadaka from all coz anajua after Ruto, he just h
- eng=726 label=hate p_hate=0.782: BREAKING NEWS  ITUMBI has directed police to deal mercilessly with Kikuyus who were  Caught protecting DCP votes  When did protecting votes 
- eng=561 label=hate p_hate=0.626: "WAKIKUYU ni bure kabisa, that is a negative investment." ~ Fernandes Barasa to Ruto https://t.co/zyeMO9lRET
- eng=557 label=offensive p_hate=0.323: Luyhas are waking up to the reality that Sifuna is a Kikuyu/Gachagua puppet to return power to Mt. Kenya and continue Kikuyu ethnic dominati
- eng=477 label=hate p_hate=0.763: @HonOscarSudi Kalenjin presidency is always a curse to the nation NEVER AGAIN Kasongo should be the last
- eng=466 label=hate p_hate=0.779: RT @AokoOtieno_: Murkomen. Cheruiyot and Maandago are all at war with each other  Uskie Farouq is collecting sadaka from all coz anajua afte
- eng=466 label=hate p_hate=0.779: RT @AokoOtieno_: Murkomen. Cheruiyot and Maandago are all at war with each other  Uskie Farouq is collecting sadaka from all coz anajua afte

## Review queues (deduped)

- `A_REVIEW_QUEUE.csv` — n=31 after RT-dedupe + audit tags
- `B_REVIEW_QUEUE.csv` — n=60 engagement-bucket sample

### Lexicon∩neither after RT-dedupe (n=44)

| audit_tag | n |
|---|---:|
| needs_human | 9 |
| likely_true_coded | 9 |
| needs_human_escalation_slang | 8 |
| likely_fp_sports_or_literal | 7 |
| needs_human_expulsion_context | 5 |
| needs_human_veiled_threat_or_boast | 4 |
| likely_fp_political_expulsion | 2 |

**Key:** of 81 lexicon∩neither rows, only **9 unique** look like true coded menace on a cheap heuristic; many hits are political slogans (`Fukuza Ruto`), sports literal `nyoka`, or boastive `watajua hawajui`. Do **not** treat raw lexicon hits as train labels.

### Likely-true coded misses (model neither, p_hate≈0)

- `['41_vs_1']` p_hate=0.006: 20th May, 2026  STATEMENT BY THE DCP DEPUTY PARTY LEADER SENATOR CLEOPHAS MALALA ON ETHNIC PROFILING IN KENYA BY UDA GOVERNMENT.   Dear Keny
- `['madoadoa']` p_hate=0.003: People who have been arguing that Kibaki caused 2007 should follow this guy.   We, who lived through 2007 with the skin of our teeth, should
- `['41_vs_1']` p_hate=0.002: ‘Ruto, You Joined Raila in 2007 and Mobilised 41 Against 1,’ Gachagua Claims https://t.co/iNoRApuliS
- `['mende']` p_hate=0.004: @abbiezuena Hii ndio tunataka. Kaende Kaende ...Kifo Cha Mende hio siku itakua lit 🔥.  Tackles za Sifuna na chocha kaongo. Ayam ready for de
- `['madoadoa']` p_hate=0.003: Just like Gachagua gave William Ruto Tutam by coiling wantam,Muturi's support for Kalonzo Musyoka is another way of giving William Ruto a pu
- `['41_vs_1']` p_hate=0.004: @ngoya_derrek He can recreate 41 vs 1
- `['41_vs_1']` p_hate=0.001: @BoboWamboiKuria Hii nayo mkijaribu 41  vs 1 will be a reality ni hayo kwa sasa
- `['41_vs_1']` p_hate=0.006: @gabrieloguda Remember they will not be in the Ballot come  2027 so 41 against 1 is not mathing. We will pour our votes to the the United op
- `['madoadoa']` p_hate=0.002: @YusuphCabdi @HonAdenDuale Tukimaliza kutoa wantam, next ni kutoa madoadoa Kenya. Tuwe safi kama pamba
