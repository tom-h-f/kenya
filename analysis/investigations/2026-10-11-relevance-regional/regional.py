"""Regional vocabulary: which neighbouring country a post is about, by keyword.

Used only to FIND candidates for labelling and to bucket the evaluation. The
labels come from the headless reader, never from these patterns.
"""

import re

REGIONS = {
    "tanzania": r"tanzania\w*|dar\s*es\s*salaam|dodoma|zanzibar|samia\s*suluhu|samiamuuaji|magufuli|chadema|"
                r"tundu\s*lissu|\blissu\b|bunge\s+la\s+tanzania|mwanza|arusha|mbeya|tanganyika|"
                r"oktoba\s*29|katibampya|act[\s-]*wazalendo|\bccm\b|bongo\b|watanzania",
    "uganda": r"uganda\w*|kampala|museveni|bobi\s*wine|kyagulanyi|muhoozi|entebbe|besigye|\bnup\b|\bm7\b|jinja|gulu\b",
    "rwanda": r"rwanda\w*|kigali|kagame|\brpf\b",
    "burundi": r"burundi\w*|bujumbura|ndayishimiye",
    "ethiopia": r"ethiopia\w*|addis\s*ababa|abiy\s*ahmed|tigray\w*|amhara|oromia|\btplf\b|eritrea\w*",
    "somalia": r"somalia\w*|mogadishu|somaliland|puntland|hassan\s*sheikh|al[\s-]*shabaab|kismayo|\bfgs\b",
    "south_sudan": r"south\s*sudan\w*|\bjuba\b|salva\s*kiir|riek\s*machar|\bspla\b",
    "sudan": r"\bsudan\w*|khartoum|\brsf\b|darfur|hemedti",
    "drc": r"\bdrc\b|congo\w*|kinshasa|goma\b|\bm23\b|tshisekedi|bukavu",
    "nigeria": r"nigeria\w*|tinubu|abuja|lagos|buhari|\bapc\b|\bpdp\b|naija|atiku|peter\s*obi|obidient|\bnnpc\b",
    "ghana": r"ghana\w*|accra|mahama|akufo|kumasi|\bndc\b|\bnpp\b",
    "south_africa": r"south\s*africa\w*|ramaphosa|johannesburg|pretoria|cape\s*town|\banc\b|\beff\b|malema|"
                    r"\bmk\s*party\b|jacob\s*zuma|soweto|durban|\bsaps\b|mzansi",
    "other_africa": r"zimbabwe\w*|zanu|mnangagwa|zambia\w*|lusaka|hichilema|malawi\w*|mozambique|"
                    r"cameroon\w*|biya|senegal\w*|ivory\s*coast|cote\s*d.ivoire|mali\b|niger\b|chad\b|"
                    r"burkina|ouattara|togo\b|madagascar|namibia\w*|botswana|angola\w*|egypt\w*|libya\w*|"
                    r"tunisia\w*|morocco|algeria\w*|sahel|ecowas|african\s*union",
}
REGION_RE = {k: re.compile(r"(?<!\w)(?:" + v + r")(?!\w)", re.I) for k, v in REGIONS.items()}

DIASPORA = re.compile(
    r"diaspora|kenyans?\s+(?:in|living\s+in|abroad)|kenyan\s+(?:in|living)|"
    r"(?:ke|kenyan)\s+(?:in\s+)?(?:usa|uk|canada|dubai|qatar|saudi|australia|germany)|"
    r"remittance|nurses?\s+in\s+(?:uk|usa)|\bkenyans\s+abroad", re.I)


def regions_hit(text: str) -> list[str]:
    return [k for k, rx in REGION_RE.items() if rx.search(text)]
