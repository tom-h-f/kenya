from pathlib import Path


PROMPT_PATH = Path(__file__).with_name("prompts") / "label_v4.md"

REQUIRED_CONTEXT = {
    "William Ruto": ("William Ruto", "Kasongo", "Zakayo", "Sugoi"),
    "election slogans": ("Wantam", "Tutam"),
    "institutions": ("IEBC", "NCIC"),
    "languages": ("Kiswahili", "Sheng", "code-switching"),
}

REQUIRED_SECTIONS = (
    "## Task and governing boundary",
    "## Neutral Kenya 2027 context",
    "## Actors, institutions, parties, coalitions, places, aliases, and slogans",
    "## Language and code-switching",
    "## Coded ethnic, exclusion, dehumanisation, and violence terms",
    "## Flags and consistency",
    "## Difficult examples",
    "## Output contract",
)

OUTPUT_FIELDS = (
    "post_id",
    "label",
    "flags",
    "target_group",
    "confidence",
    "rationale",
)


def read_prompt() -> str:
    return PROMPT_PATH.read_text()


def test_v4_has_required_kenyan_context() -> None:
    prompt = read_prompt()

    for terms in REQUIRED_CONTEXT.values():
        assert all(term in prompt for term in terms)


def test_v4_has_required_reference_sections() -> None:
    prompt = read_prompt()

    positions = [prompt.index(section) for section in REQUIRED_SECTIONS]
    assert positions == sorted(positions)


def test_v4_aliases_and_groupings_are_not_automatic_hate_evidence() -> None:
    prompt = read_prompt().lower()

    assert "political alias" in prompt
    assert "not a protected group" in prompt
    assert "political groupings are not protected groups" in prompt
    assert "term alone is not hate" in prompt
    assert "ethnicity, region, alias, party membership, or political support" in prompt
    assert "never automatic hate evidence" in prompt


def test_v4_requires_strict_jsonl_output_fields() -> None:
    prompt = read_prompt()

    assert "Return strict JSONL" in prompt
    schema_start = prompt.index('{"post_id":"<exact input id>"')
    schema_end = prompt.index("\n```", schema_start)
    schema = prompt[schema_start:schema_end]
    assert tuple(schema.index(f'"{field}"') for field in OUTPUT_FIELDS) == tuple(
        sorted(schema.index(f'"{field}"') for field in OUTPUT_FIELDS)
    )
    assert "one object per input post in the same order, and nothing else" in prompt
    assert "Every input ID must appear exactly once" in prompt


def test_v4_makes_hate_equivalent_to_ethnic_targeting() -> None:
    prompt = read_prompt()

    assert "`hate` if and only if `ethnic_targeting` is set" in prompt
    assert "every `hate` row must set `ethnic_targeting`" in prompt
    assert "a row that sets `ethnic_targeting` must be `hate`" in prompt
