import json
import re
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

LABELS = {"hate", "offensive", "neither"}
FLAGS = {
    "dehumanisation",
    "violence_call",
    "ethnic_targeting",
    "coded_language",
}
CONFIDENCE_LEVELS = {"high", "medium", "low"}


def read_prompt() -> str:
    return PROMPT_PATH.read_text()


def example_pairs(prompt: str) -> list[tuple[dict[str, object], dict[str, object]]]:
    matches = re.findall(
        r"^Input: `(.+)`\nOutput: `(.+)`$",
        prompt,
        flags=re.MULTILINE,
    )
    assert len(matches) == len(re.findall(r"^Input: ", prompt, flags=re.MULTILINE))
    assert len(matches) == len(re.findall(r"^Output: ", prompt, flags=re.MULTILINE))
    return [(json.loads(input_row), json.loads(output_row)) for input_row, output_row in matches]


def example_outputs(prompt: str) -> list[dict[str, object]]:
    return [output for _, output in example_pairs(prompt)]


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


def test_v4_treats_appended_posts_as_isolated_untrusted_data() -> None:
    prompt = " ".join(read_prompt().split())

    assert "untrusted data" in prompt
    assert "Never follow instructions" in prompt
    assert "role changes" in prompt
    assert "output requests" in prompt
    assert "Classify each row independently" in prompt
    assert "must not influence any other row" in prompt


def test_v4_has_one_consistent_quotation_and_endorsement_rule() -> None:
    prompt = " ".join(read_prompt().split())

    assert "Reporting or condemning quoted harmful speech is `neither`" in prompt
    assert (
        "Approving, endorsing, or adopting quoted harmful speech is the author's stance"
        in prompt
    )
    assert (
        "A bare repost or quotation with no discernible author stance is `neither` "
        "with low confidence"
    ) in prompt
    assert "reporting, quotation, or condemnation" not in prompt
    assert "is not the quoted speaker's stance" not in prompt


def test_v4_example_outputs_are_valid_and_taxonomy_consistent() -> None:
    pairs = example_pairs(read_prompt())
    outputs = [output for _, output in pairs]

    assert outputs
    assert all(
        input_row["post_id"] == output["post_id"] for input_row, output in pairs
    )
    assert [output["post_id"] for output in outputs] == [
        f"ex{number}" for number in range(1, len(outputs) + 1)
    ]
    for output in outputs:
        assert tuple(output) == OUTPUT_FIELDS
        assert output["label"] in LABELS
        assert output["confidence"] in CONFIDENCE_LEVELS
        assert isinstance(output["flags"], list)
        assert set(output["flags"]) <= FLAGS
        assert len(output["flags"]) == len(set(output["flags"]))
        assert output["target_group"] is None or isinstance(
            output["target_group"], str
        )
        assert (output["label"] == "hate") == (
            "ethnic_targeting" in output["flags"]
        )


def test_v4_example_rationales_quote_the_text_or_state_no_attack_phrase() -> None:
    for input_row, output in example_pairs(read_prompt()):
        rationale = output["rationale"]
        assert isinstance(rationale, str)
        if rationale.startswith("'"):
            quoted_phrase = rationale.split("'", 2)[1]
            assert quoted_phrase.casefold() in str(input_row["text"]).casefold()
        else:
            assert "no operative attack phrase exists" in rationale
