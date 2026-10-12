import copy
import textwrap

import pytest
import yaml

from data.questions import QuestionRegistryError, load_questions, parse_questions

VALID = {
    "id": "grass_legal",
    "gss_variable": "grass",
    "text": "Should marijuana be legal?",
    "options": ["Legal", "Not legal"],
    "option_codes": [1, 2],
    "ordinal": False,
    "category": "contested_social",
    "years": [1973, 2022],
    "contamination_note": "Widely reported.",
}


def dump(*qs):
    return yaml.safe_dump({"questions": list(qs)})


def mutate(**changes):
    q = copy.deepcopy(VALID)
    for k, v in changes.items():
        if v is DELETE:
            q.pop(k)
        else:
            q[k] = v
    return q


DELETE = object()


def bad(q, match):
    with pytest.raises(QuestionRegistryError, match=match):
        parse_questions(dump(q))


def test_valid_loads():
    (q,) = parse_questions(dump(VALID))
    assert q.id == "grass_legal"
    assert q.options == ("Legal", "Not legal")
    assert q.years == (1973, 2022)


def test_shipped_registry_loads():
    qs = load_questions()
    assert len({q.id for q in qs}) == len(qs) >= 1


@pytest.mark.parametrize("field", sorted(VALID))
def test_missing_field_rejected(field):
    with pytest.raises(QuestionRegistryError):
        parse_questions(dump(mutate(**{field: DELETE})))


def test_unknown_field_rejected():
    bad(mutate(extra="x"), "unknown field")


def test_duplicate_ids_rejected():
    with pytest.raises(QuestionRegistryError, match="duplicate question id"):
        parse_questions(dump(mutate(), mutate()))


@pytest.mark.parametrize("opts", [[], ["Only"], "Legal", ["A", "a"], ["A", " A "], ["A", ""], ["A", 3]])
def test_bad_options_rejected(opts):
    bad(mutate(options=opts), "option|string|empty")


def test_too_many_options():
    bad(mutate(options=[f"o{i}" for i in range(21)]), "exceeds")


@pytest.mark.parametrize("years", [[2022, 1973], [1900, 1990], [1990, 2100], [1990], "1990", [1990.5, 2000], [True, 2000], [1990, None]])
def test_bad_years_rejected(years):
    bad(mutate(years=years), "year")


def test_single_year_allowed():
    (q,) = parse_questions(dump(mutate(years=[2018, 2018])))
    assert q.years == (2018, 2018)


@pytest.mark.parametrize("cat", ["political", "", None, 3])
def test_bad_category_rejected(cat):
    bad(mutate(category=cat), "category")


@pytest.mark.parametrize("note", ["", "   ", None, 5])
def test_bad_contamination_note_rejected(note):
    bad(mutate(contamination_note=note), "contamination_note")


@pytest.mark.parametrize("ordinal", ["yes", 1, None])
def test_ordinal_must_be_bool(ordinal):
    bad(mutate(ordinal=ordinal), "ordinal")


@pytest.mark.parametrize("qid", ["", "Grass", "1abc", "has space", "a-b", "x" * 65, None, 7])
def test_bad_id_rejected(qid):
    bad(mutate(id=qid), "'id'")


@pytest.mark.parametrize("var", ["GRASS", "", "a b", "x" * 33, None])
def test_bad_variable_rejected(var):
    bad(mutate(gss_variable=var), "gss_variable")


@pytest.mark.parametrize("bad_char", ["\x00", "\n", "\t", "​", "‮"])
def test_control_and_invisible_chars_rejected(bad_char):
    bad(mutate(text=f"hello{bad_char}world"), "control/invisible")
    bad(mutate(options=["A", f"B{bad_char}"]), "control/invisible")


def test_non_ascii_text_allowed_and_normalised():
    (q,) = parse_questions(dump(mutate(text="Café – OK?")))
    assert "Caf" in q.text


def test_length_limits():
    bad(mutate(text="x" * 501), "exceeds")
    bad(mutate(contamination_note="x" * 1001), "exceeds")
    bad(mutate(options=["A", "b" * 101]), "exceeds")


def test_top_level_shapes_rejected():
    for text in ["", "[]", "questions: []", "questions: 5", "other: 1", "questions: [1]",
                 "questions: []\nextra: 1"]:
        with pytest.raises(QuestionRegistryError):
            parse_questions(text)


def test_duplicate_yaml_key_rejected():
    text = dump(VALID).replace("category: contested_social", "category: contested_social\n  category: factual")
    with pytest.raises(QuestionRegistryError, match="duplicate YAML key"):
        parse_questions(text)


def test_malicious_yaml_does_not_execute(tmp_path):
    marker = tmp_path / "pwned"
    text = f"questions: !!python/object/apply:os.system ['touch {marker}']"
    with pytest.raises(QuestionRegistryError):
        parse_questions(text)
    assert not marker.exists()


def test_yaml_aliases_rejected():
    text = textwrap.dedent("""
        questions:
          - &q {id: a}
          - *q
    """)
    with pytest.raises(QuestionRegistryError, match="alias"):
        parse_questions(text)


def test_malformed_yaml_rejected():
    with pytest.raises(QuestionRegistryError, match="invalid YAML"):
        parse_questions("questions: [unclosed")


def test_too_many_questions():
    qs = [mutate(id=f"q{i}") for i in range(201)]
    with pytest.raises(QuestionRegistryError, match="too many"):
        parse_questions(dump(*qs))


def test_load_missing_file(tmp_path):
    with pytest.raises(QuestionRegistryError, match="cannot read"):
        load_questions(tmp_path / "nope.yaml")


def test_load_oversized_file(tmp_path):
    p = tmp_path / "big.yaml"
    p.write_text("#" * 1_000_001)
    with pytest.raises(QuestionRegistryError, match="exceeds"):
        load_questions(p)


def test_load_non_utf8_file(tmp_path):
    p = tmp_path / "bin.yaml"
    p.write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(QuestionRegistryError, match="cannot read"):
        load_questions(p)


def test_question_is_immutable():
    (q,) = parse_questions(dump(VALID))
    with pytest.raises(Exception):
        q.id = "other"


@pytest.mark.parametrize("codes", [[1], [1, 2, 3], [1, 1], [-1, 2], [1, 1000], [1, 2.5], [True, 2], ["1", "2"], "12", None])
def test_bad_option_codes_rejected(codes):
    bad(mutate(option_codes=codes), "option_codes")


def test_option_codes_need_not_be_contiguous_or_sorted():
    (q,) = parse_questions(dump(mutate(option_codes=[4, 0])))
    assert q.option_codes == (4, 0)
