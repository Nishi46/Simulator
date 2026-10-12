"""Question registry: loading and strict validation of config/questions.yaml.

Nothing downstream should see a question that did not pass `load_questions`.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from simulator.config import ROOT

DEFAULT_PATH = ROOT / "config" / "questions.yaml"

CATEGORIES = ("contested_social", "factual", "behavioral")
YEAR_MIN, YEAR_MAX = 1972, 2030  # GSS began in 1972

MAX_FILE_BYTES = 1_000_000
MAX_QUESTIONS = 200
MAX_OPTIONS = 20
MAX_TEXT = 500
MAX_LABEL = 100
MAX_NOTE = 1000

_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_VAR_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_FIELDS = {
    "id", "gss_variable", "text", "options", "ordinal",
    "category", "years", "contamination_note",
}


class QuestionRegistryError(ValueError):
    """Raised for any invalid question registry content."""


@dataclass(frozen=True)
class Question:
    id: str
    gss_variable: str
    text: str
    options: tuple[str, ...]
    ordinal: bool
    category: str
    years: tuple[int, int]
    contamination_note: str


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys."""


def _construct_mapping(loader: _StrictLoader, node: yaml.MappingNode, deep: bool = False):
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in seen:
            raise QuestionRegistryError(f"duplicate YAML key: {key!r}")
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def _parse_yaml(text: str) -> Any:
    try:
        # Aliases can expand a tiny file into a huge structure; not needed here.
        for event in yaml.parse(text, Loader=yaml.SafeLoader):
            if isinstance(event, yaml.AliasEvent):
                raise QuestionRegistryError("YAML aliases/anchors are not allowed")
        return yaml.load(text, Loader=_StrictLoader)  # noqa: S506 (SafeLoader subclass)
    except yaml.YAMLError as e:
        raise QuestionRegistryError(f"invalid YAML: {e}") from e


def _check_str(value: Any, field: str, max_len: int, qid: str) -> str:
    if not isinstance(value, str):
        raise QuestionRegistryError(f"{qid}: '{field}' must be a string")
    if not value.strip():
        raise QuestionRegistryError(f"{qid}: '{field}' must not be empty")
    if len(value) > max_len:
        raise QuestionRegistryError(f"{qid}: '{field}' exceeds {max_len} characters")
    for ch in value:
        # Cc = control chars (incl. newline/NUL), Cf = invisible/bidi format chars
        if unicodedata.category(ch) in ("Cc", "Cf"):
            raise QuestionRegistryError(
                f"{qid}: '{field}' contains a control/invisible character (U+{ord(ch):04X})"
            )
    return unicodedata.normalize("NFC", value).strip()


def _check_int(value: Any, field: str, qid: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise QuestionRegistryError(f"{qid}: '{field}' must be an integer")
    return value


def _validate_question(raw: Any, index: int) -> Question:
    label = f"question[{index}]"
    if not isinstance(raw, dict):
        raise QuestionRegistryError(f"{label}: must be a mapping")
    qid = raw.get("id")
    if not isinstance(qid, str) or not _ID_RE.match(qid):
        raise QuestionRegistryError(f"{label}: 'id' must match {_ID_RE.pattern}")
    qid_label = qid

    missing = _FIELDS - raw.keys()
    if missing:
        raise QuestionRegistryError(f"{qid_label}: missing field(s): {sorted(missing)}")
    extra = raw.keys() - _FIELDS
    if extra:
        raise QuestionRegistryError(f"{qid_label}: unknown field(s): {sorted(map(str, extra))}")

    var = raw["gss_variable"]
    if not isinstance(var, str) or not _VAR_RE.match(var):
        raise QuestionRegistryError(
            f"{qid_label}: 'gss_variable' must be lowercase and match {_VAR_RE.pattern}"
        )

    text = _check_str(raw["text"], "text", MAX_TEXT, qid_label)

    opts_raw = raw["options"]
    if not isinstance(opts_raw, list):
        raise QuestionRegistryError(f"{qid_label}: 'options' must be a list")
    if len(opts_raw) < 2:
        raise QuestionRegistryError(f"{qid_label}: 'options' needs at least 2 entries")
    if len(opts_raw) > MAX_OPTIONS:
        raise QuestionRegistryError(f"{qid_label}: 'options' exceeds {MAX_OPTIONS} entries")
    options = tuple(_check_str(o, "options[]", MAX_LABEL, qid_label) for o in opts_raw)
    folded = [o.casefold() for o in options]
    if len(set(folded)) != len(folded):
        raise QuestionRegistryError(f"{qid_label}: duplicate option labels")

    ordinal = raw["ordinal"]
    if not isinstance(ordinal, bool):
        raise QuestionRegistryError(f"{qid_label}: 'ordinal' must be true or false")

    category = raw["category"]
    if category not in CATEGORIES:
        raise QuestionRegistryError(f"{qid_label}: 'category' must be one of {CATEGORIES}")

    years = raw["years"]
    if not isinstance(years, list) or len(years) != 2:
        raise QuestionRegistryError(f"{qid_label}: 'years' must be [start, end]")
    start = _check_int(years[0], "years[0]", qid_label)
    end = _check_int(years[1], "years[1]", qid_label)
    if start > end:
        raise QuestionRegistryError(f"{qid_label}: years start {start} is after end {end}")
    if start < YEAR_MIN or end > YEAR_MAX:
        raise QuestionRegistryError(
            f"{qid_label}: years must be within {YEAR_MIN}-{YEAR_MAX}"
        )

    note = _check_str(raw["contamination_note"], "contamination_note", MAX_NOTE, qid_label)

    return Question(qid, var, text, options, ordinal, category, (start, end), note)


def parse_questions(text: str) -> list[Question]:
    data = _parse_yaml(text)
    if not isinstance(data, dict) or set(data) != {"questions"}:
        raise QuestionRegistryError("top level must be a mapping with only a 'questions' key")
    raw_list = data["questions"]
    if not isinstance(raw_list, list) or not raw_list:
        raise QuestionRegistryError("'questions' must be a non-empty list")
    if len(raw_list) > MAX_QUESTIONS:
        raise QuestionRegistryError(f"too many questions (max {MAX_QUESTIONS})")

    questions = [_validate_question(r, i) for i, r in enumerate(raw_list)]
    ids = [q.id for q in questions]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise QuestionRegistryError(f"duplicate question id(s): {dupes}")
    return questions


def load_questions(path: str | Path = DEFAULT_PATH) -> list[Question]:
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError as e:
        raise QuestionRegistryError(f"cannot read {path}: {e}") from e
    if size > MAX_FILE_BYTES:
        raise QuestionRegistryError(f"{path} exceeds {MAX_FILE_BYTES} bytes")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise QuestionRegistryError(f"cannot read {path}: {e}") from e
    return parse_questions(text)
