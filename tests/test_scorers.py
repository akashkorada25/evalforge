from evalforge.config import Case
from evalforge.scorers import cosine_similarity, parse_judge, run_check, validate_schema


def case(**kw):
    return Case(id="c", vars={"ticket": "x"}, **kw)


def test_exact_ignores_case_and_whitespace():
    r = run_check({"type": "exact", "value": "Paris"}, "  paris ", case(), {})
    assert r.passed and r.score == 1.0


def test_contains_partial_score():
    r = run_check({"type": "contains", "value": ["refund", "49.99", "march"]}, "Refund of $49.99", case(), {})
    assert not r.passed
    assert abs(r.score - 2 / 3) < 1e-9
    assert "march" in r.reason


def test_not_contains():
    assert run_check({"type": "not_contains", "value": ["sorry"]}, "All good", case(), {}).passed
    assert not run_check({"type": "not_contains", "value": ["sorry"]}, "Sorry!", case(), {}).passed


def test_regex():
    r = run_check({"type": "regex", "pattern": r'"priority"\s*:\s*"urgent"'}, '{"priority": "urgent"}', case(), {})
    assert r.passed


def test_json_schema_valid_with_code_fence():
    schema = {"type": "object", "required": ["a"], "properties": {"a": {"type": "string", "enum": ["x", "y"]}}}
    r = run_check({"type": "json_schema", "schema": schema}, '```json\n{"a": "x"}\n```', case(), {})
    assert r.passed


def test_json_schema_errors():
    schema = {"type": "object", "required": ["a", "b"], "properties": {"a": {"type": "integer"}}}
    errors = validate_schema({"a": "nope"}, schema)
    assert any("missing required key 'b'" in e for e in errors)
    assert any("expected integer" in e for e in errors)
    assert validate_schema(True, {"type": "integer"})  # bool is not an integer


def test_invalid_json_fails():
    r = run_check({"type": "json_schema", "schema": {"type": "object"}}, "{broken", case(), {})
    assert not r.passed and r.score == 0.0


def test_max_words():
    assert run_check({"type": "max_words", "value": 3}, "one two three", case(), {}).passed
    assert not run_check({"type": "max_words", "value": 3}, "one two three four", case(), {}).passed


def test_similarity_uses_expected():
    c = case(expected="The order arrived damaged")
    assert run_check({"type": "similarity", "threshold": 0.5}, "order arrived damaged", c, {}).passed
    assert cosine_similarity("apples", "spaceships") == 0.0


def test_unknown_check_type_fails_cleanly():
    r = run_check({"type": "nope"}, "x", case(), {})
    assert not r.passed and "Unknown" in r.reason


def test_parse_judge_formats():
    assert parse_judge('{"score": 4, "reason": "good"}') == (4, "good")
    assert parse_judge('Sure! ```{"score": 2, "reason": "meh"}```')[0] == 2
    assert parse_judge("I would rate this 5 out of 5")[0] == 5


def test_llm_judge_without_judge_fails():
    r = run_check({"type": "llm_judge"}, "x", case(), {"judge": None})
    assert not r.passed


def test_check_name_defaults_to_type_and_can_be_set():
    assert run_check({"type": "contains", "value": "a"}, "a", case(), {}).name == "contains"
    assert run_check({"type": "contains", "value": "a", "name": "mentions-a"}, "a", case(), {}).name == "mentions-a"
