from __future__ import annotations

from polimillionaire.strategies import parse_answer_prediction


def test_parser_handles_valid_json(sample_question):
    prediction = parse_answer_prediction(
        '{"option_id": 1, "confidence": 0.72, "reason": "Brief."}',
        sample_question,
    )
    assert prediction.option_id == 1
    assert prediction.confidence == 0.72
    assert prediction.reasoning == "Brief."


def test_parser_falls_back_on_malformed_output(sample_question):
    prediction = parse_answer_prediction("I do not know", sample_question)
    assert prediction.option_id == sample_question.first_option().id
    assert prediction.metadata["fallback"] is True


def test_parser_falls_back_on_wrong_option_id(sample_question):
    prediction = parse_answer_prediction('{"option_id": 99, "confidence": 0.9}', sample_question)
    assert prediction.option_id == sample_question.first_option().id
    assert prediction.metadata["fallback"] is True


def test_parser_allows_missing_confidence(sample_question):
    prediction = parse_answer_prediction('{"option_id": 1, "reason": "Brief."}', sample_question)
    assert prediction.option_id == 1
    assert prediction.confidence is None


def test_parser_accepts_bare_option_id(sample_question):
    prediction = parse_answer_prediction("1", sample_question)
    assert prediction.option_id == 1
    assert prediction.metadata["fallback"] is False


def test_parser_handles_truncated_json_reason(sample_question):
    raw = '```json\n{\n"option_id": 1,\n"confidence": 0.9,\n"reason": "The film heavily implies the'
    prediction = parse_answer_prediction(raw, sample_question)
    assert prediction.option_id == 1
    assert prediction.confidence == 0.9
    assert prediction.reasoning == "The film heavily implies the"
