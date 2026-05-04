from __future__ import annotations

import os

import pytest

from polimillionaire.strategies import GemmaStrategy
from polimillionaire.types import AnswerOption, Question


@pytest.mark.skipif(os.environ.get("RUN_GEMMA_SMOKE_TEST") != "1", reason="Gemma smoke test is opt-in")
def test_gemma_smoke_generation():
    question = Question(
        id=1,
        text="What is 2 + 2?",
        options=[AnswerOption(0, "3"), AnswerOption(1, "4")],
    )
    prediction = GemmaStrategy().answer(question)
    assert prediction.option_id in question.valid_option_ids()
