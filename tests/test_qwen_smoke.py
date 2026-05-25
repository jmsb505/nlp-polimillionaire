from __future__ import annotations

import os

import pytest

from polimillionaire.strategies import QwenLLMConfig, QwenStrategy
from polimillionaire.types import AnswerOption, Question


@pytest.mark.skipif(os.environ.get("RUN_QWEN_SMOKE_TEST") != "1", reason="Qwen smoke test is opt-in")
def test_qwen_smoke_generation():
    question = Question(
        id=1,
        text="What is 2 + 2?",
        options=[AnswerOption(0, "3"), AnswerOption(1, "4")],
    )
    strategy = QwenStrategy(
        model_config=QwenLLMConfig(max_new_tokens=256, generation_max_time_seconds=60.0)
    )
    prediction = strategy.answer(question)
    assert prediction.option_id in question.valid_option_ids()
