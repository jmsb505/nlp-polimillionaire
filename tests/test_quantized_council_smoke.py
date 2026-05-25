from __future__ import annotations

import os

import pytest

from polimillionaire.strategies import CouncilStrategy, GemmaLLM, GemmaLLMConfig, QwenLLM, QwenLLMConfig
from polimillionaire.types import AnswerOption, Question


@pytest.mark.skipif(
    os.environ.get("RUN_QUANTIZED_COUNCIL_SMOKE_TEST") != "1",
    reason="Quantized council smoke test is opt-in",
)
def test_quantized_mixed_council_smoke_generation():
    question = Question(
        id=1,
        text="What is 2 + 2?",
        options=[AnswerOption(0, "3"), AnswerOption(1, "4"), AnswerOption(2, "5")],
    )
    gemma = GemmaLLM(GemmaLLMConfig(quantize_4bit=True, max_new_tokens=16, generation_max_time_seconds=20.0))
    qwen = QwenLLM(
        QwenLLMConfig(
            quantize_4bit=True,
            enable_thinking=False,
            max_new_tokens=24,
            generation_max_time_seconds=20.0,
        )
    )
    strategy = CouncilStrategy(
        candidate_llms=[gemma, qwen],
        judge_llm=gemma,
        candidate_max_new_tokens=24,
        judge_max_new_tokens=8,
        max_time_per_call=20.0,
    )

    prediction = strategy.answer(question)

    assert prediction.option_id == 1
    assert prediction.metadata["fallback"] is False
    placements = [gemma.device_summary, qwen.device_summary]
    assert not any(token in " ".join(placements).lower() for token in ("'cpu'", "'disk'", "meta"))
