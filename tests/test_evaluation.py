from __future__ import annotations

from polimillionaire.runner import benchmark_strategy, summarize_attempts
from polimillionaire.strategies import BaseStrategy
from polimillionaire.types import AnswerOption, AnswerPrediction, Question


def test_summarize_attempts_counts_accuracy():
    summary = summarize_attempts(
        [
            {"elapsed_seconds": 1.0, "result": {"correct": True, "timed_out": False}},
            {"elapsed_seconds": 3.0, "result": {"correct": False, "timed_out": True}},
        ]
    )
    assert summary["total"] == 2
    assert summary["correct"] == 1
    assert summary["accuracy"] == 0.5
    assert summary["timed_out"] == 1
    assert summary["avg_elapsed_seconds"] == 2.0


class FixedStrategy(BaseStrategy):
    name = "fixed"

    def __init__(self, option_id: int, votes: list[dict] | None = None):
        self.option_id = option_id
        self.votes = votes or []

    def answer(self, question: Question) -> AnswerPrediction:
        option = question.require_option(self.option_id)
        return AnswerPrediction(
            option_id=option.id,
            answer_text=option.text,
            metadata={"fallback": False, "votes": self.votes},
        )


def test_benchmark_strategy_reports_speed_and_disagreement():
    question = Question(1, "What is 2 + 2?", [AnswerOption(0, "3"), AnswerOption(1, "4")])
    strategy = FixedStrategy(1, votes=[{"option_id": 0}, {"option_id": 1}])
    summary = benchmark_strategy(strategy, [(question, 1)])
    assert summary["accuracy"] == 1.0
    assert summary["fallbacks"] == 0
    assert summary["max_elapsed_seconds"] >= 0.0
    assert summary["disagreements"] == 1
