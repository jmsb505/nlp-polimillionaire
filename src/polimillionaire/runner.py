from __future__ import annotations

"""Game runner, JSONL logs, and small result summaries."""

import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from polimillionaire.strategies import BaseStrategy
from polimillionaire.types import AnswerOption, AnswerPrediction, Question


def from_client_question(client_question: Any) -> Question:
    return Question(
        id=int(getattr(client_question, "id")),
        text=str(getattr(client_question, "text")),
        options=[
            AnswerOption(id=int(getattr(option, "id")), text=str(getattr(option, "text")))
            for option in getattr(client_question, "options")
        ],
        level=int(getattr(client_question, "level", 0) or 0),
        metadata={"source": "millionaire_client"},
    )


class SafeDelay:
    def __init__(self, seconds: float = 1.0):
        self.seconds = max(0.0, seconds)
        self._last_call: float | None = None

    def wait(self) -> None:
        now = time.monotonic()
        if self._last_call is not None:
            remaining = self.seconds - (now - self._last_call)
            if remaining > 0:
                time.sleep(remaining)
        self._last_call = time.monotonic()


class GameRunner:
    def __init__(
        self,
        client: Any,
        safe_delay_seconds: float = 1.0,
        answer_timeout_seconds: float = 25.0,
        logger: "RunLogger | None" = None,
    ):
        self.client = client
        self.safe_delay = SafeDelay(safe_delay_seconds)
        self.answer_timeout_seconds = answer_timeout_seconds
        self.logger = logger

    def play(self, competition_id: int, strategy: BaseStrategy) -> Any:
        self.safe_delay.wait()
        game = self.client.game.start(competition_id=competition_id)
        while game.in_progress:
            if game.current_question is None:
                break
            question = from_client_question(game.current_question)
            started_at = time.monotonic()
            prediction = self._safe_answer(strategy, question)
            elapsed = time.monotonic() - started_at
            self.safe_delay.wait()
            try:
                result = game.answer(prediction.option_id)
            except Exception as exc:
                result = SubmissionErrorResult(error=str(exc))
                self._log(question, prediction, result, elapsed, strategy)
                break
            self._log(question, prediction, result, elapsed, strategy)
            if getattr(result, "game_over", False):
                break
        return game

    def _safe_answer(self, strategy: BaseStrategy, question: Question) -> AnswerPrediction:
        fallback = _fallback_prediction(question, "Strategy failed or timed out.")
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(strategy.answer, question)
            prediction = future.result(timeout=self.answer_timeout_seconds)
        except TimeoutError:
            executor.shutdown(wait=False, cancel_futures=True)
            fallback.metadata["error"] = f"Timed out after {self.answer_timeout_seconds}s"
            return fallback
        except Exception as exc:
            executor.shutdown(wait=False, cancel_futures=True)
            fallback.metadata["error"] = str(exc)
            return fallback
        executor.shutdown(wait=False)
        if prediction.option_id not in question.valid_option_ids():
            fallback.metadata["error"] = "Strategy returned invalid option_id"
            fallback.metadata["original_prediction"] = prediction.metadata
            return fallback
        return prediction

    def _log(
        self,
        question: Question,
        prediction: AnswerPrediction,
        result: Any,
        elapsed_seconds: float,
        strategy: BaseStrategy,
    ) -> None:
        if self.logger:
            self.logger.log_attempt(
                question=question,
                prediction=prediction,
                result=result,
                elapsed_seconds=elapsed_seconds,
                strategy_name=getattr(strategy, "name", strategy.__class__.__name__),
            )


class RunLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_attempt(
        self,
        question: Question,
        prediction: AnswerPrediction,
        result: Any,
        elapsed_seconds: float,
        strategy_name: str,
    ) -> None:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy_name": strategy_name,
            "question": asdict(question),
            "prediction": asdict(prediction),
            "elapsed_seconds": elapsed_seconds,
            "result": {
                "correct": getattr(result, "correct", None),
                "game_over": getattr(result, "game_over", None),
                "earned_amount": getattr(result, "earned_amount", None),
                "timed_out": getattr(result, "timed_out", None),
                "status": getattr(result, "status", None),
                "current_level": getattr(result, "current_level", None),
                "error": getattr(result, "error", None),
            },
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize_attempts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(1 for row in rows if row.get("result", {}).get("correct") is True)
    timed_out = sum(1 for row in rows if row.get("result", {}).get("timed_out") is True)
    elapsed_values = [row.get("elapsed_seconds", 0.0) for row in rows]
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "timed_out": timed_out,
        "avg_elapsed_seconds": sum(elapsed_values) / total if total else 0.0,
    }


def _fallback_prediction(question: Question, reason: str) -> AnswerPrediction:
    option = question.first_option()
    return AnswerPrediction(
        option_id=option.id,
        answer_text=option.text,
        confidence=0.0,
        reasoning=reason,
        metadata={"fallback": True},
    )


@dataclass
class SubmissionErrorResult:
    correct: bool | None = None
    game_over: bool = True
    earned_amount: float | None = None
    timed_out: bool = False
    status: str = "submission_error"
    current_level: int | None = None
    error: str = ""
