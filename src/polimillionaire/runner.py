from __future__ import annotations

"""Game runner, JSONL logs, and small result summaries."""

import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

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

    def _safe_answer(
        self,
        strategy: BaseStrategy,
        question: Question,
        timeout_seconds: float | None = None,
    ) -> AnswerPrediction:
        fallback = _fallback_prediction(question, "Strategy failed or timed out.")
        timeout_seconds = self.answer_timeout_seconds if timeout_seconds is None else timeout_seconds
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(strategy.answer, question)
            prediction = future.result(timeout=timeout_seconds)
        except TimeoutError:
            executor.shutdown(wait=False, cancel_futures=True)
            fallback.metadata["error"] = f"Timed out after {timeout_seconds}s"
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


class SpeechGameRunner(GameRunner):
    def __init__(
        self,
        client: Any,
        safe_delay_seconds: float = 1.0,
        answer_timeout_seconds: float = 25.0,
        logger: "RunLogger | None" = None,
        transcriber: Callable[[bytes], str] | None = None,
        audio_dir: str | Path | None = None,
        audio_fetch_delay_seconds: float = 0.2,
    ):
        super().__init__(client, safe_delay_seconds, answer_timeout_seconds, logger)
        self.transcriber = transcriber
        self.audio_dir = Path(audio_dir) if audio_dir is not None else None
        self.audio_fetch_delay_seconds = max(0.0, audio_fetch_delay_seconds)

    def play(self, competition_id: int, strategy: BaseStrategy) -> Any:
        self.safe_delay.wait()
        game = self.client.game.start(competition_id=competition_id, mode="speech")
        while game.in_progress:
            if game.current_question is None:
                break
            self.safe_delay.wait()
            fetch_started_at = time.monotonic()
            question_audio = game.fetch_audio_question()
            self._sleep_between_audio_requests()
            option_audios = []
            for _ in range(len(getattr(game.current_question, "options", [])) or 4):
                option_audios.append(game.fetch_audio_option_next())
                self._sleep_between_audio_requests()
            audio_fetch_seconds = time.monotonic() - fetch_started_at
            if hasattr(game, "refresh_state"):
                try:
                    game.refresh_state()
                except Exception:
                    pass

            started_at = time.monotonic()
            question = self._question_from_audio(game, question_audio, option_audios, audio_fetch_seconds)
            remaining_before_strategy = getattr(game, "time_remaining", None)
            question.metadata["time_remaining_before_strategy"] = remaining_before_strategy
            strategy_timeout = self.answer_timeout_seconds
            if remaining_before_strategy is not None:
                strategy_timeout = min(strategy_timeout, max(1.0, float(remaining_before_strategy) - 1.0))
            if remaining_before_strategy is not None and remaining_before_strategy <= 1.0:
                prediction = _fallback_prediction(question, "Not enough time after speech transcription.")
            else:
                prediction = self._safe_answer(strategy, question, timeout_seconds=strategy_timeout)
            elapsed = time.monotonic() - started_at
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

    def _question_from_audio(
        self,
        game: Any,
        question_audio: bytes,
        option_audios: list[bytes],
        audio_fetch_seconds: float,
    ) -> Question:
        client_question = game.current_question
        option_ids = [
            int(getattr(option, "id", index))
            for index, option in enumerate(getattr(client_question, "options", []))
        ]
        if not option_ids:
            option_ids = list(range(len(option_audios)))

        q_text, q_error = self._safe_transcribe(question_audio)
        option_texts = []
        option_errors = []
        for index, audio in enumerate(option_audios):
            text, error = self._safe_transcribe(audio)
            option_texts.append(text or f"Option {chr(65 + index)}")
            option_errors.append(error)

        question_id = int(getattr(client_question, "id", getattr(game, "current_level", 0) or 0))
        level = int(getattr(client_question, "level", getattr(game, "current_level", 0) or 0) or 0)
        question_text = q_text or "Speech question transcript unavailable."
        options = [
            AnswerOption(id=option_ids[index], text=option_texts[index])
            for index in range(min(len(option_ids), len(option_texts)))
        ]
        if not options:
            options = [AnswerOption(id=0, text="Option A")]
        self._save_audio(question_audio, question_id, level, "question")
        for index, audio in enumerate(option_audios):
            self._save_audio(audio, question_id, level, f"option_{chr(65 + index)}")
        return Question(
            id=question_id,
            text=question_text,
            options=options,
            level=level,
            metadata={
                "source": "millionaire_client",
                "mode": "speech",
                "audio_fetch_seconds": audio_fetch_seconds,
                "time_remaining_after_audio_fetch": getattr(game, "time_remaining", None),
                "question_transcription_error": q_error,
                "option_transcription_errors": option_errors,
            },
        )

    def _safe_transcribe(self, audio_bytes: bytes) -> tuple[str, str | None]:
        try:
            transcriber = self.transcriber
            if transcriber is None:
                from polimillionaire.transcribe import transcribe

                transcriber = transcribe
            return str(transcriber(audio_bytes)).strip(), None
        except Exception as exc:
            return "", str(exc)

    def _save_audio(self, audio_bytes: bytes, question_id: int, level: int, label: str) -> None:
        if self.audio_dir is None:
            return
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        path = self.audio_dir / f"level_{level}_question_{question_id}_{label}.wav"
        path.write_bytes(audio_bytes)

    def _sleep_between_audio_requests(self) -> None:
        if self.audio_fetch_delay_seconds > 0:
            time.sleep(self.audio_fetch_delay_seconds)


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


def benchmark_strategy(strategy: BaseStrategy, cases: list[tuple[Question, int]]) -> dict[str, Any]:
    rows = []
    for question, gold_id in cases:
        started_at = time.monotonic()
        prediction = strategy.answer(question)
        elapsed = time.monotonic() - started_at
        votes = prediction.metadata.get("votes") or []
        vote_options = {vote.get("option_id") for vote in votes if isinstance(vote, dict)}
        rows.append(
            {
                "question_id": question.id,
                "prediction": prediction,
                "gold_id": gold_id,
                "correct": prediction.option_id == gold_id,
                "elapsed_seconds": elapsed,
                "fallback": bool(prediction.metadata.get("fallback")),
                "disagreement": len(vote_options) > 1,
            }
        )

    total = len(rows)
    elapsed_values = [row["elapsed_seconds"] for row in rows]
    return {
        "total": total,
        "correct": sum(1 for row in rows if row["correct"]),
        "accuracy": sum(1 for row in rows if row["correct"]) / total if total else 0.0,
        "fallbacks": sum(1 for row in rows if row["fallback"]),
        "avg_elapsed_seconds": sum(elapsed_values) / total if total else 0.0,
        "max_elapsed_seconds": max(elapsed_values) if elapsed_values else 0.0,
        "disagreements": sum(1 for row in rows if row["disagreement"]),
        "rows": rows,
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
