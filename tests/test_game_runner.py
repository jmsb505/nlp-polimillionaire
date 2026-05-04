from __future__ import annotations

from dataclasses import dataclass

from polimillionaire.runner import GameRunner, RunLogger
from polimillionaire.strategies import BaseStrategy
from polimillionaire.types import AnswerPrediction, Question


@dataclass
class ClientOption:
    id: int
    text: str


@dataclass
class ClientQuestion:
    id: int
    text: str
    options: list[ClientOption]
    level: int = 1


class FakeResult:
    correct = False
    game_over = True
    earned_amount = 0
    timed_out = False
    status = "failed"
    current_level = 1


class FakeGame:
    def __init__(self):
        self.current_question = ClientQuestion(
            id=1,
            text="Question?",
            options=[ClientOption(0, "A"), ClientOption(1, "B")],
        )
        self.in_progress = True
        self.received_answer = None
        self.current_level = 1
        self.earned_amount = 0

    def answer(self, option_id: int):
        self.received_answer = option_id
        self.in_progress = False
        return FakeResult()


class FakeGameModule:
    def __init__(self):
        self.game = FakeGame()

    def start(self, competition_id: int):
        return self.game


class FakeClient:
    def __init__(self):
        self.game = FakeGameModule()


class FailingStrategy(BaseStrategy):
    name = "failing"

    def answer(self, question: Question) -> AnswerPrediction:
        raise RuntimeError("boom")


def test_game_runner_falls_back_on_strategy_failure():
    client = FakeClient()
    runner = GameRunner(client=client, safe_delay_seconds=0, answer_timeout_seconds=1)
    runner.play(competition_id=0, strategy=FailingStrategy())
    assert client.game.game.received_answer == 0


class GoodStrategy(BaseStrategy):
    name = "good"

    def answer(self, question: Question) -> AnswerPrediction:
        option = question.first_option()
        return AnswerPrediction(option_id=option.id, answer_text=option.text)


class DisconnectingGame(FakeGame):
    def answer(self, option_id: int):
        self.received_answer = option_id
        raise RuntimeError("remote disconnected")


class DisconnectingGameModule(FakeGameModule):
    def __init__(self):
        self.game = DisconnectingGame()


class DisconnectingClient(FakeClient):
    def __init__(self):
        self.game = DisconnectingGameModule()


def test_game_runner_logs_submission_error(tmp_path):
    client = DisconnectingClient()
    log_path = tmp_path / "run.jsonl"
    runner = GameRunner(
        client=client,
        safe_delay_seconds=0,
        answer_timeout_seconds=1,
        logger=RunLogger(log_path),
    )
    runner.play(competition_id=0, strategy=GoodStrategy())
    text = log_path.read_text(encoding="utf-8")
    assert "submission_error" in text
    assert "remote disconnected" in text
