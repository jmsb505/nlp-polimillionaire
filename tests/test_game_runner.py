from __future__ import annotations

from dataclasses import dataclass

from polimillionaire.runner import GameRunner, RunLogger, SpeechGameRunner
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


class FakeSpeechGame(FakeGame):
    def __init__(self):
        super().__init__()
        self.current_question = ClientQuestion(
            id=7,
            text=None,
            options=[ClientOption(10, None), ClientOption(20, None)],
        )
        self.mode = "speech"
        self._option_audios = [b"option-a", b"option-b"]
        self.started_mode = None

    @property
    def time_remaining(self):
        return 29.0

    def fetch_audio_question(self):
        return b"question"

    def fetch_audio_option_next(self):
        return self._option_audios.pop(0)

    def refresh_state(self):
        return None


class FakeSpeechGameModule:
    def __init__(self):
        self.game = FakeSpeechGame()

    def start(self, competition_id: int, mode: str = "text"):
        self.game.started_mode = mode
        return self.game


class FakeSpeechClient:
    def __init__(self):
        self.game = FakeSpeechGameModule()


class PickSecondStrategy(BaseStrategy):
    name = "pick-second"

    def answer(self, question: Question) -> AnswerPrediction:
        option = question.options[1]
        return AnswerPrediction(option_id=option.id, answer_text=option.text)


def test_speech_game_runner_transcribes_and_submits(tmp_path):
    client = FakeSpeechClient()
    log_path = tmp_path / "speech.jsonl"

    def transcriber(audio_bytes: bytes) -> str:
        return {
            b"question": "What is the answer?",
            b"option-a": "Wrong answer",
            b"option-b": "Right answer",
        }[audio_bytes]

    runner = SpeechGameRunner(
        client=client,
        safe_delay_seconds=0,
        answer_timeout_seconds=1,
        logger=RunLogger(log_path),
        transcriber=transcriber,
        audio_fetch_delay_seconds=0,
    )
    runner.play(competition_id=0, strategy=PickSecondStrategy())

    assert client.game.game.started_mode == "speech"
    assert client.game.game.received_answer == 20
    text = log_path.read_text(encoding="utf-8")
    assert '"mode": "speech"' in text
    assert "Right answer" in text


def test_speech_game_runner_limits_strategy_timeout_after_transcription():
    client = FakeSpeechClient()

    def transcriber(audio_bytes: bytes) -> str:
        return {
            b"question": "What is the answer?",
            b"option-a": "Wrong answer",
            b"option-b": "Right answer",
        }[audio_bytes]

    class CapturingSpeechRunner(SpeechGameRunner):
        captured_timeout = None

        def _safe_answer(self, strategy, question, timeout_seconds=None):
            self.captured_timeout = timeout_seconds
            return super()._safe_answer(strategy, question, timeout_seconds=timeout_seconds)

    client.game.game._deadline_time_remaining = 3.0
    type(client.game.game).time_remaining = property(lambda self: self._deadline_time_remaining)
    runner = CapturingSpeechRunner(
        client=client,
        safe_delay_seconds=0,
        answer_timeout_seconds=25,
        transcriber=transcriber,
        audio_fetch_delay_seconds=0,
    )
    runner.play(competition_id=0, strategy=PickSecondStrategy())

    assert runner.captured_timeout == 2.0
