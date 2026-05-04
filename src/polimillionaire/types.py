from __future__ import annotations

"""Small dataclasses shared by the assignment notebook helpers."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AnswerOption:
    id: int
    text: str


@dataclass(frozen=True)
class Question:
    id: int
    text: str
    options: list[AnswerOption]
    level: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def valid_option_ids(self) -> set[int]:
        return {option.id for option in self.options}

    def first_option(self) -> AnswerOption:
        if not self.options:
            raise ValueError("Question has no answer options")
        return self.options[0]

    def get_option(self, option_id: int) -> AnswerOption | None:
        for option in self.options:
            if option.id == option_id:
                return option
        return None

    def require_option(self, option_id: int) -> AnswerOption:
        return self.get_option(option_id) or self.first_option()


@dataclass
class AnswerPrediction:
    option_id: int
    answer_text: str
    confidence: float | None = None
    reasoning: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
