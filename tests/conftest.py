from __future__ import annotations

import pytest

from polimillionaire.types import AnswerOption, Question


@pytest.fixture
def sample_question() -> Question:
    return Question(
        id=123,
        text="What is 2 + 2?",
        options=[
            AnswerOption(0, "3"),
            AnswerOption(1, "4"),
            AnswerOption(2, "5"),
            AnswerOption(3, "22"),
        ],
        level=1,
    )
