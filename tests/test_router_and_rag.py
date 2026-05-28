from __future__ import annotations

from dataclasses import dataclass

from polimillionaire.runner import benchmark_strategy
from polimillionaire.strategies import (
    FakeLLM,
    GemmaStrategy,
    RAGConfig,
    RAGStrategy,
    RoutedStrategy,
    route_question,
)
from polimillionaire.types import AnswerOption, Question


@dataclass
class FakeDocument:
    page_content: str
    metadata: dict[str, str]


def _question(text: str, options: list[str]) -> Question:
    return Question(
        id=1,
        text=text,
        options=[AnswerOption(index, option) for index, option in enumerate(options)],
    )


def test_router_sends_math_away_from_rag():
    question = _question("What is 2 + 2?", ["3", "4", "5", "22"])
    decision = route_question(question)
    assert decision.route == "direct"
    assert decision.reason == "math"


def test_router_allows_factual_questions_to_use_rag():
    question = _question(
        "Which Beatles song introduced Paul McCartney to a falsetto vocal style?",
        ["Let It Be", "I'm Down", "Yesterday", "Hey Jude"],
    )
    decision = route_question(question)
    assert decision.route == "rag"
    assert decision.reason == "factual"


def test_router_handles_negation_as_direct_model_prompt():
    question = _question(
        "Which song was NOT written by Bob Dylan?",
        ["Like a Rolling Stone", "Blowin' in the Wind", "The Times They Are A-Changin'", "Hound Dog"],
    )
    decision = route_question(question)
    assert decision.route == "direct"
    assert decision.reason == "negation"


def test_routed_strategy_bypasses_rag_for_math(sample_question):
    direct = GemmaStrategy(llm=FakeLLM(['{"option_id": 1, "confidence": 1.0, "reason": "math"}']))
    rag = RAGStrategy(
        llm=FakeLLM(['{"option_id": 2, "confidence": 1.0, "reason": "bad retrieval"}']),
        retriever=lambda query, cfg, llm: (_raise("RAG should not run"), [], 0.0),
    )
    prediction = RoutedStrategy(direct_strategy=direct, rag_strategy=rag).answer(sample_question)
    assert prediction.option_id == 1
    assert prediction.metadata["route"] == "direct"
    assert prediction.metadata["routed_to"] == "gemma"


def test_routed_strategy_uses_backup_for_low_confidence_rag():
    rag = RAGStrategy(
        llm=FakeLLM(['{"option_id": 0, "confidence": 0.4, "reason": "weak"}']),
        retriever=lambda query, cfg, llm: ("weak evidence", [], 0.01),
    )
    backup = GemmaStrategy(llm=FakeLLM(['{"option_id": 1, "confidence": 0.8, "reason": "backup"}']))
    question = _question(
        "Which Beatles song introduced Paul McCartney to a falsetto vocal style?",
        ["Let It Be", "I'm Down", "Yesterday", "Hey Jude"],
    )
    prediction = RoutedStrategy(
        direct_strategy=backup,
        rag_strategy=rag,
        fallback_strategy=backup,
        low_confidence_strategy=backup,
        rag_min_confidence=0.7,
    ).answer(question)
    assert prediction.option_id == 1
    assert prediction.metadata["backup_for_low_confidence_rag"] is True
    assert prediction.metadata["rag_prediction"]["option_id"] == 0


def test_rag_prompt_warns_about_overlap_and_causal_questions():
    question = _question(
        "What event directly led to the decision?",
        ["Earlier cause", "Later action"],
    )
    prompt = __import__("polimillionaire.strategies", fromlist=["build_rag_prompt"]).build_rag_prompt(
        question,
        "Earlier cause happened before later action.",
    )
    assert "Options may overlap" in prompt
    assert "earlier cause" in prompt


def test_rag_metadata_uses_fake_local_documents():
    docs = [
        FakeDocument(
            page_content="Whitney Houston was the debut studio album by American singer Whitney Houston.",
            metadata={"title": "Whitney Houston album", "url": "local://whitney"},
        )
    ]

    def fake_retriever(query, cfg, llm):
        return "Whitney Houston was the debut studio album.", docs, 0.12

    strategy = RAGStrategy(
        llm=FakeLLM(['{"option_id": 0, "confidence": 0.9, "reason": "evidence"}']),
        retrieval_config=RAGConfig(final_top_k=1),
        retriever=fake_retriever,
    )
    question = _question(
        "What was Whitney Houston's debut album?",
        ["Whitney Houston", "Just Whitney", "I'm Your Baby Tonight", "Whitney"],
    )
    prediction = strategy.answer(question)
    assert prediction.option_id == 0
    assert prediction.metadata["fallback"] is False
    assert prediction.metadata["retrieval_seconds"] == 0.12
    assert prediction.metadata["evidence_sources"][0]["url"] == "local://whitney"
    assert "Whitney Houston" in prediction.metadata["evidence_preview"]


def test_benchmark_summary_flags_misleading_rag_answer():
    strategy = RAGStrategy(
        llm=FakeLLM(['{"option_id": 0, "confidence": 0.9, "reason": "misleading evidence"}']),
        retriever=lambda query, cfg, llm: ("Let It Be has falsetto vocals.", [], 0.01),
    )
    question = _question(
        "Which Beatles song is known for its unique falsetto vocal technique?",
        ["Let It Be", "I'm Down", "Yesterday", "Hey Jude"],
    )
    summary = benchmark_strategy(strategy, [(question, 1)])
    assert summary["accuracy"] == 0.0
    assert summary["fallbacks"] == 0
    assert summary["max_elapsed_seconds"] >= 0.0


def _raise(message: str):
    raise AssertionError(message)
