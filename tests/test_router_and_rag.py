from __future__ import annotations

from dataclasses import dataclass

from polimillionaire.runner import benchmark_strategy
from polimillionaire.strategies import (
    CalculatorStrategy,
    FakeLLM,
    GemmaStrategy,
    RAGConfig,
    RoutedRAGCouncilStrategy,
    RAGStrategy,
    RoutedStrategy,
    _retrieval_query,
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


def test_router_does_not_treat_time_signature_option_as_math():
    question = _question(
        "Which of the following is a key characteristic of hard bop?",
        ["Incorporation of blues and gospel elements", "Use of electronic synthesizers", "No structured form", "Strict adherence to 4/4 time"],
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


def test_router_sends_recent_report_questions_to_rag():
    question = _question(
        "According to the 2026-05-17 report, which head coach of Panama said this?",
        ["Diego Maradona", "Thomas Christiansen", "Jorge Sampaoli", "Luis Suarez"],
    )
    decision = route_question(question)
    assert decision.route == "rag"
    assert decision.reason == "recent_or_report"


def test_calculator_handles_sum_of_squares_question():
    question = _question(
        "The sum $1^2 + 2^2 + 3^2 + 4^2 + \\cdots + n^2 = n(n+1)(2n+1) \\div 6$. What is the value of $21^2 + 22^2 + \\cdots + 40^2$?",
        ["41", "2870", "22140", "19270"],
    )
    prediction = CalculatorStrategy().answer(question)
    assert prediction.option_id == 3
    assert prediction.metadata["tool"] == "calculator"


def test_calculator_handles_binomial_question():
    question = _question(
        "Compute \\dbinom{85}{82}.",
        ["252", "101170", "98770", "4680"],
    )
    prediction = CalculatorStrategy().answer(question)
    assert prediction.option_id == 2
    assert prediction.metadata["calculation_method"] == "combination"


def test_calculator_handles_wave_frequency_question():
    question = _question(
        "An organ pipe has a wavelength of 2.72 m. What is the frequency if the speed of sound is 348 m/s?",
        ["466 Hz", "85.7 Hz", "128 Hz", "260 Hz"],
    )
    prediction = CalculatorStrategy().answer(question)
    assert prediction.option_id == 2
    assert prediction.metadata["calculation_method"] == "wave_frequency"


def test_calculator_handles_wave_frequency_without_of_word():
    question = _question(
        "An organ pipe produces a note with wavelength 2.67 m. If the speed of sound is 343 m/s, what is the frequency?",
        ["85.7 Hz", "128 Hz", "343 Hz", "2.67 Hz"],
    )
    prediction = CalculatorStrategy().answer(question)
    assert prediction.option_id == 1
    assert prediction.metadata["calculation_method"] == "wave_frequency"


def test_calculator_handles_explicit_sum_of_squares():
    question = _question(
        "What is 1^2 + 2^2 + 3^2 + 4^2?",
        ["16", "24", "30", "40"],
    )
    prediction = CalculatorStrategy().answer(question)
    assert prediction.option_id == 2
    assert prediction.metadata["calculation_method"] == "sum_of_squares"


def test_calculator_handles_distance_speed_time_question():
    question = _question(
        "How much time is required for a bicycle to travel a distance of 100 meters at an average speed of 2 meters per second?",
        ["200 seconds", "50 seconds", "0.02 seconds", "100 seconds"],
    )
    prediction = CalculatorStrategy().answer(question)
    assert prediction.option_id == 1
    assert prediction.metadata["calculation_method"] == "time_distance_speed"


def test_calculator_handles_correlation_transform_question():
    question = _question(
        "Suppose the correlation between two variables is 0.19. What is the new correlation if 0.23 is added to all values of x, y is doubled, and the variables are interchanged?",
        ["0.84", "0.42", "0.19", "-0.19"],
    )
    prediction = CalculatorStrategy().answer(question)
    assert prediction.option_id == 2
    assert prediction.metadata["calculation_method"] == "correlation_transform"


def test_calculator_handles_matched_pairs_statistics_question():
    question = _question(
        "Volunteers tried the old formula on one side of their face and the new formula on the other. The response variable was the difference in pimples. Which significance test is correct?",
        ["A two-sample t-test", "A matched pairs t-test", "A two-proportion z-test", "A chi-square test of independence"],
    )
    prediction = CalculatorStrategy().answer(question)
    assert prediction.option_id == 1
    assert prediction.metadata["calculation_method"] == "statistics_test"


def test_calculator_handles_basic_group_homomorphism_count():
    question = _question(
        "How many homomorphisms are there of Z into Z_2?",
        ["infinitely many", "0", "1", "2"],
    )
    prediction = CalculatorStrategy().answer(question)
    assert prediction.option_id == 3
    assert prediction.metadata["calculation_method"] == "homomorphism_count"


def test_calculator_handles_linear_transformation_from_plane():
    question = _question(
        "If f is a linear transformation from the plane to the real numbers and if f(1, 1) = 1 and f(-1, 0) = 2, then f(3, 5) =",
        ["8", "-5", "0", "9"],
    )
    prediction = CalculatorStrategy().answer(question)
    assert prediction.option_id == 3
    assert prediction.metadata["calculation_method"] == "linear_transformation"


def test_calculator_handles_linear_interpolation_word_problem():
    question = _question(
        "In 1960, there were 450,000 cases of measles reported in the U.S. In 1996, there were 500 cases reported. How many cases of measles would have been reported in 1987 if the number of cases reported from 1960 to 1996 decreased linearly?",
        ["449500", "27", "337125", "112875"],
    )
    prediction = CalculatorStrategy().answer(question)
    assert prediction.option_id == 3
    assert prediction.metadata["calculation_method"] == "linear_interpolation"


def test_calculator_handles_proportion_sample_size():
    question = _question(
        "A major polling organization wants to predict the outcome of an upcoming national election. They intend to use a 95% confidence interval with margin of error of no more than 2.5%. What is the minimum sample size needed?",
        ["1537", "39", "1536", "40"],
    )
    prediction = CalculatorStrategy().answer(question)
    assert prediction.option_id == 0
    assert prediction.metadata["calculation_method"] == "proportion_sample_size"


def test_calculator_handles_normal_distribution_iqr():
    question = _question(
        "Random variable X is normally distributed, with a mean of 25 and a standard deviation of 4. Which of the following is the approximate interquartile range for this distribution?",
        ["27.70 - 22.30 = 5.40", "27.70 / 22.30 = 1.24", "2.00(4.00) = 8.00", "25.00 - 22.30 = 2.70"],
    )
    prediction = CalculatorStrategy().answer(question)
    assert prediction.option_id == 0
    assert prediction.metadata["calculation_method"] == "normal_iqr"


def test_router_sends_binomial_question_to_direct_calculator_path():
    question = _question(
        "Compute \\binom{10}{3}.",
        ["30", "120", "720", "1000"],
    )
    decision = route_question(question)
    assert decision.route == "direct"
    assert decision.reason == "math"


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
        "Which album was Whitney Houston's debut studio album?",
        ["Whitney Houston", "Just Whitney", "I'm Your Baby Tonight", "Whitney"],
    )
    prediction = strategy.answer(question)
    assert prediction.option_id == 0
    assert prediction.metadata["fallback"] is False
    assert prediction.metadata["retrieval_seconds"] == 0.12
    assert prediction.metadata["evidence_sources"][0]["url"] == "local://whitney"
    assert "Whitney Houston" in prediction.metadata["evidence_preview"]


def test_generic_factual_retrieval_query_does_not_append_options():
    question = _question(
        "What was Whitney Houston's debut album?",
        ["Whitney Houston", "Just Whitney", "I'm Your Baby Tonight", "Whitney"],
    )
    query = _retrieval_query(question)
    assert query == question.text
    assert "Just Whitney" not in query


def test_routed_rag_council_adds_general_evidence_verifier_vote():
    first = FakeLLM(['{"option_id": 3, "confidence": 1.0, "reason": "shorter related title"}'], model_name="candidate")
    judge = FakeLLM(["0"], model_name="judge")

    def fake_retriever(query, cfg, llm):
        return "[1] Whitney Houston album\nURL: local://whitney\nWhitney Houston was the debut studio album by American singer Whitney Houston.", [], 0.05

    strategy = RoutedRAGCouncilStrategy(
        candidate_llms=[first],
        judge_llm=judge,
        retriever=fake_retriever,
        always_judge=True,
    )
    question = _question(
        "What was Whitney Houston's debut album?",
        ["Whitney Houston", "Just Whitney", "I'm Your Baby Tonight", "Whitney"],
    )
    prediction = strategy.answer(question)
    assert prediction.option_id == 0
    assert prediction.metadata["decision_method"] == "evidence_verifier"
    assert any(vote["style"] == "evidence_verifier" for vote in prediction.metadata["votes"])
    assert len(first.calls) == 1
    assert len(judge.calls) == 0


def test_routed_rag_council_trusts_strong_evidence_before_qwen_judge():
    first = FakeLLM(['{"option_id": 3, "confidence": 0.95, "reason": "Evidence [1] explicitly states"}'], model_name="candidate-a")
    second = FakeLLM(['{"option_id": 3, "confidence": 0.0, "reason": "Option 3"}'], model_name="candidate-b")
    judge = FakeLLM(["3"], model_name="qwen-judge")

    def fake_retriever(query, cfg, llm):
        return "[1] Whitney Houston (album)\nURL: local://whitney\nWhitney Houston is the debut studio album by American singer Whitney Houston.", [], 0.05

    strategy = RoutedRAGCouncilStrategy(
        candidate_llms=[first, second],
        judge_llm=judge,
        retriever=fake_retriever,
        always_judge=True,
    )
    question = _question(
        "What was Whitney Houston's debut album?",
        ["Whitney Houston", "Just Whitney", "I'm Your Baby Tonight", "Whitney"],
    )
    prediction = strategy.answer(question)
    assert prediction.option_id == 0
    assert prediction.metadata["decision_method"] == "evidence_verifier"
    assert len(judge.calls) == 0


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


def test_routed_rag_council_retrieves_once_then_uses_judge():
    docs = [
        FakeDocument(
            page_content="Whitney Houston was the debut studio album by Whitney Houston.",
            metadata={"title": "Whitney Houston album", "url": "local://whitney"},
        )
    ]
    first = FakeLLM(['{"option_id": 0, "confidence": 0.8, "reason": "evidence"}'], model_name="e2b-a")
    second = FakeLLM(['{"option_id": 3, "confidence": 0.6, "reason": "short title"}'], model_name="e2b-b")
    judge = FakeLLM(["0"], model_name="e4b-judge")
    calls = []

    def fake_retriever(query, cfg, llm):
        calls.append(query)
        return "Whitney Houston was the debut studio album.", docs, 0.05

    strategy = RoutedRAGCouncilStrategy(
        candidate_llms=[first, second],
        judge_llm=judge,
        retriever=fake_retriever,
        candidate_styles=["evidence_checker", "option_eliminator"],
    )
    question = _question(
        "Which album was Whitney Houston's debut studio album?",
        ["Whitney Houston", "Just Whitney", "I'm Your Baby Tonight", "Whitney"],
    )
    prediction = strategy.answer(question)
    assert prediction.option_id == 0
    assert prediction.metadata["strategy"] == "routed_rag_council"
    assert prediction.metadata["route"] == "rag"
    assert prediction.metadata["judge_model_name"] == "e4b-judge"
    assert prediction.metadata["candidate_styles"] == ["evidence_checker", "option_eliminator"]
    assert [vote["style"] for vote in prediction.metadata["votes"]] == ["evidence_checker", "option_eliminator"]
    assert len(calls) == 1


def test_routed_rag_council_filters_unsupported_low_confidence_vote():
    first = FakeLLM(['{"option_id": 2, "confidence": 0.55, "reason": "evidence supports this"}'], model_name="checker")
    second = FakeLLM(['{"option_id": 1, "confidence": 0.0, "reason": "not mentioned in the text"}'], model_name="eliminator")
    judge = FakeLLM(["1"], model_name="judge")

    def fake_retriever(query, cfg, llm):
        return "The report mentions a consultant and a Japan research trip.", [], 0.05

    strategy = RoutedRAGCouncilStrategy(
        candidate_llms=[first, second],
        judge_llm=judge,
        retriever=fake_retriever,
        candidate_styles=["evidence_checker", "option_eliminator"],
        always_judge=True,
    )
    question = _question(
        "According to the 2026-05-15 report, which element was emphasized?",
        ["Game company collaboration", "Advertising", "Cultural consulting and research", "Random survey"],
    )
    prediction = strategy.answer(question)
    assert prediction.option_id == 2
    assert prediction.metadata["decision_method"] == "support_filter"
    assert prediction.metadata["judge_raw_text"] is None
    assert len(judge.calls) == 0


def test_routed_rag_council_can_swap_judge_and_candidates():
    class UnloadableFakeLLM(FakeLLM):
        def __init__(self, responses, model_name):
            super().__init__(responses, model_name=model_name)
            self.unload_count = 0

        def unload(self):
            self.unload_count += 1

    candidate = UnloadableFakeLLM(
        ['{"option_id": 0, "confidence": 0.8, "reason": "candidate"}'],
        "quantized-e2b",
    )
    judge = UnloadableFakeLLM(["0"], "normal-e2b-judge")
    strategy = RoutedRAGCouncilStrategy(
        candidate_llms=[candidate],
        judge_llm=judge,
        always_judge=True,
        unload_candidates_before_judge=True,
        unload_judge_before_candidates=True,
    )
    question = _question(
        "Which album was Whitney Houston's debut studio album?",
        ["Whitney Houston", "Just Whitney", "I'm Your Baby Tonight", "Whitney"],
    )
    prediction = strategy.answer(question)
    assert prediction.option_id == 0
    assert judge.unload_count == 1
    assert candidate.unload_count == 1
    assert prediction.metadata["unload_candidates_before_judge"] is True
    assert prediction.metadata["unload_judge_before_candidates"] is True


def _raise(message: str):
    raise AssertionError(message)
