from __future__ import annotations

from polimillionaire.strategies import FakeLLM, GemmaStrategy, HeuristicStrategy, RandomStrategy


def test_random_strategy_contract(sample_question):
    prediction = RandomStrategy(seed=7).answer(sample_question)
    assert prediction.option_id in sample_question.valid_option_ids()
    assert prediction.answer_text


def test_heuristic_strategy_contract(sample_question):
    prediction = HeuristicStrategy().answer(sample_question)
    assert prediction.option_id in sample_question.valid_option_ids()
    assert prediction.answer_text


def test_gemma_strategy_contract_with_fake_llm(sample_question):
    llm = FakeLLM(['{"option_id": 1, "confidence": 0.8, "reason": "Arithmetic."}'])
    prediction = GemmaStrategy(llm=llm).answer(sample_question)
    assert prediction.option_id == 1
    assert prediction.answer_text == "4"
    assert prediction.metadata["model_name"] == "fake-llm"


def test_gemma_strategy_accepts_backend_config_without_loading_model():
    strategy = GemmaStrategy(model_config={"kind": "gemma", "inference_backend": "pipeline_any_to_any"})
    assert strategy.llm.config.model_id == "google/gemma-4-E2B-it"
    assert strategy.llm.config.inference_backend == "pipeline_any_to_any"
    assert strategy.llm.is_loaded is False


def test_gemma_strategy_accepts_deterministic_config_without_loading_model():
    strategy = GemmaStrategy(model_config={"max_new_tokens": 8, "num_beams": 1, "seed": 42})
    assert strategy.llm.config.max_new_tokens == 8
    assert strategy.llm.config.num_beams == 1
    assert strategy.llm.config.seed == 42
