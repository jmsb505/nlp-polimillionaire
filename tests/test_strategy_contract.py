from __future__ import annotations

import sys
from types import ModuleType

import torch

import polimillionaire.strategies as strategies_module
from polimillionaire.strategies import (
    CouncilStrategy,
    FakeLLM,
    GemmaLLM,
    GemmaLLMConfig,
    GemmaStrategy,
    HeuristicStrategy,
    QwenLLM,
    QwenLLMConfig,
    QwenStrategy,
    RandomStrategy,
    build_council_vote_prompt,
)


def test_rag_models_are_lazy_loaded_on_import():
    assert strategies_module._RAG_RUNTIME_READY is False
    assert strategies_module._RAG_EMBED_MODEL is None
    assert strategies_module._RAG_CROSS_ENCODER is None


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


def test_qwen_strategy_contract_with_fake_llm(sample_question):
    prediction = QwenStrategy(llm=FakeLLM(["Thought.\noption_id: 1"])).answer(sample_question)
    assert prediction.option_id == 1
    assert prediction.answer_text == "4"
    assert prediction.metadata["model_name"] == "fake-llm"
    assert prediction.metadata["thinking"] is True


def test_qwen_strategy_accepts_thinking_config_without_loading_model():
    strategy = QwenStrategy(model_config={"max_new_tokens": 128, "enable_thinking": True})
    assert isinstance(strategy.config, QwenLLMConfig)
    assert strategy.llm.config.model_id == "Qwen/Qwen3.5-2B"
    assert strategy.llm.config.max_new_tokens == 128
    assert strategy.llm.config.enable_thinking is True
    assert strategy.llm.is_loaded is False


def test_quantized_model_loaders_pass_course_bitsandbytes_config(monkeypatch):
    calls = []
    fake_transformers = ModuleType("transformers")
    fake_transformers.__version__ = "5.7.0"

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, model_id):
            return cls()

    class FakeTokenizer(FakeProcessor):
        pass

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append((model_id, kwargs))
            return cls()

        def eval(self):
            return self

    fake_transformers.BitsAndBytesConfig = FakeBitsAndBytesConfig
    fake_transformers.AutoProcessor = FakeProcessor
    fake_transformers.AutoTokenizer = FakeTokenizer
    fake_transformers.AutoModelForCausalLM = FakeModel
    fake_transformers.AutoModelForImageTextToText = FakeModel
    monkeypatch.setitem(sys.modules, "bitsandbytes", ModuleType("bitsandbytes"))
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    GemmaLLM(GemmaLLMConfig(quantize_4bit=True))._load()
    QwenLLM(QwenLLMConfig(quantize_4bit=True))._load()

    assert len(calls) == 2
    for _, kwargs in calls:
        bnb = kwargs["quantization_config"].kwargs
        assert bnb["load_in_4bit"] is True
        assert bnb["bnb_4bit_use_double_quant"] is True
        assert bnb["bnb_4bit_quant_type"] == "nf4"
        assert bnb["bnb_4bit_compute_dtype"] == torch.bfloat16


def test_council_uses_stochastic_votes_then_majority(sample_question):
    llm = FakeLLM(
        [
            '{"option_id": 1, "confidence": 0.2, "reason": "Arithmetic."}',
            '{"option_id": 0, "confidence": 0.99, "reason": "Weak guess."}',
            '{"option_id": 1, "confidence": 0.2, "reason": "Arithmetic."}',
        ]
    )
    prediction = CouncilStrategy(llm=llm, num_votes=3, base_seed=7, shuffle_options=True).answer(sample_question)
    assert prediction.option_id == 1
    assert prediction.metadata["decision_method"] == "majority_vote"
    assert [call["seed"] for call in llm.calls[:3]] == [7, 8, 9]
    assert all(call["do_sample"] is True for call in llm.calls[:3])
    assert len(llm.calls) == 3
    assert len({tuple(vote["option_order"]) for vote in prediction.metadata["votes"]}) > 1


def test_council_falls_back_to_weighted_vote_when_judge_is_invalid(sample_question):
    llm = FakeLLM(
        [
            '{"option_id": 0, "confidence": 0.2, "reason": "Guess."}',
            '{"option_id": 1, "confidence": 0.7, "reason": "Arithmetic."}',
            "I cannot choose.",
        ]
    )
    prediction = CouncilStrategy(llm=llm, num_votes=2).answer(sample_question)
    assert prediction.option_id == 1
    assert prediction.metadata["decision_method"] == "weighted_vote"
    assert prediction.metadata["fallback"] is False
    assert prediction.metadata["judge_rejected"] is True


def test_council_does_not_let_judge_override_unanimous_votes(sample_question):
    llm = FakeLLM(
        [
            '{"option_id": 1, "confidence": 0.8, "reason": "Arithmetic."}',
            '{"option_id": 1, "confidence": 0.8, "reason": "Arithmetic."}',
            '{"option_id": 1, "confidence": 0.8, "reason": "Arithmetic."}',
            "0",
        ]
    )
    prediction = CouncilStrategy(llm=llm).answer(sample_question)
    assert prediction.option_id == 1
    assert prediction.metadata["decision_method"] == "unanimous_vote"
    assert prediction.metadata["judge_raw_text"] is None
    assert len(llm.calls) == 3


def test_council_accepts_novel_judge_option_by_default(sample_question):
    llm = FakeLLM(
        [
            '{"option_id": 1, "confidence": 0.9, "reason": "Arithmetic."}',
            '{"option_id": 0, "confidence": 0.2, "reason": "Guess."}',
            "2",
        ]
    )
    prediction = CouncilStrategy(llm=llm, num_votes=2).answer(sample_question)
    assert prediction.option_id == 2
    assert prediction.metadata["decision_method"] == "judge"
    assert prediction.metadata["judge_novel_choice"] is True


def test_council_rejects_novel_judge_option_when_candidate_only(sample_question):
    llm = FakeLLM(
        [
            '{"option_id": 1, "confidence": 0.9, "reason": "Arithmetic."}',
            '{"option_id": 0, "confidence": 0.2, "reason": "Guess."}',
            "2",
        ]
    )
    prediction = CouncilStrategy(llm=llm, num_votes=2, judge_scope="candidate_only").answer(sample_question)
    assert prediction.option_id == 1
    assert prediction.metadata["decision_method"] == "weighted_vote"
    assert prediction.metadata["judge_rejected"] is True
    assert prediction.metadata["judge_option_id"] == 2


def test_council_can_use_primary_candidate_instead_of_cross_model_confidence(sample_question):
    primary = FakeLLM(['{"option_id": 1, "confidence": 0.2, "reason": "Arithmetic."}'], model_name="primary")
    secondary = FakeLLM(['{"option_id": 0, "confidence": 0.99, "reason": "Guess."}'], model_name="secondary")
    judge = FakeLLM(["2"], model_name="judge")
    prediction = CouncilStrategy(
        candidate_llms=[primary, secondary],
        judge_llm=judge,
        judge_scope="candidate_only",
        rejected_judge_fallback="primary_candidate",
    ).answer(sample_question)
    assert prediction.option_id == 1
    assert prediction.metadata["decision_method"] == "primary_candidate"
    assert prediction.metadata["judge_rejected"] is True


def test_council_accepts_different_candidate_models(sample_question):
    first = FakeLLM(['{"option_id": 0, "confidence": 0.3, "reason": "Guess."}'], model_name="first")
    second = FakeLLM(['{"option_id": 1, "confidence": 0.9, "reason": "Arithmetic."}'], model_name="second")
    judge = FakeLLM(["1"], model_name="judge")
    prediction = CouncilStrategy(candidate_llms=[first, second], judge_llm=judge).answer(sample_question)
    assert prediction.option_id == 1
    assert [vote["model_name"] for vote in prediction.metadata["votes"]] == ["first", "second"]
    assert prediction.metadata["judge_model_name"] == "judge"


def test_council_prompt_does_not_anchor_a_specific_option(sample_question):
    prompt = build_council_vote_prompt(sample_question)
    assert '"option_id": 0' not in prompt
