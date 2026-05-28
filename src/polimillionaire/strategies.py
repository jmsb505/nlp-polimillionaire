from __future__ import annotations

"""Strategies, local model loading, prompting, and output parsing."""

import json
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Any, Protocol
from langchain_core.tools import tool, render_text_description
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import Dict, Any, Optional

import asyncio
import hashlib
import time
import logging

import httpx
import nest_asyncio
import numpy as np
import trafilatura
from ddgs import DDGS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder, SentenceTransformer

from packaging.version import Version

from polimillionaire.types import AnswerPrediction, Question


try:
    from newspaper import Article as NewspaperArticle
    HAS_NEWSPAPER = True
except ImportError:
    HAS_NEWSPAPER = False

nest_asyncio.apply()


class BaseStrategy(ABC):
    name = "base"

    @abstractmethod
    def answer(self, question: Question) -> AnswerPrediction:
        raise NotImplementedError


class RandomStrategy(BaseStrategy):
    name = "random"

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def answer(self, question: Question) -> AnswerPrediction:
        option = self._rng.choice(question.options)
        return AnswerPrediction(
            option_id=option.id,
            answer_text=option.text,
            confidence=1.0 / max(1, len(question.options)),
            reasoning="Random option selected.",
            metadata={"strategy": self.name},
        )


class HeuristicStrategy(BaseStrategy):
    name = "heuristic"

    def answer(self, question: Question) -> AnswerPrediction:
        question_terms = set(_words(question.text))
        option = max(
            question.options,
            key=lambda item: len(question_terms & set(_words(item.text))),
        )
        return AnswerPrediction(
            option_id=option.id,
            answer_text=option.text,
            confidence=0.35,
            reasoning="Simple word-overlap heuristic selected the option.",
            metadata={"strategy": self.name},
        )


class LocalLLM(Protocol):
    model_name: str

    def generate(self, prompt: str, **kwargs: object) -> str:
        ...


class FakeLLM:
    def __init__(self, responses: list[str] | None = None, model_name: str = "fake-llm"):
        self.responses = list(responses or ['{"option_id": 0, "confidence": 0.5, "reason": "fake"}'])
        self.model_name = model_name
        self.prompts: list[str] = []
        self.calls: list[dict[str, object]] = []

    def generate(self, prompt: str, **kwargs: object) -> str:
        self.prompts.append(prompt)
        self.calls.append(dict(kwargs))
        if len(self.responses) == 1:
            return self.responses[0]
        return self.responses.pop(0)


@dataclass
class GemmaLLMConfig:
    model_id: str = "google/gemma-4-E2B-it"
    inference_backend: str = "auto_model"
    device_map: str = "auto"
    dtype: str = "auto"
    max_new_tokens: int = 8
    temperature: float = 0.0
    do_sample: bool = False
    num_beams: int = 1
    seed: int | None = 42
    generation_max_time_seconds: float | None = 18.0
    timeout_seconds: float | None = 25.0
    quantize_4bit: bool = False


@dataclass
class QwenLLMConfig:
    model_id: str = "Qwen/Qwen3.5-2B"
    device_map: str = "auto"
    dtype: str = "auto"
    max_new_tokens: int = 256
    temperature: float = 1.0
    do_sample: bool = True
    top_p: float = 0.95
    top_k: int = 20
    seed: int | None = 42
    enable_thinking: bool = True
    generation_max_time_seconds: float | None = None
    timeout_seconds: float | None = 25.0
    quantize_4bit: bool = False


class GemmaLLM:
    def __init__(self, config: GemmaLLMConfig | None = None, **overrides: Any):
        self.config = config or GemmaLLMConfig()
        for key, value in overrides.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self.model_name = self.config.model_id
        self._model: Any = None
        self._processor: Any = None
        self._tokenizer: Any = None
        self._pipeline: Any = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None or self._pipeline is not None

    @property
    def device_summary(self) -> str:
        if self._model is not None:
            device_map = getattr(self._model, "hf_device_map", None)
            if device_map:
                return str(device_map)
            try:
                return str(next(self._model.parameters()).device)
            except Exception:
                return "unknown"
        if self._pipeline is not None:
            return str(getattr(self._pipeline, "device", "pipeline"))
        return "not_loaded"

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self._load()
        seed = kwargs.pop("seed", self.config.seed)
        self._seed(seed)
        generation_kwargs = self._generation_kwargs(kwargs)
        if self.config.inference_backend == "auto_model":
            text = self._generate_auto_model(prompt, generation_kwargs)
        elif self.config.inference_backend == "pipeline_any_to_any":
            text = self._generate_pipeline(prompt, generation_kwargs)
        else:
            raise ValueError(f"Unsupported inference_backend: {self.config.inference_backend}")
        return text.strip()
    
    def invoke(self, input_data: Any, config: Any = None) -> Any:
        from langchain_core.outputs import ChatResult, ChatGeneration
        from langchain_core.messages import BaseMessage
        
    
        if hasattr(input_data, "to_string"):
            prompt_text = input_data.to_string()
        elif isinstance(input_data, list) and len(input_data) > 0:
            prompt_text = getattr(input_data[-1], "content", str(input_data))
        else:
            prompt_text = str(input_data)
            
        generated_text = self.generate(prompt_text)
        
    
        class CustomContent:
            def __init__(self, text):
                self.content = [{'text': text}]
                self.text_content = text
            @property
            def str_output(self):
                return self.text_content
            def __str__(self):
                return self.text_content
                g
        return CustomContent(generated_text)

    def _load(self) -> None:
        if self.is_loaded:
            return
        if self.config.inference_backend == "auto_model":
            self._load_auto_model()
        elif self.config.inference_backend == "pipeline_any_to_any":
            self._load_pipeline()
        else:
            raise ValueError(f"Unsupported inference_backend: {self.config.inference_backend}")

    def _load_auto_model(self) -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
        import transformers

        _require_supported_transformers(transformers.__version__)
        try:
            self._processor = AutoProcessor.from_pretrained(self.config.model_id)
        except Exception:
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_id, extra_special_tokens={})
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            device_map=self.config.device_map,
            dtype=self.config.dtype,
            **_quantization_kwargs(self.config.quantize_4bit),
        )
        self._model.eval()

    def _load_pipeline(self) -> None:
        from transformers import pipeline

        kwargs: dict[str, Any] = {
            "task": "any-to-any",
            "model": self.config.model_id,
            "device_map": self.config.device_map,
            "dtype": self.config.dtype,
        }
        if self.config.quantize_4bit:
            kwargs["model_kwargs"] = _quantization_kwargs(True)
        self._pipeline = pipeline(**kwargs)

    def _generation_kwargs(self, overrides: dict[str, Any]) -> dict[str, Any]:
        kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "do_sample": self.config.do_sample,
            "num_beams": self.config.num_beams,
        }
        if self.config.generation_max_time_seconds is not None:
            kwargs["max_time"] = self.config.generation_max_time_seconds
        kwargs.update({key: value for key, value in overrides.items() if value is not None})
        if not kwargs["do_sample"]:
            kwargs.pop("temperature", None)
        return kwargs

    def _seed(self, seed: int | None) -> None:
        if seed is None:
            return
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

    def _generate_auto_model(self, prompt: str, generation_kwargs: dict[str, Any]) -> str:
        import torch

        processor = self._processor or self._tokenizer
        inputs = _tokenize_prompt(processor, prompt)
        try:
            device = next(self._model.parameters()).device
            inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        except Exception:
            pass
        input_length = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, **generation_kwargs)
        generated_ids = output_ids[0][input_length:]
        return processor.decode(generated_ids, skip_special_tokens=True)

    def _generate_pipeline(self, prompt: str, generation_kwargs: dict[str, Any]) -> str:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        result = self._pipeline(messages, return_full_text=False, generate_kwargs=generation_kwargs)
        if isinstance(result, list) and result:
            item = result[0]
            if isinstance(item, dict):
                return str(item.get("generated_text", ""))
            return str(item)
        return str(result)


class QwenLLM:
    def __init__(self, config: QwenLLMConfig | None = None, **overrides: Any):
        self.config = config or QwenLLMConfig()
        for key, value in overrides.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self.model_name = self.config.model_id
        self._model: Any = None
        self._processor: Any = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device_summary(self) -> str:
        if self._model is None:
            return "not_loaded"
        device_map = getattr(self._model, "hf_device_map", None)
        if device_map:
            return str(device_map)
        try:
            return str(next(self._model.parameters()).device)
        except Exception:
            return "unknown"

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self._load()
        seed = kwargs.pop("seed", self.config.seed)
        self._seed(seed)
        generation_kwargs = self._generation_kwargs(kwargs)
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=self.config.enable_thinking,
        )
        try:
            device = next(self._model.parameters()).device
            inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        except Exception:
            pass

        import torch

        input_length = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, **generation_kwargs)
        generated_ids = output_ids[0][input_length:]
        return self._processor.decode(generated_ids, skip_special_tokens=True).strip()
    
    def invoke(self, input_data: Any, config: Any = None) -> Any:
    
        from langchain_core.outputs import ChatResult, ChatGeneration
        from langchain_core.messages import BaseMessage
        
    
        if hasattr(input_data, "to_string"):
            prompt_text = input_data.to_string()
        elif isinstance(input_data, list) and len(input_data) > 0:
            prompt_text = getattr(input_data[-1], "content", str(input_data))
        else:
            prompt_text = str(input_data)
            
        generated_text = self.generate(prompt_text)
        
        class CustomContent:
            def __init__(self, text):
                self.content = [{'text': text}]
                self.text_content = text
            @property
            def str_output(self):
                return self.text_content
            def __str__(self):
                return self.text_content
                g
        return CustomContent(generated_text)

    def _load(self) -> None:
        if self.is_loaded:
            return
        from transformers import AutoModelForImageTextToText, AutoProcessor
        import transformers

        _require_supported_transformers(transformers.__version__)
        self._processor = AutoProcessor.from_pretrained(self.config.model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.config.model_id,
            device_map=self.config.device_map,
            dtype=self.config.dtype,
            **_quantization_kwargs(self.config.quantize_4bit),
        )
        self._model.eval()

    def _generation_kwargs(self, overrides: dict[str, Any]) -> dict[str, Any]:
        kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
        }
        if self.config.do_sample:
            kwargs.update(
                {
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "top_k": self.config.top_k,
                }
            )
        if self.config.generation_max_time_seconds is not None:
            kwargs["max_time"] = self.config.generation_max_time_seconds
        kwargs.update({key: value for key, value in overrides.items() if value is not None})
        return kwargs

    def _seed(self, seed: int | None) -> None:
        if seed is None:
            return
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass


class GemmaStrategy(BaseStrategy):
    name = "gemma"

    def __init__(self, llm: LocalLLM | None = None, model_config: GemmaLLMConfig | dict[str, Any] | None = None):
        if llm is None:
            if isinstance(model_config, dict):
                allowed = {field.name for field in fields(GemmaLLMConfig)}
                model_config = GemmaLLMConfig(**{key: value for key, value in model_config.items() if key in allowed})
            llm = GemmaLLM(model_config)
        self.llm = llm

    def answer(self, question: Question) -> AnswerPrediction:
        raw_text = self.llm.generate(build_prompt(question))
        prediction = parse_answer_prediction(raw_text, question, strategy_name=self.name)
        prediction.metadata["strategy"] = self.name
        prediction.metadata["model_name"] = getattr(self.llm, "model_name", "unknown")
        prediction.metadata["device"] = getattr(self.llm, "device_summary", "unknown")
        return prediction


class QwenStrategy(BaseStrategy):
    name = "qwen3.5_thinking"

    def __init__(self, llm: LocalLLM | None = None, model_config: QwenLLMConfig | dict[str, Any] | None = None):
        if isinstance(model_config, dict):
            allowed = {field.name for field in fields(QwenLLMConfig)}
            model_config = QwenLLMConfig(**{key: value for key, value in model_config.items() if key in allowed})
        self.config = model_config or QwenLLMConfig()
        self.llm = llm or QwenLLM(self.config)

    def answer(self, question: Question) -> AnswerPrediction:
        raw_text = self.llm.generate(build_qwen_prompt(question))
        prediction = parse_answer_prediction(raw_text, question, strategy_name=self.name)
        prediction.metadata["strategy"] = self.name
        prediction.metadata["model_name"] = getattr(self.llm, "model_name", "unknown")
        prediction.metadata["device"] = getattr(self.llm, "device_summary", "unknown")
        prediction.metadata["thinking"] = self.config.enable_thinking
        return prediction


class CouncilStrategy(BaseStrategy):
    name = "council"

    def __init__(
        self,
        llm: LocalLLM | None = None,
        judge_llm: LocalLLM | None = None,
        candidate_llms: list[LocalLLM] | None = None,
        num_votes: int = 3,
        base_seed: int = 100,
        candidate_temperature: float = 0.8,
        candidate_top_p: float = 0.9,
        candidate_max_new_tokens: int = 48,
        judge_max_new_tokens: int = 8,
        max_time_per_call: float | None = None,
        shuffle_options: bool = False,
        judge_scope: str = "any_option",
        rejected_judge_fallback: str = "confidence_weighted",
    ):
        if candidate_llms:
            self.candidate_llms = list(candidate_llms)
            self.llm = llm or self.candidate_llms[0]
        elif llm is not None:
            if num_votes < 1:
                raise ValueError("num_votes must be at least 1")
            self.llm = llm
            self.candidate_llms = [llm] * num_votes
        else:
            raise ValueError("Provide llm or candidate_llms")
        if not self.candidate_llms:
            raise ValueError("num_votes must be at least 1")
        self.judge_llm = judge_llm or self.llm
        self.num_votes = len(self.candidate_llms)
        self.base_seed = base_seed
        self.candidate_temperature = candidate_temperature
        self.candidate_top_p = candidate_top_p
        self.candidate_max_new_tokens = candidate_max_new_tokens
        self.judge_max_new_tokens = judge_max_new_tokens
        self.max_time_per_call = max_time_per_call
        self.shuffle_options = shuffle_options
        if judge_scope not in {"candidate_only", "any_option"}:
            raise ValueError("judge_scope must be 'candidate_only' or 'any_option'")
        self.judge_scope = judge_scope
        if rejected_judge_fallback not in {"confidence_weighted", "primary_candidate"}:
            raise ValueError("rejected_judge_fallback must be 'confidence_weighted' or 'primary_candidate'")
        self.rejected_judge_fallback = rejected_judge_fallback

    def answer(self, question: Question) -> AnswerPrediction:
        votes: list[AnswerPrediction] = []
        for vote_index, candidate_llm in enumerate(self.candidate_llms):
            option_order = self._option_order(question, vote_index)
            raw_text = candidate_llm.generate(
                build_council_vote_prompt(question, option_order),
                max_new_tokens=self.candidate_max_new_tokens,
                do_sample=True,
                temperature=self.candidate_temperature,
                top_p=self.candidate_top_p,
                seed=self.base_seed + vote_index,
                **self._time_kwargs(),
            )
            vote = parse_answer_prediction(raw_text, question, strategy_name="council_vote")
            if not vote.metadata["fallback"]:
                vote.metadata["model_name"] = getattr(candidate_llm, "model_name", "unknown")
                vote.metadata["sample_seed"] = self.base_seed + vote_index
                vote.metadata["option_order"] = [option.id for option in option_order]
                votes.append(vote)

        if not votes:
            return _council_fallback(question, "No valid candidate votes.", [])

        supported_options = {vote.option_id for vote in votes}
        majority_option = _majority_option(votes)
        if majority_option is not None:
            result = _selected_vote_prediction(question, votes, majority_option)
            method = "unanimous_vote" if len(supported_options) == 1 else "majority_vote"
            result.metadata.update(self._metadata(votes, None, method))
            return result

        judge_raw_text = self.judge_llm.generate(
            build_judge_prompt(question, votes, judge_scope=self.judge_scope),
            max_new_tokens=self.judge_max_new_tokens,
            do_sample=False,
            seed=self.base_seed + self.num_votes,
            **self._time_kwargs(),
        )
        judged = parse_answer_prediction(judge_raw_text, question, strategy_name=self.name)
        if not judged.metadata["fallback"] and (
            self.judge_scope == "any_option" or judged.option_id in supported_options
        ):
            judged.metadata.update(self._metadata(votes, judge_raw_text, "judge"))
            judged.metadata["judge_novel_choice"] = judged.option_id not in supported_options
            return judged

        if self.rejected_judge_fallback == "primary_candidate":
            result = _selected_vote_prediction(question, votes, votes[0].option_id)
            result.reasoning = "Primary candidate selected after judge returned an unsupported answer."
            method = "primary_candidate"
        else:
            result = _weighted_vote(question, votes)
            method = "weighted_vote"
        result.metadata.update(self._metadata(votes, judge_raw_text, method))
        result.metadata["judge_rejected"] = True
        result.metadata["judge_option_id"] = None if judged.metadata["fallback"] else judged.option_id
        return result

    def _time_kwargs(self) -> dict[str, float]:
        if self.max_time_per_call is None:
            return {}
        return {"max_time": self.max_time_per_call}

    def _option_order(self, question: Question, vote_index: int) -> list[Any]:
        options = list(question.options)
        if self.shuffle_options:
            random.Random(self.base_seed + vote_index).shuffle(options)
        return options

    def _metadata(self, votes: list[AnswerPrediction], judge_raw_text: str | None, method: str) -> dict[str, Any]:
        return {
            "strategy": self.name,
            "model_name": getattr(self.llm, "model_name", "unknown"),
            "judge_model_name": getattr(self.judge_llm, "model_name", "unknown"),
            "device": getattr(self.llm, "device_summary", "unknown"),
            "decision_method": method,
            "judge_scope": self.judge_scope,
            "rejected_judge_fallback": self.rejected_judge_fallback,
            "candidate_devices": [
                getattr(candidate_llm, "device_summary", "unknown") for candidate_llm in self.candidate_llms
            ],
            "judge_device": getattr(self.judge_llm, "device_summary", "unknown"),
            "votes": [
                {
                    "option_id": vote.option_id,
                    "confidence": vote.confidence,
                    "reasoning": vote.reasoning,
                    "raw_text": str(vote.metadata.get("raw_text", ""))[:300],
                    "model_name": vote.metadata.get("model_name"),
                    "sample_seed": vote.metadata.get("sample_seed"),
                    "option_order": vote.metadata.get("option_order"),
                }
                for vote in votes
            ],
            "judge_raw_text": judge_raw_text,
        }


def build_prompt(question: Question) -> str:
    options = "\n".join(f"{option.id}) {option.text}" for option in question.options)
    return "\n\n".join(
        [
            "Pick the best answer. Return only the option id number.",
            f"Q: {question.text}",
            options,
            "Answer:",
        ]
    )


def build_qwen_prompt(question: Question) -> str:
    options = "\n".join(f"{option.id}) {option.text}" for option in question.options)
    return "\n\n".join(
        [
            "Choose the best answer to this multiple-choice question.",
            f"Q: {question.text}",
            options,
            "After thinking, finish with exactly: option_id: <number>",
        ]
    )


def build_council_vote_prompt(question: Question, option_order: list[Any] | None = None) -> str:
    options = "\n".join(f"{option.id}) {option.text}" for option in (option_order or question.options))
    return "\n\n".join(
        [
            "Pick the best answer. Check words such as NOT and EXCEPT.",
            "Return JSON with keys option_id, confidence, and reason. Use the listed numeric ID. Keep the reason short.",
            f"Q: {question.text}",
            options,
        ]
    )


def build_judge_prompt(
    question: Question,
    votes: list[AnswerPrediction],
    judge_scope: str = "candidate_only",
) -> str:
    options = "\n".join(f"{option.id}) {option.text}" for option in question.options)
    summaries = "\n".join(
        f"vote {index + 1}: option={vote.option_id}, confidence={vote.confidence}, reason={vote.reasoning or ''}"
        for index, vote in enumerate(votes)
    )
    supported = ", ".join(str(option_id) for option_id in sorted({vote.option_id for vote in votes}))
    scope_text = (
        f"Choose only one proposed option id: {supported}."
        if judge_scope == "candidate_only"
        else "You may choose any listed option."
    )
    return "\n\n".join(
        [
            f"Choose the final answer from the candidates. {scope_text} Return only the option id number.",
            f"Q: {question.text}",
            options,
            summaries,
            "Answer:",
        ]
    )


def parse_answer_prediction(raw_text: str, question: Question, strategy_name: str = "llm") -> AnswerPrediction:
    payload = _parse_payload(raw_text)
    option_id = _coerce_int(payload.get("option_id"))
    if option_id not in question.valid_option_ids():
        option_id = _match_option_text(raw_text, question)

    fallback = option_id not in question.valid_option_ids()
    if fallback:
        option_id = question.first_option().id

    option = question.require_option(option_id)
    reason = payload.get("reason")
    if reason is not None:
        reason = str(reason).strip()[:300]

    return AnswerPrediction(
        option_id=option.id,
        answer_text=option.text,
        confidence=_clamp_confidence(payload.get("confidence")),
        reasoning=reason,
        metadata={
            "strategy": strategy_name,
            "raw_text": raw_text,
            "fallback": fallback,
            "parsed_payload": payload,
        },
    )


def _majority_option(votes: list[AnswerPrediction]) -> int | None:
    counts: dict[int, int] = {}
    for vote in votes:
        counts[vote.option_id] = counts.get(vote.option_id, 0) + 1
    option_id, count = max(counts.items(), key=lambda item: (item[1], -item[0]))
    return option_id if count > len(votes) / 2 else None


def _selected_vote_prediction(
    question: Question,
    votes: list[AnswerPrediction],
    option_id: int,
) -> AnswerPrediction:
    supporters = [vote for vote in votes if vote.option_id == option_id]
    confidence_values = [vote.confidence for vote in supporters if vote.confidence is not None]
    option = question.require_option(option_id)
    return AnswerPrediction(
        option_id=option.id,
        answer_text=option.text,
        confidence=sum(confidence_values) / len(confidence_values) if confidence_values else None,
        reasoning="Council majority selected this answer.",
        metadata={"strategy": "council", "fallback": False},
    )


def _weighted_vote(question: Question, votes: list[AnswerPrediction]) -> AnswerPrediction:
    scores: dict[int, float] = {}
    counts: dict[int, int] = {}
    for vote in votes:
        weight = vote.confidence if vote.confidence is not None else 0.5
        scores[vote.option_id] = scores.get(vote.option_id, 0.0) + weight
        counts[vote.option_id] = counts.get(vote.option_id, 0) + 1
    option_id = max(scores, key=lambda item: (counts[item], scores[item], -item))
    option = question.require_option(option_id)
    total_score = sum(scores.values())
    return AnswerPrediction(
        option_id=option.id,
        answer_text=option.text,
        confidence=scores[option_id] / total_score if total_score else None,
        reasoning="Confidence-weighted council vote.",
        metadata={"strategy": "council", "fallback": False},
    )


def _council_fallback(question: Question, reason: str, votes: list[AnswerPrediction]) -> AnswerPrediction:
    option = question.first_option()
    return AnswerPrediction(
        option_id=option.id,
        answer_text=option.text,
        confidence=0.0,
        reasoning=reason,
        metadata={
            "strategy": "council",
            "fallback": True,
            "decision_method": "fallback",
            "votes": votes,
        },
    )


def _parse_payload(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    for candidate in [text, *_json_blocks(text)]:
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    payload: dict[str, Any] = {}
    bare_option_match = re.fullmatch(r"\s*(-?\d+)\s*[\.\)]?\s*", text)
    if bare_option_match:
        payload["option_id"] = bare_option_match.group(1)
        return payload
    option_match = re.search(r"(?:option_id|option|answer)\D+(-?\d+)", text, flags=re.IGNORECASE)
    if option_match:
        payload["option_id"] = option_match.group(1)
    confidence_match = re.search(r"confidence\D+([01](?:\.\d+)?)", text, flags=re.IGNORECASE)
    if confidence_match:
        payload["confidence"] = confidence_match.group(1)
    reason_match = re.search(
        r'["\']?(?:reason|justification)["\']?\s*[:=]\s*["\']?([^"\'\n\r}]*)',
        text,
        flags=re.IGNORECASE,
    )
    if reason_match:
        reason = reason_match.group(1).strip(" ,")
        if reason:
            payload["reason"] = reason
    return payload


def _json_blocks(text: str) -> list[str]:
    return re.findall(r"\{.*?\}", text, flags=re.DOTALL)


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clamp_confidence(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def _match_option_text(raw_text: str, question: Question) -> int | None:
    lowered = raw_text.lower()
    for option in question.options:
        if option.text.lower() in lowered:
            return option.id
    return None


def _tokenize_prompt(processor: Any, prompt: str) -> dict[str, Any]:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    if hasattr(processor, "apply_chat_template"):
        try:
            return processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception:
            pass
    return processor(_text_chat_prompt(prompt), return_tensors="pt")


def _text_chat_prompt(prompt: str) -> str:
    return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"


def _require_supported_transformers(version: str) -> None:
    if Version(version) < Version("5.7.0"):
        raise RuntimeError(
            "The local Gemma/Qwen models require transformers>=5.7.0 in this notebook kernel. "
            f"Current transformers version is {version}. Run the dependency install cell "
            "with INSTALL_DEPS=True, then restart the kernel and rerun setup."
        )


def _quantization_kwargs(quantize_4bit: bool) -> dict[str, Any]:
    if not quantize_4bit:
        return {}
    try:
        import bitsandbytes as _bitsandbytes
        del _bitsandbytes
    except ImportError as exc:
        raise RuntimeError(
            "4-bit quantization needs bitsandbytes in this notebook kernel. "
            "Run `%pip install -U bitsandbytes`, restart the kernel, and rerun setup."
        ) from exc

    import torch
    from transformers import BitsAndBytesConfig

    return {
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    }


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


@tool
def calculator_tool(expression: str) -> float:
    """
    Computes mathematical operations, this tool does.
    An argument it takes, a basic Python math string like '2 + 2' or '1.95 ** 10' it must be.
    """
    clean_expr = re.sub(r'[^0-9\+\-\*\/\(\)\.\s]', '', expression)
    return float(eval(clean_expr, {"__builtins__": None}, {}))

@tool
def wikipedia_tool(search_term: str) -> str:
    """
    Historical facts, biographies or general knowledge, this tool finds.
    A single entity or concept as search_term string, it requires.
    """
    try:
        import urllib.request
        import urllib.parse
        from bs4 import BeautifulSoup
        
        query = urllib.parse.quote(search_term.strip())
        url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json"
        req = urllib.request.Request(url, headers={'User-Agent': 'PoliMillionaireAgent/1.0'})
        
        with urllib.request.urlopen(req, timeout=4) as response:
            data = json.loads(response.read().decode())
            
        results = data.get("query", {}).get("search", [])
        if not results:
            return "No information found in Wikipedia."
            
        snippet = BeautifulSoup(results[0].get("snippet", ""), "html.parser").get_text()
        return f"Fact context found: {snippet}"
    except Exception as e:
        return f"Error during search: {str(e)}"



class LangChainAgenticStrategy(BaseStrategy):
    name = "langchain_agent"

    def __init__(self, raw_llm: LocalLLM, fallback_strategy: BaseStrategy | None = None):
        # Wraps your existing LocalLLM (Gemma/Qwen) into a LangChain compatible interface
        self.raw_llm = raw_llm
        self.fallback = fallback_strategy or HeuristicStrategy()
        self.tools = [calculator_tool, wikipedia_tool]
        self._setup_agent_prompts()

    def _setup_agent_prompts(self):
        rendered_tools = render_text_description(self.tools)
        
        system_routing = f"""
You are an advanced orchestrator for the quiz game 'Who wants to be a PoliMillionaire?'.
Your job is to decide if you need an external tool to answer the question accurately.

Here are the names and descriptions for each tool available:
{rendered_tools}

Given the question, you must return a JSON blob with 'name' and 'arguments' keys.
If you DO NOT need a tool to answer the question, you MUST set 'name': "none" and 'arguments': {{"query": ""}}.
If you need a tool, choose one from the list.

The response format MUST be exactly a valid JSON block like this:
{{"name": "tool_name", "arguments": {{"expression": "2+2"}}}}
"""
        self.routing_prompt = ChatPromptTemplate.from_messages([
            ("system", system_routing),
            ("user", "Question: {{input}}\nOptions: {{options_text}}")
        ], template_format="jinja2")

        system_answering = """
You are a brilliant contestant playing 'Who wants to be a PoliMillionaire?'.
Additional verified context from your tools: {{context}}

Analyze the question and select the single best option ID.
You MUST return your response ONLY as a JSON blob matching this structure:
{"option_id": 0, "confidence": 0.95, "reasoning": "Your step-by-step logic here"}
"""
        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", system_answering),
            ("user", "Question: {{input}}\nOptions:\n{{options_text}}")
        ], template_format="jinja2")

    def _invoke_tool(self, tool_name: str, tool_args: dict) -> str:
        if tool_name == "none" or not tool_name:
            return "No additional context needed."
            
        tool_map = {t.name: t for t in self.tools}
        if tool_name in tool_map:
            try:
                result = tool_map[tool_name].invoke(tool_args)
                return str(result)
            except Exception as e:
                return f"Tool execution failed: {str(e)}"
        return "Unknown tool requested."

    def answer(self, question: Question) -> AnswerPrediction:
        try:
            options_str = "\n".join([f"ID {o.id}: {o.text}" for o in question.options])
            
            # --- PHASE 1: TOOL ROUTING ---
            # Use LangChain pipe syntax to call the local wrapper model
            routing_chain = self.routing_prompt | self.raw_llm | JsonOutputParser()
            
            routing_res = routing_chain.invoke({
                "input": question.text,
                "options_text": options_str
            })
            
            tool_name = routing_res.get("name", "none")
            tool_arguments = routing_res.get("arguments", {})
            
            # --- PHASE 2: TOOL EXECUTION ---
            context_data = self._invoke_tool(tool_name, tool_arguments)
            
            # --- PHASE 3: FINAL ANSWER GENERATION ---
            answer_chain = self.answer_prompt | self.raw_llm | JsonOutputParser()
            final_res = answer_chain.invoke({
                "context": context_data,
                "input": question.text,
                "options_text": options_str
            })
            
            selected_id = int(final_res["option_id"])
            chosen_option = next((o for o in question.options if o.id == selected_id), question.options[0])
            
            return AnswerPrediction(
                option_id=chosen_option.id,
                answer_text=chosen_option.text,
                confidence=float(final_res.get("confidence", 0.7)),
                reasoning=str(final_res.get("reasoning", "Decided using LangChain Agent pipeline.")),
                metadata={
                    "strategy": self.name,
                    "used_tool": tool_name,
                    "tool_arguments": str(tool_arguments),
                    "tool_output": context_data
                }
            )
            
        except Exception as e:
            print(f"[LangChainAgent] Error captured, deploying fallback: {str(e)}")
            fallback_pred = self.fallback.answer(question)
            fallback_pred.metadata["fallback"] = True
            fallback_pred.metadata["fallback_reason"] = str(e)
            return fallback_pred



# ==============================================================================
# RAG STRATEGY
# ==============================================================================

# ===============================================================================
# MODULE-LEVEL SINGLETONS  — loaded once at import, shared across all instances
# ===============================================================================

_rag_log = logging.getLogger(f"{__name__}.rag")

# Splitter is fixed at module level; chunk_size/overlap are not runtime-tunable
# without reloading the module. RAGConfig exposes them for documentation only.
_RAG_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " ", ""],
)

_rag_log.info("RAG: loading dense encoder …")
_RAG_EMBED_MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5")

_rag_log.info("RAG: loading cross-encoder …")
_RAG_CROSS_ENCODER = CrossEncoder("BAAI/bge-reranker-base")


# ===============================================================================
# CONFIG
# ===============================================================================


@dataclass
class RAGConfig:
    # Search
    max_ddg_results: int = 7
    timeout_ddg: int = 10
    # Extra query variants generated on top of the original query (0 = disabled)
    num_extra_queries: int = 1

    # Fetching
    fetch_timeout: float = 8.0
    max_concurrent_fetches: int = 6
    max_text_chars: int = 20_000
    total_fetch_budget: float = 9.0      # hard wall-clock cap for the fetch phase

    # Retrieval
    bm25_top_k: int = 10
    dense_top_k: int = 10
    rrf_k: int = 60                      # standard RRF constant
    diversity_max_per_url: int = 2

    # Reranking
    # RAGStrategy overrides this when calling llm.generate(), 
    # so the LLM config default no longer matters.
    answer_max_new_tokens: int = 128

    # Reranking
    final_top_k: int = 5


# ===============================================================================
# PROMPT
# ===============================================================================

_RAG_EXPANSION_PROMPT = """\
Generate {n} alternative search queries for the question below.
Rules: each must be semantically distinct, under 12 words, no numbering.
Output ONLY the queries, one per line.

Question: {query}"""

# Use <<<EVIDENCE>>>, <<<QUESTION>>>, <<<OPTIONS>>> as delimiters
# instead of {evidence}/{question}/{options} with str.format(), which raises
# KeyError if any option text or evidence chunk contains literal braces { }.
_RAG_ANSWER_PROMPT = """\
Answer this multiple-choice question using the evidence below.
Return ONLY a JSON object with keys: option_id, confidence, reason.

Evidence:
<<<EVIDENCE>>>

Question: <<<QUESTION>>>
<<<OPTIONS>>>"""


def build_rag_prompt(question: Question, evidence: str) -> str:
    options = "\n".join(f"{o.id}) {o.text}" for o in question.options)
    return (
        _RAG_ANSWER_PROMPT
        .replace("<<<EVIDENCE>>>", evidence)
        .replace("<<<QUESTION>>>", question.text)
        .replace("<<<OPTIONS>>>", options)
    )


# ===============================================================================
# QUERY EXPANSION  — uses the same LocalLLM passed to the strategy
# ===============================================================================


def _expand_query_rag(query: str, cfg: RAGConfig, llm: LocalLLM) -> list[str]:
    if cfg.num_extra_queries == 0:
        return [query]
    # Same brace-safety fix for the expansion prompt
    prompt = (
        _RAG_EXPANSION_PROMPT
        .replace("{n}", str(cfg.num_extra_queries))
        .replace("{query}", query)
    )
    try:
        raw = llm.generate(prompt, max_new_tokens=80, do_sample=False)
        variants = [
            line.strip()
            for line in raw.strip().splitlines()
            if line.strip() and line.strip().lower() != query.lower()
        ][: cfg.num_extra_queries]
    except Exception as exc:
        _rag_log.warning("RAG query expansion failed: %s", exc)
        variants = []
    all_queries = [query] + variants
    _rag_log.info("RAG expanded queries: %s", all_queries)
    return all_queries


# ===============================================================================
# SEARCH
# ===============================================================================


def _rag_search_all(queries: list[str], cfg: RAGConfig) -> list[dict]:
    seen: set[str] = set()
    merged: list[dict] = []
    for q in queries:
        try:
            with DDGS(timeout=cfg.timeout_ddg) as ddgs:
                results = list(ddgs.text(q, max_results=cfg.max_ddg_results))
        except Exception as exc:
            _rag_log.warning("RAG DDG search failed for %r: %s", q, exc)
            continue
        for r in results:
            url = (r.get("href") or "").strip().lower().rstrip("/")
            if url and url not in seen:
                seen.add(url)
                merged.append(r)
    _rag_log.info("RAG: %d unique URLs after search", len(merged))
    return merged


# ===============================================================================
# ASYNC FETCHING
# ===============================================================================

_RAG_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0)"}
_RAG_BLOCKED_EXTENSIONS = {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}
_RAG_BLOCKED_DOMAINS = {
    "twitter.com", "x.com", "instagram.com",
    "facebook.com", "tiktok.com", "reddit.com",
}


def _rag_is_fetchable(url: str) -> bool:
    from urllib.parse import urlparse
    parsed = urlparse(url.lower())
    if any(parsed.netloc.endswith(d) for d in _RAG_BLOCKED_DOMAINS):
        return False
    if any(parsed.path.endswith(ext) for ext in _RAG_BLOCKED_EXTENSIONS):
        return False
    return True


def _rag_extract_text(html: str) -> str:
    text = trafilatura.extract(html, include_comments=False, include_tables=False, favor_recall=True)
    if text and len(text) >= 200:
        return text
    if HAS_NEWSPAPER:
        try:
            article = NewspaperArticle(url="")
            article.set_html(html)
            article.parse()
            if article.text and len(article.text) >= 200:
                return article.text
        except Exception:
            pass
    return ""


async def _rag_fetch_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    result: dict,
    cfg: RAGConfig,
) -> list[Document]:
    url: str = result.get("href", "")
    title: str = result.get("title", "No title")
    snippet: str = result.get("body", "")

    def _snippet_doc() -> list[Document]:
        return (
            [Document(page_content=snippet, metadata={"title": title, "url": url, "snippet": snippet, "chunk_id": 0})]
            if snippet else []
        )

    if not url or not _rag_is_fetchable(url):
        return _snippet_doc()

    async with semaphore:
        try:
            resp = await client.get(url, headers=_RAG_HEADERS, timeout=cfg.fetch_timeout)
            resp.raise_for_status()
            html = resp.text
        except Exception as exc:
            _rag_log.debug("RAG fetch failed for %s: %s", url, exc)
            return _snippet_doc()

    text = _rag_extract_text(html) or snippet
    text = " ".join(text.split())[: cfg.max_text_chars]
    chunks = _RAG_SPLITTER.split_text(text)
    return [
        Document(page_content=chunk, metadata={"title": title, "url": url, "snippet": snippet, "chunk_id": i})
        for i, chunk in enumerate(chunks)
    ]


async def _rag_fetch_all_async(results: list[dict], cfg: RAGConfig) -> list[Document]:
    semaphore = asyncio.Semaphore(cfg.max_concurrent_fetches)
    async with httpx.AsyncClient(follow_redirects=True) as client:
        tasks = [asyncio.ensure_future(_rag_fetch_one(client, semaphore, r, cfg)) for r in results]
        done, pending = await asyncio.wait(tasks, timeout=cfg.total_fetch_budget)
        if pending:
            _rag_log.info("RAG: cancelling %d slow fetches (budget %.1fs exceeded)", len(pending), cfg.total_fetch_budget)
            for t in pending:
                t.cancel()

    docs: list[Document] = []
    for fut in done:
        try:
            batch = fut.result()
            if not isinstance(batch, Exception):
                docs.extend(batch)
        except Exception as exc:
            _rag_log.debug("RAG fetch task raised: %s", exc)

    _rag_log.info("RAG: %d chunks after fetch", len(docs))
    return docs


def _rag_fetch_all(results: list[dict], cfg: RAGConfig) -> list[Document]:
    """
    Sync entry-point — compatible with both Jupyter and plain scripts.
    Same brace-safety fix for the expansion prompt
    """
    try:
        loop = asyncio.get_running_loop()       # raises RuntimeError if not running
        return loop.run_until_complete(_rag_fetch_all_async(results, cfg))   # Jupyter path
    except RuntimeError:
        return asyncio.run(_rag_fetch_all_async(results, cfg))               # script path


# ===============================================================================
# HYBRID RETRIEVAL: BM25 + DENSE + RRF
# ===============================================================================


def _rag_dense_retrieve(query: str, docs: list[Document], top_k: int) -> list[Document]:
    """Bi-encoder retrieval using the module-level SentenceTransformer (on-device, no network)."""
    if not docs:
        return []
    texts = [d.page_content for d in docs]
    doc_embs = _RAG_EMBED_MODEL.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    q_emb = _RAG_EMBED_MODEL.encode(query, normalize_embeddings=True)
    scores = np.dot(doc_embs, q_emb)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [docs[i] for i in top_idx]


def _rag_rrf_fuse(ranked_lists: list[list[Document]], cfg: RAGConfig) -> list[Document]:
    """Reciprocal Rank Fusion — parameter-free merging of multiple ranked lists."""
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            key = "{}::{}".format(
                doc.metadata.get("url", ""),
                doc.metadata.get("chunk_id", hashlib.md5(doc.page_content.encode()).hexdigest()[:8]),
            )
            scores[key] = scores.get(key, 0.0) + 1.0 / (cfg.rrf_k + rank)
            doc_map[key] = doc
    ranked_keys = sorted(scores, key=scores.__getitem__, reverse=True)
    fused = []
    for key in ranked_keys:
        doc = doc_map[key]
        doc.metadata["rrf_score"] = scores[key]
        fused.append(doc)
    _rag_log.info("RAG RRF: merged -> %d docs", len(fused))
    return fused


def _rag_hybrid_retrieve(query: str, docs: list[Document], cfg: RAGConfig) -> list[Document]:
    bm25 = BM25Retriever.from_documents(docs, k=cfg.bm25_top_k)
    bm25_results = bm25.invoke(query)
    dense_results = _rag_dense_retrieve(query, docs, cfg.dense_top_k)
    return _rag_rrf_fuse([bm25_results, dense_results], cfg)


# ===============================================================================
# DIVERSITY FILTER + CROSS-ENCODER RERANK
# ===============================================================================


def _rag_diversity_filter(docs: list[Document], max_per_url: int) -> list[Document]:
    seen: dict[str, int] = {}
    filtered = []
    for doc in docs:
        url = doc.metadata.get("url", "")
        if seen.get(url, 0) >= max_per_url:
            continue
        filtered.append(doc)
        seen[url] = seen.get(url, 0) + 1
    return filtered


def _rag_cross_encode_rerank(query: str, docs: list[Document], cfg: RAGConfig) -> list[Document]:
    if not docs:
        return docs
    pairs = [(query, d.page_content) for d in docs]
    try:
        scores = _RAG_CROSS_ENCODER.predict(pairs)
    except Exception as exc:
        _rag_log.warning("RAG cross-encoder failed: %s", exc)
        return docs[: cfg.final_top_k]
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[: cfg.final_top_k]]


# ===============================================================================
# FORMAT
# ===============================================================================


def _rag_format_evidence(docs: list[Document]) -> str:
    if not docs:
        return "No evidence found."
    blocks = []
    for i, doc in enumerate(docs, 1):
        blocks.append(
            f"[{i}] {doc.metadata.get('title', 'N/A')}\n"
            f"URL: {doc.metadata.get('url', 'N/A')}\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(blocks)


# ===============================================================================
# PIPELINE (internal, called by RAGStrategy)
# ===============================================================================


def _rag_retrieve(query: str, cfg: RAGConfig, llm: LocalLLM) -> str:
    t0 = time.perf_counter()

    # 1. Query expansion using the local LLM
    queries = _expand_query_rag(query, cfg, llm)

    # 2. Web search across all variants
    raw_results = _rag_search_all(queries, cfg)
    if not raw_results:
        return "No results found."

    # 3. Parallel async fetch with hard time budget
    docs = _rag_fetch_all(raw_results, cfg)
    if not docs:
        return "No results found."

    # 4. Hybrid BM25 + dense + RRF fusion
    fused_docs = _rag_hybrid_retrieve(query, docs, cfg)

    # 5. Diversity filter BEFORE cross-encoder so final_top_k slots stay diverse
    diverse_docs = _rag_diversity_filter(fused_docs, cfg.diversity_max_per_url)

    # 6. Cross-encoder rerank on the diverse candidate set
    top_docs = _rag_cross_encode_rerank(query, diverse_docs, cfg)

    _rag_log.info("RAG retrieve() done in %.2fs → %d chunks", time.perf_counter() - t0, len(top_docs))
    return _rag_format_evidence(top_docs)


# ===============================================================================
# STRATEGY
# ===============================================================================


class RAGStrategy(BaseStrategy):
    """
    Retrieval-Augmented Generation strategy.

    Plugs into the existing LocalLLM interface (GemmaLLM / QwenLLM).
    The same LLM is used for optional query expansion and for the final answer.

    Usage
    -----
        strategy = RAGStrategy(llm=GemmaLLM())
        prediction = strategy.answer(question)

        # Custom retrieval config:
        cfg = RAGConfig(num_extra_queries=0, final_top_k=3)
        strategy = RAGStrategy(llm=QwenLLM(), retrieval_config=cfg)
    """

    name = "rag"
    _log = _rag_log

    def __init__(
        self,
        llm: LocalLLM,
        retrieval_config: RAGConfig | None = None,
        fallback_strategy: BaseStrategy | None = None,
    ):
        self.llm = llm
        self.cfg = retrieval_config or RAGConfig()
        self.fallback = fallback_strategy or HeuristicStrategy()

    def answer(self, question: Question) -> AnswerPrediction:
        try:
            evidence = _rag_retrieve(question.text, self.cfg, self.llm)
            prompt = build_rag_prompt(question, evidence)
            # Override max_new_tokens here instead of relying on the LLM config default
            raw_text = self.llm.generate(prompt, max_new_tokens=self.cfg.answer_max_new_tokens)
            prediction = parse_answer_prediction(raw_text, question, strategy_name=self.name)
            prediction.metadata.update({
                "strategy": self.name,
                "model_name": getattr(self.llm, "model_name", "unknown"),
                "device": getattr(self.llm, "device_summary", "unknown"),
                "num_evidence_chunks": self.cfg.final_top_k,
            })
            return prediction

        except Exception as exc:
            self._log.warning("RAGStrategy error, using fallback: %s", exc)
            fallback_pred = self.fallback.answer(question)
            fallback_pred.metadata["fallback"] = True
            fallback_pred.metadata["fallback_reason"] = str(exc)
            return fallback_pred