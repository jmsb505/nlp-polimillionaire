from __future__ import annotations

"""Strategies, local model loading, prompting, and output parsing."""

import json
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Any, Protocol

from packaging.version import Version

from polimillionaire.types import AnswerPrediction, Question


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
