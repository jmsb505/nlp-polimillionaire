from __future__ import annotations

"""Strategies, local Gemma loading, prompting, and output parsing."""

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

    def generate(self, prompt: str, **kwargs: object) -> str:
        self.prompts.append(prompt)
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
        self._seed()
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
            torch_dtype=self.config.dtype,
        )
        self._model.eval()

    def _load_pipeline(self) -> None:
        from transformers import pipeline

        self._pipeline = pipeline(
            task="any-to-any",
            model=self.config.model_id,
            device_map=self.config.device_map,
            dtype=self.config.dtype,
        )

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

    def _seed(self) -> None:
        if self.config.seed is None:
            return
        try:
            import torch

            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.seed)
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
            "Gemma 4 E2B requires transformers>=5.7.0 in this notebook kernel. "
            f"Current transformers version is {version}. Run the dependency install cell "
            "with INSTALL_DEPS=True, then restart the kernel and rerun setup."
        )


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())
