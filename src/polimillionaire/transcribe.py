from __future__ import annotations

"""Whisper + Silero VAD transcription utilities."""

import os
import re
import warnings
from collections import Counter
from typing import Any

import numpy as np


WHISPER_MODEL_ID = "openai/whisper-large-v3-turbo"
WHISPER_SR = 16_000

# Clips shorter than this are passed directly to Whisper without VAD filtering.
# Short option clips (2-4s) have their speech spread across the whole clip;
# VAD splitting hurts more than it helps for them.
VAD_MIN_DURATION_S = 6.0

_whisper_model: Any = None
_whisper_processor: Any = None
_whisper_signature: tuple[str, str, str] | None = None
_whisper_cache: dict[tuple[str, str, str], tuple[Any, Any]] = {}
_faster_whisper_model: Any = None
_faster_whisper_signature: tuple[str, str, str] | None = None
_faster_whisper_cache: dict[tuple[str, str, str], Any] = {}
_vad_model: Any = None

# Punctuation/whitespace only.
_PUNCT_ONLY = re.compile(r'^[\s\W]+$')
# "Option X" label prefix that Whisper sometimes adds.
_OPTION_LABEL = re.compile(r'^\s*option\s+\w+\s*[:\-.,]?\s*', re.IGNORECASE)
_MISHEARD_OPTION_LABEL = re.compile(
    r"^\s*[-\u2013]?\s*(?:option\s+[a-d]|options\s+[a-d]|top\s*shun\s*d|topshun(?:dee|d)?|topshin\s*d|topson\s*d|topshundee|tobson\s*d|thompson\s*d)\b\s*[:!\-.,]?\s*",
    re.IGNORECASE,
)
_FILLER_PREFIX = re.compile(r"^\s*(?:haha|hmmm|um|uh|ah)\b\s*[,!.\-]?\s*", re.IGNORECASE)
_NOISE_TOKEN = re.compile(
    r"\*(?:pfft|panting|laughs?|laughter|music|noise|applause|breath(?:ing)?)\*",
    re.IGNORECASE,
)


def get_vad() -> Any:
    global _vad_model
    if _vad_model is None:
        from silero_vad import load_silero_vad
        _vad_model = load_silero_vad()
    return _vad_model


def get_whisper(
    model_id: str | None = None,
    device: str | None = None,
    dtype: str | None = None,
) -> tuple[Any, Any]:
    global _whisper_model, _whisper_processor, _whisper_signature
    model_id = model_id or os.environ.get("POLIMILLIONAIRE_WHISPER_MODEL_ID", WHISPER_MODEL_ID)
    device_name = _resolve_device(device or os.environ.get("POLIMILLIONAIRE_WHISPER_DEVICE", "auto"))
    dtype_name = dtype or os.environ.get("POLIMILLIONAIRE_WHISPER_DTYPE", "auto")
    signature = (model_id, device_name, dtype_name)
    if signature in _whisper_cache:
        _whisper_model, _whisper_processor = _whisper_cache[signature]
        _whisper_signature = signature
        return _whisper_model, _whisper_processor

    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    torch_dtype = _resolve_dtype(dtype_name, device_name)
    print(f"Loading {model_id} on {device_name} ...")

    # Transformers reads HF_HOME automatically
    _whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device_name)

    # None these stay, so Whisper skips extra suppression processors.
    _whisper_model.generation_config.forced_decoder_ids = None
    _whisper_model.generation_config.suppress_tokens = None
    _whisper_model.generation_config.begin_suppress_tokens = None
    _whisper_model.generation_config.do_sample = False
    _whisper_model.generation_config.temperature = 0.0
    _whisper_model.generation_config.num_beams = 1

    _whisper_processor = AutoProcessor.from_pretrained(
        model_id,
        clean_up_tokenization_spaces=False,
    )
    _whisper_signature = signature
    _whisper_cache[signature] = (_whisper_model, _whisper_processor)

    print(f"Whisper loaded on {device_name} (checkpoint max_length={_whisper_model.generation_config.max_length})")
    return _whisper_model, _whisper_processor


def get_faster_whisper(
    model_id: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
) -> Any:
    global _faster_whisper_model, _faster_whisper_signature
    model_id = model_id or os.environ.get("POLIMILLIONAIRE_FASTER_WHISPER_MODEL_ID", "distil-large-v3")
    device_name = (device or os.environ.get("POLIMILLIONAIRE_FASTER_WHISPER_DEVICE", "cpu")).lower()
    if device_name == "auto":
        device_name = "cpu"
    compute_type_name = compute_type or os.environ.get("POLIMILLIONAIRE_FASTER_WHISPER_COMPUTE_TYPE", "int8")
    signature = (model_id, device_name, compute_type_name)
    if signature in _faster_whisper_cache:
        _faster_whisper_model = _faster_whisper_cache[signature]
        _faster_whisper_signature = signature
        return _faster_whisper_model

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is needed for the CPU int8 ASR fallback. "
            "Install it with `pip install faster-whisper`, restart the kernel, and rerun setup."
        ) from exc

    print(f"Loading faster-whisper {model_id} on {device_name} ({compute_type_name}) ...")
    _faster_whisper_model = WhisperModel(model_id, device=device_name, compute_type=compute_type_name)
    _faster_whisper_signature = signature
    _faster_whisper_cache[signature] = _faster_whisper_model
    print(f"faster-whisper loaded on {device_name}")
    return _faster_whisper_model


def unload_whisper() -> None:
    global _whisper_model, _whisper_processor, _whisper_signature, _whisper_cache
    global _faster_whisper_model, _faster_whisper_signature, _faster_whisper_cache, _vad_model
    _whisper_model = None
    _whisper_processor = None
    _whisper_signature = None
    _whisper_cache = {}
    _faster_whisper_model = None
    _faster_whisper_signature = None
    _faster_whisper_cache = {}
    _vad_model = None
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def make_transcriber(
    model_id: str | None = None,
    device: str | None = None,
    dtype: str | None = None,
    backend: str = "transformers_whisper",
    faster_compute_type: str | None = None,
    fallback_model_id: str | None = None,
    fallback_device: str | None = None,
    fallback_dtype: str | None = None,
    fallback_backend: str | None = None,
    fallback_faster_compute_type: str | None = None,
):
    def _configured_transcribe(audio_bytes: bytes) -> str:
        primary = transcribe(
            audio_bytes,
            model_id=model_id,
            device=device,
            dtype=dtype,
            backend=backend,
            faster_compute_type=faster_compute_type,
        )
        if not _needs_transcription_retry(primary) or not fallback_model_id:
            return primary
        fallback = transcribe(
            audio_bytes,
            model_id=fallback_model_id,
            device=fallback_device or device,
            dtype=fallback_dtype or dtype,
            backend=fallback_backend or backend,
            faster_compute_type=fallback_faster_compute_type or faster_compute_type,
        )
        return fallback or primary

    return _configured_transcribe


def _needs_transcription_retry(text: str) -> bool:
    normalized = _normalize_transcript(text).strip().lower()
    if not normalized:
        return True
    words = normalized.split()
    artifact_markers = (
        "topsh",
        "topson",
        "topshin",
        "topshun",
        "tops and d",
        "tobson",
        "thompson d",
        "options c",
        "iotola",
        "eburnum",
        "soadabu",
        "paulahooded",
    )
    if any(marker in normalized for marker in artifact_markers):
        return True
    return len(words) <= 4 and normalized.startswith("and ")


def _resolve_device(device: str) -> str:
    normalized = device.lower()
    if normalized != "auto":
        return normalized
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(dtype: str, device: str):
    import torch

    normalized = dtype.lower()
    if normalized == "auto":
        return torch.float16 if device == "cuda" else torch.float32
    if normalized in {"float16", "fp16"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported Whisper dtype: {dtype}")


def _load_wav_mono_16k(audio_bytes: bytes) -> np.ndarray:
    import io
    import scipy.io.wavfile as wav

    src_sr, raw = wav.read(io.BytesIO(audio_bytes))
    if raw.dtype == np.int16:
        f32 = raw.astype(np.float32) / np.iinfo(np.int16).max
    elif raw.dtype == np.int32:
        f32 = raw.astype(np.float32) / np.iinfo(np.int32).max
    else:
        f32 = raw.astype(np.float32)
    if f32.ndim == 2:
        f32 = f32.mean(axis=1)
    if src_sr != WHISPER_SR:
        from scipy.signal import resample_poly
        f32 = resample_poly(f32, WHISPER_SR, src_sr)
    return f32


def _extract_speech(f32: np.ndarray, threshold: float = 0.5, pad_ms: int = 100) -> np.ndarray:
    import torch
    from silero_vad import get_speech_timestamps

    vad = get_vad()
    pad = int(pad_ms * WHISPER_SR / 1000)
    gap = np.zeros(int(0.05 * WHISPER_SR), dtype=np.float32)
    ts = get_speech_timestamps(
        torch.from_numpy(f32), vad,
        sampling_rate=WHISPER_SR,
        threshold=threshold,
        min_speech_duration_ms=100,
        min_silence_duration_ms=200,
    )
    if not ts:
        return f32
    chunks = []
    for t in ts:
        s = max(0, t["start"] - pad)
        e = min(len(f32), t["end"] + pad)
        chunks.append(f32[s:e])
        chunks.append(gap)
    return np.concatenate(chunks)


def _whisper_infer(
    f32: np.ndarray,
    model_id: str | None = None,
    device: str | None = None,
    dtype: str | None = None,
) -> str:
    import torch

    model, processor = get_whisper(model_id=model_id, device=device, dtype=dtype)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    inputs = processor(f32, sampling_rate=WHISPER_SR, return_tensors="pt")
    features = inputs["input_features"].to(device, dtype=dtype)
    generate_kwargs: dict[str, Any] = {
        "do_sample": False,
        "temperature": 0.0,
        "num_beams": 1,
    }
    if getattr(model.generation_config, "is_multilingual", True):
        generate_kwargs.update({"language": "english", "task": "transcribe"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ids = model.generate(features, **generate_kwargs)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def _faster_whisper_infer(
    f32: np.ndarray,
    model_id: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
) -> str:
    model = get_faster_whisper(model_id=model_id, device=device, compute_type=compute_type)
    segments, _ = model.transcribe(
        f32.astype(np.float32),
        language="en",
        task="transcribe",
        beam_size=1,
        vad_filter=False,
        condition_on_previous_text=False,
        temperature=0.0,
    )
    return " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()


def _infer_asr(
    f32: np.ndarray,
    model_id: str | None = None,
    device: str | None = None,
    dtype: str | None = None,
    backend: str = "transformers_whisper",
    faster_compute_type: str | None = None,
) -> str:
    if backend == "transformers_whisper":
        return _whisper_infer(f32, model_id=model_id, device=device, dtype=dtype)
    if backend == "faster_whisper":
        return _faster_whisper_infer(
            f32,
            model_id=model_id,
            device=device,
            compute_type=faster_compute_type,
        )
    raise ValueError(f"Unsupported ASR backend: {backend}")


def _looks_like_hallucination(text: str) -> bool:
    if not text:
        return False
    tokens = text.replace("-", " ").split()
    if len(tokens) < 6:
        return False
    for n in (1, 2):
        grams = [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        top_count = Counter(grams).most_common(1)[0][1]
        if top_count > max(4, len(grams) * 0.35):
            return True
    return False


def _normalize_transcript(text: str) -> str:
    if not text:
        return ""
    text = text.strip('"\'')
    text = _NOISE_TOKEN.sub(" ", text)
    text = _FILLER_PREFIX.sub("", text)
    text = _MISHEARD_OPTION_LABEL.sub("", text)
    text = _OPTION_LABEL.sub("", text).strip()
    text = re.sub(r"^\s*[-\u2013]\s*", "", text)
    text = _NOISE_TOKEN.sub(" ", text)
    text = _FILLER_PREFIX.sub("", text)
    if not text:
        return ""
    if _PUNCT_ONLY.match(text):
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text


def transcribe(
    audio_bytes: bytes,
    model_id: str | None = None,
    device: str | None = None,
    dtype: str | None = None,
    backend: str = "transformers_whisper",
    faster_compute_type: str | None = None,
) -> str:
    """
    Robust transcription for speech mixed with background music/noise.

    - Short clips: skip VAD, use raw audio directly.
    - Long clips: apply VAD, then fall back to raw audio if needed.
    - Normalize: strip option labels and punctuation-only output.
    """
    f32 = _load_wav_mono_16k(audio_bytes)
    duration_s = len(f32) / WHISPER_SR

    if duration_s < VAD_MIN_DURATION_S:
        return _normalize_transcript(
            _infer_asr(
                f32,
                model_id=model_id,
                device=device,
                dtype=dtype,
                backend=backend,
                faster_compute_type=faster_compute_type,
            )
        )

    try:
        speech = _extract_speech(f32)
    except Exception:
        return _normalize_transcript(
            _infer_asr(
                f32,
                model_id=model_id,
                device=device,
                dtype=dtype,
                backend=backend,
                faster_compute_type=faster_compute_type,
            )
        )

    result = _infer_asr(
        speech,
        model_id=model_id,
        device=device,
        dtype=dtype,
        backend=backend,
        faster_compute_type=faster_compute_type,
    )

    if not _looks_like_hallucination(result):
        return _normalize_transcript(result)

    result = _infer_asr(
        f32,
        model_id=model_id,
        device=device,
        dtype=dtype,
        backend=backend,
        faster_compute_type=faster_compute_type,
    )
    if not _looks_like_hallucination(result):
        return _normalize_transcript(result)

    # Last resort: return whichever is shorter (less repetition)
    raw_result = result
    vad_result = _infer_asr(
        speech,
        model_id=model_id,
        device=device,
        dtype=dtype,
        backend=backend,
        faster_compute_type=faster_compute_type,
    )
    best = min(raw_result, vad_result, key=len) if raw_result and vad_result else (raw_result or vad_result)
    return _normalize_transcript(best)
