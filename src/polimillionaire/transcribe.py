from __future__ import annotations

"""Whisper + Silero VAD transcription utilities."""

import re
import sys
import types
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
_vad_model: Any = None

# Punctuation/whitespace only.
_PUNCT_ONLY = re.compile(r'^[\s\W]+$')
# "Option X" label prefix that Whisper sometimes adds.
_OPTION_LABEL = re.compile(r'^\s*option\s+\w+\s*[:\-.,]?\s*', re.IGNORECASE)


def get_vad() -> Any:
    global _vad_model
    if _vad_model is None:
        from silero_vad import load_silero_vad
        _vad_model = load_silero_vad()
    return _vad_model


def get_whisper(model_id: str = WHISPER_MODEL_ID) -> tuple[Any, Any]:
    global _whisper_model, _whisper_processor
    if _whisper_model is not None:
        return _whisper_model, _whisper_processor

    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    print(f"Loading {model_id} ...")
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Transformers reads HF_HOME automatically
    _whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)

    # suppress_tokens/begin_suppress_tokens must be None (not []) so
    # Whisper's _retrieve_logit_processors skips building those processors —
    # an empty list still triggers processor creation and causes hallucination loops.
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

    print(f"Whisper loaded on {device} (checkpoint max_length={_whisper_model.generation_config.max_length})")
    return _whisper_model, _whisper_processor


def unload_whisper() -> None:
    global _whisper_model, _whisper_processor, _vad_model
    _whisper_model = None
    _whisper_processor = None
    _vad_model = None
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


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


def _whisper_infer(f32: np.ndarray) -> str:
    import torch

    model, processor = get_whisper()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    inputs = processor(f32, sampling_rate=WHISPER_SR, return_tensors="pt")
    features = inputs["input_features"].to(device, dtype=dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ids = model.generate(
            features,
            language="english",
            task="transcribe",
            do_sample=False,
            temperature=0.0,
            num_beams=1,
        )
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


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
    text = _OPTION_LABEL.sub("", text).strip()
    if not text:
        return ""
    if _PUNCT_ONLY.match(text):
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text


def transcribe(audio_bytes: bytes) -> str:
    """
    Robust transcription for speech mixed with background music/noise.

    - Short clips (< VAD_MIN_DURATION_S): skip VAD, use raw audio directly.
      Option clips are 2-4s; VAD mangles them by splitting short speech bursts.
    - Long clips (>= VAD_MIN_DURATION_S): apply VAD to strip music-only regions,
      with fallback to raw audio if the VAD result looks like a hallucination.
    - Normalize: strip "option X" label prefix, remove punctuation-only output.
    """
    f32 = _load_wav_mono_16k(audio_bytes)
    duration_s = len(f32) / WHISPER_SR

    if duration_s < VAD_MIN_DURATION_S:
        return _normalize_transcript(_whisper_infer(f32))

    speech = _extract_speech(f32)
    result = _whisper_infer(speech)

    if not _looks_like_hallucination(result):
        return _normalize_transcript(result)

    result = _whisper_infer(f32)
    if not _looks_like_hallucination(result):
        return _normalize_transcript(result)

    # Last resort: return whichever is shorter (less repetition)
    raw_result = result
    vad_result = _whisper_infer(speech)
    best = min(raw_result, vad_result, key=len) if raw_result and vad_result else (raw_result or vad_result)
    return _normalize_transcript(best)
