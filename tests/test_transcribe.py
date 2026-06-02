from __future__ import annotations

import numpy as np
from types import SimpleNamespace

from polimillionaire import transcribe as transcribe_module


def test_transcribe_falls_back_to_raw_whisper_when_vad_fails(monkeypatch):
    monkeypatch.setattr(
        transcribe_module,
        "_load_wav_mono_16k",
        lambda audio_bytes: np.zeros(transcribe_module.WHISPER_SR * 7, dtype=np.float32),
    )
    monkeypatch.setattr(
        transcribe_module,
        "_extract_speech",
        lambda f32: (_ for _ in ()).throw(ModuleNotFoundError("silero_vad")),
    )
    monkeypatch.setattr(
        transcribe_module,
        "_whisper_infer",
        lambda f32, **kwargs: " What is the question? ",
    )

    assert transcribe_module.transcribe(b"audio") == "What is the question?"


def test_make_transcriber_loads_configured_whisper_before_transcribing(monkeypatch):
    loaded = {}

    def fake_get_whisper(model_id=None, device=None, dtype=None):
        loaded["model_id"] = model_id
        loaded["device"] = device
        loaded["dtype"] = dtype
        return object(), object()

    monkeypatch.setattr(transcribe_module, "get_whisper", fake_get_whisper)
    def fake_transcribe(audio_bytes, model_id=None, device=None, dtype=None, **kwargs):
        loaded["transcribe_model_id"] = model_id
        loaded["transcribe_device"] = device
        loaded["transcribe_dtype"] = dtype
        return "done"

    monkeypatch.setattr(transcribe_module, "transcribe", fake_transcribe)

    transcriber = transcribe_module.make_transcriber(
        model_id="openai/whisper-small.en",
        device="cpu",
        dtype="float32",
    )

    assert transcriber(b"audio") == "done"
    assert loaded == {
        "transcribe_model_id": "openai/whisper-small.en",
        "transcribe_device": "cpu",
        "transcribe_dtype": "float32",
    }


def test_make_transcriber_retries_empty_transcript_with_fallback(monkeypatch):
    calls = []

    def fake_transcribe(audio_bytes, model_id=None, device=None, dtype=None, **kwargs):
        calls.append({"model_id": model_id, "device": device, "dtype": dtype})
        if model_id == "openai/whisper-base.en":
            return ""
        return "six"

    monkeypatch.setattr(transcribe_module, "transcribe", fake_transcribe)

    transcriber = transcribe_module.make_transcriber(
        model_id="openai/whisper-base.en",
        device="cpu",
        dtype="float32",
        fallback_model_id="openai/whisper-large-v3-turbo",
        fallback_device="cpu",
        fallback_dtype="float32",
    )

    assert transcriber(b"audio") == "six"
    assert calls == [
        {"model_id": "openai/whisper-base.en", "device": "cpu", "dtype": "float32"},
        {"model_id": "openai/whisper-large-v3-turbo", "device": "cpu", "dtype": "float32"},
    ]


def test_make_transcriber_retries_garbled_short_transcript(monkeypatch):
    calls = []

    def fake_transcribe(audio_bytes, model_id=None, device=None, dtype=None, **kwargs):
        calls.append(model_id)
        if model_id == "openai/whisper-base.en":
            return "and Eburnum."
        return "Andy Burnham."

    monkeypatch.setattr(transcribe_module, "transcribe", fake_transcribe)

    transcriber = transcribe_module.make_transcriber(
        model_id="openai/whisper-base.en",
        fallback_model_id="openai/whisper-large-v3-turbo",
    )

    assert transcriber(b"audio") == "Andy Burnham."
    assert calls == ["openai/whisper-base.en", "openai/whisper-large-v3-turbo"]


def test_transcribe_uses_configured_whisper_for_short_audio(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        transcribe_module,
        "_load_wav_mono_16k",
        lambda audio_bytes: np.zeros(transcribe_module.WHISPER_SR, dtype=np.float32),
    )

    def fake_infer(f32, model_id=None, device=None, dtype=None):
        seen["model_id"] = model_id
        seen["device"] = device
        seen["dtype"] = dtype
        return "short transcript"

    monkeypatch.setattr(transcribe_module, "_whisper_infer", fake_infer)

    assert (
        transcribe_module.transcribe(
            b"audio",
            model_id="openai/whisper-base.en",
            device="cpu",
            dtype="float32",
        )
        == "short transcript"
    )
    assert seen == {
        "model_id": "openai/whisper-base.en",
        "device": "cpu",
        "dtype": "float32",
    }


def test_whisper_infer_omits_language_for_english_only_models(monkeypatch):
    class FakeTensor:
        device = "cpu"
        dtype = "float32"

        def to(self, device, dtype):
            return self

    class FakeModel:
        generation_config = type("GenerationConfig", (), {"is_multilingual": False})()

        def parameters(self):
            return iter([FakeTensor()])

        def generate(self, features, **kwargs):
            assert "language" not in kwargs
            assert "task" not in kwargs
            return [[1, 2, 3]]

    class FakeProcessor:
        def __call__(self, f32, sampling_rate, return_tensors):
            return {"input_features": FakeTensor()}

        def batch_decode(self, ids, skip_special_tokens):
            return ["decoded"]

    monkeypatch.setattr(
        transcribe_module,
        "get_whisper",
        lambda **kwargs: (FakeModel(), FakeProcessor()),
    )

    assert transcribe_module._whisper_infer(np.zeros(10, dtype=np.float32)) == "decoded"


def test_faster_whisper_infer_uses_cpu_int8_model(monkeypatch):
    seen = {}

    class FakeFasterWhisper:
        def transcribe(self, audio, **kwargs):
            seen["audio_dtype"] = audio.dtype
            seen["kwargs"] = kwargs
            return [SimpleNamespace(text=" first "), SimpleNamespace(text="second")], object()

    def fake_get_faster_whisper(model_id=None, device=None, compute_type=None):
        seen["model_id"] = model_id
        seen["device"] = device
        seen["compute_type"] = compute_type
        return FakeFasterWhisper()

    monkeypatch.setattr(transcribe_module, "get_faster_whisper", fake_get_faster_whisper)

    text = transcribe_module._faster_whisper_infer(
        np.zeros(10, dtype=np.float64),
        model_id="distil-large-v3",
        device="cpu",
        compute_type="int8",
    )

    assert text == "first second"
    assert seen["model_id"] == "distil-large-v3"
    assert seen["device"] == "cpu"
    assert seen["compute_type"] == "int8"
    assert seen["audio_dtype"] == np.float32
    assert seen["kwargs"]["condition_on_previous_text"] is False


def test_transcribe_can_use_faster_whisper_backend(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        transcribe_module,
        "_load_wav_mono_16k",
        lambda audio_bytes: np.zeros(transcribe_module.WHISPER_SR, dtype=np.float32),
    )

    def fake_infer(f32, model_id=None, device=None, compute_type=None):
        seen["model_id"] = model_id
        seen["device"] = device
        seen["compute_type"] = compute_type
        return "fast transcript"

    monkeypatch.setattr(transcribe_module, "_faster_whisper_infer", fake_infer)

    assert (
        transcribe_module.transcribe(
            b"audio",
            model_id="distil-large-v3",
            device="cpu",
            backend="faster_whisper",
            faster_compute_type="int8",
        )
        == "fast transcript"
    )
    assert seen == {
        "model_id": "distil-large-v3",
        "device": "cpu",
        "compute_type": "int8",
    }


def test_normalize_transcript_removes_common_option_audio_artifacts():
    assert transcribe_module._normalize_transcript("Topshundee! *laughs* So...") == "So..."
    assert transcribe_module._normalize_transcript("-Tobson D. -President Maya Sandu") == "President Maya Sandu"
    assert transcribe_module._normalize_transcript("Haha, according to documents") == "according to documents"
    assert transcribe_module._normalize_transcript("Option D: Soul") == "Soul"
