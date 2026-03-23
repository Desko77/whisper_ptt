#!/usr/bin/env python3
"""
Benchmark script for Whisper PTT — measures transcription and LLM transform speed.

Usage:
    python benchmark.py "path/to/audio.m4a" --runs 3
    python benchmark.py "path/to/audio.m4a" --runs 3 --no-llm

Config is read from .env (same as whisper_ptt_cuda.py).
Change WHISPER_MODEL / LLM_MODEL / LLM_URL in .env between runs to compare.
"""

import argparse
import os
import sys
import time

# Add CUDA DLL paths from venv nvidia packages (Windows)
_venv_nvidia = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "Lib", "site-packages", "nvidia")
if os.path.isdir(_venv_nvidia):
    _dll_dirs = []
    for sub in os.listdir(_venv_nvidia):
        bin_dir = os.path.join(_venv_nvidia, sub, "bin")
        if os.path.isdir(bin_dir):
            os.add_dll_directory(bin_dir)
            _dll_dirs.append(bin_dir)
    if _dll_dirs:
        os.environ["PATH"] = os.pathsep.join(_dll_dirs) + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import requests
from faster_whisper import WhisperModel

# ---------------------------------------------------------------------------
# Config (reuses .env logic from whisper_ptt_cuda.py)
# ---------------------------------------------------------------------------

_script_dir = os.path.dirname(os.path.abspath(__file__))
_env_path = os.path.join(_script_dir, ".env")
try:
    from dotenv import load_dotenv
    load_dotenv(_env_path)
except ImportError:
    pass


def _env(key, default, *, type_=str):
    full_key = key if key.startswith("WHISPER_PTT_") else f"WHISPER_PTT_{key}"
    raw = os.environ.get(full_key, os.environ.get(key, default))
    if type_ is bool:
        return str(raw).strip().lower() in ("1", "true", "yes", "on")
    if type_ is int:
        return int(raw)
    if type_ is float:
        return float(raw)
    return str(raw)


WHISPER_MODEL = _env("WHISPER_MODEL", "large-v3")
WHISPER_LANGUAGE = _env("WHISPER_LANGUAGE", "en")
WHISPER_INITIAL_PROMPT = _env("WHISPER_INITIAL_PROMPT", "English speech.")

USE_LLM_TRANSFORM = _env("USE_LLM_TRANSFORM", "false", type_=bool)
LLM_BACKEND = _env("LLM_BACKEND", "ollama").strip().lower()
LLM_MODEL = _env("LLM_MODEL", _env("OLLAMA_MODEL", "gemma3:12b"))
LLM_URL = _env("LLM_URL", _env("OLLAMA_URL",
    "http://localhost:11434/api/generate" if LLM_BACKEND == "ollama"
    else "http://localhost:1234/v1/chat/completions"))
LLM_API_KEY = _env("LLM_API_KEY", "")
DEFAULT_LLM_TRANSFORM_PROMPT = """Fix the following speech-to-text transcription. Rules:
- Fix grammar, punctuation, and capitalization
- Remove filler words (um, uh, like, etc.)
- Keep the original language ({detected_lang})
- Keep the original meaning — do NOT add or change content
- If it's already clean, return as-is
- Return ONLY the cleaned text, nothing else

Transcription: {raw_text}"""
LLM_TRANSFORM_PROMPT = _env("LLM_TRANSFORM_PROMPT", DEFAULT_LLM_TRANSFORM_PROMPT)

# ---------------------------------------------------------------------------
# Whisper
# ---------------------------------------------------------------------------


def load_model():
    print(f"Loading Whisper model: {WHISPER_MODEL} ...")
    t0 = time.time()
    model = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
    print(f"Model loaded in {time.time() - t0:.1f}s")
    return model


def transcribe(model, audio_path):
    t0 = time.time()
    segments, info = model.transcribe(
        audio_path,
        language=WHISPER_LANGUAGE,
        initial_prompt=WHISPER_INITIAL_PROMPT,
        beam_size=1,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    elapsed = time.time() - t0
    return text, info.language, elapsed

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------


def _llm_request_ollama(prompt):
    r = requests.post(
        LLM_URL,
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False,
              "options": {"temperature": 0.1, "num_predict": len(prompt) * 2}},
        timeout=30,
    )
    return r.json()["response"].strip()


def _llm_request_openai(prompt):
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    # Bypass proxy for localhost
    proxies = {"http": None, "https": None} if "localhost" in LLM_URL or "127.0.0.1" in LLM_URL else None
    r = requests.post(
        LLM_URL,
        headers=headers,
        json={"model": LLM_MODEL,
              "messages": [{"role": "user", "content": prompt}],
              "temperature": 0.1, "max_tokens": len(prompt) * 2, "stream": False},
        timeout=60,
        proxies=proxies,
    )
    return r.json()["choices"][0]["message"]["content"].strip()


def transform_with_llm(raw_text, detected_lang):
    if not raw_text.strip():
        return raw_text, 0.0
    prompt = LLM_TRANSFORM_PROMPT.format(detected_lang=detected_lang, raw_text=raw_text)
    t0 = time.time()
    if LLM_BACKEND == "openai":
        result = _llm_request_openai(prompt)
    else:
        result = _llm_request_ollama(prompt)
    elapsed = time.time() - t0
    return result, elapsed

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def run_benchmark(audio_path, runs, skip_llm):
    model = load_model()

    print(f"\nConfig: whisper={WHISPER_MODEL}, lang={WHISPER_LANGUAGE}")
    if not skip_llm:
        print(f"        llm={LLM_MODEL}, backend={LLM_BACKEND}, url={LLM_URL}")
    print(f"Audio:  {audio_path}")
    print(f"Runs:   {runs}\n")
    print("=" * 70)

    whisper_times = []
    llm_times = []

    for i in range(runs):
        print(f"\n--- Run {i + 1}/{runs} ---")

        text, lang, w_time = transcribe(model, audio_path)
        whisper_times.append(w_time)
        print(f"  Whisper: {w_time:.2f}s")
        print(f"  Text:    {text[:120]}{'...' if len(text) > 120 else ''}")

        if not skip_llm:
            try:
                llm_text, l_time = transform_with_llm(text, lang)
                llm_times.append(l_time)
                print(f"  LLM:     {l_time:.2f}s")
                print(f"  Result:  {llm_text[:120]}{'...' if len(llm_text) > 120 else ''}")
            except Exception as e:
                print(f"  LLM error: {e}")

    print("\n" + "=" * 70)
    print("\nResults:")
    w = np.array(whisper_times)
    print(f"  Whisper — min: {w.min():.2f}s  avg: {w.mean():.2f}s  max: {w.max():.2f}s")

    if llm_times:
        l = np.array(llm_times)
        print(f"  LLM     — min: {l.min():.2f}s  avg: {l.mean():.2f}s  max: {l.max():.2f}s")
        print(f"  Total   — avg: {w.mean() + l.mean():.2f}s")
    else:
        print(f"  Total   — avg: {w.mean():.2f}s (LLM skipped)")

    # Full transcription from last run
    print(f"\nFull transcription (last run):")
    print(f"  {text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Whisper PTT pipeline")
    parser.add_argument("audio", help="Path to audio file (WAV, M4A, MP3, etc.)")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs (default: 3)")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM transform")
    args = parser.parse_args()

    if not os.path.isfile(args.audio):
        print(f"File not found: {args.audio}")
        sys.exit(1)

    run_benchmark(args.audio, args.runs, args.no_llm)
