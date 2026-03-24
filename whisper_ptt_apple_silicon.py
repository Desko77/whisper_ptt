#!/usr/bin/env python3
"""
Whisper-PTT (Apple Silicon): push-to-talk voice-to-text using mlx-whisper on Metal.
Hold hotkey -> speak -> release -> transcription pasted into the active window.

Config: WHISPER_PTT_* env vars or .env file (see .env.example-apple-silicon).

Dependencies: mlx-whisper, pyaudio, keyboard, pyperclip, requests.
Optional: Ollama for LLM transform.
"""

import os
import queue
import subprocess
import time
import threading
import collections
import numpy as np
import pyaudio
import pyperclip
import requests
import mlx_whisper

_script_dir = os.path.dirname(os.path.abspath(__file__))
_env_path = os.path.join(_script_dir, ".env")
try:
    from dotenv import load_dotenv
    load_dotenv(_env_path)
except ImportError:
    if os.path.isfile(_env_path):
        print()
        print("  ❌  NOTE: You have a .env file but python-dotenv is not installed, so it is not loaded.")
        print("      Config comes from environment variables only.")
        print("      To use .env, run:  pip install python-dotenv")
        print()


def _env(key, default, *, type_=str):
    """Read env var with type coercion. WHISPER_PTT_ prefix is optional."""
    full_key = key if key.startswith("WHISPER_PTT_") else f"WHISPER_PTT_{key}"
    raw = os.environ.get(full_key, os.environ.get(key, default))
    if type_ is bool:
        s = str(raw).strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off", ""):
            return False
        return False
    if type_ is int:
        return int(raw)
    if type_ is float:
        return float(raw)
    return str(raw)


# -----------------------------------------------------------------------------
# Config (from env; values below are defaults)
# -----------------------------------------------------------------------------

# Whisper (DEVICE and COMPUTE_TYPE not needed — MLX uses Metal automatically)
WHISPER_MODEL = _env("WHISPER_MODEL", "large-v3-turbo")
WHISPER_LANGUAGE = _env("WHISPER_LANGUAGE", "en")
WHISPER_INITIAL_PROMPT = _env("WHISPER_INITIAL_PROMPT", "English speech.")

# Hotkey (hold to record, release to stop). Default: option
HOTKEY = _env("HOTKEY", "option").strip().lower().replace(" ", "")
if "+" in HOTKEY:
    _parts = HOTKEY.split("+", 1)
    HOTKEY_MODIFIER, HOTKEY_KEY = _parts[0].strip(), _parts[1].strip()
else:
    HOTKEY_MODIFIER, HOTKEY_KEY = None, HOTKEY

# LLM transform — optional, OFF by default
USE_LLM_TRANSFORM = _env("USE_LLM_TRANSFORM", "false", type_=bool)
# Backend: "ollama" (Ollama native API) or "openai" (OpenAI-compatible: LM Studio, llama.cpp, etc.)
LLM_BACKEND = _env("LLM_BACKEND", "ollama").strip().lower()
if LLM_BACKEND not in ("ollama", "openai"):
    raise SystemExit(f"Invalid config: LLM_BACKEND must be 'ollama' or 'openai' (got {LLM_BACKEND!r}).")
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

# Output: copy to clipboard and/or paste to active window
COPY_TO_CLIPBOARD = _env("COPY_TO_CLIPBOARD", "true", type_=bool)
PASTE_TO_ACTIVE_WINDOW = _env("PASTE_TO_ACTIVE_WINDOW", "true", type_=bool)
CLIPBOARD_AFTER_PASTE_POLICY = _env("CLIPBOARD_AFTER_PASTE_POLICY", "restore").strip().lower()
if CLIPBOARD_AFTER_PASTE_POLICY not in ("restore", "clear", "preserve"):
    raise SystemExit(
        f"Invalid config: CLIPBOARD_AFTER_PASTE_POLICY must be one of restore, clear, preserve (got {CLIPBOARD_AFTER_PASTE_POLICY!r})."
    )
KEYS_AFTER_PASTE = _env("KEYS_AFTER_PASTE", "enter").strip().lower()
if KEYS_AFTER_PASTE in ("", "none"):
    KEYS_AFTER_PASTE = None

# Audio
SAMPLE_RATE = _env("SAMPLE_RATE", "16000", type_=int)
CHANNELS = 1
CHUNK_SIZE = _env("CHUNK_SIZE", "1024", type_=int)
AUDIO_FORMAT = pyaudio.paInt16

# Prebuffer and padding
PREBUFFER_SEC = _env("PREBUFFER_SEC", "0.5", type_=float)
PADDING_SEC = _env("PADDING_SEC", "0.2", type_=float)
MIN_FRAMES = _env("MIN_FRAMES", "5", type_=int)
# Simple silence gate: max int16 amplitude below this is treated as silence.
SILENCE_AMPLITUDE_THRESHOLD = _env("SILENCE_AMPLITUDE", "750", type_=int)

# Chunked transcription for long recordings (0 = disabled)
CHUNK_DURATION_SEC = _env("CHUNK_DURATION_SEC", "15", type_=float)
CHUNK_OVERLAP_SEC = _env("CHUNK_OVERLAP_SEC", "2.0", type_=float)


# -----------------------------------------------------------------------------
# MLX model name → HuggingFace repo mapping
# -----------------------------------------------------------------------------

_MLX_MODEL_MAP = {
    "tiny": "mlx-community/whisper-tiny",
    "tiny.en": "mlx-community/whisper-tiny.en",
    "base": "mlx-community/whisper-base",
    "base.en": "mlx-community/whisper-base.en",
    "small": "mlx-community/whisper-small",
    "small.en": "mlx-community/whisper-small.en",
    "medium": "mlx-community/whisper-medium",
    "medium.en": "mlx-community/whisper-medium.en",
    "large": "mlx-community/whisper-large-v3-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "turbo": "mlx-community/whisper-turbo",
}


def _resolve_model(name):
    """Resolve short model name to mlx-community HuggingFace repo. Pass-through if already a repo path."""
    return _MLX_MODEL_MAP.get(name, name)


# -----------------------------------------------------------------------------
# Recording state and prebuffer
# -----------------------------------------------------------------------------

_recording = False
_audio_frames = []
_prebuffer_deque = None
_prebuffer_lock = threading.Lock()
_prebuffer_running = True
_pyaudio_instance = None
_mlx_model_path = None
_transcribe_queue = queue.Queue()
_model_ready = threading.Event()

# Chunked transcription state
_chunk_results = []            # [(chunk_index, raw_text, lang)]
_chunk_results_lock = threading.Lock()
_chunk_index = 0
_next_chunk_frame = 0          # frame index where next chunk starts
_chunking_active = False
_prev_chunk_tail_text = ""     # last ~30 words for initial_prompt context
_chunk_transcribe_queue = queue.Queue()  # chunks enqueued during recording


def _prebuffer_size():
    return max(1, int(PREBUFFER_SEC * SAMPLE_RATE / CHUNK_SIZE))


# -----------------------------------------------------------------------------
# Audio: prebuffer and numpy conversion
# -----------------------------------------------------------------------------

_MIC_MAX_RETRIES = 5
_MIC_RETRY_DELAY = 1.0


def _open_microphone_stream():
    """Open mic stream, retrying with PyAudio re-init on transient PortAudio errors."""
    global _pyaudio_instance
    for attempt in range(1, _MIC_MAX_RETRIES + 1):
        try:
            return _pyaudio_instance.open(
                format=AUDIO_FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE,
            )
        except OSError as e:
            print(f"⚠️  Mic open failed (attempt {attempt}/{_MIC_MAX_RETRIES}): {e}")
            try:
                _pyaudio_instance.terminate()
            except Exception:
                pass
            time.sleep(_MIC_RETRY_DELAY * attempt)
            _pyaudio_instance = pyaudio.PyAudio()
    raise RuntimeError("Could not open microphone after retries — check audio permissions and device.")


def prebuffer_worker():
    """Background thread: read mic into ring buffer; when recording, also append to _audio_frames."""
    global _recording, _audio_frames
    stream = _open_microphone_stream()
    while _prebuffer_running:
        try:
            chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        except OSError as e:
            print(f"⚠️  Mic read error: {e} — reopening stream...")
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
            try:
                stream = _open_microphone_stream()
            except RuntimeError:
                print("❌ Mic recovery failed, prebuffer stopping.")
                return
            continue
        except Exception:
            break
        with _prebuffer_lock:
            _prebuffer_deque.append(chunk)
            if _recording:
                _audio_frames.append(chunk)
                # Chunked transcription: extract chunk when threshold exceeded
                if CHUNK_DURATION_SEC > 0:
                    chunk_threshold = _sec_to_frames(CHUNK_DURATION_SEC)
                    frames_since = len(_audio_frames) - _next_chunk_frame
                    if frames_since >= chunk_threshold:
                        _extract_and_submit_chunk()
    stream.stop_stream()
    stream.close()


def start_recording():
    """Start recording: copy prebuffer into _audio_frames; _recording flag lets worker append."""
    global _recording, _audio_frames
    with _prebuffer_lock:
        _audio_frames[:] = list(_prebuffer_deque)
    _recording = True
    print("🎙️ Recording...")


def frames_to_numpy(frames, prepend_silence_sec=0):
    """Raw PCM int16 frames → float32 numpy array normalised to [-1, 1]."""
    raw = b"".join(frames)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if prepend_silence_sec > 0:
        silence = np.zeros(int(prepend_silence_sec * SAMPLE_RATE), dtype=np.float32)
        audio = np.concatenate([silence, audio])
    return audio


# -----------------------------------------------------------------------------
# Chunked transcription helpers
# -----------------------------------------------------------------------------

def _sec_to_frames(sec):
    """Convert seconds to number of audio chunk frames."""
    return max(1, int(sec * SAMPLE_RATE / CHUNK_SIZE))


def _reset_chunk_state():
    """Reset all chunking globals for the next recording."""
    global _chunk_results, _chunk_index, _next_chunk_frame
    global _chunking_active, _prev_chunk_tail_text
    with _chunk_results_lock:
        _chunk_results = []
    _chunk_index = 0
    _next_chunk_frame = 0
    _chunking_active = False
    _prev_chunk_tail_text = ""
    # Drain chunk queue
    while not _chunk_transcribe_queue.empty():
        try:
            _chunk_transcribe_queue.get_nowait()
        except queue.Empty:
            break


def _merge_overlapping_text(left, right):
    """Merge two text segments, deduplicating overlapping words at the boundary."""
    if not left.strip():
        return right
    if not right.strip():
        return left

    left_words = left.split()
    right_words = right.split()

    max_overlap = min(len(left_words), len(right_words), 40)
    best_overlap = 0
    for overlap_len in range(1, max_overlap + 1):
        left_suffix = [w.lower().strip(".,!?;:\"'") for w in left_words[-overlap_len:]]
        right_prefix = [w.lower().strip(".,!?;:\"'") for w in right_words[:overlap_len]]
        if left_suffix == right_prefix:
            best_overlap = overlap_len

    if best_overlap > 0:
        return left + " " + " ".join(right_words[best_overlap:])
    return left + " " + right


def _stitch_chunks(chunk_results):
    """Stitch ordered chunk transcriptions with overlap deduplication."""
    if not chunk_results:
        return "", WHISPER_LANGUAGE

    chunk_results.sort(key=lambda x: x[0])
    langs = [lang for _, _, lang in chunk_results if lang]
    primary_lang = max(set(langs), key=langs.count) if langs else WHISPER_LANGUAGE
    texts = [text for _, text, _ in chunk_results]

    if len(texts) == 1:
        return texts[0], primary_lang

    stitched = texts[0]
    for i in range(1, len(texts)):
        stitched = _merge_overlapping_text(stitched, texts[i])
    return stitched.strip(), primary_lang


# -----------------------------------------------------------------------------
# Transcription and LLM
# -----------------------------------------------------------------------------

def transcribe(audio_np):
    """Transcribe float32 numpy audio with mlx-whisper. Returns (text, language_code)."""
    print("🔄 Transcribing...")
    t0 = time.time()
    result = mlx_whisper.transcribe(
        audio_np,
        path_or_hf_repo=_mlx_model_path,
        language=WHISPER_LANGUAGE,
        initial_prompt=WHISPER_INITIAL_PROMPT,
        fp16=True,
    )
    text = result["text"].strip()
    lang = result.get("language", WHISPER_LANGUAGE)
    print(f"📝 Whisper ({time.time() - t0:.1f}s): {text}")
    return text, lang


def _transcribe_chunk(audio_np, initial_prompt):
    """Transcribe a single chunk with custom initial_prompt for context continuity."""
    t0 = time.time()
    result = mlx_whisper.transcribe(
        audio_np,
        path_or_hf_repo=_mlx_model_path,
        language=WHISPER_LANGUAGE,
        initial_prompt=initial_prompt,
        fp16=True,
    )
    text = result["text"].strip()
    lang = result.get("language", WHISPER_LANGUAGE)
    print(f"   📝 Chunk ({time.time() - t0:.1f}s): {text[:80]}{'...' if len(text) > 80 else ''}")
    return text, lang


def _llm_request_ollama(prompt):
    """Send request to Ollama native API."""
    r = requests.post(
        LLM_URL,
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": len(prompt) * 2},
        },
        timeout=30,
    )
    return r.json()["response"].strip()


def _llm_request_openai(prompt):
    """Send request to OpenAI-compatible API (LM Studio, llama.cpp, etc.)."""
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    r = requests.post(
        LLM_URL,
        headers=headers,
        json={
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": len(prompt) * 2,
            "stream": False,
        },
        timeout=30,
    )
    return r.json()["choices"][0]["message"]["content"].strip()


def transform_with_llm(raw_text, detected_lang):
    """LLM transform: post-process transcription via configured backend."""
    if not raw_text.strip():
        return raw_text
    print(f"🔄 LLM transform ({LLM_BACKEND})...")
    t0 = time.time()
    prompt = LLM_TRANSFORM_PROMPT.format(detected_lang=detected_lang, raw_text=raw_text)
    try:
        if LLM_BACKEND == "openai":
            result = _llm_request_openai(prompt)
        else:
            result = _llm_request_ollama(prompt)
        print(f"✨ LLM ({time.time() - t0:.1f}s): {result}")
        return result
    except Exception as e:
        print(f"❌ LLM error: {e}, using raw text")
        return raw_text


# -----------------------------------------------------------------------------
# Output: clipboard and/or paste to active window
# -----------------------------------------------------------------------------

_KEY_CODES = {
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7,
    "c": 8, "v": 9, "b": 11, "q": 12, "w": 13, "e": 14, "r": 15,
    "y": 16, "t": 17, "1": 18, "2": 19, "3": 20, "4": 21, "6": 22,
    "5": 23, "9": 25, "7": 26, "8": 28, "0": 29, "o": 31, "u": 32,
    "i": 34, "p": 35, "l": 37, "j": 38, "k": 40, "n": 45, "m": 46,
    "enter": 36, "return": 36, "tab": 48, "space": 49, "delete": 51,
    "escape": 53,
}

_MODIFIERS = {
    "command": "command down",
    "control": "control down",
    "option": "option down",
    "shift": "shift down",
}


def _applescript_key_code(key_name, modifier=None):
    """Send a key via AppleScript key code (layout-independent)."""
    code = _KEY_CODES.get(key_name.lower())
    using = _MODIFIERS.get(modifier) if modifier else None
    if code is not None:
        if using:
            script = f'tell application "System Events" to key code {code} using {{{using}}}'
        else:
            script = f'tell application "System Events" to key code {code}'
    elif using:
        script = f'tell application "System Events" to keystroke "{key_name}" using {{{using}}}'
    else:
        script = f'tell application "System Events" to keystroke "{key_name}"'
    subprocess.run(["osascript", "-e", script], check=False)


def _send_keys_after_paste():
    """Parse KEYS_AFTER_PASTE (e.g. 'enter', 'ctrl+enter') and send via AppleScript."""
    if not KEYS_AFTER_PASTE:
        return
    parts = KEYS_AFTER_PASTE.split("+")
    if len(parts) == 1:
        _applescript_key_code(parts[0])
    else:
        modifier = parts[0].replace("ctrl", "control").replace("cmd", "command")
        _applescript_key_code(parts[1], modifier=modifier)


def paste_to_front(text):
    """Copy to clipboard and/or paste to active window (Cmd+V via AppleScript on macOS)."""
    if not text.strip():
        print("❌ Empty text, skipping")
        return
    if not COPY_TO_CLIPBOARD and not PASTE_TO_ACTIVE_WINDOW:
        print("✅ Done (console only)")
        return
    old = pyperclip.paste()
    pyperclip.copy(text)
    if COPY_TO_CLIPBOARD:
        print("📋 Copied to clipboard!")
    if PASTE_TO_ACTIVE_WINDOW:
        _applescript_key_code("v", modifier="command")
        time.sleep(0.1)
        if KEYS_AFTER_PASTE:
            time.sleep(0.05)
            _send_keys_after_paste()
        suffix = f' + "{KEYS_AFTER_PASTE.upper()}"' if KEYS_AFTER_PASTE else ""
        print(f"✅ Pasted to active window{suffix}!")
        if CLIPBOARD_AFTER_PASTE_POLICY == "restore":
            pyperclip.copy(old)
        elif CLIPBOARD_AFTER_PASTE_POLICY == "clear":
            pyperclip.copy("")


# -----------------------------------------------------------------------------
# Chunked transcription: extraction and submission (macOS)
# -----------------------------------------------------------------------------

def _submit_chunk_for_transcription(chunk_idx, frames, initial_prompt):
    """Enqueue chunk for the single MLX transcription worker thread."""
    _chunk_transcribe_queue.put((chunk_idx, frames, initial_prompt))


def _extract_and_submit_chunk():
    """Extract the next chunk from _audio_frames and submit for transcription.
    Called under _prebuffer_lock."""
    global _next_chunk_frame, _chunk_index, _chunking_active

    overlap_frames = _sec_to_frames(CHUNK_OVERLAP_SEC)
    chunk_frames = _sec_to_frames(CHUNK_DURATION_SEC)

    start = _next_chunk_frame
    if _chunk_index > 0 and start >= overlap_frames:
        start -= overlap_frames

    end = _next_chunk_frame + chunk_frames
    frames_copy = _audio_frames[start:end]

    # Silence check — skip silent chunks
    raw = b"".join(frames_copy)
    audio_int16 = np.frombuffer(raw, dtype=np.int16)
    if audio_int16.size == 0 or np.max(np.abs(audio_int16)) < SILENCE_AMPLITUDE_THRESHOLD:
        idx = _chunk_index
        _chunk_index += 1
        _next_chunk_frame = end
        _chunking_active = True
        with _chunk_results_lock:
            _chunk_results.append((idx, "", WHISPER_LANGUAGE))
        print(f"   ⏭️ Chunk {idx}: silence, skipping")
        return

    _next_chunk_frame = end
    idx = _chunk_index
    _chunk_index += 1
    _chunking_active = True

    initial_prompt = _prev_chunk_tail_text if _prev_chunk_tail_text else WHISPER_INITIAL_PROMPT
    duration = len(frames_copy) * CHUNK_SIZE / SAMPLE_RATE
    print(f"   🔀 Chunk {idx}: {duration:.1f}s (frames {start}–{end})")

    _submit_chunk_for_transcription(idx, frames_copy, initial_prompt)


def _extract_final_chunk(frames):
    """Extract and submit the remaining audio after the last full chunk."""
    global _chunk_index
    overlap_frames = _sec_to_frames(CHUNK_OVERLAP_SEC)

    start = _next_chunk_frame
    if _chunk_index > 0 and start >= overlap_frames:
        start -= overlap_frames

    remaining = frames[start:]
    if len(remaining) < MIN_FRAMES:
        return

    raw = b"".join(remaining)
    audio_int16 = np.frombuffer(raw, dtype=np.int16)
    if audio_int16.size == 0 or np.max(np.abs(audio_int16)) < SILENCE_AMPLITUDE_THRESHOLD:
        return

    initial_prompt = _prev_chunk_tail_text if _prev_chunk_tail_text else WHISPER_INITIAL_PROMPT
    idx = _chunk_index
    _chunk_index += 1
    duration = len(remaining) * CHUNK_SIZE / SAMPLE_RATE
    print(f"   🔀 Final chunk {idx}: {duration:.1f}s")
    _submit_chunk_for_transcription(idx, remaining, initial_prompt)


def _assemble_and_output():
    """Stitch all chunk results, LLM, paste. Called on the MLX worker thread."""
    with _chunk_results_lock:
        results = list(_chunk_results)

    stitched, lang = _stitch_chunks(results)
    print(f"🧩 Stitched {len(results)} chunks: {stitched[:120]}{'...' if len(stitched) > 120 else ''}")

    if USE_LLM_TRANSFORM and stitched.strip():
        final_text = transform_with_llm(stitched, lang)
    else:
        final_text = stitched

    paste_to_front(final_text)
    _reset_chunk_state()


# -----------------------------------------------------------------------------
# Transcription worker (all MLX/Metal ops on a single thread)
# -----------------------------------------------------------------------------

def _transcription_worker():
    """Persistent thread owning all MLX operations — Metal requires same-thread access."""
    global _mlx_model_path, _prev_chunk_tail_text
    _mlx_model_path = _resolve_model(WHISPER_MODEL)
    print(f"⏳ Loading mlx-whisper model '{_mlx_model_path}'... (first run downloads from HuggingFace)")
    warmup_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
    mlx_whisper.transcribe(
        warmup_audio,
        path_or_hf_repo=_mlx_model_path,
        language=WHISPER_LANGUAGE,
        fp16=True,
        verbose=False,
    )
    print("✅ mlx-whisper loaded!")
    _model_ready.set()

    while True:
        # Process chunk queue first (non-blocking)
        try:
            item = _chunk_transcribe_queue.get_nowait()
            if item == "ASSEMBLE":
                _assemble_and_output()
                continue
            chunk_idx, frames, initial_prompt = item
            audio_np = frames_to_numpy(frames, prepend_silence_sec=PADDING_SEC)
            text, lang = _transcribe_chunk(audio_np, initial_prompt)
            with _chunk_results_lock:
                _chunk_results.append((chunk_idx, text, lang))
                if text.strip():
                    words = text.strip().split()
                    _prev_chunk_tail_text = " ".join(words[-30:])
            continue
        except queue.Empty:
            pass

        # Then check main queue (blocking with timeout so we can poll chunk queue)
        try:
            frames = _transcribe_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if frames is None:
            break
        audio_np = frames_to_numpy(frames, prepend_silence_sec=PADDING_SEC)
        raw_text, lang = transcribe(audio_np)
        if USE_LLM_TRANSFORM and raw_text.strip():
            final_text = transform_with_llm(raw_text, lang)
        else:
            final_text = raw_text
        paste_to_front(final_text)


def stop_recording_and_process():
    """Stop recording, wait for last frames, then enqueue for transcription."""
    global _recording
    if not _recording:
        return
    _recording = False
    time.sleep(0.15)

    frames = list(_audio_frames)
    duration_sec = len(frames) * CHUNK_SIZE / SAMPLE_RATE
    print(f"⏹️ Recorded {duration_sec:.1f}s (with {PREBUFFER_SEC}s prebuffer)")

    # Only process recordings longer than 0.7 seconds in total.
    if duration_sec <= 0.7 or len(frames) < MIN_FRAMES:
        print("❌ Recording too short")
        _reset_chunk_state()
        return

    # Simple silence / noise gate: skip very low-energy audio.
    raw = b"".join(frames)
    audio_int16 = np.frombuffer(raw, dtype=np.int16)
    if audio_int16.size == 0 or np.max(np.abs(audio_int16)) < SILENCE_AMPLITUDE_THRESHOLD:
        print("❌ Audio too quiet / silence, skipping")
        _reset_chunk_state()
        return

    if not _chunking_active:
        # Short recording — single-pass (existing behavior)
        _reset_chunk_state()
        _transcribe_queue.put(frames)
    else:
        # Long recording — extract final chunk and assemble all
        _extract_final_chunk(frames)
        _chunk_transcribe_queue.put("ASSEMBLE")


# -----------------------------------------------------------------------------
# Hotkey and banner
# -----------------------------------------------------------------------------

def _on_hotkey_press(_event=None):
    if not _recording:
        start_recording()


def _on_hotkey_release(_event=None):
    stop_recording_and_process()


def _start_hotkey_listener_mac():
    """Hotkey listener using pynput on macOS (no root required, Option supported)."""
    try:
        from pynput import keyboard as pynput_keyboard
    except ImportError:
        print("❌ pynput is required on macOS. Install with:")
        print("   pip install pynput")
        return

    Key = pynput_keyboard.Key
    pressed = set()

    def _spec_from_name(name):
        if not name:
            return None
        n = str(name).strip().lower()
        if n in ("cmd", "command", "⌘"):
            return Key.cmd
        if n in ("option", "opt", "alt", "⌥"):
            return Key.alt
        if n in ("ctrl", "control"):
            return Key.ctrl
        if n == "shift":
            return Key.shift
        if n.startswith("f") and n[1:].isdigit():
            return getattr(Key, n, None)
        if len(n) == 1:
            return n
        return None

    hotkey_key_spec = _spec_from_name(HOTKEY_KEY)
    hotkey_mod_spec = _spec_from_name(HOTKEY_MODIFIER) if HOTKEY_MODIFIER else None

    def _matches(key, spec):
        if spec is None:
            return False
        if isinstance(spec, str):
            return getattr(key, "char", None) == spec
        return key == spec

    def on_press(key):
        pressed.add(key)
        if HOTKEY_MODIFIER is None:
            if _matches(key, hotkey_key_spec):
                _on_hotkey_press()
        else:
            if _matches(key, hotkey_key_spec) and any(
                _matches(k, hotkey_mod_spec) for k in pressed
            ):
                _on_hotkey_press()

    def on_release(key):
        if key == Key.esc:
            print("\n👋 Exiting...")
            return False
        if HOTKEY_MODIFIER is None:
            if _matches(key, hotkey_key_spec):
                _on_hotkey_release()
        else:
            if _matches(key, hotkey_key_spec) or _matches(key, hotkey_mod_spec):
                _on_hotkey_release()
        pressed.discard(key)

    with pynput_keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


def _format_banner():
    w = 70
    def line(s, width=None):
        width = width or w
        padded = (s + " " * width)[:width]
        return "║" + padded + "║"
    parts = [
        "╔" + "═" * w + "╗\n",
        line("     🎤 Whisper-PTT (Apple Silicon / MLX) ready!", w - 1) + "\n",
        line("") + "\n",
        line(f'     Hotkey: "{HOTKEY.upper()}" (hold to record, release to transcribe)') + "\n",
        line(f"     Model: {_mlx_model_path}") + "\n",
        line(f"     LLM transform: {'ON (' + LLM_BACKEND + ')' if USE_LLM_TRANSFORM else 'OFF'}") + "\n",
        line(f"     Copy to clipboard: {'ON' if COPY_TO_CLIPBOARD else 'OFF'}") + "\n",
        line(f"     Paste to active window: {'ON' if PASTE_TO_ACTIVE_WINDOW else 'OFF'}") + "\n",
        line(f"     Chunked transcription: {'ON (' + str(CHUNK_DURATION_SEC) + 's chunks, ' + str(CHUNK_OVERLAP_SEC) + 's overlap)' if CHUNK_DURATION_SEC > 0 else 'OFF'}") + "\n",
    ]
    if PASTE_TO_ACTIVE_WINDOW:
        parts.append((line(f'     Keys after paste: "{KEYS_AFTER_PASTE.upper()}"') if KEYS_AFTER_PASTE else line("     Keys after paste: —")) + "\n")
    parts.extend([line("") + "\n", line('     "CTRL+C" to exit') + "\n", "╚" + "═" * w + "╝"])
    return "".join(parts)


def main():
    global _pyaudio_instance, _prebuffer_deque

    threading.Thread(target=_transcription_worker, daemon=True).start()
    _model_ready.wait()

    _pyaudio_instance = pyaudio.PyAudio()
    _prebuffer_deque = collections.deque(maxlen=_prebuffer_size())

    print(f"🎧 Prebuffer active (last {PREBUFFER_SEC}s)")
    threading.Thread(target=prebuffer_worker, daemon=True).start()

    print(_format_banner())
    print(f'👂 Listening — hold "{HOTKEY.upper()}" to start recording.')

    _start_hotkey_listener_mac()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
        raise SystemExit(0)
