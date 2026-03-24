#!/usr/bin/env python3
"""
Whisper-PTT (CUDA): push-to-talk voice-to-text using faster-whisper on CUDA.
Hold hotkey → speak → release → transcription pasted into the active window.

Config: WHISPER_PTT_* env vars or .env file (see .env.example-cuda).

Dependencies: faster_whisper, pyaudio, keyboard, pyperclip, requests.
Optional: Ollama for LLM transform.
"""

import io
import os
import wave
import time
import threading
import collections
import keyboard
import pyaudio
import pyperclip
import requests
import numpy as np
from faster_whisper import WhisperModel

# Load .env from script directory (so it works regardless of CWD)
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
        return False  # any other value → off
    if type_ is int:
        return int(raw)
    if type_ is float:
        return float(raw)
    return str(raw)


# -----------------------------------------------------------------------------
# Config (from env; values below are defaults)
# -----------------------------------------------------------------------------

# Whisper (CUDA only — no CPU fallback)
WHISPER_MODEL = _env("WHISPER_MODEL", "large-v3")
WHISPER_LANGUAGE = _env("WHISPER_LANGUAGE", "en")
WHISPER_INITIAL_PROMPT = _env("WHISPER_INITIAL_PROMPT", "English speech.")

# Hotkey (hold to record, release to stop). Default: alt
HOTKEY = _env("HOTKEY", "alt").strip().lower().replace(" ", "")
# Parse combo (e.g. "ctrl+f12" -> ("ctrl", "f12")) for hook; single key -> (None, hotkey)
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
DEFAULT_LLM_TRANSFORM_PROMPT_RU = """Исправь следующую расшифровку речи. Правила:
- Исправь ТОЛЬКО грамматику, пунктуацию и заглавные буквы
- Убери слова-паразиты (эм, ну, типа, вот, короче и т.д.)
- НЕ перефразируй — сохраняй порядок слов и структуру предложения
- Технические термины, названия и специальную лексику оставляй как есть
- Пиши ТОЛЬКО на русском языке, НЕ транслитерируй в латиницу
- Если текст уже чистый, верни как есть
- Верни ТОЛЬКО исправленный текст, без пояснений

Расшифровка: {raw_text}"""

DEFAULT_LLM_TRANSFORM_PROMPT = """Fix the following speech-to-text transcription. Rules:
- Fix ONLY grammar, punctuation, and capitalization
- Remove filler words (um, uh, like, etc.)
- Do NOT rephrase — preserve word order and sentence structure
- Keep technical terms, names, and domain-specific vocabulary as-is
- Keep the original language ({detected_lang}) — do NOT transliterate to Latin script
- If it's already clean, return as-is
- Return ONLY the cleaned text, no explanations

Transcription: {raw_text}"""

def _get_llm_prompt():
    custom = _env("LLM_TRANSFORM_PROMPT", "")
    if custom:
        return custom
    if WHISPER_LANGUAGE and WHISPER_LANGUAGE.startswith("ru"):
        return DEFAULT_LLM_TRANSFORM_PROMPT_RU
    return DEFAULT_LLM_TRANSFORM_PROMPT

LLM_TRANSFORM_PROMPT = _get_llm_prompt()

# Output: copy to clipboard and/or paste to active window
COPY_TO_CLIPBOARD = _env("COPY_TO_CLIPBOARD", "true", type_=bool)
PASTE_TO_ACTIVE_WINDOW = _env("PASTE_TO_ACTIVE_WINDOW", "true", type_=bool)
# Paste method: auto (detect terminals) | ctrl+v | ctrl+shift+v | shift+insert
PASTE_METHOD = _env("PASTE_METHOD", "auto").strip().lower()
if PASTE_METHOD not in ("auto", "ctrl+v", "ctrl+shift+v", "shift+insert"):
    raise SystemExit(f"Invalid config: PASTE_METHOD must be auto, ctrl+v, ctrl+shift+v, or shift+insert (got {PASTE_METHOD!r}).")
# Clipboard after paste (only when Paste is on): restore | clear | preserve
CLIPBOARD_AFTER_PASTE_POLICY = _env("CLIPBOARD_AFTER_PASTE_POLICY", "restore").strip().lower()
if CLIPBOARD_AFTER_PASTE_POLICY not in ("restore", "clear", "preserve"):
    raise SystemExit(
        f"Invalid config: CLIPBOARD_AFTER_PASTE_POLICY must be one of restore, clear, preserve (got {CLIPBOARD_AFTER_PASTE_POLICY!r})."
    )
# Keys after paste: key(s) to send (e.g. enter, ctrl+enter). Empty or "none" = no key.
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
# Windows: add CUDA DLL path (nvidia.* packages)
# -----------------------------------------------------------------------------

def _setup_cuda_dll_path():
    """Add nvidia.cublas/cudnn/cuda_runtime bin dirs to PATH for DLL loading."""
    for name in ("nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_runtime"):
        try:
            mod = __import__(name, fromlist=[""])
            bin_dir = os.path.join(mod.__path__[0], "bin")
            if os.path.isdir(bin_dir):
                os.add_dll_directory(bin_dir)
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]
        except ImportError:
            pass


_setup_cuda_dll_path()


# -----------------------------------------------------------------------------
# Recording state and prebuffer
# -----------------------------------------------------------------------------

_recording = False
_audio_frames = []
_prebuffer_deque = None
_prebuffer_lock = threading.Lock()
_prebuffer_running = True
_pyaudio_instance = None
_whisper_model = None

# Chunked transcription state
_chunk_results = []            # [(chunk_index, raw_text, lang)]
_chunk_results_lock = threading.Lock()
_chunk_index = 0
_next_chunk_frame = 0          # frame index where next chunk starts
_chunking_active = False
_prev_chunk_tail_text = ""     # last ~30 words for initial_prompt context
_pending_chunk_threads = []


def _prebuffer_size():
    return max(1, int(PREBUFFER_SEC * SAMPLE_RATE / CHUNK_SIZE))


# -----------------------------------------------------------------------------
# Audio: prebuffer and WAV
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
        except Exception as e:
            print(f"⚠️  Unexpected mic error: {e} — stopping prebuffer.")
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
    try:
        stream.stop_stream()
        stream.close()
    except Exception:
        pass


def start_recording():
    """Start recording: copy prebuffer into _audio_frames; _recording flag lets worker append."""
    global _recording, _audio_frames
    with _prebuffer_lock:
        _audio_frames[:] = list(_prebuffer_deque)
    _recording = True
    print("🎙️ Recording...")


def frames_to_wav(frames, prepend_silence_sec=0):
    """Bytes frames list → WAV in memory (BytesIO). Optionally prepend silence."""
    if prepend_silence_sec > 0:
        sample_width = _pyaudio_instance.get_sample_size(AUDIO_FORMAT)
        silence_len = int(prepend_silence_sec * SAMPLE_RATE) * sample_width
        frames = [b"\x00" * silence_len] + list(frames)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(_pyaudio_instance.get_sample_size(AUDIO_FORMAT))
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(b"".join(frames))
    buf.seek(0)
    return buf


# -----------------------------------------------------------------------------
# Chunked transcription helpers
# -----------------------------------------------------------------------------

def _sec_to_frames(sec):
    """Convert seconds to number of audio chunk frames."""
    return max(1, int(sec * SAMPLE_RATE / CHUNK_SIZE))


def _reset_chunk_state():
    """Reset all chunking globals for the next recording."""
    global _chunk_results, _chunk_index, _next_chunk_frame
    global _chunking_active, _prev_chunk_tail_text, _pending_chunk_threads
    with _chunk_results_lock:
        _chunk_results = []
    _chunk_index = 0
    _next_chunk_frame = 0
    _chunking_active = False
    _prev_chunk_tail_text = ""
    _pending_chunk_threads = []


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

def transcribe(wav_buffer):
    """Transcribe WAV with Whisper. Returns (text, language_code)."""
    print("🔄 Transcribing...")
    t0 = time.time()
    segments, info = _whisper_model.transcribe(
        wav_buffer,
        language=WHISPER_LANGUAGE,
        initial_prompt=WHISPER_INITIAL_PROMPT,
        beam_size=1,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    print(f"📝 Whisper ({time.time() - t0:.1f}s): {text}")
    return text, info.language


def _transcribe_chunk(wav_buffer, initial_prompt):
    """Transcribe a single chunk with custom initial_prompt for context continuity."""
    t0 = time.time()
    segments, info = _whisper_model.transcribe(
        wav_buffer,
        language=WHISPER_LANGUAGE,
        initial_prompt=initial_prompt,
        beam_size=1,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    print(f"   📝 Chunk ({time.time() - t0:.1f}s): {text[:80]}{'...' if len(text) > 80 else ''}")
    return text, info.language


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

# Terminal process names — these windows use Ctrl+Shift+V for paste
_TERMINAL_PROCESSES = frozenset({
    "windowsterminal.exe", "cmd.exe", "powershell.exe", "pwsh.exe",
    "conhost.exe", "wezterm-gui.exe", "alacritty.exe", "hyper.exe",
    "mintty.exe", "wsl.exe", "bash.exe", "git-bash.exe",
    "claude.exe",
})


def _get_foreground_process_name():
    """Get the executable name of the foreground window process."""
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return ""
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
    if not handle:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(260)
        size = wintypes.DWORD(260)
        kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size))
        return os.path.basename(buf.value).lower()
    finally:
        kernel32.CloseHandle(handle)


def _resolve_paste_method():
    """Return the actual paste keystroke string for the current foreground window."""
    if PASTE_METHOD != "auto":
        return PASTE_METHOD
    proc = _get_foreground_process_name()
    if proc in _TERMINAL_PROCESSES:
        return "ctrl+shift+v"
    return "ctrl+v"


def _send_paste():
    """Send paste keystroke via keyboard lib."""
    method = _resolve_paste_method()
    if method == "ctrl+shift+v":
        keyboard.send("ctrl+shift+v")
    elif method == "shift+insert":
        keyboard.send("shift+insert")
    else:
        keyboard.send("ctrl+v")


def _send_keys_after():
    """Send KEYS_AFTER_PASTE via keyboard lib."""
    keyboard.send(KEYS_AFTER_PASTE)


def paste_to_front(text):
    """Copy to clipboard and/or paste to active window via SendInput."""
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
        import winsound
        winsound.MessageBeep(winsound.MB_OK)
    if PASTE_TO_ACTIVE_WINDOW:
        actual_method = _resolve_paste_method()
        _send_paste()
        time.sleep(0.15)
        if KEYS_AFTER_PASTE:
            time.sleep(0.05)
            _send_keys_after()
        suffix = f' + "{KEYS_AFTER_PASTE.upper()}"' if KEYS_AFTER_PASTE else ""
        print(f"✅ Pasted ({actual_method}){suffix}!")
        # Wait for paste to complete before touching clipboard
        time.sleep(0.3)
        if CLIPBOARD_AFTER_PASTE_POLICY == "restore":
            pyperclip.copy(old)
        elif CLIPBOARD_AFTER_PASTE_POLICY == "clear":
            pyperclip.copy("")


# -----------------------------------------------------------------------------
# Chunked transcription: extraction and submission
# -----------------------------------------------------------------------------

def _submit_chunk_for_transcription(chunk_idx, frames, initial_prompt):
    """Spawn a daemon thread to transcribe this chunk."""
    global _prev_chunk_tail_text

    def _worker():
        global _prev_chunk_tail_text
        wav = frames_to_wav(frames, prepend_silence_sec=PADDING_SEC)
        text, lang = _transcribe_chunk(wav, initial_prompt)
        with _chunk_results_lock:
            _chunk_results.append((chunk_idx, text, lang))
            if text.strip():
                words = text.strip().split()
                _prev_chunk_tail_text = " ".join(words[-30:])

    t = threading.Thread(target=_worker, daemon=True)
    _pending_chunk_threads.append(t)
    t.start()


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
    """Wait for all chunk transcriptions, stitch, LLM, paste."""
    for t in _pending_chunk_threads:
        t.join(timeout=60)

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
# Process recording (background thread)
# -----------------------------------------------------------------------------

def _process_recorded_frames(frames):
    """Pipeline: frames → WAV → Whisper → optional LLM → paste."""
    wav = frames_to_wav(frames, prepend_silence_sec=PADDING_SEC)
    raw_text, lang = transcribe(wav)
    if USE_LLM_TRANSFORM and raw_text.strip():
        final_text = transform_with_llm(raw_text, lang)
    else:
        final_text = raw_text
    paste_to_front(final_text)


def stop_recording_and_process():
    """Stop recording, wait for last frames, then transcribe and paste in background."""
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
        threading.Thread(target=_process_recorded_frames, args=(frames,), daemon=True).start()
    else:
        # Long recording — extract final chunk and assemble all
        _extract_final_chunk(frames)
        threading.Thread(target=_assemble_and_output, daemon=True).start()


# -----------------------------------------------------------------------------
# Hotkey and banner
# -----------------------------------------------------------------------------

def _on_hotkey_press(_event=None):
    if not _recording:
        if HOTKEY_MODIFIER is None or keyboard.is_pressed(HOTKEY_MODIFIER):
            start_recording()


def _on_hotkey_release(_event=None):
    stop_recording_and_process()


def _format_banner():
    w = 70
    def line(s, width=None):
        width = width or w
        padded = (s + " " * width)[:width]
        return "║" + padded + "║"
    parts = [
        "╔" + "═" * w + "╗\n",
        line("     🎤 Whisper-PTT ready!", w - 1) + "\n",
        line("") + "\n",
        line(f'     Hotkey: "{HOTKEY.upper()}" (hold to record, release to transcribe)') + "\n",
        line(f"     LLM transform: {'ON (' + LLM_BACKEND + ')' if USE_LLM_TRANSFORM else 'OFF'}") + "\n",
        line(f"     Copy to clipboard: {'ON' if COPY_TO_CLIPBOARD else 'OFF'}") + "\n",
        line(f"     Paste to active window: {'ON (' + PASTE_METHOD + ')' if PASTE_TO_ACTIVE_WINDOW else 'OFF'}") + "\n",
        line(f"     Chunked transcription: {'ON (' + str(CHUNK_DURATION_SEC) + 's chunks, ' + str(CHUNK_OVERLAP_SEC) + 's overlap)' if CHUNK_DURATION_SEC > 0 else 'OFF'}") + "\n",
    ]
    if PASTE_TO_ACTIVE_WINDOW:
        parts.append((line(f'     Keys after paste: "{KEYS_AFTER_PASTE.upper()}"') if KEYS_AFTER_PASTE else line("     Keys after paste: —")) + "\n")
    parts.extend([line("") + "\n", line('     "CTRL+C" to exit') + "\n", "╚" + "═" * w + "╝"])
    return "".join(parts)


def main():
    global _pyaudio_instance, _whisper_model, _prebuffer_deque

    print("⏳ Loading Whisper model... (first run may download the model)")
    _whisper_model = WhisperModel(
        WHISPER_MODEL,
        device="cuda",
        compute_type="float16",
    )
    print("✅ Whisper loaded!")

    _pyaudio_instance = pyaudio.PyAudio()
    _prebuffer_deque = collections.deque(maxlen=_prebuffer_size())

    print(f"🎧 Prebuffer active (last {PREBUFFER_SEC}s)")
    threading.Thread(target=prebuffer_worker, daemon=True).start()

    print(_format_banner())
    print(f'👂 Listening — hold "{HOTKEY.upper()}" to start recording.')

    # Suppress the hotkey so the OS doesn't process it (e.g. Alt opening menus, Pause freezing terminal)
    _suppress = HOTKEY_KEY in ("alt", "pause")
    keyboard.on_press_key(HOTKEY_KEY, _on_hotkey_press, suppress=_suppress)
    keyboard.on_release_key(HOTKEY_KEY, _on_hotkey_release, suppress=_suppress)

    # Block forever — exit only via Ctrl+C
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
        raise SystemExit(0)
