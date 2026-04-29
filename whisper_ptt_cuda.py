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
import sys
import wave
import time
import logging
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

# Whisper anti-repetition (prevents hallucinated duplicate text)
WHISPER_NO_REPEAT_NGRAM_SIZE = _env("WHISPER_NO_REPEAT_NGRAM_SIZE", "3", type_=int)
WHISPER_REPETITION_PENALTY = _env("WHISPER_REPETITION_PENALTY", "1.1", type_=float)
WHISPER_HALLUCINATION_SILENCE_THRESHOLD = _env("WHISPER_HALLUCINATION_SILENCE_THRESHOLD", "2.0", type_=float)

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
# Reasoning/thinking control for models that support it (e.g. Gemma 4).
# "none" = disable thinking (fast), "low"/"medium"/"high" = enable with budget.
LLM_REASONING_EFFORT = _env("LLM_REASONING_EFFORT", "none").strip().lower()
# z.ai thinking parameter: "disabled" | "enabled" | "" (default, not sent).
# Needed for GLM models on api.z.ai - they think by default unless disabled.
LLM_THINKING = _env("LLM_THINKING", "").strip().lower()
# Fallback LLM (used when primary is unreachable / timeouts / 5xx / 429).
# Applies to openai backend only. Typical use: cloud primary + local LM Studio fallback.
LLM_FALLBACK_ENABLED = _env("LLM_FALLBACK_ENABLED", "false", type_=bool)
LLM_FALLBACK_URL = _env("LLM_FALLBACK_URL", "")
LLM_FALLBACK_MODEL = _env("LLM_FALLBACK_MODEL", "")
LLM_FALLBACK_API_KEY = _env("LLM_FALLBACK_API_KEY", "")
LLM_FALLBACK_THINKING = _env("LLM_FALLBACK_THINKING", "").strip().lower()
# Per-request timeout in seconds (applies to primary and fallback).
LLM_TIMEOUT = _env("LLM_TIMEOUT", "15", type_=int)
# After a primary failure, skip primary for this many seconds and go straight
# to fallback. Avoids paying the primary timeout on every transcription when
# the cloud is known-down. 0 = always try primary first.
LLM_FALLBACK_STICKY_SEC = _env("LLM_FALLBACK_STICKY_SEC", "60", type_=int)
# Replace profanity/obscene language with neutral equivalents in voice-to-text transform
LLM_CLEAN_PROFANITY = _env("LLM_CLEAN_PROFANITY", "true", type_=bool)
# Replace typographic AI-symbols in LLM output with plain ASCII equivalents
# (em/en dash -> '-', curly quotes -> straight, ellipsis -> '...', ё -> е, nbsp -> space)
LLM_STRIP_AI_SYMBOLS = _env("LLM_STRIP_AI_SYMBOLS", "true", type_=bool)
DEFAULT_LLM_TRANSFORM_PROMPT_RU = """Исправь следующую расшифровку речи. Правила:
- Исправь пунктуацию, заглавные буквы и явные грамматические ошибки
- Убери слова-паразиты (эм, ну, типа, вот, короче и т.д.)
- При сомнении — НЕ меняй. Лучше оставить как есть, чем испортить
- НЕ перефразируй — сохраняй порядок слов и структуру предложения
- Технические термины, названия и специальную лексику оставляй как есть
- Пиши ТОЛЬКО на русском языке, НЕ транслитерируй в латиницу
- Верни ТОЛЬКО исправленный текст, без пояснений{profanity_rule}

Расшифровка: {raw_text}"""

DEFAULT_LLM_TRANSFORM_PROMPT = """Fix the following speech-to-text transcription. Rules:
- Fix punctuation, capitalization, and obvious grammar errors
- Remove filler words (um, uh, like, etc.)
- When in doubt — do NOT change. Better to leave as-is than to break it
- Do NOT rephrase — preserve word order and sentence structure
- Keep technical terms, names, and domain-specific vocabulary as-is
- Keep the original language ({detected_lang}) — do NOT transliterate to Latin script
- If it's already clean, return as-is
- Return ONLY the cleaned text, no explanations{profanity_rule}

Transcription: {raw_text}"""

def _get_llm_prompt():
    custom = _env("LLM_TRANSFORM_PROMPT", "")
    if custom:
        return custom
    if WHISPER_LANGUAGE and WHISPER_LANGUAGE.startswith("ru"):
        return DEFAULT_LLM_TRANSFORM_PROMPT_RU
    return DEFAULT_LLM_TRANSFORM_PROMPT

LLM_TRANSFORM_PROMPT = _get_llm_prompt()

# SpellCheck: hotkey to capture selected text, fix via LLM, paste back
SPELLCHECK_ENABLED = _env("SPELLCHECK_ENABLED", "true", type_=bool)
SPELLCHECK_HOTKEY = _env("SPELLCHECK_HOTKEY", "ctrl+t").strip().lower().replace(" ", "")
# Parse spellcheck combo (e.g. "ctrl+t" -> ("ctrl", "t"))
if "+" in SPELLCHECK_HOTKEY:
    _sc_parts = SPELLCHECK_HOTKEY.split("+", 1)
    SPELLCHECK_MODIFIER, SPELLCHECK_KEY = _sc_parts[0].strip(), _sc_parts[1].strip()
else:
    SPELLCHECK_MODIFIER, SPELLCHECK_KEY = None, SPELLCHECK_HOTKEY
SPELLCHECK_LANGUAGE = _env("SPELLCHECK_LANGUAGE", "auto").strip().lower()
# Replace profanity/obscene language with neutral equivalents
SPELLCHECK_CLEAN_PROFANITY = _env("SPELLCHECK_CLEAN_PROFANITY", "true", type_=bool)

_PROFANITY_RULE_RU = "\n- ОБЯЗАТЕЛЬНО заменяй обсценную лексику, мат и грубые выражения (уебанство, пиздец, жопа, бля и пр.) на нейтральные аналоги"
_PROFANITY_RULE_EN = "\n- ALWAYS replace profanity, obscene language, and vulgar expressions (fuck, shit, damn, etc.) with neutral equivalents"

SPELLCHECK_PROMPT_RU = """Исправь следующий текст (язык: {detected_lang}). Правила:
- Исправь пунктуацию, заглавные буквы и явные грамматические ошибки
- Исправь опечатки (перестановки букв, пропущенные/лишние буквы)
- При сомнении - НЕ меняй. Лучше оставить как есть, чем испортить
- НЕ перефразируй - сохраняй порядок слов и структуру предложения
- Технические термины, названия и специальную лексику оставляй как есть
- НЕ изменяй URL, пути файлов и фрагменты кода
- Пиши ТОЛЬКО на русском языке, НЕ транслитерируй в латиницу
- Верни ТОЛЬКО исправленный текст, без пояснений{profanity_rule}

Текст: {raw_text}"""

SPELLCHECK_PROMPT_EN = """Fix the following text. Rules:
- Fix punctuation, capitalization, and obvious grammar errors
- Fix typos (letter transpositions, missing/extra letters)
- When in doubt - do NOT change. Better to leave as-is than to break it
- Do NOT rephrase - preserve word order and sentence structure
- Keep technical terms, names, and domain-specific vocabulary as-is
- Do NOT modify URLs, file paths, or code snippets
- Keep the original language ({detected_lang}) - do NOT transliterate to Latin script
- If it's already clean, return as-is
- Return ONLY the cleaned text, no explanations{profanity_rule}

Text: {raw_text}"""

def _get_spellcheck_prompt():
    custom = _env("SPELLCHECK_PROMPT", "")
    if custom:
        return custom
    if SPELLCHECK_LANGUAGE == "ru":
        return SPELLCHECK_PROMPT_RU
    if SPELLCHECK_LANGUAGE == "en":
        return SPELLCHECK_PROMPT_EN
    return None  # auto — will be chosen per-text

SPELLCHECK_PROMPT = _get_spellcheck_prompt()

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
# Audio device: "default" or device name substring (matched case-insensitively)
AUDIO_DEVICE = _env("AUDIO_DEVICE", "default").strip()

# Prebuffer and padding
PREBUFFER_SEC = _env("PREBUFFER_SEC", "0.5", type_=float)
PADDING_SEC = _env("PADDING_SEC", "0.2", type_=float)
MIN_FRAMES = _env("MIN_FRAMES", "5", type_=int)
# Simple silence gate: max int16 amplitude below this is treated as silence.
SILENCE_AMPLITUDE_THRESHOLD = _env("SILENCE_AMPLITUDE", "750", type_=int)
# Prebuffer mode:
#   "always"  - mic stream stays open continuously (zero first-press latency, more battery use)
#   "timeout" - release mic stream after N idle seconds; first press after release reopens the stream
#               (battery friendly, BT headsets can return to A2DP profile when idle)
PREBUFFER_MODE = _env("PREBUFFER_MODE", "timeout").strip().lower()
if PREBUFFER_MODE not in ("always", "timeout"):
    PREBUFFER_MODE = "timeout"
# Idle seconds before mic stream is released (only when PREBUFFER_MODE=timeout). Default: 30 min.
PREBUFFER_IDLE_TIMEOUT_SEC = _env("PREBUFFER_IDLE_TIMEOUT_SEC", "1800", type_=int)

# Chunked transcription for long recordings (0 = disabled)
CHUNK_DURATION_SEC = _env("CHUNK_DURATION_SEC", "15", type_=float)
CHUNK_OVERLAP_SEC = _env("CHUNK_OVERLAP_SEC", "2.0", type_=float)

# Logging
LOG_ENABLED = _env("LOG_ENABLED", "false", type_=bool)
SHOW_NOTIFICATIONS = _env("SHOW_NOTIFICATIONS", "true", type_=bool)
LOG_FILE = _env("LOG_FILE", os.path.join(_script_dir, "whisper_ptt.log"))

# Logger setup
_logger = logging.getLogger("whisper_ptt")
_logger.setLevel(logging.DEBUG)
_log_handler = None


def _setup_logging():
    """Configure file logging if enabled. Idempotent."""
    global _log_handler
    if _log_handler:
        return
    if not LOG_ENABLED:
        _logger.addHandler(logging.NullHandler())
        return
    _log_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    _log_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    _logger.addHandler(_log_handler)
    _logger.info("--- Logging started (pid=%d) ---", os.getpid())


_setup_logging()


def get_log_path():
    """Return the log file path."""
    return LOG_FILE


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
_prebuffer_thread = None
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

# GUI callbacks (optional — None in console mode)
_audio_level_callback = None   # fn(peak: float) called from prebuffer thread
_event_callback = None         # fn(event: str, data: dict) called from various threads


def set_audio_level_callback(fn):
    """Register callback for audio level updates. fn(peak_amplitude: float)."""
    global _audio_level_callback
    _audio_level_callback = fn


def set_event_callback(fn):
    """Register callback for events. fn(event: str, data: dict).
    Events: recording_started, recording_stopped, processing_started,
            transcription_done, error."""
    global _event_callback
    _event_callback = fn


def _fire_event(event, data=None):
    """Fire event callback if registered."""
    cb = _event_callback
    if cb:
        try:
            cb(event, data or {})
        except Exception:
            pass


def _prebuffer_size():
    return max(1, int(PREBUFFER_SEC * SAMPLE_RATE / CHUNK_SIZE))


# -----------------------------------------------------------------------------
# Audio: prebuffer and WAV
# -----------------------------------------------------------------------------

_MIC_MAX_RETRIES = 5
_MIC_RETRY_DELAY = 1.0
_mic_switch_event = threading.Event()  # signals prebuffer to reopen stream
_wake_event = threading.Event()        # PTT/SpellCheck press signals worker to wake from SLEEPING
_last_activity_ts = time.monotonic()   # updated on PTT and SpellCheck press; idle timer reference


def _fix_device_name(name):
    """Fix PyAudio device name encoding (UTF-8 bytes misread as system codepage on Windows)."""
    # PyAudio/PortAudio returns UTF-8 bytes, Python decodes them using the system
    # ANSI codepage (cp1251 on Russian Windows). Re-encode to get raw bytes back,
    # then decode as UTF-8.
    import locale
    codepage = locale.getpreferredencoding(False)  # e.g. 'cp1251'
    for enc in (codepage, 'cp1251', 'latin-1'):
        try:
            return name.encode(enc).decode('utf-8')
        except (UnicodeDecodeError, UnicodeEncodeError):
            continue
    return name


def list_audio_devices():
    """Return list of input devices: [{"index": int, "name": str, "is_default": bool}, ...].

    Always creates a fresh PyAudio instance to get up-to-date device info
    and avoid thread-safety issues with _pyaudio_instance (owned by prebuffer_worker).
    """
    pa = pyaudio.PyAudio()
    try:
        default_idx = None
        try:
            default_idx = pa.get_default_input_device_info()["index"]
        except (OSError, KeyError):
            pass
        devices = []
        for i in range(pa.get_device_count()):
            try:
                info = pa.get_device_info_by_index(i)
            except OSError:
                continue
            if info.get("maxInputChannels", 0) > 0:
                devices.append({
                    "index": i,
                    "name": _fix_device_name(info["name"]),
                    "is_default": i == default_idx,
                })
        return devices
    finally:
        pa.terminate()


def _probe_default_device_info():
    """Get current system default input device (index, name) using a fresh PortAudio scan.

    Creates a temporary PyAudio instance to force PortAudio to rescan devices,
    avoiding stale cached device lists that miss hot-plug/unplug events.
    Returns (index, name) tuple, or (None, None) if no input device is available.

    NOTE: on Windows MME, the default index can be stable (sound mapper) even when
    the underlying device changes — callers MUST also compare by name.
    """
    pa = None
    try:
        pa = pyaudio.PyAudio()
        info = pa.get_default_input_device_info()
        return (info["index"], _fix_device_name(info["name"]))
    except (OSError, KeyError):
        return (None, None)
    finally:
        if pa:
            try:
                pa.terminate()
            except Exception:
                pass


def _resolve_device_index():
    """Resolve AUDIO_DEVICE config to a PyAudio device index (None = system default)."""
    if not AUDIO_DEVICE or AUDIO_DEVICE.lower() == "default":
        return None
    needle = AUDIO_DEVICE.lower()
    for dev in list_audio_devices():
        if needle in dev["name"].lower():
            return dev["index"]
    print(f"⚠️  Audio device '{AUDIO_DEVICE}' not found, using system default.")
    return None


def get_active_device_name():
    """Return the name of the currently configured audio device."""
    idx = _resolve_device_index()
    if idx is None:
        try:
            info = _pyaudio_instance.get_default_input_device_info()
            return _fix_device_name(info["name"])
        except (OSError, AttributeError):
            return "Default"
    try:
        info = _pyaudio_instance.get_device_info_by_index(idx)
        return _fix_device_name(info["name"])
    except (OSError, AttributeError):
        return AUDIO_DEVICE


def _open_microphone_stream():
    """Open mic stream, retrying with PyAudio re-init on transient PortAudio errors.

    When AUDIO_DEVICE="default", resolves the current system default to an EXPLICIT
    device index at open time (instead of passing None). This pins the stream to a
    real physical device so stream.read() will raise OSError reliably if the device
    disappears — otherwise PortAudio on Windows may silently keep reading zeros.
    """
    global _pyaudio_instance
    device_index = _resolve_device_index()
    if device_index is None:
        # Pin to the current system default explicitly (avoid Windows MME "sound mapper" abstraction).
        try:
            device_index = _pyaudio_instance.get_default_input_device_info()["index"]
        except (OSError, KeyError):
            device_index = None  # fall back to PortAudio-chosen default
    for attempt in range(1, _MIC_MAX_RETRIES + 1):
        try:
            kwargs = dict(
                format=AUDIO_FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE,
            )
            if device_index is not None:
                kwargs["input_device_index"] = device_index
            stream = _pyaudio_instance.open(**kwargs)
            # Log which physical device was actually opened (helps diagnose Windows MME mapping issues).
            try:
                if device_index is not None:
                    opened_name = _fix_device_name(
                        _pyaudio_instance.get_device_info_by_index(device_index)["name"]
                    )
                    print(f"🎧 Mic stream opened: idx={device_index} name='{opened_name}'")
                    _logger.info(f"Mic stream opened: idx={device_index} name='{opened_name}'")
            except Exception:
                pass
            return stream
        except OSError as e:
            print(f"⚠️  Mic open failed (attempt {attempt}/{_MIC_MAX_RETRIES}): {e}")
            try:
                _pyaudio_instance.terminate()
            except Exception:
                pass
            time.sleep(_MIC_RETRY_DELAY * attempt)
            _pyaudio_instance = pyaudio.PyAudio()
            # Re-resolve device index after reinit (default may have finished initializing, e.g. BT).
            device_index = _resolve_device_index()
            if device_index is None:
                try:
                    device_index = _pyaudio_instance.get_default_input_device_info()["index"]
                except (OSError, KeyError):
                    device_index = None
    raise RuntimeError("Could not open microphone after retries - check audio permissions and device.")


def _reinit_pyaudio():
    """Terminate and re-create the global PyAudio instance.

    Must be called from prebuffer_worker thread only (which owns _pyaudio_instance).
    Forces PortAudio to rescan devices so newly connected/disconnected mics are visible.
    """
    global _pyaudio_instance
    try:
        _pyaudio_instance.terminate()
    except Exception:
        pass
    _pyaudio_instance = pyaudio.PyAudio()


def prebuffer_worker():
    """Background thread: read mic into ring buffer; when recording, also append to _audio_frames.

    Two states:
    - ACTIVE   : stream is open, ring buffer being filled.
    - SLEEPING : stream is closed (PREBUFFER_MODE=timeout and idle > timeout).
                 Worker waits on _wake_event; PTT/SpellCheck press wakes it.

    First press after wake-up has no prebuffer pre-roll (deque is empty) and incurs
    the cold-start latency of opening the mic stream (especially noticeable on BT).
    """
    global _recording, _audio_frames

    stream = None
    _last_default_info = (None, None)
    _default_check_interval = max(1, int(SAMPLE_RATE / CHUNK_SIZE * 2))  # ~every 2 sec
    _default_check_counter = 0

    # Initial open: try to open at startup. If it fails, start in SLEEPING (will retry on first PTT).
    try:
        stream = _open_microphone_stream()
        if not AUDIO_DEVICE or AUDIO_DEVICE.lower() == "default":
            _last_default_info = _probe_default_device_info()
    except RuntimeError:
        print("⚠️  Mic not available at startup — prebuffer SLEEPING until first hotkey press.")
        _logger.warning("Initial mic open failed; entering SLEEPING state.")
        stream = None

    while _prebuffer_running:
        # ---- SLEEPING state: wait for wake signal (PTT/SpellCheck) or shutdown ----
        if stream is None:
            woken = _wake_event.wait(timeout=1.0)
            if not _prebuffer_running:
                break
            # Discard a pending mic switch — stream is closed and AUDIO_DEVICE was
            # already updated by switch_microphone(); the upcoming open will use it.
            # Without this, the ACTIVE-state mic-switch handler would double-open.
            if _mic_switch_event.is_set():
                _mic_switch_event.clear()
            if not woken:
                continue
            _wake_event.clear()
            try:
                stream = _open_microphone_stream()
                if not AUDIO_DEVICE or AUDIO_DEVICE.lower() == "default":
                    _last_default_info = _probe_default_device_info()
                else:
                    _last_default_info = (None, None)
                _default_check_counter = 0
                print("🎤 Mic stream reopened (woken by hotkey)")
                _logger.info("Prebuffer ACTIVE (woken from SLEEPING)")
            except RuntimeError:
                print("❌ Mic wake failed — staying SLEEPING.")
                _logger.error("Mic wake failed; remaining SLEEPING.")
                stream = None
                continue
            # fall through to ACTIVE-state logic

        # ---- ACTIVE state ----

        # Check if mic switch was requested
        if _mic_switch_event.is_set():
            _mic_switch_event.clear()
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
            _reinit_pyaudio()
            try:
                stream = _open_microphone_stream()
                if not AUDIO_DEVICE or AUDIO_DEVICE.lower() == "default":
                    _last_default_info = _probe_default_device_info()
                else:
                    _last_default_info = (None, None)
                print(f"🎤 Switched to: {get_active_device_name()}")
            except RuntimeError:
                print("Mic switch failed, prebuffer stopping.")
                return
            continue

        # Auto-detect system default device change (only when AUDIO_DEVICE="default" and not recording)
        _default_check_counter += 1
        if _default_check_counter >= _default_check_interval:
            _default_check_counter = 0
            if (not AUDIO_DEVICE or AUDIO_DEVICE.lower() == "default") and not _recording:
                new_info = _probe_default_device_info()
                new_idx, new_name = new_info
                last_idx, last_name = _last_default_info
                # Detect change by EITHER index OR name (Windows MME may keep index stable).
                changed = (
                    new_idx is not None
                    and (new_idx != last_idx or (new_name and new_name != last_name))
                )
                if changed:
                    _logger.info(
                        f"Default device changed: last=(idx={last_idx}, name='{last_name}') "
                        f"new=(idx={new_idx}, name='{new_name}')"
                    )
                    try:
                        stream.stop_stream()
                        stream.close()
                    except Exception:
                        pass
                    _reinit_pyaudio()
                    try:
                        stream = _open_microphone_stream()
                        _last_default_info = new_info
                        dev_name = get_active_device_name()
                        print(f"🎤 Default device changed, switched to: {dev_name}")
                        _logger.info(f"Auto-switched mic to: {dev_name}")
                        _fire_event("mic_auto_switched", dev_name)
                    except RuntimeError:
                        print("❌ Auto mic switch failed, prebuffer stopping.")
                        return
                    continue
        try:
            chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        except OSError as e:
            print(f"⚠️  Mic read error: {e} — reopening stream...")
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
            _reinit_pyaudio()
            try:
                stream = _open_microphone_stream()
                if not AUDIO_DEVICE or AUDIO_DEVICE.lower() == "default":
                    _last_default_info = _probe_default_device_info()
            except RuntimeError:
                print("❌ Mic recovery failed, prebuffer stopping.")
                return
            continue
        except Exception as e:
            print(f"⚠️  Unexpected mic error: {e} — stopping prebuffer.")
            break
        # Compute audio level OUTSIDE lock to avoid contention
        level_cb = _audio_level_callback
        if level_cb:
            try:
                audio_int16 = np.frombuffer(chunk, dtype=np.int16)
                peak = float(np.max(np.abs(audio_int16))) if audio_int16.size > 0 else 0.0
                level_cb(peak)
            except Exception:
                pass
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

        # ---- Idle timeout check: ACTIVE -> SLEEPING ----
        if (PREBUFFER_MODE == "timeout"
                and not _recording
                and PREBUFFER_IDLE_TIMEOUT_SEC > 0
                and (time.monotonic() - _last_activity_ts) > PREBUFFER_IDLE_TIMEOUT_SEC):
            # Atomic re-check + deque clear under the same lock that start_recording
            # acquires. If a PTT press raced in, _recording is True and we skip closing;
            # otherwise we clear the deque so a (very narrowly racing) start_recording
            # snapshot sees an empty buffer instead of stale audio.
            with _prebuffer_lock:
                if _recording or (time.monotonic() - _last_activity_ts) <= PREBUFFER_IDLE_TIMEOUT_SEC:
                    continue
                _prebuffer_deque.clear()
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
            stream = None
            _last_default_info = (None, None)
            _default_check_counter = 0
            # NOTE: do NOT clear _wake_event here — if PTT raced in just now, we want
            # to wake immediately on next loop iteration.
            print(f"💤 Mic released after {PREBUFFER_IDLE_TIMEOUT_SEC}s idle (prebuffer SLEEPING)")
            _logger.info("Prebuffer SLEEPING (idle > %ds)", PREBUFFER_IDLE_TIMEOUT_SEC)
            _fire_event("prebuffer_sleeping")

    if stream is not None:
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass


def start_recording():
    """Start recording: copy prebuffer into _audio_frames; _recording flag lets worker append.

    Also resets idle timer and wakes the prebuffer worker if it is SLEEPING
    (PREBUFFER_MODE=timeout and stream was released after idle timeout).
    """
    global _recording, _audio_frames, _last_activity_ts
    _last_activity_ts = time.monotonic()
    _wake_event.set()  # no-op if worker is ACTIVE
    with _prebuffer_lock:
        _audio_frames[:] = list(_prebuffer_deque)
    _recording = True
    print("🎙️ Recording...")
    _logger.info("Recording started")
    _fire_event("recording_started")


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
        no_repeat_ngram_size=WHISPER_NO_REPEAT_NGRAM_SIZE,
        repetition_penalty=WHISPER_REPETITION_PENALTY,
        hallucination_silence_threshold=WHISPER_HALLUCINATION_SILENCE_THRESHOLD,
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    elapsed = time.time() - t0
    print(f"📝 Whisper ({elapsed:.1f}s): {text}")
    _logger.info("Whisper %.1fs: %s", elapsed, text)
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
        no_repeat_ngram_size=WHISPER_NO_REPEAT_NGRAM_SIZE,
        repetition_penalty=WHISPER_REPETITION_PENALTY,
        hallucination_silence_threshold=WHISPER_HALLUCINATION_SILENCE_THRESHOLD,
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    elapsed = time.time() - t0
    print(f"   📝 Chunk ({elapsed:.1f}s): {text[:80]}{'...' if len(text) > 80 else ''}")
    _logger.info("Chunk %.1fs: %s", elapsed, text[:120])
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


class _LLMRetryable(Exception):
    """Primary endpoint failure that warrants trying the fallback."""


_llm_fallback_sticky_until = 0.0  # monotonic timestamp; primary is skipped while now < this


def _llm_try_endpoint(url, model, api_key, reasoning_effort, thinking, prompt, timeout):
    """One attempt against an OpenAI-compatible endpoint. Raises _LLMRetryable
    for conditions that warrant fallback (network, 5xx, 429, empty response),
    or raises directly (HTTPError/ValueError) for fatal errors (4xx non-429)."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": len(prompt) * 2,
        "stream": False,
    }
    if reasoning_effort and reasoning_effort != "off":
        body["reasoning_effort"] = reasoning_effort
    if thinking:
        body["thinking"] = {"type": thinking}
    try:
        r = requests.post(url, headers=headers, json=body, timeout=timeout)
    except (requests.ConnectionError, requests.Timeout) as e:
        raise _LLMRetryable(f"{type(e).__name__}: {e}")
    if r.status_code in (429, 500, 502, 503, 504):
        raise _LLMRetryable(f"HTTP {r.status_code}")
    r.raise_for_status()  # other 4xx = fatal (bad request / bad model / bad key)
    data = r.json()
    msg = data["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    if not content:
        # Some providers put the answer into reasoning_content when thinking was
        # active and ran out of max_tokens before content. Accept it.
        content = (msg.get("reasoning_content") or "").strip()
    if not content:
        raise _LLMRetryable("empty response")
    return content


def _llm_request_openai(prompt):
    """Route an OpenAI-compatible request with optional fallback endpoint."""
    global _llm_fallback_sticky_until
    has_fallback = bool(LLM_FALLBACK_ENABLED and LLM_FALLBACK_URL and LLM_FALLBACK_MODEL)
    now = time.monotonic()

    if has_fallback and now < _llm_fallback_sticky_until:
        print(f"🔁 LLM fallback (sticky) → {LLM_FALLBACK_MODEL}")
        return _llm_try_endpoint(
            LLM_FALLBACK_URL, LLM_FALLBACK_MODEL, LLM_FALLBACK_API_KEY,
            "", LLM_FALLBACK_THINKING, prompt, LLM_TIMEOUT,
        )

    try:
        return _llm_try_endpoint(
            LLM_URL, LLM_MODEL, LLM_API_KEY,
            LLM_REASONING_EFFORT, LLM_THINKING, prompt, LLM_TIMEOUT,
        )
    except _LLMRetryable as e:
        if not has_fallback:
            raise RuntimeError(f"LLM primary failed: {e}")
        print(f"⚠ LLM primary failed ({e}), falling back to {LLM_FALLBACK_MODEL}")
        _llm_fallback_sticky_until = now + LLM_FALLBACK_STICKY_SEC
        try:
            return _llm_try_endpoint(
                LLM_FALLBACK_URL, LLM_FALLBACK_MODEL, LLM_FALLBACK_API_KEY,
                "", LLM_FALLBACK_THINKING, prompt, LLM_TIMEOUT,
            )
        except Exception as fb_err:
            raise RuntimeError(f"LLM primary AND fallback failed. primary={e}; fallback={fb_err}")


def transform_with_llm(raw_text, detected_lang):
    """LLM transform: post-process transcription via configured backend."""
    if not raw_text.strip():
        return raw_text
    print(f"🔄 LLM transform ({LLM_BACKEND})...")
    t0 = time.time()
    if LLM_CLEAN_PROFANITY:
        prof_rule = _PROFANITY_RULE_RU if (detected_lang or "").startswith("ru") else _PROFANITY_RULE_EN
    else:
        prof_rule = ""
    kwargs = {"detected_lang": detected_lang, "raw_text": raw_text}
    if "{profanity_rule}" in LLM_TRANSFORM_PROMPT:
        kwargs["profanity_rule"] = prof_rule
    prompt = LLM_TRANSFORM_PROMPT.format(**kwargs)
    try:
        if LLM_BACKEND == "openai":
            result = _llm_request_openai(prompt)
        else:
            result = _llm_request_ollama(prompt)
        if LLM_STRIP_AI_SYMBOLS:
            result = _strip_ai_symbols(result)
        elapsed = time.time() - t0
        print(f"✨ LLM ({elapsed:.1f}s): {result}")
        _logger.info("LLM %s %.1fs: %s", LLM_BACKEND, elapsed, result)
        return result
    except Exception as e:
        print(f"❌ LLM error: {e}, using raw text")
        _logger.error("LLM error: %s", e)
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


def _resolve_paste_method(saved_fg_process=""):
    """Return the actual paste keystroke string. Uses saved process name if available."""
    if PASTE_METHOD != "auto":
        return PASTE_METHOD
    proc = saved_fg_process or _get_foreground_process_name()
    # Strip Electron auto-update suffix: "claude.exe.old.1775632536186" -> "claude.exe"
    base_proc = proc.split(".exe")[0] + ".exe" if ".exe" in proc else proc
    if base_proc in _TERMINAL_PROCESSES:
        return "ctrl+shift+v"
    return "ctrl+v"


# --- SendInput structures (defined once at module level) ---
import ctypes
from ctypes import wintypes

_KEYEVENTF_KEYUP = 0x0002
_INPUT_KEYBOARD = 1

class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]

class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]

class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]

class _INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("mi", _MOUSEINPUT),
        ("ki", _KEYBDINPUT),
        ("hi", _HARDWAREINPUT),
    ]

class _INPUT(ctypes.Structure):
    _fields_ = [
        ("type", wintypes.DWORD),
        ("union", _INPUT_UNION),
    ]

_VK_CONTROL = 0x11
_VK_SHIFT   = 0x10
_VK_V       = 0x56
_VK_C       = 0x43
_VK_INSERT  = 0x2D
_extra = ctypes.c_ulong(0)


_VK_MENU    = 0x12  # Alt
_VK_LWIN    = 0x5B
_VK_RWIN    = 0x5C
_ALL_MODIFIERS = (_VK_CONTROL, _VK_SHIFT, _VK_MENU, _VK_LWIN, _VK_RWIN)


def _release_stuck_modifiers():
    """Release any modifier keys stuck in pressed state (e.g. after suppress)."""
    get_state = ctypes.windll.user32.GetAsyncKeyState
    stuck = []
    for vk in _ALL_MODIFIERS:
        if get_state(vk) & 0x8000:
            stuck.append(vk)
    if not stuck:
        return
    n = len(stuck)
    arr = (_INPUT * n)()
    for i, vk in enumerate(stuck):
        arr[i].type = _INPUT_KEYBOARD
        arr[i].union.ki.wVk = vk
        arr[i].union.ki.dwFlags = _KEYEVENTF_KEYUP
        arr[i].union.ki.dwExtraInfo = ctypes.pointer(_extra)
    ctypes.windll.user32.SendInput(n, arr, ctypes.sizeof(_INPUT))
    names = {_VK_CONTROL: "Ctrl", _VK_SHIFT: "Shift", _VK_MENU: "Alt",
             _VK_LWIN: "LWin", _VK_RWIN: "RWin"}
    _logger.debug("Released stuck modifiers: %s", [names.get(v, hex(v)) for v in stuck])


def _sendinput_combo(*vk_codes):
    """Send a key combination atomically via SendInput."""
    n = len(vk_codes) * 2
    arr = (_INPUT * n)()
    idx = 0
    for vk in vk_codes:
        arr[idx].type = _INPUT_KEYBOARD
        arr[idx].union.ki.wVk = vk
        arr[idx].union.ki.dwFlags = 0
        arr[idx].union.ki.dwExtraInfo = ctypes.pointer(_extra)
        idx += 1
    for vk in reversed(vk_codes):
        arr[idx].type = _INPUT_KEYBOARD
        arr[idx].union.ki.wVk = vk
        arr[idx].union.ki.dwFlags = _KEYEVENTF_KEYUP
        arr[idx].union.ki.dwExtraInfo = ctypes.pointer(_extra)
        idx += 1
    sent = ctypes.windll.user32.SendInput(n, arr, ctypes.sizeof(_INPUT))
    if sent != n:
        _logger.warning("SendInput: sent %d/%d events (blocked by UIPI?)", sent, n)
    return sent


def _send_paste(method=None):
    """Send paste keystroke via SendInput (atomic, layout-safe)."""
    _release_stuck_modifiers()
    method = method or _resolve_paste_method()
    if method == "ctrl+shift+v":
        _sendinput_combo(_VK_CONTROL, _VK_SHIFT, _VK_V)
    elif method == "shift+insert":
        _sendinput_combo(_VK_SHIFT, _VK_INSERT)
    else:
        _sendinput_combo(_VK_CONTROL, _VK_V)


def _send_keys_after():
    """Send KEYS_AFTER_PASTE via keyboard lib."""
    keyboard.send(KEYS_AFTER_PASTE)


def paste_to_front(text, saved_fg_process=""):
    """Copy to clipboard and/or paste to active window via SendInput."""
    if not text.strip():
        print("❌ Empty text, skipping")
        return
    if not COPY_TO_CLIPBOARD and not PASTE_TO_ACTIVE_WINDOW:
        print("✅ Done (console only)")
        _fire_event("transcription_done", {"text": text})
        return
    old = pyperclip.paste()
    pyperclip.copy(text)
    if COPY_TO_CLIPBOARD:
        print("📋 Copied to clipboard!")
        import winsound
        winsound.MessageBeep(winsound.MB_OK)
    if PASTE_TO_ACTIVE_WINDOW:
        actual_method = _resolve_paste_method(saved_fg_process)
        fg_now = _get_foreground_process_name() if os.name == "nt" else ""
        _logger.info("Paste target: saved=%s, now=%s, method=%s",
                      saved_fg_process, fg_now, actual_method)
        _send_paste(actual_method)
        time.sleep(0.15)
        if KEYS_AFTER_PASTE:
            time.sleep(0.05)
            _send_keys_after()
        suffix = f' + "{KEYS_AFTER_PASTE.upper()}"' if KEYS_AFTER_PASTE else ""
        print(f"✅ Pasted ({actual_method}) to {fg_now or 'unknown'}{suffix}!")
        _logger.info("Pasted (%s): %s", actual_method, text[:120])
        # Wait for paste to complete before touching clipboard
        time.sleep(0.3)
        if CLIPBOARD_AFTER_PASTE_POLICY == "restore":
            pyperclip.copy(old)
        elif CLIPBOARD_AFTER_PASTE_POLICY == "clear":
            pyperclip.copy("")
    _fire_event("transcription_done", {"text": text})


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


def _assemble_and_output(fg_process=""):
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

    paste_to_front(final_text, saved_fg_process=fg_process)
    _reset_chunk_state()


# -----------------------------------------------------------------------------
# Process recording (background thread)
# -----------------------------------------------------------------------------

def _process_recorded_frames(frames, fg_process=""):
    """Pipeline: frames -> WAV -> Whisper -> optional LLM -> paste."""
    wav = frames_to_wav(frames, prepend_silence_sec=PADDING_SEC)
    raw_text, lang = transcribe(wav)
    if USE_LLM_TRANSFORM and raw_text.strip():
        final_text = transform_with_llm(raw_text, lang)
    else:
        final_text = raw_text
    paste_to_front(final_text, saved_fg_process=fg_process)


def stop_recording_and_process():
    """Stop recording, wait for last frames, then transcribe and paste in background."""
    global _recording
    if not _recording:
        return
    _recording = False
    # Capture foreground window NOW (before processing delay changes focus)
    fg_process = _get_foreground_process_name() if os.name == "nt" else ""
    _fire_event("recording_stopped")
    time.sleep(0.15)

    with _prebuffer_lock:
        frames = list(_audio_frames)
    duration_sec = len(frames) * CHUNK_SIZE / SAMPLE_RATE
    print(f"⏹️ Recorded {duration_sec:.1f}s (with {PREBUFFER_SEC}s prebuffer)")
    _logger.info("Recorded %.1fs", duration_sec)

    # Only process recordings longer than 0.7 seconds in total.
    if duration_sec <= 0.7 or len(frames) < MIN_FRAMES:
        print("❌ Recording too short")
        _logger.info("Skipped: too short (%.1fs)", duration_sec)
        _reset_chunk_state()
        return

    # Simple silence / noise gate: skip very low-energy audio.
    raw = b"".join(frames)
    audio_int16 = np.frombuffer(raw, dtype=np.int16)
    if audio_int16.size == 0 or np.max(np.abs(audio_int16)) < SILENCE_AMPLITUDE_THRESHOLD:
        print("❌ Audio too quiet / silence, skipping")
        _logger.info("Skipped: silence")
        _reset_chunk_state()
        return

    _fire_event("processing_started")

    if not _chunking_active:
        # Short recording - single-pass (existing behavior)
        _reset_chunk_state()
        threading.Thread(target=_process_recorded_frames, args=(frames, fg_process), daemon=True).start()
    else:
        # Long recording - extract final chunk and assemble all
        _extract_final_chunk(frames)
        threading.Thread(target=_assemble_and_output, args=(fg_process,), daemon=True).start()


# -----------------------------------------------------------------------------
# SpellCheck: capture selected text, fix via LLM, paste back
# -----------------------------------------------------------------------------

# win32clipboard for full clipboard save/restore (pywin32)
try:
    import win32clipboard
    _HAS_WIN32 = True
except ImportError:
    _HAS_WIN32 = False

_CF_UNICODETEXT = 13
_spellcheck_lock = threading.Lock()


def _detect_language(text):
    """Detect language from text using Unicode character ranges."""
    if SPELLCHECK_LANGUAGE != "auto":
        return SPELLCHECK_LANGUAGE
    if not text:
        return "en"
    cyrillic = 0
    latin = 0
    for ch in text:
        cp = ord(ch)
        if 0x0400 <= cp <= 0x04FF:
            cyrillic += 1
        elif (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A) or (0x00C0 <= cp <= 0x024F):
            latin += 1
    total = cyrillic + latin
    if total == 0:
        return "en"
    return "ru" if cyrillic / total > 0.3 else "en"


def _save_clipboard():
    """Save all clipboard formats. Returns list of (format, data) tuples."""
    if not _HAS_WIN32:
        text = pyperclip.paste() or ""
        return [(_CF_UNICODETEXT, text.encode("utf-16-le"))]
    saved = []
    try:
        win32clipboard.OpenClipboard()
        fmt = 0
        while True:
            fmt = win32clipboard.EnumClipboardFormats(fmt)
            if fmt == 0:
                break
            try:
                data = win32clipboard.GetClipboardData(fmt)
                if isinstance(data, str):
                    data = data.encode("utf-16-le")
                elif not isinstance(data, bytes):
                    continue
                saved.append((fmt, data))
            except Exception:
                continue
        win32clipboard.CloseClipboard()
    except Exception as e:
        _logger.debug("Failed to save clipboard: %s", e)
        try:
            win32clipboard.CloseClipboard()
        except Exception:
            pass
    return saved


def _restore_clipboard(saved):
    """Restore clipboard from previously saved formats."""
    if not saved:
        return
    if not _HAS_WIN32:
        for fmt, data in saved:
            if fmt == _CF_UNICODETEXT:
                pyperclip.copy(data.decode("utf-16-le"))
                return
        return
    try:
        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        for fmt, data in saved:
            try:
                if fmt == _CF_UNICODETEXT:
                    win32clipboard.SetClipboardData(fmt, data.decode("utf-16-le"))
                else:
                    win32clipboard.SetClipboardData(fmt, data)
            except Exception:
                continue
        win32clipboard.CloseClipboard()
    except Exception as e:
        _logger.debug("Failed to restore clipboard: %s", e)
        try:
            win32clipboard.CloseClipboard()
        except Exception:
            pass


def _get_clipboard_text():
    """Get current clipboard text via win32clipboard or pyperclip."""
    if _HAS_WIN32:
        try:
            win32clipboard.OpenClipboard()
            try:
                text = win32clipboard.GetClipboardData(_CF_UNICODETEXT)
            except Exception:
                text = ""
            win32clipboard.CloseClipboard()
            return text or ""
        except Exception:
            try:
                win32clipboard.CloseClipboard()
            except Exception:
                pass
            return ""
    return pyperclip.paste() or ""


def _set_clipboard_text(text):
    """Set clipboard to plain text via win32clipboard or pyperclip."""
    if _HAS_WIN32:
        try:
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(_CF_UNICODETEXT, text)
            win32clipboard.CloseClipboard()
            return
        except Exception:
            try:
                win32clipboard.CloseClipboard()
            except Exception:
                pass
    pyperclip.copy(text)


def _send_copy():
    """Send Ctrl+C via SendInput."""
    _sendinput_combo(_VK_CONTROL, _VK_C)


def _clean_llm_response(text):
    """Strip common LLM artifacts: surrounding quotes, markdown wrappers."""
    import re
    result = text.strip()
    for pat in [
        r"^Here'?s?\s+the\s+corrected\s+text\s*:\s*\n?",
        r"^Corrected\s+text\s*:\s*\n?",
        r"^Исправленный\s+текст\s*:\s*\n?",
    ]:
        result = re.sub(pat, "", result, flags=re.IGNORECASE).strip()
    if len(result) >= 2:
        for q in ('"', "'", "`"):
            if result.startswith(q) and result.endswith(q):
                result = result[1:-1].strip()
                break
    return result


_AI_SYMBOL_MAP = str.maketrans({
    "—": "-",     # — em dash
    "–": "-",     # – en dash
    "−": "-",     # − minus sign
    "‐": "-",     # ‐ hyphen
    "‑": "-",     # ‑ non-breaking hyphen
    "…": "...",   # … horizontal ellipsis
    "“": '"',     # " left double quote
    "”": '"',     # " right double quote
    "„": '"',     # „ low double quote
    "‟": '"',     # ‟ high reversed double quote
    "‘": "'",     # ' left single quote
    "’": "'",     # ' right single quote
    "‚": "'",     # ‚ low single quote
    "‛": "'",     # ‛ high reversed single quote
    " ": " ",     #   no-break space
    " ": " ",     #   narrow no-break space
    " ": " ",     #   thin space
    " ": " ",     #   figure space
    " ": " ",     #   punctuation space
    " ": " ",     #   hair space
    "​": "",      # zero-width space
    "‌": "",      # zero-width non-joiner
    "‍": "",      # zero-width joiner
    "﻿": "",      # zero-width no-break space (BOM)
    "ё": "е",
    "Ё": "Е",
})


def _strip_ai_symbols(text):
    """Replace typographic AI-symbols with plain ASCII equivalents."""
    if not text:
        return text
    return text.translate(_AI_SYMBOL_MAP)


def _spellcheck_process():
    """SpellCheck pipeline: capture selected text -> LLM fix -> paste back."""
    if not _spellcheck_lock.acquire(blocking=False):
        print("SpellCheck: already processing, skipping")
        return
    try:
        _fire_event("spellcheck_started")

        # Wait for hotkey modifiers to be physically released before sending Ctrl+C
        # (uses Win32 GetAsyncKeyState directly, does not touch keyboard hooks).
        try:
            _get_state = ctypes.windll.user32.GetAsyncKeyState
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                if not any(_get_state(vk) & 0x8000 for vk in _ALL_MODIFIERS):
                    break
                time.sleep(0.02)
        except Exception:
            time.sleep(0.3)
        _release_stuck_modifiers()

        # Save clipboard and capture selected text
        saved_clipboard = _save_clipboard()
        old_text = _get_clipboard_text()

        _send_copy()

        # Poll clipboard for changes (up to ~2s for slow apps)
        new_text = old_text
        for i in range(40):
            time.sleep(0.05)
            new_text = _get_clipboard_text()
            if new_text != old_text:
                break

        if new_text == old_text or not new_text.strip():
            _logger.info("SpellCheck capture failed: clipboard unchanged (old=%r, len=%d)",
                         (old_text or "")[:60], len(old_text or ""))
            print("SpellCheck: no text selected")
            _restore_clipboard(saved_clipboard)
            _fire_event("spellcheck_done", {"text": "", "changed": False})
            return

        print(f"SpellCheck: captured {len(new_text)} chars")
        _logger.info("SpellCheck captured: %s", new_text[:120])

        # Detect language
        lang = _detect_language(new_text)

        # Choose prompt
        if SPELLCHECK_PROMPT is not None:
            prompt_template = SPELLCHECK_PROMPT
        elif lang.startswith("ru"):
            prompt_template = SPELLCHECK_PROMPT_RU
        else:
            prompt_template = SPELLCHECK_PROMPT_EN

        if SPELLCHECK_CLEAN_PROFANITY:
            prof_rule = _PROFANITY_RULE_RU if lang.startswith("ru") else _PROFANITY_RULE_EN
        else:
            prof_rule = ""
        prompt = prompt_template.format(detected_lang=lang, raw_text=new_text, profanity_rule=prof_rule)

        # Send to LLM
        print(f"SpellCheck: LLM ({LLM_BACKEND}, lang={lang})...")
        t0 = time.time()
        try:
            if LLM_BACKEND == "openai":
                result = _llm_request_openai(prompt)
            else:
                result = _llm_request_ollama(prompt)
            result = _clean_llm_response(result)
            if LLM_STRIP_AI_SYMBOLS:
                result = _strip_ai_symbols(result)
            elapsed = time.time() - t0
            print(f"SpellCheck: LLM ({elapsed:.1f}s): {result[:120]}")
            _logger.info("SpellCheck LLM %.1fs: %s", elapsed, result[:120])
        except Exception as e:
            print(f"SpellCheck: LLM error: {e}")
            _logger.error("SpellCheck LLM error: %s", e)
            _restore_clipboard(saved_clipboard)
            _fire_event("spellcheck_done", {"text": new_text, "changed": False, "error": str(e)})
            return

        # Skip if no changes or empty
        if not result or result.strip() == new_text.strip():
            print("SpellCheck: no changes needed")
            _restore_clipboard(saved_clipboard)
            _fire_event("spellcheck_done", {"text": new_text, "changed": False})
            return

        # Paste corrected text
        _set_clipboard_text(result)
        fg_process = _get_foreground_process_name() if os.name == "nt" else ""
        actual_method = _resolve_paste_method(fg_process)
        _send_paste(actual_method)
        time.sleep(0.3)

        # Restore original clipboard
        _restore_clipboard(saved_clipboard)

        # Sound notification
        import winsound
        winsound.MessageBeep(winsound.MB_OK)

        print(f"SpellCheck: done! ({actual_method})")
        _logger.info("SpellCheck pasted (%s): %s", actual_method, result[:120])
        _fire_event("spellcheck_done", {"text": result, "changed": True})

    except Exception as e:
        _logger.error("SpellCheck error: %s", e, exc_info=True)
        print(f"SpellCheck error: {e}")
        _fire_event("spellcheck_done", {"text": "", "error": str(e)})
    finally:
        _spellcheck_lock.release()


def _on_spellcheck_key(_event=None):
    """SpellCheck hotkey handler (on key press). Checks modifier and spawns processing."""
    try:
        if not SPELLCHECK_ENABLED or _recording:
            return
        if SPELLCHECK_MODIFIER is not None and not keyboard.is_pressed(SPELLCHECK_MODIFIER):
            return
        # SpellCheck does not need the mic but counts as activity (resets prebuffer idle timer).
        # Don't wake the worker — SpellCheck processes selected text via LLM, not audio.
        global _last_activity_ts
        _last_activity_ts = time.monotonic()
        threading.Thread(target=_spellcheck_process, daemon=True).start()
    except Exception as e:
        _logger.error("SpellCheck hotkey handler error: %s", e, exc_info=True)


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
        line(f"     Microphone: {get_active_device_name()}") + "\n",
        line(f"     Anti-repeat: ngram={WHISPER_NO_REPEAT_NGRAM_SIZE}, penalty={WHISPER_REPETITION_PENALTY}, halluc_thr={WHISPER_HALLUCINATION_SILENCE_THRESHOLD}") + "\n",
    ]
    if PASTE_TO_ACTIVE_WINDOW:
        parts.append((line(f'     Keys after paste: "{KEYS_AFTER_PASTE.upper()}"') if KEYS_AFTER_PASTE else line("     Keys after paste: -")) + "\n")
    sc_status = f'ON ("{SPELLCHECK_HOTKEY.upper()}")' if SPELLCHECK_ENABLED else "OFF"
    parts.append(line(f"     SpellCheck: {sc_status}") + "\n")
    parts.extend([line("") + "\n", line('     "CTRL+C" to exit') + "\n", "╚" + "═" * w + "╝"])
    return "".join(parts)


# -----------------------------------------------------------------------------
# Init / shutdown / config API (used by GUI; console mode calls main())
# -----------------------------------------------------------------------------

def init_whisper():
    """Load the Whisper model. Blocking, may take a while on first run (download)."""
    global _whisper_model
    print("⏳ Loading Whisper model... (first run may download the model)")
    _whisper_model = WhisperModel(
        WHISPER_MODEL,
        device="cuda",
        compute_type="float16",
    )
    print("✅ Whisper loaded!")


def init_audio():
    """Initialize PyAudio, prebuffer deque, and start prebuffer worker thread."""
    global _pyaudio_instance, _prebuffer_deque, _prebuffer_running, _prebuffer_thread
    global _last_activity_ts
    _prebuffer_running = True
    _last_activity_ts = time.monotonic()
    _wake_event.clear()
    _pyaudio_instance = pyaudio.PyAudio()
    _prebuffer_deque = collections.deque(maxlen=_prebuffer_size())
    if PREBUFFER_MODE == "timeout":
        print(f"🎧 Prebuffer active (last {PREBUFFER_SEC}s; releases mic after {PREBUFFER_IDLE_TIMEOUT_SEC}s idle)")
    else:
        print(f"🎧 Prebuffer active (last {PREBUFFER_SEC}s; mode=always, mic stays open)")
    _prebuffer_thread = threading.Thread(target=prebuffer_worker, daemon=True)
    _prebuffer_thread.start()


def switch_microphone(device_name=None):
    """Switch microphone on the fly. device_name: device name or 'default'. None = re-read from config."""
    global AUDIO_DEVICE
    if device_name is not None:
        AUDIO_DEVICE = device_name
    _mic_switch_event.set()


def register_hotkeys():
    """Register keyboard hotkey hooks for push-to-talk and spellcheck."""
    _suppress = HOTKEY_KEY in ("alt", "pause")
    keyboard.on_press_key(HOTKEY_KEY, _on_hotkey_press, suppress=_suppress)
    keyboard.on_release_key(HOTKEY_KEY, _on_hotkey_release, suppress=_suppress)
    # SpellCheck hotkey (same pattern as PTT: on_press_key + modifier check)
    if SPELLCHECK_ENABLED:
        keyboard.on_press_key(SPELLCHECK_KEY, _on_spellcheck_key, suppress=False)


def unregister_hotkeys():
    """Remove keyboard hotkey hooks."""
    try:
        keyboard.unhook_all()
    except Exception:
        pass


def shutdown():
    """Stop prebuffer worker, terminate PyAudio. Call before exit."""
    global _prebuffer_running
    _prebuffer_running = False
    _wake_event.set()  # break the SLEEPING wait so worker exits promptly
    if _prebuffer_thread and _prebuffer_thread.is_alive():
        _prebuffer_thread.join(timeout=3)
    if _pyaudio_instance:
        try:
            _pyaudio_instance.terminate()
        except Exception:
            pass


def reload_config():
    """Re-read .env and update module globals. Returns dict of changed keys.
    Some settings require restart: WHISPER_MODEL (model reload), SAMPLE_RATE/CHUNK_SIZE (audio restart)."""
    global WHISPER_LANGUAGE, WHISPER_INITIAL_PROMPT
    global USE_LLM_TRANSFORM, LLM_BACKEND, LLM_MODEL, LLM_URL, LLM_API_KEY, LLM_REASONING_EFFORT, LLM_THINKING
    global LLM_FALLBACK_ENABLED, LLM_FALLBACK_URL, LLM_FALLBACK_MODEL, LLM_FALLBACK_API_KEY, LLM_FALLBACK_THINKING
    global LLM_TIMEOUT, LLM_FALLBACK_STICKY_SEC
    global LLM_CLEAN_PROFANITY, LLM_STRIP_AI_SYMBOLS, LLM_TRANSFORM_PROMPT
    global COPY_TO_CLIPBOARD, PASTE_TO_ACTIVE_WINDOW, PASTE_METHOD
    global CLIPBOARD_AFTER_PASTE_POLICY, KEYS_AFTER_PASTE
    global PREBUFFER_SEC, PADDING_SEC, MIN_FRAMES, SILENCE_AMPLITUDE_THRESHOLD
    global PREBUFFER_MODE, PREBUFFER_IDLE_TIMEOUT_SEC
    global CHUNK_DURATION_SEC, CHUNK_OVERLAP_SEC
    global SHOW_NOTIFICATIONS, LOG_ENABLED, _log_handler
    global AUDIO_DEVICE
    global SPELLCHECK_ENABLED, SPELLCHECK_HOTKEY, SPELLCHECK_MODIFIER, SPELLCHECK_KEY
    global SPELLCHECK_LANGUAGE, SPELLCHECK_CLEAN_PROFANITY, SPELLCHECK_PROMPT

    # Re-read .env
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path, override=True)
    except ImportError:
        pass

    old = get_config()

    # Instant-apply settings
    WHISPER_LANGUAGE = _env("WHISPER_LANGUAGE", "en")
    WHISPER_INITIAL_PROMPT = _env("WHISPER_INITIAL_PROMPT", "English speech.")
    USE_LLM_TRANSFORM = _env("USE_LLM_TRANSFORM", "false", type_=bool)
    LLM_BACKEND = _env("LLM_BACKEND", "ollama").strip().lower()
    LLM_MODEL = _env("LLM_MODEL", _env("OLLAMA_MODEL", "gemma3:12b"))
    LLM_URL = _env("LLM_URL", _env("OLLAMA_URL",
        "http://localhost:11434/api/generate" if LLM_BACKEND == "ollama"
        else "http://localhost:1234/v1/chat/completions"))
    LLM_API_KEY = _env("LLM_API_KEY", "")
    LLM_REASONING_EFFORT = _env("LLM_REASONING_EFFORT", "none").strip().lower()
    LLM_THINKING = _env("LLM_THINKING", "").strip().lower()
    LLM_FALLBACK_ENABLED = _env("LLM_FALLBACK_ENABLED", "false", type_=bool)
    LLM_FALLBACK_URL = _env("LLM_FALLBACK_URL", "")
    LLM_FALLBACK_MODEL = _env("LLM_FALLBACK_MODEL", "")
    LLM_FALLBACK_API_KEY = _env("LLM_FALLBACK_API_KEY", "")
    LLM_FALLBACK_THINKING = _env("LLM_FALLBACK_THINKING", "").strip().lower()
    LLM_TIMEOUT = _env("LLM_TIMEOUT", "15", type_=int)
    LLM_FALLBACK_STICKY_SEC = _env("LLM_FALLBACK_STICKY_SEC", "60", type_=int)
    LLM_CLEAN_PROFANITY = _env("LLM_CLEAN_PROFANITY", "true", type_=bool)
    LLM_STRIP_AI_SYMBOLS = _env("LLM_STRIP_AI_SYMBOLS", "true", type_=bool)
    LLM_TRANSFORM_PROMPT = _get_llm_prompt()
    COPY_TO_CLIPBOARD = _env("COPY_TO_CLIPBOARD", "true", type_=bool)
    PASTE_TO_ACTIVE_WINDOW = _env("PASTE_TO_ACTIVE_WINDOW", "true", type_=bool)
    PASTE_METHOD = _env("PASTE_METHOD", "auto").strip().lower()
    CLIPBOARD_AFTER_PASTE_POLICY = _env("CLIPBOARD_AFTER_PASTE_POLICY", "restore").strip().lower()
    KEYS_AFTER_PASTE = _env("KEYS_AFTER_PASTE", "enter").strip().lower()
    if KEYS_AFTER_PASTE in ("", "none"):
        KEYS_AFTER_PASTE = None
    PREBUFFER_SEC = _env("PREBUFFER_SEC", "0.5", type_=float)
    PADDING_SEC = _env("PADDING_SEC", "0.2", type_=float)
    MIN_FRAMES = _env("MIN_FRAMES", "5", type_=int)
    SILENCE_AMPLITUDE_THRESHOLD = _env("SILENCE_AMPLITUDE", "750", type_=int)
    new_prebuffer_mode = _env("PREBUFFER_MODE", "timeout").strip().lower()
    if new_prebuffer_mode not in ("always", "timeout"):
        new_prebuffer_mode = "timeout"
    PREBUFFER_MODE = new_prebuffer_mode
    PREBUFFER_IDLE_TIMEOUT_SEC = _env("PREBUFFER_IDLE_TIMEOUT_SEC", "1800", type_=int)
    CHUNK_DURATION_SEC = _env("CHUNK_DURATION_SEC", "15", type_=float)
    CHUNK_OVERLAP_SEC = _env("CHUNK_OVERLAP_SEC", "2.0", type_=float)
    AUDIO_DEVICE = _env("AUDIO_DEVICE", "default").strip()

    SPELLCHECK_ENABLED = _env("SPELLCHECK_ENABLED", "true", type_=bool)
    SPELLCHECK_HOTKEY = _env("SPELLCHECK_HOTKEY", "ctrl+t").strip().lower().replace(" ", "")
    if "+" in SPELLCHECK_HOTKEY:
        _sc_parts = SPELLCHECK_HOTKEY.split("+", 1)
        SPELLCHECK_MODIFIER, SPELLCHECK_KEY = _sc_parts[0].strip(), _sc_parts[1].strip()
    else:
        SPELLCHECK_MODIFIER, SPELLCHECK_KEY = None, SPELLCHECK_HOTKEY
    SPELLCHECK_LANGUAGE = _env("SPELLCHECK_LANGUAGE", "auto").strip().lower()
    SPELLCHECK_CLEAN_PROFANITY = _env("SPELLCHECK_CLEAN_PROFANITY", "true", type_=bool)
    SPELLCHECK_PROMPT = _get_spellcheck_prompt()

    SHOW_NOTIFICATIONS = _env("SHOW_NOTIFICATIONS", "true", type_=bool)
    new_log = _env("LOG_ENABLED", "false", type_=bool)
    if new_log != LOG_ENABLED:
        LOG_ENABLED = new_log
        for h in list(_logger.handlers):
            _logger.removeHandler(h)
        if _log_handler:
            _log_handler.close()
            _log_handler = None
        _setup_logging()

    new = get_config()
    changed = {k: new[k] for k in new if old.get(k) != new[k]}
    if changed:
        print(f"🔄 Config reloaded: {', '.join(changed.keys())}")
    return changed


def get_config():
    """Return dict of current configuration values."""
    return {
        "WHISPER_MODEL": WHISPER_MODEL,
        "WHISPER_LANGUAGE": WHISPER_LANGUAGE,
        "WHISPER_INITIAL_PROMPT": WHISPER_INITIAL_PROMPT,
        "WHISPER_NO_REPEAT_NGRAM_SIZE": WHISPER_NO_REPEAT_NGRAM_SIZE,
        "WHISPER_REPETITION_PENALTY": WHISPER_REPETITION_PENALTY,
        "WHISPER_HALLUCINATION_SILENCE_THRESHOLD": WHISPER_HALLUCINATION_SILENCE_THRESHOLD,
        "HOTKEY": HOTKEY,
        "USE_LLM_TRANSFORM": USE_LLM_TRANSFORM,
        "LLM_BACKEND": LLM_BACKEND,
        "LLM_MODEL": LLM_MODEL,
        "LLM_URL": LLM_URL,
        "LLM_API_KEY": LLM_API_KEY,
        "LLM_REASONING_EFFORT": LLM_REASONING_EFFORT,
        "LLM_THINKING": LLM_THINKING,
        "LLM_FALLBACK_ENABLED": LLM_FALLBACK_ENABLED,
        "LLM_FALLBACK_URL": LLM_FALLBACK_URL,
        "LLM_FALLBACK_MODEL": LLM_FALLBACK_MODEL,
        "LLM_FALLBACK_API_KEY": LLM_FALLBACK_API_KEY,
        "LLM_FALLBACK_THINKING": LLM_FALLBACK_THINKING,
        "LLM_TIMEOUT": LLM_TIMEOUT,
        "LLM_FALLBACK_STICKY_SEC": LLM_FALLBACK_STICKY_SEC,
        "LLM_CLEAN_PROFANITY": LLM_CLEAN_PROFANITY,
        "LLM_STRIP_AI_SYMBOLS": LLM_STRIP_AI_SYMBOLS,
        "COPY_TO_CLIPBOARD": COPY_TO_CLIPBOARD,
        "PASTE_TO_ACTIVE_WINDOW": PASTE_TO_ACTIVE_WINDOW,
        "PASTE_METHOD": PASTE_METHOD,
        "CLIPBOARD_AFTER_PASTE_POLICY": CLIPBOARD_AFTER_PASTE_POLICY,
        "KEYS_AFTER_PASTE": KEYS_AFTER_PASTE,
        "SAMPLE_RATE": SAMPLE_RATE,
        "CHUNK_SIZE": CHUNK_SIZE,
        "PREBUFFER_SEC": PREBUFFER_SEC,
        "PADDING_SEC": PADDING_SEC,
        "MIN_FRAMES": MIN_FRAMES,
        "SILENCE_AMPLITUDE": SILENCE_AMPLITUDE_THRESHOLD,
        "PREBUFFER_MODE": PREBUFFER_MODE,
        "PREBUFFER_IDLE_TIMEOUT_SEC": PREBUFFER_IDLE_TIMEOUT_SEC,
        "CHUNK_DURATION_SEC": CHUNK_DURATION_SEC,
        "CHUNK_OVERLAP_SEC": CHUNK_OVERLAP_SEC,
        "AUDIO_DEVICE": AUDIO_DEVICE,
        "LOG_ENABLED": LOG_ENABLED,
        "SHOW_NOTIFICATIONS": SHOW_NOTIFICATIONS,
        "SPELLCHECK_ENABLED": SPELLCHECK_ENABLED,
        "SPELLCHECK_HOTKEY": SPELLCHECK_HOTKEY,
        "SPELLCHECK_LANGUAGE": SPELLCHECK_LANGUAGE,
        "SPELLCHECK_CLEAN_PROFANITY": SPELLCHECK_CLEAN_PROFANITY,
    }


def main():
    """Console mode: init everything, register hotkeys, block forever."""
    init_whisper()
    init_audio()

    print(_format_banner())
    print(f'👂 Listening - hold "{HOTKEY.upper()}" to start recording.')

    register_hotkeys()

    # Block forever - exit only via Ctrl+C
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        unregister_hotkeys()
        shutdown()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
        raise SystemExit(0)
