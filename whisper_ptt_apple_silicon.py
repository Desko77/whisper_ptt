#!/usr/bin/env python3
"""
Whisper-PTT (Apple Silicon): push-to-talk voice-to-text using mlx-whisper on Metal.
Hold hotkey -> speak -> release -> transcription pasted into the active window.

Config: WHISPER_PTT_* env vars or .env file (see .env.example-apple-silicon).

Dependencies: mlx-whisper, pyaudio, keyboard, pyperclip, requests.
Optional: Ollama for LLM transform.
"""

import os
import logging
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
# Replace typographic AI-symbols in LLM output with plain ASCII equivalents
# (em/en dash -> '-', curly quotes -> straight, ellipsis -> '...', ё -> е, nbsp -> space)
LLM_STRIP_AI_SYMBOLS = _env("LLM_STRIP_AI_SYMBOLS", "true", type_=bool)
DEFAULT_LLM_TRANSFORM_PROMPT = """Fix the following speech-to-text transcription. Rules:
- Fix punctuation, capitalization, and obvious grammar errors
- Remove filler words (um, uh, like, etc.)
- When in doubt — do NOT change. Better to leave as-is than to break it
- Do NOT rephrase — preserve word order and sentence structure
- Keep technical terms, names, and domain-specific vocabulary as-is
- Keep the original language ({detected_lang})
- If it's already clean, return as-is
- Return ONLY the cleaned text, no explanations

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

_logger = logging.getLogger("whisper_ptt")
_logger.setLevel(logging.DEBUG)
_log_handler = None


def _setup_logging():
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
    return LOG_FILE


# -----------------------------------------------------------------------------
# MLX model name -> HuggingFace repo mapping
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
_prebuffer_thread = None
_pyaudio_instance = None
_mic_switch_event = threading.Event()  # signals prebuffer to reopen stream
_mlx_model_path = None
_transcribe_queue = queue.Queue()
_model_ready = threading.Event()
_transcription_thread = None
_hotkey_listener = None

# Chunked transcription state
_chunk_results = []            # [(chunk_index, raw_text, lang)]
_chunk_results_lock = threading.Lock()
_chunk_index = 0
_next_chunk_frame = 0          # frame index where next chunk starts
_chunking_active = False
_prev_chunk_tail_text = ""     # last ~30 words for initial_prompt context
_chunk_transcribe_queue = queue.Queue()  # chunks enqueued during recording

# GUI callbacks (optional - None in console mode)
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
# Audio: prebuffer and numpy conversion
# -----------------------------------------------------------------------------

_MIC_MAX_RETRIES = 5
_MIC_RETRY_DELAY = 1.0
_wake_event = threading.Event()        # PTT press signals worker to wake from SLEEPING
_last_activity_ts = time.monotonic()   # updated on PTT press; idle timer reference


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
                    "name": info["name"],
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

    Callers should compare by name too — on some backends the index can remain
    stable while the underlying device changes.
    """
    pa = None
    try:
        pa = pyaudio.PyAudio()
        info = pa.get_default_input_device_info()
        return (info["index"], info["name"])
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
            return info["name"]
        except (OSError, AttributeError):
            return "Default"
    try:
        info = _pyaudio_instance.get_device_info_by_index(idx)
        return info["name"]
    except (OSError, AttributeError):
        return AUDIO_DEVICE


def switch_microphone(device_name=None):
    """Switch microphone on the fly. device_name: device name or 'default'. None = re-read from config."""
    global AUDIO_DEVICE
    if device_name is not None:
        AUDIO_DEVICE = device_name
    _mic_switch_event.set()


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


def _open_microphone_stream():
    """Open mic stream, retrying with PyAudio re-init on transient PortAudio errors.

    When AUDIO_DEVICE="default", resolves the current system default to an EXPLICIT
    device index at open time. This pins the stream to a real physical device so
    stream.read() will raise OSError reliably if the device disappears.
    """
    global _pyaudio_instance
    device_index = _resolve_device_index()
    if device_index is None:
        try:
            device_index = _pyaudio_instance.get_default_input_device_info()["index"]
        except (OSError, KeyError):
            device_index = None
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
            try:
                if device_index is not None:
                    opened_name = _pyaudio_instance.get_device_info_by_index(device_index)["name"]
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
            device_index = _resolve_device_index()
            if device_index is None:
                try:
                    device_index = _pyaudio_instance.get_default_input_device_info()["index"]
                except (OSError, KeyError):
                    device_index = None
    raise RuntimeError("Could not open microphone after retries — check audio permissions and device.")


def prebuffer_worker():
    """Background thread: read mic into ring buffer; when recording, also append to _audio_frames.

    Two states:
    - ACTIVE   : stream is open, ring buffer being filled.
    - SLEEPING : stream is closed (PREBUFFER_MODE=timeout and idle > timeout).
                 Worker waits on _wake_event; PTT press wakes it.

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
        # ---- SLEEPING state: wait for wake signal (PTT) or shutdown ----
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
    elapsed = time.time() - t0
    print(f"📝 Whisper ({elapsed:.1f}s): {text}")
    _logger.info("Whisper %.1fs: %s", elapsed, text)
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
    elapsed = time.time() - t0
    print(f"   📝 Chunk ({elapsed:.1f}s): {text[:80]}{'...' if len(text) > 80 else ''}")
    _logger.info("Chunk %.1fs: %s", elapsed, text[:120])
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
    " ": " ",     #   no-break space
    " ": " ",     #   narrow no-break space
    " ": " ",     #   thin space
    " ": " ",     #   figure space
    " ": " ",     #   punctuation space
    " ": " ",     #   hair space
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
        _fire_event("transcription_done", {"text": text})
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
        _logger.info("Pasted: %s", text[:120])
        if CLIPBOARD_AFTER_PASTE_POLICY == "restore":
            pyperclip.copy(old)
        elif CLIPBOARD_AFTER_PASTE_POLICY == "clear":
            pyperclip.copy("")
    _fire_event("transcription_done", {"text": text})


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
        _reset_chunk_state()
        return

    _fire_event("processing_started")

    if not _chunking_active:
        # Short recording - single-pass (existing behavior)
        _reset_chunk_state()
        _transcribe_queue.put(frames)
    else:
        # Long recording - extract final chunk and assemble all
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


# -----------------------------------------------------------------------------
# Init / shutdown / config API (used by GUI; console mode calls main())
# -----------------------------------------------------------------------------

def init_whisper():
    """Start MLX transcription worker and wait for model to load. Blocking."""
    global _transcription_thread
    _transcription_thread = threading.Thread(target=_transcription_worker, daemon=True)
    _transcription_thread.start()
    _model_ready.wait()


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


def register_hotkeys():
    """Start pynput hotkey listener in a background thread (non-blocking)."""
    global _hotkey_listener
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
        if n in ("cmd", "command"):
            return Key.cmd
        if n in ("option", "opt", "alt"):
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
        if HOTKEY_MODIFIER is None:
            if _matches(key, hotkey_key_spec):
                _on_hotkey_release()
        else:
            if _matches(key, hotkey_key_spec) or _matches(key, hotkey_mod_spec):
                _on_hotkey_release()
        pressed.discard(key)

    _hotkey_listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
    _hotkey_listener.start()


def unregister_hotkeys():
    """Stop pynput hotkey listener."""
    global _hotkey_listener
    if _hotkey_listener:
        try:
            _hotkey_listener.stop()
        except Exception:
            pass
        _hotkey_listener = None


def shutdown():
    """Stop prebuffer worker, transcription worker, terminate PyAudio."""
    global _prebuffer_running
    _prebuffer_running = False
    _wake_event.set()  # break the SLEEPING wait so worker exits promptly
    if _prebuffer_thread and _prebuffer_thread.is_alive():
        _prebuffer_thread.join(timeout=3)
    # Signal transcription worker to stop
    _transcribe_queue.put(None)
    if _transcription_thread and _transcription_thread.is_alive():
        _transcription_thread.join(timeout=5)
    if _pyaudio_instance:
        try:
            _pyaudio_instance.terminate()
        except Exception:
            pass


def reload_config():
    """Re-read .env and update module globals. Returns dict of changed keys."""
    global WHISPER_LANGUAGE, WHISPER_INITIAL_PROMPT
    global USE_LLM_TRANSFORM, LLM_BACKEND, LLM_MODEL, LLM_URL, LLM_API_KEY, LLM_STRIP_AI_SYMBOLS, LLM_TRANSFORM_PROMPT
    global COPY_TO_CLIPBOARD, PASTE_TO_ACTIVE_WINDOW
    global CLIPBOARD_AFTER_PASTE_POLICY, KEYS_AFTER_PASTE
    global PREBUFFER_SEC, PADDING_SEC, MIN_FRAMES, SILENCE_AMPLITUDE_THRESHOLD
    global PREBUFFER_MODE, PREBUFFER_IDLE_TIMEOUT_SEC
    global CHUNK_DURATION_SEC, CHUNK_OVERLAP_SEC
    global SHOW_NOTIFICATIONS, LOG_ENABLED, _log_handler
    global AUDIO_DEVICE

    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path, override=True)
    except ImportError:
        pass

    old = get_config()

    WHISPER_LANGUAGE = _env("WHISPER_LANGUAGE", "en")
    WHISPER_INITIAL_PROMPT = _env("WHISPER_INITIAL_PROMPT", "English speech.")
    USE_LLM_TRANSFORM = _env("USE_LLM_TRANSFORM", "false", type_=bool)
    LLM_BACKEND = _env("LLM_BACKEND", "ollama").strip().lower()
    LLM_MODEL = _env("LLM_MODEL", _env("OLLAMA_MODEL", "gemma3:12b"))
    LLM_URL = _env("LLM_URL", _env("OLLAMA_URL",
        "http://localhost:11434/api/generate" if LLM_BACKEND == "ollama"
        else "http://localhost:1234/v1/chat/completions"))
    LLM_API_KEY = _env("LLM_API_KEY", "")
    LLM_STRIP_AI_SYMBOLS = _env("LLM_STRIP_AI_SYMBOLS", "true", type_=bool)
    LLM_TRANSFORM_PROMPT = _env("LLM_TRANSFORM_PROMPT", DEFAULT_LLM_TRANSFORM_PROMPT)
    COPY_TO_CLIPBOARD = _env("COPY_TO_CLIPBOARD", "true", type_=bool)
    PASTE_TO_ACTIVE_WINDOW = _env("PASTE_TO_ACTIVE_WINDOW", "true", type_=bool)
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
        "HOTKEY": HOTKEY,
        "USE_LLM_TRANSFORM": USE_LLM_TRANSFORM,
        "LLM_BACKEND": LLM_BACKEND,
        "LLM_MODEL": LLM_MODEL,
        "LLM_URL": LLM_URL,
        "LLM_API_KEY": LLM_API_KEY,
        "LLM_STRIP_AI_SYMBOLS": LLM_STRIP_AI_SYMBOLS,
        "COPY_TO_CLIPBOARD": COPY_TO_CLIPBOARD,
        "PASTE_TO_ACTIVE_WINDOW": PASTE_TO_ACTIVE_WINDOW,
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
    }


def main():
    """Console mode: init everything, register hotkeys, block forever."""
    init_whisper()
    init_audio()

    print(_format_banner())
    print(f'👂 Listening - hold "{HOTKEY.upper()}" to start recording.')

    # In console mode, use blocking listener (original behavior)
    _start_hotkey_listener_mac()


def _start_hotkey_listener_mac():
    """Blocking hotkey listener for console mode."""
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
        if n in ("cmd", "command"):
            return Key.cmd
        if n in ("option", "opt", "alt"):
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


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
        raise SystemExit(0)
