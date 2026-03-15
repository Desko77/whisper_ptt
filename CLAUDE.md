# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Whisper-PTT is a local push-to-talk voice-to-text tool. Hold a hotkey, speak, release — transcribed text is pasted into the active window. Fully offline, GPU-accelerated only. Optional Ollama-based LLM post-processing (grammar fix, translation, etc.).

## Architecture

Two independent single-file scripts — one per platform, no shared modules:

| Script | Platform | Whisper backend | Key input | Paste mechanism |
|--------|----------|-----------------|-----------|-----------------|
| `whisper_ptt_cuda.py` | Windows/Linux | `faster-whisper` (CUDA) | `keyboard` lib | `keyboard.send("ctrl+v")` |
| `whisper_ptt_apple_silicon.py` | macOS | `mlx-whisper` (Metal) | `pynput` | AppleScript `key code` |

Both scripts share the same pipeline and config schema but differ in platform-specific details:
- **Audio → WAV (CUDA) vs numpy (macOS)**: CUDA script builds WAV in-memory via `wave` module; macOS script converts PCM frames to float32 numpy array directly.
- **Threading model**: macOS uses a dedicated `_transcription_worker` thread with a queue (Metal requires same-thread access for MLX ops). CUDA script spawns a new daemon thread per transcription.
- **Hotkey handling**: macOS uses `pynput.keyboard.Listener` (no root needed, Option key support). CUDA uses `keyboard` lib (needs root on Linux).

### Pipeline (both scripts)

```
prebuffer_worker (ring buffer) → start_recording → stop_recording_and_process
  → silence/duration gate → transcribe (Whisper) → transform_with_llm (optional)
  → paste_to_front (clipboard + simulated Ctrl+V/Cmd+V)
```

## Running

```bash
# Windows/Linux (NVIDIA CUDA)
pip install -r requirements-cuda.txt
cp .env.example-cuda .env
python whisper_ptt_cuda.py

# macOS (Apple Silicon)
pip install -r requirements-apple-silicon.txt
cp .env.example-apple-silicon .env
python whisper_ptt_apple_silicon.py
```

Linux requires `sudo` for global hotkeys (`keyboard` lib limitation).

## Configuration

All config via `WHISPER_PTT_*` env vars or `.env` file. The `_env()` helper reads vars with type coercion. See `.env.example-cuda` / `.env.example-apple-silicon` for all available settings.

## Key Constraints

- No test suite exists.
- No build system or linter configured.
- No CPU-only fallback — GPU acceleration is required.
- Scripts are intentionally self-contained single files ("so you know exactly what you're running").
- When modifying one script, check if the same change applies to the other platform script.
