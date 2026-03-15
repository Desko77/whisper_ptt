# Whisper-PTT

**Stupidly Simple Local Push-to-Talk** Voice-to-Text You Can Review Over a Coffee. Powered by Whisper. Fully offline — nothing leaves your machine.

Hold a hotkey, speak, release. Transcribed text is pasted into the active window (or copied to clipboard). Optionally, a local LLM (Ollama) can transform the output — fix grammar, reformat as an email, translate, or anything else you can describe in a prompt.

<p align="center"><img src="assets/demo_apple.gif" alt="Whisper-PTT demo" width="820"></p>

---

## Platforms

Single script per platform — so you know exactly what you're running.

GPU acceleration is required — there is no CPU-only mode. One self-contained script per platform:

| Platform | Script | Whisper backend | Accelerator |
|----------|--------|-----------------|-------------|
| Windows / Linux | `whisper_ptt_cuda.py` | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | NVIDIA CUDA |
| macOS | `whisper_ptt_apple_silicon.py` | [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) | Apple Silicon (Metal) |

---

## Quick start

### Windows / Linux (NVIDIA CUDA)

```bash
git clone https://github.com/sancau/whisper-ptt.git
cd whisper-ptt
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements-cuda.txt
cp .env.example-cuda .env       # edit as needed
python whisper_ptt_cuda.py
```

On Linux, the `keyboard` library needs root for global hotkeys: `sudo python whisper_ptt_cuda.py`.

### macOS (Apple Silicon)

```bash
git clone https://github.com/sancau/whisper-ptt.git
cd whisper-ptt
python -m venv venv
source venv/bin/activate
pip install -r requirements-apple-silicon.txt
cp .env.example-apple-silicon .env   # edit as needed
python whisper_ptt_apple_silicon.py
```

macOS will prompt for **Accessibility** and **Input Monitoring** permissions on first run. Grant those and run **without** `sudo`. The Whisper model is downloaded from HuggingFace on first launch.

### LLM transform (optional, off by default)

Supports two backends for LLM post-processing:

- **Ollama** — native Ollama API
- **OpenAI-compatible** — LM Studio, llama.cpp server, or any OpenAI-compatible endpoint

```bash
# Option 1: Ollama
ollama pull gemma3:12b

# Option 2: LM Studio — download a model via LM Studio UI, start the server
```

Set `WHISPER_PTT_USE_LLM_TRANSFORM=true` and configure the backend in `.env`:

```bash
# Ollama (default)
WHISPER_PTT_LLM_BACKEND=ollama
WHISPER_PTT_LLM_MODEL=gemma3:12b
WHISPER_PTT_LLM_URL=http://localhost:11434/api/generate

# OpenAI-compatible (LM Studio, llama.cpp, etc.)
WHISPER_PTT_LLM_BACKEND=openai
WHISPER_PTT_LLM_MODEL=google/gemma-3-12b
WHISPER_PTT_LLM_URL=http://localhost:1234/v1/chat/completions
```

The default prompt cleans up grammar and filler words; replace it via `WHISPER_PTT_LLM_TRANSFORM_PROMPT` to translate, reformat, or do anything else.

---

## Configuration

All settings are read from `WHISPER_PTT_*` environment variables or a `.env` file. Both scripts use the same variable names.

| Variable | Description | Default |
|----------|-------------|---------|
| `WHISPER_PTT_WHISPER_MODEL` | Whisper model (`tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo`) | `large-v3` / `large-v3-turbo` |
| `WHISPER_PTT_WHISPER_LANGUAGE` | Language code (`en`, `ru`, `de`, `fr`, ...) | `en` |
| `WHISPER_PTT_HOTKEY` | Hold-to-record key. Combos like `alt+f12` work. | `alt` / `option` |
| `WHISPER_PTT_USE_LLM_TRANSFORM` | Enable LLM transform | `false` |
| `WHISPER_PTT_LLM_BACKEND` | LLM backend: `ollama` or `openai` (LM Studio, llama.cpp) | `ollama` |
| `WHISPER_PTT_LLM_MODEL` | LLM model name | `gemma3:12b` |
| `WHISPER_PTT_LLM_URL` | LLM API URL | auto by backend |
| `WHISPER_PTT_COPY_TO_CLIPBOARD` | Copy result to clipboard | `true` |
| `WHISPER_PTT_PASTE_TO_ACTIVE_WINDOW` | Paste into focused window | `true` |
| `WHISPER_PTT_PASTE_METHOD` | Paste method: `ctrl+v`, `ctrl+shift+v`, `shift+insert` | `ctrl+v` |
| `WHISPER_PTT_CLIPBOARD_AFTER_PASTE_POLICY` | After paste: `restore`, `clear`, or `preserve` | `restore` |
| `WHISPER_PTT_KEYS_AFTER_PASTE` | Key(s) to send after paste (`enter`, `ctrl+enter`, `none`) | `enter` |

<details>
<summary>Advanced settings</summary>

| Variable | Description | Default |
|----------|-------------|---------|
| `WHISPER_PTT_WHISPER_INITIAL_PROMPT` | Whisper initial prompt (language hint) | `English speech.` |
| `WHISPER_PTT_LLM_API_KEY` | API key for OpenAI-compatible servers (if required) | — |
| `WHISPER_PTT_LLM_TRANSFORM_PROMPT` | Custom LLM prompt (`{detected_lang}`, `{raw_text}` placeholders) | built-in |
| `WHISPER_PTT_SAMPLE_RATE` | Audio sample rate (Hz) | `16000` |
| `WHISPER_PTT_CHUNK_SIZE` | Audio chunk size | `1024` |
| `WHISPER_PTT_PREBUFFER_SEC` | Prebuffer duration (captures the first word) | `0.5` |
| `WHISPER_PTT_PADDING_SEC` | Silence padding before transcription | `0.2` |
| `WHISPER_PTT_MIN_FRAMES` | Min frames to process (skip accidental taps) | `5` |
| `WHISPER_PTT_SILENCE_AMPLITUDE` | Amplitude below which audio is treated as silence | `750` |

</details>

---

## Usage

Default hotkey: **Alt** (Windows/Linux) or **Option** (macOS). Hold to record, release to transcribe. Exit with Ctrl+C. The defaults are convenient but can interfere with other shortcuts — consider remapping to `pause` or a combo like `option+f2` via `WHISPER_PTT_HOTKEY`.

With LLM transform enabled, the raw transcription is passed through Ollama before pasting. What the LLM does is entirely controlled by the prompt — grammar cleanup, email formatting, translation, or any other text transformation.

---

## License

MIT — see [LICENSE](LICENSE).
