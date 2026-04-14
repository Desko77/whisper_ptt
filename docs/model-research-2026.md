# Model Research 2026: STT and LLM Post‑processing

Consolidated research on whether to replace Whisper, swap the default post‑processing LLM (`gemma3:12b`), or fold STT into a single multimodal model (Gemma 4 audio). Compiled April 2026.

Triggered by a [vc.ru article about 30 voice AI engines](https://vc.ru/ai/2815088-obzor-30-golosovyh-ai-dvizhkov-i-sozdanie-perevodchika-bystree-google-meet) and follow‑up questions on Gemma 4.

---

## TL;DR

| Question | Verdict |
|---|---|
| Replace Whisper with another STT engine? | **No.** Whisper Large V3 stays. |
| Drop LLM post‑processing entirely? | **No.** `LLM Transform` is already opt‑in; `SpellCheck` is an independent, useful feature. |
| Swap the default LLM (`gemma3:12b`)? | **Yes, likely.** `gemma4:e4b` is the recommended target. Confirm with `benchmark_llm.py` on real hardware before changing the default. |
| Use Gemma 4 audio as a Whisper replacement? | **No.** ~3× worse WER than Whisper, tends to paraphrase, weak long‑form, RU unknown. |

---

## Project constraints (why the usual recommendations don't apply)

From `CLAUDE.md`:
- Offline only — no cloud STT/LLM.
- GPU required — no CPU‑only fallback.
- Two intentionally self‑contained single‑file scripts (`whisper_ptt_cuda.py`, `whisper_ptt_apple_silicon.py`).
- Primary users transcribe in RU and EN.

Any engine that's cloud‑only, CPU‑only, single‑language, or ships a heavy framework (NeMo, etc.) loses on principle.

---

## Part 1 — STT: should we replace Whisper?

### What the vc.ru article actually says

The article is about a **cloud streaming translator**, not offline PTT. Its STT recommendations:
- Picks **Deepgram Nova‑3** (cloud, ~$0.0059/min, latency <300 ms).
- Rejects **Groq Whisper** (503 errors, ~2.8 s latency).
- Provides no WER benchmarks, no Russian‑specific tests, no offline context.

Verdict: the article's STT advice doesn't transfer to this project.

### Independent benchmarks

| Model | RU WER | EN WER | Speed | Offline GPU | License |
|---|---|---|---|---|---|
| **Whisper Large V3** (current) | **4.2%** | 7.44% | 1× (baseline) | CUDA + Metal | MIT |
| GigaAM v3‑rnnt (Sber) | **3.3%** | — | CPU ~0.7 s | CPU only (ONNX) | MIT |
| NVIDIA Parakeet v3 (0.6 B) | 5.5% | 6.32% | ~10× faster | CUDA + MLX | CC‑BY‑4.0 |
| NVIDIA Canary Qwen 2.5 B | — | 5.63% | slower | CUDA | CC‑BY‑4.0 |
| Vosk | 13% | — | CPU | CPU | Apache 2.0 |

Sources: [Open ASR Leaderboard (HF)](https://huggingface.co/blog/open-asr-leaderboard), [Parakeet V3 vs Whisper on Mac](https://whispernotes.app/blog/parakeet-v3-default-mac-model), [Habr — GigaAM vs Whisper vs Vosk](https://habr.com/ru/articles/1002260/), [Northflank 2026 STT benchmarks](https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks).

### Why we don't switch

1. **Whisper Large V3 is already competitive on RU (4.2% WER).** Parakeet v3 is *worse* on RU (5.5%).
2. **GigaAM is the only model meaningfully better on RU (3.3%)** but:
   - CPU‑only (ONNX Runtime) — breaks the GPU‑only rule in `CLAUDE.md`.
   - Russian‑only — kills EN and auto‑detect (`info.language` in CUDA:686, `result["language"]` in Apple Silicon:608).
3. **Parakeet v3 is ~10× faster**, but on short PTT clips Whisper already returns in <1 s on GPU — no perceptible win. Cost: add NeMo (gigabytes, PyTorch branch), breaking the single‑file intent.
4. **Integration is already clean** (`_whisper_model.transcribe(...) → segments, info` at `whisper_ptt_cuda.py:676‑686`, `mlx_whisper.transcribe(...) → result["text"], result["language"]` at `whisper_ptt_apple_silicon.py:584‑590`). Swap cost is low if a future model becomes clearly better.

### If/when we want to verify empirically

Small change to `benchmark.py`:
- Add `--backend whisper|gigaam|parakeet`.
- GigaAM: `pip install onnx-asr`; `onnx_asr.load_model("gigaam-v3-e2e-rnnt").recognize(audio_float32, 16000)`.
- Parakeet v3: `pip install nemo_toolkit[asr]` or `parakeet-mlx` on Mac.
- Run on 10–20 of your own PTT clips (RU + EN) with known transcripts; compute WER via `jiwer`.
- Only switch if a candidate clearly beats Whisper on *your* audio in both WER and latency.

Critical files for a future swap:
- `whisper_ptt_cuda.py:25, :65‑72, :676‑691, :697‑712, :1460‑1469`
- `whisper_ptt_apple_silicon.py:23, :62‑64, :584‑596, :602‑614, :851‑857`
- `benchmark.py:86‑106`
- `requirements-cuda.txt`, `requirements-apple-silicon.txt`
- `.env.example-cuda`, `.env.example-apple-silicon`

---

## Part 2 — LLM post‑processing: keep it and swap the default

### Current state

Default: `gemma3:12b` (~8 GB Q4, both platforms). Two distinct features use the LLM:

| Feature | Default | Location |
|---|---|---|
| **LLM Transform** — cleans transcription (grammar, punctuation, fillers) | **OFF** (opt‑in) | `whisper_ptt_cuda.py:748`, `whisper_ptt_apple_silicon.py:652` |
| **SpellCheck** — select text → `Ctrl+T` → LLM fixes → paste back | **ON** | `whisper_ptt_cuda.py:1304` (CUDA only; macOS not implemented) |

Prompts are well‑crafted and conservative ("when in doubt — do NOT change"), with dedicated RU and EN versions. `temperature=0.1`, `max_tokens = len(prompt)*2`, synchronous calls, 30 s timeout, supports both Ollama and OpenAI‑compatible backends, plus `WHISPER_PTT_LLM_REASONING_EFFORT` for thinking models.

### Should we remove LLM post‑processing entirely?

No.
- **Transform** is already opt‑in — no user is forced to pay its cost.
- **SpellCheck** is independent of STT and is the more valuable of the two — removing it would be a regression.

### Candidates to replace `gemma3:12b` (April 2026)

| Model | Size | VRAM Q4 | RU | License | Notes |
|---|---|---|---|---|---|
| `gemma3:12b` (current) | 12 B | ~8 GB | good | Gemma ToS | baseline |
| **`gemma4:e4b`** | 8 B (4.5 B active) | ~5 GB | 140+ languages, better than g3 | **Apache 2.0** | thinking off via `REASONING_EFFORT=none` |
| `qwen3:4b` | 4 B | ~3 GB | 119 languages, strong RU | Apache 2.0 | ~50–60 tok/s on RTX 4090; thinking off via `/no_think` |
| `qwen3:8b` / `qwen3:14b` | 8/14 B | 5/9 GB | even better | Apache 2.0 | balanced quality/speed |
| `phi-4-mini` | 3.8 B | 3 GB | weak RU | MIT | fast but not bilingual enough |
| `vikhr` / `saiga` | 7 B | 5 GB | **best RU** | Apache 2.0 | weaker EN, smaller ecosystem |

Sources: [Ollama — gemma4 library](https://ollama.com/library/gemma4), [Local AI Master — SLM Guide 2026](https://localaimaster.com/blog/small-language-models-guide-2026), [ai.rs — Gemma 4 vs Qwen 3.5 vs Llama 4](https://ai.rs/ai-developer/gemma-4-vs-qwen-3-5-vs-llama-4-compared), [Vikhr paper](https://arxiv.org/abs/2405.13929), [SiliconFlow — Best open‑source LLM for Russian 2026](https://www.siliconflow.com/articles/en/best-open-source-LLM-for-Russian).

### Recommendation

Switch the default to **`gemma4:e4b`**:
- Same family as `gemma3:12b` → prompts behave similarly, no re‑tuning required.
- Smaller and faster (~5 GB vs ~8 GB).
- Newer, broader multilingual (140+ languages), better RU.
- **Apache 2.0** (Gemma 3 ships under the Gemma ToS, which imposes usage restrictions).
- Codebase is already Gemma‑4‑ready (`WHISPER_PTT_LLM_REASONING_EFFORT=none` in `.env.example-*`).

For users who value speed over maximum quality: **`qwen3:4b`** as an alternative preset.

### Verification

`benchmark_llm.py` was rewritten (commit [c61b8da](https://github.com/Desko77/whisper_ptt/commit/c61b8da)) to compare multiple LLMs side‑by‑side on 6 realistic PTT prompts (3 EN + 3 RU: filler‑heavy, clean, technical). It measures latency + tokens/s and prints each model's actual output so quality can be eyeballed.

```bash
ollama pull gemma3:12b gemma4:e4b qwen3:4b qwen3:8b

# default list
python benchmark_llm.py

# custom list, more rounds
python benchmark_llm.py --models gemma3:12b,gemma4:e4b,qwen3:4b --rounds 5

# important for Qwen 3 / Gemma 4 on Ollama — skips thinking mode (5-10× faster)
python benchmark_llm.py --no-think

# LM Studio instead of Ollama
python benchmark_llm.py --backend openai --url http://localhost:1234/v1/chat/completions \
  --models "google/gemma-3-12b,google/gemma-4-e4b"
```

After benchmarking on real hardware:
1. Update `.env.example-cuda` and `.env.example-apple-silicon` (`WHISPER_PTT_LLM_MODEL=gemma4:e4b`).
2. Update the LLM section of `README.md`.
3. Update the LLM‑model hint in `docs/images/settings-llm.png` if it references a specific model.

---

## Part 3 — Gemma 4 audio: can it replace Whisper?

The user raised a sensible hypothesis: Gemma 4 E2B/E4B have native audio input — could one multimodal model replace Whisper + LLM entirely? **No**, for the reasons below.

### Architecture

- Audio encoder compressed to 305 M parameters (down from Gemma 3n's 681 M).
- Frame duration 40 ms (down from 160 ms).
- Accepts raw audio at 16 kHz.
- Runs offline via llama.cpp / ONNX / MediaPipe. Ollama multimodal audio support is still immature.

### Quality (the blocker)

| Model | WER |
|---|---|
| Whisper Large | **~4.4%** |
| Gemma 3n 8B (audio) | **~13.0%** |

Gemma is ~3× worse at pure transcription. Source: [Medium — Whisper vs Gemma 3n audio scribe](https://medium.com/@ajjay.ferrari/one-model-or-two-my-on-device-speech-stack-debate-whisper-vs-gemma-3n-audio-scribe-4aca133258ff).

Worse: Gemma is a language model, so it does **intent inference** — it paraphrases what it heard rather than transcribing verbatim. For a dictation PTT tool this is a dealbreaker; users expect their exact words.

### Other limitations

- **Long audio breaks the KV cache.** Whisper's 30‑second windowing stays consistent over 20‑minute meetings; Gemma degrades fast.
- **Russian.** No official or community WER for RU Gemma 4 audio. Given Google's English‑centric audio training, regression from Whisper's 4.2% RU WER is almost guaranteed.
- **Ollama / LM Studio.** Native audio input in these runtimes is flaky. Google's official audio docs point to llama.cpp / ONNX / MediaPipe — switching would require rewriting the audio pipeline in both scripts.
- **Speed.** The 40 ms frames only speed up the encoder. The decoder is a full LLM, not a small specialized ASR decoder — no evidence Gemma audio is actually faster than `faster-whisper` / `mlx-whisper` on a GPU.

### Reasonable use case for Gemma audio (not now)

Unified STT + cleanup in one step — audio → already‑cleaned text. But the 13% WER *is* the transcription error, so there's nothing to clean up post‑hoc. And "rephrasing instead of transcribing" is a correctness bug, not a style preference. Not worth the pipeline rewrite for a regression.

Sources: [MindStudio — Gemma 4 Audio Encoder](https://www.mindstudio.ai/blog/gemma-4-audio-encoder-e2b-e4b-speech-recognition), [Google AI — Gemma Audio docs](https://ai.google.dev/gemma/docs/capabilities/audio), [arXiv 2506.13596 — Qwen/Gemma + Whisper in multilingual SpeechLLM](https://arxiv.org/html/2506.13596v1).

---

## Decisions

1. **Keep Whisper Large V3** as the STT engine. Re‑evaluate only if a new model beats it on RU WER *and* runs offline on GPU *and* preserves multilingual auto‑detect.
2. **Keep LLM post‑processing** (Transform opt‑in, SpellCheck on). Do not remove.
3. **Plan to switch default LLM to `gemma4:e4b`** pending `benchmark_llm.py` results on real hardware. `qwen3:4b` stays on the shortlist as a speed‑first alternative.
4. **Do not use Gemma 4 audio for STT.** Revisit only if a future Gemma version closes the WER gap below 2× Whisper and Ollama audio support stabilises.
