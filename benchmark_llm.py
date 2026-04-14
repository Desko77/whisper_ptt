#!/usr/bin/env python3
"""Compare LLM candidates for whisper_ptt post-processing.

Runs several models side-by-side on realistic EN + RU transcription-cleanup
prompts (same rules as DEFAULT_LLM_TRANSFORM_PROMPT[_RU] in whisper_ptt_cuda.py).
For each model it does a warmup call, then N timed rounds, and prints latency,
tokens/sec, plus the model's actual output so you can eyeball quality.

Usage:
    # Pull the candidates first, e.g. for Ollama:
    #   ollama pull gemma3:12b
    #   ollama pull gemma4:e4b
    #   ollama pull qwen3:4b
    #   ollama pull qwen3:8b

    python benchmark_llm.py
    python benchmark_llm.py --models gemma3:12b,gemma4:e4b,qwen3:4b --rounds 3
    python benchmark_llm.py --backend openai --url http://localhost:1234/v1/chat/completions \\
        --models "google/gemma-3-12b,google/gemma-4-e4b,qwen/qwen3-4b"
    python benchmark_llm.py --skip-ru              # English only
    python benchmark_llm.py --no-think             # prepend /no_think for Qwen/Gemma4 on Ollama

Config defaults are read from .env (WHISPER_PTT_LLM_BACKEND / _URL / _API_KEY /
_REASONING_EFFORT). CLI flags override.
"""

import argparse
import os
import sys
import time

import requests

# ---------------------------------------------------------------------------
# Config (.env)
# ---------------------------------------------------------------------------

_script_dir = os.path.dirname(os.path.abspath(__file__))
_env_path = os.path.join(_script_dir, ".env")
try:
    from dotenv import load_dotenv
    load_dotenv(_env_path)
except ImportError:
    pass


def _env(key, default):
    full_key = key if key.startswith("WHISPER_PTT_") else f"WHISPER_PTT_{key}"
    return os.environ.get(full_key, os.environ.get(key, default))


# ---------------------------------------------------------------------------
# Prompt templates — copied verbatim from whisper_ptt_cuda.py
# ---------------------------------------------------------------------------

PROMPT_EN = """Fix the following speech-to-text transcription. Rules:
- Fix punctuation, capitalization, and obvious grammar errors
- Remove filler words (um, uh, like, etc.)
- When in doubt — do NOT change. Better to leave as-is than to break it
- Do NOT rephrase — preserve word order and sentence structure
- Keep technical terms, names, and domain-specific vocabulary as-is
- Keep the original language ({detected_lang}) — do NOT transliterate to Latin script
- If it's already clean, return as-is
- Return ONLY the cleaned text, no explanations

Transcription: {raw_text}"""

PROMPT_RU = """Исправь следующую расшифровку речи. Правила:
- Исправь пунктуацию, заглавные буквы и явные грамматические ошибки
- Убери слова-паразиты (эм, ну, типа, вот, короче и т.д.)
- При сомнении — НЕ меняй. Лучше оставить как есть, чем испортить
- НЕ перефразируй — сохраняй порядок слов и структуру предложения
- Технические термины, названия и специальную лексику оставляй как есть
- Пиши ТОЛЬКО на русском языке, НЕ транслитерируй в латиницу
- Верни ТОЛЬКО исправленный текст, без пояснений

Расшифровка: {raw_text}"""


# Realistic short PTT-style transcriptions. Mix of filler-heavy, clean, and
# technical cases — the last catch models that unnecessarily rephrase or
# translate identifiers.
TEST_CASES = [
    ("EN filler", "en",
     "so um I was thinking about like the the project and uh we need to basically "
     "finish the um the documentation by friday and also like make sure that the "
     "tests are all passing before we can uh merge it into main"),
    ("EN clean", "en",
     "The deployment pipeline requires three stages: build, test, and release."),
    ("EN technical", "en",
     "we should use faster-whisper with large-v3 on CUDA float16 and set beam_size to 1 um yeah"),
    ("RU filler", "ru",
     "ну короче вот я подумал что нам эээ надо типа переделать этот ну интерфейс "
     "потому что он вот как бы не очень удобный и короче надо добавить темную тему"),
    ("RU clean", "ru",
     "Нам необходимо завершить документацию до пятницы и убедиться, что все тесты проходят."),
    ("RU technical", "ru",
     "короче надо в whisper_ptt_cuda.py поменять faster-whisper на gigaam v3 rnnt ну или оставить как есть"),
]


DEFAULT_MODELS = ["gemma3:12b", "gemma4:e4b", "qwen3:4b", "qwen3:8b"]


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------


def call_ollama(url, model, prompt, max_tokens, timeout):
    r = requests.post(
        url,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": max_tokens},
        },
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    text = (data.get("response") or "").strip()
    comp_tokens = data.get("eval_count") or 0
    return text, comp_tokens


def call_openai(url, model, prompt, api_key, reasoning_effort, max_tokens, timeout):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if reasoning_effort and reasoning_effort != "off":
        body["reasoning_effort"] = reasoning_effort
    r = requests.post(url, headers=headers, json=body, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    text = data["choices"][0]["message"]["content"].strip()
    comp_tokens = data.get("usage", {}).get("completion_tokens") or 0
    return text, comp_tokens


def call_model(backend, url, model, prompt, api_key, reasoning_effort, timeout):
    # Mirror whisper_ptt_cuda.py: num_predict/max_tokens = len(prompt) * 2
    max_tokens = len(prompt) * 2
    if backend == "openai":
        return call_openai(url, model, prompt, api_key, reasoning_effort, max_tokens, timeout)
    return call_ollama(url, model, prompt, max_tokens, timeout)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def format_prompt(case_lang, raw_text, no_think):
    template = PROMPT_RU if case_lang == "ru" else PROMPT_EN
    prompt = template.format(detected_lang=case_lang, raw_text=raw_text)
    if no_think:
        # Qwen 3 / Gemma 4 directive. Inert for models that don't know it.
        prompt = "/no_think\n\n" + prompt
    return prompt


def strip_think_tags(text):
    """Remove <think>...</think> blocks some models emit before the answer."""
    while "<think>" in text and "</think>" in text:
        start = text.index("<think>")
        end = text.index("</think>") + len("</think>")
        text = text[:start] + text[end:]
    return text.strip()


def run_case(backend, url, model, api_key, reasoning_effort, case, rounds, timeout, no_think):
    """Warmup + N timed rounds. Return dict with times, tok_per_s, output or error."""
    _, case_lang, raw_text = case
    prompt = format_prompt(case_lang, raw_text, no_think)

    try:
        call_model(backend, url, model, prompt, api_key, reasoning_effort, timeout)
    except Exception as e:
        return {"error": f"warmup failed: {e}"}

    times, tps_list, last = [], [], ""
    for i in range(rounds):
        try:
            t0 = time.time()
            text, comp_tokens = call_model(
                backend, url, model, prompt, api_key, reasoning_effort, timeout,
            )
            elapsed = time.time() - t0
            times.append(elapsed)
            if comp_tokens and elapsed > 0:
                tps_list.append(comp_tokens / elapsed)
            last = strip_think_tags(text)
        except Exception as e:
            return {"error": f"round {i + 1}: {e}"}

    return {"times": times, "tps": tps_list, "output": last}


def main():
    parser = argparse.ArgumentParser(
        description="Compare LLM candidates for whisper_ptt post-processing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:", 1)[-1],
    )
    default_backend = _env("LLM_BACKEND", "ollama").strip().lower()
    default_url_ollama = "http://localhost:11434/api/generate"
    default_url_openai = "http://localhost:1234/v1/chat/completions"
    default_url = _env(
        "LLM_URL",
        _env("OLLAMA_URL", default_url_ollama if default_backend == "ollama" else default_url_openai),
    )
    default_api_key = _env("LLM_API_KEY", "")
    default_reasoning = _env("LLM_REASONING_EFFORT", "none").strip().lower()

    parser.add_argument("--backend", choices=("ollama", "openai"), default=default_backend)
    parser.add_argument("--url", default=default_url)
    parser.add_argument("--api-key", default=default_api_key)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS),
                        help="comma-separated list of models to compare")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120,
                        help="per-request timeout in seconds (default: 120)")
    parser.add_argument("--reasoning-effort", default=default_reasoning,
                        help="reasoning for OpenAI-compatible backends: none|low|medium|high|off")
    parser.add_argument("--no-think", action="store_true",
                        help="prepend /no_think to prompts (Qwen 3 / Gemma 4 directive)")
    parser.add_argument("--skip-ru", action="store_true", help="skip Russian cases")
    parser.add_argument("--skip-en", action="store_true", help="skip English cases")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    cases = [
        c for c in TEST_CASES
        if not (args.skip_ru and c[1] == "ru")
        and not (args.skip_en and c[1] == "en")
    ]

    if not models:
        print("No models specified.", file=sys.stderr)
        sys.exit(1)
    if not cases:
        print("No test cases selected.", file=sys.stderr)
        sys.exit(1)

    print(f"Backend:   {args.backend}")
    print(f"URL:       {args.url}")
    print(f"Models:    {', '.join(models)}")
    print(f"Rounds:    {args.rounds}  (+1 warmup per case)")
    print(f"Cases:     {len(cases)}  "
          f"({sum(1 for c in cases if c[1] == 'en')} EN, "
          f"{sum(1 for c in cases if c[1] == 'ru')} RU)")
    print(f"No-think:  {args.no_think}")
    if args.backend == "openai":
        print(f"Reasoning: {args.reasoning_effort}")
    print("=" * 78)

    # results[(model, case_name)] = {"times"|"tps"|"output"|"error"}
    results = {}

    for model in models:
        print(f"\n### {model}")
        for case in cases:
            case_name, case_lang, raw_text = case
            print(f"\n  [{case_name}] in: {raw_text[:90]}{'...' if len(raw_text) > 90 else ''}")
            res = run_case(
                args.backend, args.url, model, args.api_key,
                args.reasoning_effort, case, args.rounds, args.timeout,
                args.no_think,
            )
            results[(model, case_name)] = res
            if "error" in res:
                print(f"    ERROR: {res['error']}")
                continue
            avg_t = sum(res["times"]) / len(res["times"])
            best_t = min(res["times"])
            tps_str = f"{sum(res['tps']) / len(res['tps']):.1f} tok/s" if res["tps"] else "tok/s n/a"
            print(f"    avg={avg_t:.2f}s  best={best_t:.2f}s  {tps_str}")
            out = res["output"]
            print(f"    out: {out[:220]}{'...' if len(out) > 220 else ''}")

    # ------- Summary table (avg latency per case) -------
    print("\n" + "=" * 78)
    print("\nSummary — average latency per case (seconds):\n")
    col_w = max(14, max((len(m) for m in models), default=14) + 2)
    header = "case".ljust(18) + "".join(m.ljust(col_w) for m in models)
    print(header)
    print("-" * len(header))
    for case in cases:
        name = case[0]
        row = name.ljust(18)
        for m in models:
            res = results.get((m, name), {})
            if "error" in res:
                row += "ERROR".ljust(col_w)
            elif res.get("times"):
                row += f"{sum(res['times']) / len(res['times']):.2f}s".ljust(col_w)
            else:
                row += "-".ljust(col_w)
        print(row)

    # ------- Per-model averages -------
    print("\nPer-model average across all cases:\n")
    for m in models:
        all_times = []
        all_tps = []
        errors = 0
        for c in cases:
            res = results.get((m, c[0]), {})
            if "error" in res:
                errors += 1
            else:
                all_times.extend(res.get("times", []))
                all_tps.extend(res.get("tps", []))
        if all_times:
            avg = sum(all_times) / len(all_times)
            tps = f"{sum(all_tps) / len(all_tps):.1f} tok/s" if all_tps else "tok/s n/a"
            err_note = f"  ({errors} cases failed)" if errors else ""
            print(f"  {m.ljust(col_w)} avg={avg:.2f}s  {tps}{err_note}")
        else:
            print(f"  {m.ljust(col_w)} all cases failed")

    print(
        "\nTips:\n"
        "  • Don't pick on speed alone — scroll up and eyeball the 'out:' lines.\n"
        "    A 'fast' model that rephrases or drops words is worse than a slow one.\n"
        "  • Qwen 3 / Gemma 4 on Ollama default to thinking mode. Re-run with --no-think\n"
        "    for a fair speed comparison (they emit <think>...</think> and take 5-10x longer).\n"
        "  • On OpenAI-compatible backends use --reasoning-effort none instead.\n"
        "  • 404 on Ollama = model not pulled: run 'ollama pull <model>' first.\n"
    )


if __name__ == "__main__":
    main()
