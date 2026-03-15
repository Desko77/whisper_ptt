"""Quick benchmark: compare LLM response time between local and remote server."""

import time
import requests

LOCAL_URL = "http://localhost:1234/v1/chat/completions"
REMOTE_URL = "http://192.168.11.150:1234/v1/chat/completions"
MODEL = "google/gemma-3-12b"

# Simulates a typical whisper-ptt transcription cleanup request
PROMPT = """Fix the following speech-to-text transcription. Rules:
- Fix grammar, punctuation, and capitalization
- Remove filler words (um, uh, like, etc.)
- Keep the original language (en)
- Keep the original meaning — do NOT add or change content
- If it's already clean, return as-is
- Return ONLY the cleaned text, nothing else

Transcription: So um I was thinking about like the the project and uh we need to basically finish the um the documentation by friday and also like make sure that the tests are all passing before we can uh merge it into main"""

ROUNDS = 3


def query(url, label):
    try:
        t0 = time.time()
        r = requests.post(
            url,
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": PROMPT}],
                "temperature": 0.1,
                "max_tokens": 256,
                "stream": False,
            },
            timeout=60,
        )
        elapsed = time.time() - t0
        data = r.json()
        text = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", "?")
        return elapsed, text, completion_tokens
    except Exception as e:
        return None, str(e), 0


def main():
    print(f"Model: {MODEL}")
    print(f"Rounds: {ROUNDS}")
    print(f"Prompt length: {len(PROMPT)} chars")
    print("=" * 70)

    for label, url in [("LOCAL ", LOCAL_URL), ("REMOTE", REMOTE_URL)]:
        times = []
        print(f"\n--- {label} ({url}) ---")
        for i in range(ROUNDS):
            elapsed, text, tokens = query(url, label)
            if elapsed is None:
                print(f"  Round {i+1}: FAILED - {text}")
                continue
            times.append(elapsed)
            tps = tokens / elapsed if isinstance(tokens, int) and tokens > 0 else "?"
            print(f"  Round {i+1}: {elapsed:.2f}s, {tokens} tokens, ~{tps:.1f} tok/s" if tps != "?" else f"  Round {i+1}: {elapsed:.2f}s")
            if i == 0:
                print(f"  Output: {text[:120]}...")

        if times:
            avg = sum(times) / len(times)
            best = min(times)
            print(f"  Avg: {avg:.2f}s | Best: {best:.2f}s")

    print("\n" + "=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()
