#!/usr/bin/env python3
"""
Translate an English story file to Japanese using a local Ollama model.

Usage:
    python translate_story.py stories/foo.txt
    python translate_story.py stories/foo.txt --model qwen-ja
    python translate_story.py stories/foo.txt --output stories/foo_ja.txt

Writes:
    - stories_ja/<name>.txt          (translated story)
    - stories_ja/<name>_glossary.json (name mapping, kept for reuse)
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen-ja"
CHUNK_TARGET = 2500     # chars — soft target, won't split mid-paragraph
NUM_CTX = 8192          # bump from Ollama's default 2048
TIMEOUT_SECONDS = 600


def ollama_chat(model: str, messages: list, num_ctx: int = NUM_CTX) -> str:
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"num_ctx": num_ctx, "temperature": 0.3},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
        data = json.loads(resp.read())
    return data["message"]["content"]


def chunk_paragraphs(text: str, target: int = CHUNK_TARGET) -> list[str]:
    """Split text into chunks of ~target chars on paragraph boundaries."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, buf = [], []
    size = 0
    for p in paragraphs:
        if size + len(p) > target and buf:
            chunks.append("\n\n".join(buf))
            buf, size = [], 0
        buf.append(p)
        size += len(p) + 2
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


def extract_glossary(text: str, model: str) -> dict:
    """Ask the model for a Name → Katakana mapping for the story."""
    sample = text[:6000]
    prompt = (
        "Read the following English text and extract every proper name: "
        "characters, places, distinctive nicknames. For each, provide the "
        "Japanese rendering you would use if translating this story — "
        "katakana for foreign names, existing kanji/hiragana for names that "
        "are already Japanese. Be consistent. Output ONLY a JSON object "
        "mapping the original name to its Japanese rendering, with no "
        "commentary or code fences. Example: "
        '{"Sarah": "サラ", "James": "ジェームズ"}\n\n'
        f"TEXT:\n{sample}"
    )
    raw = ollama_chat(model, [{"role": "user", "content": prompt}])
    raw = raw.strip()
    # Strip code fences if the model added them despite instructions.
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.rstrip().endswith("```"):
            raw = raw.rstrip()[:-3]
    # Find the first { and last } to tolerate stray text.
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        mapping = json.loads(raw[start:end + 1])
        return {k: v for k, v in mapping.items() if isinstance(v, str)}
    except json.JSONDecodeError:
        return {}


def translate_chunk(chunk: str, model: str, glossary: dict) -> str:
    glossary_block = ""
    if glossary:
        lines = "\n".join(f"  {k} → {v}" for k, v in glossary.items())
        glossary_block = (
            "Use these name renderings consistently throughout:\n"
            f"{lines}\n\n"
        )
    system = (
        "You are a literary translator rendering English fiction into natural, "
        "flowing Japanese. Your output reads like a Japanese novel, not a "
        "textbook translation. You preserve the author's register, emotional "
        "nuance, and sensory detail. You preserve ALL content — including "
        "intimate, sensual, or explicit passages — without softening, "
        "abbreviating, or censoring. You do not add commentary, explanations, "
        "footnotes, or romaji glosses. You output only the Japanese "
        "translation of the passage you are given."
    )
    user = (
        f"{glossary_block}"
        "Translate the following passage into natural literary Japanese. "
        "Use 「」 for dialogue. Keep narration in a consistent register "
        "(past-tense plain form / だ・である調 is usually right for fiction). "
        "Output only the translation.\n\n"
        f"PASSAGE:\n{chunk}"
    )
    raw = ollama_chat(model, [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])
    # Strip accidental code fences
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.rstrip().endswith("```"):
            raw = raw.rstrip()[:-3]
    return raw.strip()


def main():
    ap = argparse.ArgumentParser(description="Translate a story to Japanese via Ollama.")
    ap.add_argument("input", help="Path to English .txt story")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"Ollama model name (default: {DEFAULT_MODEL})")
    ap.add_argument("--output", default=None,
                    help="Output path (default: <input>_ja.txt next to input)")
    ap.add_argument("--glossary", default=None,
                    help="Path to existing glossary JSON (skip extraction)")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        print(f"Error: {src} not found", file=sys.stderr)
        sys.exit(1)

    stem = src.stem.removesuffix("_ja")
    ja_dir = Path(__file__).parent / "stories_ja"
    ja_dir.mkdir(exist_ok=True)
    out_path = Path(args.output) if args.output else ja_dir / f"{stem}.txt"
    gloss_path = ja_dir / f"{stem}_glossary.json"

    text = src.read_text(encoding="utf-8", errors="replace")
    print(f"Loaded {src.name}: {len(text):,} chars")

    # ── Glossary ──────────────────────────────────────────
    if args.glossary:
        glossary = json.loads(Path(args.glossary).read_text(encoding="utf-8"))
        print(f"Using supplied glossary: {len(glossary)} names")
    elif gloss_path.exists():
        glossary = json.loads(gloss_path.read_text(encoding="utf-8"))
        print(f"Reusing cached glossary at {gloss_path.name}: {len(glossary)} names")
    else:
        print("Extracting name glossary...")
        t0 = time.time()
        glossary = extract_glossary(text, args.model)
        print(f"  → {len(glossary)} names in {time.time() - t0:.1f}s")
        if glossary:
            gloss_path.write_text(json.dumps(glossary, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
            print(f"  saved to {gloss_path.name}")

    if glossary:
        for k, v in list(glossary.items())[:10]:
            print(f"    {k} → {v}")
        if len(glossary) > 10:
            print(f"    … ({len(glossary) - 10} more)")

    # ── Translate in chunks ───────────────────────────────
    chunks = chunk_paragraphs(text)
    print(f"\nSplit into {len(chunks)} chunks (~{CHUNK_TARGET} chars each)")
    translated = []
    for i, chunk in enumerate(chunks, 1):
        t0 = time.time()
        print(f"  [{i}/{len(chunks)}] {len(chunk):,} chars → ", end="", flush=True)
        try:
            ja = translate_chunk(chunk, args.model, glossary)
        except urllib.error.URLError as e:
            print(f"\nOllama error: {e}", file=sys.stderr)
            sys.exit(1)
        dt = time.time() - t0
        print(f"{len(ja):,} chars  ({dt:.1f}s)")
        translated.append(ja)

    out_text = "\n\n".join(translated)
    out_path.write_text(out_text, encoding="utf-8")
    print(f"\nWrote {out_path} ({len(out_text):,} chars)")


if __name__ == "__main__":
    main()
