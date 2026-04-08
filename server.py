#!/usr/bin/env python3
"""
Web server for the interactive story roleplay app.
Run: python server.py
Then open http://localhost:5000
"""

import io
import json
import random
import re
import struct
import threading
import urllib.request
import urllib.error
from pathlib import Path
from flask import Flask, request, Response, send_file, stream_with_context

from story import analyze_story, build_system_prompt, list_models, OLLAMA_URL

STORIES_DIR = Path(__file__).parent / "stories"
KOKORO_MODEL = Path(__file__).parent / "kokoro_models" / "kokoro-v1.0.onnx"
KOKORO_VOICES = Path(__file__).parent / "kokoro_models" / "voices-v1.0.bin"

# Narrator: British female — distinguished, literary quality
NARRATOR_VOICE = "bf_emma"
NARRATOR_LANG  = "en-gb"

# Male character: British male
MALE_CHAR_VOICE = "bm_george"
MALE_CHAR_LANG  = "en-gb"

# Default female character voice (British)
CHARACTER_VOICE = "bf_isabella"
CHARACTER_LANG  = "en-gb"

# ── Voice selection by name/setting origin ────────────────────
# Each entry: (keywords, voice, lang)
# Checked against character name and story setting (case-insensitive).
_ORIGIN_VOICE_MAP = [
    # Japanese
    (["japan", "tokyo", "kyoto", "osaka", "yoko", "yuki", "hana", "aiko",
      "akiko", "haruki", "sakura", "keiko", "naomi", "rin", "yui", "asahi",
      "mizuki", "natsuki", "setsuko", "tomoko", "yoshiko"],
     "jf_alpha", "ja"),
    # French
    (["france", "paris", "lyon", "marseille", "bordeaux", "provence",
      "marie", "camille", "celine", "céline", "amélie", "amelie", "isabelle",
      "margot", "claire", "elise", "élise", "colette", "brigitte", "juliette"],
     "ff_siwis", "fr-fr"),
    # Italian
    (["italy", "italian", "rome", "roma", "milan", "milano", "venice",
      "florence", "firenze", "naples", "napoli",
      "giulia", "chiara", "valentina", "francesca", "elena", "sofia",
      "alessia", "beatrice", "aurora", "ginevra"],
     "if_sara", "it"),
    # Spanish
    (["spain", "spanish", "madrid", "barcelona", "seville", "sevilla",
      "mexico", "argentina", "colombia", "chile",
      "sofia", "isabella", "carmen", "lucia", "lucía", "catalina",
      "pilar", "consuelo", "dolores", "rosario", "paloma", "lola"],
     "ef_dora", "es"),
    # Portuguese / Brazilian
    (["brazil", "brasil", "portugal", "lisbon", "lisboa", "porto",
      "rio", "são paulo", "sao paulo",
      "ana", "beatriz", "bruna", "fernanda", "larissa", "leticia",
      "mariana", "natalia", "natália", "patricia", "patrícia"],
     "pf_dora", "pt-br"),
    # Chinese
    (["china", "chinese", "beijing", "shanghai", "guangzhou", "chengdu",
      "hong kong", "taiwan", "taipei",
      "mei", "xia", "lan", "fang", "ying", "jing", "qian", "yan",
      "xiaomei", "xiaoling", "xiaoyu", "lingling"],
     "zf_xiaoxiao", "zh"),
    # Hindi / Indian
    (["india", "indian", "delhi", "mumbai", "bangalore", "kolkata",
      "priya", "ananya", "aarti", "arti", "deepa", "divya", "geeta",
      "kavya", "lakshmi", "meena", "nisha", "pooja", "puja", "radha",
      "rekha", "rina", "sita", "sunita", "usha"],
     "hf_alpha", "hi"),
    # British (explicit)
    (["england", "london", "scotland", "wales", "ireland", "uk",
      "british", "victorian", "edwardian"],
     "bf_lily", "en-gb"),
    # American
    (["america", "american", "usa", "new york", "los angeles", "chicago",
      "texas", "california", "southern", "midwest"],
     "af_heart", "en-us"),
]


def _pick_char_voice(profile: dict) -> tuple[str, str]:
    """Return (voice, lang) for the female character based on name + setting."""
    name    = profile.get("name", "").lower()
    setting = profile.get("setting", "").lower()
    haystack = f"{name} {setting}"

    for keywords, voice, lang in _ORIGIN_VOICE_MAP:
        if any(kw in haystack for kw in keywords):
            return voice, lang

    return CHARACTER_VOICE, CHARACTER_LANG  # default: British

app = Flask(__name__)

# ── Kokoro TTS (lazy-loaded, thread-safe) ─────────────────────
_kokoro = None
_kokoro_lock = threading.Lock()

def get_kokoro():
    global _kokoro
    if _kokoro is None:
        with _kokoro_lock:
            if _kokoro is None:
                from kokoro_onnx import Kokoro
                _kokoro = Kokoro(str(KOKORO_MODEL), str(KOKORO_VOICES))
    return _kokoro


def _float32_to_wav(samples, sample_rate: int = 24000) -> bytes:
    """Convert float32 numpy array to WAV bytes without scipy."""
    import numpy as np
    pcm = np.clip(samples, -1.0, 1.0)
    pcm_int16 = (pcm * 32767).astype(np.int16)
    data = pcm_int16.tobytes()
    buf = io.BytesIO()
    num_channels = 1
    bits = 16
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(data)))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<H", 1))                          # PCM
    buf.write(struct.pack("<H", num_channels))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", sample_rate * num_channels * bits // 8))
    buf.write(struct.pack("<H", num_channels * bits // 8))
    buf.write(struct.pack("<H", bits))
    buf.write(b"data")
    buf.write(struct.pack("<I", len(data)))
    buf.write(data)
    return buf.getvalue()


_SPEECH_VERBS = r"(?:said|replied|answered|whispered|murmured|called|asked|demanded|muttered|added|continued|shouted|growled|breathed)"

def _parse_segments(text: str, male_name: str = "") -> list[tuple[str, str]]:
    """
    Split narrator text into (segment, speaker) tuples.
    speaker: "narrator" | "female" | "male"
    Quoted text is attributed by checking the preceding narrator clause for
    'he <verb>' or '<male_name> <verb>' patterns.
    Also strips *action* markers so they're read naturally.
    """
    # Remove HTML/markdown artifacts from the text
    clean = re.sub(r"\*([^*\n]+)\*", r"\1", text)  # strip *action* markers
    clean = re.sub(r"<[^>]+>", "", clean)           # strip any HTML tags

    # Build a pattern that matches male attribution before a quote
    name_alts = "|".join(filter(None, ["he", re.escape(male_name) if male_name and male_name.lower() not in ("you", "") else ""]))
    male_attr_re = re.compile(
        rf'\b(?:{name_alts})\s+{_SPEECH_VERBS}\b',
        re.IGNORECASE,
    )

    segments: list[tuple[str, str]] = []
    pattern = re.compile(r'"[^"]{1,500}"')
    last = 0
    last_narr = ""
    for m in pattern.finditer(clean):
        narr = clean[last:m.start()].strip()
        if narr:
            segments.append((narr, "narrator"))
            last_narr = narr
        else:
            last_narr = ""
        # Attribute dialogue to male if the preceding narrator text says so
        speaker = "male" if male_attr_re.search(last_narr) else "female"
        segments.append((m.group(), speaker))
        last = m.end()
    tail = clean[last:].strip()
    if tail:
        segments.append((tail, "narrator"))
    return segments


# ── In-memory session store (single user) ─────────────────────
session = {
    "messages": [],
    "profile": None,
    "model": "romance:latest",
    "char_voice": CHARACTER_VOICE,
    "char_lang":  CHARACTER_LANG,
    "male_name":  "You",
}


@app.route("/")
def index():
    return send_file("index.html")


@app.route("/models")
def models():
    return {"models": list_models()}


@app.route("/start", methods=["POST"])
def start():
    data = request.json
    model = data.get("model", "romance:latest")

    stories = list(STORIES_DIR.glob("*.txt"))
    if not stories:
        return {"error": f"No .txt files found in {STORIES_DIR}"}, 400

    chosen = random.choice(stories)
    story_text = chosen.read_text(encoding="utf-8", errors="replace")

    try:
        profile, story_context = analyze_story(story_text, model)
    except Exception as e:
        return {"error": str(e)}, 500

    if profile.get("name", "Her").strip().lower() in ("her", "she", ""):
        profile["name"] = random.choice([
            "Amelia", "Clara", "Elena", "Isla", "Lyra",
            "Mara", "Nora", "Rose", "Sarah", "Vera",
        ])

    system_prompt = build_system_prompt(profile, story_context)

    print("\n" + "═" * 60)
    print(f"  STORY: {chosen.name}")
    print("═" * 60)
    print(json.dumps(profile, indent=2, ensure_ascii=False))
    print("═" * 60 + "\n")

    char_voice, char_lang = _pick_char_voice(profile)

    session["model"] = model
    session["profile"] = profile
    session["messages"] = [{"role": "system", "content": system_prompt}]
    session["char_voice"] = char_voice
    session["char_lang"]  = char_lang
    session["male_name"]  = profile.get("male_name", "You")

    print(f"  Voice: {char_voice} ({char_lang})\n")

    return {
        "name": profile.get("name", "Her"),
        "male_name": profile.get("male_name", "You"),
        "setting": profile.get("setting", ""),
        "stage": profile.get("relationship_stage", ""),
        "summary": profile.get("story_summary", ""),
        "personality": profile.get("personality", []),
    }


def _ollama_stream(messages):
    """Shared generator: streams tokens from Ollama, yields SSE lines, appends to session."""
    payload = json.dumps({
        "model": session["model"],
        "messages": messages,
        "stream": True,
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    full = []
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for line in resp:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        full.append(token)
                        yield f"data: {json.dumps({'token': token})}\n\n"
                    if chunk.get("done"):
                        break
                except json.JSONDecodeError:
                    continue
    except urllib.error.URLError as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        return

    assembled = "".join(full)
    session["messages"].append({"role": "assistant", "content": assembled})
    yield f"data: {json.dumps({'done': True})}\n\n"


@app.route("/open", methods=["POST"])
def open_story():
    trigger_msg = {"role": "user", "content": (
        "[Begin the story. 3–4 paragraphs maximum. "
        "Open in medias res — the scene is already in motion. "
        "Weave in one or two vivid physical details about her naturally as the scene unfolds; do not front-load a description block. "
        "End on a single unresolved beat: she says one thing or does one thing that demands a response. Stop there.]"
    )}
    messages = session["messages"] + [trigger_msg]
    return Response(
        stream_with_context(_ollama_stream(messages)),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "").strip()

    if not user_input:
        return {"error": "Empty message"}, 400
    if not session["messages"]:
        return {"error": "No active story. Call /start first."}, 400

    session["messages"].append({"role": "user", "content": user_input})
    return Response(
        stream_with_context(_ollama_stream(session["messages"])),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/tts", methods=["POST"])
def tts():
    data = request.json or {}
    text = data.get("text", "").strip()
    if not text:
        return {"error": "No text"}, 400

    try:
        import numpy as np
        kokoro = get_kokoro()
        segments = _parse_segments(text, male_name=session.get("male_name", ""))

        SR = 24000
        silence = np.zeros(int(SR * 0.18), dtype=np.float32)
        parts = []

        for seg_text, speaker in segments:
            seg_text = seg_text.strip()
            if not seg_text:
                continue
            if speaker == "male":
                voice, lang = MALE_CHAR_VOICE, MALE_CHAR_LANG
            elif speaker == "female":
                voice, lang = session["char_voice"], session["char_lang"]
            else:  # narrator
                voice, lang = NARRATOR_VOICE, NARRATOR_LANG
            samples, _ = kokoro.create(seg_text, voice=voice, speed=1.0, lang=lang)
            parts.append(samples)
            parts.append(silence)

        if not parts:
            return {"error": "Nothing to synthesize"}, 400

        combined = np.concatenate(parts)
        wav_bytes = _float32_to_wav(combined, SR)

        return Response(wav_bytes, mimetype="audio/wav",
                        headers={"Cache-Control": "no-cache"})

    except Exception as e:
        return {"error": str(e)}, 500


if __name__ == "__main__":
    print("Story Roleplay Server")
    print("Open http://localhost:5000 in your browser")
    # Pre-warm kokoro in background so first TTS call is fast
    threading.Thread(target=get_kokoro, daemon=True).start()
    app.run(debug=False, port=5000, threaded=True)
