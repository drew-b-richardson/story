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

# Primary character voices (main NPC and player)
PRIMARY_MALE_CHARACTER_VOICE   = "am_michael"
PRIMARY_MALE_CHARACTER_LANG    = "en-us"
PRIMARY_FEMALE_CHARACTER_VOICE = "af_heart"
PRIMARY_FEMALE_CHARACTER_LANG  = "en-us"

# Secondary character voices (side characters who enter the scene)
SECONDARY_MALE_CHARACTER_VOICE   = "bm_fable"
SECONDARY_MALE_CHARACTER_LANG    = "en-gb"
SECONDARY_FEMALE_CHARACTER_VOICE = "af_bella"
SECONDARY_FEMALE_CHARACTER_LANG  = "en-us"

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


# ── TTS segment parsing ───────────────────────────────────────
# Past-tense verbs used for NPC / secondary character attribution
_SPEECH_VERBS = r"(?:said|replied|answered|whispered|murmured|called|asked|demanded|muttered|added|continued|shouted|growled|breathed)"
# Present AND past tense for player (second-person narration uses present tense: "you say", "you ask")
_PLAYER_SPEECH_VERBS = r"(?:say|said|reply|replied|answer|answered|whisper|whispered|murmur|murmured|call|called|ask|asked|demand|demanded|mutter|muttered|add|added|continue|continued|shout|shouted|growl|growled|breathe|breathed)"
_NAMED_ATTR_RE = re.compile(rf'\b([A-Z][a-z]{{2,}})\s+{_SPEECH_VERBS}\b')


def _detect_speaker(narr: str, player_name: str, other_name: str, other_pronoun: str) -> str:
    """
    Classify the speaker of a quoted passage from surrounding narrator text.
    Returns: "player" | "other" | "secondary_male" | "secondary_female"

    Uses position-based priority: whichever attribution tag appears earliest in
    the text wins, so "she replied and you answered" correctly reads "she replied"
    rather than letting the fixed-order player check override it.
    """
    if not narr:
        return "other"

    candidates: list[tuple[int, str]] = []  # (match_start, speaker_type)

    # Player: "you said/asked/…" (second-person, present or past)
    for m in re.finditer(rf'\byou\s+{_PLAYER_SPEECH_VERBS}\b', narr, re.IGNORECASE):
        candidates.append((m.start(), "player"))

    # Player by name: "Andrew said" (third-person fallback)
    if player_name:
        for m in re.finditer(rf'\b{re.escape(player_name)}\s+{_SPEECH_VERBS}\b', narr, re.IGNORECASE):
            candidates.append((m.start(), "player"))

    # Primary NPC by name: "Amy said"
    if other_name:
        for m in re.finditer(rf'\b{re.escape(other_name)}\s+{_SPEECH_VERBS}\b', narr, re.IGNORECASE):
            candidates.append((m.start(), "other"))

    # Primary NPC by pronoun: "she said" / "he said"
    if other_pronoun:
        for m in re.finditer(rf'\b{re.escape(other_pronoun)}\s+{_SPEECH_VERBS}\b', narr, re.IGNORECASE):
            candidates.append((m.start(), "other"))

    # Secondary characters: any other capitalised name with a speech verb
    known = {n.lower() for n in (other_name, player_name, "you") if n}
    for m in _NAMED_ATTR_RE.finditer(narr):
        if m.group(1).lower() not in known:
            # Infer gender from pronouns near this name only (±40 chars),
            # so a female primary NPC's "she" doesn't bleed into unrelated names.
            start = max(0, m.start() - 40)
            end = min(len(narr), m.end() + 40)
            local = narr[start:end]
            has_she = bool(re.search(r'\bshe\b', local, re.IGNORECASE))
            has_he = bool(re.search(r'\bhe\b', local, re.IGNORECASE))
            if has_she and not has_he:
                candidates.append((m.start(), "secondary_female"))
            elif has_he and not has_she:
                candidates.append((m.start(), "secondary_male"))
            elif other_pronoun == "she":
                # Primary NPC is female; default secondary to male to avoid confusion
                candidates.append((m.start(), "secondary_male"))
            else:
                candidates.append((m.start(), "secondary_female"))

    # Opposite-gender pronoun (cannot be the primary NPC)
    if other_pronoun == "she":
        for m in re.finditer(rf'\bhe\s+{_SPEECH_VERBS}\b', narr, re.IGNORECASE):
            candidates.append((m.start(), "secondary_male"))
    elif other_pronoun == "he":
        for m in re.finditer(rf'\bshe\s+{_SPEECH_VERBS}\b', narr, re.IGNORECASE):
            candidates.append((m.start(), "secondary_female"))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    return "other"  # default to primary NPC


def _parse_segments(
    text: str,
    player_name: str = "",
    player_pronoun: str = "you",
    other_name: str = "",
    other_pronoun: str = "she",
) -> list[tuple[str, str]]:
    """
    Split narrator text into (segment, speaker) tuples.
    speaker: "narrator" | "other" | "player" | "secondary_male" | "secondary_female"
    Also strips *action* markers so they're read naturally.
    """
    clean = re.sub(r"\*([^*\n]+)\*", r"\1", text)
    clean = re.sub(r"<[^>]+>", "", clean)

    segments: list[tuple[str, str]] = []
    pattern = re.compile(r'"[^"]{1,500}"')
    last = 0
    for m in pattern.finditer(clean):
        narr = clean[last:m.start()].strip()
        if narr:
            segments.append((narr, "narrator"))
        # Pre-attribution: last ≤80 chars before the opening quote.
        # Captures "She whispered, [quote]" lead-ins without reaching back
        # into the previous quote's own attribution tag.
        pre_narr = narr[-80:].strip() if narr else ""

        # Post-attribution: text after the closing quote, but only up to the
        # first sentence-ending punctuation (.!?) — this catches the dominant
        # pattern '"Hello," she said.' without crossing into the next sentence
        # where the OTHER character might be described with a speech verb
        # ("You answered with a nod.").
        post_raw = clean[m.end():m.end() + 120]
        sent_end = re.search(r'[.!?]', post_raw)
        post_narr = post_raw[:sent_end.end()].strip() if sent_end else post_raw.strip()

        combined_narr = (pre_narr + " " + post_narr).strip()
        speaker = _detect_speaker(combined_narr, player_name, other_name, other_pronoun)
        segments.append((m.group(), speaker))
        last = m.end()
    tail = clean[last:].strip()
    if tail:
        segments.append((tail, "narrator"))
    return segments


# ── In-memory session store (single user) ─────────────────────
# All character fields are None until /start assigns them.
session = {
    "messages":       [],
    "profile":        None,
    "model":          "romance:latest",
    "other_voice":    PRIMARY_FEMALE_CHARACTER_VOICE,
    "other_lang":     PRIMARY_FEMALE_CHARACTER_LANG,
    "other_name":     None,
    "other_pronoun":  None,
    "player_voice":   PRIMARY_MALE_CHARACTER_VOICE,
    "player_lang":    PRIMARY_MALE_CHARACTER_LANG,
    "player_name":    None,
    "player_pronoun": None,
}


@app.route("/")
def index():
    return send_file("index.html")


@app.route("/models")
def models():
    return {"models": list_models()}


@app.route("/stories")
def stories():
    files = sorted(p.name for p in STORIES_DIR.glob("*.txt"))
    return {"stories": files}


@app.route("/start", methods=["POST"])
def start():
    data = request.json
    model = data.get("model", "romance:latest")

    all_stories = list(STORIES_DIR.glob("*.txt"))
    if not all_stories:
        return {"error": f"No .txt files found in {STORIES_DIR}"}, 400

    story_pick = data.get("story", "random")
    if story_pick and story_pick != "random":
        chosen = (STORIES_DIR / story_pick).resolve()
        try:
            chosen.relative_to(STORIES_DIR.resolve())
        except ValueError:
            return {"error": "Invalid story path"}, 400
        if not chosen.exists():
            return {"error": f"Story not found: {story_pick}"}, 400
    else:
        chosen = random.choice(all_stories)

    story_text = chosen.read_text(encoding="utf-8", errors="replace")

    try:
        profile, story_context = analyze_story(story_text, model)
    except Exception as e:
        return {"error": str(e)}, 500

    _FEMALE_NAMES = ["Amelia", "Clara", "Elena", "Isla", "Lyra", "Mara", "Nora", "Rose", "Sarah", "Vera"]
    _MALE_NAMES   = ["Alex", "Daniel", "Ethan", "James", "Liam", "Marcus", "Noah", "Oliver", "Ryan", "Sebastian"]

    other_gender  = profile.get("other_gender", "female").lower()
    player_gender = profile.get("player_gender", "male").lower()

    # Ensure both characters have real names
    other_name = profile.get("other_name", "").strip()
    if not other_name or other_name.lower() in ("her", "she", "him", "he", "they"):
        profile["other_name"] = random.choice(_FEMALE_NAMES if other_gender == "female" else _MALE_NAMES)

    player_name = profile.get("player_name", "").strip()
    if not player_name or player_name.lower() in ("you", "her", "she", "him", "he", "they"):
        profile["player_name"] = random.choice(_MALE_NAMES if player_gender == "male" else _FEMALE_NAMES)

    # Assign PRIMARY voices deterministically by gender.
    # Secondary characters that enter the scene use SECONDARY voices (handled in /tts).
    if other_gender == "male":
        other_voice, other_lang = PRIMARY_MALE_CHARACTER_VOICE, PRIMARY_MALE_CHARACTER_LANG
    else:
        other_voice, other_lang = PRIMARY_FEMALE_CHARACTER_VOICE, PRIMARY_FEMALE_CHARACTER_LANG

    # When both characters share a gender, use the secondary voice for the player
    # so they're distinguishable in TTS.
    if player_gender == other_gender:
        if player_gender == "male":
            player_voice, player_lang = SECONDARY_MALE_CHARACTER_VOICE, SECONDARY_MALE_CHARACTER_LANG
        else:
            player_voice, player_lang = SECONDARY_FEMALE_CHARACTER_VOICE, SECONDARY_FEMALE_CHARACTER_LANG
    elif player_gender == "male":
        player_voice, player_lang = PRIMARY_MALE_CHARACTER_VOICE, PRIMARY_MALE_CHARACTER_LANG
    else:
        player_voice, player_lang = PRIMARY_FEMALE_CHARACTER_VOICE, PRIMARY_FEMALE_CHARACTER_LANG

    other_pronoun  = "he" if other_gender == "male" else "she"
    player_pronoun = "you"  # player is always narrated in second person

    system_prompt = build_system_prompt(profile, story_context)

    print("\n" + "═" * 60)
    print(f"  STORY : {chosen.name}")
    print("═" * 60)
    print(json.dumps(profile, indent=2, ensure_ascii=False))
    print("═" * 60)
    print(f"  Narrator                          → {NARRATOR_VOICE} ({NARRATOR_LANG})")
    print(f"  NPC    [{other_gender:6}] {profile['other_name']:20} → {other_voice} ({other_lang})")
    print(f"  Player [{player_gender:6}] {profile['player_name']:20} → {player_voice} ({player_lang})")
    print(f"  2nd ♂  [male  ] secondary male         → {SECONDARY_MALE_CHARACTER_VOICE} ({SECONDARY_MALE_CHARACTER_LANG})")
    print(f"  2nd ♀  [female] secondary female        → {SECONDARY_FEMALE_CHARACTER_VOICE} ({SECONDARY_FEMALE_CHARACTER_LANG})")
    print("═" * 60 + "\n")

    session["model"]          = model
    session["profile"]        = profile
    session["messages"]       = [{"role": "system", "content": system_prompt}]
    session["other_voice"]    = other_voice
    session["other_lang"]     = other_lang
    session["other_name"]     = profile["other_name"]
    session["other_pronoun"]  = other_pronoun
    session["player_voice"]   = player_voice
    session["player_lang"]    = player_lang
    session["player_name"]    = profile["player_name"]
    session["player_pronoun"] = player_pronoun

    return {
        "name":        profile["other_name"],
        "setting":     profile.get("setting", ""),
        "stage":       profile.get("relationship_stage", ""),
        "summary":     profile.get("story_summary", ""),
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
        "Weave in one or two vivid physical details about the NPC naturally as the scene unfolds; do not front-load a description block. "
        "End on a single unresolved beat: they say one thing or do one thing that demands a response. Stop there.]"
    )}
    session["messages"].append(trigger_msg)
    return Response(
        stream_with_context(_ollama_stream(session["messages"])),
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
        segments = _parse_segments(
            text,
            player_name=session["player_name"] or "",
            player_pronoun=session["player_pronoun"] or "you",
            other_name=session["other_name"] or "",
            other_pronoun=session["other_pronoun"] or "she",
        )

        SR = 24000
        silence = np.zeros(int(SR * 0.18), dtype=np.float32)
        parts = []

        for seg_text, speaker in segments:
            seg_text = seg_text.strip()
            if not seg_text:
                continue
            if speaker == "player":
                voice, lang = session["player_voice"], session["player_lang"]
            elif speaker == "other":
                voice, lang = session["other_voice"], session["other_lang"]
            elif speaker == "secondary_male":
                voice, lang = SECONDARY_MALE_CHARACTER_VOICE, SECONDARY_MALE_CHARACTER_LANG
            elif speaker == "secondary_female":
                voice, lang = SECONDARY_FEMALE_CHARACTER_VOICE, SECONDARY_FEMALE_CHARACTER_LANG
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
