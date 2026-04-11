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
PRIMARY_MALE_CHARACTER_VOICE   = "bm_daniel"
PRIMARY_MALE_CHARACTER_LANG    = "en-gb"
PRIMARY_FEMALE_CHARACTER_VOICE = "bf_isabella"
PRIMARY_FEMALE_CHARACTER_LANG  = "en-gb"

# Secondary character voices (side characters who enter the scene)
SECONDARY_MALE_CHARACTER_VOICE   = "am_onyx"
SECONDARY_MALE_CHARACTER_LANG    = "en-us"
SECONDARY_FEMALE_CHARACTER_VOICE = "af_aoede"
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
# Speech verbs (both present and past tense) for attribution detection
_SPEECH_VERBS = (
    r"(?:said|say|says|replied|reply|replies|answered|answer|answers|"
    r"whispered|whisper|whispers|murmured|murmur|murmurs|"
    r"called|call|calls|asked|ask|asks|demanded|demand|demands|"
    r"muttered|mutter|mutters|added|add|adds|continued|continue|continues|"
    r"shouted|shout|shouts|growled|growl|growls|breathed|breathe|breathes|"
    r"urged|urge|urges|insisted|insist|insists|snapped|snap|snaps|"
    r"hissed|hiss|hisses|sighed|sigh|sighs|laughed|laugh|laughs|"
    r"cried|cry|cries|begged|beg|begs|pleaded|plead|pleads|"
    r"offered|offer|offers|suggested|suggest|suggests|exclaimed|exclaim|exclaims|"
    r"declared|declare|declares|announced|announce|announces|"
    r"groaned|groan|groans|moaned|moan|moans|purred|purr|purrs|"
    r"teased|tease|teases|cooed|coo|coos|stammered|stammer|stammers|"
    r"interrupted|interrupt|interrupts|protested|protest|protests|"
    r"admitted|admit|admits|confessed|confess|confesses)"
)
# Player-specific version kept for backwards compat but now same coverage
_PLAYER_SPEECH_VERBS = _SPEECH_VERBS
_NAMED_ATTR_RE = re.compile(rf'\b([A-Z][a-z]{{2,}})\s+{_SPEECH_VERBS}\b')


def _detect_speaker(narr: str, player_name: str, other_name: str, other_pronoun: str,
                    secondary_characters: dict[str, str] | None = None) -> str:
    """
    Classify the speaker of a quoted passage from surrounding narrator text.
    Returns: "player" | "other" | "secondary_male" | "secondary_female"

    secondary_characters: dict mapping lowercase name → gender ("male"/"female")

    Uses position-based priority: whichever attribution tag appears earliest in
    the text wins, so "she replied and you answered" correctly reads "she replied"
    rather than letting the fixed-order player check override it.

    Name-based matches are authoritative and take priority over pronoun-based
    matches at the same position.
    """
    if not narr:
        return "other"

    if secondary_characters is None:
        secondary_characters = {}

    # (match_start, speaker_type, is_name_match) — name matches break ties
    candidates: list[tuple[int, str, bool]] = []

    # Player: "you said/asked/…" (second-person, present or past)
    for m in re.finditer(rf'\byou\s+{_PLAYER_SPEECH_VERBS}\b', narr, re.IGNORECASE):
        candidates.append((m.start(), "player", True))

    # Player by name: "Andrew said" (third-person fallback)
    if player_name:
        for m in re.finditer(rf'\b{re.escape(player_name)}\s+{_SPEECH_VERBS}\b', narr, re.IGNORECASE):
            candidates.append((m.start(), "player", True))

    # Primary NPC by name: "Amy said"
    if other_name:
        for m in re.finditer(rf'\b{re.escape(other_name)}\s+{_SPEECH_VERBS}\b', narr, re.IGNORECASE):
            candidates.append((m.start(), "other", True))

    # Secondary characters by name (from registry)
    known = {n.lower() for n in (other_name, player_name, "you") if n}
    for sc_name, sc_gender in secondary_characters.items():
        if sc_name in known:
            continue
        # Use the original casing for the regex (capitalize first letter)
        display_name = sc_name.capitalize()
        for m in re.finditer(rf'\b{re.escape(display_name)}\s+{_SPEECH_VERBS}\b', narr, re.IGNORECASE):
            speaker = "secondary_male" if sc_gender == "male" else "secondary_female"
            candidates.append((m.start(), speaker, True))

    # Also catch any capitalized name + speech verb not in our registry
    for m in _NAMED_ATTR_RE.finditer(narr):
        name_lower = m.group(1).lower()
        if name_lower in known:
            continue
        if name_lower in secondary_characters:
            # Already handled above with correct gender
            continue
        # Unknown secondary character — infer gender from pronouns in full narr
        # context that explicitly reference this name (look for "Name... she/he")
        has_she = bool(re.search(rf'\b{re.escape(m.group(1))}\b[^.!?]{{0,60}}\bshe\b', narr, re.IGNORECASE))
        has_he = bool(re.search(rf'\b{re.escape(m.group(1))}\b[^.!?]{{0,60}}\bhe\b', narr, re.IGNORECASE))
        if has_she and not has_he:
            candidates.append((m.start(), "secondary_female", True))
        elif has_he and not has_she:
            candidates.append((m.start(), "secondary_male", True))
        elif other_pronoun == "she":
            candidates.append((m.start(), "secondary_male", True))
        else:
            candidates.append((m.start(), "secondary_female", True))

    # Pronoun-based attribution (weaker signal — only used when no name match exists)
    # Primary NPC by pronoun: "she said" / "he said"
    if other_pronoun:
        for m in re.finditer(rf'\b{re.escape(other_pronoun)}\s+{_SPEECH_VERBS}\b', narr, re.IGNORECASE):
            candidates.append((m.start(), "other", False))

    # Opposite-gender pronoun (cannot be the primary NPC)
    if other_pronoun == "she":
        for m in re.finditer(rf'\bhe\s+{_SPEECH_VERBS}\b', narr, re.IGNORECASE):
            candidates.append((m.start(), "secondary_male", False))
    elif other_pronoun == "he":
        for m in re.finditer(rf'\bshe\s+{_SPEECH_VERBS}\b', narr, re.IGNORECASE):
            candidates.append((m.start(), "secondary_female", False))

    if candidates:
        # Sort by position, but prefer name-based matches over pronoun matches
        # at similar positions (within 5 chars of each other)
        candidates.sort(key=lambda x: (x[0], not x[2]))
        return candidates[0][1]

    return "other"  # default to primary NPC


def _parse_segments(
    text: str,
    player_name: str = "",
    player_pronoun: str = "you",
    other_name: str = "",
    other_pronoun: str = "she",
    secondary_characters: dict[str, str] | None = None,
) -> list[tuple[str, str]]:
    """
    Split narrator text into (segment, speaker) tuples.
    speaker: "narrator" | "other" | "player" | "secondary_male" | "secondary_female"
    Also strips *action* markers so they're read naturally.

    secondary_characters: dict mapping lowercase name → gender ("male"/"female")

    Attribution priority:
    1. Post-attribution (text after the closing quote up to the next sentence end)
       — this is the dominant English pattern: "Hello," she said.
    2. Tight pre-attribution (only the final clause before the quote, after the
       last sentence-ending punctuation) — catches: She whispered, "Hello."
    3. Default to "other" (primary NPC) if neither matches.
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

        # Post-attribution: text after the closing quote, up to first sentence end.
        # This is the most reliable signal — "Hello," she said.
        post_raw = clean[m.end():m.end() + 120]
        sent_end = re.search(r'[.!?]', post_raw)
        post_narr = post_raw[:sent_end.end()].strip() if sent_end else post_raw.strip()

        # Try post-attribution first (most reliable)
        speaker = _detect_speaker(post_narr, player_name, other_name, other_pronoun,
                                  secondary_characters)

        # If post-attribution found nothing, try tight pre-attribution:
        # Only the final clause of the narration before the quote (after the last
        # sentence-ending punctuation). This prevents "She said something. You flinched."
        # from bleeding "She said" into the next quote's attribution.
        if speaker == "other" and narr:
            # Find the last sentence boundary in pre-narration
            last_sent_end = None
            for sent_m in re.finditer(r'[.!?]\s+', narr):
                last_sent_end = sent_m.end()
            # Take only the final clause (text after last sentence boundary)
            if last_sent_end is not None:
                pre_clause = narr[last_sent_end:].strip()
            else:
                # No sentence boundary — use last 60 chars (tighter than before)
                pre_clause = narr[-60:].strip()
            if pre_clause:
                pre_speaker = _detect_speaker(pre_clause, player_name, other_name,
                                             other_pronoun, secondary_characters)
                if pre_speaker != "other":
                    # Only override default if pre-attribution found something specific
                    speaker = pre_speaker

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
    "model":          "hf.co/mradermacher/mistralai-Mistral-Nemo-Instruct-2407-extensive-BP-abliteration-12B-GGUF:Q4_K_M",
    "other_voice":    PRIMARY_FEMALE_CHARACTER_VOICE,
    "other_lang":     PRIMARY_FEMALE_CHARACTER_LANG,
    "other_name":     None,
    "other_pronoun":  None,
    "player_voice":   PRIMARY_MALE_CHARACTER_VOICE,
    "player_lang":    PRIMARY_MALE_CHARACTER_LANG,
    "player_name":    None,
    "player_pronoun": None,
    "secondary_characters": {},
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
    model = data.get("model", "hf.co/mradermacher/mistralai-Mistral-Nemo-Instruct-2407-extensive-BP-abliteration-12B-GGUF:Q4_K_M")

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

    # If the user requested a specific gender, swap player/other roles when needed.
    preferred_gender = data.get("player_gender", "auto").lower()
    if preferred_gender in ("male", "female"):
        assigned_player_gender = profile.get("player_gender", "male").lower()
        if assigned_player_gender != preferred_gender:
            # Swap all player ↔ other fields so the user plays the right character.
            for key_pair in [("player_name", "other_name"),
                             ("player_gender", "other_gender")]:
                pk, ok = key_pair
                profile[pk], profile[ok] = profile.get(ok, ""), profile.get(pk, "")
            # Swap any appearance/personality fields that are character-specific.
            # The LLM always describes "other", so after a swap those fields now
            # describe the new "other" — which is correct; no extra work needed.

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

    # Build secondary character registry: {name_lower: gender}
    secondary_characters = {}
    for sc in profile.get("secondary_characters", []):
        name = sc.get("name", "").strip()
        gender = sc.get("gender", "").lower()
        if name and gender in ("male", "female"):
            secondary_characters[name.lower()] = gender

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
    if secondary_characters:
        print("  ─── Secondary character registry ───")
        for sc_name, sc_gender in secondary_characters.items():
            voice = SECONDARY_MALE_CHARACTER_VOICE if sc_gender == "male" else SECONDARY_FEMALE_CHARACTER_VOICE
            print(f"    {sc_name.capitalize():20} [{sc_gender:6}] → {voice}")
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
    session["secondary_characters"] = secondary_characters

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


@app.route("/suggest", methods=["POST"])
def suggest():
    """Return 2-3 short action options the player could take, based on current context."""
    if not session["messages"]:
        return {"suggestions": []}

    # System prompt + last few turns for context (keep it short)
    system = session["messages"][:1]
    recent = [m for m in session["messages"][1:] if m["role"] != "system"]
    context = system + recent[-6:]

    suggest_prompt = (
        "Suggest exactly 3 brief actions or responses the player could take next in the scene. "
        "Each must be 1–2 sentences, written in first person as something the player does or says. "
        "Vary the tone: one bold/direct, one tender/warm, one cautious/indirect. "
        "Return ONLY a JSON array of 3 strings — no prose, no markdown fences, no commentary. "
        'Example: ["I reach for her hand.", "\\"Tell me,\\" I say softly.", "I look away, unsure."]'
    )
    messages = context + [{"role": "user", "content": suggest_prompt}]

    payload = json.dumps({
        "model": session["model"],
        "messages": messages,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
        content = result.get("message", {}).get("content", "").strip()
        # Strip markdown fences if present
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content.strip())
        suggestions = json.loads(content)
        if isinstance(suggestions, list):
            suggestions = [str(s) for s in suggestions[:3]]
        else:
            suggestions = []
    except Exception as e:
        print(f"Suggest error: {e}")
        suggestions = []

    return {"suggestions": suggestions}


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
            secondary_characters=session.get("secondary_characters", {}),
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


@app.route("/kokoro_test")
def kokoro_test():
    return send_file("kokoro_test.html")


@app.route("/tts_preview", methods=["POST"])
def tts_preview():
    """Render a short sample monologue in a specific Kokoro voice."""
    data = request.json or {}
    voice = data.get("voice", "").strip()
    lang  = data.get("lang", "en-us").strip()
    if not voice:
        return {"error": "No voice specified"}, 400

    sample = (
        "Good evening. I hope you don't mind me saying — you have the most "
        "extraordinary way of making an ordinary room feel entirely different. "
        "I've been watching you from across the hall, trying to work up the nerve "
        "to introduce myself. My name is… well, that hardly matters yet, does it? "
        "What matters is that I have a feeling this conversation is going to be "
        "one I remember for a very long time."
    )

    try:
        import numpy as np
        kokoro = get_kokoro()
        samples, _ = kokoro.create(sample, voice=voice, speed=1.0, lang=lang)
        wav_bytes = _float32_to_wav(samples, 24000)
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
