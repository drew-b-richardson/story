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

from story import analyze_story, enrich_character_profile, build_system_prompt, list_models, OLLAMA_URL

STORIES_DIR = Path(__file__).parent / "stories"
SUMMARIES_DIR = Path(__file__).parent / "story_summaries"
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


def _character_section(role_label: str, name: str, data: dict) -> str:
    """Build a single character section for character.md."""
    def bullet_list(items) -> str:
        if isinstance(items, list) and items:
            return "\n".join(f"- {i}" for i in items)
        return "_Not specified_"

    def field(val, fallback="_Not specified_") -> str:
        return val if val and str(val).strip() else fallback

    traits = data.get("personality", [])
    traits_str = ", ".join(traits) if traits else "_Not specified_"

    return f"""# {role_label}: {name}

## Appearance
{field(data.get('appearance'))}

### Hair
{field(data.get('hair'))}

### Eyes
{field(data.get('eyes'))}

### Scent
{field(data.get('scent'))}

### Style
{field(data.get('clothing_style'))}

## Personality
{traits_str}

## What They Love
{bullet_list(data.get('loves'))}

## What They Hate
{bullet_list(data.get('hates'))}

## Desires & Fears
{field(data.get('desires'))}
"""


def _profile_to_markdown(profile: dict) -> str:
    """Convert an analyzed profile dict to role-labeled character sections."""
    player_name   = profile.get("player_name", "Player")
    player_gender = profile.get("player_gender", "male").lower()
    other_name    = profile.get("other_name", "Unknown")
    other_gender  = profile.get("other_gender", "female").lower()

    player_role = "PRIMARY_MALE"   if player_gender == "male" else "PRIMARY_FEMALE"
    other_role  = "PRIMARY_MALE"   if other_gender  == "male" else "PRIMARY_FEMALE"

    # Build player data dict from player_ prefixed fields
    player_data = {
        "appearance":     profile.get("player_appearance"),
        "hair":           profile.get("player_hair"),
        "eyes":           profile.get("player_eyes"),
        "scent":          profile.get("player_scent"),
        "clothing_style": profile.get("player_clothing_style"),
        "personality":    profile.get("player_personality", []),
        "loves":          profile.get("player_loves", []),
        "hates":          profile.get("player_hates", []),
        "desires":        profile.get("player_desires"),
    }

    # Build other_name data dict from top-level fields
    other_data = {
        "appearance":     profile.get("appearance"),
        "hair":           profile.get("hair"),
        "eyes":           profile.get("eyes"),
        "scent":          profile.get("scent"),
        "clothing_style": profile.get("clothing_style"),
        "personality":    profile.get("personality", []),
        "loves":          profile.get("other_loves", []),
        "hates":          profile.get("other_hates", []),
        "desires":        profile.get("desires"),
    }

    sections = [
        _character_section(player_role, player_name, player_data),
        _character_section(other_role, other_name, other_data),
    ]

    # Add a speech/affection block to the NPC section
    speech_block = ""
    if profile.get("speech_style"):
        speech_block += f"\n## How They Speak\n{profile['speech_style']}\n"
    if profile.get("affection_style"):
        speech_block += f"\n## How They Show Affection\n{profile['affection_style']}\n"
    behaviors = profile.get("key_behaviors", [])
    if behaviors:
        speech_block += "\n## Key Behaviors\n" + "\n".join(f"- {b}" for b in behaviors) + "\n"
    if profile.get("dealbreakers"):
        speech_block += f"\n## What Pushes Them Away\n{profile['dealbreakers']}\n"
    if speech_block:
        sections[1] = sections[1].rstrip() + "\n" + speech_block

    # Secondary characters
    for sc in profile.get("secondary_characters", []):
        sc_name   = sc.get("name", "Unknown")
        sc_gender = sc.get("gender", "unknown").lower()
        sc_role   = "SECONDARY_MALE" if sc_gender == "male" else "SECONDARY_FEMALE"
        sc_data   = {
            "appearance":  sc.get("appearance"),
            "hair":        sc.get("hair"),
            "eyes":        sc.get("eyes"),
            "scent":       sc.get("scent"),
            "clothing_style": sc.get("clothing_style"),
            "personality": sc.get("personality", []),
            "loves":       sc.get("loves", []),
            "hates":       sc.get("hates", []),
            "desires":     sc.get("desires"),
        }
        sections.append(_character_section(sc_role, sc_name, sc_data))

    return "\n\n---\n\n".join(sections)


def _trim_characters(profile: dict) -> dict:
    """
    Enforce role limits: at most one SECONDARY_MALE and one SECONDARY_FEMALE.
    Keeps the first male and first female secondary character; drops the rest.
    """
    secondary = profile.get("secondary_characters", [])
    if not secondary:
        return profile
    kept_male = None
    kept_female = None
    for sc in secondary:
        g = sc.get("gender", "").lower()
        if g == "male" and kept_male is None:
            kept_male = sc
        elif g == "female" and kept_female is None:
            kept_female = sc
    profile["secondary_characters"] = [sc for sc in (kept_male, kept_female) if sc]
    return profile


def _profile_to_story_markdown(profile: dict) -> str:
    """Build story.md from profile fields using role labels instead of character names."""
    player_gender = profile.get("player_gender", "male").lower()
    other_gender  = profile.get("other_gender", "female").lower()
    player_role   = "PRIMARY_MALE" if player_gender == "male" else "PRIMARY_FEMALE"
    other_role    = "PRIMARY_MALE" if other_gender  == "male" else "PRIMARY_FEMALE"

    # Replace character names with role labels in the summary text
    summary = profile.get("story_summary", "_Not provided_")
    player_name = profile.get("player_name", "")
    other_name  = profile.get("other_name", "")
    if player_name:
        summary = re.sub(rf'\b{re.escape(player_name)}\b', player_role, summary)
    if other_name:
        summary = re.sub(rf'\b{re.escape(other_name)}\b', other_role, summary)

    beats = profile.get("story_beats", [])
    beats_text = "\n".join(f"{i+1}. {b}" for i, b in enumerate(beats)) if beats else "_None provided_"

    return f"""# Story

## Setting
{profile.get('setting', '_Not provided_')}

## Starting Point
{profile.get('relationship_stage', '_Not provided_')}

## Summary
{summary}

## Story Beats
{beats_text}
"""


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
    "story_beats":    [],
    "beat_index":     0,
}


@app.route("/")
def index():
    return send_file("index.html")


@app.route("/index_stories")
def index_stories():
    return send_file("index_stories.html")


@app.route("/models")
def models():
    return {"models": list_models()}


@app.route("/stories")
def stories():
    files = sorted(p.name for p in STORIES_DIR.glob("*.txt"))
    return {"stories": files}


@app.route("/analyze", methods=["POST"])
def analyze():
    """Analyze a story and save character + story summaries as .md files."""
    data = request.json or {}
    story_name = data.get("story", "").strip()

    if not story_name:
        return {"error": "No story name provided"}, 400

    # Sanitize path to prevent directory traversal
    story_file = STORIES_DIR / f"{story_name}.txt"
    try:
        story_file = story_file.resolve()
        story_file.relative_to(STORIES_DIR.resolve())
    except (ValueError, RuntimeError):
        return {"error": "Invalid story path"}, 400

    if not story_file.exists():
        return {"error": f"Story not found: {story_name}.txt"}, 404

    try:
        # Create summaries directory if needed
        SUMMARIES_DIR.mkdir(exist_ok=True)

        # Read and analyze story
        story_text = story_file.read_text(encoding="utf-8", errors="replace")
        model = data.get("model", "hf.co/mradermacher/mistralai-Mistral-Nemo-Instruct-2407-extensive-BP-abliteration-12B-GGUF:Q4_K_M")

        profile, story_context = analyze_story(story_text, model)

        # Enforce role limits before enrichment (avoid wasting LLM calls on dropped chars)
        profile = _trim_characters(profile)

        # Second pass: fill in physical descriptions, loves/hates, secondary details
        profile = enrich_character_profile(profile, story_context, model)

        # Save character summary as markdown
        char_md = _profile_to_markdown(profile)
        char_file = SUMMARIES_DIR / f"{story_name}_character.md"
        char_file.write_text(char_md, encoding="utf-8")

        # Build story.md from profile fields using role names (not character names)
        story_md = _profile_to_story_markdown(profile)
        story_file_md = SUMMARIES_DIR / f"{story_name}_story.md"
        story_file_md.write_text(story_md, encoding="utf-8")

        # Save enriched profile as JSON (for game use)
        profile_file = SUMMARIES_DIR / f"{story_name}_profile.json"
        profile_file.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")

        return {
            "success": True,
            "story": story_name,
            "character": profile.get("other_name", "Unknown"),
        }
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}, 500


@app.route("/check-analysis/<story_name>")
def check_analysis(story_name):
    """Check if a story has been analyzed (summaries exist)."""
    try:
        story_name = story_name.strip()
        char_file = SUMMARIES_DIR / f"{story_name}_character.md"
        return {"analyzed": char_file.exists()}
    except Exception:
        return {"analyzed": False}


@app.route("/start", methods=["POST"])
def start():
    data = request.json
    model = data.get("model", "hf.co/mradermacher/mistralai-Mistral-Nemo-Instruct-2407-extensive-BP-abliteration-12B-GGUF:Q4_K_M")

    all_stories = list(STORIES_DIR.glob("*.txt"))
    if not all_stories:
        return {"error": f"No .txt files found in {STORIES_DIR}"}, 400

    story_pick = data.get("story", "random")
    if story_pick and story_pick != "random":
        # Remove .txt extension if included
        story_name = story_pick.replace(".txt", "")
        chosen = (STORIES_DIR / f"{story_name}.txt").resolve()
        try:
            chosen.relative_to(STORIES_DIR.resolve())
        except ValueError:
            return {"error": "Invalid story path"}, 400
        if not chosen.exists():
            return {"error": f"Story not found: {story_pick}"}, 400
    else:
        chosen = random.choice(all_stories)
        story_name = chosen.stem

    # Check if summaries exist for this story
    char_file = SUMMARIES_DIR / f"{story_name}_character.md"
    profile_file = SUMMARIES_DIR / f"{story_name}_profile.json"
    story_md_file = SUMMARIES_DIR / f"{story_name}_story.md"

    try:
        if profile_file.exists() and story_md_file.exists():
            # Load from saved summaries
            profile = json.loads(profile_file.read_text(encoding="utf-8"))
            story_context = story_md_file.read_text(encoding="utf-8")
        else:
            # Analyze fresh
            story_text = chosen.read_text(encoding="utf-8", errors="replace")
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
    session["story_beats"]    = profile.get("story_beats", [])
    session["beat_index"]     = 0

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

    # Use a neutral system message so the model isn't locked into story-narrator mode,
    # which causes it to ignore JSON instructions after a few turns.
    suggest_system = (
        "You are a creative writing assistant. Your only job is to suggest player actions. "
        "You must respond with ONLY a valid JSON array of exactly 3 strings. No prose, no markdown."
    )
    recent = [m for m in session["messages"] if m["role"] != "system"]
    context = recent[-6:]

    suggest_prompt = (
        "Suggest exactly 3 brief actions or responses the player could take next in this scene. "
        "Each must be 1–2 sentences, written in first person as something the player does or says. "
        "Vary the tone: one bold/direct, one tender/warm, one cautious/indirect. "
        "Return ONLY a JSON array of 3 strings — no prose, no markdown fences, no commentary. "
        'Example: ["I reach for her hand.", "\\"Tell me,\\" I say softly.", "I look away, unsure."]'
    )
    messages = [{"role": "system", "content": suggest_system}] + context + [{"role": "user", "content": suggest_prompt}]

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
        parsed = json.loads(content)
        if isinstance(parsed, list):
            suggestions = [str(s) for s in parsed[:3]]
        elif isinstance(parsed, dict):
            # JSON mode may wrap the array: {"suggestions": [...]} or {"actions": [...]}
            for val in parsed.values():
                if isinstance(val, list):
                    suggestions = [str(s) for s in val[:3]]
                    break
            else:
                suggestions = []
        else:
            suggestions = []
    except Exception as e:
        print(f"Suggest error: {e!r}")
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

    # Store the clean message in history
    session["messages"].append({"role": "user", "content": user_input})

    # Every 3 user turns inject a beat nudge into what the LLM sees,
    # but NOT into the stored session history.
    beats = session.get("story_beats", [])
    messages_for_llm = session["messages"]
    if beats:
        turn_count = sum(1 for m in session["messages"] if m["role"] == "user")
        if turn_count % 3 == 0:
            beat_idx = session["beat_index"] % len(beats)
            beat = beats[beat_idx]
            session["beat_index"] += 1
            nudge = (
                f"\n\n[Scene director: You have not yet brought this beat into the action — "
                f"introduce it naturally within this exchange or the next. "
                f"Engineer the situation; don't announce it: {beat}]"
            )
            messages_for_llm = session["messages"][:-1] + [
                {"role": "user", "content": user_input + nudge}
            ]

    return Response(
        stream_with_context(_ollama_stream(messages_for_llm)),
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
