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
import subprocess
import sys
import threading
import urllib.request
import urllib.error
from pathlib import Path
from flask import Flask, request, Response, send_file, stream_with_context

from story import analyze_story, enrich_character_profile, build_system_prompt, list_models, OLLAMA_URL

STORIES_DIR = Path(__file__).parent / "stories"
STORIES_DIR_JA = Path(__file__).parent / "stories_ja"
SUMMARIES_DIR = Path(__file__).parent / "story_summaries"
SUMMARIES_DIR_JA = Path(__file__).parent / "story_summaries_jp"


def _summaries_dir(lang: str) -> Path:
    return SUMMARIES_DIR_JA if (lang or "").lower() == "ja" else SUMMARIES_DIR


def _stories_dir(lang: str) -> Path:
    return STORIES_DIR_JA if (lang or "").lower() == "ja" else STORIES_DIR
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

# Japanese voice set (Kokoro v1.0 jf_/jm_ prefixes). Used when session["lang"] == "ja".
JA_NARRATOR_VOICE                = "jf_alpha"
JA_PRIMARY_MALE_VOICE            = "jm_kumo"
JA_PRIMARY_FEMALE_VOICE          = "jf_gongitsune"
JA_SECONDARY_MALE_VOICE          = "jm_kumo"      # Kokoro ships only one jm_ voice
JA_SECONDARY_FEMALE_VOICE        = "jf_nezumi"
JA_LANG                          = "ja"


def _voice_for(speaker: str, lang: str) -> tuple[str, str]:
    """Return (voice, lang) for a speaker role. Narrator/secondary are language-global;
    player/other voices are resolved from session for primaries."""
    if lang == "ja":
        mapping = {
            "narrator":          (JA_NARRATOR_VOICE,         JA_LANG),
            "secondary_male":    (JA_SECONDARY_MALE_VOICE,   JA_LANG),
            "secondary_female":  (JA_SECONDARY_FEMALE_VOICE, JA_LANG),
        }
    else:
        mapping = {
            "narrator":          (NARRATOR_VOICE,                   NARRATOR_LANG),
            "secondary_male":    (SECONDARY_MALE_CHARACTER_VOICE,   SECONDARY_MALE_CHARACTER_LANG),
            "secondary_female": (SECONDARY_FEMALE_CHARACTER_VOICE, SECONDARY_FEMALE_CHARACTER_LANG),
        }
    return mapping.get(speaker, mapping["narrator"])

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


# Misaki JA G2P — kokoro-onnx does NOT auto-use misaki; it always phonemizes
# via espeak-ng, which mangles Japanese. We phonemize JA text ourselves with
# misaki, then pass IPA phonemes to Kokoro with is_phonemes=True.
_ja_g2p = None
_ja_g2p_lock = threading.Lock()

def get_ja_g2p():
    global _ja_g2p
    if _ja_g2p is None:
        with _ja_g2p_lock:
            if _ja_g2p is None:
                from misaki import ja
                _ja_g2p = ja.JAG2P()
    return _ja_g2p


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


# Localized heading/fallback labels for the generated markdown files.
_MD_LABELS = {
    "en": {
        "appearance":  "Appearance",
        "hair":        "Hair",
        "eyes":        "Eyes",
        "scent":       "Scent",
        "style":       "Style",
        "personality": "Personality",
        "loves":       "What They Love",
        "hates":       "What They Hate",
        "desires":     "Desires & Fears",
        "speaks":      "How They Speak",
        "affection":   "How They Show Affection",
        "behaviors":   "Key Behaviors",
        "pushes_away": "What Pushes Them Away",
        "setting":     "Setting",
        "starting":    "Starting Point",
        "summary":     "Summary",
        "beats":       "Story Beats",
        "not_specified": "_Not specified_",
        "not_provided":  "_Not provided_",
        "none_provided": "_None provided_",
        "traits_sep":  ", ",
    },
    "ja": {
        "appearance":  "外見",
        "hair":        "髪",
        "eyes":        "目",
        "scent":       "香り",
        "style":       "服装",
        "personality": "性格",
        "loves":       "好きなもの",
        "hates":       "嫌いなもの",
        "desires":     "望みと恐れ",
        "speaks":      "話し方",
        "affection":   "愛情の示し方",
        "behaviors":   "特徴的な行動",
        "pushes_away": "遠ざけるもの",
        "setting":     "舞台",
        "starting":    "関係の出発点",
        "summary":     "あらすじ",
        "beats":       "ストーリービート",
        "not_specified": "_未記載_",
        "not_provided":  "_未記載_",
        "none_provided": "_なし_",
        "traits_sep":  "、",
    },
}


def _character_section(role_label: str, name: str, data: dict, lang: str = "en") -> str:
    """Build a single character section for character.md."""
    L = _MD_LABELS.get(lang, _MD_LABELS["en"])
    sep = L["traits_sep"]

    def flatten(val) -> str:
        """Collapse dict/list values into a readable string. LLMs sometimes
        return nested objects for fields we expected to be flat strings
        (e.g. hair: {color, length, texture}); render those as prose rather
        than exposing the raw Python repr."""
        if isinstance(val, dict):
            return sep.join(str(v).strip() for v in val.values() if v)
        if isinstance(val, list):
            return sep.join(str(v).strip() for v in val if v)
        return str(val).strip() if val is not None else ""

    def bullet_list(items) -> str:
        if isinstance(items, list) and items:
            return "\n".join(f"- {flatten(i)}" for i in items if i)
        return L["not_specified"]

    def field(val) -> str:
        s = flatten(val)
        return s if s else L["not_specified"]

    traits = data.get("personality", [])
    traits_str = sep.join(flatten(t) for t in traits if t) if traits else L["not_specified"]

    return f"""# {role_label}: {name}

## {L["appearance"]}
{field(data.get('appearance'))}

### {L["hair"]}
{field(data.get('hair'))}

### {L["eyes"]}
{field(data.get('eyes'))}

### {L["scent"]}
{field(data.get('scent'))}

### {L["style"]}
{field(data.get('clothing_style'))}

## {L["personality"]}
{traits_str}

## {L["loves"]}
{bullet_list(data.get('loves'))}

## {L["hates"]}
{bullet_list(data.get('hates'))}

## {L["desires"]}
{field(data.get('desires'))}
"""


def _profile_to_markdown(profile: dict, lang: str = "en") -> str:
    """Convert an analyzed profile dict to role-labeled character sections."""
    L = _MD_LABELS.get(lang, _MD_LABELS["en"])
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
        _character_section(player_role, player_name, player_data, lang=lang),
        _character_section(other_role, other_name, other_data, lang=lang),
    ]

    # Add a speech/affection block to the NPC section
    speech_block = ""
    if profile.get("speech_style"):
        speech_block += f"\n## {L['speaks']}\n{profile['speech_style']}\n"
    if profile.get("affection_style"):
        speech_block += f"\n## {L['affection']}\n{profile['affection_style']}\n"
    behaviors = profile.get("key_behaviors", [])
    if behaviors:
        speech_block += f"\n## {L['behaviors']}\n" + "\n".join(f"- {b}" for b in behaviors) + "\n"
    if profile.get("dealbreakers"):
        speech_block += f"\n## {L['pushes_away']}\n{profile['dealbreakers']}\n"
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
        sections.append(_character_section(sc_role, sc_name, sc_data, lang=lang))

    return "\n\n---\n\n".join(sections)


def _trim_characters(profile: dict) -> dict:
    """
    Enforce role limits: at most one SECONDARY_MALE and one SECONDARY_FEMALE.
    Keeps the first male and first female secondary character; drops the rest.
    """
    secondary = [sc for sc in profile.get("secondary_characters", []) if isinstance(sc, dict)]
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


def _profile_to_story_markdown(profile: dict, lang: str = "en") -> str:
    """Build story.md from profile fields using role labels instead of character names."""
    L = _MD_LABELS.get(lang, _MD_LABELS["en"])
    player_gender = profile.get("player_gender", "male").lower()
    other_gender  = profile.get("other_gender", "female").lower()
    player_role   = "PRIMARY_MALE" if player_gender == "male" else "PRIMARY_FEMALE"
    other_role    = "PRIMARY_MALE" if other_gender  == "male" else "PRIMARY_FEMALE"

    # Replace character names with role labels in the summary text.
    # The word-boundary \b doesn't work for CJK, so for JA names use plain replace.
    summary = profile.get("story_summary", L["not_provided"])
    player_name = profile.get("player_name", "")
    other_name  = profile.get("other_name", "")
    if lang == "ja":
        if player_name: summary = summary.replace(player_name, player_role)
        if other_name:  summary = summary.replace(other_name,  other_role)
    else:
        if player_name: summary = re.sub(rf'\b{re.escape(player_name)}\b', player_role, summary)
        if other_name:  summary = re.sub(rf'\b{re.escape(other_name)}\b',  other_role,  summary)

    beats = profile.get("story_beats", [])
    beats_text = "\n".join(f"{i+1}. {b}" for i, b in enumerate(beats)) if beats else L["none_provided"]
    title = "物語" if lang == "ja" else "Story"

    return f"""# {title}

## {L["setting"]}
{profile.get('setting', L["not_provided"])}

## {L["starting"]}
{profile.get('relationship_stage', L["not_provided"])}

## {L["summary"]}
{summary}

## {L["beats"]}
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


# ── Japanese segmenter ────────────────────────────────────────
# Verb stems covering speech, reactions, and voice-adjacent acts. Any trailing
# hiragana is consumed so 言う/言った/言って/囁いた/etc. all match.
_JA_SPEECH_VERB_STEMS = (
    r"(?:"
    r"言|話|答|聞|尋|訊|呟|囁|叫|怒鳴|呼|告|続|返|"
    r"笑|微笑|頷|ため息|息|"
    r"口を開|口にし|声を"
    r")"
)
_JA_SPEECH_VERBS = rf"{_JA_SPEECH_VERB_STEMS}[\u3040-\u309F]*"
# Attribution signal — either:
#   (a) quotative と/って at the start of the post-quote clause  (strongest JA signal), or
#   (b) a speech verb anywhere in the clause (covers pre-attribution + action beats).
_JA_ATTR_LEADING = re.compile(r'^\s*(?:と|って)')
_JA_ATTR_VERB_RE = re.compile(_JA_SPEECH_VERBS)
def _ja_has_attribution(clause: str) -> bool:
    return bool(_JA_ATTR_LEADING.match(clause) or _JA_ATTR_VERB_RE.search(clause))


def _detect_speaker_ja(narr_before: str, narr_after: str,
                       player_name: str, other_name: str, other_gender: str,
                       secondary_characters: dict[str, str] | None = None) -> str:
    """
    Japanese attribution is overwhelmingly POST-quote: 「…」と彼女は囁いた。
    narr_after is the primary signal; narr_before is a fallback for less-common
    pre-attribution forms (彼女は囁いた。「…」).
    """
    secondary_characters = secondary_characters or {}
    other_gender = (other_gender or "female").lower()

    sent_end = re.search(r'[。！？]', narr_after)
    attr = narr_after[:sent_end.end()] if sent_end else narr_after[:80]

    if not _ja_has_attribution(attr):
        # Fallback: final clause of pre-narration
        last_end = 0
        for m in re.finditer(r'[。！？]', narr_before):
            last_end = m.end()
        pre_clause = narr_before[last_end:] if last_end else narr_before[-60:]
        if _ja_has_attribution(pre_clause):
            attr = pre_clause
        else:
            return "other"

    # Name-based attribution (authoritative). Subject particles は/が/も or comma/space.
    def name_role(name: str, role: str):
        if name and re.search(rf'{re.escape(name)}(?:は|が|も|、|\s)', attr):
            return role
        return None

    if (r := name_role(player_name, "player")): return r
    if (r := name_role(other_name, "other")):   return r
    known = {n for n in (other_name, player_name) if n}
    for sc_name, sc_gender in secondary_characters.items():
        if sc_name in {n.lower() for n in known}:
            continue
        role = "secondary_male" if sc_gender == "male" else "secondary_female"
        if re.search(rf'{re.escape(sc_name)}(?:は|が|も|、|\s)', attr):
            return role

    # Pronouns: 彼女 (she) must be checked before 彼 (he) since it contains 彼.
    if "彼女" in attr:
        return "other" if other_gender == "female" else "secondary_female"
    if re.search(r'彼(?!女)', attr):
        return "other" if other_gender == "male" else "secondary_male"

    # Second-person markers — rare in JA fiction but decisive when present.
    if "あなた" in attr or re.search(r'君(?!.*さん)', attr):
        return "player"

    return "other"


def _parse_segments_ja(
    text: str,
    player_name: str = "",
    other_name: str = "",
    other_gender: str = "female",
    secondary_characters: dict[str, str] | None = None,
) -> list[tuple[str, str]]:
    """Japanese equivalent of _parse_segments. Recognizes 「」 and 『』 quotes."""
    clean = re.sub(r"\*([^*\n]+)\*", r"\1", text)
    clean = re.sub(r"<[^>]+>", "", clean)

    segments: list[tuple[str, str]] = []
    # 「…」 primary speech, 『…』 nested/emphasized. Also accept stray ASCII "..." the LLM may emit.
    pattern = re.compile(r'「[^「」]{1,500}」|『[^『』]{1,500}』|"[^"]{1,500}"')
    last = 0
    for m in pattern.finditer(clean):
        narr_before = clean[last:m.start()].strip()
        if narr_before:
            segments.append((narr_before, "narrator"))

        narr_after = clean[m.end():m.end() + 120]
        speaker = _detect_speaker_ja(narr_before, narr_after,
                                     player_name, other_name, other_gender,
                                     secondary_characters)
        segments.append((m.group(), speaker))
        last = m.end()

    tail = clean[last:].strip()
    if tail:
        segments.append((tail, "narrator"))
    return segments


# ── TTS chunking ──────────────────────────────────────────────
# Kokoro caps each synthesis call at 510 phoneme tokens. Phoneme density
# varies wildly: kanji-heavy Japanese can expand 1 char → 4-5 phonemes via
# kana lookup, so the char budget has to be conservative. A 90-char budget
# empirically keeps kanji-dense passages under the cap. On overflow we catch
# the error and recursively split, so the budget is a soft target, not a
# hard guarantee.
_TTS_MAX_CHARS_JA = 90
_TTS_MAX_CHARS_EN = 380


def _chunk_for_tts(text: str, lang: str) -> list[str]:
    """Split a TTS segment on sentence boundaries so each piece fits Kokoro's
    per-call token cap. For JA, split on 。！？ (and comma 、 as fallback).
    For EN, split on .!? (and comma , as fallback)."""
    max_chars = _TTS_MAX_CHARS_JA if lang == "ja" else _TTS_MAX_CHARS_EN
    if len(text) <= max_chars:
        return [text]

    sent_end_re = re.compile(r'[。！？]' if lang == "ja" else r'[.!?]')
    soft_break_re = re.compile(r'[、]' if lang == "ja" else r'[,;:]')

    # First pass: hard sentence breaks.
    pieces: list[str] = []
    buf = ""
    i = 0
    for m in sent_end_re.finditer(text):
        sentence = text[i:m.end()]
        if len(buf) + len(sentence) > max_chars and buf:
            pieces.append(buf)
            buf = sentence
        else:
            buf += sentence
        i = m.end()
    tail = text[i:]
    if tail:
        buf += tail
    if buf:
        pieces.append(buf)

    # Second pass: any piece still too long gets split on soft breaks,
    # then on hard char limit as last resort.
    final: list[str] = []
    for p in pieces:
        if len(p) <= max_chars:
            final.append(p)
            continue
        sub = ""
        last_i = 0
        for m in soft_break_re.finditer(p):
            part = p[last_i:m.end()]
            if len(sub) + len(part) > max_chars and sub:
                final.append(sub)
                sub = part
            else:
                sub += part
            last_i = m.end()
        sub += p[last_i:]
        # Last resort: slice any remaining over-long piece.
        while len(sub) > max_chars:
            final.append(sub[:max_chars])
            sub = sub[max_chars:]
        if sub:
            final.append(sub)

    return [s.strip() for s in final if s.strip()]


# Language-annotation patterns some models (Qwen especially) like to inject
# into narration — e.g. "(in Japanese) 彼女は…" or "（中国語で）...". Kokoro
# reads them literally, so scrub before synthesis.
_LANG_TAG_RE = re.compile(
    r"""
      [（(\[]\s*                              # opening bracket: ( [ （
      (?:in\s+)?                              # optional "in "
      (?:japanese|chinese|english|korean|
         日本語|中国語|英語|韓国語)
      (?:\s*[で:：])?                         # optional particle/colon inside
      \s*[)）\]]                              # closing bracket
      \s*[:：]?\s*                            # consume any trailing colon too
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _scrub_for_tts(text: str) -> str:
    """Remove inline language annotations the LLM may have emitted."""
    cleaned = _LANG_TAG_RE.sub("", text)
    # Strip leftover leading punctuation/whitespace from removed tags.
    cleaned = re.sub(r'^[\s:：、,;；\-—–]+', '', cleaned)
    return cleaned.strip()


def _synthesize_with_fallback(kokoro, text: str, voice: str, lang: str, depth: int = 0):
    """Call kokoro.create, and on Kokoro's 510-token overflow error, split the
    text in half and retry recursively. Yields audio samples for each piece.

    For JA, phonemize with misaki first and feed phonemes directly to Kokoro
    (kokoro-onnx's built-in phonemizer is espeak-ng, which garbles Japanese
    with multilingual 'In Chinese/In Japanese' fallback annotations)."""
    if lang == "ja":
        phonemes = get_ja_g2p()(text)
        # JAG2P returns either a str or (str, tokens); normalize.
        if isinstance(phonemes, tuple):
            phonemes = phonemes[0]
        try:
            samples, _ = kokoro.create(phonemes, voice=voice, speed=1.0,
                                       lang="ja", is_phonemes=True)
            yield samples
            return
        except IndexError as e:
            if "out of bounds" not in str(e) or depth > 6 or len(text) < 8:
                raise
    else:
        try:
            samples, _ = kokoro.create(text, voice=voice, speed=1.0, lang=lang)
            yield samples
            return
        except IndexError as e:
            if "out of bounds" not in str(e) or depth > 6 or len(text) < 8:
                raise

    # Split roughly in half at the nearest sentence/comma/space boundary.
    mid = len(text) // 2
    split_at = mid
    for delta in range(min(mid, 40)):
        for offset in (mid - delta, mid + delta):
            if 0 < offset < len(text) and text[offset] in "。！？.!?、,; ":
                split_at = offset + 1
                break
        else:
            continue
        break
    left, right = text[:split_at].strip(), text[split_at:].strip()
    if left:
        yield from _synthesize_with_fallback(kokoro, left, voice, lang, depth + 1)
    if right:
        yield from _synthesize_with_fallback(kokoro, right, voice, lang, depth + 1)


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
    "lang":           "en",   # "en" | "ja" — selects segmenter + voice set
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
    lang = (request.args.get("lang") or "en").lower()
    analyzed_only = request.args.get("analyzed") == "1"
    src_dir = _stories_dir(lang)
    all_txt = sorted(p.name for p in src_dir.glob("*.txt")) if src_dir.exists() else []
    if analyzed_only:
        summaries = _summaries_dir(lang)
        if summaries.exists():
            available = {p.name.replace("_character.md", "")
                         for p in summaries.glob("*_character.md")}
            all_txt = [s for s in all_txt if s.replace(".txt", "") in available]
    return {"stories": all_txt}


@app.route("/analyze", methods=["POST"])
def analyze():
    """Analyze a story and save character + story summaries as .md files."""
    data = request.json or {}
    story_name = data.get("story", "").strip()

    if not story_name:
        return {"error": "No story name provided"}, 400

    analyze_lang = (data.get("lang") or "en").lower()
    if analyze_lang not in ("en", "ja"):
        analyze_lang = "en"
    stories_src = _stories_dir(analyze_lang)

    # Sanitize path to prevent directory traversal
    story_file = stories_src / f"{story_name}.txt"
    try:
        story_file = story_file.resolve()
        story_file.relative_to(stories_src.resolve())
    except (ValueError, RuntimeError):
        return {"error": "Invalid story path"}, 400

    if not story_file.exists():
        return {"error": f"Story not found: {story_name}.txt"}, 404

    try:
        # Read and analyze story
        story_text = story_file.read_text(encoding="utf-8", errors="replace")
        model = data.get("model", "hf.co/mradermacher/mistralai-Mistral-Nemo-Instruct-2407-extensive-BP-abliteration-12B-GGUF:Q4_K_M")

        summaries_dir = _summaries_dir(analyze_lang)
        summaries_dir.mkdir(exist_ok=True)

        profile, story_context = analyze_story(story_text, model, lang=analyze_lang)

        # Enforce role limits before enrichment (avoid wasting LLM calls on dropped chars)
        profile = _trim_characters(profile)

        # Second pass: fill in physical descriptions, loves/hates, secondary details
        profile = enrich_character_profile(profile, story_context, model, lang=analyze_lang)

        # Save character summary as markdown
        char_md = _profile_to_markdown(profile, lang=analyze_lang)
        char_file = summaries_dir / f"{story_name}_character.md"
        char_file.write_text(char_md, encoding="utf-8")

        # Build story.md from profile fields using role names (not character names)
        story_md = _profile_to_story_markdown(profile, lang=analyze_lang)
        story_file_md = summaries_dir / f"{story_name}_story.md"
        story_file_md.write_text(story_md, encoding="utf-8")

        # Save enriched profile as JSON (for game use)
        profile_file = summaries_dir / f"{story_name}_profile.json"
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
        lang = (request.args.get("lang") or "en").lower()
        char_file = _summaries_dir(lang) / f"{story_name}_character.md"
        return {"analyzed": char_file.exists()}
    except Exception:
        return {"analyzed": False}


@app.route("/check-translation/<story_name>")
def check_translation(story_name):
    """Check if an English story has already been translated to JA."""
    try:
        story_name = story_name.strip().replace(".txt", "")
        translated = STORIES_DIR_JA / f"{story_name}.txt"
        return {"translated": translated.exists()}
    except Exception:
        return {"translated": False}


@app.route("/translate", methods=["POST"])
def translate():
    """Stream translate_story.py output as SSE for a given EN story."""
    data = request.json or {}
    story_name = (data.get("story") or "").strip().replace(".txt", "")
    model = data.get("model") or "qwen-ja"

    if not story_name:
        return {"error": "No story name provided"}, 400

    story_file = STORIES_DIR / f"{story_name}.txt"
    try:
        story_file = story_file.resolve()
        story_file.relative_to(STORIES_DIR.resolve())
    except (ValueError, RuntimeError):
        return {"error": "Invalid story path"}, 400

    if not story_file.exists():
        return {"error": f"Story not found: {story_name}.txt"}, 404

    translate_script = Path(__file__).parent / "translate_story.py"

    def generate():
        try:
            proc = subprocess.Popen(
                [sys.executable, "-u", str(translate_script), str(story_file), "--model", model],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"},
            )
            for line in proc.stdout:
                yield f"data: {json.dumps(line.rstrip())}\n\n"
            proc.wait()
            if proc.returncode == 0:
                yield "data: {\"done\": true}\n\n"
            else:
                yield f"data: {{\"error\": \"Process exited with code {proc.returncode}\"}}\n\n"
        except Exception as e:
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


@app.route("/start", methods=["POST"])
def start():
    data = request.json
    model = data.get("model", "hf.co/mradermacher/mistralai-Mistral-Nemo-Instruct-2407-extensive-BP-abliteration-12B-GGUF:Q4_K_M")

    story_lang = (data.get("lang") or "en").lower()
    if story_lang not in ("en", "ja"):
        story_lang = "en"

    stories_src = _stories_dir(story_lang)
    all_stories = list(stories_src.glob("*.txt")) if stories_src.exists() else []
    if not all_stories:
        return {"error": f"No .txt files found in {stories_src}"}, 400

    story_pick = data.get("story", "random")
    if story_pick and story_pick != "random":
        # Remove .txt extension if included
        story_name = story_pick.replace(".txt", "")
        chosen = (stories_src / f"{story_name}.txt").resolve()
        try:
            chosen.relative_to(stories_src.resolve())
        except ValueError:
            return {"error": "Invalid story path"}, 400
        if not chosen.exists():
            return {"error": f"Story not found: {story_pick}"}, 400
    else:
        chosen = random.choice(all_stories)
        story_name = chosen.stem

    # Check if summaries exist for this story (in the language-scoped folder)
    summaries_dir = _summaries_dir(story_lang)
    char_file = summaries_dir / f"{story_name}_character.md"
    profile_file = summaries_dir / f"{story_name}_profile.json"
    story_md_file = summaries_dir / f"{story_name}_story.md"

    try:
        if profile_file.exists() and story_md_file.exists():
            # Load from saved summaries
            profile = json.loads(profile_file.read_text(encoding="utf-8"))
            story_context = story_md_file.read_text(encoding="utf-8")
        else:
            story_text = chosen.read_text(encoding="utf-8", errors="replace")
            profile, story_context = analyze_story(story_text, model, lang=story_lang)
    except Exception as e:
        return {"error": str(e)}, 500

    # story_lang already parsed above; drives segmenter + voice set.
    if story_lang == "ja":
        _FEMALE_NAMES = ["美咲", "さくら", "ゆい", "あかり", "花音", "千尋", "葵", "凛", "七海", "結衣"]
        _MALE_NAMES   = ["翔太", "健", "直樹", "涼", "蓮", "悠斗", "大輝", "颯", "拓海", "遥斗"]
        _PRONOUN_STOPS = ("you", "her", "she", "him", "he", "they",
                          "あなた", "彼", "彼女")
    else:
        _FEMALE_NAMES = ["Amelia", "Clara", "Elena", "Isla", "Lyra", "Mara", "Nora", "Rose", "Sarah", "Vera"]
        _MALE_NAMES   = ["Alex", "Daniel", "Ethan", "James", "Liam", "Marcus", "Noah", "Oliver", "Ryan", "Sebastian"]
        _PRONOUN_STOPS = ("you", "her", "she", "him", "he", "they")

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
    if not other_name or other_name.lower() in _PRONOUN_STOPS:
        profile["other_name"] = random.choice(_FEMALE_NAMES if other_gender == "female" else _MALE_NAMES)

    player_name = profile.get("player_name", "").strip()
    if not player_name or player_name.lower() in _PRONOUN_STOPS:
        profile["player_name"] = random.choice(_MALE_NAMES if player_gender == "male" else _FEMALE_NAMES)

    # Assign PRIMARY voices deterministically by gender.
    # Secondary characters that enter the scene use SECONDARY voices (handled in /tts).
    if story_lang == "ja":
        primary_m = (JA_PRIMARY_MALE_VOICE,   JA_LANG)
        primary_f = (JA_PRIMARY_FEMALE_VOICE, JA_LANG)
        secondary_m = (JA_SECONDARY_MALE_VOICE,   JA_LANG)
        secondary_f = (JA_SECONDARY_FEMALE_VOICE, JA_LANG)
    else:
        primary_m = (PRIMARY_MALE_CHARACTER_VOICE,   PRIMARY_MALE_CHARACTER_LANG)
        primary_f = (PRIMARY_FEMALE_CHARACTER_VOICE, PRIMARY_FEMALE_CHARACTER_LANG)
        secondary_m = (SECONDARY_MALE_CHARACTER_VOICE,   SECONDARY_MALE_CHARACTER_LANG)
        secondary_f = (SECONDARY_FEMALE_CHARACTER_VOICE, SECONDARY_FEMALE_CHARACTER_LANG)

    other_voice, other_lang = primary_m if other_gender == "male" else primary_f

    # When both characters share a gender, use the secondary voice for the player
    # so they're distinguishable in TTS.
    if player_gender == other_gender:
        player_voice, player_lang = secondary_m if player_gender == "male" else secondary_f
    else:
        player_voice, player_lang = primary_m if player_gender == "male" else primary_f

    other_pronoun  = "he" if other_gender == "male" else "she"
    player_pronoun = "you"  # player is always narrated in second person

    system_prompt = build_system_prompt(profile, story_context, lang=story_lang)

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
    session["lang"]           = story_lang

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
        "options": {"num_ctx": 8192},
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
    is_ja = session.get("lang", "en") == "ja"
    if is_ja:
        suggest_system = (
            "あなたは創作アシスタントです。プレイヤーの行動候補を提案するのが唯一の役割です。"
            "必ず有効なJSON配列（3つの日本語文字列）のみを返してください。"
            "前置きやマークダウン、中国語、英語は一切混ぜないこと。"
        )
        suggest_prompt = (
            "このシーンで、プレイヤーが次に取り得る行動・台詞を3つ、自然な日本語で提案してください。"
            "それぞれ1〜2文。プレイヤー（「私」または主語省略）の視点で書くこと。"
            "トーンを変えること：1つは大胆・直接的、1つは優しく温かい、1つは慎重・控えめ。"
            "必ず日本語（中国語や英語を混ぜない）。"
            "JSON配列のみを返すこと。コードフェンスや説明文は一切含めない。\n"
            '例：["彼女の手を取る。", "「教えて」と静かに言う。", "視線を逸らし、言葉を探す。"]'
        )
    else:
        suggest_system = (
            "You are a creative writing assistant. Your only job is to suggest player actions. "
            "You must respond with ONLY a valid JSON array of exactly 3 strings. No prose, no markdown."
        )
        suggest_prompt = (
            "Suggest exactly 3 brief actions or responses the player could take next in this scene. "
            "Each must be 1–2 sentences, written in first person as something the player does or says. "
            "Vary the tone: one bold/direct, one tender/warm, one cautious/indirect. "
            "Return ONLY a JSON array of 3 strings — no prose, no markdown fences, no commentary. "
            'Example: ["I reach for her hand.", "\\"Tell me,\\" I say softly.", "I look away, unsure."]'
        )
    recent = [m for m in session["messages"] if m["role"] != "system"]
    context = recent[-6:]
    messages = [{"role": "system", "content": suggest_system}] + context + [{"role": "user", "content": suggest_prompt}]

    payload = json.dumps({
        "model": session["model"],
        "messages": messages,
        "stream": False,
        "options": {"num_ctx": 8192},
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
        story_lang = session.get("lang", "en")
        other_gender = "male" if (session.get("other_pronoun") == "he") else "female"

        if story_lang == "ja":
            segments = _parse_segments_ja(
                text,
                player_name=session["player_name"] or "",
                other_name=session["other_name"] or "",
                other_gender=other_gender,
                secondary_characters=session.get("secondary_characters", {}),
            )
        else:
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
            seg_text = _scrub_for_tts(seg_text.strip())
            if not seg_text:
                continue
            # Log what's actually going to Kokoro — helps diagnose leaked
            # annotations, unexpected prefixes, etc.
            print(f"[tts/{speaker}] {seg_text[:80]}{'…' if len(seg_text) > 80 else ''}")
            if speaker == "player":
                voice, lang = session["player_voice"], session["player_lang"]
            elif speaker == "other":
                voice, lang = session["other_voice"], session["other_lang"]
            else:  # narrator / secondary_male / secondary_female
                voice, lang = _voice_for(speaker, story_lang)
            # Kokoro caps each synthesis call at 510 phoneme tokens. Chunk on
            # sentence boundaries first, then fall back to recursive splitting
            # on overflow (kanji-heavy JA can still blow the budget even after
            # sentence-level chunking).
            for chunk in _chunk_for_tts(seg_text, story_lang):
                for piece in _synthesize_with_fallback(kokoro, chunk, voice, lang):
                    parts.append(piece)
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
