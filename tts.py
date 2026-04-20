"""TTS engine: voice constants, segment parsing, Kokoro synthesis."""

import io
import re
import struct
import threading
from pathlib import Path

KOKORO_MODEL  = Path(__file__).parent / "kokoro_models" / "kokoro-v1.0.onnx"
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
JA_NARRATOR_VOICE         = "jf_alpha"
JA_PRIMARY_MALE_VOICE     = "jm_kumo"
JA_PRIMARY_FEMALE_VOICE   = "jf_gongitsune"
JA_SECONDARY_MALE_VOICE   = "jm_kumo"      # Kokoro ships only one jm_ voice
JA_SECONDARY_FEMALE_VOICE = "jf_nezumi"
JA_LANG                   = "ja"


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


def _voice_for(speaker: str, lang: str) -> tuple[str, str]:
    """Return (voice, lang) for a narrator/secondary speaker role."""
    if lang == "ja":
        mapping = {
            "narrator":         (JA_NARRATOR_VOICE,         JA_LANG),
            "secondary_male":   (JA_SECONDARY_MALE_VOICE,   JA_LANG),
            "secondary_female": (JA_SECONDARY_FEMALE_VOICE, JA_LANG),
        }
    else:
        mapping = {
            "narrator":         (NARRATOR_VOICE,                   NARRATOR_LANG),
            "secondary_male":   (SECONDARY_MALE_CHARACTER_VOICE,   SECONDARY_MALE_CHARACTER_LANG),
            "secondary_female": (SECONDARY_FEMALE_CHARACTER_VOICE, SECONDARY_FEMALE_CHARACTER_LANG),
        }
    return mapping.get(speaker, mapping["narrator"])


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
# Matches "Name said/whispered/etc." — a capitalized proper name followed by a speech verb,
# used to attribute a quoted line to a named character without relying on pronouns.
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
    for m in re.finditer(rf'\byou\s+{_SPEECH_VERBS}\b', narr, re.IGNORECASE):
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
            continue
        has_she = bool(re.search(rf'\b{re.escape(m.group(1))}\b[^.!?]{{0,60}}\bshe\b', narr, re.IGNORECASE))
        has_he  = bool(re.search(rf'\b{re.escape(m.group(1))}\b[^.!?]{{0,60}}\bhe\b',  narr, re.IGNORECASE))
        if has_she and not has_he:
            candidates.append((m.start(), "secondary_female", True))
        elif has_he and not has_she:
            candidates.append((m.start(), "secondary_male", True))
        elif other_pronoun == "she":
            candidates.append((m.start(), "secondary_male", True))
        else:
            candidates.append((m.start(), "secondary_female", True))

    # Pronoun-based attribution (weaker signal — only used when no name match exists)
    if other_pronoun:
        for m in re.finditer(rf'\b{re.escape(other_pronoun)}\s+{_SPEECH_VERBS}\b', narr, re.IGNORECASE):
            candidates.append((m.start(), "other", False))

    if other_pronoun == "she":
        for m in re.finditer(rf'\bhe\s+{_SPEECH_VERBS}\b', narr, re.IGNORECASE):
            candidates.append((m.start(), "secondary_male", False))
    elif other_pronoun == "he":
        for m in re.finditer(rf'\bshe\s+{_SPEECH_VERBS}\b', narr, re.IGNORECASE):
            candidates.append((m.start(), "secondary_female", False))

    if candidates:
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

        post_raw = clean[m.end():m.end() + 120]
        sent_end = re.search(r'[.!?]', post_raw)
        post_narr = post_raw[:sent_end.end()].strip() if sent_end else post_raw.strip()

        speaker = _detect_speaker(post_narr, player_name, other_name, other_pronoun,
                                  secondary_characters)

        if speaker == "other" and narr:
            last_sent_end = None
            for sent_m in re.finditer(r'[.!?]\s+', narr):
                last_sent_end = sent_m.end()
            if last_sent_end is not None:
                pre_clause = narr[last_sent_end:].strip()
            else:
                pre_clause = narr[-60:].strip()
            if pre_clause:
                pre_speaker = _detect_speaker(pre_clause, player_name, other_name,
                                             other_pronoun, secondary_characters)
                if pre_speaker != "other":
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
        last_end = 0
        for m in re.finditer(r'[。！？]', narr_before):
            last_end = m.end()
        pre_clause = narr_before[last_end:] if last_end else narr_before[-60:]
        if _ja_has_attribution(pre_clause):
            attr = pre_clause
        else:
            return "other"

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

    sent_end_re  = re.compile(r'[。！？]' if lang == "ja" else r'[.!?]')
    soft_break_re = re.compile(r'[、]'   if lang == "ja" else r'[,;:]')

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
