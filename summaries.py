"""Profile-to-Markdown conversion for story character and story summary files."""

# Localized heading/fallback labels for the generated markdown files.
_MD_LABELS = {
    "en": {
        "appearance":    "Appearance",
        "hair":          "Hair",
        "eyes":          "Eyes",
        "scent":         "Scent",
        "style":         "Style",
        "personality":   "Personality",
        "loves":         "What They Love",
        "hates":         "What They Hate",
        "desires":       "Desires & Fears",
        "speaks":        "How They Speak",
        "affection":     "How They Show Affection",
        "behaviors":     "Key Behaviors",
        "pushes_away":   "What Pushes Them Away",
        "setting":       "Setting",
        "starting":      "Starting Point",
        "summary":       "Summary",
        "beats":         "Story Beats",
        "not_specified": "_Not specified_",
        "not_provided":  "_Not provided_",
        "none_provided": "_None provided_",
        "traits_sep":    ", ",
    },
    "ja": {
        "appearance":    "外見",
        "hair":          "髪",
        "eyes":          "目",
        "scent":         "香り",
        "style":         "服装",
        "personality":   "性格",
        "loves":         "好きなもの",
        "hates":         "嫌いなもの",
        "desires":       "望みと恐れ",
        "speaks":        "話し方",
        "affection":     "愛情の示し方",
        "behaviors":     "特徴的な行動",
        "pushes_away":   "遠ざけるもの",
        "setting":       "舞台",
        "starting":      "関係の出発点",
        "summary":       "あらすじ",
        "beats":         "ストーリービート",
        "not_specified": "_未記載_",
        "not_provided":  "_未記載_",
        "none_provided": "_なし_",
        "traits_sep":    "、",
    },
}


def _str(val) -> str:
    """Flatten any LLM-returned value to a plain string (guards against [object Object])."""
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        return ", ".join(str(v) for v in val.values() if v)
    if isinstance(val, list):
        return " · ".join(str(v) for v in val if v)
    return str(val) if val else ""


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
    """Convert an analyzed profile dict to character sections (char_a / char_b format)."""
    L = _MD_LABELS.get(lang, _MD_LABELS["en"])

    sections = []
    for key in ("a", "b"):
        name   = profile.get(f"char_{key}_name", f"Character {key.upper()}")
        gender = profile.get(f"char_{key}_gender", "female").lower()
        role   = "PRIMARY_MALE" if gender == "male" else "PRIMARY_FEMALE"
        data = {
            "appearance":     profile.get(f"char_{key}_appearance"),
            "hair":           profile.get(f"char_{key}_hair"),
            "eyes":           profile.get(f"char_{key}_eyes"),
            "scent":          profile.get(f"char_{key}_scent"),
            "clothing_style": profile.get(f"char_{key}_clothing_style"),
            "personality":    profile.get(f"char_{key}_personality", []),
            "loves":          profile.get(f"char_{key}_loves", []),
            "hates":          profile.get(f"char_{key}_hates", []),
            "desires":        profile.get(f"char_{key}_desires"),
        }
        section = _character_section(role, name, data, lang=lang)

        speech_block = ""
        if profile.get(f"char_{key}_speech_style"):
            speech_block += f"\n## {L['speaks']}\n{profile[f'char_{key}_speech_style']}\n"
        if profile.get(f"char_{key}_affection_style"):
            speech_block += f"\n## {L['affection']}\n{profile[f'char_{key}_affection_style']}\n"
        behaviors = profile.get(f"char_{key}_key_behaviors", [])
        if behaviors:
            speech_block += f"\n## {L['behaviors']}\n" + "\n".join(f"- {b}" for b in behaviors) + "\n"
        if profile.get(f"char_{key}_dealbreakers"):
            speech_block += f"\n## {L['pushes_away']}\n{profile[f'char_{key}_dealbreakers']}\n"
        if speech_block:
            section = section.rstrip() + "\n" + speech_block

        sections.append(section)

    for sc in profile.get("secondary_characters", []):
        sc_name   = sc.get("name", "Unknown")
        sc_gender = sc.get("gender", "unknown").lower()
        sc_role   = "SECONDARY_MALE" if sc_gender == "male" else "SECONDARY_FEMALE"
        sc_data   = {
            "appearance":     sc.get("appearance"),
            "hair":           sc.get("hair"),
            "eyes":           sc.get("eyes"),
            "scent":          sc.get("scent"),
            "clothing_style": sc.get("clothing_style"),
            "personality":    sc.get("personality", []),
            "loves":          sc.get("loves", []),
            "hates":          sc.get("hates", []),
            "desires":        sc.get("desires"),
        }
        sections.append(_character_section(sc_role, sc_name, sc_data, lang=lang))

    return "\n\n---\n\n".join(sections)


def _profile_to_story_markdown(profile: dict, lang: str = "en") -> str:
    """Build story.md from profile fields. Uses character names directly in beats/summary."""
    L = _MD_LABELS.get(lang, _MD_LABELS["en"])
    summary    = profile.get("story_summary", L["not_provided"])
    beats      = profile.get("story_beats", [])
    beats_text = "\n".join(f"{i+1}. {b}" for i, b in enumerate(beats)) if beats else L["none_provided"]
    title      = "物語" if lang == "ja" else "Story"

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
