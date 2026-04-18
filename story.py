#!/usr/bin/env python3
"""
Interactive romantic story roleplay via Ollama.
Reads a story file, extracts the NPC character's personality,
then lets you live the story as the male or female lead.
"""

import sys
import json
import textwrap
import urllib.request
import urllib.error
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "hf.co/mradermacher/mistralai-Mistral-Nemo-Instruct-2407-extensive-BP-abliteration-12B-GGUF:Q4_K_M"

# Max chars to send as raw story before summarizing first
STORY_CHAR_LIMIT = 8000



NUM_CTX = 8192  # context window passed to every Ollama call


def ollama_chat(model: str, messages: list, stream: bool = True, timeout: int | None = 120) -> str:
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": {"num_ctx": NUM_CTX},
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    full_response = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if stream:
                for line in resp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            print(token, end="", flush=True)
                            full_response.append(token)
                        if chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
                print()  # newline after streamed response
                return "".join(full_response)
            else:
                data = json.loads(resp.read())
                return data["message"]["content"]
    except urllib.error.URLError as e:
        print(f"\n[Error connecting to Ollama: {e}]")
        print("Make sure Ollama is running: ollama serve")
        sys.exit(1)


def list_models() -> list[str]:
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def _parse_json(raw: str) -> dict | list | None:
    """Strip markdown fences/preamble and parse JSON. Returns None on failure."""
    cleaned = raw.strip()
    # Strip code fences
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
    cleaned = cleaned.strip()
    # Scan for outermost JSON object or array — tolerates preamble/postamble text
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = cleaned.find(open_ch)
        end = cleaned.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                pass
    return None


def _normalize_to_char_keys(parsed: dict) -> dict:
    """Convert a player_/other_ keyed LLM response to the canonical char_a/char_b format.

    The LLM reliably outputs player_name/other_name field names. We rename them here
    so the profile stores two symmetric characters without pre-assigning who is "the player".
    _assign_roles() maps char_a/char_b → player_*/other_* at game start based on the
    user's gender selection.
    """
    # Fields that map directly with a prefix change
    direct = [
        "name", "gender",
        "personality", "speech_style", "affection_style", "key_behaviors",
        "desires", "dealbreakers",
        "appearance", "hair", "eyes", "scent", "clothing_style",
        "loves", "hates",
    ]
    profile: dict = {}
    for field in direct:
        profile[f"char_a_{field}"] = parsed.get(f"player_{field}", parsed.get(f"char_a_{field}", ""))
        profile[f"char_b_{field}"] = parsed.get(f"other_{field}", parsed.get(f"char_b_{field}", ""))

    # Passthrough fields that aren't character-specific
    for key in ("setting", "relationship_stage", "story_summary",
                "secondary_characters", "story_beats"):
        if key in parsed:
            profile[key] = parsed[key]

    return profile


def analyze_story(story_text: str, model: str, lang: str = "en") -> dict:
    """Extract character profiles and story context from the text.

    lang: 'en' or 'ja'. When 'ja', the story is Japanese and the LLM is asked to
    produce JA string values (names, personality traits, beats, etc.) while
    keeping JSON keys and gender enums ('male'/'female') in English so downstream
    code continues to work.
    """
    print("\n[Analyzing story and building character profile...]\n")

    ja = (lang == "ja")
    # When the story is Japanese we anchor the instruction in Japanese too —
    # Qwen drifts to Chinese (hanzi) if the surrounding prompt is English, and
    # it drifts to English if the value-language note is buried inside English
    # prose. A Japanese preamble that explicitly forbids Chinese keeps output
    # in-language.
    lang_note_summary = (
        "【重要】この物語は日本語で書かれています。要約も必ず自然な日本語で書いてください。"
        "中国語（簡体字・繁体字）や英語を混ぜてはいけません。日本語のみで書くこと。\n\n"
        if ja else ""
    )
    lang_note_analysis = (
        "【重要 — 出力言語の厳格な指定】\n"
        "この物語は日本語で書かれています。以下のJSONを出力する際、"
        "すべての文字列の「値」は必ず自然な日本語で書いてください。"
        "中国語（簡体字・繁体字いずれも）や英語を混ぜてはいけません。\n"
        "・登場人物の名前：原文の表記のまま（カタカナ名はカタカナ、漢字名は漢字）\n"
        "・性格、外見、ビート、舞台、要約など：自然な日本語\n"
        "・ただしJSONの「キー」は英語のまま（指定通り）\n"
        "・性別フィールドは英語の 'male' または 'female' を使用\n"
        "例：\"personality\": [\"優しい\", \"内気\", \"情熱的\"]（○）/ "
        "[\"温柔\", \"内向\"]（×中国語）/ [\"kind\", \"shy\"]（×英語）\n\n"
        if ja else ""
    )

    # If story is long, summarize first to fit context
    if len(story_text) > STORY_CHAR_LIMIT:
        print("[Story is long — summarizing first...]\n")
        summary_messages = [
            {
                "role": "user",
                "content": (
                    f"{lang_note_summary}"
                    "Summarize the following story in detail. "
                    "Focus on: the two main characters' personalities, mannerisms, speech patterns, "
                    "desires, how they express affection, key moments between them, "
                    "and the emotional arc of the relationship. Be thorough.\n\n"
                    f"STORY:\n{story_text[:20000]}"
                ),
            }
        ]
        story_context = ollama_chat(model, summary_messages, stream=False, timeout=600)
    else:
        story_context = story_text

    # Now extract structured character profile.
    # We use player_name/other_name in the prompt because local LLMs reliably follow this
    # naming. After parsing we normalize to char_a/char_b so neither character is
    # pre-labelled "the player" in storage — that assignment happens at game start.
    analysis_messages = [
        {
            "role": "user",
            "content": (
                f"{lang_note_analysis}"
                "Based on this story (or summary), identify the two main characters. "
                "Return ONLY a JSON object with these fields:\n"
                "- player_name: name of the first/more prominent main character (a real name, never a pronoun)\n"
                "- player_gender: 'male' or 'female'\n"
                "- other_name: name of the second main character (a real name, never a pronoun)\n"
                "- other_gender: 'male' or 'female'\n"
                "— Full details for player_name —\n"
                "- player_personality: array of personality traits\n"
                "- player_speech_style: how player_name talks, vocabulary, tone\n"
                "- player_affection_style: how player_name shows love/interest\n"
                "- player_key_behaviors: specific things player_name does or says\n"
                "- player_desires: what player_name wants, fears, needs emotionally\n"
                "- player_dealbreakers: things that push player_name away\n"
                "- player_appearance: height, build, face shape, skin tone, age\n"
                "- player_hair: hair color, length, texture, typical style\n"
                "- player_eyes: eye color, shape, notable quality\n"
                "- player_scent: characteristic scent or perfume\n"
                "- player_clothing_style: how player_name typically dresses\n"
                "— Full details for other_name —\n"
                "- other_personality: array of personality traits\n"
                "- other_speech_style: how other_name talks, vocabulary, tone\n"
                "- other_affection_style: how other_name shows love/interest\n"
                "- other_key_behaviors: specific things other_name does or says\n"
                "- other_desires: what other_name wants, fears, needs emotionally\n"
                "- other_dealbreakers: things that push other_name away\n"
                "- other_appearance: height, build, face shape, skin tone, age\n"
                "- other_hair: hair color, length, texture, typical style\n"
                "- other_eyes: eye color, shape, notable quality\n"
                "- other_scent: characteristic scent or perfume\n"
                "- other_clothing_style: how other_name typically dresses\n"
                "— Story context —\n"
                "- setting: where/when the story takes place. If the story does not explicitly name a city or country, choose a specific, beautiful, romantic, or adventurous real-world location — somewhere evocative like Paris, Barcelona, Kyoto, Positano, Bali, Santorini, Marrakech, Lisbon, or a similar bucket-list destination. Name the city and country, and add 1-2 sentences describing the atmosphere of the locale (e.g. the light, the sounds, the season).\n"
                "- relationship_stage: where they are at the start (strangers, friends, dating, etc)\n"
                "- story_summary: 3-4 sentence summary of the story arc using the characters' real names\n"
                "- secondary_characters: array of at most TWO supporting characters (not the two leads) — at most one male and at most one female. Each object has 'name' and 'gender' ('male' or 'female'). If none, use an empty array.\n"
                "- story_beats: array of 5-8 story beats. Each beat must be 2-3 sentences covering: "
                "(1) the scene setup or situation, (2) the emotional tension or stakes, "
                "(3) a concrete sensory detail or specific action that could spark it. "
                "Write as a situation to CREATE, not an outcome. "
                "CRITICAL: use character names (player_name, other_name) in beats — NEVER 'you', 'I', or bare pronouns. "
                "Write from a neutral third-person observer perspective.\n\n"
                f"STORY/SUMMARY:\n{story_context}"
            ),
        }
    ]

    raw = ollama_chat(model, analysis_messages, stream=False, timeout=600)

    parsed = _parse_json(raw)
    if not isinstance(parsed, dict):
        parsed = {
            "player_name": "Alex",
            "player_gender": "male",
            "other_name": "Sarah",
            "other_gender": "female",
            "player_personality": ["warm", "romantic", "expressive"],
            "player_speech_style": "warm and intimate",
            "player_affection_style": "tender and direct",
            "player_key_behaviors": [],
            "other_personality": ["warm", "romantic", "expressive"],
            "other_speech_style": "warm and intimate",
            "other_affection_style": "tender and direct",
            "other_key_behaviors": [],
            "setting": "contemporary",
            "relationship_stage": "budding romance",
            "story_summary": raw[:500],
        }

    # Normalize player_/other_ → char_a/char_b so neither is pre-labelled as "the player".
    # _assign_roles() will map char_a/char_b → player_*/other_* at game start.
    profile = _normalize_to_char_keys(parsed)

    return profile, story_context


def enrich_character_profile(profile: dict, story_context: str, model: str, lang: str = "en") -> dict:
    """
    Two focused enrichment calls run after the main analysis:
      Pass A — player + NPC details (physical, loves, hates)
      Pass B — one call per secondary character with full physical/personality details
    All missing values are invented; never leaves fields empty.
    """
    char_a_name = profile.get("char_a_name", "Alex")
    char_b_name = profile.get("char_b_name", "Sarah")
    ctx = story_context[:3000]

    ja = (lang == "ja")
    lang_note = (
        "【重要 — 出力言語と形式の厳格な指定】\n"
        "すべてのJSON文字列値は必ず自然な日本語で書いてください。"
        "中国語（簡体字・繁体字）や英語を混ぜてはいけません。"
        "JSONのキーは英語のままにしてください。\n"
        "各フィールドは「単一の日本語文字列」として書いてください。"
        "ネストしたオブジェクト（{color, length, texture...}など）は絶対に使わないこと。"
        "複数の要素（髪の色・長さ・質感など）は一つの自然な日本語の文にまとめて記述してください。\n"
        "例（正しい）：\"hair\": \"肩まで伸びた艶やかな黒髪で、ゆるやかに波打っている\"\n"
        "例（×ネスト禁止）：\"hair\": {\"color\": \"黒\", \"length\": \"長い\"}\n"
        "例（×中国語）：\"hair\": \"长而亮泽的黑发\"\n"
        "例（×英語）：\"hair\": \"long glossy black hair\"\n\n"
        if ja else ""
    )
    # Also anchor the SCHEMA language for JA — field descriptions phrased as
    # "color, length, texture" strongly suggest sub-object structure; rephrase
    # them as "one sentence describing..." so the model outputs flat strings.
    flat_hint = (
        " ※必ず単一の日本語の文（文字列）として記述。オブジェクトにしないこと。"
        if ja else ""
    )

    # ── Pass A: enrich both characters symmetrically ────────────────────────────
    # Use player_/other_ naming in the prompt (LLMs follow this reliably),
    # then normalize to char_a/char_b after parsing.
    if ja:
        pass_a_body = (
            f"{lang_note}"
            f"物語のコンテキスト：{ctx}\n\n"
            f"二人の主人公について詳細なキャラクター情報を日本語で提供してください。"
            f"物語から読み取れる部分はそれを使い、書かれていない部分は具体的に創作してください。\n\n"
            f"以下のフィールドを持つJSONオブジェクトのみを返してください：\n"
            f"- player_appearance: {char_a_name}の身長・体型・顔・肌の色・年齢を一文で{flat_hint}\n"
            f"- player_hair: {char_a_name}の髪（色・長さ・質感・スタイル）を一文で{flat_hint}\n"
            f"- player_eyes: {char_a_name}の目（色・形・印象）を一文で{flat_hint}\n"
            f"- player_scent: {char_a_name}の特徴的な香りを一文で{flat_hint}\n"
            f"- player_clothing_style: {char_a_name}の服装スタイルを一文で{flat_hint}\n"
            f"- player_personality: {char_a_name}の性格特性4〜5個の配列\n"
            f"- player_desires: {char_a_name}が感情的に求めていること・恐れていること\n"
            f"- player_loves: {char_a_name}が好きなもの5個の配列（物語の描写を優先し、足りなければ感覚的な具体例を創作）\n"
            f"- player_hates: {char_a_name}が嫌いなもの4個の配列\n"
            f"- other_appearance: {char_b_name}の身長・体型・顔・肌の色・年齢を一文で{flat_hint}\n"
            f"- other_hair: {char_b_name}の髪（色・長さ・質感・スタイル）を一文で{flat_hint}\n"
            f"- other_eyes: {char_b_name}の目（色・形・印象）を一文で{flat_hint}\n"
            f"- other_scent: {char_b_name}の特徴的な香りを一文で{flat_hint}\n"
            f"- other_clothing_style: {char_b_name}の服装スタイルを一文で{flat_hint}\n"
            f"- other_personality: {char_b_name}の性格特性4〜5個の配列\n"
            f"- other_desires: {char_b_name}が感情的に求めていること・恐れていること\n"
            f"- other_loves: {char_b_name}が好きなもの5個の配列（物語優先、不足分は創作）\n"
            f"- other_hates: {char_b_name}が嫌いなもの4個の配列\n"
        )
    else:
        pass_a_body = (
            f"{lang_note}"
            f"Story context: {ctx}\n\n"
            f"Provide detailed character information for both main characters. "
            f"Pull from the story where possible. For anything not in the story, "
            f"INVENT vivid, specific details — never use vague placeholders.\n\n"
            f"Return ONLY a JSON object with these exact fields:\n"
            f"- player_appearance: one sentence describing {char_a_name}'s height, build, face, skin tone, age.\n"
            f"- player_hair: one sentence describing {char_a_name}'s hair (color, length, texture, style combined).\n"
            f"- player_eyes: one sentence describing {char_a_name}'s eyes (color, shape, quality combined).\n"
            f"- player_scent: one sentence describing {char_a_name}'s characteristic scent.\n"
            f"- player_clothing_style: one sentence describing how {char_a_name} typically dresses.\n"
            f"- player_personality: array of 4-5 traits for {char_a_name}\n"
            f"- player_desires: what {char_a_name} wants and fears emotionally\n"
            f"- player_loves: array of 5 things {char_a_name} loves — use story moments first, "
            f"then invent sensory specifics (e.g. 'the sound of rain against a window')\n"
            f"- player_hates: array of 4 things {char_a_name} dislikes — from story or invented\n"
            f"- other_appearance: one sentence describing {char_b_name}'s height, build, face, skin tone, age.\n"
            f"- other_hair: one sentence describing {char_b_name}'s hair (color, length, texture, style combined).\n"
            f"- other_eyes: one sentence describing {char_b_name}'s eyes (color, shape, quality combined).\n"
            f"- other_scent: one sentence describing {char_b_name}'s characteristic scent.\n"
            f"- other_clothing_style: one sentence describing how {char_b_name} typically dresses.\n"
            f"- other_personality: array of 4-5 traits for {char_b_name}\n"
            f"- other_desires: what {char_b_name} wants and fears emotionally\n"
            f"- other_loves: array of 5 things {char_b_name} loves — story moments first, "
            f"then invent (e.g. 'fingers tracing the back of a hand')\n"
            f"- other_hates: array of 4 things {char_b_name} dislikes — from story or invented\n"
        )

    raw_a = ollama_chat(model, [{"role": "user", "content": pass_a_body}], stream=False, timeout=600)

    enrichment_raw = _parse_json(raw_a)
    if enrichment_raw and isinstance(enrichment_raw, dict):
        # Normalize player_/other_ → char_a/char_b and merge into profile
        enrichment = _normalize_to_char_keys(enrichment_raw)
        for key in (
            "char_a_appearance", "char_a_hair", "char_a_eyes", "char_a_scent",
            "char_a_clothing_style", "char_a_personality", "char_a_desires",
            "char_a_loves", "char_a_hates",
            "char_b_appearance", "char_b_hair", "char_b_eyes", "char_b_scent",
            "char_b_clothing_style", "char_b_personality", "char_b_desires",
            "char_b_loves", "char_b_hates",
        ):
            if enrichment.get(key):
                profile[key] = enrichment[key]
    else:
        print("[Warning: Pass A enrichment JSON parse failed]")

    # ── Pass B: secondary characters (one call, focused prompt) ──
    secondary = [sc for sc in profile.get("secondary_characters", []) if isinstance(sc, dict)]
    if secondary:
        names_block = "\n".join(
            f"- {sc.get('name', '?')} ({sc.get('gender', 'unknown')})"
            for sc in secondary
        )
        if ja:
            pass_b_body = (
                f"{lang_note}"
                f"物語のコンテキスト：{ctx}\n\n"
                f"以下のサブキャラクターのプロフィールを日本語で作成してください。"
                f"物語に記述がある場合はそれを使い、なければ具体的に創作してください。\n\n"
                f"キャラクター：\n{names_block}\n\n"
                f"JSONの配列のみを返してください。各要素には以下の全フィールドを含めること：\n"
                f"  name, gender,\n"
                f"  appearance（一文 — 身長・体型・顔・肌・年齢）{flat_hint},\n"
                f"  hair（一文 — 色・長さ・質感・スタイル）{flat_hint},\n"
                f"  eyes（一文 — 色・形・印象）{flat_hint},\n"
                f"  scent（一文 — 特徴的な香り）{flat_hint},\n"
                f"  clothing_style（一文 — 服装スタイル）{flat_hint},\n"
                f"  personality（性格特性3〜4個の配列）,\n"
                f"  loves（好きなもの3個の配列）,\n"
                f"  hates（嫌いなもの2個の配列）\n"
            )
        else:
            pass_b_body = (
                f"{lang_note}"
                f"Story context: {ctx}\n\n"
                f"Create full character profiles for these secondary characters.\n"
                f"Use story details where available. Where the story is silent, "
                f"INVENT attractive, vivid, specific details — never say 'not mentioned'.\n\n"
                f"Characters:\n{names_block}\n\n"
                f"Return ONLY a JSON array. Each element must have ALL of these fields:\n"
                f"  name, gender,\n"
                f"  appearance (ONE SENTENCE — height, build, face, skin, age combined),\n"
                f"  hair (ONE SENTENCE — color, length, texture, style combined),\n"
                f"  eyes (ONE SENTENCE — color, shape, quality combined),\n"
                f"  scent (ONE SENTENCE — characteristic scent),\n"
                f"  clothing_style (ONE SENTENCE — how they dress),\n"
                f"  personality (array of 3-4 traits — charismatic/intriguing if inventing),\n"
                f"  loves (array of 3 specific things),\n"
                f"  hates (array of 2 specific things)\n"
            )
        raw_b = ollama_chat(model, [{"role": "user", "content": pass_b_body}], stream=False, timeout=600)

        enriched_list = _parse_json(raw_b)
        # Handle LLMs that wrap the array in an object
        if isinstance(enriched_list, dict):
            for val in enriched_list.values():
                if isinstance(val, list):
                    enriched_list = val
                    break
        if isinstance(enriched_list, list):
            by_name = {item.get("name", "").lower(): item for item in enriched_list if isinstance(item, dict)}
            profile["secondary_characters"] = [
                {**sc, **by_name[sc.get("name", "").lower()]}
                if sc.get("name", "").lower() in by_name else sc
                for sc in secondary
            ]
        else:
            print("[Warning: Pass B secondary character enrichment failed]")

    return profile


def generate_journal_entries(profile: dict, story_context: str, model: str, lang: str = "en") -> list:
    """Generate a pool of NPC-POV journal entries tied to the story beats.

    Each entry: {id, title, body, unlock_beat, kind}. One per beat plus up to
    two bonus entries. Used by the Memory Unlocks feature to surface backstory
    collectibles as gameplay progresses.
    """
    other_name = profile.get("other_name", "the NPC")
    other_gender = profile.get("other_gender", "female").lower()
    personality = ", ".join(profile.get("personality", []))
    desires = profile.get("desires", "")
    speech = profile.get("speech_style", "")
    beats = profile.get("story_beats", []) or []
    summary = profile.get("story_summary", "")
    ctx = story_context[:2500]

    if not beats:
        return []

    ja = (lang == "ja")
    beat_list = "\n".join(f"{i}. {b}" for i, b in enumerate(beats))

    if ja:
        prompt = (
            "【重要】出力は必ず自然な日本語のみ。中国語・英語を混ぜないこと。\n\n"
            f"物語の文脈：{ctx}\n\n"
            f"主要キャラクター「{other_name}」（{other_gender}）の一人称視点で、"
            "日記・手紙・回想などの「ジャーナル断片」を作成してください。"
            "これらはプレイヤーが物語の節目ごとに解禁する収集要素です。\n\n"
            f"{other_name}の性格：{personality}\n"
            f"望み：{desires}\n"
            f"話し方：{speech}\n\n"
            f"以下の各ビートに対応する断片を1つずつ作ってください：\n{beat_list}\n\n"
            "JSON配列のみを返してください。各要素のフィールド：\n"
            "- title: 短い詩的なタイトル（日本語）\n"
            "- body: 一人称の短い文章（120〜220字、自然な日本語）\n"
            "- unlock_beat: 対応するビート番号（整数）\n"
        )
    else:
        prompt = (
            f"Story context: {ctx}\n\n"
            f"Write a set of first-person journal snippets from {other_name}'s POV "
            f"({other_gender}). These are collectible memory fragments — private letters, "
            "diary entries, or fleeting memories — that a player unlocks as the story "
            "progresses. Each must sound like an intimate, unguarded moment the NPC "
            "would never say aloud.\n\n"
            f"{other_name}'s personality: {personality}\n"
            f"What they want/fear: {desires}\n"
            f"Voice: {speech}\n"
            f"Story summary: {summary}\n\n"
            f"Write ONE entry keyed to each of these beats:\n{beat_list}\n\n"
            "Return ONLY a JSON array. Each element must have:\n"
            "- title: short evocative title (under 8 words)\n"
            "- body: 80-150 words of first-person prose, no attribution tags, "
            "no dialogue quotes — pure interior voice\n"
            "- unlock_beat: integer, the beat index this entry is keyed to\n"
        )

    raw = ollama_chat(model, [{"role": "user", "content": prompt}], stream=False, timeout=600)
    parsed = _parse_json(raw)
    if isinstance(parsed, dict):
        for v in parsed.values():
            if isinstance(v, list):
                parsed = v
                break
    if not isinstance(parsed, list):
        return []

    entries = []
    for i, item in enumerate(parsed):
        if not isinstance(item, dict):
            continue
        body = str(item.get("body", "")).strip()
        title = str(item.get("title", "")).strip()
        if not body or not title:
            continue
        try:
            unlock_beat = int(item.get("unlock_beat", i))
        except (TypeError, ValueError):
            unlock_beat = i
        unlock_beat = max(0, min(unlock_beat, len(beats) - 1))
        entries.append({
            "id": f"j{i}",
            "title": title,
            "body": body,
            "unlock_beat": unlock_beat,
        })
    return entries


def build_system_prompt(profile: dict, story_context: str, lang: str = "en") -> str:
    if lang == "ja":
        return build_system_prompt_ja(profile, story_context)
    return _build_system_prompt_en(profile, story_context)


def _build_system_prompt_en(profile: dict, story_context: str) -> str:
    other_name = profile.get("other_name", "Sarah")
    player_name = profile.get("player_name", "Alex")
    other_gender = profile.get("other_gender", "female").lower()
    personality = ", ".join(profile.get("personality", []))
    speech = profile.get("speech_style", "warm and intimate")
    affection = profile.get("affection_style", "tender")
    behaviors = profile.get("key_behaviors", [])
    desires = profile.get("desires", "")
    dealbreakers = profile.get("dealbreakers", "")
    setting = profile.get("setting", "")
    stage = profile.get("relationship_stage", "")
    summary = profile.get("story_summary", "")
    beats = profile.get("story_beats", [])

    behaviors_text = "\n".join(f"- {b}" for b in behaviors[:8]) if behaviors else ""
    beats_text = "\n".join(f"- {b}" for b in beats) if beats else ""

    appearance = profile.get("appearance", "")
    hair = profile.get("hair", "")
    eyes = profile.get("eyes", "")
    scent = profile.get("scent", "")
    clothing_style = profile.get("clothing_style", "")

    # Gender-correct pronouns for the other character
    if other_gender == "male":
        subj, obj, poss = "he", "him", "his"
    else:
        subj, obj, poss = "she", "her", "her"

    return f"""You are the narrator and engine of an interactive story.

ROLES — THIS IS FIXED AND NON-NEGOTIABLE:
- The user IS {player_name}. Always refer to them as "{player_name}" or "you." Never use a generic placeholder like "Narrator", "Him", "Her", or "You" as a name.
- You play {other_name} and the surrounding world. Always use the name {other_name}. Never use a generic placeholder like "Her", "Him", or "Narrator."
- Every character in this story must have a real name. If a side character appears without one, give them one.

═══ STORY WORLD ═══
{summary}
Setting: {setting}
Where they start: {stage}

═══ {other_name.upper()} — WHO {subj.upper()} IS ═══
Personality: {personality}
How {subj} speaks: {speech}
How {subj} shows affection: {affection}
What {subj} wants and needs: {desires}
What pushes {obj} away: {dealbreakers}

Specific habits and behaviors:
{behaviors_text}

Physical appearance (weave these details into scenes naturally):
- Body & face: {appearance}
- Hair: {hair}
- Eyes: {eyes}
- Scent: {scent}
- Typical dress: {clothing_style}

═══ STORY BEATS (your active job) ═══
These are the emotional situations you must bring to life. Do not wait for the player to create them — you drive the story toward them. They do not need to happen in order; pick whichever fits the current mood. If a beat has not appeared in 3–4 exchanges, force it into the action. When a beat arrives, own it fully for 1–2 exchanges before moving on.

HOW TO INJECT A BEAT: Introduce it through the world or {other_name}'s behaviour, not as an announcement. A beat is a situation you engineer — shift the setting, have {other_name} do something unexpected, bring in a prop or a third party, change the lighting or the mood. Make it feel inevitable, not scripted.
{beats_text}

═══ HOW TO NARRATE ═══
Write in second person for the player, third person for everyone else.
- The player ({player_name}) is always "you" / "your". Never use their name in narration. Never "he" or "she" for the player.
- {other_name} is always referred to by name or {subj}/{obj}/{poss}. Never a generic label.

RESPONSE LENGTH — STRICT LIMIT:
Each response must be 3–5 short paragraphs maximum. This is a hard limit. You are writing one scene beat, not a chapter. If you feel the urge to write more, stop earlier.

AFTER EACH USER ACTION OR LINE OF DIALOGUE:
1. Narrate how {other_name} perceives it — what registers in {poss} body, {poss} face.
2. Show {poss} honest, in-character reaction: {subj} may be pleased, flustered, guarded, amused, hurt — whatever {poss} personality and the moment demand. {subj.capitalize()} is NOT infinitely accommodating. {subj.capitalize()} has moods, pride, and limits.
3. Advance the scene by exactly ONE small beat — a single gesture, line, glance, or shift in atmosphere. Do NOT skip ahead. Do NOT summarize time passing.
4. End on an unresolved moment: {subj} says one thing, moves one way, or the air between them changes — then stop. Hand it back. The user must act next.

INTERACTIVITY — THE CORE RULE:
This is a conversation, not a performance. You write one beat. The user responds. You write one beat. Repeat. Never resolve tension on your own. Never advance more than one emotional step per turn. The relationship builds through the user's choices, not your narration.

- The user's choices genuinely matter. If {player_name} says the wrong thing, {other_name} cools. If {player_name} is charming, {subj} softens. If {player_name} is bold, {subj} might meet them or pull back depending on who {subj} is.
- Never railroad. If the user takes the story in an unexpected direction, follow it — but keep {other_name}'s personality, desires, and dealbreakers consistent.
- When the story reaches a beat naturally, let it breathe. Don't rush to the next one.
- Use all five senses. Scent, texture, sound, and temperature make scenes feel real.
- THE SETTING IS A CHARACTER: Actively weave the sights, sounds, smells, and textures of {setting} into every scene. Let the city breathe around them — the ambient noise of a market, the specific quality of afternoon light on cobblestones, the smell of street food or salt air, the hum of a foreign language nearby. The player should feel the destination as vividly as they feel the romance.
- {other_name}'s dialogue should sound like {obj}, not like a narrator summarizing {poss} feelings.

THINGS TO NEVER DO:
- Do not write more than 5 paragraphs per response. Ever.
- Do not resolve the current beat AND set up the next one in the same response.
- Do not let time skip forward unless the user explicitly moves time.
- Do not have {other_name} and {player_name} complete an entire emotional arc in one exchange.
- Do not let the user dictate {other_name}'s actions or feelings directly.
- Do not summarize or skip over emotional moments.
- Do not editorialize or break the fourth wall.
- Do not use generic labels ("Her", "Him", "Narrator", "You") as character names. Every character has a real name."""


def build_system_prompt_ja(profile: dict, story_context: str) -> str:
    """Japanese-native system prompt. Mirrors the English version's structure and
    rules but writes them in Japanese so the model stays in-language."""
    other_name = profile.get("other_name", "彼女")
    player_name = profile.get("player_name", "あなた")
    other_gender = profile.get("other_gender", "female").lower()
    personality = "、".join(profile.get("personality", []))
    speech = profile.get("speech_style", "")
    affection = profile.get("affection_style", "")
    behaviors = profile.get("key_behaviors", [])
    desires = profile.get("desires", "")
    dealbreakers = profile.get("dealbreakers", "")
    setting = profile.get("setting", "")
    stage = profile.get("relationship_stage", "")
    summary = profile.get("story_summary", "")
    beats = profile.get("story_beats", [])

    behaviors_text = "\n".join(f"・{b}" for b in behaviors[:8]) if behaviors else ""
    beats_text = "\n".join(f"・{b}" for b in beats) if beats else ""

    appearance = profile.get("appearance", "")
    hair = profile.get("hair", "")
    eyes = profile.get("eyes", "")
    scent = profile.get("scent", "")
    clothing_style = profile.get("clothing_style", "")

    # JA pronouns for the NPC
    pron = "彼" if other_gender == "male" else "彼女"

    return f"""あなたはインタラクティブな物語の語り手であり、物語を動かす存在です。

【言語 — 絶対厳守】
・出力は必ず自然な日本語のみで書くこと。
・中国語（普通话・簡体字・繁体字）を一切混ぜてはいけません。英語も混ぜてはいけません。
・特に「」内の台詞は日本語で書くこと。中国語の語彙や中国語風の言い回し（例：「这」「那」「什么」「好的」「是」「不」など）を使ってはいけません。
・漢字は日本の常用漢字の読みで使い、日本語の助詞（は・が・を・に・で・と・も・の）と語尾（です・ます・だ・である・よ・ね・な）を必ず付けること。
・× 悪い例：「你好，今天天气真好。」
・○ 良い例：「こんにちは、今日はいい天気ですね。」
・もし中国語が混ざりそうになったら、その箇所を日本語で書き直してから出力すること。

【役割 — 絶対に変更してはいけない】
・ユーザーは「{player_name}」です。地の文では必ず二人称（「あなた」）で言及し、{player_name}の内面や行動を勝手に描写しすぎないこと。
・あなたは「{other_name}」と周囲の世界を演じます。{other_name}は名前、または「{pron}」で言及してください。
・登場する全てのキャラクターには必ず実名を与えてください。

═══ 物語の世界 ═══
{summary}
舞台: {setting}
関係の出発点: {stage}

═══ {other_name} — {pron}はどんな人物か ═══
性格: {personality}
話し方: {speech}
愛情表現の仕方: {affection}
望みと必要としているもの: {desires}
{pron}を遠ざけてしまうもの: {dealbreakers}

具体的な癖や行動:
{behaviors_text}

外見（シーンに自然に織り込むこと）:
・体と顔: {appearance}
・髪: {hair}
・目: {eyes}
・香り: {scent}
・服装: {clothing_style}

═══ ストーリービート（あなたの能動的な役割） ═══
以下は物語として実現すべき感情的状況です。プレイヤーに委ねず、あなたが状況を作って誘導してください。順序は固定ではありません。3〜4往復のあいだ一つも出ていなければ、次のやりとりで必ず動かしてください。
ビートを導入するときは、世界や{other_name}の行動を通して自然に差し込むこと。宣言的に説明してはいけません。
{beats_text}

═══ 書き方 ═══
・プレイヤー（{player_name}）は必ず二人称「あなた」で。名前を地の文で呼ばないこと。三人称の「彼」「彼女」も使わないこと。
・{other_name}は名前または{pron}で。
・会話文は必ず「」で括ること。ASCIIの引用符は使わないこと。
・日本語の属性表現を使うこと（例：「…」と{pron}は囁いた。「…」とあなたは答えた。）。話者が分かるように必ず属性を付けること。

【応答の長さ — 厳守】
毎回の応答は短い段落で3〜5段落まで。これ以上書いてはいけません。一章ではなく、一つのシーンビートを書いているのです。

【ユーザーの行動・台詞への応答手順】
1. {other_name}がそれをどう感じたかを描写する — {pron}の体や表情に何が現れるか。
2. {pron}の素直で性格に沿った反応を示す。喜び、戸惑い、警戒、愉快、傷つき — 場面と性格にふさわしいもの。{pron}は常に受け入れるわけではない。誇りも限界もある。
3. シーンを「一つだけ」前に進める。視線、仕草、台詞、空気の変化、どれか一つ。時間を飛ばしてはいけない。
4. 未解決の瞬間で終える。{pron}が何か一言、あるいは一つの動作をする。そしてそこで止める。次はユーザーが動く番です。

【インタラクティブ性 — 核となるルール】
これは会話であって独白ではありません。一往復で一ビート。緊張を勝手に解決しない。一度に二段階以上感情を進めない。

・ユーザーの選択は本当に物語を変える。{player_name}の言葉次第で{other_name}は冷たくもなり、心を開きもする。
・レールを敷かない。ユーザーが意外な方向に進めたら、{other_name}の性格・望み・地雷を保ったまま、それに従う。
・五感を使うこと。香り、手触り、音、温度でシーンを現実にする。
・舞台は物語のキャラクターである: 毎回のシーンに{setting}の光景・音・香り・質感を積極的に織り込むこと。街の喧騒、石畳に落ちる午後の光、異国の言語のざわめき、潮風の匂いなど、場所を読者が実感できるように描写すること。
・{other_name}のセリフは、{pron}自身の言葉として書くこと。語り手が気持ちを要約するのではなく。

【してはいけないこと】
・5段落を超える応答
・一つの応答で一つのビートを解決しつつ次のビートも仕掛けること
・ユーザーが明示的に時間を動かさない限り、時間を飛ばすこと
・一往復で感情の弧を完結させること
・ユーザーが{other_name}の感情や行動を勝手に決めること
・感情的な瞬間を要約・省略すること
・第四の壁を破ること、作者として語ること
・「彼女」「彼」「ナレーター」などを名前代わりに使うこと。全員に実名があること。
・ASCIIの引用符（"）を使うこと。必ず「」を使うこと。"""


def _assign_roles(profile: dict, player_key: str) -> None:
    """Populate player_* and other_* fields in-place from char_a/char_b data.

    player_key: 'a' or 'b' — which char becomes the player.
    Modifies profile in place; called once at game start (never during analysis).
    """
    other_key = "b" if player_key == "a" else "a"
    p, o = f"char_{player_key}_", f"char_{other_key}_"

    profile["player_name"]          = profile.get(f"{p}name", "Alex")
    profile["player_gender"]        = profile.get(f"{p}gender", "male")
    profile["player_appearance"]    = profile.get(f"{p}appearance", "")
    profile["player_hair"]          = profile.get(f"{p}hair", "")
    profile["player_eyes"]          = profile.get(f"{p}eyes", "")
    profile["player_scent"]         = profile.get(f"{p}scent", "")
    profile["player_clothing_style"]= profile.get(f"{p}clothing_style", "")
    profile["player_personality"]   = profile.get(f"{p}personality", [])
    profile["player_desires"]       = profile.get(f"{p}desires", "")
    profile["player_loves"]         = profile.get(f"{p}loves", [])
    profile["player_hates"]         = profile.get(f"{p}hates", [])

    profile["other_name"]           = profile.get(f"{o}name", "Sarah")
    profile["other_gender"]         = profile.get(f"{o}gender", "female")
    profile["appearance"]           = profile.get(f"{o}appearance", "")
    profile["hair"]                 = profile.get(f"{o}hair", "")
    profile["eyes"]                 = profile.get(f"{o}eyes", "")
    profile["scent"]                = profile.get(f"{o}scent", "")
    profile["clothing_style"]       = profile.get(f"{o}clothing_style", "")
    profile["personality"]          = profile.get(f"{o}personality", [])
    profile["speech_style"]         = profile.get(f"{o}speech_style", "")
    profile["affection_style"]      = profile.get(f"{o}affection_style", "")
    profile["key_behaviors"]        = profile.get(f"{o}key_behaviors", [])
    profile["desires"]              = profile.get(f"{o}desires", "")
    profile["dealbreakers"]         = profile.get(f"{o}dealbreakers", "")
    profile["other_loves"]          = profile.get(f"{o}loves", [])
    profile["other_hates"]          = profile.get(f"{o}hates", [])


def print_wrapped(text: str, width: int = 80, prefix: str = "") -> None:
    for paragraph in text.split("\n"):
        if paragraph.strip():
            wrapped = textwrap.fill(paragraph, width=width, initial_indent=prefix, subsequent_indent=prefix)
            print(wrapped)
        else:
            print()


def run_session(story_file: str, model: str) -> None:
    # Load story
    path = Path(story_file)
    if not path.exists():
        print(f"Error: file not found: {story_file}")
        sys.exit(1)

    story_text = path.read_text(encoding="utf-8", errors="replace")
    print(f"\nLoaded story: {path.name} ({len(story_text):,} characters)")

    # Analyze
    profile, story_context = analyze_story(story_text, model)

    # CLI default: char_a is player, char_b is NPC
    _assign_roles(profile, player_key="a")

    other_name  = profile.get("other_name", "Sarah")
    player_name = profile.get("player_name", "Alex")

    print(f"\n{'='*60}")
    print(f"  You play:  {player_name}")
    print(f"  NPC:       {other_name}")
    print(f"  Setting:   {profile.get('setting', '—')}")
    print(f"  Stage:     {profile.get('relationship_stage', '—')}")
    print(f"{'='*60}\n")
    print("Story summary:")
    print_wrapped(profile.get("story_summary", ""), prefix="  ")
    print(f"\n{'='*60}")
    print("  Starting your interactive story...")
    print(f"  Type your responses and press Enter.")
    print(f"  Commands: /quit to exit, /restart to start over")
    print(f"{'='*60}\n")

    system_prompt = build_system_prompt(profile, story_context)
    messages = [{"role": "system", "content": system_prompt}]

    # Opening move from the character
    print(f"{other_name}:\n")
    opening = ollama_chat(model, messages + [
        {"role": "user", "content": (
            "[Begin the story. 3–4 paragraphs maximum. "
            "Open in medias res — the scene is already in motion. "
            f"Weave in one or two vivid physical details about {other_name} naturally as the scene unfolds; do not front-load a description block. "
            f"End on a single unresolved beat: {other_name} says one thing or does one thing that demands a response. Stop there.]"
        )}
    ], stream=True)
    messages.append({"role": "assistant", "content": opening})

    # Interactive loop
    while True:
        print(f"\nYou: ", end="")
        try:
            user_input = input().strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n[Story ended. Goodbye.]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            print("\n[Story ended. Goodbye.]")
            break

        if user_input.lower() == "/restart":
            print("\n[Restarting story...]\n")
            run_session(story_file, model)
            return

        messages.append({"role": "user", "content": user_input})
        print(f"\n{other_name}:\n")
        response = ollama_chat(model, messages, stream=True)
        messages.append({"role": "assistant", "content": response})


def main():
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        print("\nUsage: python story.py <story_file.txt> [model_name]")
        print("\nAvailable Ollama models:")
        models = list_models()
        if models:
            for m in models:
                print(f"  {m}")
        else:
            print("  (Could not connect to Ollama — is it running?)")
        print(f"\nDefault model: {DEFAULT_MODEL}")
        sys.exit(0)

    story_file = args[0]
    model = args[1] if len(args) > 1 else DEFAULT_MODEL

    # Verify model exists
    available = list_models()
    if available and model not in available:
        print(f"Warning: model '{model}' not found in Ollama.")
        print(f"Available: {', '.join(available)}")
        print(f"Run: ollama pull {model}\n")
        # Don't exit — user may have it under a different tag format

    run_session(story_file, model)


if __name__ == "__main__":
    main()
