#!/usr/bin/env python3
"""
Web server for the interactive story roleplay app.
Run: python server.py
Then open http://localhost:5000
"""

import json
import os
import random
import re
import subprocess
import sys
import threading
import urllib.request
import urllib.error
from pathlib import Path
from flask import Flask, request, Response, send_file, stream_with_context

from story import analyze_story, enrich_character_profile, build_system_prompt, generate_journal_entry, list_models, _assign_roles, OLLAMA_URL
import affinity
from tts import (
    get_kokoro, get_ja_g2p,
    _float32_to_wav, _voice_for,
    _parse_segments, _parse_segments_ja,
    _chunk_for_tts, _scrub_for_tts, _synthesize_with_fallback,
    NARRATOR_VOICE, NARRATOR_LANG,
    PRIMARY_MALE_CHARACTER_VOICE, PRIMARY_MALE_CHARACTER_LANG,
    PRIMARY_FEMALE_CHARACTER_VOICE, PRIMARY_FEMALE_CHARACTER_LANG,
    SECONDARY_MALE_CHARACTER_VOICE, SECONDARY_MALE_CHARACTER_LANG,
    SECONDARY_FEMALE_CHARACTER_VOICE, SECONDARY_FEMALE_CHARACTER_LANG,
    JA_PRIMARY_MALE_VOICE, JA_PRIMARY_FEMALE_VOICE,
    JA_SECONDARY_MALE_VOICE, JA_SECONDARY_FEMALE_VOICE, JA_LANG,
)
from summaries import _profile_to_markdown, _profile_to_story_markdown, _str, _trim_characters

DEFAULT_MODEL = "nemo"

STORIES_DIR = Path(__file__).parent / "stories"
STORIES_DIR_JA = Path(__file__).parent / "stories_ja"
SUMMARIES_DIR = Path(__file__).parent / "story_summaries"
SUMMARIES_DIR_JA = Path(__file__).parent / "story_summaries_jp"
LOGS_DIR = Path(__file__).parent / "logs"
SAVES_DIR = Path(__file__).parent / "saves"

def _summaries_dir(lang: str) -> Path:
    return SUMMARIES_DIR_JA if (lang or "").lower() == "ja" else SUMMARIES_DIR

def _stories_dir(lang: str) -> Path:
    return STORIES_DIR_JA if (lang or "").lower() == "ja" else STORIES_DIR

def _append_log(role: str, content: str) -> None:
    """Append a single message turn to the active session log file."""
    log_path = session.get("log_file")
    if not log_path:
        return
    try:
        import datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n### [{ts}] {role.upper()}\n\n{content}\n")
    except OSError:
        pass


# How many user turns between context compressions
_COMPRESS_EVERY = 10
# How many recent exchanges (user+assistant pairs) to keep verbatim after compression
_KEEP_RECENT_EXCHANGES = 3


def _compress_history() -> None:
    """Summarize the conversation history and replace old messages with the summary.

    Keeps the system prompt and the most recent _KEEP_RECENT_EXCHANGES exchanges
    verbatim. Everything older is replaced with a single injected assistant message
    containing a concise narrative recap.
    """
    import datetime

    messages = session.get("messages", [])
    system_msg = messages[0] if messages and messages[0]["role"] == "system" else None

    # Separate non-system messages
    history = [m for m in messages if m["role"] != "system"]

    # Pair up user/assistant turns; keep last N pairs verbatim
    pairs = []
    i = 0
    unpaired_user_at_end = None
    while i < len(history) - 1:
        if history[i]["role"] == "user" and history[i + 1]["role"] == "assistant":
            pairs.append((history[i], history[i + 1]))
            i += 2
        else:
            i += 1

    # Check if there's an unpaired user message at the end (no assistant response yet)
    if history and history[-1]["role"] == "user":
        unpaired_user_at_end = history[-1]

    to_summarize = pairs[:-_KEEP_RECENT_EXCHANGES] if len(pairs) > _KEEP_RECENT_EXCHANGES else []
    keep_pairs   = pairs[-_KEEP_RECENT_EXCHANGES:]

    if not to_summarize:
        return  # nothing old enough to compress

    # Build a flat transcript for the summarization prompt
    transcript_parts = []
    for user_m, asst_m in to_summarize:
        transcript_parts.append(f"Player: {user_m['content']}")
        transcript_parts.append(f"Story: {asst_m['content']}")
    transcript = "\n\n".join(transcript_parts)

    lang = session.get("lang", "en")
    if lang == "ja":
        summary_instruction = (
            "以下はインタラクティブストーリーの会話履歴です。"
            "登場人物の関係性、重要な出来事、感情の変化、伏線を含む簡潔な要約を書いてください。"
            "散文形式で3〜5段落にまとめ、説明や前置きは不要です。"
        )
    else:
        summary_instruction = (
            "The following is a transcript from an interactive story roleplay session. "
            "Write a concise narrative recap covering: the characters' relationship dynamic, "
            "key events and emotional beats, any plot threads introduced, and where things stand. "
            "Plain prose, 3–5 paragraphs, no preamble or meta-commentary."
        )

    sum_messages = [
        {"role": "system", "content": summary_instruction},
        {"role": "user",   "content": transcript},
    ]

    payload = json.dumps({
        "model": session["model"],
        "messages": sum_messages,
        "stream": False,
        "options": {"num_ctx": 8192, "num_predict": 600},
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
        summary_text = result.get("message", {}).get("content", "").strip()
    except Exception as e:
        summary_text = f"[Summary unavailable: {e}]"

    # Inject summary as a special assistant message so the model has full context
    summary_injection = {
        "role": "assistant",
        "content": (
            "[STORY SO FAR — narrative recap of earlier events]\n\n"
            f"{summary_text}\n\n"
            "[End of recap. Continuing story from here.]"
        ),
    }

    # Reconstruct message list: system + summary + recent verbatim pairs + unpaired user (if any)
    new_messages = []
    if system_msg:
        new_messages.append(system_msg)
    new_messages.append(summary_injection)
    for user_m, asst_m in keep_pairs:
        new_messages.append(user_m)
        new_messages.append(asst_m)
    if unpaired_user_at_end:
        new_messages.append(unpaired_user_at_end)

    before_count = len(messages)
    after_count  = len(new_messages)
    session["messages"] = new_messages

    # Log the compression event
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    log_path = session.get("log_file")
    if log_path:
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"\n---\n\n"
                    f"### [{ts}] CONTEXT COMPRESSION\n\n"
                    f"- Messages before: {before_count}  →  after: {after_count}\n"
                    f"- Summarized {len(to_summarize)} exchange(s), kept {len(keep_pairs)} verbatim\n\n"
                    f"**Summary injected:**\n\n{summary_text}\n\n"
                    f"---\n"
                )
        except OSError:
            pass

def _generate_save_summary(lang: str = "en") -> str:
    """Generate a comprehensive multi-paragraph narrative summary of the current session
    for save file. Unlike _compress_history, this does not mutate session state."""
    messages = session.get("messages", [])
    if not messages or len(messages) < 2:
        return ""

    # Build transcript from all non-system messages
    transcript_lines = []
    for m in messages:
        if m["role"] == "system":
            continue
        role = m["role"].upper()
        if role == "ASSISTANT":
            role = session.get("other_name", "NPC").upper()
        elif role == "USER":
            role = "PLAYER"
        transcript_lines.append(f"{role}: {m['content']}")

    if not transcript_lines:
        return ""

    transcript = "\n\n".join(transcript_lines)
    if lang == "ja":
        sys_prompt = (
            "このストーリーセッションの包括的な叙述的要約を書いてください。保存ファイルに使用します。"
            "カバーする内容: キャラクター間の関係ダイナミクス、すべての重要なイベントと転換点、"
            "感情的なビート、どのように変化したか、未解決のスレッドや緊張。平文で、3～5段落です。"
            "このサマリーは、プレイヤーがストーリーを再開するときにコンテキストを再構築するために使用されます。"
        )
    else:
        sys_prompt = (
            "Write a comprehensive narrative recap of this story session for use as a save file. "
            "Cover: the relationship dynamic between the characters, all significant events and turning points, "
            "emotional beats and how they shifted, any unresolved threads or tensions. Plain prose, 3–5 paragraphs. "
            "This will be used to reconstruct context when the player resumes the story later."
        )

    from story import ollama_chat
    result = ollama_chat(
        session.get("model", ""),
        [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Transcript:\n\n{transcript}"},
        ],
        stream=False,
    )
    return (result or "").strip()


app = Flask(__name__)


# ── In-memory session store (single user) ─────────────────────
# All character fields are None until /start assigns them.
session = {
    "messages":       [],
    "profile":        None,
    "model":          DEFAULT_MODEL,
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
    "story_name":     None,
    "affinity":       None,
    "affinity_history": [],
    "suggest_count":  0,
    "suggest_next_env": 3,
}

_affinity_lock = threading.Lock()


def _score_turn_async(profile_snap, msgs_snap, current_snap, model, lang):
    try:
        delta = affinity.score_turn(profile_snap, msgs_snap, current_snap, model, lang)
        print(f"[affinity] score_turn delta: {delta}", flush=True)
        if not delta:
            return
        with _affinity_lock:
            current = session.get("affinity")
            if not current:
                print("[affinity] no current affinity in session", flush=True)
                return
            # Drop if we've fallen too far behind live turns (user typed faster than scorer).
            if session.get("affinity_turn_counter", current["turn"]) > current["turn"] + 3:
                print("[affinity] dropped (too far behind)", flush=True)
                return
            new_current, entry = affinity.apply_delta(current, delta, current["turn"] + 1)
            session["affinity"] = new_current
            print(f"[affinity] applied: trust={new_current['trust']} intimacy={new_current['intimacy']} tension={new_current['tension']} reason={delta.get('reason','')}", flush=True)
            history = session.setdefault("affinity_history", [])
            history.append(entry)
            if len(history) > affinity.HISTORY_CAP:
                del history[: len(history) - affinity.HISTORY_CAP]
            # Refresh the system prompt in messages[0] so the story LLM sees the updated state.
            profile = session.get("profile")
            story_context = session.get("story_context", "")
            messages = session.get("messages")
            if profile and messages and messages[0]["role"] == "system":
                messages[0]["content"] = build_system_prompt(
                    profile, story_context, lang=lang, affinity=new_current
                )
    except Exception as e:
        import traceback
        print(f"[affinity] _score_turn_async ERROR: {e}\n{traceback.format_exc()}", flush=True)


@app.route("/")
def index():
    return send_file("index.html")


@app.route("/index_stories")
def index_stories():
    return send_file("index_stories.html")


@app.route("/models")
def models():
    return {"models": list_models()}


def _extract_story_summary(story_md_path: Path) -> str:
    """Extract the ## Summary section from a _story.md file."""
    try:
        text = story_md_path.read_text(encoding="utf-8")
        in_section = False
        lines = []
        for line in text.splitlines():
            if line.strip() == "## Summary":
                in_section = True
                continue
            if in_section:
                if line.startswith("## "):
                    break
                lines.append(line)
        return "\n".join(lines).strip()
    except Exception:
        return ""


@app.route("/stories")
def stories():
    lang = (request.args.get("lang") or "en").lower()
    analyzed_only = request.args.get("analyzed") == "1"
    src_dir = _stories_dir(lang)
    all_txt = sorted(p.name for p in src_dir.glob("*.txt")) if src_dir.exists() else []
    summaries_dir = _summaries_dir(lang)
    if analyzed_only:
        if summaries_dir.exists():
            available = {p.name.replace("_character.md", "")
                         for p in summaries_dir.glob("*_character.md")}
            all_txt = [s for s in all_txt if s.replace(".txt", "") in available]
    result = []
    for name in all_txt:
        story_name = name.replace(".txt", "")
        summary = ""
        if summaries_dir.exists():
            summary = _extract_story_summary(summaries_dir / f"{story_name}_story.md")
        result.append({"name": name, "summary": summary})
    return {"stories": result}


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
        model = data.get("model", DEFAULT_MODEL)

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

        # Journals are generated lazily at game start (after player/NPC roles are known).

        char_a = profile.get("char_a_name", "?")
        char_b = profile.get("char_b_name", "?")
        return {
            "success": True,
            "story": story_name,
            "character": f"{char_a} / {char_b}",
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Analysis failed: {str(e)}"}, 500


@app.route("/check-analysis/<story_name>")
def check_analysis(story_name):
    """Check if a story has been analyzed (summaries exist)."""
    try:
        story_name = story_name.strip()
        lang = (request.args.get("lang") or "en").lower()
        base = _summaries_dir(lang).resolve()
        char_file = (base / f"{story_name}_character.md").resolve()
        char_file.relative_to(base)
        return {"analyzed": char_file.exists()}
    except (ValueError, RuntimeError):
        return {"analyzed": False}, 400
    except Exception:
        return {"analyzed": False}


@app.route("/check-translation/<story_name>")
def check_translation(story_name):
    """Check if an English story has already been translated to JA."""
    try:
        story_name = story_name.strip().replace(".txt", "")
        base = STORIES_DIR_JA.resolve()
        translated = (base / f"{story_name}.txt").resolve()
        translated.relative_to(base)
        return {"translated": translated.exists()}
    except (ValueError, RuntimeError):
        return {"translated": False}, 400
    except Exception:
        return {"translated": False}


@app.route("/translate", methods=["POST"])
def translate():
    """Stream translate_story.py output as SSE for a given EN story."""
    data = request.json or {}
    story_name = (data.get("story") or "").strip().replace(".txt", "")
    model = data.get("model") or "qwen-ja"
    if not re.match(r'^[\w\.\-/:]+$', model):
        return {"error": "Invalid model name"}, 400

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
    data = request.json or {}
    model = data.get("model", DEFAULT_MODEL)
    story_lang = (data.get("lang") or "en").lower()
    if story_lang not in ("en", "ja"):
        story_lang = "en"

    summaries_dir = _summaries_dir(story_lang)
    analyzed = [p.stem.replace("_profile", "") for p in summaries_dir.glob("*_profile.json")] if summaries_dir.exists() else []
    if not analyzed:
        return {"error": f"No analyzed stories found in {summaries_dir}. Use /index_stories to analyze a story first."}, 400

    story_pick = data.get("story", "random")
    if story_pick and story_pick != "random":
        story_name = story_pick.replace(".txt", "")
    else:
        story_name = random.choice(analyzed)

    profile_file = summaries_dir / f"{story_name}_profile.json"
    story_md_file = summaries_dir / f"{story_name}_story.md"

    try:
        profile = json.loads(profile_file.read_text(encoding="utf-8"))
        story_context = story_md_file.read_text(encoding="utf-8")
    except Exception as e:
        return {"error": str(e)}, 500

    # story_lang already parsed above; drives segmenter + voice set.
    if story_lang == "ja":
        _FEMALE_NAMES = ["美咲", "さくら", "ゆい", "あかり", "花音", "千尋", "葵", "凛", "七海", "結衣"]
        _MALE_NAMES   = ["翔太", "健", "直樹", "涼", "蓮", "悠斗", "大輝", "颯", "拓海", "遥斗"]
        _PRONOUN_STOPS = ("you", "her", "she", "him", "he", "they",
                          "あなた", "彼", "彼女")
    else:
        _FEMALE_NAMES = ["Amelia", "Clara", "Elena", "Isla", "Lyra", "Mara", "Natalie", "Rose", "Sarah", "Vera"]
        _MALE_NAMES   = ["Alex", "Daniel", "Ethan", "James", "Liam", "Marcus", "Noah", "Oliver", "Ryan", "Sebastian"]
        _PRONOUN_STOPS = ("you", "her", "she", "him", "he", "they")

    # Detect old-format profiles (pre-char_a/char_b redesign) and reject them.
    if "char_a_name" not in profile:
        return {"error": "Story needs re-analysis. Please open the indexer and click Analyze."}, 400

    # Assign player/NPC based on user gender preference — no swapping.
    # Analysis stores both characters as char_a/char_b (gender-neutral).
    # We pick which char becomes player here, at game start.
    preferred_gender = data.get("player_gender", "auto").lower()
    char_a_gender = profile.get("char_a_gender", "female").lower()
    char_b_gender = profile.get("char_b_gender", "male").lower()

    if preferred_gender == "female":
        player_key = "a" if char_a_gender == "female" else "b"
    elif preferred_gender == "male":
        player_key = "a" if char_a_gender == "male" else "b"
    else:
        # "auto" — char_a is the natural first character
        player_key = "a"

    _assign_roles(profile, player_key)

    other_gender  = profile.get("other_gender", "female").lower()
    player_gender = profile.get("player_gender", "male").lower()

    # Ensure both characters have real names; always assign Japanese names in JA mode.
    other_name = profile.get("other_name", "").strip()
    if story_lang == "ja" or not other_name or other_name.lower() in _PRONOUN_STOPS:
        profile["other_name"] = random.choice(_FEMALE_NAMES if other_gender == "female" else _MALE_NAMES)

    player_name = profile.get("player_name", "").strip()
    if story_lang == "ja" or not player_name or player_name.lower() in _PRONOUN_STOPS:
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

    system_prompt = build_system_prompt(profile, story_context, lang=story_lang,
                                        affinity=session.get("affinity"))

    # Build secondary character registry: {name_lower: gender}
    secondary_characters = {}
    for sc in profile.get("secondary_characters", []):
        name = sc.get("name", "").strip()
        gender = sc.get("gender", "").lower()
        if name and gender in ("male", "female"):
            secondary_characters[name.lower()] = gender

    print("\n" + "═" * 60)
    print(f"  STORY : {story_name}")
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

    # Create log file for this session
    import datetime
    LOGS_DIR.mkdir(exist_ok=True)
    _ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _safe_story = re.sub(r"[^\w\-]", "_", story_name)
    _log_path = LOGS_DIR / f"{_ts}_{_safe_story}.md"
    _log_path.write_text(
        f"# Session log\n\n"
        f"- **Story:** {story_name}\n"
        f"- **Model:** {model}\n"
        f"- **Player:** {profile['player_name']} ({profile.get('player_gender', '?')})\n"
        f"- **NPC:** {profile['other_name']} ({profile.get('other_gender', '?')})\n"
        f"- **Lang:** {story_lang}\n"
        f"- **Started:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"---\n\n"
        f"### [SYSTEM PROMPT]\n\n{system_prompt}\n\n---\n",
        encoding="utf-8",
    )

    session["model"]          = model
    session["profile"]        = profile
    session["messages"]       = [{"role": "system", "content": system_prompt}]
    session["log_file"]       = _log_path
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
    session["beat_next_turn"] = random.randint(5, 8)
    session["lang"]           = story_lang
    session["story_name"]     = story_name
    session["story_context"]  = story_context
    session["affinity"]       = affinity.initial_affinity(profile)
    session["affinity_history"] = []
    session["suggest_count"]  = 0
    session["suggest_next_env"] = random.randint(2, 5)

    # Journal entries are generated dynamically as beats fire, based on the
    # actual conversation. No pre-generation needed.
    session["journal"] = []
    session["journal_unlocked"] = set()

    return {
        "name":         profile["other_name"],
        "other_gender": profile.get("other_gender", "female"),
        "player_name":  profile["player_name"],
        "setting":      _str(profile.get("setting", "")),
        "stage":        _str(profile.get("relationship_stage", "")),
        "summary":      _str(profile.get("story_summary", "")),
        "personality":  profile.get("player_personality", []),
        "affinity":     session["affinity"],
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
    url_error = None
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
        url_error = str(e)
    finally:
        assembled = "".join(full)
        if assembled:
            session["messages"].append({"role": "assistant", "content": assembled})
            # Skip logging the internal story-open trigger message as a user turn
            user_turns = [m for m in session.get("messages", []) if m["role"] == "user"
                          and not m["content"].startswith("[Begin the story")]
            if user_turns:
                _append_log("user", user_turns[-1]["content"])
            _append_log("assistant", assembled)
            # Fire-and-forget affinity scoring on a daemon thread. Snapshots avoid races.
            if session.get("profile") and session.get("affinity") is not None:
                session["affinity_turn_counter"] = session["affinity"]["turn"] + 1
                threading.Thread(
                    target=_score_turn_async,
                    args=(
                        dict(session["profile"]),
                        list(session["messages"][-6:]),
                        dict(session["affinity"]),
                        session["model"],
                        session.get("lang", "en"),
                    ),
                    daemon=True,
                ).start()

    if url_error:
        yield f"data: {json.dumps({'error': url_error})}\n\n"
        return

    yield f"data: {json.dumps({'done': True, 'affinity': session.get('affinity')})}\n\n"


@app.route("/open", methods=["POST"])
def open_story():
    _pname = session.get("player_name", "the player")
    _oname = session.get("other_name", "the NPC")
    trigger_msg = {"role": "user", "content": (
        f"[Begin the story. 3–4 paragraphs maximum. "
        f"Remember: the player is {_pname} — always 'you'. {_oname} is the character you portray. "
        f"Open in medias res — the scene is already in motion. "
        f"Weave in one or two vivid physical details about {_oname} naturally as the scene unfolds; do not front-load a description block. "
        f"End on a single unresolved beat: {_oname} says one thing or does one thing that demands a response. Stop there.]"
    )}
    session["messages"].append(trigger_msg)
    return Response(
        stream_with_context(_ollama_stream(session["messages"])),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/affinity", methods=["GET"])
def get_affinity():
    return {
        "affinity": session.get("affinity"),
        "history": session.get("affinity_history", []),
    }


@app.route("/recap", methods=["POST"])
def recap():
    profile = session.get("profile")
    history = session.get("affinity_history", [])
    current = session.get("affinity")
    if not profile or not current:
        return {"error": "no active session"}, 400
    if not history:
        return {"final": current, "history": [], "narrative": ""}

    lang = session.get("lang", "en")
    narrative = affinity.generate_recap(history, profile, session["model"], lang)
    affinity.save_final(
        session.get("story_name") or "",
        lang,
        current,
        history,
        narrative,
        current.get("turn", len(history)),
    )
    return {
        "final": {k: current.get(k, 0) for k in ("trust", "intimacy", "tension")},
        "history": history,
        "narrative": narrative,
    }


@app.route("/save", methods=["POST"])
def save_session():
    """Save current session to disk, generating a comprehensive summary."""
    profile = session.get("profile")
    if not profile or not session.get("messages"):
        return {"error": "no active session"}, 400

    story_name = session.get("story_name")
    if not story_name:
        return {"error": "story not identified"}, 400

    import datetime
    lang = session.get("lang", "en")
    try:
        summary = _generate_save_summary(lang)
    except Exception as e:
        summary = "(Summary generation failed)"

    # Get last N verbatim messages (excluding system prompt)
    recent = [m for m in session.get("messages", []) if m["role"] != "system"][-6:]

    save_id = f"{story_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dict = {
        "id": save_id,
        "story": story_name,
        "lang": lang,
        "model": session.get("model", ""),
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "player_name": session.get("player_name", ""),
        "other_name": session.get("other_name", ""),
        "other_gender": profile.get("other_gender", ""),
        "player_gender": profile.get("player_gender", ""),
        "beat_index": session.get("beat_index", 0),
        "affinity": session.get("affinity"),
        "affinity_history": session.get("affinity_history", []),
        "journal": session.get("journal", []),
        "journal_unlocked": sorted(list(session.get("journal_unlocked", set()))),
        "summary": summary,
        "recent_messages": recent,
    }

    SAVES_DIR.mkdir(parents=True, exist_ok=True)
    save_path = SAVES_DIR / f"{save_id}.json"
    try:
        save_path.write_text(json.dumps(save_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as e:
        return {"error": f"Failed to save: {e}"}, 500

    return {"save_id": save_id, "saved_at": save_dict["saved_at"]}


@app.route("/saves", methods=["GET"])
def list_saves():
    """List all saved sessions with metadata."""
    if not SAVES_DIR.exists():
        return {"saves": []}

    saves = []
    try:
        for f in sorted(SAVES_DIR.glob("*.json"), reverse=True):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                saves.append({
                    "id": data.get("id", ""),
                    "story": data.get("story", ""),
                    "player_name": data.get("player_name", ""),
                    "other_name": data.get("other_name", ""),
                    "saved_at": data.get("saved_at", ""),
                })
            except (json.JSONDecodeError, OSError):
                pass
    except OSError:
        pass

    return {"saves": saves}


@app.route("/saves/<save_id>", methods=["DELETE"])
def delete_save(save_id):
    """Delete a saved session."""
    save_id = save_id.strip()
    if not save_id:
        return {"error": "no save_id provided"}, 400

    try:
        base = SAVES_DIR.resolve()
        save_path = (base / f"{save_id}.json").resolve()
        save_path.relative_to(base)
    except (ValueError, RuntimeError):
        return {"error": "invalid save_id"}, 400

    if not save_path.exists():
        return {"error": "save not found"}, 404

    try:
        save_path.unlink()
        return {"deleted": True}
    except OSError as e:
        return {"error": f"failed to delete: {e}"}, 500


@app.route("/resume", methods=["POST"])
def resume_session():
    """Resume a saved session."""
    import datetime
    data = request.json or {}
    save_id = data.get("save_id", "").strip()
    if not save_id:
        return {"error": "no save_id provided"}, 400

    try:
        base = SAVES_DIR.resolve()
        save_path = (base / f"{save_id}.json").resolve()
        save_path.relative_to(base)
    except (ValueError, RuntimeError):
        return {"error": "invalid save_id"}, 400

    if not save_path.exists():
        return {"error": "save not found"}, 404

    try:
        save_data = json.loads(save_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"error": "failed to load save"}, 500

    story_name = save_data.get("story", "")
    story_lang = save_data.get("lang", "en")
    if story_lang not in ("en", "ja"):
        story_lang = "en"

    # Load profile from disk
    summaries_dir = _summaries_dir(story_lang)
    profile_file = summaries_dir / f"{story_name}_profile.json"
    if not profile_file.exists():
        return {"error": "story profile not found"}, 404

    try:
        profile = json.loads(profile_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"error": "failed to load profile"}, 500

    # Re-assign player/NPC roles from the saved player_name so build_system_prompt works.
    saved_player = save_data.get("player_name", "")
    if saved_player and saved_player == profile.get("char_b_name", ""):
        _assign_roles(profile, player_key="b")
    else:
        _assign_roles(profile, player_key="a")

    # Journal entries are stored in the save file directly (dynamically generated).
    journal = save_data.get("journal", [])

    # Reconstruct system prompt
    story_file = _stories_dir(story_lang) / f"{story_name}.txt"
    try:
        story_context = story_file.read_text(encoding="utf-8") if story_file.exists() else ""
    except OSError:
        story_context = ""

    system_prompt = build_system_prompt(profile, story_context, story_lang,
                                        affinity=session.get("affinity"))

    # Restore session state
    session["profile"] = profile
    session["story_name"] = story_name
    session["lang"] = story_lang
    session["model"] = save_data.get("model", "")
    session["player_name"] = save_data.get("player_name", "")
    session["other_name"] = save_data.get("other_name", "")
    session["other_pronoun"] = profile.get("other_pronoun", "she")
    session["player_pronoun"] = profile.get("player_pronoun", "you")
    # Build secondary character registry: {name_lower: gender}
    secondary_characters = {}
    for sc in profile.get("secondary_characters", []):
        if isinstance(sc, dict):
            name = sc.get("name", "").strip()
            gender = sc.get("gender", "").lower()
            if name and gender in ("male", "female"):
                secondary_characters[name.lower()] = gender
    session["secondary_characters"] = secondary_characters
    session["story_beats"] = profile.get("story_beats", [])
    session["beat_index"] = save_data.get("beat_index", 0)
    session["beat_next_turn"] = random.randint(5, 8)
    # Restore saved affinity if present, otherwise compute baseline
    saved_affinity = save_data.get("affinity")
    session["affinity"] = saved_affinity if isinstance(saved_affinity, dict) else affinity.initial_affinity(profile)
    session["affinity_history"] = []
    session["suggest_count"]  = 0
    session["suggest_next_env"] = random.randint(2, 5)
    session["journal"] = journal if isinstance(journal, list) else []
    session["journal_unlocked"] = set(save_data.get("journal_unlocked", []))

    # Assign voices based on gender from save
    player_gender = save_data.get("player_gender", "male").lower()
    other_gender = save_data.get("other_gender", "female").lower()
    session["player_voice"], session["player_lang"] = (
        (PRIMARY_MALE_CHARACTER_VOICE, PRIMARY_MALE_CHARACTER_LANG)
        if player_gender == "male"
        else (PRIMARY_FEMALE_CHARACTER_VOICE, PRIMARY_FEMALE_CHARACTER_LANG)
    )
    session["other_voice"], session["other_lang"] = (
        (PRIMARY_FEMALE_CHARACTER_VOICE, PRIMARY_FEMALE_CHARACTER_LANG)
        if other_gender == "female"
        else (PRIMARY_MALE_CHARACTER_VOICE, PRIMARY_MALE_CHARACTER_LANG)
    )
    # If genders match, use secondary voices for player
    if player_gender == other_gender:
        if player_gender == "male":
            session["player_voice"], session["player_lang"] = SECONDARY_MALE_CHARACTER_VOICE, SECONDARY_MALE_CHARACTER_LANG
        else:
            session["player_voice"], session["player_lang"] = SECONDARY_FEMALE_CHARACTER_VOICE, SECONDARY_FEMALE_CHARACTER_LANG

    # Reconstruct messages: system prompt + summary only (no recent verbatim history)
    messages = [{"role": "system", "content": system_prompt}]
    if save_data.get("summary"):
        messages.append({
            "role": "assistant",
            "content": f"[STORY SO FAR]\n\n{save_data['summary']}"
        })
    session["messages"] = messages

    # Create a log file for this resumed session
    LOGS_DIR.mkdir(exist_ok=True)
    _ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _safe_story = re.sub(r"[^\w\-]", "_", story_name)
    _log_path = LOGS_DIR / f"{_ts}_{_safe_story}_resumed.md"
    _log_path.write_text(
        f"# Resumed Session Log\n\n"
        f"- **Save ID:** {save_id}\n"
        f"- **Original Story:** {story_name}\n"
        f"- **Model:** {session.get('model', '')}\n"
        f"- **Resumed:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"---\n\n",
        encoding="utf-8",
    )
    session["log_file"] = _log_path
    session["story_context"]  = story_context

    return {
        "name": profile.get("other_name", ""),
        "player_name": save_data.get("player_name", ""),
        "setting": _str(profile.get("setting", "")),
        "stage": _str(profile.get("relationship_stage", "")),
        "summary": save_data.get("summary", ""),
        "personality": profile.get("player_personality", []),
        "affinity": session.get("affinity"),
        "resumed": True,
        "beat_index": session.get("beat_index"),
    }


@app.route("/suggest", methods=["POST"])
def suggest():
    """Return 2-3 short action options the player could take, based on current context."""
    if not session["messages"]:
        return {"suggestions": []}

    # Use a neutral system message so the model isn't locked into story-narrator mode,
    # which causes it to ignore JSON instructions after a few turns.
    is_ja = session.get("lang", "en") == "ja"

    # Decide whether this turn gets an environment suggestion (once every 2–5 rounds).
    session["suggest_count"] = session.get("suggest_count", 0) + 1
    count = session["suggest_count"]
    next_env = session.get("suggest_next_env", 3)
    include_env = (count >= next_env)
    if include_env:
        session["suggest_next_env"] = count + random.randint(2, 5)

    # Pull the story setting for concrete environmental context.
    profile = session.get("profile") or {}
    setting_raw = profile.get("setting", "")
    if isinstance(setting_raw, dict):
        setting_str = ", ".join(str(v) for v in setting_raw.values() if v)
    elif isinstance(setting_raw, list):
        setting_str = "; ".join(str(v) for v in setting_raw if v)
    else:
        setting_str = str(setting_raw).strip()

    if is_ja:
        n = 3 if include_env else 2
        suggest_system = (
            "あなたは創作アシスタントです。プレイヤーの行動候補を提案するのが唯一の役割です。"
            f"必ず有効なJSON配列（{n}つの日本語文字列）のみを返してください。"
            "前置きやマークダウン、中国語、英語は一切混ぜないこと。"
        )
        env_setting_hint_ja = f"（舞台：{setting_str}）" if setting_str else ""
        env_item = (
            f"[2] 環境・場所に関わる行動{env_setting_hint_ja}（相手との関係ではなく）。"
            "現在のシーンの場所から自然につながる提案にすること（現在いる場所と無関係な観光地や有名スポットを突然出さない）。"
            "場所の移動を提案する場合は「〜に行きたい」「〜を見たい」の形にすること。（1〜2文）\n"
        ) if include_env else ""
        suggest_prompt = (
            f"JSON配列で{n}つの行動候補を返してください。順番どおりに:\n"
            "[0] 相手との関係に関する行動・台詞（大胆・直接的）。場所や環境の話題は一切含めないこと。（1〜2文）\n"
            "[1] 相手との関係に関する行動・台詞（優しく温かい）。場所や環境の話題は一切含めないこと。（1〜2文）\n"
            f"{env_item}"
            "重要：[0]と[1]は必ずキャラクター同士の関係についての内容にすること。場所・環境の話題は[2]のみ。\n"
            "プレイヤー（「私」または主語省略）の視点で書くこと。必ず日本語のみ。"
            "JSON配列のみを返すこと。コードフェンスや説明文は一切含めない。"
        )
    else:
        n = 3 if include_env else 2
        suggest_system = (
            "You are a creative writing assistant. Your only job is to suggest player actions. "
            f"You must respond with ONLY a valid JSON array of exactly {n} strings. No prose, no markdown."
        )
        env_setting_hint = f", grounded in this setting: {setting_str}" if setting_str else ""
        env_item = (
            "[2] One action about the environment or a nearby place — NOT about the relationship. "
            "It must follow naturally from where the characters currently are; do NOT invent unrelated famous landmarks or distant locations. "
            f"If suggesting going somewhere, phrase it as 'I want to go to...' or 'I want to see...' (1-2 sentences, first person{env_setting_hint})\n"
        ) if include_env else ""
        suggest_prompt = (
            f"Return a JSON array of exactly {n} player actions, in this order:\n"
            "[0] A bold or direct action or line of dialogue toward the other character — about the relationship only, NOT about location or environment (1-2 sentences, first person)\n"
            "[1] A tender or warm action or line of dialogue toward the other character — about the relationship only, NOT about location or environment (1-2 sentences, first person)\n"
            f"{env_item}"
            f"IMPORTANT: items [0] and [1] must be about the relationship between characters, not the setting.\n"
            f"Return ONLY a JSON array of {n} strings. No prose, no markdown fences, no commentary."
        )
    recent = [m for m in session["messages"] if m["role"] != "system"]
    context = recent[-6:]
    messages = [{"role": "system", "content": suggest_system}] + context + [{"role": "user", "content": suggest_prompt}]

    payload = json.dumps({
        "model": session["model"],
        "messages": messages,
        "stream": False,
        "options": {"num_ctx": 8192, "num_predict": 300},
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

        suggestions = []
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                suggestions = [str(s) for s in parsed[:3]]
            elif isinstance(parsed, dict):
                for val in parsed.values():
                    if isinstance(val, list):
                        suggestions = [str(s) for s in val[:3]]
                        break
        except json.JSONDecodeError:
            # Malformed JSON — extract quoted strings directly as a fallback
            suggestions = re.findall(r'"((?:[^"\\]|\\.)+)"', content)[:3]

        # Drop anything that's too short to be a real action (stray punctuation,
        # JSON key names, single words the regex picked up by mistake).
        suggestions = [s for s in suggestions if len(s.strip()) >= 10]

    except Exception as e:
        print(f"Suggest error: {e!r}")
        suggestions = []

    return {"suggestions": suggestions}


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_input = data.get("message", "").strip()

    if not user_input:
        return {"error": "Empty message"}, 400
    if not session["messages"]:
        return {"error": "No active story. Call /start first."}, 400

    # Store the clean message in history
    session["messages"].append({"role": "user", "content": user_input})

    # Compress history every N user turns to prevent context overflow
    user_turn_count = sum(1 for m in session["messages"] if m["role"] == "user")
    if user_turn_count > 0 and user_turn_count % _COMPRESS_EVERY == 0:
        _compress_history()

    # Every 3 user turns inject a beat nudge into what the LLM sees,
    # but NOT into the stored session history.
    beats = session.get("story_beats", [])
    messages_for_llm = session["messages"]
    beat_just_fired = None
    if beats:
        turn_count = sum(1 for m in session["messages"] if m["role"] == "user")
        beat_idx = session["beat_index"]
        if turn_count >= session.get("beat_next_turn", 5) and beat_idx < len(beats):
            beat = beats[beat_idx]
            beat_just_fired = beat_idx
            session["beat_index"] += 1
            session["beat_next_turn"] = turn_count + random.randint(5, 8)
            nudge = (
                f"\n\n[Scene director: You have not yet brought this beat into the action — "
                f"introduce it naturally within this exchange or the next. "
                f"Engineer the situation; don't announce it: {beat}]"
            )
            messages_for_llm = session["messages"][:-1] + [
                {"role": "user", "content": user_input + nudge}
            ]

    npc_name = session.get("other_name", "")
    fired_beat_idx = beat_just_fired
    fired_profile  = dict(session.get("profile", {})) if fired_beat_idx is not None else None
    fired_messages = list(session["messages"]) if fired_beat_idx is not None else None
    fired_model    = session.get("model", "")
    fired_lang     = session.get("lang", "en")

    def wrapped():
        yield from _ollama_stream(messages_for_llm)
        if fired_beat_idx is not None and fired_profile is not None:
            entry = generate_journal_entry(
                fired_profile, fired_messages, fired_beat_idx,
                fired_model, fired_lang,
            )
            if entry:
                session["journal"].append(entry)
                session["journal_unlocked"].add(entry["id"])
                yield f"data: {json.dumps({'journal_unlock': {'id': entry['id'], 'title': entry['title'], 'npc_name': npc_name}})}\n\n"

    return Response(
        stream_with_context(wrapped()),
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


@app.route("/journal", methods=["GET"])
def journal_list():
    """Return the journal entries for the current session.

    Unlocked entries include full body; locked entries return only title placeholder.
    """
    journal = session.get("journal") or []
    unlocked = session.get("journal_unlocked") or set()
    items = []
    for e in journal:
        if e.get("id") in unlocked:
            items.append({
                "id": e["id"],
                "title": e["title"],
                "body": e["body"],
                "unlocked": True,
            })
        else:
            items.append({
                "id": e["id"],
                "title": "???",
                "unlocked": False,
            })
    return {"entries": items, "unlocked_count": len(unlocked), "total": len(journal)}


@app.route("/journal/tts", methods=["POST"])
def journal_tts():
    """Render a journal entry with the NPC's voice."""
    data = request.json or {}
    entry_id = (data.get("id") or "").strip()
    if not entry_id:
        return {"error": "No entry id"}, 400
    journal = session.get("journal") or []
    unlocked = session.get("journal_unlocked") or set()
    entry = next((e for e in journal if e.get("id") == entry_id), None)
    if not entry or entry_id not in unlocked:
        return {"error": "Entry not unlocked"}, 404

    try:
        import numpy as np
        kokoro = get_kokoro()
        story_lang = session.get("lang", "en")
        voice = session.get("other_voice")
        lang = session.get("other_lang")
        if not voice:
            return {"error": "No voice configured"}, 500

        SR = 24000
        silence = np.zeros(int(SR * 0.18), dtype=np.float32)
        parts = []
        text = _scrub_for_tts(entry.get("body", "").strip())
        for chunk in _chunk_for_tts(text, story_lang):
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
    port = int(os.environ.get("PORT", 5000))
    print("Story Roleplay Server")
    print(f"Open http://localhost:{port} in your browser")
    # Pre-warm kokoro in background so first TTS call is fast
    threading.Thread(target=get_kokoro, daemon=True).start()
    app.run(debug=False, port=port, threaded=True)
