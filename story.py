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



def ollama_chat(model: str, messages: list, stream: bool = True) -> str:
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": stream,
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    full_response = []
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
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


def analyze_story(story_text: str, model: str) -> dict:
    """Extract character profiles and story context from the text."""
    print("\n[Analyzing story and building character profile...]\n")

    # If story is long, summarize first to fit context
    if len(story_text) > STORY_CHAR_LIMIT:
        print("[Story is long — summarizing first...]\n")
        summary_messages = [
            {
                "role": "user",
                "content": (
                    "Summarize the following story in detail. "
                    "Focus on: the two main characters' personalities, mannerisms, speech patterns, "
                    "desires, how they express affection, key moments between them, "
                    "and the emotional arc of the relationship. Be thorough.\n\n"
                    f"STORY:\n{story_text[:20000]}"
                ),
            }
        ]
        story_context = ollama_chat(model, summary_messages, stream=False)
    else:
        story_context = story_text

    # Now extract structured character profile
    analysis_messages = [
        {
            "role": "user",
            "content": (
                "Based on this story (or summary), identify the two main characters. "
                "The player character is the protagonist or POV character. "
                "The other character is their primary counterpart (romantic interest, etc). "
                "Return ONLY a JSON object with these fields:\n"
                "- player_name: name of the character the user will play (use a real name, never 'You' or a pronoun)\n"
                "- player_gender: 'male' or 'female'\n"
                "- other_name: name of the primary character the AI will portray (use a real name, never 'Her' or a pronoun)\n"
                "- other_gender: 'male' or 'female'\n"
                "- personality: array of personality traits for other_name\n"
                "- speech_style: how other_name talks, vocabulary, tone\n"
                "- affection_style: how other_name shows love/interest\n"
                "- key_behaviors: specific things other_name does or says in the story\n"
                "- desires: what other_name wants, fears, and needs emotionally\n"
                "- dealbreakers: things that would push other_name away or make them withdraw\n"
                "- setting: where/when the story takes place\n"
                "- relationship_stage: where they are at the start (strangers, friends, dating, etc)\n"
                "- story_summary: 3-4 sentence summary of the story arc\n"
                "- story_beats: array of 5-8 key emotional situations or turning points from the story, "
                "described as situations to encounter (not outcomes) — e.g. 'they are alone together for the first time and tension builds' not 'they kiss'\n"
                "- appearance: detailed physical description of other_name — height, build, face shape, skin tone, age\n"
                "- hair: other_name's hair color, length, texture, and typical style\n"
                "- eyes: other_name's eye color, shape, and any notable quality\n"
                "- scent: other_name's characteristic scent or perfume if mentioned, otherwise infer something fitting\n"
                "- clothing_style: how other_name typically dresses, fabrics, colors, formality\n\n"
                f"STORY/SUMMARY:\n{story_context}"
            ),
        }
    ]

    raw = ollama_chat(model, analysis_messages, stream=False)

    # Parse JSON, tolerating markdown code fences (```json ... ```)
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            # Strip opening fence line (```json or ```)
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            # Strip closing fence
            if cleaned.rstrip().endswith("```"):
                cleaned = cleaned.rstrip()[:-3]
        profile = json.loads(cleaned.strip())
    except json.JSONDecodeError:
        profile = {
            "player_name": "Alex",
            "player_gender": "male",
            "other_name": "Sarah",
            "other_gender": "female",
            "personality": ["warm", "romantic", "expressive"],
            "speech_style": "warm and intimate",
            "affection_style": "tender and direct",
            "key_behaviors": [],
            "setting": "contemporary",
            "relationship_stage": "budding romance",
            "story_summary": raw[:500],
        }

    return profile, story_context


def build_system_prompt(profile: dict, story_context: str) -> str:
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

═══ STORY BEATS (your secret roadmap) ═══
These are the emotional situations the story should move through — in rough order, but not rigidly. Guide the story toward each one organically. The user's choices can change HOW each beat unfolds, delay or accelerate beats, or produce entirely different outcomes. The beats are situations to arrive at, not scripts to follow.
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
