# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the app

```bash
source .venv/bin/activate
python server.py          # web UI at http://localhost:5000
python story.py <story_file.txt> [model]   # CLI mode
```

Ollama must be running separately (`ollama serve`). The default model is `nemo` (Mistral Nemo 12B, renamed via `ollama cp`).

## Dependencies

Managed via `.venv`. Key packages: `flask`, `kokoro_onnx`, `misaki`, `numpy`, `onnxruntime`. Install with:

```bash
python -m venv .venv && source .venv/bin/activate
pip install flask kokoro-onnx misaki numpy onnxruntime
```

Kokoro model files (`kokoro-v1.0.onnx`, `voices-v1.0.bin`) live in `kokoro_models/` (gitignored). Story `.txt` files live in `stories/` (gitignored).

## Story Indexing & Summaries

Stories are analyzed once and cached as markdown summaries to avoid re-running the expensive LLM analysis on every game start.

**Workflow:**
1. Open http://localhost:5000/index_stories in your browser
2. Select a story from the dropdown and click "Analyze"
3. The app calls Ollama to extract character profiles and story context
4. Results are saved as three files in `story_summaries/`:
   - `{story}_character.md` — structured character summary (appearance, personality, behaviors, etc.)
   - `{story}_story.md` — story context and summary text
   - `{story}_profile.json` — raw profile JSON for reference

**Re-indexing:** Click "Analyze" again to re-analyze a story and update its summaries.

**Language isolation:** Visit with `?lang=ja` to switch to Japanese mode. JA mode reads source stories from `stories_ja/` and writes analyses to `story_summaries_jp/`. EN mode reads from `stories/` and writes to `story_summaries/`. The game-page story dropdown filters by summary availability in the matching folder. Pre-existing JA summaries in `story_summaries/` must be moved by hand to `story_summaries_jp/`.

**At game start:** When you pick a story, the app loads from the cached `.md` files. If summaries don't exist yet, it will analyze the `.txt` file fresh and save summaries automatically.

## Architecture

The app has two entry points that share `story.py`:

**Web UI** (`server.py` + `index.html` + `index_stories.html` + `kokoro_test.html`)
- Flask server, single-user, `threaded=True`
- All conversation state lives in a module-level `session` dict — no database, no Flask sessions
- Request flow for a new story: `POST /start` → `POST /open` (SSE stream) → repeated `POST /chat` (SSE stream)
- `POST /tts` is called by the frontend after each AI response completes; it parses the response into segments and renders each with the appropriate Kokoro voice
- `GET /kokoro_test` + `POST /tts_preview` serve the voice sampler page

**CLI** (`story.py`)
- `run_session()` drives an interactive terminal loop using the same `analyze_story` / `build_system_prompt` functions

**Supporting modules:**
- `tts.py` — voice constants, Kokoro lazy-init, TTS segment parsers, WAV encoding, chunking
- `summaries.py` — profile→markdown conversion (`_profile_to_markdown`, `_profile_to_story_markdown`)
- `affinity.py` — trust/intimacy/tension scoring, delta application, persistence
- `translate_story.py` — CLI tool to translate EN stories to JA via Ollama; results go to `stories_ja/`

## Key design details

**Story analysis (`story.py:analyze_story` + `enrich_character_profile`)**: Three-pass Ollama call. If the story text exceeds 8000 chars, a summarization call runs first. The second call extracts a structured JSON profile using `player_/other_` naming (what local LLMs reliably follow). After parsing, `_normalize_to_char_keys()` renames these to the symmetric `char_a/char_b` format so neither character is pre-labelled "the player" in storage. A third enrichment pass (`enrich_character_profile`) fills physical descriptions, loves/hates, and secondary character details (Pass A for both leads, Pass B for secondary characters).

**Profile naming convention**: Analysis stores characters as `char_a_*` / `char_b_*`. At game start, `_assign_roles(profile, player_key)` writes the `player_*` / `other_*` keys that all downstream code (system prompt, TTS, affinity) reads. The NPC's appearance/personality fields are always keyed under `other_*` (not `char_a_*`) after role assignment.

**System prompt (`story.py:build_system_prompt`)**: Assembled from the profile dict. Contains hard rules for the model: second-person for the player, third-person for NPCs, 3–5 paragraph response limit, one emotional beat per turn. The "story beats" array is the model's secret roadmap through the narrative arc. Dispatches to `build_system_prompt_ja` for JA sessions.

**Beat injection**: In `/chat`, every `beat_next_turn` turns (randomized 5–8 between beats) a story beat is appended to the *ephemeral* user message sent to the LLM — not stored in history. After the stream completes, if a beat fired, `generate_journal_entry` writes an NPC-POV journal entry from the actual conversation.

**Context compression** (`server.py:_compress_history`): Every 10 user turns, older exchanges are replaced with a single summarized "STORY SO FAR" assistant injection. The 3 most recent user/assistant pairs are always kept verbatim. This prevents context overflow without losing narrative continuity.

**Affinity system** (`affinity.py`): After each assistant response, a daemon thread calls `affinity.score_turn()` with a snapshot of the last 6 messages. The scorer uses rule-based overrides for obvious violence/cruelty, then falls back to an LLM call with EN/JA prompt templates. Deltas are clamped to ±2 per turn and to [0, 10] absolute. The updated state is injected into `messages[0]` (the system prompt) so the story LLM sees the current relationship reality every turn.

**TTS segmentation (`tts.py:_parse_segments` / `_parse_segments_ja`)**: Splits AI responses into `(text, speaker)` tuples — `narrator`, `other` (NPC), `player`, `secondary_male`, `secondary_female`. English speaker detection is position-priority: whichever attribution tag appears earliest in surrounding narration wins (post-attribution "she said" beats pre-attribution "she whispered,"). Japanese uses `と/って` quotative particles as the primary signal. JA text is phonemized with `misaki` before passing to Kokoro (`is_phonemes=True`) because kokoro-onnx's built-in espeak-ng phonemizer mangles Japanese.

**Voice constants** (top of `tts.py`): `NARRATOR_VOICE`, `PRIMARY_MALE/FEMALE_CHARACTER_VOICE`, `SECONDARY_MALE/FEMALE_CHARACTER_VOICE`, and a parallel `JA_*` set. When player and NPC share a gender, the player is assigned the secondary voice to keep them distinguishable.

**Player/NPC role assignment**: Analysis stores both characters symmetrically as `char_a/char_b`. At game start, `/start` picks which char becomes the player based on the user's gender preference (`player_gender` param), then calls `_assign_roles(profile, player_key)` to write `player_*/other_*` fields before any further processing.

**SSE streaming**: `_ollama_stream()` is a shared generator used by both `/open` and `/chat`. It appends the fully-assembled assistant reply to `session["messages"]` after the stream completes. `/open` appends its trigger message to `session["messages"]` before calling the generator, so the history always has proper alternating user/assistant turns.

**Save/resume** (`/save`, `/saves`, `/resume`): Sessions are saved to `saves/` as JSON files. Saves include a full LLM-generated narrative summary plus the last 6 verbatim message pairs; on resume, the system prompt is reconstructed from the cached profile and the summary is injected as an assistant message to restore context without replaying every turn.

**Translation** (`translate_story.py`): Translates EN `.txt` files to JA via Ollama in paragraph-sized chunks (~1500 chars). Extracts a name→kanji glossary first to keep names consistent. The web UI streams translation progress via `POST /translate` (SSE). Default model: `qwen-ja`.
