# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the app

```bash
source .venv/bin/activate
python server.py          # web UI at http://localhost:5000
python story.py <story_file.txt> [model]   # CLI mode
```

Ollama must be running separately (`ollama serve`). The default model is `hf.co/mradermacher/mistralai-Mistral-Nemo-Instruct-2407-extensive-BP-abliteration-12B-GGUF:Q4_K_M`.

## Dependencies

Managed via `.venv`. Key packages: `flask`, `kokoro_onnx`, `numpy`, `onnxruntime`. Install with:

```bash
python -m venv .venv && source .venv/bin/activate
pip install flask kokoro-onnx numpy onnxruntime
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

**At game start:** When you pick a story, the app loads from the cached `.md` files. If summaries don't exist yet, it will analyze the `.txt` file fresh and save summaries automatically.

## Architecture

The app has two entry points that share `story.py`:

**Web UI** (`server.py` + `index.html` + `kokoro_test.html`)
- Flask server, single-user, `threaded=True`
- All conversation state lives in a module-level `session` dict — no database, no Flask sessions
- Request flow for a new story: `POST /start` → `POST /open` (SSE stream) → repeated `POST /chat` (SSE stream)
- `POST /tts` is called by the frontend after each AI response completes; it parses the response into segments and renders each with the appropriate Kokoro voice
- `GET /kokoro_test` + `POST /tts_preview` serve the voice sampler page

**CLI** (`story.py`)
- `run_session()` drives an interactive terminal loop using the same `analyze_story` / `build_system_prompt` functions

## Key design details

**Story analysis (`story.py:analyze_story`)**: Two-pass Ollama call. If the story text exceeds 8000 chars, a summarization call runs first. The second call extracts a structured JSON profile (character names, personality, appearance, story beats, etc.). The JSON may be wrapped in markdown fences — stripping handles both opening and closing fences.

**System prompt (`story.py:build_system_prompt`)**: Assembled from the profile dict. Contains hard rules for the model: second-person for the player, third-person for NPCs, 3–5 paragraph response limit, one emotional beat per turn. The "story beats" array is the model's secret roadmap through the narrative arc.

**TTS segmentation (`server.py:_parse_segments` / `_detect_speaker`)**: Splits AI responses into `(text, speaker)` tuples — `narrator`, `other` (NPC), `player`, `secondary_male`, `secondary_female`. Speaker is detected by position-priority: whichever attribution tag (e.g. "she whispered", "you said") appears earliest in the surrounding narr text wins. Each speaker maps to a distinct Kokoro voice defined at the top of `server.py`.

**Voice constants** (top of `server.py`): `NARRATOR_VOICE`, `PRIMARY_MALE/FEMALE_CHARACTER_VOICE`, `SECONDARY_MALE/FEMALE_CHARACTER_VOICE`. When player and NPC share a gender, the player is assigned the secondary voice to keep them distinguishable.

**Player/NPC role assignment**: The LLM assigns player/other roles during analysis. If the user selects a specific gender on the setup screen, `/start` swaps `player_name`/`player_gender` with `other_name`/`other_gender` in the profile before any further processing. The LLM's appearance/personality fields always describe `other_name`, so they remain correct after a swap.

**SSE streaming**: `_ollama_stream()` is a shared generator used by both `/open` and `/chat`. It appends the fully-assembled assistant reply to `session["messages"]` after the stream completes. `/open` appends its trigger message to `session["messages"]` before calling the generator, so the history always has proper alternating user/assistant turns.
