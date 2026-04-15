"""Character affinity scoring and persistence.

Tracks three dimensions (trust, intimacy, tension) between the player and the
NPC, updated after each NPC response via a non-streaming Ollama call. All
public functions fail soft — on any error they return None / "" / {} so the
main game flow is never blocked by scoring.
"""

import json
import time
import datetime
import urllib.request
import urllib.error
from pathlib import Path

from story import OLLAMA_URL, NUM_CTX, _parse_json

SUMMARIES_DIR = Path(__file__).parent / "story_summaries"
SUMMARIES_DIR_JA = Path(__file__).parent / "story_summaries_jp"

DEFAULT_AFFINITY = {"trust": 5, "intimacy": 2, "tension": 3}
CLAMP_MIN = 0
CLAMP_MAX = 10
MAX_DELTA = 2
HISTORY_CAP = 200
DECAY_EVERY = 10
DECAY_TENSION_ABOVE = 6


# ── Prompt templates ──────────────────────────────────────────────────────

SCORER_SYS_EN = """You are a relationship analyst. Given an NPC profile and the last few message exchanges, output a JSON object scoring how the most recent NPC response shifted three dimensions between the player and the NPC. Integer deltas only, each in [-2, +2].

Dimensions:
  trust     — does the NPC feel safer with the player / more betrayed?
  intimacy  — emotional or physical closeness gained / lost?
  tension   — unresolved friction, stakes, or conflict rising / falling.

Bias toward 0. Only move a dimension if the last exchange CLEARLY earned it. Most turns should score {"trust":0,"intimacy":0,"tension":0}.

NPC: {other_name}
Relationship stage: {relationship_stage}
Desires: {desires}
Dealbreakers: {dealbreakers}

Return ONLY JSON, no prose, no code fences. Schema:
{"trust": <int in [-2,2]>, "intimacy": <int in [-2,2]>, "tension": <int in [-2,2]>, "reason": "<one short sentence>"}
"""

SCORER_SYS_JA = """あなたは恋愛関係の分析者です。NPCのプロフィールと直近の会話を踏まえ、最新のNPC応答がプレイヤーとNPCの関係の3つの次元をどう変化させたかをJSONで出力してください。整数の差分のみ、各[-2, +2]の範囲。

次元:
  trust(信頼)     — NPCはプレイヤーに安心を感じたか、裏切られたと感じたか?
  intimacy(親密さ) — 感情的・身体的な近さが増したか減ったか?
  tension(緊張)   — 未解決の摩擦、緊張、対立が高まったか和らいだか?

基本は0寄りに。直近のやり取りが明確に値を動かす場合のみ変化させる。ほとんどのターンは {"trust":0,"intimacy":0,"tension":0} であるべき。

NPC: {other_name}
関係の段階: {relationship_stage}
望むもの: {desires}
許せないこと: {dealbreakers}

JSONのみ返す(前置き・コードフェンス禁止)。スキーマ:
{"trust": <[-2,2]の整数>, "intimacy": <[-2,2]の整数>, "tension": <[-2,2]の整数>, "reason": "<日本語で一文>"}
"""

RECAP_SYS_EN = """You are a narrator summarizing how a romantic scene played out. Given the final affinity scores and a timeline of how they shifted, write 2-3 sentences (no more) describing the emotional arc of the playthrough. Second person, addressing the player. Concrete and evocative, not clinical."""

RECAP_SYS_JA = """あなたは恋愛シーンの結末を語るナレーターです。最終的な親密度スコアとその推移を踏まえ、2〜3文以内でプレイスルーの感情的な弧を描写してください。プレイヤーに語りかける二人称で。具体的で詩的に、説明調は避ける。"""


# ── Safe Ollama HTTP helper (doesn't sys.exit on failure) ────────────────

def _safe_chat(model: str, messages: list, temperature: float = 0.2, timeout: int = 60) -> str | None:
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"num_ctx": NUM_CTX, "temperature": temperature},
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return data.get("message", {}).get("content", "")
    except (urllib.error.URLError, json.JSONDecodeError, OSError):
        return None


# ── Initial baseline tuned from relationship_stage ────────────────────────

def initial_affinity(profile: dict) -> dict:
    stage = (profile.get("relationship_stage") or "").lower()
    if "stranger" in stage or "見知らぬ" in stage or "初対面" in stage:
        base = {"trust": 3, "intimacy": 1, "tension": 4}
    elif "estranged" in stage or "疎遠" in stage or "別れ" in stage:
        base = {"trust": 3, "intimacy": 5, "tension": 7}
    elif "lover" in stage or "partner" in stage or "恋人" in stage or "夫婦" in stage:
        base = {"trust": 7, "intimacy": 7, "tension": 3}
    elif "friend" in stage or "友" in stage:
        base = {"trust": 6, "intimacy": 4, "tension": 2}
    else:
        base = dict(DEFAULT_AFFINITY)
    return {**base, "turn": 0, "last_reason": "", "updated_at": time.time()}


# ── Scoring ───────────────────────────────────────────────────────────────

def _format_recent(msgs: list, other_name: str) -> str:
    lines = []
    for m in msgs:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content or role == "system":
            continue
        label = "PLAYER" if role == "user" else (other_name.upper() if other_name else "NPC")
        lines.append(f"{label}: {content}")
    return "\n\n".join(lines[-4:])


def score_turn(profile: dict, recent_msgs: list, current: dict, model: str, lang: str = "en") -> dict | None:
    other_name = profile.get("other_name") or "the NPC"
    stage = profile.get("relationship_stage") or ""
    desires = ", ".join(profile.get("desires", []) or []) or "—"
    dealbreakers = ", ".join(profile.get("dealbreakers", []) or []) or "—"

    template = SCORER_SYS_JA if lang == "ja" else SCORER_SYS_EN
    # .format() would trip on literal JSON braces; use explicit replace.
    sys_prompt = (template
                  .replace("{other_name}", other_name)
                  .replace("{relationship_stage}", stage)
                  .replace("{desires}", desires)
                  .replace("{dealbreakers}", dealbreakers))

    user_prompt = _format_recent(recent_msgs, other_name)
    if not user_prompt:
        return None

    raw = _safe_chat(model, [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ])
    if not raw:
        return None
    parsed = _parse_json(raw)
    if not isinstance(parsed, dict):
        return None
    try:
        return {
            "trust": int(parsed.get("trust", 0)),
            "intimacy": int(parsed.get("intimacy", 0)),
            "tension": int(parsed.get("tension", 0)),
            "reason": str(parsed.get("reason", ""))[:240],
        }
    except (TypeError, ValueError):
        return None


# ── Apply delta with clamping + decay ─────────────────────────────────────

def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def apply_delta(current: dict, delta: dict, turn: int) -> tuple[dict, dict]:
    d_trust = _clamp(int(delta.get("trust", 0)), -MAX_DELTA, MAX_DELTA)
    d_intimacy = _clamp(int(delta.get("intimacy", 0)), -MAX_DELTA, MAX_DELTA)
    d_tension = _clamp(int(delta.get("tension", 0)), -MAX_DELTA, MAX_DELTA)

    trust = _clamp(current["trust"] + d_trust, CLAMP_MIN, CLAMP_MAX)
    intimacy = _clamp(current["intimacy"] + d_intimacy, CLAMP_MIN, CLAMP_MAX)
    tension = _clamp(current["tension"] + d_tension, CLAMP_MIN, CLAMP_MAX)

    # Tension decay
    if turn > 0 and turn % DECAY_EVERY == 0 and tension > DECAY_TENSION_ABOVE:
        tension -= 1

    reason = str(delta.get("reason", ""))[:240]
    now = time.time()
    new_current = {
        "trust": trust, "intimacy": intimacy, "tension": tension,
        "turn": turn, "last_reason": reason, "updated_at": now,
    }
    entry = {
        "turn": turn,
        "trust": trust, "intimacy": intimacy, "tension": tension,
        "d_trust": d_trust, "d_intimacy": d_intimacy, "d_tension": d_tension,
        "reason": reason, "ts": now,
    }
    return new_current, entry


# ── Recap ─────────────────────────────────────────────────────────────────

def _recap_user_prompt(history: list, profile: dict, lang: str) -> str:
    other = profile.get("other_name") or ("NPC")
    if not history:
        return ""
    first = history[0]
    last = history[-1]
    rows = [f"t{e['turn']}: trust={e['trust']} intimacy={e['intimacy']} tension={e['tension']} ({e.get('reason','')})"
            for e in history[-20:]]
    if lang == "ja":
        return (
            f"NPC: {other}\n"
            f"開始時: trust={first['trust']-first.get('d_trust',0)} intimacy={first['intimacy']-first.get('d_intimacy',0)} tension={first['tension']-first.get('d_tension',0)}\n"
            f"終了時: trust={last['trust']} intimacy={last['intimacy']} tension={last['tension']}\n"
            f"推移:\n" + "\n".join(rows) +
            "\n\n2〜3文で、プレイヤーに向けてこのプレイスルーの感情的な弧を描写してください。"
        )
    return (
        f"NPC: {other}\n"
        f"Start: trust={first['trust']-first.get('d_trust',0)} intimacy={first['intimacy']-first.get('d_intimacy',0)} tension={first['tension']-first.get('d_tension',0)}\n"
        f"End: trust={last['trust']} intimacy={last['intimacy']} tension={last['tension']}\n"
        f"Timeline:\n" + "\n".join(rows) +
        "\n\nWrite 2-3 sentences to the player describing the emotional arc of this playthrough."
    )


def generate_recap(history: list, profile: dict, model: str, lang: str = "en") -> str:
    if not history:
        return ""
    sys_prompt = RECAP_SYS_JA if lang == "ja" else RECAP_SYS_EN
    user_prompt = _recap_user_prompt(history, profile, lang)
    raw = _safe_chat(model, [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ], temperature=0.7)
    return (raw or "").strip()


# ── Persistence ───────────────────────────────────────────────────────────

def _summaries_dir(lang: str) -> Path:
    return SUMMARIES_DIR_JA if (lang or "").lower() == "ja" else SUMMARIES_DIR


def _affinity_path(story: str, lang: str) -> Path:
    return _summaries_dir(lang) / f"{story}_affinity.json"


def load_prior(story: str, lang: str) -> dict | None:
    if not story:
        return None
    path = _affinity_path(story, lang)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def save_final(story: str, lang: str, final: dict, history: list, narrative: str, turns: int) -> None:
    """Persist affinity to disk. If narrative is empty, preserves any existing
    narrative from a prior save so per-turn autosaves don't clobber a
    previously-generated recap."""
    if not story:
        return
    path = _affinity_path(story, lang)
    try:
        if not narrative:
            existing = load_prior(story, lang)
            if existing and existing.get("narrative"):
                narrative = existing["narrative"]
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "story": story,
            "lang": lang,
            "last_played": datetime.datetime.now().isoformat(timespec="seconds"),
            "final": {k: final.get(k, 0) for k in ("trust", "intimacy", "tension")},
            "narrative": narrative,
            "history": history,
            "turns": turns,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass
