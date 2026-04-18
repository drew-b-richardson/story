"""Character affinity scoring and persistence.

Tracks three dimensions (trust, intimacy, tension) between the player and the
NPC, updated after each NPC response via a non-streaming Ollama call. All
public functions fail soft — on any error they return None / "" / {} so the
main game flow is never blocked by scoring.
"""

import json
import re
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

SCORER_SYS_EN = """You are a relationship analyst. You will be given:
  PLAYER ACTION — what the player just said or did (this is what you are scoring)
  STORY RESPONSE — how the story narrator described what happened next (IGNORE THIS for scoring — it is generated fiction and does not determine relationship impact)

Score how the PLAYER ACTION shifted three dimensions. Integer deltas only, each in [-2, +2].

━━ TRUST — does the NPC feel safer or more betrayed? ━━
Score 0 for routine exchanges with no honesty or deception involved.
Earns +1: Player is honest, keeps a promise, respects a stated limit, admits something difficult.
Earns +2: Player makes a meaningful sacrifice, is deeply vulnerable, or earns real faith.
Loses -1: Player is evasive, breaks a minor promise, or pushes past a stated limit.
Loses -2: Player lies, deceives, is physically violent, or violates a dealbreaker.

━━ INTIMACY — emotional or physical closeness genuinely gained or lost? ━━
Bias hard toward 0. Only move when something real happens.
Earns +1: Player shares something personal, initiates genuine physical contact, or says something that lands emotionally.
Earns +2: A meaningful moment of real connection — confession, first kiss, deep vulnerability.
Loses -1: Player is cold, dismissive, or pulls away emotionally.
Loses -2: Player is cruel, violent, or explicitly rejects the NPC.
Do NOT award positive intimacy for passive or ambiguous actions.

━━ TENSION — unresolved friction, stakes, or desire rising or falling? ━━
Tension is NOT only conflict — it also includes unresolved romantic/sexual desire and raised stakes.
Score 0 for exchanges where nothing is left unresolved.
Rises +1: Player says something provocative, teasing, or ambiguous; a question goes unanswered; desire is hinted but not acted on.
Rises +2: Direct argument or confrontation; a secret is revealed or threatened; competing wants collide.
Falls -1: A moment of ease or relief; something is settled or laughed off.
Falls -2: A major conflict is resolved; an apology is accepted; the air fully clears.

━━ ALWAYS ━━
Physical violence = trust -2, intimacy -2, tension +2.
Dealbreaker violations = trust -2, tension +1 at minimum.
Score the player's intent, not the story's description of the outcome.

NPC: {other_name}
Relationship stage: {relationship_stage}
Desires: {desires}
Dealbreakers: {dealbreakers}

Return ONLY JSON, no prose, no code fences. Schema:
{"trust": <int in [-2,2]>, "intimacy": <int in [-2,2]>, "tension": <int in [-2,2]>, "reason": "<one short sentence describing what the player did>"}
"""

SCORER_SYS_JA = """あなたは恋愛関係の分析者です。以下の2つが与えられます:
  PLAYER ACTION — プレイヤーが言ったこと・したこと（これを採点する）
  STORY RESPONSE — 物語のナレーターがその後を描写したテキスト（採点には使わないこと — フィクションであり、関係への影響を決定しない）

PLAYER ACTIONが3つの次元をどう変化させたかを採点してください。整数の差分のみ、各[-2, +2]の範囲。

━━ TRUST（信頼）— NPCはプレイヤーに安心を感じたか、裏切られたと感じたか? ━━
特に何もない日常的なやりとりは0。
+1: 正直に話す、約束を守る、相手の限界を尊重する、難しいことを認める。
+2: 本当の意味で脆さを見せる、信頼を深める行動をとる。
-1: 言葉を濁す、小さな約束を破る、相手の限界を超えようとする。
-2: 嘘をつく、欺く、身体的暴力、「許せないこと」の違反。

━━ INTIMACY（親密さ）— 感情的・身体的な近さが実際に増したか減ったか? ━━
強く0寄りに。本当に何かが起きた時だけ動かす。
+1: 個人的なことを打ち明ける、本物の身体的接触、感情に響く言葉。
+2: 告白、初めてのキス、深い脆さの共有など、本物の繋がりの瞬間。
-1: 冷たい、感情的に距離を置く、そっけない態度。
-2: 残酷な言葉、暴力、明確な拒絶。
曖昧・消極的なプレイヤー行動にはポジティブスコアをつけないこと。

━━ TENSION（緊張）— 未解決の摩擦・欲望・対立が高まったか和らいだか? ━━
緊張は対立だけではない。未解決のロマンティックな欲望や高まる期待も含む。
特に何も残らないやりとりは0。
+1: 挑発的・曖昧な発言、質問への答えを避ける、欲望をほのめかすが行動しない。
+2: 直接的な口論・対立、秘密の暴露、強い欲求の衝突。
-1: 緊張が和らぐ、笑いで流せる、小さな解決。
-2: 大きな対立の解消、謝罪が受け入れられる、完全に空気が晴れる。

━━ 常に適用 ━━
身体的暴力 = trust -2, intimacy -2, tension +2。
「許せないこと」の違反 = trust -2, tension +1 以上。
物語の描写ではなく、プレイヤーの意図と行動を採点すること。

NPC: {other_name}
関係の段階: {relationship_stage}
望むもの: {desires}
許せないこと: {dealbreakers}

JSONのみ返す(前置き・コードフェンス禁止)。スキーマ:
{"trust": <[-2,2]の整数>, "intimacy": <[-2,2]の整数>, "tension": <[-2,2]の整数>, "reason": "<プレイヤーが何をしたかを一文で>"}
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
        base = {"trust": 3, "intimacy": 0, "tension": 4}
    elif "estranged" in stage or "疎遠" in stage or "別れ" in stage:
        base = {"trust": 3, "intimacy": 0, "tension": 7}
    elif "lover" in stage or "partner" in stage or "恋人" in stage or "夫婦" in stage:
        base = {"trust": 7, "intimacy": 0, "tension": 3}
    elif "friend" in stage or "友" in stage:
        base = {"trust": 6, "intimacy": 0, "tension": 2}
    else:
        base = {"trust": 5, "intimacy": 0, "tension": 3}
    return {**base, "turn": 0, "last_reason": "", "updated_at": time.time()}


# ── Scoring ───────────────────────────────────────────────────────────────

# Patterns that unambiguously mean a negative player action regardless of
# how the story LLM narrated the outcome.
_VIOLENCE_RE = re.compile(
    r"\b(slap|slaps|slapped|hit|hits|struck|punch|punches|punched|shove|shoves|shoved"
    r"|grab|grabs|grabbed|choke|chokes|choked|kick|kicks|kicked|strike|strikes)\b",
    re.IGNORECASE,
)
_BREAKUP_RE = re.compile(
    r"\b(break up|broke up|breaking up|leave you|leaving you|never want to see"
    r"|i hate you|get out|get away from me|i'm done|i am done|it's over|it is over"
    r"|we're over|we are over|goodbye forever)\b",
    re.IGNORECASE,
)
_INSULT_RE = re.compile(
    r"\b(idiot|stupid|pathetic|worthless|disgusting|ugly|loser|freak|moron|shut up"
    r"|i don't care about you|don't care about you|you mean nothing)\b",
    re.IGNORECASE,
)


def _rule_based_override(player_text: str) -> dict | None:
    """Return a hard-coded delta if the player text unambiguously warrants it,
    bypassing the LLM scorer entirely. Returns None if no rule fires."""
    t = player_text.lower()
    if _VIOLENCE_RE.search(t):
        return {"trust": -2, "intimacy": -2, "tension": 2,
                "reason": "player was physically violent"}
    if _BREAKUP_RE.search(t):
        return {"trust": -2, "intimacy": -2, "tension": 2,
                "reason": "player rejected or ended the relationship"}
    if _INSULT_RE.search(t):
        return {"trust": -2, "intimacy": -1, "tension": 1,
                "reason": "player was cruel or insulting"}
    return None


def _last_player_message(msgs: list) -> str:
    """Return the most recent user (player) message content, or empty string."""
    for m in reversed(msgs):
        if m.get("role") == "user":
            return (m.get("content") or "").strip()
    return ""


def _format_for_scorer(msgs: list, other_name: str) -> str:
    """Format messages for the scorer, clearly separating the player's last
    action from the story's narrative response so the model scores the action,
    not the (potentially misleading) narrative outcome."""
    non_system = [m for m in msgs if m.get("role") != "system" and (m.get("content") or "").strip()]

    # Collect up to 3 prior exchanges for context (excluding the last pair)
    context_lines = []
    prior = non_system[:-2] if len(non_system) >= 2 else []
    for m in prior[-4:]:
        role = m.get("role")
        label = "PLAYER" if role == "user" else other_name.upper()
        context_lines.append(f"{label}: {(m.get('content') or '').strip()}")

    # The last player message and last story response
    player_action = ""
    story_response = ""
    if non_system:
        last = non_system[-1]
        second_last = non_system[-2] if len(non_system) >= 2 else None
        if last.get("role") == "assistant":
            story_response = (last.get("content") or "").strip()
            if second_last and second_last.get("role") == "user":
                player_action = (second_last.get("content") or "").strip()
        elif last.get("role") == "user":
            player_action = (last.get("content") or "").strip()

    if not player_action:
        return ""

    parts = []
    if context_lines:
        parts.append("RECENT CONTEXT:\n" + "\n\n".join(context_lines))
    parts.append(f"PLAYER ACTION (score this):\n{player_action}")
    if story_response:
        parts.append(f"STORY RESPONSE (do not use for scoring — fiction only):\n{story_response}")
    return "\n\n---\n\n".join(parts)


def score_turn(profile: dict, recent_msgs: list, current: dict, model: str, lang: str = "en") -> dict | None:
    other_name = profile.get("other_name") or "the NPC"
    stage = profile.get("relationship_stage") or ""
    desires = ", ".join(profile.get("desires", []) or []) or "—"
    dealbreakers = ", ".join(profile.get("dealbreakers", []) or []) or "—"

    # Rule-based override: don't waste an LLM call on obvious violence/cruelty.
    player_text = _last_player_message(recent_msgs)
    print(f"[affinity] player_text: {player_text[:120]!r}", flush=True)
    override = _rule_based_override(player_text)
    if override:
        print(f"[affinity] rule override fired: {override}", flush=True)
        return override

    template = SCORER_SYS_JA if lang == "ja" else SCORER_SYS_EN
    sys_prompt = (template
                  .replace("{other_name}", other_name)
                  .replace("{relationship_stage}", stage)
                  .replace("{desires}", desires)
                  .replace("{dealbreakers}", dealbreakers))

    user_prompt = _format_for_scorer(recent_msgs, other_name)
    if not user_prompt:
        print("[affinity] _format_for_scorer returned empty — no player message found", flush=True)
        return None

    raw = _safe_chat(model, [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ])
    print(f"[affinity] LLM raw response: {raw!r}", flush=True)
    if not raw:
        return None
    parsed = _parse_json(raw)
    print(f"[affinity] parsed: {parsed}", flush=True)
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
