"""Jaipur web UI — play against AI in the browser.

FastAPI + HTMX + Jinja2. Zero JS frameworks.
Run: uvicorn web.app:app --reload
"""
from __future__ import annotations

import random
import uuid
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from jaipur.cards import GOODS, GoodType
from jaipur.game_fast import (
    ACT_EXCHANGE,
    ACT_SELL,
    ACT_TAKE_CAMELS,
    ACT_TAKE_ONE,
    GameState,
    _G2I,
    _N_GOODS,
)

_DIR = Path(__file__).parent
app = FastAPI(title="Jaipur")
app.mount("/static", StaticFiles(directory=_DIR / "static"), name="static")
templates = Jinja2Templates(directory=_DIR / "templates")

# --------------- AI opponent ---------------

# Try loading neural agent, fall back to greedy
_ai_agent = None

def _get_ai():
    global _ai_agent
    if _ai_agent is not None:
        return _ai_agent
    try:
        from ai.agents import NeuralAgent
        from pathlib import Path as P
        model_path = P(__file__).parent.parent / "models" / "best.pt"
        if model_path.exists():
            _ai_agent = NeuralAgent(str(model_path), device="cpu")
            print(f"[web] Loaded neural AI from {model_path}")
            return _ai_agent
    except Exception as e:
        print(f"[web] Could not load neural agent: {e}")
    from jaipur.agents import GreedyAgent
    _ai_agent = GreedyAgent()
    print("[web] Using GreedyAgent as AI opponent")
    return _ai_agent


# --------------- Session store (in-memory) ---------------

_sessions: dict[str, dict] = {}


def _new_game() -> dict:
    rng = random.Random()
    gs = GameState.new_round(rng)
    return {
        "gs": gs,
        "rng": rng,
        "round": 1,
        "wins": [0, 0],  # player, ai
        "log": ["Round 1 started. Your turn!"],
        "game_over": False,
        "match_result": None,
    }


def _get_session(session_id: str) -> dict:
    if session_id not in _sessions:
        _sessions[session_id] = _new_game()
    return _sessions[session_id]


# --------------- Helpers ---------------

GOOD_NAMES = {g: g.value.capitalize() for g in GOODS}
GOOD_EMOJI = {
    GoodType.DIAMOND: "💎",
    GoodType.GOLD: "🪙",
    GoodType.SILVER: "🥈",
    GoodType.CLOTH: "🧶",
    GoodType.SPICE: "🌶️",
    GoodType.LEATHER: "🐄",
}


def _format_action(action: tuple) -> str:
    atype = action[0]
    if atype == ACT_TAKE_ONE:
        g = GOODS[action[1]]
        return f"Take {GOOD_EMOJI[g]} {GOOD_NAMES[g]}"
    elif atype == ACT_TAKE_CAMELS:
        return "Take 🐪 Camels"
    elif atype == ACT_SELL:
        g = GOODS[action[1]]
        return f"Sell {action[2]}× {GOOD_EMOJI[g]} {GOOD_NAMES[g]}"
    elif atype == ACT_EXCHANGE:
        take = ", ".join(f"{GOOD_EMOJI[GOODS[i]]}" for i in action[1])
        give_g = ", ".join(f"{GOOD_EMOJI[GOODS[i]]}" for i in action[2])
        give_c = f"+{action[3]}🐪" if action[3] else ""
        give = give_g + (" " + give_c if give_c else "")
        return f"Exchange: get {take} / give {give or give_c}"
    return str(action)


def _market_display(gs: GameState) -> list[dict]:
    items = []
    for i, g in enumerate(GOODS):
        for _ in range(gs.market[i]):
            items.append({"emoji": GOOD_EMOJI[g], "name": GOOD_NAMES[g]})
    for _ in range(gs.market[_N_GOODS]):
        items.append({"emoji": "🐪", "name": "Camel"})
    return items


def _hand_display(gs: GameState, player: int) -> list[dict]:
    p = gs.players[player]
    items = []
    for i, g in enumerate(GOODS):
        if p.hand[i] > 0:
            items.append({
                "emoji": GOOD_EMOJI[g],
                "name": GOOD_NAMES[g],
                "count": p.hand[i],
            })
    return items


def _tokens_display(gs: GameState) -> list[dict]:
    result = []
    for i, g in enumerate(GOODS):
        pile = gs.tokens[i]
        result.append({
            "emoji": GOOD_EMOJI[g],
            "name": GOOD_NAMES[g],
            "top": pile[0] if pile else "-",
            "remaining": len(pile),
        })
    return result


def _build_context(session: dict, request: Request) -> dict:
    gs = session["gs"]
    actions = []
    if not session["game_over"] and gs.current_player == 0 and not gs.round_over:
        actions = [(i, _format_action(a)) for i, a in enumerate(gs.get_legal_actions())]

    return {
        "request": request,
        "market": _market_display(gs),
        "hand": _hand_display(gs, 0),
        "ai_hand_size": gs.players[1].hand_size,
        "player_camels": gs.players[0].camels,
        "ai_camels": gs.players[1].camels,
        "player_score": gs.players[0].score,
        "ai_score": gs.players[1].score,
        "tokens": _tokens_display(gs),
        "actions": actions,
        "log": session["log"][-8:],
        "round": session["round"],
        "wins": session["wins"],
        "deck_remaining": gs.deck_size,
        "game_over": session["game_over"],
        "match_result": session["match_result"],
        "round_over": gs.round_over,
    }


# --------------- Routes ---------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    sid = request.cookies.get("jaipur_sid")
    if not sid:
        sid = uuid.uuid4().hex
    session = _get_session(sid)
    ctx = _build_context(session, request)
    resp = templates.TemplateResponse("game.html", ctx)
    resp.set_cookie("jaipur_sid", sid, max_age=3600 * 24)
    return resp


@app.post("/action", response_class=HTMLResponse)
async def do_action(request: Request, action_idx: int = Form(...)):
    sid = request.cookies.get("jaipur_sid", "")
    session = _get_session(sid)
    gs = session["gs"]

    if session["game_over"] or gs.round_over:
        return RedirectResponse("/", status_code=303)

    legal = gs.get_legal_actions()
    if action_idx < 0 or action_idx >= len(legal):
        return RedirectResponse("/", status_code=303)

    # Player move
    action = legal[action_idx]
    session["log"].append(f"You: {_format_action(action)}")
    gs = gs.apply_action(action)
    session["gs"] = gs

    # Check round end
    if gs.round_over:
        _handle_round_end(session)
    else:
        # AI turn
        _ai_turn(session)

    ctx = _build_context(session, request)
    return templates.TemplateResponse("game.html", ctx)


@app.post("/new-game", response_class=HTMLResponse)
async def new_game(request: Request):
    sid = request.cookies.get("jaipur_sid", uuid.uuid4().hex)
    _sessions[sid] = _new_game()
    return RedirectResponse("/", status_code=303)


def _ai_turn(session: dict):
    gs = session["gs"]
    if gs.current_player != 1:
        return
    ai = _get_ai()
    while gs.current_player == 1 and not gs.round_over:
        actions = gs.get_legal_actions()
        if not actions:
            break
        action = ai.choose(gs, actions)
        session["log"].append(f"AI: {_format_action(action)}")
        gs = gs.apply_action(action)
    session["gs"] = gs
    if gs.round_over:
        _handle_round_end(session)


def _handle_round_end(session: dict):
    gs = session["gs"]
    s0, s1 = gs.score_round()
    winner = gs.round_winner()

    if winner == 0:
        session["wins"][0] += 1
        session["log"].append(f"🏆 You win the round! ({s0} vs {s1})")
    elif winner == 1:
        session["wins"][1] += 1
        session["log"].append(f"💀 AI wins the round! ({s1} vs {s0})")
    else:
        session["log"].append(f"🤝 Round tied! ({s0} vs {s1})")

    # Check match (best of 3)
    if session["wins"][0] >= 2:
        session["game_over"] = True
        session["match_result"] = "win"
        session["log"].append("🎉 You win the match!")
    elif session["wins"][1] >= 2:
        session["game_over"] = True
        session["match_result"] = "lose"
        session["log"].append("😤 AI wins the match!")
    elif session["round"] >= 3:
        session["game_over"] = True
        if session["wins"][0] > session["wins"][1]:
            session["match_result"] = "win"
            session["log"].append("🎉 You win the match!")
        else:
            session["match_result"] = "lose"
            session["log"].append("😤 AI wins the match!")
    else:
        # Next round
        session["round"] += 1
        session["gs"] = GameState.new_round(session["rng"])
        session["log"].append(f"--- Round {session['round']} ---")

        # If AI goes first in new round, let it play
        if session["gs"].current_player == 1:
            _ai_turn(session)
