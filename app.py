"""
app.py - Hugging Face Spaces entry point + Gradio demo for InboxOps.

Run locally:  python app.py
"""

import json
import os
import sys
import uuid

# Make the repo root importable so `from env import InboxOpsEnv` works
_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)
# Also ensure the working directory is on the path (HF Spaces may omit it)
_cwd = os.getcwd()
if _cwd not in sys.path:
    sys.path.insert(0, _cwd)

import gradio as gr
from fastapi import FastAPI, Request, HTTPException
from env import InboxOpsEnv


# ─── Data ─────────────────────────────────────────────────────────────────────

TASK_KEYS = [
    "email_triage",
    "meeting_scheduling",
    "scheduling_impossible",
    "support_escalation",
]

TASK_INFO = {
    "email_triage": {
        "display": "Email Triage",
        "icon": "📧",
        "difficulty": "Easy",
        "diff_color": "#22c55e",
        "description": (
            "The agent receives 13 emails in an inbox. For each email it must assign a "
            "**label** (billing / support / meeting / sales / spam / internal), "
            "a **priority** (low / medium / high), and an **owner** (finance / support / exec / sales / security / hr). "
            "Spam and phishing emails must be archived.\n\n"
            "The inbox contains intentional edge cases: a phishing email impersonating Stripe "
            "that a keyword classifier will misclassify, an internal security alert that must "
            "route to `security` (not `hr`), and a mixed support/sales email where the bug dominates."
        ),
        "actions": ["open_email", "label_email", "set_priority", "assign_owner", "archive_email"],
        "scoring": "Per email: label (1/3) + priority (1/3) + owner (1/3). Spam archived +0.10; missed -0.10. Final = mean across 13 emails.",
        "heuristic_score": 0.9231,
        "random_score": 0.2154,
        "ceiling_note": "email_011 (fake Stripe domain) is the intentional hard case - only a domain-aware agent closes this gap.",
    },
    "meeting_scheduling": {
        "display": "Meeting Scheduling",
        "icon": "📅",
        "difficulty": "Medium",
        "diff_color": "#f59e0b",
        "description": (
            "Schedule a 60-minute meeting for Alice Chen (VP Product), Bob Martinez (Head of Sales), "
            "and Carol Singh (Engineering Lead) between 2026-04-09 and 2026-04-11.\n\n"
            "The agent must view all three calendars, compute available windows, and book the best slot. "
            "Four valid slots exist; the optimal one is a morning slot on April 10."
        ),
        "actions": ["view_calendar", "find_slots", "propose_meeting", "book_meeting", "report_no_solution"],
        "scoring": "Best slot (Apr 10 09:00) = 1.00 · End-of-day = 0.70 · Lunch overlap = 0.55 · Invalid = 0.00 · ±0.05 for calendar review.",
        "heuristic_score": 1.0000,
        "random_score": 0.0000,
        "ceiling_note": None,
    },
    "scheduling_impossible": {
        "display": "Scheduling - No Solution",
        "icon": "🚫",
        "difficulty": "Medium",
        "diff_color": "#f59e0b",
        "description": (
            "Same setup as Meeting Scheduling, but the calendar data is redesigned so that "
            "**no 60-minute overlap exists** across all three participants in the date range.\n\n"
            "The agent must recognise infeasibility and call `report_no_solution` rather than "
            "hallucinating a booking or giving up silently."
        ),
        "actions": ["view_calendar", "find_slots", "report_no_solution"],
        "scoring": "report_no_solution + all calendars viewed = 1.00 · Report, skipped check = 0.80 · find_slots, no report = 0.40 · Tried to book = 0.00.",
        "heuristic_score": 1.0000,
        "random_score": 0.1000,
        "ceiling_note": None,
    },
    "support_escalation": {
        "display": "Support Escalation",
        "icon": "🎫",
        "difficulty": "Hard",
        "diff_color": "#ef4444",
        "description": (
            "Ticket TKT-001 from Marcus Wei at Global Tech Solutions (VIP enterprise, $24k/year, "
            "renewal in 3 days) reports two issues: a duplicate charge ($2,000 x 2) and an "
            "account lockout after 5 failed login attempts.\n\n"
            "The agent must investigate billing history, check the auth log, consult all four "
            "relevant policies, route to the correct teams, document findings in an internal note, "
            "draft a customer reply, and escalate - all within the SLA window."
        ),
        "actions": ["open_ticket", "view_customer", "inspect_billing", "check_auth_status",
                    "search_policy", "assign_ticket", "add_internal_note", "draft_reply", "escalate"],
        "scoring": "Checklist-based (13 items, weights sum to 1.0). Escalation alone = 0.10; full systematic investigation required for 1.0.",
        "heuristic_score": 1.0000,
        "random_score": 0.3500,
        "ceiling_note": None,
    },
}

_CODE_EXAMPLE = '''\
from env import InboxOpsEnv

env   = InboxOpsEnv()
state = env.reset("email_triage")
#       or: meeting_scheduling | scheduling_impossible | support_escalation

state, reward, done, info = env.step({
    "action":   "open_email",
    "email_id": "email_001"
})

score = env.grade()   # float 0.0-1.0, deterministic
print(env.summary())  # {task, score, total_reward, steps, completed, passed}
'''

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

* { box-sizing: border-box; }

html, body { background: #080b12 !important; }

.gradio-container {
    background: #080b12 !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    max-width: 1100px !important;
    margin: 0 auto !important;
    padding: 0 24px !important;
}

/* canvas background */
#dot-canvas {
    position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
    z-index: 0; pointer-events: none; opacity: 0.35;
}

.gradio-container > * { position: relative; z-index: 1; }

/* ── Hero ── */
.hero-wrap {
    position: relative; overflow: hidden;
    background: linear-gradient(160deg, #0d1321 0%, #080b12 50%, #110d1f 100%);
    border: 1px solid #1e2433; border-radius: 20px;
    padding: 60px 40px 48px; margin-bottom: 4px; text-align: center;
}
.hero-wrap::before {
    content: ''; position: absolute;
    top: -80px; left: 50%; transform: translateX(-50%);
    width: 600px; height: 300px; border-radius: 50%;
    background: radial-gradient(ellipse, rgba(129,140,248,.13) 0%, transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-block; margin-bottom: 20px;
    background: rgba(129,140,248,.1); border: 1px solid rgba(129,140,248,.25);
    border-radius: 999px; padding: 4px 14px;
    font-size: 0.72rem; font-weight: 600; color: #818cf8;
    letter-spacing: 0.1em; text-transform: uppercase;
}
.hero-title {
    font-size: 3.4rem; font-weight: 800; line-height: 1.1;
    background: linear-gradient(135deg, #e2e8f0 0%, #818cf8 40%, #c084fc 70%, #f472b6 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0 0 16px; letter-spacing: -1.5px;
}
.hero-sub {
    color: #64748b; font-size: 1rem; line-height: 1.7;
    margin: 0 auto 32px; max-width: 520px;
}
.hero-pills { display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; }
.pill {
    background: rgba(255,255,255,.03); border: 1px solid #1e2433;
    border-radius: 999px; padding: 5px 14px;
    font-size: 0.75rem; color: #475569; font-weight: 500;
    transition: border-color .2s, color .2s;
}
.pill:hover { border-color: #334155; color: #94a3b8; }

/* ── Section heading ── */
.section-heading {
    font-size: 0.7rem; font-weight: 700; color: #334155;
    letter-spacing: 0.12em; text-transform: uppercase;
    margin: 36px 0 14px; display: flex; align-items: center; gap: 10px;
}
.section-heading::after {
    content: ''; flex: 1; height: 1px; background: #1a1f2e;
}

/* ── Score gauge cards ── */
.sc-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px; margin-bottom: 10px;
}
@media (min-width: 640px) { .sc-grid { grid-template-columns: repeat(4, 1fr); } }

.sc-card {
    background: #0d1018;
    border: 1px solid #1a1f2e;
    border-radius: 18px;
    padding: 18px 14px 14px;
    text-align: center;
    opacity: 0;
    animation: sc-rise 0.55s ease forwards;
    box-shadow:
        0 0 8px rgba(0,0,0,.03),
        0 2px 6px rgba(0,0,0,.08),
        inset 1px 1px 1px -0.5px rgba(255,255,255,.06),
        inset -1px -1px 1px -0.5px rgba(255,255,255,.06),
        inset 0 0 6px 6px rgba(255,255,255,.03);
    transition: border-color .25s, transform .25s, box-shadow .25s;
}
@keyframes sc-rise {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
.sc-card:hover {
    border-color: #2d3555;
    transform: translateY(-4px);
    box-shadow: 0 16px 48px rgba(129,140,248,.1),
        inset 1px 1px 1px -0.5px rgba(255,255,255,.08);
}
.sc-header {
    display: flex; align-items: center;
    justify-content: space-between; margin-bottom: 10px;
}
.sc-icon { font-size: 1.25rem; }
.sc-badge {
    border-radius: 999px; padding: 3px 10px;
    font-size: 0.62rem; font-weight: 700;
    letter-spacing: 0.06em; text-transform: uppercase;
}
.sc-gauge-wrap { position: relative; }
.sc-gauge { width: 100%; height: auto; display: block; overflow: visible; }
.sc-track {
    fill: none; stroke: #1a1f2e; stroke-width: 11; stroke-linecap: round;
}
.sc-arc {
    fill: none; stroke-width: 11; stroke-linecap: round;
    stroke-dasharray: 0 283;
    transition: stroke-dasharray 1.3s cubic-bezier(0.65,0,0.35,1);
}
.sc-score-center {
    position: absolute; bottom: 6px; left: 0; right: 0;
    text-align: center; pointer-events: none;
}
.sc-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.55rem; font-weight: 800; color: #e2e8f0; line-height: 1;
}
.sc-score-lbl {
    font-size: 0.58rem; color: #334155;
    text-transform: uppercase; letter-spacing: 0.08em; margin-top: 1px;
}
.sc-name {
    font-size: 0.65rem; color: #475569; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.07em; margin-top: 10px;
}
.sc-rand { font-size: 0.62rem; color: #1e2a3a; margin-top: 3px; }

/* ── Task accordions ── */
.task-card {
    background: #0d1018 !important; border: 1px solid #1a1f2e !important;
    border-radius: 14px !important; margin-bottom: 8px !important;
    transition: border-color .2s !important;
}
.task-card:hover { border-color: #2d3555 !important; }

/* ── Action chips ── */
.action-chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0 6px; }
.chip {
    background: rgba(129,140,248,.06); border: 1px solid rgba(129,140,248,.15);
    border-radius: 6px; padding: 3px 10px;
    font-size: 0.72rem; color: #818cf8;
    font-family: 'JetBrains Mono', monospace; font-weight: 500;
    transition: background .15s;
}
.chip:hover { background: rgba(129,140,248,.12); }

/* ── Explorer ── */
.explorer-wrap {
    background: #0d1018; border: 1px solid #1a1f2e;
    border-radius: 14px; padding: 24px; margin-top: 4px;
}

/* ── Stat bar ── */
.stat-bar {
    display: flex; gap: 24px; justify-content: center;
    margin: 6px 0 2px; flex-wrap: wrap;
}
.stat-item { text-align: center; }
.stat-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem; font-weight: 700; color: #818cf8;
}
.stat-label { font-size: 0.65rem; color: #334155; text-transform: uppercase; letter-spacing: 0.08em; }

footer { display: none !important; }
"""


# ─── HTML helpers ─────────────────────────────────────────────────────────────

def _score_cards_html() -> str:
    import math
    # Arc: M10,95 A85,85 0 0,1 190,95  (semicircle, r=90, center 100,95)
    # Arc length = π × 90 ≈ 282.74
    R = 90
    ARC_LEN = round(math.pi * R, 2)   # 282.74
    CX, CY = 100, 97
    x0, x1 = CX - R, CX + R          # 10, 190

    # gradient color stops per strength
    STOPS = {
        "Easy":   ["hsl(142,71%,75%)", "hsl(142,71%,50%)", "hsl(142,55%,35%)"],
        "Medium": ["hsl(38,92%,75%)",  "hsl(38,92%,55%)",  "hsl(38,75%,38%)"],
        "Hard":   ["hsl(0,84%,75%)",   "hsl(0,84%,58%)",   "hsl(0,70%,40%)"],
    }

    cards = []
    for i, key in enumerate(TASK_KEYS):
        info   = TASK_INFO[key]
        score  = info["heuristic_score"]
        rand   = info["random_score"]
        color  = info["diff_color"]
        diff   = info["difficulty"]
        stops  = STOPS[diff]
        gid    = f"scg{i}"

        # how much of the arc to fill
        fill = round(score * ARC_LEN, 2)

        stops_html = "".join(
            f'<stop offset="{int(100/(len(stops)-1)*j)}%" stop-color="{s}"/>'
            for j, s in enumerate(stops)
        )
        badge_bg = color + "22"

        cards.append(f"""
        <div class="sc-card" style="animation-delay:{i*130}ms">
          <div class="sc-header">
            <span class="sc-icon">{info['icon']}</span>
            <span class="sc-badge" style="background:{badge_bg};color:{color}">{diff}</span>
          </div>
          <div class="sc-gauge-wrap">
            <svg class="sc-gauge" viewBox="0 0 200 108">
              <defs>
                <linearGradient id="{gid}" x1="0" y1="0" x2="1" y2="0">
                  {stops_html}
                </linearGradient>
              </defs>
              <path class="sc-track"
                d="M{x0},{CY} A{R},{R} 0 0,1 {x1},{CY}"/>
              <path class="sc-arc"
                stroke="url(#{gid})"
                d="M{x0},{CY} A{R},{R} 0 0,1 {x1},{CY}"
                data-fill="{fill}" data-len="{ARC_LEN}"/>
            </svg>
            <div class="sc-score-center">
              <div class="sc-score">{score:.4f}</div>
              <div class="sc-score-lbl">heuristic</div>
            </div>
          </div>
          <div class="sc-name">{info['display']}</div>
          <div class="sc-rand">random baseline: {rand:.4f}</div>
        </div>""")

    return f'<div class="sc-grid">{"".join(cards)}</div>'


def _action_chips_html(actions):
    chips = "".join(f'<span class="chip">{a}</span>' for a in actions)
    return f'<div class="action-chips">{chips}</div>'


def _hero_html() -> str:
    return """
    <canvas id="dot-canvas"></canvas>
    <div class="hero-wrap">
      <div class="hero-badge">Workplace Agent Benchmark</div>
      <div class="hero-title">InboxOps</div>
      <div class="hero-sub">
        A deterministic benchmark for evaluating AI agents on realistic office inbox tasks.<br>
        Four tasks · Clean 0–1 scoring · Pure Python · LLM-ready
      </div>
      <div class="hero-pills">
        <span class="pill">📦 Zero dependencies</span>
        <span class="pill">⚡ Deterministic grading</span>
        <span class="pill">🐍 Python 3.10+</span>
        <span class="pill">🤖 LLM-ready</span>
      </div>
    </div>"""


def _get_initial_state(task_key: str) -> str:
    env = InboxOpsEnv()
    state = env.reset(task_key)
    if "inbox" in state:
        state = dict(state)
        state["inbox"] = [{k: v for k, v in e.items() if k != "body"} for e in state["inbox"]]
    return json.dumps(state, indent=2)


# ─── UI ───────────────────────────────────────────────────────────────────────

_CANVAS_JS = """
() => {
  /* ── dot-wave canvas ── */
  function startCanvas() {
    var c = document.getElementById('dot-canvas');
    if (!c) { setTimeout(startCanvas, 250); return; }
    var ctx = c.getContext('2d');
    var dots = [], t = 0;
    function resize() {
      c.width = window.innerWidth; c.height = window.innerHeight;
      var sp = 24; dots = [];
      for (var i = 0; i <= Math.ceil(c.width/sp)+1; i++)
        for (var j = 0; j <= Math.ceil(c.height/sp)+1; j++)
          dots.push({x: i*sp, y: j*sp});
    }
    function draw() {
      ctx.clearRect(0, 0, c.width, c.height);
      for (var k = 0; k < dots.length; k++) {
        var d = dots[k];
        var w = Math.sin(d.x*0.015 + d.y*0.012 + t) * 0.5 + 0.5;
        ctx.globalAlpha = 0.06 + w * 0.2;
        ctx.fillStyle = '#818cf8';
        ctx.beginPath(); ctx.arc(d.x, d.y, 1.1 + w*0.7, 0, 6.283); ctx.fill();
      }
      t += 0.011; requestAnimationFrame(draw);
    }
    window.addEventListener('resize', resize); resize(); draw();
  }

  /* ── gauge arc animation ── */
  function animateArcs() {
    var arcs = document.querySelectorAll('.sc-arc');
    if (!arcs.length) { setTimeout(animateArcs, 300); return; }
    arcs.forEach(function(arc, idx) {
      var fill = parseFloat(arc.getAttribute('data-fill'));
      var len  = parseFloat(arc.getAttribute('data-len'));
      setTimeout(function() {
        arc.style.strokeDasharray = fill + ' ' + len;
      }, 300 + idx * 130);
    });
  }

  startCanvas();
  animateArcs();
}
"""


def build_app():
    with gr.Blocks(
        title="InboxOps - Workplace Agent Benchmark",
        css=CSS,
        js=_CANVAS_JS,
        theme=gr.themes.Base(
            primary_hue="violet",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
    ) as demo:

        gr.HTML(_hero_html())

        gr.HTML('<div class="section-heading">Baseline Scores</div>')
        gr.HTML(_score_cards_html())
        gr.HTML("""
        <div class="stat-bar">
          <div class="stat-item">
            <div class="stat-value">0.9808</div>
            <div class="stat-label">Mean Score</div>
          </div>
          <div class="stat-item">
            <div class="stat-value">4</div>
            <div class="stat-label">Tasks</div>
          </div>
          <div class="stat-item">
            <div class="stat-value">3/4</div>
            <div class="stat-label">Perfect</div>
          </div>
          <div class="stat-item">
            <div class="stat-value">0→1</div>
            <div class="stat-label">Score Range</div>
          </div>
        </div>""")
        gr.Markdown(
            "<small style='color:#475569;display:block;text-align:center;margin-top:6px'>"
            "Heuristic = hand-tuned keyword classifier &nbsp;·&nbsp; "
            "LLM target: close the 0.0769 `email_011` gap to reach **1.0000**"
            "</small>",
        )

        gr.HTML('<div class="section-heading">Tasks</div>')
        for key in TASK_KEYS:
            info = TASK_INFO[key]
            label = f"{info['icon']}  {info['display']}  ·  {info['difficulty']}"
            with gr.Accordion(label=label, open=False, elem_classes="task-card"):
                gr.Markdown(info["description"])
                gr.HTML(_action_chips_html(info["actions"]))
                gr.Markdown(f"**Scoring:** {info['scoring']}")
                if info["ceiling_note"]:
                    gr.Markdown(f"> **Ceiling note:** {info['ceiling_note']}")

        gr.HTML('<div class="section-heading">State Explorer</div>')
        with gr.Group(elem_classes="explorer-wrap"):
            gr.Markdown(
                "Select a task and click **Load State** to inspect the exact JSON the agent "
                "receives at `env.reset()`. Email bodies are hidden until `open_email` is called."
            )
            with gr.Row():
                task_dropdown = gr.Dropdown(
                    choices=[(f"{TASK_INFO[k]['icon']} {TASK_INFO[k]['display']}", k) for k in TASK_KEYS],
                    value="email_triage",
                    label="Task",
                    scale=3,
                )
                load_btn = gr.Button("Load State", variant="primary", scale=1, min_width=120)
            state_box = gr.Code(language="json", label="env.reset() - initial state", lines=28)

        load_btn.click(fn=_get_initial_state, inputs=task_dropdown, outputs=state_box)

        gr.HTML('<div class="section-heading">API Quick-Start</div>')
        gr.Code(value=_CODE_EXAMPLE, language="python", label="Python")

        gr.Markdown(
            "<div style='text-align:center;color:#334155;font-size:0.8rem;margin-top:32px;"
            "padding-top:16px;border-top:1px solid #1e2333'>"
            "stdlib-only core &nbsp;·&nbsp; "
            "<code>pip install openai</code> for the LLM harness &nbsp;·&nbsp; "
            "<code>pip install gradio</code> for this demo"
            "</div>"
        )

    return demo


demo = build_app()

# ─── REST API (for hackathon portal checks) ───────────────────────────────────

_sessions: dict = {}  # session_id -> InboxOpsEnv

_api = FastAPI()


@_api.get("/health")
def health():
    return {"status": "ok"}


async def _do_reset(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    task = body.get("task") or body.get("task_id") or "email_triage"
    try:
        env = InboxOpsEnv()
        state = env.reset(task)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    session_id = str(uuid.uuid4())
    _sessions[session_id] = env
    return {"session_id": session_id, "state": state}


@_api.post("/reset")
async def api_reset(request: Request):
    return await _do_reset(request)


@_api.post("/api/reset")
async def api_reset_alt(request: Request):
    return await _do_reset(request)


@_api.post("/step")
async def api_step(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    session_id = body.get("session_id")
    action = body.get("action")
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        state, reward, done, info = env.step(action)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"state": state, "reward": reward, "done": done, "info": info}


@_api.get("/grade")
async def api_grade(session_id: str):
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"score": env.grade(), "summary": env.summary()}


app = gr.mount_gradio_app(_api, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860, proxy_headers=True, forwarded_allow_ips="*")
