"""
space_app.py – Gradio demo for the InboxOps benchmark.

Usage:
    pip install gradio
    python app/space_app.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import gradio as gr
except ImportError:
    print("ERROR: gradio not installed. Run: pip install gradio")
    sys.exit(1)

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
        "scoring": "Per email: label (1/3) + priority (1/3) + owner (1/3). Spam archived → **+0.10**; missed → **−0.10**. Final = mean across 13 emails.",
        "heuristic_score": 0.9231,
        "random_score": 0.2154,
        "ceiling_note": "email_011 (fake Stripe domain) is the intentional hard case — only a domain-aware agent closes this gap.",
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
        "scoring": "Best slot (Apr 10 09:00) → **1.00** · End-of-day → **0.70** · Lunch overlap → **0.55** · Invalid → **0.00** · ±0.05 for calendar review.",
        "heuristic_score": 1.0000,
        "random_score": 0.0000,
        "ceiling_note": None,
    },
    "scheduling_impossible": {
        "display": "Scheduling — No Solution",
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
        "scoring": "report_no_solution + all calendars viewed → **1.00** · Report, skipped check → **0.80** · find_slots, no report → **0.40** · Tried to book → **0.00**.",
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
            "renewal in 3 days) reports two issues: a duplicate charge ($2,000 × 2) and an "
            "account lockout after 5 failed login attempts.\n\n"
            "The agent must investigate billing history, check the auth log, consult all four "
            "relevant policies, route to the correct teams, document findings in an internal note, "
            "draft a customer reply, and escalate — all within the SLA window."
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
#                  ^ or: meeting_scheduling | scheduling_impossible | support_escalation

# Step the agent
state, reward, done, info = env.step({
    "action":   "open_email",
    "email_id": "email_001"
})

# Grade at any point — deterministic, no judge model
score = env.grade()    # float 0.0–1.0
print(env.summary())   # {task, score, total_reward, steps, completed, passed}
'''

# ─── Custom CSS ───────────────────────────────────────────────────────────────

CSS = """
/* ── Page background ───────────────────────────────────────────── */
.gradio-container {
    background: #0f1117 !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}

/* ── Hero banner ───────────────────────────────────────────────── */
.hero-wrap {
    background: linear-gradient(135deg, #1a1f2e 0%, #0f1117 60%, #1a1228 100%);
    border: 1px solid #2a2d3a;
    border-radius: 16px;
    padding: 48px 40px 36px;
    margin-bottom: 8px;
    text-align: center;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 12px;
    letter-spacing: -0.5px;
}
.hero-sub {
    color: #94a3b8;
    font-size: 1.05rem;
    margin: 0 0 28px;
    line-height: 1.6;
}
.hero-pills {
    display: flex;
    gap: 10px;
    justify-content: center;
    flex-wrap: wrap;
}
.pill {
    background: #1e2333;
    border: 1px solid #2e3347;
    border-radius: 999px;
    padding: 5px 14px;
    font-size: 0.78rem;
    color: #94a3b8;
    font-weight: 500;
}

/* ── Section headings ──────────────────────────────────────────── */
.section-heading {
    font-size: 1.1rem;
    font-weight: 700;
    color: #e2e8f0;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin: 32px 0 12px;
    padding-left: 4px;
    border-left: 3px solid #818cf8;
    padding-left: 12px;
}

/* ── Score cards row ───────────────────────────────────────────── */
.score-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 8px;
}
.score-card {
    background: #1a1f2e;
    border: 1px solid #2a2d3a;
    border-radius: 12px;
    padding: 18px 16px;
    text-align: center;
    transition: border-color 0.2s;
}
.score-card:hover { border-color: #818cf8; }
.score-card .task-icon { font-size: 1.6rem; margin-bottom: 6px; }
.score-card .task-name { font-size: 0.75rem; color: #64748b; font-weight: 600;
                          text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; }
.score-card .diff-badge {
    display: inline-block;
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 0.68rem;
    font-weight: 700;
    margin-bottom: 10px;
    letter-spacing: 0.03em;
}
.score-card .heur-score { font-size: 1.7rem; font-weight: 800; color: #e2e8f0; line-height: 1; }
.score-card .score-label { font-size: 0.68rem; color: #475569; margin-top: 2px; }
.score-card .rand-score { font-size: 0.78rem; color: #475569; margin-top: 6px; }

/* ── Task accordion cards ──────────────────────────────────────── */
.task-card {
    background: #1a1f2e !important;
    border: 1px solid #2a2d3a !important;
    border-radius: 12px !important;
    margin-bottom: 10px !important;
}
.task-card > .label-wrap { color: #c4cde0 !important; font-weight: 600 !important; }

/* ── Action chips ──────────────────────────────────────────────── */
.action-chips { display: flex; flex-wrap: wrap; gap: 7px; margin: 10px 0 4px; }
.chip {
    background: #0f1117;
    border: 1px solid #2e3347;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.74rem;
    color: #818cf8;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-weight: 500;
}

/* ── State explorer ────────────────────────────────────────────── */
.explorer-wrap {
    background: #1a1f2e;
    border: 1px solid #2a2d3a;
    border-radius: 12px;
    padding: 24px;
    margin-top: 4px;
}

/* ── Gradio overrides ──────────────────────────────────────────── */
.gr-button-primary {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
.gr-button-primary:hover { opacity: 0.9 !important; }
footer { display: none !important; }
"""

# ─── HTML helpers ─────────────────────────────────────────────────────────────

def _score_cards_html() -> str:
    cards = []
    for key in TASK_KEYS:
        info = TASK_INFO[key]
        h = info["heuristic_score"]
        r = info["random_score"]
        color = info["diff_color"]
        cards.append(f"""
        <div class="score-card">
          <div class="task-icon">{info['icon']}</div>
          <div class="task-name">{info['display']}</div>
          <div class="diff-badge" style="background:{color}22;color:{color};">{info['difficulty']}</div>
          <div class="heur-score">{h:.4f}</div>
          <div class="score-label">heuristic</div>
          <div class="rand-score">random: {r:.4f}</div>
        </div>""")
    return f'<div class="score-grid">{"".join(cards)}</div>'


def _action_chips_html(actions: list[str]) -> str:
    chips = "".join(f'<span class="chip">{a}</span>' for a in actions)
    return f'<div class="action-chips">{chips}</div>'


def _hero_html() -> str:
    return """
    <div class="hero-wrap">
      <div class="hero-title">InboxOps</div>
      <div class="hero-sub">
        A deterministic workplace agent benchmark — 4 tasks, clean 0–1 scoring, pure Python.<br>
        <em>Can your agent handle a Monday morning inbox?</em>
      </div>
      <div class="hero-pills">
        <span class="pill">📦 Zero dependencies</span>
        <span class="pill">⚡ Deterministic grading</span>
        <span class="pill">🐍 Python 3.10+</span>
        <span class="pill">🤖 LLM-ready</span>
      </div>
    </div>"""


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get_initial_state(task_key: str) -> str:
    env = InboxOpsEnv()
    state = env.reset(task_key)
    if "inbox" in state:
        state = dict(state)
        state["inbox"] = [
            {k: v for k, v in e.items() if k != "body"}
            for e in state["inbox"]
        ]
    return json.dumps(state, indent=2)


# ─── UI ───────────────────────────────────────────────────────────────────────

def build_app():
    with gr.Blocks(
        title="InboxOps — Workplace Agent Benchmark",
        css=CSS,
        theme=gr.themes.Base(
            primary_hue="violet",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
    ) as demo:

        # ── Hero ─────────────────────────────────────────────────────────────
        gr.HTML(_hero_html())

        # ── Score cards ───────────────────────────────────────────────────────
        gr.HTML('<div class="section-heading">Baseline Scores</div>')
        gr.HTML(_score_cards_html())
        gr.Markdown(
            "<small style='color:#475569'>"
            "Heuristic = hand-tuned keyword classifier &nbsp;·&nbsp; "
            "Random = seed 42 &nbsp;·&nbsp; "
            "Gap = **+0.8144** &nbsp;·&nbsp; "
            "LLM target: close the 0.0769 email_011 gap to reach **1.0000**"
            "</small>",
        )

        # ── Task cards ────────────────────────────────────────────────────────
        gr.HTML('<div class="section-heading">Tasks</div>')

        for key in TASK_KEYS:
            info = TASK_INFO[key]
            label = f"{info['icon']}  {info['display']}  ·  {info['difficulty']}"
            with gr.Accordion(label=label, open=False, elem_classes="task-card"):
                gr.Markdown(info["description"])
                gr.HTML(_action_chips_html(info["actions"]))
                gr.Markdown(f"**Scoring:** {info['scoring']}")
                if info["ceiling_note"]:
                    gr.Markdown(
                        f"> **Ceiling note:** {info['ceiling_note']}",
                    )

        # ── State explorer ────────────────────────────────────────────────────
        gr.HTML('<div class="section-heading">State Explorer</div>')

        with gr.Group(elem_classes="explorer-wrap"):
            gr.Markdown(
                "Select a task and click **Load State** to inspect the exact JSON the agent "
                "receives at `env.reset()`. Email bodies are hidden until `open_email` is called.",
                elem_id="explorer-desc",
            )
            with gr.Row():
                task_dropdown = gr.Dropdown(
                    choices=[(f"{TASK_INFO[k]['icon']} {TASK_INFO[k]['display']}", k) for k in TASK_KEYS],
                    value="email_triage",
                    label="Task",
                    scale=3,
                )
                load_btn = gr.Button("Load State", variant="primary", scale=1, min_width=120)
            state_box = gr.Code(language="json", label="env.reset() → initial state", lines=28)

        load_btn.click(fn=_get_initial_state, inputs=task_dropdown, outputs=state_box)

        # ── API quick-start ───────────────────────────────────────────────────
        gr.HTML('<div class="section-heading">API Quick-Start</div>')
        gr.Code(value=_CODE_EXAMPLE, language="python", label="Python")

        # ── Footer ────────────────────────────────────────────────────────────
        gr.Markdown(
            "<div style='text-align:center;color:#334155;font-size:0.8rem;margin-top:32px;padding-top:16px;"
            "border-top:1px solid #1e2333'>"
            "stdlib-only core &nbsp;·&nbsp; "
            "<code>pip install openai</code> for the LLM harness &nbsp;·&nbsp; "
            "<code>pip install gradio</code> for this demo"
            "</div>"
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
