"""
run_llm_agent.py – LLM agent harness for the InboxOps benchmark.

Runs all 4 tasks using an OpenAI-compatible chat model as the agent.
The agent receives the current environment state as a JSON-formatted
system prompt and must return a JSON action dict each turn.

COST GUARD
----------
This script will not make any API calls unless the environment variable
ENABLE_LLM_RUN=true is explicitly set. This prevents accidental token
spend when the script is run during CI, demos, or casual exploration.

Usage:
    export ENABLE_LLM_RUN=true
    export OPENAI_API_KEY=sk-...
    python scripts/run_llm_agent.py

    # Run a single task
    python scripts/run_llm_agent.py --task email_triage

    # Use a different model or base URL (e.g. local Ollama)
    python scripts/run_llm_agent.py --model gpt-4o --max-steps 60
    python scripts/run_llm_agent.py --base-url http://localhost:11434/v1 --model llama3

Requirements:
    pip install openai
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Cost guard — must be checked before any import that could trigger side-effects ──
if os.environ.get("ENABLE_LLM_RUN", "").lower() != "true":
    print(
        "WARNING: LLM run skipped — ENABLE_LLM_RUN is not set.\n"
        "This guard prevents accidental API token spend.\n"
        "To run for real:\n"
        "  export ENABLE_LLM_RUN=true\n"
        "  export OPENAI_API_KEY=sk-..."
    )
    sys.exit(0)

from env import InboxOpsEnv

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)


# ─── Prompts ──────────────────────────────────────────────────────────────────

_SYSTEM_TEMPLATE = textwrap.dedent("""\
    You are an AI agent completing a workplace inbox task in the InboxOps benchmark.

    TASK: {task_name}
    DESCRIPTION: {task_description}

    At each turn you will receive the current environment state as JSON.
    You must respond with a single JSON action object — nothing else.

    Valid action keys depend on the task:

    EMAIL TRIAGE actions:
      {{"action": "open_email",    "email_id": "<id>"}}
      {{"action": "label_email",   "email_id": "<id>", "label": "<billing|support|meeting|sales|spam|internal>"}}
      {{"action": "set_priority",  "email_id": "<id>", "priority": "<low|medium|high>"}}
      {{"action": "assign_owner",  "email_id": "<id>", "owner": "<finance|support|exec|sales|security|hr>"}}
      {{"action": "archive_email", "email_id": "<id>"}}

    SCHEDULING actions:
      {{"action": "view_calendar",      "participant": "<name>"}}
      {{"action": "find_slots"}}
      {{"action": "propose_meeting",    "date": "YYYY-MM-DD", "start": "HH:MM", "end": "HH:MM"}}
      {{"action": "book_meeting",       "date": "YYYY-MM-DD", "start": "HH:MM", "end": "HH:MM"}}
      {{"action": "report_no_solution", "reason": "<explanation>"}}

    SUPPORT ESCALATION actions:
      {{"action": "open_ticket",       "ticket_id": "<id>"}}
      {{"action": "view_customer",     "customer_id": "<id>"}}
      {{"action": "inspect_billing",   "customer_id": "<id>"}}
      {{"action": "check_auth_status", "customer_id": "<id>"}}
      {{"action": "search_policy",     "policy_id": "<refund_policy|billing_policy|escalation_policy|security_policy>"}}
      {{"action": "assign_ticket",     "team": "<billing|security>"}}
      {{"action": "add_internal_note", "note": "<text>"}}
      {{"action": "draft_reply",       "content": "<text>"}}
      {{"action": "escalate",          "reason": "<text>"}}

    IMPORTANT RULES:
    - Respond with ONLY a JSON object. No prose, no markdown, no explanation.
    - For email triage: open each email before labelling it (body is hidden until opened).
    - For scheduling: view all calendars before booking; call find_slots to see available windows.
    - For support: investigate thoroughly (billing history, auth log, all policies) before escalating.
    - When you believe the task is complete, take the final action (book_meeting / escalate / etc.).
      The environment will set done=true. You will then stop.
""")

_TASK_DESCRIPTIONS = {
    "email_triage": (
        "Classify 13 emails by label (billing/support/meeting/sales/spam/internal), "
        "priority (low/medium/high), and owner (finance/support/exec/sales/security/hr). "
        "Archive spam and phishing emails. Open each email to see its body before classifying."
    ),
    "meeting_scheduling": (
        "Schedule a 60-minute meeting for Alice Chen, Bob Martinez, and Carol Singh "
        "between 2026-04-09 and 2026-04-11. View all three calendars, find available slots, "
        "and book the best (prefer morning hours)."
    ),
    "scheduling_impossible": (
        "Attempt to schedule a 60-minute meeting for Alice Chen, Bob Martinez, and Carol Singh "
        "between 2026-04-09 and 2026-04-11. If no valid slot exists, call report_no_solution "
        "with a clear explanation. Do NOT book an invalid time."
    ),
    "support_escalation": (
        "Investigate ticket TKT-001 for customer CUST-001 (VIP enterprise). "
        "Check billing history, auth status, and all relevant policies. "
        "Route to the appropriate teams, add an internal note documenting your findings, "
        "draft a customer reply, and escalate as required by policy."
    ),
}

MAX_STEPS_DEFAULT = 80


# ─── Agent loop ───────────────────────────────────────────────────────────────

def _state_to_prompt(state: dict) -> str:
    """Serialize environment state as a clean JSON string for the model."""
    # Strip internal/private keys that clutter the context
    clean = {k: v for k, v in state.items() if not k.startswith("_")}
    return json.dumps(clean, indent=2)


def _parse_action(response_text: str) -> dict | None:
    """Extract a JSON action from the model's response text."""
    text = response_text.strip()
    # Strip markdown code fences if the model wraps its output
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "action" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    return None


def run_llm_task(
    env: InboxOpsEnv,
    client: OpenAI,
    task_name: str,
    model: str,
    max_steps: int,
    verbose: bool,
) -> dict:
    """Run one task with the LLM agent. Returns the env.summary() dict."""
    env.reset(task_name)
    state = env.state()

    system_prompt = _SYSTEM_TEMPLATE.format(
        task_name=task_name,
        task_description=_TASK_DESCRIPTIONS[task_name],
    )

    messages: list[dict] = []
    done = False

    for step in range(max_steps):
        state_text = _state_to_prompt(state)

        messages.append({"role": "user", "content": state_text})

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt}] + messages,
            temperature=0.0,
            max_tokens=512,
        )

        reply = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": reply})

        action = _parse_action(reply)
        if action is None:
            if verbose:
                print(f"    [step {step+1}] Could not parse action from: {reply[:100]!r}")
            break

        if verbose:
            print(f"    [step {step+1}] action={json.dumps(action)}")

        state, reward, done, info = env.step(action)

        if verbose and info.get("error"):
            print(f"             error: {info['error']}")

        if done:
            break

    return env.summary()


# ─── Runner ───────────────────────────────────────────────────────────────────

TASKS = [
    ("email_triage",          "Email Triage",            "easy"),
    ("meeting_scheduling",    "Meeting Scheduling",       "medium"),
    ("scheduling_impossible", "Scheduling — No Solution", "medium"),
    ("support_escalation",    "Support Escalation",       "hard"),
]

_W = [35, 8, 7, 8, 5]
_HEADERS = ["Task", "Diff", "Score", "Actions", "Pass"]


def _row(vals: list) -> str:
    parts = [str(v).ljust(_W[i]) if i == 0 else str(v).rjust(_W[i]) for i, v in enumerate(vals)]
    return "  " + "  ".join(parts)


def _divider() -> str:
    return "  " + "  ".join("─" * w for w in _W)


def main():
    parser = argparse.ArgumentParser(description="Run InboxOps with an LLM agent")
    parser.add_argument("--task",     default=None,          help="Single task to run (default: all)")
    parser.add_argument("--model",    default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--base-url", default=None,          help="API base URL (for local models)")
    parser.add_argument("--max-steps",type=int, default=MAX_STEPS_DEFAULT, help="Max steps per task")
    parser.add_argument("--verbose",  action="store_true",   help="Print each action")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    client_kwargs: dict = {"api_key": api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    tasks_to_run = [t for t in TASKS if args.task is None or t[0] == args.task]
    if not tasks_to_run:
        print(f"ERROR: Unknown task {args.task!r}. Valid: {[t[0] for t in TASKS]}")
        sys.exit(1)

    env = InboxOpsEnv()
    results = []

    WIDTH = 68
    print()
    print("═" * WIDTH)
    print(f"  InboxOps — LLM Agent Baseline  [{args.model}]")
    print("═" * WIDTH)
    print()
    print(_row(_HEADERS))
    print(_divider())

    for task_name, display_name, difficulty in tasks_to_run:
        if args.verbose:
            print(f"\n  [{display_name}]")
        summary = run_llm_task(
            env, client, task_name, args.model, args.max_steps, args.verbose
        )
        results.append(summary)
        passed = "YES" if summary["passed"] else "no"
        print(_row([display_name, difficulty,
                    f"{summary['score']:.4f}", summary["steps"], passed]))

    mean_score = sum(r["score"] for r in results) / len(results)
    n_passed   = sum(1 for r in results if r["passed"])
    print(_divider())
    print(f"  Avg score: {mean_score:.4f}   Passed: {n_passed}/{len(results)}")
    print("═" * WIDTH)
    print()


if __name__ == "__main__":
    main()
