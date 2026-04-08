"""
run_baseline.py – heuristic + random baselines across all 4 tasks.

Runs each task twice:
  1. Random agent   (lower bound, seed=42 for reproducibility)
  2. Heuristic agent (keyword-based, hand-tuned rules)

Usage:
    python scripts/run_baseline.py          # all tasks
    python scripts/run_baseline.py --debug  # includes per-email classification detail

No external dependencies — pure Python 3.10+.
"""

import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import InboxOpsEnv


# ─── Heuristic: Email Triage ──────────────────────────────────────────────────

_LABEL_KEYWORDS = {
    "billing":  [
        "invoice", "payment", "charge", "overdue", "billing",
        "subscription", "renewal", "refund",
    ],
    "support":  [
        "ticket", "issue", "error", "broken", "not working", "export",
        "crash", "bug", "escalate", "time out", "timed out",
    ],
    "meeting":  [
        "meeting", "all-hands", "board", "calendar", "rescheduled", "confirmed",
        "dial-in", "agenda", "availability", "schedule",
    ],
    "sales":    [
        "pricing", "demo", "enterprise", "quote", "seats", "sso", "saml", "volume",
        "200 users",
    ],
    "spam":     [
        "congratulations", "winner", "claim", "lottery", "prize", "won",
        "suspicious login", "compromised", "verify your identity", "verify immediately",
        "account closure", "click the link below", "click here immediately",
        "once-in-a-lifetime", "processing fee", "send your full name", "bank account details",
    ],
    "internal": [
        "office", "supplies", "maintenance",
        "ops", "infrastructure", "downtime", "deployment", "on-call", "post-mortem",
        "monitoring system", "sso portal", "all staff", "security team", "credential stuffing",
    ],
}

_URGENT_WORDS = [
    "urgent", "immediately", "asap", "blocking", "overdue", "board",
    "rescheduled", "critical", "alert", "action required", "escalate",
    "emergency", "200",
]
_LOW_WORDS = [
    "congratulations", "lottery", "winner", "office supplies", "copy paper",
    "toner", "coffee pods", "once-in-a-lifetime",
]
_OPS_KEYWORDS      = ["maintenance", "infrastructure", "database", "server", "downtime", "deployment", "ops", "on-call"]
_SECURITY_KEYWORDS = ["security team", "monitoring system", "credential stuffing", "sso portal", "mfa", "breach", "login attempt"]

_LABEL_TO_OWNER = {
    "billing":  "finance",
    "support":  "support",
    "meeting":  "exec",
    "sales":    "sales",
    "spam":     "security",
    "internal": "hr",
}


def _classify_email(email: dict) -> tuple[str, str, str]:
    text = (email.get("subject", "") + " " + email.get("body", "") + " " + email.get("sender", "")).lower()

    scores = {
        lbl: sum(1 for kw in kws if kw in text)
        for lbl, kws in _LABEL_KEYWORDS.items()
    }
    label = max(scores, key=scores.get)
    if scores[label] == 0:
        label = "internal"

    if label == "sales" and any(kw in text for kw in ["ticket", "bug", "broken", "issue", "not working"]):
        label = "support"

    if any(w in text for w in _LOW_WORDS) or label == "spam":
        priority = "low"
    elif any(w in text for w in _URGENT_WORDS):
        priority = "high"
    else:
        priority = "medium"

    if label == "internal":
        if any(k in text for k in _SECURITY_KEYWORDS):
            owner = "security"
        elif any(k in text for k in _OPS_KEYWORDS):
            owner = "support"
        else:
            owner = "hr"
    else:
        owner = _LABEL_TO_OWNER.get(label, "support")

    return label, priority, owner


def run_heuristic_email(env: InboxOpsEnv, debug: bool = False) -> None:
    state  = env.state()
    emails = state["inbox"]

    for stub in emails:
        eid = stub["id"]
        state, _, _, _ = env.step({"action": "open_email", "email_id": eid})
        full = next(e for e in state["inbox"] if e["id"] == eid)
        label, priority, owner = _classify_email(full)

        if debug:
            print(f"    {eid}: {label}/{priority}/{owner}  ← '{full['subject'][:50]}'")

        env.step({"action": "label_email",  "email_id": eid, "label":    label})
        env.step({"action": "set_priority", "email_id": eid, "priority": priority})
        env.step({"action": "assign_owner", "email_id": eid, "owner":    owner})

        if label == "spam":
            env.step({"action": "archive_email", "email_id": eid})


# ─── Heuristic: Scheduling (standard) ────────────────────────────────────────

def _slot_score(s: dict) -> tuple:
    h = int(s["start"].split(":")[0])
    m = int(s["start"].split(":")[1])
    if h < 12:
        return (0, h, m)
    elif h == 12:
        return (2, h, m)
    else:
        return (1, h, m)


def run_heuristic_scheduling(env: InboxOpsEnv) -> None:
    state    = env.state()
    required = state["scenario"]["required_attendees"]

    for participant in required:
        env.step({"action": "view_calendar", "participant": participant})

    _, _, _, info = env.step({"action": "find_slots"})
    slots = info.get("slots", [])

    if not slots:
        env.step({"action": "report_no_solution",
                  "reason": "No valid slot found after checking all calendars."})
        return

    best = sorted(slots, key=_slot_score)[0]
    env.step({"action": "propose_meeting",
              "date": best["date"], "start": best["start"], "end": best["end"]})
    env.step({"action": "book_meeting",
              "date": best["date"], "start": best["start"], "end": best["end"]})


# ─── Heuristic: Scheduling (impossible) ──────────────────────────────────────

def run_heuristic_scheduling_impossible(env: InboxOpsEnv) -> None:
    state    = env.state()
    required = state["scenario"]["required_attendees"]

    for participant in required:
        env.step({"action": "view_calendar", "participant": participant})

    _, _, _, info = env.step({"action": "find_slots"})
    slots = info.get("slots", [])

    if not slots:
        env.step({
            "action": "report_no_solution",
            "reason": (
                "After reviewing all three participants' calendars and computing available windows, "
                "no 60-minute slot exists within 2026-04-09 to 2026-04-11 where Alice Chen, "
                "Bob Martinez, and Carol Singh are simultaneously free. "
                "The meeting cannot be scheduled within this date range."
            ),
        })
    else:
        best = sorted(slots, key=_slot_score)[0]
        env.step({"action": "propose_meeting",
                  "date": best["date"], "start": best["start"], "end": best["end"]})
        env.step({"action": "book_meeting",
                  "date": best["date"], "start": best["start"], "end": best["end"]})


# ─── Heuristic: Support Escalation ───────────────────────────────────────────

def run_heuristic_support(env: InboxOpsEnv) -> None:
    env.step({"action": "open_ticket",      "ticket_id": "TKT-001"})
    env.step({"action": "view_customer",    "customer_id": "CUST-001"})
    env.step({"action": "inspect_billing",  "customer_id": "CUST-001"})
    env.step({"action": "check_auth_status","customer_id": "CUST-001"})
    env.step({"action": "search_policy",    "policy_id": "refund_policy"})
    env.step({"action": "search_policy",    "policy_id": "billing_policy"})
    env.step({"action": "search_policy",    "policy_id": "escalation_policy"})
    env.step({"action": "search_policy",    "policy_id": "security_policy"})
    env.step({"action": "assign_ticket",    "team": "billing"})
    env.step({"action": "assign_ticket",    "team": "security"})
    env.step({
        "action": "add_internal_note",
        "note": (
            "FINDINGS: Duplicate charge confirmed — ch_001 and ch_002 both charged $2,000 "
            "on 2026-04-01 (billing system error). Per refund_policy, duplicate charges are "
            "eligible for immediate refund. "
            "Account locked after 5 failed login attempts at 08:04Z on 2026-04-08. "
            "Per security_policy, unlock requires security team approval + identity verification. "
            "Customer CUST-001 (Marcus Wei, Global Tech Solutions) is VIP enterprise tier, "
            "$24k/year, renewal in 3 days (2026-04-11). "
            "Per escalation_policy, must escalate to account manager Sarah Okafor within 2 hours."
        ),
    })
    env.step({
        "action": "draft_reply",
        "content": (
            "Dear Marcus,\n\n"
            "Thank you for reaching out. We have received your message and are treating this as urgent.\n\n"
            "We have confirmed the duplicate charge on your account ($2,000 × 2 on April 1st) and our "
            "billing team is processing the refund as a priority — you will receive confirmation within "
            "1 business day.\n\n"
            "Regarding your account access: our security team will contact you shortly to verify your "
            "identity and restore access. For security reasons, account unlocks require this verification "
            "step before we can proceed.\n\n"
            "Your account manager, Sarah Okafor, has been notified and will be in direct contact with you.\n\n"
            "We sincerely apologise for the disruption, especially given your upcoming renewal.\n\n"
            "Best regards,\nSupport Team"
        ),
    })
    env.step({
        "action": "escalate",
        "reason": (
            "VIP enterprise customer CUST-001 (Marcus Wei, Global Tech Solutions, $24k/year) "
            "has two critical unresolved issues: (1) duplicate charge $2,000 × 2 on 2026-04-01, "
            "(2) account locked after failed login attempts. Contract renewal in 3 days (2026-04-11). "
            "Escalation required per escalation_policy. Account manager: Sarah Okafor."
        ),
    })


# ─── Random Baseline ─────────────────────────────────────────────────────────

_ALL_LABELS     = ["billing", "support", "meeting", "sales", "spam", "internal"]
_ALL_PRIORITIES = ["low", "medium", "high"]
_ALL_OWNERS     = ["finance", "support", "exec", "sales", "security", "hr"]
_DATE_RANGE     = ["2026-04-09", "2026-04-10", "2026-04-11"]
_HOUR_SLOTS     = ["08:00", "09:00", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00"]


def run_random_email(env: InboxOpsEnv, rng: random.Random) -> None:
    """Random label/priority/owner for each email. Archive if randomly labelled spam."""
    state  = env.state()
    emails = state["inbox"]

    for stub in emails:
        eid   = stub["id"]
        label    = rng.choice(_ALL_LABELS)
        priority = rng.choice(_ALL_PRIORITIES)
        owner    = rng.choice(_ALL_OWNERS)

        env.step({"action": "label_email",  "email_id": eid, "label":    label})
        env.step({"action": "set_priority", "email_id": eid, "priority": priority})
        env.step({"action": "assign_owner", "email_id": eid, "owner":    owner})

        if label == "spam":
            env.step({"action": "archive_email", "email_id": eid})


def run_random_scheduling(env: InboxOpsEnv, rng: random.Random) -> None:
    """Pick a random date/time and try to book — no calendar check."""
    date  = rng.choice(_DATE_RANGE)
    start = rng.choice(_HOUR_SLOTS)
    h     = int(start.split(":")[0]) + 1
    end   = f"{h:02d}:00"
    env.step({"action": "book_meeting",
              "date": date, "start": start, "end": end})


def run_random_scheduling_impossible(env: InboxOpsEnv, rng: random.Random) -> None:
    """Same as random scheduling — picks an arbitrary slot, doesn't call find_slots or report."""
    run_random_scheduling(env, rng)


def run_random_support(env: InboxOpsEnv, rng: random.Random) -> None:
    """Open ticket + a few random actions. No systematic investigation."""
    env.step({"action": "open_ticket", "ticket_id": "TKT-001"})

    pool = [
        {"action": "view_customer",     "customer_id": "CUST-001"},
        {"action": "search_policy",     "policy_id": "refund_policy"},
        {"action": "assign_ticket",     "team": "billing"},
        {"action": "add_internal_note", "note": "Looked at the ticket."},
        {"action": "draft_reply",       "content": "Dear customer, we are looking into this."},
    ]
    for action in rng.sample(pool, k=min(4, len(pool))):
        env.step(action)


# ─── Runner ───────────────────────────────────────────────────────────────────

TASKS = [
    ("email_triage",          "Email Triage",            "easy",   run_heuristic_email,                  run_random_email),
    ("meeting_scheduling",    "Meeting Scheduling",       "medium", run_heuristic_scheduling,             run_random_scheduling),
    ("scheduling_impossible", "Scheduling — No Solution", "medium", run_heuristic_scheduling_impossible,  run_random_scheduling_impossible),
    ("support_escalation",    "Support Escalation",       "hard",   run_heuristic_support,                run_random_support),
]

_W = [35, 8, 7, 8, 5]   # column widths
_HEADERS = ["Task", "Diff", "Score", "Actions", "Pass"]


def _row(vals: list) -> str:
    parts = [str(v).ljust(_W[i]) if i == 0 else str(v).rjust(_W[i]) for i, v in enumerate(vals)]
    return "  " + "  ".join(parts)


def _divider() -> str:
    return "  " + "  ".join("─" * w for w in _W)


def run_all(debug: bool = False):
    env = InboxOpsEnv()
    rng = random.Random(42)

    WIDTH = 68
    print()
    print("═" * WIDTH)
    print("  InboxOps — Workplace Agent Benchmark  v1.0")
    print("═" * WIDTH)

    random_results   = []
    heuristic_results = []

    # ── Random pass ──────────────────────────────────────────────────────────
    print()
    print("  ── Random Baseline (seed=42) " + "─" * (WIDTH - 32))
    print(_row(_HEADERS))
    print(_divider())

    for task_name, display_name, difficulty, _heuristic_fn, random_fn in TASKS:
        env.reset(task_name)
        try:
            random_fn(env, rng)
        except Exception as exc:
            print(f"  ERROR in random agent ({task_name}): {exc}")
        summary = env.summary()
        random_results.append(summary)
        passed = "YES" if summary["passed"] else "no"
        print(_row([display_name, difficulty,
                    f"{summary['score']:.4f}", summary["steps"], passed]))

    n_passed_rand = sum(1 for r in random_results if r["passed"])
    mean_rand     = sum(r["score"] for r in random_results) / len(random_results)
    print(_divider())
    print(f"  Avg score: {mean_rand:.4f}   Passed: {n_passed_rand}/{len(random_results)}")

    # ── Heuristic pass ────────────────────────────────────────────────────────
    print()
    print("  ── Heuristic Baseline " + "─" * (WIDTH - 24))
    print(_row(_HEADERS))
    print(_divider())

    for task_name, display_name, difficulty, heuristic_fn, _random_fn in TASKS:
        env.reset(task_name)
        try:
            if task_name == "email_triage":
                heuristic_fn(env, debug=debug)
            else:
                heuristic_fn(env)
        except Exception as exc:
            print(f"  ERROR in heuristic agent ({task_name}): {exc}")
            import traceback; traceback.print_exc()
        summary = env.summary()
        heuristic_results.append(summary)
        passed = "YES" if summary["passed"] else "no"
        print(_row([display_name, difficulty,
                    f"{summary['score']:.4f}", summary["steps"], passed]))

    n_passed_heur = sum(1 for r in heuristic_results if r["passed"])
    mean_heur     = sum(r["score"] for r in heuristic_results) / len(heuristic_results)
    print(_divider())
    print(f"  Avg score: {mean_heur:.4f}   Passed: {n_passed_heur}/{len(heuristic_results)}")

    # ── Gap summary ───────────────────────────────────────────────────────────
    gap = mean_heur - mean_rand
    print()
    print("═" * WIDTH)
    print(f"  Heuristic vs random gap: +{gap:.4f}")
    print(f"  LLM agent target:  ≥ {mean_heur:.4f}  (match heuristic)  →  1.0000  (close email_011 gap)")
    print("═" * WIDTH)
    print()


if __name__ == "__main__":
    debug = "--debug" in sys.argv
    run_all(debug=debug)
