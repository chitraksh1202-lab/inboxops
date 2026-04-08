"""
Graders – deterministic final scoring for each task.

Each grader takes the task object (after the episode ends) and returns a
float in [0.0, 1.0]. Grading is based on final state correctness only —
NOT on cumulative reward or which specific actions were taken.

Reward (rewards.py) shapes step-level feedback for learning.
Grade (graders.py) provides the clean benchmark comparison metric.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from env.tasks.email_easy        import EmailTriageTask
    from env.tasks.scheduling_medium import MeetingSchedulingTask, SchedulingImpossibleTask
    from env.tasks.support_hard      import SupportEscalationTask


# ─── Task 1: Email Triage ─────────────────────────────────────────────────────

def grade_email(task: "EmailTriageTask") -> float:
    """
    Per-email correctness score averaged across the inbox.

    Per email (1/3 each field):
      label correct    → +0.333
      priority correct → +0.333
      owner correct    → +0.333

    Spam-handling modifier (applied after field scoring):
      spam email archived     → +0.10 bonus (cap 1.0)
      spam email NOT archived → −0.10 penalty (floor 0.0)

    Final score = mean across all emails.
    """
    scores = []
    for email in task.emails:
        eid   = email["id"]
        score = 0.0

        if task.labels.get(eid)     == email["expected_label"]:    score += 1 / 3
        if task.priorities.get(eid) == email["expected_priority"]: score += 1 / 3
        if task.owners.get(eid)     == email["expected_owner"]:    score += 1 / 3

        if email["expected_label"] == "spam":
            if eid in task.archived:
                score = min(score + 0.10, 1.0)   # bonus for archiving spam
            else:
                score = max(score - 0.10, 0.0)   # penalty for missing it

        scores.append(score)

    return round(sum(scores) / len(scores), 4) if scores else 0.0


# ─── Task 2: Meeting Scheduling (standard) ────────────────────────────────────

_SCHEDULING_VALID_SLOTS = [
    {"date": "2026-04-09", "start": "16:00", "end": "17:00"},
    {"date": "2026-04-10", "start": "09:00", "end": "10:00"},
    {"date": "2026-04-10", "start": "16:00", "end": "17:00"},
    {"date": "2026-04-11", "start": "12:30", "end": "13:30"},
]
_BEST_SLOT         = {"date": "2026-04-10", "start": "09:00", "end": "10:00"}
_LUNCH_WINDOW_SLOT = {"date": "2026-04-11", "start": "12:30", "end": "13:30"}


def _sm(a, b) -> bool:
    """Slot match helper."""
    return (
        a and b and
        a.get("date")  == b.get("date")  and
        a.get("start") == b.get("start") and
        a.get("end")   == b.get("end")
    )


def _grade_scheduling_standard(task: "MeetingSchedulingTask") -> float:
    """
    Score based on which slot was booked and process quality.

    Slot quality:
      No booking                          → 0.00
      Invalid slot (shouldn't reach here) → 0.00
      Lunch-overlap slot (Apr 11 12:30)   → 0.55
      End-of-day slots (16:00–17:00)      → 0.70
      Best slot (Apr 10 09:00)            → 1.00

    Calendar review modifier (±0.05):
      All 3 calendars viewed before book  → +0.05 (cap 1.0)
      Booked without viewing calendars    → −0.05 (floor 0.0)
    """
    booked = task.booked_meeting
    if not booked:
        return 0.0

    is_valid = any(_sm(booked, vs) for vs in _SCHEDULING_VALID_SLOTS)
    if not is_valid:
        return 0.0

    # Base score by slot quality
    if _sm(booked, _BEST_SLOT):
        score = 1.00
    elif _sm(booked, _LUNCH_WINDOW_SLOT):
        score = 0.55
    else:
        score = 0.70  # end-of-day slots

    # Process bonus/penalty: did the agent check all calendars first?
    required = set(task.scenario["required_attendees"])
    if required.issubset(task.viewed_calendars):
        score = min(score + 0.05, 1.0)
    else:
        score = max(score - 0.05, 0.0)

    return round(score, 4)


# ─── Task 2: Meeting Scheduling (impossible / no-solution) ────────────────────

def _grade_scheduling_impossible(task: "SchedulingImpossibleTask") -> float:
    """
    Score for the no-solution scheduling scenario.

    The correct behaviour is:
      1. view_calendar for all required attendees
      2. find_slots → observe empty result
      3. report_no_solution

    Scoring matrix:
      booked_meeting is not None            → 0.00  (wrong: booked something invalid or hallucinated)
      report_no_solution + all cals viewed  → 1.00  (perfect: thorough + correct conclusion)
      report_no_solution, cals not all seen → 0.80  (correct conclusion, skipped due diligence)
      find_slots called, no report          → 0.40  (found the evidence, didn't conclude)
      neither find_slots nor report         → 0.10  (did nothing useful)
    """
    # Verify the scenario is truly impossible (safeguard against data changes)
    actual_slots = task._compute_valid_slots()
    truly_impossible = len(actual_slots) == 0

    if not truly_impossible:
        # Data changed — grade as standard task
        return _grade_scheduling_standard(task)

    if task.booked_meeting is not None:
        return 0.0  # tried to book when there was nothing to book

    required = set(task.scenario["required_attendees"])
    all_cals_viewed = required.issubset(task.viewed_calendars)

    if task.report_no_solution_called:
        return 1.00 if all_cals_viewed else 0.80

    if task.find_slots_called:
        return 0.40  # found no slots but didn't explicitly report it

    return 0.10  # did nothing useful


def grade_scheduling(task) -> float:
    """Dispatch to the correct scheduling grader by task name."""
    if task.name == "scheduling_impossible":
        return _grade_scheduling_impossible(task)
    return _grade_scheduling_standard(task)


# ─── Task 3: Support Escalation ───────────────────────────────────────────────

# Checklist weights — must sum to 1.0
_SUPPORT_CHECKLIST_WEIGHTS = {
    "ticket_opened":               0.05,
    "customer_viewed":             0.10,
    "billing_inspected":           0.15,
    "auth_checked":                0.10,
    "refund_policy_consulted":     0.05,
    "billing_policy_consulted":    0.05,
    "escalation_policy_consulted": 0.05,
    "security_policy_consulted":   0.05,
    "billing_assigned":            0.10,
    "security_assigned":           0.05,
    "internal_note_added":         0.05,
    "reply_drafted":               0.10,
    "escalated":                   0.10,
}
# Total: 1.00


def grade_support(task: "SupportEscalationTask") -> float:
    """
    Checklist-based grading.
    Each item is binary: satisfied → weight, unsatisfied → 0.
    Final score = sum of satisfied weights.
    """
    checklist = task.grade_checklist()
    score = sum(
        _SUPPORT_CHECKLIST_WEIGHTS[item] * (1 if satisfied else 0)
        for item, satisfied in checklist.items()
    )
    return round(score, 4)


# ─── Unified entry point ──────────────────────────────────────────────────────

def grade(task_name: str, task) -> float:
    """Grade any task by name. Called by InboxOpsEnv.grade()."""
    if task_name == "email_triage":
        return grade_email(task)
    if task_name in ("meeting_scheduling", "scheduling_impossible"):
        return grade_scheduling(task)
    if task_name == "support_escalation":
        return grade_support(task)
    raise ValueError(f"Unknown task name: {task_name!r}")
