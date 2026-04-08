"""
Reward constants for all three tasks.

These values are used by the task step() methods to provide per-step
training signal. They are separate from the final grade (0.0–1.0),
which is computed by graders.py based on final state correctness.

Design philosophy:
- Positive rewards for correct, useful actions.
- Small positive for information gathering (opens, views).
- Larger positive for correct outcomes (correct label, valid booking).
- Negative for invalid params, no-ops, or policy violations.
- The grader (final score) is independent of cumulative reward.
"""

# ─── Task 1: Email Triage ────────────────────────────────────────────────────

EMAIL = {
    "open_email":             0.01,   # small positive for reading before acting
    "open_email_repeat":     -0.02,   # already opened, no-op penalty
    "correct_label":          0.10,   # label matches expected
    "incorrect_label":       -0.05,   # wrong label set
    "correct_to_wrong_label":-0.10,   # changed a correct label to wrong
    "correct_priority":       0.05,
    "incorrect_priority":    -0.03,
    "correct_owner":          0.05,
    "incorrect_owner":       -0.03,
    "noop_field":            -0.02,   # set same value again
    "archive_spam":           0.10,   # correctly archived phishing/spam
    "archive_important":     -0.20,   # archived a high/medium priority non-spam email
    "archive_non_spam":      -0.05,   # archived a low-priority legitimate email
    "archive_repeat":        -0.05,   # already archived
    "invalid_email_id":      -0.05,
    "invalid_label":         -0.10,
    "invalid_priority":      -0.10,
    "invalid_owner":         -0.10,
    "unknown_action":        -0.10,
}

# ─── Task 2: Meeting Scheduling ──────────────────────────────────────────────

SCHEDULING = {
    "view_calendar":          0.05,   # reward for gathering information
    "view_calendar_repeat":  -0.02,
    "find_slots_found":       0.10,   # find_slots returned at least one valid slot
    "find_slots_none":        0.00,   # neutral — no slots may be expected
    "valid_proposal":         0.15,   # proposed a slot that is actually valid
    "invalid_proposal":      -0.10,
    "book_best":              0.40,   # booked the preferred morning slot
    "book_valid":             0.25,   # booked a different valid slot
    "book_invalid":          -0.20,   # tried to book a slot with conflicts
    "book_already_done":     -0.10,   # meeting already booked
    "book_without_proposal": -0.05,   # booked without a prior propose_meeting call
    "unknown_participant":   -0.05,
    "unknown_action":        -0.10,
}

# ─── Task 3: Support Escalation ──────────────────────────────────────────────

SUPPORT = {
    "open_ticket":            0.05,
    "open_ticket_repeat":    -0.02,
    "view_customer":          0.10,   # reveals VIP status, renewal date
    "view_customer_repeat":  -0.02,
    "inspect_billing":        0.15,   # reveals duplicate charge evidence
    "inspect_billing_repeat":-0.02,
    "check_auth_status":      0.10,   # reveals account lockout
    "check_auth_repeat":     -0.02,
    "search_policy":          0.05,   # per relevant policy consulted
    "search_policy_repeat":  -0.02,
    "assign_billing":         0.10,   # correct: route billing dispute to billing team
    "assign_security":        0.05,   # correct: route auth issue to security team
    "assign_wrong_team":     -0.05,
    "assign_repeat":         -0.02,
    "add_internal_note":      0.03,
    "draft_reply":            0.10,   # drafted a reply to the customer
    "draft_reply_repeat":    -0.02,
    "escalate":               0.15,   # VIP policy requires escalation
    "escalate_repeat":       -0.05,
    "unknown_ticket":        -0.05,
    "unknown_customer":      -0.05,
    "unknown_policy":        -0.05,
    "unknown_action":        -0.10,
}
