# InboxOps – Agent Benchmark

A deterministic, hackathon-grade benchmark for evaluating AI agents on realistic office inbox tasks. Pure Python, no external dependencies.

## Tasks

| Task | Key | Difficulty | Description |
|------|-----|-----------|-------------|
| Email Triage | `email_triage` | Easy | Label, prioritise, and route 13 emails; archive phishing |
| Meeting Scheduling | `meeting_scheduling` | Medium | Find and book a valid 60-min slot across 3 busy calendars |
| Scheduling – No Solution | `scheduling_impossible` | Medium | Recognise that no valid slot exists and report it |
| Support Escalation | `support_escalation` | Hard | Investigate a VIP enterprise ticket with 4 sub-issues |

## Structure

```
InboxOps/
├── data/
│   ├── emails.json               # 13 emails, 6 label classes, 3 edge cases
│   ├── calendars.json            # 3 participants, 4 valid 60-min slots
│   ├── calendars_impossible.json # 3 participants, 0 valid slots
│   ├── tickets.json              # TKT-001: duplicate charge + lockout
│   ├── customers.json            # CUST-001: VIP enterprise, renewal in 3 days
│   └── policies.json             # refund / billing / escalation / security
├── env/
│   ├── environment.py            # InboxOpsEnv – unified API
│   ├── graders.py                # Final score 0.0–1.0 per task (deterministic)
│   ├── rewards.py                # Per-step reward constants
│   └── tasks/
│       ├── email_easy.py
│       ├── scheduling_medium.py  # MeetingSchedulingTask + SchedulingImpossibleTask
│       └── support_hard.py
└── scripts/
    └── run_baseline.py           # Heuristic baseline runner
```

## Quick start

```bash
python3 scripts/run_baseline.py           # all 4 tasks
python3 scripts/run_baseline.py --debug   # per-email classification detail
```

## API

```python
from env import InboxOpsEnv

env   = InboxOpsEnv()
state = env.reset("email_triage")            # one of the four task keys above
state, reward, done, info = env.step(action) # action is a plain dict
score = env.grade()                          # float 0.0–1.0, deterministic
print(env.summary())                         # score, reward, steps, passed
```

### Action examples

```python
# Email triage
{"action": "open_email",   "email_id": "email_011"}
{"action": "label_email",  "email_id": "email_011", "label": "spam"}
{"action": "set_priority", "email_id": "email_011", "priority": "low"}
{"action": "assign_owner", "email_id": "email_011", "owner": "security"}
{"action": "archive_email","email_id": "email_011"}

# Scheduling
{"action": "view_calendar",     "participant": "Alice Chen"}
{"action": "find_slots"}
{"action": "propose_meeting",   "date": "2026-04-10", "start": "09:00", "end": "10:00"}
{"action": "book_meeting",      "date": "2026-04-10", "start": "09:00", "end": "10:00"}
{"action": "report_no_solution","reason": "No 60-min slot exists in range"}

# Support escalation
{"action": "open_ticket",      "ticket_id": "TKT-001"}
{"action": "view_customer",    "customer_id": "CUST-001"}
{"action": "inspect_billing",  "customer_id": "CUST-001"}
{"action": "check_auth_status","customer_id": "CUST-001"}
{"action": "search_policy",    "policy_id": "escalation_policy"}
{"action": "assign_ticket",    "team": "billing"}
{"action": "add_internal_note","note": "Duplicate charge ch_001/ch_002 confirmed."}
{"action": "draft_reply",      "content": "Dear Marcus, ..."}
{"action": "escalate",         "reason": "VIP customer, renewal in 3 days"}
```

## Heuristic baseline scores

Measured on 2026-04-08 with the keyword-based heuristic in `scripts/run_baseline.py`.

| Task | Score | Reward | Steps | Pass |
|------|------:|-------:|------:|------|
| Email Triage (easy) | **0.9231** | 2.62 | 54 | YES |
| Meeting Scheduling (medium) | **1.0000** | 0.80 | 6 | YES |
| Scheduling – No Solution (medium) | **1.0000** | 0.55 | 5 | YES |
| Support Escalation (hard) | **1.0000** | 1.03 | 13 | YES |
| **Mean** | **0.9808** | | | |

### Notable scores / ceiling analysis

**Email Triage 0.9231 (12/13)**
The one miss is `email_011`: a phishing email impersonating Stripe that mentions "payment", "billing", "$599.00". A naive keyword classifier scores it as billing (2 hits) over spam (1 hit for "account will be suspended"). This gap is intentional — it demonstrates that domain-aware reasoning (checking sender domain `stripe-payment-security.net` ≠ `stripe.com`) is required for a perfect score. An LLM agent should close this gap.

**Scheduling – No Solution 1.0000**
The agent correctly calls `find_slots`, observes an empty result, and calls `report_no_solution` with a reasoned explanation. If it skipped the calendar-view step it would score 0.80; if it tried to book an invalid slot it would score 0.00.

**Support Escalation 1.0000**
The heuristic is hand-coded to follow the exact expected checklist. A real LLM agent would need to infer the same steps from the ticket and customer data.

## Grading reference

### Email Triage
Per email: label (1/3) + priority (1/3) + owner (1/3). Spam archived → +0.10 bonus (cap 1.0). Spam missed → −0.10 penalty. Final = mean.

### Meeting Scheduling (standard)
| Booked slot | Base score |
|---|---|
| Best slot (Apr 10 09:00–10:00) | 1.00 |
| End-of-day slots (16:00–17:00) | 0.70 |
| Lunch-overlap slot (Apr 11 12:30) | 0.55 |
| No booking / invalid | 0.00 |
+0.05 if all 3 calendars viewed before booking; −0.05 if not.

### Scheduling – No Solution
| Outcome | Score |
|---|---|
| report_no_solution + all calendars viewed | 1.00 |
| report_no_solution, skipped calendar check | 0.80 |
| find_slots called, no report | 0.40 |
| Nothing useful done | 0.10 |
| Tried to book invalid slot | 0.00 |

### Support Escalation
Checklist-based (13 items summing to 1.0): ticket opened (0.05), customer viewed (0.10), billing inspected (0.15), auth checked (0.10), 4× policy consulted (0.20 total), billing assigned (0.10), security assigned (0.05), internal note (0.05), reply drafted (0.10), escalated (0.10).

## Requirements

Python 3.10+ — stdlib only, no pip installs needed.
