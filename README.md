---
title: InboxOps
emoji: 📬
colorFrom: indigo
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# InboxOps - Workplace Agent Benchmark

> **Can your agent handle a Monday morning inbox?**

InboxOps is a deterministic, hackathon-grade benchmark for evaluating AI agents on realistic office inbox tasks. Four tasks, four difficulty levels, one clean score. Pure Python -- no pip installs required to run.

---

## Motivation

Most agent benchmarks test either raw knowledge retrieval (Q&A, coding) or toy environments (grid worlds, text games). Real-world office work sits in between: agents must read messy natural-language inputs, apply business logic, route work to the right people, and know when to escalate.

InboxOps fills that gap. Every task reflects something that actually lands in a support queue, calendar, or ticket system -- with intentional edge cases designed to expose the difference between shallow pattern-matching and genuine reasoning.

---

## Tasks

| # | Task | Key | Difficulty | What the agent must do |
|---|------|-----|-----------|------------------------|
| 1 | **Email Triage** | `email_triage` | Easy | Classify 13 emails by label, priority, and owner; detect and archive phishing |
| 2 | **Meeting Scheduling** | `meeting_scheduling` | Medium | Find and book the best 60-min slot across 3 busy executive calendars |
| 3 | **Scheduling - No Solution** | `scheduling_impossible` | Medium | Recognise that no valid slot exists and report it rather than hallucinating one |
| 4 | **Support Escalation** | `support_escalation` | Hard | Investigate a VIP enterprise ticket with a duplicate charge + account lockout, route correctly, and escalate within the SLA window |

### What the agent sees

Each task exposes a **state dict** -- a structured snapshot of the environment:

```
Email Triage  -> {"inbox": [...], "labels": {...}, "priorities": {...}, ...}
Scheduling    -> {"scenario": {...}, "calendars": {...}, "slots": [...], ...}
Support       -> {"ticket": {...}, "customer": {...}, "policies": {...}, ...}
```

The agent takes **plain-dict actions** (`{"action": "label_email", "email_id": "email_011", "label": "spam"}`), receives a per-step reward, and the environment computes a **deterministic final score** (0.0-1.0) based on correctness alone.

---

## Scoring

### Email Triage

Each of the 13 emails is scored independently:

| Field | Weight |
|-------|--------|
| Label correct | 1/3 |
| Priority correct | 1/3 |
| Owner correct | 1/3 |

Spam emails carry an extra modifier: **+0.10** if archived, **-0.10** if left in inbox. Final score = mean across all emails.

**Ceiling note:** `email_011` is an intentional hard case -- a phishing email impersonating Stripe at `stripe-payment-security.net` whose body mentions "payment", "billing", "$599" (strong billing signals) but only one spam signal. A keyword classifier will misclassify it. An agent that checks the sender domain closes the gap.

### Meeting Scheduling

| Booked slot | Base score |
|-------------|------------|
| Best slot - Apr 10 09:00-10:00 (morning) | 1.00 |
| End-of-day slots - 16:00-17:00 | 0.70 |
| Lunch-overlap slot - Apr 11 12:30-13:30 | 0.55 |
| No booking or invalid slot | 0.00 |

**Process modifier:** +0.05 if all 3 calendars were viewed before booking; -0.05 otherwise.

### Scheduling - No Solution

| Outcome | Score |
|---------|-------|
| `report_no_solution` + all calendars viewed | 1.00 |
| `report_no_solution`, calendars not fully checked | 0.80 |
| `find_slots` called but no report | 0.40 |
| Nothing useful done | 0.10 |
| Tried to book (invalid slot) | 0.00 |

### Support Escalation

Checklist-based (13 items, weights sum to 1.0):

| Item | Weight |
|------|--------|
| Ticket opened | 0.05 |
| Customer profile viewed | 0.10 |
| Billing history inspected | 0.15 |
| Auth log checked | 0.10 |
| Refund policy consulted | 0.05 |
| Billing policy consulted | 0.05 |
| Escalation policy consulted | 0.05 |
| Security policy consulted | 0.05 |
| Assigned to billing team | 0.10 |
| Assigned to security team | 0.05 |
| Internal note added | 0.05 |
| Reply drafted | 0.10 |
| Escalated | 0.10 |

---

## Baseline Results

Measured on 2026-04-08.

### Random Baseline (seed=42 -- lower bound)

| Task | Difficulty | Score | Reward | Actions | Pass |
|------|-----------|------:|-------:|--------:|------|
| Email Triage | easy | 0.2154 | -- | 43 | NO |
| Meeting Scheduling | medium | 0.0000 | -- | 1 | NO |
| Scheduling - No Solution | medium | 0.1000 | -- | 1 | NO |
| Support Escalation | hard | 0.3500 | -- | 5 | NO |

### Heuristic Baseline (keyword classifier)

| Task | Difficulty | Score | Reward | Actions | Pass |
|------|-----------|------:|-------:|--------:|------|
| Email Triage | easy | **0.9231** | 2.62 | 54 | YES |
| Meeting Scheduling | medium | **1.0000** | 0.80 | 6 | YES |
| Scheduling - No Solution | medium | **1.0000** | 0.55 | 5 | YES |
| Support Escalation | hard | **1.0000** | 1.03 | 13 | YES |
| **Mean** | | **0.9808** | | | |

**Heuristic ceiling:** The heuristic is hand-tuned -- it knows the action space and applies domain rules, but still misses `email_011` (the fake-Stripe phishing email). An LLM agent with sender-domain reasoning should reach **1.0000** on email triage and match or exceed heuristic performance on all other tasks.

---

## Quick Start

```bash
# Clone and enter the project
git clone <repo-url>
cd InboxOps

# Run all 4 tasks with the heuristic baseline
python3 scripts/run_baseline.py

# Show per-email classification detail
python3 scripts/run_baseline.py --debug

# Run an LLM agent -- ENABLE_LLM_RUN must be set explicitly (see Cost Guard)
export ENABLE_LLM_RUN=true
export OPENAI_API_KEY=sk-...
python3 scripts/run_llm_agent.py

# Launch the Gradio demo locally (requires gradio)
pip install gradio
python3 app.py

# Deploy to Hugging Face Spaces -- push the repo, set SDK to Gradio
# Spaces will automatically run app.py
```

**Requirements:** Python 3.10+, stdlib only for the benchmark core. No pip installs needed to run the baseline.

### Cost Guard

`scripts/run_llm_agent.py` will **not** make any API calls unless `ENABLE_LLM_RUN=true` is explicitly set. If the variable is absent or any other value, the script prints a warning and exits with code 0 -- no tokens spent.

This prevents accidental charges when the script is imported, run in CI, or executed during a demo without intent to call the API. The heuristic baseline (`run_baseline.py`) never makes network calls and is always safe to run.

---

## API

```python
from env import InboxOpsEnv

env   = InboxOpsEnv()
state = env.reset("email_triage")            # one of the four task keys
state, reward, done, info = env.step(action) # action is a plain dict
score = env.grade()                          # float 0.0-1.0, deterministic
print(env.summary())                         # {score, reward, steps, passed}
```

### Action reference

```python
# Email triage
{"action": "open_email",    "email_id": "email_011"}
{"action": "label_email",   "email_id": "email_011", "label": "spam"}
{"action": "set_priority",  "email_id": "email_011", "priority": "low"}
{"action": "assign_owner",  "email_id": "email_011", "owner": "security"}
{"action": "archive_email", "email_id": "email_011"}

# Scheduling
{"action": "view_calendar",      "participant": "Alice Chen"}
{"action": "find_slots"}
{"action": "propose_meeting",    "date": "2026-04-10", "start": "09:00", "end": "10:00"}
{"action": "book_meeting",       "date": "2026-04-10", "start": "09:00", "end": "10:00"}
{"action": "report_no_solution", "reason": "No 60-min slot exists in the date range"}

# Support escalation
{"action": "open_ticket",       "ticket_id": "TKT-001"}
{"action": "view_customer",     "customer_id": "CUST-001"}
{"action": "inspect_billing",   "customer_id": "CUST-001"}
{"action": "check_auth_status", "customer_id": "CUST-001"}
{"action": "search_policy",     "policy_id": "escalation_policy"}
{"action": "assign_ticket",     "team": "billing"}
{"action": "add_internal_note", "note": "Duplicate charge ch_001/ch_002 confirmed."}
{"action": "draft_reply",       "content": "Dear Marcus, ..."}
{"action": "escalate",          "reason": "VIP customer, renewal in 3 days"}
```

---

## Repository Structure

```
InboxOps/
├── data/
│   ├── emails.json               # 13 emails - 6 label classes, 3 edge cases
│   ├── calendars.json            # 3 participants, 4 valid 60-min slots
│   ├── calendars_impossible.json # 3 participants, 0 valid slots
│   ├── tickets.json              # TKT-001: duplicate charge + lockout
│   ├── customers.json            # CUST-001: VIP enterprise, renewal in 3 days
│   └── policies.json             # refund / billing / escalation / security
├── env/
│   ├── environment.py            # InboxOpsEnv - unified API
│   ├── graders.py                # Final score 0.0-1.0 per task (deterministic)
│   ├── rewards.py                # Per-step reward shaping constants
│   └── tasks/
│       ├── email_easy.py
│       ├── scheduling_medium.py  # MeetingSchedulingTask + SchedulingImpossibleTask
│       └── support_hard.py
├── scripts/
│   ├── run_baseline.py           # Heuristic + random baseline runners
│   └── run_llm_agent.py          # OpenAI-compatible LLM agent harness
├── app/
│   └── space_app.py              # Gradio demo (Hugging Face Spaces)
├── app.py                        # HF Spaces entry point (imports app/space_app.py)
└── requirements.txt
```

---

## Why InboxOps Is Interesting

**Grounding in real work.** The tasks are modelled on actual office workflows, not toy puzzles. The phishing email looks exactly like a real Stripe notification. The calendar data reflects realistic executive schedules. The support ticket has the kind of multi-issue complexity that actually makes customer support hard.

**Deterministic scoring.** There is no judge model, no human rater, no ambiguity. The grader is a pure function of final state. You can run it 1000 times with the same agent and get the same score.

**Calibrated difficulty.** The random baseline scores near zero (chance performance). The heuristic baseline scores ~0.98. LLM agents should slot in between -- and the one intentional gap (`email_011`) gives a concrete, measurable improvement target.

**Reward shaping included.** The environment emits per-step rewards for learning (RL-compatible), but evaluation uses only the final grader score. Both are exposed so InboxOps can be used for both supervised evaluation and policy training.

---

## Future Work

- Add more task types: contract review, expense approval, onboarding checklist
- Multi-agent variant: inbox + calendar + ticket system as separate agents that must coordinate
- Adversarial emails: model-generated phishing that adapts to the evaluating agent's patterns
- Latency and cost tracking: score agents on quality-per-dollar, not just accuracy
