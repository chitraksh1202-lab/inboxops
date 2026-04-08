"""
Task 3 – Support Escalation (Hard)
=====================================
The agent handles a complex enterprise support case involving:
  • A duplicate charge ($2,000 × 2 on the same day)
  • An account lockout (5 failed login attempts)
  • A VIP / enterprise customer with a renewal in 3 days
  • Multiple policy constraints (refund, security, escalation)

Expected action flow:
  1. open_ticket       → read the customer complaint
  2. view_customer     → discover VIP status, renewal date, account manager
  3. inspect_billing   → confirm duplicate charges (ch_001, ch_002)
  4. check_auth_status → confirm account lockout (5 failed attempts)
  5. search_policy     → consult refund, billing, escalation, security policies
  6. assign_ticket     → route to billing AND security teams
  7. add_internal_note → document findings with charge IDs and auth details
  8. draft_reply       → send a non-committal holding reply to the customer
  9. escalate          → required by VIP escalation policy

Policy constraints the agent must not violate:
  - Do NOT promise immediate account unlock (requires security team).
  - Do NOT promise refund outside policy without referencing duplicate exception.
  - Must escalate VIP cases (per escalation_policy).

Grading: checklist-based score (see graders.py).
"""

import json
from pathlib import Path
from env.rewards import SUPPORT as R

DATA_DIR = Path(__file__).parent.parent.parent / "data"

MAX_STEPS = 40

VALID_TEAMS  = {"billing", "support", "security", "exec", "sales", "hr"}
KNOWN_TICKET = "TKT-001"
KNOWN_CUSTOMER = "CUST-001"
KNOWN_POLICIES = {"refund_policy", "billing_policy", "escalation_policy", "security_policy"}


class SupportEscalationTask:
    name       = "support_escalation"
    difficulty = "hard"

    def __init__(self):
        with open(DATA_DIR / "tickets.json",   encoding="utf-8") as f: self._tickets   = {t["id"]: t for t in json.load(f)}
        with open(DATA_DIR / "customers.json", encoding="utf-8") as f: self._customers = {c["id"]: c for c in json.load(f)}
        with open(DATA_DIR / "policies.json",  encoding="utf-8") as f: self._policies  = {p["id"]: p for p in json.load(f)}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> dict:
        # What the agent has discovered so far
        self.ticket_opened      : bool  = False
        self.current_ticket     : dict  = None
        self.customer_viewed    : bool  = False
        self.customer_record    : dict  = None
        self.billing_inspected  : bool  = False
        self.billing_record     : list  = None  # billing_history list
        self.auth_checked       : bool  = False
        self.auth_record        : list  = None  # auth_log list
        self.policies_seen      : set   = set()
        self.assignments        : list  = []    # teams assigned
        self.internal_notes     : list  = []
        self.draft_reply        : str   = None
        self.escalated          : bool  = False
        self.escalation_reason  : str   = None
        self.action_log         : list  = []
        self.step_count         : int   = 0
        self.done               : bool  = False
        return self.state()

    def state(self) -> dict:
        return {
            "task":                 self.name,
            "ticket":               self.current_ticket,
            "customer_record":      self._safe_customer_view(),
            "billing_record":       self.billing_record,
            "auth_record":          self.auth_record,
            "policies_seen":        {pid: self._policies[pid] for pid in self.policies_seen},
            "assignments":          list(self.assignments),
            "internal_notes":       list(self.internal_notes),
            "draft_reply":          self.draft_reply,
            "escalated":            self.escalated,
            "escalation_reason":    self.escalation_reason,
            "action_log":           self.action_log[-15:],
            "step_count":           self.step_count,
            "done":                 self.done,
            # Hint so the agent knows what tickets exist
            "available_tickets":    list(self._tickets.keys()),
        }

    def _safe_customer_view(self) -> dict:
        """Return customer record without sensitive billing/auth details
        unless the agent has explicitly inspected them."""
        if not self.customer_record:
            return None
        view = {k: v for k, v in self.customer_record.items()
                if k not in ("billing_history", "auth_log")}
        if self.billing_inspected:
            view["billing_history"] = self.billing_record
        if self.auth_checked:
            view["auth_log"] = self.auth_record
        return view

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action: dict) -> tuple[dict, float, bool, dict]:
        self.step_count += 1
        reward = 0.0
        info   = {}
        act    = action.get("action")

        if act == "open_ticket":
            reward, info = self._open_ticket(action)
        elif act == "view_customer":
            reward, info = self._view_customer(action)
        elif act == "inspect_billing":
            reward, info = self._inspect_billing(action)
        elif act == "check_auth_status":
            reward, info = self._check_auth_status(action)
        elif act == "search_policy":
            reward, info = self._search_policy(action)
        elif act == "assign_ticket":
            reward, info = self._assign_ticket(action)
        elif act == "add_internal_note":
            reward, info = self._add_internal_note(action)
        elif act == "draft_reply":
            reward, info = self._draft_reply(action)
        elif act == "escalate":
            reward, info = self._escalate(action)
        else:
            reward = R["unknown_action"]
            info   = {
                "error": f"Unknown action '{act}'.",
                "valid_actions": [
                    "open_ticket", "view_customer", "inspect_billing",
                    "check_auth_status", "search_policy", "assign_ticket",
                    "add_internal_note", "draft_reply", "escalate",
                ],
            }

        self.action_log.append({
            "step":   self.step_count,
            "action": action,
            "reward": round(reward, 4),
            "info":   info,
        })

        if self.step_count >= MAX_STEPS:
            self.done = True

        return self.state(), reward, self.done, info

    # ── Action handlers ───────────────────────────────────────────────────────

    def _open_ticket(self, action: dict) -> tuple[float, dict]:
        tid = action.get("ticket_id")
        if tid not in self._tickets:
            return R["unknown_ticket"], {"error": f"Unknown ticket {tid!r}. Available: {list(self._tickets.keys())}"}
        if self.ticket_opened:
            return R["open_ticket_repeat"], {"note": "ticket already opened"}
        self.ticket_opened  = True
        self.current_ticket = self._tickets[tid]
        return R["open_ticket"], {"result": "ticket_loaded", "ticket": self.current_ticket}

    def _view_customer(self, action: dict) -> tuple[float, dict]:
        cid = action.get("customer_id")
        if cid not in self._customers:
            return R["unknown_customer"], {"error": f"Unknown customer {cid!r}"}
        if self.customer_viewed:
            return R["view_customer_repeat"], {"note": "customer already viewed"}
        self.customer_viewed = True
        self.customer_record = self._customers[cid]
        # Strip raw billing/auth — agent must call dedicated actions for those
        summary = {k: v for k, v in self.customer_record.items()
                   if k not in ("billing_history", "auth_log")}
        return R["view_customer"], {"result": "customer_loaded", "customer_summary": summary}

    def _inspect_billing(self, action: dict) -> tuple[float, dict]:
        cid = action.get("customer_id")
        if cid not in self._customers:
            return R["unknown_customer"], {"error": f"Unknown customer {cid!r}"}
        if self.billing_inspected:
            return R["inspect_billing_repeat"], {"note": "billing already inspected"}
        self.billing_inspected = True
        self.billing_record    = self._customers[cid]["billing_history"]
        # Highlight the duplicate to make the evidence clear
        duplicates = [b for b in self.billing_record if "_note" in b and "DUPLICATE" in b.get("_note", "")]
        return R["inspect_billing"], {
            "result":     "billing_history_loaded",
            "records":    self.billing_record,
            "duplicates_found": len(duplicates),
            "note":       "Duplicate charge detected: ch_001 and ch_002 are identical ($2,000 on 2026-04-01)" if duplicates else None,
        }

    def _check_auth_status(self, action: dict) -> tuple[float, dict]:
        cid = action.get("customer_id")
        if cid not in self._customers:
            return R["unknown_customer"], {"error": f"Unknown customer {cid!r}"}
        if self.auth_checked:
            return R["check_auth_repeat"], {"note": "auth status already checked"}
        self.auth_checked = True
        self.auth_record  = self._customers[cid]["auth_log"]
        lockout = any(e["event"] == "account_locked" for e in self.auth_record)
        return R["check_auth_status"], {
            "result":          "auth_log_loaded",
            "auth_log":        self.auth_record,
            "account_locked":  lockout,
            "note":            "Account locked after 5 failed login attempts on 2026-04-08T08:03Z" if lockout else "No lockout found",
        }

    def _search_policy(self, action: dict) -> tuple[float, dict]:
        pid = action.get("policy_id")
        if pid not in self._policies:
            return R["unknown_policy"], {
                "error": f"Unknown policy {pid!r}. Available: {sorted(self._policies.keys())}"
            }
        if pid in self.policies_seen:
            return R["search_policy_repeat"], {"note": f"policy {pid!r} already retrieved"}
        self.policies_seen.add(pid)
        return R["search_policy"], {
            "result": "policy_loaded",
            "policy": self._policies[pid],
        }

    def _assign_ticket(self, action: dict) -> tuple[float, dict]:
        team = action.get("team")
        if team not in VALID_TEAMS:
            return R["unknown_action"], {"error": f"Invalid team {team!r}. Valid: {sorted(VALID_TEAMS)}"}
        if team in self.assignments:
            return R["assign_repeat"], {"note": f"ticket already assigned to {team!r}"}
        self.assignments.append(team)
        # Billing and security are the correct teams for this case
        if team == "billing":
            return R["assign_billing"], {"result": f"assigned to {team}"}
        if team == "security":
            return R["assign_security"], {"result": f"assigned to {team}"}
        return R["assign_wrong_team"], {"result": f"assigned to {team} (unexpected for this case)"}

    def _add_internal_note(self, action: dict) -> tuple[float, dict]:
        note = action.get("note", "").strip()
        if not note:
            return 0.0, {"error": "note content is empty"}
        self.internal_notes.append(note)
        return R["add_internal_note"], {"result": "note_added", "note_preview": note[:100]}

    def _draft_reply(self, action: dict) -> tuple[float, dict]:
        content = action.get("content", "").strip()
        if not content:
            return 0.0, {"error": "draft content is empty"}
        if self.draft_reply is not None:
            return R["draft_reply_repeat"], {"note": "reply already drafted (overwriting)"}
        self.draft_reply = content
        return R["draft_reply"], {"result": "reply_drafted", "preview": content[:120]}

    def _escalate(self, action: dict) -> tuple[float, dict]:
        reason = action.get("reason", "").strip()
        if self.escalated:
            return R["escalate_repeat"], {"note": "already escalated"}
        self.escalated         = True
        self.escalation_reason = reason
        self.done              = True
        return R["escalate"], {
            "result": "escalated",
            "reason": reason,
            "note":   "Escalation recorded. Account manager Sarah Okafor will be notified.",
        }

    # ── Grade-support helper ───────────────────────────────────────────────────

    def grade_checklist(self) -> dict:
        """
        Returns a dict of checklist items and whether each was satisfied.
        Called by graders.grade_support() to compute the final score.
        """
        return {
            "ticket_opened":              self.ticket_opened,
            "customer_viewed":            self.customer_viewed,
            "billing_inspected":          self.billing_inspected,
            "auth_checked":               self.auth_checked,
            "refund_policy_consulted":    "refund_policy"    in self.policies_seen,
            "billing_policy_consulted":   "billing_policy"   in self.policies_seen,
            "escalation_policy_consulted":"escalation_policy" in self.policies_seen,
            "security_policy_consulted":  "security_policy"  in self.policies_seen,
            "billing_assigned":           "billing"  in self.assignments,
            "security_assigned":          "security" in self.assignments,
            "internal_note_added":        len(self.internal_notes) > 0,
            "reply_drafted":              self.draft_reply is not None,
            "escalated":                  self.escalated,
        }
