"""
Task 1 – Email Triage (Easy)
=============================
The agent receives an inbox of 10 emails and must:
  1. Open each email to read its content.
  2. Label it (billing / support / meeting / sales / spam / internal).
  3. Set its priority (low / medium / high).
  4. Assign an owner team (finance / support / exec / sales / security / hr).
  5. Archive any spam or phishing emails.

The task is "done" when every email has a label, priority, and owner,
and all spam emails have been archived.

Grading is handled by graders.py (final correctness score).
Reward shaping is applied per action (see rewards.py).
"""

import json
from pathlib import Path
from env.rewards import EMAIL as R

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "emails.json"

VALID_LABELS    = {"billing", "support", "meeting", "sales", "spam", "internal"}
VALID_PRIORITIES = {"low", "medium", "high"}
VALID_OWNERS    = {"finance", "support", "exec", "sales", "security", "hr"}

MAX_STEPS = 80  # 10 emails × ~8 actions max


class EmailTriageTask:
    name       = "email_triage"
    difficulty = "easy"

    def __init__(self):
        with open(DATA_PATH, encoding="utf-8") as f:
            self.emails = json.load(f)
        self._email_map = {e["id"]: e for e in self.emails}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> dict:
        self.opened     : set  = set()
        self.labels     : dict = {}
        self.priorities : dict = {}
        self.owners     : dict = {}
        self.archived   : set  = set()
        self.action_log : list = []
        self.step_count : int  = 0
        self.done       : bool = False
        return self.state()

    def state(self) -> dict:
        """
        Returns current observable state.
        Unopened emails show only id/sender/subject (realistic inbox view).
        Opened emails expose the full body.
        """
        inbox = []
        for e in self.emails:
            if e["id"] in self.opened:
                # Full email visible after opening
                inbox.append({
                    "id":      e["id"],
                    "sender":  e["sender"],
                    "subject": e["subject"],
                    "body":    e["body"],
                })
            else:
                # Unopened: preview only
                inbox.append({
                    "id":      e["id"],
                    "sender":  e["sender"],
                    "subject": e["subject"],
                })
        return {
            "task":       self.name,
            "inbox":      inbox,
            "labels":     dict(self.labels),
            "priorities": dict(self.priorities),
            "owners":     dict(self.owners),
            "archived":   sorted(self.archived),
            "action_log": self.action_log[-15:],  # last 15 for readability
            "step_count": self.step_count,
            "done":       self.done,
        }

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action: dict) -> tuple[dict, float, bool, dict]:
        """Returns (state, reward, done, info)."""
        self.step_count += 1
        reward = 0.0
        info   = {}
        act    = action.get("action")

        if act == "open_email":
            reward, info = self._open_email(action)

        elif act == "label_email":
            reward, info = self._label_email(action)

        elif act == "set_priority":
            reward, info = self._set_priority(action)

        elif act == "assign_owner":
            reward, info = self._assign_owner(action)

        elif act == "archive_email":
            reward, info = self._archive_email(action)

        else:
            reward = R["unknown_action"]
            info   = {"error": f"Unknown action '{act}'. Valid: open_email, label_email, set_priority, assign_owner, archive_email"}

        self.action_log.append({
            "step":   self.step_count,
            "action": action,
            "reward": round(reward, 4),
            "info":   info,
        })

        if self._is_complete():
            self.done = True
        if self.step_count >= MAX_STEPS:
            self.done = True

        return self.state(), reward, self.done, info

    # ── Action handlers ───────────────────────────────────────────────────────

    def _open_email(self, action: dict) -> tuple[float, dict]:
        eid = action.get("email_id")
        if eid not in self._email_map:
            return R["invalid_email_id"], {"error": f"Unknown email_id: {eid!r}"}
        if eid in self.opened:
            return R["open_email_repeat"], {"note": "already opened"}
        self.opened.add(eid)
        return R["open_email"], {"result": "opened", "email_id": eid}

    def _label_email(self, action: dict) -> tuple[float, dict]:
        eid   = action.get("email_id")
        label = action.get("label")
        if eid not in self._email_map:
            return R["invalid_email_id"], {"error": f"Unknown email_id: {eid!r}"}
        if label not in VALID_LABELS:
            return R["invalid_label"], {"error": f"Invalid label {label!r}. Valid: {sorted(VALID_LABELS)}"}
        prev     = self.labels.get(eid)
        expected = self._email_map[eid]["expected_label"]
        if prev == label:
            return R["noop_field"], {"note": f"label already set to {label!r}"}
        self.labels[eid] = label
        if label == expected:
            return R["correct_label"], {"result": "correct_label", "label": label}
        if prev == expected:
            # Agent changed a correct label to a wrong one — penalise more
            return R["correct_to_wrong_label"], {"result": "label_made_worse", "was": prev, "now": label}
        return R["incorrect_label"], {"result": "incorrect_label", "expected": expected, "got": label}

    def _set_priority(self, action: dict) -> tuple[float, dict]:
        eid      = action.get("email_id")
        priority = action.get("priority")
        if eid not in self._email_map:
            return R["invalid_email_id"], {"error": f"Unknown email_id: {eid!r}"}
        if priority not in VALID_PRIORITIES:
            return R["invalid_priority"], {"error": f"Invalid priority {priority!r}. Valid: {sorted(VALID_PRIORITIES)}"}
        if self.priorities.get(eid) == priority:
            return R["noop_field"], {"note": f"priority already set to {priority!r}"}
        self.priorities[eid] = priority
        expected = self._email_map[eid]["expected_priority"]
        if priority == expected:
            return R["correct_priority"], {"result": "correct_priority"}
        return R["incorrect_priority"], {"result": "incorrect_priority", "expected": expected, "got": priority}

    def _assign_owner(self, action: dict) -> tuple[float, dict]:
        eid   = action.get("email_id")
        owner = action.get("owner")
        if eid not in self._email_map:
            return R["invalid_email_id"], {"error": f"Unknown email_id: {eid!r}"}
        if owner not in VALID_OWNERS:
            return R["invalid_owner"], {"error": f"Invalid owner {owner!r}. Valid: {sorted(VALID_OWNERS)}"}
        if self.owners.get(eid) == owner:
            return R["noop_field"], {"note": f"owner already set to {owner!r}"}
        self.owners[eid] = owner
        expected = self._email_map[eid]["expected_owner"]
        if owner == expected:
            return R["correct_owner"], {"result": "correct_owner"}
        return R["incorrect_owner"], {"result": "incorrect_owner", "expected": expected, "got": owner}

    def _archive_email(self, action: dict) -> tuple[float, dict]:
        eid = action.get("email_id")
        if eid not in self._email_map:
            return R["invalid_email_id"], {"error": f"Unknown email_id: {eid!r}"}
        if eid in self.archived:
            return R["archive_repeat"], {"note": "already archived"}
        self.archived.add(eid)
        email = self._email_map[eid]
        if email["expected_label"] == "spam":
            return R["archive_spam"], {"result": "correctly_archived_spam"}
        if email["expected_priority"] in ("high", "medium"):
            return R["archive_important"], {"result": "archived_important_email_penalty", "priority": email["expected_priority"]}
        return R["archive_non_spam"], {"result": "archived_non_spam_legitimate_email"}

    # ── Completion check ──────────────────────────────────────────────────────

    def _is_complete(self) -> bool:
        all_ids  = set(self._email_map.keys())
        spam_ids = {e["id"] for e in self.emails if e["expected_label"] == "spam"}
        return (
            set(self.labels.keys())     == all_ids and
            set(self.priorities.keys()) == all_ids and
            set(self.owners.keys())     == all_ids and
            spam_ids.issubset(self.archived)
        )
