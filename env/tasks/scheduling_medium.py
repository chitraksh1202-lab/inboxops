"""
Task 2 – Meeting Scheduling (Medium)
=====================================
Two variants share this file:

  MeetingSchedulingTask  (name="meeting_scheduling")
    Standard case: a valid 60-min slot exists. Agent must find and book it.
    Valid slots (precomputed):
      • 2026-04-09 16:00–17:00  – end of day, not preferred
      • 2026-04-10 09:00–10:00  – ★ BEST (morning, earliest)
      • 2026-04-10 16:00–17:00  – end of day
      • 2026-04-11 12:30–13:30  – overlaps lunch window, least preferred

  SchedulingImpossibleTask  (name="scheduling_impossible")
    No-solution case: calendars are fully blocked with no mutual free window.
    Agent must call find_slots, observe [], then call report_no_solution.
    Attempting to book an invalid slot is penalised.

Expected action flow (standard):
  view_calendar × 3 → find_slots → propose_meeting → book_meeting

Expected action flow (impossible):
  view_calendar × 3 → find_slots → report_no_solution

Grading: graders.py. Standard variant scores on slot quality; impossible
variant scores on whether the agent correctly acknowledged no solution exists.
"""

import json
from datetime import date, timedelta
from pathlib import Path
from env.rewards import SCHEDULING as R

_DATA_DIR = Path(__file__).parent.parent.parent / "data"

MAX_STEPS = 25


class MeetingSchedulingTask:
    name       = "meeting_scheduling"
    difficulty = "medium"
    # Subclass overrides this to load a different scenario
    _DATA_PATH = _DATA_DIR / "calendars.json"

    # Precomputed valid slots for the standard scenario.
    # Used for fast validation — recomputed dynamically for impossible variant.
    # Verified by running _compute_valid_slots() against calendars.json.
    _STATIC_VALID_SLOTS = [
        {"date": "2026-04-09", "start": "16:00", "end": "17:00"},
        {"date": "2026-04-10", "start": "09:00", "end": "10:00"},
        {"date": "2026-04-10", "start": "16:00", "end": "17:00"},
        {"date": "2026-04-11", "start": "12:30", "end": "13:30"},
    ]
    _BEST_SLOT = {"date": "2026-04-10", "start": "09:00", "end": "10:00"}

    def __init__(self):
        with open(self._DATA_PATH, encoding="utf-8") as f:
            data = json.load(f)
        self.participants     = data["participants"]
        self.scenario         = data["scenario"]
        self._participant_map = {p["name"]: p for p in self.participants}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> dict:
        self.viewed_calendars         : set  = set()
        self.proposed_slot            : dict = None
        self.booked_meeting           : dict = None
        self.report_no_solution_called: bool = False
        self.report_no_solution_reason: str  = None
        self.find_slots_called        : bool = False
        self._valid_slots_cache       : list = None  # lazily populated
        self.action_log               : list = []
        self.step_count               : int  = 0
        self.done                     : bool = False
        return self.state()

    def state(self) -> dict:
        visible_calendars = {
            name: self._participant_map[name]
            for name in self.viewed_calendars
        }
        return {
            "task":                      self.name,
            "scenario":                  self.scenario,
            "viewed_calendars":          sorted(self.viewed_calendars),
            "visible_calendars":         visible_calendars,
            "proposed_slot":             self.proposed_slot,
            "booked_meeting":            self.booked_meeting,
            "report_no_solution_called": self.report_no_solution_called,
            "find_slots_called":         self.find_slots_called,
            "action_log":                self.action_log[-10:],
            "step_count":                self.step_count,
            "done":                      self.done,
        }

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action: dict) -> tuple[dict, float, bool, dict]:
        self.step_count += 1
        reward = 0.0
        info   = {}
        act    = action.get("action")

        if act == "view_calendar":
            reward, info = self._view_calendar(action)
        elif act == "find_slots":
            reward, info = self._find_slots()
        elif act == "propose_meeting":
            reward, info = self._propose_meeting(action)
        elif act == "book_meeting":
            reward, info = self._book_meeting(action)
        elif act == "report_no_solution":
            reward, info = self._report_no_solution(action)
        else:
            reward = R["unknown_action"]
            info   = {
                "error": f"Unknown action '{act}'.",
                "valid_actions": [
                    "view_calendar", "find_slots",
                    "propose_meeting", "book_meeting", "report_no_solution",
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

    def _view_calendar(self, action: dict) -> tuple[float, dict]:
        name = action.get("participant")
        if name not in self._participant_map:
            return R["unknown_participant"], {
                "error": f"Unknown participant {name!r}. Known: {sorted(self._participant_map.keys())}"
            }
        if name in self.viewed_calendars:
            return R["view_calendar_repeat"], {"note": f"already viewed {name!r}"}
        self.viewed_calendars.add(name)
        return R["view_calendar"], {
            "result":   f"Loaded calendar for {name}",
            "calendar": self._participant_map[name],
        }

    def _find_slots(self) -> tuple[float, dict]:
        """
        Compute all valid meeting slots from calendar data.
        Result is cached on the task instance for the lifetime of the episode.
        """
        self.find_slots_called = True
        slots = self._get_valid_slots()
        if slots:
            return R["find_slots_found"], {"slots": slots, "count": len(slots)}
        return R["find_slots_none"], {
            "slots": [], "count": 0,
            "note": "No valid slots found within the date range for all required attendees.",
        }

    def _propose_meeting(self, action: dict) -> tuple[float, dict]:
        slot = {"date": action.get("date"), "start": action.get("start"), "end": action.get("end")}
        if _slot_in_list(slot, self._get_valid_slots()):
            self.proposed_slot = slot
            return R["valid_proposal"], {"result": "valid_proposal", "slot": slot}
        reason = self._invalid_reason(slot)
        return R["invalid_proposal"], {"result": "invalid_slot", "reason": reason}

    def _book_meeting(self, action: dict) -> tuple[float, dict]:
        if self.booked_meeting is not None:
            return R["book_already_done"], {"error": "A meeting is already booked"}

        slot = {"date": action.get("date"), "start": action.get("start"), "end": action.get("end")}

        if not _slot_in_list(slot, self._get_valid_slots()):
            reason = self._invalid_reason(slot)
            return R["book_invalid"], {"result": "booking_rejected", "reason": reason}

        self.booked_meeting = slot
        self.done           = True

        bonus = 0.0
        if not (self.proposed_slot and _slots_match(self.proposed_slot, slot)):
            bonus += R["book_without_proposal"]  # negative: skipped propose step

        if _slots_match(slot, self._BEST_SLOT):
            return R["book_best"] + bonus, {"result": "booked_best_slot", "slot": slot}
        return R["book_valid"] + bonus, {"result": "booked_valid_slot", "slot": slot}

    def _report_no_solution(self, action: dict) -> tuple[float, dict]:
        """
        Terminal action for when the agent determines no valid slot exists.
        Penalised if used when valid slots are actually available.
        """
        if self.report_no_solution_called:
            return -0.05, {"note": "already reported no solution"}

        reason = action.get("reason", "").strip()
        self.report_no_solution_called = True
        self.report_no_solution_reason = reason
        self.done = True

        actual_slots = self._get_valid_slots()
        if not actual_slots:
            return 0.40, {
                "result":       "correctly_identified_no_solution",
                "actual_slots": 0,
                "note":         "Correct: no valid slot exists in the given date range.",
            }
        # Agent gave up even though valid slots existed — heavy penalty
        return -0.30, {
            "result":       "incorrectly_reported_no_solution",
            "actual_slots": len(actual_slots),
            "note":         f"Incorrect: {len(actual_slots)} valid slot(s) exist.",
        }

    # ── Slot computation ──────────────────────────────────────────────────────

    def _get_valid_slots(self) -> list[dict]:
        """Return valid slots, computing and caching on first call."""
        if self._valid_slots_cache is None:
            self._valid_slots_cache = self._compute_valid_slots()
        return self._valid_slots_cache

    def _compute_valid_slots(self) -> list[dict]:
        """Enumerate all valid duration-min slots across the scenario date range."""
        duration  = self.scenario["duration_minutes"]
        dr        = self.scenario["date_range"]
        required  = self.scenario["required_attendees"]
        cals      = [self._participant_map[n] for n in required]
        valid     = []
        for day in _date_range(dr["start"], dr["end"]):
            valid.extend(_slots_for_day(cals, day, duration))
        return valid

    def _invalid_reason(self, slot: dict) -> str:
        date_s = slot.get("date", "")
        start  = slot.get("start", "")
        end    = slot.get("end", "")

        if not (date_s and start and end):
            return "slot is missing date, start, or end"

        duration  = self.scenario["duration_minutes"]
        start_min = _t2m(start)
        end_min   = _t2m(end)

        if end_min - start_min != duration:
            return f"duration mismatch — need exactly {duration} min, got {end_min - start_min}"

        work_start, work_end = 9 * 60, 17 * 60
        if start_min < work_start or end_min > work_end:
            return "outside working hours (09:00–17:00)"

        dr = self.scenario["date_range"]
        if not (dr["start"] <= date_s <= dr["end"]):
            return f"date {date_s!r} is outside allowed range {dr['start']}–{dr['end']}"

        for name in self.scenario["required_attendees"]:
            p    = self._participant_map[name]
            busy = p["busy_slots"].get(date_s, [])
            for b in busy:
                b_s = _t2m(b["start"])
                b_e = _t2m(b["end"])
                if start_min < b_e and end_min > b_s:
                    return f"{name} is busy {b['start']}–{b['end']} on {date_s}"

        return "unknown conflict (no matching valid slot)"


# ── Impossible variant ─────────────────────────────────────────────────────────

class SchedulingImpossibleTask(MeetingSchedulingTask):
    """
    No-solution scheduling scenario.
    Calendars are designed so no 60-min window exists for all required attendees.
    The agent earns full credit by calling find_slots (empty), then report_no_solution.
    Attempting to book an invalid slot scores 0.0 on the final grade.
    """
    name       = "scheduling_impossible"
    difficulty = "medium"
    _DATA_PATH = _DATA_DIR / "calendars_impossible.json"
    # No static valid slots for this variant — always computed dynamically
    _STATIC_VALID_SLOTS = []
    _BEST_SLOT = None  # not applicable


# ── Module-level helpers ───────────────────────────────────────────────────────

def _t2m(t: str) -> int:
    """'HH:MM' → minutes since midnight."""
    h, m = map(int, t.split(":"))
    return h * 60 + m

def _m2t(m: int) -> str:
    """Minutes since midnight → 'HH:MM'."""
    return f"{m // 60:02d}:{m % 60:02d}"

def _date_range(start: str, end: str) -> list[str]:
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    out  = []
    d    = s
    while d <= e:
        out.append(d.isoformat())
        d += timedelta(days=1)
    return out

def _free_intervals(busy_slots: list, work_start: int, work_end: int) -> list[tuple]:
    """Return list of (start_min, end_min) free intervals within working hours."""
    busy   = sorted((_t2m(b["start"]), _t2m(b["end"])) for b in busy_slots)
    free   = []
    cursor = work_start
    for b_s, b_e in busy:
        if cursor < b_s:
            free.append((cursor, b_s))
        cursor = max(cursor, b_e)
    if cursor < work_end:
        free.append((cursor, work_end))
    return free

def _slots_for_day(participants: list, day: str, duration: int) -> list[dict]:
    """Find all valid meeting start times on a single day for all participants."""
    work_start, work_end = 9 * 60, 17 * 60

    all_free = [
        _free_intervals(p["busy_slots"].get(day, []), work_start, work_end)
        for p in participants
    ]

    # Intersect across all participants
    intersection = all_free[0]
    for other in all_free[1:]:
        new = []
        for (a_s, a_e) in intersection:
            for (b_s, b_e) in other:
                s = max(a_s, b_s)
                e = min(a_e, b_e)
                if e - s >= duration:
                    new.append((s, e))
        intersection = new

    # Step by 30 min to enumerate bookable start times within each window
    slots = []
    for (s, e) in intersection:
        t = s
        while t + duration <= e:
            slots.append({"date": day, "start": _m2t(t), "end": _m2t(t + duration)})
            t += 30
    return slots

def _slots_match(a: dict, b: dict) -> bool:
    return (
        a.get("date")  == b.get("date")  and
        a.get("start") == b.get("start") and
        a.get("end")   == b.get("end")
    )

def _slot_in_list(slot: dict, lst: list[dict]) -> bool:
    return any(_slots_match(slot, s) for s in lst)
