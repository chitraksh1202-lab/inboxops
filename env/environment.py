"""
InboxOpsEnv – main environment interface.

Usage:
    from env import InboxOpsEnv

    env = InboxOpsEnv()
    state = env.reset("email_triage")       # or "meeting_scheduling" / "support_escalation"
    state, reward, done, info = env.step({"action": "open_email", "email_id": "email_001"})
    score = env.grade()                     # float 0.0–1.0, final deterministic score

The environment is a thin wrapper that delegates to the task object.
It tracks cumulative reward and exposes a grade() method.
"""

from .tasks.email_easy        import EmailTriageTask
from .tasks.scheduling_medium import MeetingSchedulingTask, SchedulingImpossibleTask
from .tasks.support_hard      import SupportEscalationTask
from .graders                 import grade

TASK_REGISTRY = {
    "email_triage":          EmailTriageTask,
    "meeting_scheduling":    MeetingSchedulingTask,
    "scheduling_impossible": SchedulingImpossibleTask,
    "support_escalation":    SupportEscalationTask,
}


class InboxOpsEnv:
    """
    Unified environment wrapper for all InboxOps benchmark tasks.

    reset(task_name) → initial state dict
    step(action)     → (state, reward, done, info)
    state()          → current state dict
    grade()          → float 0.0–1.0 final score
    summary()        → dict with task name, score, total reward, completion
    """

    def __init__(self):
        self._task      = None
        self._task_name = None
        self._total_reward = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self, task_name: str = "email_triage") -> dict:
        if task_name not in TASK_REGISTRY:
            raise ValueError(
                f"Unknown task {task_name!r}. "
                f"Available: {sorted(TASK_REGISTRY.keys())}"
            )
        self._task_name    = task_name
        self._task         = TASK_REGISTRY[task_name]()
        self._total_reward = 0.0
        return self._task.reset()

    def step(self, action: dict) -> tuple[dict, float, bool, dict]:
        self._assert_task()
        state, reward, done, info = self._task.step(action)
        self._total_reward += reward
        return state, reward, done, info

    def state(self) -> dict:
        self._assert_task()
        return self._task.state()

    def grade(self) -> float:
        """Return the deterministic final score (0.0–1.0) for the current episode."""
        self._assert_task()
        return grade(self._task_name, self._task)

    def summary(self) -> dict:
        """Convenience dict for run_baseline.py output."""
        self._assert_task()
        done  = self._task.done
        score = self.grade()
        return {
            "task":          self._task_name,
            "difficulty":    self._task.difficulty,
            "score":         score,
            "total_reward":  round(self._total_reward, 4),
            "steps":         self._task.step_count,
            "completed":     done,
            "passed":        score >= 0.7,  # informal pass threshold for demo
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _assert_task(self):
        if self._task is None:
            raise RuntimeError("Call env.reset(task_name) before using the environment.")
