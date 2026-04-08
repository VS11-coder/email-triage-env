"""
EmailTriageEnv — implements the OpenEnv interface.

Three tasks:
  1. email_classification (Easy)   — classify 15 emails by category + priority
  2. email_response      (Medium)  — reply appropriately to 8 work emails
  3. inbox_management    (Hard)    — triage a mixed inbox of 17 emails end-to-end
"""
import copy
from typing import Any, Dict, List, Optional, Set, Tuple

from env.models import (
    Action, Email, Observation, InboxState, StepResult, ResetResult, StateResult
)
from env.graders import grade_easy, grade_medium, grade_hard
from env.data.emails import EASY_EMAILS, MEDIUM_EMAILS, HARD_EMAILS


TASK_CONFIG = {
    "email_classification": {
        "emails": EASY_EMAILS,
        "max_steps": 50,
        "description": (
            "Classify each of the 15 emails in your inbox. "
            "For each email, use the 'classify' action with:\n"
            "  - classification: one of spam | personal | work | newsletter\n"
            "  - notes: priority — one of urgent | medium | low\n"
            "Work through all emails, then use 'submit' to finish."
        ),
        "available_actions": ["classify", "next_email", "skip", "submit"],
        "grader": grade_easy,
    },
    "email_response": {
        "emails": MEDIUM_EMAILS,
        "max_steps": 40,
        "description": (
            "You have 8 work emails requiring attention. "
            "For each email, decide: reply with appropriate text, flag for follow-up, "
            "or archive if no action is needed. "
            "Use 'reply' with reply_text for emails needing a response. "
            "Use 'archive' or 'flag' for others. Then 'next_email' to move on."
        ),
        "available_actions": ["reply", "archive", "flag", "mark_read", "next_email", "skip", "submit"],
        "grader": grade_medium,
    },
    "inbox_management": {
        "emails": HARD_EMAILS,
        "max_steps": 80,
        "description": (
            "You are managing a busy inbox of 17 emails. "
            "For each email, take the most appropriate action(s): "
            "reply (with reply_text), flag (important follow-up), archive (FYI only), "
            "delete (spam/irrelevant), mark_read, or schedule_meeting (if a meeting is requested). "
            "You may take multiple actions per email. Then 'next_email' to move on. "
            "Prioritise urgent emails. Complete all emails, then 'submit'."
        ),
        "available_actions": [
            "reply", "archive", "flag", "delete", "mark_read",
            "schedule_meeting", "next_email", "skip", "submit"
        ],
        "grader": grade_hard,
    },
}


def _build_email(raw: Dict) -> Email:
    return Email(
        id=raw["id"],
        sender=raw["sender"],
        subject=raw["subject"],
        body=raw["body"],
        timestamp=raw["timestamp"],
    )


class EmailTriageEnv:
    """OpenEnv-compliant Email Triage environment."""

    VALID_TASKS = list(TASK_CONFIG.keys())

    def __init__(self, task: str = "email_classification"):
        if task not in TASK_CONFIG:
            raise ValueError(f"Unknown task '{task}'. Choose from: {self.VALID_TASKS}")
        self.task_name = task
        self._cfg = TASK_CONFIG[task]
        self._inbox: List[Email] = []
        self._current_idx: int = 0
        self._step_count: int = 0
        self._actions_taken: List[Dict[str, Any]] = []
        self._done: bool = False
        self._score: float = 0.0
        # Enhancement B: track actions per email to detect duplicates
        self._actions_per_email: Dict[str, Set[str]] = {}

    # ── OpenEnv Interface ──────────────────────────────────────────────────

    def reset(self) -> ResetResult:
        self._inbox = [_build_email(r) for r in self._cfg["emails"]]
        self._current_idx = 0
        self._step_count = 0
        self._actions_taken = []
        self._done = False
        self._score = 0.0
        self._actions_per_email = {}
        return ResetResult(
            observation=self._make_observation("Environment reset. Start processing emails."),
            task=self.task_name,
        )

    def step(self, action: Action) -> StepResult:
        if self._done:
            obs = self._make_observation("Episode already finished.")
            obs.done = True
            return StepResult(observation=obs, reward=0.0, done=True,
                              info={"error": "Episode already done"})

        self._step_count += 1
        reward = 0.0
        info: Dict[str, Any] = {}
        message = ""

        # ── Infinite-loop / excessive-step penalty ─────────────────────────
        if self._step_count > self._cfg["max_steps"]:
            self._done = True
            reward = -0.5
            info["error"] = "max_steps_exceeded"
            obs = self._make_observation("Max steps exceeded. Episode terminated.")
            obs.done = True
            return StepResult(observation=obs, reward=reward, done=True, info=info)

        # ── Terminal action ────────────────────────────────────────────────
        if action.action_type == "submit":
            result = self._cfg["grader"](self._actions_taken)
            self._score = result["score"]
            self._done = True
            reward = self._score
            info["grading_result"] = result
            message = f"Submitted! Final score: {self._score:.4f}"
            obs = self._make_observation(message)
            obs.done = True
            return StepResult(observation=obs, reward=reward, done=True, info=info)

        # ── Navigation ────────────────────────────────────────────────────
        if action.action_type == "next_email":
            if self._current_idx < len(self._inbox) - 1:
                self._current_idx += 1
                reward = 0.02  # small reward for making progress
                message = f"Moved to email {self._current_idx + 1}/{len(self._inbox)}."
            else:
                message = "Already at last email. Use 'submit' to finish."
                reward = 0.0
            obs = self._make_observation(message)
            return StepResult(observation=obs, reward=reward, done=False, info=info)

        if action.action_type == "skip":
            reward = -0.05  # small penalty for skipping
            message = f"Skipped email {self._current_idx + 1}."
            if self._current_idx < len(self._inbox) - 1:
                self._current_idx += 1
            obs = self._make_observation(message)
            return StepResult(observation=obs, reward=reward, done=False, info=info)

        # ── Email actions ──────────────────────────────────────────────────
        current_email = self._current_email()
        if current_email is None:
            obs = self._make_observation("No current email.")
            return StepResult(observation=obs, reward=-0.01, done=False, info={"error": "no_email"})

        email_id = action.email_id or current_email.id

        # Validate action is allowed
        if action.action_type not in self._cfg["available_actions"]:
            obs = self._make_observation(f"Action '{action.action_type}' not available for this task.")
            return StepResult(observation=obs, reward=-0.1, done=False,
                              info={"error": f"invalid_action: {action.action_type}"})

        # ── Enhancement B: Duplicate-action guardrail ──────────────────────
        if email_id not in self._actions_per_email:
            self._actions_per_email[email_id] = set()

        is_duplicate = action.action_type in self._actions_per_email[email_id]
        self._actions_per_email[email_id].add(action.action_type)

        if is_duplicate:
            # Still record for grading (grader dedupes), but zero reward + warning
            action_record = {
                "action_type": action.action_type,
                "email_id": email_id,
                "classification": action.classification,
                "reply_text": action.reply_text,
                "flag_reason": action.flag_reason,
                "meeting_time": action.meeting_time,
                "notes": action.notes,
                "step": self._step_count,
                "duplicate": True,
            }
            self._actions_taken.append(action_record)
            message = (
                f"Duplicate action: '{action.action_type}' already taken on '{email_id}'. "
                f"No additional reward. Proceed to the next email or submit."
            )
            info["warning"] = "duplicate_action"
            obs = self._make_observation(message)
            return StepResult(observation=obs, reward=0.0, done=False, info=info)

        # Record action
        action_record = {
            "action_type": action.action_type,
            "email_id": email_id,
            "classification": action.classification,
            "reply_text": action.reply_text,
            "flag_reason": action.flag_reason,
            "meeting_time": action.meeting_time,
            "notes": action.notes,
            "step": self._step_count,
        }
        self._actions_taken.append(action_record)

        # Apply action to email state
        if action.action_type == "mark_read":
            for e in self._inbox:
                if e.id == email_id:
                    e.read = True
            reward = 0.01
            message = f"Marked email {email_id} as read."

        elif action.action_type == "classify":
            if not action.classification:
                reward = -0.05
                message = "classify action requires a 'classification' value."
            else:
                reward = 0.05  # incremental reward; full score at submit
                message = (
                    f"Classified '{email_id}' as {action.classification} "
                    f"(priority: {action.notes or 'unset'})."
                )

        elif action.action_type == "reply":
            if not action.reply_text:
                reward = -0.05
                message = "reply action requires 'reply_text'."
            elif len(action.reply_text.strip()) < 20:
                reward = -0.05
                message = "Reply too short — please write a meaningful response."
            else:
                reward = 0.1
                message = f"Reply recorded for '{email_id}'."

        elif action.action_type == "flag":
            reward = 0.05
            message = f"Email '{email_id}' flagged for follow-up."

        elif action.action_type == "archive":
            reward = 0.03
            for e in self._inbox:
                if e.id == email_id:
                    e.labels.append("archived")
            message = f"Email '{email_id}' archived."

        elif action.action_type == "delete":
            reward = 0.03
            for e in self._inbox:
                if e.id == email_id:
                    e.labels.append("deleted")
            message = f"Email '{email_id}' deleted."

        elif action.action_type == "schedule_meeting":
            if not action.meeting_time:
                reward = -0.02
                message = "schedule_meeting requires 'meeting_time'."
            else:
                reward = 0.08
                message = f"Meeting scheduled for '{action.meeting_time}' re: email '{email_id}'."

        else:
            reward = 0.0
            message = f"Action '{action.action_type}' acknowledged."

        obs = self._make_observation(message)
        return StepResult(observation=obs, reward=reward, done=False, info=info)

    def state(self) -> StateResult:
        return StateResult(
            inbox=InboxState(
                emails=self._inbox,
                current_index=self._current_idx,
                total_emails=len(self._inbox),
                processed=self._current_idx,
                task_name=self.task_name,
                step_count=self._step_count,
                max_steps=self._cfg["max_steps"],
            ),
            score=self._score,
            actions_taken=self._actions_taken,
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    def _current_email(self) -> Optional[Email]:
        if 0 <= self._current_idx < len(self._inbox):
            return self._inbox[self._current_idx]
        return None

    def _make_observation(self, message: str = "") -> Observation:
        email = self._current_email()
        processed_ids = {a["email_id"] for a in self._actions_taken if a.get("email_id")}

        # Enhancement E: lightweight action history for the agent
        actions_summary = [
            {"email_id": a["email_id"], "action_type": a["action_type"]}
            for a in self._actions_taken
            if a.get("email_id") and not a.get("duplicate")
        ]

        return Observation(
            current_email=email,
            inbox_summary={
                "total": len(self._inbox),
                "current_index": self._current_idx + 1,
                "steps_taken": self._step_count,
                "max_steps": self._cfg["max_steps"],
                "emails_touched": len(processed_ids),
                "score_so_far": self._score,
            },
            task_description=self._cfg["description"],
            available_actions=self._cfg["available_actions"],
            step=self._step_count,
            done=self._done,
            message=message,
            actions_taken_summary=actions_summary,
        )
