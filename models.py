from pydantic import BaseModel, Field, field_validator
from typing import Any, Dict, List, Optional, Literal
import warnings


class Email(BaseModel):
    id: str
    sender: str
    subject: str
    body: str
    timestamp: str
    read: bool = False
    labels: List[str] = []


class InboxState(BaseModel):
    emails: List[Email]
    current_index: int = 0
    total_emails: int
    processed: int = 0
    task_name: str
    step_count: int = 0
    max_steps: int = 50


class Observation(BaseModel):
    current_email: Optional[Email]
    inbox_summary: Dict[str, Any]
    task_description: str
    available_actions: List[str]
    step: int
    done: bool = False
    message: str = ""
    actions_taken_summary: List[Dict[str, str]] = []


VALID_CLASSIFICATIONS = {
    "spam", "junk", "phishing", "scam",
    "work", "professional", "office", "business",
    "personal", "family", "friend", "social",
    "newsletter", "subscription", "news", "marketing",
    "urgent", "critical", "emergency",
}


class Action(BaseModel):
    action_type: Literal[
        "classify",
        "reply",
        "archive",
        "flag",
        "delete",
        "mark_read",
        "schedule_meeting",
        "next_email",
        "skip",
        "submit"
    ]
    email_id: Optional[str] = None
    classification: Optional[str] = Field(None, max_length=50)
    reply_text: Optional[str] = Field(None, max_length=2000)
    flag_reason: Optional[str] = Field(None, max_length=500)
    meeting_time: Optional[str] = None
    notes: Optional[str] = Field(None, max_length=200)

    @field_validator("classification")
    @classmethod
    def warn_unknown_classification(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v.lower().strip() not in VALID_CLASSIFICATIONS:
            warnings.warn(
                f"Classification '{v}' is not in the expected set. "
                f"Expected one of: spam, personal, work, newsletter, urgent",
                UserWarning,
                stacklevel=2,
            )
        return v


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = {}


class ResetResult(BaseModel):
    observation: Observation
    task: str


class StateResult(BaseModel):
    inbox: InboxState
    score: float
    actions_taken: List[Dict[str, Any]]
