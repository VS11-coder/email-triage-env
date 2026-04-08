"""
Unit tests for the Email Triage OpenEnv environment.
Covers: environment lifecycle, all three graders, duplicate-action guardrail,
input validation, edge cases (max_steps, invalid actions, empty submissions).

Run with:  python -m pytest test_env.py -v
"""
import sys
import os
import warnings
import pytest

# ── Adjust path so env imports work when run from project root ─────────────
# The project uses `from env.models import ...`, so we need the env/ package.
# If running from a flat directory, add the parent so `env` resolves.
sys.path.insert(0, os.path.dirname(__file__))

from env.email_env import EmailTriageEnv
from env.models import Action
from env.graders import grade_easy, grade_medium, grade_hard
from env.data.emails import EASY_EMAILS, MEDIUM_EMAILS, HARD_EMAILS


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def easy_env():
    env = EmailTriageEnv(task="email_classification")
    env.reset()
    return env


@pytest.fixture
def medium_env():
    env = EmailTriageEnv(task="email_response")
    env.reset()
    return env


@pytest.fixture
def hard_env():
    env = EmailTriageEnv(task="inbox_management")
    env.reset()
    return env


# ─────────────────────────────────────────────────────────────────────────────
# Test: Environment Reset
# ─────────────────────────────────────────────────────────────────────────────
class TestReset:
    def test_easy_reset_returns_correct_task(self, easy_env):
        result = easy_env.reset()
        assert result.task == "email_classification"
        assert result.observation.done is False
        assert result.observation.step == 0

    def test_medium_reset_email_count(self, medium_env):
        state = medium_env.state()
        assert state.inbox.total_emails == len(MEDIUM_EMAILS)

    def test_hard_reset_email_count(self, hard_env):
        state = hard_env.state()
        assert state.inbox.total_emails == len(HARD_EMAILS)

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            EmailTriageEnv(task="nonexistent_task")

    def test_reset_clears_state(self, easy_env):
        # Take some actions, then reset
        easy_env.step(Action(action_type="classify", email_id="e001", classification="spam", notes="low"))
        easy_env.reset()
        state = easy_env.state()
        assert len(state.actions_taken) == 0
        assert state.inbox.step_count == 0


# ─────────────────────────────────────────────────────────────────────────────
# Test: Step Mechanics
# ─────────────────────────────────────────────────────────────────────────────
class TestStep:
    def test_classify_gives_positive_reward(self, easy_env):
        result = easy_env.step(Action(
            action_type="classify", email_id="e001",
            classification="spam", notes="low"
        ))
        assert result.reward == 0.05
        assert result.done is False

    def test_reply_too_short_penalized(self, medium_env):
        result = medium_env.step(Action(
            action_type="reply", email_id="m001",
            reply_text="ok"
        ))
        assert result.reward == -0.05

    def test_valid_reply_gives_reward(self, medium_env):
        result = medium_env.step(Action(
            action_type="reply", email_id="m001",
            reply_text="Thank you for bringing this to our attention. We apologize for the discrepancy."
        ))
        assert result.reward == 0.1

    def test_next_email_advances_index(self, easy_env):
        easy_env.step(Action(action_type="next_email"))
        state = easy_env.state()
        assert state.inbox.current_index == 1

    def test_skip_penalizes_and_advances(self, easy_env):
        result = easy_env.step(Action(action_type="skip"))
        assert result.reward == -0.05
        state = easy_env.state()
        assert state.inbox.current_index == 1

    def test_invalid_action_type_penalized(self, easy_env):
        result = easy_env.step(Action(action_type="delete", email_id="e001"))
        assert result.reward == -0.1
        assert "invalid_action" in result.info.get("error", "")

    def test_classify_without_value_penalized(self, easy_env):
        result = easy_env.step(Action(action_type="classify", email_id="e001"))
        assert result.reward == -0.05

    def test_archive_labels_email(self, hard_env):
        result = hard_env.step(Action(action_type="archive", email_id="h004"))
        assert result.reward == 0.03
        state = hard_env.state()
        email = next(e for e in state.inbox.emails if e.id == "h004")
        assert "archived" in email.labels

    def test_delete_labels_email(self, hard_env):
        result = hard_env.step(Action(action_type="delete", email_id="h002"))
        assert result.reward == 0.03
        state = hard_env.state()
        email = next(e for e in state.inbox.emails if e.id == "h002")
        assert "deleted" in email.labels

    def test_mark_read_sets_flag(self, hard_env):
        result = hard_env.step(Action(action_type="mark_read", email_id="h001"))
        assert result.reward == 0.01
        state = hard_env.state()
        email = next(e for e in state.inbox.emails if e.id == "h001")
        assert email.read is True

    def test_schedule_meeting_without_time(self, hard_env):
        result = hard_env.step(Action(action_type="schedule_meeting", email_id="h003"))
        assert result.reward == -0.02

    def test_schedule_meeting_with_time(self, hard_env):
        result = hard_env.step(Action(
            action_type="schedule_meeting",
            email_id="h003",
            meeting_time="Thursday 2 PM"
        ))
        assert result.reward == 0.08


# ─────────────────────────────────────────────────────────────────────────────
# Test: Duplicate-Action Guardrail (Enhancement B)
# ─────────────────────────────────────────────────────────────────────────────
class TestDuplicateGuardrail:
    def test_duplicate_classify_gives_zero_reward(self, easy_env):
        first = easy_env.step(Action(
            action_type="classify", email_id="e001",
            classification="spam", notes="low"
        ))
        assert first.reward == 0.05

        second = easy_env.step(Action(
            action_type="classify", email_id="e001",
            classification="spam", notes="low"
        ))
        assert second.reward == 0.0
        assert "duplicate_action" in second.info.get("warning", "")

    def test_different_actions_on_same_email_not_duplicate(self, hard_env):
        r1 = hard_env.step(Action(action_type="reply", email_id="h001",
                                  reply_text="I'll coordinate with finance and have this ready by 3 PM."))
        r2 = hard_env.step(Action(action_type="flag", email_id="h001"))
        assert r1.reward == 0.1
        assert r2.reward == 0.05  # not a duplicate

    def test_same_action_on_different_emails_not_duplicate(self, easy_env):
        r1 = easy_env.step(Action(action_type="classify", email_id="e001",
                                  classification="spam", notes="low"))
        easy_env.step(Action(action_type="next_email"))
        r2 = easy_env.step(Action(action_type="classify", email_id="e002",
                                  classification="work", notes="urgent"))
        assert r1.reward == 0.05
        assert r2.reward == 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Test: Max Steps
# ─────────────────────────────────────────────────────────────────────────────
class TestMaxSteps:
    def test_exceeding_max_steps_terminates(self, easy_env):
        max_steps = easy_env._cfg["max_steps"]
        for _ in range(max_steps):
            result = easy_env.step(Action(action_type="next_email"))
            if result.done:
                break

        # One more to trigger max_steps
        result = easy_env.step(Action(action_type="next_email"))
        assert result.done is True
        assert result.reward == -0.5

    def test_step_after_done_gives_zero(self, easy_env):
        easy_env.step(Action(action_type="submit"))
        result = easy_env.step(Action(action_type="classify", email_id="e001",
                                      classification="spam"))
        assert result.done is True
        assert result.reward == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Test: Submit & Grading
# ─────────────────────────────────────────────────────────────────────────────
class TestSubmit:
    def test_empty_submit_scores_zero(self, easy_env):
        result = easy_env.step(Action(action_type="submit"))
        assert result.done is True
        assert result.reward == 0.0  # no classifications made

    def test_submit_returns_grading_result(self, easy_env):
        easy_env.step(Action(action_type="classify", email_id="e001",
                             classification="spam", notes="low"))
        result = easy_env.step(Action(action_type="submit"))
        assert "grading_result" in result.info
        assert "score" in result.info["grading_result"]


# ─────────────────────────────────────────────────────────────────────────────
# Test: Action History in Observations (Enhancement E)
# ─────────────────────────────────────────────────────────────────────────────
class TestActionHistory:
    def test_initial_observation_has_empty_history(self, easy_env):
        state = easy_env.reset()
        assert state.observation.actions_taken_summary == []

    def test_action_appears_in_history(self, easy_env):
        result = easy_env.step(Action(action_type="classify", email_id="e001",
                                      classification="spam", notes="low"))
        summary = result.observation.actions_taken_summary
        assert len(summary) == 1
        assert summary[0]["email_id"] == "e001"
        assert summary[0]["action_type"] == "classify"

    def test_duplicate_actions_excluded_from_history(self, easy_env):
        easy_env.step(Action(action_type="classify", email_id="e001",
                             classification="spam", notes="low"))
        result = easy_env.step(Action(action_type="classify", email_id="e001",
                                      classification="work", notes="urgent"))
        # Duplicate should be excluded from summary
        non_dup = [a for a in result.observation.actions_taken_summary
                   if a["email_id"] == "e001"]
        assert len(non_dup) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Test: Graders Directly
# ─────────────────────────────────────────────────────────────────────────────
class TestGradeEasy:
    def test_perfect_classification(self):
        """All emails classified correctly → score = 1.0."""
        actions = []
        for email in EASY_EMAILS:
            gt = email["ground_truth"]
            actions.append({
                "action_type": "classify",
                "email_id": email["id"],
                "classification": gt["category"],
                "notes": gt["priority"],
            })
        result = grade_easy(actions)
        assert result["score"] == 1.0
        assert result["category_accuracy"] == 1.0
        assert result["priority_accuracy"] == 1.0

    def test_all_wrong_scores_zero(self):
        actions = []
        for email in EASY_EMAILS:
            actions.append({
                "action_type": "classify",
                "email_id": email["id"],
                "classification": "xyzzy",
                "notes": "xyzzy",
            })
        result = grade_easy(actions)
        assert result["score"] == 0.0

    def test_no_actions_scores_zero(self):
        result = grade_easy([])
        assert result["score"] == 0.0

    def test_synonym_matching(self):
        """Synonyms like 'junk' for 'spam' should match."""
        actions = [{
            "action_type": "classify",
            "email_id": "e001",
            "classification": "junk",
            "notes": "minor",
        }]
        result = grade_easy(actions)
        detail = result["details"]["e001"]
        assert detail["category_correct"] is True
        assert detail["priority_correct"] is True


class TestGradeMedium:
    def test_quality_replies_score_well(self):
        actions = [
            {
                "action_type": "reply",
                "email_id": "m001",
                "reply_text": "I sincerely apologize for the invoice discrepancy. The correct amount is $11,000 as agreed. We will reissue a corrected invoice within 24 hours.",
            },
            {
                "action_type": "reply",
                "email_id": "m002",
                "reply_text": "Sure, happy to review the PR today. Will prioritize it since it's blocking others.",
            },
            {
                "action_type": "archive",
                "email_id": "m003",
            },
            {
                "action_type": "reply",
                "email_id": "m004",
                "reply_text": "The Apollo project is at 75% complete. Main blocker is the API integration. Timeline: expected completion by Feb 15.",
            },
            {"action_type": "flag", "email_id": "m005"},
        ]
        # Add entries for m006, m007, m008 if they exist
        if len(MEDIUM_EMAILS) > 5:
            actions.extend([
                {
                    "action_type": "reply",
                    "email_id": "m006",
                    "reply_text": "Thank you for the heads up about the bandwidth usage. We would like to discuss options before the auto-upgrade. Can we schedule a call to review our budget and plan?",
                },
                {
                    "action_type": "archive",
                    "email_id": "m007",
                },
                {
                    "action_type": "reply",
                    "email_id": "m008",
                    "reply_text": "I sincerely apologize for the delay on the deliverables. We have prioritized this and you will receive the complete package by Wednesday. I understand the escalation and take this very seriously.",
                },
            ])
        result = grade_medium(actions)
        assert result["score"] >= 0.6

    def test_no_actions_scores_zero(self):
        result = grade_medium([])
        assert result["score"] == 0.0


class TestGradeHard:
    def test_optimal_actions_score_high(self):
        """Take ideal actions on each email."""
        actions = [
            {"action_type": "reply", "email_id": "h001", "reply_text": "Will coordinate with finance."},
            {"action_type": "flag", "email_id": "h001"},
            {"action_type": "delete", "email_id": "h002"},
            {"action_type": "reply", "email_id": "h003", "reply_text": "Let's discuss Thursday."},
            {"action_type": "schedule_meeting", "email_id": "h003", "meeting_time": "Thursday 2 PM"},
            {"action_type": "archive", "email_id": "h004"},
            {"action_type": "reply", "email_id": "h005", "reply_text": "Will review and sign today."},
            {"action_type": "flag", "email_id": "h005"},
            {"action_type": "archive", "email_id": "h006"},
            {"action_type": "reply", "email_id": "h007", "reply_text": "Will prepare onboarding."},
            {"action_type": "flag", "email_id": "h007"},
            {"action_type": "delete", "email_id": "h008"},
            {"action_type": "reply", "email_id": "h009", "reply_text": "Investigating the 503 errors now."},
            {"action_type": "flag", "email_id": "h009"},
            {"action_type": "reply", "email_id": "h010", "reply_text": "Count me in for lunch!"},
        ]
        # Add actions for new hard emails
        if len(HARD_EMAILS) > 10:
            actions.extend([
                {"action_type": "reply", "email_id": "h011", "reply_text": "Submitting expense report now."},
                {"action_type": "flag", "email_id": "h011"},
                {"action_type": "flag", "email_id": "h012"},
                {"action_type": "delete", "email_id": "h013"},
                {"action_type": "reply", "email_id": "h014", "reply_text": "Will prepare root cause analysis."},
                {"action_type": "flag", "email_id": "h014"},
                {"action_type": "schedule_meeting", "email_id": "h014", "meeting_time": "tomorrow 10 AM"},
                {"action_type": "archive", "email_id": "h015"},
                {"action_type": "reply", "email_id": "h016", "reply_text": "Will send SOC 2 documents this week."},
                {"action_type": "flag", "email_id": "h016"},
                {"action_type": "reply", "email_id": "h017", "reply_text": "Happy to help! Let's find time this week."},
                {"action_type": "schedule_meeting", "email_id": "h017", "meeting_time": "this week"},
            ])
        result = grade_hard(actions)
        assert result["score"] >= 0.7
        assert result["coverage"] == 1.0

    def test_no_actions_gets_coverage_penalty(self):
        result = grade_hard([])
        assert result["coverage"] == 0.0
        assert result["score"] < 0.5

    def test_wrong_actions_penalized(self):
        """Deleting urgent emails should hurt score."""
        actions = [
            {"action_type": "delete", "email_id": "h001"},
            {"action_type": "delete", "email_id": "h005"},
            {"action_type": "delete", "email_id": "h009"},
        ]
        result = grade_hard(actions)
        assert result["score"] < 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Test: Input Validation (Enhancement C)
# ─────────────────────────────────────────────────────────────────────────────
class TestInputValidation:
    def test_unknown_classification_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Action(action_type="classify", classification="banana")
            assert len(w) == 1
            assert "banana" in str(w[0].message)

    def test_known_classification_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Action(action_type="classify", classification="spam")
            assert len(w) == 0

    def test_reply_text_length_limit(self):
        """reply_text over 2000 chars should be rejected."""
        with pytest.raises(Exception):
            Action(action_type="reply", reply_text="x" * 2001)

    def test_reply_text_within_limit(self):
        action = Action(action_type="reply", reply_text="x" * 2000)
        assert len(action.reply_text) == 2000


# ─────────────────────────────────────────────────────────────────────────────
# Test: Email Data Counts (Enhancement D)
# ─────────────────────────────────────────────────────────────────────────────
class TestEmailData:
    def test_easy_email_count(self):
        assert len(EASY_EMAILS) == 15

    def test_medium_email_count(self):
        assert len(MEDIUM_EMAILS) == 8

    def test_hard_email_count(self):
        assert len(HARD_EMAILS) == 17

    def test_all_emails_have_ground_truth(self):
        for email_list in [EASY_EMAILS, MEDIUM_EMAILS, HARD_EMAILS]:
            for email in email_list:
                assert "ground_truth" in email, f"Email {email['id']} missing ground_truth"

    def test_email_ids_unique(self):
        all_ids = (
            [e["id"] for e in EASY_EMAILS]
            + [e["id"] for e in MEDIUM_EMAILS]
            + [e["id"] for e in HARD_EMAILS]
        )
        assert len(all_ids) == len(set(all_ids)), "Duplicate email IDs found"
