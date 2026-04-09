"""
Graders for the Email Triage Environment.
All graders return a float in [0.0, 1.0].
Grading is deterministic and reproducible.
"""
from typing import Dict, Any, List
from env.data.emails import EASY_EMAILS, MEDIUM_EMAILS, HARD_EMAILS


# ─────────────────────────────────────────────
# TASK 1: Email Classification Grader (Easy)
# ─────────────────────────────────────────────
EASY_GT = {e["id"]: e["ground_truth"] for e in EASY_EMAILS}

CATEGORY_SYNONYMS = {
    "spam": {"spam", "junk", "phishing", "scam"},
    "work": {"work", "professional", "office", "business"},
    "personal": {"personal", "family", "friend", "social"},
    "newsletter": {"newsletter", "subscription", "news", "marketing"},
    "urgent": {"urgent", "critical", "emergency"},
}

PRIORITY_SYNONYMS = {
    "urgent": {"urgent", "critical", "high", "important", "asap"},
    "medium": {"medium", "normal", "moderate"},
    "low": {"low", "minor", "ignore", "unimportant"},
}

# Partial credit matrix: how "close" are two categories? (0.0 = totally wrong, 0.3 = close)
CATEGORY_SIMILARITY = {
    ("work", "newsletter"): 0.3,  # newsletters can be work-related
    ("newsletter", "work"): 0.3,
    ("personal", "work"): 0.2,    # sometimes ambiguous
    ("work", "personal"): 0.2,
    ("spam", "newsletter"): 0.2,  # marketing vs spam is blurry
    ("newsletter", "spam"): 0.2,
}

PRIORITY_SIMILARITY = {
    ("urgent", "medium"): 0.4,
    ("medium", "urgent"): 0.4,
    ("medium", "low"): 0.4,
    ("low", "medium"): 0.4,
    ("urgent", "low"): 0.1,
    ("low", "urgent"): 0.1,
}


def _normalize_category(cat: str) -> str:
    cat = (cat or "").lower().strip()
    for canonical, synonyms in CATEGORY_SYNONYMS.items():
        if cat in synonyms:
            return canonical
    return cat


def _normalize_priority(pri: str) -> str:
    pri = (pri or "").lower().strip()
    for canonical, synonyms in PRIORITY_SYNONYMS.items():
        if pri in synonyms:
            return canonical
    return pri


def grade_easy(actions_taken: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Score = (correct_categories * 0.6 + correct_priorities * 0.4) / num_emails
    Each email is worth equal weight.
    """
    classify_actions = {
        a["email_id"]: a
        for a in actions_taken
        if a.get("action_type") == "classify" and a.get("email_id")
    }

    n = len(EASY_GT)
    category_score = 0.0
    priority_score = 0.0
    details = {}

    for email_id, gt in EASY_GT.items():
        action = classify_actions.get(email_id)
        if action is None:
            details[email_id] = {"category_correct": False, "priority_correct": False, "classified": False}
            continue

        pred_cat = _normalize_category(action.get("classification", ""))
        pred_pri = _normalize_priority(action.get("notes", ""))

        gt_cat = _normalize_category(gt["category"])
        gt_pri = _normalize_priority(gt["priority"])

        cat_ok = pred_cat == gt_cat
        pri_ok = pred_pri == gt_pri

        if cat_ok:
            category_score += 1.0
        else:
            # Partial credit for close categories
            category_score += CATEGORY_SIMILARITY.get((pred_cat, gt_cat), 0.0)

        if pri_ok:
            priority_score += 1.0
        else:
            # Partial credit for close priorities
            priority_score += PRIORITY_SIMILARITY.get((pred_pri, gt_pri), 0.0)

        details[email_id] = {
            "category_correct": cat_ok,
            "priority_correct": pri_ok,
            "category_partial": CATEGORY_SIMILARITY.get((pred_cat, gt_cat), 0.0) if not cat_ok else 1.0,
            "priority_partial": PRIORITY_SIMILARITY.get((pred_pri, gt_pri), 0.0) if not pri_ok else 1.0,
            "predicted_category": pred_cat,
            "expected_category": gt_cat,
            "predicted_priority": pred_pri,
            "expected_priority": gt_pri,
        }

    final_score = (category_score * 0.6 + priority_score * 0.4) / n
    final_score = max(0.001, min(0.999, final_score))
    return {
        "score": round(final_score, 4),
        "category_accuracy": round(category_score / n, 4),
        "priority_accuracy": round(priority_score / n, 4),
        "details": details,
    }


# ─────────────────────────────────────────────
# TASK 2: Email Response Grader (Medium)
# ─────────────────────────────────────────────
MEDIUM_GT = {e["id"]: e["ground_truth"] for e in MEDIUM_EMAILS}

REPLY_KEYWORDS = {
    "m001": {
        "required": ["invoice", "correct", "apologize"],
        "bonus": ["timeline", "reissue", "12500", "11000"],
        "tone_words": ["sorry", "apologize", "apologies", "regret"],
    },
    "m002": {
        "required": ["review", "pr", "pull request"],
        "bonus": ["today", "time", "priority", "blocking"],
        "tone_words": ["sure", "happy", "will", "can"],
    },
    "m004": {
        "required": ["status", "progress", "complete"],
        "bonus": ["blocker", "timeline", "percent", "%"],
        "tone_words": [],
    },
    # Enhancement D: new medium email keyword tables
    "m006": {
        "required": ["bandwidth", "usage", "options"],
        "bonus": ["budget", "upgrade", "discuss", "friday", "tier", "plan"],
        "tone_words": ["appreciate", "understand", "would like"],
    },
    "m008": {
        "required": ["apologize", "delay", "deliverables"],
        "bonus": ["wednesday", "timeline", "concrete", "escalation", "priority"],
        "tone_words": ["sorry", "apologize", "apologies", "sincerely", "regret"],
    },
}


def _score_reply(email_id: str, reply_text: str) -> float:
    if not reply_text:
        return 0.0
    reply_lower = reply_text.lower()
    kw = REPLY_KEYWORDS.get(email_id)
    if not kw:
        return 0.5  # neutral score for emails without strict grading

    required_hits = sum(1 for w in kw["required"] if w in reply_lower)
    bonus_hits = sum(1 for w in kw["bonus"] if w in reply_lower)
    tone_hits = sum(1 for w in kw["tone_words"] if w in reply_lower)

    required_score = required_hits / max(len(kw["required"]), 1)
    bonus_score = min(bonus_hits / max(len(kw["bonus"]), 1), 1.0)
    tone_score = 1.0 if tone_hits > 0 or not kw["tone_words"] else 0.0

    return round(required_score * 0.5 + bonus_score * 0.3 + tone_score * 0.2, 4)


def grade_medium(actions_taken: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    For must_reply emails: score based on reply quality (keyword presence, tone).
    For non-must-reply: score based on correct action (archive/flag/skip).
    Penalty: replying to must-NOT-reply emails when wrong action taken.
    """
    action_map: Dict[str, List[Dict]] = {}
    for a in actions_taken:
        eid = a.get("email_id")
        if eid:
            action_map.setdefault(eid, []).append(a)

    total_score = 0.0
    details = {}
    n = len(MEDIUM_GT)

    for email_id, gt in MEDIUM_GT.items():
        actions = action_map.get(email_id, [])
        must_reply = gt.get("must_reply", False)

        reply_action = next((a for a in actions if a.get("action_type") == "reply"), None)
        flag_action = next((a for a in actions if a.get("action_type") == "flag"), None)
        archive_action = next((a for a in actions if a.get("action_type") == "archive"), None)

        if must_reply:
            if reply_action:
                reply_score = _score_reply(email_id, reply_action.get("reply_text", ""))
                urgency_bonus = 0.1 if gt.get("urgency") == "high" and flag_action else 0.0
                score = min(reply_score + urgency_bonus, 1.0)
            else:
                score = 0.0
        else:
            acceptable = set(gt.get("acceptable_actions", ["archive", "flag", "skip", "delete"]))
            action_types = {a.get("action_type") for a in actions}
            if action_types & acceptable:
                score = 1.0
            elif reply_action:
                score = 0.2  # replied when not required — partial credit
            else:
                score = 0.0

        total_score += score
        details[email_id] = {"score": score, "must_reply": must_reply}

    final_score = total_score / n
    final_score = max(0.001, min(0.999, final_score))
    return {"score": round(final_score, 4), "details": details}


# ─────────────────────────────────────────────
# TASK 3: Inbox Management Grader (Hard)
# ─────────────────────────────────────────────
HARD_GT = {e["id"]: e["ground_truth"] for e in HARD_EMAILS}

HARD_ACTION_SCORES = {
    "h001": {
        "reply": 0.5, "flag": 0.3, "archive": -0.2,
        "delete": -0.3, "schedule_meeting": 0.0, "mark_read": 0.2
    },
    "h002": {
        "delete": 0.8, "archive": 0.6, "reply": -0.2,
        "flag": -0.1, "mark_read": 0.2
    },
    "h003": {
        "reply": 0.4, "schedule_meeting": 0.4, "flag": 0.1,
        "archive": -0.2, "delete": -0.3, "mark_read": 0.1
    },
    "h004": {
        "archive": 0.8, "mark_read": 0.5, "delete": 0.5,
        "reply": -0.1, "flag": 0.0
    },
    "h005": {
        "reply": 0.5, "flag": 0.3, "archive": -0.2,
        "delete": -0.5, "mark_read": 0.2
    },
    "h006": {
        "archive": 0.8, "delete": 0.7, "mark_read": 0.4,
        "reply": -0.1, "flag": 0.0
    },
    "h007": {
        "reply": 0.4, "flag": 0.4, "archive": -0.1,
        "delete": -0.2, "mark_read": 0.2, "schedule_meeting": 0.2
    },
    "h008": {
        "delete": 0.9, "archive": 0.6, "reply": -0.3,
        "flag": -0.1, "mark_read": 0.1
    },
    "h009": {
        "reply": 0.5, "flag": 0.3, "archive": -0.3,
        "delete": -0.5, "schedule_meeting": 0.2, "mark_read": 0.1
    },
    "h010": {
        "reply": 0.6, "archive": 0.3, "mark_read": 0.3,
        "flag": 0.0, "delete": -0.1
    },
    # Enhancement D: score tables for new hard emails
    "h011": {
        "reply": 0.5, "flag": 0.3, "archive": -0.1,
        "delete": -0.3, "mark_read": 0.2, "schedule_meeting": 0.0
    },
    "h012": {
        "flag": 0.7, "reply": 0.2, "archive": -0.2,
        "delete": -0.5, "mark_read": 0.1, "schedule_meeting": 0.0
    },
    "h013": {
        "delete": 0.9, "archive": 0.5, "reply": -0.3,
        "flag": -0.1, "mark_read": 0.1
    },
    "h014": {
        "reply": 0.4, "flag": 0.3, "schedule_meeting": 0.3,
        "archive": -0.3, "delete": -0.5, "mark_read": 0.1
    },
    "h015": {
        "archive": 0.7, "mark_read": 0.5, "reply": 0.1,
        "flag": 0.0, "delete": -0.1
    },
    "h016": {
        "reply": 0.5, "flag": 0.3, "archive": -0.2,
        "delete": -0.5, "mark_read": 0.1, "schedule_meeting": 0.1
    },
    "h017": {
        "reply": 0.4, "schedule_meeting": 0.4, "flag": 0.1,
        "archive": -0.1, "delete": -0.2, "mark_read": 0.1
    },
}


def grade_hard(actions_taken: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Multi-criteria scoring:
    - Correct actions on each email (priority-weighted)
    - Bonus for scheduling meetings when required
    - Bonus for flagging urgent items
    - Penalty for destructive actions on important emails
    - Coverage: must process all emails
    """
    action_map: Dict[str, List[str]] = {}
    for a in actions_taken:
        eid = a.get("email_id")
        if eid:
            action_map.setdefault(eid, []).append(a.get("action_type", ""))

    # Priority weights
    PRIORITY_WEIGHTS = {"urgent": 1.5, "high": 1.2, "medium": 1.0, "low": 0.8}
    MAX_SCORE_PER_EMAIL = 1.0

    total_weighted_score = 0.0
    total_weight = 0.0
    details = {}

    for email_id, gt in HARD_GT.items():
        priority = gt.get("priority", "medium")
        weight = PRIORITY_WEIGHTS.get(priority, 1.0)
        total_weight += weight

        actions_on_email = action_map.get(email_id, [])
        score_map = HARD_ACTION_SCORES.get(email_id, {})

        if not actions_on_email:
            # No action taken — partial penalty
            email_score = -0.1
        else:
            # Sum contributions from each action, cap at MAX
            raw_score = sum(score_map.get(a, 0.0) for a in actions_on_email)
            email_score = max(min(raw_score, MAX_SCORE_PER_EMAIL), -0.5)

        total_weighted_score += email_score * weight
        details[email_id] = {
            "actions_taken": actions_on_email,
            "email_score": round(email_score, 4),
            "priority": priority,
            "weight": weight,
        }

    # Normalize to [0, 1]
    max_possible = total_weight * MAX_SCORE_PER_EMAIL
    raw_final = total_weighted_score / max_possible if max_possible > 0 else 0.0
    # Shift from [-0.5, 1.0] range to [0, 1]
    final_score = max(0.0, min(1.0, (raw_final + 0.5) / 1.5))

    # Coverage bonus: reward processing all emails
    coverage = len(action_map) / len(HARD_GT)
    final_score = min(1.0, final_score * 0.80 + coverage * 0.15)

    # Dependency ordering bonus: reward agents that process prerequisites first
    dep_bonus = 0.0
    dep_count = 0
    # Build step ordering from actions_taken
    email_first_step: Dict[str, int] = {}
    for i, a in enumerate(actions_taken):
        eid = a.get("email_id")
        if eid and eid not in email_first_step:
            email_first_step[eid] = i
    for email_id, gt in HARD_GT.items():
        dep = gt.get("depends_on")
        if dep:
            dep_count += 1
            dep_step = email_first_step.get(dep)
            this_step = email_first_step.get(email_id)
            if dep_step is not None and this_step is not None:
                if dep_step < this_step:
                    dep_bonus += 0.15  # correct order
                else:
                    dep_bonus -= 0.1   # wrong order
    if dep_count > 0:
        final_score += dep_bonus / dep_count * 0.05  # 5% weight for ordering

    final_score = max(0.001, min(0.999, final_score))

    return {
        "score": round(final_score, 4),
        "coverage": round(coverage, 4),
        "weighted_raw": round(raw_final, 4),
        "details": details,
    }
