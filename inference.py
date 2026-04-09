"""
inference.py — OpenEnv Hackathon submission script for Email Triage Environment.

Reads environment variables:
  API_BASE_URL   — LLM API endpoint  (default: https://api.openai.com/v1)
  MODEL_NAME     — model identifier   (default: gpt-4.1-mini)
  HF_TOKEN       — Hugging Face / API token (required)

Runs all three tasks and emits [START], [STEP], and [END] lines to stdout.
"""
import os
import sys
import json
import time
import requests
from openai import OpenAI

# ── Environment variables ──────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4.1-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ── Server configuration ───────────────────────────────────────────────────
SERVER_URL  = os.getenv("ENV_SERVER_URL", "http://localhost:7860")
SESSION_ID  = "inference_session"
BENCHMARK   = "email-triage-env"

TASKS = ["email_classification", "email_response", "inbox_management"]

# ── Retry configuration (Enhancement G) ───────────────────────────────────
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
RETRY_BASE_DELAY = float(os.getenv("LLM_RETRY_DELAY", "1.0"))

# ── System prompts per task ────────────────────────────────────────────────
SYSTEM_PROMPTS = {
    "email_classification": """You are an expert email triage assistant. Classify each email accurately.

STRATEGY:
1. Read the sender, subject, and body carefully.
2. Identify the category: spam (junk/phishing/scam), personal (family/friends/social), work (professional/office/business), or newsletter (subscriptions/marketing).
3. Assess priority: urgent (needs immediate action), medium (respond today), or low (can wait/informational).
4. Look for red flags: unknown senders with urgency cues = likely spam. Known internal senders = likely work.

For each email, respond with:
{"action_type": "classify", "email_id": "<id>", "classification": "<spam|personal|work|newsletter>", "notes": "<urgent|medium|low>"}

After classifying the current email, use {"action_type": "next_email"} to advance.
After ALL emails are classified, use {"action_type": "submit"} to finish.
Always respond with ONLY a valid JSON object, no other text.""",

    "email_response": """You are a senior executive assistant managing work communications.

STRATEGY:
1. For emails requiring a reply: write professional, substantive responses addressing ALL key points mentioned.
2. Include specific details: acknowledge the issue, provide concrete next steps, mention timelines.
3. Match the tone to the situation: apologetic for complaints, collaborative for requests, concise for status updates.
4. For FYI/informational emails: archive them. For important non-reply items: flag them.
5. Check the actions_taken_summary to avoid duplicating work on emails you've already handled.

Actions: reply, archive, flag, mark_read, next_email, skip, submit

For replies: {"action_type": "reply", "email_id": "<id>", "reply_text": "<detailed professional response>"}
For archive: {"action_type": "archive", "email_id": "<id>"}
For flag: {"action_type": "flag", "email_id": "<id>", "flag_reason": "<reason>"}
After each email, use {"action_type": "next_email"} to advance.
When done with ALL emails, use {"action_type": "submit"}.
Always respond with ONLY a valid JSON object.""",

    "inbox_management": """You are an expert executive inbox manager handling a complex inbox of 17 emails.

CRITICAL STRATEGY — READ CAREFULLY:
1. PRIORITIZE: Handle urgent/high-priority emails first. Scan the inbox summary to identify critical items.
2. DEPENDENCIES: Some emails are related. For example, responding to an escalation is better after understanding the underlying technical issue. Look for contextual connections between emails.
3. MULTI-ACTION: You can take multiple actions on one email before moving to the next (e.g., reply + flag, or reply + schedule_meeting).
4. DECISIVE ACTIONS per email type:
   - Spam/phishing → delete immediately
   - FYI/newsletters → archive or mark_read
   - Needs response → reply with professional, detailed reply_text
   - Important follow-up → flag with clear reason
   - Meeting request → schedule_meeting with proposed meeting_time
   - Urgent + needs response → reply + flag
5. COVERAGE: Process ALL 17 emails. Unprocessed emails hurt your score.
6. Check actions_taken_summary to track what you've already handled and avoid duplicates.

Format: {"action_type": "...", "email_id": "...", "reply_text": "...", "flag_reason": "...", "meeting_time": "..."}
After each email, use {"action_type": "next_email"} to advance.
After ALL emails, use {"action_type": "submit"}.
Always respond with ONLY a valid JSON object.""",
}


# ── HTTP helpers ───────────────────────────────────────────────────────────
def api_reset(task: str) -> dict:
    r = requests.post(
        f"{SERVER_URL}/reset",
        json={"task": task, "session_id": SESSION_ID},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def api_step(action: dict) -> dict:
    r = requests.post(
        f"{SERVER_URL}/step",
        json={"action": action, "session_id": SESSION_ID},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def api_close():
    try:
        requests.delete(f"{SERVER_URL}/session", params={"session_id": SESSION_ID}, timeout=10)
    except Exception:
        pass


# ── Enhancement G: Retry wrapper with exponential backoff ─────────────────
def _call_llm_with_retry(messages: list, temperature: float = 0.2, max_tokens: int = 512) -> str:
    """Call the LLM API with retry logic. Returns raw response text."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(
                    f"[RETRY] attempt={attempt}/{MAX_RETRIES} error={exc} "
                    f"retrying_in={delay:.1f}s",
                    file=sys.stderr,
                )
                time.sleep(delay)
            else:
                print(
                    f"[RETRY] all {MAX_RETRIES} attempts failed: {exc}",
                    file=sys.stderr,
                )
    raise last_error  # type: ignore[misc]


# ── Enhancement H: Smart context window management ────────────────────────
def _build_context(history: list, max_recent: int = 4, keep_first: int = 2) -> list:
    """
    Build a context window that always includes the first N entries
    (initial email context) plus the most recent entries, to avoid
    losing important early context while staying within token limits.
    """
    if len(history) <= keep_first + max_recent:
        return list(history)

    first_entries = history[:keep_first]
    recent_entries = history[-max_recent:]

    # Add a bridging message so the model knows there's a gap
    bridge = {
        "role": "user",
        "content": (
            f"[... {len(history) - keep_first - max_recent} earlier steps omitted "
            f"for brevity ...]"
        ),
    }

    return first_entries + [bridge] + recent_entries


def _parse_json_action(raw: str) -> dict:
    """Parse a JSON action from LLM response, handling markdown fences."""
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: try to extract JSON
        import re
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        return {"action_type": "skip"}


# ── LLM call ──────────────────────────────────────────────────────────────
def get_action(task: str, observation: dict, history: list) -> dict:
    """Call the LLM to get the next action given the current observation."""
    email = observation.get("current_email")
    summary = observation.get("inbox_summary", {})
    message_text = observation.get("message", "")
    available = observation.get("available_actions", [])
    actions_summary = observation.get("actions_taken_summary", [])

    # Build user prompt
    if email:
        email_str = (
            f"ID: {email['id']}\n"
            f"From: {email['sender']}\n"
            f"Subject: {email['subject']}\n"
            f"Body: {email['body']}\n"
            f"Timestamp: {email['timestamp']}"
        )
    else:
        email_str = "(no current email)"

    # Enhancement E integration: show completed actions to the agent
    actions_context = ""
    if actions_summary:
        action_lines = [f"  - {a['action_type']} on {a['email_id']}" for a in actions_summary]
        actions_context = f"\nActions completed so far:\n" + "\n".join(action_lines) + "\n"

    # Enhancement 10: surface dependency hints and urgency bonus to the LLM
    strategic_hints = ""
    dep_hint = summary.get("dependency_hint")
    urgency_avail = summary.get("urgency_bonus_available", False)
    if dep_hint:
        strategic_hints += f"\n⚠️ DEPENDENCY: {dep_hint}"
    if urgency_avail:
        strategic_hints += "\n⚡ URGENCY BONUS: This is an urgent email. Handle it now for an early-action bonus!"

    user_content = f"""Current Email:
{email_str}

Inbox Summary: {json.dumps(summary)}
Last message: {message_text}
Available actions: {available}{actions_context}{strategic_hints}

What action do you take? Respond with ONLY a JSON object."""

    messages = [{"role": "system", "content": SYSTEM_PROMPTS[task]}]
    # Enhancement H: smart context window instead of fixed sliding window
    context = _build_context(history, max_recent=4, keep_first=2)
    for h in context:
        messages.append(h)
    messages.append({"role": "user", "content": user_content})

    raw = _call_llm_with_retry(messages)
    return _parse_json_action(raw)


# ── Main episode runner ────────────────────────────────────────────────────
def run_task(task: str) -> None:
    reset_data = api_reset(task)
    observation = reset_data["observation"]
    done = observation.get("done", False)

    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}")
    sys.stdout.flush()

    step_n = 0
    rewards = []
    history = []
    last_error = "null"
    success = False
    final_score = 0.0

    try:
        while not done:
            # Get action from LLM
            action = get_action(task, observation, history)

            # Record in history
            history.append({"role": "assistant", "content": json.dumps(action)})

            # Step the environment
            step_result = api_step(action)
            reward = step_result["reward"]
            done = step_result["done"]
            new_obs = step_result["observation"]
            info = step_result.get("info", {})

            step_n += 1
            rewards.append(reward)
            error_val = info.get("error", None)
            last_error = str(error_val) if error_val else "null"

            action_str = json.dumps(action).replace("\n", " ")
            print(f"[STEP] step={step_n} action={action_str} reward={reward:.2f} done={str(done).lower()} error={last_error}")
            sys.stdout.flush()

            # Add env response to history
            history.append({
                "role": "user",
                "content": f"Action result: {new_obs.get('message', '')}",
            })

            observation = new_obs

            if done:
                grading = info.get("grading_result", {})
                final_score = grading.get("score", 0.0)
                success = final_score >= 0.5

    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
    finally:
        final_score = max(0.001, min(0.999, final_score))
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(f"[END] success={str(success).lower()} steps={step_n} score={final_score:.3f} rewards={rewards_str}")
        sys.stdout.flush()
        api_close()


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Optionally run a single task via CLI arg, else run all
    if len(sys.argv) > 1 and sys.argv[1] in TASKS:
        run_task(sys.argv[1])
    else:
        for task in TASKS:
            run_task(task)
            time.sleep(1)
