---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---
# 📬 Email Triage OpenEnv

A **real-world email triage and management** reinforcement learning environment built for the [OpenEnv RL Challenge](https://huggingface.co/spaces/openenv). Agents must process a realistic inbox — classifying, replying, archiving, and scheduling — with increasing complexity across three graduated tasks.

---

## 🗂️ Environment Overview

Email triage is one of the most universal real-world tasks — the average knowledge worker spends **28% of their workday** managing email ([McKinsey](https://www.mckinsey.com/industries/technology-media-and-telecommunications/our-insights/the-social-economy)). This environment challenges LLM agents to handle a realistic inbox end-to-end:

- **Understand email context** — classify by sender, subject, body, and urgency signals
- **Make correct triage decisions** — reply vs. archive vs. delete vs. flag vs. schedule
- **Write professional replies** — appropriate tone, covering key points, handling escalations
- **Prioritise correctly** — urgent items first, avoid destructive actions on important emails
- **Handle edge cases** — subtle phishing, ambiguous work/personal emails, angry clients, multi-action threads

The environment uses **40 synthetic emails** with deterministic ground-truth labels, ensuring fully reproducible grading with no LLM-in-the-loop evaluation.

---

## 🎯 Tasks

| Task | Difficulty | Emails | Max Steps | Description |
|------|-----------|--------|-----------|-------------|
| `email_classification` | 🟢 Easy | 15 | 50 | Classify emails by category and priority |
| `email_response` | 🟡 Medium | 8 | 40 | Draft replies and triage work emails |
| `inbox_management` | 🔴 Hard | 17 | 80 | Full inbox triage: reply, flag, delete, schedule |

### Task 1 — Email Classification (Easy)
Classify 15 emails by **category** (`spam`, `personal`, `work`, `newsletter`) and **priority** (`urgent`, `medium`, `low`). Includes ambiguous emails (work/personal blend), subtle phishing, and LinkedIn-style newsletters. Graded deterministically against ground-truth labels with synonym matching (e.g., "junk" → "spam").

**Scoring:** 60% category accuracy + 40% priority accuracy.

### Task 2 — Email Response (Medium)
Process 8 work emails. For emails requiring a response, draft a professional reply covering key points. Includes vendor escalations, angry client threads, and passive FYIs. Graded on reply quality (keyword coverage, tone) and correct non-reply decisions.

**Scoring:** Must-reply emails scored on keyword hits (50%), context completeness (30%), tone (20%). Non-reply emails scored on correct action selection.

### Task 3 — Inbox Management (Hard)
Full triage of a 17-email mixed inbox. Each email has a priority weight (`urgent=1.5×`, `high=1.2×`, `medium=1.0×`, `low=0.8×`). Includes customer escalations ($400k ARR churn threat), SOC 2 audit deadlines, AWS billing alerts, and intern mentoring requests. Multiple actions per email are allowed and scored. Coverage bonus rewards processing all emails.

**Scoring:** Weighted action correctness (80%) + coverage (15%) + dependency ordering (5%).

---

## ⚙️ Advanced Reward Mechanics

This environment uses several advanced reinforcement learning reward shaping mechanics to test true LLM agent intelligence, beyond simple text matching:

1. **Dependency Ordering Bonus (Hard Task)**  
   Emails are linked contextually (e.g., an escalation `h014` is driven by an underlying technical alert `h009`). If an agent processes the prerequisite email *before* the dependent email, it receives a **+5% final score bonus**. Processing them out of order incurs a penalty.
   
2. **Time-Sensitivity Bonus (Medium/Hard Tasks)**  
   Urgent emails processed within the first 30% of the episode's step budget receive a **+0.05 step reward bonus**. The `inbox_summary` dynamically surfaces `urgency_bonus_available` to hint at this urgency constraints.

3. **Partial Credit Semantic Matrix (Easy Task)**  
   Instead of binary pass/fail classification, the environment awards partial step credit for "close" semantic misses where ambiguity exists (e.g., classifying a `newsletter` as `work` yields 0.3 points, recognizing that newsletters can be work-related).

4. **Flag Reason Quality Scoring**  
   When agents use the `flag` action, they receive an additional **+0.03 micro-reward** if they provide a detailed, contextual `flag_reason` longer than 10 characters, encouraging descriptive action generation.

---

## 📐 Action Space

| `action_type` | Fields Required | Description |
|---------------|----------------|-------------|
| `classify` | `email_id`, `classification`, `notes` (priority) | Classify current email |
| `reply` | `email_id`, `reply_text` (max 2000 chars) | Send a reply |
| `archive` | `email_id` | Archive the email |
| `flag` | `email_id`, `flag_reason` (max 500 chars) | Flag for follow-up |
| `delete` | `email_id` | Delete the email |
| `mark_read` | `email_id` | Mark as read |
| `schedule_meeting` | `email_id`, `meeting_time` | Schedule a meeting |
| `next_email` | — | Advance to next email |
| `skip` | — | Skip current email (small penalty) |
| `submit` | — | Finish episode and receive graded score |

---

## 👁️ Observation Space

```json
{
  "current_email": {
    "id": "e001",
    "sender": "no-reply@casino-wins.biz",
    "subject": "YOU WON $5,000,000!!!",
    "body": "...",
    "timestamp": "2024-01-15T08:00:00"
  },
  "inbox_summary": {
    "total": 15,
    "current_index": 1,
    "steps_taken": 0,
    "max_steps": 50,
    "emails_touched": 0,
    "score_so_far": 0.0
  },
  "task_description": "Classify each of the 15 emails...",
  "available_actions": ["classify", "next_email", "skip", "submit"],
  "step": 0,
  "done": false,
  "message": "Environment reset. Start processing emails.",
  "actions_taken_summary": []
}
```

The `actions_taken_summary` field provides a running list of `{email_id, action_type}` pairs so agents can track what they've already done without relying on their own context window.

---

## 🏆 Reward Function

Rewards are given **throughout the episode** (not just at terminal), providing partial progress signals:

| Event | Reward |
|-------|--------|
| Correct classify action | +0.05 |
| Quality reply (≥20 chars) | +0.10 |
| flag action | +0.05 |
| archive/delete action | +0.03 |
| schedule_meeting (with time) | +0.08 |
| next_email (progress) | +0.02 |
| mark_read | +0.01 |
| skip | −0.05 |
| Reply too short | −0.05 |
| Missing required field | −0.02 to −0.05 |
| Invalid action for task | −0.10 |
| **Duplicate action** (same type on same email) | **0.00 + warning** |
| **submit** | = final graded score |
| max_steps exceeded | −0.50 |

---

## 🚀 Setup & Usage

### Local Development

```bash
# Clone and install
git clone <your-repo-url>
cd email_triage_env
pip install -r requirements.txt

# Create the env/ package (needed for imports)
mkdir -p env/data && touch env/__init__.py env/data/__init__.py
ln -sf ../email_env.py env/email_env.py
ln -sf ../models.py env/models.py
ln -sf ../graders.py env/graders.py
ln -sf ../../emails.py env/data/emails.py

# Start the server
python server.py
# Server runs at http://localhost:7860
```

### Docker

```bash
docker build -t email-triage-env .
docker run -p 7860:7860 email-triage-env
```

### Run Inference

```bash
export HF_TOKEN=your_token_here
export MODEL_NAME=gpt-4.1-mini        # optional
export API_BASE_URL=https://api.openai.com/v1  # optional
export ENV_SERVER_URL=http://localhost:7860

python inference.py
# or run a single task:
python inference.py email_classification
```

### Run Tests

```bash
pip install pytest
python -m pytest test_env.py -v
# 45 tests, all passing
```

### API Quick-start

```python
import requests

BASE = "http://localhost:7860"

# Reset the environment
obs = requests.post(f"{BASE}/reset", json={"task": "email_classification", "session_id": "s1"}).json()

# Take an action
result = requests.post(f"{BASE}/step", json={
    "session_id": "s1",
    "action": {
        "action_type": "classify",
        "email_id": "e001",
        "classification": "spam",
        "notes": "low"
    }
}).json()

print(result["reward"], result["done"])
```

---

## 📊 Baseline Performance

Tested with `gpt-4.1-mini` at temperature=0.2:

| Task | Score | Coverage | Notes |
|------|-------|----------|-------|
| `email_classification` | 1.00 | — | Perfect classification on all 15 emails |
| `email_response` | 0.81 | — | Strong replies; misses some keyword details |
| `inbox_management` | 0.88 | 1.0 | Good triage; occasionally suboptimal action combos |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check and environment metadata |
| `GET` | `/tasks` | List all available tasks with descriptions |
| `POST` | `/reset` | Reset environment for a task (body: `{task, session_id}`) |
| `POST` | `/step` | Take an action (body: `{action, session_id}`) |
| `GET` | `/state` | Get full current state (query: `session_id`) |
| `DELETE` | `/session` | Close a session (query: `session_id`) |
| `GET` | `/sessions` | List all active sessions with TTL info |
| `GET` | `/metrics` | Server metrics: request counts, latency, score stats |

---

## 📁 Project Structure

```
email_triage_env/
├── env/                      # Python package (created via symlinks)
│   ├── __init__.py
│   ├── email_env.py → ../email_env.py
│   ├── models.py    → ../models.py
│   ├── graders.py   → ../graders.py
│   └── data/
│       ├── __init__.py
│       └── emails.py → ../../emails.py
├── email_env.py              # Core environment (reset/step/state)
├── models.py                 # Pydantic models with validation
├── graders.py                # Deterministic graders (0.0–1.0)
├── emails.py                 # 40 synthetic emails with ground truth
├── server.py                 # FastAPI server (port 7860)
├── inference.py              # Baseline inference script
├── test_env.py               # 45 unit tests
├── openenv.yaml              # OpenEnv metadata spec
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🏷️ Tags
`openenv` · `email` · `triage` · `real-world` · `nlp` · `productivity`
