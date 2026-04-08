"""
FastAPI server for Email Triage OpenEnv environment.
Exposes REST endpoints compatible with the OpenEnv interface.
"""
import time
import logging
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import asyncio

from env.email_env import EmailTriageEnv
from env.models import Action, StepResult, ResetResult, StateResult

# ── Structured logging ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("email_triage")

# ── Session storage with timestamps ───────────────────────────────────────
SESSION_TTL_SECONDS = int(__import__("os").getenv("SESSION_TTL", "3600"))  # 1 hour default

_envs: dict[str, EmailTriageEnv] = {}
_env_timestamps: dict[str, float] = {}


def _touch_session(session_id: str) -> None:
    """Update the last-access timestamp for a session."""
    _env_timestamps[session_id] = time.time()


def _prune_stale_sessions() -> int:
    """Remove sessions older than SESSION_TTL_SECONDS. Returns count pruned."""
    now = time.time()
    stale = [
        sid for sid, ts in _env_timestamps.items()
        if now - ts > SESSION_TTL_SECONDS
    ]
    for sid in stale:
        _envs.pop(sid, None)
        _env_timestamps.pop(sid, None)
    if stale:
        logger.info(f"Pruned {len(stale)} stale sessions: {stale}")
    return len(stale)


# ── Metrics ────────────────────────────────────────────────────────────────
_metrics = {
    "requests_total": 0,
    "reset_count": 0,
    "step_count": 0,
    "total_response_time_ms": 0.0,
    "scores_by_task": {},  # task_name -> list of final scores
}


# ── Background session cleanup ────────────────────────────────────────────
async def _session_cleanup_loop():
    """Periodically prune stale sessions every 5 minutes."""
    while True:
        await asyncio.sleep(300)
        _prune_stale_sessions()


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_session_cleanup_loop())
    yield
    task.cancel()


app = FastAPI(
    title="Email Triage OpenEnv",
    description="A real-world email triage and management RL environment.",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Timing middleware ──────────────────────────────────────────────────────
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed_ms = (time.time() - start) * 1000
    _metrics["requests_total"] += 1
    _metrics["total_response_time_ms"] += elapsed_ms
    logger.info(
        f"{request.method} {request.url.path} — {elapsed_ms:.1f}ms — {response.status_code}"
    )
    return response


def _get_env(session_id: str) -> EmailTriageEnv:
    if session_id not in _envs:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. Call /reset first.")
    _touch_session(session_id)
    return _envs[session_id]


# ── Health & Metadata ──────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "email-triage-env",
        "version": "1.1.0",
        "tasks": ["email_classification", "email_response", "inbox_management"],
        "status": "running",
    }


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "name": "email_classification",
                "difficulty": "easy",
                "description": "Classify 15 emails by category and priority.",
                "max_steps": 50,
            },
            {
                "name": "email_response",
                "difficulty": "medium",
                "description": "Draft replies and triage 8 work emails.",
                "max_steps": 40,
            },
            {
                "name": "inbox_management",
                "difficulty": "hard",
                "description": "Full triage of a 17-email inbox: reply, flag, archive, delete, schedule.",
                "max_steps": 80,
            },
        ]
    }


# ── Core OpenEnv Endpoints ─────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task: str = "email_classification"
    session_id: str = "default"


@app.post("/reset", response_model=ResetResult)
def reset(req: Optional[ResetRequest] = None):
    """Reset (or initialise) the environment for a given task."""
    if req is None:
        req = ResetRequest()
    try:
        env = EmailTriageEnv(task=req.task)
        _envs[req.session_id] = env
        _touch_session(req.session_id)
        _metrics["reset_count"] += 1
        logger.info(f"Session '{req.session_id}' reset for task '{req.task}'")
        return env.reset()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class StepRequest(BaseModel):
    action: Action
    session_id: str = "default"


@app.post("/step", response_model=StepResult)
def step(req: StepRequest):
    """Take one action in the environment."""
    env = _get_env(req.session_id)
    result = env.step(req.action)
    _metrics["step_count"] += 1

    # Track final scores for metrics
    if result.done and "grading_result" in result.info:
        task_name = env.task_name
        score = result.info["grading_result"].get("score", 0.0)
        _metrics["scores_by_task"].setdefault(task_name, []).append(score)

    return result


@app.get("/state", response_model=StateResult)
def state(session_id: str = Query(default="default")):
    """Return the full current state of the environment."""
    env = _get_env(session_id)
    return env.state()


@app.delete("/session")
def close_session(session_id: str = Query(default="default")):
    """Remove a session."""
    _envs.pop(session_id, None)
    _env_timestamps.pop(session_id, None)
    logger.info(f"Session '{session_id}' closed")
    return {"status": "closed", "session_id": session_id}


# ── Enhancement A: Session listing ────────────────────────────────────────

@app.get("/sessions")
def list_sessions():
    """List all active sessions with age information."""
    now = time.time()
    sessions = []
    for sid, env in _envs.items():
        age_s = now - _env_timestamps.get(sid, now)
        sessions.append({
            "session_id": sid,
            "task": env.task_name,
            "step_count": env._step_count,
            "done": env._done,
            "age_seconds": round(age_s, 1),
            "ttl_remaining_seconds": max(0, round(SESSION_TTL_SECONDS - age_s, 1)),
        })
    return {"active_sessions": len(sessions), "ttl_seconds": SESSION_TTL_SECONDS, "sessions": sessions}


# ── Enhancement F: Metrics endpoint ───────────────────────────────────────

@app.get("/metrics")
def metrics():
    """Server-side metrics: request counts, timing, and score distributions."""
    avg_response_ms = (
        _metrics["total_response_time_ms"] / _metrics["requests_total"]
        if _metrics["requests_total"] > 0 else 0.0
    )
    score_stats = {}
    for task_name, scores in _metrics["scores_by_task"].items():
        score_stats[task_name] = {
            "count": len(scores),
            "mean": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "min": round(min(scores), 4) if scores else 0.0,
            "max": round(max(scores), 4) if scores else 0.0,
        }
    return {
        "requests_total": _metrics["requests_total"],
        "reset_count": _metrics["reset_count"],
        "step_count": _metrics["step_count"],
        "avg_response_time_ms": round(avg_response_ms, 2),
        "active_sessions": len(_envs),
        "scores_by_task": score_stats,
    }


def start():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    start()
