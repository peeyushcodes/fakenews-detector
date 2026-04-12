"""
FastAPI server for FakeNews Detector OpenEnv Environment
Exposes: POST /reset, POST /step, GET /state, GET /health, GET /openenv.yaml
"""

import os
import sys
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

# Ensure root directory is in the import path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from fakenews_env import FakeNewsAction, FakeNewsEnv, FakeNewsObservation, TASKS

app = FastAPI(
    title="FakeNews Detector OpenEnv",
    description="An environment where AI agents fact-check news claims and assign credibility scores.",
    version="1.0.0",
)

# Global env instances per task (simple in-memory state)
_envs: Dict[str, FakeNewsEnv] = {}


def get_env(task_name: str) -> FakeNewsEnv:
    if task_name not in _envs:
        _envs[task_name] = FakeNewsEnv(task_name=task_name)
    return _envs[task_name]


# ─────────────────────────────────────────────
#  Request/Response Models
# ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_name: Optional[str] = "obvious_fake"


class StepRequest(BaseModel):
    task_name: Optional[str] = "obvious_fake"
    verdict: str
    confidence: float
    reasoning: str
    key_evidence: list


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


# ─────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "environment": "fakenews-detector", "tasks": list(TASKS.keys())}


@app.post("/reset")
def reset(req: ResetRequest = None):
    task_name = (req.task_name if req else None) or "obvious_fake"
    if task_name not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task '{task_name}'. Valid: {list(TASKS.keys())}")
    env = FakeNewsEnv(task_name=task_name)
    _envs[task_name] = env
    obs = env.reset()
    return {"observation": obs.model_dump(), "task_name": task_name}


@app.post("/step")
def step(req: StepRequest):
    task_name = req.task_name or "obvious_fake"
    env = get_env(task_name)
    if env._done:
        raise HTTPException(status_code=400, detail="Episode done. Call /reset first.")

    action = FakeNewsAction(
        verdict=req.verdict,
        confidence=req.confidence,
        reasoning=req.reasoning,
        key_evidence=req.key_evidence,
    )
    result = env.step(action)
    safe_reward = round(min(max(float(result.reward), 0.01), 0.99), 3)
    return StepResponse(
        observation=result.observation.model_dump(),
        reward=safe_reward,
        done=result.done,
        info=result.info,
    )


@app.get("/state")
def state(task_name: str = "obvious_fake"):
    env = get_env(task_name)
    return env.state()


@app.get("/tasks")
def list_tasks():
    return {
        name: {
            "difficulty": task["difficulty"],
            "description": task["description"],
            "num_claims": len(task["claims"]),
            "max_steps": task["max_steps"],
        }
        for name, task in TASKS.items()
    }


OPENENV_YAML = """
name: fakenews-detector
version: "1.0.0"
description: >
  An OpenEnv environment where AI agents must fact-check news claims,
  assign verdicts (true/false/misleading/unverifiable), provide reasoning,
  and cite evidence. Rewards partial progress on reasoning quality.

tasks:
  - name: obvious_fake
    difficulty: easy
    description: Identify clearly fabricated or satirical news claims
    max_steps: 3

  - name: subtle_misinfo
    difficulty: medium
    description: Detect partially true but misleading or out-of-context claims
    max_steps: 4

  - name: complex_claim
    difficulty: hard
    description: Evaluate nuanced scientific, economic, or political claims
    max_steps: 5

observation_space:
  type: object
  fields:
    claim: string
    context: string
    source: string
    task_name: string
    step: integer
    max_steps: integer
    previous_feedback: string (optional)

action_space:
  type: object
  fields:
    verdict: "one of: true, false, misleading, unverifiable"
    confidence: "float 0.0-1.0"
    reasoning: "string, minimum 20 words"
    key_evidence: "list of 2-4 strings"

reward:
  range: [0.0, 1.0]
  components:
    verdict_correct: 0.40
    confidence_appropriate: 0.20
    reasoning_quality: 0.25
    evidence_quality: 0.15

endpoints:
  reset: POST /reset
  step: POST /step
  state: GET /state
  health: GET /health
"""


@app.get("/openenv.yaml", response_class=PlainTextResponse)
def openenv_yaml():
    return OPENENV_YAML


def main():
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
