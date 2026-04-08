"""
Inference Script — FakeNews Detector OpenEnv
=============================================
Follows the mandatory [START] / [STEP] / [END] stdout format.

Environment variables required:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier.
    HF_TOKEN       Your Hugging Face / API key.
    SPACE_URL      Your HF Space URL (e.g. https://your-space.hf.space)

Run:
    SPACE_URL=https://your-space.hf.space HF_TOKEN=hf_xxx python inference.py
"""

import json
import os
import sys
import textwrap
from typing import List, Optional

import requests
from openai import OpenAI

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
SPACE_URL = os.getenv("SPACE_URL", "http://localhost:7860").rstrip("/")
BENCHMARK = "fakenews-detector"
MAX_STEPS = 5
TEMPERATURE = 0.3
MAX_TOKENS = 500

TASKS = ["obvious_fake", "subtle_misinfo", "complex_claim"]

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert fact-checker and investigative journalist.
Your job is to analyze news claims and determine their veracity.

For each claim, respond with ONLY a valid JSON object with these exact fields:
{
  "verdict": "<one of: true, false, misleading, unverifiable>",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<your detailed analysis, minimum 50 words>",
  "key_evidence": ["<evidence point 1>", "<evidence point 2>", "<evidence point 3>"]
}

Verdict definitions:
- "true": The claim is accurate and well-supported by evidence
- "false": The claim is factually incorrect or fabricated
- "misleading": The claim has some truth but is distorted, taken out of context, or leads to false conclusions
- "unverifiable": The claim cannot be verified with available information

Rules:
- Be thorough in your reasoning (at least 50 words)
- Always provide exactly 3 key evidence points
- Set confidence based on how certain you are (0.9+ for obvious cases, 0.6-0.8 for nuanced ones)
- Output ONLY the JSON object, no other text
""").strip()


# ─────────────────────────────────────────────
#  Logging (exact format required)
# ─────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Truncate action for log readability
    action_log = action[:120].replace("\n", " ") if len(action) > 120 else action.replace("\n", " ")
    print(
        f"[STEP] step={step} action={action_log} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ─────────────────────────────────────────────
#  Model Interaction
# ─────────────────────────────────────────────

def get_model_action(client: OpenAI, observation: dict) -> dict:
    """Ask the LLM to fact-check the claim and return structured action."""
    user_prompt = textwrap.dedent(f"""
    Claim to fact-check: {observation['claim']}

    Context: {observation['context']}

    Source: {observation['source']}

    {f"Previous feedback: {observation['previous_feedback']}" if observation.get('previous_feedback') else ""}

    Analyze this claim carefully and respond with the JSON object.
    """).strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()

        # Strip markdown code blocks if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        action = json.loads(text)

        # Validate and sanitize
        verdict = action.get("verdict", "unverifiable").lower()
        if verdict not in ["true", "false", "misleading", "unverifiable"]:
            verdict = "unverifiable"

        confidence = float(action.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        reasoning = str(action.get("reasoning", "No reasoning provided."))

        evidence = action.get("key_evidence", [])
        if not isinstance(evidence, list):
            evidence = [str(evidence)]
        evidence = [str(e) for e in evidence[:4]]
        while len(evidence) < 2:
            evidence.append("Insufficient evidence provided.")

        return {
            "verdict": verdict,
            "confidence": confidence,
            "reasoning": reasoning,
            "key_evidence": evidence,
        }

    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"[DEBUG] Parse error: {exc} | Raw: {text[:200] if 'text' in dir() else 'N/A'}", flush=True)
        return {
            "verdict": "unverifiable",
            "confidence": 0.3,
            "reasoning": "Unable to parse claim due to processing error. Defaulting to unverifiable.",
            "key_evidence": ["Processing error occurred", "Could not analyze claim", "Default response used"],
        }
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return {
            "verdict": "unverifiable",
            "confidence": 0.3,
            "reasoning": "Model request failed. Defaulting to unverifiable verdict.",
            "key_evidence": ["API error", "Could not reach model", "Default fallback response"],
        }


# ─────────────────────────────────────────────
#  Env HTTP Client
# ─────────────────────────────────────────────

def env_reset(task_name: str) -> dict:
    resp = requests.post(f"{SPACE_URL}/reset", json={"task_name": task_name}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_step(task_name: str, action: dict) -> dict:
    payload = {"task_name": task_name, **action}
    resp = requests.post(f"{SPACE_URL}/step", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ─────────────────────────────────────────────
#  Main Episode Runner
# ─────────────────────────────────────────────

def run_task(client: OpenAI, task_name: str) -> dict:
    """Run a single task episode. Returns summary dict."""
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    try:
        reset_data = env_reset(task_name)
        observation = reset_data["observation"]

        for step in range(1, MAX_STEPS + 1):
            if observation.get("claim") == "[EPISODE COMPLETE]":
                break

            action = get_model_action(client, observation)
            action_str = f"verdict={action['verdict']} conf={action['confidence']:.2f}"

            try:
                result = env_step(task_name, action)
            except Exception as e:
                log_step(step=step, action=action_str, reward=0.0, done=True, error=str(e))
                break

            reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", False))
            error = None

            rewards.append(reward)
            steps_taken = step
            observation = result.get("observation", observation)

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        score = sum(rewards) / max(len(rewards), 1) if rewards else 0.0
        score = round(min(max(score, 0.0), 1.0), 3)
        success = score >= 0.3  # at least 30% average across claims

    except Exception as exc:
        print(f"[DEBUG] Task error: {exc}", flush=True)

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task": task_name, "success": success, "score": score, "rewards": rewards, "steps": steps_taken}


def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print(f"[INFO] Running FakeNews Detector inference on {SPACE_URL}", flush=True)
    print(f"[INFO] Model: {MODEL_NAME}", flush=True)
    print(f"[INFO] Tasks: {TASKS}", flush=True)
    print("", flush=True)

    all_results = []
    for task_name in TASKS:
        result = run_task(client, task_name)
        all_results.append(result)
        print("", flush=True)

    # Summary
    total_score = sum(r["score"] for r in all_results) / len(all_results)
    total_success = all(r["success"] for r in all_results)
    print(f"[SUMMARY] overall_score={total_score:.3f} all_tasks_success={str(total_success).lower()}", flush=True)


if __name__ == "__main__":
    main()
