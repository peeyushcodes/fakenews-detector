"""
Inference Script — FakeNews Detector OpenEnv
=============================================
Uses an OpenAI-compatible client + Tavily (web search) for real-time fact-checking.

Every claim is first searched on the web via Tavily,
then the LLM analyzes claim + search results to give a verdict.
"""

import json
import os
import textwrap
from typing import List, Optional

import requests
import openai
from tavily import TavilyClient

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────
API_BASE_URL   = os.getenv("API_BASE_URL")
API_KEY        = os.getenv("API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
MODEL_NAME     = os.getenv("MODEL_NAME", "gpt-4o-mini")
SPACE_URL      = os.getenv("SPACE_URL", "http://localhost:7860").rstrip("/")

BENCHMARK  = "fakenews-detector"
MAX_STEPS  = 5
TASKS      = ["obvious_fake", "subtle_misinfo", "complex_claim"]

# ─────────────────────────────────────────────
#  System Prompt for LLM
# ─────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
You are an expert fact-checker and investigative journalist with access to real web search results.

Your job is to analyze news claims using the provided web search evidence and return a verdict.

You MUST respond with ONLY a valid JSON object with these exact fields:
{
  "verdict": "<one of: true, false, misleading, unverifiable>",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<your detailed analysis using the search results, minimum 50 words>",
  "key_evidence": ["<evidence point 1>", "<evidence point 2>", "<evidence point 3>"]
}

Verdict definitions:
- "true"         : The claim is accurate and well-supported by evidence
- "false"        : The claim is factually incorrect or fabricated
- "misleading"   : Partially true but distorted, out of context, or leads to false conclusions
- "unverifiable" : Cannot be verified with available information

Rules:
- Use the web search results provided to ground your verdict in real evidence
- Reasoning must be at least 50 words
- Always provide exactly 3 key evidence points drawn from search results
- Cite specific facts from the search results in your evidence points
- Output ONLY the JSON object — no preamble, no markdown, no explanation outside JSON
""").strip()


# ─────────────────────────────────────────────
#  Logging — EXACT required format
# ─────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    action_log = action[:120].replace("\n", " ")
    print(f"[STEP] step={step} action={action_log} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


# ─────────────────────────────────────────────
#  Step 1: Tavily Web Search
# ─────────────────────────────────────────────

def search_web(tavily, claim: str) -> str:
    """
    Search the web for evidence about the claim.
    Returns a formatted string of search results to feed into LLM.
    """
    if not tavily:
        return "Web search unavailable. Please evaluate using internal knowledge and reasoning."
        
    try:
        query = claim[:200]  # Keep it focused

        response = tavily.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_answer=True,
        )

        parts = []

        if response.get("answer"):
            parts.append(f"SEARCH SUMMARY: {response['answer']}")

        results = response.get("results", [])
        for i, r in enumerate(results, 1):
            title   = r.get("title", "Unknown Source")
            url     = r.get("url", "")
            content = r.get("content", "")[:400]  # limit per result
            score   = r.get("score", 0)
            parts.append(
                f"SOURCE {i} (relevance: {score:.2f}): {title}\n"
                f"URL: {url}\n"
                f"EXCERPT: {content}"
            )

        if not parts:
            return "No web search results found for this claim."

        return "\n\n".join(parts)

    except Exception as exc:
        print(f"[DEBUG] Tavily search failed: {exc}", flush=True)
        return f"Web search unavailable: {exc}"


# ─────────────────────────────────────────────
#  Step 2: LLM Fact-Check
# ─────────────────────────────────────────────

def get_llm_verdict(client, observation: dict, search_results: str) -> dict:
    """
    Send claim + web search results to the LLM via OpenAI API and get a structured verdict.
    """
    previous = ""
    if observation.get("previous_feedback"):
        previous = f"\nPrevious feedback: {observation['previous_feedback']}"

    user_prompt = textwrap.dedent(f"""
    CLAIM TO FACT-CHECK:
    {observation['claim']}

    ORIGINAL CONTEXT PROVIDED:
    {observation['context']}

    CLAIMED SOURCE:
    {observation['source']}
    {previous}

    ═══════════════════════════════════════
    REAL-TIME WEB SEARCH RESULTS:
    ═══════════════════════════════════════
    {search_results}
    ═══════════════════════════════════════

    Based on the claim AND the web search results above, provide your fact-check verdict.
    Remember: respond ONLY with the JSON object.
    """).strip()

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=600,
            response_format={"type": "json_object"}
        )

        text = response.choices[0].message.content.strip()

        # Strip markdown code blocks if LLM wraps in ```json
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        action = json.loads(text)

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
            evidence.append("Insufficient evidence in search results.")

        return {
            "verdict":      verdict,
            "confidence":   confidence,
            "reasoning":    reasoning,
            "key_evidence": evidence,
        }

    except json.JSONDecodeError as exc:
        print(f"[DEBUG] JSON parse failed: {exc} | Raw: {text[:300] if 'text' in dir() else 'N/A'}", flush=True)
        return _fallback_action("JSON parsing failed")
    except Exception as exc:
        print(f"[DEBUG] LLM request failed: {exc}", flush=True)
        return _fallback_action(str(exc))


def _fallback_action(reason: str) -> dict:
    """"""
    return {
        "verdict":      "unverifiable",
        "confidence":   0.2,
        "reasoning":    (
            f"Unable to complete fact-check due to processing error ({reason}). "
            "The claim requires further verification from reliable independent sources. "
            "Multiple corroborating sources should be consulted before drawing conclusions."
        ),
        "key_evidence": [
            "Processing error prevented full analysis",
            "Web search results could not be fully evaluated",
            "Manual verification recommended from authoritative sources",
        ],
    }


# ─────────────────────────────────────────────
#  Combined: Search + Fact-check
# ─────────────────────────────────────────────

def get_model_action(client, tavily: TavilyClient, observation: dict) -> dict:
    claim = observation.get("claim", "")

    print(f"[DEBUG] Searching web for: {claim[:80]}...", flush=True)
    search_results = search_web(tavily, claim)
    print(f"[DEBUG] Got {len(search_results)} chars of search results", flush=True)

    print(f"[DEBUG] Sending to LLM for analysis...", flush=True)
    action = get_llm_verdict(client, observation, search_results)
    print(f"[DEBUG] LLM verdict: {action['verdict']} (confidence: {action['confidence']:.2f})", flush=True)

    return action


# ─────────────────────────────────────────────
#  Environment HTTP Client
# ─────────────────────────────────────────────

def env_reset(task_name: str) -> dict:
    resp = requests.post(f"{SPACE_URL}/reset", json={"task_name": task_name}, timeout=60)
    resp.raise_for_status()
    return resp.json()


def env_step(task_name: str, action: dict) -> dict:
    payload = {"task_name": task_name, **action}
    resp = requests.post(f"{SPACE_URL}/step", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


# ─────────────────────────────────────────────
#  Episode Runner
# ─────────────────────────────────────────────

def run_task(client, tavily: TavilyClient, task_name: str) -> dict:
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    rewards:     List[float] = []
    steps_taken: int         = 0
    success:     bool        = False
    score:       float       = 0.0

    try:
        reset_data  = env_reset(task_name)
        observation = reset_data["observation"]

        for step in range(1, MAX_STEPS + 1):
            if observation.get("claim") == "[EPISODE COMPLETE]":
                break

            action     = get_model_action(client, tavily, observation)
            action_str = f"verdict={action['verdict']} conf={action['confidence']:.2f}"

            try:
                result = env_step(task_name, action)
            except Exception as e:
                log_step(step=step, action=action_str, reward=0.0, done=True, error=str(e)[:100])
                steps_taken = step
                break

            reward = float(result.get("reward", 0.0))
            done   = bool(result.get("done",   False))

            rewards.append(reward)
            steps_taken = step
            observation = result.get("observation", observation)

            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            if done:
                break

        score   = sum(rewards) / max(len(rewards), 1) if rewards else 0.0
        score   = round(min(max(score, 0.01), 0.99), 3)
        success = score >= 0.3

    except Exception as exc:
        print(f"[DEBUG] Task error: {exc}", flush=True)
        if not rewards:
            steps_taken = 1
            rewards     = [0.0]

    log_end(success=success, steps=steps_taken, rewards=rewards)
    return {
        "task":    task_name,
        "success": success,
        "score":   score,
        "rewards": rewards,
        "steps":   steps_taken,
    }


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    if "API_KEY" not in os.environ or "API_BASE_URL" not in os.environ:
        print("[ERROR] API_KEY or API_BASE_URL not set in os.environ.", flush=True)
        # Proceed anyway to attempt fallback or let it crash on openai init
        
    if not TAVILY_API_KEY:
        print("[WARNING] TAVILY_API_KEY not set. Proceeding without web search.", flush=True)

    try:
        client = openai.OpenAI(base_url=os.environ["API_BASE_URL"], api_key=os.environ["API_KEY"])
        print(f"[INFO] OpenAI client loaded: {MODEL_NAME}", flush=True)
    except Exception as exc:
        print(f"[ERROR] LLM init failed: {exc}", flush=True)
        for task_name in TASKS:
            log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
            log_step(step=1, action="llm_init_failed", reward=0.0, done=True, error=str(exc)[:100])
            log_end(success=False, steps=1, rewards=[0.0])
        return

    try:
        tavily = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None
        print(f"[INFO] Tavily client ready", flush=True)
    except Exception as exc:
        print(f"[WARNING] Tavily init failed: {exc}. Proceeding without web search.", flush=True)
        tavily = None

    print(f"[INFO] Space URL  : {SPACE_URL}", flush=True)
    print(f"[INFO] Tasks      : {TASKS}", flush=True)
    print(f"[INFO] Pipeline   : Claim → Tavily Web Search → OpenAI Analysis → Verdict", flush=True)

    all_results = []
    for task_name in TASKS:
        print(f"\n[INFO] ═══ Starting task: {task_name} ═══", flush=True)
        result = run_task(client, tavily, task_name)
        all_results.append(result)
        print(f"[INFO] Task {task_name} done — score: {result['score']:.3f}", flush=True)

    total_score   = sum(r["score"] for r in all_results) / len(all_results)
    total_success = all(r["success"] for r in all_results)
    print(f"\n[SUMMARY] overall_score={total_score:.3f} all_tasks_success={str(total_success).lower()}", flush=True)

if __name__ == "__main__":
    main()
