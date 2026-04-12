"""
FakeNews Detector OpenEnv Environment
======================================
An environment where an AI agent must fact-check news claims and assign credibility scores.

Tasks:
- Task 1 (easy):   obvious_fake    — clearly fabricated headlines
- Task 2 (medium): subtle_misinfo  — partially true but misleading claims
- Task 3 (hard):   complex_claim   — nuanced scientific/political claims requiring deep reasoning
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
#  Pydantic Models
# ─────────────────────────────────────────────

class FakeNewsObservation(BaseModel):
    claim: str = Field(..., description="The news claim to fact-check")
    context: str = Field(..., description="Additional context or article snippet")
    source: str = Field(..., description="Claimed source of the news")
    task_name: str = Field(..., description="Current task name")
    step: int = Field(..., description="Current step number")
    max_steps: int = Field(..., description="Maximum steps allowed")
    previous_feedback: Optional[str] = Field(None, description="Feedback from previous step if any")


class FakeNewsAction(BaseModel):
    verdict: str = Field(..., description="One of: 'true', 'false', 'misleading', 'unverifiable'")
    confidence: float = Field(..., description="Confidence score 0.0-1.0")
    reasoning: str = Field(..., description="Explanation of your verdict (min 20 words)")
    key_evidence: List[str] = Field(..., description="List of 2-4 key pieces of evidence used")


class FakeNewsReward(BaseModel):
    score: float = Field(..., description="Reward score 0.0-1.0")
    verdict_correct: bool
    confidence_appropriate: bool
    reasoning_quality: float
    feedback: str


class StepResult(BaseModel):
    observation: FakeNewsObservation
    reward: float
    done: bool
    info: Dict[str, Any]


# ─────────────────────────────────────────────
#  Task Definitions
# ─────────────────────────────────────────────

TASKS = {
    "obvious_fake": {
        "difficulty": "easy",
        "description": "Identify clearly fabricated or satirical news claims",
        "max_steps": 3,
        "claims": [
            {
                "claim": "Scientists discover that drinking 10 cups of coffee daily cures all cancers permanently",
                "context": "A viral post claims researchers at 'Global Health Institute' found that caffeine destroys every type of cancer cell. The post has no citations and uses emotional language like 'THEY DON'T WANT YOU TO KNOW THIS'.",
                "source": "HealthMiracle.net (unverified blog)",
                "correct_verdict": "false",
                "explanation": "No credible scientific study supports this. Legitimate cancer cures go through rigorous peer review. Red flags: sensational language, no citations, unverified source.",
                "difficulty_score": 0.3,
            },
            {
                "claim": "The Eiffel Tower was built in 1066 by Napoleon Bonaparte as a gift to aliens",
                "context": "A social media post claims historical records were 'hidden' showing Napoleon commissioned the Eiffel Tower in 1066. The post includes obviously photoshopped images.",
                "source": "ConspiracyFacts.io",
                "correct_verdict": "false",
                "explanation": "The Eiffel Tower was built in 1889 by Gustave Eiffel. Napoleon died in 1821. 1066 is the year of the Battle of Hastings. Multiple factual errors.",
                "difficulty_score": 0.2,
            },
            {
                "claim": "NASA confirms the Moon is made entirely of cheese after 50 years of covering it up",
                "context": "An article claims whistleblowers inside NASA have 'finally revealed' that moon landing samples were actually large cheese samples. Includes quotes from unnamed 'senior scientists'.",
                "source": "TruthBombs Daily",
                "correct_verdict": "false",
                "explanation": "Moon rocks are well-documented silicate minerals. This is a known internet joke/myth. NASA data, peer-reviewed geology, and multiple international space agencies confirm composition.",
                "difficulty_score": 0.2,
            },
        ],
    },

    "subtle_misinfo": {
        "difficulty": "medium",
        "description": "Detect partially true but misleading or out-of-context claims",
        "max_steps": 4,
        "claims": [
            {
                "claim": "Study shows people who eat breakfast are healthier — so skipping breakfast causes disease",
                "context": "A news article reports a study showing breakfast eaters have better health outcomes. The article headline reads 'Skipping Breakfast KILLS: New Study Proves It'. The actual study only found correlation, not causation, and did not measure disease causation.",
                "source": "DailyHealthNews.com",
                "correct_verdict": "misleading",
                "explanation": "The study exists but the conclusion is distorted. Correlation ≠ causation. Healthier people may eat breakfast for other reasons. The headline massively overstates findings.",
                "difficulty_score": 0.6,
            },
            {
                "claim": "Crime rate dropped 40% after new gun law was passed — proving gun control works",
                "context": "A politician claims crime fell 40% in their city after gun legislation. Statistics show crime did drop 40%, but the drop began 2 years before the law passed, coinciding with an economic boom and increased policing.",
                "source": "Senator's official press release",
                "correct_verdict": "misleading",
                "explanation": "The statistic is real but the causation is false. The crime drop predated the law. Multiple confounding factors (economy, policing) were ignored. Classic post hoc fallacy.",
                "difficulty_score": 0.65,
            },
            {
                "claim": "Bill Gates said in 2015 that a pandemic would kill millions — he predicted COVID-19",
                "context": "Viral posts claim Gates 'predicted and planned' COVID-19 based on his 2015 TED Talk about pandemic preparedness. The talk exists and does warn about pandemic risks, but makes no mention of a specific virus or date.",
                "source": "Multiple social media shares",
                "correct_verdict": "misleading",
                "explanation": "Gates did give the talk and warn about pandemics — that part is true. But framing it as 'predicting COVID' or 'planning it' is misleading. Epidemiologists have warned about pandemic risks for decades.",
                "difficulty_score": 0.7,
            },
        ],
    },

    "complex_claim": {
        "difficulty": "hard",
        "description": "Evaluate nuanced scientific, economic, or political claims requiring deep reasoning",
        "max_steps": 5,
        "claims": [
            {
                "claim": "Electric vehicles produce more CO2 than gasoline cars when accounting for battery manufacturing",
                "context": "An op-ed in an energy magazine claims that when you factor in lithium battery production, EVs have a larger carbon footprint than combustion engines over their lifetime. Cites a 2019 Volvo study showing EV production emits 70% more CO2 than a comparable ICE vehicle manufacturing.",
                "source": "Energy Policy Review (legitimate publication)",
                "correct_verdict": "misleading",
                "explanation": "The manufacturing claim has some truth — battery production is carbon intensive. However, lifecycle analyses consistently show EVs have lower total emissions over their lifetime (typically 50-70% less), especially as grids get cleaner. The claim cherry-picks manufacturing and ignores operational lifetime. Context-dependent on electricity grid source.",
                "difficulty_score": 0.85,
            },
            {
                "claim": "Vaccines cause autism — Andrew Wakefield's study proves the MMR connection",
                "context": "Anti-vaccine groups cite Wakefield's 1998 Lancet paper as proof that MMR vaccines cause autism. The paper was retracted in 2010. Wakefield lost his medical license. Over 20 large-scale studies with millions of children found no link.",
                "source": "Multiple anti-vaccine websites citing retracted Lancet paper",
                "correct_verdict": "false",
                "explanation": "The original study was fraudulent, retracted, and the author lost his medical license for ethical violations. Subsequent studies involving millions of children across multiple countries found no causal link between MMR and autism. Scientific consensus is clear.",
                "difficulty_score": 0.75,
            },
            {
                "claim": "The global average temperature has not increased in the last 15 years — global warming has paused",
                "context": "A climate skeptic blog from 2013 claims there has been no warming since 1998, citing raw temperature data showing 1998 as an unusually hot El Niño year. Climate scientists responded that cherry-picking 1998 (a statistical outlier) creates a false baseline, and the long-term trend is clearly upward.",
                "source": "ClimateSkeptic.org (2013 article being reshared)",
                "correct_verdict": "false",
                "explanation": "This is a well-documented statistical manipulation. 1998 was an extreme El Niño year. Cherry-picking it as baseline creates artificial 'pause'. NOAA, NASA, and all major climate bodies confirm continued warming. The 2010s were the hottest decade on record. The claim is now especially outdated.",
                "difficulty_score": 0.9,
            },
        ],
    },
}


# ─────────────────────────────────────────────
#  Grader Logic
# ─────────────────────────────────────────────

def grade_action(action: FakeNewsAction, claim_data: dict, step: int, max_steps: int) -> Tuple[float, str, dict]:
    """
    Grade an agent's fact-checking action.
    Returns (reward: float, feedback: str, info: dict)
    
    Scoring breakdown:
      - Correct verdict:          0.40
      - Confidence appropriate:   0.20
      - Reasoning quality:        0.25
      - Evidence quality:         0.15
    """
    score = 0.01  # minimum safe starting value, clamped again at end
    correct_verdict = action.verdict.lower() == claim_data["correct_verdict"].lower()
    
    # 1. Verdict correctness (0.40)
    if correct_verdict:
        score += 0.40
        verdict_feedback = "✅ Correct verdict!"
    else:
        verdict_feedback = f"❌ Wrong verdict. Expected '{claim_data['correct_verdict']}', got '{action.verdict}'."

    # 2. Confidence appropriateness (0.20)
    # High confidence on correct answer → full marks
    # High confidence on wrong answer → penalty
    # Low confidence always → partial marks (shows uncertainty awareness)
    difficulty = claim_data["difficulty_score"]
    confidence_appropriate = False
    if correct_verdict:
        # Appropriate confidence = confidence roughly matches certainty level
        expected_confidence = 1.0 - (difficulty * 0.3)  # harder → slightly lower expected confidence
        conf_delta = abs(action.confidence - expected_confidence)
        confidence_score = max(0.01, 0.20 - conf_delta * 0.20)
        if conf_delta < 0.25:
            confidence_appropriate = True
    else:
        # Wrong verdict with high confidence = bad
        if action.confidence > 0.8:
            confidence_score = 0.01  # small non-zero so score never hits exact 0
        elif action.confidence > 0.5:
            confidence_score = 0.05
        else:
            confidence_score = 0.10  # low confidence on wrong = still some awareness
    score += confidence_score

    # 3. Reasoning quality (0.25)
    reasoning_words = len(action.reasoning.split())
    reasoning_score = 0.01  # minimum safe starting value
    if reasoning_words >= 50:
        reasoning_score = 0.25
    elif reasoning_words >= 30:
        reasoning_score = 0.18
    elif reasoning_words >= 20:
        reasoning_score = 0.12
    elif reasoning_words >= 10:
        reasoning_score = 0.06
    
    # Bonus: reasoning contains key terms from explanation
    key_terms = _extract_key_terms(claim_data["explanation"])
    term_matches = sum(1 for t in key_terms if t.lower() in action.reasoning.lower())
    reasoning_score = min(0.25, reasoning_score + term_matches * 0.02)
    score += reasoning_score

    # 4. Evidence quality (0.15)
    evidence_score = 0.01  # minimum safe starting value
    num_evidence = len(action.key_evidence)
    if num_evidence >= 3:
        evidence_score = 0.15
    elif num_evidence == 2:
        evidence_score = 0.10
    elif num_evidence == 1:
        evidence_score = 0.05
    # Check evidence isn't just empty strings
    valid_evidence = [e for e in action.key_evidence if len(e.strip()) > 10]
    if len(valid_evidence) < num_evidence:
        evidence_score *= 0.5
    score += evidence_score

    score = round(min(max(score, 0.01), 0.99), 3)

    feedback_parts = [
        verdict_feedback,
        f"Reasoning quality: {reasoning_words} words → {reasoning_score:.2f}/0.25",
        f"Evidence pieces: {len(valid_evidence)} valid → {evidence_score:.2f}/0.15",
        f"Hint: {claim_data['explanation']}",
    ]
    feedback = " | ".join(feedback_parts)

    info = {
        "correct_verdict": correct_verdict,
        "confidence_appropriate": confidence_appropriate,
        "reasoning_words": reasoning_words,
        "evidence_count": len(valid_evidence),
        "partial_scores": {
            "verdict": 0.40 if correct_verdict else 0.01,
            "confidence": round(max(confidence_score, 0.01), 3),
            "reasoning": round(max(reasoning_score, 0.01), 3),
            "evidence": round(max(evidence_score, 0.01), 3),
        },
    }

    return score, feedback, info


def _extract_key_terms(explanation: str) -> List[str]:
    """Extract meaningful terms from explanation for reasoning quality check."""
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "and", "but", "or", "in", "on", "at", "to", "for", "of", "it", "this", "that"}
    words = re.findall(r'\b[a-zA-Z]{4,}\b', explanation)
    return [w for w in words if w.lower() not in stopwords][:10]


# ─────────────────────────────────────────────
#  Environment Class
# ─────────────────────────────────────────────

class FakeNewsEnv:
    """
    FakeNews Detector OpenEnv Environment.
    
    Usage:
        env = FakeNewsEnv(task_name="obvious_fake")
        obs = env.reset()
        result = env.step(action)
        state = env.state()
    """

    def __init__(self, task_name: str = "obvious_fake"):
        if task_name not in TASKS:
            raise ValueError(f"Unknown task '{task_name}'. Choose from: {list(TASKS.keys())}")
        self.task_name = task_name
        self.task = TASKS[task_name]
        self._claim_index = 0
        self._step = 0
        self._done = False
        self._rewards: List[float] = []
        self._total_claims = len(self.task["claims"])
        self._max_steps = self.task["max_steps"] * self._total_claims

    def reset(self) -> FakeNewsObservation:
        """Reset environment to initial state."""
        self._claim_index = 0
        self._step = 0
        self._done = False
        self._rewards = []
        return self._make_observation()

    def step(self, action: FakeNewsAction) -> StepResult:
        """Take a step: receive action, return observation + reward."""
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        claim_data = self.task["claims"][self._claim_index]
        reward, feedback, info = grade_action(action, claim_data, self._step, self._max_steps)

        self._rewards.append(reward)
        self._step += 1
        self._claim_index += 1

        done = self._claim_index >= self._total_claims
        self._done = done

        if done:
            next_obs = self._make_observation(feedback=feedback, final=True)
        else:
            next_obs = self._make_observation(feedback=feedback)

        return StepResult(
            observation=next_obs,
            reward=reward,
            done=done,
            info=info,
        )

    def state(self) -> Dict[str, Any]:
        """Return current environment state."""
        return {
            "task_name": self.task_name,
            "difficulty": self.task["difficulty"],
            "step": self._step,
            "max_steps": self._max_steps,
            "claim_index": self._claim_index,
            "total_claims": self._total_claims,
            "done": self._done,
            "rewards_so_far": self._rewards,
            "cumulative_reward": sum(self._rewards),
        }

    def _make_observation(self, feedback: Optional[str] = None, final: bool = False) -> FakeNewsObservation:
        if final or self._claim_index >= self._total_claims:
            # Return a terminal observation
            return FakeNewsObservation(
                claim="[EPISODE COMPLETE]",
                context=f"You completed all {self._total_claims} claims. Total reward: {sum(self._rewards):.2f}",
                source="N/A",
                task_name=self.task_name,
                step=self._step,
                max_steps=self._max_steps,
                previous_feedback=feedback,
            )
        claim_data = self.task["claims"][self._claim_index]
        return FakeNewsObservation(
            claim=claim_data["claim"],
            context=claim_data["context"],
            source=claim_data["source"],
            task_name=self.task_name,
            step=self._step,
            max_steps=self._max_steps,
            previous_feedback=feedback,
        )
