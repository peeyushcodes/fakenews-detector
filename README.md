---
title: Fakenews Detector
emoji: 📰
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
tags:
  - openenv
---

# 📰 FakeNews Detector — OpenEnv Environment

> An AI agent environment for real-time fact-checking using web search + LLM reasoning.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-green)]()
[![Tasks](https://img.shields.io/badge/Tasks-3%20(Easy→Hard)-blue)]()
[![Reward](https://img.shields.io/badge/Reward-Partial%20Credit-orange)]()

---

## 🎯 Motivation

Misinformation is one of the most critical challenges of our time. Yet there is **no standardized RL environment** for training or evaluating AI agents on fact-checking tasks.

This environment fills that gap — agents must analyze real news claims, search for evidence, and produce structured verdicts with reasoning and citations.

---

## 🏗️ How It Works

```
Claim arrives
     ↓
Agent searches web (Tavily) for real evidence
     ↓
Agent reads claim + search results
     ↓
Agent returns: verdict + confidence + reasoning + evidence
     ↓
Environment scores the response (0.0 → 1.0)
```

---

## 📋 Tasks

| Task | Difficulty | Description |
|------|-----------|-------------|
| `obvious_fake` | Easy (0.2–0.3) | Clearly fabricated or satirical claims |
| `subtle_misinfo` | Medium (0.6–0.7) | Partially true but misleading claims |
| `complex_claim` | Hard (0.75–0.9) | Nuanced scientific/political claims |

### Task 1: `obvious_fake` (Easy)
Claims that are obviously false — tests baseline fact-checking.
- "Scientists discover coffee cures all cancers"
- "NASA confirms Moon is made of cheese"
- "Eiffel Tower built in 1066 by Napoleon for aliens"

### Task 2: `subtle_misinfo` (Medium)
Partially true but misleading claims requiring logical analysis.
- Correlation presented as causation
- Statistics used selectively to mislead
- Real events framed with false conclusions

### Task 3: `complex_claim` (Hard)
Nuanced claims requiring deep domain knowledge.
- EV vs gasoline carbon footprint (lifecycle analysis needed)
- Vaccine safety claims (requires knowledge of retracted studies)
- Climate data cherry-picking (requires statistical literacy)

---

## 🏆 Reward Function

Each step scored **0.0 → 1.0** with partial credit:

| Component | Weight | Criteria |
|-----------|--------|----------|
| Verdict correct | **40%** | Matches ground truth label |
| Confidence calibrated | **20%** | Appropriate certainty for difficulty |
| Reasoning quality | **25%** | Word count + key domain terms |
| Evidence quality | **15%** | Number of valid evidence points |

**Partial credit design:** Even a wrong verdict with excellent reasoning scores ~0.40, giving richer training signal than binary pass/fail.

**Score ranges:**
- Perfect agent: 0.96–0.99
- Good agent: 0.65–0.85
- Random agent: 0.13–0.25

---

## 🚀 Setup & Usage

### Local Development

```bash
git clone https://huggingface.co/spaces/Peeyushkumar5317/fakenews-detector
cd fakenews-detector
pip install -r requirements.txt
python server.py
```

### Docker

```bash
docker build -t fakenews-env .
docker run -p 7860:7860 fakenews-env
```

---

## 🔌 API Reference

### `POST /reset`
```json
Request:  {"task_name": "obvious_fake"}
Response: {
  "observation": {
    "claim": "...",
    "context": "...",
    "source": "...",
    "task_name": "obvious_fake",
    "step": 0,
    "max_steps": 9,
    "previous_feedback": null
  },
  "task_name": "obvious_fake"
}
```

### `POST /step`
```json
Request: {
  "task_name": "obvious_fake",
  "verdict": "false",
  "confidence": 0.95,
  "reasoning": "This claim is clearly fabricated because...",
  "key_evidence": [
    "No peer-reviewed study supports this claim",
    "Source is an unverified conspiracy blog",
    "Contradicts established scientific consensus"
  ]
}
Response: {
  "observation": {...},
  "reward": 0.90,
  "done": false,
  "info": {
    "correct_verdict": true,
    "confidence_appropriate": true,
    "reasoning_words": 45,
    "evidence_count": 3,
    "partial_scores": {
      "verdict": 0.40,
      "confidence": 0.19,
      "reasoning": 0.25,
      "evidence": 0.15
    }
  }
}
```

### `GET /health`
Returns `{"status": "ok", "environment": "fakenews-detector", "tasks": [...]}`

### `GET /tasks`
Returns all task descriptions with difficulty and metadata.

### `GET /state?task_name=obvious_fake`
Returns current episode state.

---

## 🤖 Running the Inference Script

```bash
export GEMINI_API_KEY=your_gemini_key
export TAVILY_API_KEY=your_tavily_key
export SPACE_URL=https://peeyushkumar5317-fakenews-detector.hf.space

python inference.py
```

### Expected Output

```
[START] task=obvious_fake env=fakenews-detector model=gemini-1.5-flash
[STEP] step=1 action=verdict=false conf=0.95 reward=0.90 done=false error=null
[STEP] step=2 action=verdict=false conf=0.92 reward=0.87 done=false error=null
[STEP] step=3 action=verdict=false conf=0.91 reward=0.88 done=true error=null
[END] success=true steps=3 rewards=0.90,0.87,0.88
```

---

## 📊 Baseline Scores

| Task | Agent | Avg Score |
|------|-------|-----------|
| obvious_fake | Gemini-1.5-flash + Tavily | ~0.90 |
| subtle_misinfo | Gemini-1.5-flash + Tavily | ~0.75 |
| complex_claim | Gemini-1.5-flash + Tavily | ~0.62 |

---

## 📁 Project Structure

```
fakenews-detector/
├── fakenews_env.py   # Core environment — tasks, graders, models
├── server.py         # FastAPI HTTP server
├── inference.py      # Gemini + Tavily inference script
├── openenv.yaml      # OpenEnv spec metadata
├── requirements.txt  # Python dependencies
├── Dockerfile        # Container definition
└── README.md         # This file
```

---

## 🧠 Design Decisions

1. **Partial rewards** — Richer training signal than binary pass/fail
2. **Confidence calibration** — Penalizes overconfidence on wrong answers
3. **Evidence requirement** — Forces grounded reasoning, not pattern matching
4. **Real web search** — Tavily gives agents access to live internet evidence
5. **3-tier difficulty** — Easy confirms agent works; hard challenges frontier models
