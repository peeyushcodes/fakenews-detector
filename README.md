# 📰 FakeNews Detector — OpenEnv Environment

> An AI agent environment for fact-checking news claims with structured reasoning and evidence citation.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-green)](https://openenv.ai)
[![Difficulty](https://img.shields.io/badge/Tasks-Easy%20→%20Hard-blue)]()
[![HF Space](https://img.shields.io/badge/HuggingFace-Space-yellow)]()

---

## 🎯 Motivation

Misinformation is one of the most pressing challenges of our time. Yet there is **no standardized RL environment** for training or evaluating AI agents on fact-checking tasks. This environment fills that gap.

An AI agent that can:
- Distinguish fabricated claims from real ones
- Detect subtle misleading framing
- Reason about nuanced scientific/political claims

...is genuinely useful for building media literacy tools, content moderation systems, and next-generation fact-checking assistants.

---

## 🏗️ Environment Description

The agent is presented with a **news claim**, **context snippet**, and **claimed source**. It must return:

| Field | Type | Description |
|-------|------|-------------|
| `verdict` | `"true"/"false"/"misleading"/"unverifiable"` | The fact-check result |
| `confidence` | `float [0.0–1.0]` | How certain the agent is |
| `reasoning` | `string (min 20 words)` | Explanation of the verdict |
| `key_evidence` | `list[2–4 strings]` | Evidence points used |

---

## 📋 Tasks

### Task 1: `obvious_fake` (Easy)
**Difficulty score: 0.2–0.3**

Clearly fabricated or satirical claims. Tests baseline fact-checking ability.

Examples:
- "Scientists discover coffee cures all cancers"
- "NASA confirms Moon is made of cheese"
- "Eiffel Tower built in 1066 by Napoleon for aliens"

**Expected agent score:** 0.85+ for strong models, 0.5+ for baseline

---

### Task 2: `subtle_misinfo` (Medium)
**Difficulty score: 0.6–0.7**

Partially true claims that are misleading, out of context, or misrepresent causation.

Examples:
- "Study shows breakfast eaters are healthier → therefore skipping breakfast causes disease" (correlation ≠ causation)
- Crime statistics cited selectively to support a policy claim
- Bill Gates "predicted" COVID-19 (he warned about pandemics, not this specifically)

**Expected agent score:** 0.65+ for strong models, 0.3+ for baseline

---

### Task 3: `complex_claim` (Hard)
**Difficulty score: 0.75–0.9**

Nuanced claims requiring domain expertise in science, economics, or politics.

Examples:
- EVs vs gasoline cars carbon footprint (lifecycle analysis needed)
- Vaccine-autism claim (requires knowing Wakefield fraud + meta-study consensus)
- Climate "pause" claim (requires statistical literacy + understanding El Niño baseline manipulation)

**Expected agent score:** 0.55+ for frontier models, 0.15+ for baseline

---

## 🏆 Reward Function

Each step returns a reward in `[0.0, 1.0]` with **partial credit**:

| Component | Weight | Criteria |
|-----------|--------|----------|
| Verdict correct | **40%** | Matches ground truth label |
| Confidence calibrated | **20%** | Appropriate certainty for difficulty level |
| Reasoning quality | **25%** | Word count + key domain terms mentioned |
| Evidence quality | **15%** | Number of valid, substantive evidence points |

This ensures agents are rewarded for:
- **Knowing what they don't know** (appropriate confidence)
- **Showing their work** (reasoning quality)
- **Using evidence** (not just guessing)

---

## 🚀 Setup & Usage

### Local Development

```bash
# Clone the repo
git clone https://huggingface.co/spaces/YOUR_USERNAME/fakenews-detector
cd fakenews-detector

# Install dependencies
pip install -r requirements.txt

# Run the server
python server.py

# Test it
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "obvious_fake"}'
```

### Docker

```bash
docker build -t fakenews-env .
docker run -p 7860:7860 fakenews-env
```

---

## 🔌 API Reference

### `POST /reset`
Start a new episode.

```json
Request:  {"task_name": "obvious_fake"}
Response: {"observation": {...}, "task_name": "obvious_fake"}
```

### `POST /step`
Submit an action.

```json
Request:
{
  "task_name": "obvious_fake",
  "verdict": "false",
  "confidence": 0.95,
  "reasoning": "This claim is clearly fabricated because...",
  "key_evidence": [
    "No peer-reviewed study supports this claim",
    "Source is an unverified conspiracy blog",
    "Claims contradict established medical consensus"
  ]
}

Response:
{
  "observation": {...},
  "reward": 0.87,
  "done": false,
  "info": {
    "correct_verdict": true,
    "confidence_appropriate": true,
    "reasoning_words": 45,
    "evidence_count": 3,
    "partial_scores": {"verdict": 0.40, "confidence": 0.18, "reasoning": 0.20, "evidence": 0.15}
  }
}
```

### `GET /state`
Get current environment state.

### `GET /health`
Health check — returns 200 if running.

---

## 🤖 Running the Baseline Inference Script

```bash
export HF_TOKEN=hf_your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export SPACE_URL=https://your-space.hf.space

python inference.py
```

### Expected Output Format

```
[START] task=obvious_fake env=fakenews-detector model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=verdict=false conf=0.95 reward=0.87 done=false error=null
[STEP] step=2 action=verdict=false conf=0.92 reward=0.90 done=false error=null
[STEP] step=3 action=verdict=false conf=0.88 reward=0.85 done=true error=null
[END] success=true steps=3 score=0.873 rewards=0.87,0.90,0.85
```

---

## 📊 Baseline Scores

| Task | Model | Avg Score |
|------|-------|-----------|
| obvious_fake | Qwen2.5-72B | ~0.82 |
| subtle_misinfo | Qwen2.5-72B | ~0.61 |
| complex_claim | Qwen2.5-72B | ~0.48 |

---

## 📁 Project Structure

```
fakenews-detector/
├── fakenews_env.py     # Core environment (tasks, graders, models)
├── server.py           # FastAPI HTTP server
├── inference.py        # Baseline inference script
├── openenv.yaml        # OpenEnv spec metadata
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container definition
└── README.md           # This file
```

---

## 🧠 Design Decisions

1. **Partial rewards over binary**: Agents get credit for good reasoning even with wrong verdicts — this provides richer training signal.

2. **Confidence calibration scoring**: Penalizes overconfidence on wrong answers, rewards epistemic humility on hard cases.

3. **Evidence requirement**: Forces agents to ground their verdicts in specific reasoning, not just pattern-match to expected outputs.

4. **3-tier difficulty**: Easy cases are obvious to humans; hard cases challenge frontier models on domain knowledge.
