# Reasoning Models & Inference Scaling — The New Paradigm

> June 2026

The most significant architectural shift in LLMs since the transformer itself: **inference-time compute scaling**. Rather than just training bigger models, reasoning models spend more compute at inference to think through problems step by step, achieving dramatically better performance on math, science, coding, and logic.

---

## 1. The Reasoning Revolution

### 1.1 What Changed

In late 2024, OpenAI's o1 model introduced a fundamentally new paradigm: **extended thinking** at inference time. Unlike traditional LLMs that generate the first plausible token sequence, reasoning models:

1. Explore multiple solution paths internally
2. Backtrack and correct when stuck
3. Verify and reflect on intermediate steps
4. Allocate more compute for harder problems
5. Produce visible or hidden "chains of thought"

This was followed by:
- **OpenAI o3 / o4** (2025) — Further scaling of reasoning, tool use
- **DeepSeek-R1** (Jan 2025) — Open-source reasoning model, MIT license
- **Claude 3.5 Opus + extended thinking** (2025) — Extended thinking mode
- **Gemini 2.5 Pro / Flash Thinking** (2025–2026) — Google's reasoning models
- **Qwen3 / QwQ** — Alibaba's open reasoning model
- **DeepSeek-V3 / R2** (2026) — Hybrid reasoning + general model

### 1.2 Why It Matters

| Benchmark | GPT-4o (no reasoning) | o4 (with reasoning) | Improvement |
|-----------|----------------------|---------------------|-------------|
| MATH-500 | 76% | 97% | +21pp |
| GPQA Diamond | 56% | 87% | +31pp |
| AIME 2025 | 12% | 93% | +81pp |
| LiveCodeBench | 38% | 75% | +37pp |
| ARC-AGI-2 | 8% | 87% | +79pp |

---

## 2. Architecture of Reasoning Models

### 2.1 The Scaling Law of Inference

Traditional scaling laws are about **training compute** (FLOPs for pretraining). Reasoning models introduce a second axis: **inference-time compute**.

```
Performance = f( training_compute , inference_compute )
                                ↑            ↑
                     Traditional axis    New axis: "thinking tokens"
```

### 2.2 Chain-of-Thought (CoT) Reasoning

**Core idea**: Generate intermediate reasoning steps before the final answer.

```
Standard LLM:
Input: "How many Rs are in the word strawberry?"
Output: "2" ❌ (guesses based on pattern matching)

Reasoning Model:
Input: "How many Rs are in the word strawberry?"
Thinking: "Let me count: s-t-r-a-w-b-e-r-r-y. 
           Positions: 1:s, 2:t, 3:r, 4:a, 5:w, 6:b, 7:e, 8:r, 9:r, 10:y. 
           The letter 'r' appears at positions 3, 8, and 9. That's 3 occurrences."
Output: "3" ✅
```

### 2.3 DeepSeek-R1 Architecture

DeepSeek-R1 was the first fully open reasoning model, trained via **reinforcement learning** to discover reasoning patterns:

```
Stage 1: Cold-Start Fine-Tuning
├── Collect thousands of high-quality CoT examples
├── Fine-tune DeepSeek-V3-Base on CoT data
└── Gives model initial reasoning ability

Stage 2: Reinforcement Learning (RL)
├── Use GRPO (Group Relative Policy Optimization)
├── Reward: correctness + format compliance + thinking token efficiency
├── Model discovers emergent behaviors: backtracking, self-verification
└── LANGUAGE MIXING: Model spontaneously switches to English for reasoning
     (trained primarily on Chinese, "thinks" in English!)

Stage 3: Rejection Sampling
├── Generate thousands of outputs per prompt
├── Select highest-quality (correct + well-reasoned)
└── Fine-tune on selected samples

Stage 4: Supervised Fine-Tuning (SFT)
├── Combine rejection-sampled data with general SFT data
├── Final model: DeepSeek-R1
└── Distill into smaller models: R1-Distill-1.5B, 7B, 8B, 14B, 32B, 70B
```

### 2.4 Key Technical Innovations

| Technique | Description | Used By |
|-----------|-------------|---------|
| **GRPO** (Group Relative Policy Optimization) | No critic model needed, group-based advantage estimation | DeepSeek-R1 |
| **Extended Thinking Tokens** | Model learns to use special reasoning tokens internally | o3, o4, R1 |
| **Tool-Integrated Reasoning** | Model can call search, code execution, calculators mid-reasoning | o4, Claude extended thinking |
| **Constitutional Reasoning** | Safety constraints embedded in thinking process | Claude extended thinking |
| **Self-Consistency Decoding** | Sample multiple reasoning paths, vote on final answer | Gemini 2.5 Flash |
| **Monte Carlo Tree Search (MCTS)** | Tree search over reasoning steps at inference | o3, AlphaProof |
| **Budget Forcing** | Control compute spent per problem | o3, R1 |

---

## 3. Inference Scaling Strategies

### 3.1 Compute Budgets

Models can operate at multiple compute levels:

| Level | Tokens Used | Example Problem | Cost |
|-------|-------------|-----------------|------|
| **Low** | 100–500 | Simple QA | $0.01 |
| **Medium** | 1K–10K | Competition math | $0.10 |
| **High** | 10K–50K | Research-level science | $0.50 |
| **Extreme** | 50K–1M | Novel theorem proving | $5.00+ |

### 3.2 Adaptive Compute Allocation

Models can **self-judge** how much compute a problem deserves:

```
def adaptive_reasoning(prompt, max_budget=10000):
    # Quick scan — is this a simple question?
    difficulty_estimate = model.estimate_difficulty(prompt)
    if difficulty_estimate < 0.3:
        return direct_answer(prompt)
    
    # Moderate — allocate more tokens
    if difficulty_estimate < 0.7:
        return full_reasoning(prompt, budget=max_budget * 0.3)
    
    # Hard — go deep
    return full_reasoning(prompt, budget=max_budget)
```

### 3.3 Parallel Thinking (PVE)

**Process-Verified Ensembles** — Sample N reasoning paths, verify each, pick best:

```
Prompt → Path 1 (reasoning) → Process Verifier → Score
       → Path 2 (reasoning) → Process Verifier → Score
       → Path N (reasoning) → Process Verifier → Score
                          ↓
              Argmax over verified paths
```

---

## 4. Model Comparison (June 2026)

### 4.1 Reasoning Model Landscape

| Model | Maker | Open? | Thinking Visible? | Best For |
|-------|-------|-------|-------------------|----------|
| **o4** | OpenAI | ❌ | ✗ (hidden) | Hard math, science, research |
| **o4-mini** | OpenAI | ❌ | ✗ (hidden) | Coding, cost-sensitive reasoning |
| **Claude Extended Thinking** | Anthropic | ❌ | ✓ (visible) | Analysis, writing, safety-critical |
| **Gemini 2.5 Pro (Thinking)** | Google | ❌ | ✓ | Long-context reasoning, multimodal |
| **Gemini 2.5 Flash (Thinking)** | Google | ❌ | ✓ | Fast, cost-effective reasoning |
| **DeepSeek-R1** | DeepSeek | ✓ (MIT) | ✓ | Open research, fine-tuning |
| **DeepSeek-R2** | DeepSeek | ✓ (MIT) | ✓ | Hybrid reasoning + general (2026) |
| **QwQ-32B-Preview** | Alibaba (Qwen) | ✓ | ✓ | Open reasoning, math |
| **Qwen3-235B** | Alibaba (Qwen) | ✓ (Apache 2.0) | ✓ | Hybrid reasoning + general |

### 4.2 Performance Benchmarks

| Benchmark | GPT-4o | DeepSeek-R1 | Gemini 2.5 Pro | o4 | Claude Extended |
|-----------|--------|-------------|----------------|-----|-----------------|
| AIME 2025 | 12% | 79.8% | 83% | 93% | 89% |
| MATH-500 | 76% | 97.3% | 97% | 98% | 96% |
| GPQA Diamond | 56% | 71.5% | 82% | 87% | 84% |
| SWE-bench Verified | 38% | 49.2% | 63% | 75% | 70% |
| ARC-AGI-2 | 8% | 68% | 79% | 87% | 82% |
| HLE (Humanity's Last Exam) | 3% | 29% | 35% | 43% | 38% |

---

## 5. Applications & Use Cases

### 5.1 Mathematics & Science
- **Competition math**: AIME, IMO, Putnam problems solved at gold medal level
- **Theoretical physics**: Deriving equations, checking consistency
- **Biology**: Protein folding analysis, pathway reasoning
- **Chemistry**: Reaction mechanism prediction

### 5.2 Software Engineering
- **Complex debugging**: Trace through multiple files, infer causality
- **Architecture reasoning**: Design decisions with trade-off analysis
- **SWE-bench**: Solving real GitHub issues end-to-end

### 5.3 Research & Analysis
- **Literature review**: Cross-paper reasoning, contradiction detection
- **Hypothesis generation**: Novel combinations of known results
- **Proof verification**: Checking mathematical proofs step by step

### 5.4 Safety & Alignment
- **Adversarial robustness**: Thinking through potential attacks
- **Constitutional reasoning**: Explicitly reasoning about safety constraints
- **Value alignment**: Modeling downstream impacts of decisions

---

## 6. Distillation & Small Reasoning Models

A key finding from DeepSeek-R1: reasoning can be **distilled** into small models:

| Distilled Model | Base | Performance on MATH-500 |
|-----------------|------|------------------------|
| R1-Distill-1.5B | Qwen2.5-Math-1.5B | 85% |
| R1-Distill-7B | Qwen2.5-Math-7B | 93% |
| R1-Distill-8B | Llama-3.1-8B | 89% |
| R1-Distill-14B | Qwen2.5-14B | 94% |
| R1-Distill-32B | Qwen2.5-32B | 95% |
| R1-Distill-70B | Llama-3.3-70B | 96% |

This means a **1.5B model** can outperform GPT-4o on math — a 50× efficiency gain.

---

## 7. Open Research Questions

1. **Scaling limits**: Does thinking compute have diminishing returns beyond a threshold?
2. **Overthinking**: Models can waste compute on simple problems — how to calibrate?
3. **Transparency vs. safety**: Hidden thinking prevents monitoring, visible thinking reveals vulnerabilities
4. **Multimodal reasoning**: Extending chain-of-thought to images, video, audio
5. **World model reasoning**: Can thinking lead to genuine causal understanding?
6. **Agentic reasoning**: Combining extended thinking with tool use and memory
7. **Reward hacking**: RL-trained reasoners may find shortcuts rather than genuine understanding

---

*This document is part of the AI Knowledge Library — 29-Reasoning-and-Inference-Scaling directory.*