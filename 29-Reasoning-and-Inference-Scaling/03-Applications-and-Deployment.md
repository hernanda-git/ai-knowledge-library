# Applied Reasoning — Use Cases, Distillation & Deployment

> June 2026

This document covers practical applications of reasoning models, how to deploy them cost-effectively, and the art of prompt engineering for extended thinking.

---

## 1. Prompt Engineering for Reasoning Models

### 1.1 The Thinking Prompt

Reasoning models work best with explicit instruction to think:

```
❌ Bad for reasoning models:
"Solve this equation: 3x² + 5x - 2 = 0"

✅ Good for reasoning models:
"Let's solve step by step: 3x² + 5x - 2 = 0

First, identify the coefficients: a = 3, b = 5, c = -2
Apply the quadratic formula: x = (-b ± √(b² - 4ac)) / 2a
Compute the discriminant..."
```

### 1.2 Extended Thinking Parameters

| Parameter | Effect | Typical Range |
|-----------|--------|--------------|
| `thinking_budget` (OpenAI) | Max thinking tokens | 1K–20K |
| `reasoning_effort` (OpenAI) | Low/Medium/High compute | Auto-scaling |
| `extended_thinking` (Claude) | Enable/disable | Boolean |
| `temperature` | Lower for reasoning | 0.0–0.3 (vs 0.7–1.0 for creative) |
| `top_p` | Lower for precision | 0.9–1.0 |

### 1.3 Chain-of-Thought Templates

**For math/science**:
```
Problem: {problem}

Key information:
- What are we solving for?
- What constraints apply?
- What formulas might be relevant?

Solution approach:
1. Parse the problem
2. Apply relevant principles
3. Compute intermediate values
4. Verify intermediate results
5. Produce final answer

Reasoning:
```

**For code/debugging**:
```
Bug description: {bug}

Hypothesis:
- What could cause this?
- Which component is failing?

Investigation:
1. Trace the error path
2. Check state at each step
3. Identify root cause
4. Design fix
5. Verify fix doesn't break tests

Analysis:
```

---

## 2. Distillation for Production

### 2.1 Why Distill?

Full reasoning models (70B–500B+) are expensive. Distillation transfers reasoning ability to smaller, cheaper models:

| Model | Cost per 1M tokens | Latency | Quality (MATH-500) |
|-------|-------------------|---------|-------------------|
| o4 | $15/$60 | ~10s | 97% |
| DeepSeek-R1 (671B) | $2.19/$8.28 | ~8s | 97% |
| R1-Distill-70B | $0.75/$2.40 | ~2s | 96% |
| R1-Distill-32B | $0.35/$0.90 | ~1s | 95% |
| R1-Distill-7B | $0.15/$0.30 | ~0.5s | 93% |
| R1-Distill-1.5B | $0.02/$0.05 | ~0.2s | 85% |

### 2.2 Distillation Process

```
Stage 1: Generate Teacher Traces
├── Full reasoning model (teacher) generates N traces per problem
├── Keep correct traces with high process reward scores
└── Filter to diverse solution methods

Stage 2: Student SFT
├── Small model (student) trained on teacher traces
├── Cross-entropy loss on thinking tokens + final answer
└── No RL needed — just distillation

Stage 3: Optional RL Tuning
├── Lightweight GRPO to recover any lost reasoning
└── Significantly cheaper than training from scratch
```

### 2.3 Architecture Choices for Small Reasoners

| Size | Best Architecture | Use Case |
|------|------------------|----------|
| <3B | Encoder-decoder (T5-small) | Mobile, browser |
| 3–8B | Dense decoder (Qwen2.5, Llama 3) | Edge, latency-sensitive |
| 8–20B | Dense decoder | Cost-sensitive API |
| 20–70B | MoE or dense | Balanced production |

---

## 3. Cost Optimization

### 3.1 Pricing Comparison

```
Problem: "Prove that √2 is irrational"

Without reasoning (GPT-4o):
  - ~300 tokens output
  - Cost: $0.003
  - Quality: Good explanation, may skip steps

With reasoning (o4-mini):
  - ~2,000 thinking tokens + 200 answer tokens
  - Cost: $0.022 (7× more)
  - Quality: Rigorous proof, verified steps

With full reasoning (o4):
  - ~8,000 thinking tokens + 200 answer tokens
  - Cost: $0.48 (160× more)
  - Quality: Multiple approaches, self-verification
```

### 3.2 Optimization Strategies

| Strategy | Savings | Quality Impact |
|----------|---------|----------------|
| Adaptive budget (easy vs. hard) | 40–70% | None (on easy) |
| Best-of-N with small model | 50% | Small gain |
| Distilled small model | 90% | Moderate loss |
| Prompt caching (thinking prefix) | 30% | None |
| Batch processing | 30–50% | None |

---

## 4. Real-World Deployment Patterns

### 4.1 Tiered Reasoning Architecture

```
                    ┌──────────────┐
                    │  Request     │
                    └──────┬───────┘
                           ↓
               ┌───────────────────┐
               │ Difficulty Router  │
               │ (small classifier) │
               └────┬──────┬───────┘
                    │      │
             Easy   │      │  Hard
                    ↓      ↓
           ┌──────────┐ ┌──────────┐
           │ 1.5B R1  │ │ 32B R1   │
           │ Distill  │ │ Distill  │
           │ ~100ms   │ │ ~1.5s    │
           └──────────┘ └──────────┘
                    │      │
                    ↓      ↓
               ┌───────────────────┐
               │   Verifier (PRM)   │
               └───────────────────┘
                           │
                    Confidence ≥ threshold?
                    ├── Yes → Return
                    └── No → Escalate to larger model
```

### 4.2 Caching Strategies

- **Semantic caching**: Cache reasoning traces for similar problem templates
- **Partial reuse**: Cache common sub-problem solutions (factoring, derivative rules)
- **Intermediate value store**: Avoid recomputing verified steps

---

## 5. Case Studies

### 5.1 Automated Theorem Proving

**System**: AlphaProof + Gemini 2.5
**Achievement**: Silver medal at IMO 2024 (first AI to medal)
**Architecture**:
- Formal language (Lean) for verification
- MCTS over proof steps
- Gemini for natural language to Lean translation

### 5.2 AI Research Assistant

**System**: o4 + retrieval
**Use case**: Novel research hypothesis generation
**Workflow**:
1. Retrieve relevant papers from arXiv
2. Identify contradictions and gaps
3. Generate novel hypotheses
4. Design experiments to test
5. Write structured research proposal

### 5.3 Production Code Generation

**System**: o4-mini + SWE-suite
**Use case**: Automated bug fixing on real GitHub repos
**Pipeline**:
1. Parse issue description
2. Extract relevant files
3. Generate fix with explanation
4. Run tests
5. Iterate on failures

---

## 6. Tool-Integrated Reasoning

### 6.1 Architecture

Modern reasoning models can use tools **during** their thinking process:

```
Prompt → [Think] "I need to calculate..." → [Tool: Calculator] → 
[Think] "Now I need the current exchange rate..." → [Tool: Web Search] →
[Think] "Let me compute the final value..." → [Tool: Python] →
[Answer] "The total is $1,247.50"
```

### 6.2 Code Execution Loop

```python
def reasoning_with_code(prompt, max_iterations=10):
    state = """We need to solve: {prompt}

Let me write code to explore this..."""
    
    for _ in range(max_iterations):
        # Model generates reasoning + code
        output = model.generate(state)
        
        if "<ANSWER>" in output:
            return extract_answer(output)
        
        if "```python" in output:
            code = extract_code(output)
            result = execute(code)
            state += f"\nExecution result: {result}"
            state += "\n[Reasoning continues...]"
    
    return None
```

---

## 7. Future Trajectory (2026–2028)

| Timeline | Expected Advancement |
|----------|---------------------|
| **Late 2026** | Reasoning in native multimodal (images, video, audio) |
| **2027** | Real-time reasoning for agents (sub-second thinking) |
| **2027** | Specialist reasoning models for law, medicine, engineering |
| **2027–2028** | Open-source models matching frontier reasoning |
| **2028** | Reasoning + memory = persistent self-improving agents |

---

*This document is part of the AI Knowledge Library — 29-Reasoning-and-Inference-Scaling directory.*

---
**See also:**
- [03 — Green AI: Sustainable Practices for Model Development and Deployment](42-AI-for-Science-and-Drug-Discovery/35-AI-Energy-and-Sustainability/03-Green-AI.md)
- [Enterprise AI Deployment: Production Infrastructure and Operations](05-Enterprise/01-Enterprise-AI-Deployment.md)
- [Small Language Models — Efficiency, Edge Deployment & On-Device AI](30-Small-Language-Models/01-Overview-and-Efficiency.md)
