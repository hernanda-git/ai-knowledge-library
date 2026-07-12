# AI Product Management — Core Topics

> The practical craft of AI PM: choosing the right problems, running evaluation-driven development, making the prompt-vs-RAG-vs-fine-tune decision, building data flywheels, writing AI PRDs, and roadmapping under model uncertainty.

This document goes deep on the day-to-day decisions an AI PM makes. Each topic includes decision frameworks and concrete examples.

---

## 1. Choosing the Right AI Problems

Not every problem should be solved with AI. Use this scoring rubric before committing:

| Criterion | Question | Weight |
|-----------|----------|--------|
| Error tolerance | Can the workflow absorb occasional wrong answers? | High |
| Value density | Is the automated judgment expensive/slow when done by humans? | High |
| Data availability | Do we have (or can we get) representative data + eval? | High |
| Unstructured input | Does the task involve text/image/audio/ambiguity? | Medium |
| Defensibility | Does a data flywheel or proprietary data create a moat? | Medium |
| Reversibility | Are model actions reversible or reviewable? | High (for agents) |

**Green-light example:** Support-ticket triage. Errors are recoverable (human re-routes), volume is high, labeled history exists, and misclassification costs minutes not millions.

**Red-flag example:** Auto-approving loan disbursements with no review. Irreversible, regulated, low error tolerance, high blast radius.

### The "AI-shaped problem" test
A problem is AI-shaped when: (1) the input is fuzzy/natural, (2) a "good enough" answer beats no answer, and (3) getting from good-enough to great creates compounding value via a feedback loop.

---

## 2. Evaluation-Driven Development (EDD)

EDD is to AI PM what test-driven development is to software engineering. **You define the eval before you build.**

### The eval hierarchy
```
Level 0: Vibe checks        (manual, a few examples — prototyping only)
Level 1: Golden dataset     (curated I/O pairs with expected quality)
Level 2: Automated metrics  (accuracy, faithfulness, LLM-as-judge)
Level 3: Online eval        (A/B, production quality sampling)
Level 4: Continuous eval    (drift detection, regression gates in CI)
```

### Building a golden dataset
- Start with 50–200 representative examples covering the *distribution of real usage*, not just happy paths.
- Deliberately include hard cases, edge cases, adversarial inputs, and known failure modes.
- Stratify by user segment, input type, and difficulty so you can read quality *by slice*.
- Version it. Treat it as a product asset that grows as you learn.

### Choosing metrics
| Task type | Primary metric | Notes |
|-----------|---------------|-------|
| Classification | Precision/Recall/F1 | Watch class imbalance |
| RAG / QA | Faithfulness, answer relevance, context recall | See `04-RAG` |
| Summarization | Factual consistency, coverage, LLM-as-judge | Human spot-checks |
| Generation | Win-rate vs baseline (pairwise) | ELO-style ranking |
| Agents | Task success rate, steps-to-completion, tool-call accuracy | See `20-Agent-Infrastructure` |

**Rule:** Every launch decision maps to a metric and a threshold, agreed with stakeholders *before* the sprint.

---

## 3. The Build Decision: Prompt vs RAG vs Fine-Tune vs Agent

This is the most consequential technical-product decision an AI PM shapes.

```
Start
  │
  ├─ Need current/proprietary knowledge? ──▶ RAG (add retrieval)
  │
  ├─ Need specific format/style/tone consistently? ──▶ Fine-tune (or few-shot first)
  │
  ├─ Need multi-step actions/tools? ──▶ Agent
  │
  └─ General capability, static knowledge OK? ──▶ Prompt engineering
```

| Approach | Cost to build | Latency | When to use | Downside |
|----------|--------------|---------|-------------|----------|
| Prompt engineering | Lowest | Low | First attempt, general tasks | Ceiling on control |
| RAG | Medium | Medium | Proprietary/fresh knowledge, citations | Retrieval quality is the bottleneck |
| Fine-tuning | High | Low (inference) | Consistent style, narrow domain, cost reduction at scale | Data + retraining burden |
| Agents | Highest | High | Multi-step workflows, tool use | Reliability, cost, debugging |

**PM guidance:** Always exhaust prompt engineering and RAG before fine-tuning. Fine-tuning is a commitment (data pipeline, retraining, drift). See `64-Model-Fine-Tuning-and-Post-Training` and `04-RAG`.

---

## 4. The Data Flywheel

The durable moat in AI products is not the model (everyone rents the same foundation models) — it is the **proprietary data loop**.

```
     Users interact
          │
          ▼
   Capture signals ──────┐
   (clicks, edits,       │
    thumbs, corrections) │
          │              │
          ▼              │
   Curate into eval/     │
   training data         │
          │              │
          ▼              │
   Improve model/prompt/ │
   retrieval             │
          │              │
          ▼              │
   Better product ───────┘  (attracts more users → more signal)
```

**PM responsibilities in the flywheel:**
- Instrument implicit feedback (accept/reject, edit distance, dwell time) from day one.
- Design lightweight explicit feedback (👍/👎, "report") without nagging users.
- Establish data governance and consent (see `40-AI-Data-Sovereignty-and-Privacy`).
- Close the loop: ensure captured data actually flows back into evals and training.

---

## 5. Writing an AI PRD

An AI PRD extends the classic PRD with AI-specific sections:

1. **Problem & user** — standard.
2. **Success metrics** — product + model quality + unit economics.
3. **Eval plan** — golden dataset description, metrics, thresholds, launch bar.
4. **Model/approach hypothesis** — prompt/RAG/fine-tune/agent, with rationale.
5. **Data strategy** — sources, licensing, privacy, flywheel design.
6. **Failure design** — expected failure modes, fallbacks, human-in-the-loop.
7. **Safety & guardrails** — red-team plan, content policies, bias checks.
8. **Cost model** — projected inference cost per task and at scale.
9. **Rollout plan** — shadow → canary → staged → GA, with kill switch.

---

## 6. Roadmapping Under Model Uncertainty

Foundation models improve (and get deprecated) on a timeline you don't control. Practical tactics:

- **Model-agnostic architecture.** Abstract the model behind a gateway so you can swap providers. See `25-Multi-Cloud-AI-Strategy` and `48-MCP-Cloud-Infrastructure-Agent-as-a-Service`.
- **"Capability overhang" bets.** Design features that get *better for free* as models improve.
- **Deprecation runbooks.** Every model dependency needs a migration + re-eval plan.
- **Buy-vs-build cadence.** Reassess quarterly: what you fine-tuned last quarter may now be beaten by a base model prompt.

---

## 7. Prioritization for AI Features

Classic frameworks (RICE, value/effort) still apply, but add two AI-specific axes:
- **Eval confidence** — how sure are we the quality bar is reachable?
- **Cost trajectory** — does this feature's unit cost improve or worsen at scale?

A high-value feature with low eval confidence belongs in a *research spike*, not the committed roadmap.

---

## Cross-References
- `06-Advanced/03-Evaluation-Benchmarks.md` — evaluation fundamentals
- `04-RAG` — retrieval systems
- `64-Model-Fine-Tuning-and-Post-Training` — fine-tuning tradeoffs
- `41-AI-Cost-Optimization-and-Enterprise-ROI` — cost modeling
- `16-AI-Business-Models-Playbooks` — monetization
