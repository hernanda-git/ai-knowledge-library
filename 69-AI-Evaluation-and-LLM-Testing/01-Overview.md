# AI Evaluation and LLM Testing — Overview

> Evaluation ("evals") is how teams measure whether an LLM or agent system actually works, keeps working, and improves over time. In 2025–2026 it is consistently ranked among the top in-demand AI engineering skills — because you cannot ship, iterate, or trust a system you cannot measure.

## Why This Category Exists

Traditional software has deterministic tests: given input X, assert output equals Y. LLM systems are **non-deterministic, open-ended, and subjective**. The same prompt can produce different valid answers; "correctness" is often a matter of degree; and failures are silent (a confident, fluent, wrong answer looks identical to a correct one).

Evaluation is the discipline that closes this gap. It answers questions like:

- Is the new prompt/model/RAG pipeline **better or worse** than the last one?
- Did my change **regress** any capability while improving another?
- How often does the system **hallucinate**, go off-policy, or fail safety constraints?
- What is the system's **quality distribution** in production, not just on a curated demo?

Without evals, teams "vibe-check" changes — a manual, unrepeatable process that collapses as systems grow. With evals, iteration becomes a measurable, compounding loop.

## The Core Mental Model

```
  ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
  │  Dataset    │ ──> │   System     │ ──> │  Outputs    │
  │ (inputs +   │     │ (prompt/RAG/ │     │             │
  │  references)│     │  agent/model)│     │             │
  └─────────────┘     └──────────────┘     └──────┬──────┘
                                                   │
                          ┌────────────────────────▼─────────┐
                          │        Evaluators / Scorers        │
                          │  (exact match, LLM-judge, rubric,  │
                          │   code checks, human labels)       │
                          └────────────────────────┬──────────┘
                                                    │
                                          ┌─────────▼─────────┐
                                          │  Aggregate Metrics │
                                          │  + Regression Gate │
                                          └────────────────────┘
```

Every eval system has four parts:
1. **Dataset** — representative inputs, ideally with expected outputs or reference material.
2. **System under test** — the prompt, chain, RAG pipeline, or agent.
3. **Evaluators (scorers)** — functions that turn an output into a score.
4. **Aggregation + gating** — roll scores into metrics and decide pass/fail.

## Types of Evaluation

| Type | When it runs | What it answers | Cost | Example |
|------|-------------|-----------------|------|---------|
| **Offline / batch** | Pre-deploy, CI | Did this change regress? | Medium | Run 500-case suite before merge |
| **Online / live** | Production | How is it doing on real traffic? | Low–Med | Sample 2% of prod, LLM-judge them |
| **Human eval** | Periodic / gold set | Ground truth, edge cases | High | SME rates 100 answers weekly |
| **Automated metric** | Everywhere | Cheap continuous signal | Very low | BLEU, exact-match, JSON-valid % |
| **LLM-as-judge** | Offline + online | Nuanced quality at scale | Medium | GPT-judge scores helpfulness 1–5 |
| **A/B test** | Production | Real user impact | Low | 50/50 split, measure task success |

## Offline vs Online Evaluation

**Offline evaluation** happens against a fixed dataset before shipping. It is your regression safety net and the engine of iteration. Fast, repeatable, cheap enough to run on every PR.

**Online evaluation** happens against live production traffic. It catches the distribution shift between your curated dataset and reality — the queries you never imagined, the adversarial users, the seasonal drift. Online evals feed back into offline datasets (mine failures → add to test suite).

The mature workflow is a **flywheel**:

```
 production traffic → capture traces → find failures →
 add to offline dataset → fix system → re-run offline suite →
 ship → back to production
```

## Key Metrics Landscape

- **Task-level**: task success rate, answer accuracy, tool-call correctness.
- **Quality**: helpfulness, relevance, coherence, groundedness/faithfulness (RAG).
- **Safety**: toxicity, PII leakage, jailbreak resistance, policy adherence.
- **Format**: JSON validity, schema conformance, citation presence.
- **Operational**: latency (p50/p95), cost per query, token usage.
- **RAG-specific**: context precision, context recall, faithfulness, answer relevancy (see `04-RAG/`).

## Common Anti-Patterns

1. **Demo-driven development** — tuning until the 5 demo examples work, then shipping to reality.
2. **Metric gaming** — optimizing BLEU/ROUGE that don't correlate with real quality.
3. **No baseline** — reporting "85% accuracy" with nothing to compare against.
4. **Judge = generator** — using the same model to generate and grade without calibration.
5. **Static datasets** — never updating evals as the product and user base evolve.
6. **Averaging away failures** — a 4.2/5 average hides the 5% catastrophic answers that lose customers.

## How This Category Is Organized

- **02-Core-Topics.md** — dataset design, evaluator types, LLM-as-judge, metrics, regression gating.
- **03-Technical-Deep-Dive.md** — building an eval harness from scratch, statistical rigor, judge calibration, CI integration.
- **04-Tools-and-Frameworks.md** — OpenAI Evals, Braintrust, LangSmith, Ragas, DeepEval, promptfoo, Inspect, Arize Phoenix.
- **05-Future-Outlook.md** — agent evals, self-improving eval loops, benchmark saturation, certification/regulation.

## Cross-References

- `04-RAG/` — RAG-specific evaluation (faithfulness, context recall).
- `52-AI-Hallucination-Detection-and-Mitigation/` — hallucination scoring overlaps heavily with eval.
- `18-Agent-Security-and-Trust/` — red-teaming and safety evals.
- `20-Agent-Infrastructure-and-Observability/` — tracing feeds online eval.
- `03-Agents/` — agent trajectory evaluation.
- `64-Model-Fine-Tuning-and-Post-Training/` — evals drive the RLHF/DPO reward signal.
- `68-Context-Engineering/` — evals validate context strategies.

## Key Takeaway

> If it isn't measured, it isn't engineered. Evaluation converts LLM development from art into engineering — turning subjective "seems better" into objective, gated, compounding improvement. Teams that build an eval flywheel early ship faster and break less than those who bolt it on after the first production incident.
