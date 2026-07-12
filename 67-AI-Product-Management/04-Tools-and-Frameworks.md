# AI Product Management — Tools and Frameworks

> The AI PM tool stack: evaluation platforms, LLM observability, experimentation, prompt management, analytics, and the decision frameworks that tie them together.

An AI PM doesn't need to master every tool, but must know the categories, pick a stack, and understand what each layer buys you. This document maps the ecosystem as of 2026 and provides selection guidance.

---

## The AI PM Tool Stack (Layers)

```
┌───────────────────────────────────────────────┐
│  Analytics & BI (product metrics, funnels)     │
├───────────────────────────────────────────────┤
│  Experimentation (A/B, feature flags)          │
├───────────────────────────────────────────────┤
│  Observability & Tracing (production quality)   │
├───────────────────────────────────────────────┤
│  Evaluation (offline eval, LLM-as-judge)        │
├───────────────────────────────────────────────┤
│  Prompt / Dataset Management (versioning)       │
├───────────────────────────────────────────────┤
│  Model Gateway (routing, caching, cost)         │
└───────────────────────────────────────────────┘
```

---

## 1. Evaluation Platforms

| Tool | Strength | Notes |
|------|----------|-------|
| **LangSmith** | Tracing + eval tightly integrated with LangChain | Broad ecosystem |
| **Braintrust** | Eval-first workflow, dataset + scoring UI | Popular with AI-native teams |
| **Arize Phoenix** | Open-source tracing + eval | Self-hostable |
| **Weights & Biases (Weave)** | Experiment tracking + LLM eval | Strong for teams already on W&B |
| **Ragas** | RAG-specific metrics (faithfulness, context recall) | Library, see `04-RAG` |
| **DeepEval** | Pytest-style LLM unit tests | Great for CI gating |
| **OpenAI Evals** | Simple, provider-native | Good starting point |

**Selection guidance:** Start with a library (DeepEval/Ragas) wired into CI for regression gating, then add a platform (Braintrust/LangSmith) when you need a UI for non-engineers to curate datasets and review results.

### Minimal eval with DeepEval
```python
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric

def test_summary_faithfulness():
    tc = LLMTestCase(
        input="Summarize the earnings call.",
        actual_output=model_summary,
        retrieval_context=[transcript],
    )
    assert_test(tc, [FaithfulnessMetric(threshold=0.8)])
```

---

## 2. LLM Observability & Tracing

Production AI needs distributed tracing that captures prompts, completions, tokens, latency, cost, and tool calls per request.

| Tool | Notes |
|------|-------|
| **Langfuse** | Open-source, self-hostable, strong tracing + eval |
| **Helicone** | Proxy-based, easy drop-in, cost analytics |
| **Arize / Phoenix** | ML + LLM observability, drift detection |
| **Datadog LLM Observability** | For teams already on Datadog |

See `20-Agent-Infrastructure-and-Observability` for the full observability treatment.

**What the PM watches on the dashboard:** quality-sampled scores over time, cost per task trend, latency p95/p99, error/refusal rates, and drift alerts.

---

## 3. Experimentation & Feature Flags

| Tool | Strength |
|------|----------|
| **Statsig** | Stats-rigorous experimentation, popular with AI teams |
| **LaunchDarkly** | Enterprise feature flags, staged rollout |
| **GrowthBook** | Open-source A/B testing |
| **PostHog** | Analytics + flags + experiments in one |

These enable shadow mode, canary rollout, and A/B tests described in `03-Technical-Deep-Dive.md`.

---

## 4. Prompt & Dataset Management

Prompts are product artifacts and must be versioned, reviewed, and rolled back like code.

- **PromptLayer / Langfuse Prompts / Braintrust** — prompt versioning + linking to eval results.
- **Git** — many teams keep prompts in-repo with PR review (recommended default).
- **Dataset versioning** — treat golden datasets as versioned assets (DVC, or platform-native).

**Anti-pattern:** prompts pasted into a codebase as inline strings with no version history — you lose the ability to attribute quality changes to prompt changes.

---

## 5. Model Gateways / Routers

A gateway decouples your product from any single model provider.

| Tool | Notes |
|------|-------|
| **LiteLLM** | Open-source, unified API across 100+ models |
| **OpenRouter** | Hosted routing marketplace |
| **Portkey** | Gateway + guardrails + caching + observability |
| **Cloudflare AI Gateway** | Edge caching + analytics |

Benefits: provider failover, cost-based routing, caching, and a clean seam for model migration. See `25-Multi-Cloud-AI-Strategy` and `48-MCP-Cloud-Infrastructure-Agent-as-a-Service`.

---

## 6. Product Analytics for AI

Standard analytics (Amplitude, Mixpanel, PostHog) still apply, instrumented with AI-specific events:
- `ai_suggestion_shown`, `ai_suggestion_accepted`, `ai_suggestion_edited`, `ai_suggestion_rejected`
- `ai_regeneration_requested` (a strong dissatisfaction signal)
- `ai_feedback_thumbs` (explicit signal)
- Edit distance between AI output and final user-submitted content (implicit quality proxy)

These events feed both product decisions and the data flywheel (`02-Core-Topics.md`).

---

## 7. Reference Architecture (Small AI-Native Team)

```
User ─▶ App ─▶ Model Gateway (LiteLLM) ─▶ Providers
                     │
                     ├─▶ Tracing (Langfuse) ──▶ Quality sampling + dashboards
                     ├─▶ Feature flags (Statsig) ──▶ Canary / A-B
                     └─▶ Prompts in Git (PR-reviewed)

CI: DeepEval golden-set gate ──▶ block merge on regression
Flywheel: analytics events ──▶ curate ──▶ golden set + fine-tune data
```

This stack is cheap to start, mostly open-source, and covers eval, observability, experimentation, and cost control.

---

## 8. Buy vs Build Guidance

| Layer | Default | Build when |
|-------|---------|-----------|
| Eval library | Buy/OSS | Never (huge maintenance) |
| Eval UI/platform | Buy | Highly custom workflows at scale |
| Observability | Buy/OSS | Extreme data-residency needs |
| Gateway | OSS (LiteLLM) | Unique routing/compliance logic |
| Golden datasets | **Build** | Always — this is your moat |

The one thing you should always build in-house is your **evaluation dataset and rubric** — it encodes your definition of quality and cannot be bought.

---

## Cross-References
- `20-Agent-Infrastructure-and-Observability` — observability depth
- `04-RAG` — Ragas and RAG eval
- `41-AI-Cost-Optimization-and-Enterprise-ROI` — gateway cost routing
- `33-AI-Native-Software-Development` — CI/CD integration
- `25-Multi-Cloud-AI-Strategy` — multi-provider strategy
