# AI Product Management — Technical Deep Dive

> The technical machinery an AI PM must understand to make defensible decisions: building eval harnesses, LLM-as-judge, offline vs online evaluation, A/B testing probabilistic systems, statistical rigor, and regression gating in CI.

An AI PM does not need to train models, but must be fluent enough to design evaluation, read results critically, and challenge engineering claims. This document provides that working depth with runnable examples.

---

## 1. Anatomy of an Eval Harness

An eval harness runs a system under test against a dataset and produces scored, sliceable results.

```python
from dataclasses import dataclass
from typing import Callable, Any
import statistics

@dataclass
class EvalCase:
    id: str
    inputs: dict
    expected: Any
    slice: str  # e.g. "easy", "adversarial", "segment:enterprise"

@dataclass
class EvalResult:
    case_id: str
    score: float
    passed: bool
    slice: str

def run_eval(cases: list[EvalCase],
             system: Callable[[dict], Any],
             scorer: Callable[[Any, Any], float],
             threshold: float) -> dict:
    results = []
    for c in cases:
        output = system(c.inputs)
        score = scorer(output, c.expected)
        results.append(EvalResult(c.id, score, score >= threshold, c.slice))

    # aggregate overall + by slice
    def agg(rs):
        return {
            "n": len(rs),
            "mean_score": round(statistics.mean(r.score for r in rs), 3),
            "pass_rate": round(sum(r.passed for r in rs) / len(rs), 3),
        }

    slices = {}
    for s in set(r.slice for r in results):
        slices[s] = agg([r for r in results if r.slice == s])

    return {"overall": agg(results), "by_slice": slices}
```

**The by-slice breakdown is the AI PM's superpower.** A model at 90% overall might be at 60% on your enterprise segment — the number that actually determines a deal.

---

## 2. LLM-as-Judge

When there is no ground-truth label (summaries, open-ended generation), use a strong model to score outputs against a rubric.

```python
JUDGE_PROMPT = """You are a strict evaluator. Score the RESPONSE on FAITHFULNESS
to the SOURCE on a 1-5 scale.
5 = every claim supported by source; 1 = major fabrications.
Return JSON: {{"score": <int>, "reason": "<short>"}}

SOURCE:
{source}

RESPONSE:
{response}
"""

def llm_judge(source, response, judge_model):
    out = judge_model(JUDGE_PROMPT.format(source=source, response=response))
    return parse_json(out)  # {"score": 4, "reason": "..."}
```

### Making LLM-as-judge trustworthy
- **Calibrate against humans.** Score a sample both ways; measure agreement (Cohen's kappa). If agreement is low, fix the rubric.
- **Use pairwise, not absolute, when possible.** "Is A or B better?" is more reliable than "rate A from 1–10."
- **Control for position/verbosity bias.** Randomize order; judges favor longer answers and the first option.
- **Pin the judge model version.** A judge upgrade silently shifts your metrics.

---

## 3. Offline vs Online Evaluation

| | Offline | Online |
|---|---------|--------|
| When | Before ship, in CI | In production |
| Data | Golden dataset | Real traffic |
| Speed | Fast feedback | Slow (needs traffic) |
| Fidelity | Proxy for reality | Ground truth |
| Risk | Zero user impact | Real user impact |

**Best practice:** Offline eval gates every code/prompt change; online eval (A/B + production sampling) validates real-world impact. Offline catches regressions cheaply; online catches the things your golden set didn't anticipate.

---

## 4. A/B Testing Probabilistic Systems

AI A/B tests have extra failure modes over classic experiments:

- **High variance.** Model outputs vary run-to-run; you need larger samples or paired designs.
- **Novelty effects.** Users behave differently when a feature is new; run long enough to reach steady state.
- **Metric interference.** A quality win can *reduce* usage if it makes a task finish faster — decide whether that's good.
- **Guardrail metrics.** Always pair the primary metric (e.g., acceptance rate) with guardrails (latency, cost, safety incidents).

### Sizing the experiment
```python
from math import ceil

def sample_size_per_arm(baseline_rate, mde, alpha=0.05, power=0.8):
    # normal-approx for two-proportion test
    from scipy.stats import norm
    z_a = norm.ppf(1 - alpha/2)
    z_b = norm.ppf(power)
    p1, p2 = baseline_rate, baseline_rate + mde
    pbar = (p1 + p2) / 2
    n = ((z_a * (2*pbar*(1-pbar))**0.5 + z_b * (p1*(1-p1)+p2*(1-p2))**0.5)**2) / (mde**2)
    return ceil(n)

# detect a 3-point lift on a 40% acceptance baseline
print(sample_size_per_arm(0.40, 0.03))
```

---

## 5. Shadow Mode & Canary Rollouts

For high-stakes AI, run the new model in **shadow mode** first: it processes real traffic and logs outputs, but users never see them. You compare shadow outputs to the incumbent (or to human decisions) with zero risk.

```
Request ──▶ Production model ──▶ User
        └──▶ Shadow model ─────▶ Log + compare (no user impact)
```

Progression: shadow → 1% canary → 5% → 25% → 100%, with automated rollback if guardrail metrics breach thresholds.

---

## 6. Regression Gating in CI

Wire offline eval into the deployment pipeline so quality regressions block merges.

```yaml
# .github/workflows/ai-eval.yml (illustrative)
name: ai-eval-gate
on: [pull_request]
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r requirements.txt
      - name: Run eval suite
        run: python eval/run.py --dataset golden.jsonl --report report.json
      - name: Enforce thresholds
        run: |
          python eval/gate.py --report report.json \
            --min-pass-rate 0.85 \
            --max-regression 0.02   # fail if any slice drops >2pts vs baseline
```

This turns "did we make it worse?" from a subjective debate into a build status. See `33-AI-Native-Software-Development/03-AI-Native-CI-CD-and-DevOps.md`.

---

## 7. Reading Results Critically (PM Checklist)

- **Is the sample representative?** A demo set of easy cases inflates every number.
- **What's the confidence interval?** A 2-point difference on 50 examples is noise.
- **Which slice moved?** Overall gains can hide segment regressions.
- **Did cost/latency change?** A quality win at 3× cost may not ship.
- **Is the judge trustworthy?** Check human agreement before trusting LLM-as-judge deltas.
- **Did we overfit the eval?** If the golden set never changes, teams optimize to it. Rotate held-out cases.

---

## 8. Cost & Latency Instrumentation

```python
@dataclass
class RequestTrace:
    tokens_in: int
    tokens_out: int
    model: str
    latency_ms: float
    success: bool

PRICES = {"model-a": (0.005, 0.015)}  # $/1k in, $/1k out

def cost(t: RequestTrace) -> float:
    pin, pout = PRICES[t.model]
    return (t.tokens_in/1000)*pin + (t.tokens_out/1000)*pout

def cost_per_successful_task(traces: list[RequestTrace]) -> float:
    total = sum(cost(t) for t in traces)
    succ = sum(1 for t in traces if t.success)
    return total / max(succ, 1)
```

`cost_per_successful_task` is the north-star unit-economics metric. See `41-AI-Cost-Optimization-and-Enterprise-ROI`.

---

## Cross-References
- `20-Agent-Infrastructure-and-Observability/58-AI-Evaluation-and-Benchmarking-at-Scale`
- `52-AI-Hallucination-Detection-and-Mitigation`
- `04-RAG` — RAG-specific eval (faithfulness, context recall)
- `33-AI-Native-Software-Development` — CI/CD for AI
