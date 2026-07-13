# AI Evaluation — Technical Deep Dive

> Building a production-grade eval harness from scratch: architecture, statistical rigor, judge calibration, agent trajectory evaluation, and CI integration. This is the engineering layer beneath the concepts.

## 1. Anatomy of an Eval Harness

A minimal but complete harness has five components:

```
┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐
│  Dataset   │─> │   Runner   │─> │ Evaluators │─> │ Aggregator │─> │  Reporter  │
│  loader    │   │ (calls SUT)│   │  (scorers) │   │  + stats   │   │ + gate     │
└────────────┘   └────────────┘   └────────────┘   └────────────┘   └────────────┘
```

### A from-scratch implementation

```python
import asyncio, json, statistics
from dataclasses import dataclass, field
from typing import Callable, Awaitable

@dataclass
class Case:
    id: str
    input: str
    reference: str | None = None
    context: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

@dataclass
class Result:
    case: Case
    output: str
    scores: dict[str, float]
    latency_ms: float

Scorer = Callable[[Case, str], Awaitable[float]]

class EvalHarness:
    def __init__(self, system, scorers: dict[str, Scorer], concurrency: int = 8):
        self.system = system
        self.scorers = scorers
        self.sem = asyncio.Semaphore(concurrency)

    async def _run_case(self, case: Case) -> Result:
        async with self.sem:
            import time
            t0 = time.perf_counter()
            output = await self.system(case.input, case.context)
            latency = (time.perf_counter() - t0) * 1000
            scores = {}
            for name, scorer in self.scorers.items():
                scores[name] = await scorer(case, output)
            return Result(case, output, scores, latency)

    async def run(self, cases: list[Case]) -> list[Result]:
        return await asyncio.gather(*(self._run_case(c) for c in cases))

    def aggregate(self, results: list[Result]) -> dict:
        agg = {}
        for name in self.scorers:
            vals = [r.scores[name] for r in results]
            agg[name] = {
                "mean": statistics.mean(vals),
                "min": min(vals),
                "pass_rate": sum(v >= 0.8 for v in vals) / len(vals),
            }
        agg["latency_p95"] = percentile([r.latency_ms for r in results], 95)
        return agg

def percentile(data, p):
    s = sorted(data)
    k = (len(s) - 1) * p / 100
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)
```

### Wiring in an LLM-judge scorer

```python
async def make_faithfulness_scorer(judge_client):
    async def score(case: Case, output: str) -> float:
        prompt = FAITHFULNESS_RUBRIC.format(
            context="\n".join(case.context), response=output)
        resp = await judge_client.complete(prompt, temperature=0,
                                            response_format={"type": "json_object"})
        return json.loads(resp)["score"] / 5.0  # normalize to 0-1
    return score
```

## 2. Statistical Rigor

Eval numbers are **estimates with uncertainty**. Treat them like a scientist.

### Confidence intervals

A pass rate of 0.85 on 50 cases has a wide interval. Report it:

```python
import math
def wilson_interval(successes, n, z=1.96):
    if n == 0: return (0, 0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return (center - margin, center + margin)

# 43/50 passes -> (0.727, 0.928)  -> big uncertainty; grow the dataset
```

### Is the improvement real? (significance)

Comparing v1 vs v2 pass rates, use a two-proportion test or, for paired per-case scores, a paired bootstrap:

```python
import random
def bootstrap_diff(a_scores, b_scores, iters=10000):
    n = len(a_scores)
    diffs = []
    for _ in range(iters):
        idx = [random.randrange(n) for _ in range(n)]
        diffs.append(statistics.mean(b_scores[i]-a_scores[i] for i in idx))
    diffs.sort()
    return diffs[int(0.025*iters)], diffs[int(0.975*iters)]
# If the 95% CI for the diff excludes 0, the change is significant.
```

> A 1-point move on 40 cases is noise. Know your minimum detectable effect before celebrating.

### Sample size intuition

| Cases | ±margin (95%) at p≈0.8 | Verdict |
|-------|------------------------|---------|
| 20 | ±0.18 | too noisy |
| 50 | ±0.11 | rough |
| 200 | ±0.055 | usable |
| 1000 | ±0.025 | tight |

## 3. Judge Calibration

Never trust a judge you haven't measured against humans.

```python
from sklearn.metrics import cohen_kappa_score

# human_labels and judge_labels are aligned lists of ordinal scores
kappa = cohen_kappa_score(human_labels, judge_labels, weights="quadratic")
# kappa > 0.6 = substantial agreement; < 0.4 = revise your rubric/judge
```

### Debiasing techniques

```python
# Position-bias mitigation for pairwise judging: run both orders, average
async def pairwise_debiased(judge, a, b):
    s1 = await judge(a, b)   # A first
    s2 = await judge(b, a)   # B first
    return (s1 + (1 - s2)) / 2  # symmetric average
```

- **Ensemble judges** — average across 3 model families to reduce self-preference.
- **Temperature 0** on the judge for reproducibility.
- **Rubric anchoring** — provide 1–2 example gradings per score level (few-shot the judge).

## 4. Agent & Multi-Step Evaluation

Agents add a whole new axis: you evaluate the **trajectory**, not just the final answer (see `03-Agents/`).

| Dimension | Question | Scorer |
|-----------|----------|--------|
| Final outcome | Did the task succeed? | task-success check |
| Tool selection | Right tools chosen? | trajectory match |
| Tool arguments | Correct params? | arg validation |
| Efficiency | Minimal steps? | step count vs optimal |
| Recovery | Handled errors? | error-recovery flag |

```python
def trajectory_score(actual_calls, expected_calls):
    """Order-aware tool-sequence match (ratio of correct prefix)."""
    correct = sum(1 for a, e in zip(actual_calls, expected_calls)
                  if a["tool"] == e["tool"])
    return correct / max(len(expected_calls), 1)
```

For agents, prefer **reference-free outcome checks** (did the DB row get created? did the ticket close?) over matching a rigid golden trajectory — there are many valid paths.

## 5. Online Evaluation Pipeline

```
prod request → LLM response → async sampler (2%) → judge queue →
scores → dashboard + alert if pass_rate drops below threshold
```

```python
async def online_eval_middleware(request, response):
    if random.random() < 0.02:  # sample 2%
        await eval_queue.put({
            "input": request.text, "output": response.text,
            "trace_id": response.trace_id, "ts": time.time(),
        })
    return response  # never block the user on eval
```

Key rules: **sample, don't eval everything** (cost); **never block the response path**; **route low-scoring traces to a review queue** that feeds the offline dataset.

## 6. CI Integration

```yaml
# .github/workflows/evals.yml
name: LLM Evals
on: [pull_request]
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -r requirements.txt
      - name: Run eval suite
        env: { OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }} }
        run: python -m evals.run --dataset datasets/core.jsonl --gate baseline.json
      - name: Upload report
        uses: actions/upload-artifact@v4
        with: { name: eval-report, path: report.html }
```

The `--gate` step compares against a committed `baseline.json` and exits non-zero on regression, blocking the merge.

## 7. Cost & Latency Management

- **Cache judge calls** keyed on `hash(prompt+output)` — datasets are re-run often.
- **Tier your suite**: fast smoke set (~30 cases) on every push; full set (~500) nightly.
- **Cheaper judge model** for continuous online eval, strong judge for offline gold.
- **Deterministic scorers first** — only invoke LLM-judge when code checks can't decide.

## Cross-References

- `03-Agents/` — agent architectures being evaluated.
- `20-Agent-Infrastructure-and-Observability/` — tracing backbone for online eval.
- `29-Reasoning-and-Inference-Scaling/` — evaluating reasoning chains.
- `64-Model-Fine-Tuning-and-Post-Training/` — evals as reward/validation signal.

## Key Takeaway

> A real eval harness is 20% scoring and 80% engineering discipline: statistical honesty about uncertainty, calibrated judges, sliced reporting, and CI gates. Build the harness once, and every future model/prompt/agent change becomes a measured experiment instead of a gamble.
