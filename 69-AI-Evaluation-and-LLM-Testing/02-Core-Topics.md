# AI Evaluation — Core Topics

> The building blocks every practitioner must master: datasets, evaluators, LLM-as-judge, metrics, and regression gating. This is the working vocabulary of eval engineering.

## 1. Evaluation Datasets

The dataset is the foundation. A weak dataset makes every downstream metric meaningless.

### Anatomy of an eval case

```json
{
  "id": "case-0042",
  "input": "What is the refund window for enterprise plans?",
  "reference": "Enterprise plans have a 30-day refund window.",
  "context": ["Enterprise: 30-day money-back guarantee..."],
  "metadata": { "category": "billing", "difficulty": "easy", "source": "prod" }
}
```

### Dataset design principles

| Principle | Why | Practice |
|-----------|-----|----------|
| **Representative** | Match real distribution | Sample from production logs |
| **Balanced** | Avoid blind spots | Cover each category/intent |
| **Includes hard cases** | Averages hide failures | Curate known-tricky inputs |
| **Versioned** | Reproducibility | Git or a dataset registry |
| **Sized right** | Signal vs cost | Start ~50–100, grow to 500+ |
| **Golden subset** | Human-verified truth | 30–100 SME-labeled cases |

### Where datasets come from

1. **Hand-written** — fast start, low coverage, author bias.
2. **Production mining** — real, diverse; requires logging + PII scrubbing.
3. **Synthetic generation** — LLM generates variations (see `51-Synthetic-Data-Generation/`).
4. **Failure harvesting** — every prod incident becomes a permanent test case.

> Rule of thumb: your dataset should grow every time something breaks. A stale eval set is a decaying asset.

## 2. Evaluators (Scorers)

An evaluator maps `(input, output, reference?) → score`. Categories:

### a) Deterministic / code-based
Cheap, fast, unambiguous. Best when correctness is well-defined.

```python
def json_valid(output: str) -> float:
    try:
        json.loads(output)
        return 1.0
    except Exception:
        return 0.0

def contains_citation(output: str) -> float:
    return 1.0 if re.search(r"\[\d+\]", output) else 0.0

def exact_match(output: str, reference: str) -> float:
    return 1.0 if output.strip().lower() == reference.strip().lower() else 0.0
```

### b) Statistical / reference-based
BLEU, ROUGE, METEOR, BERTScore, embedding cosine similarity. Useful for translation/summarization, but **weak proxies** for open-ended quality.

```python
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_similarity(output: str, reference: str) -> float:
    e = model.encode([output, reference])
    return float(util.cos_sim(e[0], e[1]))
```

### c) LLM-as-judge
A strong model grades the output against a rubric. Scales nuanced judgment. (Section 3.)

### d) Human
SME labeling — the gold standard for calibration and edge cases, but slow and expensive. Used to validate that your automated evaluators agree with humans.

## 3. LLM-as-Judge

The workhorse of modern evals. A capable model scores outputs where code can't.

### Three judge patterns

| Pattern | Prompt asks judge to... | Use when |
|---------|------------------------|----------|
| **Reference-based** | Compare output to gold answer | You have references |
| **Reference-free** | Rate against a rubric | Open-ended tasks |
| **Pairwise** | Pick better of A vs B | Comparing two systems |

### A solid judge prompt

```
You are an impartial evaluator. Score the RESPONSE on FAITHFULNESS to
the provided CONTEXT using this rubric:

5 - Every claim is fully supported by the context.
4 - Mostly supported; one minor unsupported detail.
3 - Half supported, half unsupported.
2 - Mostly unsupported claims.
1 - Contradicts the context or fully fabricated.

CONTEXT:
{context}

RESPONSE:
{response}

Think step by step, list each claim and whether the context supports it,
then output JSON: {"reasoning": "...", "score": <1-5>}
```

### Judge best practices

- **Use a rubric, not "rate 1–10"** — anchored scales are far more consistent.
- **Ask for reasoning first** (chain-of-thought), score last — raises accuracy.
- **Prefer pairwise for subtle differences** — humans and models compare better than they absolute-rate.
- **Mitigate biases**: position bias (swap A/B order and average), verbosity bias (judges favor longer answers), self-preference (a model favors its own outputs — use a different judge family).
- **Calibrate against humans** — measure judge-human agreement (Cohen's κ); don't trust a judge you haven't validated.
- **Use a strong judge** — the judge should generally be as capable or more capable than the system under test.

### Judge failure modes

```
position bias    → answer shown first scores higher
verbosity bias   → longer = perceived better
sycophancy       → agrees with assertive but wrong claims
self-preference  → GPT judges prefer GPT outputs
rubric drift     → vague rubric = inconsistent scores across runs
```

## 4. Metrics and Aggregation

Raw scores must roll up into decision-ready metrics.

### Aggregation choices

```python
# Mean hides tails — always look at distribution too
metrics = {
    "mean_score": statistics.mean(scores),
    "p10_score": percentile(scores, 10),   # worst-case floor
    "pass_rate": sum(s >= 4 for s in scores) / len(scores),
    "fail_count": sum(s <= 2 for s in scores),
}
```

### Slice-based analysis
Never report a single number. Break down by segment:

| Category | N | Pass rate | Mean |
|----------|---|-----------|------|
| billing | 120 | 0.94 | 4.5 |
| technical | 200 | 0.71 | 3.8 |
| edge-cases | 80 | 0.52 | 3.1 |

The overall 0.78 hides that `edge-cases` is failing — slicing surfaces it.

## 5. Regression Gating in CI

Turn evals into a merge gate:

```python
def gate(new_results, baseline):
    regressions = []
    for metric, value in new_results.items():
        base = baseline[metric]
        if value < base - 0.02:  # tolerance for noise
            regressions.append(f"{metric}: {base:.3f} -> {value:.3f}")
    if regressions:
        raise SystemExit("BLOCKED — regressions:\n" + "\n".join(regressions))
    print("PASSED — no regressions.")
```

Wire this into GitHub Actions so every PR runs the suite and blocks on regression (see `03-Technical-Deep-Dive.md`).

## 6. The Eval Flywheel

```
 1. Ship v1 with a small eval set
 2. Capture production traces
 3. Label / mine failures
 4. Add failures to dataset (it grows)
 5. Fix system, re-run offline suite
 6. Gate the PR, ship v2
 7. Repeat — dataset & quality compound
```

## Cross-References

- `04-RAG/` — RAG metric definitions (faithfulness, context recall).
- `51-Synthetic-Data-Generation/` — generating eval cases.
- `52-AI-Hallucination-Detection-and-Mitigation/` — faithfulness/groundedness scoring.
- `20-Agent-Infrastructure-and-Observability/` — trace capture for online eval.

## Key Takeaway

> Master four primitives — datasets, evaluators, judges, and gates — and you can measure any LLM system. Everything else is refinement. The judge is powerful but biased; always calibrate it against humans, and always slice your metrics instead of trusting a single average.
