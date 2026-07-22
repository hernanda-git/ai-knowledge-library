# AI Evaluation — Tools and Frameworks

> The 2025–2026 eval tooling landscape: open-source harnesses, commercial platforms, RAG-specific libraries, and how to choose. Includes code snippets and a decision matrix.

## Landscape Map

```
                        EVAL TOOLING
     ┌───────────────┬────────────────┬──────────────────┐
     │  Frameworks   │   Platforms    │   Specialized    │
     │ (code-first)  │ (hosted UI)    │  (domain focus)  │
     ├───────────────┼────────────────┼──────────────────┤
     │ OpenAI Evals  │ Braintrust     │ Ragas (RAG)      │
     │ promptfoo     │ LangSmith      │ DeepEval (pytest)│
     │ Inspect (UK)  │ Arize Phoenix  │ TruLens          │
     │ lm-eval-harness│ Langfuse      │ Giskard (safety) │
     │ EvalLite      │ Humanloop      │ Promptfoo redteam│
     └───────────────┴────────────────┴──────────────────┘
```

## Open-Source Frameworks

### promptfoo — fast, config-driven, great for CI
Declarative YAML; excellent for comparing prompts/models side by side.

```yaml
# promptfooconfig.yaml
prompts:
  - "Answer concisely: {{question}}"
providers:
  - openai:gpt-4o-mini
  - anthropic:claude-3-5-sonnet
tests:
  - vars: { question: "Capital of France?" }
    assert:
      - type: contains
        value: "Paris"
      - type: llm-rubric
        value: "Response is concise and correct"
      - type: latency
        threshold: 3000
```
```bash
npx promptfoo eval && npx promptfoo view
```

### DeepEval — "pytest for LLMs"
Pythonic, integrates with existing test suites; ships many metrics.

```python
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

def test_rag_answer():
    tc = LLMTestCase(
        input="What is our refund window?",
        actual_output=rag_pipeline("What is our refund window?"),
        retrieval_context=["Enterprise plans: 30-day refund."],
    )
    assert_test(tc, [FaithfulnessMetric(threshold=0.8),
                     AnswerRelevancyMetric(threshold=0.7)])
```

### OpenAI Evals
Registry-based framework; good for standardized model benchmarks and custom evals shared across teams.

### Inspect (UK AI Safety Institute)
Rigorous, research-grade framework for capability and safety evals; strong for agentic tasks and dangerous-capability testing.

```python
from inspect_ai import Task, task
from inspect_ai.dataset import example_dataset
from inspect_ai.scorer import model_graded_qa
from inspect_ai.solver import generate

@task
def qa_eval():
    return Task(
        dataset=example_dataset("theory_of_mind"),
        solver=generate(),
        scorer=model_graded_qa(),
    )
```

### lm-evaluation-harness (EleutherAI)
The standard for **academic model benchmarks** (MMLU, GSM8K, HellaSwag, etc.). Use it to benchmark base/foundation models, not product pipelines.

## Commercial / Hosted Platforms

| Platform | Sweet spot | Notable features |
|----------|-----------|------------------|
| **Braintrust** | Eval-driven dev loop | Playground, dataset versioning, CI SDK, scoring UI |
| **LangSmith** | LangChain ecosystems | Trace-linked evals, datasets from prod, annotation queues |
| **Arize Phoenix** | Open-source observability + eval | OTel traces, drift, embedding analysis (self-host) |
| **Langfuse** | Open-source tracing + eval | Prompt mgmt, scores, cheap self-host |
| **Humanloop** | PM/SME collaboration | Human review workflows, prompt versioning |

### LangSmith eval example

```python
from langsmith import Client
from langsmith.evaluation import evaluate

def correctness(run, example):
    score = judge(run.outputs["answer"], example.outputs["answer"])
    return {"key": "correctness", "score": score}

evaluate(
    lambda inputs: my_app(inputs["question"]),
    data="my-dataset",
    evaluators=[correctness],
    experiment_prefix="v2-prompt",
)
```

### Braintrust example

```python
from braintrust import Eval
from autoevals import Factuality

Eval(
    "refund-bot",
    data=lambda: load_cases(),
    task=lambda input: my_app(input),
    scores=[Factuality],
)
```

## RAG-Specific: Ragas

Purpose-built metrics for retrieval pipelines (pairs with `04-RAG/`).

```python
from ragas import evaluate
from ragas.metrics import (faithfulness, answer_relevancy,
                           context_precision, context_recall)
from datasets import Dataset

ds = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": retrieved_contexts,   # list[list[str]]
    "ground_truth": references,
})
report = evaluate(ds, metrics=[faithfulness, answer_relevancy,
                               context_precision, context_recall])
print(report)  # {'faithfulness': 0.91, 'context_recall': 0.78, ...}
```

| Ragas metric | Measures |
|--------------|----------|
| faithfulness | Answer grounded in retrieved context |
| answer_relevancy | Answer addresses the question |
| context_precision | Retrieved chunks are relevant (ranking) |
| context_recall | All needed info was retrieved |

## Safety / Red-Team Tooling

- **Giskard** — automated vulnerability scanning (bias, injection, robustness).
- **promptfoo redteam** — generates adversarial jailbreak/injection test cases.
- **Garak** — LLM vulnerability scanner (prompt injection, toxicity, leakage).

See `18-Agent-Security-and-Trust/` and `65-AI-for-Cybersecurity/`.

## Decision Matrix

| Your situation | Recommended |
|----------------|-------------|
| CI prompt/model comparison, minimal setup | **promptfoo** |
| Python test-suite integration | **DeepEval** |
| RAG pipeline metrics | **Ragas** |
| Full eval-driven dev with UI + datasets | **Braintrust** or **LangSmith** |
| Open-source, self-hosted observability + eval | **Phoenix** / **Langfuse** |
| Benchmarking foundation models | **lm-eval-harness** |
| Safety / dangerous-capability research | **Inspect** |
| Red-teaming / security | **Giskard**, **promptfoo redteam**, **Garak** |

## Build vs Buy

**Build (custom harness)** when: unusual scoring logic, tight data-residency needs, deep integration with internal systems (see `03-Technical-Deep-Dive.md`).

**Buy/adopt** when: you want dataset versioning, a review UI, and trace linkage without maintaining infra. Most teams start with promptfoo/DeepEval in CI, then adopt a platform (Braintrust/LangSmith/Phoenix) as scale grows.

> A common mature stack: **promptfoo/DeepEval in CI** + **Ragas for RAG metrics** + **Phoenix/Langfuse for prod tracing & online eval** + a small **custom scorer library** for domain-specific checks.

## Cross-References

- `04-RAG/` — retrieval pipelines these tools evaluate.
- `20-Agent-Infrastructure-and-Observability/` — Phoenix/Langfuse tracing overlap.
- `18-Agent-Security-and-Trust/` — red-team tooling.
- `52-AI-Hallucination-Detection-and-Mitigation/` — faithfulness scoring tools.

## Key Takeaway

> Don't over-tool. Start with one code-first framework in CI (promptfoo or DeepEval) plus Ragas if you do RAG, and add a hosted platform only when dataset management and team collaboration become the bottleneck. The tool matters far less than the discipline of running evals on every change.

---
**See also:**
- [03 — LLM Architectures 2026: Beyond the Vanilla Transformer](07-Emerging/17-Research-Frontiers-2026/03-LLM-Architectures-2026.md)
- [04 — Local LLM Indexing and Search](23-Local-AI-Inference-Self-Hosting/04-Local-LLM-Indexing-and-Search.md)
