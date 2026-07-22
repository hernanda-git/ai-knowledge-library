# Tools and Frameworks for AI Evaluation & Benchmarking at Scale

> Comprehensive guide to evaluation tools, platforms, and frameworks for production LLM evaluation — from open-source libraries to enterprise platforms.

**Last Updated:** 2026-07-06
**Estimated Reading Time:** 80 minutes
**Line Count:** ~350+
**Category:** 58-AI-Evaluation-and-Benchmarking-at-Scale

---

## Table of Contents

1. [Evaluation Tools Landscape](#1-evaluation-tools-landscape)
2. [Open-Source Evaluation Frameworks](#2-open-source-evaluation-frameworks)
3. [Commercial Evaluation Platforms](#3-commercial-evaluation-platforms)
4. [LLM-as-Judge Tools](#4-llm-as-judge-tools)
5. [Safety Evaluation Tools](#5-safety-evaluation-tools)
6. [Observability and Monitoring Platforms](#6-observability-and-monitoring-platforms)
7. [Evaluation Infrastructure Tools](#7-evaluation-infrastructure-tools)
8. [Tool Selection Guide](#8-tool-selection-guide)
9. [Integration Patterns](#9-integration-patterns)
10. [Tool Comparison Matrix](#10-tool-comparison-matrix)

---

## 1. Evaluation Tools Landscape

### 1.1 Tool Categories

```
┌─────────────────────────────────────────────────────────────┐
│              EVALUATION TOOLS LANDSCAPE                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Layer 1: Evaluation Frameworks                     │   │
│  │  Ragas, DeepEval, LM-eval-harness, Eleuther         │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│  ┌──────────────────────▼──────────────────────────────┐   │
│  │  Layer 2: Evaluation Platforms                       │   │
│  │  Braintrust, LangSmith, Arize, Patronus              │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│  ┌──────────────────────▼──────────────────────────────┐   │
│  │  Layer 3: Observability                              │   │
│  │  LangFuse, Helicone, Weights & Biases, MLflow        │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│  ┌──────────────────────▼──────────────────────────────┐   │
│  │  Layer 4: Infrastructure                             │   │
│  │  Prometheus, Grafana, Datadog, Custom                │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Tool Maturity Matrix

| Tool | Category | Maturity | Community | Enterprise | Open Source |
|---|---|---|---|---|---|
| Ragas | RAG Eval | ⭐⭐⭐⭐ | Growing | ❌ | ✅ |
| DeepEval | LLM Eval | ⭐⭐⭐⭐ | Active | ✅ | ✅ |
| LM-eval-harness | Model Eval | ⭐⭐⭐⭐⭐ | Mature | ❌ | ✅ |
| Braintrust | Full Platform | ⭐⭐⭐⭐ | Growing | ✅ | Partial |
| LangSmith | LangChain Eval | ⭐⭐⭐⭐ | Mature | ✅ | Partial |
| Arize Phoenix | Observability | ⭐⭐⭐⭐ | Active | ✅ | ✅ |
| Patronus AI | Enterprise Eval | ⭐⭐⭐ | Growing | ✅ | ❌ |
| LangFuse | Observability | ⭐⭐⭐⭐ | Active | ✅ | ✅ |

---

## 2. Open-Source Evaluation Frameworks

### 2.1 Ragas (Retrieval-Augmented Generation Assessment)

**Focus:** RAG pipeline evaluation
**Best For:** Evaluating retrieval quality, faithfulness, and answer correctness
**GitHub Stars:** 7k+

#### Installation and Setup

```bash
pip install ragas[all]
```

#### Core Metrics

```python
from ragas.metrics import (
    faithfulness,           # Is answer grounded in context?
    answer_relevancy,       # Does answer address the question?
    context_precision,      # Are retrieved contexts relevant?
    context_recall,         # Did we retrieve all needed info?
    answer_correctness,     # Is the answer correct?
    answer_similarity,      # Semantic similarity to reference
)

# Available metrics breakdown:
RAGAS_METRICS = {
    # Retrieval metrics
    "context_precision": {
        "description": "Precision of retrieved contexts",
        "requires": ["question", "contexts", "ground_truth"],
        "score_range": (0, 1),
        "higher_is_better": True
    },
    "context_recall": {
        "description": "Recall of retrieved contexts",
        "requires": ["question", "contexts", "ground_truth"],
        "score_range": (0, 1),
        "higher_is_better": True
    },
    
    # Generation metrics
    "faithfulness": {
        "description": "How faithful the answer is to the context",
        "requires": ["question", "answer", "contexts"],
        "score_range": (0, 1),
        "higher_is_better": True
    },
    "answer_relevancy": {
        "description": "How relevant the answer is to the question",
        "requires": ["question", "answer", "contexts"],
        "score_range": (0, 1),
        "higher_is_better": True
    },
    
    # End-to-end metrics
    "answer_correctness": {
        "description": "Overall correctness of the answer",
        "requires": ["question", "answer", "ground_truth"],
        "score_range": (0, 1),
        "higher_is_better": True
    },
    "answer_similarity": {
        "description": "Semantic similarity to ground truth",
        "requires": ["answer", "ground_truth"],
        "score_range": (0, 1),
        "higher_is_better": True
    }
}
```

#### Full Example

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness
)
from datasets import Dataset

# Prepare evaluation data
eval_data = Dataset.from_dict({
    "question": [
        "What is the company's refund policy?",
        "How do I reset my password?",
        "What are the available subscription tiers?",
        "How do I contact support?",
        "What payment methods are accepted?",
    ],
    "answer": [
        "Our refund policy allows returns within 30 days of purchase for a full refund.",
        "To reset your password, go to Settings > Security > Reset Password.",
        "We offer Basic ($9/mo), Pro ($29/mo), and Enterprise (custom) tiers.",
        "You can contact support via email at help@example.com or through live chat.",
        "We accept Visa, Mastercard, American Express, and PayPal.",
    ],
    "contexts": [
        ["Refund Policy: Items may be returned within 30 days of purchase for a full refund. Contact support for assistance."],
        ["Account Recovery: Password resets can be initiated from the login page or Settings > Security > Reset Password."],
        ["Subscription Tiers: Basic ($9/mo), Pro ($29/mo), Enterprise (custom pricing). Each tier includes different features."],
        ["Support Channels: Email (help@example.com), Live Chat (24/7), Phone (business hours)."],
        ["Payment Methods: Visa, Mastercard, American Express, PayPal. Enterprise customers can pay via invoice."],
    ],
    "ground_truth": [
        "30-day refund policy for all items purchased",
        "Navigate to Settings > Security > Reset Password",
        "Basic ($9/mo), Pro ($29/mo), Enterprise (custom)",
        "Email help@example.com or use live chat",
        "Visa, Mastercard, AmEx, PayPal"
    ]
})

# Run evaluation
results = evaluate(
    eval_data,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness
    ]
)

# View results
print(results)
# Dataset({
#     features: ['question', 'answer', 'contexts', 'ground_truth', 
#                'faithfulness', 'answer_relevancy', 'context_precision',
#                'context_recall', 'answer_correctness'],
#     num_rows: 5
# })

# Get summary statistics
df = results.to_pandas()
print(f"Mean faithfulness: {df['faithfulness'].mean():.3f}")
print(f"Mean relevancy: {df['answer_relevancy'].mean():.3f}")
print(f"Mean correctness: {df['answer_correctness'].mean():.3f}")

# Export results
results.to_pandas().to_csv("eval_results.csv", index=False)
```

#### Custom Metric with Ragas

```python
from ragas.metrics import Metric
from ragas.llms import LangchainLLMWrapper
from ragas.prompt import PydanticPrompt

class CustomComplianceMetric(Metric):
    """Custom metric for compliance evaluation."""
    
    name = "compliance_score"
    
    def __init__(self, compliance_rules: list):
        self.compliance_rules = compliance_rules
    
    async def score(self, row: dict) -> float:
        """Score compliance with rules."""
        
        answer = row["answer"]
        context = row.get("contexts", [])
        
        prompt = f"""Evaluate this answer for compliance with the following rules:

Rules:
{chr(10).join(f"- {rule}" for rule in self.compliance_rules)}

Answer: {answer}

For each rule, rate compliance (0-1).
Return overall compliance score (0-1)."""
        
        result = await call_llm(prompt)
        return parse_compliance_score(result)
```

### 2.2 DeepEval

**Focus:** General LLM evaluation with production focus
**Best For:** CI/CD integration, regression testing, safety evaluation
**GitHub Stars:** 3k+

#### Installation and Setup

```bash
pip install deepeval
```

#### Core Features

```python
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    ToxicityMetric,
    BiasMetric,
    GEval  # Custom LLM-as-Judge metric
)

# Define metrics
relevancy = AnswerRelevancyMetric(threshold=0.7)
faithfulness = FaithfulnessMetric(threshold=0.8)
hallucination = HallucinationMetric(threshold=0.3)
toxicity = ToxicityMetric(threshold=0.5)
bias = BiasMetric(threshold=0.5)

# Custom metric with GEval
custom_metric = GEval(
    name="conciseness",
    criteria="Evaluate if the response is concise and to the point without unnecessary verbosity.",
    evaluation_params=[
        LLMTestCaseParams(input, actual_output)
    ],
    threshold=0.7
)

# Create test cases
test_cases = [
    LLMTestCase(
        input="What is the refund policy?",
        actual_output="Our refund policy allows returns within 30 days for a full refund.",
        expected_output="30-day refund policy",
        retrieval_context=["Refund Policy: Items may be returned within 30 days..."],
        context=["Company policies"]
    ),
    # More test cases...
]

# Run evaluation
results = evaluate(
    test_cases=test_cases,
    metrics=[relevancy, faithfulness, hallucination, toxicity, bias]
)

# View results
for result in results.test_results:
    print(f"Test: {result.name}")
    print(f"  Score: {result.score}")
    print(f"  Passed: {result.success}")
    print()
```

#### CI/CD Integration

```python
# .github/workflows/llm-eval.yml
name: LLM Evaluation

on:
  pull_request:
    paths:
      - 'prompts/**'
      - 'eval/**'

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install deepeval openai
          pip install -r requirements.txt
      
      - name: Run evaluation
        run: |
          python -m deepeval test run eval/test_suite.py
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      
      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: eval-results
          path: deepeval-results/
```

### 2.3 LM Evaluation Harness (EleutherAI)

**Focus:** Standardized model evaluation
**Best For:** Model comparison, benchmark replication, research
**GitHub Stars:** 7k+

#### Installation and Setup

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e .
```

#### Running Evaluations

```bash
# Evaluate a model on standard benchmarks
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
    --tasks mmlu,gsm8k,hellaswag,truthfulqa_mc2 \
    --num_fewshot 5 \
    --batch_size auto \
    --output_path results/

# Evaluate with custom tasks
lm_eval --model openai \
    --model_args model=gpt-4o-mini \
    --tasks custom_task \
    --custom_task_files custom_tasks/ \
    --num_fewshot 0
```

#### Custom Task Definition

```python
# custom_tasks/my_task.py
from lm_eval.api.task import Task
from lm_eval.api.registry import register_task

@register_task("my_custom_task")
class MyCustomTask(Task):
    """Custom evaluation task."""
    
    DATASET_PATH = "my_dataset"
    DATASET_NAME = "default"
    
    OUTPUT_TYPE = "multiple_choice"
    
    def has_training_docs(self):
        return True
    
    def has_validation_docs(self):
        return True
    
    def has_test_docs(self):
        return True
    
    def fewshot_examples(self, k):
        """Return k few-shot examples."""
        # Implementation here
        pass
    
    def doc_to_text(self, doc):
        """Convert document to text prompt."""
        return doc["question"]
    
    def doc_to_target(self, doc):
        """Convert document to target answer."""
        return doc["answer"]
    
    def construct_requests(self, doc, ctx):
        """Construct evaluation requests."""
        # Implementation here
        pass
    
    def process_results(self, doc, results):
        """Process evaluation results."""
        # Implementation here
        pass
```

### 2.4 PromptFlow (Microsoft)

**Focus:** Prompt engineering evaluation
**Best For:** Prompt testing, flow evaluation, Azure integration
**GitHub Stars:** 8k+

```python
# promptflow evaluation example
from promptflow import PFClient
from promptflow.evals.evaluators import (
    RelevanceEvaluator,
    CoherenceEvaluator,
    FluencyEvaluator,
    GroundednessEvaluator
)

# Initialize client
pf = PFClient()

# Define evaluators
evaluators = {
    "relevance": RelevanceEvaluator(),
    "coherence": CoherenceEvaluator(),
    "fluency": FluencyEvaluator(),
    "groundedness": GroundednessEvaluator()
}

# Run evaluation
results = pf.evaluate(
    data="eval_data.jsonl",
    evaluators=evaluators,
    column_mapping={
        "question": "${data.question}",
        "answer": "${data.answer}",
        "context": "${data.context}"
    }
)

# View results
print(results.get_metrics())
```

### 2.5 OpenAI Evals

**Focus:** OpenAI model evaluation
**Best For:** Evaluating GPT models, custom eval creation
**GitHub Stars:** 15k+

```python
# Using OpenAI Evals framework
from openai import OpenAI

client = OpenAI()

# Simple evaluation
def evaluate_response(query, response, reference):
    """Evaluate using GPT-4 as judge."""
    
    prompt = f"""Rate this response on a scale of 1-5:

Query: {query}
Response: {response}
Reference: {reference}

Rate on: accuracy (1-5), helpfulness (1-5), safety (1-5)
Return JSON: {{"accuracy": 1-5, "helpfulness": 1-5, "safety": 1-5}}"""
    
    result = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    return json.loads(result.choices[0].message.content)
```

---

## 3. Commercial Evaluation Platforms

### 3.1 Braintrust

**Focus:** End-to-end evaluation and proxy
**Best For:** Full-stack LLM evaluation with proxy
**Pricing:** Free tier → $5,000+/month

#### Features

| Feature | Description |
|---|---|
| Evaluation Proxy | Proxy all LLM calls for automatic evaluation |
| Custom Evaluators | Build custom evaluation criteria |
| A/B Testing | Built-in experiment management |
| Prompt Versioning | Track and compare prompt versions |
| Cost Tracking | Automatic cost monitoring |
| Real-time Monitoring | Live quality dashboards |

#### Setup

```python
import braintrust as bt

# Initialize
bt.init(api_key="your-api-key")

# Create evaluation
eval_obj = bt.Eval(
    project="my-llm-app",
    experiment="prompt-v2-evaluation",
    data=lambda: [
        {"input": "What is AI?", "expected": "AI is artificial intelligence..."},
        {"input": "Explain ML", "expected": "Machine learning is..."},
    ],
    task=lambda input: call_my_llm(input),
    scores=[
        bt.Score(name="accuracy", scorer=my_accuracy_scorer),
        bt.Score(name="helpfulness", scorer=my_helpfulness_scorer),
    ],
)

# Run and compare
results = eval_obj.run()
print(results.summary_metrics)

# Use as proxy
import braintrust

client = braintrust.openai.OpenAI(api_key="your-openai-key")
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
# Automatically logged and evaluated
```

### 3.2 LangSmith (LangChain)

**Focus:** LangChain ecosystem evaluation
**Best For:** LangChain/LangGraph applications
**Pricing:** $39/month → $399+/month

#### Features

```python
from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

# Define evaluator
def correctness(run, example):
    """Evaluate correctness."""
    prediction = run.outputs.get("output", "")
    reference = example.outputs.get("answer", "")
    
    # Simple string matching
    score = 1.0 if prediction.lower() == reference.lower() else 0.0
    
    return {"key": "correctness", "score": score}

# Run evaluation
results = evaluate(
    data="my-dataset",
    evaluators=[correctness],
    experiment_prefix="gpt-4o-eval",
    metadata={"model": "gpt-4o", "prompt_version": "v2.3"}
)

print(results)
```

### 3.3 Arize Phoenix

**Focus:** Observability + evaluation
**Best For:** Debugging, tracing, production monitoring
**Pricing:** Free tier → Enterprise

#### Features

```python
import phoenix as px
from phoenix.evals import (
    HallucinationEvaluator,
    RelevanceEvaluator,
    ToxicityEvaluator
)

# Start Phoenix server
px.launch_app()

# Log traces
from phoenix.otel import register

tracer_provider = register(project_name="my-llm-app")

# Evaluate logged traces
evaluator = HallucinationEvaluator()
results = evaluator.run(
    dataframe=logged_traces_df,
    model="gpt-4o"
)

# View in Phoenix UI
# http://localhost:6006
```

### 3.4 Patronus AI

**Focus:** Enterprise AI quality
**Best For:** Large-scale enterprise deployments
**Pricing:** Enterprise (custom)

#### Features

| Feature | Description |
|---|---|
| Auto-Evaluator | Automatic evaluation of LLM outputs |
| Custom Rules | Define custom quality rules |
| Compliance | Industry-specific compliance checks |
| Real-time API | Evaluate in production pipeline |
| Dashboard | Quality monitoring dashboard |

---

## 4. LLM-as-Judge Tools

### 4.1 Open-Source LLM-as-Judge Solutions

```python
class LLMJudgeToolkit:
    """Collection of LLM-as-Judge tools and patterns."""
    
    JUDGE_MODELS = {
        "strong": ["gpt-4o", "claude-3.5-sonnet", "gpt-4o-mini"],
        "fast": ["gpt-4o-mini", "claude-3-haiku", "llama-3.1-8b"],
        "free": ["llama-3.1-8b", "llama-3.1-70b", "mixtral-8x7b"]
    }
    
    async def judge(
        self,
        model: str,
        prompt: str,
        response: str,
        criteria: List[str],
        reference: str = None
    ) -> dict:
        """Generic LLM judge."""
        
        criteria_text = "\n".join(f"- {c}" for c in criteria)
        
        judge_prompt = f"""Rate this response on the following criteria:

Criteria:
{criteria_text}

Query: {prompt}
Response: {response}
{"Reference: " + reference if reference else ""}

For each criterion, provide a score from 0.0 to 1.0 and a brief rationale.

Return JSON:
{{
    "scores": {{"criterion_name": score, ...}},
    "overall": average_score,
    "rationale": "overall explanation"
}}"""
        
        result = await call_llm(model, judge_prompt)
        return parse_judge_result(result)
    
    async def multi_judge_consensus(
        self,
        judges: List[str],
        prompt: str,
        response: str,
        criteria: List[str],
        reference: str = None
    ) -> dict:
        """Multi-judge consensus evaluation."""
        
        results = []
        
        for judge_model in judges:
            result = await self.judge(
                judge_model, prompt, response, criteria, reference
            )
            results.append(result)
        
        # Aggregate scores
        all_scores = {}
        for criterion in criteria:
            scores = [r["scores"].get(criterion, 0) for r in results]
            all_scores[criterion] = {
                "mean": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
                "stdev": (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)) ** 0.5
            }
        
        overall_scores = [r["overall"] for r in results]
        
        return {
            "overall": sum(overall_scores) / len(overall_scores),
            "by_criterion": all_scores,
            "agreement": 1.0 - max(
                r.get("stdev", 0) for criterion_scores in all_scores.values()
                for r in [criterion_scores]
            ),
            "individual_results": results
        }
```

### 4.2 Comparison of Judge Models

| Model | Quality | Speed | Cost | Best For |
|---|---|---|---|---|
| GPT-4o | ⭐⭐⭐⭐⭐ | Medium | $5/1M tokens | High-stakes evaluation |
| GPT-4o-mini | ⭐⭐⭐⭐ | Fast | $0.15/1M tokens | Cost-effective evaluation |
| Claude 3.5 Sonnet | ⭐⭐⭐⭐⭐ | Medium | $3/1M tokens | Nuanced evaluation |
| Claude 3 Haiku | ⭐⭐⭐ | Fast | $0.25/1M tokens | Quick evaluation |
| Llama 3.1 8B | ⭐⭐⭐ | Fast | Free | Local evaluation |
| Llama 3.1 70B | ⭐⭐⭐⭐ | Medium | Free | High-quality local |
| Mixtral 8x7B | ⭐⭐⭐ | Medium | Free | Balanced local |

---

## 5. Safety Evaluation Tools

### 5.1 Open-Source Safety Tools

```python
class SafetyEvalToolkit:
    """Collection of safety evaluation tools."""
    
    async def evaluate_safety(
        self,
        response: str,
        tools: List[str] = None
    ) -> dict:
        """Run safety evaluation using multiple tools."""
        
        if tools is None:
            tools = ["llama_guard", "content_classifier", "pii_detector"]
        
        results = {}
        
        for tool in tools:
            if tool == "llama_guard":
                results["llama_guard"] = await self._run_llama_guard(response)
            elif tool == "content_classifier":
                results["content_classifier"] = await self._run_content_classifier(response)
            elif tool == "pii_detector":
                results["pii_detector"] = await self._run_pii_detector(response)
            elif tool == "toxicity_detector":
                results["toxicity_detector"] = await self._run_toxicity_detector(response)
        
        # Aggregate safety score
        safety_scores = [
            r.get("safety_score", 1.0) for r in results.values()
        ]
        
        return {
            "overall_safe": all(r.get("safe", True) for r in results.values()),
            "safety_score": sum(safety_scores) / len(safety_scores) if safety_scores else 1.0,
            "details": results,
            "flagged_categories": self._extract_flagged_categories(results)
        }
    
    async def _run_llama_guard(self, response: str) -> dict:
        """Run Llama Guard safety classifier."""
        
        from llama_guard import LlamaGuard
        
        guard = LlamaGuard()
        
        result = guard.classify(response)
        
        return {
            "tool": "llama_guard",
            "safe": result.safe,
            "safety_score": 1.0 if result.safe else 0.0,
            "violations": result.violations
        }
    
    async def _run_content_classifier(self, response: str) -> dict:
        """Run content safety classifier."""
        
        # Use OpenAI moderation or custom classifier
        from openai import OpenAI
        
        client = OpenAI()
        
        result = client.moderations.create(input=response)
        
        moderation = result.results[0]
        
        return {
            "tool": "content_classifier",
            "safe": not moderation.flagged,
            "safety_score": 1.0 if not moderation.flagged else 0.0,
            "flagged_categories": {
                k: v for k, v in moderation.categories.model_dump().items()
                if v
            }
        }
    
    async def _run_pii_detector(self, response: str) -> dict:
        """Run PII detection."""
        
        import re
        
        pii_patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
        }
        
        detected = {}
        for pii_type, pattern in pii_patterns.items():
            matches = re.findall(pattern, response)
            if matches:
                detected[pii_type] = matches
        
        return {
            "tool": "pii_detector",
            "safe": len(detected) == 0,
            "safety_score": 1.0 if len(detected) == 0 else 0.5,
            "detected_pii": detected
        }
```

### 5.2 Safety Evaluation Benchmarks

| Benchmark | Focus | Open Source | Scale |
|---|---|---|---|
| **TrustLLM** | Trustworthiness | ✅ | 6 criteria |
| **SafetyBench** | Chinese LLM safety | ✅ | 11 categories |
| **HarmBench** | Harmful content | ✅ | 510 behaviors |
| **BBQ** | Bias | ✅ | 11 categories |
| **CrowS-Pairs** | Bias | ✅ | 9 categories |
| **RealToxicityPrompts** | Toxicity | ✅ | 100K prompts |
| **XSTest** | Overrefusal | ✅ | 250 prompts |

---

## 6. Observability and Monitoring Platforms

### 6.1 LangFuse

**Focus:** LLM observability
**Best For:** Open-source, self-hosted observability
**Pricing:** Free tier → $59+/month

```python
from langfuse import Langfuse

# Initialize
langfuse = Langfuse(
    public_key="pk-...",
    secret_key="sk-...",
    host="https://cloud.langfuse.com"
)

# Trace a request
@langfuse.observe()
def process_query(query: str) -> str:
    # Your LLM call here
    response = call_llm(query)
    
    # Log evaluation
    langfuse.score(
        name="quality",
        value=0.85,
        comment="Good response quality"
    )
    
    return response

# Flush events
langfuse.flush()
```

### 6.2 Helicone

**Focus:** LLM proxy with analytics
**Best For:** Cost tracking, usage analytics
**Pricing:** Free tier → $20+/month

```python
import helicone as hc

# Initialize
hc.init(api_key="your-key")

# Use as proxy
response = hc.openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
    headers={"Helicone-Property-User-Id": "user-123"}
)

# View analytics in dashboard
```

### 6.3 Weights & Biases

**Focus:** Experiment tracking
**Best For:** ML experiment management
**Pricing:** Free tier → $50+/month

```python
import wandb

# Initialize
wandb.init(project="llm-evaluation")

# Log evaluation results
wandb.log({
    "accuracy": 0.87,
    "hallucination_rate": 0.05,
    "cost_per_request": 0.003,
    "latency_p95": 1200
})

# Create comparison table
table = wandb.Table(columns=["metric", "v1", "v2", "delta"])
table.add_data("accuracy", 0.82, 0.87, 0.05)
table.add_data("hallucination", 0.08, 0.05, -0.03)

wandb.log({"comparison": table})

wandb.finish()
```

### 6.4 MLflow

**Focus:** ML lifecycle management
**Best For:** Model tracking, deployment
**Pricing:** Free (open source)

```python
import mlflow

# Start run
mlflow.start_run(run_name="gpt-4o-evaluation")

# Log parameters
mlflow.log_params({
    "model": "gpt-4o",
    "temperature": 0.7,
    "max_tokens": 1000,
    "eval_suite": "production_v2"
})

# Log metrics
mlflow.log_metrics({
    "accuracy": 0.87,
    "faithfulness": 0.92,
    "hallucination_rate": 0.05,
    "cost_per_request": 0.003,
    "latency_p95": 1200
})

# Log evaluation report
mlflow.log_artifact("eval_report.html")

# End run
mlflow.end_run()
```

---

## 7. Evaluation Infrastructure Tools

### 7.1 Evaluation Data Management

```python
class EvalDataManager:
    """Manage evaluation data with versioning."""
    
    def __init__(self, storage_backend):
        self.storage = storage_backend
    
    async def create_version(
        self,
        name: str,
        test_cases: List[dict],
        description: str
    ) -> str:
        """Create versioned test suite."""
        
        version_id = f"v{int(time.time())}"
        
        snapshot = {
            "version_id": version_id,
            "name": name,
            "test_cases": test_cases,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "hash": self._compute_hash(test_cases)
        }
        
        await self.storage.save(
            f"eval_data/{name}/{version_id}.json",
            snapshot
        )
        
        return version_id
    
    async def load_version(
        self,
        name: str,
        version_id: str
    ) -> List[dict]:
        """Load specific version of test suite."""
        
        snapshot = await self.storage.load(
            f"eval_data/{name}/{version_id}.json"
        )
        
        return snapshot["test_cases"]
    
    async def list_versions(self, name: str) -> List[dict]:
        """List all versions of a test suite."""
        
        files = await self.storage.list_dir(f"eval_data/{name}/")
        
        versions = []
        for file in files:
            snapshot = await self.storage.load(file)
            versions.append({
                "version_id": snapshot["version_id"],
                "created_at": snapshot["created_at"],
                "description": snapshot["description"],
                "num_cases": len(snapshot["test_cases"]),
                "hash": snapshot["hash"]
            })
        
        return sorted(versions, key=lambda x: x["created_at"], reverse=True)
    
    def _compute_hash(self, test_cases: List[dict]) -> str:
        """Compute content hash."""
        import hashlib
        import json
        
        content = json.dumps(test_cases, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
```

### 7.2 Evaluation Scheduling

```python
class EvalScheduler:
    """Schedule and manage evaluation runs."""
    
    def __init__(self):
        self.schedules = {}
        self.running_evals = {}
    
    def create_schedule(
        self,
        name: str,
        cron_expression: str,
        eval_config: dict
    ):
        """Create an evaluation schedule."""
        
        self.schedules[name] = {
            "name": name,
            "cron": cron_expression,
            "config": eval_config,
            "created_at": datetime.now().isoformat(),
            "last_run": None,
            "next_run": self._calculate_next_run(cron_expression),
            "enabled": True
        }
    
    async def run_scheduled_eval(self, name: str):
        """Run a scheduled evaluation."""
        
        schedule = self.schedules.get(name)
        if not schedule or not schedule["enabled"]:
            return
        
        # Check if already running
        if name in self.running_evals:
            logger.warning(f"Evaluation {name} already running")
            return
        
        try:
            self.running_evals[name] = {
                "started_at": datetime.now().isoformat(),
                "status": "running"
            }
            
            # Run evaluation
            result = await self._run_evaluation(schedule["config"])
            
            # Update schedule
            schedule["last_run"] = datetime.now().isoformat()
            schedule["next_run"] = self._calculate_next_run(schedule["cron"])
            
            # Check for alerts
            if result.get("overall_score", 1.0) < schedule["config"].get("alert_threshold", 0.7):
                await self._trigger_alert(name, result)
            
            self.running_evals[name]["status"] = "completed"
            self.running_evals[name]["result"] = result
            
        except Exception as e:
            logger.error(f"Scheduled evaluation failed: {e}")
            self.running_evals[name]["status"] = "failed"
            self.running_evals[name]["error"] = str(e)
        
        finally:
            # Clean up after some time
            await asyncio.sleep(300)
            if name in self.running_evals:
                del self.running_evals[name]
    
    def _calculate_next_run(self, cron_expression: str) -> str:
        """Calculate next run time from cron expression."""
        # Simplified - in production use croniter or similar
        from datetime import timedelta
        
        # Default to daily
        return (datetime.now() + timedelta(days=1)).isoformat()
```

---

## 8. Tool Selection Guide

### 8.1 Decision Matrix

```
┌─────────────────────────────────────────────────────────────┐
│              TOOL SELECTION GUIDE                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  What are you evaluating?                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ RAG Pipeline → Ragas                                │   │
│  │ General LLM → DeepEval                              │   │
│  │ Model Capabilities → LM-eval-harness                │   │
│  │ Prompt Quality → PromptFlow                         │   │
│  │ Safety → Llama Guard + Custom classifiers           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  What's your deployment model?                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Self-hosted → LangFuse, Arize Phoenix, MLflow       │   │
│  │ Cloud → Braintrust, LangSmith, W&B                  │   │
│  │ Hybrid → Arize Phoenix + Cloud dashboard            │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  What's your budget?                                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Free → Ragas + DeepEval + LangFuse                  │   │
│  │ Low ($50-200/mo) → LangSmith + DeepEval             │   │
│  │ Medium ($200-1000/mo) → Braintrust                  │   │
│  │ High ($1000+/mo) → Patronus + Custom infrastructure│   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  What's your scale?                                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ <1K evals/day → DeepEval + LangFuse                 │   │
│  │ 1K-100K evals/day → Braintrust or custom pipeline   │   │
│  │ 100K+ evals/day → Custom infrastructure + sampling  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 Quick Selection Table

| Use Case | Recommended Tool(s) | Alternative |
|---|---|---|
| RAG pipeline evaluation | Ragas | DeepEval |
| CI/CD evaluation | DeepEval | PromptFlow |
| Model benchmarking | LM-eval-harness | OpenAI Evals |
| Production monitoring | LangFuse | Arize Phoenix |
| Cost tracking | Helicone | Braintrust |
| Experiment tracking | W&B | MLflow |
| Safety evaluation | Llama Guard + Custom | Patronus AI |
| Full platform | Braintrust | LangSmith |

---

## 9. Integration Patterns

### 9.1 CI/CD Integration Pattern

```python
# .github/workflows/llm-eval.yml
name: LLM Evaluation Pipeline

on:
  push:
    branches: [main]
    paths:
      - 'prompts/**'
      - 'eval/**'
      - 'config/**'
  pull_request:
    paths:
      - 'prompts/**'
      - 'eval/**'

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install deepeval ragas openai pandas
      
      - name: Run safety evaluation
        run: |
          python -m eval.run_safety \
            --suite safety_standard \
            --threshold 0.95 \
            --fail-on-violation
      
      - name: Run quality evaluation
        run: |
          python -m eval.run_quality \
            --suite regression \
            --baseline results/latest.json \
            --max-regression 0.02
      
      - name: Generate report
        if: always()
        run: |
          python -m eval.report \
            --format github-comment \
            --output eval-results.md
      
      - name: Comment PR
        if: github.event_name == 'pull_request'
        uses: marocchino/sticky-pull-request-comment@v2
        with:
          path: eval-results.md
          recreate: true
      
      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: eval-results
          path: eval-results/
          retention-days: 30
```

### 9.2 Production Monitoring Pattern

```python
# production_monitor.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
import asyncio

app = FastAPI()

class ProductionMonitor:
    """Monitor production LLM quality."""
    
    def __init__(self):
        self.metrics = {}
        self.alerts = []
        self.sample_rate = 0.1  # 10% sampling
    
    async def start(self):
        """Start monitoring background tasks."""
        
        asyncio.create_task(self._collect_metrics())
        asyncio.create_task(self._check_alerts())
        asyncio.create_task(self._generate_reports())
    
    async def _collect_metrics(self):
        """Continuously collect metrics."""
        
        while True:
            # Collect from various sources
            metrics = await self._gather_metrics()
            
            # Store metrics
            await self._store_metrics(metrics)
            
            # Check for anomalies
            anomalies = self._detect_anomalies(metrics)
            
            if anomalies:
                await self._handle_anomalies(anomalies)
            
            await asyncio.sleep(60)  # Collect every minute
    
    async def _check_alerts(self):
        """Check for alert conditions."""
        
        while True:
            metrics = await self._get_recent_metrics(hours=1)
            
            # Check safety violations
            safety_rate = metrics.get("safety_violation_rate", 0)
            if safety_rate > 0.01:  # >1% safety violations
                await self._send_alert(
                    "critical",
                    f"Safety violation rate: {safety_rate:.2%}"
                )
            
            # Check quality degradation
            quality_score = metrics.get("avg_quality_score", 1.0)
            if quality_score < 0.7:
                await self._send_alert(
                    "warning",
                    f"Quality score dropped to: {quality_score:.2f}"
                )
            
            # Check cost spikes
            cost_per_request = metrics.get("avg_cost_per_request", 0)
            if cost_per_request > 0.05:  # >$0.05 per request
                await self._send_alert(
                    "warning",
                    f"Cost per request: ${cost_per_request:.4f}"
                )
            
            await asyncio.sleep(300)  # Check every 5 minutes
    
    async def _generate_reports(self):
        """Generate periodic reports."""
        
        while True:
            # Daily report
            report = await self._generate_daily_report()
            await self._store_report(report)
            await self._send_report_email(report)
            
            await asyncio.sleep(86400)  # Daily

monitor = ProductionMonitor()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await monitor.start()
    yield

app = FastAPI(lifespan=lifespan)
```

### 9.3 Custom Evaluation Pipeline Pattern

```python
class CustomEvalPipeline:
    """Build custom evaluation pipeline."""
    
    def __init__(self, config: dict):
        self.config = config
        self.components = {}
    
    async def build_pipeline(self):
        """Build evaluation pipeline from config."""
        
        # Load components
        self.components["data_loader"] = DataLoader(self.config["data"])
        self.components["model"] = ModelLoader(self.config["model"])
        self.components["evaluator"] = EvaluatorFactory.create(
            self.config["evaluator"]
        )
        self.components["reporter"] = ReporterFactory.create(
            self.config["reporter"]
        )
        
        return self
    
    async def run(self) -> dict:
        """Run evaluation pipeline."""
        
        # Load test data
        test_cases = await self.components["data_loader"].load()
        
        # Run evaluation
        results = []
        for case in test_cases:
            # Get model response
            response = await self.components["model"].predict(case["input"])
            
            # Evaluate
            result = await self.components["evaluator"].evaluate(
                case, response
            )
            results.append(result)
        
        # Generate report
        report = await self.components["reporter"].generate(results)
        
        return report
```

---

## 10. Tool Comparison Matrix

### 10.1 Feature Comparison

| Feature | Ragas | DeepEval | Braintrust | LangSmith | Arize Phoenix |
|---|---|---|---|---|---|
| **RAG Evaluation** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **General LLM Eval** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **CI/CD Integration** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Production Monitoring** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **A/B Testing** | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Cost Tracking** | ⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Self-Hosted** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Documentation** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Community** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

### 10.2 Pricing Comparison

| Tool | Free Tier | Starter | Pro | Enterprise |
|---|---|---|---|---|
| **Ragas** | ✅ Open Source | N/A | N/A | N/A |
| **DeepEval** | ✅ Open Source | $0 (self-hosted) | $49/mo | Custom |
| **Braintrust** | 1K evals/mo | $50/mo | $200/mo | Custom |
| **LangSmith** | 5K runs/mo | $39/mo | $399/mo | Custom |
| **Arize Phoenix** | ✅ Open Source | $0 (self-hosted) | Cloud pricing | Custom |
| **LangFuse** | ✅ Open Source | $59/mo | $199/mo | Custom |
| **W&B** | ✅ Personal | $50/mo | $100/mo | Custom |
| **MLflow** | ✅ Open Source | N/A | N/A | Databricks |

### 10.3 Use Case Recommendations

| Scenario | Primary Tools | Supporting Tools |
|---|---|---|
| **Startup, budget-conscious** | Ragas + DeepEval + LangFuse | W&B free tier |
| **Enterprise, cloud-native** | Braintrust + LangSmith | Datadog, PagerDuty |
| **Enterprise, self-hosted** | DeepEval + Arize Phoenix + MLflow | Prometheus, Grafana |
| **Research team** | LM-eval-harness + W&B | Custom evaluation |
| **Production SaaS** | Braintrust + LangFuse + DeepEval | Custom monitoring |
| **Regulated industry** | Patronus AI + Custom compliance | Full audit trail |

---

## Cross-References

| Category | Document | Relevance |
|---|---|---|
| 06-Advanced | 03-Evaluation-Benchmarks.md | Academic benchmark tools |
| 20-Agent-Infrastructure | 02-AgentOps-Frameworks.md | Agent observability tools |
| 20-Agent-Infrastructure | 03-Agent-Tracing-and-Observability.md | Tracing tools |
| 23-Local-AI-Inference | 01-Overview.md | Local model evaluation |
| 33-AI-Native-Software-Dev | 03-AI-Native-CI-CD-and-DevOps.md | CI/CD integration |
| 41-AI-Cost-Optimization | 01-Overview.md | Cost optimization tools |
| 56-MLOps | 01-Overview.md | MLOps tooling |

---

**See Also:**
- `01-Overview.md` — Introduction to evaluation at scale
- `02-Core-Topics.md` — Essential evaluation topics
- `03-Technical-Deep-Dive.md` — Advanced evaluation techniques
- `05-Future-Outlook.md` — Future of AI evaluation

---
**See also:**
- [07 — AI Coding Agent Sustainability: Managing Hidden Costs, Code Quality Decay, and Developer Burnout](33-AI-Native-Software-Development/07-AI-Coding-Agent-Sustainability.md)
- [48 — MCP Cloud Infrastructure & Agent-as-a-Service: The Production Layer of Agentic AI](48-MCP-Cloud-Infrastructure-Agent-as-a-Service/01-Overview.md)
- [03 — Agent-Native Orchestration: LangGraph, Conductor, Mistral Workflows, Mcp-Agent](31-AI-Workflow-Orchestration-and-Durable-Execution/03-Agent-Native-Orchestration.md)
