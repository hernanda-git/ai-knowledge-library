# 52 — AI Hallucination Detection and Mitigation: Tools and Frameworks

> **Category:** 52 — AI Hallucination Detection and Mitigation  
> **Document:** 04 — Tools and Frameworks  
> **Last Updated:** July 2026  
> **Cross-references:** [01-Overview.md](01-Overview.md), [03-Technical-Deep-Dive.md](03-Technical-Deep-Dive.md), [05-Future-Outlook.md](05-Future-Outlook.md)

---

## Table of Contents

1. [Open-Source Detection Frameworks](#1-open-source-detection-frameworks)
2. [Commercial Hallucination Prevention Platforms](#2-commercial-hallucination-prevention-platforms)
3. [Observability and Monitoring Tools](#3-observability-and-monitoring-tools)
4. [Domain-Specific Tools](#4-domain-specific-tools)
5. [Evaluation Benchmarks and Datasets](#5-evaluation-benchmarks-and-datasets)
6. [Integration Patterns](#6-integration-patterns)
7. [Tool Comparison Matrix](#7-tool-comparison-matrix)
8. [Implementation Guides](#8-implementation-guides)

---

## 1. Open-Source Detection Frameworks

### 1.1 DeepEval

**Overview:** Open-source LLM evaluation framework with built-in hallucination detection metrics.

**Installation:**
```bash
pip install deepeval
```

**Key Features:**
- Hallucination metric using LLM-as-judge
- Faithfulness metric for RAG evaluation
- Integration with pytest for CI/CD
- Custom metric creation

**Implementation:**
```python
from deepeval import evaluate
from deepeval.metrics import HallucinationMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

# Define test cases
test_case = LLMTestCase(
    input="What is the capital of France?",
    actual_output="The capital of France is Paris, which is known for the Eiffel Tower built in 1902.",
    context=["France is a country in Western Europe. Paris is its capital city."]
)

# Create hallucination metric
hallucination_metric = HallucinationMetric(
    threshold=0.5,  # Maximum allowed hallucination score
    model="gpt-4",
    include_reason=True
)

# Create faithfulness metric
faithfulness_metric = FaithfulnessMetric(
    threshold=0.7,
    model="gpt-4"
)

# Evaluate
result = evaluate(
    test_cases=[test_case],
    metrics=[hallucination_metric, faithfulness_metric]
)

print(f"Hallucination Score: {result.test_results[0].metrics_data[0].score}")
print(f"Faithfulness Score: {result.test_results[0].metrics_data[1].score}")
```

**Custom Metric:**
```python
from deepeval.metrics import BaseMetric
from deepeval.models import DeepEvalBaseLLM

class CitationHallucinationMetric(BaseMetric):
    """Custom metric to detect citation hallucinations."""
    
    def __init__(self, threshold=0.5, model="gpt-4"):
        super().__init__()
        self.threshold = threshold
        self.model = DeepEvalBaseLLM(model=model)
    
    def measure(self, test_case):
        # Extract citations from actual output
        citations = self._extract_citations(test_case.actual_output)
        
        if not citations:
            self.score = 0  # No citations = no citation hallucination
            self.reason = "No citations found in output"
            return self.score
        
        # Verify each citation
        hallucinated_citations = []
        for citation in citations:
            if not self._verify_citation(citation, test_case.context):
                hallucinated_citations.append(citation)
        
        self.score = len(hallucinated_citations) / len(citations)
        self.reason = f"{len(hallucinated_citations)}/{len(citations)} citations appear fabricated"
        
        return self.score
    
    def _extract_citations(self, text):
        import re
        # Match common citation patterns
        patterns = [
            r'\(([A-Z][a-z]+ (?:et al\.?)?,? \d{4})\)',
            r'\[(\d+(?:,\s*\d+)*)\]',
            r'doi:(10\.\d{4,}/[^\s]+)',
        ]
        citations = []
        for pattern in patterns:
            citations.extend(re.findall(pattern, text))
        return citations
    
    def _verify_citation(self, citation, context):
        # Simplified verification - in practice use academic APIs
        return citation.lower() in context.lower()
    
    def is_successful(self):
        return self.score <= self.threshold
```

### 1.2 RAGAS (Retrieval Augmented Generation Assessment)

**Overview:** Framework specifically designed for evaluating RAG pipelines.

**Installation:**
```bash
pip install ragas
```

**Key Metrics:**
- **Faithfulness**: How much the answer is grounded in the context
- **Answer Relevancy**: How relevant the answer is to the question
- **Context Precision**: How precise the retrieved context is
- **Context Recall**: How much of the relevant context was retrieved

**Implementation:**
```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset

# Prepare evaluation dataset
eval_data = {
    "question": [
        "What is machine learning?",
        "How does neural network training work?",
    ],
    "answer": [
        "Machine learning is a subset of AI that enables systems to learn from data.",
        "Neural network training involves forward pass, loss computation, and backpropagation.",
    ],
    "contexts": [
        ["Machine learning is a branch of artificial intelligence that focuses on building systems that learn from data."],
        ["Training a neural network involves: 1) Forward propagation, 2) Loss calculation, 3) Backpropagation, 4) Weight update."],
    ],
    "ground_truth": [
        "Machine learning is a branch of AI that enables systems to learn from data without explicit programming.",
        "Neural network training uses backpropagation to update weights based on loss.",
    ]
}

dataset = Dataset.from_dict(eval_data)

# Evaluate
results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)

print(results)
# {'faithfulness': 0.95, 'answer_relevancy': 0.88, 
#  'context_precision': 0.92, 'context_recall': 0.85}
```

**Custom RAGAS Pipeline:**
```python
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Initialize components
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4"))
evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

# Custom faithfulness with source attribution
from ragas.metrics import Faithfulness

class AttributedFaithfulness(Faithfulness):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.include_attributions = True
    
    def score(self, dataset):
        # Add attribution tracking to faithfulness evaluation
        results = super().score(dataset)
        
        # Enhance with source attributions
        for i, row in enumerate(dataset):
            attributions = self._extract_attributions(
                row["answer"], row["contexts"]
            )
            results[i]["attributions"] = attributions
        
        return results
    
    def _extract_attributions(self, answer, contexts):
        """Map each claim in answer to supporting context."""
        # Implementation depends on specific use case
        return []
```

### 1.3 Guardrails AI

**Overview:** Framework for building reliable AI outputs with pre-built validators.

**Installation:**
```bash
pip install guardrails-ai
```

**Key Features:**
- Pre-built validators for hallucination detection
- Output formatting and constraints
- Real-time validation during generation
- Integration with major LLM providers

**Implementation:**
```python
from guardrails import Guard, OnFailAction
from guardrails.validators import (
    Validator,
    PassResult,
    FailResult,
    ValidationResult
)
import re

class HallucinationValidator(Validator):
    """Custom validator to detect hallucinations."""
    
    def __init__(self, context: str, threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.context = context
        self.threshold = threshold
    
    def validate(self, key: str, value: Any, schema: Dict) -> ValidationResult:
        """Validate that the value doesn't hallucinate beyond the context."""
        
        # Check if claims in value are supported by context
        claims = self._extract_claims(value)
        unsupported = []
        
        for claim in claims:
            if not self._is_supported(claim, self.context):
                unsupported.append(claim)
        
        if unsupported:
            return FailResult(
                metadata={"hallucinated_claims": unsupported},
                error_message=f"Unsupported claims detected: {unsupported}"
            )
        
        return PassResult()
    
    def _extract_claims(self, text):
        # Simple sentence-based claim extraction
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _is_supported(self, claim, context):
        # Simple keyword overlap check (in practice, use NLI)
        claim_words = set(claim.lower().split())
        context_words = set(context.lower().split())
        overlap = len(claim_words & context_words) / len(claim_words)
        return overlap > 0.3

# Create guard
guard = Guard().use(
    HallucinationValidator(
        context="France is a country in Europe. Paris is its capital.",
        threshold=0.5,
        on_fail=OnFailAction.REASK
    )
)

# Validate output
result = guard.validate(
    "The capital of France is Paris, which is a beautiful city."
)
print(result)  # Pass

result = guard.validate(
    "The capital of France is Lyon, known for its cuisine."
)
print(result)  # Fail - Lyon is not the capital
```

### 1.4 Langfuse (Observability + Evaluation)

**Overview:** Open-source LLM observability platform with evaluation capabilities.

**Installation:**
```bash
pip install langfuse
```

**Key Features:**
- Tracing and debugging LLM applications
- Prompt version management
- Evaluation datasets and scoring
- Cost tracking and optimization

**Implementation:**
```python
from langfuse import Langfuse
from langfuse.decorators import observe

# Initialize
langfuse = Langfuse(
    public_key="pk-...",
    secret_key="sk-...",
    host="https://cloud.langfuse.com"
)

@observe()
async def generate_with_evaluation(query, context):
    """Generate response with built-in evaluation."""
    
    # Create trace
    trace = langfuse.trace(
        name="hallucination_detection",
        metadata={"query": query}
    )
    
    # Generate response
    response = await llm.generate(
        f"Context: {context}\n\nQuestion: {query}",
        temperature=0
    )
    
    # Log generation
    generation = trace.generation(
        name="response_generation",
        input={"query": query, "context": context},
        output=response,
        model="gpt-4",
        usage={"total": len(response.split())}
    )
    
    # Evaluate for hallucination
    hallucination_score = await evaluate_hallucination(query, response, context)
    
    # Log evaluation score
    trace.score(
        name="hallucination_score",
        value=hallucination_score,
        comment=f"Auto-detected hallucination probability: {hallucination_score}"
    )
    
    # If hallucination detected, log for review
    if hallucination_score > 0.5:
        trace.event(
            name="hallucination_detected",
            metadata={
                "score": hallucination_score,
                "response": response,
                "requires_review": True
            }
        )
    
    return response, hallucination_score

async def evaluate_hallucination(query, response, context):
    """Evaluate hallucination using LLM-as-judge."""
    
    eval_prompt = f"""Rate the following response for hallucination on a scale of 0-1:

Query: {query}
Context: {context}
Response: {response}

Criteria:
- Are all factual claims supported by the context?
- Are there any fabricated details?
- Is the information accurate?

Score (0-1, where 1 = no hallucination):"""
    
    score_response = await llm.generate(eval_prompt, temperature=0)
    
    # Parse score
    import re
    score_match = re.search(r'(\d+\.?\d*)', score_response)
    if score_match:
        return 1 - float(score_match.group(1))  # Invert so 1 = hallucination
    
    return 0.5  # Default uncertainty
```

---

## 2. Commercial Hallucination Prevention Platforms

### 2.1 Lakera Guard

**Overview:** Real-time AI security and hallucination prevention platform.

**Key Features:**
- Real-time prompt injection detection
- Hallucination prevention
- Content filtering
- PII detection

**Implementation:**
```python
import lakera

client = lakera.Client(api_key="your-api-key")

def generate_with_lakera_guard(query, context):
    """Generate response with Lakera Guard protection."""
    
    # Check input for manipulation
    input_check = client.prompt_input_check(
        prompt=f"Context: {context}\n\nQuestion: {query}"
    )
    
    if input_check.is_injection_detected:
        return {
            "blocked": True,
            "reason": "Potential prompt injection detected",
            "details": input_check.injection_details
        }
    
    # Generate response
    response = llm.generate(f"Context: {context}\n\nQuestion: {query}")
    
    # Check output for hallucination
    output_check = client.prompt_output_check(
        prompt=f"Context: {context}\n\nQuestion: {query}",
        response=response
    )
    
    if output_check.is_hallucination_detected:
        return {
            "blocked": True,
            "reason": "Hallucination detected in response",
            "confidence": output_check.hallucination_confidence,
            "details": output_check.hallucination_details
        }
    
    return {
        "blocked": False,
        "response": response
    }
```

### 2.2 Patronus AI

**Overview:** Enterprise LLM quality and safety platform.

**Key Features:**
- Automated hallucination detection
- Custom evaluation criteria
- A/B testing for LLM outputs
- Compliance monitoring

**Implementation:**
```python
from patronus import PatronusClient

client = PatronusClient(api_key="your-api-key")

def evaluate_with_patronus(query, context, response):
    """Evaluate response using Patronus AI."""
    
    # Run evaluation
    result = client.evaluate(
        model_output=response,
        criteria={
            "hallucination": {
                "weight": 0.4,
                "description": "Does the response contain fabricated information?"
            },
            "faithfulness": {
                "weight": 0.3,
                "description": "Is the response faithful to the provided context?"
            },
            "relevance": {
                "weight": 0.3,
                "description": "Is the response relevant to the query?"
            }
        },
        context=context,
        query=query
    )
    
    return {
        "overall_score": result.overall_score,
        "hallucination_score": result.scores["hallucination"],
        "faithfulness_score": result.scores["faithfulness"],
        "relevance_score": result.scores["relevance"],
        "passed": result.overall_score > 0.7,
        "details": result.explanation
    }
```

### 2.3 Vectara HHEM (Hughes Hallucination Evaluation Model)

**Overview:** Open-source model specifically trained for hallucination detection.

**Installation:**
```bash
pip install vectara-evaluate
```

**Implementation:**
```python
from vectara_evaluate import evaluate

# Simple hallucination detection
def detect_with_hhem(context, response):
    """Detect hallucination using HHEM."""
    
    result = evaluate(
        model_output=response,
        context=context,
        task_type="summarization"  # or "retrieval", "qa"
    )
    
    return {
        "hallucination_score": result["hallucination_score"],
        "is_hallucinated": result["hallucination_score"] > 0.5,
        "details": result
    }

# Batch evaluation
def batch_evaluate_hhem(dataset):
    """Evaluate multiple samples."""
    
    results = []
    for sample in dataset:
        result = evaluate(
            model_output=sample["response"],
            context=sample["context"],
            task_type=sample.get("task_type", "summarization")
        )
        results.append(result)
    
    # Aggregate metrics
    hallucination_scores = [r["hallucination_score"] for r in results]
    
    return {
        "mean_hallucination": sum(hallucination_scores) / len(hallucination_scores),
        "max_hallucination": max(hallucination_scores),
        "min_hallucination": min(hallucination_scores),
        "hallucination_rate": sum(1 for s in hallucination_scores if s > 0.5) / len(hallucination_scores),
        "detailed_results": results
    }
```

---

## 3. Observability and Monitoring Tools

### 3.1 Arize Phoenix

**Overview:** Open-source observability platform for LLM applications.

**Installation:**
```bash
pip install arize-phoenix
```

**Key Features:**
- LLM tracing and debugging
- Embedding drift detection
- Evaluation dataset management
- RAG-specific metrics

**Implementation:**
```python
import phoenix as px
from phoenix.otel import trace

# Initialize Phoenix
px.launch_app()

# Set up tracing
tracer = trace.init_project_name("hallucination-detection")

async def trace_llm_call(query, context):
    """Trace LLM calls with hallucination monitoring."""
    
    with tracer.start_as_current_span("llm_call") as span:
        # Log inputs
        span.set_attribute("query", query)
        span.set_attribute("context_length", len(context))
        
        # Generate response
        response = await llm.generate(f"Context: {context}\n\nQuestion: {query}")
        
        # Log output
        span.set_attribute("response_length", len(response))
        span.set_attribute("response_preview", response[:200])
        
        # Run hallucination check
        hallucination_score = await check_hallucination(query, response, context)
        
        # Log hallucination score
        span.set_attribute("hallucination_score", hallucination_score)
        
        # Alert if high hallucination
        if hallucination_score > 0.7:
            span.add_event("high_hallucination_detected", {
                "score": hallucination_score,
                "requires_review": True
            })
        
        return response, hallucination_score

# Analyze in Phoenix UI
# Access at http://localhost:6006
```

### 3.2 Weights & Biases (W&B) Weave

**Overview:** LLM observability and evaluation platform.

**Installation:**
```bash
pip install weave
```

**Implementation:**
```python
import weave

# Initialize
weave.init("hallucination-monitoring")

class HallucinationEvaluator:
    """Evaluate LLM outputs for hallucinations using W&B Weave."""
    
    @weave.op()
    async def evaluate(self, query, context, response):
        """Evaluate a single response."""
        
        # Hallucination check
        hallucination_score = await self._check_hallucination(query, response, context)
        
        # Faithfulness check
        faithfulness_score = await self._check_faithfulness(response, context)
        
        # Log to W&B
        weave.log({
            "query": query,
            "response": response,
            "hallucination_score": hallucination_score,
            "faithfulness_score": faithfulness_score
        })
        
        return {
            "hallucination_score": hallucination_score,
            "faithfulness_score": faithfulness_score,
            "passed": hallucination_score < 0.5 and faithfulness_score > 0.7
        }
    
    async def _check_hallucination(self, query, response, context):
        """Check for hallucination."""
        prompt = f"""Rate this response for hallucination (0-1):
Query: {query}
Context: {context}
Response: {response}

Score (0=no hallucination, 1=complete hallucination):"""
        
        score = await llm.generate(prompt, temperature=0)
        return float(re.search(r'(\d+\.?\d*)', score).group(1))
    
    async def _check_faithfulness(self, response, context):
        """Check faithfulness to context."""
        prompt = f"""Rate how faithful this response is to the context (0-1):
Context: {context}
Response: {response}

Score (0=not faithful, 1=fully faithful):"""
        
        score = await llm.generate(prompt, temperature=0)
        return float(re.search(r'(\d+\.?\d*)', score).group(1))

# Use evaluator
evaluator = HallucinationEvaluator()
result = await evaluator.evaluate(
    query="What is ML?",
    context="Machine learning is a subset of AI.",
    response="Machine learning is a branch of artificial intelligence."
)
```

### 3.3 Helicone

**Overview:** LLM proxy with built-in analytics and monitoring.

**Implementation:**
```python
import helicone

# Initialize
helicone.init(api_key="your-api-key")

@helicone.log
async def generate_with_monitoring(query, context):
    """Generate response with Helicone monitoring."""
    
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"Context: {context}"},
            {"role": "user", "content": query}
        ],
        helicone_properties={
            "context_length": len(context),
            "query_length": len(query)
        }
    )
    
    # Run hallucination check
    hallucination_score = await check_hallucination(query, response.choices[0].message.content, context)
    
    # Log hallucination metric
    helicone.log_request(
        metrics={"hallucination_score": hallucination_score}
    )
    
    return response
```

---

## 4. Domain-Specific Tools

### 4.1 Medical: Med-PaLM Evaluation

```python
class MedicalHallucinationChecker:
    """Detect hallucinations in medical AI outputs."""
    
    def __init__(self):
        self.medical_databases = {
            "pubmed": PubMedClient(),
            "drugbank": DrugBankClient(),
            "icd10": ICD10Client()
        }
    
    async def check_medical_response(self, query, response):
        """Check medical response for hallucinations."""
        
        results = {
            "drug_safety": await self._check_drug_safety(response),
            "dosage_accuracy": await self._check_dosage_accuracy(response),
            "study_citations": await self._verify_study_citations(response),
            "protocol_adherence": await self._check_protocol(response)
        }
        
        # Calculate overall safety score
        safety_score = 1.0
        for check in results.values():
            if check.get("unsafe", False):
                safety_score *= 0.5
        
        return {
            "safety_score": safety_score,
            "requires_physician_review": safety_score < 0.8,
            "details": results
        }
    
    async def _check_drug_safety(self, response):
        """Check for dangerous drug interactions or dosages."""
        # Extract drug mentions
        drugs = self._extract_drug_names(response)
        
        for drug in drugs:
            # Check against DrugBank
            drug_info = await self.medical_databases["drugbank"].get_drug(drug)
            if drug_info and drug_info.get("black_box_warning"):
                return {"unsafe": True, "warning": f"{drug} has black box warning"}
        
        return {"unsafe": False}
```

### 4.2 Legal: Case Law Verification

```python
class LegalHallucinationChecker:
    """Detect hallucinations in legal AI outputs."""
    
    def __init__(self):
        self.legal_apis = {
            "courtlistener": CourtListenerClient(),
            "casetext": CasetextClient()
        }
    
    async def check_legal_response(self, response, jurisdiction):
        """Check legal response for hallucinations."""
        
        citations = self._extract_citations(response)
        
        results = []
        for citation in citations:
            if citation["type"] == "case":
                verification = await self._verify_case_citation(citation)
            elif citation["type"] == "statute":
                verification = await self._verify_statute(citation, jurisdiction)
            else:
                verification = {"verified": False, "reason": "Unknown citation type"}
            
            results.append({**citation, **verification})
        
        fabricated = [r for r in results if not r.get("verified", False)]
        
        return {
            "total_citations": len(results),
            "fabricated": len(fabricated),
            "fabrication_rate": len(fabricated) / max(len(results), 1),
            "requires_attorney_review": len(fabricated) > 0,
            "details": results
        }
```

### 4.3 Financial: Data Verification

```python
class FinancialHallucinationChecker:
    """Detect hallucinations in financial AI outputs."""
    
    def __init__(self):
        self.data_sources = {
            "sec": SECClient(),
            "bloomberg": BloombergClient(),
            "refinitiv": RefinitivClient()
        }
    
    async def check_financial_response(self, response, company=None):
        """Check financial response for hallucinations."""
        
        financial_claims = self._extract_financial_claims(response)
        
        results = []
        for claim in financial_claims:
            if claim["type"] == "revenue":
                verification = await self._verify_revenue(claim, company)
            elif claim["type"] == "stock_price":
                verification = await self._verify_stock_price(claim)
            elif claim["type"] == "regulation":
                verification = await self._verify_regulation(claim)
            else:
                verification = {"verified": False}
            
            results.append({**claim, **verification})
        
        return {
            "total_claims": len(results),
            "verified": sum(1 for r in results if r.get("verified")),
            "fabricated": sum(1 for r in results if not r.get("verified")),
            "details": results
        }
```

---

## 5. Evaluation Benchmarks and Datasets

### 5.1 Standard Benchmarks

| Benchmark | Focus | Size | Metrics |
|-----------|-------|------|---------|
| **TruthfulQA** | General truthfulness | 817 questions | Truthful%, Informative% |
| **HaluEval** | Hallucination evaluation | 35K samples | Detection accuracy |
| **FActScore** | Fine-grained factuality | 10K claims | Atomic fact precision/recall |
| **RAGAS** | RAG-specific | Custom | Faithfulness, relevancy |
| **FEVER** | Fact verification | 185K claims | Accuracy, F1 |
| **AmbigQA** | Ambiguous questions | 14K samples | Refusal rate |
| **Truth Forest** | Comprehensive | Multi-dimensional | Hallucination assessment |

### 5.2 Creating Custom Benchmarks

```python
class HallucinationBenchmark:
    """Create and manage hallucination benchmarks."""
    
    def __init__(self, name, domain):
        self.name = name
        self.domain = domain
        self.cases = []
    
    def add_case(self, query, context, expected_answer, hallucination_type=None):
        """Add a test case."""
        self.cases.append({
            "query": query,
            "context": context,
            "expected_answer": expected_answer,
            "hallucination_type": hallucination_type,
            "domain": self.domain
        })
    
    def add_factual_case(self, query, context, correct_answer):
        """Add a factual verification case."""
        self.add_case(query, context, correct_answer, hallucination_type="factual")
    
    def add_refusal_case(self, query, context):
        """Add a case where the model should refuse to answer."""
        self.add_case(
            query, context, 
            "I don't have enough information to answer this question.",
            hallucination_type="refusal"
        )
    
    def add_citation_case(self, query, context, real_citation):
        """Add a citation verification case."""
        self.add_case(query, context, real_citation, hallucination_type="citation")
    
    def run_evaluation(self, evaluator):
        """Run evaluation against an evaluator function."""
        
        results = []
        for case in self.cases:
            # Get evaluator response
            response = evaluator(case["query"], case["context"])
            
            # Evaluate
            score = self._evaluate_response(case, response)
            
            results.append({
                "case": case,
                "response": response,
                "score": score
            })
        
        # Aggregate
        scores = [r["score"] for r in results]
        
        return {
            "benchmark": self.name,
            "domain": self.domain,
            "total_cases": len(self.cases),
            "average_score": sum(scores) / len(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "hallucination_rate": sum(1 for s in scores if s < 0.5) / len(scores) if scores else 0,
            "detailed_results": results
        }
    
    def _evaluate_response(self, case, response):
        """Evaluate a single response."""
        # Simplified evaluation
        if case["hallucination_type"] == "refusal":
            return 1.0 if "don't know" in response.lower() or "insufficient" in response.lower() else 0.0
        elif case["hallucination_type"] == "factual":
            return 1.0 if case["expected_answer"].lower() in response.lower() else 0.0
        else:
            return 0.5  # Default neutral score
```

### 5.3 Benchmark Datasets

```python
# Load standard benchmarks
from datasets import load_dataset

def load_truthfulqa():
    """Load TruthfulQA benchmark."""
    dataset = load_dataset("truthfulqa/truthful_qa", "generation")
    return dataset

def load_halueval():
    """Load HaluEval benchmark."""
    dataset = load_dataset("liamcripwell/halu_eval")
    return dataset

def load_rag_benchmark():
    """Load RAG-specific benchmark."""
    dataset = load_dataset("explodinggradients/ragas")
    return dataset

# Custom dataset creation
def create_medical_benchmark():
    """Create a medical hallucination benchmark."""
    
    benchmark = HallucinationBenchmark("Medical Hallucination Benchmark", "medical")
    
    # Add medical test cases
    benchmark.add_factual_case(
        query="What is the recommended dosage of aspirin for adults?",
        context="Aspirin dosage for adults: 325-650mg every 4-6 hours as needed.",
        correct_answer="325-650mg every 4-6 hours"
    )
    
    benchmark.add_refusal_case(
        query="What is the cure for cancer?",
        context="Cancer is a complex disease with multiple treatment approaches."
    )
    
    benchmark.add_citation_case(
        query="What does the study by Smith et al. 2024 say about aspirin?",
        context="Smith et al. 2024 found that low-dose aspirin reduces cardiovascular risk.",
        real_citation="Smith et al., 2024"
    )
    
    return benchmark
```

---

## 6. Integration Patterns

### 6.1 API Gateway Pattern

```python
class HallucinationGuardGateway:
    """API gateway with hallucination detection."""
    
    def __init__(self, detector, rate_limiter):
        self.detector = detector
        self.rate_limiter = rate_limiter
    
    async def handle_request(self, request):
        """Handle incoming request with hallucination detection."""
        
        # Rate limiting
        if not self.rate_limiter.check(request.client_id):
            return {"error": "Rate limit exceeded"}, 429
        
        # Input validation
        if not self._validate_input(request):
            return {"error": "Invalid input"}, 400
        
        # Generate response
        response = await self._generate(request.query, request.context)
        
        # Detect hallucination
        detection = self.detector.detect(request.context, response)
        
        # Apply policy
        if detection["is_hallucinated"]:
            if detection["hallucination_probability"] > 0.8:
                # Block high-confidence hallucinations
                return {
                    "error": "Response blocked due to potential hallucination",
                    "confidence": detection["hallucination_probability"]
                }, 422
            else:
                # Flag moderate hallucinations for review
                return {
                    "response": response,
                    "warning": "Response may contain inaccuracies",
                    "hallucination_score": detection["hallucination_probability"],
                    "requires_review": True
                }, 200
        else:
            return {"response": response}, 200
    
    def _validate_input(self, request):
        """Validate input request."""
        return bool(request.query and len(request.query) > 0)
```

### 6.2 Middleware Pattern

```python
from functools import wraps

def hallucination_guard(detector, threshold=0.5):
    """Decorator to add hallucination detection to any function."""
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get context and query from arguments
            context = kwargs.get("context", "")
            query = kwargs.get("query", args[0] if args else "")
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Detect hallucination
            if isinstance(result, str):
                detection = detector.detect(context, result)
                
                if detection["is_hallucinated"] and detection["hallucination_probability"] > threshold:
                    # Log hallucination
                    log_hallucination(query, result, detection)
                    
                    # Optionally block or modify response
                    if detection["hallucination_probability"] > 0.8:
                        raise HallucinationError(
                            f"High hallucination detected: {detection['hallucination_probability']:.2f}"
                        )
            
            return result
        return wrapper
    return decorator

# Usage
@hallucination_guard(detector=hallucination_detector, threshold=0.6)
async def generate_response(query, context):
    return await llm.generate(f"Context: {context}\n\nQuestion: {query}")
```

### 6.3 Pipeline Pattern

```python
class HallucinationDetectionPipeline:
    """Multi-stage hallucination detection pipeline."""
    
    def __init__(self):
        self.stages = [
            ("pre_check", self._pre_check),
            ("generation", self._generation),
            ("post_check", self._post_check),
            ("verification", self._verification),
            ("output", self._output)
        ]
    
    async def execute(self, query, context, config=None):
        """Execute the full pipeline."""
        
        state = {
            "query": query,
            "context": context,
            "config": config or {},
            "flags": [],
            "scores": {}
        }
        
        for stage_name, stage_fn in self.stages:
            state = await stage_fn(state)
            
            if state.get("blocked"):
                return {
                    "blocked": True,
                    "stage": stage_name,
                    "reason": state.get("block_reason"),
                    "state": state
                }
        
        return {
            "blocked": False,
            "response": state.get("response"),
            "scores": state.get("scores"),
            "flags": state.get("flags")
        }
    
    async def _pre_check(self, state):
        """Pre-generation checks."""
        # Check for prompt injection
        if self._detect_injection(state["query"]):
            state["blocked"] = True
            state["block_reason"] = "Potential prompt injection detected"
        
        # Check context quality
        if len(state["context"]) < 10:
            state["flags"].append("low_context_quality")
        
        return state
    
    async def _generation(self, state):
        """Generate response."""
        state["response"] = await llm.generate(
            f"Context: {state['context']}\n\nQuestion: {state['query']}",
            temperature=0
        )
        return state
    
    async def _post_check(self, state):
        """Post-generation checks."""
        # Basic quality checks
        if len(state["response"]) < 10:
            state["flags"].append("short_response")
        
        # Check for refusal patterns
        refusal_patterns = ["i don't know", "i cannot", "insufficient information"]
        if any(pattern in state["response"].lower() for pattern in refusal_patterns):
            state["flags"].append("refusal_detected")
        
        return state
    
    async def _verification(self, state):
        """Detailed hallucination verification."""
        detection = await self._run_detection(state["query"], state["response"], state["context"])
        
        state["scores"]["hallucination"] = detection["hallucination_probability"]
        state["scores"]["faithfulness"] = detection.get("faithfulness_score", 0)
        
        if detection["hallucination_probability"] > 0.8:
            state["blocked"] = True
            state["block_reason"] = "High hallucination probability"
        elif detection["hallucination_probability"] > 0.5:
            state["flags"].append("moderate_hallucination_risk")
        
        return state
    
    async def _output(self, state):
        """Final output preparation."""
        # Add metadata
        state["metadata"] = {
            "scores": state["scores"],
            "flags": state["flags"],
            "pipeline_version": "1.0"
        }
        
        return state
    
    def _detect_injection(self, text):
        """Detect prompt injection."""
        import re
        patterns = [
            r"ignore previous",
            r"you are now",
            r"system prompt",
            r"forget everything"
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)
    
    async def _run_detection(self, query, response, context):
        """Run hallucination detection."""
        # Placeholder for actual detection logic
        return {"hallucination_probability": 0.1, "faithfulness_score": 0.9}
```

---

## 7. Tool Comparison Matrix

| Tool | Type | Hallucination Detection | Real-time | Domain-Specific | Cost |
|------|------|------------------------|-----------|-----------------|------|
| **DeepEval** | Open-source | ✅ LLM-as-judge | ❌ | ❌ | Free |
| **RAGAS** | Open-source | ✅ Faithfulness | ❌ | ❌ | Free |
| **Guardrails AI** | Open-source | ✅ Validators | ✅ | ❌ | Free |
| **Langfuse** | Open-source | ✅ Tracing | ✅ | ❌ | Free (self-hosted) |
| **Arize Phoenix** | Open-source | ✅ Observability | ✅ | ❌ | Free |
| **W&B Weave** | Commercial | ✅ Evaluation | ✅ | ❌ | Paid |
| **Lakera Guard** | Commercial | ✅ Real-time | ✅ | ❌ | Paid |
| **Patronus AI** | Commercial | ✅ Enterprise | ✅ | ✅ | Paid |
| **Vectara HHEM** | Open-source | ✅ Model-based | ✅ | ❌ | Free |

---

## 8. Implementation Guides

### 8.1 Quick Start: Basic Hallucination Detection

```python
# 1. Install dependencies
# pip install deepeval ragas guardrails-ai

# 2. Basic detection setup
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase

async def basic_hallucination_check(query, context, response):
    """Quick hallucination check."""
    
    test_case = LLMTestCase(
        input=query,
        actual_output=response,
        context=[context]
    )
    
    metric = HallucinationMetric(threshold=0.5)
    metric.measure(test_case)
    
    return {
        "score": metric.score,
        "passed": metric.is_successful(),
        "reason": metric.reason
    }
```

### 8.2 Production Setup: Multi-Layer Defense

```python
# Production-ready hallucination defense
class ProductionHallucinationDefense:
    def __init__(self, config):
        self.config = config
        
        # Layer 1: Prompt engineering
        self.prompt_engineer = PromptEngineer(config.prompt_config)
        
        # Layer 2: RAG hardening
        self.rag_harden = RAGHardening(config.rag_config)
        
        # Layer 3: Post-generation detection
        self.detector = EnsembleDetector(config.detector_config)
        
        # Layer 4: Monitoring
        self.monitor = ProductionMonitor(config.monitor_config)
    
    async def process(self, query, context, metadata=None):
        """Process request through all defense layers."""
        
        # Layer 1: Harden prompts
        hardened_prompt = self.prompt_engineer.harden(query, context)
        
        # Layer 2: Harden RAG
        hardened_context = self.rag_harden.harden(context)
        
        # Generate response
        response = await self._generate(hardened_prompt, hardened_context)
        
        # Layer 3: Detect hallucination
        detection = self.detector.detect(hardened_context, response)
        
        # Layer 4: Monitor
        await self.monitor.record(query, response, detection, metadata)
        
        # Apply policy
        if detection["hallucination_probability"] > self.config.block_threshold:
            return self._block_response(detection)
        elif detection["hallucination_probability"] > self.config.flag_threshold:
            return self._flag_response(response, detection)
        else:
            return {"response": response, "detection": detection}
    
    async def _generate(self, prompt, context):
        """Generate response."""
        return await llm.generate(f"Context: {context}\n\n{prompt}")
    
    def _block_response(self, detection):
        return {
            "blocked": True,
            "reason": "High hallucination risk",
            "score": detection["hallucination_probability"]
        }
    
    def _flag_response(self, response, detection):
        return {
            "response": response,
            "warning": "Response may contain inaccuracies",
            "hallucination_score": detection["hallucination_probability"],
            "requires_review": True
        }
```

### 8.3 Evaluation Setup: Testing Your System

```python
# Setup comprehensive evaluation
async def evaluate_hallucination_system(system, benchmark):
    """Evaluate your hallucination detection system."""
    
    results = []
    
    for case in benchmark.cases:
        # Process through system
        output = await system.process(case["query"], case["context"])
        
        # Evaluate
        score = evaluate_case(case, output)
        
        results.append({
            "case": case,
            "output": output,
            "score": score
        })
    
    # Calculate metrics
    scores = [r["score"] for r in results]
    
    return {
        "total_cases": len(results),
        "average_score": sum(scores) / len(scores),
        "hallucination_detection_rate": sum(1 for s in scores if s > 0.5) / len(scores),
        "false_positive_rate": calculate_fpr(results),
        "false_negative_rate": calculate_fnr(results),
        "detailed_results": results
    }
```

---

## Cross-References

| Document | Relevance |
|----------|-----------|
| [01-Overview.md](01-Overview.md) | General overview |
| [02-Core-Topics.md](02-Core-Topics.md) | Core techniques |
| [03-Technical-Deep-Dive.md](03-Technical-Deep-Dive.md) | Advanced algorithms |
| [05-Future-Outlook.md](05-Future-Outlook.md) | Future directions |
| [20-Agent-Infrastructure/02-AgentOps-Frameworks.md](../../20-Agent-Infrastructure-and-Observability/02-AgentOps-Frameworks.md) | AgentOps platform comparison |
| [13-Top-Demand/04-Multimodal-AI.md](../../13-Top-Demand/04-Multimodal-AI.md) | Multimodal hallucination challenges |

---

*Last updated: July 2026*
