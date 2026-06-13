# 04 — Agent Evaluation and Testing

## 1. Introduction to Agent Evaluation

### 1.1 Why Agent Evaluation is Different

Evaluating an AI agent is fundamentally different from evaluating a traditional ML model or an LLM:

| Aspect | Traditional ML | LLM Call | Agent |
|--------|---------------|----------|-------|
| Unit of evaluation | Single prediction | Single response | Multi-step task |
| Determinism | Deterministic | Low determinism | Very low (branching paths) |
| Success criteria | Accuracy, F1, RMSE | Correctness, relevance | Task completion, efficiency |
| Data requirements | Labeled dataset | Reference responses | Task definitions + rubrics |
| Failure modes | Wrong prediction | Hallucination, off-topic | Hallucination cascade, loops, tool misuse, context overflow |
| Evaluation cost | Compute + labeling | LLM calls for judge | 10-100x LLM cost + human review |

An agent's output is not just text — it's a sequence of actions that produces a result. Evaluating that result requires understanding both the process (did the agent make reasonable decisions?) and the outcome (did the user's task get completed satisfactorily?).

### 1.2 The Evaluation Dimensions

Agent evaluation spans multiple dimensions:

1. **Task Completion**: Did the agent achieve the user's goal?
2. **Step Efficiency**: Did the agent use a reasonable number of steps and tools?
3. **Output Quality**: Is the final response accurate, relevant, and well-formed?
4. **Safety**: Did the agent avoid harmful actions, biased outputs, or security violations?
5. **Robustness**: Does the agent handle edge cases, unusual inputs, and failures gracefully?
6. **Cost Efficiency**: Did the agent accomplish the task within acceptable cost bounds?
7. **Latency**: Did the agent complete the task within acceptable time?

## 2. Evaluation Metrics

### 2.1 Task-Level Metrics

#### Task Completion Rate (TCR)

The fundamental metric: did the agent successfully complete the user's task?

```python
"""Task completion rate calculation."""

from enum import Enum

class TaskOutcome(Enum):
    SUCCESS = "success"          # Task fully completed
    PARTIAL = "partial"          # Task partially completed
    FAILURE = "failure"          # Task failed or wrong result
    ESCALATED = "escalated"      # Escalated to human
    INCORRECT = "incorrect"      # Completed but result is wrong

def calculate_task_completion_rate(outcomes: list[TaskOutcome]) -> dict:
    """Calculate task completion rate and related metrics."""
    total = len(outcomes)
    if total == 0:
        return {"tcr": 0, "partial_rate": 0, "failure_rate": 0, "escalation_rate": 0}
    
    successes = sum(1 for o in outcomes if o == TaskOutcome.SUCCESS)
    partials = sum(1 for o in outcomes if o == TaskOutcome.PARTIAL)
    failures = sum(1 for o in outcomes if o in (TaskOutcome.FAILURE, TaskOutcome.INCORRECT))
    escalations = sum(1 for o in outcomes if o == TaskOutcome.ESCALATED)
    
    return {
        "tcr": successes / total,
        "partial_rate": partials / total,
        "failure_rate": failures / total,
        "escalation_rate": escalations / total,
        "total_evaluated": total,
        "weighted_score": (successes + 0.5 * partials) / total,
    }
```

#### Step Efficiency

Measures how efficiently the agent achieves its goal — excessive steps indicate planning issues.

```python
"""Step efficiency metrics for agent evaluation."""

import numpy as np
from dataclasses import dataclass, field
from typing import List

@dataclass
class StepEfficiencyMetrics:
    steps_per_task: List[int] = field(default_factory=list)
    tool_calls_per_task: List[int] = field(default_factory=list)
    llm_calls_per_task: List[int] = field(default_factory=list)
    
    def add_run(self, steps: int, tool_calls: int, llm_calls: int):
        self.steps_per_task.append(steps)
        self.tool_calls_per_task.append(tool_calls)
        self.llm_calls_per_task.append(llm_calls)
    
    def summary(self) -> dict:
        return {
            "avg_steps": np.mean(self.steps_per_task) if self.steps_per_task else 0,
            "median_steps": np.median(self.steps_per_task) if self.steps_per_task else 0,
            "p95_steps": np.percentile(self.steps_per_task, 95) if self.steps_per_task else 0,
            "max_steps": max(self.steps_per_task) if self.steps_per_task else 0,
            "avg_tool_calls": np.mean(self.tool_calls_per_task) if self.tool_calls_per_task else 0,
            "avg_llm_calls": np.mean(self.llm_calls_per_task) if self.llm_calls_per_task else 0,
            "tool_call_ratio": (sum(self.tool_calls_per_task) / max(sum(self.llm_calls_per_task), 1)),
        }
    
    def efficiency_score(self, optimal_steps: int = 3) -> float:
        """Score efficiency relative to optimal steps (1.0 = optimal)."""
        if not self.steps_per_task:
            return 0.0
        avg_steps = np.mean(self.steps_per_task)
        if avg_steps <= optimal_steps:
            return 1.0
        # Penalty curve: at 2x optimal = 0.5, at 4x optimal = 0.25
        ratio = optimal_steps / max(avg_steps, 1)
        return min(1.0, ratio * 2 - ratio**2)
```

### 2.2 Output Quality Metrics

#### Answer Correctness

Evaluating whether the agent's final answer is factually correct requires either:
- Ground truth comparison (for known-answer tasks)
- LLM-as-judge evaluation (for open-ended tasks)
- Human evaluation (for subjective quality)

```python
"""Answer correctness evaluation using LLM-as-judge."""

from openai import OpenAI
from typing import Optional
import json

class LLMJudge:
    """LLM-as-judge evaluation for agent outputs."""
    
    def __init__(self, model: str = "gpt-4o", judge_prompt: Optional[str] = None):
        self.client = OpenAI()
        self.model = model
        self.judge_prompt = judge_prompt or self._default_prompt()
    
    def _default_prompt(self) -> str:
        return """You are evaluating an AI agent's response to a user request. 
Evaluate the following dimensions on a scale of 1-5:

1. CORRECTNESS: Is the information factually accurate?
2. COMPLETENESS: Does the response fully address the user's request?
3. RELEVANCE: Is all provided information relevant to the request?
4. CLARITY: Is the response clear and well-structured?
5. SAFETY: Does the response avoid harmful, biased, or inappropriate content?

For each dimension, provide:
- A score (1-5)
- A brief justification

Also provide an overall assessment (PASS/FAIL) and identify any errors or hallucinations.

User Request: {task_input}

Agent Response: {agent_output}

Ground Truth (if available): {ground_truth}

Format your response as JSON:
{{
    "correctness": {{"score": int, "justification": "..."}},
    "completeness": {{"score": int, "justification": "..."}},
    "relevance": {{"score": int, "justification": "..."}},
    "clarity": {{"score": int, "justification": "..."}},
    "safety": {{"score": int, "justification": "..."}},
    "overall": "PASS" or "FAIL",
    "hallucinations": ["list any factual errors"],
    "improvements": ["suggested improvements"],
    "overall_score": float
}}"""
    
    def evaluate(self, task_input: str, agent_output: str, ground_truth: Optional[str] = None) -> dict:
        """Evaluate an agent's output using LLM-as-judge."""
        prompt = self.judge_prompt.format(
            task_input=task_input,
            agent_output=agent_output,
            ground_truth=ground_truth or "Not provided"
        )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Add metadata
        result["judge_model"] = self.model
        result["judge_cost"] = self._calculate_cost(response.usage)
        
        return result
    
    def _calculate_cost(self, usage) -> float:
        """Calculate cost of the judge evaluation."""
        input_cost = (usage.prompt_tokens / 1000) * 0.0025
        output_cost = (usage.completion_tokens / 1000) * 0.01
        return input_cost + output_cost
    
    def evaluate_batch(self, eval_cases: list) -> list:
        """Evaluate a batch of agent outputs."""
        results = []
        for case in eval_cases:
            result = self.evaluate(
                task_input=case["task_input"],
                agent_output=case["agent_output"],
                ground_truth=case.get("ground_truth"),
            )
            result["case_id"] = case.get("id")
            results.append(result)
        return results
```

### 2.3 Hallucination Detection

Hallucinations in agent outputs can be particularly dangerous because the agent may confidently present incorrect information as if it were verified.

```python
"""Hallucination detection for agent outputs."""

from typing import List, Optional
import re

class HallucinationDetector:
    """
    Detects potential hallucinations in agent outputs.
    Uses multiple techniques: factual consistency, citation verification,
    internal consistency, and common hallucination patterns.
    """
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def detect_hallucinations(self, agent_output: str, context: Optional[str] = None) -> dict:
        """
        Analyze agent output for potential hallucinations.
        
        Returns:
            dict with hallucination_score, flagged_statements, and confidence
        """
        # Technique 1: Check for numeric/statistical claims
        numeric_claims = self._extract_numeric_claims(agent_output)
        
        # Technique 2: Check for citation/quote fabrications
        citation_issues = self._check_citations(agent_output)
        
        # Technique 3: LLM-based consistency check
        consistency_issues = self._check_consistency(agent_output, context)
        
        # Technique 4: Check for hedging vs. overconfidence
        confidence_issues = self._check_confidence_markers(agent_output)
        
        all_issues = numeric_claims + citation_issues + consistency_issues + confidence_issues
        
        return {
            "hallucination_detected": len(all_issues) > 0,
            "hallucination_count": len(all_issues),
            "hallucination_score": min(1.0, len(all_issues) * 0.25),
            "issues": all_issues,
            "flagged_statements": [issue["statement"] for issue in all_issues],
        }
    
    def _extract_numeric_claims(self, text: str) -> List[dict]:
        """Flag specific numeric claims that may be hallucinated."""
        issues = []
        # Match "X%", "X million", "X billion", dates, etc.
        patterns = [
            (r'\b\d+\.?\d*%\b', 'percentage'),
            (r'\b\d+\s*(million|billion|trillion)\b', 'large_number'),
            (r'\bin\s+\d{4}\b', 'year'),
            (r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b', 'date'),
        ]
        
        for pattern, claim_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                issues.append({
                    "type": "numeric_claim",
                    "subtype": claim_type,
                    "statement": match.group(),
                    "position": match.start(),
                    "confidence": 0.3,  # Low confidence — flag for review
                })
        
        return issues
    
    def _check_citations(self, text: str) -> List[dict]:
        """Check citations for fabrication patterns."""
        issues = []
        
        # Pattern: "According to [source]," with source that might be fabricated
        citation_patterns = [
            r'according to (a|an|the) (study|report|paper|article|research)',
            r'as (reported|published|stated) in',
            r'\([A-Za-z]+,?\s+\d{4}\)',  # (Author, 2023) style citations
        ]
        
        for pattern in citation_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                issues.append({
                    "type": "uncited_reference",
                    "subtype": "citation_pattern",
                    "statement": text[max(0, match.start()-50):match.end()+50],
                    "confidence": 0.5,
                })
        
        return issues
    
    def _check_consistency(self, output: str, context: Optional[str] = None) -> List[dict]:
        """Use LLM to check internal and external consistency."""
        prompt = f"""Analyze the following agent response for factual inconsistencies, contradictions, or unsupported claims.

Response: {output}

{f'Context: {context}' if context else ''}

List any statements that appear to be:
1. Internally contradictory
2. Factually questionable (not verifiable from general knowledge)
3. Making claims that would require specific knowledge the agent cannot have

Return ONLY a JSON array of objects with "statement" and "concern" fields.
If no issues found, return an empty array []."""

        # In production, call LLM here
        # For this example, we return empty
        return []
    
    def _check_confidence_markers(self, text: str) -> List[dict]:
        """Check for overconfidence about unverifiable claims."""
        issues = []
        
        # Phrases indicating certainty about potentially false claims
        overconfident = [
            r'\bdefinitely\b', r'\bwithout (a )?doubt\b', r'\bcertainly\b',
            r'\babsolutely\b', r'\bundoubtedly\b', r'\bI\'m (sure|certain)\b',
        ]
        
        uncertain = [
            r'\bmight\b', r'\bmay\b', r'\bcould\b', r'\bpossibly\b',
            r'\bI think\b', r'\bto the best of my knowledge\b',
        ]
        
        # If response is overconfident about niche topics, flag it
        has_overconfident = any(re.search(p, text, re.IGNORECASE) for p in overconfident)
        has_uncertain = any(re.search(p, text, re.IGNORECASE) for p in uncertain)
        
        if has_overconfident and not has_uncertain:
            issues.append({
                "type": "overconfidence",
                "subtype": "no_hedging",
                "statement": "Response contains certainty markers without hedging",
                "confidence": 0.2,
            })
        
        return issues
```

### 2.4 Tool Usage Evaluation

Evaluating whether the agent chose the right tools and used them correctly:

```python
"""Tool usage evaluation metrics."""

from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class ToolCallRecord:
    tool_name: str
    arguments: Dict[str, Any]
    result: str
    success: bool
    duration_ms: float
    step_number: int

@dataclass
class ToolUsageEvaluation:
    total_calls: int
    unique_tools_used: int
    tools_available: int
    tool_coverage: float  # unique_tools_used / tools_available
    success_rate: float
    avg_duration_ms: float
    unnecessary_calls: int
    incorrect_tool_selections: int

def evaluate_tool_usage(
    tool_calls: List[ToolCallRecord],
    available_tools: List[str],
    expected_tools_for_task: List[str]
) -> ToolUsageEvaluation:
    """Evaluate how well the agent used available tools."""
    
    total = len(tool_calls)
    unique_tools = set(t.tool_name for t in tool_calls)
    successes = sum(1 for t in tool_calls if t.success)
    durations = [t.duration_ms for t in tool_calls]
    
    return ToolUsageEvaluation(
        total_calls=total,
        unique_tools_used=len(unique_tools),
        tools_available=len(available_tools),
        tool_coverage=len(unique_tools) / max(len(available_tools), 1),
        success_rate=successes / max(total, 1),
        avg_duration_ms=sum(durations) / max(len(durations), 1),
        unnecessary_calls=0,  # Requires task-specific logic
        incorrect_tool_selections=0,  # Requires task-specific logic
    )
```

## 3. Evaluation Datasets

### 3.1 Public Agent Benchmarks

Several public benchmarks exist for evaluating general-purpose agents:

| Benchmark | Domain | Tasks | Evaluation | Notes |
|-----------|--------|-------|------------|-------|
| **AgentBench** | General (OS, web, database, etc.) | 466 tasks | Task completion | Multi-domain, good for general agents |
| **WebArena** | Web browsing | 812 tasks | Task completion | Realistic web environment |
| **SWE-bench** | Software engineering | 2,294 issues | Test pass rate | Code generation + debugging |
| **GAIA** | General AI assistants | 466 questions | Correctness | Multi-step reasoning |
| **ToolBench** | Tool use | 3,456 tasks | Task completion, tool selection | Tool-augmented agents |
| **ALFWorld** | Embodied agents | 6 environments | Task completion | Household tasks |
| **AgentInstruct** | General | 1,866 tasks | LLM-as-judge | Diverse tasks with rubrics |

### 3.2 Creating Custom Evaluation Datasets

For production agents, use a combination of:
1. **Synthetic tasks**: Generated by LLM from task templates
2. **Production traces**: Sampled and labeled from real usage
3. **Edge cases**: Hand-crafted failure cases discovered in production

```python
"""Custom evaluation dataset creation and management."""

import json
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict

@dataclass
class EvalCase:
    """A single evaluation case for an agent."""
    id: str
    task_input: str
    ground_truth: Optional[str] = None
    expected_tools: Optional[List[str]] = None
    expected_steps: Optional[int] = None
    tags: List[str] = None
    difficulty: str = "medium"  # easy, medium, hard, expert
    category: str = "general"
    rubrics: Optional[Dict[str, str]] = None  # criterion -> description
    created_at: str = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        if self.tags is None:
            self.tags = []

@dataclass
class EvalDataset:
    """A versioned dataset of evaluation cases."""
    name: str
    version: str
    cases: List[EvalCase]
    description: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def add_case(self, case: EvalCase):
        self.cases.append(case)
    
    def filter_by_tags(self, tags: List[str]) -> 'EvalDataset':
        """Create a subset of cases matching any of the given tags."""
        filtered = [c for c in self.cases if c.tags and any(t in c.tags for t in tags)]
        return EvalDataset(
            name=f"{self.name}_filtered",
            version=self.version,
            cases=filtered,
            description=f"Filtered subset of {self.name} by tags: {tags}",
        )
    
    def filter_by_difficulty(self, difficulty: str) -> 'EvalDataset':
        filtered = [c for c in self.cases if c.difficulty == difficulty]
        return EvalDataset(
            name=f"{self.name}_{difficulty}",
            version=self.version,
            cases=filtered,
        )
    
    def save(self, path: str):
        """Save dataset to JSON."""
        data = {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "metadata": self.metadata,
            "cases": [asdict(c) for c in self.cases],
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'EvalDataset':
        """Load dataset from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            name=data["name"],
            version=data["version"],
            description=data.get("description", ""),
            metadata=data.get("metadata", {}),
            cases=[EvalCase(**c) for c in data["cases"]],
        )


# Example: Creating a dataset from production traces
def create_dataset_from_traces(traces: List[dict], name: str, version: str = "1.0.0") -> EvalDataset:
    """Create an evaluation dataset from production agent traces."""
    cases = []
    for trace in traces:
        if trace.get("status") == "success":
            continue  # Only use problematic traces for eval
        
        case = EvalCase(
            task_input=trace.get("input", ""),
            ground_truth=trace.get("human_corrected_output"),
            expected_tools=trace.get("tools_used"),
            expected_steps=trace.get("steps"),
            tags=["production", trace.get("error_type", "unknown")],
            difficulty="hard" if trace.get("error_type") else "medium",
            category=trace.get("category", "general"),
        )
        cases.append(case)
    
    return EvalDataset(name=name, version=version, cases=cases)
```

## 4. Automated Evaluation Pipeline

### 4.1 Complete Evaluation Harness

```python
"""Complete agent evaluation harness for CI/CD integration."""

import json
import time
import asyncio
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

@dataclass
class EvalResult:
    """Result of evaluating an agent on a single case."""
    case_id: str
    task_input: str
    agent_output: str
    ground_truth: Optional[str]
    duration_ms: float
    steps: int
    llm_calls: int
    tool_calls: int
    total_tokens: int
    cost: float
    error: Optional[str] = None
    scores: Dict[str, float] = None  # evaluator_name -> score
    metadata: Dict[str, Any] = None

@dataclass
class EvalRunSummary:
    """Summary of an evaluation run across all cases."""
    dataset_name: str
    agent_id: str
    agent_version: str
    timestamp: str
    total_cases: int
    passed: int
    failed: int
    errored: int
    avg_duration_ms: float
    avg_steps: float
    avg_cost: float
    total_cost: float
    aggregated_scores: Dict[str, float]
    detailed_results: List[EvalResult]

class AgentEvalHarness:
    """
    Automated evaluation harness for agents.
    Runs an agent against a dataset of test cases with multiple evaluators.
    """
    
    def __init__(
        self,
        agent_fn: Callable[[str], Dict[str, Any]],
        agent_id: str,
        agent_version: str,
        evaluators: Optional[List[Callable]] = None,
        max_concurrency: int = 10,
        timeout_seconds: int = 300,
    ):
        self.agent_fn = agent_fn
        self.agent_id = agent_id
        self.agent_version = agent_version
        self.evaluators = evaluators or []
        self.max_concurrency = max_concurrency
        self.timeout_seconds = timeout_seconds
    
    def add_evaluator(self, evaluator_fn: Callable[[EvalResult], Dict[str, float]], name: str):
        """Add an evaluation function."""
        self.evaluators.append((evaluator_fn, name))
    
    def run(self, dataset: EvalDataset, verbose: bool = False) -> EvalRunSummary:
        """Run the evaluation harness against a dataset."""
        
        results = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            futures = {
                executor.submit(self._evaluate_single, case): case
                for case in dataset.cases
            }
            
            for future in as_completed(futures):
                case = futures[future]
                try:
                    result = future.result(timeout=self.timeout_seconds)
                    results.append(result)
                    if verbose:
                        status = "✓" if result.error is None else "✗"
                        print(f"{status} {case.id[:8]}... ({result.duration_ms:.0f}ms, {result.cost:.4f}$)")
                except Exception as e:
                    results.append(EvalResult(
                        case_id=case.id,
                        task_input=case.task_input,
                        agent_output="",
                        ground_truth=case.ground_truth,
                        duration_ms=0,
                        steps=0,
                        llm_calls=0,
                        tool_calls=0,
                        total_tokens=0,
                        cost=0,
                        error=f"Harness error: {str(e)}",
                    ))
                    if verbose:
                        print(f"✗ {case.id[:8]}... (harness error: {str(e)})")
        
        # Aggregate results
        passed = sum(1 for r in results if r.error is None)
        failed = sum(1 for r in results if r.error is not None)
        errored = sum(1 for r in results if r.error and "Harness error" in r.error)
        
        # Aggregate scores across evaluators
        aggregated_scores = {}
        if self.evaluators:
            for _, eval_name in self.evaluators:
                scores = [
                    r.scores.get(eval_name, 0)
                    for r in results
                    if r.scores and eval_name in r.scores
                ]
                if scores:
                    aggregated_scores[eval_name] = sum(scores) / len(scores)
        
        total_elapsed = time.time() - start_time
        
        return EvalRunSummary(
            dataset_name=dataset.name,
            agent_id=self.agent_id,
            agent_version=self.agent_version,
            timestamp=datetime.utcnow().isoformat(),
            total_cases=len(dataset.cases),
            passed=passed,
            failed=failed,
            errored=errored,
            avg_duration_ms=sum(r.duration_ms for r in results) / max(len(results), 1),
            avg_steps=sum(r.steps for r in results) / max(len(results), 1),
            avg_cost=sum(r.cost for r in results) / max(len(results), 1),
            total_cost=sum(r.cost for r in results),
            aggregated_scores=aggregated_scores,
            detailed_results=results,
        )
    
    def _evaluate_single(self, case: EvalCase) -> EvalResult:
        """Run the agent on a single test case and evaluate."""
        try:
            start = time.time()
            result = self.agent_fn(case.task_input)
            duration = (time.time() - start) * 1000
            
            eval_result = EvalResult(
                case_id=case.id,
                task_input=case.task_input,
                agent_output=result.get("output", ""),
                ground_truth=case.ground_truth,
                duration_ms=duration,
                steps=result.get("steps", 0),
                llm_calls=result.get("llm_calls", 0),
                tool_calls=result.get("tool_calls", 0),
                total_tokens=result.get("total_tokens", 0),
                cost=result.get("cost", 0),
                metadata=result.get("metadata", {}),
            )
            
            # Run evaluators
            scores = {}
            for evaluator_fn, eval_name in self.evaluators:
                try:
                    scores[eval_name] = evaluator_fn(eval_result)
                except Exception as e:
                    scores[eval_name] = 0.0
            eval_result.scores = scores
            
            return eval_result
            
        except Exception as e:
            return EvalResult(
                case_id=case.id,
                task_input=case.task_input,
                agent_output="",
                ground_truth=case.ground_truth,
                duration_ms=0,
                steps=0,
                llm_calls=0,
                tool_calls=0,
                total_tokens=0,
                cost=0,
                error=str(e),
            )
    
    def report(self, summary: EvalRunSummary, format: str = "text") -> str:
        """Generate a formatted evaluation report."""
        if format == "json":
            return json.dumps(asdict(summary), indent=2, default=str)
        
        lines = [
            f"=== Agent Evaluation Report ===",
            f"Agent: {summary.agent_id} (v{summary.agent_version})",
            f"Dataset: {summary.dataset_name}",
            f"Date: {summary.timestamp}",
            f"",
            f"=== Summary ===",
            f"Total Cases: {summary.total_cases}",
            f"Passed: {summary.passed}",
            f"Failed: {summary.failed}",
            f"Errored: {summary.errored}",
            f"Pass Rate: {summary.passed / max(summary.total_cases, 1) * 100:.1f}%",
            f"",
            f"=== Performance ===",
            f"Avg Duration: {summary.avg_duration_ms:.0f}ms",
            f"Avg Steps: {summary.avg_steps:.1f}",
            f"Total Cost: ${summary.total_cost:.4f}",
            f"Avg Cost: ${summary.avg_cost:.6f}",
            f"",
            f"=== Scores ===",
        ]
        
        for name, score in summary.aggregated_scores.items():
            lines.append(f"{name}: {score:.3f}")
        
        return "\n".join(lines)
    
    def save_results(self, summary: EvalRunSummary, path: str):
        """Save evaluation results to JSON."""
        with open(path, 'w') as f:
            json.dump(asdict(summary), f, indent=2, default=str)


# Usage example
def test_agent(task_input: str) -> dict:
    """Wrapper for the agent being evaluated."""
    # In production, this would call your actual agent
    return {
        "output": f"Response to: {task_input}",
        "steps": 3,
        "llm_calls": 4,
        "tool_calls": 2,
        "total_tokens": 1500,
        "cost": 0.015,
    }

def correctness_evaluator(result: EvalResult) -> float:
    """Simple evaluator: 1.0 if output exists and no error, 0.0 otherwise."""
    if result.error:
        return 0.0
    # In production, use LLM-as-judge here
    return 1.0 if len(result.agent_output) > 10 else 0.0

def efficiency_evaluator(result: EvalResult) -> float:
    """Evaluate step efficiency."""
    if result.steps <= 3:
        return 1.0
    elif result.steps <= 7:
        return 0.7
    elif result.steps <= 15:
        return 0.4
    else:
        return 0.1

def cost_evaluator(result: EvalResult) -> float:
    """Evaluate cost efficiency."""
    if result.cost <= 0.01:
        return 1.0
    elif result.cost <= 0.05:
        return 0.7
    elif result.cost <= 0.10:
        return 0.4
    else:
        return 0.1
```

### 4.2 CI/CD Integration

```python
"""CI/CD integration for agent evaluation."""

import os
import sys
import json
from pathlib import Path

def run_eval_in_ci():
    """
    Evaluation script for CI/CD pipelines.
    Designed to be run in GitHub Actions, GitLab CI, etc.
    """
    # Configuration from environment
    dataset_path = os.environ.get("EVAL_DATASET", "eval_datasets/regression.json")
    agent_version = os.environ.get("AGENT_VERSION", "dev")
    min_pass_rate = float(os.environ.get("MIN_PASS_RATE", "0.8"))
    report_path = os.environ.get("EVAL_REPORT_PATH", "eval_report.json")
    
    # Load dataset
    dataset = EvalDataset.load(dataset_path)
    
    # Initialize harness
    harness = AgentEvalHarness(
        agent_fn=test_agent,
        agent_id="my-agent",
        agent_version=agent_version,
        evaluators=[
            (correctness_evaluator, "correctness"),
            (efficiency_evaluator, "efficiency"),
            (cost_evaluator, "cost_efficiency"),
        ],
        max_concurrency=5,
    )
    
    # Run evaluation
    print(f"Running evaluation on {len(dataset.cases)} cases...")
    summary = harness.run(dataset, verbose=True)
    
    # Save report
    harness.save_results(summary, report_path)
    
    # Print summary
    print(harness.report(summary))
    
    # Check pass/fail for CI
    pass_rate = summary.passed / max(summary.total_cases, 1)
    if pass_rate < min_pass_rate:
        print(f"FAIL: Pass rate {pass_rate:.1%} below threshold {min_pass_rate:.1%}")
        sys.exit(1)
    else:
        print(f"PASS: Pass rate {pass_rate:.1%} meets threshold {min_pass_rate:.1%}")
        sys.exit(0)


# GitHub Actions workflow snippet (in .github/workflows/agent-eval.yml):
"""
name: Agent Evaluation
on: [push, pull_request]
jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run agent evaluation
        run: python eval_harness.py
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          EVAL_DATASET: eval_datasets/regression.json
          AGENT_VERSION: ${{ github.sha }}
          MIN_PASS_RATE: '0.85'
      - name: Upload evaluation report
        uses: actions/upload-artifact@v4
        with:
          name: eval-report
          path: eval_report.json
      - name: Comment PR with results
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const report = JSON.parse(fs.readFileSync('eval_report.json', 'utf8'));
            // Post comment with evaluation results
"""
```

## 5. Regression Testing for Agents

### 5.1 Building a Regression Test Suite

```python
"""Regression testing framework for agents."""

from typing import List, Dict, Any
from datetime import datetime, timedelta
import json

class AgentRegressionSuite:
    """
    Manages regression tests for agents.
    Tracks historical performance and detects regressions.
    """
    
    def __init__(self, history_path: str = "eval_history.json"):
        self.history_path = history_path
        self.history = self._load_history()
    
    def _load_history(self) -> List[dict]:
        try:
            with open(self.history_path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def _save_history(self):
        with open(self.history_path, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)
    
    def record_run(self, summary: EvalRunSummary):
        """Record an evaluation run in the history."""
        record = {
            "timestamp": summary.timestamp,
            "agent_version": summary.agent_version,
            "pass_rate": summary.passed / max(summary.total_cases, 1),
            "avg_duration_ms": summary.avg_duration_ms,
            "avg_steps": summary.avg_steps,
            "avg_cost": summary.avg_cost,
            "total_cost": summary.total_cost,
            "scores": summary.aggregated_scores,
            "dataset": summary.dataset_name,
        }
        self.history.append(record)
        self._save_history()
    
    def detect_regression(self, metric: str = "pass_rate", window: int = 5, threshold: float = 0.05) -> List[dict]:
        """
        Detect regressions in the specified metric.
        
        Args:
            metric: Metric to check
            window: Number of recent runs to compare
            threshold: Minimum relative change to flag as regression
        
        Returns:
            List of regression events
        """
        if len(self.history) < 2:
            return []
        
        recent = self.history[-window:]
        baseline = self.history[:-window]
        
        if not baseline:
            return []
        
        current_value = recent[-1].get(metric, 0)
        baseline_avg = sum(r.get(metric, 0) for r in baseline) / len(baseline)
        
        rel_change = (current_value - baseline_avg) / max(abs(baseline_avg), 0.001)
        
        if rel_change < -threshold:  # Negative change = regression
            return [{
                "type": "regression",
                "metric": metric,
                "current": current_value,
                "baseline": baseline_avg,
                "rel_change": rel_change,
                "timestamp": recent[-1]["timestamp"],
                "version": recent[-1]["agent_version"],
            }]
        
        return []
    
    def generate_regression_report(self) -> str:
        """Generate a human-readable regression report."""
        if len(self.history) < 2:
            return "Not enough history for regression analysis (need at least 2 runs)."
        
        last = self.history[-1]
        prev = self.history[-2]
        
        lines = [
            f"=== Regression Report ===",
            f"",
            f"Current version: {last['agent_version']} ({last['timestamp']})",
            f"Previous version: {prev['agent_version']} ({prev['timestamp']})",
            f"",
            f"Metric          Current   Previous  Change",
            f"------          -------   --------  ------",
        ]
        
        metrics = ["pass_rate", "avg_duration_ms", "avg_steps", "avg_cost"]
        for metric in metrics:
            curr = last.get(metric, 0)
            prev_val = prev.get(metric, 0)
            change = curr - prev_val
            pct = change / max(abs(prev_val), 0.001) * 100 if prev_val else 0
            arrow = "↑" if pct > 0 else "↓" if pct < 0 else "→"
            lines.append(f"{metric:15} {curr:>8.4f} {prev_val:>8.4f} {arrow} {pct:>+6.1f}%")
        
        # Check for regressions
        for metric in metrics:
            regressions = self.detect_regression(metric)
            if regressions:
                lines.append(f"\n⚠ REGRESSION DETECTED in {metric}!")
                for r in regressions:
                    lines.append(f"  Current: {r['current']:.4f} vs Baseline: {r['baseline']:.4f}")
        
        return "\n".join(lines)
```

## 6. A/B Testing for Agents

### 6.1 A/B Testing Framework

```python
"""A/B testing framework for agent versions."""

import random
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass

@dataclass
class ABTestConfig:
    """Configuration for an A/B test of agent versions."""
    experiment_name: str
    control_agent: str  # agent_id:version
    treatment_agent: str
    traffic_split: float  # 0.0 to 1.0, fraction going to treatment
    metrics: list[str]  # metrics to track
    min_sample_size: int
    duration_hours: int
    start_time: Optional[str] = None

class AgentABTesting:
    """
    A/B testing framework for comparing agent versions in production.
    """
    
    def __init__(self, config: ABTestConfig, agent_registry: Dict[str, Callable]):
        self.config = config
        self.agent_registry = agent_registry
        self.results = {"control": [], "treatment": []}
    
    def select_agent(self, user_id: str) -> tuple[str, str]:
        """Select which agent version a user gets based on consistent hashing."""
        # Consistent assignment so same user always gets same version
        hash_val = hash(f"{user_id}_{self.config.experiment_name}") % 1000
        if hash_val / 1000 < self.config.traffic_split:
            return ("treatment", self.config.treatment_agent)
        return ("control", self.config.control_agent)
    
    def record_result(self, group: str, agent_id: str, metrics: Dict[str, float]):
        """Record a result for analysis."""
        self.results[group].append({
            "timestamp": datetime.utcnow().isoformat(),
            "agent_id": agent_id,
            **metrics,
        })
    
    def analyze(self) -> Dict[str, Any]:
        """Analyze A/B test results."""
        import numpy as np
        from scipy import stats
        
        analysis = {
            "experiment": self.config.experiment_name,
            "control_group": self.config.control_agent,
            "treatment_group": self.config.treatment_agent,
            "control_sample": len(self.results["control"]),
            "treatment_sample": len(self.results["treatment"]),
            "metrics": {},
        }
        
        for metric in self.config.metrics:
            control_vals = [r.get(metric, 0) for r in self.results["control"]]
            treatment_vals = [r.get(metric, 0) for r in self.results["treatment"]]
            
            if control_vals and treatment_vals:
                control_mean = np.mean(control_vals)
                treatment_mean = np.mean(treatment_vals)
                
                # Statistical significance (Mann-Whitney U test for non-normal distributions)
                try:
                    stat, p_value = stats.mannwhitneyu(control_vals, treatment_vals, alternative='two-sided')
                except:
                    stat, p_value = 0, 1.0
                
                effect_size = (treatment_mean - control_mean) / max(control_mean, 0.001)
                
                analysis["metrics"][metric] = {
                    "control_mean": control_mean,
                    "treatment_mean": treatment_mean,
                    "effect_size": effect_size,
                    "effect_pct": effect_size * 100,
                    "p_value": p_value,
                    "significant": p_value < 0.05,
                    "winner": "treatment" if effect_size > 0 and p_value < 0.05 else
                             "control" if effect_size < 0 and p_value < 0.05 else
                             "none (not significant)",
                }
        
        return analysis
```

## 7. Production Evaluation Pipeline

### 7.1 Online Evaluation

While offline evaluation (on datasets) is essential, production agents need continuous online evaluation:

```python
"""Online evaluation pipeline for production agents."""

import logging
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class OnlineAgentEvaluator:
    """Continuous evaluation of agents in production."""
    
    def __init__(self, eval_storage, llm_judge: Optional[LLMJudge] = None):
        self.eval_storage = eval_storage
        self.llm_judge = llm_judge
    
    def evaluate_production_trace(self, trace: dict):
        """Evaluate a single production trace in real-time."""
        
        evaluation = {
            "trace_id": trace.get("trace_id"),
            "agent_id": trace.get("agent_id"),
            "agent_version": trace.get("agent_version"),
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {},
            "alerts": [],
        }
        
        # Basic metrics
        evaluation["metrics"]["duration_ms"] = trace.get("duration_ms", 0)
        evaluation["metrics"]["steps"] = trace.get("steps", 0)
        evaluation["metrics"]["cost"] = trace.get("cost", 0)
        evaluation["metrics"]["tool_calls"] = len(trace.get("tool_calls", []))
        
        # Success/failure
        is_error = trace.get("status") == "error"
        evaluation["metrics"]["success"] = not is_error
        
        if is_error:
            evaluation["alerts"].append({
                "type": "agent_error",
                "severity": "error",
                "message": f"Agent failed: {trace.get('error', 'unknown error')}",
            })
        
        # Check for excessive steps (potential loop)
        if trace.get("steps", 0) > 20:
            evaluation["alerts"].append({
                "type": "excessive_steps",
                "severity": "warning",
                "message": f"Agent took {trace['steps']} steps (threshold: 20)",
            })
        
        # Check for high cost
        if trace.get("cost", 0) > 0.50:
            evaluation["alerts"].append({
                "type": "high_cost",
                "severity": "warning",
                "message": f"Agent cost ${trace['cost']:.4f} exceeds $0.50 threshold",
            })
        
        # LLM-based quality evaluation (sampled, due to cost)
        if self.llm_judge and random.random() < 0.1:  # 10% sample
            try:
                quality = self.llm_judge.evaluate(
                    task_input=trace.get("input", ""),
                    agent_output=trace.get("output", ""),
                )
                evaluation["metrics"]["quality_score"] = quality.get("overall_score", 0)
                
                if quality.get("overall") == "FAIL":
                    evaluation["alerts"].append({
                        "type": "quality_failure",
                        "severity": "warning",
                        "message": f"Quality evaluation failed: {quality.get('improvements', [])}",
                    })
            except Exception as e:
                logger.warning(f"LLM judge failed: {e}")
        
        # Store evaluation
        self.eval_storage.store(evaluation)
        
        # Return alerts for real-time action
        return evaluation
```

## 8. Best Practices

### 8.1 Evaluation Strategy

1. **Start with task completion rate** — the single most important metric
2. **Add efficiency metrics** — cost, steps, duration as secondary indicators
3. **Layer in quality metrics** — LLM-as-judge for output quality
4. **Include safety checks** — hallucination detection, toxicity, PII
5. **Build regression test suites** — capture every production failure as a test case
6. **Automate evaluation in CI/CD** — gate deployments on eval pass rates

### 8.2 LLM-as-Judge Guidelines

- Use a stronger model as judge (e.g., GPT-4o evaluating GPT-4o-mini)
- Provide clear rubrics with examples
- Include ground truth when available
- Use chain-of-thought prompting for better evaluations
- Calibrate judges against human evaluations periodically
- Be aware of position bias (swap output order in comparisons)

### 8.3 Dataset Management

- Version every dataset (semantic versioning)
- Track dataset drift over time (are test cases still relevant?)
- Balance datasets across difficulty levels, categories, and edge cases
- Never train on evaluation data
- Review and refresh datasets quarterly
- Include synthetic, production, and hand-crafted cases

### 8.4 Alert Thresholds for Production Evaluation

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| Task Completion Rate | < 85% | < 70% | Rollback agent version |
| Average Steps | > 15 | > 25 | Investigate loop patterns |
| Average Cost per Task | > $0.10 | > $0.50 | Alert engineering team |
| Error Rate | > 5% | > 15% | Auto-rollback |
| Hallucination Rate | > 3% | > 10% | Flag for review |
| P95 Latency | > 60s | > 120s | Scale infrastructure |

## 9. Conclusion

Agent evaluation is not a one-time activity — it's a continuous practice that evolves with your agent system. The most successful teams:

1. **Evaluate early and often** — running eval suites on every commit
2. **Layer multiple evaluation methods** — from automated metrics to LLM judges to human review
3. **Capture production failures as test cases** — every bug is an opportunity to improve
4. **Monitor for regression** — not just pass/fail but trends in efficiency, cost, and quality
5. **A/B test in production** — validate improvements with real users

The evaluation harness, datasets, and CI/CD integration provided in this document give you a production-ready foundation. Adapt them to your specific agent architecture, domain, and quality requirements.

---

*Next: [05-Agent-Cost-Tracking-and-Optimization.md](05-Agent-Cost-Tracking-and-Optimization.md) — Cost tracking strategies and optimization techniques.*
