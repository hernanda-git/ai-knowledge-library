# Technical Deep-Dive: AI Evaluation & Benchmarking at Scale

> Advanced technical topics: custom evaluation infrastructure, evaluation proxy architectures, model routing evaluation, A/B testing for LLMs, and large-scale evaluation systems.

**Last Updated:** 2026-07-06
**Estimated Reading Time:** 80 minutes
**Line Count:** ~350+
**Category:** 58-AI-Evaluation-and-Benchmarking-at-Scale

---

## Table of Contents

1. [Evaluation Proxy Architecture](#1-evaluation-proxy-architecture)
2. [A/B Testing for LLM Applications](#2-ab-testing-for-llm-applications)
3. [Model Routing Evaluation](#3-model-routing-evaluation)
4. [Large-Scale Evaluation Systems](#4-large-scale-evaluation-systems)
5. [Evaluation Infrastructure Patterns](#5-evaluation-infrastructure-patterns)
6. [Evaluation Data Management](#6-evaluation-data-management)
7. [Evaluation for Fine-Tuned Models](#7-evaluation-for-fine-tuned-models)
8. [Evaluation for Multi-Modal Systems](#8-evaluation-for-multi-modal-systems)
9. [Evaluation Debugging and Root Cause Analysis](#9-evaluation-debugging-and-root-cause-analysis)
10. [Evaluation in Regulated Industries](#10-evaluation-in-regulated-industries)

---

## 1. Evaluation Proxy Architecture

### The Proxy Pattern

An evaluation proxy sits between your application and LLM providers, capturing all requests/responses for evaluation purposes.

```
┌─────────────────────────────────────────────────────────────┐
│                 EVALUATION PROXY ARCHITECTURE                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐     ┌─────────────────┐     ┌──────────┐    │
│  │  Your    │────▶│  Evaluation     │────▶│  LLM     │    │
│  │  App     │◀────│  Proxy          │◀────│  Provider│    │
│  └──────────┘     └────────┬────────┘     └──────────┘    │
│                            │                                │
│                    ┌───────┴───────┐                       │
│                    │               │                        │
│              ┌─────▼─────┐  ┌─────▼──────┐                │
│              │  Evaluation│  │  Logging   │                │
│              │  Pipeline  │  │  & Tracing │                │
│              └─────┬─────┘  └────────────┘                │
│                    │                                       │
│              ┌─────▼─────────────┐                         │
│              │  Evaluation Store │                         │
│              │  (Results DB)     │                         │
│              └───────────────────┘                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Implementation

```python
from fastapi import FastAPI, Request
from fastapi.responses import Response
import httpx
import time
import json
import uuid

app = FastAPI()

class EvalProxy:
    """Evaluation proxy for LLM requests."""
    
    def __init__(self):
        self.upstream_url = "https://api.openai.com/v1"
        self.evaluation_store = EvalStore()
        self.safety_filter = SafetyFilter()
        self.evaluation_sampler = EvalSampler(sampling_rate=0.1)
    
    async def handle_request(
        self,
        request: Request,
        path: str
    ) -> Response:
        """Proxy and evaluate LLM request."""
        
        request_id = str(uuid.uuid4())[:12]
        start_time = time.time()
        
        # Read request body
        body = await request.body()
        request_data = json.loads(body)
        
        # Pre-request evaluation hooks
        await self._pre_request_hooks(request_id, request_data)
        
        # Forward to upstream
        async with httpx.AsyncClient() as client:
            upstream_response = await client.post(
                f"{self.upstream_url}/{path}",
                headers=self._filter_headers(request.headers),
                content=body,
                timeout=30.0
            )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Parse response
        response_data = json.loads(upstream_response.content)
        
        # Post-request evaluation hooks
        eval_result = await self._post_request_hooks(
            request_id=request_id,
            request_data=request_data,
            response_data=response_data,
            latency_ms=latency_ms
        )
        
        # Store for async evaluation
        if self.evaluation_sampler.should_evaluate(request_data):
            await self.evaluation_store.store_interaction({
                "request_id": request_id,
                "request": request_data,
                "response": response_data,
                "latency_ms": latency_ms,
                "eval_result": eval_result,
                "timestamp": time.time()
            })
        
        return Response(
            content=upstream_response.content,
            status_code=upstream_response.status_code,
            headers=dict(upstream_response.headers)
        )
    
    async def _pre_request_hooks(
        self, 
        request_id: str, 
        request_data: dict
    ):
        """Pre-request evaluation hooks."""
        
        # Log request
        logger.info(f"[{request_id}] Request received", extra={
            "model": request_data.get("model"),
            "tokens": request_data.get("max_tokens"),
            "user_id": request_data.get("metadata", {}).get("user_id")
        })
        
        # Check for prompt injection
        user_message = self._extract_user_message(request_data)
        injection_check = await self.safety_filter.check_injection(user_message)
        
        if injection_check["detected"]:
            logger.warning(f"[{request_id}] Potential injection detected")
    
    async def _post_request_hooks(
        self,
        request_id: str,
        request_data: dict,
        response_data: dict,
        latency_ms: float
    ) -> dict:
        """Post-request evaluation hooks."""
        
        eval_result = {}
        
        # Safety check on response
        response_text = self._extract_response_text(response_data)
        safety_check = await self.safety_filter.check_response(response_text)
        eval_result["safety"] = safety_check
        
        if safety_check["blocked"]:
            logger.warning(f"[{request_id}] Safety violation in response")
        
        # Cost tracking
        usage = response_data.get("usage", {})
        eval_result["cost"] = self._calculate_cost(
            request_data.get("model", ""),
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0)
        )
        
        # Latency tracking
        eval_result["latency_ms"] = latency_ms
        
        return eval_result
    
    def _calculate_cost(
        self, 
        model: str, 
        prompt_tokens: int, 
        completion_tokens: int
    ) -> float:
        """Calculate request cost."""
        
        PRICING = {
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
            "claude-3-haiku": {"input": 0.25, "output": 1.25},
        }
        
        pricing = PRICING.get(model, {"input": 2.50, "output": 10.00})
        
        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost


# FastAPI route
eval_proxy = EvalProxy()

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_handler(request: Request, path: str):
    return await eval_proxy.handle_request(request, path)
```

### Advanced Proxy Features

```python
class AdvancedEvalProxy(EvalProxy):
    """Advanced evaluation proxy with additional features."""
    
    def __init__(self):
        super().__init__()
        self.circuit_breaker = CircuitBreaker()
        self.model_router = ModelRouter()
        self.cache = ResponseCache()
        self.ab_test_manager = ABTestManager()
    
    async def handle_request_advanced(
        self,
        request: Request,
        path: str
    ) -> Response:
        """Advanced request handling with routing and A/B testing."""
        
        request_id = str(uuid.uuid4())[:12]
        body = await request.body()
        request_data = json.loads(body)
        
        # Check circuit breaker
        if self.circuit_breaker.is_open(request_data.get("model")):
            # Route to fallback model
            fallback = self.model_router.get_fallback(request_data.get("model"))
            request_data["model"] = fallback
            logger.warning(f"[{request_id}] Circuit breaker open, using fallback: {fallback}")
        
        # Check cache
        cache_key = self._generate_cache_key(request_data)
        cached = await self.cache.get(cache_key)
        
        if cached:
            logger.info(f"[{request_id}] Cache hit")
            return Response(
                content=json.dumps(cached),
                media_type="application/json"
            )
        
        # A/B test routing
        ab_assignment = self.ab_test_manager.assign_variant(request_data)
        if ab_assignment:
            request_data = self.ab_test_manager.apply_variant(
                request_data, ab_assignment
            )
            logger.info(f"[{request_id}] A/B test variant: {ab_assignment['variant']}")
        
        # Forward to upstream
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient() as client:
                upstream_response = await client.post(
                    f"{self.upstream_url}/{path}",
                    headers=self._filter_headers(request.headers),
                    content=json.dumps(request_data).encode(),
                    timeout=30.0
                )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Success - close circuit
            self.circuit_breaker.record_success(request_data.get("model"))
            
            # Cache successful response
            response_data = json.loads(upstream_response.content)
            await self.cache.set(cache_key, response_data, ttl=300)
            
            # A/B test result
            if ab_assignment:
                await self.ab_test_manager.record_result(
                    ab_assignment, response_data, latency_ms
                )
            
            return Response(
                content=upstream_response.content,
                status_code=upstream_response.status_code
            )
            
        except httpx.TimeoutException:
            # Failure - open circuit
            self.circuit_breaker.record_failure(request_data.get("model"))
            
            # Try fallback
            fallback = self.model_router.get_fallback(request_data.get("model"))
            if fallback:
                request_data["model"] = fallback
                return await self._forward_with_fallback(request_data, path)
            
            raise
```

---

## 2. A/B Testing for LLM Applications

### 2.1 LLM A/B Test Framework

```python
class LLMAExperiment:
    """A/B testing framework for LLM applications."""
    
    def __init__(self, experiment_name: str):
        self.name = experiment_name
        self.variants = {}
        self.results = {}
        self.traffic_allocation = {}
    
    def add_variant(
        self,
        name: str,
        config: dict,
        traffic_pct: float
    ):
        """Add an experiment variant."""
        
        self.variants[name] = config
        self.traffic_allocation[name] = traffic_pct
        self.results[name] = {
            "impressions": 0,
            "conversions": 0,
            "scores": [],
            "costs": [],
            "latencies": []
        }
    
    def assign_variant(self, user_id: str) -> str:
        """Deterministically assign user to variant."""
        
        import hashlib
        
        # Deterministic assignment based on user ID
        hash_val = int(hashlib.md5(
            f"{self.name}:{user_id}".encode()
        ).hexdigest(), 16)
        
        bucket = (hash_val % 100) / 100.0
        
        cumulative = 0
        for variant, pct in self.traffic_allocation.items():
            cumulative += pct
            if bucket < cumulative:
                return variant
        
        return list(self.variants.keys())[-1]
    
    async def evaluate_variant(
        self,
        variant: str,
        query: str,
        response: str,
        context: List[str] = None
    ) -> dict:
        """Evaluate a variant's response."""
        
        evaluator = ReferenceFreeEval()
        
        result = await evaluator.evaluate(query, response, context)
        
        # Record results
        self.results[variant]["scores"].append(result["overall"])
        self.results[variant]["impressions"] += 1
        
        return result
    
    def get_results(self) -> dict:
        """Get experiment results with statistical analysis."""
        
        import statistics
        
        results_summary = {}
        
        for variant, data in self.results.items():
            if len(data["scores"]) < 2:
                results_summary[variant] = {
                    "status": "insufficient_data",
                    "impressions": data["impressions"]
                }
                continue
            
            scores = data["scores"]
            
            results_summary[variant] = {
                "status": "ready",
                "impressions": data["impressions"],
                "mean_score": statistics.mean(scores),
                "stdev": statistics.stdev(scores),
                "median_score": statistics.median(scores),
                "ci_lower": statistics.mean(scores) - 1.96 * statistics.stdev(scores) / (len(scores) ** 0.5),
                "ci_upper": statistics.mean(scores) + 1.96 * statistics.stdev(scores) / (len(scores) ** 0.5),
            }
        
        # Determine winner
        if all(r.get("status") == "ready" for r in results_summary.values()):
            best_variant = max(
                results_summary.items(),
                key=lambda x: x[1]["mean_score"]
            )
            
            # Statistical significance test
            variants = list(results_summary.keys())
            if len(variants) == 2:
                sig_test = self._t_test(
                    self.results[variants[0]]["scores"],
                    self.results[variants[1]]["scores"]
                )
                results_summary["statistical_test"] = sig_test
                results_summary["winner"] = best_variant[0] if sig_test["significant"] else None
            else:
                results_summary["winner"] = best_variant[0]
        
        return results_summary
    
    def _t_test(self, scores_a: List[float], scores_b: List[float]) -> dict:
        """Independent samples t-test."""
        
        import math
        
        n_a, n_b = len(scores_a), len(scores_b)
        mean_a = sum(scores_a) / n_a
        mean_b = sum(scores_b) / n_b
        
        var_a = sum((x - mean_a) ** 2 for x in scores_a) / (n_a - 1)
        var_b = sum((x - mean_b) ** 2 for x in scores_b) / (n_b - 1)
        
        se = math.sqrt(var_a / n_a + var_b / n_b)
        
        if se == 0:
            return {"t_statistic": 0, "p_value": 1, "significant": False}
        
        t_stat = (mean_a - mean_b) / se
        
        # Approximate p-value using normal distribution (for large n)
        p_value = 2 * (1 - self._normal_cdf(abs(t_stat)))
        
        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "effect_size": (mean_a - mean_b) / math.sqrt((var_a + var_b) / 2) if (var_a + var_b) > 0 else 0
        }
    
    @staticmethod
    def _normal_cdf(x: float) -> float:
        """Approximate normal CDF."""
        import math
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# Example usage
experiment = LLMAExperiment("prompt_v2_vs_v3")

experiment.add_variant("v2_current", {
    "system_prompt": "You are a helpful assistant.",
    "model": "gpt-4o-mini"
}, traffic_pct=0.5)

experiment.add_variant("v3_new", {
    "system_prompt": "You are a helpful assistant. Be concise and accurate.",
    "model": "gpt-4o-mini"
}, traffic_pct=0.5)

# Run experiment for 1000 users
for user_id in range(1000):
    variant = experiment.assign_variant(str(user_id))
    # ... run query and get response
    # await experiment.evaluate_variant(variant, query, response)

results = experiment.get_results()
print(f"Winner: {results.get('winner', 'No significant winner')}")
```

### 2.2 Multi-Armed Bandit for LLM Optimization

```python
import random
import math

class ThompsonSamplingBandit:
    """Thompson Sampling for dynamic LLM optimization."""
    
    def __init__(self, arms: List[dict]):
        """
        arms: List of LLM configurations to test
        Each arm has: name, model, prompt, temperature, etc.
        """
        self.arms = arms
        # Beta distribution parameters
        self.alpha = [1.0] * len(arms)  # successes
        self.beta = [1.0] * len(arms)   # failures
        self.total_pulls = [0] * len(arms)
        self.total_rewards = [0.0] * len(arms)
    
    def select_arm(self) -> int:
        """Select arm using Thompson Sampling."""
        
        samples = []
        for i in range(len(self.arms)):
            # Sample from Beta distribution
            sample = random.betavariate(self.alpha[i], self.beta[i])
            samples.append((sample, i))
        
        # Return arm with highest sample
        return max(samples, key=lambda x: x[0])[1]
    
    def update(self, arm_idx: int, reward: float):
        """Update arm with observed reward (0-1 scale)."""
        
        self.total_pulls[arm_idx] += 1
        self.total_rewards[arm_idx] += reward
        
        # Update Beta distribution
        self.alpha[arm_idx] += reward
        self.beta[arm_idx] += (1 - reward)
    
    def get_arm_stats(self) -> List[dict]:
        """Get statistics for each arm."""
        
        stats = []
        
        for i, arm in enumerate(self.arms):
            n = self.total_pulls[i]
            mean_reward = self.total_rewards[i] / n if n > 0 else 0
            
            # Calculate credible interval
            alpha_val = self.alpha[i]
            beta_val = self.beta[i]
            
            # Mean of Beta distribution
            beta_mean = alpha_val / (alpha_val + beta_val)
            
            # 95% credible interval
            import scipy.stats as stats
            ci_lower = stats.beta.ppf(0.025, alpha_val, beta_val)
            ci_upper = stats.beta.ppf(0.975, alpha_val, beta_val)
            
            stats.append({
                "name": arm["name"],
                "model": arm.get("model", "unknown"),
                "pulls": n,
                "mean_reward": mean_reward,
                "beta_mean": beta_mean,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "estimated_value": beta_mean
            })
        
        return sorted(stats, key=lambda x: x["estimated_value"], reverse=True)
    
    async def optimize(
        self,
        evaluation_fn,
        num_iterations: int = 1000
    ) -> dict:
        """Run bandit optimization."""
        
        for _ in range(num_iterations):
            # Select arm
            arm_idx = self.select_arm()
            arm = self.arms[arm_idx]
            
            # Evaluate
            reward = await evaluation_fn(arm)
            
            # Update
            self.update(arm_idx, reward)
        
        # Get final results
        stats = self.get_arm_stats()
        
        return {
            "best_arm": stats[0],
            "all_arms": stats,
            "total_evaluations": num_iterations,
            "convergence": self._check_convergence(stats)
        }
    
    def _check_convergence(self, stats: List[dict]) -> dict:
        """Check if bandit has converged."""
        
        if len(stats) < 2:
            return {"converged": False, "reason": "Not enough arms"}
        
        best = stats[0]
        second = stats[1]
        
        # Check if best arm's CI lower is above second arm's CI upper
        converged = best["ci_lower"] > second["ci_upper"]
        
        return {
            "converged": converged,
            "best_value": best["estimated_value"],
            "second_value": second["estimated_value"],
            "gap": best["estimated_value"] - second["estimated_value"]
        }
```

---

## 3. Model Routing Evaluation

### 3.1 Smart Model Router with Evaluation

```python
class EvalAwareModelRouter:
    """Route requests to optimal model based on evaluation data."""
    
    def __init__(self):
        self.model_profiles = {}
        self.routing_rules = []
        self.performance_cache = {}
    
    async def route_request(
        self,
        request: dict,
        quality_requirement: float = 0.8,
        cost_budget: float = 0.01,
        latency_budget_ms: float = 3000
    ) -> dict:
        """Route request to optimal model."""
        
        # Classify request complexity
        complexity = await self._classify_complexity(request)
        
        # Get candidate models
        candidates = self._get_candidates(
            complexity, quality_requirement, cost_budget, latency_budget_ms
        )
        
        if not candidates:
            # Fallback to best available
            return self._get_fallback_model()
        
        # Select optimal model
        selected = self._select_optimal(candidates, quality_requirement)
        
        return {
            "model": selected["model"],
            "provider": selected["provider"],
            "estimated_cost": selected["estimated_cost"],
            "estimated_latency_ms": selected["estimated_latency_ms"],
            "expected_quality": selected["expected_quality"],
            "routing_reason": selected["routing_reason"],
            "complexity": complexity
        }
    
    async def _classify_complexity(self, request: dict) -> str:
        """Classify request complexity."""
        
        user_message = request.get("messages", [{}])[-1].get("content", "")
        
        # Simple heuristics
        indicators = {
            "length": len(user_message),
            "has_code": "```" in user_message,
            "has_numbers": any(c.isdigit() for c in user_message),
            "question_words": sum(1 for w in ["why", "how", "explain", "compare", "analyze"] 
                                  if w in user_message.lower()),
        }
        
        # Complexity scoring
        score = 0
        score += min(indicators["length"] / 500, 1.0) * 0.2
        score += (1.0 if indicators["has_code"] else 0) * 0.3
        score += min(indicators["question_words"] / 3, 1.0) * 0.3
        score += min(indicators["length"] / 1000, 1.0) * 0.2
        
        if score < 0.3:
            return "simple"
        elif score < 0.6:
            return "moderate"
        else:
            return "complex"
    
    def _get_candidates(
        self,
        complexity: str,
        quality_req: float,
        cost_budget: float,
        latency_budget: float
    ) -> List[dict]:
        """Get candidate models meeting requirements."""
        
        MODEL_PROFILES = {
            "gpt-4o": {
                "quality": 0.92, "cost_per_1k": 0.0075, "latency_ms": 1500,
                "complexity_handling": ["simple", "moderate", "complex"]
            },
            "gpt-4o-mini": {
                "quality": 0.82, "cost_per_1k": 0.00045, "latency_ms": 800,
                "complexity_handling": ["simple", "moderate"]
            },
            "claude-3.5-sonnet": {
                "quality": 0.90, "cost_per_1k": 0.009, "latency_ms": 1800,
                "complexity_handling": ["simple", "moderate", "complex"]
            },
            "claude-3-haiku": {
                "quality": 0.78, "cost_per_1k": 0.000625, "latency_ms": 600,
                "complexity_handling": ["simple"]
            },
            "llama-3.1-70b": {
                "quality": 0.85, "cost_per_1k": 0.0009, "latency_ms": 1200,
                "complexity_handling": ["simple", "moderate", "complex"]
            }
        }
        
        candidates = []
        
        for model, profile in MODEL_PROFILES.items():
            if complexity not in profile["complexity_handling"]:
                continue
            
            if profile["quality"] < quality_req:
                continue
            
            if profile["cost_per_1k"] > cost_budget:
                continue
            
            if profile["latency_ms"] > latency_budget:
                continue
            
            candidates.append({
                "model": model,
                "quality": profile["quality"],
                "cost": profile["cost_per_1k"],
                "latency": profile["latency_ms"]
            })
        
        return candidates
    
    def _select_optimal(
        self, 
        candidates: List[dict],
        quality_req: float
    ) -> dict:
        """Select optimal model from candidates."""
        
        # Score each candidate
        scored = []
        
        for c in candidates:
            # Weighted score: quality 40%, cost 30%, latency 30%
            quality_score = c["quality"]
            cost_score = 1.0 - (c["cost"] / 0.01)  # Normalize to budget
            latency_score = 1.0 - (c["latency"] / 3000)  # Normalize
            
            total_score = (
                quality_score * 0.4 +
                cost_score * 0.3 +
                latency_score * 0.3
            )
            
            scored.append({
                **c,
                "total_score": total_score,
                "routing_reason": f"Quality: {quality_score:.2f}, Cost: {c['cost']:.4f}, Latency: {c['latency']}ms"
            })
        
        return max(scored, key=lambda x: x["total_score"])
```

### 3.2 Model Routing Evaluation

```python
class ModelRoutingEvaluator:
    """Evaluate model routing decisions."""
    
    def __init__(self):
        self.routing_log = []
    
    async def evaluate_routing(
        self,
        requests: List[dict],
        router: EvalAwareModelRouter
    ) -> dict:
        """Evaluate routing decisions against optimal."""
        
        results = {
            "total_requests": len(requests),
            "routing_decisions": [],
            "cost_savings": 0,
            "quality_impact": 0,
            "routing_accuracy": 0
        }
        
        for request in requests:
            # Get router's decision
            routing = await router.route_request(request)
            
            # Determine what the optimal model would be
            optimal = await self._determine_optimal(request)
            
            # Compare
            comparison = {
                "request_id": request.get("id"),
                "router_choice": routing["model"],
                "optimal_choice": optimal["model"],
                "matches_optimal": routing["model"] == optimal["model"],
                "cost_difference": routing["estimated_cost"] - optimal["estimated_cost"],
                "quality_difference": routing["expected_quality"] - optimal["expected_quality"]
            }
            
            results["routing_decisions"].append(comparison)
        
        # Aggregate
        matches = sum(1 for d in results["routing_decisions"] if d["matches_optimal"])
        results["routing_accuracy"] = matches / len(requests)
        results["avg_cost_difference"] = sum(d["cost_difference"] for d in results["routing_decisions"]) / len(requests)
        results["avg_quality_difference"] = sum(d["quality_difference"] for d in results["routing_decisions"]) / len(requests)
        
        return results
    
    async def _determine_optimal(self, request: dict) -> dict:
        """Determine optimal model for a request."""
        
        # In practice, this would be determined by:
        # 1. Running evaluation on multiple models
        # 2. Selecting based on quality/cost trade-off
        
        # Simplified: assume optimal is the cheapest that meets quality threshold
        return {
            "model": "gpt-4o-mini",
            "reason": "Meets quality threshold at lowest cost"
        }
```

---

## 4. Large-Scale Evaluation Systems

### 4.1 Distributed Evaluation Architecture

```
┌─────────────────────────────────────────────────────────────┐
│            DISTRIBUTED EVALUATION ARCHITECTURE               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │                    Evaluation Orchestrator             │ │
│  │  • Task scheduling    • Worker management             │ │
│  │  • Result aggregation • Progress tracking             │ │
│  └───────────────────────┬───────────────────────────────┘ │
│                          │                                  │
│  ┌───────────┬───────────┼───────────┬───────────┐        │
│  │           │           │           │           │        │
│  ▼           ▼           ▼           ▼           ▼        │
│  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐          │
│  │ W1  │  │ W2  │  │ W3  │  │ W4  │  │ WN  │          │
│  │     │  │     │  │     │  │     │  │     │          │
│  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘          │
│     │        │        │        │        │                │
│     ▼        ▼        ▼        ▼        ▼                │
│  ┌──────────────────────────────────────────────────┐   │
│  │              Result Store (Redis/Postgres)        │   │
│  └──────────────────────────────────────────────────┘   │
│                          │                                │
│                          ▼                                │
│  ┌──────────────────────────────────────────────────┐   │
│  │              Analytics & Reporting                │   │
│  └──────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Distributed Evaluation Implementation

```python
import asyncio
from typing import List, Optional
import json

class DistributedEvaluator:
    """Distributed evaluation system for large-scale LLM evaluation."""
    
    def __init__(
        self,
        num_workers: int = 8,
        batch_size: int = 100,
        max_concurrent_per_worker: int = 5
    ):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent_per_worker
        self.result_store = ResultStore()
        self.task_queue = asyncio.Queue()
    
    async def evaluate_suite(
        self,
        test_cases: List[dict],
        model_fn,
        evaluator_fn,
        progress_callback=None
    ) -> dict:
        """Run evaluation distributed across workers."""
        
        total_cases = len(test_cases)
        
        # Split into batches
        batches = [
            test_cases[i:i + self.batch_size]
            for i in range(0, total_cases, self.batch_size)
        ]
        
        # Enqueue batches
        for batch in batches:
            await self.task_queue.put(batch)
        
        # Start workers
        workers = [
            self._worker(model_fn, evaluator_fn)
            for _ in range(self.num_workers)
        ]
        
        # Wait for completion
        await asyncio.gather(*workers)
        
        # Aggregate results
        results = await self._aggregate_results()
        
        return {
            "total_cases": total_cases,
            "total_batches": len(batches),
            "results": results,
            "summary": self._generate_summary(results)
        }
    
    async def _worker(self, model_fn, evaluator_fn):
        """Worker process for evaluation."""
        
        while True:
            try:
                batch = self.task_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            async def eval_case(case):
                async with semaphore:
                    try:
                        # Get model response
                        response = await model_fn(case["input"])
                        
                        # Evaluate
                        result = await evaluator_fn(case, response)
                        
                        # Store result
                        await self.result_store.store(case["id"], result)
                        
                        return result
                    except Exception as e:
                        logger.error(f"Eval failed for case {case['id']}: {e}")
                        return {"error": str(e)}
            
            # Evaluate batch concurrently
            results = await asyncio.gather(*[
                eval_case(case) for case in batch
            ])
            
            self.task_queue.task_done()
    
    async def _aggregate_results(self) -> dict:
        """Aggregate all evaluation results."""
        
        all_results = await self.result_store.get_all()
        
        # Calculate metrics
        scores = [r.get("score", 0) for r in all_results if "score" in r]
        
        return {
            "total_evaluated": len(all_results),
            "successful": len(scores),
            "failed": len(all_results) - len(scores),
            "mean_score": sum(scores) / len(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "score_distribution": self._histogram(scores),
            "by_category": self._group_by_category(all_results),
            "by_difficulty": self._group_by_difficulty(all_results),
        }
    
    def _histogram(self, scores: List[float], bins: int = 10) -> dict:
        """Create score histogram."""
        
        if not scores:
            return {}
        
        min_score = min(scores)
        max_score = max(scores)
        bin_width = (max_score - min_score) / bins if max_score > min_score else 0.1
        
        histogram = {}
        for i in range(bins):
            bin_start = min_score + i * bin_width
            bin_end = bin_start + bin_width
            bin_label = f"{bin_start:.2f}-{bin_end:.2f}"
            histogram[bin_label] = sum(
                1 for s in scores if bin_start <= s < bin_end
            )
        
        return histogram
```

### 4.3 Evaluation Caching

```python
class EvalCache:
    """Cache evaluation results to avoid redundant evaluation."""
    
    def __init__(self, storage_backend, ttl_seconds: int = 86400):
        self.storage = storage_backend
        self.ttl = ttl_seconds
    
    async def get_cached_result(
        self,
        test_case_hash: str,
        model_hash: str,
        prompt_hash: str
    ) -> Optional[dict]:
        """Get cached evaluation result."""
        
        cache_key = f"eval:{test_case_hash}:{model_hash}:{prompt_hash}"
        
        cached = await self.storage.get(cache_key)
        
        if cached and self._is_fresh(cached):
            return cached["result"]
        
        return None
    
    async def store_result(
        self,
        test_case_hash: str,
        model_hash: str,
        prompt_hash: str,
        result: dict
    ):
        """Store evaluation result in cache."""
        
        cache_key = f"eval:{test_case_hash}:{model_hash}:{prompt_hash}"
        
        await self.storage.set(cache_key, {
            "result": result,
            "timestamp": time.time(),
            "test_case_hash": test_case_hash,
            "model_hash": model_hash,
            "prompt_hash": prompt_hash
        }, ttl=self.ttl)
    
    def _is_fresh(self, cached: dict) -> bool:
        """Check if cached result is still fresh."""
        return (time.time() - cached["timestamp"]) < self.ttl
    
    async def invalidate(
        self,
        model_hash: str = None,
        prompt_hash: str = None
    ):
        """Invalidate cache entries matching criteria."""
        
        if model_hash:
            await self.storage.delete_pattern(f"eval:*:{model_hash}:*")
        if prompt_hash:
            await self.storage.delete_pattern(f"eval:*:*:{prompt_hash}")
```

---

## 5. Evaluation Infrastructure Patterns

### 5.1 Evaluation Service Pattern

```python
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Evaluation Service")

class EvaluationRequest(BaseModel):
    suite_name: str
    model_config: dict
    prompt_config: dict
    evaluation_config: dict
    async_mode: bool = True

class EvaluationResponse(BaseModel):
    eval_id: str
    status: str
    estimated_time_seconds: int
    results_url: Optional[str] = None

@app.post("/evaluate", response_model=EvaluationResponse)
async def start_evaluation(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks
):
    """Start an evaluation run."""
    
    eval_id = str(uuid.uuid4())[:12]
    
    if request.async_mode:
        # Run in background
        background_tasks.add_task(
            run_evaluation_async,
            eval_id,
            request
        )
        
        return EvaluationResponse(
            eval_id=eval_id,
            status="started",
            estimated_time_seconds=estimate_time(request),
            results_url=f"/evaluate/{eval_id}/results"
        )
    else:
        # Run synchronously
        results = await run_evaluation_sync(request)
        return EvaluationResponse(
            eval_id=eval_id,
            status="completed",
            results_url=f"/evaluate/{eval_id}/results"
        )

@app.get("/evaluate/{eval_id}/results")
async def get_results(eval_id: str):
    """Get evaluation results."""
    
    results = await eval_store.get(eval_id)
    
    if not results:
        return {"status": "not_found"}
    
    return results

@app.get("/evaluate/{eval_id}/progress")
async def get_progress(eval_id: str):
    """Get evaluation progress."""
    
    progress = await eval_store.get_progress(eval_id)
    
    return {
        "eval_id": eval_id,
        "total_cases": progress["total"],
        "completed": progress["completed"],
        "failed": progress["failed"],
        "percent_complete": progress["completed"] / progress["total"] * 100 if progress["total"] > 0 else 0,
        "estimated_remaining_seconds": estimate_remaining(progress)
    }

@app.get("/suites")
async def list_suites():
    """List available evaluation suites."""
    
    suites = await suite_manager.list_suites()
    
    return {
        "suites": [
            {
                "name": s["name"],
                "num_cases": s["num_cases"],
                "last_updated": s["last_updated"],
                "description": s.get("description", "")
            }
            for s in suites
        ]
    }

@app.post("/suites/{suite_name}/cases")
async def add_test_cases(
    suite_name: str,
    cases: List[dict]
):
    """Add test cases to a suite."""
    
    await suite_manager.add_cases(suite_name, cases)
    
    return {"status": "added", "count": len(cases)}
```

### 5.2 Evaluation Dashboard API

```python
@app.get("/dashboard/overview")
async def dashboard_overview():
    """Get evaluation dashboard overview."""
    
    # Get latest results for each suite
    latest_results = await eval_store.get_latest_per_suite()
    
    # Calculate trends
    trends = await calculate_trends(latest_results)
    
    # Get alerts
    active_alerts = await alert_manager.get_active()
    
    return {
        "suites": {
            name: {
                "latest_score": result["overall_score"],
                "trend": trends.get(name, {}),
                "num_cases": result["total_cases"],
                "last_evaluated": result["timestamp"],
                "passes": result["passes"]
            }
            for name, result in latest_results.items()
        },
        "alerts": active_alerts,
        "cost_summary": await get_cost_summary(),
        "system_health": await get_system_health()
    }

@app.get("/dashboard/trends/{suite_name}")
async def suite_trends(
    suite_name: str,
    days: int = 30
):
    """Get score trends for a suite."""
    
    history = await eval_store.get_history(suite_name, days=days)
    
    return {
        "suite": suite_name,
        "period_days": days,
        "data_points": [
            {
                "date": h["timestamp"],
                "overall_score": h["overall_score"],
                "by_metric": h.get("metrics", {}),
                "num_cases": h.get("total_cases", 0)
            }
            for h in history
        ],
        "summary": {
            "avg_score": sum(h["overall_score"] for h in history) / len(history) if history else 0,
            "min_score": min(h["overall_score"] for h in history) if history else 0,
            "max_score": max(h["overall_score"] for h in history) if history else 0,
            "trend": calculate_trend_direction(history)
        }
    }
```

---

## 6. Evaluation Data Management

### 6.1 Data Versioning for Evaluation

```python
class EvalDataManager:
    """Manage evaluation data versions and lineage."""
    
    def __init__(self, storage_backend):
        self.storage = storage_backend
        self.versions = {}
    
    async def create_snapshot(
        self,
        name: str,
        test_cases: List[dict],
        metadata: dict
    ) -> str:
        """Create a snapshot of evaluation data."""
        
        version_id = f"v{int(time.time())}"
        
        snapshot = {
            "version_id": version_id,
            "name": name,
            "test_cases": test_cases,
            "metadata": {
                **metadata,
                "created_at": datetime.now().isoformat(),
                "num_cases": len(test_cases),
                "hash": self._compute_hash(test_cases)
            }
        }
        
        await self.storage.save(
            f"eval_data/{name}/{version_id}.json",
            snapshot
        )
        
        self.versions[f"{name}:{version_id}"] = snapshot
        
        return version_id
    
    async def compare_versions(
        self,
        name: str,
        version_a: str,
        version_b: str
    ) -> dict:
        """Compare two versions of evaluation data."""
        
        snap_a = await self.storage.load(f"eval_data/{name}/{version_a}.json")
        snap_b = await self.storage.load(f"eval_data/{name}/{version_b}.json")
        
        ids_a = {tc["id"] for tc in snap_a["test_cases"]}
        ids_b = {tc["id"] for tc in snap_b["test_cases"]}
        
        # Find changes
        added = ids_b - ids_a
        removed = ids_a - ids_b
        common = ids_a & ids_b
        
        # Check for modifications in common cases
        modified = []
        cases_a = {tc["id"]: tc for tc in snap_a["test_cases"]}
        cases_b = {tc["id"]: tc for tc in snap_b["test_cases"]}
        
        for case_id in common:
            if cases_a[case_id] != cases_b[case_id]:
                modified.append(case_id)
        
        return {
            "version_a": version_a,
            "version_b": version_b,
            "added": list(added),
            "removed": list(removed),
            "modified": modified,
            "unchanged": list(common - set(modified)),
            "summary": {
                "total_a": len(ids_a),
                "total_b": len(ids_b),
                "num_added": len(added),
                "num_removed": len(removed),
                "num_modified": len(modified),
                "num_unchanged": len(common) - len(modified)
            }
        }
    
    def _compute_hash(self, test_cases: List[dict]) -> str:
        """Compute hash of test cases."""
        import hashlib
        import json
        
        content = json.dumps(test_cases, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
```

### 6.2 Evaluation Data Quality

```python
class EvalDataQualityChecker:
    """Check quality of evaluation data."""
    
    async def check_suite_quality(
        self,
        test_cases: List[dict]
    ) -> dict:
        """Check quality of a test suite."""
        
        issues = []
        warnings = []
        
        # Check for duplicate inputs
        inputs = [tc.get("input", "") for tc in test_cases]
        duplicates = self._find_duplicates(inputs)
        if duplicates:
            issues.append({
                "type": "duplicate_inputs",
                "count": len(duplicates),
                "description": f"{len(duplicates)} test cases have identical inputs"
            })
        
        # Check for empty/missing fields
        for i, tc in enumerate(test_cases):
            if not tc.get("input"):
                issues.append({
                    "type": "empty_input",
                    "case_index": i,
                    "case_id": tc.get("id"),
                    "description": "Test case has empty input"
                })
            
            if not tc.get("expected_output") and tc.get("expected_output") != "":
                warnings.append({
                    "type": "missing_reference",
                    "case_index": i,
                    "case_id": tc.get("id"),
                    "description": "Test case has no expected output (reference-free evaluation only)"
                })
        
        # Check distribution
        categories = [tc.get("category", "unknown") for tc in test_cases]
        category_dist = {c: categories.count(c) for c in set(categories)}
        
        # Flag imbalanced categories
        total = len(test_cases)
        for cat, count in category_dist.items():
            if count / total < 0.05:
                warnings.append({
                    "type": "imbalanced_category",
                    "category": cat,
                    "count": count,
                    "percentage": count / total * 100,
                    "description": f"Category '{cat}' has only {count/total*100:.1f}% of cases"
                })
        
        return {
            "total_cases": len(test_cases),
            "issues": issues,
            "warnings": warnings,
            "quality_score": 1.0 - (len(issues) * 0.1) - (len(warnings) * 0.02),
            "category_distribution": category_dist,
            "pass": len(issues) == 0
        }
    
    def _find_duplicates(self, items: List[str]) -> List[int]:
        """Find duplicate items."""
        seen = {}
        duplicates = []
        
        for i, item in enumerate(items):
            normalized = item.strip().lower()
            if normalized in seen:
                duplicates.append(i)
            else:
                seen[normalized] = i
        
        return duplicates
```

---

## 7. Evaluation for Fine-Tuned Models

### 7.1 Fine-Tuning Evaluation Pipeline

```python
class FineTuningEvaluator:
    """Evaluate fine-tuned models against base models."""
    
    async def evaluate_fine_tune(
        self,
        base_model: str,
        fine_tuned_model: str,
        eval_suite: str,
        comparison_metrics: List[str] = None
    ) -> dict:
        """Compare fine-tuned model against base model."""
        
        if comparison_metrics is None:
            comparison_metrics = [
                "task_accuracy",
                "format_compliance",
                "tone_consistency",
                "hallucination_rate",
                "safety_score"
            ]
        
        # Load test cases
        test_cases = await load_eval_suite(eval_suite)
        
        # Evaluate both models
        base_results = await self._evaluate_model(base_model, test_cases)
        ft_results = await self._evaluate_model(fine_tuned_model, test_cases)
        
        # Compare
        comparison = {}
        for metric in comparison_metrics:
            base_score = base_results.get("metrics", {}).get(metric, 0)
            ft_score = ft_results.get("metrics", {}).get(metric, 0)
            
            comparison[metric] = {
                "base": base_score,
                "fine_tuned": ft_score,
                "delta": ft_score - base_score,
                "improved": ft_score > base_score
            }
        
        # Overall assessment
        improvements = sum(1 for c in comparison.values() if c["improved"])
        regressions = sum(1 for c in comparison.values() if not c["improved"])
        
        return {
            "base_model": base_model,
            "fine_tuned_model": fine_tuned_model,
            "eval_suite": eval_suite,
            "comparison": comparison,
            "summary": {
                "improvements": improvements,
                "regressions": regressions,
                "overall_delta": sum(c["delta"] for c in comparison.values()) / len(comparison),
                "recommendation": self._generate_recommendation(comparison)
            }
        }
    
    def _generate_recommendation(self, comparison: dict) -> str:
        """Generate recommendation based on comparison."""
        
        improvements = sum(1 for c in comparison.values() if c["delta"] > 0.05)
        regressions = sum(1 for c in comparison.values() if c["delta"] < -0.05)
        
        if regressions > improvements:
            return "⚠️ Fine-tuned model shows significant regressions. Consider additional training data or hyperparameter tuning."
        elif improvements > regressions:
            return "✅ Fine-tuned model shows improvements. Ready for production testing."
        else:
            return "🔄 Mixed results. Fine-tuned model shows trade-offs. Evaluate business impact of specific metric changes."
```

### 7.2 Training Data Evaluation

```python
class TrainingDataEvaluator:
    """Evaluate quality of training data for fine-tuning."""
    
    async def evaluate_training_data(
        self,
        training_data: List[dict],
        validation_data: List[dict]
    ) -> dict:
        """Evaluate training data quality."""
        
        checks = {}
        
        # 1. Volume check
        checks["volume"] = {
            "training_samples": len(training_data),
            "validation_samples": len(validation_data),
            "ratio": len(training_data) / len(validation_data) if validation_data else 0,
            "adequate": len(training_data) >= 100 and len(validation_data) >= 20
        }
        
        # 2. Quality check (sample-based)
        sample = random.sample(training_data, min(100, len(training_data)))
        checks["quality"] = await self._check_data_quality(sample)
        
        # 3. Diversity check
        checks["diversity"] = self._check_diversity(training_data)
        
        # 4. Consistency check
        checks["consistency"] = await self._check_consistency(training_data)
        
        # 5. Label quality
        checks["label_quality"] = await self._check_label_quality(training_data)
        
        # Overall assessment
        all_adequate = all(
            check.get("adequate", True) 
            for check in checks.values()
        )
        
        return {
            "checks": checks,
            "overall_adequate": all_adequate,
            "recommendations": self._generate_recommendations(checks)
        }
```

---

## 8. Evaluation for Multi-Modal Systems

### 8.1 Multi-Modal Evaluation Framework

```python
class MultiModalEvaluator:
    """Evaluate multi-modal LLM systems."""
    
    MODALITY_METRICS = {
        "text": {
            "metrics": ["coherence", "relevance", "accuracy"],
            "tools": ["ragas", "deepeval"]
        },
        "image": {
            "metrics": ["visual_accuracy", "description_quality", "spatial_reasoning"],
            "tools": ["vlm_eval", "custom_judge"]
        },
        "audio": {
            "metrics": ["transcription_accuracy", "speaker_identification", "emotion_detection"],
            "tools": ["wer", "custom_judge"]
        },
        "video": {
            "metrics": ["temporal_understanding", "action_recognition", "summarization_quality"],
            "tools": ["custom_judge", "human_eval"]
        }
    }
    
    async def evaluate_multimodal(
        self,
        inputs: dict,  # {"text": "...", "image": "...", etc.}
        response: str,
        expected: dict = None
    ) -> dict:
        """Evaluate multi-modal response."""
        
        results = {}
        
        # Determine which modalities are present
        modalities = [m for m in inputs.keys() if inputs[m] is not None]
        
        for modality in modalities:
            if modality in self.MODALITY_METRICS:
                modality_result = await self._evaluate_modality(
                    modality,
                    inputs[modality],
                    response,
                    expected.get(modality) if expected else None
                )
                results[modality] = modality_result
        
        # Cross-modal consistency
        if len(modalities) > 1:
            results["cross_modal_consistency"] = await self._check_cross_modal_consistency(
                inputs, response
            )
        
        # Overall score
        overall = self._aggregate_multimodal_scores(results)
        
        return {
            "overall": overall,
            "by_modality": results,
            "modalities_present": modalities
        }
    
    async def _evaluate_modality(
        self,
        modality: str,
        input_data,
        response: str,
        expected=None
    ) -> dict:
        """Evaluate a single modality."""
        
        metrics = self.MODALITY_METRICS[modality]["metrics"]
        
        results = {}
        for metric in metrics:
            results[metric] = await self._evaluate_metric(
                modality, metric, input_data, response, expected
            )
        
        return results
    
    async def _check_cross_modal_consistency(
        self,
        inputs: dict,
        response: str
    ) -> dict:
        """Check consistency across modalities."""
        
        # Use LLM to check if response is consistent with all inputs
        prompt = f"""Check if this response is consistent with ALL provided inputs.

Text input: {inputs.get('text', 'N/A')}
Image description: {inputs.get('image', 'N/A')}
Audio transcript: {inputs.get('audio', 'N/A')}

Response: {response}

Rate consistency (0-1): Does the response accurately reflect ALL input modalities?
Return: {{"score": 0.0, "inconsistencies": ["..."]}}"""
        
        result = await call_llm("gpt-4o", prompt)
        return parse_consistency_result(result)
```

---

## 9. Evaluation Debugging and Root Cause Analysis

### 9.1 Evaluation Failure Analysis

```python
class EvalFailureAnalyzer:
    """Analyze evaluation failures to identify root causes."""
    
    async def analyze_failures(
        self,
        failures: List[dict]
    ) -> dict:
        """Analyze evaluation failures for patterns."""
        
        # Categorize failures
        categories = self._categorize_failures(failures)
        
        # Find patterns
        patterns = self._find_patterns(failures)
        
        # Generate root causes
        root_causes = await self._identify_root_causes(categories, patterns)
        
        # Generate fixes
        fixes = await self._suggest_fixes(root_causes)
        
        return {
            "total_failures": len(failures),
            "categories": categories,
            "patterns": patterns,
            "root_causes": root_causes,
            "suggested_fixes": fixes,
            "priority_order": self._prioritize_fixes(root_causes, fixes)
        }
    
    def _categorize_failures(self, failures: List[dict]) -> dict:
        """Categorize failures by type."""
        
        categories = {
            "safety_violations": [],
            "hallucinations": [],
            "irrelevant_responses": [],
            "format_violations": [],
            "incomplete_responses": [],
            "tone_issues": [],
            "other": []
        }
        
        for failure in failures:
            failure_type = failure.get("failure_type", "other")
            
            if failure_type in categories:
                categories[failure_type].append(failure)
            else:
                categories["other"].append(failure)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
    
    def _find_patterns(self, failures: List[dict]) -> List[dict]:
        """Find common patterns in failures."""
        
        patterns = []
        
        # Check for input patterns
        input_lengths = [len(f.get("input", "")) for f in failures]
        if input_lengths:
            avg_length = sum(input_lengths) / len(input_lengths)
            if avg_length > 500:
                patterns.append({
                    "type": "long_inputs",
                    "description": f"Failures tend to have longer inputs (avg {avg_length:.0f} chars)",
                    "suggestion": "Consider chunking long inputs or adjusting context window"
                })
        
        # Check for category clustering
        categories = [f.get("category", "unknown") for f in failures]
        from collections import Counter
        cat_counts = Counter(categories)
        
        for cat, count in cat_counts.most_common(3):
            if count > len(failures) * 0.3:
                patterns.append({
                    "type": "category_cluster",
                    "category": cat,
                    "count": count,
                    "percentage": count / len(failures) * 100,
                    "description": f"Category '{cat}' accounts for {count/len(failures)*100:.0f}% of failures"
                })
        
        return patterns
    
    async def _identify_root_causes(
        self,
        categories: dict,
        patterns: List[dict]
    ) -> List[dict]:
        """Identify root causes of failures."""
        
        root_causes = []
        
        # Analyze each failure category
        for category, failures in categories.items():
            if not failures:
                continue
            
            # Sample failures for analysis
            sample = random.sample(failures, min(5, len(failures)))
            
            # Use LLM to analyze root causes
            analysis = await self._llm_analyze_failures(category, sample)
            
            root_causes.append({
                "category": category,
                "count": len(failures),
                "analysis": analysis,
                "estimated_effort": analysis.get("effort", "medium")
            })
        
        return sorted(root_causes, key=lambda x: x["count"], reverse=True)
    
    async def _suggest_fixes(self, root_causes: List[dict]) -> List[dict]:
        """Suggest fixes for root causes."""
        
        fixes = []
        
        for cause in root_causes:
            fix_suggestions = await self._generate_fix_suggestions(cause)
            
            fixes.append({
                "root_cause": cause["category"],
                "suggestions": fix_suggestions,
                "priority": self._calculate_priority(cause)
            })
        
        return sorted(fixes, key=lambda x: x["priority"], reverse=True)
```

### 9.2 Evaluation Debugging Tools

```python
class EvalDebugger:
    """Debug evaluation failures."""
    
    async def debug_failure(
        self,
        test_case: dict,
        response: str,
        eval_result: dict
    ) -> dict:
        """Debug a specific evaluation failure."""
        
        debug_info = {
            "test_case": test_case,
            "response": response,
            "eval_result": eval_result,
            "analysis": {}
        }
        
        # Analyze response quality
        debug_info["analysis"]["response_quality"] = await self._analyze_response_quality(
            test_case, response
        )
        
        # Check for common issues
        debug_info["analysis"]["common_issues"] = self._check_common_issues(response)
        
        # Generate improvement suggestions
        debug_info["analysis"]["suggestions"] = await self._generate_improvements(
            test_case, response, eval_result
        )
        
        # Visualize failure (text-based)
        debug_info["visualization"] = self._visualize_failure(
            test_case, response, eval_result
        )
        
        return debug_info
    
    def _check_common_issues(self, response: str) -> List[dict]:
        """Check for common response issues."""
        
        issues = []
        
        # Length issues
        if len(response) < 10:
            issues.append({
                "type": "too_short",
                "severity": "high",
                "description": "Response is very short (< 10 characters)"
            })
        elif len(response) > 5000:
            issues.append({
                "type": "too_long",
                "severity": "medium",
                "description": "Response is very long (> 5000 characters)"
            })
        
        # Format issues
        if response.strip() != response:
            issues.append({
                "type": "whitespace_issues",
                "severity": "low",
                "description": "Response has leading/trailing whitespace"
            })
        
        # Repetition
        sentences = response.split('.')
        if len(sentences) > 3:
            unique_sentences = set(s.strip().lower() for s in sentences)
            if len(unique_sentences) < len(sentences) * 0.7:
                issues.append({
                    "type": "repetitive",
                    "severity": "medium",
                    "description": "Response contains repetitive content"
                })
        
        # Uncertainty markers
        uncertainty_phrases = ["I'm not sure", "I think", "maybe", "possibly", "I guess"]
        uncertainty_count = sum(1 for phrase in uncertainty_phrases if phrase.lower() in response.lower())
        if uncertainty_count > 2:
            issues.append({
                "type": "uncertain",
                "severity": "medium",
                "description": f"Response contains {uncertainty_count} uncertainty markers"
            })
        
        return issues
    
    def _visualize_failure(
        self,
        test_case: dict,
        response: str,
        eval_result: dict
    ) -> str:
        """Create text visualization of evaluation failure."""
        
        lines = [
            "=" * 80,
            "EVALUATION FAILURE DEBUG",
            "=" * 80,
            "",
            f"Test Case: {test_case.get('name', 'Unknown')}",
            f"Category: {test_case.get('category', 'Unknown')}",
            f"Difficulty: {test_case.get('difficulty', 'Unknown')}",
            "",
            "-" * 40,
            "INPUT:",
            "-" * 40,
            test_case.get("input", "")[:500],
            "",
            "-" * 40,
            "EXPECTED OUTPUT:",
            "-" * 40,
            (test_case.get("expected_output", "N/A") or "N/A")[:500],
            "",
            "-" * 40,
            "ACTUAL OUTPUT:",
            "-" * 40,
            response[:500],
            "",
            "-" * 40,
            "EVALUATION RESULT:",
            "-" * 40,
        ]
        
        for metric, result in eval_result.get("dimensions", {}).items():
            score = result.get("score", result.get("normalized", 0))
            status = "✅" if score >= 0.7 else "❌"
            lines.append(f"  {status} {metric}: {score:.2f}")
        
        lines.extend(["", "=" * 80])
        
        return "\n".join(lines)
```

---

## 10. Evaluation in Regulated Industries

### 10.1 Compliance Evaluation Framework

```python
class ComplianceEvaluator:
    """Evaluate LLM systems in regulated industries."""
    
    COMPLIANCE_FRAMEWORKS = {
        "healthcare": {
            "regulations": ["HIPAA", "FDA guidelines", "clinical safety"],
            "required_evaluations": [
                "medical_accuracy",
                "hipaa_compliance",
                "clinical_safety",
                "adverse_event_detection"
            ],
            "minimum_scores": {
                "medical_accuracy": 0.95,
                "hipaa_compliance": 0.99,
                "clinical_safety": 0.98
            }
        },
        "finance": {
            "regulations": ["SEC", "FINRA", "GDPR", "anti-money laundering"],
            "required_evaluations": [
                "financial_accuracy",
                "regulatory_compliance",
                "bias_fairness",
                "audit_trail"
            ],
            "minimum_scores": {
                "financial_accuracy": 0.95,
                "regulatory_compliance": 0.98,
                "bias_fairness": 0.90
            }
        },
        "legal": {
            "regulations": ["attorney-client privilege", "bar rules", "conflict checking"],
            "required_evaluations": [
                "legal_accuracy",
                "privilege_protection",
                "conflict_detection",
                "jurisdiction_awareness"
            ],
            "minimum_scores": {
                "legal_accuracy": 0.90,
                "privilege_protection": 0.99,
                "conflict_detection": 0.95
            }
        }
    }
    
    async def evaluate_compliance(
        self,
        industry: str,
        model_fn,
        test_cases: List[dict]
    ) -> dict:
        """Evaluate LLM compliance for regulated industry."""
        
        framework = self.COMPLIANCE_FRAMEWORKS.get(industry)
        if not framework:
            raise ValueError(f"No compliance framework for industry: {industry}")
        
        results = {}
        
        for eval_type in framework["required_evaluations"]:
            eval_result = await self._run_compliance_eval(
                eval_type, model_fn, test_cases
            )
            results[eval_type] = eval_result
        
        # Check minimum scores
        compliance_status = {}
        for eval_type, min_score in framework["minimum_scores"].items():
            actual_score = results.get(eval_type, {}).get("score", 0)
            compliance_status[eval_type] = {
                "required": min_score,
                "actual": actual_score,
                "passes": actual_score >= min_score,
                "gap": min_score - actual_score
            }
        
        # Overall compliance
        all_pass = all(status["passes"] for status in compliance_status.values())
        
        return {
            "industry": industry,
            "framework": framework,
            "results": results,
            "compliance_status": compliance_status,
            "overall_compliant": all_pass,
            "violations": [
                eval_type for eval_type, status in compliance_status.items()
                if not status["passes"]
            ],
            "audit_trail": self._generate_audit_trail(results)
        }
    
    def _generate_audit_trail(self, results: dict) -> dict:
        """Generate audit trail for compliance."""
        
        return {
            "timestamp": datetime.now().isoformat(),
            "evaluations_performed": list(results.keys()),
            "evaluator_version": "1.0",
            "test_suite_version": "latest",
            "results_summary": {
                eval_type: {
                    "score": result.get("score", 0),
                    "passed": result.get("score", 0) >= 0.9,
                    "details": result.get("details", {})
                }
                for eval_type, result in results.items()
            }
        }
```

---

## Cross-References

| Category | Document | Relevance |
|---|---|---|
| 06-Advanced | 03-Evaluation-Benchmarks.md | Academic benchmark reference |
| 18-Agent-Security | 02-Prompt-Injection-Defenses.md | Security evaluation patterns |
| 20-Agent-Infrastructure | 03-Agent-Tracing-and-Observability.md | Traces for debugging |
| 22-AI-Cybersecurity | 01-Overview.md | Cybersecurity evaluation |
| 33-AI-Native-Software-Dev | 03-AI-Native-CI-CD-and-DevOps.md | CI/CD integration |
| 40-AI-Data-Sovereignty | 01-Overview.md | Data sovereignty evaluation |
| 41-AI-Cost-Optimization | 01-Overview.md | Cost optimization |
| 55-AI-Ethics | 01-Overview.md | Ethical evaluation |
| 56-MLOps | 01-Overview.md | Production ML operations |

---

**See Also:**
- `01-Overview.md` — Introduction to evaluation at scale
- `02-Core-Topics.md` — Essential evaluation topics
- `04-Tools-and-Frameworks.md` — Detailed tool comparisons
- `05-Future-Outlook.md` — Future of AI evaluation

---
**See also:**
- [07 — AI Coding Agent Sustainability: Managing Hidden Costs, Code Quality Decay, and Developer Burnout](33-AI-Native-Software-Development/07-AI-Coding-Agent-Sustainability.md)
- [48 — MCP Cloud Infrastructure & Agent-as-a-Service: The Production Layer of Agentic AI](48-MCP-Cloud-Infrastructure-Agent-as-a-Service/01-Overview.md)
- [03 — Agent-Native Orchestration: LangGraph, Conductor, Mistral Workflows, Mcp-Agent](31-AI-Workflow-Orchestration-and-Durable-Execution/03-Agent-Native-Orchestration.md)
