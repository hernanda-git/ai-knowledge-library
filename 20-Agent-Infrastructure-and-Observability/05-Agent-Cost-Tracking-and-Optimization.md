# 05 — Agent Cost Tracking and Optimization

## 1. Introduction to Agent Cost Management

### 1.1 The Cost Challenge of Agents

Agent systems introduce a fundamental cost challenge: a single user request can trigger orders of magnitude more LLM calls than a traditional chat application. Where a chatbot might make 1–2 LLM calls per query, an agent may make 10–50+ calls including:

- Initial task planning (1–3 calls)
- Tool selection and invocation (2–10 calls per step)
- Observation and analysis (1–2 calls per step)
- Error recovery and retry (1–5 calls per failure)
- Final response synthesis (1–2 calls)
- Internal reasoning / chain-of-thought (1–5 calls)

The cost multiplier is real and significant:

| Scenario | LLM Calls per Task | Typical Cost | Cost Ratio vs Chat |
|----------|-------------------|-------------|-------------------|
| Simple Q&A (chat) | 1–2 | $0.002–0.01 | 1x |
| Simple agent (1 tool) | 3–5 | $0.01–0.03 | 3–5x |
| Research agent (web + analysis) | 10–25 | $0.05–0.25 | 25–50x |
| Coding agent | 20–50 | $0.10–1.00 | 50–200x |
| Multi-agent system | 50–200 | $0.25–5.00 | 100–500x |

At scale (100K tasks/day), even a simple agent can cost $1,000–$3,000/day in LLM API costs alone.

### 1.2 Cost Components

Agent costs break down into multiple categories:

1. **LLM API Costs** (70–90% of total)
   - Input tokens (system prompt, conversation history, tool descriptions, context)
   - Output tokens (responses, reasoning, code generation)
   - Reasoning tokens (internal chain-of-thought)
   - Per-call fixed costs (minimum charges per request)

2. **Tool Execution Costs** (5–20% of total)
   - External API calls (search APIs, data providers, SaaS tools)
   - Compute resources (code execution environments, containers)
   - Database queries (vector DB lookups, SQL queries)
   - Storage (file I/O, blob storage)

3. **Infrastructure Costs** (5–15% of total)
   - Agent runtime compute (containers, serverless functions)
   - Memory/vector store costs
   - Observability pipeline costs
   - Networking and data transfer

## 2. Cost Tracking Architecture

### 2.1 Per-Request Cost Accounting

The foundation of cost tracking is capturing cost at the individual LLM call level:

```python
"""Per-request LLM cost tracking."""

from dataclasses import dataclass
from typing import Dict, Optional
from datetime import datetime

# Model pricing database ($ per 1K tokens)
# Update these as pricing changes
MODEL_PRICING = {
    # OpenAI
    "gpt-4o": {"input": 0.0025, "output": 0.01, "cached_input": 0.00125},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006, "cached_input": 0.000075},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03, "cached_input": 0.005},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015, "cached_input": 0.00025},
    "o1": {"input": 0.015, "output": 0.06, "cached_input": 0.0075},
    "o1-mini": {"input": 0.003, "output": 0.012, "cached_input": 0.0015},
    "o3-mini": {"input": 0.0011, "output": 0.0044, "cached_input": 0.00055},
    # Anthropic
    "claude-3.5-sonnet": {"input": 0.003, "output": 0.015, "cached_input": 0.0003},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125, "cached_input": 0.000025},
    "claude-3-opus": {"input": 0.015, "output": 0.075, "cached_input": 0.0015},
    # Google
    "gemini-2.0-flash": {"input": 0.0001, "output": 0.0004, "cached_input": 0.000025},
    "gemini-2.0-pro": {"input": 0.002, "output": 0.008, "cached_input": 0.0005},
    # Meta / Open-source (approximate hosted cost)
    "llama-3.1-70b": {"input": 0.0009, "output": 0.0009, "cached_input": 0.00045},
    "llama-3.1-405b": {"input": 0.005, "output": 0.005, "cached_input": 0.0025},
    # DeepSeek
    "deepseek-v3": {"input": 0.0005, "output": 0.0015, "cached_input": 0.00005},
    "deepseek-r1": {"input": 0.00055, "output": 0.00219, "cached_input": 0.000055},
}

@dataclass
class LLMCallCost:
    """Cost breakdown for a single LLM call."""
    model: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    
    @property
    def input_cost(self) -> float:
        pricing = MODEL_PRICING.get(self.model, {"input": 0.001, "output": 0.002, "cached_input": 0.0005})
        # Account for cached input discount
        cached_cost = (self.cached_input_tokens / 1000) * pricing["cached_input"]
        fresh_input = max(0, self.input_tokens - self.cached_input_tokens)
        fresh_cost = (fresh_input / 1000) * pricing["input"]
        return cached_cost + fresh_cost
    
    @property
    def output_cost(self) -> float:
        pricing = MODEL_PRICING.get(self.model, {"output": 0.002})
        return (self.output_tokens / 1000) * pricing["output"]
    
    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost
    
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.reasoning_tokens


@dataclass
class AgentSessionCost:
    """Complete cost breakdown for an agent session."""
    session_id: str
    agent_id: str
    agent_version: str
    user_id: str
    task_type: str
    llm_calls: list[LLMCallCost]
    tool_costs: Dict[str, float]  # tool_name -> cost
    infrastructure_cost: float
    start_time: datetime
    end_time: datetime
    
    @property
    def total_llm_cost(self) -> float:
        return sum(call.total_cost for call in self.llm_calls)
    
    @property
    def total_tool_cost(self) -> float:
        return sum(self.tool_costs.values())
    
    @property
    def total_cost(self) -> float:
        return self.total_llm_cost + self.total_tool_cost + self.infrastructure_cost
    
    @property
    def total_tokens(self) -> int:
        return sum(call.total_tokens for call in self.llm_calls)
    
    @property
    def duration_seconds(self) -> float:
        return (self.end_time - self.start_time).total_seconds()
    
    def summary(self) -> dict:
        """Generate a human-readable cost summary."""
        model_breakdown = {}
        for call in self.llm_calls:
            if call.model not in model_breakdown:
                model_breakdown[call.model] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost": 0.0,
                }
            model_breakdown[call.model]["calls"] += 1
            model_breakdown[call.model]["input_tokens"] += call.input_tokens
            model_breakdown[call.model]["output_tokens"] += call.output_tokens
            model_breakdown[call.model]["cost"] += call.total_cost
        
        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "duration_seconds": self.duration_seconds,
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "llm_calls": len(self.llm_calls),
            "llm_cost": self.total_llm_cost,
            "tool_cost": self.total_tool_cost,
            "infrastructure_cost": self.infrastructure_cost,
            "cost_per_second": self.total_cost / max(self.duration_seconds, 0.001),
            "model_breakdown": model_breakdown,
            "tool_breakdown": dict(self.tool_costs),
        }
```

### 2.2 Cost Tracking Service

```python
"""Centralized cost tracking service for agents."""

import json
import time
from datetime import datetime
from typing import Optional
from collections import defaultdict

class AgentCostTracker:
    """
    Tracks agent costs across sessions, users, and models.
    Supports multiple storage backends and provides aggregation.
    """
    
    def __init__(self, storage_backend=None, alert_thresholds: Optional[dict] = None):
        self.storage = storage_backend or InMemoryCostStorage()
        self.alert_thresholds = alert_thresholds or {
            "max_cost_per_session": 1.0,       # Alert if any session costs > $1
            "max_cost_per_user_per_day": 5.0,   # Alert if user exceeds $5/day
            "max_cost_per_agent_per_hour": 10.0, # Alert if agent costs > $10/hour
        }
        self.alerts = []
    
    def record_llm_call(self, session_id: str, user_id: str, agent_id: str,
                        model: str, input_tokens: int, output_tokens: int,
                        cached_input_tokens: int = 0, reasoning_tokens: int = 0) -> LLMCallCost:
        """Record and cost an LLM call."""
        cost = LLMCallCost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            reasoning_tokens=reasoning_tokens,
        )
        
        record = {
            "type": "llm_call",
            "session_id": session_id,
            "user_id": user_id,
            "agent_id": agent_id,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_input_tokens": cached_input_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cost": cost.total_cost,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        self.storage.store(record)
        self._check_alerts(record)
        
        return cost
    
    def record_tool_cost(self, session_id: str, agent_id: str, 
                         tool_name: str, cost: float, metadata: dict = None):
        """Record a tool execution cost."""
        record = {
            "type": "tool_cost",
            "session_id": session_id,
            "agent_id": agent_id,
            "tool_name": tool_name,
            "cost": cost,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.storage.store(record)
    
    def record_infrastructure_cost(self, session_id: str, agent_id: str,
                                    cost: float, resource_type: str = "compute"):
        """Record infrastructure cost (compute, memory, etc.)."""
        record = {
            "type": "infrastructure_cost",
            "session_id": session_id,
            "agent_id": agent_id,
            "resource_type": resource_type,
            "cost": cost,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.storage.store(record)
    
    def get_session_cost(self, session_id: str) -> AgentSessionCost:
        """Get complete cost breakdown for a session."""
        # In production, query from storage
        records = self.storage.get_by_session(session_id)
        # Build the session cost object
        # ...
        return None  # Placeholder
    
    def get_user_cost(self, user_id: str, since: datetime) -> float:
        """Get total cost for a user since a given time."""
        records = self.storage.get_by_user(user_id, since)
        return sum(r.get("cost", 0) for r in records)
    
    def get_agent_cost(self, agent_id: str, since: datetime) -> dict:
        """Get cost aggregation for an agent."""
        records = self.storage.get_by_agent(agent_id, since)
        
        total = sum(r.get("cost", 0) for r in records)
        calls = len([r for r in records if r["type"] == "llm_call"])
        tokens = sum(r.get("input_tokens", 0) + r.get("output_tokens", 0) 
                     for r in records if r["type"] == "llm_call")
        
        return {
            "agent_id": agent_id,
            "total_cost": total,
            "llm_calls": calls,
            "total_tokens": tokens,
            "avg_cost_per_call": total / max(calls, 1),
            "record_count": len(records),
        }
    
    def get_model_breakdown(self, since: datetime) -> dict:
        """Get cost breakdown by model."""
        records = self.storage.get_by_type("llm_call", since)
        breakdown = defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0})
        
        for r in records:
            model = r.get("model", "unknown")
            breakdown[model]["calls"] += 1
            breakdown[model]["input_tokens"] += r.get("input_tokens", 0)
            breakdown[model]["output_tokens"] += r.get("output_tokens", 0)
            breakdown[model]["cost"] += r.get("cost", 0)
        
        return dict(breakdown)
    
    def _check_alerts(self, record: dict):
        """Check cost thresholds and generate alerts."""
        alerts = []
        
        # Per-session cost threshold
        session_cost = self.get_session_cost(record.get("session_id"))
        # Check thresholds in production
        
        for alert in alerts:
            self.alerts.append(alert)


class InMemoryCostStorage:
    """Simple in-memory storage for cost records (use a real DB in production)."""
    
    def __init__(self):
        self.records = []
    
    def store(self, record: dict):
        self.records.append(record)
    
    def get_by_session(self, session_id: str) -> list:
        return [r for r in self.records if r.get("session_id") == session_id]
    
    def get_by_user(self, user_id: str, since: datetime) -> list:
        return [r for r in self.records 
                if r.get("user_id") == user_id 
                and r.get("timestamp", "") >= since.isoformat()]
    
    def get_by_agent(self, agent_id: str, since: datetime) -> list:
        return [r for r in self.records 
                if r.get("agent_id") == agent_id
                and r.get("timestamp", "") >= since.isoformat()]
    
    def get_by_type(self, record_type: str, since: datetime) -> list:
        return [r for r in self.records
                if r.get("type") == record_type
                and r.get("timestamp", "") >= since.isoformat()]
```

### 2.3 Cost Attribution Model

Attributing costs to the right dimensions is essential for understanding where money goes:

```python
"""Multi-dimensional cost attribution for agents."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List

@dataclass
class CostAttribution:
    """Multi-dimensional cost attribution."""
    by_agent: Dict[str, float]           # agent_id -> cost
    by_user: Dict[str, float]            # user_id -> cost
    by_model: Dict[str, float]           # model -> cost
    by_tool: Dict[str, float]            # tool_name -> cost
    by_task_type: Dict[str, float]       # task_type -> cost
    by_hour: Dict[str, float]            # hour -> cost (for time-of-day patterns)
    by_agent_version: Dict[str, float]   # version -> cost
    total: float
    period_start: datetime
    period_end: datetime

def compute_cost_attribution(records: List[dict], 
                              period_start: datetime,
                              period_end: datetime) -> CostAttribution:
    """Compute cost attribution across all dimensions."""
    
    by_agent = defaultdict(float)
    by_user = defaultdict(float)
    by_model = defaultdict(float)
    by_tool = defaultdict(float)
    by_task_type = defaultdict(float)
    by_hour = defaultdict(float)
    by_agent_version = defaultdict(float)
    total = 0.0
    
    for record in records:
        cost = record.get("cost", 0)
        total += cost
        
        by_agent[record.get("agent_id", "unknown")] += cost
        by_user[record.get("user_id", "unknown")] += cost
        by_model[record.get("model", "unknown")] += cost
        by_tool[record.get("tool_name", record.get("type", "unknown"))] += cost
        by_task_type[record.get("task_type", "unknown")] += cost
        by_agent_version[record.get("agent_version", "unknown")] += cost
        
        # Hourly breakdown
        ts = record.get("timestamp", "")
        if ts:
            hour_key = ts[:13]  # "2026-06-13T10"
            by_hour[hour_key] += cost
    
    return CostAttribution(
        by_agent=dict(by_agent),
        by_user=dict(by_user),
        by_model=dict(by_model),
        by_tool=dict(by_tool),
        by_task_type=dict(by_task_type),
        by_hour=dict(by_hour),
        by_agent_version=dict(by_agent_version),
        total=total,
        period_start=period_start,
        period_end=period_end,
    )
```

## 3. Cost Optimization Strategies

### 3.1 Prompt Compression

Reducing the size of prompts (especially system prompts and conversation history) directly reduces token costs:

```python
"""Prompt compression techniques for cost reduction."""

import re
from typing import List

class PromptCompressor:
    """
    Compresses agent prompts to reduce token usage.
    Multiple compression strategies that can be combined.
    """
    
    def __init__(self, compression_strategies: List[str] = None):
        self.strategies = compression_strategies or [
            "remove_comments",
            "shorten_labels",
            "remove_redundant_examples",
            "compress_tool_descriptions",
            "prune_history",
        ]
    
    def compress(self, system_prompt: str, tools: List[dict] = None,
                 history: List[dict] = None, target_reduction: float = 0.3) -> tuple:
        """
        Compress prompt components to achieve target token reduction.
        
        Returns:
            Tuple of (compressed_system_prompt, compressed_tools, compressed_history)
        """
        compressed_system = system_prompt
        compressed_tools = tools or []
        compressed_history = history or []
        
        if "remove_comments" in self.strategies:
            compressed_system = self._remove_comments(compressed_system)
        
        if "shorten_labels" in self.strategies:
            compressed_system = self._shorten_labels(compressed_system)
        
        if "compress_tool_descriptions" in self.strategies and compressed_tools:
            compressed_tools = self._compress_tool_descriptions(compressed_tools)
        
        if "prune_history" in self.strategies and compressed_history:
            compressed_history = self._prune_history(compressed_history)
        
        # Calculate reduction
        original_tokens = self._estimate_tokens(system_prompt, tools, history)
        compressed_tokens = self._estimate_tokens(compressed_system, compressed_tools, compressed_history)
        reduction = 1 - (compressed_tokens / max(original_tokens, 1))
        
        return compressed_system, compressed_tools, compressed_history, reduction
    
    def _remove_comments(self, text: str) -> str:
        """Remove comments and unnecessary whitespace."""
        # Remove markdown comments
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        # Remove single-line comments
        text = re.sub(r'# .*\n?', '', text)
        # Collapse multiple blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    def _shorten_labels(self, text: str) -> str:
        """Shorten instruction labels to minimal form."""
        replacements = {
            r'(?i)instructions': 'Instr.',
            r'(?i)guidelines': 'Gdlns.',
            r'(?i)constraints': 'Constr.',
            r'(?i)requirements': 'Req.',
            r'(?i)important notes': 'Notes',
            r'(?i)please ensure that': 'Ensure',
            r'(?i)you should always': 'Always',
            r'(?i)make sure to': '',
            r'(?i)in order to': 'To',
        }
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        return text
    
    def _compress_tool_descriptions(self, tools: List[dict]) -> List[dict]:
        """Compress tool descriptions by removing redundant parts."""
        compressed = []
        for tool in tools:
            desc = tool.get("description", "")
            # Remove examples from descriptions
            desc = re.sub(r'(?i)examples?.*?(\n|$)', '', desc)
            # Remove usage notes
            desc = re.sub(r'(?i)(usage|note|tip):.*?(\n|$)', '', desc)
            # Keep only first sentence if multiple
            sentences = desc.split('. ')
            if len(sentences) > 2:
                desc = '. '.join(sentences[:2]) + '.'
            
            compressed.append({
                **tool,
                "description": desc[:200],  # Hard limit
            })
        
        return compressed
    
    def _prune_history(self, history: List[dict]) -> List[dict]:
        """Prune conversation history to keep the most relevant parts."""
        if len(history) <= 6:  # Keep recent messages
            return history
        
        # Keep system message, last 2 exchanges, and summary
        system_msgs = [m for m in history if m.get("role") == "system"]
        user_assistant = [m for m in history if m.get("role") in ("user", "assistant", "tool")]
        
        # Keep last 4 messages (2 exchanges) and first message for context
        kept = system_msgs + user_assistant[:1] + user_assistant[-4:]
        return kept
    
    def _estimate_tokens(self, system: str, tools: List[dict], history: List[dict]) -> int:
        """Rough token estimation (4 chars ≈ 1 token)."""
        total = 0
        total += len(system) // 4
        for tool in (tools or []):
            total += len(str(tool)) // 4
        for msg in (history or []):
            total += len(str(msg.get("content", ""))) // 4
        return total
```

### 3.2 Semantic Caching with GPTCache

Caching LLM responses for similar queries can dramatically reduce costs:

```python
"""Semantic caching for agent LLM calls using GPTCache."""

import hashlib
import time
from typing import Optional, Callable, Any
from dataclasses import dataclass

# GPTCache integration
try:
    from gptcache import Cache
    from gptcache.processor.pre import last_content
    from gptcache.embedding import OpenAI as CacheEmbedding
    from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
    from gptcache.adapter import openai as cached_openai
    GPT_CACHE_AVAILABLE = True
except ImportError:
    GPT_CACHE_AVAILABLE = False

@dataclass
class CacheConfig:
    enabled: bool = True
    similarity_threshold: float = 0.85  # Cosine similarity threshold
    max_cache_size: int = 10000
    cache_ttl_seconds: int = 3600  # 1 hour
    exclude_models: list = None  # Models to never cache
    exclude_tools: list = None   # Tool names to never cache results for

class AgentSemanticCache:
    """
    Semantic cache for agent LLM calls.
    Uses GPTCache for embedding-based similarity matching.
    Returns cached responses for semantically similar prompts.
    """
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self.cache = None
        self.stats = {"hits": 0, "misses": 0, "saved_tokens": 0, "saved_cost": 0.0}
        
        if self.config.enabled and GPT_CACHE_AVAILABLE:
            self._init_cache()
    
    def _init_cache(self):
        """Initialize GPTCache with embedding-based similarity."""
        self.cache = Cache()
        self.cache.init(
            pre_embedding_func=last_content,
            embedding_func=CacheEmbedding(),
            similarity_evaluation=SearchDistanceEvaluation(),
        )
    
    def get_or_compute(self, messages: list, model: str, 
                       compute_fn: Callable, **kwargs) -> dict:
        """
        Get cached result or compute and cache.
        
        Args:
            messages: The messages sent to the LLM
            model: The model name
            compute_fn: Function to call if cache miss
            **kwargs: Additional kwargs for compute_fn
        
        Returns:
            LLM response dict (or compatible cached response)
        """
        if not self.config.enabled or not GPT_CACHE_AVAILABLE:
            return compute_fn(messages=messages, model=model, **kwargs)
        
        if self.config.exclude_models and model in self.config.exclude_models:
            return compute_fn(messages=messages, model=model, **kwargs)
        
        # Generate cache key from messages and model
        cache_key = self._make_key(messages, model)
        
        # Try cache
        cached = self.cache.get(cache_key)
        
        if cached is not None:
            self.stats["hits"] += 1
            # Estimate tokens saved
            tokens_saved = sum(len(m.get("content", "")) // 4 for m in messages)
            self.stats["saved_tokens"] += tokens_saved
            # Estimate cost saved (approximate)
            cost_per_token = 0.0025 / 1000  # Rough input cost
            self.stats["saved_cost"] += tokens_saved * cost_per_token
            return cached
        
        self.stats["misses"] += 1
        
        # Compute and cache
        result = compute_fn(messages=messages, model=model, **kwargs)
        self.cache.set(cache_key, result)
        
        return result
    
    def _make_key(self, messages: list, model: str) -> str:
        """Generate a cache key from messages and model."""
        content = json.dumps([
            {"role": m.get("role"), "content": m.get("content", "")}
            for m in messages
        ])
        return hashlib.sha256(f"{content}:{model}".encode()).hexdigest()
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self.stats["hits"] + self.stats["misses"]
        return {
            **self.stats,
            "hit_rate": self.stats["hits"] / max(total, 1),
            "miss_rate": self.stats["misses"] / max(total, 1),
        }
    
    def clear(self):
        """Clear the cache."""
        if self.cache:
            self.cache.flush()


# Example: Using Cache in an Agent
class CachedAgent:
    """Agent that uses semantic caching for LLM calls."""
    
    def __init__(self, llm_client, tools: dict):
        self.llm = llm_client
        self.tools = tools
        self.cache = AgentSemanticCache(CacheConfig(
            similarity_threshold=0.9,
            exclude_models=["gpt-4o"],  # Don't cache GPT-4o (high stakes)
        ))
    
    def think(self, messages: list) -> str:
        """LLM call with caching for non-critical thinking steps."""
        result = self.cache.get_or_compute(
            messages=messages,
            model="gpt-4o-mini",  # Use cheaper model for routine thinking
            compute_fn=self.llm.chat,
            temperature=0.7,
        )
        return result.content
    
    def critical_decision(self, messages: list) -> str:
        """Critical decision always uses uncached GPT-4o."""
        return self.llm.chat(
            messages=messages,
            model="gpt-4o",
            temperature=0.1,
        ).content
```

### 3.3 Model Routing

Using cheaper models for routine operations and expensive models for critical decisions:

```python
"""Intelligent model routing for cost optimization."""

from typing import Optional, Callable
import re

class ModelRouter:
    """
    Routes agent operations to the appropriate LLM model based on
    task complexity, criticality, and cost constraints.
    """
    
    def __init__(self, llm_clients: dict):
        """
        Args:
            llm_clients: dict of model_name -> client
        """
        self.clients = llm_clients
        
        # Routing rules
        self.routes = [
            # (condition_fn, model, weight)
            self._route_simple,      # Simple questions -> cheap model
            self._route_complex,     # Complex reasoning -> expensive model
            self._route_code,        # Code generation -> code-optimized model
            self._route_tool_call,   # Tool selection -> balanced model
            self._route_critical,    # Critical decisions -> most capable model
        ]
    
    def get_optimal_model(self, messages: list, task_type: str = "general",
                          critical: bool = False) -> str:
        """Determine the best model for a given task."""
        if critical:
            return "gpt-4o"
        
        task_content = " ".join(m.get("content", "") for m in messages[-3:])
        
        for router in self.routes:
            result = router(task_content, task_type)
            if result:
                return result
        
        return "gpt-4o-mini"  # Default to cheap model
    
    def _route_simple(self, content: str, task_type: str) -> Optional[str]:
        """Route simple factual queries to cheap model."""
        # Check if it's a simple Q&A
        simple_patterns = [
            r'^(what|who|when|where)\s+(is|was|are|were)',
            r'^(how\s+(many|much|long|far))',
            r'^(yes|no|maybe|true|false)',
            r'^(define|explain|summarize)\s',
        ]
        
        content_lower = content.lower()
        if len(content_lower) < 100 and any(
            re.match(p, content_lower) for p in simple_patterns
        ):
            return "gpt-4o-mini"
        
        return None
    
    def _route_complex(self, content: str, task_type: str) -> Optional[str]:
        """Route complex multi-step reasoning to capable model."""
        complexity_signals = [
            r'step by step',
            r'compare and contrast',
            r'analyze',
            r'evaluate',
            r'synthesize',
            r'recommend',
        ]
        
        content_lower = content.lower()
        signal_count = sum(
            1 for s in complexity_signals if s in content_lower
        )
        
        if signal_count >= 2 or len(content_lower) > 500:
            return "gpt-4o"
        
        return None
    
    def _route_code(self, content: str, task_type: str) -> Optional[str]:
        """Route code generation to code-optimized model."""
        if task_type == "code" or any(
            keyword in content.lower() 
            for keyword in ["write code", "implement", "function", "class",
                           "algorithm", "debug", "fix bug", "refactor"]
        ):
            # Check complexity
            if "```" in content or len(content) > 200:
                return "gpt-4o"
            return "gpt-4o-mini"
        return None
    
    def _route_tool_call(self, content: str, task_type: str) -> Optional[str]:
        """Route tool selection decisions."""
        if task_type in ("tool_selection", "tool_call"):
            return "gpt-4o-mini"  # Tool selection is usually straightforward
        return None
    
    def _route_critical(self, content: str, task_type: str) -> Optional[str]:
        """Route critical decisions to most capable model."""
        critical_signals = [
            r'financial', r'transaction', r'payment',
            r'legal', r'contract', r'compliance',
            r'medical', r'health', r'diagnosis',
            r'security', r'authentication',
            r'deploy', r'release', r'production',
        ]
        
        if any(re.search(s, content, re.IGNORECASE) for s in critical_signals):
            return "gpt-4o"
        
        return None
    
    def route(self, messages: list, task_type: str = "general",
              critical: bool = False, fallback_models: list = None) -> tuple:
        """
        Route to the best model and return (model_name, client).
        
        Returns:
            Tuple of (model_name, client)
        """
        model = self.get_optimal_model(messages, task_type, critical)
        
        if model in self.clients:
            return model, self.clients[model]
        
        # Fallback chain
        for fallback in (fallback_models or ["gpt-4o", "gpt-4o-mini"]):
            if fallback in self.clients:
                return fallback, self.clients[fallback]
        
        raise ValueError(f"No LLM client available for model {model}")
```

### 3.4 Cost Optimization Pipeline

Combining all optimization strategies into a coherent pipeline:

```python
"""Complete cost optimization pipeline for agents."""

import time
from typing import Optional

class CostOptimizationPipeline:
    """
    Combines multiple cost optimization strategies for agent LLM calls.
    Order: Cache check → Model routing → Prompt compression → LLM call → Cost recording
    """
    
    def __init__(
        self,
        llm_clients: dict,
        cache: Optional[AgentSemanticCache] = None,
        router: Optional[ModelRouter] = None,
        compressor: Optional[PromptCompressor] = None,
        cost_tracker: Optional[AgentCostTracker] = None,
    ):
        self.cache = cache or AgentSemanticCache()
        self.router = router or ModelRouter(llm_clients)
        self.compressor = compressor or PromptCompressor()
        self.cost_tracker = cost_tracker
        self.total_savings = 0.0
    
    def execute_llm_call(
        self,
        messages: list,
        task_type: str = "general",
        critical: bool = False,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> dict:
        """
        Execute an optimized LLM call with caching, routing, and compression.
        
        Returns:
            dict with 'content', 'model', 'tokens', 'cost', 'savings' keys
        """
        start_time = time.time()
        
        # Step 1: Model routing
        model, client = self.router.route(messages, task_type, critical)
        original_model = model
        
        # Step 2: Cache check (only for non-critical, not for best model)
        if not critical and model != "gpt-4o":
            cached = self.cache.get_or_compute(
                messages=messages,
                model=model,
                compute_fn=client.chat,
            )
            
            if cached and cached.get("from_cache"):
                savings = self._estimate_savings(messages, model, cached)
                self.total_savings += savings
                
                return {
                    "content": cached["content"],
                    "model": model,
                    "from_cache": True,
                    "savings": savings,
                    "duration_ms": (time.time() - start_time) * 1000,  # Cache is fast
                }
        
        # Step 3: Prompt compression (for high-volume, non-critical)
        if not critical and len(str(messages)) > 2000:
            compressed_system = None
            if len(messages) > 0 and messages[0].get("role") == "system":
                compressed_system, _, history, reduction = self.compressor.compress(
                    system_prompt=messages[0]["content"],
                    history=messages[1:],
                    target_reduction=0.2,
                )
                if reduction > 0.1:
                    messages[0]["content"] = compressed_system
        
        # Step 4: Execute LLM call
        response = client.chat(messages=messages, model=model)
        
        # Step 5: Record cost
        if self.cost_tracker:
            usage = getattr(response, 'usage', None)
            if usage:
                self.cost_tracker.record_llm_call(
                    session_id=session_id,
                    user_id=user_id,
                    agent_id=agent_id,
                    model=model,
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                )
        
        return {
            "content": response.choices[0].message.content,
            "model": model,
            "from_cache": False,
            "savings": 0.0,
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
            "duration_ms": (time.time() - start_time) * 1000,
            "cost": 0.0,  # Will be calculated by cost_tracker
        }
    
    def _estimate_savings(self, messages: list, model: str, cached: dict) -> float:
        """Estimate cost savings from cache hit."""
        total_input_tokens = sum(len(m.get("content", "")) // 4 for m in messages)
        # Assume we saved the input tokens + some output
        pricing = MODEL_PRICING.get(model, {"input": 0.001, "output": 0.002})
        saved_input_cost = (total_input_tokens / 1000) * pricing["input"]
        saved_output_cost = (100 / 1000) * pricing["output"]  # Assume ~100 output tokens
        return saved_input_cost + saved_output_cost
```

### 3.5 Batch Processing and Speculative Execution

```python
"""Batch processing and speculative execution for cost efficiency."""

import asyncio
from typing import List, Callable, Any

class BatchProcessor:
    """
    Batch multiple independent LLM calls into a single request to reduce
    per-call overhead and enable parallel processing.
    """
    
    async def batch_llm_calls(self, calls: List[dict], llm_client, 
                               model: str, max_batch_size: int = 5) -> List[str]:
        """
        Batch multiple independent LLM calls into one request.
        
        Works best for:
        - Evaluating multiple candidate responses
        - Scoring/ranking multiple options
        - Simple parallel tool calls
        """
        results = []
        
        for i in range(0, len(calls), max_batch_size):
            batch = calls[i:i + max_batch_size]
            
            # Create a batched prompt
            batched_messages = [{
                "role": "system",
                "content": "You will receive multiple independent requests. "
                           "Respond to each one separately, numbering your responses."
            }]
            
            for j, call in enumerate(batch):
                batched_messages.append({
                    "role": "user",
                    "content": f"Request {j+1}: {call['input']}"
                })
            
            # Single LLM call for the batch
            response = await llm_client.chat(
                messages=batched_messages,
                model=model,
            )
            
            # Parse batched response into individual responses
            batch_results = self._parse_batched_response(
                response.content, len(batch)
            )
            results.extend(batch_results)
        
        return results
    
    def _parse_batched_response(self, content: str, expected_count: int) -> List[str]:
        """Parse a batched response into individual responses."""
        results = []
        current = []
        
        for line in content.split('\n'):
            for i in range(1, expected_count + 1):
                if line.strip().startswith(f"{i}.") or line.strip().startswith(f"**{i}.**"):
                    if current:
                        results.append('\n'.join(current))
                    current = [line]
                    break
            else:
                if current:
                    current.append(line)
        
        if current:
            results.append('\n'.join(current))
        
        # Pad if we got fewer results than expected
        while len(results) < expected_count:
            results.append("")
        
        return results[:expected_count]
```

## 4. Cost Dashboard

### 4.1 Grafana Dashboard Template

Below is a Grafana dashboard JSON model for agent cost monitoring. Save this as a JSON model and import into Grafana.

```json
{
  "dashboard": {
    "title": "Agent Cost Dashboard",
    "tags": ["agents", "cost", "observability"],
    "time": { "from": "now-24h", "to": "now" },
    "panels": [
      {
        "title": "Total Cost (24h)",
        "type": "stat",
        "datasource": "Prometheus",
        "targets": [{
          "expr": "sum(agent_llm_cost_total[24h])",
          "legendFormat": "Total LLM Cost"
        }],
        "options": {
          "colorMode": "value",
          "graphMode": "area",
          "justifyMode": "center"
        }
      },
      {
        "title": "Cost by Model",
        "type": "piechart",
        "datasource": "Prometheus",
        "targets": [{
          "expr": "sum(agent_llm_cost_total) by (model)"
        }]
      },
      {
        "title": "Cost by Agent",
        "type": "barchart",
        "datasource": "Prometheus",
        "targets": [{
          "expr": "topk(10, sum(agent_llm_cost_total) by (agent_id))"
        }]
      },
      {
        "title": "Cost by User (Top 10)",
        "type": "table",
        "datasource": "Prometheus",
        "targets": [{
          "expr": "topk(10, sum(agent_llm_cost_total) by (user_id))"
        }]
      },
      {
        "title": "Cost per Session Distribution",
        "type": "heatmap",
        "datasource": "Prometheus",
        "targets": [{
          "expr": "agent_session_cost"
        }]
      },
      {
        "title": "Daily Cost Trend",
        "type": "timeseries",
        "datasource": "Prometheus",
        "targets": [{
          "expr": "sum(rate(agent_llm_cost_total[1d]))",
          "legendFormat": "Daily cost rate"
        }]
      },
      {
        "title": "Cache Hit Rate",
        "type": "gauge",
        "datasource": "Prometheus",
        "targets": [{
          "expr": "agent_cache_hit_rate"
        }],
        "options": {
          "min": 0,
          "max": 1,
          "thresholds": [
            { "value": 0.5, "color": "red" },
            { "value": 0.7, "color": "yellow" },
            { "value": 0.9, "color": "green" }
          ]
        }
      }
    ]
  }
}
```

### 4.2 Prometheus Cost Metrics Export

```python
"""Prometheus metrics for agent cost tracking."""

from prometheus_client import Counter, Histogram, Gauge, Summary
import time

# Cost metrics
AGENT_LLM_COST = Counter(
    'agent_llm_cost_total',
    'Total LLM API cost in USD',
    ['agent_id', 'model', 'user_id']
)

AGENT_TOOL_COST = Counter(
    'agent_tool_cost_total',
    'Total tool execution cost in USD',
    ['agent_id', 'tool']
)

AGENT_SESSION_COST = Histogram(
    'agent_session_cost',
    'Cost per agent session in USD',
    ['agent_id'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

AGENT_TOKENS_PER_CALL = Histogram(
    'agent_tokens_per_llm_call',
    'Tokens per LLM call',
    ['model'],
    buckets=[100, 500, 1000, 2000, 4000, 8000, 16000, 32000]
)

AGENT_CACHE_HITS = Counter(
    'agent_cache_hits_total',
    'Total semantic cache hits',
    ['agent_id']
)

AGENT_CACHE_MISSES = Counter(
    'agent_cache_misses_total',
    'Total semantic cache misses',
    ['agent_id']
)

# Cost-saving metrics
AGENT_COST_SAVED = Counter(
    'agent_cost_saved_total',
    'Total cost saved via optimization in USD',
    ['strategy']  # strategy: caching, model_routing, compression
)

def record_llm_cost(agent_id: str, model: str, user_id: str, input_tokens: int, output_tokens: int):
    """Record LLM call cost to Prometheus."""
    pricing = MODEL_PRICING.get(model, {"input": 0.001, "output": 0.002})
    cost = (input_tokens / 1000) * pricing["input"] + (output_tokens / 1000) * pricing["output"]
    AGENT_LLM_COST.labels(agent_id=agent_id, model=model, user_id=user_id).inc(cost)
    AGENT_TOKENS_PER_CALL.labels(model=model).observe(input_tokens + output_tokens)

def record_session_cost(agent_id: str, cost: float):
    AGENT_SESSION_COST.labels(agent_id=agent_id).observe(cost)
```

## 5. Cost Alerting

### 5.1 Alert Rules

```python
"""Cost-based alerting rules for agents."""

from datetime import datetime, timedelta
from typing import Callable

class CostAlertManager:
    """
    Manages cost-based alerts for agent systems.
    Supports per-user, per-agent, and global thresholds.
    """
    
    def __init__(self, notification_fn: Callable):
        self.notification_fn = notification_fn  # Slack, email, PagerDuty, etc.
        self.alerts_enabled = True
    
    def check_session_cost(self, session_cost: float, threshold: float = 1.0):
        """Alert if a single session exceeds cost threshold."""
        if session_cost > threshold and self.alerts_enabled:
            self.notification_fn({
                "type": "session_cost_exceeded",
                "severity": "warning",
                "message": f"Session cost ${session_cost:.2f} exceeds ${threshold:.2f} threshold",
                "timestamp": datetime.utcnow().isoformat(),
            })
    
    def check_daily_user_cost(self, user_id: str, daily_cost: float, 
                                threshold: float = 10.0):
        """Alert if a user's daily cost exceeds threshold."""
        if daily_cost > threshold:
            self.notification_fn({
                "type": "daily_user_cost_exceeded",
                "severity": "warning",
                "message": f"User {user_id} daily cost ${daily_cost:.2f} exceeds ${threshold:.2f}",
                "timestamp": datetime.utcnow().isoformat(),
            })
    
    def check_anomalous_spike(self, current_hour_cost: float, 
                               historical_avg_hourly: float,
                               multiplier: float = 3.0):
        """Alert on anomalous cost spikes."""
        if current_hour_cost > historical_avg_hourly * multiplier:
            self.notification_fn({
                "type": "cost_spike",
                "severity": "critical",
                "message": f"Cost spike: ${current_hour_cost:.2f}/hour vs "
                           f"${historical_avg_hourly:.2f}/hour average (x{multiplier})",
                "timestamp": datetime.utcnow().isoformat(),
            })
```

## 6. Conclusion

Cost management for agent systems requires a multi-layered approach:

1. **Track everything** — every LLM call, every tool invocation, every infrastructure cost
2. **Attribute costs** — by agent, user, session, model, and task type
3. **Optimize aggressively** — semantic caching, model routing, prompt compression
4. **Alert on anomalies** — cost spikes, budget thresholds, inefficient patterns
5. **Review regularly** — model pricing changes, new optimization techniques, usage patterns

The typical cost optimization journey:
- **Phase 1** (2–4 weeks): Implement cost tracking and basic alerts → 10–20% savings from visibility
- **Phase 2** (4–8 weeks): Add model routing and prompt compression → 20–40% additional savings
- **Phase 3** (8–12 weeks): Implement semantic caching and speculative execution → 30–60% additional savings

With all optimizations in place, most agent systems can reduce costs by 50–80% compared to naive implementations.

---

*Next: [06-Agent-Logging-and-Monitoring.md](06-Agent-Logging-and-Monitoring.md) — Structured logging, metrics, and alerting for production agents.*

---
**See also:**
- [07 — AI Coding Agent Sustainability: Managing Hidden Costs, Code Quality Decay, and Developer Burnout](33-AI-Native-Software-Development/07-AI-Coding-Agent-Sustainability.md)
- [48 — MCP Cloud Infrastructure & Agent-as-a-Service: The Production Layer of Agentic AI](48-MCP-Cloud-Infrastructure-Agent-as-a-Service/01-Overview.md)
- [03 — Agent-Native Orchestration: LangGraph, Conductor, Mistral Workflows, Mcp-Agent](31-AI-Workflow-Orchestration-and-Durable-Execution/03-Agent-Native-Orchestration.md)
