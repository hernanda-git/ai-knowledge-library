# Technical Deep Dive: AI Agent Financial Governance and Cost Control

> Advanced implementation architectures, middleware patterns, and engineering details for building production-grade cost control systems for AI agents. This document covers the low-level technical components that power agent financial governance.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Cost Interceptor Middleware](#2-cost-interceptor-middleware)
3. [Token Counting and Estimation](#3-token-counting-and-estimation)
4. [Budget State Management](#4-budget-state-management)
5. [Real-Time Cost Stream Processing](#5-real-time-cost-stream-processing)
6. [Circuit Breaker Implementations](#6-circuit-breaker-implementations)
7. [Cost-Aware Agent Orchestration](#7-cost-aware-agent-orchestration)
8. [Storage and Persistence](#8-storage-and-persistence)
9. [API Design for Cost Control](#9-api-design-for-cost-control)
10. [Testing Cost Control Systems](#10-testing-cost-control-systems)

---

## 1. Architecture Overview

### 1.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            AGENT RUNTIME                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Agent      │  │   Agent      │  │   Agent      │  │   Agent      │  │
│  │   Instance 1 │  │   Instance 2 │  │   Instance 3 │  │   Instance N │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
├─────────┼──────────────────┼──────────────────┼──────────────────┼───────────┤
│         │            COST CONTROL LAYER       │                  │           │
│  ┌──────┴──────────────────┴──────────────────┴──────────────────┴───────┐  │
│  │                     COST INTERCEPTOR MIDDLEWARE                        │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐     │  │
│  │  │   Token    │  │   Budget   │  │   Model    │  │   Circuit  │     │  │
│  │  │   Counter  │  │   Checker  │  │   Router   │  │   Breaker  │     │  │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────────┤
│                           DATA LAYER                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Budget     │  │   Cost       │  │   Usage      │  │   Audit      │  │
│  │   Store      │  │   Stream     │  │   Database   │  │   Log        │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Request Flow

```
1. Agent initiates action (inference call, tool execution)
          │
          ▼
2. Cost Interceptor Middleware intercepts the call
          │
          ▼
3. Token Counter estimates cost
          │
          ▼
4. Budget Checker validates against budget
          │  ├─ Approved → Continue to execution
          │  └─ Denied → Return budget exceeded error
          ▼
5. Model Router selects optimal model (if applicable)
          │
          ▼
6. Circuit Breaker checks for anomalies
          │  ├─ Normal → Execute action
          │  └─ Tripped → Apply intervention
          ▼
7. Action executes
          │
          ▼
8. Cost Tracker records actual cost
          │
          ▼
9. Budget State updated
          │
          ▼
10. Metrics emitted to monitoring
```

---

## 2. Cost Interceptor Middleware

### 2.1 Core Interceptor Implementation

```python
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import time
import asyncio

class ActionType(Enum):
    INFERENCE = "inference"
    TOOL_CALL = "tool_call"
    SUB_AGENT = "sub_agent"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"

@dataclass
class CostEstimate:
    action_type: ActionType
    estimated_cost: float
    confidence: float  # 0.0 to 1.0
    model: Optional[str] = None
    tokens: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class CostCheckResult:
    allowed: bool
    reason: Optional[str] = None
    estimated_cost: float = 0.0
    remaining_budget: float = 0.0
    recommended_action: Optional[str] = None
    alternative_model: Optional[str] = None

class CostInterceptorMiddleware:
    """Core middleware for intercepting and controlling agent costs."""
    
    def __init__(
        self,
        budget_manager: 'BudgetManager',
        token_counter: 'TokenCounter',
        model_router: 'ModelRouter',
        circuit_breaker: 'CircuitBreaker',
        cost_tracker: 'CostTracker'
    ):
        self.budget_manager = budget_manager
        self.token_counter = token_counter
        self.model_router = model_router
        self.circuit_breaker = circuit_breaker
        self.cost_tracker = cost_tracker
        self.hooks = {
            "before_check": [],
            "after_check": [],
            "on_denied": [],
            "on_approved": [],
            "after_execution": []
        }
    
    async def intercept_inference(
        self,
        agent_id: str,
        session_id: str,
        model: str,
        messages: list,
        max_tokens: int
    ) -> CostCheckResult:
        """Intercept an inference call."""
        
        # Run pre-check hooks
        await self._run_hooks("before_check", {
            "action_type": ActionType.INFERENCE,
            "agent_id": agent_id,
            "model": model
        })
        
        # Estimate tokens
        estimated_input = self.token_counter.estimate_input_tokens(messages)
        estimated_output = min(max_tokens, estimated_input * 0.3)
        
        # Estimate cost
        cost_estimate = self._estimate_inference_cost(
            model, estimated_input, estimated_output
        )
        
        # Check budget
        budget_check = await self.budget_manager.check_budget(
            agent_id=agent_id,
            amount=cost_estimate.estimated_cost,
            category="inference"
        )
        
        if not budget_check.allowed:
            # Try to find alternative model
            alternative = await self.model_router.find_alternative(
                agent_id=agent_id,
                max_cost=self.budget_manager.get_remaining(agent_id),
                task_requirements={"input_tokens": estimated_input}
            )
            
            result = CostCheckResult(
                allowed=False,
                reason=budget_check.reason,
                estimated_cost=cost_estimate.estimated_cost,
                remaining_budget=self.budget_manager.get_remaining(agent_id),
                recommended_action="use_alternative_model",
                alternative_model=alternative.model if alternative else None
            )
            
            await self._run_hooks("on_denied", result)
            return result
        
        # Check circuit breaker
        cb_result = self.circuit_breaker.check(
            agent_id=agent_id,
            cost=cost_estimate.estimated_cost,
            action_type=ActionType.INFERENCE
        )
        
        if cb_result.blocked:
            return CostCheckResult(
                allowed=False,
                reason=cb_result.reason,
                estimated_cost=cost_estimate.estimated_cost,
                recommended_action=cb_result.action
            )
        
        # Approve
        result = CostCheckResult(
            allowed=True,
            estimated_cost=cost_estimate.estimated_cost,
            remaining_budget=self.budget_manager.get_remaining(agent_id)
        )
        
        await self._run_hooks("on_approved", result)
        return result
    
    async def record_actual_cost(
        self,
        agent_id: str,
        session_id: str,
        action_type: ActionType,
        actual_cost: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record actual cost after execution."""
        
        # Record in budget manager
        await self.budget_manager.record_spend(
            agent_id=agent_id,
            amount=actual_cost,
            category=action_type.value
        )
        
        # Record in cost tracker
        await self.cost_tracker.record(
            agent_id=agent_id,
            session_id=session_id,
            action_type=action_type,
            cost=actual_cost,
            metadata=metadata
        )
        
        # Update circuit breaker
        self.circuit_breaker.record_action(
            agent_id=agent_id,
            cost=actual_cost,
            action_type=action_type
        )
        
        # Run post-execution hooks
        await self._run_hooks("after_execution", {
            "agent_id": agent_id,
            "action_type": action_type,
            "cost": actual_cost
        })
    
    def _estimate_inference_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> CostEstimate:
        """Estimate inference cost."""
        pricing = self._get_model_pricing(model)
        
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost
        
        return CostEstimate(
            action_type=ActionType.INFERENCE,
            estimated_cost=total_cost,
            confidence=0.9,  # High confidence for known models
            model=model,
            tokens={"input": input_tokens, "output": output_tokens}
        )
    
    async def _run_hooks(self, hook_name: str, data: Any):
        """Run registered hooks."""
        for hook in self.hooks.get(hook_name, []):
            await hook(data)
```

### 2.2 Interceptor Registration

```python
class InterceptorRegistry:
    """Manages multiple cost interceptors."""
    
    def __init__(self):
        self.interceptors: Dict[str, CostInterceptorMiddleware] = {}
        self.middleware_chain: List[str] = []
    
    def register(self, name: str, interceptor: CostInterceptorMiddleware):
        """Register a cost interceptor."""
        self.interceptors[name] = interceptor
        self.middleware_chain.append(name)
    
    async def process(
        self,
        agent_id: str,
        action_type: ActionType,
        action_data: Dict[str, Any]
    ) -> CostCheckResult:
        """Process action through interceptor chain."""
        
        for interceptor_name in self.middleware_chain:
            interceptor = self.interceptors[interceptor_name]
            
            result = await interceptor.intercept(
                agent_id=agent_id,
                action_type=action_type,
                **action_data
            )
            
            if not result.allowed:
                return result
        
        return CostCheckResult(allowed=True)
```

---

## 3. Token Counting and Estimation

### 3.1 Token Estimation Engine

```python
import tiktoken
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class TokenEstimate:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost: float
    model: str
    confidence: float

class TokenEstimator:
    """Estimates token counts for various inputs."""
    
    # Model to encoding mapping
    MODEL_ENCODINGS = {
        "gpt-4o": "o200k_base",
        "gpt-4o-mini": "o200k_base",
        "o1-preview": "o200k_base",
        "o1-mini": "o200k_base",
        "claude-3-opus": "cl100k_base",
        "claude-3.5-sonnet": "cl100k_base",
        "claude-3-haiku": "cl100k_base",
        "gemini-1.5-pro": "cl100k_base",
        "gemini-1.5-flash": "cl100k_base",
    }
    
    # Pricing per 1M tokens (USD)
    MODEL_PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "o1-preview": {"input": 15.00, "output": 60.00},
        "o1-mini": {"input": 3.00, "output": 12.00},
        "claude-3-opus": {"input": 15.00, "output": 75.00},
        "claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
        "claude-3-haiku": {"input": 0.25, "output": 1.25},
        "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    }
    
    def __init__(self):
        self.encodings = {}
    
    def _get_encoding(self, model: str):
        """Get tokenizer encoding for model."""
        if model not in self.encodings:
            encoding_name = self.MODEL_ENCODINGS.get(model, "cl100k_base")
            self.encodings[model] = tiktoken.get_encoding(encoding_name)
        return self.encodings[model]
    
    def count_tokens(self, text: str, model: str) -> int:
        """Count tokens in text."""
        encoding = self._get_encoding(model)
        return len(encoding.encode(text))
    
    def estimate_messages_tokens(
        self,
        messages: List[Dict[str, str]],
        model: str
    ) -> int:
        """Estimate tokens for message list."""
        encoding = self._get_encoding(model)
        
        total_tokens = 0
        for message in messages:
            # Each message has overhead
            total_tokens += 4  # Message framing
            
            for key, value in message.items():
                if isinstance(value, str):
                    total_tokens += len(encoding.encode(value))
        
        # Conversation overhead
        total_tokens += 2  # Reply priming
        
        return total_tokens
    
    def estimate_inference_cost(
        self,
        model: str,
        input_tokens: int,
        max_output_tokens: int
    ) -> TokenEstimate:
        """Estimate inference cost."""
        pricing = self.MODEL_PRICING.get(model, {"input": 1.0, "output": 3.0})
        
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (max_output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost
        
        return TokenEstimate(
            input_tokens=input_tokens,
            output_tokens=max_output_tokens,
            total_tokens=input_tokens + max_output_tokens,
            estimated_cost=total_cost,
            model=model,
            confidence=0.85
        )
    
    def find_optimal_model(
        self,
        input_tokens: int,
        max_output_tokens: int,
        budget_remaining: float,
        quality_requirement: str = "standard"
    ) -> List[Dict]:
        """Find models that fit within budget."""
        candidates = []
        
        for model, pricing in self.MODEL_PRICING.items():
            estimate = self.estimate_inference_cost(
                model, input_tokens, max_output_tokens
            )
            
            if estimate.estimated_cost <= budget_remaining:
                quality_score = self._get_quality_score(model, quality_requirement)
                
                candidates.append({
                    "model": model,
                    "estimated_cost": estimate.estimated_cost,
                    "quality_score": quality_score,
                    "cost_effectiveness": quality_score / estimate.estimated_cost
                })
        
        # Sort by cost effectiveness
        candidates.sort(key=lambda x: x["cost_effectiveness"], reverse=True)
        
        return candidates
    
    def _get_quality_score(self, model: str, requirement: str) -> float:
        """Get quality score for model."""
        quality_scores = {
            "gpt-4o": 0.95,
            "gpt-4o-mini": 0.75,
            "o1-preview": 0.98,
            "o1-mini": 0.85,
            "claude-3-opus": 0.97,
            "claude-3.5-sonnet": 0.92,
            "claude-3-haiku": 0.70,
            "gemini-1.5-pro": 0.90,
            "gemini-1.5-flash": 0.72,
        }
        
        base_score = quality_scores.get(model, 0.5)
        
        multiplier = {
            "basic": 0.8,
            "standard": 1.0,
            "high": 1.1,
            "critical": 1.2
        }.get(requirement, 1.0)
        
        return base_score * multiplier
```

### 3.2 Streaming Token Counter

```python
import asyncio
from typing import AsyncIterator

class StreamingTokenCounter:
    """Counts tokens in streaming responses."""
    
    def __init__(self, model: str):
        self.model = model
        self.encoding = tiktoken.get_encoding(
            MODEL_ENCODINGS.get(model, "cl100k_base")
        )
        self.input_tokens = 0
        self.output_tokens = 0
        self.buffer = ""
    
    async def count_stream(
        self,
        stream: AsyncIterator[str],
        max_tokens: int
    ) -> AsyncIterator[str]:
        """Count tokens while streaming."""
        async for chunk in stream:
            self.buffer += chunk
            
            # Count complete tokens
            tokens = self.encoding.encode(self.buffer)
            
            if len(tokens) > self.output_tokens:
                new_tokens = len(tokens) - self.output_tokens
                self.output_tokens = len(tokens)
                
                # Check against limit
                if self.output_tokens > max_tokens:
                    raise TokenLimitExceeded(
                        f"Output tokens {self.output_tokens} exceeds limit {max_tokens}"
                    )
            
            yield chunk
    
    def get_total_cost(self) -> float:
        """Get total cost of streamed response."""
        pricing = MODEL_PRICING.get(self.model, {"input": 1.0, "output": 3.0})
        
        input_cost = (self.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.output_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost
```

---

## 4. Budget State Management

### 4.1 In-Memory Budget Store

```python
from datetime import datetime, timedelta
from typing import Dict, Optional
from dataclasses import dataclass, field
import threading

@dataclass
class BudgetState:
    agent_id: str
    total_budget: float
    spent: float = 0.0
    category_spent: Dict[str, float] = field(default_factory=dict)
    session_spent: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    alerts_sent: Dict[str, datetime] = field(default_factory=dict)
    
    @property
    def remaining(self) -> float:
        return self.total_budget - self.spent
    
    @property
    def utilization_pct(self) -> float:
        return (self.spent / self.total_budget * 100) if self.total_budget > 0 else 0
    
    def can_spend(self, amount: float) -> bool:
        return self.spent + amount <= self.total_budget
    
    def record_spend(self, amount: float, category: str):
        self.spent += amount
        self.category_spent[category] = self.category_spent.get(category, 0) + amount
        self.session_spent += amount
        self.last_updated = datetime.now()

class InMemoryBudgetStore:
    """Thread-safe in-memory budget store."""
    
    def __init__(self):
        self.budgets: Dict[str, BudgetState] = {}
        self.lock = threading.RLock()
    
    def get_budget(self, agent_id: str) -> Optional[BudgetState]:
        """Get budget state for agent."""
        with self.lock:
            return self.budgets.get(agent_id)
    
    def set_budget(self, agent_id: str, total_budget: float):
        """Set budget for agent."""
        with self.lock:
            if agent_id in self.budgets:
                self.budgets[agent_id].total_budget = total_budget
            else:
                self.budgets[agent_id] = BudgetState(
                    agent_id=agent_id,
                    total_budget=total_budget
                )
    
    def record_spend(self, agent_id: str, amount: float, category: str):
        """Record spending for agent."""
        with self.lock:
            if agent_id not in self.budgets:
                raise ValueError(f"No budget found for agent {agent_id}")
            
            self.budgets[agent_id].record_spend(amount, category)
    
    def check_budget(self, agent_id: str, amount: float) -> bool:
        """Check if spending is within budget."""
        with self.lock:
            budget = self.budgets.get(agent_id)
            if not budget:
                return False
            return budget.can_spend(amount)
    
    def get_all_budgets(self) -> Dict[str, BudgetState]:
        """Get all budget states."""
        with self.lock:
            return self.budgets.copy()
    
    def reset_session(self, agent_id: str):
        """Reset session spending."""
        with self.lock:
            if agent_id in self.budgets:
                self.budgets[agent_id].session_spent = 0.0
```

### 4.2 Persistent Budget Store

```python
import json
from pathlib import Path
from typing import Dict
import aiofiles

class PersistentBudgetStore:
    """Persistent budget store with file backing."""
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.cache = InMemoryBudgetStore()
        self._load_all()
    
    def _load_all(self):
        """Load all budgets from disk."""
        for file_path in self.storage_path.glob("*.json"):
            agent_id = file_path.stem
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            budget = BudgetState(
                agent_id=agent_id,
                total_budget=data["total_budget"],
                spent=data["spent"],
                category_spent=data.get("category_spent", {}),
                session_spent=data.get("session_spent", 0.0)
            )
            self.cache.budgets[agent_id] = budget
    
    async def save(self, agent_id: str):
        """Save budget to disk."""
        budget = self.cache.get_budget(agent_id)
        if not budget:
            return
        
        file_path = self.storage_path / f"{agent_id}.json"
        data = {
            "agent_id": budget.agent_id,
            "total_budget": budget.total_budget,
            "spent": budget.spent,
            "category_spent": budget.category_spent,
            "session_spent": budget.session_spent,
            "last_updated": budget.last_updated.isoformat()
        }
        
        async with aiofiles.open(file_path, 'w') as f:
            await f.write(json.dumps(data, indent=2))
    
    async def record_spend(self, agent_id: str, amount: float, category: str):
        """Record spending and persist."""
        self.cache.record_spend(agent_id, amount, category)
        await self.save(agent_id)
```

### 4.3 Redis-Backed Budget Store

```python
import redis.asyncio as redis
from typing import Optional

class RedisBudgetStore:
    """Redis-backed distributed budget store."""
    
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
        self.key_prefix = "agent:budget:"
    
    async def get_budget(self, agent_id: str) -> Optional[BudgetState]:
        """Get budget from Redis."""
        key = f"{self.key_prefix}{agent_id}"
        data = await self.redis.hgetall(key)
        
        if not data:
            return None
        
        return BudgetState(
            agent_id=agent_id,
            total_budget=float(data.get(b"total_budget", 0)),
            spent=float(data.get(b"spent", 0)),
            category_spent=json.loads(data.get(b"category_spent", b"{}")),
            session_spent=float(data.get(b"session_spent", 0))
        )
    
    async def record_spend(self, agent_id: str, amount: float, category: str):
        """Record spending atomically in Redis."""
        key = f"{self.key_prefix}{agent_id}"
        
        async with self.redis.pipeline(transaction=True) as pipe:
            # Increment total spent
            pipe.hincrbyfloat(key, "spent", amount)
            
            # Increment category spent
            category_key = f"category:{category}"
            pipe.hincrbyfloat(key, category_key, amount)
            
            # Update timestamp
            pipe.hset(key, "last_updated", datetime.now().isoformat())
            
            await pipe.execute()
    
    async def check_budget(self, agent_id: str, amount: float) -> bool:
        """Check budget atomically."""
        key = f"{self.key_prefix}{agent_id}"
        
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.hget(key, "total_budget")
            pipe.hget(key, "spent")
            results = await pipe.execute()
        
        total_budget = float(results[0] or 0)
        spent = float(results[1] or 0)
        
        return spent + amount <= total_budget
    
    async def decrement_session(self, agent_id: str):
        """Decrement session spending."""
        key = f"{self.key_prefix}{agent_id}"
        await self.redis.hset(key, "session_spent", 0)
```

---

## 5. Real-Time Cost Stream Processing

### 5.1 Cost Event Stream

```python
import asyncio
from typing import AsyncIterator, Callable, Any
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class CostEvent:
    event_id: str
    agent_id: str
    session_id: str
    action_type: str
    model: Optional[str]
    tokens_input: int
    tokens_output: int
    cost: float
    timestamp: datetime
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "action_type": self.action_type,
            "model": self.model,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "cost": self.cost,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

class CostEventStream:
    """Manages real-time cost event streaming."""
    
    def __init__(self):
        self.subscribers: Dict[str, list[Callable]] = {}
        self.buffer: list[CostEvent] = []
        self.buffer_size = 100
        self.flush_interval = 5.0  # seconds
    
    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to cost events."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
    
    async def emit(self, event: CostEvent):
        """Emit a cost event."""
        # Add to buffer
        self.buffer.append(event)
        
        # Notify subscribers
        for callback in self.subscribers.get(event.action_type, []):
            await callback(event)
        
        # Flush buffer if full
        if len(self.buffer) >= self.buffer_size:
            await self._flush_buffer()
    
    async def _flush_buffer(self):
        """Flush event buffer to storage."""
        if not self.buffer:
            return
        
        events_to_flush = self.buffer.copy()
        self.buffer.clear()
        
        # Persist events
        await self._persist_events(events_to_flush)
    
    async def stream_events(
        self,
        agent_id: Optional[str] = None,
        action_type: Optional[str] = None,
        since: Optional[datetime] = None
    ) -> AsyncIterator[CostEvent]:
        """Stream cost events with filtering."""
        # Implementation depends on backend (Kafka, Redis Streams, etc.)
        pass
```

### 5.2 Cost Aggregator

```python
from collections import defaultdict
from datetime import datetime, timedelta

class CostAggregator:
    """Aggregates cost events in real-time."""
    
    def __init__(self):
        self.aggregations = {
            "by_agent": defaultdict(float),
            "by_session": defaultdict(float),
            "by_model": defaultdict(float),
            "by_action": defaultdict(float),
            "by_hour": defaultdict(float),
            "by_team": defaultdict(float)
        }
        self.window_size = timedelta(minutes=5)
        self.window_data = defaultdict(list)
    
    def process_event(self, event: CostEvent):
        """Process a cost event and update aggregations."""
        # Update aggregations
        self.aggregations["by_agent"][event.agent_id] += event.cost
        self.aggregations["by_session"][event.session_id] += event.cost
        self.aggregations["by_model"][event.model or "unknown"] += event.cost
        self.aggregations["by_action"][event.action_type] += event.cost
        
        hour_key = event.timestamp.strftime("%Y-%m-%d-%H")
        self.aggregations["by_hour"][hour_key] += event.cost
        
        if "team" in event.metadata:
            self.aggregations["by_team"][event.metadata["team"]] += event.cost
        
        # Update window data
        self.window_data[event.agent_id].append(event)
        self._cleanup_old_events(event.agent_id)
    
    def _cleanup_old_events(self, agent_id: str):
        """Remove events outside the current window."""
        cutoff = datetime.now() - self.window_size
        self.window_data[agent_id] = [
            e for e in self.window_data[agent_id]
            if e.timestamp > cutoff
        ]
    
    def get_window_cost(self, agent_id: str) -> float:
        """Get cost within the current time window."""
        return sum(e.cost for e in self.window_data.get(agent_id, []))
    
    def get_window_rate(self, agent_id: str) -> float:
        """Get cost rate (cost per minute) within window."""
        events = self.window_data.get(agent_id, [])
        if not events:
            return 0.0
        
        window_seconds = self.window_size.total_seconds()
        total_cost = sum(e.cost for e in events)
        
        return total_cost / (window_seconds / 60)
    
    def get_aggregation(self, dimension: str) -> Dict[str, float]:
        """Get aggregation for a dimension."""
        return dict(self.aggregations.get(dimension, {}))
```

### 5.3 Anomaly Detector

```python
from typing import List, Tuple
import statistics

class CostAnomalyDetector:
    """Detects cost anomalies in real-time."""
    
    def __init__(self, config: Dict[str, Any]):
        self.spike_threshold = config.get("spike_threshold", 3.0)
        self.trend_threshold = config.get("trend_threshold", 0.2)
        self.min_samples = config.get("min_samples", 10)
        self.baseline_window = config.get("baseline_window_days", 7)
        
        self.cost_history = defaultdict(list)
        self.anomalies = []
    
    def detect_anomalies(
        self,
        agent_id: str,
        current_cost: float,
        window_cost: float
    ) -> List[Dict]:
        """Detect anomalies for an agent."""
        anomalies = []
        
        # Record cost
        self.cost_history[agent_id].append({
            "timestamp": datetime.now(),
            "cost": current_cost,
            "window_cost": window_cost
        })
        
        # Check for spike
        if self._detect_spike(agent_id, current_cost):
            anomalies.append({
                "type": "spike",
                "severity": "high",
                "agent_id": agent_id,
                "current_cost": current_cost,
                "baseline": self._get_baseline(agent_id)
            })
        
        # Check for trend
        trend = self._detect_trend(agent_id)
        if trend > self.trend_threshold:
            anomalies.append({
                "type": "trend",
                "severity": "medium",
                "agent_id": agent_id,
                "trend_rate": trend
            })
        
        # Check for loop pattern
        if self._detect_loop(agent_id):
            anomalies.append({
                "type": "loop",
                "severity": "critical",
                "agent_id": agent_id
            })
        
        return anomalies
    
    def _detect_spike(self, agent_id: str, current_cost: float) -> bool:
        """Detect cost spike."""
        baseline = self._get_baseline(agent_id)
        if baseline == 0:
            return False
        
        return current_cost > baseline * self.spike_threshold
    
    def _get_baseline(self, agent_id: str) -> float:
        """Get baseline cost for agent."""
        history = self.cost_history.get(agent_id, [])
        if len(history) < self.min_samples:
            return 0.0
        
        costs = [h["cost"] for h in history[-100:]]
        return statistics.mean(costs)
    
    def _detect_trend(self, agent_id: str) -> float:
        """Detect cost trend."""
        history = self.cost_history.get(agent_id, [])
        if len(history) < self.min_samples * 2:
            return 0.0
        
        recent = [h["cost"] for h in history[-self.min_samples:]]
        earlier = [h["cost"] for h in history[-self.min_samples*2:-self.min_samples]]
        
        recent_avg = statistics.mean(recent)
        earlier_avg = statistics.mean(earlier)
        
        if earlier_avg == 0:
            return 0.0
        
        return (recent_avg - earlier_avg) / earlier_avg
    
    def _detect_loop(self, agent_id: str) -> bool:
        """Detect repetitive action loop."""
        history = self.cost_history.get(agent_id, [])
        if len(history) < 5:
            return False
        
        recent_costs = [h["cost"] for h in history[-5:]]
        
        # Check if all recent costs are identical (loop indicator)
        return len(set(recent_costs)) == 1
```

---

## 6. Circuit Breaker Implementations

### 6.1 State Machine Circuit Breaker

```python
from enum import Enum
from datetime import datetime, timedelta
from typing import Optional

class CircuitState(Enum):
    CLOSED = "closed"          # Normal operation
    HALF_OPEN = "half_open"    # Testing recovery
    OPEN = "open"              # Blocked

class StateMachineCircuitBreaker:
    """Circuit breaker with state machine pattern."""
    
    def __init__(self, config: Dict[str, Any]):
        self.failure_threshold = config.get("failure_threshold", 5)
        self.recovery_timeout = config.get("recovery_timeout", 300)
        self.half_open_max = config.get("half_open_max", 3)
        
        self.states: Dict[str, CircuitState] = {}
        self.failure_counts: Dict[str, int] = {}
        self.last_failure_times: Dict[str, datetime] = {}
        self.half_open_counts: Dict[str, int] = {}
    
    def get_state(self, agent_id: str) -> CircuitState:
        """Get current state for agent."""
        state = self.states.get(agent_id, CircuitState.CLOSED)
        
        # Check for recovery timeout
        if state == CircuitState.OPEN:
            last_failure = self.last_failure_times.get(agent_id)
            if last_failure and datetime.now() - last_failure > timedelta(seconds=self.recovery_timeout):
                self.states[agent_id] = CircuitState.HALF_OPEN
                self.half_open_counts[agent_id] = 0
                return CircuitState.HALF_OPEN
        
        return state
    
    def record_success(self, agent_id: str):
        """Record successful operation."""
        state = self.get_state(agent_id)
        
        if state == CircuitState.HALF_OPEN:
            self.half_open_counts[agent_id] = self.half_open_counts.get(agent_id, 0) + 1
            
            if self.half_open_counts[agent_id] >= self.half_open_max:
                # Recovery successful
                self.states[agent_id] = CircuitState.CLOSED
                self.failure_counts[agent_id] = 0
        
        elif state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_counts[agent_id] = 0
    
    def record_failure(self, agent_id: str):
        """Record failed operation."""
        state = self.get_state(agent_id)
        
        if state == CircuitState.HALF_OPEN:
            # Fail during half-open goes back to open
            self.states[agent_id] = CircuitState.OPEN
            self.last_failure_times[agent_id] = datetime.now()
        
        elif state == CircuitState.CLOSED:
            self.failure_counts[agent_id] = self.failure_counts.get(agent_id, 0) + 1
            
            if self.failure_counts[agent_id] >= self.failure_threshold:
                self.states[agent_id] = CircuitState.OPEN
                self.last_failure_times[agent_id] = datetime.now()
    
    def should_allow(self, agent_id: str) -> bool:
        """Check if operation should be allowed."""
        state = self.get_state(agent_id)
        
        if state == CircuitState.CLOSED:
            return True
        elif state == CircuitState.HALF_OPEN:
            return self.half_open_counts.get(agent_id, 0) < self.half_open_max
        else:  # OPEN
            return False
    
    def force_open(self, agent_id: str):
        """Force circuit open (for testing/manual intervention)."""
        self.states[agent_id] = CircuitState.OPEN
        self.last_failure_times[agent_id] = datetime.now()
    
    def force_close(self, agent_id: str):
        """Force circuit closed."""
        self.states[agent_id] = CircuitState.CLOSED
        self.failure_counts[agent_id] = 0
```

### 6.2 Adaptive Circuit Breaker

```python
class AdaptiveCircuitBreaker:
    """Circuit breaker that adapts thresholds based on patterns."""
    
    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config
        self.cost_history = defaultdict(list)
        self.adaptation_factor = 1.0
    
    def get_dynamic_threshold(self, agent_id: str) -> float:
        """Get dynamically adapted threshold."""
        base_threshold = self.base_config.get("failure_threshold", 5)
        
        # Adapt based on historical failure rate
        history = self.cost_history.get(agent_id, [])
        if len(history) >= 20:
            recent_failures = sum(1 for h in history[-20:] if not h["success"])
            failure_rate = recent_failures / 20
            
            if failure_rate > 0.5:
                # High failure rate - be more sensitive
                self.adaptation_factor = 0.7
            elif failure_rate < 0.1:
                # Low failure rate - be less sensitive
                self.adaptation_factor = 1.3
        
        return base_threshold * self.adaptation_factor
    
    def get_dynamic_timeout(self, agent_id: str) -> int:
        """Get dynamically adapted recovery timeout."""
        base_timeout = self.base_config.get("recovery_timeout", 300)
        
        # Adapt based on cost patterns
        history = self.cost_history.get(agent_id, [])
        if len(history) >= 10:
            avg_cost = sum(h["cost"] for h in history[-10:]) / 10
            
            if avg_cost > 10.0:
                # High cost - longer recovery
                return int(base_timeout * 1.5)
            elif avg_cost < 1.0:
                # Low cost - shorter recovery
                return int(base_timeout * 0.7)
        
        return base_timeout
```

---

## 7. Cost-Aware Agent Orchestration

### 7.1 Cost-Aware Orchestrator

```python
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class AgentTask:
    task_id: str
    agent_type: str
    priority: int  # 1-10
    estimated_cost: float
    dependencies: List[str]
    metadata: Dict[str, Any]

class CostAwareOrchestrator:
    """Orchestrates agents with cost awareness."""
    
    def __init__(self, budget_manager: BudgetManager):
        self.budget_manager = budget_manager
        self.task_queue: List[AgentTask] = []
        self.active_tasks: Dict[str, AgentTask] = {}
    
    async def schedule_tasks(
        self,
        tasks: List[AgentTask],
        agent_id: str
    ) -> List[AgentTask]:
        """Schedule tasks with budget awareness."""
        budget_remaining = self.budget_manager.get_remaining(agent_id)
        
        # Sort tasks by priority (highest first)
        sorted_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)
        
        scheduled = []
        total_cost = 0.0
        
        for task in sorted_tasks:
            if total_cost + task.estimated_cost <= budget_remaining:
                scheduled.append(task)
                total_cost += task.estimated_cost
            else:
                # Try to find cheaper alternative
                alternative = await self._find_cheaper_alternative(task)
                if alternative and total_cost + alternative.estimated_cost <= budget_remaining:
                    scheduled.append(alternative)
                    total_cost += alternative.estimated_cost
                else:
                    # Skip this task
                    continue
        
        return scheduled
    
    async def _find_cheaper_alternative(
        self,
        task: AgentTask
    ) -> Optional[AgentTask]:
        """Find a cheaper alternative for a task."""
        # Implement logic to find alternative agent type or model
        alternatives = {
            "code_review": ["code_review_mini", "code_review_fast"],
            "research": ["research_lite", "research_quick"],
            "analysis": ["analysis_basic", "analysis_summary"]
        }
        
        alt_types = alternatives.get(task.agent_type, [])
        
        for alt_type in alt_types:
            alt_cost = self._estimate_task_cost(alt_type, task.metadata)
            if alt_cost < task.estimated_cost * 0.7:
                return AgentTask(
                    task_id=f"{task.task_id}_alt",
                    agent_type=alt_type,
                    priority=task.priority,
                    estimated_cost=alt_cost,
                    dependencies=task.dependencies,
                    metadata={**task.metadata, "alternative": True}
                )
        
        return None
```

### 7.2 Cost-Aware Model Router

```python
class CostAwareModelRouter:
    """Routes to models based on cost constraints."""
    
    def __init__(self, token_estimator: TokenEstimator):
        self.token_estimator = token_estimator
        self.model_performance = defaultdict(lambda: {"success": 0, "total": 0})
    
    async def route(
        self,
        agent_id: str,
        messages: List[Dict],
        budget_remaining: float,
        quality_requirement: str = "standard"
    ) -> Dict[str, Any]:
        """Route to optimal model."""
        
        # Estimate tokens
        estimated_tokens = self.token_estimator.estimate_messages_tokens(
            messages, "gpt-4o"  # Use reference model
        )
        
        # Find candidates within budget
        candidates = self.token_estimator.find_optimal_model(
            input_tokens=estimated_tokens,
            max_output_tokens=int(estimated_tokens * 0.3),
            budget_remaining=budget_remaining,
            quality_requirement=quality_requirement
        )
        
        if not candidates:
            # Fallback to cheapest model
            return {
                "model": "gpt-4o-mini",
                "reason": "budget_constrained",
                "estimated_cost": self.token_estimator.estimate_inference_cost(
                    "gpt-4o-mini", estimated_tokens, int(estimated_tokens * 0.3)
                ).estimated_cost
            }
        
        # Select best candidate
        selected = candidates[0]
        
        # Adjust based on performance history
        performance = self.model_performance.get(selected["model"], {"success": 0, "total": 0})
        if performance["total"] > 10:
            success_rate = performance["success"] / performance["total"]
            if success_rate < 0.8:
                # Low success rate - try next candidate
                if len(candidates) > 1:
                    selected = candidates[1]
        
        return {
            "model": selected["model"],
            "reason": "cost_effective",
            "estimated_cost": selected["estimated_cost"],
            "quality_score": selected["quality_score"]
        }
    
    def record_outcome(self, model: str, success: bool):
        """Record model outcome for performance tracking."""
        self.model_performance[model]["total"] += 1
        if success:
            self.model_performance[model]["success"] += 1
```

---

## 8. Storage and Persistence

### 8.1 Cost Data Schema

```sql
-- Cost events table
CREATE TABLE cost_events (
    id UUID PRIMARY KEY,
    agent_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    action_type VARCHAR(50) NOT NULL,
    model VARCHAR(100),
    tokens_input INTEGER,
    tokens_output INTEGER,
    cost DECIMAL(10, 6) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    metadata JSONB,
    
    INDEX idx_agent_id (agent_id),
    INDEX idx_session_id (session_id),
    INDEX idx_timestamp (timestamp),
    INDEX idx_action_type (action_type)
);

-- Budget configurations table
CREATE TABLE budget_configs (
    id UUID PRIMARY KEY,
    entity_type VARCHAR(50) NOT NULL,  -- 'organization', 'team', 'project', 'agent_type'
    entity_id VARCHAR(255) NOT NULL,
    budget_type VARCHAR(50) NOT NULL,  -- 'monthly', 'daily', 'session'
    limit_amount DECIMAL(10, 2) NOT NULL,
    current_usage DECIMAL(10, 6) DEFAULT 0,
    alert_thresholds JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE (entity_type, entity_id, budget_type)
);

-- Cost summaries table (materialized for reporting)
CREATE TABLE cost_summaries (
    id UUID PRIMARY KEY,
    period_start TIMESTAMP NOT NULL,
    period_end TIMESTAMP NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(255) NOT NULL,
    total_cost DECIMAL(10, 6) NOT NULL,
    breakdown JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_period (period_start, period_end),
    INDEX idx_entity (entity_type, entity_id)
);
```

### 8.2 Event Sourcing for Cost Data

```python
from typing import List
from datetime import datetime
import json

class CostEventStore:
    """Event-sourced cost data store."""
    
    def __init__(self, db_connection):
        self.db = db_connection
    
    async def append_event(self, event: CostEvent):
        """Append a cost event."""
        query = """
        INSERT INTO cost_events 
        (id, agent_id, session_id, action_type, model, 
         tokens_input, tokens_output, cost, timestamp, metadata)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """
        
        await self.db.execute(
            query,
            event.event_id,
            event.agent_id,
            event.session_id,
            event.action_type,
            event.model,
            event.tokens_input,
            event.tokens_output,
            event.cost,
            event.timestamp,
            json.dumps(event.metadata)
        )
    
    async def get_events(
        self,
        agent_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[CostEvent]:
        """Get events for an agent within a time range."""
        query = """
        SELECT * FROM cost_events
        WHERE agent_id = $1
        AND timestamp >= $2
        AND timestamp <= $3
        ORDER BY timestamp
        """
        
        rows = await self.db.fetch(query, agent_id, start_time, end_time)
        
        return [
            CostEvent(
                event_id=row["id"],
                agent_id=row["agent_id"],
                session_id=row["session_id"],
                action_type=row["action_type"],
                model=row["model"],
                tokens_input=row["tokens_input"],
                tokens_output=row["tokens_output"],
                cost=row["cost"],
                timestamp=row["timestamp"],
                metadata=json.loads(row["metadata"])
            )
            for row in rows
        ]
    
    async def get_aggregated_costs(
        self,
        entity_type: str,
        entity_id: str,
        period_start: datetime,
        period_end: datetime
    ) -> Dict[str, float]:
        """Get aggregated costs for an entity."""
        query = """
        SELECT 
            action_type,
            SUM(cost) as total_cost,
            COUNT(*) as event_count
        FROM cost_events
        WHERE timestamp >= $1 AND timestamp <= $2
        AND metadata->>'$3' = $4
        GROUP BY action_type
        """
        
        rows = await self.db.fetch(
            query, period_start, period_end, entity_type, entity_id
        )
        
        return {
            row["action_type"]: {
                "cost": row["total_cost"],
                "count": row["event_count"]
            }
            for row in rows
        }
```

---

## 9. API Design for Cost Control

### 9.1 REST API Endpoints

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI(title="Agent Cost Control API")

# Models
class BudgetCreate(BaseModel):
    entity_type: str
    entity_id: str
    budget_type: str
    limit_amount: float
    alert_thresholds: Optional[dict] = None

class CostQuery(BaseModel):
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    group_by: Optional[str] = None

# Endpoints
@app.post("/api/v1/budgets")
async def create_budget(budget: BudgetCreate):
    """Create or update a budget."""
    # Implementation
    pass

@app.get("/api/v1/budgets/{entity_type}/{entity_id}")
async def get_budget(entity_type: str, entity_id: str):
    """Get budget for an entity."""
    # Implementation
    pass

@app.get("/api/v1/costs/summary")
async def get_cost_summary(query: CostQuery):
    """Get cost summary."""
    # Implementation
    pass

@app.get("/api/v1/costs/realtime/{agent_id}")
async def get_realtime_costs(agent_id: str):
    """Get real-time cost data for an agent."""
    # Implementation
    pass

@app.post("/api/v1/costs/check")
async def check_budget(
    agent_id: str,
    amount: float,
    category: str = "inference"
):
    """Check if a cost is within budget."""
    # Implementation
    pass

@app.get("/api/v1/anomalies")
async def get_anomalies(
    agent_id: Optional[str] = None,
    since: Optional[datetime] = None
):
    """Get detected anomalies."""
    # Implementation
    pass

@app.post("/api/v1/circuit-breaker/{agent_id}/reset")
async def reset_circuit_breaker(agent_id: str):
    """Reset circuit breaker for an agent."""
    # Implementation
    pass
```

### 9.2 WebSocket for Real-Time Updates

```python
from fastapi import WebSocket, WebSocketDisconnect
import asyncio

class CostWebSocketManager:
    """Manages WebSocket connections for real-time cost updates."""
    
    def __init__(self):
        self.connections: Dict[str, List[WebSocket]] = defaultdict(list)
        self.cost_stream = CostEventStream()
    
    async def connect(self, websocket: WebSocket, agent_id: str):
        """Connect a WebSocket for an agent."""
        await websocket.accept()
        self.connections[agent_id].append(websocket)
    
    async def disconnect(self, websocket: WebSocket, agent_id: str):
        """Disconnect a WebSocket."""
        self.connections[agent_id].remove(websocket)
    
    async def broadcast_cost_update(self, event: CostEvent):
        """Broadcast cost update to connected clients."""
        agent_connections = self.connections.get(event.agent_id, [])
        
        message = {
            "type": "cost_update",
            "data": event.to_dict()
        }
        
        for connection in agent_connections:
            try:
                await connection.send_json(message)
            except WebSocketDisconnect:
                self.connections[event.agent_id].remove(connection)

@app.websocket("/ws/costs/{agent_id}")
async def websocket_costs(websocket: WebSocket, agent_id: str):
    """WebSocket endpoint for real-time cost updates."""
    manager = CostWebSocketManager()
    await manager.connect(websocket, agent_id)
    
    try:
        while True:
            # Keep connection alive and listen for messages
            data = await websocket.receive_text()
            
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        await manager.disconnect(websocket, agent_id)
```

---

## 10. Testing Cost Control Systems

### 10.1 Unit Tests

```python
import pytest
from unittest.mock import Mock, AsyncMock

class TestCostInterceptor:
    """Tests for cost interceptor middleware."""
    
    @pytest.fixture
    def interceptor(self):
        budget_manager = Mock()
        budget_manager.check_budget = AsyncMock(return_value=Mock(allowed=True))
        budget_manager.get_remaining = Mock(return_value=100.0)
        
        token_counter = Mock()
        token_counter.estimate_input_tokens = Mock(return_value=1000)
        
        model_router = Mock()
        model_router.find_alternative = AsyncMock(return_value=None)
        
        circuit_breaker = Mock()
        circuit_breaker.check = Mock(return_value=Mock(blocked=False))
        
        cost_tracker = Mock()
        cost_tracker.record = AsyncMock()
        
        return CostInterceptorMiddleware(
            budget_manager=budget_manager,
            token_counter=token_counter,
            model_router=model_router,
            circuit_breaker=circuit_breaker,
            cost_tracker=cost_tracker
        )
    
    @pytest.mark.asyncio
    async def test_inference_approved(self, interceptor):
        """Test that inference is approved within budget."""
        result = await interceptor.intercept_inference(
            agent_id="test-agent",
            session_id="test-session",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100
        )
        
        assert result.allowed is True
        assert result.estimated_cost > 0
    
    @pytest.mark.asyncio
    async def test_inference_denied_budget_exceeded(self, interceptor):
        """Test that inference is denied when budget exceeded."""
        interceptor.budget_manager.check_budget.return_value = Mock(
            allowed=False,
            reason="budget_exceeded"
        )
        
        result = await interceptor.intercept_inference(
            agent_id="test-agent",
            session_id="test-session",
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100
        )
        
        assert result.allowed is False
        assert result.reason == "budget_exceeded"

class TestBudgetManager:
    """Tests for budget manager."""
    
    @pytest.fixture
    def budget_manager(self):
        store = InMemoryBudgetStore()
        store.set_budget("test-agent", 100.0)
        return BudgetManager(store)
    
    def test_check_budget_within_limit(self, budget_manager):
        """Test budget check within limit."""
        result = budget_manager.check_budget_sync("test-agent", 50.0)
        assert result is True
    
    def test_check_budget_exceeds_limit(self, budget_manager):
        """Test budget check exceeds limit."""
        result = budget_manager.check_budget_sync("test-agent", 150.0)
        assert result is False
    
    def test_record_spend(self, budget_manager):
        """Test recording spend."""
        budget_manager.record_spend_sync("test-agent", 25.0, "inference")
        
        remaining = budget_manager.get_remaining("test-agent")
        assert remaining == 75.0
```

### 10.2 Integration Tests

```python
class TestCostControlIntegration:
    """Integration tests for cost control system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_cost_control(self):
        """Test complete cost control flow."""
        # Setup
        budget_store = InMemoryBudgetStore()
        budget_store.set_budget("test-agent", 10.0)
        
        interceptor = CostInterceptorMiddleware(
            budget_manager=BudgetManager(budget_store),
            token_counter=TokenEstimator(),
            model_router=CostAwareModelRouter(TokenEstimator()),
            circuit_breaker=StateMachineCircuitBreaker({}),
            cost_tracker=CostTracker()
        )
        
        # Execute multiple calls
        results = []
        for i in range(5):
            result = await interceptor.intercept_inference(
                agent_id="test-agent",
                session_id="test-session",
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"Test {i}"}],
                max_tokens=100
            )
            results.append(result)
            
            if result.allowed:
                await interceptor.record_actual_cost(
                    agent_id="test-agent",
                    session_id="test-session",
                    action_type=ActionType.INFERENCE,
                    actual_cost=result.estimated_cost
                )
        
        # Verify
        allowed_count = sum(1 for r in results if r.allowed)
        assert allowed_count >= 1  # At least one should be allowed
```

---

## Summary

### Key Technical Components

1. **Cost Interceptor Middleware** — Core layer that intercepts and controls all agent costs
2. **Token Counting** — Accurate estimation and counting of tokens for cost calculation
3. **Budget State Management** — Thread-safe, persistent storage of budget states
4. **Real-Time Stream Processing** — Event streaming and aggregation for cost monitoring
5. **Circuit Breakers** — State machine pattern for automatic intervention
6. **Cost-Aware Orchestration** — Task scheduling and model routing with budget awareness
7. **Persistent Storage** — Database schemas and event sourcing for cost data
8. **API Layer** — REST and WebSocket APIs for cost control management
9. **Testing** — Unit and integration tests for reliability

### Implementation Priorities

1. Start with in-memory budget store and basic interceptor
2. Add token counting and cost estimation
3. Implement circuit breaker for critical protection
4. Add persistent storage for production use
5. Implement real-time monitoring and alerting
6. Add advanced features (adaptive thresholds, ML-based anomaly detection)

---

*Last updated: July 2026*
*Part of the AI Knowledge Library — Category 59: AI Agent Financial Governance and Cost Control*
