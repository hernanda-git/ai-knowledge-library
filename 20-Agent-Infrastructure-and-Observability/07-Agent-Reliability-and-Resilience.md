# 07 — Agent Reliability and Resilience

## 1. Introduction to Agent Reliability

### 1.1 The Reliability Challenge

Agent systems introduce a fundamentally new reliability challenge: they combine the non-determinism of LLMs with the real-world consequences of autonomous action. An agent that makes a bad decision can:

- Delete production database records
- Deploy broken code to production
- Make incorrect financial transactions
- Leak sensitive information
- Enter infinite loops that burn thousands of dollars in API costs

Unlike traditional software, where failures are usually deterministic and reproducible, agent failures are:
- **Non-deterministic**: Same input may produce different failures
- **Path-dependent**: Failure depends on the sequence of decisions, not just the final state
- **Context-dependent**: Context window state influences behavior in complex ways
- **Intermittent**: May work 99% of the time and fail catastrophically 1% of the time

### 1.2 Reliability Goals

For production agent systems, define reliability targets:

| Metric | Target | Measurement |
|--------|--------|------------|
| Task Completion Rate | > 95% | Successful sessions / total sessions |
| Error Rate | < 2% | Error sessions / total sessions |
| P99 Latency | < 120s | 99th percentile session duration |
| Availability | > 99.9% | Uptime / total time |
| Cost per Task | < $0.10 | Average cost per completed task |
| False Action Rate | < 0.1% | Harmful/incorrect actions / total actions |

## 2. Failure Modes for Agents

### 2.1 Failure Mode Taxonomy

```python
"""Complete taxonomy of agent failure modes."""

from enum import Enum

class AgentFailureMode(str, Enum):
    """All known failure modes for agent systems."""
    
    # === LLM-Related Failures ===
    
    HALLUCINATION = "hallucination"
    """Agent generates factually incorrect information with high confidence."""
    
    HALLUCINATION_CASCADE = "hallucination_cascade"
    """One hallucination leads to more, compounding the error."""
    
    CONTEXT_OVERFLOW = "context_overflow"
    """Agent exceeds context window limits, losing information."""
    
    CONTEXT_ATTENTION_FAILURE = "context_attention_failure"
    """LLM loses track of information within the context window (lost in the middle)."""
    
    MODEL_DEGRADATION = "model_degradation"
    """LLM model performance degrades or model is deprecated."""
    
    TONE_OR_STYLE_DRIFT = "tone_drift"
    """Agent output deviates from expected tone/constraints."""
    
    # === Loop and Control Flow Failures ===
    
    INFINITE_LOOP = "infinite_loop"
    """Agent repeats same reasoning/tool call pattern without progress."""
    
    ITERATION_EXCEEDED = "iteration_exceeded"
    """Agent exceeds maximum allowed steps."""
    
    BACKTRACKING_LOOP = "backtracking_loop"
    """Agent keeps reversing previous decisions without making progress."""
    
    OVER_THINKING = "over_thinking"
    """Agent spends excessive tokens on reasoning without action."""
    
    PREMATURE_TERMINATION = "premature_termination"
    """Agent stops before task is complete, claiming it's done."""
    
    # === Tool-Related Failures ===
    
    TOOL_FAILURE = "tool_failure"
    """Tool API call fails (network, auth, server error)."""
    
    TOOL_MISUSE = "tool_misuse"
    """Agent uses correct tool with wrong parameters."""
    
    TOOL_SELECTION_ERROR = "tool_selection_error"
    """Agent chooses wrong tool for the task."""
    
    TOOL_OUTPUT_PARSING_FAILURE = "tool_output_parsing_failure"
    """Agent fails to correctly parse or use tool output."""
    
    # === Rate and Resource Failures ===
    
    RATE_LIMITED = "rate_limited"
    """LLM API or tool API rate limit exceeded."""
    
    TIMEOUT = "timeout"
    """Operation exceeds configured timeout."""
    
    QUOTA_EXCEEDED = "quota_exceeded"
    """API quota exhausted (daily/monthly limits)."""
    
    CONCURRENCY_LIMIT = "concurrency_limit"
    """Too many concurrent agent executions."""
    
    # === Reasoning Failures ===
    
    CIRCULAR_REASONING = "circular_reasoning"
    """Agent goes in circles with reasoning, never progressing."""
    
    INCORRECT_PLANNING = "incorrect_planning"
    """Agent creates flawed plan to accomplish the task."""
    
    OVERLY_CAUTIOUS = "overly_cautious"
    """Agent refuses to act or asks for confirmation excessively."""
    
    OVERLY_CONFIDENT = "overly_confident"
    """Agent takes actions without sufficient verification."""
    
    # === Safety and Security Failures ===
    
    PROMPT_INJECTION = "prompt_injection"
    """User input successfully injects instructions that override system prompt."""
    
    DATA_LEAKAGE = "data_leakage"
    """Agent inadvertently reveals sensitive information."""
    
    PRIVILEGE_ESCALATION = "privilege_escalation"
    """Agent performs actions beyond its intended scope."""
    
    TOXIC_OUTPUT = "toxic_output"
    """Agent generates harmful, biased, or inappropriate content."""
    
    # === Infrastructure Failures ===
    
    DEPENDENCY_FAILURE = "dependency_failure"
    """External service dependency goes down."""
    
    DATABASE_FAILURE = "database_failure"
    """State/memory database becomes unavailable."""
    
    DEPLOYMENT_FAILURE = "deployment_failure"
    """New agent version fails to deploy or crashes."""
    
    CONFIGURATION_ERROR = "configuration_error"
    """Misconfigured agent settings, API keys, or tool definitions."""
```

### 2.2 Failure Detection

```python
"""Failure detection for agent systems."""

from typing import Optional, List
from datetime import datetime

class FailureDetector:
    """
    Detects agent failures in real-time by monitoring execution patterns.
    Uses heuristic and ML-based detection strategies.
    """
    
    def __init__(self):
        self.detection_strategies = [
            self._detect_loop,
            self._detect_excessive_cost,
            self._detect_context_overflow,
            self._detect_repetitive_output,
            self._detect_reckless_behavior,
        ]
    
    def analyze_step(self, step_data: dict, history: List[dict]) -> Optional[AgentFailureMode]:
        """Analyze a single agent step for failure patterns."""
        for strategy in self.detection_strategies:
            result = strategy(step_data, history)
            if result:
                return result
        return None
    
    def _detect_loop(self, step: dict, history: List[dict]) -> Optional[AgentFailureMode]:
        """Detect if agent is in a loop."""
        if len(history) < 6:
            return None
        
        # Check for repeated tool calls with same arguments
        recent_tool_calls = [
            s.get("tool_call") for s in history[-6:]
            if s.get("type") == "tool_call"
        ]
        
        if len(recent_tool_calls) >= 3:
            # Check if same tool with same args appears 3+ times
            from collections import Counter
            tool_signatures = [
                f"{t.get('tool_name')}:{str(t.get('args', {}))}"
                for t in recent_tool_calls
            ]
            most_common = Counter(tool_signatures).most_common(1)
            if most_common and most_common[0][1] >= 3:
                return AgentFailureMode.INFINITE_LOOP
        
        # Check for repeating thought patterns
        if len(history) >= 4:
            recent_thoughts = [
                s.get("thought", "")[:50] for s in history[-4:]
                if s.get("type") == "reasoning"
            ]
            if len(set(recent_thoughts)) <= 1 and len(recent_thoughts) >= 3:
                return AgentFailureMode.CIRCULAR_REASONING
        
        return None
    
    def _detect_excessive_cost(self, step: dict, history: List[dict]) -> Optional[AgentFailureMode]:
        """Detect if agent is accumulating excessive cost."""
        total_steps = len([s for s in history if s.get("type") in ("llm_call", "tool_call")])
        total_cost = sum(s.get("cost", 0) for s in history)
        
        if total_steps > 20 and total_cost > 1.0:
            return AgentFailureMode.ITERATION_EXCEEDED
        
        # Check if cost is accumulating very quickly
        if len(history) >= 5:
            recent_cost = sum(s.get("cost", 0) for s in history[-5:])
            if recent_cost > 0.50:
                return AgentFailureMode.OVER_THINKING
        
        return None
    
    def _detect_context_overflow(self, step: dict, history: List[dict]) -> Optional[AgentFailureMode]:
        """Detect if agent is approaching context limits."""
        total_tokens = sum(s.get("tokens", 0) for s in history)
        context_limit = 128000  # gpt-4o context limit
        
        if total_tokens > context_limit * 0.85:
            return AgentFailureMode.CONTEXT_OVERFLOW
        
        return None
    
    def _detect_repetitive_output(self, step: dict, history: List[dict]) -> Optional[AgentFailureMode]:
        """Detect if agent is producing repetitive output."""
        if len(history) < 3:
            return None
        
        recent_outputs = [
            s.get("output", "")[:100] for s in history[-3:]
            if s.get("type") == "llm_call"
        ]
        
        if len(recent_outputs) >= 2:
            # Check for near-duplicate outputs
            from difflib import SequenceMatcher
            if SequenceMatcher(None, recent_outputs[-2], recent_outputs[-1]).ratio() > 0.9:
                return AgentFailureMode.INFINITE_LOOP
        
        return None
    
    def _detect_reckless_behavior(self, step: dict, history: List[dict]) -> Optional[AgentFailureMode]:
        """Detect if agent is acting recklessly (overly confident)."""
        content = step.get("output", "") or step.get("thought", "")
        
        # Check for language indicating certainty about unverifiable claims
        reckless_phrases = [
            "I'm certain", "definitely", "without question",
            "I can guarantee", "trust me", "just do it",
            "don't worry about", "it's fine",
        ]
        
        content_lower = content.lower()
        if any(phrase in content_lower for phrase in reckless_phrases):
            return AgentFailureMode.OVERLY_CONFIDENT
        
        return None
```

## 3. Retry Strategies

### 3.1 Exponential Backoff with Jitter

```python
"""Exponential backoff with jitter for agent retries."""

import random
import time
from typing import Optional, Callable, Any
from functools import wraps

class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: tuple = (Exception,),
        on_retry: Optional[Callable] = None,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions
        self.on_retry = on_retry


def with_retry(config: RetryConfig = None):
    """
    Decorator that retries a function with exponential backoff and jitter.
    Specifically designed for agent operations (LLM calls, tool calls).
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_retries:
                        raise  # Max retries exceeded
                    
                    # Calculate delay with exponential backoff and jitter
                    delay = min(
                        config.base_delay * (config.exponential_base ** attempt),
                        config.max_delay
                    )
                    
                    if config.jitter:
                        delay = delay * (0.5 + random.random() * 0.5)  # 50-100% of delay
                    
                    if config.on_retry:
                        config.on_retry(attempt=attempt + 1, delay=delay, error=e)
                    
                    time.sleep(delay)
            
            raise last_exception  # Should not reach here
        
        return wrapper
    
    return decorator


# Specific retry configurations for different agent operations

def llm_retry_config() -> RetryConfig:
    """Retry config for LLM API calls."""
    return RetryConfig(
        max_retries=5,
        base_delay=1.0,
        max_delay=30.0,
        exponential_base=2.0,
        jitter=True,
        retryable_exceptions=(
            ConnectionError,
            TimeoutError,
            Exception,  # Catch-all for API errors
        ),
        on_retry=lambda attempt, delay, error: print(
            f"LLM call failed (attempt {attempt}), retrying in {delay:.1f}s: {error}"
        ),
    )

def tool_retry_config() -> RetryConfig:
    """Retry config for tool API calls."""
    return RetryConfig(
        max_retries=2,
        base_delay=0.5,
        max_delay=10.0,
        exponential_base=2.0,
        jitter=True,
        retryable_exceptions=(ConnectionError, TimeoutError),
    )

def agent_step_retry_config() -> RetryConfig:
    """Retry config for entire agent steps."""
    return RetryConfig(
        max_retries=2,
        base_delay=2.0,
        max_delay=30.0,
        exponential_base=3.0,
        jitter=True,
    )


# Example usage in an agent
class ResilientAgent:
    """Agent that uses retry strategies for all operations."""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    @with_retry(config=llm_retry_config())
    def call_llm(self, messages: list, model: str = "gpt-4o-mini"):
        """LLM call with automatic retry on failure."""
        return self.llm.chat(messages=messages, model=model)
    
    @with_retry(config=tool_retry_config())
    def call_tool(self, tool_fn, **kwargs):
        """Tool call with automatic retry on transient failures."""
        return tool_fn(**kwargs)
```

### 3.2 Circuit Breaker Pattern

```python
"""Circuit breaker for agent dependencies."""

import time
import threading
from enum import Enum
from typing import Optional, Callable, Any

class CircuitState(Enum):
    CLOSED = "closed"          # Normal operation, requests pass through
    OPEN = "open"              # Requests are rejected immediately
    HALF_OPEN = "half_open"    # Testing if service has recovered

class CircuitBreaker:
    """
    Circuit breaker pattern for agent dependencies (LLM APIs, tools, databases).
    Prevents cascading failures by failing fast when a dependency is unhealthy.
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,          # Failures before opening circuit
        recovery_timeout: float = 30.0,       # Seconds before testing recovery
        half_open_max_requests: int = 1,      # Test requests in half-open state
        success_threshold: int = 2,           # Successes to close circuit again
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_requests = half_open_max_requests
        self.success_threshold = success_threshold
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.half_open_requests = 0
        self.lock = threading.Lock()
        
        # Metrics
        self.total_requests = 0
        self.total_failures = 0
        self.total_rejected = 0
    
    def call(self, func: Callable, fallback: Optional[Callable] = None, **kwargs) -> Any:
        """
        Execute a function with circuit breaker protection.
        
        Args:
            func: The function to call (e.g., LLM call, tool call)
            fallback: Optional fallback function if circuit is open
            **kwargs: Arguments passed to the function
        
        Returns:
            Function result or fallback result
        """
        self.total_requests += 1
        
        if not self._should_allow_request():
            self.total_rejected += 1
            if fallback:
                return fallback(**kwargs)
            raise CircuitBreakerOpenError(f"Circuit breaker '{self.name}' is OPEN")
        
        try:
            result = func(**kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_allow_request(self) -> bool:
        """Determine if a request should be allowed through."""
        with self.lock:
            if self.state == CircuitState.CLOSED:
                return True
            
            elif self.state == CircuitState.OPEN:
                # Check if recovery timeout has elapsed
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_requests = 0
                    return True
                return False
            
            elif self.state == CircuitState.HALF_OPEN:
                # Allow limited test requests
                if self.half_open_requests < self.half_open_max_requests:
                    self.half_open_requests += 1
                    return True
                return False
            
            return True
    
    def _on_success(self):
        """Handle successful request."""
        with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    print(f"Circuit breaker '{self.name}' CLOSED (recovered)")
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0  # Reset failure count on success
    
    def _on_failure(self):
        """Handle failed request."""
        self.total_failures += 1
        with self.lock:
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self.success_count = 0
                print(f"Circuit breaker '{self.name}' OPEN (half-open test failed)")
            
            elif self.state == CircuitState.CLOSED:
                self.failure_count += 1
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitState.OPEN
                    print(f"Circuit breaker '{self.name}' OPEN (failure threshold reached)")


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


# Circuit breaker registry for managing multiple dependencies
class CircuitBreakerRegistry:
    """Manages circuit breakers for all agent dependencies."""
    
    def __init__(self):
        self.breakers = {}
    
    def get_or_create(self, name: str, **config) -> CircuitBreaker:
        """Get existing circuit breaker or create a new one."""
        if name not in self.breakers:
            self.breakers[name] = CircuitBreaker(name=name, **config)
        return self.breakers[name]
    
    def status_report(self) -> dict:
        """Get status of all circuit breakers."""
        return {
            name: {
                "state": breaker.state.value,
                "failure_count": breaker.failure_count,
                "total_requests": breaker.total_requests,
                "total_failures": breaker.total_failures,
                "total_rejected": breaker.total_rejected,
            }
            for name, breaker in self.breakers.items()
        }


# Example: Using circuit breakers in an agent
class AgentWithCircuitBreakers:
    """Agent that protects all dependencies with circuit breakers."""
    
    def __init__(self):
        self.breakers = CircuitBreakerRegistry()
        
        # Create circuit breakers for each dependency
        self.llm_breaker = self.breakers.get_or_create(
            "llm-api", failure_threshold=5, recovery_timeout=30
        )
        self.tools_breakers = {
            "web_search": self.breakers.get_or_create(
                "tool-web_search", failure_threshold=3, recovery_timeout=15
            ),
            "database": self.breakers.get_or_create(
                "tool-database", failure_threshold=2, recovery_timeout=30
            ),
        }
    
    def call_llm(self, messages: list, **kwargs) -> str:
        """Protected LLM call."""
        def llm_call(messages=messages, **kwargs):
            # Actual API call
            return "response"
        
        def fallback_llm(**kwargs):
            """Fallback: use a simpler model or cached response."""
            return "fallback response (using cache)"
        
        return self.llm_breaker.call(
            func=llm_call,
            fallback=fallback_llm,
            **kwargs
        )
    
    def call_tool(self, tool_name: str, **kwargs) -> str:
        """Protected tool call."""
        breaker = self.tools_breakers.get(tool_name)
        if not breaker:
            raise ValueError(f"No circuit breaker for tool '{tool_name}'")
        
        def tool_call(**kwargs):
            # Actual tool implementation
            return f"tool result for {kwargs}"
        
        def fallback_tool(**kwargs):
            """Fallback: return a safe error message."""
            return f"Tool {tool_name} is currently unavailable. Please try again later."
        
        return breaker.call(func=tool_call, fallback=fallback_tool, **kwargs)
```

## 4. Timeout Management

### 4.1 Multi-Level Timeouts

```python
"""Timeout management at multiple levels in agent execution."""

import asyncio
import signal
from contextlib import contextmanager
from typing import Optional

class AgentTimeouts:
    """
    Manages timeouts at multiple levels of agent execution.
    Each level wraps the one below for progressive escalation.
    """
    
    def __init__(
        self,
        llm_timeout: float = 30.0,       # Individual LLM call
        tool_timeout: float = 15.0,       # Individual tool call
        step_timeout: float = 60.0,       # Single agent step
        session_timeout: float = 600.0,   # Entire agent session
        thinking_timeout: float = 30.0,   # LLM thinking/reasoning phase
    ):
        self.llm_timeout = llm_timeout
        self.tool_timeout = tool_timeout
        self.step_timeout = step_timeout
        self.session_timeout = session_timeout
        self.thinking_timeout = thinking_timeout


class SyncTimeout:
    """Timeout context manager for synchronous code using signal-based alarms."""
    
    def __init__(self, seconds: float, error_message: str = "Operation timed out"):
        self.seconds = seconds
        self.error_message = error_message
    
    def __enter__(self):
        if self.seconds > 0:
            signal.signal(signal.SIGALRM, self._handler)
            signal.alarm(int(self.seconds))
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.seconds > 0:
            signal.alarm(0)
        return False
    
    def _handler(self, signum, frame):
        raise TimeoutError(self.error_message)


class AsyncTimeout:
    """Timeout context manager for async code."""
    
    def __init__(self, seconds: float, error_message: str = "Operation timed out"):
        self.seconds = seconds
        self.error_message = error_message
    
    async def __aenter__(self):
        self.task = asyncio.current_task()
        self.timer = asyncio.create_task(self._timeout())
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.timer.cancel()
        return False
    
    async def _timeout(self):
        await asyncio.sleep(self.seconds)
        if self.task:
            self.task.cancel()
            raise asyncio.TimeoutError(self.error_message)


# Usage in an agent
class AgentWithTimeouts:
    """Agent with timeout protection at every level."""
    
    def __init__(self, timeouts: AgentTimeouts = None):
        self.timeouts = timeouts or AgentTimeouts()
    
    def run(self, task: str) -> str:
        """Run agent with session-level timeout."""
        with SyncTimeout(
            self.timeouts.session_timeout,
            f"Agent session timed out after {self.timeouts.session_timeout}s"
        ):
            return self._execute(task)
    
    def _execute(self, task: str) -> str:
        """Execute agent loop with step-level timeout."""
        # ... agent loop ...
        return "result"
    
    def _execute_step(self, step_data: dict) -> dict:
        """Execute a single step with step timeout."""
        with SyncTimeout(
            self.timeouts.step_timeout,
            f"Agent step timed out after {self.timeouts.step_timeout}s"
        ):
            # LLM call with its own timeout
            llm_result = self._call_llm(step_data)
            # Tool calls with their own timeout
            tool_result = self._call_tool(step_data.get("tool"))
            
            return {"llm": llm_result, "tool": tool_result}
    
    def _call_llm(self, data: dict) -> str:
        """LLM call with dedicated timeout."""
        with SyncTimeout(
            self.timeouts.llm_timeout,
            f"LLM call timed out after {self.timeouts.llm_timeout}s"
        ):
            # Actual LLM call
            return "llm response"
    
    def _call_tool(self, tool_name: str) -> str:
        """Tool call with dedicated timeout."""
        with SyncTimeout(
            self.timeouts.tool_timeout,
            f"Tool call timed out after {self.timeouts.tool_timeout}s"
        ):
            # Actual tool call
            return "tool response"
```

## 5. Fallback Agents and Graceful Degradation

### 5.1 Fallback Agent Architecture

```python
"""Fallback agent system for graceful degradation."""

from typing import Optional, List, Callable
from dataclasses import dataclass

@dataclass
class FallbackLevel:
    """A level of fallback when the primary agent fails."""
    name: str
    agent_fn: Callable
    max_cost: float
    max_latency: float
    capabilities: List[str]  # What this fallback can handle

class FallbackAgentManager:
    """
    Manages a hierarchy of fallback agents.
    When the primary agent fails, tries progressively simpler/costlier fallbacks.
    """
    
    def __init__(self, primary_agent: Callable, fallbacks: List[FallbackLevel]):
        """
        Args:
            primary_agent: The main agent function
            fallbacks: Ordered list of fallback levels (tried in order)
        """
        self.primary = primary_agent
        self.fallbacks = fallbacks
    
    def execute(self, task: str, max_retries: int = 1) -> dict:
        """
        Execute with fallback chain.
        Tries primary, then each fallback level in order.
        """
        errors = []
        
        # Try primary agent
        try:
            result = self.primary(task)
            if self._is_result_acceptable(result):
                return {"level": "primary", "result": result, "errors": errors}
        except Exception as e:
            errors.append({"level": "primary", "error": str(e)})
        
        # Try fallback levels
        for fallback in self.fallbacks:
            for attempt in range(max_retries):
                try:
                    result = fallback.agent_fn(task)
                    if self._is_result_acceptable(result):
                        return {
                            "level": fallback.name,
                            "result": result,
                            "errors": errors,
                            "fallback_used": True,
                        }
                except Exception as e:
                    errors.append({"level": fallback.name, "error": str(e)})
                    continue
                break
        
        # All fallbacks failed
        return {
            "level": "failed",
            "result": None,
            "errors": errors,
            "fallback_used": True,
        }
    
    def _is_result_acceptable(self, result) -> bool:
        """Check if the result is acceptable (not a generic error response)."""
        if result is None:
            return False
        if isinstance(result, str) and len(result) < 10:
            return False
        if isinstance(result, dict) and result.get("error"):
            return False
        return True


# Example: Degradation levels for a research agent
def create_research_agent_with_fallbacks() -> FallbackAgentManager:
    """Create a research agent with graceful degradation."""
    
    def full_research_agent(task: str) -> dict:
        """Full agent with web search, analysis, and synthesis."""
        # ... implementation ...
        return {"output": "Full research result"}
    
    def analysis_only_agent(task: str) -> dict:
        """Agent that only analyzes provided context (no web search)."""
        return {"output": "Analysis based on available knowledge"}
    
    def simple_qa_agent(task: str) -> dict:
        """Simple Q&A agent (no tools, just LLM knowledge)."""
        return {"output": "Answer based on training data"}
    
    return FallbackAgentManager(
        primary_agent=full_research_agent,
        fallbacks=[
            FallbackLevel(
                name="analysis_only",
                agent_fn=analysis_only_agent,
                max_cost=0.01,
                max_latency=10.0,
                capabilities=["analysis", "reasoning"],
            ),
            FallbackLevel(
                name="simple_qa",
                agent_fn=simple_qa_agent,
                max_cost=0.002,
                max_latency=5.0,
                capabilities=["simple_qa"],
            ),
        ]
    )
```

### 5.2 Graceful Degradation Patterns

```python
"""Graceful degradation patterns for agents."""

class DegradationManager:
    """
    Manages feature degradation based on available resources.
    When a dependency fails, degrade the corresponding capability
    instead of failing completely.
    """
    
    def __init__(self):
        self.capabilities = {
            "web_search": True,
            "code_execution": True,
            "database_access": True,
            "file_operations": True,
            "image_generation": True,
        }
    
    def disable_capability(self, capability: str):
        """Disable a capability due to dependency failure."""
        if capability in self.capabilities:
            self.capabilities[capability] = False
            print(f"Capability '{capability}' disabled (degraded mode)")
    
    def enable_capability(self, capability: str):
        """Re-enable a capability after dependency recovers."""
        if capability in self.capabilities:
            self.capabilities[capability] = True
    
    def get_available_tools(self, all_tools: List[dict]) -> List[dict]:
        """Filter tools based on available capabilities."""
        capability_to_tool = {
            "web_search": "web_search",
            "code_execution": "python_interpreter",
            "database_access": "query_database",
            "file_operations": ["read_file", "write_file"],
            "image_generation": "generate_image",
        }
        
        disabled_tools = set()
        for cap, tool_names in capability_to_tool.items():
            if not self.capabilities[cap]:
                if isinstance(tool_names, list):
                    disabled_tools.update(tool_names)
                else:
                    disabled_tools.add(tool_names)
        
        return [
            tool for tool in all_tools
            if tool["name"] not in disabled_tools
        ]
    
    def get_system_prompt_addendum(self) -> str:
        """Get system prompt modifications based on degradation state."""
        if all(self.capabilities.values()):
            return ""
        
        disabled = [k for k, v in self.capabilities.items() if not v]
        return (
            f"\n\nNOTE: The following capabilities are currently unavailable: "
            f"{', '.join(disabled)}. Do not attempt to use tools related to "
            f"these capabilities. Work with what's available."
        )
```

## 6. Redundant Model Providers

### 6.1 Multi-Provider Strategy

```python
"""Multi-provider model redundancy for agent reliability."""

from typing import Optional, Dict, Any, List
import random

class ModelProvider:
    """A model provider with fallback models."""
    
    def __init__(self, name: str, models: Dict[str, Any],
                 primary_model: str, fallback_models: List[str] = None):
        self.name = name
        self.models = models
        self.primary_model = primary_model
        self.fallback_models = fallback_models or []
        self.circuit_breaker = CircuitBreaker(name=name)

class MultiProviderRouter:
    """
    Routes LLM calls across multiple providers for redundancy.
    If one provider fails, automatically tries the next.
    """
    
    def __init__(self, providers: List[ModelProvider]):
        self.providers = providers
        self.current_provider_idx = 0
    
    def call(self, messages: list, **kwargs) -> str:
        """Call LLM with automatic failover across providers."""
        errors = []
        
        # Try providers in order, cycling through them
        for _ in range(len(self.providers)):
            provider = self.providers[self.current_provider_idx]
            self.current_provider_idx = (self.current_provider_idx + 1) % len(self.providers)
            
            try:
                return provider.circuit_breaker.call(
                    func=self._make_provider_call,
                    provider=provider,
                    messages=messages,
                    **kwargs,
                )
            except Exception as e:
                errors.append(f"{provider.name}: {e}")
                continue
        
        raise RuntimeError(f"All providers failed: {'; '.join(errors)}")
    
    def _make_provider_call(self, provider: ModelProvider, messages: list, **kwargs) -> str:
        """Make the actual provider call."""
        # In production, this would call the provider's API
        raise NotImplementedError
```

## 7. Canary Deployments for Agents

### 7.1 Canary Release Strategy

```python
"""Canary deployment strategy for agent updates."""

import random
from typing import Optional, Callable

class AgentCanaryDeployment:
    """
    Manages canary deployments for agent versions.
    Gradually shifts traffic from current to new version,
    with automatic rollback on failure detection.
    """
    
    def __init__(
        self,
        agent_id: str,
        current_version_fn: Callable,
        new_version_fn: Callable,
        canary_percentages: List[float] = None,
        evaluation_fn: Optional[Callable] = None,
        auto_rollback: bool = True,
    ):
        self.agent_id = agent_id
        self.current = current_version_fn
        self.new = new_version_fn
        self.canary_percentages = canary_percentages or [1, 5, 10, 25, 50, 100]
        self.evaluation_fn = evaluation_fn
        self.auto_rollback = auto_rollback
        
        self.current_percentage = 0
        self.stage = 0
        self.metrics = {"current": [], "canary": []}
        self.rolled_back = False
    
    def promote(self) -> bool:
        """Promote to the next canary stage."""
        if self.stage >= len(self.canary_percentages):
            print(f"Canary fully rolled out (100% new version)")
            return True
        
        self.current_percentage = self.canary_percentages[self.stage]
        print(f"Promoting canary to {self.current_percentage}%")
        self.stage += 1
        return False
    
    def rollback(self):
        """Rollback to the current (previous) version."""
        print(f"ROLLING BACK canary for agent {self.agent_id}")
        self.current_percentage = 0
        self.stage = 0
        self.rolled_back = True
    
    def select_version(self, user_id: str) -> tuple:
        """Select which version a user gets based on consistent hashing."""
        if self.rolled_back or self.current_percentage >= 100:
            return ("current", self.current)
        
        # Consistent assignment
        hash_val = hash(f"{user_id}_{self.agent_id}") % 100
        if hash_val < self.current_percentage:
            return ("canary", self.new)
        return ("current", self.current)
    
    def record_metric(self, version: str, metric_name: str, value: float):
        """Record a metric for monitoring."""
        self.metrics[version].append({
            "metric": metric_name,
            "value": value,
            "timestamp": time.time(),
        })
    
    def check_rollback_condition(self) -> bool:
        """Check if rollback is needed based on canary metrics."""
        if not self.evaluation_fn:
            return False
        
        canary_metrics = self.metrics.get("canary", [])
        current_metrics = self.metrics.get("current", [])
        
        if len(canary_metrics) < 10 or len(current_metrics) < 10:
            return False  # Need more data
        
        canary_score = self.evaluation_fn(canary_metrics)
        current_score = self.evaluation_fn(current_metrics)
        
        # Rollback if canary is significantly worse
        if canary_score < current_score * 0.8:
            print(f"Canary score {canary_score:.3f} vs current {current_score:.3f} — ROLLING BACK")
            return True
        
        return False
```

## 8. Production Reliability Runbook

### 8.1 Incident Response

```python
"""Production runbook for agent reliability incidents."""

class AgentReliabilityRunbook:
    """
    Playbook for responding to agent reliability incidents.
    """
    
    RESPONSES = {
        "infinite_loop": {
            "detection": "Steps > 20 with no progress, repetitive tool calls",
            "action": [
                "1. KILL the agent session immediately",
                "2. Add step limit (max_steps=25) to agent configuration",
                "3. Add loop detection (check for repeated tool call patterns)",
                "4. Implement early termination when no progress detected",
                "5. Review agent prompt for ambiguity",
            ],
        },
        "hallucination_cascade": {
            "detection": "Subsequent LLM calls build on hallucinated facts",
            "action": [
                "1. IDENTIFY the first hallucination in the trace",
                "2. ADD fact-checking step after each LLM call (for critical agents)",
                "3. IMPLEMENT retrieval-augmented generation (RAG) for grounding",
                "4. ADD confidence threshold — re-verify if below threshold",
                "5. CONSIDER structured output formats that reduce hallucination",
            ],
        },
        "tool_failure": {
            "detection": "Tool API returns error, timeout, or unexpected response",
            "action": [
                "1. CHECK tool API status and authentication",
                "2. ADD retry logic with exponential backoff",
                "3. IMPLEMENT circuit breaker for the failing tool",
                "4. PROVIDE better error messages in tool response",
                "5. CONSIDER alternative tool implementations",
            ],
        },
        "rate_limited": {
            "detection": "HTTP 429 responses from LLM or tool API",
            "action": [
                "1. CHECK current rate limit usage",
                "2. ADD rate limit awareness in agent (pre-check before calling)",
                "3. IMPLEMENT request queuing and batching",
                "4. INCREASE rate limit (if possible) or reduce concurrency",
                "5. ADD fallback to alternative provider",
            ],
        },
        "context_overflow": {
            "detection": "Token count exceeds context window",
            "action": [
                "1. IMPLEMENT conversation summarization",
                "2. TRIM older messages from context",
                "3. USE sliding window of recent context",
                "4. PRIORITIZE important information in prompt",
                "5. SWITCH to model with larger context window",
            ],
        },
    }
    
    @classmethod
    def get_response(cls, failure_mode: str) -> dict:
        """Get the incident response for a failure mode."""
        response = cls.RESPONSES.get(failure_mode)
        if not response:
            return {
                "detection": "Unknown failure mode",
                "action": [
                    "1. COLLECT full trace and logs",
                    "2. Triage based on blast radius (number of users affected)",
                    "3. ROLLBACK if recently deployed",
                    "4. ESCALATE to agent-eng team",
                ],
            }
        return response
    
    @classmethod
    def print_response(cls, failure_mode: str):
        """Print the runbook response for a failure mode."""
        response = cls.get_response(failure_mode)
        print(f"=== Incident: {failure_mode} ===")
        print(f"Detection: {response['detection']}")
        print("Actions:")
        for action in response['action']:
            print(f"  {action}")
```

## 9. Conclusion

Building reliable agent systems requires a defense-in-depth approach:

1. **Detect failures early** — loop detection, cost monitoring, context overflow detection
2. **Retry intelligently** — exponential backoff with jitter, retry budgets
3. **Isolate failures** — circuit breakers prevent cascading failures
4. **Set timeouts at every level** — LLM calls, tool calls, steps, sessions
5. **Degrade gracefully** — fallback agents, reduced capabilities instead of full failure
6. **Redundancy** — multiple model providers, multiple tool implementations
7. **Deploy safely** — canary releases, automatic rollback, A/B testing

The most resilient agent systems assume everything will fail and are designed to handle failure at every level. They are not afraid of failure — they are prepared for it.

---

*Next: [08-Agent-Registry-and-Versioning.md](08-Agent-Registry-and-Versioning.md) — Agent versioning, registry, and deployment strategies.*

---
**See also:**
- [07 — AI Coding Agent Sustainability: Managing Hidden Costs, Code Quality Decay, and Developer Burnout](33-AI-Native-Software-Development/07-AI-Coding-Agent-Sustainability.md)
- [48 — MCP Cloud Infrastructure & Agent-as-a-Service: The Production Layer of Agentic AI](48-MCP-Cloud-Infrastructure-Agent-as-a-Service/01-Overview.md)
- [03 — Agent-Native Orchestration: LangGraph, Conductor, Mistral Workflows, Mcp-Agent](31-AI-Workflow-Orchestration-and-Durable-Execution/03-Agent-Native-Orchestration.md)
