# 03 — Agent Tracing and Observability with OpenTelemetry

## 1. Introduction to Distributed Tracing for Agent Systems

### 1.1 Why Tracing Matters for Agents

Traditional request/response tracing models break down for autonomous agents. A single user request to an agent may result in:

- 5–50 LLM calls (each with thousands of tokens)
- 10–200 tool invocations (APIs, databases, filesystem operations)
- Complex reasoning chains with backtracking and replanning
- Multi-step state transformations across context windows

Without distributed tracing, debugging an agent failure is like trying to debug a crashed program with only the final error message — you know something went wrong, but you have no idea where, why, or how to fix it.

### 1.2 What Agent Tracing Must Capture

A comprehensive agent trace captures:

```
Trace (Agent Session)
├── Span: Agent Run (root)
│   ├── Span: LLM Call (system prompt + user message)
│   ├── Span: Reasoning / Planning
│   │   ├── Span: LLM Call (thinking)
│   │   └── Span: Thought Process
│   ├── Span: Tool Decision
│   │   ├── Span: LLM Call (tool selection)
│   │   └── Span: Tool Invocation
│   │       ├── Span: Tool Input Preparation
│   │       ├── Span: External API Call
│   │       └── Span: Tool Output Processing
│   ├── Span: Observation Processing
│   │   ├── Span: LLM Call (analyzing tool output)
│   │   └── Span: Memory Update
│   ├── Span: Next Action Decision
│   │   ├── Span: LLM Call (deciding next step)
│   │   └── Span: Continue or Finish
│   └── Span: Final Response Generation
│       └── Span: LLM Call (final answer)
```

Each span contains:
- **Operation name** (what happened)
- **Timestamps** (start and end time, duration)
- **Attributes** (structured metadata)
- **Events** (log messages within the span)
- **Status** (ok, error, unset)
- **Context** (trace ID, span ID, parent span ID)

## 2. OpenTelemetry Primer for Agent Systems

### 2.1 What is OpenTelemetry?

OpenTelemetry (OTel) is the industry standard for observability instrumentation. It provides:
- **APIs and SDKs** for generating traces, metrics, and logs
- **Protocols** (OTLP) for transmitting observability data
- **Semantic conventions** for standardizing attribute names
- **Instrumentation libraries** that auto-instrument popular frameworks

### 2.2 OTel Concepts for Agent Tracing

| Concept | Agent Tracing Analog |
|---------|---------------------|
| **Tracer** | Instrumentation entry point, creates spans |
| **Span** | A single operation in the agent loop (LLM call, tool invocation, reasoning step) |
| **Trace** | The complete tree of spans for one agent execution |
| **Span Attributes** | Key-value pairs describing the span (model name, tool name, token count, cost) |
| **Span Events** | Timestamped log messages within a span (warning, error, info) |
| **Span Links** | Connections between spans across trace boundaries (multi-agent handoffs) |
| **Context Propagation** | Carrying trace context across async boundaries, threads, and network calls |
| **Sampler** | Decides which traces to retain |
| **SpanProcessor** | Processes spans on start/end (batch export, filtering) |
| **Exporter** | Sends spans to a backend (console, OTLP, Jaeger, Zipkin) |

### 2.3 Setting Up OpenTelemetry for Agent Tracing

```python
"""Basic OpenTelemetry setup for agent tracing."""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes

# Create resource identifying your service
resource = Resource(attributes={
    ResourceAttributes.SERVICE_NAME: "agent-runtime",
    ResourceAttributes.SERVICE_VERSION: "1.0.0",
    ResourceAttributes.DEPLOYMENT_ENVIRONMENT: "production",
    "agent.framework": "langgraph",
    "agent.version": "2.1.0",
})

# Configure tracer provider
provider = TracerProvider(resource=resource)

# Add span processors
# Console exporter for development
provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

# OTLP exporter for production (send to collector/backend)
otlp_exporter = OTLPSpanExporter(
    endpoint="http://otel-collector:4317",
    insecure=True,  # Use TLS in production
)
provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

# Set the global tracer provider
trace.set_tracer_provider(provider)

# Get a tracer for your agent
tracer = trace.get_tracer(__name__, "1.0.0")
```

## 3. Instrumenting Agent Components

### 3.1 LLM Call Instrumentation

Every LLM call in an agent loop should be traced with rich attributes capturing model details, token usage, and timing.

```python
"""Instrumentation for LLM calls within an agent loop."""

import time
from typing import Optional
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

tracer = trace.get_tracer(__name__)

def trace_llm_call(
    model: str,
    messages: list[dict],
    response: Optional[str] = None,
    token_usage: Optional[dict] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    error: Optional[Exception] = None,
    agent_step: int = 0,
) -> None:
    """Trace an LLM call with comprehensive attributes."""
    with tracer.start_as_current_span(
        f"llm.call.{model.split('/')[-1]}",
        kind=trace.SpanKind.CLIENT,
    ) as span:
        # Set span attributes
        span.set_attribute("llm.model", model)
        span.set_attribute("llm.temperature", temperature)
        span.set_attribute("llm.max_tokens", max_tokens)
        span.set_attribute("llm.system_prompt_length", len(messages[0]["content"]) if messages else 0)
        span.set_attribute("llm.messages_count", len(messages))
        span.set_attribute("llm.agent_step", agent_step)
        
        # Calculate total input tokens (approximate)
        input_tokens = sum(len(m.get("content", "").split()) for m in messages)
        span.set_attribute("llm.input_tokens_estimated", input_tokens)
        
        if token_usage:
            span.set_attribute("llm.prompt_tokens", token_usage.get("prompt_tokens", 0))
            span.set_attribute("llm.completion_tokens", token_usage.get("completion_tokens", 0))
            span.set_attribute("llm.total_tokens", token_usage.get("total_tokens", 0))
            
            # Cost calculation (approximate, update with your pricing)
            cost_per_1k_input = 0.0025  # gpt-4o pricing
            cost_per_1k_output = 0.01
            cost = (
                token_usage.get("prompt_tokens", 0) / 1000 * cost_per_1k_input +
                token_usage.get("completion_tokens", 0) / 1000 * cost_per_1k_output
            )
            span.set_attribute("llm.cost_usd", cost)
        
        if response:
            span.set_attribute("llm.response_length", len(response))
            span.set_attribute("llm.response_tokens_estimated", len(response.split()))
        
        if error:
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.record_exception(error)
        else:
            span.set_status(Status(StatusCode.OK))
```

### 3.2 Tool Call Instrumentation

Tool calls are where agents interact with the real world. Each tool invocation must be traced with input parameters, response, duration, and error information.

```python
"""Instrumentation for tool calls within an agent loop."""

import time
import json
from functools import wraps
from typing import Any, Callable
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

tracer = trace.get_tracer(__name__)

def instrumented_tool(tool_name: str):
    """Decorator that instruments a tool with OpenTelemetry tracing."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with tracer.start_as_current_span(
                f"tool.{tool_name}",
                kind=trace.SpanKind.CLIENT,
            ) as span:
                span.set_attribute("tool.name", tool_name)
                span.set_attribute("tool.args", json.dumps(kwargs, default=str)[:1000])
                
                # Truncate args for cardinality control
                if len(json.dumps(kwargs, default=str)) > 1000:
                    span.set_attribute("tool.args_truncated", True)
                
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    span.set_attribute("tool.duration_ms", duration * 1000)
                    span.set_attribute("tool.success", True)
                    
                    # Summarize result to avoid huge spans
                    result_str = str(result)
                    span.set_attribute("tool.result_length", len(result_str))
                    if len(result_str) > 500:
                        span.set_attribute("tool.result_preview", result_str[:500] + "...")
                    else:
                        span.set_attribute("tool.result_preview", result_str)
                    
                    span.set_status(Status(StatusCode.OK))
                    return result
                    
                except Exception as e:
                    duration = time.time() - start_time
                    span.set_attribute("tool.duration_ms", duration * 1000)
                    span.set_attribute("tool.success", False)
                    span.set_attribute("tool.error_type", type(e).__name__)
                    span.set_attribute("tool.error_message", str(e))
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
                    
        return wrapper
    return decorator


# Usage example
class MyAgent:
    def __init__(self):
        self.tools = {
            "web_search": self.web_search,
            "database_query": self.database_query,
            "code_executor": self.code_executor,
        }
    
    @instrumented_tool("web_search")
    def web_search(self, query: str, max_results: int = 5) -> str:
        """Search the web for information."""
        # Actual implementation
        return f"Search results for '{query}'..."
    
    @instrumented_tool("database_query")
    def database_query(self, sql: str) -> str:
        """Execute a database query."""
        # Actual implementation
        return f"Query results..."
```

### 3.3 Agent Reasoning Loop Instrumentation

The agent's main reasoning loop — the planning, thinking, and decision-making steps — forms the backbone of the trace.

```python
"""Instrumentation for the agent reasoning loop."""

import time
import uuid
from typing import Any, Optional
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from opentelemetry.context import attach, detach

tracer = trace.get_tracer(__name__)

class AgentTracer:
    """
    Manages tracing for a single agent execution session.
    Creates a root span representing the entire agent run
    and provides context for child spans.
    """
    
    def __init__(self, agent_id: str, agent_version: str, session_id: Optional[str] = None):
        self.agent_id = agent_id
        self.agent_version = agent_version
        self.session_id = session_id or str(uuid.uuid4())
        self.root_span = None
        self.current_step = 0
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.tool_calls = 0
        self.errors = 0
    
    def start_trace(self, input_text: str, user_id: str = "anonymous") -> None:
        """Start a new agent execution trace."""
        self.root_span = tracer.start_span(
            f"agent.{self.agent_id}",
            kind=trace.SpanKind.SERVER,
            attributes={
                "agent.id": self.agent_id,
                "agent.version": self.agent_version,
                "agent.session_id": self.session_id,
                "agent.user_id": user_id,
                "agent.input_length": len(input_text),
                "agent.input_preview": input_text[:200],
            }
        )
        self.root_span_context = trace.set_span_in_context(self.root_span)
        attach(self.root_span_context)
    
    def trace_reasoning_step(self, thought: str) -> trace.Span:
        """Trace a single reasoning/thinking step."""
        self.current_step += 1
        span = tracer.start_span(
            f"reasoning.step.{self.current_step}",
            context=self.root_span_context,
            attributes={
                "reasoning.step": self.current_step,
                "reasoning.thought_length": len(thought),
                "reasoning.thought_preview": thought[:200],
            }
        )
        return span
    
    def trace_llm_call(
        self,
        model: str,
        messages: list[dict],
        response: str,
        usage: Optional[dict] = None,
        parent_span: Optional[trace.Span] = None,
    ) -> None:
        """Trace an LLM call within the agent loop."""
        context = trace.set_span_in_context(parent_span or self.root_span)
        with tracer.start_span(f"llm.{model}", context=context) as span:
            span.set_attribute("llm.model", model)
            span.set_attribute("llm.agent_step", self.current_step)
            span.set_attribute("llm.messages", len(messages))
            
            if usage:
                span.set_attribute("llm.prompt_tokens", usage.get("prompt_tokens", 0))
                span.set_attribute("llm.completion_tokens", usage.get("completion_tokens", 0))
                span.set_attribute("llm.total_tokens", usage.get("total_tokens", 0))
                self.token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                self.token_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                self.token_usage["total_tokens"] += usage.get("total_tokens", 0)
    
    def trace_tool_call(self, tool_name: str, args: dict, result: Any, duration_ms: float, error: Optional[Exception] = None) -> None:
        """Trace a tool invocation within the agent loop."""
        self.tool_calls += 1
        with tracer.start_span(f"tool.{tool_name}") as span:
            span.set_attribute("tool.name", tool_name)
            span.set_attribute("tool.duration_ms", duration_ms)
            span.set_attribute("tool.step", self.current_step)
            
            if error:
                self.errors += 1
                span.set_status(Status(StatusCode.ERROR, str(error)))
                span.record_exception(error)
            else:
                span.set_status(Status(StatusCode.OK))
    
    def end_trace(self, final_output: str, error: Optional[Exception] = None) -> None:
        """End the agent execution trace with final statistics."""
        if self.root_span:
            self.root_span.set_attribute("agent.total_steps", self.current_step)
            self.root_span.set_attribute("agent.total_tool_calls", self.tool_calls)
            self.root_span.set_attribute("agent.total_errors", self.errors)
            self.root_span.set_attribute("agent.total_prompt_tokens", self.token_usage["prompt_tokens"])
            self.root_span.set_attribute("agent.total_completion_tokens", self.token_usage["completion_tokens"])
            self.root_span.set_attribute("agent.total_tokens", self.token_usage["total_tokens"])
            self.root_span.set_attribute("agent.output_length", len(final_output))
            
            if error:
                self.root_span.set_status(Status(StatusCode.ERROR, str(error)))
                self.root_span.record_exception(error)
            else:
                self.root_span.set_status(Status(StatusCode.OK))
            
            self.root_span.end()
            detach(self.root_span_context)
```

### 3.4 Complete Agent with Full Tracing

```python
"""Example of a fully traced agent using the AgentTracer."""

import time
from typing import Any
from opentelemetry import trace

class TracedAgent:
    """An agent with full OpenTelemetry instrumentation."""
    
    def __init__(self, agent_id: str, agent_version: str, llm_client, tools: dict):
        self.agent_id = agent_id
        self.agent_version = agent_version
        self.llm = llm_client
        self.tools = tools
        self.tracer_obj = None
    
    def run(self, input_text: str, user_id: str = "anonymous", max_steps: int = 25) -> dict:
        """Execute the agent with full tracing."""
        
        self.tracer_obj = AgentTracer(self.agent_id, self.agent_version)
        self.tracer_obj.start_trace(input_text, user_id)
        
        try:
            messages = [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": input_text},
            ]
            
            for step in range(max_steps):
                # === LLM Call ===
                llm_start = time.time()
                response = self.llm.chat(messages=messages)
                llm_duration = time.time() - llm_start
                
                self.tracer_obj.trace_llm_call(
                    model=self.llm.model,
                    messages=messages,
                    response=response.content,
                    usage=getattr(response, 'usage', None),
                )
                
                # Parse response for reasoning and action
                reasoning, action = self._parse_response(response.content)
                
                # Trace reasoning step
                if reasoning:
                    thought_span = self.tracer_obj.trace_reasoning_step(reasoning)
                    thought_span.end()
                
                # If no action needed, we're done
                if not action:
                    self.tracer_obj.end_trace(response.content)
                    return {
                        "output": response.content,
                        "steps": step + 1,
                        "tokens": self.tracer_obj.token_usage,
                    }
                
                # === Tool Call ===
                tool_name = action.get("tool")
                tool_args = action.get("args", {})
                
                if tool_name in self.tools:
                    tool_start = time.time()
                    try:
                        tool_result = self.tools[tool_name](**tool_args)
                        tool_duration = time.time() - tool_start
                        self.tracer_obj.trace_tool_call(tool_name, tool_args, tool_result, tool_duration * 1000)
                    except Exception as e:
                        tool_duration = time.time() - tool_start
                        self.tracer_obj.trace_tool_call(tool_name, tool_args, None, tool_duration * 1000, error=e)
                        tool_result = f"Error: {str(e)}"
                    
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "tool", "content": str(tool_result), "tool_call_id": action.get("id", "")})
                else:
                    error_msg = f"Unknown tool: {tool_name}"
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "tool", "content": error_msg})
            
            # Max steps reached
            final = "I've reached the maximum number of steps without completing the task."
            self.tracer_obj.end_trace(final)
            return {"output": final, "steps": max_steps, "tokens": self.tracer_obj.token_usage}
            
        except Exception as e:
            self.tracer_obj.end_trace("", error=e)
            raise
    
    def _system_prompt(self) -> str:
        return "You are a helpful AI assistant with access to tools."
    
    def _parse_response(self, content: str) -> tuple:
        """Parse agent response into reasoning and action."""
        # Implement your parsing logic
        reasoning = None
        action = None
        return reasoning, action
```

## 4. Multi-Agent Trace Correlation

### 4.1 The Challenge of Multi-Agent Tracing

In multi-agent systems (e.g., CrewAI, AutoGen, LangGraph supervisor/worker patterns), a single user request may involve multiple agents working in sequence or in parallel. Each agent may run on different processes, different machines, or even different services.

Correlating these traces into a single end-to-end view requires:

1. **Trace ID Propagation**: All spans across all agents must share the same trace ID
2. **Context Propagation**: Parent-child relationships must be preserved across agent boundaries
3. **Linkage**: When agents run asynchronously, span links connect related but non-nested spans

### 4.2 Context Propagation Across Agents

```python
"""Context propagation for multi-agent tracing."""

import uuid
from opentelemetry import trace
from opentelemetry.propagate import inject, extract
from opentelemetry.trace.propagation import set_span_in_context

tracer = trace.get_tracer(__name__)

class SupervisorAgent:
    """Supervisor agent that delegates to worker agents with context propagation."""
    
    def __init__(self, workers: dict):
        self.workers = workers
    
    def run(self, task: str) -> str:
        """Execute task by coordinating workers."""
        
        # Start the top-level trace
        trace_id = str(uuid.uuid4())
        
        with tracer.start_as_current_span("supervisor.run") as root_span:
            root_span.set_attribute("supervisor.task", task[:200])
            
            # Plan subtasks
            subtasks = self._plan(task)
            root_span.set_attribute("supervisor.subtasks", len(subtasks))
            
            results = {}
            for i, (agent_name, subtask) in enumerate(subtasks):
                # Create a child span for delegation
                with tracer.start_as_current_span(
                    f"supervisor.delegate.{agent_name}"
                ) as delegate_span:
                    delegate_span.set_attribute("supervisor.worker", agent_name)
                    delegate_span.set_attribute("supervisor.subtask", subtask[:200])
                    
                    # Propagate context to worker
                    # In a real system, this would be via headers/message metadata
                    carrier = {}
                    inject(set_span_in_context(delegate_span))
                    
                    # Call worker (in-process for example, but same concept for distributed)
                    worker = self.workers[agent_name]
                    result = worker.run(subtask, parent_context=trace.get_current_span().__class__)
                    
                    results[agent_name] = result
                    delegate_span.set_attribute("supervisor.result_length", len(result))
            
            # Synthesize final response
            final = self._synthesize(results)
            root_span.set_attribute("supervisor.final_length", len(final))
            return final
    
    def _plan(self, task: str) -> list:
        """Plan subtasks for workers."""
        return [
            ("research_agent", f"Research: {task}"),
            ("analysis_agent", f"Analyze: {task}"),
        ]
    
    def _synthesize(self, results: dict) -> str:
        return "\n".join(f"{k}: {v}" for k, v in results.items())


class WorkerAgent:
    """Worker agent that participates in a distributed trace."""
    
    def __init__(self, name: str, llm_client):
        self.name = name
        self.llm = llm_client
    
    def run(self, task: str, parent_context=None) -> str:
        """Execute task with inherited tracing context."""
        
        # Use the propagated context if available
        ctx = parent_context or trace.get_current_span().__class__
        
        with tracer.start_as_current_span(
            f"worker.{self.name}",
            context=ctx,
        ) as span:
            span.set_attribute("worker.name", self.name)
            span.set_attribute("worker.task", task[:200])
            
            # LLM call within worker
            with tracer.start_span(f"worker.{self.name}.llm", context=trace.set_span_in_context(span)):
                response = self.llm.chat(messages=[{"role": "user", "content": task}])
                span.set_attribute("worker.response_length", len(response.content))
                return response.content
```

### 4.3 Span Links for Async Multi-Agent Handoffs

When agents operate asynchronously (publish-subscribe, message queues), span links connect related spans across trace boundaries:

```python
"""Using span links for async agent handoffs."""

from opentelemetry import trace
from opentelemetry.trace import Link, NonRecordingSpan, SpanContext, TraceFlags

tracer = trace.get_tracer(__name__)

# When publishing a task to a queue:
def publish_agent_task(task_data: dict) -> str:
    """Publish a task with trace context for the consumer."""
    span = trace.get_current_span()
    span_context = span.get_span_context()
    
    # Serialize context into the message
    message = {
        "task": task_data,
        "traceparent": f"00-{span_context.trace_id:032x}-{span_context.span_id:016x}-01",
    }
    # Send to queue...
    message_id = "msg-123"
    return message_id


# When consuming the task:
def process_agent_task(message: dict) -> None:
    """Process a task, linking back to the producer trace."""
    # Extract context from message
    from opentelemetry.propagate import extract
    carrier = {"traceparent": message["traceparent"]}
    ctx = extract(carrier)
    
    with tracer.start_as_current_span(
        "agent.task.process",
        context=ctx,
        links=[Link(trace.get_current_span().get_span_context())],
    ) as span:
        span.set_attribute("agent.task", str(message["task"]))
        # Process the task...
```

## 5. Trace Sampling Strategies

### 5.1 Why Sampling is Essential

At scale, agents generate enormous volumes of trace data:
- 100 agents × 1000 tasks/day × 50 spans/task = 5 million spans/day
- At 1KB per span = 5GB/day of tracing data
- Storage costs: ~$0.02/GB/month for object storage, $0.10/GB/month for queryable storage

Without sampling, trace storage costs can exceed LLM API costs.

### 5.2 Head-Based Sampling

Decide at the start of a trace whether to sample it. Simple but cannot guarantee capturing rare events.

```python
"""Head-based sampling for agent traces."""

import random
from opentelemetry.sdk.trace.sampling import Sampler, Decision, SamplingResult
from opentelemetry import trace

class AgentHeadSampler(Sampler):
    """
    Head-based sampler that adjusts rate based on agent and user criteria.
    """
    
    def __init__(self, base_rate: float = 0.1):
        self.base_rate = base_rate
    
    def should_sample(
        self,
        parent_context,
        trace_id,
        name,
        kind,
        attributes,
        links,
        trace_state,
    ) -> SamplingResult:
        # Always sample errors and high-value agents
        agent_id = attributes.get("agent.id", "")
        
        # 100% sample for critical agents
        if agent_id in ("payment_agent", "compliance_agent"):
            return SamplingResult(Decision.RECORD_AND_SAMPLE)
        
        # 50% for new agent versions (recently deployed)
        if attributes.get("agent.version", "").startswith("2."):
            return SamplingResult(
                Decision.RECORD_AND_SAMPLE if random.random() < 0.5 else Decision.DROP
            )
        
        # Base rate for everything else
        return SamplingResult(
            Decision.RECORD_AND_SAMPLE if random.random() < self.base_rate else Decision.DROP
        )
    
    def get_description(self) -> str:
        return f"AgentHeadSampler(base_rate={self.base_rate})"


# Usage
from opentelemetry.sdk.trace import TracerProvider

provider = TracerProvider(sampler=AgentHeadSampler(base_rate=0.1))
```

### 5.3 Tail-Based Sampling

Make sampling decisions after the trace completes, based on the attributes and status of all spans.

```python
"""Tail-based sampling using a custom SpanProcessor."""

from collections import defaultdict
from typing import Optional
from opentelemetry.sdk.trace import Span
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import StatusCode

class TailBasedSampler(SpanExporter):
    """
    Tail-based sampler that decides whether to export a trace after all its spans
    have been collected. Uses a buffer to hold spans until the trace is complete.
    """
    
    def __init__(self, downstream_exporter: SpanExporter, error_rate: float = 1.0, slow_threshold_ms: float = 30000):
        self.downstream = downstream_exporter
        self.error_rate = error_rate
        self.slow_threshold_ms = slow_threshold_ms
        self.buffer = defaultdict(list)  # trace_id -> list of spans
        self.buffer_limit = 10000  # max traces in buffer
        self.max_trace_duration_ms = 120000  # 2 minutes max wait
    
    def export(self, spans) -> SpanExportResult:
        """Buffer spans and decide on export when trace is complete."""
        for span in spans:
            trace_id = span.context.trace_id
            self.buffer[trace_id].append(span)
        
        # Process complete traces
        complete_trace_ids = []
        for trace_id, trace_spans in self.buffer.items():
            if self._is_trace_complete(trace_spans):
                if self._should_sample(trace_spans):
                    self.downstream.export(trace_spans)
                complete_trace_ids.append(trace_id)
        
        # Clean up processed traces
        for trace_id in complete_trace_ids:
            del self.buffer[trace_id]
        
        # Evict oldest traces if buffer is too large
        if len(self.buffer) > self.buffer_limit:
            excess = len(self.buffer) - self.buffer_limit
            for trace_id in sorted(self.buffer.keys())[:excess]:
                # Export a sample of evicted traces
                if random.random() < 0.01:
                    self.downstream.export(self.buffer[trace_id])
                del self.buffer[trace_id]
        
        return SpanExportResult.SUCCESS
    
    def _is_trace_complete(self, spans: list) -> bool:
        """Check if a trace is complete (root span has ended)."""
        for span in spans:
            if span.parent is None:  # root span
                return True
        return False
    
    def _should_sample(self, spans: list) -> bool:
        """Decide whether to sample a complete trace."""
        has_error = any(
            span.status.status_code == StatusCode.ERROR for span in spans
        )
        is_slow = any(
            (span.end_time - span.start_time) / 1e6 > self.slow_threshold_ms
            for span in spans
        )
        has_high_cost = any(
            span.attributes.get("llm.total_tokens", 0) > 10000
            for span in spans
        )
        
        # Always sample traces with errors or high costs
        if has_error or has_high_cost:
            return True
        
        # Higher sample rate for slow traces
        if is_slow:
            return random.random() < 0.5
        
        # Base rate
        return random.random() < 0.1
    
    def shutdown(self):
        self.downstream.shutdown()
```

### 5.4 Sampling Strategy Recommendations

| Scenario | Recommended Strategy | Sample Rate |
|----------|-------------------|-------------|
| Development / Testing | Head-based, high rate | 100% |
| Production — Simple Agents | Head-based, low rate + error capture | 10% + 100% errors |
| Production — Complex Agents | Tail-based, dynamic | 25% + 100% errors/slow |
| Production — Compliance | No sampling | 100% |
| High-volume — Chat | Head-based, very low rate | 1% + 100% errors |
| Canary / New Version | Head-based, high rate | 100% during evaluation |

## 6. Trace Storage and Querying

### 6.1 Storage Backend Options

| Backend | Strengths | Weaknesses | Best For |
|---------|-----------|------------|----------|
| Jaeger | Purpose-built for traces, good UI | Limited query capabilities | Development, small-medium scale |
| Grafana Tempo | Scalable, integrated with Grafana | Complex setup | Production, existing Grafana stack |
| Elastic APM | Full observability (traces + logs) | Heavy, expensive at scale | ELK users, enterprise |
| SigNoz | OpenTelemetry-native, open-source | Smaller community | Self-hosted OTel |
| Datadog | Best-in-class UI, integrated | Very expensive | Enterprise, budget available |
| LangSmith/LangFuse | Agent-optimized | Vendor-specific schema | Agent-focused teams |

### 6.2 Trace Query Patterns for Debugging

```python
"""Example trace query patterns using the OpenTelemetry API and a trace backend."""

# Query pattern 1: Find all traces for a specific agent version
def find_agent_traces(agent_id: str, version: str, limit: int = 100):
    """Query traces matching agent and version."""
    # This would use your trace backend's query API
    pass


# Query pattern 2: Find traces with tool failures
def find_tool_failures(agent_id: str, tool_name: str, since_hours: int = 24):
    """Find traces where a specific tool failed."""
    pass


# Query pattern 3: Find expensive traces
def find_expensive_traces(min_cost: float = 1.0, since_hours: int = 24):
    """Find traces that cost more than a threshold."""
    pass


# Query pattern 4: Find slow traces
def find_slow_traces(p99_duration_ms: int = 30000, since_hours: int = 24):
    """Find traces exceeding p99 latency threshold."""
    pass
```

### 6.3 Trace Aggregation for Dashboards

```python
"""Aggregating trace data for operational dashboards."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional
from collections import defaultdict

@dataclass
class AgentTraceSummary:
    agent_id: str
    agent_version: str
    total_runs: int
    success_rate: float
    avg_duration_ms: float
    p50_duration_ms: float
    p95_duration_ms: float
    p99_duration_ms: float
    total_tokens: int
    total_cost: float
    avg_steps: float
    error_counts: dict  # error_type -> count
    tool_usage: dict  # tool_name -> count

def aggregate_traces(traces: list, time_window: timedelta = timedelta(hours=1)) -> List[AgentTraceSummary]:
    """Aggregate trace data into summaries for dashboard consumption."""
    
    from datetime import datetime
    
    # Group by agent_id + agent_version
    groups = defaultdict(list)
    for trace in traces:
        key = (trace["agent_id"], trace["agent_version"])
        groups[key].append(trace)
    
    summaries = []
    for (agent_id, agent_version), agent_traces in groups.items():
        durations = [t["duration_ms"] for t in agent_traces]
        durations.sort()
        
        total_runs = len(agent_traces)
        successes = sum(1 for t in agent_traces if t["status"] == "success")
        
        # Error counts
        error_counts = defaultdict(int)
        for t in agent_traces:
            if t.get("error_type"):
                error_counts[t["error_type"]] += 1
        
        # Tool usage
        tool_usage = defaultdict(int)
        for t in agent_traces:
            for tool in t.get("tools_used", []):
                tool_usage[tool] += 1
        
        # Percentile calculations
        def percentile(sorted_data, p):
            idx = int(len(sorted_data) * p / 100)
            return sorted_data[min(idx, len(sorted_data) - 1)]
        
        summaries.append(AgentTraceSummary(
            agent_id=agent_id,
            agent_version=agent_version,
            total_runs=total_runs,
            success_rate=successes / total_runs if total_runs > 0 else 0,
            avg_duration_ms=sum(durations) / len(durations) if durations else 0,
            p50_duration_ms=percentile(durations, 50) if durations else 0,
            p95_duration_ms=percentile(durations, 95) if durations else 0,
            p99_duration_ms=percentile(durations, 99) if durations else 0,
            total_tokens=sum(t.get("total_tokens", 0) for t in agent_traces),
            total_cost=sum(t.get("cost", 0) for t in agent_traces),
            avg_steps=sum(t.get("steps", 0) for t in agent_traces) / total_runs if total_runs > 0 else 0,
            error_counts=dict(error_counts),
            tool_usage=dict(tool_usage),
        ))
    
    return summaries
```

## 7. LangChain/LangGraph Instrumentation with OpenTelemetry

### 7.1 Using OpenLLMetry / Traceloop

OpenLLMetry provides automatic OpenTelemetry instrumentation for LangChain, LangGraph, LlamaIndex, and other agent frameworks.

```python
"""Automatic LangChain instrumentation with OpenLLMetry."""

from opentelemetry import trace
from traceloop.sdk import Traceloop
from traceloop.sdk.instruments import Instruments

# Initialize Traceloop with LangChain + LangGraph instrumentation
Traceloop.init(
    app_name="agent-prod",
    api_endpoint="http://otel-collector:4318",
    instruments={Instruments.LANGCHAIN, Instruments.LANGGRAPH},
)

# That's it! All LangChain/LangGraph operations are now traced.
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class AgentState(TypedDict):
    messages: List
    next_agent: str

# Build a LangGraph agent
workflow = StateGraph(AgentState)

def agent_node(state: AgentState):
    llm = ChatOpenAI(model="gpt-4o")
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response], "next_agent": "END"}

workflow.add_node("agent", agent_node)
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)

app = workflow.compile()

# This run will be fully traced with OpenTelemetry
result = app.invoke({"messages": [{"role": "user", "content": "Hello!"}]})
```

### 7.2 Manual LangChain Instrumentation with OpenTelemetry Callbacks

For more control, you can use OpenTelemetry directly with LangChain's callback system:

```python
"""Custom OpenTelemetry callback handler for LangChain."""

from typing import Any, Dict, List, Optional
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

tracer = trace.get_tracer(__name__)

class OpenTelemetryLangChainHandler(BaseCallbackHandler):
    """LangChain callback handler that creates OpenTelemetry spans."""
    
    def __init__(self):
        super().__init__()
        self.llm_spans: Dict[str, Any] = {}
        self.chain_spans: Dict[str, Any] = {}
        self.tool_spans: Dict[str, Any] = {}
        self.run_id_to_span_id: Dict[str, str] = {}
    
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs
    ) -> None:
        """Create span when LLM call starts."""
        run_id = str(kwargs.get("run_id", ""))
        span = tracer.start_span(
            f"llm.{serialized.get('name', 'unknown')}",
            kind=trace.SpanKind.CLIENT,
            attributes={
                "llm.model": serialized.get("kwargs", {}).get("model_name", "unknown"),
                "llm.prompts": len(prompts),
                "llm.temperature": serialized.get("kwargs", {}).get("temperature", 0),
            }
        )
        self.llm_spans[run_id] = span
        self.run_id_to_span_id[run_id] = f"llm.{run_id[:8]}"
    
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """End span when LLM call completes."""
        run_id = str(kwargs.get("run_id", ""))
        span = self.llm_spans.pop(run_id, None)
        if span:
            if response.llm_output:
                token_usage = response.llm_output.get("token_usage", {})
                span.set_attribute("llm.prompt_tokens", token_usage.get("prompt_tokens", 0))
                span.set_attribute("llm.completion_tokens", token_usage.get("completion_tokens", 0))
                span.set_attribute("llm.total_tokens", token_usage.get("total_tokens", 0))
            span.set_status(Status(StatusCode.OK))
            span.end()
    
    def on_llm_error(self, error: Exception, **kwargs) -> None:
        """Record error on LLM span."""
        run_id = str(kwargs.get("run_id", ""))
        span = self.llm_spans.pop(run_id, None)
        if span:
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.record_exception(error)
            span.end()
    
    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs
    ) -> None:
        """Create span when tool starts."""
        run_id = str(kwargs.get("run_id", ""))
        span = tracer.start_span(
            f"tool.{serialized.get('name', 'unknown')}",
            kind=trace.SpanKind.CLIENT,
            attributes={
                "tool.name": serialized.get("name", "unknown"),
                "tool.input": input_str[:500],
            }
        )
        self.tool_spans[run_id] = span
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        """End span when tool completes."""
        run_id = str(kwargs.get("run_id", ""))
        span = self.tool_spans.pop(run_id, None)
        if span:
            span.set_attribute("tool.output_length", len(output))
            span.set_status(Status(StatusCode.OK))
            span.end()
    
    def on_tool_error(self, error: Exception, **kwargs) -> None:
        """Record error on tool span."""
        run_id = str(kwargs.get("run_id", ""))
        span = self.tool_spans.pop(run_id, None)
        if span:
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.record_exception(error)
            span.end()
    
    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs
    ) -> None:
        """Create span when chain/agent starts."""
        run_id = str(kwargs.get("run_id", ""))
        span = tracer.start_span(
            f"agent.{serialized.get('name', 'unknown')}",
            kind=trace.SpanKind.SERVER,
            attributes={
                "agent.name": serialized.get("name", "unknown"),
                "agent.input": str(inputs)[:500],
            }
        )
        self.chain_spans[run_id] = span
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """End span when chain/agent completes."""
        run_id = str(kwargs.get("run_id", ""))
        span = self.chain_spans.pop(run_id, None)
        if span:
            span.set_attribute("agent.output_length", len(str(outputs)))
            span.set_status(Status(StatusCode.OK))
            span.end()
    
    def on_chain_error(self, error: Exception, **kwargs) -> None:
        """Record error on chain span."""
        run_id = str(kwargs.get("run_id", ""))
        span = self.chain_spans.pop(run_id, None)
        if span:
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.record_exception(error)
            span.end()


# Usage
handler = OpenTelemetryLangChainHandler()
llm = ChatOpenAI(model="gpt-4o", callbacks=[handler])
result = llm.invoke("Hello!")
```

## 8. Trace Visualization and Debugging

### 8.1 Critical Trace Views for Agent Debugging

When debugging an agent failure, engineers need these views:

1. **Waterfall View**: Timeline of all spans to identify slow operations or unexpected ordering
2. **Error Path Highlighting**: Tree view highlighting spans that errored, with their ancestors
3. **Token Heatmap**: Visual representation of token consumption across steps
4. **Cost Waterfall**: Cumulative cost attribution per span
5. **Decision Points**: Where the agent made choices (which tool to call, whether to continue)

### 8.2 Adding Custom Debug Events to Spans

```python
"""Adding structured debug events to agent spans."""

from opentelemetry import trace
from opentelemetry.trace import Span
import json

def add_agent_debug_event(span: Span, event_type: str, data: dict):
    """Add a structured debug event to a span."""
    span.add_event(
        name=event_type,
        attributes={
            "debug.type": event_type,
            "debug.data": json.dumps(data, default=str)[:2000],
        }
    )

# Usage in agent loop
with tracer.start_as_current_span("agent.reasoning") as span:
    add_agent_debug_event(span, "tool_selection", {
        "candidate_tools": ["web_search", "database_query"],
        "selected_tool": "web_search",
        "confidence": 0.85,
        "reasoning": "User asked about current events"
    })
    
    add_agent_debug_event(span, "context_usage", {
        "context_window_used": 45,  # percent
        "oldest_message_index": 3,
        "total_messages": 12,
        "estimated_tokens": 3400,
    })
    
    add_agent_debug_event(span, "token_usage", {
        "step": 3,
        "prompt_tokens": 1200,
        "completion_tokens": 450,
        "cumulative_tokens": 5200,
        "estimated_cost": 0.0075,
    })
```

## 9. Production Deployment Considerations

### 9.1 OpenTelemetry Collector Deployment

For production agent tracing, deploy an OpenTelemetry Collector as a gateway/sidecar:

```yaml
# otel-collector-config.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:
    timeout: 1s
    send_batch_size: 1024
  memory_limiter:
    check_interval: 1s
    limit_mib: 512
  attributes:
    actions:
      - key: environment
        value: production
        action: upsert
  filter:
    spans:
      exclude:
        match_type: regexp
        span_names:
          - "healthcheck.*"

exporters:
  otlp:
    endpoint: "jaeger:4317"
    tls:
      insecure: true
  prometheus:
    endpoint: "0.0.0.0:8889"
  logging:
    loglevel: warn

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, batch, attributes, filter]
      exporters: [otlp, logging]
    metrics:
      receivers: [otlp]
      processors: [batch]
      exporters: [prometheus]
```

### 9.2 Performance Overhead Considerations

Agent tracing adds overhead:
- **CPU**: Serialization, context propagation, export
- **Memory**: Span buffering, attribute storage
- **Network**: Trace export traffic
- **Latency**: Synchronous export (avoid — always use async/batch)

Typical overhead measurements:
- Well-instrumented agent: 1–3% CPU overhead
- Memory: 50–100MB for span buffer (configurable)
- Network: ~1–2KB per span, scalable with batching
- Latency impact: <1ms per span (async export)

## 10. Summary and Best Practices

### Tracing Best Practices Checklist

1. **Trace every LLM call** — model, tokens, cost, latency, system prompt length
2. **Trace every tool invocation** — tool name, input summary, output summary, duration, success/failure
3. **Trace reasoning steps** — thought process, context window usage, decision metadata
4. **Propagate context across async boundaries** — thread pools, message queues, HTTP calls
5. **Use consistent attribute naming** — follow OTel semantic conventions or define your own standard
6. **Implement sampling** — head-based for volume, tail-based for errors/slow traces
7. **Add structured debug events** — tool selection reasoning, context usage, decision criteria
8. **Set meaningful span status** — OK/ERROR with descriptive error messages
9. **Right-size span attributes** — truncate long values, avoid high-cardinality attributes
10. **Ship traces asynchronously** — batch export, never block agent execution on trace delivery

---

*Next: [04-Agent-Evaluation-and-Testing.md](04-Agent-Evaluation-and-Testing.md) — Comprehensive agent evaluation methodology and automated testing.*
