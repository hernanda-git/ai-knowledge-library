# 06 — Agent Logging and Monitoring

## 1. Introduction to Agent Observability

### 1.1 The Three Pillars of Observability for Agents

Observability for agent systems builds on the three classic pillars — logs, metrics, and traces — with agent-specific semantics for each:

1. **Logs**: Structured records of agent events (LLM calls, tool invocations, state transitions, errors)
2. **Metrics**: Aggregated measurements (latency, error rate, throughput, cost, token usage)
3. **Traces**: End-to-end records of agent sessions (covered in depth in document 03)

This document focuses on logging and metrics. For tracing, see [03-Agent-Tracing-and-Observability.md](03-Agent-Tracing-and-Observability.md).

### 1.2 Why Agent Monitoring Needs Specialized Approach

Traditional application monitoring assumes:
- Request/response pairs that complete in milliseconds to seconds
- Well-defined error modes (HTTP status codes, exceptions)
- Linear execution paths
- Stateless or simple state management

Agents violate all these assumptions:
- Sessions can last minutes to hours
- Error modes include novel patterns (hallucination, infinite loops, context overflow)
- Execution paths are non-deterministic and branching
- State is complex (context windows, tool outputs, memories)

## 2. Structured Agent Logging

### 2.1 Log Event Taxonomy

Agent logs should be structured around a clear event taxonomy:

```python
"""Agent event taxonomy and log schema."""

from typing import Optional, Any, Dict
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

class AgentEventType(str, Enum):
    """All event types in an agent system."""
    # Session lifecycle
    SESSION_START = "session.start"
    SESSION_END = "session.end"
    
    # LLM operations
    LLM_CALL_START = "llm.call.start"
    LLM_CALL_END = "llm.call.end"
    LLM_CALL_ERROR = "llm.call.error"
    
    # Tool operations
    TOOL_CALL_START = "tool.call.start"
    TOOL_CALL_END = "tool.call.end"
    TOOL_CALL_ERROR = "tool.call.error"
    
    # Reasoning
    REASONING_STEP = "reasoning.step"
    DECISION_MADE = "decision.made"
    
    # Agent lifecycle
    AGENT_THINK = "agent.think"
    AGENT_ACT = "agent.act"
    AGENT_OBSERVE = "agent.observe"
    
    # State management
    MEMORY_READ = "memory.read"
    MEMORY_WRITE = "memory.write"
    STATE_CHANGE = "state.change"
    
    # Errors and warnings
    ERROR = "error"
    WARNING = "warning"
    RETRY = "retry"
    TIMEOUT = "timeout"
    
    # User interaction
    USER_INPUT = "user.input"
    USER_FEEDBACK = "user.feedback"
    USER_CORRECTION = "user.correction"
    
    # System events
    RATE_LIMITED = "rate.limited"
    CONTEXT_OVERFLOW = "context.overflow"
    CIRCUIT_BREAKER = "circuit.breaker"

class AgentLogEvent(BaseModel):
    """Standard schema for an agent log event."""
    # Core identification
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    event_type: AgentEventType
    agent_id: str
    agent_version: str
    session_id: str
    trace_id: str
    
    # Actors
    user_id: Optional[str] = None
    actor_id: Optional[str] = None  # Which agent in multi-agent system
    
    # Event data
    event_data: Dict[str, Any] = Field(default_factory=dict)
    
    # Context
    parent_event_id: Optional[str] = None
    span_id: Optional[str] = None
    
    # Performance
    duration_ms: Optional[float] = None
    
    # Error
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    
    # Metadata
    environment: str = "production"
    region: str = "us-east-1"
    tags: Dict[str, str] = Field(default_factory=dict)
```

### 2.2 Structured Logging Implementation

```python
"""Structured logging for agent systems using Python's logging + JSON format."""

import json
import logging
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from pythonjsonlogger import jsonlogger

class AgentJSONFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter for agent logs with consistent schema."""
    
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        
        # Ensure consistent field ordering
        log_record['timestamp'] = datetime.utcnow().isoformat()
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        
        # Add default agent fields if not present
        if 'event_type' not in log_record:
            log_record['event_type'] = 'log'
        if 'session_id' not in log_record:
            log_record['session_id'] = ''
        if 'agent_id' not in log_record:
            log_record['agent_id'] = ''
        if 'trace_id' not in log_record:
            log_record['trace_id'] = ''


class AgentLogger:
    """
    Structured logger for agent systems.
    Produces JSON-formatted logs compatible with Loki, ELK, and other log aggregators.
    """
    
    def __init__(self, 
                 agent_id: str,
                 agent_version: str,
                 environment: str = "production",
                 log_level: str = "INFO",
                 json_output: bool = True):
        
        self.agent_id = agent_id
        self.agent_version = agent_version
        self.environment = environment
        
        # Configure Python logger
        self.logger = logging.getLogger(f"agent.{agent_id}")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.logger.handlers = []  # Clear default handlers
        
        # Add console handler with JSON formatting
        handler = logging.StreamHandler()
        if json_output:
            formatter = AgentJSONFormatter(
                fmt='%(timestamp)s %(level)s %(name)s %(message)s'
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def _log(self, level: int, event_type: str, message: str,
             session_id: Optional[str] = None,
             trace_id: Optional[str] = None,
             user_id: Optional[str] = None,
             extra: Optional[Dict[str, Any]] = None):
        """Internal logging method with consistent structure."""
        
        log_data = {
            'event_type': event_type,
            'agent_id': self.agent_id,
            'agent_version': self.agent_version,
            'session_id': session_id or '',
            'trace_id': trace_id or str(uuid.uuid4()),
            'user_id': user_id or '',
            'environment': self.environment,
        }
        
        if extra:
            log_data.update(extra)
        
        self.logger.log(level, message, extra=log_data)
    
    def log_event(self, event_type: str, message: str, **kwargs):
        """Log a structured agent event at INFO level."""
        self._log(logging.INFO, event_type, message, **kwargs)
    
    def log_error(self, event_type: str, message: str, error: Exception = None, **kwargs):
        """Log an agent error with exception details."""
        extra = kwargs.pop('extra', {})
        if error:
            extra['error_type'] = type(error).__name__
            extra['error_message'] = str(error)
        self._log(logging.ERROR, event_type, message, extra=extra, **kwargs)
    
    def log_warning(self, event_type: str, message: str, **kwargs):
        self._log(logging.WARNING, event_type, message, **kwargs)
    
    def log_debug(self, event_type: str, message: str, **kwargs):
        self._log(logging.DEBUG, event_type, message, **kwargs)
    
    # Convenience methods for common agent events
    def session_start(self, session_id: str, user_id: str, input_text: str):
        self.log_event(
            event_type="session.start",
            message="Agent session started",
            session_id=session_id,
            user_id=user_id,
            extra={"input_length": len(input_text), "input_preview": input_text[:100]}
        )
    
    def session_end(self, session_id: str, duration_ms: float, success: bool, output: str = ""):
        self.log_event(
            event_type="session.end",
            message="Agent session ended",
            session_id=session_id,
            extra={
                "duration_ms": duration_ms,
                "success": success,
                "output_length": len(output),
            }
        )
    
    def llm_call(self, session_id: str, model: str, input_tokens: int, 
                 output_tokens: int, duration_ms: float, cost: float,
                 error: Optional[Exception] = None):
        if error:
            self.log_error(
                event_type="llm.call.error",
                message=f"LLM call failed: {error}",
                session_id=session_id,
                error=error,
                extra={"model": model, "duration_ms": duration_ms}
            )
        else:
            self.log_event(
                event_type="llm.call.end",
                message=f"LLM call to {model} completed",
                session_id=session_id,
                extra={
                    "model": model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "duration_ms": duration_ms,
                    "cost": cost,
                }
            )
    
    def tool_call(self, session_id: str, tool_name: str, duration_ms: float,
                  success: bool, error: Optional[Exception] = None):
        if error:
            self.log_error(
                event_type="tool.call.error",
                message=f"Tool {tool_name} failed: {error}",
                session_id=session_id,
                error=error,
                extra={"tool": tool_name, "duration_ms": duration_ms}
            )
        else:
            self.log_event(
                event_type="tool.call.end",
                message=f"Tool {tool_name} completed",
                session_id=session_id,
                extra={"tool": tool_name, "duration_ms": duration_ms, "success": True}
            )
```

## 3. Log Aggregation

### 3.1 Loki Configuration

For Grafana Loki log aggregation, configure promtail to collect agent logs:

```yaml
# promtail-config.yaml
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: agent-logs
    static_configs:
      - targets: [localhost]
        labels:
          job: agent-runtime
          __path__: /var/log/agents/*.log
    pipeline_stages:
      - json:
          expressions:
            timestamp: timestamp
            level: level
            event_type: event_type
            agent_id: agent_id
            session_id: session_id
            trace_id: trace_id
            user_id: user_id
            error_type: error_type
      - timestamp:
          source: timestamp
          format: RFC3339
      - labels:
          level:
          event_type:
          agent_id:
          user_id:
```

### 3.2 Log Query Patterns for Debugging

```python
"""Common log query patterns using Loki's LogQL."""

# Query 1: Find all sessions where an agent errored
"""
{agent_id="customer-support-agent", level="error"}
| json
| filter event_type == "session.end" and success == "false"
"""

# Query 2: Find sessions with high cost (from structured logs)
"""
{agent_id="research-agent", event_type="session.end"}
| json
| filter extra.total_cost > 0.50
"""

# Query 3: Trace a specific session across all events
"""
{session_id="sess_abc123"}
| json
"""

# Query 4: Find agent loops (many steps without session end)
"""
{event_type="agent.think"}
| json
| filter session_id in (
    {event_type="session.start"}
    | json
    | filter agent_id="research-agent"
    | label_format session_id="{{.session_id}}"
)
"""

# Query 5: Rate limited requests by agent
"""
{event_type="rate.limited"}
| json
| rate by (agent_id) [1h]
"""

# Query 6: Tool failure rate
"""
{event_type=~"tool.call.*"}
| json
| filter success == "false"
| rate by (tool) [1h]
"""
```

## 4. Prometheus Metrics for Agents

### 4.1 Core Agent Metrics

```python
"""Prometheus metrics definition for agent monitoring."""

from prometheus_client import Counter, Histogram, Gauge, Summary
import time
from functools import wraps

# === Counter Metrics ===

# Total agent runs
AGENT_RUNS_TOTAL = Counter(
    'agent_runs_total',
    'Total number of agent runs',
    ['agent_id', 'agent_version', 'status']  # status: success, error, timeout
)

# LLM calls
AGENT_LLM_CALLS_TOTAL = Counter(
    'agent_llm_calls_total',
    'Total LLM API calls made by agents',
    ['agent_id', 'model', 'provider']
)

# Tool calls
AGENT_TOOL_CALLS_TOTAL = Counter(
    'agent_tool_calls_total',
    'Total tool calls made by agents',
    ['agent_id', 'tool_name', 'status']
)

# Token usage
AGENT_TOKENS_TOTAL = Counter(
    'agent_tokens_total',
    'Total tokens consumed',
    ['agent_id', 'model', 'token_type']  # token_type: input, output, reasoning
)

# Errors by category
AGENT_ERRORS_TOTAL = Counter(
    'agent_errors_total',
    'Total agent errors by category',
    ['agent_id', 'error_category']  # error_category: llm_error, tool_error, timeout, rate_limit, hallucination, loop
)

# User feedback
AGENT_USER_FEEDBACK = Counter(
    'agent_user_feedback_total',
    'User feedback counts',
    ['agent_id', 'rating']  # rating: positive, negative
)

# === Histogram Metrics ===

# Session duration
AGENT_SESSION_DURATION = Histogram(
    'agent_session_duration_seconds',
    'Duration of agent sessions',
    ['agent_id', 'status'],
    buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600]
)

# Step duration
AGENT_STEP_DURATION = Histogram(
    'agent_step_duration_seconds',
    'Duration of individual agent steps',
    ['agent_id', 'step_type'],  # step_type: think, act, observe
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

# LLM call latency
AGENT_LLM_LATENCY = Histogram(
    'agent_llm_latency_seconds',
    'Latency of LLM API calls',
    ['agent_id', 'model'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0]
)

# Tool latency
AGENT_TOOL_LATENCY = Histogram(
    'agent_tool_latency_seconds',
    'Latency of tool calls',
    ['agent_id', 'tool_name'],
    buckets=[0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

# Steps per session
AGENT_STEPS_PER_SESSION = Histogram(
    'agent_steps_per_session',
    'Number of steps per agent session',
    ['agent_id'],
    buckets=[1, 3, 5, 10, 15, 20, 30, 50, 100]
)

# Cost per session
AGENT_COST_PER_SESSION = Histogram(
    'agent_cost_per_session_usd',
    'Cost per agent session in USD',
    ['agent_id'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# === Gauge Metrics ===

# Active sessions
AGENT_ACTIVE_SESSIONS = Gauge(
    'agent_active_sessions',
    'Currently active agent sessions',
    ['agent_id']
)

# Queue depth
AGENT_QUEUE_DEPTH = Gauge(
    'agent_queue_depth',
    'Number of agent tasks waiting to be processed',
    ['agent_id', 'priority']
)

# Context utilization
AGENT_CONTEXT_UTILIZATION = Gauge(
    'agent_context_utilization_percent',
    'Context window utilization percentage',
    ['agent_id', 'model']
)

# === Summary Metrics (for quantiles) ===

AGENT_LATENCY_SUMMARY = Summary(
    'agent_latency_summary_seconds',
    'Latency summary for agent operations',
    ['agent_id', 'operation_type']
)


# === Decorator for automatic metric recording ===

def monitor_agent_run(agent_id: str, agent_version: str):
    """Decorator that automatically records agent run metrics."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            AGENT_ACTIVE_SESSIONS.labels(agent_id=agent_id).inc()
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                AGENT_SESSION_DURATION.labels(agent_id=agent_id, status="success").observe(duration)
                AGENT_RUNS_TOTAL.labels(agent_id=agent_id, agent_version=agent_version, status="success").inc()
                return result
            except Exception as e:
                duration = time.time() - start_time
                AGENT_SESSION_DURATION.labels(agent_id=agent_id, status="error").observe(duration)
                AGENT_RUNS_TOTAL.labels(agent_id=agent_id, agent_version=agent_version, status="error").inc()
                AGENT_ERRORS_TOTAL.labels(agent_id=agent_id, error_category="unknown").inc()
                raise
            finally:
                AGENT_ACTIVE_SESSIONS.labels(agent_id=agent_id).dec()
        return wrapper
    return decorator
```

### 4.2 Metrics Middleware for Agent Runtimes

```python
"""Metrics middleware for agent runtimes."""

from prometheus_client import start_http_server
import threading
from typing import Callable

class AgentMetricsMiddleware:
    """
    Middleware that records Prometheus metrics for agent operations.
    Attach to an agent runtime or use as a decorator.
    """
    
    def __init__(self, agent_id: str, metrics_port: int = 8000):
        self.agent_id = agent_id
        self.metrics_port = metrics_port
        
        # Start Prometheus HTTP server in background
        self._start_metrics_server()
    
    def _start_metrics_server(self):
        """Start the Prometheus metrics endpoint."""
        def serve():
            start_http_server(self.metrics_port)
            print(f"Metrics server started on port {self.metrics_port}")
        
        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
    
    def wrap_agent(self, agent_fn: Callable) -> Callable:
        """Wrap an agent function with metric recording."""
        @wraps(agent_fn)
        def wrapper(task_input: str, **kwargs):
            session_start = time.time()
            AGENT_ACTIVE_SESSIONS.labels(agent_id=self.agent_id).inc()
            
            try:
                result = agent_fn(task_input, **kwargs)
                duration = time.time() - session_start
                
                AGENT_SESSION_DURATION.labels(
                    agent_id=self.agent_id, status="success"
                ).observe(duration)
                AGENT_RUNS_TOTAL.labels(
                    agent_id=self.agent_id, agent_version=result.get("version", "unknown"),
                    status="success"
                ).inc()
                
                if "steps" in result:
                    AGENT_STEPS_PER_SESSION.labels(
                        agent_id=self.agent_id
                    ).observe(result["steps"])
                
                return result
                
            except Exception as e:
                duration = time.time() - session_start
                AGENT_SESSION_DURATION.labels(
                    agent_id=self.agent_id, status="error"
                ).observe(duration)
                AGENT_RUNS_TOTAL.labels(
                    agent_id=self.agent_id, agent_version="unknown",
                    status="error"
                ).inc()
                raise
            finally:
                AGENT_ACTIVE_SESSIONS.labels(agent_id=self.agent_id).dec()
        
        return wrapper


# === Usage ===
# In agent runtime:
# middleware = AgentMetricsMiddleware(agent_id="research-agent", metrics_port=8000)
# research_agent = middleware.wrap_agent(research_agent.run)
```

## 5. Alerting

### 5.1 Alert Rules Configuration

```yaml
# prometheus-agent-alerts.yml
groups:
  - name: agent_alerts
    interval: 30s
    rules:
      # === Critical Alerts ===
      
      - alert: AgentHighErrorRate
        expr: rate(agent_errors_total[5m]) / rate(agent_runs_total[5m]) > 0.15
        for: 5m
        labels:
          severity: critical
          team: agent-ops
        annotations:
          summary: "Agent {{ $labels.agent_id }} error rate > 15%"
          description: "Error rate is {{ $value | humanizePercentage }} over the last 5 minutes"
      
      - alert: AgentCostSpike
        expr: rate(agent_cost_per_session_usd_sum[1h]) > 50
        for: 5m
        labels:
          severity: critical
          team: cost-ops
        annotations:
          summary: "Agent cost spike > $50/hour"
          description: "Current hourly cost rate is ${{ $value | humanize }}"
      
      - alert: AgentLatencyHigh
        expr: histogram_quantile(0.95, rate(agent_session_duration_seconds_bucket[5m])) > 120
        for: 5m
        labels:
          severity: critical
          team: agent-ops
        annotations:
          summary: "Agent p95 latency > 120s"
          description: "P95 session latency is {{ $value | humanizeDuration }}"
      
      # === Warning Alerts ===
      
      - alert: AgentHighWarningRate
        expr: rate(agent_errors_total{error_category=~"timeout|rate_limit|retry"}[5m]) > 10
        for: 5m
        labels:
          severity: warning
          team: agent-ops
        annotations:
          summary: "Agent {{ $labels.agent_id }} transient errors > 10/min"
          description: "Transient error rate is {{ $value | humanize }} per minute"
      
      - alert: AgentStepsExcessive
        expr: histogram_quantile(0.95, rate(agent_steps_per_session_bucket[1h])) > 30
        for: 10m
        labels:
          severity: warning
          team: agent-ops
        annotations:
          summary: "Agent {{ $labels.agent_id }} p95 steps > 30"
          description: "P95 steps per session is {{ $value | humanize }}"
      
      - alert: AgentThroughputDrop
        expr: rate(agent_runs_total[30m]) < rate(agent_runs_total[1h]) * 0.5
        for: 10m
        labels:
          severity: warning
          team: agent-ops
        annotations:
          summary: "Agent throughput dropped > 50%"
          description: "Current throughput is {{ $value | humanize }} runs/min, down 50% from baseline"
      
      # === Info Alerts ===
      
      - alert: AgentModelDeprecated
        expr: rate(agent_llm_calls_total{model=~"gpt-3.5|text-davinci"}[5m]) > 0
        for: 1h
        labels:
          severity: info
          team: agent-eng
        annotations:
          summary: "Agent {{ $labels.agent_id }} using deprecated model"
          description: "Agent is making calls to deprecated model {{ $labels.model }}"
      
      - alert: AgentContextHigh
        expr: avg(agent_context_utilization_percent) by (agent_id) > 80
        for: 15m
        labels:
          severity: info
          team: agent-eng
        annotations:
          summary: "Agent {{ $labels.agent_id }} context utilization > 80%"
          description: "Average context utilization is {{ $value | humanize }}%"
```

### 5.2 Alert Manager Configuration

```yaml
# alertmanager.yml
route:
  receiver: agent-team
  group_by: [alertname, agent_id]
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  routes:
    - match:
        severity: critical
      receiver: agent-pagerduty
      repeat_interval: 30m
    - match:
        severity: warning
      receiver: agent-slack
    - match:
        severity: info
      receiver: agent-email

receivers:
  - name: agent-pagerduty
    pagerduty_configs:
      - routing_key: "<pagerduty-key>"
        severity: critical
        description: '{{ template "pagerduty.description" . }}'
  
  - name: agent-slack
    slack_configs:
      - api_url: "<slack-webhook-url>"
        channel: "#agent-alerts"
        title: '{{ template "slack.title" . }}'
        text: '{{ template "slack.text" . }}'
  
  - name: agent-email
    email_configs:
      - to: "agent-team@company.com"
        from: "agent-alerts@company.com"
```

### 5.3 Webhook Alert Handler

```python
"""Webhook-based alert handler for agent systems."""

import json
import requests
from typing import Optional, Dict, Any

class AgentAlertWebhook:
    """
    Sends alerts to various destinations (Slack, PagerDuty, custom webhook).
    """
    
    def __init__(self, slack_webhook: Optional[str] = None,
                 pagerduty_key: Optional[str] = None,
                 custom_webhooks: Optional[list] = None):
        self.slack_webhook = slack_webhook
        self.pagerduty_key = pagerduty_key
        self.custom_webhooks = custom_webhooks or []
    
    def send_alert(self, title: str, message: str, severity: str = "warning",
                   agent_id: Optional[str] = None, metadata: Optional[Dict] = None):
        """Send alert to all configured destinations."""
        alert_data = {
            "title": title,
            "message": message,
            "severity": severity,
            "agent_id": agent_id,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        if self.slack_webhook:
            self._send_slack(alert_data)
        
        if self.pagerduty_key:
            self._send_pagerduty(alert_data)
        
        for webhook in self.custom_webhooks:
            self._send_custom(webhook, alert_data)
    
    def _send_slack(self, alert: Dict[str, Any]):
        """Send alert to Slack."""
        color = {"critical": "danger", "warning": "warning", "info": "good"}.get(
            alert["severity"], "warning"
        )
        
        payload = {
            "attachments": [{
                "color": color,
                "title": alert["title"],
                "text": alert["message"],
                "fields": [
                    {"title": "Agent", "value": alert.get("agent_id", "N/A"), "short": True},
                    {"title": "Severity", "value": alert["severity"], "short": True},
                ],
                "footer": "Agent Monitoring",
                "ts": datetime.utcnow().timestamp(),
            }]
        }
        
        try:
            requests.post(self.slack_webhook, json=payload, timeout=5)
        except requests.RequestException as e:
            print(f"Failed to send Slack alert: {e}")
    
    def _send_pagerduty(self, alert: Dict[str, Any]):
        """Send critical alert to PagerDuty."""
        payload = {
            "routing_key": self.pagerduty_key,
            "event_action": "trigger",
            "payload": {
                "summary": alert["title"],
                "severity": alert["severity"],
                "source": alert.get("agent_id", "agent-system"),
                "custom_details": alert.get("metadata", {}),
            },
        }
        
        try:
            response = requests.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload,
                timeout=5,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Failed to send PagerDuty alert: {e}")
```

## 6. Agent Metrics Dashboard

### 6.1 Grafana Dashboard JSON

Below is a subset of the full Grafana dashboard for agent monitoring. Import this JSON into Grafana with Prometheus as the data source.

```json
{
  "dashboard": {
    "title": "Agent Operations Dashboard",
    "tags": ["agents", "operations"],
    "time": { "from": "now-1h", "to": "now" },
    "panels": [
      {
        "title": "Active Sessions",
        "type": "stat",
        "datasource": "Prometheus",
        "targets": [{
          "expr": "sum(agent_active_sessions)",
          "legendFormat": "Active"
        }],
        "options": { "colorMode": "value", "graphMode": "area" }
      },
      {
        "title": "Session Throughput",
        "type": "timeseries",
        "datasource": "Prometheus",
        "targets": [{
          "expr": "rate(agent_runs_total[5m])",
          "legendFormat": "{{agent_id}}"
        }],
        "options": { "legend": { "displayMode": "table" } }
      },
      {
        "title": "Error Rate",
        "type": "timeseries",
        "datasource": "Prometheus",
        "targets": [{
          "expr": "rate(agent_errors_total[5m]) / rate(agent_runs_total[5m])",
          "legendFormat": "{{agent_id}}"
        }]
      },
      {
        "title": "P95 Session Latency",
        "type": "timeseries",
        "datasource": "Prometheus",
        "targets": [{
          "expr": "histogram_quantile(0.95, sum(rate(agent_session_duration_seconds_bucket[5m])) by (le, agent_id))",
          "legendFormat": "{{agent_id}}"
        }]
      },
      {
        "title": "Steps per Session",
        "type": "heatmap",
        "datasource": "Prometheus",
        "targets": [{
          "expr": "sum(rate(agent_steps_per_session_bucket[1h])) by (le)",
          "format": "heatmap"
        }]
      },
      {
        "title": "Errors by Category",
        "type": "piechart",
        "datasource": "Prometheus",
        "targets": [{
          "expr": "sum(rate(agent_errors_total[1h])) by (error_category)"
        }]
      },
      {
        "title": "Top Agents by Cost",
        "type": "barchart",
        "datasource": "Prometheus",
        "targets": [{
          "expr": "topk(10, sum(rate(agent_cost_per_session_usd_sum[1h])) by (agent_id))"
        }]
      },
      {
        "title": "Cost per Session Distribution",
        "type": "timeseries",
        "datasource": "Prometheus",
        "targets": [{
          "expr": "histogram_quantile(0.5, sum(rate(agent_cost_per_session_usd_bucket[1h])) by (le))",
          "legendFormat": "p50"
        }, {
          "expr": "histogram_quantile(0.95, sum(rate(agent_cost_per_session_usd_bucket[1h])) by (le))",
          "legendFormat": "p95"
        }, {
          "expr": "histogram_quantile(0.99, sum(rate(agent_cost_per_session_usd_bucket[1h])) by (le))",
          "legendFormat": "p99"
        }]
      },
      {
        "title": "Model Usage Distribution",
        "type": "piechart",
        "datasource": "Prometheus",
        "targets": [{
          "expr": "sum(rate(agent_llm_calls_total[1h])) by (model)"
        }]
      },
      {
        "title": "Context Utilization",
        "type": "timeseries",
        "datasource": "Prometheus",
        "targets": [{
          "expr": "avg(agent_context_utilization_percent) by (agent_id)",
          "legendFormat": "{{agent_id}}"
        }]
      },
      {
        "title": "Tool Call Rate",
        "type": "timeseries",
        "datasource": "Prometheus",
        "targets": [{
          "expr": "sum(rate(agent_tool_calls_total[5m])) by (tool_name)",
          "legendFormat": "{{tool_name}}"
        }]
      },
      {
        "title": "LLM Call Latency by Model",
        "type": "timeseries",
        "datasource": "Prometheus",
        "targets": [{
          "expr": "histogram_quantile(0.95, sum(rate(agent_llm_latency_seconds_bucket[5m])) by (le, model))",
          "legendFormat": "p95 - {{model}}"
        }]
      }
    ]
  }
}
```

## 7. Production Monitoring Runbook

### 7.1 Incident Response for Agent Failures

When an alert fires, follow this runbook:

```
INCIDENT: Agent High Error Rate
─────────────────────────────────
1. ACKNOWLEDGE the alert
2. IDENTIFY the affected agent(s) from alert labels
3. CHECK recent deployments:
   - Has the agent version changed recently?
   - Has the model/prompt/tool set changed?
4. RETRIEVE failing traces:
   - Query Loki: {agent_id="<name>", event_type="session.end", success="false"}
   - Look at the error types (tool errors, LLM errors, timeouts)
5. CHECK dependency status:
   - Are LLM APIs healthy? (OpenAI status page)
   - Are external tool APIs healthy?
   - Are rate limits being hit?
6. CORRECTIVE actions:
   - Tool errors: Check tool API status, verify API keys
   - LLM errors: Check quota, rate limits, model availability
   - Timeout errors: Increase timeout, check tool latency
   - Hallucination/Loop: Roll back agent version
7. ESCALATE if needed:
   - If error persists, escalate to agent-eng team
   - If cost spike, escalate to cost-ops team
```

### 7.2 Common Incident Patterns

| Symptom | Likely Cause | Action |
|---------|-------------|--------|
| Sudden error rate spike | Model API outage or tool API change | Check provider status, switch to fallback |
| Gradual error rate increase | Prompt drift, data drift, model degradation | Run eval suite, compare with baseline |
| Cost spike | Agent in loop, prompt too long, model switch | Kill stale sessions, add step limit, check cost opt |
| Latency spike | Tool slowdown, context too large, rate limiting | Check tool perf, reduce context, increase rate limit |
| Throughput drop | Rate limiting, infrastructure scaling | Check rate limit quotas, auto-scale |
| Steps per session increasing | Prompt degradation, tool selection issues | Run eval, review tool descriptions |
| User feedback turning negative | Quality regression, behavior change | Compare with eval baseline, roll back |

## 8. Conclusion

Effective logging and monitoring for agent systems requires:

1. **Structured logging** with a consistent event taxonomy and JSON format
2. **Rich metrics** covering latency (p50/p95/p99), error rates, throughput, cost, token usage, and context utilization
3. **Comprehensive alerting** with tiered severity and actionable runbooks
4. **Integrated dashboards** that surface the most important metrics at a glance
5. **Production runbooks** that guide incident response for common failure patterns

The key insight is that agent monitoring is not just about watching for errors — it's about understanding the quality, cost, and behavior of autonomous decision-making systems. A well-monitored agent system gives operators confidence to deploy, iterate, and scale.

---

*Next: [07-Agent-Reliability-and-Resilience.md](07-Agent-Reliability-and-Resilience.md) — Failure modes, retry strategies, and resilient agent architecture.*
