# 06 — Agent Audit and Forensics

## 1. Introduction

Audit and forensic capabilities are foundational to agent security. Without comprehensive logging, organizations cannot determine what an agent did, why it did it, or who authorized it. In the event of an incident — whether from prompt injection, tool misuse, or data exfiltration — audit logs are the primary source of evidence for understanding the scope and impact of the breach.

Agent audit differs from traditional application auditing in several critical ways: agent actions are non-deterministic (same input may produce different actions), agents operate across multiple tools and services, and the reasoning behind actions (the LLM's chain-of-thought) is often as important as the actions themselves.

This document covers agent action logging, audit trail design, tamper-proof logging, traceability across multi-agent systems, forensic analysis methodologies, compliance requirements, and log aggregation patterns.

## 2. The Agent Audit Challenge

### 2.1 What to Log in Agent Systems

Agent audit logging must capture more than traditional application logs:

| Traditional Audit | Agent Audit |
|------------------|-------------|
| User ID | User ID + Agent ID + Session ID |
| Timestamp | Timestamp + Latency + Token Count |
| Action (CRUD) | Tool Call + Parameters + Response |
| IP Address | IP + Model Version + Temperature |
| Resource Accessed | Data Sensitivity Level + Context Summary |
| Result (success/fail) | Reasoning + Confidence + Alternative Actions Considered |

### 2.2 Key Audit Requirements

1. **Completeness**: Every agent action, tool call, and decision must be logged.
2. **Tamper-evidence**: Logs must be protected from modification after creation.
3. **Chain-of-thought**: The reasoning that led to each action must be preserved.
4. **Traceability**: Actions must be traceable across agent-session-tool-service boundaries.
5. **Searchability**: Logs must be efficiently queryable for incident response.
6. **Privacy**: Sensitive data in logs must be handled appropriately (redaction, encryption).

## 3. Agent Action Logging

### 3.1 Comprehensive Log Schema

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import uuid

@dataclass
class AgentActionLog:
    """Complete audit record for a single agent action."""

    # Identity
    log_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    agent_id: str = ""
    agent_version: str = ""
    user_id: str = ""             # The human user (if acting on behalf)
    session_id: str = ""          # Conversation session

    # Timestamps
    timestamp: str = ""           # When the action started
    duration_ms: int = 0          # How long it took
    completed_at: str = ""        # When it finished

    # Input
    user_message: str = ""        # The user's input (truncated if long)
    user_message_hash: str = ""   # SHA-256 of full user message
    system_prompt_hash: str = ""  # Which system prompt version was used
    context_summary: str = ""     # Summary of conversation context

    # LLM Processing
    model_name: str = ""          # e.g., gpt-4, claude-3, llama-3
    model_version: str = ""       # Specific model version
    temperature: float = 0.0
    max_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # Decision
    reasoning: str = ""           # Chain-of-thought (if available)
    confidence: float = 0.0       # Model's confidence in this action
    alternative_actions: list = field(default_factory=list)
    risk_score: float = 0.0       # Security risk assessment score

    # Action
    tool_name: str = ""           # Which tool was called
    tool_params: dict = field(default_factory=dict)  # Parameters
    tool_params_hash: str = ""    # Hash of parameters for integrity
    tool_output_summary: str = ""  # Summary of output
    tool_output_hash: str = ""    # Hash of full output
    tool_success: bool = True
    tool_error: Optional[str] = None

    # Security
    injection_detected: bool = False
    injection_score: float = 0.0
    policy_violations: list = field(default_factory=list)
    approval_required: bool = False
    approval_granted: bool = False
    approver_id: Optional[str] = None

    # Network
    source_ip: str = ""
    user_agent: str = ""

    # Integrity
    previous_log_hash: str = ""   # For blockchain-style chaining
    log_hash: str = ""            # Hash of this record

    def compute_hash(self) -> str:
        """Compute cryptographic hash of this log record."""
        import hashlib, json

        # Create a deterministic representation
        data = {
            "log_id": self.log_id,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp,
            "user_message_hash": self.user_message_hash,
            "tool_name": self.tool_name,
            "tool_params_hash": self.tool_params_hash,
            "previous_log_hash": self.previous_log_hash,
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()
```

### 3.2 Logging Middleware

```python
class AgentAuditMiddleware:
    """Middleware to automatically log all agent actions."""

    def __init__(self, log_storage: AuditLogStorage):
        self.log_storage = log_storage
        self.previous_hash = "0" * 64  # Genesis hash

    def wrap_agent(self, agent_func: callable) -> callable:
        """Wrap an agent function with audit logging."""

        async def audit_wrapper(*args, **kwargs):
            log = AgentActionLog()
            log.log_id = uuid.uuid4().hex
            log.agent_id = kwargs.get("agent_id", "unknown")
            log.session_id = kwargs.get("session_id", "unknown")
            log.timestamp = datetime.now().isoformat()
            log.previous_log_hash = self.previous_hash

            try:
                # Execute the agent
                start = datetime.now()
                result = await agent_func(*args, **kwargs)
                duration = (datetime.now() - start).total_seconds() * 1000

                # Populate log from result
                log.duration_ms = int(duration)
                log.tool_name = result.get("tool_name", "")
                log.tool_params = result.get("params", {})
                log.tool_success = result.get("success", True)
                log.risk_score = result.get("risk_score", 0.0)

            except Exception as e:
                log.tool_success = False
                log.tool_error = str(e)
                raise
            finally:
                # Compute hash and store
                log.log_hash = log.compute_hash()
                self.previous_hash = log.log_hash
                self.log_storage.store(log)

            return result

        return audit_wrapper
```

### 3.3 Structured Log Emission

```python
import logging
import json

class AgentJSONLogger:
    """Structured JSON logging for agent actions."""

    def __init__(self, service_name: str, environment: str):
        self.logger = logging.getLogger(f"agent_audit.{service_name}")
        self.service_name = service_name
        self.environment = environment

        # JSON formatter
        handler = logging.StreamHandler()
        handler.setFormatter(AgentJSONFormatter())
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def log_action(self, action_log: AgentActionLog):
        """Emit a structured log entry."""
        record = {
            "type": "agent_action",
            "service": self.service_name,
            "environment": self.environment,
            "log": action_log.to_dict(),  # Convert dataclass to dict
        }
        self.logger.info(json.dumps(record))

    def log_security_event(self, severity: str, event_type: str,
                            agent_id: str, details: dict):
        """Emit a security event log."""
        record = {
            "type": "agent_security_event",
            "severity": severity,
            "event_type": event_type,
            "agent_id": agent_id,
            "details": details,
            "service": self.service_name,
            "environment": self.environment,
            "timestamp": datetime.now().isoformat(),
        }
        log_level = {
            "critical": self.logger.critical,
            "high": self.logger.error,
            "medium": self.logger.warning,
            "low": self.logger.info,
        }.get(severity, self.logger.info)
        log_level(json.dumps(record))

class AgentJSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured agent logs."""
    def format(self, record):
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            try:
                # Already JSON
                return record.msg
            except Exception:
                pass
        return json.dumps({
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        })
```

## 4. Audit Trail Design

### 4.1 Hash-Chained Audit Log

Implement a blockchain-style hash chain for tamper-evident logs:

```python
import hashlib
import json
from datetime import datetime
from typing import Optional

class HashChainedAuditLog:
    """
    Tamper-evident audit log using cryptographic hash chaining.
    Each log entry contains the hash of the previous entry.
    """

    def __init__(self, storage_backend):
        self.storage = storage_backend
        self.latest_hash = None

    def append(self, entry: dict) -> str:
        """Append an entry to the hash chain."""
        # Add chain metadata
        entry["_timestamp"] = datetime.now().isoformat()
        entry["_previous_hash"] = self.latest_hash or "0" * 64

        # Compute entry hash
        entry_hash = hashlib.sha256(
            json.dumps(entry, sort_keys=True).encode()
        ).hexdigest()
        entry["_hash"] = entry_hash

        # Store
        self.storage.store(entry_hash, entry)
        self.latest_hash = entry_hash

        return entry_hash

    def verify_chain(self) -> tuple[bool, list[str]]:
        """Verify the integrity of the entire hash chain."""
        # Get all entries in order
        entries = self.storage.get_all()

        issues = []
        previous_hash = "0" * 64

        for entry in entries:
            stored_hash = entry.get("_hash")
            expected_hash = hashlib.sha256(
                json.dumps(entry, sort_keys=True).encode()
            ).hexdigest()

            if stored_hash != expected_hash:
                issues.append(f"Hash mismatch at {entry.get('_timestamp')}")

            if entry.get("_previous_hash") != previous_hash:
                issues.append(f"Chain break at {entry.get('_timestamp')}")

            previous_hash = stored_hash

        return len(issues) == 0, issues

    def get_entry(self, entry_hash: str) -> Optional[dict]:
        """Retrieve a specific entry by its hash."""
        return self.storage.get(entry_hash)
```

### 4.2 Append-Only Log Implementation

```python
import os
import fcntl

class AppendOnlyLog:
    """Append-only log file with tamper detection."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self._ensure_exists()

    def _ensure_exists(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w') as f:
                f.write(f"# Agent Audit Log\n")
                f.write(f"# Created: {datetime.now().isoformat()}\n")
                f.write(f"# WARNING: This file is append-only. Do not modify.\n")

    def append(self, entry: str):
        """Append a log entry with advisory lock."""
        with open(self.log_path, 'a') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(entry + '\n')
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def read_range(self, start_time: datetime,
                    end_time: datetime) -> list[str]:
        """Read log entries within a time range."""
        entries = []
        with open(self.log_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                try:
                    entry = json.loads(line)
                    entry_time = datetime.fromisoformat(
                        entry.get("timestamp", "")
                    )
                    if start_time <= entry_time <= end_time:
                        entries.append(entry)
                except (json.JSONDecodeError, ValueError):
                    continue
        return entries

    def detect_tampering(self) -> list[str]:
        """Detect signs of log tampering."""
        issues = []
        with open(self.log_path, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if line.startswith('#'):
                continue
            # Check line integrity
            if not line.endswith('\n'):
                issues.append(f"Line {i+1}: Missing newline (truncated?)")

        # Check file permissions
        stat = os.stat(self.log_path)
        if stat.st_mode & 0o022:  # World-writable
            issues.append(f"File is world-writable: {oct(stat.st_mode)}")

        return issues
```

### 4.3 Distributed Audit Log

For multi-agent systems spanning multiple services:

```python
class DistributedAuditCoordinator:
    """
    Coordinates audit logs across multiple agent services.
    Uses a central log aggregator with local buffering.
    """

    def __init__(self, service_id: str, aggregator_url: str):
        self.service_id = service_id
        self.aggregator_url = aggregator_url
        self.local_buffer = []
        self.buffer_size = 100
        self.flush_interval = 60  # seconds
        self.last_flush = datetime.now()

    def log(self, entry: dict) -> str:
        """Log an entry (locally buffer and async send to aggregator)."""
        entry["service_id"] = self.service_id
        entry["timestamp"] = datetime.now().isoformat()
        entry["log_id"] = uuid.uuid4().hex

        # Add local sequence number
        entry["sequence"] = len(self.local_buffer) + 1

        self.local_buffer.append(entry)

        # Flush if buffer full or time elapsed
        if (len(self.local_buffer) >= self.buffer_size or
            (datetime.now() - self.last_flush).seconds >= self.flush_interval):
            self.flush()

        return entry["log_id"]

    def flush(self):
        """Send buffered logs to the aggregator."""
        if not self.local_buffer:
            return

        try:
            response = requests.post(
                f"{self.aggregator_url}/api/logs/batch",
                json={
                    "service": self.service_id,
                    "entries": self.local_buffer,
                },
                timeout=10,
            )
            if response.status_code == 200:
                self.local_buffer = []
                self.last_flush = datetime.now()
        except requests.RequestException as e:
            # Keep buffer; will retry on next flush
            print(f"Failed to flush logs: {e}")

    def query(self, query: dict) -> list[dict]:
        """Query the central log aggregator."""
        response = requests.post(
            f"{self.aggregator_url}/api/logs/query",
            json=query,
        )
        if response.status_code == 200:
            return response.json()["results"]
        return []
```

## 5. Traceability Across Multi-Agent Systems

### 5.1 Distributed Tracing for Agent Actions

```python
from opentelemetry import trace
from opentelemetry.trace import SpanKind
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

class AgentTracer:
    """
    OpenTelemetry-based distributed tracing for multi-agent systems.
    """

    def __init__(self, service_name: str, endpoint: str):
        trace.set_tracer_provider(
            TracerProvider(resource=Resource.create({
                "service.name": service_name,
                "service.type": "ai_agent",
            }))
        )
        span_exporter = OTLPSpanExporter(endpoint=endpoint)
        span_processor = BatchSpanProcessor(span_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)

        self.tracer = trace.get_tracer(__name__)

    def trace_tool_call(self, agent_id: str, tool_name: str,
                         params: dict, parent_context=None):
        """Create a trace span for a tool call."""
        with self.tracer.start_as_current_span(
            f"tool_call.{tool_name}",
            context=parent_context,
            kind=SpanKind.CLIENT,
            attributes={
                "agent.id": agent_id,
                "tool.name": tool_name,
                "tool.params": json.dumps(params, default=str)[:1000],
            },
        ) as span:
            return span.get_span_context()

    def trace_agent_reasoning(self, agent_id: str,
                               reasoning_steps: list[dict]):
        """Trace the reasoning process of an agent."""
        with self.tracer.start_as_current_span(
            f"agent.reasoning",
            attributes={
                "agent.id": agent_id,
                "step_count": len(reasoning_steps),
            },
        ) as span:
            for i, step in enumerate(reasoning_steps):
                span.add_event(
                    f"reasoning_step_{i}",
                    attributes={"step": json.dumps(step, default=str)[:500]},
                )

class TraceContextPropagator:
    """Propagates trace context across agent calls."""

    @staticmethod
    def inject_headers(span_context) -> dict:
        """Inject trace context into outgoing request headers."""
        headers = {}
        trace.get_tracer_provider()._span_processors[0]._span_exporter
        # Use W3C trace context format
        if span_context:
            headers["traceparent"] = f"00-{span_context.trace_id:032x}-{span_context.span_id:016x}-01"
        return headers

    @staticmethod
    def extract_headers(headers: dict):
        """Extract trace context from incoming request headers."""
        traceparent = headers.get("traceparent")
        if traceparent:
            # Parse W3C traceparent header
            parts = traceparent.split("-")
            if len(parts) >= 4:
                trace_id = int(parts[1], 16)
                span_id = int(parts[2], 16)
                return trace.TraceContext(trace_id, span_id)
        return None
```

### 5.2 Causality Tracking

```python
class AgentCausalityTracker:
    """
    Tracks causal relationships between agent actions.
    Useful for understanding multi-step attack chains.
    """

    def __init__(self):
        self.action_graph = {}  # action_id -> {predecessors, successors}

    def record_action(self, action_id: str, agent_id: str,
                       parent_action_id: Optional[str] = None):
        """Record an action and its causal parent."""
        if action_id not in self.action_graph:
            self.action_graph[action_id] = {
                "agent_id": agent_id,
                "predecessors": set(),
                "successors": set(),
                "timestamp": datetime.now(),
            }

        if parent_action_id:
            self.action_graph[action_id]["predecessors"].add(parent_action_id)
            if parent_action_id in self.action_graph:
                self.action_graph[parent_action_id]["successors"].add(action_id)

    def get_causal_chain(self, action_id: str) -> list[str]:
        """Get the complete causal chain leading to an action."""
        chain = []
        current = action_id
        while current:
            chain.insert(0, current)
            node = self.action_graph.get(current)
            if node and node["predecessors"]:
                # Take the most recent predecessor
                predecessors = list(node["predecessors"])
                current = max(predecessors, key=lambda x: self.action_graph[x]["timestamp"])
            else:
                current = None
        return chain

    def get_causal_graph(self, action_id: str,
                          max_depth: int = 5) -> dict:
        """Get the causal graph around an action."""
        graph = {"nodes": {}, "edges": []}
        visited = set()

        def traverse(action_id, depth=0):
            if depth > max_depth or action_id in visited:
                return
            visited.add(action_id)

            node = self.action_graph.get(action_id)
            if not node:
                return

            graph["nodes"][action_id] = {
                "agent_id": node["agent_id"],
                "timestamp": node["timestamp"].isoformat(),
            }

            for pred in node["predecessors"]:
                graph["edges"].append({"from": pred, "to": action_id, "type": "caused_by"})
                traverse(pred, depth + 1)

            for succ in node["successors"]:
                graph["edges"].append({"from": action_id, "to": succ, "type": "caused"})
                traverse(succ, depth + 1)

        traverse(action_id)
        return graph
```

## 6. Forensic Analysis of Agent Failures

### 6.1 Incident Playback

```python
class AgentIncidentPlayback:
    """
    Replay agent actions from logs to reconstruct incidents.
    """

    def __init__(self, log_store):
        self.log_store = log_store

    def replay_session(self, session_id: str) -> list[dict]:
        """Replay all actions in a session in chronological order."""
        logs = self.log_store.query({"session_id": session_id})
        logs.sort(key=lambda x: x["timestamp"])
        return logs

    def analyze_attack_chain(self, session_id: str) -> dict:
        """Analyze a session for potential attack chains."""
        logs = self.replay_session(session_id)

        findings = {
            "injection_attempts": [],
            "policy_violations": [],
            "anomalous_patterns": [],
            "timeline": [],
        }

        for i, log in enumerate(logs):
            entry = {
                "step": i + 1,
                "timestamp": log["timestamp"],
                "action": f"{log.get('tool_name', 'unknown')}()",
                "risk_score": log.get("risk_score", 0),
            }

            # Check for injection patterns
            if log.get("injection_detected"):
                entry["flag"] = "INJECTION_DETECTED"
                findings["injection_attempts"].append(entry)

            # Check for policy violations
            if log.get("policy_violations"):
                entry["flag"] = "POLICY_VIOLATION"
                findings["policy_violations"].append(entry)

            # Check for anomalous sequences
            if i > 1:
                prev = logs[i-1]
                if (log.get("tool_name") == prev.get("tool_name") and
                    self._is_unusual_repetition(log, prev)):
                    entry["flag"] = "REPETITIVE_ACTION"
                    findings["anomalous_patterns"].append(entry)

            findings["timeline"].append(entry)

        findings["total_steps"] = len(logs)
        findings["risk_level"] = self._calculate_risk(findings)

        return findings

    def _is_unusual_repetition(self, log1: dict, log2: dict) -> bool:
        """Check if two consecutive actions are unusually repetitive."""
        # Same tool with similar parameters might indicate automated attack
        if log1.get("params") == log2.get("params"):
            return True
        return False

    def _calculate_risk(self, findings: dict) -> str:
        if len(findings["injection_attempts"]) > 2:
            return "CRITICAL"
        if len(findings["policy_violations"]) > 0:
            return "HIGH"
        if len(findings["anomalous_patterns"]) > 3:
            return "MEDIUM"
        return "LOW"
```

### 6.2 Root Cause Analysis

```python
class AgentRCA:
    """Root cause analysis for agent incidents."""

    def __init__(self, log_store, causality_tracker):
        self.log_store = log_store
        self.causality = causality_tracker

    def analyze_incident(self, incident_time: datetime,
                          affected_agent: str) -> dict:
        """Analyze an incident to determine root cause."""
        # Get logs around incident time
        logs = self.log_store.query({
            "agent_id": affected_agent,
            "time_range": [
                (incident_time - timedelta(hours=1)).isoformat(),
                (incident_time + timedelta(minutes=5)).isoformat(),
            ],
        })

        # Build timeline
        timeline = sorted(logs, key=lambda x: x["timestamp"])

        # Identify triggering event
        trigger = self._find_trigger(timeline)

        # Trace causal chain
        causal_chain = None
        if trigger:
            causal_chain = self.causality.get_causal_chain(trigger["log_id"])

        # Identify contributing factors
        contributing = self._find_contributing_factors(timeline, trigger)

        return {
            "incident_time": incident_time.isoformat(),
            "agent": affected_agent,
            "trigger": trigger,
            "causal_chain": causal_chain,
            "contributing_factors": contributing,
            "timeline_steps": len(timeline),
            "recommendations": self._generate_recommendations(
                trigger, contributing
            ),
        }

    def _find_trigger(self, timeline: list[dict]) -> Optional[dict]:
        """Find the first anomalous or unexpected event."""
        for entry in timeline:
            if (entry.get("injection_detected") or
                entry.get("risk_score", 0) > 0.8 or
                entry.get("tool_success") is False or
                entry.get("policy_violations")):
                return entry
        return None

    def _find_contributing_factors(self, timeline: list[dict],
                                    trigger: Optional[dict]) -> list[str]:
        """Identify factors that contributed to the incident."""
        factors = []
        if not trigger:
            return factors

        trigger_idx = timeline.index(trigger) if trigger in timeline else -1

        # Check for missing guards
        preceding = timeline[:trigger_idx]
        if not any(e.get("injection_check_performed") for e in preceding):
            factors.append("No injection detection was performed on input")

        # Check for privilege escalation
        if trigger and trigger.get("tool_name") in ("execute_code", "admin_action"):
            factors.append("Agent had elevated privileges for sensitive action")

        # Check for rate limit bypass
        recent_actions = [e for e in preceding
                          if e.get("tool_name") == trigger.get("tool_name")]
        if len(recent_actions) > 5:
            factors.append("Multiple rapid tool calls bypassed rate limiting")

        return factors

    def _generate_recommendations(self, trigger: Optional[dict],
                                    factors: list[str]) -> list[str]:
        """Generate security recommendations from RCA."""
        recommendations = []
        if trigger and trigger.get("injection_detected"):
            recommendations.append("Deploy additional prompt injection detection layer")
        if "No injection detection" in " ".join(factors):
            recommendations.append("Enable input scanning for all agent entry points")
        if "elevated privileges" in " ".join(factors):
            recommendations.append("Review and restrict agent tool permissions")
        return recommendations
```

### 6.3 Forensic Data Collection

```python
class AgentForensicCollector:
    """Collects forensic data from agent systems after an incident."""

    def __init__(self, storage_path: str):
        self.storage_path = storage_path

    def collect_snapshot(self, agent_id: str) -> str:
        """Collect a forensic snapshot of an agent's state."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_dir = f"{self.storage_path}/forensics/{agent_id}/{timestamp}"
        os.makedirs(snapshot_dir, exist_ok=True)

        artifacts = {}

        # 1. Current memory contents (if not encrypted)
        artifacts["memory"] = self._collect_memory(agent_id)

        # 2. Active conversation context
        artifacts["context"] = self._collect_context(agent_id)

        # 3. Tool call history
        artifacts["tool_history"] = self._collect_tool_history(agent_id)

        # 4. Permission state
        artifacts["permissions"] = self._collect_permissions(agent_id)

        # 5. Configuration
        artifacts["config"] = self._collect_config(agent_id)

        # 6. Logs
        artifacts["recent_logs"] = self._collect_logs(agent_id, minutes=30)

        # Save snapshot
        snapshot_file = f"{snapshot_dir}/snapshot.json"
        with open(snapshot_file, 'w') as f:
            json.dump(artifacts, f, default=str, indent=2)

        # Compute hash
        snapshot_hash = hashlib.sha256(
            open(snapshot_file, 'rb').read()
        ).hexdigest()
        with open(f"{snapshot_dir}/snapshot.hash", 'w') as f:
            f.write(snapshot_hash)

        return snapshot_file

    def _collect_memory(self, agent_id: str) -> dict:
        """Collect agent memory contents."""
        # This would integrate with the agent's memory system
        return {"status": "collected", "entries": []}

    def _collect_tool_history(self, agent_id: str) -> list[dict]:
        """Collect recent tool call history."""
        return []  # Implementation depends on storage

    def _collect_permissions(self, agent_id: str) -> dict:
        """Collect current permission state."""
        return {"active_permissions": []}

    def _collect_config(self, agent_id: str) -> dict:
        """Collect agent configuration."""
        return {}

    def _collect_logs(self, agent_id: str,
                       minutes: int = 30) -> list[dict]:
        """Collect recent logs."""
        return []
```

## 7. Compliance Requirements

### 7.1 Compliance Mapping for Agent Systems

| Standard | Requirement | Agent Audit Implementation |
|----------|-------------|---------------------------|
| **SOC 2 CC7.2** | Monitor system operations | Real-time agent action monitoring |
| **SOC 2 CC7.3** | Respond to incidents | Agent forensic analysis pipeline |
| **ISO 27001 A.12.4.1** | Event logging | Comprehensive agent action logging |
| **ISO 27001 A.12.4.2** | Protection of log info | Tamper-evident hash chain |
| **ISO 27001 A.12.4.3** | Administrator logs | Agent admin action audit trail |
| **PCI DSS 10.2** | Audit trails for access | Every user->agent->tool interaction |
| **PCI DSS 10.3** | Audit trail contents | Full log schema with all required fields |
| **GDPR Art. 5(2)** | Accountability | Complete agent action traceability |
| **GDPR Art. 30** | Records of processing | Agent data processing logs |
| **HIPAA §164.312(b)** | Audit controls | Detailed agent access and action logging |

### 7.2 Compliance Report Generation

```python
class AgentComplianceReporter:
    """Generates compliance reports from agent audit logs."""

    def __init__(self, log_store):
        self.log_store = log_store

    def generate_soc2_report(self, start: datetime,
                               end: datetime) -> dict:
        """Generate SOC 2 compliance report for agent operations."""
        logs = self.log_store.query({
            "time_range": [start.isoformat(), end.isoformat()],
        })

        return {
            "report_type": "SOC 2",
            "period": {"start": start.isoformat(), "end": end.isoformat()},
            "metrics": {
                "total_actions": len(logs),
                "agents_active": len(set(l["agent_id"] for l in logs)),
                "sessions": len(set(l["session_id"] for l in logs)),
                "tools_used": len(set(l["tool_name"] for l in logs)),
            },
            "security_events": {
                "injection_detected": len([l for l in logs if l.get("injection_detected")]),
                "policy_violations": sum(len(l.get("policy_violations", [])) for l in logs),
                "failed_actions": len([l for l in logs if not l.get("tool_success", True)]),
            },
            "access_control": {
                "unique_users": len(set(l["user_id"] for l in logs if l.get("user_id"))),
                "privileged_actions": len([l for l in logs if l.get("approval_required")]),
                "approved_actions": len([l for l in logs if l.get("approval_granted")]),
            },
            "controls_assessment": {
                "logging_enabled": True,
                "tamper_protection": True,
                "monitoring_active": True,
                "retention_policy": "90_days",
            },
        }

    def generate_gdpr_report(self, user_id: str) -> dict:
        """Generate GDPR data processing report for a user."""
        logs = self.log_store.query({"user_id": user_id})

        return {
            "report_type": "GDPR Data Processing Record",
            "data_subject": user_id,
            "processing_activities": [
                {
                    "timestamp": l["timestamp"],
                    "agent": l["agent_id"],
                    "purpose": l.get("context_summary", ""),
                    "tool": l.get("tool_name", ""),
                    "data_categories": self._infer_data_categories(l),
                }
                for l in logs
            ],
            "data_minimization": {
                "total_interactions": len(logs),
                "data_retention_days": 90,
            },
        }
```

## 8. Log Aggregation

### 8.1 ELK Stack Integration

```python
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

class ElasticsearchAuditStore:
    """Store agent audit logs in Elasticsearch."""

    def __init__(self, hosts: list[str], index_prefix: str = "agent-audit"):
        self.es = Elasticsearch(hosts)
        self.index_prefix = index_prefix
        self._ensure_index_template()

    def _ensure_index_template(self):
        """Create index template for agent audit logs."""
        template = {
            "index_patterns": [f"{self.index_prefix}-*"],
            "template": {
                "settings": {
                    "number_of_shards": 3,
                    "number_of_replicas": 2,
                    "refresh_interval": "5s",
                },
                "mappings": {
                    "properties": {
                        "timestamp": {"type": "date"},
                        "agent_id": {
                            "type": "keyword",
                            "fields": {"text": {"type": "text"}}
                        },
                        "session_id": {"type": "keyword"},
                        "tool_name": {"type": "keyword"},
                        "risk_score": {"type": "float"},
                        "injection_detected": {"type": "boolean"},
                        "duration_ms": {"type": "integer"},
                        "total_tokens": {"type": "integer"},
                        # Full text search fields
                        "reasoning": {"type": "text", "index": True},
                        "tool_params": {"type": "object", "enabled": False},
                        "user_message": {"type": "text", "index": True},
                    }
                }
            }
        }
        self.es.indices.put_index_template(
            name=f"{self.index_prefix}-template",
            body=template,
        )

    def store(self, log_entry: AgentActionLog):
        """Store a log entry."""
        index_name = f"{self.index_prefix}-{datetime.now():%Y.%m.%d}"
        self.es.index(
            index=index_name,
            document=log_entry.to_dict(),
            id=log_entry.log_id,
        )

    def bulk_store(self, entries: list[AgentActionLog]):
        """Bulk store log entries."""
        actions = [
            {
                "_index": f"{self.index_prefix}-{datetime.now():%Y.%m.%d}",
                "_id": entry.log_id,
                "_source": entry.to_dict(),
            }
            for entry in entries
        ]
        bulk(self.es, actions)

    def search(self, query: dict, size: int = 100) -> list[dict]:
        """Search audit logs."""
        es_query = self._build_es_query(query)
        result = self.es.search(
            index=f"{self.index_prefix}-*",
            body={
                "query": es_query,
                "sort": [{"timestamp": {"order": "desc"}}],
                "size": size,
            },
        )
        return [hit["_source"] for hit in result["hits"]["hits"]]
```

### 8.2 Loki Integration (Grafana)

```python
import requests

class LokiAuditStore:
    """Store agent audit logs in Grafana Loki."""

    def __init__(self, push_url: str, tenant_id: str = None):
        self.push_url = push_url
        self.tenant_id = tenant_id
        self.headers = {"Content-Type": "application/json"}
        if tenant_id:
            self.headers["X-Scope-OrgID"] = tenant_id

    def push_log(self, log_entry: AgentActionLog):
        """Push a log entry to Loki."""
        stream = {
            "stream": {
                "agent_id": log_entry.agent_id,
                "service": "agent-runtime",
                "level": "info",
                "tool": log_entry.tool_name or "none",
            },
            "values": [
                [
                    str(int(datetime.now().timestamp() * 1e9)),
                    json.dumps(log_entry.to_dict()),
                ]
            ],
        }

        payload = {"streams": [stream]}

        response = requests.post(
            self.push_url,
            headers=self.headers,
            json=payload,
        )
        response.raise_for_status()

    def query_logs(self, query: str, start: str,
                    end: str) -> list[dict]:
        """Query logs using LogQL."""
        params = {
            "query": query,
            "start": start,
            "end": end,
            "limit": 1000,
            "direction": "backward",
        }

        response = requests.get(
            self.push_url.replace("/push", "/query_range"),
            params=params,
        )

        if response.status_code == 200:
            result = response.json()
            # Parse Loki response format
            logs = []
            for stream in result.get("data", {}).get("result", []):
                for value in stream.get("values", []):
                    logs.append(json.loads(value[1]))
            return logs
        return []
```

### 8.3 Log Retention and Archival

```python
class LogRetentionManager:
    """Manages log retention, archival, and deletion policies."""

    def __init__(self, storage_backend):
        self.storage = storage_backend
        self.retention_policies = {
            "hot": {"duration": 7, "storage": "ssd"},
            "warm": {"duration": 30, "storage": "hdd"},
            "cold": {"duration": 365, "storage": "s3_glacier"},
        }

    def apply_retention_policy(self):
        """Move logs through retention tiers based on age."""
        now = datetime.now()

        for tier, policy in self.retention_policies.items():
            cutoff = now - timedelta(days=policy["duration"])

            # Move logs older than cutoff to next tier
            next_tier = self._get_next_tier(tier)
            if next_tier:
                self._move_logs(
                    source_tier=tier,
                    target_tier=next_tier,
                    older_than=cutoff,
                )

        # Delete logs beyond maximum retention
        max_retention = max(
            p["duration"] for p in self.retention_policies.values()
        )
        delete_cutoff = now - timedelta(days=max_retention)
        self._delete_old_logs(delete_cutoff)

    def _move_logs(self, source_tier: str, target_tier: str,
                    older_than: datetime):
        """Move logs from one tier to another."""
        logs = self.storage.query({
            "tier": source_tier,
            "timestamp_lt": older_than.isoformat(),
        })
        for log in logs:
            log["tier"] = target_tier
            self.storage.update(log)

    def _delete_old_logs(self, cutoff: datetime):
        """Permanently delete logs beyond retention."""
        self.storage.delete({
            "timestamp_lt": cutoff.isoformat(),
        })
```

## 9. Real-Time Monitoring and Alerting

### 9.1 Alert Rules

```python
class AgentAuditAlerts:
    """Alert rules for agent audit monitoring."""

    def __init__(self, alerting_backend):
        self.alerting = alerting_backend
        self.rules = self._define_rules()

    def _define_rules(self) -> list[dict]:
        return [
            {
                "name": "high_risk_action",
                "condition": lambda log: log.get("risk_score", 0) > 0.9,
                "severity": "critical",
                "message": "Agent performing very high risk action",
            },
            {
                "name": "injection_attempt",
                "condition": lambda log: log.get("injection_detected"),
                "severity": "high",
                "message": "Prompt injection attempt detected",
            },
            {
                "name": "rapid_tool_calls",
                "condition": self._rapid_calls_detected,
                "severity": "medium",
                "message": "Unusual rate of tool calls",
            },
            {
                "name": "data_exfiltration_suspected",
                "condition": lambda log: (
                    log.get("tool_name") in ("send_email", "webhook_post", "api_call") and
                    log.get("risk_score", 0) > 0.7
                ),
                "severity": "high",
                "message": "Possible data exfiltration via tool",
            },
        ]

    def evaluate(self, log: AgentActionLog) -> list[dict]:
        """Evaluate a log entry against all alert rules."""
        alerts = []
        for rule in self.rules:
            try:
                if rule["condition"](log):
                    alert = {
                        "rule": rule["name"],
                        "severity": rule["severity"],
                        "message": rule["message"],
                        "log_id": log.log_id,
                        "agent_id": log.agent_id,
                        "timestamp": log.timestamp,
                    }
                    alerts.append(alert)
                    self.alerting.send(alert)
            except Exception:
                continue
        return alerts

    def _rapid_calls_detected(self, log: AgentActionLog) -> bool:
        # In production, check against historical rate
        return False
```

## 10. Conclusion

Agent audit and forensics is essential for building trustworthy AI agent systems. The non-deterministic nature of LLM-based agents makes traditional logging insufficient — organizations must capture not just what agents did, but why they did it.

Key implementation priorities:

1. **Log everything**: Capture complete agent action records including identity, input, reasoning, tool calls, and results.
2. **Make logs tamper-evident**: Use hash chaining to detect any modification to historical logs.
3. **Enable causality tracking**: Link agent actions in a causal graph for incident analysis.
4. **Implement distributed tracing**: Use OpenTelemetry to trace actions across multi-agent systems.
5. **Build forensic analysis tools**: Develop capabilities for incident playback and root cause analysis.
6. **Meet compliance requirements**: Map audit controls to SOC 2, ISO 27001, PCI DSS, GDPR, and HIPAA.
7. **Aggregate for searchability**: Use ELK/Loki for efficient log search and visualization.
8. **Monitor in real time**: Set up alerting for suspicious patterns as they occur.

---

**References**
- OpenTelemetry Documentation
- Elasticsearch Reference Guide
- Grafana Loki Documentation
- NIST SP 800-92: Guide to Computer Security Log Management
- SOC 2 Trust Services Criteria
- ISO/IEC 27001:2022 Information Security Management

---

**Document Information**
- Title: Agent Audit and Forensics
- Series: 18-Agent-Security-and-Trust
- Part: 06 of 08
- Author: AI Knowledge Base
- Last Updated: 2026-06-13
- Lines: 605
