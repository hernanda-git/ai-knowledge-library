# 54 — AI Agent State Management & Persistence: Core Topics

> Last updated: July 5, 2026

This document dives deep into the core concepts, patterns, and techniques that form the foundation of AI agent state management. Each topic includes practical examples, trade-offs, and implementation guidance.

---

## Table of Contents

1. [State Representation](#1-state-representation)
2. [Checkpointing Strategies](#2-checkpointing-strategies)
3. [State Serialization and Encoding](#3-state-serialization-and-encoding)
4. [State Recovery and Resumption](#4-state-recovery-and-resumption)
5. [State Scoping and Lifecycle](#5-state-scoping-and-lifecycle)
6. [Concurrent State Access](#6-concurrent-state-access)
7. [State Consistency Models](#7-state-consistency-models)
8. [State Schema Evolution](#8-state-schema-evolution)
9. [State Cleanup and Retention](#9-state-cleanup-and-retention)
10. [State Observability](#10-state-observability)

---

## 1. State Representation

### 1.1 The State Tree

Agent state is best represented as a **tree** (or graph) rather than a flat dictionary. This enables partial updates, lazy loading, and fine-grained checkpointing.

```
AgentState (root)
├── metadata
│   ├── task_id
│   ├── agent_version
│   ├── created_at
│   ├── last_checkpoint_at
│   └── schema_version
├── execution
│   ├── plan
│   │   ├── steps[]
│   │   └── current_step_index
│   ├── status (pending|running|paused|completed|failed)
│   └── error (if failed)
├── accumulated_data
│   ├── tool_results[]
│   ├── user_inputs[]
│   └── intermediate_conclusions[]
├── resource_handles
│   ├── open_connections[]
│   ├── locks_held[]
│   └── file_handles[]
├── cost_tracking
│   ├── tokens_used
│   ├── api_calls_made
│   ├── estimated_cost
│   └── budget_remaining
├── user_context
│   ├── preferences
│   ├── session_id
│   └── conversation_history_summary
└── recovery_metadata
    ├── last_checkpoint_id
    ├── checkpoint_chain[]
    └── recovery_count
```

### 1.2 Typed State vs Dynamic State

**Typed state** (Pydantic, dataclasses, protobuf):
```python
from pydantic import BaseModel
from typing import List, Optional
from enum import Enum

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"

class ToolResult(BaseModel):
    tool_name: str
    args: dict
    result: str
    latency_ms: float
    cost: float
    timestamp: str

class AgentState(BaseModel):
    task_id: str
    agent_version: str
    status: TaskStatus
    current_step: int
    total_steps: int
    plan: List[str]
    tool_results: List[ToolResult]
    tokens_used: int = 0
    estimated_cost: float = 0.0
    error: Optional[str] = None
    started_at: str
    last_checkpoint_at: str
    
    class Config:
        # Enable schema versioning
        schema_version = 1
```

**Dynamic state** (dict, JSON):
```python
state = {
    "task_id": "abc-123",
    "status": "running",
    "step": 3,
    "results": [...],  # Can be anything
}
```

| Aspect | Typed | Dynamic |
|--------|-------|---------|
| Validation | ✅ Automatic | ❌ Manual |
| IDE support | ✅ Full | ❌ None |
| Schema evolution | ⚠️ Migration needed | ✅ Flexible |
| Serialization | ⚠️ Requires model | ✅ Direct |
| Performance | ✅ Faster (pre-allocated) | ⚠️ Slower (dict overhead) |
| Debugging | ✅ Clear structure | ⚠️ Hard to inspect |

**Recommendation:** Use typed state for production systems, dynamic state for prototyping.

### 1.3 State Compression

Large agent state (e.g., accumulated search results, code artifacts) can be compressed:

```python
import gzip
import base64

class CompressedField:
    """Compress large fields in agent state."""
    
    @staticmethod
    def compress(data: bytes) -> str:
        compressed = gzip.compress(data, compresslevel=6)
        return base64.b64encode(compressed).decode('ascii')
    
    @staticmethod
    def decompress(encoded: str) -> bytes:
        compressed = base64.b64decode(encoded.encode('ascii'))
        return gzip.decompress(compressed)

# Example: compressing large tool results
state["large_result"] = CompressedField.compress(huge_search_results.encode())
# 10MB → ~1MB after compression
```

---

## 2. Checkpointing Strategies

### 2.1 Eager vs Lazy Checkpointing

**Eager checkpointing:** Write state to durable storage after every mutation.

```python
class EagerCheckpointer:
    def __init__(self, storage):
        self.storage = storage
    
    async def update(self, state: AgentState):
        # Every state change triggers a write
        await self.storage.write(state.task_id, state.serialize())
    
    async def on_tool_result(self, state: AgentState, result: ToolResult):
        state.tool_results.append(result)
        await self.update(state)  # Write immediately
    
    async def on_step_complete(self, state: AgentState):
        state.current_step += 1
        await self.update(state)  # Write immediately
```

**Lazy checkpointing:** Batch writes and only flush periodically or on significant events.

```python
class LazyCheckpointer:
    def __init__(self, storage, flush_interval_s=30):
        self.storage = storage
        self.flush_interval = flush_interval_s
        self.dirty = False
        self.last_flush = time.time()
    
    async def mark_dirty(self, state: AgentState):
        self.dirty = True
        # Flush if interval exceeded or state is large
        if (time.time() - self.last_flush > self.flush_interval or
            state.estimated_size() > 1_000_000):
            await self.flush(state)
    
    async def flush(self, state: AgentState):
        if self.dirty:
            await self.storage.write(state.task_id, state.serialize())
            self.dirty = False
            self.last_flush = time.time()
```

### 2.2 Checkpoint Granularity

| Granularity | Frequency | Overhead | Recovery Point | Use Case |
|-------------|-----------|----------|---------------|----------|
| **Per LLM call** | Every API call | High | Maximum precision | Debugging, compliance |
| **Per step** | Every plan step | Medium | Good precision | General purpose |
| **Per phase** | Major milestones | Low | May lose inter-step work | Long-running, cost-sensitive |
| **On error only** | Failure moments | Minimal | May lose significant work | Stateless-ish agents |
| **Time-based** | Every N seconds | Configurable | Up to N seconds of work | Background agents |

### 2.3 Async Checkpointing with Write-Behind

The most common production pattern:

```python
import asyncio
from collections import deque

class AsyncCheckpointer:
    def __init__(self, storage, max_batch_size=10, max_wait_ms=1000):
        self.storage = storage
        self.queue = deque()
        self.max_batch = max_batch_size
        self.max_wait = max_wait_ms / 1000
        self._running = False
    
    async def checkpoint(self, task_id: str, state: AgentState):
        """Non-blocking checkpoint — queues the write."""
        self.queue.append((task_id, state.copy(), time.time()))
        if not self._running:
            asyncio.create_task(self._flush_loop())
    
    async def _flush_loop(self):
        self._running = True
        while self.queue:
            batch = []
            deadline = time.time() + self.max_wait
            
            while self.queue and len(batch) < self.max_batch:
                item = self.queue.popleft()
                batch.append(item)
                if time.time() >= deadline:
                    break
            
            # Write batch to storage
            await self.storage.batch_write([
                {"key": f"state:{item[0]}", "value": item[1].serialize()}
                for item in batch
            ])
        
        self._running = False
    
    async def force_checkpoint(self, task_id: str, state: AgentState):
        """Synchronous checkpoint for critical moments."""
        await self.storage.write(f"state:{task_id}", state.serialize())
```

### 2.4 Incremental Checkpointing

Instead of writing full state every time, only write what changed:

```python
class IncrementalCheckpointer:
    def __init__(self, storage):
        self.storage = storage
        self.last_checkpoint = {}
        self.sequence_number = 0
    
    async def checkpoint(self, task_id: str, full_state: dict):
        # Compute diff against last checkpoint
        diff = self._compute_diff(self.last_checkpoint, full_state)
        
        if diff:
            self.sequence_number += 1
            delta = {
                "seq": self.sequence_number,
                "diff": diff,
                "timestamp": time.time()
            }
            await self.storage.append(
                f"state:{task_id}:log", 
                delta
            )
            self.last_checkpoint = full_state.copy()
    
    async def restore(self, task_id: str) -> dict:
        # Load base snapshot
        base = await self.storage.get(f"state:{task_id}:base")
        # Apply all deltas
        deltas = await self.storage.get_all(f"state:{task_id}:log")
        for delta in sorted(deltas, key=lambda d: d["seq"]):
            base = self._apply_diff(base, delta["diff"])
        return base
    
    def _compute_diff(self, old: dict, new: dict) -> dict:
        """Simple recursive diff."""
        diff = {}
        all_keys = set(list(old.keys()) + list(new.keys()))
        for key in all_keys:
            if key not in old:
                diff[key] = {"op": "add", "value": new[key]}
            elif key not in new:
                diff[key] = {"op": "remove"}
            elif old[key] != new[key]:
                diff[key] = {"op": "replace", "value": new[key]}
        return diff
```

---

## 3. State Serialization and Encoding

### 3.1 Choosing a Format

| Format | Latency (1KB) | Size | Schema Support | Python | JS | Go |
|--------|--------------|------|---------------|--------|----|----|
| JSON | 45μs | 1.0x | None | ✅ | ✅ | ✅ |
| MessagePack | 12μs | 0.7x | None | ✅ | ✅ | ✅ |
| Protocol Buffers | 8μs | 0.5x | Excellent | ✅ | ✅ | ✅ |
| Avro | 15μs | 0.6x | Excellent | ✅ | ✅ | �ora |
| BSON | 35μs | 0.9x | Partial | ✅ | ✅ | ❌ |
| FlatBuffers | 2μs | 0.5x | Excellent | ⚠️ | ✅ | ✅ |
| Cap'n Proto | 1μs | 0.5x | Excellent | ⚠️ | ❌ | ✅ |

### 3.2 Python Serialization Examples

```python
# JSON (simple, debuggable)
import json
state_bytes = json.dumps(state, default=str).encode()
state = json.loads(state_bytes.decode())

# MessagePack (fast, compact)
import msgpack
state_bytes = msgpack.packb(state, default=str)
state = msgpack.unpackb(state_bytes, raw=False)

# Protocol Buffers (schema-enforced, compact)
# Define in .proto file:
# message AgentState {
#   string task_id = 1;
#   int32 current_step = 2;
#   repeated ToolResult tool_results = 3;
# }
from agent_state_pb2 import AgentState as PBState
pb_state = PBState(task_id="abc", current_step=3)
state_bytes = pb_state.SerializeToString()
pb_state.ParseFromString(state_bytes)
```

### 3.3 Handling Non-Serializable Objects

Agents often accumulate non-serializable objects (connections, locks, open files):

```python
import json
from dataclasses import dataclass, field, asdict

class NonSerializableHandler:
    """Handles objects that can't be directly serialized."""
    
    SPECIAL_TYPES = {
        "connection": lambda obj: {"type": "connection", "url": obj.url, "pool_id": id(obj)},
        "lock": lambda obj: {"type": "lock", "resource": obj.resource, "holder": obj.holder},
        "file_handle": lambda obj: {"type": "file", "path": obj.name, "mode": obj.mode},
    }
    
    @classmethod
    def serialize(cls, state: dict) -> str:
        def default_handler(obj):
            for type_name, serializer in cls.SPECIAL_TYPES.items():
                if hasattr(obj, '__class__') and type_name in obj.__class__.__name__.lower():
                    return serializer(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        return json.dumps(state, default=default_handler)
    
    @classmethod
    def deserialize(cls, state_str: str, reconnect_fn=None) -> dict:
        def object_hook(obj):
            if isinstance(obj, dict) and "type" in obj:
                if obj["type"] == "connection" and reconnect_fn:
                    return reconnect_fn(obj["url"])
                # Other types: store metadata, don't reconnect
                return {"_reconnect_metadata": obj}
            return obj
        
        return json.loads(state_str, object_hook=object_hook)
```

---

## 4. State Recovery and Resumption

### 4.1 Recovery Strategies

**Full recovery:** Reconstruct exact state from checkpoint.
```python
async def recover_full(task_id: str, storage) -> AgentState:
    checkpoint = await storage.get_latest_checkpoint(task_id)
    return AgentState.deserialize(checkpoint)
```

**Replay recovery:** Start from base checkpoint, replay event log.
```python
async def recover_replay(task_id: str, storage) -> AgentState:
    base = await storage.get_base_checkpoint(task_id)
    events = await storage.get_event_log(task_id)
    state = AgentState.deserialize(base)
    for event in events:
        state = apply_event(state, event)
    return state
```

**Approximate recovery:** Reconstruct from available data, accept some loss.
```python
async def recover_approximate(task_id: str, storage) -> AgentState:
    # Try latest checkpoint
    latest = await storage.get_latest_checkpoint(task_id)
    if latest:
        return AgentState.deserialize(latest)
    
    # Fall back to any available partial state
    partial = await storage.get_any_state(task_id)
    if partial:
        state = AgentState.deserialize(partial)
        state.status = "recovering"  # Signal incomplete state
        return state
    
    # Start fresh
    return AgentState(task_id=task_id, status="fresh_start")
```

### 4.2 Idempotent Recovery

When recovering, agent steps must be **idempotent** — executing them again should produce the same result:

```python
class IdempotentAgent:
    def __init__(self):
        self.completed_steps = set()
    
    async def execute_step(self, step: int, state: AgentState):
        if step in self.completed_steps:
            print(f"Step {step} already completed, skipping")
            return state
        
        # Execute with idempotency key
        result = await self._do_step(step, state, idempotency_key=state.task_id)
        self.completed_steps.add(step)
        state.current_step = step + 1
        state.tool_results.append(result)
        return state
```

### 4.3 Recovery Time Objectives

| RTO Target | Strategy | Trade-off |
|------------|----------|-----------|
| < 1 second | In-memory + async flush to durable | Risk of 1s data loss |
| < 10 seconds | Redis + periodic sync | Moderate complexity |
| < 1 minute | PostgreSQL + WAL | Strong consistency |
| < 5 minutes | S3 + event log replay | Slow but very durable |
| Any time | Full event replay | Maximum flexibility, maximum replay time |

---

## 5. State Scoping and Lifecycle

### 5.1 State Scopes

```
Global State (shared across all agents)
├── Organization settings
├── Shared resource pools
├── Global counters
└── Configuration

Agent Pool State (shared across agents of same type)
├── Model configuration
├── Tool registry
├── Rate limits
└── Shared cache

Task State (unique per task execution)
├── Execution plan
├── Accumulated results
├── Cost tracking
└── Error context

Turn State (unique per LLM call within a task)
├── Current prompt
├── Context window contents
├── Tool call parameters
└── Partial response
```

### 5.2 State Transitions

```
                    ┌──────────────┐
                    │   Created    │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
              ┌─────│   Running    │─────┐
              │     └──────┬───────┘     │
              │            │             │
        ┌─────▼─────┐     │      ┌──────▼──────┐
        │  Paused   │     │      │  Failed     │
        └─────┬─────┘     │      └──────┬──────┘
              │            │             │
              │     ┌──────▼───────┐     │
              └────►│  Completed   │◄────┘
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Archived    │
                    └──────────────┘
```

### 5.3 State Ownership

| Scope | Owner | Access | Lifecycle |
|-------|-------|--------|-----------|
| **Task state** | Single agent | Read/write | Task start → completion |
| **Shared state** | Coordination layer | Read/write (atomic) | Session start → end |
| **Read-only state** | Any agent | Read only | Configuration lifetime |
| **Ephemeral state** | Single LLM call | Read/write | One turn |

---

## 6. Concurrent State Access

### 6.1 The Concurrency Problem

When multiple agents or parallel branches share state:

```
Agent A reads state.step = 3
Agent B reads state.step = 3
Agent A writes state.step = 4
Agent B writes state.step = 4  ← Lost update! Should be 5
```

### 6.2 Locking Strategies

**Pessimistic locking (distributed lock):**
```python
import redis

class DistributedLock:
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def acquire(self, resource: str, timeout_s: int = 30) -> bool:
        return await self.redis.set(
            f"lock:{resource}", 
            "locked", 
            nx=True, 
            ex=timeout_s
        )
    
    async def release(self, resource: str):
        await self.redis.delete(f"lock:{resource}")

# Usage
lock = DistributedLock(redis)
if await lock.acquire(f"state:{task_id}"):
    try:
        state = await load_state(task_id)
        state.step += 1
        await save_state(task_id, state)
    finally:
        await lock.release(f"state:{task_id}")
```

**Optimistic locking (version check):**
```python
class OptimisticLock:
    def __init__(self, storage):
        self.storage = storage
    
    async def update_with_version(self, task_id: str, updater):
        """Retry loop for optimistic concurrency."""
        max_retries = 5
        for attempt in range(max_retries):
            state, version = await self.storage.get_with_version(task_id)
            new_state = updater(state)
            
            success = await self.storage.compare_and_swap(
                task_id, 
                expected_version=version,
                new_state=new_state
            )
            if success:
                return new_state
            
            # Version changed, retry
            await asyncio.sleep(0.01 * (2 ** attempt))
        
        raise ConcurrencyError(f"Failed to update {task_id} after {max_retries} attempts")
```

**CRDTs (Conflict-free Replicated Data Types):**
```python
# For state that can be merged without conflicts
from crdt import GCounter, ORSet

class ConcurrentAgentState:
    def __init__(self):
        self.step_counter = GCounter()  # Always increments
        self.tool_results = ORSet()      # Set union
        self.aggregated_metrics = PNCounter()  # Additive metrics
    
    def merge(self, other: "ConcurrentAgentState"):
        """Merge is always safe — no conflicts possible."""
        self.step_counter.merge(other.step_counter)
        self.tool_results.merge(other.tool_results)
        self.aggregated_metrics.merge(other.aggregated_metrics)
```

---

## 7. State Consistency Models

### 7.1 Consistency Spectrum

```
Strong ←─────────────────────────────────────────→ Eventual

│ Linearizability │ Sequential │ Causal │ Eventual │
│ (strictest)     │            │        │ (weakest)│

Use cases:
  Financial ops  │  Agent    │  Agent  │  Monitoring│
  Compliance     │  workflows│  memory │  Metrics   │
```

### 7.2 Consistency by Component

| Component | Recommended Consistency | Why |
|-----------|------------------------|-----|
| Checkpoint writes | Linearizable | Must not lose state |
| Cost tracking | Eventual | Approximate is OK |
| Tool result storage | Causal | Order matters within task |
| Audit log | Linearizable | Compliance requirement |
| Metrics | Eventual | Approximate is OK |
| Shared state (multi-agent) | Linearizable or Causal | Depends on coordination needs |

---

## 8. State Schema Evolution

### 8.1 Versioned Schemas

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class AgentStateV1:
    task_id: str
    step: int
    results: list

@dataclass
class AgentStateV2:
    task_id: str
    step: int
    results: list
    cost: float  # New field
    user_id: Optional[str] = None  # New field with default

@dataclass  
class AgentStateV3:
    task_id: str
    step: int
    results: list
    cost: float
    user_id: Optional[str] = None
    tags: dict = None  # New field with default

# Migration functions
def migrate_v1_to_v2(state: dict) -> dict:
    state["cost"] = 0.0
    state["user_id"] = None
    state["_schema_version"] = 2
    return state

def migrate_v2_to_v3(state: dict) -> dict:
    state["tags"] = {}
    state["_schema_version"] = 3
    return state

MIGRATIONS = {
    1: migrate_v1_to_v2,
    2: migrate_v2_to_v3,
}

def migrate_state(state: dict) -> dict:
    version = state.get("_schema_version", 1)
    while version in MIGRATIONS:
        state = MIGRATIONS[version](state)
        version = state["_schema_version"]
    return state
```

### 8.2 Forward and Backward Compatibility

- **Forward compatible:** New fields have defaults; old code ignores unknown fields
- **Backward compatible:** New code handles missing fields gracefully
- **Field deprecation:** Mark fields as deprecated before removing

```python
from pydantic import BaseModel, Field

class AgentState(BaseModel):
    task_id: str
    step: int
    results: list = []
    
    # Deprecated in v2 — use cost_breakdown instead
    cost: float = Field(default=0.0, deprecated=True)
    
    # New in v2
    cost_breakdown: dict = Field(default_factory=dict)
    
    # New in v3
    tags: dict = Field(default_factory=dict)
```

---

## 9. State Cleanup and Retention

### 9.1 Retention Policies

```python
from datetime import datetime, timedelta
from enum import Enum

class RetentionPolicy(Enum):
    KEEP_FOR_1_HOUR = timedelta(hours=1)
    KEEP_FOR_24_HOURS = timedelta(hours=24)
    KEEP_FOR_7_DAYS = timedelta(days=7)
    KEEP_FOR_30_DAYS = timedelta(days=30)
    KEEP_FOREVER = None
    KEEP_UNTIL_SIZE_LIMIT = "size_based"

class StateRetentionPolicy:
    def __init__(self, policy: RetentionPolicy, max_size_mb: int = 100):
        self.policy = policy
        self.max_size_mb = max_size_mb
    
    def should_delete(self, state_metadata: dict) -> bool:
        if self.policy == RetentionPolicy.KEEP_FOREVER:
            return False
        
        if self.policy == RetentionPolicy.KEEP_UNTIL_SIZE_LIMIT:
            return state_metadata.get("size_mb", 0) > self.max_size_mb
        
        if isinstance(self.policy.value, timedelta):
            age = datetime.now() - datetime.fromisoformat(state_metadata["created_at"])
            return age > self.policy.value
        
        return False
    
    def get_archive_action(self, state_metadata: dict) -> str:
        """What to do with state past retention."""
        if state_metadata.get("status") == "completed":
            return "archive_to_cold_storage"
        elif state_metadata.get("status") == "failed":
            return "delete"  # Failed tasks, no audit needed
        else:
            return "archive_with_alert"  # Incomplete — investigate
```

### 9.2 Storage Tiering

```
Hot (Redis/DynamoDB)     → Active state, < 1 hour old
  ↓ (after 1 hour)
Warm (PostgreSQL)        → Recent state, 1-7 days old
  ↓ (after 7 days)  
Cold (S3/GCS)            → Archived state, 30+ days old
  ↓ (after 30 days)
Glacier/Deep Archive     → Compliance retention
  ↓ (after retention period)
Delete                   → Permanent removal
```

---

## 10. State Observability

### 10.1 Key Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|----------------|
| `checkpoint_duration_ms` | Time to write a checkpoint | > 100ms |
| `checkpoint_size_bytes` | Size of serialized state | > 10MB |
| `recovery_time_seconds` | Time to recover from checkpoint | > 30s |
| `state_loss_incidents` | Number of state loss events | > 0 |
| `checkpoint_frequency` | Checkpoints per minute | < 1/min for long tasks |
| `state_drift_ratio` | % of state changed since last checkpoint | > 80% (checkpoint too infrequent) |

### 10.2 Monitoring Setup

```python
from prometheus_client import Histogram, Counter, Gauge

checkpoint_duration = Histogram(
    'agent_checkpoint_duration_seconds',
    'Time to write a checkpoint',
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
)

checkpoint_size = Histogram(
    'agent_checkpoint_size_bytes',
    'Size of serialized state',
    buckets=[1000, 10000, 100000, 1000000, 10000000]
)

recovery_time = Histogram(
    'agent_recovery_time_seconds',
    'Time to recover from checkpoint',
    buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0]
)

state_loss = Counter(
    'agent_state_loss_total',
    'Number of state loss incidents',
    ['agent_type', 'severity']
)

active_checkpoints = Gauge(
    'agent_active_checkpoints',
    'Number of active checkpoint operations'
)
```

### 10.3 Debugging State Issues

```python
class StateDebugger:
    """Tools for debugging agent state issues."""
    
    @staticmethod
    async def diff_checkpoints(storage, task_id: str, seq1: int, seq2: int):
        """Compare two checkpoints to understand state changes."""
        state1 = await storage.get_checkpoint(task_id, seq1)
        state2 = await storage.get_checkpoint(task_id, seq2)
        
        print(f"=== State diff: checkpoint {seq1} → {seq2} ===")
        print(f"Size: {len(state1)} → {len(state2)} bytes")
        
        # Find changed keys
        keys1 = set(state1.keys()) if isinstance(state1, dict) else set()
        keys2 = set(state2.keys()) if isinstance(state2, dict) else set()
        
        added = keys2 - keys1
        removed = keys1 - keys2
        modified = {k for k in keys1 & keys2 if state1[k] != state2[k]}
        
        print(f"Added: {added}")
        print(f"Removed: {removed}")
        print(f"Modified: {modified}")
    
    @staticmethod
    async def replay_and_inspect(storage, task_id: str, from_seq: int = 0):
        """Replay state changes and print each step."""
        events = await storage.get_events(task_id, from_seq)
        state = {}
        
        for i, event in enumerate(events):
            state = apply_event(state, event)
            print(f"\n--- Event {i + from_seq}: {event['type']} ---")
            print(f"State keys: {list(state.keys())}")
            print(f"State size: {len(json.dumps(state))} bytes")
```

---

## Cross-References

| Document | Relationship |
|----------|-------------|
| [01-Overview.md](01-Overview.md) | Introduction and high-level concepts |
| [03-Technical-Deep-Dive.md](03-Technical-Deep-Dive.md) | Advanced implementation patterns |
| [04-Tools-and-Frameworks.md](04-Tools-and-Frameworks.md) | Specific tools and platforms |
| [05-Future-Outlook.md](05-Future-Outlook.md) | Emerging trends and predictions |
| [31-AI-Workflow-Orchestration](../31-AI-Workflow-Orchestration-and-Durable-Execution/) | Workflow-level state coordination |
| [32-Agent-Memory-Systems](../32-Agent-Memory-Systems/) | Cross-session knowledge persistence |
| [20-Agent-Infrastructure](../20-Agent-Infrastructure-and-Observability/) | Observability for state tracking |

---

*This is the second document in the 54-AI-Agent-State-Management-and-Persistence series. See [01-Overview.md](01-Overview.md) for introduction and [03-Technical-Deep-Dive.md](03-Technical-Deep-Dive.md) for advanced patterns.*
