# Core Topics: Event-Driven Agent Architectures

> Deep dive into the core technical topics of event-driven agent architectures: event sourcing for agents, reactive patterns, stream processing, backpressure, schema evolution, and multi-agent coordination.

**Last Updated:** 2026-07-06  
**Estimated Reading Time:** 70 minutes  
**Prerequisites:** [01-Overview](./01-Overview.md)

---

## Table of Contents

1. [Event Sourcing for AI Agents](#1-event-sourcing-for-ai-agents)
2. [CQRS in Agent Systems](#2-cqrs-in-agent-systems)
3. [Reactive Streams and Backpressure](#3-reactive-streams-and-backpressure)
4. [Event Schema Design and Evolution](#4-event-schema-design-and-evolution)
5. [Multi-Agent Event Coordination](#5-multi-agent-event-coordination)
6. [Saga Pattern for Agent Workflows](#6-saga-pattern-for-agent-workflows)
7. [Event-Driven State Machines](#7-event-driven-state-machines)
8. [Dead Letter Queues and Error Handling](#8-dead-letter-queues-and-error-handling)
9. [Event Replay and Time Travel](#9-event-replay-and-time-travel)
10. [Security and Access Control](#10-security-and-access-control)
11. [Performance Optimization](#11-performance-optimization)
12. [Testing Event-Driven Agent Systems](#12-testing-event-driven-agent-systems)

---

## 1. Event Sourcing for AI Agents

### Why Event Sourcing?

Traditional state management for agents:

```
State = database.get(agent_id)
agent.process(event)
database.set(agent_id, new_state)
```

Event sourcing for agents:

```
events = event_store.get_events(agent_id)
state = replay(events)  # Derive current state from events
agent.process(event)
event_store.append(new_event)  # Never modify, only append
```

### Benefits for Agent Systems

1. **Complete audit trail** — Every decision an agent made is recorded
2. **Debugging** — Replay events to understand why an agent behaved a certain way
3. **Temporal queries** — What was the agent's state at time T?
4. **Compensation** — Undo agent actions by replaying compensating events
5. **Learning** — Use historical events for agent training and improvement

### Implementation

```python
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from datetime import datetime
import json
import hashlib

@dataclass
class AgentEvent:
    """Immutable event in the agent's event stream."""
    event_id: str
    event_type: str
    agent_id: str
    timestamp: str
    data: dict
    metadata: dict = field(default_factory=dict)
    version: int = 1
    
    def checksum(self) -> str:
        """Verify event integrity."""
        content = f"{self.event_id}{self.event_type}{self.agent_id}{self.timestamp}{json.dumps(self.data)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

class AgentEventStore:
    """Event store for a single agent."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.events: List[AgentEvent] = []
        self.snapshots: dict = {}
    
    def append(self, event: AgentEvent):
        """Append an event to the agent's stream."""
        assert event.agent_id == self.agent_id
        self.events.append(event)
    
    def get_events(self, after_version: int = 0) -> List[AgentEvent]:
        """Get events after a specific version."""
        return [e for e in self.events if e.version > after_version]
    
    def get_events_of_type(self, event_type: str) -> List[AgentEvent]:
        """Get all events of a specific type."""
        return [e for e in self.events if e.event_type == event_type]
    
    def create_snapshot(self, version: int, state: dict):
        """Create a state snapshot at a specific version."""
        self.snapshots[version] = {
            "state": state,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_latest_snapshot(self) -> Optional[dict]:
        """Get the most recent snapshot."""
        if not self.snapshots:
            return None
        latest_version = max(self.snapshots.keys())
        return self.snapshots[latest_version]

class EventSourcedAgent:
    """Agent that uses event sourcing for state management."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.store = AgentEventStore(agent_id)
        self.state = {}
        self._apply: dict = {}
    
    def register_handler(self, event_type: str, handler: Callable):
        """Register a state transition handler for an event type."""
        self._apply[event_type] = handler
    
    def process_event(self, event: AgentEvent):
        """Apply an event to update state."""
        if event.event_type in self._apply:
            self.state = self._apply[event.event_type](self.state, event)
        self.store.append(event)
    
    def emit_event(self, event_type: str, data: dict) -> AgentEvent:
        """Create and emit a new event."""
        version = len(self.store.events) + 1
        event = AgentEvent(
            event_id=f"evt_{self.agent_id}_{version}",
            event_type=event_type,
            agent_id=self.agent_id,
            timestamp=datetime.utcnow().isoformat(),
            data=data,
            version=version
        )
        self.process_event(event)
        return event
    
    def rebuild_state(self, up_to_version: int = None):
        """Rebuild state from events (event sourcing)."""
        snapshot = self.store.get_latest_snapshot()
        if snapshot:
            self.state = snapshot["state"]
            start_version = snapshot.get("version", 0)
        else:
            self.state = {}
            start_version = 0
        
        events = self.store.get_events(start_version)
        if up_to_version:
            events = [e for e in events if e.version <= up_to_version]
        
        for event in events:
            if event.event_type in self._apply:
                self.state = self._apply[event.event_type](self.state, event)
```

### Example: Research Agent with Event Sourcing

```python
class ResearchAgent(EventSourcedAgent):
    def __init__(self):
        super().__init__("research_agent_001")
        
        # Register state transition handlers
        self.register_handler("research.started", self._on_started)
        self.register_handler("source.found", self._on_source_found)
        self.register_handler("source.analyzed", self._on_source_analyzed)
        self.register_handler("research.completed", self._on_completed)
    
    def _on_started(self, state: dict, event: AgentEvent) -> dict:
        return {
            **state,
            "topic": event.data["topic"],
            "status": "in_progress",
            "sources_found": 0,
            "sources_analyzed": 0,
            "findings": []
        }
    
    def _on_source_found(self, state: dict, event: AgentEvent) -> dict:
        return {
            **state,
            "sources_found": state.get("sources_found", 0) + 1
        }
    
    def _on_source_analyzed(self, state: dict, event: AgentEvent) -> dict:
        return {
            **state,
            "sources_analyzed": state.get("sources_analyzed", 0) + 1,
            "findings": state.get("findings", []) + [event.data["finding"]]
        }
    
    def _on_completed(self, state: dict, event: AgentEvent) -> dict:
        return {
            **state,
            "status": "completed",
            "summary": event.data["summary"]
        }

# Usage
agent = ResearchAgent()

# Process events
agent.process_event(AgentEvent(
    event_id="evt_1",
    event_type="research.started",
    agent_id="research_agent_001",
    timestamp="2026-07-06T10:00:00Z",
    data={"topic": "AI event-driven architectures"}
))

agent.process_event(AgentEvent(
    event_id="evt_2",
    event_type="source.found",
    agent_id="research_agent_001",
    timestamp="2026-07-06T10:01:00Z",
    data={"source": "https://example.com/article1"}
))

# State is derived from events
print(agent.state)
# {'topic': 'AI event-driven architectures', 'status': 'in_progress', 
#  'sources_found': 1, 'sources_analyzed': 0, 'findings': []}

# Can rebuild state at any point
agent.store.create_snapshot(2, agent.state)
# Later, rebuild: agent.rebuild_state()
```

---

## 2. CQRS in Agent Systems

### Separating Commands and Queries

In CQRS, agent systems separate:

- **Command side:** Handles actions (emit events, change state)
- **Query side:** Handles reads (serve current state, generate reports)

```python
from abc import ABC, abstractmethod
from typing import Dict, Any

class Command(ABC):
    """Base class for commands (write operations)."""
    pass

class Query(ABC):
    """Base class for queries (read operations)."""
    pass

class CommandHandler(ABC):
    """Handles commands and emits events."""
    
    @abstractmethod
    async def handle(self, command: Command) -> List[AgentEvent]:
        pass

class QueryHandler(ABC):
    """Handles queries against read models."""
    
    @abstractmethod
    async def handle(self, query: Query) -> Any:
        pass

# Commands
class StartResearchCommand(Command):
    def __init__(self, agent_id: str, topic: str):
        self.agent_id = agent_id
        self.topic = topic

class AnalyzeSourceCommand(Command):
    def __init__(self, agent_id: str, source_url: str):
        self.agent_id = agent_id
        self.source_url = source_url

# Queries
class GetAgentStateQuery(Query):
    def __init__(self, agent_id: str):
        self.agent_id = agent_id

class GetResearchFindingsQuery(Query):
    def __init__(self, agent_id: str, min_confidence: float = 0.5):
        self.agent_id = agent_id
        self.min_confidence = min_confidence

# Command Handler
class ResearchCommandHandler(CommandHandler):
    def __init__(self, event_store: AgentEventStore):
        self.event_store = event_store
    
    async def handle(self, command: Command) -> List[AgentEvent]:
        if isinstance(command, StartResearchCommand):
            event = AgentEvent(
                event_id=f"evt_{command.agent_id}_{len(self.event_store.events)+1}",
                event_type="research.started",
                agent_id=command.agent_id,
                timestamp=datetime.utcnow().isoformat(),
                data={"topic": command.topic}
            )
            self.event_store.append(event)
            return [event]
        
        elif isinstance(command, AnalyzeSourceCommand):
            event = AgentEvent(
                event_id=f"evt_{command.agent_id}_{len(self.event_store.events)+1}",
                event_type="source.analyzed",
                agent_id=command.agent_id,
                timestamp=datetime.utcnow().isoformat(),
                data={"source_url": command.source_url}
            )
            self.event_store.append(event)
            return [event]

# Query Handler
class ResearchQueryHandler(QueryHandler):
    def __init__(self, read_model: Dict[str, Any]):
        self.read_model = read_model
    
    async def handle(self, query: Query) -> Any:
        if isinstance(query, GetAgentStateQuery):
            return self.read_model.get(query.agent_id, {})
        
        elif isinstance(query, GetResearchFindingsQuery):
            state = self.read_model.get(query.agent_id, {})
            findings = state.get("findings", [])
            return [f for f in findings if f.get("confidence", 0) >= query.min_confidence]
```

### Read Model Projections

Read models are updated by event handlers (projections):

```python
class AgentStateProjection:
    """Projects events into a read model for agent state."""
    
    def __init__(self, read_model: dict):
        self.read_model = read_model
        self.handlers = {
            "research.started": self._on_started,
            "source.found": self._on_source_found,
            "source.analyzed": self._on_source_analyzed,
            "research.completed": self._on_completed,
        }
    
    def project(self, event: AgentEvent):
        """Update read model based on event."""
        handler = self.handlers.get(event.event_type)
        if handler:
            handler(event)
    
    def _on_started(self, event: AgentEvent):
        self.read_model[event.agent_id] = {
            "status": "in_progress",
            "topic": event.data["topic"],
            "started_at": event.timestamp,
            "findings": [],
            "source_count": 0
        }
    
    def _on_source_found(self, event: AgentEvent):
        if event.agent_id in self.read_model:
            self.read_model[event.agent_id]["source_count"] += 1
    
    def _on_source_analyzed(self, event: AgentEvent):
        if event.agent_id in self.read_model:
            self.read_model[event.agent_id]["findings"].append(event.data)
    
    def _on_completed(self, event: AgentEvent):
        if event.agent_id in self.read_model:
            self.read_model[event.agent_id]["status"] = "completed"
            self.read_model[event.agent_id]["completed_at"] = event.timestamp

# Usage
read_model = {}
projection = AgentStateProjection(read_model)

# As events are appended, update read model
for event in event_store.get_events():
    projection.project(event)

# Now queries are fast against the read model
query_handler = ResearchQueryHandler(read_model)
state = await query_handler.handle(GetAgentStateQuery("research_agent_001"))
```

---

## 3. Reactive Streams and Backpressure

### The Problem

When agents process events faster than they can handle, the system becomes overwhelmed:

```
Event Producer: 1000 events/second
Agent Processing: 100 events/second
Queue: growing infinitely → OOM → crash
```

### Backpressure Mechanisms

```python
import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Optional
import time

@dataclass
class BackpressureConfig:
    max_queue_size: int = 1000
    high_water_mark: int = 800  # Start slowing at 80%
    low_water_mark: int = 200   # Resume at 20%
    max_processing_rate: float = 100.0  # events/second

class BackpressureAgent:
    """Agent with backpressure support."""
    
    def __init__(self, config: BackpressureConfig):
        self.config = config
        self.event_queue = deque()
        self.processing = False
        self.processed_count = 0
        self.last_rate_check = time.time()
        self.current_rate = 0.0
        self.paused = False
    
    async def receive_event(self, event):
        """Receive an event with backpressure."""
        if len(self.event_queue) >= self.config.high_water_mark:
            self.paused = True
            await self._wait_for_drain()
        
        self.event_queue.append(event)
    
    async def _wait_for_drain(self):
        """Wait until queue drains below low water mark."""
        while len(self.event_queue) > self.config.low_water_mark:
            await asyncio.sleep(0.1)
        self.paused = False
    
    async def process_loop(self):
        """Main processing loop with rate limiting."""
        while True:
            if self.paused or not self.event_queue:
                await asyncio.sleep(0.01)
                continue
            
            # Rate limiting
            now = time.time()
            elapsed = now - self.last_rate_check
            if elapsed >= 1.0:
                self.current_rate = self.processed_count / elapsed
                self.processed_count = 0
                self.last_rate_check = now
            
            if self.current_rate >= self.config.max_processing_rate:
                await asyncio.sleep(0.1)
                continue
            
            event = self.event_queue.popleft()
            await self.process(event)
            self.processed_count += 1
    
    async def process(self, event):
        """Override in subclass."""
        raise NotImplementedError
```

### Reactive Streams Pattern

```python
class ReactiveStream:
    """Reactive stream for agent event processing."""
    
    def __init__(self):
        self.subscribers = []
        self.buffer = []
        self.backpressure = BackpressureConfig()
    
    def subscribe(self, subscriber):
        self.subscribers.append(subscriber)
    
    async def push(self, event):
        """Push event to stream with backpressure."""
        if len(self.buffer) >= self.backpressure.high_water_mark:
            await self._apply_backpressure()
        
        self.buffer.append(event)
        await self._distribute(event)
    
    async def _apply_backpressure(self):
        """Signal upstream to slow down."""
        # In practice, this would use protocol-level signals
        # For Kafka: consumer.poll() with longer timeout
        # For Redis: BLPOP with timeout
        # For custom: pause acknowledgments
        while len(self.buffer) > self.backpressure.low_water_mark:
            await asyncio.sleep(0.01)
    
    async def _distribute(self, event):
        """Distribute event to all subscribers."""
        tasks = [sub.receive(event) for sub in self.subscribers]
        await asyncio.gather(*tasks, return_exceptions=True)
```

---

## 4. Event Schema Design and Evolution

### Schema Versioning Strategy

```python
from typing import Dict, Any, Optional
from dataclasses import dataclass
import json

@dataclass
class EventSchema:
    """Schema definition for an event type."""
    event_type: str
    version: int
    fields: Dict[str, str]  # field_name -> type
    required_fields: list
    optional_fields: list
    description: str

class SchemaRegistry:
    """Registry for event schemas with versioning."""
    
    def __init__(self):
        self.schemas: Dict[str, Dict[int, EventSchema]] = {}
    
    def register(self, schema: EventSchema):
        """Register a new schema version."""
        key = schema.event_type
        if key not in self.schemas:
            self.schemas[key] = {}
        self.schemas[key][schema.version] = schema
    
    def get_schema(self, event_type: str, version: int) -> Optional[EventSchema]:
        """Get a specific schema version."""
        return self.schemas.get(event_type, {}).get(version)
    
    def get_latest(self, event_type: str) -> Optional[EventSchema]:
        """Get the latest schema version."""
        versions = self.schemas.get(event_type, {})
        if not versions:
            return None
        return versions[max(versions.keys())]
    
    def validate(self, event: dict) -> bool:
        """Validate an event against its schema."""
        event_type = event.get("event_type")
        version = event.get("version", 1)
        
        schema = self.get_schema(event_type, version)
        if not schema:
            return False
        
        # Check required fields
        for field in schema.required_fields:
            if field not in event.get("data", {}):
                return False
        
        return True
    
    def migrate(self, event: dict, to_version: int) -> dict:
        """Migrate an event to a newer schema version."""
        event_type = event.get("event_type")
        from_version = event.get("version", 1)
        
        if from_version >= to_version:
            return event
        
        # Apply migrations step by step
        migrated = event.copy()
        for v in range(from_version, to_version):
            migration = self._get_migration(event_type, v, v + 1)
            if migration:
                migrated["data"] = migration(migrated["data"])
        
        migrated["version"] = to_version
        return migrated
    
    def _get_migration(self, event_type: str, from_version: int, to_version: int):
        """Get migration function between versions."""
        # Define migrations as needed
        migrations = {
            ("document.classified", 1, 2): lambda data: {
                **data,
                "confidence": data.get("confidence", 0.0),
                "classifier_version": "v2"
            }
        }
        return migrations.get((event_type, from_version, to_version))

# Example: Schema Evolution
registry = SchemaRegistry()

# Version 1
registry.register(EventSchema(
    event_type="document.classified",
    version=1,
    fields={"document_id": "str", "document_type": "str"},
    required_fields=["document_id", "document_type"],
    optional_fields=[],
    description="Document classification event v1"
))

# Version 2 (added confidence field)
registry.register(EventSchema(
    event_type="document.classified",
    version=2,
    fields={"document_id": "str", "document_type": "str", "confidence": "float"},
    required_fields=["document_id", "document_type"],
    optional_fields=["confidence"],
    description="Document classification event v2 with confidence"
))
```

### Schema Evolution Rules

1. **Never remove fields** — Only add new optional fields
2. **Make new fields optional** — Don't break existing consumers
3. **Use version numbers** — Increment when schema changes
4. **Document changes** — Keep a changelog for each event type
5. **Backward compatible** — New consumers can handle old events
6. **Forward compatible** — Old consumers can handle new events (ignoring unknown fields)

---

## 5. Multi-Agent Event Coordination

### Coordination Patterns

#### Pattern 1: Scatter-Gather

One agent fans out work to multiple agents and collects results:

```python
class ScatterGatherCoordinator:
    def __init__(self, event_bus):
        self.event_bus = event_bus
        self.pending_results = {}
    
    async def scatter(self, task_event: Event, agent_ids: List[str]):
        """Send task to multiple agents."""
        task_id = task_event.data["task_id"]
        self.pending_results[task_id] = {
            "expected": len(agent_ids),
            "received": [],
            "timeout": 30.0
        }
        
        for agent_id in agent_ids:
            await self.event_bus.emit(Event(
                event_type="task.assigned",
                source="coordinator",
                data={"task_id": task_id, "agent_id": agent_id},
                metadata={"correlation_id": task_event.event_id}
            ))
    
    @event_handler("task.completed")
    async def gather(self, event: Event):
        """Collect results from agents."""
        task_id = event.data["task_id"]
        
        if task_id not in self.pending_results:
            return
        
        result = self.pending_results[task_id]
        result["received"].append(event.data["result"])
        
        if len(result["received"]) >= result["expected"]:
            # All results received, aggregate
            aggregated = self.aggregate(result["received"])
            await self.event_bus.emit(Event(
                event_type="task.aggregated",
                source="coordinator",
                data={"task_id": task_id, "aggregated_result": aggregated}
            ))
            del self.pending_results[task_id]
    
    def aggregate(self, results: List[dict]) -> dict:
        """Aggregate results from multiple agents."""
        # Custom aggregation logic
        return {"results": results, "count": len(results)}
```

#### Pattern 2: Competing Consumers

Multiple agents compete to handle the same event:

```python
class CompetingConsumerAgent:
    """Agent that competes for events in a consumer group."""
    
    def __init__(self, agent_id: str, consumer_group: str, event_bus):
        self.agent_id = agent_id
        self.consumer_group = consumer_group
        self.event_bus = event_bus
    
    async def process_event(self, event: Event):
        """Process event and claim ownership."""
        # Only one agent in the group should process each event
        claimed = await self.event_bus.claim_event(
            event.event_id, 
            self.consumer_group, 
            self.agent_id
        )
        
        if claimed:
            result = await self.handle(event)
            await self.event_bus.emit(Event(
                event_type=f"{event.type}.processed",
                source=self.agent_id,
                data={"original_event_id": event.event_id, "result": result}
            ))
```

#### Pattern 3: Event-Based Consensus

Multiple agents reach consensus through event exchange:

```python
class ConsensusAgent:
    """Agent that participates in consensus protocol."""
    
    def __init__(self, agent_id: str, quorum: int, event_bus):
        self.agent_id = agent_id
        self.quorum = quorum
        self.event_bus = event_bus
        self.proposals = {}
        self.votes = {}
    
    async def propose(self, proposal: dict):
        """Submit a proposal for consensus."""
        proposal_id = str(uuid.uuid4())
        self.proposals[proposal_id] = proposal
        
        await self.event_bus.emit(Event(
            event_type="consensus.proposed",
            source=self.agent_id,
            data={"proposal_id": proposal_id, "proposal": proposal}
        ))
    
    @event_handler("consensus.proposed")
    async def vote(self, event: Event):
        """Vote on a proposal."""
        proposal_id = event.data["proposal_id"]
        
        # Agent decides to vote yes or no
        vote = await self.evaluate_proposal(event.data["proposal"])
        
        if proposal_id not in self.votes:
            self.votes[proposal_id] = {"yes": 0, "no": 0, "voters": []}
        
        self.votes[proposal_id][vote] += 1
        self.votes[proposal_id]["voters"].append(self.agent_id)
        
        await self.event_bus.emit(Event(
            event_type="consensus.voted",
            source=self.agent_id,
            data={"proposal_id": proposal_id, "vote": vote}
        ))
        
        # Check if quorum reached
        if self.votes[proposal_id]["yes"] >= self.quorum:
            await self.event_bus.emit(Event(
                event_type="consensus.reached",
                source=self.agent_id,
                data={"proposal_id": proposal_id, "result": "accepted"}
            ))
```

---

## 6. Saga Pattern for Agent Workflows

### What is a Saga?

A saga is a sequence of local transactions where each step emits an event that triggers the next step. If a step fails, compensating events are emitted to undo previous steps.

### Implementation

```python
from enum import Enum
from typing import List, Callable, Optional
from dataclasses import dataclass, field
import uuid

class SagaStepStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATED = "compensated"

@dataclass
class SagaStep:
    name: str
    agent_id: str
    action: str
    compensation: str
    status: SagaStepStatus = SagaStepStatus.PENDING

class AgentSaga:
    """Saga orchestrator for agent workflows."""
    
    def __init__(self, saga_id: str, steps: List[SagaStep]):
        self.saga_id = saga_id
        self.steps = steps
        self.current_step = 0
        self.context = {}
        self.event_log = []
    
    async def execute(self, event_bus):
        """Execute the saga steps."""
        while self.current_step < len(self.steps):
            step = self.steps[self.current_step]
            
            # Emit event to trigger step
            await event_bus.emit(Event(
                event_type=f"saga.{step.action}",
                source=f"saga_{self.saga_id}",
                data={
                    "saga_id": self.saga_id,
                    "step": step.name,
                    "context": self.context
                }
            ))
            
            # Wait for completion or failure
            result = await self.wait_for_step_result(step.name)
            
            if result["status"] == "completed":
                self.context.update(result["data"])
                step.status = SagaStepStatus.COMPLETED
                self.current_step += 1
            elif result["status"] == "failed":
                step.status = SagaStepStatus.FAILED
                await self.compensate(event_bus)
                return {"status": "compensated", "failed_step": step.name}
        
        return {"status": "completed", "context": self.context}
    
    async def compensate(self, event_bus):
        """Execute compensating actions in reverse order."""
        completed_steps = [
            s for s in self.steps 
            if s.status == SagaStepStatus.COMPLETED
        ]
        
        for step in reversed(completed_steps):
            await event_bus.emit(Event(
                event_type=f"saga.{step.compensation}",
                source=f"saga_{self.saga_id}",
                data={
                    "saga_id": self.saga_id,
                    "step": step.name,
                    "context": self.context
                }
            ))
            
            step.status = SagaStepStatus.COMPENSATED
    
    async def wait_for_step_result(self, step_name: str) -> dict:
        """Wait for step completion event (simplified)."""
        # In production, use event store or message broker
        # This is a simplified version
        pass

# Example: Order Processing Saga
saga = AgentSaga(
    saga_id="order_123",
    steps=[
        SagaStep(
            name="validate_order",
            agent_id="validation_agent",
            action="order.validate",
            compensation="order.cancel_validation"
        ),
        SagaStep(
            name="process_payment",
            agent_id="payment_agent",
            action="payment.process",
            compensation="payment.refund"
        ),
        SagaStep(
            name="reserve_inventory",
            agent_id="inventory_agent",
            action="inventory.reserve",
            compensation="inventory.release"
        ),
        SagaStep(
            name="ship_order",
            agent_id="shipping_agent",
            action="order.ship",
            compensation="order.cancel_shipment"
        ),
    ]
)
```

---

## 7. Event-Driven State Machines

### Agent as State Machine

Agents can be modeled as state machines where events trigger transitions:

```python
from enum import Enum
from typing import Dict, Tuple, Callable

class AgentState(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING = "waiting"
    ERROR = "error"
    COMPLETED = "completed"

class EventDrivenStateMachine:
    """State machine for agents driven by events."""
    
    def __init__(self):
        self.state = AgentState.IDLE
        self.transitions: Dict[Tuple[AgentState, str], Tuple[AgentState, Callable]] = {}
    
    def add_transition(self, from_state: AgentState, event_type: str, 
                       to_state: AgentState, action: Callable = None):
        """Add a state transition."""
        self.transitions[(from_state, event_type)] = (to_state, action)
    
    async def handle_event(self, event: Event) -> bool:
        """Handle an event and potentially transition state."""
        transition = self.transitions.get((self.state, event.event_type))
        
        if transition is None:
            return False  # No valid transition
        
        new_state, action = transition
        
        if action:
            await action(event)
        
        old_state = self.state
        self.state = new_state
        
        return True

# Usage: Agent State Machine
sm = EventDrivenStateMachine()

# Define transitions
sm.add_transition(AgentState.IDLE, "task.received", AgentState.PROCESSING)
sm.add_transition(AgentState.PROCESSING, "task.completed", AgentState.COMPLETED)
sm.add_transition(AgentState.PROCESSING, "task.failed", AgentState.ERROR)
sm.add_transition(AgentState.ERROR, "task.retry", AgentState.PROCESSING)
sm.add_transition(AgentState.ERROR, "task.cancel", AgentState.IDLE)
sm.add_transition(AgentState.COMPLETED, "new.task", AgentState.IDLE)

# Handle events
await sm.handle_event(Event(event_type="task.received", source="user", data={}))
# State: IDLE -> PROCESSING

await sm.handle_event(Event(event_type="task.completed", source="agent", data={}))
# State: PROCESSING -> COMPLETED
```

---

## 8. Dead Letter Queues and Error Handling

### Dead Letter Queue Pattern

```python
import asyncio
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class DeadLetterEvent:
    original_event: Event
    error: str
    attempts: int
    first_failure: str
    last_failure: str
    agent_id: str

class DeadLetterQueue:
    """Dead letter queue for failed events."""
    
    def __init__(self, max_size: int = 10000):
        self.events: List[DeadLetterEvent] = []
        self.max_size = max_size
        self.alert_callback = None
    
    async def add(self, event: Event, error: str, agent_id: str, attempts: int):
        """Add a failed event to the dead letter queue."""
        dlq_event = DeadLetterEvent(
            original_event=event,
            error=error,
            attempts=attempts,
            first_failure=datetime.utcnow().isoformat(),
            last_failure=datetime.utcnow().isoformat(),
            agent_id=agent_id
        )
        
        self.events.append(dlq_event)
        
        # Trim if over max size
        if len(self.events) > self.max_size:
            self.events = self.events[-self.max_size:]
        
        # Alert if configured
        if self.alert_callback:
            await self.alert_callback(dlq_event)
    
    async def retry(self, event_bus, count: int = 1):
        """Retry events from the dead letter queue."""
        for _ in range(min(count, len(self.events))):
            dlq_event = self.events.pop(0)
            await event_bus.emit(dlq_event.original_event)

class ResilientEventProcessor:
    """Event processor with dead letter queue support."""
    
    def __init__(self, agent_id: str, event_bus, dlq: DeadLetterQueue, 
                 max_retries: int = 3):
        self.agent_id = agent_id
        self.event_bus = event_bus
        self.dlq = dlq
        self.max_retries = max_retries
    
    async def process_with_retry(self, event: Event, handler: Callable):
        """Process event with retry and dead letter queue."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                await handler(event)
                return  # Success
            except Exception as e:
                last_error = str(e)
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        # All retries failed, send to DLQ
        await self.dlq.add(
            event=event,
            error=last_error,
            agent_id=self.agent_id,
            attempts=self.max_retries
        )
```

---

## 9. Event Replay and Time Travel

### Replaying Events

```python
class EventReplayer:
    """Replay events to rebuild state or debug issues."""
    
    def __init__(self, event_store: AgentEventStore):
        self.event_store = event_store
    
    async def replay_to_point(self, timestamp: str) -> dict:
        """Replay events up to a specific timestamp."""
        events = self.event_store.get_events()
        state = {}
        
        for event in events:
            if event.timestamp <= timestamp:
                state = self.apply_event(state, event)
        
        return state
    
    async def replay_from_event(self, event_id: str) -> dict:
        """Replay events starting from a specific event."""
        events = self.event_store.get_events()
        start_index = next(
            (i for i, e in enumerate(events) if e.event_id == event_id),
            0
        )
        
        state = {}
        for event in events[start_index:]:
            state = self.apply_event(state, event)
        
        return state
    
    async def diff_states(self, timestamp1: str, timestamp2: str) -> dict:
        """Compare state at two different points in time."""
        state1 = await self.replay_to_point(timestamp1)
        state2 = await self.replay_to_point(timestamp2)
        
        return {
            "state_at_t1": state1,
            "state_at_t2": state2,
            "diff": self._compute_diff(state1, state2)
        }
    
    def apply_event(self, state: dict, event: AgentEvent) -> dict:
        """Apply an event to state (simplified)."""
        # In practice, use registered handlers
        if event.event_type == "research.started":
            return {**state, "topic": event.data.get("topic")}
        elif event.event_type == "source.found":
            return {**state, "sources": state.get("sources", []) + [event.data]}
        return state
    
    def _compute_diff(self, state1: dict, state2: dict) -> dict:
        """Compute diff between two states."""
        diff = {}
        all_keys = set(state1.keys()) | set(state2.keys())
        
        for key in all_keys:
            if key not in state1:
                diff[key] = {"added": state2[key]}
            elif key not in state2:
                diff[key] = {"removed": state1[key]}
            elif state1[key] != state2[key]:
                diff[key] = {"changed_from": state1[key], "changed_to": state2[key]}
        
        return diff
```

---

## 10. Security and Access Control

### Event Security Patterns

```python
from typing import Set
from enum import Enum

class Permission(Enum):
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"

class EventSecurityPolicy:
    """Security policy for event-driven agent systems."""
    
    def __init__(self):
        self.agent_permissions: Dict[str, Set[Permission]] = {}
        self.event_type_permissions: Dict[str, Set[Permission]] = {}
    
    def grant_permission(self, agent_id: str, permission: Permission):
        """Grant a permission to an agent."""
        if agent_id not in self.agent_permissions:
            self.agent_permissions[agent_id] = set()
        self.agent_permissions[agent_id].add(permission)
    
    def require_permission(self, event_type: str, permission: Permission):
        """Require a permission to emit/consume an event type."""
        if event_type not in self.event_type_permissions:
            self.event_type_permissions[event_type] = set()
        self.event_type_permissions[event_type].add(permission)
    
    def can_emit(self, agent_id: str, event_type: str) -> bool:
        """Check if an agent can emit an event type."""
        required = self.event_type_permissions.get(event_type, set())
        agent_perms = self.agent_permissions.get(agent_id, set())
        return required.issubset(agent_perms)
    
    def can_consume(self, agent_id: str, event_type: str) -> bool:
        """Check if an agent can consume an event type."""
        # Similar logic to can_emit
        return self.can_emit(agent_id, event_type)

class SecureEventBus:
    """Event bus with access control."""
    
    def __init__(self, policy: EventSecurityPolicy):
        self.policy = policy
        self.subscribers = defaultdict(list)
    
    async def emit(self, agent_id: str, event: Event):
        """Emit event with security check."""
        if not self.policy.can_emit(agent_id, event.event_type):
            raise PermissionError(f"Agent {agent_id} cannot emit {event.event_type}")
        
        # Add security metadata
        event.metadata["emitted_by"] = agent_id
        event.metadata["emitted_at"] = datetime.utcnow().isoformat()
        
        # Distribute to authorized subscribers
        for subscriber_id, handler in self.subscribers.get(event.event_type, []):
            if self.policy.can_consume(subscriber_id, event.event_type):
                await handler(event)
```

---

## 11. Performance Optimization

### Batching Events

```python
class BatchedEventProcessor:
    """Process events in batches for better throughput."""
    
    def __init__(self, batch_size: int = 100, flush_interval: float = 1.0):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.batch = []
        self.last_flush = time.time()
    
    async def add_event(self, event: Event):
        """Add event to batch."""
        self.batch.append(event)
        
        if (len(self.batch) >= self.batch_size or 
            time.time() - self.last_flush >= self.flush_interval):
            await self.flush()
    
    async def flush(self):
        """Process current batch."""
        if not self.batch:
            return
        
        batch = self.batch.copy()
        self.batch.clear()
        self.last_flush = time.time()
        
        # Process batch
        await self.process_batch(batch)
    
    async def process_batch(self, batch: List[Event]):
        """Override to implement batch processing."""
        pass
```

### Event Compression

```python
class CompressedEventStore:
    """Event store with compression for storage efficiency."""
    
    def __init__(self):
        self.events = []
        self.compression_threshold = 1000  # Compress after N events
    
    async def append(self, event: AgentEvent):
        """Append event, compressing older events."""
        self.events.append(event)
        
        if len(self.events) > self.compression_threshold:
            await self.compress_old_events()
    
    async def compress_old_events(self):
        """Compress events older than threshold."""
        # In practice, use LZ4, Snappy, or Zstandard
        # This is a simplified version
        pass
```

---

## 12. Testing Event-Driven Agent Systems

### Unit Testing Event Handlers

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

class TestResearchAgent:
    """Test research agent event handling."""
    
    @pytest.fixture
    def agent(self):
        return ResearchAgent()
    
    @pytest.fixture
    def event_bus(self):
        return AsyncMock()
    
    async def test_research_started(self, agent, event_bus):
        """Test research started event handling."""
        event = AgentEvent(
            event_id="test_1",
            event_type="research.started",
            agent_id="test_agent",
            timestamp="2026-07-06T10:00:00Z",
            data={"topic": "test topic"}
        )
        
        agent.process_event(event)
        
        assert agent.state["topic"] == "test topic"
        assert agent.state["status"] == "in_progress"
    
    async def test_source_analyzed(self, agent, event_bus):
        """Test source analyzed event handling."""
        # Start research first
        agent.process_event(AgentEvent(
            event_id="test_1",
            event_type="research.started",
            agent_id="test_agent",
            timestamp="2026-07-06T10:00:00Z",
            data={"topic": "test topic"}
        ))
        
        # Analyze source
        agent.process_event(AgentEvent(
            event_id="test_2",
            event_type="source.analyzed",
            agent_id="test_agent",
            timestamp="2026-07-06T10:01:00Z",
            data={"finding": {"content": "test finding", "confidence": 0.9}}
        ))
        
        assert len(agent.state["findings"]) == 1
        assert agent.state["sources_analyzed"] == 1
```

### Integration Testing

```python
class TestEventDrivenWorkflow:
    """Integration test for event-driven agent workflow."""
    
    async def test_document_processing_pipeline(self):
        """Test full document processing pipeline."""
        event_bus = EventBus()
        
        # Set up agents
        ocr_agent = OCRAgent(event_bus)
        classification_agent = ClassificationAgent(event_bus)
        processing_agent = ProcessingAgent(event_bus)
        
        # Track events
        events_received = []
        
        @event_handler("document.processed")
        async def track_completion(event):
            events_received.append(event)
        
        event_bus.subscribe("document.processed", track_completion)
        
        # Start pipeline
        await event_bus.emit(Event(
            event_type="document.uploaded",
            source="test",
            data={"document_id": "test_doc", "content": "test content"}
        ))
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Verify
        assert len(events_received) == 1
        assert events_received[0].data["document_id"] == "test_doc"
```

---

## Summary

Event-driven agent architectures provide:
- **Loose coupling** between agents
- **Scalability** through horizontal scaling
- **Resilience** via independent failure domains
- **Auditability** through event sourcing
- **Flexibility** in workflow composition

The key patterns are:
1. Event sourcing for state management
2. CQRS for read/write separation
3. Reactive streams with backpressure
4. Sagas for long-running workflows
5. Dead letter queues for error handling

---

*This document is part of the AI Knowledge Library — a comprehensive reference for AI practitioners and researchers.*
