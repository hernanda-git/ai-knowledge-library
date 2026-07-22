# AI Event-Driven Agent Architectures

> A comprehensive guide to designing, building, and operating AI agents on event-driven architectures — covering event sourcing, reactive patterns, streaming pipelines, and the convergence of event-driven design with autonomous agent systems.

**Last Updated:** 2026-07-06  
**Estimated Reading Time:** 60 minutes  
**Category:** 57 — AI Event-Driven Agent Architectures  
**Cross-References:** [03-Agents](../03-Agents/), [32-Agent-Memory-Systems](../32-Agent-Memory-Systems/), [54-Agent-State-Management](../54-Agent-State-Management-and-Persistence/), [44-Agentic-Platforms](../44-Agentic-Platforms-and-Enterprise-Collaboration/), [31-Workflow-Orchestration](../31-AI-Workflow-Orchestration-and-Durable-Execution/)

---

## Table of Contents

1. [What Is Event-Driven Agent Architecture?](#1-what-is-event-driven-agent-architecture)
2. [Why Event-Driven for AI Agents?](#2-why-event-driven-for-ai-agents)
3. [Core Concepts and Terminology](#3-core-concepts-and-terminology)
4. [Architecture Patterns](#4-architecture-patterns)
5. [Event-Driven vs Request-Response for Agents](#5-event-driven-vs-request-response-for-agents)
6. [The Event-Driven Agent Lifecycle](#6-the-event-driven-agent-lifecycle)
7. [Key Design Principles](#7-key-design-principles)
8. [Industry Landscape 2026](#8-industry-landscape-2026)
9. [Real-World Use Cases](#9-real-world-use-cases)
10. [Getting Started Guide](#10-getting-started-guide)
11. [Common Anti-Patterns](#11-common-anti-patterns)
12. [Cross-References](#12-cross-references)

---

## 1. What Is Event-Driven Agent Architecture?

Event-driven agent architecture (EDAA) is a design paradigm where autonomous AI agents communicate, coordinate, and react to changes in their environment through **events** — immutable records of something that happened — rather than through direct synchronous calls or polling.

### Definition

> An **event-driven agent architecture** is a system design where AI agents are triggered, coordinated, and informed by events propagated through a message broker or event stream, enabling loosely-coupled, scalable, and responsive autonomous systems.

### The Core Idea

In a traditional request-response model:

```
User → Agent A → Agent B → Agent C → Response
```

In an event-driven model:

```
Event Bus ← Agent A (emits "data_ready")
           ← Agent B (listens for "data_ready", processes, emits "analysis_complete")
           ← Agent C (listens for "analysis_complete", generates report)
```

The agents don't know about each other. They only know about **events**.

### Key Characteristics

| Characteristic | Description |
|---------------|-------------|
| **Loose Coupling** | Agents don't directly reference each other |
| **Asynchronous** | Agents react to events when they arrive, not on demand |
| **Scalable** | New agents can subscribe to existing events without modification |
| **Resilient** | If one agent fails, others continue processing |
| **Observable** | Every event is a record of something that happened |
| **Composable** | Complex workflows emerge from simple event chains |

### Historical Context

Event-driven architecture predates AI agents by decades:
- **1990s:** Enterprise Application Integration (EAI) with message queues
- **2000s:** Event-Driven Architecture (EDA) formalized by Gartner
- **2010s:** Apache Kafka popularized event streaming at scale
- **2020s:** Serverless functions + event triggers (AWS Lambda, Cloud Functions)
- **2025-2026:** AI agents adopt event-driven patterns for coordination

The convergence in 2026 is driven by:
1. Agent autonomy requiring reactive architectures
2. Multi-agent coordination needing decoupled communication
3. Production reliability requirements demanding event sourcing
4. Cost optimization requiring efficient resource utilization

---

## 2. Why Event-Driven for AI Agents?

### The Problem with Synchronous Agent Systems

Most early AI agent implementations use synchronous request-response patterns:

```
Orchestrator → calls Agent A → waits for response → calls Agent B → waits → ...
```

**This breaks down because:**

1. **Latency accumulation:** 5 agents × 2s each = 10s minimum
2. **Fragility:** One slow or failed agent blocks the entire chain
3. **Resource waste:** Threads blocked waiting for LLM inference
4. **Scaling ceiling:** Can't parallelize work effectively
5. **No replay:** Can't re-process events after the fact

### The Event-Driven Advantage

| Aspect | Synchronous | Event-Driven |
|--------|------------|--------------|
| Latency | Sequential (N × T) | Parallel (max(T)) |
| Fault tolerance | Single point of failure | Independent failure domains |
| Scalability | Limited by orchestrator | Horizontal scaling per agent |
| Resource usage | Threads blocked idle | Efficient async processing |
| Auditability | Requires separate logging | Events ARE the audit trail |
| Extensibility | Modify orchestrator | Add new subscriber |
| Cost | Idle waiting costs money | Pay per event processed |

### Real-World Signal (July 2026)

Research from HN and industry shows massive adoption:
- **Calfkit** — SDK for building distributed event-driven AI agents on Kafka
- **Arvo** — TypeScript toolkit for event-driven agentic systems and mesh
- **Loom** — Event-driven OS for AI agents (built in 10 days by a college junior)
- **Stream0** — HTTP-native messaging layer for AI agents
- **DocuFlow** — Open-source event-driven AI invoice ingestion pipeline
- **SensorHub** — Event-driven version of Clawhub for giving AI agents "ears"
- Multiple blog posts on "Event-Driven Patterns for AI Agents" trending on HN

This is not theoretical — it's the pattern that production teams are converging on.

---

## 3. Core Concepts and Terminology

### Events

An **event** is an immutable record of something that happened in the system.

```json
{
  "event_id": "evt_20260706_a1b2c3",
  "event_type": "document.ingested",
  "timestamp": "2026-07-06T14:30:00Z",
  "source": "ingestion-agent",
  "version": "1.0",
  "data": {
    "document_id": "doc_xyz",
    "document_type": "invoice",
    "page_count": 3,
    "language": "en"
  },
  "metadata": {
    "correlation_id": "corr_123",
    "causation_id": "evt_20260706_a1b2c2"
  }
}
```

### Event Types

| Type | Description | Example |
|------|-------------|---------|
| **Domain Event** | Business-significant occurrence | `order.placed`, `document.verified` |
| **Integration Event** | Cross-boundary communication | `payment.processed`, `email.sent` |
| **Command Event** | Request for action | `agent.analyze.request`, `agent.summarize.request` |
| **Notification Event** | Information broadcast | `system.health.ok`, `model.updated` |
| **Timer Event** | Scheduled trigger | `daily.summary.due`, `cache.expired` |

### Event Producers

Entities that emit events:
- **Agents** — emit events when they complete tasks, make decisions, or encounter errors
- **External Systems** — webhooks, APIs, IoT sensors
- **Schedulers** — cron jobs, timers, SLA monitors
- **Users** — explicit actions, approvals, rejections

### Event Consumers (Subscribers)

Entities that react to events:
- **Agents** — subscribe to events relevant to their capabilities
- **Workflows** — step functions triggered by events
- **Monitors** — track events for observability
- **Archives** — store events for replay and audit

### Event Channels

The transport layer for events:
- **Message Queues** (RabbitMQ, SQS) — point-to-point delivery
- **Event Streams** (Kafka, Kinesis, Pulsar) — pub/sub with replay
- **Event Buses** (EventBridge, SNS) — content-based routing
- **Webhooks** — HTTP callbacks to external systems
- **In-Process Buses** (MediatR, custom) — within a single application

### Event Sourcing

Storing the **sequence of events** rather than just the current state:

```
Event Store:
  1. document.ingested(doc_123)
  2. document.classified(doc_123, type=invoice)
  3. document.extracted(doc_123, fields={...})
  4. document.validated(doc_123, status=pass)
  5. document.approved(doc_123, approver=user_456)
```

From this event log, you can reconstruct the state at any point in time, replay events to rebuild state, or audit every decision made.

### CQRS (Command Query Responsibility Segregation)

Separating the write model (commands → events) from the read model (events → queries):

```
Commands → [Command Handler] → Event Store → [Event Handlers] → Read Model
                                                              → Agent Triggers
                                                              → Audit Logs
```

---

## 4. Architecture Patterns

### Pattern 1: Event-Driven Agent Chain

Agents form a chain where each agent's output events trigger the next:

```
User Request
  ↓
[Orchestrator Agent] emits: request.received
  ↓
[Research Agent] listens → processes → emits: research.complete
  ↓
[Analysis Agent] listens → processes → emits: analysis.complete
  ↓
[Response Agent] listens → generates response → emits: response.sent
```

**Best for:** Linear workflows, document processing, multi-step analysis.

### Pattern 2: Event-Driven Fan-Out/Fan-In

One event triggers multiple agents in parallel, and results are aggregated:

```
Task Received
  ↓
[Event Router] emits: task分解
  ↓
  ├── [Agent A] → emits: subtask.a.complete
  ├── [Agent B] → emits: subtask.b.complete
  └── [Agent C] → emits: subtask.c.complete
  ↓
[Aggregator] listens for ALL three → emits: task.complete
```

**Best for:** Parallel processing, multi-modal analysis, distributed computation.

### Pattern 3: Event-Driven Saga

Long-running workflows with compensating actions for failure handling:

```
Order Placed
  ↓
[Payment Agent] → payment.processed OR payment.failed
  ↓ (if failed)
[Compensation Agent] → order.cancelled
  ↓ (if processed)
[Inventory Agent] → inventory.reserved OR inventory.failed
  ↓ (if failed)
[Compensation Agent] → payment.refunded, order.cancelled
  ↓ (if reserved)
[Fulfillment Agent] → order.shipped
```

**Best for:** Multi-step business processes, transactions spanning multiple agents.

### Pattern 4: Event-Driven Mesh

Agents form a mesh where any agent can emit events and any agent can subscribe:

```
         [Agent A]
        ↗ ↕     ↘
[Agent B] ←→ [Event Mesh] ←→ [Agent C]
        ↘ ↕     ↗
         [Agent D]
```

Every agent is both a producer and consumer. The mesh handles routing.

**Best for:** Complex multi-agent systems, emergent behavior, adaptive workflows.

### Pattern 5: Event Sourcing with Snapshot

For long-lived agent processes, combine event sourcing with periodic snapshots:

```
Events: e1, e2, e3, [Snapshot at e3], e4, e5, e6, [Snapshot at e6], ...
```

To restore state: load latest snapshot + replay subsequent events.

**Best for:** Long-running agent tasks, systems requiring audit trails, debugging.

### Pattern 6: Reactive Streams

Agents process events as a continuous stream with backpressure:

```
Event Stream → [Agent Processing Pipeline]
                 ↓
              [Backpressure Signal] → slow down producer
```

**Best for:** High-throughput scenarios, real-time processing, IoT + AI.

---

## 5. Event-Driven vs Request-Response for Agents

### Decision Matrix

| Criterion | Request-Response | Event-Driven | Winner |
|-----------|-----------------|--------------|--------|
| Simple task orchestration | ✅ Direct, easy | ⚠️ Overhead | Request-Response |
| Multi-agent coordination | ❌ Tight coupling | ✅ Loose coupling | Event-Driven |
| Real-time responsiveness | ⚠️ Depends on chain | ✅ Immediate reaction | Event-Driven |
| Fault tolerance | ❌ Cascading failures | ✅ Independent | Event-Driven |
| Audit and compliance | ❌ Extra work needed | ✅ Built-in | Event-Driven |
| Cost efficiency at scale | ❌ Idle resources | ✅ Pay per event | Event-Driven |
| Debugging complexity | ✅ Linear trace | ⚠️ Distributed trace | Request-Response |
| Learning curve | ✅ Familiar | ⚠️ Steeper | Request-Response |
| Scalability ceiling | ❌ Single orchestrator | ✅ Horizontal | Event-Driven |

### When to Use Each

**Use Request-Response when:**
- Simple agent chains (2-3 steps)
- Low-latency requirements with known topology
- Team is new to event-driven patterns
- Agent interactions are stateless and short-lived

**Use Event-Driven when:**
- Multiple agents need to coordinate
- Workflows are long-running or complex
- Fault isolation is critical
- You need audit trails for compliance
- You're scaling beyond a single machine
- Agents need to react to environmental changes

**Use Hybrid when:**
- You have both synchronous and asynchronous needs
- Some agent interactions are real-time, others are background
- You're migrating from request-response to event-driven

### The Hybrid Approach (Most Common in Production)

```python
# Synchronous for user-facing responses
@app.post("/chat")
async def chat(request: ChatRequest):
    # Synchronous: get immediate response from primary agent
    response = await primary_agent.process(request.message)
    
    # Asynchronous: emit event for background processing
    await event_bus.emit("conversation.completed", {
        "conversation_id": request.conversation_id,
        "message": request.message,
        "response": response.text
    })
    
    return response

# Asynchronous for background work
@event_handler("conversation.completed")
async def handle_conversation(event):
    # Background agents process the event
    await analytics_agent.analyze(event.data)
    await memory_agent.store(event.data)
    await quality_agent.evaluate(event.data)
```

---

## 6. The Event-Driven Agent Lifecycle

### Phase 1: Event Detection

Agents detect events from various sources:

```python
class EventDetection:
    """Agents detect events from multiple sources."""
    
    sources = {
        "user_input": UserInputDetector(),
        "system_state": SystemStateDetector(),
        "time_based": ScheduleDetector(),
        "webhook": WebhookDetector(),
        "stream": StreamDetector(),
        "file": FileWatcherDetector()
    }
    
    async def detect(self) -> Event:
        for name, detector in self.sources.items():
            event = await detector.check()
            if event:
                return event
```

### Phase 2: Event Classification

Events are classified to determine routing:

```python
class EventClassifier:
    """Classifies events for routing to appropriate agents."""
    
    classification_rules = {
        "urgent": lambda e: e.priority == "high" or e.data.get("deadline"),
        "batch": lambda e: e.type.startswith("batch."),
        "user_facing": lambda e: e.source == "user_input",
        "system": lambda e: e.source == "system_state",
    }
    
    def classify(self, event: Event) -> EventClassification:
        labels = []
        for label, rule in self.classification_rules.items():
            if rule(event):
                labels.append(label)
        return EventClassification(labels=labels, event=event)
```

### Phase 3: Event Routing

Events are routed to subscribers:

```python
class EventRouter:
    """Routes events to registered agent subscribers."""
    
    def __init__(self):
        self.subscriptions = defaultdict(list)
    
    def subscribe(self, agent_id: str, event_types: List[str]):
        for event_type in event_types:
            self.subscriptions[event_type].append(agent_id)
    
    def route(self, event: Event) -> List[str]:
        subscribers = self.subscriptions.get(event.type, [])
        # Also check wildcard subscriptions
        subscribers += self.subscriptions.get("*", [])
        return subscribers
```

### Phase 4: Agent Processing

Agents process events and potentially emit new events:

```python
class EventProcessingAgent:
    """Base class for event-driven agents."""
    
    async def handle_event(self, event: Event) -> Optional[Event]:
        """Process an event and optionally emit a new event."""
        try:
            result = await self.process(event)
            return Event(
                type=f"{self.agent_type}.completed",
                source=self.agent_id,
                data=result,
                metadata={"causation_id": event.event_id}
            )
        except Exception as e:
            return Event(
                type=f"{self.agent_type}.failed",
                source=self.agent_id,
                data={"error": str(e), "original_event": event.event_id}
            )
```

### Phase 5: Event Storage and Replay

Events are stored for audit and replay:

```python
class EventStore:
    """Stores events for audit and replay."""
    
    async def append(self, event: Event):
        await self.db.insert("events", event.to_dict())
        await self.notify_subscribers(event)
    
    async def get_events(self, aggregate_id: str) -> List[Event]:
        return await self.db.query(
            "events", 
            {"aggregate_id": aggregate_id},
            order_by="timestamp"
        )
    
    async def replay(self, aggregate_id: str, from_timestamp: datetime):
        events = await self.get_events(aggregate_id)
        for event in events:
            if event.timestamp >= from_timestamp:
                await self.process_event(event)
```

---

## 7. Key Design Principles

### Principle 1: Events Are Facts

Events represent things that **happened**, not things that **should happen**.

```
✅ "document.classified" — this document WAS classified
❌ "document.should.be.classified" — this is a command, not an event
```

### Principle 2: Events Are Immutable

Once emitted, an event cannot be changed. If you need to correct something, emit a new event:

```
Event 1: order.placed(amount=100)
Event 2: order.corrected(amount=90, reason="discount_applied")
```

### Principle 3: Event-Driven Agents Are Stateless Between Events

Agents should not rely on in-memory state between event processing. State should be:
- Derived from the event stream (event sourcing)
- Stored in external state stores
- Computed on demand from the event log

### Principle 4: Design for Failure

Every agent that processes events must handle:
- **Duplicate events** (at-least-once delivery)
- **Out-of-order events** (temporal ordering)
- **Missing events** (gap detection)
- **Poison pill events** (dead letter queues)

### Principle 5: Observable by Default

Every event should carry enough metadata for debugging:
```json
{
  "event_id": "unique_id",
  "correlation_id": "end_to_end_trace_id",
  "causation_id": "what_caused_this_event",
  "timestamp": "ISO_8601",
  "source": "which_agent_or_system",
  "version": "schema_version"
}
```

### Principle 6: Schema Evolution

Event schemas must evolve without breaking consumers:
- Use versioned event types (`document.classified.v2`)
- Make new fields optional
- Never remove fields, only deprecate
- Use a schema registry

### Principle 7: Backpressure and Rate Limiting

Agents must signal when they're overwhelmed:
- Use consumer groups for load balancing
- Implement circuit breakers
- Set processing rate limits
- Monitor queue depths

---

## 8. Industry Landscape 2026

### Event Infrastructure for AI Agents

| Tool | Description | Status |
|------|-------------|--------|
| **Apache Kafka** | Distributed event streaming platform | Production standard |
| **Apache Pulsar** | Multi-tenant event streaming | Growing adoption |
| **AWS EventBridge** | Serverless event bus | AWS-native |
| **Google Cloud Eventarc** | Event-driven for Cloud Run | GCP-native |
| **Azure Event Grid** | Event routing service | Azure-native |
| **NATS** | Lightweight messaging | Edge/IoT focus |
| **Redpanda** | Kafka-compatible, lower latency | Performance-focused |
| **WarpStream** | Kafka on object storage | Cost-optimized |

### AI Agent-Specific Event Tools

| Tool | Description | Origin |
|------|-------------|--------|
| **Calfkit** | SDK for event-driven agents on Kafka | Open Source |
| **Arvo** | TypeScript toolkit for agentic event mesh | Open Source |
| **Loom** | Event-driven OS for AI agents | Indie Project |
| **Stream0** | HTTP-native messaging for agents | Open Source |
| **DocuFlow** | Event-driven document processing | Open Source |
| **SensorHub** | Event-driven agent sensor layer | Open Source |

### Cloud Provider Agent Event Services

| Service | Provider | Features |
|---------|----------|----------|
| **Amazon Bedrock Agents** | AWS | Event-driven agent orchestration |
| **Azure AI Agent Service** | Azure | Event Grid integration |
| **Google Vertex AI Agent Builder** | GCP | Eventarc event triggers |
| **Confluent Cloud for Apache Kafka** | Confluent | Managed Kafka for agents |

---

## 9. Real-World Use Cases

### Use Case 1: Document Processing Pipeline

**Scenario:** An organization processes thousands of documents daily — invoices, contracts, reports.

```
Document Uploaded
  ↓
[OCR Agent] → document.extracted
  ↓
[Classification Agent] → document.classified (invoice/contract/report)
  ↓ (fan-out based on type)
  ├── [Invoice Agent] → invoice.validated → [Payment Agent] → payment.initiated
  ├── [Contract Agent] → contract.reviewed → [Legal Agent] → contract.approved
  └── [Report Agent] → report.summarized → [Distribution Agent] → report.distributed
```

**Event flow:**
1. User uploads document
2. Storage system emits `document.uploaded`
3. OCR agent subscribes, extracts text, emits `document.extracted`
4. Classification agent determines type, emits `document.classified`
5. Router emits type-specific events
6. Type-specific agents process and emit completion events
7. Aggregator waits for all completions, emits `document.processed`

### Use Case 2: Customer Support Agent System

**Scenario:** Multi-agent customer support with escalation.

```
Customer Message Received
  ↓
[Intent Detection Agent] → message.classified(intent=billing)
  ↓
[Billing Agent] → billing.query.resolved OR billing.escalation_needed
  ↓ (if escalation)
[Human-in-the-Loop Agent] → escalation.created → [Notification Agent] → agent.notified
```

### Use Case 3: Real-Time Monitoring and Alerting

**Scenario:** Monitor AI system health and trigger remediation.

```
Metrics Stream
  ↓
[Anomaly Detection Agent] → anomaly.detected(type=latency_spike)
  ↓
[Diagnosis Agent] → diagnosis.complete(root_cause=memory_leak)
  ↓
[Remediation Agent] → remediation.initiated(action=scale_up)
  ↓
[Validation Agent] → remediation.validated(success=true)
```

### Use Case 4: Multi-Modal Content Processing

**Scenario:** Process content that arrives in multiple modalities.

```
Content Received (text + image + audio)
  ↓
[Event Router] → content.received
  ↓ (fan-out)
  ├── [Text Agent] → text.analyzed
  ├── [Image Agent] → image.analyzed
  └── [Audio Agent] → audio.analyzed
  ↓
[Synthesis Agent] → content.synthesized(merged_analysis)
```

---

## 10. Getting Started Guide

### Step 1: Choose Your Event Infrastructure

For most teams starting out:

| Scale | Recommended | Why |
|-------|------------|-----|
| Prototype | In-process event bus (MediatR, custom) | Zero infrastructure |
| Small production | Redis Streams or NATS | Simple, fast |
| Medium production | AWS SQS + SNS or RabbitMQ | Managed, reliable |
| Large production | Apache Kafka or Redpanda | Scalable, replay |

### Step 2: Define Your Event Schema

Start with a minimal schema:

```json
{
  "event_id": "string (UUID)",
  "event_type": "string (domain.action)",
  "timestamp": "ISO 8601",
  "source": "string (agent_id or system)",
  "data": "object (event-specific payload)",
  "metadata": {
    "correlation_id": "string (trace ID)",
    "causation_id": "string (parent event ID)"
  }
}
```

### Step 3: Implement Your First Event-Driven Agent

```python
import asyncio
from dataclasses import dataclass, field
from typing import Callable, Dict, List
import uuid
from datetime import datetime

@dataclass
class Event:
    event_type: str
    source: str
    data: dict
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: dict = field(default_factory=dict)

class EventBus:
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
    
    def subscribe(self, event_type: str, handler: Callable):
        self.subscribers.setdefault(event_type, []).append(handler)
    
    async def emit(self, event: Event):
        handlers = self.subscribers.get(event.event_type, [])
        handlers += self.subscribers.get("*", [])
        for handler in handlers:
            await handler(event)

class SummarizationAgent:
    def __init__(self, event_bus: EventBus):
        self.bus = event_bus
        self.bus.subscribe("document.extracted", self.handle_document)
    
    async def handle_document(self, event: Event):
        # Process with LLM
        summary = await self.summarize(event.data["text"])
        
        # Emit completion event
        await self.bus.emit(Event(
            event_type="document.summarized",
            source="summarization_agent",
            data={"document_id": event.data["document_id"], "summary": summary},
            metadata={"causation_id": event.event_id}
        ))

# Usage
bus = EventBus()
agent = SummarizationAgent(bus)

# Trigger the pipeline
await bus.emit(Event(
    event_type="document.extracted",
    source="ocr_agent",
    data={"document_id": "doc_123", "text": "..."}
))
```

### Step 4: Add Error Handling

```python
class ResilientAgent:
    def __init__(self, event_bus: EventBus, max_retries: int = 3):
        self.bus = event_bus
        self.max_retries = max_retries
    
    async def handle_event(self, event: Event):
        for attempt in range(self.max_retries):
            try:
                result = await self.process(event)
                await self.bus.emit(Event(
                    event_type=f"{self.agent_type}.completed",
                    source=self.agent_id,
                    data=result,
                    metadata={"causation_id": event.event_id}
                ))
                return
            except Exception as e:
                if attempt == self.max_retries - 1:
                    # Emit failure event
                    await self.bus.emit(Event(
                        event_type=f"{self.agent_type}.failed",
                        source=self.agent_id,
                        data={"error": str(e), "attempts": attempt + 1},
                        metadata={"causation_id": event.event_id}
                    ))
                else:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

### Step 5: Add Observability

```python
import logging
from contextvars import ContextVar

correlation_id: ContextVar[str] = ContextVar('correlation_id', default='')

class ObservableEventBus(EventBus):
    async def emit(self, event: Event):
        # Set correlation ID for distributed tracing
        if "correlation_id" not in event.metadata:
            event.metadata["correlation_id"] = event.event_id
        
        logging.info(f"Emitting: {event.event_type} from {event.source}")
        
        handlers = self.subscribers.get(event.event_type, [])
        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logging.error(f"Handler failed: {e}")
                # Emit error event
                await self.emit(Event(
                    event_type="system.handler_error",
                    source="event_bus",
                    data={"handler": handler.__name__, "error": str(e)},
                    metadata={"correlation_id": event.metadata.get("correlation_id")}
                ))
```

---

## 11. Common Anti-Patterns

### Anti-Pattern 1: Event soup

**Problem:** Everything is an event, with no structure or schema.

**Solution:** Define clear event types with versioned schemas. Use a schema registry.

### Anti-Pattern 2: Implicit event contracts

**Problem:** Producers and consumers assume the same event structure without formal contracts.

**Solution:** Use schema registries (Confluent Schema Registry, AWS Glue) and contract testing.

### Anti-Pattern 3: Synchronous event processing

**Problem:** Event handlers make synchronous calls to external services, blocking the event loop.

**Solution:** Use async/await throughout. Offload blocking I/O to thread pools.

### Anti-Pattern 4: No dead letter queue

**Problem:** Failed events are lost or retried infinitely.

**Solution:** Implement dead letter queues with alerts and manual review processes.

### Anti-Pattern 5: Tight coupling through event data

**Problem:** Events contain too much data, making consumers dependent on producer internals.

**Solution:** Events should contain only what consumers need. Use enrichment patterns.

### Anti-Pattern 6: Ignoring event ordering

**Problem:** Events arrive out of order, causing incorrect state.

**Solution:** Include sequence numbers. Use event sourcing with idempotent handlers.

---

## 12. Cross-References

| Related Category | Relevance |
|-----------------|-----------|
| [03-Agents](../03-Agents/) | Core agent concepts and multi-agent systems |
| [32-Agent-Memory-Systems](../32-Agent-Memory-Systems/) | Memory patterns for event-driven agents |
| [54-Agent-State-Management](../54-Agent-State-Management-and-Persistence/) | State persistence in event-driven architectures |
| [31-Workflow-Orchestration](../31-AI-Workflow-Orchestration-and-Durable-Execution/) | Workflow patterns complementary to event-driven |
| [44-Agentic-Platforms](../44-Agentic-Platforms-and-Enterprise-Collaboration/) | Enterprise platforms adopting event-driven |
| [20-Agent-Infrastructure](../20-Agent-Infrastructure-and-Observability/) | Infrastructure for running event-driven agents |
| [56-MLOps](../56-MLOps-and-AI-Platform-Engineering/) | Platform engineering for event-driven AI |
| [04-RAG](../04-RAG/) | RAG pipelines can be event-driven |

---

*This document is part of the AI Knowledge Library — a comprehensive reference for AI practitioners and researchers.*

---
**See also:**
- [07 — AI Coding Agent Sustainability: Managing Hidden Costs, Code Quality Decay, and Developer Burnout](33-AI-Native-Software-Development/07-AI-Coding-Agent-Sustainability.md)
- [48 — MCP Cloud Infrastructure & Agent-as-a-Service: The Production Layer of Agentic AI](48-MCP-Cloud-Infrastructure-Agent-as-a-Service/01-Overview.md)
- [AI Agent Commerce & Agent-to-Agent (A2A) Payments: The 2026 Frontier](28-AI-Agent-Commerce-and-A2A-Payments/01-Overview.md)
