# Technical Deep Dive: Event-Driven Agent Architectures

> Implementation-level deep dive covering Kafka-based agent pipelines, event mesh design, exactly-once delivery semantics, distributed tracing, and production deployment patterns.

**Last Updated:** 2026-07-06  
**Estimated Reading Time:** 80 minutes  
**Prerequisites:** [01-Overview](./01-Overview.md), [02-Core-Topics](./02-Core-Topics.md)

---

## Table of Contents

1. [Kafka-Based Agent Pipelines](#1-kafka-based-agent-pipelines)
2. [Event Mesh for Agent Coordination](#2-event-mesh-for-agent-coordination)
3. [Delivery Semantics: At-Least-Once, Exactly-Once, At-Most-Once](#3-delivery-semantics)
4. [Distributed Tracing in Event-Driven Agent Systems](#4-distributed-tracing)
5. [Event Store Implementation](#5-event-store-implementation)
6. [Real-Time Event Processing with Streaming](#6-real-time-event-processing)
7. [Multi-Tenant Agent Event Systems](#7-multi-tenant-agent-event-systems)
8. [Event-Driven Agent Deployment Patterns](#8-deployment-patterns)
9. [Monitoring and Alerting](#9-monitoring-and-alerting)
10. [Performance Tuning](#10-performance-tuning)
11. [Disaster Recovery](#11-disaster-recovery)
12. [Production Case Studies](#12-production-case-studies)

---

## 1. Kafka-Based Agent Pipelines

### Why Kafka for Agent Systems?

Apache Kafka provides:
- **Durability** — Events survive broker failures
- **Replay** — Re-process events from any point in time
- **Scalability** — Partition-based parallelism
- **Ordering** — Guaranteed ordering within a partition
- **Exactly-once** — With proper configuration

### Kafka Topic Design for Agents

```python
# Topic naming convention
TOPICS = {
    # Raw events from agents
    "agent.events.raw": {
        "partitions": 12,
        "replication": 3,
        "retention": "7d",
        "description": "Raw events emitted by agents"
    },
    
    # Processed events
    "agent.events.processed": {
        "partitions": 12,
        "replication": 3,
        "retention": "30d",
        "description": "Events after initial processing"
    },
    
    # Agent-specific topics
    "agent.research.events": {
        "partitions": 6,
        "replication": 3,
        "retention": "14d",
        "description": "Research agent events"
    },
    
    # Dead letter queue
    "agent.events.dlq": {
        "partitions": 3,
        "replication": 3,
        "retention": "90d",
        "description": "Failed events for manual review"
    },
    
    # Commands to agents
    "agent.commands": {
        "partitions": 6,
        "replication": 3,
        "retention": "1d",
        "description": "Commands sent to agents"
    }
}
```

### Kafka Producer for Agents

```python
from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic
import json
import uuid
from datetime import datetime

class KafkaAgentProducer:
    """Kafka producer for agent events."""
    
    def __init__(self, config: dict):
        self.producer = Producer({
            'bootstrap.servers': config['bootstrap_servers'],
            'client.id': config.get('client.id', 'agent-producer'),
            'acks': 'all',  # Wait for all replicas
            'enable.idempotence': True,  # Exactly-once semantics
            'max.in.flight.requests.per.connection': 5,
            'retries': 2147483647,
            'compression.type': 'snappy',
            'linger.ms': 5,
            'batch.size': 16384,
        })
    
    def emit_event(self, topic: str, event: dict, key: str = None):
        """Emit an event to Kafka."""
        if key is None:
            key = str(uuid.uuid4())
        
        # Serialize event
        value = json.dumps(event, default=str).encode('utf-8')
        
        # Produce with callback
        self.producer.produce(
            topic=topic,
            key=key,
            value=value,
            callback=self._delivery_callback
        )
        self.producer.poll(0)
    
    def _delivery_callback(self, err, msg):
        """Handle delivery confirmation."""
        if err:
            print(f"Delivery failed: {err}")
        else:
            print(f"Delivered to {msg.topic()}[{msg.partition()}] at offset {msg.offset()}")
    
    def flush(self):
        """Wait for all pending messages to be delivered."""
        self.producer.flush()

# Usage
producer = KafkaAgentProducer({
    'bootstrap.servers': 'localhost:9092',
    'client.id': 'research-agent-001'
})

# Emit agent event
producer.emit_event(
    topic="agent.events.raw",
    event={
        "event_id": str(uuid.uuid4()),
        "event_type": "research.started",
        "agent_id": "research_agent_001",
        "timestamp": datetime.utcnow().isoformat(),
        "data": {"topic": "AI event-driven architectures"},
        "metadata": {
            "correlation_id": "corr_123",
            "causation_id": None
        }
    }
)
```

### Kafka Consumer for Agents

```python
from confluent_kafka import Consumer, KafkaError
import json
import asyncio
from typing import Callable, Dict

class KafkaAgentConsumer:
    """Kafka consumer for agent event processing."""
    
    def __init__(self, config: dict):
        self.consumer = Consumer({
            'bootstrap.servers': config['bootstrap_servers'],
            'group.id': config['group.id'],
            'auto.offset.reset': config.get('auto.offset.reset', 'earliest'),
            'enable.auto.commit': False,  # Manual commit for exactly-once
            'max.poll.interval.ms': 300000,  # 5 minutes
            'session.timeout.ms': 30000,
        })
        self.handlers: Dict[str, Callable] = {}
        self.running = False
    
    def subscribe(self, topics: list, handler: Callable):
        """Subscribe to topics with a handler."""
        self.consumer.subscribe(topics)
        self.handlers[topics[0]] = handler  # Simplified
    
    async def start(self):
        """Start consuming events."""
        self.running = True
        
        while self.running:
            msg = self.consumer.poll(1.0)
            
            if msg is None:
                continue
            
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                print(f"Consumer error: {msg.error()}")
                continue
            
            try:
                # Deserialize event
                event = json.loads(msg.value().decode('utf-8'))
                
                # Process event
                handler = self.handlers.get(msg.topic())
                if handler:
                    await handler(event)
                
                # Commit offset
                self.consumer.commit(msg, asynchronous=False)
                
            except Exception as e:
                print(f"Error processing event: {e}")
                # Send to DLQ
                self._send_to_dlq(msg, str(e))
    
    def _send_to_dlq(self, msg, error: str):
        """Send failed event to dead letter queue."""
        # Implementation depends on your DLQ strategy
        pass
    
    def stop(self):
        """Stop consuming events."""
        self.running = False
        self.consumer.close()
```

### Exactly-Once Semantics with Kafka

```python
class ExactlyOnceAgentProcessor:
    """Agent processor with exactly-once semantics."""
    
    def __init__(self, kafka_config: dict, state_store: dict):
        self.producer = Producer({
            **kafka_config,
            'enable.idempotence': True,
            'transactional.id': f"agent-processor-{uuid.uuid4()}"
        })
        self.consumer = Consumer({
            **kafka_config,
            'group.id': 'exactly-once-processor',
            'isolation.level': 'read_committed'
        })
        self.state_store = state_store  # Could be Kafka Streams state store
    
    async def process(self):
        """Process events with exactly-once semantics."""
        self.producer.init_transactions()
        
        while True:
            msg = self.consumer.poll(1.0)
            if msg is None:
                continue
            
            try:
                # Begin transaction
                self.producer.begin_transaction()
                
                # Process event
                event = json.loads(msg.value().decode('utf-8'))
                result = await self.process_event(event)
                
                # Update state store (idempotent)
                self.state_store[event['event_id']] = result
                
                # Produce output event
                self.producer.produce(
                    'agent.events.processed',
                    key=msg.key(),
                    value=json.dumps(result).encode('utf-8')
                )
                
                # Commit consumer offset within transaction
                self.producer.send_offsets_to_transaction(
                    self.consumer.position(self.consumer.assignment()),
                    self.consumer.consumer_group()
                )
                
                # Commit transaction
                self.producer.commit_transaction()
                
            except Exception as e:
                self.producer.abort_transaction()
                print(f"Transaction aborted: {e}")
```

---

## 2. Event Mesh for Agent Coordination

### What is an Event Mesh?

An event mesh is a dynamic, interconnected fabric of event brokers that enables agents to communicate across different environments (cloud, edge, on-premises).

### Event Mesh Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Event Mesh                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Cloud    │  │ Edge     │  │ On-Prem  │  │ SaaS     │   │
│  │ Broker   │←→│ Broker   │←→│ Broker   │←→│ Broker   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│       ↕              ↕              ↕              ↕        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Agent A  │  │ Agent B  │  │ Agent C  │  │ Agent D  │   │
│  │ (Cloud)  │  │ (Edge)   │  │ (On-Prem)│  │ (SaaS)   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Implementation with Solace/Solace-like Patterns

```python
class EventMesh:
    """Event mesh for distributed agent communication."""
    
    def __init__(self):
        self.brokers = {}  # broker_id -> broker_config
        self.routes = {}  # topic_pattern -> [broker_ids]
        self.agent_broker_map = {}  # agent_id -> broker_id
    
    def register_broker(self, broker_id: str, config: dict):
        """Register a broker in the mesh."""
        self.brokers[broker_id] = config
    
    def add_route(self, topic_pattern: str, broker_ids: list):
        """Add routing rule for topic pattern."""
        self.routes[topic_pattern] = broker_ids
    
    def assign_agent(self, agent_id: str, broker_id: str):
        """Assign an agent to a broker."""
        self.agent_broker_map[agent_id] = broker_id
    
    async def route_event(self, event: dict, source_agent: str):
        """Route event through the mesh."""
        source_broker = self.agent_broker_map.get(source_agent)
        if not source_broker:
            raise ValueError(f"Agent {source_agent} not assigned to any broker")
        
        # Find target brokers based on topic
        topic = event.get("event_type", "")
        target_brokers = []
        
        for pattern, brokers in self.routes.items():
            if self._matches_pattern(topic, pattern):
                target_brokers.extend(brokers)
        
        # Remove source broker from targets
        target_brokers = [b for b in set(target_brokers) if b != source_broker]
        
        # Forward event to target brokers
        for broker_id in target_brokers:
            await self._forward_to_broker(broker_id, event)
    
    def _matches_pattern(self, topic: str, pattern: str) -> bool:
        """Check if topic matches pattern (supports wildcards)."""
        import fnmatch
        return fnmatch.fnmatch(topic, pattern)
    
    async def _forward_to_broker(self, broker_id: str, event: dict):
        """Forward event to a specific broker."""
        # Implementation depends on broker type
        pass
```

### Event Mesh with NATS

```python
import nats

class NATSEventMesh:
    """Event mesh using NATS for agent communication."""
    
    def __init__(self):
        self.nc = None
        self.js = None
    
    async def connect(self, servers: str = "nats://localhost:4222"):
        """Connect to NATS."""
        self.nc = await nats.connect(servers)
        self.js = self.nc.jetstream()
    
    async def publish(self, subject: str, data: dict):
        """Publish event to NATS."""
        await self.js.publish(
            subject=subject,
            payload=json.dumps(data).encode()
        )
    
    async def subscribe(self, subject: str, handler):
        """Subscribe to events."""
        async def msg_handler(msg):
            data = json.loads(msg.data.decode())
            await handler(data)
            await msg.ack()
        
        await self.js.subscribe(
            subject=subject,
            cb=msg_handler,
            durable="agent-durable",
            deliver_policy="all"
        )
    
    async def request(self, subject: str, data: dict, timeout: float = 5.0):
        """Request-reply pattern for synchronous agent communication."""
        response = await self.nc.request(
            subject=subject,
            payload=json.dumps(data).encode(),
            timeout=timeout
        )
        return json.loads(response.data.decode())

# Usage
mesh = NATSEventMesh()
await mesh.connect()

# Agent publishes event
await mesh.publish("agent.research.started", {
    "agent_id": "research_001",
    "topic": "event-driven architectures"
})

# Another agent subscribes
await mesh.subscribe("agent.research.started", handle_research_started)
```

---

## 3. Delivery Semantics

### At-Least-Once Delivery

```python
class AtLeastOnceProcessor:
    """Event processor with at-least-once delivery."""
    
    def __init__(self, event_bus, state_store):
        self.event_bus = event_bus
        self.state_store = state_store
        self.processed_events = set()  # Track processed event IDs
    
    async def process(self, event: Event):
        """Process event with at-least-once semantics."""
        event_id = event.event_id
        
        # Check if already processed (idempotency check)
        if event_id in self.processed_events:
            return  # Already processed
        
        # Process event
        result = await self.handle_event(event)
        
        # Store result
        await self.state_store.set(event_id, result)
        
        # Mark as processed
        self.processed_events.add(event_id)
        
        # Acknowledge
        await self.event_bus.ack(event)
    
    async def handle_event(self, event: Event) -> dict:
        """Handle event (idempotent operation)."""
        # Must be idempotent - same event produces same result
        pass
```

### Exactly-Once Delivery

```python
class ExactlyOnceProcessor:
    """Event processor with exactly-once semantics."""
    
    def __init__(self, event_bus, transaction_manager):
        self.event_bus = event_bus
        self.tx_manager = transaction_manager
    
    async def process(self, event: Event):
        """Process event with exactly-once semantics."""
        async with self.tx_manager.transaction():
            # Process event
            result = await self.handle_event(event)
            
            # Store result (within transaction)
            await self.tx_manager.store(event.event_id, result)
            
            # Emit output event (within transaction)
            await self.tx_manager.emit(Event(
                event_type="event.processed",
                data={"original_event_id": event.event_id, "result": result}
            ))
            
            # Acknowledge input event (within transaction)
            await self.tx_manager.ack(event)
            
            # Transaction commits atomically
```

### At-Most-Once Delivery

```python
class AtMostOnceProcessor:
    """Event processor with at-most-once semantics."""
    
    def __init__(self, event_bus):
        self.event_bus = event_bus
    
    async def process(self, event: Event):
        """Process event with at-most-once semantics."""
        try:
            # Acknowledge first
            await self.event_bus.ack(event)
            
            # Then process (if fails, event is lost)
            await self.handle_event(event)
            
        except Exception as e:
            # Event already acknowledged, can't retry
            print(f"Event lost due to error: {e}")
```

---

## 4. Distributed Tracing

### OpenTelemetry Integration

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
import uuid

# Setup tracing
provider = TracerProvider()
processor = BatchSpanProcessor(OTLPSpanExporter())
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("agent-event-system")

class TracedEventBus:
    """Event bus with distributed tracing."""
    
    def __init__(self):
        self.handlers = {}
    
    async def emit(self, event: dict):
        """Emit event with tracing."""
        with tracer.start_as_current_span("emit_event") as span:
            span.set_attribute("event.type", event.get("event_type"))
            span.set_attribute("event.source", event.get("source"))
            span.set_attribute("event.id", event.get("event_id"))
            
            # Add trace context to event metadata
            ctx = trace.set_span_in_context(trace.get_current_span())
            event["metadata"] = event.get("metadata", {})
            event["metadata"]["trace_id"] = span.get_span_context().trace_id
            event["metadata"]["span_id"] = span.get_span_context().span_id
            
            # Publish event
            await self._publish(event)
    
    async def handle(self, event: dict):
        """Handle event with tracing."""
        trace_id = event.get("metadata", {}).get("trace_id")
        span_id = event.get("metadata", {}).get("span_id")
        
        with tracer.start_as_current_span("handle_event") as span:
            span.set_attribute("event.type", event.get("event_type"))
            span.set_attribute("event.id", event.get("event_id"))
            
            # Process event
            handler = self.handlers.get(event.get("event_type"))
            if handler:
                await handler(event)

class TracedAgent:
    """Agent with distributed tracing."""
    
    def __init__(self, agent_id: str, event_bus: TracedEventBus):
        self.agent_id = agent_id
        self.event_bus = event_bus
    
    async def process_event(self, event: dict):
        """Process event with tracing."""
        with tracer.start_as_current_span(f"agent.{self.agent_id}.process") as span:
            span.set_attribute("agent.id", self.agent_id)
            span.set_attribute("event.type", event.get("event_type"))
            
            try:
                result = await self._process(event)
                span.set_status(trace.StatusCode.OK)
                return result
            except Exception as e:
                span.set_status(trace.StatusCode.ERROR, str(e))
                span.record_exception(e)
                raise
```

### Trace Context Propagation

```python
class TraceContext:
    """Manages trace context across event boundaries."""
    
    @staticmethod
    def inject(event: dict, span) -> dict:
        """Inject trace context into event metadata."""
        event["metadata"] = event.get("metadata", {})
        event["metadata"]["trace_id"] = format(span.get_span_context().trace_id, '032x')
        event["metadata"]["span_id"] = format(span.get_span_context().span_id, '016x')
        event["metadata"]["trace_flags"] = span.get_span_context().trace_flags
        return event
    
    @staticmethod
    def extract(event: dict):
        """Extract trace context from event metadata."""
        metadata = event.get("metadata", {})
        return {
            "trace_id": metadata.get("trace_id"),
            "span_id": metadata.get("span_id"),
            "trace_flags": metadata.get("trace_flags")
        }
```

---

## 5. Event Store Implementation

### PostgreSQL Event Store

```python
import asyncpg
from typing import List, Optional
import json

class PostgresEventStore:
    """PostgreSQL-based event store for agents."""
    
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.pool = None
    
    async def connect(self):
        """Connect to PostgreSQL."""
        self.pool = await asyncpg.create_pool(self.dsn)
        
        # Create tables
        await self.pool.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id UUID PRIMARY KEY,
                event_type VARCHAR(255) NOT NULL,
                agent_id VARCHAR(255) NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                data JSONB NOT NULL,
                metadata JSONB DEFAULT '{}',
                version INTEGER NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            
            CREATE INDEX IF NOT EXISTS idx_events_agent_id ON events(agent_id);
            CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
            CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
        """)
    
    async def append(self, event: dict):
        """Append event to store."""
        await self.pool.execute("""
            INSERT INTO events (event_id, event_type, agent_id, timestamp, data, metadata, version)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
            event["event_id"],
            event["event_type"],
            event["agent_id"],
            event["timestamp"],
            json.dumps(event["data"]),
            json.dumps(event.get("metadata", {})),
            event.get("version", 1)
        )
    
    async def get_events(self, agent_id: str, after_version: int = 0) -> List[dict]:
        """Get events for an agent."""
        rows = await self.pool.fetch("""
            SELECT * FROM events 
            WHERE agent_id = $1 AND version > $2
            ORDER BY version ASC
        """, agent_id, after_version)
        
        return [self._row_to_event(row) for row in rows]
    
    async def get_events_by_type(self, event_type: str, 
                                  after: str = None) -> List[dict]:
        """Get events by type."""
        if after:
            rows = await self.pool.fetch("""
                SELECT * FROM events 
                WHERE event_type = $1 AND timestamp > $2
                ORDER BY timestamp ASC
            """, event_type, after)
        else:
            rows = await self.pool.fetch("""
                SELECT * FROM events 
                WHERE event_type = $1
                ORDER BY timestamp ASC
            """, event_type)
        
        return [self._row_to_event(row) for row in rows]
    
    async def create_snapshot(self, agent_id: str, version: int, state: dict):
        """Create a state snapshot."""
        await self.pool.execute("""
            INSERT INTO snapshots (agent_id, version, state, created_at)
            VALUES ($1, $2, $3, NOW())
            ON CONFLICT (agent_id) DO UPDATE
            SET version = EXCLUDED.version, state = EXCLUDED.state, created_at = NOW()
        """, agent_id, version, json.dumps(state))
    
    async def get_latest_snapshot(self, agent_id: str) -> Optional[dict]:
        """Get latest snapshot for an agent."""
        row = await self.pool.fetchrow("""
            SELECT * FROM snapshots WHERE agent_id = $1
        """, agent_id)
        
        if row:
            return {"version": row["version"], "state": json.loads(row["state"])}
        return None
    
    def _row_to_event(self, row) -> dict:
        """Convert database row to event dict."""
        return {
            "event_id": str(row["event_id"]),
            "event_type": row["event_type"],
            "agent_id": row["agent_id"],
            "timestamp": row["timestamp"].isoformat(),
            "data": json.loads(row["data"]),
            "metadata": json.loads(row["metadata"]),
            "version": row["version"]
        }
```

### Redis Event Store

```python
import redis.asyncio as redis
import json
from typing import List, Optional

class RedisEventStore:
    """Redis-based event store for agents (fast, in-memory)."""
    
    def __init__(self, redis_url: str = "redis://localhost"):
        self.redis = redis.from_url(redis_url)
    
    async def append(self, event: dict):
        """Append event to Redis stream."""
        agent_id = event["agent_id"]
        stream_key = f"agent:{agent_id}:events"
        
        # Add to stream
        await self.redis.xadd(
            stream_key,
            {
                "event_id": event["event_id"],
                "event_type": event["event_type"],
                "timestamp": event["timestamp"],
                "data": json.dumps(event["data"]),
                "metadata": json.dumps(event.get("metadata", {})),
                "version": str(event.get("version", 1))
            }
        )
        
        # Update current version
        await self.redis.set(
            f"agent:{agent_id}:version",
            event.get("version", 1)
        )
    
    async def get_events(self, agent_id: str, after_version: int = 0) -> List[dict]:
        """Get events from Redis stream."""
        stream_key = f"agent:{agent_id}:events"
        
        # Read from stream
        events = await self.redis.xrange(
            stream_key,
            min="-",
            max="+",
            count=1000
        )
        
        result = []
        for event_id, fields in events:
            version = int(fields[b"version"])
            if version > after_version:
                result.append({
                    "event_id": fields[b"event_id"].decode(),
                    "event_type": fields[b"event_type"].decode(),
                    "agent_id": agent_id,
                    "timestamp": fields[b"timestamp"].decode(),
                    "data": json.loads(fields[b"data"]),
                    "metadata": json.loads(fields[b"metadata"]),
                    "version": version
                })
        
        return result
    
    async def get_current_version(self, agent_id: str) -> int:
        """Get current version for an agent."""
        version = await self.redis.get(f"agent:{agent_id}:version")
        return int(version) if version else 0
```

---

## 6. Real-Time Event Processing

### Stream Processing with Flink-like Patterns

```python
import asyncio
from typing import Callable, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class Window:
    start: datetime
    end: datetime
    events: List[dict]

class StreamingEventProcessor:
    """Stream processor for real-time event analysis."""
    
    def __init__(self, window_size: timedelta, slide_interval: timedelta):
        self.window_size = window_size
        self.slide_interval = slide_interval
        self.windows: List[Window] = []
        self.current_window = None
    
    async def process_event(self, event: dict):
        """Process event in streaming fashion."""
        event_time = datetime.fromisoformat(event["timestamp"])
        
        # Create new window if needed
        if self.current_window is None or event_time > self.current_window.end:
            await self._close_window()
            self.current_window = Window(
                start=event_time,
                end=event_time + self.window_size,
                events=[]
            )
        
        # Add event to current window
        self.current_window.events.append(event)
        
        # Check if window should be emitted
        if event_time >= self.current_window.start + self.slide_interval:
            await self._emit_window()
    
    async def _emit_window(self):
        """Emit current window for processing."""
        if self.current_window and self.current_window.events:
            await self._process_window(self.current_window)
            self.windows.append(self.current_window)
            self.current_window = None
    
    async def _close_window(self):
        """Close current window."""
        if self.current_window:
            await self._emit_window()
    
    async def _process_window(self, window: Window):
        """Process a window of events (override in subclass)."""
        pass

class AgentMetricsProcessor(StreamingEventProcessor):
    """Process agent metrics in real-time."""
    
    async def _process_window(self, window: Window):
        """Calculate metrics for a window of events."""
        events = window.events
        
        # Calculate metrics
        event_types = {}
        for event in events:
            event_type = event.get("event_type", "unknown")
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        metrics = {
            "window_start": window.start.isoformat(),
            "window_end": window.end.isoformat(),
            "total_events": len(events),
            "event_type_counts": event_types,
            "avg_events_per_second": len(events) / max((window.end - window.start).total_seconds(), 1)
        }
        
        # Emit metrics event
        await self._emit_metrics(metrics)
    
    async def _emit_metrics(self, metrics: dict):
        """Emit calculated metrics."""
        # Send to monitoring system
        pass
```

### Complex Event Processing (CEP)

```python
class CEPattern:
    """Complex Event Processing pattern matcher."""
    
    def __init__(self):
        self.patterns = {}
        self.partial_matches = {}
    
    def register_pattern(self, name: str, pattern: list):
        """Register a CEP pattern.
        
        Pattern is a list of (event_type, condition) tuples.
        Example: [("order.placed", None), ("payment.processed", lambda e: e["amount"] > 100)]
        """
        self.patterns[name] = pattern
        self.partial_matches[name] = []
    
    async def process_event(self, event: dict) -> list:
        """Process event against all patterns."""
        matched_patterns = []
        
        for name, pattern in self.patterns.items():
            # Check if this event continues any partial matches
            new_partials = []
            for partial in self.partial_matches.get(name, []):
                expected_type, condition = pattern[partial["step"]]
                
                if (event["event_type"] == expected_type and
                    (condition is None or condition(event))):
                    
                    new_step = partial["step"] + 1
                    
                    if new_step >= len(pattern):
                        # Pattern complete!
                        matched_patterns.append(name)
                    else:
                        new_partials.append({
                            "step": new_step,
                            "events": partial["events"] + [event]
                        })
            
            # Check if this event starts a new pattern match
            first_type, first_condition = pattern[0]
            if (event["event_type"] == first_type and
                (first_condition is None or first_condition(event))):
                
                if len(pattern) > 1:
                    new_partials.append({
                        "step": 1,
                        "events": [event]
                    })
            
            self.partial_matches[name] = new_partials
        
        return matched_patterns

# Usage: Detect suspicious agent behavior
cep = CEPattern()

# Pattern: Agent fails 3 times in 5 minutes
cep.register_pattern("agent_failure_spike", [
    ("agent.failed", None),
    ("agent.failed", lambda e: True),  # Any failure
    ("agent.failed", lambda e: True),  # Third failure
])

# Pattern: High-value transaction followed by immediate refund
cep.register_pattern("suspicious_refund", [
    ("transaction.completed", lambda e: e.get("amount", 0) > 1000),
    ("transaction.refunded", None),
])
```

---

## 7. Multi-Tenant Agent Event Systems

### Tenant Isolation

```python
class MultiTenantEventBus:
    """Event bus with tenant isolation."""
    
    def __init__(self):
        self.tenant_topics = {}  # tenant_id -> [topics]
        self.tenant_permissions = {}  # tenant_id -> {permissions}
    
    def create_tenant(self, tenant_id: str, config: dict):
        """Create isolated event bus for a tenant."""
        self.tenant_topics[tenant_id] = {
            "events": f"tenant.{tenant_id}.events",
            "commands": f"tenant.{tenant_id}.commands",
            "dlq": f"tenant.{tenant_id}.dlq"
        }
        
        self.tenant_permissions[tenant_id] = config.get("permissions", {
            "max_events_per_second": 100,
            "max_topics": 10,
            "retention_days": 7
        })
    
    async def emit(self, tenant_id: str, event: dict):
        """Emit event for a specific tenant."""
        # Validate tenant
        if tenant_id not in self.tenant_topics:
            raise ValueError(f"Unknown tenant: {tenant_id}")
        
        # Check rate limits
        await self._check_rate_limit(tenant_id)
        
        # Add tenant context
        event["tenant_id"] = tenant_id
        event["metadata"] = event.get("metadata", {})
        event["metadata"]["tenant_id"] = tenant_id
        
        # Route to tenant-specific topic
        topic = self.tenant_topics[tenant_id]["events"]
        await self._publish(topic, event)
    
    async def subscribe(self, tenant_id: str, handler):
        """Subscribe to tenant events."""
        topic = self.tenant_topics[tenant_id]["events"]
        await self._subscribe(topic, handler)
    
    async def _check_rate_limit(self, tenant_id: str):
        """Check tenant rate limit."""
        # Implement rate limiting logic
        pass
```

### Cross-Tenant Event Sharing

```python
class CrossTenantEventBridge:
    """Bridge for sharing events across tenants."""
    
    def __init__(self, event_bus: MultiTenantEventBus):
        self.event_bus = event_bus
        self.sharing_rules = {}  # (source_tenant, event_type) -> [target_tenants]
    
    def add_sharing_rule(self, source_tenant: str, event_type: str, 
                         target_tenants: list):
        """Add rule for cross-tenant event sharing."""
        self.sharing_rules[(source_tenant, event_type)] = target_tenants
    
    async def process_event(self, tenant_id: str, event: dict):
        """Process event and share if rules exist."""
        # Emit to source tenant
        await self.event_bus.emit(tenant_id, event)
        
        # Check sharing rules
        key = (tenant_id, event.get("event_type"))
        target_tenants = self.sharing_rules.get(key, [])
        
        # Share with target tenants
        for target_tenant in target_tenants:
            shared_event = {
                **event,
                "metadata": {
                    **event.get("metadata", {}),
                    "source_tenant": tenant_id,
                    "shared": True
                }
            }
            await self.event_bus.emit(target_tenant, shared_event)
```

---

## 8. Deployment Patterns

### Kubernetes Deployment

```yaml
# agent-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: research-agent
  labels:
    app: research-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: research-agent
  template:
    metadata:
      labels:
        app: research-agent
    spec:
      containers:
      - name: agent
        image: myregistry/research-agent:latest
        env:
        - name: KAFKA_BOOTSTRAP_SERVERS
          value: "kafka-cluster:9092"
        - name: AGENT_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: research-agent-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: research-agent
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Docker Compose for Development

```yaml
# docker-compose.yaml
version: '3.8'

services:
  kafka:
    image: confluentinc/cp-kafka:latest
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
    ports:
      - "9092:9092"
    depends_on:
      - zookeeper

  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: agent_events
      POSTGRES_USER: agent
      POSTGRES_PASSWORD: secret
    ports:
      - "5432:5432"

  research-agent:
    build: ./agents/research
    environment:
      KAFKA_BOOTSTRAP_SERVERS: kafka:9092
      REDIS_URL: redis://redis:6379
      DATABASE_URL: postgres://agent:secret@postgres:5432/agent_events
    depends_on:
      - kafka
      - redis
      - postgres

  classification-agent:
    build: ./agents/classification
    environment:
      KAFKA_BOOTSTRAP_SERVERS: kafka:9092
    depends_on:
      - kafka
```

---

## 9. Monitoring and Alerting

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define metrics
events_emitted = Counter(
    'agent_events_emitted_total',
    'Total events emitted by agents',
    ['agent_id', 'event_type']
)

events_processed = Counter(
    'agent_events_processed_total',
    'Total events processed by agents',
    ['agent_id', 'event_type', 'status']
)

event_processing_duration = Histogram(
    'agent_event_processing_seconds',
    'Time spent processing events',
    ['agent_id', 'event_type']
)

agent_active = Gauge(
    'agent_active',
    'Whether agent is currently active',
    ['agent_id']
)

queue_depth = Gauge(
    'agent_queue_depth',
    'Number of events in agent queue',
    ['agent_id']
)

class MonitoredAgent:
    """Agent with Prometheus metrics."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        agent_active.labels(agent_id=agent_id).set(1)
    
    async def process_event(self, event: dict):
        """Process event with metrics."""
        event_type = event.get("event_type", "unknown")
        
        with event_processing_duration.labels(
            agent_id=self.agent_id, 
            event_type=event_type
        ).time():
            try:
                await self._process(event)
                events_processed.labels(
                    agent_id=self.agent_id,
                    event_type=event_type,
                    status="success"
                ).inc()
            except Exception as e:
                events_processed.labels(
                    agent_id=self.agent_id,
                    event_type=event_type,
                    status="error"
                ).inc()
                raise
    
    def emit_event(self, event_type: str, data: dict):
        """Emit event with metrics."""
        events_emitted.labels(
            agent_id=self.agent_id,
            event_type=event_type
        ).inc()
        
        # Actually emit event
        pass
```

### Alerting Rules

```yaml
# alerts.yaml
groups:
  - name: agent-alerts
    rules:
      - alert: AgentProcessingSlow
        expr: rate(agent_event_processing_seconds_sum[5m]) / rate(agent_event_processing_seconds_count[5m]) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Agent processing is slow"
          description: "Agent {{ $labels.agent_id }} is processing events slowly"

      - alert: AgentErrorRateHigh
        expr: rate(agent_events_processed_total{status="error"}[5m]) / rate(agent_events_processed_total[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Agent error rate is high"
          description: "Agent {{ $labels.agent_id }} has >10% error rate"

      - alert: AgentQueueDepthHigh
        expr: agent_queue_depth > 1000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Agent queue depth is high"
          description: "Agent {{ $labels.agent_id }} has {{ $value }} events queued"

      - alert: AgentDown
        expr: agent_active == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Agent is down"
          description: "Agent {{ $labels.agent_id }} is not responding"
```

---

## 10. Performance Tuning

### Batch Processing

```python
class BatchedProcessor:
    """Process events in batches for throughput."""
    
    def __init__(self, batch_size: int = 100, flush_interval: float = 1.0):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.batch = []
        self.last_flush = time.time()
    
    async def add_event(self, event: dict):
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
    
    async def process_batch(self, batch: list):
        """Override to implement batch processing."""
        pass
```

### Connection Pooling

```python
class PooledEventProcessor:
    """Event processor with connection pooling."""
    
    def __init__(self, pool_size: int = 10):
        self.pool = asyncio.Queue(maxsize=pool_size)
        self._initialize_pool(pool_size)
    
    def _initialize_pool(self, size: int):
        """Initialize connection pool."""
        for _ in range(size):
            connection = self._create_connection()
            self.pool.put_nowait(connection)
    
    def _create_connection(self):
        """Create a new connection (override in subclass)."""
        pass
    
    async def get_connection(self):
        """Get connection from pool."""
        return await self.pool.get()
    
    async def release_connection(self, connection):
        """Return connection to pool."""
        await self.pool.put(connection)
    
    async def process_with_connection(self, event: dict):
        """Process event using pooled connection."""
        connection = await self.get_connection()
        try:
            await self._process_with_connection(connection, event)
        finally:
            await self.release_connection(connection)
```

---

## 11. Disaster Recovery

### Event Store Backup

```python
class EventStoreBackup:
    """Backup and restore event store."""
    
    def __init__(self, event_store, backup_storage):
        self.event_store = event_store
        self.backup_storage = backup_storage
    
    async def backup(self, agent_id: str):
        """Backup events for an agent."""
        events = await self.event_store.get_events(agent_id)
        snapshot = await self.event_store.get_latest_snapshot(agent_id)
        
        backup = {
            "agent_id": agent_id,
            "events": events,
            "snapshot": snapshot,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.backup_storage.store(f"backup/{agent_id}", backup)
    
    async def restore(self, agent_id: str):
        """Restore events for an agent."""
        backup = await self.backup_storage.load(f"backup/{agent_id}")
        
        # Restore snapshot
        if backup.get("snapshot"):
            await self.event_store.create_snapshot(
                agent_id,
                backup["snapshot"]["version"],
                backup["snapshot"]["state"]
            )
        
        # Restore events
        for event in backup.get("events", []):
            await self.event_store.append(event)
```

### Failover Strategy

```python
class FailoverEventBus:
    """Event bus with automatic failover."""
    
    def __init__(self, primary_broker, secondary_broker):
        self.primary = primary_broker
        self.secondary = secondary_broker
        self.active = primary_broker
        self.failover_threshold = 3  # Failures before failover
        self.failure_count = 0
    
    async def emit(self, event: dict):
        """Emit event with failover."""
        try:
            await self.active.emit(event)
            self.failure_count = 0
        except Exception as e:
            self.failure_count += 1
            
            if self.failure_count >= self.failover_threshold:
                await self._failover()
                await self.active.emit(event)
            else:
                raise
    
    async def _failover(self):
        """Switch to secondary broker."""
        if self.active == self.primary:
            self.active = self.secondary
        else:
            self.active = self.primary
        
        self.failure_count = 0
        # Log failover event
```

---

## 12. Production Case Studies

### Case Study 1: Document Processing Pipeline (Calfkit-style)

**Company:** Financial services firm processing 100K+ documents daily

**Challenge:**
- Documents arrive in multiple formats (PDF, images, emails)
- Need to extract, classify, and route to appropriate processing agents
- Must handle高峰 (10x normal load during month-end)

**Solution:**
- Kafka-based event streaming with 12 partitions
- Event-driven agents: OCR → Classification → Extraction → Validation → Routing
- Auto-scaling based on queue depth
- Dead letter queue for manual review

**Results:**
- 99.9% uptime
- Processing time reduced from 24 hours to 2 hours
- Cost reduced by 60% through efficient resource utilization

### Case Study 2: Customer Support Agent System (Arvo-style)

**Company:** SaaS company with 50K+ support tickets daily

**Challenge:**
- Multiple agent types (billing, technical, general)
- Need to route tickets to appropriate agents
- Must handle escalation to human agents
- Need full audit trail for compliance

**Solution:**
- Event mesh connecting cloud and edge agents
- Complex event processing for ticket routing
- Event sourcing for complete audit trail
- Saga pattern for multi-step resolution workflows

**Results:**
- 40% reduction in resolution time
- 95% automated routing accuracy
- Complete audit trail for all agent decisions

### Case Study 3: Real-Time Monitoring System (Stream0-style)

**Company:** IoT company with 1M+ sensors

**Challenge:**
- Process millions of events per second
- Detect anomalies in real-time
- Trigger automated remediation
- Handle backpressure during traffic spikes

**Solution:**
- Reactive streams with backpressure
- Windowed aggregation for metrics
- Event-driven agents for anomaly detection
- Saga pattern for remediation workflows

**Results:**
- Sub-second anomaly detection
- 99.99% event processing reliability
- Automated remediation for 80% of issues

---

## Summary

Key technical patterns for production event-driven agent systems:

1. **Kafka for durability and replay** — Use idempotent producers and manual commits
2. **Event mesh for distributed agents** — Connect agents across environments
3. **Exactly-once semantics** — Use transactions and idempotent handlers
4. **Distributed tracing** — OpenTelemetry for end-to-end visibility
5. **Event store** — PostgreSQL for durability, Redis for speed
6. **Streaming for real-time** — Windowed aggregation and CEP
7. **Multi-tenant isolation** — Tenant-specific topics and rate limits
8. **Kubernetes deployment** — HPA for auto-scaling
9. **Prometheus monitoring** — Metrics and alerting
10. **Disaster recovery** — Backup, restore, and failover

---

*This document is part of the AI Knowledge Library — a comprehensive reference for AI practitioners and researchers.*

---
**See also:**
- [07 — AI Coding Agent Sustainability: Managing Hidden Costs, Code Quality Decay, and Developer Burnout](33-AI-Native-Software-Development/07-AI-Coding-Agent-Sustainability.md)
- [48 — MCP Cloud Infrastructure & Agent-as-a-Service: The Production Layer of Agentic AI](48-MCP-Cloud-Infrastructure-Agent-as-a-Service/01-Overview.md)
- [AI Agent Commerce & Agent-to-Agent (A2A) Payments: The 2026 Frontier](28-AI-Agent-Commerce-and-A2A-Payments/01-Overview.md)
