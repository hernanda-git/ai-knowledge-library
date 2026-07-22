# Tools and Frameworks: Event-Driven Agent Architectures

> Comprehensive survey of tools, frameworks, and platforms for building event-driven agent systems — covering message brokers, event streaming, agent frameworks, workflow engines, and observability tools.

**Last Updated:** 2026-07-06  
**Estimated Reading Time:** 60 minutes  
**Prerequisites:** [01-Overview](./01-Overview.md)

---

## Table of Contents

1. [Message Brokers and Event Streaming](#1-message-brokers-and-event-streaming)
2. [Agent-Specific Event Frameworks](#2-agent-specific-event-frameworks)
3. [Workflow and Orchestration Engines](#3-workflow-and-orchestration-engines)
4. [Event Store Solutions](#4-event-store-solutions)
5. [Observability and Monitoring](#5-observability-and-monitoring)
6. [Cloud-Native Event Services](#6-cloud-native-event-services)
7. [Edge and IoT Event Systems](#7-edge-and-iot-event-systems)
8. [Comparison Matrix](#8-comparison-matrix)
9. [Selection Guide](#9-selection-guide)
10. [Getting Started Stacks](#10-getting-started-stacks)

---

## 1. Message Brokers and Event Streaming

### Apache Kafka

**Overview:** Distributed event streaming platform capable of handling trillions of events per day.

**Key Features:**
- Durable, replicated event logs
- Exactly-once semantics
- Stream processing with Kafka Streams/ksqlDB
- Schema registry for event evolution
- Consumer groups for parallel processing

**Best For:** High-throughput, durable event streaming; multi-agent coordination at scale.

**Agent Use Cases:**
- Agent event buses
- Long-term event storage and replay
- Multi-agent coordination channels

```python
# Kafka producer example
from confluent_kafka import Producer

producer = Producer({
    'bootstrap.servers': 'localhost:9092',
    'acks': 'all',
    'enable.idempotence': True
})

producer.produce(
    topic='agent.events',
    key='agent_001',
    value=json.dumps({
        'event_type': 'task.completed',
        'agent_id': 'agent_001',
        'data': {'result': 'success'}
    }).encode()
)
```

### Apache Pulsar

**Overview:** Multi-tenant, geo-replicated messaging and streaming.

**Key Features:**
- Multi-tenancy built-in
- Geo-replication
- Tiered storage
- Functions (serverless processing)
- Schema evolution

**Best For:** Multi-tenant agent systems; geo-distributed deployments.

### RabbitMQ

**Overview:** Traditional message broker with flexible routing.

**Key Features:**
- Multiple messaging protocols (AMQP, MQTT, STOMP)
- Flexible routing (exchanges, bindings)
- Priority queues
- Dead letter queues
- Management UI

**Best For:** Simple agent communication; task queues; priority-based routing.

```python
# RabbitMQ example
import pika

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

channel.exchange_declare(exchange='agent_events', exchange_type='topic')
channel.queue_declare(queue='research_agent', durable=True)
channel.queue_bind(
    exchange='agent_events',
    queue='research_agent',
    routing_key='agent.research.*'
)

channel.basic_publish(
    exchange='agent_events',
    routing_key='agent.research.started',
    body=json.dumps({'topic': 'AI architectures'})
)
```

### NATS

**Overview:** High-performance messaging for cloud-native applications.

**Key Features:**
- Extremely low latency
- JetStream for persistence
- Request-reply pattern
- Leaf nodes for edge
- Super-cluster for global

**Best For:** Edge agent communication; real-time agent coordination; IoT agents.

```python
# NATS example
import nats

async def main():
    nc = await nats.connect("nats://localhost:4222")
    js = nc.jetstream()
    
    # Publish
    await js.publish("agent.events", b'{"event": "started"}')
    
    # Subscribe
    async def handler(msg):
        data = json.loads(msg.data.decode())
        await process_event(data)
        await msg.ack()
    
    await js.subscribe("agent.events", cb=handler, durable="agent-durable")
```

### Redis Streams

**Overview:** Redis-based event streaming with consumer groups.

**Key Features:**
- In-memory performance
- Consumer groups
- Stream trimming
- Blocking reads
- Redisson for Java

**Best For:** Fast event processing; small-to-medium scale; caching layer.

```python
# Redis Streams example
import redis.asyncio as redis

r = redis.from_url("redis://localhost")

# Add event
await r.xadd("agent:events", {
    "event_type": "task.started",
    "agent_id": "agent_001",
    "data": json.dumps({"task": "research"})
})

# Read events with consumer group
events = await r.xreadgroup(
    groupname="agent_group",
    consumername="consumer_001",
    streams={"agent:events": ">"},
    count=10
)
```

### Redpanda

**Overview:** Kafka-compatible streaming platform with lower latency and simpler ops.

**Key Features:**
- Kafka API compatible
- No JVM (C++ implementation)
- Lower latency
- Tiered storage
- Schema registry

**Best For:** Teams wanting Kafka semantics with simpler operations.

### WarpStream

**Overview:** Kafka-compatible streaming on object storage.

**Key Features:**
- No local storage (S3/GCS)
- Lower cost
- Kafka compatible
- Serverless-like

**Best For:** Cost-optimized event streaming; archival event storage.

---

## 2. Agent-Specific Event Frameworks

### Calfkit

**Overview:** SDK for building distributed event-driven AI agents on Kafka.

**Key Features:**
- Agent lifecycle management
- Event-driven agent coordination
- Built on Apache Kafka
- Type-safe event schemas
- Built-in observability

**Best For:** Production agent systems needing Kafka durability.

```python
# Calfkit-style pattern
from calfkit import Agent, Event, EventBus

class ResearchAgent(Agent):
    @event_handler("task.received")
    async def handle_task(self, event: Event):
        # Process task
        result = await self.research(event.data["topic"])
        
        # Emit completion event
        await self.emit(Event(
            type="task.completed",
            data={"result": result}
        ))

# Agent automatically subscribes to "task.received" events
agent = ResearchAgent()
event_bus = EventBus(brokers=["localhost:9092"])
event_bus.register(agent)
```

### Arvo

**Overview:** TypeScript toolkit for event-driven agentic systems and mesh.

**Key Features:**
- TypeScript-first
- Event mesh support
- Type-safe event definitions
- Agent composition
- Built-in tracing

**Best For:** TypeScript/Node.js agent systems; event mesh architectures.

```typescript
// Arvo-style pattern
import { Agent, Event, EventBus } from 'arvo';

interface ResearchEvents {
  'task.received': { topic: string };
  'task.completed': { result: string };
}

const researchAgent = new Agent<ResearchEvents>({
  name: 'research-agent',
  handlers: {
    'task.received': async (event) => {
      const result = await research(event.data.topic);
      return { type: 'task.completed', data: { result } };
    }
  }
});

const bus = new EventBus({ brokers: ['localhost:9092'] });
bus.register(researchAgent);
```

### Loom

**Overview:** Event-driven OS for AI agents.

**Key Features:**
- Agent operating system
- Event-driven scheduling
- Resource management
- Inter-agent communication
- State persistence

**Best For:** Complex multi-agent systems needing an OS-level abstraction.

### Stream0

**Overview:** HTTP-native messaging layer for AI agents.

**Key Features:**
- HTTP-based (no special protocol)
- Webhook-native
- Simple API
- Cloudflare Workers compatible
- No infrastructure needed

**Best For:** Simple agent communication; webhook-based integrations.

```python
# Stream0-style pattern (HTTP-based)
import httpx

class Stream0Agent:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.handlers = {}
    
    async def subscribe(self, event_type: str, handler):
        """Subscribe to event type via webhook."""
        self.handlers[event_type] = handler
        await httpx.post(f"{self.endpoint}/subscribe", json={
            "event_type": event_type,
            "callback_url": f"{self.callback_url}/events/{event_type}"
        })
    
    async def emit(self, event_type: str, data: dict):
        """Emit event via HTTP."""
        await httpx.post(f"{self.endpoint}/events", json={
            "type": event_type,
            "data": data
        })
```

### DocuFlow

**Overview:** Open-source event-driven AI document processing pipeline.

**Key Features:**
- Document-specific agents
- OCR → Classification → Extraction pipeline
- Event-driven architecture
- Pluggable backends

**Best For:** Document processing workflows.

### SensorHub

**Overview:** Event-driven agent sensor layer for IoT/edge.

**Key Features:**
- Sensor data ingestion
- Event-driven processing
- Edge computing support
- Protocol translation

**Best For:** IoT agent systems; edge AI; sensor-driven agents.

---

## 3. Workflow and Orchestration Engines

### Temporal

**Overview:** Durable execution platform for reliable agent workflows.

**Key Features:**
- Durable execution (survives crashes)
- Workflow versioning
- Saga support
- Activity retries
- TypeScript/Go/Java SDKs

**Best For:** Long-running agent workflows; saga patterns; reliable execution.

```python
# Temporal workflow for agent coordination
from temporalio import workflow

@workflow.defn
class AgentOrchestrationWorkflow:
    @workflow.run
    async def run(self, task: dict):
        # Step 1: Research
        research_result = await workflow.execute_activity(
            research_activity,
            task["topic"],
            start_to_close_timeout=timedelta(minutes=5)
        )
        
        # Step 2: Analyze (parallel)
        analysis_results = await asyncio.gather(*[
            workflow.execute_activity(
                analyze_activity,
                finding,
                start_to_close_timeout=timedelta(minutes=2)
            )
            for finding in research_result["findings"]
        ])
        
        # Step 3: Synthesize
        final_result = await workflow.execute_activity(
            synthesize_activity,
            analysis_results,
            start_to_close_timeout=timedelta(minutes=3)
        )
        
        return final_result
```

### Apache Airflow

**Overview:** Workflow orchestration platform for data pipelines.

**Key Features:**
- DAG-based workflows
- Rich scheduling
- Extensible operators
- Large ecosystem
- Monitoring UI

**Best For:** Data-intensive agent workflows; scheduled agent tasks.

### Prefect

**Overview:** Modern workflow orchestration with Python.

**Key Features:**
- Python-native
- Dynamic workflows
- Event-driven triggers
- Cloud and self-hosted
- Rich monitoring

**Best For:** Python agent workflows; dynamic agent orchestration.

### Dagster

**Overview:** Data asset platform with strong typing.

**Key Features:**
- Software-defined assets
- Strong typing
- Rich testing
- Event-driven
- Good DX

**Best For:** Data pipeline agents; strongly-typed agent workflows.

### AWS Step Functions

**Overview:** Serverless workflow orchestration on AWS.

**Key Features:**
- Visual workflows
- Built-in retries
- Parallel execution
- Human approval steps
- Integration with AWS services

**Best For:** AWS-native agent workflows; serverless agent orchestration.

---

## 4. Event Store Solutions

### EventStoreDB

**Overview:** Purpose-built database for event sourcing.

**Key Features:**
- Native event sourcing
- Projections
- Subscriptions
- Temporal queries
- Catch-up subscriptions

**Best For:** Event-sourced agent systems; temporal queries.

```python
# EventStoreDB example
from eventstoredb import EventStoreDBClient, NewEvent, StreamPosition

client = EventStoreDBClient(uri="esdb://localhost:2113")

# Append event
await client.append_to_stream(
    stream_name="agent-research-001",
    events=[
        NewEvent(
            type="ResearchStarted",
            data=json.dumps({"topic": "AI architectures"}).encode()
        )
    ]
)

# Read events
events = await client.read_stream(
    stream_name="agent-research-001",
    stream_position=StreamPosition.START
)
```

### Marten

**Overview:** .NET document database with event sourcing.

**Key Features:**
- Event sourcing on PostgreSQL
- Document storage
- Projections
- CQRS support
- .NET ecosystem

**Best For:** .NET agent systems; event sourcing with PostgreSQL.

### Axon Framework

**Overview:** Java/Kotlin framework for event-driven microservices.

**Key Features:**
- CQRS/Event Sourcing
- Command handling
- Event handling
- Sagas
- Tracking processors

**Best For:** Java/Kotlin agent systems; enterprise event-driven architectures.

---

## 5. Observability and Monitoring

### OpenTelemetry

**Overview:** Vendor-neutral observability framework.

**Key Features:**
- Distributed tracing
- Metrics collection
- Log correlation
- Auto-instrumentation
- Multiple exporters

**Best For:** End-to-end visibility in agent systems.

```python
# OpenTelemetry example
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Setup
provider = TracerProvider()
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="otel-collector:4317"))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("agent-system")

# Use
async def process_event(event):
    with tracer.start_as_current_span("process_event") as span:
        span.set_attribute("event.type", event["type"])
        result = await handle_event(event)
        span.set_attribute("result", str(result))
        return result
```

### Langfuse

**Overview:** Open-source LLM observability platform.

**Key Features:**
- LLM call tracking
- Cost tracking
- Evaluation
- Prompt management
- Self-hosted

**Best For:** LLM agent observability; cost tracking.

### Phoenix (Arize)

**Overview:** Open-source LLM observability.

**Key Features:**
- Tracing
- Evaluation
- drift detection
- Embeddings visualization

**Best For:** LLM debugging; evaluation.

### Prometheus + Grafana

**Overview:** Metrics collection and visualization.

**Key Features:**
- Time-series metrics
- Alerting
- Rich dashboards
- Large ecosystem

**Best For:** Agent metrics; operational monitoring.

```yaml
# Prometheus config for agents
scrape_configs:
  - job_name: 'agent-metrics'
    static_configs:
      - targets: ['agent-1:8080', 'agent-2:8080']
    metrics_path: '/metrics'
```

### Jaeger

**Overview:** Distributed tracing system.

**Key Features:**
- OpenTelemetry native
- Trace visualization
- Service dependency maps
- Performance analysis

**Best For:** Distributed trace analysis in multi-agent systems.

---

## 6. Cloud-Native Event Services

### AWS EventBridge

**Overview:** Serverless event bus for AWS services.

**Key Features:**
- Content-based routing
- Schema registry
- Partner integrations
- Archive and replay
- 35+ AWS service targets

**Best For:** AWS-native agent systems; serverless event routing.

```python
# EventBridge example
import boto3

eb = boto3.client('events')

# Put event
eb.put_events(
    Entries=[{
        'Source': 'agent.research',
        'DetailType': 'TaskCompleted',
        'Detail': json.dumps({
            'agent_id': 'research_001',
            'result': 'success'
        })
    }]
)
```

### Google Cloud Eventarc

**Overview:** Event routing for Google Cloud.

**Key Features:**
- Cloud Run integration
- AuditLog events
- Storage events
- Pub/Sub backend

**Best For:** GCP-native agent systems; Cloud Run agents.

### Azure Event Grid

**Overview:** Event routing service for Azure.

**Key Features:**
- Event domains
- Custom topics
- Event subscriptions
- Schema validation

**Best For:** Azure-native agent systems.

### Confluent Cloud

**Overview:** Managed Kafka platform.

**Key Features:**
- Fully managed Kafka
- Schema registry
- ksqlDB
- Connectors
- Governance

**Best For:** Production Kafka without operational overhead.

---

## 7. Edge and IoT Event Systems

### Eclipse Mosquitto

**Overview:** MQTT broker for IoT.

**Key Features:**
- MQTT 5.0
- Lightweight
- TLS support
- Bridge support

**Best For:** IoT agent communication; edge agents.

### EMQX

**Overview:** Scalable MQTT broker.

**Key Features:**
- High performance
- Rule engine
- WebHook
- MQTT bridge

**Best For:** Large-scale IoT agent systems.

### EdgeX Foundry

**Overview:** Edge computing framework.

**Key Features:**
- Service-oriented
- Protocol abstraction
- Device services
- Application services

**Best For:** Edge AI agents; IoT device integration.

---

## 8. Comparison Matrix

### Message Brokers

| Broker | Throughput | Latency | Durability | Complexity | Best For |
|--------|-----------|---------|------------|------------|----------|
| Kafka | Very High | Medium | Excellent | High | Large-scale streaming |
| Pulsar | High | Low-Medium | Excellent | High | Multi-tenant |
| RabbitMQ | Medium | Low | Good | Medium | Simple routing |
| NATS | Very High | Very Low | Good (JetStream) | Low | Edge/real-time |
| Redis Streams | High | Very Low | Good | Low | Fast processing |
| Redpanda | Very High | Low | Excellent | Medium | Kafka alternative |
| WarpStream | High | Medium | Excellent | Low | Cost-optimized |

### Agent Frameworks

| Framework | Language | Event-Driven | Persistence | Learning Curve | Production Ready |
|-----------|----------|-------------|-------------|----------------|------------------|
| Calfkit | Python | Yes | Kafka | Medium | Yes |
| Arvo | TypeScript | Yes | Pluggable | Low | Yes |
| Loom | Multi | Yes | Built-in | High | Experimental |
| Stream0 | Multi | Yes | HTTP | Low | Yes |

### Workflow Engines

| Engine | Durability | Saga Support | Language | Complexity | Best For |
|--------|-----------|-------------|----------|------------|----------|
| Temporal | Excellent | Yes | Multi | Medium | Long-running workflows |
| Airflow | Good | No | Python | Medium | Data pipelines |
| Prefect | Good | No | Python | Low | Dynamic workflows |
| Dagster | Good | No | Python | Medium | Typed pipelines |
| Step Functions | Excellent | Yes | JSON | Low | AWS-native |

### Event Stores

| Store | Event Sourcing | Projections | Performance | Complexity | Best For |
|-------|---------------|-------------|-------------|------------|----------|
| EventStoreDB | Native | Yes | Good | Medium | Pure event sourcing |
| Marten | Yes | Yes | Good | Medium | .NET systems |
| PostgreSQL | Custom | Custom | Good | Medium | General purpose |
| Redis | Custom | Custom | Very High | Low | Fast access |

---

## 9. Selection Guide

### Decision Tree

```
Start
  │
  ├── Do you need durable event streaming?
  │   ├── Yes, at scale → Kafka / Redpanda / WarpStream
  │   ├── Yes, simple → RabbitMQ / Redis Streams
  │   └── No, fire-and-forget → NATS / HTTP webhooks
  │
  ├── Do you need agent-specific features?
  │   ├── Yes, production → Calfkit / Arvo
  │   ├── Yes, experimental → Loom
  │   └── No, generic → Use message broker directly
  │
  ├── Do you need workflow orchestration?
  │   ├── Yes, long-running → Temporal
  │   ├── Yes, data pipelines → Airflow / Dagster
  │   ├── Yes, dynamic → Prefect
  │   └── No, simple event chains → Custom
  │
  └── What's your deployment environment?
      ├── AWS → EventBridge + Lambda
      ├── GCP → Eventarc + Cloud Run
      ├── Azure → Event Grid
      ├── Kubernetes → Kafka + custom operators
      └── Edge → NATS / MQTT
```

### By Scale

| Scale | Recommended Stack |
|-------|------------------|
| Prototype | Redis Streams + Python |
| Small (100 events/sec) | NATS + lightweight agents |
| Medium (1K events/sec) | RabbitMQ / Redis Streams + Temporal |
| Large (10K+ events/sec) | Kafka + Kubernetes + Temporal |
| Very Large (100K+ events/sec) | Kafka (multi-cluster) + custom orchestration |

### By Team Size

| Team | Recommended Approach |
|------|---------------------|
| Solo developer | Stream0 + simple event bus |
| Small team (2-5) | Arvo/Calfkit + managed Kafka |
| Medium team (5-15) | Full Kafka + Temporal + OpenTelemetry |
| Large team (15+) | Custom event mesh + distributed systems |

---

## 10. Getting Started Stacks

### Stack 1: Quick Prototype

**Goal:** Get an event-driven agent system running in hours.

```yaml
# docker-compose.yaml
services:
  redis:
    image: redis:7-alpine
  agent:
    build: .
    environment:
      REDIS_URL: redis://redis:6379
    depends_on:
      - redis
```

**Tools:**
- Redis Streams for events
- Python asyncio for agents
- In-memory state

### Stack 2: Production-Ready

**Goal:** Reliable, observable agent system for production.

```yaml
# docker-compose.yaml
services:
  kafka:
    image: confluentinc/cp-kafka:latest
  zookeeper:
    image: confluentinc/cp-zookeeper:latest
  postgres:
    image: postgres:16-alpine
  temporal:
    image: temporalio/auto-setup:latest
  otel-collector:
    image: otel/opentelemetry-collector:latest
  grafana:
    image: grafana/grafana:latest
  research-agent:
    build: ./agents/research
    depends_on:
      - kafka
      - postgres
      - temporal
```

**Tools:**
- Kafka for event streaming
- Temporal for workflows
- PostgreSQL for event store
- OpenTelemetry + Grafana for observability

### Stack 3: Enterprise

**Goal:** Multi-tenant, geo-distributed agent system.

**Tools:**
- Confluent Cloud (managed Kafka)
- Temporal Cloud (managed workflows)
- EventStoreDB (event sourcing)
- OpenTelemetry + Datadog (observability)
- Kubernetes (deployment)
- Istio (service mesh)

---

## Summary

| Category | Top Choices | When to Use |
|----------|-------------|-------------|
| Message Broker | Kafka, NATS, Redis Streams | Depends on scale and durability needs |
| Agent Framework | Calfkit, Arvo | Production agent systems |
| Workflow Engine | Temporal, Prefect | Complex multi-step workflows |
| Event Store | EventStoreDB, PostgreSQL | Event sourcing needs |
| Observability | OpenTelemetry, Langfuse | LLM agent monitoring |
| Cloud Events | EventBridge, Eventarc | Cloud-native deployments |

---

*This document is part of the AI Knowledge Library — a comprehensive reference for AI practitioners and researchers.*

---
**See also:**
- [07 — AI Coding Agent Sustainability: Managing Hidden Costs, Code Quality Decay, and Developer Burnout](33-AI-Native-Software-Development/07-AI-Coding-Agent-Sustainability.md)
- [48 — MCP Cloud Infrastructure & Agent-as-a-Service: The Production Layer of Agentic AI](48-MCP-Cloud-Infrastructure-Agent-as-a-Service/01-Overview.md)
- [AI Agent Commerce & Agent-to-Agent (A2A) Payments: The 2026 Frontier](28-AI-Agent-Commerce-and-A2A-Payments/01-Overview.md)
