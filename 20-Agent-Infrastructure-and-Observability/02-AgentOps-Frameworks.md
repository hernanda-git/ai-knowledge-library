# 02 — AgentOps Frameworks: Platforms, Comparison, and Architecture Decisions

## 1. Introduction to AgentOps Platforms

The AgentOps ecosystem has rapidly matured from a handful of startups to a diverse landscape of platforms offering overlapping but distinct capabilities. Choosing the right platform — or combination of platforms — is one of the most consequential infrastructure decisions an agent-building team will make.

### 1.1 Platform Categories

AgentOps platforms fall into several categories:

1. **Full-stack AgentOps Platforms** (LangSmith, LangFuse): Built specifically for LLM-powered applications and agents, offering end-to-end tracing, evaluation, datasets, prompts, and monitoring
2. **ML Observability Platforms** (Weights & Biases, Arize AI, WhyLabs): Originally focused on traditional ML model monitoring, now extending into LLM and agent observability
3. **LLM-specific Monitoring** (Helicone, Agenta): Lightweight solutions focused on LLM call monitoring with agent extensions
4. **General Observability Platforms with LLM support** (Datadog, Grafana, New Relic): Broad infrastructure monitoring with emerging LLM/agent instrumentation
5. **Open Source Agent Observability Kits** (OpenTelemetry, Traceloop, OpenLLMetry): Instrumentation libraries that export to any OpenTelemetry-compatible backend

### 1.2 Evaluation Criteria

When evaluating AgentOps platforms, consider these dimensions:

| Dimension | What to Evaluate |
|-----------|-----------------|
| Trace Fidelity | Does it capture every step of the agent loop? Tool calls? LLM calls? Thinking/reasoning steps? |
| Evaluation Integration | Can you run evaluations against production traces? Dataset management? LLM-as-judge? |
| Cost Tracking | Per-trace, per-model, per-user token accounting? Cost attribution? Budget alerts? |
| Prompt Management | Version control for prompts? A/B testing? Deployment of prompt variants? |
| Scalability | How many traces/day? Search performance? Retention policies? |
| Data Residency | Self-hosted option? SOC 2? GDPR? Data processing location? |
| Framework Support | LangChain, LangGraph, LlamaIndex, CrewAI, AutoGen, custom agents? |
| API & Extensibility | Can you export data to your own systems? Custom dashboards? Webhook integration? |
| Pricing | Per-trace, per-seat, per-organization? Free tier? Enterprise pricing? |

## 2. Detailed Platform Analysis

### 2.1 LangSmith

**Overview**: LangSmith is LangChain's official observability platform, deeply integrated with LangChain, LangGraph, and LangServe. It is the most popular AgentOps platform for teams already using the LangChain ecosystem.

**Key Features**:
- **Tracing**: Automatic instrumentation for all LangChain/LangGraph runs. Captures LLM calls, tool invocations, retriever operations, and chain/agent steps with full input/output logging
- **Evaluation**: Built-in evaluators (correctness, relevance, hallucination detection), custom evaluators, dataset management, test runs with comparison views
- **Hub**: Prompt versioning, sharing, and deployment for prompts and LLM configurations
- **Datasets**: Versioned datasets for evaluation, with support for creating datasets from production traces
- **Monitoring**: Real-time dashboards for latency, cost, error rates, and custom metrics with configurable alerts
- **Playground**: Interactive prompt testing and model comparison
- **Feedback**: Human feedback collection on traces (thumbs up/down, ratings, comments)

**Architecture**:
LangSmith uses a push-based telemetry model. The LangSmith SDK (part of langsmith package) collects traces locally, batches them, and sends them to LangSmith's cloud API. Self-hosted options exist via LangSmith Self-Hosted (enterprise).

**Code Example — Instrumenting with LangSmith**:
```python
import os
from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls_..."
os.environ["LANGCHAIN_PROJECT"] = "my-agent-project"

client = Client()

@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for information."""
    # Tool implementation
    return f"Results for: {query}"

llm = ChatOpenAI(model="gpt-4o")
agent = create_react_agent(llm, [search_knowledge_base])
agent_executor = AgentExecutor(agent=agent, tools=[search_knowledge_base])

# All runs are automatically traced via environment variables
result = agent_executor.invoke({"input": "What is the capital of France?"})
```

**Pricing**: Free tier (up to 5K traces/month), Developer ($59/mo), Team ($149/mo), Enterprise (custom). Self-hosted starts at ~$25K/year.

**Pros**: Deepest LangChain integration, rich evaluation framework, prompt hub, strong dataset management, active development
**Cons**: LangChain-centric (less useful for non-LangChain agents), can be expensive at scale, cloud dependency (unless self-hosted)

### 2.2 LangFuse

**Overview**: LangFuse is an open-source observability and evaluation platform for LLM applications. It started as a lightweight tracing tool and has grown into a full-featured AgentOps platform with a strong open-source community.

**Key Features**:
- **Tracing**: OpenTelemetry-compatible tracing with Python/JS SDKs. Captures LLM calls, tool usage, agent steps with hierarchical span trees
- **Evaluation**: LLM-as-judge evaluators, dataset management, manual scoring, online evaluation of production traces
- **Prompt Management**: Version-controlled prompt templates with variables, deployment environments (dev/staging/prod)
- **Datasets**: Curated datasets for evaluation, created from traces or uploaded
- **Cost Tracking**: Token usage and cost per model per trace, configurable model pricing
- **Monitoring**: Dashboards for cost, latency, quality scores, user feedback
- **Sessions**: Session-level grouping of traces across multiple interactions
- **Self-Hosted**: Fully open-source, self-hostable with Docker Compose

**Architecture**:
LangFuse uses an OpenTelemetry-based ingestion pipeline. Traces can be sent via:
1. LangFuse Python/JS SDK (direct HTTP)
2. OpenTelemetry SDK (OTLP protocol)
3. LangChain/LlamaIndex callbacks
4. Custom instrumentation via the LangFuse API

**Code Example — LangFuse with LangChain**:
```python
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool

langfuse = Langfuse(
    public_key="pk-...",
    secret_key="sk-...",
    host="https://cloud.langfuse.com"  # or self-hosted URL
)

# Use as LangChain callback
langfuse_handler = CallbackHandler(
    session_id="user-session-123",
    user_id="user-456",
    tags=["production", "v2"],
    metadata={"environment": "production"}
)

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 72°F in {city}"

llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_tool_calling_agent(llm, [get_weather])
agent_executor = AgentExecutor(agent=agent, tools=[get_weather])

result = agent_executor.invoke(
    {"input": "What's the weather in Tokyo?"},
    config={"callbacks": [langfuse_handler]}
)
```

**Pricing**: Open-source (self-hosted), Cloud: Free (50K observations/month), Team ($59/mo), Enterprise (custom). No per-trace pricing — primarily feature-based.

**Pros**: Open-source core, self-hostable, OpenTelemetry-native, rich evaluation, active community, generous free tier
**Cons**: Smaller ecosystem than LangSmith, fewer pre-built evaluators, cloud UI can be slower at scale

### 2.3 Weights & Biases (W&B)

**Overview**: W&B is a mature MLOps platform that has expanded into LLM observability with W&B Prompts. It is particularly strong for teams that already use W&B for model training.

**Key Features**:
- **Trace Tables**: Table-based trace visualization for LLM calls, tool usage, and agent steps
- **Chain Viewer**: DAG visualization of agent execution paths
- **Eval Tables**: Side-by-side comparison of agent outputs across different configurations
- **Model Registry**: Version control for models, which can include agent definitions
- **Datasets**: Versioned datasets for evaluation and fine-tuning
- **Artifacts**: Store agent traces, evaluation results, and datasets as versioned artifacts
- **Reports**: Collaborative reports with embedded trace tables and visualizations

**Code Example — W&B with Custom Agent**:
```python
import wandb
from wandb.integration.langchain import WandbTracer
from langchain.agents import create_react_agent, AgentExecutor

# Initialize W&B run
wandb.init(project="my-agent-project", config={"agent_type": "react"})

@tool
def database_query(sql: str) -> str:
    """Execute SQL query."""
    return f"Query results for: {sql}"

llm = ChatOpenAI(model="gpt-4o")
agent = create_react_agent(llm, [database_query])
agent_executor = AgentExecutor(agent=agent, tools=[database_query])

# Traces will be logged to W&B
with WandbTracer() as tracer:
    result = agent_executor.invoke({"input": "How many users signed up last week?"})
```

**Pricing**: Free (Teams up to 5), Premium ($150/user/mo), Enterprise (custom). W&B Prompts included in all tiers.

**Pros**: Excellent for ML teams already using W&B, rich artifact system, strong collaboration features, mature platform
**Cons**: Not agent-specialized (LLM tracing is an add-on), more expensive for large teams, less agent-specific evaluation

### 2.4 Arize AI

**Overview**: Arize AI is an ML observability platform with strong LLM and agent monitoring capabilities. It focuses on production monitoring, drift detection, and troubleshooting.

**Key Features**:
- **LLM Tracing**: OpenTelemetry-based tracing with span attributes for prompts, completions, token counts, tool calls
- **Embedding Monitoring**: Vector drift monitoring for LLM embeddings (critical for RAG agents)
- **Performance Monitoring**: Latency, throughput, error rate dashboards with configurable alerts
- **Drift Detection**: Automated detection of model performance degradation over time
- **Correlation Analysis**: Root cause analysis tools linking performance changes to data drift
- **Validation**: Schema validation for agent inputs and outputs
- **Integrations**: LangChain, LlamaIndex, Hugging Face, custom model serving

**Code Example — Arize with OpenTelemetry**:
```python
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

# Configure OpenTelemetry to send to Arize
provider = TracerProvider()
exporter = OTLPSpanExporter(endpoint="https://api.arize.com/v1/traces")
provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(provider)

# Auto-instrument LangChain
LangChainInstrumentor().instrument()

# Now all agent runs are traced to Arize
```

**Pricing**: Free (up to 10M inference points/month), Pro ($1,500/mo), Enterprise (custom).

**Pros**: Strong drift detection, embedding monitoring is unique, OpenTelemetry-native, generous free tier
**Cons**: Less agent-specific than LangSmith/LangFuse, fewer evaluation tools, steep learning curve

### 2.5 WhyLabs

**Overview**: WhyLabs is an AI observability platform with a focus on data quality and model monitoring. It acquired LangKit for LLM security and observability.

**Key Features**:
- **LLM Monitoring**: Token counts, prompt/response length, sentiment, toxicity detection via LangKit
- **Data Quality Monitoring**: Schema validation, missing values, type checking for agent inputs/outputs
- **Drift Monitoring**: Statistical drift detection for distributions of agent behavior over time
- **Anomaly Detection**: Automated detection of unusual patterns in agent metrics
- **Security Scanning**: Prompt injection detection, PII detection, toxic content filtering
- **WhyLabs AI Control**: Guardrails for LLM outputs

**Code Example — WhyLabs with LangKit**:
```python
import whylogs as why
from langkit import llm_metrics
from langkit.llm_metrics import LLMMetrics

# Configure WhyLabs logging
why.init(session_type="llm-agent")

# Log agent interactions
metrics = LLMMetrics()

def log_agent_interaction(prompt, response, agent_id, user_id):
    """Log agent interaction to WhyLabs."""
    profile = why.log({
        "prompt": prompt,
        "response": response,
        "agent_id": agent_id,
        "user_id": user_id,
        "prompt_tokens": len(prompt.split()),
        "response_tokens": len(response.split()),
        "prompt_sentiment": metrics.sentiment(prompt),
        "response_toxicity": metrics.toxicity(response),
        "response_pii": metrics.pii_entities(response),
    })
    return profile

# In agent execution loop
log_agent_interaction(
    prompt="What is the capital of France?",
    response="The capital of France is Paris.",
    agent_id="travel-agent-v2",
    user_id="user-789"
)
```

**Pricing**: Free (up to 5 models), Team ($2,500/mo), Enterprise (custom).

**Pros**: Strong data quality focus, security scanning (prompt injection, PII), generous free tier, good for regulated industries
**Cons**: Less agent-tracing depth, no prompt management, fewer agent-specific evaluation tools

### 2.6 Helicone

**Overview**: Helicone is a lightweight, open-source LLM observability platform focused on cost tracking, caching, and rate limiting. It acts as a proxy between your application and LLM APIs.

**Key Features**:
- **LLM Proxy**: Transparent proxy for OpenAI, Anthropic, and other providers
- **Cost Tracking**: Per-request cost calculation, cost by model/user/agent, budget alerts
- **Caching**: Response caching to reduce costs and latency (exact match and semantic)
- **Rate Limiting**: Per-user, per-agent, and per-API-key rate limits
- **Logging**: Full request/response logging with search and filtering
- **Experiments**: A/B testing of prompts and model configurations
- **Key Management**: API key management with usage quotas per key

**Architecture**: Helicone operates as a reverse proxy. You route your LLM API calls through Helicone, which logs, caches, and rate-limits before forwarding to the provider.

**Code Example — Helicone Proxy**:
```python
import openai

# Configure OpenAI client to use Helicone proxy
openai.api_base = "https://oai.hconeai.com/v1"
openai.api_key = "sk-your-openai-key"

# Helicone adds custom headers for user/agent tracking
result = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
    headers={
        "Helicone-User-Id": "user-123",
        "Helicone-Agent-Id": "customer-support-agent-v2",
        "Helicone-Session-Id": "session-abc-456"
    }
)
```

**Pricing**: Free (15K requests/mo), Developer ($20/mo), Growth ($99/mo), Enterprise (custom).

**Pros**: Excellent cost tracking, built-in caching and rate limiting, lightweight and simple, open-source core
**Cons**: Only captures LLM calls (not full agent traces), proxy-based architecture adds latency, less evaluation support

### 2.7 Agenta

**Overview**: Agenta is an open-source LLM application platform that combines prompt management, evaluation, and observability. It positions itself as a collaborative platform for building LLM-powered apps.

**Key Features**:
- **Prompt Versioning**: UI-based prompt editing with version history and deployment
- **Evaluation**: Side-by-side comparison of model outputs, custom evaluation criteria, auto-evaluation
- **Human Annotation**: Manual labeling and scoring of agent outputs for quality monitoring
- **Observability**: Monitoring dashboards for cost, latency, and usage metrics
- **A/B Testing**: Run experiments comparing different prompts, models, or configurations
- **Dataset Management**: Create and manage evaluation datasets
- **CrewAI Integration**: Native support for multi-agent CrewAI traces

**Pricing**: Open-source (self-hosted), Cloud: Free (1K evaluations), Scale ($199/mo), Enterprise (custom).

**Pros**: Strong prompt management UI, good evaluation workflow, open-source, CrewAI support
**Cons**: Smaller community, less production-proven, fewer integrations than LangSmith

## 3. Feature Comparison Matrix

| Feature | LangSmith | LangFuse | W&B | Arize | WhyLabs | Helicone | Agenta |
|---------|-----------|----------|-----|-------|---------|----------|--------|
| Agent Tracing | ✅ Deep | ✅ Deep | ✅ Basic | ✅ Basic | ❌ | ❌ | ✅ Medium |
| LLM Call Tracing | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Tool Call Tracing | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Multi-Agent Tracing | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Evaluation Framework | ✅ Rich | ✅ Rich | ✅ Basic | ✅ Medium | ❌ | ❌ | ✅ Rich |
| LLM-as-Judge Eval | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Dataset Management | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |
| Prompt Versioning | ✅ Hub | ✅ | ✅ Reg | ❌ | ❌ | ✅ Exp | ✅ |
| Cost Tracking | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ Best | ✅ Basic |
| Caching | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| Rate Limiting | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| CI/CD Integration | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| Self-Hosted | ✅ Ent | ✅ OSS | ❌ | ❌ | ❌ | ✅ | ✅ OSS |
| OpenTelemetry | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Security Scanning | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| Drift Detection | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ |
| Prompt Injection Detection | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |

## 4. Architecture Decisions for AgentOps

### 4.1 Telemetry Data Flow Architecture

The architecture of an AgentOps system has several key components:

```
┌─────────────────────────────────────────────────────────┐
│                    Agent Runtime                          │
│  ┌─────────┐  ┌──────────┐  ┌────────┐  ┌──────────┐   │
│  │ LLM Call │  │ Tool Call│  │ Memory │  │ Reasoning│   │
│  │ Tracing  │  │ Tracing  │  │ Logging│  │  Tracing │   │
│  └────┬────┘  └────┬─────┘  └────┬───┘  └────┬─────┘   │
│       │             │             │            │          │
│  ┌────▼─────────────▼─────────────▼────────────▼──────┐  │
│  │              Instrumentation Layer                   │  │
│  │  (OpenTelemetry SDK / LangSmith SDK / Custom Client) │  │
│  └─────────────────────┬───────────────────────────────┘  │
└────────────────────────┼──────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Ingestion / Collection Layer                  │
│  ┌────────┐ ┌──────────┐ ┌─────────┐ ┌───────────────┐  │
│  │ OTLP   │ │ REST API │ │Batch    │ │ Stream        │  │
│  │Receiver│ │ Gateway  │ │Processor│ │ Processor     │  │
│  └────────┘ └──────────┘ └─────────┘ └───────────────┘  │
└─────────────────────┬──────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                 Storage Layer                              │
│  ┌──────────┐ ┌──────────┐ ┌────────┐ ┌──────────────┐ │
│  │Trace DB  │ │Blob Store│ │Metrics │ │ Vector DB    │ │
│  │(Postgres)│ │ (S3/GCS) │ │(TSDB)  │ │(Embeddings)  │ │
│  └──────────┘ └──────────┘ └────────┘ └──────────────┘ │
└─────────────────────┬──────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│               Consumption Layer                            │
│  ┌────────┐ ┌──────────┐ ┌───────┐ ┌────────┐ ┌──────┐ │
│  │Dashbrd │ │ Alerting │ │ Search│ │ Export │ │ API  │ │
│  │ (Graf) │ │(PagerDut)│ │       │ │ (JSON) │ │      │ │
│  └────────┘ └──────────┘ └───────┘ └────────┘ └──────┘ │
└─────────────────────────────────────────────────────────┘
```

### 4.2 Key Architecture Decisions

#### Decision 1: Agent Framework Coupling

**Question**: How tightly should your AgentOps platform be coupled to your agent framework?

**Options**:
- **Tightly coupled** (LangSmith + LangChain): Simplest integration, richest feature set, but creates framework lock-in
- **Loosely coupled** (LangFuse + OTel): More portable across frameworks, but may lack framework-specific optimizations
- **Framework-agnostic** (OpenTelemetry + custom): Maximum flexibility, most engineering investment

**Recommendation**: Start tightly coupled with your chosen framework, but instrument via OpenTelemetry where possible to maintain portability. Most mature teams end up with a hybrid approach.

#### Decision 2: Self-Hosted vs. Cloud

**Question**: Should you self-host your AgentOps infrastructure or use a cloud platform?

**Self-Hosted Considerations**:
- Data sovereignty and compliance (GDPR, HIPAA, SOC 2)
- No per-trace costs at scale
- Full control over retention, sampling, and data processing
- Operational burden (maintenance, scaling, upgrades)
- Requires DevOps investment (Kubernetes, observability stack)

**Cloud Considerations**:
- Lower initial investment, faster time-to-value
- Automatic updates and scalability
- May restrict data processing location
- Per-trace pricing can become expensive at high volume
- SLA dependency on external provider

**Recommendation**: Start with cloud (low friction), plan for self-hosted or hybrid when traces exceed 1M/month or compliance requirements demand it.

#### Decision 3: OpenTelemetry Adoption

**Question**: Should you adopt OpenTelemetry as your instrumentation standard?

**Pros**:
- Industry standard for observability, broad ecosystem
- Vendor-agnostic (switch backends without re-instrumenting)
- Growing LLM/agent instrumentation (OpenLLMetry, Traceloop)
- Rich span attributes, context propagation, sampling

**Cons**:
- More configuration than framework-native SDKs
- Some agent-specific semantics still emerging in OTel conventions
- Requires an OTLP-compatible backend (or bridge)

**Recommendation**: Yes, adopt OpenTelemetry for custom instrumentation and as a transport layer. Use framework-native SDKs for their convenience, but bridge to OTel where possible.

#### Decision 4: Sampling Strategy

**Question**: How do you handle the volume of traces from thousands of agent sessions?

**Options**:
- **Head-based sampling**: Decide at trace start (all or nothing). Simple but may miss rare failures
- **Tail-based sampling**: Decide after trace completes based on attributes (error, latency, custom criteria). More complex but captures important traces
- **Dynamic sampling**: Adjust rate based on system load, trace priority, or time of day
- **No sampling (retain all)**: Maximum data, maximum cost. Often impractical at scale

**Recommendation**: Use head-based sampling for high-volume, low-importance traces (e.g., 1:100 for successful simple queries) and tail-based sampling to guarantee capture of errors, high-latency traces, and traces matching specific criteria (e.g., high cost, specific agent versions).

#### Decision 5: Multi-Layer Observability

**Question**: How many layers of observability do you need?

A mature agent observability strategy includes:

1. **Agent-level**: Full trace capture (LLM calls, tools, reasoning)
2. **Application-level**: HTTP requests, database queries, queue operations
3. **Infrastructure-level**: CPU, memory, network, GPU utilization
4. **Business-level**: Task completion rates, user satisfaction, ROI

**Recommendation**: Implement all four layers for production agents. Start with agent-level (critical for development) and add infrastructure-level (critical for reliability) concurrently. Add application-level and business-level as the system matures.

## 5. Multi-Platform Strategy

Most mature agent deployments use multiple tools rather than a single platform:

### Recommended Stack Composition

| Layer | Recommended Tool | Purpose |
|-------|-----------------|---------|
| Agent Tracing | LangSmith or LangFuse | Deep agent trace capture and evaluation |
| LLM Proxy/Caching | Helicone | Cost optimization, rate limiting, caching |
| Infrastructure Monitoring | Grafana + Prometheus | Server metrics, custom agent metrics |
| Log Aggregation | Grafana Loki or ELK | Agent log storage and search |
| Alerting | PagerDuty or Opsgenie | Incident notification |
| Security Scanning | WhyLabs / LangKit | Prompt injection, PII detection |
| Evaluation | LangSmith/LangFuse + custom | Automated agent evaluation |

### Integration Example

```python
"""
Multi-platform AgentOps integration example.
Combines LangFuse for tracing, Helicone for LLM proxy/caching,
and custom Prometheus metrics for monitoring.
"""

import time
from prometheus_client import Counter, Histogram, start_http_server
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool
import openai

# === Prometheus Metrics ===
AGENT_TASKS = Counter('agent_tasks_total', 'Total agent tasks', ['agent_id', 'status'])
AGENT_LATENCY = Histogram('agent_task_duration_seconds', 'Agent task duration',
                          ['agent_id'], buckets=[1, 5, 10, 30, 60, 120, 300])
AGENT_COST = Counter('agent_cost_usd_total', 'Total agent cost in USD', ['agent_id', 'model'])
AGENT_TOOL_CALLS = Counter('agent_tool_calls_total', 'Total tool calls', ['agent_id', 'tool'])

start_http_server(8000)  # Prometheus metrics endpoint

# === LangFuse ===
langfuse = Langfuse(public_key="pk-...", secret_key="sk-...")
langfuse_handler = CallbackHandler(tags=["production"])

# === Helicone (via proxy) ===
openai.api_base = "https://oai.hconeai.com/v1"

@tool
def web_search(query: str) -> str:
    AGENT_TOOL_CALLS.labels(agent_id="research-assistant", tool="web_search").inc()
    # Tool implementation
    return f"Search results for: {query}"

llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_tool_calling_agent(llm, [web_search])
agent_executor = AgentExecutor(agent=agent, tools=[web_search])

def run_agent(input_text: str, session_id: str, user_id: str):
    start_time = time.time()
    try:
        result = agent_executor.invoke(
            {"input": input_text},
            config={
                "callbacks": [langfuse_handler],
                "metadata": {"session_id": session_id, "user_id": user_id}
            }
        )
        AGENT_TASKS.labels(agent_id="research-assistant", status="success").inc()
        return result
    except Exception as e:
        AGENT_TASKS.labels(agent_id="research-assistant", status="error").inc()
        raise
    finally:
        duration = time.time() - start_time
        AGENT_LATENCY.labels(agent_id="research-assistant").observe(duration)
```

## 6. Migration Paths

### From No Observability to Basic Observability

1. Add LangFuse or LangSmith tracing to existing agents (1–2 days)
2. Set up basic dashboards for latency and cost (1 day)
3. Add critical alerts (error rate, cost spikes) (1 day)

### From Basic to Intermediate Observability

1. Implement OpenTelemetry-based instrumentation alongside framework SDK (1 week)
2. Add automated evaluation pipeline in CI/CD (1 week)
3. Implement cost tracking per user/session with alerts (2 days)
4. Set up log aggregation with structured logging (2 days)

### From Intermediate to Advanced Observability

1. Implement tail-based sampling for intelligent trace retention (1 week)
2. Build multi-agent trace correlation (1 week)
3. Implement MLOps-style drift detection for agent behavior (2 weeks)
4. Build agent registry with versioning and A/B testing (2 weeks)
5. Integrate security scanning (prompt injection, PII detection) (1 week)

## 7. Platform Selection Decision Tree

```
Do you use LangChain/LangGraph extensively?
├── YES → Use LangSmith as primary AgentOps platform
│   └── Need self-hosted? → LangSmith Self-Hosted or LangFuse
├── NO → Use LangFuse (open-source, framework-agnostic)
│
Evaluate cost management needs:
├── High LLM costs (>$1K/month) → Add Helicone as LLM proxy
├── Low LLM costs → Skip, rely on LangSmith/LangFuse cost tracking
│
Evaluate security needs:
├── PII/Prompt Injection concerns → Add WhyLabs/LangKit
├── No specific security requirements → Skip
│
Evaluate existing observability stack:
├── Using Grafana/Prometheus → Export agent metrics to existing stack
├── Using Datadog/New Relic → Check their LLM monitoring features or bridge via OTel
├── No existing stack → Start with Grafana + Loki for logs + metrics
│
Evaluate scaling needs:
├── <100K traces/month → Cloud tier of chosen platform
├── 100K-1M traces/month → Cloud with aggressive sampling
├── >1M traces/month → Evaluate self-hosted options and sampling
```

## 8. Conclusion

The AgentOps platform landscape is diverse and still maturing. No single platform covers all needs perfectly. The most effective approach is:

1. **Start simple** with a platform that matches your agent framework (LangSmith for LangChain, LangFuse for generic)
2. **Add layers as needed** — proxy caching, security scanning, custom metrics, log aggregation
3. **Plan for OpenTelemetry** — even if you don't use it today, instrument in a way that allows future migration
4. **Budget for observability costs** — expect 2–5% of total LLM spend on observability infrastructure
5. **Re-evaluate quarterly** — the AgentOps market is evolving rapidly; what's best today may not be best in six months

---

*Next: [03-Agent-Tracing-and-Observability.md](03-Agent-Tracing-and-Observability.md) — Deep dive into distributed tracing for agent systems with OpenTelemetry instrumentation.*
