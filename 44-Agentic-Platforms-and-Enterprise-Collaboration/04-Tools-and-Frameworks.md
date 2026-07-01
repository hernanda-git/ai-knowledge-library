# Tools and Frameworks: AI Agents in Enterprise Collaboration Platforms

> Comprehensive guide to the tools, frameworks, SDKs, and platforms available for building, deploying, and managing AI agents in enterprise collaboration environments.

---

## Table of Contents

1. [Agent Development Frameworks](#1-agent-development-frameworks)
2. [Platform SDKs and APIs](#2-platform-sdks-and-apis)
3. [LLM Providers and Routers](#3-llm-providers-and-routers)
4. [Memory and State Management](#4-memory-and-state-management)
5. [Observability and Monitoring](#5-observability-and-monitoring)
6. [Security and Compliance Tools](#6-security-and-compliance-tools)
7. [Testing and Evaluation](#7-testing-and-evaluation)
8. [Deployment and Infrastructure](#8-deployment-and-infrastructure)
9. [Comparison Matrix](#9-comparison-matrix)

---

## 1. Agent Development Frameworks

### 1.1 LangChain / LangGraph

LangChain is the most widely used framework for building LLM-powered agents:

**Key Features:**
- Modular chain composition
- Built-in tool integrations
- Memory management
- LangGraph for stateful agent workflows

**LangGraph for Collaboration Agents:**

```python
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

# Define agent state
class CollaborationState(MessagesState):
    channel_id: str
    user_id: str
    conversation_history: list

# Define tools
tools = [
    search_workspace,
    send_message,
    create_task,
    summarize_thread,
]

# Build agent graph
graph = StateGraph(CollaborationState)
graph.add_node("agent", create_react_agent(llm, tools))
graph.add_node("tools", ToolNode(tools))
graph.add_edge("agent", "tools")
graph.add_edge("tools", "agent")
graph.set_entry_point("agent")

app = graph.compile()
```

**Pros:** Large ecosystem, active community, extensive documentation
**Cons:** Can be complex, rapid API changes, abstraction overhead

### 1.2 CrewAI

CrewAI focuses on multi-agent collaboration:

```python
from crewai import Agent, Task, Crew

# Define agents
researcher = Agent(
    role="Research Analyst",
    goal="Find relevant information across the workspace",
    backstory="Expert at searching and synthesizing information",
    tools=[search_workspace, search_docs],
    llm=llm
)

communicator = Agent(
    role="Communications Specialist",
    goal="Draft clear and concise messages",
    backstory="Expert at workplace communication",
    tools=[send_message, create_thread],
    llm=llm
)

# Define tasks
research_task = Task(
    description="Research the latest updates on project X",
    agent=researcher
)

communication_task = Task(
    description="Draft a status update for the team",
    agent=communicator
)

# Create crew
crew = Crew(
    agents=[researcher, communicator],
    tasks=[research_task, communication_task],
    verbose=True
)

result = crew.kickoff()
```

**Pros:** Intuitive multi-agent design, good for complex workflows
**Cons:** Less flexible than raw LangChain, limited memory options

### 1.3 AutoGen (Microsoft)

AutoGen enables multi-agent conversations:

```python
from autogen import AssistantAgent, UserProxyAgent

# Create assistant agent
assistant = AssistantAgent(
    name="workspace_assistant",
    llm_config={
        "config_list": [{"model": "gpt-4", "api_key": api_key}],
        "temperature": 0.7,
    },
    system_message="You are a helpful workspace assistant."
)

# Create user proxy (represents the platform)
user_proxy = UserProxyAgent(
    name="platform_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config=False
)

# Start conversation
user_proxy.initiate_chat(
    assistant,
    message="Summarize the latest messages in #engineering channel"
)
```

**Pros:** Strong multi-agent support, Microsoft backing, research-oriented
**Cons:** Steeper learning curve, less production-ready

### 1.4 Semantic Kernel (Microsoft)

Semantic Kernel is designed for enterprise integration:

```python
import semantic_kernel as sk

kernel = sk.Kernel()

# Add AI services
kernel.add_chat_service("gpt4", AzureChatCompletion("gpt-4", endpoint, key))

# Add plugins (tools)
kernel.import_plugin_from_object(WorkspacePlugin(), "workspace")
kernel.import_plugin_from_object(TaskPlugin(), "tasks")

# Create function
summarize = kernel.create_function_from_prompt(
    prompt="{{$user_message}}",
    function_name="summarize",
    description="Summarize the conversation"
)

# Execute
result = await kernel.invoke(summarize, user_message="What happened today?")
```

**Pros:** Enterprise-focused, strong Microsoft ecosystem integration
**Cons:** Primarily .NET focused, smaller community

### 1.5 Comparison: Agent Frameworks

| Framework | Language | Multi-Agent | Memory | Enterprise | Learning Curve |
|-----------|----------|-------------|--------|------------|----------------|
| LangChain/LangGraph | Python/JS | Yes | Built-in | Good | Medium |
| CrewAI | Python | Yes | Limited | Fair | Low |
| AutoGen | Python | Yes | Limited | Fair | High |
| Semantic Kernel | Python/.NET | Limited | Built-in | Excellent | Medium |
| Haystack | Python | Limited | Built-in | Good | Medium |

---

## 2. Platform SDKs and APIs

### 2.1 Slack SDKs

| SDK | Language | Use Case | Maturity |
|-----|----------|----------|----------|
| Bolt for Python | Python | Agent development | Production |
| Bolt for JavaScript | JS/TS | Agent development | Production |
| Bolt for Java | Java | Enterprise agents | Production |
| Bolt for Go | Go | High-performance agents | Beta |
| Slack SDK (Web API) | Any | Direct API calls | Production |

**Slack Bolt Example:**

```python
from slack_bolt import App
from slack_bolt.adapter.fastapi import SlackRequestHandler

app = App(token="xoxb-...", signing_secret="...")

@app.event("message")
def handle_message(event, say):
    if "AI" in event.get("text", ""):
        response = generate_response(event["text"])
        say(text=response, thread_ts=event["ts"])

handler = SlackRequestHandler(app)

# FastAPI integration
@app.post("/slack/events")
async def slack_events(request: Request):
    return await handler.handle(request)
```

### 2.2 Microsoft Teams SDKs

| SDK | Language | Use Case | Maturity |
|-----|----------|----------|----------|
| Bot Framework SDK | C#/JS/Python | Bot development | Production |
| Teams Toolkit | VS Code | Quick start | Production |
| Graph API | Any | Platform integration | Production |
| Adaptive Cards | Any | Rich responses | Production |

### 2.3 Notion SDK

| SDK | Language | Use Case | Maturity |
|-----|----------|----------|----------|
| notion-client | Python | API access | Production |
| @notionhq/client | JS/TS | API access | Production |
| Notion SDK (REST) | Any | Direct API | Production |

### 2.4 GitHub SDK

| SDK | Language | Use Case | Maturity |
|-----|----------|----------|----------|
| PyGithub | Python | GitHub API | Production |
| Octokit | JS/TS | GitHub API | Production |
| go-github | Go | GitHub API | Production |
| GitHub CLI | CLI | Automation | Production |

---

## 3. LLM Providers and Routers

### 3.1 LLM Provider Comparison

| Provider | Models | Context Window | Price (Input/Output per 1M tokens) | Best For |
|----------|--------|---------------|-----------------------------------|----------|
| OpenAI | GPT-4o, GPT-4o-mini, GPT-5 | 128K-1M | $2.50/$10 (GPT-4o) | General purpose |
| Anthropic | Claude 3.5, Claude 4 | 200K | $3/$15 (Claude 3.5 Sonnet) | Reasoning, code |
| Google | Gemini 1.5, Gemini 2.0 | 1M | $1.25/$5 (Gemini 1.5 Pro) | Multimodal, long context |
| Meta (via AWS) | Llama 3.1, Llama 4 | 128K | Varies | Self-hosted |
| Mistral | Mistral Large, Small | 128K | $2/$6 (Mistral Large) | European compliance |
| Cohere | Command R+ | 128K | $2.5/$10 | RAG, enterprise |

### 3.2 LLM Router Implementation

```python
class LLMRouter:
    def __init__(self, config):
        self.providers = {
            "openai": OpenAIProvider(config.openai),
            "anthropic": AnthropicProvider(config.anthropic),
            "google": GoogleProvider(config.google),
        }
        self.default_provider = config.default_provider
    
    async def route(self, query, context, requirements=None):
        """Route to optimal provider based on requirements."""
        
        if requirements:
            # Route based on specific requirements
            if requirements.get("long_context"):
                return await self.providers["google"].generate(query, context)
            elif requirements.get("code_heavy"):
                return await self.providers["anthropic"].generate(query, context)
            elif requirements.get("cost_optimized"):
                return await self.providers["openai"].generate_mini(query, context)
        
        # Default routing
        return await self.providers[self.default_provider].generate(query, context)
    
    async def fallback(self, query, context, failed_provider):
        """Fallback to another provider if primary fails."""
        for name, provider in self.providers.items():
            if name != failed_provider:
                try:
                    return await provider.generate(query, context)
                except Exception:
                    continue
        
        raise AllProvidersFailed()
```

### 3.3 Model Router for Cost Optimization

```python
class CostOptimizedRouter:
    COMPLEXITY_THRESHOLDS = {
        "simple": 0.3,    # Factual questions, simple lookups
        "moderate": 0.7,  # Analysis, summarization
        "complex": 1.0,   # Reasoning, code generation
    }
    
    MODEL_COSTS = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
        "claude-3-opus": {"input": 15.00, "output": 75.00},
    }
    
    def select_model(self, query_complexity, budget_per_query=0.01):
        """Select model based on complexity and budget."""
        for model in ["gpt-4o-mini", "gpt-4o", "claude-3.5-sonnet"]:
            cost = self.estimate_cost(model, estimated_tokens=500)
            if cost <= budget_per_query:
                return model
        
        return "gpt-4o-mini"  # Fallback to cheapest
```

---

## 4. Memory and State Management

### 4.1 Memory Framework Comparison

| Framework | Type | Features | Best For |
|-----------|------|----------|----------|
| Mem0 | Vector + Graph | Entity extraction, dedup | Enterprise memory |
| Zep | Temporal | Time-aware, fact extraction | Conversational agents |
| Letta (MemGPT) | OS-inspired | Virtual context, self-editing | Long-running agents |
| LangMem | LangChain native | Integration with LangChain | LangChain users |
| Qdrant | Vector DB | High-performance search | Custom memory |

### 4.2 Mem0 Integration

```python
from mem0 import Memory

class AgentMemory:
    def __init__(self, config):
        self.memory = Memory.from_config(config)
    
    async def store_interaction(self, user_id, message, response):
        """Store a conversation interaction."""
        await self.memory.add(
            message,
            user_id=user_id,
            metadata={
                "type": "conversation",
                "timestamp": datetime.now().isoformat()
            }
        )
    
    async def recall_context(self, user_id, query):
        """Recall relevant context for a query."""
        results = await self.memory.search(
            query,
            user_id=user_id,
            limit=10
        )
        return results
    
    async def get_user_facts(self, user_id):
        """Get all known facts about a user."""
        return await self.memory.get_all(user_id=user_id)
```

### 4.3 Zep Integration

```python
from zep import ZepClient

class ZepMemory:
    def __init__(self, api_key):
        self.client = ZepClient(api_key)
    
    async def add_message(self, session_id, role, content):
        """Add a message to session history."""
        await self.client.memory.add_message(
            session_id=session_id,
            message={"role": role, "content": content}
        )
    
    async def get_context(self, session_id, query):
        """Get relevant context using Zep's temporal memory."""
        memory = await self.client.memory.get(session_id)
        
        # Zep provides:
        # - Relevant facts extracted from conversation
        # - Temporal context (what happened when)
        # - User preferences
        return {
            "facts": memory.facts,
            "recent": memory.recent_messages,
            "summary": memory.summary
        }
```

### 4.4 Vector Store Comparison

| Store | Speed | Features | Self-Hosted | Best For |
|-------|-------|----------|-------------|----------|
| Qdrant | Fast | Filtering, payloads | Yes | Production |
| Pinecone | Fast | Managed, serverless | No | Quick setup |
| Weaviate | Medium | GraphQL, modules | Yes | Flexibility |
| Chroma | Medium | Simple, lightweight | Yes | Prototyping |
| pgvector | Medium | SQL integration | Yes | Existing Postgres |

---

## 5. Observability and Monitoring

### 5.1 Observability Stack

| Tool | Type | Features | Best For |
|------|------|----------|----------|
| LangSmith | Tracing | LLM-specific tracing | LangChain users |
| Langfuse | Tracing | Open-source LLM tracing | Self-hosted |
| Helicone | Proxy | LLM proxy with analytics | Quick setup |
| Arize Phoenix | Tracing | Open-source, evaluation | Research |
| Prometheus + Grafana | Metrics | Infrastructure metrics | Operations |
| Datadog APM | Full-stack | Enterprise APM | Large teams |

### 5.2 LangSmith Integration

```python
import langsmith

client = langsmith.Client()

# Trace agent execution
with client.trace(
    name="collaboration_agent",
    metadata={
        "channel": "C123",
        "user": "U456",
        "platform": "slack"
    }
) as trace:
    # LLM call
    response = llm.generate(
        messages,
        langsmith_extra={"run_id": trace.id}
    )
    
    # Tool call
    if response.tool_calls:
        tool_result = execute_tools(response.tool_calls)
    
    trace.set outputs={"response": response.text}
```

### 5.3 Langfuse (Open-Source)

```python
from langfuse import Langfuse

langfuse = Langfuse(
    public_key="pk-...",
    secret_key="sk-...",
    host="https://cloud.langfuse.com"
)

# Create trace
trace = langfuse.trace(
    name="agent_conversation",
    metadata={"channel": "C123", "user": "U456"}
)

# Log generation
generation = trace.generation(
    name="llm_call",
    model="gpt-4o",
    input=messages,
    output=response,
    usage={"input": 500, "output": 200}
)

# Log tool call
span = trace.span(
    name="tool_execution",
    input={"tool": "search_workspace", "query": "project status"},
    output={"results": 5}
)

langfuse.flush()
```

### 5.4 Custom Metrics Dashboard

```python
# Key metrics to track
METRICS = {
    # Performance
    "response_time_p50": Histogram("response_time_p50", "Median response time"),
    "response_time_p95": Histogram("response_time_p95", "95th percentile response time"),
    "response_time_p99": Histogram("response_time_p99", "99th percentile response time"),
    
    # Quality
    "user_satisfaction": Gauge("user_satisfaction", "User satisfaction score"),
    "response_accuracy": Gauge("response_accuracy", "Response accuracy rate"),
    
    # Usage
    "requests_per_minute": Counter("requests_per_minute", "Requests per minute"),
    "active_conversations": Gauge("active_conversations", "Active conversations"),
    "unique_users": Counter("unique_users", "Unique users served"),
    
    # Cost
    "total_tokens": Counter("total_tokens", "Total tokens consumed"),
    "cost_per_request": Histogram("cost_per_request", "Cost per request"),
    "daily_cost": Gauge("daily_cost", "Total daily cost"),
    
    # Errors
    "error_rate": Counter("error_rate", "Error count by type"),
    "tool_failure_rate": Counter("tool_failure_rate", "Tool execution failures"),
}
```

---

## 6. Security and Compliance Tools

### 6.1 Security Tool Stack

| Tool | Purpose | Features |
|------|---------|----------|
| Guardrails AI | Input/Output validation | LLM-specific guards |
| Rebuff | Prompt injection detection | Self-hardening |
| LLM Guard | Content filtering | PII, toxicity, topics |
| OWASP ZAP | Web security | API vulnerability scanning |
| Snyk | Dependency scanning | Code security |

### 6.2 Guardrails AI Integration

```python
from guardrails import Guard
from guardrails.validators import (
    RegexMatch,
    TwoWords,
    ReadingTime
)

# Define output validation
guard = Guard().use(
    RegexMatch(
        regex=r"^[A-Z][a-zA-Z\s,.'!?]+$",
        on_fail="rephrase"
    ),
    ReadingTime(
        max_tokens=500,
        on_fail="trim"
    )
)

# Validate agent response
validated_response = guard.validate(
    raw_llm_output=agent_response,
    metadata={"task": "meeting_summary"}
)
```

### 6.3 Compliance Checklist

| Requirement | Implementation | Verification |
|-------------|---------------|--------------|
| Data Encryption | AES-256 at rest, TLS 1.3 in transit | Audit logs |
| Access Control | RBAC + OAuth 2.0 | Permission reviews |
| Audit Logging | All agent actions logged | Log analysis |
| Data Retention | Configurable retention policies | Policy enforcement |
| Right to Deletion | User data deletion API | Deletion confirmation |
| PII Detection | Automated PII scanning | Scan reports |

---

## 7. Testing and Evaluation

### 7.1 Testing Tools

| Tool | Type | Features | Best For |
|------|------|----------|----------|
| pytest | Unit testing | Python standard | Component tests |
| Playwright | E2E testing | Browser automation | UI testing |
| Locust | Load testing | Distributed load | Performance |
| DeepEval | LLM evaluation | Quality metrics | Response quality |
| RAGAS | RAG evaluation | RAG-specific metrics | Knowledge retrieval |

### 7.2 Agent Evaluation Framework

```python
from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric
)

# Define metrics
relevancy = AnswerRelevancyMetric(threshold=0.7)
faithfulness = FaithfulnessMetric(threshold=0.8)

# Evaluate agent responses
test_cases = [
    {
        "input": "What's the status of project X?",
        "actual_output": agent_response,
        "expected_output": "Project X is on track...",
        "retrieval_context": retrieved_docs
    }
]

results = evaluate(
    test_cases=test_cases,
    metrics=[relevancy, faithfulness]
)

print(f"Relevancy: {results[0].score}")
print(f"Faithfulness: {results[1].score}")
```

### 7.3 Load Testing

```python
from locust import HttpUser, task, between

class AgentUser(HttpUser):
    wait_time = between(1, 3)
    
    @task(3)
    def send_message(self):
        self.client.post("/api/agent/message", json={
            "channel": "C123",
            "user": "U456",
            "message": "What's the status of project X?"
        })
    
    @task(1)
    def search_workspace(self):
        self.client.post("/api/agent/search", json={
            "query": "latest updates",
            "scope": "all"
        })
    
    @task(2)
    def create_task(self):
        self.client.post("/api/agent/task", json={
            "title": "Test task",
            "assignee": "U456",
            "priority": "medium"
        })
```

---

## 8. Deployment and Infrastructure

### 8.1 Deployment Platforms

| Platform | Type | Features | Best For |
|----------|------|----------|----------|
| AWS ECS/EKS | Container | Full control, AWS ecosystem | Enterprise |
| Google Cloud Run | Serverless | Auto-scaling, simple | Startups |
| Azure Container Apps | Serverless | Azure integration | Microsoft shops |
| Railway | PaaS | Simple deployment | Prototyping |
| Fly.io | Global | Edge deployment | Global teams |

### 8.2 Infrastructure as Code

```terraform
# AWS ECS deployment
resource "aws_ecs_service" "agent" {
  name            = "collaboration-agent"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.agent.arn
  desired_count   = 3
  launch_type     = "FARGATE"

  network_configuration {
    subnets         = var.private_subnet_ids
    security_groups = [aws_security_group.agent.id]
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.agent.arn
    container_name   = "agent"
    container_port   = 8080
  }
}

resource "aws_ecs_task_definition" "agent" {
  family                   = "collaboration-agent"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 512
  memory                   = 1024

  container_definitions = jsonencode([{
    name  = "agent"
    image = "${var.ecr_repository_url}:latest"
    
    portMappings = [{
      containerPort = 8080
      hostPort      = 8080
    }]

    environment = [
      { name = "DATABASE_URL", value = var.database_url },
      { name = "REDIS_URL", value = var.redis_url },
    ]

    secrets = [
      { name = "LLM_API_KEY", valueFrom = aws_ssm_parameter.llm_key.arn }
    ]
  }])
}
```

### 8.3 CI/CD Pipeline

```yaml
# GitHub Actions workflow
name: Deploy Agent
on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run tests
        run: |
          pip install -r requirements.txt
          pytest tests/ -v
      
      - name: Run security scan
        uses: snyk/actions/python@master

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Build and push Docker image
        run: |
          docker build -t $ECR_REGISTRY/$IMAGE_NAME:$GITHUB_SHA .
          docker push $ECR_REGISTRY/$IMAGE_NAME:$GITHUB_SHA
      
      - name: Deploy to ECS
        run: |
          aws ecs update-service \
            --cluster collaboration-agent \
            --service agent \
            --force-new-deployment
```

---

## 9. Comparison Matrix

### 9.1 Framework Selection Guide

| Use Case | Recommended Framework | Why |
|----------|----------------------|-----|
| Simple Slack bot | Bolt for Python | Direct platform integration |
| Complex multi-agent | CrewAI or AutoGen | Multi-agent orchestration |
| Enterprise Microsoft | Semantic Kernel | Azure/M365 integration |
| Custom architecture | LangChain/LangGraph | Maximum flexibility |
| RAG-focused agent | Haystack | RAG optimization |

### 9.2 Platform Selection Guide

| Primary Platform | SDK | Agent Framework | Memory |
|-----------------|-----|-----------------|--------|
| Slack | Bolt | LangChain | Mem0 |
| Microsoft Teams | Bot Framework | Semantic Kernel | Zep |
| Notion | notion-client | LangChain | Qdrant |
| Google Workspace | Google AI | LangChain | Pinecone |
| Multiple | ACP protocol | LangGraph | Mem0 |

### 9.3 Cost Estimation

| Component | Low Volume | Medium Volume | High Volume |
|-----------|-----------|---------------|-------------|
| LLM API | $50/mo | $500/mo | $5,000/mo |
| Infrastructure | $100/mo | $500/mo | $2,000/mo |
| Memory/DB | $50/mo | $200/mo | $1,000/mo |
| Monitoring | $0 (OSS) | $100/mo | $500/mo |
| **Total** | **$200/mo** | **$1,300/mo** | **$8,500/mo** |

---

## Cross-References

| Document | Relevance |
|----------|-----------|
| [03-Agents/03-Agentic-Frameworks](../../03-Agents/03-Agentic-Frameworks.md) | Framework overview |
| [15-Community/10-Tools-Ecosystem](../../15-Community-Resources-Templates/10-Tools-Ecosystem.md) | Tool ecosystem |
| [32-Agent-Memory/02-Frameworks](../../32-Agent-Memory-Systems/02-Frameworks-Mem0-Zep-Letta-MemGPT-LangMem.md) | Memory frameworks |
| [20-Agent-Observability/02-AgentOps](../../20-Agent-Infrastructure-and-Observability/02-AgentOps-Frameworks.md) | Observability tools |

---

*Last updated: July 2026*
