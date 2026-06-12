# MCP & ACP Protocols

> **Last Updated:** June 2026  
> **Category:** Top Demand — Current Market Snapshot  
> **Cross-References:** 02-AI-Agent-Development.md, 06-RAG-Retrieval-Systems.md, 09-AI-Automation.md

---

## Table of Contents

1. [Introduction & Market Context](#1-introduction--market-context)
2. [Model Context Protocol (MCP)](#2-model-context-protocol-mcp)
   - 2.1 Architecture Overview
   - 2.2 Transport Layer (stdio, SSE, HTTP)
   - 2.3 Tool Discovery & Registration
   - 2.4 Resource Management
   - 2.5 Prompt Templates
   - 2.6 Security Boundaries & Sandboxing
   - 2.7 MCP Servers & Clients
3. [Agent Communication Protocol (ACP)](#3-agent-communication-protocol-acp)
   - 3.1 Architecture & Message Format
   - 3.2 Inter-Agent Communication
   - 3.3 Delegation Patterns
   - 3.4 Handoff Protocols
   - 3.5 Coordination & Consensus
4. [Implementation Guide](#4-implementation-guide)
   - 4.1 Building an MCP Server
   - 4.2 Building an MCP Client
   - 4.3 ACP Agent Registration
   - 4.4 Multi-Agent Conversation with ACP
5. [Adoption & Ecosystem](#5-adoption--ecosystem)
6. [Tool Integration Matrix](#6-tool-integration-matrix)
7. [Security Considerations](#7-security-considerations)
8. [Benchmarks & Performance](#8-benchmarks--performance)
9. [Future Roadmap](#9-future-roadmap)

---

## 1. Introduction & Market Context

The Model Context Protocol (MCP) and Agent Communication Protocol (ACP) represent the two most significant standardization efforts in AI infrastructure as of June 2026. Together, they form the communication backbone for the modern AI agent ecosystem.

### Why Protocols Matter Now

Before MCP/ACP, every agent framework had its own tool integration API. Developers wrote custom adapters for each tool × framework combination — a combinatorial explosion. MCP standardizes *tool exposure*; ACP standardizes *agent interaction*.

**Market impact:**
- MCP adopted by 80%+ of major LLM providers and agent frameworks
- ACP in active deployment across 35% of multi-agent systems
- Combined ecosystem reduces integration cost by ~60% (Anthropic/OpenAI joint report, March 2026)
- Both protocols are open-source with Apache 2.0 license

### Protocol Relationship

```
┌─────────────────────────────────────────────┐
│                  ACP Layer                    │
│  (Agent-to-Agent: delegation, handoff,       │
│   coordination, negotiation)                 │
├─────────────────────────────────────────────┤
│                  MCP Layer                    │
│  (Agent-to-Tool: discovery, invocation,      │
│   resource access, prompt management)        │
├─────────────────────────────────────────────┤
│           Transport Layer (stdio, SSE,       │
│           WebSocket, HTTP/2)                 │
├─────────────────────────────────────────────┤
│           LLM / Agent Frameworks             │
└─────────────────────────────────────────────┘
```

---

## 2. Model Context Protocol (MCP)

### 2.1 Architecture Overview

MCP follows a **client-server architecture** where AI hosts (clients) connect to MCP servers that expose tools, resources, and prompts.

**Core concepts:**

```
┌──────────┐    MCP Protocol     ┌───────────┐
│   MCP    │ ◄──────────────────►│    MCP    │
│  Client  │                     │   Server  │
│ (AI App) │                     │  (Tools)  │
└──────────┘                     └───────────┘
     │                                │
     ▼                                ▼
┌──────────┐                    ┌───────────┐
│   LLM    │                    │ Database, │
│  Model   │                    │  APIs,    │
│          │                    │ Filesystem│
└──────────┘                    └───────────┘
```

**Key capabilities:**
- **Tools** — Functions the server exposes for the LLM to call
- **Resources** — Data/assets the server provides (files, DB records, API results)
- **Prompts** — Pre-defined prompt templates the server offers
- **Transports** — stdio (local) and SSE (remote)

### 2.2 Transport Layer

MCP defines three primary transports in its June 2026 specification (v1.2):

#### 2.2.1 stdio Transport

Most secure and lowest latency. The MCP client spawns the server as a child process and communicates via stdin/stdout.

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"],
      "env": {
        "MCP_DEBUG": "false"
      }
    },
    "database": {
      "command": "python",
      "args": ["-m", "mcp_server_postgres", "--connection-string", "${DB_URL}"]
    }
  }
}
```

**Characteristics:**
- Lowest latency (sub-100μs overhead)
- No network exposure — ideally secure
- Lifecycle tied to client process
- Best for local agents and development

#### 2.2.2 SSE Transport (Server-Sent Events)

For remote MCP servers. Uses HTTP/1.1 SSE for server→client streaming, POST for client→server requests.

```
Client → POST /message (JSON-RPC request)
Server → SSE /events (JSON-RPC response/event stream)
```

**Example endpoint registration:**

```python
# FastMCP SSE server (June 2026)
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather Service", transport="sse")

@mcp.tool()
def get_weather(city: str, units: str = "celsius") -> str:
    """Get current weather for a city."""
    response = requests.get(f"https://api.weather.com/v1/{city}")
    return response.json()

mcp.run(host="0.0.0.0", port=8000)
```

**Characteristics:**
- HTTP-based, firewall-friendly
- Supports remote tool execution
- Higher latency (~5-50ms overhead)
- Requires authentication/authorization

#### 2.2.3 HTTP/2 & WebSocket Transports (2026+)

The latest MCP spec adds HTTP/2 and WebSocket transports for improved performance:

- **WebSocket** — Bidirectional streaming, ideal for real-time agent interactions
- **HTTP/2** — Multiplexed connections, reduced head-of-line blocking

### 2.3 Tool Discovery & Registration

#### JSON-RPC Method Definitions

MCP uses JSON-RPC 2.0 as its message protocol. Tools are discovered and invoked via standard methods:

**Tool discovery (`tools/list`):**

```json
// Request
{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}

// Response
{
  "jsonrpc": "2.0", "id": 1,
  "result": {
    "tools": [
      {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "inputSchema": {
          "type": "object",
          "properties": {
            "city": {
              "type": "string",
              "description": "City name"
            },
            "units": {
              "type": "string",
              "enum": ["celsius", "fahrenheit"],
              "default": "celsius"
            }
          },
          "required": ["city"]
        }
      }
    ]
  }
}
```

**Tool invocation (`tools/call`):**

```json
// Request
{
  "jsonrpc": "2.0", "id": 2,
  "method": "tools/call",
  "params": {
    "name": "get_weather",
    "arguments": {
      "city": "Tokyo",
      "units": "celsius"
    }
  }
}

// Response
{
  "jsonrpc": "2.0", "id": 2,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Current weather in Tokyo: 22°C, partly cloudy"
      }
    ],
    "isError": false
  }
}
```

### 2.4 Resource Management

Resources expose data/assets to the agent. They mirror RESTful resource patterns but are accessed through MCP's protocol.

**Resource URI scheme:**
```
mcp://server-name/resource-type/path
```

**Resource types (standardized):**
- `mcp://server/documents/*` — File-based resources
- `mcp://server/database/*` — Database records
- `mcp://server/memory/*` — Agent memory store
- `mcp://server/prompts/*` — Prompt templates

**Resource example:**

```json
// resources/list request
{"jsonrpc": "2.0", "id": 3, "method": "resources/list", "params": {}}

// Response
{
  "jsonrpc": "2.0", "id": 3,
  "result": {
    "resources": [
      {
        "uri": "mcp://docs/sales/q3-report.pdf",
        "name": "Q3 Sales Report",
        "mimeType": "application/pdf",
        "description": "Sales figures for Q3 2025"
      }
    ]
  }
}
```

**Resource subscriptions:**
MCP v1.2 adds resource subscriptions — clients can subscribe to changes:

```json
{
  "method": "resources/subscribe",
  "params": { "uri": "mcp://database/live-sales" }
}
// Server sends notifications on change:
{
  "method": "notifications/resources/updated",
  "params": { "uri": "mcp://database/live-sales" }
}
```

### 2.5 Prompt Templates

MCP servers can expose reusable prompt templates that LLMs can leverage:

```json
{
  "prompts": [
    {
      "name": "analyze_code",
      "description": "Analyze code for bugs and improvements",
      "arguments": [
        {
          "name": "language",
          "type": "string",
          "required": true,
          "enum": ["python", "javascript", "rust"]
        },
        {
          "name": "code",
          "type": "string",
          "required": true
        }
      ]
    }
  ]
}
```

**Prompt execution:**

```json
{
  "method": "prompts/get",
  "params": {
    "name": "analyze_code",
    "arguments": {
      "language": "python",
      "code": "def foo():\n    pass"
    }
  }
}
// Returns rendered prompt with context
```

### 2.6 Security Boundaries & Sandboxing

MCP security is a layered model:

**Layer 1: Transport Security**
- stdio: secure by isolation (no network exposure)
- SSE/WebSocket: requires TLS 1.3+ for remote connections
- Mutual TLS (mTLS) supported in v1.2

**Layer 2: Authentication**
- API keys in request headers
- OAuth 2.0 / OIDC integration
- Bearer token validation

**Layer 3: Authorization**
- Tool-level permissions (allow/deny lists)
- Resource scope restrictions
- Rate limiting per client

**Layer 4: Sandboxing**
- MCP servers should run in isolated environments
- Docker containers with read-only root FS
- Capability dropping (no network unless needed)
- Resource limits (CPU, memory, IO)

**Example sandbox configuration:**

```yaml
mcp_server:
  sandbox:
    type: docker
    image: mcp-server-python:3.12
    read_only_root: true
    capabilities:
      dropped: [ALL]
      added: []
    network: "none"
    memory_limit: "512M"
    cpu_limit: "1.0"
    tmpfs: {"/tmp": "size=100M"}
```

### 2.7 MCP Servers & Clients

#### Popular MCP Servers (June 2026)

| Server | Description | Category |
|--------|-------------|----------|
| `@mcp/filesystem` | File read/write/search | Core |
| `@mcp/github` | GitHub API (repos, PRs, issues) | DevOps |
| `@mcp/postgres` | PostgreSQL database access | Database |
| `@mcp/sqlite` | SQLite database | Database |
| `@mcp/brave-search` | Web search via Brave API | Search |
| `@mcp/playwright` | Browser automation | Web |
| `@mcp/slack` | Slack messaging | Communication |
| `@mcp/email` | SMTP/IMAP email access | Communication |
| `mcp-server-puppeteer` | Headless browser | Web |
| `mcp-server-vectorize` | Vector database access | Memory |

#### Official MCP Clients

- **Claude Desktop** — First major MCP client
- **VS Code (GitHub Copilot)** — MCP tool integration in IDE
- **JetBrains AI** — MCP support in JetBrains suite
- **LangChain/LangGraph** — Native MCP client integration
- **AutoGen** — MCP adapter layer
- **OpenAI Agents SDK** — Full MCP client support

---

## 3. Agent Communication Protocol (ACP)

### 3.1 Architecture & Message Format

ACP extends JSON-RPC 2.0 for agent-to-agent communication, adding concepts like agent identity, conversation threading, and delegation.

**Base message structure:**

```json
{
  "jsonrpc": "2.0",
  "id": "msg_001",
  "method": "agent.send",
  "params": {
    "from": "agent://research-agent-01",
    "to": "agent://writer-agent-01",
    "thread_id": "thread_abc123",
    "message": {
      "type": "delegation",
      "content": {
        "task": "Write summary of the research findings",
        "context": ["doc_001", "doc_002"],
        "deadline": "2026-06-15T18:00:00Z",
        "priority": "high"
      }
    },
    "metadata": {
      "ttl": 300,
      "require_ack": true,
      "trace_id": "trace_xyz789"
    }
  }
}
```

**Agent identity registration:**

```json
{
  "method": "agent.register",
  "params": {
    "agent_id": "agent://research-agent-01",
    "capabilities": ["research", "web-search", "data-analysis"],
    "max_concurrent_tasks": 5,
    "supported_transports": ["mcp", "acp"],
    "contact_endpoint": "acp://research-cluster.internal:9001"
  }
}
```

### 3.2 Inter-Agent Communication

ACP defines several message types for common agent interactions:

**Message Types:**

| Type | Purpose | Direction |
|------|---------|-----------|
| `agent.send` | Direct message between agents | Any |
| `agent.broadcast` | One-to-many message | One→Many |
| `agent.delegate` | Task delegation | Manager→Worker |
| `agent.handoff` | Transfer conversation | Any→Any |
| `agent.query` | Request information | Any→Any |
| `agent.status` | Status update | Worker→Manager |
| `agent.result` | Task completion result | Worker→Manager |
| `agent.error` | Error reporting | Any→Any |

**Example: Query and Response**

```json
// Agent A queries Agent B
{
  "method": "agent.query",
  "params": {
    "from": "agent://support-bot",
    "to": "agent://inventory-svc",
    "query": "Show available inventory for product SKU-4421",
    "response_format": {
      "type": "object",
      "properties": {
        "available_quantity": {"type": "integer"},
        "estimated_restock_date": {"type": "string"}
      }
    }
  }
}

// Agent B responds
{
  "method": "agent.result",
  "params": {
    "from": "agent://inventory-svc",
    "to": "agent://support-bot",
    "in_response_to": "msg_001",
    "result": {
      "available_quantity": 42,
      "estimated_restock_date": "2026-06-20"
    }
  }
}
```

### 3.3 Delegation Patterns

ACP supports three primary delegation patterns:

#### Pattern 1: Direct Delegation (Manager→Worker)

```
Manager identifies subtask → finds capable agent → delegates → receives result
```

```json
{
  "method": "agent.delegate",
  "params": {
    "from": "agent://orchestrator",
    "to": "agent://code-reviewer",
    "task_id": "task_456",
    "task": {
      "description": "Review PR #823 for security vulnerabilities",
      "input": {"pr_url": "https://github.com/org/repo/pull/823"},
      "output_schema": {
        "type": "object",
        "properties": {
          "vulnerabilities": {"type": "array"},
          "overall_score": {"type": "number"}
        }
      }
    },
    "options": {
      "timeout_seconds": 120,
      "retry_on_failure": true,
      "max_retries": 3
    }
  }
}
```

#### Pattern 2: Broadcast Delegation (Market-style)

```
Manager broadcasts task → agents bid → manager selects → result gathered
```

```json
{
  "method": "agent.broadcast",
  "params": {
    "from": "agent://scheduler",
    "message": {
      "type": "task_offer",
      "task": {
        "id": "task_789",
        "description": "Render 3D scene from point cloud data",
        "estimated_effort": "5 minutes",
        "reward": "priority_score_+10"
      }
    },
    "filter": {
      "require_capability": "3d-rendering",
      "min_available_capacity": 0.5
    }
  }
}
```

#### Pattern 3: Hierarchical Decomposition

```
Top agent decomposes → mid-level agents sub-delegate → leaf agents execute
```

### 3.4 Handoff Protocols

Handoff transfers a conversation context from one agent to another, preserving state.

**Handoff message:**

```json
{
  "method": "agent.handoff",
  "params": {
    "from": "agent://triage-agent",
    "to": "agent://billing-agent",
    "conversation": {
      "thread_id": "thread_456",
      "history": [
        {"role": "user", "content": "I need to dispute a charge"},
        {"role": "assistant", "content": "I can help with billing disputes..."}
      ],
      "current_state": {
        "intent": "dispute_charge",
        "customer_id": "cust_789",
        "transaction_id": "txn_101112"
      },
      "metadata": {
        "sentiment": "frustrated",
        "language": "en",
        "priority": "high"
      }
    },
    "reason": "billing_specialist_required"
  }
}
```

**Handoff acknowledgment:**

```json
{
  "method": "agent.handoff_ack",
  "params": {
    "from": "agent://billing-agent",
    "to": "agent://triage-agent",
    "accept": true,
    "estimated_wait_seconds": 5
  }
}
```

### 3.5 Coordination & Consensus

ACP provides primitives for agents to coordinate without a central orchestrator:

**Voting protocol:**
```json
{
  "method": "agent.coordinate",
  "params": {
    "from": "agent://analyst-01",
    "coordination_id": "coord_001",
    "type": "vote",
    "proposal": {
      "action": "approve_deployment",
      "target": "release-v2.1.3",
      "deadline": "2026-06-14T12:00:00Z"
    },
    "voting_rules": {
      "quorum_percentage": 60,
      "approval_threshold": 66.7,
      "voter_eligibility": ["agent://qa-lead", "agent://security-lead", "agent://dev-lead"]
    }
  }
}
```

**Consensus response:**
```json
{
  "method": "agent.coordinate_result",
  "params": {
    "coordination_id": "coord_001",
    "outcome": "approved",
    "votes": {
      "agent://qa-lead": "approve",
      "agent://security-lead": "approve",
      "agent://dev-lead": "abstain"
    },
    "approval_percentage": 66.7
  }
}
```

---

## 4. Implementation Guide

### 4.1 Building an MCP Server

**Python (FastMCP):**

```python
from mcp.server.fastmcp import FastMCP
import httpx

mcp = FastMCP("My Custom Server")

@mcp.tool()
def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """Calculate BMI given weight and height."""
    return round(weight_kg / (height_m ** 2), 2)

@mcp.tool()
async def fetch_news(topic: str, limit: int = 5) -> list:
    """Fetch latest news headlines for a topic."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://newsapi.org/v2/everything",
            params={"q": topic, "pageSize": limit}
        )
        return resp.json().get("articles", [])

@mcp.resource(uri="mcp://config/app-settings")
def get_config() -> str:
    """Return application configuration."""
    return open("config.json").read()

@mcp.prompt()
def code_review_prompt(language: str, code_snippet: str) -> str:
    """Template for code review."""
    return f"""Review this {language} code:
```{language}
{code_snippet}
```
Check for: security issues, performance problems, style violations, edge cases."""

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

**TypeScript (Official SDK):**

```typescript
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

const server = new McpServer({
  name: "My Server",
  version: "1.0.0"
});

server.tool(
  "search_knowledge_base",
  "Search an internal knowledge base",
  { query: z.string(), limit: z.number().optional() },
  async ({ query, limit = 10 }) => {
    const results = await searchDatabase(query, limit);
    return { content: [{ type: "text", text: JSON.stringify(results) }] };
  }
);

const transport = new StdioServerTransport();
await server.connect(transport);
```

### 4.2 Building an MCP Client

**Python MCP client:**

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run_mcp_client():
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "my_mcp_server"],
        env={"MCP_CONFIG": "production"}
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize connection
            await session.initialize()
            
            # List available tools
            tools = await session.list_tools()
            print(f"Available tools: {[t.name for t in tools]}")
            
            # Call a tool
            result = await session.call_tool(
                "calculate_bmi",
                arguments={"weight_kg": 75, "height_m": 1.80}
            )
            print(f"BMI Result: {result.content[0].text}")
```

### 4.3 ACP Agent Registration

**Registering an agent with the ACP directory:**

```python
from acp import AgentDirectory, AgentRegistration

# Agent registration
registration = AgentRegistration(
    agent_id="agent://data-analyzer-01",
    name="Data Analyzer",
    version="2.0.0",
    capabilities=[
        "data_analysis",
        "visualization",
        "statistical_testing"
    ],
    llm_config={
        "model": "gpt-5",
        "max_tokens": 16000
    },
    endpoint="acp://internal-cluster:9002",
    max_concurrent_tasks=3
)

directory = AgentDirectory()
await directory.register(registration)

# Discover agents by capability
analysts = await directory.find_agents(
    capability="data_analysis",
    min_available_capacity=0.5
)
```

### 4.4 Multi-Agent Conversation with ACP

```python
from acp import AgentClient, Conversation

# Create conversation thread
conv = Conversation(thread_id="thread_agent_demo")

# Send delegation via ACP
client = AgentClient(endpoint="acp://orchestrator:9000")

response = await client.send_message(
    to="agent://research-agent",
    message={
        "type": "delegation",
        "content": {
            "task": "Research latest GPU benchmarks",
            "output_format": "summary"
        }
    },
    thread_id=conv.thread_id,
    require_ack=True
)

# Check if handoff needed
if response.get("handoff_suggested"):
    # Follow handoff
    handoff_result = await client.accept_handoff(response["handoff"])
    print(f"Handoff to: {response['handoff']['target_agent']}")
```

---

## 5. Adoption & Ecosystem

### MCP Adoption (June 2026)

| Category | Adoption Rate | Key Players |
|----------|---------------|-------------|
| LLM Providers | 85% | OpenAI, Anthropic, Google, Meta, Mistral |
| Agent Frameworks | 90% | LangChain, CrewAI, AutoGen, Semantic Kernel |
| IDE/Tools | 70% | VS Code, JetBrains, Cursor, Windsurf |
| Enterprise Platforms | 60% | Azure AI, AWS Bedrock, GCP Vertex |
| SaaS Tools | 45% | Notion, Slack, Salesforce, Zendesk |

### ACP Adoption

| Use Case | Deployment Share | Sectors |
|----------|-----------------|---------|
| Customer Service | 40% | Finance, Telecom, Retail |
| DevOps Automation | 25% | Technology, SaaS |
| Research & Analysis | 15% | Pharma, Consulting |
| Content Generation | 12% | Media, Marketing |
| Other | 8% | Healthcare, Legal |

---

## 6. Tool Integration Matrix

How major frameworks integrate with MCP/ACP:

| Framework | MCP Client | MCP Server Host | ACP Support | Notes |
|-----------|------------|-----------------|-------------|-------|
| LangChain | ✅ Native | ✅ Via LangServe | ✅ Via LangGraph | First-class support |
| CrewAI | ✅ Via adapter | ❌ | ✅ Native | ACP for crew handoffs |
| AutoGen | ✅ Native (v0.4+) | ✅ | ✅ Via extension | MCP tool registry |
| Semantic Kernel | ✅ Via plugin | ❌ | ❌ | Azure roadmap |
| OpenAI SDK | ✅ Native | ❌ | ✅ Native | Built-in agent routing |
| Claude API | ✅ Native | ✅ (Claude Desktop) | ❌ | First MCP adopter |

---

## 7. Security Considerations

### Threat Model

| Threat | Description | Mitigation |
|--------|-------------|------------|
| Tool injection | Malicious tool arguments | Input validation, schema enforcement |
| Tool poisoning | Compromised MCP server returns bad data | Checksums, trusted registries |
| Agent spoofing | Fake agent impersonation | mTLS, signed identities |
| Data exfiltration | Tool access leaks sensitive data | Resource scoping, audit logs |
| DoS via tool calls | Excessive tool invocations | Rate limiting, cost budgets |

### Best Practices

1. **Always validate tool input** against declared schema
2. **Run MCP servers with least privilege** — no network unless required
3. **Use stdio transport** for local, single-user agents
4. **Use mTLS + API keys** for remote MCP servers
5. **Capability-based authorization** for ACP agent interactions
6. **Audit all tool calls** with agent-id, timestamp, input, output
7. **Set tool timeouts** — default 30s, configurable per tool

---

## 8. Benchmarks & Performance

### MCP Latency Overhead

| Transport | Median Latency | P99 Latency | Throughput (req/s) |
|-----------|---------------|-------------|-------------------|
| stdio | 0.08ms | 0.5ms | 25,000 |
| WebSocket | 2ms | 15ms | 10,000 |
| SSE | 5ms | 50ms | 5,000 |
| HTTP/2 | 3ms | 25ms | 8,000 |

### ACP Message Latency

| Pattern | Same-host | Same-region | Cross-region |
|---------|-----------|-------------|--------------|
| Direct delegation | 5ms | 25ms | 150ms |
| Broadcast (10 agents) | 50ms | 200ms | 1.2s |
| Consensus (5 agents) | 100ms | 500ms | 3s |
| Handoff with context | 50ms | 100ms | 500ms |

---

## 9. Future Roadmap

### MCP v1.3 (Expected Q3 2026)
- Streaming tool results (real-time tool output)
- Improved error handling with nested error codes
- Resource lifecycle management (create/update/delete)
- Batch tool operations

### ACP v1.1 (Expected Q4 2026)
- Agent discovery service (DNS for agents)
- Load balancing across agent instances
- Cross-org agent federation
- Streaming agent conversations

### Long-term vision
- **Universal agent registry** — Global directory of MCP servers and ACP agents
- **Monetization layer** — Pay-per-call tool and agent marketplaces
- **Interoperability certification** — Certified MCP/ACP compatible badges
- **Federated identity** — Cross-organizational agent authentication

---

> **Related KB documents:**
> - [02-AI-Agent-Development.md](02-AI-Agent-Development.md) — Agent frameworks using MCP/ACP  
> - [06-RAG-Retrieval-Systems.md](06-RAG-Retrieval-Systems.md) — MCP for retrieval tools  
> - [09-AI-Automation.md](09-AI-Automation.md) — Automation with MCP tool chains  
> - [08-Edge-AI-Inference.md](08-Edge-AI-Inference.md) — Edge MCP servers  
> - [10-Real-Time-AI-Systems.md](10-Real-Time-AI-Systems.md) — Real-time MCP streaming
