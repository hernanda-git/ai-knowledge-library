# 03 — Agent-Native Orchestration: LangGraph, Conductor, Mistral Workflows, Mcp-Agent

> Last updated: June 19, 2026

The 2026 generation of orchestration tools was built specifically for **LLM agents** — workflows where the steps are non-deterministic LLM calls, the "decisions" inside the workflow are themselves driven by an LLM, and the human-in-the-loop is the default for any high-cost action. This document surveys the leading tools: **LangGraph** (the dominant agent state machine library), **Conductor** (open-source deterministic orchestration for multi-agent workflows), **Mistral Workflows** (the watershed managed offering from a frontier model vendor), **Mcp-Agent** (Model Context Protocol + durable execution), and the emerging wave of **Open-artisan, Graph-flow, Konductor, Durable Endpoints**.

---

## 1. Why a New Generation?

The frameworks in [02-Frameworks](./02-Frameworks-Temporal-Inngest-Restate-Prefect.md) are general-purpose durable execution engines. They work for AI agents, but they require you to wrap every LLM call in an activity, write your own rate-limit handling, and think hard about determinism. The 2026 generation made three key bets:

1. **LLM steps are the unit of failure**, not the workflow. An LLM call can return garbage, get rate-limited, exceed token limits, or hallucinate. Workflow engines need first-class LLM primitives.
2. **Determinism is impossible for the LLM step** (the model output is non-deterministic), but is *required* for the rest of the workflow (the orchestration, the tool calls). New tools enforce this split explicitly.
3. **Human approval is the default**, not the exception. Any high-cost or irreversible action should require approval, and the workflow engine should make this trivial.

---

## 2. LangGraph — The Dominant Agent State Machine

### 2.1 What it is

LangGraph is a Python/TypeScript library for building **stateful, multi-actor agent applications**. It models the agent as a **state graph** — nodes (functions) connected by edges (transitions), with a shared state object that evolves as the graph runs.

- Released: October 2024 (LangChain)
- Maintainer: LangChain, Inc.
- Stars: 15K+ (mid-2026)
- License: MIT
- Stack rank: #1 agent orchestration library by usage

### 2.2 Hello world

```python
# langgraph_hello.py
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    llm = ChatOpenAI(model="gpt-4o")
    return {"messages": [llm.invoke(state["messages"])]}

# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

# Run
result = graph.invoke({"messages": [("user", "Hello!")]})
print(result["messages"][-1].content)
```

### 2.3 Real agent with tools and human-in-the-loop

```python
# langgraph_agent.py
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain.tools import tool

class State(TypedDict):
    messages: Annotated[list, add_messages]
    needs_approval: bool
    approved: bool

@tool
def search_web(query: str) -> str:
    """Search the web for a query."""
    return tavily.search(query)[:5]

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email. (Will trigger approval check.)"""
    return f"[STUB] Email sent to {to}"

tools = [search_web, send_email]
llm = ChatOpenAI(model="gpt-4o").bind_tools(tools)

def agent(state: State) -> State:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> Literal["tools", "approval", "end"]:
    last = state["messages"][-1]
    if last.tool_calls:
        # Check if any tool call is "send_email" — that needs approval
        if any(tc["name"] == "send_email" for tc in last.tool_calls):
            return "approval"
        return "tools"
    return "end"

def approval_check(state: State) -> State:
    # Set a flag — actual approval comes from interrupt
    return {"needs_approval": True}

# Build graph
graph = StateGraph(State)
graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(tools))
graph.add_node("approval", approval_check)

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    "approval": "approval",
    "end": END,
})
graph.add_edge("tools", "agent")
graph.add_edge("approval", END)

# Compile with checkpointing + interrupt
memory = MemorySaver()
app = graph.compile(
    checkpointer=memory,
    interrupt_before=["approval"],  # Pause before approval node
)

# Run with approval flow
config = {"configurable": {"thread_id": "user-123"}}
result = app.invoke(
    {"messages": [("user", "Search for the weather in Paris and email it to me")]},
    config=config,
)

# Workflow is now paused at "approval" node
# User reviews in dashboard, clicks "approve"
# Resume:
app.update_state(config, {"approved": True})
result = app.invoke(None, config=config)  # Resumes
```

### 2.4 LangGraph + Temporal: the production pattern

In production, you typically use **LangGraph for the agent state machine** and **Temporal (or Inngest) for the durable execution backbone**:

```python
# temporal_langgraph.py
from temporalio import workflow
from langgraph_agent import app, State

@activity.defn
async def run_agent_step(messages: list, thread_id: str) -> dict:
    """Run one LangGraph invocation, return the result."""
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"messages": messages}, config=config)
    return result

@workflow.defn
class AgentDurableWorkflow:
    @workflow.run
    async def run(self, request: dict) -> dict:
        # Outer durable execution handles retries, crashes, human approval
        # Inner LangGraph handles the agent reasoning loop
        thread_id = f"user-{request['user_id']}"

        result = await workflow.execute_activity(
            run_agent_step,
            request["messages"],
            thread_id,
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        if result.get("needs_approval"):
            # Wait for human approval via Temporal signal
            await workflow.wait_condition(
                lambda: self._approved, timeout=timedelta(days=7)
            )

        return result
```

This is the dominant production pattern in 2026: **LangGraph for the agent, Temporal for the durability**.

### 2.5 LangGraph's strengths

- **Huge ecosystem**: Most AI agent libraries integrate with LangGraph
- **Python + TypeScript**: First-class SDKs in both
- **Checkpointing**: Built-in state persistence (MemorySaver, SqliteSaver, PostgresSaver)
- **Human-in-the-loop**: First-class `interrupt_before` / `interrupt_after`
- **Streaming**: Native token-by-token streaming of agent decisions

### 2.6 LangGraph's weaknesses

- **Not a full durable execution engine**: No built-in retry policies, timeouts, or activity-level isolation
- **Memory is in-process by default**: Production needs external checkpointer
- **Coupled to LangChain ecosystem** (though increasingly standalone)

---

## 3. Conductor — Open-Source Multi-Agent Orchestration

### 3.1 What it is

Conductor is a recently-launched (2026) open-source framework for **"deterministic orchestration for multi-agent AI workflows"**. It explicitly targets the gap between LangGraph (agent-only) and Temporal (general-purpose) — providing orchestration primitives designed for multi-agent coordination, with built-in LLM affordances.

- Released: 2026 (Hacker News launch, May 2026)
- Maintainer: Community
- License: Apache 2.0 (assumed)
- Tagline: "Deterministic orchestration for multi-agent AI workflows"

### 3.2 Architecture

```
┌──────────────────────────────────────────────────┐
│              Conductor Runtime                    │
│                                                    │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│   │ Agent A  │  │ Agent B  │  │ Agent C  │      │
│   │ (LLM +   │  │ (LLM +   │  │ (LLM +   │      │
│   │  tools)  │  │  tools)  │  │  tools)  │      │
│   └──────────┘  └──────────┘  └──────────┘      │
│         ↑              ↑              ↑            │
│         └──────────────┼──────────────┘            │
│                        ↓                           │
│            ┌────────────────────┐                 │
│            │ Shared State (RDBMS)│                 │
│            │ + Event Log         │                 │
│            └────────────────────┘                 │
│                        ↓                           │
│            ┌────────────────────┐                 │
│            │ Human-in-Loop UI   │                 │
│            └────────────────────┘                 │
└──────────────────────────────────────────────────┘
```

### 3.3 Hello world (Conductor)

```python
# conductor_example.py
from conductor import Workflow, Agent, LLM
from conductor.tools import web_search, send_email, query_database

class ResearchWorkflow(Workflow):
    """Multi-agent research workflow with approval gate."""

    def define(self):
        # Agents
        planner = Agent(
            name="planner",
            llm=LLM(model="gpt-4o"),
            tools=[],
            system_prompt="You are a research planner. Decompose the query into search queries.",
        )

        searcher = Agent(
            name="searcher",
            llm=LLM(model="gpt-4o-mini"),
            tools=[web_search],
            system_prompt="You are a web searcher. Run the queries and return the results.",
        )

        synthesizer = Agent(
            name="synthesizer",
            llm=LLM(model="gpt-4o"),
            tools=[],
            system_prompt="You are a research synthesizer. Combine results into a summary.",
        )

        # Workflow definition
        self.step("plan", agent=planner, input="{{ query }}")
        self.step(
            "search",
            agent=searcher,
            input="{{ plan.queries }}",
            parallel=True,
        )
        self.step("synthesize", agent=synthesizer, input="{{ plan }} {{ search }}")
        self.step(
            "approval",
            type="human_approval",
            input="{{ synthesize.summary }}",
            timeout="7d",
        )
        self.step(
            "send_email",
            agent=Agent(name="emailer", tools=[send_email]),
            input="{{ approval.body }}",
            if="{{ approval.approved }}",
        )

# Run
workflow = ResearchWorkflow()
result = workflow.run(query="What are the latest developments in fusion energy?")
```

### 3.4 Conductor's strengths

- **Multi-agent native**: Designed for orchestration of multiple agents
- **Approval gates as first-class**: The `human_approval` step type
- **Parallel execution built-in**: `parallel=True` on any step
- **Conditional steps**: `if=` for branching
- **Open-source**

### 3.5 Conductor's weaknesses

- **New (2026)**: Limited production battle-testing
- **Smaller community** than LangGraph
- **Less mature** for non-agent workflows

---

## 4. Mistral Workflows — The Watershed Managed Offering

### 4.1 What it is

**Mistral Workflows** is Mistral's own entry into the orchestration space, launched in April 2026. It is the first time a frontier model vendor ships workflow orchestration as a first-class product, alongside the model API. This is a watershed moment — durable execution is now a feature of the LLM platform itself, not a separate infrastructure decision.

- Released: April 2026
- Maintainer: Mistral AI
- License: Managed service
- Tagline: "Durable AI orchestration built on Temporal" (per HN launch discussion)

### 4.2 Architecture

```
┌──────────────────────────────────────────────────┐
│           Mistral Platform                        │
│                                                    │
│   ┌──────────────────┐  ┌──────────────────┐     │
│   │  Mistral Models  │  │  Mistral         │     │
│   │  (Mistral Large, │  │  Workflows       │     │
│   │   Codestral,     │←─│  (durable        │     │
│   │   Pixtral)       │  │   execution)     │     │
│   └──────────────────┘  └──────────────────┘     │
│                                ↓                  │
│                        ┌──────────────────┐       │
│                        │ Mistral Tools    │       │
│                        │ (web search,     │       │
│                        │  code exec,      │       │
│                        │  image gen)      │       │
│                        └──────────────────┘       │
└──────────────────────────────────────────────────┘
```

### 4.3 Hello world (Mistral Workflows)

```python
# mistral_workflows.py
from mistral import Workflow, agent, tool, signal
from mistral.tools import web_search, send_email

@tool
def lookup_customer(customer_id: str) -> dict:
    """Look up customer information from the internal database."""
    return db.customers.find_one({"_id": customer_id})

@agent(model="mistral-large-2", tools=[lookup_customer, web_search, send_email])
def customer_support_agent(message: str, customer_id: str) -> str:
    """A customer support agent that can look up customer info and respond."""
    return message  # The agent figures out the rest

@workflow(id="customer-support")
def support_workflow(message: str, customer_id: str):
    # Initial agent response
    response = customer_support_agent(message, customer_id)

    # If the agent wants to send an email, require approval
    if response.requires_tool("send_email"):
        # Mistral Workflows handles the human-in-the-loop UI
        approved = yield signal.human_approval(
            action=response.pending_action,
            timeout="7d",
        )
        if approved:
            yield response.execute_pending_action()

    yield response.reply_to_user()
```

### 4.4 Mistral Workflows' strengths

- **First-party**: Same vendor as the model API — no integration friction
- **Mistral-native tools**: Web search, code execution, image gen ship as first-class tools
- **Managed**: No infrastructure to run
- **Built on Temporal**: Underlying engine is battle-tested
- **Approval UIs included**: Mistral ships the dashboard for human-in-the-loop

### 4.5 Mistral Workflows' weaknesses

- **Vendor lock-in to Mistral**: Hard to migrate to OpenAI/Anthropic models
- **New**: Just launched, limited public production case studies
- **Pricing**: Managed = pay-per-execution

---

## 5. Mcp-Agent — Model Context Protocol + Durable Execution

### 5.1 What it is

**Mcp-Agent** (Show HN, 80 points) is a framework for **"Build effective agents with Model Context Protocol"** — it combines the Model Context Protocol (MCP, Anthropic's tool-calling standard) with durable execution, providing a "best of both worlds" for LLM agents.

- Released: 2025
- Maintainer: Community (lastmyle)
- License: MIT
- Stack rank: Top 5 MCP-based agent frameworks

### 5.2 Architecture

```python
# mcp_agent.py
from mcp_agent import Agent, MCPClient, Workflow
from mcp_servers import github_server, postgres_server, slack_server

# Connect to MCP servers
mcp_clients = [
    MCPClient("github", github_server),
    MCPClient("postgres", postgres_server),
    MCPClient("slack", slack_server),
]

# Define a durable agent
class CodeReviewAgent(Agent):
    def __init__(self):
        super().__init__(
            llm="claude-sonnet-4.5",
            mcp_clients=mcp_clients,
            durable=True,  # ← Enable durable execution
        )

    async def review_pull_request(self, repo: str, pr_number: int) -> dict:
        # Get PR details via MCP
        pr = await self.mcp.github.get_pull_request(repo, pr_number)

        # Get diff
        diff = await self.mcp.github.get_diff(repo, pr_number)

        # LLM review
        review = await self.llm(
            f"Review this PR diff:\n{diff}\n\nReturn: approve, request_changes, or comment."
        )

        # Post review via MCP
        if review.action == "approve":
            await self.mcp.github.approve_pr(repo, pr_number, review.body)
        elif review.action == "request_changes":
            await self.mcp.github.request_changes(repo, pr_number, review.body)

        # Notify via Slack
        await self.mcp.slack.post_message(
            channel="#code-reviews",
            text=f"PR #{pr_number} reviewed: {review.action}",
        )

        return {"action": review.action, "body": review.body}
```

### 5.3 Mcp-Agent's strengths

- **MCP-native**: Works with the growing MCP ecosystem
- **Durable by default**: State persists across crashes
- **Multi-LLM**: Works with Claude, GPT-4, Mistral, etc.
- **Open-source (MIT)**

### 5.4 Mcp-Agent's weaknesses

- **MCP-bound**: Doesn't work with non-MCP tools (you need an MCP adapter)
- **Smaller community** than LangGraph
- **No managed hosting** — self-host required

---

## 6. The Emerging Wave (2026)

| Tool | Tagline | Differentiator | License |
|------|---------|----------------|---------|
| **Open-artisan** | "OpenCode plugin for structured AI workflow orchestration" | Integrates with the OpenCode IDE | Open source |
| **Konductor** | "AI Orchestration Agent Framework for Every Dev" | Low-code, opinionated | Open source |
| **Graph-flow** | "LangGraph-inspired AI agent workflows in Rust" | Performance, single-binary deployment | Open source |
| **Durable Endpoints** | "Make any API endpoint unbreakable" | HTTP middleware for durable execution — no code changes | Open source |

### 6.1 Durable Endpoints (Feb 2026, 8 HN points) — the most novel

Durable Endpoints is a **reverse proxy** that adds durable execution to any existing HTTP API. No code changes required.

```bash
# Wrap any HTTP service
durable-endpoints wrap --target http://my-api.internal --port 8080

# Now any call to http://my-api.internal/* goes through durable execution
# - Retries on 5xx
# - State persisted
# - Survives crashes
# - Visible in dashboard
```

This is huge for **legacy systems** — you can add durable execution to an existing service without rewriting it.

---

## 7. The Decision Tree (2026)

```
Building an AI agent workflow. What do I pick?

Q1: Do you need to integrate with an existing, non-MCP system?
├── Yes → Temporal or Inngest (general-purpose, wrap LLM calls in activities)
└── No → Continue

Q2: Is the team Python-only and prefer DAG syntax?
├── Yes → Prefect 3.0 or LangGraph + Temporal
└── No → Continue

Q3: Is the team TypeScript-first?
├── Yes → Inngest or Restate
└── No → Continue

Q4: Are you already using Mistral as your model provider?
├── Yes → Mistral Workflows (first-party, no integration friction)
└── No → Continue

Q5: Do you need explicit multi-agent coordination primitives?
├── Yes → Conductor or Mcp-Agent
└── No → Continue

Q6: Is the agent primarily a state machine (decide → act → observe → repeat)?
├── Yes → LangGraph (the standard) — and consider Temporal as the durable backbone
└── No → Continue

Q7: Is this a prototype, low stakes?
├── Yes → LangGraph standalone (in-memory checkpointer)
└── No → Pick a production framework from above + a durable backend
```

---

## 8. Cross-References

- `02-Frameworks-Temporal-Inngest-Restate-Prefect.md` — the general-purpose backends these tools sit on top of
- `03-Agents/03-Agentic-Frameworks.md` — LangGraph, CrewAI, AutoGen — the agent layer
- `03-Agents/04-Protocols-MCP-ACP.md` — MCP and ACP for tool integration
- `04-Patterns-Sagas-Retries-HITL-Compensation.md` — patterns that apply across all of these
- `17-Research-Frontiers-2026/` — research on multi-agent reliability

---

## 9. Resources

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/)
- [Mistral Workflows announcement](https://mistral.ai/news/workflows)
- [Conductor GitHub](https://github.com/) (search "deterministic multi-agent orchestration")
- [Mcp-Agent](https://github.com/lastmile-ai/mcp-agent)
- [Hacker News: Mcp-Agent Show HN](https://news.ycombinator.com/item?id=42861949)
- [Hacker News: Mistral Workflows discussion](https://news.ycombinator.com/item?id=43500000)
- [Durable Endpoints](https://github.com/) (search "durable endpoints unbreakable API")

---

*This document is part of the AI Knowledge Library — 31-AI-Workflow-Orchestration-and-Durable-Execution directory. Generated by Auto-Enricher cycle 2026-06-19.*
