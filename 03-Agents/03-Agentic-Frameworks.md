# Agentic Frameworks

> Comprehensive reference and comparison of major agentic frameworks: LangGraph, CrewAI, AutoGen, Semantic Kernel, smolagents, DSPy, OpenAI Agents SDK, Anthropic MCP, Google ADK, Amazon Bedrock Agents, Agno/Phidata, and Pydantic AI.

---

## Table of Contents

1. [LangGraph](#langgraph)
2. [CrewAI](#crewai)
3. [AutoGen / AutoGen Studio](#autogen--autogen-studio)
4. [Semantic Kernel](#semantic-kernel)
5. [smolagents](#smolagents)
6. [DSPy](#dspy)
7. [OpenAI Agents SDK](#openai-agents-sdk)
8. [Anthropic MCP Client](#anthropic-mcp-client)
9. [Google Agent Development Kit (ADK)](#google-agent-development-kit-adk)
10. [Amazon Bedrock Agents](#amazon-bedrock-agents)
11. [Agno / Phidata](#agno--phidata)
12. [Pydantic AI](#pydantic-ai)
13. [Comparison Table](#comparison-table)

---

## LangGraph

LangGraph is a framework developed by LangChain for building stateful, multi-agent applications using graph-based state machines. It enables developers to define agent workflows as directed graphs where nodes are computation steps and edges define control flow.

### Core Architecture

LangGraph models agent workflows as a **state machine** where each node in the graph represents a step that transforms the shared state.

#### StateGraph

`StateGraph` is the primary graph type. It operates on a shared state object that persists across nodes.

**Key Components:**
- **State Schema**: A TypedDict or Pydantic model defining the structure of the shared state.
- **Nodes**: Functions or runnable objects that take state as input and return state updates.
- **Edges**: Define the flow between nodes (directed connections).
- **Conditional Edges**: Dynamic routing based on state content.
- **Entry Point**: The first node to execute.
- **Finish Points**: Nodes where execution terminates.

**Basic StateGraph Example:**
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import operator

class GraphState(TypedDict):
    input: str
    messages: List[str]
    next_step: str

def node_a(state: GraphState) -> GraphState:
    # Process and return updates (only returning changes)
    return {"messages": state["messages"] + ["Node A processed"]}

def node_b(state: GraphState) -> GraphState:
    return {"messages": state["messages"] + ["Node B processed"]}

def router(state: GraphState) -> str:
    """Conditional edge: decide which node to go to next."""
    if "error" in state:
        return "error_handler"
    elif state["next_step"] == "b":
        return "node_b"
    return END

# Build the graph
builder = StateGraph(GraphState)
builder.add_node("node_a", node_a)
builder.add_node("node_b", node_b)
builder.add_node("error_handler", error_handler)

builder.set_entry_point("node_a")
builder.add_edge("node_a", "node_b")
builder.add_conditional_edges("node_b", router, {
    "node_a": "node_a",
    "error_handler": "error_handler",
    END: END
})

# Compile into an executable app
app = builder.compile()
```

#### MessageGraph

`MessageGraph` is a specialized graph for chat-based applications. It operates on lists of messages rather than arbitrary state.

```python
from langgraph.graph import MessageGraph, END
from langchain_core.messages import HumanMessage, AIMessage

def assistant_node(state):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": messages + [response]}

def human_review_node(state):
    # Interrupt for human review
    return {"messages": state["messages"]}

# MessageGraph is simpler - just message passing
graph = MessageGraph()
graph.add_node("assistant", assistant_node)
graph.add_node("human_review", human_review_node)

graph.set_entry_point("assistant")
graph.add_edge("assistant", "human_review")
graph.add_conditional_edges("human_review", 
    lambda state: "assistant" if state["approved"] else END)
```

### Node Types and Patterns

Nodes in LangGraph are the fundamental building blocks. They can be:

1. **Function Nodes**: Simple Python functions that take state and return state updates.
```python
def my_node(state: dict) -> dict:
    # Process state
    return {"key": "value"}
```

2. **Runnable Nodes**: LangChain Runnable objects (chains, LCEL pipelines).
```python
from langchain_core.runnables import RunnableLambda

chain = prompt | llm | parser
builder.add_node("llm_call", chain)
```

3. **Subgraph Nodes**: Entire graphs used as nodes within a larger graph.
```python
subgraph = create_subgraph().compile()
builder.add_node("sub_task", subgraph)
```

4. **Agent Nodes**: Full agent loops that can make multiple tool calls.
```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(model, tools)
builder.add_node("agent_step", agent)
```

### Edges and Control Flow

LangGraph provides several types of edges for different control flow patterns:

#### Direct Edges
Simple, unconditional transitions between nodes.
```python
graph.add_edge("node_a", "node_b")
```

#### Conditional Edges
Edges that route based on a function evaluating the current state.
```python
def route_decision(state: dict) -> str:
    if state["quality"] > 0.8:
        return "finalize"
    elif state["attempts"] < 3:
        return "retry"
    return "fallback"

graph.add_conditional_edges(
    "evaluator",
    route_decision,
    {
        "finalize": "output_node",
        "retry": "generator",
        "fallback": "human_review"
    }
)
```

#### Entry and Finish Points
```python
graph.set_entry_point("start_node")
graph.set_finish_point("end_node")
# Can also set conditional entry:
graph.set_conditional_entry_point(entry_router)
```

### Checkpointing and Persistent State

LangGraph provides built-in checkpointing for state persistence, enabling pause/resume, human-in-the-loop, and fault recovery.

```python
from langgraph.checkpoint import MemorySaver

# In-memory checkpointing
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# SQLite checkpointing (persistent)
from langgraph.checkpoint import SqliteSaver
with SqliteSaver.from_conn_string("checkpoints.db") as saver:
    app = graph.compile(checkpointer=saver)
    
    # Run with thread ID for conversation continuity
    config = {"configurable": {"thread_id": "conversation_1"}}
    result = app.invoke({"input": "Hello"}, config)
```

**Checkpointing Features:**
- **Automatic State Saving**: State is saved after each node execution.
- **Thread-Based Isolation**: Each conversation/run gets a unique thread_id.
- **Pause/Resume**: Execution can be paused and resumed later.
- **State Inspection**: View state at any checkpoint.
- **Branching**: Fork from any checkpoint for exploration.

### Streaming

LangGraph supports multiple streaming modes for real-time output:

```python
# Stream all steps
for event in app.stream({"input": "Hello"}):
    for node_name, output in event.items():
        print(f"Node {node_name}: {output}")

# Stream only LLM tokens
for event in app.stream_events({"input": "Hello"}):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content)

# Stream values (final state after each node)
for state in app.stream_values({"input": "Hello"}):
    print(state)
```

**Streaming Modes:**
- **`stream()`**: Yields events for each node execution.
- **`stream_events()`**: Yields fine-grained events (LLM tokens, tool calls, etc.).
- **`stream_values()`**: Yields the full state after each node.
- **`astream()`**: Async version of stream().

### Human-in-the-Loop Interrupt

LangGraph supports interrupting execution for human input:

```python
def human_review_node(state: dict) -> dict:
    # INTERRUPT: Wait for human input
    human_response = interrupt(
        "Please review the generated content and provide feedback."
    )
    state["human_feedback"] = human_response
    return state

# Resume execution with human input
# (from a different process/session)
config = {"configurable": {"thread_id": "thread_1"}}
# Get the state at the interrupt point
state = app.get_state(config)
# Resume with human input
result = app.invoke(
    None,
    config,
    interrupt="Looks good, proceed."
)
```

### Subgraphs

Subgraphs enable modular composition of agent workflows:

```python
def create_research_subgraph():
    """A self-contained research subgraph."""
    builder = StateGraph(ResearchState)
    builder.add_node("search_web", search_web)
    builder.add_node("extract_content", extract_content)
    builder.add_node("summarize", summarize)
    builder.set_entry_point("search_web")
    builder.add_edge("search_web", "extract_content")
    builder.add_edge("extract_content", "summarize")
    return builder.compile()

# Use in parent graph
main_graph = StateGraph(MainState)
main_graph.add_node("research", create_research_subgraph())
main_graph.add_node("write_report", write_report)
main_graph.set_entry_point("research")
main_graph.add_edge("research", "write_report")
```

**Subgraph Features:**
- **Encapsulation**: Internal state is isolated from parent graph.
- **Reusability**: Subgraphs can be shared across projects.
- **Composability**: Subgraphs can contain subgraphs.
- **Input/Output Mapping**: Map parent state to subgraph state and back.

### Parallel Execution

LangGraph supports parallel node execution through fan-out patterns:

```python
from langgraph.graph import add_labels

def fan_out(state):
    """Split state into parallel tasks."""
    tasks = state["input"].split("\n")
    return {"tasks": tasks}

def parallel_worker(state):
    """Worker that processes a single task."""
    task = state["current_task"]
    result = process_task(task)
    return {"results": [result]}

# Fan-out pattern
builder.add_node("split", fan_out)
for i in range(NUM_WORKERS):
    builder.add_node(f"worker_{i}", parallel_worker)

builder.set_entry_point("split")
# Dynamic fan-out edges
for i in range(NUM_WORKERS):
    builder.add_edge("split", f"worker_{i}")
```

### LangGraph Cloud / Platform

LangGraph Cloud provides managed deployment infrastructure for LangGraph applications.

**Key Features:**
- **Serverless Deployment**: Automatic scaling of agent applications.
- **Persistent State**: Managed checkpoint storage.
- **API Endpoints**: REST API for interacting with graphs.
- **Monitoring**: Built-in tracing and observability.
- **Versioning**: Deploy multiple versions of your graph.
- **Cron Jobs**: Scheduled graph execution.

**Architecture:**
```
LangGraph Cloud
├── API Gateway
├── Execution Runtime
│   ├── Graph Executor
│   ├── Checkpoint Manager
│   └── Stream Manager
├── State Store (PostgreSQL)
├── Event Store
└── Monitoring Stack
```

**Deployment Configuration:**
```yaml
# langgraph.json
{
  "dependencies": ["."],
  "graphs": {
    "my_agent": "./agent.py:graph"
  },
  "env": {
    "OPENAI_API_KEY": "env:OPENAI_API_KEY"
  }
}
```

### Superstep and Pregel Architecture

LangGraph is built on a **Pregel-inspired architecture**, named after Google's Pregel system for large-scale graph processing.

**Key Concepts:**
- **Superstep**: A round of computation where all active nodes execute in parallel.
- **Message Passing**: Nodes communicate by writing to shared state.
- **Voting to Halt**: Nodes can vote to stop execution.
- **Deterministic Execution**: Same input produces same output (deterministic ordering).

**Superstep Execution Model:**
```
Superstep 0: Entry node(s) execute
Superstep 1: Fan-out to parallel nodes
Superstep 2: Process and aggregate
Superstep 3: Conditional routing
...
Superstep N: Finish node(s)
```

**Advantages of Pregel Model:**
- **Parallelism**: Natural parallel execution of independent nodes.
- **Scalability**: Can distribute across multiple machines.
- **Fault Tolerance**: Superstep boundaries enable checkpointing.
- **Determinism**: Predictable execution ordering.

### Advanced Patterns

#### Agent Supervisor Pattern
```python
from langgraph.graph import StateGraph, END

def supervisor(state):
    """Supervisor agent that routes to specialized workers."""
    messages = state["messages"]
    # LLM decides which worker to call next
    response = supervisor_llm.invoke([
        system_message,
        *messages
    ])
    return {"next_worker": response.content}

def worker_1(state):
    # Specialized worker
    pass

def worker_2(state):
    # Specialized worker
    pass

graph = StateGraph(State)
graph.add_node("supervisor", supervisor)
graph.add_node("worker_1", worker_1)
graph.add_node("worker_2", worker_2)

graph.add_conditional_edges(
    "supervisor",
    lambda s: s["next_worker"],
    {"worker_1": "worker_1", "worker_2": "worker_2", END: END}
)
```

#### Multi-Agent Debate Pattern
```python
def debater_a(state):
    return {"argument_a": generate_argument(state, side="pro")}

def debater_b(state):
    return {"argument_b": generate_argument(state, side="con")}

def judge(state):
    verdict = evaluate_arguments(state["argument_a"], state["argument_b"])
    return {"verdict": verdict}

graph = StateGraph(DebateState)
graph.add_node("debater_a", debater_a)
graph.add_node("debater_b", debater_b)
graph.add_node("judge", judge)

# Parallel execution
graph.set_entry_point("__start__")
graph.add_edge("__start__", "debater_a")
graph.add_edge("__start__", "debater_b")
graph.add_edge("debater_a", "judge")
graph.add_edge("debater_b", "judge")
```

### Limitations

- **Steep Learning Curve**: Graph-based programming model requires mental shift.
- **Debugging Complexity**: State-based debugging can be challenging.
- **Serialization Constraints**: State must be serializable for checkpointing.
- **Performance Overhead**: Checkpointing adds latency per node.
- **Documentation Gaps**: Rapid evolution means docs can be outdated.

---

## CrewAI

CrewAI is a framework for orchestrating role-based AI agents that work together on complex tasks. It provides a high-level abstraction for multi-agent systems.

### Core Architecture

CrewAI is built around four core entities: Agents, Tasks, Crews, and Processes.

```
Crew
├── Process (sequential / hierarchical)
├── Agents
│   ├── Agent 1 (role, goal, backstory, tools)
│   ├── Agent 2 (role, goal, backstory, tools)
│   └── Agent N (...)
├── Tasks
│   ├── Task 1 (description, agent, tools)
│   ├── Task 2 (description, agent, tools)
│   └── Task N (...)
└── Memory (optional)
    ├── Short-term
    ├── Long-term
    └── Entity memory
```

### Agents

Agents in CrewAI are defined by their role, goal, backstory, and available tools.

```python
from crewai import Agent

researcher = Agent(
    role="Senior Research Analyst",
    goal="Uncover cutting-edge developments in AI and technology",
    backstory="""You are a seasoned research analyst with over 15 years 
    of experience in technology analysis. Known for deep dives into 
    complex topics and identifying emerging trends.""",
    allow_delegation=False,
    verbose=True,
    tools=[search_tool, scrape_tool],
    llm="gpt-4-turbo",
    max_iter=15,
    max_rpm=10,
    memory=True
)
```

**Agent Parameters:**
| Parameter | Description |
|-----------|-------------|
| `role` | Agent's function within the crew |
| `goal` | Individual objective the agent is working towards |
| `backstory` | Context and personality for the agent |
| `allow_delegation` | Whether agent can delegate tasks to other agents |
| `tools` | List of tools available to the agent |
| `llm` | Language model to use (default: gpt-4) |
| `max_iter` | Maximum iterations before forced answer |
| `max_rpm` | Maximum requests per minute (rate limiting) |
| `verbose` | Enable detailed logging |
| `memory` | Enable memory for this agent |

### Tasks

Tasks define what needs to be done and can be assigned to specific agents.

```python
from crewai import Task

research_task = Task(
    description="""Research the latest developments in {topic}. 
    Focus on major breakthroughs in the last 6 months.""",
    expected_output="""A comprehensive 3-page report covering:
    - Key breakthroughs and their implications
    - Companies and researchers leading the work
    - Timeline of major developments""",
    agent=researcher,
    tools=[search_tool],  # Override tools for this specific task
    context=[],  # Context from other tasks
    async_execution=False,  # Whether to run asynchronously
    callback=task_callback_function,
    human_input=False
)
```

**Task Features:**
- **Context Passing**: Tasks can receive context from previous tasks.
- **Async Execution**: Tasks can run in parallel.
- **Callbacks**: Hooks for task lifecycle events.
- **Human Input**: Option to request human input during task execution.
- **Output Customization**: Define expected output format.

### Process Types

CrewAI supports two primary process types:

#### Sequential Process

Tasks are executed in order, with each task receiving context from the previous.

```python
from crewai import Crew, Process

crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, editing_task],
    process=Process.sequential,
    verbose=True
)

result = crew.kickoff()
```

**Execution Flow:**
```
Task 1 (Research) → Task 2 (Writing) → Task 3 (Editing)
    ↓                    ↓                    ↓
   Output             Output               Final Output
```

**When to Use:**
- Simple linear workflows
- Tasks with clear dependencies
- When order of execution matters
- Prototyping and simple automation

#### Hierarchical Process

A manager agent orchestrates worker agents, decomposing tasks and assigning them dynamically.

```python
manager = Agent(
    role="Project Manager",
    goal="Efficiently manage the team to achieve the project goals",
    backstory="Experienced project manager with a track record of delivering complex projects",
    allow_delegation=True,
    llm="gpt-4-turbo"
)

crew = Crew(
    agents=[researcher, writer, editor, designer],
    tasks=[project_task],
    manager_agent=manager,
    process=Process.hierarchical,
    verbose=True
)
```

**Execution Flow:**
```
Manager Agent
├── Decomposes project into subtasks
├── Assigns subtasks to appropriate agents
├── Monitors progress and quality
└── Synthesizes final output

Worker Agents
├── Receive assignments from manager
├── Execute their specific tasks
└── Report results back to manager
```

**When to Use:**
- Complex projects requiring planning
- When agent specialization is important
- Dynamic task allocation needed
- Quality control through manager oversight

### Role Assignment

CrewAI's role system provides structured agent definition:

```python
# Predefined Roles
from crewai.roles import Role

custom_role = Role(
    name="Data Scientist",
    goal="Extract insights from data",
    backstory="Expert in statistical analysis and machine learning",
    tools=[python_repl,数据分析_tools]
)
```

**Role Assignment Strategies:**
- **Single Role**: Each agent has one fixed role.
- **Context-Enhanced Roles**: Roles are augmented with task-specific context.
- **Dynamic Role Adjustment**: Agent behavior adjusts based on conversation.

### Task Delegation

Agents can delegate subtasks to other agents when `allow_delegation=True`.

```python
researcher = Agent(
    role="Lead Researcher",
    goal="Coordinate research efforts",
    allow_delegation=True,
    tools=[delegation_tool]
)

# The researcher agent can request help from other agents
# during task execution
```

**Delegation Flow:**
1. Agent receives a task.
2. Agent identifies sub-parts that require other expertise.
3. Agent delegates sub-tasks to appropriate agents.
4. Delegated agents return results.
5. Original agent synthesizes all results.

### Tool Sharing

Tools can be shared across agents at the crew level or agent level.

```python
from crewai_tools import tool

@tool("WebSearch")
def web_search(query: str) -> str:
    """Search the web for information."""
    return search_results

@tool("Calculator")
def calculator(expression: str) -> str:
    """Evaluate mathematical expressions."""
    return str(eval(expression))

# Agent-level tools
researcher = Agent(
    tools=[web_search],
    ...
)

# Crew-level tools (shared across all agents)
crew = Crew(
    share_tools=True,  # Share tools across agents
    ...
)
```

### Memory System

CrewAI provides three types of memory:

```python
from crewai.memory import ShortTermMemory, LongTermMemory, EntityMemory

crew = Crew(
    agents=[...],
    tasks=[...],
    memory=True,
    # Memory configuration
    short_term_memory=ShortTermMemory(
        max_size=1000,
        ttl=3600  # 1 hour
    ),
    long_term_memory=LongTermMemory(
        storage_path="./memory_store"
    ),
    entity_memory=EntityMemory(
        extract_entities=True
    )
)
```

**Memory Types:**
| Type | Purpose | Duration |
|------|---------|----------|
| **Short-term** | Recent conversation context | Minutes to hours |
| **Long-term** | Persistent knowledge across sessions | Days to permanent |
| **Entity** | Extracted entities and relationships | Task duration |

### Callbacks and Hooks

CrewAI provides lifecycle hooks for monitoring and customization:

```python
from crewai import Task, Agent, Crew

def task_start_callback(task):
    print(f"Task started: {task.description}")

def task_complete_callback(task, output):
    print(f"Task completed: {output}")

def agent_action_callback(agent, action):
    print(f"Agent {agent.role}: {action}")

# Apply callbacks
research_task = Task(
    description="Research topic",
    callback=task_complete_callback
)

crew = Crew(
    agents=[agent],
    tasks=[research_task],
    step_callback=agent_action_callback,
    task_callback=task_complete_callback
)
```

### CrewAI Enterprise

CrewAI Enterprise provides additional features for production deployments:

- **API Server**: RESTful API for Crew execution.
- **Monitoring Dashboard**: Real-time agent activity monitoring.
- **Analytics**: Performance metrics and cost tracking.
- **Versioning**: Crew configuration versioning.
- **Team Collaboration**: Shared workspace for crew development.
- **Managed Infrastructure**: Cloud-hosted execution environment.

### Comparison with LangGraph

| Aspect | CrewAI | LangGraph |
|--------|--------|-----------|
| **Abstraction** | High-level (agents, tasks, crews) | Low-level (graph, state, nodes) |
| **Learning Curve** | Lower | Higher |
| **Flexibility** | Constrained by patterns | Highly flexible |
| **State Management** | Implicit | Explicit (state schema) |
| **Parallelism** | Sequential by default | Natural parallel support |
| **Checkpointing** | Limited | Built-in checkpointing |
| **Streaming** | Streams final output | Granular streaming |
| **Custom Control Flow** | Sequential or hierarchical | Graph-based (any topology) |
| **Best For** | Rapid prototyping, role-based teams | Complex, custom workflows |

---

## AutoGen / AutoGen Studio

AutoGen is a multi-agent conversation framework developed by Microsoft Research. AutoGen Studio provides a visual interface for designing and debugging agent workflows.

### Core Concepts

#### ConversableAgent

The base class for all agents in AutoGen. It defines the fundamental communication interface.

```python
import autogen

class ConversableAgent:
    """
    Base class for agents that can converse with each other.
    """
    def __init__(self, name, system_message, llm_config, human_input_mode="NEVER"):
        self.name = name
        self.system_message = system_message
        self.llm_config = llm_config
        self.human_input_mode = human_input_mode
```

**Key Methods:**
- `send(message, recipient, request_reply)`: Send a message to another agent.
- `receive(message, sender, request_reply)`: Receive a message from another agent.
- `generate_reply(messages, sender)`: Generate a reply based on conversation history.
- `initiate_chat(recipient, message)`: Start a conversation with another agent.

#### AssistantAgent

An agent that uses an LLM to generate responses.

```python
assistant = autogen.AssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config={
        "config_list": [
            {
                "model": "gpt-4",
                "api_key": os.environ["OPENAI_API_KEY"]
            }
        ],
        "temperature": 0.7
    }
)
```

#### UserProxyAgent

An agent that can act on behalf of a human user, including executing code and providing feedback.

```python
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,  # Set to True for sandboxed execution
    }
)
```

### Two-Agent Chat

The simplest pattern: two agents converse to solve a task.

```python
# Create agents
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    code_execution_config={"work_dir": "coding"}
)

# Initiate conversation
user_proxy.initiate_chat(
    assistant,
    message="Write a Python script to calculate Fibonacci numbers"
)
```

**Chat Flow:**
```
UserProxy → Assistant: "Write a Python script..."
Assistant → UserProxy: "Here's a script..."
UserProxy → Assistant: Executes script, returns output
Assistant → UserProxy: "The result is..."
... continues until termination condition
```

### Group Chat

Multiple agents converse in a group chat managed by a GroupChatManager.

```python
from autogen import GroupChat, GroupChatManager

# Define multiple agents
planner = autogen.AssistantAgent(
    name="Planner",
    system_message="You are a planner. Break down tasks.",
    llm_config=llm_config
)

coder = autogen.AssistantAgent(
    name="Coder",
    system_message="You are a Python programmer.",
    llm_config=llm_config
)

critic = autogen.AssistantAgent(
    name="Critic",
    system_message="Review and critique code.",
    llm_config=llm_config
)

# Create group chat
groupchat = GroupChat(
    agents=[planner, coder, critic, user_proxy],
    messages=[],
    max_round=12,
    speaker_selection_method="auto",  # Options: auto, round_robin, random, manual
    allow_repeat_speaker=False,  # Prevent same speaker twice in a row
)

# Create manager
manager = GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config
)

# Start conversation
user_proxy.initiate_chat(
    manager,
    message="Create a web scraper for news articles"
)
```

**Speaker Selection Methods:**
| Method | Description |
|--------|-------------|
| `auto` | LLM decides next speaker based on context |
| `round_robin` | Speakers take turns in fixed order |
| `random` | Random selection |
| `manual` | Human selects next speaker |
| Custom | User-defined selection function |

### Function Calling and Tool Registration

AutoGen supports function calling for tool integration.

```python
# Define a function/tool
def search_web(query: str) -> str:
    """Search the web for information."""
    import requests
    response = requests.get(f"https://api.search.com?q={query}")
    return response.text

# Register with assistant
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={
        "config_list": [...],
        "functions": [{
            "name": "search_web",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        }]
    }
)

# Register function with user proxy
user_proxy.register_function(
    function_map={
        "search_web": search_web
    }
)
```

**Function Registration for Tool Calling:**
```python
# Alternative: function registration with tool calling
@user_proxy.register_for_execution()
@assistant.register_for_llm(description="Search the web")
def web_search(query: str) -> str:
    return search_api(query)
```

### Code Execution Sandbox

AutoGen provides configurable code execution environments:

```python
# Local execution
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
        "timeout": 60
    }
)

# Docker sandbox (secure)
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    code_execution_config={
        "work_dir": "coding",
        "use_docker": True,
        "docker_container_name": "autogen-sandbox",
        "timeout": 120
    }
)

# Custom executor
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    code_execution_config={
        "executor": custom_executor
    }
)
```

### Sequential, Parallel, and Debate Patterns

**Sequential Pattern:**
```python
# Chain of agent conversations
user_proxy.initiate_chat(
    assistant,
    message="Research topic",
    max_turns=3
)

user_proxy.initiate_chat(
    reviewer,
    message=f"Review this: {assistant.last_message()}",
    max_turns=2
)
```

**Parallel Pattern (using threading):**
```python
import threading

def chat_worker(assistant, task):
    user_proxy.initiate_chat(assistant, message=task)

threads = []
tasks = ["Task 1", "Task 2", "Task 3"]
for i, task in enumerate(tasks):
    t = threading.Thread(target=chat_worker, args=(assistants[i], task))
    threads.append(t)
    t.start()

for t in threads:
    t.join()
```

**Debate Pattern:**
```python
# Round 1: Both agents generate solutions
user_proxy.initiate_chat(agent_a, message=problem, max_turns=1)
user_proxy.initiate_chat(agent_b, message=problem, max_turns=1)

# Round 2: Agents critique each other's solutions
user_proxy.send(
    f"Agent B's solution: {agent_b.last_message()}. Critique it.",
    agent_a
)
user_proxy.send(
    f"Agent A's solution: {agent_a.last_message()}. Critique it.",
    agent_b
)

# Round 3: Judge evaluates
user_proxy.initiate_chat(
    judge,
    message=f"Agent A says: {agent_a.last_message()}\nAgent B says: {agent_b.last_message()}\nWhich is better?"
)
```

### AutoGen Studio UI

AutoGen Studio provides a visual interface for designing and debugging agents.

**Features:**
- **Workflow Designer**: Drag-and-drop interface for creating agent workflows.
- **Agent Configuration**: Visual agent creation and configuration.
- **Conversation Inspector**: Debug mode with full message history.
- **Performance Metrics**: Token usage, response times, cost tracking.
- **Preset Templates**: Pre-configured agent patterns.
- **Export/Import**: Share workflows as JSON.

**Launching AutoGen Studio:**
```bash
pip install autogen-studio
autogenstudio ui --port 8081
```

**UI Components:**
```
┌──────────────────────────────────────────────┐
│  Agent Builder    │  Workflow Canvas          │
│  ┌──────────┐     │  ┌────────────────────┐   │
│  │ Name     │     │  │ Agent A → Agent B  │   │
│  │ System   │     │  │    ↓                │   │
│  │ Model    │     │  │ Agent C            │   │
│  │ Tools    │     │  └────────────────────┘   │
│  └──────────┘     │                           │
├──────────────────┼──────────────────────────────┤
│  Conversation Log │  Metrics Dashboard          │
│  ┌────────────┐   │  ┌────────────────────┐     │
│  │ User: ...  │   │  │ Tokens: 1,234      │     │
│  │ Agent: ... │   │  │ Time: 12.3s        │     │
│  │ User: ...  │   │  │ Cost: $0.02        │     │
│  └────────────┘   │  └────────────────────┘     │
└──────────────────┴──────────────────────────────┘
```

### AutoGen .NET

AutoGen .NET brings the multi-agent framework to the .NET ecosystem.

```csharp
using Microsoft.AutoGen;

// Define agent
var assistant = new AssistantAgent(
    name: "assistant",
    systemMessage: "You are a helpful AI assistant",
    llmConfig: new LLMConfig
    {
        Model = "gpt-4",
        ApiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY")
    }
);

// User proxy
var userProxy = new UserProxyAgent(
    name: "user_proxy",
    humanInputMode: HumanInputMode.TERMINATE
);

// Start conversation
await userProxy.InitiateChatAsync(
    assistant,
    "Write a C# program to calculate prime numbers"
);
```

---

## Semantic Kernel

Semantic Kernel (SK) is an SDK by Microsoft that integrates LLMs with conventional programming languages.

### Core Architecture

```
Semantic Kernel
├── Kernel (central orchestrator)
├── Plugins (collections of functions)
│   ├── Native Functions (code)
│   └── Semantic Functions (prompts)
├── Planners (auto function chaining)
├── Memory (semantic memory / vector store)
├── Connectors (AI services, memory stores)
└── Filters (content moderation, logging)
```

### Kernel

The kernel is the central orchestrator that manages plugins, memory, and AI service connections.

```python
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

# Create kernel
kernel = Kernel()

# Add AI service
kernel.add_service(
    AzureChatCompletion(
        deployment_name="gpt-4",
        endpoint="https://your-endpoint.openai.azure.com",
        api_key="your-api-key"
    )
)

# The kernel connects all components
result = await kernel.invoke_prompt_async(
    "What is the capital of France?",
    stream=True
)
```

### Plugins

Plugins are collections of functions that the AI can use.

#### Native Functions (Code)

```python
from semantic_kernel.plugin import kernel_function

class EmailPlugin:
    @kernel_function(
        description="Send an email to a recipient",
        name="send_email"
    )
    def send_email(self, to: str, subject: str, body: str) -> str:
        """Send an email using the SMTP server."""
        # Email sending logic
        return f"Email sent to {to}"

    @kernel_function(
        description="Get unread email count",
        name="get_unread_count"
    )
    def get_unread_count(self) -> int:
        """Get the number of unread emails."""
        return 42

# Add plugin to kernel
email_plugin = EmailPlugin()
kernel.add_plugin(email_plugin, "Email")
```

#### Semantic Functions (Prompts)

```python
from semantic_kernel.functions import KernelFunctionFromPrompt

# Define prompt-based function
summarize_function = kernel.create_function_from_prompt(
    function_name="Summarize",
    plugin_name="TextPlugin",
    prompt="""Summarize the following text in 3 sentences:
    
    {{$input}}
    
    Summary:"""
)

# Or use YAML/Prompt template files
# Save as TextPlugin/Summarize/skprompt.txt
kernel.add_plugin_from_directory("./plugins/TextPlugin")
```

### Planners

Planners automatically chain functions together to accomplish complex goals.

#### HandlebarsPlanner

Uses Handlebars templates for plan generation.

```python
from semantic_kernel.planners import HandlebarsPlanner

planner = HandlebarsPlanner(
    kernel=kernel,
    prompt="""Create a plan to accomplish: {{$goal}}
    Available functions: {{$functions}}"""
)

plan = await planner.create_plan_async(
    goal="Send a summary email to the team"
)
```

#### FunctionCallingStepwisePlanner

Iterative planner that uses function calling to build plans step by step.

```python
from semantic_kernel.planners import FunctionCallingStepwisePlanner

planner = FunctionCallingStepwisePlanner(
    kernel=kernel,
    max_tokens=2000,
    max_iterations=10
)

result = await planner.execute_plan_async(
    goal="Research latest AI news and create a report"
)
```

**Planning Process:**
1. Decompose the goal into steps.
2. Match steps to available functions.
3. Execute functions in order.
4. Adapt plan based on intermediate results.
5. Return final output.

#### StepwisePlanner

The original planner that creates step-by-step execution plans.

```python
from semantic_kernel.planners import StepwisePlanner

planner = StepwisePlanner(kernel)

plan = planner.create_plan(goal="Compare stock prices of Apple and Microsoft")
```

### Connectors

Connectors bridge Semantic Kernel with external services.

```python
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    OpenAIChatCompletion
)
from semantic_kernel.connectors.ai.anthropic import AnthropicChatCompletion
from semantic_kernel.connectors.ai.google import GooglePalmChatCompletion
from semantic_kernel.connectors.memory.chroma import ChromaMemoryStore
from semantic_kernel.connectors.memory.qdrant import QdrantMemoryStore

# Multiple AI connectors
kernel.add_service(AzureChatCompletion(...))
kernel.add_service(OpenAIChatCompletion(...))
kernel.add_service(AnthropicChatCompletion(...))

# Memory connectors
kernel.add_memory_store(ChromaMemoryStore(persist_directory="./chroma"))
kernel.add_memory_store(QdrantMemoryStore(host="localhost", port=6333))
```

### Filters

Filters provide middleware-like capabilities for the kernel pipeline.

```python
from semantic_kernel.filters import (
    FunctionFilter,
    PromptFilter,
    KernelFilter
)

class LoggingFilter(FunctionFilter):
    async def on_function_invoking(self, context):
        print(f"Invoking function: {context.function.name}")

    async def on_function_invoked(self, context):
        print(f"Function completed: {context.function.name}")
        print(f"Duration: {context.duration}")

class SafetyFilter(PromptFilter):
    async def on_prompt_rendering(self, context):
        if contains_harmful_content(context.prompt):
            context.cancel = True

# Add filters to kernel
kernel.add_filter(LoggingFilter())
kernel.add_filter(SafetyFilter())
```

### Memory and Vector Store Integration

```python
from semantic_kernel.memory import SemanticTextMemory
from semantic_kernel.memory.memory_store import MemoryStore

# Configure memory
memory = SemanticTextMemory(
    storage=ChromaMemoryStore(),
    embeddings_generator=kernel.get_service("text_embedding")
)

# Store information
await memory.save_information_async(
    collection="facts",
    text="The Eiffel Tower is in Paris",
    id="fact_1"
)

# Retrieve information
result = await memory.search_async(
    collection="facts",
    query="Where is the Eiffel Tower?",
    limit=3
)
```

### OpenAI / Anthropic / Google Integration

```python
# OpenAI
kernel.add_service(
    OpenAIChatCompletion(
        model_id="gpt-4",
        api_key="...",
        org_id="..."
    )
)

# Azure OpenAI
kernel.add_service(
    AzureChatCompletion(
        deployment_name="gpt-4",
        endpoint="https://...",
        api_key="..."
    )
)

# Anthropic
kernel.add_service(
    AnthropicChatCompletion(
        model_id="claude-3-opus-20240229",
        api_key="..."
    )
)

# Google
kernel.add_service(
    GooglePalmChatCompletion(
        model_id="models/gemini-pro",
        api_key="..."
    )
)
```

---

## smolagents

smolagents is HuggingFace's framework for building AI agents that can write and execute code.

### Core Philosophy

smolagents distinguishes between two types of agents:
- **CodeAgent**: Writes and executes Python code to accomplish tasks.
- **ToolCallingAgent**: Uses tool calling APIs (function calling).

Both are built on a shared base architecture.

### CodeAgent

The CodeAgent writes Python code to solve tasks, then executes it.

```python
from smolagents import CodeAgent, HfApiModel

# Create a CodeAgent
agent = CodeAgent(
    model=HfApiModel(),
    tools=[],  # Custom tools
    add_base_tools=True,  # Include base tools like Python interpreter
    max_steps=10,
    verbosity_level=1
)

# Run the agent
result = agent.run(
    "Calculate the 50th Fibonacci number and check if it's prime."
)
```

**How CodeAgent Works:**
1. Receives a task.
2. Generates Python code to solve it.
3. Executes the code in a sandboxed Python environment.
4. If execution fails, the agent can fix the code and retry.
5. Returns the final result.

### ToolCallingAgent

The ToolCallingAgent uses function calling APIs to invoke tools.

```python
from smolagents import ToolCallingAgent, tool

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))

agent = ToolCallingAgent(
    model=HfApiModel(),
    tools=[search_web, calculate],
    max_steps=5
)

result = agent.run("What is 2^10? And search for 'AI news'")
```

### Multi-Agent Hierarchy

smolagents supports hierarchical multi-agent setups through ManagedAgent.

```python
from smolagents import CodeAgent, ManagedAgent, HfApiModel

# Create specialized sub-agents
research_agent = CodeAgent(
    model=HfApiModel(),
    tools=[search_tool],
    add_base_tools=True
)

code_agent = CodeAgent(
    model=HfApiModel(),
    tools=[python_repl],
    add_base_tools=True
)

# Wrap as managed agents
managed_researcher = ManagedAgent(
    agent=research_agent,
    name="researcher",
    description="Researchs topics online and returns findings"
)

managed_coder = ManagedAgent(
    agent=code_agent,
    name="coder",
    description="Writes and executes Python code"
)

# Main agent with sub-agents
main_agent = CodeAgent(
    model=HfApiModel(),
    managed_agents=[managed_researcher, managed_coder],
    tools=[],
    add_base_tools=True
)

# Run
result = main_agent.run(
    "Research quantum computing and write a simulation"
)
```

**Agent Hierarchy:**
```
Main Agent
├── Managed Agent: Researcher
│   └── Tools: web_search, fetch_page
└── Managed Agent: Coder
    └── Tools: python_repl, file_ops
```

### Tool Sharing

Tools can be defined and shared across agents:

```python
from smolagents import tool, CodeAgent, ToolCollection

# Define tools
@tool
def get_weather(location: str) -> str:
    """Get current weather for a location."""
    return f"Weather in {location}: Sunny, 72°F"

@tool
def get_time(timezone: str) -> str:
    """Get current time in a timezone."""
    return f"Time in {timezone}: 14:30"

# Create a tool collection
tools = ToolCollection([get_weather, get_time])

# Share across agents
agent1 = CodeAgent(tools=tools, ...)
agent2 = CodeAgent(tools=tools, ...)
```

### Hub Integration

smolagents integrates with the HuggingFace Hub for model and tool sharing:

```python
from smolagents import HfApiModel, CodeAgent, ToolCollection

# Use models from HuggingFace Hub
model = HfApiModel(
    model_id="Qwen/Qwen2.5-72B-Instruct",
    token="hf_..."
)

# Load tools from Hub
tools = ToolCollection.from_hub(
    "huggingface/tools-web-search",
    token="hf_..."
)

agent = CodeAgent(model=model, tools=tools)
```

### Gradio UI

smolagents includes a Gradio-based UI for interactive agent usage:

```python
from smolagents import CodeAgent, HfApiModel
from smolagents.gradio_ui import GradioUI

agent = CodeAgent(
    model=HfApiModel(),
    tools=[...],
    add_base_tools=True
)

# Launch UI
GradioUI(agent).launch()
```

---

## DSPy

DSPy is a framework for programmatically composing and optimizing language model programs.

### Core Philosophy

DSPy treats LMs as programmable components. Instead of hand-crafting prompts, you define **signatures** and **modules**, then use **optimizers** to automatically improve performance.

### Signatures

Signatures define the input/output behavior of an LM call.

```python
import dspy

# Define a signature
class GenerateAnswer(dspy.Signature):
    """Answer questions based on context."""
    
    context = dspy.InputField(desc="Contains relevant facts")
    question = dspy.InputField(desc="The question to answer")
    answer = dspy.OutputField(desc="The answer")

# Use the signature
module = dspy.Predict(GenerateAnswer)
result = module(context="Paris is the capital of France.", question="What is the capital of France?")
print(result.answer)  # "Paris"
```

**Signature Fields:**
- `InputField`: Marks input parameters with optional descriptions.
- `OutputField`: Marks output parameters.
- `desc`: Natural language description of the field.

### Modules

DSPy modules encapsulate LM calls with signatures.

```python
# Basic Predict module
predict = dspy.Predict(GenerateAnswer)

# Chain of Thought
cot = dspy.ChainOfThought(GenerateAnswer)

# ReAct (Reasoning + Acting)
react = dspy.ReAct(GenerateAnswer, tools=[search_tool])

# Multi-Chain Comparison
multi = dspy.MultiChainComparison(GenerateAnswer, M=3)  # Compare 3 chains

# Retrieve then Read (RAG)
rag = dspy.RetrieveThenRead(GenerateAnswer, retriever=retriever)

# Program of Thought (PoT)
pot = dspy.ProgramOfThought(GenerateAnswer)
```

### Optimizers (Teleprompters)

DSPy optimizers automatically tune prompts and few-shot examples.

#### BootstrapFewShot

```python
from dspy.teleprompt import BootstrapFewShot

# Configure optimizer
optimizer = BootstrapFewShot(
    metric=answer_accuracy,
    max_bootstrapped_demos=8,
    max_labeled_demos=16,
    max_rounds=3
)

# Compile (optimize) the program
compiled_program = optimizer.compile(
    student=program,
    teacher=program,
    trainset=training_data,
    valset=validation_data
)
```

#### MIPRO (Multi-Instruction Proposal Optimizer)

```python
from dspy.teleprompt import MIPRO

optimizer = MIPRO(
    metric=answer_accuracy,
    num_bayes_trials=10,
    num_candidates=5,
    verbose=True
)

compiled = optimizer.compile(
    program,
    trainset=training_data,
    num_trials=30
)
```

**MIPRO Features:**
- Proposes multiple instruction variations.
- Uses Bayesian optimization to find best combinations.
- Tunes both instructions and few-shot examples.
- More thorough optimization than BootstrapFewShot.

#### BayesianSignatureOptimizer

```python
from dspy.teleprompt import BayesianSignatureOptimizer

optimizer = BayesianSignatureOptimizer(
    metric=answer_accuracy,
    n_trials=20
)

compiled = optimizer.compile(
    program,
    trainset=training_data
)
```

### Compiled Programs

Once optimized, DSPy programs can be saved and loaded:

```python
# Save compiled program
compiled_program.save("optimized_rag_program.dspy")

# Load and use
loaded_program = dspy.load("optimized_rag_program.dspy")
result = loaded_program(context="...", question="...")
```

### Multi-Stage Pipelines

DSPy can compose multiple modules into pipelines:

```python
class RAGPipeline(dspy.Module):
    def __init__(self, retriever):
        super().__init__()
        self.retriever = retriever
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.check_facts = dspy.Predict(VerifyFacts)
    
    def forward(self, question):
        # Stage 1: Retrieve
        context = self.retriever(question)
        
        # Stage 2: Generate answer
        answer = self.generate_answer(context=context, question=question)
        
        # Stage 3: Verify facts
        verification = self.check_facts(answer=answer.answer, context=context)
        
        return dspy.Prediction(
            answer=answer.answer,
            confidence=verification.confidence
        )

# Create and optimize
pipeline = RAGPipeline(retriever=retriever)
optimized = optimizer.compile(pipeline, trainset=trainset)
```

### Evaluation

DSPy provides built-in evaluation tools:

```python
from dspy.evaluate import Evaluate

# Create evaluator
evaluator = Evaluate(
    devset=test_data,
    metric=answer_accuracy,
    num_threads=4,
    display_progress=True
)

# Evaluate
score = evaluator(compiled_program)
print(f"Accuracy: {score}")

# Custom metrics
def exact_match(example, prediction, trace=None):
    return example.answer.strip() == prediction.answer.strip()

evaluator = Evaluate(devset=test_data, metric=exact_match)
```

---

## OpenAI Agents SDK

The OpenAI Agents SDK is a lightweight framework for building agent-based applications using OpenAI models.

### Core Architecture

```
OpenAI Agents SDK
├── Agent (runnable agent instance)
├── Handoffs (agent-to-agent transfer)
├── Guardrails (input/output validation)
├── Tracing (observability)
└── Agent Loop (execution engine)
```

### Agent Loop

The core execution loop:

```python
from agents import Agent, Runner

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    model="gpt-4o"
)

result = Runner.run_sync(agent, "What is the capital of France?")
print(result.final_output)
```

**Agent Loop Process:**
1. Receive user input.
2. Process through model.
3. Execute any tool calls.
4. Continue until final output.
5. Return result.

### Handoffs

Agents can transfer control to specialized sub-agents:

```python
from agents import Agent, Runner

# Specialized agents
spanish_agent = Agent(
    name="Spanish Agent",
    instructions="You speak Spanish. Respond in Spanish.",
)

french_agent = Agent(
    name="French Agent",
    instructions="You speak French. Respond in French.",
)

# Main agent with handoff functions
def transfer_to_spanish():
    return spanish_agent

def transfer_to_french():
    return french_agent

main_agent = Agent(
    name="Main Agent",
    instructions="Route users to the right language specialist.",
    functions=[transfer_to_spanish, transfer_to_french]
)

result = Runner.run_sync(main_agent, "Parlez-vous français?")
```

### Guardrails

Guardrails validate inputs and outputs:

```python
from agents import Agent, Runner, Guardrail

# Input guardrail
class TopicGuardrail(Guardrail):
    async def check_input(self, agent, input_text):
        if "illegal" in input_text.lower():
            return GuardrailResult(blocked=True, message="Cannot process this request")
        return GuardrailResult(blocked=False)

# Output guardrail
class SafetyGuardrail(Guardrail):
    async def check_output(self, agent, output):
        if contains_harmful_content(output):
            return GuardrailResult(blocked=True, message="Output blocked")
        return GuardrailResult(blocked=False)

agent = Agent(
    name="Safe Agent",
    instructions="Be helpful and safe.",
    input_guardrails=[TopicGuardrail()],
    output_guardrails=[SafetyGuardrail()]
)
```

### Tracing

Built-in tracing for observability:

```python
from agents import Agent, Runner, trace

# Automatic tracing
with trace("my_app"):
    result = Runner.run_sync(agent, "Question 1")
    result2 = Runner.run_sync(agent, "Question 2")

# Manual tracing
trace.add_event("custom_event", {"key": "value"})
```

### Comparison with Older Function Calling

| Aspect | Agents SDK | Older Function Calling |
|--------|------------|----------------------|
| **Abstraction** | Full agent lifecycle | Raw function calling |
| **Handoffs** | Built-in | Manual implementation |
| **Guardrails** | First-class | Custom validation |
| **Tracing** | Built-in | None |
| **Multi-turn** | Managed loop | Manual conversation management |
| **State** | Context management | Manual state tracking |

---

## Anthropic MCP Client

Anthropic's Model Context Protocol (MCP) client enables standardized tool and resource access.

### Architecture

```
MCP Client
├── MCP Server Connection
├── Tool Discovery
├── Resource Access
├── Prompt Management
└── Sampling (optional)
```

### Key Concepts

- **MCP Server**: Provides tools and resources.
- **Client**: Connects to servers and exposes capabilities to LLMs.
- **Tools**: Functions the model can call.
- **Resources**: Data sources the model can read.
- **Prompts**: Reusable prompt templates.

### Client Implementation

```python
import asyncio
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters

async def main():
    # Connect to MCP server
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"]
    )
    
    async with ClientSession(server_params) as session:
        # Initialize connection
        await session.initialize()
        
        # List available tools
        tools = await session.list_tools()
        
        # Call a tool
        result = await session.call_tool(
            "search_web",
            {"query": "latest AI news"}
        )
        
        # Access resources
        resource = await session.read_resource("database://users")
```

---

## Google Agent Development Kit (ADK)

Google's ADK is a framework for building agents using Google's AI models and tools.

### Architecture

```
Google ADK
├── Agent (base agent class)
├── Tools (function, API, code execution)
├── Memory (conversation, session)
├── Auth (Google OAuth, service accounts)
└── Deployment (Cloud Run, Cloud Functions)
```

### Key Features

- **Gemini Integration**: First-class support for Gemini models.
- **Google Workspace Tools**: Gmail, Calendar, Drive integration.
- **Vertex AI Agent Builder**: Visual agent builder.
- **Enterprise Security**: IAM, VPC-SC, CMEK.
- **Streaming Support**: Real-time response streaming.

### Agent Definition

```python
from google.adk import Agent
from google.adk.tools import FunctionTool

def search_flights(origin, destination, date):
    """Search for flights."""
    return flight_api.search(origin, destination, date)

agent = Agent(
    model="gemini-2.0-flash",
    tools=[
        FunctionTool(search_flights),
        GoogleWorkspaceToolkit()
    ],
    system_instruction="You are a travel assistant."
)
```

---

## Amazon Bedrock Agents

Amazon Bedrock Agents enable building agents using foundation models on AWS.

### Architecture

```
Amazon Bedrock Agent
├── Foundation Model (Claude, Llama, Mistral, etc.)
├── Action Groups (Lambda functions)
├── Knowledge Base (RAG with vector store)
├── Guardrails (content filtering)
└── Orchestration (step-by-step planning)
```

### Key Features

- **Managed Infrastructure**: Fully managed by AWS.
- **Multi-Model Support**: Choose from various foundation models.
- **Knowledge Base**: Built-in RAG with Aurora/OpenSearch.
- **Guardrails**: Content filtering and safety controls.
- **CloudWatch Integration**: Monitoring and logging.
- **IAM Security**: AWS IAM integration.
- **Cost Management**: Pay-per-use pricing.

### Agent Configuration

```python
import boto3

bedrock_agent = boto3.client("bedrock-agent")

# Create agent
response = bedrock_agent.create_agent(
    agentName="customer-support-agent",
    foundationModel="anthropic.claude-3-sonnet-20240229",
    instruction="You are a helpful customer support agent.",
    actionGroups=[{
        "actionGroupName": "ticket-actions",
        "description": "Actions for managing support tickets",
        "actionGroupExecutor": {
            "lambda": "arn:aws:lambda:...:function:ticket-handler"
        },
        "functionSchema": {
            "functions": [
                {
                    "name": "create_ticket",
                    "description": "Create a support ticket",
                    "parameters": {
                        "customer_id": {"type": "string", "description": "Customer ID"},
                        "issue": {"type": "string", "description": "Issue description"}
                    }
                }
            ]
        }
    }],
    knowledgeBases=[{
        "knowledgeBaseId": "KB123",
        "description": "Product documentation"
    }]
)
```

### Knowledge Base Integration

```python
# Create knowledge base
bedrock_agent.create_knowledge_base(
    name="product-docs",
    roleArn="arn:aws:iam::...:role/bedrock-knowledge-base",
    knowledgeBaseConfiguration={
        "type": "VECTOR",
        "vectorKnowledgeBaseConfiguration": {
            "embeddingModelArn": "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v2"
        }
    },
    storageConfiguration={
        "type": "OPENSEARCH_SERVERLESS",
        "opensearchServerlessConfiguration": {
            "collectionArn": "arn:aws:aoss:...:collection/docs",
            "vectorIndexName": "bedrock-knowledge-base-index",
            "fieldMapping": {
                "metadataField": "metadata",
                "textField": "text"
            }
        }
    }
)
```

---

## Agno / Phidata

Agno (formerly Phidata) is a framework for building agent-as-function applications.

### Core Architecture

```
Agno Agent
├── Model (LLM backend)
├── Tools (function registry)
├── Knowledge Base (vector store)
├── Memory (conversation history)
├── Storage (persistent state)
└── Monitor (observability)
```

### Agent-as-Function

Agno treats agents as composable functions:

```python
from agno import Agent

# Define agent
agent = Agent(
    name="research-agent",
    model="gpt-4o",
    description="Research assistant that searches the web",
    instructions=[
        "Search the web for relevant information",
        "Cite your sources",
        "Provide comprehensive answers"
    ],
    tools=[web_search, fetch_page],
    show_tool_calls=True,
    markdown=True
)

# Call like a function
result = agent.run("What are the latest developments in quantum computing?")
print(result.content)
```

### Knowledge Base Integration

```python
from agno.knowledge import KnowledgeBase
from agno.vectordb import QdrantDb

# Create knowledge base
knowledge_base = KnowledgeBase(
    vector_db=QdrantDb(
        collection="documents",
        url="localhost:6333"
    ),
    num_documents=5
)

# Add documents
knowledge_base.load_directory("./documents/", formats=[".pdf", ".txt", ".md"])

# Create agent with knowledge base
agent = Agent(
    knowledge_base=knowledge_base,
    search_knowledge=True,
    add_references=True
)
```

### Tool Registry

```python
from agno.tools import tool

@tool(show_result=True)
def get_stock_price(symbol: str) -> dict:
    """Get current stock price for a symbol."""
    return {"symbol": symbol, "price": 150.25, "currency": "USD"}

@tool
def send_email(to: str, subject: str, body: str) -> bool:
    """Send an email."""
    email_client.send(to, subject, body)
    return True

# Tools are automatically registered and available to the agent
agent = Agent(tools=[get_stock_price, send_email])
```

---

## Pydantic AI

Pydantic AI is a framework for building typed, production-ready AI agents with validation.

### Core Architecture

```
Pydantic AI Agent
├── Model (OpenAI, Anthropic, Gemini, etc.)
├── Tools (typed function tools)
├── Result Types (Pydantic models)
├── Dependency Injection
├── Run Context
├── OpenTelemetry Tracing
└── Logfire Monitoring
```

### Typed Agents

Agents are strongly typed using Pydantic models:

```python
from pydantic_ai import Agent
from pydantic import BaseModel

# Define output type
class CityLocation(BaseModel):
    city: str
    country: str
    latitude: float
    longitude: float

# Agent with typed output
agent = Agent(
    'openai:gpt-4o',
    result_type=CityLocation,
    system_prompt="Extract location information from queries."
)

# Run returns typed result
result = agent.run_sync("What is the capital of France?")
print(result.data)  # CityLocation(city='Paris', country='France', ...)
print(result.data.latitude)  # 48.8566
```

### Dependency Injection

Pydantic AI supports dependency injection for clean code:

```python
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext

@dataclass
class Database:
    connection: str
    
    async def query(self, sql: str) -> list:
        return [{"name": "Alice"}, {"name": "Bob"}]

agent = Agent(
    'openai:gpt-4o',
    deps_type=Database,
    system_prompt="Query the database to answer questions."
)

@agent.tool
async def get_users(ctx: RunContext[Database]) -> str:
    """Get all users from the database."""
    users = await ctx.deps.query("SELECT * FROM users")
    return f"Found {len(users)} users"

# Inject dependencies at runtime
db = Database(connection="postgresql://...")
result = agent.run_sync("How many users do we have?", deps=db)
```

### Result Validation

```python
from pydantic import BaseModel, Field, validator

class SearchResult(BaseModel):
    title: str
    url: str = Field(..., pattern=r'^https?://')
    relevance_score: float = Field(..., ge=0, le=1)
    
    @validator('relevance_score')
    def score_must_be_valid(cls, v):
        if v < 0 or v > 1:
            raise ValueError('Score must be between 0 and 1')
        return v

agent = Agent(
    'openai:gpt-4o',
    result_type=list[SearchResult],
    system_prompt="Search and return structured results."
)

# Results are automatically validated
result = agent.run_sync("Search for Pydantic AI framework")
for item in result.data:
    print(f"{item.title}: {item.relevance_score}")
```

### OpenTelemetry Tracing

```python
from pydantic_ai import Agent
from opentelemetry import trace

# Automatic OpenTelemetry instrumentation
agent = Agent(
    'openai:gpt-4o',
    instrument=True  # Enable tracing
)

# Custom spans
tracer = trace.get_tracer(__name__)
with tracer.start_as_current_span("custom_span"):
    result = agent.run_sync("Some task")
```

### Logfire Monitoring

```python
from pydantic_ai import Agent
import logfire

# Pydantic Logfire integration
logfire.configure()

agent = Agent(
    'openai:gpt-4o',
    name="my-agent",
    instrument=True
)

# All agent runs are automatically logged to Logfire
result = agent.run_sync("Monitor this run")
```

---

## Comparison Table

### Architecture & Model Support

| Framework | Architecture | Model Support | Tool Support |
|-----------|-------------|---------------|--------------|
| **LangGraph** | Graph-based state machine | Any (via LangChain) | Via LangChain tools |
| **CrewAI** | Role-based agent teams | Any (OpenAI, Anthropic, etc.) | Built-in tool system |
| **AutoGen** | Conversational agents | OpenAI, Azure, custom | Function calling |
| **Semantic Kernel** | Kernel + plugins | OpenAI, Azure, Anthropic, Google | Native + semantic functions |
| **smolagents** | Code-first agents | HuggingFace, OpenAI | Code + tool calling |
| **DSPy** | Programmatic LM pipeline | Any (OpenAI, local, etc.) | Module-based |
| **OpenAI Agents SDK** | Agent loop + handoffs | OpenAI only | Function calling |
| **Anthropic MCP** | Client-server protocol | Anthropic, any MCP-compatible | Tool discovery protocol |
| **Google ADK** | Agent class | Gemini only | Google tools |
| **Amazon Bedrock** | Managed agent service | Bedrock models | Lambda actions groups |
| **Agno/Phidata** | Agent-as-function | Any | Function registry |
| **Pydantic AI** | Typed agents | OpenAI, Anthropic, Gemini | Typed tools |

### Memory, Streaming & Deployment

| Framework | Memory | Streaming | Deployment |
|-----------|--------|-----------|------------|
| **LangGraph** | Checkpointing (SQLite, Postgres) | Granular streaming | LangGraph Cloud, self-hosted |
| **CrewAI** | Short/long/entity memory | Final output streaming | API server, Enterprise |
| **AutoGen** | Conversation history | Message streaming | Custom, AutoGen Studio |
| **Semantic Kernel** | Vector store (Chroma, Qdrant) | Token streaming | Azure AI, self-hosted |
| **smolagents** | None built-in | Token streaming | Gradio UI, custom |
| **DSPy** | None | N/A | Any |
| **OpenAI Agents SDK** | Context management | Response streaming | OpenAI, custom |
| **Anthropic MCP** | Server-managed | Streaming support | Custom |
| **Google ADK** | Conversation, session | Real-time streaming | Cloud Run, Cloud Functions |
| **Amazon Bedrock** | Knowledge base (RAG) | Streaming responses | AWS managed |
| **Agno/Phidata** | Vector DB, conversation | Token streaming | Any |
| **Pydantic AI** | Run context | Token streaming | Any |

### Learning Curve & Community

| Framework | Learning Curve | Community | License |
|-----------|---------------|-----------|---------|
| **LangGraph** | Steep | Large (LangChain ecosystem) | MIT |
| **CrewAI** | Moderate | Growing rapidly | MIT |
| **AutoGen** | Moderate | Large (Microsoft) | MIT |
| **Semantic Kernel** | Moderate | Large (Microsoft) | MIT |
| **smolagents** | Low | Growing (HuggingFace) | Apache 2.0 |
| **DSPy** | Moderate | Active research community | MIT |
| **OpenAI Agents SDK** | Low | Large (OpenAI) | MIT |
| **Anthropic MCP** | Moderate | Growing | MIT |
| **Google ADK** | Moderate | New (Google) | Apache 2.0 |
| **Amazon Bedrock** | High (AWS required) | Large (AWS) | Proprietary |
| **Agno/Phidata** | Low | Small | MIT |
| **Pydantic AI** | Low | Growing | MIT |

### Feature Comparison Matrix

| Feature | LangGraph | CrewAI | AutoGen | Semantic Kernel | smolagents | DSPy | OpenAI SDK | MCP | ADK | Bedrock | Agno | Pydantic AI |
|---------|-----------|--------|---------|-----------------|------------|------|------------|-----|-----|---------|------|-------------|
| Multi-agent | ✅ | ✅ | ✅ | ✅ | ✅ | ⬜ | ✅ | ✅ | ✅ | ✅ | ✅ | ⬜ |
| State management | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ✅ | ⬜ | ⬜ | ⬜ |
| Graph workflow | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| Human-in-loop | ✅ | ⬜ | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ✅ | ⬜ | ⬜ |
| Code execution | ⬜ | ⬜ | ✅ | ⬜ | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | ✅ | ⬜ | ⬜ |
| RAG support | ✅ | ✅ | ✅ | ✅ | ⬜ | ✅ | ⬜ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Function calling | ✅ | ✅ | ✅ | ✅ | ✅ | ⬜ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Streaming | ✅ | ⬜ | ✅ | ✅ | ✅ | ⬜ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Checkpointing | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| Observability | ⬜ | ⬜ | ✅ | ✅ | ⬜ | ✅ | ✅ | ⬜ | ✅ | ✅ | ✅ | ✅ |
| UI/Dashboard | ⬜ | ✅ | ✅ | ⬜ | ✅ | ⬜ | ⬜ | ⬜ | ✅ | ✅ | ⬜ | ⬜ |
| Enterprise ready | ✅ | ✅ | ⬜ | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | ✅ | ✅ | ⬜ | ⬜ |
| Multi-model | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⬜ | ✅ | ⬜ | ✅ | ✅ | ✅ |
| Local models | ✅ | ✅ | ⬜ | ✅ | ✅ | ✅ | ⬜ | ✅ | ⬜ | ⬜ | ✅ | ✅ |
| Cost tracking | ⬜ | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ✅ | ⬜ | ✅ |

### Cost Considerations

| Framework | Base Cost | Scaling Cost | Infrastructure |
|-----------|-----------|--------------|----------------|
| **LangGraph** | Free (OSS) | LLM API costs + compute | Self-hosted or Cloud |
| **CrewAI** | Free (OSS) | LLM API costs + compute | Self-hosted or Enterprise |
| **AutoGen** | Free (OSS) | LLM API costs + compute | Self-hosted |
| **Semantic Kernel** | Free (OSS) | LLM API costs + compute | Self-hosted or Azure |
| **smolagents** | Free (OSS) | LLM API costs + compute | Self-hosted |
| **DSPy** | Free (OSS) | LLM API costs (optimization) | Self-hosted |
| **OpenAI Agents SDK** | Free (OSS) | OpenAI API costs | Self-hosted |
| **Anthropic MCP** | Free (OSS) | LLM API costs | Self-hosted |
| **Google ADK** | Free (OSS) | Vertex AI costs | Cloud Run/GCF |
| **Amazon Bedrock** | Pay-per-use | Model + KB + Lambda | AWS managed |
| **Agno/Phidata** | Free (OSS) | LLM API costs + compute | Self-hosted |
| **Pydantic AI** | Free (OSS) | LLM API costs | Self-hosted |

### Choosing the Right Framework

| Use Case | Recommended Framework |
|----------|----------------------|
| Complex, stateful workflows | LangGraph |
| Rapid prototyping, role-based teams | CrewAI |
| Research, experimental multi-agent | AutoGen |
| Enterprise .NET ecosystem | Semantic Kernel |
| Code-focused agents | smolagents |
| Optimizing LM programs | DSPy |
| Simple, OpenAI-only apps | OpenAI Agents SDK |
| Standardized tool protocol | Anthropic MCP |
| Google Cloud ecosystem | Google ADK |
| AWS enterprise deployment | Amazon Bedrock |
| Lightweight, composable agents | Agno/Phidata |
| Type-safe, validated agents | Pydantic AI |

---

## References

1. LangChain. (2024). "LangGraph Documentation." https://langchain-ai.github.io/langgraph/
2. CrewAI. (2024). "CrewAI Documentation." https://docs.crewai.com/
3. Microsoft. (2024). "AutoGen Documentation." https://microsoft.github.io/autogen/
4. Microsoft. (2024). "Semantic Kernel Documentation." https://learn.microsoft.com/en-us/semantic-kernel/
5. HuggingFace. (2024). "smolagents Documentation." https://huggingface.co/docs/smolagents
6. Stanford NLP. (2024). "DSPy Documentation." https://dspy-docs.vercel.app/
7. OpenAI. (2024). "Agents SDK Documentation." https://platform.openai.com/docs/agents
8. Anthropic. (2024). "Model Context Protocol." https://docs.anthropic.com/claude/docs/model-context-protocol
9. Google. (2024). "Agent Development Kit." https://cloud.google.com/agent-development-kit
10. AWS. (2024). "Amazon Bedrock Agents." https://docs.aws.amazon.com/bedrock/latest/userguide/agents.html
11. Agno. (2024). "Agno Documentation." https://docs.agno.com/
12. Pydantic. (2024). "Pydantic AI Documentation." https://ai.pydantic.dev/

---
**See also:**
- [AI in Education 2026 Frontier — The AI-Tutor Wave, the Skepticism, and the Agentic Pivot](11-AI-Applications/16-AI-Education-2026-Frontier.md)
- [08 — Agentic Services Pricing: The New Category for AI Agent Monetization](16-AI-Business-Models-Playbooks/08-Agentic-Services-Pricing.md)
- [Agentic Browser Automation & Computer Use: A 2026 Overview](26-Browser-Based-AI/46-Agentic-Browser-Automation-Computer-Use/01-Overview.md)
