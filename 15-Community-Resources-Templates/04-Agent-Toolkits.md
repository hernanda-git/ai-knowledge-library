# 04 — Agent Toolkits & Frameworks

> **Purpose:** Comprehensive comparison of agent-building frameworks, toolkits, and orchestration platforms.
> **Last Updated:** 2026-06-12
> **Maintainer:** AI Knowledge Base Team

---

## Table of Contents

1. [Introduction](#introduction)
2. [Framework Comparison Overview](#framework-comparison-overview)
3. [LangChain & LangGraph](#langchain--langgraph)
4. [CrewAI](#crewai)
5. [AutoGen (Microsoft)](#autogen-microsoft)
6. [Semantic Kernel (Microsoft)](#semantic-kernel-microsoft)
7. [Eliza (Vercel)](#eliza-vercel)
8. [Haystack (deepset)](#haystack-deepset)
9. [Dify](#dify)
10. [Flowise](#flowise)
11. [n8n](#n8n)
12. [Other Notable Frameworks](#other-notable-frameworks)
13. [Decision Flowchart](#decision-flowchart)
14. [Benchmarking & Evaluation](#benchmarking--evaluation)
15. [Further Reading](#further-reading)

---

## Introduction

The agent toolkit ecosystem has exploded since 2023. This document provides a structured comparison of the major frameworks, helping practitioners choose the right tool for their use case.

### Framework Categories

| Category | Examples | Best For |
|----------|----------|----------|
| **Agent Orchestration** | LangChain, LangGraph, Semantic Kernel | Complex multi-step agent workflows |
| **Multi-Agent Systems** | AutoGen, CrewAI, AgentScope | Multiple agents collaborating |
| **RAG Platforms** | Haystack, LlamaIndex, Canopy | Retrieval-augmented generation |
| **Low-Code/No-Code** | Dify, Flowise, n8n, Langflow | Visual workflow builders |
| **Conversational** | Eliza, Rasa, Botpress | Chatbots and assistants |
| **Enterprise** | Semantic Kernel, Cognigy, Kore.ai | Enterprise integration |

---

## Framework Comparison Overview

### Summary Table

| Framework | Type | Language | License | GitHub Stars | Key Strength |
|-----------|------|----------|---------|-------------|--------------|
| LangChain/LangGraph | Orchestration | Python, JS | MIT | ~98K / ~9K | Ecosystem size, flexibility |
| CrewAI | Multi-Agent | Python | MIT | ~25K | Simple multi-agent orchestration |
| AutoGen | Multi-Agent | Python | MIT | ~36K | Deep multi-agent conversations |
| Semantic Kernel | Orchestration | C#, Python, Java | MIT | ~22K | Enterprise .NET integration |
| Eliza | Agent Framework | TypeScript | MIT | ~8K | Vercel/Edge ecosystem |
| Haystack | RAG Platform | Python | Apache-2.0 | ~18K | Production RAG |
| Dify | Low-Code Platform | Python, TS | Apache-2.0 | ~55K | Easiest visual builder |
| Flowise | Low-Code Platform | TypeScript | Apache-2.0 | ~32K | LangChain visual wrapper |
| n8n | Workflow Automation | TypeScript | Sustainable Use License | ~50K | General automation + AI |
| LlamaIndex | RAG/Data Framework | Python | MIT | ~37K | Data indexing & retrieval |

### Architecture Comparison

| Framework | Agent Model | State Management | Memory | Tool Calling | Multi-Model |
|-----------|-------------|-----------------|--------|--------------|-------------|
| LangChain | Runnable + Graph | Explicit state | Memory classes | Built-in tools | Yes |
| LangGraph | StateGraph | Custom state | Checkpointing | LangChain tools | Yes |
| CrewAI | Agent + Task | Sequential/async | Context window | Custom tools | Yes |
| AutoGen | Agent + AssistantAgent | Conversational | History | Function calling | Yes |
| Semantic Kernel | Plugin-based | Context objects | Semantic memory | Native/OpenAPI | Yes |
| Haystack | Pipeline | Component state | DocumentStore | Custom components | Pluggable |
| Eliza | Actor model | Stream state | Buffer memory | Tool registry | Pluggable |
| Dify | Workflow nodes | Variable system | Conversation | Built-in tools | Yes |
| Flowise | Visual graph | Chatflow state | Vector memory | LangChain tools | Yes |
| n8n | Workflow DAG | Node state | n8n memory | Node credentials | HTTP |

---

## LangChain & LangGraph

### Overview

LangChain is the most widely adopted LLM framework, providing a comprehensive ecosystem for building context-aware reasoning applications. LangGraph extends LangChain with graph-based state management for complex agent workflows.

**Repository:** https://github.com/langchain-ai/langchain  
**Documentation:** https://python.langchain.com/  
**Stars:** ~98K (LangChain), ~9K (LangGraph)  
**Latest Version:** 0.3.x (LangChain), 0.2.x (LangGraph)

### Core Concepts

```python
# LangChain: Basic LLM Chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template(
    "Answer the question: {question}"
)
model = ChatOpenAI(model="gpt-4o")
output_parser = StrOutputParser()

chain = prompt | model | output_parser
result = chain.invoke({"question": "What is RAG?"})
print(result)
```

### Agent Execution with Tool Calling

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    # Implementation
    return f"Results for: {query}"

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))

tools = [search_web, calculate]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with web search and calculation abilities."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(ChatOpenAI(model="gpt-4o"), tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_executor.invoke({"input": "What is 25 * 4 + 10? Search for population of Tokyo."})
print(result["output"])
```

### LangGraph: Stateful Multi-Step Agents

```python
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_step: str

def router(state: AgentState):
    if state["next_step"] == "research":
        return "research_node"
    elif state["next_step"] == "write":
        return "write_node"
    else:
        return END

def research_node(state: AgentState):
    # Perform research
    return {
        "messages": [AIMessage(content="Research complete.")],
        "next_step": "write"
    }

def write_node(state: AgentState):
    # Write based on research
    return {
        "messages": [AIMessage(content="Writing complete.")],
        "next_step": "review"
    }

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("research_node", research_node)
workflow.add_node("write_node", write_node)
workflow.set_conditional_edge_source("router", router)
workflow.set_entry_point("research_node")
workflow.add_edge("write_node", END)

app = workflow.compile()
```

### Key Features

| Feature | Description |
|---------|-------------|
| **Model agnostic** | 100+ LLM providers supported |
| **Rich tool ecosystem** | 700+ integrations |
| **LangSmith** | Built-in observability and debugging |
| **LangServe** | Deploy chains as REST APIs |
| **LangGraph** | Graph-based state machines for agents |
| **Hub** | Community prompt sharing |
| **Templates** | Production-ready application templates |

### Pros & Cons

| Pros | Cons |
|------|------|
| Largest ecosystem and community | Abstraction complexity can be overwhelming |
| Excellent documentation and tutorials | Rapid API changes (breaking changes common) |
| Flexible — works for simple and complex use cases | Heavy dependency tree |
| Strong enterprise adoption | Debugging can be challenging |
| LangSmith provides great observability | Performance overhead from abstraction layers |

### When to Use

✅ **Use LangChain when:** You need broad model/tool compatibility, have complex chains, or want production-ready observability.  
✅ **Use LangGraph when:** You need cyclical agent loops, persistent state, or human-in-the-loop workflows.  
❌ **Avoid when:** You need minimal dependencies, ultra-low latency, or a simple one-off script.

---

## CrewAI

### Overview

CrewAI provides a simple, high-level API for creating teams of AI agents that collaborate on tasks. It emphasizes role-based agent design with minimal boilerplate.

**Repository:** https://github.com/crewAIInc/crewAI  
**Documentation:** https://docs.crewai.com/  
**Stars:** ~25K  
**Latest Version:** 0.70.0+

### Quick Start

```python
from crewai import Agent, Task, Crew, Process

# Define agents
researcher = Agent(
    role="Senior Research Analyst",
    goal="Uncover cutting-edge developments in AI",
    backstory="You are an expert research analyst with 15 years of experience.",
    allow_delegation=False,
    verbose=True
)

writer = Agent(
    role="Technical Writer",
    goal="Create compelling technical content",
    backstory="You are a skilled writer who makes complex topics accessible.",
    allow_delegation=True,
    verbose=True
)

# Define tasks
research_task = Task(
    description="Research the latest developments in {{topic}}",
    expected_output="A detailed research brief with 5 key findings",
    agent=researcher
)

write_task = Task(
    description="Write a blog post based on the research findings",
    expected_output="A 1000-word blog post in markdown format",
    agent=writer
)

# Create crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,  # or Process.hierarchical
    verbose=True
)

# Execute
result = crew.kickoff(inputs={"topic": "multi-agent systems"})
print(result)
```

### Key Features

| Feature | Description |
|---------|-------------|
| **Role-based agents** | Define agents by role, goal, and backstory |
| **Task delegation** | Agents can delegate subtasks to others |
| **Sequential & hierarchical processes** | Choose workflow patterns |
| **Built-in memory** | Short-term, long-term, and entity memory |
| **Tool integration** | Custom tool creation and LangChain tools |
| **Human input** | Optional human-in-the-loop checkpoints |

### Pros & Cons

| Pros | Cons |
|------|------|
| Very easy to get started | Limited to Python-only |
| Role-based design is intuitive | Less flexibility for complex workflows |
| Good documentation | Smaller community than LangChain |
| Built-in memory management | State management is less explicit |

### When to Use

✅ **Use when:** You want to quickly prototype multi-agent systems with clear role separation.  
❌ **Avoid when:** You need fine-grained control over agent behavior or graph-level state.

---

## AutoGen (Microsoft)

### Overview

AutoGen is Microsoft's framework for building multi-agent conversations. It enables multiple LLM agents, tool agents, and human agents to interact through structured conversations.

**Repository:** https://github.com/microsoft/autogen  
**Documentation:** https://microsoft.github.io/autogen/  
**Stars:** ~36K  
**Latest Version:** 0.7.x (v2 in early access)

### Quick Start

```python
import autogen

# Configure LLM
config_list = [
    {"model": "gpt-4o", "api_key": "your-api-key"}
]

# Create agents
assistant = autogen.AssistantAgent(
    name="Assistant",
    llm_config={"config_list": config_list},
    system_message="You are a helpful AI assistant."
)

user_proxy = autogen.UserProxyAgent(
    name="User",
    human_input_mode="NEVER",  # or "ALWAYS" or "TERMINATE"
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False
    }
)

# Define a function tool
@user_proxy.register_for_execution()
@assistant.register_for_llm(description="Search the web")
def web_search(query: str) -> str:
    """Mock web search"""
    return f"Results for: {query}"

# Start conversation
user_proxy.initiate_chat(
    assistant,
    message="Research the latest trends in AI agents and summarize them."
)
```

### Group Chat (Multi-Agent)

```python
from autogen import GroupChat, GroupChatManager

# Create specialized agents
researcher = autogen.AssistantAgent(
    name="Researcher",
    system_message="You are a researcher. Find and analyze information."
)

critic = autogen.AssistantAgent(
    name="Critic",
    system_message="You are a critic. Review and provide constructive feedback."
)

writer = autogen.AssistantAgent(
    name="Writer",
    system_message="You are a writer. Synthesize information into clear content."
)

# Create group chat
groupchat = GroupChat(
    agents=[researcher, critic, writer],
    messages=[],
    max_round=10,
    speaker_selection_method="auto"  # or "round_robin" or "random"
)

manager = GroupChatManager(
    groupchat=groupchat,
    llm_config={"config_list": config_list}
)

user_proxy = autogen.UserProxyAgent(
    name="User",
    human_input_mode="TERMINATE"
)

user_proxy.initiate_chat(
    manager,
    message="Create a report on advancements in multimodal AI"
)
```

### Key Features

| Feature | Description |
|---------|-------------|
| **Multi-agent conversations** | Structured agent-to-agent and agent-human dialogs |
| **Code execution** | Built-in sandboxed code execution |
| **Tool/function integration** | Register function tools for agents |
| **Group chat management** | Automated speaker selection and turn management |
| **Human-in-the-loop** | Multiple human input modes |
| **Asynchronous messaging** | Non-blocking agent communication |

### Pros & Cons

| Pros | Cons |
|------|------|
| Excellent multi-agent conversation management | Steeper learning curve |
| Code execution is well-integrated | Focused mainly on conversational patterns |
| Strong academic/enterprise backing | Configuration can be verbose |
| Flexible agent types (LLM, Tool, Human) | Limited visual tooling |

### When to Use

✅ **Use when:** You need structured multi-agent conversations, code-executing agents, or human-in-the-loop workflows.  
❌ **Avoid when:** You need simple single-agent chains or low-code visual building.

---

## Semantic Kernel (Microsoft)

### Overview

Semantic Kernel is an SDK that integrates LLMs with conventional programming languages (C#, Python, Java). It's designed for enterprise applications that need to combine AI with existing code.

**Repository:** https://github.com/microsoft/semantic-kernel  
**Documentation:** https://learn.microsoft.com/en-us/semantic-kernel/  
**Stars:** ~22K  
**Latest Version:** 1.20+ (Python), 1.15+ (C#)

### Quick Start (C#)

```csharp
using Microsoft.SemanticKernel;

var builder = Kernel.CreateBuilder();
builder.AddOpenAIChatCompletion("gpt-4o", "your-api-key");

var kernel = builder.Build();

// Import a native plugin
var plugin = kernel.ImportPluginFromType<TimePlugin>();

// Run a prompt with function calling
var result = await kernel.InvokePromptAsync(
    "Today is {{TimePlugin.Today}}. What notable events happened on this date?"
);
Console.WriteLine(result);
```

### Quick Start (Python)

```python
import asyncio
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import kernel_function

class SearchPlugin:
    @kernel_function(description="Search the web")
    async def search(self, query: str) -> str:
        # Implementation
        return f"Results for: {query}"

async def main():
    kernel = Kernel()
    kernel.add_service(OpenAIChatCompletion(ai_model_id="gpt-4o"))
    kernel.add_plugin(SearchPlugin(), plugin_name="search")
    
    result = await kernel.invoke_prompt(
        function_name="search_example",
        prompt="{{search.search $query}}",
        arguments={"query": "AI trends 2026"}
    )
    print(result)

asyncio.run(main())
```

### Key Features

| Feature | Description |
|---------|-------------|
| **Planners** | Automatic function orchestration |
| **Native functions** | Turn existing C#/Python code into AI-callable functions |
| **Semantic memory** | Vector-based memory with embeddings |
| **OpenAPI integration** | Import REST APIs as plugins |
| **Enterprise-grade** | Azure integration, managed identity, RBAC |
| **Multi-language** | C#, Python, Java (same patterns) |

### Pros & Cons

| Pros | Cons |
|------|------|
| Excellent .NET enterprise integration | Smaller community than LangChain |
| Strong typing and IDE support | Python/Java lag behind C# features |
| Built-in telemetry with AppInsights | Documentation can be Microsoft-centric |
| Planner orchestration is powerful | Fewer community plugins and tools |

### When to Use

✅ **Use when:** You're building enterprise .NET applications, need Azure integration, or want to make existing code AI-callable.  
❌ **Avoid when:** You're in a polyglot stack without .NET or need maximum community plugin availability.

---

## Eliza (Vercel)

### Overview

Eliza is a lightweight, TypeScript-first AI agent framework built by Vercel. It's designed for the edge computing paradigm with streaming, serverless functions, and minimal overhead.

**Repository:** https://github.com/ai-community/eliza (verify)  
**Documentation:** https://eliza.vercel.ai/  
**Stars:** ~8K  

### Quick Start

```typescript
import { Eliza, createTool } from "@ai-community/eliza";

const searchTool = createTool({
  name: "search",
  description: "Search the web",
  parameters: {
    query: { type: "string", description: "Search query" }
  },
  execute: async ({ query }) => {
    // Implementation
    return `Results for: ${query}`;
  }
});

const agent = new Eliza({
  model: "gpt-4o",
  tools: [searchTool],
  systemPrompt: "You are a helpful assistant.",
  streaming: true
});

const response = await agent.chat("What's the weather in Tokyo?");
console.log(response);
```

### Key Features

| Feature | Description |
|---------|-------------|
| **Streaming-first** | Built for real-time streaming responses |
| **Edge-ready** | Works on Vercel Edge, Cloudflare Workers |
| **TypeScript native** | Full type safety across the framework |
| **Minimal dependencies** | Lightweight compared to LangChain |
| **Tool registry** | Simple type-safe tool definitions |

### Pros & Cons

| Pros | Cons |
|------|------|
| Great developer experience (TypeScript) | Smaller ecosystem |
| Excellent streaming support | Primarily TypeScript/Node.js |
| Minimal bundle size | Fewer integrations |
| Perfect for Vercel deployment | Less suitable for complex orchestrations |

### When to Use

✅ **Use when:** You're building TypeScript/Node.js applications, need edge deployment, or want minimal overhead.  
❌ **Avoid when:** You need Python ecosystem or complex multi-agent workflows.

---

## Haystack (deepset)

### Overview

Haystack is a production-ready framework for building RAG (Retrieval-Augmented Generation) pipelines. It provides modular components for indexing, retrieval, and generation.

**Repository:** https://github.com/deepset-ai/haystack  
**Documentation:** https://docs.haystack.deepset.ai/  
**Stars:** ~18K  
**Latest Version:** 2.8+

### Quick Start

```python
from haystack import Pipeline
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.document_stores.in_memory import InMemoryDocumentStore

# Create document store and index documents
document_store = InMemoryDocumentStore()
document_store.write_documents([
    {"content": "Python is a programming language.", "meta": {"source": "wiki"}},
    {"content": "RAG stands for Retrieval-Augmented Generation.", "meta": {"source": "guide"}},
])

# Build pipeline
prompt_template = """
Answer the question based on the provided context.

Context:
{% for doc in documents %}
{{ doc.content }}
{% endfor %}

Question: {{ question }}
Answer:
"""

rag_pipeline = Pipeline()
rag_pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=document_store))
rag_pipeline.add_component("prompt_builder", PromptBuilder(template=prompt_template))
rag_pipeline.add_component("llm", OpenAIGenerator(model="gpt-4o"))

rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")

# Query
result = rag_pipeline.run({
    "retriever": {"query": "What is RAG?"},
    "prompt_builder": {"question": "What is RAG?"}
})
print(result["llm"]["replies"][0])
```

### Key Features

| Feature | Description |
|---------|-------------|
| **Modular pipeline architecture** | Composable components |
| **Multiple retrieval methods** | BM25, embedding, hybrid, sparse |
| **Document stores** | Weaviate, Pinecone, Qdrant, Elasticsearch |
| **File converters** | PDF, Docx, HTML, Markdown parsers |
| **Evaluation framework** | Built-in RAG evaluation metrics |
| **Tracing & monitoring** | OpenTelemetry integration |

### Pros & Cons

| Pros | Cons |
|------|------|
| Best-in-class RAG pipeline support | Less focused on agent/orchestration |
| Excellent production readiness | Fewer LLM integrations than LangChain |
| Great document processing | Agents are a newer feature (v2.8+) |
| Strong evaluation framework | Smaller community |

### When to Use

✅ **Use when:** You're building production RAG systems with complex document processing.  
❌ **Avoid when:** You need general-purpose agent orchestration or multi-agent systems.

---

## Dify

### Overview

Dify is an open-source LLM application development platform with a visual workflow designer. It supports RAG pipelines, agent capabilities, and workflow automation through a drag-and-drop interface.

**Repository:** https://github.com/langgenius/dify  
**Documentation:** https://docs.dify.ai/  
**Stars:** ~55K  
**Latest Version:** 1.0.x

### Key Features

| Feature | Description |
|---------|-------------|
| **Visual workflow builder** | Drag-and-drop LLM application design |
| **RAG engine** | Built-in document indexing and retrieval |
| **Agent capabilities** | Tool calling, multi-step reasoning |
| **Model providers** | OpenAI, Anthropic, Azure, local models |
| **API management** | Rate limiting, key management, logging |
| **Conversation history** | Built-in memory and context management |
| **Plugin system** | Extensible with community plugins |
| **App templates** | Pre-built application templates |

### Quick Start (API)

```python
import requests

# Dify API interaction
response = requests.post(
    "http://localhost:5001/v1/chat-messages",
    json={
        "inputs": {},
        "query": "What is machine learning?",
        "response_mode": "blocking",
        "conversation_id": "",
        "user": "user-123"
    },
    headers={"Authorization": "Bearer your-api-key"}
)
print(response.json()["answer"])
```

### Pros & Cons

| Pros | Cons |
|------|------|
| Easiest visual builder for non-developers | Less flexible than code-based frameworks |
| Quick prototyping | Self-hosting requires resources |
| Built-in RAG and agent support | Limited customization at scale |
| Active community (55K stars) | API changes can be breaking |

### When to Use

✅ **Use when:** You want rapid prototyping, need a visual builder, or are building internal tools.  
❌ **Avoid when:** You need fine-grained control or are building a production system at scale.

---

## Flowise

### Overview

Flowise is a low-code/no-code visual LLM flow builder built on top of LangChain. It provides a drag-and-drop interface for creating LLM applications without writing code.

**Repository:** https://github.com/FlowiseAI/Flowise  
**Documentation:** https://docs.flowiseai.com/  
**Stars:** ~32K  
**Latest Version:** 2.1.x

### Key Features

| Feature | Description |
|---------|-------------|
| **Visual node editor** | Drag-and-drop LangChain components |
| **LangChain under the hood** | Access to full LangChain ecosystem |
| **Multiple LLM support** | 20+ model providers |
| **Vector database support** | Pinecone, Weaviate, Qdrant, Chroma |
| **Chat & API endpoints** | Built-in deployment options |
| **Custom components** | Build custom nodes with JavaScript |
| **Authentication** | API key management |

### Pros & Cons

| Pros | Cons |
|------|------|
| Visual access to LangChain ecosystem | Limited compared to writing code |
| Quick prototyping | Can be slow for complex flows |
| No-code friendly | Debugging visual flows is harder |
| Self-hostable | Documentation can be sparse |

### When to Use

✅ **Use when:** You want visual LangChain development or no-code LLM app building.  
❌ **Avoid when:** You need production-scale performance or complex custom logic.

---

## n8n

### Overview

n8n is a workflow automation platform that has expanded into AI/LLM capabilities. It connects 400+ services and provides AI agent nodes for building intelligent workflows.

**Repository:** https://github.com/n8n-io/n8n  
**Documentation:** https://docs.n8n.io/  
**Stars:** ~50K  
**Latest Version:** 1.60+  
**License:** Sustainable Use License (source-available)

### Quick Start (AI Agent Node)

```json
{
  "name": "AI Agent Workflow",
  "nodes": [
    {
      "name": "Webhook Trigger",
      "type": "n8n-nodes-base.webhook",
      "parameters": { "path": "ai-agent" }
    },
    {
      "name": "AI Agent",
      "type": "@n8n/n8n-nodes-langchain.agent",
      "parameters": {
        "agentType": "toolsAgent",
        "options": {
          "systemMessage": "You are a helpful assistant.",
          "model": {
            "name": "gpt-4o",
            "apiKey": "={{ $credentials.openAiApi.apiKey }}"
          }
        }
      }
    },
    {
      "name": "HTTP Request",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "url": "https://api.example.com/search",
        "options": {}
      }
    }
  ]
}
```

### Key Features

| Feature | Description |
|---------|-------------|
| **400+ integrations** | Services, databases, APIs |
| **AI agent nodes** | LangChain integration for LLM agents |
| **Visual workflow builder** | Drag-and-drop automation |
| **Credentials management** | Centralized API key storage |
| **Execution history** | Full workflow execution logs |
| **Self-host & cloud** | Both deployment options |
| **Error handling** | Retry, error workflows |

### Pros & Cons

| Pros | Cons |
|------|------|
| Most extensive general integrations | AI features are newer (less mature) |
| Excellent workflow automation | License is not fully open-source |
| Large community | AI agent design not as polished |
| Great error handling | Less control over LLM internals |

### When to Use

✅ **Use when:** You need to integrate AI into business workflows with 400+ services.  
❌ **Avoid when:** You need a pure AI agent framework without general automation.

---

## Other Notable Frameworks

### LlamaIndex (formerly GPT Index)

- **Focus:** Data framework for LLM applications
- **Repo:** https://github.com/run-llama/llama_index
- **Stars:** ~37K
- **Key strength:** Best-in-class data indexing and retrieval
- **Best for:** RAG-heavy applications with complex data sources

### Langflow

- **Focus:** Visual LangChain builder
- **Repo:** https://github.com/langflow-ai/langflow
- **Stars:** ~32K
- **Key strength:** Similar to Flowise but with more customization
- **Best for:** Visual prototyping with Python export

### AgentScope

- **Focus:** Multi-agent distributed platform
- **Repo:** https://github.com/modelscope/agentscope
- **Stars:** ~6K
- **Key strength:** Distributed agent deployment
- **Best for:** Large-scale multi-agent simulations

### Rasa

- **Focus:** Conversational AI / chatbots
- **Repo:** https://github.com/RasaHQ/rasa
- **Stars:** ~19K
- **Key strength:** NLU pipeline for intent-based chatbots
- **Best for:** Customer service and conversational AI

### MetaGPT

- **Focus:** Multi-agent for software development
- **Repo:** https://github.com/geekan/MetaGPT
- **Stars:** ~45K
- **Key strength:** Role-based software team simulation
- **Best for:** Automated software development workflows

---

## Decision Flowchart

```
What's your primary need?
│
├─ Build a RAG system?
│   ├─ Production-ready → Haystack
│   └─ Quick prototyping → Dify / Flowise
│
├─ Build a single-agent application?
│   ├─ Python → LangChain
│   ├─ TypeScript → Eliza
│   ├─ .NET/C# → Semantic Kernel
│   └─ No-code → Dify / Flowise
│
├─ Build a multi-agent system?
│   ├─ Role-based delegation → CrewAI
│   ├─ Conversational agents → AutoGen
│   ├─ Graph-based state → LangGraph
│   └─ Distributed agents → AgentScope
│
├─ Integrate AI into business workflows?
│   └─ n8n (400+ integrations)
│
├─ Automate software development?
│   └─ MetaGPT
│
└─ Build enterprise .NET applications?
    └─ Semantic Kernel
```

---

## Benchmarking & Evaluation

### Performance Metrics

| Metric | What It Measures | Tooling |
|--------|-----------------|---------|
| **Latency (P50, P95, P99)** | Response time | LangSmith, custom |
| **Throughput** | Requests per second | k6, locust |
| **Cost per task** | API costs | LangSmith, custom |
| **Success rate** | % of tasks completed | Custom evaluation |
| **Tool call accuracy** | Correct tool selection | LangSmith, MLflow |
| **Hallucination rate** | Factual correctness | RAGAS, TruLens |

### Evaluation Tools

| Tool | Description | Link |
|------|-------------|------|
| **LangSmith** | LangChain observability and evaluation | https://smith.langchain.com/ |
| **RAGAS** | RAG evaluation metrics | https://github.com/explodinggradients/ragas |
| **TruLens** | LLM app evaluation and monitoring | https://www.trulens.org/ |
| **DeepEval** | Unit testing for LLMs | https://github.com/confident-ai/deepeval |
| **Phoenix (Arize)** | LLM observability | https://github.com/Arize-AI/phoenix |

---

## Further Reading

- [02-SOUL-SKILL-Templates.md](02-SOUL-SKILL-Templates.md) — SOUL/SKILL integration patterns
- [03-Prompt-Libraries.md](03-Prompt-Libraries.md) — Prompts for agent orchestration
- [10-Tools-Ecosystem.md](10-Tools-Ecosystem.md) — Supporting tools and infrastructure
- [LangChain Documentation](https://python.langchain.com/) — Official docs
- [AutoGen Documentation](https://microsoft.github.io/autogen/) — Microsoft's guide
- [CrewAI Documentation](https://docs.crewai.com/) — Official getting started
- [Semantic Kernel Learning](https://learn.microsoft.com/en-us/semantic-kernel/) — Microsoft Learn

---

*Document version 1.0 — Last updated 2026-06-12*

---
**See also:**
- [07 — AI Coding Agent Sustainability: Managing Hidden Costs, Code Quality Decay, and Developer Burnout](33-AI-Native-Software-Development/07-AI-Coding-Agent-Sustainability.md)
- [48 — MCP Cloud Infrastructure & Agent-as-a-Service: The Production Layer of Agentic AI](48-MCP-Cloud-Infrastructure-Agent-as-a-Service/01-Overview.md)
- [03 — Agent-Native Orchestration: LangGraph, Conductor, Mistral Workflows, Mcp-Agent](31-AI-Workflow-Orchestration-and-Durable-Execution/03-Agent-Native-Orchestration.md)
