# MCP (Model Context Protocol) and ACP (Agent Communication Protocol)

**Last Updated:** May 2026
**Author:** Hermes Agent Knowledge Base

---

## Table of Contents

1. [Overview](#overview)
2. [Part I: Model Context Protocol (MCP)](#part-i-model-context-protocol-mcp)
   - [What is MCP?](#what-is-mcp)
   - [Historical Context and Motivation](#historical-context-and-motivation)
   - [Architecture: Host, Client, Server](#architecture-host-client-server)
   - [Transport Layer](#transport-layer)
   - [Protocol Primitives](#protocol-primitives)
   - [How MCP Enables LLM-External System Interaction](#how-mcp-enables-llm-external-system-interaction)
   - [Comparison to Function Calling](#comparison-to-function-calling)
   - [Security Considerations](#security-considerations)
   - [Real-World Usage](#real-world-usage)
3. [Part II: Agent Communication Protocol (ACP)](#part-ii-agent-communication-protocol-acp)
   - [What is ACP?](#what-is-acp)
   - [Purpose and Goals](#purpose-and-goals)
   - [How Agents Discover and Communicate](#how-agents-discover-and-communicate)
   - [Relationship with MCP](#relationship-with-mcp)
   - [Current State of the Protocol](#current-state-of-the-protocol)
   - [Differences Between ACP and MCP](#differences-between-acp-and-mcp)
4. [Part III: Comparative Analysis](#part-iii-comparative-analysis)
5. [Part IV: Future Directions](#part-iv-future-directions)
6. [References and Further Reading](#references-and-further-reading)

---

## Overview

The emergence of Large Language Models (LLMs) as general-purpose reasoning engines has created a new challenge: how do these models interact with the outside world? Two protocols have emerged to address this question from different angles:

- **Model Context Protocol (MCP)** — An open protocol developed by Anthropic that standardizes how LLMs connect to external tools, data sources, and services.
- **Agent Communication Protocol (ACP)** — An emerging specification that defines how autonomous AI agents discover each other and communicate in a multi-agent ecosystem.

While both protocols deal with AI system interoperability, they serve fundamentally different purposes and operate at different layers of the stack. This document provides a comprehensive examination of each protocol, their architectures, use cases, and relationship to one another.

---

## Part I: Model Context Protocol (MCP)

### What is MCP?

The **Model Context Protocol (MCP)** is an open-standard, client-server protocol developed by Anthropic that provides a standardized interface for LLMs to interact with external systems — databases, file systems, APIs, search engines, code repositories, and any other tool or data source. It was first announced in November 2024 and released as an open specification and reference implementation.

MCP solves a fundamental problem in the LLM ecosystem: every integration between an LLM and an external system has historically been a bespoke, one-off implementation. Before MCP, connecting an LLM to a GitHub repository required custom code — and connecting the same LLM to a database required entirely different custom code. This created a fragmented landscape where each integration was brittle, non-portable, and costly to maintain.

MCP addresses this by introducing a **universal protocol layer** analogous to how USB standardized peripheral connections for computers. Just as USB allows any compliant device (keyboard, mouse, storage) to connect to any compliant host (laptop, desktop, tablet) through a standard interface, MCP allows any LLM application (the "host") to connect to any tool or data source (the "server") through a standard protocol.

**Key characteristics of MCP:**

- **Open standard:** The specification, SDKs, and reference implementations are publicly available under permissive licenses.
- **Protocol-agnostic transport:** Supports both local (stdio) and remote (SSE) communication channels.
- **Provider-neutral:** Designed to work with any LLM provider, not just Anthropic's Claude.
- **Bidirectional:** Servers can both respond to requests and push updates to clients.
- **Extensible:** The primitive set (tools, resources, prompts) can be extended as the ecosystem evolves.

### Historical Context and Motivation

MCP emerged from a specific pain point at Anthropic. As Claude gained capabilities and users began integrating it into their workflows, Anthropic observed that each integration required a custom adapter layer. This was unsustainable for three reasons:

1. **Duplication of effort:** Every MCP-like capability (database access, file operations, API calls) was being reimplemented by every integrator.
2. **Fragmentation:** No standard way for tools to declare their capabilities or for models to discover what's available.
3. **Context window pressure:** Without a structured protocol, tool descriptions and invocation patterns consumed precious context tokens with inconsistent formatting.

The protocol was designed to be **universal** — not tied to any specific LLM or provider. While Anthropic developed it, the specification is vendor-neutral and can be implemented by any model provider. This is a deliberate architectural choice: the protocol sits between the application layer and the tool layer, not inside any particular model.

### Architecture: Host, Client, Server

MCP uses a three-tier architecture with clearly defined roles:

```
+----------------------------------+
|         HOST APPLICATION         |
|  (Claude Desktop, IDE, CLI,      |
|   custom application)            |
|                                  |
|  +----------------------------+  |
|  |        MCP CLIENT         |  |
|  |  (embedded in host)       |  |
|  +----------+----------------+  |
+-------------|-------------------+
              |  MCP Protocol
              |  (JSON-RPC over stdio/SSE)
              |
+-------------|-------------------+
|  +----------+----------------+  |
|  |        MCP SERVER         |  |
|  |  (adapter for a tool or   |  |
|  |   data source)            |  |
|  +----------------------------+  |
|                                  |
|  +----------------------------+  |
|  |   EXTERNAL SYSTEM          |  |
|  |  (database, API, files,    |  |
|  |   search engine, etc.)     |  |
|  +----------------------------+  |
+----------------------------------+
```

#### Host

The **host** is the application or environment where the LLM operates and the user interacts. It is the top-level orchestrator that:

- Manages the lifecycle of MCP client connections
- Handles user authentication and authorization
- Controls which servers are available to the LLM
- Processes the LLM's decisions about which tools to invoke
- Presents results back to the user

Examples of hosts include:
- **Claude Desktop** — Anthropic's desktop application
- **IDE plugins** — VS Code extension, JetBrains plugin, etc.
- **CLI tools** — Command-line interfaces that wrap LLM interactions
- **Custom applications** — Any application that embeds an LLM

The host is responsible for **security policy enforcement**. It decides which MCP servers a given LLM session can access, under what conditions, and with what permissions.

#### Client

The **client** is a protocol-level component embedded within the host application. It establishes and manages a 1:1 connection with an MCP server. The client:

- Initiates the connection using the configured transport
- Negotiates protocol version and capabilities during initialization
- Sends requests on behalf of the LLM (tool calls, resource reads, prompt retrievals)
- Receives responses and optional notifications from the server
- Manages connection state, reconnection, and error handling

**Important architectural note:** The MCP client is not the LLM itself. The client is a protocol adapter within the host application that translates between the LLM's decisions and the MCP protocol messages. The LLM operates at a higher level — it decides which tool to call based on descriptions provided through the protocol, but the actual protocol negotiation and message formatting is handled by the client.

A single host may run multiple clients simultaneously, each connected to a different server. For example, Claude Desktop might maintain concurrent connections to a filesystem server, a database server, and a GitHub server.

#### Server

The **server** is a lightweight adapter that exposes a specific tool, data source, or service through the MCP protocol. Each server:

- Implements the MCP protocol specification
- Registers its capabilities (tools, resources, prompts) during initialization
- Handles incoming requests from the client
- Communicates with the underlying external system
- Can send proactive notifications to the client

Servers are designed to be **single-purpose** and **independently deployable**. A typical MCP server wraps one underlying service — for example:

- `mcp-server-filesystem` — Exposes local file system operations
- `mcp-server-postgres` — Exposes PostgreSQL database queries
- `mcp-server-github` — Exposes GitHub API operations
- `mcp-server-slack` — Exposes Slack messaging operations
- `mcp-server-puppeteer` — Exposes browser automation

Servers can run locally (started as subprocesses by the host) or remotely (accessed over the network). The server is stateless with respect to protocol state — each request contains enough information to be processed independently.

#### Initialization Sequence

The MCP connection lifecycle follows a structured handshake:

1. **Client sends `initialize` request** — includes protocol version, client capabilities, and client metadata
2. **Server responds with `initialized`** — includes protocol version, server capabilities (list of supported features), and server metadata
3. **Client sends `initialized` notification** — acknowledges the server's response
4. **Normal operation begins** — tools, resources, and prompts can be listed and used
5. **Termination** — either side can close the connection gracefully

During initialization, both sides negotiate the protocol version (using semantic versioning) and declare which capabilities they support. This allows the protocol to evolve while maintaining backward compatibility.

### Transport Layer

MCP defines two primary transport mechanisms, plus extensibility for custom transports.

#### 1. Stdio Transport (Standard Input/Output)

The **stdio transport** is used for local, subprocess-based communication between a host and a server running on the same machine.

**How it works:**
- The host launches the MCP server as a child process
- The host writes JSON-RPC messages to the server's stdin
- The server writes JSON-RPC responses to its stdout
- The server writes log/debug information to stderr (which the host can capture separately)
- The host monitors the child process for termination

**Characteristics:**
- **Low latency:** No network overhead; communication happens via OS pipes
- **High security:** The server runs as a local process with the host's permissions; no network exposure
- **Simple deployment:** No network configuration, authentication, or TLS setup
- **Process isolation:** Each server runs in its own process; crashes are contained
- **Single connection:** One client connects to one server per process

**Use cases:**
- Desktop applications connecting to local tools (file system, local database, local services)
- Development tools (IDE plugins connecting to code analyzers, linters, local git)
- CLI-based LLM workflows where all tools run on the local machine

**Example configuration (from Claude Desktop's config.json):**
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/path/to/allowed/directory"
      ]
    },
    "github": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-github"
      ],
      "env": {
        "GITHUB_TOKEN": "<your-token>"
      }
    }
  }
}
```

#### 2. SSE Transport (Server-Sent Events)

The **SSE transport** enables remote communication between a client and a server over HTTP. It uses a combination of:

- **SSE** (Server-Sent Events) — for the server-to-client direction (notifications, responses)
- **HTTP POST** — for the client-to-server direction (requests)

**How it works:**
- The client connects to the server's SSE endpoint
- The server sends an `endpoint` event containing the URL for client-to-server requests
- The client sends JSON-RPC requests via HTTP POST to that endpoint
- The server streams responses and notifications back through the SSE connection

**Characteristics:**
- **Remote access:** Servers can be anywhere on the network
- **Persistent connection:** The SSE channel stays open for the session duration
- **Standard web infrastructure:** Works through proxies, load balancers, and CDNs
- **HTTP-based security:** Can leverage TLS, API keys, OAuth, and other web security mechanisms
- **Multiple clients:** A remote server can serve multiple clients simultaneously

**Use cases:**
- Connecting to cloud-based services (SaaS tools, cloud databases, web APIs)
- Multi-user or shared server deployments
- Scenarios where the tool runs in a different environment than the LLM

#### 3. Custom Transports

The protocol specification allows for custom transports beyond stdio and SSE. The SDK provides interfaces for implementing alternative transports, which could include:

- WebSocket-based transport
- gRPC-based transport (for high-performance scenarios)
- Message queue transports (RabbitMQ, Kafka for enterprise deployments)
- Unix domain sockets (for local inter-process communication with better lifecycle management than pipes)

#### Message Format

All MCP messages use **JSON-RPC 2.0** as the wire format. This is a lightweight, well-established RPC protocol that uses JSON for data serialization. The protocol defines three types of messages:

- **Requests** — Client sends to server, expects a response (e.g., `tools/call`, `resources/read`)
- **Responses** — Server replies to a request (success with result, or error)
- **Notifications** — Either side sends without expecting a response (e.g., `notifications/initialized`, `notifications/cancelled`)

**Example JSON-RPC message (tool call request):**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "read_file",
    "arguments": {
      "path": "/path/to/document.txt"
    }
  }
}
```

**Example JSON-RPC response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Contents of the file..."
      }
    ],
    "isError": false
  }
}
```

### Protocol Primitives

MCP defines three core primitives that servers can expose: **Tools**, **Resources**, and **Prompts**. These form the building blocks of LLM-external system interaction.

#### 1. Tools

**Tools** are executable operations that the LLM can invoke through the host application. They represent actions the model can take — the "do something" primitive.

**Characteristics:**
- **Model-initiated:** The LLM decides when to call a tool based on its description and parameters
- **User-approved:** The host typically presents tool calls to the user for confirmation before execution
- **Stateless:** Each tool call is independent; servers maintain state if needed, but the protocol is stateless
- **Batched results:** Tool responses can contain multiple content items (text, images, embedded resources)

**Tool definition structure:**
```json
{
  "name": "read_file",
  "description": "Read the contents of a file at the specified path",
  "inputSchema": {
    "type": "object",
    "properties": {
      "path": {
        "type": "string",
        "description": "Absolute path to the file"
      }
    },
    "required": ["path"]
  }
}
```

**Protocol methods for tools:**

| Method | Direction | Description |
|--------|-----------|-------------|
| `tools/list` | Client -> Server | List available tools with their schemas |
| `tools/call` | Client -> Server | Execute a specific tool with given arguments |

**Typical tool categories:**
- **File operations:** read, write, edit, list, search files
- **Database operations:** query, insert, update, delete records
- **API integrations:** send messages, create issues, manage resources
- **Code operations:** compile, lint, test, analyze code
- **Web operations:** fetch URL, search, scrape content
- **System operations:** execute commands, manage processes

#### 2. Resources

**Resources** are data sources that can be read by the LLM. They represent the "know something" primitive — structured or unstructured data that provides context.

**Characteristics:**
- **Static data:** Resources are read-only; they provide data, not actions
- **URI-addressed:** Each resource has a unique URI for identification and access
- **Structured or unstructured:** Resources can be text, binary, or structured data (JSON, etc.)
- **MIME-typed:** Resources include content type information for proper handling
- **Subscribable:** Clients can subscribe to resource changes (if the server supports it)

**Resource definition structure:**
```json
{
  "uri": "file:///path/to/document.txt",
  "name": "Project README",
  "description": "The README file for the current project",
  "mimeType": "text/plain"
}
```

**Protocol methods for resources:**

| Method | Direction | Description |
|--------|-----------|-------------|
| `resources/list` | Client -> Server | List available resources |
| `resources/read` | Client -> Server | Read the content of a specific resource |
| `resources/subscribe` | Client -> Server | Subscribe to changes on a resource |
| `resources/unsubscribe` | Client -> Server | Unsubscribe from resource changes |

**Resource templates** allow servers to expose dynamic resources with URI patterns:
```
file:///{path}
git:///{repository}/commits/{sha}
postgres:///{database}/tables/{table}/schema
```

This is useful when the set of accessible resources is too large to enumerate (e.g., all files in a filesystem, all rows in a database, all documents in a search index).

#### 3. Prompts

**Prompts** are pre-defined, reusable prompt templates that servers can provide. They represent the "structured interaction" primitive.

**Characteristics:**
- **Template-based:** Prompts can include dynamic arguments that get substituted at retrieval time
- **Multi-message:** A prompt can consist of multiple messages (system, user, assistant) with different roles
- **Reusable:** Prompts are designed to be used multiple times, possibly by different users
- **Discoverable:** The client can list all available prompts from a server

**Prompt definition structure:**
```json
{
  "name": "code_review",
  "description": "Review code changes for quality and correctness",
  "arguments": [
    {
      "name": "code",
      "description": "The code to review",
      "required": true
    }
  ]
}
```

**Protocol methods for prompts:**

| Method | Direction | Description |
|--------|-----------|-------------|
| `prompts/list` | Client -> Server | List available prompts |
| `prompts/get` | Client -> Server | Get a specific prompt with arguments substituted |

**Use cases for prompts:**
- Domain-specific interaction templates (e.g., "Review this pull request", "Explain this error")
- Pre-configured workflows (e.g., "Debug this issue", "Generate documentation")
- Contextual assistance (e.g., "Help me write a SQL query for this schema")
- Multi-step agent patterns (e.g., "Plan, then execute, then analyze")

#### Primitive Comparison

| Aspect | Tools | Resources | Prompts |
|--------|-------|-----------|---------|
| **Primary action** | Execute | Read | Retrieve |
| **Initiated by** | LLM (via host) | LLM (via host) | User or LLM |
| **Side effects** | Yes (modifies state) | No (read-only) | No (template retrieval) |
| **Parameters** | Required (input schema) | URI (addressing) | Template arguments |
| **Return type** | Content items | Content + MIME type | Messages |
| **Subscription** | No | Yes (optional) | No |
| **Analogy** | Functions/APIs | Files/Documents | Macros/Templates |

### How MCP Enables LLM-External System Interaction

The interaction flow between an LLM and an external system through MCP follows a structured pattern:

#### Step-by-step flow:

1. **Server Registration:** The host starts one or more MCP servers, each configured with credentials and access permissions for its underlying system.

2. **Capability Discovery:** The MCP client sends `tools/list`, `resources/list`, and `prompts/list` to each connected server. The servers respond with their available primitives, including schemas, descriptions, and URI patterns.

3. **Context Building:** The host incorporates the discovered tool/resource/prompt information into the LLM's system prompt or context window. This tells the LLM what capabilities are available and how to use them.

4. **LLM Reasoning:** The LLM processes the user's request, taking into account the available tools and resources. It may decide that invoking a tool would help accomplish the task.

5. **Tool Selection:** The LLM generates a tool call (typically in a structured format like function-calling JSON or XML). The host intercepts this and validates it against the tool's schema.

6. **User Confirmation (optional):** The host may present the proposed tool call to the user for approval, especially for operations that modify state or access sensitive data.

7. **Protocol Invocation:** The MCP client sends a `tools/call` JSON-RPC request to the appropriate server, passing the tool name and arguments.

8. **Server Execution:** The MCP server receives the request, translates it into the native API call or operation on the underlying system, and returns the result.

9. **Result Incorporation:** The MCP client receives the response and feeds it back to the LLM, typically by appending it as a tool result message in the conversation context.

10. **Continuation:** The LLM continues reasoning with the new information, potentially calling more tools or generating the final response.

#### Example Scenario: Database Query

A user asks: *"What were our top 10 products by revenue last quarter?"*

1. The host has an MCP server for PostgreSQL connected
2. The LLM sees the `query_database` tool with schema: database, query
3. The LLM decides to call `query_database` with a SQL query
4. The host shows the user: "Claude wants to run: SELECT p.name, SUM(o.total) as revenue FROM..."
5. The user approves
6. The MCP client sends the request to the postgres server
7. The server executes the query against the database
8. Results are returned to the LLM
9. The LLM processes the results and responds with a natural language answer

This flow demonstrates how MCP provides **structured, safe, and auditable** access to external systems. Every tool invocation is explicit, logged, and (optionally) user-verified.

### Comparison to Function Calling

MCP is often compared to function calling (also known as tool use, function calling, or tool-augmented generation). While related, they operate at different layers:

| Aspect | Function Calling | Model Context Protocol (MCP) |
|--------|-----------------|------------------------------|
| **Layer** | Model-level capability | Application-level protocol |
| **Scope** | How an LLM declares intent to use a tool | How tools are discovered, described, and invoked |
| **Standardization** | Provider-specific (OpenAI, Anthropic, Google each define their own format) | Protocol-level standard agnostic to model provider |
| **Tool definition** | Defined in the prompt/system message | Defined via `tools/list` and `tools/call` endpoints |
| **Discovery** | Static — tools are enumerated in the prompt | Dynamic — tools can be listed at runtime |
| **Transport** | N/A (within the LLM API call) | External (stdio or network) |
| **Lifecycle** | Per-request (tools resubmitted each time) | Persistent connection (tools available across requests) |
| **Security** | API-layer (what the LLM API enforces) | Protocol + host-layer (access control, user confirmation) |
| **Extensibility** | Limited to what the model provider supports | Open — anyone can write an MCP server |

**Key insight:** Function calling and MCP are complementary, not competitive. Function calling is how the LLM expresses "I want to use this tool with these arguments." MCP is how the application infrastructure discovers, describes, and invokes those tools. You could use MCP with any LLM that supports function calling — the MCP client translates between the protocol and the LLM's native tool format.

**Comparison with OpenAI Plugins:** OpenAI's now-deprecated plugins system (from 2023) shared some goals with MCP but was tied to OpenAI's platform. MCP is intentionally provider-agnostic and open. The plugin manifest specification was OpenAI-specific; MCP's primitive definitions and JSON-RPC transport are designed to work with any LLM provider.

### Security Considerations

MCP introduces several security considerations that stem from its architecture of connecting LLMs to external systems:

#### 1. Tool Authorization

MCP servers represent powerful capabilities — file system access, database queries, API calls, code execution. The protocol itself does not enforce authorization; this is the responsibility of the **host application** and the **server implementation**.

**Best practices:**
- Each server should authenticate to its underlying service using secure credentials (API keys, OAuth tokens, service accounts)
- Servers should implement the **principle of least privilege** — only expose the minimum necessary capabilities
- Hosts should ask for user confirmation before executing tools that have side effects (write, delete, modify operations)

#### 2. Data Leakage via Resources

Resources can expose sensitive data to the LLM. Since LLMs may include this data in their context (and potentially in their training data, depending on the provider), careful controls are needed.

**Best practices:**
- Restrict resource access to only the data the user is authorized to see
- Consider resource-level access controls (who can list/read which resources)
- Be aware of context window limits — large resources can consume significant context
- Implement PII redaction on sensitive resources before serving them to the LLM

#### 3. Server Isolation

Stdio-transport servers run as local processes. A malicious or compromised server could potentially:
- Access the filesystem beyond its configured boundaries
- Execute arbitrary code
- Leak credentials from environment variables

**Mitigations:**
- Run MCP servers in sandboxed environments (containers, restricted user accounts, seccomp profiles)
- Use environment variables for credentials (keeps them out of the protocol messages)
- Apply filesystem-level access controls
- Consider running untrusted servers in separate containers or VMs

#### 4. Prompt Injection

Since prompts can be provided by external servers, there is a risk of prompt injection — a server providing a malicious prompt template that manipulates the LLM's behavior.

**Mitigations:**
- Hosts should validate or sanitize prompts from untrusted servers
- Separate system-level instructions from user-provided and server-provided content
- Consider treating server-provided prompts as untrusted input

#### 5. Remote Server Security

SSE transport exposes MCP servers over the network, introducing standard web security considerations:

- **TLS/HTTPS:** All remote MCP connections should use TLS encryption
- **Authentication:** Servers should authenticate clients (API keys, bearer tokens, mTLS)
- **Rate limiting:** Protect servers from abuse
- **Input validation:** Validate all JSON-RPC payloads at the server boundary
- **Network segmentation:** MCP servers should not be directly exposed to the internet unless necessary

#### 6. Credential Management

MCP servers commonly require credentials to access their underlying services. The protocol recommends:

- Credentials should be provided via environment variables, not embedded in server code
- Hosts should store credentials securely (OS keychain, encrypted config)
- Servers should not log or transmit credentials in cleartext
- Credential rotation should be supported without server restart (if the underlying service allows)

### Real-World Usage

MCP has seen rapid adoption since its release in late 2024. Below are notable use cases and implementations.

#### Adoption by Tools and Platforms

1. **Claude Desktop (Anthropic)** — The flagship MCP host. Claude Desktop allows users to configure MCP servers through a JSON configuration file. Anthropic provides several reference server implementations: filesystem, GitHub, PostgreSQL, SQLite, Puppeteer, Slack, Google Drive, and more.

2. **Reference Implementations** — Anthropic maintains official MCP server packages on npm (`@modelcontextprotocol/server-*`) covering:
   - Filesystem operations
   - Git operations
   - GitHub API
   - GitLab API
   - Google Drive
   - Google Maps
   - PostgreSQL and SQLite databases
   - Puppeteer (browser automation)
   - Slack messaging
   - Sentry error tracking
   - Sequential thinking (chain-of-thought assistant)

3. **Community Ecosystem** — A growing ecosystem of community-built MCP servers for services including:
   - Kubernetes cluster management
   - AWS resource management
   - Jira and Linear issue tracking
   - Notion, Obsidian, and other note-taking apps
   - Docker container management
   - Redis, MongoDB, and other databases
   - Web scraping and crawling
   - Email (IMAP/SMTP)
   - Cloudflare, Vercel, and other deployment platforms

4. **IDE Integration** — VS Code and other IDEs have adopted MCP for coding agents, allowing the IDE to:
   - Read and write files through the filesystem server
   - Query codebases via search servers
   - Run tests and linters through tool servers
   - Access git history and repository metadata

5. **CLI Tools** — Command-line tools like `mcp-cli` allow running MCP interactions directly from the terminal, useful for scripting and automation.

#### Use Case Categories

**Development and Coding:**
- Automated code review with access to git, linters, and test runners
- Documentation generation from codebase analysis
- Bug reproduction using browser automation (Puppeteer MCP server)
- Database schema exploration and query generation

**Data Analysis:**
- Querying databases through postgres/sqlite servers
- Reading and analyzing files from the filesystem
- Searching documentation and knowledge bases
- Processing data through web APIs

**Operations and DevOps:**
- Kubernetes cluster management
- Cloud resource provisioning via AWS/GCP/Azure servers
- Incident response with monitoring tool integration
- Log analysis through log aggregation services

**Content Creation:**
- Drafting and publishing blog posts or documentation
- Managing content in CMS platforms
- Image generation and manipulation through API servers
- Social media scheduling and posting

**Enterprise Automation:**
- CRM data management
- ERP system queries and updates
- HR system integrations
- Financial data analysis

#### Example: MCP in a Data Analysis Workflow

A practical example of MCP in action:

1. User asks: "Analyze our Q1 sales data and create a visualization"
2. The MCP host (Claude Desktop) has servers for filesystem, postgres, and slack
3. Claude uses the `postgres` server to query Q1 sales data from the database
4. Claude uses the `filesystem` server to write a Python script that creates a visualization
5. Claude uses the `filesystem` server again to run the script and capture the output image
6. Claude uses the `slack` server to post the visualization to the team channel
7. The user gets: a summary of the analysis, the generated chart, and a Slack notification

All of this happens through standardized protocol interactions without any custom glue code.

---

## Part II: Agent Communication Protocol (ACP)

### What is ACP?

The **Agent Communication Protocol (ACP)** is an emerging open specification that defines how autonomous AI agents discover each other, exchange messages, coordinate actions, and share information in a multi-agent environment. While MCP standardizes LLM-to-tool communication, ACP standardizes agent-to-agent communication.

ACP addresses a different but equally important problem: as AI agents become capable of performing complex, multi-step tasks, there is an increasing need for agents to work together. A single agent might be good at code generation but not at deployment; another might excel at data analysis but not at visualization. ACP provides the communication substrate that enables these agents to collaborate.

The protocol is being developed as a community effort with contributions from multiple organizations and researchers working on multi-agent systems. It is not owned by any single company, unlike MCP's origin at Anthropic.

### Purpose and Goals

ACP is designed to address several fundamental challenges in multi-agent systems:

#### Primary Goals:

1. **Agent Discovery** — Allow agents to find each other based on capabilities, without requiring a central registry or manual configuration. Agents should be able to join an ecosystem and announce "I can do X, Y, Z."

2. **Standardized Communication** — Define a common message format and protocol for agent-to-agent interaction, regardless of what underlying LLM or technology each agent uses internally.

3. **Task Delegation** — Enable one agent to request another agent's services. "Agent A, who is good at code generation, can ask Agent B, who is good at security auditing, to review the generated code."

4. **State Sharing** — Allow agents to share context, intermediate results, and state information so that collaboration is coherent and agents don't need to rediscover information.

5. **Error Handling** — Define how agents communicate failures, timeouts, and unexpected conditions to each other.

6. **Interoperability** — Ensure that agents built with different frameworks, LLMs, or architectures can communicate productively.

#### Design Principles:

- **Decentralized:** No central authority or registry required; agents can form ad-hoc networks
- **Extensible:** The message types and interaction patterns can grow with the ecosystem
- **Language-agnostic:** Works with any programming language or AI framework
- **Security-aware:** Includes mechanisms for authentication, authorization, and trust
- **Graceful degradation:** Communication failures shouldn't crash the system

### How Agents Discover and Communicate

ACP defines a multi-layered approach to agent interaction:

#### Agent Discovery

ACP supports several discovery mechanisms, ranging from simple to sophisticated:

**1. Static Configuration (Direct Connection):**
Agents are configured with the addresses of other agents they should communicate with. This is the simplest approach, suitable for small, controlled deployments.

```
Agent A -> configured to know about Agent B at agent-b.example.com:8080
Agent B -> configured to know about Agent A at agent-a.example.com:8080
```

**2. Capability Registry (Centralized Discovery):**
A discovery service maintains a registry where agents register their capabilities and connection endpoints. Other agents query the registry to find agents with specific skills.

```
Agent A registers: {"capabilities": ["code_generation", "python"], "endpoint": "..."}
Agent B queries: "Find agents with code_review capability"
Registry returns: [Agent C (code_review), Agent D (security_review)]
```

**3. Peer-to-Peer Discovery (Decentralized):**
Agents use gossip protocols or distributed hash tables to discover each other without a central point of failure. This is more complex but more resilient and scalable.

**4. DNS-Based Discovery:**
Using DNS SRV records or well-known endpoints to discover agent services, similar to how service mesh architectures work.

#### Communication Patterns

ACP defines several fundamental communication patterns:

**1. Request-Response (Synchronous):**
An agent sends a request and waits for a response. Used for queries, calculations, or any interaction that expects an immediate answer.

```
Agent A -> Agent B: "Analyze this code for security vulnerabilities"
Agent B -> Agent A: "Found 2 vulnerabilities: SQL injection at line 42, XSS at line 78"
```

**2. Task Delegation (Asynchronous):**
An agent delegates a task and receives a task ID. The delegating agent can check back later or receive a callback/notification when the task completes.

```
Agent A -> Agent B: {"action": "delegate", "task": "audit_code", "payload": {...}, "callback": "agent-a/callback"}
Agent B -> Agent A: {"task_id": "task-123", "status": "accepted"}
... later ...
Agent B -> Agent A: {"task_id": "task-123", "status": "completed", "result": {...}}
```

**3. Broadcast/Multicast:**
An agent sends a message to multiple agents simultaneously. Used for announcements, questions ("Does anyone know X?"), or coordination signals.

```
Agent A -> All: "Starting deployment of version 2.1.0, pausing all dependent operations"
Agent B, Agent C, Agent D -> Agent A: "Acknowledged"
```

**4. Publish-Subscribe:**
Agents subscribe to specific topics or event types and receive notifications when relevant events occur. This is useful for ongoing monitoring and alerting scenarios.

```
Agent A subscribes to: "security_events", "deployment_status"
Agent B publishes: {"topic": "security_events", "severity": "high", "description": "..."}
Agent A receives the notification and takes action
```

**5. Negotiation Protocol:**
Agents engage in multi-turn conversations to agree on plans, resolve conflicts, or coordinate complex workflows.

```
Agent A: "I propose we split the task: I'll generate the code, you test it, she deploys it."
Agent B: "I can test, but I need a test plan first."
Agent C: "I'll handle deployment, but only after tests pass with 90%+ coverage."
Agent A: "Agreed. Let's proceed with this plan."
```

#### Message Format

ACP uses a structured message format (typically JSON) with common fields:

```json
{
  "protocol": "acp/1.0",
  "message_id": "msg-abc-123",
  "sender_id": "agent-code-gen-01",
  "receiver_id": "agent-security-audit-02",
  "timestamp": "2026-05-31T10:30:00Z",
  "message_type": "request",
  "interaction_id": "interaction-456",
  "payload": {
    "action": "security_audit",
    "parameters": {
      "code": "def login(username, password): ...",
      "language": "python",
      "depth": "thorough"
    }
  },
  "metadata": {
    "ttl": 300,
    "priority": "normal",
    "signature": "base64signature..."
  }
}
```

Key fields:
- **protocol:** Version of ACP being used
- **message_id:** Unique identifier for this message (for deduplication)
- **sender_id/receiver_id:** Agent identities
- **interaction_id:** Correlates messages in a multi-turn interaction
- **message_type:** request, response, notification, broadcast, etc.
- **payload:** The actual content, structured by the interaction type
- **metadata:** Routing, security, and lifecycle information

### Relationship with MCP

ACP and MCP are complementary protocols that operate at different layers of an AI system architecture. They are not competitors — they solve different problems.

#### Architectural Layering

```
+----------------------------------------------------+
|                  USER INTERFACE                     |
+----------------------------------------------------+
|              AGENT COMMUNICATION LAYER              |
|                   (ACP Layer)                       |
|  Multi-agent coordination, task delegation,         |
|  knowledge sharing between agents                   |
+----------------------------------------------------+
|              TOOL INTEGRATION LAYER                 |
|                   (MCP Layer)                       |
|  Tool discovery, resource access, prompt templates  |
+----------------------------------------------------+
|                    AI MODELS                        |
|  (LLMs, vision models, specialized models)          |
+----------------------------------------------------+
|                 INFRASTRUCTURE                      |
|  (Compute, storage, networking)                     |
+----------------------------------------------------+
```

In this architecture:
- **MCP** handles the vertical integration: how an AI agent talks to databases, APIs, files, and other external tools.
- **ACP** handles the horizontal integration: how multiple AI agents talk to each other.

#### How They Can Work Together

A realistic multi-agent system might use both protocols:

1. A **coordinator agent** (operating over ACP) receives a complex user request
2. It decomposes the task and uses ACP to delegate subtasks to specialized agents:
   - A **code generator agent**
   - A **data analysis agent**
   - A **deployment agent**
3. Each specialized agent uses MCP internally to interact with the tools needed for its subtask:
   - The code generator uses MCP to access the filesystem and git repositories
   - The data analyst uses MCP to query databases and run analysis scripts
   - The deployment agent uses MCP to interact with cloud APIs and CI/CD systems
4. The agents coordinate through ACP, sharing intermediate results and status updates
5. The coordinator agent synthesizes the final result for the user

This two-protocol approach provides clean separation of concerns:
- **MCP** answers: "How does an agent use tools?"
- **ACP** answers: "How do agents work together?"

### Current State of the Protocol

As of May 2026, ACP is in an earlier stage of development compared to MCP. Here is the current status:

#### Development Status

- **Specification:** Early draft stages. The protocol concepts are being formalized by a community of researchers and practitioners working on multi-agent systems.
- **Adoption:** Limited to experimental and research deployments. No major production systems have publicly adopted ACP as a standard.
- **Implementations:** A few proof-of-concept implementations exist, primarily in Python and TypeScript, but there is no canonical reference implementation comparable to Anthropic's MCP SDK.
- **Standardization:** There is no single governing body. Multiple groups are working on similar ideas; convergence toward a unified standard is an ongoing process.

#### Key Challenges

1. **Lack of a dominant implementation:** Unlike MCP, which was launched by Anthropic with a concrete SDK and reference servers, ACP lacks a single driving organization. This makes standardization slower.

2. **Agent identity and trust:** How do agents prove their identity? How do they establish trust before delegating tasks? This is an open research problem involving cryptographic identity, reputation systems, and capability verification.

3. **Semantic understanding:** For agents to collaborate effectively, they need to understand not just the message format but the semantics of each other's capabilities. This requires shared ontologies or dynamic capability negotiation.

4. **Error resilience:** What happens when an agent goes offline mid-task? How are tasks reassigned? The protocol needs robust failure semantics.

5. **Security at scale:** As agent networks grow, ensuring secure communication, preventing impersonation, and managing permissions becomes complex.

6. **Performance overhead:** Multi-agent communication adds latency. For time-sensitive tasks, the overhead of protocol negotiation, serialization, and routing must be minimized.

#### Ongoing Work

Several research and development efforts are exploring the ACP design space:

- **Protocol design:** Defining the core message types, interaction patterns, and state management semantics.
- **Capability description languages:** Creating standard ways for agents to describe what they can do, analogous to OpenAPI for APIs.
- **Discovery mechanisms:** Evaluating different approaches from simple registries to fully decentralized discovery.
- **Security frameworks:** Developing authentication, authorization, and trust models appropriate for agent networks.
- **Integration with existing standards:** Exploring how ACP can work with other protocols (MCP, gRPC, HTTP, WebSocket, message queues).

### Differences Between ACP and MCP

| Dimension | Model Context Protocol (MCP) | Agent Communication Protocol (ACP) |
|-----------|------------------------------|------------------------------------|
| **Primary purpose** | Connect LLMs to tools and data | Connect agents to other agents |
| **Origin** | Anthropic (open-sourced) | Community-driven (multiple orgs) |
| **Maturity** | Released, SDK available, growing ecosystem | Early stage, experimental |
| **Architecture** | Host-Client-Server (three-tier) | Peer-to-peer (agent network) |
| **Transport** | stdio (local), SSE (remote) | HTTP, WebSocket, message queues (to be defined) |
| **Primitives** | Tools, Resources, Prompts | Messages, Tasks, Capabilities (evolving) |
| **Initiation** | LLM-driven (via function calling) | Agent-driven (autonomous delegation) |
| **Discovery** | Server discovery by host (config) | Agent discovery by capability |
| **State model** | Stateless (per-request) | Stateful (conversations, task tracking) |
| **Security model** | Host-enforced + server permissions | Distributed trust, identity, attestation |
| **Complexity** | Lower (tool calls are simple operations) | Higher (negotiation, delegation, coordination) |
| **Adoption** | Growing rapidly in production | Experimental/research phase |
| **Reference impl.** | Yes (TypeScript SDK, Python SDK) | No canonical implementation yet |
| **Problem analogy** | "USB for LLM tools" | "TCP/IP for AI agents" |

#### Can They Be Used Together?

**Yes, and this is the expected pattern.** In a mature multi-agent system:

1. ACP handles **inter-agent communication** — the protocol layer where agents discover each other, negotiate task divisions, and share results.
2. MCP handles **agent-to-tool communication** — the protocol layer where each individual agent accesses the external systems it needs to do its work.

An agent's stack might look like:

```
+---------------------------+
|    Agent Logic (LLM)      |
+---------------------------+
| ACP Client Module         |  <-> Other agents
| MCP Client Module         |  <-> Tools/Services
+---------------------------+
```

This layered approach allows each protocol to focus on its specific domain while enabling powerful multi-agent systems that can also interact with the world.

---

## Part III: Comparative Analysis

### When to Use Which Protocol

| Scenario | Use MCP | Use ACP | Use Both |
|----------|---------|---------|----------|
| Single agent needs database access | Yes | No | No |
| Single agent needs to read/write files | Yes | No | No |
| Multiple agents collaborating on a task | No | Yes | Yes |
| Agent needs to query an external API | Yes | No | No |
| Agent network with specialized roles | No | Yes | Yes |
| Building a coding assistant | Yes | No | No |
| Building a multi-agent coding team | Yes | Yes | Yes |
| Connecting LLM to a CRM system | Yes | No | No |
| Coordinating across 10+ agents | No | Yes | Yes |
| Rapid prototyping of tool access | Yes | No | No |

### Maturity and Risk Assessment

**MCP:**
- **Maturity:** High (production-ready SDK, clear specification, growing ecosystem)
- **Risk:** Low (backed by Anthropic, open source, active community)
- **Investment readiness:** Ready for production use today
- **Learning curve:** Moderate (understand JSON-RPC, the three primitives, and transport options)

**ACP:**
- **Maturity:** Low (early specification, limited implementations)
- **Risk:** High (no standard implementation, potential for fragmentation)
- **Investment readiness:** Experimental only; expect breaking changes
- **Learning curve:** Steep (multiple design dimensions still being explored)

### Ecosystem Comparison

| Metric | MCP | ACP |
|--------|-----|-----|
| Official servers | ~20+ reference implementations | None |
| Community servers | 200+ (and growing rapidly) | A few experimental |
| SDK languages | TypeScript, Python (official) | Python, TypeScript (experimental) |
| Integration platforms | Claude Desktop, VS Code, CLI | None mainstream |
| Documentation | Comprehensive, with tutorials | Limited, academic |
| Conference presence | Multiple talks at AI/developer conferences | Research workshops |

---

## Part IV: Future Directions

### MCP Evolution

1. **Broader Model Support:** While MCP works with any LLM via function calling, deeper integration with more model providers' native tool formats will reduce friction.

2. **Enhanced Security Model:** Expect more sophisticated authorization frameworks, including role-based access control (RBAC) for MCP servers and fine-grained permission scopes.

3. **Streaming Responses:** Ability for tools to stream partial results back to the LLM (useful for long-running operations, progress updates, and real-time data).

4. **Transactional Operations:** Support for multi-step atomic operations across multiple servers (e.g., "read from DB, write to file, post to Slack" as a single transaction).

5. **Caching and Optimization:** Standardized caching mechanisms for resources and tool results to reduce latency and save API calls.

6. **Multi-Server Coordination:** Better primitives for MCP servers to coordinate with each other, potentially via the host as an intermediary.

7. **Mobile and Edge Support:** Adaptations of the stdio transport for mobile environments and the SSE transport for edge computing scenarios.

### ACP Evolution

1. **Standardization:** The most critical path for ACP is convergence around a single specification. This may happen organically through community consensus or through adoption by a major AI platform.

2. **Reference Implementation:** A robust, well-documented reference implementation in one or two languages is essential for adoption.

3. **Integration with MCP:** As both protocols mature, expect tighter integration — perhaps ACP messages carrying MCP-style tool descriptions, or MCP servers being wrapped as ACP-capable agents.

4. **Identity and Trust Frameworks:** Development of practical, cryptographically sound approaches to agent identity and trust, potentially leveraging existing standards like DID (Decentralized Identifiers) and Verifiable Credentials.

5. **Capability Ontologies:** Standardized vocabularies for describing agent capabilities, making discovery more meaningful and automated.

6. **Testing and Verification Tools:** Frameworks for testing multi-agent interactions, simulating failures, and verifying protocol compliance.

7. **Enterprise Features:** Audit logging, compliance tracking, SLA management, and billing integration for agent-to-agent interactions.

### Convergence Possibilities

The long-term trajectory suggests a convergence where:

1. **MCP becomes the standard tool-access layer** for virtually all LLM applications, analogous to how HTTP/SSL became the standard for web communication.

2. **ACP builds on MCP's patterns** but extends them to the multi-agent domain, possibly reusing MCP's primitive definitions for agent capability descriptions.

3. **A unified ecosystem emerges** where tools are discoverable via MCP, agents are discoverable via ACP, and they share common patterns for security, identity, and message serialization.

4. **Protocol gateways** may emerge that translate between MCP and ACP, allowing single tools to be accessed by both individual LLMs (via MCP) and agent collectives (via ACP).

---

## Part V: Implementation Notes

### Getting Started with MCP

**Minimum setup to use MCP (as a user):**
1. Install Claude Desktop (or another MCP-compatible host)
2. Configure MCP servers in the host's configuration file
3. Start interacting with tools through natural language

**Minimum setup to create an MCP server:**
1. Choose an SDK (TypeScript or Python)
2. Import the MCP SDK and create a server instance
3. Define tools using the `tool()` decorator or `addTool()` method
4. Implement the handler functions that execute the tool operations
5. Run the server with stdio or SSE transport

**Example (Python, minimal server):**
```python
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions

app = Server("my-server")

@app.list_tools()
async def list_tools():
    return [
        {
            "name": "greet",
            "description": "Greet a person by name",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name to greet"}
                },
                "required": ["name"]
            }
        }
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "greet":
        return {"content": [{"type": "text", "text": f"Hello, {arguments['name']}!"}]}
    raise ValueError(f"Unknown tool: {name}")

if __name__ == "__main__":
    from mcp.server.stdio import stdio_server
    import anyio
    anyio.run(stdio_server(app))
```

### Getting Started with ACP

Since ACP is still in an experimental phase, the best approach is:

1. **Follow the community discussions:** Watch for developments on GitHub and AI research forums
2. **Experiment with early implementations:** Try proof-of-concept agent coordination systems
3. **Contribute to the specification:** The protocol is still being shaped; community input is valuable
4. **Build on MCP first:** For practical needs today, build MCP servers for your tools; the multi-agent layer can be added later as ACP matures

---

## References and Further Reading

### MCP Resources

- **Official Specification:** https://spec.modelcontextprotocol.io/
- **GitHub Repository:** https://github.com/modelcontextprotocol
- **Python SDK:** https://github.com/modelcontextprotocol/python-sdk
- **TypeScript SDK:** https://github.com/modelcontextprotocol/typescript-sdk
- **Anthropic Blog Post (Nov 2024):** Introducing the Model Context Protocol
- **MCP Servers Directory:** https://github.com/modelcontextprotocol/servers

### ACP Resources

- **Community Discussions:** Various GitHub repositories and forums (search "Agent Communication Protocol")
- **Academic Papers:** Multi-agent communication protocols in AI research (AAAI, NeurIPS, ICML workshops)
- **Related Standards:**
  - FIPA ACL (Foundation for Intelligent Physical Agents - Agent Communication Language) — an early academic standard
  - OWL-S / Semantic Web Services — ontology-based service descriptions
  - DID / Verifiable Credentials — W3C standards relevant to agent identity

### General AI System Architecture

- **Function Calling in LLMs:** OpenAI, Anthropic, Google, and other provider documentation
- **Multi-Agent Systems:** Research from AutoGPT, BabyAGI, Microsoft AutoGen, LangChain, CrewAI
- **Tool-Augmented LLMs:** Papers on Toolformer, Gorilla, and related research

---

## Cross-References

| Reference | Description |
|-----------|-------------|
| [04-AI-Agents-and-Orchestrators.md](04-AI-Agents-and-Orchestrators.md) | How agents and orchestrators use MCP/ACP for communication |
| [05-OpenCode-ClaudeCode-and-Hermes-Agent.md](05-OpenCode-ClaudeCode-and-Hermes-Agent.md) | Real implementations of MCP in coding agents |
| [07-Glossary.md](07-Glossary.md) | Definitions for MCP Host/Client/Server, ACP, Tool, Resource, etc. |
| [08-AI-Roadmap.md](08-AI-Roadmap.md) | Protocol convergence and standardisation trends |

---

*This document is part of the AiBaseKnowledge library. It was compiled from publicly available specifications, documentation, and community resources as of May 2026. Given the rapid evolution of both MCP and ACP, readers should verify specific details against the latest official documentation.*

---
**See also:**
- [07 — AI Coding Agent Sustainability: Managing Hidden Costs, Code Quality Decay, and Developer Burnout](33-AI-Native-Software-Development/07-AI-Coding-Agent-Sustainability.md)
- [48 — MCP Cloud Infrastructure & Agent-as-a-Service: The Production Layer of Agentic AI](48-MCP-Cloud-Infrastructure-Agent-as-a-Service/01-Overview.md)
- [03 — Agent-Native Orchestration: LangGraph, Conductor, Mistral Workflows, Mcp-Agent](31-AI-Workflow-Orchestration-and-Durable-Execution/03-Agent-Native-Orchestration.md)
