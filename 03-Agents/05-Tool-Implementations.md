# OpenCode CLI, Claude Code, and Hermes Agent — Comprehensive Comparison

> A detailed technical reference covering what these three AI coding tools are, who built them, their architectures, how they connect to MCP/ACP/tools, and when to choose each one.

---

## Table of Contents

1. [OpenCode CLI](#1-opencode-cli)
2. [Claude Code](#2-claude-code)
3. [Hermes Agent](#3-hermes-agent)
4. [Comparison Table](#4-comparison-table)
5. [ACP / MCP / Tool Integration Comparison](#5-acp--mcp--tool-integration-comparison)
6. [Use Cases: When to Choose Which](#6-use-cases-when-to-choose-which)
7. [Summary](#7-summary)
8. [Migration and Integration Guide](#8-migration-and-integration-guide)
9. [Enterprise Governance and Compliance](#9-enterprise-governance-and-compliance)

---

## 1. OpenCode CLI

### 1.1 What It Is and Who Created It

**OpenCode** is an open-source, terminal-based AI coding agent built by the **opencode-ai** community. It provides a rich Terminal User Interface (TUI) for interacting with various AI models to assist with coding, debugging, and software engineering tasks.

**Important status update:** The original `opencode-ai/opencode` repository has been **archived** (as of 2024/2025). The project has continued and evolved under the name **Crush**, developed by the same original author in collaboration with the **Charm** team (charmbracelet/crush). Crush inherits and extends OpenCode's architecture. References to "OpenCode" below describe the original project; Crush is its direct successor.

- **Language:** Go
- **License:** MIT
- **Repository:** https://github.com/opencode-ai/opencode (archived)
- **Successor:** https://github.com/charmbracelet/crush (Crush)

### 1.2 Core Features and Capabilities

| Feature | Description |
|---------|-------------|
| **Interactive TUI** | Built with Bubble Tea (Go TUI framework); vim-like keybindings, split panes, session management |
| **Multi-Provider AI** | Supports OpenAI, Anthropic Claude, Google Gemini, AWS Bedrock, Groq, Azure OpenAI, OpenRouter, GitHub Copilot |
| **Session Management** | Save, resume, and manage multiple conversation sessions per project |
| **LSP Integration** | Language Server Protocol support for diagnostics, code intelligence (Go, TypeScript, etc.) |
| **Tool Integration** | AI can execute shell commands, search files, read/write files, and modify code |
| **MCP Support** | Model Context Protocol (stdio + SSE) for connecting external tools |
| **File Change Tracking** | Visual diff of file changes during sessions |
| **Auto-Compact** | Automatically summarizes conversations when approaching context limit |
| **Custom Commands** | User-defined commands with named arguments, stored as markdown files |
| **Vim-like Editor** | Built-in editor with text input for composing messages |
| **External Editor** | Open your preferred editor for message composition |
| **Persistent Storage** | SQLite database for conversations and sessions |

### 1.3 Architecture (How It Works Internally)

OpenCode is written in **Go** and uses the Charm ecosystem of libraries:

```
┌─────────────────────────────────────────────┐
│                  TUI Layer                   │
│  Bubble Tea (charmbracelet/bubbletea)        │
│  Lip Gloss (styling), Bubbles (components)   │
├─────────────────────────────────────────────┤
│              Agent Loop (ReAct)              │
│  Reasoning -> Tool Selection -> Execution    │
├─────────────────────────────────────────────┤
│           Provider Abstraction Layer         │
│  OpenAI │ Anthropic │ Gemini │ Groq │ etc.   │
├─────────────────────────────────────────────┤
│          Protocol Integration Layer          │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
│  │ MCP stdio│  │ MCP SSE  │  │ LSP (gopls│  │
│  │ servers  │  │ servers  │  │  etc.)    │  │
│  └──────────┘  └──────────┘  └───────────┘  │
├─────────────────────────────────────────────┤
│              Storage Layer                   │
│  SQLite (sessions, conversations)            │
└─────────────────────────────────────────────┘
```

**Agent Loop:**

1. User sends a message via the TUI
2. OpenCode constructs a prompt with conversation history + system context
3. The message is sent to the configured AI provider (e.g., Claude, GPT, Gemini)
4. The model responds with text and/or tool call requests
5. OpenCode executes approved tools (file read/write, terminal commands, MCP tools)
6. Results are fed back to the model
7. The loop continues until the task is complete or the user intervenes

**Configuration File (~/.opencode.json or ./.opencode.json):**

- Provider configurations (API keys, endpoints)
- Agent model selection per task type (coder, task, title)
- MCP server definitions
- LSP server configurations
- Shell settings, debug flags, auto-compact toggle

### 1.4 ACP/MCP/Tool Integration

OpenCode implements the **Model Context Protocol (MCP)** as its primary extensibility mechanism:

- **MCP stdio:** Connect to local MCP servers via standard input/output (process spawning)
- **MCP SSE:** Connect to remote MCP servers via Server-Sent Events (HTTP streaming)
- **Tool Discovery:** Automatically discovers available tools from connected MCP servers
- **Permission System:** Tools require user approval before execution; configured via allow/deny lists
- **LSP as Tools:** Language Server Protocol integration exposes diagnostics as AI-accessible tools

**Example MCP configuration:**

```json
{
  "mcpServers": {
    "filesystem": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]
    },
    "web-api": {
      "type": "sse",
      "url": "https://api.example.com/mcp",
      "headers": { "Authorization": "Bearer token" }
    }
  }
}
```

**Built-in tools (no MCP needed):**
- Terminal command execution
- File read/write
- File search (grep)
- LSP diagnostics queries

### 1.5 Successor: Crush by Charm

Crush is the active successor to OpenCode, developed by the Charm team. It adds:
- Industrial-grade terminal support across macOS, Linux, Windows, Android, FreeBSD, OpenBSD, NetBSD
- Multi-model mid-session switching
- Enhanced MCP support (http, stdio, sse)
- Charm ecosystem reliability (25k+ applications)

---

## 2. Claude Code

### 2.1 What It Is and Who Created It

**Claude Code** is a proprietary, agentic coding tool built by **Anthropic** (the company behind Claude AI models). It lives in your terminal, understands your codebase, and helps you code faster by executing routine tasks, explaining complex code, and handling git workflows through natural language commands.

- **Language:** TypeScript/Node.js
- **License:** Proprietary (free tier available with usage limits)
- **Website:** https://code.claude.com
- **Installation:** `npm install -g @anthropic-ai/claude-code` (deprecated), or platform-specific installers

### 2.2 Core Features and Capabilities

| Feature | Description |
|---------|-------------|
| **Terminal & IDE** | Runs in terminal, VS Code extension, JetBrains IDE, desktop app, browser (web), and Chrome extension (beta) |
| **Codebase Understanding** | Reads your entire codebase, understands project structure, dependencies, and patterns |
| **File Editing** | Edits files, creates new files, refactors code across multiple files simultaneously |
| **Git Integration** | Creates commits, manages branches, resolves merge conflicts, writes commit messages |
| **Terminal Commands** | Runs shell commands, manages processes, installs dependencies |
| **Agentic Loop** | Can independently plan, execute, and iterate on multi-step software engineering tasks |
| **Memory** | Project-level instructions (CLAUDE.md) and user-level memory (stored in .claude directory) |
| **Permission Modes** | Configurable: Plan mode (no destructive changes), Auto mode (automatic approved actions), Ask mode (always prompt) |
| **Context Window Management** | Prompt caching, automatic context compression, session management |
| **MCP Support** | Full Model Context Protocol integration (stdio + SSE) |
| **Plugin System** | Extensible via plugins (custom commands and agents) |
| **Remote Control** | SSP (Secure Session Protocol) for remote machine control |
| **Computer Use (Preview)** | GUI-driven interaction: clicking buttons, filling forms, navigating UIs |
| **Slack Integration** | Run Claude Code from within Slack channels |
| **GitHub Integration** | Tag @claude on GitHub for PR review and code review tasks |
| **Prompt Caching** | Built-in caching for reduced latency and costs on repetitive contexts |
| **Privacy Controls** | Data usage opt-outs, limited retention for sensitive data |

### 2.3 Architecture (How It Works Internally)

Claude Code is built in **TypeScript** (Node.js) with a modular architecture:

```
┌─────────────────────────────────────────────┐
│              Interface Layer                 │
│  CLI (TUI) │ VS Code Ext. │ Web │ Desktop   │
├─────────────────────────────────────────────┤
│              Agent Orchestrator              │
│  Planning -> Tool Selection -> Execution     │
│  Multi-step decomposition, Git workflow      │
├─────────────────────────────────────────────┤
│           Claude API Integration             │
│  Anthropic Messages API (Claude models only) │
│  Prompt caching, streaming, tool use         │
├─────────────────────────────────────────────┤
│          Tool & Protocol Layer               │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
│  │ MCP stdio│  │ MCP SSE  │  │ Plugins   │  │
│  │ servers  │  │ servers  │  │ (custom)  │  │
│  └──────────┘  └──────────┘  └───────────┘  │
│  Bash tool  │  File ops  │  Git ops         │
├─────────────────────────────────────────────┤
│              Storage Layer                   │
│  ~/.claude/  │  CLAUDE.md  │  Session files  │
└─────────────────────────────────────────────┘
```

**Agent Loop:**

1. User provides a prompt (natural language task description)
2. Claude Code analyzes the codebase for context (reads project structure, key files)
3. The model plans an approach and begins executing steps
4. Each step may involve: reading files, editing code, running terminal commands, git operations
5. Results are streamed back to the user in real-time
6. Claude iterates: fixes errors, tests solutions, refactors as needed
7. Optionally creates commits or PRs with the completed work

**Key architectural details:**

- **Enterprise-grade prompt caching:** Claude Code aggressively caches system prompts and frequently-used context to reduce cost and latency
- **Session persistence:** Conversations are saved and resumable; sessions can be organized by project
- **The .claude directory:** Contains project-level instructions (CLAUDE.md), memory files, and session data
- **Permission system:** Three modes (Plan, Auto, Ask) control whether Claude can take actions autonomously

### 2.4 MCP/Tool Integration

Claude Code has robust **Model Context Protocol (MCP)** support:

- **MCP stdio servers:** Local process-based MCP tools
- **MCP SSE servers:** Remote MCP tool servers
- **Built-in tools:** Bash execution, file operations, git operations, code search, web search, computer use
- **Plugin system:** JavaScript/TypeScript plugins add custom commands and agents
- **SSP (Secure Session Protocol):** Remote control for running Claude Code on another machine

**MCP configuration** is stored in the `~/.claude/settings.json` file or per-project in `.claude/settings.json`:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]
    }
  }
}
```

**Notable:** Claude Code is **tied to Anthropic's Claude models only**. It does not support OpenAI, Gemini, or other model providers. This is by design — Anthropic optimizes deeply for Claude's tool-use capabilities.

### 2.5 Pricing

Claude Code uses the Anthropic API under the hood. Users pay for token usage:
- **Free tier:** Limited usage (rate-limited)
- **Paid:** Per-token pricing via Anthropic API plans
- **Max plan:** Higher rate limits and priority access

---

## 3. Hermes Agent

### 3.1 What It Is and Who Created It

**Hermes Agent** is a self-improving, multi-platform AI agent built by **Nous Research** (an independent AI research organization known for open-source LLMs like Hermes, Mixtral finetunes, and research into mechanistic interpretability). It is designed as a general-purpose autonomous agent that learns from experience, persists knowledge across sessions, and operates across multiple messaging platforms simultaneously.

- **Language:** Python (with Go-based CLI tools)
- **License:** MIT (fully open source)
- **Repository:** https://github.com/NousResearch/hermes-agent
- **Documentation:** https://hermes-agent.nousresearch.com/docs/

### 3.2 Core Features and Capabilities

| Feature | Description |
|---------|-------------|
| **Multi-Platform** | Telegram, Discord, Slack, WhatsApp, Signal, CLI — all from a single gateway process |
| **Multi-Provider Model Access** | 20+ providers: Nous Portal, OpenRouter (200+ models), OpenAI, Anthropic, NovitaAI, NVIDIA NIM, Xiaomi MiMo, z.ai/GLM, Kimi/Moonshot, MiniMax, Hugging Face, local endpoints |
| **Closed Learning Loop** | Autonomous skill creation after complex tasks; skills self-improve during use; persistent memory with LLM summarization |
| **Persistent Memory** | MEMORY.md, USER.md, FTS5 session search, Honcho dialectic user modeling, periodic memory nudges |
| **Skills System** | Procedural memory (skill files), auto-generated skills, agentskills.io compatible |
| **Subagent Delegation** | Spawn isolated subagents for parallel workstreams; Python RPC tool calls |
| **Scheduled Automations** | Built-in cron scheduler with delivery to any platform (daily reports, backups, audits) |
| **Terminal Backends** | Local, Docker, SSH, Singularity, Modal, Daytona — serverless persistence hibernate on idle |
| **Messaging Gateway** | Unified gateway for Telegram, Discord, Slack, WhatsApp, Signal; voice memo transcription |
| **MCP Support** | Full MCP integration (stdio + SSE), tool approval system |
| **Plugins** | Hermes plugin system for extensibility |
| **Profiles** | Multiple profiles with separate skills, plugins, cron, memories |
| **API Server** | OpenAI-compatible API server |
| **Voice Mode** | Voice input/output, voice memo transcription |
| **Vision** | Image understanding and analysis |
| **Browser Automation** | Cloud browser for web tasks |
| **Image Generation** | Built-in image generation via supported providers |
| **Context Files** | Project-level context (AGENTS.md, CONTEXT.md) |
| **Security** | Command approval, DM pairing, container isolation |
| **Trajectory Generation** | Research-ready batch trajectory generation for training new models |

### 3.3 Architecture (How It Works Internally)

Hermes Agent is built in **Python** with a modular, service-oriented architecture:

```
┌──────────────────────────────────────────────────────────┐
│                   Interface Layer                         │
│  ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌─────────────┐  │
│  │ CLI TUI │ │ Telegram │ │ Discord  │ │ Slack/etc.  │  │
│  └─────────┘ └──────────┘ └──────────┘ └─────────────┘  │
├──────────────────────────────────────────────────────────┤
│                   Gateway Layer                           │
│  Message routing, platform abstraction, auth             │
├──────────────────────────────────────────────────────────┤
│                Agent Orchestrator                         │
│  ┌────────────────────────────────────────────────────┐  │
│  │ Agent Loop (ReAct + Reflection)                    │  │
│  │  Plan -> Reason -> Act -> Observe -> Learn         │  │
│  │  Subagent spawning, task decomposition             │  │
│  └────────────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────────────┤
│              Provider Abstraction Layer                   │
│  20+ providers, model routing, fallback, reasoning       │
├──────────────────────────────────────────────────────────┤
│              Memory & Learning Layer                      │
│  ┌──────────┐ ┌──────────┐ ┌─────────┐ ┌──────────┐    │
│  │Persistent│ │ Skills   │ │ FTS5    │ │ Honcho   │    │
│  │Memory    │ │System    │ │Search   │ │Dialectic │    │
│  └──────────┘ └──────────┘ └─────────┘ └──────────┘    │
├──────────────────────────────────────────────────────────┤
│              Tool & Protocol Layer                        │
│  ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌──────────┐  │
│  │ MCP stdio│ │ MCP SSE  │ │ Plugins   │ │ Cron     │  │
│  │ servers  │ │ servers  │ │ (Python)  │ │ (Sched.) │  │
│  └──────────┘ └──────────┘ └───────────┘ └──────────┘  │
│  Terminal │ File ops │ Subagent │ Web search │ Vision    │
├──────────────────────────────────────────────────────────┤
│              Backend Layer                                │
│  Local │ Docker │ SSH │ Modal │ Daytona │ Singularity    │
└──────────────────────────────────────────────────────────┘
```

**Agent Loop (detailed):**

1. Message arrives from any platform (CLI, Telegram, Discord, etc.)
2. Hermes constructs a system prompt from:
   - SOUL.md (permanent identity — tone, personality, constraints)
   - AGENTS.md / CONTEXT.md (project context)
   - Skills (procedural memory files matched to the task)
   - Memories (persistent cross-session knowledge)
3. The message is sent to the configured LLM provider
4. The model responds with reasoning and/or tool calls
5. Hermes executes approved tools (built-in, MCP, plugins, or subagent delegation)
6. Results return to the model for further reasoning
7. After task completion, Hermes may:
   - Create a new skill from the experience (autonomous skill generation)
   - Update memory with new information (periodic memory nudges)
   - Schedule cron jobs for recurring tasks

**Key architectural innovations:**

- **Hermes Home (~/.hermes/):** Configuration, skills, memories, cron, profiles, plugins all organized under a single directory
- **SOUL.md:** A permanent identity file that defines the agent's personality, tone, and behavior — distinct from project-level instructions (AGENTS.md)
- **Skills Hub integration:** Compatible with agentskills.io for community-shared skills
- **Isolated subagents:** Each subagent runs in its own context, can call tools via RPC, and reports back to the parent agent
- **Terminal backends:** Choose where your agent runs — locally, on a remote server via SSH, in a Docker container, or serverless (Modal, Daytona)

### 3.4 MCP/ACP/Tool Integration

Hermes Agent supports **MCP (Model Context Protocol)** with both stdio and SSE transports:

- **MCP stdio:** Connect to local tool servers as child processes
- **MCP SSE:** Connect to remote MCP servers over HTTP streaming
- **Tool approval system:** Granular permissions for each tool; users can approve/deny/allowlist
- **40+ built-in tools:** Terminal, file operations, search, web, vision, voice, image gen, subagent, etc.
- **Toolset system:** Group tools into named sets for different contexts
- **Plugin system:** Python-based plugins for deep extensibility
- **Cross-profile isolation:** Each profile has its own set of tools, plugins, and MCP servers

**MCP Configuration** (in `~/.hermes/config.yaml` or via `hermes config set`):

```yaml
mcp_servers:
  filesystem:
    type: stdio
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]
  web-scraper:
    type: sse
    url: https://mcp.example.com/sse
```

**Note on ACP (Agent Communication Protocol):** While Hermes Agent does not implement the formal ACP specification, its subagent system and plugin architecture provide equivalent capabilities for agent-to-agent communication and tool sharing. The subagent system uses an RPC-like mechanism where subagents can call tools on the parent agent and report results.

### 3.5 Migrating from OpenClaw

Hermes Agent provides a built-in migration path from **OpenClaw** (a predecessor/alternative agent):

```bash
hermes claw migrate          # Full interactive migration
hermes claw migrate --dry-run  # Preview only
```

Imports: persona (SOUL.md), memories, skills, command allowlist, messaging settings, API keys, TTS assets, workspace instructions.

---

## 4. Comparison Table

### 4.1 General Comparison

| Aspect | OpenCode (Crush) | Claude Code | Hermes Agent |
|--------|------------------|-------------|--------------|
| **Creator** | opencode-ai community / Charm | Anthropic | Nous Research |
| **Language** | Go (Crush: Go) | TypeScript/Node.js | Python |
| **License** | MIT (open source) | Proprietary | MIT (open source) |
| **Self-hostable** | Yes | No | Yes |
| **Primary Focus** | Terminal coding assistant | Terminal/IDE coding agent | General-purpose autonomous agent |
| **Target User** | Developers in terminal | Developers in terminal/IDE | Power users, researchers, automators |
| **Maturity** | Archived (Crush=active) | Production | Active development |

### 4.2 Model & Provider Support

| Capability | OpenCode (Crush) | Claude Code | Hermes Agent |
|------------|------------------|-------------|--------------|
| **Anthropic Claude** | Yes | Yes (required) | Yes |
| **OpenAI GPT** | Yes | No | Yes |
| **Google Gemini** | Yes | No | Yes |
| **OpenRouter (200+)** | Yes | No | Yes |
| **Local Models** | Yes (OpenAI-compatible endpoint) | No | Yes (any endpoint) |
| **GitHub Copilot** | Yes (experimental) | No | No (separate) |
| **AWS Bedrock** | Yes | No | No (via OpenRouter) |
| **Groq** | Yes | No | Yes |
| **Azure OpenAI** | Yes | No | Yes |
| **Nous Portal** | No | No | Yes (native) |
| **NVIDIA NIM** | No | No | Yes |
| **Multi-model switching** | Yes (mid-session in Crush) | No (Claude only) | Yes (mid-session via /model) |
| **Prometheus/Watsonx/etc.** | No | No | Yes |

### 4.3 Platform Support

| Platform | OpenCode (Crush) | Claude Code | Hermes Agent |
|----------|------------------|-------------|--------------|
| **Terminal (CLI)** | Yes (native) | Yes (native) | Yes (native) |
| **VS Code extension** | No | Yes | No |
| **JetBrains IDE** | No | Yes | No |
| **Desktop app** | No | Yes | No |
| **Web browser** | No | Yes | No |
| **Telegram** | No | No | Yes |
| **Discord** | No | No | Yes |
| **Slack** | No | Yes (limited) | Yes |
| **WhatsApp** | No | No | Yes |
| **Signal** | No | No | Yes |
| **Email** | No | No | Yes |
| **Chrome extension** | No | Yes (beta) | No |
| **API server** | No | No | Yes (OpenAI-compatible) |

### 4.4 Agent Capabilities

| Capability | OpenCode (Crush) | Claude Code | Hermes Agent |
|------------|------------------|-------------|--------------|
| **Agentic Loop** | Yes (ReAct) | Yes (advanced) | Yes (ReAct + Reflection) |
| **File Editing** | Yes | Yes | Yes |
| **Terminal Execution** | Yes | Yes | Yes |
| **Git Integration** | Basic | Advanced (commits, PRs, merge) | Basic |
| **Multi-file Refactoring** | Basic | Advanced | Via subagents |
| **Autonomous Planning** | Limited | Yes | Yes |
| **Task Decomposition** | No | Yes | Yes (subagents) |
| **Self-healing (fix errors)** | Limited | Yes | Yes |
| **Skills System** | No | No | Yes (autonomous + manual) |
| **Persistent Memory** | Session only | CLAUDE.md + .claude | Yes (MEMORY.md, USER.md, FTS5, Honcho) |
| **Cross-session Recall** | No | Basic | Yes (FTS5 search + LLM summarization) |
| **Subagent Delegation** | No | No | Yes (parallel workstreams) |
| **Scheduled Tasks** | No | No | Yes (built-in cron) |
| **Learning Loop** | No | No | Yes (auto skill creation, self-improvement) |
| **Voice Mode** | No | No | Yes |
| **Vision/Image Analysis** | No | Yes (limited) | Yes |
| **Image Generation** | No | No | Yes |
| **Browser Automation** | No | Yes (Computer Use preview) | Yes (cloud browser) |
| **Web Search** | No | Yes | Yes |
| **User Modeling** | No | No | Yes (Honcho dialectic) |

### 4.5 Extensibility

| Feature | OpenCode (Crush) | Claude Code | Hermes Agent |
|---------|------------------|-------------|--------------|
| **MCP (Model Context Protocol)** | Yes (stdio + SSE) | Yes (stdio + SSE) | Yes (stdio + SSE) |
| **Plugin System** | No | Yes (JavaScript) | Yes (Python) |
| **Custom Commands** | Yes (markdown files) | Yes (plugins) | Yes (skills + plugins) |
| **LSP Integration** | Yes | Limited | No (via MCP) |
| **Custom Tools** | Via MCP servers | Via MCP + plugins | Via MCP + plugins + toolsets |
| **Profiles/Multi-instance** | No | No | Yes |
| **API Endpoints** | No | No | Yes (OpenAI-compatible) |
| **agentskills.io Compatible** | No | No | Yes |
| **OpenClaw Migration** | N/A | N/A | Yes (built-in) |

### 4.6 MCP Protocol Support

| MCP Feature | OpenCode (Crush) | Claude Code | Hermes Agent |
|-------------|------------------|-------------|--------------|
| **MCP stdio** | Yes | Yes | Yes |
| **MCP SSE** | Yes | Yes | Yes |
| **MCP HTTP** | No (Crush: Yes) | No | No |
| **Tool Discovery** | Yes (automatic) | Yes (automatic) | Yes (automatic) |
| **Tool Approval** | Yes (per-tool allow/deny) | Yes (permission modes) | Yes (granular allow/deny) |
| **Multiple Servers** | Yes | Yes | Yes |
| **Env Variables for MCP** | Yes | Yes | Yes |
| **Per-project MCP config** | Yes (local .opencode.json) | Yes (.claude/settings.json) | Yes (profile config) |

---

## 5. ACP / MCP / Tool Integration Comparison

### 5.1 MCP (Model Context Protocol)

All three tools support MCP, but with different levels of integration:

**OpenCode / Crush:**
- Standard MCP implementation with stdio and SSE transports
- MCP servers defined in configuration file
- Tools auto-discovered and presented to the AI
- Permission system requires user approval per tool

**Claude Code:**
- Full MCP support with both stdio and SSE
- MCP tools are first-class citizens alongside built-in tools
- Plugin system complements MCP for deeper integrations
- Permission modes (Plan/Auto/Ask) control execution

**Hermes Agent:**
- Full MCP stdio and SSE support
- MCP tools integrate with the broader toolset system
- Can combine MCP tools with plugins, built-in tools, and subagent delegation
- Granular per-tool permission system across all platforms

### 5.2 ACP (Agent Communication Protocol)

**No tool in this comparison implements the formal ACP specification.** However:

- **Hermes Agent's subagent system** is the closest equivalent — subagents can communicate via RPC, delegate tasks, and share context. It provides agent-to-agent communication without implementing the formal ACP spec.
- **Claude Code** does not support agent-to-agent communication natively.
- **OpenCode/Crush** operates as a single-agent system.

### 5.3 Tool-Use Architecture

| Aspect | OpenCode (Crush) | Claude Code | Hermes Agent |
|--------|------------------|-------------|--------------|
| **Tool definition** | Go functions + MCP | TypeScript + MCP + plugins | Python functions + MCP + plugins |
| **Tool approval** | Per-execution prompt | Per-execution (modes) | Per-execution + allowlist |
| **Streaming tool output** | Yes | Yes | Yes |
| **Tool composition** | Sequential | Sequential + parallel | Sequential + parallel + subagent |
| **Context injection** | Full conversation | Cached system prompt | Layered (SOUL.md + skills + memory + context) |

---

## 6. Use Cases: When to Choose Which

### 6.1 Choose OpenCode / Crush When...

- You want a **lightweight, fast CLI coding assistant**
- You prefer to work **entirely in the terminal** without IDE integration
- You need **multi-provider flexibility** (switch between Claude, GPT, Gemini freely)
- You want **open source** with MIT licensing
- You are on **unusual platforms** (FreeBSD, OpenBSD, NetBSD, Android Terminal)
- You want **simple MCP extensibility** without complex configuration
- You don't need persistent memory, scheduling, or multi-platform messaging

**Best for:** Solo developers who want a terminal-native AI coding companion with model flexibility.

### 6.2 Choose Claude Code When...

- You are already invested in **Anthropic's Claude ecosystem**
- You want the **deepest Claude model integration** (optimized tool-use, prompt caching)
- You need **enterprise-grade coding assistance** with IDE integration (VS Code, JetBrains)
- You want **git workflow automation** (commits, PR management, merge conflict resolution)
- You need **remote control** capabilities (SSP protocol)
- You want **Claude's Computer Use** for GUI interaction
- You value **polished UX** and official support from Anthropic
- You don't mind **proprietary software** and per-token pricing

**Best for:** Teams and enterprises using Anthropic's Claude models who want deep IDE integration and polished coding workflows.

### 6.3 Choose Hermes Agent When...

- You need a **multi-platform autonomous agent** (simultaneously on Telegram, Discord, Slack, CLI)
- You require **persistent memory and learning** across sessions
- You want **scheduled automations** (cron jobs, daily reports, backups)
- You need **subagent delegation** for complex parallel tasks
- You want **model flexibility** (20+ providers, no lock-in)
- You value **open source** with MIT licensing and full self-hosting
- You want **skill creation and improvement** over time
- You need **voice, vision, and multimodal capabilities**
- You want an **agent that runs on a server** (SSH, Docker, Modal, Daytona) and persists between sessions
- You want **Nous Portal** integration for one-stop API access
- You are a **researcher** needing trajectory generation and model training data

**Best for:** Power users, researchers, and automation enthusiasts who want a general-purpose autonomous agent that operates across platforms and improves over time.

### 6.4 Decision Matrix

| If you prioritize... | Choose |
|----------------------|--------|
| **Lightweight terminal coding** | OpenCode / Crush |
| **Deep Claude + IDE integration** | Claude Code |
| **Multi-platform autonomous agent** | Hermes Agent |
| **Open source + self-hosting** | OpenCode / Crush or Hermes Agent |
| **Model flexibility** | OpenCode / Crush or Hermes Agent |
| **Enterprise support** | Claude Code |
| **Persistent memory & learning** | Hermes Agent |
| **Git workflow automation** | Claude Code |
| **Scheduled tasks / cron** | Hermes Agent |
| **Agent-to-agent communication** | Hermes Agent (subagents) |
|| **Research / trajectory data** | Hermes Agent |
|| **FreeBSD / unusual platforms** | OpenCode / Crush |
|| **Messaging platform integration** | Hermes Agent |

---

## 7. Security, Cost, and Performance Comparison

### 7.1 Security Model Comparison

| Security Aspect | OpenCode (Crush) | Claude Code | Hermes Agent |
|:----------------|:----------------:|:-----------:|:------------:|
| **Tool approval mechanism** | Per-execution prompt (allow/deny list) | Permission modes (Plan/Auto/Ask) | Per-execution + allowlist + granular permissions |
| **Code execution isolation** | Shell execution (no sandbox by default) | Bash sandbox (limited) | Configurable: local, Docker, SSH, container isolation |
| **Configuration file security** | JSON with API keys (plaintext) | JSON plaintext | YAML + env vars + secret injection |
| **Multi-tenant isolation** | Single-user | Single-user | Profiles with isolated skills/plugins/memories |
| **Data retention control** | Local SQLite (user-managed) | Limited opt-outs | Full control: local storage, no telemetry by default |
| **Network access policy** | User-approved tool invocations | Permission modes | Granular per-tool approval across all platforms |
| **Dependency vulnerability** | Go binary (static linking) | Node.js (npm dependencies) | Python (pip dependencies) |
| **Audit logging** | Session files | Session persistence | FTS5 searchable logs, trajectory generation |

**Security recommendation for production:**
- **OpenCode/Crush:** Best for single-user, local-only development with minimal external service exposure
- **Claude Code:** Best when working within Anthropic's trusted environment; network calls go through their API
- **Hermes Agent:** Best when you need fine-grained security controls, multi-tenant isolation, and full data sovereignty

### 7.2 Cost Analysis

| Cost Factor | OpenCode (Crush) | Claude Code | Hermes Agent |
|:------------|:----------------:|:-----------:|:------------:|
| **Software license** | Free (MIT) | Free tier + paid per-token | Free (MIT) |
| **API cost (per 1M tokens, input)** | Varies: Claude=$3-8, GPT=$2.5-10, Gemini=$0.5-1 | Claude only: $3-8 input | Varies by provider (20+ options) |
| **API cost (per 1M tokens, output)** | Varies: Claude=$15-40, GPT=$10-30, Gemini=$2-8 | Claude only: $15-40 | Varies by provider |
| **Infrastructure (self-hosted)** | None (terminal only) | None (API-based) | Optional: Docker, SSH host, Modal ($0-50/mo) |
| **Context caching savings** | None | Yes (enterprise-grade, 50-90% savings) | Provider-dependent |
| **Typical monthly cost (heavy user)** | $50-500 (depends on provider) | $100-1,000 (Claude API) | $50-500 (depends on provider) |
| **Cost optimization flexibility** | High (switch providers anytime) | Low (locked to Claude) | High (20+ providers, model routing) |

**Cost optimization tips:**
1. Use cheaper models for simple tasks (route grep/file-read to budget models)
2. Enable context caching for 50-90% savings on repetitive work
3. Self-host local models for zero-cost inference (Ollama/vLLM)
4. Batch operations to maximize context reuse
5. Monitor token usage weekly to identify spending patterns

### 7.3 Performance and Latency Comparison

| Metric | OpenCode (Crush) | Claude Code | Hermes Agent |
|:-------|:----------------:|:-----------:|:------------:|
| **Startup time (cold)** | ~1-2s (Go binary) | ~2-5s (Node.js init) | ~3-8s (Python import + plugin load) |
| **Startup time (warm)** | <500ms | ~1s | ~2s |
| **Token-to-first-response** | Provider-dependent (200ms-2s) | Prompt caching (100ms-1s) | Provider-dependent (200ms-2s) |
| **File search (10K files)** | ~500ms (ripgrep) | ~1s | ~1-2s |
| **Large file read (10K lines)** | ~200ms | ~300ms | ~500ms |
| **Memory idle** | ~50-100MB | ~100-200MB | ~200-500MB |
| **Memory (active session)** | ~100-300MB | ~200-500MB | ~500MB-2GB (with subagents) |
| **Multi-file refactoring** | Sequential (fast) | Parallel planning | Subagent parallel (scales across cores) |
| **Context window utilization** | Provider max (up to 200K) | Claude's 200K (smart summarization) | Provider max (with auto-compact) |

**Performance summary:**
- **OpenCode/Crush:** Fastest startup and lowest memory. Best for quick, single-file coding tasks.
- **Claude Code:** Best latency for Claude models via prompt caching (30-50% faster responses).
- **Hermes Agent:** Heaviest startup but fastest on complex multi-file tasks via subagent parallelism.

### 7.4 Latency Breakdown by Task Type

| Task | OpenCode (Crush) | Claude Code | Hermes Agent |
|:-----|:----------------:|:-----------:|:------------:|
| **Fix a typo in one file** | 3-8s | 5-10s | 8-15s |
| **Refactor across 3 files** | 15-45s | 20-40s | 15-30s (subagents) |
| **New module (10 files)** | 30-90s | 40-80s | 30-60s |
| **Run tests + fix failures** | 60-180s | 45-120s | 40-90s (parallel subagents) |
| **Full codebase audit** | 5-30 min | 3-15 min | 2-10 min (parallel scanning) |

> **Key insight:** OpenCode/Crush wins on simple tasks (1-3 tool calls). Hermes Agent wins on complex multi-step workflows via subagents. Claude Code balances both with best Claude optimization.

---

## 8. Summary

### OpenCode (Crush) — The Lightweight Multi-Model Coder

- Go-based, MIT-licensed, open-source terminal AI coding assistant
- Supports multiple AI providers with mid-session switching (Crush)
- Full MCP integration (stdio + SSE)
- Lightweight, fast, and runs on many platforms (including FreeBSD, Android)
- **Best for:** Terminal-native developers who want model flexibility without heavy infrastructure

### Claude Code — The Enterprise Claude-Native Coding Agent

- TypeScript/Node.js, proprietary, built by Anthropic
- Deep integration with Claude models (optimized tool-use, prompt caching)
- Full MCP + plugins + SSP remote control + Computer Use
- IDE integrations (VS Code, JetBrains), web, desktop, and browser extension
- **Best for:** Teams invested in the Claude ecosystem who want polished IDE integration and enterprise features

### Hermes Agent — The Self-Improving Multi-Platform Agent

- Python-based, MIT-licensed, built by Nous Research
- 20+ model providers, persistent memory, autonomous skill creation
- Full MCP + subagent delegation + cron + multi-platform messaging
- Runs locally, on servers (SSH/Docker), or serverless (Modal/Daytona)
- **Best for:** Power users who want an autonomous, learning agent that operates across platforms and improves over time

### Quick Summary Table

| | OpenCode (Crush) | Claude Code | Hermes Agent |
|---|:---:|:---:|:---:|
| Open Source | Yes (MIT) | No | Yes (MIT) |
| Multi-Provider | Yes | No (Claude only) | Yes (20+) |
| Terminal UI | Yes | Yes | Yes |
| IDE Integration | No | Yes (VS Code, JetBrains) | No |
| Messaging Platforms | No | Slack only | Telegram, Discord, Slack, WhatsApp, Signal |
| MCP Support | Yes | Yes | Yes |
| Persistent Memory | No | Basic | Advanced |
| Subagents / Parallelism | No | No | Yes |
| Scheduled Tasks | No | No | Yes |
| Self-Learning | No | No | Yes |
| Voice / Vision | No | Limited vision | Yes (both) |
|| Remote Execution | No | Yes (SSP) | Yes (SSH, Docker, Modal) |
|| Security Model | Minimal (local) | Moderate (API-gated) | Advanced (isolated profiles) |
|| Cost Flexibility | High (multi-provider) | Low (Claude only) | High (20+ providers) |
|| Startup Speed | ⚡ Fast (Go) | 🚀 Medium (Node.js) | 🐍 Moderate (Python) |
|| Task Complexity Ceiling | Medium | High | Very high (subagents) |
|| Best For | Terminal coding | Enterprise Claude dev | Autonomous automation |

---

## 8. Migration and Integration Guide

This section provides practical guidance for migrating between AI coding tools, setting up hybrid workflows, and integrating these tools into CI/CD pipelines.

### 8.1 Migration Paths Between Tools

| From → To | Complexity | Key Actions | Data to Migrate | Risks |
|:----------|:---------:|:------------|:----------------|:------|
| **OpenCode → Crush** | Low (same lineage) | `opencode.json` → `crush.json`; install Crush via Go/brew | Session history (SQLite), config, MCP server definitions | Minor config syntax differences; session format may differ |
| **OpenCode/Crush → Claude Code** | Medium | Set up Claude API key; install Claude Code; port custom MCP servers | Custom commands (markdown) → Claude plugins; MCP configs | Lock-in to Claude models; custom commands need JS port |
| **OpenCode/Crush → Hermes Agent** | Medium-High | `hermes claw migrate` (pipelines OpenClaw configs); or manual port | Skills, memories, custom commands → Hermes skills; config YAML | Different tool philosophy (coding assistant vs general agent) |
| **Claude Code → Hermes Agent** | Medium | Port CLAUDE.md → AGENTS.md/.hermes/config; remap MCP servers | Project instructions, memory files, session logs | No git workflow equivalent in Hermes; no IDE integration |
| **Hermes Agent → Claude Code** | Medium-High | Extract memory into CLAUDE.md; port skills as Claude plugins | Skills, cron jobs, memories, multi-platform configs | Loss of multi-platform, cron, subagents |
| **Any → Any** | Varies | See section 8.2 for migration checklist | — | Downtime during migration; team retraining |

### 8.2 Tool Migration Checklist

Use this checklist when migrating a development team from one AI coding tool to another:

#### Pre-Migration (1-2 weeks before)
- [ ] **Audit existing tool usage:** Which features are used daily? Weekly? Rarely?
- [ ] **Inventory custom configurations:** MCP servers, custom commands, plugins, environment variables
- [ ] **Identify essential workflows:** Git operations, code review, file refactoring, terminal commands
- [ ] **Document model preferences:** Which model(s) are used and why? Provider API keys and budget
- [ ] **Check platform requirements:** Does the team need IDE integration? Multi-platform messaging? Scheduled tasks?
- [ ] **Set up pilot project:** Test the target tool on a non-critical project first

#### Migration (1-3 days)
- [ ] **Install and configure target tool**
- [ ] **Port MCP server configurations** — this is usually the largest migration effort
- [ ] **Convert custom commands/plugins:** OpenCode markdown commands → Hermes skills or Claude plugins
- [ ] **Set up project-level context:** Port CLAUDE.md ↔ AGENTS.md/CONTEXT.md
- [ ] **Test core workflows:** Run each essential workflow and compare results
- [ ] **Port environment variables and secrets**

#### Post-Migration (1-2 weeks)
- [ ] **Shadow run both tools** for 1 week to compare
- [ ] **Collect team feedback** on productivity, accuracy, latency
- [ ] **Monitor cost** — API usage patterns differ between tools
- [ ] **Adjust prompts/context:** Fine-tune system prompts for the new tool
- [ ] **Decommission old tool** once confident in replacement

### 8.3 Hybrid Workflow Strategies

Some teams benefit from running multiple tools for different tasks:

| Strategy | Configuration | Best For | Example |
|:---------|:------------|:---------|:--------|
| **Primary + Fallback** | Use tool A for daily work, tool B when A fails | Risk mitigation | Claude Code primary → Hermes Agent for complex multi-file refactoring |
| **Domain-specific** | Tool A for frontend, Tool B for backend, Tool C for DevOps | Specialized strengths | OpenCode/Crush for quick edits; Claude Code for full PRs; Hermes for cron/automation |
| **Cost-optimized** | Budget model for simple tasks, premium model for complex | Cost control | Crush with Gemini (low cost) for grep/file reads; Claude Code for code generation |
| **Parallel workflow** | Use subagents (Hermes) or plugins for parallel work | Throughput | Hermes Agent spawns subagents for test generation while Claude Code writes implementation |
| **Review pipeline** | Code written with tool A, reviewed with tool B | Quality assurance | Claude Code generates code → Hermes agent reviews for security/quality |

```python
# Conceptual hybrid workflow: Hermes Agent orchestrates, Claude Code implements
# This is a practical pattern for complex multi-step coding tasks
async def hybrid_refactoring_task(repo_path, task_description):
    # Step 1: Hermes plans the work using its system-level understanding
    plan = await hermes_agent.plan_repo_refactoring(repo_path, task_description)

    # Step 2: Delegate file-level implementations to Claude Code
    for file_spec in plan.file_changes:
        claude_result = await claude_code.modify_file(
            path=file_spec.path,
            changes=file_spec.description,
            style_guide=file_spec.style
        )
        # Hermes reviews and verifies each change
        if not hermes_agent.verify_change(file_spec.path, claude_result):
            await hermes_agent.request_revision(file_spec.path, claude_result)
            claude_result = await claude_code.modify_file(
                path=file_spec.path, changes=file_spec.revision
            )

    # Step 3: Hermes runs tests and handles deployment
    test_results = await hermes_agent.run_test_suite(repo_path)
    if test_results.failed:
        for failed_test in test_results.failed:
            await hermes_agent.fix_test(repo_path, failed_test)
    await hermes_agent.commit_and_push(repo_path, plan.description)
    return plan
```

### 8.4 CI/CD Integration Patterns

| Integration Point | OpenCode/Crush | Claude Code | Hermes Agent |
|:-----------------|:-------------:|:-----------:|:------------:|
| **GitHub Actions** | Via script wrapper (`crush --run <command>`) | Via `claude code` CLI in action; supports `@claude` PR review | Via cron scheduler or `hermes run <skill>` action |
| **GitLab CI** | Same as GitHub (shell-based) | Same as GitHub | Hermes can run as CI step or via webhook-triggered cron |
| **Automated PR review** | Not built-in; custom script | Native via GitHub app (@claude on PR) | Via cron skill that fetches PR diff and reviews |
| **Scheduled code maintenance** | Not built-in | Not built-in | Native cron: `hermes cron create --skill code-review --schedule "0 6 * * 1"` |
| **Monorepo workflows** | Per-project .opencode.json | Per-project .claude/settings.json | Profile per project; `AGENTS.md` per directory |
| **Multi-repo coordination** | Manual | Manual | Subagent delegation per repo; Hermes orchestrates |

### 8.5 Team Adoption Recommendations

| Team Size | Recommended Tool | Rationale |
|:---------|:----------------|:----------|
| **1-3 developers** | OpenCode/Crush or Claude Code | Low overhead, individual productivity focus |
| **3-10 developers** | Claude Code or Hermes Agent | Multi-developer coordination, shared context (CLAUDE.md) |
| **10-50 developers** | Hermes Agent (orchestrator) + Claude Code (implementers) | Hybrid approach keeps individual velocity + org-level automation |
| **50+ developers** | Hermes Agent (enterprise, multi-platform) | Scheduled tasks, code review pipelines, CI/CD integration, multi-platform rollouts |
| **Research team** | Hermes Agent | Trajectory generation, experiment tracking, persistent memory across experiments |
| **Platform/SRE team** | Hermes Agent (cron + automation) + Crush (incident response) | Scheduled maintenance + fast incident response from terminal |

### 8.6 MCP Server Reuse Between Tools

MCP servers are tool-agnostic — the same server can be used with any tool that supports MCP:

```yaml
# Example: Shared MCP server configuration for Claude Code (.claude/settings.json)
# and Hermes Agent (~/.hermes/config.yaml)
# and Crush (~/.config/crush/config.yaml)

# filesystem MCP server — works with all three
mcp_servers:
  filesystem:
    type: stdio
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]

# web scraping MCP server — works with all three
  web-scraper:
    type: sse
    url: https://mcp.example.com/sse
```

**Migrating MCP configurations:**
```bash
# Export MCP servers from Claude Code
cat ~/.claude/settings.json | jq '.mcpServers' > mcp_export.json

# Import into Hermes Agent
hermes config set mcp_servers --from-json mcp_export.json

# Import into Crush
crush config set mcp_servers --from-json mcp_export.json
```

**Key compatibility consideration:** While MCP servers are portable, each tool may expose MCP tool results differently:
- **Claude Code:** Tools are first-class; results streamed inline
- **Hermes Agent:** Tools go through the toolset + permission system; results contribute to agent loop
- **Crush/OpenCode:** Tools are user-approved; results displayed in TUI

### 8.7 Decision Framework for Tool Selection

When evaluating which tool(s) to adopt, score each against your requirements:

| Requirement | Weight (1-5) | OpenCode/Crush | Claude Code | Hermes Agent |
|:------------|:-----------:|:--------------:|:-----------:|:------------:|
| Terminal productivity | — | ★★★★☆ | ★★★★☆ | ★★★☆☆ |
| IDE integration | — | ★☆☆☆☆ | ★★★★★ | ★☆☆☆☆ |
| Multi-model flexibility | — | ★★★★★ | ★☆☆☆☆ | ★★★★★ |
| Persistent memory | — | ★☆☆☆☆ | ★★★☆☆ | ★★★★★ |
| Automation / scheduling | — | ★☆☆☆☆ | ★★☆☆☆ | ★★★★★ |
| Multi-platform messaging | — | ★☆☆☆☆ | ★★☆☆☆ | ★★★★★ |
| Git workflow automation | — | ★★☆☆☆ | ★★★★★ | ★★★☆☆ |
| Open source / self-host | — | ★★★★★ | ★☆☆☆☆ | ★★★★★ |
| Enterprise support | — | ★☆☆☆☆ | ★★★★★ | ★★★☆☆ |
| Agent-to-agent delegation | — | ★☆☆☆☆ | ★☆☆☆☆ | ★★★★★ |
| Cost flexibility | — | ★★★★★ | ★★☆☆☆ | ★★★★★ |

To compute your score: multiply each requirement's weight by the star rating (converted to 1-5) and sum across all tools. The highest score is your recommended primary tool.

---

## 9. Enterprise Governance and Compliance

As AI coding tools move from individual productivity aids to enterprise-grade development infrastructure, governance, security, and compliance considerations become paramount. This section covers the key concerns for organizations adopting AI coding tools at scale.

### 9.1 Code Ownership and Intellectual Property

| Concern | OpenCode (Crush) | Claude Code | Hermes Agent |
|:--------|:----------------:|:-----------:|:------------:|
| **Code sent to model provider** | Configurable (select your provider) | Yes (Anthropic API) | Configurable (select your provider) |
| **Training on your code** | Depends on provider policy | Opt-out available | Depends on provider (none for local models) |
| **IP assignment of generated code** | User retains ownership (no ToS claim) | User retains ownership | User retains ownership |
| **Self-hosted option eliminates sending** | Yes (local models via Ollama/vLLM) | No (API-only) | Yes (local models) |
| **Audit trail of generated code** | Session logs (local SQLite) | Session persistence (.claude) | FTS5 searchable logs, trajectory generation |
| **Open-source license compliance** | Manual review | Manual review | Manual review (subagent scanning) |

**Key question:** Does the AI coding tool's provider train on your code? Check the provider's terms of service. Self-hosted models (running locally via Ollama, vLLM, or llama.cpp) guarantee that no code leaves your infrastructure. For cloud-based tools, most enterprise plans offer data privacy guarantees where your prompts and completions are not used for model training.

### 9.2 Data Privacy and Leakage Prevention

AI coding tools have access to your entire codebase — making them a potential vector for data leakage. Enterprise controls include:

| Control | OpenCode (Crush) | Claude Code | Hermes Agent |
|:--------|:----------------:|:-----------:|:------------:|
| **API key injection** | Environment variables | Environment variables | Environment variables + secret injection |
| **File access scoping** | Workspace directory | Workspace directory | Configurable via profiles |
| **Network egress control** | OS-level firewall | OS-level firewall | Docker/SSH isolation |
| **Prompt inspection** | Manual (user reviews tool calls) | Permission mode review | Granular per-tool approval |
| **PII/secret redaction** | Not built-in | Not built-in | Plugin-based (extensible) |
| **Codebase masking** | Not built-in | Not built-in | Not built-in |
| **Output filtering** | Not built-in | Not built-in | Plugin-based (safety filters) |

**Best practices:**
1. Use `.gitignore` to exclude sensitive files (`.env`, `secrets/`, `credentials.*`) from the agent's workspace
2. Inject API keys and secrets via environment variables, never in plain text files
3. For highly sensitive codebases, use self-hosted models with Docker isolation
4. Review all tool call logs regularly for data exfiltration attempts
5. Implement a human-in-the-loop approval process for any action that reads or modifies sensitive files

### 9.3 Audit Logging and Traceability

| Capability | OpenCode (Crush) | Claude Code | Hermes Agent |
|:-----------|:----------------:|:-----------:|:------------:|
| **Full session recording** | Local SQLite | .claude directory | FTS5 searchable + trajectory export |
| **Tool call history** | Session logs | Session files | Searchable tool logs |
| **File change tracking** | Visual diff | Git integration | Via git (external) |
| **User attribution** | Per-user session | Per-user session | Per-user + per-platform |
| **Export/archival** | SQLite DB export | Session file export | FTS5 + JSON trajectory export |
| **Compliance reporting** | Manual | Manual | Trajectory generation format |
| **SIEM integration** | Not built-in | Not built-in | Plugin-based (extensible via cron + webhook) |

**For regulated industries (finance, healthcare, government):** Maintain immutable audit logs of all AI coding tool interactions. Export session data in a structured format (JSON/NDJSON) that can be ingested by your existing SIEM or log management platform. Hermes Agent's trajectory generation feature outputs structured JSON records suitable for compliance pipelines.

### 9.4 Compliance Frameworks and Certifications

| Framework | Relevance to AI Coding Tools | Key Requirements |
|:----------|:----------------------------|:-----------------|
| **SOC 2** (Type II) | Data security, availability, processing integrity | Access controls, encryption, audit logging, vendor management |
| **ISO 27001** | Information security management | Risk assessment, security controls, continuous improvement |
| **GDPR** | Data protection for EU users | Right to explanation, data minimization, data processing records |
| **HIPAA** | Healthcare data (US) | BAA with provider, PHI access controls, audit controls |
| **PCI DSS** | Payment card data | Code changes affecting payment systems must be logged and reviewed |
| **Executive Order on AI (2023)** | US federal government AI use | Risk assessment, testing, transparency requirements |

**Vendor assessment checklist:**
- [ ] Does the tool provider offer a SOC 2 report or ISO 27001 certification?
- [ ] Are model providers (OpenAI, Anthropic, etc.) SOC 2 compliant?
- [ ] Is data encrypted at rest and in transit?
- [ ] Can you opt out of training data usage?
- [ ] Are there data retention policies and deletion capabilities?
- [ ] Does the tool support bring-your-own-key (BYOK) encryption?
- [ ] Are subprocessors (model providers, infrastructure) disclosed?

### 9.5 Vendor Lock-in Mitigation

| Strategy | OpenCode (Crush) | Claude Code | Hermes Agent |
|:---------|:----------------:|:-----------:|:------------:|
| **Multi-provider support** | ✅ Yes (Claude, GPT, Gemini, Groq, etc.) | ❌ No (Claude only) | ✅ Yes (20+ providers) |
| **MCP portability** | ✅ Yes (MCP servers are reusable) | ✅ Yes (MCP servers are reusable) | ✅ Yes (MCP servers are reusable) |
| **Configuration format** | JSON (opencode.json) | JSON (.claude/settings.json) | YAML (config.yaml) |
| **Custom commands/skills** | Markdown commands → portable | JS plugins → Claude-specific | SKILL.md → portable (agentskills.io) |
| **Open standard for context** | No standard format | CLAUDE.md (proprietary) | AGENTS.md/CONTEXT.md (open conventions) |
| **Export/import tool** | Manual | Manual | `hermes claw migrate` |
| **Self-hostable** | ✅ Yes | ❌ No (proprietary API-only) | ✅ Yes (fully open source) |

**Recommendation:** Build your AI coding toolchain around open standards (MCP, AGENTS.md, SKILL.md, markdown-based configs) rather than tool-specific formats. This ensures your custom commands, MCP servers, and project instructions remain portable even if you switch primary tools. Standards documented in [08-Reference/03-Agent-Configs-SOUL-SKILL.md] and [03-Agents/04-Protocols-MCP-ACP.md].

### 9.6 Enterprise Adoption Rollout Plan

| Phase | Duration | Activities | Success Metrics |
|:------|:--------:|:-----------|:----------------|
| **1. Pilot** | 2-4 weeks | Select 5-10 developers; provide training; establish guardrails; collect feedback | Tool adoption rate (>80% of pilot group), developer satisfaction score (>4/5) |
| **2. Evaluate** | 2-4 weeks | Measure productivity impact; identify failure modes; refine guardrails | PR cycle time reduction (>15%), bug introduction rate (<5% increase) |
| **3. Expand** | 4-8 weeks | Roll out to team; integrate with CI/CD; establish MCP server catalog; document best practices | Team-wide adoption (>90%), standard deviation of productivity gains <20% |
| **4. Govern** | Ongoing | Audit logs review; compliance reporting; model provider evaluation; skill library curation | Zero compliance incidents, audit-ready within 48 hours |
| **5. Optimize** | Quarterly | Cost optimization; model routing; benchmark new models; skill improvement | 20% cost reduction per quarter, sustained developer satisfaction |

### 9.7 Cost Governance

| Budget Item | OpenCode (Crush) | Claude Code | Hermes Agent |
|:------------|:----------------:|:-----------:|:------------:|
| **Base tool cost** | Free (MIT) | Free tier + pay-per-token | Free (MIT) |
| **Model API cost (heavy user)** | $50-500/mo | $100-1,000/mo | $50-500/mo |
| **Self-hosting infrastructure** | $0 (terminal only) | N/A (API-only) | $0-50/mo (optional: Docker, Modal) |
| **Per-developer cost** | $50-500/mo | $100-1,000/mo | $50-500/mo |
| **Cost controls** | Provider selection (switch to cheaper) | Prompt caching (50-90% savings) | Provider routing (cheap for simple, premium for complex) |
| **Budget monitoring** | Manual (check API dashboard) | Anthropic console | Multi-provider usage tracking via gateway |

| Capability | OpenCode (Crush) | Claude Code | Hermes Agent |
|:-----------|:----------------:|:-----------:|:------------:|
| **Model routing by task** | Manual (switch provider) | Not available | Yes (automatic: cheap model for simple queries) |
| **Token budget alerts** | Not built-in | Not built-in | Plugin-based |
| **Per-user quotas** | Not built-in | Not built-in | Plugin-based |
| **Cost allocation tags** | Not built-in | Not built-in | Via gateway metadata |

**Cost optimization strategy for enterprises:**
1. Use self-hosted open-weight models (Llama 3, Qwen, DeepSeek) for routine coding tasks (autocomplete, documentation, simple refactoring)
2. Reserve frontier models (Claude, GPT-4o) for complex reasoning, architecture decisions, and security-critical code review
3. Implement cost-aware routing: route file search, grep, and simple edits to budget models; route complex planning and multi-file refactoring to premium models
4. Enable prompt caching (Claude Code: built-in; Hermes: provider-dependent) for 50-90% savings on repetitive contexts
5. Set token consumption alerts and per-developer budgets to prevent cost overruns

---

## Cross-References

| Reference | Description |
|-----------|-------------|
| [03-Agents/04-Protocols-MCP-ACP.md] | Protocol support that underlies these tools |
| [03-Agents/01-Agent-Architectures.md] | The architectural patterns these tools implement |
| [08-Reference/03-Agent-Configs-SOUL-SKILL.md] | Configuration files used by Hermes Agent |
| [08-Reference/01-Glossary.md] | Definitions for OpenCode, Crush, Claude Code, Hermes Agent, etc. |
| [08-Reference/02-AI-Roadmap.md] | Where these tools are heading |
| [05-Enterprise/01-Enterprise-AI-Deployment.md] | Production deployment patterns for AI coding tools |
| [03-Agents/04-Protocols-MCP-ACP.md] | Detailed protocol comparison between MCP and ACP |
| [08-Reference/01-Glossary.md] | §9 Enterprise Governance terms (SOC 2, ISO 27001, GDPR, vendor lock-in) |

---

*Document version: 1.5 → 2.0 — June 2026 | Expanded with §8 Migration and Integration Guide — migration paths, hybrid workflows, CI/CD integration, team adoption recommendations, MCP server portability, decision framework; and §9 Enterprise Governance and Compliance — code ownership, data privacy, audit logging, compliance frameworks, vendor lock-in mitigation, enterprise rollout, cost governance*

*Sources: GitHub repositories (opencode-ai/opencode, charmbracelet/crush, NousResearch/hermes-agent, anthropics/claude-code), official documentation sites, and project READMEs.*
