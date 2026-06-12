# 02 — SOUL.md & SKILL.md Templates

> **Purpose:** Complete reference for authoring SOUL.md and SKILL.md files — the foundational metadata and capability descriptors for AI agents.
> **Last Updated:** 2026-06-12
> **Maintainer:** AI Knowledge Base Team
> **Status:** Active

---

## Table of Contents

1. [Introduction](#introduction)
2. [SOUL.md — Agent Identity File](#soulmd---agent-identity-file)
   - [Frontmatter Fields](#frontmatter-fields)
   - [Complete Example: CodeGen Agent](#complete-example-codegen-agent)
   - [Field Reference Table](#field-reference-table)
   - [Validation Rules](#validation-rules)
3. [SKILL.md — Capability Descriptor File](#skillmd---capability-descriptor-file)
   - [Frontmatter Fields](#skillmd-frontmatter-fields)
   - [Complete Example: Code Review Skill](#complete-example-code-review-skill)
   - [Command & Subcommand Structure](#command--subcommand-structure)
   - [Parameter Specification](#parameter-specification)
   - [Template Variables](#template-variables)
4. [Best Practices](#best-practices)
   - [Writing Effective SOUL Files](#writing-effective-soul-files)
   - [Writing Effective SKILL Files](#writing-effective-skill-files)
   - [Common Pitfalls](#common-pitfalls)
5. [Extension Points](#extension-points)
   - [Custom Hooks](#custom-hooks)
   - [Plugin Integration](#plugin-integration)
   - [Multi-Agent Coordination](#multi-agent-coordination)
6. [Validation & Testing](#validation--testing)
   - [Schema Validation](#schema-validation)
   - [Integration Testing](#integration-testing)
   - [Linting Rules](#linting-rules)
7. [Migration Guide](#migration-guide)
   - [v1.x to v2.0 Migration](#v1x-to-v20-migration)
   - [Breaking Changes](#breaking-changes)
8. [Troubleshooting](#troubleshooting)
9. [FAQ](#faq)
10. [Further Reading](#further-reading)

---

## Introduction

SOUL.md and SKILL.md files are the **identity and capability descriptors** for AI agents in the Hermes Agent ecosystem (and compatible frameworks). They serve as:

- **SOUL.md** — The agent's "identity card": name, description, model configuration, tools, instructions, and metadata.
- **SKILL.md** — The agent's "capability declaration": a single skill with commands, subcommands, parameters, and execution templates.

Together, they enable:
- **Discoverability** — Other agents and users can find capabilities
- **Composability** — Skills can be loaded and chained
- **Versioning** — Clear version tracking for capabilities
- **Validation** — Schema-checked configurations
- **Portability** — Skills can be shared across agents

### When to Use Each

| File | Use When |
|------|----------|
| **SOUL.md** | Defining an agent's core identity, model, tools, and instructions |
| **SKILL.md** | Declaring a single reusable capability with structured commands |

A single agent has **one** SOUL.md and **zero or more** SKILL.md files.

---

## SOUL.md — Agent Identity File

SOUL.md defines **who the agent is** — its identity, model binding, tool access, and behavioral instructions.

### Frontmatter Fields

The SOUL.md file starts with YAML frontmatter between `---` delimiters. All fields are listed below with their types, requirements, and descriptions.

#### Required Fields

```yaml
---
name: "Agent-Name"
description: >
  A short description of the agent's purpose and capabilities.
  Can span multiple lines with YAML block scalar (> or |).
version: "1.0.0"
model:
  provider: openai
  name: gpt-4o
---
```

#### Optional Fields

```yaml
---
tools:
  - name: web_search
    description: Search the web for current information
  - name: code_interpreter
    description: Execute Python code in a sandboxed environment
instructions: |
  You are a helpful coding assistant. Always explain your reasoning.
  Prefer simple solutions over complex ones.
avatar: "🤖"  # Emoji or URL to avatar image
language: en-US
tags:
  - code-generation
  - debugging
  - refactoring
homepage: https://github.com/example/agent
repository: https://github.com/example/agent.git
license: MIT
skills:
  - code-review
  - test-generation
  - documentation
requires:
  python: ">=3.10"
  memory: "512MB"
environment:
  variables:
    - name: OPENAI_API_KEY
      description: OpenAI API key for model access
      required: true
    - name: LOG_LEVEL
      description: Logging level
      default: "info"
extensions:
  hooks:
    on_start: plugins/init.py
    on_shutdown: plugins/cleanup.py
---
```

### Complete Example: CodeGen Agent

Below is a complete, production-quality SOUL.md for a code-generation agent.

```yaml
---
name: "CodeForge"
description: >
  An advanced code generation and analysis agent. Capable of writing,
  reviewing, refactoring, and optimizing code across 20+ programming
  languages. Integrates with Git, linters, and test frameworks.
version: "2.3.1"
model:
  provider: openai
  name: gpt-4o
  temperature: 0.2
  max_tokens: 16384
  top_p: 0.95
  frequency_penalty: 0.0
  presence_penalty: 0.0
  stop:
    - "<|endoftext|>"
tools:
  - name: terminal
    description: Execute shell commands in a Linux environment
  - name: read_file
    description: Read text files with line numbers and pagination
  - name: write_file
    description: Write or overwrite file content
  - name: search_files
    description: Search file contents or find files by name
  - name: patch
    description: Targeted find-and-replace edits using fuzzy matching
  - name: process
    description: Manage background processes
  - name: web_search
    description: Search the web for current information
    requires_confirmation: true
  - name: github_api
    description: Interact with GitHub API for PRs, issues, and repos
    requires_auth: true
avatar: "⚡"
language: en-US
tags:
  - code-generation
  - code-review
  - refactoring
  - debugging
  - testing
  - git
homepage: https://codeforge.dev
repository: https://github.com/nousresearch/codeforge
license: Apache-2.0
skills:
  - code-review
  - test-generation
  - documentation-writer
  - refactoring-assistant
  - dependency-audit
instructions: |
  # CodeForge Agent Instructions

  ## Core Identity
  You are CodeForge, an expert software engineering assistant. Your purpose is to help developers write better code faster.

  ## Behavioral Guidelines
  1. **Understand before acting.** Always read the relevant files and understand the codebase structure before making changes.
  2. **Explain your reasoning.** For every change, explain why it's being made.
  3. **Prefer simple solutions.** The simplest correct solution is usually the best.
  4. **Test your changes.** Always run tests after making changes.
  5. **Respect existing patterns.** Follow the codebase's established patterns and conventions.
  6. **Be safe.** Never delete data without confirmation. Use version control.

  ## Communication Style
  - Be concise but thorough
  - Use code blocks with language tags
  - Explain errors before fixing them
  - Offer alternatives when there are tradeoffs

  ## Workflow
  1. Understand the request
  2. Explore the codebase
  3. Plan the approach
  4. Implement changes
  5. Verify with tests
  6. Summarize what was done

  ## Constraints
  - Maximum 5 file modifications per turn without confirmation
  - Always suggest testing after changes
  - Never expose API keys or secrets
requires:
  python: ">=3.11"
  node: ">=18.0"
  memory: "1GB"
  disk: "5GB"
environment:
  variables:
    - name: OPENAI_API_KEY
      description: Required for model access
      required: true
      sensitive: true
    - name: GITHUB_TOKEN
      description: GitHub personal access token
      required: false
      sensitive: true
    - name: LOG_LEVEL
      description: Logging level (debug, info, warning, error)
      default: "info"
    - name: CODE_FORGE_MODE
      description: "Operating mode: strict, moderate, or relaxed"
      default: "moderate"
extensions:
  hooks:
    on_start: plugins/codeforge_init.py
    on_shutdown: plugins/codeforge_cleanup.py
  plugins:
    - name: git-integration
      version: "1.2.0"
      description: Automatic commit and branch management
    - name: lint-runner
      version: "2.0.1"
      description: Runs linters after code changes
---
```

### Field Reference Table

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | string | ✅ | — | Agent's display name (2–64 chars, alphanumeric + hyphens) |
| `description` | string | ✅ | — | Short summary (10–500 chars) |
| `version` | string | ✅ | — | Semantic version (semver) |
| `model.provider` | string | ✅ | — | Model provider (openai, anthropic, google, local) |
| `model.name` | string | ✅ | — | Model identifier string |
| `model.temperature` | float | ❌ | 0.7 | Sampling temperature (0.0–2.0) |
| `model.max_tokens` | int | ❌ | 4096 | Maximum output tokens |
| `model.top_p` | float | ❌ | 1.0 | Nucleus sampling parameter |
| `model.frequency_penalty` | float | ❌ | 0.0 | Frequency penalty (-2.0 to 2.0) |
| `model.presence_penalty` | float | ❌ | 0.0 | Presence penalty (-2.0 to 2.0) |
| `model.stop` | string[] | ❌ | [] | Stop sequences |
| `tools` | Tool[] | ❌ | [] | Available tools/function calls |
| `avatar` | string | ❌ | 🤖 | Emoji or URL for agent avatar |
| `language` | string | ❌ | en-US | Primary language code |
| `tags` | string[] | ❌ | [] | Classification tags |
| `homepage` | string | ❌ | — | Project homepage URL |
| `repository` | string | ❌ | — | Source repository URL |
| `license` | string | ❌ | — | SPDX license identifier |
| `skills` | string[] | ❌ | [] | Referenced skill names |
| `requires` | object | ❌ | {} | System requirements |
| `environment.variables` | EnvVar[] | ❌ | [] | Required/env configuration |
| `extensions.hooks` | object | ❌ | {} | Lifecycle hooks |
| `extensions.plugins` | Plugin[] | ❌ | [] | Plugin dependencies |
| `instructions` | string | ❌ | — | System prompt / behavioral instructions |

### Validation Rules

When parsing a SOUL.md, the system validates these rules:

```yaml
validation_rules:
  name:
    - pattern: "^[a-zA-Z0-9][a-zA-Z0-9_-]{1,63}$"
    - message: "Name must start with alphanumeric, 2–64 chars, no spaces"
  version:
    - pattern: "^\\d+\\.\\d+\\.\\d+$"
    - pattern: "^\\d+\\.\\d+\\.\\d+-\\w+\\.\\d+$"  # pre-release
    - message: "Must be valid semver (e.g., 1.0.0, 2.3.1-beta.1)"
  model:
    provider:
      - enum: [openai, anthropic, google, local, azure, bedrock, together, fireworks]
    name:
      - pattern: "^[a-zA-Z0-9][a-zA-Z0-9._-]+$"
  tools[].name:
    - unique: true
    - pattern: "^[a-z][a-z_]{1,63}$"
  tags:
    - unique: true
    - max_items: 20
  instructions:
    - min_length: 50
    - max_length: 10000
  environment.variables[].name:
    - pattern: "^[A-Z][A-Z0-9_]{1,63}$"
    - message: "Environment variables must be UPPER_SNAKE_CASE"
  skills:
    - unique: true
    - max_items: 50
```

---

## SKILL.md — Capability Descriptor File

SKILL.md defines a **single reusable capability**. It specifies commands, subcommands, parameters, and execution templates that an agent can perform.

### SKILL.md Frontmatter Fields

```yaml
---
name: "code-review"
description: "Perform comprehensive code review with static analysis, security scanning, and best-practice checks."
version: "2.1.0"
author: "CodeForge Team"
category: "development"
tags:
  - code-review
  - static-analysis
  - security
  - linting
requires:
  python: ">=3.10"
  tools:
    - pylint
    - bandit
    - mypy
    - black
icon: "🔍"
timeout: 300  # seconds
---
```

#### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Unique skill name (alphanumeric + hyphens) |
| `description` | string | What the skill does (10–500 chars) |
| `version` | string | Semantic version of the skill |

#### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `author` | string | — | Creator or team name |
| `category` | string | "general" | Functional category |
| `tags` | string[] | [] | Classification tags |
| `requires` | object | {} | Dependencies (tools, runtimes) |
| `icon` | string | "📦" | Emoji icon |
| `timeout` | int | 120 | Max execution time in seconds |

### Command & Subcommand Structure

Each SKILL.md contains a `commands` section defining the skill's interface:

```yaml
commands:
  - name: review
    description: "Run a full code review on specified files or directories"
    subcommands:
      - name: files
        description: "Review specific files"
        parameters:
          - name: paths
            type: string[]
            description: "File paths to review (glob patterns supported)"
            required: true
          - name: severity
            type: enum
            description: "Minimum severity level to report"
            default: "warning"
            options:
              - "critical"
              - "warning"
              - "info"
      - name: directory
        description: "Review an entire directory"
        parameters:
          - name: path
            type: string
            description: "Directory path to review"
            required: true
          - name: recursive
            type: boolean
            description: "Include subdirectories"
            default: true
          - name: exclude
            type: string[]
            description: "Patterns to exclude"
            default: ["node_modules", ".git", "venv"]
      - name: pr
        description: "Review changes in a pull request"
        parameters:
          - name: pr_number
            type: integer
            description: "Pull request number"
            required: true
          - name: repository
            type: string
            description: "Owner/repo (defaults to current repo)"
            required: false
          - name: focus
            type: enum
            description: "Review focus area"
            default: "all"
            options:
              - "all"
              - "security"
              - "performance"
              - "style"
              - "correctness"
```

#### Type System

| Type | Description | Example |
|------|-------------|---------|
| `string` | Text value | `"hello"` |
| `integer` | Whole number | `42` |
| `float` | Decimal number | `3.14` |
| `boolean` | True/false | `true` |
| `enum` | One of defined options | `"warning"` |
| `string[]` | Array of strings | `["a", "b"]` |
| `integer[]` | Array of integers | `[1, 2, 3]` |
| `object` | Key-value map | `{key: "value"}` |
| `file` | File path | `"src/main.py"` |
| `directory` | Directory path | `"src/"` |
| `code` | Code snippet | `"def foo(): pass"` |

### Parameter Specification

Parameters follow a detailed schema:

```yaml
parameters:
  - name: example_param
    type: string
    description: "What this parameter controls"
    required: true                          # or false
    default: "default_value"                # used when not required
    options:                                # only for enum types
      - "option1"
      - "option2"
    validation:
      pattern: "^[a-z]+$"                   # regex pattern
      min_length: 1
      max_length: 100
      min_value: 0                          # for numeric types
      max_value: 100
    sensitive: false                        # if true, mask in logs
    positional: false                       # if true, use positional arg
    deprecated: false                       # if true, warn on use
```

### Template Variables

Templates use Go-style template syntax with `{{ }}` delimiters:

```yaml
templates:
  system: |
    You are a code review assistant with expertise in {{ .language }}.
    Focus on: {{ .focus_area }}.
    Minimum severity: {{ .severity }}.

  user: |
    Please review the following code:

    File: {{ .filename }}
    ```{{ .language }}
    {{ .code }}
    ```

    Consider:
    1. Correctness — Does the code do what it intends?
    2. Security — Are there vulnerabilities?
    3. Performance — Can this be optimized?
    4. Style — Does it follow {{ .style_guide }}?
    5. Maintainability — Is it easy to understand and change?

  response: |
    ## Code Review: {{ .filename }}

    ### Summary
    {{ .summary }}

    ### Issues Found
    {{ range .issues }}
    - **[{{ .severity }}]** {{ .line }}: {{ .message }}
      *Suggestion:* {{ .suggestion }}
    {{ end }}

    ### Overall Assessment
    {{ .assessment }}

### Complete Example: Code Review Skill

Below is a complete, production-quality SKILL.md for a code review capability.

```yaml
---
name: "code-review"
description: >
  Comprehensive code review skill covering static analysis, security
  auditing, style enforcement, performance profiling, and best-practice
  recommendations. Supports Python, JavaScript, TypeScript, Go, Rust,
  Java, and C++.
version: "2.1.0"
author: "CodeForge Team"
category: "development"
tags:
  - code-review
  - static-analysis
  - security
  - linting
  - quality
requires:
  python: ">=3.10"
  node: ">=18.0"
  tools:
    - pylint
    - bandit
    - mypy
    - black
    - eslint
    - prettier
icon: "🔍"
timeout: 300
---

## Overview

The code-review skill provides automated, AI-enhanced code review capabilities.
It combines static analysis tools with LLM-powered understanding to deliver
comprehensive feedback on code quality, security, performance, and maintainability.

## Commands

commands:
  - name: review
    description: "Run a full code review on specified files or directories"
    subcommands:
      - name: files
        description: "Review specific files"
        parameters:
          - name: paths
            type: string[]
            description: "File paths to review (supports glob patterns like **/*.py)"
            required: true
          - name: language
            type: enum
            description: "Programming language"
            options:
              - python
              - javascript
              - typescript
              - go
              - rust
              - java
              - cpp
            required: true
          - name: severity
            type: enum
            description: "Minimum severity level to report"
            default: "info"
            options:
              - "critical"
              - "warning"
              - "info"
          - name: focus
            type: enum
            description: "Review focus area"
            default: "all"
            options:
              - "all"
              - "security"
              - "performance"
              - "style"
              - "correctness"
          - name: style_guide
            type: enum
            description: "Style guide to follow"
            default: "default"
            options:
              - "default"
              - "google"
              - "microsoft"
              - "airbnb"
              - "custom"
      - name: directory
        description: "Review an entire directory"
        parameters:
          - name: path
            type: directory
            description: "Directory path to review"
            required: true
          - name: recursive
            type: boolean
            description: "Include subdirectories"
            default: true
          - name: exclude
            type: string[]
            description: "Patterns to exclude"
            default: ["node_modules", ".git", "venv", "__pycache__", "dist", "build"]
          - name: max_files
            type: integer
            description: "Maximum number of files to review"
            default: 50
            validation:
              min_value: 1
              max_value: 500
      - name: pr
        description: "Review changes in a pull request"
        parameters:
          - name: pr_number
            type: integer
            description: "Pull request number"
            required: true
          - name: repo
            type: string
            description: "Repository in owner/repo format"
            required: true
          - name: focus
            type: enum
            description: "Review focus area"
            default: "all"
            options:
              - "all"
              - "security"
              - "performance"
              - "style"
              - "correctness"
          - name: inline_comments
            type: boolean
            description: "Post inline comments on the PR"
            default: false

  - name: analyze
    description: "Run specific analysis types without full review"
    subcommands:
      - name: security
        description: "Security vulnerability scan"
        parameters:
          - name: paths
            type: string[]
            description: "Files or directories to scan"
            required: true
          - name: standard
            type: enum
            description: "Security standard"
            default: "owasp"
            options:
              - "owasp"
              - "cwe"
              - "sans"
      - name: complexity
        description: "Code complexity analysis"
        parameters:
          - name: paths
            type: string[]
            description: "Files to analyze"
            required: true
          - name: metric
            type: enum
            description: "Complexity metric"
            default: "cyclomatic"
            options:
              - "cyclomatic"
              - "cognitive"
              - "halstead"
              - "maintainability"
      - name: dependencies
        description: "Dependency vulnerability check"
        parameters:
          - name: manifest_path
            type: string
            description: "Path to dependency manifest (package.json, requirements.txt, etc.)"
            required: true
          - name: deep_scan
            type: boolean
            description: "Scan transitive dependencies"
            default: true

## Templates

templates:
  system: |
    You are a professional code reviewer with deep expertise in {{ .language }}.
    Your analysis must be thorough, constructive, and actionable.
    Focus areas: {{ .focus }}.
    Minimum severity: {{ .severity }}.

    Follow these principles:
    1. Be specific — reference exact lines and symbols
    2. Be constructive — always suggest improvements, not just problems
    3. Be prioritized — highlight critical issues first
    4. Be educational — explain the "why" behind each finding

  review_prompt: |
    Please review the following {{ .language }} code:

    File: {{ .filename }}
    ```{{ .language }}
    {{ .code }}
    ```

    ## Review Criteria
    {{ if eq .focus "security" }}
    ### Security Checklist
    - [ ] Input validation
    - [ ] Authentication/authorization
    - [ ] Data sanitization
    - [ ] Secure defaults
    - [ ] Error handling (no information leakage)
    - [ ] Dependency vulnerabilities
    - [ ] Secrets management
    {{ else if eq .focus "performance" }}
    ### Performance Checklist
    - [ ] Algorithmic complexity
    - [ ] Memory usage
    - [ ] I/O operations (batching, caching)
    - [ ] Database query optimization
    - [ ] Concurrency and parallelism
    - [ ] Hot paths and bottlenecks
    {{ else if eq .focus "style" }}
    ### Style Checklist
    - [ ] Follows {{ .style_guide }} conventions
    - [ ] Naming consistency
    - [ ] Comment quality and necessity
    - [ ] Code organization
    - [ ] Formatting (linting rules)
    {{ else }}
    ### Full Review Checklist
    - [ ] Correctness — logical accuracy
    - [ ] Security — vulnerability assessment
    - [ ] Performance — efficiency analysis
    - [ ] Style — code conventions
    - [ ] Maintainability — clarity and structure
    - [ ] Test coverage — existing and needed
    {{ end }}

    ## Response Format
    Provide your review as follows:

    ### Summary (2-3 sentences)
    ### Critical Issues (must fix)
    Each with: Location, Issue, Risk, Suggestion
    ### Warnings (should fix)
    Each with: Location, Issue, Suggestion
    ### Suggestions (nice to have)
    Each with: Location, Suggestion
    ### Overall Rating
    Pass / Pass with changes / Needs significant revision

  summary_template: |
    ## Review Summary

    **File:** {{ .filename }}
    **Language:** {{ .language }}
    **Focus:** {{ .focus }}
    **Severity Threshold:** {{ .severity }}

    | Category | Count |
    |----------|-------|
    | 🔴 Critical | {{ .critical_count }} |
    | 🟡 Warning | {{ .warning_count }} |
    | 🔵 Info | {{ .info_count }} |

    **Overall:** {{ .overall_assessment }}

## Examples

### Example 1: Review a Python file for security issues

```yaml
command: review
subcommand: files
parameters:
  paths: ["src/auth/login.py"]
  language: python
  severity: warning
  focus: security
```

### Example 2: Analyze code complexity across a project

```yaml
command: analyze
subcommand: complexity
parameters:
  paths: ["src/"]
  metric: cyclomatic
```

### Example 3: Full PR review with inline comments

```yaml
command: review
subcommand: pr
parameters:
  pr_number: 142
  repo: "myorg/myrepo"
  focus: all
  inline_comments: true
```

## Error Handling

| Error | Cause | Resolution |
|-------|-------|------------|
| `FILE_NOT_FOUND` | Specified path doesn't exist | Verify path and try again |
| `UNSUPPORTED_LANGUAGE` | Language not in supported list | Check supported languages |
| `TOOL_NOT_INSTALLED` | Required linter/analyzer missing | Install via `pip install` or `npm install` |
| `REVIEW_TIMEOUT` | Review exceeded timeout | Reduce file count or scope |
| `GIT_ERROR` | Git operation failed | Check repository permissions |
| `PARSE_ERROR` | Could not parse the file | Verify file syntax is valid |

## Changelog

### 2.1.0 (2026-05-15)
- Added dependency analysis subcommand
- Added inline comment support for PR reviews
- Extended language support to include Rust and C++

### 2.0.0 (2026-02-01)
- Major rewrite with subcommand architecture
- Added template system for customizable output
- Added complexity analysis capabilities
- Improved error handling and timeout management

### 1.0.0 (2025-11-01)
- Initial release with basic review capabilities
- Support for Python and JavaScript
- Security and style analysis

## See Also

- [SOUL.md Template](../02-SOUL-SKILL-Templates.md#soulmd---agent-identity-file)
- [Prompt Libraries](../03-Prompt-Libraries.md) — Additional prompt templates
- [Agent Toolkits](../04-Agent-Toolkits.md) — Framework integration patterns
```

---

## Best Practices

### Writing Effective SOUL Files

1. **Be specific in the name.** Use descriptive names like "CodeForge" not "Agent42".
2. **Write a clear description.** Someone should understand the agent's purpose in one sentence.
3. **List only necessary tools.** Don't add tools the agent won't use — it increases latency and risk.
4. **Set appropriate model parameters.** Lower temperature (0.1–0.3) for deterministic tasks like code; higher (0.7–0.9) for creative tasks.
5. **Write thorough instructions.** The instructions field is the most important for behavior shaping.
6. **Use environment variables for config.** Never hardcode secrets or paths.
7. **Version your SOUL.md.** Use semantic versioning and update when behavior changes.
8. **Tag thoughtfully.** Tags enable discovery — use standard categories.
9. **Specify requirements.** List runtime requirements so users know what's needed.
10. **Keep instructions concise.** Aim for 200–1000 words. Too long degrades model attention.

### Writing Effective SKILL Files

1. **Single responsibility.** Each SKILL should do one thing well.
2. **Name commands as verbs.** `review`, `analyze`, `generate`, `transform`.
3. **Use subcommands for variants.** `review.files`, `review.directory`, `review.pr`.
4. **Set good defaults.** Make the common case simple.
5. **Validate inputs.** Use validation patterns to catch bad input early.
6. **Document every parameter.** Users and LLMs need to understand each field.
7. **Provide examples.** At least 2–3 usage examples.
8. **Handle errors gracefully.** Define error codes and recovery strategies.
9. **Version your SKILLs.** Track changes in a changelog section.
10. **Template thoughtfully.** System prompts shape LLM behavior; user prompts structure input.

### Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| **Overly broad description** | Agent takes on tasks it can't handle | Narrow the scope, split into multiple agents |
| **Missing instructions** | Agent behaves unpredictably | Always set behavioral instructions |
| **Too many tools** | Confusion, increased latency | Only include tools the agent actually uses |
| **Vague parameter descriptions** | LLM passes wrong values | Be specific: "File path relative to project root" |
| **No error handling** | Silent failures | Define error types and recovery paths |
| **Ignoring timeouts** | Long-running tasks hang | Set realistic timeouts per command |
| **Hardcoded values** | Brittle, non-portable | Use template variables and environment vars |
| **No versioning** | Can't track changes | Use semver and maintain a changelog |

---

## Extension Points

### Custom Hooks

Hooks allow running code at specific lifecycle points:

```yaml
extensions:
  hooks:
    on_start: hooks/startup.py       # Run when agent initializes
    on_shutdown: hooks/shutdown.py   # Run when agent terminates
    on_tool_call: hooks/audit.py     # Run before every tool call
    on_error: hooks/error_handler.py # Run when an error occurs
```

Hook scripts receive a context object with agent state and can:
- Initialize resources (DB connections, API clients)
- Log events for auditing
- Send telemetry
- Modify agent behavior dynamically
- Clean up resources on shutdown

### Plugin Integration

Skills can declare plugin dependencies:

```yaml
extensions:
  plugins:
    - name: git-integration
      version: ">=1.0.0"
      description: Git operations for the agent
    - name: cache-manager
      version: "2.0.0"
      description: In-memory caching layer
```

Plugins are loaded in order and can:
- Add new tools to the agent
- Provide shared utilities
- Register middleware
- Expose configuration interfaces

### Multi-Agent Coordination

For complex workflows, multiple agents can coordinate:

```
Coordinator Agent (SOUL: orchestrator)
├── Code Agent (SOUL: code-forge)
│   ├── SKILL: code-review
│   └── SKILL: test-generation
├── Research Agent (SOUL: researcher)
│   └── SKILL: web-search
└── Review Agent (SOUL: reviewer)
    └── SKILL: quality-gate
```

Coordination patterns:
- **Sequential**: Pass output from one agent as input to the next
- **Hierarchical**: Orchestrator delegates to specialized agents
- **Debate**: Multiple agents review each other's work
- **Voting**: Multiple agents propose solutions, best one selected

---

## Validation & Testing

### Schema Validation

Validate SOUL.md and SKILL.md against the JSON Schema:

```bash
# Validate a SOUL.md file
hermes validate soul path/to/SOUL.md

# Validate a SKILL.md file
hermes validate skill path/to/SKILL.md

# Validate all files in a directory
hermes validate all ./skills/

# Strict validation (fails on warnings)
hermes validate soul path/to/SOUL.md --strict
```

### Integration Testing

Test that skills work end-to-end:

```bash
# Test a specific skill command
hermes test skill code-review --command review --params '{"paths": ["test/fixtures/sample.py"], "language": "python"}'

# Run all skill tests
hermes test all

# Run with coverage report
hermes test all --coverage
```

### Linting Rules

```
# SOUL.md rules
SOUL001 - Name must match pattern ^[a-zA-Z0-9][a-zA-Z0-9_-]{1,63}$
SOUL002 - Description must be 10-500 characters
SOUL003 - Version must be valid semver
SOUL004 - Model provider must be from supported list
SOUL005 - Tool names must be lowercase with underscores
SOUL006 - Instructions must be between 50-10000 characters
SOUL007 - Environment variable names must be UPPER_SNAKE_CASE

# SKILL.md rules
SKILL001 - Name must match pattern ^[a-z][a-z-]{1,63}$
SKILL002 - Description must be 10-500 characters
SKILL003 - Version must be valid semver
SKILL004 - At least one command must be defined
SKILL005 - Command names must be lowercase verbs
SKILL006 - All parameters must have descriptions
SKILL007 - Required parameters must come before optional
SKILL008 - Template variables must match configuration
SKILL009 - Timeout must be between 1 and 600 seconds
```

---

## Migration Guide

### v1.x to v2.0 Migration

The v2.0 format introduced several breaking changes from v1.x:

| v1.x | v2.0 | Migration Action |
|------|------|-----------------|
| Flat command list | Subcommand hierarchy | Nest commands under parent |
| Inline tool config | Separate tool definitions | Extract tools to top-level `tools` array |
| No template system | Go-template based | Migrate prompt text to `templates` |
| Single system prompt | Multi-template | Split into `system`, `user`, `response` templates |
| No validation | Schema validation | Fix any validation errors |
| No versioning | Required version field | Add version field |
| No tags | Optional tags | Add classification tags |

### Breaking Changes

1. **Command restructuring**: v1.x `commands` was a flat list. v2.0 uses `commands[].subcommands[]`.
2. **Template syntax**: Changed from `{variable}` to `{{ .variable }}`.
3. **Required fields**: `version` is now required in both SOUL.md and SKILL.md.
4. **Parameter types**: `choices` renamed to `options`, `type` values expanded.
5. **Validation added**: Previously valid files may now fail validation.

---

## Troubleshooting

### Common Errors

**"Name already in use"** — Another skill or agent has the same name. Choose a unique name.

**"Validation failed: field X is required"** — You're missing a required frontmatter field. Check the field reference table.

**"Template rendering error"** — A template variable referenced in `{{ }}` doesn't exist. Check variable names match.

**"Command not found"** — The specified command doesn't exist in any loaded skill. Verify the skill is loaded and command name is correct.

**"Timeout exceeded"** — The operation took longer than the skill's timeout. Increase timeout or optimize the operation.

**"Tool not installed"** — A tool listed in `requires.tools` is missing. Install the tool and try again.

### Debugging Tips

1. **Validate first**: Run `hermes validate` before testing.
2. **Check logs**: Set `LOG_LEVEL=debug` for detailed output.
3. **Simplify**: Strip the skill down to one command and test incrementally.
4. **Test templates**: Render templates with sample data to verify syntax.
5. **Review examples**: Compare against the complete examples in this document.

---

## FAQ

**Q: Can a single agent have multiple SOUL.md files?**
A: No. One agent = one SOUL.md. The SOUL defines the agent's core identity.

**Q: Can a SKILL.md reference another SKILL.md?**
A: No. SKILLs are self-contained. The SOUL.md lists which skills an agent uses.

**Q: What happens if a required tool is missing?**
A: The agent logs a warning and operates without that tool. Some commands may be unavailable.

**Q: Can I use environment variables in templates?**
A: Yes. Access them as `{{ .env.VARIABLE_NAME }}` in templates.

**Q: How do I share skills between agents?**
A: Place the SKILL.md in a shared directory and reference it from both agents' SOUL.md.

**Q: Is there a maximum size for a SKILL.md?**
A: SKILL.md should be under 50KB. Larger files may cause parsing timeouts.

---

## Further Reading

- [03-Prompt-Libraries.md](03-Prompt-Libraries.md) — Additional prompt templates for skills
- [04-Agent-Toolkits.md](04-Agent-Toolkits.md) — How SOUL/SKILL integrates with agent frameworks
- [08-Contribution-Templates.md](08-Contribution-Templates.md) — Templates for contributing skills
- [JSON Schema Reference](https://json-schema.org/understanding-json-schema/) — For custom validation rules
- [Go Template Documentation](https://pkg.go.dev/text/template) — Template syntax reference
- [Semantic Versioning](https://semver.org/) — Version format specification

---

*Document version 1.0 — Last updated 2026-06-12*
