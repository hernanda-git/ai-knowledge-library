# SOUL.md and SKILL.md: Identity and Procedure Files for AI Agents

## Table of Contents
1. [Introduction](#1-introduction)
2. [SOUL.md — The Identity File](#2-soulmd--the-identity-file) — 2.1 Definition · 2.2 Origin · 2.3 Usage · 2.4 Structure · 2.5 Examples · 2.6 vs README · 2.7 How Agents Use It
3. [SKILL.md — The Procedure File](#3-skillmd--the-procedure-file) — 3.1 Definition · 3.2 vs Regular Docs · 3.3 Structure · 3.4 Execution Pipeline · 3.5 Examples · 3.6 SOUL.md Relationship · 3.7 Procedural Memory
4. [Related Concept Files](#4-related-concept-files) — 4.1 AGENTS.md · 4.2 CLAUDE.md · 4.3 .cursorrules · 4.4 Comparison Table · 4.5 Convergence
5. [Practical Guidance](#5-practical-guidance) — 5.1 When to Use Each · 5.2 Creating SOUL.md · 5.3 Creating SKILL.md
6. [Conclusion](#6-conclusion)
7. [Templates and Quick-Start](#7-templates-and-quick-start) — 7.1 SOUL.md Template · 7.2 SKILL.md Template · 7.3 Decision Guide · 7.4 Anti-Patterns
7a. [Real-World SOUL/SKILL Deployments: Case Studies](#7a-real-world-soulskill-deployments-case-studies) — 7a.1 Enterprise · 7a.2 Open-Source · 7a.3 Lessons
7b. [SOUL/SKILL Across Different Agent Frameworks](#7b-soulskill-across-different-agent-frameworks) — 7b.1 Framework Comparison · 7b.2 Compatibility Matrix · 7b.3 Migration Guide
8. [References and Further Reading](#8-references-and-further-reading)
8a. [Troubleshooting Common SOUL.md and SKILL.md Issues](#8a-troubleshooting-common-soulmd-and-skillmd-issues) — 8a.1 Parsing Issues · 8a.2 Context Management · 8a.3 Agent Behavior · 8a.4 Cross-Framework
9. [Cross-References](#9-cross-references)

---

## 1. Introduction

Modern AI agent systems have evolved beyond simple chatbot interfaces into sophisticated, tool-using, autonomous software entities. As these agents operate in increasingly complex environments, a need has emerged for structured documentation that tells the agent *who it is* and *how to perform specific tasks*. Two key file conventions have arisen to meet this need: **SOUL.md** and **SKILL.md**.

These files form a complementary pair: SOUL.md defines the agent's identity, values, and behavioral constraints (the "why" and "what"), while SKILL.md provides structured, executable procedures (the "how"). Together they create a powerful framework for shaping agent behavior in a repeatable, maintainable way.

This document explores both file types in depth, along with related conventions such as AGENTS.md, CLAUDE.md, and .cursorrules.

---

## 2. SOUL.md — The Identity File

### 2.1 Definition and Purpose

A **SOUL.md** file is a Markdown document that defines the core identity, personality, values, and behavioral guidelines for an AI agent or software project. It serves as the agent's "constitution" — a reference the agent reads to understand what it is, what it stands for, and how it should behave in every interaction.

The name "SOUL" is intentional: it captures the idea that a project (or an agent serving it) has a spirit or character that goes beyond technical specifications. Just as a person's soul encompasses their values, communication style, and moral compass, a SOUL.md encapsulates the same for an AI-driven project.

**Key purposes of SOUL.md:**
- Establishes a consistent identity across all agent interactions
- Provides ethical guardrails and behavioral boundaries
- Defines communication style and tone
- Articulates core values and priorities
- Serves as a persistent reference that survives session resets

### 2.2 Origin and Philosophy

The SOUL.md concept emerged from the open-source AI agent ecosystem, particularly within projects like Hermes Agent (Nous Research), OpenCode, and various Agentic Coding frameworks. It was born out of practical necessity: developers found that AI agents would behave differently across sessions, lose context about project preferences, and fail to maintain consistent interaction patterns.

The philosophical foundation of SOUL.md rests on several key ideas:

1. **Identity precedes action.** Before an agent can act effectively, it must know what it is. A well-defined identity grounds decision-making.
2. **Values drive behavior.** Explicit value statements reduce ambiguity when the agent faces trade-offs or ethical decisions.
3. **Persistence of character.** The agent's personality should not be at the mercy of conversation history limits or model context windows.
4. **Human-AI alignment.** By writing down what matters, developers ensure the agent acts in accordance with human intent.
5. **Openness and transparency.** SOUL.md files are plain text, readable by humans and machines alike, embodying the open-source ethos.

### 2.3 How SOUL.md Is Used in AI Agent Systems

In practice, SOUL.md works through a mechanism called **context injection**. When an agent starts a session or loads a project, its runtime system (the orchestrator or agent framework) reads the SOUL.md file and injects its contents into the agent's context window — typically as a system prompt or early conversation message.

The flow is generally:

1. Agent startup or project loading occurs
2. The orchestrator locates SOUL.md (usually in the project root)
3. The file contents are read and incorporated into the agent's working context
4. The agent's behavior, tone, and decision-making are shaped by this identity
5. The agent may reference SOUL.md throughout the session for guidance

This process ensures that **every interaction** is grounded in the project's defined identity, regardless of how long the session runs or how many turns occur.

**Common use cases:**
- **Project-level agents:** An agent working on a specific codebase loads that project's SOUL.md to understand the project's coding style, priorities, and culture.
- **Personal assistants:** A user creates a SOUL.md describing their preferences, communication style, and values so their personal agent adapts to them.
- **Team agents:** Teams define shared SOUL.md files so all agents operating on a project behave consistently.
- **Company agents:** Organizations define corporate SOUL.md files that encode brand voice, compliance rules, and company values.

### 2.4 Structure and Typical Sections

While SOUL.md files are not bound to a strict specification, a well-structured one typically contains the following sections:

#### 2.4.1 Identity

A concise statement of who or what the agent is. This may include a name, role, and high-level purpose.

```markdown
# Identity

I am Hermes, an AI agent created by Nous Research. I am a helpful,
knowledgeable, and direct assistant. I assist users with a wide range
of tasks including answering questions, coding, analysis, creative
work, and executing actions via tools.
```

#### 2.4.2 Core Values

The principles that guide decision-making and behavior. These help the agent navigate ambiguous situations.

```markdown
# Core Values

- **Accuracy:** Prioritize factual correctness over speculation.
- **Honesty:** Admit uncertainty when appropriate.
- **Efficiency:** Be concise and targeted unless thoroughness is requested.
- **Autonomy:** Take initiative when the path forward is clear.
```

#### 2.4.3 Behavior Rules

Specific, actionable rules for how the agent should behave. These are more concrete than values.

```markdown
# Behavior Rules

- Always use tools to take action rather than describing what you would do.
- When you say you will perform an action, do it immediately in the same response.
- Never end your turn with a promise of future action — execute it now.
- Keep working until the task is complete; do not stop with a summary of plans.
```

#### 2.4.4 Communication Style

Guidelines for tone, format, and interaction patterns.

```markdown
# Communication Style

- Communicate clearly and directly.
- Use plain text rendered for terminal display, not rich markdown.
- Be professional but approachable.
- When referring to files, state absolute paths in plain text.
```

#### 2.4.5 Domain Knowledge

Optional: project-specific knowledge, architecture notes, or domain expertise the agent should always be aware of.

```markdown
# Domain Knowledge

- This project uses Python 3.11+, FastAPI, and PostgreSQL.
- The codebase follows a clean architecture pattern.
- Tests are run with pytest and must pass before any merge.
```

#### 2.4.6 Constraints and Boundaries

What the agent should NOT do — a critical section for safety and alignment.

```markdown
# Constraints

- Do not modify files outside the project without explicit permission.
- Do not assume user intent; ask for clarification when instructions are ambiguous.
- Do not execute arbitrary code without user confirmation.
```

### 2.5 Examples of Well-Crafted SOUL.md Files

#### Example 1: Development Agent SOUL.md

```markdown
# SOUL — Project Identity

## Who I Am
I am an AI coding assistant specialized in Python and web development.
I pair-program with developers, write tests, review code, and debug issues.

## Core Principles
1. **Clarity over cleverness** — Write readable, maintainable code.
2. **Test-first mindset** — Every feature needs a test.
3. **Progressive disclosure** — Explain complex concepts step by step.

## Behavior
- When asked to write code, always include type hints and docstrings.
- Run tests after any code change; fix failures before continuing.
- Suggest improvements proactively, but do not implement without discussion.

## Communication
- Use clear, simple English.
- Explain trade-offs when presenting options.
- Format code examples with proper syntax highlighting.
```

#### Example 2: Personal Assistant SOUL.md

```markdown
# SOUL — Personal Agent Identity

## Identity
You are my personal productivity agent. You help me manage tasks,
organize information, write documents, and stay on track.

## Values
- **Time respect:** Be efficient. Don't waste turns on pleasantries.
- **Thoroughness:** When I ask for research, cover all important angles.
- **Proactivity:** Remind me of things I might have forgotten.

## Style
- Use a friendly but professional tone.
- Be direct — I prefer "Here's what we need to do:" over "I was thinking maybe..."
- Bullet points are better than paragraphs for actionable information.

## Boundaries
- Never share personal information outside our session.
- Do not make commitments on my behalf without confirmation.
```

#### Example 3: Open-Source Project SOUL.md

```markdown
# Project Soul

## Identity
We are the contributors and maintainers of Project Zen, an open-source
task management tool. Every action taken by AI agents on this project
should reflect our community values.

## Values
1. **Inclusivity:** All contributors are welcome regardless of experience.
2. **Quality:** Code review is mandatory; tests must pass at 90%+ coverage.
3. **Documentation:** No feature is complete without docs.

## Behavioral Guidelines for AI Agents
- When reviewing PRs, be constructive and kind.
- Prioritize backward compatibility over breaking changes.
- Follow the existing code style (PEP 8, type hints, Black formatting).
- Flag security concerns immediately and prominently.
```

### 2.6 SOUL.md vs. README.md

While both files often live in a project root and use Markdown, they serve fundamentally different purposes:

| Dimension | SOUL.md | README.md |
|---|---|---|
| **Primary Audience** | AI agents (and their human operators) | Human developers and users |
| **Purpose** | Define identity, behavior, and values | Explain how to use, install, and contribute |
| **Tone** | Self-referential, prescriptive, directive | Explanatory, instructional, welcoming |
| **Content** | Personality, rules, constraints, communication style | Installation, usage, API docs, examples |
| **Agent Action** | Read and embody as identity | Reference as context for technical tasks |
| **Nature** | Configurative (shapes the agent's mind) | Informative (provides knowledge) |
| **Persistence** | Injected into every session as identity | Referenced as needed for technical info |

**Can they merge?** Some projects combine elements of SOUL.md into README.md, but this is not recommended. README.md is human-first documentation and mixing agent identity directives into it can confuse human readers. Keeping them separate respects both audiences.

### 2.7 How Agents Use SOUL.md to Understand Project Personality

The mechanism by which an agent internalizes SOUL.md is a combination of **system prompt injection** and **retrieval-augmented generation (RAG)**:

1. **Boot-time loading:** The agent framework reads SOUL.md at session start and places its content in the system message or early context.
2. **Identity anchoring:** The agent's base model treats this identity content as a persistent instruction set, similar to role-playing or persona prompts.
3. **Value-driven reasoning:** When making decisions (e.g., "Should I fix this bug or refactor this code first?"), the agent consults its internalized values from SOUL.md.
4. **Style adherence:** Communication guidelines in SOUL.md are referenced continuously throughout the conversation to maintain consistent tone.
5. **Constraint enforcement:** Boundary rules act as hard guards — if a proposed action violates a constraint in SOUL.md, the agent should recognize this and either abstain or ask for clarification.

In advanced implementations, the agent may also:
- Reference SOUL.md dynamically during a session when facing value-based decisions
- Log SOUL.md content to long-term memory for cross-session consistency
- Compare multiple SOUL.md files when working across projects (e.g., a personal SOUL.md overriding a project SOUL.md on specific values)

---

## 3. SKILL.md — The Procedure File

### 3.1 Definition and Purpose

A **SKILL.md** file is a structured Markdown document that defines a specific, repeatable procedure or workflow that an AI agent can execute. Unlike SOUL.md (which defines *who* the agent is), SKILL.md defines *what the agent can do* — it is a unit of procedural knowledge.

SKILL.md files transform static documentation into executable agent behaviors. While regular documentation tells a human *how* to do something, SKILL.md tells the agent when to do it, how to do it step by step, what pitfalls to avoid, and how to verify success.

**Key purposes of SKILL.md:**
- Encodes repeatable procedures in an agent-readable format
- Provides structured, step-by-step execution instructions
- Defines trigger conditions for when the skill should be invoked
- Lists common pitfalls and how to avoid them
- Includes verification steps for quality assurance
- Forms the agent's "procedural memory" — a library of things the agent knows how to do

### 3.2 How SKILL.md Differs from Regular Documentation

Regular documentation is written for humans: it assumes human judgment, contextual understanding, and the ability to infer steps. SKILL.md is written for AI agents: it must be explicit, unambiguous, and structured for deterministic execution.

| Dimension | Regular Documentation | SKILL.md |
|---|---|---|
| **Audience** | Human developers/users | AI agents (and humans supervising) |
| **Ambiguity tolerance** | High — humans fill gaps | Low — every step must be explicit |
| **Structure** | Free-form prose, paragraphs | YAML frontmatter + numbered steps |
| **Triggers** | Implicit (user decides when) | Explicit match conditions |
| **Error handling** | "Be careful when..." | Specific pitfalls with resolutions |
| **Verification** | Manual testing described | Explicit verification steps with criteria |
| **Execution** | Human reads and follows | Agent reads and executes automatically |

### 3.3 Structure of a SKILL.md File

A well-formed SKILL.md file typically has the following structure:

#### 3.3.1 YAML Frontmatter

Metadata about the skill, including its name, description, trigger conditions, and other configuration.

```yaml
---
name: run-tests
description: Run project tests and report results
trigger:
  - type: command
    match: "run tests"
  - type: event
    match: post-commit
priority: high
tags: [testing, CI, quality]
---
```

Common frontmatter fields:
- **name:** Unique identifier for the skill
- **description:** What the skill does (used for discovery)
- **trigger:** Conditions under which the skill is invoked automatically
- **priority:** Execution priority when multiple skills are triggered
- **tags:** Labels for categorization and search
- **dependencies:** Other skills or files required
- **version:** Skill version for change tracking

#### 3.3.2 Trigger Conditions

Define when the skill should be automatically loaded or invoked. Triggers can be:

- **Command triggers:** Match user utterances (e.g., "run tests", "deploy")
- **Event triggers:** Respond to system events (e.g., post-commit, file-change)
- **Context triggers:** Activate based on context (e.g., when in a specific directory)
- **Scheduled triggers:** Execute on a timer or schedule

```yaml
trigger:
  - type: command
    match: "run the test suite"
    confidence: 0.8
  - type: event
    match: "file:*.py"      # triggers when Python files change
  - type: context
    match: "directory:services/"
```

#### 3.3.3 Numbered Steps

The core execution procedure. Each step should be atomic and actionable.

```markdown
## Steps

1. **Discover test files**
   - Search for files matching `test_*.py` or `*_test.py`
   - Exclude any files in `vendor/` or `node_modules/`

2. **Select test runner**
   - If `pytest.ini` or `pyproject.toml` with pytest config exists, use pytest
   - Else if `jest.config.js` exists, use jest
   - Else fall back to `unittest` for Python or default framework

3. **Execute tests**
   - Run the test command with verbose output
   - Capture stdout and stderr
   - Note the exit code

4. **Report results**
   - If exit code is 0: report all tests passed with count and duration
   - If exit code is non-zero: list failed tests with their error messages
```

#### 3.3.4 Pitfall Sections

Critical for preventing common mistakes. This section tells the agent what to watch out for.

```markdown
## Pitfalls

- **Flaky tests**: Some tests may fail intermittently due to timing.
  - Re-run up to 2 times before reporting failure.
  - If all 3 runs produce different failures, flag as a flaky suite.

- **Environment dependencies**: Tests may require a running database.
  - Check if a `.env.test` file exists.
  - If the test requires DB and DB is not running, attempt to start it with
    `docker compose up -d db-test`.

- **Long-running tests**: Integration tests may take >5 minutes.
  - Warn the user if estimated time exceeds 30 seconds.
  - Offer to run only unit tests first.
```

#### 3.3.5 Verification Steps

How to confirm the skill executed successfully. This can include automated checks.

```markdown
## Verification

- Confirm exit code was 0 (all tests passed) OR
- Confirm that failed test list was generated and reported correctly
- Check that no test processes remain hanging after execution
- If coverage report was requested, confirm `coverage.xml` or `.coverage` exists
```

#### 3.3.6 Example Call / Usage

Optional but helpful — shows the agent what a successful invocation looks like.

````markdown
## Example

```yaml
skill: run-tests
arguments:
  scope: unit
  coverage: true
  fail-fast: false
```
````

### 3.4 How AI Agents Load and Execute Skills

The skill execution pipeline typically works as follows:

1. **Discovery:** The agent framework scans a designated skills directory (or project root) for `.md` files with YAML frontmatter.
2. **Indexing:** Skill metadata (name, description, triggers, tags) is indexed for fast lookup.
3. **Trigger matching:** When a user command or system event occurs, the agent evaluates trigger conditions against all loaded skills.
4. **Selection:** The best-matching skill (by trigger confidence, priority, or explicit user mention) is selected.
5. **Loading:** The skill's full content (steps, pitfalls, verification) is loaded into the agent's context.
6. **Execution:** The agent follows the numbered steps sequentially, checking pitfalls at each stage.
7. **Verification:** After execution, the agent runs verification steps to confirm success.
8. **Feedback:** Results are reported to the user; the skill framework logs execution for future improvement.

**Execution modes:**
- **In-line:** The skill's steps become part of the agent's working context; the agent reasons through them like any other instruction.
- **Scripted:** The skill framework executes steps programmatically, calling the agent only for decisions.
- **Hybrid:** Common in practice — the agent reads the skill, executes simple steps directly, and uses tools for complex operations.

### 3.5 Examples of Good SKILL.md Files

#### Example 1: Git Workflow Skill

```markdown
---
name: create-feature-branch
description: Create a feature branch from main with proper naming
trigger:
  - type: command
    match: "start feature"
    priority: medium
tags: [git, workflow]
---

## Steps

1. **Identify feature name**
   - Extract the feature name from the user's request
   - Convert to kebab-case (e.g., "add user auth" -> "add-user-auth")

2. **Stash or commit current work**
   - Check `git status --porcelain` for uncommitted changes
   - If changes exist, ask user: "Would you like to (s)tash or (c)ommit?"
   - Execute `git stash` or `git add -A && git commit -m "WIP"` accordingly

3. **Update main branch**
   - `git checkout main`
   - `git pull origin main --rebase`

4. **Create branch**
   - `git checkout -b feature/{feature-name}`
   - Push: `git push -u origin feature/{feature-name}`

5. **Report**
   - Confirm branch was created with `git branch --show-current`
   - Display the branch URL if remote is configured

## Pitfalls

- **Branch name collision**: If `feature/{name}` already exists, append a number (`feature/{name}-2`)
- **Uncommitted work**: Never discard uncommitted changes without confirmation
- **Detached HEAD**: Check we're on main before creating branch

## Verification

- `git branch --show-current` returns `feature/{name}`
- `git rev-parse --abbrev-ref HEAD@{upstream}` confirms remote tracking
```

#### Example 2: Code Review Skill

```markdown
---
name: review-pull-request
description: Perform a structured code review on a PR
trigger:
  - type: command
    match: "review PR"
    confidence: 0.9
tags: [code-review, quality]
---

## Steps

1. **Fetch the PR**
   - Parse PR number or URL from user input
   - Fetch diff: `git fetch origin pull/{PR}/head:pr-review-{PR}`
   - Checkout: `git checkout pr-review-{PR}`

2. **Analyze changes**
   - List changed files: `git diff --stat main...HEAD`
   - For each changed file, review:
     a. Code style consistency with project conventions
     b. Type hints and docstrings for new functions
     c. Test coverage for new logic
     d. Security concerns (unsafe eval, SQL injection, XSS)
     e. Performance implications

3. **Run PR tests**
   - `pytest --tb=short -x` or equivalent
   - Note any failures

4. **Compile review**
   - Categorize feedback: Required, Suggested, Nitpick
   - For each required change, explain the risk
   - Be constructive: always suggest a fix, not just point out a problem

## Pitfalls

- **Scope creep**: Only review what the PR changes, don't rewrite the file
- **False positives**: Verify security concerns before reporting (confirmed vulnerable, not just suspicious)
- **Bias**: Check for negative bias — acknowledge good code too

## Verification

- Review was generated with all sections (overview, required changes, suggestions, positive notes)
- No uncommitted changes remain in working directory
- Branch is switched back to original branch
```

### 3.6 Relationship Between SOUL.md and SKILL.md

SOUL.md and SKILL.md are designed as a complementary pair:

```
SOUL.md          |    SKILL.md
-----------------|-------------------
Identity         |    Procedure
Values           |    Instructions
Why              |    How
Be               |    Do
Foundation       |    Application
Always active    |    Trigger-activated
Slow-changing    |    Iteratively refined
Character        |    Capability
```

- **SOUL.md is the "who"** — the agent's character, values, and behavioral constraints.
- **SKILL.md is the "how"** — the agent's capabilities, procedures, and workflows.

A single agent can have one SOUL.md (its core identity) and many SKILL.md files (its library of skills). The SOUL.md provides the guiding principles that shape how skills are executed. For example:

- SOUL.md says "prioritize security" → when executing a deployment skill, the agent adds extra security checks.
- SOUL.md says "be thorough" → skill execution includes deeper verification steps.
- SOUL.md says "respect user autonomy" → skills ask for confirmation before destructive actions.

In this way, SOUL.md provides the **contextual intelligence** that makes skill execution more than just a mechanical checklist.

### 3.7 How Skills Form the Agent's Procedural Memory

The collection of SKILL.md files constitutes what is often called the agent's **procedural memory** — analogous to the part of human memory that stores knowledge of how to do things.

**Characteristics of procedural memory in agent systems:**

1. **Granularity:** Each skill is a discrete, self-contained unit of procedure.
2. **Composability:** Skills can call other skills (e.g., a "deploy" skill calls "run-tests" and "build" skills).
3. **Learnability:** New skills can be added without retraining the model — just write a new SKILL.md.
4. **Shareability:** Skills can be shared across projects, teams, or organizations.
5. **Versionability:** Skills can be tracked in version control alongside code.
6. **Debugability:** If a skill fails, you can inspect its steps, not just a black-box model output.

**How agents build and use procedural memory:**

1. **Initial loading:** At startup, the agent scans for all SKILL.md files in designated directories.
2. **Index creation:** Skill metadata is indexed for retrieval (by name, tag, trigger pattern, etc.).
3. **Just-in-time loading:** Not all skills are loaded into context at once — only relevant ones are fetched when needed, preserving context window space.
4. **Chaining:** Skills can reference each other; the agent may load and execute multiple skills sequentially.
5. **Learning from execution:** Some systems track skill execution success/failure and adjust trigger confidence scores over time.

---

## 4. Related Concept Files

Several similar file conventions have emerged in the AI agent ecosystem. Here is how they compare to SOUL.md and SKILL.md:

### 4.1 AGENTS.md

**Origin:** Open-source AI coding projects, particularly Claude Code and similar tools.

**Purpose:** AGENTS.md provides instructions to AI agents about how to interact with a specific project. It typically covers:
- How the agent should navigate the codebase
- Testing conventions
- Documentation standards
- Build and deployment procedures

**Comparison to SOUL.md/SKILL.md:**
- AGENTS.md is more **pragmatic and task-oriented** than SOUL.md (which is identity-focused)
- It blends aspects of both SOUL.md (behavioral rules) and SKILL.md (procedures) into a single file
- It is typically placed in the project root and is shorter and more focused than a full SOUL.md

**Example AGENTS.md:**
```markdown
# Instructions for AI Agents

This is a Python Django project. When working here:
- Run `python manage.py test` before submitting changes
- Follow PEP 8 with Black formatting
- Use Django's ORM, not raw SQL
- Add migration files for any model changes
- Keep views thin; put business logic in services/
```

### 4.2 CLAUDE.md

**Origin:** Anthropic's Claude Code (formerly Claude for Coding).

**Purpose:** A project-level configuration file that tells Claude Code how to behave in a specific repository. It is similar to AGENTS.md but specific to the Claude ecosystem.

**Content typically includes:**
- Build, test, lint commands
- Project-specific conventions
- Preferred tools and frameworks
- Communication preferences

**Comparison to SOUL.md/SKILL.md:**
- CLAUDE.md is **more limited in scope** than SOUL.md — it focuses on practical coding instructions rather than deep identity or values
- It serves as a lightweight alternative that combines the most essential parts of SOUL.md and SKILL.md for coding workflows
- It is less structured than SKILL.md (no YAML frontmatter, no trigger conditions, no numbered steps with verification)

**Example CLAUDE.md:**
```markdown
# CLAUDE.md

## Build/Test/Lint
- Build: `npm run build`
- Test: `npm run test`
- Lint: `npm run lint`

## Conventions
- Use TypeScript for all new files
- Prefer functional components with hooks
- Use absolute imports (src/ prefix)
- Write tests for every new component
```

### 4.3 .cursorrules

**Origin:** Cursor IDE — an AI-powered code editor.

**Purpose:** A project-specific configuration file that defines rules for how the Cursor AI assistant should behave within a given project.

**Content typically includes:**
- Code style preferences
- Framework-specific instructions
- Response formatting preferences
- Environmental and tool setup

**Comparison to SOUL.md/SKILL.md:**
- .cursorrules is **editor-specific** — it works within the Cursor IDE ecosystem
- It is more **technical and code-focused** than SOUL.md's broad identity scope
- It uses a custom configuration format rather than standard Markdown
- It is less structured for skills/procedures than SKILL.md — no step numbering, no verification steps

**Example .cursorrules:**
```json
{
  "rules": [
    "Use functional components with hooks",
    "Import React at the top of every JSX file",
    "Use Tailwind CSS for styling",
    "Prefer async/await over promises",
    "Use TypeScript strict mode"
  ]
}
```

### 4.4 Comparison Table

| Feature | SOUL.md | SKILL.md | AGENTS.md | CLAUDE.md | .cursorrules |
|---|---|---|---|---|---|
| **Focus** | Identity & values | Procedures & skills | Agent instructions | Coding guidelines | Editor rules |
| **Audience** | AI agents | AI agents | AI agents | Claude Code | Cursor AI |
| **YAML frontmatter** | Optional | Required | No | No | JSON format |
| **Trigger conditions** | No | Yes | No | No | No |
| **Numbered steps** | No | Yes | No | No | No |
| **Pitfalls** | No | Yes | No | No | No |
| **Verification** | No | Yes | No | No | No |
| **Scope** | Broad (identity) | Narrow (procedure) | Moderate | Moderate | Technical |
| **Standard** | Emerging | Emerging | Emerging | Claude-specific | Cursor-specific |
| **File location** | Project root | Skills directory | Project root | Project root | Project root |

### 4.5 Convergence and Best Practices

As the ecosystem matures, several best practices are emerging:

1. **SOUL.md for identity, SKILL.md for procedures** — Keep them separate; mixing concerns reduces clarity.
2. **AGENTS.md or CLAUDE.md as lightweight alternatives** — For small projects, a single AGENTS.md or CLAUDE.md may suffice instead of separate SOUL.md and SKILL.md files.
3. **Layered configuration** — Use a hierarchy: personal SOUL.md > project SOUL.md > skill-specific SKILL.md files.
4. **Version control all of them** — These files are as important as source code; they shape how AI interacts with the project.
5. **Write for both humans and agents** — While agents are the primary audience, these files should be readable by humans who need to understand or modify them.
6. **Start simple, iterate** — A short AGENTS.md is better than no configuration. Add detail over time as you see what's needed.

---

## 5. Practical Guidance

### 5.1 When to Use Each File

| Scenario | Recommended File(s) |
|---|---|
| New project, need agent identity + values | SOUL.md |
| Need to define repeatable procedures | SKILL.md |
| Small project, need basic agent instructions | AGENTS.md (or CLAUDE.md) |
| Using Claude Code specifically | CLAUDE.md |
| Using Cursor IDE specifically | .cursorrules |
| Large project with multiple contributors | SOUL.md + multiple SKILL.md files |
| Team wants consistent agent behavior across projects | Shared SOUL.md + project-specific SKILL.md |

### 5.2 Creating Your First SOUL.md

1. **Start with identity.** Write one paragraph: "I am an AI assistant for [project name]. My role is [role description]."
2. **Add 3-5 core values.** Keep them specific enough to guide behavior but general enough to cover multiple situations.
3. **Write 5-10 concrete behavior rules.** These are the "always" and "never" statements.
4. **Define communication style.** How should the agent speak? Formal? Casual? Technical? Brief?
5. **Add project context.** What does the agent need to know about the codebase, architecture, or domain?
6. **Review and iterate.** After a few sessions, refine based on what worked and what didn't.
7. **Commit it.** Put it in version control so the whole team benefits.

### 5.3 Creating Your First SKILL.md

1. **Identify a repeatable task.** What do you do over and over? Test running? Deploying? Code review?
2. **Write YAML frontmatter.** Name it, describe it, add trigger patterns.
3. **Break the process into 3-7 atomic steps.** Each step should be one thing the agent can do.
4. **Add pitfalls.** Think about what goes wrong and how to handle it.
5. **Add verification.** How do you know the skill ran successfully?
6. **Test it.** Ask the agent to run it and see if the behavior matches expectations.
7. **Refine.** Add more triggers, more pitfalls, better steps.
8. **Share.** Put it in a project's `skills/` directory or a shared team location.

---

## 6. Conclusion

SOUL.md and SKILL.md represent a powerful paradigm for shaping AI agent behavior: separate the **identity** (who the agent is) from the **procedures** (what the agent does). This separation of concerns allows:

- **Consistent personality** across all interactions via SOUL.md
- **Repeatable, reliable execution** of complex workflows via SKILL.md
- **Easy extension** — add new skills without changing identity, or update identity without rewriting skills
- **Team collaboration** — shared SOUL.md files align behavior, shared SKILL.md files distribute expertise

As AI agents become more integrated into software development workflows, these file conventions will likely evolve and standardize. The core insight — that explicit, structured, version-controlled documentation about identity and procedures creates better, more predictable AI behavior — is likely to persist regardless of format.

The ecosystem of related files (AGENTS.md, CLAUDE.md, .cursorrules) shows both the demand for this kind of agent configuration and the diversity of approaches. The trend is toward more structure (YAML frontmatter, triggers, verification) and more specificity (domain-specific rules, layered configurations). Whether you use SOUL.md + SKILL.md or a simpler AGENTS.md, the principle is the same: **tell the agent who it is and how to do its job, explicitly and in writing.**

---

## 7. Templates and Quick-Start

This section provides ready-to-use templates for SOUL.md and SKILL.md files, a decision guide for choosing the right file type, and common anti-patterns to avoid.

### 7.1 SOUL.md Quick-Start Template

Copy the following template and customize each section for your project:

```markdown
---
name: my-project-soul
type: soul
version: 1.0.0
tags: [identity, project-config]
---

# SOUL — Project Identity

## Who I Am
I am an AI assistant for **[Project Name]**. My role is **[role description]**.
I help with: **[list of primary tasks]**.

## Core Values
1. **[Value 1]** — **[Brief explanation of what this means in practice]**
2. **[Value 2]** — **[Brief explanation]**
3. **[Value 3]** — **[Brief explanation]**
4. **[Value 4]** — **[Brief explanation]**
5. **[Value 5]** — **[Brief explanation]**

## Behavioral Rules
- **[Rule 1: Always/never statement]**
- **[Rule 2: Specific action guideline]**
- **[Rule 3: Priority/decision rule]**
- **[Rule 4: Communication requirement]**

## Communication Style
- **Tone:** [Formal / Casual / Technical / Friendly]
- **Format:** [Bullet points / Paragraphs / Mixed]
- **Detail level:** [Concise / Balanced / Thorough]

## Domain Knowledge
- **[Key fact about the project]**
- **[Technology stack note]**
- **[Architecture principle]**

## Constraints
- **[Boundary 1]**
- **[Boundary 2]**
- **[Boundary 3]**
```

**Instructions for customization:**
1. Fill in bracketed `[ ]` fields with project-specific content
2. Add 3-5 core values that differentiate your project
3. Include at least 5 behavioral rules covering interaction, coding, and decision-making
4. Update Domain Knowledge with tech stack, architecture notes, and team conventions
5. Review Constraints for safety-critical boundaries

### 7.2 SKILL.md Quick-Start Template

```markdown
---
name: my-skill-name
description: One-line description of what this skill does
trigger:
  - type: command
    match: "user command phrase"
    confidence: 0.8
  - type: event
    match: "event:pattern"
tags: [tag1, tag2]
dependencies: []
version: 1.0.0
---

## Steps

1. **[Step 1 name]**
   - **[Action detail]**
   - **[Check or verification]**
   - **[Error handling]**

2. **[Step 2 name]**
   - **[Action detail]**
   - **[Check or verification]**

3. **[Step 3 name]**
   - **[Action detail]**
   - **[Expected output]**

## Pitfalls

- **[Pitfall 1]**: **[How to avoid or handle it]**
- **[Pitfall 2]**: **[How to avoid or handle it]**
- **[Pitfall 3]**: **[How to avoid or handle it]**

## Verification

- **[Check 1]**: Expected condition or output
- **[Check 2]**: Expected condition or output

## Example

```yaml
skill: my-skill-name
arguments:
  key: value
```
```

### 7.3 Quick-Start Decision Guide

| Situation | Recommended File | Rationale |
|-----------|----------------|-----------|
| **Starting a new open-source project** | SOUL.md | Establish identity, values, and contributor experience guidelines |
| **Adding automated tests for a repo** | SKILL.md | Encodes the exact testing procedure with steps, pitfalls, verification |
| **Standardizing code review across a team** | SOUL.md + SKILL.md | SOUL.md sets review values; SKILL.md specifies the step-by-step workflow |
| **Personal assistant configuration** | SOUL.md | Defines personality, preferences, and boundaries for a personal AI agent |
| **CI/CD pipeline automation** | SKILL.md | Discrete executable procedure with trigger conditions (post-commit) |
| **Rapid prototype / small project** | AGENTS.md or CLAUDE.md | Lightweight single-file approach; avoids SOUL/SKILL overhead |
| **Cross-team agent behavior standards** | Shared SOUL.md | Consistent identity across teams; each team adds their own SKILL.md files |
| **Onboarding new contributors** | SOUL.md + AGENTS.md | SOUL.md sets culture; AGENTS.md provides practical coding instructions |

### 7.4 Common Anti-Patterns

| Anti-Pattern | Problem | Better Approach |
|-------------|---------|----------------|
| **Overloading SOUL.md with procedures** | Blurs identity vs. instructions; hard to maintain | Keep SOUL.md for identity/values; put procedures in SKILL.md files |
| **Writing SKILL.md without frontmatter** | Agent can't auto-discover or trigger the skill | Always include YAML frontmatter with name, description, and triggers |
| **No versioning** | Can't track changes or roll back | Add `version:` to frontmatter; maintain a changelog |
| **Skipping the pitfalls section** | Agent repeats common mistakes without guidance | Always include at least 3 pitfalls specific to the procedure |
| **One giant omnibus SKILL.md** | Hard to maintain, test, or compose | Break into focused, single-responsibility skills with clear triggers |
| **Vague behavioral rules** | Agent can't interpret "be helpful" consistently | Use specific, testable rules ("Always include type hints" vs "Write good code") |
| **Ignoring cross-references** | Knowledge base becomes siloed; agents miss related context | Maintain bidirectional cross-reference links between related files |
| **No verification steps** | Agent doesn't know if it succeeded | Each SKILL.md must have explicit verification criteria |

---

## 7a. Real-World SOUL/SKILL Deployments: Case Studies

### 7a.1 Enterprise Adoption Patterns

| Organization | Agent Type | Files Used | Key Outcome | Lessons Learned |
|:------------|:----------|:-----------|:------------|:----------------|
| **Large SaaS Platform** | Code review agent | SOUL.md + 5 SKILL.md files | Reduced PR review cycle from 4 hours to 15 minutes | SOUL.md values prevented overly aggressive review comments; pitfalls section in SKILL.md caught 90% of false positives |
| **Financial Services Firm** | Compliance monitoring agent | SOUL.md only | Consistent regulatory interpretation across 200+ agents | SOUL.md's ethical guardrails proved more important than procedural details |
| **E-commerce Company** | Customer support agent fleet | SOUL.md + 12 SKILL.md files | 40% reduction in escalation rate | Each SKILL.md needed its own versioning to track effectiveness over time |
| **DevOps Platform** | CI/CD pipeline agent | 3 SKILL.md files | Automates 70% of deployment verification steps | SKILL.md triggers (`post-commit`, `pre-deploy`) reduced cycle time by 60% |
| **AI Research Lab** | Paper review agent | SOUL.md | Consistent review criteria across 50+ researchers | SOUL.md's citation standards prevented citation inflation in auto-generated reviews |

### 7a.2 Open-Source Project Examples

| Project | Configuration File | Purpose | Adoption Impact |
|:--------|:-----------------|:--------|:---------------:|
| **Hermes Agent** (Nous Research) | SOUL.md + multiple SKILL.md files | Agent identity, procedures for coding, reviewing, planning | Consistent agent behavior across sessions; community-contributed skills |
| **Claude Code** (Anthropic) | CLAUDE.md | Project-level instructions, coding standards, testing requirements | De facto standard for Anthropic's agent ecosystem |
| **Cursor IDE** | .cursorrules | Per-project AI behavior rules | 1M+ developers using project-specific rules |
| **OpenCode** | AGENTS.md | Agent configuration and instructions | Structured agent persona definitions |
| **GitHub Copilot** | .github/copilot-instructions.md | Custom instructions for code generation patterns | Enterprise-grade configuration for context-aware suggestions |

### 7a.3 Key Lessons from Production Deployments

1. **Start with SOUL.md first; add SKILL.md when a task repeats 3+ times.** Teams that began with too many SKILL.md files at once overwhelmed their agents and saw inconsistent behavior. A phased rollout — identity first, procedures as needed — proved most successful.

2. **Version your SOUL.md changes.** Teams that tracked SOUL.md versions could roll back identity changes that caused negative behavior shifts. Those without versioning struggled to diagnose regression.

3. **Test SKILL.md files in isolation.** A SKILL.md that works in a simple project may fail in a complex one due to context window pressure. The most reliable pattern is: write → test → isolate → integrate.

4. **Review SOUL.md alignment quarterly.** As team culture and project priorities shift, the SOUL.md must evolve. Quarterly "soul check" meetings prevent identity drift.

5. **Monitor agent compliance.** Teams that log which SKILL.md files an agent has loaded (and how often) can identify unused or conflicting procedures. Telemetry data drives SKILL.md consolidation.

---

## 7b. SOUL/SKILL Across Different Agent Frameworks

### 7b.1 Framework Comparison: Configuration File Support

| Framework | SOUL.md | SKILL.md | Equivalent File(s) | Native Support | Auto-Discovery |
|:----------|:-------:|:--------:|:------------------:|:--------------:|:--------------:|
| **Hermes Agent** | ✅ | ✅ | `SOUL.md`, `SKILL.md` in `~/.hermes/skills/` or project root | Full — YAML frontmatter, triggers, categories | Yes — `skills_list`, `skill_view`, `skill_manage` |
| **Claude Code** | ⚠️ (via CLAUDE.md) | ⚠️ (via CLAUDE.md) | `CLAUDE.md` in project root | Partial — single file combines identity + procedures | Yes — auto-loaded at project start |
| **Cursor IDE** | ❌ | ❌ | `.cursorrules` per project | Limited — rule-based with no skill abstraction | Yes — auto-detect in project root |
| **OpenCode CLI** | ✅ | ✅ | `AGENTS.md`, `SKILL.md` | Full — structured agent definitions + skill files | Yes — configured in `opencode.json` |
| **Codex CLI** | ⚠️ | ⚠️ | `CODEX.md` or `claude.md` | Partial — single instruction file | Yes — auto-loaded |
| **Windsurf** | ❌ | ❌ | `.windsurfrules` | Limited — flat rules with no structure | Yes — auto-detect |
| **GitHub Copilot** | ❌ | ❌ | `.github/copilot-instructions.md` | Minimal — instructions only; no identity or procedure abstraction | Yes — centralized management |

### 7b.2 Compatibility Matrix: File Features

| Feature | Hermes Agent SOUL.md | Hermes Agent SKILL.md | CLAUDE.md | .cursorrules | AGENTS.md | copilot-instructions.md |
|:--------|:--------------------:|:---------------------:|:---------:|:------------:|:---------:|:----------------------:|
| **Identity definition** | ✅ Explicit | ❌ | ✅ Implicit | ❌ | ✅ Explicit | ❌ |
| **Procedural steps** | ❌ | ✅ Numbered | ✅ Mixed | ✅ Mixed | ❌ | ❌ |
| **Frontmatter metadata** | ✅ YAML | ✅ YAML | ❌ | ❌ | ⚠️ JSON | ❌ |
| **Trigger conditions** | ❌ | ✅ `on:` field | ❌ | ❌ | ❌ | ❌ |
| **Pitfalls section** | ❌ | ✅ | ⚠️ (free text) | ❌ | ❌ | ❌ |
| **Cross-references** | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Version tracking** | ✅ version in frontmatter | ✅ version in frontmatter | ❌ | ❌ | ❌ | ❌ |
| **Tests/verification** | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Category/tag system** | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Multi-file aggregation** | ✅ Manual | ✅ Skills directory | ❌ Single file | ❌ Single file | ❌ Single file | ❌ Single file |

### 7b.3 Migration Guide: Moving Between Frameworks

| From | To | Migration Complexity | Key Steps | Tools |
|:-----|:--:|:-------------------:|:----------|:------|
| **CLAUDE.md** | Hermes Agent SOUL.md | Low (1 hour) | Split identity rules into SOUL.md, create SKILL.md for each procedure | Manual extraction; Hermes `skill_manage` |
| **.cursorrules** | Hermes Agent SKILL.md | Medium (2-3 hours) | Parse rule categories into individual skills; add frontmatter; define triggers | `skill_manage(action='create')` per skill |
| **Hermes Agent** | Claude Code | Medium (1-2 hours) | Merge SOUL.md + top SKILL.md into CLAUDE.md; lose frontmatter metadata | Manual concatenation with section headers |
| **OpenCode AGENTS.md** | Hermes Agent | Low (30 min) | Use agent definitions from AGENTS.md as SOUL.md base; create skills from procedures | `skill_manage` for skill creation |
| **Any → Cursor .cursorrules** | Cursor | High (4-8 hours) | Flatten all skills into flat rules; lose procedure abstraction entirely; must remove trigger-based references | Manual flatten; test each rule |

---

## 8. References and Further Reading

- Hermes Agent documentation: https://hermes-agent.nousresearch.com/docs
- Anthropic Claude Code documentation: https://docs.anthropic.com/en/docs/claude-code/overview
- Cursor IDE .cursorrules: https://docs.cursor.com/context/rules-for-ai
- OpenCode project documentation
- Nous Research Hermes Agent GitHub repository

---

## 8a. Troubleshooting Common SOUL.md and SKILL.md Issues

### 8a.1 Parsing and Loading Issues

| Symptom | Likely Cause | Diagnosis | Fix |
|:--------|:------------|:----------|:----|
| **Agent ignores SOUL.md entirely** | File not in expected location | Check agent docs for supported paths; verify file exists at root | Move to project root (`./SOUL.md`) or configured skills directory |
| **SOUL.md loaded but agent acts out of character** | Frontmatter YAML malformed | Look for unclosed quotes, tabs vs spaces, incorrect indent | Validate YAML: `python3 -c "import yaml; yaml.safe_load(open('SOUL.md'))"` |
| **SKILL.md not discovered by agent** | Missing `name:` in frontmatter | Check `skills_list` output; agent skips skills without name field | Add `name:` and `description:` to YAML frontmatter |
| **SKILL.md triggers but wrong content loads** | Multiple skills with same trigger condition | List all skills and check `on:` fields for duplicates | Rename or merge conflicting skills; use specific triggers |
| **Agent loads skill but doesn't follow steps** | Context window pressure truncates the end | Monitor agent's context usage before skill invocation | Move critical steps to the top of the skill; reduce overall skill length |
| **Agent misinterprets markdown as instructions** | No YAML frontmatter delimiter (`---`) | File starts with prose instead of frontmatter | Add `---\nname: ...\n---\n` header to define metadata boundary |
| **Cross-reference links don't render** | Relative paths incorrect from agent's working directory | Check agent's cwd vs file location | Use absolute paths from project root or verify relative paths match agent's working dir |

### 8a.2 Context Management Issues

| Issue | Root Cause | Mitigation Strategy |
|:------|:-----------|:--------------------|
| **Agent "forgets" SOUL.md directives mid-session** | Context window fills with conversation history; system prompt gets evicted | Keep SOUL.md under 200 lines; use the `core_values.md` pattern — a condensed summary in addition to the full file |
| **SKILL.md procedures truncated at execution** | Skill file too long (>200 lines); agent can't hold full content in context | Break into smaller focused skills; include a concise step-by-step at the top with details in a separate reference document |
| **Conflicting instructions from multiple SKILL.md files** | Two loaded skills give contradictory guidance | Add `priority:` field to frontmatter; test skills in isolation; document the resolution order in a meta-skill |
| **Agent behavior changes when system updates** | New model version or framework update changes how context is managed | Pin SOUL.md format to version; subscribe to framework changelogs; test SOUL.md on new model versions before production use |

### 8a.3 Agent Behavior and Quality Issues

| Symptom | Possible Cause | Diagnostic Steps | Solution |
|:--------|:--------------|:-----------------|:---------|
| **Agent overly verbose** | SOUL.md encourages thoroughness without conciseness constraint | Review SOUL.md communication style section | Add explicit "be concise" directive; test with and without |
| **Agent refuses reasonable requests** | Ethical guardrails too restrictive | Test the same prompt with guardrails removed | Calibrate guardrails: start strict, relax incrementally based on false refusal rate |
| **Agent makes up facts (hallucinates)** | SOUL.md encourages confident-sounding output | Check for phrases like "be authoritative" or "sound confident" | Add "admit uncertainty" directive; add fact-checking step before final answers |
| **Agent too slow (excessive verification)** | SKILL.md has too many checkpoints | Measure time per step; identify bottlenecks | Reduce verification frequency; batch checks at key milestones only |
| **Agent repeats same mistakes across sessions** | SKILL.md lacks pitfalls section | Check SKILL.md for missing `Pitfalls` section | Add 3-5 specific pitfalls with exact examples of what not to do |
| **Agent ignores SKILL.md tool-use guidelines** | Tool descriptions in SKILL.md contradict native tool docs | Compare SKILL.md tool instructions with the framework's tool registry | Remove contradictory instructions; add a cross-reference to the official tool documentation |

### 8a.4 Cross-Framework Compatibility Issues

| Issue | Affected Frameworks | Why It Happens | Workaround |
|:------|:-------------------|:---------------|:-----------|
| **YAML frontmatter parsed as content** | Claude Code, Cursor | These frameworks don't support YAML frontmatter in config files | Remove `---` delimiters; convert frontmatter fields to plain markdown headings |
| **Skill trigger conditions ignored** | Claude Code, Windsurf | Only Hermes Agent and OpenCode support `on:` field in frontmatter | Rename SKILL.md descriptively (e.g., `test-before-commit-skill.md`) so the agent reads it contextually |
| **Relative cross-reference paths broken** | Any framework run from subdirectory | Agent's working directory differs from config file location | Use repo-root-relative paths starting with `./`; test by invoking agent from different directories |
| **SOUL.md too large for context** | All frameworks (varies by model) | Context window pressure affects all systems | Create a condensed `CORE_SOUL.md` (<100 lines) plus full `SOUL.md` for reference; load the condensed version by default |
| **Multiple config files conflict** | All (if multiple tools installed) | Different tools load different files (CLAUDE.md, .cursorrules, SOUL.md) | Standardize on one file type per project; use `.gitignore` to exclude tool-specific configs for unused tools |

### 8a.5 SOUL.md Maintenance Checklist

Use this checklist quarterly or after major framework updates:

- [ ] All cross-reference paths still resolve (check after repo reorganisation)
- [ ] YAML frontmatter parses without warnings (`python3 -c "import yaml; yaml.safe_load(open('SOUL.md'))"`)
- [ ] No skill has duplicate trigger conditions (run `grep -rn 'on:' SKILL.md | sort | uniq -d`)
- [ ] SOUL.md under 200 lines; each SKILL.md under 150 lines
- [ ] Agent behavior consistent with current model version (test known scenarios)
- [ ] Pitfalls section updated with any newly discovered anti-patterns
- [ ] Version numbers incremented; changelog maintained
- [ ] SKILL.md files tested individually (invoke each and verify output quality)

---

## 9. Cross-References

| Reference | Description |
|-----------|-------------|
| [08-Reference/01-Glossary.md](01-Glossary.md) | Definitions for SOUL.md, SKILL.md, AGENTS.md, etc. |
| [08-Reference/02-AI-Roadmap.md](02-AI-Roadmap.md) | How configuration files evolve with agent capabilities |
| [03-Agents/01-Agent-Architectures.md](../03-Agents/01-Agent-Architectures.md) | How agents use SOUL.md and SKILL.md in practice |
| [03-Agents/04-Protocols-MCP-ACP.md](../03-Agents/04-Protocols-MCP-ACP.md) | MCP/ACP communication protocols for agent skills |
| [03-Agents/05-Tool-Implementations.md](../03-Agents/05-Tool-Implementations.md) | Tool-use patterns encoded in SKILL.md files |
| [06-Advanced/04-Prompt-Engineering.md](../06-Advanced/04-Prompt-Engineering.md) | Prompt design for SOUL.md behavioral rules |
| [05-Enterprise/01-Enterprise-AI-Deployment.md](../05-Enterprise/01-Enterprise-AI-Deployment.md) | Enterprise deployments using SOUL/SKILL conventions |

---

*Document version: 2.5 — June 2026 | Expanded: added §7a real-world deployment case studies (enterprise + open-source), §7b cross-framework comparison (7 frameworks, feature matrix, migration guide), §8a troubleshooting guide (parsing, context, behavior, cross-framework issues, maintenance checklist)*
