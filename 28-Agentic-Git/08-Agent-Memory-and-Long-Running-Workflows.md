# 08 — Agentic Git: Agent Memory and Long-Running Workflows

> How an agent remembers across sessions, why git is the natural substrate for that memory, and the patterns for stateful multi-session agent work — from the agent's episodic log to cross-repo semantic memory.

## 1. The memory problem for agents

Every fresh agent invocation starts with a context window that doesn't know:

- What the previous agent in this repo learned
- Which decisions were already made and why
- What files have already been touched this sprint
- What the user's style preferences are
- Which patterns the team has rejected

For a one-shot task, this is fine. For an agent that works on the same codebase over weeks — the reality in 2026 — it's a dealbreaker. The agent re-derives everything, re-litigates rejected decisions, makes the same mistakes.

Git is the cheapest, most universally available substrate for agent memory. It already exists in every project. It has versioning. It has signed commits. It has diffs. The trick is using it as memory, not just history.

## 2. The four layers of agent memory

| Layer | Substrate | Scope | TTL | Example |
|---|---|---|---|---|
| **Working memory** | Context window | Single turn | < 1 min | The current diff being written |
| **Episodic memory** | Git notes, side-files | Single session | Days-weeks | Agent's reasoning trace during this task |
| **Semantic memory** | Lore trailers, docs | Multi-session | Months-years | Project-wide constraints and decisions |
| **Procedural memory** | Skills, .cursorrules, AGENTS.md | Cross-project | Permanent | "Always run linter before commit" |

Git-native memory lives in layers 2 and 3. Layer 4 is configuration files that also live in git. Layer 1 is the model's context window — not git's concern.

## 3. Episodic memory: capturing what the agent did this session

The simplest pattern: after every agent turn, append a structured entry to a per-session log file, then `git add` and commit.

### 3.1 The session-log pattern

```text
.agent_radio/
  sessions/
    2026-06-17-abc123/
      prompt.md          # what the user asked
      plan.md            # the agent's plan
      tool_calls.jsonl   # every tool invocation
      diff.patch         # the resulting diff
      result.md          # final summary
      metrics.json       # tokens, duration, model
```

Each entry committed at the end of the turn:

```bash
git add .agent_radio/sessions/2026-06-17-abc123/
git commit -m "agent(turn): session 2026-06-17-abc123 — implemented pagination

Lore-id: <random>
Agent-DID: did:key:z6Mk…
Agent-Model: claude-opus-4.5
Agent-Tokens: 41203
Agent-Duration-s: 187
Refs: ticket-142"
```

`entireio/cli` (4.5k★) is the canonical implementation. It "hooks into your Git workflow to capture AI agent sessions as you work." Sessions are replayable.

### 3.2 The git-notes alternative

`git notes` is git's built-in mechanism for attaching out-of-band data to commits. Use it instead of a side-file if you want everything in git without bloating the tree:

```bash
git notes --ref agent-session add -F session.md HEAD
git notes --ref agent-reasoning add -F reasoning.md HEAD
git notes --ref agent-metrics add -F metrics.json HEAD
```

Notes can be pushed to remote (`git push origin refs/notes/*`) without affecting the visible history.

### 3.3 What to capture

The minimum useful capture:

```json
{
  "session_id": "2026-06-17-abc123",
  "agent_did": "did:key:z6Mk…",
  "agent_model": "claude-opus-4.5",
  "started_at": "2026-06-17T07:42:11Z",
  "ended_at": "2026-06-17T07:45:18Z",
  "duration_s": 187,
  "tokens": {
    "input": 18420,
    "output": 22783,
    "cache_read": 0,
    "cache_write": 0
  },
  "task_summary": "Implement /v2/users pagination per ticket-142",
  "files_touched": ["src/api/v2/users.ts", "tests/api/v2/users.test.ts"],
  "tools_used": ["Read", "Edit", "Grep", "Bash"],
  "test_outcome": "12/12 passing",
  "decisions": [
    {
      "choice": "Cursor-based pagination",
      "rejected": ["offset-based (slow at scale)", "keyset (requires schema migration)"],
      "rationale": "Cursor is stateless from the server side and works with the existing index."
    }
  ],
  "next_steps": ["add rate-limiting", "integrate with user-service"],
  "prompt": "<verbatim user prompt>",
  "diff_hash": "sha256:9a2f…"
}
```

A few sessions in, you have a queryable history of every choice your agent made.

## 4. Semantic memory: Lore + decision shadow

Already covered in chapter 07. The summary: agents that commit with structured Lore trailers (`Constraint:`, `Rejected:`, `Confidence:`, `Reversibility:`) build a queryable semantic graph across the whole repo.

The cross-session payoff:

```bash
# New agent, day 30 of working on the same project
$ lore query --module auth --type constraint
a1b2c3d4: session TTL must not exceed 24h per compliance policy
e5f6g7h8: must support horizontal scaling -- use Redis-backed store

# The new agent now knows:
# - Don't propose local-memory sessions (violates horizontal-scaling constraint)
# - Don't propose long TTLs (violates compliance)
# - The team has already considered and rejected JWT-with-short-expiry
# → Fewer bad PRs, faster iteration
```

## 5. Procedural memory: skills, rules, conventions

Stored in repo-root files that the agent reads at session start.

### 5.1 The AGENTS.md pattern

`AGENTS.md` (or `CLAUDE.md`, `.cursorrules`, `.github/copilot-instructions.md`) is a free-form markdown file the agent reads before doing anything. Conventions:

- **Anthropic / Claude Code:** `CLAUDE.md` at repo root
- **Cursor:** `.cursorrules` (legacy) or `.cursor/rules/*.mdc`
- **OpenCode:** `AGENTS.md` at repo root
- **GitHub Copilot:** `.github/copilot-instructions.md`
- **Codex:** `AGENTS.md` at repo root
- **Aider:** `CONVENTIONS.md` or `~/.aider.conf.yml`

A canonical 2026 AGENTS.md:

```markdown
# AGENTS.md

## Build / test / lint
- Install: `pnpm install`
- Test: `pnpm test`
- Lint: `pnpm lint`
- Type-check: `pnpm typecheck`
- Build: `pnpm build`

## Before every commit
- Run `pnpm test` and `pnpm lint` — both must pass
- Run `gitleaks protect` — no secrets
- Sign the commit with `git commit -S` (your SSH key)
- Subject line: `<type>(<scope>): <subject>` (conventional commits)

## Code style
- TypeScript strict mode, no `any`
- Prefer `readonly` for module-level state
- All public functions need JSDoc with `@example`
- Use the existing `Result<T, E>` type for fallible operations — never throw

## Architecture decisions (do not violate)
- All DB access goes through `src/db/` — never query directly
- All external HTTP calls use the shared `httpClient` with retry policy
- Auth uses server-side sessions, NOT JWT — see Lore-id a1b2c3d4

## What not to do
- Do NOT modify `prisma/schema.prisma` without human review
- Do NOT touch `infrastructure/terraform/**`
- Do NOT add new top-level dependencies without discussion
- Do NOT use `eval()` or `new Function()` anywhere

## Agent-specific
- Always commit as: `Co-authored-by: <agent-name> <noreply@vendor>`
- Use the project's pre-commit hooks; do not bypass with `--no-verify`
- If a task spans > 1000 lines, ask before starting
- When uncertain, prefer smaller PRs over larger ones
```

This file is the agent's "house rules." Committed to git, it survives across sessions, across humans, across agent runs.

### 5.2 The skills pattern (loadable modules)

Where AGENTS.md is general-purpose instructions, **skills** are loadable, scoped, named bundles of context. The OpenCode / Claude Code / openclaude pattern:

```text
skills/
  backend-api/
    SKILL.md             # the skill's instructions
    examples/            # worked examples
    scripts/             # helper scripts
  frontend-react/
    SKILL.md
  devops-terraform/
    SKILL.md
```

When the agent starts a backend API task, it loads `skills/backend-api/SKILL.md` and inherits the conventions there. Skills are git-tracked, version-controlled, sharable.

## 6. Cross-session agent identity

The deepest form of agent memory: **the agent knows it's the same agent across sessions.**

Mechanisms:

1. **Persistent Ed25519 DID.** Generated once, stored at `~/.agents/<name>.pem`, referenced as `Agent-DID: did:key:z6Mk…` in every commit. Future sessions load the same key.
2. **Per-identity agent state.** A JSON file at `~/.agents/<name>/state.json` that records:
   - Total commits signed
   - Repos touched
   - Reputation score
   - Last task summary
3. **State files in the repo.** `.agent_state/<agent-did>/` directory tracked in git, containing the agent's evolving notes about the project. Updated at the end of each session.
4. **Self-models in Lore.** The agent's commits to Lore form an implicit autobiography. `lore query --author did:key:z6Mk…` returns every decision the agent has made.

### 6.1 The agent's "memory budget"

A practical limit: a 2026 agent's effective long-term memory is bounded by:

```
agent_memory ≈
    lore_protocol_atoms
  + AGENTS.md + skills/
  + last_30_days_of_session_logs
  + reputation_graph
```

For projects beyond ~100k LOC, the Lore protocol alone is too much to fit in any single context window. The agent must *navigate* its memory, not *load* it. The query API (`lore query`) becomes essential.

## 7. Long-running workflows (multi-day agent work)

The hardest case: an agent works on a feature for 5 days, across many sessions, with humans reviewing in between. How does state survive?

### 7.1 The handoff commit pattern

```text
feat(checkout): stage 3 of 5 — validation layer

[CONTEXT] Stages 1-2 implemented the schema and service layer (commits abc, def).
Tests passing locally. Schema deployed to staging.
[HANDOFF] Stage 4 needs: rate-limiting middleware, integration with user-service.
The rate-limit config lives at /config/ratelimit.yaml — read it before designing.
[STATE] Branch agent-a/checkout-stage-3, 7 commits, all green.
CI: green on commit ghi789.
Next session: branch from here.
```

A human (or another agent) can `git checkout` the branch, read the body, and know exactly what to do next.

### 7.2 The state-machine pattern

Some 2026 agent frameworks persist a state machine in a git-tracked file:

```json
// .agent_state/<session-id>/state.json
{
  "session_id": "ticket-289-impl",
  "current_state": "awaiting_human_review",
  "stages": [
    {"id": "schema", "status": "done", "commit": "abc123"},
    {"id": "service", "status": "done", "commit": "def456"},
    {"id": "validation", "status": "in_review", "commit": "ghi789", "pr": "#142"},
    {"id": "rate_limit", "status": "pending"},
    {"id": "deploy", "status": "pending"}
  ],
  "context_for_next_session": "..."
}
```

The file is committed at every state transition. A new agent (or human) reads the file, knows exactly where the work is, picks up.

### 7.3 The "Git as message queue" pattern

For truly long workflows (weeks), the repo itself becomes the message bus:

```text
.agent_queue/
  in_progress/
    ticket-289.json     # active work
  completed/
    ticket-142.json     # finished
  failed/
    ticket-156.json     # rolled back
```

Each entry is a JSON file with the task spec, current state, and result. Agents claim tasks by moving files between directories (via `git mv`) — git's atomicity guarantees no double-claim.

```bash
# Agent A claims a task
git mv .agent_queue/pending/ticket-289.json .agent_queue/in_progress/
git commit -m "agent: claim ticket-289"

# ... does work ...

# Agent A completes
git mv .agent_queue/in_progress/ticket-289.json .agent_queue/completed/
git commit -m "agent: complete ticket-289"
```

This is the agentic-git equivalent of a job queue, with git as the lock manager. Gitlawb bounties are the economic layer on top of this same primitive.

## 8. Cross-repo agent memory

A 2026 agent typically works across multiple repos. Memory that crosses repo boundaries is harder.

### 8.1 The monorepo case

If your repos are already a monorepo (Nx, Turborepo, Bazel), git itself is the cross-project memory. `AGENTS.md` at the root applies to all packages. Lore is queryable across the whole monorepo.

### 8.2 The polyrepo case

For polyrepo setups:

- **Dedicated memory repo.** `~/memory/<agent-did>.git` — a separate git repo the agent owns, holds its long-term notes. Cloned into any project as `.agent_memory` (gitignored).
- **External knowledge base.** A Qdrant / Postgres / vector DB the agent queries at session start. Less git-native but scales better.
- **Cross-repo Lore federation.** Multiple repos publish Lore atoms to a shared query service. New agents can query "all constraints across the org's repos" via a federated `lore query`.

## 9. The decay problem (memory that goes stale)

Memory decays. A constraint written in 2024 may be obsolete in 2026. The patterns to manage decay:

| Pattern | Mechanism | Cost |
|---|---|---|
| **Confidence + reviewed-at trailers** | `Confidence: low (last-reviewed 2025-01)` | Cheap; relies on agents checking the date |
| **Time-based expiry** | `Valid-until: 2026-12-31` | Cheap; brittle if dates wrong |
| **Dependency tracking** | Lore atom includes `depends-on: package@version`; atom expires when the dep changes | Medium; needs dep tracking |
| **Periodic re-validation** | Scheduled agent task that re-reads all Lore atoms and confirms them | Expensive; should be off-peak |
| **Supersedes chain** | `Supersedes: f7e8d9c0` — old atoms are marked obsolete | Cheap; requires discipline |

## 10. The "second-day" problem

When an agent returns to a repo on day 2, the worst experience is:

- It forgets what it did yesterday
- It re-derives context from files (slow, lossy)
- It re-litigates decisions that were already settled

The fix is everything in this chapter, in priority order:

1. **AGENTS.md at the repo root.** Immediate, cheap, 80% of the value.
2. **Lore trailers on every commit.** Medium effort, large long-term payoff.
3. **Session logs at `.agent_radio/`.** Cheap, gives you observability and replay.
4. **Persistent agent identity with DID.** Necessary for reputation and trust.
5. **State machine in `.agent_state/`.** Only needed for long-running multi-stage work.
6. **Cross-repo memory repo.** Only if you have ≥ 5 repos and the same agent works them.

## 11. Tools that operationalize this

| Tool | What it does | URL |
|---|---|---|
| `entireio/cli` | Captures agent sessions into git-attached JSONL logs | github.com/entireio/cli |
| `h5i` | Agent Radio (commit-mediated agent-to-agent messaging) + token-efficient sessions | h5i.dev |
| `lore-protocol` | Structured trailers + query API | github.com/Ian-stetsenko/lore-protocol |
| `Mybono/ai-orchestrator` | Self-learning loop that records outcomes and evolves skills | github.com/Mybono/ai-orchestrator |
| `agent-radio` (concept) | Multi-agent messaging via commit logs | (no canonical implementation yet) |
| `.agent_state/` (convention) | State files in git for long-running workflows | (no tool; convention) |

## 12. See also

- Cat 02 — *AI Prompt Engineering* (how prompt-level memory interacts)
- Cat 03 — *RAG and Knowledge Systems* (vector-DB memory patterns)
- Cat 12 — *AI Agent Architecture* (state machines in agent frameworks)
- **03-Technical-Deep-Dive** — implementation of identity + commit signing
- **07-Prompt-and-Commit-Patterns** — Lore trailers in detail
- **09-Replay-Debug-and-Observability** — how memory supports debugging
