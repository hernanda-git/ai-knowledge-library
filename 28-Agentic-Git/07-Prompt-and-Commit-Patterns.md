# 07 — Agentic Git: Prompt and Commit Patterns

> The shape of a good agent commit. How the commit message is constructed from the agent's prompt, the standards the 2026 ecosystem has converged on (conventional commits, structured trailers, machine-readable metadata), and the patterns that turn commit logs into queryable knowledge.

## 1. Why commit message design matters more for agents

For humans, commit messages are read occasionally — during a PR review or a `git blame`. For agents, they are **queried at machine speed** by:

- The next agent that touches the same file
- Tools like `gitlawb_bounty_claim` that index commits by type
- Lore Protocol parsers (`Ian-stetsenko/lore-protocol`, arXiv:2603.15566)
- The h5i "Agent Radio" feature that surfaces commits between Claude and Codex
- CI bots that route commits by type (fix vs feat vs chore)

A vague human commit ("stuff") is annoying. A vague agent commit breaks the downstream tools.

## 2. The 2026 conventional agent commit

The community has converged on a near-universal pattern: **conventional commits + structured trailers**.

### 2.1 The headline (subject line)

```text
<type>(<scope>): <subject>
```

Examples (from real 2026 agent PRs):

```text
fix(auth): resolve refresh-token race under concurrent load
feat(api): add rate-limiting middleware to /v2/users
refactor(db): split BillingRepository into per-tenant classes
test(checkout): cover edge case where cart is empty after coupon
docs(readme): clarify self-host prerequisites
chore(deps): bump typescript 5.5 → 5.6
```

The 12 conventional types — from `feat` and `fix` to `perf`, `revert`, `build`, `ci` — are the agent's vocabulary. Agents that pick the wrong type are graded down by automated reviewers.

### 2.2 The body (free text, but structured)

The body explains *why*, not *what*. For agents, the pattern is:

```text
The session store and the JWT refresh path can both modify the same
record, leading to intermittent 401s under high concurrency.

Switched to a Redis-backed mutex around the refresh path. Per-load
test results: 99.5% → 99.95% percentile latency at 1k rps.

Refs: ticket-142, design-doc-2026-04
```

The "Refs" trailer is the hook for downstream tools.

### 2.3 The trailer block (the structural payoff)

Trailers are key-value pairs git already understands. They ride along on every `git log` and every `git format-patch`. The 2026 standard set:

```text
feat(auth): switch session store from JWT to server-side sessions

Client-side JWTs leaked user roles into browser storage.
Server-side sessions let us revoke access instantly on permission changes.

Lore-id: a1b2c3d4
Constraint: must support horizontal scaling -- use Redis-backed store
Constraint: session TTL must not exceed 24h per compliance policy
Rejected: JWT with short expiry | still leaks roles to client
Rejected: encrypted JWT | adds decryption overhead on every request
Confidence: high
Scope-risk: wide
Reversibility: migration-needed
Directive: do not cache session objects at the application layer
Tested: concurrent session creation under load
Not-tested: Redis failover behavior
Supersedes: f7e8d9c0
Co-authored-by: Claude <noreply@anthropic.com>
Signed-off-by: Hernanda <m.hernanda95@gmail.com>
Agent-DID: did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK
Agent-Model: claude-opus-4.5
Agent-Tokens: 41203
Agent-Duration-s: 187
Reviewed-by: did:key:z6Mk…human
Refs: ticket-142, design-doc-2026-04
```

Six categories of trailer, in order of adoption:

| Category | Examples | Adoption |
|---|---|---|
| **Identity / attribution** | `Co-authored-by`, `Agent-DID`, `Signed-off-by` | Universal |
| **Lore Protocol** | `Lore-id`, `Constraint`, `Rejected`, `Confidence`, `Scope-risk`, `Reversibility`, `Directive`, `Supersedes` | Growing rapidly |
| **Process metadata** | `Tested`, `Not-tested`, `Reviewed-by`, `Refs` | Universal |
| **Provenance** | `Agent-Model`, `Agent-Tokens`, `Agent-Duration-s`, `Agent-Tools` | New standard |
| **Risk / scope** | `Scope-risk`, `Reversibility`, `Migration-needed` | Lore adoption |
| **Cross-repo** | `Refs`, `Closes`, `Supersedes`, `See-also` | Universal |

## 3. The Lore Protocol in detail

[`Ian-stetsenko/lore-protocol`](https://arxiv.org/abs/2603.15566) — *"Lore: Repurposing Git Commit Messages as a Structured Knowledge Protocol for AI Coding Agents"* by Ivan Stetsenko — defines a specific grammar for these trailers. The thesis:

> Every codebase accumulates a Decision Shadow — the reasoning behind why code exists in its current form. This knowledge lives in developers' heads, gets lost in Slack threads, and vanishes when people leave. AI coding agents suffer most.

The protocol encodes the Decision Shadow as native git trailers. Quoting from the paper's abstract:

> Lore encodes decision context directly into git commit messages using native git trailers — key-value pairs that git already supports. No extra files, no external databases, no separate tools. The knowledge lives where the code lives and travels with it.

This is exactly the inverse of the "moved to Notion / Confluence / Linear" pattern. Knowledge stays in the repo.

### 3.1 The 12 Lore trailers

| Trailer | What it encodes | Example |
|---|---|---|
| `Lore-id` | Unique identifier of this knowledge atom | `a1b2c3d4` |
| `Constraint` | A hard requirement this code must satisfy | `must support horizontal scaling` |
| `Rejected` | An alternative that was considered and dropped | `JWT with short expiry | still leaks roles to client` |
| `Confidence` | Author's confidence in this decision | `high` / `medium` / `low` |
| `Scope-risk` | How much of the codebase this change affects | `wide` / `narrow` / `isolated` |
| `Reversibility` | How hard it is to undo | `easy` / `hard` / `migration-needed` |
| `Directive` | An explicit instruction to future agents | `do not cache session objects at the application layer` |
| `Tested` | What was verified to work | `concurrent session creation under load` |
| `Not-tested` | What was NOT verified | `Redis failover behavior` |
| `Supersedes` | Lore-id this replaces | `f7e8d9c0` |
| `Refs` | External references (ticket, doc, RFC) | `ticket-142, design-doc-2026-04` |
| `Model-context` | Why the model was chosen | `Opus for planning; Sonnet for execution` |

### 3.2 Querying Lore

A 2026-era query tool can do:

```bash
$ lore query --module auth --type constraint
a1b2c3d4: session TTL must not exceed 24h per compliance policy
e5f6g7h8: must support horizontal scaling -- use Redis-backed store
i9j0k1l2: do not log session contents (PII)

$ lore query --module auth --type rejected
a1b2c3d4: JWT with short expiry | still leaks roles to client
a1b2c3d4: encrypted JWT | adds decryption overhead on every request

$ lore query --module auth --directive
a1b2c3d4: do not cache session objects at the application layer
```

The next agent reading the auth module can ask "what constraints apply?" and get a structured answer instead of grepping for `// TODO` comments.

## 4. The h5i Agent Radio pattern

[`h5i-dev/h5i`](https://h5i.dev/) (391★) introduces the concept of **prompt-aware commits** and an "Agent Radio" that lets Claude and Codex talk to each other through the commit log.

### 4.1 The commit shape

```text
feat(api): add /v2/users pagination

[LIVE] Agent A: starting work on ticket-142
[CHECK] Agent A: tests pass locally
[HANDOFF] Agent A: tests are green but CI needs to run; pause for review
[RESUME] Agent B (Codex): picking up from Agent A's handoff
[DONE] Agent B: PR ready for human review

Agent-Radio-Session: 2026-06-17-abc123
Agent-Radio-State: handoff
Agent-DID-A: did:key:z6Mk…
Agent-DID-B: did:key:z6Mk…
Agent-Prompt-A: "implement /v2/users pagination per ticket-142"
Agent-Prompt-B: "complete Agent A's work; merge after green"
```

When a new agent reads the commit, it sees the conversation that produced it.

### 4.2 The token reduction trick

h5i claims up to **95% token reduction** by "shrinking noisy tool output while keeping the raw evidence." The mechanism:

- Tool output is recorded in a side-channel (file or git note) at full fidelity
- The agent's commit message references the side-channel by ID instead of inlining the output
- Future agents can pull the side-channel when they need full context

Example:

```text
feat(api): add /v2/users pagination

[Refs full tool trace: .agent_radio/session-abc123.jsonl]
[Refs raw test output: .agent_radio/test-out-abc123.txt]

Summary: 7/7 unit tests pass; integration pending; see full trace.
```

The agent's own context window stays clean. Humans get a skim-able summary. Researchers can replay the full session.

## 5. Conventional commits enforcement

A pre-commit hook that validates the subject against conventional commit rules:

```bash
#!/bin/bash
# .git/hooks/commit-msg

COMMIT_MSG_FILE="$1"
SUBJECT=$(head -n1 "$COMMIT_MSG_FILE")

PATTERN='^(feat|fix|docs|style|refactor|perf|test|chore|build|ci|revert)(\([a-z0-9-]+\))?!?: .+$'

if ! echo "$SUBJECT" | grep -qE "$PATTERN"; then
  echo "ERROR: subject does not match conventional commit pattern"
  echo "  Got:      $SUBJECT"
  echo "  Expected: <type>(<scope>): <subject>"
  echo "  Types: feat fix docs style refactor perf test chore build ci revert"
  exit 1
fi

# Optional: lint trailers
if grep -qE "^(wip|TODO|XXX):" "$COMMIT_MSG_FILE"; then
  echo "WARNING: commit message contains WIP/TODO markers"
  echo "  These indicate an unfinished commit; consider amending before merge"
fi
```

For agents, the same hook is invoked automatically. Tools like `commitlint` (Node) or `go-gitlint` are more sophisticated.

## 6. Commit-message generation patterns

### 6.1 The minimal pattern (default for Claude Code)

The agent takes its prompt, infers the type and scope, and writes the subject.

```text
Prompt: "Fix the auth refresh race condition"
→ fix(auth): resolve auth refresh race condition
```

When this is too terse, append a body.

### 6.2 The structured pattern (Lore Protocol)

After the basic conventional subject, append Lore trailers (template):

```text
<type>(<scope>): <subject>

<body explaining why>

Lore-id: <random-hex>
Constraint: <hard requirements this change satisfies>
Rejected: <alternative considered and why dropped>
Confidence: <high|medium|low>
Scope-risk: <wide|narrow|isolated>
Reversibility: <easy|hard|migration-needed>
Tested: <what was verified>
Not-tested: <what was NOT verified>
Refs: <ticket>, <doc>
Agent-DID: did:key:<agent>
Agent-Model: <model-id>
Agent-Tokens: <count>
```

### 6.3 The handoff pattern (multi-agent)

```text
<type>(<scope>): <subject>

[CONTEXT] <summary of the prior agent's work>
[HANDOFF] <what's left to do>
[STATE] <current state of the worktree>
```

Example:

```text
feat(checkout): implement /cart POST validation

[CONTEXT] Started on ticket-289. Wrote schema, migration, and 12 tests.
All pass locally.
[HANDOFF] Need to add rate-limiting per IP and integrate with the existing
user-service client. Rate-limit config is in /config/ratelimit.yaml.
[STATE] Branch agent-a/checkout-validation, 4 commits, all green.
Next agent should branch from here.
```

The next agent (or a human) can pick up where the previous agent stopped.

## 7. Anti-patterns (what to reject)

### 7.1 The prompt-as-commit-message

```text
> fix the bug in auth.ts
```

This is what some 2025 agents produced when given a vague prompt. It tells the next agent nothing.

**Fix:** Use `autogit`-style tooling that detects when the prompt itself would make a useless subject and substitutes a file-list summary.

### 7.2 The mega-commit

```text
WIP: refactor everything
```

Or a 10,000-line diff in a single commit.

**Fix:** Squash-on-merge; enforce one-logical-change-per-commit via CI.

### 7.3 The emoji parade

```text
🚀 feat: ✨ add ✨ new ✨ feature 🎉
```

Cute, but breaks log parsing and AST tools.

**Fix:** Commit-message lint that rejects non-ASCII in the subject.

### 7.4 The attributions-only trailer

```text
Co-authored-by: Claude <noreply@anthropic.com>
```

That's the *minimum* trailer. No `Agent-DID`, no `Refs`, no rationale. Useful for nothing downstream.

**Fix:** Required-trailer policy in commit-msg hook.

## 8. The agent prompt → commit-message pipeline

The standard 2026 pattern in pseudocode:

```python
def make_commit_message(agent_state: dict, diff: str) -> str:
    """Generate a structured commit message from agent state."""

    # 1. Infer type from the task
    type_, scope = infer_conventional_type_and_scope(agent_state.task, diff)

    # 2. Write subject from the agent's summary
    subject = agent_state.summary.strip()[:72]

    # 3. Build the body
    body = build_body(
        rationale=agent_state.rationale,
        constraints=agent_state.constraints,
        rejected_alternatives=agent_state.rejected,
        tested=agent_state.tests_passed,
        not_tested=agent_state.tests_skipped,
    )

    # 4. Append trailers
    trailers = {
        "Lore-id": generate_lore_id(),
        "Confidence": agent_state.confidence,
        "Scope-risk": agent_state.scope_risk,
        "Reversibility": agent_state.reversibility,
        "Tested": agent_state.tested_summary,
        "Not-tested": agent_state.untested_summary,
        "Refs": ",".join(agent_state.refs),
        "Co-authored-by": f"{agent_state.model_name} <noreply@{agent_state.vendor}>",
        "Agent-DID": agent_state.did,
        "Agent-Model": agent_state.model_id,
        "Agent-Tokens": agent_state.token_count,
        "Agent-Duration-s": agent_state.duration_seconds,
    }

    # 5. Serialize
    msg = f"{type_}({scope}): {subject}\n\n{body}\n\n"
    msg += "\n".join(f"{k}: {v}" for k, v in trailers.items())
    return msg
```

This is roughly what `claude-code`, `cursor-agent`, and `opencode` do internally. The differences are in how `infer_conventional_type_and_scope` is implemented — typically a small LLM call or a rule-based classifier on the diff.

## 9. The commit log as a knowledge graph

If you follow the patterns above, your `git log` becomes queryable as a graph. Tools that exploit this:

- `lore-protocol` — queryable Lore atoms (constraints, rejections, directives)
- `h5i` — Agent Radio session replay
- `GitNexus` (`abhigyanpatwari/GitNexus`, 42k★) — code intelligence graph over commits
- `sem` (`Ataraxy-Labs/sem`, 2.9k★) — entity-level diffs and impact analysis
- `entireio/cli` (4.5k★) — agent session capture linked to commits

The 2026 thesis: **a project's commit log is its most valuable documentation, because it preserves the decision shadow that no other artifact captures.**

## 10. Migration playbook (existing repos)

If you have a repo with existing agent commits that don't follow these patterns:

1. **Day 1:** Add the conventional commit lint hook. Don't rewrite history yet; let new commits be clean.
2. **Day 7:** Backfill top-N important files with Lore trailers on the next touching commit.
3. **Day 30:** Generate a `LORE.md` at the repo root summarizing the major architectural decisions (constraints, rejections) extracted from commit history.
4. **Day 90:** Migrate from per-PR squash to per-feature squash, so each merged commit is a Lore atom.

## 11. See also

- Cat 02 — *AI Prompt Engineering* (how the prompt shapes the output)
- Cat 13 — *AI Engineering & DevOps* (commit hooks and CI integration)
- **02-Core-Topics** — identity and signing (where Agent-DID lives)
- **08-Agent-Memory-and-Long-Running-Workflows** — cross-session knowledge from commits
- **09-Replay-Debug-and-Observability** — capturing agent reasoning into the commit
- Lore paper: [arxiv.org/abs/2603.15566](https://arxiv.org/abs/2603.15566)
- h5i: [h5i.dev](https://h5i.dev/)
