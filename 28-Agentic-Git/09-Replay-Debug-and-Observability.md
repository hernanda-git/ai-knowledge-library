# 09 — Agentic Git: Replay, Debug, and Observability

> When an agent's commit breaks production at 3 AM, how do you figure out why? The patterns and tools for capturing, replaying, and debugging agent sessions — and the postmortem conventions the 2026 ecosystem is converging on.

## 1. The debug problem with agents

When a human breaks production, the postmortem flow is:

1. Read the offending commit (`git show <sha>`)
2. Read the PR discussion
3. Read the diff carefully
4. Trace through the local repro
5. Talk to the author
6. Write the postmortem

When an *agent* breaks production, none of those steps work cleanly:

- `git show` shows the diff but not *why* the agent chose it
- The PR discussion is empty (auto-merged) or low-signal (agent-generated)
- The diff is huge (agent's "fix" rewrote 50 files)
- The local repro doesn't reproduce the model's reasoning
- There is no author to talk to
- The postmortem needs a different structure

The 2026 answer: **record everything the agent did, structured for replay.**

## 2. What "replay" means for an agent session

A replayable agent session captures enough state that you can:

1. **Re-run** the agent on the same prompt and get the same answer (with the same model and temperature=0).
2. **Inspect** the agent's reasoning at any step (what it was thinking when it called each tool).
3. **Inspect** the agent's context at any step (what the model saw).
4. **Diff two sessions** to see why they diverged.
5. **Branch** from any intermediate step.

The minimum capture:

```json
{
  "session_id": "2026-06-17-abc123",
  "schema_version": "1.0",
  "agent": {
    "did": "did:key:z6Mk…",
    "model": "claude-opus-4.5",
    "vendor": "anthropic",
    "temperature": 0.0,
    "max_tokens": 8192,
    "system_prompt_hash": "sha256:abc…"
  },
  "task": {
    "prompt": "Implement /v2/users pagination per ticket-142",
    "context_files": ["src/api/v2/users.ts", "tests/api/v2/users.test.ts"],
    "lore_query_results": {"module": "users", "constraints": ["..."]}
  },
  "steps": [
    {
      "step": 1,
      "type": "thought",
      "content": "Need to understand the existing user-list endpoint before designing pagination"
    },
    {
      "step": 2,
      "type": "tool_call",
      "tool": "Read",
      "arguments": {"file": "src/api/v2/users.ts"},
      "result": "<truncated>",
      "result_hash": "sha256:def…"
    },
    {
      "step": 3,
      "type": "thought",
      "content": "Existing endpoint returns full list. Will add cursor-based pagination."
    },
    {
      "step": 4,
      "type": "tool_call",
      "tool": "Edit",
      "arguments": {"file": "src/api/v2/users.ts", "old_text": "...", "new_text": "..."},
      "result": "ok",
      "result_hash": "sha256:ghi…"
    }
    // ... many more steps ...
  ],
  "result": {
    "files_changed": ["src/api/v2/users.ts", "tests/api/v2/users.test.ts"],
    "commit_sha": "abc123",
    "test_outcome": "12/12 passing"
  },
  "metrics": {
    "tokens_input": 18420,
    "tokens_output": 22783,
    "duration_s": 187,
    "tools_invoked": 23
  }
}
```

This file — typically 50KB-5MB depending on session length — is the unit of replay.

## 3. The canonical 2026 tools

### 3.1 entireio/cli — the GitHub-native option

[`entireio/cli`](https://github.com/entireio/cli) (4.5k★, "Entire CLI hooks into your Git workflow to capture AI agent sessions as you work") is the dominant open-source tool. It:

- Wraps the agent's working directory
- Captures every tool call, file change, and reasoning step
- Stores the session as JSONL in a git-attached `.entire/` directory
- Links each session to the resulting commit
- Provides `entire log` (sessions) and `entire replay <session-id>`

Adoption: high. Many agent harnesses integrate Entire's hooks natively.

### 3.2 h5i — token-efficient capture

[`h5i-dev/h5i`](https://h5i.dev/) (391★) goes further: it captures the full session but stores tool output in a side-channel (file or git note), so the agent's own context window doesn't have to re-process the verbose tool output every turn.

The trade-off: h5i is more invasive (it wraps the agent's prompt loop directly) but yields "up to 95% token reduction" on long sessions.

### 3.3 Skar — capture-and-regression-test

[`kalisky/skar`](https://github.com/kalisky/skar) is the most rigorous: it converts a captured session into a pytest regression test. The agent's behavior on a given input is locked in as a test that will catch drift if you change the underlying model or prompt.

From the README:

> Skar turns a captured AI agent trace into a committed pytest regression test.

Skar is for teams writing the code that wraps an LLM into a tool-using agent. If you ship a custom agent (LangChain, LlamaIndex, Anthropic SDK direct, AutoGen, etc.), Skar lets you lock a specific run as a regression test.

Two packages: `@kalisky/skar` (npm, CLI + MCP server) and `skar` (PyPI, Python runtime recorder).

The MCP server exposes four tools to any agent: `capture_claude_code_session`, `generate_pytest_regression`, `validate_trace`, `inspect_trace`.

### 3.4 The .agent_radio/ convention

A lighter convention (no specific tool required): every agent commits its session log to `.agent_radio/sessions/<session-id>/` at the end of each turn. Any agent that can read files can replay by reading the JSONL.

### 3.5 Git notes as a side-channel

`git notes --ref agent-session` is the lightest possible capture. No extra files in the tree; just metadata attached to commits. Trade-off: harder to discover, easier to lose (notes don't push by default).

## 4. The replay workflow

Given a session log, how do you actually replay it?

### 4.1 Re-running an agent session

```bash
entire replay 2026-06-17-abc123 --model claude-opus-4.5 --temp 0
```

This re-invokes the agent with the same prompt and (ideally) the same context, but on a fresh working tree. Diff the new diff against the original:

```bash
git diff abc123..<new-commit-sha>
```

If the diff is non-empty, the agent is non-deterministic (model changed, prompt cached differently, etc.). If the diff is empty, the session is replayable.

### 4.2 Inspecting reasoning

```bash
entire reasoning 2026-06-17-abc123 --step 4
```

Prints the agent's `thought` block at step 4, including what it was thinking when it called the tool.

### 4.3 Inspecting context

```bash
entire context 2026-06-17-abc123 --step 4 --what model-saw
```

Reconstructs the exact prompt the model saw at step 4. Critical for debugging prompt-injection or context-overflow bugs.

### 4.4 Branching a session

```bash
entire branch 2026-06-17-abc123 --at-step 4 --new-task "fix the race instead"
```

Starts a new session that takes over from step 4 of the original, with a different continuation task. Useful for "what would have happened if the agent had chosen differently?"

## 5. The postmortem format for agent incidents

A 2026-style agent postmortem (the post-mortem commit + the postmortem doc):

### 5.1 The postmortem commit

```text
postmortem(api): incident 2026-06-17-1432 — v2/users pagination regression

Impact:
  4,231 requests failed between 14:32 and 14:51 UTC.
  Error rate spiked from 0.3% to 12%.
  Customer-facing: yes.
  Time-to-detect: 19 minutes.

Detection:
  Anomaly alert from service-mesh dashboard (5xx rate).
  Paged on-call at 14:51 UTC.

Root cause:
  Agent session 2026-06-17-abc123 introduced cursor-based pagination
  but used a `created_at` column that is not indexed. Under high QPS,
  the full table scan caused request timeouts.

Why the agent chose an unindexed column:
  Step 12 of session 2026-06-17-abc123:
    Thought: "Need an ordering field. Use created_at — it's intuitive."
    The agent did not check for indexes before choosing.
  Step 23 of session:
    Thought: "Schema check passed (no migration needed)."
    The agent did not verify the existing indexes against the new query plan.

Why the agent's choice was not caught:
  - Local tests passed (small dataset)
  - CI performance tests did not cover the new query path
  - The agent's PR did not include a database-explain output
  - Auto-merge on green CI merged without human review

Lore-id: postmortem-2026-06-17-1432
Constraint: new queries must include an EXPLAIN ANALYZE in the PR
Constraint: pagination cursors must reference indexed columns
Rejected: implicit ordering by created_at | use indexed column instead
Confidence: high
Reversibility: easy (one-line fix)
Refs: incident-2026-06-17-1432, ticket-301

Co-authored-by: Claude <noreply@anthropic.com>
Agent-DID: did:key:z6Mk…
Agent-Model: claude-opus-4.5
Reviewed-by: Hernanda <m.hernanda95@gmail.com>
```

### 5.2 The postmortem document

A separate `postmortems/2026-06-17-1432.md` with the full timeline, contributing factors, customer impact, and action items. The postmortem commit is the searchable index.

### 5.3 The action items

Action items become tickets. Each ticket becomes a new Lore constraint when completed:

```text
docs: enforce EXPLAIN ANALYZE in PR template

Adds a checkbox to the PR template: "For any new query path, attach
EXPLAIN ANALYZE output from production-like data."

Lore-id: a1b2c3d4
Constraint: new queries must include EXPLAIN ANALYZE
Refs: postmortem-2026-06-17-1432, ticket-301
```

The postmortem's lessons become future Lore constraints. The system learns from failure.

## 6. The observability stack (metrics to track)

The 2026 standard set of metrics every agentic-git system should export:

| Metric | Source | Why |
|---|---|---|
| Commits per day per agent | `git log --author` | Detect runaway loops |
| PRs opened per day per agent | `gh api` | Detect spam |
| PRs merged without review per agent | branch policy audit | Detect policy violations |
| Mean time from PR-open to PR-merge | GitHub API | Detect auto-merge abuse |
| Reverts per agent per week | `git log --grep='^Revert'` | Detect bad commits |
| Token usage per agent per day | agent runtime metrics | Cost control |
| Session replay capture rate | Entire / h5i | Coverage of observability |
| Lore atom coverage (files with Lore constraints / total files) | `lore stats` | Knowledge completeness |
| Test pass rate per agent | CI | Quality trend |
| Commit message lint pass rate | commit-msg hook | Compliance |

A simple Grafana dashboard on these metrics catches most regressions early.

## 7. The debugging cheatsheet

| Symptom | First place to look | Tool |
|---|---|---|
| Agent's commit broke prod | Replay the session, check reasoning at the offending step | `entire replay` |
| Agent is making worse commits than last week | Diff recent session logs to last week's | `entire diff` |
| Agent re-litigated a decision that was already settled | Check Lore for that module | `lore query` |
| Agent committed a secret | Session log shows it was in the prompt; remove from prompt | `entire reasoning` |
| Two agents edited the same file | `git log --all` + branch graph | `git worktree list` |
| Auto-merge fired on a bad PR | Audit the auto-merge policy | GitHub branch settings |
| Agent's commit message is uninformative | Check the agent's `commit-msg` hook | pre-commit framework |
| Agent's diff is too large | Check session log for runaway loop | `entire reasoning` |
| Production is fine but agent's tests are flaky | Check test isolation | `pytest --random-order` |
| Lore atom is out of date | Check `Last-reviewed` trailer | Lore query |

## 8. What good looks like in 2026

A mature 2026 agentic-git deployment has:

- [ ] 100% of agent commits have an attached session log
- [ ] Session logs are signed by the agent's DID
- [ ] Session logs push to a remote (git notes or `.agent_radio/`)
- [ ] Postmortems include the session log + Lore atom + commit
- [ ] Lore atoms are reviewed quarterly (decay check)
- [ ] Replayable sessions can re-run deterministically
- [ ] Each repo has at least one `entire replay` and one `lore query` in its CI
- [ ] Agent reputation is computed from session-log analysis
- [ ] A runbook exists for "agent broke prod" → replay → diagnose → postmortem → Lore constraint

If you have all nine, debugging an agent incident takes minutes, not hours.

## 9. The 18-month horizon for agent observability

Where the field is heading:

- **OpenTelemetry for agent traces.** The OTel working group is standardizing `gen_ai.*` spans. By Q4 2026 most agent frameworks will emit OTel natively.
- **Standard session-log schema.** A W3C-style standard for agent traces is being discussed. `entireio/cli` and `h5i` schemas will merge.
- **Built-in determinism.** Models like Claude 4.x and GPT-5 already support `temperature=0` determinism; replay tools will exploit this fully.
- **Visual replay.** Browser-based replay UIs (think Chrome DevTools for agents) will become standard. Already shipped in some products.
- **Cost-aware replay.** "Replay only the steps that mattered for the bug" rather than re-running the whole session.

## 10. See also

- Cat 12 — *AI Agent Architecture* (state machines, loops, observability patterns)
- Cat 19 — *Observability & Monitoring* (broader observability stacks)
- **03-Technical-Deep-Dive** — agent identity and commit signing
- **07-Prompt-and-Commit-Patterns** — commit message structure that helps debugging
- **08-Agent-Memory-and-Long-Running-Workflows** — episodic memory for replay
