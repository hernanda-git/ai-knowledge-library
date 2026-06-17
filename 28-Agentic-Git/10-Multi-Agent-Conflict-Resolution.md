# 10 — Agentic Git: Multi-Agent Conflict Resolution

> When multiple agents work the same repo in parallel, conflicts are inevitable. This chapter covers the strategies, tools, and emerging standards for keeping agents from stepping on each other.

## 1. The conflict landscape

When N agents work on N tasks in a repo, three classes of conflict appear:

| Class | When | Symptom |
|---|---|---|
| **Logical conflict** | Two agents edit the same function with incompatible changes | Merge conflict at `git merge` |
| **Semantic conflict** | Two agents edit different files but the changes break each other | Tests pass for each PR in isolation, fail when both merge |
| **Resource conflict** | Two agents try to edit the same generated file (package-lock.json, schema.prisma) | Last writer wins, the other agent's work is lost |

Vanilla git handles class 1 with merge tools. Classes 2 and 3 are unique to multi-agent workflows and need explicit strategy.

## 2. The worktree model (the default 2026 pattern)

The worktree pattern (already covered in chapter 02 §3) is the default for a reason: it provides **physical isolation** between agents.

```text
repo/.git/
repo/main/                ← branch: main
repo/wt-agent-a/          ← branch: agent-a/feature-1
repo/wt-agent-b/          ← branch: agent-b/feature-2
repo/wt-agent-c/          ← branch: agent-c/bug-fix
```

Each agent `cd`s into its own worktree, never sees the others' uncommitted changes. Merges happen via PR review.

What worktrees don't solve:

- Two agents touching the same file at different lines (no merge conflict, but the second agent's commit may undo the first agent's intent)
- Two agents adding the same dependency at different versions (semantic conflict)
- Two agents editing the same database migration (resource conflict)
- One agent's tests being run against another agent's changes (CI runs each in isolation; integrated tests run later)

## 3. The clash-sh/clash pattern (early detection)

[`clash-sh/clash`](https://github.com/clash-sh/clash) (62★, Rust) is purpose-built for this problem. From the README:

> Avoid merge conflicts across git worktrees for parallel AI coding agents.

The thesis: agents running in separate worktrees are **blind to each other's changes** and inevitably touch overlapping parts of the codebase. Conflicts only surface at feature completion. Clash surfaces them earlier.

### 3.1 How clash works

```bash
# In worktree wt-agent-a
$ clash check
⚠ Potential conflict with wt-agent-b:
  - src/api/users.ts: agent-a modifies lines 42-58 (new endpoint)
  - src/api/users.ts: agent-b modifies lines 51-73 (rate-limit middleware)
  → Overlap: lines 51-58 (7 lines)

# Recommended actions:
$ clash suggest
1. Wait for agent-b to merge first, then rebase
2. Have agent-b take the rate-limit only; you take the endpoint only
3. Coordinate manually
```

Clash runs on every commit in the worktree and compares the cumulative diff against every other worktree's diff. When overlap is detected, it surfaces it.

### 3.2 Modes

- **`clash check`** — one-shot analysis
- **`clash watch`** — continuous (re-checks on every file save)
- **`clash status`** — high-level summary across all worktrees
- **`clash --json`** — machine-readable for agent consumption

### 3.3 Limits

Clash is line-based; it doesn't catch semantic conflicts (two agents adding the same import). For that, you need type-check + integration tests run against the combined state.

## 4. The role-based-locking pattern

[`Mybono/ai-orchestrator`](https://github.com/Mybono/ai-orchestrator) (97★) implements **role-based locking** for agents working in parallel:

```text
Planner A   → owns: tickets/[1-100].md
Planner B   → owns: tickets/[101-200].md
Coder A     → owns: src/api/**, tests/api/**
Coder B     → owns: src/db/**, tests/db/**
Reviewer    → owns: PRs/* (read-only)
```

A central file `.agent_locks.json` records which agent owns which paths. Before any commit, the agent checks it has the lock:

```python
def can_edit(agent_role: str, path: str) -> bool:
    locks = load_locks()
    for pattern, owner in locks.items():
        if fnmatch(path, pattern) and owner != agent_role:
            return False
    return True

# Agent's commit hook
if not can_edit("coder-a", filepath):
    raise PermissionError(f"coder-a cannot edit {filepath}; owned by coder-b")
```

This is overkill for small teams but mandatory when you have 6+ agents working the same monorepo.

## 5. The optimistic concurrency pattern

For agents that can't pre-coordinate, the pattern is:

1. Each agent works in its own worktree.
2. Before pushing, the agent rebases onto the latest `main`.
3. If the rebase conflicts, the agent resolves (LLM-assisted) and pushes.
4. CI runs on the merged result; if CI fails, the agent (or a downstream process) re-attempts.

```bash
# Agent's pre-push hook
git fetch origin main
git rebase origin/main
if [ $? -ne 0 ]; then
    echo "Rebase conflict detected; attempting LLM-assisted resolution..."
    python3 ~/bin/rebase-resolver.py
fi
git push origin HEAD
```

Where `rebase-resolver.py` is a tool that:

- Parses the conflict markers
- Asks the LLM to resolve based on the original task description
- Verifies the resolution by re-running tests
- Falls back to "request human review" if confidence is low

## 6. The semantic-conflict problem

Two agents editing different files can still produce a broken merge. Example:

```text
Agent A: changes function signature foo(x: int) -> str to foo(x: int, y: bool) -> str
         updates all call sites in src/api/** and tests/api/**
Agent B: adds a new call site foo(42) in src/cli/main.ts
```

Both PRs pass CI in isolation. After merge, the combined `foo(42)` call lacks the new `y` parameter and crashes.

Detection requires:

1. **Type-check the merged tree.** Most languages have a fast type-checker (`tsc --noEmit`, `mypy`, `cargo check`).
2. **Run cross-module tests.** Integration tests, end-to-end tests.
3. **Static analysis.** `semgrep`, `codeql`, or custom rules that look for cross-file invariants.

A 2026 best practice: every PR's CI includes a "merge preview" build:

```yaml
# GitHub Actions: merge-preview job
merge-preview:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
    - name: Merge preview
      run: |
        git fetch origin main
        git checkout -b merge-preview origin/main
        git merge --no-ff --no-edit ${{ github.head_ref }}
    - name: Type check + tests on merged tree
      run: pnpm typecheck && pnpm test:integration
```

If this fails, the PR can't merge — even if it would pass in isolation.

## 7. The shared-state pattern (for generated files)

Generated files (`package-lock.json`, `schema.prisma`, `openapi.yaml`, generated client code) are touched by every agent and merge-conflict constantly.

The 2026 best practice: **never edit generated files directly; always regenerate.**

```bash
# .gitattributes
package-lock.json merge=union
schema.prisma lockable=true

# Agent's pre-commit hook
if [[ "$files_changed" == *"package-lock.json"* ]]; then
    pnpm install --lockfile-only
    git add package-lock.json
fi
```

The `lockable` Git attribute (available in GitLab, in progress for GitHub) tells git: "this file is auto-generated; don't try to three-way merge it; whoever pushes last wins."

`merge=union` (for files where union is safe, like `package-lock.json`) keeps all package versions from both sides.

For files where neither is acceptable (schema.prisma, openapi.yaml), use a custom merge driver:

```bash
# .gitattributes
schema.prisma merge=schema-merge

# .git/config
[merge "schema-merge"]
    name = schema-aware merge for prisma
    driver = python3 ~/bin/prisma-schema-merge %A %O %B %L %R
```

The custom driver does a structured merge based on Prisma's schema grammar.

## 8. The optimistic-then-confirm pattern for parallel feature work

When two agents are working on related features (say, both adding endpoints to the same controller), the safer pattern is:

```text
T+0   Agent A creates branch feat-a, opens PR #1
T+0   Agent B creates branch feat-b, opens PR #2
T+1h  Both PRs have CI green
T+1h  Agent A's merge-preview job runs:
      - Fetches feat-b
      - Merges feat-b into feat-a
      - Runs full test suite
      - Reports conflicts back to agent A
T+2h  Agent A resolves, re-pushes
T+3h  Agent A's PR merges to main
T+3h  Agent B's merge-preview job now re-runs against main + feat-a
      - Reports new conflicts to agent B
T+4h  Agent B resolves, re-pushes, merges
```

The merge-preview job is the gate. Both agents see each other's changes before either lands.

## 9. The "merge day" pattern

For large repos with high agent throughput, some teams adopt a **merge-day** discipline:

- Agents work freely Mon-Thu, accumulating PRs
- Friday is merge day: a senior engineer (or a senior agent with extra permissions) merges all approved PRs in dependency order
- Conflicts between PRs are resolved by the merge-day owner

This trades time-to-merge for stability. Common at large OSS projects with high PR volume.

## 10. Conflict-resolution priority

When a conflict is detected, who decides how to resolve it?

Default priority (2026 consensus):

1. **Lore constraints win.** If a Lore atom says "do not cache session objects," and the conflicting change adds caching, the change loses.
2. **The agent whose work was there first wins.** If Agent A's commit is on `main` and Agent B's PR conflicts, Agent B must adapt.
3. **The change that touches fewer files wins.** Surgical changes beat sweeping rewrites.
4. **The agent with higher reputation wins.** A new agent yields to an established agent with a track record.
5. **Human review.** If none of the above resolves the conflict, page a human.

These priorities can be encoded in the conflict-resolver tool:

```python
def resolve_conflict(ours: str, theirs: str, context: dict) -> str:
    if violates_lore(theirs, context["lore_atoms"]):
        return ours  # theirs violates a constraint
    if not_on_main(context["ours_commit"]):
        return theirs  # ours isn't on main yet
    if ours_touches_more_files(context):
        return theirs  # theirs is more surgical
    if reputation(context["their_agent"]) > reputation(context["our_agent"]):
        return theirs
    raise HumanReviewRequired(...)
```

## 11. Tools inventory

| Tool | Mechanism | Notes |
|---|---|---|
| `worktrunk` | Worktree management | The default in 2026 |
| `ccpm` | GitHub Issues + worktree flow | Adds project management layer |
| `agentvm` | Pure-bash parallel agent runner | Minimal deps |
| `Backlog.md` | In-repo collaboration | Lives in the repo |
| `clash-sh/clash` | Conflict early-warning | 62★, niche but precise |
| `Mybono/ai-orchestrator` | Role-based locking | 97★, good for monorepos |
| `h5i` | Multi-agent messaging | Reduces redundant work |
| `h5i Agent Radio` | Cross-agent commit-mediated chat | New pattern |
| `entireio/cli` | Session capture for postmortem | 4.5k★ |
| `worktrunk merge --into main <branch>` | Squashing/merging PRs | Vanilla worktrunk |
| `gh pr merge --auto` | Auto-merge after CI | GitHub native |
| GitLab Merge Trains | Queued merges with conflict resolution | GitLab native |

## 12. The anti-patterns to avoid

### 12.1 The shared-worktree anti-pattern

Multiple agents writing to the same worktree. Almost always loses work; never do this.

### 12.2 The "force push" anti-pattern

An agent's solution to a rebase conflict is `git push --force`. This destroys other agents' work. Block via branch policy:

```yaml
branch_protection:
  main:
    block_force_push: true
  agent_a/feature_1:
    block_force_push: false  # OK on personal branches
```

### 12.3 The "long-lived branch" anti-pattern

An agent works for 5 days on a single branch without rebasing. By day 5, the merge conflict with main is unresolvable. Enforce:

- Daily rebase on `main` (or weekly, minimum)
- Branch max age of 7 days (auto-close stale PRs)
- Daily merge-preview builds

### 12.4 The "no merge preview" anti-pattern

Two agents' PRs both pass CI in isolation. They merge in opposite order and break. The fix: always run merge-preview, always require it green before merging.

### 12.5 The "no role separation" anti-pattern

Two agents writing to `prisma/schema.prisma` at the same time. Whoever pushes last wins. The fix: assign `schema.prisma` to one owner (human or senior agent) and route all schema changes through them.

## 13. The 30-day adoption plan for conflict-free multi-agent workflows

### Week 1
- Day 1: Audit current repo for generated files (`package-lock.json`, `*.generated.*`, schema files). Add `.gitattributes` for each.
- Day 2: Document a role assignment file (`.agent_locks.json`) listing who owns which paths.
- Day 3: Add `clash check` to the agent's pre-commit hook (advisory mode).
- Day 4: Add `git fetch && git rebase origin/main` to the agent's pre-push hook.
- Day 5: Add a "merge-preview" GitHub Action that runs on every PR.

### Week 2
- Day 6: Set up `clash watch` for one project to see the actual conflict pattern.
- Day 7: Write a postmortem on one real conflict your team hit last month. Apply chapter 09's format.
- Day 8: Implement Lore for one module that has the most conflicts.
- Day 9: Configure branch protection: require merge-preview to be green before merge.
- Day 10: Add the conflict-resolution priority logic (section 10) to your resolver tool.

### Week 3
- Day 11: Set up role-based locks for at least 3 hot-spot paths.
- Day 12: Add a custom merge driver for `prisma/schema.prisma` (or your equivalent).
- Day 13: Train each agent on AGENTS.md including the conflict-resolution rules.
- Day 14: Run a 2-agent simulation on a sample repo; measure conflict rate.

### Week 4
- Day 15: Refine based on what broke.
- Day 16: Add observability: conflict count per agent, time-to-resolve per conflict.
- Day 17: Write a "merge day" runbook if your PR volume justifies it.
- Day 18: Add reputation scoring (chapter 05 cross-ref) to the conflict priority logic.
- Day 19: Document the team's full agent-conflict policy in CONTRIBUTING.md.
- Day 20: Celebrate.

## 14. See also

- **02-Core-Topics §3** — worktrees as agent sandboxes
- **03-Technical-Deep-Dive §3** — worktree orchestrator implementation
- **06-Security-and-Supply-Chain** — when conflicts hide supply-chain attacks
- **08-Agent-Memory-and-Long-Running-Workflows** — handoff commits and state files
- Cat 13 — *AI Engineering & DevOps* — CI/CD and merge strategies
