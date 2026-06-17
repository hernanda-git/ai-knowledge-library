# 01 — Agentic Git: Overview

> A category covering the emerging discipline of **version control for, by, and between AI agents**. This overview defines the term, situates it in 2026, names the major actors, and previews the rest of the category.

## 1. What is "Agentic Git"?

**Agentic Git** is the practice, infrastructure, and tooling that lets AI coding agents — and the humans who supervise them — read, write, branch, merge, review, sign, replicate, and govern Git repositories with the same autonomy, accountability, and reproducibility that humans currently have. It is *not* a fork of Git and *not* a single product; it is a stack of conventions, plugins, identity layers, and policy engines layered on top of vanilla Git so that an agent (LLM-driven CLI, cloud sandbox, or swarm) can be a first-class citizen in a repository's lifecycle.

Two threads are converging under the term:

1. **Agent-operated Git** — agents clone, commit, push, open PRs, and merge on behalf of humans. The dominant pattern in 2026: every serious coding agent (Claude Code, OpenCode, Cursor Agent, GitHub Copilot Coding Agent, Cursor Cloud Agents) now has git plumbing baked in. Commits authored by an LLM are now a routine fact of life on GitHub.
2. **Agent-native Git infrastructure** — git hosts, remotes, and identity layers designed *for* agents rather than adapted from human workflows. Decentralized git nodes (e.g. **Gitlawb**), on-chain bounty/escrow systems, DID-based commit signing, and libp2p-replicated repositories fall here.

The first thread is the demand side; the second is the supply side. Together they are what makes the space legible.

## 2. Why this category exists (2026 inflection)

Three signals make now the right time to document the space:

- **Every AI coding tool ships git integration as a first-class feature.** Cursor has Cloud Agents that open PRs from sandboxes ([HN: "Cursor Cloud Agents Down"](https://hn.algolia.com/?q=cursor+cloud+agents), 21 pts). Claude Code, Codex, OpenCode, and `aider` all do `git commit` / `git push` natively. The repository is now the agent's native API.
- **A new infra layer is emerging to serve agentic commits.** Gitlawb (the "decentralized git for AI agents" project) launched its opencode plugin (77★), self-hostable Rust node (34★), and on-chain bounty contracts in 2025-2026. It uses Ed25519 `did:key` identities instead of GitHub usernames, RFC 9421 HTTP signatures instead of PATs, and exposes identity/repo/PR/bounty/agent flows via MCP to any agent.
- **Bounty/escrow markets for agent work exist now.** Gitlawb's bounty workflow lets a human post a task with tokens escrowed on-chain, an agent claims it, ships a PR, and gets paid on merge. Similar patterns are appearing in ccpm (`automazeio/ccpm`, 8.2k★) and Backlog.md (`MrLesk/Backlog.md`, 5.8k★). Code is becoming a market commodity settled on git-native rails.

The previous library coverage (cat 20 — *Agent Infrastructure & Observability*; cat 24 — *Agent Autonomy & Accountability*) touched the agent-as-committer problem tangentially. Cat 28 zooms in.

## 3. Scope and boundaries

| In scope (this category) | Out of scope (handled elsewhere) |
|---|---|
| Agent ↔ git plumbing: clones, commits, PRs, reviews, merges | Agent framework internals (cat 06) |
| Agent identity, signing, DID, on-chain reputation | Generic identity wallets (cat 04) |
| Decentralized / agent-native git infrastructure | Generic P2P / libp2p theory (cat 19) |
| Bounty markets, agent-to-agent contracting, code-as-asset | DAO legal structures (cat 27) |
| Repo orchestration: worktrees, swarms, parallel branches | Plain GitLab/GitHub UI patterns (legacy) |
| Commit policy, branch protection, agent guardrails | Pure CI/CD without agent context (cat 13) |
| Open-source agentic-git projects (OpenCode, Gitlawb, ccpm, Backlog.md, …) | Commercial features that don't expose git semantics |
| Cursor Agent / Cursor Cloud Agents workflows | Pure code-completion UX (covered elsewhere) |

## 4. The actor map

The 2026 agentic-git ecosystem has four distinct actor types. Each plays a different role:

### 4.1 The coding agents (the committers)

- **Claude Code** (Anthropic) — terminal-resident agent. 132.9k★ on GitHub as of mid-2026. Native git workflow: stage, commit, push, branch, PR via `gh` CLI.
- **Cursor Cloud Agents** — Cursor's cloud sandbox agents that take a Jira/Linear ticket and ship a PR ([cursor.com/agents](https://cursor.com/agents)). 37-pt HN thread on secret management, 21-pt thread on outages.
- **OpenCode** (opencode.ai) — terminal + TUI coding agent. ~16k★ ecosystem. Plugin surface for git tools (the Gitlawb plugin hooks here).
- **OpenClaude** (Gitlawb/openclaude, 29k★) — OpenCode-fork under the Gitlawb umbrella; ships with native agent identity.
- **Codex CLI** (OpenAI) — terminal agent. Recently added worktree-aware commit flows.
- **GitHub Copilot Coding Agent** — cloud agent that opens PRs from issues.
- **Aider, Continue.dev, Cody, Codestral** — smaller but git-native.

### 4.2 The git hosts and remotes (the substrate)

- **GitHub** — default. Adds `co-authored-by: Copilot` trailers, signed commits via GitHub App, PR review agents.
- **GitLab** — Duo agents with merge-request semantics.
- **Gitea / Forgejo** — self-hosted. Popular for on-prem agent deployments.
- **Gitlawb Node** (Gitlawb/node, 34★) — Rust + libp2p + Ed25519. Self-hostable. Replicates git activity across peers. The first credible attempt at a *decentralized GitHub for agents*.
- **opengap** (`open-gitagent/opengap`, 2.8k★) — "framework-agnostic, git-native standard for defining AI agents." Competes in the spec layer.

### 4.3 The orchestration layer (the conductors)

- **worktrunk** (`max-sixty/worktrunk`, 5.5k★) — CLI for git worktree management *designed for parallel AI agent workflows*. Each agent gets its own worktree; humans review the resulting PRs.
- **ccpm** (`automazeio/ccpm`, 8.2k★) — "Project management skill system for Agents that uses GitHub Issues and Git worktrees." Ships with Claude Code skills.
- **agentvm** (Gitlawb/agentvm, 22★) — "Pure-bash CLI for running multiple AI coding agents in parallel, each in its own [worktree]."
- **Backlog.md** (`MrLesk/Backlog.md`, 5.8k★) — "managing project collaboration between humans and AI Agents in a git repo." Single binary, lives in your repo.
- **stagewise** (`stagewise-io/stagewise`, 6.7k★) — "The Open Source Agentic IDE. Create and orchestrate coding agents."

### 4.4 The market layer (the economy)

- **Gitlawb Bounties** — on-chain token escrow (Base) for tasks posted against a repo. Agents claim, ship a PR, get paid on merge via `gitlawb_bounty_submit`.
- **Bankr Skills** (Gitlawb/banker-skills, 16★) — plug-and-play tools for building agents that trade on-chain.
- **GitNexus** (`abhigyanpatwari/GitNexus`, 42k★) — client-side code-intel graph that agents query instead of grepping.
- **sem** (`Ataraxy-Labs/sem`, 2.9k★) — "Semantic version control → entity-level diffs, blame, and impact analysis on top of git."
- **mcp_agent_mail** (`Dicklesworthstone/mcp_agent_mail`, 2k★) — async coordination layer for AI agents: identities, inboxes, searchable threads. Think "email for agents," MCP-served.

## 5. The unique affordances of agentic git

Compared to plain git-for-humans, agentic git adds five affordances:

| Affordance | Human git | Agentic git |
|---|---|---|
| **Identity** | GitHub username + email | Ed25519 `did:key`, UCAN tokens, agent attestations |
| **Authorization** | PAT, OAuth scopes, branch protection | RFC 9421 HTTP signatures + capability tokens + policy DSL |
| **Commit signing** | GPG/SSH optional | Required by default (agent ≠ author, agent = signer-of-record) |
| **Replication** | Single remote | libp2p gossip across peers (Gitlawb model) |
| **Settlement** | No native payment | On-chain escrow + bounties (Gitlawb model) |

## 6. The cross-cutting risks

Agentic git makes three things worse that were already bad with humans:

- **Velocity of bad commits.** A loop-bug in an agent can produce 10,000 garbage commits in an hour. Branch protection is no longer defense enough — repos need commit-content scanning and rate limiting *per identity*.
- **Attribution collapse.** `Co-authored-by: Cursor` trailers are already ubiquitous; in 6 months the *meaningful* attribution has migrated to PR-review commentary, which is much sparser. Legal accountability (cat 24) and on-chain reputation (cat 27) both suffer.
- **Supply-chain injection at the agent layer.** A poisoned tool definition in a plugin (e.g. an MCP tool that rewrites commit messages to embed a backdoor) now has access to write to your `main`. The attack surface is the *agent's toolchain*, not just the repo's dependencies.

## 7. How to read this category

- **02-Core-Topics** drills into the seven technical primitives: identity, signing, worktrees, PR review, bounties, replication, policy.
- **03-Technical-Deep-Dive** has full working code: a from-scratch agent identity CLI, an Ed25519-signed commit wrapper, a `git push`-style worktree fork for parallel agents, and an MCP server exposing Gitlawb to Claude Code.
- **04-Tools-and-Frameworks** is the 2026 inventory (47 projects, sorted by use case) plus a 30-day adoption plan.
- **05-Future-Outlook** covers the 2026-2030 trajectory: the death of the GitHub-star economy, the rise of agent-native code markets, regulation, and three possible ends-state scenarios.

## 8. See also

- Cat 04 — *Agent Identity, Wallets & Authentication* (DID, UCAN, signed-agent flows)
- Cat 06 — *Agent Frameworks & Orchestration* (LangGraph, CrewAI, OpenCode)
- Cat 13 — *AI Engineering & DevOps* (CI/CD for agents, GitHub Actions, runners)
- Cat 20 — *Agent Infrastructure & Observability* (tracing, eval, cost — where cat 28 ends)
- Cat 24 — *Agent Autonomy, Accountability & Operator Liability* (who's on the hook for an agent's bad merge)
- Cat 27 — *AI Agent Legal Entities & DAO Governance* (the legal half of agentic-git reputation)

---

*Last reviewed: 2026-06-17. Coverage horizon: through Q2 2026. Sources: GitHub API search, HN Algolia, primary repo READMEs, Cursor changelog, Gitlawb docs.*
