# 04 — Agentic Git: Tools and Frameworks

> The 2026 inventory. Forty-seven active projects, grouped by role, ranked by adoption signal (stars, HN mentions, npm/pip downloads). Plus a 30-day adoption plan and a curated "what to learn first" reading list.

## 1. The 2026 landscape at a glance

```text
                     ┌──────────────────────────────────────┐
                     │            AGENTIC GIT STACK         │
                     └──────────────────────────────────────┘
                                       │
   ┌───────────────┬───────────────────┼────────────────────┬──────────────────┐
   │               │                   │                    │                  │
┌──▼──────┐   ┌────▼─────┐       ┌──────▼─────┐       ┌──────▼────┐      ┌──────▼────┐
│ Coding  │   │  Git     │       │ Orchestr-  │       │ Identity  │      │  Markets  │
│ Agents  │   │  Hosts   │       │ ation      │       │ & Signing │      │ & Pay     │
└─────────┘   └──────────┘       └────────────┘       └───────────┘      └───────────┘
   Claude       GitHub              worktrunk            Ed25519/DID       Gitlawb
   Cursor       GitLab              ccpm                 Sigstore          bounties
   OpenCode     Gitlawb             agentvm              UCAN              Bankr
   OpenClaude   Gitea               Backlog.md           GitNexus          Polymarket
   Codex CLI    opengap             stagewise            Gitlawb node      MCP Mail
```

## 2. Coding agents (the committers)

These are the agents that actually push commits. Star counts and HN signal from mid-2026.

| Project | Stars | Type | Git-native? | Notes |
|---|---|---|---|---|
| **Claude Code** (`anthropics/claude-code`) | 132.9k | CLI | ✅ Built-in `git commit/push/PR` | Reference implementation. Drives most of the "agent commits" pattern in 2026. |
| **Cursor Cloud Agents** | — | Cloud sandbox | ✅ PR-from-issue | Cursor's hosted runner. 37-pt HN thread on secrets, 21-pt on outages. |
| **OpenCode** (`sst/opencode`) | ~16k | CLI + TUI | ✅ Plugin model | The substrate Gitlawb hooks into. |
| **OpenClaude** (`Gitlawb/openclaude`) | 29k | CLI (fork) | ✅ Built-in DID identity | OpenCode fork with first-class agent identity. |
| **Codex CLI** (OpenAI) | — | CLI | ✅ Worktree-aware | Terminal agent, GPT-5 family. |
| **GitHub Copilot Coding Agent** | — | Cloud | ✅ PR from issue | Lives in GitHub Issues. |
| **Aider** | ~35k | CLI | ✅ Built-in | Long-time favorite; structured commit messages. |
| **Continue.dev** | ~25k | VS Code/JetBrains | ✅ Via commands | Open-source, model-agnostic. |
| **Cody** (Sourcegraph) | — | Editor | ✅ Via commands | Codebase-aware. |
| **Cline / Roo Code** | ~30k | VS Code | ✅ Commit-PR cycle | Direct PR creation. |
| **stagewise** (`stagewise-io/stagewise`) | 6.7k | IDE | ✅ Built-in | Orchestrates other agents. |

**What to learn first:** Claude Code's `commit` / `PR` flow is the most documented. Master that, then layer Cursor Cloud Agents for asynchronous work.

## 3. Git hosts and remotes (the substrate)

Where the commits live.

| Project | Type | Identity model | Agent-aware | Notes |
|---|---|---|---|---|
| **GitHub** | SaaS | Username + email | `Co-authored-by:` trailers, GitHub App signing | Default. PR review agents standard. |
| **GitLab** | SaaS / self-hosted | Username + email | Duo agents, MR semantics | Strong enterprise story. |
| **Gitea / Forgejo** | Self-hosted | Username + email | Bot accounts | The on-prem pick for agent deployments. |
| **Gitlawb Node** (`Gitlawb/node`) | Decentralized (Rust + libp2p) | Ed25519 `did:key` | ✅ First-class | The first credible *agent-native* git host. 34★ but rapidly growing. |
| **opengap** (`open-gitagent/opengap`) | Standardization effort | DID-based | ✅ Spec layer | "Framework-agnostic, git-native standard for defining AI agents." 2.8k★. |
| **entireio/cli** (`entireio/cli`) | CLI wrapper | — | ✅ Captures agent sessions | "Hooks into your Git workflow to capture AI agent sessions as you work." 4.5k★. |
| **sem** (`Ataraxy-Labs/sem`) | Git layer | — | ✅ Semantic diffs | "Entity-level diffs, blame, and impact analysis on top of git." 2.9k★. |

**What to learn first:** Get fluent in vanilla git's worktree, notes, and refs layers. Every agentic-git tool is built on top.

## 4. Orchestration (the conductors)

The glue that runs N agents on N branches.

| Project | Stars | What it does | Why you'd use it |
|---|---|---|---|
| **worktrunk** (`max-sixty/worktrunk`) | 5.5k | Git worktree CLI for parallel AI workflows | Smallest learning curve. Each agent = one worktree. |
| **ccpm** (`automazeio/ccpm`) | 8.2k | Project management skill system using GitHub Issues + worktrees | Ships Claude Code skills for the full backlog→PR loop. |
| **agentvm** (`Gitlawb/agentvm`) | 22 | Pure-bash CLI for N parallel agents | Drop-in for shell scripts; minimal deps. |
| **Backlog.md** (`MrLesk/Backlog.md`) | 5.8k | Manages human+agent collaboration in a git repo | Single binary; lives inside your repo. |
| **stagewise** (`stagewise-io/stagewise`) | 6.7k | Agentic IDE; orchestrates coding agents | Browser preview + agent control plane. |
| **mcp_agent_mail** (`Dicklesworthstone/mcp_agent_mail`) | 2.0k | Async identity/inbox layer for agents | Solves "agent A needs to message agent B without sharing memory." |

**What to learn first:** `worktrunk` — install, run `wt switch -c my-feature`, push a branch, open a PR. The concept is identical for all the rest.

## 5. Identity, signing, and reputation

| Project | Type | Crypto | Used by |
|---|---|---|---|
| **Gitlawb identity** (`gl identity new`) | CLI | Ed25519 + `did:key` | Gitlawb ecosystem |
| **Sigstore / Fulcio / Rekor** | Service | OIDC + transparency log | sigstore.dev ecosystem |
| **UCAN** (`did:ucan/spec`) | Token format | Capability chaining | Any project doing delegated authority |
| **GitNexus** (`abhigyanpatwari/GitNexus`) | Code-intel graph | — | Agent queries, not grep |
| **Sem** | Semantic VCS | — | Entity-level diffs |
| **did:key / did:web resolvers** | Library | Multi | Universal |

**What to learn first:** The W3C DID spec (skim the primer). Then read UCAN's 10-minute intro. You'll see the pattern behind every identity tool.

## 6. Markets and code-as-asset

Where agentic commits become economic events.

| Project | Settlement | What it sells | Status |
|---|---|---|---|
| **Gitlawb Bounties** | Base (Ethereum L2) | Token-escrowed tasks against a repo | Live (early) |
| **Gitlawb contracts** (`Gitlawb/contracts`) | Base | On-chain protocol contracts | Live, audited pending |
| **Bankr Skills** (`Gitlawb/banker-skills`) | — | Plug-and-play agent trading skills | Live |
| **opengap** | — | Standard for agent identity & comms | Spec layer |

The pattern is identical to Upwork's, just git-native: post task → escrow funds → agent claims → ships PR → merge → payout.

**What to learn first:** Read Gitlawb's `gl bounty create` flow once end-to-end. The economic primitive is simple; the agent integration is the hard part.

## 7. The MCP integration layer

MCP (Model Context Protocol) is how an agent *talks* to a git tool. The 2026 catalog of MCP servers that touch git:

| MCP server | Git op | Best for |
|---|---|---|
| `@modelcontextprotocol/server-github` | Full GitHub API | Anything GitHub |
| `@modelcontextprotocol/server-gitlab` | GitLab API | Anything GitLab |
| `@gitlawb/opencode` (npm) | Gitlawb identity/repo/PR/bounty/agent | Agent-native flow |
| Custom (see §4 of chapter 03) | Worktree + signed commit | Minimal surface |

Every serious agent in 2026 — Claude Code, Cursor, OpenCode, OpenClaude, Codex CLI — speaks MCP. If you're building agentic-git tooling, ship an MCP server.

## 8. Comparison: top 5 picks by use case

| Use case | Pick #1 | Pick #2 | Pick #3 |
|---|---|---|---|
| **I want my AI assistant to commit code today** | Claude Code | Cursor Cloud Agents | Codex CLI |
| **I want to run 4 agents in parallel on my repo** | worktrunk | agentvm | stagewise |
| **I want agents with portable cryptographic identity** | Gitlawb Node | Sigstore (via Cosign) | Custom Ed25519 |
| **I want agents to earn money shipping code** | Gitlawb Bounties | (nothing else live yet) | — |
| **I want a PR-review agent that learns my codebase** | Cursor Bugbot | Claude Code Review | CodeRabbit |
| **I want a decentralized git host that won't disappear** | Gitlawb Node | opengap | (run your own Gitea) |
| **I want to capture every agent session for replay** | entireio/cli | Backlog.md | Manual `git reflog` |

## 9. The 30-day adoption plan

A pragmatic ramp for a single developer or a small team.

### Week 1 — Foundations
- Day 1: Read 01-Overview and 02-Core-Topics end-to-end.
- Day 2: Pick a coding agent (Claude Code recommended). Run a single non-trivial task end-to-end. Watch every `git` invocation.
- Day 3: Read `git worktree` docs. Run `git worktree add ../wt-test -b test/wt-1`. Make a commit on it.
- Day 4: Install `worktrunk` (`cargo install worktrunk` or `brew install worktrunk`). Run `wt switch -c demo`. Compare UX to raw git.
- Day 5: Set up a personal GitHub repo with branch protection on `main`. Try to push a non-signed commit. Then enable required signed commits.
- Day 6: Sign your own commits with an SSH key (`git config commit.gpgsign true`). Confirm GitHub shows the "Verified" badge.
- Day 7: Review. Where did the friction come from? What did the agent skip that you wouldn't?

### Week 2 — Identity and signing
- Day 8: Generate an Ed25519 keypair with `agentid new <label>` (chapter 03). Save the DID somewhere safe.
- Day 9: Wrap one of your repos with `gitlawb-commit` (chapter 03). Confirm `git notes show HEAD` returns your attestation.
- Day 10: Install the Sigstore `cosign` CLI. Sign a container image OR a git commit with OIDC.
- Day 11: Read the UCAN spec primer. Understand the difference between identity (DID), authorization (capability), and attestation (signature).
- Day 12: Pick a small repo and post a fake bounty via Gitlawb CLI. (`gl bounty create --repo <repo> --amount 5 --token USDC`). Watch it appear on the node.
- Day 13: Clone the `Gitlawb/opencode-gitlawb` repo and read the README again with fresh eyes. Try to identify three gaps in your own setup that the plugin would solve.
- Day 14: Review. Decide: are you going to commit on a public DID-signed basis going forward?

### Week 3 — Orchestration
- Day 15: Run `agent-orchestrate` from chapter 03 on a real backlog of three tasks. Use whatever agent you have.
- Day 16: Tweak the orchestrator. Add automatic PR creation. Add a "merge only if CI passes" gate.
- Day 17: Install `Backlog.md` (`npm i -g backlog.md`) in a repo. Try the `backlog task create` + `backlog task start` flow.
- Day 18: Compare `Backlog.md` to `ccpm`. Pick whichever matches your repo's existing conventions.
- Day 19: Set up `entireio/cli`. Run a coding session. Replay the captured log. See what the agent actually did vs. what it claimed.
- Day 20: Add a CI step that runs your favorite linter + tests on every agent PR. Gate `main` on green.
- Day 21: Review. Is the throughput worth the orchestration overhead? Most teams overestimate this.

### Week 4 — Markets and next steps
- Day 22: Read the `Gitlawb/contracts` repo. Understand what's actually on-chain vs. off-chain.
- Day 23: Try `opengap` — read the spec, run a minimal demo.
- Day 24: Set up a self-hosted Gitlawb Node (Docker). Confirm you can push to it.
- Day 25: Write an MCP server that exposes your favorite three git ops to any agent. Use chapter 03's `git-mcp` as a starting point.
- Day 26: Read Cursor's changelog (cursor.com/changelog). Find one feature that materially changed how an agent commits code in your stack.
- Day 27: Build a `git notes --ref attestations`-based dashboard: which agents in your org committed the most this week, what was the verdict on their PRs.
- Day 28: Run a "red team" exercise: spawn an agent with a poisoned tool definition that rewrites commit messages. Confirm your policy layer catches it.
- Day 29: Pick one vendor to stop using. (Optional but recommended.)
- Day 30: Write a one-page internal doc: "How we do agentic git at <org>." Pin it.

## 10. Cross-references

- **01-Overview** — what this category is
- **02-Core-Topics** — the seven primitives
- **03-Technical-Deep-Dive** — four from-scratch implementations
- **05-Future-Outlook** — the 2027-2030 trajectory

## 11. Further reading

### Primary sources (READMEs)
- `anthropics/claude-code` README
- `sst/opencode` README + plugins docs
- `Gitlawb/node` README + `SECURITY.md`
- `Gitlawb/opencode-gitlawb` README
- `max-sixty/worktrunk` README + `wt --help`
- `automazeio/ccpm` README

### Blog posts and announcements
- Cursor Blog: "Self-driving codebases" (2026)
- Cursor Changelog (filter on "Cloud Agents", "git", "PR")
- Gitlawb announcement thread on HN

### Papers / RFCs (skim for grounding)
- W3C DID Core 1.0
- RFC 9421 — HTTP Message Signatures
- UCAN Spec (ucan.xyz)
- Sigstore whitepaper

### Communities
- HN: `?q=agentic+git`, `?q=cursor+cloud+agents`
- Discord: Cursor, Claude Code, OpenCode servers all have #git / #agents channels
