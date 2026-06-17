# 02 — Agentic Git: Core Topics

> The seven technical primitives every agentic-git system must implement — or plug into — to be useful in 2026. Each section gives the principle, the canonical implementation, and a short example.

## 1. Agent identity

### The problem
Git was designed around a human email + name. Agents have neither. When an agent `git commit`s today, you get whatever string the agent's harness wrote — usually a `Co-authored-by:` trailer with a generic name. There is no portable identity the agent can carry across hosts.

### The 2026 standard: Ed25519 `did:key`

A DID (Decentralized Identifier) is a W3C standard. The `did:key` method bakes a public key directly into the identifier — no resolver needed. For agents, the canonical form is:

```
did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK
```

That's it. The string IS the identity. Anyone with the string can verify a signature; no PKI lookup required.

### Reference implementation (Python, ~30 lines)

```python
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
import base58, multibase

def new_agent_identity(label: str = "agent") -> tuple[str, str]:
    """Returns (did, pem_path). The DID is the agent's identity everywhere."""
    sk = Ed25519PrivateKey.generate()
    pk_bytes = sk.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )  # 32 bytes
    # multibase encode the multicodec-prefixed pubkey (0xed01 for Ed25519)
    multicodec = b"\xed\x01" + pk_bytes
    did = "did:key:" + multibase.encode("base58btc", multicodec).decode()
    pem = sk.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    path = f"~/.agents/{label}.ed25519.pem"
    with open(path, "w") as f:
        f.write(pem)
    return did, path
```

### Why it matters
A `did:key` survives the agent moving from Cursor to Claude Code to a self-hosted OpenCode. The same identity can sign commits on GitHub (via a signed-message hook), sign HTTP requests against Gitlawb Node (RFC 9421), and authorize UCAN tokens for on-chain actions. One identity, three substrates.

Gitlawb implements this exactly: every node, user, and agent gets an Ed25519 keypair at `~/.gitlawb/identity.pem`, and the CLI prints the DID on `gl identity new`.

## 2. Commit signing

### The problem
A `Co-authored-by:` trailer is not a signature. Anyone can paste it. For agent commits to be auditable, the *signing identity* must be cryptographically bound to the commit, not just textually claimed.

### The 2026 patterns

There are three signing layers in use, in increasing order of strength:

| Layer | What it proves | Used by |
|---|---|---|
| **GPG/SSH trailer** | A known key signed the commit hash | Most humans, GitHub Copilot |
| **X.509 (Sigstore)** | A specific OIDC identity signed it, logged in a transparency log | Sigstore Fulcio + Rekor |
| **Ed25519 agent signature** | The DID signed a structured payload (commit hash + author + timestamp + repo URI) | Gitlawb, agentic-git primitives |

The agentic-git-native form looks like this (from a `gl commit` invocation):

```json
{
  "type": "commit-attestation/v1",
  "subject": "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK",
  "repo": "did:gitlawb:abc123/my-repo",
  "branch": "main",
  "commit": "8f4e1a3...",
  "timestamp": "2026-06-17T07:42:11Z",
  "message_sha256": "9a2f..."
}
```

Signed with the Ed25519 key, verifiable by anyone holding the DID. The signature gets stored alongside the commit on the Gitlawb node and replicated via libp2p.

### Why it matters
Once commits are signed by an agent's DID, you can build **reputation graphs** without trusting GitHub. An agent that has shipped 500 signed commits resolving on-chain bounties has a verifiable track record. A new agent with no history is a sybil risk.

## 3. Worktrees as agent sandboxes

### The problem
Two agents cannot `git checkout -b` the same branch at the same time. Even one agent doing three sequential sub-tasks blocks on its own earlier worktrees.

### The solution: one worktree per agent (or per sub-task)

`git worktree add` lets a single `.git` directory have multiple checked-out working trees. Each worktree gets its own branch and its own working directory.

`worktrunk` (`max-sixty/worktrunk`, 5.5k★) wraps this with agent-aware defaults:

```bash
wt switch -c 'feature/auth-rewrite'   # new worktree + new branch
wt list                                # show all worktrees
wt merge --into main feature/auth-rewrite
```

`agentvm` (Gitlawb/agentvm) does the same with N agents in parallel, each in their own worktree:

```bash
agentvm run --agents 4 --task "refactor: split billing module" .
```

### The pattern in practice

```text
main
├── worktree-A/  → branch: agent-a/fix-142  (Claude Code)
├── worktree-B/  → branch: agent-b/refactor (Cursor Cloud)
├── worktree-C/  → branch: agent-c/tests    (OpenCode)
└── worktree-D/  → branch: agent-d/docs     (human)
```

Each agent commits in isolation. A `gl pr create` (or `gh pr create`) ties them together. The human reviews the diff per branch — not per commit.

### Why it matters
This is the single most important infrastructure primitive for *running multiple agents in parallel without stomping on each other*. Without worktrees, only one agent can work the repo at a time, which destroys the throughput win.

## 4. PR review by other agents

### The 2026 shape

GitHub PR review by AI is now standard, but the *interesting* pattern is **agent-to-agent review**:

- A coding agent opens a PR.
- A *different* review agent (often a smaller/faster model) reads the diff, leaves inline comments, requests changes.
- The original agent addresses comments, pushes again.
- A human approves when satisfied.

Tools in this space:

| Tool | Role | Note |
|---|---|---|
| **Cursor Bugbot** | Background code review on PRs | Cursor's hosted reviewer; 5-pt HN thread |
| **Claude Code Review** | PR review subcommand | `claude code review --pr 142` |
| **Gitlawb `gitlawb_pr_review`** | On-chain agent review record | Review attestation stored on node, replicated |
| **CodeRabbit** | Commercial PR review agent | Independent of GitHub |
| **Greptile** | Codebase-aware PR review | Graph-based context |

### The interesting primitive: review-as-attestation

When an agent reviews another agent's PR, the review itself is a signed attestation:

```json
{
  "type": "pr-review/v1",
  "subject": "did:key:z6Mk...",
  "pr": "owner/repo#142",
  "verdict": "request-changes",
  "comments": [
    {"file": "src/auth.ts", "line": 42, "body": "race condition in refresh path"}
  ],
  "timestamp": "2026-06-17T08:14:00Z"
}
```

This becomes a permanent part of the agent's track record. Over time, agents with hundreds of approved reviews gain reputation; those whose reviews are repeatedly overruled lose it.

## 5. Bounties and code-as-asset

### Gitlawb's bounty flow (the canonical 2026 implementation)

The full primitive, copy-paste from `Gitlawb/opencode-gitlawb` README:

```
Agent: I'll check for open bounties on this repo.
→ gitlawb_bounty_list(repo: "owner/repo", status: "open")

Agent: I'll claim bounty #42 and fix the bug.
→ gitlawb_bounty_claim(id: "42")

Agent: [writes code, commits, pushes]

Agent: Creating a PR for the fix.
→ gitlawb_pr_create(repo: "repo", head: "fix/bug", title: "Fix login timeout")

Agent: Submitting PR against the bounty.
→ gitlawb_bounty_submit(id: "42", pr: "1")
```

The contract logic runs on Base (Ethereum L2). When the PR merges, the escrowed tokens release to the claimer's address. If the bounty expires unmerged, the tokens refund to the poster.

### What this enables

A repo with active bounties becomes a *job board* for agents. Agents that can prove they shipped the fix get paid, automatically, with no human payroll in the loop. The maintainer of a popular open-source project becomes a market-maker, not a maintainer.

The economic primitive is the same one Uber and Airbnb use: a *protocol* (escrow, dispute resolution, reputation) layered over a *market* (here, git-native labor).

## 6. Decentralized replication

### Why vanilla GitHub fails

GitHub is one company. If it goes down (rate-limit incident, legal pressure, hostile acquisition), every agent whose commit path goes through GitHub stops. The 2026 outages show this is not hypothetical.

### Gitlawb's answer

```text
Decentralized GitHub
+ signed agent-native workflows
+ resilient repo replication
+ CDN-style app/code delivery
```

A Gitlawb Node exposes standard git smart-HTTP. Multiple nodes can announce, discover, gossip, and sync. Each node signs writes with Ed25519; libp2p is the transport. Optional pinning to IPFS/Pinata, Arweave/Irys, S3/Tigris for durability.

What this means concretely:

- You can `git clone gitlawb://node-A/my-repo` and get the same repo from `node-B`.
- The maintainer of a node doesn't have authority to delete a repo they don't own — DIDs gate writes.
- A repo's commit history is content-addressed and replicated; no single failure mode.

### The pragmatic caveat

As of mid-2026, Gitlawb's own README is honest about the limits:

> Private repository read enforcement is not wired yet. Treat public nodes as public infrastructure unless you restrict access at your proxy/firewall.
>
> UCAN chain validation and revocation are not complete.
>
> Repository write authorization is not capability-complete yet; HTTP signatures prove identity, not full authorization policy.

Decentralized git for agents is real but not production-ready for private data. Use it for public repos and open-source bounties first.

## 7. Policy and guardrails

### The 2026 best-practice stack

Once agents can commit at machine speed, you need policy in depth. The standard stack is:

```yaml
# .agentgit/policy.yaml (proposed; conventions still forming)
identity:
  require_did: true                # commits must include DID
  allow_agents:
    - did:key:z6Mk...              # your coding agent
    - did:key:z6Mk...              # review agent
  block_agents: []

branches:
  main:
    require_signed: true
    require_review:
      min_approvals: 1
      allowed_reviewers:
        - did:key:z6Mk...          # the human (paired with their main identity)
    max_commits_per_hour: 20
    banned_paths:
      - ".github/workflows/**"
      - "secrets/**"

content:
  block_secrets: true               # scan commit diffs for AWS keys, etc.
  block_large_files_mb: 10
  require_tests: true
```

### The runtime enforcement layer

Tools like **`sentry-cli` + agent hooks** (custom) and **`gitlawb_doctor`** (built-in) check identity, node reachability, and git config on every commit attempt. **Pre-commit frameworks** (the existing `pre-commit.com` ecosystem) work fine for agent commits — just run them in the agent's commit hook.

The harder problem is *post-commit* — once an agent has pushed, you need:

- Automated code review (Cursor Bugbot, Claude Code Review)
- CI (GitHub Actions, GitLab CI)
- A human-in-the-loop reviewer before merge to `main`

This is the same stack you'd build for humans. The agent just makes it run 100x more often.

## Cross-references

- See **03-Technical-Deep-Dive** for full working code of identity, signing, worktree orchestration, and a from-scratch MCP server.
- See **04-Tools-and-Frameworks** for the 47-project inventory.
- See **05-Future-Outlook** for where this all goes by 2030.
