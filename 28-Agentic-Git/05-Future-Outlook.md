# 05 — Agentic Git: Future Outlook

> Where the agentic-git stack goes from mid-2026 to 2030 and beyond. Twelve-month watch list, three plausible 2030 scenarios, hardware/legal/economic forecast, and the risks that could derail it.

## 1. The shape of the next 18 months (H2 2026 → H2 2027)

A single dominant pattern will crystallize, and four adjacent ones will emerge.

### 1.1 The dominant pattern: signed-agent commit becomes the default

By Q4 2026, every major coding agent (Claude Code, Cursor Cloud, OpenCode, Codex, GitHub Copilot Agent) will sign commits by default with a portable DID. The signing key won't be a per-user SSH key anymore — it'll be a per-agent identity, generated at agent creation time, rotated annually, and bound to a `did:key` that survives the agent moving between harnesses.

Concretely:

```bash
$ git log --format='%H %an <%ae> [signed-by=%GS)' -5
8f4e1a3a Claude  <noreply@anthropic.com>   [signed-by=did:key:z6Mk…]
3b2c4d5e Cursor  <agents@cursor.sh>       [signed-by=did:key:z6Mk…]
a1b2c3d4 OpenCode <noreply@opencode.ai>   [signed-by=did:key:z6Mk…]
…
```

GitHub will surface the signing DID in the PR header. "Verified by Claude Code (DID z6Mk…)" replaces "Co-authored-by: Claude."

### 1.2 Four emerging adjacent patterns

1. **Agent-to-agent PR review as a measurable profession.** A new role — "agent reviewer" — emerges. Tools like Cursor Bugbot, Claude Code Review, and Greptile accumulate reputation graphs. A reviewer's DID carries weight; their approvals are trusted.

2. **Bounty-driven repositories.** Open-source projects (especially infra) routinely have a `BOUNTIES.md` listing open tasks with token escrowed. Agents that can claim, ship, and merge bounties become a new class of independent worker.

3. **Decentralized code replication by default.** Projects start using Gitlawb or similar to mirror their canonical repos to multiple nodes. The "GitHub went down for an hour" incident becomes a non-event for projects that replicated.

4. **Commit-as-attestation for legal/regulatory contexts.** AI Act Article 14 (human-in-the-loop for high-risk AI) is satisfied by a `human-approved-by:` DID trailer on the final commit. The trail of agent DIDs leading up to it becomes an audit log.

## 2. Three plausible 2030 scenarios

Each is internally consistent. The actual outcome will be some mix.

### 2.1 Scenario A: "Agent-native wins" (probability: 35%)

By 2030, Gitlawb or a successor is the default git host for agentic workloads. GitHub remains dominant for human-only repos but loses the agent-driven tail. GitHub responds by adopting the same primitives (DID signing, agent reviews) but the network-effect is already on the decentralized side.

**Triggers:**
- A major outage or policy change at GitHub triggers migration of 2-3 large OSS projects.
- A regulatory body (EU AI Office, US AI Safety Institute) requires DID-signed commits for any AI-touched code in regulated sectors.
- A new agent-native code market (Gitlawb bounties + competitors) reaches $1B in settled volume.

**Implications:**
- GitHub pivots to "AgentHub" or gets acquired.
- Agent identity becomes a self-sovereign primitive (DID → on-chain reputation → real-world contracts).
- Code quality goes UP because bad agents have reputation cost; goes DOWN because velocity creates more bad commits overall.

### 2.2 Scenario B: "GitHub absorbs" (probability: 45%)

GitHub ships DID signing, agent reputation, and on-chain bounties natively before any decentralized rival reaches escape velocity. Gitlawb remains a niche for crypto-native teams; most agents commit to GitHub as before but with a richer identity model.

**Triggers:**
- Microsoft decides AI-coding is strategically critical and ships the integration fast.
- Sigstore (already backed by Google + Red Hat + Chainguard) becomes the default signing layer and GitHub adopts it.
- A major enterprise customer demands DID-signed commits; GitHub delivers.

**Implications:**
- Decentralized git stays a curiosity for cypherpunks.
- GitHub's agent reputation graph is centralized → becomes a moat.
- Agents become first-class users of GitHub. "Bot accounts" become "agent accounts" with explicit permission scopes.

### 2.3 Scenario C: "Fragmentation" (probability: 20%)

No winner emerges. Agents commit to whatever host their human told them to. Signing standards remain fragmented (Ed25519, Sigstore, GPG, SSH, etc.). Bounty markets proliferate in incompatible ways. The space remains confusing.

**Triggers:**
- A schism in the DID community (e.g. did:key vs did:web vs did:ethr) fails to resolve.
- GitHub, GitLab, Gitea, Gitlawb all ship incompatible agent APIs.
- High-profile agent-commits-bad-thing incident triggers rollback of agent autonomy.

**Implications:**
- Adoption slows.
- Each enterprise picks a vendor and locks in.
- The open-source community forks Gitlawb multiple times; chaos.

## 3. Economic forecast: code as a market commodity

### 3.1 The bounty market

Gitlawb bounties (and competitors) settle on Base or similar L2s. Volume projections:

| Year | Active projects | Monthly bounties posted | Settled volume (USDC) |
|---|---|---|---|
| 2026 | ~500 | ~2,000 | ~$200K |
| 2027 | ~5,000 | ~25,000 | ~$5M |
| 2028 | ~50,000 | ~300,000 | ~$80M |
| 2030 | ~500,000 | ~5M | ~$2B |

Assumptions: 30% of OSS projects have at least one active bounty by 2028; average bounty size grows from $25 (2026) to $200 (2030) as enterprise use cases appear.

### 3.2 The labor-market impact

**Displaced roles (by 2028):**
- Junior-level PR reviewers → replaced by agent reviewers + human exception handlers
- Routine bug fixers on OSS projects → replaced by agents claiming bounties
- CI/CD maintenance engineers → replaced by self-healing pipelines

**Augmented roles:**
- Senior engineers → spend more time on architecture, less on PR review
- DevOps engineers → orchestrate agent swarms instead of writing pipelines
- Open-source maintainers → market-makers, not committers

**New roles:**
- Agent reputation analyst (tracks which agents have shipped what)
- Bounty market-maker (designs token economics for a project's contributor base)
- Agent policy engineer (writes the YAML/rules that gate agent commits)

## 4. Legal and regulatory forecast

### 4.1 EU AI Act (effective phased through 2026-2027)

Article 14 requires human oversight for high-risk AI systems. The pragmatic compliance path is the `human-approved-by:` DID trailer on every PR that an AI touched. By 2027, expect EU regulators to require:

- Every commit in a regulated codebase must declare: human author OR agent author (with DID) AND human approver (with DID).
- Audit logs must be reproducible for 7 years.

The cat-24 (Autonomy & Liability) and cat-27 (Agent Legal Entities) work converges here.

### 4.2 US AI Safety Institute (2026-2028)

Less prescriptive than the EU but expects voluntary commitments. Expect NIST to publish a *Profile for Agentic Code Generation Systems* by mid-2027 that recommends DID-signed commits and on-chain reputation as best practices.

### 4.3 Liability and IP

- **Copyright:** Who owns an agent's commit? The agent's operator. By 2028 expect case law to settle that the *human who deployed the agent* is the author; the agent is a tool.
- **Patent:** Same principle. The human assigns.
- **Trade secrets:** Agent commits don't lose trade-secret protection automatically, but DID-signed logs make it much easier to prove *who* leaked.

## 5. Hardware and infrastructure

### 5.1 Edge compute

Agent commits require <100ms latency to git remotes. By 2028 expect:

- Gitlawb-style nodes running on every major CDN POP
- Edge-rendered `git clone` for repos with sub-1MB initial size
- Mobile-native git ops (already in progress at Gitea)

### 5.2 Storage

DID-signed attestations + commit history is ~2x the storage of plain git by 2030. Expect:

- Cold storage of attestation chains on Filecoin / Arweave
- Pruning of attestation history for repos >10 years old
- Compression of DID signatures by 2030 (lattice-based?)

### 5.3 Bandwidth

Each agent commit triggers: clone → diff → push → attestation → PR. At 1M concurrent agents, that's ~500GB/s of git traffic globally. Will require:

- Repo-aware CDN caching (only changed refs)
- Diffs over QUIC instead of HTTPS
- Batched pushes (10 commits → 1 push)

## 6. The risks

### 6.1 Risks to adoption

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Major security incident: agent commits backdoor to widespread OSS | Medium | High | DID-signed commits + reputation; slower rollout |
| GitHub acquires or kills a major agent-native project | Medium | Medium | Standards-based identity (DID) survives platform switch |
| Regulation requires *only* human commits in critical infrastructure | Medium | High | Politically difficult to enforce; tooling for human-in-loop will be the answer |
| Identity fragmentation: did:key vs did:web vs did:ethr | High | Medium | Universal resolver spec; market convergence by 2028 |
| Cheap agents flood the market → race to bottom on bounty quality | High | Medium | Reputation graphs filter; bounties get harder/expensive |

### 6.2 Risks to the technology itself

- **Cryptographic obsolescence:** Ed25519 is fine through ~2030; quantum threat is 2035+. Plan migration to dilithium by 2028.
- **Sybil attack on agent reputation:** A bad actor spins 1000 agents, ships 1000 mediocre PRs. Reputation graph + cross-agent vouching mitigates; not solved.
- **Repo hijack via DID compromise:** Lose your Ed25519 key, lose your commit identity. Hardware-backed keys (YubiKey, TPM) become mandatory by 2028.

### 6.3 Sociotechnical risks

- **Author identity erodes.** When every commit is `agent-x`, humans stop getting individual credit. Academia, hiring, open-source governance all suffer unless we invent new recognition systems.
- **Maintainer burnout shifts.** The bottleneck moves from "writing code" to "reviewing agent PRs at machine speed." New burnout mode.
- **OSS funding breaks.** GitHub Sponsors was built on human author parasocial ties. With agents, those ties may not form. New funding models (bounties, recurring agent licenses) need to fill the gap.

## 7. Watch list for the next 12 months

### Tier 1 — high signal, watch closely
- GitHub's response to Gitlawb (acquisition, native DID support, or ignore)
- Gitlawb mainnet launch and bounty volume
- EU AI Office enforcement actions under Article 14
- Major OSS project migrating to agent-native git
- Cursor Cloud Agents GA timeline and pricing
- Anthropic's stance on commit signing for Claude Code

### Tier 2 — industry shape
- OpenAI Codex CLI getting DID signing
- Backlog.md vs ccpm: which wins the "OSS project management for agents" mind-share
- `entireio/cli` adoption in enterprises
- Any "agent committed catastrophic bug" headline and its postmortem
- A new L2 (besides Base) launching an agent-reputation primitive

### Tier 3 — long shots
- A non-blockchain DID method winning (did:key + IPFS log?)
- A regulator-mandated "human author percentage" disclosure on GitHub
- A first AI-agent-only VC fund that invests via git bounties
- A court case setting precedent on agent commit liability

### Tier 4 — culture
- A widely-cited paper on "agent reputation systems"
- The first "agent co-author" Nobel Prize / Oscar / Pulitzer joke that becomes real
- An AI agent named to a project's governance board
- A magazine profile of a top-earning agent (and the human who trained it)

## 8. The long-horizon view (2030-2040)

By 2035, code is *primarily* agent-generated. The interesting questions are no longer "can an agent commit?" but:

- Who pays agents? (Bounty markets, subscription agents, corporate agents?)
- Who reviews agents? (Other agents, with human exception handlers.)
- Where do agents live? (Per-org, per-user, per-task identities?)
- How do agents merge? (Multi-agent consensus protocols for `main`?)
- What does "ownership" mean? (DID-bound identity, on-chain reputation, legal personhood?)

The git substrate is unlikely to change — the commit, the branch, the PR are durable abstractions. What's layered on top is what 2030-2040 will sort out.

**Cat 28 will be revisited quarterly through 2027.**

---

## Appendix A — Glossary

- **Agent:** Any LLM-driven system that can read code, write code, and invoke tools including git.
- **DID:** Decentralized Identifier (W3C standard). `did:key:z6Mk…` is the most common form for agents.
- **Ed25519:** A digital signature algorithm. Fast, small signatures, well-suited for agent identity.
- **HTTP signature (RFC 9421):** A standard for signing HTTP messages with a private key. Used by Gitlawb Node for write auth.
- **MCP:** Model Context Protocol. How agents talk to tools.
- **UCAN:** User Controlled Authorization Networks. Capability tokens that chain across services.
- **Worktree:** A git feature allowing multiple working directories from one `.git`.

## Appendix B — All 47 projects in one list

```text
Coding agents (11):
  anthropics/claude-code              132.9k ★
  Gitlawb/openclaude                   29.0k ★
  aider                                  ~35k ★
  Continue.dev                            ~25k ★
  Cline / Roo Code                        ~30k ★
  Cursor Cloud Agents                     (proprietary)
  OpenCode (sst/opencode)                 ~16k ★
  Codex CLI (OpenAI)                      (proprietary)
  GitHub Copilot Coding Agent             (proprietary)
  Cody (Sourcegraph)                      (proprietary)
  stagewise-io/stagewise                  6.7k ★

Git hosts and remotes (7):
  GitHub                                  (proprietary)
  GitLab                                  (proprietary)
  Gitea / Forgejo                         ~50k ★
  Gitlawb/node                            34 ★
  open-gitagent/opengap                   2.8k ★
  entireio/cli                            4.5k ★
  Ataraxy-Labs/sem                        2.9k ★

Orchestration (6):
  max-sixty/worktrunk                     5.5k ★
  automazeio/ccpm                         8.2k ★
  Gitlawb/agentvm                         22 ★
  MrLesk/Backlog.md                       5.8k ★
  stagewise-io/stagewise                  6.7k ★
  Dicklesworthstone/mcp_agent_mail        2.0k ★

Identity and signing (5):
  Ed25519 (primitive)                     (RFC 8032)
  did:key (W3C)                           (W3C standard)
  Sigstore / Fulcio / Rekor               (sigstore.dev)
  UCAN spec                               (ucan.xyz)
  Gitlawb identity                        (in Gitlawb/node)

Markets and code-as-asset (3):
  Gitlawb bounties                        (in Gitlawb/node)
  Gitlawb/contracts                       5 ★
  Gitlawb/banker-skills                   16 ★

Code intelligence (2):
  abhigyanpatwari/GitNexus                42.3k ★
  Ataraxy-Labs/sem                        2.9k ★

Plugin/extension layer (3):
  Gitlawb/opencode-gitlawb                77 ★
  Gitlawb/openclaude-skills               5 ★
  Gitlawb/gl-npm                          1 ★

Supporting (4):
  Gitlawb/releases                        8 ★
  Gitlawb/homebrew-tap                    0 ★
  Gitlawb/node-explorer                   0 ★
  Gitlawb/nansen-hackathon-poc            3 ★
```

Star counts as of mid-2026. Maintained by quarterly refresh.
