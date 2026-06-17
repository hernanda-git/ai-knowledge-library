# 12 — Agentic Git: Legal Precedent and Intellectual Property

> Who owns an AI agent's commit? Is it copyrightable? Can it infringe? What about GPL contamination when an LLM was trained on copyleft code? This chapter covers the 2026 legal landscape — what the cases have said, what regulators are about to say, and the practical patterns for keeping an agentic-git deployment legally clean.

## 1. The four legal questions every agentic-git deployment faces

1. **Authorship.** Who is the legal author of a commit authored by an agent? The operator? The vendor? The model itself?
2. **Copyrightability.** Is the commit's content copyrightable at all? (US: requires human authorship.)
3. **Infringement risk.** Did the agent reproduce licensed code (training data, copyleft contamination, or attribution-required snippets) without permission?
4. **Liability.** When an agent's commit causes harm (data leak, IP infringement, breach of contract), who pays?

The answers are converging but not yet stable. Here is the 2026 state of play.

## 2. Authorship (the leading question)

### 2.1 The US position

The US Copyright Office has issued three rounds of guidance (2023, 2024, 2025) on AI-generated works. The consistent position:

> Pure AI output, without sufficient human creative contribution, is not copyrightable.

But the guidance does **not** say "AI assistance disqualifies the work." It says "the human contributions must be more than de minimis."

For agentic-git commits specifically:

| Level of human involvement | Copyright status (US) |
|---|---|
| Human writes the spec, agent writes the code | Probably copyrightable to the human (the spec is the creative contribution) |
| Human approves the agent's PR with minor edits | Borderline; depends on whether the edits reflect creative input |
| Agent operates fully autonomously with no human review | Likely not copyrightable; public domain |
| Human provides prompt + agent generates code with no review | Likely not copyrightable |

For a business this matters because:

- **Uncopyrightable = no protection from competitors.** Anyone can copy your agent-generated code.
- **Uncopyrightable = no standing to sue for infringement** by others.

### 2.2 The EU position

The EU's 2024 AI Act (effective phased through 2026-2027) treats AI-generated content via the Database Directive and the Software Directive. The practical position:

- AI-generated software may be eligible for copyright if there is "the author's own intellectual creation" (the human's creative input).
- The threshold is lower than the US: a human who steers an agent sufficiently may qualify.

### 2.3 The practical pattern

Most 2026 legal guidance for engineering teams:

1. **Document the human's creative contribution.** The PR description, the spec, the test cases — these are the human's. Make sure they're recorded.
2. **Have a human reviewer sign off.** A simple `Reviewed-by: <human>` trailer is evidence of human creative oversight.
3. **Don't claim AI-generated code is "novel."** The copyright office has rejected novelty claims for purely AI-generated content.
4. **For copyrightable work, register the human-authored parts.** Don't try to register the agent's output.

## 3. The Co-authored-by question

The `Co-authored-by:` trailer is now ubiquitous. The legal question: does putting your name next to Claude's actually make you a co-author?

### 3.1 The consensus position

No. `Co-authored-by` is a **git convention**, not a legal construct. GitHub's help page on trailers explicitly says they're for attribution, not copyright assignment.

To actually share copyright, you need:

- A written agreement
- All parties' consent
- A work that qualifies for copyright (i.e., has human creative input)

### 3.2 The trailer hygiene

Despite not being legally binding, the trailer matters because:

- It documents the **process** (a human + an agent collaborated)
- It creates a **rebuttable presumption** that a human was involved
- It sets **expectations** for downstream users (other contributors know it's AI-assisted)

The 2026 best practice: every agent commit has both `Co-authored-by: <agent>` (truthful) and `Reviewed-by: <human-did>` (evidence of human creative oversight).

```text
feat(auth): add rate-limiting middleware

Lore-id: a1b2c3d4
Co-authored-by: Claude <noreply@anthropic.com>
Agent-DID: did:key:z6Mk…
Reviewed-by: Hernanda <m.hernanda95@gmail.com>
Human-DID: did:key:z6Mk…
```

## 4. License contamination

### 4.1 The training-data problem

LLMs are trained on public code, much of which is GPL-AGPL-MIT-Apache-BSD licensed. When an agent generates code, it may (often unintentionally) reproduce snippets from its training data.

This is the most legally fraught area of agentic-git in 2026. The unsolved questions:

- Is output of an LLM a "derivative work" of its training data?
- If the LLM was trained on GPL code, is its output GPL-encumbered?
- Does the size of the snippet matter (substring length, number of lines)?
- Does the LLM vendor bear liability, or the user, or both?

### 4.2 The current case law

As of mid-2026, no major jurisdiction has definitively ruled that LLM output is a derivative work of training data. The pending cases to watch:

- **Doe v. OpenAI** (US, ND CA, 2024-present) — class action alleging training on copyrighted code; status: in discovery
- **Getty v. Stability AI** (UK, 2024-present) — UK High Court; first major decision in late 2025 was partial win for Getty on training-data use but did not directly address output copyright
- **Authors Guild v. OpenAI** (US, SDNY, 2023-present) — settlement discussions ongoing
- **Programmer's Guild v. GitHub Copilot** (US, 2022-present) — narrow holding on a specific reproduction; broader claims pending

The pattern emerging: **training is legal in most jurisdictions; specific output reproduction may not be.**

### 4.3 The agentic-git implications

For your team's commits:

1. **Avoid verbatim reproduction.** If an agent outputs 50+ lines that match a known OSS project's source, that's a red flag. Pre-commit license scanners (ScanCode, FOSSology, licensee) can detect this.
2. **Watch for copyleft contamination.** If your repo is MIT-licensed and an agent outputs code that resembles GPL code, your repo's license integrity is at risk.
3. **Document the operator's edits.** The more the human changed the agent's output before commit, the stronger the "human creative contribution" argument.

### 4.4 Pre-commit license scanning

The 2026 stack:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/marketplace/scan-oss
    rev: v1.2.3
    hooks:
      - id: scan-oss
        # Block commits that introduce copyleft code into a permissive repo
        policy: strict-permissive

  - repo: https://github.com/licensee/licensee
    rev: latest
    hooks:
      - id: licensee
        # Verify the repo's LICENSE matches the agent's expected output license
```

Tools:

- **ScanCode Toolkit** (Linux Foundation) — detects ~1,000 licenses
- **FOSSology** — multi-license scanner with web UI
- **licensee** (GitHub) — quick LICENSE file detection
- **CodeQL** — query-based code scanning that can flag suspicious similarity
- **Copyleft Detector** — Perl-based, focused on GPL contamination

## 5. The training-data output traceability

A 2026 best practice: every agent commit includes provenance metadata about the model's training corpus:

```text
feat(api): add /v2/users endpoint

Lore-id: a1b2c3d4
Agent-DID: did:key:z6Mk…
Agent-Model: claude-opus-4.5
Training-Cutoff: 2026-03
Training-Data-Provenance: https://www.anthropic.com/data-provenance
Co-authored-by: Claude <noreply@anthropic.com>
Reviewed-by: Hernanda <m.hernanda95@gmail.com>
```

This metadata doesn't prove non-infringement, but it documents the provenance chain. If a dispute arises, the existence of the chain is evidence of due diligence.

## 6. Liability for agentic-git commits

### 6.1 The contractual chain

When an agent commits, several parties are in the chain:

```text
Vendor (e.g., Anthropic)
  │ provides the model
  ▼
Operator (you, the human who runs the agent)
  │ configures the agent, sets policies
  ▼
Agent (the LLM-driven process)
  │ produces the commit
  ▼
Repo maintainer / employer
  │ owns the resulting code
  ▼
End user / customer
  │ affected by the deployed service
```

Liability flows down the chain. When an agent's commit causes harm:

- The **operator** is primarily liable (they chose to deploy the agent).
- The **vendor** may share liability if their model produced demonstrably defective output.
- The **repo maintainer** may be liable if they merged without review.
- The **end user** has standing to sue.

### 6.2 The 2026 case law direction

Pending cases are testing whether AI-vendor terms of service (which usually disclaim liability for output) are enforceable. The trend:

- **B2C**: Terms that disclaim all liability are increasingly challenged.
- **B2B**: Sophisticated parties can contractually allocate risk; terms hold more often.
- **Regulated industries** (health, finance, defense): Human-in-the-loop requirements make the operator's liability clearer.

### 6.3 The "operator liability" standard

In practice, the 2026 standard for safe deployment:

1. **You (operator) are responsible for your agent's commits.** Period. No terms of service change this.
2. **The defense is process.** Show you had policies, code review, security scanning, pre-commit hooks.
3. **Insurance is becoming mandatory.** Several major insurers now require AI-liability riders for any deployment that produces external commits.

## 7. Cross-border considerations

If your team is global (which most 2026 dev teams are), you need to track:

| Jurisdiction | Key law | Agent-implication |
|---|---|---|
| US (federal) | Copyright Act | Requires human authorship |
| US (state) | Various AI laws (CO, CA, IL) | Disclosure required for AI in hiring, healthcare |
| EU | AI Act (2024) | High-risk systems require human oversight |
| EU | GDPR | Agent processing personal data → DPIA required |
| UK | AISI guidance (2024) | Voluntary; emergent standards |
| China | Generative AI rules (2023) | Content labeling required |
| Japan | AI Promotion Act (2025) | Light-touch; innovation-friendly |
| Singapore | AI Verify framework (2023) | Voluntary testing standard |

For a global team, the safest baseline:

- Human review for any production commit.
- AI-generated content labeled as such in any user-facing output.
- EU GDPR-compliant data handling in agent prompts.
- License provenance documented.

## 8. Patent implications

A subtler issue: if an agent's commit reads as novel and non-obvious, it could be **patentable** by the operator. If it reads as obvious (or as a re-combination of training data), it's not.

The 2026 emerging practice:

- **Operators are filing patents on agent-generated inventions** that meet the non-obviousness threshold.
- **The US Patent Office** (as of 2025 guidance) requires that the inventor be a human; an AI cannot be named. The human operator who conceived the underlying idea is the inventor.

For an engineering team:

- Document the **human's inventive concept** in PR descriptions (the "why" of the change).
- Treat the agent's output as a tool, like a compiler; the human's specification is the inventive concept.
- Consult IP counsel before filing patents on agent-assisted work.

## 9. The AI Act and agentic git specifically

The EU AI Act's **Article 14** (Human Oversight) requires that high-risk AI systems "enable individuals to understand the AI system's capabilities and limitations, monitor its operation, and intervene on its output."

For agentic-git, the practical compliance pattern:

```text
Commit metadata required for AI Act Article 14 compliance:

1. System identification: model name, version, vendor
2. Operator identification: human DID or name
3. Decision rationale: PR description, design docs
4. Human oversight evidence: Reviewed-by trailer
5. Reversibility mechanism: revert commit, force-push block

Example:
Agent-System: claude-opus-4.5 (Anthropic)
Agent-DID: did:key:z6Mk…
Operator-DID: did:key:z6Mk…
Operator-Role: senior-engineer
Decision-Rationale: PR-description.md (attached)
Human-Oversight: reviewed-and-approved by Hernanda <...>
Reversibility: revert-by-bf7e8d9 revert-this-commit
```

By 2027, expect every enterprise agent commit to carry this metadata as standard.

## 10. Practical compliance checklist

For an engineering team running agentic-git in 2026:

- [ ] Every agent commit has `Co-authored-by: <agent>` and `Reviewed-by: <human>` trailers
- [ ] Every agent commit has a PR description documenting human creative contribution
- [ ] Pre-commit license scanning is enabled and enforced
- [ ] No copyleft code is introduced into a permissive-licensed repo (or is explicitly accepted via PR review)
- [ ] AI Act Article 14 metadata is captured for any production code
- [ ] The team has an AI liability insurance rider
- [ ] The team's AGENTS.md includes a license-compliance section
- [ ] Agent output is reviewed for verbatim training-data reproduction
- [ ] The team's CODEOWNERS file requires human review for sensitive paths
- [ ] Annual legal review of agentic-git practices (or as regulations change)

## 11. The 18-month horizon

By Q2 2027 expect:

- **EU AI Act Article 14 enforcement actions** that establish precedent on what counts as human oversight for code commits.
- **A landmark US case** ruling on whether LLM output can be a derivative of training data.
- **GitHub's Co-authored-by becoming a recognized (but not legally binding) attribution standard**.
- **A "License Provenance" trailer** becoming standard on agent commits, similar to today's Co-authored-by.
- **First API vendor indemnification clauses** specifically for agentic-git commits.
- **Lore-id trademarked** as a standard for knowledge-graph commit metadata.

## 12. See also

- Cat 21 — *AI Regulation & Antitrust* (broader regulatory landscape)
- Cat 24 — *Agent Autonomy & Operator Liability* (the autonomy side)
- Cat 27 — *AI Agent Legal Entities & DAO Governance* (the legal-personhood side)
- **06-Security-and-Supply-Chain** — supply-chain attacks that hide in agent commits
- **07-Prompt-and-Commit-Patterns** — the trailer conventions that support legal hygiene
- **08-Agent-Memory-and-Long-Running-Workflows** — documenting decision shadow as a legal record
