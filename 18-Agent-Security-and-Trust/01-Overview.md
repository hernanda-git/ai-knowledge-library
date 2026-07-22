# 01 — Agent Security and Trust: Overview

## 1. Introduction

The rapid proliferation of AI agents — autonomous software systems that perceive their environment, reason about goals, and execute actions — has created an entirely new attack surface in modern computing. Unlike traditional software, agents combine large language models (LLMs), tool-calling capabilities, external data access, and autonomous decision-making into systems that can act on behalf of users without direct human supervision at every step. This autonomy, while powerful, introduces profound security challenges that threaten enterprise adoption.

Agent security is not merely an extension of traditional application security. It encompasses novel threat vectors unique to the intersection of AI reasoning, tool execution, and autonomous action. This document provides the foundational overview of the agent security landscape, the threat model, and maps the remaining seven documents in this series.

## 2. Why Agent Security Is Critical

### 2.1 Enterprise Adoption Blocker

Security concerns are consistently cited as the #1 barrier to enterprise deployment of AI agents. In surveys conducted across Fortune 500 organizations, security and trust risks rank above cost, performance, and integration complexity as the primary reason agents remain in pilot phases rather than production deployment.

Key enterprise concerns include:

- **Data leakage**: Agents process sensitive corporate data (PII, financial records, trade secrets) through third-party LLM APIs and tool chains, creating new exfiltration pathways.
- **Unauthorized actions**: Autonomous agents with tool access can execute API calls, write to databases, send emails, or modify infrastructure — actions that, if taken maliciously or erroneously, cause real damage.
- **Regulatory compliance**: Regulated industries (finance, healthcare, legal) require audit trails, access controls, and data governance that agent systems must satisfy before deployment.
- **Liability allocation**: When an agent makes a harmful decision, responsibility is unclear — the developer, the operator, or the model provider?

### 2.2 Unique Characteristics of Agent Security

Agent security differs from traditional software security in several fundamental ways:

**Non-deterministic execution**: LLMs do not produce deterministic outputs. The same prompt can yield different responses, making traditional vulnerability scanning and static analysis insufficient.

**Indirect prompt injection**: Agents process external content (web pages, documents, emails) that may contain adversarial instructions invisible to the user but interpreted by the agent.

**Tool misuse**: Even well-intentioned agents can misuse tools through ambiguous or incomplete instructions, leading to unintended consequences.

**Multi-step reasoning attacks**: Attackers can guide agents through a chain of seemingly benign actions that collectively achieve a malicious goal (the "fragile-to-chain" problem).

**Training data poisoning**: Models used by agents may have been compromised during training, embedding backdoors or vulnerabilities that manifest during deployment.

## 3. Threat Landscape Overview

### 3.1 Attack Surface of an AI Agent

A modern AI agent exposes the following attack surfaces:

```
┌─────────────────────────────────────────────────┐
│                  User Interface                  │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│               LLM / Agent Engine                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │  Prompt  │  │ Reasoning│  │  Memory/      │   │
│  │  Engine  │  │  Engine  │  │  Context      │   │
│  └──────────┘  └──────────┘  └──────────────┘   │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│                  Tool Orchestrator               │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │  Search  │  │  Code    │  │  API/         │   │
│  │  Tools   │  │  Interp. │  │  Database     │   │
│  └──────────┘  └──────────┘  └──────────────┘   │
└─────────────────────────────────────────────────┘
```

Each layer presents distinct threats:

| Layer | Threat Examples |
|-------|----------------|
| User Interface | Cross-agent injection, social engineering via agent |
| LLM Engine | Prompt injection, jailbreaking, adversarial inputs |
| Reasoning Engine | Multi-turn manipulation, goal hijacking |
| Memory/Context | Context poisoning, memory extraction |
| Tool Orchestrator | Tool misuse, command injection, parameter tampering |
| External Tools | Data exfiltration, API abuse, supply chain attacks |

### 3.2 Threat Actor Categories

- **External attackers**: Individuals or groups attempting to manipulate agents from outside the organization via prompt injection, crafted inputs, or social engineering.
- **Insider threats**: Malicious or negligent employees who exploit agent capabilities beyond authorized scope.
- **Compromised dependencies**: Third-party plugins, models, or tools that introduce vulnerabilities through the supply chain.
- **Competing agents**: In multi-agent systems, one agent may be compromised and used to attack others.
- **Model providers**: While generally trusted, model providers with access to telemetry could extract proprietary information from agent interactions.

### 3.3 Attack Taxonomy

The following taxonomy organizes agent-specific attacks:

**Injection Attacks**
- Direct prompt injection
- Indirect prompt injection (via tools, web content, documents)
- Multi-turn conversation manipulation
- Context window overflow attacks
- Delimiter confusion attacks

**Autonomy Exploitation**
- Tool overuse (resource exhaustion)
- Tool misuse (performing unintended actions)
- Goal misalignment (reward hacking)
- Delegation abuse (in multi-agent systems)

**Information Disclosure**
- Model inversion (reconstructing training data)
- Membership inference (determining if data was in training set)
- Side-channel attacks (timing, error messages)
- Embedding extraction (capturing vector representations)

**Supply Chain Attacks**
- Malicious plugin/model marketplace entries
- Compromised training pipelines
- Dependency confusion (typosquatting packages)
- Model weight tampering

**Integrity Attacks**
- Output manipulation (hallucination injection)
- Decision poisoning (subtle bias insertion)
- Audit trail tampering

## 4. The Agent Security Stack

A comprehensive agent security posture operates across multiple layers:

### 4.1 Layer 1: Input Security

Protecting the agent from malicious inputs at the prompt level. This includes prompt injection detection, input sanitization, and context isolation.

**Covered in**: 02-Prompt-Injection-Defenses.md

### 4.2 Layer 2: Access Control

Restricting what tools, data, and actions an agent can perform based on identity, role, and context. This includes fine-grained permission models and least-privilege principles.

**Covered in**: 03-Tool-Access-Control.md

### 4.3 Layer 3: Data Protection

Preventing data exfiltration through monitoring, filtering, rate limiting, and confidential computing techniques.

**Covered in**: 04-Data-Exfiltration-Prevention.md

### 4.4 Layer 4: Identity and Authentication

Establishing and verifying agent identity across services, including credential management, mutual TLS, and decentralized identity.

**Covered in**: 05-Agent-Authentication-and-Identity.md

### 4.5 Layer 5: Audit and Forensics

Maintaining tamper-proof logs of agent actions for incident response, compliance, and post-mortem analysis.

**Covered in**: 06-Agent-Audit-and-Forensics.md

### 4.6 Layer 6: Supply Chain Integrity

Securing the agent's dependencies including models, frameworks, plugins, and infrastructure.

**Covered in**: 07-Supply-Chain-Security-for-Agents.md

### 4.7 Layer 7: Trust and Reliability

Building confidence in agent behavior through verification, testing, human oversight, and certification.

**Covered in**: 08-Trust-Reliability-Frameworks.md

## 5. Security Principles for Agent Systems

### 5.1 Principle of Least Privilege

Every agent component should have only the minimum permissions necessary to perform its function. An agent that only needs to read calendar events should not have email send capabilities. This principle is violated more often in agent systems than in traditional software because of the convenience of giving agents broad tool access.

### 5.2 Defense in Depth

No single security control is sufficient. A robust agent security architecture layers multiple independent controls:

- Input validation → Prompt injection detection → Output verification → Audit logging

### 5.3 Human-in-the-Loop

For high-risk actions, require human approval before execution. The threshold for what constitutes "high-risk" should be configurable and context-aware.

### 5.4 Minimize Attack Surface

Each tool, plugin, or capability added to an agent increases its attack surface. Organizations should critically evaluate whether each capability is necessary and whether it can be scoped more narrowly.

### 5.5 Assume Compromise

Design agent systems assuming that at some point, the LLM will generate unexpected or malicious outputs. All safety-critical actions should have guardrails independent of the model's judgment.

### 5.6 Auditability

Every action an agent takes must be logged in a tamper-evident manner. If you cannot audit what an agent did, you cannot trust it.

## 6. Key Terminology

| Term | Definition |
|------|------------|
| **Agent** | Autonomous system that uses an LLM to reason and execute actions via tools |
| **Direct Prompt Injection** | Attacker-controlled input in the user prompt that overrides system instructions |
| **Indirect Prompt Injection** | Adversarial content in data retrieved by the agent (web, documents) that modifies behavior |
| **Tool** | A function that an agent can call (API, database, code execution, etc.) |
| **Function Calling** | The mechanism by which LLMs generate structured calls to tools |
| **Guardrails** | Runtime constraints that validate and restrict agent behavior |
| **Sandbox** | Isolated execution environment for untrusted code or actions |
| **HITL** | Human-in-the-loop: requiring human approval for certain actions |
| **TEE** | Trusted Execution Environment: hardware-level isolation |
| **SBOM** | Software Bill of Materials: inventory of software dependencies |
| **SLSA** | Supply-chain Levels for Software Artifacts: security framework |
| **SPIFFE** | Secure Production Identity Framework for Everyone |
| **DID** | Decentralized Identifier |

## 7. Regulatory and Compliance Landscape

### 7.1 Current Regulatory Environment

- **EU AI Act**: Classifies AI systems by risk level. Agent systems with autonomous tool access may fall under "high-risk" categorization, requiring conformity assessments, risk management, and human oversight.
- **NIST AI Risk Management Framework**: Provides guidance on mapping, measuring, and managing AI risks including agent-specific concerns.
- **Executive Order 14110 (US)**: Requires safety testing, red-teaming, and reporting for AI systems.
- **ISO/IEC 42001**: AI management system standard covering governance, risk assessment, and controls.

### 7.2 Emerging Standards

- **OWASP Top 10 for LLM Applications**: Includes prompt injection, supply chain, and access control vulnerabilities specific to LLM-based systems.
- **MITRE ATLAS**: Adversarial Threat Landscape for Artificial-Intelligence Systems — a knowledge base of adversary tactics and techniques for AI systems.
- **CISA's AI Security Guidance**: Guidelines for secure AI deployment including agent-specific considerations.

## 8. Document Map

This series comprises eight documents that together form a comprehensive reference for agent security and trust:

| # | Document | Focus | Key Audience |
|---|----------|-------|-------------|
| 01 | **Overview** | Threat landscape, security principles, document map | All stakeholders |
| 02 | **Prompt Injection Defenses** | Injection types, detection, prevention frameworks | Security engineers, ML engineers |
| 03 | **Tool Access Control** | Permission models, sandboxing, least privilege | Platform engineers, architects |
| 04 | **Data Exfiltration Prevention** | DLP, output filtering, confidential computing | Security analysts, compliance |
| 05 | **Agent Authentication & Identity** | Identity models, OAuth, mTLS, credential management | DevOps, platform security |
| 06 | **Agent Audit & Forensics** | Logging, audit trails, compliance | Compliance, SOC teams |
| 07 | **Supply Chain Security** | Model signing, dependency scanning, provenance | SecOps, ML engineers |
| 08 | **Trust & Reliability Frameworks** | Scoring, testing, HITL, certification | Risk management, leadership |

## 9. Risk Assessment Methodology

Organizations deploying agents should perform a structured risk assessment using the following approach:

### 9.1 Asset Identification
- What data can the agent access?
- What actions can the agent perform?
- What systems can the agent reach?

### 9.2 Threat Modeling
- Use STRIDE or PASTA methodologies adapted for agent systems.
- Identify specific agent attack paths (see Section 3.3).
- Model both accidental and malicious scenarios.

### 9.3 Control Assessment
- Evaluate existing security controls at each stack layer.
- Identify gaps specific to agent autonomy.
- Test controls through red-teaming and penetration testing.

### 9.4 Residual Risk Acceptance
- Document accepted risks and their rationale.
- Establish monitoring triggers for risk escalation.
- Define escalation procedures for agent incidents.

## 10. Conclusion

Agent security is not a static goal but an ongoing practice that must evolve alongside the capabilities of AI agents. The documents in this series provide detailed technical guidance for securing each layer of the agent stack. Organizations that invest in a comprehensive security posture — covering prompt injection defenses, access control, data protection, authentication, audit, supply chain integrity, and trust frameworks — will be best positioned to realize the benefits of autonomous agents while managing their risks.

The following seven documents dive deep into each of these areas with architecture guidance, code examples, framework recommendations, and real-world case studies.

---

**Document Information**
- Title: Agent Security and Trust — Overview
- Series: 18-Agent-Security-and-Trust
- Part: 01 of 08
- Author: AI Knowledge Base
- Last Updated: 2026-06-13
- Lines: 445

---
**See also:**
- [07 — AI Coding Agent Sustainability: Managing Hidden Costs, Code Quality Decay, and Developer Burnout](33-AI-Native-Software-Development/07-AI-Coding-Agent-Sustainability.md)
- [48 — MCP Cloud Infrastructure & Agent-as-a-Service: The Production Layer of Agentic AI](48-MCP-Cloud-Infrastructure-Agent-as-a-Service/01-Overview.md)
- [03 — Agent-Native Orchestration: LangGraph, Conductor, Mistral Workflows, Mcp-Agent](31-AI-Workflow-Orchestration-and-Durable-Execution/03-Agent-Native-Orchestration.md)
