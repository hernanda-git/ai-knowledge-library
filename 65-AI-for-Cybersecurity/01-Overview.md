# AI for Cybersecurity — Overview

> How artificial intelligence is reshaping offensive and defensive security: autonomous SOCs, AI-assisted threat detection, vulnerability discovery, Deepfake/phishing detection, and AI red teaming. A practical map of the 2026 landscape, its architectures, risks, and how it relates to the rest of this library.

## 1. Why this category exists

Cybersecurity is undergoing its largest structural shift since the move to cloud. The driver is not a single new algorithm but a convergence:

1. **Attackers adopted generative AI first.** Phishing is now multilingual, grammar-perfect, and personalized at scale. Deepfake voice/Video is cheap enough for vishing. Malware writes itself. The asymmetry that defenders relied on (attacker effort > payoff) collapsed.
2. **Data volume outran humans.** A mid-size SOC ingests 10k–100k+ alerts/day. Pure human triage is mathematically impossible; most alerts are ignored or auto-closed.
3. **Foundation models became cheap and good enough.** LLMs, encoders, and vision models can parse logs, write detections, correlate signals, and converse with analysts.

The result: **AI for Cybersecurity** is the discipline of applying ML/LLMs to protect systems — distinct from category `18-Agent-Security-and-Trust` (which is about *securing the agents themselves*). This category is about *using* AI as a defensive and offensive security tool.

See also: `20-Agent-Infrastructure-and-Observability` (telemetry plumbing), `52-Hallucination-Detection-and-Mitigation` (relevant when LLMs write detections), `40-AI-Data-Sovereignty-and-Privacy` (data handling constraints), `55-AI-Ethics-and-Responsible-AI` (bias/dual-use concerns).

## 2. Scope of the category

| Sub-area | What it covers | Library cross-ref |
|---|---|---|
| Threat detection | Anomaly detection, UEBA, log/network ML | `20-Agent-Infrastructure-and-Observability` |
| Autonomous SOC | Alert triage, SOAR+LLM, case management | `31-AI-Workflow-Orchestration-and-Durable-Execution` |
| Vulnerability discovery | SAST/DAST with LLMs, fuzzing, patching | `33-AI-Native-Software-Development` |
| AI red teaming | Adversarial testing of models/systems | `18-Agent-Security-and-Trust` |
| Content authenticity | Deepfake/phishing/synth-media detection | `50-Multimodal-AI` |
| Threat intel | Automated IOC extraction, attribution | `10-Industry` |
| Security copilots | Analyst assistants, natural-language query | `02-LLMs` |

## 3. The core problem: the alert funnel

```
Raw telemetry (EDR, NDR, firewall, cloud, IDP)
        │
        ▼
  [Normalization & enrichment]  ← embeddings, entity resolution
        │
        ▼
  [Detection layer]  ← rules + ML + LLM semantic detectors
        │
        ▼
  [Triage/ranking]   ← severity scoring, dedup, correlation
        │
        ▼
  [Investigation]    ← graph traversal, LLM hypothesis gen
        │
        ▼
  [Response]         ← SOAR playbooks, auto-contain
        │
        ▼
   Analyst (human-in-the-loop for high stakes)
```

The LLM's job is mostly in the middle: turning noisy, heterogeneous signals into a small set of *explained, prioritized* cases a human can act on in minutes instead of hours.

## 4. Two paradigms: predictive ML vs generative AI

| Dimension | Classical ML security | Generative-AI security |
|---|---|---|
| Training data | Labeled attacks, flows | General corpora + fine-tuned on sec data |
| Strength | Precise, fast, cheap at scale | Reasoning, summarization, nuance |
| Weakness | Brittle to novel attacks | Hallucination, latency, cost |
| Typical use | Malware classification, anomaly | Triage, reporting, query, red team |
| Latency | ms | seconds–minutes |

Modern stacks use **both**: fast ML for first-pass filtering, LLM for reasoning over the survivors.

## 5. Key architectures

### 5.1 Detection-as-Code with LLM assist
Security engineers describe intent in English; an LLM emits a detection rule (Sigma, YARA, KQL, SPL). The rule is version-controlled (see `08-Reference`).

### 5.2 Retrieval-augmented investigation
The LLM is grounded in your telemetry via RAG (see `04-RAG`): it can answer "show me every login from a new device in the last 24h that then touched a domain controller" by generating and running a query, then citing the logs.

### 5.3 Agentic SOC
A multi-step agent (see `03-Agents`) plans an investigation: enrich IP → pull related alerts → check MITRE ATT&CK mapping → propose containment. Bounded by tools and guardrails (see `18-Agent-Security-and-Trust`).

## 6. The MITRE ATT&CK anchor

Almost every AI security product maps findings to [MITRE ATT&CK](https://attack.mitre.org). An LLM that emits a narrative should also emit technique IDs (T1059, T1566, …). This is the lingua franca between tools and analysts — treat it as a required output schema.

## 7. Build vs buy (2026 reality)

| Need | Build | Buy |
|---|---|---|
| Log parsing/normalization | Rarely | Splunk, Sentinel, CrowdStrike |
| LLM triage layer | Common (open-weight) | Microsoft Security Copilot, Google Gemini in SecOps |
| Custom detection dev | Yes (your environment is unique) | Hybrid |
| Red teaming | Yes (open tools) | Pentest vendors + AI |

Most orgs land on **buy the pipeline, build the reasoning layer** with an open-weight model (see `30-Small-Language-Models`, `23-Local-AI-Inference-Self-Hosting`) to keep telemetry on-prem.

## 8. Risks introduced by AI itself

- **Hallucinated IOCs**: an LLM may invent an IP/domain. Never auto-action on an unverified model output (see `52-Hallucination-Detection-and-Mitigation`).
- **Prompt injection via logs**: attacker crafts a log line that hijacks the investigating agent (see `18-Agent-Security-and-Trust`).
- **Data exfiltration**: sending logs to a public LLM. Use local or tenant-isolated models.
- **Adversarial drift**: attackers learn your detector and craft evasions.

## 9. Regulatory pressure

Several regimes now *expect* AI-assisted monitoring (e.g., financial-sector SES, NIS2 incident timelines). At the same time, AI use in security is caught by `21-AI-Regulation-Antitrust` and `55-AI-Ethics-and-Responsible-AI` (bias in surveillance, dual-use). Document model version, training data, and decisions for auditability.

## 10. Maturity model

| Level | Behavior |
|---|---|
| 0 | Manual, rule-only |
| 1 | ML anomaly detection added |
| 2 | LLM summarization of alerts |
| 3 | LLM triage + suggested response |
| 4 | Agentic investigation, human approves |
| 5 | Semi-autonomous response for low-risk, full auto for contained blast radius |

Most enterprises in 2026 sit at 2–3.

## 11. How to read the rest of this category

- `02-Core-Topics.md` — threat detection, SOC automation, vuln discovery, content authenticity.
- `03-Technical-Deep-Dive.md` — embeddings, detection engineering, RAG investigation, agentic SOC, eval.
- `04-Tools-and-Frameworks.md` — open-source and commercial landscape, code samples.
- `05-Future-Outlook.md` — autonomous defense, AI-vs-AI, standards, 12-month bets.

## 12. Quick glossary

- **UEBA**: User and Entity Behavior Analytics.
- **NDR/EDR/XDR**: Network / Endpoint / Extended Detection & Response.
- **SOAR**: Security Orchestration, Automation, and Response.
- **IOC**: Indicator of Compromise (IP, hash, domain).
- **TTP**: Tactics, Techniques, and Procedures.
- **Purple teaming**: attack + defense exercise combined.

## 13. First principles for practitioners

1. AI reduces *noise*, it does not remove *judgment*.
2. Every model output that drives an action must be **verifiable and reversible**.
3. Keep a human in the loop for anything with blast radius > one asset.
4. Measure: mean-time-to-detect (MTTD), mean-time-to-respond (MTTR), false-positive rate, analyst hours saved.
5. Attack your own AI (see `18-Agent-Security-and-Trust`).

## 14. Relationship to agent security

This category is the *user* of agents; `18-Agent-Security-and-Trust` is about *hardening* them. A secure SOC agent needs: tool allow-lists, output schema validation, log-injection defenses, and rollback. Read both.

## 15. Common failure modes

- Treating the LLM as a classifier (it's not; use ML for that).
- Skipping eval (see `03` deep dive) and shipping a triage agent that misses critical alerts.
- Under-investing in normalization — garbage telemetry, garbage reasoning.
- No feedback loop from analysts back into the model / prompts.

## 16. A note on benchmarks

Public security benchmarks (MITRE EMBER for malware, Cylab, SecEval) exist but rarely match your environment. Build an internal golden set of past incidents and replay them.

## 17. Data minimization

Security data is among the most sensitive. Prefer on-device inference (`23-Local-AI-Inference-Self-Hosting`, `62-Edge-AI-and-On-Device-Inference`) and tenant isolation. Redact PII before any LLM call.

## 18. Cost reality

A naive "send every alert to GPT" design bankrupts you. The pattern that works: cheap ML pre-filter → only top-N cases to LLM → cache aggressively → use small local models (`30-Small-Language-Models`) for summarization.

## 19. Team shape

You need: detection engineers, ML engineers, a threat-intel function, and a security-AI red team. The "AI security engineer" role blends all four.

## 20. Where this is headed

Toward *autonomous defense*: systems that detect, decide, and contain at machine speed for well-bounded risk, while escalating ambiguous cases to humans. The 2026 differentiator is not model quality — it's **data plumbing, eval, and guardrails**.

## 21. References & further reading

- MITRE ATT&CK — https://attack.mitre.org
- OWASP Top 10 for LLM Applications — https://owasp.org/www-project-top-10-for-large-language-model-applications/
- MITRE ATLAS (adversarial ML) — https://atlas.mitre.org
- NIST AI Risk Management Framework — https://www.nist.gov/itl/ai-risk-management-framework

## 22. Checklist to start

- [ ] Inventory telemetry sources and normalize
- [ ] Stand up a cheap ML anomaly layer
- [ ] Build an internal incident golden set
- [ ] Pilot LLM summarization on a sample of alerts
- [ ] Define human-in-the-loop boundaries
- [ ] Red-team the agent before production

## 23. Cross-category reading path

New to the library? Read in this order: `02-LLMs` → `04-RAG` → `03-Agents` → `18-Agent-Security-and-Trust` → this category → `20-Agent-Infrastructure-and-Observability`.

## 24. Myths

- *"AI will replace my SOC team."* No — it replaces the part of the job that was impossible anyway (triage of 100k alerts).
- *"LLMs are magic detectors."* They're reasoners over context, not classifiers.
- *"Local models are too weak."* For summarization/triage, a 7–14B model is plenty (`30-Small-Language-Models`).

## 25. Adoption blockers (and how to clear them)

| Blocker | Fix |
|---|---|
| "We don't trust the model" | Start read-only (summarization), prove on golden set |
| "Our data is too sensitive" | Local models (`23`), redaction (`Presidio`) |
| "Alerts are unmanageable" | Normalize + ML pre-filter first |
| "No ML talent" | Buy pipeline, build small reasoning layer |
| "Audit/compliance fear" | Log model version + decisions; human approvals |

## 26. Build a 30-day pilot

- Week 1: OCSF normalize one telemetry source; build a 50-incident golden set.
- Week 2: Stand up an ML anomaly pre-filter; measure FP/recall.
- Week 3: Add LLM summarization on top-1% alerts via local 8B model.
- Week 4: Add a guardrailed triage agent; red-team with Garak; report to stakeholders.

## 27. Operating model

Security-AI is a product, not a project. Assign an owner, a roadmap, weekly eval reviews, and an incident path for the AI itself (`20-Agent-Infrastructure-and-Observability`).

## 28. Vendor questions to ask

- Can detections export as open Sigma/OCSF?
- Is my telemetry tenant-isolated?
- What model version is running, and can I pin it?
- How do you defend the agent from prompt injection in logs?
- What eval evidence do you publish?

## 29. Training your team

- Detection engineers learn LLM prompting + RAG.
- Analysts learn to critique model output (keep agency).
- Red team learns adversarial ML (`18-Agent-Security-and-Trust`).
- Leadership learns the metrics (MTTD/MTTR/FP).

## 30. Ethics guardrails

Security AI can become surveillance AI. Bound it: minimize PII, peer-review queries, log who asked what, respect `55-AI-Ethics-and-Responsible-AI` and `27-AI-in-HR-and-Recruiting` privacy norms.

## 31. The "no AI" baseline comparison

A controlled A/B (AI triage vs human-only on the same alert sample) is the most persuasive internal artifact. Publish MTTD/MTTR deltas.

## 32. Common metrics pitfalls

- Optimizing recall while FP explodes burns analysts.
- Ignoring "critical missed" — one missed ransomware beats 1000 FPs.
- Not counting analyst hours saved (the real ROI).

## 33. Relationship to AI cost optimization

Security AI is a top token-spend area. Apply `41-AI-Cost-Optimization-and-Enterprise-ROI`: pre-filter, cache, small models, batch.

## 34. Incident-response comms

LLMs draft stakeholder updates fast — but a human owns the message. Keep a template library; never auto-send external comms.

## 35. Knowledge management

Store runbooks, past cases, and model behaviors in the library itself (`08-Reference`, `15-Community-Resources-Templates`). The SOC agent should retrieve them at runtime (`04-RAG`, `32-Agent-Memory-Systems`).

## 36. Scaling across business units

Once the pattern works in one BU, replicate via shared OCSF schema + shared golden set + per-BU fine-tune (`64-Model-Fine-Tuning-and-Post-Training`).

## 37. Summary

AI for Cybersecurity is the imperative application of ML+LLMs to a domain drowning in data and attacked by AI-wielding adversaries. The winning 2026 pattern is hybrid (fast ML + reasoned LLM), local-first for data sensitivity, eval-driven, and human-bounded. Start boring, prove with a golden set, red-team relentlessly, and scale via shared plumbing. The rest of this category goes deep.

## 38. Pre-flight self-assessment

Score your org 0–5 on: telemetry normalization, ML pre-filter, LLM summarization, guardrailed triage, autonomous bounded-response, red-team-in-CI. Anything under 3 is your next investment.

## 39. One-page exec brief

"AI lets our SOC handle 10x alert volume at the same headcount by auto-summarizing and pre-triaging, while keeping humans on every high-stakes decision. We start local and private, prove value on past incidents, then expand. Risk is managed by guardrails and continuous red-teaming, not trust."
