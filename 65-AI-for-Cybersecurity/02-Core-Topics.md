# AI for Cybersecurity — Core Topics

> The five load-bearing domains of AI-driven security: threat detection, autonomous SOC, vulnerability discovery, AI red teaming, and content authenticity. Each with how it works, 2026 tooling, and code sketches.

## 1. Threat Detection (the ML backbone)

Before any LLM, you need a fast, reliable first-pass filter. Classical ML still dominates here because it is precise, cheap, and explainable.

### 1.1 Anomaly detection on flows
Train an isolation forest / autoencoder on normal host/network behavior; score deviations.

```python
from sklearn.ensemble import IsolationForest
import numpy as np

# features: bytes_in, bytes_out, conn_count, dst_unique, hour
X = np.load("netflow_features.npy")
clf = IsolationForest(contamination=0.01, random_state=42)
clf.fit(X)
scores = clf.decision_function(X)   # higher = more normal
anomalies = X[scores < clf.threshold_]
```

### 1.2 UEBA (User & Entity Behavior Analytics)
Per-entity baselines (peer-group, time-of-day). Flag impossible travel, off-hours admin, mass file access.

### 1.3 Signature + ML hybrid
YARA / Sigma catch known; ML catches novel. Route both into the same pipeline.

| Approach | Precision | Recall (novel) | Latency | Cost |
|---|---|---|---|---|
| Signatures | High | Low | ms | Low |
| ML anomaly | Med | High | ms | Low |
| LLM semantic | Med | Med | s | High |

## 2. Autonomous SOC (where LLMs earn their keep)

### 2.1 The triage agent
Input: a normalized alert + its correlated telemetry. Output: a structured case.

```python
from pydantic import BaseModel

class TriageVerdict(BaseModel):
    severity: str                  # critical|high|medium|low
    summary: str
    attack_techniques: list[str]   # MITRE ATT&CK IDs
    confidence: float
    recommended_actions: list[str]
    needs_human: bool

# system prompt grounds the model in your runbook + ATT&CK
verdict = llm.with_structured_output(TriageVerdict).invoke(context)
```

### 2.2 SOAR + LLM
Legacy SOAR is brittle (hard-coded playbooks). LLM-SOAR *generates* the response plan per case, then executes only approved steps.

### 2.3 Case management
The agent maintains a case thread: hypotheses, evidence, actions taken — an auditable narrative for the analyst (see `31-AI-Workflow-Orchestration-and-Durable-Execution` for durable execution).

### 2.4 Human-in-the-loop boundaries
```python
AUTO_ACTIONS = {"enrich", "lookup", "snapshot"}
NEEDS_APPROVAL = {"isolate_host", "block_ip", "disable_account"}

def execute(plan):
    for step in plan:
        if step.kind in NEEDS_APPROVAL:
            request_approval(step)   # pauses durable workflow
        else:
            run(step)
```

## 3. Vulnerability Discovery & Patching

### 3.1 LLM-assisted SAST
Feed a function + CWE catalog; ask for a vuln explanation + fix.

```python
prompt = f"""Review this code for CWE-79 (XSS) and CWE-89 (SQLi).
Return JSON: {{findings: [{{line, cwe, severity, fix}}]}}
Code:
{source}"""
```

### 3.2 AI-guided fuzzing
LLMs generate semantically valid inputs near parser boundaries (see `33-AI-Native-Software-Development`).

### 3.3 Auto-patch proposals
Model emits a diff; CI runs tests; human merges. Risk: the patch may be wrong — require tests to pass and a security review.

### 3.4 SBOM + AI
Cross-reference your SBOM against CVE feeds automatically; LLM drafts the mitigation note.

## 4. AI Red Teaming (attack your own AI)

Distinct from `18-Agent-Security-and-Trust` (which is defensive hardening); red teaming is *offensive* testing.

### 4.1 What to test
- Prompt injection (see `18`)
- Jailbreaks / policy bypass
- Data extraction (membership inference)
- Tool misuse by an agent
- Log/email injection that hijacks an investigating agent

### 4.2 Tooling
PyRIT, Garak, Agentic AI Red Teaming, MITRE ATLAS.

```python
# Garak: scan a model for known failure modes
# garak --model_type openai --model_name gpt-4 --probes promptinject
```

### 4.3 Loop
Continuous red team → fix → re-test. Treat it like fuzzing for behavior.

## 5. Content Authenticity (Deepfake / Phishing / Synth-media)

### 5.1 Phishing detection
Classify by linguistic markers + URL features + sender graph. LLM adds nuance for novel lures.

```python
def score_email(msg):
    url_feats = extract_url_features(msg.urls)
    ml = phishing_clf.predict_proba([url_feats])[0][1]
    sem = llm.invoke(f"Is this a phishing attempt? {msg.text}")  # guardrail-bounded
    return 0.7*ml + 0.3*sem  # ensemble
```

### 5.2 Deepfake detection
Vision models flag temporal inconsistencies, blending artifacts, lip-sync mismatch. Watermark detection (C2PA) for provenance.

### 5.3 Provenance standards
C2PA, watermarking (SynthID, invisible markers). Verify provenance before trusting media in investigations.

### 5.4 Vishing
Voice-clone detection by acoustic artifact analysis; challenge-response protocols for high-value transfers.

## 6. Threat Intelligence Automation

- Extract IOCs from reports/threads automatically (NER over text).
- Map to ATT&CK.
- Push to detection pipeline.

```python
# LLM extracts structured IOCs
ioc = llm.invoke(f"Extract IOCs (ip, domain, hash, url) from:\n{report}")
# -> feed into blocklist + detection-as-code
```

## 7. Security Copilots

Natural-language query over your estate: "who accessed the payroll bucket last weekend?" The copilot generates and runs the query, returns cited evidence. Ground it with RAG (`04-RAG`) over your schema docs.

## 8. Network detection (NDR)

ML on packet metadata (not payload) for lateral movement, C2 beacons (periodic low-volume), DNS tunneling.

| Signal | Indicates |
|---|---|
| Beaconing | C2 |
| East-west spikes | Lateral movement |
| DNS TXT volume | Tunneling |
| New TLS JA3 | Unknown tooling |

## 9. Cloud security (CSPM)

LLMs review IaC (Terraform) for misconfig: public buckets, over-permissive IAM, missing encryption. Map to CIS benchmarks.

## 10. Identity & auth

- Impossible travel detection
- Compromised-credential signaling (breach corpus match)
- Risk-based step-up auth driven by a risk score (ML)

## 11. Ransomware early warning

Behavioral: mass file rename/encrypt, shadow-copy deletion, volume shadow changes. ML on file-event streams catches it pre-exfiltration.

## 12. Insider threat

UEBA + NLP over communications (privacy-bounded, see `27-AI-in-HR-and-Recruiting` for ethics) flags data hoarding, unusual access before departure.

## 13. Detection engineering with LLMs

| Task | Prompt pattern |
|---|---|
| Sigma from intent | "Write a Sigma rule detecting …" |
| Explain alert | "Explain this Splunk result to a tier-1 analyst" |
| False-positive triage | "Given context X, is this a FP? Why?" |
| Hypothesis gen | "What 3 techniques could explain these IOCs?" |

## 14. The ensemble pattern (most important)

Never rely on one model. Ensemble:
```
score = w1*signature + w2*ml_anomaly + w3*llm_semantic + w4*graph
```
Tune weights per environment. The LLM is usually the smallest weight — it adds reasoning, not precision.

## 15. Evaluation you must run

- Replay past incidents; measure recall at fixed FP budget.
- Red-team the triage agent with poisoned logs.
- Ablation: does the LLM actually help vs ML-only? (Often it helps *explain*, not *detect*.)

## 16. Data pipeline essentials

- Normalize to OCSF (Open Cybersecurity Schema Framework).
- Entity resolution (merge identities across sources).
- Time-series store for fast lookback.
- Feature store for ML.

## 17. Privacy constraints

Security logs contain PII. Before LLM calls: redact emails, tokens, secrets. Prefer local models (`23-Local-AI-Inference-Self-Hosting`) for sensitive telemetry.

## 18. Cost levers

- Pre-filter aggressively (ML) before LLM.
- Cache identical alerts (dedup).
- Use small models (`30-Small-Language-Models`) for summarization.
- Batch where latency allows.

## 19. Common anti-patterns

- LLM-as-classifier (use ML).
- Auto-blocking on model output (never, unverified).
- No eval set (ship blind).
- Trusting model IOCs without verification.

## 20. Metrics dashboard

| Metric | Target |
|---|---|
| MTTD | ↓ 50% vs baseline |
| MTTR | ↓ 40% |
| FP rate | < 5% |
| Analyst hours saved | +30% |
| Critical missed | 0 tolerated |

## 21. Tooling shortlist (detailed in 04)

Open: Sigma, YARA, Suricata, MITRE CALDERA, Garak, PyRIT, OpenCTI.
Commercial: CrowdStrike, Microsoft Security Copilot, Google SecOps (Gemini), Sentinel, Splunk.

## 22. Integration with agent infra

The SOC agent is just another agent — give it the same observability (`20-Agent-Infrastructure-and-Observability`) and guardrails (`18-Agent-Security-and-Trust`).

## 23. Incident-response acceleration

LLMs draft the IR report, timeline, and stakeholder comms in minutes. Keep a human editor; log the model version for audit.

## 24. Summary

The five core topics form a stack: ML detects, LLM reasons/triages, agents investigate, red teams attack the AI, and authenticity tooling defends the human-facing surface. None works alone; the value is in the *ensemble* and the *plumbing*. Next: the technical deep dive.

## 25. API security

LLMs infer from API traffic: abnormal call volume, token stuffing, BOLA/IDOR patterns. Pair with `33-AI-Native-Software-Development` for securing the apps themselves.

## 26. Fraud detection

Transaction graphs + ML score risk in real time; LLM explains declines to analysts and customers. Ensemble of gradient-boosted trees + GNN + LLM narrative.

## 27. OT / IoT security

Industrial telemetry is low-frequency but high-consequence. Anomaly detection on SCADA; LLM maps to ATT&CK for ICS. Latency tolerance is higher; privacy is extreme (`62-Edge-AI-and-On-Device-Inference`).

## 28. Secrets & data leakage

LLM scans repos, logs, and tickets for leaked keys/tokens; auto-rotate via integration. Complements `40-AI-Data-Sovereignty-and-Privacy`.

## 29. Supply-chain security

SBOM + CVE + LLM impact analysis: "if log4j-class bug hits X, what breaks?" Provenance checks on dependencies (`38-AI-Supply-Chain-and-Chip-Design`).

## 30. Brand / external threat surface

LLM monitors dark web, fake domains, impersonation; alerts trust & safety. Multimodal (`50-Multimodal-AI`) for logo/brand Deepfakes.

## 31. The human-facing surface

Phishing, vishing, Deepfake, and social engineering target people, not perimeter. Content-authenticity tooling + training is the last line. See `55-AI-Ethics-and-Responsible-AI`.

## 32. Detection lifecycle

```
idea → Sigma/YARA draft (LLM) → PR → CI lint/test → SIEM → eval → tune
```
Treat detections as code (`08-Reference`, `19-Voice-AI-and-Agents` for call-surface).

## 33. Cross-team handoffs

SOC → IR → legal → comms. The agent should hand off structured cases, not raw alerts. Durable workflows (`31-AI-Workflow-Orchestration-and-Durable-Execution`) carry state across teams.

## 34. Measuring triage quality weekly

Track per-week: recall on golden set, FP rate, escalation rate, analyst override rate. A rising override rate signals prompt/model drift — investigate before it becomes a miss.

## 35. The "explain, don't decide" principle

For the first 90 days, the LLM should *only* explain and rank, never act. Earn the right to act with 90 days of clean eval evidence. This single rule prevents most catastrophic early failures.
