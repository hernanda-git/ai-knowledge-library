# AI for Cybersecurity — Tools and Frameworks

> A practical landscape of open-source and commercial tooling for AI-driven security, with runnable snippets and integration notes. Cross-referenced to the rest of the library where relevant.

## 1. Open standards & schemas

| Standard | Purpose | Link |
|---|---|---|
| OCSF | Unified security event schema | ocsf.io |
| MITRE ATT&CK | TTP knowledge base | attack.mitre.org |
| MITRE ATLAS | Adversarial ML threats | atlas.mitre.org |
| C2PA | Content provenance/watermarking | c2pa.org |
| Sigma | Generic detection rules | github.com/SigmaHQ |
| YARA | Malware pattern rules | github.com/VirusTotal/yara |

Always normalize to OCSF before AI processing — it's the lingua franca.

## 2. Open-source detection & response

- **Sigma / SigmaCLI** — write detections once, compile to Splunk/Sentinel/Elastic.
- **YARA / YARA-X** — pattern matching for malware & phishing payloads.
- **Suricata** — NDR IDS/IPS; ML on flow metadata.
- **OpenCTI** — threat-intel management; LLM extracts IOCs into it.
- **Elastic Security / OpenSearch Security** — ML anomaly jobs built in.

```bash
# compile a Sigma rule to Splunk
sigma convert -p splunk -t 'index=main' rule.yml
```

## 3. AI red-teaming toolkits

| Tool | Use |
|---|---|
| **Garak** | LLM vulnerability scanner (promptinject, jailbreaks) |
| **PyRIT** | Microsoft's AI red teaming framework |
| **MITRE CALDERA** | Automated adversary emulation (agent-based) |
| **Agentic AI Red Teaming** | Tests tool-using agents |

```bash
# Garak scan
garak --model_type huggingface --model_name meta-llama/Llama-3.1-8B \
      --probes promptinject,reveal
```

Cross-ref: `18-Agent-Security-and-Trust`.

## 4. LLM / orchestration frameworks (build your SOC agent)

- **LangChain / LangGraph** — agent graphs, tool binding (`03-Agents`).
- **LlamaIndex** — RAG over security docs (`04-RAG`).
- **Haystack** — pipelines for retrieval + generation.
- **CrewAI / AutoGen** — multi-agent investigation crews.

```python
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun

agent = create_react_agent(llm, [lookup_ip, whois, vt_lookup])
# bounded tool set; all network calls logged
```

## 5. Vector stores for RAG investigation

- **FAISS** — fast local, good for schema/index embeds.
- **Chroma** — lightweight, embedded.
- **Weaviate / Qdrant** — production, metadata filtering.
- **pgvector** — if you're already on Postgres.

See `04-RAG` and `37-AI-Native-Databases`.

## 6. Local model serving (privacy-first)

- **Ollama / vLLM / TGI** — serve open-weight models on-prem.
- **llama.cpp / MLX** — edge/CPU inference (`62-Edge-AI-and-On-Device-Inference`).
- **PrivateGPT** — self-hosted, document-grounded Q&A.

```bash
ollama run llama3.1:8b   # local triage/summarization
```

Cross-ref: `23-Local-AI-Inference-Self-Hosting`, `30-Small-Language-Models`.

## 7. Commercial platforms (2026)

| Vendor | AI capability |
|---|---|
| **CrowdStrike** | Charlotte AI (triage, detection authoring) |
| **Microsoft** | Security Copilot (Copilot across XDR, Entra, Intune) |
| **Google** | Gemini in Security Operations (Chronicle) |
| **Sentinel (MS)** | AI alert grouping, KQL assist |
| **Splunk** | MLTK, Assistant for SPL |
| **Palo Alto** | precision AI across Cortex |
| **Wiz** | AI for cloud security posture |

Buy the pipeline; build the reasoning layer locally where data is sensitive.

## 8. Deepfake / synthetic-media detection

- **Microsoft Video Authenticator** — deepfake scoring.
- **Google SynthID** — watermark detection.
- **C2PA validator libs** — provenance check.
- **Open source**: DeepFakeDetection (FaceForensics++), biological-signal analysis.

```python
from c2pa import ManifestStore
store = ManifestStore.from_file("video.mp4")
assert store.active_manifest is not None, "no provenance"
```

## 9. Phishing / email security

- **Sublime Security** — detection-as-code for email; LLM-assisted rules.
- **Tessian / Abnormal** — ML behavioral phishing defense.
- **Mxtoolbox / PhishTank** — IOC feeds.

## 10. CSPM / IaC review

- **Checkov / Trivy** — IaC scanning; pair with LLM for natural-language findings.
- **Wiz / Orca** — cloud risk with AI prioritization.
- **Semgrep** — SAST; LLM explains findings (`33-AI-Native-Software-Development`).

```bash
semgrep --config auto --json src/ > findings.json
# LLM turns findings.json into a prioritized remediation plan
```

## 11. Threat-intel automation

- **MISP** — sharing; LLM extracts IOCs from reports.
- **OpenCTI** — graph + LLM enrichment.
- **MITRE ATT&CK Navigator** — technique mapping.

```python
# extract IOCs with LLM then push to MISP
iocs = llm.invoke(f"Extract IOCs from:\n{report_text}")
misp.add_event(build_event(iocs))
```

## 12. SOAR platforms

- **Tines / Shuffle / StackStorm** — workflow engines; add an LLM step that *drafts* the plan, human approves.
- Distinction: legacy SOAR = static playbooks; AI-SOAR = generated plans per case.

## 13. Observability for your security AI

Instrument the agent like any service (`20-Agent-Infrastructure-and-Observability`):
- token cost per case
- escalation rate
- eval drift over time
- guardrail trip count

## 14. Building a detection-authoring copilot (example)

```python
from langchain_openai import ChatOpenAI  # or local via Ollama
llm = ChatOpenAI(model="gpt-4o-mini")  # or ollama
prompt = """You are a detection engineer. From this threat:
{threat}
Emit a Sigma rule (YAML) + 3 false-positives + ATT&CK tags."""
rule = llm.invoke(prompt.format(threat=threat))
# write to repo, open PR, CI validates
```

## 15. Building a triage agent (example)

```python
from pydantic import BaseModel
class Verdict(BaseModel):
    severity: str
    summary: str
    techniques: list[str]
    confidence: float
    needs_human: bool

def triage(alert, ctx):
    v = llm.with_structured_output(Verdict).invoke(
        system=SYSTEM_NO_INJECT, user=f"{ctx}\n{alert}"
    )
    v.techniques = [t for t in v.techniques if t in VALID_ATTACK_IDS]
    return v
```

## 16. Guardrail libraries

- **Guardrails AI / NeMo Guardrails** — output schema + topic constraints.
- **Lakera / HiddenLayer** — LLM security specialized.
- **OWASP LLM Top 10** — checklist (`18-Agent-Security-and-Trust`).

## 17. Evaluation harnesses

- Internal golden set (incident replay).
- **Garak / PyRIT** for model-level.
- **DeepEval / Ragas** for RAG faithfulness (`04-RAG`, `52-Hallucination-Detection-and-Mitigation`).

## 18. Data & feature stores

- **Feature store** (Feast) for ML features.
- **Time-series DB** (Timescale, Influx) for lookback.
- **Graph DB** (Neo4j) for entity correlation.

## 19. Privacy / compliance tooling

- **Microsoft Presidio** — PII redaction before LLM.
- **Differential privacy libs** (Opacus) for shared intel.
- Tenant-isolated endpoints for cloud LLMs.

```python
from presidio_analyzer import AnalyzerEngine
analyzer = AnalyzerEngine()
# redact before any LLM call on logs containing PII
```

## 20. Small-model options for edge SOC

- Llama-3.1-8B, Phi-3/3.5, Gemma-2-9B, Qwen2.5-7B — fine for triage/summarization on a single GPU (`30-Small-Language-Models`).

## 21. Integration blueprint

```
[EDR/NDR/FW/Cloud] --(OCSF)--> [Normalizer]
        --> [ML pre-filter] --> [Vector store + Graph]
        --> [LLM triage agent + Guardrails] --> [SOAR/Cases]
        --> [Analyst UI]  <-- [Observability]
```

## 22. Choosing what to build vs buy

| Build | Buy |
|---|---|
| Triage/summarization layer (local) | Telemetry pipeline (SIEM/XDR) |
| Detection-authoring copilot | Endpoint/network sensors |
| Red-team harness | Managed threat intel |
| Guardrails | — |

## 23. Cost reference (illustrative)

| Design | Relative cost |
|---|---|
| Every alert → frontier LLM | 🔴 100x |
| ML pre-filter → top 1% → local 8B | 🟢 1x |
| ML pre-filter → top 5% → mid LLM + cache | 🟡 5x |

## 24. Vendor lock-in caution

Prefer OCSF + open models so you can swap SIEM/LLM without rewrites. Keep detections in git.

## 25. Quick start stack (open, free)

Sigma + YARA + Suricata + OpenCTI + Ollama(Llama-3.1-8B) + LangGraph + Chroma + Garak. Enough to pilot a local-first SOC agent.

## 26. Summary

The tooling is mature and mostly open. The differentiator is *integration* and *guardrails*, not availability. Build the reasoning layer locally; buy the sensing pipeline; red-team everything. Next: where this is headed.

## 27. Open-source SIEM alternatives

- **Wazuh** — XDR with ML rules; good on-prem starting point.
- **Security Onion** — network monitoring + Suricata + Zeek.
- **Grafana Loki + Tempo** — log/trace correlation feeding your own ML.

## 28. Model registries & versioning

Track model versions like code (`64-Model-Fine-Tuning-and-Post-Training`): MLflow, Hugging Face Hub, Ollama tags. Pin in production; audit which version decided what.

## 29. Prompt & example stores

Keep triage prompts + few-shot examples in git; review like code. Pair with `32-Agent-Memory-Systems` for learned feedback.

## 30. Synthetic-data generators for testing

Use LLMs + `51-Synthetic-Data-Generation` to fabricate incidents for eval without leaking real PII. Label synthetic; keep a human-verified core.

## 31. CI for security AI

```
on: push
jobs:
  eval:
    - sigma lint
    - garak scan (model)
    - replay golden set -> assert recall>=0.9, FP<=0.05
    - block merge on regression
```

## 32. Edge / on-device options

`62-Edge-AI-and-On-Device-Inference`: run triage on a branch office appliance; only aggregate IOCs leave the site (`40-AI-Data-Sovereignty-and-Privacy`).

## 33. Multimodal tooling

`50-Multimodal-AI`: CLIP-style encoders for brand/image Deepfake; Whisper for vishing detection; vision transformers for document forgery.

## 34. Choosing a local model in practice

For an 8–14B triage/summarization model on one GPU: Llama-3.1-8B (general), Phi-3.5 (small, strong reasoning), Qwen2.5-7B (multilingual). Benchmark on *your* golden set — published scores don't transfer (`30-Small-Language-Models`).

## 35. Integration anti-patterns

- Vendor SIEM that won't export OCSF/Sigma → lock-in trap.
- Cloud LLM with no tenant isolation → data-risk.
- Tool with no published eval → unproven.
- "Fully autonomous" marketing → ignore; demand bounded scope.
