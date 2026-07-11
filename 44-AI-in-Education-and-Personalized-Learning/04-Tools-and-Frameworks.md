# AI in Education — Tools and Frameworks

> A practitioner's catalog of AIED vendors, platforms, and open-source building blocks, plus patterns for assembling your own tutoring/learning system. Cross-references `04-RAG/`, `03-Agents/`, `30-Small-Language-Models/`, `62-Edge-AI-and-On-Device-Inference/`, `41-AI-Cost-Optimization-and-Enterprise-ROI/`.

---

## Table of Contents

1. [Landscape Overview](#1-landscape-overview)
2. [Vendor Categories](#2-vendor-categories)
3. [Generative-AI Tutoring Apps](#3-generative-ai-tutoring-apps)
4. [Institutional / LMS-AI Platforms](#4-institutional--lms-ai-platforms)
5. [Open-Source Building Blocks](#5-open-source-building-blocks)
6. [Knowledge Tracing & Analytics Libraries](#6-knowledge-tracing--analytics-libraries)
7. [RAG & Content Grounding Stacks](#7-rag--content-grounding-stacks)
8. [Voice & Multimodal Toolkits](#8-voice--multimodal-toolkits)
9. [On-Device Runtimes](#9-on-device-runtimes)
10. [Build-Your-Own: Reference Stack](#10-build-your-own-reference-stack)
11. [Evaluation & Safety Tooling](#11-evaluation--safety-tooling)
12. [Selection Criteria](#12-selection-criteria)
13. [Cross-References](#13-cross-references)

---

## 1. Landscape Overview

AIED tooling spans four buyer personas:

| Persona | Needs | Typical buy |
|---|---|---|
| **Student (consumer)** | Homework help, study buddy | Tutoring app, chatbot |
| **Teacher** | Planning, grading, insights | Add-on to LMS / standalone |
| **Institution** | Analytics, admin, compliance | Platform + SIS integration |
| **Enterprise L&D** | Upskilling, compliance training | Corporate learning suite |

> Note: Web search was unavailable during this run (no API key), so specific vendor names are kept generic/category-level to avoid fabricating current product claims. Replace with verified vendors when researching.

---

## 2. Vendor Categories

1. **AI Tutoring / Homework Helpers** — conversational solvers across subjects.
2. **Writing & Feedback Assistants** — essay feedback, originality checks.
3. **Adaptive Learning Platforms** — sequenced, mastery-based courses.
4. **Authoring / Content Gen** — lesson, quiz, and differentiation generators.
5. **Learning Analytics / Early-Warning** — at-risk prediction dashboards.
6. **Language Learning** — conversational practice bots.
7. **Accessibility Suites** — captioning, translation, reading support.
8. **Corporate L&D AI** — upskilling aligned to `34-AI-Workforce-Transformation/`.

---

## 3. Generative-AI Tutoring Apps

Characteristics:
- Chat + image input (multimodal, `50-Multimodal-AI/`).
- Step-by-step solving with explanations.
- Subscription model; consumer-grade UX.

**Build-vs-buy:** these are fine for individual study but rarely FERPA-compliant for institutional deployment — prefer grounded, logged systems for schools.

---

## 4. Institutional / LMS-AI Platforms

Integration points with Learning Management Systems (LMS):
- **LTI / API** to pull roster, grades, assignments.
- **SIS sync** for student data (`40-AI-Data-Sovereignty-and-Privacy/`).
- **Teacher console** for HITL review (`55-AI-Ethics-and-Responsible-AI/`).

```
LMS ──LTI──▶ AI Service ──▶ Teacher Console (approve/override)
 ▲              │
 └──grades──────┘
```

---

## 5. Open-Source Building Blocks

| Need | Project family | Notes |
|---|---|---|
| Orchestration | LangGraph, AutoGen, CrewAI | `03-Agents/` |
| RAG | LlamaIndex, LangChain | `04-RAG/` |
| Vector DB | Qdrant, Weaviate, pgvector | curriculum store |
| Memory | Redis, Postgres | `32-Agent-Memory-Systems/` |
| Eval | Ragas, DeepEval, custom | `52-...` |
| Serving | vLLM, TGI, llama.cpp | `48-...`, `62-...` |

---

## 6. Knowledge Tracing & Analytics Libraries

- **pyBKT** — Bayesian KT parameter estimation.
- **EduCT / EduData** — datasets and DKT tooling.
- **pandas / scikit-learn** — custom mastery models and early-warning classifiers.
- **MOOC datasets** (e.g., KDD Cup, Open University OULAD) — benchmarking.

```python
# pyBKT quickstart (conceptual)
from pybkt import BKT
model = BKT(seed_data=logs, num_fits=5)
model.fit()
print(model.params_)                 # P(L0), P(T), P(S), P(G)
pred = model.predict(next_item="fractions")
```

---

## 7. RAG & Content Grounding Stacks

For curriculum-grounded tutoring (see `03-Technical-Deep-Dive.md` §3):

```
Textbook PDF ─▶ parser ─▶ semantic chunker ─▶ embed ─▶ vector store
                                                          │
Student Q ─▶ embed ─▶ retrieve(top-k) ─▶ LLM (grounded) ─▶ answer
```

Recommended: LlamaIndex for ingestion + pgvector/Qdrant for storage + a small/fast embedder. Validate retrieved context with `52-AI-Hallucination-Detection-and-Mitigation/`.

---

## 8. Voice & Multimodal Toolkits

- **Speech:** Whisper (ASR), open TTS (e.g., Coqui/piper) — `19-Voice-AI-and-Agents/`.
- **Vision:** multimodal LLMs for whiteboard/worksheet understanding — `50-Multimodal-AI/`.
- **Language practice:** ASR + TTS loop with pronunciation scoring.

---

## 9. On-Device Runtimes

| Runtime | Use | Link |
|---|---|---|
| llama.cpp | CPU/GPU inference, GGUF | `62-Edge-AI-and-On-Device-Inference/` |
| Ollama | Easy local models | `62-...` |
| ONNX Runtime | Cross-platform | `62-...` |
| TensorRT-LLM | NVIDIA optimised | `63-GPU-Kernel-and-Inference-Performance-Engineering/` |
| MLX | Apple Silicon | `62-...` |

Great for privacy-first (`40-...`) and offline school deployments.

---

## 10. Build-Your-Own: Reference Stack

A minimal, privacy-respecting tutor:

```
Frontend (web/app)
   │  chat + image upload
   ▼
API (FastAPI) ── auth (student id)
   │
   ├─▶ RAG (LlamaIndex + pgvector) ...... curriculum
   ├─▶ Student model (Redis/Postgres) ... 32-Agent-Memory-Systems
   ├─▶ Tutor policy (LangGraph agent) ... 03-Agents
   ├─▶ LLM (vLLM on-prem OR small model on-device)
   │
   ├─▶ Guardrails (PII filter, grounding check) ... 52-, 18-
   └─▶ Observability (logs, dashboards) ........... 20-
Teacher console (HITL override, review)
```

**Why on-prem/small-model first:** compliance + cost (`41-...`) + offline (`62-...`).

---

## 11. Evaluation & Safety Tooling

- **Learning-gain study harness** — pre/post test + randomization.
- **Hallucination checker** — `52-AI-Hallucination-Detection-and-Mitigation/`.
- **Bias/fairness audit** — `55-AI-Ethics-and-Responsible-AI/`.
- **Red-teaming** — `18-Agent-Security-and-Trust/61-AI-Red-Teaming-for-LLMs/`.
- **Observability** — `20-Agent-Infrastructure-and-Observability/`.

---

## 12. Selection Criteria

| Criterion | Why it matters | Related file |
|---|---|---|
| FERPA/COPPA/GDPR compliance | Legal must | `40-...` |
| Grounding (RAG) | Reduces hallucination | `04-RAG/` |
| HITL controls | Teacher trust/oversight | `55-...` |
| Cost per student | Sustainability | `41-...` |
| On-device option | Privacy/offline | `62-...` |
| Explainability | Audit, trust | `06-Advanced/64-AI-Model-Explainability-and-XAI/` |
| Equity | Fair access | `55-...` |

---

## 13. Cross-References

- Agents: `03-Agents/`
- RAG: `04-RAG/`
- Memory: `32-Agent-Memory-Systems/`
- Small models: `30-Small-Language-Models/`
- Edge: `62-Edge-AI-and-On-Device-Inference/`
- Multimodal: `50-Multimodal-AI/`
- Voice: `19-Voice-AI-and-Agents/`
- Hallucination: `52-AI-Hallucination-Detection-and-Mitigation/`
- Security: `18-Agent-Security-and-Trust/`
- Ethics/privacy: `55-AI-Ethics-and-Responsible-AI/`, `40-AI-Data-Sovereignty-and-Privacy/`
- Observability: `20-Agent-Infrastructure-and-Observability/`
- Cost: `41-AI-Cost-Optimization-and-Enterprise-ROI/`
- GPU perf: `63-GPU-Kernel-and-Inference-Performance-Engineering/`
- Workforce: `34-AI-Workforce-Transformation/`

---

*Part of `44-AI-in-Education-and-Personalized-Learning/`. See `01-Overview.md`, `03-Technical-Deep-Dive.md`, `05-Future-Outlook.md`.*
