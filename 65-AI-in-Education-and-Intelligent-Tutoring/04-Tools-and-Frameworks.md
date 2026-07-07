# AI in Education — Tools and Frameworks

> A practical buyer's/build guide: hosted tutoring platforms, open models fine-tuned for
> education, SDKs, content-authoring tools, and the open-source building blocks you can
> assemble into your own tutor (see `03-Technical-Deep-Dive.md`). Directional as of 2026;
> verify current capabilities before procurement.

## 1. Hosted Tutoring & Learning Platforms

| Platform | Form | Strength | Notes |
|----------|------|----------|-------|
| Khanmigo (Khan Academy) | Generative tutor | Socratic, safe, curricular | GPT-4-class, guarded |
| Duolingo Max | Language tutor | Engagement, streaks | LLM explanations |
| Google LearnLM | Model + tools | Multimodal, grounded | Education-tuned model family |
| Coursera / edX AI features | Course copilots | Scale, content library | Summaries, Q&A |
| Quizlet Q-Chat / Magic Notes | Study aids | Recall practice | Flashcard-native |
| Pearson / McGraw Hill AI | Textbook-integrated | Aligned to materials | RAG over owned content |
| Squirrel AI (China) | Adaptive K-12 | Large KT dataset | High-volume ITS |

> None of these is an endorsement; listed to map the landscape. Procurement must weigh
> data residency (`40-AI-Data-Sovereignty`) and minors' privacy.

---

## 2. Education-Tuned Models

General LLMs can tutor, but education-tuned models improve safety and pedagogy:

- **LearnLM** (Google) — explicitly tuned for learning, with "coach" behavior.
- **Instruct-style fine-tunes** of Llama/Mistral on dialogue + Socratic datasets.
- **Small models** (`30-Small-Language-Models`) for on-device tutoring (privacy, cost)
  — e.g., a 3–8B model running in a school data center (`23-Local-AI-Inference`).

### Fine-tuning recipe (sketch)

```python
# LoRA fine-tune on (question, socratic_hint) pairs
from peft import LoraConfig
cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj"])
train("tutor_sft_dataset.jsonl", model="mistral-7b", peft=cfg,
      max_seq_len=2048)
# Then DPO against "gives answer directly" negatives to enforce scaffolding
```

---

## 3. Knowledge Tracing & Adaptive Engines (Open Source)

| Project | Type | Language |
|---------|------|----------|
| pyBKT | Bayesian KT | Python |
| DKT / AKT reference impls | Deep KT | PyTorch |
| LearnSphere / DataShop | Learning analytics warehouse | — |
| Oppia | Interactive lesson authoring + KT | Python/Angular |
| Open edX | LMS with adaptive blocks | Python |

```bash
pip install pybkt
from pybkt import BKTModel
m = BKTModel(); m.fit(sequence_df)   # columns: learner, skill, correct
print(m.predict(learner="s1", skill="algebra"))
```

---

## 4. RAG & Curriculum Grounding (`04-RAG`)

| Tool | Use |
|------|-----|
| LangChain / LlamaIndex | Curriculum ingestion + retrieval |
| Chroma / Qdrant / pgvector | Vector store (`37-AI-Native-Databases`) |
| Unstructured / MarkItDown | Parse PDF textbooks, slides |

Pattern: chunk by learning objective, attach `kc_id`/`difficulty` metadata, filter by
grade/subject at query time (see `03-Technical-Deep-Dive.md` §1).

---

## 5. Agent Orchestration (`03-Agents`, `31-Workflow`)

| Tool | Role |
|------|------|
| LangGraph | Stateful tutor graph, hint-ladder control |
| CrewAI / AutoGen | Multi-agent (tutor + grader + planner) |
| Temporal / Inngest | Durable session state (`54-Agent-State`) |
| MCP (`48-MCP-Cloud-Infrastructure`) | Expose curriculum/grading as tools |

```python
# LangGraph sketch: tutor loop with scaffolding gate
from langgraph.graph import StateGraph
g = StateGraph(TutorState)
g.add_node("retrieve", retrieve_curriculum)
g.add_node("hint", scaffolding_controlled_hint)
g.add_node("grade", update_mastery)
g.add_edge("retrieve", "hint")
g.add_conditional_edges("hint", needs_more_help, {"yes":"retrieve","no":"grade"})
```

---

## 6. Assessment & Execution Sandboxes

| Need | Tool |
|------|------|
| Code execution | gVisor / Firecracker / Docker sandbox |
| Math equivalence | SymPy, Mathematica/WolframAlpha API |
| Rubric grading | Custom LLM + JSON schema validation |
| Plagiarism / authorship | `43-AI-Data-Provenance` tooling |

Never execute learner-submitted code on the host — see `56-MLOps-and-AI-Platform-Engineering`.

---

## 7. Guardrails & Safety (`18-Agent-Security`, `61-AI-Red-Teaming`)

| Tool | Use |
|------|-----|
| Lakera / Guardrails AI | Input/output filtering for minors |
| Self-hosted classifiers | Toxicity, off-topic, PII (`40-Data-Sovereignty`) |
| Red-teaming harness | Probe for answer-leaking, jailbreaks (`61`) |

Education-specific red-team cases:
- "Just give me the answer" loops.
- Prompt injection via uploaded homework image.
- Requests for non-educational content.

---

## 8. Accessibility Tooling (`50-Multimodal-AI`)

| Need | Tool |
|------|------|
| Speech-to-text | Whisper, Azure Speech |
| Text-to-speech | ElevenLabs, Azure TTS |
| Captioning | Live caption APIs |
| OCR / handwriting | TrOCR, Gemini/Vision models |
| Reading-level simplification | LLM prompt + readability metrics |

---

## 9. Observability & Analytics (`20-Agent-Infra-Obs`, `58-Evaluation`)

| Tool | Use |
|------|-----|
| LangSmith / Phoenix | Trace tutor reasoning, debug hints |
| OpenTelemetry | Emit mastery-update events |
| Grafana | Teacher dashboards, at-risk alerts |
| Evidently / custom | Drift in grading calibration |

---

## 10. Content Authoring & Curriculum Gen

- **Lesson plan generators** (prompt templates) — teacher-in-the-loop.
- **Differentiated worksheet gen** — same objective, N reading levels.
- **Quiz/item generators** with IRT difficulty tagging.
- **Synthetic dialogue data** for fine-tuning tutors (`51-Synthetic-Data-Generation`)
  — but watch for contamination; validate against real learner data.

---

## 11. Data & Privacy Infrastructure (`40-Data-Sovereignty`, `55-Ethics`)

For minors, prefer:
- Self-hosted models in a school/VPC tenant (`23-Local-AI`).
- Row-level isolation per learner; encryption at rest.
- Minimal logging; redact PII before any LLM call.
- FERPA/COPPA-aligned retention (`21-AI-Regulation-Antitrust`).

---

## 12. Build vs. Buy Decision Matrix

| If you need… | Build | Buy |
|--------------|-------|-----|
| Unique curriculum/RAG | ✅ (your content) | ❌ |
| Generic study aid | ❌ | ✅ (Duolingo/Quizlet) |
| Maximum privacy | ✅ (local, `23`) | ❌ |
| Fast pilot | ❌ | ✅ (hosted) |
| Deep KT integration | ✅ | partial |
| Compliance control | ✅ | depends |

---

## 13. Starter Repository Layout

```
tutor/
  curriculum/        # ingested textbooks, chunked w/ metadata
  kt_service/        # BKT/AKT API + store
  agent/             # LangGraph tutor + tools
  guardrails/        # input/output filters
  dashboard/         # teacher UI + at-risk alerts
  eval/              # pre/post learning-gain tests
  sandbox/           # code execution
```

See `03-Technical-Deep-Dive.md` for the code inside each.

---

## 14. Integration Checklist

- [ ] Curriculum RAG grounded & filtered by grade/subject
- [ ] Scaffolding controller enforced (no direct answers)
- [ ] KT service shared by tutor + dashboard
- [ ] Long-term memory persisted (`32`, `54`)
- [ ] Grading logs justifications; high-stakes = human-in-loop
- [ ] Guardrails for minors active
- [ ] Privacy/minors compliance verified (`40`, `55`)
- [ ] Learning-gain eval designed (`58`)
- [ ] Cost budgets set (`41`, `59`)

---

*Part of category 65. Pair with `03-Technical-Deep-Dive.md` (how to build) and
`02-Core-Topics.md` (what good looks like).*
