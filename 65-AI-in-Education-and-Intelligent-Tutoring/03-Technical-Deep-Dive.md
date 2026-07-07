# AI in Education — Technical Deep Dive

> Engineering blueprints for building safe, measurable AI tutors and adaptive learning
> systems. Covers the hybrid architecture (LLM generation + classic ITS instrumentation),
> RAG over curricula, tutoring agents with long-term memory, knowledge-tracing services,
> guardrails, and evaluation. Companion to `02-Core-Topics.md`.

## Architecture at a Glance

```
                         ┌─────────────────────────────┐
   Learner ──prompt─────>│  Tutor Agent (LLM + tools)  │
                         │  - Socratic hint policy      │
                         │  - scaffolding controller    │
                         └───┬───────────┬───────────┬──┘
                             │           │           │
                    ┌────────▼──┐ ┌──────▼─────┐ ┌────▼──────────┐
                    │ Curriculum │ │ Student     │ │ Assessment    │
                    │ RAG store  │ │ Memory / KT │ │ & Grading svc │
                    │ (textbook, │ │ service     │ │ (rubric, exec)│
                    │  syllabus) │ │ (32,54)     │ │               │
                    └────────────┘ └─────────────┘ └───────────────┘
                             │           │
                             └─────┬─────┘
                                   ▼
                         Teacher Dashboard / Analytics
                         (mastery estimates, alerts)
```

The design principle: **generate with an LLM, measure with instrumentation.** The LLM
gives fluid explanation; the surrounding services give the rigor classic ITS always had.

---

## 1. Curriculum RAG (Grounding the Tutor)

An ungrounded tutor hallucinates facts and "teaches" things not in the syllabus. Ground
it in the official material using `04-RAG`.

### 1.1 Ingestion

```python
# Chunk the textbook by section + learning objective, not fixed length
from langchain.text_splitter import MarkdownHeaderTextSplitter
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
    ("#", "unit"), ("##", "lesson"), ("###", "objective")])
docs = splitter.split_text(textbook_md)
# Attach metadata: grade, subject, KC_id, difficulty
```

### 1.2 Retrieval + Prompt

```python
def tutor_response(question, learner_id, k=4):
    ctx = curriculum_vectorstore.similarity_search(
        question, k=k,
        filter={"subject": "algebra", "grade": 8})
    mastery = kt_service.get_mastery(learner_id)        # Section 4
    hint_level = scaffolding.get_hint_level(learner_id, current_problem)
    prompt = f"""You are a Socratic tutor for 8th-grade algebra.
    Use ONLY the CURRICULUM below. Do NOT give the final answer.
    Current learner mastery: {mastery}. Hint level allowed: {hint_level}.
    Give the next hint up the ladder, then stop.

    CURRICULUM:
    {ctx}

    STUDENT QUESTION: {question}
    """
    return llm(prompt)
```

This single pattern addresses hallucination (`52-AI-Hallucination-Detection`) and
scaffolding simultaneously.

---

## 2. The Tutoring Agent (LLM + Tools)

Model the tutor as an agent (`03-Agents`) with tools, not a raw chat completion.

### 2.1 Tool Set

```python
tools = [
    retrieve_curriculum,        # RAG over syllabus
    generate_problem,           # make a practice item at target difficulty
    check_answer,               # symbolic/executable checker
    update_mastery,             # write to KT service
    request_human_teacher,      # escalate
    log_reflection,             # store learner's explanation
]
```

### 2.2 Scaffolding Controller (the key guardrail)

```python
class ScaffoldingController:
    def __init__(self, max_hints=4, struggle_budget=3):
        self.max_hints = max_hints
        self.struggle_budget = struggle_budget

    def policy(self, learner_id, problem_id, asked_for_answer: bool):
        level = hint_store.get(learner_id, problem_id, default=0)
        if asked_for_answer and level >= self.max_hints:
            # ran out of hints → learner needs practice, not answer
            return ("I won't give the answer yet. Try the first step: "
                    f"{generate_first_step(problem_id)}", level)
        new_level = min(level + 1, self.max_hints)
        hint_store.set(learner_id, problem_id, new_level)
        return next_hint(problem_id, new_level), new_level
```

The hint level reached becomes a mastery signal logged to the KT service.

---

## 3. Long-Term Memory (`32-Agent-Memory-Systems`, `54-Agent-State`)

A tutor that forgets the learner between sessions is useless. Persist:

| Memory type | Store | Example |
|-------------|-------|---------|
| Episodic | Vector DB + log | "On 2026-06-30 struggled with fractions" |
| Semantic (mastery) | KT service / KV | Per-KC probability |
| Procedural | Rule store | Preferred explanation style |
| Metadata | SQL | Grade, IEP flags, consent |

```python
# Persist after each interaction (see 54-Agent-State-Management)
session_log = {
    "learner_id": lid, "problem_id": pid, "hint_level": lvl,
    "self_explanation": text, "correct": bool, "ts": now}
memory.append_episodic(session_log)
kt_service.update(lid, pid, correct=bool)
```

Use durable execution (`31-AI-Workflow-Orchestration`) so a crash mid-session doesn't
lose progress.

---

## 4. Knowledge Tracing as a Service

Expose KT behind an API so the tutor, dashboard, and early-warning system share one
source of truth.

```python
# BKT update (see 02-Core-Topics §4.1)
def bkt_update(p_known, correct, p_T=0.1, p_S=0.1, p_G=0.2):
    if correct:
        likelihood = p_known * (1 - p_S) + (1 - p_known) * p_G
    else:
        likelihood = p_known * p_S + (1 - p_known) * (1 - p_G)
    posterior = (p_known * (1 - p_S)) / likelihood if correct else \
                (p_known * p_S) / likelihood
    next_known = posterior + (1 - posterior) * p_T
    return next_known

@app.post("/kt/update")
def kt_update(req: KTUpdate):
    cur = db.get_mastery(req.learner_id, req.kc_id)
    new = bkt_update(cur, req.correct)
    db.set_mastery(req.learner_id, req.kc_id, new)
    return {"mastery": new}
```

For deep variants, swap BKT for an AKT model served via ONNX/TorchServe.

---

## 5. Assessment & Grading Service

### 5.1 Code Grading (execution-based)

```python
import subprocess, tempfile
def grade_code(submission: str, tests: str) -> dict:
    with tempfile.TemporaryDirectory() as d:
        with open(f"{d}/sol.py","w") as f: f.write(submission)
        with open(f"{d}/t.py","w") as f: f.write(tests)
        r = subprocess.run(["pytest", f"{d}/t.py"], capture_output=True,
                           timeout=30, text=True)
    return {"passed": r.returncode == 0, "log": r.stdout[-2000:]}
```

Isolate execution (`56-MLOps`, sandboxes) — never run learner code on the host.

### 5.2 Essay Rubric Grading (with justification + calibration)

```python
def grade_essay(text, rubric, n_samples=3):
    scores = []
    for _ in range(n_samples):
        out = llm_json(RUBRIC_PROMPT.format(rubric=rubric, submission=text))
        scores.append(out)
    # majority/mean, require justification present, flag low agreement → human
    return aggregate(scores)
```

Log every justification for audit (`55-AI-Ethics`, `64-XAI`). For high-stakes grading,
require human confirmation.

---

## 6. Guardrails & Safety

Education involves minors → safety bar is high.

| Risk | Control |
|------|---------|
| Toxic/off-topic output | Output classifier + refusal policy |
| Hallucinated facts | Curriculum RAG + fact-check tool |
| Giving away answers | Scaffolding controller (Section 2.2) |
| PII leakage | Redaction + `40-AI-Data-Sovereignty` |
| Prompt injection from learner | Separate system prompt, no tool from free text |
| Self-harm disclosures | Crisis protocol → escalate to human |

Use an input/output guardrail layer (`18-Agent-Security-and-Trust`); never let the
model bypass the scaffolding controller via tool calls.

---

## 7. Multimodal Tutoring (`50-Multimodal-AI`)

```python
# Learner uploads a photo of handwritten work
img = load("hw.jpg")
ocr_text = vision_model.transcribe_handwriting(img)     # multimodal
feedback = tutor_response("Check my work: " + ocr_text)
# diagrams: VLM explains the graph; speech: ASR for spoken questions
```

This unlocks math-on-paper and science-diagram tutoring that pure text can't.

---

## 8. Personalization Loop (Adaptive Engine)

```
every interaction:
  1. read mastery vector (KT service)
  2. select next problem near ability θ (IRT, §3.2 of 02)
  3. set allowed hint level from mastery
  4. after response: update KT, append episodic memory
  5. if mastery[KC] > threshold: unlock next KC (CBE)
  6. if at-risk (mastery dropping / disengaged): alert teacher
```

Use event-driven architecture (`57-AI-Event-Driven-Agent-Architectures`) to fan out
updates to dashboard and early-warning without blocking the tutor.

---

## 9. Teacher Dashboard & Early-Warning

```sql
-- At-risk learners: mastery below threshold AND declining
SELECT learner_id, kc_id, mastery, trend
FROM mastery_snapshots
WHERE mastery < 0.6 AND trend < 0
ORDER BY trend ASC
LIMIT 50;
```

Surface as a dashboard; trigger teacher notification. Predictive retention models
(`58-AI-Evaluation`) should report recall/precision, not just accuracy, to avoid
stigmatizing false positives.

---

## 10. Reference Stack (Build-It-Yourself)

| Layer | Options |
|-------|---------|
| LLM | LearnLM, GPT-4o-class, Claude, Llama/Mistral (local, `23-Local-AI`) |
| Orchestration | LangGraph / CrewAI (`03-Agents`) |
| Vector store | pgvector / Chroma / Qdrant (`37-AI-Native-Databases`) |
| KT service | Custom (BKT/AKT) + Redis/Postgres |
| Execution sandbox | gVisor / Firecracker (`56-MLOps`) |
| Guardrails | Lakera / self-hosted classifiers (`18-Agent-Security`) |
| Durable state | Temporal / Inngest (`31-Workflow-Orchestration`, `54-State`) |
| Observability | LangSmith / OpenTelemetry (`20-Agent-Infra-Obs`) |
| UI | Next.js + WebSocket streaming |

See `04-Tools-and-Frameworks.md` for hosted platforms if you'd rather buy than build.

---

## 11. Evaluation (Does It Teach?)

Do **not** evaluate an educational AI like a chatbot. Measure learning.

```python
# Pre/post randomized control
def ab_test(tutor_variant, n=200):
    learners = random_split(cohort, n)
    pre = assess(learners, pretest)
    expose(learners[:n//2], tutor_variant)        # treatment
    post = assess(all, posttest)
    gain = (post - pre).groupby(arm).mean()
    return ttest(gain.treatment, gain.control)    # significance
```

Also track: KT-AUC (prediction), hint-level distribution, engagement (DFD), and
human-grade agreement (QWK). Full method in `58-AI-Evaluation-and-Benchmarking`.

---

## 12. Cost & Scale (`41-AI-Cost-Optimization`, `59-Financial-Governance`)

Tutoring is high-volume, low-margin per user. Optimize:
- Route simple hints to small models (`30-Small-Language-Models`, `53-Model-Cascading`).
- Cache curriculum retrieval.
- Batch dashboard aggregation.
- Set per-learner token budgets (`59-Financial-Governance`).

---

*Part of category 65. Build with `02-Core-Topics.md` (why) and
`04-Tools-and-Frameworks.md` (what to buy).*
