# AI in Education — Core Topics

> The learning-science foundations and the four core capabilities of AI in education:
> intelligent tutoring, adaptive learning, automated assessment, and feedback. This file
> explains *what good looks like* and the research that backs it, complementing the
> technical build guidance in `03-Technical-Deep-Dive.md`.

## 1. The Learning Sciences Primer

Before applying AI, you must know what you are optimizing. The relevant constructs:

### 1.1 Mastery Learning
Bloom (1984) showed that one-on-one mastery learning moved the average student from the
50th to the 98th percentile versus lecture-based instruction. The catch: it requires
~2× the adult time. **AI's value proposition is to deliver mastery-learning economics at
scale.** This is the original justification for ITS.

### 1.2 Spacing and Interleaving
- **Spacing effect:** distributed practice beats massed practice.
- **Interleaving:** mixing topics improves discrimination and long-term retention.
An adaptive engine should schedule reviews using a spaced-repetition scheduler
(e.g., Leitner boxes, SM-2, FSRS).

### 1.3 Retrieval Practice
Testing is *not* just measurement — it is a learning event (the testing effect). Good
tutors force the learner to retrieve, then give feedback.

### 1.4 Cognitive Load Theory
Working memory is limited. Scaffolding (`01-Overview.md`) reduces extraneous load and
frees capacity for germane (schema-building) load. An LLM that dumps a 40-line answer
overloads the learner; a hint ladder does not.

### 1.5 Productive Struggle
Some difficulty is necessary for learning. Tuning the *desirable difficulty* is the
pedagogical control knob: too easy → no gain; too hard → disengage.

---

## 2. Intelligent Tutoring Systems (ITS)

### 2.1 The Classic Four-Component Architecture

```
+-------------------+      +--------------------+
|  Domain Model     |<---->|  Student Model      |
| (what expertise   |      | (what the learner   |
|  looks like)      |      |  knows / misconceptions)|
+-------------------+      +--------------------+
        ^                            ^
        v                            v
+-------------------+      +--------------------+
|  Tutor Model      |<---->|  Interface Model    |
| (pedagogical      |      | (how it's presented)|
|  strategy)        |      |                     |
+-------------------+      +--------------------+
```

- **Domain model:** a skill graph / knowledge components (KCs). Example: "solving
  linear equations" decomposes into "isolate variable", "distribute", "combine like
  terms".
- **Student model:** a per-learner estimate of mastery over each KC. Updated by
  knowledge tracing (Section 4).
- **Tutor model:** policy that picks the next problem, hint, or explanation.
- **Interface:** step-based (best for KT) vs. conversational (best for engagement).

### 2.2 Why ITS Works

Meta-analyses (e.g., Kulik & Fletcher) find ITS yields ~0.3–0.5 SD learning gains vs.
traditional instruction — comparable to human tutoring's ~0.4 SD. The limitation
historically was the *authoring cost*: encoding a domain model took experts months.
**LLMs collapse that cost** — they can generate problems, hints, and explanations on
demand, which is why generative tutors exploded after 2023.

### 2.3 Generative Tutors vs. Classic ITS

| Dimension | Classic ITS | Generative (LLM) Tutor |
|-----------|-------------|------------------------|
| Authoring cost | Very high | Near-zero (prompt + RAG) |
| Explanation quality | Narrow, rigid | Fluid, multi-style |
| Mastery tracking | Precise (step logs) | Weak unless instrumented |
| Hallucination risk | Low (curated) | High (needs guardrails) |
| Scalability | Poor (per-domain build) | Excellent |
| Best use | Well-defined STEM skills | Open-ended, language, breadth |

**Takeaway:** the winning architecture is *hybrid* — LLM for generation, classic ITS
instrumentation for measurement. See `03-Technical-Deep-Dive.md`.

---

## 3. Adaptive Learning & Personalization

### 3.1 Personalization Axes

| Axis | Example adaptation |
|------|--------------------|
| Pace | Faster learners skip drills; slower get more practice |
| Difficulty | Item-response-theory-based selection |
| Modality | Visual learner → diagrams; auditory → audio summaries |
| Pathway | Different order of KCs based on prerequisites mastered |
| Language | Translate/code-switch for multilingual learners |
| Scaffolding | Reduce hints as mastery rises |

### 3.2 Item Response Theory (IRT)

IRT models the probability that a learner of ability θ answers an item of difficulty β
correctly:

```
P(correct | θ, β) = 1 / (1 + exp(-(θ - β)))     # 1-parameter logistic (Rasch)
```

Adaptive tests (e.g., computerized adaptive testing) pick the next item near the
learner's estimated ability to maximize information. This is the math behind
"the test got harder as I answered right."

### 3.3 Competency-Based Education (CBE)

Learners advance on *demonstrated mastery*, not seat time. AI makes CBE operable by
continuously estimating mastery (knowledge tracing) and unlocking the next competency
automatically.

---

## 4. Knowledge Tracing (KT)

The quantitative core of personalization. Given a learner's past responses, predict
mastery of each skill and the probability of correct response on the next item.

### 4.1 Bayesian Knowledge Tracing (BKT)

A hidden Markov model per skill with four parameters:
- `p(L0)` prior probability the skill is known
- `p(T)` probability of transition from not-known to known after an opportunity
- `p(S)` slip (known but answered wrong)
- `p(G)` guess (unknown but answered right)

Update after each response:

```
P(L_n | obs) = [ P(obs | L_n) * P(L_{n-1}) ] / normalizer
P(L_{n+1})  = P(L_n) + (1 - P(L_n)) * p(T)     # if success
```

### 4.2 Deep Knowledge Tracing (DKT)

An LSTM that ingests the interaction sequence `(skill_id, correctness)` and outputs a
mastery vector. Pros: captures inter-skill transfer. Cons: opaque (ties to `64-XAI`).

### 4.3 Modern Variants (2022–2026)

- **SAKT / AKT** — self-attention knowledge tracing; AKT uses a memory-augmented
  attention with monotonic attention over time.
- **Contextual KT** — incorporates problem text embeddings (often from an LLM).
- **LLM-as-KT** — prompt an LLM with the history and ask for a mastery estimate; cheap
  but needs calibration.

---

## 5. Automated Assessment & Grading

### 5.1 What Can Be Auto-Graded Well

| Type | Maturity | Method |
|------|----------|--------|
| Multiple choice / fill-in | Excellent | Exact / IRT |
| Code submissions | Good | Execution + test suites (see `33-AI-Native-Software-Development`) |
| Math expressions | Good | Symbolic equivalence (SymPy, CAS) |
| Short answer | Moderate | Embedding similarity + LLM rubric |
| Essays | Moderate | LLM rubric grading + retrieval of source |
| Open-ended projects | Weak | Human-in-the-loop required |

### 5.2 Rubric-Based LLM Grading Pattern

```python
def grade(submission: str, rubric: list[str], model="learnlm-1.5") -> dict:
    prompt = f"""You are a strict grader. Assess the submission against EACH rubric
    criterion. Return JSON: {{criterion: {{score, justification}}}}.

    RUBRIC:
    {chr(10).join(f'- {r}' for r in rubric)}

    SUBMISSION:
    {submission}
    """
    return llm_json(prompt, model=model)   # validate schema, log justifications
```

Principles:
- **Always ask for justifications** (auditability → `55-AI-Ethics`, `64-XAI`).
- **Calibrate** against human grades; report agreement (quadratic weighted kappa).
- **Never grade on a single call** for high-stakes decisions; use multi-sample or
  human review.

### 5.3 Academic Integrity

Generative AI breaks "write an essay at home" assignments. Responses:
- **Shift to process:** grade drafts, reflection, and in-class demonstration.
- **Authorship detection** (weak, adversarial) — see `43-AI-Data-Provenance`.
- **AI-permitted assignments** with cited use, teaching *about* AI (`34-AI-Workforce`).

---

## 6. Feedback Quality

Feedback is where tutors live or die. Hattie's meta-analyses rank *feedback* among the
highest-impact interventions (~0.7 SD) — but only *corrective, timely, and specific*
feedback.

### 6.1 Feedback Dimensions

| Dimension | Bad | Good |
|-----------|-----|------|
| Timing | Next week | Immediate |
| Specificity | "Good job" | "Your thesis lacks evidence in para 2" |
| Location | Final grade only | At the point of error |
| Control | Tells the answer | Guides discovery |

### 6.2 The Hint Ladder (Socratic Scaffolding)

```
Level 0: Re-state the goal / relevant concept
Level 1: Point to the relevant principle
Level 2: Suggest an approach (not the step)
Level 3: Give the first step
Level 4: Worked example (last resort)
```

A tutor agent tracks the learner's *hint level reached* as a mastery signal: needing
only Level 0 ⇒ mastered; needing Level 4 ⇒ not.

---

## 7. Accessibility & Inclusion

AI is a force multiplier for accessibility:
- **Speech-to-text / captioning** for Deaf/hard-of-hearing learners.
- **Text-to-speech** for dyslexia / low-vision.
- **Simplification** of reading level on demand.
- **Translation** for multilingual/ELL students (ties to `50-Multimodal-AI`).

But models must be evaluated for bias across dialects and languages
(`55-AI-Ethics-and-Responsible-AI`) — an English-tuned tutor can fail non-native speakers.

---

## 8. Teacher Augmentation (Not Replacement)

The highest-ROI deployments keep the teacher central:
- **Lesson planning** drafts → teacher edits.
- **Differentiated worksheets** generated per reading level.
- **Progress insights** from learning analytics dashboards.
- **Parent communication** drafts.

This is "copilot for teachers" (`33-AI-Native-Software-Development` patterns apply).

---

## 9. Common Failure Modes

| Failure | Cause | Mitigation |
|---------|-------|------------|
| False confidence | Tutor affirms wrong reasoning | Require learner explanation first |
| Shortcutting | Gives answers directly | Hint-ladder, productive-struggle budget |
| Hallucinated facts | Untethered LLM | RAG over curriculum (`04-RAG`) |
| Bias in grading | Training data skew | Calibrate, human-in-loop, audit |
| Privacy breach | Logging minors' data | `40-AI-Data-Sovereignty`, minimize |
| Engagement crash | Repetitive generation | Vary style, gamify, human check-ins |

---

## 10. Metrics That Matter

| Metric | What it measures | Trap |
|--------|------------------|------|
| Learning gain | Pre/post assessment delta | Confounds with maturity |
| Knowledge-tracing AUC | Prediction accuracy | Doesn't prove learning |
| Engagement / DFD | Days frequency of use | Engagement ≠ learning |
| Hint-level distribution | Scaffolding effectiveness | Needs calibration |
| Human-grade agreement | Grading validity | Kappa, not accuracy |
| Dropout prediction recall | At-risk identification | False alarms stigmatize |

---

## 11. Relationship to Adjacent Categories

- `29-Reasoning-and-Inference-Scaling` — math/science tutoring needs reasoning.
- `52-AI-Hallucination-Detection` — factual grounding of explanations.
- `58-AI-Evaluation-and-Benchmarking` — learning-gain eval methodology.
- `55-AI-Ethics` & `21-AI-Regulation` — fairness, minors' data, policy.
- `34-AI-Workforce-Transformation` — teaching *about* AI is part of education.

---

*Part of category 65 — AI in Education and Intelligent Tutoring. See `01-Overview.md`
for the taxonomy and `03-Technical-Deep-Dive.md` to build.*
