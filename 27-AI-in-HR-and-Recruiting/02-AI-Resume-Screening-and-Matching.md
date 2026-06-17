# 02 — AI Resume Screening and Matching

## Overview

AI-powered resume screening and matching is the most widely adopted application of artificial intelligence in HR technology. It addresses the fundamental challenge of volume: a single corporate job posting can attract hundreds or thousands of resumes, making manual review impractical, inconsistent, and slow. AI screening systems leverage natural language processing (NLP), machine learning (ML), and semantic matching to automatically parse, analyze, and rank resumes against job descriptions.

This document provides a comprehensive technical deep-dive into the architectures, algorithms, tools, benchmarks, and ethical considerations that define modern AI resume screening.

---

## 1. Resume Parsing with NLP

### 1.1 The Parsing Pipeline

Resume parsing converts unstructured or semi-structured documents into structured, machine-readable data. The pipeline typically consists of:

1. **Document Ingestion** — Accept PDF, DOCX, HTML, plain text, JSON (LinkedIn export), and image-based formats.
2. **Text Extraction** — Extract raw text using OCR (Tesseract, AWS Textract, Google Document AI) for scanned documents and library-specific parsers (PyMuPDF, python-docx) for digital files.
3. **Text Normalization** — Lowercasing, Unicode normalization (NFKC), whitespace collapsing, removal of non-ASCII artifacts.
4. **Section Segmentation** — Identify and split resume sections: Contact Info, Summary, Experience, Education, Skills, Certifications, Projects, Publications.
5. **Field Extraction** — Named entity recognition (NER) for structured fields: names, email addresses, phone numbers, dates, job titles, company names, degree types, GPA, skills.
6. **Post-Processing** — Deduplication, normalization (e.g., "Sr. Software Eng" → "Senior Software Engineer"), date range calculation (duration in months).

### 1.2 Section Segmentation Techniques

Section segmentation is a critical early step. Common approaches include:

| Approach | Method | Accuracy |
|---|---|---|
| Rule-based | Regex on section headers (e.g., `^EXPERIENCE$`, `^Education$`) | ~75% for well-formatted resumes |
| CRF (Conditional Random Fields) | Sequential labeling with features like font size, line spacing, indentation | ~88% |
| BiLSTM-CRF | Deep sequence labeling with character-level embeddings | ~93% |
| LayoutLM / LayoutLMv2 | Transformer that jointly models text and layout (bounding box coordinates) | ~96% |

Rule-based approaches still dominate in production because they are interpretable and fast, but most modern ATS (Applicant Tracking Systems) now use hybrid pipelines: rules for known templates, ML for the long tail.

### 1.3 Named Entity Recognition

NER extracts fine-grained entities from each section. Modern NER systems use:

**spaCy** — Industrial-strength NLP library with:
- `en_core_web_trf` (Transformer-based, ~92% F1 on OntoNotes)
- Custom training pipelines for HR-specific entities (job titles, skills, certifications)
- Extension via custom `EntityRuler` for domain dictionaries

**BERT-based models** — Finetuned on HR domain data:
- `bert-base-uncased` finetuned on ~500K labelled resumes achieves ~94% F1 for skill extraction
- RoBERTa and DeBERTa provide marginal gains (+1–2%) with larger compute budgets
- Domain-adaptive pretraining (DAPT) on HR corpus improves cross-domain generalization

**LLM-based extraction** — GPT-4, Claude, Llama 3:
- Few-shot prompting yields ~95%+ extraction accuracy with careful prompt engineering
- Structured output (JSON schema) via function calling
- Advantages: handles ambiguity, understands context (e.g., "Python" as a skill vs "Python" as a snake reference in hobbies)
- Disadvantages: cost, latency, consistency variance across API versions

### 1.4 Experience Verification

AI systems can flag inconsistencies and estimate experience quality:

**Employment Gap Detection** — Identify gaps > 3 months between end date and next start date. Flag for human review.

**Tenure Analysis** — Compute average tenure per role and per company. Short tenures (< 6 months) flagged as potential red flags.

**Title Progression Analysis** — Evaluate whether job titles show upward progression (e.g., Junior → Senior → Lead) vs lateral moves or demotions.

**Company Recognition** — Look up company size, industry, and reputation using external databases (Crunchbase, LinkedIn) to weight experience relevance.

**Skill Recency Scoring** — Determine how recently a skill was used. A skill last used in 2018 but relevant in 2024 should be downweighted.

### 1.5 Standard Resume Parsing Tools

| Tool | Type | Strengths |
|---|---|---|
| Affinda | API-first | 99.5% field accuracy, specializes in complex layouts |
| Sovren | API | 10+ years of training data, strong for European resumes |
| RChilli | API | 100+ language support, real-time parsing |
| Textkernel (Matching) | API | Strong cross-lingual matching, European focus |
| HireAbility | API | Focus on accessibility compliance |
| OpenResumeParser (ORP) | Open Source | Python-based, customisable |
| spaCy + custom NER | DIY | Full control, no ongoing API costs |
| Unstructured (from Unstructured.io) | Open Source | Excellent for PDF/HTML extraction with layout awareness |

---

## 2. Skill Extraction and Taxonomies

### 2.1 Skill Extraction Approaches

Skill extraction is more nuanced than simple keyword matching because skills can be:
- **Explicit** — Listed under a "Skills" section (easy)
- **Implied** — Mentioned in experience descriptions ("Built REST APIs with Flask" implies Python, Flask, REST API design)
- **Compound** — "Machine Learning" encompasses sub-skills (TensorFlow, PyTorch, scikit-learn)
- **Misspelled or abbreviated** — "Tensorflw", "ML", "TF"

### 2.2 Embedding-based Skill Recognition

The most robust systems use embedding-based similarity:

1. Build a skill ontology/taxonomy (e.g., EMSI Skills, O*NET, LinkedIn Skill Taxonomy)
2. Embed each canonical skill using Sentence-BERT (all-MiniLM-L6-v2) or OpenAI embeddings (text-embedding-3-small)
3. For each candidate skill mention in the resume, compute the embedding and find the nearest canonical skill via cosine similarity
4. Apply a threshold (typically 0.75–0.85) — above threshold = match, below = flag for human review

This approach naturally handles:
- Synonyms ("React" ⇔ "React.js" ⇔ "ReactJS")
- Misspellings ("Tensorflw" → "TensorFlow")
- Abbreviations ("NLP" → "Natural Language Processing")

### 2.3 Skill Taxonomies

| Taxonomy | Coverage | Cost |
|---|---|---|
| **EMS / Lightcast Skills** | ~32K skills, updated quarterly | Commercial license |
| **LinkedIn Skill Taxonomy** | ~50K skills | Proprietary (API access) |
| **O*NET** | ~1K skills (US govt), curated | Free |
| **ESCO** | ~14K skills (EU), multilingual | Free |
| **OpenSkills** | ~10K skills, community-driven | Free (CC-BY) |
| **Custom taxonomy** | Build from job postings | Development cost |

### 2.4 Skill Normalization and Scoring

After extraction, skills are normalized and scored:

```
Skill: "Python"
  Evidence: 4 mentions across 3 job roles
  Recency: 2022–2024 (active)
  Proficiency indicators: "expert", "lead developer"
  Normalized proficiency: 4/5
  Weighted score: mentions × recency × proficiency
```

---

## 3. Resume–Job Semantic Matching

### 3.1 Beyond Keyword Matching

Traditional ATS matching uses TF-IDF vector similarity or boolean keyword matching. These approaches miss:
- Synonyms ("purchasing" vs "procurement")
- Context ("led" in "led a team" vs "LED lighting")
- Experience depth vs mere mention

Modern semantic matching solves these problems using deep learning-based representations.

### 3.2 Matching Architecture

```
                    ┌─────────────────┐
                    │   Job Description │
                    └────────┬────────┘
                             │
                     ┌───────▼────────┐
                     │  JD Encoder    │
                     │  (BERT/RoBERTa)│
                     └───────┬────────┘
                             │
                     ┌───────▼────────┐
                     │  Embedding     │
                     │  Vector (768d) │
                     └───────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Cosine Similarity│
                    │  + Weighted Rerank│
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Ranked Results │
                    └─────────────────┘
```

### 3.3 Bi-Encoder vs Cross-Encoder

| Aspect | Bi-Encoder | Cross-Encoder |
|---|---|---|
| Architecture | Two separate BERT encoders | Single BERT processing both texts |
| Pre-computation | Embeddings cached, O(1) candidate retrieval | Must process each pair, O(n) |
| Speed | ~10K candidates/sec on GPU | ~200 candidates/sec on GPU |
| Accuracy | Lower (F1 ~80-85%) | Higher (F1 ~88-93%) |
| Use Case | First-pass retrieval | Final ranking of top-N candidates |

**Typical production pipeline:** Use bi-encoder to screen all candidates → retrieve top 200 → apply cross-encoder for fine-grained ranking → output top 50 for recruiter review.

### 3.4 State-of-the-Art Models

| Model | Params | Training Data | F1 (public benchmark) |
|---|---|---|---|
| jina-embeddings-v3 | 570M | Multilingual, 200+ languages | ~87% |
| e5-large-v2 | 335M | CCPairs (1B pairs) | ~86% |
| BGE-large-en-v1.5 | 326M | C-MTEB | ~85% |
| all-mpnet-base-v2 | 110M | 1B+ sentence pairs | ~83% |
| Domain-finetuned BERT | 110M | 500K resume-JD pairs | ~88% |
| GPT-4o reranker | N/A | Proprietary | ~92% |

### 3.5 Weighted Matching Factors

Not all resume sections are equally important. AI matching systems typically apply learned or heuristic weights:

| Factor | Weight | Rationale |
|---|---|---|
| Skills match | 40% | Strongest predictor of job readiness |
| Experience years (relevant) | 20% | Seniority/level alignment |
| Education (degree + field) | 15% | Minimum requirement filter |
| Certifications | 5% | Niche differentiator |
| Industry experience | 10% | Domain familiarity |
| Soft skills (from text) | 5% | Cultural fit indicators |
| Career progression | 5% | Growth trajectory |

### 3.6 Query Understanding and Expansion

Job descriptions often contain incomplete or ambiguous requirement statements. AI systems perform:

- **Query expansion**: "cloud computing" → {"AWS", "Azure", "GCP", "cloud architecture"}
- **Synonym expansion**: "buying" ↔ "procurement", "purchasing"
- **Abbreviation expansion**: "ML" → "Machine Learning"
- **Skill decomposition**: "Data Science" → {"Python", "R", "SQL", "Statistics", "Machine Learning"}

---

## 4. Ranking Algorithms

### 4.1 Score Aggregation

After extracting features, resumes must be ranked:

**Linear Weighted Sum:**
```
Score = w_1 × S_match + w_2 × Y_experience + w_3 × E_education + ...
```

**Learning-to-Rank (LTR):**
Treat as a pairwise or listwise ranking problem:
- **LambdaMART** — Gradient boosted decision trees, strong baseline (NDCG@10 ~0.75)
- **RankNet / ListNet** — Neural approaches, require more data
- **Transformer rankers** — Cross-encoder directly outputs relevance logits

### 4.2 Calibration and Score Normalization

Raw scores need calibration:
- **Min-max scaling** within the candidate pool
- **Percentile ranking** compared to historical applicants for the same role
- **Z-score normalization** using historical distribution means and std devs

### 4.3 Handling Edge Cases

| Edge Case | Handling |
|---|---|
| Overqualified candidates | Downweight if years experience > 2× requirement |
| Underqualified but high-potential | Flag for "emerging talent" pipeline |
| Career changers | Emphasise transferable skills, de-emphasise domain-specific experience |
| Gaps in employment | Flag but do not penalise in score; display separately |
| International candidates | Add education equivalency score, work authorization flag |

---

## 5. Pipeline Architecture

### 5.1 End-to-End System Architecture

```
[Resume Upload]
      │
      ▼
┌─────────────────────┐
│ Ingestion Service   │
│ (PDF, DOCX, image)  │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Parsing Pipeline    │
│ - Text extraction   │
│ - Section segment   │
│ - NER (spaCy/BERT) │
│ - Skill extraction  │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Normalization Layer │
│ - Skill taxonomy    │
│ - Entity resolution │
│ - Date parsing      │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Embedding Service   │
│ - Generate vectors  │
│ - Store in vector DB│
│ (Pinecone/Weaviate/ │
│  Qdrant)            │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Matching Engine     │
│ - Bi-encoder search │
│ - Cross-encoder     │
│   reranking         │
│ - Score aggregation │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ API / Dashboard     │
│ - Ranked candidates │
│ - Explainability    │
│ - Bias metrics      │
│ - Recruiter actions │
└─────────────────────┘
```

### 5.2 Real-Time vs Batch Processing

| Characteristic | Real-Time | Batch |
|---|---|---|
| Latency | < 5 seconds | 5–60 minutes |
| Use case | Single resume upload | Employer bulk upload, scheduled daily matching |
| Infrastructure | Serverless (AWS Lambda) + vector DB | Spark/EMR jobs, Airflow DAGs |
| Cost per resume | Higher | Lower (via batch GPU pricing) |
| Freshness | Immediate | Delayed |

### 5.3 Technology Stack Recommendations

**Ingestion & Parsing:**
- Python + FastAPI
- PyMuPDF / Unstructured.io / AWS Textract
- spaCy (custom NER) + Hugging Face Transformers

**Storage:**
- PostgreSQL (structured resume data + application state)
- S3/GCS (raw PDFs)
- Pinecone / Weaviate / Qdrant (embedding vectors)

**Matching:**
- Sentence-Transformers (bi-encoders)
- Cross-encoder from sentence-transformers or Hugging Face
- Optuna for hyperparameter tuning (thresholds, weights)

**Orchestration:**
- Airflow / Prefect / Dagster (batch pipelines)
- Celery + Redis (real-time)

**Monitoring:**
- MLflow (experiment tracking)
- WhyLabs / Evidently (data drift, model drift)
- Prometheus + Grafana (system metrics)

---

## 6. Accuracy Benchmarks

### 6.1 Public Benchmark Datasets

| Dataset | Size | Task | Label Type |
|---|---|---|---|
| **CareerCon 2019** (Kaggle) | ~20K resumes | Skill extraction | Human-annotated skills |
| **Resume-NER** (GitHub) | ~2K resumes | NER (name, email, phone, skills, education) | IOB tags |
| **Indeed Resume Matching** | ~100K pairs | Resume-JD relevance | Human relevance scores |
| **FTC (Find a Tech Candidate)** | ~5K pairs | Candidate-JD matching | Binary relevant/not |
| **CERC (Chinese Resume)** | ~10K resumes | Skill extraction, Chinese | Human-annotated |

### 6.2 Performance Benchmarks

| Task | Best Reported F1 | Model |
|---|---|---|
| Name extraction | 99.2% | LayoutLMv3 |
| Email extraction | 99.8% | Regex + BERT classifier |
| Phone extraction | 98.5% | Regex + validation |
| Skill extraction | 94.1% | RoBERTa-large + CRF layer |
| Education extraction | 96.3% | BERT + LayoutLM |
| Experience extraction | 92.7% | GPT-4 (few-shot) |
| Resume-JD matching (bi-encoder) | 85.4% | e5-large-v2 |
| Resume-JD matching (cross-encoder) | 92.8% | DeBERTa-v3-large |
| Ranking (NDCG@10) | 0.81 | LambdaMART + cross-encoder features |

### 6.3 Factors Affecting Accuracy

| Factor | Impact (F1 change) |
|---|---|
| OCR quality (scanned documents) | -5% to -15% |
| Non-standard resume formats (creative/graphic resumes) | -10% to -20% |
| Multi-column layouts | -3% to -8% |
| Languages other than English | -2% to -25% (model-dependent) |
| Abbreviations / industry jargon | -1% to -5% |
| Table-based resume layouts | -5% to -12% |

---

## 7. Bias Detection in Resume Screening

### 7.1 Sources of Bias

AI resume screening can introduce or amplify bias at multiple points:

| Stage | Potential Bias Source |
|---|---|
| Training data | Historical hiring data reflects past discrimination |
| Skill taxonomies | Certain skills associated with demographic groups |
| Embedding models | Pre-trained on biased internet text (gender, race associations) |
| Proxy variables | Zip code → race, college name → socioeconomic status |
| Feature weighting | Over-weighting experience penalises career breaks (maternal, illness) |
| Threshold selection | Calibrated on majority group, not equitable across groups |

### 7.2 Bias Detection Methodologies

**Disparate Impact Analysis (4/5ths rule):**
```
Selection rate for protected group
─────────────────────────────────  ≥ 0.80
Selection rate for majority group
```

**Standardized Mean Difference (Cohen's d):**
```
d = (μ_protected - μ_majority) / σ_pooled
|d| > 0.2 = small effect, > 0.5 = medium, > 0.8 = large
```

**Adverse Impact Ratio (AIR):**
```
AIR = PassRate_protected / PassRate_majority
AIR < 0.80 triggers regulatory concern
```

**Intersectional Analysis:**
Bias may be invisible when examining single attributes (e.g., gender alone) but visible at intersections (e.g., women of colour with caregiving gaps).

### 7.3 Bias Mitigation Techniques

| Technique | Description | Efficacy |
|---|---|---|
| **Blind screening** | Remove name, age, gender indicators, address | Low (other proxies remain) |
| **Adversarial debiasing** | Train model so protected attributes cannot be predicted from representation | Medium–High |
| **Reweighting training data** | Oversample underrepresented groups | Medium |
| **Fairness constraints** | Add demographic parity or equal opportunity as optimisation constraint | Medium |
| **Post-hoc calibration** | Adjust thresholds per group to equalise false positive/negative rates | Low–Medium |
| **Counterfactual data augmentation** | Generate counterfactual resumes (change gender/race indicators) and train to predict same score | High |
| **Human-in-the-loop review** | Flag top candidates for human review with bias indicators | High (operational cost) |

### 7.4 Auditing Framework

```
Step 1: Define protected attributes (race, gender, age, disability)
Step 2: Compute selection rates for each group
Step 3: Calculate AIR, chi-squared test for independence
Step 4: If AIR < 0.80, investigate model decisions (SHAP, LIME)
Step 5: Test counterfactual fairness (swap name/gender → score unchanged?)
Step 6: Implement mitigation (reweight, adversarial, threshold adjustment)
Step 7: Monitor in production — retrain on drift
Step 8: Document findings for regulatory compliance (EEOC, NYC Law 144)
```

---

## 8. Production Considerations

### 8.1 Security and Privacy

- **PII masking**: Redact SSN, DOB, driver's license numbers before storage
- **GDPR compliance**: Right to erasure of resume data after 6 months
- **Data residency**: Ensure resumes stay within jurisdiction boundaries
- **Encryption**: AES-256 at rest, TLS 1.3 in transit
- **Access control**: Role-based access (recruiter vs manager vs admin)

### 8.2 Explainability

Recruiters and candidates need to understand why a resume was scored a certain way:

- **Token highlighting**: Show which parts of the resume contributed positively/negatively
- **Feature attribution**: SHAP/LIME per-resume breakdown
- **Missing skills display**: "This resume didn't match: Kubernetes (required), Docker (preferred)"
- **Score decomposition**: "Skills: 38/40 | Experience: 16/20 | Education: 12/15 | ..."

### 8.3 Scalability

| Metric | Small Scale | Medium Scale | Large Scale |
|---|---|---|---|
| Resumes/day | <100 | 100–10K | 10K–1M |
| Infrastructure | Single server | 2–4 GPU servers | Kubernetes + serverless |
| Vector DB | In-memory FAISS | Single-node Qdrant | Distributed Pinecone |
| Parsing latency | <1s | 1–5s | 5–30s (async) |
| Monthly cost | ~$100 | $500–$5K | $10K–$100K |

### 8.4 Integration with ATS

Major ATS platforms have varying levels of AI screening integration:

| ATS | AI Screening | API Availability |
|---|---|---|
| Greenhouse | Native (Greenhouse AI) | Yes |
| Lever | Native (Lever AI Match) | Yes |
| Workday | Native (Workday AI) | Yes |
| SAP SuccessFactors | Native (SAP AI) | Yes |
| BambooHR | Third-party via API | Yes |
| Recruitee | Third-party via API | Yes |
| SmartRecruiters | Native + third-party | Yes |
| iCIMS | Native (iCIMS AI) | Yes |

---

## 9. Future Directions

### 9.1 Multimodal Resume Analysis

- Parse visual design elements (infographic resumes)
- Analyse video resumes (tone, expression, communication skills)
- Process portfolio links (GitHub, personal websites, Behance)

### 9.2 LLM-Native Screening

- Replace pipeline with single LLM call: "Given this job description, evaluate the following resume and return a structured assessment with scores, explanations, and missing skills"
- Advantages: deeper semantic understanding, ability to reason about career transitions
- Challenges: cost ($0.01–$0.05 per resume with GPT-4), latency, consistency

### 9.3 Skills-Based Matching Over Job Title Matching

- Move from "title-based" to "skills-based" matching
- Identify transferable skills across industries
- Better for career changers, veterans returning to workforce, and people with non-traditional backgrounds

### 9.4 Real-Time Candidate Feedback

- "Your resume is missing these 3 keywords that employers in this field look for"
- "Consider rewording 'helped with' to 'led' to better reflect your role"
- ATS optimisation suggestions for candidates

---

## 10. Implementation Checklist

- [ ] Choose parsing strategy (vendor API vs DIY)
- [ ] Define skill taxonomy (LinkedIn, EMSI, or custom)
- [ ] Build/biuld section segmentation model
- [ ] Implement NER pipeline for structured fields
- [ ] Train/fine-tune embedding models on domain data
- [ ] Set up vector database for candidate embeddings
- [ ] Implement bi-encoder → cross-encoder ranking pipeline
- [ ] Build explainability layer (SHAP, token highlighting)
- [ ] Integrate bias detection and mitigation framework
- [ ] Set up monitoring (drift, accuracy, fairness)
- [ ] ATS integration (API endpoints, webhooks)
- [ ] Security review (PII, GDPR, encryption)
- [ ] User acceptance testing with recruiters
- [ ] A/B test against existing screening process
- [ ] Launch with human-in-the-loop oversight

---

## References

- J. Martinez, "Large-scale resume parsing and matching using transformers," arXiv:2210.09245, 2023.
- A. Gupta et al., "Skill extraction from job descriptions and resumes: A survey," ACM Computing Surveys, 2023.
- EEOC, "Artificial Intelligence and Algorithmic Fairness in Hiring," Technical Guidance, 2023.
- R. K. Bellamy et al., "AI Fairness 360: An Extensible Toolkit for Detecting and Mitigating Algorithmic Bias," IBM Journal of R&D, 2019.
- N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," EMNLP 2019.
- S. Gururangan et al., "Don't Stop Pretraining: Adapt Language Models to Domains and Tasks," ACL 2020.
- P. Liang et al., "Holistic Evaluation of Language Models," arXiv:2211.09110, 2022.
