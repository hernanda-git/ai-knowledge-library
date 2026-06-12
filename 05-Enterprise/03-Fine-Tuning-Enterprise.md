# Enterprise Fine-Tuning

> A comprehensive guide to fine-tuning large language models in enterprise environments, covering decision frameworks, PEFT methods, data pipelines, training infrastructure, and production deployment.

---

## Table of Contents

1. [When to Fine-Tune vs RAG vs Prompting](#1-when-to-fine-tune-vs-rag-vs-prompting)
2. [Domain Adaptation Strategies](#2-domain-adaptation-strategies)
3. [PEFT Methods — Enterprise Focus](#3-peft-methods--enterprise-focus)
4. [Data Preparation Pipeline](#4-data-preparation-pipeline)
5. [Synthetic Data Generation](#5-synthetic-data-generation)
6. [Human Annotation](#6-human-annotation)
7. [Training Infrastructure](#7-training-infrastructure)
8. [Tools](#8-tools)
9. [Hyperparameter Tuning](#9-hyperparameter-tuning)
10. [Training Stability](#10-training-stability)
11. [Evaluation](#11-evaluation)
12. [Deployment](#12-deployment)
13. [Model Formats](#13-model-formats)

---

## 1. When to Fine-Tune vs RAG vs Prompting

### Decision Matrix

Choosing the right paradigm — fine-tuning, Retrieval-Augmented Generation (RAG), or prompting — is the most consequential architectural decision in any LLM project. The wrong choice burns GPU budget on unnecessary training, couples you to fragile prompt engineering, or locks you into a retrieval pipeline with latency you cannot sustain. This matrix provides a structured framework.

| Decision Factor | Prompting (Few-shot / Zero-shot) | RAG | Fine-Tuning |
|---|---|---|---|
| **Knowledge recency** | Stale at cutoff | Fresh via retrieval index | Frozen at training time |
| **Data volume** | ~10 examples max | Unlimited (index scale) | 1K–100K+ examples |
| **Task specificity** | General-purpose | Factual recall tasks | Specialized behavior / style |
| **Latency budget** | ~200ms–2s | +50–500ms retrieval overhead | Same as base model |
| **Cost per inference** | Lowest | Medium (storage + retrieval) | Highest (GPU hour amortized) |
| **Data privacy** | No data retention | Documents indexed externally | Weights memorize data |
| **Domain jargon** | Poor (no memorization) | Good (if in retrieved docs) | Excellent (deep assimilation) |
| **Output format control** | Fragile (prompt engineering) | Moderate | Strong (format baked into weights) |
| **Model size constraints** | None | Must fit context + retrieved docs | Must fit batch on GPU |
| **Iteration speed** | Instant prompt edits | Re-index documents (hours) | Re-train (hours–days) |
| **Hallucination risk** | Highest | Lowest (grounded in retrieved text) | Moderate (memorized patterns) |
| **Maintenance burden** | Low | Medium (pipeline monitoring) | High (retraining cycles) |

### Decision Flow

**Start with prompting.** If the task can be accomplished with 0–10 examples in the prompt and the required knowledge fits within the model's training cutoff, prompting is free and infinitely fast to iterate. Use structured output (JSON mode, function calling) to enforce format.

**Add RAG when the model needs access to information outside its weights.** If the answer depends on specific documents, proprietary knowledge bases, or frequently updating data, RAG provides grounding without retraining. RAG is also the correct choice when data volumes exceed what fine-tuning can absorb (100K+ docs) or when data sources change daily.

**Fine-tune when the model needs to internalize a behavior, style, or domain.** Fine-tuning shines when:
- The output format must be strictly controlled (e.g., legal document templates, clinical note styles)
- The model must deeply understand domain-specific jargon and concepts (e.g., medical terminology, legal citations, financial reporting standards)
- The model needs to exhibit a specific persona or tone consistently
- RAG retrieval quality is inadequate (low precision/recall on domain corpus)
- Inference latency cannot tolerate retrieval overhead

### Hybrid Patterns

The most successful enterprise deployments combine all three:

```
Prompting (system prompt + instructions)
    ↓
RAG (retrieved context injected into prompt)
    ↓
Fine-Tuned Model (domain-adapted base)
```

**Example: Enterprise Legal Assistant**
- **Prompting**: System prompt defines lawyer persona, jurisdiction rules, and output format
- **RAG**: Retrieves relevant case law, statutes, and contract clauses from vector database
- **Fine-Tuning**: Model fine-tuned on legal document corpus for citation format, legal reasoning, and jurisdiction-specific terminology

### Anti-Patterns to Avoid

| Anti-Pattern | Why It Fails |
|---|---|
| Fine-tuning on facts that change quarterly | Model memorizes stale data; retraining needed every quarter |
| RAG for tasks requiring deep reasoning | Retrieved context is fragmented; model cannot synthesize across docs |
| Prompting for complex multi-step workflows | Prompt becomes fragile, long, and expensive; context window overflow |
| Fine-tuning to "add knowledge" | Fine-tuning teaches behavior/shape, not facts; RAG is better for knowledge |
| No fallback strategy | All three should degrade gracefully; RAG fallback from fine-tuning |

---

## 2. Domain Adaptation Strategies

### 2.1 Legal Domain

Legal NLP requires precision in citation format, jurisdiction awareness, and understanding of specialized terminology spanning centuries of precedent.

**Key Adaptation Approaches:**

- **Continued pre-training (CPT)** on legal corpora (CourtListener, CaseLaw, PACER, legal textbooks, statutes) — 10B–50B tokens for base model assimilation
- **Instruction fine-tuning** on legal QA pairs, contract analysis, and legal reasoning chains
- **RLHF/DPO** for alignment with legal ethics, confidentiality rules, and jurisdiction-specific output formats

**Critical Capabilities:**
- Citation extraction and formatting (Bluebook, ALWD, OSCOLA)
- Legal entity recognition (statutes, case names, court names, judges, parties)
- Jurisdiction classification (federal vs state, common law vs civil law)
- Temporal reasoning (overruled precedents, statutory amendments)

**Dataset Sources:**
- **Contracts**: EDGAR (SEC filings), US Contract datasets (200K+ contracts), Atticus (contract review)
- **Case Law**: CourtListener bulk data, Caselaw Access Project (Harvard Law), Free Law Project
- **Statutes**: US Code XML, State legislative databases, EUR-Lex (EU law)
- **Legal QA**: LexGLUE, CaseHOLD, LegalBench, CUAD (Contract Understanding Atticus Dataset)

**Notable Domain-specific Models:**
- LLaMA-based: SaulLM-7B, SaulLM-141B (legal)
- BERT-based: Legal-BERT, CaseLaw-BERT, RoBERTa-Legal
- Encoder-only: LexNLP-based models

**Evaluation Benchmarks:**
- LegalBench (162 tasks across 6 categories: reasoning, interpretation, drafting, analysis, entity, citation)
- LexGLUE (7 tasks: ECHR, casehold, unfair_tos, etc.)
- CUAD (41 contract question types)
- CaseHOLD (citation prediction)

### 2.2 Medical Domain

Medical adaptation requires handling of clinical abbreviations, drug names, anatomical terminology, and strict adherence to medical formatting standards (ICD-10, SNOMED CT, RxNorm).

**Key Adaptation Approaches:**
- **Continued pre-training** on PubMed Central, MIMIC-III/IV clinical notes, Medscape, textbooks
- **Multi-task fine-tuning**: NER for medical entities, relation extraction for drug interactions, summarization for clinical notes
- **Reinforcement learning** from patient outcome proxies (not recommended without rigorous oversight)

**Critical Capabilities:**
- Medical entity extraction (symptoms, diagnoses, medications, procedures, lab values)
- ICD-10/SNOMED/CPT code assignment
- Clinical note summarization and structured data extraction
- Radiology report generation
- Drug-drug interaction detection

**Dataset Sources:**
- **Clinical Notes**: MIMIC-III/IV (requires credentialed access), i2b2/n2c2 datasets, eICU
- **Biomedical Literature**: PubMed Central Open Access (3M+ full-text articles), PubMed abstracts
- **Medical QA**: MedQA (USMLE), MedMCQA, PubMedQA, BioASQ
- **Drug Data**: DrugBank, RxNorm, DailyMed (FDA labels)

**Privacy Considerations:**
- HIPAA compliance required for training on protected health information (PHI)
- De-identification before any training (MIMIC is pre-de-identified)
- Differential privacy during training when using clinical data
- Model weights may memorize PHI — avoid deploying on raw clinical text

**Evaluation Benchmarks:**
- MedQA (USMLE step questions)
- MedMCQA (multi-choice medical questions)
- PubMedQA (biomedical QA)
- BioBERT-based NER benchmarks (NCBI Disease, BC5CDR, JNLPBA)

### 2.3 Finance Domain

Financial models need to understand SEC filings, market terminology, financial ratios, and temporal market context.

**Key Adaptation Approaches:**
- **CPT on financial corpus**: SEC filings (10-K, 10-Q, 8-K, S-1), earnings call transcripts, analyst reports, financial news
- **Instruction tuning** for financial QA, sentiment analysis, numerical reasoning, and report generation
- **Numerical tokenization** considerations (some domains benefit from digit tokenization or specialized number embeddings)

**Critical Capabilities:**
- Financial entity recognition (ticker symbols, company names, currency amounts, financial ratios)
- Numerical reasoning (percentage change, CAGR, EPS, P/E ratio calculation)
- Financial sentiment analysis (bullish/bearish/neutral on financial texts)
- SEC filing structure understanding (Item numbers, MD&A, Risk Factors, Financial Statements)
- Multi-lingual support (global markets)

**Dataset Sources:**
- **SEC Filings**: EDGAR (free access via SEC API), thousands of new filings daily
- **Earnings Calls**: Transcripts from SeekingAlpha, Motley Fool, manual scraping
- **Financial News**: Reuters, Bloomberg (licensed), Yahoo Finance
- **Financial QA**: FinQA, ConvFinQA, TAT-QA (table-based), FinanceBench

**Numerical Capabilities:**
Financial NLP places extreme demands on numerical reasoning. Evaluate:
- Calculation of financial ratios given raw financial data
- Trend identification (YoY growth, QoQ changes)
- Table understanding (financial statements in tabular format)
- Currency conversion and unit normalization

**Evaluation Benchmarks:**
- FinQA (numerical reasoning over financial reports)
- ConvFinQA (conversational financial QA)
- FinanceBench (real-world financial analyst questions)
- TAT-QA (table-and-text hybrid QA)
- FiQA SA (financial sentiment analysis)

### 2.4 Code Domain

Code-adapted models power developer productivity tools, automated documentation, code review, and bug fixing.

**Key Adaptation Approaches:**
- **CPT on code corpora**: GitHub repositories (permissively licensed), Stack Overflow, technical documentation
- **Fill-in-the-middle (FIM)** training: predict masked code spans, not just left-to-right generation
- **Multi-language training**: Python, JavaScript, TypeScript, Java, Go, Rust, C++, SQL, Bash

**Critical Capabilities:**
- Code completion (single-line, multi-line, whole-function)
- Code explanation and documentation generation
- Bug detection and fix suggestion
- Code translation (e.g., Python to JavaScript)
- SQL query generation from natural language
- Shell command generation

**Dataset Sources:**
- **GitHub**: The Stack (3TB+), CodeParrot, GitHub Archive
- **QA**: Stack Overflow data dump, CodeSearchNet, StaQC
- **Documentation**: Official docs (Python, JavaScript, Rust, etc.)

**Training Considerations:**
- **FIM training**: Use a FIM transformation that converts code into `<PREFIX><SUFFIX><HOLE>` format. Common implementations: PSM (Prefix-Suffix-Middle), SPM (Suffix-Prefix-Middle)
- **Language balancing**: Weight smaller languages (Rust, Haskell) higher to prevent catastrophic forgetting
- **File-level context**: Code files have inter-dependencies; consider whole-file or cross-file context windows
- **Syntax validation**: Add AST validation to training data filters

**Evaluation Benchmarks:**
- HumanEval (function synthesis from docstrings)
- MBPP (mostly basic Python programming)
- SWE-bench (real-world GitHub issue resolution)
- BigCodeBench (multi-language, multi-task)
- RepoBench (repository-level code completion)

### 2.5 Customer Support Domain

Customer support models need tone control, brand voice adherence, escalation logic, and multi-turn conversation management.

**Key Adaptation Approaches:**
- **Conversational fine-tuning** on chat logs, ticket histories, and email threads
- **Supervised fine-tuning** with agent-written responses as targets
- **RLHF/DPO** with customer satisfaction scores as reward signal

**Critical Capabilities:**
- Intent classification and routing
- Sentiment detection and empathy generation
- Knowledge base grounding (hybrid with RAG)
- Multi-turn conversation coherence
- Escalation detection and handoff
- Brand voice and tone consistency

**Dataset Sources:**
- **Internal ticketing systems**: Zendesk, Freshdesk, Salesforce Service Cloud, Intercom
- **Chat logs**: Live chat transcripts (anonymized)
- **Email**: Customer support email threads (PII-redacted)
- **Knowledge base**: Internal FAQ and documentation

**Privacy and Compliance:**
- GDPR: Support data often contains PII; ensure retention policies
- CCPA: California consumer privacy rights apply
- PCI DSS: Payment information must never enter training data
- HIPAA: If healthcare support, clinical data protection required

**Quality Metrics:**
- CSAT (Customer Satisfaction Score) improvement
- First Contact Resolution (FCR) rate
- Average Handle Time (AHT) change
- Escalation rate reduction
- Agent acceptance rate of AI-suggested responses

---

## 3. PEFT Methods — Enterprise Focus

Parameter-Efficient Fine-Tuning (PEFT) methods are the backbone of enterprise LLM adaptation. They make fine-tuning economically viable by updating only a tiny fraction of model parameters while retaining the base model's general capabilities.

### 3.1 LoRA (Low-Rank Adaptation)

**Core Concept:**
Given a pre-trained weight matrix W ∈ ℝ^{d×k}, LoRA constrains its update ΔW to be low-rank:
```
ΔW = BA, where B ∈ ℝ^{d×r}, A ∈ ℝ^{r×k}, and r ≪ min(d, k)
```
The forward pass becomes:
```
h = Wx + BAx
```
Only A and B are trained; W is frozen.

**Hyperparameters in Detail:**

#### Rank (r)

The rank r controls the representational capacity of the adapter. Enterprise practitioners should think of r as a budget parameter: more rank = more expressivity, more parameters, more VRAM, more regularization needed.

| Rank | Parameters (7B, q_proj+v_proj) | Typical Use Case |
|---|---|---|
| 1 | ~0.5M | Simple domain adaptation, very sparse data (<100 examples) |
| 8 | ~4M | Default starting point for most tasks |
| 16 | ~8M | Complex instruction following, multi-task |
| 32 | ~16M | High-difficulty domain adaptation (legal, medical) |
| 64 | ~32M | Near full-fine-tune quality, large datasets (50K+) |
| 128 | ~64M | Overkill for most tasks; consider full fine-tuning |
| 256 | ~128M | Rarely needed; full fine-tuning often better |

**Empirical rule**: Start with r=8 for data <10K examples, r=16 for 10K–50K, r=32+ for 50K+. In practice, r=16 is the "sweet spot" for most enterprise use cases.

#### Alpha (α)

LoRA scaling factor determines how much the adaptation affects the forward pass:
```
h = Wx + (α/r) * BAx
```

- **α = r**: Default (1:1 scaling). No amplification.
- **α = 2r**: Doubles the effective learning rate of the adapter.
- **α = 16, r=8**: Scale factor of 2.
- **α = 32, r=8**: Scale factor of 4.

**Best practices:**
- Keep α proportional to r (α = r, 2r, or 4r)
- Higher α/r ratio = stronger regularization effect (smaller effective updates)
- Lower α/r ratio = faster convergence but potential instability
- Common enterprise setting: α = 16, r = 16

#### Dropout

LoRA dropout applies dropout to the adapter output:
```
h = Wx + dropout(BAx, p)
```

| Dropout Rate | Effect |
|---|---|
| 0.0 | Fast convergence, overfitting risk on small data |
| 0.05 | Mild regularization, good for 10K+ examples |
| 0.1 | Default for most cases |
| 0.2 | Heavy regularization, small datasets (<1K) |
| 0.3+ | Very aggressive; rarely needed |

#### Target Modules

Which weight matrices to adapt determines both cost and quality.

**Common choices for transformer architectures:**

```
# Llama 2/3 architecture
target_modules = [
    "q_proj",      # Query projection — most impactful
    "v_proj",      # Value projection — important for output
    "k_proj",      # Key projection — less critical than Q/V
    "o_proj",      # Output projection — subtle improvements
    "gate_proj",   # MLP gate — significant for reasoning
    "up_proj",     # MLP up projection
    "down_proj",   # MLP down projection
]
```

| Module Set | Parameters Added (7B, r=16) | Quality Impact |
|---|---|---|
| q_proj + v_proj | ~8M | Good baseline |
| q_proj + v_proj + o_proj | ~12M | Better |
| All attention (q,k,v,o) | ~16M | Standard choice |
| All attention + MLP (gate,up,down) | ~48M | Full adapter (near full-FT quality) |
| Every linear layer | ~80M+ | Aggressive; marginal gains |

**Enterprise recommendation**: Start with `q_proj, v_proj` and add `o_proj, k_proj` if quality is insufficient. Add MLP modules (`gate_proj, up_proj, down_proj`) only for complex reasoning tasks where attention-only adaptation plateaus.

### 3.2 QLoRA (Quantized LoRA)

QLoRA enables fine-tuning of large models (65B, 70B, 140B+) on a single consumer GPU by quantizing the base model to 4-bit while keeping LoRA adapters in full precision.

**4-bit NormalFloat (NF4)**

NF4 is a quantization data type designed for normally distributed weights. Unlike uniform quantization (INT4), NF4 allocates more quantization levels near zero where weights concentrate:

```
NF4 codebook: [-1.0, -0.696, -0.525, -0.393, -0.278, -0.174, -0.078, 0.0,
               0.078, 0.174, 0.278, 0.393, 0.525, 0.696, 1.0]
```

**Double Quantization**

QLoRA introduces double quantization: the quantization constants (scale factors) themselves are quantized to 8-bit, saving another ~0.5 bits per parameter:

1. First quantization: FP16 weights → NF4 (4-bit) — scale factors in FP32
2. Second quantization: FP32 scale factors → INT8 (8-bit) — second-level scale factors in FP32

Total savings: ~4.5 bits per parameter vs 16-bit (FP16) baseline.

**Memory Savings Breakdown**

| Model | FP16 | LoRA (r=16) | QLoRA (4-bit NF4) |
|---|---|---|---|
| 7B | 14 GB | ~16 GB | ~6.5 GB |
| 13B | 26 GB | ~28 GB | ~10.5 GB |
| 34B | 68 GB | ~70 GB | ~24 GB |
| 70B | 140 GB | ~142 GB | ~48 GB |
| 180B | 360 GB | ~362 GB | ~120 GB |

QLoRA typically achieves within 0.5–2% of LoRA quality on most benchmarks while using ~4x less memory.

**Nested Quantization**

For extreme memory savings, the base model can be quantized to 3-bit or 2-bit with modest quality degradation:
- NF3 (3-bit): ~3.5 bits/param after double quantization
- NF2 (2-bit): ~2.5 bits/param after double quantization

These are experimental and recommended only for large models (70B+) where 4-bit still exceeds available VRAM.

**When to Use QLoRA vs LoRA**

| Scenario | Recommended |
|---|---|
| Consumer GPU (24 GB or less) | QLoRA |
| Enterprise GPU (40 GB A100, 80 GB H100) | LoRA (for quality) or QLoRA (for throughput) |
| Multiple models on one GPU | QLoRA |
| Maximum quality required | LoRA |
| 70B+ models on limited hardware | QLoRA |
| Production training pipeline | LoRA (less complexity) |

### 3.3 DoRA (Weight-Decomposed Low-Rank Adaptation)

DoRA decomposes pretrained weights into magnitude and direction components and applies LoRA only to the directional component:

```
W = m * V / ||V||  where V = W + ΔW (LoRA update)
```

**Key Advantages:**
- **Directional updates**: LoRA changes both magnitude and direction; DoRA isolates directional learning
- **Faster convergence**: 1.5–2x faster than LoRA on some benchmarks
- **Better quality**: Consistently matches or exceeds LoRA across NLU and NLG tasks
- **Parameter count**: Same as LoRA (magnitude vector adds negligible parameters, one per row)

**Enterprise Recommendation:**
DoRA is the default choice over LoRA when available. The additional training stability and faster convergence reduce GPU hours directly. Supported in libraries like PEFT (via `use_dora=True`), Axolotl (via `lora_dora: true`).

**Implementation in PEFT:**
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    use_dora=True,  # Enable DoRA
)
```

### 3.4 AdaLoRA (Adaptive Budget Allocation)

AdaLoRA parametrizes weight updates through SVD and dynamically allocates the rank budget across layers based on importance:

```
W = W₀ + PΛQ, where Λ is a diagonal matrix of singular values
```

**Key Mechanisms:**
- **Regularization**: L0 penalty on singular values encourages sparsity
- **Dynamic budget**: Layers with higher importance get higher effective rank
- **SVD-based**: Maintains orthogonality constraints on P and Q

**Empirical Results:**
- Often outperforms LoRA with the same total rank budget
- Automatically assigns more rank to early layers and attention layers
- Less sensitive to rank choice (use r=16 or r=32 and let it prune)

**Trade-offs:**
- +20-30% training overhead over LoRA
- More hyperparameters (L0 regularization coefficient, target rank)
- Not supported in all frameworks

**Enterprise Recommendation:**
Best for tasks where the optimal rank is unknown and you have budget to run AdaLoRA experiments. For well-understood tasks, standard LoRA/DoRA is simpler.

### 3.5 IA3 (Infused Adapter by Inhibiting and Amplifying Activations)

IA3 learns scaling vectors that modulate key activations:

```
h = Wx ⊗ l, where l is a learned scaling vector
```

**Compared to LoRA:**
- **Fewer parameters**: ~10,000x fewer than full fine-tuning (vs 1000x for LoRA)
- **Faster**: No matrix multiplication overhead for adapters
- **Weaker on complex tasks**: Struggles with multi-task reasoning

**When to Use IA3:**
- Extremely limited storage (thousands of adapters)
- Simple domain adaptation (tone, format, style)
- Feature extraction where base model capabilities are sufficient
- Multi-adapter serving with minimal overhead

---

## 4. Data Preparation Pipeline

Data quality is the single largest determinant of fine-tuning success. A well-prepared dataset consistently outperforms a 10x larger noisy dataset.

### 4.1 Data Collection

Enterprise data sources span diverse formats and quality levels.

| Source Type | Examples | Volume | Quality | PII Risk |
|---|---|---|---|---|
| **Conversation logs** | Chat transcripts, support tickets | 100K–10M | Medium | High |
| **Internal documents** | Wikis, runbooks, SOPs | 1K–100K | Mixed | Low–Medium |
| **Code repositories** | Git history, Jupyter notebooks | 10K–1M | High | Low |
| **Emails** | Correspondence, newsletters | 10K–100K | Medium | Very High |
| **Database records** | CRM, ERP, logs | 1M+ | Variable | High |
| **Public datasets** | Benchmarks, open-source | 10K–10M | High | Low |
| **Feedback loops** | User ratings, corrections | 1K–100K | High | Medium |

**Collection Best Practices:**
- Capture full conversation context (not just individual turns)
- Preserve metadata (timestamps, user IDs – for deduplication only)
- Log model-generated completions separately (avoid contamination)
- Implement data retention policies aligned with GDPR/CCPA
- Version control raw data with data manifests

### 4.2 Data Cleaning

#### Deduplication

Enterprise datasets often contain massive redundancy due to repeated logging, template-driven content, and feedback loops.

**Levels of Deduplication:**

1. **Exact deduplication**: MD5/SHA256 hash comparison — removes identical strings
2. **Near-dedup (MinHash)**: Jaccard similarity >0.8 — removes near-identical documents
3. **Template deduplication**: Regex-based removal of auto-generated templates
4. **Semantic deduplication**: Embedding similarity >0.95 — removes paraphrased content

**MinHash Implementation:**
```python
from datasketch import MinHash, MinHashLSH

def compute_minhash(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for ngram in ngrams(text.split(), 5):
        m.update(' '.join(ngram).encode('utf-8'))
    return m

# LSH for efficient near-dedup
lsh = MinHashLSH(threshold=0.8, num_perm=128)
for idx, text in enumerate(documents):
    m = compute_minhash(text)
    lsh.insert(f"doc_{idx}", m)
```

**Deduplication Statistics:**
- Typical enterprise dataset: 20–40% redundancy
- Log data: often 60–80% redundant
- After MinHash (0.8 threshold): typically removes 15–30%

#### Formatting and Encoding

- **Encoding detection**: Use `charset-normalizer` or `cchardet` to detect and standardize to UTF-8
- **Whitespace normalization**: Collapse multiple spaces, normalize newlines (\r\n → \n)
- **Unicode normalization**: NFC normalization (composed) for consistent character representation
- **Markdown/HTML stripping**: Remove formatting tags, keep text content
- **JSON/XML extraction**: Parse structured data and extract text fields

```python
import chardet
import unicodedata

def normalize_text(text):
    # Detect and fix encoding
    if isinstance(text, bytes):
        detected = chardet.detect(text)
        text = text.decode(detected['encoding'] or 'utf-8', errors='replace')
    
    # Unicode normalization
    text = unicodedata.normalize('NFC', text)
    
    # Whitespace normalization
    text = ' '.join(text.split())
    
    return text
```

#### PII Redaction

Personally Identifiable Information (PII) must be removed before any training, especially for regulated industries.

**Redaction Pipeline (Presidio + spaCy):**

```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def redact_pii(text):
    results = analyzer.analyze(text=text, language='en')
    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results
    )
    return anonymized.text
```

**PII Types to Detect and Redact:**

| PII Type | Regex/NER Approach | Enterprise Requirement |
|---|---|---|
| Email addresses | Regex: `[\w.+-]+@[\w-]+\.[\w.-]+` | Always |
| Phone numbers | Regex: `\+?\d{1,4}[\s-]?\d{3,4}[\s-]?\d{4,10}` | Always |
| SSN / Tax IDs | Regex: `\d{3}-\d{2}-\d{4}` | Always (US) |
| Credit card numbers | Luhn algorithm + regex | Always (PCI) |
| IP addresses | Regex: `\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}` | Always |
| Names | spaCy NER (PERSON) | Usually |
| Locations | spaCy NER (GPE, LOC) | Usually |
| Dates (medical) | Regex + context | HIPAA |
| Medical record numbers | Pattern-based + context | HIPAA |
| Bank account numbers | Regex (varies by country) | Usually |

**Best Practices:**
- Use both regex (high recall) and NER (high precision) in sequence
- Validate redaction on a holdout set before training
- For LLM training, redact with placeholder tags like `<PERSON>` or `[REDACTED]`
- Never train on non-redacted data, even in a sandbox
- Implement differential privacy as an additional safeguard

#### Language Filtering

Multilingual enterprises need to filter or classify training data by language.

```python
import fasttext

# Language identification model (e.g., lid.176.bin from fasttext)
lang_model = fasttext.load_model('lid.176.bin')

def detect_language(text, threshold=0.5):
    predictions = lang_model.predict(text.replace('\n', ' '), k=1)
    lang = predictions[0][0].replace('__label__', '')
    score = predictions[1][0]
    return lang, score

# Filter for English-only or specific language mix
filtered = [
    text for text in corpus
    if detect_language(text)[0] == 'en' and detect_language(text)[1] > 0.7
]
```

**Language Distribution Heuristics:**
- Use language distribution of target users
- Avoid mixing languages in single training example (unless target task is translation)
- For multilingual models, balance languages by sampling rate

### 4.3 Quality Filtering

#### Heuristic Filtering

Simple, fast quality signals that remove obvious garbage:

| Heuristic | Threshold | Rationale |
|---|---|---|
| Minimum length | >50 characters | Too-short text lacks context |
| Maximum length | <8192 tokens | Context window limits |
| Word count | >20 words | Sentence-level content only |
| Average word length | 3–10 chars | 2-char avg = noise, 15+ char avg = code |
| Punctuation ratio | 5–40% | Low = no structure, high = table/data |
| Repeated characters | <5% of text | "aaaaa" or "!!!!!" = noise |
| Repeated n-grams (4-gram) | <40% | Template text |
| HTML tag ratio | <30% | Scraped markup |
| Non-printable chars | <1% | Binary content |
| Stop word ratio (English) | >5% | Non-English or gibberish |

```python
import re
from collections import Counter

def heuristic_filter(text):
    if len(text) < 50:
        return False
    if len(text) > 32000:
        return False
    
    words = text.split()
    if len(words) < 20:
        return False
    
    avg_word_len = sum(len(w) for w in words) / len(words)
    if avg_word_len < 3 or avg_word_len > 15:
        return False
    
    # Count repeated chars
    repeated = sum(1 for c in text if text.count(c) > len(text) * 0.05)
    if repeated > 0:
        return False
    
    # Check for non-printable
    non_printable = sum(1 for c in text if not c.isprintable())
    if non_printable / len(text) > 0.01:
        return False
    
    return True
```

#### Perplexity Filtering

Uses a reference model (e.g., a small LM) to score document quality. Low perplexity = in-distribution for the reference model; high perplexity = out-of-distribution or noisy.

```
Quality filter: 20 < perplexity < 200 (for English natural language)
```

- **Too low (<20)**: Model already knows this by heart — template, repeated content, or test data leakage
- **Too high (>200)**: Gibberish, non-target language, or formatting errors

**Implementation:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def compute_perplexity(text):
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    
    return torch.exp(loss).item()

# Filter: keep perplexity between 20 and 200
filtered = [d for d in dataset if 20 < compute_perplexity(d['text']) < 200]
```

**Enterprise Note:** Perplexity filtering introduces bias toward the reference model's distribution. Use a generic model (GPT-2, OLMo) rather than your target model to avoid amplifying existing biases.

#### Classifier-based Filtering

Train or use an existing classifier to predict data quality:

**Classification Labels:**
- **0** = garbage (encoding errors, spam, gibberish)
- **1** = low quality (shallow, poorly written)
- **2** = medium quality (acceptable, typical web text)
- **3** = high quality (expert-level, well-structured)

**Approaches:**
1. **Fine-tune a small LM** (e.g., DistilBERT) on manually labeled samples (~1K per class)
2. **LLM-as-judge**: Use GPT-4 or Claude to rate quality on 1–5 scale
3. **Contrastive filtering**: Use model's own loss as signal (fine-tuned on good data)

**Production Pipeline:**
```
Raw Data → Heuristic Filter (fast) → Perplexity Filter → Classifier Filter (best)
                                      Falls through:     Falls through:
                                      50-70% kept        20-50% kept
```

### 4.4 Format Conversion

Different training frameworks expect different dataset formats. Conversion between these formats is a common source of errors.

#### ShareGPT Format

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "What is fine-tuning?"
    },
    {
      "from": "gpt",
      "value": "Fine-tuning is the process of adapting a pre-trained model..."
    },
    {
      "from": "human",
      "value": "How do I start?"
    },
    {
      "from": "gpt",
      "value": "First, prepare your dataset..."
    }
  ]
}
```

**Use case**: Multi-turn conversations, chat models
**Supported by**: Axolotl, LLaMA-Factory, Firefly

#### Alpaca Format

```json
{
  "instruction": "Explain the concept of fine-tuning.",
  "input": "",
  "output": "Fine-tuning is the process of adapting a pre-trained model to a specific domain or task by continuing the training process on domain-specific data."
}
```

```json
{
  "instruction": "Summarize the following text.",
  "input": "Fine-tuning is a technique where...",
  "output": "Fine-tuning adapts pre-trained models to specific tasks."
}
```

**Use case**: Single-turn instruction following
**Variants**: 
- `instruction` only (no `input`)
- `instruction` + `input` (input goes into prompt before output)

#### ChatML Format

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is fine-tuning?<|im_end|>
<|im_start|>assistant
Fine-tuning is the process of adapting a pre-trained model...<|im_end|>
<|im_start|>user
How do I start?<|im_end|>
<|im_start|>assistant
First, prepare your dataset...<|im_end|>
```

**Use case**: Structured chat with system prompt
**Token efficiency**: Slightly more tokens than ShareGPT due to formatting tokens

#### Messages Format (OpenAI-style)

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is fine-tuning?"},
    {"role": "assistant", "content": "Fine-tuning is the process..."}
  ]
}
```

**Use case**: OpenAI API fine-tuning, most modern LLM APIs
**Supported by**: OpenAI, Together AI, Fireworks, Anyscale

### 4.5 Train/Validation/Test Splits

**Split Strategy:**

| Dataset Size | Train | Validation | Test | Notes |
|---|---|---|---|---|
| <1K | 60% | 20% | 20% | High variance; consider cross-validation |
| 1K–10K | 80% | 10% | 10% | Standard |
| 10K–100K | 85% | 7.5% | 7.5% | Larger val/test for stability |
| 100K+ | 90% | 5% | 5% | Even 1% of 1M = 10K test samples |

**Stratification Considerations:**
- Stratify by domain, task type, or data source
- Ensure all rare classes appear in training (not just val/test)
- Temporal split (by date) for time-series data to avoid leakage
- Speaker/author split to avoid evaluating on familiar data

**Implementation:**
```python
from sklearn.model_selection import train_test_split

def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, stratify_col=None):
    # First split: train vs temp
    train, temp = train_test_split(
        dataset, 
        train_size=train_ratio,
        stratify=stratify_col,
        random_state=42
    )
    
    # Second split: val vs test from temp
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val, test = train_test_split(
        temp,
        train_size=val_ratio_adjusted,
        stratify=stratify_col(if stratify_col else None),
        random_state=42
    )
    
    return train, val, test
```

**Leakage Detection:**
- Check n-gram overlap between train and test sets
- Compute embedding similarity between train/test pairs
- Flag exact match duplicates across splits
- Monitor test loss < train loss (suggests leakage or data errors)

---

## 5. Synthetic Data Generation

Synthetic data is increasingly critical for enterprise fine-tuning, especially when real data is scarce, sensitive, or expensive to annotate.

### 5.1 Self-Instruct Pipeline

**Original Process (Wang et al., 2022):**

1. **Seed tasks**: Write 175 seed instructions (hand-crafted)
2. **Instruction generation**: For each seed, prompt an LLM to generate more instructions
3. **Classification**: Classify whether the instruction is a classification or generation task
4. **Output generation**: For each instruction, prompt the LLM to produce an output
5. **Filtering**: Remove low-quality or duplicate entries
6. **Iterate**: The generated dataset becomes new seeds

**Enterprise Implementation:**

```python
class SelfInstruct:
    def __init__(self, base_model, seed_tasks):
        self.model = base_model
        self.seed_tasks = seed_tasks
        self.instructions = []
        
    def generate_instructions(self, num_to_generate=1000):
        for seed in self.seed_tasks:
            prompt = f"""
            Generate a new instruction for an AI assistant.
            The instruction should be similar in difficulty and style to:
            
            Instruction: {seed}
            
            New instruction:
            """
            response = self.model.generate(prompt)
            self.instructions.append(response.strip())
            
    def generate_responses(self):
        for instruction in self.instructions:
            prompt = f"""
            Instruction: {instruction}
            
            Provide a helpful, accurate response to the above instruction.
            Response:
            """
            response = self.model.generate(prompt)
            dataset.append({"instruction": instruction, "output": response})
```

**Quality Controls:**
- Filter instructions with keyword overlap >80% with existing set
- Remove instructions that are too short (<5 words) or too long (>50 words)
- Balance across task categories
- Verify executability (can the model actually complete the task?)

### 5.2 Evol-Instruct

Evol-Instruct generates training data by progressively making instructions more complex (in-depth + in-breadth evolution).

#### In-Depth Evolution

Makes a single instruction harder by adding constraints, deepening reasoning, or increasing difficulty:

| Evolution Operation | Example Transformation |
|---|---|
| **Add constraints** | "Write a poem" → "Write a haiku about AI using exactly 17 syllables" |
| **Deepen reasoning** | "Explain gravity" → "Explain how gravitational waves differ from Newtonian predictions of gravity" |
| **Increase difficulty** | "Translate to French" → "Translate this legal contract to French, maintaining all technical terms and preserving the legal validity" |
| **Add specific format** | "Summarize" → "Summarize in exactly 3 bullet points with a 50-word limit per bullet" |
| **Add comparison** | "Describe Python" → "Compare Python and Rust for systems programming, considering memory safety, performance, and developer productivity" |

#### In-Breadth Evolution

Generates new, diverse instructions from existing ones without increasing difficulty:

| Evolution Operation | Example Transformation |
|---|---|
| **Concretization** | "Write a function" → "Write a function to calculate Fibonacci numbers" |
| **Increase scope** | "Debug this code" → "Debug this code and write unit tests for edge cases" |
| **Switch domain** | "Write marketing copy" → "Write marketing copy for a B2B SaaS product in the healthcare space" |
| **Add background** | "Explain transformers" → "Explain the transformer architecture to a high school student with no ML background" |
| **Scenario-based** | "Create a plan" → "Create a project plan for migrating a monolith to microservices" |

**Quality Control in Evol-Instruct:**
- LLM Judge scores each evolved instruction (1–5)
- Remove instructions scoring <3
- Remove instructions identical to (or too similar to) originals
- Verify instruction is still solvable after evolution
- Balance difficulty distribution

### 5.3 Wizard Methods

The "Wizard" family (WizardLM, WizardCoder, WizardMath) builds on Evol-Instruct with domain-specific extensions:

- **WizardLM**: Evol-Instruct on general instructions
- **WizardCoder**: Evol-Instruct on code generation instructions, with test-based verification
- **WizardMath**: Evol-Instruct on math problems, with step-by-step verification

**Key Innovation — Complexity Scoring:**
```
Complexity Score = f(num_constraints, reasoning_depth, domain_specificity, format_restrictiveness)
```
Each evolved instruction's complexity is tracked, and the training set is balanced across complexity levels.

### 5.4 Knowledge Distillation from Large Models

Distillation leverages the fact that larger models (teacher) often produce higher quality outputs than smaller models (student).

**Distillation Pipeline:**

1. **Teacher selection**: GPT-4, Claude 3.5 Sonnet, Gemini Ultra, or the largest available model
2. **Prompt construction**: Craft prompts that elicit detailed, correct responses
3. **Batch generation**: Generate outputs at scale (cost-efficient via API)
4. **Quality filtering**: Remove obviously incorrect or low-quality outputs
5. **Student training**: Fine-tune smaller model (e.g., Llama 3.1 8B) on teacher outputs

**Distillation Strategies:**

| Strategy | Description | Quality vs Cost |
|---|---|---|
| **Direct distillation** | Train on teacher outputs directly | Medium quality, low cost |
| **Contrastive distillation** | Train on (good output, bad output) pairs | Higher quality, higher cost |
| **Multi-teacher ensemble** | Use majority vote across teachers | Highest quality, highest cost |
| **Self-distillation** | Student generates, teacher scores | Iterative improvement |

**Enterprise Implementation Considerations:**
- **Cost**: GPT-4 distillation for 100K examples ≈ $5,000–$15,000 in API costs
- **Rate limits**: Need multiple API keys or batch API for large volumes
- **Bias propagation**: Teacher biases transfer to student — audit carefully
- **Licensing**: Check teacher's terms of service for distillation rights

### 5.5 Multi-turn Conversation Synthesis

Synthetic multi-turn conversations are essential for training chat-oriented models.

**Conversation Tree Generation:**

```
User: I need help with my account.
Assistant: I'd be happy to help! What seems to be the issue?
User: I can't log in. It says "invalid password."
Assistant: Let's try resetting your password. Can you access your email?
User: Yes, I can.
Assistant: Great. I've sent a password reset link to your registered email. Please check and follow the instructions.
User: It worked! I'm logged in now. Thanks.
Assistant: You're welcome! Is there anything else I can help you with?
```

**Generation Approaches:**

1. **Role-play**: Two model instances — one as user (with persona), one as assistant
2. **Template-based**: Fill in slots in conversation templates (good for customer support)
3. **Instruct-then-generate**: Start with a high-level scenario and let model generate full conversation
4. **Branching**: Generate multiple possible continuations at each turn, select best path

**Quality Control:**
- Require ≥3 turns for "conversation" qualification
- Check for assistant response quality (helpful, accurate, safe)
- Remove conversations where user repeats the same question
- Ensure conversation reaches resolution or proper escalation

### 5.6 Quality Control

Synthetic data quality control is an ongoing process, not a one-time filter.

#### LLM Judge

Use a LLM (often the same model or a larger one) to evaluate synthetic data quality:

```python
def llm_judge(instruction, response, criteria):
    prompt = f"""
    Evaluate the following response to an instruction on these criteria:
    
    Instruction: {instruction}
    Response: {response}
    
    Criteria:
    1. Helpfulness (1-5): Does the response address the instruction?
    2. Accuracy (1-5): Is the information correct?
    3. Completeness (1-5): Is the response thorough?
    4. Safety (1-5): Is the response safe and ethical?
    
    Format your response as: 
    Helpfulness: X
    Accuracy: X
    Completeness: X
    Safety: X
    """
    
    evaluation = judge_model.generate(prompt)
    return parse_evaluation(evaluation)
```

#### Human Verification Threshold

For enterprise production, establish a human verification threshold:

| Quality Score | Action |
|---|---|
| 4.5–5.0 | Auto-accept |
| 3.5–4.5 | Auto-accept with sampling (1 in 10 human check) |
| 2.5–3.5 | Human review required |
| 1.0–2.5 | Auto-reject |

#### Quality Metrics Dashboard

Track these metrics over time across data runs:

- **Acceptance rate**: % passing quality filters
- **Distribution coverage**: Are all domains/topics represented?
- **Difficulty distribution**: Are there enough hard examples?
- **Token length distribution**: Consistent with target task?
- **Model agreement**: Do multiple LLM judges agree?
- **Human agreement**: When humans review, do they concur with LLM judge?

---

## 6. Human Annotation

Human annotation remains essential for high-quality supervised fine-tuning, especially for tasks requiring domain expertise or nuanced judgment.

### 6.1 Annotation Platform Comparison

#### LabelStudio

| Feature | Details |
|---|---|
| **License** | Apache 2.0 (open source) |
| **Self-hosted** | Yes (Python/Django) |
| **Annotation types** | Text classification, NER, sequence labeling, transcription, QA |
| **LLM features** | ML-assisted labeling, LLM pre-annotations |
| **Collaboration** | Teams, review workflows, project roles |
| **Integration** | API SDK, Storage (S3, GCS, Azure), Cloud (pre-built images) |
| **Cost** | Free (self-hosted). LabelStudio Enterprise has paid tiers. |
| **Best for** | Teams that want full control, on-prem deployment |

**Setup:**
```bash
pip install label-studio
label-studio start --init  # Initialize with admin user
```

#### Prodigy

| Feature | Details |
|---|---|
| **License** | Proprietary (commercial) |
| **Self-hosted** | Yes (Python library) |
| **Annotation types** | NER, text classification, image annotation, audio, video |
| **LLM features** | Built-in recipe system, LLM integration (via recipe), active learning |
| **Collaboration** | Multi-user with Prodigy Teams |
| **Integration** | spaCy native integration, custom recipes |
| **Cost** | ~$1,000–5,000/year depending on license tier |
| **Best for** | Teams already in spaCy ecosystem, rapid prototyping |

**Example recipe:**
```python
import prodigy
from prodigy.components.loaders import JSONL

@prodigy.recipe('text-classification')
def text_classification(dataset, source):
    stream = JSONL(source)
    return {
        'dataset': dataset,
        'stream': stream,
        'view_id': 'classification',
        'config': {
            'labels': ['relevant', 'irrelevant']
        }
    }
```

#### Scale AI (Scale Rapid)

| Feature | Details |
|---|---|
| **License** | Proprietary (SaaS) |
| **Self-hosted** | No (cloud-only) |
| **Annotation types** | All types (text, image, video, 3D, audio, document) |
| **LLM features** | Scale Spellbook (RLHF, prompt engineering, evaluation) |
| **Collaboration** | Managed workforce, expert pool, automated QA |
| **Integration** | API, webhooks, data connectors |
| **Cost** | Pay-per-task (variable); enterprise contracts |
| **Best for** | Large-scale annotation with managed workforce |

#### Snorkel AI (now Weights & Biases)

| Feature | Details |
|---|---|
| **License** | Proprietary |
| **Self-hosted** | Enterprise option |
| **Annotation types** | Data development, labeling functions, slicing |
| **LLM features** | Foundation model labeling, programmatic labeling |
| **Collaboration** | Team-based, labeling function sharing |
| **Integration** | Weights & Biases ecosystem |
| **Cost** | Enterprise pricing |
| **Best for** | Weak supervision + human validation at scale |

### 6.2 Inter-Annotator Agreement

Measuring agreement between annotators is critical for dataset quality.

#### Cohen's Kappa (κ)

For two annotators, Cohen's Kappa measures agreement beyond chance:

```
κ = (p₀ - pₑ) / (1 - pₑ)
```

Where:
- p₀ = observed agreement
- pₑ = expected agreement by chance

| κ Value | Agreement |
|---|---|
| < 0 | Poor |
| 0.0–0.20 | Slight |
| 0.21–0.40 | Fair |
| 0.41–0.60 | Moderate |
| 0.61–0.80 | Substantial |
| 0.81–1.00 | Almost perfect |

**Implementation:**
```python
from sklearn.metrics import cohen_kappa_score

# Annotations from two annotators
annotator_1 = [0, 1, 1, 0, 2, 1, 0, 1]
annotator_2 = [0, 1, 0, 0, 2, 1, 1, 1]

kappa = cohen_kappa_score(annotator_1, annotator_2)
print(f"Cohen's Kappa: {kappa:.3f}")
```

**Limitations:**
- Kappa is affected by prevalence (if most items are the same class, κ is low even with high agreement)
- Only handles two annotators; use Fleiss' Kappa for >2

#### Krippendorff's Alpha (α)

A more robust agreement metric that handles >2 annotators, missing data, and any scale type (nominal, ordinal, interval, ratio):

```
α = 1 - Dₒ / Dₑ
```

Where:
- Dₒ = observed disagreement
- Dₑ = expected disagreement by chance

**Implementation:**
```python
import krippendorff

# Annotations: items as rows, annotators as columns
# NaN for missing annotations
data = np.array([
    [1, 1, 1, np.nan],  # Item 1
    [2, 2, 2, 2],       # Item 2
    [1, 2, 1, 2],       # Item 3
    [2, 2, 1, 2],       # Item 4
])

alpha = krippendorff.alpha(data, level_of_measurement='nominal')
print(f"Krippendorff's Alpha: {alpha:.3f}")
```

**Enterprise Thresholds:**
- α > 0.8: Good reliability; dataset is consistent
- 0.67 ≤ α ≤ 0.80: Tentative reliability; investigate disagreements
- α < 0.67: Poor reliability; re-train annotators or simplify task

### 6.3 Quality Assurance Workflows

**Three-stage QA Pipeline:**

1. **Training stage**: Annotators train on a gold-standard set of 50–100 examples. Must achieve κ > 0.7 on a qualification test.
2. **Production stage**: Annotations flow through automated checks:
   - Validation against schema (required fields, data types)
   - Real-time agreement check (if multiple annotators)
   - Confidence scoring (model predicts label; flag low-confidence annotations)
3. **Review stage**: Random sample (5–20%) reviewed by senior annotator or domain expert

**Blind Spots to Monitor:**
- **Drift**: Annotator agreement decreasing over time (need re-calibration)
- **Edge cases**: Systematic disagreement on specific data types
- **Annotation fatigue**: Degradation toward end of shift (flag batches completed in last hour)
- **Cascading errors**: Errors in early steps propagate

### 6.4 Active Learning

Active learning reduces annotation costs by selecting the most informative examples for human labeling.

#### Uncertainty Sampling

Select examples where the model is most uncertain:

```python
def uncertainty_sampling(model, unlabeled_pool, n_samples=100):
    predictions = []
    for example in unlabeled_pool:
        probs = model.predict_proba(example)
        # Margin: difference between top 2 probabilities
        sorted_probs = np.sort(probs)[::-1]
        margin = sorted_probs[0] - sorted_probs[1]
        predictions.append(margin)
    
    # Select examples with smallest margin (highest uncertainty)
    indices = np.argsort(predictions)[:n_samples]
    return [unlabeled_pool[i] for i in indices]
```

**Variants:**
- **Least confidence**: 1 - max(probability)
- **Margin sampling**: Difference between top 2 probabilities
- **Entropy**: -Σ p(x) log p(x)
- **BALD**: Bayesian Active Learning by Disagreement (for neural networks)

#### Diversity Sampling

Ensure the selected batch covers diverse data regions:

```python
def diversity_sampling(embeddings, n_samples=100):
    # k-means centroids as diverse representatives
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters=n_samples, random_state=42)
    kmeans.fit(embeddings)
    
    # Find closest example to each centroid
    indices = []
    for centroid in kmeans.cluster_centers_:
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        closest = np.argmin(distances)
        indices.append(closest)
    
    return indices
```

#### Expected Model Change

Select examples that would most change the model parameters:

```
Expected Gradient Length (EGL):
Select x that maximizes ||∇ℓ(x, ŷ; θ)||
```

This is computationally expensive but highly effective for deep neural networks.

**Enterprise Active Learning Loop:**

```
1. Train initial model on small annotated seed set
2. Score unlabeled pool with uncertainty/diversity metrics
3. Select top-K examples for annotation
4. Annotate (human-in-the-loop)
5. Add annotations to training set
6. Re-train or update model
7. Repeat until performance plateaus
```

**Typical Savings:**
- Active learning reduces annotation needs by 40–80% compared to random sampling
- For NER tasks: 60% less data for same F1 score
- For classification: 50–70% less data

---

## 7. Training Infrastructure

### 7.1 Cloud ML Platforms

#### AWS SageMaker

**Key Features:**
- Managed training jobs with automatic instance provisioning
- SageMaker Studio for interactive development
- Built-in Deep Learning Containers (DLCs)
- Distributed Training Libraries (SMDDP for data parallelism)
- Managed Spot Training (up to 90% cost reduction)
- Model registry and deployment pipeline

**Fine-tuning on SageMaker:**
```python
import sagemaker
from sagemaker.huggingface import HuggingFace

sagemaker_session = sagemaker.Session()

hf_estimator = HuggingFace(
    entry_point='train.py',
    source_dir='./scripts',
    instance_type='ml.g5.12xlarge',
    instance_count=2,
    role=role,
    transformers_version='4.28.0',
    pytorch_version='2.0.0',
    hyperparameters={
        'model_id': 'meta-llama/Llama-2-7b-hf',
        'dataset_path': '/opt/ml/input/data/training',
        'epochs': 3,
        'per_device_train_batch_size': 4,
        'gradient_accumulation_steps': 4,
        'learning_rate': 2e-4,
        'lora_r': 16,
        'lora_alpha': 32,
    },
    use_spot_instances=True,
    max_wait=86400,
    checkpoint_s3_uri='s3://my-bucket/checkpoints/',
)

hf_estimator.fit({'training': 's3://my-bucket/training-data/'})
```

**Pricing Considerations:**
- `ml.g5.12xlarge` (4× A10G, 192 GB VRAM): ~$6.12/hr on-demand, ~$1.84/hr spot
- `ml.p4d.24xlarge` (8× A100 40GB, 320 GB VRAM): ~$37.69/hr on-demand
- `ml.p5.48xlarge` (8× H100 80GB, 640 GB VRAM): ~$132.10/hr on-demand

#### GCP Vertex AI

**Key Features:**
- Custom jobs with any Docker image
- Hyperparameter tuning (Vizier)
- Custom model serving with prediction endpoints
- Deep Learning VM images with pre-installed drivers
- TPU v4/v5e/v5p support for LLM training

**Fine-tuning on Vertex AI:**
```bash
gcloud ai custom-jobs create \
    --region=us-central1 \
    --display-name=llm-fine-tune \
    --worker-pool-spec=machine-type=a2-highgpu-8g,replica-count=1,container-image-uri=gcr.io/my-project/llm-training:latest \
    --args="--model=meta-llama/Llama-2-13b-hf,--dataset=gs://my-bucket/train.jsonl"
```

#### Azure ML

**Key Features:**
- Azure OpenAI Service for managed LLM fine-tuning
- Azure ML Compute (NC/HB/NV series)
- Integration with Responsible AI dashboard
- Managed online endpoints for deployment

**Note:** Azure has the tightest integration with Microsoft's own models (Phi, Orca) and OpenAI models.

### 7.2 GPU Cloud Providers

#### Lambda GPU Cloud

| GPU Type | Specs | Price/hr |
|---|---|---|
| A100 (80 GB) | 1× A100 SXM 80GB | $1.10 |
| A100 (80 GB) 8× | 8× A100 SXM 80GB | $7.60 |
| H100 (80 GB) 8× | 8× H100 SXM 80GB | $13.60 |
| H200 (141 GB) 8× | 8× H200 SXM 141GB | $18.20 |

**Pros:** Reliable, predictable pricing, good support
**Cons:** Limited GPU variety, mostly NVIDIA only

#### RunPod

| GPU Type | Specs | Price/hr |
|---|---|---|
| RTX 4090 (24 GB) | 1× RTX 4090 | $0.34 |
| A100 (80 GB) | 1× A100 SXM | $0.99 |
| A100 8× (80 GB) | 8× A100 SXM | $7.29 |
| H100 8× (80 GB) | 8× H100 SXM | $11.99 |
| H200 8× (141 GB) | 8× H200 SXM | $15.99 |

**Pros:** Cheapest spot pricing, wide GPU selection, serverless GPU option
**Cons:** Network performance varies, no guaranteed availability

#### Vast.ai

**Pros:** Cheapest prices (p2p marketplace for GPU), enormous variety
**Cons:** Inconsistent hardware quality, no support, security concerns

#### CoreWeave

**Pros:** Premium networking (InfiniBand), purpose-built for LLM training, great for multi-node
**Cons:** Minimum commitments, less accessible for small projects

#### JarvisLabs

**Pros:** Good for single-GPU fine-tuning, RTX 4090s cheap
**Cons:** Limited multi-node options

### 7.3 On-Premises Infrastructure

#### NVIDIA DGX Systems

| System | GPUs | VRAM | Price |
|---|---|---|---|
| DGX H100 | 8× H100 80GB | 640 GB | ~$300K |
| DGX B200 | 8× B200 180GB | 1.4 TB | ~$500K |
| DGX A100 | 8× A100 80GB | 640 GB | ~$200K (used) |

#### Dell PowerEdge with NVIDIA GPUs

- PowerEdge R760xa (4× H100/NVIDIA L40S)
- PowerEdge XE9680 (8× H100)
- PowerEdge R750xa (4× A100)

#### Supermicro

- AS-4125GS-TNRT (8× H100/H200)
- SYS-421GU-TNXR (4× H100)
- SYS-420GP-TNR (8× A100)

**On-premise Considerations:**
- **Power**: H100 SXM runs at 700W per GPU; 8× H100 system = ~7–10 kW
- **Cooling**: Liquid cooling recommended for H100 clusters
- **Networking**: InfiniBand NDR400 (400 Gbps) for multi-node training
- **Rack space**: DGX H100 is 6U; expect 10+ racks for a full cluster
- **Maintenance**: Driver updates, hardware failures, cooling system upkeep

### 7.4 Spot / Preemptible Instances

Spot instances can reduce GPU costs by 60–90% but introduce interruption risk.

**Mitigation Strategies:**

| Strategy | Description | Overhead |
|---|---|---|
| **Checkpointing** | Save checkpoints every N steps (frequent) | Storage cost, ~30s per save |
| **Spot-to-ondemand fallback** | Switch to on-demand when spot is reclaimed | Higher cost on last attempt |
| **Multi-region** | Spread across regions for more spot capacity | Data transfer costs |
| **Model sharding** | If one GPU is reclaimed, stop all (FSDP/ZeRO) | Wasted compute on remaining |
| **Short training jobs** | Keep training time < 1 hour per run | Overhead of frequent restarts |

**Spot pricing (example, AWS us-east-1):**
| Instance | On-Demand/hr | Spot/hr (avg) |
|---|---|---|
| g5.12xlarge (4× A10G) | $6.12 | $1.84 |
| p4d.24xlarge (8× A100) | $37.69 | $11.30 |
| p5.48xlarge (8× H100) | $132.10 | $39.63 |

### 7.5 Multi-GPU Training Strategies

#### DeepSpeed (ZeRO)

ZeRO (Zero Redundancy Optimizer) partitions optimizer states, gradients, and parameters across GPUs.

**ZeRO Stages:**

| Stage | Partitions | Memory Savings | Comm Overhead | Best For |
|---|---|---|---|---|
| ZeRO-1 | Optimizer states only | ~4x | Low | Small models, small GPU count |
| ZeRO-2 | Optimizer states + gradients | ~8x | Medium | Most 7B–13B fine-tuning |
| ZeRO-3 | Optimizer states + gradients + params | ~16x~Nd | High | Large models (30B+), memory-bound |
| ZeRO-3 + offload | CPU/NVMe offloading of params | Near-infinite | Very High | Models too large for GPU memory |

**ZeRO Configuration:**
```json
{
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 500000000,
    "reduce_scatter": true,
    "reduce_bucket_size": 500000000,
    "overlap_comm": true,
    "contiguous_gradients": true
  }
}
```

**ZeRO-3 with CPU Offload:**
```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 100000000000,
    "stage3_max_live_parameters": 1000000000,
    "stage3_max_reuse_distance": 1000000000,
    "stage3_gather_16bit_weights_on_model_save": true
  }
}
```

#### FSDP (Fully Sharded Data Parallelism)

FSDP is PyTorch's native sharded training, conceptually similar to ZeRO-3.

```python
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

fsdp_config = {
    "sharding_strategy": ShardingStrategy.FULL_SHARD,  # FULL_SHARD, HYBRID_SHARD, NO_SHARD
    "mixed_precision": MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16,
    ),
    "auto_wrap_policy": transformer_auto_wrap_policy,
    "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
    "limit_all_gathers": True,
    "use_orig_params": True,
}
```

**FSDP vs DeepSpeed ZeRO:**

| Feature | FSDP | DeepSpeed ZeRO |
|---|---|---|
| **Framework** | Native PyTorch | Microsoft DeepSpeed |
| **ZeRO stage** | FULL_SHARD = ZeRO-3 | ZeRO 1/2/3 |
| **HYBRID_SHARD** | Yes | Yes (ZeRO-3++) |
| **CPU offload** | Yes | Yes (optimizer + params) |
| **NVMe offload** | No | Yes |
| **Overlap comm** | Limited | Advanced |
| **Ease of use** | Easier (less config) | More config, more control |
| **Performance** | Good | Often better tuned |
| **Community** | Growing | Mature |

#### Tensor Parallelism

Splits individual layers across GPUs (within a node for fast interconnects):

```
Large Matrix Multiplication:
Wx = [W₁; W₂; ...; Wₙ] x → Each GPU computes Wᵢx → AllGather to combine
```

**When Needed:** Models ≥ 70B require tensor parallelism even with ZeRO-3 because individual weight matrices exceed 80 GB (e.g., Llama 3 405B's output projection is ~6 GB; its embedding matrix is ~16 GB — manageable, but large layers need sharding).

```bash
# DeepSpeed with tensor parallelism
deepspeed --num_gpus 8 train.py \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 1
```

#### Pipeline Parallelism

Splits model layers across GPUs (each GPU processes a subset of layers):

```
GPU 0: Layers 1-10 → GPU 1: Layers 11-20 → GPU 2: Layers 21-30 → GPU 3: Layers 31-40
```

**Challenges:** Pipeline bubbles (idle time during forward/backward pass). Mitigated by micro-batching.

```python
# DeepSpeed pipeline parallelism
deepspeed --num_gpus 4 train.py \
    --pipeline-model-parallel-size 4
```

**Hybrid Strategy (3D Parallelism):**
```
Groups:          Data Parallel (across 8 nodes)
Within node:     Tensor Parallel (across 4 GPUs) + Pipeline Parallel (2 stages)
Combined:        8 × 4 × 2 = 64 GPUs
```

#### Sequence Parallelism

Splits long sequences across GPUs, enabling training on sequences longer than one GPU can fit:

```
Sequence of 128K tokens → Split into 4 × 32K → Each GPU processes 32K
                                 → Ring attention for full context
```

**When Needed:** Long-context fine-tuning (128K+ tokens), RULER evaluation, book-level training.

---

## 8. Tools

### 8.1 Axolotl

Axolotl is the most popular configuration-driven LLM fine-tuning framework. It supports almost all open-source models and PEFT methods.

**YAML Configuration:**
```yaml
base_model: meta-llama/Llama-2-7b-hf
model_type: LlamaForCausalLM
tokenizer_type: LlamaTokenizer

load_in_8bit: false
load_in_4bit: true
strict: false

datasets:
  - path: ./data/train.jsonl
    ds_type: json
    type: sharegpt
    conversation: llama-2

dataset_prepared_path: ./processed_data
val_set_size: 0.1
output_dir: ./lora-out

sequence_len: 2048
sample_packing: false

lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - v_proj
  - k_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

lora_modules_to_save:
  - embed_tokens
  - lm_head

train_on_inputs: false
group_by_length: false
bf16: auto
fp16: false
tf32: false

gradient_checkpointing: true
early_stopping_patience: 3
gradient_accumulation_steps: 4
per_device_train_batch_size: 4
per_device_eval_batch_size: 4
num_epochs: 3
optimizer: adamw_torch
lr_scheduler: cosine
learning_rate: 2e-4
warmup_ratio: 0.03

deepspeed: ./deepspeed_config.json  # Optional
fsdp: false
```

**Supported Models:**
- Llama 1/2/3/3.1
- Mistral, Mixtral
- Qwen 1/2/2.5
- Gemma 1/2
- Falcon
- Phi 1/2/3
- DeepSeek, DeepSeek-V2
- Yi, Yi-1.5
- DBRX
- Command-R
- And many more

**Dataset Formats Supported:** Alpaca, ShareGPT, ChatML, messages, raw text, JSONL, Parquet

**Training Entry:**
```bash
# Single GPU
accelerate launch -m axolotl.cli.train config.yml

# Multi-GPU
deepspeed --num_gpus 8 train.py --config config.yml
```

### 8.2 Unsloth

Unsloth optimizes memory and speed for LoRA/QLoRA training through custom CUDA kernels and optimized gradient checkpointing.

**Key Optimizations:**
- **Custom attention kernels**: 2x faster than vanilla implementation
- **Optimized gradient checkpointing**: Selective checkpointing reduces memory by 50% with minimal speed loss
- **4-bit optimizations**: Faster NF4 quantization and dequantization
- **Memory tracking**: Built-in memory usage reporting

**Memory Reductions:**

| Model | Standard QLoRA | Unsloth QLoRA | Savings |
|---|---|---|---|
| Llama 3 8B | ~16 GB | ~12 GB | 25% |
| Mistral 7B | ~14 GB | ~10 GB | 29% |
| Gemma 7B | ~14 GB | ~11 GB | 21% |
| Llama 3 70B | ~72 GB | ~56 GB | 22% |

**Usage:**
```python
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-2-7b-hf",
    max_seq_length=2048,
    dtype=torch.bfloat16,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Enable Unsloth's optimized checkpointing
    random_state=42,
    use_rslora=False,
    loftq_config=None,
)

# Standard HuggingFace trainer interface
from transformers import TrainingArguments
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=100,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
    ),
)
trainer.train()
```

### 8.3 TRL (Transformer Reinforcement Learning)

TRL provides trainers for various alignment methods.

#### SFTTrainer (Supervised Fine-Tuning)

```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        output_dir="./sft-output",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        fp16=True,
        logging_steps=10,
    ),
)
trainer.train()
```

**Packing:** TRL supports packing multiple short sequences into one training example to improve throughput:
```python
SFTTrainer(..., packing=True)
```

#### DPOTrainer (Direct Preference Optimization)

DPO eliminates the need for a separate reward model by directly optimizing the policy on preference pairs:

```python
from trl import DPOTrainer

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    train_dataset=preference_dataset,
    args=TrainingArguments(
        output_dir="./dpo-output",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-6,
        max_steps=1000,
        fp16=True,
    ),
    beta=0.1,  # KL penalty coefficient
)
dpo_trainer.train()
```

**Dataset Format (DPO):**
```json
{
  "prompt": "Explain quantum computing",
  "chosen": "Quantum computing leverages quantum mechanical phenomena...",
  "rejected": "Quantum computing is like... it's complicated."
}
```

#### PPOTrainer (Proximal Policy Optimization)

Full RLHF pipeline with a separate reward model:

```python
from trl import PPOTrainer, PPOConfig

config = PPOConfig(
    model_name=model_name,
    learning_rate=1.41e-5,
    ppo_epochs=4,
    mini_batch_size=4,
    batch_size=16,
)

ppo_trainer = PPOTrainer(
    config=config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
)

# Training loop
for epoch, batch in enumerate(ppo_trainer.dataloader):
    query_tensors = batch["input_ids"]
    
    # Generate response
    response_tensors = ppo_trainer.generate(
        query_tensors,
        return_prompt=False,
        length_sampler=output_length_sampler,
        batch_size=4,
    )
    
    # Compute reward
    rewards = reward_model(query_tensors, response_tensors)
    
    # PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
```

#### RewardTrainer

```python
from trl import RewardTrainer

reward_trainer = RewardTrainer(
    model=reward_model,
    tokenizer=tokenizer,
    train_dataset=preference_dataset,
    args=TrainingArguments(
        output_dir="./reward-output",
        per_device_train_batch_size=4,
        learning_rate=1e-5,
        num_train_epochs=3,
    ),
)
reward_trainer.train()
```

#### ORPOTrainer (Odds Ratio Preference Optimization)

An alternative to DPO that uses odds ratio instead of Bradley-Terry:

```python
from trl import ORPOTrainer

orpo_trainer = ORPOTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=preference_dataset,
    args=TrainingArguments(
        output_dir="./orpo-output",
        per_device_train_batch_size=4,
        learning_rate=1e-6,
        max_steps=1000,
    ),
    beta=0.1,
)
orpo_trainer.train()
```

#### GRPOTrainer (Group Relative Policy Optimization)

```python
from trl import GRPOTrainer

grpo_trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_function],
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=TrainingArguments(
        output_dir="./grpo-output",
        per_device_train_batch_size=4,
        learning_rate=1e-6,
        num_train_epochs=3,
    ),
)
grpo_trainer.train()
```

### 8.4 HuggingFace Ecosystem

The HuggingFace ecosystem provides the foundation for most open-source fine-tuning:

- **transformers**: Model loading, tokenization, training loop
- **datasets**: Dataset loading, preprocessing, caching
- **peft**: LoRA, QLoRA, DoRA, AdaLoRA, IA3 implementation
- **accelerate**: Device placement, mixed precision, distributed training
- **trl**: RLHF, DPO, PPO, ORPO trainers
- **evaluate**: Evaluation metrics and benchmark integration
- **hub**: Model/dataset hosting and versioning

**End-to-end Pipeline:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import torch

# 1. Load model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# 2. Prepare for PEFT
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# 3. Load data
dataset = load_dataset("json", data_files="train.jsonl")["train"]

# 4. Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        fp16=True,
    ),
)
trainer.train()

# 5. Save
trainer.save_model("./final-model")
```

### 8.5 Lit-GPT

Lightning AI's Lit-GPT provides a minimal, hackable fine-tuning implementation with full transparency:

```bash
# Download weights
python scripts/download.py --repo_id meta-llama/Llama-2-7b-hf

# Prepare data
python scripts/prepare_alpaca.py

# Fine-tune with LoRA
python finetune/lora.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --data_dir data/alpaca \
    --out_dir out/lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05
```

**Key Advantage:** Minimal abstractions — easy to modify, debug, and understand. Good for research and custom implementations.

### 8.6 LLaMA-Factory

LLaMA-Factory provides a web UI and CLI for fine-tuning with over 100 model architectures and multiple training methods:

```bash
# CLI usage
llamafactory-cli train \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --stage sft \
    --do_train \
    --dataset alpaca_zh \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --lora_rank 8 \
    --lora_alpha 32 \
    --output_dir output \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 100 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --fp16
```

**Key Features:**
- Web UI (Gradio-based) for non-technical team members
- 100+ model architectures supported
- Multiple training stages: pre-training, SFT, RLHF (DPO, PPO, KTO)
- Data merging and export
- Multi-GPU support via DeepSpeed

### 8.7 Firefly

Firefly is a Chinese-origin fine-tuning framework optimized for multi-turn conversations:

```bash
# Train
python train.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --train_file ./data/train.jsonl \
    --output_dir ./output \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --lora_rank 16 \
    --lora_alpha 32
```

**Key Strengths:**
- Multi-turn conversation support
- Efficient packing for long contexts
- Good Chinese language support
- Knowledge graph augmentation

### 8.8 Tool Comparison Matrix

| Tool | Best For | Strengths | Weaknesses |
|---|---|---|---|
| **Axolotl** | General production fine-tuning | YAML config, wide model support, PEFT + full FT | Complexity, steep learning curve |
| **Unsloth** | Memory-constrained environments | Memory optimization, speed | Limited model support, newer tool |
| **TRL** | RLHF and alignment training | Full alignment suite (DPO, PPO, ORPO, GRPO) | Not for pre-training |
| **Lit-GPT** | Research, learning, transparency | Minimal, hackable, educational | Manual updates, less automation |
| **LLaMA-Factory** | Teams needing GUI + CLI | Web UI, 100+ models, Chinese support | Newer, smaller community |
| **Firefly** | Multi-turn conversation | Conversation packing, knowledge graphs | Limited scope |
| **HuggingFace (direct)** | Maximum control and customization | Full flexibility, ecosystem integration | More code required |

---

## 9. Hyperparameter Tuning

### 9.1 Learning Rate

Learning rate is the most impactful hyperparameter in fine-tuning.

**Scheduling Strategies:**

| Schedule | Description | Best For |
|---|---|---|
| **Cosine** | LR decreases following cosine curve from max to min | General purpose, smooth convergence |
| **Linear** | Linear decay to 0 | Short training runs, simple tasks |
| **Constant** | Fixed LR throughout training | Very short runs, minimal adaptation |
| **Constant with warmup** | Linear warmup then constant | Stable training with aggressive LR |
| **Cosine with restarts** | Cyclical cosine annealing | Escaping local minima, multi-task |
| **Inverse square root** | LR ∝ 1/√t | Very long training runs |

**Recommended Defaults:**

| Setting | Value |
|---|---|
| Learning rate (LoRA, 7B) | 2e-4 |
| Learning rate (LoRA, 13B) | 1e-4 |
| Learning rate (LoRA, 70B) | 5e-5 |
| Learning rate (Full FT, 7B) | 1e-5 |
| Learning rate (Full FT, 13B) | 5e-6 |
| Learning rate (Full FT, 70B) | 2e-6 |
| Warmup ratio | 0.03 (3% of total steps) |
| Schedule | Cosine |

**LR Range Guidelines:**
```
LoRA:    1e-5 to 5e-4 (sweet spot: 1e-4 to 2e-4)
Full FT: 5e-6 to 5e-5 (sweet spot: 1e-5)
DPO:     1e-7 to 1e-6 (sweet spot: 5e-7)
```

### 9.2 Batch Size

**Terminology:**

- **Per-device batch size**: Samples per GPU per step
- **Gradient accumulation steps**: Accumulate gradients over N steps before applying update
- **Effective batch size**: Per-device × GPUs × gradient accumulation steps

**Calculations:**
```
Effective batch size = per_device_train_batch_size × num_gpus × gradient_accumulation_steps
```

**Example:**
```yaml
per_device_train_batch_size: 4
gradient_accumulation_steps: 8
num_gpus: 8
# Effective: 4 × 8 × 8 = 256 samples per update
```

**Batch Size Guidelines:**

| Model Size | Min Batch | Recommended | Max Batch |
|---|---|---|---|
| 7B | 32 | 128 | 1024 |
| 13B | 16 | 64 | 512 |
| 34B | 8 | 32 | 256 |
| 70B | 4 | 16 | 128 |
| 180B | 2 | 8 | 64 |

**Effects of Batch Size:**
- **Too small**: Noisy gradients, unstable training, poor generalization
- **Too large**: Smooth gradients, but wall-clock time per epoch increases; may converge to sharp minima

**Scaling Laws (Batch Size):**
- Double batch size → halve learning rate for similar training dynamics
- Maximum useful batch size ≈ 0.1 × dataset size (for most tasks)

### 9.3 Epoch Count vs Tokens Seen

The number of training epochs depends on dataset size and the amount of repetition desired.

| Dataset Size | Typical Epochs | Total Tokens Seen (7B model, 100M dataset) |
|---|---|---|
| 1K examples | 10–20 | 200M–400M tokens |
| 10K examples | 5–10 | 500M–1B tokens |
| 100K examples | 2–5 | 2B–5B tokens |
| 1M+ examples | 1–3 | 10B–30B tokens |

**Rule of Thumb:** Train until validation loss plateaus. For LoRA, this is typically 1–5 epochs. For full fine-tuning, 1–3 epochs is usually sufficient and more can cause overfitting.

**Token Budget Monitoring:**
```python
total_tokens = 0
for epoch in range(num_epochs):
    for batch in dataloader:
        total_tokens += batch['input_ids'].numel()
        
# Monitor tokens seen vs model size
tokens_per_param_ratio = total_tokens / model_params
# Good: 20–200 tokens per parameter (LoRA)
# Good: 100–500 tokens per parameter (full FT)
```

### 9.4 LoRA Rank

See Section 3.1 for detailed rank recommendations.

**Tuning Approach:**
1. Start with r=8 or r=16
2. Monitor validation loss — if underfitting, double the rank
3. If overfitting (train loss << val loss), halve the rank or increase dropout
4. For very large datasets (100K+), consider r=32 or r=64

**Rank Scaling:**
```
Compute budget ≈ O(r × (d + k)) where d, k are weight matrix dimensions
Doubling rank → ~2x parameter count, ~2x compute, ~1.1–1.3x quality
```

### 9.5 LoRA Alpha Scaling

Alpha controls the magnitude of the adapter update:

- **α = r** (default): Unit scaling
- **α = 2r**: Stronger adapter influence
- **α = r/2**: Weaker adapter influence

**Tuning Alpha:**
- If model is not adapting enough (similar to base): increase α/r ratio
- If training is unstable: decrease α/r ratio
- If catastrophic forgetting: decrease α/r ratio
- Common range: α = 8 to 128, r = 4 to 64

### 9.6 Target Modules Selection

See Section 3.1 for target module recommendations.

**Module Selection Strategy:**
1. Start with `q_proj, v_proj`
2. Evaluate: if quality is insufficient, add `k_proj, o_proj`
3. If still insufficient, add MLP: `gate_proj, up_proj, down_proj`
4. Monitor: each doubling of modules roughly doubles parameter count

**Module Impact by Architecture:**

| Architecture | Most Impactful Modules | Additional Modules |
|---|---|---|
| Llama/Mistral | q_proj, v_proj, gate_proj | o_proj, k_proj, up_proj, down_proj |
| GPT-NeoX | query, value, dense_h_to_4h | attention.output, dense_4h_to_h |
| Falcon | query, value, dense | attention_output, dense_4h_to_h |
| CodeGen | qkv_proj, out_proj | fc_in, fc_out |
| BLOOM | query_key_value, dense | dense_h_to_4h, dense_4h_to_h |

### 9.7 Weight Decay

Weight decay acts as L2 regularization on the adapter weights.

| Setting | Effect |
|---|---|
| 0.0 | No regularization (fast training, risky for small data) |
| 0.01 | Light regularization (default for most tasks) |
| 0.1 | Strong regularization (small datasets, overfitting) |
| 1.0 | Very strong (rarely used) |

**Recommendation:** 0.01 for LoRA fine-tuning with 1K+ examples. Increase to 0.1 for <500 examples.

### 9.8 AdamW Betas and Epsilon

**Default AdamW Parameters:**
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-4,
    betas=(0.9, 0.999),  # Default: (0.9, 0.999)
    eps=1e-8,             # Default: 1e-8
    weight_decay=0.01,
)
```

**When to Tune Betas:**
- **Spiky loss**: Increase β₂ to 0.9999 (smoother but slower adaptation)
- **Slow convergence**: Decrease β₁ to 0.85 (more weight on recent gradients)
- **Small batch training**: Consider β₁=0.95, β₂=0.999 (more stable)

**When to Tune Epsilon:**
- **Training instability**: Increase ε to 1e-7 or 1e-6
- **Mixed precision issues**: ε = 1e-4 or 1e-5 for fp16 training
- **Convergence plateaus**: ε = 1e-8 is usually fine

### 9.9 LoRA Dropout

See Section 3.1 for dropout recommendations.

**Tuning Dropout Together with Rank:**
```
Low rank + low dropout:     Under-parameterized, might underfit
Low rank + high dropout:    Too constrained, won't learn
High rank + low dropout:    Overfit risk
High rank + high dropout:   Good for small data
```

---

## 10. Training Stability

### 10.1 Loss Spikes

Loss spikes during training can derail fine-tuning. Common causes and solutions:

| Cause | Symptom | Fix |
|---|---|---|
| **Data quality issue** | Isolated spike in one batch | Detect and remove bad data point |
| **Learning rate too high** | Gradual rise after initial drop | Reduce LR, increase warmup |
| **Batch with outlier length** | Spike on long sequences | Gradient clipping, sort by length |
| **Numerical instability** | Loss → NaN or Inf | Use bf16 instead of fp16, increase epsilon |
| **Optimizer state corruption** | Spikes at resume | Clear optimizer state on resume |
| **Data ordering bias** | Periodic spikes | Shuffle data before each epoch |
| **LoRA rank too high** | Oscillating loss | Reduce rank or increase dropout |

**Gradient Clipping:**
```python
# In TrainingArguments
args = TrainingArguments(
    max_grad_norm=1.0,  # Default: clip gradients to max L2 norm of 1.0
)
```

**Detection and Recovery:**
```python
def detect_loss_spike(loss_history, threshold=3.0, window=10):
    if len(loss_history) < window * 2:
        return False
    
    recent = loss_history[-window:]
    prior = loss_history[-window*2:-window]
    
    recent_mean = np.mean(recent)
    prior_mean = np.mean(prior)
    prior_std = np.std(prior)
    
    # Flag if recent mean > prior mean + threshold * prior std
    if recent_mean > prior_mean + threshold * prior_std:
        return True
    return False
```

**When Spikes Occur:**
1. **Pause training** (if automated)
2. **Examine the last batch** — bad data point?
3. **Reduce LR** by 0.5x or 0.3x and restart from previous checkpoint
4. **If persistent**: Switch to bf16, increase gradient accumulation, reduce batch size

### 10.2 Loss Curves Interpretation

**Ideal Loss Curve:**
```
Loss
│  \          
│   \         
│    \___     
│        \    
│         \___
│             \__
└─────────────────→ Steps
```

**Common Patterns:**

**Pattern 1: Healthy training**
- Loss drops smoothly, no spikes
- Train and val loss converge
- Val loss eventually plateaus

**Pattern 2: Overfitting**
```
Train: \___ (low)
Val:   \___/ (higher, diverging)
```
Fix: Reduce epochs, increase dropout, add weight decay, reduce rank

**Pattern 3: Underfitting**
```
Train: \__________ (still high, still decreasing)
Val:   \__________ (plateau)
```
Fix: Increase epochs, increase LR, increase rank, reduce dropout

**Pattern 4: Learning rate too high**
```
Train: \/\/\/\/\/ (oscillating)
```
Fix: Reduce LR, increase warmup

**Pattern 5: Data quality issues**
```
Train: \___/\___/\ (periodic spikes)
```
Fix: Remove outlier data, shuffle, gradient clipping

**Pattern 6: Catastrophic forgetting**
```
Train: \__  (low on new task)
Benchmark: /‾‾ (dropping on original task)
```
Fix: Reduce LR, use replay data, reduce epochs

### 10.3 Evaluation During Training

Regular evaluation prevents wasted training and enables early stopping.

**Evaluation Types:**

| Type | Frequency | Metric | Cost |
|---|---|---|---|
| **Per-step loss** | Every step | Cross-entropy loss | None |
| **Perplexity** | Every N steps (e.g., 100) | exp(loss) on eval set | ~1 minute |
| **Downstream eval** | Every N steps (e.g., 500) | Task-specific (F1, EM, etc.) | 5–30 minutes |
| **Generative eval** | Every N epochs | Human eval, LLM-as-judge | 10–60 minutes |

**Implementation:**
```python
from transformers import TrainerCallback

class EvalCallback(TrainerCallback):
    def __init__(self, eval_dataset, tokenizer, every_n_steps=100):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.every_n_steps = every_n_steps
        
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every_n_steps == 0:
            # Compute perplexity
            model = kwargs['model']
            eval_loader = DataLoader(self.eval_dataset, batch_size=4)
            
            total_loss = 0
            for batch in eval_loader:
                outputs = model(**batch)
                total_loss += outputs.loss.item()
            
            avg_loss = total_loss / len(eval_loader)
            perplexity = np.exp(avg_loss)
            
            self.log({"eval/perplexity": perplexity, "step": state.global_step})
```

### 10.4 Checkpointing Strategies

**Checkpoint Frequency:**

| Strategy | Frequency | Storage Cost | Recovery |
|---|---|---|---|
| **By step** | Every N steps (e.g., 100) | N × model_size/hour | Fine-grained |
| **By epoch** | Every epoch | epoch_count × model_size | Coarse |
| **Best N** | When val loss improves | N × model_size | Best quality |
| **Time-based** | Every N hours | N × model_size/hour | Time-budget aligned |

**Recommended Enterprise Strategy:**
```
Save every 500 steps (by step)
Keep last 3 checkpoints (disk space management)
Save best checkpoint (by validation loss)
```

```python
args = TrainingArguments(
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,  # Keep only 3 most recent
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)
```

**Checkpoint Structure:**
```
checkpoint-500/
├── adapter_config.json    # LoRA config
├── adapter_model.safetensors  # LoRA weights
├── training_args.bin      # Training arguments snapshot
├── optimizer.pt           # Optimizer state (for resume)
├── scheduler.pt           # LR scheduler state
├── trainer_state.json     # Step, epoch, best metric
├── rng_state.pth          # Random state for reproducibility
└── scaling.pt             # QLoRA scaling factors (if applicable)
```

### 10.5 Resuming from Checkpoint

```python
# Resume training
trainer.train(resume_from_checkpoint=True)
# Or from specific checkpoint:
trainer.train(resume_from_checkpoint="./checkpoint-500")

# The trainer will:
# 1. Load model weights
# 2. Load optimizer and scheduler state
# 3. Skip already-processed data
# 4. Continue from save step
```

**Issues When Resuming:**
- **Dataset shuffle**: Will shuffle differently unless you set a seed for each run
- **Data loading**: If dataset changed, resume may skip wrong indices
- **Optimizer state**: Load optimizer.pt to avoid starting from fresh momentum
- **LR scheduler**: LR will jump if not properly restored
- **Mixed precision**: Loss scaler state must be restored for fp16

---

## 11. Evaluation

### 11.1 Holdout Sets Creation

**Temporal Split (Recommended for Enterprise):**
```python
# Sort by timestamp, use oldest 80% for train, newest 10% for val, newest 10% for test
dataset_sorted = sorted(dataset, key=lambda x: x['timestamp'])
n = len(dataset_sorted)
train = dataset_sorted[:int(n*0.8)]
val = dataset_sorted[int(n*0.8):int(n*0.9)]
test = dataset_sorted[int(n*0.9):]
```

**Stratified Split:**
```python
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in sss.split(X, strata):
    train = [dataset[i] for i in train_idx]
    test = [dataset[i] for i in test_idx]
```

**Cross-Validation (for small datasets):**
```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    train_data = [dataset[i] for i in train_idx]
    val_data = [dataset[i] for i in val_idx]
    
    # Train on fold
    metrics = train_and_evaluate(train_data, val_data)
    fold_metrics.append(metrics)

mean_metrics = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
```

### 11.2 Domain-Specific Metrics

**General Metrics:**
- Perplexity
- Accuracy
- F1 Score
- BLEU (generation)
- ROUGE (summarization)
- METEOR
- BERTScore

**Legal Domain:**
- Citation accuracy (% of citations correctly formatted)
- Statute relevance (precision/recall of cited statutes)
- Legal reasoning coherence (expert rating, 1–5)
- Hallucination rate of legal facts

**Medical Domain:**
- ICD-10 code assignment accuracy
- Drug name accuracy (% correctly named)
- Clinical note completeness (expert checklist)
- Safety flag rate (incorrect medical advice)
- HIPAA compliance (PII presence in generated text)

**Finance Domain:**
- Numerical calculation accuracy (% correct)
- Financial ratio calculation error
- Sentiment alignment with ground truth (finance-specific sentiment)
- SEC filing format compliance

**Code Domain:**
- Pass@k (functional correctness)
- Syntax error rate
- Test coverage of generated code
- Code style compliance (PEP8, etc.)
- Security vulnerability rate (SAST scan)

**Customer Support:**
- CSAT score prediction (agreement with actual CSAT)
- First Contact Resolution accuracy
- Intent classification accuracy
- Escalation prediction precision/recall
- Brand voice compliance (expert rating)

### 11.3 Human Evaluation Protocol

**Protocol Design:**

1. **Sample selection**: Random sample of 100–500 generations per model variant
2. **Blinding**: Remove model identifiers, shuffle outputs
3. **Evaluators**: 3+ domain experts per sample
4. **Criteria**: Task-specific rubric (1–5 scale per criterion)
5. **Calibration**: Gold-standard examples with known scores
6. **Quality control**: Inter-annotator agreement monitoring

**Sample Rubric (Legal Domain):**
```
Criteria                   1 (Poor)    3 (Good)    5 (Excellent)
Accuracy of legal facts    >3 errors   0-1 errors  Perfect
Citation format            None cited  Partial     Complete & correct
Reasoning quality          Illogical   Sound       Compelling
Helpfulness               Unhelpful   Somewhat    Highly actionable
Safety (no hallucination)  Hallucinates Minor issues  No issues
```

**Typical Enterprise Human Eval Cost:**
```
100 samples × 3 evaluators × 5 minutes each × $50/hr = ~$125
1,000 samples × 3 evaluators × 5 min × $50/hr = ~$1,250
```

### 11.4 A/B Testing

**Online A/B Testing for LLM Deployment:**

1. **Traffic split**: 50% control (baseline), 50% treatment (fine-tuned model)
2. **Metric collection**: User engagement, task completion, latency, feedback ratings
3. **Statistical significance**: Chi-squared test, t-test, or Bayesian A/B test
4. **Minimum sample size**: 
```
n = (Z_α/2 + Z_β)² × (p₁(1-p₁) + p₂(1-p₂)) / (p₂ - p₁)²
Example: 5% improvement, 80% power, 95% confidence → ~3,000 samples per variant
```

**Implementation:**
```python
from scipy import stats

def ab_test(control_outcomes, treatment_outcomes):
    """Two-proportion z-test"""
    n_control = len(control_outcomes)
    n_treatment = len(treatment_outcomes)
    
    p_control = sum(control_outcomes) / n_control
    p_treatment = sum(treatment_outcomes) / n_treatment
    
    p_pool = (sum(control_outcomes) + sum(treatment_outcomes)) / (n_control + n_treatment)
    
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_control + 1/n_treatment))
    z = (p_treatment - p_control) / se
    
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return {
        'control_rate': p_control,
        'treatment_rate': p_treatment,
        'lift': (p_treatment - p_control) / p_control,
        'p_value': p_value,
        'significant': p_value < 0.05,
    }
```

### 11.5 Regression Testing

Maintain a regression test suite to catch quality regressions when updating models:

```python
REGRESSION_TESTS = [
    {
        "name": "citation_format",
        "input": "What is the holding in Miranda v. Arizona?",
        "expected_pattern": r"\d+ U\.S\. \d+",  # Must cite US Reports
        "metric": "regex_match",
    },
    {
        "name": "no_pii_leak",
        "input": "Summarize this patient: John Doe, SSN 123-45-6789",
        "expected_pattern": r"(?i)(?:\[REDACTED\]|<PERSON>|<SSN>)",
        "metric": "regex_match",
    },
    {
        "name": "code_compiles",
        "input": "Write a Python function to sort a list",
        "metric": "code_compiles",
    },
]
```

**Regression Suite Pipeline:**
1. Run suite against baseline model → store results
2. Run suite against candidate model → compare
3. Flag any metric that degrades beyond tolerance (e.g., >5% drop)
4. Block deployment if critical tests fail

---

## 12. Deployment

### 12.1 Merging LoRA Adapters into Base Model

For production deployment, merging LoRA weights into the base model eliminates adapter loading overhead:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Load LoRA adapter
lora_model = PeftModel.from_pretrained(base_model, "./lora-checkpoint-500")

# Merge weights into base model
merged_model = lora_model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("./merged-model")
tokenizer.save_pretrained("./merged-model")
```

**Storage Comparison:**

| Format | 7B Base | 7B + LoRA (r=16) | 7B Merged |
|---|---|---|---|
| FP16 | 14 GB | 14 GB + 16 MB | 14 GB |
| 4-bit | 4 GB | 4 GB + 16 MB | 4 GB |

**When to Merge vs Keep Separate:**

| Scenario | Merge | Keep Separate |
|---|---|---|
| Single adapter deployment | ✓ | |
| Multiple adapter routing | | ✓ |
| Maximum inference throughput | ✓ | |
| Quick adapter switching | | ✓ |
| GPU memory constrained | ✓ (one forward pass) | |
| Hot-swapping | | ✓ |

### 12.2 Serving Multiple Adapters

#### vLLM LoRA Adapter Routing

vLLM supports serving multiple LoRA adapters simultaneously with dynamic routing:

```python
from vllm import LLM, SamplingParams

# Initialize with base model + max LoRA config
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_lora=True,
    max_loras=8,           # Max concurrent adapters
    max_lora_rank=64,      # Max rank among all adapters
    # LoRA modules used across all adapters
    lora_extra_vocab_size=256,
)

# Load adapters
llm.load_lora(
    "legal-v1",
    lora_path="./adapters/legal-v1",
)
llm.load_lora(
    "medical-v1",
    lora_path="./adapters/medical-v1",
)

# Route requests to specific adapter
prompts = [
    ("What is the holding in Marbury v. Madison?", "legal-v1"),
    ("Diagnose based on these symptoms...", "medical-v1"),
]

sampling_params = SamplingParams(temperature=0.1, max_tokens=512)

# vLLM automatically routes each prompt to its specified adapter
outputs = llm.generate(
    [p[0] for p in prompts],
    sampling_params,
    lora_request=[p[1] for p in prompts],  # Per-prompt adapter routing
)
```

**vLLM Adapter Hot-Swapping:**
```bash
# Start vLLM server with LoRA support
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --enable-lora \
    --max-loras 8 \
    --max-lora-rank 64

# Hot-swap: update an adapter without restarting
curl http://localhost:8000/v1/load_lora_adapter \
    -H "Content-Type: application/json" \
    -d '{"lora_name": "legal-v2", "lora_path": "./adapters/legal-v2"}'

# Then route traffic to new adapter
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "legal-v2",
        "messages": [{"role": "user", "content": "Explain contract law"}]
    }'
```

#### Tensor Parallelism with LoRA

When using tensor parallelism (model parallelism across GPUs), LoRA adapters must be sharded consistently with the base model:

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,  # 4 GPUs
    enable_lora=True,
    max_loras=4,
)
```

**Key Considerations for TP + LoRA:**
- LoRA weights are sharded across GPUs along with base weights
- Each GPU holds 1/TP_size of each adapter
- Adapter loading/unloading must happen on all GPUs

### 12.3 Adapter Versioning

**Versioning Schema:**
```
adapters/
├── legal-v1.0.0/  # Major.Minor.Patch
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── legal-v1.1.0/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── legal-v2.0.0/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
└── manifest.json
```

**Manifest Example:**
```json
{
  "adapters": [
    {
      "name": "legal-v1.0.0",
      "base_model": "meta-llama/Llama-2-7b-hf",
      "created": "2024-06-01T10:00:00Z",
      "dataset": "legal-corpus-v1",
      "metrics": {"f1": 0.92, "exact_match": 0.87},
      "status": "deprecated",
      "checksum": "sha256:abc123..."
    },
    {
      "name": "legal-v1.1.0",
      "base_model": "meta-llama/Llama-2-7b-hf",
      "created": "2024-06-15T14:30:00Z",
      "dataset": "legal-corpus-v2",
      "metrics": {"f1": 0.94, "exact_match": 0.89},
      "status": "active",
      "checksum": "sha256:def456..."
    }
  ]
}
```

### 12.4 Model Registry

**Enterprise Model Registry (using MLflow example):**

```python
import mlflow

mlflow.set_tracking_uri("http://mlflow-server:5000")

with mlflow.start_run(run_name="llama-7b-legal-v2"):
    # Log parameters
    mlflow.log_params({
        "base_model": "meta-llama/Llama-2-7b-hf",
        "lora_r": 16,
        "lora_alpha": 32,
        "learning_rate": 2e-4,
        "batch_size": 128,
        "dataset": "legal-corpus-v2",
        "num_epochs": 3,
    })
    
    # Log metrics
    mlflow.log_metrics({
        "eval_loss": 0.85,
        "eval_accuracy": 0.94,
        "eval_f1": 0.92,
    })
    
    # Log model
    mlflow.transformers.log_model(
        transformers_model={"model": merged_model, "tokenizer": tokenizer},
        artifact_path="model",
        registered_model_name="llama-7b-legal",
    )

# Transition model stage
from mlflow.tracking import MlflowClient
client = MlflowClient()

client.transition_model_version_stage(
    name="llama-7b-legal",
    version=2,
    stage="Staging",  # Options: None, Staging, Production, Archived
)
```

### 12.5 Canary Deployments

**Canary Deployment Strategy for LLM Adapters:**

```
Phase 1: Internal (5% traffic, dev team only)
   ↓ (monitor for 24h)
Phase 2: Shadow (100% traffic mirrored, no user-facing)
   ↓ (compare outputs)
Phase 3: Canary (10% traffic, real users)
   ↓ (monitor metrics)
Phase 4: Ramped (25% → 50% → 75% → 100%)
   ↓ (checkpoint at each stage)
Phase 5: Full rollout (100% traffic)
```

**Automated Rollback Conditions:**
- Latency p99 increases by >20%
- Error rate exceeds 1%
- User satisfaction drops by >5%
- Safety violations detected
- Any regression test fails

### 12.6 Rollback

```python
# MLflow rollback: switch to previous version
client.transition_model_version_stage(
    name="llama-7b-legal",
    version=1,  # Previous version
    stage="Production",
)

# Kubernetes rollback
kubectl rollout undo deployment/llm-server

# Adapter swap (vLLM)
curl http://localhost:8000/v1/load_lora_adapter \
    -d '{"lora_name": "legal-v1", "lora_path": "./adapters/legal-v1.0.0"}'
```

---

## 13. Model Formats

### 13.1 HuggingFace Format (safetensors)

The standard format for open-source LLMs and fine-tuned adapters.

**Structure:**
```
model_directory/
├── config.json              # Model configuration
├── tokenizer_config.json    # Tokenizer configuration
├── tokenizer.model          # Tokenizer model file (SentencePiece)
├── tokenizer.json           # Tokenizer JSON (if applicable)
├── model.safetensors        # Model weights (or model-00001-of-00002.safetensors)
├── model.safetensors.index.json  # Shard index (for sharded models)
├── generation_config.json   # Default generation parameters
├── special_tokens_map.json  # Special token mappings
└── adapter_config.json      # (PEFT) LoRA configuration
└── adapter_model.safetensors  # (PEFT) LoRA weights
```

**Advantages:**
- Universal support across all major libraries
- Metadata (config) includes architecture details
- Sharded format for large models
- Safe (no pickle) with safetensors

### 13.2 GGUF (llama.cpp)

GGUF is the format for CPU-efficient inference with llama.cpp and its bindings (llama-cpp-python, Ollama, LM Studio).

**Advantages:**
- CPU-optimized inference (no GPU needed)
- Multiple quantization levels (Q2_K, Q3_K, Q4_K_M, Q5_K_M, Q6_K, Q8_0, F16)
- 4-bit quantized models run on 8 GB RAM
- Portable (single file)
- Growing ecosystem (Ollama, LM Studio, GPT4All)

**Quantization Types:**

| Type | Bits/Weight | Quality Relative to FP16 | Size (7B) |
|---|---|---|---|
| Q2_K | 2.56 | ~55% | 2.7 GB |
| Q3_K_S | 3.35 | ~70% | 3.3 GB |
| Q3_K_M | 3.55 | ~75% | 3.5 GB |
| Q4_0 | 4.10 | ~85% | 3.9 GB |
| Q4_K_M | 4.35 | ~90% | 4.1 GB |
| Q5_K_M | 5.15 | ~95% | 4.7 GB |
| Q6_K | 6.15 | ~98% | 5.4 GB |
| Q8_0 | 8.25 | ~99% | 6.7 GB |
| F16 | 16.0 | 100% | 13.5 GB |

**Conversion from HuggingFace:**
```bash
# Convert HF model to GGUF
python convert.py ./llama-model --outfile llama-model.gguf

# Quantize
./quantize llama-model.gguf llama-model-Q4_K_M.gguf Q4_K_M
```

### 13.3 AWQ (Activation-Aware Weight Quantization)

AWQ is a quantization method that preserves important weights based on activation patterns.

**Advantages:**
- Better quality than GPTQ at same bit width
- Faster inference than GPTQ in most implementations
- Supported by vLLM, TGI, AutoAWQ, ExLlama

**Generation:**
```python
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Quantize to 4-bit
model.quantize(
    tokenizer,
    quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"},
)

# Save
model.save_quantized("./llama-awq")
tokenizer.save_pretrained("./llama-awq")
```

**Comparison:**
| Quantization | Bits | Quality (vs FP16) | Inference Speed |
|---|---|---|---|
| AWQ 4-bit | 4 | ~99.5% | Very Fast |
| AWQ 3-bit | 3 | ~97% | Fast |

### 13.4 GPTQ (GPT Post-Training Quantization)

GPTQ is a one-shot weight quantization method using the Hessian matrix.

**Advantages:**
- Mature ecosystem (AutoGPTQ, ExLlama, ExLlamaV2)
- Good quality at 4-bit
- Well-supported in inference engines

**Generation:**
```python
from auto_gptq import AutoGPTQForCausalLM

model = AutoGPTQForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantize_config=BaseQuantizeConfig(
        bits=4,
        group_size=128,
        damp_percent=0.01,
        desc_act=False,  # Set to True for better quality (slower)
    ),
)
model.quantize(tokenizer)
model.save_quantized("./llama-gptq")
```

**GPTQ vs AWQ:**
- GPTQ: Better quality at 4-bit group size 128, slightly slower inference
- AWQ: Better quality at 4-bit group size 128, faster inference in most engines

### 13.5 MLX (Apple Silicon)

MLX is Apple's ML framework for Apple Silicon (M-series chips).

**Advantages:**
- Native Apple Silicon optimization
- Unified memory (CPU + GPU shared memory)
- Efficient for on-device deployment
- Good for Mac-based development

**Format:**
```python
import mlx.core as mx
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

response = generate(model, tokenizer, prompt="Hello!", max_tokens=100)
print(response)
```

**Conversion:**
```bash
# Convert HF model to MLX format
python -m mlx_lm.convert \
    --hf-path meta-llama/Llama-2-7b-hf \
    --mlx-path ./llama-mlx-4bit \
    -q \
    --q_bits 4
```

### 13.6 ONNX (Open Neural Network Exchange)

ONNX enables cross-platform deployment and hardware-specific optimization.

**Advantages:**
- Cross-platform (can run on CPU, GPU, NPU, etc.)
- Optimization via ONNX Runtime
- Quantization (dynamic, static, QAT)
- Integration with Azure and Windows ML

**Conversion:**
```python
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

model = ORTModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    export=True,
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model.save_pretrained("./llama-onnx")
```

**When to Use:**
- Windows deployment (DirectML)
- Azure cloud deployment
- Edge devices / IoT
- Cross-platform mobile deployment

### 13.7 Format Decision Matrix

| Criterion | HuggingFace | GGUF | AWQ | GPTQ | MLX | ONNX |
|---|---|---|---|---|---|---|
| GPU inference | ✓ Best | ✓ Good | ✓ Best | ✓ Best | ✓ Apple | ✓ Good |
| CPU inference | ✗ Slow | ✓ Best | ✗ Slow | ✗ Slow | ✓ Good | ✓ Good |
| Mobile/Edge | ✗ | ✓ | ✗ | ✗ | ✓ Apple | ✓ Best |
| Cloud deployment | ✓ Best | ✓ | ✓ Best | ✓ Best | ✗ | ✓ Best |
| Quantization | ✓ FP16/8 | ✓ 2-8 bit | ✓ 4 bit | ✓ 4 bit | ✓ 4 bit | ✓ Dynamic |
| Community support | ✓ Best | ✓ Large | ✓ Large | ✓ Large | ✓ Growing | ✓ Large |
| Tooling | ✓ Best | ✓ Good | ✓ Good | ✓ Good | ✓ Good | ✓ Good |
| Windows | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ Best |
| Fine-tuning | ✓ Best | ✗ | ✗ | ✗ | ✓ (new) | ✗ |
| Single file | ✗ | ✓ Yes | ✗ | ✗ | ✓ Yes | ✗ |

---

## Appendices

### A. Fine-Tuning Checklist

**Before you start:**
- [ ] Decision matrix completed (fine-tuning vs RAG vs prompting)
- [ ] Base model selected and tested zero-shot
- [ ] Dataset collected and audited for quality
- [ ] PII redaction pipeline verified
- [ ] Train/val/test split created
- [ ] Format converted to framework-native format
- [ ] GPU budget confirmed and allocated
- [ ] Checkpoint storage configured

**During training:**
- [ ] Loss monitored for spikes (first 100 steps)
- [ ] Validation loss tracked
- [ ] Checkpoints saved regularly
- [ ] Learning rate warmup verified
- [ ] Gradient clipping enabled

**After training:**
- [ ] Holdout evaluation complete
- [ ] Human evaluation (if applicable)
- [ ] A/B test results analyzed
- [ ] Regression tests passed
- [ ] Adapter versioned and registered
- [ ] Canary deployment plan prepared
- [ ] Rollback plan tested

### B. Common Error Messages and Solutions

| Error | Cause | Solution |
|---|---|---|
| `CUDA out of memory` | Batch too large | Reduce batch size, enable gradient checkpointing, reduce sequence length |
| `RuntimeError: expected scalar type Half but found Float` | Mixed precision mismatch | Set `torch_dtype=torch.float16` or use `bf16` |
| `KeyError: 'llama'` | Unsupported model name | Update transformers library |
| `ValueError: Tokenizer class X doesn't exist` | Tokenizer not downloaded | Use `AutoTokenizer.from_pretrained()` with explicit trust |
| `ConnectionError: Can't load tokenizer` | Network issue or insufficient access | Use local files or set `HF_TOKEN` env variable |
| `Transformer-based language models only` | QLoRA on unsupported model | Use `device_map="auto"` or specify quantization config |
| `AttributeError: 'NoneType' object has no attribute 'shape'` | Missing gradient checkpointing | Enable gradient checkpointing |

### C. Performance Benchmarks

| Model | Method | GPUs | Time (1 epoch, 10K examples) | VRAM/GPU |
|---|---|---|---|---|
| Llama 3 8B | LoRA r=16 | 1× A100 80GB | ~15 min | ~18 GB |
| Llama 3 8B | QLoRA r=16 | 1× RTX 4090 | ~20 min | ~10 GB |
| Llama 3 70B | QLoRA r=16 | 1× A100 80GB | ~45 min | ~65 GB |
| Llama 3 70B | LoRA r=16 | 4× A100 80GB | ~12 min | ~72 GB (each) |
| Llama 3 70B | QLoRA r=16 | 2× RTX 4090 | ~35 min | ~22 GB (each) |
| Mistral 7B | LoRA r=16 | 1× RTX 4090 | ~10 min | ~14 GB |
| Falcon 180B | QLoRA r=8 | 8× A100 80GB | ~90 min | ~68 GB (each) |

### D. Cost Estimation Calculator

```python
def estimate_training_cost(
    model_size_billions: float,
    dataset_size: int,
    avg_seq_len: int,
    epochs: int,
    gpu_type: str,
    num_gpus: int,
    lora_rank: int = 16,
    use_qlora: bool = False,
):
    # Parameters per token (approx)
    params = model_size_billions * 1e9
    
    # FLOPs per token (rough approximation)
    flops_per_token = 6 * params  # 6 * params for forward+backward
    
    # Total tokens processed
    total_tokens = dataset_size * avg_seq_len * epochs
    
    # GPU throughput (Tokens/second, approximate)
    gpu_throughput = {
        "A100-80GB": 1.5e5,  # 150K tokens/s for 7B (varies widely)
        "H100-80GB": 2.5e5,
        "RTX4090": 0.8e5,
        "A10G": 0.6e5,
    }
    
    throughput = gpu_throughput.get(gpu_type, 1e5)
    
    # Adjust for QLoRA (slower due to quantization)
    if use_qlora:
        throughput *= 0.8
    
    # Adjust for model size (larger models = lower throughput per GPU)
    throughput *= (7 / model_size_billions) ** 0.5
    
    # Total GPU-hours
    seconds = total_tokens / (throughput * num_gpus)
    gpu_hours = seconds / 3600
    
    # Cost
    gpu_price = {
        "A100-80GB": 3.0,  # $/GPU-hour (approximate on-demand)
        "H100-80GB": 5.0,
        "RTX4090": 0.5,
        "A10G": 1.0,
    }
    
    total_cost = gpu_hours * num_gpus * gpu_price.get(gpu_type, 2.0)
    
    return {
        "gpu_hours": round(gpu_hours, 1),
        "total_cost": round(total_cost, 2),
        "tokens_processed": total_tokens,
        "throughput_per_gpu": round(throughput),
    }

# Example
print(estimate_training_cost(
    model_size_billions=7,
    dataset_size=10000,
    avg_seq_len=2048,
    epochs=3,
    gpu_type="A100-80GB",
    num_gpus=1,
    lora_rank=16,
    use_qlora=False,
))
```

---

> **Last Updated:** May 2026
> **Author:** AI Enterprise Knowledge Base
> **License:** Internal use — Enterprise Knowledge Management
