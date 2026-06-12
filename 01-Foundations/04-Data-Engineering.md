# Data Engineering for AI: Building the Foundation

## Table of Contents
1. [Introduction](#1-introduction)
2. [Data Collection Strategies](#2-data-collection)
3. [Data Quality and Filtering](#3-data-quality)
4. [Data Deduplication](#4-deduplication)
5. [Data Decontamination](#5-decontamination)
6. [Data Mixing and Curriculum](#6-data-mixing)
7. [Synthetic Data Generation](#7-synthetic-data)
8. [Data Annotation and Labeling](#8-annotation)
9. [Data Governance and Compliance](#9-governance)
10. [Data Infrastructure and Pipelines](#10-infrastructure)
11. [Data for Fine-Tuning](#11-fine-tuning-data)
12. [Data for RLHF and Alignment](#12-rlhf-data)
13. [Data for Evaluation](#13-evaluation-data)
14. [Data Curation at Scale: Case Studies](#14-case-studies)
15. [Cloud-Native Data Engineering and Cost Analysis](#15-cloud-native-data-engineering-and-cost-analysis)
15a. [Data Engineering for Multimodal Models](#15a-data-engineering-for-multimodal-models)
16. [Cross-References](#16-cross-references)

---

## 1. Introduction

Data is the foundation of every AI system. The quality, quantity, and diversity of training data determine model capabilities more than architecture or compute. A poorly curated dataset cannot be compensated by a larger model, but a well-curated dataset can make smaller models competitive with much larger ones.

**The Data Trilemma:** Every dataset faces three competing constraints:
1. **Scale** (enough data for generalization)
2. **Quality** (high signal-to-noise ratio)
3. **Diversity** (coverage of desired capabilities)

Optimizing all three simultaneously is the central challenge of data engineering for AI.

### 1.1 The Data Pipeline at Scale

```
Raw Data Sources
    ↓
Data Collection → Crawling, APIs, Licensing, Partnerships
    ↓
Data Processing → Extraction, Normalization, Language Detection
    ↓
Data Filtering → Quality, Safety, PII, Deduplication, Decontamination
    ↓
Data Selection → Mixing, Curriculum, Heuristic/Metric-based Selection
    ↓
Data Preparation → Tokenization, Formatting, Packing, Shuffling
    ↓
Training Ready → Sharded, Indexed, Ready for Loaders
```

---

## 2. Data Collection Strategies

### 2.1 Web Crawling

The primary source of pre-training data for most LLMs.

**Crawling Infrastructure:**
- **Common Crawl:** Free, publicly available crawl of 50B+ web pages. Used by GPT-3, LLaMA, Mistral, most open models. Monthly snapshots, 50-80 TB per month.
- **Custom Crawling:** For domain-specific data or curation control. Frameworks: Scrapy, Apache Nutch, StormCrawler.
- **Human-Curated:** Wikipedia, Books (Project Gutenberg, Bibliotik), Academic papers (arXiv, PubMed, Semantic Scholar), Code (GitHub, Stack Overflow).

**Crawling Challenges:**
- **DNS resolution:** Billions of domain lookups needed
- **Politeness:** Rate limiting to avoid overwhelming servers
- **Freshness:** Content changes over time; need periodic re-crawls
- **Exclusion:** robots.txt compliance, content restrictions
- **Language:** Detecting/classifying languages for multi-lingual training

### 2.2 Licensed Data

Increasingly important as copyright concerns grow:

- **Reddit:** Licensed to Google (Gemini), $60M/year
- **Stack Overflow:** Licensed to OpenAI
- **News Corp:** Licensed to OpenAI ($250M+)
- **Shutterstock:** Licensed to OpenAI, Meta
- **Getty Images:** Licensed to NVIDIA, Google
- **AP News:** Licensed to OpenAI
- **GitHub Code:** Training data for code models (Copilot, Codex, StarCoder)
- **Scientific articles:** PubMed Central (open), Elsevier (licensed), Springer (licensed)

### 2.3 Domain-Specific Data Collection

**Code Models:**
- GitHub public repositories (filtered by license, language, quality)
- Stack Overflow Q&A
- Documentation (readthedocs, MDN, official docs)
- Code review datasets
- Jupyter notebooks with outputs

**Medical Models:**
- PubMed abstracts and full-text articles
- Clinical trial registries (ClinicalTrials.gov)
- Electronic Health Records (de-identified)
- Medical textbooks, guidelines
- FDA documents, drug labels

**Legal Models:**
- Court cases (PACER, case.law, CourtListener)
- Legislation (Congress.gov, state legislatures)
- Regulations (Federal Register, Code of Federal Regulations)
- Legal documents (contracts, briefs, filings)
- Patent databases (USPTO, EPO, WIPO)

---

## 3. Data Quality and Filtering

### 3.1 Heuristic Filtering

Simple, fast filters applied before model-based filtering:

| Filter Type | Description | Example Threshold |
|-------------|-------------|------------------|
| **Length filter** | Too short or too long documents are noise | 200-100,000 tokens |
| **Symbol-to-word ratio** | High symbol content = low quality | <0.5 ratio of symbols |
| **Stop word frequency** | Indicates natural language | >0.1 stop word ratio |
| **Character repetition** | Gibberish detection | <0.5 repeated character ratio |
| **Word repetition** | Keyword stuffing detection | <0.3 repeated word ratio |
| **Language detection** | Keep only desired languages | fastText, CLD2, LangDetect |
| **Perplexity filter** | High perplexity = off-distribution | Train a small LM, filter outliers |
| **Toxicity filter** | Remove explicitly harmful content | Classifier-based |
| **PII filter** | Remove or flag personal information | Regex, NER, classifier |

### 3.2 Quality Classifiers

Train a classifier to predict whether a document is "high quality":

**Training Data:**
- Positive: Wikipedia, books, curated articles
- Negative: Uncategorized web pages, spam, low-quality content

**Architecture:**
- FastText (fast, good for many languages)
- BERT/RoBERTa (more accurate, slower)
- Custom transformer (domain-specific)

**LLaMA 3 Approach:** Used a transformer-based quality classifier to score and filter 15T+ tokens from 5× that raw data. The classifier was trained on 10K+ hand-labeled examples covering 100+ quality dimensions.

### 3.3 Perplexity Filtering

Train a small reference model (e.g., KenLM, 6M-parameter transformer) and filter documents where the model assigns high perplexity (surprising content).

**Use:** Remove documents that are unlike the distribution of high-quality text. Documents about rare topics, unusual formatting, or off-domain content get filtered.

**LLaMA 2 Approach:** Perplexity filtering removed URLs with very high or very low perplexity relative to a Wikipedia-based reference model.

### 3.4 FineWeb Quality Signals

The FineWeb dataset (Hugging Face, 2024) introduced multiple quality signals for web data:

| Signal | Method | What It Measures |
|--------|--------|-----------------|
| **C4 quality** | Heuristic filters from C4 dataset | Basic web page quality |
| **WET quality** | Character-level statistics | Formatting consistency |
| **DupLM score** | LM-based duplication detection | Uniqueness of content |
| **URL score** | URL structure analysis | Likelihood of being spam/not |
| **Category score** | GNN-based on page structure | Indicates article-like quality |
| **Domain quality** | Aggregated per-domain rating | Domain-level trustworthiness |

---

## 4. Data Deduplication

Deduplication is one of the most important and impactful data processing steps. It improves model performance by preventing memorization and reducing training redundancy.

### 4.1 Exact Deduplication

Remove identical documents:

**Techniques:**
- Hash-based (SHA256, MD5 of normalized text)
- Bloom filters for memory-efficient de-duplication at scale
- Hash tables for exact matching

**Scale:** LLaMA 3 deduplicated 30 trillion tokens → removed ≈10-15% of data
**Impact:** Prevents models from memorizing specific examples, improves generalization

### 4.2 Near-Deduplication

Remove documents that are similar but not identical:

**Locality-Sensitive Hashing (LSH):**
- MinHash: Jaccard similarity estimation
- SimHash: Cosine similarity via hash collisions
- Parameters: number of hashes, bands, rows per band
- Typically: 256 hashes, 16 bands, 16 rows = similarity threshold ~0.75

**Document/Fuzzy Deduplication:**
- Prefix filtering: documents sharing long prefix are similar
- Edit distance: Levenshtein, Damerau-Levenshtein
- k-mer overlap: DNA-inspired (ATCG for nucleotides → tokens for text)

**Sentence-Level Deduplication:**
- Find repeated sentences across documents
- Flag documents with excessive repeated sentences
- Important for: news articles (same story from multiple sources)

**Embedding-Based Deduplication:**
- Embed documents → cluster embeddings → remove near neighbors
- More expensive but more accurate
- Pre-filter with LSH, then use cosine similarity for final dedup

### 4.3 Datasets and Deduplication Rates

| Dataset | Raw Size | After Basic Dedup | After Near-Dedup | Retention Rate |
|---------|:--------:|:-----------------:|:----------------:|:--------------:|
| Common Crawl (2024) | 100T tokens | ~60T tokens | ~40T tokens | 40% |
| C4 | ~800B | ~360B | ~180B | 22% |
| RefinedWeb | 5T tokens | — | ~0.6T tokens | 12% |
| FineWeb | 15T tokens | — | ~3.4T tokens | 23% |
| LLaMA 3 | 30T+ tokens | — | ~15T tokens | ~50% |

---

## 5. Data Decontamination

Decontamination ensures that evaluation data does not overlap with training data. Contamination inflates benchmark scores and masks true model capability.

### 5.1 n-gram Matching

**Method:** Check if any n-gram from an evaluation example appears in the training data.

**n-gram sizes:** typically 8-13 for text, 50+ for code
**Threshold:** If an n-gram match is found, the example is contaminated

**Issues:**
- Too short n-grams: false positives (common phrases)
- Too long n-grams: false negatives (rephrased contamination)
- Simple rephrasing bypasses n-gram detection

### 5.2 Contamination Detection at Scale

**Prefix-based matching:** Check if the first 50 characters of an eval question match training data — if so, the model likely saw the exact or near-exact question.

**Embedding-based:** Embed eval queries → find nearest neighbors in training data → manual inspection of matches.

**Training dynamics:** Monitor loss curve for eval data during training — if loss drops suspiciously fast, the model may have seen similar data earlier.

**Exact submission matching:** For coding benchmarks (HumanEval, MBPP), check if eval problems or solutions appear in training data (Stack Overflow, GitHub).

### 5.3 Known Contamination Incidents

| Model | Contamination | Impact |
|-------|---------------|--------|
| **GPT-3** | Training data included Wikipedia, which overlaps with many benchmarks | Overstated few-shot learning (memorization vs generalization) |
| **Codex** | Training on GitHub included code similar to HumanEval problems | Up to 40% of HumanEval solutions may have been seen |
| **LLaMA 1** | Training on C4/C4 overlaps with many eval datasets | Multiple benchmarks likely inflated |
| **GPT-4 Technical Report** | Acknowledged contamination but argued it was "limited" | Unknown |
| **DeepSeek-R1** | Explicit contamination analysis — reported minimal | Cleanest large-scale contamination report |

### 5.4 Best Practices

- **Publish contamination analysis** with model releases (method + results)
- **Use multiple benchmark versions** (augmented, paraphrased, translated)
- **Hide benchmark data sources** from crawlers (robots.txt)
- **Post-hoc decontamination:** Remove contaminated examples from eval + re-evaluate
- **Dynamic benchmarks:** LiveBench, HumanEval+, fresh data regularly

---

## 6. Data Mixing and Curriculum

### 6.1 The Data Mixing Problem

Given multiple data sources (web, books, code, math, science), what proportions produce the best model?

**Key Findings:**
- **Web data:** Provides broad knowledge but is noisy (typically 50-70% of mix)
- **Books/long-form:** Improves coherence, narrative understanding (5-15%)
- **Code:** Improves reasoning, logical structure, instruction following (5-20%)
- **Math/science:** Improves formal reasoning (2-10%)
- **Academic papers:** Improves technical depth (2-10%)

### 6.2 Optimal Data Mixes from Major Models

| Model | Web | Books | Code | Math | Academic | Other |
|-------|:---:|:-----:|:----:|:----:|:--------:|:----:|
| **GPT-3** | 60% | 20% | 5% | 3% | 7% | 5% |
| **LLaMA 1** | 67% | 15% | 4% | — | 4% | 10% |
| **Chinchilla** | 80% | 10% | — | — | 10% | — |
| **LLaMA 2** | 65% | 10% | 5% | 5% | 5% | 10% |
| **LLaMA 3** | 50% | 15% | 15% | 5% | 5% | 10% |
| **Qwen 2.5** | 55% | 10% | 15% | 8% | 7% | 5% |
| **DeepSeek-V2** | 50% | 12% | 15% | 10% | 8% | 5% |

### 6.3 Curriculum Learning

Train on easier data first, harder data later:

**Data Ordering Heuristics:**
- **Perplexity-based:** Lower-perplexity first (web data), higher-perplexity later (code, math)
- **Document length:** Short first (for learning token statistics), longer later (for coherence)
- **Quality score:** Higher quality first (essential patterns), lower quality later (broad coverage)
- **Domain ordering:** Wikipedia → web → books → code → math → academic

**LLaMA 3 Curriculum:**
1. Start: web data (broad coverage, learn token patterns)
2. Middle: code + web (develop reasoning)
3. Late: math + science (specialized reasoning)
4. Final mix: anneal on high-quality data

### 6.4 Annealing and Data Quality Upsampling

**Annealing:** In the final stages of training, increase the proportion of high-quality data (curated, human-written). This "polishes" the model without changing its broad knowledge.

**LLaMA 3:** Used 1.5T tokens of annealed high-quality data at the end of training
**Mistral:** Annealed on 1T tokens of high-quality curated data
**Chinchilla:** Showed longer training on more data beats training on less compute-optimal data (contradicting Kaplan scaling)

### 6.5 Data Selection by Model Feedback

**Model-based data selection:**
1. Train a small model on a subset of data
2. Evaluate on a held-out validation set
3. Identify which data points contributed most to improvement
4. Upsample those data types

**DSIR (Data Selection via Importance Resampling):** Resample from a large pool to match a target distribution of high-quality data. Used by Pythia, LLaMA.

**DoReMi (Domain Reweighting with Minimax):** Optimize data weights to minimize worst-case domain loss. Automatically finds optimal mixing ratios.

---

## 7. Synthetic Data Generation

### 7.1 When to Use Synthetic Data

| Use Case | Best For | Limitations |
|----------|----------|-------------|
| **Instruction-tuning data** | SFT data (query → response pairs) | Risk: model collapse if purely synthetic |
| **Code generation training** | Functions, tests, documentation | Works well — code is verifiable |
| **Math problem generation** | Reasoning training, RLHF steps | Quality control needed for correctness |
| **Preference data for alignment** | RLHF pairs (chosen/rejected) | Might amplify existing model biases |
| **Data augmentation** | Increasing diversity of existing data | Requires careful quality filtering |
| **Long-context data** | Need long examples for context-window training | Complex to generate correctly |
| **Domain-specific training** | Specialized knowledge areas (medicine, law) | Requires seed data from domain experts |

### 7.2 Techniques

**Self-Instruct (Stanford, 2022):**
1. Seed tasks: write 175 diverse instruction templates
2. Generate instructions: for each template, use LLM to produce concrete instructions
3. Generate responses: for each instruction, use LLM to produce a response
4. Filter: remove low-quality, duplicate, or off-topic samples
5. Repeat: with each iteration, expand the instruction pool

**Evol-Instruct (WizardLM, 2023):**
1. Depth evolving: "Make this instruction more complex by adding constraints"
2. Breadth evolving: "Generate a different, novel instruction on the same topic"
3. Each iteration makes instructions harder and more diverse
4. Result: SFT data that teaches models to handle complex instructions
          
**Data Dreaming (Meta, 2024):**
- Generate coherent multi-turn conversations from single prompts
- Simulate realistic user interactions with an LLM playing the "user" role
- Used for: chatbot training data where real conversations are scarce

### 7.3 Model Collapse

When synthetic data is used recursively (training on model outputs), quality degrades:

```
Generation 1: Real data → Model A
Generation 2: Model A's outputs → Model B (trained on A's outputs + real)
Generation 3: Model B's outputs → Model C
...
Generation N: Models drift toward low-variance, generic outputs
```

**Mitigation Strategies:**
- Always mix synthetic with real data (≥30% real)
- Filter generated data aggressively (quality classifiers, human review)
- Use diversity constraints during generation
- Periodically refresh from real data sources
- Use multiple model architectures for generation

---

## 8. Data Annotation and Labeling

### 8.1 Annotation Types

| Type | Description | Typical Cost | Best For |
|------|-------------|:-----------:|----------|
| **Binary labels** | Good/bad, relevant/irrelevant | $0.05-0.10/sample | Safety filtering, preference data |
| **Categorical labels** | Topic, sentiment, toxicity category | $0.10-0.50/sample | Quality classification |
| **Ranking labels** | Ranking of multiple responses | $0.50-2.00/comparison | RLHF reward modeling |
| **Free-text critique** | Write feedback on model output | $2.00-10.00/task | Detailed alignment data |
| **Demonstration writing** | Write ideal response from scratch | $5.00-20.00/response | SFT data |
| **Expert annotation** | Domain-expert labeled data (medical, legal) | $20.00-100+/sample | Specialized domains |

### 8.2 Annotation Platforms

| Platform | Type | Scale | Quality |
|----------|------|:----:|:-------:|
| **Scale AI** | Managed workforce | Massive | High (managed) |
| **Surge AI** | AI-specialized annotators | Large | Very high |
| **LabelBox** | Self-managed + marketplace | Medium-large | Variable |
| **Appen** | Crowd + managed | Massive | Medium |
| **Amazon SageMaker Ground Truth** | Managed + auto-labeling | Large | High |
| **Prodigy** | In-house annotation tool | Small-medium | High (in-house) |
| **Label Studio** | Open-source tool | Custom | Custom |

### 8.3 Quality Control

**Inter-annotator agreement:**
- Measure: Cohen's κ (2 annotators), Fleiss' κ (3+ annotators)
- Target: κ > 0.7 for clear categories, κ > 0.5 for subjective tasks
- Low agreement → refine instructions, retrain annotators, or use adjudication

**Adjudication:**
- Third annotator resolves disagreements
- Expert reviewer for final decisions on contested labels

**Gold Standard Questions:**
- Insert known-quality questions into annotation batches
- Annotators who fail gold standards → retraining or removal

---

## 9. Data Governance and Compliance

### 9.1 Copyright and Licensing

**Current Legal Landscape (2026):**
- Training on copyrighted data is being challenged in courts worldwide
- Authors Guild vs OpenAI (SDNY, ongoing)
- NYT vs OpenAI and Microsoft (SDNY, ongoing)
- Getty Images vs Stability AI (UK High Court, ongoing)
- European copyright directive: text and data mining exception for research
- Japan Copyright Law: allows TDM for AI training
- Singapore Copyright Act: allows TDM for AI training

**Mitigation Strategies:**
- Licensing data directly (OpenAI's approach: $1B+ in licensing deals)
- Using only public domain/Creative Commons/open-source data
- Opt-out mechanisms: websites can declare AI training exclusion
- Model output fingerprinting (detect memorized copyrighted content)

### 9.2 Privacy and PII

**PII Types to Handle:**
- Names, addresses, phone numbers, email addresses
- Social security numbers, passport numbers, driver's license numbers
- Medical records, biometric data
- Financial account numbers
- User credentials, tokens, API keys

**PII Detection:**
- Regex patterns for structured PII (SSN, credit cards)
- NER models for unstructured PII (names, addresses)
- Rule-based heuristics for context-based PII
- Embedding-based for novel PII patterns

**PII Handling:**
- **Removal:** Replace PII with placeholder ([NAME], [PHONE])
- **Generalization:** Replace "123 Main St, Springfield" with "Springfield"
- **Synthesis:** Replace real PII with realistic fake PII
- **Suppression:** Remove entire documents containing PII above threshold

---

## 10. Data Infrastructure and Pipelines

### 10.1 Storage Systems

| System | Best For | Scale | Strengths |
|--------|----------|:-----:|-----------|
| **HDFS** | Large raw data storage | 10+ PB | Batch processing native |
| **S3/GCS/Blob** | Cloud-native object storage | Unlimited | Scalable, durable, streaming |
| **NFS/Lustre** | High-throughput model access | 100+ TB | Direct mounting, GPU cluster native |
| **SQL databases** | Metadata, annotations | 10+ TB | ACID, indexing, querying |
| **Vector databases** | Embedding storage, similarity search | 10B+ vectors | ANN search, hybrid search |

### 10.2 Processing Frameworks

- **Apache Spark:** Distributed data processing, ETL at petabyte scale
- **Apache Beam:** Unified batch + streaming pipeline model
- **Ray Data:** ML-native data processing with Python-first API
- **Dask:** Parallel computing in Python for moderate scale
- **Apache Flink:** Real-time streaming data processing

### 10.3 Data Loaders for Training

| Loader | Ecosystem | Features |
|--------|-----------|----------|
| **PyTorch DataLoader** | PyTorch | Distributed, sharded, streaming |
| **MosaicML Streaming** | Composer | Cloud-native streaming from S3/GCS |
| **WebDataset** | PyTorch | Tar-based efficient data storage |
| **Hugging Face Datasets** | HF Transformers | Hub integration, caching, processing |
| **DeepSpeed Data Efficiency** | DeepSpeed | Mixed precision, on-demand loading |

---

## 11. Data for Fine-Tuning

### 11.1 Instruction-Tuning Data

**Essential Properties:**
- **Diversity:** Instructions should cover many use cases, domains, and formats
- **Difficulty:** Mix of easy, medium, and hard instructions
- **Format consistency:** Consistent instruction format for generalization
- **Answer quality:** Well-written, accurate responses

**Key Datasets:**

| Dataset | Size | Description | Quality |
|---------|:----:|-------------|:-------:|
| **ShareGPT** | 100K+ | Real user-shared conversations | Variable |
| **OpenAssistant** | 150K | Human-annotated human-AI conversations | High |
| **Dolly** | 15K | Databricks employee-written tasks | Very high |
| **User-or-LLM** | 200K | Mix of human and LLM-generated data | Medium-high |
| **No Robots** | 10K | Hand-curated by Hugging Face team | Very high |
| **LIMA** | 1K | Minimal, highest-quality SFT data | Highest |

### 11.2 Data Composition for Fine-Tuning

| Component | Proportion | Purpose |
|-----------|:----------:|---------|
| General instruction following | 40-60% | Broad capabilities |
| Chat/multi-turn | 10-20% | Conversation ability |
| Reasoning (math, logic) | 10-15% | Reasoning improvement |
| Code | 5-10% | Code generation |
| Safety/refusal | 5-10% | Safety behavior |
| Structured output (JSON) | 2-5% | Format compliance |

---

## 12. Data for RLHF and Alignment

### 12.1 Preference Data Construction

**Collection Pipeline:**
1. Generate multiple responses per prompt from different models/configurations
2. Present responses to human annotators
3. Collect pairwise preferences (which is better?)
4. Also collect: ranking, rating, justification
5. Quality metrics: inter-annotator agreement, label distribution

**Preference Data Dimensions:**
- **Helpfulness:** Which response better addresses the user's intent?
- **Harmlessness:** Which response is safer and more appropriate?
- **Honesty:** Which response is more truthful?
- **Format:** Which response is better structured?

**Data Scale for RLHF:**
- Reward model training: 100K-1M+ preference comparisons
- PPO training: 10K-100K prompts for online generation
- Anthropic's RLHF data: 1M+ preference comparisons
- OpenAI's InstructGPT: 20K+ demonstrations + 40K+ comparisons

---

## 13. Data for Evaluation

### 13.1 Benchmark Contamination Prevention

- **Fresh data:** Create new evaluation data after training data cutoff
- **Multi-version:** Generate multiple variants of each evaluation question
- **Canary strings:** GPT-2 showed that canaries in training data can be used to verify contamination
- **Dynamic evaluation:** LiveBench updates questions monthly

### 13.2 Holdout Datasets

- Reserve 0.1-1% of training data as a held-out validation set
- Never filter, deduplicate, or contaminate this holdout
- Monitor: does performance on holdout track performance on public benchmarks?

### 13.3 Data Quality Evaluation

- **Diversity metrics:** Coverage of topics, domains, formats
- **Contamination score:** n-gram overlap with known benchmarks
- **Quality score:** Model perplexity, classifier score, human rating
- **Novelty score:** How much new information does this data add?
- **Bias metrics:** Representation across demographic groups

---

## 14. Data Curation at Scale: Case Studies

### 14.1 LLaMA 3 Data Pipeline

Meta's largest data pipeline for LLaMA 3:

**Scale:** 30T+ tokens → 15T tokens after processing
**Sources:** Common Crawl, Wikipedia, Books, Code, Math, Academic Papers, Social Media
**Process:**
1. Crawl → 150T tokens (5× the final size)
2. Heuristic filtering → 60T tokens (quality: 40% retention)
3. Quality classifier → 30T tokens (ML-based filtering: 50% retention)
4. Deduplication (exact + near) → 22T tokens (27% removal)
5. Decontamination → 20T tokens (9% removal via n-gram matching)
6. Data mixing + annealing → 15T tokens final training mix
**Compute:** 300K+ GPU-hours for data processing alone

### 14.2 FineWeb (Hugging Face)

Open-source data curation benchmark:

**Approach:**
- Started with Common Crawl snapshots
- Applied 10+ quality signals
- Used ablations to measure each filter's contribution
- Published both the data and the pipeline

**Key Insight:** Different models benefit from different filtering levels. LLaMA-like models benefit from aggressive filtering; Gemini-like models from moderate filtering with broader coverage.

---

## 15. Cloud-Native Data Engineering and Cost Analysis

### 15.1 Cloud Data Pipeline Comparison

| Cloud Provider | Storage | Processing | Data Warehouse | ML Integration | Best For |
|:--------------|:--------|:-----------|:---------------|:--------------|:---------|
| **AWS** | S3 (unlimited, 11 9s durability) | EMR (Spark), Glue (ETL), Athena (SQL) | Redshift | SageMaker, Bedrock | Broadest ecosystem, mature tools |
| **GCP** | GCS (unlimited, 11 9s) | Dataflow (Beam), Dataproc (Spark), BigQuery | BigQuery (serverless) | Vertex AI, Model Garden | Serverless data warehouse, AI-native |
| **Azure** | Blob Storage (unlimited) | Azure Synapse, Data Factory, HDInsight | Synapse Analytics | Azure ML, AI Studio | Microsoft enterprise integration |
| **Self-hosted (on-prem)** | HDFS, NFS, MinIO | Apache Spark, Flink, Dask | ClickHouse, Druid, Pinot | Custom ML ops | Maximum control, data sovereignty |

### 15.2 Cost Estimation by Pipeline Scale

| Scale Stage | Monthly Data Volume | Compute (Data Proc.) | Storage | Total Estimate | Best Platform |
|:-----------|:------------------:|:--------------------:|:-------:|:--------------:|:-------------:|
| **Startup** | 1-10 TB | $200-1K (spot instances) | $50-200 (S3/GCS) | **$300-1.5K** | AWS + S3 + Athena |
| **Growth** | 10-100 TB | $1K-10K (auto-scaling clusters) | $200-2K | **$1.5K-15K** | GCP + BigQuery + Dataflow |
| **Scale** | 100 TB - 1 PB | $10K-100K (dedicated Spark) | $2K-20K | **$15K-150K** | Multi-cloud + custom infra |
| **Hyperscale** | 1 PB+ | $100K-1M+ (customized infra) | $20K-100K+ | **$150K-1M+** | Self-hosted + cloud hybrid |

**Cost optimization tips:**
- Use **spot/preemptible instances** for non-critical preprocessing (30-60% savings)
- **Object storage lifecycle policies** — auto-move cold data to cheaper tiers after N days
- **Columnar formats** (Parquet, ORC) reduce scan costs by 50-90% vs JSON/CSV
- **Data compression** — Zstandard (zstd) gives best speed/ratio trade-off (~2-5× compression)
- **Partition pruning** — partition by date/region to limit bytes scanned in queries

### 15.3 Data Versioning and Lineage

| Tool | Type | Storage Backend | Scale | Key Feature |
|:----|:----:|:---------------:|:----:|:------------|
| **DVC** | Git-based data versioning | S3, GCS, HDFS, local | Medium | Git-native workflow, pipeline tracking |
| **LakeFS** | Git-like branches for data lakes | S3, GCS, Azure Blob | Large | Zero-copy branches, rollback, GCs |
| **Pachyderm** | Data lineage + pipeline | Object + file system | Large | Automatic provenance, diff between commits |
| **Dolt** | Version-controlled SQL DB | Local/S3 | Medium | Cell-level diff, merge, branch for data |
| **Delta Lake** | ACID transactions on data lake | S3, ADLS, GCS | Very large | Time travel, schema enforcement, merge |
| **Great Expectations** | Data quality + expectations | Any (metadata store) | Any | Automated data validation, documentation |

**Example: DVC pipeline for dataset versioning:**

```bash
# Initialize DVC in your project
dvc init

# Track a dataset
dvc add data/raw_training_data.parquet

# The dataset is now versioned — commit the .dvc file
git add data/raw_training_data.parquet.dvc .gitignore
git commit -m "track raw training data v1"

# Push to remote storage
dvc remote add -d myremote s3://my-bucket/dvc-store
dvc push

# Later, switch to a previous version
git checkout <commit-hash>
dvc checkout
```

### 15.4 Data Quality Monitoring in Production

| Approach | What It Checks | Frequency | Implementation |
|----------|:-------------:|:---------:|:--------------:|
| **Schema validation** | Column types, nullability, constraints | Every batch | Great Expectations, Pandera, Pydantic |
| **Statistical profiling** | Distribution, min/max, mean, std, quantiles | Daily | whylogs, Evidently, Pandas profiling |
| **Drift detection** | Feature/target distribution shift | Daily/weekly | Evidently, NannyML, Alibi Detect |
| **Anomaly detection** | Unexpected values, outliers | Real-time or batch | Isolation Forest, Z-score, IQR |
| **Freshness checks** | Last updated timestamp, staleness | Every pipeline run | Custom scheduler + alerting |
| **Bias monitoring** | Demographic parity, equal opportunity | Monthly | AIF360, FairLearn, SHAP |

```python
# Example: Great Expectations data quality check
import great_expectations as ge

# Load a DataFrame into a Great Expectations context
df = ge.from_pandas(raw_df)

# Define expectations
df.expect_column_values_to_not_be_null("user_id")
df.expect_column_values_to_be_between("age", 0, 120)
df.expect_column_values_to_be_in_set("country", 
    ["US", "UK", "DE", "FR", "JP", "CN", "BR", "IN", "CA", "AU"])

# Run validation
results = df.validate()
assert results["success"], f"Data quality failed: {results['statistics']}"
print(f"Validated: {results['statistics']['evaluated_expectations']} expectations, "
      f"{results['statistics']['successful_expectations']} passed")
```

### 15.5 Data Engineering for LLM Fine-Tuning

| Data Stage | Key Activities | Tools | Typical Timeline |
|:-----------|:-------------|:------|:---------------:|
| **Collection** | Source identification, licensing, extraction | Custom scrapers, APIs, HuggingFace Datasets | 1-4 weeks |
| **Cleaning** | Dedup, decontamination, PII removal, formatting | MinHash, LSH, fastText, regex | 1-2 weeks |
| **Annotation** | SFT demonstrations, preference pairs, quality labels | Scale AI, Surge, Label Studio | 2-8 weeks |
| **Formatting** | Chat template, system prompt, packing, sharding | HuggingFace tokenizer, Megatron-LM | Days |
| **Validation** | Holdout creation, contamination check, quality audit | Custom scripts, embedding similarity | 1-2 weeks |
| **Iteration** | Training → eval → error analysis → data improvement | W&B, MLflow, data flywheel | Continuous |

**Data flywheel for LLM improvement:**
```
Model Training → Evaluation → Error Analysis → Data Augmentation → Model Training (improved)
```

This iterative cycle is the most effective strategy for improving model quality beyond architecture scaling.

---

## 15a. Data Engineering for Multimodal Models

Multimodal models (processing text, images, audio, video simultaneously) present unique data engineering challenges compared to unimodal training.

### 15a.1 Multimodal Data Sources and Formats

| Modality | Common Formats | Typical Source | Storage Size | Tokenization |
|:---------|:--------------|:---------------|:-----------:|:------------|
| **Text** | JSONL, Parquet, TXT | Web crawl, books, code, documents | ~1-10 TB (for 1T tokens) | BPE, SentencePiece, tiktoken |
| **Image** | JPEG, PNG, WebP | LAION, COCO, ImageNet, web scrape | ~100-500 TB (per billion images) | ViT patches (14×14, 16×16); VQ-VAE tokens |
| **Audio** | WAV, MP3, FLAC | LibriSpeech, Common Voice, YouTube | ~10-100 TB (per million hours) | Mel-spectrogram patches; HuBERT/wav2vec tokens |
| **Video** | MP4, WebM, AVI | YouTube, Kinetics, HowTo100M | ~1-10 PB (per million hours) | Frame sampling (1-30 fps); 3D patches |
| **Code** | JSONL, ZIP of source files | GitHub, Stack Overflow | ~1-10 TB | Language-specific tokenizers (tree-sitter) |

### 15a.2 Interleaving Strategies for Multimodal Data

Models like GPT-4V, Gemini, and LLaVA require **interleaved** text+image data — text documents with images naturally placed within.

**Common interleaving formats:**

| Format | Description | Model Example | Data Example |
|:-------|:-----------|:-------------|:------------|
| **Sequential** | Text tokens followed by image tokens | LLaVA, BLIP-2 | `<image_embedding> ... text ...` |
| **Interleaved** | Multiple images interspersed in text | GPT-4V, Gemini | `text<image>text<image>text` |
| **Time-synchronized** | Video frames aligned to transcript | Video-LLaMA, Video-ChatGPT | `[frame at t=0] text [frame at t=5] text` |
| **Document-level** | Full document with all embedded media | Nougat, LayoutLM | PDF pages with figures, tables, equations |

**Interleaved data preprocessing pipeline:**
```python
def build_interleaved_sample(document, image_store, max_length=4096):
    """Convert a document with embedded image references into an interleaved training sample."""
    tokens = []
    for segment in document.segments:
        if segment.type == "text":
            tokens.extend(tokenizer.encode(segment.content))
        elif segment.type == "image":
            image = image_store.load(segment.url)
            image_tokens = image_encoder(image)  # e.g., 256 ViT patch embeddings
            tokens.extend(image_tokens)
        elif segment.type == "table" or segment.type == "chart":
            # Render as text or as image depending on model capability
            tokens.extend(render_table_as_text(segment))
        # Truncate to max length
        if len(tokens) >= max_length:
            break
    return torch.tensor(tokens[:max_length])
```

### 15a.3 Data Scaling Laws for Multimodal Models

| Aspect | Text-Only | Text+Image | Text+Image+Video |
|:-------|:--------:|:----------:|:----------------:|
| **Data volume (tokens/patches)** | 1-15T tokens | 1-5T text + 1-10B image pairs | 1-5T text + 1-10B images + 100M-1B video clips |
| **Storage (raw)** | 10-50 TB | 100-1000 TB | 1-100 PB |
| **Storage (processed)** | 1-5 TB | 10-100 TB | 100-1000 TB |
| **Compute (pre-training)** | 10K-100K GPU-hours | 100K-1M GPU-hours | 1M-10M GPU-hours |
| **Key scaling finding** | More tokens → better language | Quality of image-text alignment matters more than quantity | Video scaling is compute-bound, not data-bound |

**Key insight from multimodal scaling research:** For text+image models, **image-text alignment quality** (how well captions describe images) is more important than dataset size beyond a threshold (~1B pairs). The LAION-5B paper found that filtering to 2B high-alignment pairs produced better models than using all 5B.

### 15a.4 Image-Text Alignment Quality Metrics

| Metric | What It Measures | Calculation | Typical Good Threshold |
|:-------|:----------------|:-----------|:---------------------:|
| **CLIP score** (cosine similarity) | How well image matches caption in CLIP embedding space | `cosine(CLIP(img), CLIP(text))` | >0.3 |
| **BLIP score** | How well caption describes image (BLIP-2 re-ranking) | BLIP-2 captioning likelihood | >0.25 |
| **LAION aesthetic score** | Visual quality of image (human-rated aesthetic) | MLP trained on LAION aesthetics subset | >5.0 |
| **NSFW score** | Probability of adult content | CLIP-based NSFW classifier | <0.1 |
| **OCR density** | Fraction of image containing text | OCR detection coverage | <0.3 (for non-text tasks) |
| **Face count** | Number of detectable faces | Face detection model | Task-dependent |
| **Watermark score** | Probability of watermark | Trained watermark detector | <0.5 |

```python
# Image-text pair quality filtering pipeline
import torch
from PIL import Image

def filter_image_text_pair(image_path, caption, clip_model, clip_processor,
                           aesthetic_model=None, nsfw_threshold=0.1):
    """Filter image-text pairs by quality scores."""
    image = Image.open(image_path).convert("RGB")

    # 1. CLIP score: alignment between image and text
    inputs = clip_processor(text=[caption], images=[image], return_tensors="pt")
    with torch.no_grad():
        outputs = clip_model(**inputs)
    clip_score = outputs.logits_per_image.item()
    if clip_score < 0.28:
        return False, {"reason": "low_clip_score", "clip_score": clip_score}

    # 2. NSFW filter
    if hasattr(clip_model, "nsfw_classifier"):
        nsfw_prob = clip_model.nsfw_classifier(inputs["pixel_values"]).sigmoid().item()
        if nsfw_prob > nsfw_threshold:
            return False, {"reason": "nsfw", "nsfw_prob": nsfw_prob}

    # 3. Aesthetic score (optional)
    if aesthetic_model is not None:
        features = clip_model.get_image_features(**inputs)
        aesthetic_score = aesthetic_model(features).item()
        return True, {"clip_score": clip_score, "aesthetic": aesthetic_score}

    return True, {"clip_score": clip_score}
```

### 15a.5 Video Data Processing Pipeline

Video data is the most challenging modality due to temporal dimension and massive storage requirements.

```python
def process_video_for_training(video_path, sample_rate=1.0, max_frames=128,
                               resolution=224, audio_track=False):
    """Process a video file into frame samples for multimodal training."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sample frames at the specified rate
    frame_interval = int(fps / sample_rate) if sample_rate > 0 else 1
    frames = []
    timestamps = []
    for i in range(0, total_frames, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (resolution, resolution))
            frames.append(frame)
            timestamps.append(i / fps)
        if len(frames) >= max_frames:
            break
    cap.release()

    # Return as numpy array [T, H, W, C]
    video_tensor = np.stack(frames) if frames else np.zeros((1, resolution, resolution, 3))
    return {
        "frames": video_tensor,
        "timestamps": timestamps,
        "fps": fps,
        "total_frames": total_frames,
        "duration_seconds": total_frames / fps
    }
```

**Video data storage strategies:**

| Strategy | Storage Cost | Read Speed | Best For |
|:---------|:-----------:|:---------:|:---------|
| **Store raw MP4** | 1× (lowest) | Slow (decompress per frame) | Archival, streaming pipelines |
| **Store as frame directory** | 10-50× (JPEG per frame) | Fast (random frame access) | Training, random frame sampling |
| **Store as WebDataset tar** | 5-10× (JPEG packed) | Fast (sequential read) | Distributed training |
| **Store as compressed numpy** | 2-5× (npz compressed) | Very fast (direct memory load) | Small datasets, prototyping |
| **Store as HDF5** | 1.5-3× (compressed chunks) | Fast (chunked read) | Medium-scale training |

### 15a.6 Cross-Modal Alignment Data

Alignment data teaches multimodal models to understand relationships between modalities:

| Alignment Type | Training Data | Typical Size | Loss Function | Example |
|:--------------|:-------------|:-----------:|:-------------|:--------|
| **Image-Text contrastive** | (image, caption) pairs | 1-5B pairs | InfoNCE | CLIP, SigLIP |
| **Image-Text generative** | (image, caption/QA) pairs | 100M-1B | Cross-entropy | LLaVA, BLIP-2 |
| **Video-Text** | (video clip, description) pairs | 100M-1B clips | Contrastive + generative | Video-LLaMA, InternVideo |
| **Audio-Text** | (audio clip, transcript) pairs | 100K-1M hours | Contrastive + CTC/seq2seq | Whisper, CLAP |
| **Image-Image** | (view1, view2) augmented pairs | Self-supervised | Contrastive | DINO, iBOT |
| **Interleaved multimodal** | Documents with mixed media | 100M-1B docs | Next-token prediction | GPT-4V, Gemini |

**Synthetic multimodal data generation:**
- **Describe images:** Use a captioning model (BLIP-2, LLaVA) to generate high-quality captions for images
- **Generate QA pairs:** Use LLM + caption to create visual question-answering data
- **OCR augmentation:** Run OCR on images and overlay text tokens in training data
- **Audio captioning:** Transcribe and describe audio content using Whisper + LLM

### 15a.7 Data Curation for Multimodal Models — Production Checklist

- [ ] **Source diversity:** Ensure coverage across domains (web, books, news, social, science, medical)
- [ ] **Image quality:** Minimum resolution (≥256×256 for ViT-L, ≥224×224 for ViT-B); aspect ratio preservation
- [ ] **Text quality:** Captions must be descriptive, factual, and in the target language
- [ ] **Deduplication across modalities:** Near-duplicate images that appear with different captions can cause confusion; use pHash + CLIP for cross-modal dedup
- [ ] **PII in images:** Detect and blur faces, license plates, personal documents
- [ ] **Copyright filtering:** Detect watermarks, known copyrighted imagery, trademarked content
- [ ] **Temporal freshness:** For news and current events, remove outdated image-text pairs
- [ ] **Distribution monitoring:** Track modality ratios (text tokens vs image patches), language distribution, domain distribution
- [ ] **Safety filtering:** NSFW content removal across all modalities; hateful memes detection
- [ ] **Validation set:** Held-out multimodal evaluation set with human-verified alignment

---

## 15b. Data Observability and Real-Time ML Pipelines

As data pipelines grow to petabyte scale, **data observability** — the practice of monitoring, alerting, and triaging data quality issues — becomes essential for maintaining ML system reliability.

### 15b.1 The Five Pillars of Data Observability

| Pillar | Definition | Key Questions | Monitoring Approach |
|:-------|:-----------|:--------------|:-------------------|
| **Freshness** | Is new data arriving on schedule? | "Did yesterday's batch land by 6 AM? Is the streaming pipeline keeping up?" | Time-series of last-updated timestamps; SLAs with alerting |
| **Distribution** | Is data distributed as expected? | "Did the schema change? Are null rates spiking?" | Statistical profiling (mean, std, quantiles, null ratio) per feature; drift detection |
| **Volume** | Is the expected volume of data flowing? | "Did DQN log volume drop by 90%? Did we lose a shard?" | Row count monitoring; partition-level volume checks; anomaly detection |
| **Schema** | Is the data structure well-defined and consistent? | "Did someone rename a column? Did a new enum value appear?" | Schema registry (Avro, Protobuf, JSON Schema); schema diff on every pipeline run |
| **Lineage** | Where did data come from, and how was it transformed? | "Which upstream source fed this training example? What transformations were applied?" | Automated data provenance tracking (OpenLineage, Marquez); column-level lineage |

### 15b.2 Tools Comparison for Data Observability

| Tool | Type | Freshness | Distribution | Volume | Schema | Lineage | Ease of Setup |
|:-----|:----:|:---------:|:-----------:|:-----:|:-----:|:-------:|:------------:|
| **Great Expectations** | Open-source Python | YES | YES | YES | YES | no (manual) | Easy (Python library) |
| **dbt + dbt test** | Transformation framework | no | YES | YES | YES | YES (native) | Medium (needs dbt setup) |
| **Monte Carlo** | SaaS | YES | YES | YES | YES | YES | Easy (agent install) |
| **Soda** | Open-source + SaaS | YES | YES | YES | YES | no | Easy (CLI + config) |
| **Deequ (AWS)** | Open-source (Spark) | no | YES | YES | YES | no | Medium (Spark required) |
| **Apache Griffin** | Open-source (Spark) | YES | YES | YES | no | no | Hard (Spark ecosystem) |
| **Datafold** | SaaS (diff-focused) | no | YES | no | YES | YES | Easy (CI integration) |
| **Elementl / Dagster** | Orchestrator | YES | YES | YES | YES | YES | Medium (full platform) |

### 15b.3 Real-Time Feature Engineering for ML Pipelines

Modern ML systems increasingly require real-time features — computed on the fly from streaming data rather than pre-computed in batch pipelines.

| Approach | Latency | Consistency | Compute Cost | Best For |
|:---------|:------:|:----------:|:------------:|:---------|
| **Batch pre-compute** (daily/hourly) | 1-24 hours | Strong (all features available at inference time) | Low (one-time compute) | User profiles, aggregated statistics, historical features |
| **Streaming aggregation** (Flink, Kafka Streams) | Seconds-minutes | Eventual (tumbling windows) | Medium (continuous compute) | Real-time counters, rolling averages, trending signals |
| **Online feature computation** (at inference time) | <100ms | Strong (computed on latest data) | High (computed per request) | Contextual features (time of day, device, location), request-dependent features |
| **Hybrid** (batch base + streaming delta) | Seconds | Strong (base is consistent, delta is fresh) | Medium | Most production systems — batch for stable features, streaming for freshness |

```python
# Real-time streaming feature pipeline example
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import KafkaSource
from pyflink.common import SimpleStringSchema
import json
import redis
import time

class StreamingFeatureComputer:
    def __init__(self, redis_host="localhost", redis_port=6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
    
    def compute_features(self, event):
        user_id = event["user_id"]
        event_type = event["event_type"]
        timestamp = event.get("timestamp", int(time.time()))
        
        pipe = self.redis_client.pipeline()
        pipe.hincrby(f"user:{user_id}:counts", "total_events", 1)
        pipe.hincrby(f"user:{user_id}:counts:{event_type}", "count", 1)
        pipe.set(f"user:{user_id}:last_seen", timestamp)
        pipe.execute()
        
        return {
            "user_id": user_id,
            "total_events": int(self.redis_client.hget(f"user:{user_id}:counts", "total_events") or 0),
            f"{event_type}_count": int(
                self.redis_client.hget(f"user:{user_id}:counts:{event_type}", "count") or 0
            ),
            "last_seen_seconds_ago": int(time.time()) - int(
                self.redis_client.get(f"user:{user_id}:last_seen") or timestamp
            )
        }

# Streaming Flink job: consume Kafka events -> compute features -> store in Redis
env = StreamExecutionEnvironment.get_execution_environment()
kafka_source = KafkaSource.builder()     .set_bootstrap_servers("localhost:9092")     .set_topics("user-events")     .set_group_id("feature-computer")     .set_value_only_deserializer(SimpleStringSchema())     .build()

def process_and_store(event_str):
    event = json.loads(event_str)
    computer = StreamingFeatureComputer()
    features = computer.compute_features(event)
    print(f"Features for user {event['user_id']}: {features}")
    return features

ds = env.from_source(kafka_source, watermark_strategy=None, source_name="user-events")
ds.map(process_and_store)
env.execute("real-time-feature-computer")
```

### 15b.4 Feature Store Architecture

A feature store is the central repository for ML features, providing consistent access across training and inference:

| Feature Store | Backend | Online Serving | Point-in-Time Correctness | Offline Backfill | Open Source |
|:-------------|:-------:|:-------------:|:-------------------------:|:----------------:|:----------:|
| **Feast** | Redis / DynamoDB / Firestore | YES | YES (timestamp-based joins) | YES (Spark, batch) | YES |
| **Tecton** | DynamoDB / Cassandra | YES | YES (automatic temporal joins) | YES (Spark, native) | no |
| **Hopsworks** | MySQL / RocksDB | YES | YES | YES (Hive, Spark) | YES |
| **Vertex AI Feature Store** | Bigtable / BigQuery | YES | YES | YES (native GCP) | no |
| **Databricks Feature Store** | DynamoDB / online store | YES | YES (time travel) | YES (Delta Lake) | no |
| **Azure Managed Feature Store** | Cosmos DB / Redis | YES | YES | YES (Synapse) | no |

**Key design decisions for a feature store:**

1. **Point-in-time correctness:** When training a model, each training example must use feature values that were *current at the time of the event*, not the latest values (which would leak future information). Feast calls this "point-in-time joins" — joining event timestamps to feature timestamps to ensure no time-travel data leakage.

2. **Online vs. offline separation:** Online feature store (low-latency, Redis/DynamoDB) serves real-time inference; offline feature store (Parquet/batch, BigQuery/S3) serves training. Both should be sourced from the same feature pipeline to ensure consistency.

3. **Feature validation:** Every feature registered in the feature store should include:
   - Expected type and range
   - Statistical profile (mean, std, quantiles, null rate)
   - Owner and domain (who to contact if the feature breaks)
   - Freshness SLA (how often the feature is updated)

### 15b.5 Data Contract Pattern for Cross-Team Data Pipelines

Data contracts define explicit quality guarantees between data producers and data consumers:

```yaml
# data_contract.yaml — Example contract for a "user-activity" dataset
dataset: user_activity
version: "2.1.0"
owner: team:analytics

schema:
  columns:
    - name: user_id
      type: STRING
      required: true
      description: "Unique user identifier, UUID v4"
      constraints:
        not_null: true
        unique: true
    - name: event_type
      type: STRING
      required: true
      allowed_values: ["click", "purchase", "login", "logout", "search"]
    - name: timestamp
      type: TIMESTAMP
      required: true
      freshness_sla: 30 minutes
    - name: session_id
      type: STRING
      required: false
      description: "Browser session identifier, nullable for API events"

quality_slas:
  - metric: row_count
    min: 1000000
    max: 10000000
  - metric: null_rate
    column: session_id
    max: 0.15
  - metric: distribution
    column: event_type
    reference: "production_event_distribution.json"
    tolerance: 0.05

notifications:
  - on: failure
    channel: slack
    target: "#data-pipeline-alerts"
```

Data contracts provide explicit guarantees that prevent silent data quality degradation from affecting downstream ML models and dashboards.

---

## 16. Cross-References

| Reference | Description |
|-----------|-------------|
| [01-Foundations/05-Training-Methodologies.md] | How training data is used in practice |
| [05-Enterprise/03-Fine-Tuning-Enterprise.md] | Enterprise data pipeline for fine-tuning |
| [06-Advanced/03-Evaluation-Benchmarks.md] | Evaluation data and contamination |
| [07-Emerging/02-AI-Safety.md] | Data safety, PII, toxic content filtering |
| [08-Reference/01-Glossary.md] | Key terms defined |
| [09-Papers/01-Foundational-Papers.md] | Foundational data engineering papers |
| [02-LLMs/05-NLP-Foundations.md] | Text preprocessing and tokenization |
| [06-Advanced/04-Prompt-Engineering.md] | Synthetic data generation via LLMs |
| [06-Advanced/01-Multimodal-AI.md] | Multimodal model training and data requirements |
| [01-Foundations/03-Deep-Learning.md] | Deep learning foundations for multimodal architectures |

---

*Document version: 2.0 — June 2026 | Tier 3: Expansion. [Added: §15b Data Observability and Real-Time ML Pipelines — five pillars, tools comparison, streaming feature engineering, feature store architecture, data contract pattern. Updated Cross-References.]* — June 2026 | Tier 2: Enriched — added §15a Data Engineering for Multimodal Models with interleaving strategies, scaling laws, image-text quality metrics, video processing pipeline, cross-modal alignment data, and production checklist*
