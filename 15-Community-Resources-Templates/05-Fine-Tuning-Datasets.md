# 05 — Fine-Tuning Datasets

> **Purpose:** Curated reference of fine-tuning datasets, data generation tools, and quality control methods for LLM fine-tuning.
> **Last Updated:** 2026-06-12
> **Maintainer:** AI Knowledge Base Team

---

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset Taxonomy](#dataset-taxonomy)
3. [Instruction Datasets](#instruction-datasets)
4. [Preference Datasets](#preference-datasets)
5. [Domain-Specific Datasets](#domain-specific-datasets)
6. [Multilingual Datasets](#multilingual-datasets)
7. [Synthetic Data Generation](#synthetic-data-generation)
8. [Data Filtering & Quality Control](#data-filtering--quality-control)
9. [Dataset Curation Tools](#dataset-curation-tools)
10. [Dataset Licenses & Ethics](#dataset-licenses--ethics)
11. [Training Recipes](#training-recipes)
12. [Evaluation Benchmarks](#evaluation-benchmarks)
13. [Further Reading](#further-reading)

---

## Introduction

Fine-tuning is the process of adapting a pre-trained language model to specific tasks, domains, or behaviors. The quality and composition of the fine-tuning dataset is the single most important factor in fine-tuning success.

This document provides:
- A **curated catalog** of publicly available datasets
- Links, sizes, and license information
- Tools for generating and filtering data
- Best practices for dataset composition
- Training recipe references

### The Fine-Tuning Data Landscape (2026)

```
                               INSTRUCTION DATA
                     ┌───────────────────────────┐
                     │  OpenAssistant  │  Dolly   │
                     │  ShareGPT       │  Alpaca  │
                     │  No Robots      │  LIMA    │
                     └───────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                                ▼
           PREFERENCE DATA                 DOMAIN DATA
     ┌──────────────────────┐     ┌──────────────────────┐
     │  HH-RLHF             │     │  Medical (MedQA)     │
     │  UltraFeedback       │     │  Legal (LeXFiles)    │
     │  HelpSteer           │     │  Code (CodeSearchNet)│
     │  Nectar              │     │  Math (GSM8K, MATH)  │
     └──────────────────────┘     └──────────────────────┘
                    │                                │
                    └───────────────┬───────────────┘
                                    ▼
                         SYNTHETIC DATA
                    ┌──────────────────────┐
                    │  Evol-Instruct       │
                    │  Self-Instruct       │
                    │  GLAN       │ UltraChat│
                    └──────────────────────┘
```

---

## Dataset Taxonomy

### By Purpose

| Category | Purpose | Example Datasets |
|----------|---------|-----------------|
| **Instruction Tuning** | Teach task-following behavior | OpenAssistant, Dolly, ShareGPT |
| **Preference Alignment** | Align with human preferences | HH-RLHF, UltraFeedback |
| **Domain Adaptation** | Specialize for a domain | MedQA, CodeAlpaca, LeXFiles |
| **Safety Tuning** | Reduce harmful outputs | BeaverTails, SafetyPrompts |
| **Multilingual** | Extend language coverage | Bactrian-X, Aya Dataset |
| **Reasoning** | Improve logical reasoning | GSM8K, MATH, PrOntoQA |
| **Tool Use** | Teach function calling | ToolBench, APIBank |

### By Size

| Size Range | Examples | Typical Use |
|------------|----------|-------------|
| Tiny (< 1K) | LIMA (1K), SuperNOVA | Hypothesis testing |
| Small (1K–10K) | Dolly (15K), HH-RLHF (160K) | Initial experiments |
| Medium (10K–100K) | OpenAssistant (88K), UltraFeedback (64K) | Quality-focused tuning |
| Large (100K–1M) | ShareGPT (1M+), Alpaca (52K) | General instruction tuning |
| Very Large (> 1M) | The Pile (800GB), C4 | Pre-training, not fine-tuning |

### By Quality Level

| Quality | Characteristics | Examples |
|---------|----------------|----------|
| **Gold** | Expert human-written, verified | LIMA, OpenAssistant (level 1) |
| **Silver** | Human-written, moderate filtering | Dolly, ShareGPT (filtered) |
| **Bronze** | LLM-generated, quality-filtered | Alpaca, Self-Instruct |
| **Raw** | Unfiltered or minimally filtered | The Pile, Common Crawl |

---

## Instruction Datasets

### OpenAssistant Conversations (OASST1)

- **Description:** ~88K human-written assistant conversations. Multi-turn, multi-lingual (35+ languages). Human-labeled for quality.
- **Size:** ~88K conversation trees, ~400K individual messages
- **Format:** JSON (conversation trees with replies)
- **License:** Apache 2.0
- **Languages:** 35+ (primarily English, Spanish, German, French, Russian)
- **Quality:** Multiple human quality ratings per message
- **Use case:** General instruction tuning, multi-turn conversations
- **Access:** https://huggingface.co/datasets/OpenAssistant/oasst1
- **Citation:** Köpf et al., "OpenAssistant Conversations — Democratizing Large Language Model Alignment" (2023)

**Key strengths:** Multi-turn, multi-lingual, human quality ratings.

### Databricks Dolly 2.0

- **Description:** ~15K instruction-response pairs created by Databricks employees. Covers 8 categories: brainstorming, classification, closed QA, generation, information extraction, open QA, summarization, and rewriting.
- **Size:** 15K entries
- **Format:** JSONL (instruction, context, response, category)
- **License:** CC BY-SA 3.0
- **Languages:** English only
- **Quality:** Human-written by 5K+ Databricks employees
- **Use case:** General instruction tuning, baseline dataset
- **Access:** https://huggingface.co/datasets/databricks/databricks-dolly-15k
- **Citation:** Conover et al., "Free Dolly: Introducing the World's First Truly Open Instruction-Tuned LLM" (2023)

**Key strengths:** High-quality human data, commercially permissive license.

### ShareGPT (Various Versions)

- **Description:** User conversations with ChatGPT shared via ShareGPT browser extension. Multiple versions available with different filtering levels. **Note:** Check terms of service — using ShareGPT data may violate OpenAI's ToS.
- **Size:** ~100K–1M+ conversations (varies by version)
- **Format:** JSON/JSONL (conversation format)
- **License:** Various (community collections — check each version)
- **Languages:** English (primarily), many others
- **Quality:** Variable — depends on filtering. Can contain toxic content.
- **Use case:** General instruction tuning, conversational AI
- **Access:** Various HuggingFace repositories (search "ShareGPT")
- **Notable versions:**
  - **ShareGPT-90K** (cleaner, filtered)
  - **ShareGPT-4o** (includes multimodal)
  - **ShareGPT_WMT_Filtered** (translation pairs)

**Key strengths:** Real user queries, natural conversation patterns.
**Warning:** Legal/ethical concerns — not all versions have clear licensing.

### Stanford Alpaca

- **Description:** 52K instruction-response pairs generated by text-davinci-003. Generated from 175 seed tasks via Self-Instruct.
- **Size:** 52K entries
- **Format:** JSON (instruction, input, output)
- **License:** CC BY-NC 4.0 (commercial use may be restricted)
- **Languages:** English
- **Quality:** LLM-generated, minimal filtering. Contains errors and hallucinations.
- **Use case:** Starter dataset for instruction tuning, Self-Instruct methodology reference
- **Access:** https://huggingface.co/datasets/tatsu-lab/alpaca
- **Citation:** Taori et al., "Stanford Alpaca: An Instruction-following LLaMA model" (2023)

**Key strengths:** Simple, widely used, demonstrated to work.
**Limitations:** LLM-generated data inherits teacher model biases.

### LIMA (Less Is More for Alignment)

- **Description:** 1,030 carefully curated examples (750 from community forums + 250 from manual writing + 30 from prompts). Demonstrates that high-quality data beats large quantity.
- **Size:** 1,030 entries
- **Format:** JSONL (conversation format)
- **License:** CC BY-NC 4.0
- **Languages:** English
- **Quality:** Very high — human-selected and human-written examples
- **Use case:** Alignment research, quality-focused tuning
- **Access:** https://huggingface.co/datasets/GAIR/lima
- **Citation:** Zhou et al., "LIMA: Less Is More for Alignment" (2023)

**Key strength:** Demonstrates that 1K high-quality examples can match 50K+ low-quality ones.

### No Robots

- **Description:** ~10K human-written instructions and responses. Created specifically to be a high-quality, permissively licensed dataset.
- **Size:** ~10K entries
- **Format:** JSONL (conversation format)
- **License:** CC BY 4.0 (fully permissive)
- **Languages:** English
- **Quality:** High — human-written by contractors
- **Use case:** General instruction tuning, safe baseline
- **Access:** https://huggingface.co/datasets/HuggingFaceH4/no_robots

### WizardLM Data

- **Description:** Generated via Evol-Instruct methodology. 250K+ varied and complex instructions. Multiple versions (Evol-Instruct-70K, Evol-Instruct-V2, WizardCoder).
- **Size:** 70K–250K+ (varies by version)
- **Format:** JSON (instruction, output)
- **License:** CC BY-NC 4.0
- **Languages:** English
- **Quality:** LLM-generated with evolutionary complexity, high diversity
- **Use case:** Complex instruction tuning, code, reasoning
- **Access:** https://huggingface.co/datasets/WizardLM
- **Citation:** Xu et al., "WizardLM: Empowering Large Language Models to Follow Complex Instructions" (2023)

**Key strength:** Evolutionary approach creates diverse, complex instructions.

### Comparison Table

| Dataset | Size | Source | Quality | License | Multi-turn | Multi-lingual |
|---------|------|--------|---------|---------|-----------|---------------|
| OpenAssistant OASST1 | 88K trees | Human | ★★★★★ | Apache 2.0 | ✅ | ✅ (35+) |
| Dolly 2.0 | 15K | Human | ★★★★☆ | CC BY-SA | ❌ | ❌ |
| ShareGPT | 100K–1M | User/API | ★★★☆☆ | Varies | ✅ | ✅ |
| Alpaca | 52K | LLM | ★★★☆☆ | CC BY-NC | ❌ | ❌ |
| LIMA | 1K | Human | ★★★★★ | CC BY-NC | ❌ | ❌ |
| No Robots | 10K | Human | ★★★★☆ | CC BY | ✅ | ❌ |
| WizardLM | 70K–250K | LLM | ★★★★☆ | CC BY-NC | ❌ | ❌ |

---

## Preference Datasets

Preference datasets contain pairs (or rankings) of model outputs, used for RLHF, DPO, and other alignment methods.

### HH-RLHF (Anthropic Helpful & Harmless)

- **Description:** ~170K human preference comparisons. Two categories: helpfulness (preferred vs. rejected responses) and harmlessness (safe vs. unsafe). The foundational preference dataset.
- **Size:** ~170K preference pairs
- **Format:** JSONL (chosen, rejected, context)
- **License:** MIT
- **Languages:** English
- **Use case:** RLHF training, DPO training, safety alignment
- **Access:** https://huggingface.co/datasets/Anthropic/hh-rlhf
- **Citation:** Bai et al., "Training a Helpful and Harmless Assistant from Human Feedback" (2022)

**Key strengths:** Foundational RLHF dataset, covers both helpfulness and harmlessness.

### UltraFeedback

- **Description:** ~64K prompts with 256K model responses (4 responses per prompt), rated on 4 dimensions: instruction-following, honesty, helpfulness, and harmlessness. Responses from GPT-4, GPT-3.5, LLaMA, and other models.
- **Size:** ~64K prompts, ~256K responses
- **Format:** JSONL (prompt, responses, ratings, rationales)
- **License:** MIT
- **Languages:** English
- **Use case:** DPO training, reward model training, preference learning
- **Access:** https://huggingface.co/datasets/openbmb/UltraFeedback
- **Citation:** Cui et al., "UltraFeedback: Boosting Language Models with High-quality Feedback" (2023)

**Key strengths:** Multiple responses per prompt, multi-dimensional ratings, AI feedback rationales.

### HelpSteer (NVIDIA)

- **Description:** ~37K human preference comparisons with multi-dimensional annotations: correctness, helpfulness, coherence, complexity, verbosity. Designed for fine-grained reward modeling.
- **Size:** ~37K entries
- **Format:** JSONL (prompt, response, per-dimension scores)
- **License:** CC BY 4.0
- **Languages:** English
- **Use case:** Fine-grained reward modeling, DPO
- **Access:** https://huggingface.co/datasets/nvidia/HelpSteer
- **Citation:** Wang et al., "HelpSteer: Multi-attribute Helpfulness Evaluation of LLM Responses" (2024)

**Key strengths:** Multi-dimensional preference labels, fine-grained scoring.

### Nectar (Microsoft)

- **Description:** ~180K preference pairs with GPT-4 rankings. Each prompt has 7 responses ranked by GPT-4. Covers diverse task categories.
- **Size:** ~183K preference pairs
- **Format:** JSONL (prompt, 7 ranked responses)
- **License:** CC BY-NC 4.0
- **Languages:** English
- **Quality:** LLM-ranked by GPT-4
- **Use case:** Preference learning, reward modeling
- **Access:** https://huggingface.co/datasets/teknium/Nectar
- **Citation:** Teknium, "Nectar: A GPT-4 Quality Preference Dataset" (2024)

### Preference Dataset Comparison

| Dataset | Size | Source | Rating Type | License | Notes |
|---------|------|--------|-------------|---------|-------|
| HH-RLHF | ~170K | Human | Binary | MIT | Original RLHF dataset |
| UltraFeedback | ~64K prompts | AI (GPT-4) | Multi-dim + ranking | MIT | 4 responses per prompt |
| HelpSteer | ~37K | Human | Multi-dim scoring | CC BY 4.0 | 5 dimensions |
| Nectar | ~183K | AI (GPT-4) | Rankings | CC BY-NC | 7 ranked responses |
| Anthropic Red team | ~38K | Human | Binary | MIT | Adversarial safety data |
| SHP (Stanford) | ~385K | Human | Preferences | MIT | Reddit-based preferences |

---

## Domain-Specific Datasets

### Medical

| Dataset | Size | Description | License | Link |
|---------|------|-------------|---------|------|
| **MedQA (USMLE)** | ~12K | Multiple-choice medical questions | MIT | HuggingFace |
| **PubMedQA** | ~1K | Biomedical QA with PubMed abstracts | MIT | HuggingFace |
| **MedMCQA** | ~194K | Indian medical exam questions | MIT | HuggingFace |
| **ChatDoctor** | ~100K | Patient-doctor conversation dataset | MIT | GitHub |
| **BioMedLM Data** | Varies | PubMed articles and summaries | Varies | HuggingFace |
| **MedAlpaca** | ~40K | Medical instruction following | CC BY-NC | HuggingFace |

**Best for:** Clinical QA, medical summarization, patient communication.

### Legal

| Dataset | Size | Description | License | Link |
|---------|------|-------------|---------|------|
| **LeXFiles** | ~2.5M | Legal documents from various sources | CC BY 4.0 | HuggingFace |
| **LegalBench** | ~1.8K | Legal reasoning benchmark (also used for training) | MIT | GitHub |
| **CaseHOLD** | ~53K | Legal citation identification | MIT | HuggingFace |
| **CUAD** | ~510 | Contract understanding (annotations) | MIT | HuggingFace |
| **LexGLUE** | ~1.2M | Multiple legal tasks | CC BY 4.0 | HuggingFace |

**Best for:** Contract analysis, legal research, document classification.

### Code

| Dataset | Size | Description | License | Link |
|---------|------|-------------|---------|------|
| **CodeSearchNet** | ~2M | Code-search pairs (6 languages) | MIT | HuggingFace |
| **Code Alpaca** | ~20K | Code instruction following | CC BY-NC | HuggingFace |
| **Magicoder Data** | ~75K | Code generation (Evol-Instruct for code) | CC BY-NC | HuggingFace |
| **The Stack** | ~6TB | Large-scale source code (v2) | MIT + per-file | HuggingFace |
| **CodeExercises** | ~10K | Programming exercise solutions | MIT | GitHub |
| **OSS-Instruct** | ~75K | Open-source code generation | CC BY-NC | HuggingFace |
| **LeanDojo** | ~98K | Lean theorem proving data | Apache 2.0 | GitHub |

**Best for:** Code generation, code completion, code translation, test generation.

### Mathematics

| Dataset | Size | Description | License | Link |
|---------|------|-------------|---------|------|
| **GSM8K** | ~8.5K | Grade school math word problems | MIT | HuggingFace |
| **MATH** | ~12.5K | Competition math problems | MIT | HuggingFace |
| **MathInstruct** | ~262K | Diverse math instruction tuning | MIT | HuggingFace |
| **MetaMathQA** | ~395K | Augmented math reasoning data | MIT | HuggingFace |
| **NuminaMath** | ~860K | Math competition + CoT data | MIT | HuggingFace |
| **ProofNet** | ~5K | Formal math proofs | MIT | GitHub |

**Best for:** Math reasoning, step-by-step solution generation.

### Science & Research

| Dataset | Size | Description | License | Link |
|---------|------|-------------|---------|------|
| **SciQ** | ~13.7K | Science QA from textbooks | MIT | HuggingFace |
| **QED** | ~27K | Scientific reasoning data | CC BY 4.0 | HuggingFace |
| **SciBench** | ~10K | College-level science problems | MIT | GitHub |
| **Arxiv Dataset** | ~2M papers | Academic papers dataset | CC0 | HuggingFace |

---

## Multilingual Datasets

### Bactrian-X

- **Description:** Instruction tuning data for 52 languages. Generated via Google Translate from Alpaca-like seeds. Covers low-resource languages.
- **Size:** ~60K per language (52 languages)
- **License:** CC BY-NC 4.0
- **Access:** https://huggingface.co/datasets/MBZUAI/Bactrian-X

### Aya Dataset (Cohere For AI)

- **Description:** ~200K human-written instruction-response pairs across 119 languages. High-quality, human-curated.
- **Size:** ~200K entries
- **License:** Apache 2.0
- **Access:** https://huggingface.co/datasets/CohereForAI/aya_dataset

### xP3 / xP3mt (BigScience)

- **Description:** Cross-lingual pretraining and prompting. xP3: ~80M examples across 46 languages. xP3mt: 13 languages with machine translation.
- **Size:** ~80M examples
- **License:** Apache 2.0
- **Access:** https://huggingface.co/datasets/bigscience/xP3

### LM-Science-Questions

- **Description:** Scientific QA in multiple languages (en, zh, ja, de, fr, etc.)
- **Access:** HuggingFace

---

## Synthetic Data Generation

Synthetic data generation is the process of using LLMs to create training data. This has become the dominant approach for creating large instruction-tuning datasets.

### Self-Instruct

**Paper:** Wang et al., "Self-Instruct: Aligning Language Models with Self-Generated Instructions" (2022)

**Process:**
1. Start with a seed set of ~200 human-written instructions
2. Prompt an LLM to generate new instructions based on seeds
3. Generate responses for each instruction
4. Filter low-quality or duplicate entries
5. Iterate

**Code:** https://github.com/yizhongw/self-instruct

```python
# Simplified Self-Instruct pipeline
seed_tasks = load_seed_tasks()
generated_tasks = []

for _ in range(num_iterations):
    # Sample exemplars from seed + previously generated
    exemplars = sample_exemplars(seed_tasks + generated_tasks, k=8)
    
    # Generate new instructions
    prompt = build_generation_prompt(exemplars)
    new_instructions = llm.generate(prompt)
    
    # Filter (dedup, quality score)
    filtered = filter_instructions(new_instructions, existing=generated_tasks)
    
    # Generate responses
    for instruction in filtered:
        response = llm.generate(f"Instruction: {instruction}\nResponse:")
        generated_tasks.append({"instruction": instruction, "response": response})
```

### Evol-Instruct

**Paper:** Xu et al., "WizardLM: Empowering Large Language Models to Follow Complex Instructions" (2023)

**Process:**
1. Take a base instruction
2. Apply evolutionary operators:
   - **Add constraints** (e.g., "write in JSON format, under 100 words")
   - **Deepen reasoning** (e.g., "explain step by step and provide evidence")
   - **Increase difficulty** (e.g., "solve without using external tools")
   - **Concretize** (e.g., "give real-world examples instead of abstract")
3. Generate responses for evolved instructions
4. Eliminate instructions that are too similar or nonsensical

**Code:** https://github.com/huggingface/alignment-handbook (includes Evol-Instruct)

### GLAN (Generated Latent Augmented Data)

**Paper:** Lee et al., "GLAN: A Graph-based Approach to Generate Instruction Data at Scale" (2024)

**Process:**
1. Define a taxonomy of skills/abilities (e.g., "reasoning about cause and effect")
2. Build a skill graph showing relationships between abilities
3. Traverse the graph to generate varied instruction seeds
4. For each seed, generate full instruction-response pairs
5. Filter by difficulty and quality

**Key advantage:** More systematic coverage of knowledge domains vs. random sampling.

### UltraChat

**Paper:** Ding et al., "UltraChat: A Large-scale Multi-turn Dialog Dataset" (2023)

**Process:**
1. Define conversation scenarios (e.g., "student asking for homework help")
2. Use a two-model setup: one as "user" and one as "assistant"
3. Generate multi-turn conversations with topic transitions
4. Apply quality filters

**Key advantage:** Multi-turn conversations with natural topic flow.

### Synthetic Data Best Practices

| Practice | Description |
|----------|-------------|
| **Diverse seeds** | Use domain-specific seeds for broader coverage |
| **Quality filtering** | Filter by response length, repetition, perplexity, or classifier |
| **Deduplication** | Remove near-duplicate instructions (embedding-based) |
| **Difficulty balance** | Mix easy, medium, and hard examples |
| **Format variety** | Include multiple response formats (lists, JSON, paragraphs) |
| **Teacher diversity** | Use multiple LLMs as teachers to reduce bias |
| **Human validation** | Sample and validate 1-5% of generated data |

---

## Data Filtering & Quality Control

### Deduplication

| Method | Description | Tooling |
|--------|-------------|---------|
| **Exact dedup** | Remove identical strings | Unix `uniq`, Python sets |
| **MinHash LSH** | Near-duplicate detection | `datasketch`, `text-dedup` |
| **Embedding dedup** | Semantic similarity clustering | Sentence Transformers |
| **Bloom filter** | Approximate membership test | `pybloom_live` |

### Quality Filters

| Filter | What It Detects | Implementation |
|--------|-----------------|----------------|
| **Length filter** | Too short (< 5 words) or too long (> 2048 tokens) | Heuristic |
| **Perplexity filter** | Unnatural or random text | GPT-2 perplexity from `transformers` |
| **Repetition filter** | Repeated n-grams | Custom script |
| **Language ID** | Wrong language | `fasttext` language classifier |
| **Toxicity filter** | Toxic or offensive content | `detoxify` or OOD detector |
| **Instruction quality** | Vague or incomplete instructions | LLM-as-judge |
| **Response quality** | Hallucinations, errors | LLM-as-judge |
| **Diversity filter** | Semantic coverage | Embedding-based clustering |

### LLM-as-Judge Filtering

```python
# Use an LLM to judge data quality
import openai

def judge_quality(instruction, response):
    prompt = f"""
    Rate this instruction-response pair on a scale of 1-5 for:
    - Instruction clarity: How specific and unambiguous is the instruction?
    - Response quality: Is the response accurate, complete, and helpful?
    - Format compliance: Does the response follow the requested format?

    Instruction: {instruction}
    Response: {response}

    Format your response as JSON:
    {{"instruction_clarity": 1-5, "response_quality": 1-5, 
      "format_compliance": 1-5, "overall_score": 1-5, 
      "accept": true/false, "reason": "..."}}
    """
    
    result = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(result.choices[0].message.content)
```

### Data Composition Best Practices

```
Optimal data mix (empirically):
- 60-70% general instruction data
- 10-15% domain-specific data
- 10-15% code/math data
- 5-10% preference pairs
- 1-5% safety/constitutional data

Quality over quantity:
- 10K high-quality > 100K low-quality
- Multiple tasks > single task
- Multi-turn > single turn
- Diverse formats > single format
```

---

## Dataset Curation Tools

### Data Prep Tools

| Tool | Description | Link |
|------|-------------|------|
| **Argilla** | Data annotation and curation platform | https://github.com/argilla-io/argilla |
| **Label Studio** | Multi-format data labeling | https://labelstud.io/ |
| **Cleanlab** | Data quality and error detection | https://github.com/cleanlab/cleanlab |
| **Snorkel** | Programmatic data labeling | https://snorkel.ai/ |
| **DVC** | Data version control | https://dvc.org/ |
| **Lux** | Data visualization for quality checks | https://lux-ai.org/ |

### Data Generation Tools

| Tool | Description | Link |
|------|-------------|------|
| **Distilabel** | LLM-based synthetic data generation and annotation | https://github.com/argilla-io/distilabel |
| **Alpaca-LoRA** | Alpaca-style data generation | https://github.com/tloen/alpaca-lora |
| **text-only-dataset** | Dataset creation from text sources | https://github.com/huggingface/text-only-dataset |
| **Datasets library** | HuggingFace dataset loading and processing | https://huggingface.co/docs/datasets/ |
| **Alignment Handbook** | Training and data recipes by HuggingFace | https://github.com/huggingface/alignment-handbook |

### Dataset Search & Discovery

| Platform | Description | Link |
|----------|-------------|------|
| **HuggingFace Datasets** | 100K+ datasets | https://huggingface.co/datasets |
| **PapersWithCode Datasets** | Research datasets | https://paperswithcode.com/datasets |
| **Zenodo** | Academic dataset hosting | https://zenodo.org/ |
| **Kaggle Datasets** | Competition datasets | https://www.kaggle.com/datasets |
| **OpenDataDiscovery** | Dataset catalog | https://opendatadiscovery.org/ |

---

## Dataset Licenses & Ethics

### Common Licenses

| License | Can Use? | Can Modify? | Commercial? | Attribution? |
|---------|----------|-------------|-------------|--------------|
| **CC0 / Public Domain** | ✅ | ✅ | ✅ | ❌ |
| **MIT** | ✅ | ✅ | ✅ | ✅ |
| **Apache 2.0** | ✅ | ✅ | ✅ | ✅ + notice |
| **CC BY 4.0** | ✅ | ✅ | ✅ | ✅ |
| **CC BY-SA 4.0** | ✅ | ✅ | ✅ | ✅ + share-alike |
| **CC BY-NC 4.0** | ✅ | ✅ | ❌ | ✅ |
| **OpenRAIL** | ✅ | ✅ | ✅ | ✅ + behavioral restrictions |
| **ODC-By** | ✅ | ✅ | ✅ | ✅ |

⚠️ **Always verify the license of derived/combined datasets.** Creating a dataset from multiple sources may inherit the most restrictive license.

### Ethical Considerations

1. **Consent**: Was the data collected with consent? ShareGPT data, for example, raises consent questions.
2. **Privacy**: Does the dataset contain PII? Always scan for and remove personal information.
3. **Representation**: Does the dataset represent diverse demographics and viewpoints?
4. **Bias**: What biases might the dataset introduce or amplify?
5. **Harmful content**: Does the dataset contain toxic or harmful content? Even safety datasets need careful handling.
6. **Copyright**: Is the training data covered by copyright? Legal landscapes vary by jurisdiction.

### Data Governance Checklist

```
[ ] Sources documented with citations
[ ] License clearly specified
[ ] PII removed (NER scan + manual check)
[ ] Toxic content filtered (or explicitly noted)
[ ] Language and dialect composition documented
[ ] Demographic diversity assessed
[ ] Potential biases documented
[ ] Data versioned with DVC or similar
[ ] Provenance tracked (where each example came from)
[ ] Terms of use specified
```

---

## Training Recipes

### Fine-Tuning Methods Overview

| Method | Description | Data Requirements | Key Reference |
|--------|-------------|-------------------|---------------|
| **Full fine-tuning** | Update all model parameters | Large (10K+) | Classic approach |
| **LoRA** | Low-rank adaptation | Small (100+) | Hu et al. 2021 |
| **QLoRA** | Quantized LoRA | Small (100+) | Dettmers et al. 2023 |
| **DoRA** | Weight-decomposed LoRA | Small (100+) | Liu et al. 2024 |
| **RLHF** | Reinforcement learning from human feedback | Preference pairs | Ouyang et al. 2022 |
| **DPO** | Direct preference optimization | Preference pairs | Rafailov et al. 2023 |
| **ORPO** | Odds ratio preference optimization | Preference pairs | Hong et al. 2024 |
| **KTO** | Kahneman-Tversky optimization | Binary preference | Ethayarajh et al. 2024 |

### Sample Training Configuration (LoRA)

```yaml
# config.yaml for fine-tuning with LoRA
model:
  base_model: meta-llama/Llama-3.1-8B
  load_in_8bit: false
  load_in_4bit: true

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

training:
  dataset: path/to/data.jsonl
  format: "alpaca"  # or "sharegpt", "oasst1"
  num_epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2e-4
  warmup_steps: 100
  logging_steps: 25
  save_steps: 500
  eval_steps: 500
  max_seq_length: 2048
  packing: true  # Pack multiple samples into one sequence
  optimizer: paged_adamw_8bit

output:
  output_dir: ./fine-tuned-model
  push_to_hub: true
  hub_model_id: username/model-name
```

### Sample DPO Training Configuration

```yaml
# config_dpo.yaml for DPO training
model:
  base_model: meta-llama/Llama-3.1-8B
  load_in_4bit: true
  use_peft: true
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05

dpo:
  beta: 0.1  # DPO temperature parameter
  loss_type: sigmoid  # sigmoid, hinge, ipom, dpo_simple

data:
  dataset: path/to/preference_data.jsonl
  format: "preference_pairs"
  prompt_field: "prompt"
  chosen_field: "chosen"
  rejected_field: "rejected"

training:
  num_epochs: 1
  per_device_batch_size: 4
  gradient_accumulation_steps: 8
  learning_rate: 5e-6
  warmup_ratio: 0.1
  max_length: 2048
  max_prompt_length: 1024
```

---

## Evaluation Benchmarks

### Post-Fine-Tuning Evaluation

| Benchmark | What It Measures | Format | Link |
|-----------|-----------------|--------|------|
| **MMLU** | Knowledge across 57 subjects | Multiple choice | https://github.com/hendrycks/test |
| **HumanEval** | Code generation | Function completion | https://github.com/openai/human-eval |
| **GSM8K** | Math reasoning | Word problems | https://github.com/openai/grade-school-math |
| **HellaSwag** | Commonsense reasoning | Multiple choice | https://github.com/rowanz/hellaswag |
| **TruthfulQA** | Factuality | Multiple choice | https://github.com/sylinrl/TruthfulQA |
| **MT-Bench** | Multi-turn conversation quality | LLM-as-judge | https://github.com/lm-sys/FastChat |
| **AlpacaEval** | Instruction following | LLM-as-judge | https://github.com/tatsu-lab/alpaca_eval |
| **BBH (BIG-Bench Hard)** | Challenge tasks | Various | https://github.com/suzgunmirac/BIG-Bench-Hard |
| **IFEval** | Instruction following accuracy | Format compliance | https://github.com/google-research/google-research/tree/master/instruction_following_eval |

### Recommended Evaluation Suite

For a comprehensive fine-tuning evaluation:

```bash
# Run standard evaluation suite
# Using lm-eval-harness (v0.4+)
lm_eval \
  --model hf \
  --model_args pretrained=./fine-tuned-model,dtype=bfloat16 \
  --tasks mmlu,gsm8k,hellaswag,truthfulqa,bbh \
  --batch_size auto \
  --output_path ./eval_results \
  --num_fewshot 5
```

---

## Further Reading

- [03-Prompt-Libraries.md](03-Prompt-Libraries.md) — Prompt engineering for data generation
- [04-Agent-Toolkits.md](04-Agent-Toolkits.md) — Fine-tuning data for agents
- [06-Awesome-AI-Repos.md](06-Awesome-AI-Repos.md) — Dataset and fine-tuning repositories
- [HuggingFace Alignment Handbook](https://github.com/huggingface/alignment-handbook) — Production training recipes
- [axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) — Easy fine-tuning framework
- [LLM Data Best Practices](https://arxiv.org/abs/2407.15513) — Survey paper
- [DataComp-LM](https://github.com/mlfoundations/datacomp-lm) — Data filtering benchmark

---

*Document version 1.0 — Last updated 2026-06-12*
