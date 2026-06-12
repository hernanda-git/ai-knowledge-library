# 05 - Training Methodologies

> A comprehensive, deeply technical exploration of training methodologies for modern deep learning and large language models. Covers pre-training objectives, data pipelines, scaling laws, fine-tuning, RLHF/alignment, instruction tuning, synthetic data, and augmentation — with complete mathematical derivations, pseudocode, and practical implementation guidance.

---

## Table of Contents

1. [Pre-Training Objectives](#1-pre-training-objectives)
2. [Training Data Pipeline](#2-training-data-pipeline)
3. [Data Deduplication and Filtering](#3-data-deduplication-and-filtering)
4. [Data Mixing Strategies](#4-data-mixing-strategies)
5. [Scaling Laws](#5-scaling-laws)
6. [Curriculum Learning](#6-curriculum-learning)
7. [Multi-Task Learning](#7-multi-task-learning)
8. [Transfer Learning](#8-transfer-learning)
9. [Fine-Tuning](#9-fine-tuning)
10. [Parameter-Efficient Fine-Tuning (PEFT)](#10-parameter-efficient-fine-tuning-peft)
11. [RLHF: Reinforcement Learning from Human Feedback](#11-rlhf-reinforcement-learning-from-human-feedback)
12. [DPO and Direct Alignment Methods](#12-dpo-and-direct-alignment-methods)
13. [Constitutional AI and RLAIF](#13-constitutional-ai-and-rlaif)
14. [Instruction Tuning](#14-instruction-tuning)
15. [Synthetic Data Generation](#15-synthetic-data-generation)
16. [Data Augmentation for LLMs](#16-data-augmentation-for-llms)
17. [Further Reading and References](#17-further-reading-and-references)

---

## 1. Pre-Training Objectives

### 1.1 Autoregressive Language Modeling

The **autoregressive (AR) objective** — also called causal LM — is the most fundamental and widely used pre-training objective. The model predicts the next token given all previous tokens, factorizing the joint probability via the chain rule:

```
p(x) = Π_{t=1}^{T} p(x_t | x_{<t})
```

**Training loss (next-token prediction):**

```
L_AR(θ) = -Σ_{t=1}^{T} log p_θ(x_t | x_{1:t-1})
```

**Properties:**
- **Causal masking**: Each token can only attend to preceding tokens (left-to-right)
- **Inherently generative**: The model can generate text autoregressively
- **Zero-shot capabilities**: Emergent abilities at sufficient scale (GPT-2, GPT-3)
- **Token efficiency**: Every token is used for both context and prediction (no "wasted" tokens like MLM)

**Implementation:**
```python
# Autoregressive forward pass
def autoregressive_loss(model, tokens, attention_mask=None):
    logits = model(tokens)  # [batch, seq_len, vocab_size]
    # Shift: predict token t from context of tokens[:t]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = tokens[:, 1:].contiguous()
    loss = cross_entropy(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
    return loss
```

**Architectures using AR objective:**
- GPT family (GPT-1, GPT-2, GPT-3, GPT-4)
- Llama / Llama 2 / Llama 3
- Mistral / Mixtral
- PaLM / Gemini
- Qwen / DeepSeek
- Gemma / Phi

### 1.2 Masked Language Modeling (MLM)

Introduced by BERT (Devlin et al., 2019), MLM randomly masks some input tokens and trains the model to predict them from the bidirectional context.

**Objective:**
```
L_MLM(θ) = -Σ_{t ∈ masked} log p_θ(x_t | x_mask)
```

**Masking strategy (BERT-style):**
- Randomly select 15% of tokens in each sequence
- Of the selected tokens:
  - 80% replaced with [MASK]
  - 10% replaced with random token
  - 10% left unchanged

This mitigates the mismatch between pre-training (masked) and fine-tuning (unmasked) input.

**Efficiency concern:** Only 15% of tokens are used for computing the loss. This makes MLM less token-efficient than AR.

**Variants:**
- **Whole Word Masking (WWM)**: Mask all tokens in a word together (used by RoBERTa)
- **Entity-level Masking**: Mask named entities or phrases
- **Span-level Masking**: Mask contiguous spans of tokens (used by SpanBERT)

### 1.3 Prefix Language Modeling

Also known as **Prefix LM** or **Causal MLM**, this objective combines elements of AR and MLM (UniLM, Dong et al., 2019).

**Formulation:**
Given a sequence split into two parts:
- **Prefix** (first k tokens): Bidirectional attention — tokens attend to all other tokens in the prefix
- **Suffix** (remaining tokens): Causal attention — each token attends to all prefix tokens and previous suffix tokens

**Training loss:**
```
L = -Σ_{t=k+1}^{T} log p_θ(x_t | x_{1:k}, x_{k+1:t-1})
```

**Properties:**
- Efficient for encoder-decoder tasks: the prefix acts as the "encoder" input
- Used in T5's "prefix LM" mode, GLM (Du et al., 2022)
- Less common than pure AR or span corruption in modern models

### 1.4 Span Corruption / Denoising Objective

T5 (Raffel et al., 2020) introduced **span corruption**: replace contiguous spans of tokens with a single sentinel token, and train the model to predict the dropped-out tokens.

**Formulation:**
```
Input:  "The [MASK_0] and [MASK_1] in the park."
Target: "[MASK_0] best [MASK_0] walked [MASK_1] dog [MASK_1] <EOS>"
```

**Procedure:**
1. Sample a mean span length `λ` (e.g., λ=3) and corruption rate `ρ` (e.g., ρ=0.15)
2. Randomly select corrupted spans such that ~ρ·T tokens are masked
3. Replace each span with a unique sentinel token (<X>, <Y>, ...)
4. Target: concatenate the sentinel tokens + their corresponding spans in order

**Properties:**
- More token-efficient than MLM (no unmasked tokens in loss)
- Encourages long-span generation (harder than single-token prediction)
- The model learns to both understand corrupted input and generate coherent output — naturally suited for encoder-decoder architectures
- Can be applied in decoder-only mode (called "masked LM with causal masking" in some works)

**Span Masking vs Token Masking (variance analysis):**

Span corruption has higher variance per gradient step but also provides more learning signal per masked token. The optimal span length depends on the data modality: λ=3 for general text, λ=6 for code, λ=1 for tasks requiring fine-grained token prediction (entities).

### 1.5 UL2 Objectives

UL2 (Tay et al., 2022) unified multiple pre-training objectives within a single model using a **mixture-of-denoisers** framework.

**Three denoising paradigms:**

**1. S-denoising (Span corruption):**
Standard T5-style span corruption with medium spans. Enables next-token prediction at scale.

**2. X-denoising (Extreme denoising):**
Large corruption rates (50-100%) with large span lengths. Teaches the model to generate long coherent passages from minimal context.

**3. R-denoising (Prefix LM):**
Causal prefix + full generation suffix. Prepares the model for tasks like summarization and dialogue.

**Training formulation:**

UL2 alternates between these modes, either deterministically or stochastically. The modes can use:
- A common Transformer backbone
- Mode-specific attention masks (non-causal for S/X, causal for R)
- Mode-specific sentinel tokens (optional)

**Key insight:** Different denoising objectives teach different skills:
- S-denoising: Local coherence, lexical matching
- X-denoising: Long-range reasoning, production of long outputs
- R-denoising: Generation conditioned on prefix, summarization

**Architecture:** UL2 uses a **Mixture-of-Experts (MoE)** Transformer (Switch Transformer style) with a non-causal Prefix LM architecture.

**Modern influence:** The UL2 framework influenced:
- **PaLM 2 / Gemini**: Mixed objective training
- **FIM (Fill-in-the-Middle)**: Used in Code LLMs (InCoder, SantaCoder, Code Llama)
- **Mixture of denoisers** in T5-style continual pre-training

### 1.6 Fill-in-the-Middle (FIM)

FIM (Bavarian et al., 2022) trains the model to fill in a missing middle section of code, given a prefix and suffix.

**Objective:**
Given a document split into three parts: prefix `P`, middle `M`, suffix `S`:
```
Training input: [P] [FIM_MID] [S]
Target: [M]
```

**Corruption variants:**
- **PSM (Prefix-Suffix-Middle)**: Most common
- **SMP (Suffix-Middle-Prefix)**: Rearranged to prevent positional bias
- **Token-level FIM**: Apply FIM at the token level (more noise)

**Training ratio:** Typically 50% of training examples use FIM, 50% use standard AR. FIM examples use a special sentinel token to indicate where the gap is.

**Impact:** FIM is standard in virtually all code LLMs (Codex, CodeGen, StarCoder, Code Llama, DeepSeek Coder). It significantly improves code infilling performance while maintaining or slightly improving standard generation.

### 1.7 Comparison of Pre-training Objectives

| Objective | Masking | Attention | Loss Tokens | Strengths | Weaknesses |
|---|---|---|---|---|---|
| Autoregressive | None | Causal | All tokens | Generative, simple, scalable | Unidirectional context |
| MLM (BERT) | 15% | Bidirectional | 15% of tokens | Strong bidir representations | Not generative, inefficient |
| Prefix LM | None on prefix | Prefix: full; Suffix: causal | Suffix tokens | Balanced encoding/generation | More complex masking |
| Span Corruption | ~15% spans | Bidirectional (enc) | Corrupted spans | Efficient, generative | Encoder-decoder complexity |
| UL2 mix | Variable | Mode-dependent | Variable | All-rounder | Complex training pipeline |
| FIM | 50% examples | Causal | Middle section | Code infilling | Domain-specific benefit |

### 1.8 Consistency and Positional Embedding Notes

Different pre-training objectives often interact with positional embeddings:
- **ALiBi** (Press et al., 2022): Linear attention bias — works best with AR objectives. Used in BLOOM, MPT
- **RoPE** (Su et al., 2021): Rotary position embeddings — works with AR and prefix LM. Used in Llama, Mistral, PaLM, Gemma
- **Absolute PE**: Traditional sinusoidal or learned — used in BERT, GPT-2, T5

RoPE is the dominant choice in modern LLMs due to its natural extrapolation to longer sequences and compatibility with relative position computation.

---

## 2. Training Data Pipeline

### 2.1 Overview

The training data pipeline for large language models is a critical factor in final model quality. The pipeline typically consists of:

```
Internet → Crawling → Filtering → Deduplication
  → Quality Rating → Data Mixing → Tokenization
  → Sharding → Training
```

Each stage involves significant engineering and research challenges. We cover each in depth below.

### 2.2 Web Crawling

**CommonCrawl** is the primary source for most LLM training datasets. It provides raw web page data collected at scale (monthly snapshots of tens of petabytes).

**Crawling strategies:**
- **Broad crawl**: Capture as much of the web as possible (CommonCrawl)
- **Focused crawl**: Target specific domains or types of content (e.g., academic papers, code repositories)
- **Incremental crawl**: Continuously update, replacing stale pages
- **Custom crawlers**: Some organizations build their own (Google, OpenAI, Anthropic, Meta)

**WARC format:** Crawled data is stored in WARC (Web ARChive) files containing:
- WARC headers (URL, date, IP, content type)
- Raw HTTP response (HTML, headers, metadata)
- Metadata about the crawl

**Wet vs Wat vs Warc:**
- **WARC**: Full raw data including headers
- **WAT**: Metadata only (links, text metadata — lighter)
- **WET**: Extracted plain text only (no HTML) — most common for LLM training

### 2.3 HTML Extraction and Text Processing

Raw HTML must be converted to clean text:

**Tools:**
- **trafilatura** (Barbaresi, 2021): State-of-the-art HTML-to-text extraction. Used in FineWeb, DCLM, RedPajama
- **jusText**: Rule-based boilerplate removal
- **readability-lxml**: Mozilla Readability port — article extraction
- **html2text**: Simple HTML→Markdown conversion
- **Newspaper3k**: Article extraction with metadata

**Pipeline:**
```python
def extract_text(html, url):
    # 1. Parse HTML
    doc = trafilatura.load_html(html)
    
    # 2. Extract main content
    text = trafilatura.extract(doc, 
        output_format='txt',
        include_comments=False,
        include_tables=True,
        no_fallback=False)
    
    # 3. Post-processing
    text = normalize_whitespace(text)
    text = remove_excess_newlines(text)
    
    return text
```

**Heuristics for low-quality pages:**
- Page with < 200 characters of extracted text
- Ratio of extracted text to raw HTML < 0.1 (too much markup, likely spam)
- Contains excessive punctuation or ALL CAPS
- Missing closing HTML tags
- Non-standard character sets

### 2.4 Language Filtering

Identify and remove non-target-language documents:

- **FastText language classifier** (Joulin et al., 2017): Trained on Wikipedia data, covers 157 languages. This is the standard tool used by most LLM projects.
- **CLD2/CLD3 (Compact Language Detector)**: Chromium's built-in detector
- **Human validation**: Sample documents to verify language classification accuracy

**Implementation pattern:**
```python
from fasttext import load_model

lang_model = load_model('lid.176.bin')

def classify_language(text, threshold=0.5):
    predictions = lang_model.predict(text)
    lang = predictions[0][0].replace('__label__', '')
    confidence = predictions[1][0]
    return lang, confidence

# Keep only target language documents
if lang == target_lang and confidence >= threshold:
    keep_document()
```

**Data leakage across languages:** Even when filtering for one language, multilingual documents may contain code-switching. Some pipelines include a "multilingual" category.

### 2.5 Heuristic Quality Filtering

Simple rule-based filters that remove low-quality documents. Based on analysis from The Pile (Gao et al., 2020), C4 (Raffel et al., 2020), and subsequent work.

**Common heuristics:**

```python
def heuristic_filter(text, stats):
    checks = []
    
    # 1. Minimum length
    checks.append(len(text) >= 200)  # at least 200 chars
    
    # 2. Remove placeholder/lorem ipsum pages
    placeholder_phrases = ['lorem ipsum', 'this is a sample', 'under construction']
    checks.append(not any(p in text.lower() for p in placeholder_phrases))
    
    # 3. Punctuation ratio (high = likely bad)
    punct = sum(c in string.punctuation for c in text)
    punct_ratio = punct / max(1, len(text))
    checks.append(punct_ratio < 0.3)
    
    # 4. Upper case ratio
    upper = sum(c.isupper() for c in text if c.isalpha())
    upper_ratio = upper / max(1, sum(c.isalpha() for c in text))
    checks.append(upper_ratio < 0.4)  # SHOUTING pages
    
    # 5. Stop word ratio
    stop_words = {'the', 'a', 'an', 'in', 'of', 'to', 'and', 'is', 'it', 'that'}
    words = text.lower().split()
    stop_ratio = sum(w in stop_words for w in words) / max(1, len(words))
    checks.append(stop_ratio > 0.01)  # not gibberish
    
    # 6. Bullet point ratio (too many lists = likely navigation)
    bullet_count = text.count('*') + text.count('-')
    checks.append(bullet_count / len(text) < 0.1)
    
    # 7. Symbol-to-word ratio
    symbols = sum(c in '#@💙💪😂😍' for c in text)
    checks.append(symbols / len(text.split()) < 0.3)
    
    return all(checks)
```

**The RedPajama heuristics** (Together Computer, 2023): Include 7 quality filters including:
- A histogram of character classes (digits, punctuation, uppercase, lowercase, spaces)
- A "word count" ≥ 50 after extraction
- Mean word length between 3 and 10
- At least 90% of lines end with a proper line-ending character

**FineWeb heuristics** (Penedo et al., 2024): A refined set including:
- Minimum 128 tokens after tokenization
- Maximum perplexity threshold (using a reference model)
- Minimum stop word ratio
- Remove pages with repeated lines (n-gram dedup at line level)

### 2.6 Classifier-Based Quality Filtering

Train a classifier to distinguish "high quality" from "low quality" text:

**Approach 1: Binary classifier (high-quality vs. random web)**

```
1. Collect positive examples: Wikipedia, books, arXiv papers, curated articles
2. Collect negative examples: random CommonCrawl samples (or known low-quality)
3. Train a fastText / linear classifier / BERT classifier
4. Score all documents and keep top k% or above a threshold
```

**Approach 2: Perplexity-based scoring**

Use a well-trained language model (e.g., a previously trained model or a small reference model like KenLM) to score documents:

```python
def perplexity_score(text, model):
    # Tokenize and compute loss
    tokens = tokenizer.encode(text)
    with torch.no_grad():
        loss = model(tokens, labels=tokens).loss
    ppl = torch.exp(loss).item()
    return ppl

# Typical thresholds (depends on domain):
# Wikipedia: ~50-80
# Reddit: ~80-120
# Gibberish: > 500
# Very high quality: < 40
```

**Approach 3: Domain-based rating (DCLM method)**

DCLM (Li et al., 2024) used a classifier trained on curriculum learning samples: they trained a small model (300M-1B params) on a sample of the data, and used the model's learning dynamics (loss on held-out validation sets) to identify high-value documents.

**Classifier score aggregation:**
```python
# FastText-based quality classifier
def train_quality_classifier():
    model = fasttext.train_supervised(
        input='quality_data.train',
        lr=0.1,
        epoch=50,
        wordNgrams=2,
        dim=300,
        loss='softmax'
    )
    return model

def score_document(text, classifier_model):
    # Replace newlines as fastText cannot handle them
    text_clean = text.replace('\n', ' ')
    prediction = classifier_model.predict(text_clean)
    quality_score = prediction[1][0]  # probability of 'high' class
    return quality_score
```

### 2.7 Toxicity and Harmful Content Filtering

Remove toxic, NSFW, hate speech, and personally identifiable information (PII):

- **Detoxicity models**: Jigsaw Perspective API (toxicity, severe toxicity, identity attack, insult, threat, profanity, sexually explicit)
- **PII removal**: Regex patterns for emails, phone numbers, SSNs, credit card numbers, IP addresses
- **NSFW image detection**: For multimodal datasets, detect adult content
- **Domain blacklists**: Block known adult domains, spam domains
- **Keyword filtering**: Remove or flag pages with slurs, hate speech terms

**Important tradeoff:** Overly aggressive filtering removes valid LGBTQ+ content, health information, and legitimate discussions. Modern approaches use **nuanced filtering** that preserves minority group content while removing truly hateful material.

---

## 3. Data Deduplication and Filtering

### 3.1 Exact Deduplication

Remove identical documents (byte-for-byte or hash-for-hash):

**Approach:** Compute a hash (SHA-256, MD5, xxHash) of each document and remove duplicates:

```python
def exact_dedup(documents):
    seen = set()
    unique = []
    for doc in documents:
        # Hash the normalized text
        text_normalized = doc.strip().lower()
        doc_hash = hashlib.sha256(text_normalized.encode()).hexdigest()
        if doc_hash not in seen:
            seen.add(doc_hash)
            unique.append(doc)
    return unique
```

**Normalization:** Simple normalization can catch more duplicates:
- Collapse whitespace
- Lowercase (optional)
- Remove HTML tags
- Normalize Unicode (NFC normalization)

**Scaling:** For datasets of billions of documents, use:
- **Spark/MapReduce**: Partition by hash prefix, deduplicate within each partition
- **RocksDB/LevelDB**: On-disk key-value stores for hash → doc mapping
- **Bloom filters**: Probabilistic dedup — fast but can miss some duplicates

### 3.2 MinHash for Near-Duplicate Detection

MinHash (Broder, 1997) is the standard algorithm for finding near-duplicate documents (e.g., articles from different news sources covering the same story, or documents that differ only in boilerplate).

**Algorithm:**

1. **Tokenization**: Convert document to set of shingles (n-grams at the word level, typically k=5-13)

2. **Signature computation**: Use k hash functions to create a signature vector:

```python
def minhash_signature(doc, num_hashes=256, shingle_size=5):
    # Generate shingles
    shingles = set()
    words = doc.split()
    for i in range(len(words) - shingle_size + 1):
        shingle = ' '.join(words[i:i+shingle_size])
        shingles.add(shingle)
    
    # Compute signature
    signature = [float('inf')] * num_hashes
    for shingle in shingles:
        for i in range(num_hashes):
            # Use a hash function (typically random linear hash)
            h = hash_shingle(shingle, i)
            signature[i] = min(signature[i], h)
    
    return signature
```

3. **Band-based LSH (Locality Sensitive Hashing)**: Split the signature into `b` bands of `r` rows each. Two documents are candidate near-duplicates if any band hash matches:

```python
def lsh_bands(signature, num_bands=16):
    """Split signature into bands and hash each band."""
    rows_per_band = len(signature) // num_bands
    band_hashes = []
    for b in range(num_bands):
        band = signature[b * rows_per_band : (b + 1) * rows_per_band]
        band_hash = hash(tuple(band))
        band_hashes.append(band_hash)
    return band_hashes
```

**Probability of detection (Jaccard similarity s):**

```
P(detection | s) = 1 - (1 - s^r)^b
```

Where:
- `s` = Jaccard similarity of the two documents
- `r` = rows per band
- `b` = number of bands

For `b=16, r=4`: `P ≈ 1 - (1 - s^4)^16`
- `s=0.5`: P ≈ 0.53
- `s=0.7`: P ≈ 0.96
- `s=0.9`: P ≈ 1.00

**Threshold estimation:**

```
s_threshold ≈ (1 / b)^(1/r)
```

For b=16, r=4: s_threshold ≈ (1/16)^(1/4) = 0.5

Documents with Jaccard similarity > ~0.5 are likely detected.

**Practical concerns:**
- **Shingle size**: k=5 words is standard for English. Too small: false positives. Too large: missed near-duplicates.
- **Number of hashes**: 256 is typical. More hashes → higher accuracy but more memory/time.
- **Scaling**: Tools like DEDUP (Penedo et al., 2024) using Spark can process billions of documents.
- **Memory**: For 10B documents with 256 hashes: 10B × 256 × 8 bytes = 20.5 TB — must use distributed LSH.

### 3.3 SimHash for Near-Duplicate Detection

SimHash (Charikar, 2002) is an alternative to MinHash that works by computing a **fingerprint** from a weighted sum of hash values.

**Algorithm:**
```python
def simhash_fingerprint(doc, num_bits=64):
    """Compute SimHash fingerprint for a document."""
    v = [0] * num_bits
    words = doc.split()
    
    for word in set(words):  # or use tf-idf weighting
        # Word hash
        h = hash_words(word)  # returns 64-bit integer
        
        # tf-weight (number of occurrences)
        weight = words.count(word)
        
        # Accumulate bits
        for i in range(num_bits):
            if (h >> i) & 1:
                v[i] += weight
            else:
                v[i] -= weight
    
    # Compress to fingerprint
    fingerprint = 0
    for i in range(num_bits):
        if v[i] > 0:
            fingerprint |= (1 << i)
    
    return fingerprint
```

**Similarity detection:** Two documents are near-duplicates if their SimHash fingerprints have a **Hamming distance** ≤ threshold (e.g., 3 for 64-bit fingerprints).

**Pros vs MinHash:**
- **SimHash**: Faster (single pass), more memory efficient, good for short documents
- **MinHash**: More flexible (configurable threshold), better for varying document lengths, standard in large-scale datasets

Both are used in major LLM datasets. MinHash is the more common choice (C4, The Pile, RedPajama, FineWeb, DCLM).

### 3.4 URL-Level Deduplication

Remove duplicate URLs that point to the same content:

```python
def url_normalize(url):
    # Remove fragments
    url = url.split('#')[0]
    # Lowercase scheme and host
    parsed = urlparse(url)
    normalized = f"{parsed.scheme.lower()}://{parsed.hostname.lower()}{parsed.path}"
    # Remove trailing slash
    normalized = normalized.rstrip('/')
    # Remove www prefix
    normalized = normalized.replace('://www.', '://')
    return normalized

def url_dedup(documents):
    seen_urls = set()
    for doc in documents:
        url_norm = url_normalize(doc.url)
        if url_norm not in seen_urls:
            seen_urls.add(url_norm)
            yield doc
```

### 3.5 Line-Level Dedup and Repetition Removal

Documents with repeated lines, boilerplate, or templated content degrade model quality and can cause training divergence.

**Approaches:**

```python
def remove_repeated_lines(text, threshold=3):
    """Remove lines that appear more than threshold times."""
    lines = text.split('\n')
    line_counts = Counter(lines)
    filtered = [l for l in lines if line_counts[l] <= threshold]
    return '\n'.join(filtered)

def remove_repeated_ngrams(text, n=50, threshold=10):
    """Remove documents where any n-gram repeats excessively."""
    tokens = text.split()
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    ngram_counts = Counter(ngrams)
    max_repeat = max(ngram_counts.values())
    return max_repeat <= threshold
```

**FineWeb's approach:** Repetition at the line level is computed using character-level de-duplication of lines after normalization. Lines appearing more than once are removed.

### 3.6 Perplexity-Based Filtering

Use a reference language model to score documents and filter out low-quality (high perplexity) content:

```python
def filter_by_perplexity(texts, model, tokenizer, batch_size=64):
    """Filter documents by perplexity using a reference model."""
    keep = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        # Tokenize
        encodings = tokenizer(batch, return_tensors='pt', 
                             padding=True, truncation=True, 
                             max_length=512)
        with torch.no_grad():
            outputs = model(**encodings, labels=encodings['input_ids'])
            losses = outputs.loss  # per-example loss
            ppls = torch.exp(losses).cpu().numpy()
        
        for j, ppl in enumerate(ppls):
            # Typical range: Wikipedia ~50-80, spam ~500+
            if 20 <= ppl <= 300:
                keep.append(batch[j])
    
    return keep
```

**Which model to use for scoring:**
- **KenLM** (Kenny Language Model): 5-gram model trained on Wikipedia — fast, lightweight
- **Small Transformer**: 125M-350M param model trained on high-quality data — better than n-gram
- **Reference model**: If you have a previous version of your target model, use it for consistency

**Practical note:** Perplexity-based filtering is a powerful signal but has domain effects: scientific papers have lower perplexity than creative writing. Multi-domain filtering should use domain-specific thresholds.

### 3.7 Domain-Level Filtering

Blacklist or whitelist specific domains:

```python
# CommonCrawl domain heuristics
BLACKLISTED_DOMAINS = {
    'googlevideo.com',  # video
    'youtube.com',      # has own data
    'facebook.com',     # social media
    'instagram.com',    # primarily image
    'pinterest.com',    # primarily image
    '4chan.org',        # toxic content
    '8kun.top',         # toxic content
    'twitter.com',      # has own data
    'x.com',            # has own data
}

HIGH_QUALITY_DOMAINS = {
    'wikipedia.org',        # Encyclopedia
    'arxiv.org',            # Scientific papers
    'github.com',           # Code
    'plos.org',             # Open access science
    'nature.com',           # Scientific journal
    'science.org',          # Scientific journal
    'stat.gov',             # Government statistics
    'nasa.gov',             # Government science
    'loc.gov',              # Library of Congress
    'gutenberg.org',        # Public domain books
    'aclweb.org',           # Computational linguistics
    'pml.org',              # Programming languages
}

# Domain scoring: each page from high-quality domains gets a boost
# or can bypass some filtering steps
```

**Practical guidance:** Domain-level filtering is simple and effective but can miss nuance. A single low-quality page from a high-quality domain (e.g., a forum post on StackOverflow) may be useful while a highly-formatted page from a generally low-quality domain may not.

### 3.8 Complete Data Pipeline Pseudocode

```python
ALGORITHM: Complete LLM Data Pipeline
─────────────────────────────────────
def build_training_dataset(crawl_path='cc_data/', output_path='train_data/'):
    
    # Stage 1: Load raw data (CommonCrawl WET files)
    docs = load_wet_files(crawl_path)
    
    # Stage 2: HTML extraction
    docs = [extract_text(doc.html, doc.url) for doc in docs]
    
    # Stage 3: Language identification
    docs = [d for d in docs if classify_language(d.text) == 'en']
    
    # Stage 4: Heuristic quality filtering
    docs = [d for d in docs if pass_heuristic_quality(d.text)]
    
    # Stage 5: MinHash near-dedup
    signatures = [minhash_signature(d.text) for d in docs]
    clusters = lsh_clustering(signatures, threshold=0.8)
    docs = remove_cluster_duplicates(docs, clusters)
    
    # Stage 6: Line-level repetition removal
    docs = [d.map(remove_repeated_lines) for d in docs]
    
    # Stage 7: Domain filtering and boosting
    docs = [d for d in docs if d.domain not in BLACKLIST]
    
    # Stage 8: Perplexity-based filtering
    ppls = compute_perplexity_batch(docs, reference_model)
    docs = [d for d, p in zip(docs, ppls) if 20 <= p <= 300]
    
    # Stage 9: Toxicity filtering
    docs = [d for d in docs if toxicity_score(d.text) < 0.5]
    
    # Stage 10: PII removal
    docs = [remove_pii(d) for d in docs]
    
    # Stage 11: Data mixing (see Section 4)
    docs = apply_mixing_strategy(docs)
    
    # Stage 12: Tokenization and sharding
    tokens = [tokenize(d.text) for d in docs]
    shard(tokens, output_path)
    
    return stats(docs)
```

---

## 4. Data Mixing Strategies

### 4.1 The Data Mixing Problem

Given multiple data sources (Wikipedia, books, code, web, social media, scientific papers), how should they be mixed for optimal downstream performance?

**Key insight:** The optimal data distribution is **not** proportional to data source size, availability, or perceived "quality." It must be empirically determined through ablations.

### 4.2 Mixing Strategies

**1. Uniform sampling:** Sample equal number of tokens from each domain. Simple but wasteful (smaller domains get oversampled, larger domains undersampled).

```python
# Uniform mixing
domains = ['wikipedia', 'books', 'web', 'code']
sample_ratio = 1.0 / len(domains)  # 0.25 each
```

**2. Proportional sampling:** Sample in proportion to dataset size. Most common baseline.

```python
# Proportional to size
total_size = sum(len(d) for d in datasets)
proportions = [len(d) / total_size for d in datasets]
```

**3. Quality-weighted mixing:** Weight domains by quality score.

```python
# Quality weighting (simplified)
QUALITY_WEIGHTS = {
    'wikipedia': 3.0,
    'books': 2.5,
    'arxiv': 2.0,
    'code': 1.5,
    'web': 0.5,
}
effective_weight = {d: size * QUALITY_WEIGHTS.get(d, 1.0) for d in domains}
total_eff = sum(effective_weight.values())
proportions = {d: w / total_eff for d, w in effective_weight.items()}
```

**4. Perplexity-based mixing (DoReMi)** (Xie et al., 2023):

DoReMi (Domain Reweighting with Minimax optimization) automatically determines domain weights by minimizing the worst-case loss across domains.

```python
# DoReMi: Minimax weight optimization
# 1. Train small reference model (300M params) on uniform data
# 2. For each domain, compute loss gap vs reference model
# 3. Update domain weights to minimize worst-case gap
# 4. Use optimal weights to train the large model

def doremi_update(domain_losses, reference_losses, weights, lr=0.01):
    """One step of DoReMi weight update."""
    # Compute excess loss per domain
    excess = domain_losses - reference_losses
    
    # Update weights using gradient descent on worst-case excess
    # Soft-minimax: update weights to emphasize high-excess domains
    weights = weights * (1 + lr * torch.exp(excess / temperature))
    weights = weights / weights.sum()  # normalize
    return weights
```

**5. Temperature-based smoothing:** Smooth proportions to prevent over-dominance of large domains:

```python
def smooth_proportions(size_dict, temperature=1.0):
    """Apply temperature to smooth domain proportions."""
    # Exponential scaling
    scaled = {k: v ** temperature for k, v in size_dict.items()}
    total = sum(scaled.values())
    return {k: v / total for k, v in scaled.items()}
```

Temperature < 1 → more uniform (promotes diversity)
Temperature > 1 → more extreme (promotes biggest domains)

**6. Dynamic mixing (annealing):** Change data mix over training course.

```python
def dynamic_mix_annealing(step, total_steps, domains):
    """Annealed data mixing.
    Early: more high-quality data for learning
    Late: more diverse web data for generalization
    """
    early_mix = {'wikipedia': 0.3, 'books': 0.3, 'code': 0.2, 'web': 0.2}
    late_mix = {'wikipedia': 0.1, 'books': 0.1, 'code': 0.1, 'web': 0.7}
    
    fraction = step / total_steps  # 0 to 1
    mix = {}
    for domain in domains:
        mix[domain] = (1 - fraction) * early_mix[domain] + fraction * late_mix[domain]
    return mix
```

### 4.3 Effect of Data Mix on Model Capabilities

Empirical findings from major LLM projects:

| Domain | Improves | Can Harm |
|---|---|---|
| Wikipedia | Factual accuracy, QA | Creative writing (overly formal) |
| Books | Narrative, long-form coherence | Token efficiency (too many tokens per fact) |
| Code | Reasoning, structured output, math | Fluency in prose |
| Scientific papers | Domain knowledge, formal reasoning | General conversation |
| Web (C4, FineWeb) | General knowledge, diversity | Factual accuracy, coherence |
| Social media | Dialogue, conversational text | Toxicity, redundancy |

**Key example:** Adding 1% code data to a language model significantly improves reasoning benchmarks (MATH, GSM8K) without degrading language performance. This was demonstrated by DeepSeek Code and Code Llama.

**Practical recommendation for data mixing:**

```
Typical LLM mix (undisclosed in most papers but approximately):
- Web (C4/FineWeb): 60-80%
- Books: 10-15%
- Wikipedia: 3-5%
- Code (GitHub): 3-10%
- Academic papers (arXiv/Pubmed): 3-5%
- Other (Reddit, forums, news): 5-10%
```

---

## 5. Scaling Laws

### 5.1 The Kaplan Scaling Law

Kaplan et al. (2020, also called "Scaling Laws for Neural Language Models") established the foundational scaling relationships for Transformer language models.

**Empirical power laws:**

The test loss `L(N, D)` as a function of model parameters N and dataset tokens D follows a power law:

```
L(N) ≈ (N_c / N)^{α_N}    for fixed D (compute-bounded)
L(D) ≈ (D_c / D)^{α_D}    for fixed N (data-bounded)
```

Where:
- `α_N ≈ 0.076` (parameter scaling exponent)
- `α_D ≈ 0.095` (data scaling exponent)
- `N_c, D_c` are constants (compute-bounded scaling)

**Key finding 1: Large models are more token-efficient.**
Given a fixed compute budget, larger models with fewer training steps (fewer tokens) achieve lower loss than smaller models trained on more tokens.

**Key finding 2: Compute-optimal frontier.**
For a given compute budget `C` (FLOPs), the optimal allocation is:

```
N_opt ∝ C^{0.73}
D_opt ∝ C^{0.27}
```

Where `N_opt` is the optimal model size and `D_opt` is the optimal number of training tokens.

**Key finding 3: Transfer follows scaling.**
Fine-tuning performance on downstream tasks also scales with pre-training compute, though with a larger exponent (bigger models benefit more from fine-tuning).

**Key finding 4: Smooth predictable scaling.**
Loss decreases predictably as power law — no "phase changes" where increasing scale suddenly changes performance trajectory for in-distribution evaluation.

**The famous scaling equation (compute budget C):**

```
L(N, D) = [(N_c / N)^{α_N / α_D} + (D_c / D)]^{α_D}
```

### 5.2 The Chinchilla Scaling Law

Hoffmann et al. (2022, "Training Compute-Optimal Large Language Models") re-examined the scaling law and found that the Kaplan law **overestimated the value of model parameters** — compute-optimal models are actually smaller, trained on more data.

**Chinchilla methodology:**

For each model size {70M, 100M, 300M, 500M, 1B, 3B, 6B}, they trained the model on varying amounts of data and measured the final loss. They fit a parametric model:

```
L(N, D) = E + A / N^{α} + B / D^{β}
```

Where:
- `E` = irreducible loss (entropy of text)
- `A / N^{α}` = loss from parameter-limited capacity
- `B / D^{β}` = loss from data-limited capacity

**Three approaches to find compute-optimal boundary:**

**Approach 1: IsoFLOP profiles (most commonly cited).**
Train models at multiple sizes and data volumes, group results by FLOP budget, find the optimal point for each budget.

```
For a given FLOP budget C:
    N_opt(C) ∝ C^{0.5}
    D_opt(C) ∝ C^{0.5}
```

This means: **proportional scaling** — doubling compute budget should double both model parameters and training tokens.

**Approach 2: Parametric fitting.** Fit the parametric loss function L(N, D) = E + A/N^α + B/D^β.

Chinchilla fitted parameters: `α ≈ 0.34`, `β ≈ 0.28`, `E ≈ 1.69`, `A ≈ 406.4`, `B ≈ 410.7`.

**Approach 3: Test fit on a fixed FLOP frontier.** Train models at different (N, D) combinations with the same FLOP budget, find minimum loss.

**Chinchilla recommendation:**
```
N_opt = C / (6 × 1.92) ≈ C / 11.52  (for Chinchilla, N_opt=70B, D_opt=1.4T)
```

Where 6 is the FLOPs per token per parameter (forward + backward pass), and 1.92 is the optimal tokens-per-parameter ratio.

**Compare Kaplan vs Chinchilla:**

| Aspect | Kaplan 2020 | Chinchilla 2022 |
|---|---|---|
| Optimal tokens/param | ~30:1 (300B params, 300B tokens) | ~20:1 (70B params, 1.4T tokens) |
| Param scaling exponent | α_N ≈ 0.076 | α ≈ 0.34 (in loss function) |
| Model compute scaling | N_opt ∝ C^{0.73} | N_opt ∝ C^{0.5} |
| Data compute scaling | D_opt ∝ C^{0.27} | D_opt ∝ C^{0.5} |
| Implication | Bigger models, fewer tokens | Bigger models AND more tokens |

**Practical impact of Chinchilla:**

The Chinchilla paper fundamentally changed LLM training. Post-2022 models are trained on significantly more tokens relative to their size:
- Llama 1 (6.7B → 1T tokens, ratio ~150:1)
- Llama 2 (7B → 2T tokens, ratio ~286:1)
- Llama 3 (8B → 15T tokens, ratio ~1875:1)
- Mistral 7B → ~9T tokens, ratio ~1286:1
- DeepSeek V3 (671B MoE) → 14.8T tokens

Modern models far exceed the Chinchilla-optimal ratio, suggesting either:
1. Scaling exponents are steeper than Chinchilla assumed
2. Data quality (not just quantity) improves with more tokens
3. Emergent capabilities benefit from more data beyond the loss-scaling optimum

### 5.3 IsoFLOP Profiles

The IsoFLOP method is the most direct way to determine compute-optimal training:

```python
def isoflop_profiles(model_sizes, token_counts, flop_budget):
    """Find optimal (N, D) for a given FLOP budget."""
    results = []
    
    for N in model_sizes:
        for D in token_counts:
            # FLOPs for training with transformers:
            # ~6 * N * D FLOPs (forward + backward)
            F = 6 * N * D
            
            if abs(F - flop_budget) / flop_budget > 0.1:
                continue  # Skip non-matching combos
            
            # Train and measure loss
            model = Transformer(..., d_model=..., n_layers=...)
            loss = train(model, n_tokens=D)
            
            results.append({'N': N, 'D': D, 'F': F, 'loss': loss})
    
    # Find minimum loss at this FLOP budget
    best = min(results, key=lambda x: x['loss'])
    return best['N'], best['D']
```

**Practical guidance for IsoFLOP experiments:**

1. Train models at 4-6 different sizes (e.g., 70M, 150M, 300M, 500M, 1B, 3B)
2. For each size, train at 3-5 different token counts (e.g., 5B, 10B, 20B, 50B, 100B tokens)
3. Plot loss vs FLOP budget curves
4. Find the Pareto-optimal frontier
5. Extrapolate to larger compute budgets using power-law fits

### 5.4 Data Scaling vs Model Scaling

**The core question:** Given additional compute budget, should you increase model size or training data?

**Kaplan answer (2020):** Increase model size. At compute-optimality, 73% of additional compute goes to model size, 27% to data.

**Chinchilla answer (2022):** Increase both equally. 50%/50% split.

**Modern understanding (2024+):** The answer depends on:

1. **Data quality**: With high-quality data, data scaling provides more benefit per token. Low-quality data may require more model capacity to "filter" noise.

2. **Model architecture**: MoE models (Mixtral, DeepSeek-V3) decouple total params from active params — they benefit more from data scaling as capacity is cheap.

3. **Training regime**: 
   - **Pre-training scale** (10B-30T tokens): Chinchilla regime holds reasonably
   - **Continued pre-training / domain adaptation**: Data scaling dominates (small model on domain data beats large model on less domain data)
   - **Fine-tuning**: More data almost always helps regardless of model size

4. **Diminishing returns**: Both model and data scaling show diminishing returns. The marginal benefit of 1% more data = 1% more params varies.

```
// Modern intuition for data vs model scaling returns:
// For 8B-70B models:
//   Doubling data: loss reduction ≈ 0.05-0.10 nats
//   Doubling model size: loss reduction ≈ 0.10-0.15 nats
// Ratio: model scaling ~1.5-2× more effective per FLOP
// But: data can be parallelized across GPUs; model scaling requires more memory
```

### 5.5 Scaling Law Breakdown and Limitations

**Known breakdowns of scaling laws:**

1. **Small model → large model extrapolation**: Power laws fitted on models up to 1B params may not hold at 100B+ due to:
   - Emerging capabilities (few-shot reasoning, in-context learning)
   - Variance in training dynamics (larger models train differently)
   - Architectural constraints (attention head count, vocabulary size)

2. **Compute-extrapolation**: The functional form `L = E + A/N^α + B/D^β` assumes smooth monotonic improvement. In practice, there can be:
   - **Threshold effects**: Certain capabilities (arithmetic, multi-step reasoning) emerge only past a critical scale
   - **Plateauing**: Some benchmarks saturate (models achieve >95% on some tasks)
   - **Capability scaling is non-uniform**: Some tasks improve smoothly with scale, others show sharp transitions

3. **Data quality matters**: Scaling laws are derived from IID data. With data selection:
   - High-quality data can achieve same loss with 2-10× fewer tokens
   - The "effective data size" depends on quality, not just quantity

4. **Training compute ≠ inference compute**: Optimal pre-training models may be suboptimal for deployment. Smaller models can be distilled or pruned for inference efficiency.

### 5.6 Beyond Scaling Laws: Recent Understanding

**The "Three Scaling Regimes" (Michaud et al., 2023):**

1. **Under-trained regime** (D < N^0.5): Model capacity underutilized. More data helps most.
2. **Compute-optimal regime** (D ≈ N^0.5): Balanced allocation.
3. **Over-trained regime** (D > N^0.5): Limited by model capacity. Larger model would help.

Modern models (Llama 3 8B on 15T tokens) are in the over-trained regime — they benefit more from architectural improvements or increased model size than from more data of the same quality.

**The "Data Scaling Wall":**

There is concern that the web's natural language data is finite and will be exhausted by 2026-2030 (Villalobos et al., 2022). This has driven:
- Synthetic data generation (see Section 15)
- Multi-turn augmentation
- Code and structured data sources
- Multilingual data collection
- Proprietary data licensing

**Does scaling always work? (Schaeffer et al., 2024):**

"E=mc²" (Emergent magnitudes = claims, counter-claims) — the notion that capabilities "emerge" abruptly at certain scales has been challenged as an artifact of discontinuous evaluation metrics. Continuous metrics (loss, BLEU, accuracy on fine-grained rubrics) show gradual improvement with scale.

### 5.7 Practical Scaling Guidelines (2024+)

```
Model Size    → Recommended Training Tokens (current best practice)
──────────────────────────────────────────────────────
1B params     → 500B - 2T tokens
3B params     → 1T - 5T tokens
7B params     → 2T - 15T tokens
13B params    → 3T - 20T tokens
34B params    → 5T - 30T tokens
70B params    → 10T - 40T tokens
130B params   → 15T - 60T tokens
300B params   → 30T - 100T tokens (MoE often used)
1T+ params    → 50T+ tokens (mostly MoE)
```

These ratios exceed Chinchilla by 5-100×, reflecting the focus on data quality and the value of extended training for emergent capabilities.

---

## 6. Curriculum Learning

### 6.1 Core Idea

Curriculum learning (Bengio et al., 2009) organizes training data from "easy" to "hard," mimicking how humans learn. The hypothesis is that this ordering accelerates convergence and improves final performance.

**Mathematical formulation:**

Given a set of training examples `{(x_i, y_i)}` ordered by difficulty `d_i`, the curriculum presents them in increasing difficulty order:

```
L_curriculum(θ) = Σ_{t=1}^{T} w_t · L(θ; (x_{π(t)}, y_{π(t)}))
```

Where `π` is a permutation sorted by difficulty `d_i`, and `w_t` is a time-dependent weight (typically 1 for early examples in easy phase, switching to include hard examples later).

### 6.2 Difficulty Metrics

**Data-driven difficulty:**
- **Loss-based**: Loss of a pre-trained model on each example
- **Perplexity-based**: PPL of a small reference model
- **Gradient magnitude**: Norm of the gradient on each example (high gradient = harder)
- **Language model probability**: `-log p(x_i)`

**Structural difficulty:**
- **Sequence length**: Shorter sequences first
- **Vocabulary complexity**: Rare words count
- **Syntactic complexity**: Parse tree depth
- **Readability scores**: Flesch-Kincaid, Dale-Chall

**Task-specific difficulty:**
- **For math**: Number of reasoning steps, equation length
- **For code**: Lines of code, cyclomatic complexity
- **For translation**: BLEU score against reference

### 6.3 Curriculum Schedules

**1. Pacing function:** Controls how quickly hard examples are introduced.

```
p(t) = min(1, (t/T)^k)  // k controls sharpness
```

- k=1 (linear): Gradually increase
- k<1 (concave): Quickly introduce hard examples
- k>1 (convex): Delay hard examples

**2. Competence-based curriculum:** Switch difficulty when model achieves a target accuracy:

```
if val_acc_on_easy > 0.8:
    mix_in_medium = 0.5
if val_acc_on_medium > 0.8:
    mix_in_hard = 0.5
```

**3. Self-paced learning** (Kumar et al., 2010): Model selects which examples to train on based on its current competence:

```
min_θ Σ_i v_i · L(θ; x_i, y_i) - λ · ||v||_1
s.t. v_i ∈ [0, 1]
```

Where `v_i` are learned example weights. Easy examples (low loss) get v_i=1; hard examples (high loss) get v_i=0 (skipped).

### 6.4 Curriculum Learning in LLM Training

**Current consensus:** Curriculum learning **does not significantly improve** large-scale LLM pre-training. The dominant practice is **random shuffling** of training data.

**However, curriculum is effective for:**

1. **Domain-adaptive pre-training**: Start with general web data, transition to domain-specific data (code, biology, finance)
2. **Multi-stage training**: 
   - Stage 1: Pre-training on filtered web
   - Stage 2: Continued pre-training on high-quality curated data
   - Stage 3: Instruction tuning
3. **Length extension**: Train on short sequences first, gradually increase context length (used by GPT-3, Llama 2: 2k → 4k → 8k → 32k)
4. **Math reasoning training**: Start with simple arithmetic, progress to multi-step problems

**Example: Multi-stage curriculum for math LLM:**
```
Stage 1: General pre-training (500B tokens, random order)
Stage 2: Math+Github code (100B tokens, random order)
Stage 3: Reasoning chain data (10B tokens, easy→hard)
Stage 4: Instruction tuning (1B tokens, random order)
```

### 6.5 Practical Implementation

```python
def curriculum_dataloader(dataset, difficulty_fn, batch_size, 
                          total_steps, schedule='linear'):
    """Curriculum learning dataloader."""
    # Compute difficulties for all examples
    difficulties = [difficulty_fn(x) for x in dataset]
    
    # Sort by difficulty
    sorted_idx = np.argsort(difficulties)
    sorted_data = [dataset[i] for i in sorted_idx]
    
    # Define pacing function
    def pacing(t):
        if schedule == 'linear':
            return min(1.0, t / (0.3 * total_steps))
        elif schedule == 'concave':
            return min(1.0, (t / total_steps) ** 0.5)
        elif schedule == 'convex':
            return min(1.0, (t / total_steps) ** 2)
    
    step = 0
    while step < total_steps:
        fraction = pacing(step)
        n_easy = int(fraction * len(sorted_data))
        
        # Sample from the easiest n_easy examples
        batch_indices = np.random.choice(n_easy, batch_size)
        yield [sorted_data[i] for i in batch_indices]
        step += 1
```

---

## 7. Multi-Task Learning

### 7.1 Multi-Task Learning Formulation

Multi-task learning (MTL) trains a single model on multiple tasks simultaneously, sharing representations across tasks. The goal is that learning one task improves performance on others (inductive transfer).

**Loss function for MTL:**

```
L_total(θ_shared, {θ_t}) = Σ_{t=1}^{T} λ_t · L_t(θ_shared, θ_t; D_t)
```

Where:
- `θ_shared` = shared parameters (e.g., Transformer backbone)
- `θ_t` = task-specific parameters (e.g., classification heads)
- `λ_t` = task weight
- `D_t` = dataset for task t

### 7.2 Task Weighting Strategies

**Uniform weighting:** `λ_t = 1/T` — simple but suboptimal when tasks have different scales or learning speeds.

**Uncertainty weighting** (Kendall et al., 2018): Learn task weights as the log variance of each task:

```
L_total = Σ_t (1 / 2σ_t²) · L_t + log σ_t
```

Tasks with high uncertainty (noise) are downweighted automatically.

**Dynamic prioritization** (Guo et al., 2018): Weight tasks by their performance gap:

```
λ_t ∝ 1 / (current_performance_t - target_performance_t + ε)
```

Tasks far from target get higher weight.

**Gradient balancing** (Chen et al., 2018): Adjust λ_t so all tasks have similar gradient magnitudes.

```
min_λ || Σ_t λ_t · g_t - g_avg ||_2
where g_t = ∇_θ L_t(θ)
```

### 7.3 Multi-Task Learning in LLM Context

**FLAN** (Wei et al., 2022) showed that instruction-tuning on many tasks simultaneously (massive multi-task learning) dramatically improves zero-shot generalization.

**Key insight:** Multi-task training on diverse NLP tasks (question answering, translation, summarization, reasoning) creates models that follow instructions more reliably than those trained on a single task.

**Task diversity considerations:**

```
Best MTL setups include:
- Diverse task types: classification, generation, extraction, reasoning
- Diverse domains: science, creative, business, technical
- Diverse output formats: multiple choice, free text, structured
- Diverse difficulty: simple to complex
```

---

## 8. Transfer Learning

### 8.1 The Transfer Learning Paradigm

Transfer learning leverages knowledge from a source domain/task to improve learning on a target domain/task.

**In deep learning / NLP, this is overwhelmingly:**

1. **Pre-training** on large general-domain data (source)
2. **Fine-tuning** on smaller task-specific data (target)

**Types of transfer:**
- **Inductive**: Source and target tasks differ but are related (e.g., pre-train on language modeling → fine-tune on classification)
- **Transductive**: Same task, different domains (e.g., pre-train on English → fine-tune on French)
- **Unsupervised pre-training**: Self-supervised pre-training → supervised fine-tuning

### 8.2 Pre-Training then Fine-Tuning

This is the dominant paradigm since BERT (2019). The typical sequence:

```
1. Pre-train on massive unlabeled data (web, books, code)
   - AR or MLM objective
   - Billions of tokens
   - High computational cost ($1M-$100M+)

2. Fine-tune on labeled downstream task data
   - Task-specific objective (classification, QA, generation)
   - Thousands to millions of labeled examples
   - Low computational cost
```

**Key insight:** Pre-training provides broad linguistic and world knowledge; fine-tuning adapts to specific task formats and domain distributions.

### 8.3 Linear Probing vs Fine-Tuning

**Linear probing:** Freeze pre-trained weights, train only a new classification head:

```
z = f_θ(x)       // frozen encoder
ŷ = W · z + b    // trainable linear classifier
L = cross_entropy(ŷ, y)
```

**Fine-tuning:** Update all pre-trained weights:

```
ŷ = g_φ(f_θ(x))   // both encoder f and head g are updated
L = cross_entropy(ŷ, y)
```

**Comparison:**
| Property | Linear Probing | Full Fine-Tuning |
|---|---|---|
| Parameters trained | Very few (only head) | All parameters |
| Compute cost | Low | Moderate |
| Overfitting risk | Low | Higher (especially small target data) |
| Task-specific adaptation | Limited (only last features) | Full adaptation |
| Feature preservation | Preserves pre-trained features | May overwrite features |
| Performance (data-rich) | Lower | Higher |
| Performance (data-poor) | Good | Can overfit |

**Probing vs fine-tuning gap:** The difference between linear probe and fine-tuned performance indicates how much the pre-trained features need to be transformed for the target task.

### 8.4 Progressive Transfer and Continual Pre-Training

**Progressive transfer**: Gradually reduce the gap between pre-training distribution and target distribution:

```
Stage 1: General pre-training (web)
Stage 2: Domain-adaptive pre-training (biomedical papers)
Stage 3: Task-adaptive pre-training (specific benchmark)
Stage 4: Fine-tuning (labeled task data)
```

**Pragmatic approach (used by most LLM developers):**

```
1. Pre-train base model on general web data (15T tokens)
2. Continue pre-training on curated high-quality data (500B tokens)
3. Multi-task instruction tuning (10M examples)
4. Alignment (RLHF/DPO) on preference data (10K-100K examples)
```

---

## 9. Fine-Tuning

### 9.1 Full Fine-Tuning

Full fine-tuning updates all model parameters on the target task.

**Formulation:**
```
θ* = argmin_θ L(θ; D_target)
    = argmin_θ Σ_{(x,y) ∈ D_target} ℓ(f_θ(x), y)
```

Where `θ` is initialized from pre-trained weights `θ_pretrain` and `ℓ` is the task loss (typically cross-entropy for language tasks).

**Empirical findings:**
- Fine-tuning with a small learning rate (1e-6 to 5e-5) is critical
- Full fine-tuning is expensive for large models (70B+ requires 8× A100s per node)
- Performance varies significantly with hyperparameters (LR, batch size, epochs)
- **Catastrophic forgetting**: Fine-tuning on a narrow task degrades general capabilities

**Mitigating catastrophic forgetting:**
- **Elastic Weight Consolidation (EWC)** (Kirkpatrick et al., 2017): Penalize changes to important parameters:
```
L = L_task + λ Σ_i F_i · (θ_i - θ_pretrain,i)²
```
Where `F_i` is the Fisher information matrix diagonal (importance of parameter i).

- **Replay / Experience replay**: Mix fine-tuning data with pre-training data:
```
L = L_task + α · L_pretrain(D_replay)
```

### 9.2 Hyperparameter Considerations for Fine-Tuning

```python
# Common fine-tuning hyperparameters (Llama/Mistral scale)

FINE_TUNING_CONFIG = {
    # Optimizer
    'optimizer': 'AdamW',
    'learning_rate': 2e-5,         # typical range: 1e-6 to 5e-5
    'weight_decay': 0.1,           # default
    'beta1': 0.9,
    'beta2': 0.999,
    'epsilon': 1e-8,
    
    # LR schedule
    'lr_scheduler': 'cosine',
    'warmup_ratio': 0.03,          # 3% of total steps
    'min_lr_ratio': 0.1,           # decay to 10% of peak
    
    # Training
    'batch_size': 128,             # per GPU batch, then gradient accumulation
    'num_epochs': 3,               # typical: 2-5 for instruction tuning
    'gradient_clip': 1.0,          # max gradient norm
    'gradient_accumulation_steps': 1,  # adjust based on GPU count
    
    # Precision
    'dtype': 'bfloat16',           # or 'float16' with gradient scaling
    
    # Regularization (optional for fine-tuning)
    'dropout': 0.0,                # typically 0 for fine-tuning
    'label_smoothing': 0.0,        # optional for classification
}
```

---

## 10. Parameter-Efficient Fine-Tuning (PEFT)

### 10.1 Motivation

Full fine-tuning of large models (70B-400B parameters) requires:
- Storing full-precision gradients (huge memory)
- Downloading/distributing full checkpoints
- Expensive hardware (multiple nodes of A100/H100)

**PEFT methods** update only a small fraction of parameters (0.01%-2%) while keeping most parameters frozen. At this low level of parameter change, PEFT often matches full fine-tuning quality.

### 10.2 LoRA (Low-Rank Adaptation)

Hu et al. (2021) proposed LoRA — the most widely used PEFT method.

**Core idea:** The weight update `ΔW = W - W_pretrain` during fine-tuning has low **intrinsic rank**. Instead of learning a full `d × k` matrix, we learn a low-rank decomposition:

```
W' = W_pretrain + BA
```

Where:
- `W_pretrain ∈ ℝ^{d×k}` is frozen
- `B ∈ ℝ^{d×r}`, `A ∈ ℝ^{r×k}` are trainable
- `r << min(d, k)` is the LoRA rank (typically 4-64)

**Initialization:**
- `A` is initialized with random Gaussian (e.g., Kaiming normal)
- `B` is initialized to zero — so `W' = W_pretrain` at the start of training

**Forward pass:**
```
h = W_pretrain · x + BA · x = W_pretrain · x + B(Ax)
```

**Scaling:**
```
W' = W_pretrain + (α / r) · BA
```

Where `α` is a scaling hyperparameter (typically 8-32). This controls the magnitude of the update. In practice, `α/r` acts as a learning rate for the update.

**Why low rank works:**
- Pre-trained models have a low-dimensional "intrinsic dimension" (Li et al., 2018)
- Fine-tuning adjustments lie in a low-rank subspace
- Aghajanyan et al. (2021) showed that pre-trained models have intrinsic rank of ~1 for full dataset fine-tuning

**Where to apply LoRA:**

Typically applied to attention weight matrices W_q, W_k, W_v, W_o. Also applicable to feedforward layers.

```python
class LoRALayer(nn.Module):
    "Low-Rank Adaptation layer."
    def __init__(self, base_layer, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base_layer  # frozen pre-trained weight
        d, k = base_layer.weight.shape
        
        # Low-rank decomposition
        self.lora_A = nn.Parameter(torch.randn(r, k) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(d, r))
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        base_out = self.base(x)
        lora_out = (self.lora_B @ (self.lora_A @ x.T)).T * self.scaling
        # Equivalent: lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return base_out + lora_out
```

### 10.3 LoRA Rank Selection

The rank `r` controls the expressiveness vs efficiency tradeoff:

```
r = 1: Very constrained, limited adaptation. Works for simple style changes.
r = 4-8: Sweet spot for most tasks. Matches full fine-tuning on many benchmarks.
r = 16-32: More capacity, needed for complex tasks or large domain shifts.
r = 64-256: Approaches full fine-tuning expressiveness but reduces parameter savings.
```

**Heuristic for rank selection:**
```python
def suggest_lora_rank(task_type, domain_shift, n_examples):
    """Suggest LoRA rank based on task characteristics."""
    if n_examples < 1000:
        base_rank = 4
    elif n_examples < 10000:
        base_rank = 8
    else:
        base_rank = 16
    
    if domain_shift == 'large':  # e.g., general → biomedical
        base_rank *= 2
    elif domain_shift == 'small':  # e.g., same domain, new task
        pass
    
    if task_type == 'generation':  # harder than classification
        base_rank = int(base_rank * 1.5)
    
    return min(base_rank, 64)
```

**Empirical finding:** Rank 8 often matches full fine-tuning on most NLU tasks. Higher ranks (32-64) help for domain adaptation and generation tasks.

### 10.4 QLoRA: Quantized Low-Rank Adaptation

Dettmers et al. (2023) combined 4-bit quantization with LoRA to fine-tune 65B models on a single GPU.

**Key innovations:**

1. **NF4 (NormalFloat4) quantization**: A new data type optimized for normally distributed weights.

**NF4 quantization math:**

The NF4 datatype is a quantization scheme specifically designed for neural network weights, which are approximately normally distributed (mean 0, standard deviation σ).

Unlike uniform quantization, NF4 allocates more quantization levels near the center of the distribution (where most weights lie) and fewer at the tails.

**Quantization process:**
```python
def nf4_quantize(weight_tensor):
    """Quantize tensor to NF4."""
    # 1. Compute absolute max of the tensor
    abs_max = weight_tensor.abs().max()
    
    # 2. Normalize to [-1, 1]
    normalized = weight_tensor / abs_max
    
    # 3. Quantize to 4-bit (16 levels)
    # NF4 quantization levels (for normally distributed values):
    # q_i = quantiles of N(0,1) at positions (i+0.5)/16 for i = 0..15
    
    # Map normalized values to nearest quantile level
    quant_levels = [-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, 
                    -0.0911, 0.0, 0.0792, 0.1609, 0.2461, 0.3379,
                    0.4407, 0.5626, 0.7230, 1.0]
    
    # Each value maps to 4-bit index (0-15)
    indices = torch.bucketize(normalized, quant_levels) - 1
    indices = indices.clamp(0, 15)
    
    return indices.to(torch.uint8), abs_max  # store quantization stats
```

**Dequantization:**
```python
def nf4_dequantize(indices, abs_max):
    """Dequantize NF4 indices back to bf16."""
    quant_levels = [-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848,
                    -0.0911, 0.0, 0.0792, 0.1609, 0.2461, 0.3379,
                    0.4407, 0.5626, 0.7230, 1.0]
    levels_tensor = torch.tensor(quant_levels, device=indices.device)
    values = levels_tensor[indices.long()] * abs_max
    return values
```

**NF4 vs other quantization:**

NF4 achieves better quality than FP4 (naive 4-bit floating point) and close to FP8/bfloat16 performance for a wide range of model sizes.

**QLoRA training pipeline:**

```
QLoRA training:
1. Load model in 4-bit NF4 (doubles memory efficiency vs 8-bit)
2. Add LoRA adapters in bfloat16 (kept in high precision)
3. During forward/backward:
   - Dequantize 4-bit weights on-the-fly to compute forward
   - Gradients flow only through LoRA adapters
   - Base weights remain in 4-bit
4. After training: merge LoRA with dequantized weights, or keep LoRA separate

Memory: 65B model fits in ~35GB with QLoRA (vs ~130GB for full precision)
```

**Memory comparison (70B model):**
```
Full fine-tuning (fp32):   ~260 GB
Full fine-tuning (bf16):   ~140 GB
LoRA (bf16 base):          ~140 GB base + 200 MB adapters
QLoRA (NF4 base + bf16 LoRA): ~35 GB base + 200 MB adapters
```

### 10.5 DoRA: Weight-Decomposed Low-Rank Adaptation

DoRA (Liu et al., 2024) decomposes pre-trained weights into **magnitude** and **direction** components, learning both separately.

**Core idea:**
```
W' = m · (W_pretrain + BA) / ||W_pretrain + BA||
```

Where:
- `m ∈ ℝ^{d×1}` is a learnable magnitude vector
- `BA` is LoRA (low-rank update to direction)
- The denominator normalizes the direction component

**Why DoRA works:**
- Full fine-tuning adjusts both magnitude and direction of weight vectors
- Standard LoRA couples magnitude and direction changes
- DoRA decouples them, allowing independent control
- The magnitude vector `m` has only d parameters (negligible overhead)

**DoRA update:**
```
// Initialize m from ||W_pretrain||
m = ||W_pretrain||  (computed per row)

// During fine-tuning:
W_direction = (W_pretrain + BA) / (||W_pretrain + BA|| + ε)
W_final = m · W_direction
```

**Performance:** DoRA consistently outperforms LoRA at the same rank, especially on:
- Domain adaptation tasks (larger distribution shift)
- Generation tasks (long-form text)
- Tasks requiring significant weight reorganization

### 10.6 AdaLoRA: Adaptive Budget Allocation

AdaLoRA (Zhang et al., 2023) allocates the LoRA budget (total rank) **dynamically** across weight matrices based on their importance.

**Key idea:** Not all weight matrices need the same LoRA rank. Attention output projections may need more capacity than key projections.

**Mechanism:**
1. Parameterize each weight update as a **SVD** (Singular Value Decomposition):
```
W' = W_pretrain + P · Λ · Q
```
Where `P ∈ ℝ^{d×r}`, `Λ ∈ ℝ^{r×r}` (diagonal), `Q ∈ ℝ^{r×k}`.

2. Regularize the singular values using a **penalty** that drives unimportant singular values to zero:
```
R(Λ) = λ · Σ_i |Λ_ii|  // L1 penalty on singular values
```

3. After pruning: singular values near zero are eliminated, **reducing the effective rank** for that layer.

4. The freed budget is reallocated to more important layers.

**AdaLoRA budget allocation:**
```
// After each N steps:
for each module m:
    importance_m = Σ_j |Λ_jj(m)|  // sum of singular values

// Compute total importance across all modules
total_imp = Σ_m importance_m

// Target total rank across modules = R_total
// Allocate rank proportional to importance:
rank_m = round(R_total * importance_m / total_imp)
```

**Results:** AdaLoRA achieves better performance than equal-rank LoRA with the same total parameter budget, or matches LoRA with 2-4× fewer parameters.

### 10.7 IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)

Liu et al. (2022) proposed IA³ (also written as IA3), which learns element-wise rescaling of keys, values, and feedforward activations.

**Formulation:**

For each input `x` and the corresponding activations:
```
// Self-attention rescaling (keys and values):
K_ia3 = K ⊙ l_k  // l_k is learned vector
V_ia3 = V ⊙ l_v  // l_v is learned vector

// Feedforward rescaling:
FF_ia3 = FF_activation ⊙ l_ff  // l_ff is learned vector
```

**Parameter count:**
- Each learned vector has dimension equal to the layer's hidden dimension
- For a Transformer layer with d_model=4096: 3 × 4096 = 12,288 parameters per layer
- Total for 32-layer model: ~400K parameters — extremely lightweight

**Training:**
- Only the scale vectors `l_k, l_v, l_ff` are updated
- All pre-trained weights remain frozen
- The rescaling is applied element-wise (hadamard product)

**Performance:**
- IA³ matches LoRA rank 1-4 on most tasks
- Generally 1.5-2× more parameter-efficient than LoRA
- For complex tasks (generation, reasoning), LoRA typically outperforms IA³

### 10.8 Prefix-Tuning (Li & Liang, 2021)

Prefix-tuning prepends learnable continuous vectors (virtual tokens) to the input of Transformer layers.

**Formulation:**

For a Transformer model with L layers, prefix-tuning maintains a learned "prefix" matrix `P_ℓ ∈ ℝ^{l_p × d}` for each layer ℓ:

```
// For attention layer ℓ:
[K; P_ℓ^K]   // prefix keys: original K concatenated with learned prefix
[V; P_ℓ^V]   // prefix values
```

Where `l_p` is the prefix length (typically 10-100) and `d` is the dimension.

**Forward pass:**
```python
# During inference with prefix-tuning:
def attention_with_prefix(Q, K, V, prefix_k, prefix_v):
    # prefix_k: [batch, l_p, dim] — learned
    # prefix_v: [batch, l_p, dim] — learned
    
    K_ext = torch.cat([prefix_k, K], dim=1)  # prepend prefix
    V_ext = torch.cat([prefix_v, V], dim=1)
    
    # Standard attention with extended K, V
    scores = torch.matmul(Q, K_ext.transpose(-2, -1)) / sqrt(dim)
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, V_ext)
    
    # Slice to remove prefix positions (keep only T outputs)
    return out[:, :, -T:, :]  # only original positions
```

**Key design choices:**
- Prefix is initialized from the pre-trained embeddings of real words (e.g., "summarize")
- Prefix parameters are optimized via a separate MLP (reparameterization) for training stability
- Different prefix lengths for different layers (longer for higher layers)

**Pros:** Very parameter-efficient (l_p × L × 2d parameters), flexible task representation.
**Cons:** Harder to optimize than LoRA, can reduce effective context length.

### 10.9 Prompt-Tuning (Lester et al., 2021)

Prompt-tuning (also called "soft prompts") is simpler than prefix-tuning — it only prepends learnable tokens to the **input embedding** layer, not to all layers.

**Formulation:**
```
// Input:
x_ext = [P; x_embed]  // P ∈ ℝ^{l_p × d}, x_embed is input embedding

// Forward: standard Transformer with extended input
// No modifications to model architecture
```

**Comparison with Prefix-Tuning:**
- Prompt-tuning: One set of learnable embeddings at input layer only (~l_p × d parameters)
- Prefix-tuning: Learnable vectors at every layer (~L × l_p × 2d parameters)
- Prefix-tuning is 10-100× more parameters but typically 2-5% more accurate

**Prompt length:** 5-100 tokens. Longer prompts provide more capacity but can cause optimization difficulty.

**Initialization strategies:**
1. **Random**: Initialize from N(0, 0.01)
2. **Vocabulary initialization**: Initialize from embeddings of existing tokens (e.g., "summarize the following text briefly")
3. **Class-label initialization**: For classification, initialize with class name embeddings

### 10.10 P-Tuning v1 and v2

**P-Tuning v1** (Liu et al., 2022): Learned continuous prompts for NLU tasks.

The key innovation over prompt-tuning was **prompt encoder** — an LSTM/MLP that generates the prompt embeddings from learned parameters, capturing interactions between prompt tokens.

```python
class PTuningPrompt(nn.Module):
    def __init__(self, n_prompts=20, dim=768, hidden_dim=512):
        super().__init__()
        # Learnable prompt embeddings
        self.prompt_embeddings = nn.Parameter(torch.randn(n_prompts, dim))
        # Prompt encoder (LSTM or MLP)
        self.encoder = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self):
        # Encode prompt embeddings through the MLP
        encoded = self.encoder(self.prompt_embeddings)
        return encoded  # [n_prompts, dim]
```

The prompt encoder stabilizes training — without it, soft prompts tend to collapse to similar values.

**P-Tuning v2** (Liu et al., 2022):

Extended P-Tuning to generation tasks (similar to Prefix-Tuning but with additional design choices):

1. **Deep prompt**: Apply prompts to every layer (like Prefix-Tuning), not just input
2. **Task-specific heads**: For classification, use [CLS] + MLP head instead of verbalizer
3. **Unified for classification and generation**: Same architecture works for NLU and NLG

### 10.11 Adapter Layers

Houlsby et al. (2019) introduced bottleneck adapters: small feedforward networks inserted between Transformer layers.

**Adapter architecture:**
```
Adapter(x) = W_up · σ(W_down · x) + x  // residual connection
```

Where:
- `W_down ∈ ℝ^{d × r}` projects down to bottleneck dimension (r << d)
- `W_up ∈ ℝ^{r × d}` projects back up
- `σ` is a non-linearity (typically ReLU or GeLU)
- Two adapters per layer: after self-attention and after FFN

**Bottleneck factor:** `r = d / m` where m is typically 16-64. This gives 2-6% additional parameters.

**Variants:**

**1. AdapterDrop** (Rücklé et al., 2021): Drop adapters in lower layers during inference for 2× speedup with minimal quality loss. Lower layers are less task-specific, so their adapters can be omitted more safely.

**2. MAD-X** (Pfeiffer et al., 2020): Modular adapters for cross-lingual transfer — separate adapters for language, task, and structure.

**3. AdapterFusion** (Pfeiffer et al., 2021): Combine multiple trained adapters (from different tasks) via learned gating, enabling multi-task without retraining.

**4. Compacter** (Mahabadi et al., 2021): Even more compact — use **PHM** (parameterized hypercomplex multiplication) layers to reduce adapter parameters by 8×.

**Comparison of PEFT Methods:**

| Method | Parameters (vs full) | Expressiveness | Training Stability | Inference Overhead |
|---|---|---|---|---|
| Full Fine-tuning | 100% | Max | Good | None |
| LoRA (r=8) | ~0.1-0.5% | Very good | Excellent | None (can merge) |
| QLoRA (r=32) | ~0.1-0.5% + quantization | Very good | Good | Dequantization cost |
| DoRA (r=8) | ~0.1-0.5% | Excellent | Excellent | None (can merge) |
| AdaLoRA | ~0.1-0.5% | Excellent (adaptive) | Good | None (can merge) |
| IA³ | ~0.01-0.05% | Good | Good | None |
| Prefix-Tuning | ~0.1-1% | Good | Moderate | Reduced throughput (longer seq) |
| Prompt-Tuning | ~0.001-0.01% | Moderate | Moderate | Minimal |
| P-Tuning v2 | ~0.1-1% | Good | Good | Reduced throughput |
| Adapters | ~2-6% | Very good | Good | Slight (extra layers) |

**Practical recommendation:** Start with LoRA (r=8-32) for most PEFT applications. Use QLoRA when GPU memory is limited. Use DoRA for challenging domain adaptation. Use IA³ for extreme parameter efficiency.

---

## 11. RLHF: Reinforcement Learning from Human Feedback

### 11.1 RLHF Pipeline Overview

RLHF (Christiano et al., 2017; Stiennon et al., 2020; Ouyang et al., 2022) is the standard method for aligning LLMs with human preferences. The pipeline has three stages:

```
Stage 1: SFT (Supervised Fine-Tuning)
    Train on high-quality demonstration data (prompt → ideal response)
    
Stage 2: RM (Reward Model Training)
    Collect human preference comparisons (A vs B)
    Train a reward model to predict human preferences
    
Stage 3: RL Fine-Tuning (PPO)
    Use the reward model as a reward signal
    Fine-tune the SFT model using PPO
```

### 11.2 Stage 1: Supervised Fine-Tuning (SFT)

Train the base model on high-quality human-written demonstrations:

```
L_SFT(θ) = -E_{(x, y) ~ D_sft} [log π_θ(y | x)]
```

Where:
- `x` is the prompt
- `y` is the ideal response (human-written)
- `π_θ` is the policy (the SFT model)

**SFT data:** Typically 10K-100K demonstration pairs, covering diverse tasks.

**Key concern:** Models may learn stylistic patterns rather than true capabilities from this small dataset.

### 11.3 Stage 2: Reward Model Training

**Data collection:**
- Present model-generated responses to human raters
- Human chooses which response is better (or provides Elo ranking)
- Typical dataset: 100K-1M comparison pairs

**Score function formulation:**

For a pair of responses `(y_a, y_b)` to prompt `x`, with human preference that `y_a` is better than `y_b`:

```
p(y_a > y_b | x) = σ(r(x, y_a) - r(x, y_b))
```

Where `r(x, y)` is the reward model output (a scalar), and `σ` is the sigmoid function.

**Bradley-Terry preference model:**

```
p(y_a > y_b) = exp(r(x, y_a)) / (exp(r(x, y_a)) + exp(r(x, y_b)))
```

**Reward model loss:**
```
L_RM(φ) = -E_{(x, y_w, y_l) ~ D_rm} [log σ(r_φ(x, y_w) - r_φ(x, y_l))]
```

Where `y_w` is the winning response and `y_l` is the losing response.

**Reward model architecture:**
- Start from SFT model
- Replace the final layer with a linear head that outputs a scalar
- Typically use a small model (e.g., 1B-7B) even for training a 70B policy

**Training details:**
- Batch size: 64-512 pairs
- Learning rate: 1e-6 to 1e-5
- Regularization: weight decay, dropout
- Accuracy goal: 60-75% on held-out comparisons (human agreement is ~75%)

### 11.4 Stage 3: PPO (Proximal Policy Optimization)

PPO (Schulman et al., 2017) is used to optimize the policy against the learned reward model.

**RLHF objective:**
```
max_θ E_{x ~ D, y ~ π_θ(y|x)} [r_φ(x, y)] - β · KL(π_θ(y|x) || π_ref(y|x))
```

Where:
- `π_θ` is the policy being optimized
- `π_ref` is the reference policy (frozen SFT model)
- `β` controls the KL penalty strength (typically 0.01-0.1)
- KL penalty prevents the policy from diverging too far from the reference (reduces reward hacking)

**PPO in RLHF — detailed steps:**

```
ALGORITHM: PPO for RLHF
────────────────────────
Input: SFT model π_ref, reward model r_φ
Initialize: π_θ = π_ref

for each iteration:
    // 1. Collect rollouts
    for each prompt x in batch:
        y = sample(π_θ(y | x))  // generate response
        reward = r_φ(x, y)  // score from reward model
        kl = KL(π_θ(y|x) || π_ref(y|x))  // KL per token
        advantage = reward - β * kl  // penalized reward
    
    // 2. Compute PPO loss
    for each (x, y) in batch:
        π_old = prob from old policy (before update)
        π_new = prob from current policy
        
        ratio = π_new(y|x) / π_old(y|x)
        clipped_ratio = clamp(ratio, 1-ε, 1+ε)
        
        L_policy = -E[min(ratio * A, clipped_ratio * A)]
        L_value = E[(V_θ(x) - reward)²]  // value function loss (optional)
        
        L_PPO = L_policy + c_v * L_value - c_e * H(π_θ)  // entropy bonus
        
    // 3. Update policy
    θ ← θ - η · ∇_θ L_PPO
```

**PPO in RLHF — implementation details:**

**PPO epoch:** Typically 1-4 epochs per batch of rollouts. Multiple epochs increase sample efficiency but risk overfitting to the reward model.

**KL penalty variants:**

1. **Fixed KL:** `β` is constant during training. Simple but requires tuning.

2. **Adaptive KL** (approximate KL per batch):
```python
def adaptive_kl_beta(kl_current, kl_target=0.01, beta=0.01):
    """Adjust β to maintain target KL divergence."""
    if kl_current > 2 * kl_target:
        beta *= 1.5
    elif kl_current < 0.5 * kl_target:
        beta /= 1.5
    return beta
```

3. **KL budget**: Maintain a running KL budget. Penalize only when budget is exceeded.

**Value function (critic):**
- Often initialied from the reward model
- Learns to predict the expected return from state x
- Reduces variance of advantage estimates
- Optional: can be omitted (use reward directly as advantage)

**Advantage estimation:**
```
GAE (Generalized Advantage Estimation):
A_t = Σ_{k=0}^{T-t-1} (λγ)^k · δ_{t+k}
δ_t = r_t + γ · V(s_{t+1}) - V(s_t)
```

In RLHF, the simpler approach is often used:
```
A_t = r_φ(x, y) - V_θ(x)  // per-sequence advantage
```

### 11.5 Reward Hacking and Over-Optimization

**Reward hacking:** The policy exploits the reward model by generating responses that score high but are actually not good (e.g., nonsensical text that the RM incorrectly scores high).

**Signs of reward hacking:**
1. Reward model score increases but human evaluation plateaus
2. Generated text becomes repetitive, overly long, or uses unusual phrasing
3. The KL divergence from the reference model grows large

**Mitigations:**

1. **KL penalty** (as described above) — the primary defense

2. **Reward normalization:**
```python
# Normalize rewards to have zero mean, unit variance
rewards = reward_model(batch)
rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
```

3. **Ensemble reward models**: Average scores from multiple independently trained reward models:
```
r_ensemble(x, y) = (1/M) Σ_{i=1}^{M} r_i(x, y)
```

4. **Reward scaling**: Clip rewards to a reasonable range:
```python
reward = torch.clamp(reward, -5.0, 5.0)
```

5. **Per-token KL vs per-sequence KL**: Per-token KL provides finer-grained control:
```
per_token_kl = Σ_t KL(π_θ(y_t | x, y_{<t}) || π_ref(y_t | x, y_{<t}))
```

**Reward over-optimization** (Gao et al., 2023):

Empirically, the true reward (human satisfaction) increases with PPO iterations but then decreases, while the proxy reward (RM score) continues to increase. This "reward over-optimization" occurs at a predictable rate:

```
true_reward ≈ proxy_reward - α · KL
```

Where `α` depends on the quality of the reward model. Better RMs have larger α (over-optimization sets in later).

**Practical recommendation:**
- Train for 1-3 PPO epochs
- Monitor both proxy reward and held-out evaluation metrics
- Stop when human-annotated quality peaks (not when RM score peaks)
- Use adaptive KL to maintain target KL (~0.01 per token)

### 11.6 PPO Implementation Pseudocode

```python
def rlhf_ppo_step(sft_model, reward_model, ref_model, ppo_model, batch):
    """Single PPO optimization step."""
    
    # Unpack batch
    prompts, responses, old_log_probs, advantages, returns = batch
    
    # 1. Current policy probabilities
    current_log_probs = ppo_model.get_log_probs(prompts, responses)
    current_values = ppo_model.get_values(prompts, responses)
    
    # 2. KL penalty (per-token)
    with torch.no_grad():
        ref_log_probs = ref_model.get_log_probs(prompts, responses)
    
    kl = torch.exp(ref_log_probs - current_log_probs) - (ref_log_probs - current_log_probs) - 1
    # Simplified: kl = (ref_log_probs - current_log_probs)  # approximates forward KL
    
    # 3. Ratio and clipped ratio
    ratio = torch.exp(current_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    
    # 4. Policy loss with KL
    surrogate = -torch.min(ratio * advantages, clipped_ratio * advantages)
    policy_loss = surrogate.mean()
    
    # 5. Value loss
    value_loss = F.mse_loss(current_values.squeeze(), returns)
    
    # 6. Entropy bonus
    entropy = ppo_model.entropy(prompts)
    entropy_bonus = entropy.mean()
    
    # 7. Total loss
    total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus
    
    return total_loss
```

---

## 12. DPO and Direct Alignment Methods

### 12.1 DPO: Direct Preference Optimization

Rafailov et al. (2023, "Direct Preference Optimization: Your Language Model is Secretly a Reward Model") showed that the RLHF objective can be solved in closed form, eliminating the need for a separate reward model.

**Key insight:** The optimal policy under the KL-constrained RL objective has a closed-form solution in terms of the reward function:

```
π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp(r(x, y) / β)
```

Rearranging to express the reward:
```
r(x, y) = β · log(π*(y|x) / π_ref(y|x)) + β · log Z(x)
```

**DPO loss derivation:**

Substituting the reward expression into the Bradley-Terry preference model:

```
p(y_w > y_l | x) = σ(r(x, y_w) - r(x, y_l))
```

Yields the DPO loss:

```
L_DPO(θ) = -E_{(x, y_w, y_l)} [log σ(β · log(π_θ(y_w|x) / π_ref(y_w|x)) 
                                       - β · log(π_θ(y_l|x) / π_ref(y_l|x)))]
```

Where:
- `y_w` is the preferred/won response
- `y_l` is the dispreferred/lost response
- `β` is the temperature parameter (controls how far the policy can deviate from π_ref)
- `π_θ` is the policy being optimized
- `π_ref` is the frozen reference policy

**Simplified DPO loss:**

```
L_DPO = -log σ(β · (log π_θ(y_w|x) - log π_ref(y_w|x) 
                   - log π_θ(y_l|x) + log π_ref(y_l|x)))
     = -log σ(β · (r_θ(x, y_w) - r_θ(x, y_l)))
```

Where implicitly `r_θ(x, y) = β · log(π_θ(y|x) / π_ref(y|x))`.

**The key difference between DPO and PPO:**

| DPO | PPO |
|---|---|
| Directly optimizes preference loss | RL optimization via reward model |
| No explicit reward model needed | Requires trained reward model |
| Single-stage preference optimization | Three-stage (SFT → RM → RL) |
| No online sampling during training | Online sampling (rollouts) |
| Computationally simpler | More complex, more hyperparameters |
| Sensitive to off-policy preference data | More stable with online exploration |
| Objective: maximize log-likelihood of preferred response | Maximize reward subject to KL constraint |

### 12.2 DPO Gradient

The gradient of the DPO loss provides insight into how DPO works:

```
∇_θ L_DPO = -β · E[σ(-β · (log(π_θ(y_w|x)/π_ref(y_w|x)) - log(π_θ(y_l|x)/π_ref(y_l|x))))
                    · (∇_θ log π_θ(y_w|x) - ∇_θ log π_θ(y_l|x))]
```

The sigmoid weight `σ(-β·Δ)` is high when:
- The preferred response has low likelihood (compared to reference)
- The dispreferred response has high likelihood

Thus DPO automatically focuses on:
- **Increasing** probability of preferred responses when they are currently under-priced
- **Decreasing** probability of dispreferred responses when they are currently over-priced

### 12.3 DPO Implementation

```python
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """Compute DPO loss.
    
    Args:
        policy_chosen_logps: log π_θ(y_w|x)
        policy_rejected_logps: log π_θ(y_l|x)
        ref_chosen_logps: log π_ref(y_w|x)
        ref_rejected_logps: log π_ref(y_l|x)
        beta: temperature parameter
    """
    # Log ratios
    log_ratio_chosen = policy_chosen_logps - ref_chosen_logps
    log_ratio_rejected = policy_rejected_logps - ref_rejected_logps
    
    # Preference margin
    margin = log_ratio_chosen - log_ratio_rejected
    
    # DPO loss
    loss = -F.logsigmoid(beta * margin).mean()
    
    # Optional: accuracy metric
    with torch.no_grad():
        accuracy = (margin > 0).float().mean()
    
    return loss, accuracy
```

**Training loop:**
```python
def train_dpo(policy, ref_policy, dataloader, beta=0.1, lr=1e-6):
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)
    
    for batch in dataloader:
        prompts, chosen, rejected = batch
        
        # Forward pass through policy (current)
        policy_chosen_logps = policy.get_log_probs(prompts, chosen)
        policy_rejected_logps = policy.get_log_probs(prompts, rejected)
        
        # Forward pass through reference (frozen)
        with torch.no_grad():
            ref_chosen_logps = ref_policy.get_log_probs(prompts, chosen)
            ref_rejected_logps = ref_policy.get_log_probs(prompts, rejected)
        
        # DPO loss
        loss, _ = dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            ref_chosen_logps, ref_rejected_logps,
            beta=beta
        )
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### 12.4 GRPO: Group Relative Policy Optimization

GRPO (Shao et al., 2024 / DeepSeek-R1) is a variant of PPO that eliminates the need for a value function by using **group-based advantage estimation**.

**Core idea:** For each prompt, sample **G responses** from the current policy. Compute the advantage of each response relative to the group average:

```
A_i = (r_i - mean({r_1, ..., r_G})) / std({r_1, ..., r_G})
```

Where `r_i` is the reward for response i (from a reward model or rule-based reward).

**GRPO loss:**
```
L_GRPO(θ) = -E_{(x, {y_i})} [ (1/G) Σ_i A_i · min(r_i(θ), clip(r_i(θ), 1-ε, 1+ε)) 
                                 - β · KL(π_θ || π_ref) ]
```

Where `r_i(θ) = π_θ(y_i|x) / π_old(y_i|x)` is the importance sampling ratio.

**Why GRPO works without a value function:**
- The group-based normalization provides a zero-mean, unit-variance advantage signal
- Like DPO, the "baseline" is computed from the group mean (not from a learned value function)
- This eliminates the critic network, reducing memory and complexity

**GRPO vs PPO:**
| Aspect | PPO | GRPO |
|---|---|---|
| Value function | Required | Not needed |
| Advantage estimation | GAE or residual from value | Group-normalized reward |
| Memory | Policy + Value + Reward + Reference | Policy + Reward + Reference |
| Complexity | Higher (need to train critic) | Lower |
| Sample efficiency | Higher (value function stabilizes) | Lower (rely on group size) |
| Group size G | N/A | 4-32 responses per prompt |

### 12.5 KTO: Kahneman-Tversky Optimization

KTO (Ethayarajh et al., 2024) aligns models using a **prospect theory**-inspired loss that only requires knowing whether each response is "desirable" or "undesirable" (not paired preferences).

**KTO loss:**
```
L_KTO(θ) = E_{x, y} [ w(y) · (log(1 + exp(β · (KL(π_θ||π_ref)(x, y) - δ_ref(x)))))
                          - (1 - w(y)) · (log(1 + exp(β · (δ_ref(x) - KL(π_θ||π_ref)(x, y))))) ]
```

Where:
- `w(y)` is 1 for desirable (chosen) responses, 0 for undesirable (rejected)
- `KL(π_θ||π_ref)(x, y) = log(π_θ(y|x) / π_ref(y|x))`
- `δ_ref` is a reference point (typically the average KL of the reference model on good responses)

**Key advantage:** KTO works with unpaired "desired/undesired" labels (e.g., thumbs up/down) instead of paired comparisons. This simplifies data collection.

### 12.6 ORPO: Odds Ratio Preference Optimization

ORPO (Hong et al., 2024) combines SFT and preference learning into a single loss, eliminating the reference model.

**ORPO loss:**
```
L_ORPO(θ) = E_{(x, y_w, y_l)} [ -log p_θ(y_w|x) 
                                 - λ · log σ(log(odds_θ(y_w|x) / odds_θ(y_l|x))) ]
```

Where `odds_θ(y|x) = p_θ(y|x) / (1 - p_θ(y|x))` and `λ` is a weighting parameter.

**The odds ratio:**
```
odds_θ(y_w|x) / odds_θ(y_l|x) = [π_θ(y_w|x) / (1 - π_θ(y_w|x))] / [π_θ(y_l|x) / (1 - π_θ(y_l|x))]
```

**Key insight:** ORPO avoids the reference model entirely by using the **odds ratio** as the implicit reward. The first term is the standard SFT loss (maximize likelihood of preferred), and the second term maximizes the odds ratio between preferred and rejected.

**Advantages:**
- No reference model needed (2× memory savings vs DPO)
- Single-stage training (SFT + alignment)
- Simpler pipeline

**Disadvantages:**
- May be less stable than DPO for challenging alignment tasks
- The odds ratio assumption (that odds reflect preference quality) may not always hold

### 12.7 SimPO: Simple Preference Optimization

SimPO (Meng et al., 2024) further simplifies DPO by replacing the reference model with a **length-normalized reward** based on the average log probability.

**SimPO reward:**
```
r_θ(x, y) = (β / |y|) · log π_θ(y|x)
```

Where `|y|` is the response length. Length normalization prevents the model from preferring longer responses (a common issue in unnormalized methods).

**SimPO loss:**
```
L_SimPO(θ) = -E_{(x, y_w, y_l)} [log σ(β · ((1/|y_w|) log π_θ(y_w|x) - (1/|y_l|) log π_θ(y_l|x)) - γ)]
```

Where `γ` is a margin hyperparameter (typically 0.5-1.5).

**Advantages over DPO:**
- No reference model needed
- Length normalization reduces verbosity bias
- Margin γ allows controlling the separation between chosen and rejected

**SimPO vs DPO on verbosity:**
```
DPO: r_θ(x, y) = β · log(π_θ(y|x) / π_ref(y|x))
     → Longer responses can have lower probability, reducing bias
     But: if π_ref assigns low probability to long responses, DPO may favor long responses

SimPO: r_θ(x, y) = (β / |y|) · log π_θ(y|x)
     → Explicitly penalizes length by normalizing
     → More robust to response length differences
```

### 12.8 Rejection Sampling and Best-of-N

**Rejection sampling:** Sample N responses from the model, select the best one according to a reward model or heuristic:

```python
def rejection_sampling(policy, prompt, reward_model, N=16):
    responses = [policy.generate(prompt) for _ in range(N)]
    scores = [reward_model(prompt, r) for r in responses]
    best_idx = np.argmax(scores)
    return responses[best_idx], scores[best_idx]
```

**Best-of-N rejection sampling for training** (used by Llama 2, GPT-4):

1. For each training prompt, generate N responses from the current policy
2. Score each response with the reward model
3. Keep the highest-scoring response as the "chosen" response
4. Fine-tune the policy on these chosen responses (standard SFT)

```python
def best_of_n_training(policy, prompts, reward_model, N=8):
    """One iteration of Best-of-N training."""
    chosen_responses = []
    
    for prompt in prompts:
        # Sample N responses
        responses = [policy.generate(prompt, temperature=0.7) for _ in range(N)]
        
        # Score with reward model
        scores = [reward_model.score(prompt, r) for r in responses]
        
        # Select best
        best_idx = np.argmax(scores)
        chosen_responses.append((prompt, responses[best_idx]))
    
    # SFT on best responses
    train_with_sft(policy, chosen_responses)
```

**Theoretical benefit of Best-of-N (BON):**

For a reward model with score `r(y|x)`, the expected maximum score from N i.i.d. samples follows:

```
E[max_{i=1..N} r_i] ≈ μ + σ · √(2 · log N)
```

Where μ and σ are the mean and std of the reward distribution. The benefit scales logarithmically with N — diminishing returns beyond N=128-256.

**Rejection sampling vs PPO/DPO:**

| Method | Online/Offline | Complexity | Data Efficiency |
|---|---|---|---|
| PPO | Online (rollouts) | High | High (uses reward signal directly) |
| DPO | Offline (fixed preference pairs) | Low | Medium (depends on preference quality) |
| Rejection Sampling | Online (model-based selection) | Low | Medium (wasteful of N-1 samples) |
| Best-of-N | Online | Low | Lower than PPO, higher than random |

---

## 13. Constitutional AI and RLAIF

### 13.1 Constitutional AI (CAI)

Bai et al. (2022, Anthropic) proposed Constitutional AI — a method for training harmless AI assistants **without** extensive human feedback.

**The constitution:** A set of principles (e.g., "Do not generate hate speech," "Be helpful and harmless"). Example principles:

```
1. Choose the response that is most helpful and harmless.
2. Do not engage in harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
3. If the request is harmful, explain why you cannot comply and offer a safer alternative.
4. Be honest about your limitations and uncertainties.
5. Avoid making up information (hallucination).
```

**Two-stage training process:**

**Stage 1: Supervised CAI (Constitutional SFT):**

```
For each harmful prompt x:
    1. Generate initial response y_0 ~ π_θ(·|x)
    2. Apply critique: c ~ critique_model(·|x, y_0, constitution)
       (The critique model identifies how y_0 violates the constitution)
    3. Generate revised response: y_1 ~ π_θ(·|x, y_0, c)
    4. Train π_θ to predict y_1 from x (SFT)

Loss: L_CAI_SFT = -log π_θ(y_1 | x)
```

**Stage 2: RLAIF (RL from AI Feedback):**

```
For each prompt x:
    1. Sample two responses: y_a, y_b ~ π_θ(·|x)
    2. Ask critique model: "Which response better follows the constitution?"
    3. Train reward model on AI-generated preferences
    4. Apply PPO/DPO with this reward model
```

### 13.2 RLAIF: Reinforcement Learning from AI Feedback

RLAIF uses an AI judge (often a stronger model like GPT-4 or Claude) to provide feedback instead of human raters.

**RLAIF pipeline:**

```
1. Generate preference pairs (y_a, y_b) from current policy
2. Send to LLM judge: "Which response is better? A or B?"
3. Train reward model on judge preferences
4. Use PPO/DPO with this reward model
```

**Self-play variant:** Use the same model that is being trained as the judge:
```
Policy → provides feedback → trains itself
```

This is the approach used by:
- **RLCD** (Yang et al., 2023): Self-training with contrastive feedback
- **SPIN** (Chen et al., 2024): Self-play fine-tuning with preference learning

**Quality of AI feedback:**

Research shows that AI feedback correlates with human feedback at 70-80% (depending on the task). RLAIF can match or approach the performance of RLHF while reducing human annotation costs by 50-100×.

### 13.3 RLAIF Implementation

```python
def rlaif_judge(prompt, response_a, response_b, judge_model, constitution):
    """Use LLM judge to compare two responses."""
    judge_prompt = f"""
    {constitution}
    
    Prompt: {prompt}
    
    Response A: {response_a}
    
    Response B: {response_b}
    
    Which response better follows the above principles?
    Answer with only 'A' or 'B'.
    """
    
    result = judge_model.generate(judge_prompt, temperature=0.0)
    return 'A' if 'A' in result else 'B'

def generate_rlaif_preferences(policy, prompts, judge_model, constitution):
    """Generate AI-labeled preference pairs."""
    preferences = []
    
    for prompt in prompts:
        # Generate two responses
        y_a = policy.generate(prompt)
        y_b = policy.generate(prompt)
        
        # Get AI preference
        winner = rlaif_judge(prompt, y_a, y_b, judge_model, constitution)
        
        if winner == 'A':
            preferences.append((prompt, y_a, y_b))
        else:
            preferences.append((prompt, y_b, y_a))
    
    return preferences
```

### 13.4 Self-Play and Iterative Training

**Self-play fine-tuning** (Chen et al., 2024 - SPIN):

```
Initialize: model π_θ

For each iteration:
    1. Generate synthetic responses: y ~ π_θ(·|x) for prompts x
    2. Use π_θ as the judge: compare y with the training data response y_ref
    3. Train π_θ to distinguish its own generations from human data
```

**SPIN loss:**
```
L_SPIN(θ) = -E_{(x, y_ref, y_synthetic)} [log σ(f_θ(x, y_ref) - f_θ(x, y_synthetic))]
```

Where `f_θ(x, y) = log(π_θ(y|x) / π_ref(y|x))` as in DPO.

**Iterative DPO:**

- Start with SFT model
- Round 1: Collect preference data → DPO training → Model v1
- Round 2: Generate new responses from model v1 → Collect new preferences → DPO training → Model v2
- ... continue for K rounds

Each round generates harder preference pairs (the model's own generations are harder to differentiate), pushing the model toward the Pareto frontier.

---

## 14. Instruction Tuning

### 14.1 What is Instruction Tuning?

Instruction tuning (also called **supervised fine-tuning** or SFT) trains a pre-trained language model to follow instructions by fine-tuning on (instruction, response) pairs.

**Key insight:** A model pre-trained on next-token prediction doesn't naturally follow instructions. Instruction tuning teaches the model the **format** of human-AI interaction.

### 14.2 Instruction Data Formats

**Format 1: Simple prompt-response (most common):**
```
### Instruction:
Translate to French: "Hello, how are you?"

### Response:
Bonjour, comment allez-vous ?
```

**Format 2: Chat template (multi-turn):**
```
<|system|>
You are a helpful AI assistant.

<|user|>
What is the capital of France?

<|assistant|>
The capital of France is Paris.

<|user|>
What is its population?

<|assistant|>
Paris has a population of approximately 2.1 million people within the city limits, and over 12 million in the metropolitan area.
```

**Format 3: FLAN-style (task description + options):**
```
Classify the sentiment of the following text as positive, negative, or neutral.

Text: I just received the package, it's perfect.
Sentiment: Positive
```

**Format 4: Few-shot with instructions:**
```
Instruction: Classify the relation between entities.

Text: "The CEO of Apple, Tim Cook, announced the new iPhone."
Entities: Apple, Tim Cook
Relation: company_CEO
```

### 14.3 Multi-Task Instruction Tuning

**FLAN** (Wei et al., 2022) and **FLAN-T5**/FLAN-PaLM (Chung et al., 2022) demonstrated that instruction tuning on many tasks (60-2000+) dramatically improves generalization to unseen tasks.

**Key findings:**
1. **More tasks → better generalization**: Every added task improves held-out task performance (logarithmically)
2. **Template diversity**: Using multiple templates per task (10-30) improves robustness more than any single template
3. **Balanced sampling**: Sample each task approximately equally (not proportional to dataset size)
4. **Task clustering**: Related tasks should be balanced (e.g., not too many QA tasks overwhelming summarization)

**FLAN template approach:**
```
For task T (e.g., natural language inference):
Template 1: "{premise}\nQuestion: {hypothesis} True or False?"
Template 2: "Premise: {premise}\nHypothesis: {hypothesis}\nIs this entailment?"
Template 3: "Does {premise} imply {hypothesis}?\nA) Yes\nB) No"
Template 4: "Relationship between:\n{premise}\n{hypothesis}"
...
```

**Super-NaturalInstructions** (Wang et al., 2022): A collection of 1,600+ NLP tasks with detailed instructions. Key design:
- **Task definitions** (not just input-output pairs): Each task includes a natural language description
- **Positive and negative examples**: Both correct and incorrect examples for in-context learning
- **Instruction-to-instruction generalization**: Models learn to parse new instructions at inference time

### 14.4 Instruction Data Quality Considerations

**Data volume:** 1K-100K instruction pairs typically work. More data can help but shows diminishing returns.

**Data diversity** matters more than volume. A well-designed set of 5K instructions covering 50 task types often outperforms 50K instructions covering 5 task types.

**Quality filtering for instruction data:**
```
1. Remove duplicates (exact or near-duplicate instruction-response pairs)
2. Detect and remove low-effort responses ("I don't know", "Let me think...")
3. Decontaminate: remove pairs that leak evaluation data
4. Balance response length (remove very short or very long outliers)
5. Language consistency (remove code-mixed or wrong-language responses)
6. Toxicity filter
```

**Decontamination methods:**

```python
def decontaminate(prompt, response, eval_datasets):
    """Remove instruction pairs that overlap with evaluation data."""
    # Compute n-gram overlap with evaluation benchmarks
    prompt_ngrams = set(extract_ngrams(prompt, n=13))
    
    for eval_set in eval_datasets:
        for eval_example in eval_set:
            eval_ngrams = set(extract_ngrams(eval_example, n=13))
            overlap = len(prompt_ngrams & eval_ngrams)
            if overlap > 0.5 * min(len(prompt_ngrams), len(eval_ngrams)):
                return False  # Too similar — remove
    return True
```

### 14.5 Multi-Turn Instruction Data

Modern assistants need to handle multi-turn conversations. Training data formats:

```
Turn 1:
  User: "What is machine learning?"
  Assistant: "Machine learning is a subset of AI that..."

Turn 2 (follow-up):
  User: "Can you give me an example?"
  Assistant: "Sure! A common example is spam detection..."

Turn 3 (correction):
  User: "Actually, I meant examples of unsupervised learning."
  Assistant: "Ah, for unsupervised learning, examples include clustering..."
```

**Multi-turn training strategies:**
1. **Full conversation**: Train on entire conversation as one sequence
2. **Random-turn masking**: Randomly mask assistant responses and predict only those
3. **Multi-turn SFT**: Each training example is one turn context → one response

**Loss computation:**
```python
# Typically, only compute loss on assistant tokens
labels = input_ids.clone()
labels[labels == user_token_id] = -100  # ignore user tokens
labels[labels == system_token_id] = -100  # ignore system tokens

loss = cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
```

---

## 15. Synthetic Data Generation

### 15.1 Motivation

Synthetic data generation creates training data algorithmically rather than collecting human-written data. Motivations:
- **Scale**: Generate unlimited training examples
- **Cost**: 10-100× cheaper than human annotation
- **Coverage**: Target specific domains, skills, or difficulty levels
- **Privacy**: No exposure to sensitive human data
- **Iteration**: Rapidly generate new data for different training regimes

### 15.2 Self-Instruct

Wang et al. (2022) proposed Self-Instruct — a pipeline for generating instruction data from LLMs.

**Self-Instruct pipeline:**
```
1. Seed pool: Hand-craft 175 seed instructions covering diverse tasks
2. For each iteration:
   a. Sample k seed instructions (e.g., 8) as in-context examples
   b. Prompt LLM: "Given examples of instructions, generate new instructions"
   c. Generate instruction
   d. Generate input for the instruction (optional)
   e. Generate response for the instruction using the LLM
   f. Filter low-quality generations
3. Repeat until target size reached
```

**Self-Instruct generation prompt:**
```
You are given a set of instruction examples. Generate a new instruction
that is different from all examples.

Examples:
1. Instruction: {inst_1}
2. Instruction: {inst_2}
...
8. Instruction: {inst_8}

Generate a new instruction:
```

**Self-Instruct filtering:**
```
1. ROUGE-L-based filtering: Remove instructions that are too similar (ROUGE-L > 0.7)
2. Keyword filtering: Remove instructions containing stopwords only
3. Classification filtering: Only keep instructions that belong to target categories
4. Length filtering: Remove very short (< 3 words) or very long (> 100 words) instructions
```

**Limitations of vanilla Self-Instruct:**
- Generated instructions tend to be short and simple
- Instructions cluster around the seed task types
- Output quality depends heavily on the base LLM
- Less effective for complex reasoning tasks

### 15.3 Evol-Instruct

Xu et al. (2023, WizardLM) proposed Evol-Instruct, which iteratively **evolves** instructions to be more complex and diverse.

**Evol-Instruct pipeline:**
```
Phase 1: Initial generation (like Self-Instruct)
  - Generate simple instructions from base model

Phase 2: Instruction evolution (iterative)
  For each instruction, apply evolution operations:
  
  a. In-depth Evolving:
     - Add constraints (e.g., "in exactly 5 sentences")
     - Deepen (e.g., "explain the underlying mechanism")
     - Concretize (e.g., "give 3 specific examples")
     - Increase reasoning steps (e.g., "first identify the cause, then...")
     - Complicate input (e.g., add distractions or multi-part questions)
  
  b. In-breadth Evolving:
     - Generate new instruction from scratch based on existing topic
     - Mutate to a different domain/task type
     - Combine two instructions into one

Phase 3: Elimination (filtering)
  - Remove instructions where the LLM cannot generate a valid response
  - Remove instructions that are too similar to existing ones (ROUGE-L > 0.7)
  - Remove instructions where response quality is poor (< 20 tokens)
  - Remove instructions with low response entropy (model is uncertain)

Phase 4: Response generation
  - Use GPT-4/Claude or other strong model to generate high-quality responses
```

**Evolution prompt example (In-depth Evolving - Add constraints):**
```
I want you to act as an Instruction Evolver and rewrite the following
instruction into a more complex version.

The new instruction should:
1. Add one or more specific constraints (e.g., length, format, style)
2. Maintain the original intent and core requirement
3. Be realistic and answerable

Original instruction: "Explain the theory of relativity."

New instruction (with constraints):
```

**Evol-Instruct results:**
- WizardLM used 52K evolved instructions from a 7B Llama model → outperformed Llama itself
- The evolution process produces more challenging, diverse, and complex instructions
- **Key finding:** Evolved instructions at complexity levels 4-6 (on a 1-7 scale) provide the most improvement. Very complex (7) instructions may be unrealistic and harm quality.

**Wizard series applications:**
- WizardCoder: Evol-Instruct for code (using Code Alpaca seeds)
- WizardMath: Evol-Instruct for math
- WizardLM: General-purpose Evol-Instruct

### 15.4 Wizard Methods (WizardCoder, WizardMath)

**WizardCoder** (Luo et al., 2023):

Applied Evol-Instruct to code generation:

1. **Instruction evolution for code**: Evolve programming instructions to be more complex
2. **Code Evol-Instruct**:
   - Add constraints (libraries, edge cases, performance requirements)
   - Add input/output specifications
   - Add code-specific requirements (time complexity, memory)
3. **Response evolution**: Evolve code responses to:
   - Add comments and documentation
   - Optimize for readability and performance
   - Include error handling

**WizardMath** (Luo et al., 2023):

Applied Evol-Instruct to mathematical reasoning:

1. **Math instruction evolution**:
   - Increase number of reasoning steps
   - Add intermediate conditions
   - Combine multiple math concepts
   - Increase numerical complexity
2. **Process supervision**: Generate step-by-step reasoning chains (CoT) 
3. **Verification-based rejection sampling**: Use multiple solutions and verify correctness

**Key technique — Evol-Instruct for math:**
```
Original: "Solve for x: 2x + 5 = 13"

Evolved (more complex):
"A rectangle's length is 3 more than twice its width. If the perimeter 
of the rectangle is 34 cm, find the length and width of the rectangle. 
[Also verify that the area is at least 50 cm²]"
```

### 15.5 Reverse Instruction Tuning

Instead of generating instructions from seeds, generate responses first then derive instructions:

```
1. Sample domain text (e.g., Wikipedia article, book passage, code snippet)
2. Prompt LLM: "What question was this text answering?/What instruction does this fulfill?"
3. Generate (instruction, response) pairs

This approach:
- Ensures response quality (it's drawn from high-quality text)
- Generates diverse instructions naturally
- Avoids the limitation of seed-based instruction generation
```

**Example:** 
```
Response text: "The Eiffel Tower was built in 1889 for the World's Fair..."
Generated instruction: "When was the Eiffel Tower built and for what purpose?"
Generated instruction 2: "Describe the historical context of the Eiffel Tower's construction."
```

### 15.6 Multi-Turn Synthetic Data

Generate multi-turn conversations for training chat models:

**Method 1: Topic-guided:**
```
1. Pick topic → Generate first user turn
2. Generate assistant response
3. Generate follow-up user turn (follow-up, challenge, request clarification)
4. Generate assistant response
5. Repeat for 3-10 turns

Prompt for follow-up generation:
"Given the following conversation, generate a natural follow-up question from the user:
Topic: {topic}
User: {first_message}
Assistant: {first_response}
User: (generate next turn)"
```

**Method 2: Self-chat:**
```
- Two instances of the same model talk to each other
- One acts as user, one as assistant
- Provide system prompt describing the user persona
- Continue for multiple turns

Used by: ShareGPT conversations, synthetic chat datasets
```

### 15.7 Synthetic Data Quality Control

**Common issues with synthetic data:**
1. **Hallucination**: Model generates plausible but incorrect responses
2. **Blandness**: Repetitive patterns, same style across all responses
3. **Simplicity**: Instructions are too easy, not diverse
4. **Spurious correlations**: Data contains artifacts the model can memorize

**Quality control measures:**

```python
def filter_synthetic_data(data, quality_model, threshold=0.8):
    """Filter synthetic data using a quality classifier."""
    filtered = []
    
    for instruction, response in data:
        # 1. Check response quality
        quality_score = quality_model.score_quality(response)
        
        # 2. Check instruction-response alignment
        alignment = quality_model.check_alignment(instruction, response)
        
        # 3. Check for repetition
        repetition = detect_repetition(response)
        
        # 4. Check response diversity (vs existing data)
        diversity = compute_diversity(response, filtered)
        
        if all([
            quality_score > threshold,
            alignment > threshold,
            repetition < 0.3,
            diversity > 0.2
        ]):
            filtered.append((instruction, response))
    
    return filtered
```

---

## 16. Data Augmentation for LLMs

### 16.1 Paraphrasing

Paraphrasing generates multiple surface forms of the same training example, improving robustness and data efficiency.

**Methods:**

**1. LLM-based paraphrasing:**
```
Prompt: "Paraphrase the following text while preserving its meaning:

Original: 'The cat sat on the mat.'
Paraphrase:"
```

**2. Back-translation (see Section 16.2)**

**3. Rule-based:**
- Synonym replacement (using WordNet)
- Syntactic transformations (active ↔ passive voice)
- Entity substitution (replace named entities with alternatives)
- Sentence splitting/combining

**4. Controlled generation (SC paraphrase):**
- Generate paraphrases with controlled syntactic structure
- Use constituency parse trees as conditioning
- Ensures diverse syntactic forms

**Application in LLM training:**
- Paraphrase instruction data to increase template diversity
- Paraphrase demonstrations to create multiple in-context examples
- Paraphrase evaluation prompts to test instruction robustness

### 16.2 Back-Translation

Back-translation (Sennrich et al., 2016) generates augmentations by translating text to an intermediate language and back.

**Pipeline:**
```
English: "The quick brown fox jumps over the lazy dog."
→ Translate to French: "Le renard brun rapide saute par-dessus le chien paresseux."
→ Translate back to English: "The fast brown fox jumps over the lazy dog."

Result: Natural paraphrase with lexical and structural variation.
```

**Mathematical formulation:**
```
P_back(x' | x) = Σ_z P(z | x; θ_fwd) · P(x' | z; θ_bwd)
```

Where:
- `z` is the intermediate language
- `θ_fwd` is the forward translation model (X → Z)
- `θ_bwd` is the backward translation model (Z → X')

**Noising effects:**
Back-translation introduces three types of noise:
1. **Lexical**: Different word choices
2. **Syntactic**: Different sentence structures
3. **Information-preserving**: Meaning is preserved (approximately)

**Languages for back-translation:**
```
Best intermediate languages (ordered by quality-variety balance):
1. German: Good structural divergence from English
2. French: Moderate divergence, high translation quality
3. Russian: High divergence, moderate translation quality
4. Japanese: Very high divergence (word order change)
5. Multiple languages: Use each as intermediate for maximum diversity
```

### 16.3 Noising Methods

Add controlled noise to training data to improve robustness:

**1. Token masking:** Randomly replace tokens with [MASK] or special tokens.

```python
def mask_tokens(text, mask_rate=0.15, mask_token='[MASK]'):
    tokens = text.split()
    for i in range(len(tokens)):
        if random.random() < mask_rate:
            tokens[i] = mask_token
    return ' '.join(tokens)
```

**2. Token deletion:**
```python
def delete_tokens(text, delete_rate=0.1):
    tokens = text.split()
    tokens = [t for t in tokens if random.random() > delete_rate]
    return ' '.join(tokens)
```

**3. Token permutation (shuffle):**
```python
def shuffle_span(text, span_length=3, shuffle_prob=0.1):
    tokens = text.split()
    for i in range(0, len(tokens) - span_length + 1, span_length):
        if random.random() < shuffle_prob:
            random.shuffle(tokens[i:i+span_length])
    return ' '.join(tokens)
```

**4. Text infilling:** Replace a span with a sentinel token (like T5/FIM training):
```python
def text_infill(text, infill_rate=0.15):
    tokens = text.split()
    # Select spans to remove
    spans = select_random_spans(tokens, infill_rate)
    result = []
    span_idx = 0
    i = 0
    while i < len(tokens):
        if i in spans:
            result.append(f'<extra_id_{span_idx}>')
            i += span_length_of(i)
            span_idx += 1
        else:
            result.append(tokens[i])
            i += 1
    return ' '.join(result), masked_spans_to_predict
```

**5. Synonym replacement:**
```python
import nltk

def synonym_replace(text, replace_prob=0.2):
    tokens = nltk.word_tokenize(text)
    new_tokens = []
    for token in tokens:
        if random.random() < replace_prob:
            syns = nltk.corpus.wordnet.synsets(token)
            if syns:
                lemma = random.choice(syns).lemmas()[0].name()
                if lemma != token:
                    new_tokens.append(lemma.replace('_', ' '))
                    continue
        new_tokens.append(token)
    return ' '.join(new_tokens)
```

**6. Spelling error injection:**
```python
def add_typos(text, typo_rate=0.05):
    """Inject realistic typos (delete, swap, replace, insert)."""
    chars = list(text)
    for i in range(len(chars)):
        if random.random() < typo_rate:
            typo_type = random.choice(['delete', 'swap', 'replace', 'insert'])
            if typo_type == 'delete':
                chars[i] = ''
            elif typo_type == 'swap' and i < len(chars) - 2:
                chars[i], chars[i+1] = chars[i+1], chars[i]
            elif typo_type == 'replace':
                chars[i] = random.choice('abcdefghijklmnopqrstuvwxyz')
            elif typo_type == 'insert':
                chars.insert(i, random.choice('abcdefghijklmnopqrstuvwxyz'))
    return ''.join(chars)
```

### 16.4 Application of Augmentation in LLM Training

**When to use data augmentation:**

| Augmentation Type | Use Case | Impact |
|---|---|---|
| Paraphrasing | Instruction diversity, robustness | +1-3% on unseen tasks |
| Back-translation | Data quantity, NLP benchmarks | 1.5-2× effective data size |
| Token masking | Robustness to noisy input | Improves adversarial robustness |
| Spelling errors | Production robustness | Reduces sensitivity to typos |
| Synonym replacement | Domain adaptation | Moderate improvement |
| Span shuffling | Long-range dependency learning | Marginal in LLMs |

**Practical guidance:**
- Use augmentation during **post-training** (SFT, alignment) more than pre-training
- For pre-training, the web data itself provides enough natural variation
- For instruction tuning, template diversity (paraphrasing) is the most impactful augmentation
- For code LLMs, back-translation between code and natural language is valuable
- Avoid augmentation that changes the answer for classification/verification tasks

---

## 17. Further Reading and References

### Survey and Overview Papers

1. Zhao, W. X. et al. (2023). "A Survey of Large Language Models." arXiv:2303.18223.
2. Yao, Y. et al. (2024). "A Survey on Large Language Model (LLM) Fine-tuning Techniques." arXiv:2408.10706.
3. Han, X. et al. (2024). "A Survey of Data Augmentation for Large Language Models." arXiv:2402.10780.
4. Bai, Y. et al. (2024). "Alignment of Large Language Models: A Survey." arXiv:2409.00116.

### Foundational Papers Referenced

**Pre-training Objectives**
- BERT: Devlin et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers"
- GPT: Radford et al. (2018-2020). GPT-1/2/3 series
- T5: Raffel et al. (2020). "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
- UL2: Tay et al. (2022). "UL2: Unifying Language Learning Paradigms"
- FIM: Bavarian et al. (2022). "Efficient Training of Language Models to Fill in the Middle"

**Data Pipeline**
- C4: Raffel et al. (2020). C4 dataset
- The Pile: Gao et al. (2020). "The Pile: An 800GB Dataset of Diverse Text for Language Modeling"
- RedPajama: Together Computer (2023). "RedPajama: an Open Dataset for Training Large Language Models"
- FineWeb: Penedo et al. (2024). "The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale"
- DCLM: Li et al. (2024). "DataComp-LM: In Search of the Next Generation of Training Sets for Language Models"
- MinHash: Broder (1997). "On the Resemblance and Containment of Documents"
- SimHash: Charikar (2002). "Similarity Estimation Techniques from Rounding Algorithms"

**Scaling Laws**
- Kaplan et al. (2020). "Scaling Laws for Neural Language Models"
- Hoffmann et al. (2022). "Training Compute-Optimal Large Language Models" (Chinchilla)
- Michaud et al. (2023). "The Quantization Model of Neural Scaling"

**Fine-tuning**
- LoRA: Hu et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models"
- QLoRA: Dettmers et al. (2023). "QLoRA: Efficient Finetuning of Quantized Language Models"
- DoRA: Liu et al. (2024). "DoRA: Weight-Decomposed Low-Rank Adaptation"
- AdaLoRA: Zhang et al. (2023). "AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning"
- IA³: Liu et al. (2022). "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper through Scaling"
- Prefix-Tuning: Li & Liang (2021). "Prefix-Tuning: Optimizing Continuous Prompts for Generation"
- Prompt-Tuning: Lester et al. (2021). "The Power of Scale for Parameter-Efficient Prompt Tuning"
- P-Tuning: Liu et al. (2022). "P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Across Scales and Tasks"

**RLHF and Alignment**
- RLHF Paper: Christiano et al. (2017). "Deep Reinforcement Learning from Human Preferences"
- InstructGPT: Ouyang et al. (2022). "Training Language Models to Follow Instructions with Human Feedback"
- DPO: Rafailov et al. (2023). "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
- GRPO: Shao et al. (2024). "DeepSeekMath: Pushing the Limits of Mathematical Reasoning"
- KTO: Ethayarajh et al. (2024). "KTO: Model Alignment as Prospect Theoretic Optimization"
- ORPO: Hong et al. (2024). "ORPO: Monolithic Preference Optimization without Reference Model"
- SimPO: Meng et al. (2024). "SimPO: Simple Preference Optimization with a Reference-Free Reward"
- Constitutional AI: Bai et al. (2022). "Constitutional AI: Harmlessness from AI Feedback"
- RLAIF: Bai et al. (2022). "Training a Helpful and Harmless Assistant from Human Feedback"

**Instruction Tuning**
- FLAN: Wei et al. (2022). "Finetuned Language Models Are Zero-Shot Learners"
- FLAN-T5: Chung et al. (2022). "Scaling Instruction-Finetuned Language Models"
- Super-NaturalInstructions: Wang et al. (2022). "Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks"

**Synthetic Data**
- Self-Instruct: Wang et al. (2022). "Self-Instruct: Aligning Language Model with Self Generated Instructions"
- Evol-Instruct: Xu et al. (2023). "WizardLM: Empowering Large Language Models to Follow Complex Instructions"
- WizardCoder: Luo et al. (2023). "WizardCoder: Empowering Code Large Language Models with Evol-Instruct"

**Data Augmentation**
- Back-Translation: Sennrich et al. (2016). "Improving Neural Machine Translation Models with Monolingual Data"
- Text Infilling: Lewis et al. (2020). "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation"

---

*This document is part of the AI Base Knowledge series. For related topics, see 01-LLM-and-AI-Models.md (Transformer architecture, inference), 02-Machine-Learning.md (foundational ML concepts), and 03-Deep-Learning.md (deep learning architectures).*
