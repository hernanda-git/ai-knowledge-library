# Natural Language Processing: Foundations and Techniques

## Table of Contents
1. [Introduction](#1-introduction)
2. [Text Preprocessing](#2-preprocessing)
3. [Word Representations](#3-word-reps)
4. [Syntactic Analysis](#4-syntax)
5. [Semantic Analysis](#5-semantics)
6. [Sequence Labeling](#6-labeling)
7. [Text Classification](#7-classification)
8. [Information Extraction](#8-ie)
9. [Machine Translation](#9-mt)
10. [Summarization](#10-summarization)
11. [Question Answering](#11-qa)
12. [Dialogue Systems](#12-dialogue)
13. [Attention and Transformers in NLP](#13-attention)
14. [Classical vs Modern NLP](#14-classical-modern)
14a. [NLP Evaluation and Benchmarking](#14a-nlp-evaluation-and-benchmarking)
14b. [Multilingual and Cross-lingual NLP](#14b-multilingual-and-cross-lingual-nlp)
15. [Production NLP Deployment](#15-production-nlp-deployment)
16. [Cross-References](#16-cross-references)

---

## 1. Introduction

NLP is the field of computational linguistics — teaching computers to understand, generate, and manipulate human language. While LLMs have revolutionized NLP, the foundational techniques (tokenization, parsing, NER, dependency analysis) remain essential for understanding what LLMs do and for building production systems.

### The NLP Pipeline (Classical)
```
Raw Text → Sentence Segmentation → Tokenization → Normalization → 
POS Tagging → Parsing → Semantic Analysis → Application
```

### Modern vs Classical NLP Differences

| Aspect | Classical NLP | Modern NLP (LLM era) |
|--------|:-------------:|:--------------------:|
| Architecture | Feature engineering + shallow models | End-to-end deep learning |
| Supervision | Task-specific labeled data | Pre-training + few-shot |
| Representation | Sparse (one-hot, TF-IDF) | Dense embeddings |
| Context | Fixed window | Full context (up to 1M+ tokens) |
| Tasks | One model per task | One model for all tasks |
| Training | Train from scratch | Fine-tune pre-trained model |

---

## 2. Text Preprocessing

### 2.1 Tokenization
Tokenization splits text into tokens (words, subwords, or characters).

| Tokenizer | Method | Vocabulary | OOV Handling | Used By |
|-----------|--------|:----------:|:------------:|---------|
| **Word-level** | Split by whitespace/punctuation | 100K+ | No | Classical NLP |
| **BPE** (Byte-Pair Encoding) | Merge most frequent character pairs | 8K-50K | Subword decomposition | GPT, RoBERTa |
| **WordPiece** | Merge based on likelihood | 30K | Subword | BERT, DistilBERT |
| **SentencePiece** | Unigram LM + BPE (no pre-tokenization) | 8K-32K | Subword | T5, LLaMA, Gemma |
| **Unigram** | Probabilistic subword segmentation | 8K-50K | Subword | XLNet, ALBERT |
| **tiktoken** | BPE (fast, Rust-based) | 100K | Subword | GPT-4, o1, Claude |

See: [02-LLMs/03-Tokenization.md] for in-depth coverage.

### 2.2 Normalization

| Technique | Description | Example | When to Use |
|-----------|-------------|---------|:-----------:|
| **Lowercasing** | Convert all text to lowercase | "US" vs "us" = same | General text |
| **Stemming** | Remove affixes aggressively | "running" → "run", "better" → "better" | Search, IR |
| **Lemmatization** | Morphological analysis with dictionary | "running" → "run", "better" → "good", "was" → "be" | Semantic analysis |
| **Stop word removal** | Remove high-frequency words | "the", "a", "is", "are" | TF-IDF, IR |
| **Punctuation removal** | Strip punctuation | "Hello!" → "Hello" | Simple classifiers |
| **Unicode normalization** | NFC/NFD normalization | "é" → "e" + combining accent | Multilingual text |

```python
import spacy
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Stemming vs Lemmatization
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = ["running", "better", "was", "studies", "flies"]
for w in words:
    print(f"{w:10} → stem: {stemmer.stem(w):10} lemma: {lemmatizer.lemmatize(w, pos='v'):10}")
# running    → stem: run       lemma: run
# better     → stem: better    lemma: run       (lemmatizer needs POS tag)
# was        → stem: wa        lemma: be
# studies    → stem: studi     lemma: study
```

### 2.3 Sentence Segmentation

| Method | Approach | Accuracy |
|--------|----------|:--------:|
| **Rule-based** | Regex on `. ! ?` + capitalization | ~95% |
| **Punkt** (NLTK) | Unsupervised boundary detection | ~97% |
| **spaCy sentencizer** | Dependency parser-based | ~99% |
| **Neural model** | Transformer SBD | ~99.5% |

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Dr. Smith went to Washington. He met with Mr. Jones, who asked: 'What about A.I.?'")
for sent in doc.sents:
    print(sent.text)
```

---

## 3. Word Representations

### 3.1 Evolution of Word Embeddings

| Era | Model | Year | Dimensions | Context | Key Innovation |
|:---:|-------|:----:|:----------:|:-------:|---------------|
| Sparse | One-hot | — | V (=100K) | None | Simple baseline |
| Sparse | TF-IDF | 1972 | V | Document | Frequency weighting |
| Static | Word2Vec (CBOW/Skip-gram) | 2013 | 100-300 | Local window | Semantic vector arithmetic |
| Static | GloVe | 2014 | 100-300 | Global co-occurrence | Matrix factorization |
| Static | fastText | 2017 | 100-300 | Subword | OOV handling |
| Contextual | ELMo | 2018 | 1024 | BiLSTM over sentence | Context-dependent embeddings |
| Contextual | BERT | 2019 | 768/1024 | Transformer (full) | Deeply bidirectional |
| Multi-task | InstructGPT/GPT-3 | 2020 | 12288 | Full context | Task-agnostic embeddings |

### 3.2 Word2Vec Details (Mikolov et al., 2013)

- **CBOW:** Predict masked word from surrounding context words
- **Skip-gram:** Predict surrounding context words from target word
- **Training:** Negative sampling (NCE loss) or hierarchical softmax

```python
# Word2Vec analogies
from gensim.models import Word2Vec

model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=4)
# king - man + woman = queen
result = model.most_similar(positive=['king', 'woman'], negative=['man'])
print(result[0])  # ('queen', 0.78)

# Learning structure: vector("Paris") - vector("France") + vector("Italy") ≈ vector("Rome")
```

**Properties of learned embeddings:**
- Analogies via vector arithmetic (king - man + woman ≈ queen)
- Semantic similarity captured (cat ≈ kitten > dog > car)
- Syntactic relationships (walked - walk ≈ ran - run)

### 3.3 Contextual Embeddings (BERT, 2019)
BERT generates different embeddings for the same word in different contexts:
```
"I went to the river bank."  → "bank" = financial_institution_embedding
"I deposited money at the bank." → "bank" = river_edge_embedding
```

---

## 4. Syntactic Analysis

### 4.1 Parts of Speech (POS) Tagging

Each word assigned a grammatical category.

| Tag Set | Size | Example Tags |
|---------|:----:|--------------|
| **Penn Treebank** | 45-50 | NN (noun), VB (verb), JJ (adjective), RB (adverb) |
| **Universal Dependencies** | 17 | NOUN, VERB, ADJ, ADV, ADP, DET |
| **Brown Corpus** | 87 | NN (singular noun), NNS (plural noun), VBZ (3rd person verb) |

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The quick brown fox jumps over the lazy dog")
for token in doc:
    print(f"{token.text:8} → POS: {token.pos_:5} → Tag: {token.tag_:5} → Dep: {token.dep_}")
# The      → POS: DET   → Tag: DT    → Dep: det
# quick    → POS: ADJ   → Tag: JJ    → Dep: amod
# brown    → POS: ADJ   → Tag: JJ    → Dep: amod
# fox      → POS: NOUN  → Tag: NN    → Dep: nsubj
# jumps    → POS: VERB  → Tag: VBZ   → Dep: ROOT
```

**Accuracy:** >97% with modern taggers (HMM → CRF → BiLSTM → Transformer).

### 4.2 Constituency Parsing
Tree structure representing phrase structure (context-free grammar):

```
[S [NP [D The] [N cat]] [VP [V sat] [PP [P on] [NP [D the] [N mat]]]]
```

| Algorithm | Type | Complexity | Accuracy |
|-----------|:----:|:----------:|:--------:|
| CKY | Grammar-based (CFG) | O(n³n|G|) | Grammar-dependent |
| Earley | Chart parsing | O(n³) worst | Grammar-dependent |
| Stanford PCFG | Probabilistic CFG | O(n³) | ~90% F1 |
| Self-attentive parser | Neural span prediction | O(n²) | >95% F1 (Kitaev & Klein, 2018) |

### 4.3 Dependency Parsing
Directed edges between words: head-dependent relations.

```
        root
         ↓
      (bought)
       ↙     ↘
    nsubj     dobj → det
    (John)   (car) → (the)
```

| Method | Type | Speed | Accuracy (UAS) |
|--------|:----:|:-----:|:--------------:|
| Transition-based (arc-standard) | Greedy, shift-reduce | Fast | ~93% |
| Graph-based (Eisner, biaffine) | Maximum spanning tree | Slower | >95% |
| BiLSTM + biaffine | Neural (Dozat & Manning, 2017) | Medium | >96% |

---

## 5. Semantic Analysis

### 5.1 Semantic Role Labeling (SRL)
Identify predicate-argument structure: who did what to whom.

```
Arg0=buyer(John), Arg1=thing(car), verb=buy
```

**Frameworks:**
| Framework | PropBank | FrameNet |
|-----------|:--------:|:--------:|
| Arguments | Arg0-Arg5, ArgM-LOC, ArgM-TMP | Frame elements (Buyer, Goods, Seller) |
| Coverage | 4,500 predicates | 1,200 frames |
| Granularity | Role types (coarse) | Frame-specific (fine) |

### 5.2 Named Entity Recognition (NER)
Identify and classify named entities.

| Entity Type | Examples |
|-------------|----------|
| Person | John Smith, Elon Musk, Taylor Swift |
| Organization | Google, OpenAI, Stanford University |
| Location | New York City, Mount Everest, Pacific Ocean |
| Date | June 5, 2026, last Tuesday, Q3 2025 |
| Money | $100 million, €50, ¥10,000 |
| Percentage | 25%, 3.5x, half |
| Product | iPhone 15, Windows 11, GPT-4o |
| Event | Olympics, WW3 (hypothetical), Super Bowl |

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple CEO Tim Cook announced iPhone 17 on September 9, 2026 in Cupertino.")
for ent in doc.ents:
    print(f"{ent.text:30} → {ent.label_:10} ({ent.kb_id_})")
# Apple                        → ORG
# Tim Cook                     → PERSON
# iPhone 17                    → PRODUCT
# September 9, 2026            → DATE
# Cupertino                    → GPE
```

**Models:** BiLSTM-CRF (state-of-the-art before transformers), BERT-NER (current SOTA, F1 ~93-95%), GPT few-shot (variable).

### 5.3 Coreference Resolution
Resolve which noun phrases refer to the same entity:

> "**John** went to the store. **He** bought milk. **John**'s wife called **him**."

All refer to the same person (John).

| Model | Approach | F1 |
|-------|----------|:--:|
| Stanford Deterministic | Rule-based | ~60% |
| Mention-ranking | Learn ranking over candidates | ~70% |
| End-to-end (Lee et al., 2017) | Span-based neural | ~75% |
| Longformer coref | Transformer-based | ~85% |

---

## 6. Sequence Labeling

Common NLP tasks framed as assigning a label to each token:

| Task | Labels | Example |
|------|--------|---------|
| POS tagging | NN, VB, JJ, ... | The/DET cat/NN sat/VBD on/IN the/DET mat/NN |
| NER | B-PER, I-PER, B-ORG, O, ... | [B-ORG Apple] CEO [B-PER Tim] [I-PER Cook] |
| Chunking | B-NP, I-NP, B-VP, ... | [B-NP The cat] [B-VP sat] [B-PP on] [B-NP the mat] |
| Sentiment | POS, NEG, NEU | I/B-POS love/I-POS this/O |

**Models:**
| Model | Era | Strengths | Weaknesses |
|-------|:---:|-----------|------------|
| HMM (generative) | 1990s | Simple, fast | Independence assumption, lower accuracy |
| CRF (discriminative) | 2000s | Exact inference, global features | Hand-engineered features |
| BiLSTM-CRF | 2016 | Deep learning, no feature eng | Sequential (slow), limited context |
| Transformer (BERT) | 2019 | Full context, SOTA | Computationally heavy |

---

## 7. Text Classification

### 7.1 Approaches Compared

| Method | Data Efficiency | Performance | Interpretability | Training Speed |
|--------|:---------------:|:-----------:|:----------------:|:-------------:|
| Naive Bayes | ★★★★★ | ★★☆ | ★★★★★ | ★★★★★ |
| SVM + features | ★★★★☆ | ★★★☆ | ★★★★ | ★★★★ |
| CNN (TextCNN) | ★★★☆ | ★★★★ | ★★★ | ★★★★ |
| BiLSTM | ★★★☆ | ★★★★ | ★★☆ | ★★★☆ |
| BERT fine-tune | ★★☆ | ★★★★★ | ★★☆ | ★★☆ |
| GPT few-shot | ★☆☆ | ★★★★☆ | ★☆ | N/A (inference only) |

### 7.2 Multi-label vs Multi-class

| Aspect | Multi-class | Multi-label |
|--------|:-----------:|:-----------:|
| Classes per sample | Exactly 1 | 0 to N |
| Output layer | Softmax (k) | Sigmoid (k) |
| Loss | Cross-entropy | Binary cross-entropy (per class) |
| Example | Sentiment: pos/neg/neu | Topics: tech+sports+finance |
| Evaluation | Accuracy, F1 macro | Precision@k, micro-F1 |

```python
import torch.nn as nn

# Multi-class classifier (sentiment)
classifier = nn.Sequential(
    nn.Linear(768, 256),
    nn.ReLU(),
    nn.Linear(256, 3),  # pos, neg, neu
    nn.Softmax(dim=-1)
)

# Multi-label classifier (topics)
multilabel = nn.Sequential(
    nn.Linear(768, 256),
    nn.ReLU(),
    nn.Linear(256, 10),  # 10 topics
    nn.Sigmoid()          # independent probability per class
)
```

---

## 8. Information Extraction

### 8.1 Relation Extraction
Extract structured relationships from text.

```
"Apple CEO Tim Cook" → (Person: Tim Cook, CEO_of, Organization: Apple)
"born in New York in 1990" → (Person: John, place_of_birth: New York, year_of_birth: 1990)
```

| Approach | Description | F1 |
|----------|-------------|:--:|
| Template-based | Hand-crafted patterns | ~50% |
| Distant supervision | Align with KB (Freebase, Wikidata) | ~65% |
| Fine-tuned BERT | Relation classification on entity pairs | ~90% |
| GPT prompting | "Extract relationships from the following text" | Variable |

### 8.2 Open Information Extraction (Open IE)
Extract relations without pre-defined schema:
```
"Google acquired DeepMind in 2014" → (Google, acquired, DeepMind, in 2014)
```

Advantage: No need for predefined relation types. Useful for knowledge base construction from unstructured text.

### 8.3 Event Extraction
Identify events and their participants (who did what, when, where):
```
"John was born in New York in 1990" 
→ Event: birth, Person: John, Place: New York, Time: 1990
```

**ACE Event Extraction:** 8 event types, 33 subtypes (life, movement, transaction, conflict, contact, personnel, justice, business).

---

## 9. Machine Translation

### Evolution

| Era | Approach | BLEU (WMT En-De) | Training Data |
|:---:|----------|:-----------------:|:-------------:|
| 1960s-1990s | Rule-based (hand-crafted grammar) | — | None |
| 2000s | Statistical SMT (phrase-based) | ~20 | Parallel corpus |
| 2015 | Neural NMT (Seq2Seq + attention) | ~25 | Parallel corpus |
| 2018 | Transformer (Vaswani et al.) | ~30 | Parallel corpus |
| 2020 | Pre-trained NMT (mBART, M2M-100) | ~35 | 100+ languages |
| 2023+ | LLM-based (zero-shot translation) | ~35+ | Multilingual pre-training |

**Key Metrics:**
| Metric | Type | Description |
|--------|:----:|-------------|
| **BLEU** | n-gram overlap | Precision of n-grams (0-100, higher=better) |
| **COMET** | Neural | Cross-lingual semantic similarity |
| **chrF** | Character-level | Character n-gram F-score |
| **TER** | Edit distance | Translation edit rate (lower=better) |

```python
# Example: BLEU score calculation
from nltk.translate.bleu_score import sentence_bleu

reference = "the cat sat on the mat".split()
candidate = "the cat is on the mat".split()
score = sentence_bleu([reference], candidate)
print(f"BLEU: {score:.3f}")  # ~0.6
```

---

## 10. Summarization

### 10.1 Extractive vs Abstractive

| Aspect | Extractive | Abstractive |
|--------|:----------:|:-----------:|
| Output | Select existing sentences | Generate new text |
| Faithfulness | Guaranteed (copies from source) | Risk of hallucination |
| Coherence | Lower (disconnected sentences) | Higher (fluent narrative) |
| Compression | Fixed (based on selection ratio) | Flexible |
| Key models | TextRank, BERTSum | BART, Pegasus, T5, GPT-4 |

### 10.2 Key Models

| Model | Year | Type | ROUGE-L (CNN/DM) |
|-------|:----:|:----:|:----------------:|
| TextRank | 2004 | Extractive | 31.2 |
| BERTSum | 2019 | Extractive | 41.0 |
| BART | 2019 | Abstractive | 42.6 |
| Pegasus | 2019 | Abstractive | 44.2 |
| T5-11B | 2020 | Abstractive | 43.9 |
| Longformer/LED | 2020 | Long-doc abstractive | 42.2 (on long docs) |
| GPT-4/Claude 3 | 2023+ | Abstractive | >45 (zero-shot) |

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
text = "Long article text... (1024 tokens)"
summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
print(summary[0]['summary_text'])
```

---

## 11. Question Answering

### 11.1 QA Types

| Type | Input | Output | Dataset |
|------|:-----:|:------:|---------|
| **Extractive** | Question + context | Answer span in text | SQuAD 2.0 |
| **Abstractive** | Question + context | Generated answer | NarrativeQA |
| **Multiple-choice** | Question + options | Selected option | RACE, MMLU |
| **Open-domain** | Question only | Retrieved context → answer | Natural Questions |
| **Table QA** | Question + table | Cell or aggregation | WikiTableQuestions |
| **Visual QA** | Question + image | Text answer | VQA 2.0 |

### 11.2 Modern QA with RAG

Modern QA systems use **Retrieval-Augmented Generation** (RAG) to ground answers in factual documents. See: [04-RAG/01-RAG-Architectures.md].

```python
# Simplified RAG QA pipeline
def rag_qa(question, corpus, embedder, llm, top_k=5):
    # 1. Embed question
    q_emb = embedder.encode(question)
    
    # 2. Retrieve relevant documents
    doc_embs = embedder.encode(corpus)
    scores = cosine_similarity(q_emb, doc_embs)
    top_docs = [corpus[i] for i in scores.argsort()[-top_k:]]
    
    # 3. Generate answer with context
    context = "\n\n".join(top_docs)
    prompt = f"Answer the question based on the context:\n\nContext: {context}\n\nQuestion: {question}"
    return llm.generate(prompt)
```

---

## 12. Dialogue Systems

### 12.1 Types of Dialogue Systems

| Type | Goal | Architecture | Examples |
|------|:----:|:-------------|----------|
| **Task-oriented** | Complete specific goal (book flight) | Intent detection + slot filling + dialog state tracking | Siri, Alexa, Google Assistant |
| **Chit-chat** | Open-ended conversation | Seq2Seq, transformer decoder | Eliza, Meena, BlenderBot |
| **LLM-powered** | General-purpose assistant | Instruction-tuned LLM | ChatGPT, Claude, Gemini |

### 12.2 Task-Oriented Dialogue Components

| Component | Function | Example |
|-----------|----------|---------|
| **NLU** | Intent detection + slot filling | Intent: BookFlight, Slots: {from: "NYC", to: "London", date: "tomorrow"} |
| **DST** (Dialog State Tracking) | Track conversation state | Current booking: NYC→London, needs: return date |
| **Policy** | Decide next action | Ask for missing slot → confirm → execute |
| **NLG** | Generate natural response | "I found a flight from New York to London tomorrow at 9 AM. Shall I book it?" |

### 12.3 Dialogue Evaluation

| Metric | What It Measures |
|--------|-----------------|
| **Task success rate** | Did the user achieve their goal? |
| **Slot error rate** | How many slots were filled incorrectly? |
| **Satisfaction** | User ratings, survey scores |
| **Coherence** | Is the response on-topic? |
| **Engagement** | Turn count, conversation length |

---

## 13. Attention and Transformers in NLP

### 13.1 Attention Mechanism (Bahdanau et al., 2015)

Before attention, Seq2Seq models used a fixed-size context vector (bottleneck). Attention allows the decoder to look at all source positions:

```
score(q, k_i) = q · k_i                          (dot product)
score(q, k_i) = q · W · k_i                      (bilinear / Luong attention)
score(q, k_i) = v^T · tanh(W[q; k_i])            (additive / Bahdanau attention)
α_i = softmax(score(q, k_i))                      (attention weights)
context = Σ α_i · v_i                             (weighted sum of values)
```

### 13.2 Self-Attention (Transformer, Vaswani et al., 2017)

Attention(Q, K, V) = softmax(QK^T / √d_k) V

```python
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    scores = Q @ K.transpose(-2, -1) / (K.size(-1) ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weights = F.softmax(scores, dim=-1)
    return weights @ V
```

**Multi-Head Attention:** h parallel attention heads, each with different learned projections. Concatenated and projected to final dimension.

### 13.3 Positional Encodings
Transformers have no inherent notion of word order. Solutions:

| Method | Description | Used By |
|--------|-------------|---------|
| **Sinusoidal** | Fixed sine/cosine functions of position | Transformer, BART |
| **Learned absolute** | Trainable position embeddings | BERT, GPT-2 |
| **Relative (T5)** | Learned pairwise distance between positions | T5 |
| **Rotary (RoPE)** | Rotate embeddings by position angle | LLaMA, GPT-NeoX |
| **ALiBi** | Linear bias based on distance | BLOOM, MPT |

### 13.4 From Transformer to LLMs

```
Transformer Encoder               Transformer Decoder
(BERT, RoBERTa)                   (GPT, LLaMA)
     ↓                                  ↓
Encoder-only:                        Decoder-only:
- Bidirectional context              - Left-to-right causal
- Best for: classification,          - Best for: generation,
  NER, QA (span extraction)            summarization, chat
```

**Encoder-Decoder (T5, BART):** Combine both — encode input, then decode output. Best for: translation, summarization, any seq2seq task.

---

## 14. Classical vs Modern NLP

### Per-Task Comparison

| Task | Classical Best | LLM Approach | Best When |
|------|:-------------:|:------------:|-----------|
| POS Tagging | BiLSTM-CRF (97%) | Few-shot or fine-tune | Classical for low-latency |
| NER | RoBERTa-CRF (94% F1) | In-context few-shot | LLM for rare entity types |
| Text Classification | BERT fine-tune (96%) | GPT zero-shot (variable) | Classical for high accuracy |
| Translation | NMT Transformer (BLEU 35) | GPT/Claude zero-shot | LLM for low-resource pairs |
| Summarization | BART/Pegasus (ROUGE 44) | GPT-4/Claude few-shot | LLM for creative summaries |
| Open-domain QA | RAG + reader (F1 75%) | Claude/GPT-4 with web search | LLM for complex reasoning |
| Sentiment Analysis | BERT fine-tune (97%) | GPT zero-shot (92%) | Classical for production |
| Relation Extraction | BERT fine-tune (90% F1) | GPT few-shot (85%) | Classical for structured IE |

### Key Trade-off

**Classical NLP:**
- ✅ Specialized, high accuracy on specific tasks
- ✅ Interpretable feature contributions
- ✅ Low latency, low cost
- ✅ Works with small labeled datasets
- ❌ Requires task-specific models and data

**LLM-based NLP:**
- ✅ One model for all tasks (zero-shot)
- ✅ Handles open-ended, creative, and complex tasks
- ✅ No task-specific training needed
- ❌ Hallucination, expensive, slower
- ❌ Hard to debug and control

**Trend:** Classical NLP handles specialized, low-resource, structured tasks. LLMs handle open-ended, zero-shot, multi-task scenarios. The best production systems often **combine both**.

---

## 14a. NLP Evaluation and Benchmarking

NLP evaluation has evolved from simple overlap metrics to sophisticated frameworks that test models across dozens of capabilities. Understanding the strengths and weaknesses of each benchmark is critical for model selection and research.

### 14a.1 Benchmarks Overview

| Benchmark | Year | Domain | # Tasks | Key Metric | Saturated? | Description |
|:---------:|:----:|:------:|:-------:|:----------:|:----------:|-------------|
| **GLUE** | 2018 | General NLU | 9 | Avg. score | ✅ (2020) | CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE, WNLI |
| **SuperGLUE** | 2019 | Harder NLU | 8 | Avg. score | ✅ (2021) | BoolQ, CB, COPA, MultiRC, ReCoRD, RTE, WiC, WSC |
| **XTREME** | 2020 | Cross-lingual | 9 | Avg. score | ✅ (2023) | 40 languages, 9 tasks spanning classification, QA, retrieval |
| **GEM** | 2021 | Generation | 13 | Task-specific | Partial | Data-to-text, summarisation, dialog, simplification |
| **HELM** | 2022 | Holistic eval | 42 | Multi-metric | No | Stanford's broad evaluation across 7 metrics per scenario |
| **BIG-bench** | 2022 | Diverse reasoning | 204 | Beyond human | No | 204 tasks testing knowledge, reasoning, social bias |
| **MMLU** | 2021 | Knowledge | 57 | Accuracy | ✅ (2024) | 57 subjects from STEM to humanities |
| **MMLU-Pro** | 2024 | Harder knowledge | 57 | Accuracy | Partial | Harder questions, more distractors, 10 options per question |
| **GPQA** | 2023 | Expert science | 448 | Accuracy | No | Graduate-level biology, physics, chemistry |
| **SWE-bench** | 2024 | Code generation | 2,294 | Pass@1 | No | Real GitHub issues with test-based verification |

### 14a.2 Evaluation Metrics Taxonomy

**Lexical Metrics (exact match):**

| Metric | What It Measures | Applicable Tasks | Known Limitations |
|--------|-----------------|:----------------:|-------------------|
| **BLEU** | n-gram precision + brevity penalty | Translation | Favours short outputs, ignores semantics |
| **ROUGE-1/2/L** | n-gram recall & LCS | Summarization | Does not capture factual correctness |
| **METEOR** | Unigram precision/recall + synonym matching | Translation, captioning | Better than BLEU but still surface-level |
| **chrF** | Character n-gram F-score | Translation (morphologically rich) | Language-dependent, ignores order beyond n-grams |
| **TER** | Edit distance (insertions/deletions/substitutions) | Translation | Lower=better; sensitive to reference length |
| **Exact Match (EM)** | Exact string equality | QA, code gen | Binary — no partial credit; harsh |
| **F1 (token overlap)** | Token-level precision/recall F1 | QA, NER | Better than EM but still lexical |

**Neural Metrics:**

```python
# BERTScore — embedding-based evaluation
from bert_score import score

P, R, F1 = score(
    ["The cat sat on the mat."],    # candidate
    ["A cat was sitting on the mat."],  # reference
    lang="en", model_type="microsoft/deberta-xlarge-mnli",
    verbose=True
)
print(f"BERTScore F1: {F1.mean():.4f}")  # ~0.96 for similar meaning
```

| Metric | Type | Correlation with Human | Computation | Use Case |
|--------|:----:|:---------------------:|:-----------:|----------|
| **BERTScore** | Embedding similarity | 0.90–0.95 | Fast (GPU) | Summarization, translation |
| **BLEURT** | Learned evaluation model | 0.92–0.97 | Medium (GPU) | Generation quality |
| **COMET** | Cross-lingual neural metric | 0.85–0.93 | Medium (GPU) | Translation |
| **UniEval** | Multi-dimension evaluator | 0.90–0.95 | Medium (GPU) | NLG quality across dimensions |
| **LLM-as-a-Judge** | LLM prompt-based scoring | 0.80–0.95 | Slow (depends on LLM) | General-purpose, open-ended |
| **Rubric-based (G-Eval)** | LLM scores against rubric criteria | 0.85–0.93 | Slow (depends on LLM) | Structured evaluation |

### 14a.3 Modern Evaluation Best Practices

**1. Use multiple metrics.** No single metric captures all quality dimensions. A robust evaluation suite includes lexical, neural, and human metrics.

**2. Report confidence intervals.** NLP metrics are noisy — report standard deviations and statistical significance (paired bootstrap or approximate randomization).

**3. Test for robustness.** Evaluate on:
- **Adversarial examples** (typos, paraphrases, reorderings)
- **Distribution shift** (different domains, time periods)
- **Subgroup performance** (demographic groups, rare categories)

**4. Use LLM-as-a-Judge carefully.**
| Pitfall | Mitigation |
|---------|------------|
| Position bias (prefers first vs last output) | Swap response order, average scores |
| Verbosity bias (prefers longer outputs) | Normalize for length, use rubrics |
| Self-enhancement bias (prefers own outputs) | Use different judge model than candidate model |
| Calibration issues (scores cluster) | Use pairwise comparison (A vs B) not absolute scores |
| Instruction sensitivity | Test with multiple prompt templates |

**5. Always include human evaluation for critical systems.** Automated metrics correlate imperfectly with human judgment, especially for creative and open-ended tasks.

```python
# Automated evaluation suite for NLP tasks
import evaluate
from typing import List, Dict

class NLPEvaluator:
    """Multi-metric NLP evaluation suite."""
    
    def __init__(self, task: str):
        self.task = task
        self.metrics = {
            "translation": ["bleu", "comet", "chrf"],
            "summarization": ["rouge", "bertscore", "bleurt"],
            "qa": ["exact_match", "f1", "bert_score"],
            "classification": ["accuracy", "f1_macro", "matthews_correlation"],
        }.get(task, ["accuracy"])
    
    def evaluate(self, predictions: List[str], 
                 references: List[str]) -> Dict[str, float]:
        results = {}
        for metric_name in self.metrics:
            metric = evaluate.load(metric_name)
            result = metric.compute(
                predictions=predictions, 
                references=references
            )
            results.update(result)
        return results

# Usage
evaluator = NLPEvaluator("translation")
scores = evaluator.evaluate(
    predictions=["the cat is on the mat"],
    references=["the cat sat on the mat"]
)
print(scores)  # {"bleu": 0.60, "comet": 0.89, "chrf": 0.75}
```

### 14a.4 Benchmark Selection Decision Tree

```
What is my task?
├── Understanding / Classification
│   ├── General NLU → GLUE, SuperGLUE
│   └── Domain-specific → Create custom benchmark
├── Generation
│   ├── Translation → WMT, FLORES-200
│   ├── Summarization → CNN/DailyMail, XSum, arXiv
│   └── Dialogue → DSTC, PersonaChat
├── Code Generation → HumanEval+, MBPP, SWE-bench
├── Reasoning
│   ├── Math → GSM8K, MATH, AIME
│   └── Science → MMLU-Pro, GPQA
└── Safety / Alignment
    ├── Harmlessness → HarmBench, XSTest
    ├── Bias → BBQ, WinoBias, CrowS-Pairs
    └── Honesty → TruthfulQA, HaluEval
```

**Key principle:** Benchmarks are a proxy, not a guarantee. A high GLUE score does not mean your model will perform well in production. Always evaluate in your specific deployment context.

---

## 14b. Multilingual and Cross-lingual NLP

Multilingual NLP addresses the challenge of processing text in hundreds of languages with varying resource availability. Cross-lingual transfer learning enables models trained on high-resource languages (English, Chinese) to generalize to low-resource languages. This section covers architectures, training strategies, and practical considerations for building multilingual NLP systems.

### 14b.1 Cross-Lingual Transfer Learning Approaches

| Approach | Description | Supervision Required | Performance | Examples |
|----------|-------------|:-------------------:|:-----------:|----------|
| **Zero-shot transfer** | Train on source language, evaluate on target | Source-language labels only | 60-80% of supervised | mBERT zero-shot NER |
| **Translate-train** | Machine-translate training data to target languages | MT system + source labels | 80-90% of supervised | Translate English SQuAD to other languages |
| **Translate-test** | Translate input to source language at inference | MT system at inference | 70-85% of supervised | Translate query → English → answer |
| **Cross-lingual fine-tuning** | Fine-tune multilingual model on small target-language data | Small labeled dataset (100-1000 examples) | 85-95% of supervised | XLM-R + target-language NER |
| **Joint training** | Train on multiple languages simultaneously | Labels in multiple languages | Near-supervised for each language | Multilingual BERT, XLM-R |
| **Adapter-based transfer** | Train language-specific adapters on frozen multilingual backbone | Small target-language data | 80-90% of supervised | MAD-X, UberText |

### 14b.2 Multilingual Models Comparison

The choice of multilingual model significantly impacts cross-lingual performance. Key models and their characteristics:

| Model | Languages | Architecture | Size | Key Strength | Best For |
|-------|:---------:|:-----------:|:----:|-------------|----------|
| **mBERT** | 104 | BERT-base | 110M | Broad language coverage, widely studied | General multilingual NLU |
| **XLM-R** (Conneau et al., 2020) | 100 | RoBERTa-base/large | 278M-550M | SOTA cross-lingual transfer, CC-100 corpus | Cross-lingual classification, NER |
| **XLM** (Lample & Conneau, 2019) | 100 | BERT + TLM objective | 250M | Translation Language Modeling | Translation, cross-lingual NLU |
| **mT5** (Xue et al., 2021) | 101 | T5 encoder-decoder | 300M-13B | Generative multilingual tasks | Translation, summarization, QA |
| **mBART** (Liu et al., 2020) | 50 | BART encoder-decoder | 680M | Seq2Seq pre-training for translation | Machine translation |
| **LaBSE** (Feng et al., 2022) | 109 | Dual BERT encoder | 470M | Cross-lingual sentence embeddings | Bitext mining, retrieval |
| **RemBERT** (Chung et al., 2021) | 110 | BERT with decoupled embedding | 561M | Improved cross-lingual transfer | Low-resource languages |
| **NLLB** (Meta, 2022) | 200 | Encoder-decoder MoE | 600M-54.5B | Massive language coverage, translation | Machine translation for 200 languages |

**Key insight:** XLM-R Large is generally the best choice for cross-lingual classification and NER. mT5 Large is preferred for generative tasks. NLLB-200 is unmatched for translation into low-resource languages.

### 14b.3 Zero-Shot Cross-Lingual Transfer

The most practical pattern for multilingual NLP: train on a single high-resource language (usually English) and evaluate on target languages without any target-language training data.

```python
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer
import torch

# Load pre-trained multilingual model
model_name = "xlm-roberta-base"
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
model = XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=3)

# English training data
train_texts_en = ["I loved this movie!", "Terrible experience.", "It was okay."]
train_labels = [2, 0, 1]
inputs = tokenizer(train_texts_en, padding=True, truncation=True, return_tensors="pt")
model(**inputs, labels=torch.tensor(train_labels))

# Zero-shot inference on Chinese, Arabic, Swahili without any training
for lang, text in [("zh", "这部电影太棒了！"), ("ar", "كانت تجربة رهيبة"), ("sw", "Ilikuwa sawa")]:
    inputs = tokenizer(text, return_tensors="pt")
    probs = torch.softmax(model(**inputs).logits, dim=-1)
    print(f"{lang}: sentiment scores: {probs[0].tolist()}")
```

**What enables zero-shot transfer?**
1. **Shared subword vocabulary:** XLM-R uses a 250K-token SentencePiece vocabulary trained on 100 languages — common subwords across languages create a shared representation space.
2. **Cross-lingual alignment:** The pre-training objective (MLM on multilingual data with TLM on translation pairs) forces the model to align representations of semantically equivalent tokens across languages.
3. **Language-agnostic features:** Higher layers learn task-specific patterns (e.g., sentiment direction, entity types) that generalize across languages because the lower layers have been cross-lingually aligned.

**Performance reality:** Zero-shot typically achieves 60-80% of supervised performance on the target language. Performance degrades for languages typologically distant from the source (e.g., English→Chinese is better than English→Swahili for most tasks).

### 14b.4 Code-Switching

Code-switching — the alternation between two or more languages within a single utterance — is a pervasive phenomenon in multilingual communities but poorly handled by most NLP systems.

| Challenge | Example | Impact on NLP |
|-----------|---------|---------------|
| **Mixed vocabulary** | "Can you buy some **chapati** from the **duka**?" (English + Swahili) | Tokenizer splits unfamiliar words; OOV increases |
| **Morphological blending** | "I'm **happi-er** than **khush**" (English + Hindi) | POS tagging fails; SRL confuses argument boundaries |
| **Syntactic divergence** | "**Mimi** like **kula** pizza" (Swahili + English + Swahili) | Dependency parser trained on monolingual treebanks fails |
| **Script mixing** | "I love **人工智能**!" (Latin + CJK) | Tokenizer/embedding alignment breaks |
| **Pragmatic switching** | "**Wallahi**, that's amazing!" (Arabic discourse marker + English) | Sentiment analysis incorrectly weighted |

**Approaches:**
| Method | Description | Data Required |
|--------|-------------|:------------:|
| **Code-switching-aware tokenizers** | Extend BPE vocabulary with code-switched n-grams | CS corpus (e.g., LICS, CALCS datasets) |
| **Multi-task learning** | Jointly train language identification + main task | Language tags per token |
| **Language-agnostic representations** | Use multilingual models with shared embedding space (XLM-R) | Only monolingual data |
| **CS-specific fine-tuning** | Fine-tune on code-switched data | Small CS corpus (100K tokens) |
| **Mixed-language data augmentation** | Generate CS-like data via random word substitution | Monolingual corpora in both languages |

### 14b.5 Practical Recommendations

| Scenario | Recommended Approach | Model |
|----------|--------------------|-------|
| **10+ languages, classification only** | Zero-shot XLM-R | xlm-roberta-large |
| **2-5 languages, high accuracy needed** | Joint fine-tuning on all languages | XLM-R or mT5 |
| **Low-resource language (< 1M speakers)** | Translate-train + adapter | XLM-R + MAD-X adapter |
| **Code-switching expected** | CS-aware fine-tuning | XLM-R + CS corpus |
| **Multilingual generation (translation, summarization)** | mT5 or NLLB | mt5-large or nllb-200-distilled |
| **Sentence similarity / retrieval across languages** | LaBSE | sentence-transformers/LaBSE |
| **Production latency critical** | Distilled XLM-R or mT5-small | xlm-roberta-base, mt5-small |

**Cross-lingual evaluation metrics:**
- **BLEU/COMET** for translation quality
- **Zero-shot F1** for cross-lingual classification
- **XNLI** for cross-lingual NLI (15 languages)
- **XQuAD / MLQA** for cross-lingual QA
- **BUCC / Tatoeba** for bitext mining accuracy
- **Language ID accuracy** for code-switching detection

---

## 15. Production NLP Deployment

### 15.1 Model Selection for Production

| Consideration | Classical Model | LLM-based | Hybrid |
|:-------------|:---------------:|:---------:|:------:|
| **Latency requirement** | <50ms (on CPU) | 500ms-5s (on GPU) | Route simple→classical, complex→LLM |
| **Cost per prediction** | $0.00001-0.001 | $0.001-0.10 | Optimized per task complexity |
| **Data volume** | 10K-1M predictions/day | 100-10K predictions/day | Scale-dependent routing |
| **Accuracy ceiling** | Task-specific SOTA | General capability | Best of both |
| **Labeled data available** | Yes (100-10K samples) | Few-shot / zero-shot | Warm-start with LLM, distill to classical |
| **Interpretability** | High (feature importance, rules) | Low (black box) | Classical for audits, LLM for edge cases |
| **Regulatory requirements** | Easier to certify | Challenging | Use classical for compliance-critical paths |

### 15.2 NLP Pipeline Architecture for Production

A robust production NLP pipeline differs significantly from research notebooks:

```python
from typing import List, Dict, Optional
import asyncio
import logging

logger = logging.getLogger(__name__)

class ProductionNLPPipeline:
    """
    Production-ready NLP pipeline with:
    - Multi-stage processing
    - Graceful degradation
    - Metrics collection
    - Caching layer
    """
    
    def __init__(self, stages: List[Dict], cache_size: int = 10_000):
        self.stages = stages  # [{'name': 'tokenizer', 'model': ...}, ...]
        self.cache = {}       # LRU cache for repeated inputs
        self.metrics = {s['name']: {'calls': 0, 'errors': 0, 'total_time_ms': 0} 
                       for s in stages}
    
    async def process(self, text: str) -> Dict:
        """Process text through the pipeline with error isolation."""
        if text in self.cache:
            return self.cache[text]
        
        result = {"text": text, "stages": {}}
        current_input = text
        
        for stage in self.stages:
            stage_name = stage['name']
            self.metrics[stage_name]['calls'] += 1
            try:
                t0 = asyncio.get_event_loop().time()
                output = await stage['model'].process(current_input)
                elapsed = (asyncio.get_event_loop().time() - t0) * 1000
                self.metrics[stage_name]['total_time_ms'] += elapsed
                result['stages'][stage_name] = output
                current_input = str(output)  # Pass downstream
            except Exception as e:
                self.metrics[stage_name]['errors'] += 1
                logger.error(f"Stage '{stage_name}' failed: {e}")
                # Graceful degradation: skip failed stage
                result['stages'][stage_name] = {"error": str(e)}
                continue
        
        if len(self.cache) < 10_000:
            self.cache[text] = result
        return result
    
    def get_health(self) -> Dict:
        """Return pipeline health metrics for monitoring."""
        return {
            "stages": self.metrics,
            "cache_size": len(self.cache),
            "healthy": all(
                m['errors'] / max(m['calls'], 1) < 0.05 
                for m in self.metrics.values()
            ),
        }
```

### 15.3 Monitoring Production NLP Systems

| Metric | What It Detects | Alert Threshold | Remediation |
|--------|----------------|:---------------:|-------------|
| **p50/p99 latency** | Model degradation or input size drift | p99 > 2× baseline | Scale up, optimize model, or add queue |
| **Error rate (5xx)** | Model or infrastructure issues | >1% of requests | Roll back, fallback model, circuit breaker |
| **Empty/null output rate** | Silent failures (model returns empty string) | >0.5% | Add output validation, default response |
| **Label distribution shift** | Distribution of predictions changes | KL divergence >0.1 from baseline | Retrain with updated data |
| **Input length distribution** | Users sending unusually long/short text | Std dev > 3× baseline | Add preprocessing guardrails |
| **Confidence score drift** | Model becoming less confident | Mean confidence drop >5% | Investigate data shift, retrain |

### 15.4 Common Production Failure Modes

| Failure Mode | Cause | Detection | Mitigation |
|:-------------|:------|:----------|:-----------|
| **Tokenization mismatch** | Training vs inference tokenizer differs | Test with edge cases before deploy | Pin tokenizer version, add integration test |
| **OOV explosion with BPE** | Very long rare-word sequences | Monitor token count per input | Truncate or chunk inputs > max_tokens |
| **Prompt injection in user text** | Users include adversarial instructions | Keyword/pattern detection | Input sanitization, separate user text from instructions |
| **Memory leak from cached embeddings** | Long-running service without eviction | Memory usage trending up over time | LRU eviction, restart periodically |
| **Language mismatch** | Input language not in training data | Language detection on each input | Fallback model per language, or reject gracefully |
| **Version skew** | Model was updated but preprocessing wasn't | Compare hash of preprocessor config | Version lock the full pipeline, test end-to-end |

### 15.5 Cost Optimization Strategies

| Strategy | Savings | Trade-off |
|----------|:-------:|-----------|
| **Distillation** (train small model on LLM outputs) | 10-50× cost reduction | 2-5% accuracy drop possible |
| **Caching** (KV cache for repeated prefixes) | 30-70% on repeated queries | Additional memory for cache |
| **Batching** (process multiple inputs together) | 2-5× throughput increase | Adds latency for first item in batch |
| **Speculative decoding** (draft+verify) | 2-3× faster generation | More complex infrastructure |
| **Adaptive routing** (simple→cheap model, hard→expensive) | 40-60% overall cost reduction | Requires a router classifier |
| **On-device deployment** (CoreML, TFLite, ONNX) | Zero inference cost | Limited model size (≤7B params) |

**Decision framework:** For every production NLP task, ask:
1. *Can a simple model (TF-IDF + logistic regression) achieve 90% of the accuracy?* → Start here
2. *Is latency critical (<100ms)?* → Avoid LLMs; use distilled or classical models
3. *Do labels change frequently?* → Prefer zero-shot LLMs over fine-tuned models
4. *Is this a safety-critical task?* → Add classical post-processing guardrails regardless of model choice

---

## 16. Cross-References

| Reference | Description |
|-----------|-------------|
| [02-LLMs/03-Tokenization.md] | Modern tokenization (BPE, WordPiece, SentencePiece) |
| [02-LLMs/01-Transformer-Architecture.md] | Transformer for NLP (self-attention, multi-head) |
| [04-RAG/01-RAG-Architectures.md] | Retrieval-augmented NLP (RAG pipeline) |
| [01-Foundations/01-LLM-and-AI-Models.md] | Current LLM capabilities |
| [06-Advanced/04-Prompt-Engineering.md] | Prompt patterns for NLP tasks |
| [06-Advanced/03-Evaluation-Benchmarks.md] | NLP benchmarks (GLUE, SuperGLUE, MMLU) |
| [05-Enterprise/01-Enterprise-AI-Deployment.md] | Production deployment for NLP systems |
| [06-Advanced/09-AI-UX-and-Interaction.md] | NLP interaction design patterns |
| [08-Reference/01-Glossary.md] | Key NLP terms |
| [04-RAG/02-Advanced-RAG.md] | Advanced retrieval for NLP pipelines |
| [08-Reference/02-AI-Roadmap.md] | Future directions in multilingual and cross-lingual NLP |

---
*Document version: 2.5 → 3.0 — June 2026 | Tier 2: Gap Fill | Expanded with code examples, attention mechanism, classical vs modern comparison, production trade-offs, NLP evaluation benchmarking section, and §14b Multilingual and Cross-lingual NLP (transfer approaches, model comparison, zero-shot code, code-switching)*
