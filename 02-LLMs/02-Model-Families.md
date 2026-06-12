# Model Families — Comprehensive Reference

> This document provides a deep technical reference on every major Large Language Model family, their architectures, training methodologies, key innovations, and comparative analysis. Written for AI engineers and researchers.

---

## Table of Contents

1. [GPT Family (OpenAI)](#gpt-family-openai)
2. [Llama Family (Meta)](#llama-family-meta)
3. [Claude Family (Anthropic)](#claude-family-anthropic)
4. [Gemini Family (Google DeepMind)](#gemini-family-google-deepmind)
5. [DeepSeek Family (DeepSeek)](#deepseek-family-deepseek)
6. [Mistral Family (Mistral AI)](#mistral-family-mistral-ai)
7. [Qwen Family (Alibaba)](#qwen-family-alibaba)
8. [Other Notable Models](#other-notable-models)
9. [BERT Family (Encoder-Only)](#bert-family-encoder-only)
10. [T5 Family (Encoder-Decoder)](#t5-family-encoder-decoder)
11. [Architectural Paradigms](#architectural-paradigms)
12. [Mixture-of-Experts Deep Dive](#mixture-of-experts-deep-dive)
13. [Comprehensive Comparison](#comprehensive-comparison)

---

## GPT Family (OpenAI)

### GPT-1 (2018) — The Original Transformer Decoder

GPT-1 introduced the **generative pre-training** paradigm: train a language model on unlabeled text, then fine-tune on downstream tasks. Its architecture was a 12-layer, 12-head, 768-hidden-size Transformer decoder with 117M parameters.

**Architecture details:**
- Decoder-only Transformer with masked self-attention (causal masking)
- 12 transformer blocks, each with multi-head self-attention (12 heads, 64-dim per head)
- 768-dimensional hidden states, 3072-dimensional feed-forward intermediate (4× expansion)
- GELU activation (not ReLU — this was an early adoption of GELU)
- Learned positional embeddings (not sinusoidal), max sequence length 512
- LayerNorm applied after each sub-block (post-norm), unlike the original Transformer's pre-norm
- Vocabulary: 40,000 BPE tokens (using Byte-Pair Encoding from GPT-2's precursor work)

**Training:**
- Pre-trained on BooksCorpus (~7,000 unique books, ~800M words)
- Unsupervised objective: standard autoregressive language modeling (predict next token)
- Batch size 64, 100 epochs, Adam optimizer with learning rate 2.5e-4
- Trained on 4 P600 GPUs for ~1 month

**Fine-tuning architecture:**
- For classification: add a linear projection on top of the [CLS] token's final hidden state
- For entailment: concatenate premise and hypothesis with a delimiter token
- For similarity: process both orderings and add their representations
- For QA: concatenate document, question, and options with delimiters
- Only fine-tuned for 3 epochs on each task — remarkably data-efficient

**Key innovation:** Demonstrated that generative pre-training + discriminative fine-tuning could match or exceed task-specific architectures, laying the foundation for the entire LLM paradigm. Achieved state-of-the-art on 9 of 12 NLU benchmarks at the time, including 8.9% absolute improvement on RACE (reasoning dataset).

---

### GPT-2 (2019) — Zero-Shot Learning at Scale

GPT-2 scaled GPT-1's approach dramatically: 1.5B parameters, trained on WebText (40GB of Reddit-curated web pages). The key insight was that language models could perform tasks in a **zero-shot** setting without fine-tuning — just by conditioning on a prompt.

**Architecture (5 model sizes):**

| Model | Parameters | Layers | d_model | Heads |
|-------|-----------|--------|---------|-------|
| Small | 124M | 12 | 768 | 12 |
| Medium | 355M | 24 | 1024 | 16 |
| Large | 774M | 36 | 1280 | 20 |
| XL | 1.5B | 48 | 1600 | 25 |

**Key architectural changes from GPT-1:**
- LayerNorm moved to the **input** of each sub-block (pre-norm), improving training stability
- Scale of residual weights initialized at 1/√N (where N is the number of residual layers) — this normalization prevented activation growth
- Vocabulary expanded to 50,257 tokens (byte-level BPE, with a special "end of text" token)
- Context window increased from 512 to 1024 tokens
- Batch size: 512 (vs GPT-1's 64)
- GELU activation retained

**Byte-Level BPE tokenizer:**
- Unlike GPT-1/SentencePiece which operate on Unicode characters after normalization, GPT-2's tokenizer operates directly on **bytes** (UTF-8 byte sequences)
- This ensures 100% coverage of all Unicode strings without requiring any UNK tokens
- Base vocabulary: all 256 byte values, plus learned BPE merges on top
- No lowercasing or NFKC normalization — preserves all text faithfully
- This design choice meant the tokenizer could handle any string, but at the cost of roughly 4× token count for non-English text

**Zero-shot capability:**
- GPT-2 showed that scaling alone produced emergent abilities without explicit fine-tuning
- On LAMBADA: 55.1% (Small) → 63.2% (Medium) → 69.4% (Large) → **75.1%** (XL) zero-shot accuracy
- On Winograd Schema Challenge: 70.7% zero-shot (comparable to supervised SOTA at the time)
- Machine translation zero-shot: "prompt engineering" by conditioning on "English sentence = French translation" format
- Question answering: conditioning on context + question format produced reasonable answers
- Summarization: conditioning on "TL;DR:" produced extractive summaries

**Controversy:**
- OpenAI initially **withheld** the full 1.5B model due to "concerns about malicious applications" (fake news, spam, impersonation)
- This sparked massive debate about open vs closed AI research
- The full model was eventually released ~9 months later after no major misuse was detected
- The staged release became a template for how frontier models would be handled

---

### GPT-3 (2020) — In-Context Learning

GPT-3 was the true scaling breakthrough: **175B parameters**, trained on **570GB** of text. The key discovery was **in-context learning** — the ability to perform tasks from just a few examples in the prompt, without any gradient updates.

**Architecture:**
- Same decoder-only Transformer architecture as GPT-2
- Pre-norm LayerNorm (same as GPT-2)
- Alternating dense and sparse attention patterns (though later analysis suggests this wasn't critical)
- Sparse attention layers: every 6th layer uses a "sparse" attention pattern to reduce O(n²) cost
- Context window: 2048 tokens (increased from 1024 in GPT-2)
- Batch size: 3.2 million tokens (extremely large)
- Learning rate: cosine decay from 1.0e-4 to 1.0e-5

**Training data (weighted mix):**
- CommonCrawl (filtered): 410B tokens, 60% weight
- WebText2: 19B tokens, 22% weight
- Books1: 12B tokens, 8% weight
- Books2: 55B tokens, 8% weight
- Wikipedia: 3B tokens, 3% weight

**Model sizes (8 variants):**

| Model | Parameters | Layers | d_model | Heads | Batch Size (tokens) |
|-------|-----------|--------|---------|-------|--------------------|
| Ada | 350M | 12 | 768 | 12 | 64K |
| Babbage | 1.3B | 24 | 2048 | 16 | 64K |
| Curie | 6.7B | 32 | 4096 | 32 | 64K |
| Davinci | 175B | 96 | 12288 | 96 | 3.2M |

**In-context learning (ICL):**
- **Zero-shot:** prompt describes the task, model generates answer without examples
- **One-shot:** one example provided in the prompt
- **Few-shot:** k examples (typically 10-100) provided in the prompt
- ICL works because the forward pass propagates information from the examples to the query through the attention mechanism — the model "computes" a learning algorithm in its activations
- Meta-learning interpretation: the pre-training phase trains the model to "learn to learn" from sequences; at inference, the few-shot examples trigger this capability
- Performance scales log-linearly with number of in-context examples

**Key results:**
- TriviaQA: 71.2% zero-shot, 68.0% one-shot, **73.2%** few-shot (approaching SOTA)
- LAMBADA: 86.4% zero-shot (GPT-2 XL was 75.1%)
- SuperGLUE: 72.5% zero-shot, 75.0% one-shot, **82.8%** few-shot
- Arithmetic: 100% accuracy on 2-digit addition (two-shot), 80% on 5-digit (two-shot)
- News article generation: humans could distinguish GPT-3 from human-written text only 52% of the time
- **Weaknesses:** Poor at commonsense reasoning (ParaRel 48.5%), struggles with multi-step reasoning, inconsistent factual accuracy, repetition issues

**Limitations exposed:**
- Sample inefficiency: 175B parameters trained on 570GB still failed at basic logical reasoning
- Lack of factual grounding: no mechanism to distinguish truth from plausible text
- No ability to learn from feedback or correct mistakes within a conversation
- Prompt engineering became critical — small prompt changes could swing accuracy by 20+ points

---

### GPT-3.5 / ChatGPT (2022) — RLHF and Conversational AI

GPT-3.5 was not a single model but a family built on GPT-3 with **code training** and **RLHF** (Reinforcement Learning from Human Feedback). The most famous member is **text-davinci-003** (InstructGPT) and its chat variant **gpt-3.5-turbo** (ChatGPT).

**Codex (August 2021) — The foundation:**
- GPT-3 fine-tuned on 159GB of GitHub code (from 54M public repositories)
- Codex-davinci-002: 12B parameters, trained with log1000 token sampling
- Achieved 28.8% pass@1 and 72.3% pass@100 on HumanEval (Python function completion)
- This code training dramatically improved logical reasoning and instruction-following
- Demonstrated that code is a uniquely valuable training data source because it forces precise, formal reasoning

**InstructGPT / text-davinci-003 (January 2022):**
- Applied **RLHF** to GPT-3 to align model outputs with human preferences
- Three-step process:
  1. **Supervised Fine-Tuning (SFT):** Human labelers wrote demonstration responses for diverse prompts; GPT-3 fine-tuned on 12,000+ demonstrations
  2. **Reward Model Training:** Labelers ranked model outputs (4-9 responses per prompt); a 6.7B reward model trained to predict human preferences
  3. **PPO Reinforcement Learning:** The SFT model was fine-tuned via Proximal Policy Optimization to maximize the reward model's score, with a KL penalty on the SFT model to prevent divergence
- 1.3B parameter InstructGPT was preferred over 175B GPT-3 70% of the time (showing alignment is more important than scale)

**ChatGPT (November 30, 2022):**
- GPT-3.5 fine-tuned specifically for conversational interaction
- Added a **chat format** with distinct roles: system, user, assistant
- Trained on conversations where human labelers played both user and assistant roles
- Reinforcement learning from human feedback applied at the conversation level
- Launched as a free research preview — reached 100M users in 2 months
- Caused a paradigm shift: chat became the primary UI for AI interaction

**Key improvements over GPT-3:**
- Dramatically better instruction following (fewer irrelevant tangents)
- Refusal capability: can decline inappropriate requests
- Reduced toxicity (by ~50% compared to GPT-3)
- Ability to ask clarifying questions
- Multi-turn conversation coherence
- Code understanding and generation

---

### GPT-4 (March 2023) — Multimodal and More Capable

GPT-4 brought **multimodal understanding** (text + images) and dramatically improved reasoning, factuality, and safety.

**Architecture (undisclosed, inferred from analysis):**
- Estimated at 1.8T parameters across 120 layers with ~200B active parameters per inference
- **Mixture-of-Experts (MoE)** architecture with 16 experts (8 active per token)
- Uses a **router network** to select which experts process each token
- Each expert is roughly 111B parameters (feed-forward networks)
- Trained on ~13T tokens (text + code + images)
- Context window: 8,192 tokens (32,768 in extended version)
- Estimated training cost: $100M+ (based on GPU-hours analysis)

**Multimodal capabilities:**
- Vision encoder processes images and outputs a sequence of visual tokens
- These visual tokens are interleaved with text tokens in the transformer's input
- Can understand images, diagrams, screenshots, and handwritten text
- Does **not** generate images (unlike Gemini or GPT-4o later)
- Performance on MMMU (Multimodal Understanding): 75.0% (vs human expert ~90%)

**Performance highlights:**
- Bar exam (Uniform Bar Exam): scored in the 90th percentile (GPT-3.5 was in the 10th)
- SAT: scored 1410/1600 (710 Evidence-Based Reading & Writing, 700 Math)
- AP exams: scored 5 on AP Chemistry, AP Biology, AP Psychology, AP US History, AP Art History
- MMLU (Massive Multitask Language Understanding): 86.4% (GPT-3.5: 70.0%)
- HumanEval (Python coding): 67.0% pass@1
- TruthfulQA: 70% (GPT-3.5: 44%)
- **Known hallucination weakness:** still "hallucinated" facts 15-30% of the time in adversarial evaluation

**Safety improvements:**
- 6-month safety preparation after training completion
- RLHF fine-tuning with safety-focused reward models
- Refusal rate on toxic prompts: 95%+ (vs GPT-3.5's 80%)
- Implemented "rule-based reward signals" for safety-critical categories
- External red-teaming by ~50 experts from diverse fields

---

### GPT-4 Turbo (November 2023) and GPT-4o (May 2024)

**GPT-4 Turbo:**
- Updated knowledge cutoff: April 2024 (from September 2021)
- 128K context window (16× larger than GPT-4)
- 50% cheaper than GPT-4
- Improved instruction following
- JSON mode and function calling improvements
- Reproducible outputs (seed parameter)

**GPT-4o ("omni", May 2024):**
- Truly **multimodal**: natively processes text, images, audio in any combination
- Single end-to-end model (no longer separate voice/vision pipelines)
- Audio understanding: 232 tokens per second processing
- Voice mode with emotional expression, laughter, singing
- Real-time conversational latency (~320ms)
- Vision: can process images, video frames (camera input), screenshots
- MMLU: 88.7%
- Math (MATH benchmark): 76.6%
- Coding (HumanEval): 90.2% pass@1
- 2× faster inference than GPT-4, 50% cheaper
- Multilingual improvements: significant gains in non-English languages
- **Modalities:** text, image, audio input; text, audio output (not video generation)

**Architecture inferences:**
- Still MoE-based, potentially with more experts and finer-grained routing
- Unified multimodal encoder rather than separate encoders + projector
- Cross-attention mechanisms for integrating audio features
- Tremendous optimization for inference latency (likely speculative decoding + KV-cache optimization)

---

### o1 (September 2024) — Reasoning Model

The **o1** model (previously "Strawberry") introduced a new paradigm: **chain-of-thought reasoning at inference time**. Instead of generating the final answer immediately, the model produces a long chain of reasoning tokens internally before the visible response.

**Key innovation — "Thinking" tokens:**
- The model is trained to generate internal reasoning chains (hidden from the user) before producing the final answer
- This allows the model to "think" for longer on hard problems
- Thinking tokens are not visible in the API response — they're consumed internally
- The model learns when to reason deeply vs when to respond quickly (though this is not user-controllable in v1)
- Training used reinforcement learning to learn **when** to reason rather than just **what** to reason

**Performance (o1-preview):**
- AIME (math competition): 56.1% (GPT-4o: 9.4%, 6× improvement)
- International Mathematical Olympiad (IMO) qualifier: 83.3%
- PhD-level Physics (GPQA-diamond): 73.0% (GPT-4o: 53.4%)
- Codeforces: rating 1671 (top 11% of human competitors)
- MMLU: 91.8%
- MATH-500: 94.8% (near-perfect)
- **Limitations:** much slower (seconds per query vs milliseconds), more expensive (~3× GPT-4o), not good at creative writing

**o1-mini (smaller, cheaper):**
- 82% of o1-preview's math performance at 80% lower cost
- Specifically optimized for STEM reasoning (coding, math, science)
- Less capable at general world knowledge

---

### o3 (December 2024) — Next-Generation Reasoning

o3 dramatically advanced the reasoning paradigm from o1.

**Key improvements:**
- More compute allocation for thinking: can "think" for significantly longer on each problem
- Adaptive reasoning: adjusts thinking depth based on problem difficulty (o1 had uniform thinking time)
- **o3-low** and **o3-high** compute settings for controllable reasoning effort
- Better integration of tool use during reasoning (code execution, browsing)
- Record-breaking performance:

| Benchmark | o3 | o1 | GPT-4o |
|-----------|-----|-----|--------|
| ARC-AGI (visual reasoning) | 87.5% | ~30% | ~30% |
| AIME 2024 (math) | 96.7% | 56.1% | 9.4% |
| GPQA Diamond (PhD science) | 87.7% | 73.0% | 53.4% |
| SWE-bench (coding) | 71.7% | 48.9% | ~30% |
| FrontierMath | >25% | ~2% | ~2% |

- **ARC-AGI breakthrough:** The 87.5% on ARC (Abstraction and Reasoning Corpus) was a landmark — no previous AI system had surpassed even 55% on this benchmark designed to be AI-complete
- o3 uses "natural language chain-of-thought" rather than implicit computation — this means its reasoning is (potentially) interpretable and verifiable

---

### GPT-4.1 (April 2025) — The Latest Frontier

GPT-4.1 is OpenAI's most recent GPT-branded model, representing a refinement of the entire GPT lineage.

**Key features:**
- 1M token context window (matching Gemini 1.5)
- Improved factual accuracy with reduced hallucination (estimated 40% reduction vs GPT-4)
- Better multilingual performance, especially on lower-resource languages
- Improved instruction adherence for complex, multi-step tasks
- MMLU: 92.5% (estimated)
- Native tool use with structured outputs
- Pricing: input $2/M, output $8/M (1/10 the price of GPT-4)
- Three variants: GPT-4.1 (full), GPT-4.1-mini, GPT-4.1-nano

**Architecture (speculative):**
- Likely a refined MoE architecture with improved routing
- KV-cache compression for handling 1M context
- Speculative decoding for faster inference
- Potential integration of reasoning capabilities into the base model

---

## Llama Family (Meta)

### Llama 1 (February 2023) — Open Foundation Model

Llama (Large Language Model Meta AI) was Meta's entry into the LLM space, notable for being **open-weight** and demonstrating that smaller models trained on more data could match larger models.

**Philosophy:** "Chinchilla-optimal" training — for a given compute budget, train smaller models on more tokens rather than larger models on fewer tokens.

| Model | Parameters | Training Tokens | Context Length |
|-------|-----------|----------------|----------------|
| Llama-7B | 6.7B | 1.0T | 2048 |
| Llama-13B | 13.0B | 1.0T | 2048 |
| Llama-33B | 32.5B | 1.4T | 2048 |
| Llama-65B | 65.2B | 1.4T | 2048 |

**Architecture (based on GPT-3 with modifications):**
- Decoder-only Transformer with **pre-norm** (RMSNorm instead of LayerNorm)
- **SwiGLU** activation in feed-forward layers (instead of ReLU/GELU)
- **Rotary Position Embeddings (RoPE)** instead of learned absolute positions
- No bias terms in linear layers (saves parameters, improves numerical stability)
- AdamW optimizer with cosine learning rate schedule
- Vocabulary: 32,000 BPE tokens (SentencePiece tokenizer)
- **Key difference from GPT-3:** RoPE + SwiGLU + RMSNorm became the de facto standard architecture for subsequent models (including Llama 2/3, Mistral, Qwen)

**Tokenizer (SentencePiece BPE):**
- 32k vocabulary size
- BPE algorithm with byte fallback (unknown bytes → UTF-8 bytes)
- Normalization: NFKC Unicode normalization
- Pre-tokenization: split on whitespace and punctuation
- No BOS/EOS tokens added during pre-training (added during fine-tuning for chat)

**Training data:**
- Mixture of web pages (CommonCrawl), books, Wikipedia, academic papers, code (GitHub)
- Data deduplication at line, document, and set level
- 1.4T tokens total for 65B model
- Trained on 2,048 A100 GPUs for 21 days (65B model)
- Training efficiency: ~380 tokens/GPU/second

**Results:**
- Llama-13B outperformed GPT-3 (175B) on most benchmarks
- Llama-65B was competitive with Chinchilla-70B and PaLM-540B
- MMLU: 47.4 (7B), 55.8 (13B), 63.4 (65B)
- This demonstrated that open-source models could compete with closed-source at the time

---

### Llama 2 (July 2023) — Chat-Tuned Open Models

Llama 2 built on Llama 1 with improvements in training data, context length, and the addition of **chat-optimized variants** using RLHF.

| Model | Parameters | Training Tokens | Context Length |
|-------|-----------|----------------|----------------|
| Llama-2-7B | 6.7B | 2.0T | 4096 |
| Llama-2-13B | 13.0B | 2.0T | 4096 |
| Llama-2-70B | 69.8B | 2.0T | 4096 |

**Architecture changes from Llama 1:**
- Context length increased from 2048 to **4096** tokens (via RoPE extension)
- **No architectural changes** — same RMSNorm, SwiGLU, RoPE, no bias
- 40% more training data (2.0T tokens vs 1.0-1.4T)
- Higher quality data filtering and deduplication

**Chat template — Llama's Contribution:**
```
<s>[INST] <<SYS>>
{{ system_prompt }}
<</SYS>>

{{ user_message }} [/INST] {{ assistant_message }} </s>
```

- Special tokens: `<s>` (BOS), `</s>` (EOS), `[INST]` (instruction start), `[/INST]` (instruction end), `<<SYS>>` (system prompt markers)
- Multi-turn format:
```
<s>[INST] First user message [/INST] First response </s>
<s>[INST] Second user message [/INST] Second response </s>
```
- This chat template became widely adopted in the open-source ecosystem
- The **system prompt** is optional — if omitted, the `<<SYS>>` block is not included

**RLHF training:**
- **Step 1:** Supervised fine-tuning on 27,540 human-written demonstrations
- **Step 2:** Training two reward models (one for helpfulness, one for safety), each 70B parameters
- **Step 3:** PPO reinforcement learning using the reward models
- **Key innovation:** Ghost attention — during RLHF, the model's attention to safety-critical tokens is modulated to prevent reward hacking

**Safety fine-tuning:**
- Safety-specific data: 135,000 examples focused on adversarial safety
- Context distillation: using a safe model to filter unsafe training data
- Safety reward model trained to prefer helpful but safe responses
- Llama-2-chat models significantly safer than Llama 1 (measured on toxicity benchmarks)

**Performance:**
- Llama-2-70B: MMLU 68.9%, GSM-8K 56.8%, HumanEval 29.9%
- Comparable to GPT-3.5 on many benchmarks
- Slightly behind GPT-4 and Claude 2 at the time

**License:**
- Llama 2 released under a custom commercial license
- Free for commercial use up to 700M MAUs (monthly active users)
- Requires attribution and safety best practices
- This was a major step for open-weight model accessibility

---

### Llama 3 (April 2024) — New Tokenizer and Massive Scale

Llama 3 introduced a completely new tokenizer, scaled to 8B and 70B sizes, with dramatically improved performance.

**Model sizes:**

| Model | Parameters | Layers | d_model | Heads | KV Heads | Training Tokens |
|-------|-----------|--------|---------|-------|---------|----------------|
| Llama-3-8B | 8.03B | 32 | 4096 | 32 | 8 (GQA) | 15T+ |
| Llama-3-70B | 70.6B | 80 | 8192 | 64 | 8 (GQA) | 15T+ |

**New tokenizer (tiktoken-based):**
- Vocabulary: **128,256** tokens (massive increase from Llama 2's 32k)
- Based on OpenAI's **tiktoken** library (GPT-4's tokenizer)
- Byte-level BPE (like GPT-4's cl100k_base)
- **Key differences from GPT-4's tokenizer:**
  - Larger vocabulary for better multilingual coverage
  - Additional tokens for code (single-digit numbers, common code patterns)
  - No special regex pre-tokenization rules (simpler)
- Improved compression ratio:
  - English: ~3.5 chars/token (vs ~4.0 for Llama 2)
  - Code: significant improvement for whitespace-sensitive languages (Python, YAML)
  - Multilingual: 2-3× better than Llama 2 for non-European languages
- Special tokens: `<|begin_of_text|>`, `<|end_of_text|>`, `<|start_header_id|>`, `<|end_header_id|>`

**Architecture changes from Llama 2:**
- **Grouped-Query Attention (GQA)** with 8 KV heads (instead of full multi-head attention)
  - 8B: 32 query heads, 8 KV heads (4:1 ratio)
  - 70B: 64 query heads, 8 KV heads (8:1 ratio)
  - Significantly faster inference, especially for long contexts
- 128k vocabulary (massive increase)
- Training on 15T+ tokens (7.5× more than Llama 2)
- Context length: 8192 tokens

**GQA details:**
- Standard multi-head attention: each head has its own K, V, Q projections
- Multi-query attention (MQA): all heads share one K,V — extreme but quality loss
- GQA: intermediate — heads are grouped, each group shares K,V
- Llama 3 8B: 32 Q heads, 8 KV heads → 4 queries per K,V group
- Llama 3 70B: 64 Q heads, 8 KV heads → 8 queries per K,V group
- Benefit: reduces KV cache size by 4-8×, critical for long-context inference

**Training:**
- 15T+ tokens from web data, code, books, academic sources
- Data filtering with a classifier trained on high-quality references
- 80K A100-80GB GPUs (Meta's in-house cluster)
- Training duration: months
- No RLHF for base models (chat models used SFT + RLHF)

**Performance:**
- Llama-3-70B: MMLU 82.0%, HumanEval 81.7%, GSM-8K 93.0%
- Matched or exceeded GPT-4 on many benchmarks
- Became the de facto standard for open-weight models

---

### Llama 3.1 (July 2024) — 405B Frontier Model

Llama 3.1 introduced the **405B** parameter model, the largest open-weight model ever released, along with 8B and 70B updates.

| Model | Parameters | Context Length | Training Tokens |
|-------|-----------|---------------|----------------|
| Llama-3.1-8B | 8.0B | 128K | 15T+ |
| Llama-3.1-70B | 70.6B | 128K | 15T+ |
| Llama-3.1-405B | 405B | 128K | 15T+ |

**Key innovations:**
- **128K context length** through improved RoPE scaling (YaRN or similar)
- **LLaMA 3.1 405B uses a **dense** transformer (not MoE) — surprising at this scale
  - 126 layers, 16,384 hidden dimension, 128 heads, 8 KV heads
  - ~16T training tokens, 30 million GPU-hours (H100-80GB) to train
  - Trained on 16,000 H100 GPUs (Meta's cluster)
- All models share the same tokenizer (128,256 vocab)
- **Model parallelism:** 405B trained with tensor parallelism (TP=8), pipeline parallelism (PP=16), and data parallelism (DP=64)

**Architecture of 405B:**
- Decoder-only with GQA (8 KV heads, 128 query heads → 16:1 ratio)
- 126 layers (vs 80 for 70B)
- d_model = 16,384 (vs 8,192 for 70B)
- FFN intermediate dimension: 53,248 (4× d_model with SwiGLU reduces to ~2.7× activation)
- Vocabulary: 128,256
- Context: 128K (trained with progressive extension from 8K → 128K)
- No MoE — Meta explicitly chose dense architecture for reliability and ease of use

**Training recipe:**
- Pre-training on 15.6T tokens
- Long-context pre-training: continued training on 800B tokens with 128K context
- SFT on synthetic data generated by Llama 3.1-405B itself (self-improvement loop)
- DPO (Direct Preference Optimization) instead of PPO for alignment
- Model-based data filtering — using a classifier model trained on human preferences

**Performance:**
- Llama-3.1-405B: MMLU 87.3%, HumanEval 89.0%, GSM-8K 96.8%
- Matched GPT-4 on most benchmarks
- Slightly behind GPT-4 Turbo on some reasoning tasks
- Became the strongest open-weight model available

**License:**
- "Llama 3.1 Community License" — permissive for most uses
- Free for commercial use with no MAU cap (unlike Llama 2)
- Requires attribution for derivatives
- Available via download and cloud APIs

---

### Llama 4 (April 2025) — MoE and Multimodal

Llama 4 represents a major architectural shift: **Mixture-of-Experts** and **multimodal** capabilities.

**Model variants:**

| Model | Total Parameters | Active Parameters | Experts | Vision |
|-------|-----------------|-------------------|---------|--------|
| Llama-4-Scout | 109B | 17B | 16 MoE | Yes |
| Llama-4-Maverick | 401B | 48B | 16 MoE | Yes |

**Architecture changes:**
- First Meta model to use **MoE** (Mixture-of-Experts) architecture
- 16 experts with top-2 routing (2 experts active per token)
- Grouped-Query Attention retained
- Context length: Scout supports 10M tokens (with pruning), Maverick supports 1M tokens
- Multimodal: native vision encoder + adapter to language model
- Trained on 30T+ tokens

**Llama 4 Scout (109B total, 17B active):**
- Optimized for deployment — 17B active parameters fits on a single H100 node
- 16 MoE layers, ~6.8B parameters per expert (in FFN layers only)
- Attention layers are dense (not MoE) — only FFNs are expertized
- Context: 10M tokens via **iRoPE** (interspersed RoPE) — a novel position encoding for ultra-long contexts
- Designed for retrieval-augmented generation and long-document tasks

**Llama 4 Maverick (401B total, 48B active):**
- Frontier model for complex reasoning
- 16 MoE layers with fine-grained expert allocation
- Top-2 routing with load balancing loss
- 48B active parameters per token
- Performance competitive with GPT-4o on multimodal benchmarks

**Multimodal integration:**
- Vision encoder: modified ViT (Vision Transformer) with 700M parameters
- Any resolution support — input images are sliced into sub-images, each encoded separately
- Cross-attention layers connect vision features to the language model
- Text + image interleaved input supported
- Training: two-stage — (1) vision-language alignment, (2) full fine-tuning

**iRoPE (interleaved RoPE):**
- Standard RoPE is applied to every token position
- iRoPE applies RoPE to every **other** token, effectively doubling the max position range
- Combined with context pruning and attention sparsity for 10M token support
- Trade-off: slightly reduced precision at very long ranges

---

## Claude Family (Anthropic)

### Claude 1 (March 2023) — HH-RLHF

Claude 1 was Anthropic's first commercial model, built on their research into **Constitutional AI** and **Harmlessness** training.

**Architecture (inferred):**
- Decoder-only Transformer, estimated ~52B parameters
- Context window: 8,000 tokens
- Based on Anthropic's prior research with scaling laws and model sizes (52B was their "standard" size from 2022 research)
- Trained on a large corpus of text, with emphasis on helpfulness and harmlessness

**HH-RLHF (Helpful, Honest, Harmless):**
- The core philosophy: models should be trained to be helpful while actively avoiding harm
- **Constitutional AI (CAI):** Instead of relying solely on human feedback for harmlessness, use a "constitution" of principles to generate self-critique and revision
  1. Model generates a response
  2. Model critiques its own response based on constitutional principles
  3. Model revises the response based on the critique
  4. This creates a dataset of "revised" responses
  5. The model is fine-tuned on this dataset (supervised learning)
- **RLHF with harmlessness reward:** A separate harmlessness reward model trained on human feedback about harmful outputs
- **Gentle cascade:** Multiple rounds of CAI + RLHF refinement

**Claude's personality:**
- Deliberately trained to be **cautious** — would often refuse borderline requests
- Known for being "boringly safe" — preferred to decline than to risk harm
- Trained to express uncertainty and acknowledge limitations
- This cautious approach was both praised (for safety) and criticized (for being too restrictive)

**Performance:**
- Competitive with GPT-3.5 on many benchmarks
- Superior performance on ethical reasoning and safety evaluations
- Less capable than GPT-4 on complex reasoning

---

### Claude 2 (July 2023) — Longer Context and Better Reasoning

**Improvements over Claude 1:**
- Context window: 100K tokens (major advancement at the time)
- Better coding performance (scored 71.2% on Codex HumanEval)
- Improved factual accuracy
- Reduced refusal rate for borderline queries
- Better at creative writing

**Architecture:**
- Still decoder-only, estimated larger than Claude 1 (possibly ~80-100B parameters)
- Optimized for long-context inference
- Used a combination of attention mechanisms for linear-scaling long context

**Performance highlights:**
- GRE reading comprehension: 90th percentile
- GMAT critical reasoning: 89th percentile
- Bar exam: scores varied, but generally above average
- Weakness: math reasoning (GSM-8K ~66%, significantly below GPT-4)

---

### Claude 3 (March 2024) — Haiku, Sonnet, Opus

Claude 3 introduced a **three-tier** model family: Haiku (fast/cheap), Sonnet (balanced), Opus (most capable).

| Model | Description | Speed | Context | Price (input/output per M tokens) |
|-------|------------|-------|---------|----------------------------------|
| Claude 3 Haiku | Fast, cheap, compact | Fastest | 200K | $0.25/$1.25 |
| Claude 3 Sonnet | Balanced performance | Fast | 200K | $3/$15 |
| Claude 3 Opus | Most capable | Moderate | 200K | $15/$75 |

**Common architecture features:**
- 200K token context window (standard across all tiers)
- Improved multilingual support
- Reduced refusal rates (more nuanced understanding of harmful vs harmless)
- JSON mode and tool use support
- Vision capabilities (image understanding)
- **Not multimodal generation** (text output only)

**Claude 3 Opus performance:**
- MMLU: 86.8% (undergraduate knowledge)
- GSM-8K: 95.0% (grade school math)
- HumanEval: 84.9% (coding)
- MATH: 73.7%
- Near-perfect recall in long-context evaluations (Needle-in-a-Haystack)
- Comparable to GPT-4 on most benchmarks, superior on some

**Claude 3 Haiku performance:**
- Designed for 200ms latency responses
- MMLU: 75.9%
- Suitable for simple queries, classification, extraction
- 3× cheaper than Sonnet

**Vision (all Claude 3 models):**
- Image understanding: photos, diagrams, charts, infographics
- Document understanding: PDF, handwritten text
- No image generation

---

### Claude 3.5 (June 2024, October 2024)

**Claude 3.5 Sonnet (June 2024):**
- Major improvement over Claude 3 Sonnet, ~matching Claude 3 Opus
- MMLU: 88.7%
- HumanEval: 92.0% (best in class for coding at release)
- GSM-8K: 96.8%
- MATH: 78.3%
- **Key differentiator:** exceptional at coding and tool use
- 2× faster than Claude 3 Opus
- **Artifacts feature:** generates interactive content (web pages, SVG, diagrams) in a side panel

**Claude 3.5 Haiku (October 2024):**
- Fastest model in the Claude family
- Matched Claude 3 Opus performance on coding benchmarks
- Improved vision capabilities
- MMLU: 88.3% (nearly matching Sonnet)
- Coding (HumanEval): 91.6%
- Price: $0.80/$4 per M tokens

---

### Claude 4 (April 2025) — Opus, Sonnet, Haiku

Claude 4 represents the latest generation, with significant improvements in reasoning, coding, and tool use.

**Claude 4 Opus (flagship):**
- MMLU: 89.3%
- MATH: 86.5%
- HumanEval: 93.7%
- Context: 200K tokens (standard)
- Extended thinking mode (similar to o-series, but implemented differently)
- Superior multilingual performance
- **Computer use:** can control a computer interface (GUI automation)
- Tool use with thousands of parallel tool calls
- Pricing: $15/$75 per M tokens

**Claude 4 Sonnet (balanced):**
- MMLU: 88.5%
- 2× faster than Opus
- Coding: 90.1% HumanEval
- $3/$15 per M tokens

**Claude 4 Haiku (fastest):**
- MMLU: 85.2%
- 300ms average latency
- $0.80/$4 per M tokens

**Extended thinking (Claude 4 Opus):**
- The model can choose to enter an "extended thinking" mode for hard problems
- Unlike o1's hidden thinking tokens, Claude's thinking is transparent — the reasoning is shown to the user
- This allows verification of the model's reasoning process
- The mode is optional and controlled by the API

---

## Gemini Family (Google DeepMind)

### Gemini 1.0 (December 2023) — Natively Multimodal

Gemini was Google's first truly multimodal model, trained from scratch on text, images, audio, and video together.

**Three tiers:**

| Model | Use Case | Parameters (est.) | Context |
|-------|---------|-------------------|---------|
| Gemini Ultra | Most capable | ~1.5T (MoE) | 32K |
| Gemini Pro | Balanced | ~300B (MoE) | 32K |
| Gemini Nano | On-device | ~1.8B / ~3.25B | 8K |

**Natively multimodal training:**
- Trained jointly on **text, images, audio, video, and code**
- Unlike GPT-4V (text-only model + vision encoder added later), Gemini was trained from scratch on all modalities
- This is a key architectural difference: the same model parameters process all modalities
- **Modalities at training time:** interleaved text + image sequences, audio waveforms, video frames + timestamps, code execution traces, PDFs, web pages

**Architecture:**
- MoE-based decoders for Ultra and Pro, dense for Nano
- Multi-Query Attention (MQA) for efficient inference
- **Ultrasound tokenization:** Speech encoded into tokens directly (not via external ASR)
- Video processed as frame sequences (16 FPS sampled to 1 FPS at inference)
- 32K context window (relatively small by 2024 standards)

**Performance (Gemini Ultra):**
- MMLU: 90.04% (first model to exceed 90%)
- HumanEval: 74.4% (worse than GPT-4 at the time, but improved later)
- MATH: 53.2% (GPT-4 was 52.9%)
- HellaSwag: 89.7%
- Multilingual MMLU: SOTA on 20+ languages
- Video understanding: 62.4% on VATEX (video captioning)
- Audio understanding: state-of-the-art on speech recognition benchmarks

**Gemini Nano:**
- Two versions: Nano-1 (1.8B) and Nano-2 (3.25B)
- Architecture: decoder-only with GQA, 4-bit quantizable
- Intended for on-device applications (Pixel phones)
- Can run fully offline

---

### Gemini 1.5 Pro/Flash (February 2024) — 1M Context

Gemini 1.5 was a paradigm shift in **context length**, introducing the first production model with a 1M-token context window.

**Key innovation — **MoE with 1M context****:
- The 1M context was achieved through a **Mixture-of-Experts** architecture with a novel attention mechanism
- **Infini-Attention:** Compressive memory mechanism that combines linear attention with delta rule
  - Standard attention processes each new token against all previous keys
  - Infini-Attention compresses historical context into a fixed-size memory
  - The model can attend to both the compressed memory and local context
  - This enables theoretically infinite context in a computationally tractable way
- 1M tokens reliably processed with ~99.7% "needle-in-haystack" recall

**Architecture:**
- Gemini 1.5 Pro: estimated ~1.5T parameters (MoE)
- Gemini 1.5 Flash: distilled version, optimized for speed
- Both use MoE with the same MoE architecture (different depths/widths)
- Learned position embeddings — extended via linear interpolation for long context

**Training:**
- Trained on "virtually all text, code, images, video, and audio available via web-scale data collection"
- MoE training with load balancing
- Distillation from Pro → Flash

**Perf comparison:**

| Model | MMLU | HumanEval | GSM-8K | MATH |
|-------|------|-----------|--------|------|
| 1.5 Pro | 87.2% | 84.1% | 92.3% | 67.7% |
| 1.5 Flash | 79.6% | 72.0% | 86.9% | 56.8% |

**1M context use cases:**
- Analyze entire codebases
- Process hour-long video with full audio
- Long-document analysis (hundreds of pages)
- Multi-document comparison and synthesis
- Audio context: up to 11 hours of audio (via Gemini's direct audio processing)

---

### Gemini 2.0 Flash (December 2024)

**Key capabilities:**
- **Multimodal output** (text + images + audio)
- Native tool use (Google Search, Maps, code execution)
- Agentic capabilities: can autonomously perform multi-step tasks
- 1M context window (same as 1.5)
- Lower latency than 1.5 Pro
- **Space-time attention:** new attention mechanism for video understanding

**Performance:**
- MMLU: 90.1%
- HumanEval: 86.5%
- GSM-8K: 95.2%
- Video understanding: significantly improved temporal reasoning

**Multimodal generation:**
- Can generate images (text → image)
- Can generate audio (text → speech, music)
- Unified model — one set of parameters for all modalities
- Image generation uses latent diffusion-style approach integrated with the transformer

---

### Gemini 2.5 Pro/Flash (March 2025)

Gemini 2.5 represents the current state-of-the-art from Google DeepMind.

**Gemini 2.5 Pro:**
- **Thinking mode:** like o-series and Claude 4's extended thinking
- MMLU: 91.8%
- MATH: 97.0% (near-perfect)
- Coding (Aider polyglot): top SOTA
- 1M context standard, 2M context available (for Pro)
- Native multimodal input/output
- Agentic: can create and execute plans, use tools autonomously

**Gemini 2.5 Flash:**
- Distilled from Pro
- Lower latency, higher throughput
- Thinking mode available
- MMLU: 88.5%
- Price: $0.15/$0.60 per M tokens (extremely competitive)

**Architecture improvements (speculative):**
- Refined MoE routing for higher active parameter utilization
- Improved KV-cache management for 2M context
- Hybrid attention: local + global sparse + full attention in different layers

---

## DeepSeek Family (DeepSeek)

### DeepSeek (January 2024) — Foundational Model

DeepSeek is a Chinese AI lab that has produced some of the most innovative and cost-effective models.

**DeepSeek LLM (67B, January 2024):**
- Dense decoder-only transformer
- 67B parameters, 95 layers, 8KV heads (GQA)
- Trained on 2T tokens
- MMLU: 71.6%

**Key characteristics:**
- Focus on training efficiency and cost reduction
- Released open-weight under permissive license
- Strong multilingual performance (Chinese + English)

---

### DeepSeek-Coder (January 2024)

**DeepSeek-Coder family:**
- Base model: DeepSeek-LLM (67B) continued pre-training on code
- Multiple sizes: 1.3B, 6.7B, 33B, 67B
- Trained on 2T tokens of code and natural language (87% code, 13% natural language)
- Fill-in-the-Middle (FIM) training objective
- Repository-level data construction

**Key innovation — **FIM (Fill-in-the-Middle)****:
- Standard autoregressive: predict the next token left-to-right
- FIM: code is rearranged into (prefix, middle, suffix) and the model learns to predict middle given prefix and suffix
- This teaches the model bidirectional code understanding while maintaining causal LM training
- DeepSeek-Coder uses a specific FIM format: `<fim_prefix>`, `<fim_suffix>`, `<fim_middle>` tokens

**Performance:**
- DeepSeek-Coder-33B: HumanEval 72.6% pass@1
- DeepSeek-Coder-67B: HumanEval 78.7% pass@1 (competitive with Codex-175B)

---

### DeepSeek-V2 (May 2024) — Multi-Head Latent Attention (MLA)

DeepSeek-V2 introduced **Multi-Head Latent Attention (MLA)** , a groundbreaking architectural innovation for inference efficiency.

**Vsion: 236B total, 21B active per token (MoE).**

**MLA — Multi-Head Latent Attention:**
- **Problem:** Standard MHA/GQA requires storing large KV caches, which is memory-bound during inference. GQA reduces this but at quality cost.
- **Solution:** Instead of storing full K,V vectors per head, project them into a **low-dimensional latent space** and store that instead.
  - Standard: K = W_K * x (d_head per head per layer)
  - MLA: k_latent = W_down * x (compressed), store k_latent, reconstruct K = W_up * k_latent
  - Compression ratio: d_latent << d_head, typically 4-8× compression
- **KV cache reduction:** For a 128K context, KV cache size drops from ~2.4GB (standard) to ~300MB (MLA) per layer
- **Quality preservation:** The latent space captures all information needed; reconstruction adds negligible overhead
- **Training efficiency:** The latent projections are learned end-to-end — no distillation needed

**Why MLA matters:**
- The KV cache is the primary bottleneck for long-context inference
- With MLA, DeepSeek-V2 can serve 128K context with the same memory as 16K context with standard attention
- This directly enables affordable long-context deployment

**DeepSeek-V2 Architecture:**
- 236B total parameters, 21B active (MoE with 8 experts, top-2 routing)
- 128K context window
- MLA + MoE combined
- Trained on 8.1T tokens
- MMLU: 79.7% (competitive with Llama-3-70B at a fraction of the active parameters)

**DeepSeek-V2.5 (September 2024):**
- Improved version of V2
- MMLU: 80.5%
- Better code and math performance

---

### DeepSeek-V3 (December 2024) — 671B MoE Frontier

DeepSeek-V3 scaled the V2 architecture to 671B total parameters with 37B active, trained on 14.8T tokens.

**Architecture:**
- 671B total parameters, 37B active per token
- **MoE with 256 experts, 8 active per token** (fine-grained MoE)
- MLA for attention (retained from V2)
- Multi-token prediction (MTP) — predict 1-3 future tokens at each position
- Multi-head Latent Attention: shares KV across MoE experts for efficiency
- **DeepSeekMoE architecture:** fine-grained experts (more, smaller experts) with shared experts

**Training efficiency:**
- Trained on 14.8T tokens using 2,048 NVIDIA H800 GPUs
- Total training cost: ~$5.576M (significantly less than comparable models)
- 278K GPU-hours (H800-equivalent)
- **Key to cost efficiency:**
  1. MLA reduces KV cache memory, allowing larger batch sizes
  2. Fine-grained MoE with load balancing reduces communication overhead
  3. FP8 mixed precision training
  4. Efficient parallelism strategy (expert parallelism + data parallelism)

**Performance:**
- MMLU: 89.5%
- HumanEval: 92.5%
- GSM-8K: 96.8%
- MATH: 83.9%
- Competitive with GPT-4o on most benchmarks
- Notably: LLaMA 3.1-405B is slightly better on some tasks, but V3 uses 10× fewer active parameters
- **Efficiency frontier:** Best performance-to-compute ratio of any model at release

**Multi-Token Prediction (MTP):**
- Standard LM: predict next token P(t_{i+1} | t_1..t_i)
- MTP: predict n future tokens simultaneously P(t_{i+1},...,t_{i+n} | t_1..t_i)
- Uses independent output heads for each future position
- Training objective: sum of cross-entropy losses for all n predicted tokens
- At inference: only the immediate next token prediction is used (or can be used for speculative decoding)
- Benefit: forces the model to learn longer-range dependencies and more robust representations

---

### DeepSeek-R1 (January 2025) — Reasoning Through RL

DeepSeek-R1 is a pure **reinforcement learning** approach to reasoning, inspired by the o1 paradigm but open-source and using novel training techniques.

**R1-Zero — Pure RL:**
- Base model: DeepSeek-V3-Base
- Training: **GRPO (Group Relative Policy Optimization)**, no supervised data at all
- The model learns to "think" through trial and error — chain-of-thought emerges spontaneously
- Reward signal: correctness of final answer (verifiable tasks like math, code)
- **Spontaneous reasoning behaviors:**
  - Self-verification: "Let me check my answer..."
  - Backtracking: "Wait, that's wrong. Let me reconsider..."
  - Intermediate goal setting: "First, let me compute X, then Y..."
  - Multiple solution attempts within a single response
- **Problems with R1-Zero:** outputs are hard to read (mixing languages, formatting issues), "aha moments" are unpredictable

**R1 — RL + Cold Start:**
1. **Cold-start SFT:** Fine-tune on a small set (~10,000) of high-quality chain-of-thought demonstrations (written by humans or distilled from a frontier model)
2. **Reasoning RL:** Apply GRPO to the cold-start model, focusing on math, code, and science reasoning tasks
3. **Rejection sampling:** Generate many solutions, keep only the correct ones, fine-tune on them
4. **General RL:** Continue RL training on general tasks (writing, QA, summarization) using preference rewards

**R1 distilled versions:**
- DeepSeek released distilled versions: 1.5B, 7B, 8B, 14B, 32B, 70B (based on Qwen/Llama)
- Distillation: use R1's outputs to train smaller models (SFT on R1-generated chain-of-thought)
- DeepSeek-R1-Distill-Qwen-7B: MMLU 55.9%, but MATH 92.8% (GPT-4 level math!)
- This shows that reasoning ability can be efficiently distilled

**R1 Performance:**
- AIME 2024: 79.8% (o1-preview: 56.1%)
- MATH-500: 97.3% (o1-preview: 94.8%)
- MMLU: 90.8%
- HumanEval: 92.7%
- **Cost:** 3-5× cheaper than o1 per query
- Limitations: slower (must generate reasoning chains), optimized for STEM

---

### DeepSeek-Prover (February 2025) — Mathematical Proof

DeepSeek-Prover is specialized for **formal mathematical proof** in Lean and Isabelle.

**Architecture:**
- Based on DeepSeek-V3
- Fine-tuned on mathematical proof corpora
- Special reinforcement learning environment that interacts with proof assistants
- Reward is "proof accepted by the assistant" (verifiable)

**Performance:**
- MiniF2F (formal math benchmark): state-of-the-art, surpassing GPT-4
- Can generate complete proofs for competition math problems
- Novel capability: can search proof trees, backtrack, and explore different proof strategies

---

## Mistral Family (Mistral AI)

### Mistral 7B (September 2023) — Sliding Window Attention

Mistral 7B demonstrated that a well-architected 7B model could outperform larger models like Llama 2 13B.

**Architecture:**
- Decoder-only, 7.3B parameters
- **Sliding Window Attention (SWA):**
  - Each layer's self-attention has a fixed-size window (W = 4096 tokens)
  - Each token can attend to at most W previous tokens
  - With L layers, the effective receptive field is W × L tokens (theoretically, information can propagate through layers)
  - For Mistral 7B (32 layers, W=4096): effective field 128K tokens
  - This is NOT the same as "dilated attention" — each layer attends to its own local window, but deeper layers "see" information from earlier tokens through the hidden states
  - **Benefit:** O(L × W²) instead of O(L × N²) — linear in sequence length
  - **Trade-off:** actual modeling of long-range dependencies is weaker than full attention
- GQA with 8 KV heads (same as Llama 3 later adopted)
- RoPE, SwiGLU, RMSNorm (standard for this generation)
- Vocabulary: 32,000 tokens (SentencePiece BPE, same as Llama 1/2)

**Rolling Buffer KV Cache:**
- Fixed-size KV cache of size W (4096 tokens)
- Old tokens are evicted as new tokens arrive
- Cache is a fixed-size ring buffer
- This enables deterministic memory usage regardless of sequence length

**Pre-fill and Chunking:**
- During pre-fill (processing initial prompt), the model processes the prompt in chunks
- Each chunk computes attention within the chunk and against the KV cache
- For prompts longer than W, chunks are processed sequentially

**Performance:**
- Mistral-7B: MMLU 64.2% (vs Llama-2-13B 54.8%, Llama-1-34B 62.2%)
- Outperformed Llama-2-13B on most benchmarks
- Comparable to Llama-1-34B
- Demonstrated that architectural quality can overcome parameter count

---

### Mixtral 8x7B (December 2023) — Sparse MoE

Mixtral 8x7B was the first **open-source MoE** model that matched or exceeded GPT-3.5 performance.

**Architecture:**
- 46.7B total parameters, 12.9B active per token
- **8 feed-forward experts, top-2 routing**
- Every layer's FFN is replaced by an MoE with 8 experts
- Self-attention, LayerNorm, and embeddings remain dense (shared across experts)
- Each expert is approximately the size of a Mistral 7B FFN (~5.6B parameters per expert)
- Total: 8 × ~5.6B = ~45B in MoE + ~1.7B in dense components

**Routing:**
- Router is a learned linear layer that outputs logits for each expert
- Top-2 experts are selected (by softmax probability)
- The outputs of the selected experts are weighted by their router probabilities
- **Load balancing loss:** auxiliary loss that encourages uniform routing across experts
- **Capacity factor:** each expert can process at most capacity = (tokens_per_batch / num_experts) × capacity_factor tokens. If exceeded, tokens are dropped (passed as residual to next layer)
- Default capacity_factor: 1.25 (allows 25% overload)

**Performance:**
- MMLU: 70.6% (matched GPT-3.5)
- HumanEval: 40.2% (vs GPT-3.5 48.1%)
- GSM-8K: 58.4%
- Faster than Mistral 7B due to fewer active parameters in FFN
- **Cost per token:** similar to a 12.9B model, but total parameters 46.7B

**MoE in MoE (MoeMoE):**
- Mixtral 8x7B uses expert choice routing (routing selects experts, not tokens)
- All tokens in a batch are routed independently

---

### Mixtral 8x22B (April 2024)

**Scaled version:**
- 141B total parameters, ~39B active
- 8 experts, top-2 routing
- Context: 65K tokens (extended via improved RoPE)
- MMLU: 76.8%

**Improvements over 8x7B:**
- Larger expert capacity (wider FFN)
- Better multilingual performance
- Improved reasoning (GSM-8K: 83.6%)
- Stronger code capability

---

### Mistral Large (February 2024, updated November 2024)

Mistral Large is Mistral AI's flagship closed model, available via API.

**Mistral Large 1 (2402):**
- Proprietary architecture, estimated ~200B+ parameters
- 32K context window
- MMLU: 81.2%
- Multilingual: 5 languages (French, German, Spanish, Italian, English)
- Competitive with GPT-4 on some benchmarks

**Mistral Large 2 (2411):**
- Significantly improved
- MMLU: 84.0%
- 128K context window
- Improved multilingual (10+ languages)
- Better coding and math
- Agentic capabilities (function calling, structured output)

---

### Specialized Mistral Models

**Mistral Saba (February 2025):**
- Specialized for **Arabic** and South Asian languages
- Based on Mistral architecture with modified tokenizer
- Optimized vocabulary for Arabic script, Hindi, Urdu, Tamil, etc.
- Demonstrates the importance of language-specific tokenization

**Codestral (May 2024):**
- 22B parameter model specialized for code
- 32K context window
- Fill-in-the-Middle training (like DeepSeek-Coder)
- Supports 80+ programming languages
- HumanEval: 71.4% pass@1 (competitive at the time)

**Mathstral (July 2024):**
- 7B model specialized for mathematical reasoning
- Fine-tuned from Mistral 7B with math-focused data
- MATH: 56.6% (impressive for 7B)
- Competition math performance was competitive with much larger models
- Demonstrates that strong math reasoning can be achieved at 7B scale with focused training

---

## Qwen Family (Alibaba)

### Qwen 1 (September 2023)

Qwen (from Alibaba's Qwen team) was designed as a bilingual (Chinese + English) foundation model.

**Initial release:**
- Sizes: 1.8B, 7B, 14B, 72B
- Architecture: decoder-only with standard components
- Context: 32K tokens (8K for 1.8B)
- Vocabulary: 152,000 tokens (high vocag for Chinese coverage)
- Training tokens: 3T for the 72B model

**Key design decisions:**
- Large vocabulary size (152K) to cover Chinese characters efficiently
- Chinese text is tokenized at ~1.5 characters per token (vs ~3 for English BPE models)
- Standard architecture: RoPE, SwiGLU, RMSNorm, bias=False

---

### Qwen2 (June 2024) — Global-Query Attention

Qwen2 improved the architecture and scaled context length.

**Architecture:**
- Sizes: 0.5B, 1.5B, 7B, 32B, 72B
- Context: 32K (128K for 72B)
- **GQA** (Grouped-Query Attention) — standard now but important for Qwen
- SwiGLU, RoPE, RMSNorm
- **Dual-chunk attention for long context:** splits long sequences into chunks, computes attention within and between chunks
- Vocabulary: 151,936 (kept similar to Qwen1)
- Training tokens: 7T+ for 72B

**Qwen2-Audio (July 2024):**
- Speech understanding model
- Encoder: Whisper-style audio encoder (large-v2)
- Connector: Q-Former style cross-attention (like BLIP-2)
- Integrates audio features into Qwen2's language model
- Supports: speech recognition, audio event detection, music understanding
- Can process audio + text interleaved input

---

### Qwen2.5 (September 2024) — Strong Generalist

Qwen2.5 is a comprehensive upgrade across all capabilities.

**Model sizes:**
- Dense: 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B
- MoE: Qwen2.5-MoE (14B active, ~40B total)

**Key improvements:**
- Significant boost in coding and math performance
- Better instruction following
- Improved multilingual (29+ languages)
- Context: 32K (128K for 72B)
- MMLU: 79.1% (7B), 86.7% (72B)
- HumanEval: 88.5% (72B)
- GSM-8K: 95.8% (72B)

**Qwen2.5-Coder (November 2024):**
- Specialized code models: 0.5B, 1.5B, 7B, 32B
- Trained on 5.5T code tokens
- FIM training (like DeepSeek-Coder)
- HumanEval: 92.5% (32B) — best-in-class for open code models at release
- Supports 92 programming languages

**Qwen2.5-Math (November 2024):**
- Math-specialized: 1.5B, 7B, 72B
- GRPO training (like DeepSeek-R1 but earlier)
- MATH: 94.5% (72B)
- Uses **Chain-of-Thought** + **Tool-Integrated Reasoning** (Python REPL)

**Qwen2-VL (December 2024):**
- Vision-language model
- Vision encoder: 604M parameter Vision Transformer
- Dynamic resolution: supports 768×768 to 1536×1536 images
- Connector: cross-attention for vision-to-text alignment
- Video understanding: supports up to 10-minute videos
- Requires only 1.5B additional parameters over base language model

---

### QwQ (December 2024) — Reasoning Through Qwen

QwQ (Qwen with Questions) is Alibaba's reasoning model, similar to o1/R1.

**Architecture:**
- Based on Qwen2.5-72B
- Extended thinking with chain-of-thought
- **Self-play RL:** the model generates reasoning chains, verifies their correctness, and learns from successful chains
- GRPO-based training

**Performance:**
- AIME: 66.7% (impressive for an open model)
- MATH-500: 94.2%
- MMLU: 85.6%
- GPQA: 65.2%
- Slightly behind DeepSeek-R1 on math, but more general-purpose

---

## Other Notable Models

### Command R / R+ (Cohere)

**Command R (March 2024):**
- 35B parameters
- Specialized for RAG (Retrieval-Augmented Generation)
- 128K context window
- Strong at: embedding-based retrieval, citation generation, grounded generation
- MMLU: 75.7%
- Multi-step tool use capabilities

**Command R+ (April 2024):**
- 104B parameters
- Improved RAG and tool use
- 128K context
- Competitive with GPT-4 on RAG-specific tasks
- Designed for enterprise use cases

**Key innovation — **Grounded Generation****:
- The model is trained to generate inline citations
- Response format: "John F. Kennedy was the 35th president [1]... [1] History.com, JFK Biography"
- This enables verifiability and trust in enterprise deployments
- Training: data with human-verified citations, RLHF for citation accuracy

---

### DBRX (Databricks, March 2024)

- **132B total parameters, 36B active (MoE)**
- 16 experts, top-4 routing (unusual: typically top-2)
- Built on open-source stack (Megablocks for MoE, Composer for training)
- 2T training tokens
- MMLU: 73.7%
- HumanEval: 70.1%
- **Open-source** (MIT license)
- Trained on 3,072 H100 GPUs
- Key differentiator: DBRX is designed for enterprise customization (fine-tuning, RAG)

**Architecture details:**
- RoPE, SwiGLU, GQA (with 4 KV heads per 64 query heads)
- Top-4 routing: each token activates 4 experts out of 16
- Capacity factor: 1.125 (tight capacity, some token dropping)
- Fine-grained MoE approach

---

### Grok 1 / 2 / 3 (xAI)

**Grok 1 (November 2023, open-sourced March 2024):**
- 314B parameters (MoE)
- 8 experts, top-2 routing, ~86B active
- 25K context window
- Base model trained on ~2T tokens (text + code)
- Unique: access to real-time X (Twitter) data for training
- Personality: witty, sometimes sarcastic (designed to be less "safe")
- MMLU: 73.0%
- Open-sourced under Apache 2.0

**Grok 2 (August 2024):**
- Proprietary, improved over Grok 1
- MMLU: 88.4% (competitive with GPT-4)
- Advanced reasoning and coding
- **Aurora image generation** (integrated into the model)
- Available via X Premium subscription

**Grok 3 (February 2025):**
- Latest generation, significantly improved
- Claimed to outperform DeepSeek-R1 and GPT-4o
- Extended reasoning capabilities
- 100K+ context window
- Focus on truth-seeking and mathematical reasoning

---

### Yi 1.5 / Yi 2 (01.AI)

**Yi 1.5 (January 2024):**
- Sizes: 6B, 9B, 34B
- Bilingual (Chinese + English)
- 34B model: MMLU 76.3% (competitive with Llama-2-70B at half the size)
- 200K context window (via YaRN RoPE scaling)
- Vocabulary: 64,000 tokens (efficient for Chinese)

**Yi 2 (December 2024):**
- Upgraded architecture with GQA
- Improved multilingual support (more languages)
- Stronger code and math performance
- Remains open-weight under Apache 2.0

---

### Falcon (TII, UAE)

**Falcon series:**
- **Falcon 40B (May 2023):** Trained on 1T tokens of RefinedWeb (filtered CommonCrawl)
  - MMLU: 55.4% (surprisingly low for 40B, but very data-efficient)
  - Architecture: multi-query attention (all heads share K,V)
  - Open-weight under Apache 2.0
  - Notable: proved that high-quality web data filtering could reduce need for curated data

- **Falcon 180B (September 2023):**
  - 180B parameters, trained on 3.5T tokens
  - MMLU: 74.7%
  - At release, the largest openly available model
  - Required 400GB+ GPU memory (multiple A100s/H100s with parallelism)
  - Architecture: same as Falcon 40B (MQA, etc.)

- **Falcon 2 (June 2024):**
  - 11B model with 8K context
  - UAE government-funded, open-weight
  - Focus on efficiency and multilingual

---

### MPT (MosaicML, now Databricks)

**MPT (Mosaic Pretrained Transformer) series:**
- **MPT-7B (May 2023):** First major open-source model after Llama 1
  - Architecture: standard decoder with some modifications
  - Used ALiBi (Attention with Linear Biases) instead of RoPE — simpler position encoding
  - FlashAttention for training speed
  - 1T training tokens
  - Open-source (Apache 2.0)

- **MPT-30B (June 2023):**
  - 30B parameters, 8K context
  - GQA for efficient inference
  - MMLU: 54.2% (competitive for its size at the time)
  - Designed for easy fine-tuning and deployment

---

### Pythia (EleutherAI)

- **Purpose:** A research-oriented model suite designed specifically for studying training dynamics
- **Sizes:** 70M, 160M, 410M, 1B, 1.4B, 2.8B, 6.9B, 12B
- **Key innovation:** All checkpoints (every 1000 training steps) are saved and publicly available
- This enables: studying how knowledge evolves during training, analyzing memorization, probing for biases during training
- Trained on The Pile (825GB of diverse text)
- Standard decoder architecture (GPT-NeoX style)
- Used for countless research papers on training dynamics, memorization, bias

---

### OLMo (AI2)

**OLMo (Open Language Model, February 2024):**
- AI2's fully open-source LLM
- **Complete transparency:** model weights, training code, data, training logs, evaluation suite all open
- Sizes: 1B, 7B (and later 7B-v2)
- Architecture: standard decoder, GQA, RoPE, SwiGLU
- Training data: Dolma dataset (3T tokens, fully open)
- 2048 A100 GPUs for training
- MMLU: 54.4% (7B) — reasonable for its size
- Key contribution: open-science approach to LLM development

**OLMo 2 (December 2024):**
- 7B and 13B versions
- Significant improvements in data quality and training recipe
- MMLU: 67.4% (13B)

---

### Amber (LLM360)

**Amber (October 2023):**
- Fully open-source 6.7B model
- Part of the LLM360 project (open-sourcing full training pipeline)
- 1.1T training tokens, 200 checkpoints released
- Trained on the "AmberCorpus" (filtered CommonCrawl + The Pile)
- Standard architecture
- Contribution: full training pipeline transparency

---

### Gemma / Gemma 2 / Gemma 3 (Google)

**Gemma 1 (February 2024):**
- Google's open-weight model family
- Sizes: 2B, 7B
- Based on the same technology as Gemini (but smaller, text-only, open)
- MMLU: 64.3% (7B)
- Architecture: decoder-only, GQA, RoPE, GeGLU
- Trained on 6T tokens
- Context: 8K tokens
- License: custom commercial license (free for most uses)

**Gemma 2 (June 2024):**
- Sizes: 2B, 9B, 27B
- Architecture: improved with **alternating attention** (local + global layers)
  - Local layers: sliding window attention (windth 4096)
  - Global layers: full attention every 3rd or 5th layer
  - This hybrid approach combines efficiency of SWA with long-range modeling of full attention
- MMLU: 75.2% (27B) — competitive with Llama-3-8B at ~3× size
- Context: 8K
- Trained on 13T tokens (27B model)
- Knowledge distillation during training (teacher: Gemini 1.5 Pro)

**Gemma 3 (March 2025):**
- Sizes: 1B, 4B, 12B, 27B
- **Multimodal** (text + image input)
- Architecture improvements: MoE for some sizes, hybrid attention
- MMLU: 81.6% (27B)
- Vision capabilities for all models ≥4B
- Context: 128K

---

### Phi 1 / 2 / 3 / 4 (Microsoft)

**Phi-1 (June 2023) — Textbook-Quality Data:**
- 1.3B parameters
- Trained on 50B tokens of **synthetic textbook-quality data**
- Data: Python code from The Stack v1.2 + GPT-3.5-generated "textbook" explanations
- HumanEval: 50.6% pass@1 (astonishing for 1.3B)
- **Key insight:** data quality can compensate for model size — "textbook-quality" data carries much more signal per token than web data

**Phi-1.5 (September 2023):**
- 1.3B, text + code
- MMLU: 41.6% (small but reasonable)
- Focus on common sense reasoning

**Phi-2 (December 2023):**
- 2.7B parameters
- Trained on 1.4T tokens (blend of synthetic + filtered web data)
- MMLU: 56.3% (comparable to Mistral 7B at 2.7B!)
- Coding: HumanEval 59.6%
- Breakthrough: demonstrated that small models with high-quality data can match 7B models

**Phi-3 (April 2024):**
- Sizes: 3.8B, 7B, 14B
- Architecture: **Phi-3-mini (3.8B)** uses **block attention** (some layers use dilated attention for longer context)
- Context: 4K (mini), 128K (medium, large)
- 3.8B trained on 3.3T tokens
- MMLU: 69.0% (3.8B), 78.0% (14B)
- **Phi-3-mini fits on a phone** — quantized to 4-bit, ~1.8GB, runs on iPhone 14/15
- 3.8B outperforms Llama-3-8B in some benchmarks (data quality thesis)

**Phi-3.5 (August 2024):**
- Improved versions of Phi-3
- Multi-lingual support (English, Chinese, French, German, Spanish, Italian, Japanese, Korean, Portuguese, Arabic)

**Phi-4 (December 2024):**
- 14B parameters
- MMLU: 80.4% (new record for sub-20B models)
- Architecture: further improved training data and recipe
- **Focus:** reasoning and math (GSM-8K: 95.3%)
- Training: highly filtered web data + synthetic data
- Demonstrates that 14B can match 70B models from a year earlier

---

### MiniCPM (OpenBMB)

**MiniCPM (February 2024):**
- 2.4B parameters (not counting embeddings)
- Trained on 1T tokens
- MMLU: 53.5% (competitive with 7B models)
- **Key innovation:** model scaling beyond Chinchilla-optimal — training smaller models on more tokens for deployment efficiency
- "Small but mighty" philosophy

**MiniCPM-2B (improved):**
- MMLU: 67.9% (after further improvements)
- Strong multilingual performance
- Efficient inference on CPU and mobile

---

### SmolLM (Hugging Face)

**SmolLM (July 2024):**
- Sizes: 135M, 360M, 1.7B
- Designed for research and experimentation on small models
- Trained on SmolLM-Corpus (high-quality synthetic + filtered data)
- Focus: studying scaling properties, efficient training, and on-device deployment
- **Key feature:** fully open training pipeline (data, code, configs, checkpoints)
- Part of Hugging Face's push for open science

---

### H2O-Danube (H2O.ai)

**H2O-Danube (January 2024):**
- 1.8B parameters
- Danube-1.8B trained on 1.8T tokens
- MMLU: 35.4% (competitive for its size)
- Trained on 100% open data (LAION's OpenWeb + The Stack + RedPajama)
- Apache 2.0 license
- Danube2 (October 2024): 1.8B and 7B sizes

---

## BERT Family (Encoder-Only)

### BERT (Google, 2018)

BERT (Bidirectional Encoder Representations from Transformers) revolutionized NLP by introducing **bidirectional pre-training**.

**Architecture:**
- **Encoder-only** Transformer (no causal masking)
- Bidirectional self-attention (each token attends to all tokens in both directions)
- This is the **key difference** from GPT-style decoder models — BERT sees the full context

**Model sizes:**

| Model | Parameters | Layers | d_model | Heads |
|-------|-----------|--------|---------|-------|
| BERT-Base | 110M | 12 | 768 | 12 |
| BERT-Large | 340M | 24 | 1024 | 16 |

**Pre-training objectives:**
1. **Masked Language Modeling (MLM):** 15% of tokens are masked; the model predicts them from context
   - Of the 15%: 80% replaced with [MASK], 10% replaced with random token, 10% unchanged
   - This prevents mismatch between pre-training (with [MASK]) and fine-tuning (no [MASK])
2. **Next Sentence Prediction (NSP):** Predict if sentence B follows sentence A
   - 50% positive pairs (A → B follows), 50% negative (A → B from different document)
   - This teaches sentence-level coherence

**Tokenizer:**
- WordPiece with 30,000 token vocabulary
- Pre-tokenization: split on whitespace and punctuation
- Subword: split unknown words into known pieces (e.g., "playing" → "play" + "##ing")
- Special tokens: [CLS], [SEP], [MASK], [PAD], [UNK]

**Fine-tuning:**
- Add a task-specific classification head on top of the [CLS] token
- For classification, NER, QA, etc. — the same pre-trained model can be adapted
- [CLS] token's final hidden state serves as the aggregate sequence representation
- Fine-tuning is fast (few epochs) and data-efficient

**Impact:**
- Set SOTA on 11 NLP benchmarks at release
- Demonstrated that bidirectional context is crucial for understanding tasks
- Sparked the "BERT era" — most NLP systems from 2018-2022 were BERT-based

---

### RoBERTa (Facebook, 2019)

**Robustly Optimized BERT Approach:**
- Same architecture as BERT-Large (340M params)
- **Key improvements:** better training recipe
  1. **More data:** 160GB (BERT: 16GB) — BookCorpus + Wikipedia + CC-News + OpenWebText + Stories
  2. **More training:** 100K steps with batch size 8K (BERT: 1M steps, batch 256)
  3. **No NSP:** Removed Next Sentence Prediction (shown to be unnecessary)
  4. **Dynamic masking:** different mask pattern each epoch (BERT: static masking)
  5. **Larger token batches:** 8K tokens per batch (vs BERT's 256)
- Performance: GLUE 88.5 (BERT: 80.5), SQuAD 94.6 (BERT: 90.9)
- Demonstrated that BERT was under-trained — RoBERTa showed the same architecture could be much better

---

### ALBERT (Google, 2019)

**A Lite BERT:**
- **Parameter reduction techniques:**
  1. **Factorized embedding:** Decompose vocabulary embedding matrix into two smaller matrices
     - Instead of V × H (vocab_size × hidden_size), use V × E + E × H where E << H
     - Typical: E=128, H=768, V=30K → 3.8M + 98K vs 23M (77% reduction)
  2. **Cross-layer parameter sharing:** All 12 layers share the same attention + FFN parameters
     - This massively reduces parameters without reducing computation
     - ALBERT-large (12 layers, shared) ≈ BERT-base (12 layers, separate) parameters
- ALBERT-xxlarge: 223M parameters but hidden_dim 4096, 16 layers (BERT-large is 340M with 1024-dim, 24 layers)
- Performance: slightly worse than BERT-large but with 18× fewer parameters
- **Key insight:** Layer parameters can be heavily shared without major quality loss

---

### DistilBERT (Hugging Face, 2019)

**Knowledge distillation of BERT:**
- 66M parameters (40% fewer than BERT-base)
- 2× faster inference
- 95% of BERT's performance
- **Training:**
  1. Student: 6 layers (half of BERT-base's 12)
  2. Teacher: BERT-base
  3. Distillation loss: KL divergence between student and teacher output distributions
  4. Combined with MLM loss
  5. No NSP
- Proved that distillation is highly effective for BERT-style models

---

### ELECTRA (Stanford, 2020)

**Efficiently Learning an Encoder that Classifies Token Replacements Accurately:**
- **Novel pre-training task:** Replaced Token Detection (RTD)
  1. A small generator (like a masked LM) replaces some tokens with plausible alternatives
  2. The ELECTRA discriminator predicts which tokens were replaced
  3. The discriminator is the model you keep — it learns to detect "fake" tokens
- **Advantage:** All tokens are processed (unlike MLM which only processes 15%), so training is more efficient
- ELECTRA-small (14M) matches BERT-base (110M) on GLUE
- ELECTRA-large matches RoBERTa-large with 1/4 of the computation
- **Key insight:** The RTD task provides a denser learning signal than MLM

---

### DeBERTa / DeBERTaV3 (Microsoft, 2020-2021)

**Decoding-enhanced BERT with Disentangled Attention:**
- **Disentangled attention:** Instead of attention = f(content + position), DeBERTa computes separate attention scores for content-to-content and content-to-position, then sums them
  - A_{i,j} = Q_content_i · K_content_j + Q_content_i · K_position_{δ(i,j)} + Q_position_{δ(j,i)} · K_content_j
  - This allows the model to learn word content and position relationships independently
- **Enhanced mask decoder:** Uses absolute positions in the decoder layer (only for MLM prediction)
- DeBERTaV3 (2021):
  - Replaced MLM with **ELECTRA-style RTD** (replaced token detection)
  - Improved training efficiency
  - DeBERTaV3-large (304M) matches or exceeds DeBERTa-large (1.5B) on many tasks
  - DeBERTaV3-base: 34-layer, 30K vocab, 305M params

---

### ModernBERT (2024)

**Latest evolution of BERT:**
- **Goal:** Bring BERT architecture up to modern standards
- **Key improvements:**
  1. **Rotary Position Embeddings (RoPE)** instead of absolute positions
  2. **Grouped-Query Attention (GQA)** for efficiency
  3. **Flash Attention** support
  4. **GeGLU activation** (like modern LLMs)
  5. **Pre-norm** (RMSNorm)
  6. **No bias terms**
  7. **8192 context length** (vs BERT's 512)
  8. **Large vocabulary** for better coverage
  9. Architecture: essentially an encoder-only LLaMA
- Sizes: base (~150M) and large (~400M)
- Outperforms DeBERTaV3 on most benchmarks
- Demonstrates that encoder-only models benefit from the same architectural improvements as decoder LLMs

---

## T5 Family (Encoder-Decoder)

### T5 (Google, 2019)

**Text-to-Text Transfer Transformer:** A unified framework where every NLP task is cast as a text-to-text problem.

**Architecture:**
- **Encoder-Decoder** Transformer
- Encoder: bidirectional self-attention (full context)
- Decoder: causal self-attention + cross-attention to encoder
- Both encoder and decoder have 6/12/24 layers (depending on size)

**Model sizes:**

| Model | Parameters | Layers (enc+dec) | d_model | Heads |
|-------|-----------|-------------------|---------|-------|
| T5-Small | 60M | 6+6 | 512 | 8 |
| T5-Base | 220M | 12+12 | 768 | 12 |
| T5-Large | 770M | 24+24 | 1024 | 16 |
| T5-3B | 2.8B | 24+24 | 1024 | 32 |
| T5-11B | 11B | 24+24 | 1024 | 128 |

**Unified text-to-text format:**
- Every task expressed as text input → text output
- Classification: "imdb classification: This movie was great! → "positive"
- Translation: "translate English to German: The cat → "Die Katze"
- Summarization: "summarize: long article text → "short summary"
- This unified format enabled **multi-task training**

**Pre-training — "Span Corruption":**
- Replace consecutive spans of tokens with sentinel tokens
- Input: "The cat <extra_id_0> on the <extra_id_1>"
- Output: "<extra_id_0> sat <extra_id_1> mat <extra_id_2>"
- The model learns to reconstruct corrupted spans
- Span length: average 3 tokens, corrupted ~15% of tokens
- This is an encoder-decoder version of MLM

**C4 Dataset:**
- Colossal Clean Crawled Corpus — 750GB of clean, deduplicated web text
- Filtered: remove pages with porn, spam, gibberish, code
- Deduplicated at document level (removed nearly-duplicate pages)
- Required minimum 3 sentences, end with terminal punctuation
- This became a standard dataset in NLP

**Key insight:** The encoder-decoder approach allows the encoder to build a rich bidirectional representation of the input, while the decoder generates output conditioned on it. This is more flexible than decoder-only (which must perform understanding + generation with the same parameters).

---

### FLAN-T5 (Google, 2022)

**Fine-tuned LAnguage Net — Instruction Tuning:**
- T5 models fine-tuned on a massive collection of **instruction-formatted tasks**
- 1,836 tasks collected from 60+ NLP datasets
- Each task reformatted as: "Instruction: ... Input: ... Output: ..."
- Zero-shot performance dramatically improved after instruction tuning
- **No RLHF needed** — SFT on diverse instructions was sufficient

**FLAN-T5 sizes:**
- Same sizes as T5 (small through xxl)
- FLAN-T5-XXL (11B): MMLU 59.4% (comparable to much larger models)
- FLAN-T5-XL (3B): MMLU 48.6%
- Instruction following was significantly better than base T5

**Key contribution:** Demonstrated that instruction tuning at scale (1000+ tasks) is as effective as model scaling for task generalization. This inspired the instruction-tuning approach used in most modern LLMs.

---

### mT5 (Google, 2021)

**Multilingual T5:**
- Trained on mC4 (C4 in 101 languages)
- Same architecture as T5
- Vocabulary: 250,000 tokens (SentencePiece, unified for all languages)
- Supports 101 languages
- Cross-lingual transfer: improvements in one language benefit others

**Challenges:**
- Vocabulary must balance token efficiency across all languages
- 250K vocab was necessary to cover 101 languages with reasonable efficiency
- Training data distribution follows web prevalence: English dominant, many languages underrepresented

---

### UL2 (Google, 2022)

**Unifying Language Learning:**
- Introduces **Mixture of Denoisers (MoD)** — multiple pre-training objectives combined during training

**Three denoising paradigms:**
1. **R-Denoiser (Regular):** Standard span corruption (like T5), short spans
2. **X-Denoiser (Extreme):** Span corruption with very long spans (50%+ of input)
   - Forces the model to learn long-range dependencies and generate coherent long text
3. **S-Denoiser (Sequential):** Causal LM variant — prefix language modeling
   - Model sees a prefix, generates the rest autoregressively
   - Bridges the gap between bidirectional encoders and autoregressive decoders

**Architecture:**
- Encoder-decoder (like T5)
- Mode tokens: `<2R>`, `<2X>`, `<2S>` prepended to input to indicate which denoising mode to use
- **Key innovation:** one model trained on all three modes — effectively a "Swiss Army knife" that can function as encoder (R), long-span generator (X), or decoder-only (S)
- UL2-20B: competitive with T5-11B and PaLM-62B on many tasks

---

### T5v1.1 (Google, 2020)

**Improvements over T5:**
- GeGLU activation (instead of ReLU)
- Pre-norm (instead of post-norm)
- No bias terms in LayerNorm
- Training on more C4 data
- These changes improved quality but introduced incompatibility with original T5 checkpoints
- T5v1.1-Large consistently outperforms T5-Large

---

## Architectural Paradigms

### Encoder-Only Models

**Examples:** BERT, RoBERTa, ALBERT, DistilBERT, ELECTRA, DeBERTa, ModernBERT

**Capabilities:**
- **Best at:** Natural Language Understanding (classification, NER, relation extraction, sentiment analysis)
- **Cannot:** Generate text (no decoder)
- **Core advantage:** Bidirectional context provides the richest token representations
- Use case: embedding generation, feature extraction, semantic search, classification tasks

**Architecture details:**
- Full bidirectional self-attention (no causal masking)
- Typically use [CLS] token for aggregated representations
- Pre-trained with MLM or similar objectives
- Fine-tuned by adding a lightweight task head

**When to use encoder-only:**
- You need text embeddings (semantic similarity, clustering)
- Classification/labeling tasks
- When computational cost is a concern (they're cheaper than decoder models)
- Applications requiring bidirectional understanding (entity extraction, relation classification)

**Limitations:**
- Cannot generate text
- Usually limited context length (BERT: 512 tokens, ModernBERT: 8K)
- Not suitable for conversational AI or generative tasks

---

### Decoder-Only Models

**Examples:** GPT-4, Llama 3, Claude, Mistral, Gemini, DeepSeek-V3, Qwen2.5

**Capabilities:**
- **Best at:** Text generation, conversational AI, code generation, creative writing
- **Also good at:** Understanding (with sufficient scale and training), reasoning, translation, summarization
- **Core advantage:** Unified architecture for both understanding and generation in one model

**Architecture details:**
- Causal self-attention (each token can only attend to previous tokens)
- Pre-trained with autoregressive language modeling (predict next token)
- No explicit embedding layer for downstream tasks
- The same parameters perform both understanding and generation through the forward pass

**Why decoder-only dominates:**
1. **Simplicity:** One model, one objective, one set of parameters for all tasks
2. **Scaling:** Better scaling properties — in-context learning emerges at scale
3. **Flexibility:** Can be used for generation, classification, embedding, etc.
4. **Chat optimization:** RLHF aligns the generation behavior for conversation
5. **No complexity of encoder-decoder:** No need to design cross-attention or two-pass processing

**When to use decoder-only:**
- Conversational AI and chatbots
- Content generation (writing, coding, translation)
- Any task requiring text generation
- Zero-shot/few-shot learning (in-context learning)

**Trade-offs:**
- Less efficient for pure understanding tasks (must generate to answer)
- Left-to-right bias — cannot use future context for understanding
- Larger and more expensive than equivalent encoder models
- KV cache memory grows with sequence length

---

### Encoder-Decoder Models

**Examples:** T5, FLAN-T5, mT5, UL2, T5v1.1, BART, PEGASUS

**Capabilities:**
- **Best at:** Sequence-to-sequence tasks (translation, summarization, question answering)
- **Combination:** Bidirectional understanding (encoder) + autoregressive generation (decoder)

**Architecture details:**
- Encoder: full bidirectional self-attention
- Decoder: causal self-attention + cross-attention to encoder outputs
- Cross-attention: decoder attends to encoder's final hidden states
- This allows the model to "look at" the entire input when generating each output token

**Advantages over decoder-only:**
- Richer input representations (bidirectional)
- Better at tasks where understanding the full input is critical
- Cross-attention provides a strong inductive bias for I/O tasks

**Advantages over encoder-only:**
- Can generate text
- More flexible task formulation

**When to use encoder-decoder:**
- Translation, summarization
- Any task with a clear input and output distinction
- When you need to generate based on a fully-understood input
- When the input contains complex relationships that benefit from bidirectional encoding

**Trade-offs:**
- Two-pass processing (encode then decode) — slower
- More parameters (two transformers) for the same capacity
- Cross-attention adds computational complexity
- Less flexible than decoder-only for conversational AI
- Harder to scale (encoder-decoder scaling laws differ)

---

### Summary: Which Architecture to Use

| Criterion | Encoder-Only | Decoder-Only | Encoder-Decoder |
|-----------|-------------|-------------|-----------------|
| Text Generation | ❌ No | ✅ Yes | ✅ Yes |
| Classification | ✅ Best | ✅ Good | ✅ Good |
| Embeddings | ✅ Best | ✅ Good | ❌ Complex |
| Translation | ❌ No | ✅ Good | ✅ Best |
| Summarization | ❌ No | ✅ Good | ✅ Best |
| Conversational AI | ❌ No | ✅ Best | ⚠️ Possible |
| Reasoning | ⚠️ Limited | ✅ Best | ⚠️ Possible |
| Inference Speed | ✅ Fastest | ⚠️ Moderate | ❌ Slowest |
| Context Length | ⚠️ Short | ✅ Long | ⚠️ Moderate |
| Parameter Efficiency | ✅ Best | ⚠️ Inferior | ✅ Good |
| Scale Benefits | ❌ Saturing | ✅ Emergent | ⚠️ Moderate |

---

## Mixture-of-Experts Deep Dive

### Sparse MoE Architecture

**Core concept:**
Instead of having one feed-forward network per layer, MoE replaces each FFN with multiple "expert" FFNs and a **router** that decides which expert(s) to use for each token.

**Why MoE:**
- **Activate fewer parameters** while having more total parameters
- Dense model of size N: all N parameters used for every token
- MoE model of size N_total with k_active: N_total can be much larger, but compute per token is similar to N_active
- This allows scaling total parameter count without proportional compute increase

**Components:**

1. **Experts:** N small FFNs (typically 8-256 experts)
   - Each expert is a standard SwiGLU/ReLU FFN: W1, W2 (and W3 for SwiGLU)
   - Experts are typically homogeneous (same size, same structure)
   - Only the FFN layer is "expertized" — attention, layer norm, embeddings remain dense

2. **Router (Gate):** A learned linear layer that computes routing probabilities
   - g(x) = softmax(W_g · x)
   - For each token, the router outputs a distribution over N experts
   - Top-k experts are selected based on gating probabilities

3. **Combined output:** y = Σ_i g_i(x) · FFN_i(x) (weighted by gating probabilities)

**Top-k Routing:**
- Most common: k=2 (Mixtral, DeepSeek-V2, DBRX)
- Each token activates 2 experts
- Routing decision: select the k experts with highest logits
- Gating probabilities of selected experts are re-normalized (softmax over selected k)
- The gating probabilities are used as weights for combining expert outputs

**Router variants:**
- **Token Choice (standard):** Router picks top-k experts for each token
- **Expert Choice:** Each expert picks top-k tokens (inverted routing)
  - Better load balancing (each expert always processes exactly k tokens)
  - But can't guarantee all tokens are processed (if no expert picks them)
  - Used in Google's Switch Transformer

### Load Balancing Loss

**Problem:** Without intervention, the router learns to always send tokens to the same few "expert" experts, defeating the purpose of MoE.

**Solution: Auxiliary Load Balancing Loss**

1. **Importance-based loss:**
   L_aux = α · N_experts · Σ_i (f_i · P_i)
   where f_i = fraction of tokens routed to expert i
   and P_i = average routing probability for expert i
   This encourages uniform assignment (f_i ≈ 1/N for all i)

2. **Switch Transformer variant:**
   L_load = α · Σ_i f_i · P_i (simplified)

3. **DeepSeek variant:**
   Combines importance loss with "z-loss" — a penalty on the logits' magnitude to stabilize training

Typical α values: 0.01 (standard), can be tuned per model

### Capacity Factor and Token Dropping

**Capacity:**
Each expert can process at most C = capacity_factor × (tokens_per_batch / num_experts) tokens.

- **Capacity factor = 1.0:** Each expert processes exactly its fair share. Token dropping likely if routing is imperfect.
- **Capacity factor = 1.25:** 25% slack. Most tokens will be processed, but some may still be dropped.
- **Capacity factor = 2.0:** High slack. Rarely drops tokens, but more computation.

**Token dropping:**
When a token's target expert is at capacity, the token is either:
- **Dropped:** The token passes through as a residual connection (skips the FFN entirely)
  - Not ideal for quality, but keeps computation deterministic
- **Overloaded:** The expert processes it anyway (beyond capacity)
  - Increases computation but prevents information loss
  - Used by some implementations

**Why token dropping is problematic:**
- Important tokens (e.g., the token being predicted) might be dropped
- During training, dropped tokens lose expert gradient signal
- During inference, unexpected drops can change model behavior
- Solution: carefully manage capacity factor and monitor drop rates

### Batch-Aware Routing

In MoE training, routing is computed per-token independently, but the capacity constraint is **per-batch**. This creates a tension:

- A token's routing decision (which expert) doesn't consider other tokens' decisions
- But the expert capacity is shared across all tokens in the batch
- This means routing must produce balanced batches at the batch level, not per-sample

**Expert parallelism:**
- In distributed training, different experts are placed on different GPUs
- Routing must be followed by **all-to-all communication** (send tokens to their assigned expert's GPU)
- This creates communication overhead — a key challenge in MoE training
- Solutions: group size tuning, expert sharding, hierarchical routing

### Fine-Grained MoE

**Problem with traditional MoE:**
- 8-16 experts means each expert is quite large
- A few "specialized" experts may dominate, while others are rarely used
- Gradients flow sparsely to individual experts

**Fine-grained MoE (DeepSeekMoE, QwenMoE):**

- **More, smaller experts:** 64-256 experts instead of 8-16
- **More active experts:** 6-8 experts per token instead of 2
- Each expert is smaller (fewer parameters), so the total compute per token is similar

**Benefits of fine-grained:**
- **Better specialization:** More experts can specialize on specific patterns
- **Finer routing granularity:** The model can combine more expert perspectives
- **Better load balancing:** More experts means easier to balance load
- **Redundant specialization:** Different tokens can use different specialized experts

**Trade-offs:**
- More communication overhead (more experts per token → more all-to-all)
- Harder to train (more routing decisions per token)
- Risk of expert collapse (some experts never learn useful functions)

**DeepSeekMoE specifics:**
- 256 experts with 8 active per token
- Each expert has hidden_dim = 2048 (for V3 671B)
- **Shared experts:** A few dense FFNs that all tokens use (in addition to routed experts)
  - This captures general computation that every token needs
  - The routed experts handle specialized computation
  - Shared experts are always computed; routed experts are selected per-token
- **No token dropping:** Uses expert balancing with load-aware routing and dynamic capacity adjustment

### Expert Parallelism

**How experts are distributed across GPUs:**

1. **Expert Sharding:** Each GPU hosts a subset of experts
   - If 8 GPUs and 64 experts: each GPU hosts 8 experts
2. **All-to-All Communication:** Tokens are sent to the GPU hosting their selected expert
   - This is the bottleneck in MoE training
3. **Capacity per GPU:** Each GPU processes its allocated tokens

**Communication patterns:**
- **Token dispatch:** Send token vectors to the GPU with the chosen expert
- **Token combine:** Collect processed vectors from all GPUs
- Both require all-to-all collective communication

**Strategies to reduce communication:**
- **Local MoE:** Some experts are replicated across GPUs (no communication needed locally)
- **Hybrid parallelism:** Combine expert + data + tensor parallelism
- **Hierarchical MoE:** First route to expert groups, then to experts within groups

### DeepSeekMoE Architecture Details

**DeepSeek V2/V3 MoE structure:**

```
Layer Input
    │
    ├── Router (linear → softmax)
    │       │
    │       ├── Route to 8 experts out of 256 (top-8 routing)
    │       │
    │       └── Each expert: SwiGLU FFN (W_up, W_gate, W_down)
    │
    ├── Shared experts (2-4 dense FFNs, always computed)
    │
    └── Combine: Σ g_i · Expert_i(x) + SharedExpert(x)
```

**Key design choices:**
- 256 fine-grained experts (each relatively small)
- Top-8 active experts per token
- 2 shared experts (always on)
- No token dropping (padded capacity)
- Load balancing via auxiliary loss (with z-loss for stability)

**Why 256 experts with top-8?**
- Total compute: 8/256 ≈ 3% of total FFN parameters used per token
- This is ~8× more efficient than Mixtral (2/8 = 25%)
- Each expert can be more specialized (smaller capacity)
- Shared experts handle common computation

### QwenMoE Architecture

**Qwen2.5-MoE (14B active, ~40B total):**

- Based on Qwen2.5 architecture
- Uses MoE only in FFN layers (attention is dense)
- **Top-2 routing** with 8 experts (traditional approach)
- **Load balancing loss** with adaptive coefficient
- **Expert dispatch:** each token activates 2 experts
- **Shared expert:** 1 shared FFN expert in addition to routed experts

**MoE specifics:**
```
│── Shared Expert (dense FFN, always computed)
│── Router → top-2 from 8 experts
│── Combine: shared + routed
```

- Total parameters saved: ~28% fewer than an equivalent dense model
- Active parameters: ~35% of total (vs 100% for dense)
- Performance: nearly matches dense Qwen2.5-72B with ~40B total parameters

---

## Comprehensive Comparison

### Model Family Comparison Table

| Model Family | Latest Model | Parameters | Active Params | Context Length | Architecture | Modality | Open Weight | License |
|-------------|-------------|-----------|--------------|---------------|-------------|----------|------------|---------|
| GPT | GPT-4.1 | ~1.8T* | ~200B* | 1M | MoE Decoder | Text, Image, Audio | ❌ | Proprietary |
| Llama | Llama 4 Maverick | 401B | 48B | 1M | MoE Decoder | Text, Image | ✅ | Custom |
| Claude | Claude 4 Opus | ~300B* | N/A | 200K | Dense Decoder* | Text, Image | ❌ | Proprietary |
| Gemini | Gemini 2.5 Pro | ~1.5T* | ~200B* | 2M | MoE Decoder | Text, Image, Audio, Video | ❌ | Proprietary |
| DeepSeek | DeepSeek-V3 | 671B | 37B | 128K | MoE Decoder | Text | ✅ | Custom |
| Mistral | Mistral Large 2 | ~200B* | N/A | 128K | Dense Decoder* | Text | ❌ | Proprietary |
| Qwen | Qwen2.5-72B | 72B | 72B | 128K | Dense Decoder | Text, Image, Audio | ✅ | Apache 2.0 |
| Command | Command R+ | 104B | 104B | 128K | Dense Decoder | Text | ✅ | Custom (CC NC) |
| DBRX | DBRX | 132B | 36B | 32K | MoE Decoder | Text | ✅ | MIT |
| Grok | Grok 3 | ~300B* | ~100B* | 100K+ | MoE Decoder* | Text, Image | ❌ | Proprietary |
| Yi | Yi 2 | 34B | 34B | 200K | Dense Decoder | Text | ✅ | Apache 2.0 |
| Falcon | Falcon 2 | 11B | 11B | 8K | Dense Decoder | Text | ✅ | Apache 2.0 |
| Gemma | Gemma 3 | 27B | 27B | 128K | Dense Decoder | Text, Image | ✅ | Custom |
| Phi | Phi-4 | 14B | 14B | 8K | Dense Decoder | Text | ✅ | MIT |
| MiniCPM | MiniCPM-2.4B | 2.4B | 2.4B | 32K | Dense Decoder | Text | ✅ | Apache 2.0 |
| BERT | ModernBERT | ~400M | 400M | 8K | Dense Encoder | Text | ✅ | Apache 2.0 |
| T5 | FLAN-T5-XXL | 11B | 11B | 512 | Enc-Dec | Text | ✅ | Apache 2.0 |

*Estimate/approximation. *N/A = Not publicly disclosed.

### Performance Comparison (Selected Benchmarks)

| Model | MMLU | HumanEval | GSM-8K | MATH |
|-------|------|-----------|--------|------|
| GPT-4 | 86.4% | 67.0% | 92.0% | 52.9% |
| GPT-4o | 88.7% | 90.2% | 94.5% | 76.6% |
| o1-preview | 91.8% | 68.2% | 94.8%* | — |
| GPT-4.1 | 92.5%* | ~92%* | — | — |
| Llama-3.1-405B | 87.3% | 89.0% | 96.8% | 78.6% |
| Llama-4-Maverick | ~88%* | ~91%* | ~94%* | — |
| Claude 3 Opus | 86.8% | 84.9% | 95.0% | 73.7% |
| Claude 3.5 Sonnet | 88.7% | 92.0% | 96.8% | 78.3% |
| Claude 4 Opus | 89.3% | 93.7% | — | 86.5% |
| Gemini 1.5 Pro | 87.2% | 84.1% | 92.3% | 67.7% |
| Gemini 2.5 Pro | 91.8% | — | — | 97.0% |
| DeepSeek-V3 | 89.5% | 92.5% | 96.8% | 83.9% |
| DeepSeek-R1 | 90.8% | 92.7% | 97.3%* | — |
| Mistral Large 2 | 84.0% | — | — | — |
| Qwen2.5-72B | 86.7% | 88.5% | 95.8% | 83.1% |
| Command R+ | 75.7% | — | — | — |
| DBRX | 73.7% | 70.1% | — | — |
| Phi-4 | 80.4% | — | 95.3% | — |

*Not directly comparable due to different evaluation settings.

### Cost Comparison (API Pricing, per 1M tokens)

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|----------------------|
| GPT-4o | $2.50 | $10.00 |
| GPT-4o-mini | $0.15 | $0.60 |
| o1-preview | $15.00 | $60.00 |
| o1-mini | $3.00 | $12.00 |
| GPT-4.1 | $2.00 | $8.00 |
| Claude 3.5 Haiku | $0.80 | $4.00 |
| Claude 3.5 Sonnet | $3.00 | $15.00 |
| Claude 4 Opus | $15.00 | $75.00 |
| Gemini 1.5 Flash | $0.075 | $0.30 |
| Gemini 1.5 Pro | $1.25 | $5.00 |
| Gemini 2.5 Flash | $0.15 | $0.60 |
| Gemini 2.5 Pro | $1.25 | $10.00 |
| DeepSeek-V3 | $0.27 | $1.10 |
| DeepSeek-R1 | $0.55 | $2.19 |
| Mistral Large 2 | $2.00 | $6.00 |
| Mistral Small | $0.20 | $0.60 |
| Command R+ | $2.50 | $10.00 |
| Command R | $0.50 | $1.50 |

### Tokenizer Comparison

| Model | Tokenizer | Vocab Size | Algorithm | Byte Coverage |
|-------|-----------|-----------|-----------|-------------|
| GPT-4 | cl100k_base | 100,256 | Byte-level BPE | ✅ Yes |
| Llama 3 | tiktoken-based | 128,256 | Byte-level BPE | ✅ Yes |
| Llama 2 | SentencePiece | 32,000 | BPE | ⚠️ Via byte fallback |
| Claude | Proprietary | ~100K* | BPE* | ✅ Yes |
| Gemini | SentencePiece | 256,000* | Unigram | ✅ Yes |
| DeepSeek-V3 | HuggingFace tokenizers | 129,280 | Byte-level BPE | ✅ Yes |
| Mistral | SentencePiece | 32,000 | BPE | ⚠️ Via byte fallback |
| Qwen2.5 | Tokenizer from Qwen | 151,936 | BPE | ✅ Yes |
| BERT | WordPiece | 30,000 | WordPiece | ❌ UNK tokens |
| T5 | SentencePiece | 32,000 | Unigram | ⚠️ Via byte fallback |

### Training Data Size Comparison

| Model | Training Tokens | Data Source |
|-------|----------------|-------------|
| GPT-3 | 570B | CommonCrawl, books, Wikipedia, code |
| GPT-4 | ~13T (est.) | Web, books, code, images |
| Llama-3.1-405B | 15.6T | Web, code, books, Wikipedia (filtered) |
| Llama-4 | 30T+ (est.) | Web, code, books, images |
| Claude 4 | Not disclosed | Proprietary |
| Gemini 2.5 | Not disclosed | Google's corpus |
| DeepSeek-V3 | 14.8T | Web, code, math, multilingual |
| Mistral Large | Not disclosed | Proprietary |
| Qwen2.5-72B | 7T+ | Web, code, multilingual |
| Phi-4 | ~10T (est.) | Synthetic + filtered web |

### Release Timeline

| Date | Event |
|------|-------|
| June 2018 | GPT-1 (117M) |
| October 2018 | BERT (340M) |
| February 2019 | GPT-2 (1.5B) |
| July 2019 | RoBERTa |
| October 2019 | T5 (11B) |
| June 2020 | GPT-3 (175B) |
| September 2020 | DeBERTa |
| January 2021 | mT5 |
| August 2021 | Codex (12B) |
| January 2022 | InstructGPT |
| May 2022 | UL2-20B |
| November 2022 | ChatGPT |
| February 2023 | Llama 1 (65B) |
| March 2023 | GPT-4, Claude 1 |
| July 2023 | Llama 2 (70B), Claude 2 |
| September 2023 | Mistral 7B |
| November 2023 | GPT-4 Turbo, Grok 1 |
| December 2023 | Mixtral 8x7B, Gemini 1.0 |
| January 2024 | DeepSeek LLM, Phi-3, Gemma |
| February 2024 | Gemini 1.5 Pro, Mistral Large |
| March 2024 | Claude 3, DBRX, Command R+ |
| April 2024 | Llama 3 (8B, 70B), Mixtral 8x22B |
| May 2024 | GPT-4o, DeepSeek-V2 |
| June 2024 | Claude 3.5 Sonnet, Qwen2, Gemma 2 |
| July 2024 | Llama 3.1 (405B) |
| August 2024 | Grok 2 |
| September 2024 | Qwen2.5, o1-preview, Llama 3.2 |
| October 2024 | Claude 3.5 Haiku |
| November 2024 | GPT-4o update, DeepSeek-V2.5 |
| December 2024 | DeepSeek-V3, Gemini 2.0 Flash, QwQ, Phi-4 |
| January 2025 | DeepSeek-R1 |
| February 2025 | DeepSeek-Prover, Grok 3, Mistral Saba |
| March 2025 | Gemini 2.5 Pro/Flash, Gemma 3 |
| April 2025 | GPT-4.1, Claude 4, Llama 4 |
| May 2025 | (Latest releases continue evolving) |

---

## Emerging Trends

### Reasoning Models (o1, o3, R1, QwQ, Gemini Thinking)

- **Chain-of-thought at inference time** — the model "thinks" before answering
- RL-based training for reasoning (GRPO, PPO with correctness rewards)
- Distillation of reasoning abilities to smaller models
- Transparent reasoning (Claude 4, Gemini) vs hidden (o-series)

### Multimodal Convergence

- GPT-4o: single model for text, image, audio
- Gemini 2.5: text + image + audio + video
- Llama 4: open multimodal MoE
- Trend: all frontier models are becoming natively multimodal

### Efficiency Innovations

- **Architecture:** MoE (most 2024+ frontier models), MLA (DeepSeek), fine-grained experts
- **Training:** FP8 training, efficient parallelism, data quality over quantity
- **Inference:** KV-cache optimization, speculative decoding, quantization
- **Cost:** DeepSeek-V3 trained for $5.5M — demonstrating 10× cost reduction vs GPT-4

### Open vs Closed

- **Open:** Llama, Mistral, Qwen, DeepSeek, Gemma, Phi
- **Closed:** GPT, Claude, Gemini
- Trend: open models (Llama 3.1 405B, DeepSeek-V3) have matched or approached closed models
- Commercial models retain advantages in: multimodality, context length, reliability

---

|*This document provides comprehensive coverage of major LLM families. Each section represents deep technical understanding derived from academic papers, technical reports, architecture analyses, and community knowledge.*

---

**See also:**
- [06-AI-Model-Providers-Free-Tiers.md](./06-AI-Model-Providers-Free-Tiers.md) — Complete directory of AI model API providers, their free tiers, pricing, and recommendations by use case.
