# 01 - LLMs and AI Models

> A comprehensive introduction to Large Language Models, the Transformer architecture, training pipelines, inference, and key concepts in modern AI.

---

## Table of Contents

1. [What is an AI Model](#1-what-is-an-ai-model)
2. [What is a Large Language Model](#2-what-is-a-large-language-model)
3. [Transformer Architecture](#3-transformer-architecture)
4. [Training Pipeline](#4-training-pipeline)
5. [Base vs Instruct Models](#5-base-vs-instruct-models)
6. [Model Sizes and Parameters](#6-model-sizes-and-parameters)
7. [Quantization](#7-quantization)
8. [Inference](#8-inference)
9. [Key Concepts](#9-key-concepts)
10. [Further Reading](#10-further-reading)

---

## 1. What is an AI Model

An **AI model** is a mathematical construct — typically a neural network — that has been trained on data to recognize patterns, make predictions, or generate outputs. At its core, an AI model is a function that maps inputs to outputs:

```
y = f(x; θ)
```

Where:
- **x** is the input (e.g., text, image, audio)
- **y** is the output (e.g., classification, generated text, bounding box)
- **θ** (theta) represents the model's parameters — learned weights that determine the behavior of the function
- **f** is the architecture of the neural network

The **parameters** are what the model learns during training. They are numerical values organized into layers (matrices and vectors). A model with 7 billion parameters, for instance, has 7 billion individual numbers that collectively encode everything the model "knows."

Models are categorized broadly by what they process:
- **Language models** (text in, text out)
- **Vision models** (images in, predictions out)
- **Multimodal models** (combination of text, images, audio, video)
- **Speech models** (audio in, text or audio out)
- **Diffusion models** (generate images/video from noise)

The "intelligence" of a model is a function of its architecture, its parameter count, the quality and quantity of its training data, and the training methodology used.

---

## 2. What is a Large Language Model

A **Large Language Model (LLM)** is a type of neural network specialized for understanding and generating human language. LLMs are "large" — typically billions of parameters — trained on vast text corpora spanning books, articles, code, websites, and more.

### Key Characteristics

- **Statistical next-token prediction**: At their core, LLMs predict the next token (word or sub-word) in a sequence. Given "The capital of France is", they predict "\_\_\_".
- **Emergent abilities**: As models scale up (more parameters, more data), unexpected capabilities emerge — reasoning, translation, summarization, coding, in-context learning, etc. These abilities are not explicitly programmed but arise from the training process.
- **Few-shot and zero-shot learning**: LLMs can perform tasks with few or no examples, generalizing from patterns learned during training.
- **Contextual understanding**: They process entire sequences of tokens, allowing them to maintain context over long passages of text.

### Notable LLMs (by family)

| Family | Creator | Examples | Notes |
|---|---|---|---|
| GPT | OpenAI | GPT-3, GPT-4, GPT-4o | Among the first large-scale LLMs |
| LLaMA | Meta | LLaMA 2, LLaMA 3, LLaMA 3.1 | Open-weight, widely used in research |
| Mistral | Mistral AI | Mistral 7B, Mixtral 8x7B | Efficient models, MoE architecture |
| Qwen | Alibaba | Qwen 2.5, QwQ | Strong multilingual and reasoning |
| DeepSeek | DeepSeek | DeepSeek-V2, V3, R1 | Competitive with closed-source models |
| Gemma | Google | Gemma 2, Gemma 3 | Open-weight from Google |
| Claude | Anthropic | Claude 3, Claude 4 | Safety-focused, closed-source |
| Command | Cohere | Command R+ | Optimized for RAG and business |

---

## 3. Transformer Architecture

The Transformer architecture, introduced in the paper *"Attention Is All You Need"* (Vaswani et al., 2017), is the foundation of virtually all modern LLMs. It replaced recurrent neural networks (RNNs) and LSTMs by introducing a mechanism that processes all tokens in parallel rather than sequentially.

### 3.1 High-Level Structure

A Transformer consists of stacked layers, each containing:

1. **Multi-Head Self-Attention** — allows each token to "attend to" every other token
2. **Feed-Forward Network (FFN)** — a learned transformation applied per token
3. **Layer Normalization** — stabilizes training
4. **Residual Connections** — gradient flow bypass

Modern LLMs use a **decoder-only** architecture (described in Section 3.7).

### 3.2 Tokenization

Before text enters the model, it must be converted into a sequence of integers (token IDs). **Tokenization** is the process of splitting text into tokens.

```
Input:  "Hello, world!"
Tokens: ["Hello", ",", " world", "!"]
IDs:    [15496, 11, 995, 0]
```

#### Tokenization Methods

- **Byte-Pair Encoding (BPE)**: Starts with individual bytes/characters and iteratively merges the most frequent pairs. Used by GPT, LLaMA, Mistral, etc.
- **WordPiece**: Similar to BPE but merges based on likelihood gain rather than frequency. Used by BERT.
- **SentencePiece / Unigram**: Language-agnostic; works directly on raw text without pre-tokenization. Used by LLaMA, Gemma, etc.

#### Vocabulary Size

Typical LLM vocabularies range from 32,000 to 256,000 tokens:

| Model | Vocabulary Size |
|---|---|
| GPT-2 | 50,257 |
| LLaMA 3 | 128,000 |
| Gemma 2 | 256,000 |
| DeepSeek-V3 | 128,000 |

A larger vocabulary means fewer tokens per piece of text (faster generation), but larger embedding and output matrices (more memory).

#### Special Tokens

LLMs use special tokens to structure input and output:

- `<|begin_of_text|>` / `<s>` — marks the start of a sequence
- `<|end_of_text|>` / `</s>` — marks the end
- `<|pad|>` — padding for batched processing
- `<|im_start|>` / `<|im_end|>` — chat template markers (instruct models)
- `<eos>` — end-of-sequence (stop generation)

### 3.3 Embeddings

**Embeddings** are dense vector representations of tokens. Each token ID is mapped to a high-dimensional vector (the embedding dimension).

```
Token ID:  [15496]      →  Embedding:  [0.12, -0.45, 0.78, ..., 0.33]
Dimension:   1          →  Dim:        d_model (e.g., 4096)
```

Two types of embeddings are used:

1. **Token Embeddings**: A lookup table `V × d_model` where V is vocabulary size and d_model is the hidden dimension. Each row is a learned vector for a token.
2. **Positional Embeddings**: Since the Transformer processes tokens in parallel (not sequentially), it has no inherent sense of token order. Positional information must be injected.

#### Positional Encoding Types

- **Absolute (sinusoidal)**: Fixed sine/cosine functions at different frequencies. Used in the original Transformer.
- **Learned absolute**: Positional embeddings are trainable parameters. Used in GPT-2/3.
- **Rotary Position Embedding (RoPE)**: Encodes position by rotating the query and key vectors. Favored in modern models (LLaMA, Mistral, Qwen, Gemma). Allows better length generalization.
- **ALiBi (Attention with Linear Biases)**: Biases attention scores linearly by distance between tokens. Used in BLOOM, MPT.

#### Embedding Dimension (d_model)

| Model | d_model | Layers |
|---|---|---|
| LLaMA 3 8B | 4096 | 32 |
| LLaMA 3 70B | 8192 | 80 |
| Mistral 7B | 4096 | 32 |
| DeepSeek-V3 | 7168 | 61 (dense) |

### 3.4 Self-Attention

**Self-attention** is the core innovation of the Transformer. It allows every token to "look at" every other token in the sequence and compute a weighted representation. The weights are determined by how "relevant" each token is to the current token.

#### The Attention Mechanism

For each token, we compute three vectors from its embedding:

- **Query (Q)**: What am I looking for?
- **Key (K)**: What do I contain?
- **Value (V)**: What information do I pass along?

These are obtained by multiplying the input embedding by learned weight matrices:

```
Q = X * W_Q
K = X * W_K
V = X * W_V
```

The attention score between token i and token j is computed as the dot product of Q_i and K_j:

```
Score(i, j) = Q_i · K_j
```

Higher scores indicate stronger relevance. The scores are scaled by `1/√d_k` (to prevent softmax saturation) and passed through a softmax to produce attention weights:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) * V
```

#### Causal (Masked) Attention

In decoder-only LLMs, a token must not attend to *future* tokens (since it shouldn't "see" the answer). A **causal mask** is applied: all attention scores for `j > i` are set to -infinity before softmax, so they become zero.

```
Mask matrix:
Token 0: [1, 0, 0, 0]
Token 1: [1, 1, 0, 0]
Token 2: [1, 1, 1, 0]
Token 3: [1, 1, 1, 1]
```

### 3.5 Multi-Head Attention

Rather than performing a single attention computation, the Transformer runs **multiple attention heads in parallel**. Each head learns to focus on different types of relationships:

```python
# Pseudo for 8 heads
head_0 = attention(Q_0, K_0, V_0)  # Syntax relationships?
head_1 = attention(Q_1, K_1, V_1)  # Entity relationships?
head_2 = attention(Q_2, K_2, V_2)  # Long-distance dependencies?
# ... etc
```

All heads' outputs are concatenated and projected back to d_model:

```
MultiHead(Q, K, V) = Concat(head_0, ..., head_h) * W_O
```

Typical head counts:
- LLaMA 3 8B: 32 heads (8 KV heads — grouped-query attention)
- LLaMA 3 70B: 64 heads (8 KV heads)
- Mistral 7B: 32 heads (8 KV heads)
- GPT-3: 96 heads

#### Attention Variants

- **Multi-Head Attention (MHA)**: Full Q, K, V heads — most compute/memory intensive.
- **Grouped-Query Attention (GQA)**: Multiple query heads share fewer key/value heads. Balances speed and quality. Used in LLaMA 2/3, Mistral, Gemma.
- **Multi-Query Attention (MQA)**: All query heads share a single key/value head. More aggressive, used in PaLM, Falcon.

### 3.6 Feed-Forward Networks (FFN)

After attention, each token passes through an identical **feed-forward network** (FFN), also called a **Multi-Layer Perceptron (MLP)**. This is typically two linear transformations with a non-linear activation in between:

```
FFN(x) = act(x * W_1 + b_1) * W_2 + b_2
```

The FFN has an **intermediate (hidden) dimension** that is typically 2.5x to 4x d_model:

| Model | d_model | FFN hidden dim | Activation |
|---|---|---|---|
| LLaMA 3 8B | 4096 | 14,336 | SwiGLU |
| Mistral 7B | 4096 | 14,336 | SwiGLU |
| GPT-3 | 12,288 | 49,152 | GELU |
| Gemma 2 9B | 3584 | 28,672 | GeGLU |

#### Activation Functions

- **ReLU**: `max(0, x)` — simple, used in early transformers
- **GELU**: `x * Φ(x)` (Gaussian Error Linear Unit) — smoother than ReLU
- **SwiGLU** / **SiLU** (Sigmoid Linear Unit): `x * sigmoid(x)` — current default in most LLMs
- **GeGLU**: GELU variant with gating

Modern LLMs nearly all use **gated activations** (SwiGLU, GeGLU), which multiply the activation output by a learned gating signal.

### 3.7 Layer Normalization

**Layer normalization** (LayerNorm) stabilizes training by normalizing activations across the feature dimension:

```
LayerNorm(x) = γ * (x - μ) / √(σ² + ε) + β
```

Where:
- μ is the mean of the layer's activations
- σ is the standard deviation
- γ and β are learned scale and shift parameters
- ε is a small constant to prevent division by zero

#### Normalization Placement

- **Post-norm** (original Transformer): Norm after residual addition. Harder to train for very deep models.
- **Pre-norm** (modern default): Norm before each sub-layer. More stable gradient flow, enabling deeper models.
- **RMS Norm** (used in LLaMA, Mistral): Simplified version that only normalizes by root mean square. Faster, nearly as effective.

```
RMSNorm(x) = x / √(mean(x²) + ε) * γ
```

### 3.8 Decoder-Only Design

The original Transformer had an **encoder-decoder** structure: the encoder reads the full input (bidirectional), and the decoder generates output (causal). Modern LLMs (GPT, LLaMA, Mistral, etc.) use a **decoder-only** design.

#### Why Decoder-Only?

1. **Simplicity**: One stack of layers instead of two.
2. **Generalization**: The same architecture handles both understanding and generation — the model simply conditions on whatever context is provided (a prompt, a question, a task description).
3. **Scaling**: Decoder-only models scale more predictably.
4. **In-context learning**: The causal (left-to-right) nature inherently supports autoregressive generation, which is exactly what chat and completion tasks need.

#### Architecture Diagram (Conceptual)

```
Input:  [token_1, token_2, ..., token_n]
            ↓
        [Embeddings + Positional Encoding]
            ↓
        ┌───────────────────────┐
        │ Transformer Layer x N  │  (repeated L times)
        │                       │
        │   ┌─────────────────┐ │
        │   │ RMSNorm         │ │
        │   │   → Self-Attn   │ │  (causal mask)
        │   │   → Residual +  │ │
        │   └─────────────────┘ │
        │   ┌─────────────────┐ │
        │   │ RMSNorm         │ │
        │   │   → FFN (MLP)   │ │
        │   │   → Residual +  │ │
        │   └─────────────────┘ │
        └───────────────────────┘
            ↓
        [RMSNorm] → [LM Head] (linear projection to vocab)
            ↓
        Token probabilities
```

The **LM Head** (also called the unembedding or output projection) is a linear layer that projects the final hidden state to vocabulary-sized logits. Often, the LM head shares weights with the token embedding matrix (weight tying).

---

## 4. Training Pipeline

Training a modern LLM is a multi-stage pipeline. Each stage serves a different purpose and uses different data.

### 4.1 Pre-Training

**Pre-training** is the first and most compute-intensive stage. The model is trained on a massive, diverse corpus of text (trillions of tokens) to predict the next token in a sequence — a task called **causal language modeling** or **autoregressive language modeling**.

#### Objective

Given a sequence of tokens `[x_1, x_2, ..., x_n]`, the model maximizes:

```
L = Σ log P(x_t | x_<t, θ)
```

i.e., the log-probability of each token given all previous tokens and the model parameters θ.

#### Training Data

Pre-training data typically includes:
- Web pages (Common Crawl, web archives) — the largest source
- Books (fiction, non-fiction, textbooks)
- Scientific papers (arXiv, PubMed)
- Code (GitHub)
- Wikipedia, Reddit, forums
- News articles
- Social media

The total dataset size for modern LLMs:

| Model | Training Tokens |
|---|---|
| LLaMA 1 | 1.0T |
| LLaMA 2 | 2.0T |
| LLaMA 3 | 15.0T+ |
| DeepSeek-V3 | 14.8T |
| Mistral 7B | ~8.0T |
| GPT-4 (estimated) | ~13T |

#### Data Processing Pipeline

Raw data goes through extensive cleaning:
1. **Deduplication**: Remove near-duplicate documents (MinHash, Bloom filters)
2. **Quality filtering**: Heuristic rules, classifier-based filtering (e.g., using FastText to select "high-quality" pages)
3. **PII removal**: Strip personally identifiable information
4. **Toxicity filtering**: Remove hate speech, violence, explicit content
5. **Perplexity filtering**: Train a small reference model, filter out documents that score too high (surprising) or too low (trivial)
6. **Domain reweighting**: Increase weight for high-quality sources (books, papers), decrease for low-quality (spam, SEO garbage)

#### Data Mixing

The ratio of data sources is a critical hyperparameter. Typical mixes:

```
Web crawl:     60-70%
Books:         10-15%
Code:          10-15%
Academic:      5-10%
Other:         5%
```

#### Training Configuration

- **Optimizer**: AdamW (Adam with decoupled weight decay)
- **Learning rate schedule**: Cosine decay with warmup
- **Batch size**: Thousands of sequences (millions of tokens per step)
- **Hardware**: Thousands of GPUs (A100/H100/B200) for weeks to months
- **Precision**: Mixed precision (BF16/FP16) with gradient checkpointing, ZeRO optimization, tensor/pipeline parallelism

#### Compute Costs

| Model | GPU Hours (estimate) |
|---|---|
| LLaMA 3 8B | ~1.6M A100-hours |
| LLaMA 3 70B | ~6.4M A100-hours |
| GPT-4 (estimated) | ~100M+ GPU-hours |

### 4.2 Supervised Fine-Tuning (SFT)

After pre-training, the model knows language but doesn't follow instructions well. **Supervised Fine-Tuning (SFT)** teaches the model to respond helpfully to user prompts.

#### Process

1. Collect **instruction-response pairs**: Human annotators write prompts and ideal responses.
2. Fine-tune the pre-trained model on these pairs using the standard next-token prediction loss.

```
User: "Explain what a transformer is."
Assistant: "A transformer is a neural network architecture..."
```

The model sees the full conversation and is trained to predict the assistant's tokens (loss is computed only on the response tokens, not the prompt).

#### SFT Data Quality

A typical SFT dataset might have 10,000–100,000 examples. Unlike pre-training (more data = better), SFT benefits more from **high quality** than large quantity. Key principles:
- Diverse: Cover many tasks (coding, writing, reasoning, QA, summarization, translation, etc.)
- Correct: Answers must be factually accurate
- Well-formatted: Consistent template (e.g., `### Instruction` / `### Response`)
- Multi-turn: Some examples of conversation history

#### Common SFT Datasets

| Dataset | Size | Description |
|---|---|---|
| OpenAssistant | ~34K | Crowdsourced conversations |
| ShareGPT | ~90K | User-shared ChatGPT conversations |
| No Robots | ~10K | High-quality curated instructions |
| Dolly | 15K | Databricks-generated instructions |
| UltraChat | 1.5M | Large-scale synthetic conversations |

Many models also use **synthetic data** generated by stronger models (e.g., GPT-4 generating training examples for smaller models).

### 4.3 Reinforcement Learning from Human Feedback (RLHF)

RLHF aligns the model with human preferences — making it more helpful, honest, and harmless. It was popularized by InstructGPT/ChatGPT and is used by most frontier models.

#### Three-Step RLHF Pipeline

**Step 1: SFT (warmup)**
Fine-tune on high-quality instruction data as described above.

**Step 2: Reward Model Training**
Train a separate **reward model** (RM) that predicts human preference scores:

1. Generate multiple responses for the same prompt using the SFT model
2. Humans rank these responses (e.g., "response A is better than response B")
3. Train a reward model (usually a transformer with a scalar head) to predict the human ranking

The RM learns to assign higher scores to responses humans prefer.

**Step 3: Reinforcement Learning (PPO)**

Using **Proximal Policy Optimization (PPO)** , the SFT model is fine-tuned to maximize the reward model's score while staying close to the original SFT model (to prevent reward hacking):

```
Objective = E[r(x, y)] - β * KL(π_RL || π_SFT)
```

Where:
- r(x, y) is the reward model's score for prompt x and response y
- KL(π_RL || π_SFT) is the KL divergence between the RL-optimized policy and the SFT model
- β controls the strength of the KL penalty

The KL penalty prevents the model from exploiting the reward model (e.g., producing nonsensical but high-reward outputs).

#### Direct Preference Optimization (DPO)

**DPO** simplifies RLHF by directly optimizing the model on preference pairs without training a separate reward model or doing RL. The loss function:

```
L_DPO = -E[log σ(β * log(π_θ(y_w | x) / π_ref(y_w | x)) - β * log(π_θ(y_l | x) / π_ref(y_l | x)))]
```

Where y_w is the "winning" response and y_l is the "losing" response.

DPO is simpler than PPO (no reward model, no RL library) and competitive in many settings. It has largely replaced PPO in open-source fine-tuning pipelines.

#### Other Alignment Methods

| Method | Description |
|---|---|
| **DPO** | Direct optimization on preference pairs |
| **IPO** | Identity preference optimization (variant of DPO) |
| **KTO** | Kahneman-Tversky optimization (uses "good" vs "bad" examples instead of pairs) |
| **ORPO** | Odds Ratio Preference Optimization (combines SFT and alignment in one stage) |
| **SimPO** | Simple preference optimization (simplified reward formulation) |
| **RLAIF** | Use an AI (e.g., a stronger LLM) as the preference judge instead of humans |

---

## 5. Base vs Instruct Models

After training, a model can be released in different flavors:

### Base Model

- **Trained only on pre-training** (or pre-training + small amounts of curated data)
- Capable of **text completion** but not instruction following
- Requires prompt engineering to elicit useful behavior
- Used primarily by developers/researchers who want to fine-tune further
- Example: `meta-llama/Llama-3.1-8B` (base)

```
Prompt: "The capital of France is"
→ "Paris. It is known for the Eiffel Tower, which was built in 1889."
```

### Instruct/Chat Model

- **Base model + SFT (+ alignment)** — fine-tuned on instruction-response pairs
- Understands conversation format and follows instructions
- Designed for direct use by end users
- Example: `meta-llama/Llama-3.1-8B-Instruct`

```
User: "What is the capital of France?"
Assistant: "The capital of France is Paris."
```

### Differences summarized

| Aspect | Base Model | Instruct Model |
|---|---|---|
| Training | Pre-training only | Pre-training + SFT + RLHF/DPO |
| Output style | Completion | Conversational / instructive |
| Behavior | Open-ended generation | Follows instructions |
| Use case | Fine-tuning, research | Chatting, production apps |
| Safety | Minimal guardrails | Stronger refusal behaviors |
| Prompt format | Simple text | Chat template (special tokens) |

### Chat Templates

Instruct models use a specific format for structuring conversations. Different models use different formats:

```
# LLaMA 3 Instruct
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
What is 2+2?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
4<|eot_id|>

# Mistral Instruct
<s>[INST] What is 2+2? [/INST] 4</s>

# ChatML (GPT-4, Qwen)
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
4<|im_end|>
```

These templates must be applied correctly or the model will produce poor results.

---

## 6. Model Sizes and Parameters

### 6.1 Parameter Counts

Model size is measured in **parameters** — the total number of trainable values (weights and biases). Common sizes:

| Size Class | Parameter Range | Example Models |
|---|---|---|
| Small | 1B–3B | TinyLLaMA 1.1B, Phi-3 Mini 3.8B, Gemma 2 2B |
| Medium | 7B–14B | LLaMA 3 8B, Mistral 7B, Gemma 2 9B |
| Large | 34B–70B | LLaMA 3 70B, Command R+, Mixtral 8x22B |
| Very Large | 120B–405B | LLaMA 3.1 405B, DeepSeek-V3 |
| Frontier | 500B–2T+ | GPT-4 (est. ~1.8T), Gemini Ultra |

### 6.2 Memory Requirements

Parameter count directly determines GPU memory needed. For **inference** (not training), memory required is roughly:

```
Memory ≈ P * bytes_per_param
```

- **FP16/BF16**: 2 bytes per parameter
- **FP32**: 4 bytes per parameter
- **INT4**: 0.5 bytes per parameter

Plus overhead for KV cache, activations, and model structure:

| Model | Size | FP16 VRAM (approx) |
|---|---|---|
| LLaMA 3 8B | 8B | ~16 GB |
| LLaMA 3 70B | 70B | ~140 GB |
| Mistral 7B | 7B | ~14 GB |
| DeepSeek-V3 | 671B (MoE) | ~400 GB (FP8) |

**Training** requires significantly more memory: parameters, gradients, optimizer states (AdamW stores 2 extra states per param), and activations. Training a 7B model may need 60–80 GB of VRAM per GPU.

### 6.3 Mixture of Experts (MoE)

**Mixture of Experts** is an architecture that achieves better performance per FLOP/token by having many specialized subnetworks ("experts") but only activating a subset for each input.

#### How MoE Works

1. A **router** (gating network) computes a probability distribution over experts
2. For each token, the top-k experts are selected (typically k=2)
3. The token is processed only by those experts
4. The experts' outputs are combined with weighted by the router's probabilities

```
For each token:
  router(embedding) → probabilities over 8 experts
  top-2: expert_3 (0.6), expert_7 (0.3)
  output = 0.6 * expert_3(token) + 0.3 * expert_7(token)
```

#### Dense vs MoE Comparison

| Property | Dense | MoE |
|---|---|---|
| Total parameters | 7B | 47B (8 experts × ~5.9B each) |
| Active parameters per token | 7B | ~7B (2 experts active) |
| FLOPs per token | ~7B | ~7B |
| Memory required | ~14 GB (FP16) | ~94 GB (FP16) |
| Inference cost per token | Lower | Higher (more parameters loaded) |

MoE models have a large **memory footprint** (all experts must be loaded) but **compute footprint** similar to a smaller dense model. This means:
- Good for throughput (batch processing many tokens)
- Bad for latency-sensitive single-token generation (must load all experts)

#### MoE Capacity Factor

The **capacity factor** controls how many tokens each expert can process:
- CF = 1.0: Each expert gets at most `tokens / num_experts` slots
- CF > 1.0: More slots, but some tokens may be dropped or overflow

#### Notable MoE Models

| Model | Total Params | Active Params | Experts | Top-k |
|---|---|---|---|---|
| Mixtral 8x7B | 47B | 13B | 8 | 2 |
| Mixtral 8x22B | 141B | 39B | 8 | 2 |
| DeepSeek-V3 | 671B | 37B | 256 | 8 |
| Qwen 2.5-MoE | 42B | 16B | — | — |
| GPT-4 (est.) | ~1.8T | ~280B | 16 | 2 |
| Gemini 1.5 Pro (est.) | — | — | MoE | — |
| DBRX | 132B | 36B | 16 | 4 |

MoE is also used in **Mixture of Agents (MoA)** , covered in Section 9.9.

### 6.4 Beyond Parameters: Data and Compute

Model quality depends on more than parameter count. The **Chinchilla scaling laws** (Hoffmann et al., 2022) showed that for optimal performance, the amount of training data should scale proportionally with model size:

```
Compute-optimal: 20 tokens of data per parameter
```

- A 7B model should be trained on ~140B tokens
- A 70B model: ~1.4T tokens
- A 405B model: ~8.1T tokens

Modern models often exceed these ratios (more data is still beneficial, just not "optimal" for the compute budget).

---

## 7. Quantization

**Quantization** reduces the precision of model weights (and optionally activations) to decrease memory usage and increase inference speed, at the cost of some accuracy.

### 7.1 Precision Types

| Precision | Bits | Range | Use Case |
|---|---|---|---|
| FP32 | 32 | ±3.4×10³⁸ | Full precision, training |
| FP16 (half) | 16 | ±65,504 | Mixed-precision training |
| BF16 | 16 | ±3.4×10³⁸ | Training (same range as FP32) |
| FP8 (E5M2/E4M3) | 8 | Limited | Emerging, used in H100 |
| INT8 | 8 | -128 to 127 | Inference quantization |
| INT4 | 4 | -8 to 7 | Extremely compressed inference |
| NF4 | 4 | Normal distribution | Optimal for normally-distributed weights |
| FP4 | 4 | Limited | Experimental |
| INT3/INT2 | 3/2 | Heavily limited | Experimental, significant quality loss |
| Binary (1-bit) | 1 | {0, 1} | Extreme compression, speculative |

#### Why BF16 over FP16?

BF16 (bfloat16) has the same exponent range as FP32 (8 bits) but fewer mantissa bits. This means it can represent the same range of values with less precision — ideal for deep learning where range matters more than fine-grained precision. FP16 has a narrower range (±65K vs ±3×10³⁸), causing overflow in gradient updates during training.

### 7.2 Quantization Approaches

#### Post-Training Quantization (PTQ)

Apply quantization after training is complete:

1. **Weight-only quantization**: Only weights are quantized (most common)
2. **Weight + activation quantization**: Both weights and activations quantized (harder, more speedup)
3. **Dynamic quantization**: Scale factors computed per batch
4. **Static quantization**: Scale factors pre-computed using a calibration dataset

#### Quantization-Aware Training (QAT)

Simulate quantization during training, so the model learns to compensate for precision loss. Results in higher quality but requires re-training (expensive).

#### Calibration

For PTQ, a small **calibration dataset** (hundreds to thousands of samples) is used to determine optimal quantization parameters (scale, zero-point). The calibration data should be representative of the model's expected use.

### 7.3 Quantization Formats

#### GGUF (GPT-Generated Unified Format)

- **Origin**: Created by the llama.cpp community (continuation of GGML)
- **Storage**: Single-file format containing weights, tokenizer, and model metadata
- **Quantization levels**: `q2_k`, `q3_k_m`, `q4_0`, `q4_k_m`, `q5_0`, `q5_k_m`, `q6_k`, `q8_0`
- **Best for**: CPU inference (llama.cpp), consumer GPUs with limited VRAM
- **Pros**: Universal, widely supported, good quality-per-bit
- **Cons**: Not optimal for batched GPU inference

#### GPTQ (GPT-Post-Training Quantization)

- **Origin**: Paper "GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers"
- **Storage**: Usually as a directory of safetensors + config + quantize_config.json
- **Levels**: Generally 4-bit and 3-bit
- **Best for**: GPU inference with CUDA (AutoGPTQ, ExLlama)
- **Pros**: Good GPU performance, efficient batching
- **Cons**: Calibration dataset needed, format fragmentation

#### AWQ (Activation-aware Weight Quantization)

- **Origin**: Paper "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"
- **Storage**: Similar to GPTQ (safetensors + config)
- **Levels**: Usually 4-bit
- **Best for**: GPU inference (vLLM, AutoAWQ, TensorRT-LLM)
- **Pros**: Better quality than GPTQ at same bit-width, faster
- **Cons**: Fewer tools support it (but growing)

#### Bitsandbytes (BNB)

- **Origin**: Hugging Face ecosystem, authored by Tim Dettmers
- **Levels**: 8-bit, 4-bit (QLoRA style)
- **Best for**: Fine-tuning with QLoRA, Hugging Face Transformers
- **Pros**: Easy to use with HF ecosystem, NF4 datatype
- **Cons**: CPU inference not well supported

#### AQLM (Additive Quantization of Language Models)

- **Origin**: More recent (2024)
- **Levels**: 2-bit compression
- **Best for**: Extreme compression
- **Pros**: Theoretically optimal for very low bits
- **Cons**: Still experimental

### 7.4 Quantization Comparison

| Format | Quality Retention | Inference Speed | VRAM Savings | Ecosystem Support |
|---|---|---|---|---|
| GGUF Q4_K_M | Good | Good (CPU/GPU) | ~75% | llama.cpp, LM Studio, Ollama |
| GPTQ 4-bit | Good | Excellent (GPU) | ~75% | AutoGPTQ, ExLlama, HF TGI |
| AWQ 4-bit | Very Good | Excellent (GPU) | ~75% | vLLM, AutoAWQ, TRT-LLM |
| BNB 4-bit (NF4) | Very Good | Good (GPU) | ~75% | HF Transformers, PEFT (QLoRA) |

### 7.5 Practical Guidelines

- **For CPU inference**: Use GGUF (Q4_K_M or Q5_K_M for quality)
- **For GPU inference with vLLM**: Use AWQ or FP8
- **For GPU inference with ExLlama**: Use GPTQ
- **For fine-tuning on consumer GPUs**: Use QLoRA with BNB NF4
- **For production serving**: Use FP8 (on H100) or INT8/AWQ (on A100)

---

## 8. Inference

**Inference** is the process of generating text from a trained model given an input prompt.

### 8.1 Decoding Strategies

Given the model's output probabilities over the vocabulary, **how** do we choose the next token?

#### Greedy Decoding

Always pick the most likely token:

```
token = argmax(probabilities)
```

- **Pros**: Deterministic, simple
- **Cons**: Repetitive, dull output; no creativity

#### Random Sampling

Sample from the probability distribution:

```
token = sample(probabilities)
```

- **Pros**: Diverse, creative output
- **Cons**: May produce incoherent or unlikely tokens

#### Top-k Sampling

Restrict sampling to the k most likely tokens:

```
Top-k tokens → renormalize → sample
```

Typical k values: 10–100. Lower = more focused, higher = more diverse.

#### Top-p (Nucleus) Sampling

Restrict sampling to the smallest set of tokens whose cumulative probability exceeds p:

```
Cumulative probability sorted descending.
Keep tokens until sum ≥ p.
Renormalize → sample from this set.
```

Typical p values: 0.8–0.95. More adaptive than top-k.

#### Temperature

A scaling parameter applied to logits before softmax:

```
logits_scaled = logits / temperature
probabilities = softmax(logits_scaled)
```

- **Low temperature (0.1–0.5)**: More deterministic, focused
- **High temperature (0.8–1.5)**: More random, creative
- **Temperature = 1.0**: No change
- **Temperature → 0**: Equivalent to greedy
- **Temperature → ∞**: Uniform random

#### Min-P Sampling

A newer strategy that sets a minimum probability threshold (relative to the top token's probability):

```
threshold = min_p * max_probability
Keep tokens with prob ≥ threshold
```

Helps maintain coherence while allowing diversity. Default min_p is typically 0.01–0.1.

#### Repetition Penalty

Scales down the probability of tokens that have already appeared:

```
Repeat penalty: 1.0–1.2 (1.0 = no penalty)
Higher penalty = less repetition
```

#### Typical Sampling

Keep tokens whose probability is close to the "expected" probability (entropy-based). Removes both too-likely and too-unlikely tokens.

#### Beam Search

Breadth-first search: maintain multiple candidate sequences (beams) and select the best one at the end. Used more for translation/summarization than open-ended generation.

#### Contrastive Search

Combines likelihood with a "degeneration penalty" to prevent repetitive patterns.

### 8.2 Context Window

The **context window** (or context length) is the maximum number of tokens the model can process at once. It includes both the prompt and the generated response.

| Model | Max Context Length |
|---|---|
| GPT-3 | 2K |
| GPT-4 Turbo | 128K |
| LLaMA 2 | 4K |
| LLaMA 3 | 8K |
| LLaMA 3.1 | 128K |
| Mistral 7B | 8K (32K via sliding window) |
| Mixtral 8x7B | 32K |
| Qwen 2.5 | 32K–128K (varies) |
| DeepSeek-V3 | 128K |
| Gemini 1.5 Pro | 2M |
| Claude 3 | 200K |
| Command R+ | 128K |

#### Extending Context

Techniques for handling longer sequences:

- **RoPE extrapolation**: Scale or adjust the RoPE frequencies for longer positions (YaRN, NTK-aware scaling)
- **Ring Attention**: Distribute long sequences across multiple GPUs
- **Recurrent memory / cache compression**: Compress earlier context into summary tokens
- **Sliding window attention**: Each token attends only to a local window (Mistral, StreamingLLM)
- **Strided / sparse attention**: Attend to a subset of positions for efficiency

### 8.3 KV Cache

During generation, the model processes tokens one at a time (autoregressive). For each new token, it recomputes attention over all previous tokens. Without optimization, this would be O(n²) per token — prohibitively slow for long sequences.

The **KV Cache** stores the Key and Value matrices from previous tokens and reuses them:

```
Without KV cache:
  Step 1: compute attn over [t1]        → O(1)
  Step 2: compute attn over [t1, t2]    → O(2)
  Step N: compute attn over [t1..tN]     → O(N)
  Total: O(N²)

With KV cache:
  Step 1: compute K1, V1, store → produce token 1
  Step 2: load K1,V1, compute K2,V2, store → produce token 2
  Step N: load K1..K(N-1), compute KN,VN, store → produce token N
  Total: O(N)
```

#### KV Cache Memory

The KV cache grows linearly with sequence length and batch size:

```
KV cache per token = 2 (K+V) * d_model * num_layers * 2 bytes (FP16)
```

For LLaMA 3 8B (d_model=4096, layers=32):
- Per token: 2 × 4096 × 32 × 2 = ~1 MB
- 8K context: 8,192 × 1 MB = ~8 GB
- 128K context: ~128 GB (impractical, needs optimization)

#### KV Cache Optimizations

- **GQA/MQA**: Grouped/Multi-Query Attention reduces KV head count, shrinking cache
- **PagedAttention (vLLM)**: Store KV blocks like OS virtual memory pages, reducing fragmentation
- **Cache quantization**: Store KV cache in INT8 or FP8
- **Sliding window**: Only cache recent tokens
- **Prefix caching**: Reuse KV cache for common prompt prefixes
- **Shared KV cache**: Share cache between requests with the same prefix (e.g., system prompt)

### 8.4 Inference Frameworks

#### llama.cpp

- **Language**: C/C++
- **Hardware**: CPU, Apple Silicon (Metal), CUDA, Vulkan, SYCL, etc.
- **Best for**: Local inference, consumer hardware, CPU inference
- **Format**: GGUF
- **Features**:
  - Runs on CPU efficiently (no high-end GPU needed)
  - Supports all major quantization formats
  - Prompt caching, continuous batching
  - Server mode with OpenAI-compatible API
  - Low memory footprint
- **Limitations**: Not optimized for high-throughput serving
- **Use case**: Running models on a laptop, edge devices, or personal computer

#### ollama

- Built on top of llama.cpp
- Provides a user-friendly interface (CLI + API)
- Simplifies model downloading and running
- Focus: Local, easy, personal use

#### vLLM

- **Language**: Python/C++ (CUDA)
- **Hardware**: NVIDIA GPUs (AMD ROCm experimental)
- **Best for**: High-throughput production serving
- **Format**: Hugging Face, AWQ, GPTQ, FP8
- **Features**:
  - **PagedAttention**: Efficient memory management
  - **Continuous batching**: Dynamically add/remove requests from the batch
  - **Prefix caching**: Reuse shared prompt prefixes
  - **Speculative decoding**: Use a draft model to speed up generation
  - **Multi-LoRA**: Serve many fine-tuned adapters with one base model
  - **OpenAI-compatible API**
  - **Quantization**: AWQ, GPTQ, FP8, INT8
- **Limitations**: GPU-only, more complex setup
- **Use case**: Production API serving, large-scale deployments

#### TensorRT-LLM (NVIDIA)

- **Language**: C++/Python (NVIDIA-only)
- **Hardware**: NVIDIA GPUs (optimized for H100/B200)
- **Best for**: Maximum throughput on NVIDIA hardware
- **Format**: TensorRT engine (compiled, not plain weights)
- **Features**:
  - **In-flight batching**: Batch scheduling at runtime
  - **FP8 quantization**: Native FP8 support (H100)
  - **Multi-GPU / multi-node**: Tensor Parallel, Pipeline Parallel
  - **Plugin system**: Custom attention kernels, MoE kernels
  - **Speculative decoding, Medusa, Eagle, Redrafting**
  - **CV CUDA integration**: Zero-copy data transfer
- **Limitations**: NVIDIA-only, compilation step needed, complex
- **Use case**: Maximum throughput on NVIDIA clusters

#### Other Frameworks

| Framework | Language | Focus |
|---|---|---|
| Hugging Face TGI | Rust/Python | Simple deployment on HF infrastructure |
| ExLlamaV2 | Python/CUDA | Fast GPU inference for GPTQ models |
| MLC-LLM | Python | Cross-platform (CUDA, Metal, Vulkan, WebGPU) |
| Aphrodite | Python | vLLM fork with additional features (samplers, etc.) |
| SGLang | Python | Structured generation, constraint-guided decoding |
| MLX | Python/Metal | Apple Silicon optimized (by Apple) |

### 8.5 Speculative Decoding

A technique to speed up generation without quality loss. A small **draft model** generates several tokens quickly, and the large **target model** verifies them in parallel.

```
Draft model (fast, e.g., 1B): generates 5 candidate tokens
Target model (slow, e.g., 70B): evaluates all 5 in a single forward pass
→ Accept correct candidates, reject and resample at rejection point
```

Result: 2–3x speedup without quality degradation.

### 8.6 Batching

To maximize throughput, multiple requests are processed together:

- **Static batching**: Fixed batch size, wait for all requests to finish
- **Dynamic (continuous/in-flight) batching**: Add new requests as others finish, no waiting

vLLM's continuous batching was a breakthrough, nearly doubling throughput compared to static batching.

---

## 9. Key Concepts

### 9.1 Tokens

A **token** is the fundamental unit of text that the model processes. Tokens are not words — they are sub-word units determined by the tokenizer.

- One token ≈ 0.75 words (English, average)
- One token ≈ 3.5 characters
- 100 tokens ≈ 75 words
- Pricing is typically per token (input and output billed separately)

Examples of tokenization:

```
"Hello, world!" → ["Hello", ",", " world", "!"]
"unbelievable" → ["un", "believe", "able"]
"ChatGPT" → ["Chat", "G", "PT"]
```

**Tokens are not smart** — the model doesn't "see" words, it sees token IDs. The tokenizer's quality (vocabulary, handling of rare words, multilingual coverage) directly affects model performance.

### 9.2 Embeddings

**Embeddings** are the dense vector representations that encode semantic meaning. Key properties:

- **Dimensionality**: Typically 1024–8192 (d_model)
- **Dense**: Every dimension carries information (unlike sparse one-hot vectors)
- **Learned**: Trained end-to-end with gradient descent
- **Semantic**: Similar words have similar vectors
- **Contextual**: The same word in different contexts has different representations (at deeper layers)

The embedding space captures surprising relationships:

```
vector("king") - vector("man") + vector("woman") ≈ vector("queen")
```

In LLMs, the **final hidden state** (at the last token position) is typically used as the "contextual" representation for downstream tasks.

### 9.3 Hallucination

**Hallucination** occurs when an LLM generates information that is false, fabricated, or nonsensical but presented as factual.

#### Types

- **Intrinsic**: Contradicts the provided context (e.g., says "Paris is in Germany" after being told it's in France)
- **Extrinsic**: Not verifiable from context (e.g., invents a citation that sounds plausible but doesn't exist)
- **Factual**: States incorrect facts (e.g., wrong historical dates)
- **Source confusion**: Mixes up sources or attributes correctly to the wrong entity

#### Causes

- **Statistical generation**: The model is optimized for plausible-sounding text, not factual accuracy
- **Knowledge cutoff**: The model doesn't know events after training
- **Overconfidence**: The model may assign high probability to incorrect completions
- **Pressure to be helpful**: RLHF incentivizes answering rather than refusing
- **No grounding**: The model has no mechanism to verify facts against external sources

#### Mitigation

| Technique | Description |
|---|---|
| **RAG** (Retrieval-Augmented Generation) | Retrieve relevant documents before generating |
| **Chain-of-thought** | Force step-by-step reasoning before answer |
| **Self-consistency** | Generate multiple answers and take majority vote |
| **Constrained decoding** | Force the model to only use tokens from retrieved passages |
| **In-context examples** | Provide examples showing correct fact-checking behavior |
| **Temperature reduction** | Lower temperature reduces creativity (and hallucination) |
| **Prompting techniques** | "Say 'I don't know' if unsure" |

### 9.4 Prompt Engineering

**Prompt engineering** is the practice of designing inputs (prompts) to elicit desired behavior from an LLM.

#### Basic Principles

1. **Be specific and clear**: Vague prompts → vague answers
2. **Provide context**: Give the model relevant background
3. **Use instructions**: Tell the model what to do (and what not to do)
4. **Format the output**: Specify desired format (JSON, bullet points, etc.)

#### Techniques

**Zero-shot prompting**
```
Translate to French: "Hello, how are you?"
→ "Bonjour, comment allez-vous ?"
```

**Few-shot prompting**
```
Classify the sentiment:
Text: "This movie was amazing!" → Positive
Text: "I hated every minute" → Negative
Text: "It was okay, I guess" →
```

**Chain-of-Thought (CoT)**
```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
   Each can has 3 balls. How many does he have now?
A: Roger starts with 5 balls. 2 cans × 3 balls = 6 balls.
   5 + 6 = 11. The answer is 11.
Q: The cafeteria had 23 apples. They used 20 for lunch and bought 6 more.
   How many apples do they have?
A:
```

**System prompts** (for instruct models)
```
You are a helpful AI assistant. You respond concisely and accurately.
If you don't know the answer, say so. Always cite sources when possible.
```

**Structured output prompts**
```
Generate a JSON object with fields: name, age, email.
Example: {"name": "Alice", "age": 30, "email": "alice@example.com"}
Input: Bob is 25 years old, his email is bob@test.com
```

**Role prompting**
```
You are an expert physicist explaining quantum mechanics to a 10-year-old.
Use simple analogies and no math.
```

**Negative prompting**
```
Do NOT mention politics. Do NOT use technical jargon.
```

### 9.5 RAG vs Fine-Tuning

Both **Retrieval-Augmented Generation (RAG)** and **fine-tuning** adapt a model to domain-specific knowledge, but they work differently.

#### RAG (Retrieval-Augmented Generation)

**How it works**:
1. User asks a question
2. Retrieve relevant documents from a knowledge base (vector database, search engine)
3. Inject the retrieved text into the model's context
4. Model generates an answer grounded in the retrieved documents

```
User: "What was Q3 revenue?"
        ↓
  [Retrieval: search "Q3 revenue report 2024"]
        ↓
  [Context: "Q3 revenue was $2.1B..."]
        ↓
  Model generates answer based on context
```

**Pros**: No training needed, up-to-date information, can cite sources, easy to update
**Cons**: Requires infrastructure (vector DB, retrieval pipeline), adds latency, limited by context window

#### Fine-Tuning

**How it works**: Update the model's weights on a dataset of domain-specific examples.

**Pros**: Model learns deep patterns, works without retrieval infrastructure, lower latency (no retrieval step)
**Cons**: Expensive (GPU time), dataset creation, risk of forgetting (catastrophic forgetting), needs retraining for updates

#### When to Use Which

| Scenario | Use |
|---|---|
| Need up-to-date information | RAG |
| Need to cite specific sources | RAG |
| Limited budget / no GPUs | RAG |
| Teach new style/format/tone | Fine-tuning |
| Teach new domain knowledge that doesn't fit in context | Fine-tuning |
| Reduce latency (remove retrieval step) | Fine-tuning |
| Privacy (keep data on-device with small model) | Fine-tuning |
| Best of both worlds | RAG + Fine-tuning together |

#### Hybrid Approach: RAG + Fine-Tuning

1. Fine-tune on domain-specific data (curated Q&A, writing style)
2. Add RAG for grounding in current information

This is the production pattern for many enterprise applications.

### 9.6 Model Distillation

**Knowledge Distillation** (or model distillation) trains a smaller **student model** to mimic a larger **teacher model**.

#### How It Works

1. Run the teacher model on a dataset to generate "soft labels" (probability distributions over tokens)
2. Train the student model to match the teacher's probability distribution, not just the hard labels

```
Loss = α * KL(student_logits || teacher_logits) + (1-α) * CrossEntropy(student, hard_labels)
```

The KL divergence term transfers the teacher's "soft knowledge" — e.g., which tokens are plausible alternatives, not just the single best token.

#### Benefits

- **Smaller model**: 1/10th the size of the teacher
- **Faster inference**: Lower latency, higher throughput
- **Similar quality**: Often 95%+ of the teacher's performance
- **Deployable on edge**: Phones, laptops, embedded devices

#### Notable Distilled Models

| Model | Teacher | Size Ratio |
|---|---|---|
| DistilBERT | BERT | 40% smaller, 95% performance |
| TinyLLaMA | LLaMA 2 7B | ~15% of size |
| Phi-3 Mini | GPT-4 / larger models (synthetic data) | — |
| Orca 2 | GPT-4 | ~7B model, teacher was GPT-4 |

#### Distillation vs Pruning vs Quantization

| Technique | What changes | Quality Impact |
|---|---|---|
| Distillation | Architecture (smaller model) | Minimal if done well |
| Pruning | Remove weights/neurons | Moderate |
| Quantization | Lower precision | Small (especially 8-bit) |

All three can be combined for maximum compression.

### 9.7 Mixture of Agents (MoA)

**Mixture of Agents (MoA)** is a paradigm that combines multiple LLMs to produce better outputs than any single model. Introduced by Together AI in 2024.

#### How MoA Works

1. **Multiple models** generate responses independently (layer 1)
2. Each model sees its own output plus the outputs of others (cross-attention at the agent level)
3. A final **aggregator model** combines all responses into a single best answer

```
Layer 1:
  [LLaMA 3] → Response A
  [Mistral] → Response B
  [Qwen]   → Response C

Layer 2 (optional):
  [LLaMA 3] sees A+B+C → Response D
  [Mistral] sees A+B+C → Response E

Aggregator: [GPT-4] sees A+B+C+D+E → Final Answer
```

#### Key Differences from MoE

| Aspect | Mixture of Experts (MoE) | Mixture of Agents (MoA) |
|---|---|---|
| Level | Inside one model (sub-networks) | Multiple complete models |
| Training | Trained end-to-end | Pre-trained models, no joint training |
| Activation | Router selects top-k experts | All agents generate independently |
| Use case | Efficient scaling | Ensemble for quality |

#### MoA vs Simple Ensemble

- **Simple ensemble**: Average/vote across model outputs (e.g., majority vote, self-consistency)
- **MoA**: Models see each other's outputs, allowing cross-influence; an aggregator model synthesizes the final answer

MoA has been shown to outperform GPT-4 on benchmarks by combining smaller open-source models.

### 9.8 Additional Key Concepts

#### Attention Sinks

In streaming models, the first token often acts as an "attention sink" — the model allocates disproportionate attention to it. This can be exploited for infinite context extensions.

#### Scaling Laws

Empirical relationships showing how model performance improves with:
- More parameters
- More training data
- More compute

Key insight: For a given compute budget, there's an optimal balance between parameters and data.

#### Emergent Abilities

Capabilities that appear only above a certain model size threshold:
- Multi-step reasoning
- In-context learning
- Instruction following
- Code generation

Some emergent abilities are related to **model size**; others emerge from **training quality** and **data diversity**.

#### Long-Context Effects

As context windows grow, models often perform worse on the "middle" of the context (lost-in-the-middle phenomenon). Strategies like **attention bias** and **position interpolation** help.

#### Chain-of-Thought (CoT)

Generating intermediate reasoning steps before the final answer. Significantly improves performance on math, logic, and multi-step reasoning tasks.

#### System Prompt vs User Prompt

- **System prompt**: Sets the model's behavior, persona, and constraints (typically hidden from the user)
- **User prompt**: The actual request from the user

#### Tool Use / Function Calling

Modern LLMs can call external tools (APIs, calculators, search engines, databases) by generating structured output (typically JSON) that specifies which function to call and with what arguments.

#### Safety and Alignment

- **Refusal behavior**: The model should refuse harmful requests (e.g., "how to build a bomb")
- **Jailbreaking**: Adversarial prompts designed to bypass safety guardrails
- **Red teaming**: Systematic testing for safety vulnerabilities
- **Constitutional AI**: Anthropic's approach of training models with a set of principles (constitution) rather than human feedback alone

---

## 10. Further Reading

### Foundational Papers

| Paper | Key Contribution |
|---|---|
| *Attention Is All You Need* (Vaswani et al., 2017) | Transformer architecture |
| *Language Models are Few-Shot Learners* (Brown et al., 2020) | GPT-3, scaling, in-context learning |
| *Training Language Models to Follow Instructions* (Ouyang et al., 2022) | InstructGPT / RLHF |
| *LLaMA: Open and Efficient Foundation Language Models* (Touvron et al., 2023) | LLaMA family |
| *Direct Preference Optimization* (Rafailov et al., 2023) | DPO |
| *Scaling Laws for Neural Language Models* (Kaplan et al., 2020) | Scaling laws |
| *Training Compute-Optimal Large Language Models* (Hoffmann et al., 2022) | Chinchilla scaling |
| *GPTQ: Accurate Post-Training Quantization* (Frantar et al., 2022) | GPTQ |
| *AWQ: Activation-aware Weight Quantization* (Lin et al., 2023) | AWQ |
| *Efficient Memory Management for LLM Serving with PagedAttention* (Kwon et al., 2023) | vLLM |

### Online Resources

- **Hugging Face Course**: https://huggingface.co/learn/nlp-course
- **The Annotated Transformer**: https://nlp.seas.harvard.edu/2018/04/03/attention.html
- **llama.cpp**: https://github.com/ggerganov/llama.cpp
- **vLLM**: https://github.com/vllm-project/vllm
- **TensorRT-LLM**: https://github.com/NVIDIA/TensorRT-LLM
- **Ollama**: https://ollama.com
- **lmstudio.ai**: Desktop app for running local models

---

*Document version 1.0 — May 2026. This document is part of the AiBaseKnowledge library. The field evolves rapidly; check for updated versions periodically.*

## Cross-References

| Reference | Description |
|-----------|-------------|
| [02-RAG-Retrieval-Augmented-Generation.md](02-RAG-Retrieval-Augmented-Generation.md) | How LLMs are augmented with external knowledge retrieval |
| [03-MCP-and-ACP-Protocols.md](03-MCP-and-ACP-Protocols.md) | How LLMs connect to external tools via MCP |
| [04-AI-Agents-and-Orchestrators.md](04-AI-Agents-and-Orchestrators.md) | How LLMs power autonomous agents and orchestrators |
| [07-Glossary.md](07-Glossary.md) | Definitions for all key terms (Transformer, Token, Quantization, etc.) |
| [08-AI-Roadmap.md](08-AI-Roadmap.md) | Future directions for LLM architecture and training |
