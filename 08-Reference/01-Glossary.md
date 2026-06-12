# Glossary — Unified Terminology for the AI Base Knowledge Library

> A comprehensive, cross-referenced glossary of all key terms used across the AI Base Knowledge documents. Each entry includes a concise definition (2–4 sentences) and links to the document(s) that cover the topic in depth.

---

## Table of Contents

1. [Foundation Model Concepts](#1-foundation-model-concepts)
2. [Transformer & Architecture](#2-transformer--architecture)
3. [Training & Alignment](#3-training--alignment)
4. [Quantization & Model Formats](#4-quantization--model-formats)
5. [Inference & Decoding](#5-inference--decoding)
6. [RAG & Retrieval](#6-rag--retrieval)
7. [MCP Protocol](#7-mcp-protocol)
8. [ACP Protocol](#8-acp-protocol)
9. [AI Agents & Orchestrators](#9-ai-agents--orchestrators)
10. [Agent Identity & Configuration Files](#10-agent-identity--configuration-files)
11. [AI Coding Tools](#11-ai-coding-tools)
12. [Agent Frameworks & Systems](#12-agent-frameworks--systems)
13. [Evaluation & Benchmarks](#13-evaluation--benchmarks)
14. [Deployment & Serving](#14-deployment--serving)
15. [Emerging Topics](#15-emerging-topics)
16. [Infrastructure & MLOps](#16-infrastructure--mlops)
17. [Performance Metrics & Training Concepts](#17-performance-metrics--training-concepts)
18. [Advanced Learning Paradigms](#18-advanced-learning-paradigms)
19. [Data Engineering & MLOps Pipeline Concepts](#19-data-engineering--mlops-pipeline-concepts)

---

## 1. Foundation Model Concepts

### LLM (Large Language Model)
A neural network model with billions of parameters trained on vast text corpora to understand and generate human language by predicting the next token in a sequence. LLMs exhibit emergent abilities such as reasoning, translation, summarization, and in-context learning that arise from scale rather than explicit programming. They are the core reasoning engine behind modern AI agents, chatbots, and coding assistants.
**See:** 01-LLM-and-AI-Models.md (§2)

### Foundation Model
A large-scale AI model (typically an LLM or multimodal model) trained on broad data at massive scale, designed to be adapted to a wide range of downstream tasks. Foundation models serve as the base layer upon which specialized models (instruct models, fine-tuned variants) are built. The term was popularized by the Stanford CRFM and encompasses models like GPT-4, LLaMA 3, and Claude.
**See:** 01-LLM-and-AI-Models.md (§1, §2)

### Base Model
A model that has undergone only pre-training (or pre-training plus minimal curation) and is capable of text completion but not instruction following. Base models require careful prompt engineering to elicit useful behavior and are primarily used by developers and researchers who wish to fine-tune further. Example: `meta-llama/Llama-3.1-8B` (base).
**See:** 01-LLM-and-AI-Models.md (§5 — Base vs Instruct Models)

### Instruct Model (Chat Model)
A model built by fine-tuning a base model on instruction-response pairs via Supervised Fine-Tuning (SFT), optionally followed by alignment techniques such as RLHF or DPO. Instruct models understand conversation formats, follow user instructions, and are designed for direct use in applications. Example: `meta-llama/Llama-3.1-8B-Instruct`.
**See:** 01-LLM-and-AI-Models.md (§4.2, §5)

### Scaling Laws
Empirical findings that model performance improves predictably with increases in model size (parameters), dataset size (tokens), and compute budget (FLOPs), following power-law relationships. Kaplan et al. (2020) showed that larger models are more sample-efficient, while the Chinchilla scaling law (Hoffmann et al., 2022) demonstrated that for optimal training, model size and training tokens should be scaled proportionally — many existing models are undertrained. Scaling laws guide resource allocation decisions in training frontier models.
**See:** 01-LLM-and-AI-Models.md (§4.1)

### Emergent Abilities
Capabilities that arise spontaneously in sufficiently large language models but are not present in smaller ones, such as multi-step arithmetic, code generation, and in-context learning. Emergent abilities are unpredictable from small-scale extrapolations and often exhibit phase-transition-like behavior where performance jumps sharply at a certain model scale. The existence of emergent abilities motivates the pursuit of ever-larger foundation models and raises important questions about model evaluation.
**See:** 01-LLM-and-AI-Models.md (§2)

### Grokking
A phenomenon where a neural network suddenly generalizes from memorization to true understanding after prolonged training far beyond the point of overfitting. First observed in small algorithmic tasks (modular arithmetic), grokking manifests as a sharp drop in test loss long after training loss has already converged to zero. The mechanism is believed to involve the network slowly compressing memorized solutions into more general, structured representations via mechanisms like weight decay.
**See:** 01-LLM-and-AI-Models.md (§4.4)

### Alignment Tax
The reduction in a model's raw capability (reasoning, creativity, knowledge breadth) incurred during alignment procedures such as RLHF or DPO, which optimize for safety and helpfulness rather than pure task performance. Alignment tax is measured as the performance gap between the base model and its aligned variant on standard benchmarks. Techniques such as conditional RLHF and inference-time alignment aim to reduce this tax.
**See:** 01-LLM-and-AI-Models.md (§4.3)

### Capability vs Alignment
The fundamental tension in AI development between maximizing a model's raw capabilities (intelligence, autonomy, skill breadth) and ensuring its behavior is aligned with human values, safety constraints, and intended use. A purely capable model may find unintended or harmful ways to achieve goals, while over-alignment can reduce helpfulness and autonomy. Managing this trade-off is a central challenge in frontier AI research and deployment.
**See:** 01-LLM-and-AI-Models.md (§4)

### Inference-Time Compute
The computational budget allocated during model inference rather than during training, enabling techniques that trade latency for improved output quality. Examples include chain-of-thought reasoning (using more tokens for intermediate steps), self-consistency (sampling multiple outputs and selecting the best), tree-of-thought search, and best-of-N sampling. Inference-time compute scaling is a growing research area as models are deployed in settings where quality matters more than raw speed.
**See:** 01-LLM-and-AI-Models.md (§8.5, §9.4)

### Tree-of-Thought (ToT)
An extension of Chain-of-Thought prompting that explores multiple reasoning paths simultaneously, evaluating intermediate states and branching only when promising directions are identified. Unlike CoT's linear reasoning trace, ToT treats reasoning as a tree search with deliberate exploration, backtracking, and pruning. ToT significantly improves performance on tasks requiring planning, combinatorial reasoning, and strategic thinking, but at higher inference cost.
**See:** 04-AI-Agents-and-Orchestrators.md (§1.2 — Reasoning)

### Self-Consistency
A decoding strategy that improves answer reliability by generating multiple independent outputs for the same prompt (using temperature sampling) and selecting the most consistent answer via majority voting or marginal aggregation. Self-consistency is particularly effective for math, reasoning, and factoid QA tasks where the correct answer is unambiguous but the reasoning path varies. It provides a simple, training-free method to boost accuracy at the cost of increased inference compute.
**See:** 01-LLM-and-AI-Models.md (§8.5)

---

## 2. Transformer & Architecture

### Transformer
The neural network architecture introduced in the paper "Attention Is All You Need" (Vaswani et al., 2017) that serves as the foundation of virtually all modern LLMs. Transformers replace recurrent layers with a multi-head self-attention mechanism, processing all tokens in parallel rather than sequentially, which enables efficient training on massive datasets. Modern LLMs use a decoder-only variant that stacks transformer layers, each containing self-attention, feed-forward networks, layer normalization, and residual connections.
**See:** 01-LLM-and-AI-Models.md (§3)

### Attention
The core mechanism in Transformer models that allows each token to "attend to" every other token in the sequence, computing weighted representations based on relevance. Attention is implemented via Query, Key, and Value (QKV) projections: each token produces a query vector, and the model computes attention scores by taking the dot product of that query with all key vectors, then using the resulting weights to aggregate the corresponding value vectors. Multi-head attention runs this process in parallel across multiple representation subspaces.
**See:** 01-LLM-and-AI-Models.md (§3.1, §3.5)

### Token
The fundamental unit of text that a language model processes — typically a sub-word unit rather than a whole word. Tokenization converts raw text into token IDs using algorithms such as Byte-Pair Encoding (BPE), WordPiece, or SentencePiece. One token corresponds to approximately 0.75 English words on average, and pricing for LLM APIs is typically per token (input and output billed separately).
**See:** 01-LLM-and-AI-Models.md (§3.2, §9.1)

### Embedding
A dense vector representation that encodes semantic meaning, typically with 1024–8192 dimensions. Each token ID is mapped to a high-dimensional embedding vector via a lookup table (token embeddings), and positional information is injected separately (e.g., via RoPE). Embedding spaces capture semantic relationships — similar words have similar vectors — and the final hidden state at the last token position is commonly used as a contextual representation for downstream tasks.
**See:** 01-LLM-and-AI-Models.md (§3.3, §9.2)

### Context Window
The maximum number of tokens a model can process in a single forward pass, including both the prompt and the generated response. Modern models have context windows ranging from 8K (LLaMA 3) to 128K-200K (GPT-4 Turbo, Claude 3, LLaMA 3.1), with Gemini 1.5 Pro supporting up to 2M tokens. Managing context window limits is a central challenge in agent systems and RAG pipelines.
**See:** 01-LLM-and-AI-Models.md (§8.2)

### KV Cache (Key-Value Cache)
A memory optimization technique that stores the Key and Value matrices from previously generated tokens during autoregressive decoding, avoiding redundant recomputation. The KV cache grows linearly with sequence length and batch size (approximately 1 MB per token for a 8B-parameter model). Optimizations such as PagedAttention, grouped-query attention (GQA), and cache quantization are used to manage its memory footprint.
**See:** 01-LLM-and-AI-Models.md (§8.3)

### MoE (Mixture of Experts)
An architecture that achieves better performance per FLOP by having many specialized sub-networks ("experts") while activating only a subset for each input token. A router network selects the top-k experts (typically k=2) per token, enabling the model to have a large total parameter count (e.g., 671B in DeepSeek-V3) while keeping active parameters per token much smaller (e.g., 37B). MoE models have a large memory footprint but compute cost similar to a smaller dense model.
**See:** 01-LLM-and-AI-Models.md (§6.3)

---

## 3. Training & Alignment

### Fine-tuning
The process of updating a pre-trained model's weights on a dataset of domain-specific examples to adapt it for a particular task or behavior. Fine-tuning can range from full parameter updates to parameter-efficient methods like LoRA (Low-Rank Adaptation) that train small adapter modules. It is distinct from RAG in that it modifies the model's internal weights rather than augmenting its input context.
**See:** 01-LLM-and-AI-Models.md (§4.2, §9.5)

### SFT (Supervised Fine-Tuning)
A training stage after pre-training in which the model is fine-tuned on instruction-response pairs to teach it to follow user instructions and produce helpful, formatted responses. SFT typically uses 10,000–100,000 high-quality examples and optimizes the standard next-token prediction loss only on the response tokens. This transforms a base model (text completion) into an instruct model (conversational).
**See:** 01-LLM-and-AI-Models.md (§4.2)

### RLHF (Reinforcement Learning from Human Feedback)
An alignment technique that trains a model to produce outputs preferred by humans through a three-step pipeline: (1) SFT warmup, (2) training a separate reward model on human preference rankings, and (3) using Proximal Policy Optimization (PPO) to fine-tune the model to maximize the reward model's score while staying close to the original SFT model via a KL penalty. RLHF was popularized by InstructGPT/ChatGPT and is used by most frontier models.
**See:** 01-LLM-and-AI-Models.md (§4.3)

### DPO (Direct Preference Optimization)
A simpler alternative to RLHF that directly optimizes the model on preference pairs without training a separate reward model or using reinforcement learning. DPO uses a loss function that increases the relative log-probability of preferred responses over dispreferred responses. It has largely replaced PPO in open-source fine-tuning pipelines due to its simplicity and competitive performance.
**See:** 01-LLM-and-AI-Models.md (§4.3)

### Distillation (Knowledge Distillation)
A technique that trains a smaller "student" model to mimic the output probability distribution of a larger "teacher" model, using the KL divergence between their logits as an additional loss term. This transfers the teacher's "soft knowledge" — including which alternative tokens are plausible — resulting in a model that is 1/10th the size while retaining 95%+ of the teacher's performance. Distillation differs from pruning (removing weights) and quantization (lowering precision).
**See:** 01-LLM-and-AI-Models.md (§9.6)

### Hallucination
The phenomenon where an LLM generates information that is false, fabricated, or nonsensical but presented as factual. Hallucinations can be intrinsic (contradicting provided context), extrinsic (inventing unverifiable information), or factual (stating incorrect facts). Mitigation strategies include RAG (grounding generation in retrieved documents), chain-of-thought prompting, self-consistency, and constrained decoding.
**See:** 01-LLM-and-AI-Models.md (§9.3)

### Curriculum Learning
A training strategy where examples are presented to the model in order of increasing difficulty, starting with simple patterns and gradually introducing more complex ones. Curriculum learning can accelerate convergence, improve final performance, and reduce catastrophic forgetting by structuring the learning trajectory. In LLM training, curricula are applied during pre-training (ordering data by complexity) and fine-tuning (starting with simple instructions before complex reasoning tasks).
**See:** 01-LLM-and-AI-Models.md (§4)

### RLAIF (Reinforcement Learning from AI Feedback)
A variant of RLHF where preference labels for training the reward model are generated by an LLM (e.g., Claude, GPT-4) rather than gathered from human annotators. RLAIF scales preference data collection dramatically — the judge model compares outputs and provides rankings, often with chain-of-thought reasoning for its ratings. Constitutional AI builds on this by also using AI-generated critiques to revise outputs during training.
**See:** 01-LLM-and-AI-Models.md (§4.3)

### Constitutional AI (CAI)
An alignment technique developed by Anthropic that uses a written constitution (a set of principles and rules) to train models to be helpful, harmless, and honest through AI self-critique and revision. In the first stage (supervised), the model generates outputs, critiques them against the constitution, and revises them. In the second stage (RL), a preference model trained on AI-generated comparisons replaces human feedback, enabling scalable alignment without expensive human annotation.
**See:** 01-LLM-and-AI-Models.md (§4.3)

### KTO (Kahneman-Tversky Optimization)
A preference optimization method inspired by prospect theory (Kahneman & Tversky) that aligns models using binary feedback (thumbs up/thumbs down) rather than pairwise comparisons. Unlike DPO which requires paired preferred/dispreferred responses, KTO works with unpaired examples by optimizing a utility function that treats wins and losses asymmetrically. KTO simplifies data collection and performs competitively with DPO.
**See:** 01-LLM-and-AI-Models.md (§4.3)

### ORPO (Odds Ratio Preference Optimization)
A preference optimization technique that combines supervised fine-tuning and preference alignment into a single training stage by adding an odds ratio loss term to the standard SFT objective. ORPO penalizes the model when it assigns high probability to dispreferred responses relative to preferred ones, eliminating the need for a separate alignment phase. It achieves strong results with fewer training steps than DPO or RLHF.
**See:** 01-LLM-and-AI-Models.md (§4.3)

### SimPO (Simple Preference Optimization)
A simplified preference optimization method that uses the average log-probability of response tokens as an implicit reward, bypassing the need for a reference model entirely. SimPO eliminates the KL divergence term against a frozen reference model (required by DPO), instead directly optimizing the gap between preferred and dispreferred responses. It is computationally lighter than DPO while maintaining competitive alignment quality.
**See:** 01-LLM-and-AI-Models.md (§4.3)

### Preference Optimization
A family of post-training alignment techniques that fine-tune a language model to favor outputs that humans (or AI judges) consider more helpful, harmless, or accurate. Methods include RLHF (PPO with a learned reward model), DPO (direct optimization on preference pairs), KTO (binary feedback optimization), ORPO (combined SFT + alignment), and SimPO (reference-free optimization). Preference optimization is the dominant paradigm for aligning frontier models after the SFT stage.
**See:** 01-LLM-and-AI-Models.md (§4.3)

### Instruction Tuning
The process of fine-tuning a base model on a diverse dataset of (instruction, response) pairs to teach it to follow user instructions in a conversational format. Instruction tuning transforms a text-completion model into a chat model capable of understanding prompts, following formatting rules, and generating helpful responses. High-quality instruction tuning datasets (e.g., OpenAssistant, ShareGPT) are a critical factor in model usability and are distinct from preference data used in alignment.
**See:** 01-LLM-and-AI-Models.md (§4.2)

---

## 4. Quantization & Model Formats

### Quantization
The process of reducing the numerical precision of model weights (and optionally activations) to decrease memory usage and increase inference speed, at the cost of some accuracy. Common precision levels include FP16, INT8, INT4, and NF4. Quantization can be applied post-training (PTQ) or simulated during training (QAT), and the optimal format depends on the target hardware (CPU, consumer GPU, or data-center GPU).
**See:** 01-LLM-and-AI-Models.md (§7)

### GGUF (GPT-Generated Unified Format)
A single-file quantization format created by the llama.cpp community (successor to GGML) that bundles model weights, tokenizer, and metadata into one file. GGUF offers quantization levels from q2_k to q8_0 and is optimized for CPU inference via llama.cpp, making it the standard choice for running models on consumer hardware, laptops, and edge devices.
**See:** 01-LLM-and-AI-Models.md (§7.3)

### GPTQ (GPT-Post-Training Quantization)
A GPU-optimized quantization format based on the paper "GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers." GPTQ stores weights as safetensors with a quantization config and is best used with CUDA-based inference engines such as AutoGPTQ and ExLlama. It offers good GPU performance and efficient batching but requires a calibration dataset.
**See:** 01-LLM-and-AI-Models.md (§7.3)

### AWQ (Activation-aware Weight Quantization)
A quantization format that achieves better quality than GPTQ at the same bit-width by accounting for activation magnitudes during weight quantization. AWQ is optimized for GPU inference via engines such as vLLM, AutoAWQ, and TensorRT-LLM, and has become a preferred choice for production serving due to its quality retention and speed.
**See:** 01-LLM-and-AI-Models.md (§7.3)

---

## 5. Inference & Decoding

### Speculative Decoding
A technique that accelerates text generation without quality loss by using a small "draft" model to quickly generate several candidate tokens, which are then verified in parallel by the large "target" model in a single forward pass. This typically yields a 2–3x speedup and is supported by inference frameworks such as vLLM and TensorRT-LLM.
**See:** 01-LLM-and-AI-Models.md (§8.5)

### Prompt Engineering
The practice of designing input prompts to elicit desired behavior from an LLM, including techniques such as zero-shot prompting, few-shot prompting, chain-of-thought reasoning, role prompting, negative prompting, and structured output formatting. Effective prompt engineering is critical for getting reliable, accurate results from base and instruct models alike.
**See:** 01-LLM-and-AI-Models.md (§9.4)

### System Prompt
A special prompt that sets the model's behavioral guidelines, persona, and constraints, typically hidden from the end user and injected at the start of every conversation. System prompts define the model's role (e.g., "You are a helpful coding assistant"), establish safety guardrails, and configure output formatting preferences. They are distinct from user prompts, which contain the actual request.
**See:** 01-LLM-and-AI-Models.md (§9.4, §9.8)

### Logprobs (Log Probabilities)
The log-probability scores that a language model assigns to each generated token, indicating the model's confidence in its predictions at each generation step. Logprobs are accessible via API parameters (`logprobs=True` in OpenAI/Anthropic APIs) and are used for: detecting low-confidence generations that may need human review, reranking alternative outputs, computing generation uncertainty, implementing threshold-based fallback logic, and debugging model behavior. Higher logprobs (closer to 0) indicate higher confidence; very low logprobs (< -5) indicate the model was uncertain about a token choice. Logprob analysis is an essential tool for production LLM quality monitoring.
**See:** 01-LLM-and-AI-Models.md (§8.1 — Autoregressive Generation)

### Sampling Parameters
The configuration knobs that control how a language model selects the next token from its probability distribution during generation. Key parameters include:
- **Temperature** (0–2): Controls randomness — lower values (0.1-0.3) produce more deterministic, focused outputs; higher values (0.8-1.5) produce more creative, diverse outputs. Temperature 0 is greedy decoding (always pick the most likely token).
- **Top-p / Nucleus Sampling** (0–1): Selects from the smallest set of tokens whose cumulative probability exceeds p. Lower values (0.1-0.3) make output more focused; higher values (0.9-1.0) allow more diversity. Often used together with temperature.
- **Top-k** (1-N): Limits the next token selection to the k most likely tokens. Lower values produce more conservative outputs; k=1 is equivalent to greedy decoding.
- **Frequency / Presence Penalty** (0–2): Discourages token repetition. Frequency penalty reduces the probability of tokens already generated (proportional to their frequency). Presence penalty reduces the probability of any token that has appeared at least once. Both help reduce repetition loops.
- **Max tokens / Stop sequences**: Control output length and termination conditions.

Production recommendation: Start with temperature=0.2, top_p=0.9 for factual tasks; temperature=0.8, top_p=0.95 for creative tasks. Always tune sampling parameters per task — the default rarely suits all use cases.
**See:** 01-LLM-and-AI-Models.md (§8.1)

### Function Calling (Tool Use)
A capability of modern LLMs to generate structured output (typically JSON) specifying which external function or API to call and with what arguments, enabling the model to interact with tools, databases, search engines, and other external systems. Function calling is the model-level mechanism that expresses intent to use a tool, while MCP is the application-level protocol that discovers, describes, and invokes those tools.
**See:** 01-LLM-and-AI-Models.md (§9.8); 03-MCP-and-ACP-Protocols.md (Comparison to Function Calling)

### Prefix Caching
An optimization technique that stores the KV cache for repeated prompt prefixes (e.g., system prompts, long context documents, shared instruction templates) and reuses them across multiple requests. When a new request starts with the same prefix, recomputation of attention for those initial tokens is skipped, reducing time-to-first-token (TTFT) by 30-80% in multi-turn conversations and agent loops. Prefix caching is supported by vLLM (automatic prefix caching), TensorRT-LLM, and most LLM inference APIs.
**See:** 01-LLM-and-AI-Models.md (§8.3)

### Continuous Batching
A serving technique where the inference engine dynamically adds and removes sequences from the active batch as they finish generation, rather than waiting for an entire batch to complete before processing new requests. Continuous batching maximizes GPU utilization by keeping the accelerator busy at all times, especially under variable-length workloads. It is a key differentiator of production-grade inference engines like vLLM and TensorRT-LLM compared to naive batched generation.
**See:** 01-LLM-and-AI-Models.md (§8.4)

### Prompt Caching
A broader optimization that caches the processed representation of common prompt text (including the KV cache and tokenized form) to reduce redundant computation across requests sharing the same prefix. Prompt caching is a superset of prefix caching and is a key feature of Claude API (prompt caching at 1/10th the input token cost for cached content) and other commercial LLM services. Effective prompt cache hit rates can reduce latency by 50-80% and significantly lower API costs.
**See:** 01-LLM-and-AI-Models.md (§8.3)

---

## 6. RAG & Retrieval

### RAG (Retrieval-Augmented Generation)
An AI paradigm that combines a retrieval system with a generative LLM: relevant document chunks are first retrieved from an external knowledge base (vector database or search index) and then fed as context to the LLM to produce a grounded, factually accurate response. RAG addresses LLM limitations including stale knowledge, hallucination, lack of attributability, and domain-specificity, without requiring model retraining.
**See:** 02-RAG-Retrieval-Augmented-Generation.md (§1, §2); 01-LLM-and-AI-Models.md (§9.5)

### Chunking
The process of dividing documents into manageable segments (chunks) before embedding and indexing for RAG retrieval. Chunking strategies include fixed-size overlap splitting, semantic chunking (splitting at natural boundaries), recursive splitting, and agentic chunking. The chunk size, overlap, and strategy directly impact retrieval quality and the amount of context passed to the LLM.
**See:** 02-RAG-Retrieval-Augmented-Generation.md (§3.1, §6)

### Embedding Model
A specialized neural network that converts text into fixed-dimensional dense vectors (embeddings) that capture semantic meaning, enabling similarity search. The same embedding model must be used to embed both documents (during indexing) and queries (during retrieval) to ensure they map to the same vector space. Popular embedding models include OpenAI's text-embedding-3, Cohere's Embed, and open-source models like BGE and E5.
**See:** 02-RAG-Retrieval-Augmented-Generation.md (§3.1, §7)

### Vector Database (Vector Store)
A database optimized for storing embedding vectors and performing efficient Approximate Nearest Neighbor (ANN) search, enabling semantic similarity retrieval at scale. Vector databases support indexing (building search structures over embeddings) and querying (finding the top-k most similar vectors to a query embedding). Examples include Pinecone, Weaviate, ChromaDB, Qdrant, and Milvus.
**See:** 02-RAG-Retrieval-Augmented-Generation.md (§3.1, §8)

### Semantic Search
A search paradigm that retrieves results based on meaning rather than exact keyword matching, powered by embedding models and vector databases. Unlike lexical search (BM25) which matches literal terms, semantic search captures conceptual relationships — e.g., a query about "canine health" can retrieve documents about "dog nutrition." Production RAG systems often combine both approaches in hybrid retrieval with reciprocal rank fusion.
**See:** 02-RAG-Retrieval-Augmented-Generation.md (§3.2)

### Hybrid Search
A retrieval strategy that combines lexical (keyword-based) search using BM25 with semantic (vector-based) search using embedding models, then merges results using Reciprocal Rank Fusion (RRF) or weighted scoring. Hybrid search captures both exact term matches (important for named entities, codes, and technical terms) and semantic similarity (important for conceptual queries), providing more robust retrieval than either method alone. Production RAG systems almost universally employ hybrid search.
**See:** 02-RAG-Retrieval-Augmented-Generation.md (§3.2)

### Re-ranking
A two-stage retrieval process where an initial fast retrieval (e.g., hybrid search returning 50-200 candidates) is followed by a more accurate but slower cross-encoder model that scores each candidate against the query. Re-ranking significantly improves precision by moving the most relevant results to the top, often improving downstream generation quality by 10-30%. Common re-rankers include Cohere Rerank, BGE Reranker, and cross-encoder models.
**See:** 02-RAG-Retrieval-Augmented-Generation.md (§3.3)

### Contextual Retrieval
An advanced RAG technique developed by Anthropic that prepends document-level context to each chunk before embedding, improving retrieval accuracy by helping the embedding model understand the chunk's place within the broader document. For example, a chunk containing "the revenue grew by 20%" might be prefixed with "This chunk is from the 2024 annual report discussing financial performance." Contextual retrieval significantly reduces embedding ambiguity on long, multi-topic documents.
**See:** 02-RAG-Retrieval-Augmented-Generation.md (§6)

### Query Rewriting
A RAG optimization technique where the user's original query is first processed by an LLM to generate a more effective search query — expanding abbreviations, resolving ambiguity, adding synonyms, or decomposing complex questions into sub-queries. Query rewriting bridges the gap between natural language questions and optimal retrieval queries, often improving recall by 20-40%. It is a core component of agentic RAG and advanced retrieval pipelines.
**See:** 02-RAG-Retrieval-Augmented-Generation.md (§3.4)

### Fusion Retrieval
A retrieval strategy that generates multiple query variants from the original question (via query expansion or rewriting) and retrieves documents for each, then merges results through Reciprocal Rank Fusion or weighted voting. Fusion retrieval improves recall by covering different facets of the query — for example, searching for "AI safety," "LLM alignment," and "model risks" from a single question about dangerous AI capabilities. It is particularly effective for complex, multi-faceted information needs.
**See:** 02-RAG-Retrieval-Augmented-Generation.md (§3.4)

### CRAG (Corrective RAG)
A RAG framework that adds a retrieval evaluator to assess whether retrieved documents are relevant before generation. If retrieval quality is low, CRAG triggers corrective actions: it may expand the knowledge source (web search), rewrite the query, or trigger a new retrieval round. If no relevant information can be found, the model is trained to admit its lack of knowledge. CRAG improves robustness against retrieval failures that plague standard RAG pipelines.
**See:** 02-RAG-Retrieval-Augmented-Generation.md (§4.2)

### Self-RAG (Self-Reflective RAG)
An RAG framework where the LLM is trained to generate special reflection tokens that control retrieval and critique its own outputs. Self-RAG interleaves generation with retrieval decisions: the model decides when to retrieve (using a retrieval flag), processes retrieved passages, and evaluates each passage's relevance and support level before incorporating it. This enables on-demand, selective retrieval tailored to each generation step rather than a single retrieval at the start.
**See:** 02-RAG-Retrieval-Augmented-Generation.md (§4.2)

### Agentic RAG
An evolution of RAG where an AI agent manages the entire retrieval process — planning multi-step retrieval strategies, deciding when to search vs. generate, selecting among different data sources (vector DB, web search, APIs, structured databases), and iterating based on intermediate results. Unlike simple RAG which performs a single retrieval pass, Agentic RAG uses tool-use, reasoning loops, and adaptive planning to handle complex information needs that require synthesis across multiple sources.
**See:** 02-RAG-Retrieval-Augmented-Generation.md (§4)

---

## 7. MCP Protocol

### MCP (Model Context Protocol)
An open-standard, client-server protocol developed by Anthropic that provides a standardized interface for LLMs to interact with external systems — databases, file systems, APIs, search engines, code repositories, and any other tool or data source. MCP solves the fragmentation problem where every LLM-to-tool integration was previously a bespoke implementation, analogous to how USB standardized peripheral connections for computers. The protocol uses JSON-RPC 2.0 over transport channels (stdio or SSE) and defines three core primitives: Tools, Resources, and Prompts.
**See:** 03-MCP-and-ACP-Protocols.md (§Part I)

### MCP Host
The top-level application or environment where the LLM operates and the user interacts. The host manages MCP client connections, handles authentication and authorization, controls which servers are available, processes the LLM's tool invocation decisions, and presents results to the user. Examples include Claude Desktop, VS Code extensions, CLI tools, and custom applications that embed an LLM.
**See:** 03-MCP-and-ACP-Protocols.md (Architecture: Host, Client, Server)

### MCP Client
A protocol-level component embedded within the host application that establishes and manages a 1:1 connection with an MCP server. The client handles transport initialization, protocol version negotiation, sending requests (tool calls, resource reads, prompt retrievals), receiving responses, and managing connection state. A single host may run multiple clients simultaneously, each connected to a different server, but the client is distinct from the LLM itself — it is a protocol adapter.
**See:** 03-MCP-and-ACP-Protocols.md (Architecture: Host, Client, Server)

### MCP Server
A lightweight, single-purpose adapter that exposes a specific tool, data source, or service through the MCP protocol. Each server implements the MCP specification, registers its capabilities during initialization (tools, resources, prompts), handles incoming requests, and communicates with the underlying external system. Servers are independently deployable and can run locally (as subprocesses via stdio transport) or remotely (over HTTP via SSE transport).
**See:** 03-MCP-and-ACP-Protocols.md (Architecture: Host, Client, Server)

### Tool (in MCP)
An executable operation that the LLM can invoke through the host application — the "do something" primitive of MCP. Tools are model-initiated, stateless operations with defined input schemas (JSON Schema), and their execution typically requires user approval. Common tool categories include file operations, database queries, API integrations, code operations, web operations, and system commands.
**See:** 03-MCP-and-ACP-Protocols.md (Protocol Primitives — Tools)

### Resource (in MCP)
A data source that can be read by the LLM — the "know something" primitive of MCP. Resources are read-only, URI-addressed data items (text, binary, or structured) that provide context. Clients can list available resources, read their contents, and optionally subscribe to change notifications. Resource templates allow servers to expose dynamic resources with URI patterns (e.g., `file:///{path}`).
**See:** 03-MCP-and-ACP-Protocols.md (Protocol Primitives — Resources)

### Prompt (in MCP)
A pre-defined, reusable prompt template that an MCP server can provide — the "structured interaction" primitive of MCP. Prompts can include dynamic arguments that get substituted at retrieval time and can consist of multiple messages with different roles (system, user, assistant). They serve as domain-specific interaction templates, pre-configured workflows, or contextual assistance patterns.
**See:** 03-MCP-and-ACP-Protocols.md (Protocol Primitives — Prompts)

---

## 8. ACP Protocol

### ACP (Agent Communication Protocol)
An emerging open specification that defines how autonomous AI agents discover each other, exchange messages, coordinate actions, and share information in a multi-agent environment. While MCP standardizes LLM-to-tool communication (vertical integration), ACP standardizes agent-to-agent communication (horizontal integration). ACP is still in early-stage development with no canonical reference implementation, unlike the more mature MCP.
**See:** 03-MCP-and-ACP-Protocols.md (§Part II)

### ACP Transport
The communication channel underlying ACP agent interactions, which may include HTTP, WebSocket, and message queues (to be formally defined as the protocol matures). Unlike MCP's well-defined stdio and SSE transports, ACP's transport layer is still being designed and will need to support discovery mechanisms (static configuration, capability registries, peer-to-peer) and communication patterns (request-response, task delegation, broadcast, pub-sub, negotiation).
**See:** 03-MCP-and-ACP-Protocols.md (§Part II — How Agents Discover and Communicate)

### SSH-SCP Transport (SSP)
Not an ACP transport per se, but a related remote-execution protocol. **SSP (Secure Session Protocol)** is a proprietary protocol used by Claude Code for remote machine control, allowing the coding agent to execute commands and manage files on a remote host. This is distinct from MCP's stdio/SSE transports and ACP's emerging transport specifications. The term "SSH-SCP" is sometimes used conversationally to refer to SSH-based remote agent execution (supported by Hermes Agent via its terminal backends).
**See:** 05-OpenCode-ClaudeCode-and-Hermes-Agent.md (§2.3, §2.4)

---

## 9. AI Agents & Orchestrators

### AI Agent
An autonomous software entity that perceives its environment, processes information, makes decisions, and takes actions to achieve specific goals by leveraging LLMs or other AI models for reasoning. Agents operate in a perception-action cycle: they observe, reason about what to do next, execute actions (typically via tool calls), observe results, and repeat. Core characteristics include autonomy, tool-use, reasoning (ReAct, CoT, reflection), memory (short-term, long-term, episodic), and planning.
**See:** 04-AI-Agents-and-Orchestrators.md (§1)

### AI Orchestrator
An intelligent coordination layer that manages multiple AI agents, tools, sub-systems, and data sources to accomplish complex, multi-step objectives. Unlike a simple agent that executes tasks directly, an orchestrator delegates to specialized sub-agents, monitors progress, aggregates results, and makes high-level decisions about task prioritization, reallocation, and synthesis. An orchestrator is essentially an agent that manages other agents.
**See:** 04-AI-Agents-and-Orchestrators.md (§2)

### Agent Loop
The core execution cycle of an AI agent, typically following a ReAct pattern: the agent receives input, reasons about what to do (Thought), executes a tool call or action (Action), processes the result (Observation), and repeats until the goal is achieved. The agent loop is the fundamental operational pattern that distinguishes autonomous agents from stateless LLM calls.
**See:** 04-AI-Agents-and-Orchestrators.md (§1.4); 05-OpenCode-ClaudeCode-and-Hermes-Agent.md (§1.3, §2.3, §3.3)

### ReAct (Reasoning + Acting)
A widely used agent architecture proposed by Yao et al. (2022) that interleaves reasoning traces with action execution. Instead of generating a full plan upfront, the agent alternates between Thought (reasoning about current state), Action (calling a tool), and Observation (processing the result). ReAct forms the basis of LangChain agents, OpenAI Assistants, and most modern agent implementations.
**See:** 04-AI-Agents-and-Orchestrators.md (§1.4)

### Chain-of-Thought (CoT)
A prompting and reasoning technique where the model generates intermediate reasoning steps before arriving at a final answer, significantly improving performance on math, logic, and multi-step reasoning tasks. CoT can be triggered zero-shot (by adding "Let's think step by step") or via few-shot examples. It is distinct from ReAct in that CoT focuses on reasoning alone, while ReAct interleaves reasoning with external actions.
**See:** 04-AI-Agents-and-Orchestrators.md (§1.2 — Reasoning); 01-LLM-and-AI-Models.md (§9.4, §9.8)

### Task Decomposition
The process by which an orchestrator or agent breaks down a high-level goal into smaller, well-defined sub-tasks that can be assigned to specialized agents or tools. Decomposition can be top-down (recursive breakdown until tasks are atomic), template-based (using predefined workflow patterns), or dynamic (LLM-generated on the fly). Effective decomposition considers dependencies, granularity, and resource requirements.
**See:** 04-AI-Agents-and-Orchestrators.md (§2.3 — Task Decomposition, §4.1)

### Subagent
A specialized AI agent spawned by an orchestrator or parent agent to execute a specific sub-task independently. Subagents operate in their own context with their own tools and instructions, and they report results back to the parent. They enable parallel execution of independent workstreams and are a core building block of scalable multi-agent systems.
**See:** 04-AI-Agents-and-Orchestrators.md (§2.3 — Subagent Management, §4.1)

### Tool-use
The capability of an AI agent to call external tools, APIs, and services to perform actions beyond text generation — such as reading files, executing code, searching the web, sending messages, or querying databases. Tool-use transforms an LLM from a static text generator into an active agent and is typically exposed through a function-calling interface where the agent outputs structured JSON specifying which tool to call and with what parameters.
**See:** 04-AI-Agents-and-Orchestrators.md (§1.2 — Tool-Use); 01-LLM-and-AI-Models.md (§9.8)

### Autonomous Agent
An AI agent that operates without human intervention for extended periods, making independent decisions about tool selection, error recovery, and goal satisfaction. The degree of autonomy varies: some agents require human-in-the-loop approval for critical actions (e.g., file deletion, API calls with side effects), while others operate fully autonomously with only high-level goal supervision.
**See:** 04-AI-Agents-and-Orchestrators.md (§1.2 — Autonomy)

### Human-in-the-Loop (HITL)
A design pattern where human oversight is integrated into the agent's decision-making process at specific checkpoints, typically for high-risk actions, ambiguous situations, or quality gates. HITL can be implemented at the per-action level (user must approve each tool call), the per-task level (user reviews intermediate results), or the workflow level (user approves the overall plan before execution).
**See:** 04-AI-Agents-and-Orchestrators.md (§1.2, §2.3, §3.3); 03-MCP-and-ACP-Protocols.md (Security Considerations)

### Orchestration Pattern
A structured approach to coordinating multiple sub-tasks or agents within an orchestrator workflow. The five primary patterns are:
- **Sequential**: Sub-tasks execute one after another in dependency order; simple but slow.
- **Parallel**: Independent sub-tasks execute concurrently; faster but requires dependency analysis.
- **Hierarchical**: A tree structure with manager agents overseeing worker agents; scalable but adds latency.
- **Recursive**: Complex sub-tasks are further decomposed on demand; handles unknown complexity gracefully.
- **Hybrid**: Combines multiple patterns within a single workflow; used in virtually all production systems.
**See:** 04-AI-Agents-and-Orchestrators.md (§5)

### Multi-Agent System
A system in which multiple AI agents collaborate, coordinate, and communicate to accomplish goals that exceed the capability of any single agent. Multi-agent systems can use various coordination mechanisms (ACP for communication, orchestrators for task delegation, shared memory for state) and patterns (role-based teams, debate/consensus, market-based allocation). Frameworks like CrewAI, AutoGen, and LangGraph specialize in building multi-agent systems.
**See:** 04-AI-Agents-and-Orchestrators.md (§1.5); 03-MCP-and-ACP-Protocols.md (§Part II)

### Agent Framework
A software library or platform that provides the infrastructure for building, deploying, and managing AI agents. Key frameworks include **LangChain/LangGraph** (most widely adopted, extensive integration ecosystem), **CrewAI** (role-based multi-agent teams), **AutoGen** (Microsoft Research, flexible conversation patterns), **Semantic Kernel** (enterprise, Azure-integrated), and **OpenAI Assistants API** (managed platform). These frameworks abstract away common agent patterns such as tool-use, memory, and orchestration.
**See:** 04-AI-Agents-and-Orchestrators.md (§1.5)

### Episodic Memory
A component of an agent's memory system that stores specific past experiences, actions, and outcomes in sequential order, analogous to human episodic memory of events. In agent systems, episodic memory enables the agent to recall what happened in previous sessions, learn from past mistakes, and maintain continuity across interactions. It is typically implemented as a database of past episodes (conversations, tool calls, outcomes) indexed by time and retrievable by semantic similarity.
**See:** 04-AI-Agents-and-Orchestrators.md (§1.3 — Memory)

### Semantic Memory
An agent's long-term store of factual knowledge, concepts, and relationships extracted from past experiences, analogous to human semantic memory of facts and meanings. Unlike episodic memory (which stores raw experiences), semantic memory stores distilled knowledge — patterns, entity attributes, and procedural insights. In Hermes Agent, semantic memory is implemented through MEMORY.md files, FTS5 search, and the Honcho dialectic modeling system.
**See:** 04-AI-Agents-and-Orchestrators.md (§1.3 — Memory); 05-OpenCode-ClaudeCode-and-Hermes-Agent.md (§3.2)

### Reflection
An agent's ability to evaluate its own outputs, reasoning, and decisions to identify errors, inconsistencies, or areas for improvement before taking further action. Reflection can be applied at multiple levels: token-level (self-consistency checks), step-level (critiquing intermediate reasoning), task-level (reviewing completed work for quality), and meta-level (evaluating overall strategy). It is a key capability for building reliable, self-improving agents.
**See:** 04-AI-Agents-and-Orchestrators.md (§1.2 — Reasoning, §2.3)

### Mixture of Agents (MoA)
A multi-agent architecture where multiple LLM agents generate responses independently, and a separate aggregator agent synthesizes the best response from the pool. MoA improves output quality by leveraging the diverse strengths and knowledge of different models — each "expert" contributes its perspective, and the aggregator resolves differences and selects optimal content. This approach has been shown to outperform individual frontier models on benchmarks like AlpacaEval and MT-Bench.
**See:** 04-AI-Agents-and-Orchestrators.md (§1.5)

### Agent Supervisor
A specialized agent or orchestration layer responsible for monitoring, coordinating, and managing the lifecycle of worker agents in a multi-agent system. The supervisor assigns tasks, tracks progress, handles errors, manages resource allocation, and enforces guardrails. It operates at a higher level of abstraction than individual agents, focusing on workflow integrity rather than specific task execution.
**See:** 04-AI-Agents-and-Orchestrators.md (§2.2)

### Delegation
The ability of an agent to offload sub-tasks to other agents (subagents) or external systems, enabling parallel execution and specialization. Effective delegation requires the delegator to clearly specify the task, provide necessary context, set success criteria, and process the results upon completion. Delegation is a core orchestration primitive and is distinct from tool-use in that the delegate has its own agency and decision-making capability.
**See:** 04-AI-Agents-and-Orchestrators.md (§2.3)

### Task Queue
A data structure or service that manages the ordered execution of pending tasks in an agent or orchestration system. Tasks are enqueued with metadata (priority, dependencies, timeout, retry policy) and processed by available workers. Task queues enable concurrency control, backpressure management, and guaranteed execution semantics (at-least-once, exactly-once) in production agent deployments.
**See:** 04-AI-Agents-and-Orchestrators.md (§2.3)

### Guardrails
Programmable constraints and validation rules that govern an AI agent's behavior, ensuring outputs stay within defined boundaries of safety, legality, accuracy, and appropriateness. Guardrails can be applied pre-generation (input validation, prompt injection detection, topic filtering), during generation (output length limits, structured output validation), or post-generation (toxicity checks, factuality verification, policy compliance scanning). They are a critical safety layer between the LLM and end users.
**See:** 04-AI-Agents-and-Orchestrators.md (§3.3)

### Safety Filters
Post-processing checks that screen agent outputs for harmful content including toxicity, bias, personally identifiable information (PII), malicious code, unsafe URLs, and policy violations. Safety filters typically use a combination of blocklists, classifier models (e.g., Llama Guard, Azure Content Safety), and rule-based checks. They serve as the last line of defense before delivering agent responses to users and are complementary to alignment techniques applied during training.
**See:** 04-AI-Agents-and-Orchestrators.md (§3.3)

---

## 10. Agent Identity & Configuration Files

### SOUL.md
A Markdown file that defines the core identity, personality, values, and behavioral guidelines for an AI agent or software project. SOUL.md serves as the agent's "constitution" — a persistent reference the agent reads at session start to understand what it is, what it stands for, and how it should behave. It is injected into the agent's context window as a system prompt, ensuring consistent behavior across sessions. The name is intentionally philosophical: it captures the idea that a project has a spirit or character beyond technical specifications.
**See:** 06-SOUL-and-SKILL.md (§2)

### SKILL.md
A structured Markdown file that defines a specific, repeatable procedure or workflow that an AI agent can execute — a unit of procedural knowledge. Unlike SOUL.md (which defines who the agent is), SKILL.md defines what the agent can do. It includes YAML frontmatter (name, description, triggers), numbered execution steps, pitfalls to avoid, and verification criteria. Skills form the agent's "procedural memory" and can be shared, versioned, and composed.
**See:** 06-SOUL-and-SKILL.md (§3)

### AGENTS.md
A project-level instruction file that tells AI agents how to interact with a specific codebase, covering navigation conventions, testing standards, build procedures, and documentation expectations. AGENTS.md blends aspects of both SOUL.md (behavioral rules) and SKILL.md (procedures) into a single, shorter file. It originated in the open-source AI coding ecosystem and is commonly used as a lightweight alternative to separate SOUL.md and SKILL.md files.
**See:** 06-SOUL-and-SKILL.md (§4.1)

### CLAUDE.md
A project-level configuration file specific to Anthropic's Claude Code that tells the assistant how to behave in a given repository. It typically includes build/test/lint commands, project conventions, and communication preferences. CLAUDE.md is more limited in scope than SOUL.md (focusing on practical coding instructions rather than deep identity) and less structured than SKILL.md (no YAML frontmatter or numbered steps).
**See:** 06-SOUL-and-SKILL.md (§4.2)

### .cursorrules
A project-specific configuration file for Cursor IDE (an AI-powered code editor) that defines rules for how the AI assistant should behave within a project. It includes code style preferences, framework-specific instructions, and response formatting preferences. .cursorrules is editor-specific and uses a JSON configuration format, making it less flexible than SOUL.md/SKILL.md but well-integrated into the Cursor ecosystem.
**See:** 06-SOUL-and-SKILL.md (§4.3)

### Agent Identity
The concept, formalized in SOUL.md, that an AI agent should have a defined character encompassing its name, role, core values, communication style, behavioral constraints, and domain knowledge. Agent identity is injected as context at session start and persists across interactions, providing a stable foundation for decision-making and human-AI alignment. It is the "who" that precedes the "how" (skills).
**See:** 06-SOUL-and-SKILL.md (§2.1, §2.4)

### Procedural Memory
An agent's library of encoded procedures for performing specific tasks, analogous to the part of human memory that stores knowledge of how to do things. In AI agent systems, procedural memory is implemented as a collection of SKILL.md files (or equivalent skill definitions) that the agent loads and executes when triggered. Procedural memory is granular, composable, learnable (new skills can be added without retraining), shareable, versionable, and debuggable.
**See:** 06-SOUL-and-SKILL.md (§3.7)

### AGENTS.md / CONTEXT.md
Project-level instruction files that provide task-specific context to an AI agent for a given repository or workspace. AGENTS.md declares project conventions, domain knowledge, tool preferences, and workflow instructions that the agent loads at session start alongside its permanent identity (SOUL.md). CONTEXT.md serves a similar role but is typically lighter — focusing on session-specific context rather than persistent project rules. These files complement the agent's permanent memory (SOUL.md, MEMORY.md) by supplying ephemeral project context that may differ between workspaces. Hermes Agent loads both AGENTS.md and CONTEXT.md from the current working directory at session start.
**See:** 06-SOUL-and-SKILL.md (§4.4–4.5)

---

## 11. AI Coding Tools

### OpenCode
An open-source, terminal-based AI coding agent built by the opencode-ai community, written in Go with a rich TUI (Bubble Tea framework), multi-provider AI support, LSP integration, and MCP connectivity. The original `opencode-ai/opencode` repository has been archived; the project has evolved into **Crush** by the Charm team, which inherits and extends OpenCode's architecture with industrial-grade cross-platform support and enhanced MCP (HTTP, stdio, SSE).
**See:** 05-OpenCode-ClaudeCode-and-Hermes-Agent.md (§1)

### Crush
The active successor to OpenCode, developed by the Charm team (charmbracelet/crush). Crush adds industrial-grade terminal support across macOS, Linux, Windows, Android, FreeBSD, OpenBSD, and NetBSD, plus multi-model mid-session switching and enhanced MCP support. It is written in Go under the MIT license.
**See:** 05-OpenCode-ClaudeCode-and-Hermes-Agent.md (§1.5)

### Claude Code
A proprietary, agentic coding tool built by Anthropic that lives in the terminal, understands codebases, and assists with coding tasks through natural language. Written in TypeScript/Node.js, Claude Code features an advanced agentic loop, enterprise-grade prompt caching, MCP integration, SSP remote control, a plugin system, and CLAUDE.md project-level memory. It is tied exclusively to Anthropic's Claude models and is available as a free (rate-limited) or paid per-token service.
**See:** 05-OpenCode-ClaudeCode-and-Hermes-Agent.md (§2)

### Hermes Agent
A self-improving, multi-platform AI agent built by Nous Research, written in Python under the MIT license. Hermes Agent features a closed learning loop (autonomous skill creation from experience), persistent memory (MEMORY.md, USER.md, FTS5 search, Honcho dialectic modeling), a skills system (procedural memory with auto-generated skills), subagent delegation, scheduled automations (cron), multi-platform support (Telegram, Discord, Slack, WhatsApp, Signal, CLI), 20+ model providers, full MCP integration, and multiple terminal backends (Local, Docker, SSH, Modal, Daytona).
**See:** 05-OpenCode-ClaudeCode-and-Hermes-Agent.md (§3)

---

## 12. Agent Frameworks & Systems

### LangChain / LangGraph
The most widely adopted framework (Python/JS) for building LLM-powered agents. LangChain provides chains, agents, tools, memory, and callbacks; its LangGraph extension enables cyclic graphs (loops, branching) for multi-step agent workflows with streaming, human-in-the-loop, checkpointing, and persistent state.
**See:** 04-AI-Agents-and-Orchestrators.md (§1.5)

### CrewAI
A multi-agent orchestration framework that models agent teams as "crews" with role-based agent design (researcher, writer, etc.) and built-in collaboration patterns. CrewAI supports sequential and hierarchical orchestration modes and is designed for easy setup of multi-agent systems, making it popular for content generation, research, and automation workflows.
**See:** 04-AI-Agents-and-Orchestrators.md (§1.5)

### AutoGen (Microsoft)
A multi-agent conversation framework from Microsoft Research featuring AssistantAgent, UserProxyAgent, GroupChat, and tool registration with type-safe schemas. AutoGen provides built-in code execution sandboxes, flexible conversation patterns, nested chats, and human-in-the-loop capabilities, making it well-suited for complex multi-agent scenarios.
**See:** 04-AI-Agents-and-Orchestrators.md (§1.5)

---

## 13. Evaluation & Benchmarks

### Perplexity (PPL)
A measurement of how well a language model predicts a sample, calculated as the exponentiated average negative log-likelihood of the test tokens. Lower perplexity indicates the model assigns higher probability to the actual text, meaning it has learned the data distribution more accurately. While perplexity is a useful intrinsic metric for tracking training progress, it does not always correlate with downstream task performance or human preference.
**See:** 01-LLM-and-AI-Models.md (§9.7)

### BLEU (Bilingual Evaluation Understudy)
An automatic metric for evaluating generated text quality by measuring n-gram overlap between the model output and one or more reference texts, with a brevity penalty to discourage short outputs. BLEU scores range from 0 to 100 and are most commonly used for machine translation and summarization tasks. It is widely criticized for favoring lexical over semantic similarity and has been largely superseded by neural metrics and LLM-based evaluation.
**See:** 01-LLM-and-AI-Models.md (§9.7)

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
A set of automatic metrics for evaluating text summarization by comparing generated summaries to reference summaries via n-gram recall, longest common subsequence (ROUGE-L), and skip-bigram co-occurrence (ROUGE-S). Unlike BLEU which measures precision, ROUGE primarily measures recall — whether the generated summary captures the key information from the reference. Both BLEU and ROUGE are increasingly supplemented by LLM-based judges for quality evaluation.
**See:** 01-LLM-and-AI-Models.md (§9.7)

### MMLU (Massive Multitask Language Understanding)
A benchmark consisting of approximately 14,000 multiple-choice questions across 57 academic subjects spanning STEM, humanities, social sciences, and professional fields. MMLU tests a model's breadth of knowledge and reasoning ability at varying difficulty levels. It was the de facto standard for evaluating frontier model capabilities from 2021 to 2024 and has been succeeded by harder benchmarks such as GPQA and MMLU-Pro.
**See:** 01-LLM-and-AI-Models.md (§9.7)

### HumanEval
A benchmark for evaluating code generation capabilities, consisting of 164 hand-written programming problems with unit tests. Each problem specifies a function signature, docstring with description, and expected behavior. Models are evaluated on the pass@k metric (the probability that at least one of k generated samples passes all tests). HumanEval has been succeeded by more comprehensive coding benchmarks such as MBPP, SWE-bench, and BigCodeBench.
**See:** 01-LLM-and-AI-Models.md (§9.7)

### GPQA (Google-Proof Q&A)
A challenging benchmark of graduate-level multiple-choice questions in biology, physics, and chemistry, designed to be difficult even for expert humans with internet access. GPQA questions require deep domain expertise and multi-step reasoning. It serves as a test of expert-level knowledge and is one of the few benchmarks where frontier models still show significant room for improvement.
**See:** 01-LLM-and-AI-Models.md (§9.7)

### Arena Elo
A model ranking metric derived from the Chatbot Arena, a crowdsourced platform where users chat with two anonymous models and vote for the better response. Elo scores are computed from pairwise comparisons using a Bradley-Terry model, similar to chess rankings. Arena Elo captures human preference better than static benchmarks because it reflects real-world usage and subjective quality, though it is influenced by user demographics and model popularity.
**See:** 01-LLM-and-AI-Models.md (§9.7)

### MATH (Mathematics Aptitude Test of Heuristics)
A benchmark of 12,500 challenging mathematics competition problems spanning precalculus, calculus, algebra, geometry, number theory, and probability. Each problem requires multi-step reasoning with chain-of-thought generation. MATH is a standard evaluation for mathematical reasoning capability and is complemented by GSM8K for grade-school-level math and AIME/IMO for olympiad-level problems.
**See:** 01-LLM-and-AI-Models.md (§9.7)

### GSM8K (Grade School Math 8K)
A benchmark of 8,500 grade-school-level math word problems requiring 2-8 steps of arithmetic reasoning, designed to test multi-step chain-of-thought reasoning. GSM8K was instrumental in demonstrating the effectiveness of CoT prompting and is widely used as a quick check of basic reasoning capability. Despite being relatively simple, it remains a standard because it tests whether models can maintain coherent multi-step reasoning without hallucinating intermediate values.
**See:** 01-LLM-and-AI-Models.md (§9.7)

### IFEval (Instruction-Following Evaluation)
A benchmark that tests a model's ability to follow verifiable instructions such as formatting requirements, length constraints, keyword inclusion, and structural rules. Unlike knowledge-heavy benchmarks, IFEval focuses purely on instruction adherence — whether the model does exactly what it is told. It is a key metric for evaluating instruction-tuned models and has strong correlation with human evaluation of instruction following.
**See:** 01-LLM-and-AI-Models.md (§9.7)

---

## 14. Deployment & Serving

### Tensor Parallelism (TP)
A distributed inference technique that splits individual model layers across multiple GPUs by partitioning weight matrices along the column or row dimension. Each GPU holds a slice of each layer and computes its portion of the attention and feed-forward operations, communicating intermediate results via all-reduce operations. Tensor parallelism is essential for serving models too large to fit on a single GPU and is typically used alongside pipeline parallelism for maximum throughput.
**See:** 01-LLM-and-AI-Models.md (§8.4)

### Pipeline Parallelism (PP)
A distributed inference technique that partitions model layers across multiple GPUs, so each GPU computes a subset of consecutive layers. Unlike tensor parallelism (which splits within a layer), pipeline parallelism splits along the layer dimension — layers 1-8 on GPU 0, layers 9-16 on GPU 1, etc. Pipeline parallelism reduces inter-GPU communication overhead compared to TP but introduces idle time (bubble) at the start and end of each batch. It is commonly combined with TP in production deployments.
**See:** 01-LLM-and-AI-Models.md (§8.4)

### Model Router
A serving layer that directs each inference request to the optimal model based on task type, latency requirements, cost constraints, and expected quality. A model router may implement tiered serving (try the cheapest model first, escalate to more expensive models on failure), task-specific routing (math questions to a math-specialized model, creative writing to a general model), or load-balanced routing across multiple instances of the same model. Model routers are a key component of production AI infrastructure.
**See:** 01-LLM-and-AI-Models.md (§8.4)

---

## 15. Emerging Topics

### Agentic AI
A paradigm shift from passive question-answering AI to autonomous systems that can independently pursue goals, make decisions, take actions, and adapt to changing circumstances without continuous human guidance. Agentic AI systems combine LLM reasoning with tool-use, memory, planning, self-reflection, and delegation to operate autonomously over extended periods. This represents the evolution from tools that answer questions to agents that solve problems, and is the driving vision behind systems like Hermes Agent, Claude Code, and AutoGPT.
**See:** 04-AI-Agents-and-Orchestrators.md (§1); 05-OpenCode-ClaudeCode-and-Hermes-Agent.md (§3)

### World Models
Internal representations that encode an understanding of how the environment works, enabling an AI to simulate possible future states before taking action. World models allow agents to perform mental simulation, counterfactual reasoning, and planning by predicting the consequences of actions without executing them in the real world. They are a core component of System 2 thinking architectures and are foundational to reinforcement learning approaches in robotics and game-playing AI (e.g., MuZero, Dreamer).
**See:** 04-AI-Agents-and-Orchestrators.md (§1.2 — Planning)

### System 2 Thinking
A concept borrowed from cognitive science (Daniel Kahneman) describing slow, deliberate, analytical reasoning that is resource-intensive but more accurate than fast, intuitive System 1 thinking. In AI, System 2 thinking corresponds to techniques like chain-of-thought reasoning, tree-of-thought search, self-consistency, reflection, and formal verification — methods that trade additional compute for improved reliability. Building agents with explicit System 2 capabilities is an active research frontier in reliable AI.
**See:** 04-AI-Agents-and-Orchestrators.md (§1.2 — Reasoning)

### Self-Improvement
The capability of an AI system to autonomously enhance its own performance, knowledge, or capabilities through iterative cycles of self-evaluation, reflection, and adjustment. Forms of self-improvement include self-play (improving through repeated practice against itself), self-training (generating training data from its own outputs and retraining), skill acquisition (creating and saving new reusable procedures), and meta-learning (learning how to learn more effectively). Self-improvement is a key step toward autonomous AI development and a central design goal of Hermes Agent's closed learning loop.
**See:** 05-OpenCode-ClaudeCode-and-Hermes-Agent.md (§3.2)

---

## 16. Infrastructure & MLOps

### vLLM
An open-source, high-throughput LLM serving engine developed at UC Berkeley that implements PagedAttention for efficient KV-cache memory management. vLLM features continuous batching, tensor/pipeline parallelism, prefix caching, speculative decoding, FP8/INT4 quantization, and OpenAI-compatible API endpoints. It is the most widely adopted open-source model serving framework for production deployments.
**See:** 01-LLM-and-AI-Models.md (§8.3–8.5); 05-Enterprise/04-AI-Infrastructure.md

### CUDA (Compute Unified Device Architecture)
NVIDIA's parallel computing platform and programming model that enables general-purpose computing on GPUs. CUDA is the foundational software layer for virtually all deep learning frameworks (PyTorch, TensorFlow, JAX). Key components include CUDA cores/tensor cores (hardware), cuBLAS (matrix operations), cuDNN (neural network primitives), and NCCL (multi-GPU communication). Understanding CUDA memory hierarchy (global, shared, local, registers) is critical for performance optimization.

### NCCL (NVIDIA Collective Communications Library)
A GPU-accelerated library for multi-GPU and multi-node communication that implements collective operations (all-reduce, all-gather, broadcast, reduce-scatter). NCCL is the backbone of distributed training across GPUs — used by PyTorch DDP/FSDP, DeepSpeed, and Megatron-LM. It leverages NVLink (intra-node) and InfiniBand/RoCE (inter-node) for maximum bandwidth.
**See:** 05-Enterprise/04-AI-Infrastructure.md (§3 — Distributed Training)

### HNSW (Hierarchical Navigable Small World)
The state-of-the-art algorithm for Approximate Nearest Neighbor (ANN) search in vector databases. HNSW constructs a multi-layer graph where upper layers have fewer, longer-range connections (for coarse search) and lower layers have dense, short-range connections (for fine search). It offers O(log N) search complexity and 95%+ recall with 10× less memory than exhaustive search. Nearly every production vector database (Pinecone, Weaviate, Qdrant, Milvus) uses HNSW as its default index.
**See:** 04-RAG/03-Vector-Databases.md (§3 — Indexing)

### ANN (Approximate Nearest Neighbor)
A family of algorithms that trade exactness for speed in high-dimensional similarity search, returning the "good enough" nearest neighbors rather than the exact ones. ANN methods include HNSW (graph-based, state-of-the-art for quality), IVF (inverted file indexing, good for high-dimensional), and product quantization (compression-based, lowest memory). ANN indexing is the core technology enabling vector databases to search billions of vectors in milliseconds.
**See:** 04-RAG/03-Vector-Databases.md (§2 — Search Algorithms)

### MLflow
An open-source platform for the ML lifecycle, covering experiment tracking (metrics, params, artifacts), model packaging (MLflow Models format), model registry (versioning, stage transitions), and deployment (to REST APIs, batch, Spark). MLflow is the most widely adopted MLOps framework and is used by organizations of all sizes for managing the end-to-end machine learning lifecycle.
**See:** 05-Enterprise/04-AI-Infrastructure.md (§5 — MLOps)

### Weights & Biases (W&B)
A commercial MLOps platform for experiment tracking, dataset versioning, hyperparameter sweeps, model registry, and LLM monitoring (W&B Prompts). W&B provides a centralized dashboard for visualizing training runs, comparing model versions, and collaborating across teams. It integrates with all major ML frameworks and is widely used in both research and production settings for tracking deep learning experiments.
**See:** 05-Enterprise/03-Fine-Tuning-Enterprise.md (§4 — Experiment Tracking)

### Langfuse
An open-source observability and tracing platform specifically designed for LLM applications. Langfuse captures token usage, latency, prompt versions, model parameters, and generation quality for every LLM call. It provides a trace viewer for debugging complex chains and agent loops, cost tracking across models, prompt management with versioning, and evaluation datasets for regression testing. It is a critical tool for production LLM applications alongside similar platforms like Arize Phoenix, Helicone, and LangSmith.
**See:** 05-Enterprise/01-Enterprise-AI-Deployment.md (§6 — Observability)

### ONNX (Open Neural Network Exchange) / ONNX Runtime
An open-source format for representing machine learning models across frameworks (PyTorch, TensorFlow, JAX). ONNX defines a standard operator set and graph format, enabling model portability between training and inference environments. ONNX Runtime is a cross-platform inference engine that optimizes ONNX models through graph transformations, operator fusion, and quantization. It is widely used for production deployment on CPU, GPU (CUDA/TensorRT), and edge devices (mobile, IoT), offering 1.5-3× inference speedup over raw PyTorch eager mode.
**See:** 05-Enterprise/04-AI-Infrastructure.md (§4 — Inference Optimization)

### RAGAS (Retrieval-Augmented Generation Assessment)
An evaluation framework for RAG pipelines that measures four key metrics: faithfulness (does the answer align with retrieved context?), answer relevancy (does the answer address the question?), context precision (are retrieved documents relevant?), and context recall (are all relevant documents retrieved?). RAGAS uses an LLM as a judge to score these metrics without requiring ground-truth labels, enabling automated RAG quality evaluation at scale.
**See:** 04-RAG/02-Advanced-RAG.md (§8 — Evaluation); 06-Advanced/03-Evaluation-Benchmarks.md

### Docker for AI
Containerization practices specific to AI workloads. Key patterns include: model-in-image (packaging models directly in Docker images for consistent inference), sidecar containers (separate pre/post-processing services alongside the inference server), GPU passthrough (using `nvidia-container-toolkit` with `--gpus all`), multi-stage builds (separating build-time PyTorch from runtime), and Hugging Face optimized images (`huggingface/text-generation-inference`, `vllm/vllm-openai`).
**See:** 05-Enterprise/04-AI-Infrastructure.md (§4 — Containerization)

### DeepSpeed
Microsoft's deep learning optimization library that enables training of extremely large models through ZeRO (Zero Redundancy Optimizer) memory optimization. ZeRO stages: Stage 1 partitions optimizer states, Stage 2 partitions gradients, Stage 3 partitions model parameters. DeepSpeed also provides pipeline parallelism, Mixture of Experts (MoE) training, sparse attention, and inference optimizations. It is the standard library for training models larger than 10B parameters on a single GPU and beyond.
**See:** 01-Foundations/05-Training-Methodologies.md (§6 — Distributed Training); 05-Enterprise/04-AI-Infrastructure.md (§3)

### TensorRT-LLM
NVIDIA's inference optimization library that maximizes LLM throughput on NVIDIA GPUs through graph optimization, kernel fusion, INT4/INT8/FP8 quantization, in-flight batching, paged attention, and multi-GPU multi-node inference. TensorRT-LLM supports all major model architectures and is the primary serving engine for enterprise NVIDIA deployments, competing with vLLM for production inference workloads.
**See:** 05-Enterprise/04-AI-Infrastructure.md (§4 — Inference Optimization)

---

## 17. Performance Metrics & Training Concepts

This section defines fundamental ML performance metrics and training methodologies used throughout the AI Base Knowledge library.

### Accuracy
The simplest classification metric: the ratio of correct predictions to total predictions. Accuracy = (TP + TN) / (TP + TN + FP + FN). While intuitive, accuracy is misleading for imbalanced datasets — a model that always predicts the majority class can achieve high accuracy while being useless. Prefer precision, recall, or F1 for imbalanced tasks.

### Precision
The fraction of positive predictions that are actually correct: Precision = TP / (TP + FP). High precision means few false positives — critical for applications like spam detection (you don't want legitimate emails marked as spam) and medical diagnosis (you don't want to falsely flag a healthy patient).
**See:** [01-Foundations/02-Machine-Learning.md] (§4 — Classification)

### Recall (Sensitivity / True Positive Rate)
The fraction of actual positives that are correctly identified: Recall = TP / (TP + FN). High recall means few false negatives — critical for applications like cancer screening (you don't want to miss a positive case) and fraud detection (you want to catch as much fraud as possible, even if it means some false alarms).
**See:** [01-Foundations/02-Machine-Learning.md] (§4 — Classification)

### F1 Score
The harmonic mean of precision and recall: F1 = 2 · (Precision · Recall) / (Precision + Recall). F1 provides a single score that balances both precision and recall, making it ideal for comparing models on imbalanced datasets. The harmonic mean penalizes extreme imbalances — if either precision or recall is 0, F1 is 0 regardless of the other.
**See:** [01-Foundations/02-Machine-Learning.md] (§4 — Classification)

### ROC-AUC (Receiver Operating Characteristic — Area Under Curve)
A metric that measures the model's ability to distinguish between classes across all classification thresholds. The ROC curve plots TPR (recall) vs FPR (1 - specificity) at each threshold. AUC=1.0 is perfect; AUC=0.5 is random. ROC-AUC is threshold-independent and robust to class imbalance, making it a standard metric for binary classification evaluation alongside PR-AUC (Precision-Recall AUC), which is preferred for highly imbalanced datasets.
**See:** [01-Foundations/02-Machine-Learning.md] (§4 — Classification)

### Confusion Matrix
A table that summarizes classification performance by comparing actual vs predicted labels. For binary classification: four cells — True Positives (TP), True Negatives (TN), False Positives (FP, Type I error), False Negatives (FN, Type II error). All other classification metrics (accuracy, precision, recall, F1, specificity) are derived from these four values. For multi-class classification, the confusion matrix is N×N where N is the number of classes.

### Backpropagation (Backprop)
The fundamental algorithm for training neural networks, which computes the gradient of the loss function with respect to each weight by applying the chain rule backward through the network. Backprop consists of: (1) forward pass — compute predictions, (2) loss computation — compare to targets, (3) backward pass — compute gradients layer-by-layer from output to input, (4) weight update — adjust parameters using optimizer (SGD, Adam, etc.). Backprop is the computational foundation underlying virtually all deep learning training.
**See:** [01-Foundations/03-Deep-Learning.md] (§2 — Neural Networks)

### Gradient Descent
An optimization algorithm that iteratively updates model parameters in the direction of the negative gradient of the loss function. Variants: Batch GD (full dataset per step — accurate but slow), Stochastic GD (one sample per step — fast but noisy), Mini-batch GD (small batch per step — best trade-off). Learning rate controls step size. Modern optimizers (Adam, AdamW, SGD with momentum) extend basic GD with adaptive learning rates and momentum terms.
**See:** [01-Foundations/03-Deep-Learning.md] (§3 — Training)

### Normalization
Techniques that rescale data or activations to improve training stability and convergence:
- **Batch Normalization (BN):** Normalizes activations across the batch dimension; reduces internal covariate shift; allows higher learning rates; adds learnable scale/shift parameters (γ, β). BN behavior differs between training (batch statistics) and inference (running averages).
- **Layer Normalization (LN):** Normalizes across feature dimension; independent of batch size; preferred for RNNs and Transformers (used in all modern LLMs). LN computes mean/variance over the hidden dimension rather than the batch, making it suitable for variable-length sequences.
- **Instance Normalization:** Normalizes per-channel per-sample; used in style transfer.
- **Group Normalization:** Compromise between BN and LN; divides channels into groups; used when batch size is small (e.g., video, segmentation).

| Norm | Normalizes Over | Batch-Dependent | Used In |
|------|:--------------:|:---------------:|---------|
| Batch Norm | Batch × Spatial | ✅ Yes | CNNs, ResNets |
| Layer Norm | Features | ❌ No | Transformers, RNNs |
| Instance Norm | Spatial (per channel) | ❌ No | Style transfer |
| Group Norm | Channel groups × Spatial | ❌ No | Video, small-batch |
**See:** [01-Foundations/03-Deep-Learning.md] (§3.3 — Normalization)

### Regularization
Techniques that prevent overfitting by constraining model complexity. Types:
- **L1 Regularization (Lasso):** Adds λ·|w| to loss; produces sparse weights (feature selection).
- **L2 Regularization (Ridge / Weight Decay):** Adds λ·‖w‖² to loss; shrinks weights toward zero without sparsity.
- **Dropout:** Randomly zeroes a fraction of activations during training; forces redundant representations.
- **Early Stopping:** Stop training when validation loss stops improving.
- **Data Augmentation:** Generate synthetic training examples via transformations.
- **Label Smoothing:** Replace hard 0/1 targets with ε/(K-1) and 1-ε; improves calibration.
- **Mixup / CutMix:** Train on interpolations of input-label pairs; improves robustness.
**See:** [01-Foundations/03-Deep-Learning.md] (§4 — Regularization)

### Loss Functions
The objective functions that quantify the error between model predictions and targets:
- **Mean Squared Error (MSE / L2):** 1/n·Σ(y - ŷ)² — for regression; penalizes large errors heavily.
- **Mean Absolute Error (MAE / L1):** 1/n·Σ|y - ŷ| — for regression; more robust to outliers.
- **Cross-Entropy Loss (Log Loss):** -Σ y·log(ŷ) — for classification; standard for multi-class.
- **Binary Cross-Entropy (BCE):** -(y·log(ŷ) + (1-y)·log(1-ŷ)) — for binary classification.
- **Hinge Loss:** max(0, 1 - y·ŷ) — used in SVMs; encourages margin maximization.
- **Contrastive Loss:** (1-y)·d² + y·max(0, margin-d)² — for siamese networks, embeddings.
- **KL Divergence:** Σ p·log(p/q) — measures distribution difference; used in VAEs, distillation.
**See:** [01-Foundations/03-Deep-Learning.md] (§3.1 — Loss Functions)

### Learning Rate Schedules
Strategies for adjusting the learning rate during training:
- **Step Decay:** Reduce LR by factor γ every N epochs.
- **Cosine Annealing:** Cosine-shaped decay from initial LR to near-zero.
- **Cosine with Warm Restarts (SGDR):** Periodic cosine resets; escapes local minima.
- **Linear Warmup + Decay:** Ramp up LR linearly, then decay.
- **Reduce on Plateau:** Reduce LR when validation metric stops improving.
- **One-Cycle Policy:** Ramp up then down in one cycle; fast convergence.
| Schedule | Convergence Speed | Final Performance | Use Case |
|----------|:-----------------:|:-----------------:|----------|
| Step Decay | Moderate | Good | Standard training |
| Cosine Annealing | Fast | Excellent | Large-scale training |
| Warmup + Cosine | Moderate | Best | Transformers, LLMs |
| One-Cycle | Very Fast | Good | Small-medium datasets |

---

## 18. Advanced Learning Paradigms

This section defines advanced machine learning paradigms not covered in the basic performance metrics (§17). These are essential concepts for understanding modern ML training and research.

### Self-Supervised Learning (SSL)

A training paradigm where the model generates its own supervisory signal from unlabeled data by solving a pretext task, learning useful representations without human annotations. Common pretext tasks include masked language modeling (predicting masked tokens — used by BERT), contrastive prediction (distinguishing similar vs dissimilar pairs — used by SimCLR, MoCo), and rotation prediction (predicting image rotation angles). Self-supervised learning is the foundation of modern pre-training: virtually all LLMs are pre-trained with SSL objectives before any supervised fine-tuning.

**Variants:**
| Variant | Domain | Method | Downstream Benefit |
|---------|:------:|--------|-------------------|
| **Masked Language Modeling (MLM)** | NLP | Predict masked tokens | BERT, RoBERTa |
| **Causal Language Modeling (CLM)** | NLP | Predict next token | GPT, LLaMA |
| **SimCLR / MoCo** | Vision | Contrastive instance discrimination | ImageNet SOTA without labels |
| **DINO / iBOT** | Vision | Self-distillation, patch-level features | ViT training |
| **MAE (Masked Autoencoder)** | Vision | Reconstruct masked patches | Efficient ViT pre-training |
| **Data2Vec** | Speech/Text/Vision | Predict latent representations | Unified multi-modal SSL |
| **wav2vec 2.0 / HuBERT** | Speech | Masked prediction in latent space | ASR with minimal labeled data |
| **SEER** | Vision | Large-scale contrastive learning | 1B+ parameter vision models |

**See:** [01-Foundations/03-Deep-Learning.md] (§2 — Neural Networks); [02-LLMs/01-Transformer-Architecture.md] (§3 — Pre-training Objectives)

### Contrastive Learning

A self-supervised technique that learns representations by pulling similar (positive) pairs together in embedding space while pushing dissimilar (negative) pairs apart. The core idea is: *augmented views of the same sample should have similar embeddings, while different samples should have different embeddings.* Contrastive learning relies on data augmentation (crops, color jitter, noise) to create positive pairs and treats other samples in the batch as negatives. The standard loss is **NT-Xent** (Normalized Temperature-scaled Cross-Entropy Loss), also called InfoNCE.

```python
import torch
import torch.nn.functional as F

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5):
    """
    Compute NT-Xent (Normalized Temperature Cross-Entropy) loss.
    z1, z2: [batch_size, dim] — two augmented views of the same batch.
    """
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)                      # [2B, D]
    z = F.normalize(z, dim=-1)                            # L2 normalize
    sim = z @ z.T                                         # [2B, 2B]

    # Mask out self-similarity (diagonal)
    mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim = sim[mask].view(2 * batch_size, -1)             # [2B, 2B-1]

    # Positive pairs: (i, i+B) and (i+B, i)
    pos = torch.cat([sim[:batch_size, batch_size-1:], 
                     sim[batch_size:, :batch_size]], dim=0)
    # Negative pairs: everything else
    neg = torch.cat([sim[:batch_size, :batch_size-1],
                     sim[batch_size:, batch_size-1:]], dim=0)

    logits = torch.cat([pos, neg], dim=1) / temperature
    labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z.device)
    return F.cross_entropy(logits, labels)
```

**Applications:** Vision (SimCLR, MoCo, CLIP), NLP (SimCSE, sentence embeddings), multimodal (CLIP, ImageBind).

### Active Learning

A machine learning approach where the algorithm selectively chooses which data points to label next, minimizing the amount of labeled data needed to achieve a target accuracy. The active learning loop iterates: train on currently labeled data → query the most uncertain or informative unlabeled samples → obtain labels (from human annotator) → retrain.

| Query Strategy | How It Selects Samples | Best For |
|----------------|----------------------|----------|
| **Uncertainty sampling** | Highest prediction entropy or lowest max probability | Classification |
| **Margin sampling** | Smallest difference between top-2 class probabilities | Multi-class |
| **Query-by-committee (QBC)** | Highest disagreement across an ensemble of models | Small datasets |
| **Expected Model Change** | Samples that would most change the current model | Any task |
| **Diversity sampling** | Representative samples covering the input space | High-dimensional data |
| **Bayesian Active Learning** | Acquisition functions (BALD, max entropy) from Bayesian NN | Uncertainty-aware |

**See:** [01-Foundations/02-Machine-Learning.md] (§5 — Semi-supervised Learning)

### Synthetic Data

Artificially generated data used to supplement or replace real training data, addressing scarcity, privacy, or imbalance issues. Synthetic data can be generated via generative models (GANs, diffusion models, LLMs), rule-based systems (simulators, physics engines), or augmentation pipelines. Quality considerations include distributional alignment with real data, diversity, and the risk of model collapse from training on synthetic outputs.

| Generation Method | Domain | Tools / Models | Quality | Risk |
|-------------------|:------:|:--------------:|:------:|:----:|
| **LLM-generated text** | NLP | GPT-4, Claude, LLaMA | High | Model collapse, bias amplification |
| **Diffusion-based images** | Vision | Stable Diffusion, DALL-E 3 | High | Stereotype reinforcement |
| **GAN-based synthesis** | Vision, tabular | StyleGAN, CTGAN | Medium | Mode collapse |
| **Rule-based simulation** | Robotics, autonomous driving | MuJoCo, CARLA, Unity | Very High | Reality gap |
| **Data augmentation** | All domains | RandAugment, MixUp, CutMix | Medium | Limited diversity |
| **Formal generation** | Code, math, logic | Symbolic engines, Lean | Very High | Limited scope |

**See:** [01-Foundations/05-Training-Methodologies.md] (§3 — Data Preparation); [06-Advanced/02-Diffusion-Models.md]

### Reinforcement Learning (RL) Fundamentals

A paradigm where an agent learns to make sequential decisions by interacting with an environment, receiving rewards (or penalties) for its actions, and maximizing cumulative reward over time. RL differs from supervised learning: there is no labeled data — the agent must explore and learn from its own experience.

| Component | Description | Example |
|-----------|-------------|---------|
| **Agent** | The decision-maker | LLM generating tokens |
| **Environment** | The world the agent interacts with | Game, dialog context, user feedback |
| **State (s)** | Current situation | Current board position, conversation history |
| **Action (a)** | What the agent can do | Move a piece, generate next token |
| **Reward (r)** | Feedback signal | +1 for correct answer, -1 for wrong |
| **Policy (π)** | Strategy: state → action | Model's token probability distribution |
| **Value function (V)** | Expected future reward from a state | How good is this position? |
| **Q-function** | Expected future reward for state-action pair | How good is this action in this state? |

| RL Algorithm | Type | Key Idea | Used In |
|-------|:----:|----------|---------|
| **Q-Learning** | Value-based | Learn optimal Q-function via Bellman equation | Classic control |
| **Deep Q-Network (DQN)** | Value-based | Q-Learning with neural network approximation | Atari, games |
| **Policy Gradient (REINFORCE)** | Policy-based | Directly optimize policy via gradient ascent | Continuous control |
| **PPO (Proximal Policy Optimization)** | Policy-based | Clipped surrogate objective for stable updates | RLHF, robotics |
| **SAC (Soft Actor-Critic)** | Actor-critic | Max entropy RL for exploration | Robotics |
| **PPO for RLHF** | Policy-based | KL-constrained PPO with reward model | LLM alignment |

**RLHF (Reinforcement Learning from Human Feedback):** Applies PPO to align LLMs with human preferences. A reward model trained on human comparisons scores model outputs, and the LLM is fine-tuned via PPO to maximize this score while staying close to its original distribution (via KL penalty).

### Causal Machine Learning

An emerging paradigm that moves beyond correlation-based prediction to model causal relationships — answering *what would happen if* questions (interventions) rather than *what predicts what* questions (associations). Causal ML methods estimate treatment effects, perform counterfactual reasoning, and build models robust to distribution shift.

| Method | What It Estimates | When to Use |
|--------|------------------|-------------|
| **Causal Inference (DoWhy, CausalNex)** | Average Treatment Effect (ATE), Conditional ATE | Estimating impact of interventions |
| **Uplift Modeling** | Individual Treatment Effect (ITE) | Targeting which users to treat |
| **Double/Debiased ML (DML)** | ATE using ML + orthogonalization | High-dimensional confounders |
| **Instrumental Variables (IV)** | Causal effect with unobserved confounding | Natural experiments |
| **Directed Acyclic Graphs (DAGs)** | Causal structure identification | Hypothesis generation |
| **Counterfactual Reasoning** | What would have happened under alternative action | Explainability, fairness |

**Key distinction:** Standard ML predicts `P(Y | X)`; Causal ML estimates `P(Y | do(X=x))` — the distribution of Y if X were *intervened* to take value x, which is different from merely conditioning on X=x when confounders exist.

**See:** [01-Foundations/10-Causal-Inference.md] (dedicated document on causal methods)

---

## 19. Data Engineering & MLOps Pipeline Concepts

This section defines core data engineering and MLOps concepts referenced across the AI Base Knowledge library. These terms underpin the infrastructure and workflows used to build, deploy, and maintain production ML systems.

### ETL (Extract, Transform, Load)

A data pipeline pattern where data is extracted from source systems, transformed (cleaned, normalized, enriched), and loaded into a target storage or analytics system. ETL is the traditional approach for batch data processing and has been largely supplemented by ELT in modern data architectures, where raw data is first loaded into a data lake or warehouse and transformed in place.

### ELT (Extract, Load, Transform)

A modern variation of ETL where data is first extracted and loaded into the target system in raw form, then transformed using the compute power of the target platform (e.g., Snowflake, BigQuery, Databricks). ELT leverages modern cloud data warehouses that can handle massive-scale transformations efficiently, reducing the need for a separate transformation layer.

| Aspect | ETL | ELT |
|--------|:---:|:---:|
| Transformation timing | Before loading | After loading |
| Target system | Data warehouse | Data lake or lakehouse |
| Compute location | Separate transformation server | Target platform (warehouse/lake) |
| Schema | Schema-on-write | Schema-on-read |
| Best for | Legacy systems, strict governance | Cloud-native, agile analytics |
| Latency | Higher (batch-oriented) | Lower when using streaming |

### Data Pipeline

An automated sequence of data processing steps that moves data from source to destination, applying transformations, validations, and quality checks along the way. Data pipelines are the backbone of ML systems — covering data ingestion, feature computation, model training, inference, and monitoring. Pipeline orchestration tools (Apache Airflow, Prefect, Dagster, KubeFlow) manage scheduling, dependencies, retries, and alerting.

### Feature Store

A centralized repository for storing, managing, and serving ML features consistently across training and inference. Feature stores solve the "training-serving skew" problem by ensuring the same feature computation logic is used in both phases. Key capabilities include feature versioning, point-in-time correct lookups (for time-series features), online serving (low-latency via Redis/DynamoDB), offline serving (batch via Parquet/Delta Lake), and feature discovery. Popular implementations: Feast, Tecton, SageMaker Feature Store.

### Data Lake / Data Warehouse / Lakehouse

| Concept | Description | Best For | Example Technologies |
|---------|-------------|----------|---------------------|
| **Data Warehouse** | Structured, schema-on-write storage optimized for SQL analytics | Business reporting, BI dashboards | Snowflake, BigQuery, Redshift |
| **Data Lake** | Raw data storage (structured, semi-structured, unstructured) in native format | Data science exploration, ML training data, archival | S3, ADLS, GCS |
| **Lakehouse** | Combines data lake flexibility with warehouse ACID/performance via a metadata layer | Unified ML + BI workloads | Delta Lake, Iceberg, Hudi |

### Data Lineage

The practice of tracking data as it flows through pipelines — from source through transformations to final consumption. Lineage enables impact analysis (what downstream models are affected by a schema change?), root-cause debugging (where did this anomalous value originate?), and compliance auditing (is this data processed according to GDPR?). Tools: Apache Atlas, DataHub, Amundsen, Marquez, dbt.

### Data Drift and Concept Drift

- **Data drift (covariate shift):** The distribution of input features changes over time (e.g., user demographics shift). The model receives inputs it was not trained on, degrading prediction quality.
- **Concept drift:** The relationship between inputs and the target changes (e.g., customer preferences shift post-pandemic). The mapping P(Y|X) evolves even if X's distribution stays constant.
- **Label drift:** The distribution of the target variable changes (e.g., fraud rate increases seasonally).
- **Detection methods:** PSI (Population Stability Index), KS-test, Wasserstein distance, ADWIN, DDM (Drift Detection Method).

**See:** [01-Foundations/02-Machine-Learning.md] (§6 — Model Monitoring); [05-Enterprise/01-Enterprise-AI-Deployment.md] (§5 — Monitoring)

### Data Leakage

A subtle but critical error where information from the future (or test set) is inadvertently used during training, leading to overly optimistic evaluation metrics that do not reflect real-world performance. Common forms: (1) temporal leakage — training on future data in time-series tasks, (2) feature leakage — using features that are not available at prediction time, (3) duplication leakage — identical or near-identical samples appear in both train and test sets. Data leakage is one of the most common causes of ML project failure in production.

**See:** [01-Foundations/05-Training-Methodologies.md] (§4.1 — Validation Strategies)

### Train / Validation / Test Split

The standard paradigm for evaluating ML models: training data (used to fit parameters), validation data (used for hyperparameter tuning and model selection), and test data (held out for final evaluation only). Common ratios: 80/10/10 or 70/15/15. For time series, the split must respect temporal order (see [06-Advanced/07-Time-Series-Forecasting.md] §8). For small datasets, k-fold cross-validation replaces fixed splits.

### MLOps (Machine Learning Operations)

The practice of applying DevOps principles to ML systems: versioning data and models, automating training and deployment pipelines, monitoring model performance in production, and managing the full ML lifecycle. MLOps encompasses CI/CD for ML (training pipeline automation), model registry (versioned artifact storage), feature store (reusable feature computation), monitoring (drift detection, performance tracking), and governance (model approvals, audit trails). Key frameworks: MLflow, Kubeflow, TFX, SageMaker, Vertex AI.

### Model Registry

A central catalog of trained model versions with metadata: training parameters, evaluation metrics, training data version, and deployment status (staging, production, archived). The model registry is the source of truth for which model is serving in production and enables rollback, reproducibility, and compliance auditing. MLflow Model Registry is the most widely adopted open-source implementation.

### Shadow Deployment

A deployment strategy where a new model version runs in parallel with the current production model, receiving the same requests but its predictions are logged (not served to users). Shadow deployment allows safe evaluation of the new model's performance, latency, and behavior under real traffic before promoting it to production. It is the safest deployment pattern for high-stakes ML applications.

**See:** [05-Enterprise/01-Enterprise-AI-Deployment.md] (§4 — Deployment Strategies)

### A/B Testing for ML

A controlled experiment where two model versions (control A and treatment B) serve different user segments, and their performance is compared on business metrics (conversion rate, user satisfaction, revenue). Proper A/B testing requires: random assignment, sufficient sample size (power analysis), statistical significance testing, and guarding against interference between arms. A/B testing is the gold standard for evaluating model impact but requires careful infrastructure and traffic allocation.

### CI/CD for ML (ML Pipelines)

Applying continuous integration and continuous deployment practices to machine learning: (1) CI — automatically validate code, data schemas, and model quality on every commit; (2) CD — automatically deploy validated models to staging/production. ML CI/CD differs from software CI/CD in that it must validate both code and data artifacts, and model behavior can change without code changes (data drift). Tools: GitHub Actions + MLflow, Kubeflow Pipelines, Argo Workflows, Jenkins + DVC.

**See:** [05-Enterprise/01-Enterprise-AI-Deployment.md] (§4 — Deployment); [05-Enterprise/04-AI-Infrastructure.md] (§5 — MLOps)

---

## Index of Terms by Document

| Document | Key Terms Covered |
|----------|------------------|
| **01-Foundations/01-LLM-and-AI-Models.md** | LLM, Foundation Model, Base Model, Instruct Model, Transformer, Attention, Token, Embedding, Context Window, KV Cache, MoE, Scaling Laws, Emergent Abilities, Grokking, Alignment Tax, Capability vs Alignment, Inference-Time Compute, Self-Consistency, Quantization, GGUF, GPTQ, AWQ, Fine-tuning, SFT, RLHF, DPO, Distillation, Hallucination, Curriculum Learning, RLAIF, Constitutional AI, KTO, ORPO, SimPO, Preference Optimization, Instruction Tuning, Prefix Caching, Continuous Batching, Prompt Caching, Prompt Engineering, System Prompt, Function Calling, Chain-of-Thought, Speculative Decoding, Perplexity, BLEU, ROUGE, MMLU, HumanEval, GPQA, Arena Elo, MATH, GSM8K, IFEval, Tensor Parallelism, Pipeline Parallelism, Model Router |
| **04-RAG/01-RAG-Architectures.md** | RAG, Chunking, Embedding Model, Vector Database, Semantic Search, Hybrid Search, Re-ranking, Contextual Retrieval, Query Rewriting, Fusion Retrieval, CRAG, Self-RAG, Agentic RAG |
| **03-Agents/04-Protocols-MCP-ACP.md** | MCP, ACP, MCP Host, MCP Client, MCP Server, Tool (MCP), Resource (MCP), Prompt (MCP), ACP Transport, Function Calling (comparison) |
| **03-Agents/01-Agent-Architectures.md** | AI Agent, AI Orchestrator, Agent Loop, ReAct, Chain-of-Thought, Tree-of-Thought, Task Decomposition, Subagent, Tool-use, Autonomous Agent, Orchestration Pattern (Sequential/Parallel/Hierarchical/Recursive/Hybrid), Human-in-the-Loop, Multi-Agent System, LangChain, CrewAI, AutoGen, Episodic Memory, Semantic Memory, Reflection, Mixture of Agents, Agent Supervisor, Delegation, Task Queue, Guardrails, Safety Filters, Agentic AI, World Models, System 2 Thinking |
| **03-Agents/05-Tool-Implementations.md** | OpenCode, Crush, Claude Code, Hermes Agent, SSP (SSH-SCP Transport), Semantic Memory, Self-Improvement |
| **08-Reference/03-Agent-Configs-SOUL-SKILL.md** | SOUL.md, SKILL.md, AGENTS.md, CLAUDE.md, .cursorrules, Agent Identity, Procedural Memory |
| **08-Reference/01-Glossary.md (this document)** | vLLM, CUDA, NCCL, HNSW, ANN, MLflow, W&B, Langfuse, RAGAS, Docker for AI, DeepSpeed, TensorRT-LLM, ETL, ELT, Data Pipeline, Feature Store, Data Lake, Data Warehouse, Lakehouse, Data Lineage, Data Drift, Concept Drift, Data Leakage, Train/Val/Test Split, MLOps, Model Registry, Shadow Deployment, A/B Testing, CI/CD for ML |

---

*Document version: 1.5 → 2.0 — June 2026. This glossary is part of the AiBaseKnowledge series. It is cross-referenced with all other documents in the library and should be updated whenever new terms are introduced in any of the source documents. [Updated: added §17 Performance Metrics & Training Concepts — accuracy, precision, recall, F1, ROC-AUC, confusion matrix, backpropagation, gradient descent, normalization, regularization, loss functions, learning rate schedules; added §18 Advanced Learning Paradigms — self-supervised learning, contrastive learning, active learning, synthetic data, reinforcement learning fundamentals, causal ML; added §19 Data Engineering & MLOps Pipeline Concepts — ETL, ELT, Data Pipeline, Feature Store, Data Lake/Warehouse/Lakehouse, Data Lineage, Data Drift, Data Leakage, MLOps, Model Registry, Shadow Deployment, A/B Testing, CI/CD for ML; fixed stale cross-reference paths in Index table]*
