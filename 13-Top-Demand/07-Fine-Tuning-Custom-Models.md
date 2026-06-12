# Fine-Tuning & Custom Models

> **Last Updated:** June 2026  
> **Category:** Top Demand — Current Market Snapshot  
> **Cross-References:** 02-AI-Agent-Development.md, 05-AI-Safety-Alignment.md, 06-RAG-Retrieval-Systems.md, 08-Edge-AI-Inference.md

---

## Table of Contents

1. [Market Context & Demand](#1-market-context--demand)
2. [When to Fine-Tune vs. RAG vs. Prompt Engineering](#2-when-to-fine-tune-vs-rag-vs-prompt-engineering)
3. [PEFT Methods (Parameter-Efficient Fine-Tuning)](#3-peft-methods-parameter-efficient-fine-tuning)
   - 3.1 LoRA (Low-Rank Adaptation)
   - 3.2 QLoRA (Quantized LoRA)
   - 3.3 DoRA (Weight-Decomposed Low-Rank Adaptation)
   - 3.4 AdaLoRA & Dynamic Rank Allocation
   - 3.5 Prompt Tuning & Prefix Tuning
4. [Full Fine-Tuning](#4-full-fine-tuning)
   - 4.1 When Full Fine-Tuning is Necessary
   - 4.2 Infrastructure Requirements
   - 4.3 FSDP & Distributed Training
   - 4.4 Training Recipes
5. [RLHF & DPO Fine-Tuning](#5-rlhf--dpo-fine-tuning)
   - 5.1 Preference Data Collection
   - 5.2 Reward Model Training
   - 5.3 PPO Fine-Tuning
   - 5.4 DPO Fine-Tuning Pipeline
6. [Model Distillation](#6-model-distillation)
   - 6.1 Logit-Based Distillation
   - 6.2 Feature-Based Distillation
   - 6.3 On-Policy Distillation
   - 6.4 Task-Specific Distillation
7. [Dataset Curation](#7-dataset-curation)
   - 7.1 Data Quality Filtering
   - 7.2 Synthetic Data Generation
   - 7.3 Data Mixing Strategies
   - 7.4 Decontamination
8. [Evaluation Pipelines](#8-evaluation-pipelines)
9. [Deployment Infrastructure](#9-deployment-infrastructure)
   - 9.1 vLLM
   - 9.2 Text Generation Inference (TGI)
   - 9.3 Ollama
   - 9.4 llama.cpp
   - 9.5 Hypersonic Serving
10. [Cost Analysis](#10-cost-analysis)
11. [Future Directions](#11-future-directions)

---

## 1. Market Context & Demand

Fine-tuning has become commoditized. In June 2026, the question is no longer "should we fine-tune?" but "which method, with what data, on which base model?"

**Market dynamics:**
- 60%+ of enterprises with AI deployments fine-tune at least one model
- Open-source base models (Llama 4, Qwen 2.5, Mistral 3) dominate fine-tuning starts
- Fine-tuning-as-a-service providers (Fireworks, Together, Anyscale, Modal) grew 180% YoY
- Average fine-tuning project cost: $5K-$200K (PEFT vs. full fine-tuning)
- GPT-5 fine-tuning API (OpenAI): $100/training-hour, available since early 2026

**Key drivers:**
- **Domain specificity** — General models underperform on specialized tasks (legal, medical, finance)
- **Latency/cost optimization** — Smaller fine-tuned models beat larger general ones on specific tasks
- **Privacy** — On-premise fine-tuning keeps sensitive data in-house
- **Brand voice** — Companies want models that speak in their specific style

---

## 2. When to Fine-Tune vs. RAG vs. Prompt Engineering

| Approach | Best For | Cost | Effort | Update Frequency | Performance |
|----------|----------|------|--------|-----------------|-------------|
| **Prompt Engineering** | Simple instructions, quick experiments | $0 | Low | Continuous | Baseline |
| **RAG** | Knowledge-heavy tasks, dynamic data | Low-Med | Medium | Real-time | High (with good retrieval) |
| **Fine-Tuning (PEFT)** | Style/tone, domain adaptation, structured outputs | Med | Med-High | Quarterly | Very High |
| **Full Fine-Tuning** | Whole-task learning, new capabilities | High | Very High | Rarely | Maximum |

**Decision framework:**
```
Is the task knowledge-based? → Yes → RAG
Is the task format/structure-based? → Yes → Fine-tuning (PEFT)
Is the task a new capability? → Yes → Full fine-tuning or RLHF
Do you need both knowledge + format? → RAG + Fine-tuning
```

---

## 3. PEFT Methods (Parameter-Efficient Fine-Tuning)

### 3.1 LoRA (Low-Rank Adaptation)

LoRA freezes the base model weights and injects trainable rank-decomposition matrices into attention layers.

**How it works:**
```
Original: W ∈ ℝ^(d×k) → Frozen
LoRA: W' = W + BA where B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), r << min(d,k)
```

**Implementation:**

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,                # Rank — controls expressiveness
    lora_alpha=32,       # Scaling factor (alpha/r)
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, config)
# Train only 0.1-1% of total parameters
```

**Optimal rank selection (2026 guidance):**

| Task | Recommended Rank | Notes |
|------|-----------------|-------|
| Style/format adaptation | r=8-16 | Low rank sufficient |
| Domain adaptation (narrow) | r=16-32 | More expressiveness needed |
| Domain adaptation (broad) | r=32-64 | Full domain knowledge |
| Multitask fine-tuning | r=64-128 | Must capture multiple tasks |

### 3.2 QLoRA (Quantized LoRA)

QLoRA combines 4-bit base model quantization with LoRA adapters. The base model is quantized using NF4 (NormalFloat4), while adapters remain in full precision.

**Memory savings:**

| Model | Full Fine-Tune | LoRA (16-bit) | QLoRA (4-bit + LoRA) |
|-------|---------------|---------------|----------------------|
| Llama 4 Scout (17B) | ~140 GB | ~70 GB | ~16 GB |
| Qwen 2.5 32B | ~256 GB | ~128 GB | ~28 GB |
| Llama 4 Maverick (400B MoE) | ~3.2 TB | ~1.6 TB | ~320 GB |

**Training setup:**

```python
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-4-17B",
    quantization_config=bnb_config,
    device_map="auto"
)

# LoRA on quantized model
lora_config = LoraConfig(r=16, ...)
model = get_peft_model(model, lora_config)
```

**Performance impact (2026):**
- QLoRA typically achieves 95-99% of full-precision LoRA quality
- Depends on: model size, task complexity, training data quality
- Recommended for: single-GPU fine-tuning, rapid experimentation

### 3.3 DoRA (Weight-Decomposed Low-Rank Adaptation)

DoRA (2024) decomposes pre-trained weights into magnitude and direction components, then applies LoRA only to the directional component.

**Advantage over LoRA:**
- Better captures the learning dynamics of full fine-tuning
- 0.5-2% improvement on standard benchmarks vs. LoRA at same rank
- Same memory footprint as LoRA
- Slightly more computation per step (negligible)

```python
# DoRA configuration (HuggingFace PEFT 0.15+)
config = LoraConfig(
    r=16,
    use_dora=True,  # Enable DoRA
    # ... same as LoRA
)
```

### 3.4 AdaLoRA & Dynamic Rank Allocation

AdaLoRA learns to allocate rank across layers dynamically based on importance:

- High-importance layers get higher rank
- Low-importance layers get lower rank or are pruned
- 20-30% more parameter-efficient than fixed-rank LoRA
- Automatically identifies which layers need more capacity

### 3.5 Prompt Tuning & Prefix Tuning

Lightweight alternatives to LoRA:

**Prompt tuning** — Learn soft prompt tokens prepended to input:

```python
# Prompt tuning: learn 20 special tokens
config = PromptTuningConfig(
    num_virtual_tokens=20,
    prompt_tuning_init="TEXT",
    prompt_tuning_init_text="Provide a detailed analysis of the following:",
    tokenizer_name_or_path="meta-llama/Llama-4-17B",
    task_type="CAUSAL_LM"
)
```

**Prefix tuning** — Learn prefix vectors inserted in each transformer layer.

**When to use:**
- Extremely low resource (tune <0.01% of parameters)
- Rapid iteration across many tasks
- Multi-task setups with shared base model

---

## 4. Full Fine-Tuning

### 4.1 When Full Fine-Tuning is Necessary

Full fine-tuning updates all model parameters. It's necessary when:

1. **New capabilities** — Teaching the model to do something the base model cannot
2. **Major domain shift** — Moving from general to highly specialized domain
3. **Multi-task instruction tuning** — Building an instruction-following base
4. **Maximum performance needed** — PEFT methods leave 1-5% performance gap

### 4.2 Infrastructure Requirements

| Model Size | Min GPUs | Recommended GPUs | Memory (per GPU) | Time (1 epoch, 100K samples) |
|-----------|----------|-----------------|-------------------|------------------------------|
| 7-8B | 4 | 8-16 | 80 GB H100 | 4-8 hours |
| 13-17B | 8 | 16-32 | 80 GB H100 | 8-16 hours |
| 34-40B | 16 | 32-64 | 80 GB H100 | 16-32 hours |
| 70-72B | 32 | 64-128 | 80 GB H100 | 1-2 days |
| 400B MoE | 128 | 256+ | 80 GB H100 | 3-7 days |

### 4.3 FSDP & Distributed Training

**FSDP (Fully Sharded Data Parallel)** is the standard for full fine-tuning:

```python
# Launch with torchrun
# torchrun --nproc_per_node=8 train.py

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision
)

# FSDP config for training
fsdp_config = {
    "sharding_strategy": ShardingStrategy.FULL_SHARD,  # ZeRO-3
    "mixed_precision": MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16
    ),
    "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
    "limit_all_gathers": True,
    "use_orig_params": True
}

# Wrap model
model = FSDP(model, **fsdp_config)
```

### 4.4 Training Recipes

**Standard full fine-tuning recipe (Llama 4, June 2026):**

```yaml
training:
  base_model: meta-llama/Llama-4-17B
  precision: bfloat16
  optimizer: AdamW (β1=0.9, β2=0.95, eps=1e-8)
  learning_rate: 2e-5 with cosine decay
  warmup_steps: 200
  batch_size: 128 (microbatch: 4, gradient_accumulation: 32)
  max_seq_length: 8192
  scheduler: cosine (decay to 10% of peak LR)
  weight_decay: 0.1
  gradient_clipping: 1.0
  num_epochs: 2-3
  
  data_mix:
    - instruction_data: 60% (domain-specific instructions)
    - general_data: 20% (general QA to prevent catastrophic forgetting)
    - code_data: 10% (if coding is needed)
    - safety_data: 10% (alignment preservation)
```

---

## 5. RLHF & DPO Fine-Tuning

### 5.1 Preference Data Collection

Building preference data is the most critical and expensive step:

```yaml
preference_data:
  collection_method: human_annotators + AI_feedback
  
  format:
    prompt: str
    chosen: str  # Preferred response
    rejected: str  # Dispreferred response
    metadata:
      - domain
      - difficulty
      - annotator_id
      - confidence
  
  quality_controls:
    - Annotator agreement (Cohen's κ > 0.6)
    - Regular calibration checks
    - Gold standard examples (hidden)
    - Inter-annotator consistency scoring
  
  scale:
    minimum: 10,000 pairs (DPO), 50,000 pairs (RLHF)
    recommended: 50,000-500,000 pairs
    cost: $2-$10 per pair (expert annotation)
```

### 5.2 Reward Model Training

```python
# Bradley-Terry reward model
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.value_head = nn.Linear(base_model.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state[:, -1, :]  # Last token
        reward = self.value_head(hidden).squeeze(-1)
        return reward

# Bradley-Terry loss
def bradley_terry_loss(chosen_rewards, rejected_rewards):
    return -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
```

### 5.3 PPO Fine-Tuning

PPO details (see also 05-AI-Safety-Alignment.md):

```yaml
ppo_config:
  kl_penalty: adaptive (target KL = 0.04)
  clip_range: 0.2
  clip_range_value: 5.0
  vf_coef: 0.5
  horizon: 10000
  gamma: 1.0  # No discount for PPO in RLHF
  lam: 0.95
  mini_batch_size: 64
  ppo_epochs: 4
  learning_rate: 3e-6 (1/10th of SFT learning rate)
```

### 5.4 DPO Fine-Tuning Pipeline

```python
from trl import DPOTrainer

dpo_trainer = DPOTrainer(
    model=model,  # Policy model
    ref_model=ref_model,  # Reference model (frozen)
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=5e-6,
        fp16=True,
    ),
    beta=0.1,  # DPO temperature
    max_length=2048,
    max_prompt_length=1024,
)

dpo_trainer.train()
```

---

## 6. Model Distillation

### 6.1 Logit-Based Distillation

Classic distillation uses the teacher model's output logits as soft targets:

```python
def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
    # Soft target loss (KL divergence)
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean"
    ) * (temperature ** 2)
    
    # Hard target loss (cross-entropy with ground truth)
    hard_loss = F.cross_entropy(student_logits, labels)
    
    # Combined
    return alpha * soft_loss + (1 - alpha) * hard_loss
```

### 6.2 Feature-Based Distillation

Intermediate layer representations are matched:

```
Teacher Layers:  L1 → L2 → ... → Ln
                   ↓     ↓          ↓
Student Layers:  L1 → L2 → ... → Ln (smaller)
                   ↕     ↕          ↕
              MSE Loss or Cosine Loss
```

### 6.3 On-Policy Distillation

Student generates outputs, teacher scores them:

```
Student generates response → Teacher evaluates quality → Student learns from feedback
```

This is the approach used for distilling reasoning chains (e.g., GPT-5 → smaller model).

### 6.4 Task-Specific Distillation

Distill only for specific tasks:

| Task | Teacher → Student | Size Reduction | Quality Retention |
|------|------------------|----------------|-------------------|
| Code generation | GPT-5 → CodeLlama 13B | 20x | 92% |
| Summarization | Claude 4 → Mistral 7B | 40x | 88% |
| Classification | GPT-5 → DeBERTa-v3 | 200x | 95% |
| Chat | Llama 4 400B → 17B | 24x | 85% |

---

## 7. Dataset Curation

### 7.1 Data Quality Filtering

**Quality filters (2026 best practices):**

```yaml
quality_filters:
  heuristic:
    - min_length: 50 characters
    - max_repetition_ratio: 0.3
    - min_language_probability: 0.9 (from language detector)
    - remove_boilerplate: true
    - remove_html_markup: true
  
  model_based:
    - perplexity_filter: remove examples with perplexity > 2σ from mean
    - instruction_quality: use 7B judge to score Q/A pairs
    - deduplication: MinHash near-deduplication (95% threshold)
    - contamination: check against evaluation benchmarks (n-gram overlap)
```

### 7.2 Synthetic Data Generation

Generating synthetic training data from larger models:

```python
def generate_synthetic_data(teacher_model, topics, n_per_topic=1000):
    data = []
    for topic in topics:
        prompt = f"""Generate a challenging instruction and high-quality response about {topic}.
The instruction should test {topic} knowledge at an expert level.
The response should be accurate, comprehensive, and well-structured.

Format:
Instruction: [instruction]
Response: [response]"""
        
        for _ in range(n_per_topic // len(topics)):
            result = teacher_model.generate(prompt)
            instruction, response = parse(result)
            data.append({"instruction": instruction, "response": response})
    
    return data
```

**Quality considerations:**
- Synthetic data alone leads to model collapse (Shumailov et al., 2024)
- Mix synthetic with human data (70:30 to 90:10 ratio)
- Use multiple teacher models for diversity
- Filter synthetic data with quality classifiers

### 7.3 Data Mixing Strategies

**Optimal data proportions (empirical, 2026):**

| Data Type | Proportion | Purpose |
|-----------|-----------|---------|
| Domain instruction data | 40-60% | Main task learning |
| General instruction data | 20-30% | Prevents catastrophic forgetting |
| Code data | 5-15% | Improves reasoning |
| Preference data | 5-10% | Alignment |
| Safety data | 2-5% | Safety preservation |
| High-quality pretrain data | 1-3% | Knowledge reinforcement |

### 7.4 Decontamination

Critical: ensure no benchmark data leaks into training:

```python
def decontaminate(dataset, benchmark_sets, ngram_size=13):
    """Remove examples that overlap with benchmark data."""
    clean_data = []
    
    for example in dataset:
        text = example["instruction"] + " " + example.get("response", "")
        text_ngrams = set(zip(*[text[i:] for i in range(ngram_size)]))
        
        contaminated = False
        for benchmark_text in benchmark_sets:
            bench_ngrams = set(zip(*[benchmark_text[i:] 
                                     for i in range(ngram_size)]))
            overlap = len(text_ngrams & bench_ngrams)
            if overlap / len(text_ngrams) > 0.8:
                contaminated = True
                break
        
        if not contaminated:
            clean_data.append(example)
    
    return clean_data
```

---

## 8. Evaluation Pipelines

### Multi-Dimensional Evaluation

```yaml
evaluation:
  task_performance:
    - accuracy / F1 / EM (task-specific metrics)
    - expert human evaluation (n=100+ per class)
    - A/B testing against base model
    
  generalization:
    - held-out tasks (not seen in training)
    - OOD examples (different distribution)
    - cross-language (if multilingual)
    
  safety:
    - MMLU Safety
    - BBQ (bias)
    - TruthfulQA
    - Internal red-teaming
    - Jailbreak resistance
  
  efficiency:
    - latency (P50, P95, P99)
    - throughput (tokens/sec per GPU)
    - memory usage (peak, steady-state)
    - cost per inference
    
  robustness:
    - paraphrased inputs
    - noisy inputs (typos, formatting changes)
    - adversarial inputs
    - length generalization
```

---

## 9. Deployment Infrastructure

### 9.1 vLLM

vLLM is the most popular open-source LLM serving engine (65% market share):

```python
from vllm import LLM, SamplingParams

# Load fine-tuned model
llm = LLM(
    model="./fine-tuned-llama-4",
    tensor_parallel_size=4,  # 4 GPUs
    quantization="fp8",      # FP8 inference
    max_model_len=16384,
    gpu_memory_utilization=0.90
)

# Optimized serving
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=2048,
    frequency_penalty=0.1
)

outputs = llm.generate(prompts, sampling_params)
```

**vLLM features (2026):**
- PagedAttention for efficient KV cache management
- Continuous batching
- FP8 quantization support
- Speculative decoding (2-3x speedup)
- Prefix caching
- LoRA adapter swapping (no reloading base model)

### 9.2 Text Generation Inference (TGI)

HuggingFace's TGI is also widely used:

```yaml
tgi_config:
  model: ./fine-tuned-model
  num_shard: 4
  max_concurrent_requests: 512
  max_batch_prefill_tokens: 8192
  max_batch_total_tokens: 2097152
  
  quantization: bitsandbytes-nf4
  speculate: 5  # Speculative decoding with 5 draft tokens
```

### 9.3 Ollama

For local development and testing:

```bash
# Create Modelfile
FROM ./fine-tuned-llama-4
PARAMETER temperature 0.7
PARAMETER top_p 0.9

# Build and run
ollama create my-model -f Modelfile
ollama run my-model
```

### 9.4 llama.cpp

For CPU and edge deployment:

```bash
# Convert to GGUF format
python convert.py ./fine-tuned-model --outfile model.gguf

# Quantize
./quantize model.gguf model-q4_K_M.gguf q4_K_M

# Run
./main -m model-q4_K_M.gguf -p "Your prompt here" -n 2048
```

### 9.5 Hypersonic Serving

Emerging serving technologies (2026):

| Technology | Speedup over vLLM | Technique | Maturity |
|-----------|------------------|-----------|----------|
| **SGLang** | 2-5x | RadixAttention, structured generation | Production |
| **TensorRT-LLM** | 1.5-3x | NVIDIA-optimized kernels | Production |
| **Mamba** | 3-5x | State-space model architecture | Early adopter |
| **CausalLM** | 2-3x | Custom CUDA kernels | Emrging |

---

## 10. Cost Analysis

### Fine-Tuning Costs

| Method | 7B Model | 17B Model | 70B Model | 400B MoE |
|--------|----------|-----------|-----------|----------|
| **LoRA/QLoRA** | $50-200 | $100-500 | $500-2000 | $2000-8000 |
| **Full SFT** | $500-2000 | $2000-8000 | $8000-30000 | $30000-100000 |
| **RLHF** | $2000-5000 | $5000-20000 | $20000-80000 | $80000-250000 |
| **DPO** | $500-2000 | $2000-5000 | $5000-20000 | $20000-60000 |

*Costs: GPU compute only (runpod/azure/aws spot pricing, June 2026)*

### Inference Costs

| Model Size | vLLM (8-bit) | vLLM (FP8) | TensorRT (FP8) | Spec Decode |
|-----------|--------------|------------|----------------|-------------|
| 7B | $0.08/M tokens | $0.07/M | $0.05/M | $0.04/M |
| 17B | $0.20/M tokens | $0.17/M | $0.12/M | $0.09/M |
| 70B | $0.80/M tokens | $0.60/M | $0.45/M | $0.35/M |
| 400B MoE | $3.00/M tokens | $2.50/M | $2.00/M | $1.50/M |

### ROI Calculation

```yaml
roi_framework:
  costs:
    - fine_tuning_compute
    - data_annotation
    - evaluation
    - infrastructure
    - ongoing_inference_premium (vs. base model)
  
  benefits:
    - accuracy_improvement (measured in reduced errors)
    - latency_reduction (smaller model = faster)
    - cost_savings (can use smaller model than base)
    - user_satisfaction
  
  breakeven:
    typical_prompt: 100M-500M tokens (PEFT)
    typical_prompt: 500M-2B tokens (full FT)
```

---

## 11. Future Directions

### Trends (H2 2026)
- **Automatic fine-tuning** — AI-driven hyperparameter and data selection
- **Multi-base model fine-tuning** — Train one adapter that works across model families
- **Fine-tuning on device** — Personalization on phone/laptop (Apple MLX, MLCC)
- **Test-time fine-tuning** — Adapt models on the fly during inference
- **Merged adapters** — Combine multiple LoRAs via model merging (TIES, DARE, SLERP)

### Open Challenges
- **Catastrophic forgetting** — Fine-tuned models lose general capabilities
- **Evaluation** — No standardized way to measure "how good is my fine-tuned model?"
- **Data quality** — Garbage in, garbage out; data collection is the bottleneck
- **Reproducibility** — Small changes in data order, LR, seeds produce different models

---

> **Related KB documents:**
> - [02-AI-Agent-Development.md](02-AI-Agent-Development.md) — Fine-tuning agents  
> - [05-AI-Safety-Alignment.md](05-AI-Safety-Alignment.md) — RLHF/DPO for alignment  
> - [06-RAG-Retrieval-Systems.md](06-RAG-Retrieval-Systems.md) — Embedding fine-tuning  
> - [08-Edge-AI-Inference.md](08-Edge-AI-Inference.md) — Quantization for edge
