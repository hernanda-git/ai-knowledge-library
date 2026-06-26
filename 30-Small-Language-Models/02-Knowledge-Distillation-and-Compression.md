# Knowledge Distillation & Model Compression Techniques

> June 2026 — Deep technical reference for creating small language models from large teachers

---

## Table of Contents

1. [Foundations of Knowledge Distillation](#1-foundations-of-knowledge-distillation)
2. [Logit-Based Distillation](#2-logit-based-distillation)
3. [Feature-Based Distillation](#3-feature-based-distillation)
4. [On-Policy Distillation](#4-on-policy-distillation)
5. [Sequence-Level Distillation](#5-sequence-level-distillation)
6. [Quantization-Aware Training (QAT)](#6-quantization-aware-training-qat)
7. [Post-Training Quantization (PTQ)](#7-post-training-quantization-ptq)
8. [Pruning: Structured & Unstructured](#8-pruning-structured--unstructured)
9. [Weight Merging & Model Soups](#9-weight-merging--model-soups)
10. [Hardware-Aware Optimization](#10-hardware-aware-optimization)
11. [Pipeline: Distilling a Production SLM](#11-pipeline-distilling-a-production-slm)
12. [Evaluation & Benchmarking](#12-evaluation--benchmarking)
13. [References & Papers](#13-references--papers)

---

## 1. Foundations of Knowledge Distillation

Knowledge distillation (KD) is the process of transferring knowledge from a large, high-capacity **teacher model** to a smaller **student model**. The core idea, introduced by Hinton et al. (2015), is that the student learns not just from the ground-truth labels but from the **softened probability distribution** of the teacher — capturing the dark knowledge of relative class probabilities.

### 1.1 The Distillation Objective

The canonical KD loss combines two terms:

```
L = α · L_hard(y_true, y_student) + (1 − α) · τ² · L_soft(y_teacher/T, y_student/T)
```

Where:
- `L_hard` — standard cross-entropy against ground truth labels
- `L_soft` — KL divergence between teacher and student softened logits
- `T` — temperature parameter controlling softness (higher T = softer distribution)
- `τ²` — scaling factor to balance gradient magnitudes (since soft targets have variance τ² smaller)
- `α` — mixing weight between hard and soft losses (typical: 0.5–0.9)

**Temperature schedule:** Temperature can be scheduled during training:
```
T_start = 20.0  →  T_end = 1.0
# Linear decay over training steps:
T = T_start + (T_end - T_start) · (step / total_steps)
```
High initial temperature enables the student to learn the full teacher distribution; low final temperature sharpens towards the ground truth.

### 1.2 Dark Knowledge

The teacher's soft distribution reveals:
- **Relative similarities** between classes (e.g., a "car" image producing 0.1 probability for "truck" but 0.001 for "cat")
- **Uncertainty boundaries** — regions where the teacher is unsure
- **Feature correlations** — learned through the teacher's internal representations

For language models, dark knowledge includes:
- Which next-token alternatives are plausible vs. impossible
- Style and tone preferences
- Reasoning paths (captured through chain-of-thought distillation)

### 1.3 Types of Knowledge Distillation

| Type | Knowledge Source | Transfer Method | Typical Use Case | Quality Retention |
|------|-----------------|-----------------|------------------|-------------------|
| **Logit-based** | Teacher output logits | KL divergence on soft targets | Classification, next-token prediction | 80–90% |
| **Feature-based** | Teacher hidden states | MSE on intermediate representations | Cross-architecture distillation | 85–95% |
| **On-policy** | Teacher-generated completions | Self-generated student data + teacher scoring | RLHF/RLAIF, alignment transfer | 90–98% |
| **Sequence-level** | Teacher full output distribution | SeqKD, Minimum Bayes Risk | Text generation, summarization | 85–92% |
| **Relation-based** | Teacher layer relationships | Graph matching | Deep representation transfer | 80–88% |

---

## 2. Logit-Based Distillation

### 2.1 Standard Logit KD (Hinton 2015)

**Algorithm:**
```python
def logit_kd_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.7):
    # Soften both distributions
    soft_student = F.log_softmax(student_logits / T, dim=-1)
    soft_teacher = F.softmax(teacher_logits / T, dim=-1)

    # KL divergence on soft targets
    kd_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T ** 2)

    # Standard cross-entropy on hard labels
    ce_loss = F.cross_entropy(student_logits, labels)

    # Combined loss
    return alpha * ce_loss + (1 - alpha) * kd_loss
```

**Critical implementation details:**
- Teacher logits must be pre-computed and cached (offline distillation) to avoid recomputation
- Batch normalization layers in the student need special handling — use frozen teacher BN statistics during warmup
- Temperature values > 10 cause vanishing gradients; values < 2 lose dark knowledge
- The `T²` scaling factor is often omitted in practice (Hinton 2015 vs. modern implementations) — its absence biases towards the hard loss

### 2.2 Normalized KD

**NormKD (2023)** addresses the logit magnitude mismatch between teacher and student:

```
L_normkd = KL(σ(z_s / τ || z_s ||) || σ(z_t / τ || z_t ||))
```

Where `||z||` is the L2 norm of logits. This is critical when:
- Teacher and student have different dimensionalities
- Student logit magnitudes are systematically smaller
- Fine-grained classification tasks (ImageNet, C4-100M)

Empirical finding: NormKD improves ImageNet top-1 accuracy by 1.2–2.1% over standard KD.

### 2.3 Decoupled KD (DKD)

**DKD (Zhao et al., 2022)** decomposes the KL divergence into:

```
KL = TC + NC
TC = target-class knowledge (binary)
NC = non-target class knowledge (distribution)
```

With separate temperature for each:
```
L_dkd = α · TC(τ_tc) + (1 - α) · NC(τ_nc)
# τ_tc > τ_nc — preserve dark knowledge for non-target classes
```

Results: DKD achieves 1.5–3% higher accuracy than standard KD on CIFAR-100 and ImageNet with the same student architecture.

### 2.4 Distillation for LLMs (MiniLLM Approach)

**MiniLLM (Gu et al., 2024)** adapts logit KD for autoregressive language models:

```python
def mini_llm_loss(student, teacher, input_ids):
    """Reverse KL divergence for generation tasks."""
    student_logits = student(input_ids).logits
    teacher_logits = teacher(input_ids).logits

    # Reverse KL: D_KL(student || teacher) encourages student to be "broad"
    # instead of "mode-seeking" (forward KL)
    student_probs = F.softmax(student_logits, dim=-1)
    teacher_probs = F.softmax(teacher_logits, dim=-1)

    # Reverse KL per token
    kl_per_token = (student_probs * (student_probs.log() - teacher_probs.log())).sum(-1)

    # Sequence-level masking
    return kl_per_token[input_ids != PAD_TOKEN].mean()
```

**Why reverse KL for LLMs:**
- Forward KL (`D_KL(teacher || student)`) is **mode-seeking** — student learns only teacher's high-probability tokens
- Reverse KL (`D_KL(student || teacher)`) is **mean-seeking** — student covers the full distribution
- For generation, mean-seeking prevents repetitive and safe-harbor outputs

### 2.5 DisCo (Distillation via Contrastive Objective)

**DisCo (2024)** reframes logit KD as contrastive learning:

```python
def disco_loss(student_rep, teacher_rep, temperature=0.07):
    """
    student_rep: [batch, hidden_dim]
    teacher_rep: [batch, hidden_dim]
    """
    # Normalize
    student_rep = F.normalize(student_rep, dim=-1)
    teacher_rep = F.normalize(teacher_rep, dim=-1)

    # SimCLR-style contrastive loss
    logits = student_rep @ teacher_rep.T / temperature
    labels = torch.arange(len(student_rep), device=student_rep.device)
    return F.cross_entropy(logits, labels)
```

Novel insight: The student learns to map its hidden states to the **same embedding space** as the teacher, aligning internal representations without explicit feature matching.

---

## 3. Feature-Based Distillation

### 3.1 FitNets (Romero et al., 2015)

The pioneering feature-based approach introduces a **regressor** to match intermediate representations:

```python
class HintRegressor(nn.Module):
    """Project student hidden states to teacher dimension."""
    def __init__(self, student_dim, teacher_dim, hidden_dim=512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(student_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, teacher_dim),
        )

    def forward(self, student_hidden):
        return self.proj(student_hidden)

def fitnet_loss(student_hidden, teacher_hidden, regressor):
    """MSE on intermediate feature maps after regressor."""
    projected = regressor(student_hidden.detach())
    return F.mse_loss(projected, teacher_hidden.detach())
```

**Key findings:**
- Earlier layers benefit more from feature-based KD than later layers
- The hint regressor absorbs most of the learning signal — it must be large enough
- Layer selection heuristic: match layers at 25%, 50%, 75% depths

### 3.2 Progressive KD

Train the student gradually from shallow to deep layers:

```python
progressive_stages = [
    {"layers": [0, 1, 2], "student_depth": 2, "epochs": 2},
    {"layers": [0, 1, 2, 3, 4], "student_depth": 4, "epochs": 2},
    # ... gradually increase depth
]
```

Each stage trains the student up to a partial depth, then initializes the next deeper stage from the shallower checkpoint. This prevents gradient vanishing in deep feature matching.

### 3.3 Attention Transfer (AT)

**Zagoruyko & Komodakis (2017)** — transfer spatial attention maps:

```python
def attention_transfer_loss(student_act, teacher_act):
    """Sum of normalized activation maps."""
    # Compute spatial attention from activations
    student_at = (student_act ** 2).sum(1)  # [batch, H, W]
    teacher_at = (teacher_act ** 2).sum(1)

    # Normalize
    student_at = F.normalize(student_at.view(len(student_at), -1), dim=1)
    teacher_at = F.normalize(teacher_at.view(len(teacher_at), -1), dim=1)

    return F.mse_loss(student_at, teacher_at)
```

For transformers, attention transfer maps to:
- **Self-attention maps** (per-head or aggregated)
- **Cross-attention patterns** (for encoder-decoder)
- **Key-query similarity matrices**

Empirical: AT adds 1–3% to student accuracy on vision tasks; 0.5–1.5% on language tasks.

### 3.4 Transformer-Specific Feature KD (TinyBERT, DistilBERT)

**TinyBERT (Jiao et al., 2020)** distills four components:

```python
def tinybert_loss(student, teacher, batch):
    losses = {}

    # 1. Embedding layer
    losses['emb'] = F.mse_loss(student.embeddings, teacher.embeddings)

    # 2. Attention maps (per-layer, Transformer-specific)
    for s_layer, t_layer in zip(student.encoder.layer, teacher.encoder.layer):
        s_attn = s_layer.attention.self.get_attention_map()
        t_attn = t_layer.attention.self.get_attention_map()
        losses[f'attn_{s_layer}'] = F.kl_div(
            F.log_softmax(s_attn / T, dim=-1),
            F.softmax(t_attn / T, dim=-1),
            reduction='batchmean'
        )

    # 3. Hidden states
    for s_hid, t_hid in zip(student.hidden_states, teacher.hidden_states):
        losses['hidden'] += F.mse_loss(s_hid, t_hid)

    # 4. Prediction logits (standard KD)
    losses['pred'] = kd_loss(student.logits, teacher.logits, batch.labels)

    return sum(losses.values())
```

**DistilBERT (Sanh et al., 2019)** simplifies this:
- Cosine distance on hidden states
- Combined with masked LM loss and KD loss
- 40% smaller, 60% faster, 97% retained performance

---

## 4. On-Policy Distillation

### 4.1 RL-Based Distillation

The student generates tokens, which the teacher evaluates:

```python
def on_policy_kd(student, teacher, prompt, T=1.0):
    """Generate from student, evaluate with teacher."""
    # Student generates
    student_outputs = student.generate(prompt, max_new_tokens=128)
    student_log_probs = student(prompt, student_outputs).log_probs

    # Teacher evaluates the SAME sequence
    teacher_log_probs = teacher(prompt, student_outputs).log_probs

    # Policy gradient: minimize reverse KL
    reward = -F.kl_div(
        student_log_probs,
        teacher_log_probs,
        reduction='none'
    ).sum(-1)

    # REINFORCE-style update
    baseline = reward.mean().detach()
    advantage = reward - baseline
    return -(student_log_probs.sum(-1) * advantage).mean()
```

**Critical advantage:** The student explores its own generation space and gets feedback from the teacher on those specific completions. This avoids the distribution mismatch of offline KD (where the teacher evaluated on its own generations, which differ from the student's).

### 4.2 SPIN (Self-Play Fine-Tuning)

**SPIN (Chen et al., 2024)** extends on-policy distillation into a self-play framework:

```python
def spin_iteration(student, teacher, dataset):
    """One SPIN iteration."""
    # Generate responses from student
    prompts, student_responses = student.generate_batch(dataset)

    # Score with teacher (preference model)
    teacher_scores = teacher.score(prompts, student_responses)

    # Student learns from preferred completions
    preferred = student_responses[teacher_scores > 0.5]
    rejected = student_responses[teacher_scores <= 0.5]

    # DPO-like preference optimization
    return dpo_loss(student, prompts, preferred, rejected)
```

**Key insight:** The student improves, becomes the "new teacher" for the next iteration, and the process repeats. After 3–4 SPIN iterations, a Phi-3-small can match its teacher on 85% of benchmarks.

### 4.3 Self-KD

The student distills **itself** — using saved checkpoints from earlier training:

```python
def self_kd(student, previous_checkpoints, batch, alpha=0.5):
    """Distill from earlier versions of the same model."""
    teacher_preds = []
    for ckpt in previous_checkpoints:
        prev = load_model(ckpt)
        with torch.no_grad():
            pred = prev(batch)
        teacher_preds.append(pred)

    # Ensemble teacher = average of past predictions
    ensemble_teacher = torch.stack(teacher_preds).mean(0)

    # Standard KD against this ensemble
    return kd_loss(student.logits, ensemble_teacher, batch.labels) * alpha + \
           F.cross_entropy(student.logits, batch.labels) * (1 - alpha)
```

Self-KD doesn't need an external teacher — the model learns from its own history. Empirically, 3–5 past checkpoints stored every 10K steps work best. More checkpoints cause diminishing returns.

### 4.4 Online Distillation

Teacher and student are trained simultaneously:

```python
def online_kd(student, teacher, batch, student_lr=5e-5, teacher_lr=1e-5):
    """Teacher and student evolve together."""

    # Forward passes
    student_out = student(batch)
    teacher_out = teacher(batch)

    # Peer teaching loss (bidirectional KD)
    s2t_loss = kd_loss(teacher_out.logits, student_out.logits, batch.labels)
    t2s_loss = kd_loss(student_out.logits, teacher_out.logits, batch.labels)

    # Student hard loss on ground truth
    student_main_loss = F.cross_entropy(student_out.logits, batch.labels)

    # Teacher main loss on ground truth
    teacher_main_loss = F.cross_entropy(teacher_out.logits, batch.labels)

    # Combined
    student_loss = student_main_loss + t2s_loss
    teacher_loss = teacher_main_loss + s2t_loss * 0.5

    return student_loss, teacher_loss
```

Used in: **Born-Again Networks** (Furlanello et al., 2018), **Deep Mutual Learning** (Zhang et al., 2018).

---

## 5. Sequence-Level Distillation

### 5.1 Sequence-Level KD (SeqKD)

For autoregressive generation, the teacher's **full sequence distribution** matters:

```python
def seqkd_loss(student, teacher, prompts, max_len=128, num_samples=5):
    """
    Teacher generates K candidate completions; student learns from
    the marginal distribution.
    """
    total_loss = 0

    for prompt in prompts:
        # Teacher generates K completions
        completions = teacher.generate(
            prompt,
            max_new_tokens=max_len,
            num_return_sequences=num_samples,
            temperature=0.7,
            do_sample=True,
        )

        # For each completion, compute teacher probability
        teacher_probs = teacher.score(prompt, completions)

        # Weighted student loss — high teacher prob = more weight
        for comp, prob in zip(completions, teacher_probs):
            student_log_prob = student(prompt, comp).log_probs.sum()
            total_loss -= prob * student_log_prob

    return total_loss / len(prompts)
```

**Advantage over token-level KD:**
- Captures teacher's **global sequence preferences** (length, style, structure)
- Avoids the "exposure bias" of teacher-forcing
- 2–5 BLEU point improvement on machine translation tasks

### 5.2 Minimum Bayes Risk (MBR) Distillation

**MBR (Kumar et al., 2024)** selects the "safest" teacher completions:

```python
def mbr_distillation(student, teacher, prompt, num_hyps=10):
    # Generate diverse candidates from teacher
    hypotheses = teacher.generate(
        prompt, num_return_sequences=num_hyps, temperature=1.0
    )

    # Score each hypothesis against all others
    risks = []
    for i, h_i in enumerate(hypotheses):
        risk = 0
        for j, h_j in enumerate(hypotheses):
            if i != j:
                risk += 1 - similarity(h_i, h_j)  # e.g., BLEU, ROUGE
        risks.append(risk)

    # Select lowest-risk hypothesis
    best_idx = torch.argmin(torch.tensor(risks))
    best_hyp = hypotheses[best_idx]

    # Student learns to produce best_hyp via supervised fine-tuning
    return F.cross_entropy(student(prompt, best_hyp).logits, best_hyp)
```

MBR distillation is the preferred method for **domain-specific SLMs** (e.g., medical, legal, code) where the teacher must be faithfully compressed without hallucination.

### 5.3 Contrastive Distillation for Generation

**CDG (2024)** adds contrastive learning between teacher-preferred and teacher-disfavored sequences:

```python
def contrastive_distill(student, teacher, prompt, teacher_logits=None):
    # Generate candidates from teacher with temperature
    candidates = teacher.generate(prompt, num_return_seqs=5, temperature=0.8)

    # Score each candidate with teacher
    scores = F.softmax(teacher(prompt, candidates).logits / T, dim=-1)

    # Select positive (high teacher prob) and negative (low teacher prob)
    top_idx = scores.argmax()
    bottom_idx = scores.argmin()

    # Contrastive loss: maximize margin between positive and negative
    pos_log_prob = student(prompt, candidates[top_idx]).log_probs.sum()
    neg_log_prob = student(prompt, candidates[bottom_idx]).log_probs.sum()

    margin = F.relu(neg_log_prob - pos_log_prob + MARGIN)
    return margin.mean()
```

Margin `M` is typically 0.5–1.0 nats. Too high collapses to positive-only; too low doesn't differentiate.

---

## 6. Quantization-Aware Training (QAT)

### 6.1 QAT Foundations

QAT simulates quantization during training, allowing the model to learn to compensate for resolution loss:

```python
class FakeQuantize(torch.autograd.Function):
    """Straight-Through Estimator (STE) for quantization."""

    @staticmethod
    def forward(ctx, x, scale=1.0, zero_point=0, bits=8):
        """Quantize → dequantize, breaking gradient at quantization boundaries."""
        qmin, qmax = 0, 2**bits - 1
        x_int = torch.round(x / scale + zero_point)
        x_int = torch.clamp(x_int, qmin, qmax)
        x_q = (x_int - zero_point) * scale
        return x_q

    @staticmethod
    def backward(ctx, grad_output):
        """STE: pass gradient through as if quantization never happened."""
        return grad_output, None, None, None
```

**The STE trick:** During forward pass, weights are quantized (discrete). During backward, gradients pass through unaffected. This is the single most important idea in QAT — without it, the zero-gradient problem of quantization makes training impossible.

### 6.2 QAT Hyperparameters

| Parameter | Typical Value | Effect |
|-----------|--------------|--------|
| Start step | 10% of training | Warmup allows model to stabilize first |
| Bit width schedule | 8→6→4 bit | Gradual quantization avoids sudden accuracy loss |
| Scale initialization | Min-max on initial batch | Over-estimation leads to wasted precision |
| Learning rate | 1/10 of full-precision LR | Quantization noise requires smaller steps |
| Batch size | 2× larger | Quantization noise needs more averaging |

### 6.3 Weight vs. Activation Quantization

```python
def qat_forward(layer, x, bits_w=4, bits_a=8):
    """
    Different precision for weights vs activations.
    Weights can tolerate lower precision (4-bit) than activations (8-bit).
    """
    # Quantize weights
    weight_scale = layer.weight.abs().max() / (2**(bits_w-1) - 1)
    q_weight = FakeQuantize.apply(layer.weight, weight_scale, 0, bits_w)

    # Quantize activations (if specified)
    if bits_a < 16:
        act_scale = x.abs().max() / (2**(bits_a-1) - 1)
        x = FakeQuantize.apply(x, act_scale, 0, bits_a)

    return F.linear(x, q_weight, layer.bias)
```

**Industry practice (2026):**
- Weight-only quantization: **4-bit** (NF4 in QLoRA, FP4 in GPTQ) — negligible accuracy loss
- Weight + activation quantization: **W4A8** (4-bit weights, 8-bit activations) — 0.5–1% loss
- Full integer quantization: **W8A8** — 0.1–0.3% loss, 2–4× speedup on INT8 hardware

### 6.4 QAT for SLMs — LLM-QAT

**LLM-QAT (Liu et al., 2024)** adapts QAT for large language models:

```python
def llm_qat_step(model, batch, teacher=None, bits_w=4, bits_a=8):
    """QAT step with optional teacher distillation."""

    # Apply fake quantization to all linear layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.weight = FakeQuantize.apply(
                module.weight,
                module.weight_scale.detach(),
                0,
                bits_w
            )

    # Forward with quantization
    outputs = model(batch)

    if teacher is not None:
        # Distillation loss (prevents quantization drift)
        with torch.no_grad():
            teacher_outputs = teacher(batch)
        loss = kd_loss(outputs.logits, teacher_outputs.logits, batch.labels)
    else:
        loss = F.cross_entropy(outputs.logits, batch.labels)

    return loss
```

**Key insight:** Without teacher distillation during QAT, 4-bit models lose 5–15% on benchmarks. With online KD, the loss drops to 0.5–2%.

### 6.5 Quantization Formats

| Format | Bits | Range | Use Case | Vendor |
|--------|------|-------|----------|--------|
| **INT8** | 8 | [-128, 127] | Server inference | All |
| **INT4** | 4 | [-8, 7] | GPU inference | NVIDIA, AMD |
| **NF4** | 4 | Normal float | QLoRA finetuning | bitsandbytes |
| **FP8 (E4M3)** | 8 | ±448 | Training | NVIDIA H100+ |
| **FP6 (E3M2)** | 6 | ±28 | Inference | DeepSpeed |
| **FP4 (E2M1)** | 4 | ±6 | Extreme compression | DeepSpeed |
| **MXFP4** | 4 | Block-scaled | Hardware efficient | AMD, Intel |

**The NF4 innovation:** QLoRA's NormalFloat4 is information-theoretically optimal for normally distributed weights. Since neural network weights approximate a normal distribution with zero mean, NF4 allocates more precision near zero where most weights live:

```
NF4 buckets: [-1.0, -0.696, -0.525, -0.394, ... 0.0, ... 0.394, 0.525, 0.696, 1.0]
                     (narrow spacing near 0, wide spacing at extremes)
```

This 4-bit encoding retains 96%+ of the 16-bit model quality.

### 6.6 LSQ (Learned Step Size Quantization)

**LSQ (Esser et al., 2020)** makes the quantization scale a learnable parameter:

```python
class LSQQuantizer(nn.Module):
    """
    Learned Step Size Quantization.
    Scale = learnable parameter, initialized from weight statistics.
    """
    def __init__(self, bits=8, init_scale=0.01):
        super().__init__()
        self.bits = bits
        self.scale = nn.Parameter(torch.tensor(init_scale))
        self.zero_point = 0

    def forward(self, x):
        qmin, qmax = 0, 2**self.bits - 1

        # Scale gradient: g = ∂L/∂s · (∂q/∂s · ∂L/∂q)
        # Gradient scaling factor (from paper)
        g = 1.0 / math.sqrt(x.numel() * self.qmax)

        return FakeQuantize.apply(x, self.scale, self.zero_point, self.bits)
```

LSQ improves QAT accuracy by 0.5–1% over fixed-scale approaches. The gradient scaling factor `g` is critical — without it, the scale parameter's gradient magnitude is mismatched with weight gradients.

---

## 7. Post-Training Quantization (PTQ)

### 7.1 GPTQ

**GPTQ (Frantar et al., 2023)** — Optimal Brain Quantization adapted for LLMs:

```python
def gptq_quantize(weight_matrix, hessian, bits=4):
    """
    layer['weight']: [out_dim, in_dim]
    hessian: [in_dim, in_dim] — second-order gradients
    """
    W = weight_matrix.clone().float()
    H = hessian + 0.01 * torch.eye(hessian.shape[0])  # dampen

    # Cholesky decomposition for efficient inverse
    H_inv = torch.linalg.cholesky(H)
    H_inv = torch.cholesky_inverse(H_inv)

    Q = torch.zeros_like(W)

    # Sequential quantization with inverse-Hessian compensation
    for i in range(W.shape[1]):
        w = W[:, i]
        q = quantize(w, bits)  # round to nearest quantized value

        error = q - w  # quantization error
        Q[:, i] = q

        if i < W.shape[1]:
            # Update remaining columns to compensate for error
            update_cols = W[:, i+1:]
            damp = H_inv[i, i:]
            update_cols += torch.outer(error, damp) / damp[0]

    return Q
```

**Why GPTQ works so well:**
- Uses the **Hessian** of the loss function to guide quantization
- Compensates for quantization error in remaining weights
- Scales to 70B+ models on a single A100 in 4–8 hours
- 4-bit GPTQ: 0.5–1% perplexity degradation on WikiText-2

**Implementation gotchas:**
- Hessian computation requires 128–1024 calibration samples (C4 or WikiText-2)
- Use `group_size=128` for the best accuracy/speed tradeoff
- Symmetric quantization works better for weights (accidental finding)
- Activation ordering (`--act-order`) reduces perplexity by 0.05–0.1 PPL

### 7.2 AWQ (Activation-Aware Weight Quantization)

**AWQ (Lin et al., 2024)** observes that not all weights are equally important:

```python
def awq_quantize(weight, activations, bits=4, alpha=0.5):
    """
    weight: [out_dim, in_dim]
    activations: [num_calib_samples, in_dim] — calibration data
    """
    # Compute activation magnitudes per input channel
    act_scale = activations.abs().mean(0)  # [in_dim]

    # Normalize to [0, 1]
    act_scale = act_scale / act_scale.max()

    # Scale important channels (those with high activation)
    saliency = act_scale ** alpha
    scaled_weight = weight * saliency

    # Quantize scaled weights
    Q = quantize(scaled_weight, bits)

    # Rescale back
    Q = Q / saliency

    return Q
```

**Key insight:** Weights corresponding to high-activation input channels (top 1%) are 10–100× more important than the bottom 50%. AWQ protects these channels by scaling them before quantization.

**Results:**
- Outperforms GPTQ at 4-bit by 0.3–0.5 PPL
- Especially effective at 3-bit (nearly 1 PPL improvement)
- Integrated into vLLM TensorRT-LLM as the default PTQ method
- No calibration Hessian needed — 10× faster than GPTQ

### 7.3 SmoothQuant

**SmoothQuant (Xiao et al., 2024)** addresses activation outliers — the #1 obstacle to activation quantization:

```python
def smoothquant_transform(weights, activations, alpha=0.5):
    """
    Smooth out activation outliers by migrating quantization difficulty
    from activations to weights.
    """
    # Per-channel scaling factor
    s_x = activations.abs().max(dim=0).values  # [in_dim]
    s_w = weights.abs().max(dim=1).values       # [out_dim]

    # Migrate difficulty: W8A8 quantization
    # higher alpha = more difficulty moved from activations to weights
    scaling = s_x ** alpha / s_w ** (1 - alpha)
    scaling = scaling.clamp(max=1e4)

    # Smooth activations (divide by scaling)
    smoothed_act = activations / scaling

    # Boost weights (multiply by scaling)
    boosted_weight = weights * scaling

    return boosted_weight, smoothed_act
```

**Without SmoothQuant:** Activation quantization fails because 0.1% of tokens carry 90% of the magnitude (see "massive outlier" phenomenon). W8A8 loses 3–10% accuracy.

**With SmoothQuant:** W8A8 retains full-precision accuracy. W6A6 loses only 0.5%. W4A4 loses 2–3%.

**The outlier phenomenon (Dettmers et al., 2022):**
- LLM activations have ~20 hidden dimension positions where values are 10–100× larger than average
- These outliers appear in specific attention heads after layer 8
- They are **consistent across inputs** — some positions are always outliers
- SmoothQuant handles this by reducing the outlier magnitude in activations while compensating in weights

### 7.4 AQLM (Additive Quantization of Language Models)

**AQLM (2024)** is the current SOTA for extreme compression (2-bit):

```python
def aqlm_compress(weight, num_codebooks=8, codebook_size=256):
    """
    Represent each weight vector as a sum of K codebook vectors,
    each chosen from a learned codebook of size C.
    """
    W = weight  # [out_dim, in_dim]
    k, c = num_codebooks, codebook_size
    d = W.shape[1]

    # Initialize codebooks [k, c, d/k] — each codebook is a set of sub-vectors
    codebooks = init_codebooks(k, c, d // k)

    # For each output neuron, find the best combination of codebook entries
    # (k = 8, c = 256) → 8 * log2(256) = 64 bits per group
    # For d=4096 with group_size=8 → 64/8 = 8 bits per weight = 2 bits per param
    W_compressed = []
    for i in range(0, d, d // k):
        chunk = W[:, i:i + d // k]
        best_codes = find_best_codes(chunk, codebooks)
        W_compressed.append(combine(chunk, best_codes))

    return torch.cat(W_compressed, dim=1)
```

At the same size (8× compression), AQLM outperforms:
- NF4 by 1.8 PPL
- GPTQ-4bit-128g by 0.5 PPL
- AWQ-4bit by 0.4 PPL

At 2-bit (16× compression), AQLM retains chat quality while NF4 and GPTQ fail. Used in **QuIP#** for 2-bit llama.cpp support.

---

## 8. Pruning: Structured & Unstructured

### 8.1 Magnitude Pruning

The simplest and still most effective baseline:

```python
def magnitude_prune(model, sparsity=0.5):
    """Remove smallest-magnitude weights."""
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            threshold = torch.quantile(param.abs().flatten(), sparsity)
            mask = param.abs() > threshold
            param.data *= mask
```

**Issue:** Magnitude pruning at high sparsity (80%+) causes catastrophic accuracy loss for LLMs. It works for vision CNNs but not for attention-based models.

### 8.2 SparseGPT

**SparseGPT (Frantar & Alistarh, 2023)** — OBS for unstructured pruning:

```python
def sparsegpt_prune(layer_weights, hessian, sparsity=0.5):
    """
    One-shot unstructured pruning at 50%+ sparsity.
    Uses Hessian-based error compensation, like GPTQ but for pruning.
    """
    W = layer_weights.clone().float()
    H = hessian + 0.01 * eye(layer_weights.shape[1])

    H_inv = torch.linalg.inv(H)  # [in_dim, in_dim]

    # Order weights by importance: (w² / diag(H⁻¹))
    importance = W ** 2 / H_inv.diag().unsqueeze(0)

    # Prune lowest-importance weights
    threshold = torch.quantile(importance.flatten(), sparsity)
    mask = importance > threshold

    # Update remaining weights to compensate
    W_pruned = W * mask
    error = W - W_pruned
    W_updated = W_pruned + compensate(error, H_inv, mask)

    return W_updated
```

**Results:**
- 50% sparsity: no accuracy loss on OPT, BLOOM, Llama
- 60% sparsity: 0.5–1% perplexity increase
- 70% sparsity: 2–5% perplexity increase
- No retraining required (truly one-shot)

### 8.3 Wanda (Weight-Activation-Aware Pruning)

**Wanda (Sun et al., 2024)** — simpler than SparseGPT with competitive results:

```python
def wanda_prune(weight, activations, sparsity=0.5):
    """
    Prune score = |weight| · ||activation_column||₂
    No Hessian needed.
    """
    # Activation norms per input column
    act_norm = activations.norm(p=2, dim=0)  # [in_dim]

    # Per-output-neuron importance: element-wise product
    importance = weight.abs() * act_norm.unsqueeze(0)  # [out_dim, in_dim]

    # Prune per row (each output neuron has its own threshold)
    for i in range(importance.shape[0]):
        row = importance[i]
        thresh = torch.quantile(row, sparsity)
        if thresh > 0:
            weight[i][row < thresh] = 0

    return weight
```

**Key finding:** The activation norm `||x_j||₂` weights each input dimension by its typical magnitude. Dimensions that are always small are less important even if their weights are large. Wanda eliminates the Hessian computation, making it 100× faster than SparseGPT with comparable (90–95%) quality.

### 8.4 Structured Pruning (SLMC)

**SLMC (2023)** prunes entire attention heads and FFN intermediate dimensions:

```python
def prune_attention_heads(model, calibration_data, threshold=0.1):
    """
    Measure head importance via accumulated attention probability.
    Prune heads with importance below threshold.
    """
    head_importance = defaultdict(float)

    for batch in calibration_data:
        outputs = model(batch, output_attentions=True)

        for layer_idx, layer_attn in enumerate(outputs.attentions):
            # layer_attn: [batch, heads, seq, seq]
            # Head importance = sum of attention probabilities
            for head_idx in range(layer_attn.size(1)):
                head_mask = layer_attn[:, head_idx].detach()
                head_importance[(layer_idx, head_idx)] += head_mask.sum().item()

    # Normalize and prune
    max_importance = max(head_importance.values())
    to_prune = [(k, v) for k, v in head_importance.items()
                if v / max_importance < threshold]
    return to_prune
```

**LayerDrop (Fan et al., 2020):** A different approach — train with stochastic depth (randomly drop layers), then prune the least important layers at inference. This produces models where you can drop 30% of layers with <1% accuracy loss.

### 8.5 Compression-Aware Architecture Design

Modern SLMs design for compressibility from the start:

| Technique | Compression | Accuracy Impact | Models |
|-----------|-------------|-----------------|--------|
| **ALBERT** weight-sharing | 18× (vs BERT) | -2% | ALBERT-xxlarge |
| **DistilBERT** layer reduction | 2× (vs BERT) | -3% | DistilBERT-base |
| **MobileBERT** bottleneck | 4× (vs BERT) | -1.5% | MobileBERT |
| **DeLighT** group-wise | 2–4× | -1–2% | DeLighT |
| **TinyBERT** combined KD+pruning | 7.5× (vs BERT) | -3% | TinyBERT-4L |
| **Phi-3** data-centric training | 4× (vs 7B) | +5% | Phi-3-mini (3.8B) |

---

## 9. Weight Merging & Model Soups

### 9.1 Model Soups (Wortsman et al., 2022)

Simple averaging of multiple fine-tuned checkpoints:

```python
def model_soup(models, weights=None):
    """
    Average weights of N models fine-tuned from the same base.
    Each model: different hyperparameters, data order, seeds.
    """
    if weights is None:
        weights = torch.ones(len(models)) / len(models)

    averaged_state = {}
    for key in models[0].state_dict():
        averaged_state[key] = sum(
            m.state_dict()[key] * w for m, w in zip(models, weights)
        )

    return averaged_state
```

**Results:** +0.5–1.5% accuracy on ImageNet. The soup consistently outperforms any individual model. Works because different training runs converge to different but equally valid local minima.

### 9.2 TIES-Merging (Task-Arithmetic)

**TIES (Yadav et al., 2024)** resolves conflicts when merging models fine-tuned on different tasks:

```python
def ties_merge(base_model, task_models, sparsity=0.2):
    """
    1. Trim redundant task vectors (top-k sparsity)
    2. Elect sign (majority vote per parameter)
    3. Disjoint merge (average only aligned signs)
    """
    base_params = base_model.state_dict()
    merged = copy.deepcopy(base_params)

    for key in base_params:
        task_deltas = []
        for task_model in task_models:
            delta = task_model.state_dict()[key] - base_params[key]
            # Step 1: Trim — keep only top-k by magnitude
            k = int(delta.numel() * (1 - sparsity))
            threshold = torch.kthvalue(delta.abs().flatten(), k).values
            delta[delta.abs() < threshold] = 0

            task_deltas.append(delta)

        # Step 2: Elect sign — majority vote
        stack = torch.stack(task_deltas)
        sign_votes = torch.sign(stack).sum(0)
        majority_sign = torch.sign(sign_votes)

        # Step 3: Disjoint merge — average only parameters
        # where the task vector sign matches the majority sign
        valid_mask = (torch.sign(stack) == majority_sign.unsqueeze(0))
        merged_delta = (stack * valid_mask).sum(0) / valid_mask.sum(0).clamp(min=1)

        merged[key] = base_params[key] + merged_delta

    return merged
```

**Results:** TIES-merging 8 Llama-3.2-3B models fine-tuned on different tasks (math, code, QA, summarization, translation, etc.) produces a single model that matches or exceeds performance of individually fine-tuned models on each task. Test loss increase: <0.5%.

### 9.3 DARE (Drop And REscale)

**DARE (Yu et al., 2024)** — drop most delta parameters and rescale:

```python
def dare_merge(base_model, task_models, drop_rate=0.9, scale=1.0):
    """
    Randomly drop most task-specific parameters (90%+) and rescale.
    Counter-intuitively, this IMPROVES merged performance.
    """
    base_params = base_model.state_dict()
    merged = copy.deepcopy(base_params)

    for key in base_params:
        deltas = []
        for task_model in task_models:
            delta = task_model.state_dict()[key] - base_params[key]

            # Randomly drop 90% of parameters
            mask = torch.rand_like(delta) > drop_rate
            delta = delta * mask

            # Rescale to maintain expected magnitude
            delta = delta / (1 - drop_rate) * scale

            deltas.append(delta)

        merged[key] = base_params[key] + sum(deltas)

    return merged
```

**Why dropping 90% works:** Most task-specific parameters during fine-tuning are small adjustments with high noise. The 10% that remain carry 95%+ of the task signal. This explains why low-rank adaptation (LoRA) works — the effective rank of fine-tuning is extremely low.

### 9.4 Franken-Merging (Cross-Architecture)

Merging models of different architectures by aligning their hidden dimensions via a linear probe:

```python
def franken_merge(encoder_model, decoder_model, alignment_n=512):
    """
    Merge an encoder's knowledge into a decoder via a learned linear bridge.
    This creates a novel architecture impossible to train from scratch.
    """
    bridge = nn.Linear(encoder_model.hidden_size, decoder_model.hidden_size)

    # Train bridge on paired hidden states
    for batch in paired_dataset:
        with torch.no_grad():
            enc_states = encoder_model.get_hidden_states(batch.input_ids)
            dec_states = decoder_model.get_hidden_states(batch.input_ids)
        loss = F.mse_loss(bridge(enc_states), dec_states)
        loss.backward()
        optimizer.step()

    # The merged model uses the encoder's early layers and decoder's later layers
    # connected through bridge
    return MergedModel(encoder_model, bridge, decoder_model)
```

Used by **Sakana AI** to create novel model architectures from merging. Notably, they merged vision encoders with language decoders to create multimodal models 10× smaller than training from scratch.

---

## 10. Hardware-Aware Optimization

### 10.1 Apple Neural Engine (ANE)

Optimizing SLMs for Apple's ANE (40 TOPS, 16-core, Int8):

```python
def prepare_for_ane(model):
    """Optimizations required for ANE inference."""
    # 1. All weights must be FP16 or INT8 (ANE doesn't support FP32)
    model = model.half()

    # 2. Replace GeLU with ReLU (GeLU has no ANE kernel)
    for name, module in model.named_modules():
        if isinstance(module, nn.GELU):
            parent = get_parent(model, name)
            setattr(parent, name.split('.')[-1], nn.ReLU())

    # 3. Use split-SILU (ANE-specific activation optimization)
    # 4. Swin attention pattern (ANE optimized block-sparse)
    # 5. Pre-compute and freeze embedding tables (ANE fixed function units)

    return model
```

**Performance on Apple M4 (2025):**
- Phi-3-mini (3.8B): 45 tokens/sec
- Llama-3.2-3B: 38 tokens/sec
- Gemma 2 2B: 55 tokens/sec

**Key bottleneck:** ANE's shared memory bandwidth (120 GB/s on M4 Pro). Large models are bandwidth-bound, not compute-bound.

### 10.2 Qualcomm Hexagon NPU

Qualcomm's Snapdragon X Elite / 8 Gen 3:

```python
def optimize_for_hexagon(model):
    """Hexagon NPU quantization targeting."""
    # 1. Use INT8 exclusively (Hexagon native)
    # 2. Conv2D → Im2Col + GEMM (Hexagon has optimized im2col)
    # 3. Pre-compute LayerNorm parameters
    # 4. Fuse adjacent operations: add + relu, conv + bn, linear + gelu

    return model
```

**Packing tokens into batches:** Hexagon excels at batch=1 inference with pipelining. The critical path is the first token (cold start: 80–200ms). Subsequent tokens: 15–40ms.

### 10.3 Google Edge TPU

```python
def optimize_for_edgetpu(model):
    """Edge TPU constraints and optimizations."""
    # 1. Static shapes only (no dynamic padding)
    # 2. Max dimension: 2048 for Edge TPU v5
    # 3. Use depthwise separable convolutions
    # 4. Model must be fully quantized to INT8

    return model
```

Edge TPU v5 (2024) can run Bert-base at 8ms latency at batch=1. Transformer models must have fixed sequence lengths.

### 10.4 NVIDIA TensorRT-LLM

Integration path for GPU-optimized SLMs:

```python
import tensorrt_llm as trt_llm

def build_trt_engine(model_path, precision='float16', max_batch=1, max_seq_len=4096):
    """Build TensorRT-LLM engine for optimal GPU inference."""
    builder = trt_llm.Builder()

    # Config
    builder.max_batch_size = max_batch
    builder.max_input_len = max_seq_len
    builder.precision = precision  # 'float16', 'bfloat16', 'int8', 'int4'

    # INode chunking — split FFN across GPUs
    builder.enable_inflight_batching = True
    builder.gpu_weights_per_gpu = 0.5  # split across 2 GPUs

    # KV cache optimizations
    builder.paged_kv_cache = True
    builder.kv_cache_dtype = 'fp8'     # KV cache quantization

    engine = builder.build(model_path)
    return engine
```

**Performance (A100):**
- Phi-3-mini 3.8B: 2,500 tok/s (batch=32)
- Llama-3.2-3B: 2,100 tok/s (batch=32)
- Gemma 2 2B: 3,200 tok/s (batch=32)

### 10.5 ExecuTorch (Meta)

Meta's on-device deployment framework:

```python
# Export to ExecuTorch
import executorch as et

def export_for_executorch(model, example_input):
    """Export PyTorch model to ExecuTorch format for mobile/edge."""
    # Quantize
    quantized = et.quantize.quantize(model, et.quantize.PT2EQuantizer())

    # Lower to delegate (XNNPACK, CoreML, QNN)
    lowered = et.lower(quantized, example_input, backend='xnnpack')

    # Export to .pte file
    et.save(lowered, "model.pte")
```

**Platform performance (2025):**
- Pixel 9 Pro: Llama-3.2-1B at 45 tok/s
- Samsung S25: Llama-3.2-1B at 40 tok/s
- iPhone 16 Pro: Phi-3-mini at 35 tok/s

---

## 11. Pipeline: Distilling a Production SLM

### 11.1 End-to-End Distillation Pipeline

Based on the Phi-4 distillation process (Microsoft, 2025):

```python
def distill_slm_pipeline():
    """Production SLM distillation pipeline (120 GPU-hours on 8×A100)."""

    # Phase 0: Data curation (2 weeks, human-in-loop)
    # ================================================
    # Collect 15T tokens high-quality data:
    #   - 10T filtered web (C4 + CommonCrawl + all-CC-id)
    #   - 3T code (GitHub Python + Java + JS + web + system)
    #   - 1T math (proof-pile-2, GSM8K, MATH, AoPS)
    #   - 1T multilingual (mC4 filtered for 50 languages)

    # Phase 1: Teacher pre-training (expensive, done once)
    # ====================================================
    teacher = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.1-70B")
    # Teacher is frozen; only used for logit generation

    # Phase 2: Cached logit generation (128 A100-80GB, 1 week)
    # =========================================================
    # Generate teacher logits for 500B tokens (offline KD):
    #   - batch_size = 32 per GPU, sequences of 4096 tokens
    #   - Store (soft_labels, token_ids) pairs in HF Datasets
    #   - Storage: ~3TB of soft targets (FP16 logits × vocab)
    #   - Speed: ~300M tokens/hour with 128 GPUs

    # Phase 3: Token-level pretraining distillation (6 days)
    # =======================================================
    student = SmallLMConfig(
        hidden_size=2560,
        num_hidden_layers=32,
        num_attention_heads=20,
        intermediate_size=6912,
        vocab_size=128256,
    ).create_model()

    for step, batch in enumerate(cached_data_loader):
        # Load cached teacher logits for this batch
        student_logits = student(batch.input_ids).logits
        teacher_logits = batch.teacher_logits  # pre-computed

        # Three-term loss
        distill_loss = kd_loss(
            student_logits, teacher_logits, batch.labels,
            T=4.0, alpha=0.5
        )
        feature_loss = attention_transfer_loss(
            student.get_attention_maps(),
            batch.teacher_attention_maps
        )
        total_loss = distill_loss + 0.1 * feature_loss

        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if step % 1000 == 0:
            evaluate(student)

    # Phase 4: On-policy refinement (1 day)
    # ======================================
    for step, prompt in enumerate(refinement_dataset):
        student_completion = student.generate(prompt)
        teacher_scores = teacher.score(prompt, student_completion)
        on_policy_loss = reinforce_loss(student, student_completion, teacher_scores)
        on_policy_loss.backward()
        optimizer.step()

    # Phase 5: QAT + Pruning (2 days)
    # ==================================
    for step, batch in enumerate(qat_dataset):
        # Apply fake quantization
        apply_fake_quant(student, bits_w=4, bits_a=8)

        # Prune 25% of lowest-importance heads
        prune_heads(student, calibration_data, sparsity=0.25)

        # Teacher-assisted QAT (prevent drift)
        student_out = student(batch)
        teacher_out = teacher(batch)
        qat_loss = kd_loss(student_out.logits, teacher_out.logits, batch.labels)

        qat_loss.backward()
        optimizer.step()

    # Phase 6: Alignment via DPO (1 day)
    # ===================================
    for step, (prompt, chosen, rejected) in enumerate(preference_data):
        # Direct Preference Optimization
        chosen_logps = student(prompt, chosen).log_probs
        rejected_logps = student(prompt, rejected).log_probs
        dpo_loss = -F.logsigmoid(chosen_logps - rejected_logps).mean()
        dpo_loss.backward()
        optimizer.step()

    return student.eval()
```

### 11.2 Resource Requirements

| Phase | GPUs | Time | Cost (Cloud) | Data |
|-------|------|------|-------------|------|
| Teacher pre-training | 512×H100 | 2 months | $2M | 15T tokens |
| Cached logit generation | 128×A100 | 1 week | $15K | 500B tokens |
| Token-level distillation | 8×A100 | 6 days | $1.2K | 500B cached logits |
| On-policy refinement | 4×A100 | 1 day | $200 | 50M prompts |
| QAT + pruning | 4×A100 | 2 days | $400 | 10B calibration |
| Alignment (DPO) | 4×A100 | 1 day | $200 | 1M preferences |
| **Total** | — | **~15 days** | **~$17K (distill only)** | — |

### 11.3 Offline KD Data Format

```json
{
    "input_ids": [1, 1234, 5678, ...],
    "attention_mask": [1, 1, 1, ...],
    "teacher_logits": [[0.1, 0.2, ...], ...],  // [seq_len, vocab_size], FP16
    "labels": [1234, 5678, ...],
    "loss_mask": [1, 1, 0, 1, ...],  // mask padding and prompt tokens
    "token_ids": [1234, 5678, ...]
}
```

Total per-token overhead: `vocab_size × FP16 = 128K × 2 = 256KB` per token. For 500B tokens ≈ 128TB of cached logits. This is the dominant cost of offline KD and motivates **online distillation** approaches.

---

## 12. Evaluation & Benchmarking

### 12.1 Benchmark Suite for SLMs

| Benchmark | Metric | Target | Purpose |
|-----------|--------|--------|---------|
| MMLU-Pro | Accuracy | 55–72% (3B), 60–78% (7B) | Knowledge & reasoning |
| HumanEval | Pass@1 | 35–50% (3B), 45–65% (7B) | Code generation |
| GSM8K | Accuracy | 55–75% (3B), 65–85% (7B) | Math word problems |
| HellaSwag | Accuracy | 60–75% (3B), 65–80% (7B) | Commonsense reasoning |
| Arc-Challenge | Accuracy | 50–65% (3B), 60–75% (7B) | Science Grade 8 |
| TruthfulQA | Accuracy | 40–55% (3B), 45–60% (7B) | Factuality |
| IFEval | Strict Acc | 55–70% (3B), 60–75% (7B) | Instruction following |
| MT-Bench | Score (1–10) | 6.5–7.5 (3B), 7.0–8.0 (7B) | Conversational quality |

### 12.2 The MMLU-Pro Effect

Small models exhibit a **unique failure mode**: they achieve competitive accuracy on MMLU-Pro (standardized multiple-choice) but collapse on open-ended reasoning:

```python
def detect_collapse(student, teacher, prompts):
    """Measure similarity collapse — SMIs tend to memorize patterns."""
    student_outs = [student.generate(p) for p in prompts]
    teacher_outs = [teacher.generate(p) for p in prompts]

    for s, t in zip(student_outs, teacher_outs):
        embedding_sim = cosine_similarity(embed(s), embed(t))
        # Below 0.5 indicates the student is not following reasoning

    return mean_embedding_similarity
```

Collapse ratio (density of student outputs in representation space) is a better indicator of SLM quality than any single benchmark. Healthy SLMs have 0.3–0.5 collapse ratio; collapsed ones exceed 0.7.

### 12.3 Efficiency Metrics

```python
def benchmark_efficiency(model, input_length=128, output_length=256):
    results = {}

    # Latency
    for batch_size in [1, 4, 8, 16, 32]:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        model.generate(inputs, max_new_tokens=output_length)
        end.record()
        torch.cuda.synchronize()

        results[f'latency_bs{batch_size}'] = start.elapsed_time(end)

    # Memory
    results['peak_memory'] = torch.cuda.max_memory_allocated()
    results['model_memory'] = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )

    # Throughput
    results['tokens_per_second'] = output_length / (results['latency_bs1'] / 1000)

    return results
```

**Target SLM metrics (3B class):**
- First token latency: <50ms (GPU), <200ms (NPU), <500ms (CPU)
- Generation rate: >40 tok/s (GPU), >15 tok/s (NPU)
- Memory: <8GB (GPU), <4GB (NPU), <2GB (on-device)
- W4A8 quantized: 2.5GB total

---

## 13. References & Papers

### Foundational Papers

| Year | Paper | Core Idea | Citations |
|------|-------|-----------|-----------|
| 2015 | **Distilling the Knowledge in a Neural Network** (Hinton et al.) | Logit-based KD | 30K+ |
| 2015 | **FitNets: Hints for Thin Deep Nets** (Romero et al.) | Feature-based KD, hint regression | 5K+ |
| 2017 | **Attention Transfer** (Zagoruyko & Komodakis) | Attention map transfer | 2K+ |
| 2018 | **Born-Again Neural Networks** (Furlanello et al.) | Same-architecture self-KD | 1.5K+ |
| 2019 | **DistilBERT** (Sanh et al.) | BERT distillation | 7K+ |
| 2019 | **TinyBERT** (Jiao et al.) | Transformer-specific 4-component KD | 2K+ |
| 2020 | **LSQ: Learned Step Size Quantization** (Esser et al.) | Learnable quantization scale | 1K+ |
| 2020 | **LayerDrop** (Fan et al.) | Stochastic depth for structured pruning | 1K+ |
| 2021 | **Deep Mutual Learning** (Zhang et al.) | Peer-teaching distillation | 2K+ |

### Modern LLM-Specific Papers

| Year | Paper | Core Idea | Impact |
|------|-------|-----------|--------|
| 2023 | **GPTQ** (Frantar et al.) | Hessian-based post-training quantization | Industry standard for LLM 4-bit |
| 2023 | **SparseGPT** (Frantar & Alistarh) | One-shot unstructured pruning | First viable >50% LLM pruning |
| 2023 | **QLoRA** (Dettmers et al.) | NF4 quantization + LoRA finetuning | Democratized LLM finetuning |
| 2024 | **AWQ** (Lin et al.) | Activation-aware weight quantization | Default in vLLM, TensorRT-LLM |
| 2024 | **SmoothQuant** (Xiao et al.) | Migrate difficulty activations→weights | Enables W8A8 without loss |
| 2024 | **Wanda** (Sun et al.) | Activation-weighted pruning | 100× faster than SparseGPT |
| 2024 | **MiniLLM** (Gu et al.) | Reverse KL for LLM distillation | SOTA LLM→SLM transfer |
| 2024 | **SPIN** (Chen et al.) | Self-play fine-tuning | Improves SLM via iterative self-distill |
| 2024 | **DARE** (Yu et al.) | Drop And REscale merging | Enables 8+ model merging |
| 2024 | **TIES-Merging** (Yadav et al.) | Trim, elect sign, merge | Task-arithmetic model merging |
| 2024 | **AQLM** (Egiazarian et al.) | Additive quantization | SOTA at 2-bit extreme compression |
| 2024 | **QuIP#** (Tseng et al.) | Lattice-based quantization | 2-bit with coherent inference |

### Distillation of Specific SLMs

| SLM | Teacher | Distillation Method | Size | Retained Quality |
|-----|---------|-------------------|------|-----------------|
| **Phi-1** (2023) | GPT-3.5 | Textbook-quality data filtering | 1.3B | 90% on HumanEval |
| **Phi-3-mini** (2024) | GPT-4 | Data-centric + logit KD | 3.8B | 95% on MMLU |
| **Phi-4** (2025) | GPT-4o | Offline + on-policy KD + QAT | 14B | 98% on MMLU-Pro |
| **Gemma 2 2B** (2024) | Gemini | Distillation + pruning | 2.6B | 92% of Gemma 7B |
| **Gemma 2 9B** (2024) | Gemini | Logit + feature KD | 9.2B | 97% of 27B |
| **Llama 3.2 1B** (2024) | Llama 3.1 8B | Pruning + logit KD | 1B | 85% on MMLU |
| **Llama 3.2 3B** (2024) | Llama 3.1 8B | Pruning + on-policy KD | 3B | 91% on MMLU |
| **SmolLM2** (2025) | Llama 3.1 70B | Offline KD + SPIN | 1.7B | 83% of Llama 3.1 8B |
| **DeepSeek-R1-Distill-Qwen-1.5B** (2025) | DeepSeek-R1 | On-policy reasoning KD | 1.5B | 85% on MATH |

---

## Appendix: Code Repositories & Tools

| Tool | Purpose | Stars (2026) |
|------|---------|-------------|
| [Textbooks Are All You Need](https://github.com/microsoft/phi-1) | Phi-1 training pipeline | 12K |
| [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) | AWQ quantization | 5K |
| [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) | GPTQ quantization | 9K |
| [Bitsandbytes](https://github.com/TimDettmers/bitsandbytes) | NF4, FP4 quantization | 15K |
| [SparseGPT](https://github.com/IST-DASLab/sparsegpt) | One-shot pruning | 3K |
| [LLM-QAT](https://github.com/facebookresearch/llm-qat) | QAT for LLMs | 2K |
| [MergeKit](https://github.com/arcee-ai/mergekit) | Model merging framework | 8K |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | GPU inference optimization | 15K |
| [ExecuTorch](https://github.com/pytorch/executorch) | On-device deployment | 8K |
| [TinyChat](https://github.com/mit-han-lab/TinyChat) | On-device LLM inference | 3K |
| [PowerInfer-2](https://github.com/SJTU-IPADS/PowerInfer-2) | Mobile LLM inference | 2K |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | CPU/GPU inference | 100K+ |
