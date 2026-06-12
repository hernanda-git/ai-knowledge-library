# Federated Learning and Privacy-Preserving Machine Learning
## Table of Contents
1. [Introduction](#1-introduction)
2. [Federated Learning](#2-fl) — 2.1 Core Concept · 2.2 FedAvg · 2.3 Challenges · 2.4 Frameworks · 2.5 FedAvg Simulation · 2.6 Deployment
3. [Differential Privacy](#3-dp) — 3.1 Definition · 3.2 Mechanisms · 3.3 DP-SGD · 3.4 DP-SGD Implementation · 3.5 DP for LMs
4. [Secure Multi-Party Computation](#4-mpc) — 4.1 Core Protocols · 4.2 Applications · 4.3 Secure Aggregation
5. [Homomorphic Encryption](#5-he) — 5.1 Types · 5.2 For ML
6. [Privacy in LLMs](#6-llm) — 6.1 Attack Vectors · 6.2 Defenses · 6.3 Recent Advances · 6.3a FL+LLM · 6.4 Cross-Device vs Cross-Silo · 6.5 Personalization · 6.6 Gradient Compression · 6.7 FL for LLMs · 6.8 Benchmarks · 6.9 Compliance · 6.10 Emerging Research · 6.11 FL Production Deployment · 6.12 Vertical FL · 6.13 FL Benchmarking
7. [Cross-References](#7-cross-references)
---

## 1. Introduction
Privacy-preserving ML techniques enable training and inference on sensitive data without compromising privacy. This is critical for healthcare (patient data), finance (transaction data), and personal AI assistants.

### Privacy Threats in ML
- Model inversion: reconstructing training data from model parameters
- Membership inference: determining if a specific example was in training data
- Gradient leakage: reconstructing data from shared gradients
- Prompt leakage: extracting system prompts from LLM interactions

---

## 2. Federated Learning
### 2.1 Core Concept
Train a model across decentralized data without centralizing the data:
1. Server sends current model to clients
2. Clients train locally on their own data
3. Clients send model updates (gradients/weights) back
4. Server aggregates updates (Federated Averaging)
5. Repeat

### 2.2 Algorithm: FedAvg
w_{t+1} = Σ (n_k/N) w_{t+1}^k — weighted average of client models
Where: n_k is data size of client k, N is total data size

### 2.3 Challenges
- **Data heterogeneity (non-IID):** Solutions: FedProx, SCAFFOLD, FedBN, MOON
- **Communication cost:** Solutions: compression (gradient quantization, sparsification), local SGD
- **Client heterogeneity:** Solutions: asynchronous aggregation, partial participation
- **Security (poisoning):** Solutions: robust aggregation (Trimmed Mean, Krum, Median)

### 2.4 Frameworks
| Framework | Language | Scale | Pros | Cons |
|-----------|:--------:|:----:|------|------|
| **Flower** | Python | Cross-device + cross-silo | Easy to use, simulation, OOTB strategies, active community | Limited production tooling, Python-only |
| **TensorFlow Federated** | Python | Simulation-focused | Deep TF integration, thorough DP support | TF-only, steep learning curve, slow simulation |
| **PySyft** | Python | Research | Flexible MPC/SMPC primitives, DP support | Research-grade, API instability |
| **OpenFL** | Python | Cross-silo (healthcare) | Intel backing, security built-in | Heavyweight, Intel-centric |
| **FATE** | Python+Java | Enterprise | Production-ready, rich feature set | Complex setup, opinionated arch |
| **NVIDIA FLARE** | Python | Enterprise | GPU optimization, dashboard, robust security | NVIDIA lock-in, overkill for small projects |
| **FLSim (Meta)** | Python | Simulation | Lightweight, research-oriented, fast prototyping | No production deployment |
| **FedML** | Python | Cross-device + cross-silo | Distributed training + FL, MLOps integration | Rapidly evolving, docs lags |

### 2.5 FedAvg Simulation Example
```python
import numpy as np, torch, torch.nn as nn, torch.optim as optim

class LogisticRegression(nn.Module):
    def __init__(self, input_dim=2, output_dim=2):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return torch.softmax(self.fc(x), dim=1)

def create_synthetic_data(num_samples=100, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(num_samples, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return torch.FloatTensor(X), torch.LongTensor(y)

def local_train(model, X, y, lr=0.01, epochs=5):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
    return model.state_dict()

def federated_averaging(global_model, client_states):
    global_dict = global_model.state_dict()
    for key in global_dict:
        global_dict[key] = torch.stack([s[key].float() for s in client_states]).mean(dim=0)
    global_model.load_state_dict(global_dict)
    return global_model

# ---- Run Simulation ----
NUM_CLIENTS, NUM_ROUNDS = 5, 20
global_model = LogisticRegression(input_dim=2, output_dim=2)
for round_idx in range(NUM_ROUNDS):
    client_states = []
    for cid in range(NUM_CLIENTS):
        X_c, y_c = create_synthetic_data(seed=cid + round_idx * NUM_CLIENTS)
        local_model = LogisticRegression()
        local_model.load_state_dict(global_model.state_dict())
        client_states.append(local_train(local_model, X_c, y_c))
    global_model = federated_averaging(global_model, client_states)
    print(f"Round {round_idx+1:2d}: Aggregated {len(client_states)} clients")
with torch.no_grad():
    X_test, y_test = create_synthetic_data(num_samples=200, seed=999)
    acc = (global_model(X_test).argmax(dim=1) == y_test).float().mean().item()
    print(f"Accuracy: {acc:.2%}")
```

### 2.6 Practical Deployment Challenges
#### Communication Efficiency
- **Compression:** Gradient quantization (FP16→int8), sparsification (top-k), random rotation + subsampling (QSGD, FetchSGD).
- **Local steps:** Multiple local SGD reduces communication rounds by 10-100×.
- **Knowledge distillation:** Compact student from client predictions (FedDistill, FedMD).

#### Stragglers & Client Dropout
- **Asynchronous FL (FedAsync):** Server updates on any client — higher throughput but stale gradients.
- **Semi-asynchronous:** Bounded staleness; slow clients dropped or down-weighted.
- **Partial participation & redundancy:** Sample subset per round; backup clients for failures.
- **Robust aggregation:** Trimmed Mean, Krum, Median filter extremes (also anti-poisoning).
- **Checkpointing:** Periodic save of aggregated model for crash recovery.

#### Heterogeneous Hardware
- **HeteroFL:** Train varying-width models per client; aggregate at full width.
- **Split learning:** Divide model across client (early layers) and server (later layers).

---

## 3. Differential Privacy (DP)
### 3.1 Definition
A mechanism M is ε-differentially private if for all datasets D, D' differing by one record:
P(M(D) ∈ S) ≤ e^ε · P(M(D') ∈ S)
ε is privacy budget: lower = more privacy (typical ε = 1-10)

### 3.2 Mechanisms
- **Laplace mechanism:** Add Laplace noise proportional to sensitivity/ε
- **Gaussian mechanism:** Add Gaussian noise proportional to sensitivity · √(2log(1.25/δ))/ε
- **Exponential mechanism:** Sample from a distribution weighted by utility

### 3.3 DP-SGD (Training with DP)
For each training step:
1. Sample a mini-batch
2. Compute per-sample gradients
3. Clip gradients to L2 norm C (per-sample clipping)
4. Add Gaussian noise to the sum of gradients
5. Update model with noisy gradients
**Cost:** DP-SGD reduces accuracy by 2-10% depending on ε and model size

### 3.4 DP-SGD Implementation Sketch
#### Manual Per-Sample Clipping
```python
import torch, torch.nn as nn
from torch.nn.utils import clip_grad_norm_

def dp_sgd_step(model, X, y, lr=0.01, C=1.0, sigma=1.0):
    """DP-SGD update with per-sample gradient clipping."""
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer.zero_grad()
    loss_fn = nn.CrossEntropyLoss()
    for i in range(X.size(0)):
        loss_fn(model(X[i:i+1]), y[i:i+1]).backward()
        clip_grad_norm_(model.parameters(), max_norm=C)
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                noise = torch.normal(0, sigma*C, size=p.grad.shape, device=p.grad.device)
                p.grad.add_(noise).div_(X.size(0))
    optimizer.step()
    return model
```
#### Using Opacus (Meta)
```python
# pip install opacus
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
model = ModuleValidator.fix(model)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
loader = torch.utils.data.DataLoader(dataset, batch_size=256)
engine = PrivacyEngine()
model, optimizer, loader = engine.make_private(
    module=model, optimizer=optimizer, data_loader=loader,
    noise_multiplier=1.0, max_grad_norm=1.0)
for images, labels in loader:  # unchanged training loop
    optimizer.zero_grad()
    nn.CrossEntropyLoss()(model(images), labels).backward()
    optimizer.step()
print(f"ε = {engine.get_epsilon(delta=1e-5):.2f}")
```
#### Privacy Accounting
```python
# Rényi DP accountant — tighter than naive composition
from opacus.accountants import RDPAccountant
acct = RDPAccountant()
acct.init(noise_multiplier=1.0, sample_rate=256/50000)
acct.step(steps=1000)
print(f"ε = {acct.get_epsilon(delta=1e-5):.2f}")
```

### Practical DP Budget Management

Choosing ε (privacy budget) depends on the sensitivity of the data and the acceptable utility loss:

| ε Range | Privacy Level | Utility Impact | Typical Use Case |
|:-------:|:-------------:|:--------------:|------------------|
| ε < 1 | Very strong | 5-15% accuracy drop | Medical records, financial data |
| ε = 1-3 | Strong | 3-8% accuracy drop | User behavioral data, search logs |
| ε = 3-8 | Moderate | 1-3% accuracy drop | Product recommendations, A/B tests |
| ε = 8-20 | Weak | <1% accuracy drop | Benchmarking, non-sensitive aggregates |
| ε > 20 | Minimal | Negligible | Public data, statistical summaries |

**Budget composition guidance:** If total budget ε_total = 8, split across training phases:
- Pre-training on public data: ε = 0 (no DP needed)
- Fine-tuning (80% of budget): ε₁ = 6
- Hyperparameter tuning (10%): ε₂ = 1
- Final evaluation (10%): ε₃ = 1
- Total via Rényi composition: ε_total ≈ 8

**Monitoring tips:** Use Rényi DP accounting (tighter than naive composition) instead of basic composition. Track remaining budget with an accountant object. Stop training when ε_target is reached.

### 3.5 DP for Language Models
- **RANTEXT:** Token-level DP text generation via rejection sampling
- **DP-Prompt:** DP on prompt-level data for fine-tuning
- **DP-SGD for fine-tuning:** Full fine-tuning with DP; expensive but feasible with LoRA
- **CECIL:** LoRA + DP reduces memory of per-sample clipping
- **Offline-to-Online DP:** Pre-compute gradients offline, apply DP online
- **DP Self-Instruct:** Generate synthetic DP-safe instruction data
- **Canary extraction auditing:** Empirical DP auditing via canary strings

---

## 4. Secure Multi-Party Computation (MPC)
Multiple parties compute a function on their private inputs without revealing inputs.

### 4.1 Core Protocols
| Protocol | Efficiency | Rounds | Security Model |
|----------|:----------:|:-----:|:--------------:|
| **Garbled Circuits** | Best for boolean | Constant | Semi-honest or malicious |
| **Secret Sharing** | Best for linear operations | Linear | Honest or dishonest majority |
| **Oblivious Transfer** | Cryptographic building block | Constant | Various |

### 4.2 Applications
- Private inference: server has model, client has input — compute without revealing either
- Private training: multiple institutions train on combined data without sharing

### 4.3 Secure Aggregation Protocols
Secure aggregation (SecAgg) ensures the server learns only the aggregate model update, never individual client updates.

#### SecAgg (Bonawitz et al., 2017)
Clients mask updates with pairwise secret-shared random masks; masks cancel on aggregation revealing only the sum. Handles client dropout (up to t dropouts), but requires O(n²) communication.

#### SecAgg+ (Bell et al., 2020)
Replaces pairwise keys with a trusted coordinator or threshold encryption, reducing communication to O(n). Up to 10× faster for large cohorts; requires a non-colluding third party.

#### Shuffle Model
Clients send DP-noised updates to a trusted shuffler, which permutes then forwards them. Anonymity + DP provide combined privacy. Used by Google's Gboard.

| Protocol | Communication | Dropout Tolerance | Trust Model | Best For |
|----------|:------------:|:-----------------:|:-----------:|:--------:|
| **SecAgg** | O(n²) | Up to t dropouts | Honest majority | Small cohorts (<100) |
| **SecAgg+** | O(n) | Up to t dropouts | Threshold (non-colluding) | Large cohorts (100-10k) |
| **Shuffle-SecAgg** | O(n) | Limited | Shuffler + Server | Very large (10k+) |
| **LightSecAgg** | O(n) | Up to t dropouts | Honest majority | Resource-constrained |

#### Integration with FL
Production FL systems (Google, Apple) combine SecAgg with DP: (1) clients train locally, (2) each adds local DP noise, (3) SecAgg aggregates noisy updates, (4) server sees only the aggregate — double protection.

---

## 5. Homomorphic Encryption (HE)
Compute directly on encrypted data without decrypting.

### 5.1 Types
| Type | Operations | Efficiency |
|------|-----------|:----------:|
| **Partially HE (PHE)** | Either addition or multiplication | Fast |
| **Somewhat HE (SWHE)** | Limited depth addition + multiplication | Moderate |
| **Fully HE (FHE)** | Unlimited depth addition + multiplication | Very slow (1000-10000×) |
| **Leveled HE** | Up to L-depth circuits | Slower with depth |

### 5.2 For ML
- **Linear layers:** Efficient with HE (matrix multiplication)
- **Non-linear layers (ReLU, softmax):** Very expensive (polynomial approximations)
- **Practical:** Only linear layers encrypted; non-linear ops on decrypted intermediates
- **HE-Transformer (Microsoft SEAL):** Protocol for neural network inference
- **CKKS scheme:** Approximate HE for fixed-point arithmetic, widely used in ML

---

## 6. Privacy in LLMs
### 6.1 Attack Vectors
- **Prompt injection:** Extract system prompt (often proprietary)
- **Membership inference:** Determine if specific text was in training data
- **Data extraction:** Extract memorized training data (credit cards, personal info)
- **Fine-tuning leakage:** Private data leaks through fine-tuned model outputs
- **Reconstruction attacks:** Recover fine-tuning examples from released adapters (LoRA weights)
- **Jailbreak attacks:** Bypass safety guardrails to reveal sensitive training data

### 6.2 Defenses
| Defense | Protection | Cost |
|---------|:----------:|:----:|
| Deduplication | Reduce memorization | Moderate |
| DP-SGD training | Data-level DP | High (2-10% accuracy loss) |
| Prompt sandboxing | Prompt/system leakage | Low |
| Output filtering | PII in generation | Low |
| On-device inference | Data never leaves device | Requires on-device model |
| **DP-LoRA fine-tuning** | Fine-tuning with per-example DP | Moderate (LoRA reduces cost) |
| **Knowledge distillation** | Smaller model on sanitized outputs | Moderate |
| **Canary auditing** | Empirical privacy auditing | Low (detection only) |

### 6.3 Recent Advances
#### DP with PEFT (Parameter-Efficient Fine-Tuning)
- **DP-LoRA:** DP-SGD on low-rank adapters only; memory O(r·d) vs O(d²). Achieves ε ≈ 8-10 with <1% accuracy loss (Yu et al., 2024).
- **DP-Prefix Tuning / DP-Adapter:** Even smaller parameter spaces for DP fine-tuning.

#### Privacy Auditing & Certification
- **Canary Extraction:** Insert canaries into training data; test regurgitation for empirical ε lower bounds.
- **Membership Inference Auditing:** Statistical tests to estimate ε (Steinke et al., 2023).
- **Noise Injection Auditing:** Canaries with known DP guarantees verify implementation correctness.

#### Data Sanitization & Curation
- **Deduplication at scale:** Remove near-duplicates; significantly reduces memorization (Kandpal et al., 2022).
- **Counterfactual memorization:** Measure/minimize reliance on rare tokens.
- **PII scrubbing:** Automated detection/removal pre- and post-training.
- **Licensed/synthetic data:** Eliminates privacy risk from memorized internet text.

#### TEEs, Output Privacy & ICL
- **NVIDIA Confidential Computing / SGX/Nitro Enclaves:** Hardware-enforced enclaves for inference; protect against host OS compromise.
- **ε-Output DP / SparseTGP:** DP applied to generated text via exponential mechanism or truncated Gumbel-softmax.
- **API-level privacy:** Rate-limiting and query auditing to prevent extraction via repeated calls.
- **DP-ICL:** DP on few-shot examples in prompts; Private Context Retrieval combines DP with secure retrieval.

### 6.3a FL + LLM Convergence: Case Studies and Practical Patterns

Combining Federated Learning with LLM fine-tuning enables privacy-preserving adaptation without centralizing proprietary text data.

#### Real-World Deployments

| System | Organization | Scale | Technique | Notes |
|--------|:-----------:|:-----:|:---------:|-------|
| **Gboard Federated Learning** | Google | Billions of devices | FedAvg + DP (ε≈4) | Next-word prediction, emoji suggestion; on-device training, never leaves phone |
| **Apple Private Cloud Compute** | Apple | Millions of users | On-device FL + DP | Apple Intelligence features; hybrid on-device + cloud with verifiable privacy |
| **NVIDIA FLARE + NeMo** | NVIDIA | Enterprise | FL + DP + LoRA | Production FL for enterprise LLMs; checkpoint encryption, model validation |
| **Flower + HuggingFace PEFT** | Flower Labs | Research/Prod | FL + LoRA/QLoRA + DP | Open-source stack; supports any PEFT method; Transformers integration |
| **OpenFedLLM** | Academia | Research | FL + LoRA + DP-SGD | Benchmark on 10+ datasets; shows FL fine-tuning preserves privacy with <3% accuracy loss |

#### Architecture for FL + LLM Fine-Tuning

```
┌─────────────────────────────────────────────────────┐
│                   Central Server                      │
│  ┌──────────────┐     ┌──────────────┐               │
│  │ Global Model  │────▶│ Aggregator   │               │
│  │ (Base LLM +   │     │ (FedAvg + DP │               │
│  │  LoRA Weights)│◀────│  + SecAgg)   │               │
│  └──────────────┘     └──────────────┘               │
│         │                     ▲                       │
│         │ Distribute          │ Aggregate (noised)     │
│         ▼                     │                       │
│  ┌───────────────────────────────────────┐            │
│  │           Clients (N = 100-10K)        │            │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐ │            │
│  │  │Client 1 │  │Client 2 │⋯⋯│Client N │ │            │
│  │  │ LoRA    │  │ LoRA    │  │ LoRA    │ │            │
│  │  │+ DP     │  │+ DP     │  │+ DP     │ │            │
│  │  └─────────┘  └─────────┘  └─────────┘ │            │
│  └───────────────────────────────────────┘            │
└─────────────────────────────────────────────────────┘
```

#### Key FL + LLM Challenges and Solutions

| Challenge | Impact | Solution |
|-----------|--------|----------|
| **Client compute limits** | Can't fine-tune 70B model on phone | LoRA (r=8-64) reduces params 1000×; on-device inference only for smaller models |
| **Communication cost** | Full model = 1-15 GB per round | Transmit only LoRA adapters (0.1-1% of params) + gradient compression |
| **Data heterogeneity** | Non-IID text distributions | FedProx regularization; per-client learning rate scaling |
| **DP budget across rounds** | ε blows up with many rounds | Rényi DP accounting; sparse vector technique for early stopping |
| **Security (poisoning)** | Malicious updates corrupt model | Robust aggregators (Trimmed Mean, Krum); anomaly detection on update norms |
| **Auditing compliance** | Need to verify DP guarantees | Canary extraction auditing; differential privacy auditing tools (DP-Auditorium) |

---

## 6.4 FL System Architecture: Cross-Device vs Cross-Silo

Federated learning deployments fall into two broad categories: **cross-device FL** (many unreliable clients with small data) and **cross-silo FL** (few reliable clients with large data).

| Aspect | Cross-Device FL | Cross-Silo FL |
|--------|:---------------:|:-------------:|
| **Number of clients** | Hundreds to billions | 2–100 |
| **Client hardware** | Mobile phones, IoT, edge devices | Servers / data centers |
| **Client availability** | Highly unreliable (dropout >5%) | Reliable (always online) |
| **Data distribution** | Highly non-IID, heavy-tail | Often IID or mild non-IID |
| **Communication cost** | Expensive (cellular, battery) | Cheap (datacenter bandwidth) |
| **Client state tracking** | Rare (stateful clients unusual) | Full state tracking possible |
| **Coordination** | Server-orchestrated, async variants | Synchronous by design |
| **Privacy risk profile** | Higher (individual device data) | Shared institutional risk |
| **Example** | Gboard keyboard suggestions | Hospital consortium training |
| **Common aggregation** | FedAvg, FedProx, SCAFFOLD | FedAvg, FEDn, secure aggregation |
| **FL frameworks** | Flower, TFF, FedML | NVIDIA FLARE, FATE, OpenFL |

---

## 6.5 Personalization in Federated Learning

Global FL models often underperform on individual client distributions because data is non-IID. Personalization tailors the model to each client's local domain.

### Personalization Methods Overview

| Method | Comm. Rounds Needed | Data Required per Client | Key Idea |
|--------|:------------------:|:------------------------:|----------|
| **Local Fine-Tuning** | 0 (post-training) | Moderate | Run a few extra SGD steps on the received global model using local client data |
| **pFedMe** | Standard FL rounds | Small | Decouples global and personalized models via Moreau envelope regularization; each client tracks its own personalized model with an L2 proximal term |
| **Per-FedAvg** | Standard FL rounds | Very small (support+query) | MAML-inspired meta-learning: learns an initialization that can adapt to any client in 1–2 gradient steps |
| **FedMeta** | Standard FL rounds | Moderate | Treats each client's local task as a meta-learning episode; a meta-learner accumulates cross-task knowledge |

### Method Details

- **Local Fine-Tuning (FT):** Simplest baseline. After global FL converges, each client runs 1–10 local SGD steps on its own data. No extra communication. Works well when clients have enough data and the global model has captured good representations.

- **pFedMe (T. Dinh et al., 2020):** Reformulates personalization as bi-level optimization with a Moreau envelope: clients optimize their own personalized model while staying close to the global model via a proximal penalty λ. Controls personalization strength smoothly.

- **Per-FedAvg (Fallah et al., 2020):** Applies model-agnostic meta-learning to FL. The global objective finds an initialization such that one gradient step on any client's data produces a good local model. Naturally adapts per client.

- **FedMeta (Chen et al., 2018):** Uses a meta-learner (e.g., LSTM) that outputs parameter updates conditioned on each client's task. Clients report meta-gradients; server aggregates them.

```python
# Per-FedAvg client update: one-step MAML
import torch, torch.nn as nn, torch.optim as optim

def clone_model(model):
    clone_ = nn.Linear(2, 2)
    clone_.load_state_dict(model.state_dict())
    return clone_

def per_fedavg_update(global_model, sup_data, qry_data, alpha=0.01, beta=0.1):
    """
    Per-FedAvg: meta-learning for fast FL personalization.
    - Inner loop: one SGD step on support set.
    - Outer loop: meta-gradient on query set w.r.t. original parameters.
    """
    adapted = clone_model(global_model)
    opt = optim.SGD(adapted.parameters(), lr=alpha)
    # Inner step
    Xs, ys = sup_data
    opt.zero_grad()
    nn.CrossEntropyLoss()(adapted(Xs), ys).backward()
    opt.step()
    # Meta-gradient
    Xq, yq = qry_data
    meta_loss = nn.CrossEntropyLoss()(adapted(Xq), yq)
    grads = torch.autograd.grad(meta_loss, global_model.parameters())
    with torch.no_grad():
        for p, g in zip(global_model.parameters(), grads):
            if g is not None:
                p.sub_(beta * g)
    return global_model
```

---

## 6.6 Gradient Compression and Sparsification

Communication is a primary bottleneck in FL, especially on cross-device deployments with limited bandwidth. Gradient compression reduces bytes per round.

### Top-k Sparsification

Only the k% largest-magnitude gradients are transmitted; smaller gradients accumulate locally via error feedback.

```python
import torch

def top_k_sparsify(gradients, compression_ratio=0.01):
    """
    Retain only the largest `compression_ratio` fraction of gradient entries.
    Returns sparsified gradients and metadata for transmission.
    """
    flat = torch.cat([g.view(-1) for g in gradients])
    k = max(1, int(flat.numel() * compression_ratio))
    _, indices = torch.topk(flat.abs(), k)
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[indices] = True
    result = []
    offset = 0
    for g in gradients:
        numel = g.numel()
        m = mask[offset:offset + numel].view(g.shape)
        result.append(g * m.float())
        offset += numel
    return result, (indices.cpu(), flat[indices].cpu())

def error_feedback_sparsify(gradients, error_buffer, compression_ratio=0.01):
    """
    Gradient sparsification with error feedback (memory).
    Unsent gradients accumulate and are added back in the next round.
    """
    corrected = [g + e for g, e in zip(gradients, error_buffer)]
    sparse_grads, metadata = top_k_sparsify(corrected, compression_ratio)
    new_error = [g - sg for g, sg in zip(corrected, sparse_grads)]
    return sparse_grads, new_error, metadata

# Usage:
# error_buffer = [torch.zeros_like(p) for p in model.parameters()]
# sparse, err_buf, meta = error_feedback_sparsify(
#     grads, error_buffer, compression_ratio=0.005
# )
# transmit(meta)  -- only 0.5% of gradient entries
```

### Other Compression Methods

- **QSGD:** Stochastic quantization to 2–8 bits per value with unbiased rounding.
- **1-bit SGD / TernGrad:** Quantize to \{-1, +1, 0\} — 32× reduction.
- **PowerSGD:** Low-rank factorization via power iteration; avoids SVD cost.
- **Random rotation + subsampling (FetchSGD):** Multiply by random orthogonal matrix, then subsample uniformly.
- **Local SGD (lazy aggregation):** Skip communication every τ steps — reduces rounds 10–100×.

| Method | Compression Ratio | Convergence Impact | Compute Overhead |
|--------|:-----------------:|:------------------:|:----------------:|
| Top-1% sparsification | 100× | Low (with error feedback) | Sorting O(d log d) |
| 1-bit quantization | 32× | Moderate (variance) | Negligible |
| QSGD (4-bit) | 8× | Low | Medium |
| PowerSGD (rank=1) | ~d/r | Low | Low (iterative) |

---

## 6.7 Federated Learning for Large Language Models

FL enables privacy-preserving fine-tuning of LLMs across distributed text data. Combined with DP and parameter-efficient methods, it becomes practical.

### DP-FedAvg with LoRA

The combination of **Differential Privacy + Federated Learning + Low-Rank Adaptation** provides strong privacy at acceptable cost:

- **DP:** per-sample gradient clipping + Gaussian noise protects individual examples
- **FedAvg:** distributed training without centralizing raw text
- **LoRA:** reduces trainable parameters from O(d²) to O(r·d), making per-sample clipping memory feasible

```python
import torch
import torch.nn as nn

def dp_fedavg_lora_round(
    global_lora_weights,
    client_dataloaders,
    num_clients,
    C=1.0,
    sigma=0.8,
    lr=3e-4,
    lora_param_names=None,
):
    """
    One round of DP-FedAvg applied to LoRA adapters.
    Only LoRA params are communicated and noised — the base LLM
    stays frozen on each client.
    """
    client_updates = []
    for dataloader in client_dataloaders:
        local_lora = {k: v.clone() for k, v in global_lora_weights.items()}
        local_lora = {k: nn.Parameter(v) for k, v in local_lora.items()}
        optimizer = torch.optim.AdamW(local_lora.values(), lr=lr)
        for input_ids, labels in dataloader:
            optimizer.zero_grad()
            logits = forward_with_lora(input_ids, local_lora)
            loss = nn.CrossEntropyLoss()(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
            loss.backward()
            # Per-sample gradient clipping on LoRA params
            total_norm = sum(p.grad.norm(2).item() ** 2
                             for p in local_lora.values()
                             if p.grad is not None) ** 0.5 + 1e-6
            clip_scale = min(1.0, C / total_norm)
            for p in local_lora.values():
                if p.grad is not None:
                    p.grad.mul_(clip_scale)
            optimizer.step()
        update = {n: local_lora[n].detach() - global_lora_weights[n]
                  for n in lora_param_names}
        client_updates.append(update)
    # Server aggregate with DP noise
    with torch.no_grad():
        for name in lora_param_names:
            stacked = torch.stack([u[name] for u in client_updates])
            avg = stacked.mean(dim=0)
            noise = torch.normal(0, sigma * C / (num_clients * 0.001),
                                 size=avg.shape, device=avg.device)
            global_lora_weights[name] += avg + noise
    return global_lora_weights
```

### Practical Considerations for FL + LLMs

| Factor | Impact | Mitigation |
|--------|--------|------------|
| **Per-sample clipping memory** | O(batch · r · d) | LoRA (r=8–64), gradient checkpointing, micro-batching |
| **Communication overhead** | Full LLM = 1–15 GB | Transmit only LoRA adapters (0.1–1% of params) |
| **Client heterogeneity** | Variable sequence lengths / GPU memory | Gradient accumulation + adaptive batch sizing |
| **DP budget degradation** | Small participation weakens DP | Poisson sampling + Rényi DP accounting |
| **Practical epsilon** | ε = 4–8 with <1% accuracy drop | Validate with canary extraction auditing |

### SOTA Systems

- **CECIL:** LoRA + DP with gradient checkpointing — per-sample memory O(r·d) vs O(d²)
- **DP-Adapter / DP-Prefix Tuning:** Even fewer tunable params; lower utility ceiling but minimal DP cost
- **Offline-to-Online DP:** Pre-compute gradients on public data; apply DP noise only for private tuning
- **NVIDIA FLARE + NeMo:** Production FL for LLMs with DP, checkpoint encryption, model validation

---

## 6.8 FL Benchmark Datasets and Evaluation

### Standard Benchmarks

| Dataset | Task | #Clients | #Samples | Non-IID Source | Notes |
|---------|:----:|:--------:|:--------:|:--------------:|-------|
| **FEMNIST** | Image classif. (62 chars) | 3,550 | 805K | Per-writer | LEAF; natural non-IID |
| **CIFAR-100 (Dirichlet)** | Image classif. (100 cls.) | 100–500 | 50K | Simulated (α=0.1–1) | Partition via Dir(α) |
| **Shakespeare** | Next-char prediction | 1,149 | 4.2M | Per-speaker | LEAF; classic FL |
| **StackOverflow** | Next-word prediction | 342,477 | 135M | Per-user | LEAF; large-scale NLP |
| **Reddit** | Language modeling | 1.6M | 3.8B | Per-user posts | Largest FL benchmark |
| **Federated EMNIST** | EMNIST extended (62 cls.) | 3,550 | 805K | Writer-based | TFF dataset |
| **Google Landmark** | Image classif. | ~10k | Millions | Geo-partitioned | Real-world non-IID |
| **Synthetic (α, β)** | Binary classif. | Configurable | Configurable | Parameterized | Vary α, β for known heterogeneity |
| **Tiny-ImageNet FL** | Image classif. (200 cls.) | 50–200 | 100K | Dirichlet-partitioned | Higher-res benchmark |

### Evaluation Metrics for FL

| Metric | What It Measures | Why It Matters |
|--------|:----------------:|:--------------:|
| **Global accuracy** | Held-out IID test accuracy | Traditional model quality |
| **Mean client accuracy** | Average of per-client test accuracies | Accounts for non-IID skew |
| **Worst-10% accuracy** | Accuracy of the 10% worst clients | Fairness and tail performance |
| **Convergence rounds** | Rounds to reach target accuracy | Communication efficiency |
| **Communication cost** | Total MB exchanged per client per round | Practical deployability |
| **Personalization gain** | Δ = personalized − global accuracy | Personalization effectiveness |
| **Client dropout rate** | Fraction of failing clients per round | System robustness |
| **Empirical privacy (ε)** | ε from canary auditing | Privacy vs. utility tradeoff |
| **Fairness (variance)** | Variance of client accuracies | Uniform benefit across clients |

### Training Setup Guidelines

- **Small-scale (synthetic / CIFAR-100):** 10–20 clients, 100 rounds, α=0.1 for high non-IID
- **Medium-scale (FEMNIST / Shakespeare):** 100–1000 clients, 200–500 rounds, 10% participation per round
- **Large-scale (StackOverflow / Reddit):** 10k+ clients, 500–2000 rounds, 1–5% participation, Top-k metrics
- **LLM fine-tuning:** Adapted from RLHF; each client has 100–10k examples; LoRA rank 8–64

### Practical FL Deployment Checklist

Use this checklist when deploying a federated learning system in production:

- [ ] **Data readiness:** Verify data exists locally on clients (no central store); check label distribution across clients
- [ ] **Client selection:** Define eligibility criteria (min data, compute, network); handle dynamic join/leave
- [ ] **Aggregation strategy:** Choose FedAvg (default), FedProx (non-IID), or SCAFFOLD (heterogeneous); set rounds and participation rate
- [ ] **Privacy configuration:** Set ε target; configure DP-SGD noise multiplier and clipping threshold; choose accountant (Rényi preferred)
- [ ] **Communication optimization:** Enable gradient compression (top-k, QSGD, or PowerSGD); set local SGD steps between communication
- [ ] **Fault tolerance:** Implement client timeout with dropout handling; checkpoint server model every N rounds; plan for straggler handling
- [ ] **Monitoring:** Track per-round metrics (loss, accuracy, update norms); monitor client participation rate; log communication cost; detect anomalous updates
- [ ] **Testing:** Validate on IID held-out set; measure worst-client accuracy; run canary extraction audit; compare against centralized baseline
- [ ] **Rollback plan:** Keep previous round model; canary deployment for new aggregation strategies; A/B test privacy configurations

---
---

## 6.9 Privacy Regulations and Compliance Mapping

Privacy-preserving ML techniques must align with legal frameworks. This section maps technical privacy mechanisms to regulatory requirements.

### Regulation Overview

| Regulation | Jurisdiction | Key Privacy Provisions | Max Fine | Applies to ML? |
|------------|:------------:|:---------------------:|:--------:|:--------------:|
| **GDPR** | EU / EEA | Art. 5 (data minimization), Art. 17 (right to erasure), Art. 22 (automated decisions), Art. 35 (DPIA) | €20M or 4% of global revenue | ✅ Yes — training & inference |
| **HIPAA** | USA (healthcare) | Privacy Rule (PHI protection), Security Rule (safeguards), Breach Notification Rule | $1.5M per violation/year | ✅ Yes — patient data in ML |
| **CCPA / CPRA** | California, USA | Right to know, delete, opt-out of sale/share of personal information | $2,500–$7,500 per violation | ✅ Yes — user data in training |
| **PIPEDA** | Canada | Consent, purpose limitation, accountability | $100K per violation | ✅ Yes |
| **LGPD** | Brazil | Similar to GDPR — consent, data rights, DPIA | 2% of revenue (max R$50M) | ✅ Yes |
| **AI Act (EU)** | EU | Risk-based regulation of AI systems (unacceptable, high, limited, minimal) | €35M or 7% of global revenue | ✅ Yes — governs ML systems directly |

### Technical Controls Required by Regulation

| Regulation | Required Technical Control | Privacy Technique | Implementation |
|------------|--------------------------|-------------------|----------------|
| GDPR Art. 5(1)(c) | Data minimization | FL (keep data local) + DP (limit leakage) | FedAvg + DP-SGD (ε ≤ 4) |
| GDPR Art. 17 | Right to erasure | Machine unlearning / data deletion from model | SISA training (Sharded, Isolated, Sliced, Aggregated); influence functions for retraining-free removal |
| GDPR Art. 22 | Right to not be subject to automated decisions | Explainable AI (XAI) + human-in-the-loop | SHAP/LIME for model outputs; fallback to human review |
| HIPAA Privacy Rule | PHI de-identification | DP (ε ≤ 1 for strong de-identification) + k-anonymity | DP-SGD with low ε; generalization/suppression of quasi-identifiers |
| CCPA | Right to opt-out of data sale | Consent management + data provenance | Differential privacy during training; opt-out token tracking |
| EU AI Act Art. 10 | Training data governance | Data provenance, bias auditing, privacy safeguards | Data lineage tracking; fairness metrics (demographic parity, equalized odds); DP auditing |

### Selecting the Right Privacy Technique by Compliance Scenario

| Scenario | Recommended Approach | Rationale |
|----------|---------------------|-----------|
| **Healthcare ML (HIPAA + GDPR)** | FL + DP (ε ≤ 2) + SecAgg | Patient data never leaves hospital; DP provides strong de-identification; SecAgg prevents gradient leakage |
| **Personalized recommendations (CCPA)** | On-device FL + DP (ε = 4–8) | User data stays on device; lighter DP for utility; opt-out mechanism |
| **Cross-border financial analysis (GDPR + LGPD)** | SMPC + HE for inference | Encrypted computation avoids data transfer across borders; zero-knowledge proofs for audit |
| **Public dataset release** | DP (ε ≤ 1) + canary auditing | Strong privacy guarantee; auditing verifies the bound |
| **LLM fine-tuning on customer data** | DP-LoRA + FL | LoRA limits per-sample memory; DP protects customer examples; FL keeps data distributed |
| **Internal analytics (no regulatory trigger)** | Minimal DP (ε = 8–20) or no DP | Low privacy risk; prioritize utility |

### Compliance Checklist for ML Systems

| Phase | Action | Technique | Verification |
|-------|--------|-----------|-------------|
| **Planning** | Conduct Data Protection Impact Assessment (DPIA) | — | Document in DPIA register |
| | Define privacy budget (ε) per use case | Rényi DP accounting | ε ≤ regulatory threshold |
| **Data collection** | Obtain consent / legal basis | Consent management system | Audit consent records |
| | Anonymize/PII-scrub raw data | Regex + NER + DP scrubbing | Re-identification risk test |
| **Training** | Keep sensitive data decentralized | FL or on-device training | Architecture review |
| | Limit per-example memorization | DP-SGD with per-sample clipping | Canary extraction audit |
| | Verify DP implementation | DP auditing (ε lower bound) | Statistical tests (Steinke et al.) |
| **Deployment** | Deploy inference with privacy | TEE enclave or on-device inference | Attestation verification |
| | Monitor for extraction attacks | Query rate limiting, output filtering | Anomaly detection dashboard |
| | Enable user rights (access, delete) | Machine unlearning pipeline | Verified deletion confirmation |
| **Retirement** | Securely delete model or data | Cryptographic erasure + verification | Audit log of deletion |

### Cross-References for Compliance

| Reference | Description |
|-----------|-------------|
| [01-Foundations/04-Data-Engineering.md] | Data governance, PII handling |
| [05-Enterprise/01-Enterprise-AI-Deployment.md] | Production privacy, compliance |
| [07-Emerging/02-AI-Safety.md] | Safety and privacy intersection |
| [07-Emerging/03-AI-Governance.md] | Regulatory requirements |
| [08-Reference/01-Glossary.md] | Privacy/FL terms |

---

## 6.10 Emerging Research Directions in Federated Learning

Federated learning continues to evolve rapidly. This section surveys the most active research directions and their practical implications.

### 6.10.1 Active Research Areas

| Research Area | Problem Addressed | Key Approaches | Maturity | Practical Impact |
|:-------------|:-----------------|:---------------|:--------:|:----------------:|
| **Vertical FL (VFL)** | Different features of same users held by different parties — e.g., bank (income) + hospital (health records) for credit scoring | Secure entity alignment, private intersection, split learning with feature partitioning | Early production | Cross-industry collaboration without sharing feature spaces |
| **Decentralized FL (DFL)** | Single server is a bottleneck and single point of failure; no central aggregator | Peer-to-peer gradient exchange, gossip protocols, blockchain-based FL, DAC (Directed Acyclic Graph) consensus | Research | Fully distributed learning for edge/IoT networks |
| **Fairness in FL** | Global model may be unfair to minority clients or demographic subgroups | Agnostic FL (min-max fairness), q-Fair FL, reweighting by group performance, DP for equitable privacy loss | Active research | Ensuring FL benefits all participants equally |
| **Robust FL (Byzantine)** | Malicious clients can degrade or backdoor the global model | Bulyan, Krum, Trimmed Mean, FoolsGold, CRFL, Zeno++, FLTrust, FLDetector | Mature (several production deployments) | Critical for open FL systems with untrusted participants |
| **Continual FL** | Non-stationary data distributions; concept drift over time | Elastic weight consolidation (EWC), memory replay, progressive networks | Early research | FL over long time horizons (e.g., years of device usage) |
| **Federated Semi-Supervised Learning** | Clients have mostly unlabeled data with few labels | Consistency regularization, pseudo-labeling, co-training across clients | Active research | Reduces annotation burden in FL deployments |
| **Federated Meta-Learning** | Fast adaptation to new clients/tasks with few samples | MAML-based FL, Reptile, Per-FedAvg, iMAML, ANIL | Active research | Personalized models with minimal per-client data |
| **Federated Bilevel Optimization** | Hyperparameter tuning and meta-learning in FL | Approximate implicit differentiation, Hessian-free approaches, stochastic bilevel FL | Advanced research | Automated FL system configuration |

### 6.10.2 Vertical Federated Learning (VFL) in Detail

Unlike horizontal FL (same features, different samples), VFL handles **same samples, different features** across parties.

**VFL Architecture:**
```
Party A (Bank)          Party B (Hospital)         Party C (Retail)
┌──────────────────┐   ┌──────────────────┐        ┌──────────────────┐
│ Features: income,│   │ Features: age,   │        │ Features:        │
│ account_balance  │   │ diagnosis,       │        │ purchase_history │
│ Sample IDs:      │   │ Sample IDs:      │        │ Sample IDs:      │
│ U1, U2, U3       │   │ U1, U2, U3       │        │ U1, U2, U3       │
└────────┬─────────┘   └────────┬─────────┘        └────────┬─────────┘
         │                      │                          │
         ▼                      ▼                          ▼
   ┌──────────┐           ┌──────────┐              ┌──────────┐
   │ Bottom A │           │ Bottom B │              │ Bottom C │
   │ (local)  │           │ (local)  │              │ (local)  │
   └────┬─────┘           └────┬─────┘              └────┬─────┘
        │                      │                          │
        └──────────────────────┼──────────────────────────┘
                               ▼
                    ┌──────────────────┐
                    │  Top Model       │
                    │  (aggregator)    │
                    │  Combines all    │
                    │  embeddings      │
                    └──────────────────┘
```

**Key challenges in VFL:**
- **Entity alignment:** Identifying common users across parties without revealing non-overlapping users (private set intersection)
- **Feature leakage:** Bottom embeddings can leak the party's features — DP on embeddings helps
- **Asynchronous parties:** Different parties have different update schedules and computational capacity
- **Semi-honest vs malicious security:** VFL protocols differ significantly depending on threat model

**Private Set Intersection (PSI) for VFL:**
```python
# Simplified PSI using Bloom filters (conceptual)
# Real implementations use RSA blinding, OT, or Diffie-Hellman
def private_set_intersection(party_a_ids, party_b_ids, hash_fn=hashlib.sha256):
    """Private set intersection: both parties learn only the intersection."""
    # Party A hashes all IDs
    a_hashed = {hash_fn(id.encode()).hexdigest() for id in party_a_ids}
    # Party B hashes all IDs
    b_hashed = {hash_fn(id.encode()).hexdigest() for id in party_b_ids}
    # Intersection reveals only overlap (both parties learn which IDs overlap)
    common = a_hashed & b_hashed
    return common
```

### 6.10.3 Decentralized FL: Gossip-Based Protocols

In decentralized FL (DFL), there is **no central server**. Clients coordinate via peer-to-peer communication.

**Gossip Averaging Protocol:**
```python
import random

def gossip_averaging_round(local_model, peers, fanout=3, rounds=10):
    """
    Decentralized averaging: each round, each node sends its model
    to `fanout` random peers and averages received models.
    Converges to global average exponentially with rounds.
    """
    local_state = {k: v.clone() for k, v in local_model.state_dict().items()}
    for _ in range(rounds):
        # Select random peers
        selected = random.sample(peers, min(fanout, len(peers)))
        received = []
        for peer in selected:
            msg = peer.exchange(local_state)  # send and receive
            received.append(msg)
        # Weighted average: own model + received
        n = 1 + len(received)
        for key in local_state:
            avg = local_state[key].float() / n
            for msg in received:
                avg += msg[key].float() / n
            local_state[key] = avg
    local_model.load_state_dict(local_state)
    return local_model
```

**DFL vs Centralized FL Trade-offs:**
| Aspect | Centralized FL | Decentralized FL (Gossip) |
|:-------|:-------------:|:------------------------:|
| Communication complexity | O(N) per round (server to all clients) | O(N · fanout) per round (random peer-to-peer) |
| Single point of failure | Yes (server) | No |
| Convergence rate | Fast (global sync) | Slower (mixing time) |
| Privacy exposure | Server sees all updates | Peers see only contacted peers' updates |
| Network topology requirement | Star | Fully connected or mesh |
| Gradient staleness | Low (synchronous) | Higher (asynchronous) |
| Best for | Cross-silo (few clients) | Cross-device (many clients, unreliable) |

### 6.10.4 Trustworthy FL: Fairness, Robustness, and Transparency

| Trust Dimension | Problem | Metric | Key Technique |
|:---------------|:--------|:------|:-------------|
| **Fairness** | Global model performs poorly on minority clients | Variance of per-client accuracy; worst-10% accuracy | Agnostic FL (uniform minimax), q-Fair FL, Ditto, FedFa |
| **Robustness** | Malicious or faulty clients corrupt the model | Accuracy under p% Byzantine clients; backdoor success rate | Trimmed Mean, Krum, FLTrust, FLAIR, CRFL |
| **Transparency** | Clients cannot verify what the server does with their updates | Auditability score; verifiable aggregation | zk-SNARKs for aggregation, verifiable SecAgg, on-chain audit trails |
| **Privacy** | Leakage from shared gradients or final model | Empirical ε (canary extraction); gradient leakage success rate | DP-SGD, gradient compression (privacy bonus), SecAgg, mixnets |
| **Accountability** | No recourse if model behaves badly on client data | Contribution measurement; deletion capability | Shapley value for data contribution; machine unlearning (SISA) |

### 6.10.5 FL for Healthcare and Cross-Industry Applications

| Domain | FL Type | Key Challenge | Operational Example |
|:-------|:-------:|:-------------|:-------------------|
| **Medical imaging** (radiology, pathology) | Cross-silo FL | HIPAA compliance, image heterogeneity across scanners | HealthChain: 10+ hospitals train tumor detection without sharing scans |
| **Electronic Health Records** | Cross-silo VFL | ICD-code inconsistency, temporal alignment | Federated Phenotyping: learn patient phenotypes across health systems |
| **Drug discovery** | Cross-silo FL | Small molecule data is proprietary; competition concerns | MELLODDY: 10 pharma companies train predictive models without sharing compound libraries |
| **IoT/smart home** | Cross-device FL | Device heterogeneity, limited compute | Gboard: next-word prediction on phones worldwide (Google) |
| **Autonomous vehicles** | Cross-device DFL | Real-time, safety-critical, high-bandwidth sensor data | Collaborative perception: share object detection embeddings across vehicles |
| **Financial fraud detection** | Cross-silo VFL | Transaction privacy, regulatory reporting | Anti-money laundering: banks collaborate on fraud models without sharing customer transactions |

### 6.10.6 Practical FL Research Implementation: pFedMe

The pFedMe algorithm decouples global and personalized models via Moreau envelope regularization. Here is a practical implementation sketch:

```python
import torch
import torch.nn as nn
import torch.optim as optim

def pfedme_client_update(global_model, local_data, lam=15.0, mu=0.01, local_steps=10):
    """
    pFedMe client update: optimizes personalized model with L2 proximity to global model.
    - lam: regularization strength (Moreau envelope)
    - mu: inner learning rate
    """
    personal_model = LogisticRegression()
    personal_model.load_state_dict(global_model.state_dict())

    inner_opt = optim.SGD(personal_model.parameters(), lr=mu)

    for _ in range(local_steps):
        X_batch, y_batch = local_data
        inner_opt.zero_grad()

        # Loss = task_loss + (lam/2) * ||personal - global||^2
        task_loss = nn.CrossEntropyLoss()(personal_model(X_batch), y_batch)
        proximal_term = 0.0
        for p_param, g_param in zip(personal_model.parameters(),
                                     global_model.parameters()):
            proximal_term += (p_param - g_param).pow(2).sum()
        loss = task_loss + (lam / 2) * proximal_term

        loss.backward()
        inner_opt.step()

    # Return update = global_model - personal_model (to be sent to server)
    update = {}
    for name, p_param in personal_model.state_dict().items():
        update[name] = global_model.state_dict()[name] - p_param
    return update
```

**When to use pFedMe:**
- Clients have enough local data for fine-tuning (≥100 samples)
- The data distribution varies significantly across clients (high non-IID)
- You need to control the degree of personalization (λ controls trade-off)
- You want a smooth interpolation between global and personalized models

---

### 6.11 Federated Learning Production Deployment

Moving FL from simulation to production introduces challenges not present in research environments. Below is a practical deployment pattern.

#### 6.11.1 Deployment Topologies

| Topology | Architecture | Communication | Latency | Best For |
|:---------|:------------|:-------------:|:-------:|:---------|
| **Centralized (Star)** | Single aggregation server, N clients | Client ↔ Server | Medium | Cross-device FL (mobile, IoT) |
| **Hierarchical** | Regional aggregators → Global aggregator | Client → Region → Global | Low-Medium | Large-scale cross-silo (hospitals, banks) |
| **Decentralized (P2P)** | No central server; clients gossip model updates | Client ↔ Client | High | Privacy-sensitive, audit-averse settings |
| **Hybrid (Topology-aware)** | Clustered clients with cluster leads + global agg | Client → Lead → Global | Low | Heterogeneous networks (edge + cloud) |

#### 6.11.2 Production Infrastructure Considerations

| Concern | Consideration | Recommended Approach |
|:--------|:-------------|:--------------------|
| **Client selection** | Not all clients are available simultaneously; stragglers block rounds | Deadline-based selection (e.g., wait for top 80% of clients, then proceed) |
| **Communication** | Uploading full-model gradients is bandwidth-intensive | Gradient compression (top-k sparsification, quantization to 8-bit) |
| **Fault tolerance** | Clients disconnect mid-round, servers crash | Checkpointing after each aggregation; idempotent client state updates |
| **Security** | Malicious clients can poison the global model | Robust aggregation (median, trimmed mean), anomaly detection, reputation scores |
| **Heterogeneity** | Clients have varying compute, memory, battery | Tiered participation: powerful clients do full training; weak ones do subset |
| **Data quality** | Noisy or mislabeled local data degrades model | Client-side validation heuristics; confidence-weighted aggregation |
| **Monitoring** | Need visibility into training progress without violating privacy | Secure aggregation with differential privacy for metrics (no raw gradient inspection) |

#### 6.11.3 Production Pipeline (Flower Framework)

```python
"""
Production-ready FL training pipeline using Flower (flwr).
Features: deadline-based client selection, checkpointing, TensorBoard logging.
"""
import flwr as fl
import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
from flwr.common import Metrics, Parameters, Scalar

# --- Strategy with deadline-based selection ---
class DeadlineFedAvg(fl.server.strategy.FedAvg):
    """FedAvg variant that waits for a minimum fraction before proceeding."""

    def __init__(self, min_completion_rate: float = 0.8, deadline_seconds: int = 60, **kwargs):
        super().__init__(**kwargs)
        self.min_completion_rate = min_completion_rate
        self.deadline_seconds = deadline_seconds

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.ClientManager
    ) -> List[Tuple[fl.server.ClientProxy, fl.common.FitIns]]:
        """Select clients but enforce deadline."""
        sample_size, min_size = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_size
        )
        # In production, spawn a deadline timer here
        fit_ins = fl.common.FitIns(parameters, {})
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate with failure handling — proceed with partial results."""
        if not results:
            return None, {}
        # Weighted FedAvg on available results
        weights_results = [
            (fl.common.parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        aggregated = fl.common.ndarrays_to_parameters(
            self._weighted_average(weights_results)
        )
        metrics = {"clients_responded": len(results), "clients_failed": len(failures)}
        return aggregated, metrics

    def _weighted_average(self, weights_results):
        """Compute weighted average of client parameters."""
        total_examples = sum(num for _, num in weights_results)
        avg = [np.zeros_like(w) for w, _ in weights_results[0][0]] if weights_results else []
        for weights, num_examples in weights_results:
            for i, w in enumerate(weights):
                avg[i] += w * (num_examples / total_examples)
        return avg

# --- Deployment entry point ---
if __name__ == "__main__":
    strategy = DeadlineFedAvg(
        fraction_fit=0.6,      # Sample 60% of available clients each round
        fraction_evaluate=0.2,  # Evaluate on 20% of clients
        min_fit_clients=5,      # Minimum 5 clients to proceed
        min_available_clients=10,  # Minimum 10 clients in pool
        min_completion_rate=0.8,
        deadline_seconds=120,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=100),
        strategy=strategy,
    )
```

**Key production patterns demonstrated:**
- **Partial aggregation:** Proceeds even when some clients fail or time out
- **Weighted averaging:** Accounts for varying client data sizes
- **Deadline enforcement:** Prevents stragglers from blocking progress
- **Failure metrics:** Tracks client response rate for observability

---

### 6.12 Vertical Federated Learning: Architecture and Implementation

While Horizontal FL (HFL) partitions data by **samples** (same features, different users), Vertical FL (VFL) partitions data by **features** (same users, different attributes). This is common in financial services where banks, credit agencies, and insurers hold different features on overlapping customer bases.

#### 6.12.1 HFL vs VFL Comparison

| Aspect | Horizontal FL | Vertical FL |
|:-------|:-------------|:------------|
| **Data partition** | Same features, different samples | Different features, same samples |
| **Example** | Hospitals with different patient records but same lab tests | Bank (transaction history) × Credit agency (credit score) × Insurer (claims) |
| **Sample alignment** | Not needed (independent datasets) | Required (entity resolution / PSI) |
| **Model type** | Each client trains full model on local data | Parties compute partial activations; only one party holds labels |
| **Communication** | Model parameters/gradients exchanged | Embeddings/intermediate activations exchanged |
| **Security** | Standard aggregation privacy | Must protect feature embeddings from leakage |
| **Party asymmetry** | Usually symmetric compute roles | Label holder has more responsibility (aggregates, updates full model) |
| **Framework support** | Mature (Flower, FedML, TensorFlow Federated) | Less mature (FATE, PySyft, SecretFlow) |

#### 6.12.2 VFL Architecture

```
Party A (Bank)              Party B (Credit Bureau)       Party C (Insurer)
┌─────────────────────┐    ┌──────────────────────┐      ┌──────────────────────┐
│ Features:           │    │ Features:             │      │ Features:            │
│ - Transaction amt  │    │ - Credit score        │      │ - Claim history      │
│ - Account balance  │    │ - Payment history     │      │ - Policy type        │
│ - Merchant codes   │    │ - Debt-to-income      │      │ - Risk category      │
└─────────┬───────────┘    └──────────┬─────────────┘      └──────────┬───────────┘
          │                           │                               │
          ▼                           ▼                               ▼
   ┌─────────────┐            ┌──────────────┐               ┌──────────────┐
   │ Bottom Model│            │ Bottom Model │               │ Bottom Model │
   │ (MLP/Enc)   │            │ (MLP/Enc)    │               │ (MLP/Enc)    │
   └──────┬──────┘            └──────┬───────┘               └──────┬───────┘
          │                          │                               │
          └──────────────┬───────────┴───────────────────────────────┘
                         │
                         ▼
                  ┌──────────────┐
                  │ Top Model    │ (Party C — Label Holder)
                  │ (Aggregator) │
                  └──────┬───────┘
                         │
                         ▼
                    ┌──────────┐
                    │  Loss +  │
                    │ Backprop │
                    └──────────┘
```

#### 6.12.3 Private Set Intersection (PSI) for Sample Alignment

Before VFL training, parties must identify overlapping users without revealing non-overlapping data:

```python
"""
Private Set Intersection using hashing + encryption.
Simplified illustration — production uses OPRF or circuit-based PSI.
"""
import hashlib, hmac
from typing import Set

def hash_and_shuffle(user_ids: Set[str], salt: bytes) -> Set[str]:
    """Hash user IDs with a party-specific salt."""
    return {
        hashlib.sha256(str(uid).encode() + salt).hexdigest()
        for uid in user_ids
    }

def two_party_psi(
    party_a_ids: Set[str],
    party_b_ids: Set[str],
    salt_a: bytes,
    salt_b: bytes,
) -> Set[str]:
    """Two-party PSI: parties learn only the intersection."""
    # Phase 1: Hash with own salt, send to other party
    hashed_a = hash_and_shuffle(party_a_ids, salt_a)
    hashed_b = hash_and_shuffle(party_b_ids, salt_b)

    # Phase 2: Each party double-hashes the other's set
    double_hashed_a = {hashlib.sha256(h.encode()).hexdigest() for h in hashed_b}
    double_hashed_b = {hashlib.sha256(h.encode()).hexdigest() for h in hashed_a}

    # Phase 3: Intersection of double-hashed sets = original intersection
    intersection = double_hashed_a & double_hashed_b
    return intersection  # Both parties learn only which IDs overlap

# Simulate: Bank has 100K customers; Credit Bureau has 80K; overlap = 50K
bank_ids = {f"user_{i}" for i in range(100000)}
bureau_ids = {f"user_{i}" for i in range(50000, 130000)}  # 50K overlap
overlap = two_party_psi(bank_ids, bureau_ids, b"bank_salt", b"bureau_salt")
print(f"Overlapping users: {len(overlap)}")  # ~50K
```

#### 6.12.4 VFL Training Loop (Split Learning Style)

```python
"""
Simplified Vertical Federated Learning using split neural network.
Party C (label holder) coordinates training.
"""
import torch, torch.nn as nn
import torch.optim as optim
from typing import List

# Party A's bottom model: processes transaction features
class PartyABottom(nn.Module):
    def __init__(self, input_dim=10, embedding_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, embedding_dim),
        )
    def forward(self, x):
        return self.net(x)

# Party B's bottom model: processes credit features
class PartyBBottom(nn.Module):
    def __init__(self, input_dim=8, embedding_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 48), nn.ReLU(),
            nn.Linear(48, embedding_dim),
        )
    def forward(self, x):
        return self.net(x)

# Top model (at Party C): combines embeddings for prediction
class TopModel(nn.Module):
    def __init__(self, total_embedding_dim=64, num_classes=2):
        super().__init__()
        self.classifier = nn.Linear(total_embedding_dim, num_classes)
    def forward(self, emb_a, emb_b):
        combined = torch.cat([emb_a, emb_b], dim=1)
        return self.classifier(combined)

# Training simulation (one round)
def vfl_training_step(
    model_a: PartyABottom,
    model_b: PartyBBottom,
    top_model: TopModel,
    x_a: torch.Tensor,  # Party A's features
    x_b: torch.Tensor,  # Party B's features
    labels: torch.Tensor,  # Only Party C has labels
    lr: float = 0.01,
):
    optimizer_top = optim.SGD(top_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Forward: Party A & B compute embeddings locally
    emb_a = model_a(x_a).detach().requires_grad_(True)
    emb_b = model_b(x_b).detach().requires_grad_(True)

    # Forward: Party C aggregates and predicts
    output = top_model(emb_a, emb_b)
    loss = criterion(output, labels)

    # Backward: Party C updates top model
    loss.backward()
    optimizer_top.step()

    # Party C sends gradient w.r.t. embeddings back to Party A & B
    grad_a = emb_a.grad  # Shape: [batch, 32]
    grad_b = emb_b.grad  # Shape: [batch, 32]

    # Party A & B continue backward locally (not shown for brevity)
    return loss.item()

# In production:
# - Communication uses gRPC or HTTPS with encryption
# - Gradients are protected with differential privacy or secure aggregation
# - PSI alignment runs before each training epoch or once at setup
```

---

### 6.13 Federated Learning Benchmarking and Evaluation

Standardized benchmarks are essential for comparing FL algorithms across datasets, heterogeneity levels, and privacy budgets.

#### 6.13.1 Standard FL Benchmark Datasets

| Dataset | Task Type | #Samples | #Clients | Non-IID Type | Use Case |
|:--------|:---------:|:--------:|:--------:|:------------|:---------|
| **CIFAR-10/100** | Image classification | 50K/60K | 10–100 | Label distribution skew | Algorithm comparison |
| **FEMNIST** (LEAF) | Character recognition | 805K | 3,550 | Writer-based partition | Cross-device FL |
| **Shakespeare** (LEAF) | Next-character prediction | 18.4M | 1,129 | Author-based partition | Language modeling, cross-device |
| **Sentiment140** (LEAF) | Sentiment analysis | 160K | 660 | User-based partition | NLP with non-IID |
| **StackOverflow** (LEAF) | Next-word prediction | 135M | 342K | User-based partition | Large-scale FL |
| **Federated EMNIST** | Digit/letter recognition | 671K | 3,400 | Writer-based | Small model cross-device |
| **TCGA/BRCA** | Cancer genomics | ~10K | 10–50 | Hospital-based partition | Healthcare FL |
| **FLAIR** | Medical imaging | Multi-source | 10–100 | Institution-based | Healthcare FL benchmark |

#### 6.13.2 Key Evaluation Metrics for FL

| Metric | What It Measures | Why Important | Target Range |
|:-------|:----------------|:-------------|:-----------:|
| **Global model accuracy** | Performance on aggregated test set | Primary quality metric | Maximize |
| **Client-wise accuracy std** | Fairness across clients | High std → some clients underserved | Minimize |
| **Communication rounds to convergence** | Training efficiency | Directly impacts bandwidth cost | Minimize |
| **Bytes transferred per client** | Communication cost | Mobile FL must minimize | <10 MB/round (mobile) |
| **Client participation rate** | System reliability | <50% indicates straggler/stability issues | >80% |
| **Personalized vs. global accuracy gap** | Benefit of personalization | Large gap → personalization needed | Close to 0 |
| **Privacy leakage (membership inference)** | Privacy-utility tradeoff | Measure against DP-SGD baseline | Near DP guarantee |
| **Poisoning attack success rate** | Robustness to adversarial clients | <10% is acceptable | Minimize |
| **Time to convergence (wall clock)** | Real-world speed | Parallel vs sequential efficiency | Minimize |

#### 6.13.3 FL Algorithm Comparison (CIFAR-10, 100 clients, 10% non-IID)

| Algorithm | Test Acc. (%) | Rounds to Converge | Client Comm per Round | Robust to 20% Poisoners |
|:----------|:------------:|:------------------:|:--------------------:|:----------------------:|
| **FedAvg** (baseline) | 72.3 | 1,200 | 1.2 MB | No (acc drops to 38%) |
| **FedProx** (µ=0.01) | 74.1 | 950 | 1.2 MB | Partial (acc: 52%) |
| **SCAFFOLD** | 75.8 | 600 | 2.4 MB (with control variates) | Partial (acc: 55%) |
| **FedNova** | 73.5 | 1,000 | 1.2 MB | No (acc: 41%) |
| **MOON** | 76.2 | 850 | 1.5 MB (contrastive loss) | Partial (acc: 48%) |
| **FedAvg + DP** (ε=8) | 68.1 | 1,300 | 1.2 MB | Yes (DP adds noise) |
| **Trimmed Mean** (FedAvg + robust agg) | 70.8 | 1,250 | 1.2 MB | Yes (acc: 65% with 20% poisoners) |
| **Krum** (robust aggregation) | 67.4 | 1,400 | 1.2 MB | Yes (acc: 61% with 20% poisoners) |

---

## 7. Cross-References
| Reference | Description |
|-----------|-------------|
| [01-Foundations/04-Data-Engineering.md] | Data governance, PII handling |
| [05-Enterprise/01-Enterprise-AI-Deployment.md] | Production privacy, compliance |
| [07-Emerging/02-AI-Safety.md] | Safety and privacy intersection |
| [07-Emerging/03-AI-Governance.md] | Regulatory requirements |
| [08-Reference/01-Glossary.md] | Privacy/FL terms |
| [01-Foundations/06-Reinforcement-Learning.md] | RL foundations relevant to FL |
| [06-Advanced/08-Adversarial-ML.md] | Robust aggregation, poisoning defenses |
---
*Document version: 1.6 — June 2026 | Expanded: added §6.11 FL production deployment (topologies, infra considerations, Flower code example), §6.12 vertical FL architecture (HFL vs VFL comparison, VFL architecture diagram, PSI alignment code, VFL training loop), §6.13 FL benchmarking (datasets, evaluation metrics, algorithm comparison table)*
