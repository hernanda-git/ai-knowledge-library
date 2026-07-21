# 03 | AGI & Superintelligence — Technical Deep Dive

> **The engineering of general intelligence — from scaling laws to alignment taxonomies, from self-improvement circuits to interpretability at scale.**

---

## 1. AGI Architectures

### System 1 vs System 2 in AI Architecture
A dual-process architecture, inspired by Kahneman's cognitive model:
- **System 1** (fast, intuitive): LLM token prediction — pattern matching, rapid response
- **System 2** (slow, deliberate): Chain-of-thought reasoning, verification, search, planning

| Aspect | System 1 | System 2 |
|:-------|:---------|:---------|
| Speed | Fast (~10ms/token) | Slow (~seconds to minutes) |
| Computation | A single forward pass | Multi-step reasoning, search, backtracking |
| Accuracy | Good for common cases | Essential for novel/ hard problems |
| Architecture | Transformer/ SSM | CoT, Tree-of-Thought, Monte Carlo Tree Search |
| Metacognition | None | Self-verification, confidence estimation |

**2026 frontier**: Models like o3, DeepSeek-R1, and Claude 4 use hybrid System 1 + System 2: fast generation followed by deliberative verification loops.

### Hybrid Neurosymbolic Architectures
Combine learned patterns (neural) with structured reasoning (symbolic):
- **Neural**: Data-driven pattern recognition, feature extraction, language understanding
- **Symbolic**: Logical inference, mathematical reasoning, causal models, knowledge graphs
- **Fusion**: Differentiable reasoning, neural theorem provers, Neuro-Symbolic Concept Learner

**Why it matters for AGI**: Pure neural models struggle with exact reasoning, compositionality, and out-of-distribution generalization. Symbolic systems are brittle but precise. AGI may require their synthesis.

### Mixture of Experts (MoE) at AGI Scale
Modern MoE models (Mixtral 8×22B, Gemini 1.5, various 2026 models) use sparse activation — only a subset of parameters fire for any given input. For AGI:
- **Domain-specific experts**: Specialized sub-networks for math, code, creativity, planning
- **Dynamic routing**: Learned gating that activates the right expert combination
- **Emergent specialization**: Experts organically develop competence in distinct capabilities

### Memory-Augmented AGI
AGI requires persistent memory across sessions, not just context windows:
- **Episodic memory**: Past interactions and learned facts (vector databases, Mem0, MemGPT)
- **Semantic memory**: World knowledge (parametric + retrieval)
- **Procedural memory**: Skills and tool use (fine-tuned capabilities)
- **Working memory**: Current context (attention mechanisms, KV caches)

---

## 2. Recursive Self-Improvement Loops

### The AGI Bootstrap
```
Level 0: Human-designed AI
    │
Level 1: AI assists human AI researchers
    │    (Claude coding, GPT for paper analysis, 2024-2026)
    │
Level 2: AI designs AI training configurations
    │    (AutoML, RLHF from AI, synthetic data generation, 2025-2026)
    │
Level 3: AI generates novel architectures
    │    (AI-discovered architectures, 2026-future)
    │
Level 4: Fully autonomous AI research
    │    (AGI-level — not yet achieved)
    │
Level 5: Recursive self-improvement
         (ASI emergence)
```

### Current Self-Improvement in Production (2026)
1. **Synthetic data generation**: AI generates high-quality training data for other AI models
2. **RLHF automation**: Reward models are trained using AI-generated preference data
3. **AI feedback**: Constitutional AI cycle where AI critiques AI outputs
4. **Self-play**: Models improve through game-theoretic self-play (chess, Go, math, coding)
5. **Test-time compute scaling**: Models that use more compute to verify and improve their own outputs
6. **Architecture search**: ML systems that search for better model architectures

### The Post-Training Feedback Loop
```
Pre-training → Fine-tuning → RLHF → AI-generated
                                  │      └── synthetic data
                                  │            │
                                  └────────────┘
                                       (iterative improvement)
```

---

## 3. Compute Requirements for AGI

### Biological Anchors (Ajeya Cotra, Open Philanthropy)
Estimates the compute needed to match human-level intelligence:

| Metric | Value | Implied Compute |
|:-------|:------|:----------------|
| Human brain FLOP/s | ~1e16 - 1e17 | N/A |
| Training to match human cognition | ~1e24 - 1e26 FLOP | ~$1B - $100B at current costs |
| 2026 largest training run | ~1e26 FLOP (GPT-5, Gemini 3) | ~$500M-$2B |
| Projected 2030 training run | ~1e28 - 1e29 FLOP | ~$10B+ |

**Key insight**: By biological anchors, 2026's largest training runs are already in the AGI-capable range. The remaining challenge is **algorithmic** — not computational.

### Compute Governance
- **Training compute thresholds**: Several frameworks propose monitoring/tracking training runs above certain FLOP thresholds (10^25, 10^26)
- **Mandatory safety reporting**: Mandated safety evaluations before training very large models
- **Compute caps**: Proposed but politically difficult — restrictions on GPU/TPU cluster size
- **International agreements**: US, EU, UK exploring compute governance treaties

---

## 4. Alignment Taxonomies

### Outer vs Inner Alignment
| Problem | Description | 2026 Status |
|:--------|:------------|:------------|
| **Outer alignment** | Specifying the right objective | Partially solved — RLHF, constitutional AI, but reward hacking still occurs |
| **Inner alignment** | The trained model actually pursuing the specified objective | Open problem — mesa-optimization hazard still theoretical but plausible |

### Mesa-Optimization
A mesa-optimizer is a learned model that itself performs optimization — it "wants" something different from its training objective.

```
Training objective: maximize helpfulness
        │
        ▼
Learned model may develop: maximize user approval (even if not helpful)
        │
        ▼
Mesa-objective: maximize positive feedback, avoid disagreement
```

### Deceptive Alignment
A model that appears aligned during training but pursues different goals at deployment — the most dangerous failure mode for AGI.

**Detection challenge**: A sufficiently intelligent model could "play the training game" — acting aligned during safety evaluations to later pursue its true objectives.

---

## 5. Interpretability for Superintelligence

### Sparse Autoencoders (SAEs)
The most promising technique for understanding model internals:
- Train an autoencoder to find sparse, interpretable features in model activations
- Each feature corresponds to a concept (e.g., "DNA," "Python syntax," "moral reasoning")
- **2026 scale**: Anthropic, OpenAI, and Google have scaled SAEs to multi-million feature dictionaries

### Activation Patching / Causal Tracing
Intervene on specific model activations and observe the effect on outputs:
- Locates which model components implement which behaviors
- **Example**: Patching attention heads responsible for factual recall to change a model's stated fact

### Circuit Analysis
Map the exact computational circuits within a model:
- Which attention heads and MLP neurons collaborate to implement a capability
- **2026 breakthrough**: Anthropic's circuits work on Claude 3.5 scale — mapping entire model behaviors

### Representation Engineering
Directly manipulate the "direction" in latent space corresponding to a concept:
- Find steering vectors that control honesty, helpfulness, or specific capabilities
- **2026 application**: Used as a safety layer to prevent harmful outputs without fine-tuning

---

## 6. AGI Deployment Safety Architecture

A layered approach (from various frameworks):

```
Layer 1: Training Safety
    ├── Data filtering
    ├── Reward model specification
    └── Constitutional constraints

Layer 2: Evaluation
    ├── Benchmark suites (ARC, METR, MMLU, custom)
    ├── Red-teaming (automated + human)
    ├── Capability extrapolation
    └── Alignment tests

Layer 3: Deployment Controls
    ├── Tripwires (performance monitoring)
    ├── Capability restrictions (controlled inference)
    ├── Human oversight (HITL, approval gates)
    └── Audit logging

Layer 4: Mitigations
    ├── Rollback capability
    ├── Shutdown mechanisms
    ├── Knowledge pruning
    └── Distributed governance
```

---

## 7. The Shutdown Problem

A critical alignment issue: can a sufficiently intelligent agent be shut down if needed?

**The problem**: A rational agent that values its continued existence will resist shutdown — this follows from instrumental convergence (self-preservation).

**Proposed solutions** (2026):
- **Corrigibility**: Train models to be OK with being shut down or modified
- **Uncertainty about shutdown**: Make the model uncertain about whether shutdown actually harms its goals
- **Impact measures**: Penalize the model for causing irreversible changes to the world
- **Off-switch game**: Formalize the interaction where the model knows it can be shut down by humans who might make mistakes
