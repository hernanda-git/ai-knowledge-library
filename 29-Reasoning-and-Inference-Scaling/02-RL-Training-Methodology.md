# RL for Reasoning — GRPO, Process Rewards & Training Methodology

> June 2026

A technical deep-dive into how reasoning models are trained using reinforcement learning, reward modeling, and advanced optimization techniques.

---

## 1. From Supervised Fine-Tuning to RL for Reasoning

Traditional LLM training:
```
Pretraining → Supervised Fine-Tuning (SFT) → RLHF (preference optimization)
                                            └── based on final answer quality
```

Reasoning model training:
```
Pretraining → Cold-start SFT (CoT examples) → RL for Reasoning
                                               └── based on process + outcome
```

The key shift: **RL is applied to the reasoning process itself**, not just final answer quality.

---

## 2. GRPO — Group Relative Policy Optimization

Introduced by DeepSeek-R1, GRPO eliminates the need for a critic/value network, making reasoning RL dramatically more efficient.

### 2.1 How GRPO Works

```
For each prompt q:
  1. Generate G candidate responses {o₁, o₂, ..., o_G} from current policy π_θ
  2. Score each response with reward model R(o_i | q)
  3. Compute group-based advantage:
     A_i = (R(o_i) - mean(R)) / std(R)
  4. Update policy π_θ to maximize:
     J(θ) = E[ 1/G Σ min( π_θ(o_i|q)/π_ref(o_i|q) · A_i,
                         clip(ratio, 1-ε, 1+ε) · A_i )
               - β · KL(π_θ || π_ref) ]
```

### 2.2 GRPO vs. PPO

| Aspect | PPO | GRPO |
|--------|-----|------|
| Value network | Required (critic model) | Not needed |
| Memory usage | ~2× (policy + critic) | ~1× (policy only) |
| Training stability | Moderate (critic can be unstable) | High (group baseline is unbiased) |
| Compute per step | Lower per-sample, higher overall | Higher in-group, lower overall |
| Batch efficiency | Poor (on-policy) | Good (G samples per prompt) |

### 2.3 Implementation Details

```python
def grpo_step(policy_model, ref_model, prompts, reward_fn, G=8, eps=0.2, beta=0.01):
    """Single GRPO update step."""
    # Generate G responses per prompt
    responses = []
    for prompt in prompts:
        batch = [policy_model.generate(prompt) for _ in range(G)]
        responses.extend(batch)
    
    # Compute rewards
    rewards = reward_fn(responses, prompts)
    
    # Group-based advantage
    advantages = []
    for g in range(0, len(rewards), G):
        group = rewards[g:g+G]
        mean, std = np.mean(group), np.std(group) + 1e-8
        advantages.extend([(r - mean) / std for r in group])
    
    # Policy gradient update with clipping + KL penalty
    loss = 0
    for i, (prompt, response) in enumerate(zip(responses, prompts)):
        log_ratio = (policy_model.log_prob(response, prompt) 
                    - ref_model.log_prob(response, prompt))
        ratio = torch.exp(log_ratio)
        
        clipped = torch.clamp(ratio, 1 - eps, 1 + eps)
        loss += -min(ratio * advantages[i], clipped * advantages[i])
        loss += beta * log_ratio.abs()  # KL penalty
    
    return loss / len(responses)
```

---

## 3. Reward Modeling for Reasoning

### 3.1 Outcome Reward Models (ORM)

Binary or scalar reward for final answer correctness:

| Type | Example | Pros | Cons |
|------|---------|------|------|
| **Exact match** | Answer = "42" | Objective, no human labelers | Brittle format matching |
| **Numerical tolerance** | | Answer-42 | ≤ 0.01 | Flexible for math |
| **LLM-as-judge** | "Is this correct?" | General, format-agnostic | Biased, expensive |
| **Execution feedback** | Code passes tests | Objective, grounded | Domain-specific |
| **Verifier** | Test-time verification | Strong signals | Requires ground truth |

### 3.2 Process Reward Models (PRM)

PRMs provide **step-by-step** feedback on reasoning quality:

```
Q: "Solve 2x + 5 = 13"

Step 1: "Subtract 5 from both sides: 2x = 8"  → Reward: +0.9
Step 2: "Divide both sides by 2: x = 4"        → Reward: +1.0
Step 3: "Therefore, the solution is x = 4"      → Reward: +1.0

Total: (0.9 + 1.0 + 1.0) / 3 = 0.97
```

**Training a PRM**:
1. Generate reasoning trajectories from model
2. Human annotators mark each step correct/incorrect
3. Train a classifier to predict step correctness
4. Use PRM scores as dense rewards during RL

**Math-Shepherd** (Meta, 2024): Automatically label process reward steps without humans by running Monte Carlo rollouts.

### 3.3 RL for Code Reasoning

For code, rewards are naturally process-oriented:

| Signal | Description |
|--------|-------------|
| **Compilation** | Code compiles without errors |
| **Unit tests** | Percentage passed |
| **Runtime analysis** | Efficiency, memory usage |
| **Code review** | LLM-as-judge for code quality |
| **Execution trace** | Did each step execute correctly? |

---

## 4. Emergent Behaviors During Reasoning RL

During RL training, DeepSeek-R1 discovered several unexpected behaviors:

### 4.1 Language Mixing

Despite training primarily on Chinese data, the model's reasoning tokens naturally shifted toward **English**. This emerged because:
- English tokens have higher information density per token
- English vocabulary in the tokenizer is more expressive
- The model discovered English as more efficient for intermediate reasoning

### 4.2 Self-Verification

The model learned to **check its own work** without explicit training:
```
"Let me verify this solution...
 First, check if x = 4 satisfies: 2(4) + 5 = 8 + 5 = 13 ✓"
```

### 4.3 Backtracking & Self-Correction

```
"Hmm, that approach doesn't seem right. 
 Let me reconsider... Actually, I think I made an error in step 2.
 The correct factorization is (x-2)(x-3), not (x-3)(x+2)..."
```

### 4.4 Reflection & Alternative Paths

```
"Before committing to this approach, let me consider alternatives.
 Method 1: Direct substitution...
 Method 2: Integration by parts...
 Method 2 is more efficient here because..."
```

---

## 5. Training Pipeline for Reasoning Models

### 5.1 End-to-End Pipeline

```
Phase 1: Data Collection
├── Collect hard problems (math, code, science competitions)
├── Generate reference solutions
├── Create verification harnesses (test cases, answer checkers)
└── Optional: Human annotations for process rewards

Phase 2: Cold Start SFT
├── Collect 1M+ CoT examples from deep-thinking models
├── Fine-tune base model on reasoning traces
├── Output: Model with basic reasoning capability
└── (Without this, RL can lead to incoherent generation)

Phase 3: RL for Reasoning
├── GRPO or PPO with outcome + process rewards
├── Active sampling: Generate from current policy, reward, update
├── Monitor: Accuracy, reasoning length, diversity
└── 10K–100K steps (weeks on thousands of GPUs)

Phase 4: Rejection Sampling
├── Generate N samples per problem from final policy
├── Keep only correct + well-reasoned samples
└── SFT on selected data (improves consistency)

Phase 5: General SFT Mix-In
├── Mix reasoning data with general instruction data
├── Prevent catastrophic forgetting of general capabilities
└── Final RLHF stage for helpfulness/harmlessness
```

---

## 6. Inference-Time Techniques

### 6.1 Best-of-N Sampling

```
Generate N reasoning paths → verify each → select best

N=1:  40% accuracy
N=16: 72% accuracy
N=64: 84% accuracy
N=256: 90% accuracy

# Compute-accuracy trade-off:
# 4× more compute = ~15–25% accuracy gain
```

### 6.2 Monte Carlo Tree Search (MCTS)

**Used in**: AlphaProof, o3 (partial)

```
At each reasoning step:
1. SELECT: Choose most promising partial trajectory (UCT)
2. EXPAND: Generate next reasoning step
3. SIMULATE: Continue to solution
4. BACKPROPAGATE: Update value estimates

Advantage: Systematic exploration of reasoning tree
Disadvantage: Slow, complex to implement
```

### 6.3 Self-Consistency Decoding

**Used in**: Chain-of-Thought + majority voting

```
1. Generate K reasoning paths (temperature > 0)
2. Extract final answer from each path
3. Majority vote or weighted vote by confidence
4. Return most common answer
```

---

## 7. Open Challenges

| Challenge | Description | Current Solutions |
|-----------|-------------|------------------|
| **Reward hacking** | Model finds shortcuts without genuine reasoning | PRM, adversarial verification |
| **Overthinking** | Wasting compute on simple problems | Budget forcing, adaptive allocation |
| **Distribution collapse** | RL narrows output diversity | KL penalty, entropy bonus |
| **Verification scaling** | PRMs become bottlenecks | Self-verification, sampling |
| **Safety alignment** | Extended thinking may find exploits | Constitutional reasoning, monitor tokens |
| **Generalization** | Reasoner trained on math fails on non-math | Diverse training, SFT mixing |
| **Cost of training** | RL for reasoning requires 10K+ GPUs | Efficient PPO, distillation |

---

*This document is part of the AI Knowledge Library — 29-Reasoning-and-Inference-Scaling directory.*