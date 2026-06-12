# Reinforcement Learning: Foundations, Algorithms, and Modern Applications

## Table of Contents
1. [Introduction to Reinforcement Learning](#1-introduction)
2. [Markov Decision Processes](#2-mdps)
3. [Dynamic Programming](#3-dp)
4. [Model-Free Prediction](#4-prediction)
5. [Model-Free Control](#5-control)
6. [Function Approximation](#6-function-approx)
7. [Deep Q-Networks](#7-dqn)
8. [Policy Gradient Methods](#8-policy-gradient)
9. [Actor-Critic Methods](#9-actor-critic)
10. [Model-Based RL](#10-model-based)
11. [Inverse RL and Imitation Learning](#11-inverse)
12. [Multi-Agent RL](#12-marl)
13. [RL for LLM Alignment](#13-rlhf)
14. [RL for Robotics](#14-robotics) — 14.1 Sim-to-Real · 14.2 Key Systems · 14.3 DQN · 14.4 PPO · 14.5 SAC · 14.6 Distributional · 14.7 Offline · 14.8 Bandits · 14.9 Exploration · 14.10 Meta-RL · 14.11 Games · 14.12 RecSys · 14.13 Advanced RLHF · 14.14 Frameworks
15. [Open Problems](#15-open)
15a. [Practical RL Project Guide](#15a-practical-rl-project-guide)
16. [Cross-References](#16-cross-references)

---

## 1. Introduction

Reinforcement Learning (RL) is the branch of machine learning concerned with how agents ought to take actions in an environment to maximize cumulative reward. Unlike supervised learning (learning from labeled data) or unsupervised learning (finding patterns in unlabeled data), RL learns from *interaction* — the agent tries actions, observes outcomes, and learns which actions produce the most reward.

### 1.1 The RL Loop
Agent → Action → Environment → State + Reward → Agent...

### 1.2 RL vs Supervised vs Unsupervised
| Aspect | Supervised | Unsupervised | Reinforcement |
|--------|-----------|-------------|---------------|
| Data | Labeled examples | Unlabeled data | Interaction history |
| Feedback | Per-instance labels | None (patterns) | Delayed reward signal |
| Goal | Generalize from examples | Find structure | Maximize cumulative reward |
| Time | Static | Static | Sequential (credit assignment) |
| Exploration | Not needed | Not needed | Essential |

---

## 2. Markov Decision Processes

MDPs formalize sequential decision-making. Defined by (S, A, P, R, γ):
- S: state space
- A: action space  
- P: transition dynamics P(s'|s,a)
- R: reward function R(s,a,s')
- γ: discount factor [0,1]

### 2.1 Policy
π(a|s) = probability of taking action a in state s

### 2.2 Value Functions
- V^π(s) = E[Σγ^t R_t | s_0=s, π] — expected return from state s following π
- Q^π(s,a) = E[Σγ^t R_t | s_0=s, a_0=a, π] — expected return taking a in s then following π

### 2.3 Bellman Equations
V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a)[R(s,a,s') + γV^π(s')]
Q^π(s,a) = Σ_{s'} P(s'|s,a)[R(s,a,s') + γ Σ_{a'} π(a'|s')Q^π(s',a')]

### 2.4 Optimality
V*(s) = max_a Q*(s,a)
Q*(s,a) = Σ_{s'} P(s'|s,a)[R(s,a,s') + γ max_{a'} Q*(s',a')]

---

## 3. Dynamic Programming

### 3.1 Policy Iteration
1. Evaluate: compute V^π for current π
2. Improve: π'(s) = argmax_a Σ_{s'} P(s'|s,a)[R(s,a,s') + γV^π(s')]
3. Repeat until convergence

### 3.2 Value Iteration
V_{k+1}(s) = max_a Σ_{s'} P(s'|s,a)[R(s,a,s') + γV_k(s')]
Converges to V* without explicit policy representation.

### 3.3 Generalized Policy Iteration
Most RL algorithms alternate (partial) evaluation and (partial) improvement.

---

## 4. Model-Free Prediction

### 4.1 Monte Carlo Learning
Average returns from complete episodes.
V(s) ← V(s) + α(G_t - V(s))
where G_t is the actual return from state s.

### 4.2 Temporal Difference Learning
Learn from incomplete episodes by bootstrapping.
V(s_t) ← V(s_t) + α(r_t + γV(s_{t+1}) - V(s_t))
TD error: δ_t = r_t + γV(s_{t+1}) - V(s_t)

### 4.3 TD(λ)
Eligibility traces combine MC (λ=1) and TD(0) (λ=0).
V(s) ← V(s) + αδ_t e_t(s) where e_t(s) = γλ e_{t-1}(s) + 1(s_t=s)

---

## 5. Model-Free Control

### 5.1 Q-Learning (Watkins, 1989)
Off-policy TD control — learns optimal Q-function regardless of behavior policy.
Q(s,a) ← Q(s,a) + α[r + γ max_{a'} Q(s',a') - Q(s,a)]

### 5.2 SARSA
On-policy TD control — learns Q-function for the current policy.
Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]

### 5.3 Exploration vs Exploitation
- ε-greedy: with probability ε, take random action; otherwise greedy
- Boltzmann (softmax): P(a|s) ∝ exp(Q(s,a)/τ), τ controls exploration
- Upper Confidence Bound: a = argmax_a Q(s,a) + c√(ln t/N(s,a))
- Thompson Sampling: sample Q from posterior, act greedily w.r.t. sample

---

## 6. Function Approximation

When state/action spaces are too large for tables, use function approximators:
- Q(s,a) ≈ Q_θ(s,a) — parameterized by θ (neural network weights)
- V(s) ≈ V_θ(s)
- π(a|s) ≈ π_θ(a|s)

### 6.1 Linear Function Approximation
Q(s,a) = φ(s,a)ᵀw. Simple, stable, but limited expressivity.

### 6.2 Neural Function Approximation
Q(s,a) = NN_θ(s,a). Powerful but can be unstable.
Three "deadly sins" (Van Hasselt et al., 2018):
1. Function approximation (bias)
2. Bootstrapping (self-referential targets)
3. Off-policy learning (distribution mismatch)

---

## 7. Deep Q-Networks

### 7.1 DQN (Mnih et al., 2015)
First stable deep RL algorithm, playing Atari from pixels.

**Key Innovations:**
1. **Experience Replay:** Store (s,a,r,s') in a buffer, sample randomly to break correlations
2. **Target Network:** Fixed Q-target network, periodically updated to reduce moving target problem
3. **Frame Stacking:** 4 consecutive frames as input to capture motion

**Loss:** L(θ) = E[(r + γ max_{a'} Q(s',a'; θ⁻) - Q(s,a; θ))²]
where θ⁻ is the target network (frozen, periodically copied from θ)

### 7.2 DQN Improvements

| Variant | Innovation | Gain |
|---------|------------|:----:|
| Double DQN | Decouple action selection from evaluation | Reduces overestimation bias |
| Dueling DQN | Separate V(s) + A(s,a) streams | Better policy evaluation |
| Prioritized Replay | Sample important transitions more | 2-4× faster learning |
| C51 (Categorical DQN) | Learn distribution of returns, not just mean | State-of-the-art for years |
| Rainbow | Combine all improvements | Best DQN variant |

---

## 8. Policy Gradient Methods

### 8.1 The Policy Gradient Theorem
∇J(θ) = E_π[∇log π_θ(a|s) Q^π(s,a)]

Key insight: increase probability of actions with high Q-values, decrease those with low Q-values, weighted by the gradient of log probability.

### 8.2 REINFORCE (Williams, 1992)
Monte Carlo policy gradient:
∇J(θ) = Σ_t G_t ∇log π_θ(a_t|s_t)
where G_t is the actual return from step t.

**Issues:** High variance because G_t is a sample estimate.

### 8.3 Variance Reduction
- **Baseline:** Subtract a state-dependent baseline b(s) from Q-values
- **Advantage function:** A(s,a) = Q(s,a) - V(s) — how much better is a than average?
- **Actor-Critic:** Learn both policy (actor) and value function (critic)

---

## 9. Actor-Critic Methods

### 9.1 Advantage Actor-Critic (A2C)
Two networks:
- Actor: π_θ(a|s) — learns the policy
- Critic: V_φ(s) — learns state values

Losses:
- Actor: ∇J(θ) = ∇log π_θ(a|s) (r + γV(s') - V(s))
- Critic: L(φ) = (r + γV_φ(s') - V_φ(s))²

### 9.2 A3C (Asynchronous Advantage Actor-Critic)
Multiple parallel workers each with their own environment copy. Workers update a shared model asynchronously.

### 9.3 PPO (Proximal Policy Optimization, Schulman et al., 2017)
**The default RL algorithm for 2022-2026.**

Key idea: trust region optimization via clipped surrogate objective.
L_CLIP(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
where r_t(θ) = π_θ(a_t|s_t)/π_θ_old(a_t|s_t) is the probability ratio.

**Why PPO works:**
- Simple to implement (no Lagrange multipliers)
- Stable (clipping prevents large policy updates)
- Compatible with discrete and continuous actions
- Used for: RLHF, Dota 2, robotics, game playing

### 9.4 SAC (Soft Actor-Critic, Haarnoja et al., 2018)
Maximum entropy RL — optimizes both reward and entropy:
J = Σ E[r_t + α H(π(·|s_t))]

**Advantages:**
- Off-policy (sample efficient)
- Automatically tunes α (temperature)
- State-of-the-art for continuous control
- Used for: robotics, autonomous driving

---

## 10. Model-Based RL

### 10.1 Dyna Architecture
Learn a model of the environment, then use it for planning:
1. Interact with real environment → store experience
2. Learn model: P_θ(s'|s,a), R_θ(s,a) from experience
3. Plan using the model: simulate trajectories, update Q-values
4. Act: use Q-values to select actions

### 10.2 MuZero (Schrittwieser et al., 2020)
Learns a model in latent space without explicit reward or transition prediction.
- Representation function: h(s) = latent state
- Dynamics function: g(latent, a) = next latent + reward
- Prediction function: f(latent) = policy + value
- Uses MCTS for planning in latent space
- Achieves superhuman performance in Go, Chess, Shogi, Atari

---

## 11. Inverse RL and Imitation Learning

### 11.1 Behavioral Cloning
Supervised learning: learn π*(a|s) from expert demonstrations.
**Problem:** Distribution mismatch — the agent makes small mistakes, enters states the expert never was in, and compounding errors grow.

### 11.2 Inverse Reinforcement Learning
Given expert demonstrations, learn the reward function R(s) that the expert was optimizing.

**Methods:**
- Maximum Entropy IRL (Ziebart et al., 2008): find R that maximizes the likelihood of expert behavior under maximum entropy model
- GAIL (Generative Adversarial Imitation Learning): discriminator distinguishes expert vs agent, agent tries to fool it

---

## 12. Multi-Agent RL

### 12.1 Challenges
- **Non-stationarity:** Other agents' policies change → environment distribution shifts
- **Scalability:** Joint action space grows exponentially with agents
- **Credit assignment:** Which agent's action contributed to the reward?
- **Communication:** What information should agents share?

### 12.2 Key Algorithms
| Algorithm | Approach | Best For |
|-----------|----------|----------|
| **Independent Q-Learning** | Each agent learns Q-function independently | Simple tasks |
| **VDN** | Factorized Q-functions: Q_tot = Σ Q_i | Cooperative tasks |
| **QMIX** | Monotonic mixing: Q_tot = f(Q_1,...,Q_n) | More complex coordination |
| **MADDPG** | Centralized critic, decentralized actors | Mixed competitive/cooperative |
| **MAPPO** | Multi-agent PPO | General multi-agent RL |

---

## 13. RL for LLM Alignment

### 13.1 RLHF Overview
The three-stage RLHF pipeline (see: [07-Emerging/02-AI-Safety.md] §3.1):
1. SFT on demonstrations
2. Reward model training on preferences
3. PPO optimization against reward model

### 13.2 PPO in RLHF
PPO for LLMs differs from standard PPO:
- **Policy:** The LLM (generating tokens)
- **State:** The prompt + generated tokens so far
- **Action:** Next token to generate
- **Reward:** Reward model score + KL penalty

**KL Penalty:** β D_KL(π_θ || π_ref) prevents the policy from drifting too far from the SFT model, preserving capability.

### 13.3 GRPO (Group Relative Policy Optimization)
Used by DeepSeek-R1. Multiple responses are generated for each prompt, and advantage is computed as the group-relative score rather than from a learned value function.

**Benefits:** No critic network needed; simpler, more stable for language tasks.

### 13.4 RL for Reasoning
DeepSeek-R1 showed that RL alone (without SFT on reasoning chains) can teach models to reason:
1. Initialize from a base model (no reasoning training)
2. Generate responses to math problems
3. Reward: correct answer (verifiable) + format constraints
4. Run GRPO to optimize
5. Over training, the model naturally develops chain-of-thought reasoning

---

## 14. RL for Robotics

### 14.1 Sim-to-Real Transfer
Training in simulation, deploying on real robots:
- Domain randomization: randomize physics, visuals, dynamics
- System identification: match simulation to real dynamics
- Progressive net: adapt features from simulation to real

### 14.2 Key Robotic RL Systems
- **DROID (Google DeepMind):** Distributed RL for robot manipulation
- **RoboCat (DeepMind):** Multi-embodiment RL agent
- **RT-2 (Google):** VLA (Vision-Language-Action) model trained on robot data
- **Octo (UC Berkeley):** Open-source robot foundation model

---

### 14.3 DQN Implementation in PyTorch

A complete DQN agent with experience replay and Double DQN soft updates:

```python
class ReplayBuffer:
    """Experience replay buffer — stores transitions, samples i.i.d. batches."""
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return [np.array(x) for x in zip(*batch)]

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    """Deep Q-Network with two hidden layers."""
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, act_dim))

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99,
                 tau=0.005, buffer_capacity=100000, batch_size=64):
        self.gamma, self.batch_size, self.tau = gamma, batch_size, tau
        self.policy_net = DQN(obs_dim, act_dim)
        self.target_net = DQN(obs_dim, act_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optim = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

    def act(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randrange(self.policy_net.net[-1].out_features)
        with torch.no_grad():
            return self.policy_net(torch.FloatTensor(state)).argmax().item()

    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        s, a, r, s_next, d = self.buffer.sample(self.batch_size)
        s = torch.FloatTensor(s)
        a = torch.LongTensor(a).unsqueeze(1)
        r = torch.FloatTensor(r)
        s_next = torch.FloatTensor(s_next)
        d = torch.FloatTensor(d)

        # Current Q(s,a)
        q_current = self.policy_net(s).gather(1, a).squeeze()
        # Double DQN: select action with policy net, evaluate with target net
        with torch.no_grad():
            next_actions = self.policy_net(s_next).argmax(1, keepdim=True)
            q_next = self.target_net(s_next).gather(1, next_actions).squeeze()
            q_target = r + self.gamma * q_next * (1 - d)

        loss = F.mse_loss(q_current, q_target)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # Polyak soft update of target network
        for p, tp in zip(self.policy_net.parameters(),
                         self.target_net.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
```

**Key design decisions:** (1) Experience replay breaks temporal correlations by sampling i.i.d. batches. (2) Target network stabilizes learning — without it, the regression target moves with every gradient step. (3) Soft target update (Polyak averaging, τ=0.005) avoids hard parameter spikes. (4) Double DQN action selection reduces overestimation bias by decoupling selection from evaluation.

### 14.4 PPO Implementation Sketch

PPO's clipped surrogate objective in core PyTorch:

```python
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation — bias-variance tradeoff via λ."""
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards) - 1)):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    return advantages, advantages + values  # returns = advantages + values

def ppo_loss(policy, value_net, states, actions, old_log_probs,
             advantages, returns, clip_epsilon=0.2):
    # Probability ratio r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
    log_probs = policy.log_prob(states, actions)
    ratio = torch.exp(log_probs - old_log_probs)

    # Clipped surrogate objective (core PPO innovation)
    clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    actor_loss = -torch.min(ratio * advantages,
                            clipped * advantages).mean()

    # Value function loss
    value_pred = value_net(states).squeeze()
    value_loss = 0.5 * F.mse_loss(value_pred, returns)

    # Optional entropy bonus for exploration
    entropy = policy.entropy(states).mean()
    total_loss = actor_loss + value_loss - 0.01 * entropy
    return total_loss
```

**Why clipping works:** The min(ratio·A, clip(ratio)·A) objective ensures that when the probability ratio exits [1-ε, 1+ε], the gradient is zeroed — further deviation no longer improves the loss. This implements a soft trust region without costly second-order methods (TRPO's conjugate gradient), making PPO the default RL algorithm for most applications.

### 14.5 SAC Algorithm Deeper Dive

Soft Actor-Critic (Haarnoja et al., 2018) optimizes for **maximum entropy** rather than maximum reward alone:

| Component | Standard Actor-Critic | SAC |
|-----------|----------------------|-----|
| Objective | max E[Σγ^t r_t] | max E[Σγ^t (r_t + α·H(π(·|s_t)))] |
| Policy | Deterministic or Gaussian | Reparameterized Gaussian (tanh squashed) |
| Critics | Single Q-network | Two Q-networks + min ensemble |
| Temperature α | N/A | Automatically tuned to target entropy |
| Data efficiency | On-policy | Off-policy (replay buffer) |

SAC maintains three networks: two Q-functions (Q₁, Q₂) to combat overestimation, and a policy network. The reparameterization trick enables gradient flow through sampling: a = tanh(μ_θ(s) + σ_θ(s)·ε), ε ∼ N(0, I).

**Loss functions:**

- **Q-loss:** L(φ_i) = E[(Q_{φ_i}(s,a) - (r + γ(min_j Q_{φ̄_j}(s',a') - α log π(a'|s'))))²]
- **Policy loss:** L(θ) = E[α log π_θ(a|s) - min_j Q_{φ_j}(s,a)]
- **Temperature loss:** L(α) = E[-α log π_θ(a|s) - α·H_target]

The entropy bonus encourages exploration and prevents premature convergence to a deterministic policy. SAC is the default algorithm for continuous control (MuJoCo, robotics) with 3-10× better sample efficiency than PPO on dense-reward tasks.

### 14.6 Distributional RL (C51, QR-DQN)

Standard RL learns the **expected** return. Distributional RL models the **full distribution** Z(s,a), capturing risk, uncertainty, and multi-modal rewards.

**C51 (Bellemare et al., 2017):** Models Z(s,a) as a categorical distribution over 51 fixed atoms spanning [V_min, V_max]. The loss is cross-entropy between the projected target and predicted distributions. C51 achieved state-of-the-art Atari results by modeling return variance explicitly — crucial when different outcomes have the same expectation but vastly different risk profiles.

**QR-DQN (Dabney et al., 2018):** Learns quantile values at N evenly-spaced fractions τ_i = (2i-1)/(2N). The quantile Huber loss is:

ρ^κ_τ(δ) = |τ - 1{δ<0}| · L_κ(δ) / κ

QR-DQN is simpler than C51, converges faster, and adapts distribution shape naturally without requiring a predefined support range.

| Aspect | C51 | QR-DQN |
|--------|-----|--------|
| Distribution | Categorical (fixed atoms) | Implicit (learned quantiles) |
| Loss | Cross-entropy | Quantile Huber |
| Output size | 51 × |A| | N × |A| |
| Training stability | Good — fixed support | Better — adaptive quantiles |
| Key advantage | First to show distributional works | Simpler, faster, often more accurate |

Distributional RL is essential for **risk-sensitive** applications: a medical diagnosis agent should avoid high-variance actions even if the mean Q-value is high.

### 14.7 Offline RL (CQL, IQL)

Offline RL (batch RL) learns entirely from a fixed dataset without environment interaction — crucial when interaction is expensive or dangerous (healthcare, autonomous driving, robotics).

**Key challenge — distributional shift:** The learned policy takes actions not covered by the dataset, producing out-of-distribution Q-values. Standard Q-learning catastrophically overestimates OOD actions because it maximizes over unseen action choices.

**CQL (Kumar et al., 2020):** Adds a regularizer penalizing Q-values for OOD actions while boosting in-distribution Q:

L_CQL = α·E_s[log Σ_a exp(Q(s,a)) - E_{a∼D}[Q(s,a)]] + L_Bellman

This produces a conservative lower-bound Q-function that is safe to maximize, preventing the policy from exploiting erroneously high Q estimates.

**IQL (Kostrikov et al., 2022):** Avoids evaluating OOD actions entirely via expectile regression:

L_IQL = E[|τ - 1{δ<0}| · δ²], where δ = r + γV(s') - Q(s,a)

IQL never queries Q(s,a) for non-dataset actions, achieving state-of-the-art results on D4RL benchmarks.

| Method | Strategy | Handles OOD? | Best For |
|--------|----------|:------------:|----------|
| BCQ | Constrain actions toward dataset | ✓ | Simple offline tasks |
| CQL | Conservative Q regularization | ✓ | General offline RL |
| IQL | Implicit (expectile) backup | ✓ | High-dim, sparse reward |
| DT (Decision Transformer) | Sequence modeling (transformer) | ✓ | Trajectory-conditioned tasks |

Offline RL is the backbone of RL foundation models — large-scale pre-training on diverse datasets followed by minimal fine-tuning, analogous to how NLP shifted from task-specific training to pre-training + fine-tuning.

### 14.8 Multi-Armed Bandits

The simplest form of RL — a single state, multiple actions, immediate reward. Bandits formalize the explore-exploit dilemma in its purest form.

| Algorithm | Strategy | Regret | Best For |
|-----------|----------|:------:|----------|
| **ε-Greedy** | Explore with prob ε | Linear | Simple, easy to implement |
| **UCB1** | Optimism under uncertainty | O(log n) | Stationary rewards |
| **Thompson Sampling** | Sample from posterior | O(log n) | Bayesian, contextual |
| **EXP3** | Exponential weight update | O(√n) | Adversarial rewards |
| **LinUCB** | Contextual linear UCB | O(d√n) | Contextual bandits (news) |

**Contextual Bandits** extend bandits with state (context): the algorithm observes features, selects an action, and receives a reward. Used for:
- **News recommendation:** Recommend articles based on user context (LinUCB by Li et al., 2010)
- **Ad serving:** Select ads to maximize CTR given user demographics
- **Clinical trials:** Dynamically assign treatments to patients
- **A/B testing:** Reduce regret vs static A/B tests

```python
# Thompson Sampling for Bernoulli bandits
class ThompsonSampler:
    def __init__(self, n_arms):
        self.alpha = np.ones(n_arms)  # Beta prior successes
        self.beta = np.ones(n_arms)   # Beta prior failures
    
    def select_arm(self):
        samples = [np.random.beta(a, b) 
                   for a, b in zip(self.alpha, self.beta)]
        return np.argmax(samples)
    
    def update(self, arm, reward):
        self.alpha[arm] += reward
        self.beta[arm] += 1 - reward
```

### 14.9 Intrinsic Motivation and Exploration

Beyond ε-greedy, modern RL uses **intrinsic rewards** to drive exploration:

| Method | Bonus Signal | Formula |
|--------|-------------|---------|
| **Count-based** | Visit count | r⁺ = 1/√N(s) |
| **RND** (Random Network Distillation) | Prediction error of a fixed random network | r⁺ = ||f(φ(s)) - f̂(φ(s))||² |
| **ICM** (Intrinsic Curiosity Module) | Forward dynamics prediction error | r⁺ = ||φ̂(s_{t+1}) - φ(s_{t+1})||² |
| **BeBold** | Count-based in embedding space | r⁺ ∝ 1/√N(φ(s)) |
| **Go-Explore** | Archive-based (return to promising states) | Not novelty — return-based |

**RND (Burda et al., 2019):** Two networks — a fixed random target network f and a predictor f̂ trained on visited states. Novel states have high prediction error → high intrinsic reward. Used by OpenAI to achieve superhuman performance on Montezuma's Revenge (a notoriously hard exploration Atari game).

**Case Study — Montezuma's Revenge:** Before intrinsic motivation, DQN achieved ~0 reward after 200M frames. With RND, agents achieved ~12,000 reward — the first RL agent to solve the first room without human demonstrations.

### 14.10 Meta-Reinforcement Learning

Meta-RL trains agents that can quickly adapt to new tasks using experience from previous tasks — "learning to learn" in RL.

| Method | Approach | Key Innovation |
|--------|----------|---------------|
| **RL²** (Duan et al., 2016) | RNN trained with RL across tasks | Recurrent policy internalizes learning algorithm |
| **MAML-RL** (Finn et al., 2017) | Model-agnostic meta-learning | Few-shot gradient adaptation to new reward |
| **PEARL** (Rakelly et al., 2019) | Probabilistic embeddings for meta-RL | Off-policy, sample efficient |
| **VariBAD** (Zintgraf et al., 2020) | Bayes-adaptive MDP approximation | Uncertainty-aware task inference |

**PEARL Architecture:**
1. **Context encoder:** Encodes (s, a, r, s') trajectories into task embedding z
2. **Policy:** π(a|s, z) conditioned on task embedding
3. **Training:** Sample task → collect data → infer z → update policy
4. **Test time:** New task → few transitions → z captures task → policy adapts immediately

Meta-RL enables **few-shot adaptation** in robotics: a robot trained across 100 different environments can adapt to a new environment in 1-3 trials.

### 14.11 RL for Game Playing — Case Studies

| System | Game | Method | Achievement |
|--------|------|--------|-------------|
| **AlphaGo** (2016) | Go | MCTS + policy/value nets | Beat world champion Lee Sedol |
| **AlphaZero** (2017) | Chess, Go, Shogi | Self-play MCTS | Superhuman without human data |
| **MuZero** (2020) | Atari, Chess, Go | Learned model + MCTS | No game rules needed |
| **OpenAI Five** (2019) | Dota 2 | PPO + LSTM + team spirit | Beat world champions |
| **AlphaStar** (2019) | StarCraft II | Deep RL + league training | Grandmaster on all races |
| **Pluribus** (2019) | No-limit poker | CFR + self-play | Superhuman in 6-player poker |
| **Student of Games** (2023) | Multiple games | Search + RL | General game-playing agent |
| **Sokoban** | Push-block puzzles | A3C + curriculum | Human-level on hard puzzles |

**AlphaZero's Self-Play Learning Loop:**
```
1. Self-play: Current best network generates MCTS games
2. Training: Predict move probabilities (policy) + game outcome (value)
3. Evaluation: New network vs best network
4. Acceptance: If new wins >55%, replace best
5. Repeat: ~44 million self-play games for chess
```

AlphaZero-trained networks discovered human-like chess strategies (king safety, piece development) but also entirely novel patterns. The system required only the rules of chess and zero human expert data.

### 14.12 RL for Recommendation Systems

RL is increasingly applied to recommendation systems to handle the dynamic, sequential nature of user interactions:

| Application | Model | State | Action | Reward |
|-------------|-------|-------|--------|--------|
| **News recommendation** | DQN/LinUCB | User features, history | Article selection | CTR, dwell time |
| **Video recommendation** | DeepBayes + RL | Watch history, time | Video carousel order | Engagement time |
| **E-commerce** | DDPG | Session features, cart | Product ranking | Conversion, revenue |
| **Content feed** | PPO + bandit | User embedding, context | Which content to show | Long-term retention |

**Key challenge:** The reward is delayed (user who sees irrelevant content today may churn next month). RL naturally handles this via discounted returns and value functions.

### 14.13 Learning from Human Feedback (Advanced RLHF Techniques)

Beyond basic PPO-based RLHF, advanced methods have emerged:

| Technique | Innovation | Benefit |
|-----------|------------|---------|
| **DPO** (Rafailov, 2023) | Closed-form optimal policy | No reward model needed |
| **GRPO** (DeepSeek, 2024) | Group-based advantage | No critic network |
| **REINFORCE Leave-One-Out** | Monte Carlo with baseline | Simple, effective |
| **SPIN** (Chen et al., 2024) | Self-play fine-tuning | Iterative self-improvement |
| **RLEF** (Reinforcement Learning from Execution Feedback) | Code execution reward | Grounded reasoning |
| **PRO** (Preference Reward Optimization) | Preference + reward mixing | Better Pareto frontier |

**GRPO Detailed** (used by DeepSeek-R1):
```
For each prompt q:
  1. Generate G responses {o_1, ..., o_G} from old policy π_θ_old
  2. Compute rewards {r_1, ..., r_G} for each response
  3. Normalize: A_i = (r_i - mean(r)) / std(r)  # Group-relative advantage
  4. Optimize: J_GRPO(θ) = E[1/G Σ min(π_θ(o_i|q)/π_θ_old(o_i|q)·A_i, clip(...)·A_i) - β·D_KL(π_θ||π_ref)]
```

GRPO eliminates the value function critic (a major source of instability in PPO for LLMs) by using the group of responses to compute the baseline. This is particularly effective when rewards are **verifiable** (math answers, code correctness) rather than learned from preferences.

---

### 14.14 RL Frameworks and Training Ecosystem

The RL ecosystem has matured significantly. This section compares the major RL libraries and provides practical tips for training.

#### RL Library Comparison

| Library | Framework | Algorithms | Parallel Envs | Distributed | Use Case |
|---------|:---------:|:----------:|:-------------:|:-----------:|----------|
| **Stable-Baselines3** | PyTorch | PPO, A2C, DQN, SAC, TD3, ARS | ❌ (requires wrapper) | ❌ | Research, education, standard benchmarks |
| **Ray RLlib** | PyTorch/TF | PPO, DQN, SAC, APEX, IMPALA, MARL | ✅ Native | ✅ Multi-node | Production, large-scale, multi-agent |
| **CleanRL** | PyTorch | PPO, DQN, SAC, A2C, DDPG | ❌ | ❌ | Education, reproducible research |
| **Acme (DeepMind)** | TensorFlow/JAX | DQN, D4PG, R2D2, IMPALA, PPO | ✅ | ✅ | Research, distributed RL |
| **SBX (JAX SB3)** | JAX | PPO, SAC, DQN, TD3 | ❌ | ❌ | Fast JAX-native training |
| **Tianshou** | PyTorch | PPO, DQN, SAC, IQN, CQL, BC | ✅ | ✅ | Research, offline RL |
| **Dopamine (Google)** | JAX/TF | DQN, Rainbow, IQN, C51 | ❌ | ❌ | Atari research, distributional RL |
| **minRL** | PyTorch | PPO, DQN, SAC (minimal implementations) | ❌ | ❌ | Learning, educational |

**Quick start recommendation:**
- **"I want to reproduce a paper result"** → Stable-Baselines3 or CleanRL
- **"I need production-scale distributed training"** → Ray RLlib
- **"I need JAX speed"** → SBX or Dopamine
- **"I study offline RL"** → Tianshou or d3rlpy
- **"I'm learning RL for the first time"** → CleanRL (50-line implementations)

#### Hyperparameter Tuning Guide

| Algorithm | Most Sensitive Parameter | Default | Tuning Range | Impact |
|-----------|------------------------|:-------:|:------------:|--------|
| **PPO** | `clip_epsilon` | 0.2 | 0.1–0.3 | Higher = larger updates (risk of instability) |
| **PPO** | `learning_rate` | 3e-4 | 1e-4–1e-3 | Lower = more stable but slower |
| **PPO** | `entropy_coef` | 0.01 | 0.0–0.1 | Higher = more exploration |
| **DQN** | `learning_rate` | 1e-4 | 1e-5–1e-3 | Lower = more stable |
| **DQN (Rainbow)** | `n_steps` (multi-step) | 3 | 1–10 | Higher = faster credit assignment |
| **SAC** | `alpha` (temperature) | Auto-tuned | 0.05–0.5 | Higher = more entropy/exploration |
| **SAC** | `tau` (target smoothing) | 0.005 | 0.001–0.01 | Lower = slower target update (more stable) |
| **PPO (RLHF)** | `kl_coef` (β) | 0.04 | 0.01–0.2 | Higher = less drift from base model |

```python
# Example: PPO hyperparameter sweep with Ray Tune
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .environment("CartPole-v1")
    .training(
        lr=tune.loguniform(1e-4, 1e-3),
        train_batch_size=tune.choice([2048, 4096, 8192]),
        clip_param=tune.uniform(0.1, 0.3),
        entropy_coeff=tune.loguniform(0.001, 0.1),
        vf_loss_coeff=tune.uniform(0.1, 1.0),
    )
    .evaluation(evaluation_num_episodes=10)
)
tuner = tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    tune_config=tune.TuneConfig(num_samples=20, metric="episode_reward_mean", mode="max"),
)
results = tuner.fit()
print(f"Best config: {results.get_best_result().config}")
```

#### Environment Standardization

| Environment Suite | Domain | State Space | Action Space | Notes |
|------------------|--------|:-----------:|:------------:|-------|
| **Gymnasium / Gym** | General | Any | Discrete/Cont. | De facto standard; 1,000+ environments |
| **MuJoCo** | Continuous control | Continuous | Continuous | Standard for SAC/PPO benchmarking |
| **Atari (ALE)** | Arcade games | Image (84×84×4) | Discrete (18) | Classic DQN benchmark |
| **Procgen** | Generalization | Image (64×64) | Discrete (15) | Test generalization across levels |
| **DM Control** | Continuous control | Continuous/image | Continuous | DeepMind's suite; harder than MuJoCo |
| **Meta-World** | Multi-task robotics | Continuous | Continuous | 50+ manipulation tasks |
| **D4RL** | Offline RL | Various | Various | Standard offline RL benchmark |
| **Minigrid / BabyAI** | Navigation | Grid | Discrete | Goal-conditioned, instruction following |
| **NetHack** | Roguelike game | Symbolic | Discrete | Extreme exploration challenge |
| **Pokémon Showdown** | Competitive | Mixed | Mixed | Multi-agent, long horizon |

#### Training Infrastructure Tips

| Concern | Recommendation |
|---------|---------------|
| **GPU vs CPU for RL** | Most RL algorithms are CPU-bound (env simulation). Use 8-16 CPU cores per training run. GPU only helps for large neural networks (image-based policies). |
| **Vectorized environments** | Use `gym.vector.make()` or `SubprocVecEnv` for N parallel envs (N = 4-64). Environments run in separate processes on CPU. |
| **Replay buffer memory** | DQN replay buffer with 1M transitions × 84×84×4 uint8 = ~27 GB. Use frame compression (JPEG, delta encoding) or smaller buffer. |
| **Distributed RL** | Ray RLlib: 1 learner GPU + N worker CPUs. IMPALA architecture scales to 100s of workers. |
| **Experiment tracking** | Use W&B or TensorBoard. Log: episode reward, loss components, explained variance, entropy, gradient norms, KL divergence. |
| **Checkpointing** | Save model every 10-50 epochs. For long runs (1B+ frames), save replay buffer + optimizer state + RNG state for exact restarts. |
| **Deterministic mode** | Set `seed`, `torch.manual_seed(seed)`, `np.random.seed(seed)`, `random.seed(seed)`, and environment seeds for reproducibility. |

#### RL Training Checklist

- [ ] **Env correctness:** Run random policy for 100 episodes; verify reward range and episode length
- [ ] **Sanity check:** Overfit on a single state-action pair (policy should maximize reward for that specific action)
- [ ] **Seed sweep:** Run 3-5 seeds per configuration; RL results are notoriously seed-sensitive
- [ ] **Monitor:** Track episode reward, value loss, policy loss, entropy, explained variance, KL (for PPO), gradient norms
- [ ] **Evaluation:** Evaluate every N epochs on a fixed set of seeds with deterministic policy (no exploration noise)
- [ ] **Performance baseline:** Compare against a trivial policy (always take action 0) and a random policy
- [ ] **Compute budget:** Estimate total environment steps; typical benchmarks: 1M (simple), 10M (medium), 100M+ (Atari)

### Cross-References for RL Ecosystem

| Reference | Description |
|-----------|-------------|
| [07-Emerging/02-AI-Safety.md] | RLHF for alignment, reward hacking |
| [06-Advanced/03-Evaluation-Benchmarks.md] | RL benchmarks |
| [08-Reference/01-Glossary.md] | Key RL terms |

---

## 15. Open Problems

1. **Sample efficiency:** RL requires millions of interactions — far too many for real-world applications
2. **Reward specification:** Designing reward functions that don't lead to reward hacking
3. **Exploration in high dimensions:** How to explore large state-action spaces efficiently
4. **Credit assignment:** Which past actions caused the current reward?
5. **Transfer learning:** How to transfer RL policies across tasks
6. **Multi-agent coordination:** How to learn cooperative policies with communication
7. **Safety:** How to ensure RL agents don't exploit reward function loopholes
8. **Compositionality:** Can RL agents learn reusable skills?
9. **Foundation models for RL:** Can we pre-train generalist policies (like Gato, RT-2) that transfer broadly?
10. **World models:** Can learned world models (Dreamer, DayDreamer) replace explicit simulators for planning?
11. **Hierarchical RL:** How to learn useful abstractions and subgoals automatically?
12. **Biological plausibility:** How does the brain solve RL with far fewer samples and no replay?

---

## 15a. Practical RL Project Guide

This section provides a practical guide for setting up, running, debugging, and deploying RL projects — bridging the gap between textbook algorithms and working implementations.

### 15a.1 Project Setup Checklist

| Phase | Task | Recommendation |
|-------|------|---------------|
| **Environment** | Choose environment suite | Gymnasium for standard benchmarks; custom env for domain-specific tasks |
| | Verify environment | Run random policy for 100 episodes; check reward range, episode length, action bounds |
| | Vectorize | Use `gym.vector.SyncVectorEnv` or `AsyncVectorEnv` with 4-16 parallel envs |
| **Model** | Select algorithm | PPO for general use, SAC for continuous control, DQN/Rainbow for discrete |
| | Architecture | 2-3 hidden layers, 64-256 units each; layer norm for stability |
| | Seed everything | `torch.manual_seed + np.random.seed + random.seed + env seed` |
| **Training** | Learning rate | 3e-4 (Adam) — the most robust default across algorithms |
| | Batch size | 64 (DQN), 2048-4096 (PPO), 256 (SAC) |
| | Total steps | 1M (simple), 10M (medium), 100M+ (Atari, hard tasks) |
| **Tracking** | Metrics | Episode reward, value loss, policy loss, entropy, explained variance, KL |
| | Logging | WandB or TensorBoard; log every 1000 steps, eval every 10K steps |
| | Checkpoints | Save model every 10-50 epochs; include optimizer state, RNG state |
| **Evaluation** | Deterministic policy | Disable exploration noise during evaluation |
| | Multiple seeds | Run 3-5 seeds; RL results are notoriously seed-sensitive |
| | Performance baseline | Compare against random policy and trivial policy (always action 0) |

### 15a.2 Common Debugging Scenarios

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| **Reward never increases** | Learning rate too high or too low | Try 1e-4, 3e-4, 1e-3; check reward scaling |
| **Policy collapses to deterministic** | Entropy coefficient too low | Increase entropy bonus (0.01 → 0.05 for PPO); check entropy logging |
| **Q-values explode** | No target network or slow update | Add target network; reduce tau (0.005 → 0.001) |
| **Training oscillates** | Learning rate too high | Reduce LR; increase batch size; check gradient clipping |
| **Slow convergence** | Insufficient exploration | Increase ε (DQN) or entropy (policy gradient); add intrinsic motivation |
| **Evaluation inconsistent** | Seed sensitivity | Always evaluate with multiple seeds; report mean ± std over 5 seeds |
| **High variance in returns** | Episode length too short | Increase max episode steps; use GAE with λ=0.95 for lower variance |
| **Overfitting to environment** | Too many steps for simple task | Reduce total steps; add regularization; use harder environment config |

### 15a.3 Reproducibility Protocol

```python
import torch
import numpy as np
import random
import gymnasium as gym

def set_seed(seed: int = 42):
    """Set all random seeds for full reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_env(env_id: str, seed: int):
    """Create seeded environment."""
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

# Full training loop template
def train_ppo(env_id="CartPole-v1", total_steps=1_000_000, seed=42):
    set_seed(seed)
    env = make_env(env_id, seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Policy network
    policy = torch.nn.Sequential(
        torch.nn.Linear(obs_dim, 256), torch.nn.Tanh(),
        torch.nn.Linear(256, 256), torch.nn.Tanh(),
        torch.nn.Linear(256, act_dim),
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    # Training loop
    obs, _ = env.reset()
    episode_rewards = []
    ep_reward = 0

    for step in range(total_steps):
        # Action selection
        logits = policy(torch.FloatTensor(obs))
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample().item()

        # Environment step
        next_obs, reward, terminated, truncated, _ = env.step(action)
        ep_reward += reward
        done = terminated or truncated

        if done:
            episode_rewards.append(ep_reward)
            ep_reward = 0
            obs, _ = env.reset()
        else:
            obs = next_obs

        # Logging
        if (step + 1) % 10000 == 0:
            avg = np.mean(episode_rewards[-10:]) if episode_rewards else 0
            print(f"Step {step+1}: Avg reward (last 10) = {avg:.1f}")

    return episode_rewards

# Run with reproducibility
rewards = train_ppo(seed=42)
assert abs(np.mean(rewards[-10:]) - 500) > 1, "Verify reproduction"
```

### 15a.4 Production Deployment Checklist

| Concern | Check | Tool / Method |
|---------|-------|---------------|
| **Model format** | Export to ONNX or TorchScript | `torch.onnx.export()`, `torch.jit.trace()` |
| **Inference latency** | < 10ms per action | ONNX Runtime, TensorRT, NVIDIA Triton |
| **Batching** | Handle multiple environment instances | Ray Serve, TorchServe with dynamic batching |
| **Monitoring** | Action distribution drift | KS-test on action histogram vs training distribution |
| **Safety guardrails** | Action masking, reward clipping | Custom wrapper around policy output |
| **Fallback** | When model produces NaN or invalid action | Rule-based safe action; raise alert |
| **A/B testing** | Compare new policy vs baseline | Online evaluation with holdout traffic (e.g., 5% of requests) |
| **Rollback** | Quickly revert to previous policy | Versioned model registry (MLflow, W&B Model Registry) |
| **Data pipeline** | Log environment interactions for retraining | Structured logging (JSON) to S3/GCS; daily retraining |
| **Bias monitoring** | Check for demographic fairness | Slice-based evaluation on protected attributes |

### 15a.5 Resource Estimation

| Task | Total Steps | Wall Time | Hardware | Cost (Cloud) |
|------|:-----------:|:---------:|:--------:|:-----------:|
| CartPole (simple) | 100K | 2 min | 1 CPU core | $0.01 |
| MuJoCo (continuous) | 1M | 30 min | 4 CPU + GPU | $0.50 |
| Atari game (DQN) | 10M | 4 hours | 8 CPU + GPU | $10 |
| Atari (Rainbow) | 50M | 20 hours | 8 CPU + GPU | $50 |
| Dota 2 (OpenAI Five) | 900M×yr | Continuous | 128K CPU + 256 GPU | $10M+ |
| Robotics sim (Isaac) | 100M | 24 hours | 32 CPU + 8 GPU | $200 |

---

## 15b. RL in Production: Case Studies and Lessons Learned

Real-world RL deployments reveal a significant gap between academic benchmarks and production systems. This section catalogs documented production RL deployments, their architectures, and the lessons learned.

### 15b.1 Algorithm Selection Guide for Production

| Use Case | Recommended Algorithm | Why | Avoid |
|----------|---------------------|-----|-------|
| **Discrete action (game, recommendation)** | DQN / Rainbow (if 100K+ env steps) | Sample efficient off-policy | Pure policy gradient (too sample-inefficient) |
| **Continuous control (robotics)** | SAC or TD3 | Off-policy + maximum entropy | PPO (on-policy is too expensive for real hardware) |
| **General purpose, sim available** | PPO | Stable, well-understood, good defaults | SAC (harder to tune for discrete actions) |
| **LLM alignment / RLHF** | PPO + KL penalty, or GRPO | KL prevents collapse; GRPO removes critic | DQN (action space too large) |
| **Real-time bidding / ads** | Contextual bandit (LinUCB, Thompson) | Only need immediate reward | Full RL (delayed reward not needed) |
| **Supply chain optimization** | Model-based RL (Dyna, MuZero) | Sample efficient with learned model | Model-free (too many env interactions needed) |
| **Autonomous driving** | IL + RL (DAgger, PPO with safety constraints) | Imitation to bootstrap, RL to improve | Pure RL from scratch (unsafe exploration) |
| **Multi-agent systems** | MAPPO, QMIX | Stable multi-agent coordination | Independent DQN (non-stationarity issues) |

### 15b.2 Production Case Studies

| Company / System | Domain | Algorithm | Scale | Key Metric | Lesson Learned |
|------------------|--------|-----------|:-----:|:----------:|----------------|
| **Google DeepMind (Data Centers)** | Cooling optimization | Deep RL (DQN variant) | 120+ data centers | 40% reduction in cooling energy | The biggest gains came from RL exploring *non-intuitive* strategies (e.g., running some servers hotter than engineers deemed comfortable) |
| **Waymo** | Autonomous driving | PPO + IL + safety layer | 20M+ public road miles | 99.9%+ disengagement-free miles | Safety-critical RL needs a rule-based fallback; pure RL exploration on public roads is unacceptable |
| **YouTube Recommendations** | Content recommendation | Deep Q-Network + SlateQ | 2B+ monthly users | 1-2% improvement in watch time | Offline evaluation (simulated rewards) did NOT match online A/B tests — offline eval was overly optimistic |
| **Spotify Discover Weekly** | Music recommendation | Contextual bandit (Thompson sampling) | 40M+ weekly users | 25% increase in discovery rate | Bandit simplicity beats complex RL when the time horizon is short and exploration is cheap |
| **Alibaba E-Commerce** | Product recommendation | MARL (multi-agent PPO) | 500M+ users | 7.5% GMV improvement | Multi-agent coordination beats independent agents in marketplaces where products compete for attention |
| **Jasper / Microsoft** | RLHF for LLM alignment | PPO + RM + KL penalty | Millions of prompts | Improved helpfulness scores by 30%+ | KL penalty tuning was the most critical hyperparameter — too low caused capability regression, too high prevented alignment |
| **NVIDIA DRIVE** | Autonomous simulation | PPO with domain randomization | Billions of sim miles | 90% reduction in real-world testing | Sim-to-real transfer works when domain randomization thoroughly covers the sim-reality gap |
| **UC Berkeley Robot Learning** | Real robot manipulation | SAC + sim-to-real | 10+ robot platforms | 80% success on novel objects | SAC's off-policy nature was essential — they could not afford the 10M+ on-policy interactions PPO would need |

### 15b.3 Common Failure Modes in Production RL

| Failure Mode | Symptom | Root Cause | Mitigation |
|-------------|---------|------------|------------|
| **Reward hacking** | Agent finds unintended shortcut to high reward | Poorly specified reward function | Reward shaping audits; adversarial reward testing |
| **Distribution shift at deployment** | Policy performs well in training, poorly in production | Training environment doesn't match deployment | Domain randomization; online adaptation; gradual rollouts |
| **Catastrophic forgetting** | Policy loses previously learned capabilities as it learns new ones | Non-stationary learning target | Replay buffers; elastic weight consolidation; multi-task training |
| **Exploration hazard** | Agent tries dangerous actions during learning | Naive exploration in safety-critical domain | Action masking; safety layers; constrained MDP |
| **Evaluation gap** | Offline metrics show improvement but online A/B test shows degradation | Simulated evaluation doesn't capture real user behavior | Live A/B testing; counterfactual evaluation (e.g., IPS estimators) |
| **Latency constraint violation** | Policy inference takes too long for real-time decision | Complex neural network inference | Model distillation; quantization; ONNX runtime; fixed compute budget |
| **Non-stationary environment shift** | User behavior changes → previously optimal policy degrades | Environment dynamics drift over time | Online fine-tuning; bandit-style exploration; periodic retraining |

### 15b.4 Production RL Infrastructure Checklist

| Component | Production Requirement | Recommendation |
|-----------|----------------------|----------------|
| **Environment** | Fast simulation or logged data | C++ simulator for speed; replay buffer for offline data; hybrid for sim-to-real |
| **Inference** | Low latency (<10ms), high throughput | ONNX Runtime / TensorRT for GPU; ONNX Runtime for CPU; batch inference where possible |
| **Training** | Distributed, fault-tolerant | Ray RLlib for production; custom infrastructure for hyperscale |
| **Monitoring** | Reward distribution, drift, safety metrics | Prometheus + Grafana dashboards; alert on reward mean dropping >2σ |
| **Rollback** | Quick revert to previous policy | Versioned model registry (MLflow); shadow deployment before full rollout |
| **Safety** | Constraint satisfaction | Action clipping; safety critic; human-in-the-loop for critical decisions |
| **Data pipeline** | Log all (s,a,r,s') for retraining | Structured logging to S3/GCS; daily retraining on fresh data |

---

## 16. Cross-References

| Reference | Description |
|-----------|-------------|
| [07-Emerging/02-AI-Safety.md] | RLHF for alignment, reward hacking |
| [01-Foundations/02-Machine-Learning.md] | ML foundations |
| [01-Foundations/05-Training-Methodologies.md] | Training methods including DPO variants |
| [01-Foundations/04-Data-Engineering.md] | Preference data for RLHF |
| [06-Advanced/01-Multimodal-AI.md] | Robotics, vision-language-action models |
| [08-Reference/01-Glossary.md] | Key RL terms |
| [01-Foundations/06-Reinforcement-Learning.md] | This document — RL foundations and practical guide |
| [06-Advanced/03-Evaluation-Benchmarks.md] | RL evaluation benchmarks |
| [01-Foundations/03-Deep-Learning.md] | Deep learning foundations for RL |

---

*Document version: 2.5 — June 2026 | Tier 3: Expansion. [Added: §15a Practical RL Project Guide, §15b Production RL Case Studies (algorithm selection guide, production case studies, failure modes, infrastructure checklist). Updated Cross-References.]* — June 2026 | Tier 3: Expansion. [Added: §15a Practical RL Project Guide — project setup checklist, debugging scenarios, reproducibility protocol with code example, production deployment checklist, resource estimation. Updated Cross-References.]*
