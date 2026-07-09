# AI for Gaming — Technical Deep Dive

> **Category 61 — AI Knowledge Library** | Hands-on implementations: training an RL agent to play a game, building a PCG pipeline, wiring an LLM NPC with RAG, running automated playtests, and deploying on-device. Code is illustrative Python/PyTorch-style pseudocode meant to be adapted to real engines.

---

## Table of Contents

1. [Environment Interface](#environment-interface)
2. [RL Agent: PPO for a Grid Game](#rl-agent-ppo-for-a-grid-game)
3. [Reward Shaping](#reward-shaping)
4. [Self-Play and Population-Based Training](#self-play-and-population-based-training)
5. [Procedural Level Generation with WFC](#procedural-level-generation-with-wfc)
6. [PCGML: Training a Level GAN](#pcgml-training-a-level-gan)
7. [LLM NPC with Retrieval Grounding](#llm-npc-with-retrieval-grounding)
8. [Automated Playtest Harness](#automated-playtest-harness)
9. [Performance and On-Device Inference](#performance-and-on-device-inference)
10. [Evaluation Methodology](#evaluation-methodology)
11. [Worked Mini-Project](#worked-mini-project)

---

## Environment Interface

Standardize the game as an RL environment (Gymnasium-style):

```python
class GameEnv:
    def reset(self): ...
    def step(self, action): ...
    def render(self): ...
    # obs: dict | action: int/continuous | reward: float | done: bool
```

```python
import gymnasium as gym
class DungeonEnv(gym.Env):
    def __init__(self): self.action_space = gym.spaces.Discrete(4)
    def reset(self, seed=None): ...
    def step(self, action): ...
```

---

## RL Agent: PPO for a Grid Game

A minimal PPO actor-critic skeleton.

```python
import torch, torch.nn as nn

class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU())
        self.actor = nn.Linear(128, act_dim)
        self.critic = nn.Linear(128, 1)
    def forward(self, x):
        h = self.net(x)
        return torch.softmax(self.actor(h), -1), self.critic(h)

# Training loop (conceptual)
for epoch in range(1000):
    rollout = collect(env, policy, steps=2048)
    advantages = compute_gae(rollout, gamma=0.99, lam=0.95)
    for _ in range(4):  # multiple epochs per batch
        loss = ppo_loss(policy, rollout, advantages, clip=0.2)
        loss.backward(); opt.step()
```

Key hyperparameters: `gamma`, `lambda (GAE)`, `clip`, `entropy_coef`. See [Reinforcement Learning (01/06)](../01-Foundations/06-Reinforcement-Learning.md).

### Using Stable-Baselines3

```python
from stable_baselines3 import PPO
model = PPO("MlpPolicy", DungeonEnv(), verbose=1,
            n_steps=2048, batch_size=64, gamma=0.99)
model.learn(total_timesteps=1_000_000)
```

---

## Reward Shaping

Reward design matters more than model size:

```python
def reward(state, prev):
    r = 0.0
    r += 1.0 if state.killed_enemy else 0.0
    r -= 0.01                      # time penalty (avoid dawdling)
    r += 0.5 * state.health_gain
    r -= 2.0 if state.died else 0.0
    return r
```

**Potential-based shaping** avoids reward hacking:

```
F(s, s') = γ * Φ(s') − Φ(s)
```

where Φ is a hand-designed "progress" potential (e.g., distance to goal).

---

## Self-Play and Population-Based Training

Competitive NPCs improve by playing against copies of themselves.

```python
# League-based self-play
for iteration in range(N):
    new_agent = mutate(best_agent)
    for opponent in league.sample(k=5):
        train(new_agent, vs=opponent)     # maximizes win-rate vs league
    league.add(new_agent)
    best_agent = select(league)           # by ELO
```

This produces robust, non-exploitable behavior — the technique behind landmark results on StarCraft and MOBA titles. Combine with **policy snapshots** (a library of past versions) to prevent cyclic forgetting.

---

## Procedural Level Generation with WFC

Wave Function Collapse enforces local adjacency constraints.

```python
def wfc(tiles, adjacency, grid_size):
    grid = [[None]*grid_size for _ in range(grid_size)]
    while not all(all(c is not None for c in row) for row in grid):
        x, y = min_entropy_cell(grid)
        grid[x][y] = choose(entropy[x][y])
        propagate(grid, x, y, adjacency)   # remove contradictory options
    return grid
```

**Playability gate**:

```python
level = wfc(tiles, adjacency, 32)
while not solvable(level):                # run a solver
    level = wfc(tiles, adjacency, 32)
```

| Approach | Determinism | Controllability | Compute |
|----------|-------------|-----------------|---------|
| WFC | Yes | Medium | Low |
| VAE sample | No | Low–Med | Med |
| Diffusion | No | High (prompt) | High |
| LLM | No | High (prompt) | High |

---

## PCGML: Training a Level GAN

```python
import torch, torch.nn as nn

class LevelGAN(nn.Module):
    def __init__(self, z=32, size=32):
        super().__init__()
        self.G = nn.Sequential(nn.Linear(z, 256), nn.ReLU(),
                              nn.Linear(256, size*size), nn.Sigmoid())
        self.D = nn.Sequential(nn.Linear(size*size, 256), nn.ReLU(),
                              nn.Linear(256, 1), nn.Sigmoid())
    # train with BCE loss on real (from dataset) vs fake (G(z))
```

After training, sample `z`, decode, and **validate playability** before accepting.

---

## LLM NPC with Retrieval Grounding

Prevent lore hallucinations and keep characters consistent.

```python
from rag import VectorStore   # see Category 04

lore = VectorStore.load("world_eldoria")
CHAR_CARD = ("You are Borin, a gruff dwarf blacksmith. "
             "Never discuss modern technology. Speak in short, blunt sentences.")

def npc_reply(player_text: str) -> str:
    ctx = lore.search(player_text, top_k=4)
    prompt = (f"{CHAR_CARD}\n\nLore:\n{ctx}\n\nPlayer: {player_text}\nBorin:")
    raw = llm.generate(prompt, max_tokens=100, temperature=0.6)
    if safety_classifier.flag(raw):     # see Category 18
        return "Borin grunts and says nothing."
    return raw
```

Caching frequent queries and using a [Small Language Model (30)](../30-Small-Language-Models/01-Overview.md) on-device keeps latency and cost down.

### Memory Across Sessions

Attach a persistent memory store ([32](../32-Agent-Memory-Systems/01-Overview.md)) so Borin remembers the player's past deeds.

---

## Automated Playtest Harness

Spin up many parallel game instances driven by bots; collect telemetry.

```python
def playtest(build, n_bots=200, episodes=50):
    results = []
    for bot in spawn_bots(n_bots):
        for _ in range(episodes):
            env = GameEnv(build)
            traj = bot.play(env)                 # RL or scripted
            results.append({
                "coverage": state_coverage(traj),
                "crashes": env.exception_count,
                "win_rate": traj.won,
                "balance_flags": detect_anomalies(traj),
            })
    return aggregate(results)
```

Wire this into CI so every build is auto-played before shipping — the gaming equivalent of [AgentOps tracing (20/03)](../20-Agent-Infrastructure-and-Observability/03-Agent-Tracing-and-Observability.md).

---

## Performance and On-Device Inference

Console/phone NPCs must hit tight latency budgets.

- Quantize policies to INT8 (see [Quantization (02/04)](../02-LLMs/04-Quantization.md)).
- Batch inference across NPCs sharing a frame.
- Use [Edge AI (62)](../62-Edge-AI-and-On-Device-Inference/01-Overview.md) patterns for mobile.

```
Frame budget (33 ms @ 30fps)
├─ Render       18 ms
├─ Physics       6 ms
└─ AI inference  9 ms  ← shared batch, INT8
```

---

## Evaluation Methodology

Measure what players feel, not just win-rate:

| Metric | How | Target |
|--------|-----|--------|
| Believability | Human rating study | > 4/5 |
| Reactivity | Response-to-event latency | < 250 ms |
| Coverage | States reached by bots | > 90% |
| Robustness | Crash rate / 10k episodes | < 0.1% |
| Balance | Win-rate spread | 35–65% |
| Cost | Tokens or FLOPs per session | budgeted |

Tie results back to [Agent Evaluation (20/04)](../20-Agent-Infrastructure-and-Observability/04-Agent-Evaluation-and-Testing.md).

---

## Worked Mini-Project

Build a "Tiny Dungeon" with:
1. A grid `GameEnv` (Python).
2. A PPO policy trained to reach the exit.
3. WFC to generate rooms.
4. An LLM "guide" NPC grounded in a 10-line lore file.

This exercises every pillar in this category and connects to [Agent Architectures (03/01)](../03-Agents/01-Agent-Architectures.md).

### Starter skeleton

```python
env = DungeonEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(500_000)
level = generate_wfc(32)
print(npc_reply("Where is the forge?"))
```

---

*Deep dive complete. See [04-Tools-and-Frameworks.md](04-Tools-and-Frameworks.md).*
