# AI for Gaming — Core Topics

> **Category 61 — AI Knowledge Library** | This document maps the conceptual core of game AI: behavior systems, procedural content generation, player modeling, dynamic difficulty, narrative intelligence, automated playtesting, pathfinding, and economy modeling. It is the "what and why" layer that [03-Technical-Deep-Dive.md](03-Technical-Deep-Dive.md) makes concrete with code.

---

## Table of Contents

1. [NPC Behavior Systems](#npc-behavior-systems)
2. [Procedural Content Generation](#procedural-content-generation)
3. [Player Modeling and Dynamic Difficulty](#player-modeling-and-dynamic-difficulty)
4. [Narrative and Dialogue Intelligence](#narrative-and-dialogue-intelligence)
5. [Automated Playtesting](#automated-playtesting)
6. [Pathfinding and Spatial AI](#pathfinding-and-spatial-ai)
7. [Economy and Balance Modeling](#economy-and-balance-modeling)
8. [Crowd and Faction Simulation](#crowd-and-faction-simulation)
9. [Cross-Cutting Concerns](#cross-cutting-concerns)
10. [Summary](#summary)

---

## NPC Behavior Systems

Non-player characters must perceive the world, decide actions, and execute them believably.

### Behavior Trees (BT)

The industry standard for hierarchical decision-making. Nodes: Selector, Sequence, Condition, Action, Decorator.

```python
# Pseudocode: a simple combat BT
Selector:
  Sequence:
    Condition: enemy_in_range()
    Action: attack()
  Sequence:
    Condition: low_health()
    Action: retreat()
  Action: patrol()
```

**Pros**: debuggable, predictable, designer-friendly. **Cons**: authoring cost grows with complexity; hard to get truly emergent behavior.

### Utility AI

Score every possible action; pick the highest. Great for "human-like" hesitation and continuous behavior.

```python
def score_actions(state):
    scores = {
        "attack": 0.3 * state.enemy_proximity + 0.5 * state.has_ammo,
        "flee":   0.8 * (1 - state.health_ratio),
        "heal":   0.9 * state.has_medkit * (1 - state.health_ratio),
    }
    return max(scores, key=scores.get)
```

**Pros**: smooth, tunable, scales to many considerations. **Cons**: many weights to tune; not emergent.

### Learned Policies (RL)

Replace hand-authored logic with a policy π(a|s) trained via reinforcement learning. See [01-Foundations/06-Reinforcement-Learning.md](../01-Foundations/06-Reinforcement-Learning.md).

| Approach | Pros | Cons | Best For |
|----------|------|------|----------|
| BT | Debuggable, predictable | Authoring cost | Scripted bosses |
| Utility | Smooth, tunable | Hard to scale | Crowd behavior |
| RL | Emergent, adaptive | Sample-inefficient, opaque | Competitive agents |
| LLM-driven | Conversational, varied | Latency, cost | Social NPCs |

### Hybrid: BT + Learned Subpolicy

Production games increasingly wrap a learned policy inside a BT action node — keeping global control readable while delegating micro-decisions to a network. This is the most common 2026 pattern for "smart but controllable" NPCs.

### Perception and Sensing

NPCs need a perception model: what they "see/hear." Typically a cone + occlusion check, or a learned detector.

```python
def perceive(world, npc):
    visible = [e for e in world.entities
               if in_cone(npc.pos, npc.facing, e.pos, fov=90)
               and line_of_sight(npc.pos, e.pos, world.obstacles)]
    return visible
```

---

## Procedural Content Generation

PCG creates game content algorithmically. Two families:

### Search-Based PCG

Use constraint solvers / evolutionary search to satisfy design rules.

```python
# Wave Function Collapse — tile-based level gen
def collapse(grid):
    cell = min_entropy_cell(grid)
    grid[cell] = sample_allowed(cell)
    propagate_constraints(grid, cell)
    return grid
```

### PCG via Machine Learning (PCGML)

Train on existing content, then generate new.

- **Level GANs**: generator produces tile maps; discriminator enforces playability.
- **LLM level design**: prompt a model with constraints ("3-star difficulty, 2 exits").
- **Diffusion for textures/3D**: text-to-asset pipelines.

| Method | Control | Fidelity | Training Need |
|--------|---------|----------|---------------|
| L-systems | High | Low–Med | None |
| WFC | Med | Med | None |
| GAN | Low | High | Large dataset |
| Diffusion | Med (prompt) | Very High | Large dataset |
| LLM | High (prompt) | Med | Fine-tune |

**Validation is critical**: generated content must be *playable*. Always run a solver/playability check and resample on failure. See also [Synthetic Data Generation (51)](../51-Synthetic-Data-Generation/01-Overview.md).

### Quest and Item Generation

- **Quest graphs**: sample a directed graph of objectives with balance constraints.
- **Item rolls**: control loot distributions to keep economy sane (see Economy below).

---

## Player Modeling and Dynamic Difficulty

### Player Modeling

Infer skill, style, and engagement from telemetry.

```python
# Feature vectors from telemetry
features = {
    "aim_accuracy": rolling_mean(hits/shots, window=50),
    "reaction_ms": median(time_to_first_shot),
    "exploration": unique_rooms_visited / total_rooms,
}
skill = logistic_regression.predict(features)  # 0..1
```

Techniques: supervised learning (skill prediction), clustering (archetype discovery), sequence models (churn prediction).

### Dynamic Difficulty Adjustment (DDA)

A control loop that nudges parameters to keep players in flow.

```python
# Simple DDA controller
if player.win_rate > 0.7:
    enemy_damage *= 1.05
elif player.win_rate < 0.3:
    enemy_damage *= 0.95
```

Multi-armed bandits are preferred when there are many tunable knobs and sparse reward — they explore configurations while exploiting the best.

> **Ethics note**: adaptive difficulty that *hides* its adjustments is fine; difficulty that manipulates spending crosses into [Responsible AI (55)](../55-AI-Ethics-and-Responsible-AI/01-Overview.md) territory.

---

## Narrative and Dialogue Intelligence

### LLM-Driven NPCs

Ground dialogue in lore with [RAG (04)](../04-RAG/01-RAG-Architectures.md) to avoid hallucinated lore.

```python
# Retrieval-augmented NPC response
context = vector_db.search(query=player_utterance, top_k=4,
                           filter={"world": "Eldoria"})
prompt = f"Lore:\n{context}\n\nPlayer: {player_utterance}\nNPC:"
response = llm.generate(prompt, max_tokens=120, temperature=0.7)
```

### Branching Narrative Planning

Use a planner (e.g., STRIPS-style) over story beats to keep coherence across player choices.

### Guardrails

- Keep a deterministic "character card" (tone, constraints).
- Use a classifier to reject out-of-character or unsafe lines ([18](../18-Agent-Security-and-Trust/01-Overview.md)).
- Cache common responses for latency.
- Use a [Small Language Model (30)](../30-Small-Language-Models/01-Overview.md) on-device for cost.

Related: [Agent Memory Systems (32)](../32-Agent-Memory-Systems/01-Overview.md) for long-term NPC recall.

---

## Automated Playtesting

Replace manual QA with agents that *play*.

### Bot Typologies

| Bot Type | Method | Use |
|----------|--------|-----|
| Scripted | Hard-coded paths | Smoke tests |
| RL agent | PPO/SAC | Explore states |
| Fuzzer | Random inputs | Crash detection |
| Imitation | From human replays | Realistic coverage |

### Coverage Metrics

- State-space coverage (rooms, branches reached)
- Crash/exception rate per build
- Balance anomalies (e.g., an item that trivializes the game)
- Regression flags vs previous build

This is the gaming analogue of [Agent Evaluation (20/04)](../20-Agent-Infrastructure-and-Observability/04-Agent-Evaluation-and-Testing.md).

### Integrating into CI

```
build → launch 200 bots × 50 episodes → aggregate telemetry → gate release
```

---

## Pathfinding and Spatial AI

- **A\*** on grids; **Navigation Meshes (NavMesh)** for 3D.
- **Hierarchical pathfinding (HPA\*)** for large worlds.
- **Learned planners**: graph neural networks predict waypoints.

```python
def a_star(start, goal, neighbors, cost):
    frontier = [(0, start)]; came_from = {}
    g = {start: 0}
    while frontier:
        _, cur = heappop(frontier)
        if cur == goal: return reconstruct(came_from, cur)
        for nxt in neighbors(cur):
            ng = g[cur] + cost(cur, nxt)
            if ng < g.get(nxt, inf):
                g[nxt] = ng; came_from[nxt] = cur
                heappush(frontier, (ng + heur(nxt, goal), nxt))
```

### Local Avoidance

A* gives a path; **steering behaviors / RVO** handle dynamic collision with other agents.

---

## Economy and Balance Modeling

Live-service games are economies. Model them like macro-systems:

- **Agent-based simulation**: thousands of simulated players spend/craft/trade.
- **What-if analysis**: "If we nerf resource X, does inflation spike?"
- **Reinforcement-learning tuners** adjust drop rates toward target curves.

```python
# Toy economy simulation step
for agent in agents:
    decision = policy(agent.state)        # craft / sell / hoard
    agent.state = env.step(agent.state, decision)
inflation = mean(price_index) / baseline
```

### Balance Levers

| Lever | Effect |
|-------|--------|
| Drop rate | Item scarcity |
| Craft cost | Sink pressure |
| Vendor price | Currency sink |
| Respawn time | PvP pacing |

---

## Crowd and Faction Simulation

Large scenes (battles, cities) need cheap, believable masses:

- **Boids / steering** for flocks.
- **Utility + BT** for individual agents in a crowd.
- **Faction AI**: model groups with shared goals; [Multi-Agent Systems (03/02)](03-Agents/02-Multi-Agent-Systems.md) applies directly.

---

## Cross-Cutting Concerns

- **Latency budget**: NPC inference must fit frame budget (often < 30 ms).
- **Determinism**: replay/esports require reproducible behavior — rules out pure sampling in competitive modes.
- **Cost**: LLM NPCs cost per token; [Cost Optimization (41)](../41-AI-Cost-Optimization-and-Enterprise-ROI/01-Overview.md) applies directly.
- **Safety**: See [Agent Security (18)](../18-Agent-Security-and-Trust/01-Overview.md) for prompt-injection risks in player-facing LLMs.
- **Privacy**: Telemetry-based player models must respect [Data Sovereignty (40)](../40-AI-Data-Sovereignty-and-Privacy/01-Overview.md).

---

## Summary

| Topic | Primary Technique | Library Cross-Ref |
|-------|-------------------|-------------------|
| NPC behavior | RL / BT / LLM | 03, 06 |
| PCG | Search / GAN / Diffusion / LLM | 51 |
| Player model | Supervised / clustering | 01 |
| DDA | Control / bandits | 06 |
| Narrative | LLM + RAG | 04, 32 |
| Playtesting | RL bots / fuzzers | 20 |
| Pathfinding | A* / NavMesh / GNN | 07 |
| Economy | ABM / RL tuning | 41 |
| Crowds | Boids / utility | 03 |

---

*Core topics complete. Proceed to [03-Technical-Deep-Dive.md](03-Technical-Deep-Dive.md).*
