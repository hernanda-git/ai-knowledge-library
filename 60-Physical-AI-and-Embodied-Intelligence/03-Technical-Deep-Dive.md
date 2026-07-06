# Physical AI and Embodied Intelligence: Technical Deep Dive

> **Category:** 60 — Physical AI and Embodied Intelligence  
> **Last Updated:** July 2026  
> **Cross-references:** [02-Core-Topics.md](./02-Core-Topics.md), [04-Tools-and-Frameworks.md](./04-Tools-and-Frameworks.md)

---

## Table of Contents

1. [Vision-Language-Action (VLA) Models](#1-vision-language-action-vla-models)
2. [Robot Learning at Scale](#2-robot-learning-at-scale)
3. [Sim-to-Real Transfer Deep Dive](#3-sim-to-real-transfer-deep-dive)
4. [Dexterous Manipulation](#4-dexterous-manipulation)
5. [Whole-Body Control](#5-whole-body-control)
6. [Real-Time Inference on Robots](#6-real-time-inference-on-robots)
7. [Safety-Critical Control](#7-safety-critical-control)
8. [Multi-Modal Perception](#8-multi-modal-perception)

---

## 1. Vision-Language-Action (VLA) Models

### 1.1 Architecture Deep Dive

VLA models represent the fusion of vision-language models with robot action generation. Unlike traditional robotics pipelines that separate perception, planning, and control, VLA models learn end-to-end mappings from raw sensory input to motor commands.

```
DETAILED VLA ARCHITECTURE:
─────────────────────────────────────────────────────────────
                    ┌─────────────────────┐
                    │   Language Instruction│
                    │  "Pick up the red   │
                    │   cup and place it   │
                    │   on the shelf"      │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │   Language Encoder   │
                    │  (LLaMA-3, T5, etc.)│
                    │  → Token Embeddings  │
                    └─────────┬───────────┘
                              │
    ┌──────────────┐          │          ┌──────────────┐
    │ Camera Image │          │          │ Proprioception│
    │  (224×224)   │          │          │ (Joint angles)│
    └──────┬───────┘          │          └──────┬───────┘
           │                  │                 │
    ┌──────▼───────┐          │          ┌──────▼───────┐
    │ Vision Encoder│          │          │ State Encoder │
    │  (ViT-L/14)  │          │          │  (MLP)        │
    │  → Patches    │          │          │  → Embedding  │
    └──────┬───────┘          │          └──────┬───────┘
           │                  │                 │
           └──────────┬───────┴────────┬────────┘
                      │                │
              ┌───────▼────────────────▼───────┐
              │     Multimodal Transformer      │
              │     (Attention Fusion)           │
              │     12-24 layers, 768-1024 dim  │
              └───────────────┬────────────────┘
                              │
              ┌───────────────▼────────────────┐
              │      Action Decoder             │
              │  (Action Tokens → Motor Commands)│
              │  Discretized into 256 bins      │
              └───────────────┬────────────────┘
                              │
              ┌───────────────▼────────────────┐
              │   Action Head (MLP)             │
              │   → Joint velocities [7-DOF]    │
              │   → Or End-effector [x,y,z,rx,ry,rz]│
              └────────────────────────────────┘
─────────────────────────────────────────────────────────────
```

### 1.2 Action Tokenization

A key innovation in VLA models is discretizing continuous robot actions into tokens:

**Action Space Design:**

| Approach | Representation | Pros | Cons |
|----------|---------------|------|------|
| Continuous | Direct regression | Simple | Distribution shift |
| Discretized | 256-bin tokens | Stable training | Quantization error |
| Diffusion | Iterative denoising | Multimodal actions | Slow inference |
| Flow matching | Continuous normalizing flows | Fast, accurate | Complex training |

**Token vocabulary:**
```
ACTION TOKEN REPRESENTATION:
─────────────────────────────────────────────────────────────
Each action dimension is discretized into 256 bins:
  bin_idx = round((action - min) / (max - min) × 255)

Example for 7-DOF robot arm:
  Action = [0.12, -0.34, 0.56, 0.78, -0.12, 0.34, 0.01]
  Tokens = [83, 57, 158, 199, 71, 143, 80]

Action dimension ranges:
  Joint 1: [-3.14, 3.14] rad
  Joint 2: [-2.00, 2.00] rad
  ...
  Joint 7: [-3.14, 3.14] rad
─────────────────────────────────────────────────────────────
```

### 1.3 Training Pipeline

**Stage 1: Pre-training on Internet Data**
- Vision-language alignment (CLIP, SigLIP)
- Language model pre-training (LLaMA, T5)
- Web-scale image-text pairs (billions of examples)

**Stage 2: Robot Data Fine-tuning**
- Cross-embodiment robot datasets (Open X-Embodiment)
- Mixed robot types: arms, humanoids, mobile robots
- Multi-task learning across manipulation skills

**Stage 3: Task-Specific Adaptation**
- Fine-tune on target robot and task
- Few-shot adaptation (10-50 demonstrations)
- Domain-specific data augmentation

### 1.4 Key VLA Models in 2026

| Model | Developer | Parameters | Key Innovation |
|-------|----------|-----------|----------------|
| π0 | Physical Intelligence | 3B | Cross-embodiment, general manipulation |
| RT-2 | Google DeepMind | 55B | Web knowledge transfer |
| GR00T N1 | NVIDIA | 7B | Sim-to-real integrated |
| Octo | Berkeley Open X | 93M | Open-source, multi-task |
| HPT | MIT | 100M | Heterogeneous pretrained transformers |
| OpenVLA | Berkeley | 7B | Open-source VLA |

### 1.5 Code Example: Simple VLA Inference

```python
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

# Load pre-trained VLA model
model = AutoModelForVision2Seq.from_pretrained("physical-intelligence/pi0")
processor = AutoProcessor.from_pretrained("physical-intelligence/pi0")

# Process inputs
image = processor.load_image("camera_frame.jpg")
instruction = "Pick up the red cup and place it on the shelf"
proprioception = get_joint_positions()  # Current robot state

inputs = processor(
    images=image,
    text=instruction,
    proprioception=proprioception,
    return_tensors="pt"
)

# Generate action sequence
with torch.no_grad():
    action_tokens = model.generate(
        **inputs,
        max_new_tokens=32,  # 4 action steps × 7 DOF = 28 tokens
        temperature=0.1
    )

# Decode tokens to continuous actions
actions = processor.decode_actions(action_tokens)
# actions shape: [4, 7] - 4 future actions, 7 DOF each

# Execute on robot
for action in actions:
    robot.set_joint_velocities(action)
    time.sleep(0.025)  # 40 Hz control loop
```

---

## 2. Robot Learning at Scale

### 2.1 Open X-Embodiment Dataset

The largest robot learning dataset ever assembled:

```
OPEN X-EMBODIMENT STATISTICS:
─────────────────────────────────────────────────────────────
Total episodes:           1,000,000+
Robot types:              22
Tasks:                    500+
Institutions:             20+
Languages:                English, Japanese, German
Data volume:              ~3 TB
─────────────────────────────────────────────────────────────
```

**Included Robot Platforms:**
- Franka Emika Panda (most common)
- WidowX
- Google Robot (RT-1 robot)
- Kuka LBR iiwa
- UR5/UR10
- Mobile ALOHA
- Various custom platforms

**Data Format:**
```json
{
  "episode_id": "2026-01-15_143022",
  "robot_type": "franka_panda",
  "task": "pick_up_red_cup",
  "language_instruction": "Pick up the red cup",
  "observations": {
    "images": {"front": "cam_front.jpg", "wrist": "cam_wrist.jpg"},
    "joint_positions": [0.12, -0.34, 0.56, 0.78, -0.12, 0.34, 0.01],
    "gripper_position": 0.08
  },
  "actions": [
    {"joint_velocities": [0.01, -0.02, 0.03, 0.04, -0.01, 0.02, 0.001],
     "gripper_action": -1.0}
  ],
  "metadata": {
    "collection_method": "teleoperation",
    "demonstrator_skill": "expert",
    "success": true
  }
}
```

### 2.2 Scaling Laws for Robot Learning

Key findings from large-scale robot learning:

**Data Scaling:**
- Performance improves logarithmically with dataset size
- ~10K episodes needed for reliable single-task performance
- ~100K+ episodes for multi-task generalization
- Diminishing returns beyond 1M episodes without diversity

**Model Scaling:**
- VLA models show emergent capabilities above 1B parameters
- Cross-embodiment transfer improves with model size
- 7B parameter models show strong zero-shot transfer
- 55B+ models (like RT-2) enable web-knowledge transfer

**Compute Scaling:**
- Training cost: ~$100K-500K for a 7B VLA model
- Inference: 5-50 Hz on NVIDIA Jetson Orin
- Edge deployment requires model distillation (7B → 1-2B)

### 2.3 Transfer Learning Across Robots

A major challenge: can a model trained on one robot work on another?

**Cross-Embodiment Strategies:**

1. **Action space normalization**: Map all robots to a canonical action space
2. **Embodiment tokens**: Include robot identity as an input token
3. **Partial sharing**: Share vision/language encoders, use robot-specific action heads
4. **Adapters**: Small robot-specific adapter layers

**Performance by transfer type:**
| Transfer | Performance | Notes |
|----------|------------|-------|
| Same robot, new task | 85-95% | Strong generalization |
| Similar robot, same task | 70-85% | Good transfer |
| Different robot, same task | 50-70% | Moderate transfer |
| Different robot, different task | 30-50% | Weak transfer |

### 2.4 Self-Supervised Robot Learning

Learning without explicit labels:

**Contrastive Learning:**
```python
# SimCLR-style pre-training for robot vision
def contrastive_loss(z1, z2, temperature=0.07):
    """z1, z2 are augmented views of the same observation"""
    similarity = torch.mm(z1, z2.T) / temperature
    labels = torch.arange(z1.shape[0])
    return nn.CrossEntropyLoss()(similarity, labels)

# Positive pairs: same observation, different augmentations
# Negative pairs: different observations
```

**Masked Prediction:**
```python
# MAE-style pre-training for robot proprioception
def masked_proprioception_loss(obs, mask_ratio=0.75):
    """Predict masked joint positions from visible ones"""
    mask = torch.rand(obs.shape) > mask_ratio
    visible = obs * mask
    predicted = model(visible)
    loss = F.mse_loss(predicted * ~mask, obs * ~mask)
    return loss
```

---

## 3. Sim-to-Real Transfer Deep Dive

### 3.1 Physics Simulation Fidelity

The accuracy of physics simulation determines sim-to-real success:

**Contact Modeling:**

```
CONTACT FORCE MODELS:
─────────────────────────────────────────────────────────────
1. Rigid Contact (LCP)
   - Instantaneous force application
   - No penetration allowed
   - Fast but unrealistic for manipulation

2. Soft Contact (Hunt-Crossley)
   - Penetration-based force
   - F = kδ^n + cδ̇δ^n
   - Better for manipulation tasks

3. Deformable Contact (FEM)
   - Finite element modeling
   - Accurate but computationally expensive
   - Used for cloth, soft objects

4. Particle-Based (MPM)
   - Material Point Method
   - Good for fluids, granular materials
   - Used in Genesis, NVIDIA Flex
─────────────────────────────────────────────────────────────
```

**Friction Models:**
- **Coulomb friction**: F ≤ μN (simple, widely used)
- **Stribeck friction**: Velocity-dependent (more realistic)
- **Anisotropic friction**: Direction-dependent (fabric, wood grain)
- **Lubricated friction**: Fluid film effects (oily surfaces)

### 3.2 Domain Randomization Strategies

Advanced randomization techniques:

**Curriculum Randomization:**
```
TRAINING CURRICULUM:
─────────────────────────────────────────────────────────────
Phase 1 (Epoch 0-100):    Easy randomization
  - Object mass: ±20%
  - Friction: ±10%
  - Lighting: ±10%

Phase 2 (Epoch 100-500): Medium randomization
  - Object mass: ±50%
  - Friction: ±30%
  - Lighting: ±30%
  - Camera noise: σ=5

Phase 3 (Epoch 500-1000): Hard randomization
  - Object mass: ±100%
  - Friction: ±50%
  - Lighting: ±50%
  - Camera noise: σ=15
  - Action delay: 0-50ms
─────────────────────────────────────────────────────────────
```

**Automatic Domain Randomization (ADR):**
- Start with minimal randomization
- Increase range when policy is robust
- Decrease when policy fails
- Automate the curriculum with heuristics

### 3.3 Teacher-Student Training

**Architecture:**
```
TEACHER-STUDENT PIPELINE:
─────────────────────────────────────────────────────────────
Teacher (in simulation):
  - Perfect state information
  - No noise, no delay
  - Access to object poses, contacts
  - Trained with RL (PPO, SAC)

Student (for real-world):
  - Only camera images + proprioception
  - Noisy, delayed observations
  - Distilled from teacher via:
    - Behavioral cloning
    - Policy distillation
    - Feature matching

Distillation Loss:
  L = α·L_action + β·L_feature + γ·L_reward
  where L_action = MSE(student_action, teacher_action)
        L_feature = MSE(student_features, teacher_features)
─────────────────────────────────────────────────────────────
```

### 3.4 Real-World Fine-Tuning Protocols

When simulation isn't sufficient:

| Protocol | Data Required | Training Time | Performance |
|----------|--------------|---------------|-------------|
| Zero-shot | None | 0 | 40-60% |
| Few-shot (10 demos) | 10 episodes | 30 min | 60-75% |
| Fine-tuning (100 demos) | 100 episodes | 2 hours | 75-90% |
| Online RL (1000 interactions) | 1000 steps | 4-8 hours | 85-95% |
| Full real training | 10K+ steps | Days | 90-98% |

---

## 4. Dexterous Manipulation

### 4.1 The Dexterous Manipulation Challenge

Dexterous manipulation is one of the hardest problems in robotics:

**Complexity factors:**
- High-dimensional action space (20+ DOF for a dexterous hand)
- Complex contact dynamics (multiple contact points, friction)
- Uncertainty in object properties (mass, friction, deformability)
- Real-time requirements (grasping decisions in milliseconds)

### 4.2 Hand Designs

| Hand | DOF | Fingers | Key Feature |
|------|-----|---------|-------------|
| Allegro Hand | 16 | 4 | Research standard |
| Shadow Hand | 24 | 5 | Most human-like |
| LEAP Hand | 12 | 4 | Low-cost, 3D printed |
| Robotiq 3-Finger | 6 | 3 | Industrial, robust |
| Barrett Hand | 4 | 3 | Adaptive grasping |
| Custom soft | Variable | Variable | Task-specific |

### 4.3 Dexterous Tasks and Benchmarks

**Standard Benchmarks:**

| Task | Description | Difficulty | State of Art |
|------|-------------|-----------|-------------|
| Pen rotation | Rotate pen in hand | Medium | 90% success |
| In-hand reorientation | Reposition arbitrary object | Hard | 70% success |
| Sphere rolling | Roll ball in palm | Medium | 85% success |
| Valve turning | Rotate valve to target | Medium | 95% success |
| Object sorting | Pick and sort objects | Hard | 60% success |
| Tool use | Grasp and use tool | Very Hard | 40% success |

### 4.4 Learning Dexterous Manipulation

**Reinforcement Learning Approach:**
```python
# PPO for dexterous manipulation
def dexterous_manipulation_reward(state, target_pose):
    """Reward function for in-hand reorientation"""
    # Position reward
    pos_error = torch.norm(state.object_pos - target_pose[:3])
    pos_reward = torch.exp(-10 * pos_error)
    
    # Orientation reward
    rot_error = rotation_distance(state.object_rot, target_pose[3:])
    rot_reward = torch.exp(-5 * rot_error)
    
    # Smoothness penalty
    action_smoothness = torch.norm(state.action - state.prev_action)
    smooth_penalty = -0.01 * action_smoothness
    
    # Contact reward (maintain contact with object)
    contact_reward = 0.1 * state.num_contacts
    
    return pos_reward + rot_reward + smooth_penalty + contact_reward
```

**Imitation Learning Approach:**
- Record human hand demonstrations with teleoperation
- Map human hand pose to robot hand kinematics
- Train behavioral cloning model
- Fine-tune with RL for robustness

---

## 5. Whole-Body Control

### 5.1 Humanoid Whole-Body Control

Controlling a humanoid robot requires coordinating:
- **Legs**: Locomotion, balance
- **Torso**: Posture, balance recovery
- **Arms**: Manipulation, gestures
- **Hands**: Dexterous manipulation
- **Head**: Camera gaze, human attention

### 5.2 Locomotion + Manipulation

The key challenge: walk AND manipulate simultaneously:

```
WHOLE-BODY CONTROL ARCHITECTURE:
─────────────────────────────────────────────────────────────
High-Level Task Planner
  ↓ "Walk to table, pick up cup, walk to sink"
  
Locomotion Controller (RL Policy)
  ↓ Joint commands for legs, torso
  
Manipulation Controller (VLA + RL)
  ↓ Joint commands for arms, hands
  
Whole-Body Coordination Layer
  → Balance compensation during reaching
  → CoM (Center of Mass) tracking
  → Angular momentum regulation
  
Low-Level Joint Controller (PD/Impedance)
  → Motor torques
─────────────────────────────────────────────────────────────
```

### 5.3 Balance During Manipulation

When a humanoid reaches for an object, it must:
1. **Shift CoM**: Move center of mass to maintain balance
2. **Counter-balance**: Use arms/legs for balance compensation
3. **Plan support polygon**: Ensure CoM stays within foot support
4. **Handle external forces**: Object weight affects balance

**Dynamic Balance Metrics:**
- **ZMP (Zero Moment Point)**: Must stay within support polygon
- **CoM velocity**: Must be controlled to prevent falling
- **Angular momentum**: Must be regulated during fast motions
- **Foot force distribution**: Must be stable and reasonable

### 5.4 Bimanual Manipulation

Using two arms simultaneously:

**Types:**
- **Symmetric**: Both arms do the same motion (carrying large object)
- **Asymmetric**: Arms have different roles (hold + manipulate)
- **Coordinated**: Arms work together toward a common goal (folding cloth)

**Challenges:**
- Synchronization between arms
- Collision avoidance between arms
- Shared workspace management
- Dual-arm force control

---

## 6. Real-Time Inference on Robots

### 6.1 Edge Computing Hardware

| Platform | TOPS | Power | Price | Use Case |
|----------|------|-------|-------|----------|
| NVIDIA Jetson Orin Nano | 40 | 15W | $249 | Entry-level robots |
| NVIDIA Jetson Orin NX | 100 | 25W | $599 | Mid-range robots |
| NVIDIA Jetson AGX Orin | 275 | 60W | $1,999 | High-performance robots |
| Qualcomm RB5 | 15 | 7W | $349 | Drones, small robots |
| Intel NUC | 10-50 | 28W | $500-1000 | General compute |
| Custom ASIC (e.g., Google TPU Edge) | 100+ | 10W | Varies | Optimized inference |

### 6.2 Model Optimization for Edge

| Technique | Speedup | Accuracy Loss | Implementation |
|-----------|---------|---------------|----------------|
| FP16 quantization | 2x | <1% | PyTorch native |
| INT8 quantization | 3-4x | 1-3% | TensorRT, ONNX |
| INT4 quantization | 5-8x | 3-10% | GPTQ, AWQ |
| Model pruning | 2-3x | 1-5% | Structured pruning |
| Knowledge distillation | 3-10x | 2-8% | Teacher-student |
| TensorRT optimization | 2-5x | <1% | NVIDIA SDK |

### 6.3 Inference Pipeline Timing

```
REAL-TIME ROBOT INFERENCE TIMELINE:
─────────────────────────────────────────────────────────────
Time (ms)  │ Operation
─────────────────────────────────────────────────────────────
0          │ Camera frame captured
1-3        │ Image preprocessing (resize, normalize)
3-8        │ Vision encoder forward pass
8-12       │ Language encoder forward pass
12-16      │ Multimodal fusion + action decoding
16-17      │ Action post-processing
17-18      │ Safety check (collision, joint limits)
18-20      │ Send to joint controller
20-25      │ Actuator response
─────────────────────────────────────────────────────────────
Total latency: ~25ms (40 Hz control loop)
```

### 6.4 Pipelining and Asynchrony

To achieve high control rates, operations are pipelined:

```python
import threading
import queue

class RobotInferencePipeline:
    def __init__(self, model, robot):
        self.model = model
        self.robot = robot
        self.observation_queue = queue.Queue(maxsize=2)
        self.action_queue = queue.Queue(maxsize=2)
        
    def perception_thread(self):
        """Runs at camera frame rate (30 Hz)"""
        while True:
            obs = self.robot.get_observation()
            if not self.observation_queue.full():
                self.observation_queue.put(obs)
    
    def inference_thread(self):
        """Runs as fast as possible"""
        while True:
            obs = self.observation_queue.get()
            action = self.model.predict(obs)
            if not self.action_queue.full():
                self.action_queue.put(action)
    
    def control_thread(self):
        """Runs at control rate (100-1000 Hz)"""
        while True:
            action = self.action_queue.get()
            self.robot.execute(action)
```

---

## 7. Safety-Critical Control

### 7.1 Safety Guarantees

Physical AI must provide formal safety guarantees:

**Control Barrier Functions (CBFs):**
```python
def control_barrier_function(state, control, safe_set):
    """
    Ensures the system stays within the safe set.
    
    h(x) >= 0 means state x is safe.
    Control must satisfy: dh/dx * f(x,u) >= -α(h(x))
    """
    h = safe_set.boundary_distance(state)
    dh_dx = safe_set.gradient(state)
    dynamics = robot.forward_dynamics(state, control)
    
    # CBF constraint: must be satisfied
    cbf_constraint = dh_dx @ dynamics + alpha * h
    
    # Solve QP with CBF constraint
    # min ||u - u_desired||^2
    # s.t. cbf_constraint >= 0
    return solve_qp(control_desired, cbf_constraint)
```

**Safety Filter Architecture:**
```
SAFETY FILTER PIPELINE:
─────────────────────────────────────────────────────────────
Desired Action (from VLA/RL policy)
         │
    ┌────▼────┐
    │ Safety  │
    │ Filter  │
    │  (QP)   │
    └────┬────┘
         │
    ┌────▼────────────┐
    │ Check constraints│
    │ • Joint limits   │
    │ • Collision      │
    │ • Velocity       │
    │ • Force          │
    └────┬────────────┘
         │
    ┌────▼────┐
    │ Safe    │
    │ Action  │
    └─────────┘
─────────────────────────────────────────────────────────────
```

### 7.2 Emergency Stop Systems

**Hardware safety:**
- E-stop buttons (physical, always accessible)
- Joint torque limits (hardware-enforced)
- Speed limiting (software + hardware)
- Collision detection (force sensors)

**Software safety:**
- Watchdog timers (detect frozen controllers)
- Heartbeat monitoring (detect communication loss)
- Graceful degradation (reduced speed, safe stop)
- State estimation monitoring (detect sensor failures)

### 7.3 Failure Modes and Recovery

| Failure Type | Detection | Response |
|-------------|-----------|----------|
| Motor failure | Current spike, position error | Safe stop, reconfigure |
| Sensor failure | Timeout, invalid values | Switch to redundant sensor |
| Communication loss | Heartbeat timeout | Emergency stop |
| Software crash | Watchdog timeout | Restart, relocalize |
| Collision detected | Force threshold | Immediate stop, retract |
| Human too close | Safety scanner | Slow down, stop if closer |

---

## 8. Multi-Modal Perception

### 8.1 Visual-Tactile Fusion

Combining vision and touch for robust manipulation:

```
VISUAL-TACTILE FUSION ARCHITECTURE:
─────────────────────────────────────────────────────────────
Camera Image ──→ Vision Encoder ──→ Visual Features ──┐
                                                        │
Tactile Signal ──→ Tactile Encoder ──→ Tactile Features ─┤
                                                        │
                                        ┌───────────────▼──────────────┐
                                        │    Cross-Modal Attention      │
                                        │    (Fuse visual + tactile)    │
                                        └───────────────┬──────────────┘
                                                        │
                                                Action Prediction
─────────────────────────────────────────────────────────────
```

**Applications:**
- Slip detection: Visual motion + tactile force → slip early warning
- Material recognition: Visual appearance + tactile texture → material ID
- Force estimation: Visual contact + tactile pressure → contact force
- Grasp stability: Visual pose + tactile distribution → grasp quality

### 8.2 Visual-Language Navigation

Navigating complex environments using language instructions:

**Approaches:**
1. **Segmentation + Planning**: Segment scene, plan path to target
2. **End-to-end VLN**: Direct language → action mapping
3. **Hierarchical**: High-level navigation + local obstacle avoidance

**Benchmarks:**
- **R2R (Room-to-Room)**: Indoor navigation from language
- **RxR**: Multilingual navigation
- **REVERIE**: Object-grounded navigation
- **SOON**: Scene-aware object navigation

### 8.3 Scene Understanding for Manipulation

Understanding the full 3D scene before acting:

1. **3D reconstruction**: Build 3D model from camera images
2. **Object segmentation**: Identify and separate objects
3. **Semantic understanding**: Know what each object is
4. **Relationship detection**: Understand object relationships
5. **Affordance detection**: Identify graspable surfaces, functional parts
6. **State estimation**: Track object states over time

---

## Summary

The technical frontier of Physical AI in 2026 is defined by:

1. **VLA models** enabling cross-embodiment transfer with web knowledge
2. **Large-scale datasets** (Open X-Embodiment) enabling generalization
3. **Sim-to-real pipelines** making training feasible and safe
4. **Edge computing** enabling real-time inference on robots
5. **Safety-critical control** providing formal guarantees
6. **Multi-modal perception** combining vision, touch, and language

The next breakthrough will likely come from **foundation models** that can generalize across any robot embodiment, any task, and any environment — the physical equivalent of GPT for language.

---

*This document is part of the AiBaseKnowledge library. See [01-Overview.md](./01-Overview.md) for the full overview.*
