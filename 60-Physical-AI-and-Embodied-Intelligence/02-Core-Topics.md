# Physical AI and Embodied Intelligence: Core Topics

> **Category:** 60 — Physical AI and Embodied Intelligence  
> **Last Updated:** July 2026  
> **Cross-references:** [01-Overview.md](./01-Overview.md), [03-Technical-Deep-Dive.md](./03-Technical-Deep-Dive.md)

---

## Table of Contents

1. [Robot Perception Systems](#1-robot-perception-systems)
2. [Manipulation and Grasping](#2-manipulation-and-grasping)
3. [Locomotion and Mobility](#3-locomotion-and-mobility)
4. [Human-Robot Interaction](#4-human-robot-interaction)
5. [Motion Planning and Control](#5-motion-planning-and-control)
6. [Sim-to-Real Transfer](#6-sim-to-real-transfer)
7. [Multi-Robot Systems](#7-multi-robot-systems)
8. [Robot Learning Paradigms](#8-robot-learning-paradigms)

---

## 1. Robot Perception Systems

### 1.1 Visual Perception

Visual perception is the primary sense for most physical AI systems. Modern robots use multiple camera configurations to understand their environment.

**Camera Configurations:**

| Configuration | Cameras | Coverage | Use Case |
|--------------|---------|----------|----------|
| Monocular | 1 RGB | 60-90° FOV | Simple navigation |
| Stereo | 2 RGB | 60-90° + depth | Manipulation, navigation |
| RGB-D | 1 RGB + depth | 60-90° + depth | Indoor manipulation |
| Omni-directional | 3-4 fisheye | 360° | Mobile robots, AGVs |
| Eye-in-hand | 1+ on gripper | Close-up | Precision manipulation |

**Depth Sensing Technologies:**

```
DEPTH SENSING COMPARISON:
────────────────────────────────────────────────────────────
Technology      │ Range     │ Accuracy  │ Cost    │ Outdoor
────────────────────────────────────────────────────────────
Structured Light│ 0.2-10m   │ ±1mm      │ $100-500│ No
Time-of-Flight  │ 0.1-10m   │ ±5mm      │ $50-300 │ Limited
Stereo Vision   │ 0.5-50m   │ ±5mm      │ $50-200 │ Yes
LiDAR           │ 0.1-300m  │ ±2mm      │ $500-5K │ Yes
FMCW LiDAR      │ 0.1-300m  │ ±1mm      │ $1-5K   │ Yes
────────────────────────────────────────────────────────────
```

**Object Detection and Recognition:**

Modern robot perception uses transformer-based detection models:
- **YOLO v8/v9**: Real-time object detection (30-60 FPS on edge)
- **Grounding DINO**: Open-vocabulary detection from text prompts
- **Segment Anything (SAM 2)**: Universal image segmentation
- **CLIP**: Vision-language alignment for zero-shot recognition
- **DINOv2**: Self-supervised visual features for robotics

**6D Pose Estimation:**

Knowing not just *what* objects are, but their exact 3D position and orientation:
- **FoundationPose** (NVIDIA): Single-image 6D pose estimation
- **DiffPose**: Diffusion-based pose estimation
- **BundleSDF**: Temporal pose tracking
- Required for: grasping, assembly, tool use

### 1.2 Tactile Perception

Tactile sensing enables robots to feel what they touch — critical for manipulation.

**Tactile Sensor Types:**

| Type | Principle | Resolution | Bandwidth | Cost |
|------|-----------|-----------|-----------|------|
| Vision-based (GelSight) | Camera + deformable gel | ~10K points | 100 Hz | $200-500 |
| Capacitive | Capacitance change | ~100 points | 1 kHz | $50-200 |
| Piezoelectric | Charge generation | ~50 points | 10 kHz | $100-300 |
| Resistive | Resistance change | ~100 points | 1 kHz | $20-100 |
| Optical (BioTac) | Impedance + temperature | ~19 electrodes | 1 kHz | $500+ |

**Tactile Learning:**
- **Slip detection**: Preventing objects from falling during grasping
- **Texture recognition**: Identifying surfaces by touch
- **Force estimation**: Inferring contact forces without force sensors
- **In-hand manipulation**: Repositioning objects using tactile feedback

### 1.3 Audio Perception

Robots increasingly use audio for environmental awareness:
- **Sound source localization**: Identifying where sounds come from
- **Event detection**: Glass breaking, alarms, human speech
- **Contact detection**: Identifying when/where the robot touches something
- **Echolocation**: Using sound to map environments (bats/dolphins inspired)

### 1.4 Multi-Modal Sensor Fusion

Combining multiple sensor modalities for robust perception:

```
SENSOR FUSION ARCHITECTURE:
─────────────────────────────────────────────────────────
RGB Camera ──┐
              ├──→ Feature Extraction ──┐
Depth Map ────┘                         │
                                        ├──→ Attention Fusion ──→ World Model
LiDAR ───────┐                         │
              ├──→ Feature Extraction ──┘
IMU ──────────┘
─────────────────────────────────────────────────────────
```

**Fusion Approaches:**
- **Early fusion**: Combine raw sensor data before processing
- **Late fusion**: Process each modality independently, combine decisions
- **Attention-based fusion**: Learn which modalities to attend to for each task

---

## 2. Manipulation and Grasping

### 2.1 Grasp Planning

Determining how to pick up an object is a fundamental robotics problem.

**Grasp Quality Metrics:**
- **Grasp wrench space (GWS)**: Volume of wrenches the grasp can resist
- **Force closure**: Ability to resist arbitrary external forces
- **Form closure**: Geometric constraint preventing object motion
- **Success probability**: Empirical success rate from training

**Grasp Planning Pipelines:**

1. **Analyze** → Segment object, estimate pose
2. **Generate** → Propose candidate grasps
3. **Score** → Evaluate grasp quality
4. **Execute** → Move gripper to grasp pose
5. **Verify** → Confirm successful grasp via tactile/visual feedback

### 2.2 Gripper Taxonomy

| Gripper Type | DOF | Payload | Dexterity | Best For |
|-------------|-----|---------|-----------|----------|
| Parallel jaw | 1 | 0.5-10 kg | Low | Simple pick-place |
| Vacuum/suction | 0 | 0.1-5 kg | Low | Flat objects, boxes |
| 3-finger (Robotiq) | 3-4 | 2-5 kg | Medium | Mixed objects |
| Anthropomorphic | 5+ | 0.5-2 kg | High | Dexterous tasks |
| Soft gripper | Variable | 0.1-2 kg | Medium | Fragile, irregular |
| Geometric | 0 | 5-50 kg | Low | Palletizing |

### 2.3 Deformable Object Manipulation

One of the hardest problems in manipulation — handling cloth, rope, food:

- **Cloth folding**: Hierarchical planning, particle-based models
- **Rope manipulation**: Shape matching, goal-conditioned policies
- **Food handling**: Variable compliance, damage prevention
- **Cable routing**: PCB assembly, wire harness manufacturing

### 2.4 Contact-Rich Manipulation

Tasks requiring sustained contact and precise force control:
- **Insertion**: Peg-in-hole, USB插拔, screw tightening
- **Surface following**: Wiping, polishing, sanding
- **Sliding**: Drawers, switches, doors
- **Tool use**: Hammering, screwdriving, cutting

---

## 3. Locomotion and Mobility

### 3.1 Bipedal Locomotion

Humanoid robots face unique locomotion challenges:

**Control Approaches:**

| Method | Description | Robustness | Computation |
|--------|-------------|-----------|-------------|
| ZMP-based | Zero Moment Point trajectory | Medium | Low |
| MPC | Model Predictive Control | High | Medium |
| RL policy | Learned from simulation | Very High | Low (inference) |
| Hybrid MPC+RL | MPC high-level, RL low-level | Very High | Medium |

**Key Challenges:**
- **Balance recovery**: Push recovery, trip recovery
- **Stair climbing**: Dynamic foot placement, height adaptation
- **Terrain adaptation**: Gravel, slopes, wet surfaces
- **Carrying loads**: Maintaining balance under variable payloads
- **Falling safely**: Minimizing damage when balance is lost

**State of the Art (2026):**
- Unitree H1/G1: Can walk, run, jump, do backflips
- Tesla Optimus: Walking in factory environments
- Figure 02: Walking and manipulating simultaneously
- Boston Dynamics Atlas (electric): Most dynamic humanoid

### 3.2 Quadruped Locomotion

Four-legged robots are more stable and practical for many applications:

```
QUADRUPED ADVANTAGES OVER BIPEDAL:
─────────────────────────────────────────────────────────
✓ Higher stability (4 contact points)
✓ Greater payload capacity
✓ Faster terrain traversal
✓ Simpler control
✗ Cannot use human tools/infrastructure
✗ Cannot climb ladders/stairs designed for humans
✗ Less dexterous manipulation potential
─────────────────────────────────────────────────────────
```

**Leading Platforms:**
- **Boston Dynamics Spot**: Most commercially mature, $74.5K
- **Unitree B2**: High-speed, industrial-grade, lower cost
- **ANYmal** (ANYbotics): Inspection-focused, harsh environments
- **Ghost Robotics Vision 60**: Military/law enforcement focus

### 3.3 Aerial Robotics (Drones)

 drones extend physical AI into the air:

- **Fixed-wing**: Long-range survey, agriculture mapping
- **Multirotor**: Precision hovering, inspection, delivery
- **Hybrid VTOL**: Vertical takeoff + efficient forward flight
- **Swarm coordination**: Multi-drone inspection, light shows

**AI-Enhanced Drone Capabilities:**
- Autonomous obstacle avoidance (sense-and-avoid)
- Visual-inertial SLAM for GPS-denied environments
- AI-powered inspection (detecting defects, anomalies)
- Autonomous docking and recharging

### 3.4 Wheeled and Tracked Mobility

Still dominant for industrial applications:
- **AGVs (Automated Guided Vehicles)**: Follow magnetic tape, wires
- **AMRs (Autonomous Mobile Robots)**: AI-powered navigation, flexible
- **Omnidirectional**: Mecanum wheels for tight spaces
- **Tracked**: Rough terrain, construction, military

---

## 4. Human-Robot Interaction

### 4.1 Physical HRI

Physical collaboration between humans and robots:

**Safety Standards:**
- ISO/TS 15066: Collaborative robot safety
- Force limits: 150N (quasi-static), 280N (transient) for body contact
- Speed limits: 250 mm/s in collaborative workspace
- Power and force limiting: Active monitoring and stop

**Collaboration Modes:**

| Mode | Description | Example |
|------|-------------|---------|
| Co-existence | Shared workspace, no simultaneous contact | AMR + worker |
| Sequential | Take turns interacting with same workpiece | Robot welds, human inspects |
| Cooperative | Simultaneous manipulation | Human + robot carry object |
| Direct collaboration | Direct physical contact | Exoskeleton, cobot assembly |

### 4.2 Communication Modalities

| Modality | Bandwidth | Latency | Use Case |
|----------|----------|---------|----------|
| Voice | Medium | 0.5-2s | Commands, status updates |
| Gesture | High | 0.1-0.5s | Direction, approval, stop |
| Haptic | Very High | <0.01s | Force guidance, teaching |
| Visual (LEDs) | Low | 0.1s | Status indication |
| Screen | High | 0.1s | Detailed information |

### 4.3 Robot Teaching Methods

How humans teach robots new skills:

1. **Lead-through teaching**: Physically moving the robot through motions
2. **Kinesthetic teaching**: Applying forces to guide the robot
3. **Teleoperation**: Remote control with master-slave setup
4. **Learning from demonstration (LfD)**: Recording and generalizing demonstrations
5. **Natural language instructions**: "Pick up the red block and place it on the shelf"
6. **Programming by demonstration**: Visual programming of robot behaviors

---

## 5. Motion Planning and Control

### 5.1 Motion Planning Algorithms

| Algorithm | Optimality | Completeness | Speed | Use Case |
|-----------|-----------|-------------|-------|----------|
| A*/Dijkstra | Optimal | Complete | Fast | Grid navigation |
| RRT | Asymptotically | Probabilistic | Fast | High-DOF planning |
| RRT* | Optimal | Probabilistic | Medium | Optimal path planning |
| PRM | Asymptotically | Probabilistic | Fast | Multi-query planning |
| CHOMP | Local optima | No | Fast | Trajectory optimization |
| TrajOpt | Local optima | No | Fast | Constrained optimization |
| MPPI | Local optima | No | Fast | Model-predictive |

### 5.2 Control Approaches

**Classical Control:**
- PID: Simple, reliable, widely used
- Impedance/Admittance: Force-aware control for contact tasks
- Sliding mode: Robust to model uncertainty

**Learning-Based Control:**
- RL policies: Trained in simulation, deployed on real robots
- MPC + learned models: Combine model-based planning with learned dynamics
- Diffusion policies: Generate action sequences via denoising

**Hybrid Approaches (State of the Art):**
```
ARCHITECTURE: TAMP (Task and Motion Planning)
─────────────────────────────────────────────────────────
High Level: Task Planner (symbolic)
    ↓ "Pick red block, place on shelf"
Motion Planner: RRT*/TrajOpt
    ↓ "Joint trajectory avoiding obstacles"
Low Level: Controller (PID/Impedance)
    ↓ "Motor torques at each timestep"
Actuators → Robot Body → Physical World
─────────────────────────────────────────────────────────
```

### 5.3 Real-Time Constraints

Physical AI systems must meet strict timing requirements:

| Task | Required Frequency | Latency Budget |
|------|-------------------|----------------|
| Joint control | 1 kHz | 1 ms |
| Force control | 1 kHz | 1 ms |
| Obstacle avoidance | 100 Hz | 10 ms |
| Object detection | 30 Hz | 33 ms |
| Path planning | 10-100 Hz | 10-100 ms |
| Task planning | 1-10 Hz | 100ms-1s |

---

## 6. Sim-to-Real Transfer

### 6.1 The Reality Gap

Why simulation-trained policies fail on real robots:

| Gap Type | Description | Impact |
|----------|-------------|--------|
| Visual gap | Rendered images ≠ camera images | Perception failure |
| Physics gap | Sim physics ≠ real physics | Control failure |
| Dynamics gap | Friction, compliance models wrong | Manipulation failure |
| Sensor noise | Sim is noiseless, real is noisy | Robustness failure |
| Latency | Sim is instantaneous, real has delay | Stability failure |

### 6.2 Domain Randomization

The primary technique for bridging the sim-to-real gap:

Randomize simulation parameters during training:
- **Visual**: Textures, lighting, backgrounds, camera noise
- **Physics**: Friction, mass, damping, joint limits
- **Dynamics**: Action delays, sensor noise, observation dropout
- **Scene**: Object poses, positions, counts

**Randomization ranges:**
```
TYPICAL DOMAIN RANDOMIZATION RANGES:
─────────────────────────────────────────────────────────
Parameter          │ Range
─────────────────────────────────────────────────────────
Friction coefficient│ 0.2 - 1.5 (nominal: 0.8)
Object mass        │ 0.5x - 2.0x (nominal: 1.0x)
Light intensity    │ 0.3 - 2.0 (nominal: 1.0)
Camera noise σ     │ 0 - 15 pixels
Action delay       │ 0 - 50 ms
Joint damping      │ 0.5x - 2.0x
─────────────────────────────────────────────────────────
```

### 6.3 Progressive Sim-to-Real

Start easy, increase difficulty:
1. Train in clean simulation
2. Add domain randomization
3. Train with realistic sensor models
4. Fine-tune on small amount of real data
5. Deploy with online adaptation

### 6.4 Real-World Fine-Tuning

When simulation isn't enough, use real robot data:
- **Few-shot fine-tuning**: 10-100 real demonstrations
- **Online fine-tuning**: Continuously update from real interactions
- **Offline RL**: Learn from previously collected real data
- **Sim+real jointly**: Interleave sim and real training

---

## 7. Multi-Robot Systems

### 7.1 Coordination Paradigms

| Paradigm | Description | Scalability | Use Case |
|----------|-------------|------------|----------|
| Centralized | One controller plans for all | Low | Small teams |
| Decentralized | Each robot plans independently | High | Swarms |
| Hybrid | Local + global coordination | Medium | Warehouses |
| Leader-follower | One robot leads, others follow | Medium | Inspection |

### 7.2 Communication Requirements

| Application | Bandwidth | Latency | Reliability |
|------------|-----------|---------|-------------|
| Warehouse fleet | 1-10 Mbps | 10-100 ms | 99.9% |
| Multi-robot assembly | 10-100 Mbps | 1-10 ms | 99.99% |
| Swarm robotics | 100 Kbps | 100 ms | 99% |
| Remote teleoperation | 50-200 Mbps | <5 ms | 99.99% |

### 7.3 Fleet Management

Modern robot fleet management platforms:
- **AWS RoboMaker**: Cloud-based robot management
- **NVIDIA Isaac fleet**: Multi-robot simulation and management
- **ROS 2 Nav2**: Navigation stack with multi-robot support
- **Gazebo + ROS 2**: Open-source multi-robot simulation

---

## 8. Robot Learning Paradigms

### 8.1 Reinforcement Learning for Robotics

RL has become the dominant paradigm for learning robot control:

**Challenges for robotics RL:**
- Sample inefficiency (real robots are slow)
- Safety during exploration (no "reset" in real world)
- Sparse rewards (task completion is rare)
- Non-stationary environments (world changes)

**Solutions:**
- **Sim-first**: Train in simulation, deploy to real
- **Offline RL**: Learn from pre-collected datasets
- **Safe RL**: Constrained optimization with safety guarantees
- **Reward shaping**: Dense rewards to guide exploration

### 8.2 Imitation Learning

Learning from human demonstrations:

| Method | Data Requirement | Generalization | Use Case |
|--------|-----------------|---------------|----------|
| Behavioral Cloning | 100+ demos | Low | Simple tasks |
| DAgger | 50+ demos + queries | Medium | Navigation |
| Imitation + RL | 10-50 demos | High | Manipulation |
| Inverse RL | 20-100 expert demos | Medium | Complex behaviors |

### 8.3 Vision-Language-Action Models

The 2025-2026 breakthrough in robot learning:

**VLA Architecture:**
```
VLA MODEL PIPELINE:
─────────────────────────────────────────────────────────
1. Vision Encoder (ViT/CLIP)
   → Process camera image → Visual features
   
2. Language Encoder (LLaMA/T5)
   → Process instruction → Language features
   
3. Action Decoder (Transformer)
   → Fuse visual + language → Action tokens
   
4. Action Head
   → Convert tokens → Joint velocities / end-effector pose
─────────────────────────────────────────────────────────
```

**Training Data:**
- **Open X-Embodiment**: 1M+ robot episodes across 22 robot types
- **DROID**: Large-scale manipulation dataset
- **Bridge Dataset**: 60K+ robot manipulation episodes
- **RoboSet**: Multi-task, multi-robot dataset

### 8.4 Self-Supervised Robot Learning

Learning without labeled data:
- **Curiosity-driven exploration**: Robots explore and learn from surprise
- **Contrastive learning**: Learn visual representations from robot video
- **Masked prediction**: Predict masked sensor inputs (like BERT for robots)
- **World models**: Learn to predict future observations and rewards

---

## Summary

Physical AI in 2026 is characterized by:
1. **Foundation models** (VLA) enabling cross-embodiment transfer
2. **Sim-to-real** pipelines making training feasible and safe
3. **Humanoid robots** reaching commercial viability
4. **Safety standards** maturing to enable human-robot collaboration
5. **Multi-robot coordination** enabling fleet-level intelligence

The next frontier: **Physical AGI** — robots that can learn any physical skill from a single demonstration, generalize across environments, and safely operate alongside humans.

---

*This document is part of the AiBaseKnowledge library. See [01-Overview.md](./01-Overview.md) for the full overview.*
