# Tools and Frameworks for Physical AI in 2026

> **Description:** A comprehensive inventory of the 2026 Physical AI software and hardware toolchain — simulation platforms, VLA training libraries, robot hardware platforms, sensor stacks, teleoperation interfaces, evaluation benchmarks, and the integration glue that turns research code into a deployed humanoid. Includes tool comparisons, install instructions, license terms, and the canonical 2026 reference stack.

---

## Table of Contents

1. [The 2026 Reference Stack](#1-the-2026-reference-stack)
2. [Simulation Platforms](#2-simulation-platforms)
3. [VLA Training Libraries](#3-vla-training-libraries)
4. [Robot Hardware Platforms](#4-robot-hardware-platforms)
5. [Sensors and Perception](#5-sensors-and-perception)
6. [Teleoperation Interfaces](#6-teleoperation-interfaces)
7. [World Model Platforms](#7-world-model-platforms)
8. [Datasets](#8-datasets)
9. [Evaluation Benchmarks](#9-evaluation-benchmarks)
10. [Safety, Monitoring, and Observability](#10-safety-monitoring-and-observability)
11. [Middleware and Real-Time Control](#11-middleware-and-real-time-control)
12. [Edge Compute and Acceleration](#12-edge-compute-and-acceleration)
13. [DevOps for Physical AI Fleets](#13-devops-for-physical-ai-fleets)
14. [Toolchain Comparison Matrices](#14-toolchain-comparison-matrices)
15. [Getting Started: A 30-Day Plan](#15-getting-started-a-30-day-plan)

---

## 1. The 2026 Reference Stack

A canonical 2026 Physical AI deployment uses the following stack:

```
┌─────────────────────────────────────────────────────────────┐
│  ORCHESTRATION:    Ray, KubeRay, SkyPilot, Determined AI    │
├─────────────────────────────────────────────────────────────┤
│  TRAINING:         PyTorch 2.4+, JAX 0.4+, FSDP, DeepSpeed  │
│  RL LIBRARIES:     RLLib, Stable-Baselines3, Isaac Lab RL  │
│  DIFFUSION:        HuggingFace Diffusers, FlowMatching      │
├─────────────────────────────────────────────────────────────┤
│  VLA MODELS:       OpenVLA, OpenVLA-OFT, π₀, GR00T         │
│  WORLD MODELS:     Cosmos, Genesis, Dreamer V3             │
│  EMBODIMENT:       HPT, Skild Brain, NVIDIA Isaac Sim      │
├─────────────────────────────────────────────────────────────┤
│  SIMULATION:       Isaac Lab 2.0, MuJoCo XLA, Genesis      │
│  RENDERING:        Isaac Sim (RTX), ManiSkill 3, Habitat 3  │
├─────────────────────────────────────────────────────────────┤
│  MIDDLEWARE:       ROS 2 Humble/Iron, CycloneDDS           │
│  REAL-TIME CTRL:   Orocos, RT-PREEMPT, EtherCAT, SOEM      │
├─────────────────────────────────────────────────────────────┤
│  EDGE COMPUTE:     Jetson Thor, RTX 6000 Ada, Hailo-15     │
│  INFERENCE:        TensorRT, Triton, vLLM, LMDeploy        │
├─────────────────────────────────────────────────────────────┤
│  HARDWARE:         Franka, Unitree, Figure, Apptronik      │
│  SENSORS:          RealSense, ZED, Prophesee, ATI, SynTouch│
├─────────────────────────────────────────────────────────────┤
│  TELEOP:           Open-Teleop, ALOHA, GELLO, iPhone-NeRF  │
│  VR/AR:            Quest 3, Vision Pro, Manus Gloves        │
├─────────────────────────────────────────────────────────────┤
│  SAFETY:           ISO 13849, VerifAI, Reachability, ISO/TS 15066 │
│  OBSERVABILITY:    Foxglove, RViz, custom dashboards       │
├─────────────────────────────────────────────────────────────┤
│  FLEET MGMT:       HuggingFace Spaces, Weights & Biases    │
│  OTA UPDATES:      AWS IoT Greengrass, Azure IoT Edge      │
└─────────────────────────────────────────────────────────────┘
```

The library's existing `23-Local-AI-Inference-Self-Hosting/` and `20-Agent-Infrastructure-and-Observability/` cover the inference and observability layers. This category covers the **robotics-specific** parts.

---

## 2. Simulation Platforms

### 2.1 Physics simulators

| Platform | License | Speed | Differentiability | GPU | Best for |
|----------|---------|-------|-------------------|-----|----------|
| **Nvidia Isaac Lab 2.0** | Free (commercial add-on) | 50K× real-time | ✅ (Warp) | RTX | Photorealism + RL |
| **Isaac Sim 4.5** | Free (commercial) | 5K× real-time | ❌ | RTX | Photorealism, scenes |
| **MuJoCo XLA / MJX** | Apache 2.0 | 10K× real-time | ❌ | TPU/GPU | Fast RL, locomotion |
| **Genesis** | Apache 2.0 | 100K× real-time | ✅ Fully | GPU | Differentiable physics |
| **PyBullet** | Zlib | 200× real-time | ❌ | CPU | Legacy, simple |
| **Drake** | BSD | 100× real-time | ✅ (limited) | CPU | Manipulation, contact |
| **Brax** | Apache 2.0 | 50K× real-time | ✅ | TPU/GPU | Large-scale RL |
| **Newton** (Nvidia + Google + Disney) | Apache 2.0 | 30K× real-time | ✅ | GPU | 2026 release, production |
| **Raisim** | Commercial | 1K× real-time | ✅ | CPU | Contact-rich |
| **SAPIEN** | MIT | 500× real-time | ❌ | CPU | Articulated objects |

**Install (Isaac Lab 2.0):**

```bash
# Isaac Lab requires Isaac Sim 4.5+
pip install isaacsim==4.5.0 --extra-index-url https://pypi.nvidia.com
pip install isaaclab==2.0.0

# Verify
python -c "import isaaclab; print(isaaclab.__version__)"
```

**Install (MuJoCo XLA):**

```bash
pip install mujoco-mjx brax
python -c "import mujoco; print(mujoco.__version__)"
```

**Install (Genesis):**

```bash
pip install genesis-world
python -c "import genesis; print(genesis.__version__)"
```

### 2.2 Manipulation benchmarks (built on the above)

| Benchmark | Built on | Tasks | Robots | Released |
|-----------|----------|-------|--------|----------|
| **LIBERO** | MuJoCo | 130 | Franka | CMU, 2023 |
| **LIBERO-Plus** | MuJoCo | 100 | Franka | CMU, 2025 |
| **ManiSkill 3** | Isaac Lab | 1,000+ | Multiple | UCSD, 2025 |
| **RoboCasa** | MuJoCo | 100 | Franka, Tiago | Stanford, 2024 |
| **Behavior-1K** | Isaac Sim | 1,000 | Multiple | Stanford, 2024 |
| **RLBench / RLBench-2** | CoppeliaSim | 100 | Franka, Panda | Imperial, 2024 |
| **Habitat 3.0** | Habitat-Sim | 1,200 | Fetch, Spot | Meta, 2024 |
| **MimicGen** | Isaac Lab | 50+ | Franka | Nvidia, 2024 |
| **SAPIEN-ManiSkill** | SAPIEN | 201 | XArm | USC, 2024 |
| **Calvin** | PyBullet | 34 | Franka | Freiburg, 2022 |

### 2.3 Locomotion benchmarks

| Benchmark | Robot | Tasks |
|-----------|-------|-------|
| **Isaac Gym Humanoid** | Humanoid | Stand, walk, run |
| **MuJoCo Ant / Humanoid** | Ant, Humanoid | Reach, walk |
| **Genesis Humanoid Benchmark** | Unitree, Berkeley Lite | Walk, balance, run |
| **Quadruped Gym** | A1, Go1, ANYmal | Walk, climb, jump |
| **Berkeley Humanoid Lite Bench** | BHL | Walk, kick, get-up |
| **Unitree Sim** | G1, H1, Go2 | Walk, manipulation |
| **Agility Sim** | Digit | Walk, carry |

---

## 3. VLA Training Libraries

### 3.1 OpenVLA ecosystem

| Library | Use | License |
|---------|-----|---------|
| **OpenVLA** | The original VLA | Apache 2.0 |
| **OpenVLA-OFT** | The 2025 optimized version (25× faster) | Apache 2.0 |
| **Octo** | Cross-embodiment VLA (UC Berkeley) | Apache 2.0 |
| **VLA-Adapter** | Efficient fine-tuning | Apache 2.0 |
| **RoboFlamingo** | Few-shot VLA | Apache 2.0 |
| **ManipLLM** | Object-centric VLA | Apache 2.0 |

**Install:**

```bash
pip install openvla-oft
# Or
git clone https://github.com/openvla/openvla-oft
cd openvla-oft && pip install -e .
```

### 3.2 Foundation model training

| Library | Use | License | Notes |
|---------|-----|---------|-------|
| **PyTorch + FSDP** | Distributed training | BSD | Standard for VLAs |
| **DeepSpeed** | ZeRO optimization | Apache 2.0 | Memory-efficient |
| **Megatron-LM** | Tensor parallelism | Open | For 70B+ models |
| **JAX / Pax** | TPU training | Apache 2.0 | Google's preferred |
| **Composer** | MosaicML training | Apache 2.0 | Recipes |
| **torchtitan** | PyTorch native large-scale | BSD | 2025+ |
| **liger-kernel** | Memory-efficient kernels | BSD | 2025+ |
| **unsloth** | Fast LLM fine-tuning | Apache 2.0 | LoRA/QLoRA |

### 3.3 Diffusion and flow matching

| Library | Use | License |
|---------|-----|---------|
| **diffusers** (HuggingFace) | Diffusion models | Apache 2.0 |
| **flow-matching** (Meta) | Flow matching | CC-BY-NC |
| **torchcfm** | Continuous normalizing flows | MIT |
| **rectified-flow** | Rectified flow | MIT |
| **diffusion-policy** (Collab) | Original diffusion policy | MIT |
| **robomimic** | Imitation learning | MIT |

### 3.4 Imitation learning

| Library | Use | License |
|---------|-----|---------|
| **robomimic** | IL benchmarks | MIT |
| **robosuite** | Franka MuJoCo | MIT |
| **imitation** | DAgger, behavior cloning | MIT |
| **serl** | Sample-efficient RL | MIT |
| **rlkit** | RL algorithms | MIT |
| **stable-baselines3** | RL baselines | MIT |
| **rllib** (Ray) | Distributed RL | Apache 2.0 |
| **tianshou** | RL | MIT |
| **acme** | RL | Apache 2.0 |
| **Mava** | Multi-agent RL | Apache 2.0 |

---

## 4. Robot Hardware Platforms

### 4.1 Industrial / research arms

| Robot | DoF | Reach (m) | Payload (kg) | Repeatability (mm) | Price (2026) |
|-------|-----|-----------|---------------|---------------------|--------------|
| **Franka Research 3** | 7 | 0.855 | 3 | ±0.1 | $35,000 |
| **Franka Production 3** | 7 | 0.855 | 3 | ±0.1 | $45,000 |
| **UFACTORY xArm 7** | 7 | 0.7 | 3.5 | ±0.1 | $5,000 |
| **UFACTORY xArm 6** | 6 | 0.7 | 5 | ±0.1 | $5,000 |
| **UFACTORY Lite 6** | 6 | 0.5 | 1 | ±0.5 | $1,200 |
| **Kinova Gen 3** | 7 | 0.9 | 4 | ±0.1 | $30,000 |
| **Techman TM5S** | 6 | 0.9 | 6 | ±0.05 | $25,000 |
| **Aubo i16** | 6 | 0.96 | 16 | ±0.05 | $18,000 |
| **Dobot Nova 2** | 6 | 0.6 | 2.5 | ±0.05 | $3,000 |
| **Yaskawa Motoman GP12** | 6 | 1.4 | 12 | ±0.02 | $50,000 |

### 4.2 Humanoid platforms

| Robot | DoF | Height | Weight (kg) | Battery (h) | Price (2026) |
|-------|-----|--------|--------------|-------------|--------------|
| **Unitree G1** | 43 | 1.32m | 35 | 2-6 | **$16,000** |
| **Unitree H1** | 19 | 1.78m | 47 | 2-4 | $90,000 |
| **Unitree H1+** | 19+12 | 1.78m | 47 | 2-4 | $130,000 |
| **Figure 03** | 41 | 1.68m | 60 | 5 | $30K-50K |
| **1X NEO** | 26 | 1.65m | 30 | 2-4 | $20K-30K |
| **Apptronik Apollo** | 32 | 1.73m | 73 | 4 | $50K+ |
| **Agility Digit (Gen 2)** | 16+2 | 1.75m | 65 | 3 | $250K+ (target $50K) |
| **Tesla Optimus V3** | 28+11 | 1.73m | 57 | 5+ | $20K-30K (target) |
| **Fourier GR-1** | 40 | 1.65m | 55 | 2 | TBD |
| **XPeng IronX** | 60+ | 1.78m | 65 | 2-3 | TBD |
| **Galbot G1** | 40+ | 1.60m | 55 | 2 | TBD |
| **Sanctuary Phoenix** | 20+ | 1.70m | 70 | 2 | TBD |
| **Boston Dynamics Atlas (electric)** | 28 | 1.50m | 80 | 1-2 | TBD |
| **Honda Asimo** | 57 | 1.30m | 48 | 1 | Legacy |
| **PAL Talos** | 32 | 1.75m | 95 | 2 | $100K+ |
| **Toyota HSR** | 8+ | 1.20m | 37 | 2 | $50K |

### 4.3 Quadrupeds and mobile bases

| Robot | DoF | Speed (m/s) | Payload (kg) | Battery (h) | Price |
|-------|-----|-------------|---------------|-------------|-------|
| **Unitree Go2** | 12 | 3.7 | 8 | 2-4 | $1,600 |
| **Unitree B2** | 12 | 6.0 | 40 | 4-6 | $80,000 |
| **Boston Dynamics Spot** | 12 | 1.6 | 14 | 1.5 | $150,000 |
| **Boston Dynamics Spot+ (2026)** | 12 | 2.0 | 20 | 2.5 | $180,000 |
| **ANYmal C** | 12 | 1.0 | 10 | 2 | $200,000 |
| **ANYmal X (explosion-proof)** | 12 | 1.0 | 10 | 2 | $300,000 |
| **DeepRobotics Lite3** | 12 | 4.0 | 5 | 1.5 | $2,000 |
| **XPeng PX2** | 16 | 2.5 | 20 | 2 | $30K |

### 4.4 Dexterous hands

| Hand | Fingers | DoF | Sensors | Price |
|------|---------|-----|---------|-------|
| **Robot Era Star** | 5 | 6 | Tactile | $5,000 |
| **Shadow Robot Dexterous** | 5 | 24 | Tactile | $130K+ |
| **Psyonic Ability** | 5 | 6 | Tactile | $15K |
| **Clone Robotics Hand** | 5 | 27 | Proprio, tactile | $2,000 (DIY kit) |
| **Wonik Robotics Allegro** | 4 | 16 | Position | $15K |
| **LEAP Hand (Stanford)** | 5 | 16 | Optional | $2,500 |
| **DLR-HIT Hand II** | 5 | 15 | Tactile | $40K |
| **Seed Robotics RH8D** | 5 | 8 | Position | $5,000 |

---

## 5. Sensors and Perception

### 5.1 Vision

| Sensor | Resolution | FPS | Depth | Price |
|--------|------------|-----|-------|-------|
| **Intel RealSense D455** | 1280×800 | 90 | ✅ Stereo | $300 |
| **Intel RealSense D555** | 1280×800 | 90 | ✅ Stereo | $400 |
| **Stereolabs ZED 2i** | 2208×1242 | 120 | ✅ Stereo | $450 |
| **Stereolabs ZED X** | 2208×1242 | 120 | ✅ Stereo (GMSL) | $1,000 |
| **OAK-D Pro** | 12MP | 60 | ✅ Stereo | $300 |
| **Luxonis OAK-D 4** | 12MP | 60 | ✅ Stereo | $500 |
| **Prophesee EVK4** | 1280×720 | 1000 | ❌ (event) | $4,500 |
| **FLIR Blackfly S** | 20MP | 75 | ❌ | $1,500 |
| **Basler ace 2** | 24MP | 50 | ❌ | $1,200 |
| **Sony IMX585 module** | 5MP | 90 | ❌ | $50 |
| **Arducam 64MP** | 64MP | 30 | ❌ | $80 |
| **Arducam IMX477** | 12MP | 60 | ❌ | $40 |

### 5.2 LiDAR and depth

| Sensor | Range | Points/s | Price |
|--------|-------|-----------|-------|
| **Hesai XT32** | 120m | 5.6M | $1,200 |
| **Hesai AT128** | 200m | 1.5M | $2,500 |
| **Robosense RS-LiDAR-16** | 150m | 0.6M | $2,000 |
| **Robosense Helios 32** | 200m | 1.4M | $4,000 |
| **Ouster OS0-128** | 100m | 5.2M | $18,000 |
| **Livox Mid-360** | 70m | 0.2M | $600 |
| **Livox Avia** | 450m | 0.24M | $1,500 |
| **Velodyne VLP-32C** | 200m | 0.6M | $8,000 |
| **Intel RealSense L515** | 9m | 23M | $350 |
| **Microsoft Azure Kinect** | 5.5m | 1M | $400 |

### 5.3 Force / tactile

| Sensor | Axes | Range | Price |
|--------|------|-------|-------|
| **ATI Mini45** | 6 | 145 N | $5,000 |
| **ATI Nano17** | 6 | 17 N | $4,500 |
| **OnRobot HEX-E** | 6 | 200 N | $3,000 |
| **Robotiq FT300** | 6 | 300 N | $3,500 |
| **SynTouch BioTac** | 1+ | tactile | $8,000 |
| **Contactile Puck** | 3 | tactile | $2,000 |
| **Xela uSkin** | 16+ | tactile | $1,000 |
| **TakkTile 2** | 1 | tactile | $50 |
| **Pressure Profile Systems** | 16+ | tactile | $1,500 |

### 5.4 Inertial and proprioceptive

| Sensor | Type | Price |
|--------|------|-------|
| **Bosch BMI088** | 6-axis IMU | $5 |
| **InvenSense ICM-42688** | 6-axis IMU | $10 |
| **VectorNav VN-100** | IMU + mag | $400 |
| **Xsens MTi-30** | Industrial IMU | $1,000 |
| **Lord Microstrain 3DM-GX5** | High-grade IMU | $5,000 |
| **Honeywell HGuide i300** | Tactical IMU | $8,000 |

---

## 6. Teleoperation Interfaces

### 6.1 VR/AR-based

| System | Hardware | Latency | Notes |
|--------|----------|---------|-------|
| **Open-Teleop** | Quest 3 + Manus gloves | ~15 ms | Open source |
| **Aloha 2** | Custom dual-arm + 3DConnexion | ~30 ms | Stanford, $20K |
| **GELLO** | Custom 3D-printed arm | ~20 ms | Berkeley, $300 |
| **Mobile ALOHA** | Aloha + mobile base | ~40 ms | Stanford |
| **Tau Robotics** | Vision Pro + custom | ~25 ms | Apple Silicon |
| **iPhone-NeRF** | iPhone Pro + LiDAR | ~50 ms | Apple's, NeRF-based |
| **CyberRunner** | Custom haptic suit | ~10 ms | TUM, haptics |
| **Dactyl glove** | Custom tactile | ~5 ms | OpenAI legacy |

### 6.2 Direct teaching

| Method | Description | Latency |
|--------|-------------|---------|
| **Kinesthetic** | Hand-guide the arm | 0 (real-time) |
| **Lead-through** | Operator pulls the arm | 0 |
| **Pendant** | Teach pendant buttons | Variable |
| **Scripted** | Pre-written trajectory | 0 |

### 6.3 Multi-modal teleop

| System | Modalities | Notes |
|--------|------------|-------|
| **TouchTele** | Haptic + vision | CMU |
| **Holobot** | Whole-body teleop | MIT |
| **DroneNavigator** | Gaze + gesture | Stanford |
| **TactileTeleop** | Tactile feedback | Meta |
| **ForceJoystick** | Force-feedback joystick | Haption |

---

## 7. World Model Platforms

### 7.1 Industry

| Platform | Vendor | License | Specialty |
|----------|--------|---------|-----------|
| **Cosmos** | Nvidia | Open weights | Photorealistic video + robotics |
| **Genie 3** | Google DeepMind | Closed | Interactive environments |
| **Sora 2** | OpenAI | Closed | General video |
| **Veo 3** | Google | Closed | High-fidelity video |
| **GAIA-1 / GAIA-2** | Wayve | Research | Driving-specific |
| **UniSim** | Google | Research | Universal simulator |
| **Simsomnia** | Dreamer team | Research | Multi-domain |
| **Playground v3** | Meta | Research | Game environments |

### 7.2 Open source

| Platform | Lab | Specialty |
|----------|-----|-----------|
| **Dreamer V3** | DeepMind | Pixel-based RL |
| **IRIS** | DIAMOND | Transformer world model |
| **GAIA-2** | Wayve | Driving |
| **TD-MPC2** | Nikita Rudin | Model-based RL |
| **DreamerV3-X** | Various | Embodiment-specific |
| **Llama-Vision-Sim** | Community | Open VLM-sim |

> See `25-World-Models/01-Overview.md` for the full taxonomy.

---

## 8. Datasets

### 8.1 Manipulation datasets (real)

| Dataset | Episodes | Robots | Tasks | License |
|---------|----------|--------|-------|---------|
| **Open X-Embodiment** | 2M+ | 60+ | 1000+ | Various (mostly CC-BY) |
| **DROID** | 76K | Franka, Sawyer | 86 | CC-BY |
| **LIBERO** | 105K | Franka | 130 | MIT |
| **RT-1** | 130K | Everyday Robot | 700+ | Research |
| **BridgeData** | 7.2K | WidowX | 71 | Research |
| **BridgeData V2** | 60K | WidowX | 100 | Research |
| **RoboSet** | 98K | Multiple | 50+ | Research |
| **Epic Kitchens (Ego4D)** | 1.7K hours | Human | 100+ | CC-BY |
| **Something-Something V2** | 220K | Human | 174 | CC-BY |
| **Epic-Kitchens 2024** | 5K hours | Human | 1,000+ | Research |
| **Ego-Exo4D** | 1.4K hours | Human | 800+ | Research |
| **Epic-100** | 100 hours | Human | 100 | CC-BY |
| **HoloAssist** | 600 hours | Human | 220 | Research |
| **EgoSchema** | 5K | Human | 5K | Research |

### 8.2 Locomotion datasets

| Dataset | Type | Notes |
|---------|------|-------|
| **LIP** (Locomotion Intelligence Pool) | Motion capture → humanoid | Berkeley, 2025 |
| **Motion-X** | MoCap corpus | 16K sequences |
| **AMASS** | MoCap | 30K motions |
| **HumanML3D** | Text-to-motion | 45K sequences |
| **KIT** | MoCap | 4.5K motions |
| **PKUMMD** | MoCap | 5K motions |

### 8.3 Synthetic datasets

| Dataset | Method | Notes |
|---------|--------|-------|
| **Nvidia Isaac Lab Replay** | Procedural | 100M+ frames |
| **Cosmos-Synth-1M** | Video world model | 1M episodes |
| **MimicGen-Large** | Real-seed synthesis | 50K episodes |
| **RoboCasa-Synth** | Procedural | 100K |
| **Genesis-Demos** | Differentiable sim | 10K |
| **ManiSkill Synth** | Isaac Lab | 1M+ |

---

## 9. Evaluation Benchmarks

### 9.1 Manipulation

| Benchmark | Tasks | Test |
|-----------|-------|------|
| **LIBERO** | 130 | Spatial, object, goal, long-horizon |
| **RLBench** | 100 | Skill-level |
| **ManiSkill 3** | 1,000+ | SOTA benchmark |
| **Behavior-1K** | 1,000 | Long-horizon household |
| **CALVIN** | 34 | Long-horizon language-conditioned |
| **Meta-World** | 50 | Meta-learning |
| **RoboSuite** | 8 | Standard |
| **FrankaKitchen** | 5 | Kitchen tasks |
| **OpenX-Sim** | 100 | Sim-to-real transfer |
| **NIST-PAB** (draft) | 50 | Standardized 2026 |

### 9.2 Locomotion

| Benchmark | Robots | Tasks |
|-----------|--------|-------|
| **Isaac Gym Humanoid** | Humanoid | Walk, run, kick |
| **MuJoCo Ant** | Ant | Reach, walk |
| **Quadruped Gym** | ANYmal, A1 | Walk, climb |
| **BHL Bench** | Berkeley Lite | Walk, get-up |
| **Unitree Sim** | G1, H1 | Walk, manipulation |
| **Agility Sim** | Digit | Walk, carry |

### 9.3 Mobile manipulation

| Benchmark | Tasks |
|-----------|-------|
| **Habitat 3.0** | 1,200 mobile manipulation |
| **AI2-THOR** | 300 household |
| **SAPIEN** | 201 articulated |
| **ManipulaTHOR** | 200 household |
| **VirtualHome** | 1,000 household |
| **TDW** | 1,500+ household |

### 9.4 Real-world

| Benchmark | Tasks | Notes |
|-----------|-------|-------|
| **Figure BMW Pilot** | 5 (sheet metal) | Industrial first |
| **Amazon GXO** | 5 (picking) | Warehouse first |
| **Apptronik Mercedes** | 3 (assembly) | Auto first |
| **1X Home Trial** | 10 (household) | Home first |
| **Optimus Factory** | 5 (battery sort) | Internal first |
| **Unitree Industrial** | 10 (boxes) | Largest scale |

---

## 10. Safety, Monitoring, and Observability

### 10.1 Hardware safety

| Standard | Coverage | Required for |
|----------|----------|--------------|
| **ISO 13849** | Safety of machinery | All industrial robots |
| **ISO/TS 15066** | Collaborative robots | Cobots |
| **ISO 10218** | Industrial robot safety | Industrial arms |
| **IEC 61508** | Functional safety | All safety-critical |
| **IEC 62061** | Safety of control systems | Industrial |
| **ANSI/RIA R15.06** | US robot safety | US industrial |
| **UL 3300** | US service robot safety | US service |
| **ISO 13482** | Personal care robots | Elder care |
| **ASTM F45** | Mobile robot test | AMRs |
| **ISO 21448 (SOTIF)** | Safety of intended function | Autonomous vehicles |
| **ISO 26262** | Automotive | Self-driving cars |
| **EU AI Act** | High-risk AI | Embodied AI in EU |

### 10.2 Software safety libraries

| Library | Use | License |
|---------|-----|---------|
| **VerifAI** | Verification of AI | BSD |
| **Hamilton-Jacobi reachability** | Safety analysis | Research |
| **ShieldNN** | Neural network shielding | MIT |
| **CBF (Control Barrier Functions)** | Safety filters | MIT |
| **Lyapunov-stable RL** | Stability guarantees | Research |
| **DSS (Dynamic Safety Shields)** | Runtime safety | MIT |
| **AROC** | Reachability | BSD |
| **JuliaReach** | Reachability | BSD |
| **Hylaa** | Linear hybrid automata | BSD |
| **Flow** | Liveness | Apache 2.0 |

### 10.3 Observability tools

| Tool | Use | License |
|------|-----|---------|
| **Foxglove** | Robot visualization | Apache 2.0 |
| **RViz2** | ROS 2 visualization | BSD |
| **PlotJuggler** | Real-time plotting | MIT |
| **Rerun** | Robotics log visualization | MIT |
| **Foxglove Studio** | Web-based dashboard | Apache 2.0 |
| **Datadog IoT** | Fleet monitoring | Commercial |
| **New Relic IoT** | Fleet monitoring | Commercial |
| **Prometheus + Grafana** | Custom dashboards | Apache 2.0 |
| **Weights & Biases** | Experiment tracking | Commercial |

### 10.4 Fleet management

| Platform | Use |
|----------|-----|
| **AWS RoboMaker** | Cloud robotics (deprecated, replaced by IoT Greengrass) |
| **Azure IoT Edge** | Edge fleet |
| **Google Cloud IoT** | Fleet mgmt |
| **Formant** | Robotics observability |
| **Freedom Robotics** | Fleet management |
| **InOrbit** | Multi-vendor fleet |
| **Rocos** | Cloud robotics |
| **Brain Corp** | Commercial robot fleet |
| **Fetch Robotics Cloud** | AMR fleet |

---

## 11. Middleware and Real-Time Control

### 11.1 Robot middleware

| Middleware | License | Notes |
|------------|---------|-------|
| **ROS 2 Humble** | Apache 2.0 | LTS until 2027 |
| **ROS 2 Iron** | Apache 2.0 | LTS |
| **ROS 2 Jazzy** | Apache 2.0 | 2024 release |
| **ROS 2 Kilted** | Apache 2.0 | 2025 release |
| **ROS 2 Lyrical** | Apache 2.0 | 2026 release (LTS) |
| **CycloneDDS** | Apache 2.0 | Default DDS |
| **Zenoh** | Apache 2.0 | Emerging alternative |
| **LCM** | BSD | Lightweight |
| **MOOS-IvP** | GPL | Marine robotics |
| **YARP** | GPL | Italian institute |
| **Orocos** | LGPL | Real-time control |
| **OpenRTM-aist** | LGPL | Japanese standard |
| **ORB-SLAM3** | GPL | Visual SLAM |
| **Cartographer** | Apache 2.0 | LiDAR SLAM |

### 11.2 Real-time control

| Library | Use | Latency |
|---------|-----|---------|
| **Etherlab (SOEM)** | EtherCAT master | 100 µs cycle |
| **PREEMPT-RT** | Linux real-time | 50 µs |
| **Xenomai** | Hard real-time | 10 µs |
| **VxWorks** | Hard RTOS | 10 µs |
| **QNX Neutrino** | Hard RTOS | 10 µs |
| **FreeRTOS** | Embedded | 100 µs |
| **Zephyr** | Embedded | 100 µs |
| **NuttX** | Embedded | 100 µs |
| **AutoSAR** | Automotive | 1 ms |
| **ROS 2 real-time** | ROS 2 + PREEMPT-RT | 1 ms |

### 11.3 Whole-body control

| Library | Robot | License |
|---------|-------|---------|
| **OCS2** (ETH) | Generic | Apache 2.0 |
| **Pinocchio** | Generic | BSD |
| **Croccodyl** | Generic | BSD |
| **Drake** | Generic | BSD |
| **control-toolbox** | Generic | BSD |
| **WBID** | Humanoid | Research |
| **MuJoCo MPC** | Generic | Apache 2.0 |
| **CasADi** | Optimal control | LGPL |
| **ACADO** | Optimal control | LGPL |
| **HPIPM** | Interior-point MPC | BSD |

---

## 12. Edge Compute and Acceleration

### 12.1 Edge AI chips

| Chip | Vendor | FP16 TFLOPS | Power | Notes |
|------|--------|-------------|-------|-------|
| **Jetson Thor** | Nvidia | 2,000 (FP8) | 75W | 2025 release |
| **Jetson Orin Nano** | Nvidia | 40 | 15W | 2024 |
| **Jetson Orin NX** | Nvidia | 100 | 25W | 2024 |
| **Jetson AGX Orin** | Nvidia | 275 | 60W | 2023 |
| **Qualcomm RB5** | Qualcomm | 12 | 15W | Phone-class |
| **Qualcomm QRB5165** | Qualcomm | 30 | 15W | Robotics |
| **Hailo-15** | Hailo | 40 (INT8) | 7W | Efficient |
| **Apple M3 Ultra** | Apple | 2,800 FP16 | 80W | Research only |
| **Apple M4 Pro** | Apple | 700 FP16 | 38W | Research only |
| **Intel Core Ultra 7** | Intel | 25 | 28W | Laptop-class |
| **Google Edge TPU** | Google | 4 (INT8) | 2W | Coral |
| **Tenstorrent Grayskull** | Tenstorrent | 320 (BF16) | 75W | RISC-V |
| **Cerebras WSE-3** | Cerebras | 1,200,000 (sparse) | 15kW | Datacenter |
| **Groq LPU** | Groq | 188,000 (INT8) | 300W | Inference only |
| **SambaNova RDU** | SambaNova | 300,000 (INT8) | 300W | Dataflow |

### 12.2 Inference frameworks

| Framework | Use | License |
|-----------|-----|---------|
| **TensorRT** | Nvidia GPU | Commercial (free) |
| **TensorRT-LLM** | Nvidia LLM | Commercial (free) |
| **Triton Inference Server** | Serving | BSD |
| **vLLM** | LLM serving | Apache 2.0 |
| **LMDeploy** | LLM serving | Apache 2.0 |
| **ONNX Runtime** | Cross-platform | MIT |
| **OpenVINO** | Intel | Apache 2.0 |
| **TFLite** | Google | Apache 2.0 |
| **CoreML** | Apple | Closed |
| **Qualcomm SNPE** | Qualcomm | Commercial |
| **HailoRT** | Hailo | BSD |

### 12.3 Optimization tools

| Tool | Use | License |
|------|-----|---------|
| **TensorRT Model Optimizer** | Quantization, pruning | Commercial |
| **ONNX Quantizer** | INT8/FP8 | MIT |
| **nncf** (Intel) | Compression | Apache 2.0 |
| **AIMET** (Qualcomm) | Compression | BSD |
| **SmoothQuant** | W8A8 | Apache 2.0 |
| **AWQ** | 4-bit LLM | MIT |
| **GPTQ** | 4-bit LLM | Apache 2.0 |
| **bitsandbytes** | 4/8-bit | MIT |
| **torchao** | PyTorch native | BSD |
| **mlc-llm** | Compile LLM | Apache 2.0 |

---

## 13. DevOps for Physical AI Fleets

### 13.1 MLOps / experiment tracking

| Tool | Use | License |
|------|-----|---------|
| **Weights & Biases** | Tracking | Commercial |
| **MLflow** | Tracking | Apache 2.0 |
| **DVC** | Data versioning | Apache 2.0 |
| **Pachyderm** | Pipelines | Apache 2.0 |
| **Metaflow** | Workflow | Apache 2.0 |
| **ClearML** | MLOps | Apache 2.0 |
| **Comet** | Tracking | Commercial |
| **Neptune.ai** | Tracking | Commercial |
| **Hydra** | Config | MIT |
| **Omegaconf** | Config | Apache 2.0 |
| **Ray** | Distributed | Apache 2.0 |
| **SkyPilot** | Cloud orchestration | Apache 2.0 |
| **Determined AI** | Training | Apache 2.0 |
| **Lambda Labs** | GPU cloud | Commercial |

### 13.2 Versioning and data

| Tool | Use | License |
|------|-----|---------|
| **Git LFS** | Model versioning | Git |
| **DVC** | Data versioning | Apache 2.0 |
| **HuggingFace Hub** | Model sharing | Apache 2.0 |
| **W&B Artifacts** | Model + data | Commercial |
| **LakeFS** | Data lake versioning | Apache 2.0 |
| **Pachyderm** | Pipelines | Apache 2.0 |

### 13.3 Deployment

| Tool | Use | License |
|------|-----|---------|
| **Docker** | Containerization | Apache 2.0 |
| **Kubernetes** | Orchestration | Apache 2.0 |
| **K3s** | Lightweight K8s | Apache 2.0 |
| **KubeRay** | Ray on K8s | Apache 2.0 |
| **Triton** | Inference serving | BSD |
| **BentoML** | Model serving | Apache 2.0 |
| **Ray Serve** | Serving | Apache 2.0 |
| **Seldon Core** | Serving | Apache 2.0 |
| **AWS IoT Greengrass** | Edge | Commercial |
| **Azure IoT Edge** | Edge | Commercial |

---

## 14. Toolchain Comparison Matrices

### 14.1 Sim platforms (2026)

| Property | Isaac Lab | MuJoCo XLA | Genesis | Drake | PyBullet | Brax |
|----------|-----------|------------|---------|-------|----------|------|
| Speed | ★★★★★ | ★★★★ | ★★★★★ | ★★★ | ★★ | ★★★★★ |
| Realism | ★★★★★ | ★★★ | ★★★ | ★★★★ | ★★ | ★★ |
| Diff. | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ |
| License | Free | Apache 2.0 | Apache 2.0 | BSD | Zlib | Apache 2.0 |
| Learning curve | Steep | Medium | Medium | Steep | Easy | Medium |
| Production-grade | Yes | Yes | Emerging | Yes | No | Yes |
| Best for | Photorealism | Locomotion | Research | Manipulation | Teaching | RL |

### 14.2 VLA training libraries (2026)

| Library | Ease | Speed | Scale | Best for |
|---------|------|-------|-------|----------|
| **OpenVLA-OFT** | ★★★★ | ★★★★ | ★★★ | Research to production |
| **Octo** | ★★★★★ | ★★★ | ★★★★ | Cross-embodiment |
| **π₀** | ★★★ | ★★★★ | ★★★★ | Production |
| **GR00T** | ★★★ | ★★★★ | ★★★★★ | Cross-embodiment, large |
| **RoboFlamingo** | ★★★★ | ★★★ | ★★★ | Few-shot |
| **ManipLLM** | ★★★★ | ★★★ | ★★★ | Object-centric |

### 14.3 Edge compute (2026)

| Chip | AI performance | Power | Cost | Best for |
|------|----------------|-------|------|----------|
| **Jetson Thor** | 2,000 FP8 | 75W | $3K | 7B VLAs |
| **Jetson Orin NX** | 100 FP16 | 25W | $700 | 1B-3B models |
| **Apple M3 Ultra** | 2,800 FP16 | 80W | $4K | Research only |
| **Hailo-15** | 40 INT8 | 7W | $200 | Small VLAs, low power |
| **Qualcomm RB5** | 12 FP16 | 15W | $400 | Low-end |

---

## 15. Getting Started: A 30-Day Plan

If you are new to Physical AI in 2026, here is a practical 30-day plan:

### Week 1: Foundations
- **Day 1–2:** Install ROS 2 Humble, complete the official tutorials. Set up Ubuntu 22.04 in a VM.
- **Day 3–4:** Install MuJoCo XLA, run a tutorial. Simulate a pendulum, then a simple arm.
- **Day 5–7:** Read the LIBERO paper, install the LIBERO benchmark. Run the baseline.

### Week 2: Imitation learning
- **Day 8–10:** Collect 100 demos in LIBERO using the keyboard.
- **Day 11–14:** Train a behavior-cloning policy in Isaac Lab. Achieve >50% success on a simple task.

### Week 3: VLA training
- **Day 15–17:** Install OpenVLA. Run the inference demo on a LIBERO task.
- **Day 18–21:** Fine-tune OpenVLA on your 100 demos. Run on the LIBERO eval. Target: >60% success.

### Week 4: Real hardware
- **Day 22–24:** Buy or borrow a UFACTORY xArm 6 ($5K) or use a Franka simulator. Connect to ROS 2.
- **Day 25–28:** Deploy the VLA on the real arm with a safety shield. Add a RealSense camera.
- **Day 29–30:** Document, write a blog post, share on LinkedIn/HN. Join the Physical AI community.

### Reading list (in order)
1. Ha & Schmidhuber, "World Models" (2018)
2. RT-1 paper (Google, 2022)
3. RT-2 paper (Google, 2023)
4. OpenVLA paper (Stanford, 2024)
5. π₀ paper (Physical Intelligence, 2024)
6. OpenVLA-OFT paper (Stanford, 2025)
7. Helix paper (Figure AI, 2025)
8. Cosmos paper (Nvidia, 2025)
9. GR00T N1 paper (Nvidia, 2025)
10. EU AI Act (full text, 2024)

### Communities to join
- **r/MachineLearning** (Reddit)
- **r/robotics** (Reddit)
- **Robotics Stack Exchange**
- **ROS Discourse**
- **HuggingFace Discord** (Physical AI channel)
- **Twitter/X:** @chris_paxton, @svlevine, @karpathy, @hausman_k, @sirbayes, @SkildAI_, @physical_int
- **Conferences:** CoRL, ICRA, IROS, RSS, NeurIPS Robotics Workshop, ICML Workshop on Embodied AI
- **Newsletters:** The Robot Brains (Rodney Brooks / Peter Chen), The Batch (Andrew Ng)

---

*Cross-references: `25-World-Models/04-Tools-and-Frameworks.md` for world-model tools; `23-Local-AI-Inference-Self-Hosting/06-Hardware-for-Local-Inference.md` for edge compute; `20-Agent-Infrastructure-and-Observability/04-...` for fleet observability; `24-AI-Agent-Autonomy-Accountability/05-Governance-Auditing-and-Regulatory-Frameworks.md` for safety standards.*

*Next: `05-Future-Outlook.md` — the 2027–2030 roadmap for Physical AI.*
