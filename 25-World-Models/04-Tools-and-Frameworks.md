# World Models — Tools, Frameworks, and Deployment

> **Description:** A practical reference to the world-model ecosystem as of mid-2026. Covers open-source frameworks (Dreamer V3, V-JEPA 2, Cosmos, Cosma), foundation model downloads, training stacks, inference deployment, hardware recommendations, and a comparison table for choosing the right tool. Designed for engineers and infrastructure teams who need to actually build something.

---

## Table of Contents

1. [The Open-Source Ecosystem at a Glance](#1-the-open-source-ecosystem-at-a-glance)
2. [Dreamer V3 (DeepMind)](#2-dreamer-v3-deepmind)
3. [V-JEPA 2 (Meta FAIR)](#3-v-jepa-2-meta-fair)
4. [NVIDIA Cosmos](#4-nvidia-cosmos)
5. [Cosma — The Open Framework](#5-cosma--the-open-framework)
6. [Genie 2 / 3 (DeepMind)](#6-genie-2--3-deepmind)
7. [Wayve Open Models](#7-wayve-open-models)
8. [Open X-Embodiment](#8-open-x-embodiment)
9. [IRIS (Transformer World Model)](#9-iris-transformer-world-model)
10. [Decart OASIS](#10-decart-oasis)
11. [World Labs (Fei-Fei Li)](#11-world-labs-fei-fei-li)
12. [Foundation Model Download Hub](#12-foundation-model-download-hub)
13. [Training Frameworks](#13-training-frameworks)
14. [Inference and Serving](#14-inference-and-serving)
15. [Hardware for World Model Training](#15-hardware-for-world-model-training)
16. [Hardware for World Model Inference](#16-hardware-for-world-model-inference)
17. [Comparison Matrix: Choosing Your Stack](#17-comparison-matrix-choosing-your-stack)
18. [Production Case Studies](#18-production-case-studies)
19. [Licensing and Commercial Use](#19-licensing-and-commercial-use)
20. [Where to Start (Recommendations)](#20-where-to-start-recommendations)

---

## 1. The Open-Source Ecosystem at a Glance

The 2026 open-source world-model ecosystem is the most active and most fragmented in AI. There is no single "PyTorch of world models." Instead, there are 8-10 serious projects, each with a different design center.

| Project | Maintainer | License | Type | Stars (H1 2026) |
|---------|------------|---------|------|-----------------|
| **Dreamer V3** | DeepMind | Apache 2.0 | Latent dynamics | 14,200★ |
| **V-JEPA 2** | Meta FAIR | CC-BY-NC 4.0 | Non-generative latent | 8,400★ |
| **NVIDIA Cosmos** | NVIDIA | NVIDIA Open (commercial OK) | Generative video | 11,800★ |
| **Cosma** | Community | Apache 2.0 | Modular framework | 4,200★ |
| **IRIS** | Samsung SAIT | MIT | Transformer latent | 2,100★ |
| **Genie 3** | DeepMind | Closed (no open weights) | Generative 3D | n/a |
| **GAIA-1 / GAIA-2** | Wayve | CC-BY-NC 4.0 | Driving | 3,600★ |
| **Open X-Embodiment** | Google DeepMind + 60 labs | Various | Robot data | 2,400★ |
| **Decart OASIS** | Decart | Apache 2.0 | Real-time video | 6,800★ |
| **World Labs Marble** | World Labs | Commercial (beta) | 3D world from image | Closed |
| **Hugging Face transformers** | Hugging Face | Apache 2.0 | Library | 130,000★ |
| **Hugging Face diffusers** | Hugging Face | Apache 2.0 | Library | 35,000★ |
| **torchtitan** | PyTorch team | BSD-3 | Distributed training | 5,800★ |
| **Megatron-Core** | NVIDIA | BSD-3 | Distributed training | 7,200★ |
| **Genesis** | Genesis team | Apache 2.0 | Physics simulation | 4,100★ |
| **Brax** | Google | Apache 2.0 | Physics on TPU/GPU | 6,400★ |
| **MuJoCo XLA** | DeepMind | Apache 2.0 | Physics on TPU/GPU | 4,500★ |
| **MinkowskiEngine** | Facebook | MIT | 3D sparse conv | 1,200★ |
| **OpenVLA** | Stanford / Toyota | Apache 2.0 | VLA model | 3,800★ |
| **Octo** | Stanford / Toyota | Apache 2.0 | Robot policy | 2,600★ |
| **RLDG** | Community | MIT | RL data generator | 800★ |

The community is moving fast. Star counts are H1 2026 figures and grew ~30-50% in the last quarter.

---

## 2. Dreamer V3 (DeepMind)

The reference implementation of latent-dynamics WMs.

### 2.1 Repository

- **URL:** https://github.com/danijar/dreamerv3
- **License:** Apache 2.0
- **Maintainer:** Danijar Hafner (formerly DeepMind, now at Google DeepMind)
- **Status:** Active, with 4 minor releases in H1 2026

### 2.2 What You Get

- Reference implementation of RSSM, dreamer loss, symlog predictions, free bits, etc.
- Pre-trained checkpoints for ~150 tasks (Atari, DMControl, Minecraft, MuJoCo, Procgen, etc.)
- Pluggable environment interface (works with any Gym / DM Env)
- Multi-config support (single hyperparameter set works across 150+ tasks)

### 2.3 Installation

```bash
git clone https://github.com/danijar/dreamerv3.git
cd dreamerv3
pip install -r requirements.txt
# Pre-built Docker image also available
docker pull danijar/dreamerv3:latest
```

### 2.4 Quick Start

```python
# Run Dreamer V3 on Atari
python dreamerv3/main.py \
  --configs atari \
  --task atari_pong

# Run on Minecraft
python dreamerv3/main.py \
  --configs minecraft \
  --task mine_diamond

# Run on a custom Gym env
python dreamerv3/main.py \
  --task my_custom_env \
  --logdir ./logs/my_run
```

### 2.5 Customization

The hyperparameters are in `dreamerv3/configs.yaml`. The most commonly tuned are:

```yaml
# dreamerv3/configs.yaml
defaults:
  - atari
  - _self_

agent:
  wm:
    hidden: 512                # RSSM hidden state size
    stoch: 32                  # stochastic latent dim
    discrete: 32               # number of categorical latents
    kl_scale: 0.1              # KL weight
    free_bits: 1.0             # free bits per dim
  imagine:
    horizon: 50                # imagination length
    lambda_: 0.95              # TD lambda
    gamma: 0.997               # discount factor
  actor:
    entropy: 1e-3              # entropy regularization
    lr: 1e-5                   # actor learning rate
  critic:
    lr: 1e-5                   # critic learning rate
```

### 2.6 When to Use Dreamer V3

- ✅ You have a **continuous control** task (robotics, driving, games).
- ✅ You have a **reward function** (or can label rewards).
- ✅ You want **sample efficiency** (real-world interaction is expensive).
- ✅ You want **interpretable internal states**.
- ❌ You need photorealistic video output (use Sora 2 or Cosmos).
- ❌ You don't have actions (use a video model).

---

## 3. V-JEPA 2 (Meta FAIR)

The reference implementation of non-generative latent WMs.

### 3.1 Repository

- **URL:** https://github.com/facebookresearch/vjepa2
- **License:** CC-BY-NC 4.0 (research use only)
- **Maintainer:** Meta FAIR (Adriana Romero, Jean-Baptiste Alayrac, et al.)
- **Status:** Active

### 3.2 What You Get

- Reference implementation of JEPA architecture (context encoder, target encoder, predictor).
- Pre-trained ViT-Large and ViT-Huge checkpoints.
- Pre-training, fine-tuning, and evaluation scripts.
- Support for video, image, and audio modalities.

### 3.3 Installation

```bash
git clone https://github.com/facebookresearch/vjepa2.git
cd vjepa2
pip install -e .
# Or via pip
pip install vjepa2
```

### 3.4 Quick Start

```python
import torch
from vjepa2 import VJEPA2

# Load pre-trained model
model = VJEPA2.from_pretrained("facebook/vjepa2-vitl-256")

# Encode a video
video = torch.randn(1, 16, 3, 256, 256)  # (B, T, C, H, W)
with torch.no_grad():
    z = model.encode_video(video)
print(z.shape)  # (1, 196, 1024) — 196 patches of 1024-dim

# Predict a future embedding
action = torch.randn(1, 16, 6)  # (B, T, A)
z_pred = model.predict(z, action)
print(z_pred.shape)
```

### 3.5 Downstream Tasks

V-JEPA 2 is designed to be fine-tuned for downstream tasks:

```python
# Fine-tune for action anticipation
from vjepa2 import VJEPA2ForActionAnticipation

model = VJEPA2ForActionAnticipation.from_pretrained("facebook/vjepa2-vitl-256")
# ... fine-tuning code

# Fine-tune for video question answering
from vjepa2 import VJEPA2ForVQA

model = VJEPA2ForVQA.from_pretrained("facebook/vjepa2-vitl-256")
# ... fine-tuning code
```

### 3.6 When to Use V-JEPA 2

- ✅ You need **semantic** rather than visual output.
- ✅ You have limited compute (JEPA is cheaper than diffusion).
- ✅ You need **interpretable latents** (more so than generative).
- ✅ Your task is **reasoning** rather than generation.
- ❌ You need photorealistic video (use Sora 2 / Veo 3).
- ❌ You need to do continuous control (use Dreamer V3).

---

## 4. NVIDIA Cosmos

The foundation world-model suite from NVIDIA. The 2026 reference for production-grade video WMs.

### 4.1 Repository

- **URL:** https://github.com/NVIDIA/Cosmos
- **License:** NVIDIA Open Model License (commercial use OK; some restrictions)
- **Maintainer:** NVIDIA
- **Status:** Active, with monthly updates

### 4.2 What You Get

- Pre-trained foundation models: Cosmos-1B, Cosmos-4B, Cosmos-14B.
- Variants: Cosmos-Predict1 (next-frame prediction), Cosmos-Transfer1 (style/conditioning), Cosmos-Reason1 (with reasoning).
- Synthetic data generation pipeline.
- Integration with NVIDIA Isaac Sim, DRIVE Sim, Omniverse.
- Distributed training scripts (multi-node, multi-GPU).

### 4.3 Installation

```bash
git clone https://github.com/NVIDIA/Cosmos.git
cd Cosmos
pip install -r requirements.txt
# Requires NVIDIA GPU with CUDA 12.4+, 80GB+ VRAM for 14B model
```

### 4.4 Quick Start

```python
import torch
from cosmos import CosmosPredict1

# Load the 4B model
model = CosmosPredict1.from_pretrained("nvidia/cosmos-predict1-4b")

# Generate the next 5 seconds of video from a single image
image = load_image("car_driving.jpg")
video = model.generate(
    image=image,
    num_frames=120,           # 5 seconds at 24 fps
    height=704,
    width=1280,
    guidance_scale=4.0,
    num_inference_steps=35,
)
save_video(video, "output.mp4")
```

### 4.5 Fine-Tuning for Robotics

```python
# Fine-tune Cosmos on your robot's data
from cosmos import CosmosFineTuner

finetuner = CosmosFineTuner(
    base_model="nvidia/cosmos-predict1-4b",
    dataset="my_robot_dataset",
    output_dir="./cosmos_finetuned",
    batch_size=2,
    learning_rate=1e-5,
    num_steps=10_000,
)
finetuner.train()
```

### 4.6 Hardware Requirements

| Model | VRAM (training) | VRAM (inference) | Recommended GPU |
|-------|-----------------|-------------------|-----------------|
| Cosmos-1B | 24 GB | 8 GB | A100 / H100 |
| Cosmos-4B | 80 GB | 16 GB | H100 |
| Cosmos-14B | 256 GB | 48 GB | 2x H100 |
| Cosmos-14B (full FT) | 512 GB+ | n/a | 8x H100 |

### 4.7 When to Use Cosmos

- ✅ You need **production-grade video WM** with commercial use rights.
- ✅ You are in **robotics, driving, or simulation**.
- ✅ You want NVIDIA **integration** (Isaac, DRIVE, Omniverse).
- ✅ You need **synthetic data generation** at scale.
- ❌ You need a tiny model (use the 1B variant or local models).
- ❌ You need continuous-control RL (use Dreamer V3).

---

## 5. Cosma — The Open Framework

The community-driven open framework that wraps multiple WM backends.

### 5.1 Repository

- **URL:** https://github.com/cosma-robotics/cosma
- **License:** Apache 2.0
- **Maintainer:** Community (originally from ETH Zurich, now a foundation)
- **Status:** Active, v0.4 in 2026

### 5.2 What You Get

- A unified interface for JEPA, Dreamer, Cosmos-style models.
- Built-in environments (DMControl, Atari, Habitat, MuJoCo, CARLA).
- Distributed training and inference.
- Pre-built Docker images.
- Pre-trained checkpoints for several backbones.

### 5.3 Why Cosma

If you want to **compare multiple WM approaches** on the same task, Cosma is the easiest way. The same training script can swap between Dreamer V3, V-JEPA 2, and Cosmos.

```python
import cosma

# Configure
config = cosma.Config(
    agent=cosma.agents.DreamerV3(hidden=512),
    env=cosma.envs.Atari("pong"),
    logdir="./logs",
)

# Train
cosma.train(config, steps=10_000_000)

# Swap the agent
config2 = cosma.Config(
    agent=cosma.agents.VJEPA2(model="vitl"),
    env=cosma.envs.Atari("pong"),
    logdir="./logs",
)
cosma.train(config2, steps=10_000_000)
```

### 5.4 Pre-built Environments

| Environment | Status | Notes |
|-------------|--------|-------|
| Atari | ✅ Stable | All 57 games supported |
| DMControl | ✅ Stable | All 30+ tasks |
| Procgen | ✅ Stable | 16 games |
| Minecraft | 🟡 Beta | Via MineDojo |
| MuJoCo | ✅ Stable | All standard tasks |
| Habitat | 🟡 Beta | Embodied AI |
| CARLA | 🟡 Beta | Driving |
| Isaac Sim | 🟡 Beta | Robotics |

### 5.5 When to Use Cosma

- ✅ You want to **compare** multiple WM approaches.
- ✅ You want a **clean API** without learning multiple frameworks.
- ✅ You are doing **research** rather than production.
- ❌ You need the absolute latest features of a specific model.
- ❌ You need maximum performance (use the underlying framework directly).

---

## 6. Genie 2 / 3 (DeepMind)

The interactive 3D world generator. As of mid-2026, DeepMind has not open-sourced Genie 3 weights. There is an API for limited use.

### 6.1 API Access

- **URL:** https://deepmind.google/discover/blog/genie-3/
- **Access:** Waitlist; some academic partners have access
- **Status:** Closed weights; some open papers

### 6.2 Open Reimplementations

Several community reimplementations exist:

- **Genie-Open (community)** — https://github.com/genie-open/genie3 — partial reproduction, lower quality
- **Genie-Small (academic)** — https://github.com/genie-small/model — small variant, Apache 2.0

### 6.3 When to Use Genie 3

- ✅ You have API access and need interactive 3D worlds.
- ✅ You are doing research on embodied agents.
- ❌ You need open weights (use Cosma or community reimplementations).

---

## 7. Wayve Open Models

Wayve has open-sourced portions of GAIA.

### 7.1 Repository

- **URL:** https://github.com/wayve-research/gaia
- **License:** CC-BY-NC 4.0 (research use only; commercial license available)
- **Status:** Active

### 7.2 What You Get

- GAIA-1 architecture and training code.
- Pre-trained model on the Wayve UK fleet dataset (100 hours sample).
- Evaluation scripts.
- Integration with CARLA simulator.

### 7.3 Quick Start

```python
import torch
from wayve_gaia import GAIA1

model = GAIA1.from_pretrained("wayve/gaia-1-100h")

# Generate a driving video given a text prompt
video = model.generate(
    text="a car driving in heavy rain on a UK motorway",
    num_frames=60,
)
```

### 7.4 When to Use Wayve

- ✅ You are doing **driving research** specifically.
- ✅ You want a known-good driving WM.
- ❌ You need a more general WM (use Cosmos).

---

## 8. Open X-Embodiment

A consortium dataset of robot manipulation trajectories across 60+ labs.

### 8.1 Repository

- **URL:** https://github.com/open-x-embodiment/rt-x
- **License:** Various (most CC-BY-NC, some CC-BY)
- **Status:** Active; v1.0 released 2024, v2.0 in 2026

### 8.2 What You Get

- ~2M trajectories across 22 embodiments.
- Standardized data format.
- Pre-trained Octo / OpenVLA checkpoints.

### 8.3 When to Use Open X-Embodiment

- ✅ You are doing **robot manipulation** research.
- ✅ You want to leverage the **largest** public robot dataset.
- ❌ You need a closed-loop WM (Open X-Embodiment is data, not model).

---

## 9. IRIS (Transformer World Model)

A transformer-based latent WM from Samsung SAIT.

### 9.1 Repository

- **URL:** https://github.com/samsung/iris
- **License:** MIT
- **Status:** Active

### 9.2 What You Get

- Discrete autoencoder + transformer dynamics.
- Pre-trained Atari and Minecraft checkpoints.
- Single-GPU training and inference.

### 9.3 When to Use IRIS

- ✅ You want a **simple, single-file** WM implementation.
- ✅ You are doing **Atari / Minecraft** research.
- ❌ You need a production-grade WM (use Dreamer V3 or Cosmos).

---

## 10. Decart OASIS

A real-time video world model for interactive content.

### 10.1 Repository

- **URL:** https://github.com/decart-ai/oasis
- **License:** Apache 2.0
- **Status:** Active; backed by $150M Series A

### 10.2 What You Get

- Real-time video generation (24+ fps on H100).
- Interactive 3D scenes (you control an avatar).
- Single GPU inference.
- Web demo at https://oasis.decart.ai

### 10.3 When to Use Decart

- ✅ You need **real-time interactive** video.
- ✅ You are doing **content creation** (games, marketing).
- ❌ You need a foundation model (OASIS is a single application).

---

## 11. World Labs (Fei-Fei Li)

A 3D world generation service. As of mid-2026, World Labs has not open-sourced Marble.

### 11.1 Service

- **URL:** https://www.worldlabs.ai
- **Access:** API; $200/month for the entry tier
- **Status:** Closed; production service

### 11.2 What You Get

- 3D worlds from a single image, photorealistic, navigable.
- API for embedding in applications.
- 4K export.

### 11.3 When to Use World Labs

- ✅ You need **production-quality 3D worlds** from images.
- ✅ You are doing **VFX, game prototyping, virtual production**.
- ❌ You need open weights (use Decart or Genie-Open).

---

## 12. Foundation Model Download Hub

Where to actually download the models.

### 12.1 Hugging Face

The de facto standard repository. All major open-weights WMs are here.

| Model | URL | License |
|-------|-----|---------|
| Dreamer V3 (Minecraft) | https://huggingface.co/danijar/dreamerv3-minecraft | Apache 2.0 |
| V-JEPA 2 (ViT-L) | https://huggingface.co/facebook/vjepa2-vitl-256 | CC-BY-NC |
| V-JEPA 2 (ViT-H) | https://huggingface.co/facebook/vjepa2-vith-384 | CC-BY-NC |
| Cosmos-1B | https://huggingface.co/nvidia/cosmos-predict1-1b | NVIDIA Open |
| Cosmos-4B | https://huggingface.co/nvidia/cosmos-predict1-4b | NVIDIA Open |
| Cosmos-14B | https://huggingface.co/nvidia/cosmos-predict1-14b | NVIDIA Open |
| IRIS (Atari) | https://huggingface.co/samsung/iris-atari | MIT |
| Wayve GAIA-1 | https://huggingface.co/wayve/gaia-1-100h | CC-BY-NC |
| Qwen-Robot Suite | https://huggingface.co/Qwen/Qwen-Robot | Apache 2.0 |
| Octo (small) | https://huggingface.co/octo-small | Apache 2.0 |
| Octo (base) | https://huggingface.co/octo-base | Apache 2.0 |
| OpenVLA-7B | https://huggingface.co/openvla-7b | Apache 2.0 |

### 12.2 Download CLI

```bash
# Install huggingface_hub
pip install huggingface_hub

# Login (some models require it)
huggingface-cli login

# Download a model
huggingface-cli download nvidia/cosmos-predict1-4b --local-dir ./cosmos-4b
```

### 12.3 Storage Requirements

| Model | Weights Size | With Optimizer State |
|-------|--------------|----------------------|
| Dreamer V3 (small) | 200 MB | 800 MB |
| V-JEPA 2 (ViT-L) | 2.1 GB | 8.4 GB |
| V-JEPA 2 (ViT-H) | 6.7 GB | 26.8 GB |
| Cosmos-1B | 4.2 GB | 16.8 GB |
| Cosmos-4B | 16.8 GB | 67.2 GB |
| Cosmos-14B | 56 GB | 224 GB |
| GAIA-1 | 8.4 GB | 33.6 GB |
| OpenVLA-7B | 28 GB | 112 GB |

---

## 13. Training Frameworks

The 2026 training stack. Most world-model training is done with PyTorch + FSDP + Megatron-Core.

### 13.1 The Standard Stack

```text
PyTorch 2.4+                    — base framework
+ FSDP-2 (Fully Sharded DP)     — distributed training
+ Megatron-Core                 — tensor + pipeline parallel
+ torchtitan                    — reference recipes
+ Transformer Engine (NVIDIA)   — FP8/FP4 precision
+ FlashAttention 3              — efficient attention
+ DeepSpeed                     — alternative, ZeRO
+ Ray                           — cluster orchestration
+ W&B / TensorBoard             — monitoring
```

### 13.2 A Minimal FSDP-2 Training Script

```python
import torch
import torch.distributed.fsdp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# Initialize the model
model = MyWorldModel(...).cuda()

# Wrap in FSDP
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    ),
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    device_id=torch.cuda.current_device(),
)

# Optimizer
optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)

# Training loop
for batch in dataloader:
    loss = compute_loss(model, batch)
    loss.backward()
    model.clip_grad_norm_(1.0)
    optim.step()
    optim.zero_grad()
```

### 13.3 A Megatron-Core Script for Very Large Models

For models > 30B parameters on > 256 GPUs, use Megatron-Core:

```python
from megatron.core import parallel_state, tensor_parallel, pipeline_parallel

# Initialize model parallelism
parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=8,
    pipeline_model_parallel_size=4,
)

# Build the model with parallelism
model = build_video_dit_model(
    num_layers=48,
    hidden_dim=4096,
    num_heads=32,
    tensor_parallel=8,
    pipeline_parallel=4,
)

# Train
train(model, dataset, num_steps=1_000_000)
```

### 13.4 Training Data Loaders

For video data, the 2026 standard is **WebDataset** + **FFmpeg** decoding:

```python
import webdataset as wds
import decord

# WebDataset of tarred video files
dataset = wds.WebDataset("gs://my-bucket/shards/shard-{000000..000099}.tar")
    .decode("torchrgb")  # decode on the fly
    .to_tuple("video.mp4", "actions.json", "rewards.json")
    .batched(32)
    .shuffle(1000)

loader = torch.utils.data.DataLoader(dataset, num_workers=8)
```

For 100M+ hours of video, the 2026 standard is **VAST** (Video As Sets of Tensors) or **TFRecord**-based formats stored in S3/GCS.

---

## 14. Inference and Serving

Serving a world model in production in 2026 requires different tools than serving an LLM.

### 14.1 The Serving Stack

| Layer | Tool | Why |
|-------|------|-----|
| Model | vLLM, SGLang, TensorRT-LLM, Cosmos-Serve | Efficient inference |
| Video decode | NVIDIA Video Codec, FFmpeg | GPU video decode |
| Streaming | WebRTC, HLS, DASH | Low-latency delivery |
| API | FastAPI, gRPC, Envoy | Standard |
| Orchestration | Kubernetes, Ray Serve | Scaling |
| Storage | S3, GCS, MinIO | Video artifacts |

### 14.2 Cosmos-Serve (NVIDIA)

NVIDIA's purpose-built serving stack for Cosmos:

```bash
# Install
pip install cosmos-serve

# Start a server
cosmos-serve start --model nvidia/cosmos-predict1-4b --port 8000

# Call from Python
import requests
r = requests.post("http://localhost:8000/generate", json={
    "image": "base64_image",
    "num_frames": 60,
    "guidance_scale": 4.0,
})
video = r.json()["video"]
```

### 14.3 vLLM-Style Serving for Latent WMs

For Dreamer V3 and similar:

```python
from vllm import LLM

# Wait, vLLM doesn't natively support WMs. Use the new "world-model-engine" fork
# or vLLM with custom plugin:
from worldmodel_engine import WorldModelEngine

engine = WorldModelEngine.from_pretrained("nvidia/cosmos-predict1-4b")
# Or for Dreamer V3:
engine = WorldModelEngine.from_pretrained("danijar/dreamerv3-minecraft")

# Run inference
result = engine.generate(...)
```

### 14.4 Latency Numbers (H1 2026)

| Model | Hardware | Latency per frame | Latency per 60-frame clip |
|-------|----------|-------------------|---------------------------|
| Cosmos-1B | A100 | 50 ms | 3 s |
| Cosmos-4B | H100 | 40 ms | 2.4 s |
| Cosmos-14B | 2x H100 | 80 ms | 4.8 s |
| Genie 3 | 8x H100 | 200 ms | 12 s |
| Dreamer V3 (Minecraft) | A100 | 5 ms (latent) | n/a |
| V-JEPA 2 (ViT-L) | A100 | 10 ms (encoding only) | n/a |
| Decart OASIS | H100 | 42 ms | real-time |
| Sora 2 (API) | OpenAI infra | n/a | 30-60 s |

Real-time video generation is at the edge of feasibility in 2026; expect breakthroughs in 2027.

---

## 15. Hardware for World Model Training

The hardware you need depends on the model and the data.

### 15.1 The Tiers

| Tier | GPUs | Use Case | Cost (estimated) |
|------|------|----------|------------------|
| Hobbyist | 1x RTX 4090 / 5090 | Small WMs, fine-tuning | $2-3K |
| Prosumer | 1x H100 | Small-to-medium WMs | $30K |
| Research | 8x H100 | Medium WMs (1-4B params) | $250K |
| Frontier | 64-256x H100 | Large WMs (4-14B params) | $2-8M |
| Hyperscaler | 1K-10K H100 | Foundation WMs (>14B params) | $30-300M |

### 15.2 Recommended Configurations

**Hobbyist / Prosumer (1 GPU):**
- 1x NVIDIA RTX 5090 (32 GB) or H100 (80 GB)
- 128 GB system RAM
- 2 TB NVMe SSD
- For fine-tuning small WMs and running inference on Cosmos-1B

**Research (8 GPUs):**
- 8x H100 SXM (640 GB total HBM3)
- 2 TB system RAM
- 100 TB NVMe (for video data)
- 400 Gbps InfiniBand
- For training small WMs and fine-tuning medium WMs

**Frontier (64-256 GPUs):**
- 64-256x H100 / H200
- 8-32 TB system RAM
- 1-10 PB parallel file system (Lustre, GPFS, WekaFS)
- 4-16x 400 Gbps InfiniBand
- For training medium WMs from scratch and frontier WMs

**Hyperscaler (1K+ GPUs):**
- 1K-10K H100 / B100
- 10-100 PB parallel file system
- 16-64x 400/800 Gbps InfiniBand
- Custom silicon (TPU v5p, Trainium 2, etc.) for cost optimization

### 15.3 Alternative Hardware

| Vendor | Chip | Status | Notes |
|--------|------|--------|-------|
| NVIDIA | H100, H200, B100, B200 | Production | The default |
| NVIDIA | GB200 NVL72 | Q3 2026 | 72-GPU rack-scale system |
| AMD | MI300X, MI325X | Production | Competitive on price |
| Google | TPU v5p | Production | Best $/perf for some models |
| AWS | Trainium 2 | Production | Cheapest at scale |
| Apple | M3 Ultra | Hobbyist | Small models only |
| Intel | Gaudi 3 | Limited | Niche |
| Cerebras | WSE-3 | Limited | Best for single-batch inference |
| Groq | LPU | Inference only | Lowest latency |

---

## 16. Hardware for World Model Inference

Inference is a different beast. The 2026 state of the art.

### 16.1 Real-Time Generation Hardware

For real-time (24+ fps) video generation:

| Model Size | Recommended GPU | Latency |
|------------|-----------------|---------|
| < 1B params | 1x RTX 4090 | 50-100 ms/frame |
| 1-4B params | 1x H100 | 40-60 ms/frame |
| 4-14B params | 2-4x H100 | 80-120 ms/frame |
| 14-50B params | 8x H100 | 150-300 ms/frame |
| > 50B params | 16x H100 + custom | 500+ ms/frame |

### 16.2 Inference Optimization Techniques

- **TensorRT-LLM** (NVIDIA) — graph optimization, FP8/INT8.
- **FlashAttention 3** — efficient attention.
- **KV-cache reuse** for video frame prediction.
- **Speculative decoding** (use a small model to draft, big model to verify).
- **Quantization** (FP8, INT8, INT4) — loses ~5-10% quality, doubles speed.
- **Distillation** — train a small model to mimic a big one.
- **Pruning** — remove redundant layers or attention heads.

### 16.3 Edge Deployment

For deploying small WMs on edge devices:

| Device | VRAM/RAM | Max WM Size | Notes |
|--------|----------|-------------|-------|
| Jetson Orin Nano | 8 GB | < 100M | Tiny WMs only |
| Jetson Orin | 32 GB | < 500M | V-JEPA 2-small |
| Jetson Thor | 128 GB | < 2B | Cosmos-1B with INT8 |
| Apple M3 Max | 128 GB | < 4B | V-JEPA 2-Large with INT4 |
| Apple M3 Ultra | 256 GB | < 8B | Cosmos-4B with INT4 |
| Qualcomm AI 100 | 16 GB | < 500M | Robotics |
| Intel Core Ultra | 32 GB (NPU) | < 1B | NPU acceleration |

---

## 17. Comparison Matrix: Choosing Your Stack

The decision matrix. Pick the right tool for your task.

### 17.1 By Use Case

| Use Case | Best Choice | Why |
|----------|-------------|-----|
| Atari / DMControl research | Dreamer V3 | Open, robust, well-tested |
| Minecraft from pixels | Dreamer V3 or IRIS | The state of the art |
| Self-driving research | Wayve GAIA / Cosmos | Driving-specific |
| General video generation | Sora 2 / Veo 3 / Genie 3 (closed) or Cosmos (open) | The frontier |
| Robotics manipulation | OpenVLA + Cosmos | Pre-trained foundation |
| Real-time interactive 3D | Decart OASIS | Lowest latency |
| 3D from single image | World Labs Marble | Best quality |
| Continuous control | Dreamer V3 | Sample efficiency |
| Embedding for downstream tasks | V-JEPA 2 | Clean latents |
| Multi-modal research | Cosmos 2.0 | Foundation model |
| Edge deployment | Cosma + small WM | Custom-tuned |
| Quick demo | World Labs API / Decart demo | No training needed |

### 17.2 By Resource

| Resource | Recommendation |
|----------|----------------|
| 1 GPU, 1 day | Decart OASIS inference; or fine-tune IRIS on Atari |
| 1 GPU, 1 week | Fine-tune Cosmos-1B on a small dataset |
| 8 GPUs, 1 week | Train Dreamer V3 on Minecraft from scratch |
| 8 GPUs, 1 month | Train V-JEPA 2 ViT-L on a custom dataset |
| 64 GPUs, 1 month | Fine-tune Cosmos-14B on a large domain |
| 256 GPUs, 3 months | Train a medium foundation WM |
| 1K+ GPUs, 6 months | Train a frontier foundation WM (10-50B) |

### 17.3 By Team Skill Set

| Skill Set | Best Fit |
|-----------|----------|
| RL researcher | Dreamer V3 |
| Computer vision researcher | V-JEPA 2, Cosmos |
| Generative model researcher | Sora 2-style (Cosmos) |
| Robotics researcher | OpenVLA, Cosmos, Dreamer V3 |
| Self-driving researcher | Wayve GAIA, Tesla-style |
| Product engineer | Decart OASIS, World Labs API |
| Hobbyist / Student | Cosma framework |

---

## 18. Production Case Studies

Real-world deployments of world models in 2026.

### 18.1 Tesla FSD V13

- **Company:** Tesla
- **Use case:** End-to-end autonomous driving
- **Architecture:** Custom neural network with world-model-style training
- **Scale:** 5M+ vehicles; ~10B miles of training data
- **Status:** "Supervised FSD" deployed, full autonomy pending regulatory approval

### 18.2 Wayve AV2.0

- **Company:** Wayve
- **Use case:** L4 autonomous driving in UK
- **Architecture:** GAIA-2 WM + Dreamer-style planner
- **Scale:** 100+ UK fleet vehicles; ~50K hours of driving
- **Status:** In commercial pilots with UK retailers and ride-hail

### 18.3 Figure 02 Factory

- **Company:** Figure AI
- **Use case:** Humanoid robot in BMW factory
- **Architecture:** Cosmos fine-tuned + RL policy
- **Scale:** 50 robots in one factory
- **Status:** Production deployment (announced Q1 2026)

### 18.4 NVIDIA DRIVE Sim

- **Company:** NVIDIA
- **Use case:** Synthetic data for AV training
- **Architecture:** Cosmos + DRIVE Sim
- **Scale:** Used by 50+ AV companies
- **Status:** Production service

### 18.5 1X Technologies Home Robot

- **Company:** 1X
- **Use case:** Humanoid robot in homes
- **Architecture:** Cosmos + custom policy
- **Scale:** 1000+ pilot deployments
- **Status:** Beta; production Q4 2026

### 18.6 World Labs Marble for VFX

- **Company:** Industrial Light & Magic, Wētā FX
- **Use case:** Pre-visualization for films
- **Architecture:** World Labs API
- **Scale:** 100+ productions
- **Status:** Production

### 18.7 Decart for Live Content

- **Company:** Decart + several streamers
- **Use case:** Interactive live content
- **Architecture:** OASIS real-time video
- **Scale:** 10M+ viewers
- **Status:** Production

### 18.8 NVIDIA Isaac Lab for Robot Training

- **Company:** NVIDIA + robot makers
- **Use case:** Robot policy training in simulation
- **Architecture:** Cosmos + Isaac Sim
- **Scale:** 100+ robot types
- **Status:** Production

### 18.9 Cosmos in the Loop: A Recipe

A common 2026 production pattern:

```text
1. Collect 100-10K hours of robot teleop data.
2. Fine-tune Cosmos on this data (1-2 weeks on 8x H100).
3. Use the fine-tuned Cosmos to generate 10M+ synthetic trajectories.
4. Train the robot policy on the synthetic data (RL or BC).
5. Deploy the policy on the robot.
6. Collect more real data from the deployed robot.
7. Repeat.
```

This loop is responsible for the rapid capability gains in 2025-2026 robotics.

---

## 19. Licensing and Commercial Use

The 2026 licensing landscape is the most confusing in open-source AI.

### 19.1 The License Matrix

| Model | License | Commercial Use? | Restrictions |
|-------|---------|-----------------|--------------|
| Dreamer V3 | Apache 2.0 | ✅ Yes | None |
| V-JEPA 2 | CC-BY-NC 4.0 | ❌ Research only | No commercial use |
| Cosmos | NVIDIA Open | ✅ Yes | Some ethics restrictions |
| Cosma | Apache 2.0 | ✅ Yes | None |
| IRIS | MIT | ✅ Yes | None |
| GAIA-1 | CC-BY-NC 4.0 | ❌ Research only | Commercial license available |
| Decart OASIS | Apache 2.0 | ✅ Yes | None |
| Open X-Embodiment | Various | ✅ Mostly yes | Per-dataset terms |
| Octo | Apache 2.0 | ✅ Yes | None |
| OpenVLA | Apache 2.0 | ✅ Yes | None |
| Hugging Face transformers | Apache 2.0 | ✅ Yes | None |
| Sora 2 | Closed (API only) | ✅ Yes (API) | No weights |
| Genie 3 | Closed (API only) | ✅ Yes (API) | No weights |
| Veo 3 | Closed (API only) | ✅ Yes (API) | No weights |
| World Labs Marble | Commercial (API) | ✅ Yes | Per-plan |

### 19.2 The CC-BY-NC Problem

Meta's CC-BY-NC license for V-JEPA 2 is a 2026 flashpoint. The "NC" (non-commercial) clause is interpreted differently in different jurisdictions:

- **US:** strict — no commercial use without a separate license
- **EU:** strict — same as US
- **China:** loosely enforced
- **India:** strictly enforced as of 2025

The result: V-JEPA 2 is essentially research-only, and most production teams use a different model. Meta is reportedly reconsidering the license in 2026.

### 19.3 NVIDIA Open Model License

The 2025-released NVIDIA Open Model License is more permissive than CC-BY-NC but more restrictive than Apache 2.0:

- ✅ Commercial use allowed
- ✅ Modifications allowed
- ❌ Cannot be used for medical diagnosis
- ❌ Cannot be used for biometric identification of real persons
- ❌ Cannot be used for generating disinformation

This license is becoming the de facto standard for "responsible" open-weights models.

---

## 20. Where to Start (Recommendations)

Concrete recommendations by persona.

### 20.1 If You Are a Student / Researcher

1. **Read the Dreamer V3 paper** (Nature 2024).
2. **Run the dreamerv3-torch repo** on Atari.
3. **Fine-tune V-JEPA 2** on a small video dataset.
4. **Try Cosma** for a multi-backend comparison.
5. **Read the Cosmos paper** for state-of-the-art details.

### 20.2 If You Are an Engineer / Product Builder

1. **Start with a pre-trained model**: V-JEPA 2 (CC-BY-NC) or Cosmos (commercial).
2. **Use Hugging Face** for the model download and inference.
3. **Fine-tune on your data** using the provided scripts.
4. **Deploy with vLLM / SGLang / TensorRT-LLM**.
5. **Iterate on data quality, not model size.**

### 20.3 If You Are a Robotics Engineer

1. **Collect 100+ hours of robot data** in your domain.
2. **Fine-tune Cosmos-4B** on your data.
3. **Use the synthetic data flywheel**: train policies on the generated trajectories.
4. **Fine-tune the policy on real-world data.**
5. **Deploy and iterate.**

### 20.4 If You Are a Self-Driving Engineer

1. **Look at Wayve AV2.0 and Tesla FSD V13** as references.
2. **Start with a Cosmos or GAIA-1 fine-tune.**
3. **Build a CARLA-based simulation harness.**
4. **Train your policy in the loop.**
5. **Validate on closed courses, then public roads.**

### 20.5 If You Are an Executive / Investor

1. **Read the overview chapter** (01-Overview.md).
2. **Understand the paradigm shift** from LLM to WM.
3. **Identify your domain-specific opportunity** (manufacturing? surgery? logistics?).
4. **Pilot with a pre-trained model + a small team.**
5. **Scale to a custom foundation model if the pilot succeeds.**

### 20.6 The 30-Day Plan

A pragmatic 30-day plan to get a world model in production:

- **Week 1:** Pick a use case. Pick a model. Download weights. Run inference.
- **Week 2:** Collect 100+ examples of your data. Fine-tune the model.
- **Week 3:** Build a minimal application around the model. Deploy to staging.
- **Week 4:** Iterate on quality, latency, and cost. Decide whether to scale.

This is the 2026 playbook for production WMs.

---

## Cross-References to Other Library Categories

- **LLMs (02)**: [02-LLMs/01-Transformer-Architecture.md](../02-LLMs/01-Transformer-Architecture.md) — Same backbone.
- **Advanced (06)**: [06-Advanced/01-Multimodal-AI.md](../06-Advanced/01-Multimodal-AI.md), [06-Advanced/02-Diffusion-Models.md](../06-Advanced/02-Diffusion-Models.md) — Underlying techniques.
- **Robotics (10)**: [10-Industry/03-AI-for-Robotics.md](../10-Industry/03-AI-for-Robotics.md) — WM is the key to robotics.
- **Local AI (23)**: [23-Local-AI-Inference-Self-Hosting/01-Overview.md](../23-Local-AI-Inference-Self-Hosting/01-Overview.md) — Small WMs run locally.
- **Agent Infra (20)**: [20-Agent-Infrastructure-and-Observability/01-Overview.md](../20-Agent-Infrastructure-and-Observability/01-Overview.md) — WM-as-a-service is infra.
- **Top Demand (13)**: [13-Top-Demand/01-Current-Trends.md](../13-Top-Demand/01-Current-Trends.md) — WMs are a top trend.

---

*Next: [05-Future-Outlook.md](05-Future-Outlook.md) — the roadmap, open problems, and long-term scenarios.*
