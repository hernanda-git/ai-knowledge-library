# AI Video Generation — Architectures, Models & Techniques

> June 2026

A technical deep-dive into the architectures powering modern video generation: from Diffusion Transformers to causal spatiotemporal modeling.

---

## 1. Core Architecture: Diffusion Transformers (DiT)

Video generation in 2025–2026 is dominated by the **Diffusion Transformer (DiT)** architecture, first proposed by William Peebles and Saining Xie at UC Berkeley (2023), and subsequently adopted by OpenAI (Sora), Google (Veo), Meta (Movie Gen), and Stability AI.

### 1.1 Why Transformers over U-Nets?

Traditional diffusion models used **U-Net backbones** with convolutional layers. DiT replaced convolutions with transformer blocks, yielding three critical advantages:

| Aspect | U-Net (prior) | Transformer (DiT) |
|--------|--------------|-------------------|
| Scaling | Plateaus at scale | Continues improving with compute |
| Sequence length | Limited by conv kernel | Native — handles arbitrary tokens |
| Temporal modeling | Ad-hoc (3D convs) | Native — attention across time |
| Conditioning | Cross-attention + AdaGN | In-context, adaptive layer norm |

### 1.2 DiT Architecture Overview

```
Input Video (T frames × H × W × 3)
    ↓
3D VAE Encoder (compresses spatial + temporal)
    ↓
Latent Video (T' × H' × W' × C)
    ↓
Patchify (2D × time → sequence of tokens)
    ↓
Positional Encoding (sinusoidal + time embeddings)
    ↓
[N × Transformer Blocks]
    ├── Adaptive Layer Norm (conditioned on timestep + text)
    ├── Multi-Head Self-Attention (spatiotemporal mask)
    ├── Cross-Attention (text conditioning from T5/CLIP)
    └── MLP (GELU activation)
    ↓
Unpatchify → Decode to latent → 3D VAE Decoder
    ↓
Generated Video
```

### 1.3 Spatiotemporal Attention Patterns

The core innovation is how attention handles both space and time. Three approaches have emerged:

**Full Spatiotemporal Attention** (Sora, Veo 3):
- All tokens attend to all other tokens across frames
- O(T² × H² × W²) complexity — expensive but most coherent
- Requires clever optimizations: KV caching, sliding windows

**Factorized Attention** (CogVideoX, Stable Video Diffusion):
- Spatial attention within each frame + temporal attention across frames
- O(T × H² × W² + T²) — more efficient, slightly less coherent

**Hybrid / Causal Attention** (Runway Gen-4):
- Causal masking: each frame attends to past frames only
- Enables streaming / real-time generation
- O(T² × H²/2) — good balance

---

## 2. Leading Video Models: Technical Comparison

### 2.1 Sora (OpenAI)

**Architecture**: DiT with full spatiotemporal attention, operating in a unified latent space

**Key Technical Features**:
- **3D spacetime latent patches** — variable duration, resolution, aspect ratio in one model
- **Recaptioning** — Dense video captions generated for training via a video captioner
- **World model properties** — Emergent 3D consistency, object permanence, physics
- **Diffusion Transformer** — ~8B parameters, 64× compression via 3D VAE

**Technical Specs**:
| Parameter | Value |
|-----------|-------|
| VAE Compression | 64× (spatial 8× × temporal 8×) |
| Latent Size | ~256 × 256 × C per frame (at 64× compression) |
| Patch Size | 2 × 2 × 2 (spacetime patches) |
| Text Encoder | T5-XXL |
| Training Data | Large-scale (not publicly specified) |
| Inference | 50–100 DDIM steps, CFG scale 5–10 |

### 2.2 Veo 3 (Google DeepMind)

**Architecture**: Advanced DiT with flow matching, multi-stage generation

**Key Technical Features**:
- **Flow Matching** replaces diffusion for faster, higher quality generation
- **Keyframe → interpolation** two-stage generation for extended videos
- **Camera control** explicit conditioning on camera intrinsics + extrinsics
- **Extended generation** — maintains consistency across 60+ seconds

**Flow Matching vs. Diffusion**:
```
Diffusion: q(xₜ | x₀) = N(αₜ x₀, σₜ² I)  → predict noise
Flow Match: q(xₜ | x₀) = x₀ + t × v    → predict velocity field
```

### 2.3 Runway Gen-4

**Architecture**: Hybrid DiT with spatial cascade

**Key Technical Features**:
- **Multi-motion brushes** — move multiple objects independently
- **Green screen / alpha channel** — native RGBA output
- **Image + video conditioning** — start from reference images
- **Creative control architecture** — separate motion, style, and content pathways

### 2.4 CogVideoX (Zhipu AI)

**Architecture**: 3D Causal VAE + DiT (open weights)

**Key Technical Features**:
- **3D Causal VAE** — enables arbitrary-length generation (causal convolutions)
- **Expert-annotated captions** — fine-grained video understanding training
- **Open weights under Apache 2.0** — fine-tunable
- **Chinese + English bilingual** text understanding

---

## 3. Video Generation Techniques

### 3.1 Text-to-Video (T2V)

Standard pipeline: text prompt → embedding → DiT → decode → video

Challenges:
- **Temporal binding**: "A dog jumps over a fence then turns around" — sequential actions
- **Attribute binding**: "A red car and a blue truck" — color attribution per object
- **Counting**: "Three cats on a sofa" — numerical consistency

Solutions:
- **Dense captioning** of training videos with temporal annotations
- **Layout conditioning** — bounding boxes + text per region
- **Motion-specific prompting** — camera move, speed, transition cues

### 3.2 Image-to-Video (I2V)

Generate video from a single starting image. Critical for:
- Animation from concept art
- Product visualization
- Deepfake / face animation

Techniques:
- **Image as first frame** — condition all subsequent frames
- **CLIP image embedding** → cross-attention injection
- **Ip-Adapter style** — image prompt adapter for video

### 3.3 Video-to-Video (V2V)

Stylize, modify, or extend existing video:
- Style transfer: "Make this video look like Van Gogh"
- Object replacement: "Change the car to a bicycle"
- Extrapolation: "Continue this scene for 5 more seconds"

### 3.4 Camera Control

Modern models accept explicit camera parameters:
```python
camera_params = {
    'pan': [-30, 0],         # degrees, start to end
    'tilt': [0, 15],
    'zoom': [1.0, 1.5],      # zoom factor
    'dolly': ['mid', 'close'],
    'crane': [0, 45],
    'track': [-2, 2]         # meters, left-right
}

def generate_with_camera(
    prompt="A futuristic city street at night",
    camera=camera_params,
    duration=10
):
    # Camera conditioning injected into DiT via adaptive layer norm
    # Model learns camera trajectories from paired training data
    pass
```

---

## 4. Temporal Consistency and Long-Form Generation

### 4.1 The Consistency Problem

Longer videos amplify accumulation of drift, object morphing, and background inconsistency. Key solutions:

| Technique | Description | Models |
|-----------|-------------|--------|
| **Noise initialization** | Same noise seed across frames | Sora, Veo 3 |
| **Keyframe + interpolation** | Generate sparse keyframes, fill between | Veo 3, Pika 2.0 |
| **Temporal attention masking** | Hierarchical attention windows | CogVideoX |
| **3D convolutions** | Causal 3D conv preserves time axis | Runway Gen-4 |
| **Object-level tracking** | Per-object latent codes maintained across frames | Research |
| **Memory / recurrent conditioning** | Frame N conditioned on hidden state of N-1 | Streaming models |

### 4.2 Extended Generation Pipeline

```
Prompt + Camera Params
    ↓
Stage 1: Generate Keyframes (every 24th frame)
    ├── High-quality DiT (slower, more steps)
    └── Each keyframe conditioned on prior keyframe + text
    ↓
Stage 2: Interpolate In-Between Frames
    ├── Lighter DiT (fewer steps, smaller model)
    └── Conditioned on surrounding keyframes + motion vectors
    ↓
Stage 3: Upscale + Refine
    ├── Spatial super-resolution
    └── Temporal smoothing filter
```

---

## 5. Benchmarking Video Generation

### 5.1 Key Benchmarks

| Benchmark | What It Measures | Leading Model (2026) |
|-----------|-----------------|---------------------|
| **VBench** | Comprehensive: quality, consistency, dynamics | Veo 3 |
| **EvalCrafter** | Prompt adherence, temporal consistency | Sora |
| **UCF-101** | Action recognition on generated videos | CogVideoX-5B |
| **FVD (Fréchet Video Distance)** | Distribution similarity to real videos | Veo 3 (lowest FVD) |
| **CLIP Score** | Text-video alignment | Various (~35+) |
| **Human Preference** | Human raters comparing outputs | Runway Gen-4 |

### 5.2 Evaluation Dimensions

- **Per-frame quality**: IS (Inception Score), FID
- **Temporal consistency**: FVD, warping error, flow consistency
- **Text alignment**: CLIP score, VQA on generated content
- **Physics**: Physical plausibility checks (object permanence, gravity)
- **Diversity**: Intra-class variance, multi-modal coverage

---

## 6. Inference Optimization

### 6.1 Speed vs. Quality Trade-offs

| Method | Speed Gain | Quality Impact |
|--------|-----------|----------------|
| Fewer sampling steps (10 vs 50 DDIM) | 5× | Moderate (perceptually small) |
| CFG scale reduction | 1× (same steps) | Significant (less saturation) |
| Latent pruning | 2× | Small (depends on compression) |
| Parallel frame generation | 3-8× | Medium (temporal artifacts) |
| Quantization (FP16 → INT8) | 2× | Small to moderate |
| Flash Attention (FP8) | 2× | None |

### 6.2 KV Cache for Video

Video DiT generates many frames, each requiring full attention. KV caching strategies:

- **Full cache**: Store all prior frame KVs for temporal attention
- **Windowed cache**: Only last N frames' KVs (fixed memory budget)
- **Compressed cache**: Cluster/merge similar KVs across frames

---

## 7. Open-Source Ecosystem

### 7.1 Available Models & Weights

| Model | License | Params | Hugging Face |
|-------|---------|--------|-------------|
| Stable Video Diffusion | Stability Research License | 2.3B | stabilityai/stable-video-diffusion-img2vid |
| CogVideoX-5B | Apache 2.0 | 5B | THUDM/CogVideoX-5B |
| Mochi 1 (Genmo) | Apache 2.0 | 10B | genmo/mochi-1 |
| Open-Sora Plan | Apache 2.0 | ~1B | hpcaitech/Open-Sora |
| LVD (Latte) | MIT | 1.3B | Vchitect/Latte |

### 7.2 Fine-Tuning Tools

- **Hugging Face Diffusers** — Video training scripts
- **LoRA for video** — Parameter-efficient fine-tuning of DiT blocks
- **xFormers / FlashAttention** — Memory efficient attention for video
- **DeepSpeed ZeRO** — Distributed training across GPUs

---

*This document is part of the AI Knowledge Library — 28-AI-Video-Audio-Generation directory.*