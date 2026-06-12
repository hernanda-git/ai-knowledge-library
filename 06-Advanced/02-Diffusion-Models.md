# Diffusion Models: Theory, Architecture, and Applications

## Table of Contents
1. [Introduction](#1-introduction)
2. [Mathematical Foundation](#2-foundations)
3. [Forward and Reverse Processes](#3-processes)
4. [Denoising Diffusion Probabilistic Models](#4-ddpm)
5. [Noise Schedules](#5-schedules)
6. [Sampling Methods](#6-sampling)
7. [Latent Diffusion](#7-latent)
8. [Conditioning Mechanisms](#8-conditioning)
9. [Classifier-Free Guidance](#9-cfg)
10. [Score-Based Models](#10-score-based)
11. [Model Architectures](#11-architectures)
12. [Applications](#12-applications)
13. [Advanced Training Techniques](#13-training)
14. [Recent Advances](#14-advances)
13a. [Practical Training & Sampling Code](#13a-practical-training--sampling-code)
14. [Recent Advances](#14-advances)
    - 14a. [Frameworks Comparison](#14a-comparison-of-diffusion-frameworks-and-apis)
    - 14b. [Deployment & Optimization](#14b-practical-deployment-and-optimization)
15. [Cross-References](#15-cross-references)
16. [Ethics, Safety, and Societal Impact](#16-ethics-safety-and-societal-impact)

---

## 1. Introduction

Diffusion models are a class of generative models that learn to reverse a gradual noising process. Unlike GANs (adversarial training) or VAEs (variational bounds), diffusion models learn through a simple denoising objective and produce state-of-the-art image, video, audio, and 3D content.

### Timeline
- 2015: First diffusion model formulation (Sohl-Dickstein)
- 2020: DDPM — practical training and sampling (Ho et al.)
- 2021: DDIM — faster sampling (Song et al.)
- 2022: Stable Diffusion — latent diffusion, consumer-grade image generation
- 2023: DALL-E 3, Midjourney v6, SDXL — production quality
- 2024: Flux, SD3, Sora — video diffusion, transformer backbones
- 2025-2026: Real-time generation, 4K video, 3D diffusion

---

## 2. Mathematical Foundation

### 2.1 The Forward Process (Noising)

Given data x₀ ~ p_data(x), define a Markov chain that gradually adds Gaussian noise:

q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)

where β_t is the variance schedule (increasing from ~1e-4 to ~0.02).

In closed form:
x_t = √(ᾱ_t) x₀ + √(1-ᾱ_t) ε, where ε ~ N(0,I)

Key: any noisy sample x_t can be expressed directly in terms of x₀ without iterating.
ᾱ_t = ∏_{s=1}^t (1-β_s) — cumulative product.

### 2.2 The Reverse Process (Denoising)

Learn to reverse the noising process:

p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t²I)

The model predicts the mean μ_θ, which can be reparameterized as predicting:
1. Direct prediction: μ_θ(x_t, t)
2. Noise prediction: ε_θ(x_t, t) — predict the noise ε that was added
3. Data prediction: x̂_θ(x_t, t) — predict the original data

**Noise prediction is the standard formulation** because it's more stable.

### 2.3 Training Objective

Simple MSE loss on noise prediction:
L = E_{t,x₀,ε}[||ε - ε_θ(x_t, t)||²]

where t is uniformly sampled from [1,T], x₀ from the data distribution, and ε from N(0,I).

---

## 3. Forward and Reverse Processes

### 3.1 DDPM Forward Process

β_t schedule (linear):
β₁ = 1e-4, β_T = 0.02, T = 1000

Properties:
- As t→T, x_t ≈ N(0,I) (standard Gaussian)
- The process is Markovian (no dependence on past beyond immediate previous step)
- Forward process requires no learning (fixed variance schedule)

### 3.2 DDPM Reverse Process

p_θ(x_{t-1} | x_t) = N(x_{t-1}; (1/√α_t)(x_t - (β_t/√(1-ᾱ_t))ε_θ(x_t,t)), σ_t²I)

This is the **denoising step**: given x_t and predicted noise ε_θ, compute the mean of x_{t-1}.

σ_t² = β_t (when using fixed variance) or can be learned.

### 3.3 Reparameterization Trick

Deriving the reverse step:
x_{t-1} = (1/√α_t)(x_t - (β_t/√(1-ᾱ_t))ε_θ(x_t,t)) + σ_t z, where z ~ N(0,I)

This is the sampling step — repeated T times from x_T ~ N(0,I) to produce x₀.

---

## 4. Denoising Diffusion Probabilistic Models (DDPM)

### 4.1 Standard DDPM (Ho et al., 2020)

| Component | Configuration |
|-----------|--------------|
| Architecture | U-Net with attention |
| Noise schedule | Linear, β₁=1e-4, β_T=0.02 |
| Steps | T=1000 |
| Training | MSE on ε prediction |
| Sampling | Langevin-like: x_{t-1} = 1/√α_t(x_t - β_t/√(1-ᾱ_t)ε_θ) + σ_t z |
| Samples | 3.5h for 50K 32×32 samples (1000 steps) |

### 4.2 Improving DDPM

| Method | Change | Speed/Quality |
|--------|--------|:-------------:|
| DDIM | Non-Markovian, deterministic reverse | Same quality in 50-100 steps (10-20× faster) |
| Improved DDPM | Learned σ_t, cosine schedule | Better FID with fewer parameters |
| DPM-Solver | ODE solver-based sampling | 10-50 steps, high quality |
| Progressive Distillation | Distill 1000 → 100 → 10 → 1 step | One-step generation |

---

## 5. Noise Schedules

The noise schedule β_t (or the derived ᾱ_t) determines how quickly information is destroyed.

### 5.1 Linear Schedule (DDPM)
β_t increases linearly from β₁ to β_T.
- Pros: Simple, well-studied
- Cons: Too much noise destroyed early, suboptimal for high-resolution images
- ᾱ_T ≈ 0 — signal is fully destroyed at final step

### 5.2 Cosine Schedule (Improved DDPM)
ᾱ_t = cos²((t/T + s)/(1+s) · π/2), s=0.008
- Pros: Gradual information destruction, better log-likelihood, slightly better FID
- Cons: Requires more steps for convergence

### 5.3 Custom Schedules

| Schedule | Shape | Use Case |
|----------|-------|----------|
| Square-root | √(1-t/T) | Video diffusion |
| Sigmoid | σ-shaped | High diversity generation |
| Exponential | exp(-βt) | Fast diffusion (fewer steps) |
| Learned | Parameterized β_tθ | Task-specific optimization |
| v-prediction | v = α_tε - σ_tx | Stable training at high resolution |

### 5.4 v-parameterization (v-prediction)
Instead of predicting ε, predict v = α_tε - σ_tx, where:
- α_t = √(ᾱ_t), σ_t = √(1-ᾱ_t)
- v is the "velocity" along the diffusion path
- More stable for high-resolution generation
- Used by: Stable Diffusion 3, SDXL, Imagen

---

## 6. Sampling Methods

### 6.1 DDPM Sampling (1000 steps)
x_T ~ N(0,I)
For t = T,...,1:
    x_{t-1} = 1/√α_t(x_t - β_t/√(1-ᾱ_t)ε_θ(x_t,t)) + σ_t z

Quality: High (best likelihood). Speed: Slow (1000 forward passes).

### 6.2 DDIM Sampling (10-100 steps)
Deterministic reverse: x_{t-1} = √(ᾱ_{t-1})x̂_θ + √(1-ᾱ_{t-1})ε_θ
where x̂_θ = (x_t - √(1-ᾱ_t)ε_θ)/√ᾱ_t

Quality: Good (slightly lower diversity). Speed: 10-100× faster.

### 6.3 DPM-Solver
Treats the reverse process as an ODE (probability flow ODE). Uses ODE solver (DPM-Solver, DPM-Solver++) for faster sampling.

Quality: Excellent (matches DDPM). Speed: 10-50 steps.

### 6.4 Consistency Models
Train a model to directly map from any noisy state to data in one step. Uses distillation or direct training.

Quality: Acceptable (lags behind multi-step). Speed: 1 step.

### 6.5 Sampling Schedule Comparison

| Method | Steps | FID (ImageNet 256) | Sampling Time |
|--------|:----:|:------------------:|:-------------:|
| DDPM | 1000 | 3.17 | 100% |
| DDIM | 50 | 4.67 | 5% |
| DPM-Solver++ | 20 | 2.94 | 2% |
| DPM-Solver++ | 10 | 3.50 | 1% |
| Consistency Model | 1 | 6.20 | 0.1% |

---

## 7. Latent Diffusion

### 7.1 Core Idea (Rombach et al., 2022)
Move the diffusion process from pixel space to a learned latent space.

**Architecture:**
1. VAE encoder: image → latent (compression factor 8×)
2. Latent diffusion: noise/denoise in latent space
3. VAE decoder: latent → image

**Benefits:**
- 8× compression → 64× less computation per step
- Higher resolution possible (512×512 → 1024×1024 → 4K)
- Conditional generation via cross-attention on text embedding

### 7.2 Stable Diffusion Family

| Version | Base | Specialization |
|---------|------|---------------|
| SD 1.5 | UNet (860M) | General purpose, most fine-tuned models |
| SD 2.1 | UNet + v-prediction | Improved quality, depth control |
| SDXL | Larger UNet (2.6B), dual text encoder | High quality, native 1024×1024 |
| SD 3 | Replaced UNet with MMDiT | Better text rendering, anatomy |
| SD 3.5 | Improved MMDiT | Fastest high-quality generation |
| Flux | New architecture (Black Forest Labs) | Best-in-class detail and prompt adherence |
| Flux Pro | Scaled-up Flux | Photorealism benchmark leader |

---

## 8. Conditioning Mechanisms

### 8.1 Text Conditioning

**Cross-Attention (Stable Diffusion):**
text → text encoder → K,V tokens → cross-attend with image features
t = W_Q · image_feature, K = W_K · text_embedding, V = W_V · text_embedding

**Conditioning on Text Encoder Type:**
- SD 1/2: CLIP text encoder (77 tokens, 512-dim)
- SDXL: CLIP (77 tokens) + T5 (77 tokens, pooled 1280-dim)
- SD 3: T5-XXL (512 tokens, full attention over all text)
- DALL-E 3: Custom text encoder + upsampler
- Flux: T5-XXL + CLIP (multiple encoders)

### 8.2 Other Conditioning Types

| Type | Method | Examples |
|------|--------|---------|
| Image (img2img) | Add noise to image, denoise with conditioning | SD img2img, ControlNet |
| Edge maps | Canny edge → ControlNet | Architectural design |
| Depth maps | MiDaS/Depth-Anything → ControlNet | 3D-consistent generation |
| Pose | OpenPose skeleton → ControlNet | Character generation |
| Segmentation | SAM/semantic maps → ControlNet | Object manipulation |
| Style | Reference image → IP-Adapter | Style transfer |
| Personalization | Fine-tune on person/object → DreamBooth, LoRA | Custom portraits |

---

## 9. Classifier-Free Guidance (CFG)

### 9.1 The CFG Formula

ε_θ(x_t, t, c) + g × (ε_θ(x_t, t, c) - ε_θ(x_t, t, ∅))

where:
- c is the conditioning (text prompt)
- ∅ is unconditional (empty prompt)
- g is the guidance scale (typically 3-14)

**Effect:**
- g=0: unconditional generation (high diversity)
- g=1: conditional generation (standard)
- g>1: amplified conditioning (higher quality, lower diversity)
- g=7-12: typical for SD 1/2 (best quality/diversity tradeoff)
- g=10-14: typical for SDXL (allows higher guidance)
- g=2-5: typical for Flux (lower guidance needed)

### 9.2 Why CFG Works
The difference (ε_c - ε_u) points in the direction of "more conditioning-like." Amplifying this direction produces samples that are more faithful to the prompt.

### 9.3 Limitations of CFG
- Amplifies artifacts at high g (oversaturation, oversharpening)
- Reduces diversity (all samples look similar)
- Doesn't work well for all model types
- CFG can cause "burn-in" effects in video diffusion

---

## 10. Score-Based Models

### 10.1 Score Matching
Diffusion is equivalent to learning the *score function*: ∇_x log p(x) — the direction of increasing probability density.

**Relationship:** ε_θ(x_t, t) = -σ_t ∇_x log p_t(x_t)
The predicted noise points toward higher probability density (negative score = direction of increasing density).

### 10.2 Score SDE (Song et al., 2021)
Unified framework treating diffusion as a Stochastic Differential Equation (SDE):

dx = f(x,t)dt + g(t)dw (forward)
dx = [f(x,t) - g(t)²∇_x log p_t(x)]dt + g(t)dw̄ (reverse)

Where:
- f(x,t): drift coefficient (deterministic component)
- g(t): diffusion coefficient (noise)
- dw: Wiener process (Brownian motion)

### 10.3 Probability Flow ODE
The SDE has a corresponding ODE with the same marginal densities: deterministic, faster to sample, enables latent space interpolation.

---

## 11. Model Architectures

### 11.1 U-Net Backbone

Standard architecture for diffusion up to SDXL:
1. **Encoder:** Downsampling blocks (2× → 4× → 8× → 16×)
2. **Middle:** Single bottleneck with self-attention
3. **Decoder:** Upsampling blocks with skip connections from encoder
4. **Time embedding:** Sinusoidal positional encoding → MLP → add to features
5. **Cross-attention:** Insert after self-attention in each block

**SDXL U-Net:** 2.6B parameters, 3× larger than SD 1.5. Residual blocks added, more attention, dual cross-attention.

### 11.2 Transformer Backbone (MMDiT)

Used by SD3, SD3.5, and Flux. MMDiT = Multi-Modal Diffusion Transformer.

**Key advantages over U-Net:**
- Full bidirectional attention between text and image tokens
- Better text-image alignment (text rendering, complex concepts)
- More parameter-efficient (scales better with compute)
- Handles multiple modalities naturally

**Flux Architecture:**
- 12B parameters
- MMDiT backbone
- T5-XXL text encoder
- Native 1024×1024 generation
- Best-in-class photorealism

### 11.3 DiT (Diffusion Transformer, Peebles & Xie, 2023)
Pure transformer for diffusion. Patched image tokens + class labels + time steps as input. Scaled better than U-Net at large compute budgets.

---

## 12. Applications

### 12.1 Text-to-Image Generation
- Best models: Flux Pro, Midjourney v7, DALL-E 3, SD 3.5
- Resolution: 1024×1024 native, up to 4K with upscalers
- Quality metrics: FID < 5, CLIP score > 0.32

### 12.1a Image Generation Model Comparison (2026)

| Model | Quality | Speed | Prompt Adherence | Text Rendering | Cost | Control | Open? |
|-------|:------:|:-----:|:----------------:|:--------------:|:----:|:-------:|:----:|
| **Flux Pro** | ⭐⭐⭐⭐⭐ | ⚡⚡⚡ | 99% | ✅✅✅ | $0.05/image | High | ✅ Weight |
| **Midjourney v7** | ⭐⭐⭐⭐⭐ | ⚡⚡⚡ | 95% | ✅✅ | $10-30/mo | Medium | ❌ |
| **DALL-E 3** | ⭐⭐⭐⭐½ | ⚡⚡⚡⚡ | 97% | ✅✅✅ | $0.04/image | Low | ❌ |
| **SD 3.5** | ⭐⭐⭐⭐ | ⚡⚡⚡⚡ | 93% | ✅✅ | Free (local) | Very High | ✅ MIT |
| **Ideogram 3** | ⭐⭐⭐⭐ | ⚡⚡ | 96% | ✅✅✅✅ | Pay-per-use | Medium | ❌ |
| **Imagen 3** | ⭐⭐⭐⭐½ | ⚡⚡⚡ | 94% | ✅✅ | Through GCP | Low | ❌ |
| **Recraft V3** | ⭐⭐⭐⭐ | ⚡⚡⚡⚡ | 92% | ✅✅ | Freemium | High | ❌ |
| **FLUX.1 dev** | ⭐⭐⭐⭐½ | ⚡⚡ | 95% | ✅✅ | Free (local) | High | ✅ Apache |

**Recommendation by use case:**
- **Best overall quality:** Flux Pro or Midjourney v7
- **Best text rendering:** Ideogram 3 (best-in-class) or DALL-E 3
- **Best for local/self-hosted:** SD 3.5 or FLUX.1 dev (open weights, ControlNet, LoRA)
- **Best for API/automation:** Flux Pro (fastest) or DALL-E 3 (best prompt adherence)
- **Best free (cloud):** Recraft V3 (free tier available) or SD 3.5 (via Replicate free predictions)

### 12.2 Text-to-Video Generation
- Sora (OpenAI): 60-second 1080p video
- Gen-3 (Runway): Professional video editing
- Kling (Kuaishou): Physics-aware generation
- CogVideoX (Zhipu): Open-source video diffusion

### 12.3 Text-to-3D Generation
- Point-E / Shap-E (OpenAI): Diffusion on 3D point clouds/meshes
- DreamFusion/SJC: Score distillation sampling (SDS)
- Zero-1-to-3: Novel view synthesis via diffusion

### 12.4 Audio Generation
- Stable Audio (Stability AI): Diffusion on spectrograms
- MusicGen (Meta): Text-to-music diffusion
- AudioLDM: Latent diffusion for audio

---

## 13. Advanced Training Techniques

| Technique | Description | Benefit |
|-----------|-------------|---------|
| **EDM** | Preconditioning, loss weighting | Optimal training dynamics |
| **Min-SNR** | Signal-to-noise-ratio weighting | Better high-resolution training |
| **Offset Noise** | Add small offset to noise | Better dark/bright image generation |
| **Progressive Growing** | Start low-res, gradually increase | Stabilizes high-res training |
| **LSGM** | Latent + score-based in VAE | Combined generative power |
| **SiT** | Interpolating between models | Smoother latent space |

---

## 14. Recent Advances

### 14.1 Real-Time Generation (2025-2026)
- Consistency Models (1-step generation)
- Adversarial Diffusion Distillation (ADD)
- Latent Consistency Models (LCM)
- **Flow matching:** Alternative to diffusion with straighter paths

### 14.2 Flow Matching

Flow matching (Lipman et al., 2023) is an alternative generative modeling paradigm that has largely superseded diffusion for state-of-the-art image and video generation (used by Flux, SD3, Sora).

#### Mathematical Formulation

Instead of gradually adding and removing noise (diffusion), flow matching learns a **continuous, deterministic transformation** (a "flow") from a simple prior distribution p₁ (e.g., N(0,I)) to the data distribution p₀:

| Aspect | Diffusion Models | Flow Matching |
|--------|:----------------:|:-------------:|
| **Forward path** | Stochastic (SDE): add noise gradually | Deterministic (ODE): linear interpolation |
| **Reverse path** | Denoise step-by-step | Follow the learned vector field backwards |
| **Training target** | Predict noise ε added at step t | Predict velocity v = dx/dt at time t |
| **Sampling** | 10-1000 steps (depending on method) | 1-50 steps (straighter paths) |
| **Deterministic?** | Can be (DDIM) or stochastic (DDPM) | Always deterministic |
| **Latent space structure** | Complex, curved trajectories | Straight, interpolatable paths |
| **Key advantage** | Well-understood theory | Faster sampling, better latent interpolation |

#### Key Concepts

- **Velocity field v_t(x):** A vector field that specifies how a point x at time t should move to follow the flow from noise to data
- **Rectified flow (Liu et al., 2023):** The straight-line path between prior and data. Training objective: minimize ||v_θ(x_t, t) - (x₁ - x₀)||² where x_t = (1-t)·x₀ + t·x₁
- **Conditional flow matching (CFM):** Train on individual data points with known conditional paths; the marginal vector field emerges automatically

#### Why Flow Matching Won

1. **Faster sampling:** 1-4 steps for good quality (vs 10-50 for diffusion)
2. **Better latent interpolation:** Linear paths in latent space → meaningful interpolations between images
3. **Simpler training:** No noise schedule, no variance preservation, no SNR weighting
4. **Deterministic by nature:** No stochastic sampling needed → consistent, repeatable outputs
5. **Natural video extension:** Straight paths extend naturally to video (trajectories in time)

**Used by:** Flux (Black Forest Labs), Stable Diffusion 3 & 3.5 (Stability AI), Sora (OpenAI), Movie Gen (Meta)

---

## 14a. Comparison of Diffusion Frameworks and APIs

| Framework | UI Type | Model Support | ControlNets | LoRA | Video Gen | Ease of Use | Best For |
|-----------|:-------:|:-------------:|:-----------:|:----:|:---------:|:-----------:|----------|
| **Diffusers (🤗)** | Python API | 500+ models | ✅ Full | ✅ | ✅ | 🟡 Medium | Developers, researchers, custom pipelines |
| **ComfyUI** | Node graph | 300+ models | ✅ Full | ✅ | ✅ | 🔴 Workflow | Advanced users, complex pipelines, production |
| **Automatic1111** | Web UI | 200+ models | ✅ Full | ✅ | ❌ | 🟢 Easy | Beginners, SD 1.5/XL users, extension ecosystem |
| **Forge** | Web UI | 200+ models | ✅ Full | ✅ | ❌ | 🟢 Easy | A1111 successor, faster, less memory |
| **InvokeAI** | Web UI | 100+ models | ✅ | ✅ | ❌ | 🟢 Easy | Artists, canvas-based editing, inpainting |
| **Krita AI** | Plugin | 100+ models | ✅ | ✅ | ❌ | 🟢 Easy | Artists using Krita, seamless in-painting workflow |
| **Fooocus** | Web UI | SDXL/3 | ⚠️ Limited | ❌ | ❌ | 🟢 Very Easy | One-click generation, beginners, no prompt engineering |
| **SwarmUI** | Web UI | 200+ models | ✅ | ✅ | ⚠️ | 🟢 Easy | ComfyUI backend with A1111-like UI |
| **SD Next** | Web UI | 100+ models | ✅ | ✅ | ❌ | 🟢 Easy | A1111 fork with optimizations |

**Quick start recommendation:**
- **"I want the best quality now"** → ComfyUI (most capable, steep learning curve)
- **"I just want to generate images"** → Fooocus or Forge (one-click)
- **"I'm a developer"** → Diffusers (Python API, full control)
- **"I'm an artist"** → InvokeAI or Krita AI (canvas-oriented)
- **"I need video generation"** → ComfyUI or Diffusers

---

## 13a. Practical Training & Sampling Code

### 13a.1 DDPM Training Loop (PyTorch)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding used in diffusion models."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-torch.arange(half).float() * torch.log(torch.tensor(10000.0)) / half)
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class SimpleUNet(nn.Module):
    """Minimal U-Net for diffusion on small datasets (e.g., CIFAR-10, MNIST)."""
    def __init__(self, in_channels=3, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        # Encoder
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1, stride=2)
        # Bottleneck
        self.bottleneck = nn.Conv2d(256, 256, 3, padding=1)
        # Decoder (with skip connections)
        self.up1 = nn.ConvTranspose2d(256 + time_dim, 128, 4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(128 + 128, 64, 4, stride=2, padding=1)
        self.out = nn.Conv2d(64, in_channels, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        # Encode
        x1 = F.silu(self.conv1(x))
        x2 = F.silu(self.conv2(x1))
        x3 = F.silu(self.conv3(x2))
        # Bottleneck
        x3 = self.bottleneck(x3)
        # Decode with time conditioning
        t_expanded = t_emb[:, :, None, None].expand(-1, -1, x3.shape[2], x3.shape[3])
        x = torch.cat([x3, t_expanded], dim=1)
        x = F.silu(self.up1(x))
        # Skip connection: trim if sizes mismatch
        h_diff = x.shape[2] - x2.shape[2]
        if h_diff > 0:
            x = x[:, :, h_diff//2:-(h_diff - h_diff//2) or None, :]
        elif h_diff < 0:
            x2 = x2[:, :, -h_diff//2:-( -h_diff - -h_diff//2) or None, :]
        x = torch.cat([x, x2], dim=1)
        x = F.silu(self.up2(x))
        h_diff = x.shape[2] - x1.shape[2]
        if h_diff > 0:
            x = x[:, :, h_diff//2:-(h_diff - h_diff//2) or None, :]
        elif h_diff < 0:
            x1 = x1[:, :, -h_diff//2:-( -h_diff - -h_diff//2) or None, :]
        x = torch.cat([x, x1], dim=1)
        return self.out(x)

def ddpm_training_step(model, x, optimizer, device, beta_start=1e-4, beta_end=0.02, T=1000):
    """Single DDPM training step: sample random t, add noise, predict noise."""
    batch_size = x.shape[0]
    t = torch.randint(0, T, (batch_size,), device=device).long()
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    noise = torch.randn_like(x)
    alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)
    x_noisy = alpha_bar_t.sqrt() * x + (1 - alpha_bar_t).sqrt() * noise

    predicted_noise = model(x_noisy, t.float() / T)
    loss = F.mse_loss(predicted_noise, noise)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# @ddpm_sample
def ddpm_sample(model, device, img_shape=(3, 32, 32), T=1000):
    """Generate a batch of 1 sample using the reverse diffusion process."""
    betas = torch.linspace(1e-4, 0.02, T, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    x = torch.randn((1, *img_shape), device=device)
    with torch.no_grad():
        for t in reversed(range(T)):
            t_tensor = torch.full((1,), t, device=device, dtype=torch.float) / T
            z = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
            alpha_bar = alpha_bars[t]
            alpha = alphas[t]
            pred_noise = model(x, t_tensor)
            x = (1 / alpha.sqrt()) * (x - (1 - alpha) / (1 - alpha_bar).sqrt() * pred_noise) + betas[t].sqrt() * z
    return x
```

### 13a.2 Rectified Flow Training (Simplified)

```python
def rectified_flow_loss(model, x, device, n_timesteps=100):
    """Training objective for rectified flow: learn the straight path z₁ - z₀."""
    batch_size = x.shape[0]
    t = torch.rand(batch_size, 1, 1, 1, device=device)
    noise = torch.randn_like(x)
    # Interpolate: x_t = (1-t) * x_0 + t * noise
    x_t = (1 - t) * x + t * noise
    # Target velocity: v = noise - x (straight line direction)
    v_target = noise - x
    v_pred = model(x_t, t.squeeze())
    return F.mse_loss(v_pred, v_target)

def rectified_flow_sample(model, device, img_shape=(3, 32, 32), steps=50):
    """Sample from a rectified flow model using ODE Euler stepping."""
    x = torch.randn((1, *img_shape), device=device)
    dt = 1.0 / steps
    with torch.no_grad():
        for i in range(steps):
            t_tensor = torch.full((1,), i * dt, device=device, dtype=torch.float)
            v = model(x, t_tensor)
            x = x + v * dt
    return x
```

### 13a.3 Video Diffusion with Temporal Attention

Video diffusion extends image diffusion by adding a **temporal dimension** — treating video as a sequence of frames with both spatial and temporal coherence:

```python
class TemporalAttentionBlock(nn.Module):
    """Self-attention across time dimension for video diffusion."""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B, F, H*W, C) where F = frames
        B, F, N, C = x.shape
        x_in = x
        x = self.norm(x)
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, F, N, self.num_heads, -1).transpose(2, 3) for t in qkv]
        # Temporal attention: aggregate across frames (F dimension)
        attn = (q @ k.transpose(-2, -1)) * (C // self.num_heads) ** -0.5
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(2, 3).reshape(B, F, N, C)
        return self.proj(out) + x_in
```

**Key architectural differences for video:**
| Component | Image Diffusion | Video Diffusion |
|-----------|----------------|-----------------|
| Input shape | (B, C, H, W) | (B, F, C, H, W) — F = frames |
| Attention | Spatial only | Spatial + Temporal (3D) |
| Conditioning | Text prompt | Text + first frame + camera motion |
| Temporal consistency | N/A | Optical flow loss, frame interpolation |
| Compute | ~1-10 GFLOPS/frame | ~10-100 GFLOPS/frame, F× more memory |
| Key challenge | Image quality | Frame-to-frame coherence |

**Sora (OpenAI)** popularized the DiT backbone for video. Its architecture factors spatial and temporal patches: each (t, h, w) patch is treated as a token, and full spatiotemporal attention captures long-range video dynamics. Later models (CogVideoX, Kling, Runway Gen-3) use 3D-VAE + temporal attention layers inserted into existing image diffusion U-Nets or transformers.

### 13a.4 Key Architectural Innovations (2024-2026)

| Innovation | Model | Impact |
|-----------|-------|--------|
| **MMDiT** (Multi-Modal Diffusion Transformer) | SD3, Flux | Replaces U-Net; joint text-image attention |
| **3D-VAE** | CogVideoX | Compresses video in space and time |
| **Rectified Flow** | Flux, SD3 | Straight, deterministic generation paths |
| **Joint Text-Image Training** | DALL-E 3 | Caption rewriting for better alignment |
| **Adversarial Distillation** | SDXL Turbo, LCM | 1-4 step generation with GAN loss |
| **ControlNet++** | Various | Improved conditioning on edge/depth/pose/seg maps |

---

## 14b. Practical Deployment and Optimization

Deploying diffusion models in production requires careful management of memory, latency, and throughput. This section provides practical guidance.

### Memory Optimization

| Technique | Saving | Trade-off | Implementation |
|-----------|:------:|:---------:|----------------|
| **FP16 inference** | ~50% VRAM | Negligible quality loss | `.half()` on model and inputs; PyTorch AMP with `torch.cuda.amp` |
| **VAE tiling** | 30-50% VRAM | Slight boundary artifacts | Process image in overlapping tiles; merge with feathering |
| **Sequential offloading** | Up to 80% VRAM | 2-5× slower | Offload non-active layers to CPU; Diffusers `enable_sequential_cpu_offload()` |
| **Model CPU offload** | Up to 90% VRAM | 3-10× slower | Keep entire model on CPU, move to GPU per step; `enable_model_cpu_offload()` |
| **Sliced attention** | 30% VRAM | ~10% slower | Compute attention in chunks; `enable_attention_slicing()` |
| **xFormers memory-efficient attention** | 20-50% VRAM | Same speed | `enable_xformers_memory_efficient_attention()` — requires xFormers package |
| **torch.compile** | 10-20% VRAM | 1.5-2× faster compile once | `model = torch.compile(model, mode='reduce-overhead')` |

### Batch Size and Throughput Tuning

| Batch Size | VRAM (SDXL, 1024×1024) | Throughput (img/s) | Latency (s/img) | Use Case |
|:----------:|:----------------------:|:------------------:|:---------------:|----------|
| 1 | ~8 GB | 0.5-1.0 | 1.0-2.0 | Interactive / real-time |
| 2 | ~12 GB | 0.8-1.5 | 1.3-1.7 | Batch generation, low-latency |
| 4 | ~20 GB | 1.2-2.0 | 2.0-3.3 | Production batch (moderate) |
| 8 | ~36 GB | 1.5-2.5 | 3.2-5.3 | Offline batch rendering |
| 16 | ~68 GB | 1.8-2.8 | 5.7-8.9 | Bulk generation (requires 80GB GPU) |

**Guidelines:**
- Start with batch size 1, increase until VRAM is ~90% utilized
- SD 1.5 uses ~50% less VRAM than SDXL at same resolution
- Flux uses ~30% more VRAM than SDXL at same resolution
- For real-time applications (< 2s latency): keep batch size 1-2, reduce steps (LCM at 4 steps)

### Resolution and Aspect Ratio Scaling

| Resolution | SD 1.5 VRAM | SDXL VRAM | Flux VRAM | Quality Impact |
|:----------:|:----------:|:---------:|:---------:|:--------------:|
| 512×512 | 3.0 GB | — | — | Baseline (SD 1.5 native) |
| 768×768 | 5.5 GB | — | — | Good detail |
| 1024×1024 | 9.0 GB | 7.5 GB | 11 GB | Native SDXL/Flux |
| 1280×1280 | 14 GB | 11 GB | 16 GB | Very high detail |
| 1536×1536 | 21 GB | 16 GB | 24 GB | Near-artifact threshold |
| 2048×2048 | 38 GB | 28 GB | 42 GB | Upscaler required |

**Aspect ratio buckets** (SDXL training): Models are trained on multiple aspect ratios (e.g., 1024×1024, 896×1152, 768×1344, 640×1536). Picking a bucket close to your target reduces cropping.

### Serving Architecture Patterns

```python
# Production inference server with Diffusers — minimal example
import torch
from diffusers import StableDiffusionXLPipeline
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")
pipe.enable_attention_slicing()
pipe.enable_xformers_memory_efficient_attention()

class GenRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    steps: int = 30
    guidance_scale: float = 7.5
    seed: int | None = None

@app.post("/generate")
async def generate(req: GenRequest):
    generator = torch.manual_seed(req.seed) if req.seed else None
    image = pipe(
        req.prompt,
        negative_prompt=req.negative_prompt,
        num_inference_steps=req.steps,
        guidance_scale=req.guidance_scale,
        generator=generator,
    ).images[0]
    return {"image": image.tobytes(), "format": "PNG"}  # serialize appropriately
```

### Performance Benchmarks (Production Hardware)

| GPU | Model | Resolution | Steps | Latency (1 img) | Throughput (imgs/s, bs=4) |
|:---:|-------|:----------:|:----:|:---------------:|:-------------------------:|
| **RTX 4090** (24GB) | SDXL | 1024×1024 | 30 | 1.8s | 1.8 img/s |
| **RTX 4090** (24GB) | Flux.1 dev | 1024×1024 | 30 | 3.5s | 0.9 img/s |
| **RTX 6000 Ada** (48GB) | SDXL | 1024×1024 | 30 | 1.5s | 2.4 img/s |
| **A10G** (24GB, AWS) | SDXL | 1024×1024 | 30 | 2.8s | 1.2 img/s |
| **A100** (80GB) | SDXL | 1024×1024 | 30 | 0.9s | 3.8 img/s |
| **A100** (80GB) | Flux.1 dev | 1024×1024 | 30 | 2.2s | 1.6 img/s |
| **H100** (80GB) | SDXL | 1024×1024 | 30 | 0.6s | 5.2 img/s |
| **H100** (80GB) | Flux.1 dev | 1024×1024 | 30 | 1.5s | 2.4 img/s |

### Cost Estimation for Production

| Model | GPU Required | GPU Cost/hr (cloud) | imgs/hr | Cost per 1000 images |
|-------|:-----------:|:-------------------:|:-------:|:--------------------:|
| SD 1.5 | RTX 4090 or A10G | ~$1.00 | 3,600 | ~$0.28 |
| SDXL | RTX 4090 or A10G | ~$1.00 | 2,160 | ~$0.46 |
| SDXL + ControlNet | A10G (24GB) | ~$1.00 | 1,200 | ~$0.83 |
| Flux.1 dev | A100 (80GB) | ~$2.50 | 3,600 | ~$0.69 |
| Flux Pro (API) | — | API rate | N/A | ~$0.05/image |

### Common Deployment Pitfalls

| Issue | Symptom | Root Cause | Fix |
|-------|---------|------------|-----|
| **OOM on first batch** | CUDA OOM at first inference | Memory fragmentation from model loading | `torch.cuda.empty_cache()` after loading; use `lowvram` mode |
| **Gray/black images** | All outputs are gray or black | VAE underflow → latent values in wrong range | Cast VAE to fp32 for decode; check `scale_factor` |
| **NaN loss during training** | Training diverges | FP16 overflow in attention logits | Use `torch.cuda.amp.GradScaler`; reduce LR |
| **Checkerboard artifacts** | Grid pattern in outputs | Upsampling convolution imbalance | Use nearest-exact interpolation; increase decoder channels |
| **Prompt adherence degrades** | Outputs ignore prompt | CFG scale too low or model overfitted | Increase CFG (try 7.5→12); use negative prompt |
| **Slow first inference** | First generation is 10× slower | Model compilation + KV cache warmup | Warmup with 2-3 dummy inferences; save compiled model |

### Deployment Checklist

- [ ] **Model format:** Convert to FP16 and test quality delta; consider INT8/FP8 for throughput
- [ ] **Warmup:** Run 2-3 dummy generations before serving first request (cold-start mitigation)
- [ ] **Batching:** Profile VRAM vs throughput for batch sizes 1, 2, 4, 8
- [ ] **Scheduling:** Implement queue with priority (free users get batch size 1, paid users get higher throughput)
- [ ] **Safety:** Add NSFW filter (CLIP-based or classifier); implement content moderation
- [ ] **Caching:** Cache repeated prompts + seeds in Redis; implement prompt normalization (lowercase, whitespace trim)
- [ ] **Monitoring:** Track generation latency P50/P95/P99, OOM rate, prompt count, resolution distribution
- [ ] **Fallbacks:** Fallback to smaller model on OOM; reduce steps dynamically under load

### Cross-References for Deployment

| Reference | Description |
|-----------|-------------|
| [05-Enterprise/04-AI-Infrastructure.md] | GPU infrastructure, serving |
| [06-Advanced/01-Multimodal-AI.md] | Multimodal generation |
| [08-Reference/01-Glossary.md] | Key terms |


## 15. Cross-References

| Reference | Description |
|-----------|-------------|
| [06-Advanced/01-Multimodal-AI.md] | Vision models, multimodal generation |
| [01-Foundations/01-LLM-and-AI-Models.md] | Transformer architecture |
| [02-LLMs/01-Transformer-Architecture.md] | Attention mechanisms |
| [01-Foundations/06-Reinforcement-Learning.md] | RL for diffusion guidance |
| [08-Reference/01-Glossary.md] | Key terms |

|---

## 16. Ethics, Safety, and Societal Impact

### 16.1 The Dual-Use Challenge

Diffusion models, like all powerful generative AI, present a dual-use problem: the same technology that enables creative expression and productivity can be weaponized for disinformation, harassment, and fraud.

| Risk Category | Specific Threat | Mitigation Approach | Industry Standard |
|:-------------|:----------------|:--------------------|:-----------------:|
| **Deepfakes** | Non-consensual intimate images, political impersonation | C2PA content credentials, deepfake detection classifiers | C2PA 2.1, Coalition for Content Provenance |
| **Disinformation** | Synthetic news images, fake event photography | Watermarking (invisible, robust), metadata signing | SynthID (Google DeepMind), IPTC metadata |
| **Copyright infringement** | Generation in style of living artists, trademarked characters | NSFW filters, artist opt-out registries, model-distillation filters | Spawning API, Have I Been Trained |
| **Bias & representation** | Stereotypical or under-represented groups | Dataset auditing, prompt debiasing, fairness fine-tuning | LAION-5B filtering, SD bias reduction techniques |
| **Harmful content** | Violence, gore, illegal activity generation | Classifier-based safety filters, prompt blocklists | Safety Checker (Diffusers), OpenAI Content Filter |
| **Model theft** | Fine-tuning on proprietary data, unauthorized use | API rate limiting, watermark tracing, license enforcement | Model licensing (CreativeML Open RAIL-M) |

### 16.2 Technical Safety Measures

| Layer | Technique | Implementation | Effectiveness |
|:------|:----------|:---------------|:-------------:|
| **Input** | Prompt filtering (blocklist + classifier) | CLIP-based NSFW classifier on text | Catches ~95% of explicit prompts |
| **Generation** | Latent-space intervention | Redirect diffusion away from unsafe regions during sampling | Moderate — reduces harmful outputs but can be circumvented |
| **Output** | Image classification after generation | NSFW image classifier (e.g., LAION's CLIP-based detector) | ~97% recall on known harmful categories |
| **Provenance** | Invisible watermarking | DWT-DCT-SVD watermark embedded in generated images | Robust to cropping, compression, and recoloring |
| **Attribution** | C2PA metadata signing | Signed manifest attached to image file | Verifiable chain of provenance; can be stripped |

### 16.3 Watermarking and Provenance

```python
# Invisible watermark embedding using DWT-DCT-SVD (simplified)
import numpy as np
import pywt
from scipy.fftpack import dct, idct

def embed_watermark(image: np.ndarray, watermark: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    \"\"\"
    Embed an invisible watermark into an image using DWT-DCT-SVD.
    
    Args:
        image: RGB image array (H, W, 3) in [0, 255]
        watermark: Binary watermark (-1, 1) pattern
        alpha: Watermark strength (higher = more visible but more robust)
    
    Returns:
        Watermarked image
    \"\"\"
    # Convert to YUV and use luminance channel
    import cv2
    yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    Y = yuv[:, :, 0].astype(np.float32)
    
    # 2-level DWT
    coeffs = pywt.wavedec2(Y, 'haar', level=2)
    LL, (LH, HL, HH) = coeffs
    
    # Block DCT on LL subband
    blocks = []
    h, w = LL.shape
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = LL[i:i+8, j:j+8]
            if block.shape == (8, 8):
                blocks.append(dct(block))
    
    # Embed watermark in mid-frequency DCT coefficients
    wm_idx = 0
    for i, block_dct in enumerate(blocks):
        if wm_idx < len(watermark):
            # Modify a mid-frequency coefficient
            block_dct[4, 4] += alpha * watermark[wm_idx]
            wm_idx += 1
    
    # Inverse DCT
    idx = 0
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            if (i//8) * (w//8) + (j//8) < len(blocks):
                LL[i:i+8, j:j+8] = idct(blocks[idx])
                idx += 1
    
    # Inverse DWT
    yuv[:, :, 0] = pywt.waverec2((LL, (LH, HL, HH)), 'haar')[:h, :w].clip(0, 255)
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB).astype(np.uint8)


def detect_watermark(image: np.ndarray, original_watermark: np.ndarray) -> float:
    \"\"\"Extract and correlate watermark from a potentially watermarked image.\"\"\"
    import cv2
    yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    Y = yuv[:, :, 0].astype(np.float32)
    coeffs = pywt.wavedec2(Y, 'haar', level=2)
    LL, _ = coeffs
    
    # Extract from mid-frequency DCT
    extracted = []
    h, w = LL.shape
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = LL[i:i+8, j:j+8]
            if block.shape == (8, 8):
                block_dct = dct(block)
                if len(extracted) < len(original_watermark):
                    extracted.append(block_dct[4, 4])
    
    extracted = np.array(extracted)
    # Normalize and correlate
    extracted = (extracted - extracted.mean()) / extracted.std()
    correlation = np.dot(extracted[:len(original_watermark)], original_watermark) / len(original_watermark)
    return float(correlation / (extracted.std() + 1e-8))
```

### 16.4 Regulatory Landscape

| Region | Regulation | Key Requirements | Impact on Diffusion Models |
|:-------|:-----------|:-----------------|:--------------------------|
| **EU** | EU AI Act (2025) | Risk categorization, transparency, watermarking for deepfakes | Synthetic content labeling mandatory; high-risk systems require conformity assessment |
| **USA** | Executive Order 14110 (2023), state laws | Reporting requirements for foundation models, watermarking standards | Federal procurement requires C2PA; California AI safety bill |
| **China** | Deep Synthesis Provisions (2023) | Real-name registration, data labeling, watermarking | All AI-generated content must be labeled; algorithm filing required |
| **UK** | AI Safety Summit (2023), pro-innovation framework | Voluntary commitments, safety testing | Industry self-regulation, safety institute evaluations |

### 16.5 Best Practices for Responsible Deployment

1. **Always watermark generated content** — even if optional, make it the default
2. **Implement tiered safety filters** — prompt-level → generation-level → output-level
3. **Maintain an abuse reporting system** — users must be able to report harmful outputs
4. **Publish transparency reports** — quarterly metrics on flagged/blocked content volumes
5. **Audit for bias regularly** — use the MIT Bias Benchmark or SD Bias Explorer
6. **Provide content provenance** — C2PA credentials in image metadata with model ID + generation params
7. **Restrict sensitive generation** — celebrity faces, political figures, currency, official documents
8. **Enable artist opt-out** — allow creators to exclude their works from training data

### 16.6 Cross-References for Ethics

| Reference | Description |
|-----------|-------------|
| [07-Emerging/02-AI-Safety.md] | AI safety frameworks, alignment research |
| [07-Emerging/03-AI-Governance.md] | AI governance, regulation, policy |
| [08-Reference/02-AI-Roadmap.md] | Responsible AI roadmap milestones |
| [01-Foundations/09-Federated-Learning-Privacy.md] | Privacy-preserving machine learning |

---

*Document version: 2.0 — June 2026 | Expanded: added §16 Ethics, Safety & Societal Impact — dual-use risks, safety measures table, watermarking code, regulatory landscape, best practices checklist, cross-references. Fixed ToC drift (added §13a, corrected §14a/14b nesting). Previously v1.1.*
