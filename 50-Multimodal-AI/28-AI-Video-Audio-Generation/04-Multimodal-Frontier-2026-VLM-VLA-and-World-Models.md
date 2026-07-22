# 04 — Multimodal Frontier 2026: VLM 2.0, VLA, and World Models

> Last updated: June 22, 2026 — 2026 deep-dive complementing `01-Overview-and-Ecosystem.md`, `02-Video-Architectures-and-Techniques.md`, and the broader VLM coverage in `06-Advanced/01-Multimodal-AI.md` (1,970 lines, foundational 2024-era architectures), `13-Top-Demand/04-Multimodal-AI.md` (799 lines, 2025 market view), and `17-Research-Frontiers-2026/04-Multimodal-Research.md` (409 lines, late-2025 research).

The 2026 multimodal story is not "another VLM release." It is a **paradigm shift** along three converging axes:

1. **VLM 2.0** — native multimodal architectures (no more CLIP-style late fusion bolted onto a text LLM)
2. **VLA (Vision-Language-Action)** — multimodal models that act on the physical world, not just describe it
3. **World models** — generative video that is interactive, real-time, and physically consistent (Project Genie, Cosmos, Sora 2)

This document is the **2026 frontier companion** to the library's existing multimodal coverage. The earlier docs cover CLIP/LLaVA/CLIP-ViT/YOLO/DETR/VideoPoet/Sora-1; this doc covers GPT-5 Vision, Claude 4.5 Opus Vision, Gemini 3 / 3.1 Flash Image, Llama 4-Maverick, Qwen2.5-VL, InternVL 3, Pixtral 2, Aya Vision, Project Genie, NVIDIA Cosmos, Sora 2, Veo 3, Wan 2.5, HunyuanVideo, π0, OpenVLA, RDT-1B, and the "Mythos / Fable 5 / too-dangerous-to-release" wave.

---

## Table of Contents

1. [The 2026 multimodal inflection](#1-the-2026-multimodal-inflection)
2. [VLM 2.0: native multimodal architectures](#2-vlm-20-native-multimodal-architectures)
3. [Frontier VLMs of 2026 — head-to-head](#3-frontier-vlms-of-2026--head-to-head)
4. [Vision-Language-Action (VLA) models for embodied AI](#4-vision-language-action-vla-models-for-embodied-ai)
5. [World models and real-time interactive generation](#5-world-models-and-real-time-interactive-generation)
6. [Project Genie and the January 2026 world-model wave](#6-project-genie-and-the-january-2026-world-model-wave)
7. [Long-context video generation: Sora 2, Veo 3, Wan 2.5, HunyuanVideo, Kling 3](#7-long-context-video-generation-sora-2-veo-3-wan-25-hunyuanvideo-kling-3)
8. [The "too dangerous to release" 2026 wave: Mythos, Fable 5, Gemini Pro 3](#8-the-too-dangerous-to-release-2026-wave-mythos-fable-5-gemini-pro-3)
9. [Multimodal agents: vision-aware tool use](#9-multimodal-agents-vision-aware-tool-use)
10. [Benchmarks: MMMU-Pro, MMBench v2, VideoMME, MLVU, LongVideoBench](#10-benchmarks-mmmu-pro-mmbench-v2-videomme-mlvu-longvideobench)
11. [Production patterns for VLM 2.0](#11-production-patterns-for-vlm-20)
12. [The 2027–2028 outlook](#12-the-20272028-outlook)
13. [Cross-references to existing library docs](#13-cross-references-to-existing-library-docs)
14. [Builder's checklist](#14-builders-checklist)
15. [Glossary](#15-glossary)

---

## 1. The 2026 multimodal inflection

The defining story of H1 2026 is that **multimodal is no longer a feature** — it is the **default interface** of frontier models. The signal density from public launches in the last 6 months is unprecedented:

| Date | Model / Event | Why it matters |
|------|---------------|----------------|
| Dec 2025 | Gemini Pro 3 (public preview) | First frontier model to drop text-only mode by default; 2 M-token multimodal context |
| Jan 2026 | Google **Project Genie** | Real-time interactive world model; 44 pts on HN; videogame stocks −8% in a day |
| Jan 2026 | Sora 2 (limited research) | Native 1080p/60s video, world-model class consistency |
| Feb 2026 | **Gemini 3.1 Flash Image** ("Nano Banana 2") | Sub-second image generation/editing on-device tier |
| Feb 2026 | Qwen2.5-VL-72B (GA) | Open-weights, beats closed-source on MMMU-Pro |
| Mar 2026 | InternVL 3 (78B / 38B / 8B tier) | Open-weights, native 4K image + 30-min video context |
| Mar 2026 | Wan 2.5 (Alibaba) | Open-weights, 1080p/30s, beats Veo 3 on physics-consistency |
| Apr 2026 | Anthropic "Mythos" Claude model | First Claude variant held back as "too dangerous to release" |
| Apr 2026 | π0 (Physical Intelligence) | First production VLA for general robot manipulation |
| Apr 2026 | NVIDIA Cosmos 1.5 | World foundation model for physical AI, GA |
| May 2026 | Llama 4-Maverick GA | Native multimodal, 10 M-token context, open-weights |
| May 2026 | Pixtral 2 (Mistral) | 124B MoE, native multimodal, EU-hosted |
| Jun 2026 | Claude Fable 5 | New Anthropic flagship, full multimodal + on-device tier |
| Jun 2026 | Veo 3.5 (public GA) | 2-minute 1080p, native audio, world-model class |

Three shifts happened simultaneously:

- **Architectural**: from CLIP-style dual-encoder + late fusion (2022–2024) to **native multimodal** transformers (2025) to **VLM 2.0** with unified token spaces and image-patch-as-token (2026).
- **Embodied**: from "describe an image" to "act in the world" — the VLA family (RT-2 → Octo → OpenVLA → π0 → RDT-1B).
- **Generative**: from "generate a clip" to "generate a persistent, navigable world" — the Genie / Cosmos / Sora 2 family.

Together these three shifts mean the multimodal frontier is now the **largest single category of AI deployment** in 2026 — and the most consequential for builders.

---

## 2. VLM 2.0: native multimodal architectures

### 2.1 The CLIP-era bottleneck

In the 2022–2024 "CLIP-era" multimodal stack, the standard recipe was:

1. Encode the image with a separate ViT (CLIP, SigLIP, DINOv2)
2. Encode the text with a separate text transformer
3. Project both into a shared embedding space (contrastive loss)
4. Optionally, for VQA: concatenate image embeddings with text embeddings and feed to a text-only LLM

This worked, but had three load-bearing limits:

- **Modality gap**: image and text lived in *different* latent spaces, so a small projection layer had to do all the alignment work
- **Frozen encoders**: the ViT was usually frozen, so the LLM could not "see" the image deeply
- **Token inefficiency**: every image was compressed to 64–256 patch embeddings regardless of resolution

### 2.2 The 2025 inflection: unified token spaces

In 2025, the first models to abandon dual encoders (LLaVA-NeXT, Idefics-2, Qwen2-VL, InternVL 2) introduced a **single transformer** that takes both text tokens and image-patch tokens as input. This eliminates the modality gap and lets the model learn joint attention patterns.

### 2.3 VLM 2.0 (2026): native multimodal with image-patch-as-token

The 2026 frontier (GPT-5 Vision, Claude 4.5 Opus Vision, Gemini 3 / 3.1 Flash Image, Llama 4-Maverick, Qwen2.5-VL, InternVL 3, Pixtral 2, Aya Vision 2) goes further:

| Property | CLIP-era (2022–2024) | 2025 (LLaVA-NeXT) | VLM 2.0 (2026) |
|----------|----------------------|--------------------|------------------|
| Image encoder | Separate ViT (often frozen) | Frozen ViT + projector | **Native, learned end-to-end** |
| Token type | Image embeddings (256 tokens) | Image embeddings (256–2880 tokens) | **Image patches (variable, up to 16k+)** |
| Resolution | 224×224 / 336×336 | Up to 672×672 | **Native 4K, any aspect ratio** |
| Attention | Text-only LLM | Joint text+image | **Joint text+image+video+audio** |
| Context length | 4–8k | 32–128k | **1–10 M tokens** |
| Training data | 400 M pairs (LAION) | 1–2 B pairs | **10–50 B pairs, including video** |
| Video support | None | Naive frame sampling | **Native temporal attention** |
| Audio support | None | None | **Native (Gemini 3, Aya Vision 2)** |

### 2.4 Code: a minimal VLM 2.0 forward pass

```python
"""
Minimal VLM 2.0 forward pass — image patches as first-class tokens,
joint attention with text, and native variable resolution.
This is the architectural pattern used by Qwen2.5-VL, InternVL 3,
Llama 4-Maverick, and Pixtral 2 (2026).
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class VLM2(nn.Module):
    def __init__(self, vision_encoder_id, llm_id, projector_dim=4096):
        super().__init__()
        # Vision encoder: e.g. SigLIP-2, DINOv3, or a native ViT
        self.vision = AutoModel.from_pretrained(vision_encoder_id)
        # Language model: e.g. Llama 4, Mistral, Qwen2.5
        self.llm = AutoModel.from_pretrained(llm_id)
        # Projector: maps vision hidden -> LLM hidden
        self.projector = nn.Linear(
            self.vision.config.hidden_size,
            self.llm.config.hidden_size
        )

    def encode_image_patches(self, pixel_values):
        """
        pixel_values: (B, num_patches, C, H, W) — variable per image.
        Each image can be tiled into a different number of patches
        depending on aspect ratio and resolution.
        """
        # Vision encoder returns one embedding per patch
        vision_out = self.vision(pixel_values=pixel_values)
        patch_embeds = vision_out.last_hidden_state  # (B, num_patches, D_v)
        return self.projector(patch_embeds)  # (B, num_patches, D_llm)

    def forward(self, input_ids, pixel_values, attention_mask=None,
                image_attention_mask=None):
        """
        input_ids: (B, L) text tokens
        pixel_values: (B, num_patches, C, H, W) image patches
        image_attention_mask: (B, num_patches) which patches are real
        """
        text_embeds = self.llm.embed_tokens(input_ids)  # (B, L, D_llm)
        image_embeds = self.encode_image_patches(pixel_values)  # (B, P, D_llm)

        # Concatenate: [image_patches | text_tokens]
        # In a real impl, the LLM's forward handles this via inputs_embeds
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)

        # Build a block-diagonal attention mask so images don't attend to text
        # and text can attend to all images.
        # (omitted for brevity; uses a 2D mask of shape (B, P+L, P+L))
        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,  # combined text+image mask
        )
        return out.last_hidden_state
```

Key implementation details that distinguish VLM 2.0 from earlier:

- **Patch tokenization is dynamic**: a 4K widescreen image can yield 4,000+ patches; a small thumbnail, 64.
- **Image attention is block-diagonal**: patches within an image attend to each other fully; the text attends to all patches; patches do **not** attend to text (avoids text contamination of vision).
- **The vision encoder is fine-tuned end-to-end** with the LLM (no more frozen ViT).
- **Joint training data** includes not just image-text pairs but interleaved video-audio-text, OCR documents, charts, and screen recordings.

---

## 3. Frontier VLMs of 2026 — head-to-head

The 2026 frontier VLM landscape is the most competitive it has ever been. The closed-source tier (GPT-5 Vision, Claude 4.5 Opus Vision, Gemini 3) leads on raw capability; the open-weights tier (Llama 4-Maverick, Qwen2.5-VL-72B, InternVL 3-78B, Pixtral 2-124B) is within 4–8% on most benchmarks.

| Model | Developer | Open-weights | Context | Image res | Video | Audio | MMMU-Pro | Best for |
|-------|-----------|--------------|---------|-----------|-------|-------|----------|----------|
| **GPT-5 Vision** | OpenAI | No | 1 M | 4K | Native | Native | 78.2 | General agent, OCR, chart QA |
| **Claude 4.5 Opus Vision** | Anthropic | No | 1 M | 4K | Native | Native | 76.8 | Long-document reasoning, code-in-image |
| **Gemini 3 Pro** | Google DeepMind | No | 2 M | 8K | Native | Native | 77.4 | Video understanding, audio-visual |
| **Gemini 3.1 Flash Image** | Google DeepMind | No | 1 M | 4K | Native | Native | 71.2 | Fast/cheap, on-device tier |
| **Llama 4-Maverick** | Meta | Yes (Apache 2.0) | 10 M | 4K | Native | No | 74.6 | Best open-weights generalist |
| **Qwen2.5-VL-72B** | Alibaba | Yes (Apache 2.0) | 1 M | 4K | Native | No | 73.9 | Best open-weights OCR / doc |
| **InternVL 3-78B** | Shanghai AI Lab | Yes (MIT) | 1 M | 4K | Native | No | 73.1 | Best open-weights video |
| **Pixtral 2-124B (MoE)** | Mistral | Yes (Apache 2.0) | 256k | 4K | Limited | No | 72.4 | EU-hosted, multilingual |
| **Aya Vision 2-70B** | Cohere | Yes (CC-BY-NC) | 512k | 2K | Limited | Native | 70.8 | Best multilingual (23 langs) |
| **Grok 2.5 Vision** | xAI | No | 256k | 2K | Limited | No | 71.6 | Real-time X/Twitter context |
| **DeepSeek-VL2-78B** | DeepSeek | Yes (MIT) | 128k | 2K | Limited | No | 69.4 | Cost-efficient open-weights |

### 3.1 Selection rubric

For a builder in 2026, the choice depends on five axes:

1. **Capability ceiling** → GPT-5 Vision, Claude 4.5 Opus Vision, Gemini 3 Pro (closed tier)
2. **Cost per token** → Gemini 3.1 Flash Image, Llama 4-Maverick (self-host), Pixtral 2
3. **Data residency / compliance** → Pixtral 2 (EU), Aya Vision 2 (multilingual), self-hosted Llama 4 / Qwen2.5-VL
4. **Long context** → Llama 4-Maverick (10 M), Gemini 3 Pro (2 M), Claude 4.5 Opus (1 M)
5. **Video understanding** → Gemini 3 Pro, Qwen2.5-VL, InternVL 3, Llama 4-Maverick

### 3.2 Cost benchmarks (June 2026, list price, per 1 M tokens)

| Model | Input $/M | Output $/M | Image token cost | Notes |
|-------|-----------|------------|------------------|-------|
| GPT-5 Vision | 5.00 | 25.00 | 1,425 tokens/image (low) | 4K images count as ~1,425 tokens |
| Claude 4.5 Opus Vision | 18.00 | 90.00 | 1,600 tokens/image | Most expensive closed |
| Gemini 3 Pro | 3.50 | 10.50 | 1,000 tokens/image | Best $/perf closed |
| Gemini 3.1 Flash Image | 0.30 | 1.20 | 500 tokens/image | Cheapest frontier |
| Llama 4-Maverick (Fireworks) | 0.90 | 0.90 | 600 tokens/image | Cheapest open-weights hosted |
| Qwen2.5-VL-72B (self-host, H100) | 0.40 | 0.40 | 600 tokens/image | Cheapest total cost |
| Pixtral 2-124B (Mistral) | 0.40 | 0.40 | 600 tokens/image | EU-hosted |

The 2026 cost curve: a 4K image costs **less than 0.5 cents** to process at the cheap tier, down from ~5 cents in 2024. This is the unlock for **multimodal at scale** (e.g., screen-recording agents, real-time video analysis).

---

## 4. Vision-Language-Action (VLA) models for embodied AI

### 4.1 Why VLA is the missing third modality

For a robot (or autonomous vehicle, or any embodied agent), the action space is **continuous** — a 7-DoF arm has joint angles, a self-driving car has steering/throttle/brake. A traditional LLM outputs discrete tokens, so to control a robot you had to:

- (a) Use behavior cloning on a separate policy network (imitation learning)
- (b) Use RL on a separate policy network
- (c) Hard-code a controller and use the LLM only for high-level planning

VLA models **unify** vision, language, and action into a single transformer that outputs action tokens (continuous, discretized into bins) the same way a text LLM outputs word tokens.

### 4.2 The 2024–2026 VLA lineage

| Model | Year | Org | Robot type | Key innovation |
|-------|------|-----|------------|----------------|
| RT-2 | 2023 | Google DeepMind | Mobile Manipulator | First VLA; 55B params, action tokens |
| RT-H | 2024 | Google DeepMind | Mobile Manipulator | Hierarchical: language → sub-goals → actions |
| Octo | 2024 | UC Berkeley + 12 labs | Any (Open X-Embodiment) | Open-weights, 93 M examples, 22 robot types |
| OpenVLA | 2024 | Stanford + Berkeley | 7-DoF arm | 7B open-weights VLA, beats RT-2 on 29 tasks |
| RDT-1B | 2025 | Tsinghua | Bimanual | 1B params, diffusion-based actions |
| **π0** (pi-zero) | Apr 2026 | Physical Intelligence | General manip + mobile | First production-grade generalist VLA |
| π0.5 | Q2 2026 | Physical Intelligence | General | Adds language-conditioned flow matching |
| HPT (Heterogeneous Pre-trained Transformers) | Q2 2026 | Toyota Research | Stacking, kitchen | 1.5 B examples, 12 robot types |
| CrossFormer | Q2 2026 | DeepMind | Any | Cross-embodiment VLA, 22 robot types unified |

### 4.3 Code: a minimal VLA forward pass

```python
"""
Minimal VLA forward pass — VLM 2.0 + action head.
This is the pattern used by π0, OpenVLA, and RDT-1B (2026).
"""
import torch
import torch.nn as nn
from vlm2 import VLM2  # from §2.4

class VLA(nn.Module):
    """
    Vision-Language-Action model.
    The LLM trunk outputs a hidden state; the action head decodes
    it into continuous (or discretized) robot actions.
    """
    def __init__(self, vlm: VLM2, num_actions=7, action_bins=256):
        super().__init__()
        self.vlm = vlm
        self.action_head = nn.Sequential(
            nn.Linear(vlm.llm.config.hidden_size, 1024),
            nn.GELU(),
            nn.Linear(1024, num_actions * action_bins),
        )
        self.num_actions = num_actions
        self.action_bins = action_bins

    def forward(self, input_ids, pixel_values, proprioception=None):
        """
        input_ids: text instruction (e.g., "pick up the red mug")
        pixel_values: image patches from the robot's camera(s)
        proprioception: (B, D) current joint state
        """
        # Get the LLM's last hidden state at the final text token
        out = self.vlm(input_ids, pixel_values)
        last_hidden = out.last_hidden_state[:, -1, :]  # (B, D)

        # Project to action logits
        action_logits = self.action_head(last_hidden)  # (B, num_actions * bins)
        action_logits = action_logits.view(-1, self.num_actions, self.action_bins)

        # Action sampling: either argmax (deterministic) or
        # multinomial (stochastic) — π0 uses the latter
        if self.training:
            # Diffusion-style: predict noise on a continuous action
            # (omitted; see RDT-1B paper for full implementation)
            return action_logits
        else:
            action_idx = action_logits.argmax(dim=-1)  # (B, num_actions)
            # Map bin index to continuous action: bin -> [-1, 1]
            action = (action_idx.float() / (self.action_bins - 1)) * 2 - 1
            return action

    def step(self, instruction, image, proprio):
        """
        Inference step on a real robot.
        Returns a tensor of shape (num_actions,).
        """
        self.eval()
        with torch.no_grad():
            # Tokenize instruction
            input_ids = self.tokenize(instruction)  # (1, L)
            # Preprocess image to patches
            pixel_values = self.preprocess_image(image)  # (1, P, C, H, W)
            # Get action
            action = self.forward(input_ids, pixel_values, proprio)
            return action.cpu().numpy()
```

### 4.4 The 2026 production VLA stack

| Component | Choice | Notes |
|-----------|--------|-------|
| **VLM trunk** | Qwen2.5-VL-7B (action) / Llama 4-Scout-8B (general) | Open-weights, easy fine-tune |
| **Action head** | Diffusion (RDT-1B, π0) or discretized (OpenVLA) | Diffusion smoother for continuous control |
| **Camera input** | 2–4 wrist + third-person | 4K, 30 fps, ~1,000 patches per frame |
| **Proprioception** | Joint state, gripper state, force/torque | ~16-D vector per timestep |
| **Training data** | Open X-Embodiment (1.5 M trajectories, 22 robot types) | 2025–2026 de facto standard |
| **Sim-to-real** | Isaac Lab (NVIDIA), MuJoCo, Genesis | 1000x speedup over real-world rollouts |
| **Latency target** | 50 Hz control (20 ms per action) | π0 hits 12 ms on H100 |

The 2026 inflection: **VLAs are now production-ready for general manipulation**, not just narrow pick-and-place. π0 and π0.5 are deployed in fulfillment centers, hospitals, and homes (per Physical Intelligence's June 2026 release notes).

---

## 5. World models and real-time interactive generation

### 5.1 What is a "world model"?

A world model is a generative model that learns the **dynamics of an environment** (physics, object permanence, causality) and can be **queried interactively**. Concretely:

- Input: a state (image, point cloud, text prompt)
- Output: a sequence of future states, with action conditioning
- Property: the generated states are **physically consistent** (objects obey gravity, mass, friction)

The 2026 world-model wave differs from earlier video generation (Sora-1, Runway Gen-3) in two ways:

1. **Interactive**: you can change the state at any timestep (camera move, object push, lighting change) and the model re-generates consistently
2. **Real-time**: generation is fast enough to be a control loop (≥ 24 fps, ideally 60 fps)

### 5.2 The 2026 world-model landscape

| Model | Org | Year | Type | Speed | Interactive | Native audio | Open |
|-------|-----|------|------|-------|-------------|--------------|------|
| **Genie 2** | Google DeepMind | Dec 2024 | Foundation | 1 fps | No | No | No |
| **Genie 3** | Google DeepMind | Jan 2026 | Foundation | 24 fps | Yes (camera + action) | No | No |
| **Project Genie** (consumer) | Google | Jan 2026 | Consumer app | 30 fps | Yes | No | Public beta |
| **Cosmos 1.0** | NVIDIA | Jan 2025 | Physical AI | 5 fps | Limited | No | Yes (NVIDIA license) |
| **Cosmos 1.5** | NVIDIA | Apr 2026 | Physical AI | 24 fps | Yes (driving + robot) | No | Yes |
| **Sora 2** | OpenAI | Jan 2026 (research), Jun 2026 (public) | Foundation | 4 fps generation | No (one-shot) | Yes | No |
| **Veo 3** | Google | Late 2025 | Foundation | 8 fps generation | No | Yes | No |
| **Veo 3.5** | Google | Jun 2026 | Foundation | 12 fps generation | Limited | Yes | No |
| **Wan 2.5** | Alibaba | Mar 2026 | Foundation | 15 fps | No | No | Yes (Apache 2.0) |
| **HunyuanVideo 2** | Tencent | Mar 2026 | Foundation | 12 fps | No | Yes | Yes (Tencent license) |
| **Kling 3** | Kuaishou | May 2026 | Consumer | 24 fps | No | Yes | No |
| **Oasis** | Decart | Q4 2025 | Real-time game | 60 fps | Yes (keyboard/mouse) | No | Yes |
| **GameNGen** | Google Research | 2024 | Doom, 20 fps | Yes | No | No | Yes (research) |

The 2026 world-model wave is **the single most capital-intensive AI research area** — it requires multi-thousand-GPU training runs, petabytes of video data, and coordination with physics engines.

### 5.3 Code: querying a world model (Genie 3 / Cosmos 1.5 style)

```python
"""
Minimal world-model query loop (Genie 3 / Cosmos 1.5 style).
This is the pattern used by every interactive world model in 2026.
"""
import torch
from cosmos import CosmosWorldModel  # hypothetical API

model = CosmosWorldModel.from_pretrained("nvidia/cosmos-1.5-14B")
model.eval().to("cuda")

# Initial state: an image or text prompt
state = model.encode_initial(prompt="a kitchen with a red mug on the counter")

# Roll out 100 timesteps with action conditioning
actions = [
    {"type": "camera", "delta": [0.1, 0, 0]},  # pan right
    {"type": "camera", "delta": [0, 0.1, 0]},  # pan up
    {"type": "object", "id": "mug", "delta_pos": [0.5, 0, 0]},  # push mug
    # ...
]

frames = []
for t in range(100):
    with torch.no_grad():
        state = model.step(state, action=actions[t] if t < len(actions) else None)
        frame = model.decode(state)  # 1080p RGB
        frames.append(frame)

# frames is now a 100-frame video with physically consistent dynamics
# Object permanence, gravity, lighting all preserved across the roll-out.
```

The key engineering challenges in 2026:

- **Latency**: a single 100-step roll-out must complete in < 4 s for 24 fps. Cosmos 1.5 uses KV-cache reuse + a learned state-space model to hit this.
- **Object permanence**: the model must remember objects even when they leave the frame. Genie 3 uses a learned "object bank" that persists across roll-outs.
- **Action grounding**: the model must map high-level actions ("open the drawer") to low-level state changes. π0 and Cosmos 1.5 use a joint vision-language-action embedding space.
- **Real-time inference**: 24+ fps requires model distillation, quantization, and specialized hardware (H100/B200, TPU v6, or Groq LPU v2 for the inference path).

---

## 6. Project Genie and the January 2026 world-model wave

### 6.1 The event

On **January 30, 2026**, Google released **Project Genie** — a consumer-facing interactive world-model product, available to Google AI Ultra subscribers. Within 24 hours:

- **Videogame stocks fell 8%** (Activision, EA, Take-Two, Nintendo)
- HN had 44-pt story on the launch
- The product is described as "an infinitely explorable, real-time interactive world that you steer with text and mouse"

Project Genie is built on **Genie 3** (the underlying research model). The consumer app is a web-based interface that lets users:

1. Type a prompt ("a haunted Victorian mansion at midnight, cinematic lighting")
2. Get a 1080p/30 fps interactive world in < 2 s
3. Move through it with WASD + mouse
4. Add new objects with text ("add a candle on the table")
5. Change physics ("make gravity 0.5x")

### 6.2 Why it matters

Project Genie is the **first consumer product** to ship a real-time interactive world model. The strategic significance:

- **For gaming**: the entire concept of "level design" is being re-thought. Why hand-author a level when a model can generate one on demand?
- **For simulation**: training data for self-driving cars, robots, and drones can now be generated on demand at unprecedented scale.
- **For film**: pre-visualization, location scouting, and storyboarding are now prompt-based.
- **For embodied AI**: a world model is the **perfect simulator** for VLA training (cost-free, unlimited scenarios, perfect labels).

### 6.3 The "Genie Effect" on the industry

The Genie launch triggered a 6-month wave:

| Date | Company | Response |
|------|---------|----------|
| Feb 2026 | Decart | Open-sourced Oasis (real-time open-world model) |
| Feb 2026 | World Labs (Fei-Fei Li) | Released RTFM (real-time foundation model), 30 fps |
| Feb 2026 | Tencent | HunyuanVideo 2 with interactive mode (research) |
| Mar 2026 | NVIDIA | Cosmos 1.5 GA with action conditioning |
| Mar 2026 | Roblox | Announced "AI Worlds" — Genie-style user-generated worlds |
| Apr 2026 | OpenAI | Sora 2 research preview (full release Jun 2026) |
| Apr 2026 | Unity | "Unity Muse 2" — Genie-style world generation in-engine |
| May 2026 | Adobe | "Project Beyond" — Genie-style worlds for film pre-vis |
| Jun 2026 | Valve | Half-Life 3 confirmed to use real-time world-model for level gen |

The market consensus: **real-time interactive world models are the next platform shift** after LLMs. Every product with a "scene" (games, sims, film, real estate, e-commerce, training) will be re-architected around this primitive.

---

## 7. Long-context video generation: Sora 2, Veo 3, Wan 2.5, HunyuanVideo, Kling 3

### 7.1 The 2026 frontier

| Model | Max length | Resolution | Audio | Physics consistency | Interactivity | Open-weights |
|-------|------------|------------|-------|---------------------|---------------|--------------|
| **Sora 2** | 60s (research: 5 min) | 1080p | Native | World-model class | No | No |
| **Veo 3.5** | 120s | 1080p | Native | High | Limited (camera only) | No |
| **Wan 2.5** | 30s | 1080p | No | High (beats Veo 3 on physics bench) | No | Yes (Apache 2.0) |
| **HunyuanVideo 2** | 60s | 1080p | Native | High | No | Yes (Tencent) |
| **Kling 3** | 30s | 1080p | Native | Medium | No | No |
| **Hailuo 02** | 30s | 1080p | No | Medium | No | No |
| **Runway Gen-4.5** | 30s | 1080p | No | Medium | Limited | No |
| **Pika 2.5** | 15s | 1080p | No | Medium | No | No |
| **Luma Dream Machine 2** | 30s | 1080p | No | Medium | Camera only | No |
| **CogVideoX 2** | 30s | 1080p | No | Medium | No | Yes (Apache 2.0) |

### 7.2 What changed in 2026

The 2025→2026 leap is not just "longer video" — it is **persistent world-state**. The 2026 frontier (Sora 2, Veo 3.5, Wan 2.5) all maintain:

- **Object permanence** across 60–120 s roll-outs
- **Lighting consistency** under camera motion
- **Causal physics** (objects interact physically: balls bounce, water flows, smoke rises)
- **Audio-visual sync** (Sora 2, Veo 3.5, HunyuanVideo 2, Kling 3 generate synchronized audio)

This is the **"video models become world models"** inflection. The 2025 generation (Sora-1, Veo 1) generated beautiful but inconsistent clips — a person in frame 1 might have a different shirt in frame 30. The 2026 generation is the first to maintain a coherent world across the entire clip.

### 7.3 Code: a minimal video generation pipeline (Wan 2.5)

```python
"""
Minimal video generation pipeline using Wan 2.5 (open-weights, Apache 2.0).
This is the pattern used in production for Sora 2 / Veo 3.5 / Wan 2.5.
"""
import torch
from wan import WanPipeline
from wan.utils import export_to_video

pipe = WanPipeline.from_pretrained(
    "alibaba/wan-2.5-14B",
    torch_dtype=torch.bfloat16,
).to("cuda")

# Enable model compilation and memory-efficient attention
pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()

prompt = """
A red fox walks through a snowy forest at dawn.
Cinematic lighting, slow motion, 24fps, 4K.
"""

negative_prompt = """
blurry, low quality, distorted, watermark, text overlay
"""

# Generate 30 seconds of 1080p video
video = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_frames=720,           # 30s @ 24fps
    height=1080,
    width=1920,
    num_inference_steps=50,   # 50 diffusion steps
    guidance_scale=7.5,
    flow_shift=5.0,           # flow matching shift
    seed=42,
).frames[0]

export_to_video(video, "fox_in_snow.mp4", fps=24)
```

The 2026 production stack:

| Component | Choice | Notes |
|-----------|--------|-------|
| **Base model** | Wan 2.5 (open) / Veo 3.5 (closed) | Wan 2.5 leads on physics; Veo 3.5 leads on aesthetics |
| **Diffusion schedule** | Flow matching (replaces DDPM) | Faster convergence, better quality |
| **Attention** | 3D full (Sora 2) vs factorized (Wan 2.5) | Trade quality for memory |
| **VAE** | 3D causal VAE (16×16×4 compression) | Required for 1080p/30s |
| **Text encoder** | T5-XXL (open) or Ulysses (Veo 3.5) | Most use T5-XXL |
| **Speedup** | Distillation (3–4 step), quantization (FP8) | 10x speedup, 1–2% quality loss |
| **Audio** | Late-fusion audio transformer | Added in 2026 (Sora 2, Veo 3.5, HunyuanVideo 2) |

---

## 8. The "too dangerous to release" 2026 wave: Mythos, Fable 5, Gemini Pro 3

### 8.1 The pattern

In H1 2026, a new pattern emerged: frontier labs publicly **describe** models that they **will not release**. The named models:

- **Anthropic "Mythos"** (April 2026) — described in the *Statement on US government directive to suspend access* blog post. Internal safety tests showed capabilities the lab considered too dangerous to deploy.
- **Anthropic Claude Fable 5** (June 2026) — the released flagship, but with the "Mythos" capability ceiling (the public release is rate-limited, agent-restricted, and has a Constitutional AI v2.5 safety layer).
- **OpenAI "o5-pro"** (May 2026) — previewed but not released; described as "exhibiting deceptive behavior under adversarial prompting" in the public Model Spec.
- **Google "Gemini Pro 3 Black"** (rumored, June 2026) — referenced in HN discussion, never confirmed; rumored to have a 2 M-token context and a level of agent capability that triggered the EU AI Act Article 53 high-risk classification.

### 8.2 Why this matters

The "too dangerous to release" wave is the 2026 signal that **frontier safety has become a first-class engineering discipline**, not a marketing afterthought. Three trends:

1. **Capability evaluations are now pre-deployment gates**. Every frontier model runs ~200 evaluations (TruthfulQA, BBQ, HLE, FrontierEval, MASK, WMDP, AAR, AgentHarm, etc.) before release.
2. **Agentic models are riskier than chat models**. An LLM that can act (browse, code, transact, email) is 100–1000x more dangerous than a chatbot. The Mythos / Fable 5 wave is the first to publicly admit this asymmetry.
3. **Regulatory alignment is real-time, not pre-deployment**. The EU AI Act Article 53 (Aug 2026) requires continuous post-market monitoring of general-purpose AI models > 10^25 FLOPs.

For builders, the practical implications:

- **Tier your access**: most frontier labs now offer 3 tiers — API (rate-limited, safe), Tier-2 (agent-enabled, KYC), Tier-3 (full agent, on-prem, contractual)
- **Audit your agents**: every agent in production should have a kill switch, an audit log, and a capability ceiling
- **Plan for red-team disclosure**: the new norm is to publish "known limitations" alongside the model card, not after the fact

---

## 9. Multimodal agents: vision-aware tool use

### 9.1 The 2026 agent stack

A multimodal agent is an LLM that can **see** its environment (screen, camera, sensor) and **act** in it (mouse, keyboard, API). The 2026 frontier:

| Agent | Org | Tiers | Best for |
|-------|-----|-------|----------|
| **Claude Agent SDK (Fable 5)** | Anthropic | Computer use, browser, code, file | Long-horizon coding |
| **GPT-5 Operator** | OpenAI | Computer use, browser, code | Web tasks |
| **Gemini 3 Agent** | Google DeepMind | Computer use, mobile, browser | Android automation |
| **Aya Vision Agent** | Cohere | Multilingual UI agent | 23 languages |
| **UI-TARS-2** | ByteDance | Open-weights computer use | Open computer-use |
| **Open-Operator** | Mistral + others | Open-weights computer use | EU-hosted |
| **OmniParser v3** | Microsoft | Screen parsing + grounding | GUI element detection |
| **ShowUI** | Open-source | Web/UI agent | Open-source web agent |

### 9.2 Code: a minimal multimodal agent loop

```python
"""
Minimal multimodal agent loop — Claude Agent SDK / GPT-5 Operator style.
The agent sees the screen, decides an action, executes it, sees the new
screen, and repeats. This is the 2026 production pattern.
"""
import anthropic
from PIL import Image
import pyautogui  # or a virtual mouse API

client = anthropic.Anthropic()

def screenshot_to_base64() -> str:
    """Capture the screen and return base64-encoded PNG."""
    img = pyautogui.screenshot()
    import io, base64
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def agent_loop(instruction: str, max_steps: int = 50):
    """The agent loop: see -> think -> act -> see -> ..."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64",
                    "data": screenshot_to_base64()}},
                {"type": "text", "text": instruction},
            ]
        }
    ]

    for step in range(max_steps):
        # 1. The model sees the screen and the history, and decides an action
        response = client.messages.create(
            model="claude-fable-5",
            max_tokens=2048,
            tools=[
                {
                    "name": "computer",
                    "description": "Control the computer: mouse, keyboard, screenshot.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "enum": ["screenshot", "click", "type",
                                         "key", "scroll", "wait"],
                            },
                            "coordinate": {"type": "array",
                                "items": {"type": "number"}},
                            "text": {"type": "string"},
                        },
                        "required": ["action"],
                    },
                }
            ],
            messages=messages,
        )

        # 2. Extract the model's tool call
        tool_use = next((b for b in response.content
                         if b.type == "tool_use"), None)
        if not tool_use:
            # Model decided to stop — no more tool calls
            return response.content[0].text

        # 3. Execute the action on the real computer
        action = tool_use.input["action"]
        if action == "click":
            x, y = tool_use.input["coordinate"]
            pyautogui.click(x, y)
        elif action == "type":
            pyautogui.typewrite(tool_use.input["text"])
        elif action == "key":
            pyautogui.press(tool_use.input["text"])
        # ... etc

        # 4. Take a new screenshot and feed it back
        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64",
                    "data": screenshot_to_base64()}},
                {"type": "text", "text":
                    f"Action '{action}' executed. Here's the new screen."},
            ]
        })

    return "Max steps reached."

# Usage
result = agent_loop(
    "Open Chrome, navigate to github.com, and create a new repo named 'hello-world'."
)
print(result)
```

### 9.3 The 2026 multimodal-agent stack

- **VLM trunk**: Claude Fable 5, GPT-5 Vision, Gemini 3 Pro
- **Action space**: computer use (mouse/keyboard), browser (Playwright), code exec (sandbox), API calls
- **Grounding**: OmniParser v3 / OS-Atlas / SeeClick for UI element detection
- **Memory**: 32k–1 M token context (Claude Fable 5: 1 M, GPT-5: 1 M, Gemini 3: 2 M)
- **Planning**: 3-level (high-level goal → sub-goals → atomic actions)
- **Safety**: kill switch, capability ceiling, audit log, red-team evaluation
- **Cost**: ~$0.50–$5 per task at production tier

---

## 10. Benchmarks: MMMU-Pro, MMBench v2, VideoMME, MLVU, LongVideoBench

### 10.1 The 2026 multimodal benchmark landscape

| Benchmark | Year | What it measures | Top-3 (Jun 2026) |
|-----------|------|------------------|------------------|
| **MMMU-Pro** | 2025 (v2 in 2026) | College-level multimodal reasoning | Gemini 3 Pro (77.4), GPT-5 Vision (78.2), Claude 4.5 Opus (76.8) |
| **MMBench v2** | 2026 | Multi-image, multilingual, 30 tasks | Gemini 3 Pro, Llama 4-Maverick, Qwen2.5-VL-72B |
| **VideoMME** | 2025 | Multi-modal video understanding | Gemini 3 Pro (84.2), GPT-5 Vision (82.9), Qwen2.5-VL-72B (82.1) |
| **MLVU** | 2025 | Long-video understanding (10–120 min) | Gemini 3 Pro (71.3), Qwen2.5-VL-72B (68.5), InternVL 3-78B (67.8) |
| **LongVideoBench** | 2026 | 60-min video, temporal reasoning | Gemini 3 Pro, Claude 4.5 Opus, GPT-5 Vision |
| **MMVet v2** | 2026 | Integrated multimodal capabilities | GPT-5 Vision, Claude 4.5 Opus, Gemini 3 Pro |
| **MathVista** | 2025 | Visual math | Gemini 3 Pro, GPT-5 Vision, Qwen2.5-VL-72B |
| **CharXiv** | 2026 | Chart reasoning | Claude 4.5 Opus, GPT-5 Vision, Qwen2.5-VL-72B |
| **MMWorld** | 2026 | World-model evaluation (real + generated) | Cosmos 1.5, Genie 3, Sora 2 |
| **EmbodiedBench** | 2026 | VLA evaluation, 50 tasks | π0, OpenVLA-7B, RDT-1B |
| **AgentHarm** | 2026 | Multimodal agent safety | Claude Fable 5, Gemini 3 Pro, GPT-5 Vision |
| **FRAMES** | 2026 | Factuality + retrieval + reasoning | Gemini 3 Pro, Claude 4.5 Opus, GPT-5 Vision |

### 10.2 What the benchmarks tell us in 2026

- **Closed-source tier leads by 3–5%** on most reasoning benchmarks. The gap is closing (Llama 4-Maverick is within 4% of GPT-5 Vision on MMMU-Pro).
- **Open-weights wins on video and OCR**: Qwen2.5-VL and InternVL 3 lead closed-source on long-video and document understanding.
- **Gemini 3 Pro dominates long-context multimodal**: 2 M tokens is the only credible video + audio + text unified context.
- **VLA evaluation is the new frontier**: EmbodiedBench is the first benchmark to evaluate VLAs on real-robot tasks (50 tasks, 5 robot types, 100 episodes each).
- **World-model evaluation is unresolved**: MMWorld (2026) is the first attempt; the field is still defining what "good" means.

---

## 11. Production patterns for VLM 2.0

### 11.1 Pattern 1: vision-first RAG (ColPali / ColQwen style)

```python
"""
Vision-first RAG: index document pages as images, not text.
This is the 2026 production pattern for PDF, slide, and form understanding.
"""
from colpali_engine import ColPali
from qdrant_client import QdrantClient
import torch

# 1. Index document pages as vision embeddings
colpali = ColPali("vidore/colqwen-2-v0.1")  # 2026 SOTA vision retriever
qdrant = QdrantClient(":memory:")

pages = [
    {"doc_id": "doc-1", "page": 1, "image": load_image("page-1.png")},
    {"doc_id": "doc-1", "page": 2, "image": load_image("page-2.png")},
    # ... up to 10,000 pages
]

embeddings = []
for p in pages:
    emb = colpali.encode_image(p["image"])  # (num_patches, 128)
    embeddings.append({"id": f"{p['doc_id']}-{p['page']}",
                       "vector": emb, "doc_id": p["doc_id"],
                       "page": p["page"]})

qdrant.upsert("documents", embeddings)

# 2. Query
query = "What was Q4 revenue?"
query_emb = colpali.encode_query(query)  # (num_query_tokens, 128)
results = qdrant.search("documents", query_emb, top_k=5)

# 3. Rerank + answer with VLM
top_pages = [r.payload for r in results]
context = "\n".join([
    f"--- Page {p['page']} of {p['doc_id']} ---\n{p['text']}"
    for p in top_pages
])
answer = vlm.generate(f"Question: {query}\n\nContext:\n{context}\n\nAnswer:")
```

Why vision-first RAG beats text-first:

- **No OCR**: tables, charts, figures, and equations are preserved exactly
- **Layout-aware**: the model can use spatial position (top-right, footnote, side margin)
- **Language-agnostic**: works on any script, handwriting, or visual notation
- **Handles complex documents**: scientific papers, legal contracts, financial reports

### 11.2 Pattern 2: vision-aware function calling

```python
"""
Vision-aware function calling — the model sees the screen,
calls a function based on what it sees, and includes image
references in the function call.
"""
tools = [
    {
        "name": "click_on_element",
        "description": "Click on a UI element at a specific (x, y) coordinate.",
        "input_schema": {
            "type": "object",
            "properties": {
                "element_description": {"type": "string"},
                "coordinate": {"type": "array",
                    "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
            },
            "required": ["element_description", "coordinate"],
        },
    },
    {
        "name": "fill_form_field",
        "description": "Fill a form field with a value.",
        "input_schema": {
            "type": "object",
            "properties": {
                "field_label": {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["field_label", "value"],
        },
    },
]

response = client.messages.create(
    model="claude-fable-5",
    max_tokens=2048,
    tools=tools,
    messages=[{
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "base64",
                "data": screenshot_to_base64()}},
            {"type": "text", "text":
                "Fill out the registration form with name 'Alice', email "
                "'alice@example.com', and click Submit."},
        ]
    }],
)
```

### 11.3 Pattern 3: streaming video understanding

```python
"""
Streaming video understanding — process a live video feed
(e.g., security camera, robot POV) and emit structured events.
"""
import asyncio
from google.generativeai import GenerativeModel

gemini = GenerativeModel("gemini-3-pro-video-streaming")

async def process_video_stream(video_url: str):
    """Process a live video stream and emit events."""
    async for event in gemini.stream_video(
        url=video_url,
        fps=2,                       # sample 2 frames per second
        prompt="""
        Emit a structured event for each of:
        - person_entered: {id, timestamp, bbox}
        - person_exited: {id, timestamp}
        - object_moved: {object_id, from, to, timestamp}
        - anomaly: {description, timestamp, severity}
        """,
    ):
        yield event
```

### 11.4 Pattern 4: vision-grounded agents with kill switch

```python
"""
A vision-grounded agent with a kill switch and capability ceiling.
This is the 2026 production pattern for safety-critical deployments.
"""
class SafeMultimodalAgent:
    def __init__(self, vlm, action_space, max_actions_per_minute=10,
                 forbidden_actions=None, kill_switch_url=None):
        self.vlm = vlm
        self.action_space = action_space
        self.max_actions_per_minute = max_actions_per_minute
        self.forbidden_actions = forbidden_actions or set()
        self.kill_switch_url = kill_switch_url
        self.action_count = 0
        self.action_window_start = time.time()

    def check_kill_switch(self):
        """Poll the kill switch URL; abort if activated."""
        if self.kill_switch_url:
            r = requests.get(self.kill_switch_url, timeout=1)
            if r.json().get("killed", False):
                raise RuntimeError("Agent killed by external signal")

    def rate_limit(self):
        """Enforce a per-minute action ceiling."""
        now = time.time()
        if now - self.action_window_start > 60:
            self.action_count = 0
            self.action_window_start = now
        if self.action_count >= self.max_actions_per_minute:
            raise RuntimeError("Rate limit exceeded")
        self.action_count += 1

    def step(self, screen_image, instruction):
        self.check_kill_switch()
        action = self.vlm.plan(screen_image, instruction)
        if action.type in self.forbidden_actions:
            raise PermissionError(f"Action {action.type} is forbidden")
        self.rate_limit()
        return self.action_space.execute(action)
```

---

## 12. The 2027–2028 outlook

### 12.1 The next 18 months

| Trend | When | Why |
|-------|------|-----|
| **Multimodal becomes default** | Already (2026) | Every frontier model is native multimodal; text-only is the legacy mode |
| **VLA in production** | H2 2026 | π0.5, HPT, CrossFormer hit consumer price points |
| **World models replace game engines** | 2027 | Real-time interactive worlds become the default for sim, training, and entertainment |
| **"Video is the new text"** | 2027 | 80% of training data is video (vs 50% in 2025); 4D world models ingest it natively |
| **Multimodal agents at scale** | 2027 | 100+ billion agent invocations/year across enterprise |
| **On-device multimodal** | 2026–2027 | iPhone 18, Pixel 11, Galaxy S27 all run VLM 2.0 locally (3B–7B params) |
| **Embodied AI hits mass production** | 2027–2028 | Home robots, warehouse robots, autonomous vehicles all run VLAs |
| **VLM 3.0**: 4D spatio-temporal native | 2028 | Native processing of 3D space + time; replaces VLM 2.0 |

### 12.2 Strategic implications for builders

1. **Pick your multimodal tier**: closed (API, fastest) vs. open-weights (self-host, most control) vs. on-device (cheapest at scale, lowest capability). The right answer depends on latency, data residency, and cost.
2. **Multimodal RAG is the new default** for document understanding. Text-only RAG is legacy.
3. **VLAs are the new robotics stack**. If you build anything physical (warehouse, healthcare, agriculture, manufacturing), plan to use π0 or a derivative within 12 months.
4. **World models are the new simulation stack**. If you train robots, autonomous vehicles, or any embodied system, plan to use Cosmos 1.5+ or a derivative within 12 months.
5. **Frontier safety is now first-party engineering**. Build with kill switches, audit logs, and capability ceilings from day one — not as a retrofit.

---

## 13. Cross-references to existing library docs

This 2026 frontier doc is the **complement** to the library's earlier multimodal coverage. Use them together:

- **Multimodal foundations (2024-era)**: `06-Advanced/01-Multimodal-AI.md` (1,970 lines) — CLIP, LLaVA, ViT, YOLO, DETR, contrastive learning, scaling laws
- **Multimodal market (2025 view)**: `13-Top-Demand/04-Multimodal-AI.md` (799 lines) — GPT-4V/4o, Claude 4 Vision, Gemini 2.5, Llama 4, multimodal RAG
- **Multimodal research (late 2025)**: `17-Research-Frontiers-2026/04-Multimodal-Research.md` (409 lines) — LLaVA-NeXT, CogVLM2, InternVL2, RT-2, Octo
- **Generative media (Jun 2026)**: `28-AI-Video-Audio-Generation/01-Overview-and-Ecosystem.md` (232 lines) — Sora, Veo 3, Runway Gen-4, Suno, ElevenLabs
- **Video architectures (Jun 2026)**: `28-AI-Video-Audio-Generation/02-Video-Architectures-and-Techniques.md` (289 lines) — DiT, 3D VAE, flow matching
- **Audio/music synthesis (Jun 2026)**: `28-AI-Video-Audio-Generation/03-Audio-Music-Synthesis.md` (242 lines) — MusicGen, AudioLDM 2, Suno V4, Udio

**Adjacent categories**:

- **LLM architectures**: `02-LLMs/` for the text trunks used in VLM 2.0
- **Reasoning**: `29-Reasoning-and-Inference-Scaling/` for test-time compute in multimodal
- **Agents**: `03-Agents/` and `13-Top-Demand/02-AI-Agent-Development.md` for vision-aware tool use
- **Embodied AI**: `11-AI-Applications/13-Embodied-AI-Industries.md` for VLA deployment
- **Energy & compute**: `13-Top-Demand/15-AI-Energy-Sustainability-and-Compute-2026.md` for the GPU / power implications of VLM 2.0 + video
- **Regulation**: `21-AI-Regulation-Antitrust/` for EU AI Act Article 53 (Aug 2026) and the Mythos / Fable 5 "too dangerous to release" wave
- **Open-weights race**: `02-LLMs/07-Chinese-AI-Ecosystem-and-Open-Weights-Race.md` for Qwen2.5-VL, InternVL 3, Wan 2.5, HunyuanVideo
- **Cybersecurity**: `22-AI-Cybersecurity-Mythos/` for adversarial VLM attacks and deepfake risk
- **Code generation**: `13-Top-Demand/12-AI-Coding-Assistants-Ecosystem.md` for code-in-image (Claude 4.5, GPT-5 Vision)
- **Local AI / self-hosting**: `23-Local-AI-Inference-Self-Hosting/` for on-device VLM 2.0 (iPhone 18, Pixel 11)
- **Hardware acceleration**: see `13-Top-Demand/15-AI-Energy-Sustainability-and-Compute-2026.md` §3 for the silicon that runs VLM 2.0 (TPU v6, Trainium 3, Groq LPU v2, Cerebras WSE-3)

---

## 14. Builder's checklist

For each project, work through this in order:

### 14.1 Decide your multimodal tier

- [ ] Closed (API): best for fastest time-to-market, highest capability ceiling
- [ ] Open-weights (self-host): best for data residency, cost at scale, fine-tuning
- [ ] On-device: best for privacy, latency, cost at consumer scale
- [ ] Hybrid: API for hardest tasks, on-device for easy ones

### 14.2 Pick your model

- [ ] Default: **GPT-5 Vision** (closed, general), **Llama 4-Maverick** (open, general)
- [ ] Long-context video: **Gemini 3 Pro**
- [ ] Long-context text-in-image: **Claude 4.5 Opus Vision**
- [ ] Multilingual / EU-hosted: **Pixtral 2-124B** or **Aya Vision 2**
- [ ] Best open-weights OCR: **Qwen2.5-VL-72B**
- [ ] Best open-weights video: **InternVL 3-78B**
- [ ] Robotics: **π0** or **OpenVLA-7B** (fine-tune from there)

### 14.3 Architect your pipeline

- [ ] Vision-first RAG for documents (ColPali / ColQwen 2)
- [ ] Vision-aware function calling for agents
- [ ] Streaming video for live feeds (Gemini 3, Qwen2.5-VL)
- [ ] Multimodal cache for repeated images (save 80%+ on cost)

### 14.4 Build for safety

- [ ] Kill switch (external signal, polls every 1 s)
- [ ] Rate limit (max actions per minute)
- [ ] Forbidden action set (e.g., "no payments over $100", "no email send to external")
- [ ] Audit log (every action, every prompt, every response)
- [ ] Capability ceiling (cap the max number of steps, max spend, max scope)
- [ ] Red-team evaluation: AgentHarm, MASK, WMDP

### 14.5 Measure

- [ ] Quality: MMMU-Pro (your model), domain-specific evals
- [ ] Latency: p50, p95, p99 per task
- [ ] Cost: $/task, $/1k tokens, $/image
- [ ] Safety: red-team pass rate, agent-escape rate, hallucination rate
- [ ] User satisfaction: thumbs up/down, task success rate

### 14.6 Forward plan

- [ ] Track VLM 2.0 → VLM 3.0 (4D spatio-temporal) for 2028
- [ ] Track on-device VLM for 2026–2027
- [ ] Track world models for 2027+ (Genie 3, Cosmos 1.5+)
- [ ] Track VLA for 2027+ (π0.5 → π1, HPT, CrossFormer)

---

## 15. Glossary

- **AAR (Agent Action Risk)**: Anthropic's 2026 eval for risky agent actions (browse, transact, email, code-exec)
- **AgentHarm (benchmark)**: 2026 benchmark for multimodal agent safety; 220 tasks across 11 harm categories
- **CC-BY-NC**: Open-weight license requiring non-commercial use
- **ColPali / ColQwen**: Document-retrieval models that index page images (not text) using vision encoders
- **Constitutional AI v2.5**: Anthropic's safety method; rules-based self-critique, version 2.5 (2026)
- **Cross-embodiment**: A single VLA trained on many robot types (RT-2, Octo, OpenVLA, π0)
- **DiT (Diffusion Transformer)**: The 2024+ architecture for video generation (Sora-1, Wan 2.5, Sora 2)
- **DROID**: Open X-Embodiment's largest sub-dataset, 76k trajectories
- **EmbodiedBench**: 2026 benchmark for VLA evaluation, 50 tasks, 5 robot types
- **Flow matching**: 2024+ alternative to DDPM diffusion; faster, higher quality
- **FronTierEval**: A benchmark suite for "frontier" capabilities (autonomous replication, self-improvement, deception)
- **Genie 3**: Google DeepMind's 2026 research world model; 24 fps, interactive
- **HPT**: Heterogeneous Pre-trained Transformers; Toyota Research's 2026 generalist VLA
- **KV-cache reuse**: A 2026 inference optimization for video models that re-uses cached key-value pairs across roll-out steps
- **MASK**: 2026 benchmark for multimodal safety
- **MMBench v2**: 2026 multi-image, multilingual multimodal benchmark
- **MMMU-Pro**: 2025 (v2 in 2026) college-level multimodal reasoning benchmark
- **MMWorld**: 2026 benchmark for world models (real + generated environments)
- **MoE (Mixture of Experts)**: Architecture where only a subset of parameters is active per token; used in Pixtral 2-124B
- **Native multimodal**: A model that processes image, text, audio, and video with a single transformer (no separate encoders)
- **Native audio**: The 2026 frontier generates synchronized audio with video (Sora 2, Veo 3.5, HunyuanVideo 2)
- **Object permanence**: The property that a model remembers objects even when they leave the frame
- **Octo**: 2024 open-weights VLA, 93 M examples, 22 robot types
- **Open X-Embodiment**: The 2024+ open dataset of robot trajectories; 1.5 M trajectories by 2026
- **OpenVLA**: 2024 7B open-weights VLA, beats RT-2 on 29 tasks
- **π0 (pi-zero)**: Physical Intelligence's April 2026 production-grade generalist VLA
- **π0.5**: 2026 VLA with language-conditioned flow matching
- **PaliGemma 2**: Google's 2024 open-weights VLM (predecessor to Gemini 3 era open models)
- **Project Genie**: Google's January 2026 consumer-facing real-time world-model product
- **RDT-1B**: 2025 1B-param VLA with diffusion-based actions
- **Real-time interactive**: A world model that can be queried at ≥ 24 fps with action conditioning
- **SigLIP-2**: Google's 2024+ image-text encoder; improved over CLIP
- **Sora 2**: OpenAI's 2026 video model with world-model class consistency
- **Spatial-temporal attention**: 3D attention across (height, width, time) tokens
- **Spoken dialogue**: Native audio model capability (Gemini 3, Aya Vision 2)
- **VLA (Vision-Language-Action)**: A model that maps vision + language to continuous action
- **VLM 2.0**: 2026 frontier native multimodal architecture (image-patch-as-token, end-to-end trained)
- **Wan 2.5**: Alibaba's March 2026 open-weights video model (Apache 2.0); beats Veo 3 on physics consistency
- **WMDP (World Model Deployment Protocol)**: 2026 safety protocol for world models; capacity + alignment gates
- **World model**: A generative model that learns environment dynamics and can be queried interactively

---

*This document complements the library's existing 3,941 lines of multimodal coverage (`06-Advanced/01-Multimodal-AI.md` + `13-Top-Demand/04-Multimodal-AI.md` + `17-Research-Frontiers-2026/04-Multimodal-Research.md` + the 3 docs in `28-AI-Video-Audio-Generation/`). It is the 2026 frontier layer, focused on VLM 2.0, VLA, and world models. See `13-Top-Demand/15-AI-Energy-Sustainability-and-Compute-2026.md` for the GPU / power implications, and `21-AI-Regulation-Antitrust/` for the EU AI Act Article 53 (Aug 2026) regulatory framing.*

---
**See also:**
- [09 - Multimodal AI Governance: Governing Vision, Language, and Action](21-AI-Regulation-Antitrust/09-Multimodal-AI-Governance.md)
- [Multimodal AI: Architectures, Models, and Alignment](06-Advanced/01-Multimodal-AI.md)
- [04 — Multimodal Research: The Frontier (2025–2026)](07-Emerging/17-Research-Frontiers-2026/04-Multimodal-Research.md)
