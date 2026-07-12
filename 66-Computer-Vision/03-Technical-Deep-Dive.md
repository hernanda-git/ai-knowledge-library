# Computer Vision: Technical Deep Dive

> **Category:** 66 — Computer Vision  \n> **Last Updated:** July 2026  \n> **Cross-references:** [02-Core-Topics.md](02-Core-Topics.md), [01-Foundations/03-Deep-Learning.md](../01-Foundations/03-Deep-Learning.md), [01-Foundations/05-Training-Methodologies.md](../01-Foundations/05-Training-Methodologies.md), [02-LLMs/01-Transformer-Architecture.md](../02-LLMs/01-Transformer-Architecture.md), [06-Advanced/64-AI-Model-Explainability-and-XAI/](../06-Advanced/64-AI-Model-Explainability-and-XAI/), [63-GPU-Kernel-and-Inference-Performance-Engineering/](../63-GPU-Kernel-and-Inference-Performance-Engineering/)

---

## Table of Contents

1. [Backbone Internals: From Convolutions to Tokens](#1-backbone-internals-from-convolutions-to-tokens)
2. [The DETR Bipartite-Matching Loss](#2-the-detr-bipartite-matching-loss)
3. [CLIP Training Objective in Detail](#3-clip-training-objective-in-detail)
4. [SAM Architecture and Prompt Encoding](#4-sam-architecture-and-prompt-encoding)
5. [Diffusion Math (DDPM)](#5-diffusion-math-ddpm)
6. [Loss Functions by Task](#6-loss-functions-by-task)
7. [Data Augmentation Pipelines](#7-data-augmentation-pipelines)
8. [Training at Scale: Distributed and Mixed Precision](#8-training-at-scale-distributed-and-mixed-precision)
9. [Quantization and Compilation for Inference](#9-quantization-and-compilation-for-inference)
10. [Interpretability and Failure Analysis](#10-interpretability-and-failure-analysis)
11. [Reproducibility Checklist](#11-reproducibility-checklist)

---

## 1. Backbone Internals: From Convolutions to Tokens

### 1.1 Convolution math

For input $X \in \mathbb{R}^{C_{in}\times H\times W}$, a kernel $K \in \mathbb{R}^{C_{out}\times C_{in}\times k\times k}$ produces:

$$Y_{o,i,j} = \sum_{c=1}^{C_{in}}\sum_{u,v} K_{o,c,u,v}\, X_{c,\,i+u,\,j+v} + b_o$$

Output size with padding $p$ and stride $s$: $H_{out}=\lfloor(H-k+2p)/s\rfloor+1$.

### 1.2 ResNet residual block

```python
class BasicBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(c)
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + x)        # skip connection
```

The skip connection lets gradients flow: $\frac{\partial L}{\partial x^{(l)}} = \frac{\partial L}{\partial x^{(l+1)}}(\cdot) + \frac{\partial L}{\partial x^{(l+1)}}$. Prevents vanishing gradients in deep nets.

### 1.3 ViT encoder block

```python
class ViTBlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(self, x):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + a
        return x + self.mlp(self.ln2(x))    # pre-norm residuals
```

Note **pre-norm** (LayerNorm before attention) is the 2026 default — more stable than post-norm.

---

## 2. The DETR Bipartite-Matching Loss

DETR predicts a fixed set of $N$ boxes (e.g., 100) and matches predictions to ground truth via **Hungarian algorithm**:

```python
from scipy.optimize import linear_sum_assignment
import torch

def bipartite_match(pred, gt):
    # cost matrix C[i,j] = classification + bbox(1-IoU)
    C = -pred_cls_logprob[:, gt_cls] + bbox_cost(pred_box, gt_box)  # (N, M)
    pred_idx, gt_idx = linear_sum_assignment(C.cpu().numpy())
    return pred_idx, gt_idx            # unique 1:1 assignment
```

Total loss (matched pairs only):

$$\mathcal{L} = \lambda_{cls}\mathcal{L}_{cls} + \lambda_{box}\big(\mathcal{L}_{L1}(b_{\hat\sigma(i)}, b_i) + \lambda_{giou}\mathcal{L}_{GIoU}\big)$$

Unmatched predictions are trained as "no object" ($\varnothing$). This removes hand-crafted NMS and anchors.

---

## 3. CLIP Training Objective in Detail

Given a batch of $B$ image-text pairs, CLIP maximizes agreement on the diagonal:

$$\mathcal{L}_{i2t} = -\frac{1}{B}\sum_i \log \frac{\exp(\langle I_i,T_i\rangle/\tau)}{\sum_j \exp(\langle I_i,T_j\rangle/\tau)}$$
$$\mathcal{L}_{t2i} = -\frac{1}{B}\sum_i \log \frac{\exp(\langle T_i,I_i\rangle/\tau)}{\sum_j \exp(\langle T_j,I_i\rangle/\tau)}$$

```python
import torch.nn.functional as F
def clip_loss(img_emb, txt_emb, tau=0.07):
    img_emb = F.normalize(img_emb); txt_emb = F.normalize(txt_emb)
    logits = img_emb @ txt_emb.t() / tau
    labels = torch.arange(len(logits))
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
```

$\tau$ (temperature) is learned. SigLIP replaces the softmax with a **pairwise sigmoid** loss, which scales better with large batches and noisy data.

---

## 4. SAM Architecture and Prompt Encoding

SAM = **Image Encoder (ViT) + Prompt Encoder + Lightweight Mask Decoder**.

```
image ──ViT──> image_embed (B, 256, 64, 64)
prompt (point/box/mask) ──PromptEncoder──> sparse/dense emb
image_embed + prompt_emb ──MaskDecoder (2-way cross-attn)──> mask (logits)
```

Key detail: the image encoder runs **once**; prompts are cheap, enabling interactive editing.

```python
class PromptEncoder(nn.Module):
    def __init__(self, embed_dim=256, num_points=2):
        super().__init__()
        self.point_embed = nn.Embedding(num_points+1, embed_dim)
        self.box_embed   = nn.Linear(4, embed_dim)
        self.not_a_point = nn.Embedding(1, embed_dim)
    def forward(self, points=None, boxes=None):
        out = []
        if points is not None:
            out.append(self.point_embed(points[..., -1].long()) + self._pe(points[..., :2]))
        if boxes is not None:
            out.append(self.box_embed(boxes))
        return torch.cat(out, 1) if out else self.not_a_point.weight
```

SAM's mask decoder uses **two mask tokens** (whole / part) plus an IoU head to rank predictions.

---

## 5. Diffusion Math (DDPM)

Forward (diffusion) adds noise over $T$ steps:

$$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}\,x_{t-1}, \beta_t I)$$
Closed form: $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\ \epsilon\sim\mathcal{N}(0,I)$.

Reverse (denoising) trains $\epsilon_\theta$ to predict the noise:

$$\mathcal{L}_{simple} = \mathbb{E}_{x_0,t,\epsilon}\big[\|\epsilon - \epsilon_\theta(x_t, t, c)\|^2\big]$$

Sampling (DDIM, faster):
```python
@torch.no_grad()
def ddim_sample(model, cond, steps=50):
    x = torch.randn_like(cond)                 # x_T
    for t in reversed(range(steps)):
        eps = model(x, t, cond)
        x = denoise_step(x, eps, t, steps)     # one-step jump
    return x
```

Latent diffusion (SD) runs this in a VAE's compressed latent space, slashing compute (see [06-Advanced/02-Diffusion-Models.md](../06-Advanced/02-Diffusion-Models.md)).

---

## 6. Loss Functions by Task

| Task | Loss | Notes |
|---|---|---|
| Classification | Cross-entropy | label smoothing helps |
| Detection (DETR) | CE + L1 + GIoU | Hungarian matched |
| Semantic seg | CE + Dice/Focal | class imbalance |
| Instance seg | Mask CE + box loss | Mask R-CNN |
| Depth | Scale-invariant log | relative depth |
| VLM pretrain | InfoNCE (contrastive) | CLIP/SigLIP |
| Diffusion | MSE on noise | $\epsilon$-prediction |
| OCR | CTC / attention CE | variable-length |

**Focal loss** down-weights easy examples — critical for dense, imbalanced detection:
$$\text{FL}(p_t) = -(1-p_t)^\gamma \log(p_t)$$

---

## 7. Data Augmentation Pipelines

Strong augmentation is essential for robust CV.

```python
from torchvision import transforms as T
train_aug = T.Compose([
    T.RandomResizedCrop(224, scale=(0.2, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.4, 0.4, 0.4, 0.1),
    T.RandomGrayscale(p=0.2),
    T.GaussianBlur(23),
    T.ToTensor(), T.Normalize(mean=[.485,.456,.406], std=[.229,.224,.225]),
])
```

**Mixup / CutMix** blend images and labels for regularization. **RandAugment / AutoAugment** automate policy search. For self-supervision, **two correlated views** (crops, color jitter) drive contrastive losses.

---

## 8. Training at Scale: Distributed and Mixed Precision

```python
import torch
from torch.amp import autocast, GradScaler

scaler = GradScaler("cuda")
optimizer.zero_grad()
with autocast("cuda", dtype=torch.bfloat16):
    out = model(images)
    loss = criterion(out, targets)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

Distributed strategies:
- **DataParallel** (single node, legacy).
- **DistributedDataParallel (DDP)** — per-GPU replica, gradient all-reduce.
- **FSDP / ZeRO** — shards params/optim state for huge backbones.
- **Pipeline parallelism** — split layers across GPUs for ViT-H/VLMs.

bfloat16 is the 2026 default for vision transformers (stable, no loss-scaling drama). Tie to [01-Foundations/05-Training-Methodologies.md](../01-Foundations/05-Training-Methodologies.md).

---

## 9. Quantization and Compilation for Inference

**Post-training quantization (PTQ):**
```python
import torch
model.eval()
model.qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")
q = torch.ao.quantization.prepare(model)
calibrate(q, sample_loader)               # collect activation stats
q = torch.ao.quantization.convert(q)      # fold to INT8
```

**Export + compile for deployment:**
```python
example = (torch.randn(1,3,224,224),)
ep = torch.export.export(model, example)
# compile to TensorRT / ONNX / CoreML / ExecuTorch for edge
```

**INT8 vs FP16 vs INT4 trade-off:**

| Prec | Speedup | Accuracy risk | Use |
|---|---|---|---|
| FP32 | 1× | none | reference |
| FP16/BF16 | ~2× | low | GPU inference |
| INT8 | ~3–4× | low–med | server/edge |
| INT4 | ~6–8× | med | on-device VLM |

See [63-GPU-Kernel-and-Inference-Performance-Engineering/](../63-GPU-Kernel-and-Inference-Performance-Engineering/) and [62-Edge-AI-and-On-Device-Inference/](../62-Edge-AI-and-On-Device-Inference/) for the full performance picture.

---

## 10. Interpretability and Failure Analysis

Why did the model predict X? Tools:
- **Grad-CAM**: heatmap from gradient-weighted activations.
- **Saliency / Integrated Gradients**: pixel attribution.
- **Feature visualization**: DINOv2/CLIP probe neurons for "objectness", "scene", "text".

```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
g = cam(input_tensor=img, targets=[ClassifierOutputTarget(cls)])
```

VLMs can **hallucinate** grounding — a confident box on nothing (see [52-AI-Hallucination-Detection-and-Mitigation/](../52-AI-Hallucination-Detection-and-Mitigation/)). Mitigations: confidence thresholds, human-in-the-loop verification, and calibration datasets. Interpretability connects to [06-Advanced/05-Interpretability.md](../06-Advanced/05-Interpretability.md) and [06-Advanced/64-AI-Model-Explainability-and-XAI/](../06-Advanced/64-AI-Model-Explainability-and-XAI/).

---

## 11. Reproducibility Checklist

- [ ] Fixed seeds (`torch.manual_seed`, `numpy`, `random`).
- [ ] Pin library versions (torch, torchvision, transformers).
- [ ] Log config + dataset version (DVC / Weights & Biases).
- [ ] Report full eval protocol (metric, IoU threshold, splits).
- [ ] Release checkpoint + inference script.
- [ ] Calibrate quant on a fixed sample set.
- [ ] Note known failure modes from §10.

A model that tops a leaderboard but fails on a shifted domain is a liability — always validate on production-representative data (see [41-AI-Cost-Optimization-and-Enterprise-ROI/](../41-AI-Cost-Optimization-and-Enterprise-ROI/)).

---

*Deep Dive covers the math and engineering underpinnings. For the practical toolchain, see [04-Tools-and-Frameworks.md](04-Tools-and-Frameworks.md); for what is coming next, see [05-Future-Outlook.md](05-Future-Outlook.md).*
