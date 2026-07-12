# Computer Vision: Core Topics

> **Category:** 66 — Computer Vision  \n> **Last Updated:** July 2026  \n> **Cross-references:** [01-Overview.md](01-Overview.md), [01-Foundations/03-Deep-Learning.md](../01-Foundations/03-Deep-Learning.md), [02-LLMs/01-Transformer-Architecture.md](../02-LLMs/01-Transformer-Architecture.md), [06-Advanced/02-Diffusion-Models.md](../06-Advanced/02-Diffusion-Models.md), [50-Multimodal-AI/](../50-Multimodal-AI/)

---

## Table of Contents

1. [Convolutional Foundations](#1-convolutional-foundations)
2. [Attention and Vision Transformers](#2-attention-and-vision-transformers)
3. [Self-Supervised Pretraining](#3-self-supervised-pretraining)
4. [Vision-Language Alignment](#4-vision-language-alignment)
5. [Detection Architectures](#5-detection-architectures)
6. [Segmentation Paradigms](#6-segmentation-paradigms)
7. [Generative Vision (Diffusion)](#7-generative-vision-diffusion)
8. [Video Understanding](#8-video-understanding)
9. [3D and Depth](#9-3d-and-depth)
10. [Document and OCR Vision](#10-document-and-ocr-vision)
11. [On-Device and Efficient CV](#11-on-device-and-efficient-cv)

---

## 1. Convolutional Foundations

ConvNets remain the workhorse for efficient, well-understood vision tasks.

**Core operations:**
- **Convolution**: local receptive field, weight sharing, translation equivariance.
- **Pooling**: downsampling, translation invariance.
- **Residual connections** (ResNet, 2015): enable 100+ layer training.

```python
import torch.nn as nn

class TinyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))
```

**Why CNNs still matter in 2026:** they are tiny, fast, and battle-tested for edge deployment (see [62-Edge-AI-and-On-Device-Inference/](../62-Edge-AI-and-On-Device-Inference/)). Families: ResNet, EfficientNet, ConvNeXt, RegNet, MobileNet.

---

## 2. Attention and Vision Transformers

ViT (Dosovitskiy et al., 2020) treats an image as a sequence of patches:

```
IMAGE (H×W×3) ──split into 16×16 patches──> N = HW/256 tokens
                                    +[CLS] token
                                    +positional embedding
                                           │
                                     Transformer encoder
                                           │
                                       [CLS] → classification head
```

**Patch embedding:**
```python
import torch
from torch import nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch=16, in_ch=3, embed=768):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed, patch, stride=patch)
    def forward(self, x):
        x = self.proj(x)            # (B, embed, H/p, W/p)
        return x.flatten(2).transpose(1, 2)  # (B, N, embed)
```

**Trade-offs (CNN vs ViT):**

| Aspect | CNN | ViT |
|---|---|---|
| Inductive bias | Strong (locality) | Weak (data-hungry) |
| Data need | Moderate | Large (or pretrain) |
| Global context | Needs many layers | Immediate (attention) |
| Edge efficiency | Excellent | Needs distillation/quant |
| Transfer | Good | Excellent |

Modern hybrids: **ConvNeXt** (CNN modernized), **CoAtNet**, **MaxViT** blend both.

---

## 3. Self-Supervised Pretraining

Labeled data is expensive; self-supervision learns from unlabeled images.

**Contrastive (SimCLR / MoCo / BYOL):** pull same-image views together, push others apart.
**Masked image modeling (MAE, 2021):** mask ~75% patches, reconstruct pixels.
**DINO / DINOv2 (2021–2023):** self-distillation produces superb features — DINOv2 is a 2026 feature-extraction standard.

```python
# Conceptual MAE forward (pseudocode)
x_patches = patchify(image)              # (B, N, D)
masked, ids_keep = random_mask(x_patches, ratio=0.75)
feat = encoder(masked)                   # visible only
recon = decoder(feat, ids_keep)          # reconstruct all
loss = mse(recon, x_patches)
```

Self-supervised backbones are the foundation for detection/segmentation heads and for VLMs (see §4).

---

## 4. Vision-Language Alignment

The 2021 CLIP breakthrough aligned image and text embeddings so that *zero-shot classification* became possible:

```
image ──> image encoder ──> img_emb
text  ──> text encoder  ──> txt_emb      (shared space, contrastive trained)
similarity = cosine(img_emb, txt_emb)
```

**Zero-shot classification without retraining:**
```python
from transformers import CLIPProcessor, CLIPModel
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = load_image("factory_defect.jpg")
labels = ["normal part", "cracked weld", "missing bolt"]
inputs = proc(text=labels, images=image, return_tensors="pt", padding=True)
with torch.no_grad():
    out = model(**inputs)
logits = out.logits_per_image.softmax(-1)
print(dict(zip(labels, logits[0].tolist())))
```

**Key VLM families (2026):**
- **CLIP / OpenCLIP** — contrastive alignment.
- **SigLIP** — uses sigmoid loss, better at scale.
- **BLIP / BLIP-2** — image-text pretraining for captioning/VQA.
- **LLaVA / InternVL / Qwen-VL / Pixtral** — instruction-following multimodal LLMs (see [50-Multimodal-AI/](../50-Multimodal-AI/)).
- **Florence-2 / Kosmos-2** — unified "task-as-text" perception.

This unification is why CV no longer needs a model-per-task (see [01-Overview.md](01-Overview.md#3-why-cv-is-a-2026-headline-capability)).

---

## 5. Detection Architectures

**Two-stage (Faster R-CNN):** region proposals → classify/refine boxes. Accurate, slower.
**One-stage (YOLO, RetinaNet):** predict boxes directly. Fast, real-time.
**Anchor-free (FCOS, CenterNet):** predict per-pixel box params.
**Transformer (DETR):** set-prediction with bipartite matching — no NMS.

```python
# DETR-style detection (conceptual)
boxes = detr(image)                      # list of (x,y,w,h,class,score)
keep = [b for b in boxes if b.score > 0.7]
```

**YOLO in practice (ultralytics):**
```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
results = model("shop_floor.jpg")
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls, conf = int(box.cls[0]), float(box.conf[0])
```

**Open-vocabulary detection** (GroundingDINO, OWLv2) lets you detect *arbitrary text queries* — a 2026 game-changer for long-tail objects.

---

## 6. Segmentation Paradigms

| Paradigm | Output | Model |
|---|---|---|
| Semantic | per-pixel class | DeepLab, SegFormer |
| Instance | per-object mask | Mask R-CNN, Mask2Former |
| Panoptic | things + stuff | Panoptic FPN, Mask2Former |
| Promptable | mask from point/box/text | **SAM** |

**Segment Anything (SAM, 2022):** a promptable model — give it a point, box, or mask, get a segmentation. Foundation for interactive labeling.

```python
from segment_anything import sam_model_registry, SamPredictor
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)
predictor.set_image(image)
masks, scores, _ = predictor.predict(point_coords=[[x,y]], point_labels=[1])
```

**SAM + GroundingDINO = text-prompted segmentation:**
```text
TEXT "red helmet" → GroundingDINO boxes → SAM masks → labeled segmentation
```
This "grounded SAM" pattern powers zero-shot perception for agents (see [60-Physical-AI-and-Embodied-Intelligence/](../60-Physical-AI-and-Embodied-Intelligence/)).

---

## 7. Generative Vision (Diffusion)

Diffusion models generate images by iteratively denoising noise (see [06-Advanced/02-Diffusion-Models.md](../06-Advanced/02-Diffusion-Models.md)):

```
x_T ~ N(0,I)  --denoise-->  x_{T-1}  ...  -->  x_0 = image
        ε_θ(x_t, t, condition=text)
```

**Text-to-image (SDXL, Flux, Imagen):**
```python
from diffusers import StableDiffusionXLPipeline
pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/sdxl-base-1.0")
image = pipe("a robot inspecting a solar panel, photorealistic").images[0]
```

**Why it matters for CV:** generative models also produce **training data** (synthetic labels) and enable inpainting/editing — a bridge to [51-Synthetic-Data-Generation/](../51-Synthetic-Data-Generation/).

---

## 8. Video Understanding

Video adds a temporal axis. Approaches:
- **3D CNNs** (I3D, SlowFast) — spatiotemporal kernels.
- **Video Transformers** (TimeSformer, ViViT) — factorized attention.
- **Video LLMs** (Video-LLaVA, InternVideo, Qwen-VL video) — frame tokens + LLM reasoning.

```python
# Extract frames, encode, ask a VLM
frames = sample_video("assembly_line.mp4", n=8)
answer = vlm_qa(frames, "Did the operator skip step 3? Explain.")
```

Use cases: action recognition, anomaly detection in manufacturing, surveillance, sports analytics, autonomous driving (see [60-](../60-Physical-AI-and-Embodied-Intelligence/)).

---

## 9. 3D and Depth

- **Monocular depth**: Depth Anything, MiDaS estimate metric/relative depth from one image.
- **Stereo**: two cameras → disparity → depth.
- **NeRF / 3D Gaussian Splatting**: novel-view synthesis from images.
- **Point clouds**: PointNet, voxel CNNs for LIDAR (autonomous driving, robotics).

```python
from transformers import pipeline
depth_pipe = pipeline("depth-estimation", model="Intel/dpt-large")
depth = depth_pipe("street_view.jpg")["depth"]
```

Depth + segmentation feeds embodied agents' spatial understanding (see [60-](../60-Physical-AI-and-Embodied-Intelligence/)).

---

## 10. Document and OCR Vision

Specialized models read documents:
- **TrOCR** — Transformer OCR.
- **Donut / Nougat** — OCR-free document understanding (direct image→text/JSON).
- **LayoutLM** — jointly models text + layout + image for forms.

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel
proc = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-invoice")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-invoice")
pixel = proc(image, return_tensors="pt").pixel_values
out = model.generate(pixel, max_length=512,
                     decoder_input_ids=proc.tokenizer("<s_invoice>", return_tensors="pt").input_ids)
print(proc.batch_decode(out, skip_special_tokens=False)[0])
```
Tie-ins: finance/legal document automation ([49-AI-for-Legal-and-LegalTech/](../49-AI-for-Legal-and-LegalTech/)).

---

## 11. On-Device and Efficient CV

Running CV on phones/cameras/cars requires compression (see [30-Small-Language-Models/](../30-Small-Language-Models/), [62-Edge-AI-and-On-Device-Inference/)):

- **Quantization**: FP32 → INT8 (or INT4) with minimal accuracy loss.
- **Pruning**: remove redundant channels/filters.
- **Distillation**: train small student from large teacher.
- **Efficient backbones**: MobileNet, EfficientNet-Lite, MobileSAM.

```python
import torch
model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
model_prepared = torch.ao.quantization.prepare(model.eval())
# calibrate on sample data...
model_int8 = torch.ao.quantization.convert(model_prepared)
torch.jit.save(torch.jit.script(model_int8), "cv_int8.pt")
```

These techniques make privacy-preserving, offline CV practical for [62-](../62-Edge-AI-and-On-Device-Inference/) and [40-AI-Data-Sovereignty-and-Privacy/](../40-AI-Data-Sovereignty-and-Privacy/).

---

*Core Topics establishes the conceptual and architectural foundations of modern CV. For implementation internals, see [03-Technical-Deep-Dive.md](03-Technical-Deep-Dive.md); for the toolchain, see [04-Tools-and-Frameworks.md](04-Tools-and-Frameworks.md).*
