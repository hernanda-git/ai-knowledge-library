# Computer Vision: Tools and Frameworks

> **Category:** 66 — Computer Vision  \n> **Last Updated:** July 2026  \n> **Cross-references:** [03-Technical-Deep-Dive.md](03-Technical-Deep-Dive.md), [01-Foundations/03-Deep-Learning.md](../01-Foundations/03-Deep-Learning.md), [50-Multimodal-AI/](../50-Multimodal-AI/), [62-Edge-AI-and-On-Device-Inference/](../62-Edge-AI-and-On-Device-Inference/), [51-Synthetic-Data-Generation/](../51-Synthetic-Data-Generation/)

---

## Table of Contents

1. [Framework Landscape](#1-framework-landscape)
2. [PyTorch + torchvision](#2-pytorch--torchvision)
3. [Hugging Face Transformers & Diffusers](#3-hugging-face-transformers--diffusers)
4. [Detection & Segmentation Suites](#4-detection--segmentation-suites)
5. [Annotation and Labeling](#5-annotation-and-labeling)
6. [MLOps for Vision](#6-mlops-for-vision)
7. [Cloud CV APIs](#7-cloud-cv-apis)
8. [Edge and Mobile Toolchains](#8-edge-and-mobile-toolchains)
9. [Video and 3D Tooling](#9-video-and-3d-tooling)
10. [Synthetic Data & Augmentation Libraries](#10-synthetic-data--augmentation-libraries)
11. [Choosing a Stack (Decision Guide)](#11-choosing-a-stack-decision-guide)

---

## 1. Framework Landscape

| Layer | Go-to tools |
|---|---|
| Core DL | PyTorch (dominant), JAX/Flax (Google), TensorFlow (legacy) |
| Vision models | torchvision, Hugging Face Transformers/Diffusers |
| Detection/Seg | MMDetection, Detectron2, ultralytics (YOLO) |
| Annotation | Label Studio, CVAT, Roboflow |
| MLOps | Weights & Biases, ClearML, DVC, Comet |
| Deploy | ONNX Runtime, TensorRT, TorchServe, Triton, ExecuTorch |
| Cloud | AWS Rekognition, GCP Vision, Azure CV |

**PyTorch is the de-facto 2026 standard** for research and production vision; torchvision ships canonical model weights (ResNet, ViT, FCN, Mask R-CNN, SegFormer).

---

## 2. PyTorch + torchvision

```python
import torch
from torchvision import models, transforms

weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights).eval()
pre = weights.transforms()                 # correct normalization included

img = transforms.ToTensor()(load_image("cat.jpg")).unsqueeze(0)
with torch.no_grad():
    probs = torch.softmax(model(pre(img)), -1)
top5 = probs.topk(5)
for idx, p in zip(top5.indices[0], top5.values[0]):
    print(weights.meta["categories"][idx], round(p.item(), 3))
```

**torchvision detection/segmentation models:**
```python
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, maskrcnn_resnet50_fpn
det = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT").eval()
seg = maskrcnn_resnet50_fpn(weights="DEFAULT").eval()
```

**Transforms v2 (recommended):**
```python
from torchvision.transforms import v2
aug = v2.Compose([v2.RandomResizedCrop(224), v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize([.485,.456,.406],[.229,.224,.225])])
```

---

## 3. Hugging Face Transformers & Diffusers

**ViT classification:**
```python
from transformers import ViTImageProcessor, ViTForImageClassification
proc = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
```

**GroundingDINO + SAM pipeline (zero-shot perception):**
```python
from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection
from transformers import SamModel, SamProcessor

gd_proc = GroundingDinoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
gd_model = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")
inputs = gd_proc(images=image, text="a red helmet. a person.", return_tensors="pt")
with torch.no_grad():
    out = gd_model(**inputs)
results = gd_proc.post_process_grounded_object_detection(out, inputs, threshold=0.3)
boxes = results[0]["boxes"]
# feed boxes into SAM for masks
sam_proc = SamProcessor.from_pretrained("facebook/sam-vit-base")
sam = SamModel.from_pretrained("facebook/sam-vit-base")
sam_inputs = sam_proc(image, input_boxes=boxes.unsqueeze(0), return_tensors="pt")
with torch.no_grad():
    sam_out = sam(**sam_inputs)
masks = sam_proc.image_processor.post_process_masks(...)
```

**Diffusion generation:**
```python
from diffusers import StableDiffusionXLPipeline
pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/sdxl-base-1.0").to("cuda")
pipe("a warehouse robot, cinematic lighting").images[0].save("robot.png")
```

See [50-Multimodal-AI/](../50-Multimodal-AI/) and [06-Advanced/02-Diffusion-Models.md](../06-Advanced/02-Diffusion-Models.md).

---

## 4. Detection & Segmentation Suites

### 4.1 ultralytics (YOLO)
```python
from ultralytics import YOLO
m = YOLO("yolov8x.pt")
m.train(data="coco8.yaml", epochs=50, imgsz=640)      # or .predict()
m.export(format="onnx")                                # deploy
```

### 4.2 MMDetection / MMSegmentation (OpenMMLab)
```python
from mmdet.apis import init_detector, inference_detector
model = init_detector("configs/yolo/yolov3_d53_320_273e_coco.py", "ckpt.pth")
result = inference_detector(model, "street.jpg")
```

### 4.3 Detectron2 (Meta)
```python
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
cfg = get_cfg(); cfg.merge_from_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = "model_final.pth"
pred = DefaultPredictor(cfg)(image)
```

**Comparison:**

| Suite | Strength | Best for |
|---|---|---|
| ultralytics | speed, ergonomics | real-time detection, fast prototypes |
| MMDetection | breadth, research | SOTA zoo, custom heads |
| Detectron2 | maturity, Caffe2 roots | production segmentation |

---

## 5. Annotation and Labeling

- **Label Studio** — flexible, multi-modal, model-assisted.
- **CVAT** — open-source, video tracking, polygons.
- **Roboflow** — dataset management + augmentation + hosting.
- **V7 / SuperAnnotate** — enterprise QA workflows.

**Model-assisted loop (cut cost):**
```
1. Train weak model  → 2. SAM pre-segment  → 3. Human corrects  → 4. Retrain
```

```yaml
# Roboflow augmentation example (augment.yaml)
augmentation:
  - flip: horizontal
  - rotate: {limit: 20}
  - brightness: {limit: 0.2}
  - mosaic: 0.5
```

---

## 6. MLOps for Vision

| Need | Tool |
|---|---|
| Experiment tracking | Weights & Biases, Comet, ClearML |
| Data versioning | DVC, LakeFS |
| Pipeline | Kubeflow, Metaflow, Airflow |
| Serving | TorchServe, Triton, BentoML, KServe |
| Monitoring | Evidently, Arize, WhyLabs |

**Example W&B logging:**
```python
import wandb
wandb.init(project="cv-defect", config=dict(lr=1e-3, arch="vit-b"))
wandb.log({"val_mAP": mAP, "epoch": e})
wandb.save("model.pt")
```

Drift monitoring matters: a CV model deployed in a new factory lighting condition silently degrades (tie to [52-AI-Hallucination-Detection-and-Mitigation/](../52-AI-Hallucination-Detection-and-Mitigation/)).

---

## 7. Cloud CV APIs

| Provider | Service | Notes |
|---|---|---|
| AWS | Rekognition | labels, faces, text, PPE |
| GCP | Cloud Vision | labels, OCR, safe-search |
| Azure | Computer Vision | OCR, spatial, video |
| Clarifai | Clarifai | custom model hosting |
| Roboflow | Inference | self-host or cloud YOLO |

```python
import boto3
rek = boto3.client("rekognition")
resp = rek.detect_labels(Image={"S3Object": {"Bucket": "b", "Name": "img.jpg"}}, MaxLabels=10)
for l in resp["Labels"]:
    print(l["Name"], round(l["Confidence"],1))
```

Cloud APIs are fastest to start but have data-residency and cost limits (see [25-Multi-Cloud-AI-Strategy/](../25-Multi-Cloud-AI-Strategy/), [40-AI-Data-Sovereignty-and-Privacy/](../40-AI-Data-Sovereignty-and-Privacy/)).

---

## 8. Edge and Mobile Toolchains

- **ExecuTorch** (PyTorch edge runtime) — phones, microcontrollers.
- **TensorRT** (NVIDIA) — GPU inference, fp16/int8.
- **ONNX Runtime** — cross-platform, CoreML/EdgeTPU backends.
- **TFLite / TF Micro** — Android / embedded.
- **CoreML** — Apple devices.
- **Qualcomm AI Engine / NNAPI** — mobile NPUs.

```python
# Export to ONNX then run anywhere
torch.onnx.export(model, dummy, "cv.onnx", opset_version=17)
# ONNX Runtime
import onnxruntime as ort
sess = ort.InferenceSession("cv.onnx", providers=["CPUExecutionProvider"])
```

See [62-Edge-AI-and-On-Device-Inference/](../62-Edge-AI-and-On-Device-Inference/) and [63-GPU-Kernel-and-Inference-Performance-Engineering/](../63-GPU-Kernel-and-Inference-Performance-Engineering/).

---

## 9. Video and 3D Tooling

- **Video:** decord / PyAV (loading), VideoMAE (pretrain), Video-LLaVA (VLM).
- **3D / NeRF:** nerfstudio, PyTorch3D, Open3D, Gaussian-Splatting repos.
- **Depth:** Depth Anything, MiDaS (Hugging Face pipelines).
- **LIDAR/point cloud:** OpenPCDet, PDAL.

```python
from transformers import pipeline
depth = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small")
dmap = depth("road.jpg")["depth"]
```

3D/depth feeds embodied agents (see [60-Physical-AI-and-Embodied-Intelligence/](../60-Physical-AI-and-Embodied-Intelligence/)).

---

## 10. Synthetic Data & Augmentation Libraries

- **Diffusers / ComfyUI** — generate labeled training images (see [51-Synthetic-Data-Generation/](../51-Synthetic-Data-Generation/)).
- **Albumentations** — fast image augmentations.
- **kornia** — differentiable CV ops (aug, homography) in PyTorch.
- **imgaug** — legacy augmentation.
- **NVIDIA Omniverse / Blender** — photoreal 3D-rendered training sets.

```python
import albumentations as A
tf = A.Compose([A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(p=0.2), A.ShiftScaleRotate(p=0.4)])
aug = tf(image=img, mask=mask); img, mask = aug["image"], aug["mask"]
```

Synthetic data closes the long-tail gap for rare classes (LVIS problem, [02-Core-Topics.md](02-Core-Topics.md#1-convolutional-foundations)).

---

## 11. Choosing a Stack (Decision Guide)

```
START
 │
 ├─ Need a quick detector? ──────────────► ultralytics YOLO
 ├─ Research / custom head? ────────────► MMDetection + PyTorch
 ├─ Zero-shot / text prompts? ──────────► GroundingDINO + SAM (HF)
 ├─ Generate images / synthetic data? ──► Diffusers
 ├─ On-device? ─────────────────────────► ExecuTorch / TFLite / CoreML
 ├─ Video understanding? ───────────────► Video-LLaVA / InternVideo
 ├─ No ML team, fast start? ────────────► Cloud CV API
 └─ Need full control + privacy? ───────► self-host ONNX/TensorRT
```

**Rule of thumb (2026):** start with a VLM + grounding for flexibility, drop to a specialized CNN/YOLO only when latency or edge constraints demand it. Match the tool to [62-Edge-AI-and-On-Device-Inference/](../62-Edge-AI-and-On-Device-Inference/) and [41-AI-Cost-Optimization-and-Enterprise-ROI/](../41-AI-Cost-Optimization-and-Enterprise-ROI/) constraints.

---

*Tools and Frameworks maps the practical ecosystem. Return to [01-Overview.md](01-Overview.md) for the big picture or see [05-Future-Outlook.md](05-Future-Outlook.md) for the roadmap.*
