# Multimodal AI: Architectures, Models, and Alignment

## Table of Contents

1. Introduction to Multimodal AI
2. Vision Models
   - 2.1 Vision Transformers (ViT)
   - 2.2 YOLO (You Only Look Once)
   - 2.3 DETR (DEtection TRansformer)
   - 2.4 SAM (Segment Anything Model)
3. Image Generation Models
   - 3.1 Stable Diffusion 1/2/3/XL
   - 3.2 Flux
   - 3.3 DALL-E 2/3
   - 3.4 Midjourney
4. Video Generation Models
   - 4.1 Sora
   - 4.2 VideoPoet
5. Audio Models
   - 5.1 Whisper
   - 5.2 MusicGen
   - 5.3 AudioCraft
6. Multimodal LLMs
   - 6.1 CLIP
   - 6.2 LLaVA
   - 6.3 GPT-4V/o
   - 6.4 Gemini Vision
   - 6.5 Qwen-VL
   - 6.6 InternVL
7. Speech and TTS Models
   - 7.1 Bark
   - 7.2 XTTS
   - 7.3 CosyVoice
   - 7.4 ChatTTS
8. Music Generation
   - 8.1 Suno
   - 8.2 Udio
9. 3D Generation Models
   - 9.1 Point-E
   - 9.2 DreamFusion
10. Multimodal Embeddings
11. Modality Alignment
12. References and Further Reading

---

## 1. Introduction to Multimodal AI

Multimodal AI refers to artificial intelligence systems that can process, understand, and generate content across multiple modalities including text, images, audio, video, and 3D data. Unlike unimodal systems that operate on a single data type, multimodal models fuse information from different sources to achieve more robust and contextually aware understanding.

### 1.1 The Modality Gap Problem

One of the fundamental challenges in multimodal AI is the modality gap — the representational disconnect between different data types. Text is discrete and symbolic, images are continuous and spatial, audio is temporal and spectral, and video adds a temporal dimension to visual data. Bridging these gaps requires sophisticated alignment techniques.

### 1.2 Core Architectural Paradigms

Multimodal architectures generally follow one of three paradigms:

**Early Fusion**: Input features from all modalities are combined at the input level before processing through a single model. This allows the model to learn cross-modal interactions from the start but requires aligned data.

**Late Fusion**: Each modality is processed independently through separate encoders, and their representations are combined at the decision level. This is modular and allows pretrained unimodal encoders to be reused.

**Hybrid Fusion**: Intermediate representations from different modalities are fused at multiple levels of the network. Cross-attention mechanisms are commonly used to enable information flow between modalities at various depths.

### 1.3 Contrastive Learning in Multimodal Spaces

A breakthrough technique for multimodal learning is contrastive learning, where the model learns to bring representations of matching modality pairs (e.g., an image and its caption) closer in embedding space while pushing non-matching pairs apart. The InfoNCE loss is the standard objective:

```
L = -log( exp(sim(x_i, y_i)/tau) / sum_j(exp(sim(x_i, y_j)/tau)) )
```

where `sim()` is cosine similarity, `tau` is a temperature parameter, and positive pairs `(x_i, y_i)` are matched examples. This is the foundation of models like CLIP and its derivatives.

### 1.4 Modality Alignment Taxonomies

| Alignment Type | Description | Example Models |
|---------------|-------------|----------------|
| Contrastive | Learn shared embedding space via positive/negative pairs | CLIP, SigLIP, ImageBind |
| Generative | Generate one modality conditioned on another | DALL-E, Stable Diffusion |
| Translation | Directly map between modalities | Whisper (speech-to-text), Bark (text-to-speech) |
| Compositional | Combine modalities for unified reasoning | LLaVA, GPT-4V, Gemini |
| Temporal | Align across time-synchronized modalities | Video-audio alignment, VideoCLIP |

### 1.5 Scaling Laws for Multimodal Models

Research has shown that multimodal models exhibit scaling behavior similar to language models but with additional dimensions:

- **Data scaling**: Performance improves with more paired data, but the marginal benefit decreases for lower-quality pairs.
- **Model scaling**: Larger encoders and decoders improve quality, with encoder-decoder balance being critical.
- **Modality scaling**: Adding more modalities during pretraining can improve performance on each individual modality through cross-modal transfer.

---

## 2. Vision Models

Vision models are the foundational building blocks for multimodal systems that process visual information. Modern vision models have evolved from convolutional architectures to transformer-based designs.

### 2.1 Vision Transformers (ViT)

The Vision Transformer (ViT), introduced by Dosovitskiy et al. (2020) at Google Research, was a paradigm shift in computer vision. It demonstrated that a pure transformer architecture, originally designed for NLP, could achieve state-of-the-art results on image classification tasks when given sufficient training data.

#### 2.1.1 Architecture

The ViT architecture divides an input image into fixed-size patches (typically 16x16 pixels), linearly embeds each patch into a sequence of tokens, prepends a learnable [CLS] token, adds positional embeddings, and processes the sequence through standard transformer encoder blocks.

```
Input: Image H x W x C
Patches: N = (H/P) * (W/P) patches of size P x P
Patch Embedding: Linear projection to D dimensions
Position Embeddings: Learnable 1D position encodings
Transformer Encoder: L blocks of Multi-Head Self-Attention + MLP
Output: [CLS] token representation for classification
```

#### 2.1.2 Key Innovations

- **Patch-based tokenization**: Eliminates the need for convolutional feature extractors, using a straightforward linear projection of image patches.
- **Positional encodings**: Since self-attention is permutation-invariant, positional embeddings convey spatial information. ViT uses learnable 1D positional embeddings, though relative positional biases have been adopted in later variants.
- **Pre-training requirements**: ViT requires large-scale pre-training (e.g., ImageNet-21k, JFT-300M) to outperform CNNs, a limitation addressed by later works like DeiT (Data-efficient Image Transformers).

#### 2.1.3 Variants

**ViT-B, ViT-L, ViT-H**: Base, Large, and Huge variants with increasing model dimensions (e.g., ViT-H has 632M parameters with 14x14 patches).

**DeiT (Data-efficient Image Transformers)**: Introduced teacher-student distillation with a CNN teacher (RegNet) to achieve competitive performance without massive pre-training datasets.

**Swin Transformer**: Introduced hierarchical feature maps with shifted windows, enabling efficient multi-scale processing and making transformers practical for dense prediction tasks like segmentation and detection.

**ViT-22B**: Google's 22-billion-parameter ViT, demonstrating continued scaling benefits for vision transformers.

#### 2.1.4 Technical Implementation Details

```python
# Simplified ViT forward pass
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x
```

#### 2.1.5 Strengths and Limitations

**Strengths**:
- Global receptive field from the first layer
- Scales well with compute and data
- Transfers effectively to downstream tasks
- Simpler architecture than equivalent CNNs

**Limitations**:
- Requires large datasets for effective training from scratch
- Quadratic complexity in patch count (O(N²)) for full self-attention
- Less effective on small datasets without strong regularization
- Lacks built-in inductive biases for visual data (translation equivariance, locality)

### 2.2 YOLO (You Only Look Once)

YOLO is a family of real-time object detection systems that frame detection as a single regression problem, directly predicting bounding boxes and class probabilities from full images in one evaluation.

#### 2.2.1 Evolution of YOLO

**YOLOv1 (2015)**: The original paper by Joseph Redmon introduced the unified detection paradigm. The image was divided into an SxS grid, with each cell predicting B bounding boxes and C class probabilities. The loss combined localization, confidence, and classification components. The architecture used 24 convolutional layers followed by 2 fully connected layers.

**YOLOv2 (YOLO9000, 2016)**: Introduced batch normalization, anchor boxes (priors learned via k-means clustering on training data), Darknet-19 backbone, and multi-scale training. The YOLO9000 variant could detect over 9000 object categories by jointly training on detection and classification datasets using WordTree.

**YOLOv3 (2018)**: Featured a deeper Darknet-53 backbone with residual connections (inspired by ResNet), multi-scale predictions at three different feature map resolutions, logistic regression for objectness scoring, and independent logistic classifiers instead of softmax for multi-label classification.

**YOLOv4 (2020)**: A significant engineering achievement by Alexey Bochkovskiy, YOLOv4 incorporated a bag of freebies and specials: Mosaic data augmentation, DropBlock regularization, CIoU loss, Cross-stage partial connections (CSP), Mish activation, and spatial pyramid pooling (SPP). The CSPDarknet53 backbone with PANet neck and YOLO head achieved an excellent speed-accuracy trade-off.

**YOLOv5 (2020)**: Ultralytics' implementation in PyTorch, focusing on usability, deployment, and ecosystem. While not accompanied by a formal paper, YOLOv5 popularized the architecture with focus-based loss, auto-learning anchor boxes, and efficient deployment via ONNX/TensorRT.

**YOLOv6 (2022)**: By Meituan, optimized for industrial deployment with reparameterizable backbones and necks inspired by RepVGG.

**YOLOv7 (2022)**: Introduced trainable bag-of-freebies including planned re-parameterization and coarse-for-fine lead head guidance. Set state-of-the-art for speed and accuracy at the time.

**YOLOv8 (2023)**: Ultralytics' latest series supporting detection, segmentation, classification, and pose estimation in a unified framework. Uses anchor-free detection, C2f (CSP with 2 convolutions and fusing) modules, and decoupled head design.

**YOLO-NAS (2023)**: Deci AI's Neural Architecture Search-optimized YOLO variant, achieving superior accuracy-efficiency trade-offs.

**YOLOv9 (2024)**: Introduced Programmable Gradient Information (PGI) and Generalized Efficient Layer Aggregation Network (GELAN), addressing information loss in deep networks.

**YOLOv10 (2024)**: Introduced consistent dual assignments for NMS-free training, light-weight classification head, spatial-channel decoupled downsampling, and rank-guided block design. Achieved state-of-the-art latency-accuracy trade-offs.

#### 2.2.2 Core Technical Concepts

**Anchor Boxes**: Pre-defined bounding box shapes used as references. YOLO predicts offsets relative to these anchors rather than absolute coordinates. Modern YOLO variants (v8+) have moved toward anchor-free detection.

**Loss Function**: The modern YOLO loss function has three components:

```
L_total = L_box + L_obj + L_cls
```

- **L_box**: CIoU (Complete IoU) loss considering overlap, center distance, and aspect ratio
- **L_obj**: Binary cross-entropy for objectness (whether an object exists)
- **L_cls**: Binary cross-entropy for class predictions (or focal loss for class imbalance)

**Task-Aligned Assigner**: Modern YOLO variants use task-aligned positive sample assignment, selecting positive samples based on a weighted combination of classification score and IoU.

#### 2.2.3 Deployment and Inference Optimization

```python
# YOLOv8 inference example
from ultralytics import YOLO

# Load a pretrained model
model = YOLO('yolov8n.pt')  # nano variant (~3.2M params)

# Run inference
results = model('image.jpg')

# Access detections
for r in results:
    boxes = r.boxes  # Boxes object for bbox outputs
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf[0].item()
        cls = box.cls[0].item()
        print(f"Class: {cls}, Confidence: {conf:.2f}, Box: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
```

YOLO models are optimized for deployment via:
- **TensorRT FP16/INT8 quantization** for NVIDIA GPUs
- **CoreML export** for Apple Silicon
- **ONNX runtime** for cross-platform inference
- **OpenVINO** for Intel hardware
- **NCNN/TNN** for mobile and edge devices

#### 2.2.4 YOLO in Multimodal Systems

YOLO serves as the visual backbone in many multimodal pipelines:
- **Dense captioning**: YOLO detects regions, then a captioning model describes each region
- **Visual question answering**: Object detections provide structured inputs to reasoning modules
- **Video understanding**: Frame-by-frame detection with tracking (e.g., ByteTrack) enables temporal scene graphs
- **Robotics**: Real-time object detection for manipulation and navigation tasks

### 2.3 DETR (DEtection TRansformer)

DETR, introduced by Carion et al. (2020) at Facebook AI, reimagined object detection as a direct set prediction problem using transformers, eliminating the need for many hand-designed components like anchor boxes, non-maximum suppression (NMS), and region proposal networks.

#### 2.3.1 Architecture

DETR consists of three main components:

**CNN Backbone**: A standard convolutional network (typically ResNet-50 or ResNet-101) extracts feature maps from the input image. The final feature map is flattened and combined with positional encodings to form the input sequence for the transformer.

**Transformer Encoder-Decoder**: The encoder processes the image feature sequence using standard multi-head self-attention. The decoder takes a fixed set of N learnable object queries (typically N=100) and processes them via cross-attention with the encoded image features.

**Detection Head**: The decoder outputs are passed through a feed-forward network (FFN) that predicts bounding box coordinates (normalized center x, y, width, height) and class labels. An additional "no object" class (∅) handles cases where no object is present.

#### 2.3.2 Set Prediction Loss

The key innovation in DETR is the bipartite matching loss. The model produces N predictions, and the ground truth has M objects (M < N typically). The Hungarian algorithm finds the optimal assignment between predictions and ground truth that minimizes:

```
L_Hungarian = sum_i [ -log(p_σ(i)(c_i)) + 1_{c_i ≠ ∅} * L_box(b_i, b_σ(i)) ]
```

where `σ` is the optimal permutation, `p_σ(i)(c_i)` is the probability of class `c_i` for prediction `σ(i)`, and `L_box` combines L1 loss and generalized IoU loss.

```python
# Simplified Hungarian matching
from scipy.optimize import linear_sum_assignment

def hungarian_matching(cost_matrix):
    """
    cost_matrix: (num_preds, num_targets)
    Returns optimal assignment indices
    """
    pred_indices, target_indices = linear_sum_assignment(cost_matrix)
    return pred_indices, target_indices
```

#### 2.3.3 Variants and Improvements

**Deformable DETR**: Addresses DETR's slow convergence by using deformable attention, which attends only to a sparse set of key sampling points around a reference point. This reduces the quadratic complexity of attention to O(N * K) where K is the number of sampling points (typically 4-8). Deformable DETR converges 10x faster and performs better on small objects.

**DETR with Conditional DETR**: Decouples the content queries from the spatial queries, using conditional spatial embeddings that are learned from the decoder embeddings. This reduces the need for the cross-attention to simultaneously perform content matching and spatial localization.

**DAB-DETR (Dynamic Anchor Box DETR)**: Treats each object query as a 4D anchor box (x, y, w, h) and updates these anchors iteratively through the decoder layers. The anchor box serves as positional prior, making query-to-feature correspondence explicit.

**DN-DETR (Denoising DETR)**: Adds a denoising training task where the model learns to reconstruct ground-truth boxes from noisy versions, stabilizing the bipartite matching and accelerating convergence.

**DINO (DETR with Improved DeNoising anchOr boxes)**: State-of-the-art DETR variant combining DN-DETR's denoising with DAB-DETR's anchor boxes, contrastive denoising, and a look-forward-twice scheme for box prediction refinement.

**RT-DETR (Real-Time DETR)**: Baidu's real-time DETR variant using efficient hybrid encoder and IoU-aware query selection, achieving competitive speed with YOLO while maintaining DETR's end-to-end elegance.

#### 2.3.4 Strengths and Limitations

**Strengths**:
- Truly end-to-end: no NMS, no anchor boxes, no hand-crafted components
- Consistent pipeline for object detection, segmentation (via DETR variants like Mask2Former)
- Good generalization across domains due to transformer-based reasoning

**Limitations**:
- Slow convergence (especially original DETR)
- Computationally expensive for high-resolution images
- Struggles with small objects (partially addressed by Deformable DETR)
- Fixed number of predictions N (must be larger than maximum expected objects)

#### 2.3.5 DETR in Multimodal Contexts

DETR's set prediction paradigm has been extended to multimodal tasks:

- **MDETR (Modulated DETR)**: Grounds phrases in images by processing text alongside visual features in the transformer, enabling text-conditioned detection
- **GLIP (Grounded Language-Image Pre-training)**: Unifies detection and phrase grounding using a contrastive formulation similar to DETR, scaling detection to thousands of categories via language
- **SpatialVLM**: Uses DETR-like architectures to generate spatial relationship descriptions for 3D understanding

### 2.4 SAM (Segment Anything Model)

SAM, introduced by Kirillov et al. (2023) at Meta AI, represents a foundation model for image segmentation. Inspired by the success of large language models and the "prompting" paradigm, SAM is designed to segment any object in any image given flexible prompts (points, boxes, masks).

#### 2.4.1 Architecture

SAM's architecture is composed of three components:

**Image Encoder**: A MAE (Masked Autoencoder) pre-trained ViT-Huge (632M parameters) that processes the input image at 1024x1024 resolution. The encoder produces a high-dimensional feature embedding that captures the full image context. This embedding can be pre-computed once for the entire image and reused for multiple prompts, enabling real-time interactive segmentation.

**Prompt Encoder**: Handles two types of prompts:
- **Sparse prompts**: Points (positive/negative), boxes, and text. Points and boxes are encoded via positional encodings. Text prompts are encoded using a pretrained text encoder (e.g., CLIP).
- **Dense prompts**: Coarse masks, which are embedded via a convolutional network and combined with the image embedding.

**Mask Decoder**: A lightweight transformer-based decoder that maps the combined image embedding and prompt embeddings to a segmentation mask. The decoder uses a modified transformer block with two-way cross-attention between image features and prompt tokens. The output is a set of three masks corresponding to different granularity levels (whole, part, subpart).

#### 2.4.2 Training Data and Procedure

SAM was trained on the SA-1B dataset, the largest segmentation dataset ever created, containing:
- 11 million images
- 1.1 billion segmentation masks
- 2.1 billion points (average ~200 masks per image)

Masks were generated using an iterative "data engine" process:
1. **Model-assisted manual annotation**: Professional annotators click on objects, and SAM predicts masks. Annotators correct masks iteratively.
2. **Semi-automatic annotation**: SAM is improved and used to propose masks automatically; annotators review and correct.
3. **Fully automatic annotation**: SAM generates masks automatically using a grid of points as prompts, followed by post-processing (NMS, filtering).

#### 2.4.3 Zero-Shot and Prompting Capabilities

SAM demonstrates remarkable zero-shot capabilities:

- **Point prompting**: Click on a point to segment the containing object
- **Box prompting**: Draw a bounding box to segment the primary object inside
- **Mask prompting**: Provide a rough mask for refinement
- **Automatic**: Generate masks for all objects in the image (via grid sampling)
- **Text prompting** (via CLIP integration): "Segment the dog" generates a mask for the dog

```python
# SAM inference example
from segment_anything import sam_model_registry, SamPredictor

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)
predictor.set_image(image)

# Box prompt
masks, scores, logits = predictor.predict(
    box=np.array([100, 150, 600, 700]),
    multimask_output=True
)

# Point prompt (positive click)
masks, scores, logits = predictor.predict(
    point_coords=np.array([[350, 400]]),
    point_labels=np.array([1]),  # 1 = foreground, 0 = background
    multimask_output=True
)
```

#### 2.4.4 SAM 2 (2024)

SAM 2 extends the model to video, enabling:
- **Promptable video segmentation**: User clicks on any object in any frame, and SAM 2 propagates the mask across the entire video
- **Streaming memory**: As each frame is processed, the model's memory about the segmented object is updated via cross-attention with previous frames
- **Occlusion handling**: The model can predict when an object is occluded and re-emerge later
- **Real-time performance**: Capable of processing video at interactive frame rates

The SAM 2 architecture replaces the image encoder with a video backbone that processes frames while maintaining a memory bank of previous frame embeddings, object pointers, and occlusion predictions.

#### 2.4.5 Applications in Multimodal Systems

SAM serves as a versatile visual grounding component:
- **LLaVA-SAM**: Integrates SAM segmentations into LLaVA's visual pipeline for detailed visual reasoning
- **Grounded SAM**: Combines Grounding DINO (for text-to-box detection) with SAM (for box-to-mask segmentation)
- **Semantic SAM**: Extends SAM with semantic class predictions
- **Medical SAM**: Fine-tuned on medical imaging data (CT, MRI, X-ray) for clinical segmentation
- **3D SAM**: Extended to volumetric segmentation in 3D medical data

#### 2.4.6 Limitations

- **No semantic understanding by default**: SAM segments objects but doesn't classify them (though this is addressed by combining with classifiers)
- **Computational cost**: The ViT-H image encoder is heavy (632M parameters), though mobile variants (MobileSAM, EfficientSAM, FastSAM) address this
- **Temporal instability**: In video, masks may flicker between frames (partially addressed in SAM 2)
- **Ambiguous prompts**: Vague prompts (e.g., a single point on a region with multiple overlapping objects) may produce inconsistent results

---

## 3. Image Generation Models

Image generation has undergone a revolution with the advent of diffusion models, which progressively denoise random noise into coherent images conditioned on text prompts.

### 3.1 Stable Diffusion 1/2/3/XL

Stable Diffusion (SD) is a family of open-source text-to-image models developed by Stability AI in collaboration with RunwayML and CompVis LMU Munich.

#### 3.1.1 Stable Diffusion 1 (2022)

**Architecture**: SD is a latent diffusion model (LDM) that operates in the latent space of a pretrained VAE (Variational Autoencoder) rather than directly in pixel space. This significantly reduces computational requirements while maintaining high image quality.

Components:
- **VAE Encoder**: Downsamples images from pixel space (512x512x3) to latent space (64x64x4), a 48x compression rate
- **U-Net Denoiser**: A convolutional U-Net with cross-attention layers for text conditioning. Operating in latent space, the U-Net is ~860M parameters, compared to billions needed for pixel-space diffusion
- **Text Encoder**: CLIP ViT-L/14 text encoder (frozen) that encodes text prompts into 77 tokens of 768-dimensional embeddings
- **Scheduler**: DDPM, DDIM, PLMS (Pseudo-LMS), or Euler-based schedulers for controlled denoising steps

Training objective:
```
L = E_{z_0, ε, t, c} [ || ε - ε_θ(z_t, t, c) ||² ]
```

where `z_0` is the encoded latent, `ε` is the added noise, `t` is the timestep, `c` is the text conditioning, and `ε_θ` is the denoising U-Net.

**Denoising process**:
```python
# Simplified diffusion sampling
def sample(model, text_embedding, steps=50, guidance_scale=7.5):
    # Start from random noise in latent space
    latents = torch.randn(1, 4, 64, 64)
    
    for t in reversed(range(steps)):
        # Classifier-free guidance
        noise_pred_uncond = model(latents, t, empty_embedding)
        noise_pred_text = model(latents, t, text_embedding)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Denoising step (DDIM)
        latents = ddim_step(latents, noise_pred, t)
    
    # Decode latents to pixels
    image = vae.decode(latents)
    return image
```

**Classifier-Free Guidance (CFG)**: A technique that balances diversity and fidelity. During training, the model is conditioned on both text prompts and empty prompts (unconditional) with some probability. At inference, the prediction is extrapolated away from unconditional and toward conditional:

```
ε_θ_guided = ε_θ(z_t, t, ∅) + w * (ε_θ(z_t, t, c) - ε_θ(z_t, t, ∅))
```

where `w` is the guidance scale (typically 7-14). Higher `w` values produce images more aligned with the prompt but can reduce diversity and cause saturation.

#### 3.1.2 Stable Diffusion 2 (2022)

Improvements over SD1:
- **OpenCLIP text encoder** (ViT-H/14, 694M params): Larger and more capable than the original CLIP encoder, resulting in better prompt understanding
- **Resolution upgrade**: Native 768x768 generation (SD 2.0) and 512x512 (SD 2.1)
- **Depth-to-image**: Conditional generation using depth maps via MiDaS (dense prediction transformer)
- **Upscaler**: 4x super-resolution model (x4-upscaler) using latent diffusion
- **Removal of NSFW filter from training data**: Controversially, SD 2 was trained on a filtered subset, affecting stylistic diversity

#### 3.1.3 Stable Diffusion XL (2023)

SDXL represents a significant architectural upgrade:

**Larger U-Net** (2.6B params): Triple the size of SD's U-Net, using a two-stage pipeline:
- **Base model (SDXL)**: Generates 1024x1024 latents
- **Refiner model**: A separate U-Net that processes the base output with a different noise schedule to enhance detail

**Dual text encoders**: 
- OpenCLIP ViT-bigG/14 (frozen)
- CLIP ViT-L/14 (frozen)
This dual-encoder approach captures both fine-grained details and high-level semantics.

**Novel conditioning mechanisms**:
- **Size conditioning**: The model receives the target image dimensions as conditioning, enabling arbitrary aspect ratios
- **Crop conditioning**: Information about how the image was cropped during training allows the model to handle various compositions
- **Pooled text embeddings**: The pooled output of the text encoder provides global style and content guidance alongside per-token embeddings

**Architecture improvements**:
- **Double-stream blocks**: Separate paths for text-conditioned and unconditional processing
- **Logit-normal sampling**: Instead of uniform noise, logit-normal sampling for timestep distribution improves training of both large and small noise levels
- **Expert denoising**: Different time ranges benefit from different architecture capacities

```python
# SDXL sampling with refiner
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import torch

base = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

# Generate with base
image = base(
    prompt="A cinematic shot of a futuristic city at sunset, neon lights",
    num_inference_steps=40,
    denoising_end=0.8,  # Stop early for refiner to work
    output_type="latent"
).images[0]

# Refine
image = refiner(
    prompt="A cinematic shot of a futuristic city at sunset, neon lights",
    image=image[None, :],
    num_inference_steps=40,
    denoising_start=0.8,  # Start from where base stopped
).images[0]
```

#### 3.1.4 Stable Diffusion 3 (2024)

SD3 introduced the MMDiT (Multimodal Diffusion Transformer) architecture, a significant departure from the U-Net structure:

**MMDiT Architecture**:
- Replaces the U-Net with a diffusion transformer (DiT) backbone
- Uses separate sets of parameters for image and text modalities but joint attention
- Rectified flow (RF) training objective instead of standard diffusion
- Improved text rendering capabilities (significantly better at generating legible text)

**Rectified Flow**: A simpler training objective that learns to transform noise to data along straight-line paths:
```
L = E_{x_0, ε, t} [ || (x_0 - ε) - v_θ(x_t, t, c) ||² ]
```
where `x_t = (1-t) * ε + t * x_0` is the linear interpolation between noise and data, and `v_θ` predicts the velocity vector field.

**Scaling**: SD3 comes in multiple sizes:
- SD3-medium: ~2B params
- SD3-large: ~8B params

**Improved text rendering**: SD3 demonstrates markedly better performance on generating text within images, a significant limitation of previous SD versions.

### 3.2 Flux

Flux is a family of text-to-image models developed by Black Forest Labs (founded by former Stability AI researchers including Robin Rombach, the first author of the original Stable Diffusion paper). Flux represents the state of the art in open-source image generation as of mid-2024.

#### 3.2.1 Architecture

Flux uses a novel architecture combining:
- **MMDiT (Multimodal Diffusion Transformer)**: Similar to SD3 but with architectural refinements including QK-normalization and parallel attention layers
- **Rotary Position Embeddings (RoPE)**: Replaces sinusoidal positional encodings for better 2D position understanding
- **Dual text encoders**: T5-XXL (11B params) + CLIP ViT-L, enabling both detailed semantic understanding and aesthetic quality
- **Flow matching training**: Uses rectified flow with a specific noise schedule optimized for image quality

#### 3.2.2 Variants

**Flux.1 Pro**: Full precision, highest quality, access via API
**Flux.1 Dev**: Open-weight (non-commercial), distilled from Pro with guidance distillation
**Flux.1 Schnell**: Open-source (Apache 2.0), 4-step distilled model for rapid generation

#### 3.2.3 Key Capabilities

- **Superior prompt following**: Among the best in class for adhering to complex, multi-attribute prompts
- **Native multi-resolution**: Supports any aspect ratio within reasonable bounds
- **Text rendering**: State-of-the-art text generation in images, rivaling DALL-E 3
- **Anatomical accuracy**: Significantly improved hands, faces, and anatomy compared to SDXL
- **Speed**: Schnell variant generates high-quality images in 4 steps

### 3.3 DALL-E 2/3

DALL-E, developed by OpenAI, represents a closed-source but highly influential line of text-to-image models.

#### 3.3.1 DALL-E 2 (2022)

**Architecture**: DALL-E 2 uses a two-stage approach:

**Prior model**: A diffusion model that generates CLIP image embeddings from text captions. Two prior variants were explored (autoregressive and diffusion), with the diffusion prior performing better. The prior converts text embeddings (from CLIP text encoder) to image embeddings (in CLIP image space).

**Decoder model**: A diffusion model that generates images conditioned on the CLIP image embeddings. This "unCLIP" process reverses CLIP's encoding: given an image embedding, generate the corresponding image.

```
Stage 1 (Prior): text → CLIP text emb → Diffusion Prior → CLIP image emb
Stage 2 (Decoder): CLIP image emb + text → Diffusion Decoder → image
```

**Key innovations**:
- **Text-conditioned prior**: The prior model enables image variations, interpolations, and text-guided edits in CLIP embedding space
- **Image variations**: By passing a CLIP image embedding (from a real image) through the decoder, DALL-E 2 can create diverse variations of the input
- **Inpainting and editing**: The decoder supports masked region editing with text guidance

#### 3.3.2 DALL-E 3 (2023)

DALL-E 3 represents a massive leap in caption following and image-text alignment:

**Improved Captioning**: Rather than training on existing image-text pairs (which are often noisy and underspecified), DALL-E 3 uses an image captioner (likely GPT-4V) to generate detailed, descriptive captions for training images. This closes the "caption gap" — the discrepancy between user prompts and training captions.

**Seamless text rendering**: Significantly improved ability to render legible text in images.

**Safety filtering**: Input prompts are passed through a safety classifier before generation, and the model refuses to generate images in the style of living artists.

**Integration with ChatGPT**: DALL-E 3 is available through ChatGPT, which writes optimized prompts for the model based on natural language user requests.

**No compositional breakdown**: DALL-E 3 dramatically reduces failures where models miss objects or relationships specified in prompts.

### 3.4 Midjourney

Midjourney is a proprietary AI art generation platform (not a model released for download) developed by Midjourney Inc., founded by David Holz (co-founder of Leap Motion).

#### 3.4.1 Versions

**V1-V3 (2022)**: Early versions focused on artistic, painterly styles with strong aesthetic quality but limited prompt adherence.

**V4 (Nov 2022)**: Major architectural overhaul with improved:
- Anatomical coherence (especially hands and faces)
- Wider variety of styles (not just painterly)
- Better aspect ratio handling
- Image prompting and style references

**V5 (Mar 2023)**: Higher resolution (native 1024x1024 with upscaling), improved:
- Photo-realism approaching photographic quality
- Better prompt understanding of fine details
- Improved textures and surface rendering
- Removed restrictions on certain subjects

**V6 (Dec 2023)**: Complete model re-architecture:
- Significantly improved prompt following
- Better understanding of prepositions, grammar, and spatial relationships
- Coherent text rendering (rudimentary)
- Improved upscalers
- Character consistency features

**V6.1 (Jul 2024)**: Further improved coherence, detail rendering, and personalization features.

#### 3.4.2 Key Features

**Discord-based interface**: Users interact via Discord, using `/imagine` commands. The platform recently added a web interface.

**Styling options**: 
- `--style raw`: Reduces Midjourney's artistic interpretations for more literal prompt following
- `--stylize <value>` (0-1000): Controls the degree of artistic interpretation
- `--v <version>`: Selects the model version

**Parameters**:
- `--ar <w:h>`: Aspect ratio (e.g., 16:9, 9:16, 1:1)
- `--chaos <0-100>`: Variation in results
- `--quality <.25-2>`: Rendering quality/time trade-off
- `--seed <int>`: Reproducible results
- `--iw <0-2>`: Image prompt weight relative to text

**Advanced features**:
- **Image prompting**: Provide reference images for composition and style
- **Multi-prompts**: Use `::` to separate concepts for precise weighting (e.g., `lake::2 mountain::1`)
- **Remix mode**: Modify prompts after generation to iterate on results
- **Pan/zoom**: Extend images in any direction
- **Blend**: Combine multiple images
- **Style reference (`--sref`)**: Reference images for style transfer
- **Character reference (`--cref`)**: Maintain character consistency across generations

#### 3.4.3 Technical Approach (Inferred)

Midjourney's exact architecture is proprietary, but based on available information and public research:
- Likely based on diffusion models (potentially latent diffusion or similar)
- Strong emphasis on aesthetic quality via curation, human preference feedback, and fine-tuning
- Multiple specialized models for different aspects (composition, lighting, texture, upscaling)
- Iterative refinement process that combines generation with upscaling and detail enhancement

---

## 4. Video Generation Models

Video generation extends image generation to the temporal domain, requiring models to maintain consistency across frames while capturing motion and dynamics.

### 4.1 Sora

Sora, announced by OpenAI in February 2024, represents a breakthrough in text-to-video generation. As of 2024, Sora is not publicly released but detailed technical reports and demonstrations have been published.

#### 4.1.1 Architecture

Sora is a diffusion transformer (DiT) operating on video patches:

**Visual Patches**: Similar to how LLMs process text tokens, Sora processes videos as collections of spatiotemporal patches. Videos are decomposed into patches of variable duration, resolution, and aspect ratio.

**Compression Network**: A video compressor network (likely a VAE or VQ-VAE) maps raw video pixels to a lower-dimensional latent space. The compressor handles both spatial (H x W) and temporal (T) dimensions.

**Transformer Backbone**: A scalable diffusion transformer processes the patch sequence. The architecture supports:
- Variable-length sequences (videos of varying duration)
- Variable-resolution (training and inference at native resolution)
- Variable-aspect-ratio (native support without cropping)

**Latent → Pixel Decoder**: A decoder network (potentially learned simultaneously with the compressor) maps denoised latents back to pixel space.

#### 4.1.2 Key Capabilities

**Duration and resolution flexibility**: Sora can generate videos from 1 second to 1 minute at resolutions up to 1920x1080.

**Native multi-aspect ratio**: Unlike models that require square or fixed-ratio inputs, Sora handles 16:9, 9:16, 1:1, and other aspect ratios natively.

**Consistent physics**: Sora demonstrates understanding of object permanence, gravity, fluid dynamics, and rigid body motion. However, it can still produce physical implausibilities.

**3D consistency**: Objects maintain consistent 3D structure as the camera moves, suggesting internal 3D understanding.

**Emergent simulation**: When trained at scale, Sora exhibits emergent capabilities including:
- Camera motion (pan, zoom, orbit)
- Object interactions (cutting, pouring, bouncing)
- Style transfer (photorealistic, animated, cinematic)
- Video-to-video editing (changing style, extending videos forward/backward)

**Long-range coherence**: Can maintain character and scene consistency for the duration of the generated video.

#### 4.1.3 Relationship to Previous Models

Sora builds on the DiT (Diffusion Transformer) architecture from Peebles & Xie (2023) and applies it to video by:
- Extending from 2D (image) to 3D (video) patchification
- Using a video-native compression network instead of image VAE
- Scaling the transformer to handle the significantly longer sequences inherent in video

#### 4.1.4 Limitations

- **Physics imperfections**: Objects can spontaneously appear/disappear, rigid bodies can behave incorrectly
- **Spatial detail**: Fine details (text, faces) can be inconsistent across frames
- **Motion magnitude**: Very fast or complex motions can produce artifacts
- **Causality violations**: Objects' states can change without cause (e.g., a sandwich that appears half-eaten at the start)
- **Accessibility**: As of late 2024, Sora has limited public availability

### 4.2 VideoPoet

VideoPoet, developed by Google Research (announced December 2023), takes a different approach from Sora by using an autoregressive language model architecture rather than diffusion.

#### 4.2.1 Architecture

VideoPoet is a decoder-only large language model that processes video, image, audio, and text as tokens:

**Tokenizer**: A video tokenizer (MAGVIT-v2) converts video frames into discrete tokens using a vector-quantized variational autoencoder (VQ-VAE). This tokenizer handles:
- Variable-length videos
- Multiple modalities (video, image, audio)
- High compression ratios while maintaining quality

**LLM Backbone**: A causal transformer with billions of parameters trained on the token sequences. The autoregressive next-token prediction paradigm is applied to multimodal token sequences.

**Super-resolution**: A separate model (likely diffusion-based) upsamples the low-resolution token outputs to high-resolution video.

#### 4.2.2 Capabilities

- **Text-to-video**: Generate video from text descriptions
- **Image-to-video**: Animate a static image
- **Video-to-video**: Style transfer, inpainting, outpainting
- **Audio generation**: Can generate audio synchronized with video (though primarily focused on visual)
- **Video editing**: Extend, trim, or modify existing videos
- **Camera motion control**: Supports specified camera movements

#### 4.2.3 Autoregressive vs. Diffusion for Video

| Aspect | VideoPoet (Autoregressive) | Sora (Diffusion) |
|--------|---------------------------|-------------------|
| Generation | Token-by-token (autoregressive) | Iterative denoising |
| Speed | Fast at inference (one forward pass) | Requires multiple steps (but can be distilled) |
| Global coherence | Local focus can miss long-range structure | Iterative refinement improves global consistency |
| Training | Standard next-token prediction (scalable) | Diffusion loss (more complex training) |
| Tokenization | Requires VQ-VAE (information loss) | Operates on continuous latents (less lossy) |
| Control | Natural for conditional generation | Requires conditioning mechanisms |

VideoPoet demonstrates that LLM architectures can be extended to video generation, challenging the dominance of diffusion models in visual generation.

---

## 5. Audio Models

Audio processing and generation has advanced significantly with deep learning, enabling high-quality speech recognition, music generation, and sound synthesis.

### 5.1 Whisper

Whisper is OpenAI's general-purpose speech recognition model, trained on 680,000 hours of multilingual and multitask supervised data.

#### 5.1.1 Architecture

Whisper uses an encoder-decoder transformer architecture:

**Encoder**: Processes log-Mel spectrograms (80-channel, 25ms window, 10ms stride) through convolutional layers (2 layers with GELU activation) followed by transformer encoder blocks.

**Decoder**: An autoregressive transformer decoder that generates text tokens conditioned on the encoder output. The decoder supports multiple tasks controlled via special tokens:
- Speech-to-text transcription
- Language identification
- Timestamp prediction (segment-level)
- Translation (to English)

**Training objective**: Cross-entropy loss on text tokens with teacher forcing during training.

#### 5.1.2 Model Variants

| Model | Parameters | Multilingual | Relative Speed |
|-------|-----------|--------------|----------------|
| tiny | 39M | Yes | ~32x |
| base | 74M | Yes | ~16x |
| small | 244M | Yes | ~6x |
| medium | 769M | Yes | ~2x |
| large | 1,550M | Yes | 1x |
| large-v2 | 1,550M | Yes | 1x |
| large-v3 | 1,550M | Yes | 1x |

**large-v3**: Trained with improved training methodology including:
- 2x more data than large-v2
- Improved language coverage (99 languages)
- Reduced WER across most languages
- Better handling of numbers, punctuation, and formatting

#### 5.1.3 Usage

```python
import whisper

model = whisper.load_model("large-v3")

# Transcribe
result = model.transcribe("speech.mp3", language="en")
print(result["text"])

# With word-level timestamps
result = model.transcribe("speech.mp3", word_timestamps=True)
for segment in result["segments"]:
    for word in segment["words"]:
        print(f"{word['word']}: {word['start']:.2f}s - {word['end']:.2f}s")

# Translate to English
result = model.transcribe("french_speech.mp3", task="translate")
print(result["text"])
```

#### 5.1.4 Strengths and Limitations

**Strengths**:
- Robust across diverse acoustic conditions (noise, accents, reverberation)
- Strong multilingual performance (99 languages)
- Handles multiple tasks in a single unified model
- No need for language-specific fine-tuning

**Limitations**:
- Computationally expensive (large model requires ~5GB VRAM for inference)
- No streaming support natively (processes entire audio)
- Struggles with very long audio (>30s segments processed independently)
- Idiosyncratic formatting (e.g., unusual word segmentation)

### 5.2 MusicGen

MusicGen, developed by Meta AI (2023), is a text-to-music generation model capable of producing high-quality music samples from textual descriptions.

#### 5.2.1 Architecture

**EnCodec tokenizer**: Meta's neural audio codec that compresses audio into discrete tokens at multiple bitrates. MusicGen uses EnCodec at 32kHz with a 50Hz frame rate (50 tokens per second).

**Hierarchical generation**: MusicGen uses a single autoregressive transformer that predicts all codebook layers simultaneously (unlike earlier models that predicted them sequentially). This is achieved through:
- **Codebook interleaving patterns**: The model predicts multiple codebook levels at each timestep, using different patterns
  - **Delay pattern**: Each subsequent codebook is predicted with a 1-step delay, maintaining causality
  - **Flattening pattern**: All codebooks are predicted as separate token dimensions

**Conditioning**: The transformer is conditioned on text embeddings from a frozen text encoder (T5-based) and optionally on a melody condition (chromagram features from an audio input).

#### 5.2.2 Model Variants

| Model | Params | Description |
|-------|--------|-------------|
| small | 300M | Fast generation, good quality |
| medium | 1.5B | Best quality for most uses |
| large | 3.3B | Highest quality, slower |
| melody | 1.5B | Supports melody conditioning |

```python
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained("large")
model.set_generation_params(
    duration=8,  # seconds
    temperature=1.0,
    top_k=250,
    top_p=0.0,
)

wav = model.generate([
    "80s synthwave track with heavy bass and energetic drums",
    "lo-fi hip hop beat for studying, gentle piano and vinyl crackle",
])
for idx, one_wav in enumerate(wav):
    audio_write(f"output_{idx}", one_wav.cpu(), model.sample_rate)
```

#### 5.2.3 Melody Conditioning

MusicGen's melody model can be conditioned on an existing audio snippet to generate music that follows the same melodic contour:

```python
# Melody conditioning
model = MusicGen.get_pretrained("melody")
model.set_generation_params(duration=12)

melody_wav, sr = audio_read("melody_reference.wav")
wav = model.generate_with_chroma(
    ["a cheerful acoustic guitar arrangement"],
    melody_wav[None].expand(1, -1, -1),
    sample_rate=sr,
)
```

### 5.3 AudioCraft

AudioCraft is Meta AI's unified framework for audio generation, encompassing three models: MusicGen (music), AudioGen (sound effects), and EnCodec (compression).

#### 5.3.1 AudioGen

AudioGen generates general sound effects (not just music) from text descriptions:

**Training data**: 20,000 hours of sound effect data (soundtracks, foley, ambient sounds, animal sounds, etc.)

**Architecture**: Similar to MusicGen but trained on diverse sound effect data rather than primarily musical data. Uses the same EnCodec tokenizer and autoregressive transformer architecture.

**Capabilities**:
- Generate sound effects from text: `dog barking in empty room`, `rain on a tin roof`, `footsteps on gravel`
- Conditional generation: extend or interpolate between sounds
- Multi-channel generation: can generate stereo or ambisonic audio

```python
from audiocraft.models import AudioGen

model = AudioGen.get_pretrained("facebook/audiogen-medium")
model.set_generation_params(duration=5)

wav = model.generate([
    "thunderstorm with heavy rain",
    "coffee shop ambient noise with light chatter",
])
```

#### 5.3.2 EnCodec

EnCodec is Meta's neural audio compression model, serving as the tokenization backbone for AudioCraft:

**Architecture**: A streaming encoder-decoder with a residual vector quantizer (RVQ) bottleneck.

**Key features**:
- Multiple bitrates: 1.5, 3, 6, 12, 24 kbps
- Streaming inference: processes audio in chunks with stateful LSTM layers
- Bandwidth scalability: single model handles multiple compression levels
- Perceptual loss: uses discriminators trained on audio quality to improve perceptual quality

**Technical details**:
- 1D convolutional encoder with strides for downsampling
- LSTM for temporal modeling
- RVQ with 32 codebooks of 1024 entries each at higher bitrates
- Decoder mirrors encoder with transposed convolutions

**EnCodec V2 (used in MusicGen)**:
- Improved quality at low bitrates
- Better handling of music signals
- Higher temporal resolution (50 Hz frame rate vs. 75 Hz in original)

---

## 6. Multimodal LLMs

Multimodal Large Language Models (MLLMs) extend LLMs to process and reason about visual, audio, and other modalities alongside text.

### 6.1 CLIP (Contrastive Language-Image Pre-training)

CLIP, introduced by OpenAI in 2021, is a foundational multimodal model that learns aligned image and text representations.

#### 6.1.1 Architecture

CLIP consists of two encoders trained jointly via contrastive learning:

**Image Encoder**: Either a ResNet (RN50, RN101, RN50x4, RN50x16, RN50x64) or a ViT (ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px). The largest publicly available CLIP model uses ViT-L/14@336px.

**Text Encoder**: A transformer (similar to GPT-2) with 8 layers, 512-dimensional hidden states, and 512-dimensional embeddings. The input text is tokenized into 77 tokens (truncated/padded), with [SOS] and [EOS] tokens. The final text embedding is the projection of the [EOS] token.

**Contrastive Training**: Both encoders project their outputs into a shared embedding space via learned projection matrices. The training objective maximizes cosine similarity between matched image-text pairs while minimizing similarity for unmatched pairs.

```
sim(I, T) = cos(proj_I(enc_I(I)), proj_T(enc_T(T)))
```

#### 6.1.2 Zero-Shot Transfer

CLIP enables zero-shot image classification by embedding candidate class names (e.g., "a photo of a dog", "a photo of a cat") and comparing them with image embeddings:

```python
import clip
import torch

model, preprocess = clip.load("ViT-L/14")

image = preprocess(Image.open("photo.jpg")).unsqueeze(0)
text = clip.tokenize(["a photo of a dog", "a photo of a cat"])

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    logits_per_image = image_features @ text_features.T
    probs = logits_per_image.softmax(dim=-1)
```

#### 6.1.3 Model Variants

| Model | Backbone | Params (M) | Resolution | Top-1 ImageNet Zero-Shot |
|-------|---------|------------|------------|--------------------------|
| RN50 | ResNet-50 | 76 | 224 | 60.4% |
| ViT-B/32 | ViT-B/32 | 150 | 224 | 63.2% |
| ViT-B/16 | ViT-B/16 | 150 | 224 | 68.3% |
| ViT-L/14 | ViT-L/14 | 428 | 224 | 75.5% |
| ViT-L/14@336px | ViT-L/14 | 428 | 336 | 76.2% |

#### 6.1.4 CLIP's Role in Multimodal Systems

CLIP has become a ubiquitous component in multimodal AI:
- **Image generation conditioning**: Used in DALL-E 2, Stable Diffusion, and many other generators
- **Visual understanding**: Paired with LLMs in models like LLaVA and GPT-4V
- **Zero-shot detection**: Models like ViLD (Vision Language Detector) use CLIP for open-vocabulary object detection
- **Video understanding**: CLIP embeddings extended to video via VideoCLIP, CLIP4Clip
- **Multi-modal search**: CLIP enables text-to-image and image-to-text retrieval

**OpenCLIP**: An open-source reimplementation by LAION that reproduced and extended CLIP's training. Key variants:
- ViT-g/14 (LAION-2B): 1.4B params, trained on 2B image-text pairs
- ViT-H/14 (LAION-2B): 986M params
- ViT-bigG/14 (LAION-2B): 2.5B params, used in SDXL

**SigLIP**: Google's variant using sigmoid loss instead of softmax contrastive loss, enabling better performance with smaller batch sizes.

### 6.2 LLaVA (Large Language and Vision Assistant)

LLaVA, introduced by Haotian Liu et al. at UW-Madison (2023), is a simple yet effective multimodal LLM that connects a vision encoder (CLIP) with a language model (LLaMA/Vicuna) via a lightweight projection module.

#### 6.2.1 Architecture

LLaVA's architecture consists of:
- **Vision Encoder**: CLIP ViT-L/14 (frozen during initial training, later fine-tuned)
- **Projection Module**: A simple linear projection (or 2-layer MLP in LLaVA-1.5) that maps visual tokens to the LLM's embedding space
- **Language Model**: LLaMA (original), Vicuna (LLaVA-1.5), or LLaMA-3 (LLaVA-NeXT)

The total parameter count is approximately 13B-34B depending on the LLM backbone.

#### 6.2.2 Training Procedure

LLaVA uses a two-stage training procedure:

**Stage 1: Vision-Language Alignment (Pre-training)**:
- Freeze vision encoder and LLM
- Train only the projection layer
- Use CC3M (Conceptual Captions) dataset for image-caption pairs
- Objective: standard language modeling (next-token prediction) on captions

**Stage 2: Visual Instruction Tuning**:
- Fine-tune projection layer and LLM (optionally also the vision encoder)
- Use GPT-4-generated instruction-following data (LLaVA-Instruct-150K)
- Data includes: conversations, detailed descriptions, complex reasoning

#### 6.2.3 Data Generation Pipeline

LLaVA's training data was generated by using GPT-4 on COCO images with carefully designed prompts:

**Conversation data**: Free-form conversations about images
```
User: Describe this image in detail.
Assistant: A group of people are playing soccer on a grassy field...
```

**Detailed description**: Systematic scene understanding
```
User: What objects can you identify in this image?
Assistant: There are two dogs, a frisbee, a person throwing the frisbee...
```

**Complex reasoning**: Multi-step reasoning about images
```
User: Why might the person be wearing a raincoat?
Assistant: The sky appears overcast with dark clouds, suggesting rain...
```

#### 6.2.4 LLaVA Variants

**LLaVA-1.5 (2024)**:
- CLIP ViT-L/14@336px vision encoder
- MLP projection (2 layers instead of 1)
- Vicuna 1.5 7B/13B LLM
- Academic VQA datasets for training (in addition to LLaVA-Instruct)
- Visual question answering fine-tuning on VizWiz, GQA, OCR-VQA, etc.
- State-of-the-art on 11 out of 12 multimodal benchmarks at release

**LLaVA-NeXT (2024)**:
- Improved training recipes
- Support for higher resolution (any resolution via anyres)
- Support for Llama-3 backbone
- Improved OCR and document understanding
- LLaVA-NeXT-Interleave: Support for interleaved image-text data for multi-image reasoning

**LLaVA-NeXT-Video**: Extension to video understanding via frame sampling and temporal reasoning.

```python
# LLaVA inference example
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, default_image_processor

model_path = "liuhaotian/llava-v1.5-13b"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)

# Prepare image
image = Image.open("image.jpg")
image_tensor = process_images([image], image_processor, model.config)

# Prepare prompt
prompt = "USER: <image>\nDescribe what's happening in this image.\nASSISTANT:"
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

# Generate
with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        do_sample=True,
        temperature=0.2,
        max_new_tokens=1024,
    )
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
```

### 6.3 GPT-4V / GPT-4o

GPT-4V (GPT-4 Vision) and the later GPT-4o ("omni") represent OpenAI's multimodal frontier models, integrating vision, audio, and text understanding.

#### 6.3.1 GPT-4V (September 2023)

GPT-4V extended GPT-4 with image understanding capabilities:

**Capabilities**:
- Image captioning and detailed description
- Visual question answering
- Diagram and chart interpretation
- Handwriting recognition
- Spatial reasoning
- Multi-image comparison and analysis
- Text extraction from images (OCR)

**Input modalities**: Text + images (single or multiple images per conversation)
**Output modality**: Text only

**Technical approach** (inferred from public information):
- Uses a vision encoder (likely CLIP-like) to extract visual features
- Visual tokens are interleaved with text tokens in the transformer's context window
- The model is trained end-to-end on both vision-language and text-only data
- Training likely involves multiple stages: vision-language alignment, instruction tuning, and RLHF

**Limitations**:
- Cannot process video natively (only individual frames)
- Limited accuracy on fine-grained detail in very high-resolution images
- Can be confused by complex spatial relationships
- May struggle with specialized visual domains (medical imaging, satellite imagery)

#### 6.3.2 GPT-4o (May 2024)

GPT-4o ("omni" from Latin "omnis", meaning "all" or "every") is OpenAI's natively multimodal model that can process and generate text, images, and audio in a single unified architecture.

**Key architectural innovations**:
- **Single network for all modalities**: Unlike GPT-4V which might use separate encoders fused with text, GPT-4o is reportedly a single end-to-end multimodal model
- **Native audio processing**: The model can understand and generate audio without cascading through speech-to-text and text-to-speech pipelines
- **Real-time interaction**: Response latency reduced to ~232ms for audio (comparable to human conversation)
- **Emotion and tone understanding**: Can perceive and express emotion, tone, and non-verbal cues in audio

**Capabilities**:
- Text + image + audio input → text output
- Text + image + audio input → audio output (with natural speech, emotion, intonation)
- Real-time conversational AI with visual understanding
- Audio event recognition (e.g., barking, music in the background)
- Voice tone and sentiment analysis

**Performance**:
- GPT-4o matches or exceeds GPT-4 Turbo on text and vision benchmarks
- Significantly better on multilingual benchmarks
- Superior audio understanding compared to cascaded systems (Whisper → GPT-4 → TTS)
- Improved performance on non-English languages

**API access**:
```python
from openai import OpenAI
import base64

client = OpenAI()

# Image understanding
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(open("photo.png", "rb").read()).decode()
                }}
            ]
        }
    ]
)

# Audio input (via transcription then understanding)
audio_file = open("speech.mp3", "rb")
transcript = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file
)
```

### 6.4 Gemini Vision

Gemini, developed by Google DeepMind, is a multimodal model family designed from the ground up to work across text, image, audio, video, and code.

#### 6.4.1 Architecture

Gemini's architecture is based on a Transformer decoder with multimodal capabilities integrated at the architectural level:

**Key architectural features**:
- **Native multimodal training from the start**: Unlike approaches that add vision to an existing language model, Gemini was trained jointly on text, images, audio, and video from the beginning
- **Visual encoding**: Employs a vision encoder (likely ViT-based) that outputs visual tokens processed alongside text tokens
- **Variable input resolution**: Gemini Ultra natively handles inputs up to 10M tokens (theoretically), with 200K context window in production
- **Multi-query attention**: Similar to PaLM, for efficient inference

#### 6.4.2 Model Variants

**Gemini Ultra**: The largest model, designed for complex reasoning tasks. Outperforms GPT-4 on MMLU (90.0% vs 86.4%), including text, vision, and multimodal benchmarks.

**Gemini Pro**: Balanced performance and efficiency, comparable to GPT-4 in many benchmarks while being more cost-effective.

**Gemini Nano**: Models for on-device deployment, available in 1.8B and 3.25B parameter variants. Designed for Pixel and other mobile devices. Uses:
- Int8 quantization for efficient inference
- Custom kernel optimizations for mobile hardware
- Selective generation and speculative decoding for speed

#### 6.4.3 Multimodal Capabilities

**Image understanding**:
- Scene understanding, object detection, OCR
- Diagram and chart interpretation
- Information extraction from documents
- Image-to-code (generate HTML/CSS from design mockups)

**Audio understanding**:
- Native audio processing without separate ASR
- Speech transcription and translation
- Audio event detection (music, animal sounds, environmental audio)
- Sentiment analysis from speech tone

**Video understanding**:
- Frame-by-frame analysis or temporal analysis
- Action recognition and activity understanding
- Video summarization
- Question answering about video content

**Code generation**:
- Multi-language code generation
- Code explanation and documentation
- Code translation between languages

#### 6.4.4 Training Infrastructure

Gemini Ultra was trained on TPUv5e and TPUv5p accelerators in large pods:
- Trained across thousands of TPUs
- Used the multiconnect and Pathways systems for efficient distributed training
- Training data includes web documents, books, code, images, audio, and video

### 6.5 Qwen-VL

Qwen-VL is Alibaba Cloud's multimodal LLM, part of the Qwen (通义千问) model family.

#### 6.5.1 Architecture

Qwen-VL follows the LLaVA-like paradigm with several innovations:

**Components**:
- **Vision Encoder**: OpenCLIP ViT-bigG/14 (with frozen parameters initially)
- **Vision-Language Adapter**: A single-layer cross-attention module that compresses visual tokens from the encoder into a fixed-length set of learnable queries (256 visual tokens), significantly reducing the sequence length
- **Language Model**: Qwen-7B (fine-tuned with vision-language data)

**Key architectural features**:
- **Position awareness**: The vision encoder incorporates 2D positional encoding to better handle spatial relationships
- **Dynamic resolution**: Supports multiple resolution levels (224, 448, 672, etc.) via patch interpolation
- **Multi-image input**: Can process multiple images simultaneously and reason about relationships between them

#### 6.5.2 Capabilities

- **Visual question answering**: Detailed scene understanding and reasoning
- **OCR and document understanding**: Strong performance on Chinese and English text in images
- **Object recognition**: Fine-grained object identification and localization
- **Multi-image reasoning**: Compare and contrast multiple images
- **Grounding**: Can reference specific objects in images (via bounding box coordinates)

#### 6.5.3 Qwen-VL Variants

**Qwen-VL**: Base model with 7B LLM backbone, strong multimodal capabilities
**Qwen-VL-Chat**: Chat-optimized version with instruction tuning and safety alignment
**Qwen-VL-Plus** (API): Enhanced version with better detail capture and OCR
**Qwen-VL-Max** (API): Largest variant with state-of-the-art performance on multimodal benchmarks

### 6.6 InternVL

InternVL, developed by Shanghai AI Laboratory and OpenGVLab, is a large-scale multimodal model designed to scale up the vision foundation model to match the scale of the language model.

#### 6.6.1 Architecture

InternVL challenges the assumption that the vision encoder must be small relative to the LLM:

**Vision Encoder**: InternViT-6B, a 6-billion-parameter vision transformer trained on massive multimodal data. This is significantly larger than CLIP ViT-L (428M) or ViT-g (1.4B).

**Language Model**: InternLM2-20B, a 20B parameter LLM.

**MLP Projector**: A 3-layer MLP connecting the vision encoder to the LLM.

**Dynamic High Resolution**: Supports resolutions up to 4K by dynamically tiling input images and processing patches through the vision encoder.

#### 6.6.2 Key Innovations

**Progressive Alignment Training Strategy**:
1. **Stage 1**: Train a giant ViT using image-text contrastive learning (similar to CLIP, but at 6B scale)
2. **Stage 2**: Add MLP projector and train on image-text data to align visual tokens with LLM embedding space
3. **Stage 3**: Fine-tune LLM and projector (freeze ViT) on visual instruction data
4. **Stage 4**: Jointly fine-tune all components with a small learning rate

**MMPU Benchmark**: InternVL also contributed MMPU (Multi-Modal Puzzle Understanding), a benchmark for complex multimodal reasoning requiring spatial, temporal, and logical reasoning.

#### 6.6.3 Performance

InternVL-6B (vision encoder) + InternLM2-20B achieves state-of-the-art results on:
- MMBench: 83.5%
- MMMU (val): 63.6%
- MathVista: 63.7%
- ChartQA: 85.1%
- DocVQA: 93.2%

The model demonstrates that scaling vision encoders to match language model scale provides significant benefits for fine-grained visual understanding tasks.

---

## 7. Speech and TTS Models

Text-to-speech (TTS) and speech synthesis models convert text to natural-sounding speech, with modern models capable of capturing prosody, emotion, and speaker characteristics.

### 7.1 Bark

Bark is Suno AI's transformer-based text-to-audio model capable of generating speech, music, and non-verbal sounds from text prompts.

#### 7.1.1 Architecture

Bark uses a GPT-style autoregressive architecture with multiple specialized transformer models:

**Text Encoder**: A GPT-2 style model that processes text tokens and produces semantic representations.

**Coarse Acoustic Model**: Predicts coarse audio tokens (EnCodec codes, first 2 codebooks) from the semantic representations.

**Fine Acoustic Model**: Predicts fine audio tokens (remaining EnCodec codebooks) conditioned on the coarse tokens.

**Audio Decoder**: EnCodec decoder that converts the full set of audio tokens back to waveform.

```
text → [Text Encoder] → [Coarse Acoustic Model] → [Fine Acoustic Model] → [EnCodec Decoder] → audio
```

#### 7.1.2 Capabilities

- **Expressive speech**: Generates speech with natural intonation, emphasis, and pacing
- **Non-verbal sounds**: Can generate laughter, sighs, crying, whispers, and other vocalizations
- **Music generation**: Limited but functional music generation from text
- **Speaker customization**: Supports voice cloning via speaker prompts
- **Multi-language**: Supports English, German, French, Spanish, Italian, Chinese, Japanese, and Korean (with varying quality)

#### 7.1.3 Usage

```python
from bark import SAMPLE_RATE, generate_audio, preload_models

# Download and load all models
preload_models()

# Generate speech
text = "Hello, this is a demonstration of the Bark text-to-speech model."
audio_array = generate_audio(text, history_prompt="v2/en_speaker_6")

# With non-verbal cues
text = "Hello...[laughter]...this is great!"
audio_array = generate_audio(text)

# Voice cloning
from bark import clone_voice
voice_prompt = clone_voice("reference_speech.wav")
audio_array = generate_audio("This sounds like the speaker from the audio file.", voice_prompt)
```

#### 7.1.4 Strengths and Limitations

**Strengths**:
- Highly expressive with natural prosody
- Supports non-verbal sounds embedded in text
- Voice cloning capability
- Open-source (MIT license)

**Limitations**:
- Relatively slow (full generation pipeline)
- Quality varies by language (English best supported)
- Limited voice consistency over long generations
- ~12s maximum generation per call (due to context window)

### 7.2 XTTS (Coqui TTS)

XTTS is Coqui AI's text-to-speech system focusing on voice cloning and cross-lingual transfer.

#### 7.2.1 Architecture

XTTS uses a GPT-style architecture with multiple components:

**Speaker Encoder**: A model that extracts a speaker embedding from a reference audio sample. This embedding captures voice characteristics, accent, and prosody patterns.

**Text-to-Semantic Model**: Predicts semantic tokens (from HuBERT or similar speech representation model) autoregressively from text, conditioned on the speaker embedding.

**Semantic-to-Acoustic Model**: Converts semantic tokens to acoustic features (Mel spectrograms or EnCodec tokens).

**Vocoder/Decoder**: Converts acoustic features to waveform (e.g., HiFi-GAN, WaveGlow).

#### 7.2.2 Key Features

- **Cross-lingual voice cloning**: Clone a voice from one language and synthesize speech in another language
- **Voice cloning from short samples**: Works with as little as 3 seconds of reference audio
- **Multi-lingual**: Supports 17+ languages including English, Chinese, French, German, Spanish, Italian, Portuguese, Polish, Turkish, Russian, Dutch, Czech, Arabic, Japanese, Hungarian, Korean, and Hindi
- **Emotion and style control**: Can adjust speaking rate, pitch, and emphasis

```python
# XTTS inference
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Load model
model = Xtts.init_from_config(XttsConfig())
model.load_checkpoint(config, checkpoint_dir="xtts/")
model.cuda()

# Voice cloning
speaker_embedding = model.get_speaker_embedding("reference.wav")

# Generate speech in different language
output = model.inference(
    text="Bonjour, je parle français avec une voix anglaise.",
    language="fr",
    speaker_embedding=speaker_embedding,
    speed=1.0,
)

# Save audio
from scipy.io.wavfile import write
write("output.wav", 24000, output["wav"])
```

### 7.3 CosyVoice

CosyVoice is Alibaba's voice generation model with advanced voice cloning and emotion control capabilities.

#### 7.3.1 Architecture

CosyVoice uses a novel architecture combining:
- **Flow Matching**: Instead of autoregressive generation, CosyVoice uses flow matching for robust and fast generation
- **LLM-based text encoder**: Leverages large language model capabilities for better text understanding and prosody prediction
- **Speaker-conditional normalization**: Speaker identity is injected at multiple levels of the network for consistent voice cloning

#### 7.3.2 Key Capabilities

- **Zero-shot voice cloning**: Clone voices from a single short utterance
- **Cross-lingual synthesis**: Generate speech in different languages while preserving the cloned voice characteristics
- **Emotion control**: Controllable emotional expression (happy, sad, angry, surprised, neutral)
- **Speaking style transfer**: Transfer prosodic style between speakers
- **Instruction-based generation**: Can follow complex text instructions about how to speak

#### 7.3.3 Technical Innovations

**In-Context Learning (ICL)**: CosyVoice treats voice cloning as an in-context learning problem. By providing a reference audio as context, the model adapts its generation to match the voice characteristics without explicit speaker embedding extraction.

**Supervised Semantic Tokens (SST)**: A novel speech tokenization that preserves semantic content while being speaker-agnostic, enabling better cross-speaker and cross-lingual transfer.

### 7.4 ChatTTS

ChatTTS is a text-to-speech model specifically designed for conversational applications, optimized for natural dialogue generation.

#### 7.4.1 Architecture

ChatTTS uses a VAE + LLM architecture:

**VAE Encoder**: Compresses speech into latent tokens
**LLM Decoder**: Generates speech tokens autoregressively from text
**VAE Decoder**: Converts tokens back to waveform

#### 7.4.2 Features

- **Conversational prosody**: Specifically optimized for natural-sounding conversations with proper pauses, intonation, and emphasis
- **Multi-speaker support**: Can generate dialogues with multiple distinct voices
- **Fine-grained control**: Supports laugh, pause, and other paralinguistic elements via special tokens
- **Real-time capable**: Can generate speech faster than real-time on modern GPUs
- **Open-source**: Released under Apache 2.0 license

```python
# ChatTTS inference
import ChatTTS

chat = ChatTTS.Chat()
chat.load_models()

# Generate conversational speech
texts = ["Hello! How are you doing today?", "I'm doing great, thanks for asking!"]
params = {
    "temperature": 0.3,
    "top_P": 0.7,
    "top_K": 20,
    "speed": 1.0,
}

for text in texts:
    wav = chat.infer(text, params)
    audio_write(f"response.wav", wav, 24000)
```

---

## 8. Music Generation

Music generation models synthesize musical audio from text descriptions, style references, or other conditioning signals.

### 8.1 Suno

Suno is a leading AI music generation platform (developed by Suno Inc., previously known for Bark TTS). Suno's model generates complete songs with lyrics, vocals, and instrumentation from text prompts.

#### 8.1.1 Architecture (Inferred)

Suno's architecture is proprietary but likely involves:

**Multiple specialized models**:
- **Lyrics model**: A language model that generates song lyrics in various styles and structures (verse, chorus, bridge)
- **Melody model**: Generates melodic structure, chord progressions, and harmony
- **Acoustic model**: Produces the waveform with vocals and instrumentation
- **Mixing model**: Balances and masters the final output

**Audio representation**: Likely uses a combination of:
- Discrete audio tokens (similar to EnCodec)
- Continuous latent representations for fine details
- Hybrid symbolic/continuous approaches for musical structure

#### 8.1.2 Capabilities

- **Full song generation**: Complete songs with structure (intro, verse, chorus, bridge, outro)
- **Lyrics + audio**: Generates both lyrics and vocal performance
- **Style control**: Can specify genre, mood, instrumentation, tempo
- **Multi-instrumental**: Supports various instrument combinations
- **Vocals**: Generates sung vocals in multiple styles
- **Multi-language**: Supports lyrics generation in multiple languages

#### 8.1.3 Usage (via Suno Web Platform)

Users provide prompts like:
```
"A catchy pop song about summer romance with upbeat tempo, 
synth pads, and electronic drums. Female vocals."
```

#### 8.1.4 Limitations

- **Limited fine-grained control**: Cannot specify exact notes, chords, or arrangements
- **Copyright concerns**: Trained on copyrighted music, legal gray area
- **Inconsistency**: Quality can vary significantly between generations
- **Lyrics coherence**: Generated lyrics may lack deep semantic meaning
- **Instrumental separation**: Can struggle with complex arrangements with many instruments

### 8.2 Udio

Udio is an AI music generation platform (founded by ex-Google DeepMind researchers) that focuses on high-quality audio generation.

#### 8.2.1 Architecture (Inferred)

Udio appears to use a diffusion-based architecture (rather than autoregressive):

**Latent diffusion for audio**: Similar to image diffusion models but operating on audio latent representations.

**Hierarchical generation**: Generates music in multiple resolution stages:
1. **Structure stage**: Overall musical structure (bars, section transitions)
2. **Content stage**: Melody, harmony, rhythm details
3. **Texture stage**: Timbre, instrumentation, mixing

**Audio tokenization**: Likely uses a high-fidelity audio tokenizer (similar to EnCodec but with higher quality reconstruction) to enable both generation and editing.

#### 8.2.2 Key Distinctions from Suno

| Feature | Suno | Udio |
|---------|------|------|
| Architecture | Likely autoregressive | Likely diffusion-based |
| Audio quality | Good for full songs | Excellent, especially instrumental |
| Vocal quality | Strong vocal performance | Good, slightly less natural |
| Prompt following | Good style/genre matching | Better fine-grained control |
| Edit capability | Limited (regenerate) | Better (extend, remix) |
| Musical coherence | Good song structure | Excellent harmonic consistency |

#### 8.2.3 Capabilities

- **High-fidelity audio**: Produces CD-quality audio (up to 44.1kHz)
- **Long-form generation**: Can generate extended musical pieces (minutes)
- **Instrumental and vocal**: Full song generation with instruments and vocals
- **Style and genre control**: Precise genre, era, and style specification
- **Audio extension**: Extend existing audio files with AI-generated content
- **Remixing**: Modify aspects of generated songs

---

## 9. 3D Generation Models

3D generative models create 3D assets (meshes, point clouds, neural fields) from text, images, or other inputs.

### 9.1 Point-E

Point-E, developed by OpenAI (2023), is a text-to-3D model that generates 3D point clouds.

#### 9.1.1 Architecture

Point-E uses a cascade of diffusion models:

**Text-to-Image**: GLIDE (or similar diffusion model) generates an image from the text prompt.

**Image-to-3D**: A diffusion model conditioned on the generated image produces a colored point cloud:
- **Point Cloud VAE**: Encodes point clouds into a latent space (like image VAEs but for 3D)
- **Diffusion Model**: Generates point cloud latents conditioned on image features
- **Decoder**: Converts latents back to point clouds

```
text → [GLIDE] → image → [Point Cloud Diffusion] → point cloud
```

#### 9.1.2 Key Characteristics

- **Speed**: Point-E is extremely fast, generating a 3D point cloud in ~1-2 minutes on a single GPU (compared to hours for some implicit representation methods)
- **Color**: Generates colored point clouds (RGB values per point)
- **Limitations**:
  - Point clouds lack surface information (no mesh or occupancy)
  - Lower quality than implicit methods (NeRF, SDF-based)
  - Requires point cloud post-processing for surface reconstruction

#### 9.1.3 Extension: Point-E for 3D Completion and View Synthesis

Point-E can also:
- **Condition on partial point clouds**: Complete incomplete 3D scans
- **Generate novel views**: Synthesize new viewpoints of a 3D object
- **Multi-view generation**: Generate multiple consistent views

### 9.2 DreamFusion

DreamFusion, by Google Research (2023), introduced a paradigm for text-to-3D generation using 2D diffusion models and Score Distillation Sampling (SDS).

#### 9.2.1 Score Distillation Sampling (SDS)

SDS is the key innovation enabling text-to-3D without 3D training data:

**Core idea**: Use a pretrained 2D text-to-image diffusion model (e.g., Imagen) as a critic that evaluates renderings of a 3D representation. The gradient of the diffusion model's denoising loss with respect to the rendered image is backpropagated through a differentiable renderer to optimize the 3D parameters.

**SDS gradient**:
```
∇_θ L_SDS(φ, x = g(θ)) = E_[ w(t) * (ε_φ(z_t; y, t) - ε) * (∂x / ∂θ) ]
```

where:
- `θ` are the 3D representation parameters
- `g(θ)` is the differentiable renderer producing image `x`
- `z_t` is the noisy rendering at timestep `t`
- `ε_φ` is the diffusion model, conditioned on text `y`
- `w(t)` is a weighting function
- `ε` is the noise added during SDS

#### 9.2.2 3D Representation

DreamFusion uses NeRF (Neural Radiance Fields) as the 3D representation:
- An MLP maps 3D coordinates (x, y, z) + view direction to density and color
- Volume rendering produces 2D images from arbitrary viewpoints
- The NeRF is optimized from scratch for each text prompt

#### 9.2.3 Key Improvements over Prior Work

- **No 3D training data**: Only requires a pretrained 2D diffusion model
- **Consistent 3D geometry**: Optimizing from multiple viewpoints naturally produces coherent 3D structure
- **High-quality textures**: The 2D diffusion prior provides rich texture guidance

#### 9.2.4 Limitations and Extensions

**Original Limitations**:
- Slow optimization (hours per object)
- Janus problem (multiple-facing geometry, e.g., two faces)
- Over-saturation and limited color variety
- Fixed resolution (64x64 in original NeRF)

**Extensions**:

**Magic3D**: Coarse-to-fine approach using NeRF followed by mesh fine-tuning with texture maps. Higher resolution (512x512) and faster.

**ProlificDreamer**: Variational Score Distillation (VSD) addresses over-saturation and diversity issues in SDS.

**MVDream**: Multi-view diffusion model (trained on 3D data) that generates multiple consistent views simultaneously, reducing the Janus problem.

**DreamGaussian**: Uses 3D Gaussian Splatting instead of NeRF for faster optimization (3-5 minutes instead of 1-2 hours).

**Stable DreamFusion**: Open-source implementations using Stable Diffusion as the 2D prior.

---

## 10. Multimodal Embeddings

Multimodal embeddings map different modalities into a shared representational space, enabling cross-modal retrieval, comparison, and reasoning.

### 10.1 The Embedding Space

A multimodal embedding function `f` maps inputs from modality `m` to a d-dimensional vector:

```
f_m: X_m → R^d
```

The goal is that semantically similar content has similar embeddings regardless of modality:

```
sim(f_text("a red car"), f_image(<photo of red car>)) >>
sim(f_text("a red car"), f_image(<photo of blue truck>))
```

### 10.2 Key Models

**CLIP (OpenAI)**: The most widely used multimodal embedding model. Embeds images and text into a joint space with 512-768 dimensions depending on the variant.

**SigLIP (Google)**: Uses sigmoid loss instead of softmax for contrastive learning. Better performance with smaller batch sizes. Enables training on large batches more efficiently.

**ImageBind (Meta)**: Extends the CLIP paradigm to bind six modalities (images, text, audio, depth, thermal, IMU) into a single embedding space. Rather than requiring all modality pairs for training, ImageBind learns by pairing each modality with images, and the image-text binding enables information flow between all modalities.

**ALIGN (Google)**: Uses a simple dual-encoder model trained on noisy image-text pairs (1.8B) with contrastive loss. Demonstrates that scale of data can compensate for quality.

**GTR (Google Text Representation)**: Text-only embedding model that can be combined with visual encoders.

**CoCa (Contrastive Captioners)**: Combines contrastive learning with captioning loss in a single encoder-decoder architecture, producing embeddings that capture both discriminative and generative knowledge.

### 10.3 Properties of Good Multimodal Embeddings

- **Alignment**: Matched pairs are closer than unmatched pairs
- **Uniformity**: Embeddings are roughly uniformly distributed on the hypersphere
- **Cross-modal transfer**: Properties learned in one modality transfer to others
- **Linearity**: Semantic relationships are reflected as vector operations (e.g., the classic "king - man + woman = queen" analogy)
- **Robustness**: Invariant to superficial differences (lighting, background, noise)

### 10.4 Evaluation Metrics

- **Recall@K**: Percentage of queries where the correct match is in the top K results
- **MRR (Mean Reciprocal Rank)**: Average of reciprocal ranks of correct matches
- **NCDG (Normalized Cumulative Discounted Gain)**: For ranked retrieval with graded relevance
- **Zero-shot classification**: Use text embeddings as class prototypes
- **Linear probe**: Train a linear classifier on frozen embeddings

### 10.5 Applications

- **Cross-modal search**: Search images with text, or text with images
- **Zero-shot classification**: Classify images using text descriptions of classes
- **Image captioning**: Combined with generative models
- **Visual grounding**: Locate objects in images based on text descriptions
- **Multimodal clustering**: Group similar content across modalities
- **Active learning**: Select informative examples for human labeling

---

## 11. Modality Alignment

Modality alignment refers to techniques for aligning representations across different data modalities, enabling coherent multimodal understanding and generation.

### 11.1 Alignment Objectives

#### 11.1.1 Contrastive Objectives

The most common alignment method. The InfoNCE loss (as used in CLIP) maximizes mutual information between matched pairs:

```
L_NCE = -E[ log( exp(sim(x_i, y_i)/τ) / Σ_j exp(sim(x_i, y_j)/τ) ) ]
```

**Variants**:
- **Sigmoid loss (SigLIP)**: Replaces the softmax with sigmoid losses for each pair independently, allowing grouping of positives vs negatives more flexibly
- **Margin loss**: Uses a margin to separate positive and negative pairs
- **Triplet loss**: Anchor-positive distance < anchor-negative distance + margin

#### 11.1.2 Generative Objectives

Align by generating one modality from another:

- **Cross-modal reconstruction**: Encode one modality and decode another
- **Masked modeling**: Mask tokens in one modality and predict from the other
- **Prefix language modeling**: Condition text generation on visual features (Flamingo-style)
- **Causal modeling**: Generate multimodal sequences autoregressively

#### 11.1.3 Hybrid Objectives

Combine contrastive and generative losses for better alignment:

- **CoCa**: Combines contrastive loss (image-text matching) with captioning loss (text generation conditioned on images)
- **FLAVA**: Uses contrastive, masked-image-modeling, and masked-text-modeling objectives
- **BEiT-3**: Multi-way transformer with masked data modeling on both images and text

### 11.2 Architectural Approaches to Alignment

#### 11.2.1 Dual Encoder

Separate encoders for each modality with alignment via similarity in embedding space.

```
text → Text Encoder → text_emb
image → Image Encoder → img_emb
similarity = cosine(text_emb, img_emb)
```

**Pros**: Efficient, modular, supports independent encoding for retrieval
**Cons**: Limited cross-modal interaction, linear similarity may miss complex relationships

#### 11.2.2 Cross-Attention

Allow features from one modality to attend to features from another:

```
image_features → Cross-Attention(queries=text_features, keys=image_features) → aligned_text
```

**Pros**: Rich interaction, captures fine-grained alignment
**Cons**: Requires both modalities at inference, computationally expensive

#### 11.2.3 Unified Transformer

Process all modalities through a single transformer:

```
[text_tokens, image_tokens, audio_tokens] → Transformer → aligned_representations
```

**Pros**: Maximum interaction, captures all cross-modal relationships
**Cons**: Training efficiency, modality-specific tokenization challenges

### 11.3 Training Strategies

#### 11.3.1 Curriculum Learning

Progressively increase alignment difficulty:
1. Start with large batches and simple pairings
2. Introduce harder negatives (e.g., fine-grained categories)
3. Add multi-image, multi-text samples
4. Include compositional and reasoning tasks

#### 11.3.2 Multi-Task Training

Combine alignment with auxiliary tasks:
- Captioning (generation)
- Detection (localization)
- Classification (semantic understanding)
- Retrieval (discrimination)

#### 11.3.3 Knowledge Distillation

Transfer alignment knowledge from larger to smaller models:
- CLIP ViT-L → TinyCLIP (distillation reduces parameters 90%)
- Maintain alignment quality while reducing compute

### 11.4 The Modality Gap Problem

Even aligned models exhibit a "modality gap" — embeddings from different modalities occupy distinct regions of the shared space rather than being fully integrated.

**Causes**:
- Different information densities (images have more information per sample than text)
- Encoder architecture differences
- Training dynamics (contrastive loss doesn't force full integration)

**Mitigation strategies**:
- **Learnable temperature scaling**: Different temperatures for different modalities
- **Modality-specific normalization**: Apply different normalization strategies per modality
- **Cross-modal translation**: Regularize with reconstruction objectives
- **Gradient equalization**: Balance gradient contributions from different modalities

### 11.5 Downstream Evaluation of Alignment

**Zero-shot**: Direct transfer across modalities (e.g., CLIP zero-shot classification)
**Linear probing**: Linear classifier on frozen aligned features
**Full fine-tuning**: Fine-tune aligned model on downstream task
**Cross-modal retrieval**: Recall-based metrics (R@1, R@5, R@10)
**Visual grounding**: Measure alignment between text phrases and image regions

---

## 12. References and Further Reading

### Foundational Papers

- Dosovitskiy, A., et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ViT paper.
- Redmon, J., et al. (2016). "You Only Look Once: Unified, Real-Time Object Detection." YOLO paper.
- Carion, N., et al. (2020). "End-to-End Object Detection with Transformers." DETR paper.
- Kirillov, A., et al. (2023). "Segment Anything." SAM paper.
- Rombach, R., et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models." Stable Diffusion paper.
- Radford, A., et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision." CLIP paper.

### Survey Papers

- Baltrusaitis, T., et al. (2019). "Multimodal Machine Learning: A Survey and Taxonomy."
- Summaira, J., et al. (2023). "Recent Advances in Multimodal Machine Learning."
- Li, C., et al. (2024). "A Survey on Multimodal Large Language Models."

### Open-Source Implementations

- HuggingFace Diffusers: https://github.com/huggingface/diffusers
- Stable Diffusion WebUI: https://github.com/AUTOMATIC1111/stable-diffusion-webui
- LLaVA: https://github.com/haotian-liu/LLaVA
- Whisper: https://github.com/openai/whisper
- AudioCraft: https://github.com/facebookresearch/audiocraft
- SAM: https://github.com/facebookresearch/segment-anything
- YOLOv8: https://github.com/ultralytics/ultralytics
- Bark: https://github.com/suno-ai/bark
- Point-E: https://github.com/openai/point-e
- OpenCLIP: https://github.com/mlfoundations/open_clip

### Datasets

- LAION-400M / LAION-5B: Large-scale image-text datasets
- SA-1B: SAM's segmentation dataset (1B masks across 11M images)
- CC3M / CC12M: Conceptual Captions datasets
- MS-COCO: Image captioning and detection dataset
- AudioSet: Audio event dataset
- LibriSpeech: Speech recognition dataset
- YouTube-8M: Video understanding dataset

### Benchmarks

- MMBench: Multimodal LLM evaluation
- MMMU: Massively Multimodal Understanding
- MathVista: Mathematical reasoning in visual contexts
- ChartQA: Chart understanding and reasoning
- DocVQA: Document visual question answering
- SEED-Bench: Comprehensive multimodal evaluation

---

*This document is part of the AiBaseKnowledge series on Advanced Topics in AI. Last updated: May 2026.*
