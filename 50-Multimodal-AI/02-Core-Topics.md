# Multimodal AI: Core Topics — Architectures, Training, and Modalities

> **Deep dive into the technical foundations of multimodal AI systems.** This document covers the core architectural patterns, training methodologies, modality-specific encoders, and fusion strategies that power modern multimodal models.

---

## Table of Contents

1. [Modality-Specific Encoders](#modality-specific-encoders)
2. [Fusion Strategies](#fusion-strategies)
3. [Training Methodologies](#training-methodologies)
4. [Alignment and Representation Learning](#alignment-and-representation-learning)
5. [The Mixture-of-Experts Revolution](#the-mixture-of-experts-revolution)
6. [Cross-Modal Reasoning](#cross-modal-reasoning)
7. [Generation Across Modalities](#generation-across-modalities)
8. [Evaluation Benchmarks](#evaluation-benchmarks)

---

## Modality-Specific Encoders

### Vision Encoders

Vision encoders transform raw pixel data into rich feature representations that can be aligned with other modalities.

#### Vision Transformer (ViT)

The dominant architecture for visual encoding in 2026:

```python
import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=14, embed_dim=1024, 
                 depth=24, num_heads=16):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                activation='gelu',
                batch_first=True
            ),
            num_layers=depth
        )
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, images):
        # Patch embedding: (B, C, H, W) → (B, N_patches, embed_dim)
        x = self.patch_embed(images)
        
        # Prepend CLS token
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer encoding
        x = self.transformer(x)
        x = self.norm(x)
        
        return x  # (B, N_patches+1, embed_dim)
```

#### SigLIP Vision Encoder

Google's Sigmoid Loss for Image-Text Pretraining has become the standard for multimodal vision encoding:

```python
# SigLIP-style contrastive training
class SigLIPLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, image_features, text_features, labels):
        # Compute similarity matrix
        logits = torch.matmul(
            image_features, text_features.T
        ) / self.temperature
        
        # Sigmoid-based loss (not softmax like CLIP)
        loss = -(
            labels * torch.log(torch.sigmoid(logits)) +
            (1 - labels) * torch.log(1 - torch.sigmoid(logits))
        )
        
        return loss.mean()
```

**Why SigLIP won over CLIP:**
- Sigmoid loss is more stable for large batch training
- Better zero-shot transfer performance
- More efficient gradient computation

### Audio Encoders

Audio encoders process waveforms or spectrograms into representations suitable for multimodal fusion.

#### Whisper Architecture (OpenAI)

```python
class WhisperEncoder(nn.Module):
    def __init__(self, n_mels=80, n_state=1024, n_head=16, n_layer=6):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        
        self.positional_embedding = nn.Parameter(
            torch.randn(1, 1500, n_state)  # Max 30s audio
        )
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=n_state,
                nhead=n_head,
                dim_feedforward=n_state * 4,
                batch_first=True
            ),
            num_layers=n_layer
        )
        
        self.ln = nn.LayerNorm(n_state)
    
    def forward(self, mel_spectrogram):
        # mel_spectrogram: (B, n_mels, T)
        x = torch.relu(self.conv1(mel_spectrogram))
        x = torch.relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # (B, T', n_state)
        
        x = x + self.positional_embedding[:, :x.size(1), :]
        x = self.transformer(x)
        x = self.ln(x)
        
        return x
```

### Video Encoders

Video encoding extends vision encoding with temporal modeling:

```python
class VideoTransformer(nn.Module):
    def __init__(self, vision_encoder, temporal_layers=6, embed_dim=1024):
        super().__init__()
        self.spatial_encoder = vision_encoder  # Frozen ViT
        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=16,
                dim_feedforward=embed_dim * 4,
                batch_first=True
            ),
            num_layers=temporal_layers
        )
        self.temporal_pos_embed = nn.Parameter(
            torch.randn(1, 256, embed_dim)  # Max 256 frames
        )
    
    def forward(self, video_frames):
        # video_frames: (B, T, C, H, W)
        B, T = video_frames.shape[:2]
        
        # Extract spatial features for each frame
        spatial_features = []
        for t in range(T):
            feat = self.spatial_encoder(video_frames[:, t])
            feat = feat.mean(dim=1)  # Pool spatial tokens
            spatial_features.append(feat)
        
        # Stack: (B, T, embed_dim)
        x = torch.stack(spatial_features, dim=1)
        x = x + self.temporal_pos_embed[:, :T, :]
        
        # Temporal reasoning
        x = self.temporal_transformer(x)
        
        return x  # (B, T, embed_dim)
```

---

## Fusion Strategies

### 1. Concatenation Fusion

The simplest approach — concatenate token sequences from different modalities.

```python
class ConcatFusion(nn.Module):
    def __init__(self, modality_dims, output_dim):
        super().__init__()
        self.projections = nn.ModuleDict({
            mod: nn.Linear(dim, output_dim) 
            for mod, dim in modality_dims.items()
        })
    
    def forward(self, modality_inputs):
        # Project each modality to common dimension
        projected = []
        for mod, inp in modality_inputs.items():
            projected.append(self.projections[mod](inp))
        
        # Concatenate along sequence dimension
        return torch.cat(projected, dim=1)
```

### 2. Cross-Attention Fusion

One modality attends to another, allowing fine-grained interaction:

```python
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, query_modality, key_value_modality):
        # Cross-attention
        attn_out, _ = self.cross_attn(
            query=self.norm1(query_modality),
            key=key_value_modality,
            value=key_value_modality
        )
        x = query_modality + attn_out
        
        # Feed-forward
        x = x + self.ffn(self.norm2(x))
        return x
```

### 3. Gated Fusion

Learned gates control the contribution of each modality:

```python
class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.text_gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.vision_gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.audio_gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
    
    def forward(self, text_feat, vision_feat, audio_feat):
        g_t = self.text_gate(text_feat)
        g_v = self.vision_gate(vision_feat)
        g_a = self.audio_gate(audio_feat)
        
        # Normalize gates
        total = g_t + g_v + g_a + 1e-8
        g_t, g_v, g_a = g_t/total, g_v/total, g_a/total
        
        # Weighted combination
        fused = g_t * text_feat + g_v * vision_feat + g_a * audio_feat
        return fused
```

### 4. Dynamic Routing (Mixture-of-Experts)

Each token is dynamically routed to the most appropriate expert:

```python
class MoEFusion(nn.Module):
    def __init__(self, dim, num_experts=8, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            ) for _ in range(num_experts)
        ])
        self.router = nn.Linear(dim, num_experts)
        self.top_k = top_k
    
    def forward(self, x):
        # Route each token
        router_logits = self.router(x)
        top_k_indices = router_logits.topk(self.top_k, dim=-1).indices
        top_k_weights = torch.softmax(
            router_logits.gather(-1, top_k_indices), dim=-1
        )
        
        # Compute expert outputs
        output = torch.zeros_like(x)
        for k in range(self.top_k):
            indices = top_k_indices[:, :, k]
            weights = top_k_weights[:, :, k]
            
            for i, expert in enumerate(self.experts):
                mask = (indices == i)
                if mask.any():
                    expert_out = expert(x[mask])
                    output[mask] += weights[mask].unsqueeze(-1) * expert_out
        
        return output
```

---

## Training Methodologies

### Stage 1: Modality-Specific Pretraining

Each encoder is pretrained independently on its modality:

```python
# Vision encoder pretraining (e.g., MAE - Masked Autoencoder)
class MaskedAutoencoder(nn.Module):
    def __init__(self, encoder, decoder, mask_ratio=0.75):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask_ratio = mask_ratio
    
    def forward(self, images):
        # Random masking
        mask = torch.bernoulli(
            torch.full_like(images[:, 0, 0, 0], self.mask_ratio)
        ).bool()
        
        # Encode visible patches only
        encoded = self.encoder(images, mask=mask)
        
        # Decode all patches
        reconstructed = self.decoder(encoded)
        
        # Loss on masked patches only
        loss = F.mse_loss(reconstructed[mask], images[mask])
        return loss
```

### Stage 2: Cross-Modal Alignment

Align representations from different modality encoders:

```python
# Contrastive learning for cross-modal alignment
class ContrastiveAlignment(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, image_features, text_features):
        # L2 normalize
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute similarity
        logits = torch.matmul(
            image_features, text_features.T
        ) / self.temperature
        
        # Symmetric contrastive loss
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        
        return (loss_i2t + loss_t2i) / 2
```

### Stage 3: Instruction Tuning with Multimodal Data

Fine-tune the fused model on multimodal instruction-following data:

```python
# Multimodal instruction tuning
class MultimodalInstructionTuner:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def prepare_data(self, sample):
        """
        Sample format:
        {
            "instruction": "What is happening in this video?",
            "input": {"video": video_frames, "text": instruction},
            "output": "The video shows a person cooking pasta..."
        }
        """
        # Tokenize text
        text_tokens = self.tokenizer(sample["instruction"])
        
        # Encode video
        video_tokens = self.model.video_encoder(sample["input"]["video"])
        
        # Concatenate
        input_tokens = torch.cat([text_tokens, video_tokens], dim=1)
        
        # Target
        target = self.tokenizer(sample["output"])
        
        return input_tokens, target
```

---

## Alignment and Representation Learning

### Contrastive Learning

The foundation of multimodal alignment — teaching the model that certain image-text pairs are related while others are not.

### ITM (Image-Text Matching)

Binary classification: given an image and text, predict whether they match.

### ITC (Image-Text Contrastive)

Pull matching pairs together and push non-matching pairs apart in embedding space.

### Captioning Loss

Train the model to generate accurate captions for images:

```python
class CaptioningLoss(nn.Module):
    def __init__(self, vocab_size, pad_token_id):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    
    def forward(self, logits, captions):
        # Shift for autoregressive training
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = captions[:, 1:].contiguous()
        
        return self.loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
```

---

## The Mixture-of-Experts Revolution

### Why MoE Dominates Multimodal AI in 2026

The key insight: different modalities benefit from different processing strategies, but sharing a common representation space is essential.

```python
# MoE architecture for multimodal models
class MultimodalMoE(nn.Module):
    def __init__(self, dim, num_experts=16, top_k=2):
        super().__init__()
        
        # Modality-specific experts
        self.text_experts = nn.ModuleList([
            TransformerBlock(dim) for _ in range(num_experts // 4)
        ])
        self.vision_experts = nn.ModuleList([
            TransformerBlock(dim) for _ in range(num_experts // 4)
        ])
        self.audio_experts = nn.ModuleList([
            TransformerBlock(dim) for _ in range(num_experts // 4)
        ])
        self.shared_experts = nn.ModuleList([
            TransformerBlock(dim) for _ in range(num_experts // 4)
        ])
        
        # Router network
        self.router = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, num_experts)
        )
    
    def forward(self, x, modality_mask):
        all_experts = (
            self.text_experts + self.vision_experts + 
            self.audio_experts + self.shared_experts
        )
        
        # Route
        router_logits = self.router(x)
        top_k_weights, top_k_indices = router_logits.topk(2, dim=-1)
        top_k_weights = F.softmax(top_k_weights, dim=-1)
        
        # Compute weighted sum of expert outputs
        output = torch.zeros_like(x)
        for k in range(2):
            expert_idx = top_k_indices[:, :, k]
            weight = top_k_weights[:, :, k]
            
            for i, expert in enumerate(all_experts):
                mask = (expert_idx == i)
                if mask.any():
                    output[mask] += weight[mask].unsqueeze(-1) * expert(x[mask])
        
        return output
```

### Load Balancing

Ensuring all experts get utilized evenly:

```python
def load_balancing_loss(router_logits, num_experts):
    """Auxiliary loss to encourage balanced expert utilization."""
    # Probability distribution over experts
    probs = F.softmax(router_logits, dim=-1)
    
    # Fraction of tokens routed to each expert
    tokens_per_expert = probs.mean(dim=[0, 1])
    
    # Uniform distribution
    uniform = torch.ones(num_experts, device=probs.device) / num_experts
    
    # KL divergence
    loss = F.kl_div(
        tokens_per_expert.log(), 
        uniform, 
        reduction='sum'
    )
    
    return loss
```

---

## Cross-Modal Reasoning

### Chain-of-Thought Across Modalities

```python
class MultimodalCoT(nn.Module):
    def __init__(self, model, max_reasoning_steps=5):
        super().__init__()
        self.model = model
        self.max_steps = max_reasoning_steps
    
    def reason(self, question, images, audio=None):
        reasoning_chain = []
        
        for step in range(self.max_steps):
            # Generate next reasoning step
            step_output = self.model.generate(
                context=reasoning_chain,
                question=question,
                images=images,
                audio=audio
            )
            
            # Check if reasoning is complete
            if step_output.get("is_final", False):
                return step_output["answer"], reasoning_chain
            
            reasoning_chain.append(step_output["step"])
        
        return reasoning_chain[-1], reasoning_chain
```

### Visual Question Answering (VQA)

```python
class VQAModel(nn.Module):
    def __init__(self, vision_encoder, language_model):
        super().__init__()
        self.vision = vision_encoder
        self.language = language_model
        self.vision_projection = nn.Linear(1024, language_model.config.hidden_size)
    
    def answer(self, image, question):
        # Encode image
        visual_features = self.vision(image)
        visual_features = self.vision_projection(visual_features)
        
        # Tokenize question
        question_tokens = self.language.tokenize(question)
        
        # Combine and generate
        combined = torch.cat([visual_features, question_tokens], dim=1)
        answer = self.language.generate(combined)
        
        return answer
```

---

## Generation Across Modalities

### Text-to-Image (Diffusion)

```python
class TextToImageModel(nn.Module):
    def __init__(self, text_encoder, unet, vae):
        super().__init__()
        self.text_encoder = text_encoder
        self.unet = unet
        self.vae = vae
    
    def generate(self, prompt, num_steps=50, guidance_scale=7.5):
        # Encode text
        text_embeddings = self.text_encoder(prompt)
        
        # Random latent noise
        latents = torch.randn(1, 4, 64, 64)
        
        # Denoise
        for t in reversed(range(num_steps)):
            noise_pred = self.unet(latents, t, text_embeddings)
            
            # Classifier-free guidance
            noise_guided = noise_pred[0] + guidance_scale * (
                noise_pred[0] - noise_pred[1]
            )
            
            # Update latents
            latents = scheduler.step(noise_guided, t, latents)
        
        # Decode to image
        image = self.vae.decode(latents)
        return image
```

### Image-to-Text (Captioning)

```python
class ImageCaptioner(nn.Module):
    def __init__(self, vision_encoder, language_model):
        super().__init__()
        self.vision = vision_encoder
        self.language = language_model
    
    def caption(self, image, max_length=100):
        visual_features = self.vision(image)
        
        # Autoregressive generation
        tokens = [self.language.bos_token_id]
        
        for _ in range(max_length):
            input_tensor = torch.tensor([tokens])
            logits = self.language(input_tensor, visual_features)
            
            next_token = logits[:, -1, :].argmax(dim=-1)
            tokens.append(next_token.item())
            
            if next_token.item() == self.language.eos_token_id:
                break
        
        return self.language.decode(tokens)
```

### Audio-to-Text and Text-to-Audio

```python
class SpeechTranslator:
    def __init__(self, asr_model, tts_model):
        self.asr = asr_model  # Audio → Text
        self.tts = tts_model  # Text → Audio
    
    def translate_audio(self, source_audio, target_language):
        # Transcribe source audio
        text = self.asr.transcribe(source_audio, language="auto")
        
        # Translate text
        translated_text = translate(text, target_language)
        
        # Synthesize target audio
        target_audio = self.tts.synthesize(translated_text)
        
        return target_audio
```

---

## Evaluation Benchmarks

### Standard Benchmarks (2026)

| Benchmark | Task | Models Evaluated | State of Art |
|-----------|------|-----------------|--------------|
| MMMU | Multimodal reasoning | 30+ | 78.5% (Gemini 3) |
| MathVista | Math with visual reasoning | 25+ | 72.3% (GPT-5) |
| Video-MME | Video understanding | 20+ | 81.2% (Gemini 3) |
| AudioBench | Audio understanding | 15+ | 85.7% (GPT-5) |
| DocVQA | Document understanding | 25+ | 93.1% (Claude Opus 4) |
| ChartQA | Chart understanding | 20+ | 91.5% (Gemini 3) |

### Custom Evaluation Framework

```python
class MultimodalEvaluator:
    def __init__(self, model, benchmarks):
        self.model = model
        self.benchmarks = benchmarks
    
    def evaluate(self):
        results = {}
        
        for name, benchmark in self.benchmarks.items():
            correct = 0
            total = 0
            
            for sample in benchmark:
                prediction = self.model.predict(
                    sample["input"],
                    modality=sample["modality"]
                )
                
                if self.check_answer(prediction, sample["answer"]):
                    correct += 1
                total += 1
            
            results[name] = correct / total
        
        return results
```

---

## Key Takeaways

1. **ViT + SigLIP** dominate vision encoding; **Whisper-style** encoders lead audio
2. **Cross-attention** and **MoE routing** are the top fusion strategies in 2026
3. **Three-stage training** (pretrain → align → instruction-tune) is the standard pipeline
4. **MoE architectures** enable scaling to trillions of parameters efficiently
5. **Cross-modal reasoning** via chain-of-thought is the frontier capability
6. **Evaluation** remains fragmented — MMMU and Video-MME are the leading benchmarks

---

*See also: [01-Overview.md](./01-Overview.md) for high-level context, [03-Technical-Deep-Dive.md](./03-Technical-Deep-Dive.md) for implementation details*

*Last updated: July 4, 2026*

---
**See also:**
- [09 - Multimodal AI Governance: Governing Vision, Language, and Action](21-AI-Regulation-Antitrust/09-Multimodal-AI-Governance.md)
- [Multimodal AI: Architectures, Models, and Alignment](06-Advanced/01-Multimodal-AI.md)
- [04 — Multimodal Research: The Frontier (2025–2026)](07-Emerging/17-Research-Frontiers-2026/04-Multimodal-Research.md)
