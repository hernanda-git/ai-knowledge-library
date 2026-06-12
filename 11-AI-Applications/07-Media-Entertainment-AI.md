# AI in Media & Entertainment

## Table of Contents
1. [Introduction](#introduction)
2. [Content Recommendation Systems](#content-recommendation-systems)
   - [YouTube Recommendation Architecture](#youtube-recommendation-architecture)
   - [Netflix Recommendation System](#netflix-recommendation-system)
   - [Time-Weighted Collaborative Filtering](#time-weighted-collaborative-filtering)
3. [Generative Media](#generative-media)
   - [DALL-E & Text-to-Image Generation](#dall-e--text-to-image-generation)
   - [Sora & Video Generation](#sora--video-generation)
   - [Music Generation with Transformers](#music-generation-with-transformers)
   - [Text-to-Speech & Voice Synthesis](#text-to-speech--voice-synthesis)
4. [Deepfake Detection & Media Forensics](#deepfake-detection--media-forensics)
   - [GAN Fingerprint Detection](#gan-fingerprint-detection)
   - [Temporal Inconsistency Detection](#temporal-inconsistency-detection)
   - [Forensic Watermarking](#forensic-watermarking)
5. [Content Moderation](#content-moderation)
   - [Multi-Modal Moderation Models](#multi-modal-moderation-models)
   - [Toxic Content Detection](#toxic-content-detection)
   - [Copyright Infringement Detection](#copyright-infringement-detection)
6. [Game AI](#game-ai)
   - [NPC Behavior Trees & Utility AI](#npc-behavior-trees--utility-ai)
   - [Reinforcement Learning for Game Playing](#reinforcement-learning-for-game-playing)
   - [Procedural Content Generation](#procedural-content-generation)
   - [Game Testing with AI](#game-testing-with-ai)
7. [Sports Analytics](#sports-analytics)
   - [Player Tracking & Computer Vision](#player-tracking--computer-vision)
   - [Game Strategy Optimization](#game-strategy-optimization)
   - [Injury Prediction & Prevention](#injury-prediction--prevention)
8. [Case Studies](#case-studies)
9. [Cross-References](#cross-references)
10. [Summary & Conclusion](#summary--conclusion)

---

## Introduction

The media and entertainment industry has been fundamentally reshaped by AI, from the algorithms that recommend content on streaming platforms to the generative models that create new images, video, music, and text. This document provides a deep technical exploration of AI in media and entertainment, covering content recommendation, generative media, deepfake detection, content moderation, game AI, and sports analytics.

The global AI in media and entertainment market was valued at $14.4 billion in 2023 and is projected to reach $52.6 billion by 2030, driven by streaming personalization, generative AI tools, and automated content production.

## Content Recommendation Systems

### YouTube Recommendation Architecture

YouTube's recommendation system serves billions of users daily, using a sophisticated two-stage architecture:

```python
import torch
import torch.nn as nn
import tensorflow as tf

class YouTubeRecommendationModel:
    """
    Two-stage recommendation architecture:
    1. Candidate Generation (retrieval): ~100s of candidates out of billions
    2. Ranking: Precise scoring of candidates
    """
    
    # Stage 1: Candidate Generation (Deep Neural Retrieval)
    class CandidateGenerator(nn.Module):
        """
        Two-tower model for candidate generation.
        
        User tower: features about the user (watch history, demographics)
        Video tower: features about the video (content, metadata)
        
        Candidates retrieved via approximate nearest neighbor search
        in the learned embedding space.
        """
        def __init__(self, n_videos, n_categories, hidden_units=[256, 128, 64]):
            super().__init__()
            
            # User tower
            self.user_history_embed = nn.Embedding(n_videos, 32)
            self.user_search_embed = nn.Embedding(50000, 32)  # Search tokens
            self.user_geo_embed = nn.Embedding(300, 16)  # Geographic
            self.user_device_embed = nn.Embedding(10, 8)
            
            self.user_mlp = nn.Sequential(
                nn.Linear(32 + 32 + 16 + 8 + 10, hidden_units[0]),
                nn.ReLU(),
                nn.Linear(hidden_units[0], hidden_units[1]),
                nn.ReLU(),
                nn.Linear(hidden_units[1], hidden_units[2])
            )
            
            # Video tower
            self.video_id_embed = nn.Embedding(n_videos, 32)
            self.video_category_embed = nn.Embedding(n_categories, 16)
            self.video_topic_embed = nn.Embedding(1000, 32)
            
            # Video age (freshness bias)
            self.video_age_weight = nn.Parameter(torch.ones(1) * 0.1)
            
            self.video_mlp = nn.Sequential(
                nn.Linear(32 + 16 + 32 + 1, hidden_units[0]),
                nn.ReLU(),
                nn.Linear(hidden_units[0], hidden_units[1]),
                nn.ReLU(),
                nn.Linear(hidden_units[1], hidden_units[2])
            )
        
        def forward(self, user_features, video_features):
            user_emb = self.user_mlp(user_features)
            video_emb = self.video_mlp(video_features)
            
            # Normalize for cosine similarity
            user_emb = nn.functional.normalize(user_emb, p=2, dim=1)
            video_emb = nn.functional.normalize(video_emb, p=2, dim=1)
            
            return user_emb, video_emb
        
        def score(self, user_emb, video_emb):
            return (user_emb * video_emb).sum(dim=1)
    
    # Stage 2: Ranking with Deep Neural Network
    class Ranker(nn.Module):
        """
        Wide & Deep ranking model with hundreds of features.
        
        Wide component: Memorization of feature interactions
        Deep component: Generalization to unseen feature combinations
        """
        def __init__(self, n_features_sparse, n_features_dense=100):
            super().__init__()
            
            # Wide (linear) component
            # Cross-product transformations for memorization
            self.wide = nn.Linear(n_features_sparse + n_features_dense, 1)
            
            # Deep component
            self.deep = nn.Sequential(
                nn.Linear(n_features_dense, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        
        def forward(self, wide_features, deep_features):
            wide_out = self.wide(wide_features)
            deep_out = self.deep(deep_features)
            
            return torch.sigmoid(wide_out + deep_out)

# YouTube's training approach
def youtube_training_pipeline():
    """
    Training setup for YouTube's recommendation model.
    """
    model = YouTubeRecommendationModel()
    
    # Loss functions
    candidate_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    ranking_loss = 'weighted_logistic_loss'  # Custom: weight by watch time
    
    # Expected watch time per impression weighting
    # Instead of optimizing click-through rate, YouTube optimizes
    # expected watch time = P(click) * E(watch_time | click)
    
    # Weighted logistic regression:
    # loss = - (watch_time * log(sigmoid(logit)) + (1 - click) * log(1 - sigmoid(logit)))
    
    # Importance weighting for imbalanced data
    # Negative sampling: 500-1000 negatives per positive impression
    
    return model
```

**Key design decisions in YouTube's system:**

1. **Expected watch time optimization** (not CTR): Actors/creators are incented to produce engaging content rather than clickbait
2. **Freshness bias**: New videos receive a time-dependent boost in the candidate generation stage, decaying as \( \frac{1}{\sqrt{t+1}} \)
3. **Negative sampling**: For each positive impression, 500-1000 negative videos are sampled, weighted by popularity to reduce bias toward popular content

### Netflix Recommendation System

Netflix's recommendation system is a multi-algorithm ensemble that handles hundreds of personalized rows:

```yaml
netflix_recsys:
  algorithms:
    personalized_video_ranking:
      - Matrix factorization (SVD) for user-video affinity
      - Time-aware CF with temporal decay
      - Autoencode-based collaborative filtering
      - Factorization machines for feature interactions
    
    contextual_bandits:
      - "Select row arrangement based on user state"
      - "Explore new genres/titles for cold-start users"
      - "Thompson Sampling for row selection"
    
    similarity_models:
      - "Video-to-video: content embedding + collaborative signals"
      - "More like this: hybrid content + co-watch graph"
      - "Because you watched: causal co-watch patterns"
    
    trending & social:
      - "Popular in your region (seasonal adjustment)"
      - "Friends watching (social graph, where available)"
      - "News & events-aware trending"
  
  row_generation:
    - "Continue watching" (top of page, highest engagement)
    - "Top picks for you" (highest predicted rating)
    - "Trending now" (freshness + popularity)
    - "New releases" (recency-weighted)
    - "Because you watched X" (specific recommendations)
    - "Genre rows" (personalized genre ranking)
  
  AB_testing:
    - "Thousands of concurrent experiments"
    - "Multi-armed bandit for row placement optimization"
    - "Interleaved experiments for ranking comparisons"
  
  metrics:
    - "Session duration (primary)"
    - "Member retention (long-term)"
    - "Title discovery rate"
    - "Search rate (lower is better for recommendations)"
```

### Time-Weighted Collaborative Filtering

```python
class TimeWeightedMatrixFactorization(nn.Module):
    """
    TimeSVD++: Collaborative filtering with temporal dynamics.
    
    Accounts for:
    - Drifting user preferences over time
    - Item popularity changes
    - Seasonal effects
    """
    def __init__(self, n_users, n_items, n_factors=50, n_time_bins=100):
        super().__init__()
        
        # Static components
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        
        # Temporal user bias (changes over time)
        # Each user has time-dependent bias learned per time bin
        self.time_bias = nn.Embedding(n_users * n_time_bins, 1)
        
        # Temporal item popularity
        self.item_time_bias = nn.Embedding(n_items * n_time_bins, 1)
        
        # Day-of-week effects
        self.dow_user = nn.Embedding(n_users * 7, 1)
        self.dow_item = nn.Embedding(n_items * 7, 1)
    
    def forward(self, user_ids, item_ids, time_bins, day_of_week):
        # Static prediction
        pred = (self.user_factors(user_ids) * self.item_factors(item_ids)).sum(dim=1)
        pred += self.user_bias(user_ids).squeeze()
        pred += self.item_bias(item_ids).squeeze()
        
        # Temporal components
        time_idx = user_ids * len(self.time_bias) + time_bins
        pred += self.time_bias(time_idx).squeeze()
        
        item_time_idx = item_ids * len(self.item_time_bias) + time_bins
        pred += self.item_time_bias(item_time_idx).squeeze()
        
        # Day-of-week
        dow_user_idx = user_ids * 7 + day_of_week
        pred += self.dow_user(dow_user_idx).squeeze()
        
        return torch.sigmoid(pred)
```

## Generative Media

### DALL-E & Text-to-Image Generation

DALL-E and similar models have revolutionized visual content creation. The core architecture combines a text encoder with a diffusion-based image generator:

```python
import torch
import torch.nn as nn

class SimplifiedDiffusionModel(nn.Module):
    """
    Conceptual architecture of a text-to-image diffusion model.
    
    Components:
    1. Text encoder (Transformer) → text embeddings
    2. Image encoder (VAE) → latent space
    3. Diffusion backbone (U-Net with attention) → denoising
    """
    def __init__(self, text_vocab_size=49408, text_dim=768, 
                 image_size=64, latent_dim=4, n_steps=1000):
        super().__init__()
        
        # Text encoder (CLIP-like)
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=text_dim,
                nhead=12,
                dim_feedforward=text_dim * 4,
                batch_first=True
            ),
            num_layers=12
        )
        self.text_embedding = nn.Embedding(text_vocab_size, text_dim)
        
        # Image VAE encoder/decoder
        self.vae_encoder = self._build_vae_encoder(image_size, latent_dim)
        self.vae_decoder = self._build_vae_decoder(latent_dim, image_size)
        
        # Diffusion U-Net
        self.denoising_unet = self._build_unet(latent_dim, text_dim)
        
        # Noise schedule (cosine schedule)
        self.register_buffer('betas', self._cosine_betas(n_steps))
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alpha_bars', torch.cumprod(self.alphas, dim=0))
    
    def _build_vae_encoder(self, img_size, latent_dim):
        """Compress image to latent space"""
        return nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),  # 16 -> 8
            nn.ReLU(),
            nn.Conv2d(256, latent_dim * 2, 3, 1, 1),  # 8 -> 8
        )
    
    def _build_vae_decoder(self, latent_dim, img_size):
        """Reconstruct image from latent"""
        return nn.Sequential(
            nn.Conv2d(latent_dim, 256, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),  # 32 -> 64
            nn.Tanh()
        )
    
    def _build_unet(self, latent_dim, text_dim):
        """U-Net for denoising in latent space"""
        # Cross-attention layers for text conditioning
        # Time embedding for diffusion step
        # Down/up blocks with residual connections
        
        # Simplified representation
        return nn.Module()  # Full implementation is extensive
    
    def _cosine_betas(self, n_steps, s=0.008):
        """Cosine noise schedule (improved over linear)"""
        steps = torch.linspace(0, n_steps, n_steps + 1)
        f_t = torch.cos((steps / n_steps + s) / (1 + s) * torch.pi / 2) ** 2
        alphas_bar = f_t / f_t[0]
        betas = 1 - alphas_bar[1:] / alphas_bar[:-1]
        return torch.clamp(betas, max=0.999)
    
    def encode_image(self, image):
        """Encode image to latent representation"""
        mu_logvar = self.vae_encoder(image)
        mu, logvar = mu_logvar.chunk(2, dim=1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode_latent(self, latent):
        """Decode latent back to image"""
        return self.vae_decoder(latent)
    
    def forward_diffusion(self, latent, t):
        """Add noise to latent at timestep t"""
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1)
        noise = torch.randn_like(latent)
        noisy = torch.sqrt(alpha_bar) * latent + torch.sqrt(1 - alpha_bar) * noise
        return noisy, noise
    
    def text_to_image(self, text, n_steps=50):
        """Generate image from text prompt"""
        # Encode text
        tokens = self.tokenizer(text, return_tensors='pt')
        text_emb = self.text_encoder(self.text_embedding(tokens['input_ids']))
        
        # Start from pure noise
        latent = torch.randn(1, 4, 64, 64)
        
        # DDIM sampling (faster than DDPM)
        for t in reversed(range(n_steps)):
            t_tensor = torch.full((1,), t, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.denoising_unet(latent, t_tensor, text_emb)
            
            # Denoise step
            alpha = self.alphas[t]
            alpha_bar = self.alpha_bars[t]
            alpha_bar_prev = self.alpha_bars[t-1] if t > 0 else torch.tensor(1.0)
            
            # DDIM update
            pred_x0 = (latent - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)
            latent = torch.sqrt(alpha_bar_prev) * pred_x0 + torch.sqrt(1 - alpha_bar_prev) * noise_pred
        
        return self.decode_latent(latent)
```

**Key diffusion model innovations for media generation:**

| Model | Parameter Count | Key Innovation | Primary Use |
|-------|----------------|----------------|-------------|
| DALL-E 2 | 3.5B | CLIP text encoder + diffusion prior | Text-to-image |
| DALL-E 3 | ~12B | Direct caption training | Text-to-image |
| Stable Diffusion | 890M | Latent diffusion (compress in VAE space) | Open-source text-to-image |
| Imagen | 2.1B | Text-only training with frozen text encoder | High-fidelity images |
| Midjourney v6 | ~12B | Proprietary aesthetic tuning | Artistic images |

### Sora & Video Generation

OpenAI's Sora represents a breakthrough in text-to-video generation. Its architecture combines diffusion transformers with video compression networks:

```yaml
sora_architecture:
  video_compressor:
    - "3D VAE that compresses video to latent space"
    - "Spatial compression: 8x8 factor"
    - "Temporal compression: 4x factor"
    - "Output: (T/4) x (H/8) x (W/8) x C latent"
  
  diffusion_transformer:
    backbone: "DiT (Diffusion Transformer)"
    patches: "2x2x2 space-time patches"
    conditioning: "Text embeddings from re-captioner"
    
    architecture:
      - "N=28 transformer layers"
      - "Attention: ST-Attention (separate spatial + temporal)"
      - "Adaptive layer norm for timestep conditioning"
      - "Cross-attention for text conditioning"
  
  re-captioner:
    - "Dense video captioning model"
    - "Generates detailed text descriptions for training videos"
    - "Enables text-faithful video generation"
  
  sampling:
    - "Progressive scaling: 480p -> 720p -> 1080p"
    - "Variable duration: 1s to 60s"
    - "Aspect ratio: Native (not cropped)"
```

### Music Generation with Transformers

```python
class MusicTransformer(nn.Module):
    """
    Music generation using decoder-only Transformer.
    
    Represents music as discrete tokens:
    - NOTE_ON, NOTE_OFF, TIME_SHIFT, VELOCITY
    - Multi-track: piano, strings, drums, etc.
    """
    def __init__(self, vocab_size=5000, hidden=512, n_layers=12, n_heads=8):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden)
        self.pos_embedding = nn.Embedding(2048, hidden)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=hidden * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, n_layers)
        
        self.output = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, vocab_size)
        )
    
    def forward(self, tokens, mask=None):
        seq_len = tokens.size(1)
        
        # Position encoding
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
        x = self.token_embedding(tokens) + self.pos_embedding(positions)
        
        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len) * float('-inf'),
            diagonal=1
        ).to(tokens.device)
        
        # Generate autoregressively
        output = self.transformer(x, x, tgt_mask=causal_mask)
        return self.output(output)
    
    def generate(self, prompt_tokens, max_length=500, temperature=0.8):
        """Generate music continuation"""
        self.eval()
        generated = prompt_tokens.clone()
        
        for _ in range(max_length - len(prompt_tokens)):
            with torch.no_grad():
                logits = self.forward(generated.unsqueeze(0))
                next_logits = logits[0, -1, :] / temperature
                
                # Sample with top-k filtering
                top_k = 50
                top_k_logits, top_k_indices = torch.topk(next_logits, top_k)
                probs = torch.softmax(top_k_logits, dim=-1)
                next_token = top_k_indices[torch.multinomial(probs, 1)]
                
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=-1)
        
        return generated

class MusicTokenization:
    """Tokenize MIDI files for transformer training"""
    
    @staticmethod
    def midi_to_tokens(midi_file, resolution=16):
        """Convert MIDI to discrete token sequence"""
        # MIDI parsing with pretty_midi
        import pretty_midi
        midi = pretty_midi.PrettyMIDI(midi_file)
        
        tokens = []
        for instrument in midi.instruments:
            # Sort notes by start time
            notes = sorted(instrument.notes, key=lambda n: n.start)
            
            current_time = 0
            for note in notes:
                # Time shift
                time_shift = int((note.start - current_time) * resolution)
                if time_shift > 0:
                    tokens.append(('TIME_SHIFT', time_shift))
                
                # Note on
                tokens.append(('NOTE_ON', note.pitch, note.velocity // 16))
                
                # Note off (could be separate or implied by duration)
                duration = int((note.end - note.start) * resolution)
                tokens.append(('NOTE_DURATION', duration))
                
                current_time = note.start
        
        return tokens
```

**AI music generation platforms:**

| Platform | Model | Key Feature | Output |
|----------|-------|------------|--------|
| Suno v4 | Diffusion + Transformer | High-quality vocal generation | Full songs with lyrics |
| MusicLM (Google) | Hierarchical Transformer | Text-conditioned music generation | Long-form musical audio |
| Jukebox (OpenAI) | VQ-VAE + Transformer | Multi-genre with lyrics| Raw audio (not MIDI) |
| MuseNet (OpenAI) | Sparse Transformer | Multi-instrument composition | MIDI files |
| Riffusion | Fine-tuned Stable Diffusion | Spectrogram-to-audio | Real-time music generation |

## Deepfake Detection & Media Forensics

### GAN Fingerprint Detection

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepfakeDetectorCNN(nn.Module):
    """
    Forensics++ based deepfake detector.
    
    Detects GAN-generated images by analyzing:
    1. Frequency domain artifacts (FFT)
    2. Color correlation anomalies
    3. Upsampling artifacts
    4. Blending boundary inconsistencies
    """
    def __init__(self, num_classes=2):
        super().__init__()
        
        # RGB stream
        self.rgb_stream = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Frequency stream (takes FFT magnitude as input)
        self.freq_stream = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Steganalysis features (neighboring pixel correlations)
        self.srm_conv = self._make_srm_layer()
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(64 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def _make_srm_layer(self):
        """Spatial Rich Model filters for steganalysis"""
        srm = nn.Conv2d(3, 3, 5, padding=2, bias=False)
        
        # SRM filters: highlight pixel prediction errors
        kernel = torch.tensor([
            [[0, 0, 0, 0, 0],
             [0, -1, 2, -1, 0],
             [0, 2, -4, 2, 0],
             [0, -1, 2, -1, 0],
             [0, 0, 0, 0, 0]],
        ], dtype=torch.float32).repeat(3, 1, 1)
        
        srm.weight = nn.Parameter(kernel.unsqueeze(1), requires_grad=False)
        return srm
    
    def forward(self, x):
        # RGB features
        rgb_f = self.rgb_stream(x)
        rgb_f = F.adaptive_avg_pool2d(rgb_f, 1).view(x.size(0), -1)
        
        # Frequency features
        fft = torch.fft.rfft2(x, norm='ortho')
        fft_mag = torch.abs(fft)
        freq_f = self.freq_stream(fft_mag)
        freq_f = F.adaptive_avg_pool2d(freq_f, 1).view(x.size(0), -1)
        
        # SRM features
        srm_f = self.srm_conv(x)
        srm_f = F.adaptive_avg_pool2d(srm_f, 1).view(x.size(0), -1)
        
        # Concatenate and classify
        combined = torch.cat([rgb_f, freq_f, srm_f], dim=1)
        return self.fusion(combined)
```

**Deepfake detection benchmark accuracy (FaceForensics++):**

| Detection Method | Raw Deepfakes | F2F | FaceSwap | NeuralTextures |
|-----------------|---------------|-----|----------|----------------|
| Xception (full) | 99.7% | 93.2% | 94.0% | 76.5% |
| CNN + FFT | 99.3% | 92.8% | 93.5% | 75.2% |
| Steg. features | 98.5% | 85.3% | 87.1% | 62.3% |
| MesoNet | 95.6% | 83.1% | 84.5% | 60.8% |

### Temporal Inconsistency Detection for Video Deepfakes

```python
class TemporalDeepfakeDetector(nn.Module):
    """
    Detects video deepfakes by analyzing temporal inconsistencies.
    
    Real videos have consistent:
    - Head pose (smooth trajectories)
    - Eye blinking (physiological cadence)
    - Lip-sync (matching audio)
    - Illumination (consistent across frames)
    """
    def __init__(self):
        super().__init__()
        
        # Frame-level feature extractor
        self.frame_encoder = models.resnet18(pretrained=True)
        self.frame_encoder.fc = nn.Identity()  # Remove classifier
        
        # Temporal inconsistency detector
        self.temporal_model = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
    
    def forward(self, video_frames):
        """
        video_frames: (batch, n_frames, C, H, W)
        """
        batch_size, n_frames = video_frames.shape[:2]
        
        # Extract frame features
        frame_features = []
        for t in range(n_frames):
            feat = self.frame_encoder(video_frames[:, t])
            frame_features.append(feat)
        
        frame_features = torch.stack(frame_features, dim=1)
        
        # Temporal modeling
        temporal_out, _ = self.temporal_model(frame_features)
        
        # Average over time
        video_feat = temporal_out.mean(dim=1)
        
        return torch.sigmoid(self.classifier(video_feat)).squeeze()
```

## Game AI

### NPC Behavior Trees & Utility AI

```python
class BehaviorTree:
    """
    Behavior Tree for NPC decision making.
    
    Composite nodes: Sequence, Selector, Parallel
    Decorator nodes: Invert, Repeat, UntilFail
    Leaf nodes: Action, Condition
    """
    class Node:
        def tick(self, context):
            raise NotImplementedError
    
    class Sequence(Node):
        """Run children in order; fails if any child fails"""
        def __init__(self, children):
            self.children = children
        
        def tick(self, context):
            for child in self.children:
                status = child.tick(context)
                if status == 'failure':
                    return 'failure'
                if status == 'running':
                    return 'running'
            return 'success'
    
    class Selector(Node):
        """Run children in order; succeeds if any child succeeds"""
        def __init__(self, children):
            self.children = children
        
        def tick(self, context):
            for child in self.children:
                status = child.tick(context)
                if status == 'success':
                    return 'success'
                if status == 'running':
                    return 'running'
            return 'failure'
    
    class Condition(Node):
        """Check a game state condition"""
        def __init__(self, condition_fn):
            self.condition_fn = condition_fn
        
        def tick(self, context):
            return 'success' if self.condition_fn(context) else 'failure'
    
    class Action(Node):
        """Execute an action"""
        def __init__(self, action_fn, duration=0):
            self.action_fn = action_fn
            self.duration = duration
            self.elapsed = 0
        
        def tick(self, context):
            if self.elapsed < self.duration:
                self.action_fn(context, self.elapsed / self.duration)
                self.elapsed += 1
                return 'running'
            self.elapsed = 0
            return 'success'

# Example: Guard patrol behavior tree
def create_guard_bt():
    return BehaviorTree.Selector([
        # Priority 1: Investigate noise
        BehaviorTree.Sequence([
            BehaviorTree.Condition(lambda ctx: ctx['heard_noise']),
            BehaviorTree.Action(lambda ctx, t: ctx['agent'].move_to(ctx['noise_position'])),
            BehaviorTree.Action(lambda ctx, t: ctx['agent'].investigate(), duration=30)
        ]),
        
        # Priority 2: Chase intruder
        BehaviorTree.Sequence([
            BehaviorTree.Condition(lambda ctx: ctx['saw_intruder']),
            BehaviorTree.Selector([
                # Alert others if possible
                BehaviorTree.Sequence([
                    BehaviorTree.Condition(lambda ctx: ctx['near_alert_station']),
                    BehaviorTree.Action(lambda ctx, t: ctx['agent'].trigger_alarm())
                ]),
                # Chase directly
                BehaviorTree.Action(lambda ctx, t: ctx['agent'].chase(ctx['intruder_position']))
            ])
        ]),
        
        # Default: Patrol
        BehaviorTree.Sequence([
            BehaviorTree.Action(lambda ctx, t: ctx['agent'].patrol(), duration=120),
            BehaviorTree.Action(lambda ctx, t: ctx['agent'].idle(), duration=10)
        ])
    ])
```

### Reinforcement Learning for Game Playing

```python
import gym
import torch
import torch.nn as nn
import numpy as np

class RLGameAgent(nn.Module):
    """
    Deep RL agent for playing games from raw screen pixels.
    Uses DQN with experience replay and double DQN.
    """
    def __init__(self, n_actions, input_shape=(4, 84, 84)):
        super().__init__()
        
        # Convolutional encoder (Atari-style)
        self.cnn = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Compute CNN output size
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            cnn_out = self.cnn(dummy).shape[1]
        
        # Dueling architecture (advantage + value)
        self.advantage = nn.Sequential(
            nn.Linear(cnn_out, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
        self.value = nn.Sequential(
            nn.Linear(cnn_out, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        # Normalize pixel values
        x = x.float() / 255.0
        
        features = self.cnn(x)
        advantage = self.advantage(features)
        value = self.value(features)
        
        # Q = V + (A - mean(A))
        return value + (advantage - advantage.mean(dim=-1, keepdim=True))
    
    def act(self, state, epsilon=0.05):
        """Epsilon-greedy action selection"""
        if np.random.random() < epsilon:
            return np.random.randint(self.advantage[0].out_features)
        
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.forward(state)
        return q_values.argmax().item()

class ReplayBuffer:
    """Experience replay buffer for game RL"""
    def __init__(self, capacity=100000, frame_history=4):
        self.capacity = capacity
        self.frame_history = frame_history
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size=32):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones)
        )

def train_dqn(agent, env, n_episodes=10000):
    """Train DQN agent on game environment"""
    target_agent = RLGameAgent(env.action_space.n)
    target_agent.load_state_dict(agent.state_dict())
    
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-4)
    replay = ReplayBuffer()
    
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            
            replay.push(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            
            # Training step
            if len(replay.buffer) > 1000:
                states, actions, rewards, next_states, dones = replay.sample()
                
                # Double DQN
                with torch.no_grad():
                    next_actions = agent(next_states).argmax(dim=1)
                    next_q = target_agent(next_states)
                    next_q = next_q.gather(1, next_actions.unsqueeze(1)).squeeze()
                    target_q = rewards + 0.99 * next_q * (1 - dones)
                
                current_q = agent(states).gather(1, actions.unsqueeze(1)).squeeze()
                loss = nn.MSELoss()(current_q, target_q)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()
        
        # Update target network
        if episode % 10 == 0:
            target_agent.load_state_dict(agent.state_dict())
        
        print(f"Episode {episode}, Reward: {total_reward}")
```

### Procedural Content Generation

```python
class PCGML(nn.Module):
    """
    Procedural content generation using machine learning.
    
    Generates game levels, maps, and assets using Variational
    Autoencoders and GANs.
    """
    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        
        # VAE for level generation
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, latent_dim * 2)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output tile probabilities
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        params = self.encoder(x)
        mu, logvar = params.chunk(2, dim=1)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
    
    def generate_level(self, n_samples=1):
        """Generate new levels from latent space"""
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim)
            level = self.decoder(z)
        return level

# Mario-style level generation
class MarioLevelGenerator:
    """Generate Super Mario-style platformer levels"""
    def __init__(self):
        self.tile_types = {
            0: 'empty', 1: 'ground', 2: 'brick', 3: 'question_block',
            4: 'pipe', 5: 'enemy', 6: 'coin', 7: 'flag'
        }
    
    def generate_level(self, length=200):
        """Procedural level generation with heuristics + ML guidance"""
        level = np.zeros((16, length), dtype=int)
        
        # Ground layer (rows 13-15)
        level[12:16, :] = 1  # Ground tiles
        
        for x in range(length):
            # Generate terrain features
            if np.random.random() < 0.02:
                # Pipe generation
                pipe_height = np.random.randint(2, 5)
                level[12-pipe_height:12, x] = 4  # Pipe body
                level[12-pipe_height-1, x] = 4  # Pipe top
            
            # Question blocks at reasonable positions
            if np.random.random() < 0.03 and x > 5 and level[8, x] == 0:
                level[8, x] = 3
                if np.random.random() < 0.7:
                    level[8, x+1] = 2  # Adjacent brick
            
            # Enemy placement
            if np.random.random() < 0.02 and x > 10 and level[11, x] == 0:
                if level[11, x-1] != 5:  # No adjacent enemies
                    level[11, x] = 5
            
            # Gaps in ground
            if np.random.random() < 0.04:
                if x > 5 and x < length - 5:
                    gap_width = np.random.randint(1, 4)
                    level[12:16, x:x+gap_width] = 0
        
        return level
```

## Sports Analytics

### Player Tracking & Computer Vision

```python
class PlayerTracking:
    """
    Multi-object tracking for sports using Kalman filters
    and appearance-based re-identification.
    """
    def __init__(self):
        self.tracks = {}
        self.next_id = 0
        self.kalman_filters = {}
    
    def _kalman_filter(self):
        """Kalman filter for smooth tracking"""
        from filterpy.kalman import KalmanFilter
        
        kf = KalmanFilter(dim_x=6, dim_z=2)  # x, y, vx, vy, ax, ay
        kf.F = np.array([
            [1, 0, 1, 0, 0.5, 0],
            [0, 1, 0, 1, 0, 0.5],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])
        kf.R *= 10  # Measurement noise
        kf.P *= 100  # Covariance
        return kf
    
    def update(self, detections, frame):
        """
        detections: list of [x1, y1, x2, y2, confidence]
        """
        matched = set()
        
        # Hungarian algorithm for data association
        from scipy.optimize import linear_sum_assignment
        
        if self.tracks:
            # Compute cost matrix (IoU + appearance distance)
            cost_matrix = np.zeros((len(self.tracks), len(detections)))
            
            for i, (tid, track) in enumerate(self.tracks.items()):
                for j, det in enumerate(detections):
                    iou = self._compute_iou(track['bbox'], det[:4])
                    cost_matrix[i, j] = 1 - iou
            
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            for i, j in zip(row_indices, col_indices):
                if cost_matrix[i, j] < 0.7:  # Match threshold
                    tid = list(self.tracks.keys())[i]
                    self.tracks[tid]['bbox'] = detections[j][:4]
                    self.tracks[tid]['frames_since_update'] = 0
                    matched.add(j)
        
        # New tracks for unmatched detections
        for j in range(len(detections)):
            if j not in matched:
                tid = self.next_id
                self.tracks[tid] = {
                    'bbox': detections[j][:4],
                    'frames_since_update': 0,
                    'reid_embedding': None,
                    'birth_frame': frame
                }
                self.next_id += 1
        
        # Remove stale tracks
        to_remove = []
        for tid, track in self.tracks.items():
            track['frames_since_update'] += 1
            if track['frames_since_update'] > 30:
                to_remove.append(tid)
        
        for tid in to_remove:
            del self.tracks[tid]
        
        return self.tracks
```

## Case Studies

### Case Study 1: YouTube's Recommendation System

**Background**: YouTube's recommendation system serves over 2 billion logged-in monthly users, recommending from a corpus of billions of videos.

**Architecture and results:**
```yaml
youtube_recommendation:
  generation_stage:
    architecture: Two-tower DNN
    training_data: 1B+ daily user-video interactions
    candidates_reduced: From billions to ~100/video
    ANN_index: Google ScaNN (Scalable Nearest Neighbors)
    
  ranking_stage:
    features: ~1000 (including real-time features)
    objective: Expected watch time (not CTR)
    model: Wide & Deep + DCN
    training: Daily incremental updates
    
  business_impact:
    - Watch time increased by ~1% per experiment
    - 1000+ experiments run per day
    - 35%+ of views come from recommendations
```

### Case Study 2: Midjourney's AI Art Generation

**Background**: Midjourney has become the leading platform for AI-generated art, used by millions of creators.

**Technical approach:**
```yaml
midjourney:
  model_version: v6 (February 2024)
  base_architecture: Diffusion model (modified Stable Diffusion)
  
  key_differentiators:
    - "Aesthetic fine-tuning: RLHF from professional artists"
    - "Prompt understanding: re-captioning + prompt expansion"
    - "Stylization: Learned artistic style embeddings"
    - "Resolution: 1024x1024 native, upscaled to 4K"
  
  generation_pipeline:
    1. "Natural language prompt → LLM expansion → detailed prompt"
    2. "Image generation (diffusion, 50 steps)"
    3. "Variation generation (subtle latent perturbations)"
    4. "Upscaling with fine-tuned Real-ESRGAN"
    5. "Aesthetic ranking → top 4 results displayed"
  
  community:
    - 20M+ registered users
    - 10M+ images generated daily
    - Active community on Discord
```

## Cross-References

This document relates to other categories in the AI Knowledge Base:

- **[05-Education-AI.md](05-Education-AI.md)** — Engagement optimization and recommendation systems share personalization techniques
- **[06-Retail-AI.md](06-Retail-AI.md)** — Content recommendation and product recommendation share fundamental architectures
- **[08-Agriculture-AI.md](08-Agriculture-AI.md)** — Computer vision for drone/satellite imagery shares techniques with sports tracking
- **[09-Transportation-AI.md](09-Transportation-AI.md)** — Reinforcement learning for autonomous driving shares techniques with game-playing AI
- **[11-Government-AI.md](11-Government-AI.md)** — Content moderation and deepfake detection have important public policy implications

## Summary & Conclusion

AI in media and entertainment spans a diverse and rapidly evolving set of technologies:

- **Content Recommendation**: Two-tower retrieval + deep ranking models at YouTube, multi-algorithm ensembles at Netflix, time-aware collaborative filtering
- **Generative Media**: Diffusion models (DALL-E, Stable Diffusion, Sora) for image/video generation, autoregressive Transformers (MusicLM, Suno) for music and audio
- **Deepfake Detection**: CNN + frequency analysis for still images, temporal inconsistency detection for video
- **Content Moderation**: Multi-modal models for detecting toxic, violent, and illegal content at platform scale
- **Game AI**: Behavior trees for NPC logic, RL for game-playing agents, procedural content generation with VAEs
- **Sports Analytics**: Computer vision for player tracking, strategy optimization, and injury prediction

The field is characterized by rapid technological advancement — particularly in generative AI — which brings both creative opportunities and challenges around authenticity, copyright, and content safety. The most successful media AI systems carefully balance personalization with content diversity, automation with human creative control, and engagement with responsible content stewardship.
