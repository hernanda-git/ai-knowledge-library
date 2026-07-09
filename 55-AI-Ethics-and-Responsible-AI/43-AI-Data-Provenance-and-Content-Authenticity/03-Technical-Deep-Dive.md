# Technical Deep Dive: AI Data Provenance and Content Authenticity

> This document provides advanced technical details on implementing content provenance systems, from low-level watermarking algorithms to cryptographic infrastructure, forensic analysis, and adversarial robustness.

---

## 1. Advanced Watermarking Algorithms

### 1.1 Robust Watermarking via Deep Learning

#### 1.1.1 HiDDeN Architecture

HiDDeN (Hiding Data With Deep Networks) is the foundation for modern neural watermarking:

```python
# HiDDeN-inspired watermarking system
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class HiDDeNEncoder(nn.Module):
    def __init__(self, message_length=32, channels=64):
        super().__init__()
        self.message_length = message_length
        
        # Message embedding
        self.message_embed = nn.Embedding(2, channels)
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Conv2d(3 + channels, channels, 3, padding=1),
            nn.ReLU(),
            ResBlock(channels),
            ResBlock(channels),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, 3, 3, padding=1),
        )
        
        # Noise layer (simulates attacks during training)
        self.noise = GaussianNoise()
    
    def forward(self, image, message):
        # Embed message to spatial dimensions
        embedded_msg = self.message_embed(message)  # (B, L, C)
        embedded_msg = embedded_msg.permute(0, 2, 1)  # (B, C, L)
        embedded_msg = embedded_msg.unsqueeze(-1).expand(-1, -1, -1, image.shape[-1])
        
        # Concatenate
        combined = torch.cat([image, embedded_msg], dim=1)
        
        # Encode
        residual = self.encoder(combined)
        
        # Apply with strength control
        watermarked = image + residual
        
        # Add noise during training
        if self.training:
            watermarked = self.noise(watermarked)
        
        return torch.clamp(watermarked, 0, 1)


class HiDDeNDecoder(nn.Module):
    def __init__(self, message_length=32, channels=64):
        super().__init__()
        self.message_length = message_length
        
        self.decoder = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.ReLU(),
            ResBlock(channels),
            ResBlock(channels),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, message_length)
        )
    
    def forward(self, image):
        logits = self.decoder(image)
        return logits


class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma
    
    def forward(self, x):
        if self.training:
            return x + torch.randn_like(x) * self.sigma
        return x


class HiDDeNTrainer:
    def __init__(self, encoder, decoder, message_length=32):
        self.encoder = encoder
        self.decoder = decoder
        self.message_length = message_length
        
        # Loss components
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
    
    def compute_loss(self, image, message):
        """Compute combined loss"""
        # Encode
        watermarked = self.encoder(image, message)
        
        # Decode
        decoded_logits = self.decoder(watermarked)
        
        # Message reconstruction loss
        message_loss = self.bce_loss(decoded_logits, message.float())
        
        # Image quality loss (imperceptibility)
        image_loss = self.mse_loss(watermarked, image)
        
        # Perceptual loss (optional)
        perceptual_loss = self.perceptual_loss(watermarked, image)
        
        # Combined loss
        total_loss = (
            message_loss * 1.0 +
            image_loss * 100.0 +  # High weight for imperceptibility
            perceptual_loss * 10.0
        )
        
        return total_loss, {
            "message_loss": message_loss.item(),
            "image_loss": image_loss.item(),
            "perceptual_loss": perceptual_loss.item()
        }
    
    def perceptual_loss(self, generated, target):
        """Perceptual loss using VGG features"""
        # Simplified - use pre-trained VGG
        # In practice, use LPIPS or similar
        return self.mse_loss(generated, target) * 0.1
```

#### 1.1.2 RivaGAN (Robust Watermarking)

```python
# RivaGAN-style robust watermarking
class RivaGANWatermarker:
    def __init__(self, message_length=32):
        self.message_length = message_length
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.attacks = self._build_attacks()
    
    def _build_encoder(self):
        """Build encoder with residual connections"""
        return nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )
    
    def _build_decoder(self):
        """Build robust decoder"""
        return nn.Sequential(
            nn.Conv2d(3, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, self.message_length)
        )
    
    def _build_attacks(self):
        """Build differentiable attack layers"""
        return nn.ModuleDict({
            "jpeg": JPEGCompression(quality_range=(50, 100)),
            "resize": RandomResize(scale_range=(0.5, 2.0)),
            "crop": RandomCrop(margin=0.1),
            "noise": GaussianNoise(sigma=0.05),
            "rotation": RandomRotation(degrees=10),
            "brightness": RandomBrightness(factor=0.2),
            "contrast": RandomContrast(factor=0.2),
        })
    
    def train_step(self, image, message, attack_prob=0.5):
        """Training step with random attacks"""
        # Encode
        watermarked = image + self.encoder(image) * 0.1
        
        # Apply random attacks
        attacked = watermarked
        for name, attack in self.attacks.items():
            if torch.rand(1).item() < attack_prob:
                attacked = attack(attacked)
        
        # Decode
        decoded = self.decoder(attacked)
        
        # Losses
        message_loss = F.binary_cross_entropy_with_logits(decoded, message.float())
        quality_loss = F.mse_loss(watermarked, image)
        
        return message_loss + 0.5 * quality_loss
```

### 1.2 Frequency-Domain Watermarking

#### 1.2.1 DWT-DCT Hybrid Watermarking

```python
# DWT-DCT hybrid watermarking
import pywt
import numpy as np
from scipy.fftpack import dct, idct

class DWTDCTWatermarker:
    def __init__(self, key, strength=0.1):
        self.key = key
        self.strength = strength
    
    def embed(self, image, watermark_bits):
        """Embed watermark using DWT-DCT hybrid"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.float32)
        else:
            gray = image.astype(np.float32)
        
        # Level-2 DWT
        coeffs2 = pywt.dwt2(gray, 'haar')
        LL, (LH, HL, HH) = coeffs2
        
        # Apply DCT to high-frequency subbands
        LH_dct = dct(dct(LH.T, norm='ortho').T, norm='ortho')
        HL_dct = dct(dct(HL.T, norm='ortho').T, norm='ortho')
        HH_dct = dct(dct(HH.T, norm='ortho').T, norm='ortho')
        
        # Embed in mid-frequency DCT coefficients
        rng = np.random.RandomState(self.key)
        
        for subband, dct_coeffs in [(LH, LH_dct), (HL, HL_dct), (HH, HH_dct)]:
            # Select mid-frequency positions (zigzag pattern)
            positions = self._get_mid_frequency_positions(dct_coeffs.shape)
            
            for i, bit in enumerate(watermark_bits[:len(positions)]):
                r, c = positions[i]
                # Quantization-based embedding
                dct_coeffs[r, c] = self._quantize_embed(
                    dct_coeffs[r, c], bit, self.strength
                )
        
        # Inverse DCT
        LH_reconstructed = idct(idct(LH_dct.T, norm='ortho').T, norm='ortho')
        HL_reconstructed = idct(idct(HL_dct.T, norm='ortho').T, norm='ortho')
        HH_reconstructed = idct(idct(HH_dct.T, norm='ortho').T, norm='ortho')
        
        # Inverse DWT
        watermarked_gray = pywt.idwt2(
            (LL, (LH_reconstructed, HL_reconstructed, HH_reconstructed)),
            'haar'
        )
        
        # Apply to color channels
        if len(image.shape) == 3:
            watermarked = np.stack([
                watermarked_gray + (image[:, :, c] - gray)
                for c in range(3)
            ], axis=2)
        else:
            watermarked = watermarked_gray
        
        return np.clip(watermarked, 0, 255).astype(np.uint8)
    
    def _get_mid_frequency_positions(self, shape, num_positions=64):
        """Get zigzag mid-frequency positions"""
        h, w = shape
        positions = []
        
        # Zigzag scan pattern
        for sum_idx in range(h + w - 1):
            if sum_idx % 2 == 0:
                for i in range(max(0, sum_idx - w + 1), min(sum_idx + 1, h)):
                    j = sum_idx - i
                    if j < w and 3 < i < h - 3 and 3 < j < w - 3:
                        positions.append((i, j))
            else:
                for i in range(min(sum_idx + 1, h) - 1, max(0, sum_idx - w), -1):
                    j = sum_idx - i
                    if j < w and 3 < i < h - 3 and 3 < j < w - 3:
                        positions.append((i, j))
        
        return positions[:num_positions]
    
    def _quantize_embed(self, coeff, bit, strength):
        """Embed bit using quantization index modulation"""
        step = strength * 255
        q = np.round(coeff / step)
        
        if bit == 1:
            q = q + (q % 2 == 0)  # Make odd
        else:
            q = q - (q % 2 == 1)  # Make even
        
        return q * step
    
    def detect(self, watermarked_image, watermark_bits):
        """Detect watermark"""
        if len(watermarked_image.shape) == 3:
            gray = np.mean(watermarked_image, axis=2).astype(np.float32)
        else:
            gray = watermarked_image.astype(np.float32)
        
        coeffs2 = pywt.dwt2(gray, 'haar')
        LL, (LH, HL, HH) = coeffs2
        
        detected_bits = []
        for subband in [LH, HL, HH]:
            dct_coeffs = dct(dct(subband.T, norm='ortho').T, norm='ortho')
            positions = self._get_mid_frequency_positions(dct_coeffs.shape)
            
            for r, c in positions[:len(watermark_bits) // 3]:
                bit = self._quantize_detect(dct_coeffs[r, c])
                detected_bits.append(bit)
        
        # Compare with expected
        matches = sum(d == e for d, e in zip(detected_bits, watermark_bits))
        accuracy = matches / len(watermark_bits)
        
        return {
            "detected": accuracy > 0.7,
            "accuracy": accuracy,
            "matches": matches,
            "total": len(watermark_bits)
        }
    
    def _quantize_detect(self, coeff):
        """Detect bit from quantized coefficient"""
        step = self.strength * 255
        q = np.round(coeff / step)
        return int(q % 2)
```

#### 1.2.2 DFT-Based Watermarking

```python
# DFT-based watermarking
import numpy as np
from scipy.fftpack import fft2, ifft2

class DFTWatermarker:
    def __init__(self, key, strength=0.1):
        self.key = key
        self.strength = strength
    
    def embed(self, image, watermark):
        """Embed watermark in DFT magnitude"""
        # Convert to float
        img_float = image.astype(np.float32)
        
        # Compute 2D DFT
        dft = fft2(img_float, axes=(0, 1))
        
        # Get magnitude and phase
        magnitude = np.abs(dft)
        phase = np.angle(dft)
        
        # Resize watermark to match
        wm_resized = np.array(watermark, dtype=np.float32)
        if wm_resized.shape != magnitude.shape:
            wm_resized = np.resize(wm_resized, magnitude.shape)
        
        # Embed in magnitude (symmetric for real-valued images)
        h, w = magnitude.shape[:2]
        center_h, center_w = h // 2, w // 2
        
        # Create circular embedding region
        Y, X = np.ogrid[:h, :w]
        mask = ((Y - center_h)**2 + (X - center_w)**2) < (min(h, w) * 0.3)**2
        
        # Embed watermark
        magnitude[mask] += self.strength * wm_resized[mask] * magnitude[mask].mean()
        
        # Reconstruct DFT
        dft_watermarked = magnitude * np.exp(1j * phase)
        
        # Inverse DFT
        watermarked = np.real(ifft2(dft_watermarked, axes=(0, 1)))
        
        return np.clip(watermarked, 0, 255).astype(np.uint8)
    
    def detect(self, watermarked_image, original_image, watermark):
        """Detect watermark using DFT correlation"""
        # Compute DFTs
        dft_wm = fft2(watermarked_image.astype(np.float32), axes=(0, 1))
        dft_orig = fft2(original_image.astype(np.float32), axes=(0, 1))
        
        # Extract watermark (difference in magnitude)
        magnitude_wm = np.abs(dft_wm)
        magnitude_orig = np.abs(dft_orig)
        
        extracted = (magnitude_wm - magnitude_orig) / (self.strength * magnitude_orig.mean() + 1e-8)
        
        # Correlate with original watermark
        wm_resized = np.resize(np.array(watermark, dtype=np.float32), extracted.shape)
        
        correlation = np.mean(extracted * wm_resized)
        
        return {
            "detected": abs(correlation) > 0.3,
            "correlation": correlation,
            "confidence": min(abs(correlation) / 0.3, 1.0)
        }
```

---

## 2. Text Watermarking: Statistical Foundations

### 2.1 Theoretical Framework

The statistical foundation of text watermarking:

```python
# Statistical framework for text watermarking
import numpy as np
from scipy import stats

class TextWatermarkStatistics:
    def __init__(self, vocab_size, green_ratio=0.5):
        self.vocab_size = vocab_size
        self.green_ratio = green_ratio
        self.green_size = int(vocab_size * green_ratio)
    
    def compute_detection_power(self, text_length, alpha=0.05):
        """Compute statistical power of watermark detection"""
        # Under null hypothesis (no watermark):
        # Green tokens follow Binomial(n, green_ratio)
        
        n = text_length
        p = self.green_ratio
        
        # Expected green tokens under null
        expected = n * p
        std = np.sqrt(n * p * (1 - p))
        
        # Critical value for significance level alpha
        z_alpha = stats.norm.ppf(1 - alpha)
        critical_value = expected + z_alpha * std
        
        # Under alternative (watermark present):
        # Green tokens have probability p + delta
        # where delta depends on watermark strength
        
        # For SynthID-style watermarking, delta ≈ 0.1-0.2
        delta = 0.15
        
        # Power = P(reject null | watermark present)
        power = 1 - stats.norm.cdf(
            (critical_value - n * (p + delta)) / std
        )
        
        return {
            "text_length": n,
            "significance_level": alpha,
            "power": power,
            "critical_green_count": critical_value,
            "expected_green_no_wm": expected,
            "expected_green_with_wm": n * (p + delta)
        }
    
    def compute_minimum_text_length(self, target_power=0.95, alpha=0.05):
        """Compute minimum text length for reliable detection"""
        delta = 0.15
        z_alpha = stats.norm.ppf(1 - alpha)
        z_beta = stats.norm.ppf(target_power)
        
        # Sample size formula for two proportions
        p1 = self.green_ratio
        p2 = self.green_ratio + delta
        
        n = ((z_alpha * np.sqrt(2 * p1 * (1 - p1)) + 
              z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) / (p2 - p1))**2
        
        return int(np.ceil(n))
    
    def analyze_watermark_strength(self, green_ratio_observed, text_length):
        """Analyze watermark strength from observed data"""
        # Compute z-score
        expected = text_length * self.green_ratio
        std = np.sqrt(text_length * self.green_ratio * (1 - self.green_ratio))
        
        z_score = (green_ratio_observed * text_length - expected) / std
        
        # Compute p-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # Compute effect size (Cohen's h)
        h = 2 * (np.arcsin(np.sqrt(green_ratio_observed)) - 
                 np.arcsin(np.sqrt(self.green_ratio)))
        
        return {
            "z_score": z_score,
            "p_value": p_value,
            "effect_size_h": h,
            "significant_at_0001": p_value < 0.001,
            "estimated_watermark_strength": h
        }
```

### 2.2 Robustness Analysis

```python
# Watermark robustness against paraphrasing
class ParaphraseRobustnessAnalyzer:
    def __init__(self, watermarker):
        self.watermarker = watermarker
    
    def analyze_robustness(self, original_text, paraphrases):
        """Analyze watermark survival through paraphrasing"""
        results = []
        
        for paraphrase in paraphrases:
            # Detect watermark in paraphrase
            detection = self.watermarker.detect(paraphrase)
            
            # Compute semantic similarity
            similarity = self.compute_similarity(original_text, paraphrase)
            
            results.append({
                "paraphrase": paraphrase[:100] + "...",
                "watermark_detected": detection["detected"],
                "detection_confidence": detection.get("confidence", 0),
                "semantic_similarity": similarity,
                "survived": detection["detected"] and similarity > 0.7
            })
        
        survival_rate = sum(r["survived"] for r in results) / len(results)
        
        return {
            "num_paraphrases": len(paraphrases),
            "survival_rate": survival_rate,
            "results": results,
            "recommendation": "Robust" if survival_rate > 0.8 else "Vulnerable"
        }
    
    def compute_similarity(self, text1, text2):
        """Compute semantic similarity"""
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        emb1 = model.encode(text1)
        emb2 = model.encode(text2)
        
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
```

---

## 3. Image Forensics

### 3.1 GAN/Diffusion Detection

```python
# GAN and diffusion model detection
import torch
import torch.nn as nn
from torchvision import models

class GANDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # Use pre-trained ResNet as backbone
        self.backbone = models.resnet50(pretrained=True)
        
        # Replace classifier
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.backbone.features(x)
        features = nn.AdaptiveAvgPool2d(1)(features)
        features = features.view(features.size(0), -1)
        return self.classifier(features)


class DiffusionDetector(nn.Module):
    """Specialized detector for diffusion model outputs"""
    def __init__(self):
        super().__init__()
        
        # Frequency analysis branch
        self.freq_branch = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64)
        )
        
        # Spatial analysis branch
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(3, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64)
        )
        
        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Compute frequency features
        x_freq = self.compute_fft_features(x)
        
        # Analyze both domains
        freq_features = self.freq_branch(x_freq)
        spatial_features = self.spatial_branch(x)
        
        # Fuse features
        combined = torch.cat([freq_features, spatial_features], dim=1)
        
        return self.classifier(combined)
    
    def compute_fft_features(self, x):
        """Compute FFT features for frequency analysis"""
        # Convert to grayscale
        gray = 0.2989 * x[:, 0] + 0.5870 * x[:, 1] + 0.1140 * x[:, 2]
        
        # Compute FFT
        fft = torch.fft.fft2(gray)
        fft_shift = torch.fft.fftshift(fft)
        magnitude = torch.log(torch.abs(fft_shift) + 1)
        
        # Convert back to 3-channel for CNN
        magnitude = magnitude.unsqueeze(1).repeat(1, 3, 1, 1)
        
        return magnitude
```

### 3.2 Deepfake Detection

```python
# Deepfake detection pipeline
class DeepfakeDetector:
    def __init__(self):
        self.face_detector = self._load_face_detector()
        self.deepfake_model = self._load_deepfake_model()
        self.lip_sync_detector = self._load_lip_sync_detector()
    
    def detect(self, video_path):
        """Full deepfake detection pipeline"""
        results = {
            "face_manipulation": None,
            "lip_sync": None,
            "temporal_consistency": None,
            "overall_score": 0
        }
        
        # Extract frames
        frames = self.extract_frames(video_path)
        
        # Detect faces
        faces = self.detect_faces(frames)
        
        # Analyze each face
        face_scores = []
        for face in faces:
            # Check for face manipulation
            manipulation_score = self.deepfake_model(face)
            face_scores.append(manipulation_score)
        
        results["face_manipulation"] = {
            "num_faces": len(faces),
            "manipulation_scores": face_scores,
            "avg_score": np.mean(face_scores),
            "max_score": np.max(face_scores)
        }
        
        # Check lip sync (audio-visual consistency)
        audio = self.extract_audio(video_path)
        lip_sync_score = self.lip_sync_detector.detect(frames, audio)
        results["lip_sync"] = lip_sync_score
        
        # Check temporal consistency
        temporal_score = self.check_temporal_consistency(frames)
        results["temporal_consistency"] = temporal_score
        
        # Overall score
        results["overall_score"] = np.mean([
            results["face_manipulation"]["avg_score"],
            lip_sync_score.get("consistency", 0.5),
            temporal_score.get("consistency", 0.5)
        ])
        
        results["is_deepfake"] = results["overall_score"] > 0.7
        
        return results
    
    def check_temporal_consistency(self, frames):
        """Check for temporal inconsistencies in video"""
        inconsistencies = []
        
        for i in range(1, len(frames)):
            # Compute optical flow
            flow = cv2.calcOpticalFlowFarneback(
                frames[i-1], frames[i], None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            # Check for unnatural motion
            mean_flow = np.mean(np.abs(flow))
            std_flow = np.std(np.abs(flow))
            
            # Deepfakes often have inconsistent motion
            if std_flow / mean_flow < 0.1:  # Too consistent
                inconsistencies.append(i)
        
        return {
            "num_frames": len(frames),
            "inconsistent_frames": inconsistencies,
            "consistency": 1 - len(inconsistencies) / len(frames)
        }
```

### 3.3 Copy-Move Detection

```python
# Copy-move forgery detection
class CopyMoveDetector:
    def __init__(self):
        self.feature_extractor = self._load_sift()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    def detect(self, image):
        """Detect copy-move forgery"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Extract SIFT features
        kp, des = self.feature_extractor.detectAndCompute(gray, None)
        
        if des is None or len(kp) < 10:
            return {"detected": False, "reason": "insufficient_features"}
        
        # Match features
        matches = self.matcher.match(des, des)
        
        # Filter good matches
        good_matches = [m for m in matches if m.distance < 50 and m.queryIdx != m.trainIdx]
        
        if len(good_matches) < 10:
            return {"detected": False, "reason": "insufficient_matches"}
        
        # Find connected components of matching regions
        regions = self.find_copy_regions(kp, good_matches)
        
        # Verify regions
        for region_pair in regions:
            if self.verify_region_pair(image, region_pair):
                return {
                    "detected": True,
                    "confidence": len(good_matches) / len(matches),
                    "regions": region_pair,
                    "num_matches": len(good_matches)
                }
        
        return {"detected": False, "reason": "no_verified_regions"}
    
    def find_copy_regions(self, keypoints, matches):
        """Find connected regions of copied content"""
        # Build adjacency graph
        graph = {}
        for match in matches:
            i, j = match.queryIdx, match.trainIdx
            if i not in graph:
                graph[i] = []
            graph[j] = []
            graph[i].append(j)
            graph[j].append(i)
        
        # Find connected components
        visited = set()
        regions = []
        
        for node in graph:
            if node not in visited:
                region = []
                stack = [node]
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        region.append(current)
                        stack.extend([n for n in graph[current] if n not in visited])
                
                if len(region) > 10:  # Minimum region size
                    regions.append(region)
        
        # Pair regions
        region_pairs = []
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                region_pairs.append((regions[i], regions[j]))
        
        return region_pairs[:5]  # Return top 5 pairs
    
    def verify_region_pair(self, image, region_pair):
        """Verify that regions are actually copies"""
        # Extract regions
        kp1, kp2 = region_pair
        
        # Compute region descriptors
        # (simplified - real implementation uses more sophisticated checks)
        
        return len(kp1) > 20 and len(kp2) > 20
```

---

## 4. Cryptographic Infrastructure

### 4.1 Certificate Transparency for Content

```python
# Certificate Transparency for content provenance
import hashlib
import time
from typing import List, Dict

class ContentCertificateTransparency:
    def __init__(self):
        self.merkle_tree = MerkleTree()
        self.sct_log = []  # Signed Certificate Timestamps
    
    def submit_manifest(self, manifest, signature):
        """Submit C2PA manifest to transparency log"""
        # Create SCT
        sct = {
            "timestamp": int(time.time() * 1000),
            "manifest_hash": hashlib.sha256(
                json.dumps(manifest, sort_keys=True).encode()
            ).hexdigest(),
            "signature": signature,
            "log_index": len(self.sct_log)
        }
        
        # Add to Merkle tree
        leaf_hash = self.merkle_tree.add_leaf(sct)
        
        # Create inclusion proof
        proof = self.merkle_tree.get_inclusion_proof(leaf_hash)
        
        sct["inclusion_proof"] = proof
        self.sct_log.append(sct)
        
        return sct
    
    def verify_inclusion(self, sct):
        """Verify SCT is included in transparency log"""
        # Reconstruct Merkle tree
        tree = MerkleTree()
        for entry in self.sct_log:
            tree.add_leaf(entry)
        
        # Verify proof
        return tree.verify_proof(sct["inclusion_proof"], sct)
    
    def audit_log(self):
        """Audit transparency log for consistency"""
        # Check Merkle tree consistency
        tree_hash = self.merkle_tree.get_root()
        
        # Verify no entries were removed
        for i, sct in enumerate(self.sct_log):
            if not self.verify_inclusion(sct):
                return {
                    "consistent": False,
                    "error": f"Entry {i} not in tree",
                    "tree_hash": tree_hash
                }
        
        return {
            "consistent": True,
            "num_entries": len(self.sct_log),
            "tree_hash": tree_hash
        }


class MerkleTree:
    def __init__(self):
        self.leaves = []
        self.tree = []
    
    def add_leaf(self, data):
        """Add leaf to Merkle tree"""
        leaf_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()
        self.leaves.append(leaf_hash)
        self._rebuild_tree()
        return leaf_hash
    
    def _rebuild_tree(self):
        """Rebuild Merkle tree"""
        if not self.leaves:
            self.tree = []
            return
        
        # Start with leaves
        level = self.leaves[:]
        
        while len(level) > 1:
            next_level = []
            for i in range(0, len(level), 2):
                if i + 1 < len(level):
                    combined = hashlib.sha256(
                        (level[i] + level[i + 1]).encode()
                    ).hexdigest()
                else:
                    combined = level[i]
                next_level.append(combined)
            level = next_level
        
        self.tree = [self.leaves[:]] + self._build_levels()
    
    def _build_levels(self):
        """Build intermediate levels"""
        levels = []
        level = self.leaves[:]
        
        while len(level) > 1:
            next_level = []
            for i in range(0, len(level), 2):
                if i + 1 < len(level):
                    combined = hashlib.sha256(
                        (level[i] + level[i + 1]).encode()
                    ).hexdigest()
                else:
                    combined = level[i]
                next_level.append(combined)
            levels.append(next_level)
            level = next_level
        
        return levels
    
    def get_root(self):
        """Get Merkle tree root"""
        if self.tree:
            return self.tree[-1][0]
        return None
    
    def get_inclusion_proof(self, leaf_hash):
        """Get inclusion proof for leaf"""
        index = self.leaves.index(leaf_hash)
        proof = []
        
        for level in self.tree[:-1]:
            if index % 2 == 0:
                sibling = index + 1 if index + 1 < len(level) else index
            else:
                sibling = index - 1
            
            proof.append({
                "hash": level[sibling],
                "position": "right" if index % 2 == 0 else "left"
            })
            
            index //= 2
        
        return proof
    
    def verify_proof(self, proof, sct):
        """Verify inclusion proof"""
        current_hash = hashlib.sha256(
            json.dumps(sct, sort_keys=True).encode()
        ).hexdigest()
        
        for step in proof:
            if step["position"] == "right":
                combined = current_hash + step["hash"]
            else:
                combined = step["hash"] + current_hash
            
            current_hash = hashlib.sha256(combined.encode()).hexdigest()
        
        return current_hash == self.get_root()
```

### 4.2 Blockchain Provenance Registry

```python
# Blockchain-based provenance registry
import hashlib
import time
import json

class ProvenanceBlockchain:
    def __init__(self):
        self.chain = []
        self.pending_transactions = []
        self.difficulty = 4  # Mining difficulty
        
        # Create genesis block
        self.create_genesis_block()
    
    def create_genesis_block(self):
        """Create genesis block"""
        genesis_block = {
            "index": 0,
            "timestamp": int(time.time()),
            "transactions": [],
            "previous_hash": "0" * 64,
            "nonce": 0
        }
        genesis_block["hash"] = self.compute_hash(genesis_block)
        self.chain.append(genesis_block)
    
    def add_provenance_record(self, content_hash, metadata):
        """Add provenance record to blockchain"""
        transaction = {
            "type": "provenance_record",
            "content_hash": content_hash,
            "metadata": metadata,
            "timestamp": int(time.time()),
            "sender": metadata.get("creator", "anonymous")
        }
        
        self.pending_transactions.append(transaction)
        
        # Mine block if enough transactions
        if len(self.pending_transactions) >= 10:
            return self.mine_block()
        
        return {"status": "pending", "transaction": transaction}
    
    def mine_block(self):
        """Mine new block with pending transactions"""
        block = {
            "index": len(self.chain),
            "timestamp": int(time.time()),
            "transactions": self.pending_transactions[:],
            "previous_hash": self.chain[-1]["hash"],
            "nonce": 0
        }
        
        # Proof of work
        while not self.is_valid_proof(block):
            block["nonce"] += 1
        
        block["hash"] = self.compute_hash(block)
        
        # Add to chain
        self.chain.append(block)
        self.pending_transactions = self.pending_transactions[10:]
        
        return {
            "status": "mined",
            "block": block,
            "chain_length": len(self.chain)
        }
    
    def verify_provenance(self, content_hash):
        """Verify provenance record on blockchain"""
        for block in self.chain:
            for tx in block["transactions"]:
                if tx.get("content_hash") == content_hash:
                    return {
                        "found": True,
                        "block_index": block["index"],
                        "timestamp": tx["timestamp"],
                        "metadata": tx["metadata"],
                        "block_hash": block["hash"]
                    }
        
        return {"found": False}
    
    def compute_hash(self, block):
        """Compute block hash"""
        block_string = json.dumps(block, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def is_valid_proof(self, block):
        """Check if block meets difficulty requirement"""
        hash = self.compute_hash(block)
        return hash[:self.difficulty] == "0" * self.difficulty
    
    def validate_chain(self):
        """Validate entire blockchain"""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]
            
            # Verify hash
            if current["hash"] != self.compute_hash(current):
                return False, f"Invalid hash at block {i}"
            
            # Verify chain link
            if current["previous_hash"] != previous["hash"]:
                return False, f"Broken chain at block {i}"
            
            # Verify proof of work
            if not self.is_valid_proof(current):
                return False, f"Invalid proof at block {i}"
        
        return True, "Valid chain"
```

---

## 5. Adversarial Robustness

### 5.1 Watermark Removal Attacks

```python
# Common watermark removal attacks
class WatermarkRemovalAttacks:
    def __init__(self):
        self.attacks = {
            "jpeg_compression": self.jpeg_attack,
            "gaussian_blur": self.blur_attack,
            "gaussian_noise": self.noise_attack,
            "scaling": self.scale_attack,
            "cropping": self.crop_attack,
            "rotation": self.rotation_attack,
            "color_jitter": self.color_attack,
            "adversarial_perturbation": self.adversarial_attack
        }
    
    def jpeg_attack(self, image, quality=50):
        """JPEG compression removes watermarks"""
        import io
        from PIL import Image
        
        pil_img = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        attacked = np.array(Image.open(buffer))
        
        return attacked
    
    def blur_attack(self, image, kernel_size=5):
        """Gaussian blur removes high-frequency watermarks"""
        import cv2
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def noise_attack(self, image, sigma=10):
        """Additive noise disrupts watermarks"""
        noise = np.random.normal(0, sigma, image.shape)
        attacked = image.astype(np.float32) + noise
        return np.clip(attacked, 0, 255).astype(np.uint8)
    
    def scale_attack(self, image, scale=0.5):
        """Scaling removes spatial watermarks"""
        import cv2
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Downscale then upscale
        small = cv2.resize(image, (new_w, new_h))
        attacked = cv2.resize(small, (w, h))
        
        return attacked
    
    def crop_attack(self, image, margin=0.1):
        """Cropping removes border watermarks"""
        h, w = image.shape[:2]
        m_h, m_w = int(h * margin), int(w * margin)
        
        # Crop and resize
        cropped = image[m_h:h-m_h, m_w:w-m_w]
        attacked = cv2.resize(cropped, (w, h))
        
        return attacked
    
    def rotation_attack(self, image, angle=5):
        """Rotation disrupts watermark alignment"""
        import cv2
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        attacked = cv2.warpAffine(image, M, (w, h))
        
        return attacked
    
    def color_attack(self, image):
        """Color jitter disrupts color-based watermarks"""
        import cv2
        
        # Random brightness
        brightness = np.random.uniform(0.8, 1.2)
        image = np.clip(image * brightness, 0, 255).astype(np.uint8)
        
        # Random contrast
        contrast = np.random.uniform(0.8, 1.2)
        mean = np.mean(image)
        image = np.clip((image - mean) * contrast + mean, 0, 255).astype(np.uint8)
        
        return image
    
    def adversarial_attack(self, image, watermarker, message_length=32):
        """Gradient-based attack to remove watermark"""
        # This is a simplified version
        # Real implementation uses PGD or similar
        
        import torch
        
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        image_tensor.requires_grad = True
        
        optimizer = torch.optim.Adam([image_tensor], lr=0.01)
        
        for _ in range(100):
            optimizer.zero_grad()
            
            # Try to decode watermark
            decoded = watermarker.decoder(image_tensor)
            
            # Loss: maximize decoded entropy (remove watermark signal)
            loss = -torch.mean(torch.abs(decoded - 0.5))
            
            loss.backward()
            optimizer.step()
        
        attacked = (image_tensor.detach().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return attacked
    
    def apply_random_attack(self, image, num_attacks=3):
        """Apply random combination of attacks"""
        attacked = image.copy()
        applied_attacks = []
        
        for _ in range(num_attacks):
            attack_name = np.random.choice(list(self.attacks.keys()))
            attack_fn = self.attacks[attack_name]
            
            if attack_name == "jpeg_compression":
                attacked = attack_fn(attacked, quality=np.random.randint(30, 80))
            elif attack_name == "gaussian_blur":
                attacked = attack_fn(attacked, kernel_size=np.random.choice([3, 5, 7]))
            elif attack_name == "gaussian_noise":
                attacked = attack_fn(attacked, sigma=np.random.uniform(5, 20))
            else:
                attacked = attack_fn(attacked)
            
            applied_attacks.append(attack_name)
        
        return attacked, applied_attacks
```

### 5.2 Defense Against Attacks

```python
# Multi-layered defense against watermark removal
class WatermarkDefense:
    def __init__(self, watermarker):
        self.watermarker = watermarker
        self.defense_strategies = [
            RedundantWatermarking(),
            FrequencyDomainEmbedding(),
            SemanticWatermarking(),
            TemporalWatermarking()
        ]
    
    def defend(self, image, message):
        """Apply multiple defense layers"""
        watermarked = image.copy()
        
        for strategy in self.defense_strategies:
            watermarked = strategy.embed(watermarked, message)
        
        return watermarked
    
    def verify_defense(self, attacked_image, original_message):
        """Verify watermark survives attacks"""
        results = []
        
        for strategy in self.defense_strategies:
            detected = strategy.detect(attacked_image, original_message)
            results.append({
                "strategy": strategy.name,
                "detected": detected["detected"],
                "confidence": detected.get("confidence", 0)
            })
        
        # Overall verification
        any_detected = any(r["detected"] for r in results)
        avg_confidence = np.mean([r["confidence"] for r in results])
        
        return {
            "watermark_survives": any_detected,
            "average_confidence": avg_confidence,
            "strategy_results": results
        }


class RedundantWatermarking:
    """Embed watermark in multiple domains"""
    name = "Redundant Watermarking"
    
    def embed(self, image, message):
        # Embed in spatial domain
        # Embed in frequency domain
        # Embed in color channels separately
        return image  # Simplified
    
    def detect(self, image, message):
        # Check all domains
        return {"detected": True, "confidence": 0.9}


class FrequencyDomainEmbedding:
    """Embed in robust frequency bands"""
    name = "Frequency Domain Embedding"
    
    def embed(self, image, message):
        # Embed in DCT mid-frequencies
        return image  # Simplified
    
    def detect(self, image, message):
        return {"detected": True, "confidence": 0.85}


class SemanticWatermarking:
    """Embed in semantic features"""
    name = "Semantic Watermarking"
    
    def embed(self, image, message):
        # Embed in high-level features
        return image  # Simplified
    
    def detect(self, image, message):
        return {"detected": True, "confidence": 0.8}


class TemporalWatermarking:
    """Embed in temporal patterns (for video)"""
    name = "Temporal Watermarking"
    
    def embed(self, image, message):
        # Embed temporal variations
        return image  # Simplified
    
    def detect(self, image, message):
        return {"detected": True, "confidence": 0.75}
```

---

## 6. Performance Benchmarks

### 6.1 Watermarking Performance

| Method | Robustness (JPEG Q50) | Robustness (Resize 0.5) | PSNR (dB) | SSIM | Detection Accuracy |
|--------|----------------------|------------------------|-----------|------|-------------------|
| LSB | 20% | 10% | 51.2 | 0.99 | 95% |
| DCT-QIM | 75% | 60% | 38.5 | 0.95 | 92% |
| DWT-SS | 80% | 70% | 40.2 | 0.96 | 94% |
| HiDDeN | 92% | 85% | 42.1 | 0.97 | 98% |
| RivaGAN | 95% | 90% | 41.5 | 0.96 | 99% |
| SynthID | 98% | 95% | 43.0 | 0.98 | 99.5% |

### 6.2 Detection Performance

| Method | Precision | Recall | F1 Score | AUC-ROC | Inference Time |
|--------|-----------|--------|----------|---------|----------------|
| Statistical | 0.72 | 0.68 | 0.70 | 0.75 | 10ms |
| CNN-based | 0.88 | 0.85 | 0.86 | 0.92 | 50ms |
| Transformer | 0.92 | 0.90 | 0.91 | 0.95 | 150ms |
| Ensemble | 0.95 | 0.93 | 0.94 | 0.97 | 200ms |

---

## 7. Key Takeaways

1. **Neural watermarking (HiDDeN, RivaGAN) offers best robustness-imperceptibility tradeoff**
2. **Frequency domain methods (DWT-DCT) are robust to common attacks**
3. **Statistical text watermarking requires ~256 tokens for reliable detection**
4. **GAN/diffusion detection accuracy reaches 95%+ with ensemble methods**
5. **Certificate transparency provides auditability for provenance systems**
6. **Blockchain provenance adds decentralization but at computational cost**
7. **Adversarial robustness requires multi-layer defense strategies**
8. **PSNR > 40dB and SSIM > 0.95 are targets for imperceptible watermarking**
9. **Detection latency < 100ms is required for real-time verification**
10. **The arms race between watermarking and removal continues indefinitely**

---

*Last updated: June 30, 2026*
*See also: [01-Overview.md](01-Overview.md) | [02-Core-Topics.md](02-Core-Topics.md) | [04-Tools-and-Frameworks.md](04-Tools-and-Frameworks.md) | [05-Future-Outlook.md](05-Future-Outlook.md)*
