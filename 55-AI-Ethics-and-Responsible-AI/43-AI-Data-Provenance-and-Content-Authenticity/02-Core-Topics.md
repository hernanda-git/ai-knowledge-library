# Core Topics in AI Data Provenance and Content Authenticity

> This document provides a comprehensive exploration of the fundamental topics underpinning AI data provenance and content authenticity, from watermarking techniques to verification infrastructure, platform implementations, and the regulatory landscape.

---

## 1. Content Provenance Architectures

### 1.1 Provenance Chain Architecture

A complete provenance system tracks the full lifecycle of content:

```
┌─────────────────────────────────────────────────────────────────┐
│                    PROVENANCE CHAIN                              │
│                                                                  │
│  Creation ──→ Processing ──→ Distribution ──→ Consumption ──→ Archive │
│     │              │              │               │              │
│     ▼              ▼              ▼               ▼              │
│  Signing      Editing Log    Platform Store   Verification     │
│  Metadata     Versioning     C2PA Manifest    Trust Score     │
│  Watermark    Attribution    Provenance UI    Audit Trail     │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 C2PA Manifest Structure

The C2PA manifest is the core data structure for provenance:

```json
{
  "claim": {
    "generator": {
      "type": "urn:c2pa:ai",
      "name": "DALL-E 4",
      "version": "4.2.1",
      " URI": "https://openai.com/dall-e"
    },
    "signature": {
      "algorithm": "ES384",
      "certificate_chain": "..."
    },
    "timestamp": "2026-06-30T10:30:00Z",
    "actions": [
      {
        "action": "c2pa.created",
        "when": "2026-06-30T10:29:55Z",
        "softwareAgent": "DALL-E 4"
      },
      {
        "action": "c2pa.color_adjusted",
        "when": "2026-06-30T10:29:58Z",
        "softwareAgent": "Adobe Photoshop"
      }
    ],
    "ingredient": [
      {
        "title": "reference_photo.jpg",
        "relationship": "parentOf",
        "instanceID": "uuid:12345678-..."
      }
    ],
    "metadata": {
      "ai_generated": true,
      "ai_model": "dall-e-4",
      "prompt": "A sunset over mountains...",
      "content_credentials": "..."
    }
  },
  "assertions": [
    {
      "type": "c2pa.watermark",
      "data": "...",
      "algorithm": "synthid-v3"
    },
    {
      "type": "c2pa.training_data",
      "data": {
        "license": "openai-commercial",
        "opt_out": false
      }
    }
  ]
}
```

### 1.3 Nested Provenance

Content often combines multiple sources, requiring nested provenance:

```
Final Image
├── AI Generated Layer (DALL-E 4)
│   ├── C2PA Manifest: ai-generated
│   ├── Watermark: SynthID
│   └── Prompt metadata
├── Human Edited Layer (Photoshop)
│   ├── C2PA Manifest: edited
│   ├── Edit history
│   └── Original reference photo
│       └── Source: Shutterstock
│           ├── License info
│           └── Photographer credit
└── Composite C2PA Manifest
    └── Links all nested manifests
```

---

## 2. Watermarking Deep Dive

### 2.1 Watermark Taxonomy

| Type | Method | Robustness | Imperceptibility | Capacity |
|------|--------|-----------|------------------|----------|
| **Spatial** | LSB, spread spectrum | Low | High | High |
| **Frequency** | DCT, DWT, DFT | Medium | High | Medium |
| **Deep Learning** | Neural encoding | High | High | Medium |
| **Semantic** | Meaning-based | Very High | Very High | Low |
| **Metadata** | C2PA, EXIF | Low | N/A | High |
| **Hybrid** | Multi-domain | Very High | High | Medium |

### 2.2 Image Watermarking Techniques

#### 2.2.1 Spread Spectrum Watermarking

```python
# Spread spectrum watermarking
import numpy as np
from scipy.ndimage import gaussian_filter

class SpreadSpectrumWatermarker:
    def __init__(self, key, strength=0.1):
        self.key = key
        self.strength = strength
        self.rng = np.random.RandomState(key)
    
    def generate_watermark_pattern(self, shape):
        """Generate pseudo-random noise pattern"""
        # Create spread spectrum pattern
        pattern = self.rng.randn(*shape).astype(np.float32)
        
        # Apply Gaussian smoothing for imperceptibility
        pattern = gaussian_filter(pattern, sigma=1.0)
        
        # Normalize
        pattern = pattern / np.max(np.abs(pattern))
        return pattern
    
    def embed(self, image):
        """Embed watermark in image"""
        pattern = self.generate_watermark_pattern(image.shape[:2])
        
        # Apply to each color channel
        watermarked = image.astype(np.float32)
        for c in range(3):
            watermarked[:, :, c] += self.strength * pattern * 255
        
        return np.clip(watermarked, 0, 255).astype(np.uint8)
    
    def detect(self, watermarked_image, original_image):
        """Detect watermark using correlation"""
        pattern = self.generate_watermark_pattern(watermarked_image.shape[:2])
        
        # Compute difference
        diff = watermarked_image.astype(np.float32) - original_image.astype(np.float32)
        
        # Correlate with pattern
        correlation = np.mean(diff * pattern)
        
        # Statistical test
        threshold = 0.05
        return {
            "detected": abs(correlation) > threshold,
            "confidence": abs(correlation) / threshold,
            "correlation": correlation
        }
```

#### 2.2.2 DWT-Based Watermarking

```python
# Discrete Wavelet Transform watermarking
import pywt
import numpy as np

class DWTWatermarker:
    def __init__(self, key, strength=0.1):
        self.key = key
        self.strength = strength
    
    def embed(self, image, watermark_bits):
        """Embed watermark in DWT domain"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Apply 2-level DWT
        coeffs2 = pywt.dwt2(gray, 'haar')
        LL, (LH, HL, HH) = coeffs2
        
        # Embed in high-frequency subbands (more robust)
        rng = np.random.RandomState(self.key)
        mask = rng.choice([0, 1], size=LH.shape, p=[0.7, 0.3])
        
        for i, bit in enumerate(watermark_bits[:int(mask.sum())]):
            positions = np.where(mask)
            if i < len(positions[0]):
                r, c = positions[0][i], positions[1][i]
                LH[r, c] += self.strength * bit * 255
        
        # Reconstruct
        watermarked_coeffs = LL, (LH, HL, HH)
        watermarked_gray = pywt.idwt2(watermarked_coeffs, 'haar')
        
        # Apply to all channels
        if len(image.shape) == 3:
            watermarked = np.stack([
                watermarked_gray + (image[:, :, c] - gray)
                for c in range(3)
            ], axis=2)
        else:
            watermarked = watermarked_gray
        
        return np.clip(watermarked, 0, 255).astype(np.uint8)
```

#### 2.2.3 Neural Watermarking (End-to-End)

```python
# End-to-end neural watermarking
import torch
import torch.nn as nn

class NeuralWatermarkEncoder(nn.Module):
    def __init__(self, message_length=64):
        super().__init__()
        self.message_length = message_length
        
        # Message embedding
        self.message_embedding = nn.Embedding(2, 64)
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Conv2d(3 + 64, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, image, message):
        """Encode watermark into image"""
        # Embed message
        embedded = self.message_embedding(message)  # (B, L, 64)
        embedded = embedded.permute(0, 2, 1)  # (B, 64, L)
        embedded = embedded.unsqueeze(-1).expand(-1, -1, -1, image.shape[-1])
        
        # Concatenate with image
        combined = torch.cat([image, embedded], dim=1)
        
        # Generate watermark residual
        residual = self.encoder(combined)
        
        # Apply with strength control
        watermarked = image + 0.1 * residual
        
        return torch.clamp(watermarked, 0, 1)


class NeuralWatermarkDecoder(nn.Module):
    def __init__(self, message_length=64):
        super().__init__()
        self.message_length = message_length
        
        self.decoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, message_length),
            nn.Sigmoid()
        )
    
    def forward(self, image):
        """Decode watermark from image"""
        bits = self.decoder(image)
        return (bits > 0.5).float()
```

### 2.3 Text Watermarking

#### 2.3.1 List-Based Watermarking (Christ et al.)

```python
# List-based text watermarking
import hashlib
import numpy as np

class ListBasedWatermarker:
    def __init__(self, secret_key, green_list_ratio=0.5):
        self.key = secret_key
        self.green_ratio = green_list_ratio
    
    def get_green_list(self, context_tokens):
        """Generate green list based on context"""
        # Hash context to get seed
        context_str = str(context_tokens)
        seed = int(hashlib.sha256(
            (self.key + context_str).encode()
        ).hexdigest(), 16) % (2**32)
        
        rng = np.random.RandomState(seed)
        vocab_size = 50257  # GPT-2 vocab
        
        # Select green list
        green_size = int(vocab_size * self.green_ratio)
        green_indices = set(rng.choice(vocab_size, green_size, replace=False))
        
        return green_indices
    
    def watermark_logits(self, logits, context_tokens):
        """Apply watermark to logits"""
        green_list = self.get_green_list(context_tokens)
        
        watermarked = logits.clone()
        for i in range(len(logits)):
            if i in green_list:
                watermarked[i] += 1.0  # Boost green tokens
            else:
                watermarked[i] -= 1.0  # Suppress red tokens
        
        return watermarked
    
    def detect_watermark(self, tokens):
        """Detect watermark in token sequence"""
        green_count = 0
        total = len(tokens) - 1  # Need context for each token
        
        for i in range(1, len(tokens)):
            context = tokens[:i]
            green_list = self.get_green_list(context)
            if tokens[i] in green_list:
                green_count += 1
        
        # Binomial test
        expected = total * self.green_ratio
        std = np.sqrt(total * self.green_ratio * (1 - self.green_ratio))
        z_score = (green_count - expected) / std
        
        return {
            "detected": z_score > 4.0,  # p < 0.00003
            "z_score": z_score,
            "green_ratio": green_count / total
        }
```

#### 2.3.2 SynthID-Style Watermarking

```python
# SynthID-inspired text watermarking
class SynthIDTextWatermarker:
    def __init__(self, secret_key, num_buckets=32):
        self.key = secret_key
        self.num_buckets = num_buckets
        self.hash_fn = self._create_hash()
    
    def _create_hash(self):
        """Create consistent hash function"""
        def hash_fn(token_id, context_hash):
            return (token_id * 2654435761 + context_hash) % self.num_buckets
        return hash_fn
    
    def get_bucket_scores(self, logits, previous_tokens):
        """Assign tokens to buckets and score"""
        # Compute context hash
        context_hash = 0
        for t in previous_tokens[-32:]:  # Last 32 tokens
            context_hash = (context_hash * 31 + t) % (2**32)
        
        # Score each token
        scores = torch.zeros_like(logits)
        for token_id in range(logits.shape[-1]):
            bucket = self.hash_fn(token_id, context_hash)
            # Green buckets get boost, red get penalty
            if bucket < self.num_buckets // 2:
                scores[token_id] = 1.0
            else:
                scores[token_id] = -1.0
        
        return logits + 0.5 * scores
    
    def detect(self, tokens, window_size=256):
        """Detect watermark using sliding window"""
        detections = []
        
        for start in range(0, len(tokens) - window_size, window_size // 2):
            window = tokens[start:start + window_size]
            
            # Count green/red
            green_count = 0
            for i in range(1, len(window)):
                context_hash = 0
                for t in window[max(0, i-32):i]:
                    context_hash = (context_hash * 31 + t) % (2**32)
                
                bucket = self.hash_fn(window[i], context_hash)
                if bucket < self.num_buckets // 2:
                    green_count += 1
            
            expected = window_size * 0.5
            std = np.sqrt(window_size * 0.25)
            z = (green_count - expected) / std
            detections.append(z)
        
        # Aggregate detections
        mean_z = np.mean(detections)
        return {
            "detected": mean_z > 4.0,
            "mean_z_score": mean_z,
            "num_windows": len(detections)
        }
```

### 2.4 Audio Watermarking

#### 2.4.1 Frequency-Domain Audio Watermarking

```python
# Audio watermarking in frequency domain
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, istft

class AudioWatermarker:
    def __init__(self, key, strength=0.05):
        self.key = key
        self.strength = strength
    
    def embed(self, audio, sample_rate, message_bits):
        """Embed watermark in audio spectrogram"""
        # Compute STFT
        f, t, Zxx = stft(audio, fs=sample_rate, nperseg=1024)
        
        # Embed in mid-frequency range (2-8 kHz)
        # This range is less audible but carries watermark
        freq_mask = (f >= 2000) & (f <= 8000)
        
        # Generate watermark pattern
        rng = np.random.RandomState(self.key)
        pattern = rng.choice([-1, 1], size=Zxx[freq_mask].shape)
        
        # Embed bits
        for i, bit in enumerate(message_bits[:int(freq_mask.sum())]):
            positions = np.where(pattern)
            if i < len(positions[0]):
                r, c = positions[0][i], positions[1][i]
                Zxx[freq_mask][r, c] += self.strength * bit * np.abs(Zxx[freq_mask][r, c])
        
        # Reconstruct
        _, watermarked = istft(Zxx, fs=sample_rate, nperseg=1024)
        return watermarked.astype(np.float32)
    
    def detect(self, watermarked_audio, sample_rate, message_bits):
        """Detect watermark from audio"""
        f, t, Zxx = stft(watermarked_audio, fs=sample_rate, nperseg=1024)
        
        freq_mask = (f >= 2000) & (f <= 8000)
        
        rng = np.random.RandomState(self.key)
        pattern = rng.choice([-1, 1], size=Zxx[freq_mask].shape)
        
        # Correlate
        correlation = np.mean(Zxx[freq_mask] * pattern)
        
        return {
            "detected": abs(correlation) > self.strength * 0.5,
            "confidence": abs(correlation) / (self.strength * 0.5)
        }
```

#### 2.4.2 Inaudible Neural Audio Watermarking

```python
# Neural audio watermarking (conceptual)
class NeuralAudioWatermarker(nn.Module):
    def __init__(self, message_length=32):
        super().__init__()
        self.message_length = message_length
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, message_length)
        )
    
    def encode(self, audio, message):
        """Encode watermark into audio"""
        # Embed message
        embedded = self._embed_message(audio, message)
        
        # Generate residual
        residual = self.encoder(embedded)
        
        # Apply with masking (inaudible)
        mask = self._compute_audibility_mask(audio)
        watermarked = audio + mask * residual * 0.1
        
        return watermarked
    
    def decode(self, audio):
        """Decode watermark from audio"""
        return torch.sigmoid(self.decoder(audio))
    
    def _compute_audibility_mask(self, audio):
        """Compute psychoacoustic mask"""
        # Simplified - real implementation uses FFT
        return torch.ones_like(audio) * 0.1
    
    def _embed_message(self, audio, message):
        """Embed message bits into audio tensor"""
        B, C, L = audio.shape
        embedded = audio.repeat(1, 1, 1)
        
        # Repeat message to fill length
        message_repeated = message.unsqueeze(-1).repeat(1, 1, L // self.message_length)
        
        # Add message to audio
        embedded = embedded + message_repeated[:, :C, :L] * 0.01
        
        return embedded
```

---

## 3. Digital Signatures and PKI

### 3.1 Content Signing Infrastructure

```
┌─────────────────────────────────────────────────────────┐
│                 PKI FOR CONTENT PROVENANCE               │
│                                                          │
│  Root CA (e.g., C2PA Root)                               │
│    │                                                     │
│    ├── Intermediate CA (Platform-specific)               │
│    │     │                                               │
│    │     ├── Signing Key (per creator)                   │
│    │     │     └── Signs content manifests               │
│    │     │                                               │
│    │     └── Verification Key (public)                   │
│    │           └── Verifies signatures                   │
│    │                                                     │
│    └── Certificate Chain                                 │
│          └── Proves trust from root to content           │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Ed25519 Signing for Content

```python
# Ed25519 content signing
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
import hashlib
import json
import time

class ContentSigningService:
    def __init__(self):
        self.private_key = Ed25519PrivateKey.generate()
        self.public_key = self.private_key.public_key()
    
    def sign_content(self, content_bytes, metadata):
        """Sign content with full provenance"""
        # Create content hash
        content_hash = hashlib.sha256(content_bytes).digest()
        
        # Create comprehensive manifest
        manifest = {
            "spec_version": "2.0",
            "content_hash": {
                "algorithm": "SHA-256",
                "value": content_hash.hex()
            },
            "creator": {
                "name": metadata.get("creator_name"),
                "url": metadata.get("creator_url"),
                "identity": metadata.get("creator_identity")
            },
            "provenance": {
                "created_at": int(time.time()),
                "generator": metadata.get("generator"),
                "ai_model": metadata.get("ai_model"),
                "ai_generated": metadata.get("ai_generated", False),
                "training_data": metadata.get("training_data_info")
            },
            "edits": metadata.get("edit_history", []),
            "license": metadata.get("license"),
            "content_type": metadata.get("content_type")
        }
        
        # Sign manifest
        manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()
        signature = self.private_key.sign(manifest_bytes)
        
        return {
            "manifest": manifest,
            "signature": signature.hex(),
            "public_key": self.public_key.public_bytes(
                serialization.Encoding.Raw,
                serialization.PublicFormat.Raw
            ).hex(),
            "certificate_chain": self._get_certificate_chain()
        }
    
    def verify_content(self, content_bytes, signed_data):
        """Verify content provenance"""
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        
        # Reconstruct public key
        public_key = Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(signed_data["public_key"])
        )
        
        # Verify signature
        manifest_bytes = json.dumps(
            signed_data["manifest"], sort_keys=True, separators=(',', ':')
        ).encode()
        
        try:
            public_key.verify(bytes.fromhex(signed_data["signature"]), manifest_bytes)
            signature_valid = True
        except Exception:
            signature_valid = False
        
        # Verify content hash
        content_hash = hashlib.sha256(content_bytes).digest()
        hash_valid = content_hash.hex() == signed_data["manifest"]["content_hash"]["value"]
        
        return {
            "valid": signature_valid and hash_valid,
            "signature_valid": signature_valid,
            "hash_valid": hash_valid,
            "manifest": signed_data["manifest"]
        }
    
    def _get_certificate_chain(self):
        """Get certificate chain (simplified)"""
        return [
            {"subject": "C2PA Root", "issuer": "C2PA Root"},
            {"subject": "Platform CA", "issuer": "C2PA Root"}
        ]
```

### 3.3 Zero-Knowledge Provenance

```python
# Zero-knowledge proof for provenance (conceptual)
class ZKProvenance:
    def __init__(self):
        self.proving_key = None  # Generated during setup
        self.verification_key = None
    
    def create_proof(self, content_hash, secret_metadata):
        """Create ZK proof that content has valid provenance
        without revealing the metadata"""
        
        # Simplified - real implementation uses zk-SNARKs or similar
        proof = {
            "content_hash": content_hash,
            "proof_type": "zk-SNARK",
            "public_inputs": {
                "has_valid_creator": True,
                "is_ai_generated": secret_metadata["ai_generated"],
                "has_training_license": True
            },
            "proof_data": "...",  # Actual cryptographic proof
            "verification_key": self.verification_key
        }
        
        return proof
    
    def verify_proof(self, proof, public_inputs):
        """Verify ZK proof without learning secret metadata"""
        # Verify without learning:
        # - Who created it
        # - What model was used
        # - Training data details
        
        # Only learn:
        # - Has valid creator (yes/no)
        # - Is AI generated (yes/no)
        # - Has training license (yes/no)
        
        return {
            "valid": True,  # Based on actual proof verification
            "public_claims": public_inputs
        }
```

---

## 4. Platform-Specific Implementations

### 4.1 Google SynthID

```python
# SynthID API usage (conceptual)
import google.generativeai as genai

class SynthIDIntegration:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
    
    def generate_watermarked_image(self, prompt):
        """Generate image with SynthID watermark"""
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_modalities=["image"],
                # SynthID watermark is automatically applied
            )
        )
        
        # Image contains invisible watermark
        return response.candidates[0].content.parts[0].inline_data.data
    
    def verify_synthid(self, image_bytes):
        """Verify SynthID watermark"""
        # SynthID verification API
        result = genai.verify_watermark(image_bytes)
        
        return {
            "has_watermark": result.watermarked,
            "confidence": result.confidence,
            "model": result.model_id
        }
    
    def generate_watermarked_text(self, prompt):
        """Generate text with SynthID watermark"""
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                # Text watermarking is automatic
            )
        )
        
        return response.text  # Contains statistical watermark
```

### 4.2 OpenAI Content Credentials

```python
# OpenAI Content Credentials API
import openai

class OpenAIProvenance:
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)
    
    def generate_image_with_provenance(self, prompt):
        """Generate image with Content Credentials"""
        response = self.client.images.generate(
            model="dall-e-4",
            prompt=prompt,
            size="1024x1024",
            quality="hd",
            # Content Credentials are automatically added
        )
        
        image_url = response.data[0].url
        
        # Download image
        import requests
        image_data = requests.get(image_url).content
        
        return {
            "image": image_data,
            "content_credentials": {
                "c2pa_manifest": "...",  # Embedded in image
                "watermark": "openai-synthid-v2",
                "generator": "DALL-E 4",
                "license": "openai-commercial"
            }
        }
    
    def verify_content_credentials(self, image_bytes):
        """Verify OpenAI Content Credentials"""
        # Extract and verify C2PA manifest
        from c2pa import verify
        
        result = verify(image_bytes)
        
        return {
            "valid": result.valid,
            "generator": result.manifest.generator.name,
            "ai_generated": result.manifest.metadata.get("ai_generated"),
            "signature_valid": result.signature_valid
        }
```

### 4.3 Adobe Content Credentials

```python
# Adobe Content Credentials integration
from adobe_content_credentials import ContentCredentials

class AdobeProvenance:
    def __init__(self):
        self.cr = ContentCredentials()
    
    def sign_photoshop_edit(self, original_image, edit_metadata):
        """Sign Photoshop edit with Content Credentials"""
        # Original image already has Content Credentials
        signed = self.cr.sign(
            original_image,
            creator=edit_metadata["creator"],
            statement="Edited in Photoshop",
            linked_url=edit_metadata.get("portfolio_url")
        )
        
        return signed
    
    def verify_firefly_generation(self, image):
        """Verify Adobe Firefly generated image"""
        verification = self.cr.verify(image)
        
        return {
            "valid": verification.valid,
            "generator": "Adobe Firefly",
            "creator": verification.manifest.creator,
            "ai_generated": True,
            "training_data": {
                "licensed": True,
                "source": "Adobe Stock + public domain",
                "opt_out_available": True
            }
        }
    
    def get_content_credentials_ui(self, image):
        """Get user-facing Content Credentials info"""
        info = self.cr.get_info(image)
        
        return {
            "show_badge": True,
            "badge_text": "Content Credentials",
            "click_info": {
                "creator": info.creator,
                "tools_used": info.software,
                "ai_generated": info.ai_generated,
                "edit_history": info.edit_count,
                "view_full": "contentcredentials.org/verify/..."
            }
        }
```

---

## 5. Verification Infrastructure

### 5.1 Browser Integration

```javascript
// Content Credentials browser extension (conceptual)
class ContentCredentialsExtension {
    constructor() {
        this.c2paParser = new C2PAParser();
    }
    
    async verifyImage(imageUrl) {
        const response = await fetch(imageUrl);
        const arrayBuffer = await response.arrayBuffer();
        const uint8Array = new Uint8Array(arrayBuffer);
        
        // Check for C2PA manifest
        const manifest = this.c2paParser.parse(uint8Array);
        
        if (manifest) {
            return {
                hasCredentials: true,
                generator: manifest.generator.name,
                aiGenerated: manifest.metadata?.ai_generated,
                creator: manifest.creator,
                editHistory: manifest.actions?.length || 0,
                verified: await this.verifySignature(manifest)
            };
        }
        
        return { hasCredentials: false };
    }
    
    async verifySignature(manifest) {
        // Verify cryptographic signature
        const publicKey = await this.getPublicKey(manifest.signer);
        const signatureValid = await publicKey.verify(
            manifest.signature,
            manifest.toBytes()
        );
        
        return signatureValid;
    }
    
    showBadge(verificationResult) {
        // Show Content Credentials badge on image
        const badge = document.createElement('div');
        badge.className = 'content-credentials-badge';
        
        if (verificationResult.aiGenerated) {
            badge.innerHTML = '🤖 AI Generated';
        } else if (verificationResult.creator) {
            badge.innerHTML = `✍️ ${verificationResult.creator}`;
        }
        
        return badge;
    }
}
```

### 5.2 API Verification Service

```python
# Provenance verification API
from fastapi import FastAPI, UploadFile
from PIL import Image
import io

app = FastAPI()

class ProvenanceVerifier:
    def __init__(self):
        self.verifiers = {
            "c2pa": C2PAVerifier(),
            "synthid": SynthIDVerifier(),
            "openai": OpenAIVerifier(),
            "adobe": AdobeVerifier(),
            "neural": NeuralDetector()
        }
    
    async def verify(self, content, content_type):
        """Verify content provenance using all available methods"""
        results = {}
        
        for name, verifier in self.verifiers.items():
            try:
                result = await verifier.verify(content, content_type)
                results[name] = result
            except Exception as e:
                results[name] = {"error": str(e)}
        
        # Aggregate results
        return self.aggregate_results(results)
    
    def aggregate_results(self, results):
        """Aggregate verification results from multiple sources"""
        has_provenance = any(
            r.get("has_provenance", False) 
            for r in results.values() 
            if isinstance(r, dict)
        )
        
        is_ai_generated = any(
            r.get("ai_generated", False)
            for r in results.values()
            if isinstance(r, dict)
        )
        
        confidence = max(
            (r.get("confidence", 0) for r in results.values() 
             if isinstance(r, dict)),
            default=0
        )
        
        return {
            "has_provenance": has_provenance,
            "is_ai_generated": is_ai_generated,
            "confidence": confidence,
            "details": results
        }

@app.post("/verify")
async def verify_content(file: UploadFile, content_type: str):
    """API endpoint for content verification"""
    content = await file.read()
    
    verifier = ProvenanceVerifier()
    result = await verifier.verify(content, content_type)
    
    return result
```

---

## 6. Training Data Provenance

### 6.1 Model Card Provenance

```yaml
# Model Card with full training data provenance
model_card:
  name: "GPT-5"
  version: "5.0.1"
  
  training_data:
    total_tokens: 15_000_000_000_000
    
    sources:
      - name: "Common Crawl"
        url: "https://commoncrawl.org"
        tokens: 8_000_000_000_000
        license: "Open"
        provenance:
          crawl_date: "2025-01 to 2025-12"
          language_distribution:
            en: 0.45
            zh: 0.15
            other: 0.40
          content_types:
            web_pages: 0.70
            news: 0.15
            academic: 0.10
            other: 0.05
          deduplication: "MinHash, 0.8 threshold"
          
      - name: "Books3"
        tokens: 2_000_000_000_000
        license: "Licensed"
        provenance:
          publishers: 1_500
          opt_out_count: 234
          copyright_status: "In copyright"
          
      - name: "Code (GitHub)"
        tokens: 3_000_000_000_000
        license: "Mixed (MIT, Apache, GPL)"
        provenance:
          repositories: 50_000_000
          opt_out_count: 1_234_567
          languages: ["Python", "JavaScript", "Java", ...]
          
      - name: "Wikipedia"
        tokens: 1_000_000_000_000
        license: "CC BY-SA 3.0"
        provenance:
          language: "300+"
          articles: 60_000_000
          
      - name: "Academic Papers"
        tokens: 1_000_000_000_000
        license: "Licensed"
        provenance:
          sources: ["arXiv", "PubMed", "Semantic Scholar"]
          papers: 50_000_000
          copyright_compliance: "Fair use / licensed"
    
    filtering:
      - "Removed PII using regex + NER"
      - "Filtered toxic content (Perspective API > 0.8)"
      - "Deduplicated near-identical content (SimHash)"
      - "Removed copyright-claimed content (DMCA list)"
      
    opt_outs:
      total_requests: 1_500_000
      honored: 1_234_567
      pending: 265_433
```

### 6.2 Training Data Lineage Tracking

```python
# Training data lineage tracking
class TrainingDataLineage:
    def __init__(self):
        self.lineage_db = {}  # In production, use graph database
    
    def register_dataset(self, dataset_id, metadata):
        """Register dataset with full lineage"""
        lineage = {
            "id": dataset_id,
            "metadata": metadata,
            "provenance": {
                "source": metadata["source"],
                "license": metadata["license"],
                "collection_date": metadata["collection_date"],
                "transformations": [],
                "parent_datasets": [],
                "child_datasets": []
            },
            "legal": {
                "copyright_status": metadata.get("copyright_status"),
                "opt_out_count": metadata.get("opt_out_count", 0),
                "dmca_claims": metadata.get("dmca_claims", []),
                "license_violations": []
            },
            "quality": {
                "size_bytes": metadata.get("size_bytes"),
                "num_samples": metadata.get("num_samples"),
                "language_distribution": metadata.get("languages"),
                "deduplication_ratio": metadata.get("dedup_ratio")
            }
        }
        
        self.lineage_db[dataset_id] = lineage
        return lineage
    
    def add_transformation(self, dataset_id, transformation):
        """Record data transformation"""
        if dataset_id in self.lineage_db:
            self.lineage_db[dataset_id]["provenance"]["transformations"].append({
                "type": transformation["type"],
                "parameters": transformation["params"],
                "timestamp": transformation["timestamp"],
                "output_dataset": transformation["output_id"]
            })
    
    def trace_model_training(self, model_id, training_datasets):
        """Trace which datasets were used to train a model"""
        lineage = {
            "model_id": model_id,
            "training_datasets": [],
            "total_tokens": 0,
            "license_summary": {},
            "opt_out_summary": {"total": 0, "honored": 0}
        }
        
        for dataset_id in training_datasets:
            if dataset_id in self.lineage_db:
                dataset = self.lineage_db[dataset_id]
                lineage["training_datasets"].append(dataset)
                lineage["total_tokens"] += dataset["quality"]["num_samples"]
                
                # Aggregate license info
                license_type = dataset["provenance"]["license"]
                if license_type not in lineage["license_summary"]:
                    lineage["license_summary"][license_type] = 0
                lineage["license_summary"][license_type] += dataset["quality"]["num_samples"]
                
                lineage["opt_out_summary"]["total"] += dataset["legal"]["opt_out_count"]
        
        return lineage
```

---

## 7. Cross-Modal Provenance

### 7.1 Multi-Modal Content Chains

When AI content crosses modalities, provenance must chain:

```
User Prompt (Text)
    │
    ▼
DALL-E 4 (Image Generation)
    │ Provenance: AI-generated, prompt recorded
    ▼
Photoshop (Editing)
    │ Provenance: Human-edited, layer history
    ▼
Video Editor (Video Creation)
    │ Provenance: Composite, multiple sources
    ▼
YouTube (Distribution)
    │ Provenance: Platform metadata, C2PA
    ▼
Viewer (Consumption)
    │ Verification: All layers checked
    └── Result: Full provenance chain verified
```

### 7.2 Cross-Modal Verification

```python
# Cross-modal provenance verification
class CrossModalVerifier:
    def __init__(self):
        self.modal_verifiers = {
            "text": TextProvenanceVerifier(),
            "image": ImageProvenanceVerifier(),
            "audio": AudioProvenanceVerifier(),
            "video": VideoProvenanceVerifier()
        }
    
    async def verify_chain(self, content_chain):
        """Verify provenance across content chain"""
        chain_results = []
        
        for item in content_chain:
            modal = item["modality"]
            verifier = self.modal_verifiers[modal]
            
            result = await verifier.verify(item["content"])
            
            # Check link to previous item
            if chain_results:
                prev = chain_results[-1]
                link_valid = self.verify_link(prev, item, result)
                result["link_valid"] = link_valid
            
            chain_results.append(result)
        
        # Overall chain validity
        all_valid = all(r.get("valid", False) for r in chain_results)
        links_valid = all(r.get("link_valid", True) for r in chain_results[1:])
        
        return {
            "chain_valid": all_valid and links_valid,
            "num_items": len(chain_results),
            "results": chain_results
        }
    
    def verify_link(self, prev_result, current_item, current_result):
        """Verify link between content items"""
        # Check that current item references previous
        has_reference = current_item.get("parent_id") == prev_result.get("id")
        
        # Check timestamp consistency
        timestamp_valid = current_item.get("timestamp", 0) > prev_result.get("timestamp", 0)
        
        # Check signature chain
        signature_chain_valid = self.verify_signature_chain(prev_result, current_result)
        
        return has_reference and timestamp_valid and signature_chain_valid
```

---

## 8. Economic Implications

### 8.1 Provenance Premium

Content with verified provenance commands premium value:

| Content Type | Without Provenance | With Provenance | Premium |
|-------------|-------------------|-----------------|---------|
| Stock Photography | $0.50-5.00 | $1.00-15.00 | 100-200% |
| Music Tracks | $0.001/stream | $0.003/stream | 200% |
| News Articles | Standard CPM | 30% higher CPM | 30% |
| Academic Papers | Free/paywalled | Premium access | 50% |
| Art/Design | Variable | 50-300% premium | 50-300% |

### 8.2 Cost of Provenance

Implementing provenance systems has costs:

| Component | Cost (per 1M items) | Notes |
|-----------|---------------------|-------|
| Watermarking | $50-200 | Computational overhead |
| C2PA Signing | $100-500 | Certificate management |
| Storage (manifests) | $20-100 | Metadata storage |
| Verification API | $200-1000 | On-demand verification |
| Platform Integration | $5,000-50,000 | One-time development |
| **Total Annual** | **$10,000-100,000** | For medium-scale platform |

### 8.3 ROI of Provenance

```python
# ROI calculation for provenance implementation
def calculate_provenance_roi(
    monthly_content_items,
    average_revenue_per_item,
    provenance_cost_monthly,
    premium_multiplier=1.3
):
    """Calculate ROI of implementing provenance"""
    
    # Revenue without provenance
    base_revenue = monthly_content_items * average_revenue_per_item
    
    # Revenue with provenance (premium)
    provenance_revenue = base_revenue * premium_multiplier
    
    # Additional revenue from provenance
    additional_revenue = provenance_revenue - base_revenue
    
    # Net benefit
    net_benefit = additional_revenue - provenance_cost_monthly
    
    # ROI
    roi = (net_benefit / provenance_cost_monthly) * 100
    
    return {
        "base_monthly_revenue": base_revenue,
        "provenance_monthly_revenue": provenance_revenue,
        "additional_revenue": additional_revenue,
        "provenance_cost": provenance_cost_monthly,
        "net_benefit": net_benefit,
        "roi_percent": roi,
        "breakeven_items": provenance_cost_monthly / (average_revenue_per_item * (premium_multiplier - 1))
    }

# Example
result = calculate_provenance_roi(
    monthly_content_items=1_000_000,
    average_revenue_per_item=0.01,
    provenance_cost_monthly=5000,
    premium_multiplier=1.3
)
print(f"ROI: {result['roi_percent']:.1f}%")
print(f"Breakeven: {result['breakeven_items']:.0f} items/month")
```

---

## 9. Security Considerations

### 9.1 Attack Vectors

| Attack | Description | Mitigation |
|--------|-------------|------------|
| **Watermark Removal** | Adversarial attacks to remove watermarks | Robust watermarking, multiple watermarks |
| **Metadata Stripping** | Removing C2PA manifests | Platform-level enforcement, redundant metadata |
| **Signature Forgery** | Creating fake provenance | PKI infrastructure, certificate transparency |
| **Training Data Poisoning** | Injecting bad data with fake provenance | Dataset validation, source verification |
| **Model Extraction** | Stealing model to bypass watermarks | Model protection, API-only access |
| **Replay Attacks** | Reusing old provenance for new content | Timestamps, content hashing |

### 9.2 Defense Strategies

```python
# Multi-layer defense against provenance attacks
class ProvenanceDefense:
    def __init__(self):
        self.defense_layers = [
            WatermarkDefense(),
            MetadataDefense(),
            CryptographicDefense(),
            NeuralDefense()
        ]
    
    def defend(self, content, attack_type):
        """Apply multi-layer defense"""
        defenses = {}
        
        for layer in self.defense_layers:
            defense = layer.defend(content, attack_type)
            defenses[layer.name] = defense
        
        # Aggregate defense strength
        overall_strength = np.mean([
            d["strength"] for d in defenses.values()
        ])
        
        return {
            "overall_strength": overall_strength,
            "defenses": defenses,
            "recommendation": self.get_recommendation(overall_strength)
        }
    
    def get_recommendation(self, strength):
        """Get defense recommendation"""
        if strength > 0.9:
            return "Strong defense - likely resistant to attacks"
        elif strength > 0.7:
            return "Moderate defense - consider additional layers"
        elif strength > 0.5:
            return "Weak defense - significant vulnerability"
        else:
            return "Critical - immediate action required"
```

---

## 10. Key Takeaways

1. **Provenance is multi-layered** — watermarks, metadata, signatures, and detection all play roles
2. **C2PA is the emerging standard** — but implementation varies by platform
3. **Watermarking has tradeoffs** — robustness vs. imperceptibility vs. capacity
4. **Cross-modal provenance is essential** — content crosses formats frequently
5. **Economic incentives align** — provenance creates value for creators and platforms
6. **Security is an arms race** — attacks and defenses continuously evolve
7. **Privacy vs. provenance tension** — zero-knowledge proofs offer a path forward
8. **Training data provenance matters** — model cards must document data sources
9. **User education is critical** — tools are useless without awareness
10. **Regulation is accelerating** — EU, China, and US states are mandating provenance

---

*Last updated: June 30, 2026*
*See also: [01-Overview.md](01-Overview.md) | [03-Technical-Deep-Dive.md](03-Technical-Deep-Dive.md) | [04-Tools-and-Frameworks.md](04-Tools-and-Frameworks.md) | [05-Future-Outlook.md](05-Future-Outlook.md)*
