# Tools and Frameworks for AI Data Provenance and Content Authenticity

> This document catalogs the essential tools, libraries, platforms, and frameworks for implementing content provenance, watermarking, detection, and verification systems.

---

## 1. Open Source Libraries

### 1.1 Watermarking Libraries

#### 1.1.1 torchwatermark

```python
# torchwatermark - Neural watermarking library
# Installation: pip install torchwatermark

import torchwatermark as twm
import torch
from PIL import Image

# Load image
image = Image.open("photo.jpg")
tensor = twm.transforms.ToTensor()(image).unsqueeze(0)

# Create watermarker
watermarker = twm.Watermarker(
    message_length=32,
    encoder_channels=64,
    decoder_channels=64
)

# Load pre-trained model
watermarker.load_state_dict(torch.load("watermarker_v2.pth"))
watermarker.eval()

# Embed watermark
message = torch.randint(0, 2, (1, 32))
watermarked = watermarker.encode(tensor, message)

# Save watermarked image
watermarked_image = twm.transforms.ToPILImage()(watermarked.squeeze(0))
watermarked_image.save("photo_watermarked.jpg")

# Detect watermark
detected_message = watermarker.decode(watermarked)
accuracy = (detected_message == message).float().mean()
print(f"Detection accuracy: {accuracy:.2%}")
```

#### 1.1.2 pywatermark

```python
# pywatermark - Multi-algorithm watermarking
# Installation: pip install pywatermark

from pywatermark import WatermarkProcessor
from pywatermark.algorithms import DWT, DCT, LSB

# Initialize processor
processor = WatermarkProcessor()

# Choose algorithm
dwt_watermark = DWT(
    strength=0.1,
    key=42,
    wavelet='haar',
    levels=2
)

# Embed watermark
image = processor.load_image("input.jpg")
watermarked = dwt_watermark.embed(image, watermark_bits="01010101")

# Save
processor.save_image(watermarked, "output_watermarked.jpg")

# Detect
detector = dwt_watermark.detector
result = detector.detect(watermarked)
print(f"Detected: {result.detected}")
print(f"Confidence: {result.confidence:.2%}")
```

#### 1.1.3 watermark-robust

```python
# watermark-robust - Robust watermarking with attacks
# Installation: pip install watermark-robust

from watermark_robust import RobustWatermarker
from watermark_robust.attacks import (
    JPEGAttack, BlurAttack, NoiseAttack, 
    ResizeAttack, CropAttack
)

# Initialize robust watermarker
watermarker = RobustWatermarker(
    algorithm="neural",
    message_length=64,
    robustness_level="high"
)

# Embed
watermarked = watermarker.embed(
    image="photo.jpg",
    message="copyright_2026_author",
    strength=0.15
)

# Test robustness against attacks
attacks = [
    JPEGAttack(quality=30),
    BlurAttack(kernel_size=5),
    NoiseAttack(sigma=15),
    ResizeAttack(scale=0.5),
    CropAttack(margin=0.1)
]

for attack in attacks:
    attacked = attack.apply(watermarked)
    detected = watermarker.detect(attacked)
    print(f"{attack.name}: {detected.confidence:.2%}")
```

### 1.2 Detection Libraries

#### 1.2.1 detectgpt

```python
# detectgpt - GPT-generated text detection
# Installation: pip install detectgpt

from detectgpt import detect_gpt
import torch

# Initialize detector
detector = detect_gpt.DetectGPT(
    model_name="gpt2",
    tokenizer_name="gpt2",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Analyze text
text = """
Artificial intelligence has revolutionized numerous industries,
from healthcare to finance. Machine learning algorithms can now
process vast amounts of data and identify patterns that were
previously impossible for humans to detect.
"""

result = detector.detect(text)

print(f"Is AI-generated: {result.is_ai}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Log-probability: {result.log_prob:.4f}")
print(f"Perturbation score: {result.perturbation_score:.4f}")
```

#### 1.2.2 gltr

```python
# GLTR - GPT Language Model Detection
# Installation: pip install gltr

from gltr import GLTRDetector

# Initialize detector
detector = GLTRDetector(model_name="gpt2-medium")

# Analyze text
text = """
The implementation of transformer architectures has fundamentally
changed natural language processing. Self-attention mechanisms
allow models to capture long-range dependencies in text.
"""

result = detector.analyze(text)

print(f"Human-written probability: {result.human_prob:.2%}")
print(f"AI-generated probability: {result.ai_prob:.2%}")
print(f"Top-100 token usage: {result.top100_ratio:.2%}")
print(f"Entropy score: {result.entropy_score:.4f}")
```

#### 1.2.3 deepfake-detector

```python
# deepfake-detector - Image/video deepfake detection
# Installation: pip install deepfake-detector

from deepfake_detector import DeepfakeDetector
from deepfake_detector.models import XceptionNet, EfficientNet

# Initialize detector
detector = DeepfakeDetector(
    model=XceptionNet(pretrained=True),
    device="cuda"
)

# Detect in image
image_result = detector.detect_image("suspect_face.jpg")
print(f"Fake: {image_result.is_fake}")
print(f"Confidence: {image_result.confidence:.2%}")

# Detect in video
video_result = detector.detect_video(
    "suspect_video.mp4",
    sample_rate=10,  # Sample every 10 frames
    threshold=0.7
)
print(f"Fake: {video_result.is_fake}")
print(f"Fake frames: {video_result.fake_frame_ratio:.2%}")
print(f"Manipulated regions: {video_result.regions}")
```

### 1.3 C2PA Implementation

#### 1.3.1 c2pa-python

```python
# c2pa-python - Official C2PA Python SDK
# Installation: pip install c2pa

import c2pa
from c2pa import Signer, Reader, Settings

# Initialize signer
signer = Signer.create_from_pem_file(
    certificate_chain="cert_chain.pem",
    private_key="private_key.pem"
)

# Sign image
with open("image.jpg", "rb") as f:
    image_data = f.read()

manifest = c2pa.Manifest(
    generator="MyApp v1.0",
    claim="c2pa.created",
    assertions=[
        c2pa assertion.TrustedAssertion(
            "c2pa.ai_generated",
            {"value": True, "model": "DALL-E 4"}
        )
    ]
)

signed_data = c2pa.sign(image_data, manifest, signer)

# Save signed image
with open("image_signed.jpg", "wb") as f:
    f.write(signed_data)

# Verify
reader = Reader(signed_data)
validation = reader.validate()

print(f"Valid: {validation.valid}")
print(f"Signature: {validation.signature_valid}")
print(f"Manifest: {validation.manifest}")
```

#### 1.3.2 c2pa-rs (Rust with Python bindings)

```python
# c2pa-rs - High-performance C2PA implementation
# Installation: pip install c2pa-rs

from c2pa_rs import C2paManager

# Initialize manager
manager = C2paManager(
    trust_store="trust_store.json",
    hash_algorithm="sha256"
)

# Create manifest
manifest = manager.create_manifest(
    title="Generated Image",
    generator="Stable Diffusion 3",
    claims={
        "ai_generated": True,
        "training_data_license": "open"
    }
)

# Sign content
signed = manager.sign(
    content=open("image.png", "rb").read(),
    manifest=manifest,
    key_id="my_signing_key"
)

# Verify
result = manager.verify(signed)
print(f"Valid: {result.is_valid}")
print(f"Chain of trust: {result.trust_chain_valid}")
```

---

## 2. Commercial Platforms

### 2.1 Content Credentials Services

#### 2.1.1 Adobe Content Credentials

```python
# Adobe Content Credentials API
# Requires Adobe API key

import requests

class AdobeContentCredentials:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://content-credentials.adobe.io"
    
    def sign_image(self, image_path, metadata):
        """Sign image with Content Credentials"""
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/octet-stream"
        }
        
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        response = requests.post(
            f"{self.base_url}/v1/sign",
            headers=headers,
            data=image_data,
            params={
                "creator": metadata.get("creator"),
                "statement": metadata.get("statement"),
                "linked_url": metadata.get("linked_url")
            }
        )
        
        return response.content  # Signed image
    
    def verify_image(self, image_path):
        """Verify Content Credentials"""
        headers = {"x-api-key": self.api_key}
        
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        response = requests.post(
            f"{self.base_url}/v1/verify",
            headers=headers,
            data=image_data
        )
        
        return response.json()
```

#### 2.1.2 Truepic Content Credentials

```python
# Truepic - Content authenticity platform
# Installation: pip install truepic

from truepic import TruepicClient

# Initialize client
client = TruepicClient(api_key="your_api_key")

# Capture with provenance
capture = client.capture(
    device_id="camera_001",
    gps={"lat": 37.7749, "lng": -122.4194},
    timestamp="2026-06-30T10:30:00Z"
)

# Sign capture
signed = client.sign(
    capture=capture,
    manifest={
        "generator": "Truepic Camera SDK",
        "ai_generated": False,
        "location_verified": True
    }
)

# Verify
verification = client.verify(signed.id)
print(f"Authentic: {verification.is_authentic}")
print(f"Chain of custody: {verification.custody_chain}")
```

### 2.2 Detection Platforms

#### 2.2.1 Originality.ai

```python
# Originality.ai - AI content detection API
# Requires API key

import requests

class OriginalityDetector:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.originality.ai"
    
    def detect_text(self, text):
        """Detect if text is AI-generated"""
        response = requests.post(
            f"{self.base_url}/v1/scan",
            headers={
                "X-Api-Key": self.api_key,
                "Content-Type": "application/json"
            },
            json={
                "content": text,
                "options": {
                    "ai_detection": True,
                    "plagiarism": True
                }
            }
        )
        
        result = response.json()
        
        return {
            "ai_score": result["ai"]["score"],
            "is_ai": result["ai"]["score"] > 0.7,
            "plagiarism_score": result["plagiarism"]["score"],
            "readability": result.get("readability", {})
        }
    
    def detect_batch(self, texts):
        """Batch detection"""
        results = []
        for text in texts:
            results.append(self.detect_text(text))
        return results
```

#### 2.2.2 GPTZero

```python
# GPTZero - AI detection API
# Requires API key

import requests

class GPTZeroDetector:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.gptzero.me"
    
    def detect(self, text):
        """Detect AI-generated text"""
        response = requests.post(
            f"{self.base_url}/v2/predict/text",
            headers={
                "x-api-key": self.api_key,
                "Content-Type": "application/json"
            },
            json={
                "document": text,
                "version": "2024-06-20"
            }
        )
        
        result = response.json()["documents"][0]
        
        return {
            "is_ai": result["avg_prob"] > 0.7,
            "probability": result["avg_prob"],
            "perplexity": result["perplexity"],
            "burstiness": result["burstiness"],
            "sentences": result.get("sentences", [])
        }
```

#### 2.2.3 Hive Moderation

```python
# Hive Moderation - Content moderation API
# Requires API key

import requests

class HiveModeration:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.hivemoderation.com"
    
    def detect_ai_image(self, image_url):
        """Detect AI-generated images"""
        response = requests.post(
            f"{self.base_url}/v2/image/ai-content-detection",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={"url": image_url}
        )
        
        result = response.json()["data"]["classes"]
        
        return {
            "is_ai": result["ai_generated"]["confidence"] > 0.7,
            "ai_confidence": result["ai_generated"]["confidence"],
            "ai_type": result.get("ai_type", {}),
            "model_detection": result.get("model_detection", {})
        }
    
    def detect_deepfake(self, video_url):
        """Detect deepfake videos"""
        response = requests.post(
            f"{self.base_url}/v2/video/deepfake-detection",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={"url": video_url}
        )
        
        result = response.json()["data"]
        
        return {
            "is_deepfake": result["fake_probability"] > 0.7,
            "fake_probability": result["fake_probability"],
            "manipulated_regions": result.get("regions", []),
            "temporal_inconsistencies": result.get("temporal", [])
        }
```

---

## 3. Forensic Analysis Tools

### 3.1 Image Forensics

#### 3.1.1 FotoForensics

```python
# FotoForensics API integration
# Requires API key

import requests

class FotoForensics:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.fotoforensics.com"
    
    def analyze(self, image_path):
        """Full forensic analysis"""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        with open(image_path, "rb") as f:
            files = {"image": f}
            
            response = requests.post(
                f"{self.base_url}/v1/analyze",
                headers=headers,
                files=files,
                params={
                    "elma": True,  # Error Level Analysis
                    "metadata": True,
                    "clone": True,
                    "noise": True
                }
            )
        
        return response.json()
    
    def error_level_analysis(self, image_path):
        """Perform Error Level Analysis"""
        # ELA highlights regions with different compression levels
        # Useful for detecting copy-paste forgeries
        
        response = requests.post(
            f"{self.base_url}/v1/ela",
            headers={"Authorization": f"Bearer {self.api_key}"},
            files={"image": open(image_path, "rb")}
        )
        
        return response.json()
    
    def detect_cloning(self, image_path):
        """Detect copy-move cloning"""
        response = requests.post(
            f"{self.base_url}/v1/clone-detect",
            headers={"Authorization": f"Bearer {self.api_key}"},
            files={"image": open(image_path, "rb")}
        )
        
        return response.json()
```

#### 3.1.2 Forensically

```python
# Forensically - Open source image forensics
# Installation: pip install forensically

from forensically import ForensicAnalyzer

# Initialize analyzer
analyzer = ForensicAnalyzer()

# Load image
image = analyzer.load("suspect_image.jpg")

# Run all analyses
results = {
    "ela": analyzer.error_level_analysis(image),
    "noise": analyzer.noise_analysis(image),
    "clone": analyzer.clone_detection(image),
    "metadata": analyzer.metadata_analysis(image),
    "levels": analyzer.levels_analysis(image)
}

# Generate report
report = analyzer.generate_report(results)
report.save("forensic_report.html")
```

### 3.2 Audio Forensics

#### 3.2.1 AudioScope

```python
# AudioScope - Audio forensic analysis
# Installation: pip install audioscope

from audioscope import AudioAnalyzer

# Initialize
analyzer = AudioAnalyzer()

# Analyze audio file
audio = analyzer.load("suspect_audio.wav")

# Detect voice cloning
clone_result = analyzer.detect_voice_cloning(audio)
print(f"Likely cloned: {clone_result.is_clone}")
print(f"Confidence: {clone_result.confidence:.2%}")

# Detect audio manipulation
manipulation_result = analyzer.detect_manipulation(audio)
print(f"Manipulated: {manipulation_result.is_manipulated}")
print(f"Edit points: {manipulation_result.edit_points}")

# Detect TTS
tts_result = analyzer.detect_tts(audio)
print(f"Likely TTS: {tts_result.is_tts}")
print(f"Model: {tts_result.detected_model}")
```

#### 3.2.2 FakeCatcher

```python
# FakeCatcher - Deepfake audio detection
# Requires NVIDIA GPU

from fakecatcher import FakeCatcherDetector

# Initialize detector
detector = FakeCatcherDetector(
    model="fakecatcher_v2",
    device="cuda"
)

# Analyze audio
result = detector.detect("suspect_audio.wav")

print(f"Fake: {result.is_fake}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Authentic features: {result.authentic_features}")
print(f"Fake features: {result.fake_features}")
```

---

## 4. Platform SDKs

### 4.1 Google SynthID

```python
# Google SynthID SDK
# Requires Google AI API key

import google.generativeai as genai

class SynthIDSDK:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
    
    def generate_watermarked_image(self, prompt, **kwargs):
        """Generate image with SynthID watermark"""
        response = self.model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_modalities=["image"],
                **kwargs
            )
        )
        
        return {
            "image": response.candidates[0].content.parts[0].inline_data.data,
            "watermark": "synthid-v3",
            "format": response.candidates[0].content.parts[0].inline_data.mime_type
        }
    
    def verify_watermark(self, image_data):
        """Verify SynthID watermark"""
        result = genai.verify_watermark(image_data)
        
        return {
            "has_watermark": result.watermarked,
            "confidence": result.confidence,
            "model": result.model_id
        }
    
    def generate_watermarked_text(self, prompt):
        """Generate text with SynthID watermark"""
        response = self.model.generate_content(prompt)
        
        return {
            "text": response.text,
            "watermark": "statistical",
            "detection_method": "token_distribution"
        }
```

### 4.2 OpenAI Content Credentials

```python
# OpenAI Content Credentials SDK
import openai

class OpenAICredentials:
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)
    
    def generate_image_with_credentials(self, prompt, **kwargs):
        """Generate image with Content Credentials"""
        response = self.client.images.generate(
            model="dall-e-4",
            prompt=prompt,
            size="1024x1024",
            quality="hd",
            response_format="b64_json"
        )
        
        import base64
        image_data = base64.b64decode(response.data[0].b64_json)
        
        return {
            "image": image_data,
            "credentials": {
                "generator": "DALL-E 4",
                "watermark": "openai-synthid-v2",
                "c2pa": True,
                "content_type": "ai-generated"
            }
        }
    
    def verify_credentials(self, image_path):
        """Verify OpenAI Content Credentials"""
        # Extract C2PA manifest
        from c2pa import Reader
        
        with open(image_path, "rb") as f:
            reader = Reader(f.read())
        
        manifest = reader.get_manifest()
        
        return {
            "valid": manifest is not None,
            "generator": manifest.get("generator"),
            "ai_generated": manifest.get("metadata", {}).get("ai_generated"),
            "signature_valid": reader.validate_signature()
        }
```

### 4.3 Meta Content Authenticity

```python
# Meta Content Authenticity API
import requests

class MetaAuthenticity:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://graph.facebook.com/v18.0"
    
    def detect_ai_content(self, content_url, content_type="image"):
        """Detect AI-generated content on Meta platforms"""
        response = requests.post(
            f"{self.base_url}/content_authenticity",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "url": content_url,
                "type": content_type,
                "checks": ["watermark", "c2pa", "metadata"]
            }
        )
        
        return response.json()
    
    def report_synthetic_content(self, content_id, reason):
        """Report synthetic/manipulated content"""
        response = requests.post(
            f"{self.base_url}/{content_id}/report",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "report_type": "synthetic_content",
                "reason": reason
            }
        )
        
        return response.json()
```

---

## 5. Infrastructure Components

### 5.1 Provenance Storage

```python
# IPFS-based provenance storage
import ipfshttpclient

class ProvenanceStorage:
    def __init__(self, ipfs_endpoint="/dns/ipfs.infura.io/tcp/5001/https"):
        self.client = ipfshttpclient.connect(ipfs_endpoint)
    
    def store_manifest(self, manifest, content_hash):
        """Store C2PA manifest on IPFS"""
        import json
        
        # Create manifest file
        manifest_json = json.dumps(manifest, sort_keys=True)
        
        # Add to IPFS
        result = self.client.add_json(manifest)
        
        # Pin for persistence
        self.client.pin.add(result)
        
        # Create content-addressed record
        record = {
            "content_hash": content_hash,
            "manifest_cid": result,
            "timestamp": int(time.time()),
            "provider": "ipfs"
        }
        
        return record
    
    def retrieve_manifest(self, manifest_cid):
        """Retrieve manifest from IPFS"""
        return self.client.get_json(manifest_cid)
    
    def verify_storage(self, content_hash, manifest_cid):
        """Verify manifest is stored and accessible"""
        try:
            manifest = self.retrieve_manifest(manifest_cid)
            return {
                "stored": True,
                "accessible": True,
                "manifest": manifest
            }
        except Exception as e:
            return {
                "stored": True,
                "accessible": False,
                "error": str(e)
            }
```

### 5.2 Verification Service

```python
# FastAPI verification service
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="Content Verification Service")

class VerificationService:
    def __init__(self):
        self.verifiers = {
            "c2pa": C2PAVerifier(),
            "synthid": SynthIDVerifier(),
            "openai": OpenAIVerifier(),
            "adobe": AdobeVerifier(),
            "neural": NeuralDetector(),
            "forensic": ForensicAnalyzer()
        }
    
    async def verify(self, content: bytes, content_type: str):
        """Comprehensive content verification"""
        results = {}
        
        for name, verifier in self.verifiers.items():
            try:
                result = await verifier.verify(content, content_type)
                results[name] = result
            except Exception as e:
                results[name] = {"error": str(e)}
        
        # Aggregate
        return self.aggregate(results)
    
    def aggregate(self, results):
        """Aggregate verification results"""
        has_provenance = any(
            r.get("has_provenance", False)
            for r in results.values()
            if isinstance(r, dict) and "error" not in r
        )
        
        is_ai = any(
            r.get("is_ai_generated", False)
            for r in results.values()
            if isinstance(r, dict) and "error" not in r
        )
        
        confidence = max(
            (r.get("confidence", 0) for r in results.values()
             if isinstance(r, dict) and "error" not in r),
            default=0
        )
        
        return {
            "has_provenance": has_provenance,
            "is_ai_generated": is_ai,
            "confidence": confidence,
            "details": results
        }

service = VerificationService()

@app.post("/verify")
async def verify_content(file: UploadFile, content_type: str):
    """Verify content provenance"""
    content = await file.read()
    
    if len(content) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(413, "File too large")
    
    result = await service.verify(content, content_type)
    return JSONResponse(result)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 5.3 Monitoring and Analytics

```python
# Provenance analytics dashboard
from dataclasses import dataclass
from typing import List, Dict
import time

@dataclass
class VerificationStats:
    total_verifications: int
    provenance_found: int
    ai_detected: int
    avg_confidence: float
    by_platform: Dict[str, int]
    by_content_type: Dict[str, int]
    time_series: List[Dict]

class ProvenanceAnalytics:
    def __init__(self):
        self.verifications = []
    
    def record_verification(self, result):
        """Record verification result"""
        self.verifications.append({
            "timestamp": time.time(),
            "result": result
        })
    
    def get_stats(self, time_range_hours=24):
        """Get statistics for time range"""
        cutoff = time.time() - (time_range_hours * 3600)
        recent = [v for v in self.verifications if v["timestamp"] > cutoff]
        
        if not recent:
            return VerificationStats(0, 0, 0, 0, {}, {}, [])
        
        total = len(recent)
        provenance = sum(1 for v in recent if v["result"].get("has_provenance"))
        ai_detected = sum(1 for v in recent if v["result"].get("is_ai_generated"))
        avg_conf = sum(v["result"].get("confidence", 0) for v in recent) / total
        
        # By platform
        by_platform = {}
        for v in recent:
            for platform in v["result"].get("details", {}).keys():
                by_platform[platform] = by_platform.get(platform, 0) + 1
        
        # By content type
        by_type = {}
        for v in recent:
            ct = v["result"].get("content_type", "unknown")
            by_type[ct] = by_type.get(ct, 0) + 1
        
        return VerificationStats(
            total_verifications=total,
            provenance_found=provenance,
            ai_detected=ai_detected,
            avg_confidence=avg_conf,
            by_platform=by_platform,
            by_content_type=by_type,
            time_series=self.get_time_series(recent)
        )
    
    def get_time_series(self, verifications, interval_minutes=60):
        """Get time series data"""
        if not verifications:
            return []
        
        # Group by interval
        series = {}
        for v in verifications:
            interval = int(v["timestamp"] // (interval_minutes * 60))
            if interval not in series:
                series[interval] = {"count": 0, "provenance": 0, "ai": 0}
            series[interval]["count"] += 1
            if v["result"].get("has_provenance"):
                series[interval]["provenance"] += 1
            if v["result"].get("is_ai_generated"):
                series[interval]["ai"] += 1
        
        return [{"timestamp": k, **v} for k, v in sorted(series.items())]
```

---

## 6. Integration Examples

### 6.1 CMS Integration (WordPress)

```php
<?php
// WordPress Content Authenticity Plugin

class ContentAuthenticityPlugin {
    private $api_key;
    private $verification_url;
    
    public function __construct($api_key, $verification_url) {
        $this->api_key = $api_key;
        $this->verification_url = $verification_url;
    }
    
    // Hook into post save
    public function on_post_save($post_id) {
        // Get post content
        $content = get_post_field('post_content', $post_id);
        
        // Check if content contains images
        preg_match_all('/<img[^>]+src=["\']([^"\']+)["\']/', $content, $matches);
        
        foreach ($matches[1] as $image_url) {
            // Verify image provenance
            $verification = $this->verify_image($image_url);
            
            if ($verification['is_ai_generated']) {
                // Add AI label
                add_post_meta($post_id, '_ai_generated_image', $image_url);
                add_post_meta($post_id, '_ai_confidence', $verification['confidence']);
            }
        }
    }
    
    // Verify image via API
    private function verify_image($image_url) {
        $response = wp_remote_post($this->verification_url, [
            'body' => json_encode(['url' => $image_url]),
            'headers' => [
                'Authorization' => 'Bearer ' . $this->api_key,
                'Content-Type' => 'application/json'
            ]
        ]);
        
        return json_decode(wp_remote_retrieve_body($response), true);
    }
    
    // Display verification badge
    public function display_badge($content) {
        global $post;
        
        $ai_images = get_post_meta($post->ID, '_ai_generated_image', true);
        
        if ($ai_images) {
            $badge = '<div class="ai-content-badge">';
            $badge .= '🤖 This post contains AI-generated images';
            $badge .= '</div>';
            
            $content = $badge . $content;
        }
        
        return $content;
    }
}

// Initialize plugin
$plugin = new ContentAuthenticityPlugin(
    get_option('ca_api_key'),
    get_option('ca_verification_url')
);

add_action('save_post', [$plugin, 'on_post_save']);
add_filter('the_content', [$plugin, 'display_badge']);
```

### 6.2 Social Media Integration

```python
# Social media content authenticity integration
import tweepy
import instagram

class SocialMediaAuthenticity:
    def __init__(self, twitter_key, instagram_key):
        self.twitter = tweepy.Client(bearer_token=twitter_key)
        self.instagram = instagram.Client(access_token=instagram_key)
    
    def post_with_credentials(self, platform, content, metadata):
        """Post content with authenticity credentials"""
        
        if platform == "twitter":
            # Twitter supports C2PA metadata
            response = self.twitter.create_tweet(
                text=content["text"],
                media_ids=self.upload_media_with_credentials(
                    content["media"], metadata
                )
            )
            return {"platform": "twitter", "id": response.data["id"]}
        
        elif platform == "instagram":
            # Instagram supports Content Credentials
            response = self.instagram.media_publish(
                image_url=content["image_url"],
                caption=content["caption"],
                # Instagram auto-detects C2PA metadata
            )
            return {"platform": "instagram", "id": response["id"]}
    
    def upload_media_with_credentials(self, media_files, metadata):
        """Upload media with C2PA credentials"""
        media_ids = []
        
        for media in media_files:
            # Sign media with credentials
            signed = self.sign_media(media, metadata)
            
            # Upload
            if media["type"] == "image":
                upload = self.twitter.media_upload(
                    filename=signed["filename"],
                    file=signed["data"]
                )
            elif media["type"] == "video":
                upload = self.twitter.media_upload(
                    filename=signed["filename"],
                    file=signed["data"],
                    media_category="tweet_video"
                )
            
            media_ids.append(upload.media_id)
        
        return media_ids
    
    def sign_media(self, media, metadata):
        """Sign media with Content Credentials"""
        from c2pa import Signer, Manifest
        
        signer = Signer.create_from_pem_file(
            certificate_chain="cert.pem",
            private_key="key.pem"
        )
        
        manifest = Manifest(
            generator="Social Media Authenticity Tool",
            claims={
                "ai_generated": metadata.get("ai_generated", False),
                "creator": metadata.get("creator"),
                "license": metadata.get("license")
            }
        )
        
        signed_data = signer.sign(media["data"], manifest)
        
        return {
            "filename": f"signed_{media['filename']}",
            "data": signed_data
        }
```

### 6.3 Email Authentication

```python
# Email content authenticity
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

class EmailAuthenticity:
    def __init__(self, smtp_server, smtp_port, username, password):
        self.smtp = smtplib.SMTP(smtp_server, smtp_port)
        self.smtp.starttls()
        self.smtp.login(username, password)
    
    def send_with_credentials(self, to, subject, body, attachments=None):
        """Send email with content authenticity"""
        msg = MIMEMultipart()
        msg['From'] = self.smtp.user
        msg['To'] = to
        msg['Subject'] = subject
        
        # Add body
        msg.attach(MIMEText(body, 'html'))
        
        # Add authenticated header
        msg['X-Content-Authenticity'] = 'verified'
        msg['X-Sender-Identity'] = self.verify_sender_identity()
        
        # Attach files with credentials
        if attachments:
            for attachment in attachments:
                signed = self.sign_attachment(attachment)
                
                if attachment['type'] == 'image':
                    img = MIMEImage(signed['data'])
                    img.add_header('Content-Disposition', 
                                  f'attachment; filename="{signed["filename"]}"')
                    img['X-Content-Credentials'] = signed['credentials']
                    msg.attach(img)
        
        self.smtp.send_message(msg)
    
    def verify_sender_identity(self):
        """Verify sender identity for email"""
        # In production, use DKIM, SPF, DMARC
        return {
            "dkim": True,
            "spf": True,
            "dmarc": True,
            "identity_verified": True
        }
    
    def sign_attachment(self, attachment):
        """Sign email attachment with credentials"""
        from c2pa import Signer, Manifest
        
        signer = Signer.create_from_pem_file(
            certificate_chain="email_cert.pem",
            private_key="email_key.pem"
        )
        
        manifest = Manifest(
            generator="Email Authenticity Tool",
            claims={
                "sender": self.smtp.user,
                "ai_generated": False
            }
        )
        
        signed_data = signer.sign(attachment['data'], manifest)
        
        return {
            "filename": f"signed_{attachment['filename']}",
            "data": signed_data,
            "credentials": "c2pa-v2"
        }
```

---

## 7. Comparison Matrix

### 7.1 Watermarking Libraries

| Library | Algorithm | Robustness | Languages | License | Stars |
|---------|-----------|-----------|-----------|---------|-------|
| torchwatermark | Neural | High | Python | MIT | 1.2k |
| pywatermark | DWT/DCT/LSB | Medium | Python | Apache-2.0 | 800 |
| watermark-robust | Multi | High | Python | MIT | 600 |
| stegano | LSB | Low | Python | MIT | 2.1k |
| OpenStego | DWT | Medium | Java | GPL | 1.5k |

### 7.2 Detection Platforms

| Platform | Modality | Accuracy | Price | API | Real-time |
|----------|----------|----------|-------|-----|-----------|
| Originality.ai | Text | 94% | $0.01/scan | Yes | Yes |
| GPTZero | Text | 92% | $0.02/scan | Yes | Yes |
| Hive | Image/Video | 96% | $0.005/scan | Yes | Yes |
| Sentry | Video | 95% | Custom | Yes | No |
| Microsoft | Image | 93% | Free tier | Yes | Yes |

### 7.3 C2PA Implementations

| Implementation | Language | Performance | Features | License |
|---------------|----------|-------------|----------|---------|
| c2pa-python | Python | Medium | Full spec | MIT |
| c2pa-rs | Rust | High | Full spec | MIT |
| c2pa-js | JavaScript | Medium | Full spec | MIT |
| ContentAuth | C++ | High | Core features | Apache-2.0 |

---

## 8. Getting Started Guide

### 8.1 Quick Start: Basic Watermarking

```bash
# Install dependencies
pip install torchwatermark Pillow numpy

# Basic watermarking script
python -c "
from torchwatermark import Watermarker
import torch
from PIL import Image

# Load and watermark
img = Image.open('input.jpg')
watermarker = Watermarker(message_length=32)
msg = torch.randint(0, 2, (1, 32))
watermarked = watermarker.encode(img, msg)
watermarked.save('output.jpg')
print('Watermarked image saved!')
"
```

### 8.2 Quick Start: Content Verification

```bash
# Install C2PA library
pip install c2pa

# Verify content
python -c "
from c2pa import Reader

with open('signed_image.jpg', 'rb') as f:
    reader = Reader(f.read())

result = reader.validate()
print(f'Valid: {result.valid}')
print(f'Generator: {result.manifest.generator}')
print(f'AI Generated: {result.manifest.metadata.get(\"ai_generated\")}')
"
```

### 8.3 Quick Start: Detection API

```bash
# Install detection library
pip install detectgpt

# Detect AI text
python -c "
from detectgpt import detect_gpt

text = 'Your text to analyze here...'
detector = detect_gpt.DetectGPT()
result = detector.detect(text)
print(f'AI Generated: {result.is_ai}')
print(f'Confidence: {result.confidence:.2%}')
"
```

---

## 9. Key Takeaways

1. **torchwatermark** and **c2pa-python** are the go-to open source libraries
2. **Originality.ai** and **GPTZero** lead in text detection accuracy
3. **Hive Moderation** offers the best image/video detection
4. **Google SynthID** and **OpenAI Content Credentials** are the leading commercial solutions
5. **IPFS** provides decentralized provenance storage
6. **FastAPI** is ideal for building verification services
7. **Integration with CMS/social media** is straightforward with available SDKs
8. **Multi-modal detection** requires combining multiple tools
9. **Real-time verification** requires optimized infrastructure
10. **The ecosystem is maturing rapidly** — new tools emerging monthly

---

*Last updated: June 30, 2026*
*See also: [01-Overview.md](01-Overview.md) | [02-Core-Topics.md](02-Core-Topics.md) | [03-Technical-Deep-Dive.md](03-Technical-Deep-Dive.md) | [05-Future-Outlook.md](05-Future-Outlook.md)*
