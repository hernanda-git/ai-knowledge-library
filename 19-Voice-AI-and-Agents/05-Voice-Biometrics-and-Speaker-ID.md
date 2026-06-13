# Voice Biometrics and Speaker Identification

## 5.1 Introduction to Voice Biometrics

Voice biometrics is the technology that identifies or verifies a person based on the unique characteristics of their voice. Unlike speech recognition (which focuses on WHAT is being said), voice biometrics focuses on WHO is speaking. This distinction is fundamental to understanding the technology's capabilities and limitations.

### 5.1.1 Key Concepts

**Speaker Identification:** Determining WHO is speaking from a set of known speakers. The system has enrollment data for N speakers and must identify which speaker (if any) matches the input. This is a 1:N matching problem.

**Speaker Verification:** Authenticating a claimed identity. The user claims to be person X, and the system verifies whether the voice matches X's enrollment. This is a 1:1 matching problem.

**Speaker Diarization:** Determining WHO spoke WHEN in a multi-speaker conversation. The system segments audio by speaker identity without necessarily knowing who the speakers are. Returns "Speaker A spoke from 0:00-0:30, Speaker B from 0:30-1:15."

**Voice Print:** A compact numeric representation (embedding) of a person's voice characteristics. Typically a 128-512 dimensional vector. Voice prints capture: vocal tract shape, pitch range, speaking rhythm, accent patterns, and other distinctive characteristics.

### 5.1.2 Applications

| Application | Type | Description |
|-------------|------|-------------|
| Phone Banking | Verification | "Your voice is your password" for account access |
| Call Center Authentication | Verification | Replace security questions with voice verification |
| Fraud Detection | Identification | Flag known fraudsters calling into contact centers |
| Personalized Assistants | Identification | Recognize user and personalize responses |
| Access Control | Verification | Voice-based physical or logical access |
| Forensic Analysis | Identification | Law enforcement speaker identification |
| Patient Identification | Verification | Healthcare telemedicine authentication |
| Smart Home | Identification | "Hey device, who's speaking?" |

### 5.1.3 Market Size

The voice biometrics market was valued at $1.8 billion in 2024 and is projected to reach $5.5 billion by 2030, growing at a CAGR of 20.5%. Growth drivers: contact center automation, fraud prevention, passwordless authentication, and regulatory compliance (PSD2 SCA, HIPAA, GDPR).

## 5.2 Text-Dependent vs Text-Independent

### 5.2.1 Text-Dependent Verification

The user speaks a fixed passphrase (enrollment phrase is same as verification phrase).

**Advantages:**
- Higher accuracy (EER 0.5-1.5%)
- Simpler modeling (can use DTW or frame-level matching)
- Anti-spoofing via phrase verification
- Less sensitive to channel variation

**Disadvantages:**
- User must remember the passphrase
- Less flexible for natural conversation
- Passphrase can be overheard

**Examples:**
- "My voice is my password"
- "At Acme Bank, access granted"
- "Zero, one, two, three, four"

### 5.2.2 Text-Independent Verification

The user speaks any phrase (enrollment and verification phrases are different).

**Advantages:**
- More natural user experience
- Can verify from natural conversation
- No passphrase to remember
- Harder to replay (but still possible)

**Disadvantages:**
- Lower accuracy (EER 1-5%)
- More complex modeling (must separate speaker characteristics from content)
- More sensitive to channel and duration variation
- Requires longer speech for good accuracy (10-30 seconds)

### 5.2.3 Text-Prompted Verification

The system generates a random phrase for the user to repeat. Combines security of text-dependent with flexibility of text-independent.

**Advantages:**
- Highest security (prevents replay and voice synthesis attacks)
- No fixed passphrase to steal
- Content verification + speaker verification

**Disadvantages:**
- Most complex user experience
- Requires user to read unpredictable text
- Longer enrollment process

## 5.3 Voice Biometric Models and Architectures

### 5.3.1 Evolution of Speaker Recognition Models

**Pre-2015 (i-vectors):** Gaussian mixture models (GMMs) with universal background models (UBM). I-vectors compactly represent speaker characteristics. PLDA (Probabilistic Linear Discriminant Analysis) for scoring. State-of-the-art until ~2018.

**2015-2020 (x-vectors):** Deep neural networks replacing GMMs. TDNN (Time-Delay Neural Network) based x-vector extractors. PLDA scoring. Significant improvement over i-vectors. EER reduction of 20-40%.

**2020-2024 (ResNet + ECAPA):** Residual networks and attentional pooling. ECAPA-TDNN (Emphasized Channel Attention, Propagation, and Aggregation TDNN) became the dominant architecture. Self-supervised pre-training emerges.

**2024+ (Foundation Models):** Large-scale pre-trained models (WavLM, HuBERT, UniSpeech) fine-tuned for speaker recognition. Transformers replacing CNNs. EER reductions of 30-50% over ECAPA.

### 5.3.2 ECAPA-TDNN Architecture

ECAPA-TDNN is currently the most widely adopted state-of-the-art architecture for speaker verification:

```
┌────────────────────────────────────────┐
│           Input Features              │
│      (80-dim MFCC or Mel-Spec)        │
└────────────────┬───────────────────────┘
                 │
┌────────────────▼──────────────────────┐
│     1D Convolution Layer              │
│     (512 channels, kernel 5)          │
└────────────────┬──────────────────────┘
                 │
┌────────────────▼──────────────────────┐
│     SE-Res2Block × 3                  │
│  (Squeeze-and-Excitation + Res2Net)   │
│  Channels: 512, 1024, 1536            │
└────────────────┬──────────────────────┘
                 │
┌────────────────▼──────────────────────┐
│  Feature Aggregation Module           │
│  (Multi-layer feature summation)      │
└────────────────┬──────────────────────┘
                 │
┌────────────────▼──────────────────────┐
│     Attentive Statistics Pooling      │
│  (Self-attention weighted mean+std)   │
└────────────────┬──────────────────────┘
                 │
┌────────────────▼──────────────────────┐
│  Fully Connected Layers × 2           │
│  (1536 → 512 → Embedding Dim)          │
└────────────────┬──────────────────────┘
                 │
┌────────────────▼──────────────────────┐
│       Speaker Embedding               │
│      (192 or 256-dim vector)          │
└───────────────────────────────────────┘
```

**Key innovations of ECAPA-TDNN:**
1. **Squeeze-and-Excitation (SE) blocks:** Channel-wise attention that emphasizes informative frequency bands.
2. **Res2Net modules:** Multi-scale feature extraction within each residual block.
3. **Multi-layer feature aggregation:** Combines features from multiple layers for richer representation.
4. **Attentive statistics pooling:** Self-attention mechanism that weights each frame's contribution to the speaker embedding.

### 5.3.3 WavLM Architecture

WavLM (Microsoft, 2022+) is a large-scale pre-trained model for speech processing:

```
┌──────────────────────────────────────┐
│         Raw Audio Waveform           │
└────────────────┬─────────────────────┘
                 │
┌────────────────▼─────────────────────┐
│     CNN Encoder (7 layers)           │
│      (Downsampling 16kHz → 50Hz)     │
└────────────────┬─────────────────────┘
                 │
┌────────────────▼─────────────────────┐
│  Transformer Encoder × 12-24         │
│  (Relative position bias,            │
│   Gated relative position bias)      │
└────────────────┬─────────────────────┘
                 │
┌────────────────▼─────────────────────┐
│     Projection Layers                │
│     (Speaker embedding output)       │
└──────────────────────────────────────┘
```

WavLM is pre-trained on 94,000 hours of audio using:
- **Masked Speech Denoising:** Mask spans of input and predict the masked content
- **Speech Denoising:** Learn to transform noisy speech to clean representations
- **Speaker Permutation:** Learn invariant representations across utterance variations

When fine-tuned for speaker recognition, WavLM achieves state-of-the-art results on benchmarks like VoxCeleb.

### 5.3.4 ResNetSE (ResNet + Squeeze-and-Excitation)

A widely used architecture for speaker identification (N-speaker classification):

```python
import torch
import torch.nn as nn

class ResNetSEBlock(nn.Module):
    """ResNet block with Squeeze-and-Excitation."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        # Squeeze-and-Excitation
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        se_weight = self.se(out)
        out = out * se_weight
        out += residual
        out = self.relu(out)
        return out

class SpeakerEmbeddingExtractor(nn.Module):
    """Extract 256-dim speaker embeddings using ResNetSE."""

    def __init__(self, n_mels=80, embedding_dim=256):
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(
            ResNetSEBlock(64),
            ResNetSEBlock(64),
            ResNetSEBlock(64),
        )
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, embedding_dim)

    def forward(self, mel_spec):
        # mel_spec: (batch, 1, n_mels, time)
        x = self.frontend(mel_spec)
        x = self.res_blocks(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        embedding = self.fc(x)
        # L2 normalize
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        return embedding
```

## 5.4 Voice Print Creation and Matching

### 5.4.1 Enrollment Process

```
1. Record reference audio (3-30 seconds of clear speech)
2. Extract features (MFCC or mel-spectrogram)
3. Pass through speaker embedding model
4. Store embedding vector (voice print) in database
5. Optionally store multiple audio samples for robust enrollment
```

### 5.4.2 Verification Process

```
1. User speaks verification phrase (2-10 seconds)
2. Extract features from verification audio
3. Pass through same embedding model
4. Compare verification embedding with stored voice print
5. Compute similarity score (cosine similarity, PLDA, etc.)
6. Accept if score > threshold, else reject
```

### 5.4.3 Similarity Scoring

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch

class VoiceBiometricSystem:
    def __init__(self, model, threshold=0.65):
        self.model = model
        self.threshold = threshold
        self.voice_prints = {}  # user_id -> embedding

    def enroll(self, user_id, audio_path):
        """Enroll a user by extracting and storing their voice print."""
        embedding = self._extract_embedding(audio_path)
        self.voice_prints[user_id] = embedding
        return True

    def verify(self, user_id, audio_path):
        """Verify a user against their stored voice print."""
        if user_id not in self.voice_prints:
            return {"verified": False, "error": "User not enrolled"}

        verification_embedding = self._extract_embedding(audio_path)
        stored_embedding = self.voice_prints[user_id]

        # Cosine similarity
        similarity = cosine_similarity(
            verification_embedding.reshape(1, -1),
            stored_embedding.reshape(1, -1),
        )[0][0]

        verified = similarity >= self.threshold

        return {
            "verified": verified,
            "similarity": float(similarity),
            "threshold": self.threshold,
        }

    def identify(self, audio_path, top_k=3):
        """Identify who is speaking from enrolled users."""
        embedding = self._extract_embedding(audio_path)

        scores = []
        for user_id, stored_emb in self.voice_prints.items():
            similarity = cosine_similarity(
                embedding.reshape(1, -1),
                stored_emb.reshape(1, -1),
            )[0][0]
            scores.append((user_id, float(similarity)))

        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]

    def _extract_embedding(self, audio_path):
        """Extract speaker embedding from audio file."""
        # Load and preprocess audio
        audio = self._load_audio(audio_path)  # 16kHz mono PCM
        mel_spec = self._compute_mel_spec(audio)

        # Extract embedding
        self.model.eval()
        with torch.no_grad():
            embedding = self.model(mel_spec.unsqueeze(0))

        return embedding.numpy().flatten()

    def _load_audio(self, path):
        """Load and preprocess audio (16kHz, mono, normalize)."""
        import librosa
        audio, sr = librosa.load(path, sr=16000, mono=True)
        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-10)
        return audio

    def _compute_mel_spec(self, audio):
        """Compute mel-spectrogram features."""
        import librosa
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=16000,
            n_mels=80,
            n_fft=400,
            hop_length=160,
        )
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        # Normalize per sample
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-10)
        return torch.FloatTensor(mel_spec[np.newaxis, :, :])
```

### 5.4.4 Multi-Sample Enrollment

```python
class RobustVoiceBiometricSystem(VoiceBiometricSystem):
    def enroll(self, user_id, audio_paths):
        """Enroll with multiple audio samples for robustness."""
        embeddings = []
        for path in audio_paths:
            emb = self._extract_embedding(path)
            embeddings.append(emb)

        # Average embeddings
        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-10)

        self.voice_prints[user_id] = avg_embedding
        return True

    def verify(self, user_id, audio_paths):
        """Verify with multiple samples, use average score."""
        if user_id not in self.voice_prints:
            return {"verified": False, "error": "User not enrolled"}

        stored_embedding = self.voice_prints[user_id]
        scores = []

        for path in audio_paths:
            emb = self._extract_embedding(path)
            similarity = cosine_similarity(
                emb.reshape(1, -1),
                stored_embedding.reshape(1, -1),
            )[0][0]
            scores.append(similarity)

        avg_score = np.mean(scores)
        verified = avg_score >= self.threshold

        return {
            "verified": verified,
            "similarity": float(avg_score),
            "scores": [float(s) for s in scores],
            "threshold": self.threshold,
        }
```

## 5.5 Anti-Spoofing and Liveness Detection

Spoofing attacks are a critical concern for voice biometrics. Attack types:

### 5.5.1 Attack Types

**Replay Attack:** Record the target's voice and replay it to the biometric system.

**Voice Cloning / Synthesis:** Use TTS or voice conversion to generate the target's voice.

**Morphing:** Combine multiple speakers' voices to create a composite that matches multiple enrollments.

**Impersonation:** A human mimics the target's voice.

### 5.5.2 Countermeasures

**Liveness Detection (Active):**
- Challenge-response: "Please read this random number: 7-3-9-1"
- Requires the user to process and respond to unpredictable text
- Effective against replay and pre-recorded attacks

**Liveness Detection (Passive):**
- Analyze acoustic features for signs of recording:
  - Channel noise patterns
  - Speaker impedance characteristics
  - Microphone distortion signatures
  - Ambient noise consistency

**Anti-Spoofing Models:**
- Dedicated classifier (binary: real vs spoof) trained on spoofed audio
- Can be integrated into the speaker embedding model
- ASVspoof challenge datasets provide training/evaluation data

```python
# Anti-spoofing classifier using LFCC features
import numpy as np
import torch
import torch.nn as nn

class AntiSpoofingClassifier(nn.Module):
    """CNN-based binary classifier for spoof detection."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.3)

        # Adaptive pooling for variable-length input
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 2)  # real/spoof binary

    def forward(self, x):
        # x: (batch, 1, n_features, time)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x  # Raw scores (logits)

# Combined verification + anti-spoofing pipeline
class SecureVoiceBiometricSystem:
    def __init__(self, speaker_model, anti_spoof_model, threshold=0.65):
        self.speaker_model = speaker_model
        self.anti_spoof_model = anti_spoof_model
        self.threshold = threshold

    def verify(self, user_id, audio_path):
        # Step 1: Anti-spoofing check
        spoof_score = self._detect_spoof(audio_path)
        if spoof_score > 0.5:
            return {"verified": False, "rejected": True, "reason": "spoof_detected"}

        # Step 2: Speaker verification
        result = self.speaker_model.verify(user_id, audio_path)
        result["spoof_score"] = float(spoof_score)

        return result

    def _detect_spoof(self, audio_path):
        """Return spoof probability (0=real, 1=spoof)."""
        features = self._extract_lfcc(audio_path)
        self.anti_spoof_model.eval()
        with torch.no_grad():
            output = self.anti_spoof_model(features.unsqueeze(0))
            probs = torch.softmax(output, dim=1)
        return probs[0, 1].item()  # Spoof probability
```

### 5.5.3 Anti-Spoofing Effectiveness

| Attack Type | Detection Rate (EER) |
|-------------|---------------------|
| Replay (high quality) | 1-5% |
| Replay (low quality) | < 1% |
| Voice cloning (ElevenLabs) | 3-10% |
| Voice conversion | 5-15% |
| Human impersonation | 10-20% |

*Note: Detection rates depend heavily on the anti-spoofing model quality and training data. Dedicated anti-spoofing models significantly outperform generic speaker verification models at detecting attacks.*

## 5.6 Accuracy Metrics

### 5.6.1 Key Metrics

**False Acceptance Rate (FAR):** The proportion of impostor attempts that are incorrectly accepted. Also called False Match Rate (FMR).

**False Rejection Rate (FRR):** The proportion of genuine attempts that are incorrectly rejected. Also called False Non-Match Rate (FNMR).

**Equal Error Rate (EER):** The point at which FAR equals FRR. Lower EER = better accuracy. Standard metric for comparing speaker verification systems.

**Detection Cost Function (DCF):** Weighted combination of FAR and FRR, used by NIST SRE evaluations. C_{miss} × P_{miss} × P_{target} + C_{fa} × P_{fa} × (1 - P_{target}).

**Minimum DCF (minDCF):** DCF at the optimal threshold. Complementary to EER.

**Top-1 Accuracy:** For speaker identification (1:N), percentage of correct identifications.

### 5.6.2 Performance Benchmarks

| Model | VoxCeleb1 EER | VoxCeleb1-H EER | VoxCeleb1-E EER | Parameters |
|-------|---------------|-----------------|-----------------|------------|
| i-vector + PLDA | 4.8% | 8.5% | 6.7% | ~2M |
| x-vector (TDNN) | 3.0% | 5.8% | 4.3% | 4.2M |
| ResNetSE-34 | 1.8% | 3.9% | 2.8% | 6.8M |
| ECAPA-TDNN (C=1024) | 0.87% | 2.24% | 1.42% | 6.2M |
| ECAPA-TDNN (C=2048) | 0.80% | 2.14% | 1.38% | 20.8M |
| WavLM Base+ | 0.55% | 1.65% | 1.10% | 94.7M |
| WavLM Large | 0.37% | 1.21% | 0.79% | 316.6M |

*VoxCeleb1: Clean audio, known speakers. VoxCeleb1-H: Hard trials (same gender, similar nationality). VoxCeleb1-E: Extended trials (all variations).*

### 5.6.3 Factors Affecting Accuracy

**Audio Duration:**
- < 2 seconds: 5-15% EER (very unreliable)
- 2-5 seconds: 3-8% EER
- 5-15 seconds: 1-4% EER
- 15-60 seconds: 0.5-2% EER
- > 60 seconds: 0.3-1% EER

**Channel Mismatch:**
- Same microphone: 0.5-2% EER
- Different microphone: 1-5% EER (worsens)
- Telephone vs broadband: 2-8% EER
- Mobile vs landline: 3-7% EER

**Noise Conditions:**
- Clean (SNR > 30dB): 0.5-2% EER
- Moderate noise (SNR 15-20dB): 2-5% EER
- High noise (SNR 5-10dB): 5-15% EER
- Very noisy (SNR < 5dB): > 15% EER

**Time Between Enrollment and Verification:**
- Same day: 0.5-2% EER
- 1 week: 1-3% EER
- 1 month: 1.5-4% EER
- 6 months: 2-6% EER
- 1 year: 3-8% EER

*Voice changes over time due to aging, health conditions, and speaking habits. Periodic re-enrollment (every 3-6 months) is recommended for production systems.*

## 5.7 Speaker Diarization

Speaker diarization answers "who spoke when?" in multi-speaker audio.

### 5.7.1 Diarization Pipeline

```
┌─────────┐   ┌──────────┐   ┌───────────┐   ┌──────────┐   ┌──────────┐
│  Audio  │   │   VAD    │   │  Speaker  │   │  Cluster │   │  Resegm. │
│  Input  │──▶│(Speech / │──▶│  Change   │──▶│  (AHC /  │──▶│(Refine   │
│         │   │ Silence)  │   │  Detection│   │  Spec.)  │   │ Boundaries│
└─────────┘   └──────────┘   └───────────┘   └──────────┘   └──────────┘
```

**VAD:** Detect speech segments (remove silence/noise).

**Speaker Change Detection:** Identify when speaker changes occur (every 1-3 seconds within speech segments). Uses Bayesian Information Criterion (BIC) or neural models.

**Clustering:** Group segments by speaker identity. Common approach: Agglomerative Hierarchical Clustering (AHC) on speaker embeddings. Number of speakers may be unknown (estimated).

**Resegmentation:** Refine speaker boundaries using Viterbi decoding with speaker models. Improves boundary accuracy.

### 5.7.2 PyAnnote Audio for Diarization

```python
import torch
import librosa
from pyannote.audio import Pipeline

# Load pretrained diarization pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=os.getenv("HF_TOKEN"),
)

# Run diarization
diarization = pipeline("audio.wav")

# Iterate over speech turns
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{speaker}: {turn.start:.1f}s - {turn.end:.1f}s")

# Export to RTTM format
with open("audio.rttm", "w") as f:
    diarization.write_rttm(f)

# Get speaker-segmented audio
import soundfile as sf

audio, sr = sf.read("audio.wav")
for turn, _, speaker in diarization.itertracks(yield_label=True):
    start_sample = int(turn.start * sr)
    end_sample = int(turn.end * sr)
    speaker_audio = audio[start_sample:end_sample]
    sf.write(f"{speaker}_{turn.start:.0f}.wav", speaker_audio, sr)
```

### 5.7.3 WhisperX with Diarization

```python
import whisperx
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Transcribe with Whisper
model = whisperx.load_model("large-v3-turbo", device=device)
result = model.transcribe("meeting.wav")

# 2. Align word timestamps
align_model, metadata = whisperx.load_align_model(
    language_code=result["language"],
    device=device,
)
result = whisperx.align(
    result["segments"],
    align_model,
    metadata,
    "meeting.wav",
    device=device,
)

# 3. Diarization
diarize_model = whisperx.DiarizationPipeline(
    model_name="pyannote/speaker-diarization-3.1",
    use_auth_token=os.getenv("HF_TOKEN"),
    device=device,
)
diarize_segments = diarize_model("meeting.wav")

# 4. Assign speakers
result = whisperx.assign_word_speakers(diarize_segments, result)

# 5. Output
for segment in result["segments"]:
    speaker = segment.get("speaker", "UNKNOWN")
    text = segment["text"]
    start = segment["start"]
    end = segment["end"]
    print(f"[{start:.1f}-{end:.1f}] {speaker}: {text}")
```

## 5.8 Privacy and Regulatory Considerations

### 5.8.1 Voice as Biometric Data

Voice prints are considered biometric data under most privacy regulations:

**GDPR (EU):** Voice data is personal data. Biometric data used for identification is "special category" data requiring explicit consent. Data subjects have right to erasure ("right to be forgotten").

**CCPA/CPRA (California):** Voice data is personal information. Biometric data requires opt-out availability. Additional protections for sensitive personal information.

**HIPAA (US Healthcare):** Voice data in healthcare contexts is protected health information (PHI). Requires BAA with voice biometric providers. Encryption and access controls required.

**BIPA (Illinois):** Biometric Information Privacy Act. Requires written consent before collecting biometric data. Strict data retention and deletion requirements. Private right of action ($1,000-5,000 per violation).

**PSD2 SCA (EU Banking):** Voice biometrics can satisfy Strong Customer Authentication requirements if the system achieves EER < 0.5% and FAR < 0.1%.

### 5.8.2 Best Practices for Privacy

```python
# Privacy-preserving voice biometrics

# 1. Hash user IDs (never store raw identifiers with voice prints)
import hashlib

def hash_user_id(user_id, salt):
    """Create anonymized user identifier."""
    return hashlib.sha256(
        f"{user_id}:{salt}".encode()
    ).hexdigest()

# 2. Store encrypted voice prints
from cryptography.fernet import Fernet

encryption_key = Fernet.generate_key()
cipher = Fernet(encryption_key)

encrypted_embedding = cipher.encrypt(embedding.tobytes())

# 3. Implement data retention policy
class PrivacyAwareVoiceBiometrics:
    MAX_RETENTION_DAYS = 90
    AUTO_DELETE_INACTIVE_DAYS = 180

    def enroll(self, user_id, audio_path):
        """Enroll with privacy consent check."""
        if not self._has_user_consent(user_id):
            raise PermissionError("User consent not provided")

        embedding = self._extract_embedding(audio_path)
        encrypted = self._encrypt_embedding(embedding)

        self._store_voice_print(
            user_id=self._anonymize(user_id),
            encrypted_embedding=encrypted,
            enrollment_date=datetime.now(),
            consent_version="2024-07-01",
        )

    def verify(self, user_id, audio_path):
        """Verify with consent validation."""
        if not self._is_consent_current(user_id):
            return {"verified": False, "reason": "consent_expired"}

        # ... verification logic ...

    def delete_user_data(self, user_id):
        """Right to erasure implementation."""
        anonymized_id = self._anonymize(user_id)
        self._delete_voice_print(anonymized_id)
        self._log_deletion(user_id, "user_requested")
```

### 8.3 Consent Management

```
Consent Requirements for Voice Biometrics:

✓ Explicit: Opt-in, not opt-out
✓ Informed: User knows it's biometric data
✓ Specific: Purpose-limited (not for general use)
✓ Withdrawable: User can revoke at any time
✓ Time-limited: Re-consent periodically (annually recommended)
✓ Granular: Separate consent for enrollment and verification
```

### 5.8.4 Voice Deepfake Detection

As voice cloning quality improves, detection becomes essential:

**Acoustic Artifacts Detection:** Identify artifacts from TTS or voice conversion (unnatural formant transitions, spectral discontinuities, irregular micro-prosody).

**Channel Analysis:** Detect evidence of re-recording or playback (IMD distortion, background noise patterns, impedance signatures).

**Live Challenge-Response:** Present unpredictable challenges that require real-time cognitive processing, not just voice matching.

**Multi-Modal Fusion:** Combine voice with other biometrics (face, behavior) or contextual signals (device ID, location).

## 5.9 Implementation Guide

### 5.9.1 Deployment Considerations

**Model Selection:**
- High security (EER < 1%): WavLM Large + anti-spoofing
- Balanced (EER 1-2%): ECAPA-TDNN (C=1024) + basic anti-spoofing
- Speed-critical (EER 2-5%): ResNetSE-34 or x-vector
- Resource-constrained: i-vector + PLDA (lightweight, 0.5M params)

**Threshold Selection:**
```python
def select_threshold(validation_scores, target_far=0.01):
    """Select threshold to achieve target FAR."""
    genuine_scores, impostor_scores = validation_scores

    impostor_scores.sort()
    n_impostors = len(impostor_scores)
    threshold_idx = int(n_impostors * (1 - target_far))
    threshold = impostor_scores[threshold_idx]

    # Compute FRR at this threshold
    false_rejections = sum(1 for s in genuine_scores if s < threshold)
    frr = false_rejections / len(genuine_scores)

    return threshold, frr
```

**Multi-Factor Authentication:**
Voice alone is rarely sufficient for high-security applications. Recommended combination:

| Security Level | Factors | Example |
|----------------|---------|---------|
| Low | Voice only | Smart speaker personalization |
| Medium | Voice + knowledge (PIN) | Phone banking balance inquiry |
| High | Voice + possession (phone) | Payment authorization |
| Very High | Voice + face + context | High-value transaction |

### 5.9.2 Integration with Voice Agents

```python
# Voice biometrics in a voice agent pipeline
from pipecat.processors.frame_processor import FrameProcessor

class VoiceBiometricsProcessor(FrameProcessor):
    """Authenticate user via voice during conversation."""

    def __init__(self, biometric_system):
        super().__init__()
        self.biometric_system = biometric_system
        self.audio_buffer = []
        self.required_duration = 5.0  # Seconds of speech needed
        self.authenticated = False

    async def process_frame(self, frame, direction):
        if isinstance(frame, AudioRawFrame) and not self.authenticated:
            self.audio_buffer.append(frame.audio)
            total_duration = len(b"".join(self.audio_buffer)) / (16000 * 2)  # PCM16

            if total_duration >= self.required_duration:
                # Attempt authentication
                result = self.biometric_system.identify(
                    b"".join(self.audio_buffer)
                )

                if result[0][1] > 0.65:  # Similarity threshold
                    self.authenticated = True
                    user_id = result[0][0]
                    await self.push_frame(
                        AuthenticationFrame(user_id=user_id)
                    )
                else:
                    # Not authenticated yet, keep collecting
                    pass

        await self.push_frame(frame, direction)
```

## 5.10 Future Directions

**Passive Liveness Detection:** Non-intrusive detection of recording/synthesis attacks without user cooperation.

**Joint ASR + Speaker Recognition:** Multi-task models that simultaneously transcribe speech and identify the speaker.

**Self-Supervised Pre-training:** Models like WavLM and HuBERT that learn from unlabeled audio, reducing labeled data requirements.

**Adaptive Thresholds:** Dynamic thresholds that adjust based on call context (transaction value, risk score, user's recent history).

**Voice Template Protection:** Reversible and cancellable biometrics — if a voice print is compromised, issue a new one without changing the underlying voice.

**Cross-Lingual Speaker Recognition:** Identifying speakers across languages (training in English, verifying in Mandarin).

**Emotion-Aware Biometrics:** Accounting for emotional state in verification (a stressed voice shouldn't be falsely rejected).

---

*This document covers voice biometrics in depth. See 04-Speech-to-Text-and-Transcription.md for ASR technology, 06-Real-Time-Voice-Pipelines.md for audio pipeline integration, and 08-Telephony-AI-and-Calling-Agents.md for regulatory compliance in telephony applications.*
