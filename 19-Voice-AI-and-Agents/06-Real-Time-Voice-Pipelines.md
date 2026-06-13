# 06 — Real-Time Voice Pipelines for AI Agents

## Overview

Real-time voice pipelines form the critical infrastructure that enables AI agents to process, understand, and respond to spoken language with minimal latency. Unlike text-based interactions, voice agents must handle bidirectional audio streaming, manage jitter and packet loss, coordinate speech recognition with response generation, and deliver natural-sounding speech synthesis — all within tight latency budgets measured in milliseconds. This document provides a comprehensive technical reference for building, optimizing, and deploying real-time voice pipelines for AI agents.

## Table of Contents

1. [Architecture of Real-Time Voice Pipelines](#1-architecture-of-real-time-voice-pipelines)
2. [Audio Capture and Preprocessing](#2-audio-capture-and-preprocessing)
3. [Streaming Speech Recognition (ASR)](#3-streaming-speech-recognition-asr)
4. [Voice Activity Detection (VAD)](#4-voice-activity-detection-vad)
5. [End-of-Utterance Detection](#5-end-of-utterance-detection)
6. [Streaming Text-to-Speech (TTS)](#6-streaming-text-to-speech-tts)
7. [Latency Budgets and Optimization](#7-latency-budgets-and-optimization)
8. [WebRTC and Transport Protocols](#8-webrtc-and-transport-protocols)
9. [Barge-In and Interruption Handling](#9-barge-in-and-interruption-handling)
10. [Voice Pipeline Orchestration](#10-voice-pipeline-orchestration)
11. [Audio Codecs and Quality](#11-audio-codecs-and-quality)
12. [Error Recovery and Resiliency](#12-error-recovery-and-resiliency)
13. [Monitoring and Observability](#13-monitoring-and-observability)
14. [Frameworks and Deployment Options](#14-frameworks-and-deployment-options)
15. [References and Further Reading](#15-references-and-further-reading)

---

## 1. Architecture of Real-Time Voice Pipelines

### 1.1 High-Level Architecture

A real-time voice pipeline consists of several processing stages connected in a streaming dataflow:

```
User Microphone → Audio Capture → VAD → ASR → NLU/LLM → TTS → Audio Playback
                    ↑                                          ↓
              Jitter Buffer                              Audio Mixer
                    ↑                                          ↓
              Transport Layer (WebRTC/WS)           Transport Layer
```

### 1.2 Typical Pipeline Stages

**Stage 1 — Audio Acquisition**
- Capture audio from microphone (local) or receive audio packets (remote)
- Sample rate conversion, channel mixing, noise suppression
- Frame size: 10–30ms for real-time interaction

**Stage 2 — Voice Activity Detection (VAD)**
- Detect when human speech is present vs silence/background noise
- Segment audio into speech/no-speech regions
- Typically operates on 10–30ms frames

**Stage 3 — Automatic Speech Recognition (ASR)**
- Convert speech audio to text transcription
- Streaming mode: emit partial results as they become available
- Supports interim results (partial hypotheses) and final results

**Stage 4 — Natural Language Understanding / Processing**
- Process transcribed text through NLU pipeline or LLM
- Intent detection, entity extraction, or free-form response generation
- May interleave with ASR (query-and-response streaming)

**Stage 5 — Text-to-Speech Synthesis (TTS)**
- Convert text response to audio waveform
- Streaming mode: begin speaking before full response is generated
- Supports voice customization, emotions, SSML

**Stage 6 — Audio Output**
- Play audio through speaker (local) or transmit audio packets (remote)
- Jitter buffer management, audio mixing for barge-in

### 1.3 Key Design Principles

- **Minimize end-to-end latency**: Target sub-300ms for conversational feel
- **Support partial results**: Both ASR and TTS should stream incremental results
- **Enable interruption**: Users must be able to barge in and interrupt the agent
- **Graceful degradation**: Pipeline should handle packet loss, jitter, and bandwidth variations
- **Modular and swappable**: Each stage should be independently replaceable

---

## 2. Audio Capture and Preprocessing

### 2.1 Audio Sampling and Formats

| Parameter | Standard | Conversational | Music/Custom |
|-----------|----------|---------------|--------------|
| Sample Rate | 16 kHz | 8 kHz (telephony) | 44.1 kHz |
| Bit Depth | 16-bit | 8-bit μ-law | 24/32-bit float |
| Channels | 1 (mono) | 1 (mono) | 2 (stereo) |
| Frame Size | 20ms (320 samples @ 16kHz) | 30ms (240 samples @ 8kHz) | 10ms |
| Format | PCM S16LE | G.711 μ-law | FLAC/Opus |

### 2.2 Noise Suppression Techniques

**a) Spectral Subtraction**
- Estimate noise floor during silence periods
- Subtract noise spectrum from signal
- Simple but can introduce musical artifacts

**b) Wiener Filtering**
- Adaptive filter based on signal-to-noise ratio estimation
- Better noise reduction with fewer artifacts
- More computationally expensive

**c) Deep Learning Denoisers**
- RNNoise, DCCRN, Conv-TasNet
- Superior performance in non-stationary noise
- GPU-accelerated for real-time inference

**d) RNNoise Integration Example**

```python
import rnnoise
import numpy as np

class AudioDenoiser:
    def __init__(self, sample_rate=16000):
        self.denoiser = rnnoise.RNNoise()
        self.sample_rate = sample_rate

    def process_frame(self, audio_frame: np.ndarray) -> np.ndarray:
        """Denoise a 20ms audio frame (320 samples at 16kHz)."""
        if len(audio_frame) != 320:
            raise ValueError(f"Expected 320 samples, got {len(audio_frame)}")
        return self.denoiser.process_frame(audio_frame)

    def process_stream(self, audio_stream: np.ndarray) -> np.ndarray:
        """Denoise a full audio stream by processing frame by frame."""
        output = np.array([], dtype=np.float32)
        for i in range(0, len(audio_stream), 320):
            frame = audio_stream[i:i+320]
            if len(frame) == 320:
                output = np.append(output, self.denoiser.process_frame(frame))
        return output
```

### 2.3 Audio Preprocessing Pipeline

```python
class AudioPreprocessor:
    def __init__(self, sample_rate=16000, frame_size_ms=20):
        self.sample_rate = sample_rate
        self.frame_size = sample_rate * frame_size_ms // 1000
        self.denoiser = AudioDenoiser(sample_rate)
        self.resampler = None  # configure for sample rate conversion
        self.agc = self._create_agc()  # automatic gain control

    def _create_agc(self):
        """Configure automatic gain control."""
        # Uses WebRTC AGC or custom implementation
        return lambda audio: audio * 2.0  # simplified example

    async def process_frame(self, raw_audio: bytes) -> np.ndarray:
        """Process an incoming raw audio frame."""
        # Convert bytes to numpy array
        audio_array = np.frombuffer(raw_audio, dtype=np.int16)
        audio_array = audio_array.astype(np.float32) / 32768.0

        # Apply denoising
        audio_array = self.denoiser.process_frame(audio_array)

        # Apply automatic gain control
        audio_array = self.agc(audio_array)

        # Apply pre-emphasis filter (optional)
        audio_array = self._pre_emphasis(audio_array)

        return audio_array

    def _pre_emphasis(self, audio: np.ndarray, coeff=0.97) -> np.ndarray:
        """Apply pre-emphasis filter to boost high frequencies."""
        return np.append(audio[0], audio[1:] - coeff * audio[:-1])
```

---

## 3. Streaming Speech Recognition (ASR)

### 3.1 Streaming vs Batch ASR

| Feature | Streaming ASR | Batch ASR |
|---------|--------------|-----------|
| Latency | 100–500ms (first result) | Seconds to minutes |
| Output | Incremental partial results | Complete transcript only |
| Use Case | Real-time voice agents | Transcription, subtitles |
| Memory | Process-per-frame | Process entire audio |
| Accuracy | Slightly lower (less context) | Higher (full context) |

### 3.2 Streaming ASR Architecture

```
Audio Frames → Feature Extraction (Fbank/MFCC) → Encoder → Decoder → Text

                    ┌─────────────┐
Audio Stream ──────▶│  Feature    │──────▶┌──────────┐
                    │  Extractor  │       │ Encoder  │
                    └─────────────┘       └────┬─────┘
                                              │
                    ┌─────────────┐           ▼
                    │  Decoder    │◀───── Encoded Features
                    │  (CTC/RNNT) │
                    └──────┬──────┘
                           │
                           ▼
                    Partial Text Output
                    (emitted every frame/step)
```

### 3.3 Key Streaming ASR Models

**a) Whisper (OpenAI)**
- Large-scale multilingual ASR
- Streaming via WhisperX or faster-whisper
- Chunked processing with overlapping windows
- Typical latency: 500ms–2s (depends on model size)

**b) Wav2Vec 2.0 / HuBERT (Meta)**
- Self-supervised learning approach
- CTC-based streaming
- Lower latency than Whisper for streaming
- Fine-tuned for specific domains

**c) Conformer-Transducer (NVIDIA)**
- Streaming RNNT architecture
- State-of-the-art accuracy with low latency
- Optimized with TensorRT for GPU inference
- Used in production voice assistants

**d) Paraformer (Alibaba DAMO Academy)**
- Non-autoregressive streaming ASR
- Parallel decoding for lower latency
- Good for real-time applications
- Supported in FunASR toolkit

**e) Kaldi / Vosk**
- Traditional hybrid ASR systems
- Very low latency (50–100ms)
- Lightweight, runs on CPU
- Limited accuracy compared to end-to-end models

### 3.4 Streaming ASR Implementation

```python
import asyncio
import numpy as np

class StreamingASR:
    def __init__(self, model_type="whisper", language="en",
                 sample_rate=16000, chunk_size_sec=0.32):
        self.model_type = model_type
        self.language = language
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_size_sec)
        self.buffer = np.array([], dtype=np.float32)
        self.model = self._load_model()

    def _load_model(self):
        if self.model_type == "whisper":
            from faster_whisper import WhisperModel
            return WhisperModel("base", device="cpu",
                                compute_type="int8")
        elif self.model_type == "wav2vec":
            # Load Wav2Vec 2.0 model
            pass
        else:
            raise ValueError(f"Unsupported model: {self.model_type}")

    async def transcribe_chunk(self, audio_chunk: np.ndarray) -> str:
        """Transcribe an audio chunk and return partial text."""
        self.buffer = np.append(self.buffer, audio_chunk)

        # Process only if we have enough audio
        if len(self.buffer) < self.sample_rate * 0.5:  # min 0.5s
            return ""

        if self.model_type == "whisper":
            segments, info = self.model.transcribe(
                self.buffer, beam_size=1, language=self.language
            )
            text = " ".join(seg.text for seg in segments)
            return text
        return ""

    def reset(self):
        """Reset the ASR state for a new utterance."""
        self.buffer = np.array([], dtype=np.float32)

    async def transcribe_full(self, audio: np.ndarray) -> str:
        """Transcribe a complete audio segment (non-streaming fallback)."""
        segments, info = self.model.transcribe(
            audio, beam_size=5, language=self.language
        )
        return " ".join(seg.text for seg in segments)
```

### 3.5 Server-Side Streaming with gRPC

```protobuf
service StreamingASR {
  // Bidirectional streaming for real-time ASR
  rpc Recognize(stream RecognizeRequest) returns (stream RecognizeResponse);
}

message RecognizeRequest {
  oneof streaming_request {
    RecognitionConfig config = 1;  // First message only
    bytes audio_content = 2;       // Subsequent messages
  }
}

message RecognizeResponse {
  repeated RecognitionResult results = 1;
  bool is_final = 2;
  int64 end_time_ms = 3;
}

message RecognitionResult {
  string transcript = 1;
  float confidence = 2;
  repeated WordInfo words = 3;
  bool is_partial = 4;
}

message WordInfo {
  string word = 1;
  float start_time_ms = 2;
  float end_time_ms = 3;
  float confidence = 4;
}
```

---

## 4. Voice Activity Detection (VAD)

### 4.1 VAD Purpose and Importance

Voice Activity Detection segments audio into speech and non-speech regions. It is critical for:
- Reducing ASR processing on non-speech segments (cost savings)
- Determining when to start and stop transcription
- Enabling natural turn-taking in conversations
- Supporting barge-in interruption detection

### 4.2 VAD Algorithms

**a) Energy-Based VAD**
- Compare signal energy against a threshold
- Simple and fast but poor performance in noisy environments
- Threshold should be adaptive (track noise floor)

**b) Spectral-Based VAD**
- Analyze frequency content to distinguish speech from noise
- Speech has characteristic formant structure
- Better noise immunity than energy-based

**c) Machine Learning VAD**
- WebRTC VAD: Gaussian mixture model in the spectral domain
- Silero VAD: Pre-trained LSTM-based model
- MarbleNet: Convolutional neural network approach

### 4.3 Silero VAD Implementation

```python
import torch
import numpy as np

class SileroVAD:
    def __init__(self, sample_rate=16000, threshold=0.5):
        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False
        )
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.triggered = False
        self.temp_end = 0
        self.speech_segments = []

    def process_frame(self, audio_frame: np.ndarray) -> dict:
        """Process a single audio frame (typically 30ms/512 samples)."""
        audio_tensor = torch.from_numpy(audio_frame).float()

        with torch.no_grad():
            speech_prob = self.model(audio_tensor, self.sample_rate).item()

        result = {
            "speech_probability": speech_prob,
            "is_speech": speech_prob >= self.threshold,
            "triggered": self.triggered,
        }
        return result

    def process_stream(self, audio_stream: np.ndarray,
                       frame_size=512) -> list:
        """Process an audio stream and identify speech segments."""
        frames = []
        for i in range(0, len(audio_stream), frame_size):
            frame = audio_stream[i:i+frame_size]
            if len(frame) < frame_size:
                frame = np.pad(frame, (0, frame_size - len(frame)))
            result = self.process_frame(frame)
            frames.append(result)
        return frames

    def detect_speech_segments(self, audio: np.ndarray,
                               frame_size=512) -> list:
        """Return list of (start_sample, end_sample) speech segments."""
        results = self.process_stream(audio, frame_size)
        segments = []
        in_speech = False
        start = 0

        for i, r in enumerate(results):
            if r["is_speech"] and not in_speech:
                in_speech = True
                start = i * frame_size
            elif not r["is_speech"] and in_speech:
                in_speech = False
                segments.append((start, i * frame_size))

        if in_speech:
            segments.append((start, len(audio)))

        return segments
```

### 4.4 WebRTC VAD Comparison

| Feature | WebRTC VAD | Silero VAD | MarbleNet |
|---------|-----------|------------|-----------|
| Model Size | ~50KB | ~3MB | ~10MB |
| Latency per frame | <1ms | ~3ms (CPU) | ~5ms |
| Accuracy (noisy) | Moderate | High | Very High |
| Accuracy (clean) | Good | Very High | Very High |
| Configuration | 3 modes (aggressive) | Threshold parameter | Threshold parameter |
| Language | Agnostic | Agnostic | Agnostic |
| License | BSD | MIT | Apache 2.0 |

---

## 5. End-of-Utterance Detection

### 5.1 EOU Detection Strategies

**a) Silence-Based EOU**
- Detect when speech stops and silence exceeds a threshold
- Simple but can cut off speaker during pauses
- Typical silence threshold: 500–1500ms

**b) Semantic EOU**
- Use ASR decoder to predict end-of-turn
- Look for sentence endings, question marks, trailing off
- More natural but requires ASR integration

**c) Hybrid EOU**
- Combine silence detection with semantic prediction
- Use silence as primary, semantic as confirmation
- Reduces false positives and premature cutoffs

### 5.2 EOU Decision Logic

```python
class EndOfUtteranceDetector:
    def __init__(self, silence_threshold_ms=800,
                 min_utterance_ms=300, max_utterance_ms=30000):
        self.silence_threshold_ms = silence_threshold_ms
        self.min_utterance_ms = min_utterance_ms
        self.max_utterance_ms = max_utterance_ms
        self.speech_start_time = None
        self.last_speech_time = None
        self.silence_duration = 0
        self.total_speech_ms = 0

    def process_vad_result(self, result: dict,
                           current_time_ms: float) -> str:
        """
        Process VAD result and return EOU decision.
        Returns: 'continue', 'utterance_end', 'timeout'
        """
        if result["is_speech"]:
            if self.speech_start_time is None:
                self.speech_start_time = current_time_ms
            self.last_speech_time = current_time_ms
            self.silence_duration = 0
            self.total_speech_ms = current_time_ms - self.speech_start_time

            # Check max utterance duration
            if self.total_speech_ms >= self.max_utterance_ms:
                return "utterance_end"
        else:
            if self.speech_start_time is not None:
                self.silence_duration = current_time_ms - self.last_speech_time

        # Check silence threshold
        if self.silence_duration >= self.silence_threshold_ms:
            if self.total_speech_ms >= self.min_utterance_ms:
                return "utterance_end"

        return "continue"

    def reset(self):
        self.speech_start_time = None
        self.last_speech_time = None
        self.silence_duration = 0
        self.total_speech_ms = 0
```

### 5.3 Advanced EOU with ASR Integration

```python
class SemanticEOUDetector:
    def __init__(self, asr_model):
        self.asr = asr_model
        self.eou_threshold = 0.7
        self.last_partial_text = ""
        self.no_speech_frames = 0
        self.max_no_speech_frames = 40  # ~800ms at 20ms frames

    async def check_eou(self, partial_text: str,
                        is_final: bool, vad_speech: bool) -> bool:
        """Check if utterance has ended using semantic and silence cues."""
        # Check for final ASR result
        if is_final:
            return True

        # Check for sentence-ending punctuation in partial results
        if partial_text and partial_text != self.last_partial_text:
            self.last_partial_text = partial_text
            self.no_speech_frames = 0

            # Semantic end-of-turn markers
            eou_markers = ["?", ".", "!", "...", "thank you",
                           "bye", "goodbye", "that's all"]
            for marker in eou_markers:
                if partial_text.rstrip().endswith(marker):
                    return True

        # Silence-based fallback
        if not vad_speech:
            self.no_speech_frames += 1
            if self.no_speech_frames >= self.max_no_speech_frames:
                return True
        else:
            self.no_speech_frames = 0

        return False
```

---

## 6. Streaming Text-to-Speech (TTS)

### 6.1 Streaming TTS Architecture

```
Input Text → Text Frontend → Acoustic Model → Vocoder → Audio Output
                 ↓                ↓              ↓          ↓
           Tokenization      Mel-Spectrogram   Waveform   Chunked
           Normalization     Generation        Synthesis  Streaming
           Prosody Prediction
```

### 6.2 Low-Latency TTS Models

**a) VITS (Conditional Variational Autoencoder)**
- End-to-end TTS from text to waveform
- Latency: ~200ms for first audio
- Monotonic alignment for streaming
- High voice quality

**b) Piper TTS**
- Optimized for edge devices (Raspberry Pi, mobile)
- Latency: <100ms on GPU, <500ms on CPU
- VITS-based architecture
- 50+ voices available

**c) Coqui AI TTS**
- Open-source with multiple model architectures
- Supports fine-tuning and voice cloning
- Latency: 300ms–1s depending on model

**d) Microsoft Edge TTS (Neural)**
- Cloud-based neural TTS
- Extremely natural voices
- Latency: 200–500ms including network
- Rich SSML support

**e) ElevenLabs**
- Ultra-realistic AI voices
- Streaming API with low latency
- Voice cloning and generation
- Emotional control and style

### 6.3 Streaming TTS Implementation

```python
import asyncio
import numpy as np

class StreamingTTS:
    def __init__(self, model_type="piper", voice="en_US-less-low",
                 sample_rate=22050):
        self.model_type = model_type
        self.voice = voice
        self.sample_rate = sample_rate
        self.synthesizer = self._load_synthesizer()
        self.audio_buffer = b""
        self.streaming = False

    def _load_synthesizer(self):
        if self.model_type == "piper":
            from piper import PiperVoice
            import wave
            return PiperVoice.load(self.voice)
        elif self.model_type == "coqui":
            from TTS.api import TTS
            return TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
        else:
            raise ValueError(f"Unsupported TTS model: {self.model_type}")

    async def synthesize_chunk(self, text: str,
                               play_immediately=True) -> np.ndarray:
        """Synthesize text and return audio chunk."""
        if self.model_type == "piper":
            audio = self.synthesizer.synthesize(text)
            return audio
        elif self.model_type == "coqui":
            audio = self.synthesizer.tts(text)
            return np.array(audio, dtype=np.float32)

    async def stream_sentence(self, sentence: str,
                              chunk_size_ms=100) -> list:
        """Stream a sentence in audio chunks for real-time playback."""
        audio = await self.synthesize_chunk(sentence)
        chunk_samples = int(self.sample_rate * chunk_size_ms / 1000)

        chunks = []
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i+chunk_samples]
            chunks.append(chunk)
            await asyncio.sleep(0)  # yield control

        return chunks

    def text_to_audio_bytes(self, text: str) -> bytes:
        """Convert text to PCM audio bytes."""
        audio = self.synthesizer.synthesize(text)
        return (audio * 32767).astype(np.int16).tobytes()
```

### 6.4 SSML for Expressive TTS

```xml
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis">
  <voice name="en-US-JennyNeural">
    <prosody rate="medium" pitch="default">
      <emphasis level="moderate">
        Welcome to our customer service.
      </emphasis>
      <break time="500ms"/>
      How can I
      <prosody rate="slow">help you today</prosody>
      ?
    </prosody>
  </voice>
</speak>
```

### 6.5 TTS Playback Strategies

**a) Chunked Streaming**
- Begin playback as soon as first audio chunk is available
- Overlap TTS generation with audio playback
- Target: <200ms from text generation to first audio

**b) Sentence-Level Synthesis**
- Synthesize one sentence at a time
- Natural prosody across full sentence
- Latency: sentence duration × 0.3 (approximate real-time factor)

**c) Word-Level Synthesis**
- Synthesize word by word for minimum latency
- Can sound choppy without prosody planning
- Latency: <50ms per word

**d) Predictive Pre-fetching**
- Predict likely next phrases using LLM
- Pre-synthesize audio before text is finalized
- Risk of wasted computation if prediction is wrong

---

## 7. Latency Budgets and Optimization

### 7.1 Target Latency Budget

| Segment | Target | Hard Limit | Component |
|---------|--------|-----------|-----------|
| Audio Capture | 10ms | 30ms | Microphone, A/D |
| Noise Suppression | 5ms | 20ms | Preprocessing |
| VAD | 5ms | 10ms | Voice Activity Detection |
| ASR | 150ms | 300ms | Speech Recognition |
| NLU/LLM | 200ms | 1000ms | Understanding |
| TTS | 150ms | 300ms | Speech Synthesis |
| Network RTT | 50ms | 150ms | Transport |
| **Total** | **~570ms** | **~1810ms** | |

### 7.2 Optimization Techniques

**a) ASR Optimization**
- Use smaller models (Whisper tiny vs large)
- Enable prediction profiling (pre-warm model)
- Use batched inference for parallel requests
- Optimize input features (Fbank extraction on GPU)

**b) LLM Optimization**
- Use speculative decoding (draft model + target model)
- Implement prefix caching for common contexts
- Use quantized models (INT8, FP8, INT4)
- Enable KV-cache optimization

**c) TTS Optimization**
- Use lightweight vocoder (LPCNet vs WaveNet)
- Cache frequently spoken phrases (greetings, confirmations)
- Use latency profiling to overlap synthesis with playback

**d) Network Optimization**
- Use WebRTC with UDP for low-latency transport
- Enable opus audio codec with optimized bitrate
- Use regional edge servers (CDN for voice)
- Implement adaptive bitrate based on network conditions

### 7.3 Latency Monitoring Template

```yaml
latency_monitoring:
  pipeline_id: "voice-agent-prod-v2"
  sample_period_ms: 60000  # 1-minute rolling window
  
  percentiles:
    p50_total_ms: 420
    p95_total_ms: 890
    p99_total_ms: 1450
  
  stage_breakdown_ms:
    capture:
      p50: 8
      p95: 15
      p99: 28
    noise_suppression:
      p50: 4
      p95: 8
      p99: 18
    vad:
      p50: 3
      p95: 6
      p99: 12
    asr:
      p50: 120
      p95: 250
      p99: 480
    llm:
      p50: 180
      p95: 450
      p99: 920
    tts:
      p50: 85
      p95: 160
      p99: 300
    network:
      p50: 20
      p95: 50
      p99: 120
  
  slas:
    p50_total: "< 500ms"  # Status: PASS
    p95_total: "< 1000ms" # Status: WARNING (890ms)
    p99_total: "< 2000ms" # Status: PASS
```

### 7.4 Real-Time Factor (RTF) Optimization

RTF = Processing Time / Audio Duration

- **Target RTF**: < 0.3 (process 1s of audio in <300ms)
- **Excellent**: < 0.1
- **Good**: 0.1–0.3
- **Acceptable**: 0.3–0.5
- **Poor**: > 0.5 (not suitable for real-time)

---

## 8. WebRTC and Transport Protocols

### 8.1 WebRTC for Voice Agents

WebRTC is the preferred transport for real-time voice due to:
- UDP-based low-latency media transport
- Built-in echo cancellation (AEC)
- Jitter buffer management
- Adaptive bitrate (via GCC)
- NAT traversal (ICE/STUN/TURN)
- DTLS-SRTP encryption

### 8.2 WebRTC Media Constraints

```javascript
// WebRTC peer connection configuration
const peerConfig = {
  iceServers: [
    { urls: 'stun:stun.l.google.com:19302' },
    {
      urls: 'turn:turn.example.com:3478',
      username: 'user',
      credential: 'pass'
    }
  ]
};

// Audio constraints for voice agent
const audioConstraints = {
  audio: {
    sampleRate: 16000,
    sampleSize: 16,
    channelCount: 1,
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: true,
    latencyHint: 'interactive'  // prefer low latency
  }
};
```

### 8.3 Alternative Transport Protocols

| Protocol | Latency | Reliability | Use Case |
|----------|---------|-------------|----------|
| WebRTC | 50–150ms | Moderate (UDP) | Real-time conversation |
| WebSocket | 100–300ms | High (TCP) | Server-side voice agents |
| gRPC Bidirectional | 50–200ms | High (HTTP/2) | Backend pipeline services |
| RTP | 50–100ms | Low (pure UDP) | Custom voice infrastructure |
| SIP | 100–500ms | Moderate | Telephony integration |

### 8.4 WebSocket Voice Transport Example

```python
import asyncio
import websockets
import json
import numpy as np

class WebSocketVoiceTransport:
    def __init__(self, uri="ws://localhost:8765"):
        self.uri = uri
        self.websocket = None
        self.connected = False

    async def connect(self):
        self.websocket = await websockets.connect(self.uri)
        self.connected = True

        # Send initial configuration
        config = {
            "type": "config",
            "sample_rate": 16000,
            "channels": 1,
            "codec": "pcm16",
            "frame_size_ms": 20,
            "vad_enabled": True,
            "asr_model": "whisper-tiny",
            "tts_voice": "en-US-JennyNeural"
        }
        await self.websocket.send(json.dumps(config))

    async def send_audio(self, audio_frame: bytes):
        """Send an audio frame to the server."""
        if not self.connected:
            raise ConnectionError("Not connected")
        message = {
            "type": "audio",
            "data": audio_frame.hex(),
            "timestamp": time.time_ns()
        }
        await self.websocket.send(json.dumps(message))

    async def receive_messages(self):
        """Receive messages from the server."""
        while self.connected:
            try:
                message = await self.websocket.recv()
                data = json.loads(message)

                if data["type"] == "transcript":
                    yield ("transcript", data["text"], data.get("is_final", False))
                elif data["type"] == "audio_response":
                    audio_bytes = bytes.fromhex(data["data"])
                    yield ("audio", audio_bytes, data.get("format", "pcm16"))
                elif data["type"] == "state":
                    yield ("state", data["state"], None)

            except websockets.ConnectionClosed:
                self.connected = False
                break

    async def disconnect(self):
        if self.websocket:
            await self.websocket.close()
            self.connected = False
```

---

## 9. Barge-In and Interruption Handling

### 9.1 Barge-In Architecture

Barge-in allows the user to interrupt the agent while it is speaking. This is essential for natural conversation flow.

```
Agent Speaking → User Starts Speaking → VAD Detects Speech →
  → TTS Paused → ASR Processes User Speech →
  → Agent Stops Responding → Agent Processes New Input →
  → Agent Generates New Response
```

### 9.2 Barge-In Detection

```python
class BargeInDetector:
    def __init__(self, vad_model, asr_model,
                 interruption_silence_ms=300):
        self.vad = vad_model
        self.asr = asr_model
        self.interruption_threshold = interruption_silence_ms
        self.agent_is_speaking = False
        self.user_speech_during_agent = []

    def on_agent_start_speaking(self):
        self.agent_is_speaking = True
        self.user_speech_during_agent = []

    def on_agent_stop_speaking(self):
        self.agent_is_speaking = False

    async def process_frame(self, audio_frame: np.ndarray) -> dict:
        """Process audio frame and detect potential barge-in."""
        vad_result = self.vad.process_frame(audio_frame)

        if self.agent_is_speaking and vad_result["is_speech"]:
            self.user_speech_during_agent.append(audio_frame)

            # Check if user speech is sustained
            if len(self.user_speech_during_agent) > 10:  # ~200ms
                return {
                    "barge_in_detected": True,
                    "confidence": self._estimate_interruption_confidence(),
                    "agent_should_stop": True
                }

        return {
            "barge_in_detected": False,
            "confidence": 0.0,
            "agent_should_stop": False
        }

    def _estimate_interruption_confidence(self) -> float:
        """Estimate confidence that this is a genuine interruption."""
        # Factors: duration of user speech, energy level,
        # whether user is saying something meaningful
        user_audio = np.concatenate(self.user_speech_during_agent)
        energy = np.mean(np.abs(user_audio))

        # Normalize to 0-1
        conf = min(1.0, len(self.user_speech_during_agent) / 30.0)
        conf *= min(1.0, energy / 0.1)

        return conf
```

### 9.3 Interruption Response Strategy

```python
class InterruptionManager:
    def __init__(self):
        self.agent_is_speaking = False
        self.tts_stream = None
        self.barge_in_detector = BargeInDetector(None, None)

    async def handle_interruption(self):
        """Handle an interruption event."""
        if not self.agent_is_speaking:
            return

        # 1. Stop TTS playback immediately
        await self._stop_tts()

        # 2. Send stop marker to client
        await self._send_stop_signal()

        # 3. Log interruption for analytics
        self._log_interruption()

        # 4. Wait for user to finish speaking
        user_text = await self._capture_user_utterance()

        # 5. Process user's new input and generate response
        response = await self._process_new_input(user_text)

        # 6. Resume speaking with new response
        await self._start_tts(response)

    async def _stop_tts(self):
        """Stop TTS playback immediately."""
        if self.tts_stream:
            self.tts_stream.stop()
            self.tts_stream = None

    async def _send_stop_signal(self):
        """Tell the client to stop audio playback."""
        # Send WebSocket message or WebRTC signaling
        pass

    async def _capture_user_utterance(self) -> str:
        """Wait for user to finish speaking and return transcript."""
        # This would integrate with VAD + ASR pipeline
        pass

    def _log_interruption(self):
        """Log interruption for analytics."""
        # Track: frequency, timing, user intent during interruption
        pass
```

---

## 10. Voice Pipeline Orchestration

### 10.1 Orchestration Framework

```python
class VoicePipelineOrchestrator:
    def __init__(self):
        self.preprocessor = AudioPreprocessor()
        self.vad = SileroVAD()
        self.asr = StreamingASR()
        self.tts = StreamingTTS()
        self.eou = EndOfUtteranceDetector()
        self.barge_in = BargeInDetector(self.vad, self.asr)
        self.state = "idle"  # idle, listening, processing, speaking

    async def process_stream(self, audio_stream: asyncio.Queue,
                             text_output: asyncio.Queue,
                             audio_output: asyncio.Queue):
        """Main pipeline processing loop."""
        while True:
            audio_frame = await audio_stream.get()

            if audio_frame is None:  # End of stream
                break

            # Preprocess audio
            processed = await self.preprocessor.process_frame(audio_frame)

            # Check for barge-in
            barge_in = await self.barge_in.process_frame(processed)
            if barge_in["agent_should_stop"]:
                await self._handle_barge_in(text_output, audio_output)
                continue

            # Voice activity detection
            vad_result = self.vad.process_frame(processed)

            # End-of-utterance detection
            eou_decision = self.eou.process_vad_result(
                vad_result, time.time() * 1000
            )

            if eou_decision == "utterance_end":
                # Process complete utterance
                utterance_audio = self._get_utterance_audio()
                transcript = await self.asr.transcribe_full(utterance_audio)
                await text_output.put(transcript)

                # Generate response
                response_text = await self._generate_response(transcript)

                # Synthesize and play response
                await self._synthesize_and_play(response_text, audio_output)

                # Reset for next utterance
                self.eou.reset()
                self.asr.reset()
            else:
                # Streaming ASR for partial results
                partial = await self.asr.transcribe_chunk(processed)
                if partial:
                    await text_output.put(("partial", partial))
```

### 10.2 State Machine for Voice Pipeline

```
IDLE → LISTENING (on user speech detected)
LISTENING → PROCESSING (on end of utterance)
LISTENING → IDLE (on timeout)
PROCESSING → SPEAKING (on response ready)
SPEAKING → LISTENING (on barge-in)
SPEAKING → IDLE (on response complete)
```

---

## 11. Audio Codecs and Quality

### 11.1 Codec Comparison for Voice

| Codec | Bitrate | Latency | Quality | Use Case |
|-------|---------|---------|---------|----------|
| Opus | 6–510 kbps | 5–60ms | Excellent | WebRTC, VoIP |
| G.711 μ-law | 64 kbps | 0.125ms | Good | Telephony (PSTN) |
| G.722 | 48–64 kbps | 2ms | Good | Wideband telephony |
| G.729 | 8 kbps | 15ms | Fair | Narrowband VoIP |
| Speex | 2.4–44.2 kbps | 30ms | Fair | Legacy VoIP |
| Silk | 6–40 kbps | 25ms | Very Good | Skype, older WebRTC |
| GSM-FR | 13 kbps | 20ms | Fair | Legacy GSM |
| PCM (raw) | 256 kbps | 0ms | Excellent | Local processing |

### 11.2 Opus Configuration for Voice Agents

```python
def configure_opus_for_voice(sample_rate=16000, bitrate=32000):
    """Configure Opus codec for optimal voice agent quality."""
    import opuslib

    application = opuslib.APPLICATION_VOIP  # Optimized for voice
    frame_size_ms = 20  # Standard frame size

    encoder = opuslib.Encoder(sample_rate, 1, application)
    encoder.bitrate = bitrate
    encoder.complexity = 5  # 0-10, tradeoff quality vs CPU
    encoder.signal = opuslib.SIGNAL_VOICE  # Voice-optimized mode
    encoder.inband_fec = True  # Forward error correction
    encoder.packet_loss_perc = 15  # Expected packet loss %
    encoder.dtx = True  # Discontinuous Transmission (silence suppression)

    return encoder
```

### 11.3 Audio Quality Metrics

- **PESQ (Perceptual Evaluation of Speech Quality)**: -0.5 to 4.5
- **POLQA (Perceptual Objective Listening Quality Assessment)**: 1-5
- **MOS (Mean Opinion Score)**: 1-5 (subjective)
- **STOI (Short-Time Objective Intelligibility)**: 0-1
- **SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)**: dB

---

## 12. Error Recovery and Resiliency

### 12.1 Common Failure Modes

| Failure | Symptom | Recovery Strategy |
|---------|---------|-------------------|
| Packet Loss | Audio glitches, missing words | FEC, PLC (Packet Loss Concealment) |
| Jitter | Variable delay, buffer under/overrun | Adaptive jitter buffer |
| ASR Timeout | No transcription | Retry with degraded model |
| TTS Failure | No audio output | Use cached phrase, escalate |
| Network Drop | Connection lost | Reconnect, state recovery |
| High Latency | Delayed response | Degrade model size, skip enhancement |
| CPU/GPU Overload | Pipeline stall | Throttle input, reduce frame rate |

### 12.2 Resilience Pattern

```python
class ResilientVoicePipeline:
    def __init__(self):
        self.primary_asr = StreamingASR("whisper-base")
        self.fallback_asr = StreamingASR("whisper-tiny")
        self.jitter_buffer = AdaptiveJitterBuffer(target_latency_ms=100)
        self.failure_count = 0
        self.max_failures = 3

    async def transcribe_with_fallback(self, audio: np.ndarray) -> str:
        """Attempt transcription with fallback on failure."""
        try:
            result = await asyncio.wait_for(
                self.primary_asr.transcribe_chunk(audio),
                timeout=2.0
            )
            self.failure_count = 0
            return result
        except asyncio.TimeoutError:
            self.failure_count += 1
            if self.failure_count >= self.max_failures:
                # Degrade to fallback model
                return await self.fallback_asr.transcribe_chunk(audio)
            return ""

    async def recover_connection(self, session_state: dict):
        """Recover voice session after connection drop."""
        # Re-establish transport
        await self.transport.connect()

        # Restore session state
        await self.transport.send_state(session_state)

        # Play reconnection prompt
        tts_audio = await self.tts.synthesize_chunk(
            "I'm sorry, I lost audio for a moment. Can you repeat that?"
        )
        return tts_audio
```

### 12.3 Jitter Buffer Implementation

```python
import collections

class AdaptiveJitterBuffer:
    def __init__(self, target_latency_ms=100, max_size_ms=500,
                 sample_rate=16000):
        self.target_latency = int(sample_rate * target_latency_ms / 1000)
        self.max_size = int(sample_rate * max_size_ms / 1000)
        self.buffer = collections.deque()
        self.buffered_samples = 0
        self.sample_rate = sample_rate
        self._adaptive_factor = 1.0

    def add_frame(self, audio_frame: np.ndarray):
        """Add an audio frame to the jitter buffer."""
        self.buffer.append(audio_frame)
        self.buffered_samples += len(audio_frame)

        # Trim if buffer exceeds maximum
        while self.buffered_samples > self.max_size:
            oldest = self.buffer.popleft()
            self.buffered_samples -= len(oldest)

    def get_playback_frame(self, frame_size: int) -> np.ndarray:
        """Get an audio frame for playback from the buffer."""
        if self.buffered_samples < self.target_latency:
            # Not enough data, stretch or return silence
            if self.buffered_samples > 0:
                return self._get_available(frame_size)
            return np.zeros(frame_size, dtype=np.float32)

        return self._get_available(frame_size)

    def _get_available(self, frame_size: int) -> np.ndarray:
        """Read frame_size samples from the buffer."""
        output = np.array([], dtype=np.float32)
        while len(output) < frame_size and self.buffer:
            frame = self.buffer.popleft()
            self.buffered_samples -= len(frame)
            output = np.append(output, frame)

        # Pad if we don't have enough
        if len(output) < frame_size:
            output = np.pad(output, (0, frame_size - len(output)))

        return output[:frame_size]

    def adapt_to_network(self, packet_loss: float, rtt_ms: float):
        """Adapt jitter buffer based on network conditions."""
        if packet_loss > 0.05 or rtt_ms > 200:
            self.target_latency = int(self.sample_rate * 0.2)  # 200ms
        else:
            self.target_latency = int(self.sample_rate * 0.1)  # 100ms
```

---

## 13. Monitoring and Observability

### 13.1 Key Voice Pipeline Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| End-to-End Latency | Time from user speech to agent response | > 2s p95 |
| ASR Latency | Time to transcribe speech | > 500ms p95 |
| TTS Latency | Time to synthesize response | > 500ms p95 |
| Packet Loss | % of audio packets lost | > 5% |
| Jitter | Variation in packet arrival | > 100ms |
| VAD Accuracy | % correct speech/non-speech classification | < 95% |
| Barge-In Detection Rate | % of interruptions correctly detected | < 90% |
| ASR Word Error Rate | % transcription errors | > 10% |
| TTS MOS | Mean opinion score of synthesized speech | < 3.5 |
| Call Drop Rate | % of sessions that drop unexpectedly | > 1% |
| Pipeline Error Rate | % of processing steps that error | > 2% |

### 13.2 Monitoring Dashboard Template

```yaml
voice_pipeline_dashboard:
  title: "Real-Time Voice Pipeline Monitoring"
  refresh_interval: 10s
  
  panels:
    - title: "Latency (p95, ms)"
      metrics: ["e2e_latency", "asr_latency", "tts_latency", "network_rtt"]
      type: "time_series"
    
    - title: "Audio Quality"
      metrics: ["packet_loss_pct", "jitter_ms", "mos_score"]
      type: "time_series"
    
    - title: "Pipeline Health"
      metrics: ["vad_accuracy", "asr_wer", "error_rate"]
      type: "single_stat"
    
    - title: "Session Activity"
      metrics: ["active_sessions", "sessions_per_minute", "avg_duration_sec"]
      type: "time_series"
    
    - title: "Resource Usage"
      metrics: ["gpu_util_pct", "cpu_util_pct", "memory_gb"]
      type: "time_series"
    
    - title: "Barge-In Stats"
      metrics: ["barge_in_rate", "interruption_success_rate"]
      type: "single_stat"
```

### 13.3 Structured Logging for Voice Pipelines

```python
import structlog

logger = structlog.get_logger()

class VoicePipelineLogger:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.log = logger.bind(session_id=session_id)

    def log_audio_frame(self, seq: int, size_bytes: int,
                        rtp_timestamp: int):
        self.log.debug("audio_frame", seq=seq, size=size_bytes,
                       rtp_ts=rtp_timestamp)

    def log_transcription(self, text: str, is_final: bool,
                          latency_ms: float, confidence: float):
        self.log.info("transcription", text=text, final=is_final,
                      latency=latency_ms, confidence=confidence)

    def log_response(self, text: str, latency_ms: float,
                     tts_duration_ms: float):
        self.log.info("response", text=text, latency=latency_ms,
                      tts_duration=tts_duration_ms)

    def log_barge_in(self, user_text: str, agent_was_saying: str,
                     interruption_latency_ms: float):
        self.log.info("barge_in", user_text=user_text,
                      agent_context=agent_was_saying,
                      latency=interruption_latency_ms)

    def log_error(self, component: str, error: str,
                  recoverable: bool):
        self.log.error("pipeline_error", component=component,
                       error=error, recoverable=recoverable)
```

---

## 14. Frameworks and Deployment Options

### 14.1 Full-Stack Voice Agent Frameworks

**a) Vocode**
- Open-source framework for voice agents
- Supports multiple ASR/TTS providers
- Built-in telephony (Twilio, Vonage)
- Python-based, extensible pipeline

**b) LiveKit Agents**
- Real-time voice agent infrastructure
- WebRTC-native, built on LiveKit
- Integrates with OpenAI, Deepgram, ElevenLabs
- Handles transport, VAD, barge-in out of the box

**c) Daily Bots (by Daily.co)**
- Managed voice agent platform
- WebRTC-based with pre-built pipeline
- Supports custom ML models
- Low-latency infrastructure

**d) PlayAI**
- Voice agent SDK with pre-built voice
- Streaming TTS and ASR
- Emotion and style control
- Low-latency pipeline

**e) Rime.ai**
- Voice agent development platform
- Customizable voice pipelines
- Analytics and monitoring
- Enterprise-grade infrastructure

### 14.2 Component-Level Integration

| Component | Options |
|-----------|---------|
| ASR (Cloud) | Deepgram, AssemblyAI, Azure Speech, Google STT, AWS Transcribe |
| ASR (Self-Hosted) | Whisper, Wav2Vec, Paraformer, Vosk, Kaldi |
| TTS (Cloud) | ElevenLabs, Azure TTS, Google TTS, Amazon Polly, PlayHT |
| TTS (Self-Hosted) | Piper, Coqui, VITS, Tortoise TTS, Bark |
| NLP/LLM | OpenAI, Anthropic, Gemini, Llama, Mixtral |
| Transport | WebRTC, WebSocket, gRPC, SIP, RTP |
| VAD | Silero, WebRTC VAD, rnnoise |
| Telephony | Twilio, Vonage, Plivo, SignalWire |
| Monitoring | Prometheus, Grafana, Datadog, New Relic |

### 14.3 Deployment Architecture Options

**a) All-in-One Server**
```
Client ← WebRTC → Voice Agent Server (ASR + LLM + TTS)
```
- Pros: Simple, lowest latency
- Cons: Vertical scaling limits, single point of failure

**b) Microservice Pipeline**
```
Client ← WebRTC → Media Server → ASR Service → LLM Service → TTS Service → Client
```
- Pros: Independent scaling, fault isolation
- Cons: Higher latency (network hops), complex orchestration

**c) Hybrid Edge-Cloud**
```
Client ← WebRTC → Edge Node (VAD + ASR) → Cloud (LLM + TTS) → Client
```
- Pros: Low latency for ASR, powerful LLM in cloud
- Cons: Coordination complexity

---

## 15. References and Further Reading

- WebRTC Specification (W3C) — https://www.w3.org/TR/webrtc/
- Opus Codec — RFC 6716
- Silero VAD — https://github.com/snakers4/silero-vad
- Piper TTS — https://github.com/rhasspy/piper
- Faster Whisper — https://github.com/guillaumeklf/faster-whisper
- Vocode — https://github.com/vocodedev/vocode
- LiveKit Agents — https://github.com/livekit/agents
- Daily Bots — https://www.daily.co/products/bots/
- RNNoise — https://github.com/xiph/rnnoise
- Deepgram Streaming API — https://developers.deepgram.com/reference/streaming
- AssemblyAI Real-Time — https://www.assemblyai.com/docs/streaming
- Azure Speech SDK — https://learn.microsoft.com/en-us/azure/ai-services/speech-service/
- ElevenLabs Streaming — https://elevenlabs.io/docs/api-reference/streaming
- "Real-Time Voice Processing for AI Assistants" — Technical Report, 2025
- "Low-Latency Speech Enhancement for Real-World Applications" — ICASSP 2024
- "Efficient Streaming ASR with Transducer Models" — IEEE/ACM TASLP 2023
- WebRTC GCC (Google Congestion Control) — RFC 8888
- "Barge-In Detection for Voice Assistants: A Survey" — arXiv 2024
- Opus Interactive Audio Codec — https://opus-codec.org/
