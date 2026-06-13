# Speech-to-Text and Transcription: Modern ASR Technology

## 4.1 Introduction to Speech Recognition

Speech-to-text (STT), also known as Automatic Speech Recognition (ASR), is the technology that converts spoken language into written text. Modern STT systems leverage deep neural networks to achieve human-level accuracy on clean speech and dramatically improved robustness in noisy environments. This document provides comprehensive technical coverage of STT models, providers, architectures, and integration patterns.

### 4.1.1 ASR Technology Evolution

**Pre-2015 — Classic HMM/GMM:** Hidden Markov Models with Gaussian Mixture Models. Required hand-engineered acoustic features (MFCCs, filterbanks). Speaker-dependent training. Limited vocabulary. WER: 15-25% on clean speech.

**2015-2020 — Deep Learning Hybrid:** Deep Neural Networks replacing GMMs. CNN+RNN architectures. Connectionist Temporal Classification (CTC) loss. End-to-end models emerging. WER: 8-15% on clean speech. Examples: Deep Speech 2, Kaldi.

**2020-2024 — Transformer Era:** Transformer and Conformer architectures. Self-supervised pre-training (wav2vec 2.0, HuBERT, WavLM). Streaming attention mechanisms. WER: 2-8% on clean speech, approaching human parity. Examples: Whisper, Deepgram Nova, AssemblyAI Conformer.

**2024+ — Foundation Models:** Large-scale pre-trained models fine-tuned for specific domains. Multi-modal understanding. Real-time streaming optimized. WER: 1-5% on clean speech, 8-15% on challenging conditions. Examples: Whisper large-v3, Deepgram Nova-3, Chirp (Google).

### 4.1.2 ASR System Architecture

Modern ASR systems follow this high-level architecture:

```
┌──────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────┐
│  Audio   │   │  Feature     │   │  Acoustic    │   │  Decoder │
│  Input   │──▶│  Extraction  │──▶│  Model       │──▶│ + LM     │──▶ Text
│(16kHz PCM)│   │ (Mel-Spec / │   │ (Conformer / │   │          │
│          │   │  Encoder)    │   │  Streaming)  │   │          │
└──────────┘   └──────────────┘   └──────────────┘   └──────────┘
```

**Feature Extraction:** Raw audio (16kHz PCM, mono) is converted to mel-filterbank features (typically 80 mel bands). Frame size: 25ms, hop: 10ms. Some end-to-end models use raw waveform directly (SincNet frontend) or SSL features (HuBERT, WavLM).

**Acoustic Model:** The core neural network that maps acoustic features to phonetic representations. Modern architectures use:
- **Conformer:** Convolution-augmented Transformer. Self-attention captures long-range dependencies, convolution captures local patterns. The dominant architecture for high-quality ASR.
- **RNN-T (RNN Transducer):** Streaming-friendly architecture. Encoder processes audio, prediction network processes text, joint network combines them. Used by Deepgram Nova, Google USM.
- **CIF (Continuous Integrate-and-Fire):** Streaming architecture that outputs tokens at variable rate, aligned with speech rate.
- **Seq-to-Seq (Listen, Attend, Spell):** Attention-based encoder-decoder. Not streaming-friendly without modifications.

**Decoder + Language Model:** Converts acoustic model outputs to text. Can be:
- CTC decoder (greedy or beam search)
- RNN-T decoder (for streaming models)
- External language model (n-gram or neural LM) for rescoring
- Transducer joint network (for RNN-T models)

### 4.1.3 Streaming vs Batch Processing

**Batch Processing:** Process complete audio file at once. Higher accuracy (full context available). Higher latency (wait for entire file). Use cases: transcription, captioning, meeting recording.

**Real-Time Streaming:** Process audio as it arrives. Lower latency. Must manage endpointing (when is the utterance complete?). Can use:
- **Full Context:** Accumulate audio, process larger chunks. Higher latency, higher accuracy.
- **Greedy:** Process each chunk independently (no context). Lower latency, lower accuracy.
- **Context-Aware:** Maintain context window (previous chunks inform current chunk). Best balance.

```
# Streaming STT timing
# t=0: User starts speaking
# t=100ms: STT receives first audio chunk
# t=200ms: First partial result emitted
# t=800ms: User stops speaking
# t=1200ms: End-of-utterance detected (by VAD)
# t=1400ms: Final transcription result
# Total: ~1400ms from speech start to final text
```

## 4.2 OpenAI Whisper

Whisper is an open-source ASR model by OpenAI that has become a dominant force in the speech recognition landscape.

### 4.2.1 Model Architecture

Whisper uses a sequence-to-sequence transformer architecture:

```
Audio (80-channel Mel-Spectrogram)
    │
    ▼
┌──────────────┐
│  Encoder     │ 12-32 Transformer layers (model-dependent)
│  (Conformer) │
└──────────────┘
    │
    ▼
┌──────────────┐
│  Decoder     │ Cross-attention to encoder outputs
│  (Autoregr.) │ Predicts text tokens with special tokens
└──────────────┘
    │
    ▼
Text (with special tokens for task/format)
```

**Special Tokens:** Whisper uses special tokens to encode task and output format:
- `<|startoftranscript|>` + `<|en|>` (language) + `<|transcribe|>` (task) + text tokens + `<|endoftext|>`
- This allows a single model to handle: transcription, translation to English, language detection, and timestamped output

### 4.2.2 Whisper Model Variants

| Model | Params | VRAM (FP16) | Speed (relative) | WER LibriSpeech | WER CommonVoice |
|-------|--------|-------------|-------------------|-----------------|-----------------|
| tiny | 39M | ~1GB | 32x | 7.7% | 10.8% |
| base | 74M | ~1GB | 16x | 5.8% | 8.7% |
| small | 244M | ~2GB | 6x | 4.6% | 6.8% |
| medium | 769M | ~5GB | 2x | 3.8% | 5.8% |
| large | 1550M | ~10GB | 1x | 3.1% | 4.6% |
| large-v2 | 1550M | ~10GB | 1x | 2.9% | 4.3% |
| large-v3 | 1550M | ~10GB | 1x | 2.8% | 4.2% |
| large-v3-turbo | 809M | ~6GB | 3x | 3.3% | 4.8% |
| distil-large-v3 | 756M | ~5GB | 4x | 3.5% | 5.0% |

**large-v3** (released November 2024) is the current best Whisper model. Key improvements over v2:
- Trained on 5x more data (including multilingual YouTube)
- 99 languages supported
- Better accuracy on noisy speech and accented English
- Improved language identification

**large-v3-turbo:** A smaller, faster version. 809M parameters (vs 1550M). Uses distillation from large-v3. ~3x faster inference with ~10% relative WER increase.

### 4.2.3 Whisper Usage

**Basic Usage (OpenAI Whisper):**

```python
import whisper

model = whisper.load_model("large-v3-turbo")  # Or "large-v3", "medium", etc.

# Transcription
result = model.transcribe("audio.mp3")
print(result["text"])

# With options
result = model.transcribe(
    "audio.mp3",
    language="en",
    task="transcribe",  # or "translate" (to English)
    temperature=0.0,
    beam_size=5,
    word_timestamps=True,
    verbose=True,
)

# Word-level timestamps
for segment in result["segments"]:
    for word in segment["words"]:
        print(f"{word['word']}: {word['start']:.2f} - {word['end']:.2f}")
```

**Faster-Whisper (CTranslate2):**

```python
import faster_whisper

model = faster_whisper.WhisperModel(
    "large-v3",
    device="cuda",
    compute_type="float16",  # or "int8_float16" for 2x speedup
    num_workers=2,
)

# Streaming transcription
segments, info = model.transcribe(
    "audio.mp3",
    beam_size=5,
    vad_filter=True,  # Filter out silence
    vad_parameters=dict(min_silence_duration_ms=200),
)

print(f"Detected language: {info.language} ({info.language_probability:.2f})")

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

**WhisperX (with alignment and diarization):**

```python
import whisperx

# Load model
model = whisperx.load_model("large-v3-turbo", device="cuda", compute_type="float16")

# Transcribe
result = model.transcribe("audio.mp3")
print(f"Segments: {len(result['segments'])}")

# Align to improve word timestamps
align_model, metadata = whisperx.load_align_model(language_code="en", device="cuda")
result = whisperx.align(
    result["segments"],
    align_model,
    metadata,
    "audio.mp3",
    device="cuda",
)

# Speaker diarization
diarize_model = whisperx.DiarizationPipeline(
    model_name="pyannote/speaker-diarization-3.1",
    device="cuda",
)
diarize_segments = diarize_model("audio.mp3")

# Assign speakers to words
result = whisperx.assign_word_speakers(diarize_segments, result)
for seg in result["segments"]:
    print(f"Speaker {seg['speaker']}: {seg['text']}")
```

### 4.2.4 Optimizing Whisper Inference

```python
# Optimization strategies
import torch
import whisper

# 1. FP16 inference (2x faster on modern GPUs)
model = whisper.load_model("large-v3")
model = model.half()  # Convert to FP16

# 2. Torch compile (experimental)
model = torch.compile(model, mode="reduce-overhead")

# 3. Batch processing multiple audio files
audio1 = whisper.load_audio("file1.wav")
audio2 = whisper.load_audio("file2.wav")
padded = whisper.pad_or_trim(torch.stack([audio1, audio2]))
mel = whisper.log_mel_spectrogram(padded).to(model.device)
results = model.transcribe_batch(mel, batch_size=2)

# 4. Streaming with faster-whisper
from faster_whisper import BatchedInferencePipeline

model = faster_whisper.WhisperModel("large-v3", device="cuda")
batched_model = BatchedInferencePipeline(model=model)

segments, _ = batched_model.transcribe("audio.mp3", batch_size=16)
for segment in segments:
    print(segment.text)
```

### 4.2.5 Whisper Limitations

- **Latency:** Not designed for real-time streaming. Minimum ~1s for small utterances.
- **Hallucination:** Can produce text when there's no speech (silence, music, noise).
- **Language Detection:** Not always accurate, especially for short utterances.
- **Timestamp Accuracy:** Word-level timestamps can be off by 200-500ms without alignment.
- **Resource Usage:** Large models require significant GPU memory.
- **No Built-in Diarization:** Requires WhisperX or pyannote.audio for speaker labels.

## 4.3 Deepgram

Deepgram is a cloud-native ASR platform using end-to-end deep learning. It's optimized for real-time streaming with low latency.

### 4.3.1 Deepgram Model Architecture

Deepgram uses an **end-to-end deep learning architecture** that avoids the traditional pipeline. Their model is based on RNN-Transducer (RNN-T) with:

- **Speech Encoder:** Conformer architecture processing mel-filterbank features
- **Prediction Network:** LSTM-based network processing text context
- **Joint Network:** Combines encoder and prediction network outputs
- **Decoder:** RNN-T beam search decoder

Key design decisions:
- No separate language model (jointly trained end-to-end)
- Streaming-friendly by design (RNN-T supports incremental output)
- Multi-task learning: transcription + punctuation + capitalization + formatting

### 4.3.2 Deepgram Model Variants

| Model | Release | WER (LibriSpeech) | WER (internal) | Latency | Languages |
|-------|---------|-------------------|----------------|---------|-----------|
| Nova-1 | 2023 | 8.4%* | — | ~300ms | 30+ |
| Nova-2 | 2024 | 6.9%* | 8.4% | ~250ms | 30+ |
| Nova-3 | 2025 | 5.2%* | — | ~200ms | 30+ |
| Whisper distill | 2023 | — | — | ~150ms | 99 |

*Deepgram reports WER on their internal test sets, not standard benchmarks. The values above are approximate equivalents.

**Nova-3** improvements:
- 40% relative WER reduction over Nova-2
- Better accuracy on: accented speech, noisy environments, multiple speakers
- Improved custom vocabulary handling
- Faster time-to-first-transcript

### 4.3.3 Deepgram Streaming API

```python
import asyncio
import websockets
import json
import pyaudio

DEEPGRAM_API_KEY=os.get...CE = "wss://api.deepgram.com/v1/listen"

async def stream_microphone():
    """Stream microphone audio to Deepgram in real-time."""

    # Microphone setup
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=2048,
    )

    # WebSocket parameters
    params = {
        "model": "nova-2-general",
        "language": "en",
        "encoding": "linear16",
        "sample_rate": 16000,
        "channels": 1,
        "interim_results": True,       # Get partial results
        "punctuate": True,             # Auto punctuation
        "utterance_end_ms": 1000,      # End-of-utterance detection
        "vad_events": True,            # Get start/end speaking events
        "endpointing": 300,            # Silence threshold in ms
        "smart_format": True,          # Format numbers, dates, etc.
        "diarize": True,               # Speaker diarization
        "language_model": {            # Custom language model
            "boost": ["CustomWords"],
        },
    }

    extra_headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
    }

    # Connect to Deepgram
    async with websockets.connect(
        f"{ws_url}?{urlencode(params)}",
        extra_headers=extra_headers,
    ) as ws:
        print("Connected to Deepgram. Speak into your microphone.")

        # Send audio in a loop
        async def send_audio():
            while True:
                data = stream.read(2048, exception_on_overflow=False)
                await ws.send(data)

        # Receive transcripts
        async def receive_transcripts():
            async for message in ws:
                data = json.loads(message)

                if data.get("type") == "Results":
                    channel = data["channel"]
                    alternatives = channel["alternatives"]

                    if alternatives:
                        transcript = alternatives[0]["transcript"]
                        is_final = data.get("is_final", False)
                        confidence = alternatives[0].get("confidence", 0)

                        if transcript.strip():
                            if is_final:
                                print(f"✅ FINAL: {transcript} (conf: {confidence:.2f})")
                            else:
                                print(f"🔄 PARTIAL: {transcript}")

                elif data.get("type") == "UtteranceEnd":
                    print("— End of utterance —")

                elif data.get("type") == "Error":
                    print(f"❌ Error: {data['msg']}")

        await asyncio.gather(send_audio(), receive_transcripts())

asyncio.run(stream_microphone())
```

### 4.3.4 Deepgram Prerecorded API

```python
import requests
import json

# Prerecorded (batch) transcription
response = requests.post(
    "https://api.deepgram.com/v1/listen",
    headers={
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "application/json",
    },
    params={
        "model": "nova-2-general",
        "punctuate": True,
        "diarize": True,
        "smart_format": True,
        "paragraphs": True,   # Paragraph splitting
        "utterances": True,   # Utterance segmentation
        "summarize": "v2",    # Summarization (extra cost)
        "sentiment": True,    # Sentiment analysis (extra cost)
    },
    json={
        "url": "https://example.com/audio/meeting.mp3",
    },
)

result = response.json()
transcript = result["results"]["channels"][0]["alternatives"][0]["transcript"]
words = result["results"]["channels"][0]["alternatives"][0]["words"]

# Word-level details
for word in words[:10]:
    print(f"{word['word']}: start={word['start']:.2f}s end={word['end']:.2f}s "
          f"conf={word['confidence']:.2f} speaker={word.get('speaker', '?')}")
```

### 4.3.5 Deepgram Customization

```python
# Custom vocabulary to improve domain-specific accuracy
custom_vocab = {
    "model": "nova-2-general",
    "custom": {
        "boosted_words": [
            "acromegaly",
            "pseudopseudohypoparathyroidism",
            "CT scan",
            "MRI",
            "hemoglobin A1C",
            "pharmaceutical",
            "S&P 500",
            "NASDAQ",
            "Python 3.11",
        ],
        "boost_value": 15,  # Higher = stronger boost
    },
}

# Alternative: Train a custom model on domain data
# Requires uploading labeled audio or text data
# Deepgram Custom Model (DCM) can reduce domain-specific WER by 30-50%
```

## 4.4 AssemblyAI

AssemblyAI provides a cloud ASR platform with rich features and strong accuracy.

### 4.4.1 Key Features

- **Conformer-Transducer** architecture (similar to Deepgram)
- **Real-time streaming** with optional final transcript correction
- **Rich features:** sentiment analysis, topic detection, content moderation, chapter detection, entity detection, summarization
- **LeMUR:** LLM framework for applying GPT-based processing to transcripts

### 4.4.2 AssemblyAI API Usage

```python
import assemblyai as aai

aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

# Configure transcriber
config = aai.TranscriptionConfig(
    # Model selection
    speech_model=aai.SpeechModel.nova2,  # or best, nano
    language_code="en_us",

    # Formatting
    punctuate=True,
    format_text=True,

    # Diarization
    speaker_labels=True,
    speakers_expected=3,

    # Content features
    content_safety=True,
    content_safety_confidence=aai.ContentSafetyConfidenceThreshold.high,
    sentiment_analysis=True,
    entity_detection=True,
    auto_chapters=True,
    auto_highlights=True,
    summarization=True,
    summary_model=aai.SummarizationModel.informative,
    summary_type=aai.SummarizationType.bullets,

    # Custom vocabulary
    word_boost=["acromegaly", "NASDAQ"],
    boost_param=aai.BoostParam.low,
)

# Transcribe file
transcriber = aai.Transcriber()
transcript = transcriber.transcribe("audio.mp3", config)

# Access results
print(f"Text: {transcript.text}")
print(f"Confidence: {transcript.confidence}")

for utterance in transcript.utterances:
    print(f"Speaker {utterance.speaker}: {utterance.text}")

# Sentiment
for sentiment_result in transcript.sentiment_analysis:
    print(f"{sentiment_result.text} -> {sentiment_result.sentiment} "
          f"(conf: {sentiment_result.confidence})")

# Chapters
for chapter in transcript.chapters:
    print(f"Chapter: {chapter.headline} "
          f"({chapter.start:.1f}s - {chapter.end:.1f}s)")

# Streaming
stream_config = aai.StreamingConfig(
    speech_model=aai.SpeechModel.nova2,
    sample_rate=16000,
    encoding=aai.Encoding.pcm_mulaw,
    # or pcm_s16le for raw PCM
)

stream = aai.Streaming.from_microphone(stream_config)

@stream.on("open")
def on_open():
    print("Streaming...")

@stream.on("data")
def on_data(transcript: aai.Transcript):
    if transcript.text:
        if transcript.is_final:
            print(f"Final: {transcript.text}")
        else:
            print(f"Partial: {transcript.text}")

@stream.on("error")
def on_error(error):
    print(f"Error: {error}")

stream.connect()
```

## 4.5 Azure Speech

Azure Speech-to-Text is Microsoft's enterprise ASR offering with deep Azure ecosystem integration.

### 4.5.1 Key Features

- **Real-time and batch transcription**
- **Custom Speech:** Customize models with domain vocabulary and audio data
- **Language identification** for up to 10 languages simultaneously
- **Diarization** up to 10 speakers
- **Profanity filtering** with configurable sensitivity
- **Channel separation** for multi-channel call center audio
- **Pronunciation assessment**
- **On-premise deployment** via containers

### 4.5.2 Azure STT Implementation

```python
import azure.cognitiveservices.speech as speechsdk
import asyncio

speech_config = speechsdk.SpeechConfig(
    subscription=os.getenv("AZURE_SPEECH_KEY"),
    region=os.getenv("AZURE_SPEECH_REGION"),
)

# Set recognition language
speech_config.speech_recognition_language = "en-US"

# Enable diarization
speech_config.set_service_property(
    name="speakerDiarization",
    value="true",
    channel=speechsdk.ServicePropertyChannel.UriQueryParameter,
)

# Create audio configuration
audio_config = speechsdk.AudioConfig(
    use_default_microphone=True  # Or filename= for file
)

# Create recognizer with custom endpoint
speech_config.endpoint_id = os.getenv("AZURE_CUSTOM_ENDPOINT_ID")

recognizer = speechsdk.SpeechRecognizer(
    speech_config=speech_config,
    audio_config=audio_config,
)

# Event handlers
recognizer.recognizing.connect(
    lambda evt: print(f"PARTIAL: {evt.result.text}")
)

recognizer.recognized.connect(
    lambda evt: print(f"FINAL: {evt.result.text} (reason: {evt.result.reason})")
)

recognizer.canceled.connect(
    lambda evt: print(f"CANCELED: {evt.reason}")
)

# Continuous recognition
recognizer.start_continuous_recognition()
input("Press Enter to stop...")
recognizer.stop_continuous_recognition()
```

### 4.5.3 Custom Speech Model Training

```python
# Azure Custom Speech requires Azure portal workflow:
# 1. Upload text data (sentences with domain vocabulary)
# 2. Upload audio + labeled transcripts
# 3. Train custom model
# 4. Deploy to custom endpoint
# 5. Use endpoint_id in SpeechConfig (shown above)

# Benefits: Can reduce domain-specific WER by 20-60%
# Recommended: Minimum 1 hour of transcribed audio
# Best: 5-50 hours of domain-matched audio
```

## 4.6 STT Accuracy Benchmarks

### 4.6.1 Standard Benchmark Results (WER %)

| Model | LibriSpeech Clean | LibriSpeech Other | CommonVoice EN | CHiME-6 (noisy) | Telephony |
|-------|-------------------|-------------------|----------------|-----------------|-----------|
| Whisper large-v3 | 2.8% | 5.6% | 4.2% | 12.1% | 8.5% |
| Whisper large-v2 | 2.9% | 6.2% | 4.3% | 12.8% | 9.2% |
| distil-large-v3 | 3.5% | 7.2% | 5.0% | 14.5% | 10.5% |
| Deepgram Nova-3 | 4.5%* | 8.3%* | 5.1%* | 10.5%* | 6.2%* |
| Deepgram Nova-2 | 5.8%* | 10.1%* | 6.8%* | 13.2%* | 7.8%* |
| AssemblyAI | 5.1%* | 9.5%* | 6.3%* | 12.5%* | 7.5%* |
| Azure Speech | 6.8%* | 11.2%* | 7.5%* | 14.8%* | 8.2%* |
| Google Chirp | 3.8%* | 7.5%* | 5.2%* | 11.0%* | 7.0%* |

*Note: Values marked with * are approximate, based on published performance reports and independent evaluations. Conditions vary across evaluations.

### 4.6.2 Factors Affecting STT Accuracy

**Speaking Style:**
- Read speech: 1-3% WER (best case)
- Conversational: 5-10% WER
- Fast speech (>200 wpm): +2-5% WER
- Heavy accent: +3-15% WER
- Child voices: +5-20% WER

**Acoustic Conditions:**
- Clean, close-mic: Baseline
- Office noise: +3-8% WER
- Street noise: +10-20% WER
- Multi-speaker overlap: +15-30% WER
- Telephone quality (8kHz): +5-10% WER
- Music background: +15-40% WER

**Domain-Specific Vocabulary:**
- General English: Baseline
- Medical terminology: +10-25% WER (without customization)
- Legal/technical: +8-20% WER
- Product names: +15-40% WER (without customization)
- Code-switching: +15-35% WER

**Audio Quality:**
- 16kHz, clean: Baseline
- 8kHz (telephone): +5-10%
- Compressed (MP3 128kbps): +2-5%
- Volume too low/high: +5-15%
- Clipping/distortion: +15-30%

## 4.7 Cost Comparison

### 4.7.1 STT Pricing (per minute of audio)

| Provider | Real-Time | Batch | Custom Model | Notes |
|----------|-----------|-------|--------------|-------|
| Deepgram | $0.0043 | $0.0043 | $0.0065 | Volume discounts at 500K+ min/mo |
| AssemblyAI | $0.0050 | $0.0050 | Included | Rich features included |
| Azure Speech | $0.0060 | $0.0030 | $0.012-0.024 | Batch cheaper, custom premium |
| Google Cloud | $0.0060 | $0.0048 | $0.012 | First 60 min free monthly |
| Whisper (self-host) | ~$0.0004 | ~$0.0002 | — | GPU compute cost only |
| Whisper (API, Replicate) | $0.0060 | $0.0060 | — | Pay per second of audio |
| Rev AI | $0.0060 | $0.0045 | — | Human transcription $1.50/min |
| OpenAI Whisper API | — | $0.006/min | — | Supports 99 languages |

### 4.7.2 Cost Comparison Example

**Scenario:** 100,000 minutes/month of transcription

| Solution | Monthly Cost | Annual Cost |
|----------|-------------|-------------|
| Deepgram Nova-2 | $430 | $5,160 |
| AssemblyAI | $500 | $6,000 |
| Azure (batch) | $300 | $3,600 |
| Whisper (self-host, 4xA10 GPUs) | ~$1,200 | ~$14,400 |
| Whisper (self-host, 1xA100) | ~$400 | ~$4,800 |

Self-hosted Whisper costs depend on GPU prices (cloud GPU vs on-premise). At very high volumes (>500K min/mo), self-hosting Whisper becomes dramatically cheaper.

## 4.8 Selecting an STT Provider

### Decision Framework

**Choose Deepgram when:**
- Real-time streaming is primary use case
- Low latency is critical (<400ms)
- You need phone-optimized model
- You want good accuracy out of box
- Volume is moderate (< 1M min/mo)

**Choose Whisper when:**
- Cost is paramount and you have GPU infrastructure
- You need 99 languages
- Maximum accuracy is required (large-v3)
- Offline/batch processing with no latency constraints
- You need full control over model and data privacy

**Choose AssemblyAI when:**
- You need rich features (sentiment, chapters, entities)
- You want a single API for STT + NLP
- Diarization quality is important
- You want LeMUR for LLM-based analysis

**Choose Azure Speech when:**
- You're already in Azure ecosystem
- You need HIPAA/BAA compliance
- Custom model training is important
- On-premise deployment required
- You need multi-channel call center support

**Choose Google Cloud STT when:**
- You need maximum language coverage (125+)
- Chirp model quality meets requirements
- You're already in GCP ecosystem
- You need streaming V2 for word-level confidences

## 4.9 Advanced STT Techniques

### 4.9.1 Multi-Model Ensemble

Combine multiple STT models and pick the best result:

```python
class STTEnsemble:
    def __init__(self):
        self.models = {
            "whisper": WhisperTranscriber(),
            "deepgram": DeepgramTranscriber(),
            "assemblyai": AssemblyAITranscriber(),
        }

    async def transcribe(self, audio_path):
        results = {}
        for name, model in self.models.items():
            try:
                results[name] = await model.transcribe(audio_path)
            except Exception as e:
                print(f"{name} failed: {e}")

        # Score and select best result
        best = None
        best_score = 0
        for name, result in results.items():
            # Confidence-based scoring, or use a separate model
            score = result.get("confidence", 0) * len(result.get("text", ""))
            if score > best_score:
                best_score = score
                best = result

        return best
```

### 4.9.2 Voice Activity Detection Integration

```python
import webrtcvad
import collections
import pyaudio

class VADStreamingASR:
    def __init__(self, stt_service):
        self.vad = webrtcvad.Vad(2)  # Aggressiveness 0-3
        self.stt = stt_service
        self.audio_buffer = collections.deque()

    def process_chunk(self, audio_chunk):
        """Process 30ms audio chunk. Returns True if speech detected."""
        is_speech = self.vad.is_speech(audio_chunk, 16000)

        if is_speech:
            self.audio_buffer.append(audio_chunk)
        elif self.audio_buffer:
            # Process accumulated speech
            full_audio = b"".join(self.audio_buffer)
            self.audio_buffer.clear()
            return self.stt.transcribe(full_audio)

        return None

# Usage with Deepgram streaming
stt = DeepgramSTT(api_key=...)
vad_asr = VADStreamingASR(stt)

# In audio callback
def audio_callback(data):
    result = vad_asr.process_chunk(data)
    if result:
        print(f"Transcript: {result}")
```

### 4.9.3 Language Detection

```python
# Whisper multilingual detection
import whisper

model = whisper.load_model("large-v3")
audio = whisper.load_audio("multi_language.wav")

# Detect language
mel = whisper.log_mel_spectrogram(audio).unsqueeze(0)
_, probs = model.detect_language(mel)

# Top 5 languages
top_langs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
for lang, prob in top_langs:
    print(f"{lang}: {prob:.3f}")

# Deepgram language detection
# Set language="multi" in the API parameters
# Deepgram will auto-detect and transcribe in the detected language

# Azure language identification
auto_detect_source_language_config = \
    speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
        languages=["en-US", "fr-FR", "de-DE", "es-ES", "zh-CN"]
    )
recognizer = speechsdk.SpeechRecognizer(
    speech_config=speech_config,
    auto_detect_source_language_config=auto_detect_source_language_config,
    audio_config=audio_config,
)
```

## 4.10 Emerging Trends

**Audio Foundation Models:** Models like Google's AudioLM and Meta's Voicebox that can do speech recognition, speaker identification, and audio generation in a unified architecture.

**Direct Speech-to-Intent:** Skip the text transcription and directly classify user intent from audio features. Lower latency but less flexible.

**On-Device STT:** Running ASR on phones/edge devices for privacy and latency. Apple's on-device Siri models, Qualcomm's voice AI, Whisper on mobile via MediaPipe.

**Multi-Modal STT:** Combining audio with video (lip movements) for improved accuracy in noisy environments. Audio-Visual ASR.

**Real-time Translation:** STT with simultaneous translation to another language. Already available in Whisper (translate task) and Deepgram (translation API).

**Domain-Adaptive Models:** STT that automatically adapts to the domain (medical, legal, technical) without explicit fine-tuning, using in-context learning at inference time.

---

*This document covers speech-to-text technology comprehensively. See 02-Voice-Agent-Frameworks.md for STT integration in voice agent pipelines, 05-Voice-Biometrics-and-Speaker-ID.md for speaker-related technologies, and 06-Real-Time-Voice-Pipelines.md for streaming audio processing architecture.*
