# AI Audio & Music Generation — Speech Synthesis, Voice Cloning & Music AI

> June 2026

Audio AI has matured into a production-ready technology stack spanning speech synthesis, voice cloning, sound effects generation, and full music composition. This document covers the architectures, models, and applications.

---

## 1. The Audio AI Landscape

### 1.1 Market Snapshot

| Segment | Market Size (2026) | Key Players | Growth Rate |
|---------|-------------------|-------------|-------------|
| Text-to-Speech | $4.2B | ElevenLabs, Google, Amazon Polly, Azure | 18% YoY |
| Voice Cloning | $1.8B | ElevenLabs, Respeecher, PlayHT, Fish Audio | 35% YoY |
| Music Generation | $1.5B | Suno, Udio, Stability Audio, Google Lyria | 45% YoY |
| Sound Effects | $0.5B | ElevenSound, Meta AudioCraft, Stability SFX | 60% YoY |

### 1.2 Timeline of Breakthroughs

- **2022**: Coqui TTS open-source, VALL-E neural codec LM (Microsoft)
- **2023**: ElevenLabs v2, Suno Bark (early), Meta MusicGen + AudioGen
- **2024**: Suno V3 (first commercially viable music), ElevenLabs Reader, Stability Audio
- **2025**: Suno V4 (full song structure), ElevenSound, Udio, high-quality real-time TTS
- **2026**: Emotion-controllable speech, multi-speaker music generation, seamless voice conversion

---

## 2. Speech Synthesis (TTS) Architecture

### 2.1 Modern TTS Pipeline

```
Text → Text Encoder → Acoustic Model → Vocoder → Waveform
```

Modern TTS has evolved through three generations:

| Generation | Approach | Examples | Quality | Latency |
|------------|----------|----------|---------|---------|
| **1st Gen** | Concatenative (diphone/unit selection) | Ancient systems | Robotic | Low |
| **2nd Gen** | Parametric (HMM, statistical) | MaryTTS | Smooth but flat | Medium |
| **3rd Gen** | Neural (Tacotron, FastSpeech) | Tacotron 2, FastPitch | Natural | Medium |
| **4th Gen** | Neural Codec + LM | ElevenLabs, VALL-E, CosyVoice | Studio quality | Medium |
| **5th Gen** | End-to-end DiT + Codec | ElevenLabs v3+, Fish Speech | Indistinguishable | Low |

### 2.2 Neural Codec Language Models

The dominant architecture as of 2026:

```
Step 1: Encode audio → discrete tokens via neural codec
    ┌─────────────────────────────────────┐
    │ EnCodec / DAC / Mimi / SpeechTokenizer  │
    │ Input: 24kHz audio → Output: 8-32 tokens/frame  │
    │ Multiple residual RVQ levels (hierarchical)   │
    └─────────────────────────────────────┘

Step 2: Train autoregressive transformer on codec tokens
    ┌─────────────────────────────────────┐
    │ Audio Language Model (temporal)         │
    │ Conditioned on: text T5/ByT5 embeddings │
    │ Speaker embedding (for voice cloning)   │
    │ Emotion/Prosody conditioning            │
    └─────────────────────────────────────┘

Step 3: Decode predicted tokens back to waveform
    ┌─────────────────────────────────────┐
    │ Same neural codec decoder (reconstruct) │
    │ Output: 24kHz or 48kHz waveform         │
    └─────────────────────────────────────┘
```

**Key technologies**:
- **Residual Vector Quantization (RVQ)**: Multiple quantizer layers capturing coarse → fine detail
- **DAC (Descript Audio Codec)**: Open-source, high fidelity, 8kbps → 48kHz
- **Mimi (Mozilla)**: Real-time codec, 1.7kbps, streaming capable
- **Audio LM decoder**: Transformer with causal masking, trained on next-token prediction over codec indices

### 2.3 Voice Cloning

Voice cloning with limited data has become commercially viable:

| Data Requirement | Quality | Example |
|-----------------|---------|---------|
| 30 seconds | Good | ElevenLabs Instant |
| 3 minutes | Very good | PlayHT Instant |
| 30 minutes | Excellent | Respeecher |
| 3+ hours | Indistinguishable | Custom enterprise models |

**Architecture**:
```
Reference Audio → Speaker Encoder (d-vector / ECAPA-TDNN)
    → Speaker Embedding (128–512 dim) → Injected into Audio LM via cross-attention
```

---

## 3. Music Generation

### 3.1 Music Generation Approaches

| Approach | Description | Models |
|----------|-------------|--------|
| **Token-based LM** | Treat music as token sequence (like language) | Suno, Udio, MusicGen |
| **Diffusion on Spectrogram** | Generate mel-spectrograms then invert | Stability Audio, Riffusion |
| **Hybrid** | LM for structure + diffusion for timbre | Google Lyria, Suno V4 |
| **Stem-based** | Generate individual tracks (drums, bass, vocals) and mix | Suno V4 stems |
| **Score-based** | Generate symbolic notation then render | MuseNet, OmniMUSIC |

### 3.2 Suno V4 Architecture

Suno V4 represents the state of the art in full music generation:

```
Input: "A melancholic piano ballad in B minor, 90 BPM, about lost love"
    ↓
Lyric Generator (optional): LLM generates verse-chorus structure
    ↓
Structure Encoder: Maps genre/tempo/mood to structural template
    ↓
Music LM (Transformer): Generates codec tokens for full 3-minute song
    ├── Multi-track attention: Drums, Bass, Harmony, Melody, Vocals
    ├── Genre-conditional normalization
    └── Lyric-phoneme alignment (syllable-to-note timing)
    ↓
Stem Separator: Post-hoc separation into instrument stems
    ↓
Mastering Engine: EQ, compression, limiting (learned via RL from human feedback)
```

**Capabilities**:
- Full song generation (2–4 minutes)
- Genre, BPM, key, instrument control
- Lyric generation or custom lyrics
- Cover songs: "Make this song sound like a jazz trio"
- Stem export for DAW editing

### 3.3 Music Understanding Models

| Model | Task |
|-------|------|
| **CLAP** | Contrastive language-audio pretraining (text ↔ audio retrieval) |
| **MusicFM** | Music foundation model for genre, mood, instrument classification |
| **MERT** | Music understanding with acoustic + symbolic training |
| **Jukebox** | Lyrics-to-audio (prior generation) |
| **Encodec-Music** | Music-specific neural codec |

---

## 4. Sound Effects Generation

### 4.1 SFX Generation Models

| Model | Capabilities | Quality |
|-------|-------------|---------|
| **ElevenSound** | 1,000+ SFX categories, controllable duration | Studio |
| **Stability Audio SFX** | Open-source, fine-tunable | High |
| **Meta AudioGen** | Conditioned on text + reference audio | Good |
| **Soundstorm** | Efficient non-autoregressive sound generation | Good |

### 4.2 Prompt Engineering for SFX

```
# Good SFX prompts:
"The sound of footsteps on dry autumn leaves, close mic, stereo"
"A heavy wooden door creaking slowly, then slamming shut"
"Rain on a tin roof, medium intensity, with distant thunder"
"A futuristic laser gun charging up and firing, sci-fi quality"
```

---

## 5. Real-Time Speech & Voice Chat

### 5.1 Streaming Architecture

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  ASR      │────▶│  LLM     │────▶│  TTS     │────▶│  Output  │
│ Whisper   │     │ Response │     │ Voice    │     │ Speaker  │
└──────────┘     └──────────┘     └──────────┘     └──────────┘
     ▲                                                │
     │              Low-latency pipeline               │
     └─────────────────────────────────────────────────┘
                       < 200ms end-to-end
```

**Key technologies for real-time**:
- **Streaming ASR**: Whisper live, Deepgram Nova-2, AssemblyAI Streaming
- **Turn detection**: VAD (Voice Activity Detection) — Silero VAS, WebRTC VAD
- **Low-latency TTS**: ElevenLabs Turbo models (sub-100ms generation), Kokoro TTS
- **Streaming codec**: Mimi (Mozilla) for low-bitrate real-time transmission

### 5.2 Voice AI Products

| Product | Type | Features |
|---------|------|----------|
| **ElevenLabs Conversational AI** | Full-stack | Agent + voice + turn detection |
| **Pipecat** | Open-source framework | Modular, multi-provider |
| **Vapi AI** | Voice agent platform | Build-in-minutes, telephony |
| **Bland AI** | Enterprise voice agents | Scalable, analytics |
| **Retell AI** | Voice agent SDK | Customizable, WebSocket |

---

## 6. Evaluation Metrics

| Task | Metrics |
|------|---------|
| **TTS Quality** | MOS (Mean Opinion Score), CMOS, WER on TTS |
| **Voice Cloning** | Speaker similarity (d-vector cosine), naturalness MOS |
| **Music Quality** | Fréchet Audio Distance (FAD), CLAP score, human preference |
| **SFX Quality** | FAD, AudioSet accuracy, subjective fidelity |
| **Real-time** | RTF (Real-Time Factor), E2E latency (ms) |

---

## 7. Ethical Considerations

### 7.1 Voice Cloning Risks

- **Consent**: Opt-in voice cloning only (ElevenLabs, PlayHT require verification)
- **Voice scams**: Audio deepfakes used in CEO fraud (documented $25M+ cases)
- **Detection**: Speaker verification, audio watermarking (ElevenLabs Audio Watermark)

### 7.2 Music Copyright

- **Training data lawsuits**: RIAA suing Suno (2024–2025)
- **Style mimicry**: Is generating "a Taylor Swift song" copyright infringement?
- **Platform policies**: Spotify banning AI-generated music (partial reversal in 2025)

### 7.3 Best Practices

- Always obtain consent for voice cloning
- Label AI-generated audio clearly (C2PA provenance)
- Use watermarked outputs where possible
- Respect opt-out requests from artists and voice actors

---

*This document is part of the AI Knowledge Library — 28-AI-Video-Audio-Generation directory.*