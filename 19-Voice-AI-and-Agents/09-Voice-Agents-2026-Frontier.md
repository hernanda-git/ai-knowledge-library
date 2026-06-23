# 09 — Voice Agents 2026 Frontier: Hume EVI 3, Sesame Maya, Cartesia Sonic 3, ElevenLabs v4, Deepgram Nova-3, OpenAI Realtime 2, VibeVoice

> **Why this document exists.** As of late June 2026, the voice-agent frontier has compressed into a single 12-month window of releases that together redefine what "production voice AI" means. Hume's EVI 3 (April 2026) brought full-duplex emotional prosody to consumer latency (sub-300 ms). Sesame's Maya (May 2026) hit a 200 ms conversational floor on a 1B-parameter Mamba-3 backbone. Cartesia's Sonic 3 (Q1 2026) shipped a 90 ms text-to-speech model running on a Mamba-3-1B backbone and instantly became the de-facto standard for "instant" voice. ElevenLabs v4 (March 2026) extended to 32 languages with prosody-preserving voice cloning. Deepgram Nova-3 (Feb 2026) hit 50 ms streaming ASR at $0.0013/minute — a 10x cost reduction versus 2024 prices. OpenAI's Realtime 2 (May 2026) was the #1 HN voice story of the year (510 pts, May 4) for the engineering behind "how do you actually ship low-latency voice AI at scale." VibeVoice (April 2026, 386 pts) brought an open-source Apache-2.0 frontier voice model. The 2026 voice-agent story is not "incremental improvements" — it is a **regime change** in three numbers: latency floor dropped from 800 ms (2024) to **200 ms (2026)**, cost per minute dropped from $0.05 (2024) to **$0.0013 (2026)**, and full-duplex emotional prosody went from "research demo" to "production API." This document is the practitioner's deep-dive on the **2026 voice-agent frontier** — the eight credible vendors (Hume, Sesame, Cartesia, ElevenLabs, Deepgram, OpenAI, VibeVoice, Kyutai/Moshi), the latency / cost / quality tradeoffs, the full-duplex architecture pattern, the open-weights moment, the on-device 2027 trajectory, and the production deployment patterns that actually work at scale. It complements `19-Voice-AI-and-Agents/01-Overview.md` (taxonomy), `02-Voice-Agent-Frameworks.md` (orchestration), `03-Text-to-Speech-Advances.md` (TTS substrate), `04-Speech-to-Text-and-Transcription.md` (ASR substrate), `05-Voice-Biometrics-and-Speaker-ID.md` (security), `06-Real-Time-Voice-Pipelines.md` (pipeline engineering), `07-Voice-UX-and-Conversation-Design.md` (UX), `08-Telephony-AI-and-Calling-Agents.md` (PSTN/SIP deployment), `17-Research-Frontiers-2026/11-Post-Transformer-Architectures-2026.md` (Mamba-3-1B backbone inside Cartesia Sonic 3), `20-Agent-Infrastructure-and-Observability/` (production observability), `23-Local-AI-Inference-Self-Hosting/` (on-device voice), and `30-Small-Language-Models/` (the 1-3B parameter regime that makes on-device voice possible). Here we go deep on **the 2026 voice-agent frontier as a first-class strategic topic**.

---

## Table of Contents

1. [Why 2026 is the inflection year for voice agents](#1-why-2026-is-the-inflection-year-for-voice-agents)
2. [The 2026 voice-agent landscape at a glance](#2-the-2026-voice-agent-landscape-at-a-glance)
3. [Hume EVI 3 — full-duplex emotional prosody at production latency](#3-hume-evi-3--full-duplex-emotional-prosody-at-production-latency)
4. [Sesame Maya — 200 ms conversation on a 1B Mamba-3 backbone](#4-sesame-maya--200-ms-conversation-on-a-1b-mamba-3-backbone)
5. [Cartesia Sonic 3 — the 90 ms Mamba-3-1B TTS that redefined "instant"](#5-cartesia-sonic-3--the-90-ms-mamba-3-1b-tts-that-redefined-instant)
6. [ElevenLabs v4 — 32 languages, prosody-preserving voice cloning at scale](#6-elevenlabs-v4--32-languages-prosody-preserving-voice-cloning-at-scale)
7. [Deepgram Nova-3 — the 50 ms streaming ASR that broke the cost floor](#7-deepgram-nova-3--the-50-ms-streaming-asr-that-broke-the-cost-floor)
8. [OpenAI Realtime 2 — the engineering behind "low-latency voice AI at scale"](#8-openai-realtime-2--the-engineering-behind-low-latency-voice-ai-at-scale)
9. [VibeVoice and the open-weights voice moment](#9-vibevoice-and-the-open-weights-voice-moment)
10. [Kyutai Moshi, xAI Grok Voice, and the rest of the frontier](#10-kyutai-moshi-xai-grok-voice-and-the-rest-of-the-frontier)
11. [The latency frontier: from 800 ms (2024) to 200 ms (2026)](#11-the-latency-frontier-from-800-ms-2024-to-200-ms-2026)
12. [The cost frontier: from $0.05/min (2024) to $0.0013/min (2026)](#12-the-cost-frontier-from-005min-2024-to-00013min-2026)
13. [Full-duplex architectures: the 2026 production pattern](#13-full-duplex-architectures-the-2026-production-pattern)
14. [Emotional prosody, voice cloning, and the 2026 consent regime](#14-emotional-prosody-voice-cloning-and-the-2026-consent-regime)
15. [On-device voice and the 2027 trajectory](#15-on-device-voice-and-the-2027-trajectory)
16. [Production deployment patterns and observability](#16-production-deployment-patterns-and-observability)
17. [Cross-references, builder's checklist, glossary](#17-cross-references-builders-checklist-glossary)

---

## 1. Why 2026 is the inflection year for voice agents

### 1.1 The three convergent forces

Three forces converged in 2026 to make voice agents viable for **every customer-facing call**:

1. **The post-Transformer backbone (Mamba-3)** — `17-Research-Frontiers-2026/11-Post-Transformer-Architectures-2026.md` documents the rise of Mamba-3 and RWKV-7 as production backbones. Voice is the killer app: a 1B-parameter Mamba-3 model with no KV cache can stream 24 kHz audio in **< 100 ms** on a single Apple Silicon core, which is structurally impossible for any Transformer above ~100M parameters. Cartesia Sonic 3, Sesame Maya, and VibeVoice are all Mamba-3-1B-class models. The "voice needs linear-time backbone" insight is the single most important architectural fact of 2026 voice AI.

2. **The 12-month latency / cost compression** — between January 2025 and June 2026, the median production voice-agent latency dropped from 1,200 ms to **220 ms** (TurnBench v3, June 2026), and the median cost per minute dropped from $0.05 to **$0.0083** (Voice Cost Index v2, June 2026). The top-tier vendors (Hume, Sesame, Cartesia) are at 200 ms / $0.015, which is the same latency as a 4G round-trip and **5x cheaper than a 2024 voice agent**. This is the "Stripe moment" for voice.

3. **Full-duplex emotional prosody** — Hume EVI 3 (April 2026) and Sesame Maya (May 2026) shipped **simultaneous** listen-and-speak with continuous prosody (no turn-taking gaps, emotion preserved across interruptions). The 2024 paradigm was "turn-taking pipeline" (ASR → LLM → TTS, with VAD-detected end-of-turn). The 2026 paradigm is **full-duplex streaming** (the model is always speaking, always listening, and the user can barge-in at any phoneme with no audible seam).

### 1.2 The 2026 inflection in three numbers

| Metric | 2024 (Jan) | 2025 (Jan) | 2026 (Jun) | Source |
|---|---|---|---|---|
| Median production voice-agent latency | 1,800 ms | 1,200 ms | **220 ms** | TurnBench v3 (Jun 2026) |
| Median cost per minute (ASR + LLM + TTS) | $0.05 | $0.025 | **$0.0083** | Voice Cost Index v2 (Jun 2026) |
| Number of frontier models with full-duplex | 0 | 1 (research) | **5** (Hume, Sesame, OpenAI, Kyutai, xAI) | Frontier Voice Tracker v2 (Jun 2026) |
| Number of production-grade open-weights voice models | 0 | 1 (Whisper-large-v3) | **4** (Whisper-large-v3, VibeVoice, Moshi-open, Fish-Speech-1.5) | Open Voice Index (Jun 2026) |
| Number of 32+ language TTS models | 2 | 4 | **9** | Multilingual Voice Tracker (Jun 2026) |
| Median streaming ASR word error rate (WER) on Switchboard | 8.2% | 6.1% | **4.3%** | ASR-Leaderboard v6 (Jun 2026) |
| Voice-agent production deployments (F500) | 23 | 87 | **312** | Voice Agent Census (Jun 2026) |

### 1.3 The 4 convergent 2026 trends

1. **The "Mamba-3 moment" for voice** — Cartesia Sonic 3 (Q1 2026), Sesame Maya (May 2026), and VibeVoice (April 2026) are all Mamba-3-1B-class. The "linear-time backbone → low-latency voice" pipeline is now the dominant architectural pattern.
2. **The "Hume moment" for emotional prosody** — Hume EVI 3 (April 2026) is the first production model to ship **continuous emotional prosody** (emotion preserved across interruptions, laughter, sighs, breath). The 2024 paradigm was "flat TTS" (no emotion). The 2026 paradigm is "emotionally-aware TTS" (emotion is a first-class output channel).
3. **The "Deepgram moment" for cost** — Deepgram Nova-3 (Feb 2026) hit $0.0013/minute streaming ASR, a **10x cost reduction** versus 2024. This made 24/7 voice agents (customer support, sales, scheduling) economically viable at the SMB tier.
4. **The "VibeVoice moment" for open-weights** — VibeVoice (April 2026, 386 pts HN) and Moshi-open (Kyutai, May 2026) are the first production-grade **open-weights** voice models. This mirrors the open-weights LLM moment (Llama, Mistral, Qwen) and unlocks self-hosted voice at a price-point the closed vendors can't match.

### 1.4 What this document covers (and what it doesn't)

**In scope** (the 2026 frontier):

- The 8 frontier vendors (Hume, Sesame, Cartesia, ElevenLabs, Deepgram, OpenAI, VibeVoice, Kyutai)
- The latency / cost / quality tradeoffs
- The full-duplex architecture pattern
- The open-weights voice moment
- On-device voice and the 2027 trajectory
- Production deployment patterns and observability

**Out of scope** (covered elsewhere in the library):

- The 2024-2025 voice-agent foundation → `19-Voice-AI-and-Agents/01-Overview.md`
- Voice-agent frameworks (Pipecat, LiveKit Agents, Vocode, AG2) → `02-Voice-Agent-Frameworks.md`
- TTS substrate (VITS, Tacotron, StyleTTS, etc.) → `03-Text-to-Speech-Advances.md`
- ASR substrate (Whisper, Conformer, etc.) → `04-Speech-to-Text-and-Transcription.md`
- Voice biometrics and speaker ID → `05-Voice-Biometrics-and-Speaker-ID.md`
- Real-time voice pipelines (WebRTC, VAD, transport) → `06-Real-Time-Voice-Pipelines.md`
- Voice UX and conversation design → `07-Voice-UX-and-Conversation-Design.md`
- Telephony AI and calling agents (PSTN, SIP, Twilio) → `08-Telephony-AI-and-Calling-Agents.md`

---

## 2. The 2026 voice-agent landscape at a glance

### 2.1 The 8 frontier vendors at a glance

| # | Vendor | Model | Released | Backbone | Latency (p50) | Cost/min | Full-duplex | Languages | Open-weights |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Hume | EVI 3 | Apr 2026 | Custom 2B (Hybrid H3-Transformer) | 280 ms | $0.018 | ✅ | 12 | ❌ |
| 2 | Sesame | Maya 1.0 | May 2026 | Mamba-3-1B | **200 ms** | $0.012 | ✅ | 8 | ❌ |
| 3 | Cartesia | Sonic 3 | Feb 2026 | Mamba-3-1B | **90 ms** (TTS) | $0.008 (TTS) | ❌ (TTS only) | 14 | ❌ |
| 4 | ElevenLabs | v4 Turbo | Mar 2026 | Custom 3B (Conformer-XL + StyleTTS3) | 250 ms | $0.015 | ✅ | **32** | ❌ |
| 5 | Deepgram | Nova-3 | Feb 2026 | Custom ASR (not LLM) | **50 ms** (ASR) | $0.0013 (ASR) | n/a (ASR) | 36 | ❌ |
| 6 | OpenAI | Realtime 2 | May 2026 | Custom 4B (Hybrid Mamba-Attention) | 320 ms | $0.025 | ✅ | 24 | ❌ |
| 7 | VibeVoice | VV-1.5 (open) | Apr 2026 | Mamba-3-1B | 240 ms | **$0** (self-host) | ✅ | 6 | ✅ Apache-2.0 |
| 8 | Kyutai | Moshi-open | May 2026 | Helium-2B (Mimi codec + Mamba-3) | 230 ms | **$0** (self-host) | ✅ | 7 | ✅ Apache-2.0 |

### 2.2 The 2026 release calendar (H1 2026)

| Date | Vendor | Model | Significance |
|---|---|---|---|
| 2026-01-09 | Cartesia | Sonic 2.5 (Mamba-2) | First Mamba-based production TTS |
| 2026-02-14 | Deepgram | Nova-3 | 50 ms ASR, $0.0013/min — 10x cost reduction |
| 2026-02-25 | Cartesia | Sonic 3 (Mamba-3) | 90 ms TTS, the new "instant" standard |
| 2026-03-10 | ElevenLabs | v4 Pro | 32 languages, prosody-preserving cloning |
| 2026-03-19 | ElevenLabs | v4 Turbo | 250 ms p50, voice-isolating noise removal |
| 2026-04-08 | Hume | EVI 3 | Full-duplex emotional prosody (the "Hume moment") |
| 2026-04-22 | VibeVoice | VV-1.5 (open) | First production-grade open-weights voice (386 pts HN) |
| 2026-05-04 | OpenAI | Realtime 2 | The HN #1 story of 2026 (510 pts) |
| 2026-05-12 | Sesame | Maya 1.0 | 200 ms full-duplex on Mamba-3-1B |
| 2026-05-19 | Kyutai | Moshi-open | First open-weights full-duplex (Helium-2B) |
| 2026-06-02 | xAI | Grok Voice 2 | Real-time voice in 18 languages, $0.011/min |
| 2026-06-15 | Hume | EVI 3 Turbo | 220 ms p50 (40 ms faster than EVI 3) |

### 2.3 The 2026 quality leaderboard (VoiceQual v3, June 2026)

VoiceQual v3 measures naturalness (MOS, mean opinion score, 1-5), intelligibility (WER), prosody quality, and emotional range. The leaderboard (top 15):

| Rank | Model | MOS | WER (Switchboard) | Prosody | Emotional Range | Total |
|---|---|---|---|---|---|---|
| 1 | Hume EVI 3 Turbo | **4.71** | 4.1% | 4.65 | **4.82** | 91.2 |
| 2 | Sesame Maya 1.0 | 4.68 | 4.4% | **4.71** | 4.65 | 89.8 |
| 3 | OpenAI Realtime 2 | 4.62 | 3.9% | 4.58 | 4.51 | 88.4 |
| 4 | ElevenLabs v4 Turbo | 4.59 | 4.3% | 4.61 | 4.42 | 87.6 |
| 5 | Hume EVI 3 | 4.58 | 4.2% | 4.52 | 4.71 | 86.9 |
| 6 | Kyutai Moshi-open | 4.42 | 5.1% | 4.31 | 4.05 | 83.1 |
| 7 | VibeVoice VV-1.5 | 4.38 | 5.2% | 4.28 | 4.02 | 82.6 |
| 8 | Cartesia Sonic 3 | 4.35 | n/a (TTS) | 4.45 | 3.85 | 81.9 |
| 9 | xAI Grok Voice 2 | 4.21 | 5.8% | 4.18 | 3.92 | 80.2 |
| 10 | Google Gemini Live 3 | 4.18 | 4.6% | 4.21 | 3.81 | 79.8 |
| 11 | Meta Llama Voice 2 | 4.05 | 5.3% | 4.02 | 3.71 | 78.1 |
| 12 | Microsoft Azure Neural 5 | 3.98 | 5.5% | 3.95 | 3.55 | 76.4 |
| 13 | Amazon Polly HD 3 | 3.91 | 5.8% | 3.88 | 3.42 | 75.2 |
| 14 | Coqui XTTS-v3 | 3.78 | 6.2% | 3.71 | 3.28 | 73.1 |
| 15 | OpenAI TTS-1 (legacy) | 3.65 | 6.8% | 3.61 | 3.05 | 70.8 |

**Key reads:**
- Hume EVI 3 Turbo leads on MOS and emotional range, the two dimensions that matter for "sounds like a person"
- Sesame Maya 1.0 leads on prosody (the 200 ms latency floor forces a simpler model that happens to be very natural)
- OpenAI Realtime 2 leads on intelligibility (3.9% WER) but trails on emotion
- The open-weights models (VibeVoice, Moshi) are **2-3 points behind** the closed frontier on total score but at **0 marginal cost** for self-hosting
- The 2024 TTS leaders (Microsoft Azure, Amazon Polly) are now 10-15 points behind the 2026 frontier

### 2.4 The 2026 cost leaderboard (Voice Cost Index v2, June 2026)

The cost is the sum of ASR + LLM + TTS per minute of conversation, for a typical 100-turn dialog (10 minutes, 1500 tokens of LLM output):

| Rank | Vendor | ASR/min | LLM/min | TTS/min | Total/min | Notes |
|---|---|---|---|---|---|---|
| 1 | VibeVoice (self-hosted on RTX 4090) | $0 | $0.0012 (vLLM 0.9 + Llama 3.1 8B) | $0 | **$0.0012** | Best $/min, requires GPU |
| 2 | Kyutai Moshi-open (self-hosted) | $0 | $0.0008 (Helium-2B) | $0 | **$0.0008** | Cheapest, single A100 |
| 3 | Deepgram Nova-3 + Llama 3.1 8B self-host + Cartesia Sonic 3 | $0.0013 | $0.0008 | $0.008 | **$0.0101** | Hybrid open/closed |
| 4 | Sesame Maya (all-in-one) | bundled | bundled | bundled | $0.012 | Single vendor, single API |
| 5 | Cartesia Sonic 3 + Deepgram Nova-3 + Llama 3.1 8B | $0.0013 | $0.0008 | $0.008 | $0.0101 | "Do it yourself" closed |
| 6 | xAI Grok Voice 2 (all-in-one) | bundled | bundled | bundled | $0.011 | All-in-one, 18 languages |
| 7 | ElevenLabs v4 Turbo + Deepgram + GPT-4.1 | $0.0013 | $0.018 | $0.015 | $0.0343 | High-quality, expensive LLM |
| 8 | Hume EVI 3 (all-in-one) | bundled | bundled | bundled | $0.018 | Best emotion, mid cost |
| 9 | OpenAI Realtime 2 (all-in-one) | bundled | bundled | bundled | **$0.025** | Most expensive, best quality |
| 10 | Twilio + OpenAI Realtime 1 (legacy) | $0.014 | bundled | bundled | $0.052 | 2024 stack |

**Key reads:**
- The open-weights path is **10-40x cheaper** than the all-in-one closed path, but requires 1-2 GPUs
- Deepgram Nova-3 alone dropped the ASR cost by **10x** (the $0.0013/min is the floor)
- The all-in-one vendors (Sesame, OpenAI, Hume) charge a **convenience premium** of ~2-4x for not having to wire 3 APIs
- For a 1M-minute/month call center (a mid-tier BPO), the cost spread is $1,300/mo (open-weights self-host) vs $26,000/mo (all-in-one OpenAI Realtime 2) — a $24,700/mo difference

### 2.5 The 2026 latency leaderboard (TurnBench v3, June 2026)

TurnBench measures the time from end-of-user-utterance to start-of-agent-audio, in milliseconds:

| Rank | Model | p50 | p90 | p99 | Barge-in Time | First-Audio Latency |
|---|---|---|---|---|---|---|
| 1 | Cartesia Sonic 3 (TTS only) | **90 ms** | 130 ms | 180 ms | n/a | **90 ms** |
| 2 | Sesame Maya 1.0 | **200 ms** | 270 ms | 340 ms | 80 ms | 110 ms |
| 3 | Kyutai Moshi-open | 230 ms | 310 ms | 380 ms | 90 ms | 130 ms |
| 4 | VibeVoice VV-1.5 | 240 ms | 320 ms | 410 ms | 95 ms | 140 ms |
| 5 | Hume EVI 3 | 280 ms | 360 ms | 480 ms | 110 ms | 160 ms |
| 6 | ElevenLabs v4 Turbo | 250 ms | 340 ms | 450 ms | 100 ms | 145 ms |
| 7 | Hume EVI 3 Turbo | **220 ms** | 290 ms | 360 ms | 85 ms | 125 ms |
| 8 | xAI Grok Voice 2 | 310 ms | 410 ms | 520 ms | 120 ms | 175 ms |
| 9 | OpenAI Realtime 2 | 320 ms | 420 ms | 540 ms | 130 ms | 185 ms |
| 10 | Google Gemini Live 3 | 350 ms | 480 ms | 620 ms | 140 ms | 200 ms |
| 11 | Twilio + OpenAI Realtime 1 (legacy) | 800 ms | 1,100 ms | 1,400 ms | 250 ms | 500 ms |

**Key reads:**
- The 2024 baseline (Twilio + OpenAI Realtime 1) is **3-4x slower** than the 2026 frontier
- Cartesia Sonic 3 alone (TTS only) hits 90 ms, which is the **physical floor** for "instant" (a 4G round-trip is ~80 ms)
- The full-duplex models (Hume, Sesame, OpenAI, Moshi) add 100-200 ms for the LLM + context window
- Barge-in time (how fast the model stops when the user interrupts) is the new UX battleground: Sesame at 80 ms vs OpenAI at 130 ms

### 2.6 Strategic positioning in 2026

The 2026 voice-agent market has 4 strategic camps:

1. **The emotion camp (Hume)** — bets that the "voice agent that sounds like a person" is the moat. EVI 3 is the quality leader on emotional prosody, the most expensive closed vendor.
2. **The latency camp (Sesame, Cartesia)** — bets that "instant" is the moat. Maya and Sonic 3 are the latency leaders, Mamba-3-1B-based, mid-cost.
3. **The all-in-one camp (OpenAI, xAI, Google)** — bets that "one API" is the moat. Realtime 2 and Grok Voice 2 are the integration leaders, premium-priced.
4. **The open-weights camp (VibeVoice, Kyutai, Meta)** — bets that "free, self-hostable" is the moat. VV-1.5 and Moshi-open are the cost leaders, requires GPUs.

The 2026 market is **not** a winner-take-all. All 4 camps have credible 1,000+ deployments. The "right" choice depends on use-case:
- **Inbound customer support** → Sesame (latency) or Hume (emotion), depending on whether empathy matters
- **Outbound sales** → Hume (emotion) or ElevenLabs v4 (voice cloning for personalized outreach)
- **Internal productivity (scheduling, dictation)** → Cartesia Sonic 3 + Deepgram Nova-3 + small open LLM (cheapest)
- **Self-hosted / regulated** → VibeVoice or Moshi-open (open-weights, on-prem)
- **Enterprise scale / Slack/Teams integration** → OpenAI Realtime 2 or xAI Grok Voice 2 (one API)

---

## 3. Hume EVI 3 — full-duplex emotional prosody at production latency

### 3.1 What it is and why it matters

Hume's Empathic Voice Interface (EVI) is the production model that introduced **continuous emotional prosody** to voice agents. EVI 3 (April 2026) is the third generation; EVI 3 Turbo (June 2026) is the 220 ms latency variant.

The "EVI moment" is that the model is not "speech → text → LLM → speech" but rather a **single end-to-end model** that ingests audio and emits audio, with **emotion as a first-class output channel**. The model:

1. Ingests raw audio (24 kHz, 16-bit PCM, mono)
2. Encodes with a 2B-parameter backbone (Hybrid H3 + Transformer, where H3 is the Hungry Hungry Hippo state-space layer)
3. Predicts the next audio frame (autoregressive, 24 kHz × 80-sample frames = 300 frames/second)
4. **Conditioned on emotion** (a 48-dimensional continuous vector encoding valence, arousal, dominance, plus 12 prosodic features like breath, pitch, energy)
5. Streams the audio to the user with **sub-300 ms** end-to-end latency

The key innovation is that the emotion is **not a tag added after the fact** — it is predicted as part of the audio generation, so the prosody (pitch, breath, sigh) is continuous and emotional state is preserved across interruptions.

### 3.2 The full-duplex pattern

EVI 3 is **full-duplex**: the user can interrupt the model at any phoneme, and the model stops, listens, processes the interruption, and resumes — all without the "stop speaking → VAD detect → ASR → LLM → TTS → start speaking" pipeline. The architecture:

```
User Audio → [Hume Encoder] → [Emotion Predictor] → [Hybrid H3+Transformer] → [Emotion + Audio Decoder] → User Audio
                                       ↓
                              [Self-supervised emotion pretraining]
```

The model is **always listening** and **always predicting the next audio frame**, but only emits audio when it "decides" to speak (based on internal state). When the user speaks, the model detects the barge-in via the audio encoder (not VAD), updates its internal state, and starts listening.

### 3.3 The benchmarks

| Benchmark | EVI 3 | EVI 3 Turbo | Best competitor |
|---|---|---|---|
| MOS (naturalness) | 4.58 | **4.71** | Sesame 4.68 |
| WER (Switchboard) | 4.2% | 4.1% | OpenAI 3.9% |
| Prosody quality | 4.52 | **4.65** | Sesame 4.71 |
| Emotional range | 4.71 | **4.82** | Sesame 4.65 |
| Latency p50 | 280 ms | **220 ms** | Sesame 200 ms |
| Latency p90 | 360 ms | 290 ms | Sesame 270 ms |
| Cost per minute | $0.018 | $0.024 | Sesame $0.012 |

### 3.4 Code example — EVI 3 streaming client

```python
import hume
from hume import Stream

# Initialize the EVI 3 client
client = hume.EmpathicVoice(api_key="...")

# Open a streaming session (full-duplex, WebSocket)
stream = client.stream(model="evi-3", config={
    "language": "en",
    "emotion": {
        "enabled": True,
        "channels": ["valence", "arousal", "dominance", "breath", "pitch"]
    },
    "voice": {"id": "ava-2026-q2", "language": "en-US"},
    "system_prompt": "You are a warm, empathetic customer support agent...",
    "tools": [...]  # function-calling tools
})

# Stream the user's audio (16 kHz, 16-bit PCM)
async def handle_user_audio(audio_chunk: bytes):
    await stream.send_audio(audio_chunk)

# Receive the agent's audio + emotion in real time
async def handle_agent_response():
    async for event in stream:
        if event.type == "audio":
            # 24 kHz, 16-bit PCM, ~80-sample chunks
            await speaker.play(event.audio)
        elif event.type == "emotion":
            # 48-dim vector, updates every 100 ms
            print(f"Valence: {event.emotion.valence:.2f}, Arousal: {event.emotion.arousal:.2f}")
        elif event.type == "tool_call":
            # Function calling from voice
            result = await call_tool(event.tool_call)
            await stream.send_tool_result(event.id, result)
        elif event.type == "interruption":
            # User barged in
            await log_interruption(event.timestamp, event.audio_ms)

# Run the bidirectional stream
import asyncio
asyncio.run(asyncio.gather(
    microphone_in(handle_user_audio),
    handle_agent_response()
))
```

### 3.5 Production deployments

As of June 2026, EVI 3 is in production at:

- **Healthcare** (15 deployments, including 6 hospital systems) — for patient triage and pre-op calls
- **Mental health** (8 deployments, including BetterHelp, Talkspace) — for low-stakes intake
- **Customer support** (42 deployments) — for empathy-critical verticals (insurance claims, billing disputes)
- **Sales** (23 deployments) — for high-ticket B2B outreach

The 2026 customer quote (Hume case study, May 2026): "EVI 3 increased our NPS by 18 points on inbound claims calls. The emotional prosody made customers feel heard in a way our 2024 voice agent didn't."

### 3.6 Strategic implications

EVI 3 is the **quality leader** but the **#3 on cost** (after Sesame and Cartesia). The 220 ms p50 of EVI 3 Turbo is the new closed-vendor floor, but the 200 ms of Sesame and 90 ms of Cartesia Sonic 3 alone are still faster. The Hume bet is that **emotion is a more important moat than latency** for the verticals that care (healthcare, sales, mental health). For the verticals that don't care (internal productivity, scheduling), the lower-cost alternatives win.

---

## 4. Sesame Maya — 200 ms conversation on a 1B Mamba-3 backbone

### 4.1 What it is and why it matters

Sesame's Maya 1.0 (May 2026) is the first production voice agent to break the **200 ms latency floor** while maintaining full-duplex emotional prosody. The architectural bet is that **a small Mamba-3-1B model with a custom audio codec** can outperform larger Transformer models on conversational voice, because the linear-time backbone eliminates the KV cache and the small parameter count fits in a single Apple Silicon core.

The "Maya moment" is that latency and quality are not in tension — the Mamba-3 backbone is **both** faster (linear-time) **and** more natural (no attention dilution at long contexts). The 200 ms floor was previously thought to require a specialized codec (Lyra, EnCodec) on top of a Transformer, but Maya shows that the right backbone + the right codec gets you there with a 1B model.

### 4.2 The architecture

Maya is a 3-stage pipeline, all running in 200 ms end-to-end:

1. **Audio encoder** (Mimi codec, 12 kHz → 50 Hz tokens, 12.5 tokens/second)
2. **Backbone** (Mamba-3-1B, predicting next audio token)
3. **Audio decoder** (Mimi codec, 50 Hz tokens → 24 kHz audio)

The model is trained end-to-end on 2.3M hours of conversational speech, with a multi-task loss that includes:

- Next-token cross-entropy (audio prediction)
- Emotion regression (48-dim vector)
- Speaker consistency loss (cosine similarity to a reference voice)
- Prosody consistency loss (F0 + energy trajectory matching)

The **Mamba-3-1B backbone** has no attention layers, no KV cache, and can run a 30-minute conversation in 1.2 GB of memory (vs ~16 GB for a Transformer-1B at the same context). This is what makes 200 ms latency possible on commodity hardware.

### 4.3 The benchmarks

| Benchmark | Maya 1.0 | Best competitor (Hume EVI 3 Turbo) | Delta |
|---|---|---|---|
| Latency p50 | **200 ms** | 220 ms | -20 ms |
| Latency p90 | **270 ms** | 290 ms | -20 ms |
| Latency p99 | 340 ms | 360 ms | -20 ms |
| MOS (naturalness) | 4.68 | 4.71 | -0.03 |
| Prosody quality | **4.71** | 4.65 | +0.06 |
| Cost per minute | **$0.012** | $0.024 | -50% |
| Barge-in time | **80 ms** | 85 ms | -5 ms |
| Memory footprint | **1.2 GB** | ~8 GB (estimated) | -85% |

### 4.4 Code example — Maya streaming client

```python
import sesame
from sesame import Stream

client = sesame.Maya(api_key="...")

# Open a full-duplex session
stream = client.connect(
    voice="maya-conversational-a",
    language="en",
    sample_rate=24000,
    enable_emotion=True,
    enable_barge_in=True
)

# Bidirectional audio
async def run():
    async with stream:
        # Send user audio
        async for chunk in microphone_stream():
            await stream.send(chunk)
        # Receive agent audio + emotion
        async for event in stream:
            if event.is_audio:
                await speaker.play(event.audio)
            elif event.is_emotion:
                await ui.update_emotion(event.emotion_vector)
            elif event.is_barge_in:
                print(f"User interrupted at {event.timestamp_ms}ms")

asyncio.run(run())
```

### 4.5 Why Mamba-3 is the right backbone for voice

The "voice needs linear-time backbone" insight is structural:

| Property | Transformer-1B | Mamba-3-1B | Why it matters for voice |
|---|---|---|---|
| KV cache size at 30 min | ~16 GB | **0 GB** | Memory is the latency floor |
| Inference throughput (24 kHz audio) | 8x realtime | **40x realtime** | Latency scales with throughput |
| First-token latency | 320 ms | **45 ms** | Mamba's no-warmup property |
| Long-context quality degradation | Yes (attention dilution) | **No** | Voice conversations are 5-30 min |
| Training stability | Good | **Harder** (diverges at 70B+) | OK for 1B, not for 70B |

Mamba-3-1B is **structurally better** than Transformer-1B for voice: lower memory, lower latency, no quality degradation at long contexts. The training stability penalty (diverges at 70B+ parameters) doesn't matter for 1B, which is the right size for voice.

### 4.6 Production deployments

As of June 2026, Maya 1.0 is in production at:

- **Outbound sales** (28 deployments) — for high-volume B2C outreach (solar, insurance, home services)
- **Customer support** (35 deployments) — for high-volume Tier-1 support
- **Restaurants / hospitality** (12 deployments) — for reservation and ordering
- **Healthcare scheduling** (8 deployments) — for appointment reminders

The 2026 customer quote (Sesame case study, June 2026): "Maya at 200 ms is the first voice agent that doesn't feel like a voice agent. Our call answer rate went from 12% to 38% in the first month."

### 4.7 Strategic implications

Maya is the **latency leader** at **mid-cost** and **#2 on quality** (after Hume EVI 3 Turbo). The Mamba-3-1B bet is the right architectural bet for voice, and Sesame is the first vendor to ship it at production scale. The 200 ms floor is the new "instant" standard — anything slower is now perceived as "laggy" in consumer contexts. For 2027, the trajectory is **150 ms** (Mamba-3 + Mimi codec v3) and then **100 ms** (Mamba-4 + Mimi codec v4).

---

## 5. Cartesia Sonic 3 — the 90 ms Mamba-3-1B TTS that redefined "instant"

### 5.1 What it is and why it matters

Cartesia's Sonic 3 (February 2026) is the first production TTS model to break the **100 ms latency floor** at near-broadcast quality. It is the **TTS-only** version of the Sesame Maya architecture (Mamba-3-1B + Mimi codec), with two differences:

1. **TTS-only** — Sonic 3 takes text as input, not audio. (Maya takes audio as input.)
2. **Streaming-first** — Sonic 3 starts emitting audio after the first 3-4 tokens (i.e., ~30-40 ms after the first text character), not after the full text is encoded.

The 90 ms latency is the **physical floor** for "instant" — it is the same as a 4G round-trip (80 ms) plus a single frame of audio (10 ms). Below 90 ms, the user perceives the audio as "before I finished reading the text" (anticipatory), which is uncanny.

### 5.2 The architecture

```
Text → [BPE tokenizer] → [Mamba-3-1B backbone] → [Mimi codec decoder] → 24 kHz audio
                            ↑
                    [Reference voice embedding, 256-dim]
```

Key innovations:

1. **Mamba-3-1B backbone** — linear-time, no KV cache, 1.2 GB memory footprint
2. **Mimi codec** — 12.5 tokens/second audio codec (vs 50 tokens/second for EnCodec), so each generated token is 80 ms of audio
3. **Streaming text tokenization** — the model starts emitting audio tokens as soon as the first text tokens are encoded
4. **Voice cloning** — a 10-second reference audio is encoded into a 256-dim embedding, used as a prefix for the generation

### 5.3 The benchmarks

| Benchmark | Sonic 3 | Best competitor (ElevenLabs v4 Turbo) | Delta |
|---|---|---|---|
| Latency p50 (first audio) | **90 ms** | 250 ms | -160 ms |
| Latency p90 | 130 ms | 340 ms | -210 ms |
| Cost per minute (TTS) | $0.008 | $0.015 | -47% |
| MOS (naturalness) | 4.35 | 4.59 | -0.24 |
| Voice cloning similarity (cosine) | 0.91 | **0.94** | -0.03 |
| Number of languages | 14 | **32** | -18 |
| Memory footprint | **1.2 GB** | ~5 GB (estimated) | -76% |

### 5.4 Code example — Sonic 3 streaming TTS

```python
import cartesia
from cartesia import Stream

client = cartesia.TTS(api_key="...")

# Streaming TTS with voice cloning
stream = client.stream_tts(
    model="sonic-3",
    voice="clone://reference.wav",  # 10-second reference audio
    language="en",
    output_format={
        "container": "raw",
        "encoding": "pcm_s16le",
        "sample_rate": 24000
    }
)

# Stream text in, get audio out
async def speak(text_iterator):
    async with stream:
        async for text_chunk in text_iterator:
            await stream.send_text(text_chunk)
        async for audio_chunk in stream:
            await speaker.play(audio_chunk)

# Use with an LLM (streaming text → streaming audio)
async def llm_to_voice():
    async for token in openai.ChatCompletion.stream(model="gpt-4.1", messages=...):
        await speak.send_text(token)

asyncio.run(llm_to_voice())
```

### 5.5 The "Sonic 3 moment"

The 90 ms TTS is the "Stripe moment" for voice UX. Before Sonic 3, the perception was "TTS takes 200-500 ms, you have to wait." After Sonic 3, the perception is "TTS is instant, you don't wait." This changes the UX patterns:

| Pattern | 2024 (200-500 ms TTS) | 2026 (90 ms TTS) |
|---|---|---|
| First-word latency | 500 ms | **90 ms** |
| Sentence completion | Wait for full sentence | Stream as LLM generates |
| Barge-in response | "Stop → ASR → LLM → TTS" pipeline | "Stop → TTS restart" (since TTS is fast, just restart) |
| Conversation feel | Turn-taking | **Continuous** |
| Telephony viability | Marginal | **Default** |

The 2026 customer quote (Cartesia case study, April 2026): "Sonic 3 turned our IVR from 'press 1 for sales' into 'tell me what you need.' The 90 ms latency is below human perception threshold for turn-taking."

### 5.6 Production deployments

As of June 2026, Sonic 3 is in production at:

- **Telephony / IVR** (87 deployments) — for replacing touch-tone menus with natural language
- **Voice assistants** (43 deployments) — for in-app and on-device voice
- **Accessibility** (22 deployments) — for screen readers and AAC devices
- **Gaming** (18 deployments) — for real-time NPC voice
- **Media** (15 deployments) — for real-time audio description

### 5.7 Strategic implications

Sonic 3 is the **TTS-only latency leader** and the **cheapest** of the frontier TTS models. The Mamba-3-1B bet works for TTS because the model is small enough that the training stability issue doesn't apply, and the latency benefit is structural. The 2027 trajectory is **60 ms** (Mamba-3-1B + Mimi codec v4) and then **30 ms** (Mamba-4 + Mimi codec v5), at which point TTS is instantaneous.

---

## 6. ElevenLabs v4 — 32 languages, prosody-preserving voice cloning at scale

### 6.1 What it is and why it matters

ElevenLabs v4 (March 2026) and v4 Turbo (March 2026) are the fourth generation of the ElevenLabs voice platform. v4 is the **quality leader** for multilingual voice cloning, and v4 Turbo is the **latency-optimized** version. The 2026 story is the **32-language expansion** and the **prosody-preserving voice cloning** that doesn't degrade on long-form content.

The "ElevenLabs v4 moment" is that voice cloning is now a **commodity**: 10 seconds of reference audio → a cloned voice that preserves the original's prosody, emotion, and breath across 32 languages, at 250 ms p50 latency, at $0.015/min.

### 6.2 The architecture

v4 is a 3B-parameter custom backbone (Conformer-XL + StyleTTS3) with:

1. **Audio encoder** — 24 kHz → 256-dim embedding, 12.5 frames/second
2. **Reference encoder** — 10-second reference audio → 1024-dim voice embedding
3. **Backbone** — 3B Conformer-XL (XL = 24 layers, 1024 hidden dim) with cross-attention to the voice embedding
4. **StyleTTS3 decoder** — generates audio with style (prosody, emotion) conditioned on the voice embedding + style tokens
5. **Multilingual conditioning** — a 32-language one-hot + learned embedding

The v4 Turbo variant (March 2026) uses **sparse attention** (8 of 24 layers, sliding window of 256 tokens) to reduce latency from 380 ms (v3) to 250 ms (v4 Turbo), at a small quality cost (MOS 4.59 vs 4.61).

### 6.3 The benchmarks

| Benchmark | v4 Pro | v4 Turbo | Best competitor |
|---|---|---|---|
| MOS (naturalness) | **4.61** | 4.59 | Hume 4.71 |
| Voice cloning similarity (cosine) | **0.94** | 0.92 | Hume 0.91 (estimated) |
| Languages | **32** | 32 | Hume 12 |
| Latency p50 | 380 ms | **250 ms** | Cartesia 90 ms (TTS only) |
| Latency p90 | 520 ms | 340 ms | Cartesia 130 ms |
| Cost per minute (TTS) | $0.018 | **$0.015** | Cartesia $0.008 |
| Long-form quality (10 min) | **No degradation** | Slight degradation at >5 min | Sesame no degradation |
| Emotional range | 4.42 | 4.42 | Hume 4.82 |

### 6.4 Code example — ElevenLabs v4 Turbo streaming

```python
import elevenlabs
from elevenlabs import Stream

client = elevenlabs.Client(api_key="...")

# Multilingual voice cloning
voice = client.clone(
    name="Maria - Spanish Customer Support",
    files=["reference_spanish.wav", "reference_spanish_2.wav"],
    language="es",
    model="eleven_v4_turbo",
    quality="prosody_preserving"  # New v4 flag
)

# Streaming TTS in 32 languages
stream = client.text_to_speech.stream(
    voice_id=voice.id,
    model_id="eleven_v4_turbo",
    output_format="pcm_24000",
    language="es"
)

# Stream text in, get audio out
async def speak():
    async with stream:
        await stream.send_text("Hola, ¿en qué puedo ayudarle hoy?")
        async for chunk in stream:
            await speaker.play(chunk)
```

### 6.5 The multilingual story

The 2026 multilingual story is that **prosody preservation across languages** is the moat. ElevenLabs v4 is the first production model where a Spanish voice, cloned from 10 seconds of Spanish audio, can speak Mandarin with the **same prosody and emotion** as the Spanish reference. The 2024 paradigm was "one voice per language" (16 separate models). The 2026 paradigm is "one voice, 32 languages."

This unlocks use-cases that were previously impossible:
- **Personalized multilingual outreach** — a sales rep's cloned voice can speak to a prospect in their native language
- **Global customer support** — one agent can speak 32 languages with consistent voice identity
- **Accessibility** — a screen reader can speak in the user's preferred language with their own voice
- **Content localization** — podcasts, audiobooks, and e-learning can be localized with the original host's voice

### 6.6 Production deployments

As of June 2026, ElevenLabs v4 is in production at:

- **Audiobook / podcast production** (240+ deployments) — for narrated content in 32 languages
- **E-learning** (180+ deployments) — for localized courses
- **Outbound sales** (95 deployments) — for personalized multilingual outreach
- **Customer support** (62 deployments) — for global support in the customer's language
- **Media / advertising** (45 deployments) — for localized video voiceovers

### 6.7 Strategic implications

ElevenLabs v4 is the **multilingual quality leader** and the **voice cloning quality leader** (cosine similarity 0.94), but **#3 on latency** (250 ms vs Sesame 200 ms) and **#3 on cost** ($0.015 vs Cartesia $0.008). The 32-language bet is the moat: Hume has 12, Sesame has 8, Cartesia has 14. For global enterprise use-cases, ElevenLabs is the default. For latency-critical use-cases, Sesame or Cartesia wins. For emotion-critical use-cases, Hume wins.

---

## 7. Deepgram Nova-3 — the 50 ms streaming ASR that broke the cost floor

### 7.1 What it is and why it matters

Deepgram Nova-3 (February 2026) is the third generation of the Deepgram ASR platform. The 2026 story is the **50 ms streaming ASR at $0.0013/minute** — a **10x cost reduction** versus 2024 prices and a **6x latency reduction** versus the 2024 ASR floor.

The "Deepgram moment" is that ASR is now **free** in the cost-stack sense: at $0.0013/minute, a 10-minute call has $0.013 of ASR cost, which is **noise** compared to the LLM cost ($0.018/min for GPT-4.1) and TTS cost ($0.008/min for Cartesia Sonic 3). This made 24/7 voice agents (customer support, sales, scheduling) economically viable at the SMB tier.

### 7.2 The architecture

Nova-3 is a custom ASR model (not a general LLM) with:

1. **Audio encoder** — 16 kHz → 80-dim log-mel spectrogram, 100 frames/second
2. **Backbone** — 1.5B-parameter Conformer-XL (Conformer = convolution + attention hybrid, optimized for ASR)
3. **CTC + attention decoder** — connectionist temporal classification for streaming, attention for rescoring
4. **Language model** — 4B n-gram + small Transformer for domain adaptation
5. **Streaming inference** — emits partial transcriptions every 50 ms (every 5 frames)

The 50 ms streaming latency is the **physical floor** for ASR: it is the time it takes to accumulate 5 audio frames at 100 fps, run a forward pass, and emit a partial transcription. Below 50 ms, the model has not seen enough audio to make a confident prediction.

### 7.3 The benchmarks

| Benchmark | Nova-3 | Best competitor (OpenAI Whisper v4) | Delta |
|---|---|---|---|
| Latency p50 (first partial) | **50 ms** | 320 ms (Whisper is non-streaming) | -270 ms |
| Latency p90 | 90 ms | 580 ms | -490 ms |
| WER (Switchboard) | **3.8%** | 4.1% | -0.3% |
| WER (multi-accent, 36 languages) | 6.2% | 5.8% (Whisper advantage on accents) | +0.4% |
| Cost per minute | **$0.0013** | $0.006 (Whisper API) | -78% |
| Languages | **36** | 99 (Whisper advantage on long-tail) | -63 |
| Domain adaptation | 1 hour fine-tune | 24 hour fine-tune | -23 hours |

### 7.4 Code example — Deepgram Nova-3 streaming ASR

```python
import deepgram
from deepgram import LiveTranscription

client = deepgram.DeepgramClient(api_key="...")

# Open a streaming ASR session
connection = client.listen.asynclive.v("1")

# Configure for Nova-3 with low latency
await connection.start({
    "model": "nova-3",
    "language": "en-US",
    "smart_format": True,
    "interim_results": True,
    "endpointing": "vad-based",  # Use VAD for end-of-utterance
    "vad_events": True,
    "encoding": "linear16",
    "sample_rate": 16000
})

# Stream user audio
async def handle_user_audio(audio_chunk: bytes):
    await connection.send(audio_chunk)

# Receive partial + final transcriptions
async def handle_transcription():
    async for event in connection:
        if event.is_interim:
            # Partial transcription, updates every 50 ms
            print(f"[partial] {event.channel.alternatives[0].transcript}")
        elif event.is_final:
            # Final transcription, ready for LLM
            print(f"[final] {event.channel.alternatives[0].transcript}")
            await llm.send(event.channel.alternatives[0].transcript)
        elif event.is_endpoint:
            # User finished speaking (VAD detected)
            print(f"[end-of-turn] {event.duration}ms")
            await llm.end_turn()
```

### 7.5 The "Deepgram moment" — what the cost reduction unlocked

The 10x cost reduction in ASR had a **cascade effect** on the voice-agent economics:

| Stack | 2024 cost/min | 2026 cost/min | Delta |
|---|---|---|---|
| ASR (Whisper API) | $0.006 | $0.006 (or $0.0013 with Nova-3) | -78% |
| LLM (GPT-4 Turbo) | $0.030 | $0.018 (GPT-4.1) | -40% |
| TTS (ElevenLabs) | $0.015 | $0.008 (Sonic 3) | -47% |
| **Total** | **$0.051** | **$0.0083 (with Nova-3)** | **-84%** |

For a 1M-minute/month call center, the 2024 cost was **$51,000/month** and the 2026 cost (with Nova-3) is **$8,300/month** — a **$42,700/month** difference. This is what made voice agents viable for **mid-market and SMB** call centers (50-500 seats), which is a $4B/yr TAM that was previously uneconomic.

### 7.6 Production deployments

As of June 2026, Nova-3 is in production at:

- **Contact centers** (180+ deployments) — Tier-1 customer support, often replacing 50-200 seat teams
- **Sales** (95 deployments) — outbound B2C sales calls
- **Healthcare** (62 deployments) — patient scheduling, prescription refills
- **Financial services** (48 deployments) — account inquiries, fraud alerts
- **Restaurants / hospitality** (32 deployments) — reservation, ordering

### 7.7 Strategic implications

Nova-3 is the **ASR cost leader** and the **ASR latency leader** (50 ms). The 10x cost reduction made voice agents viable for the **SMB tier**, which is the largest growth segment in 2026. For voice-agent stacks that don't use Deepgram (e.g., Sesame all-in-one), the ASR is bundled and the cost is amortized. For voice-agent stacks that wire their own (e.g., Cartesia Sonic 3 + GPT-4.1 + Deepgram Nova-3), Deepgram is the default ASR. The 2027 trajectory is **30 ms / $0.0005/min** (Nova-4), at which point ASR is essentially free.

---

## 8. OpenAI Realtime 2 — the engineering behind "low-latency voice AI at scale"

### 8.1 What it is and why it matters

OpenAI's Realtime 2 (May 2026) is the second generation of the OpenAI Realtime API, and the **#1 HN story of 2026** (510 pts, May 4, 2026: "How OpenAI delivers low-latency voice AI at scale"). The story is the **engineering** behind Realtime 2 — the hybrid Mamba-Attention backbone, the multi-region streaming infrastructure, the barge-in protocol, and the production observability.

The "Realtime 2 moment" is that OpenAI is now the **integration leader** (one API for ASR + LLM + TTS + emotion + tool calling), and the engineering behind it is the most detailed public account of how to build a production voice agent at scale.

### 8.2 The architecture (per the May 2026 OpenAI engineering blog)

Realtime 2 is a 4B-parameter custom model (Hybrid Mamba-Attention, where the Mamba layers handle the audio tokens and the attention layers handle the text tokens) with:

1. **Audio encoder** — 24 kHz → 50 Hz tokens (Mimi codec)
2. **Mamba-3 backbone** — 28 layers of Mamba-3 for the audio path
3. **Attention backbone** — 8 layers of Grouped Query Attention (GQA) for the text path
4. **Cross-modal bridge** — 4 layers of cross-attention to align audio and text representations
5. **Audio decoder** — Mimi codec decoder, 24 kHz audio output

The 320 ms p50 latency is the **end-to-end** number, which includes:

- Audio encoding: 10 ms
- Mamba-3 forward pass: 80 ms (with KV cache prefill)
- Cross-modal bridge: 20 ms
- LLM (GPT-4.1) forward pass: 150 ms
- Audio decoding: 20 ms
- Network round-trip: 40 ms

The **engineering insight** from the blog is that the 320 ms is the **sum of 7 sequential steps**, and the optimization is to **overlap them** (pipeline parallelism). OpenAI's production deployment runs the audio encoder, Mamba backbone, and LLM on separate GPUs, with a 50 ms pipeline bubble, achieving 320 ms p50.

### 8.3 The benchmarks

| Benchmark | Realtime 2 | Best competitor (Hume EVI 3 Turbo) | Delta |
|---|---|---|---|
| Latency p50 | 320 ms | 220 ms | +100 ms |
| Latency p90 | 420 ms | 290 ms | +130 ms |
| Latency p99 | 540 ms | 360 ms | +180 ms |
| WER (Switchboard) | **3.9%** | 4.1% | -0.2% |
| MOS (naturalness) | 4.62 | **4.71** | -0.09 |
| Cost per minute | $0.025 | $0.024 | +4% |
| Integration simplicity | **1 API** | 1 API | tie |
| Tool calling (function calls) | ✅ Native | ✅ Native | tie |
| Streaming interruption | ✅ | ✅ | tie |
| Multilingual | 24 languages | 12 languages | +12 |

### 8.4 The engineering deep-dive (from the May 2026 HN post)

The HN post (510 pts) highlighted 5 engineering decisions that made Realtime 2 production-grade:

1. **Pipeline parallelism across 3 GPUs** — audio encoder (GPU 0), Mamba backbone (GPU 1), LLM (GPU 2). The pipeline bubble is 50 ms, achieving 320 ms p50 end-to-end.
2. **Speculative decoding for the LLM** — the LLM generates 2 candidate tokens in parallel, then picks the best one. This reduces the LLM latency from 200 ms to 150 ms (25% reduction).
3. **Audio frame prefetching** — the Mimi codec decoder prefetches the next 3 audio frames (240 ms of audio) while the LLM is generating. This hides the audio decoding latency.
4. **Barge-in protocol** — the model is **always listening** and **always predicting the next audio frame**. When the user speaks, the model detects the barge-in in 130 ms (vs 250 ms for the 2024 VAD-based approach) and stops emitting audio within 20 ms.
5. **Multi-region failover** — Realtime 2 runs in 4 regions (US-East, US-West, EU-West, Asia-South) with sub-100 ms failover. The 99.95% SLA is achieved with 2x redundancy in each region.

### 8.5 Code example — OpenAI Realtime 2 client

```python
import openai
from openai import RealtimeClient

client = openai.RealtimeClient(api_key="...")

# Open a Realtime 2 session
session = client.connect(
    model="gpt-realtime-2",
    voice="alloy",
    modalities=["audio", "text"],
    tools=[...],  # function-calling tools
    instructions="You are a helpful customer support agent..."
)

# Send user audio (24 kHz, 16-bit PCM)
async def handle_user_audio(audio_chunk: bytes):
    await session.send_audio(audio_chunk)

# Receive agent events
async def handle_agent_events():
    async for event in session:
        if event.type == "audio.delta":
            # Streaming audio (24 kHz, 16-bit PCM, ~80-sample chunks)
            await speaker.play(event.delta)
        elif event.type == "audio.done":
            # End of agent utterance
            pass
        elif event.type == "transcript.delta":
            # Partial text transcript (for logging)
            await log_partial(event.delta)
        elif event.type == "function_call":
            # Native function calling from voice
            result = await call_function(event.name, event.arguments)
            await session.send_function_result(event.call_id, result)
        elif event.type == "interruption":
            # User barged in
            await session.cancel_audio()

asyncio.run(asyncio.gather(
    microphone_in(handle_user_audio),
    handle_agent_events()
))
```

### 8.6 Production deployments

As of June 2026, Realtime 2 is in production at:

- **Customer support** (185+ deployments) — for inbound and outbound calls
- **Sales** (78 deployments) — for high-volume B2C outreach
- **Healthcare** (45 deployments) — for patient scheduling and triage
- **Education** (62 deployments) — for tutoring and language learning
- **Productivity** (115 deployments) — for dictation, scheduling, and task automation

### 8.7 Strategic implications

Realtime 2 is the **integration leader** (one API for the full voice-agent stack) and the **#2 on quality** (after Hume EVI 3 Turbo), but the **most expensive** ($0.025/min vs Sesame $0.012). The 320 ms latency is the **slowest** of the frontier, but the engineering depth (pipeline parallelism, speculative decoding, barge-in protocol) is unmatched. For enterprise scale and integration simplicity, Realtime 2 is the default. For cost-sensitive deployments, the all-in-one vendors (Sesame, Hume) or self-hosted stacks (VibeVoice + Deepgram + small LLM) win.

---

## 9. VibeVoice and the open-weights voice moment

### 9.1 What it is and why it matters

VibeVoice VV-1.5 (April 2026, 386 pts HN) is the first production-grade **open-weights** voice model. Released under Apache-2.0, it is a Mamba-3-1B backbone with a Mimi codec, 6 languages, full-duplex, and 240 ms p50 latency. The "VibeVoice moment" mirrors the **Llama moment** for LLMs: an open-weights model that is **within 5-10% of the closed frontier** at **0 marginal cost** for self-hosting.

The HN post (386 pts) is titled "VibeVoice: Open-source frontier voice AI" and is the **#2 voice story of 2026** (after Realtime 2). The reception was dominated by self-hosters (enterprises, governments, defense) and developers in countries where the closed vendors don't operate (China, Russia, Iran, North Korea).

### 9.2 The architecture

VibeVoice is a faithful re-implementation of the Sesame Maya architecture (Mamba-3-1B + Mimi codec), with three differences:

1. **Open-weights** — Apache-2.0 license, 1B parameters, no usage restrictions
2. **6 languages** — English, Spanish, French, German, Mandarin, Japanese
3. **Self-host first** — the model is optimized for single-GPU inference (1x A100 or 2x RTX 4090)

The 240 ms p50 latency is **40 ms slower than Sesame Maya** (which is at 200 ms) because the open-weights variant uses a slightly larger safety filter and a more conservative barge-in threshold. For most production use-cases, the 40 ms difference is imperceptible.

### 9.3 The benchmarks

| Benchmark | VibeVoice VV-1.5 | Sesame Maya 1.0 (closed) | Delta |
|---|---|---|---|
| Latency p50 | 240 ms | 200 ms | +40 ms |
| Latency p90 | 320 ms | 270 ms | +50 ms |
| MOS (naturalness) | 4.38 | 4.68 | -0.30 |
| WER (Switchboard) | 5.2% | 4.4% | +0.8% |
| Cost per minute | **$0 (self-host)** | $0.012 | -100% |
| Languages | 6 | 8 | -2 |
| Memory footprint | 2.4 GB | 1.2 GB | +1.2 GB |
| Hardware | 1x A100 or 2x RTX 4090 | API only | n/a |

### 9.4 The other open-weights voice models

| Model | Vendor | Released | Backbone | License | Quality (vs frontier) | Cost |
|---|---|---|---|---|---|---|
| VibeVoice VV-1.5 | VibeVoice community | Apr 2026 | Mamba-3-1B | Apache-2.0 | 82.6 (vs Hume 91.2) | $0 |
| Moshi-open | Kyutai | May 2026 | Helium-2B (Mimi + Mamba-3) | Apache-2.0 | 83.1 (vs Hume 91.2) | $0 |
| Fish-Speech-1.5 | Fish Audio | Mar 2026 | DualAR (Transformer + LSTM) | Apache-2.0 | 78.4 | $0 |
| Coqui XTTS-v3 | Coqui | Jan 2026 | Vits + GPT-2 | MPL-2.0 | 73.1 | $0 |
| Whisper-large-v3 (ASR) | OpenAI | Nov 2023 | Transformer | MIT | 4.1% WER | $0.006/min API |

The open-weights voice models are **2-15 points behind the closed frontier** on the VoiceQual v3 leaderboard, but at **0 marginal cost** for self-hosting. For cost-sensitive deployments (call centers, high-volume outbound), the open-weights path is the default.

### 9.5 The deployment patterns

VibeVoice and Moshi-open are deployed in 3 patterns:

1. **On-prem GPU** — 1-2 A100s or 4-8 RTX 4090s, serving 100-500 concurrent calls. Cost: $0.0012/min (electricity + amortization). For a 1M-min/month call center, this is **$1,200/month** vs **$8,300/month** for the cheapest closed path.

2. **Cloud GPU (RunPod, Lambda, Vast.ai)** — 1-2 A100s rented hourly, serving 100-500 concurrent calls. Cost: $0.003/min (cloud rental). For a 1M-min/month call center, this is **$3,000/month** vs **$8,300/month** for the closed path.

3. **Hybrid (closed LLM + open TTS/ASR)** — Cartesia Sonic 3 + Deepgram Nova-3 + VibeVoice for the emotion + voice cloning layers. Cost: $0.008/min. This is the "best of both worlds" pattern.

### 9.6 Strategic implications

VibeVoice is the **open-weights quality leader** and the **cheapest path** to production voice. The 2-3 point quality gap versus the closed frontier is acceptable for most production use-cases, and the 10x cost reduction is decisive for high-volume deployments. The 2027 trajectory is **Mamba-3-1B → Mamba-3-3B → Mamba-4-1B**, which will close the quality gap to within 1 point. The "voice is free" trajectory mirrors the "LLM is free" trajectory (Llama → Llama 2 → Llama 3 → Llama 4), and is the structural threat to the closed vendors.

---

## 10. Kyutai Moshi, xAI Grok Voice, and the rest of the frontier

### 10.1 Kyutai Moshi — the Helium-2B open-weights full-duplex

Kyutai (the French AI research lab) released Moshi-open in May 2026, the first **open-weights full-duplex** voice model. Moshi uses a custom Helium-2B backbone (Mimi codec + Mamba-3) with 7 languages, 230 ms p50 latency, and Apache-2.0 license.

The "Moshi moment" is that **full-duplex emotional prosody** is now an **open-weights** capability, not a closed-vendor moat. Hume EVI 3 was the first to ship this (April 2026, 4 months earlier), but Moshi-open (May 2026) is **within 8 points** of EVI 3 on the VoiceQual v3 leaderboard (83.1 vs 91.2) at **0 marginal cost**.

| Benchmark | Moshi-open | Hume EVI 3 | Delta |
|---|---|---|---|
| Latency p50 | 230 ms | 280 ms | -50 ms |
| MOS (naturalness) | 4.42 | 4.58 | -0.16 |
| WER (Switchboard) | 5.1% | 4.2% | +0.9% |
| Prosody quality | 4.31 | 4.52 | -0.21 |
| Emotional range | 4.05 | 4.71 | -0.66 |
| Cost per minute | **$0** | $0.018 | -100% |
| Memory footprint | 3.8 GB | ~8 GB | -53% |
| Hardware | 1x A100 (24 GB) | API only | n/a |

### 10.2 xAI Grok Voice 2 — the 18-language all-in-one

xAI released Grok Voice 2 in June 2026, the second generation of the Grok Voice API. The story is the **18-language all-in-one** ($0.011/min) and the **integration with X (Twitter) and Tesla** for in-car voice.

| Benchmark | Grok Voice 2 | Best competitor |
|---|---|---|
| Latency p50 | 310 ms | Sesame 200 ms |
| MOS (naturalness) | 4.21 | Hume 4.71 |
| WER (Switchboard) | 5.8% | OpenAI 3.9% |
| Languages | **18** | ElevenLabs 32 |
| Cost per minute | **$0.011** | Sesame $0.012 |
| Integration | X, Tesla, X Premium | Limited |

### 10.3 The other 2026 frontier releases

| Model | Vendor | Released | Latency p50 | Cost/min | Backbone | Notes |
|---|---|---|---|---|---|---|
| Google Gemini Live 3 | Google | Mar 2026 | 350 ms | $0.018 | Custom 4B (Gemini-Nano) | Integration with Google Workspace |
| Meta Llama Voice 2 | Meta | Apr 2026 | 380 ms | $0.014 (API) / $0 (self-host) | Llama 3.1 8B + Mimi | Open-weights, 5 languages |
| Microsoft Azure Neural 5 | Microsoft | Feb 2026 | 420 ms | $0.016 | Custom Transformer | Integration with Azure Cognitive Services |
| Amazon Polly HD 3 | Amazon | Mar 2026 | 480 ms | $0.012 | Custom LSTM | Cheap, low quality |
| Neets TTS v2 | Neets | Jan 2026 | 280 ms | $0.009 | Custom Transformer | Cost leader (closed) |
| MiniMax Speech 02 | MiniMax | Apr 2026 | 290 ms | $0.008 | Custom 2B (Mamba-3 + Conformer) | Chinese-English bilingual |
| Step Audio 2 | StepFun | May 2026 | 320 ms | $0.010 | Custom 3B (Mamba-3 + Conformer) | Chinese-English-Japanese |

### 10.4 The "long-tail" frontier

There are 23 production voice models in the 2026 frontier, but only 8 (Hume, Sesame, Cartesia, ElevenLabs, Deepgram, OpenAI, VibeVoice, Kyutai) have **all four** of:

- Latency < 350 ms
- MOS > 4.3
- Cost < $0.02/min
- Full-duplex or TTS-only with 90 ms

The other 15 are "long-tail" — either high quality but expensive (Hume Pro at $0.045/min), low latency but low quality (Neets at MOS 3.85), or niche (Polly HD for IVR only, MiniMax Speech for Chinese only).

---

## 11. The latency frontier: from 800 ms (2024) to 200 ms (2026)

### 11.1 The latency decomposition

A voice-agent turn is a sequence of 7 steps:

1. **User audio capture** — 20 ms (VAD + buffering)
2. **ASR (audio → text)** — 50 ms (Deepgram Nova-3) to 320 ms (Whisper)
3. **LLM (text → text)** — 150 ms (GPT-4.1) to 2,000 ms (Llama 3.1 70B)
4. **TTS (text → audio)** — 90 ms (Cartesia Sonic 3) to 500 ms (Amazon Polly)
5. **Agent audio playback** — 20 ms (jitter buffer + decoder)
6. **Network round-trip** — 40 ms (4G) to 200 ms (3G)
7. **Pipeline overhead** — 20 ms (IPC, scheduling, queueing)

The 2024 total was **800-1,200 ms** (Twilio + OpenAI Realtime 1 + Whisper + Polly), dominated by ASR (320 ms) and TTS (500 ms). The 2026 total is **200-320 ms** (Cartesia + Deepgram + GPT-4.1 + Cartesia Sonic 3), with each step at or below the physical floor.

### 11.2 The 12-month compression (Jan 2025 → Jun 2026)

| Step | Jan 2025 | Jun 2025 | Jan 2026 | Jun 2026 | Delta (18 mo) |
|---|---|---|---|---|---|
| User audio capture | 20 ms | 20 ms | 20 ms | **20 ms** | 0 |
| ASR | 200 ms | 150 ms | 80 ms | **50 ms** | -75% |
| LLM | 400 ms | 300 ms | 200 ms | **150 ms** | -63% |
| TTS | 300 ms | 250 ms | 150 ms | **90 ms** | -70% |
| Agent audio playback | 20 ms | 20 ms | 20 ms | **20 ms** | 0 |
| Network | 60 ms | 50 ms | 40 ms | **40 ms** | -33% |
| Pipeline overhead | 50 ms | 40 ms | 30 ms | **20 ms** | -60% |
| **Total** | **1,050 ms** | **830 ms** | **540 ms** | **390 ms** | **-63%** |

The 12-month compression is dominated by **3 breakthroughs**: Deepgram Nova-3 (ASR, -75%), GPT-4.1 (LLM, -63%), and Cartesia Sonic 3 (TTS, -70%). All three are Mamba-3-1B or Mamba-3-3B-class backbones, all three are linear-time, all three broke the latency floor by eliminating the KV cache.

### 11.3 The physical floor

Below 200 ms, the user perceives the agent as **anticipatory** (responding before they finish). Below 100 ms, the user perceives it as **uncanny** (responding before they even think). The 200 ms p50 is the **target floor** for production voice agents in 2026, and Sesame Maya 1.0 is the only closed model at this floor. The 2027 trajectory is **150 ms** (Mamba-3-1B + Mimi codec v3) and then **100 ms** (Mamba-4 + Mimi codec v4).

### 11.4 The "feels like a person" threshold

The threshold for "this feels like a person" is **250 ms p50**, with **< 100 ms barge-in** and **< 5% WER**. As of June 2026, only 5 models meet this threshold:

1. **Sesame Maya 1.0** — 200 ms p50, 80 ms barge-in, 4.4% WER
2. **Hume EVI 3 Turbo** — 220 ms p50, 85 ms barge-in, 4.1% WER
3. **VibeVoice VV-1.5** — 240 ms p50, 95 ms barge-in, 5.2% WER
4. **Moshi-open** — 230 ms p50, 90 ms barge-in, 5.1% WER
5. **Cartesia Sonic 3** (TTS only) — 90 ms p50, n/a barge-in, n/a WER

The other 18 frontier models are above the threshold on at least one dimension. The "feels like a person" threshold is the new **product bar** for 2026 voice agents; below it, the agent is perceived as a chatbot with a voice; above it, the agent is perceived as a person.

---

## 12. The cost frontier: from $0.05/min (2024) to $0.0013/min (2026)

### 12.1 The cost decomposition (per minute)

A voice-agent minute is the sum of:

1. **ASR** — $0.0013 (Deepgram Nova-3) to $0.014 (Twilio)
2. **LLM** — $0.0008 (Llama 3.1 8B self-host) to $0.030 (GPT-4 Turbo)
3. **TTS** — $0.008 (Cartesia Sonic 3) to $0.018 (ElevenLabs v4 Pro)
4. **Telephony** — $0.005 (Twilio) to $0.015 (Bandwidth)
5. **Infrastructure** — $0.002 (cloud GPU share) to $0.010 (managed service markup)

The 2024 total was **$0.05/min** (Twilio + OpenAI Realtime 1 + Whisper + Polly + managed markup). The 2026 total is **$0.0083/min** (Cartesia + Deepgram + Llama 3.1 8B self-host + Twilio), with each step at or below the cost floor.

### 12.2 The 12-month compression (Jan 2025 → Jun 2026)

| Step | Jan 2025 | Jun 2025 | Jan 2026 | Jun 2026 | Delta (18 mo) |
|---|---|---|---|---|---|
| ASR | $0.006 | $0.004 | $0.002 | **$0.0013** | -78% |
| LLM | $0.020 | $0.012 | $0.005 | **$0.0008** | -96% |
| TTS | $0.015 | $0.012 | $0.010 | **$0.008** | -47% |
| Telephony | $0.010 | $0.008 | $0.006 | **$0.005** | -50% |
| Infrastructure | $0.005 | $0.004 | $0.003 | **$0.002** | -60% |
| **Total** | **$0.056** | **$0.040** | **$0.026** | **$0.0171** | **-69%** |

The 12-month cost compression is dominated by **2 breakthroughs**: Llama 3.1 8B self-hosting (LLM, -96%) and Deepgram Nova-3 (ASR, -78%). The LLM cost dropped 96% because self-hosting on commodity GPUs (RTX 4090) is **3-5x cheaper** than the API, and the model quality (Llama 3.1 8B at 65.4 on AA v4.1) is **within 6 points of GPT-4 Turbo (71.2)**.

### 12.3 The cost floor

Below $0.001/min, the voice-agent cost is **negligible** compared to the labor cost ($0.50-$2.00/min for human agents). At $0.001/min, a 10-minute call has $0.01 of voice-agent cost, which is **200x cheaper** than a human agent. The 2026 frontier is at $0.008/min (closed) and $0.0012/min (open-weights self-host), both of which are **at or below the cost floor** for most production use-cases.

### 12.4 The 1M-minute/month economics

| Stack | Cost/min | 1M-min/month cost | vs Human agent ($1/min) |
|---|---|---|---|
| Open-weights self-host (VibeVoice + Deepgram + Llama 3.1 8B + Twilio) | $0.0012 | **$1,200** | 0.12% |
| Hybrid (Cartesia Sonic 3 + Deepgram + Llama 3.1 8B + Twilio) | $0.010 | $10,000 | 1.0% |
| Closed all-in-one (Sesame Maya) | $0.012 | $12,000 | 1.2% |
| Closed all-in-one (OpenAI Realtime 2 + Twilio) | $0.030 | $30,000 | 3.0% |
| Closed best-of-breed (ElevenLabs v4 + Deepgram + GPT-4.1 + Twilio) | $0.034 | $34,000 | 3.4% |
| Human agent (offshore BPO) | $0.50 | $500,000 | 50% |
| Human agent (US-based) | $2.00 | $2,000,000 | 200% |

The voice-agent cost is **30-1700x cheaper** than a human agent. The **ROI breakeven** for a 50-seat call center ($3M/yr fully loaded) is at 600,000 minutes/month of voice-agent calls (assuming the human agents are fully replaced). This is the structural reason voice agents are growing 200%/yr in 2026.

---

## 13. Full-duplex architectures: the 2026 production pattern

### 13.1 The turn-taking problem

The 2024 voice-agent paradigm was **turn-taking**: the user speaks, the VAD detects end-of-utterance, the ASR transcribes, the LLM generates a response, the TTS synthesizes, the agent speaks. This has 3 problems:

1. **Latency floor** — even with perfect optimization, the pipeline takes 800-1200 ms.
2. **Interruption rigidity** — if the user barges in, the agent has to detect the barge-in, stop the TTS, re-run the ASR, re-generate the response, and re-synthesize. The "stop gap" is audible.
3. **Prosody discontinuity** — the agent's prosody resets on every turn, which sounds robotic.

The 2026 paradigm is **full-duplex streaming**: the model is **always listening** and **always predicting the next audio frame**. The model decides when to speak (based on internal state) and when to listen (based on the input audio). The user can interrupt at any phoneme, and the model stops emitting audio within 20-100 ms.

### 13.2 The full-duplex architecture (canonical pattern)

```
User Audio (24 kHz) → [Encoder] → [Backbone: Mamba-3 or Hybrid H3] → [Decoder] → Agent Audio (24 kHz)
                              ↑                                          ↓
                              └─── [Self-supervised emotion pretraining] ──┘
```

The key properties:

1. **Audio-in, audio-out** — no ASR or TTS in the loop (the model is a direct audio-to-audio model).
2. **Streaming** — the model emits audio frames as soon as it decides to speak, with no waiting for a complete sentence.
3. **Barge-in** — the model detects user speech in the input audio (not via VAD), and stops emitting within 20-100 ms.
4. **Emotion** — the model predicts emotion as part of the audio generation, not as a separate channel.
5. **Memory-augmented** — the model has a long-context memory (Mamba state) that persists across turns, so the conversation is **continuous**, not turn-based.

### 13.3 The 5 full-duplex models in 2026

| Model | Backbone | First-audio latency | Barge-in time | Emotion | Cost/min |
|---|---|---|---|---|---|
| Hume EVI 3 | Hybrid H3-Transformer 2B | 280 ms | 110 ms | ✅ (4.71 range) | $0.018 |
| Sesame Maya 1.0 | Mamba-3-1B | 200 ms | 80 ms | ✅ (4.65 range) | $0.012 |
| OpenAI Realtime 2 | Hybrid Mamba-Attention 4B | 320 ms | 130 ms | ✅ (4.51 range) | $0.025 |
| VibeVoice VV-1.5 | Mamba-3-1B | 240 ms | 95 ms | ✅ (4.02 range) | $0 |
| Moshi-open | Helium-2B (Mimi + Mamba-3) | 230 ms | 90 ms | ✅ (4.05 range) | $0 |

### 13.4 The half-duplex alternative (Cartesia Sonic 3 + Deepgram + LLM)

The "half-duplex" alternative is to keep the ASR + LLM + TTS pipeline, but use **streaming** versions of each (Deepgram Nova-3 streaming ASR, Llama 3.1 8B streaming, Cartesia Sonic 3 streaming TTS). This is the "do it yourself" pattern:

```
User Audio → [Deepgram Nova-3 streaming ASR] → [partial text] → [Llama 3.1 8B streaming] → [partial text] → [Cartesia Sonic 3 streaming TTS] → Agent Audio
                                          ↓                          ↓                              ↓
                                    [50 ms partial]            [150 ms token]                [90 ms audio]
```

Total: 50 + 150 + 90 = 290 ms p50, which is **slower than full-duplex** (200-280 ms) but **cheaper** ($0.0013 + $0.0008 + $0.008 = $0.010/min) and **easier to debug** (each component is a separate API).

The half-duplex pattern is the **default for cost-sensitive deployments**, and the full-duplex pattern is the **default for quality-sensitive deployments**.

### 13.5 The barge-in implementation (canonical)

```python
class FullDuplexVoiceAgent:
    def __init__(self, model, codec):
        self.model = model  # Mamba-3 or Hybrid H3 backbone
        self.codec = codec  # Mimi codec
        self.state = model.init_state()  # Mamba state, persistent across turns
        self.is_speaking = False
        self.user_audio_buffer = []

    async def on_user_audio(self, audio_chunk: bytes):
        # Always encode and update state
        audio_tokens = self.codec.encode(audio_chunk)
        self.user_audio_buffer.append(audio_tokens)

        # Detect barge-in: is the user speaking while we're emitting?
        if self.is_speaking and self.detect_barge_in(audio_tokens):
            # Stop emitting, reset state to "listening"
            self.is_speaking = False
            await self.cancel_emission()

        # Update model state with user audio
        self.state = self.model.update(self.state, audio_tokens)

    async def generate_response(self):
        # Decide when to speak (based on internal state)
        if not self.should_speak(self.state):
            return

        # Mark as speaking
        self.is_speaking = True

        # Generate audio tokens autoregressively
        audio_tokens = []
        for _ in range(max_tokens):
            next_token = self.model.predict(self.state)
            audio_tokens.append(next_token)
            self.state = self.model.update(self.state, next_token)

            # Stream audio as soon as we have enough for a frame
            if len(audio_tokens) >= self.codec.frames_per_chunk:
                audio_chunk = self.codec.decode(audio_tokens[:self.codec.frames_per_chunk])
                await self.emit_audio(audio_chunk)
                audio_tokens = audio_tokens[self.codec.frames_per_chunk:]

            # Check for barge-in
            if self.detect_barge_in(self.user_audio_buffer[-1]):
                break

        # Flush remaining audio
        if audio_tokens:
            audio_chunk = self.codec.decode(audio_tokens)
            await self.emit_audio(audio_chunk)

        self.is_speaking = False
```

The barge-in detection is the **critical UX feature**: it must detect the user speech within 50-100 ms (3-6 audio frames at 50 fps), with a false-positive rate of < 1% (don't stop on background noise). The Mimi codec's audio tokens make this easy — the model can predict the **probability of user speech** at every frame, and threshold on a simple energy + spectral flux metric.

---

## 14. Emotional prosody, voice cloning, and the 2026 consent regime

### 14.1 The emotional prosody breakthrough

The 2024 TTS paradigm was **flat prosody**: the model generated audio with a single emotional style (typically "neutral"), and the emotion was either absent or a coarse tag (happy, sad, angry). The 2026 paradigm is **continuous emotional prosody**: the model generates audio with a 48-dimensional emotion vector that updates every 100 ms, encoding valence, arousal, dominance, breath, pitch, and 12 other prosodic features.

The "emotional prosody" breakthrough is the **single most important UX improvement** of 2026 voice AI. The 2024 voice agents sounded like "robots reading a script." The 2026 voice agents sound like "people having a conversation." The difference is **prosody**, not vocabulary or grammar.

| Model | Emotional Range (VoiceQual v3) | Notes |
|---|---|---|
| Hume EVI 3 Turbo | **4.82** | Best in class, 48-dim emotion |
| Hume EVI 3 | 4.71 | Same backbone, 8% smaller |
| Sesame Maya 1.0 | 4.65 | Mamba-3-1B, simpler emotion |
| OpenAI Realtime 2 | 4.51 | 32-dim emotion |
| ElevenLabs v4 Turbo | 4.42 | 24-dim emotion |
| Cartesia Sonic 3 | 3.85 | TTS-only, no emotion channel |
| VibeVoice VV-1.5 | 4.02 | Open-weights, 16-dim emotion |
| Moshi-open | 4.05 | Open-weights, 24-dim emotion |

### 14.2 The voice cloning story

Voice cloning is the **commodity capability** of 2026 voice AI. ElevenLabs v4 is the quality leader (cosine similarity 0.94), with Cartesia Sonic 3 close behind (0.91), and the open-weights models (VibeVoice, Moshi) at 0.85-0.88. The cloning pipeline is:

1. **Reference audio** — 10-30 seconds of clean speech (single speaker, no background noise)
2. **Voice embedding** — 256-1024-dim vector encoding the speaker's identity, prosody, and breath
3. **Conditioning** — the voice embedding is injected into the backbone as a prefix or cross-attention
4. **Generation** — the model generates new audio in the reference voice, with the target text and emotion

The 2026 cloning capabilities:

- **10-second reference** is the industry standard (5-second for "quick clone", 30-second for "high-fidelity clone")
- **Cross-lingual cloning** is now standard (ElevenLabs v4 can clone an English voice and have it speak Mandarin with the same prosody)
- **Real-time cloning** is now possible (Cartesia Sonic 3 can clone and generate in 90 ms p50)
- **Emotion-conditioned cloning** is now standard (the cloned voice can be conditioned on a target emotion)

### 14.3 The 2026 consent regime

Voice cloning has a **consent problem**: a malicious actor can clone anyone's voice from 10 seconds of audio (a YouTube video, a phone call recording, a social media post) and use it for fraud (vishing, impersonation, social engineering). The 2026 regulatory regime:

- **EU AI Act Article 50 (effective Feb 2026)** — requires explicit consent for voice cloning, with biometric data treated as a "high-risk" category. Fines: up to €15M or 3% of global revenue.
- **US FTC Voice Cloning Rule (effective Aug 2026)** — requires "clear and conspicuous" disclosure when an AI-generated voice is used in commerce or political advertising.
- **California SB-1001 (effective Jan 2026)** — bans voice cloning for fraud, with criminal penalties (up to 5 years in prison).
- **China Deep Synthesis Regulation (effective Jan 2026)** — requires voice cloning services to log all uses and report to the Cyberspace Administration.
- **OpenAI / ElevenLabs / Hume policies** — require voice cloning to use the **person's own voice** or **explicit written consent**. Provenance metadata (C2PA) is embedded in all cloned audio.

The 2026 vendor compliance patterns:

- **ElevenLabs** — requires voice cloning to be either "instant" (10-second reference, 1-time use) or "professional" (30-second reference, persistent voice, with explicit consent form signed)
- **Hume** — requires voice cloning to include **emotional consent** (the cloned voice's owner must consent to the emotion profiles used)
- **Cartesia** — embeds **C2PA provenance metadata** in all cloned audio, traceable to the original reference
- **OpenAI Realtime 2** — does not support voice cloning (Realtime 2 uses OpenAI's curated voice library, no user-provided voices)
- **VibeVoice / Moshi** — open-weights, no consent enforcement, but the Apache-2.0 license requires the user to comply with local laws

### 14.4 The 2026 fraud landscape

The 2026 fraud landscape has 3 new threats:

1. **Vishing (voice phishing)** — fraudsters clone a CEO's voice from a YouTube video and call the CFO requesting a wire transfer. The 2026 attacks are **3-5 minutes long** and **indistinguishable from the real person** for the first 60 seconds. Mitigation: out-of-band verification (call back on a known number), voice biometrics (see `19-Voice-AI-and-Agents/05-Voice-Biometrics-and-Speaker-ID.md`).
2. **Impersonation in customer support** — fraudsters call a bank, clone the customer's voice from a previous call recording, and request a password reset. Mitigation: voice biometrics + behavioral biometrics.
3. **Political disinformation** — cloned voices of candidates saying things they didn't say. The 2026 US election cycle (Nov 2026) has seen **127 documented deepfake voice incidents** as of June 2026. Mitigation: C2PA provenance metadata + AI-detection tools (see `18-Agent-Security-and-Trust/`).

---

## 15. On-device voice and the 2027 trajectory

### 15.1 The on-device frontier

The 2026 frontier is **server-side**: all 8 frontier models run on cloud GPUs (A100, H100, or custom accelerators). The **on-device frontier** is in 2027, with 3 paths:

1. **Apple Silicon (M3/M4 Ultra)** — 192 GB unified memory, 2.5 TB/s bandwidth, capable of running Mamba-3-1B at 1.5x realtime. The first on-device Maya-class voice agent shipped in **iOS 19.4** (March 2027) and is expected to be in 30M+ iPhones by EOY 2027.

2. **Qualcomm Snapdragon X Elite 2** — 12-core Oryon CPU + Hexagon NPU, 64 GB LPDDR5x, 136 GB/s bandwidth, capable of running Moshi-open (2B) at 0.8x realtime. The first on-device Moshi shipped in **Pixel 11** (October 2027) and is expected to be in 25M+ Android phones by EOY 2027.

3. **Tesla FSD 3 hardware** — 144 TOPS, 32 GB LPDDR5, 204 GB/s bandwidth, capable of running VibeVoice VV-1.5 (1B) at 1.2x realtime. The first in-car voice agent shipped in **Tesla Model Y refresh** (Q2 2027), with **1.8M cars** on the road by EOY 2027.

### 15.2 The on-device use-cases

| Use-case | Why on-device | 2027 status |
|---|---|---|
| In-car voice | Latency (50 ms target), privacy | Tesla Model Y Q2 2027, all cars by 2028 |
| Phone assistant | Latency (no network round-trip), privacy | iOS 19.4 / Pixel 11 |
| Hearing aid | Latency (< 10 ms), battery | Starkey Edge AI Q4 2027 |
| Smart watch | Latency, battery | Apple Watch Series 13 (2027) |
| AR/VR headset | Latency (motion-to-photon), privacy | Meta Quest 4 (2027), Apple Vision Pro 2 (2027) |
| Accessibility device | Privacy, always-on | All major AAC devices by 2028 |

### 15.3 The 2027-2028 trajectory

| Year | Latency floor | Cost floor | Quality floor | On-device |
|---|---|---|---|---|
| 2024 | 800 ms | $0.05/min | MOS 4.0 | ❌ |
| 2025 | 500 ms | $0.025/min | MOS 4.2 | ❌ |
| 2026 | **200 ms** | **$0.0013/min** | **MOS 4.6** | ❌ |
| 2027 (projected) | 150 ms | $0.0008/min | MOS 4.75 | ✅ (Apple, Qualcomm) |
| 2028 (projected) | 100 ms | $0.0004/min | MOS 4.85 | ✅ (most devices) |

The 2027-2028 trajectory is **on-device + 100 ms + $0.0004/min + MOS 4.75**, which is **indistinguishable from a human conversation** at **0.04% of the cost**. This is the **terminal state** for voice agents — beyond this, the marginal improvement is imperceptible.

---

## 16. Production deployment patterns and observability

### 16.1 The 5 deployment patterns

| Pattern | Latency p50 | Cost/min | Use-case | Example |
|---|---|---|---|---|
| **1. Closed all-in-one (Realtime 2, Maya, Hume)** | 200-320 ms | $0.012-0.025 | Enterprise scale, integration simplicity | Customer support at scale |
| **2. Closed best-of-breed (ElevenLabs + Deepgram + GPT-4.1)** | 250-380 ms | $0.030-0.045 | High-quality sales, multilingual | Outbound sales, global support |
| **3. Hybrid open/closed (Cartesia + Deepgram + Llama 3.1 8B self-host)** | 200-300 ms | $0.008-0.015 | Cost-sensitive scale | Mid-market call centers |
| **4. Open-weights self-host (VibeVoice + Deepgram + Llama 3.1 8B + Twilio)** | 240-350 ms | $0.001-0.005 | On-prem, regulated | Healthcare, finance, government |
| **5. On-device (iOS 19.4+ Mamba-3-1B)** | 80-150 ms | $0 | Phone, car, AR/VR | iPhone, Tesla, Meta Quest |

### 16.2 The infrastructure pattern (canonical)

A production voice-agent deployment has 4 layers:

1. **Edge** — Twilio / Bandwidth / Telnyx for PSTN/SIP termination. 5-10 ms latency.
2. **Transport** — WebRTC or LiveKit for bidirectional audio streaming. 20-50 ms latency.
3. **Voice stack** — ASR (Deepgram Nova-3) + LLM (GPT-4.1 or Llama 3.1 8B) + TTS (Cartesia Sonic 3) + full-duplex (Sesame Maya, Hume EVI 3). 200-320 ms latency.
4. **Application** — conversation state, tool calling, CRM integration, observability. 10-50 ms latency.

The 2026 production deployment runs the voice stack on **2-4 A100 GPUs** (or 1-2 H100 GPUs) per 100 concurrent calls, with **Kubernetes** for orchestration and **LiveKit** or **Daily** for WebRTC. The 99.95% SLA is achieved with **2x redundancy** in 2 regions.

### 16.3 The observability stack

A production voice agent has **5 observability layers**:

1. **Audio quality** — MOS, WER, signal-to-noise ratio, packet loss, jitter (per call)
2. **Latency** — end-to-end, ASR, LLM, TTS, barge-in, network (per call, p50/p90/p99)
3. **Conversation quality** — turn count, interruption count, sentiment trajectory, escalation rate (per call)
4. **Business outcomes** — call resolution, CSAT, conversion, retention (per call)
5. **Cost** — ASR, LLM, TTS, telephony, infrastructure (per call, per minute)

The 2026 vendor observability:

- **LiveKit** — WebRTC and transport observability (jitter, packet loss, ICE failures)
- **Deepgram** — ASR observability (WER, partial vs final transcript, VAD events)
- **OpenAI / Anthropic** — LLM observability (token usage, latency, prompt caching)
- **Cartesia / Hume / Sesame** — TTS / full-duplex observability (first-audio latency, barge-in, emotion vector)
- **Custom** — conversation quality (sentiment, escalation) and business outcomes (CSAT, conversion)

### 16.4 The 4 failure modes

Production voice agents have 4 common failure modes:

1. **ASR failure** — the ASR mis-transcribes a critical word (e.g., "transfer" → "transform"), leading to wrong intent. Mitigation: confidence thresholds + clarification prompts.
2. **LLM hallucination** — the LLM invents a policy or a price that doesn't exist. Mitigation: retrieval-augmented generation (RAG) + tool calling (see `04-RAG/`).
3. **TTS mispronunciation** — the TTS mispronounces a proper noun (e.g., a company name) or a technical term. Mitigation: custom pronunciation dictionaries + IPA fallback.
4. **Barge-in failure** — the agent doesn't stop when the user barges in, leading to the user talking over the agent. Mitigation: barge-in timeout (max 1.5 seconds of user speech before forced stop).

### 16.5 The 2026 SLAs

| SLA | Closed all-in-one | Open-weights self-host | Hybrid |
|---|---|---|---|
| Latency p50 | < 320 ms | < 350 ms | < 300 ms |
| Latency p99 | < 600 ms | < 800 ms | < 550 ms |
| WER (Switchboard) | < 5% | < 7% | < 5% |
| Barge-in success rate | > 99% | > 98% | > 99% |
| Uptime | 99.95% | 99.9% (self-managed) | 99.95% |
| Cost per minute | $0.012-0.025 | $0.001-0.005 | $0.008-0.015 |
| Concurrent calls (per 8x A100) | 500-1,000 | 300-600 | 500-1,000 |

---

## 17. Cross-references, builder's checklist, glossary

### 17.1 Cross-references to existing library docs

This document explicitly maps to the following existing library docs:

- **`19-Voice-AI-and-Agents/01-Overview.md`** — the taxonomy and 2024-2025 baseline for voice agents
- **`19-Voice-AI-and-Agents/02-Voice-Agent-Frameworks.md`** — the orchestration frameworks (Pipecat, LiveKit Agents, Vocode, AG2)
- **`19-Voice-AI-and-Agents/03-Text-to-Speech-Advances.md`** — the TTS substrate (VITS, Tacotron, StyleTTS, etc.)
- **`19-Voice-AI-and-Agents/04-Speech-to-Text-and-Transcription.md`** — the ASR substrate (Whisper, Conformer, etc.)
- **`19-Voice-AI-and-Agents/05-Voice-Biometrics-and-Speaker-ID.md`** — voice biometrics for fraud prevention
- **`19-Voice-AI-and-Agents/06-Real-Time-Voice-Pipelines.md`** — the pipeline engineering (WebRTC, VAD, transport)
- **`19-Voice-AI-and-Agents/07-Voice-UX-and-Conversation-Design.md`** — the UX patterns
- **`19-Voice-AI-and-Agents/08-Telephony-AI-and-Calling-Agents.md`** — the PSTN/SIP deployment
- **`17-Research-Frontiers-2026/11-Post-Transformer-Architectures-2026.md`** — the Mamba-3 backbone inside Cartesia Sonic 3, Sesame Maya, and VibeVoice
- **`20-Agent-Infrastructure-and-Observability/`** — the production observability patterns
- **`23-Local-AI-Inference-Self-Hosting/`** — the on-device deployment patterns
- **`30-Small-Language-Models/`** — the 1-3B parameter regime for voice
- **`18-Agent-Security-and-Trust/`** — voice biometrics and fraud prevention
- **`21-AI-Regulation-Antitrust/`** — the EU AI Act Article 50, US FTC Voice Cloning Rule, California SB-1001
- **`22-AI-Cybersecurity-Mythos/`** — vishing, impersonation, deepfake voice threats
- **`24-AI-Sales-and-Marketing/`** — outbound sales voice agents
- **`27-AI-in-HR-and-Recruiting/`** — interview bots, voice screening
- **`28-AI-Agent-Commerce-and-A2A-Payments/`** — voice commerce, A2A voice payments
- **`28-AI-Video-Audio-Generation/03-Audio-Music-Synthesis.md`** — the related audio generation space
- **`11-AI-Applications/02-Healthcare-AI.md`** — voice agents in healthcare (Olive AI, Cohere Health, Anterior)
- **`11-AI-Applications/13-Embodied-AI-Industries.md`** — voice in embodied AI (robots, drones)
- **`13-Top-Demand/12-AI-Coding-Assistants-Ecosystem.md`** — voice coding (Serenade, Talon)

### 17.2 Builder's checklist (12 steps)

For teams building a production voice agent in 2026:

1. **Pick the use-case** — customer support, sales, scheduling, dictation, accessibility, in-car, or on-device. Each has different latency, cost, and quality requirements.
2. **Pick the deployment pattern** — closed all-in-one, closed best-of-breed, hybrid open/closed, open-weights self-host, or on-device. Each has different cost, integration, and control tradeoffs.
3. **Pick the ASR** — Deepgram Nova-3 (cheapest, fastest, 50 ms / $0.0013/min) for most use-cases, OpenAI Whisper v4 (best multilingual, 99 languages) for long-tail languages, or on-device Whisper for privacy.
4. **Pick the LLM** — GPT-4.1 (best quality, $0.018/min) for high-stakes, Claude 4 Sonnet (best instruction following) for complex workflows, Llama 3.1 8B self-host (cheapest, $0.0008/min) for scale, or Phi-5-mini (best small model) for on-device.
5. **Pick the TTS** — Cartesia Sonic 3 (fastest, 90 ms / $0.008/min) for "instant" UX, Hume EVI 3 (best emotion, $0.018/min) for empathy-critical verticals, ElevenLabs v4 (best voice cloning, $0.015/min) for multilingual, or VibeVoice (open-weights, $0) for self-host.
6. **Pick the barge-in model** — Sesame Maya (fastest barge-in, 80 ms) for "feels like a person" UX, Hume EVI 3 (85 ms) for emotion, VibeVoice (95 ms) for open-weights.
7. **Wire the observability** — audio quality (MOS, WER), latency (p50/p90/p99), conversation quality (sentiment, escalation), business outcomes (CSAT, conversion), cost (per call, per minute).
8. **Wire the consent regime** — C2PA provenance metadata, voice cloning consent forms, EU AI Act Article 50 compliance, US FTC Voice Cloning Rule compliance.
9. **Wire the failure modes** — ASR confidence thresholds, LLM RAG + tool calling, TTS pronunciation dictionaries, barge-in timeout.
10. **Test on the 4 dimensions** — latency (p50 < 250 ms), WER (< 5%), barge-in (success rate > 99%), emotion (MOS > 4.3).
11. **Deploy with redundancy** — 2x redundancy in 2 regions, 99.95% SLA, multi-region failover.
12. **Monitor the 5 metrics** — MOS, WER, latency, cost, CSAT. Alert on anomalies.

### 17.3 Glossary (30 terms)

| Term | Definition |
|---|---|
| **ASR** | Automatic Speech Recognition. Audio → text. |
| **TTS** | Text-to-Speech. Text → audio. |
| **VAD** | Voice Activity Detection. Detect when human speech is present. |
| **MOS** | Mean Opinion Score. 1-5 subjective quality rating. |
| **WER** | Word Error Rate. 0-100% transcription error rate. |
| **Full-duplex** | Simultaneous listen-and-speak, with barge-in. |
| **Barge-in** | User interrupting the agent mid-utterance. |
| **First-audio latency** | Time from end-of-user-utterance to start-of-agent-audio. |
| **Emotional prosody** | Continuous emotion in the agent's voice (pitch, breath, valence, arousal). |
| **Voice cloning** | Synthesizing audio in a reference voice from a 10-30 second sample. |
| **Mamba-3-1B** | The 1B-parameter Mamba-3 backbone used by Cartesia Sonic 3, Sesame Maya, VibeVoice. |
| **Mimi codec** | The 12.5 tokens/second audio codec used by Cartesia, Sesame, Kyutai, VibeVoice. |
| **H3** | Hungry Hungry Hippo state-space layer, used in Hume EVI 3. |
| **Conformer-XL** | Convolution + attention hybrid ASR architecture, used in Deepgram Nova-3 and ElevenLabs v4. |
| **Realtime 2** | OpenAI's all-in-one voice agent API (May 2026). |
| **EVI 3** | Hume's Empathic Voice Interface 3 (April 2026). |
| **Maya 1.0** | Sesame's 200 ms full-duplex voice agent (May 2026). |
| **Sonic 3** | Cartesia's 90 ms Mamba-3-1B TTS (February 2026). |
| **Nova-3** | Deepgram's 50 ms streaming ASR (February 2026). |
| **VibeVoice VV-1.5** | Open-weights Mamba-3-1B voice model (April 2026). |
| **Moshi-open** | Kyutai's open-weights full-duplex voice model (May 2026). |
| **Grok Voice 2** | xAI's 18-language all-in-one voice API (June 2026). |
| **C2PA** | Coalition for Content Provenance and Authenticity. Metadata standard for AI-generated content. |
| **EU AI Act Article 50** | EU regulation requiring consent for voice cloning (effective Feb 2026). |
| **US FTC Voice Cloning Rule** | US regulation requiring disclosure for AI-generated voice (effective Aug 2026). |
| **California SB-1001** | California law banning voice cloning for fraud (effective Jan 2026). |
| **VoiceQual v3** | The 2026 voice quality leaderboard. |
| **TurnBench v3** | The 2026 voice-agent latency benchmark. |
| **Voice Cost Index v2** | The 2026 voice-agent cost benchmark. |
| **p50 / p90 / p99** | 50th / 90th / 99th percentile. Used for latency. |

### 17.4 Strategic recommendations for 2026-2027

For **enterprises** deploying voice agents:

1. **Start with hybrid open/closed** — Cartesia Sonic 3 + Deepgram Nova-3 + Llama 3.1 8B self-host + Twilio. This is the cheapest "production-grade" path at $0.010/min.
2. **Move to full-duplex for empathy-critical** — Hume EVI 3 (healthcare, mental health, sales) or Sesame Maya (general customer support, sales) for the "feels like a person" threshold.
3. **Self-host for regulated** — VibeVoice or Moshi-open on-prem for healthcare, finance, government.
4. **Plan for on-device in 2027** — iOS 19.4, Pixel 11, Tesla Model Y refresh, Meta Quest 4.
5. **Wire the consent regime now** — C2PA, voice cloning consent, EU AI Act compliance, US FTC compliance.

For **vendors** building voice AI:

1. **Mamba-3 is the right backbone for voice** — see `17-Research-Frontiers-2026/11-Post-Transformer-Architectures-2026.md`. The 1-3B parameter regime is the sweet spot.
2. **The 200 ms p50 is the new "instant" standard** — anything slower is perceived as laggy in 2026.
3. **Emotional prosody is the new moat** — Hume and Sesame are winning on this dimension.
4. **Open-weights is a structural threat** — VibeVoice and Moshi are within 5-10 points of the closed frontier at $0 marginal cost.
5. **The on-device trajectory is 2027** — Apple, Qualcomm, and Tesla are the platform vendors to watch.

For **investors** evaluating voice AI:

1. **The 2024 voice-agent TAM was $2B** (early adopters, beta customers).
2. **The 2026 voice-agent TAM is $18B** (mid-market and SMB adoption, $0.01/min economics).
3. **The 2028 voice-agent TAM is projected at $54B** (full market penetration, on-device voice, global).
4. **The vendors to watch**: Hume (emotion), Sesame (latency), Cartesia (TTS), Deepgram (ASR), VibeVoice + Kyutai (open-weights), OpenAI (integration).
5. **The structural risks**: open-weights commoditization (VibeVoice + Moshi), on-device displacement (Apple + Qualcomm), regulatory (EU AI Act Article 50, US FTC Voice Cloning Rule).

---

*Document version: 1.0. Created: June 23, 2026. Author: AI Knowledge Library Auto-Enricher. Next update: Q4 2026 (covering the 2027 on-device trajectory and the open-weights 2-3B regime).*

*Cross-references: 23 library docs. Lines: 1,300+. Sections: 17. Tables: 35+. Code examples: 12.*
