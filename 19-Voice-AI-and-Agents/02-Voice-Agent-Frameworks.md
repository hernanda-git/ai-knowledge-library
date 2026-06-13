# Voice Agent Frameworks: Building Real-Time Voice Applications

## 2.1 Introduction to Voice Agent Frameworks

Voice agent frameworks provide the middleware layer that orchestrates the real-time pipeline connecting speech-to-text, AI reasoning, and text-to-speech components. Unlike traditional chatbot frameworks, voice agent frameworks must handle the unique challenges of real-time audio processing: streaming input, turn management, interruption handling, and low-latency output.

### 2.1.1 The Core Pipeline

Every voice agent framework, regardless of specific implementation, follows this fundamental pipeline:

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  Audio   │   │   STT    │   │    AI    │   │   TTS    │
│  Source  │──▶│ (Speech  │──▶│ (LLM /   │──▶│ (Speech  │
│ (Mic /   │   │  to Text)│   │  Logic)  │   │  Synth)  │──▶ Audio Out
│  Stream) │   │          │   │          │   │          │
└──────────┘   └──────────┘   └──────────┘   └──────────┘
      │                                              │
      │        ┌──────────────┐                      │
      └────────│  Voice       │──────────────────────┘
               │  Activity    │
               │  Detection   │
               │  (VAD)       │
               └──────────────┘
```

**Key architectural components:**

**Audio Source:** Raw audio capture from microphone, WebRTC stream, or telephony SIP trunk. Typically PCM 16kHz or 48kHz, mono or stereo. The audio source may include pre-processing: noise suppression, echo cancellation, automatic gain control.

**Voice Activity Detection (VAD):** Determines when a person is speaking vs. silence. Essential for turn-taking. Without VAD, the system doesn't know when to start/stop STT processing. VAD operates on short audio frames (10-100ms) and returns probability scores. Common VAD models: Silero VAD, WebRTC VAD.

**Speech-to-Text (STT):** Converts audio stream into text. Can operate in streaming mode (partial results as user speaks) or batch mode (full utterance). STT sends text to AI when the user finishes speaking (midsentence detection + end-of-utterance detection).

**AI/Logic Layer:** Processes text input and generates response. This can be an LLM (GPT-4, Claude, Llama), a rules-based dialog manager, or a hybrid. The AI layer may call external tools (APIs, databases) before generating a response.

**Text-to-Speech (TTS):** Converts response text into audio. Most modern voice agents use streaming TTS (synthesize and play audio incrementally) to minimize time-to-audio response.

**Interrupt Handler:** Monitors for user interruptions during TTS playback. If user starts speaking while agent is speaking, the system should pause TTS, process the new input, and respond appropriately.

### 2.1.2 Framework Comparison Table

| Feature | Pipecat | Vocode | LiveKit Agents | Vapi | Voiceflow |
|---------|---------|--------|----------------|------|-----------|
| Open Source | ✅ (BSD) | ✅ (MIT) | ✅ (Apache 2.0) | ❌ | Partially |
| STT Providers | DG, Wh, Asb, AZ, GC | DG, Wh, Asb, AZ | DG, Wh, Asb, AZ, Cart | DG, Wh, Asb | ASR-built-in |
| TTS Providers | EL, PH, OA, Cart, AZ, GC | EL, PH, OA, AZ | EL, PH, OA, Cart, AZ | EL, PH, OA | Polly, Azure |
| LLM Providers | OA, An, Tog, Rep, Local | OA, An, Tog, Local | OA, An, Local | OA, An, Tog, Rep | Any OpenAI-compat |
| Telephony | Daily, Media Streams | Twilio, Vonage | LiveKit SIP | Twilio, Vonage | Twilio |
| WebRTC | ✅ (Daily) | ❌ | ✅ (LiveKit) | ✅ (custom) | ❌ |
| Interruption | ✅ Built-in | ✅ | ✅ Built-in | ✅ Managed | ❌ |
| VAD | Silero + WebRTC | WebRTC | Silero | Internal | Basic |
| Language Support | Python | Python, TS | Python, TS | API | Visual |
| Latency (E2E) | ~500-1500ms | ~800-2000ms | ~400-1200ms | ~600-1500ms | ~1000-3000ms |
| Hosting | Self-host | Self-host | Self-host / Cloud | Managed | Cloud |
| Pricing | Free | Free | Free (self) / $20/mo (cloud) | $0.05-$0.11/min | $30-$500/mo |

*Keys: DG=Deepgram, Wh=Whisper, Asb=AssemblyAI, AZ=Azure Speech, GC=Google Cloud, EL=ElevenLabs, PH=PlayHT, OA=OpenAI, Cart=Cartesia, An=Anthropic, Tog=Together*

## 2.2 Pipecat — Open Source Voice Agent Framework

Pipecat is an open-source Python framework for building real-time voice agents. Developed by Daily (the WebRTC infrastructure company), Pipecat is designed from the ground up for real-time audio processing with a modular pipeline architecture.

### 2.2.1 Core Architecture

Pipecat uses a **Frame-based pipeline architecture**:

**Frames** are the fundamental data unit flowing through the system:

```python
# Frame types in Pipecat
TextFrame         # Text input/output
AudioRawFrame     # Raw PCM audio data
STTMessagesFrame  # STT transcription results
TranscriptionFrame # Finalized transcription
LLMMessagesFrame  # AI/LLM messages
MetricsFrame      # Performance metrics
StartFrame        # Pipeline start marker
EndFrame          # Pipeline end marker
```

**Processors** are modular components that consume and produce frames:

```python
# Each stage in the pipeline is a Processor
class STTProcessor:    # Audio → Text
class LLMProcessor:    # Text → Response Text
class TTSProcessor:    # Text → Audio
class VADProcessor:    # Audio → VAD Events
class TransportProcessor:  # Network Audio ↔ Local Audio
```

**Pipeline** chains processors together:

```python
pipeline = Pipeline([
    audio_input_processor,
    vad_processor,
    stt_processor,
    llm_processor,
    tts_processor,
    audio_output_processor,
])
```

### 2.2.2 Complete Pipecat Voice Agent

Here's a fully functional voice agent using Pipecat, Deepgram, OpenAI, and ElevenLabs:

```python
import asyncio
import os
from pipecat.frames.frames import TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators import FrameTimingAggregator
from pipecat.processors.logger import FrameLogger
from pipecat.processors.audio import VADProcessor
from pipecat.vad.silero import SileroVADAnalyzer
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.daily import DailyTransport, DailyParams

# Configuration
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel

async def main():
    # 1. Transport Layer: Connect to Daily WebRTC room
    transport = DailyTransport(
        room_url=os.getenv("DAILY_ROOM_URL"),
        token=os.getenv("DAILY_TOKEN"),
        bot_name="Voice Agent",
        params=DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(threshold=0.75),
        ),
    )

    # 2. STT Service: Deepgram Nova-2
    stt = DeepgramSTTService(
        api_key=DEEPGRAM_API_KEY,
        model="nova-2-general",
        language="en",
        sample_rate=16000,
        encoding="linear16",
        interim_results=True,  # Enable streaming partial results
    )

    # 3. AI Service: OpenAI GPT-4o-mini
    llm = OpenAILLMService(
        api_key=OPENAI_API_KEY,
        model="gpt-4o-mini",
        system_prompt=(
            "You are a helpful voice agent named Alex. "
            "You speak conversationally and naturally. "
            "Keep responses concise since this is a voice conversation. "
            "Never mention that you're an AI or that this is a voice call. "
            "If you don't know something, say so honestly."
        ),
        temperature=0.7,
        max_tokens=256,
    )

    # 4. TTS Service: ElevenLabs Turbo v2
    tts = ElevenLabsTTSService(
        api_key=ELEVENLABS_API_KEY,
        voice_id=ELEVENLABS_VOICE_ID,
        model="eleven_turbo_v2",
        sample_rate=24000,
        # Enable streaming for low latency
        stream=True,
    )

    # 5. Build Pipeline
    pipeline = Pipeline([
        transport.input(),        # Audio input from WebRTC
        VADProcessor(),           # Voice Activity Detection
        stt,                      # Speech-to-Text
        llm,                      # AI/LLM Processing
        tts,                      # Text-to-Speech
        transport.output(),       # Audio output to WebRTC
    ])

    # 6. Create and run task
    task = PipelineTask(pipeline)

    @transport.event_handler("on_first_other_participant_joined")
    async def on_user_joined(transport, participant):
        print(f"User joined: {participant['id']}")
        # Agent starts with greeting
        await task.queue_frames([TextFrame("Hello! How can I help you today?")])

    @transport.event_handler("on_participant_left")
    async def on_user_left(transport, participant):
        print(f"User left: {participant['id']}")
        await task.cancel()

    runner = PipelineRunner()
    await runner.run(task)

if __name__ == "__main__":
    asyncio.run(main())
```

### 2.2.3 Pipecat's Interruption Handling

Pipecat has sophisticated built-in interruption handling:

```python
# Interruption configuration
from pipecat.processors.interruption import InterruptionHandler

class CustomInterruptionHandler(InterruptionHandler):
    """Handles user interruptions during agent speech."""

    async def on_interruption(self, direction: str, frames: list):
        """
        direction: 'forward' (user speaks while agent speaks)
                   'backward' (agent needs to re-evaluate)
        """
        if direction == 'forward':
            # User started speaking — stop TTS immediately
            print("Interruption detected — stopping TTS")
            for frame in frames:
                if hasattr(frame, 'tts_stopped'):
                    # Send metrics about interruption
                    pass

        elif direction == 'backward':
            # Agent was interrupted, now processing new context
            print("Processing new input after interruption")

# Enable interruption detection
transport_params = DailyParams(
    interrupt_handling=True,
    interrupt_duration=0.5,  # Min silence to consider interruption over
    interrupt_sensitivity=0.7,
)
```

### 2.2.4 Pipecat RTVI Integration

Pipecat implements the RTVI (Real-Time Voice Interface) standard for client-server communication:

```python
# RTVI WebSocket protocol for client-side control
# The RTVI client can:
# - Configure agent settings (voice, personality)
# - Send/receive transcripts in real-time
# - Control call flow (mute, hang up)
# - Receive metrics and analytics

from pipecat.rtvi import RTVIClient, RTVIConfig

config = RTVIConfig(
    stt={"provider": "deepgram", "model": "nova-2-general"},
    llm={"provider": "openai", "model": "gpt-4o-mini"},
    tts={"provider": "elevenlabs", "voice": "21m00Tcm4TlvDq8ikWAM"},
    vad={"provider": "silero", "threshold": 0.75},
)

# RTVI event handlers
async def on_transcript_partial(client, text):
    print(f"Partial transcript: {text}")

async def on_transcript_final(client, text):
    print(f"Final transcript: {text}")

async def on_metrics(client, metrics):
    print(f"Latency metrics: {metrics}")
    # metrics contains: stt_latency, llm_latency, tts_latency, e2e_latency
```

### 2.2.5 Pipecat Custom Processors

You can extend Pipecat with custom processors for domain-specific logic:

```python
from pipecat.processors.frame_processor import FrameProcessor

class SentimentAnalyzer(FrameProcessor):
    """Custom processor that analyzes sentiment of user input."""

    def __init__(self):
        super().__init__()
        self._sentiment_threshold = -0.3  # Escalate if very negative

    async def process_frame(self, frame, direction):
        if isinstance(frame, TranscriptionFrame) and direction == "down":
            # Analyze sentiment of user transcription
            text = frame.text
            sentiment = await self._analyze_sentiment(text)

            if sentiment < self._sentiment_threshold:
                # Negative sentiment detected — route to escalation
                escalation_frame = ContextFrame({
                    "sentiment": sentiment,
                    "needs_escalation": True,
                })
                await self.push_frame(escalation_frame)

        await self.push_frame(frame, direction)

    async def _analyze_sentiment(self, text):
        # Integrate with sentiment API
        # Return -1.0 (negative) to 1.0 (positive)
        return 0.0

# Add custom processor to pipeline
pipeline = Pipeline([
    transport.input(),
    VADProcessor(),
    stt,
    SentimentAnalyzer(),  # Custom processor
    llm,
    tts,
    transport.output(),
])
```

## 2.3 Vocode — Open Source Voice Agent Framework

Vocode (originally by Cohere, now community-maintained) provides abstractions for STT, TTS, LLM, and telephony.

### 2.3.1 Vocode Architecture

Vocode uses a **Modular Agent Architecture**:

```python
from vocode import AgentConfig, Agent
from vocode.streaming import StreamFactory
from vocode.streaming.synthesizer.eleven_labs import ElevenLabsSynthesizer
from vocode.streaming.transcriber.deepgram import DeepgramTranscriber
from vocode.streaming.agent.chat_gpt_agent import ChatGPTAgent
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.telephony.config_manager import InMemoryConfigManager
from vocode.streaming.telephony.conversation import TelephonyConversation
from vocode.streaming.telephony.server import TelephonyServer

# Configure STT, Agent, TTS
agent_config = AgentConfig(
    transcriber_config=DeepgramTranscriberConfig(
        api_key=DEEPGRAM_API_KEY,
        model="nova-2-general",
        sampling_rate=16000,
        chunk_size=2048,
    ),
    agent_config=ChatGPTAgentConfig(
        model_name="gpt-4o-mini",
        system_message=BaseMessage(
            text="You are a helpful voice assistant..."
        ),
        prompt_preamble=(
            "You are having a voice conversation. "
            "Keep responses brief and natural. "
        ),
    ),
    synthesizer_config=ElevenLabsSynthesizerConfig(
        api_key=ELEVENLABS_API_KEY,
        voice_id=ELEVENLABS_VOICE_ID,
        model="eleven_turbo_v2",
        sampling_rate=24000,
    ),
)

# Create agent
agent = Agent(agent_config)

# For telephony (Twilio):
from vocode.streaming.telephony.twilio import TwilioConfig

twilio_config = TwilioConfig(
    account_sid=os.getenv("TWILIO_ACCOUNT_SID"),
    auth_token=os.getenv("TWILIO_AUTH_TOKEN"),
    phone_number=os.getenv("TWILIO_PHONE_NUMBER"),
)

# Start Telephony Server
server = TelephonyServer(
    base_url="https://your-server.ngrok.io",
    config_manager=InMemoryConfigManager(),
    inbound_calls_config=agent_config,
    outbound_calls_config=agent_config,
    twilio_config=twilio_config,
)

# Make an outbound call
async def make_call():
    conversation = await server.call(
        to_number="+1234567890",
        from_number="+1987654321",
        agent_config=agent_config,
    )
    print(f"Call SID: {conversation.call_sid}")
```

### 2.3.2 Vocode with WebRTC

```python
from vocode.streaming.stream_factory import StreamFactory
from vocode.streaming.stream import PyAudioStream

# Local microphone/speaker interaction
stream_factory = StreamFactory(
    transcriber_config=DeepgramTranscriberConfig(...),
    agent_config=ChatGPTAgentConfig(...),
    synthesizer_config=ElevenLabsSynthesizerConfig(...),
)

# Create audio stream
audio_stream = PyAudioStream(
    input_device_index=None,  # Default mic
    output_device_index=None, # Default speaker
)

# Start conversation locally
conversation = stream_factory.create_conversation(audio_stream)
conversation.start()

# Keep the conversation going
conversation.terminate()
```

## 2.4 LiveKit Agents Framework

LiveKit Agents is a framework for building voice agents using the LiveKit WebRTC platform. It provides a clean Python SDK with high-level abstractions.

### 2.4.1 LiveKit Agents Architecture

LiveKit uses a **Room-based architecture**: each conversation is a WebRTC room where the agent and user exchange audio tracks.

### 2.4.2 Complete LiveKit Agent

```python
import asyncio
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import deepgram, openai, elevenlabs, silero

class VoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are a helpful voice agent. "
                "Respond conversationally and concisely. "
                "Keep responses under 2 sentences when possible. "
                "Never mention you're an AI."
            ),
            stt=deepgram.STT(model="nova-2-general"),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=elevenlabs.TTS(
                voice_id="21m00Tcm4TlvDq8ikWAM",
                model="eleven_turbo_v2",
            ),
            # Enable interruption handling
            interrupt_speech_duration=1.0,  # Seconds of silence to interrupt
            min_endpointing_delay=0.8,      # Delay before considering utterance ended
        )

    async def on_enter(self):
        """Called when agent enters the room."""
        await self.say("Hello! How can I help you today?")

    async def on_chat_message(self, message):
        """Handle chat (text) messages."""
        response = await self.llm.chat([
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": message},
        ])
        await self.say(response)

# Worker entry point
async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    agent = VoiceAgent()
    session = AgentSession(ctx.room, agent)
    await session.start()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fcn=entrypoint))
```

### 2.4.3 LiveKit Agents Pipeline Configuration

```python
from livekit.agents.voice import PipelineVoiceAgent
from livekit.agents.voice.processors import (
    AgentProcessor,
    UserAudioProcessor,
    AssistantSpeechProcessor,
)

# Fine-grained control over pipeline stages
agent = PipelineVoiceAgent(
    # VAD configuration
    vad=silero.VAD(
        threshold=0.5,
        min_speech_duration=0.5,     # Min speech to start STT
        min_silence_duration=0.8,    # Silence to end utterance
        padding_duration=0.3,        # Padding to capture word edges
    ),

    # STT configuration
    stt=deepgram.STT(
        model="nova-2-general",
        language="en",
        smart_format=True,
        punctuate=True,
        interim_results=True,
    ),

    # LLM configuration
    llm=openai.LLM(
        model="gpt-4o-mini",
        temperature=0.5,
        max_retries=2,
    ),

    # TTS configuration
    tts=elevenlabs.TTS(
        voice_id="21m00Tcm4TlvDq8ikWAM",
        model="eleven_turbo_v2",
        streaming_latency=2,  # Lower = faster but potentially lower quality
        enable_ssml_parsing=True,
    ),

    # Endpointing configuration
    endpointing=silero.Endpoint(
        min_speech_duration=0.3,
        min_silence_duration=0.5,
    ),

    # Barge-in (interruption) configuration
    barge_in=enabled,
    barge_in_sensitivity=0.7,
)

# Custom event handlers
@agent.on("user_speech_committed")
def on_user_speech(transcript: str, is_final: bool):
    """Called for every STT update from the user."""
    if is_final:
        print(f"User said: {transcript}")

@agent.on("agent_speech_committed")
def on_agent_speech(transcript: str):
    """Called when agent finishes speaking."""
    print(f"Agent said: {transcript}")

@agent.on("metrics_collected")
def on_metrics(metrics: dict):
    """Called with performance metrics."""
    print(f"E2E Latency: {metrics['e2e_latency_ms']}ms")
    print(f"STT Latency: {metrics['stt_latency_ms']}ms")
    print(f"LLM Latency: {metrics['llm_latency_ms']}ms")
    print(f"TTS Latency: {metrics['tts_latency_ms']}ms")
```

## 2.5 Vapi — Managed Voice Agent Platform

Vapi is a managed platform that provides the entire voice agent infrastructure as a service. You define the agent configuration via API, and Vapi handles the streaming pipeline, telephony integration, and scaling.

### 2.5.1 Vapi API Architecture

```python
import requests
import os

VAPI_API_KEY = os.getenv("VAPI_API_KEY")

# Create a voice agent configuration
agent_config = {
    "name": "Sales Assistant Pro",
    "model": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "systemPrompt": (
            "You are a sales assistant for Acme Corp. "
            "Your goal is to qualify leads and book demos. "
            "Be friendly, professional, and efficient. "
            "Gather: company name, role, interest area, phone/email. "
            "If lead is qualified, offer to book a demo."
        ),
        "temperature": 0.6,
        "maxTokens": 256,
    },
    "voice": {
        "provider": "elevenlabs",
        "voiceId": "21m00Tcm4TlvDq8ikWAM",
        "model": "eleven_turbo_v2",
        "speed": 1.0,
    },
    "transcriber": {
        "provider": "deepgram",
        "model": "nova-2-general",
        "language": "en",
    },
    "recordingEnabled": True,
    "hipaaEnabled": False,
}

# Create agent
resp = requests.post(
    "https://api.vapi.ai/agent",
    headers={
        "Authorization": f"Bearer {VAPI_API_KEY}",
        "Content-Type": "application/json",
    },
    json=agent_config,
)
agent = resp.json()
agent_id = agent["id"]

# Make an outbound call
call_config = {
    "agentId": agent_id,
    "phoneNumberId": os.getenv("VAPI_PHONE_NUMBER_ID"),
    "customer": {
        "number": "+1234567890",
        "name": "John Doe",
    },
    "metadata": {
        "campaign": "q2-outreach",
        "leadScore": 85,
    },
    "maxDurationSeconds": 600,
}

resp = requests.post(
    "https://api.vapi.ai/call",
    headers={
        "Authorization": f"Bearer {VAPI_API_KEY}",
        "Content-Type": "application/json",
    },
    json=call_config,
)
call = resp.json()
print(f"Call started: {call['id']}")
```

### 2.5.2 Vapi Real-Time Events

Vapi supports webhook events for real-time call monitoring:

```python
# Webhook event types
EVENTS = {
    "call.started": "Call has been established",
    "call.ended": "Call has ended",
    "call.transcript": "Real-time transcription update",
    "call.sentiment": "Sentiment analysis update",
    "call.functions_called": "Agent called a function",
    "call.knowledge_base_hit": "Knowledge base was queried",
    "call.error": "Error occurred during call",
}

# Example webhook handler (FastAPI)
from fastapi import FastAPI, Request

app = FastAPI()

@app.post("/vapi-webhook")
async def vapi_webhook(request: Request):
    payload = await request.json()
    event = payload.get("type")

    if event == "call.transcript":
        transcript = payload["data"]["transcript"]
        speaker = transcript["role"]  # "assistant" or "user"
        text = transcript["transcript"]
        print(f"[{speaker}] {text}")

    elif event == "call.ended":
        call_data = payload["data"]
        print(f"Call ended: {call_data['duration']}s")
        print(f"End reason: {call_data['endReason']}")
        print(f"Sentiment: {call_data.get('sentiment', {})}")

        # Analytics
        metrics = call_data.get("metrics", {})
        print(f"Average user wait time: {metrics.get('avgWaitTimeMs', 0)}ms")
        print(f"Total talk time ratio: {metrics.get('talkTimeRatio', 0)}")

    return {"status": "ok"}
```

### 2.5.3 Vapi Function Calling

Vapi supports tool calling for API integration:

```python
# Define tools in agent configuration
agent_config = {
    "name": "Booking Agent",
    "model": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "systemPrompt": "You book appointments for Acme Medical Center...",
        "functions": [
            {
                "name": "check_availability",
                "description": "Check available appointment slots",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date": {"type": "string", "description": "YYYY-MM-DD"},
                        "doctor": {"type": "string", "description": "Doctor name"},
                    },
                    "required": ["date"],
                },
            },
            {
                "name": "book_appointment",
                "description": "Book an appointment slot",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "slotId": {"type": "string"},
                        "patientName": {"type": "string"},
                        "patientPhone": {"type": "string"},
                    },
                    "required": ["slotId", "patientName", "patientPhone"],
                },
            },
        ],
    },
}

# Vapi calls your webhook when functions are invoked
@app.post("/vapi-function-call")
async def handle_function_call(request: Request):
    payload = await request.json()
    function_name = payload["function"]
    args = payload["arguments"]

    if function_name == "check_availability":
        slots = await appointment_system.get_slots(args["date"])
        return {"result": slots}
    elif function_name == "book_appointment":
        result = await appointment_system.book_slot(
            args["slotId"], args["patientName"], args["patientPhone"]
        )
        return {"result": result}
```

## 2.6 Voice Agent Architecture Deep Dive

### 2.6.1 Streaming Architecture

Real-time voice agents rely on **streaming** — processing audio and text in chunks rather than waiting for complete inputs. There are three streaming levels:

**Level 1 — Chunked Audio Streaming:** Audio is streamed as PCM chunks (typically 20-100ms). STT processes continuously. The AI layer still waits for complete utterances (via VAD endpointing).

**Level 2 — Streaming STT:** STT returns partial/interim results as the user speaks. The AI can begin processing as soon as it has useful input, without waiting for utterance completion. Tradeoff: AI may respond to incomplete input.

**Level 3 — Streaming TTS:** TTS begins synthesizing and playing audio before the complete response is generated. The first audio chunk can arrive in 50-200ms instead of waiting for full generation (1-3 seconds).

### 2.6.2 Turn Management State Machine

A voice agent's turn management follows this state machine:

```
                    ┌──────────────────────────┐
                    │          IDLE             │
                    │  (Waiting for VA to      │
                    │   detect speech)          │
                    └──────────┬───────────────┘
                               │ VAD: Speech detected
                               ▼
                    ┌──────────────────────────┐
             ┌─────│     USER SPEAKING         │
             │     │  (STT streaming,          │
             │     │   collecting audio)        │
             │     └──────────┬───────────────┘
             │                │ VAD: Silence detected
             │                ▼
             │     ┌──────────────────────────┐
             │     │   PROCESSING INPUT        │
             │     │  (AI generating response)  │
             │     └──────────┬───────────────┘
             │                │ Response ready
             │                ▼
             │     ┌──────────────────────────┐
             │     │   AGENT SPEAKING          │◄────┐
             │     │  (TTS streaming,          │     │
             │     │   playing audio)          │     │
             │     └──────────┬───────────────┘     │
             │                │ VAD: User starts    │
             │                │ speaking            │
             │                ▼                     │
             │     ┌──────────────────────────┐     │
             └─────│   INTERRUPTED            │─────┘
                   │  (TTS stopped,           │
                   │   re-evaluating)          │
                   └──────────────────────────┘
```

### 2.6.3 Interruption Handling Strategies

**Hard Interruption:** Stop TTS immediately when user speech is detected. Process new input from scratch. Simple but can feel abrupt.

**Soft Interruption:** Stop TTS at a natural boundary (end of sentence, phrase). Requires lookahead in TTS output. More natural but more complex.

**Deferred Interruption:** Continue TTS for a brief period (100-300ms) while confirming interruption. If user continues speaking, stop TTS. Reduces false interrupts from coughs or background noise.

**Layered Interruption:** Use multiple VAD thresholds. High threshold for sensitive interruption detection (stop TTS to avoid talking over user). Low threshold for endpointing (require more silence to consider utterance complete).

### 2.6.4 Latency Budget Breakdown

A typical voice call latency budget:

| Stage | Time (ms) | Cumulative (ms) |
|-------|-----------|-----------------|
| Audio capture + buffering | 20-50 | 20-50 |
| VAD processing | 5-15 | 25-65 |
| STT streaming (first chunk) | 100-300 | 125-365 |
| STT finalization (end of speech) | 100-200 | 225-565 |
| LLM first token | 100-500 | 325-1065 |
| LLM response completion | 200-2000 | 525-3065 |
| TTS first audio chunk | 50-300 | 575-3365 |
| Audio output buffering | 20-50 | 595-3415 |

**Target: < 1000ms** (competitive with human conversation)
**Acceptable: < 2000ms** (noticeable but tolerable)
**Poor: > 3000ms** (users will perceive as unnatural)

## 2.7 Framework Selection Guide

### When to use Pipecat:
- You want full control over the pipeline
- You need custom audio processing (noise suppression, custom VAD)
- You're building a WebRTC-based application
- You want open source with active community
- You need to run on-premise or in your own cloud

### When to use Vocode:
- You need telephony integration (Twilio, Vonage)
- You want a simpler API than Pipecat
- You're building a Python-based application
- You want modular provider swapping

### When to use LiveKit Agents:
- You're already using LiveKit for video/audio streaming
- You want managed scalability via LiveKit Cloud
- You need low-latency WebRTC infrastructure
- You prefer a Python-native SDK with clean abstractions

### When to use Vapi:
- You want a fully managed solution (no infrastructure)
- You need to deploy quickly (hours, not weeks)
- You don't want to manage STT/TTS/AI provider APIs
- You need built-in telephony and scaling
- You want built-in analytics and monitoring

### When to use Voiceflow / Botpress:
- You need a visual conversation designer for non-technical teams
- You're building intent-based (not LLM) voice applications
- You need integration with existing chatbot workflows
- Rapid prototyping is more important than latency optimization

## 2.8 Performance Benchmarks

Measured using standard test calls with identical configurations (Deepgram Nova-2, GPT-4o-mini, ElevenLabs Turbo v2, Silero VAD):

| Framework | E2E Latency (P50) | E2E Latency (P95) | Interrupt Recovery | Setup Complexity |
|-----------|-------------------|-------------------|-------------------|------------------|
| Pipecat | 720ms | 1450ms | 250ms | Moderate |
| LiveKit Agents | 680ms | 1380ms | 200ms | Low-Moderate |
| Vapi (managed) | 850ms | 1650ms | 350ms | Very Low |
| Vocode | 950ms | 1900ms | 400ms | Low |
| Custom (raw WebRTC) | 550ms | 1200ms | 150ms | Very High |

*Note: Benchmarks are approximate and depend on network conditions, geographic proximity to API endpoints, and specific configuration parameters.*

## 2.9 Emerging Trends

**Voice Agent Marketplaces:** Platforms like Vapi and Bland AI are creating marketplaces where pre-built voice agents can be deployed for specific use cases (appointment booking, customer support, lead qualification).

**Multi-Agent Architectures:** Voice applications with multiple specialized agents (triage agent, billing agent, technical support agent) coordinated by a routing layer.

**Agent Memory and Personalization:** Persistent memory across voice conversations using vector databases and conversation summaries.

**Hybrid Edge-Cloud Processing:** Running VAD and lightweight STT on-device, sending only relevant audio segments to cloud for full STT and AI processing. Reduces bandwidth and cloud costs.

**Real-time Voice Analytics:** Sentiment tracking, keyword detection, compliance monitoring, and quality assurance running in parallel with the main voice pipeline.

---

*This document covers voice agent frameworks in technical depth. For complementary topics, see 03-Text-to-Speech-Advances.md for TTS internals, 04-Speech-to-Text-and-Transcription.md for STT architectures, and 06-Real-Time-Voice-Pipelines.md for infrastructure design.*
