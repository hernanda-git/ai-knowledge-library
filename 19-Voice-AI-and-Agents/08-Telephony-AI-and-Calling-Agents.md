# 08 — Telephony AI and Calling Agents

## Overview

Telephony AI represents the convergence of voice AI agents with traditional telecommunications infrastructure. Unlike web-based voice agents that operate over data networks (WebRTC/WebSocket), telephony AI agents must interface with the Public Switched Telephone Network (PSTN), handle legacy protocols (SIP, ISDN, SS7), navigate telephony-specific regulations, and deliver reliable call handling at carrier-grade quality. This document provides a comprehensive technical reference for building, deploying, and operating AI agents that make and receive phone calls.

## Table of Contents

1. [Telephony Infrastructure Overview](#1-telephony-infrastructure-overview)
2. [SIP Trunking and PSTN Interconnection](#2-sip-trunking-and-pstn-interconnection)
3. [Twilio, Vonage, and Telephony APIs](#3-twilio-vonage-and-telephony-apis)
4. [Call Control and State Machines](#4-call-control-and-state-machines)
5. [DTMF and Keypad Interaction](#5-dtmf-and-keypad-interaction)
6. [Audio Codecs for Telephony](#6-audio-codecs-for-telephony)
7. [Call Routing and Number Management](#7-call-routing-and-number-management)
8. [SMS and MMS Integration](#8-sms-and-mms-integration)
9. [Outbound Calling Campaigns](#9-outbound-calling-campaigns)
10. [Compliance and Regulations](#10-compliance-and-regulations)
11. [STIR/SHAKEN and Call Authentication](#11-stirshaken-and-call-authentication)
12. [Scalable Call Handling](#12-scalable-call-handling)
13. [Call Analytics and Monitoring](#13-call-analytics-and-monitoring)
14. [Telephony AI Agent Frameworks](#14-telephony-ai-agent-frameworks)
15. [References and Further Reading](#15-references-and-further-reading)

---

## 1. Telephony Infrastructure Overview

### 1.1 The Telephony Stack

```
┌─────────────────────────────────────────────┐
│             AI Agent Layer                    │
│  (LLM, NLU, TTS, ASR, Conversation Logic)   │
├─────────────────────────────────────────────┤
│            Telephony Middleware               │
│  (Call Control, State Machines, Routing)     │
├─────────────────────────────────────────────┤
│         Telephony API / CPaaS Layer           │
│  (Twilio, Vonage, Plivo, SignalWire)         │
├─────────────────────────────────────────────┤
│              SIP Trunking                     │
│  (SIP Protocol, RTP Media Transport)         │
├─────────────────────────────────────────────┤
│           PSTN / Carrier Network              │
│  (SS7, ISDN, Fiber, Cellular)                │
├─────────────────────────────────────────────┤
│              End User Devices                 │
│  (Landline, Mobile, VoIP Phone)              │
└─────────────────────────────────────────────┘
```

### 1.2 How a Telephony AI Call Works

```
1. Incoming call arrives at carrier
2. Carrier routes to SIP trunk based on DID number
3. SIP INVITE arrives at telephony API provider
4. Provider triggers webhook to agent server
5. Agent answers call, establishes RTP audio stream
6. Bidirectional audio flows via provider's media servers
7. Agent processes audio (ASR → LLM → TTS)
8. Agent sends audio back through media stream
9. Agent hangs up or transfers call
10. Provider sends call completion webhook
```

### 1.3 Key Differences: WebRTC vs Telephony

| Aspect | WebRTC Voice | Telephony Voice |
|--------|-------------|----------------|
| Transport | UDP (ICE/DTLS) | SIP + RTP |
| Audio Codec | Opus (preferred) | G.711, G.722, G.729 |
| Latency | 50–200ms | 100–500ms |
| Setup | ICE/STUN/TURN | SIP INVITE/ACK |
| Number required | No | Yes (DID number) |
| Regulations | Minimal | Extensive (TCPA, FCC) |
| Reliability | Best-effort | Carrier-grade (99.999%) |
| Cost | Data-only | Per-minute charges |
| Caller ID | N/A | Required |
| Emergency calls | Not supported | 911/E911 required |
| Hardware | Computer/phone | Any phone device |

---

## 2. SIP Trunking and PSTN Interconnection

### 2.1 SIP Protocol Basics

SIP (Session Initiation Protocol) is the signaling protocol used to establish, modify, and terminate voice and video calls over IP networks.

**Key SIP Methods for AI Agents:**
- `INVITE` — Initiate a call
- `ACK` — Confirm session establishment
- `BYE` — Terminate a call
- `CANCEL` — Cancel a pending call
- `REGISTER` — Register with SIP server
- `OPTIONS` — Capability query
- `INFO` — DTMF digits, mid-call signaling
- `REFER` — Call transfer
- `NOTIFY` — Event notifications

### 2.2 SIP Message Flow for Basic Call

```
Caller                    SIP Proxy                  AI Agent Server
  │                          │                            │
  │── INVITE (SDP offer) ──▶│── INVITE (SDP offer) ────▶│
  │                          │                            │
  │                          │◀─ 100 Trying ────────────│
  │◀─ 100 Trying ──────────│                            │
  │                          │                            │
  │                          │◀─ 180 Ringing ───────────│
  │◀─ 180 Ringing ──────────│                            │
  │                          │                            │
  │                          │◀─ 200 OK (SDP answer) ───│
  │◀─ 200 OK (SDP answer) ──│                            │
  │                          │                            │
  │── ACK ─────────────────▶│── ACK ──────────────────▶│
  │                          │                            │
  │══════════ RTP Audio (bidirectional) ═══════════════│
  │                          │                            │
  │── BYE ─────────────────▶│── BYE ──────────────────▶│
  │                          │                            │
  │                          │◀─ 200 OK ────────────────│
  │◀─ 200 OK ───────────────│                            │
```

### 2.3 SIP Trunk Configuration

```yaml
sip_trunk_config:
  provider: "FlowRoute"  # or "Twilio Elastic SIP", "Bandwidth", "Telnyx"
  
  connection:
    protocol: "TLS"  # TLS for security, UDP for latency
    transport: "tcp"
    port: 5061
    
  credentials:
    username: "sip_username"
    password: "sip_password"
    realm: "sip.provider.com"
  
  codec_preferences:
    - "PCMU"     # G.711 μ-law (highest compat)
    - "PCMA"     # G.711 a-law
    - "G722"     # Wideband audio
    - "telephone-event"  # DTMF relay
  
  media:
    encryption: "SRTP"  # Secure RTP
    ice: false           # Not typical for SIP trunking
    rtcp_mux: true
  
  registration:
    enabled: true
    expiry_seconds: 3600
    retry_interval: 30
  
  call_limits:
    max_concurrent_calls: 50
    max_call_duration_minutes: 60
    max_call_rate_per_second: 10
  
  failover:
    primary: "sip-primary.provider.com"
    secondary: "sip-backup.provider.com"
    failover_threshold_ms: 5000
```

### 2.4 Media Server Architecture

For telephony AI, media servers handle the RTP audio stream and connect it to the AI processing pipeline.

```
RTP Audio → Media Server → Jitter Buffer → Audio Codec → PCM Audio → VAD → ASR → LLM → TTS → PCM Audio → Codec → RTP
```

**Popular Media Servers:**
- **FreeSWITCH**: Open-source, highly configurable, supports many codecs
- **Asterisk**: Oldest open-source PBX, extensive feature set
- **RTPEngine**: High-performance media proxy for WebRTC/SIP interop
- **Janus**: Lightweight WebRTC gateway
- **Kamailio**: SIP proxy server, can route to media servers

---

## 3. Twilio, Vonage, and Telephony APIs

### 3.1 Twilio Programmable Voice

Twilio is the most widely used telephony API platform for AI agents. It provides REST APIs and webhook-based call control.

**Key Twilio Concepts:**
- **TwiML** (Twilio Markup Language) — XML-based instructions for call flow
- **Webhooks** — HTTP callbacks for call events
- **Media Streams** — Real-time audio streaming via WebSocket
- **Functions** — Serverless call logic
- **Studio** — Visual call flow builder

### 3.2 TwiML for AI Agents

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <!-- Greeting message -->
  <Say voice="polly.Joanna-Neural">
    Hello! I'm an AI assistant calling from Acme Corp.
  </Say>
  
  <!-- Gather speech input (ASR integration) -->
  <Gather input="speech dtmf" 
          action="/process-input" 
          method="POST"
          speechTimeout="auto"
          speechModel="phone_call"
          enhanced="true"
          language="en-US"
          timeout="5"
          profanityFilter="true">
    <Say>How can I help you today?</Say>
  </Gather>
  
  <!-- Fallback if no input -->
  <Say>I didn't hear anything. Goodbye.</Say>
  <Hangup/>
</Response>
```

### 3.3 Media Streams for Real-Time AI

For advanced AI agents, Twilio Media Streams provides raw audio via WebSocket for custom processing.

```python
from fastapi import FastAPI, WebSocket
from twilio.rest import Client

app = FastAPI()

# Twilio webhook for incoming call
@app.post("/incoming-call")
async def incoming_call():
    """Return TwiML to start media stream."""
    from twilio.twiml.voice_response import VoiceResponse, Start, Stream
    response = VoiceResponse()
    start = Start()
    stream = Stream(url="wss://our-server.com/media-stream")
    start.append(stream)
    response.append(start)
    response.say("Connecting you to our AI assistant.")
    response.pause(length=1)
    return Response(content=str(response), media_type="application/xml")

# WebSocket for real-time audio
@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    await websocket.accept()
    
    async def handle_audio(audio_payload: dict):
        """Process incoming audio from Twilio."""
        if audio_payload["event"] == "media":
            # audio_payload["media"]["payload"] is base64-encoded audio
            audio_data = base64.b64decode(
                audio_payload["media"]["payload"]
            )
            # Process through ASR pipeline
            transcript = await asr.transcribe_chunk(audio_data)
            
            # If we have a response, send TTS audio back
            if transcript and agent_should_respond(transcript):
                response_audio = await tts.synthesize_chunk(
                    generate_response(transcript)
                )
                # Send audio back through WebSocket
                await websocket.send_json({
                    "event": "media",
                    "streamSid": audio_payload["streamSid"],
                    "media": {
                        "payload": base64.b64encode(
                            response_audio
                        ).decode("utf-8")
                    }
                })
    
    async for message in websocket.iter_json():
        if message["event"] == "connected":
            print("Media stream connected")
        elif message["event"] == "start":
            stream_sid = message["streamSid"]
            print(f"Stream started: {stream_sid}")
        elif message["event"] == "media":
            await handle_audio(message)
        elif message["event"] == "stop":
            print("Stream stopped")
```

### 3.4 Vonage (Nexmo) Voice API

Similar to Twilio but uses NCCO (Nexmo Call Control Objects) in JSON format.

```json
[
  {
    "action": "talk",
    "text": "Hello! I'm an AI assistant.",
    "voiceName": "Joanna",
    "level": 0,
    "bargeIn": true
  },
  {
    "action": "input",
    "type": ["speech", "dtmf"],
    "speech": {
      "uuid": ["speech-uuid"],
      "endOnSilence": 1.5,
      "language": "en-US",
      "context": ["support", "sales", "billing"]
    },
    "eventUrl": ["https://our-server.com/voice-input"],
    "timeout": 5
  }
]
```

### 3.5 CPaaS Provider Comparison

| Feature | Twilio | Vonage | Plivo | Telnyx | SignalWire |
|---------|--------|--------|-------|--------|-----------|
| PSTN Coverage | Global | Global | US/Global | US/Global | US/Global |
| SIP Trunking | Yes | Yes | Yes | Yes | Yes |
| Media Streams | Yes (WebSocket) | No | No | No | Yes |
| AI/ML Features | Yes (Twilio AI) | No | No | No | Yes |
| Serverless | Functions | Yes | No | No | Yes |
| Pricing | $$ | $$ | $ | $ | $$ |
| WebRTC | Yes | Yes | No | Yes | Yes |
| SMS | Yes | Yes | Yes | Yes | Yes |
| Number Porting | Yes | Yes | Yes | Yes | Yes |
| STIR/SHAKEN | Yes | Yes | Yes | Yes | Yes |

---

## 4. Call Control and State Machines

### 4.1 Call State Machine

```
                     ┌─────────────┐
                     │   IDLE      │
                     └──────┬──────┘
                            │
                     ┌──────▼──────┐
           ┌─────────│  RINGING    │──────────┐
           │         └──────┬──────┘          │
           │                │                  │
     ┌─────▼─────┐   ┌─────▼──────┐   ┌──────▼─────┐
     │  NO ANSWER │   │ ANSWERED   │   │  REJECTED  │
     └───────────┘   └─────┬──────┘   └────────────┘
                           │
                    ┌──────▼──────┐
                    │  IN_PROGRESS│
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  TRANSFERRING│
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  COMPLETED  │
                    └─────────────┘
```

### 4.2 Call Control Manager

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional

class CallState(Enum):
    IDLE = "idle"
    RINGING = "ringing"
    ANSWERED = "answered"
    IN_PROGRESS = "in_progress"
    TRANSFERRING = "transferring"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class CallSession:
    call_sid: str
    from_number: str
    to_number: str
    state: CallState
    start_time: float
    duration_seconds: float
    transcript: list
    sip_call_id: Optional[str] = None
    media_stream_sid: Optional[str] = None

class CallControlManager:
    def __init__(self):
        self.active_calls: Dict[str, CallSession] = {}
        self.call_history: List[CallSession] = []

    async def create_call(self, from_number: str,
                          to_number: str) -> CallSession:
        """Initiate a new outbound call."""
        call_sid = self._generate_call_sid()
        session = CallSession(
            call_sid=call_sid,
            from_number=from_number,
            to_number=to_number,
            state=CallState.RINGING,
            start_time=time.time(),
            duration_seconds=0,
            transcript=[]
        )
        self.active_calls[call_sid] = session
        return session

    async def answer_call(self, call_sid: str):
        """Answer an inbound call."""
        session = self.active_calls.get(call_sid)
        if session and session.state == CallState.RINGING:
            session.state = CallState.ANSWERED
            # Start media stream, begin conversation
            await self._initialize_agent_session(session)

    async def end_call(self, call_sid: str):
        """End an active call."""
        session = self.active_calls.get(call_sid)
        if session:
            session.state = CallState.COMPLETED
            session.duration_seconds = time.time() - session.start_time
            self.call_history.append(session)
            del self.active_calls[call_sid]
            # Cleanup agent resources
            await self._cleanup_agent_session(session)

    async def transfer_call(self, call_sid: str,
                            target_number: str):
        """Transfer call to another number."""
        session = self.active_calls.get(call_sid)
        if session:
            session.state = CallState.TRANSFERRING
            # Perform SIP REFER or Twilio <Dial>
            await self._execute_transfer(session, target_number)

    def get_active_call_count(self) -> int:
        return len(self.active_calls)

    def _generate_call_sid(self) -> str:
        return f"CA{uuid.uuid4().hex[:12].upper()}"
```

### 4.3 Call Event Webhooks

```yaml
twilio_webhook_endpoints:
  voice:
    incoming_call: "POST /voice/incoming"
    call_status: "POST /voice/call-status"
    gather_input: "POST /voice/gather"
    recording: "POST /voice/recording"
    transcription: "POST /voice/transcription"
    
  media_streams:
    websocket: "wss://server/voice/media-stream"
    
  events:
    - "initiated"
    - "ringing"
    - "answered"
    - "in-progress"
    - "completed"
    - "busy"
    - "no-answer"
    - "canceled"
    - "failed"
```

### 4.4 Call Status Webhook Handler

```python
@app.post("/voice/call-status")
async def handle_call_status(request: Request):
    """Handle call status callbacks from telephony provider."""
    data = await request.form()
    call_sid = data.get("CallSid")
    call_status = data.get("CallStatus")
    duration = data.get("CallDuration", 0)
    from_number = data.get("From")
    to_number = data.get("To")
    
    logger.info(f"Call {call_sid} status: {call_status}")
    
    if call_status == "ringing":
        # Call is ringing on the receiving end
        pass
    elif call_status == "in-progress":
        # Call was answered
        await call_manager.answer_call(call_sid)
    elif call_status == "completed":
        # Call ended normally
        await call_manager.end_call(call_sid)
        # Log call outcomes
        await log_call_outcome(call_sid, duration)
    elif call_status == "busy":
        # Called party was busy
        await call_manager.end_call(call_sid)
        await schedule_retry(call_sid)
    elif call_status == "no-answer":
        # No one answered
        await call_manager.end_call(call_sid)
        await schedule_retry(call_sid, delay_minutes=30)
    elif call_status == "failed":
        # Call failed due to network or carrier error
        await call_manager.end_call(call_sid)
        await alert_operations_team(call_sid, data)
    
    return {"status": "ok"}
```

---

## 5. DTMF and Keypad Interaction

### 5.1 DTMF in Telephony AI

DTMF (Dual-Tone Multi-Frequency) allows users to interact with AI agents using their phone keypad. This is essential for:
- Fallback when ASR fails
- Users who prefer not to speak
- Entering numeric data (account numbers, PINs)
- Menu navigation in IVR flows
- Accessibility for speech-impaired users

### 5.2 DTMF Tone Mapping

| Key | Low Frequency | High Frequency | Use |
|-----|-------------|---------------|-----|
| 1 | 697 Hz | 1209 Hz | Menu option 1 |
| 2 | 697 Hz | 1336 Hz | Menu option 2 |
| 3 | 697 Hz | 1477 Hz | Menu option 3 |
| 4 | 770 Hz | 1209 Hz | Menu option 4 |
| 5 | 770 Hz | 1336 Hz | Menu option 5 |
| 6 | 770 Hz | 1477 Hz | Menu option 6 |
| 7 | 852 Hz | 1209 Hz | Menu option 7 |
| 8 | 852 Hz | 1336 Hz | Menu option 8 |
| 9 | 852 Hz | 1477 Hz | Menu option 9 |
| 0 | 941 Hz | 1336 Hz | Menu option 0 |
| * | 941 Hz | 1209 Hz | Back/Cancel |
| # | 941 Hz | 1477 Hz | Confirm/Enter |

### 5.3 DTMF Collection Implementation

```python
class DTMFCollector:
    def __init__(self, min_digits=1, max_digits=20,
                 timeout_seconds=5, finish_on_key="#"):
        self.min_digits = min_digits
        self.max_digits = max_digits
        self.timeout = timeout_seconds
        self.finish_key = finish_on_key
        self.digits = []
        self.last_digit_time = 0

    def process_digit(self, digit: str) -> dict:
        """Process a DTMF digit and return collection status."""
        now = time.time()

        if digit == self.finish_key:
            return self._finish_collection()

        self.digits.append(digit)
        self.last_digit_time = now

        status = {
            "collected": "".join(self.digits),
            "digit_count": len(self.digits),
            "is_complete": len(self.digits) >= self.max_digits,
            "terminator": False
        }

        if len(self.digits) >= self.max_digits:
            status["is_complete"] = True

        return status

    def check_timeout(self) -> bool:
        """Check if DTMF input has timed out."""
        if not self.digits:
            return False
        return (time.time() - self.last_digit_time) > self.timeout

    def _finish_collection(self) -> dict:
        return {
            "collected": "".join(self.digits),
            "digit_count": len(self.digits),
            "is_complete": len(self.digits) >= self.min_digits,
            "terminator": True
        }

    def reset(self):
        self.digits = []
        self.last_digit_time = 0

    def get_prompt(self) -> str:
        """Generate prompt explaining expected DTMF input."""
        if self.min_digits == self.max_digits:
            return f"Please enter {self.min_digits} digits followed by the pound key."
        return f"Please enter your response followed by the pound key."


class DTMFMenu:
    def __init__(self, options: dict):
        """
        options: {
            "1": {"action": "check_balance", "label": "Check balance"},
            "2": {"action": "make_payment", "label": "Make a payment"},
            "3": {"action": "speak_to_agent", "label": "Speak to a human"},
        }
        """
        self.options = options
        self.collector = DTMFCollector(min_digits=1, max_digits=1)

    def get_menu_prompt(self) -> str:
        items = []
        for key, option in self.options.items():
            items.append(f"Press {key} to {option['label']}")
        return "Please select an option. " + " ".join(items)

    def process_selection(self, digit: str) -> Optional[str]:
        result = self.collector.process_digit(digit)
        if result["is_complete"]:
            digit = result["collected"]
            return self.options.get(digit, {}).get("action")
        return None
```

### 5.4 Mixed Speech + DTMF Interaction

Many users prefer to speak but may need to enter numeric data via DTMF for security (e.g., PIN entry). The agent should accept both.

```python
class HybridInputHandler:
    def __init__(self):
        self.input_mode = "speech"  # "speech" or "dtmf"
        self.dtmf_collector = DTMFCollector()
        self.speech_buffer = ""

    async def process_speech(self, transcript: str) -> dict:
        """Process speech input."""
        self.input_mode = "speech"
        self.speech_buffer = transcript
        return {"mode": "speech", "text": transcript}

    def process_dtmf(self, digit: str) -> dict:
        """Process DTMF input."""
        self.input_mode = "dtmf"
        result = self.dtmf_collector.process_digit(digit)
        return {"mode": "dtmf", "digits": result["collected"],
                "complete": result["is_complete"]}

    def get_secure_input_prompt(self) -> str:
        """Prompt for secure data entry (e.g., PIN)."""
        return ("For security, please enter your PIN using the keypad. "
                "Your input will not be spoken aloud.")
```

---

## 6. Audio Codecs for Telephony

### 6.1 Telephony Codec Requirements

Unlike WebRTC where Opus dominates, telephony must support a range of codecs for PSTN interoperability.

| Codec | Bitrate | Sample Rate | MOS Score | Latency | Bandwidth |
|-------|---------|-------------|-----------|---------|-----------|
| G.711 μ-law | 64 kbps | 8 kHz | 4.1 | 0.125ms | 87.2 kbps |
| G.711 a-law | 64 kbps | 8 kHz | 4.1 | 0.125ms | 87.2 kbps |
| G.722 | 64 kbps | 16 kHz | 4.5 | 2ms | 92.8 kbps |
| G.729 | 8 kbps | 8 kHz | 3.9 | 15ms | 31.2 kbps |
| G.726 | 16–40 kbps | 8 kHz | 3.8–4.1 | 0.125ms | 24–56 kbps |
| iLBC | 13.3/15.2 kbps | 8 kHz | 4.1 | 30ms | 24.5 kbps |
| AMR-NB | 4.75–12.2 kbps | 8 kHz | 3.5–4.2 | 25ms | 10–25 kbps |
| AMR-WB | 6.6–23.85 kbps | 16 kHz | 3.8–4.5 | 25ms | 15–40 kbps |
| Opus | 6–510 kbps | 8–48 kHz | 4.5+ | 5–60ms | 10–160 kbps |

### 6.2 Codec Negotiation in SIP

```python
SDP_OFFER = """
v=0
o=agent 2890844526 2890844526 IN IP4 media.agent.com
s=Voice Agent Session
c=IN IP4 203.0.113.1
t=0 0
m=audio 49170 RTP/AVP 0 8 9 101
a=rtpmap:0 PCMU/8000
a=rtpmap:8 PCMA/8000
a=rtpmap:9 G722/16000
a=rtpmap:101 telephone-event/8000
a=fmtp:101 0-16
"""

class CodecNegotiator:
    def __init__(self, preferred_codecs=None):
        self.preferred = preferred_codecs or ["PCMU", "PCMA", "G722"]

    def negotiate(self, local_codecs: list,
                  remote_codecs: list) -> str:
        """Negotiate best common codec between local and remote."""
        for codec in self.preferred:
            if codec in local_codecs and codec in remote_codecs:
                return codec
        # Fallback to most compatible
        if "PCMU" in remote_codecs:
            return "PCMU"
        if "PCMA" in remote_codecs:
            return "PCMA"
        return remote_codecs[0] if remote_codecs else "PCMU"

    def get_sdp_for_codec(self, codec: str, ip: str,
                          port: int) -> str:
        """Generate SDP for negotiated codec."""
        codec_map = {
            "PCMU": ("0", "PCMU/8000"),
            "PCMA": ("8", "PCMA/8000"),
            "G722": ("9", "G722/16000"),
        }
        pt, encoding = codec_map.get(codec, ("0", "PCMU/8000"))
        return (
            f"v=0\r\n"
            f"o=agent 2890844526 2890844526 IN IP4 {ip}\r\n"
            f"s=Voice Agent Session\r\n"
            f"c=IN IP4 {ip}\r\n"
            f"t=0 0\r\n"
            f"m=audio {port} RTP/AVP {pt} 101\r\n"
            f"a=rtpmap:{pt} {encoding}\r\n"
            f"a=rtpmap:101 telephone-event/8000\r\n"
            f"a=fmtp:101 0-16\r\n"
        )
```

### 6.3 Audio Processing for Telephony

Audio from PSTN has specific characteristics that affect ASR accuracy.

```python
class TelephonyAudioProcessor:
    def __init__(self, input_codec="PCMU"):
        self.input_codec = input_codec
        self.sample_rate = self._get_sample_rate(input_codec)
        self.denoiser = self._create_denoiser()

    def _get_sample_rate(self, codec: str) -> int:
        rates = {
            "PCMU": 8000, "PCMA": 8000,
            "G722": 16000, "G729": 8000
        }
        return rates.get(codec, 8000)

    def _create_denoiser(self):
        """Create noise suppressor optimized for telephony audio."""
        # Telephony audio has limited bandwidth (300-3400 Hz)
        # Use bandpass filter + noise gate
        return TelephonyDenoiser(
            low_cut_hz=300,
            high_cut_hz=3400,
            noise_gate_db=-45
        )

    def decode_rtp_payload(self, payload: bytes,
                           codec: str) -> np.ndarray:
        """Decode RTP audio payload to PCM samples."""
        if codec == "PCMU":
            return self._ulaw_to_pcm(payload)
        elif codec == "PCMA":
            return self._alaw_to_pcm(payload)
        # Add other codec conversions
        raise ValueError(f"Unsupported codec: {codec}")

    def _ulaw_to_pcm(self, ulaw_bytes: bytes) -> np.ndarray:
        """Convert μ-law encoded bytes to PCM float32."""
        # Standard μ-law expansion table
        ulaw_table = self._build_ulaw_table()
        pcm = np.array([ulaw_table[b] for b in ulaw_bytes],
                       dtype=np.float32)
        return pcm / 32768.0

    def _build_ulaw_table(self):
        """Build μ-law to PCM lookup table."""
        table = []
        for i in range(256):
            # μ-law decode algorithm
            mu = i ^ 0xff  # invert bits
            sign = (mu & 0x80) << 1
            exponent = (mu >> 4) & 0x07
            mantissa = mu & 0x0f
            sample = ((mantissa << 3) + 0x84) << (exponent + 2)
            sample = sample ^ sign
            table.append(sample)
        return table

    def upscale_to_16khz(self, audio_8khz: np.ndarray) -> np.ndarray:
        """Upscale 8kHz telephony audio to 16kHz for better ASR."""
        # Simple linear interpolation
        x_old = np.linspace(0, 1, len(audio_8khz))
        x_new = np.linspace(0, 1, len(audio_8khz) * 2)
        return np.interp(x_new, x_old, audio_8khz)
```

---

## 7. Call Routing and Number Management

### 7.1 DID Number Management

DID (Direct Inward Dialing) numbers are the phone numbers assigned to your agent.

```python
class NumberManager:
    def __init__(self, provider_api):
        self.provider = provider_api
        self.numbers = {}
        self.routing_rules = []

    async def provision_number(self, area_code: str,
                               features: list = None) -> dict:
        """Provision a new phone number."""
        features = features or ["voice", "sms", "mms"]
        available = await self.provider.available_phone_numbers(
            area_code=area_code,
            type="local",
            capabilities={"voice": True}
        )
        if not available:
            raise RuntimeError(f"No numbers available in {area_code}")

        number = available[0]
        result = await self.provider.buy_phone_number(
            number.phone_number,
            voice_url="https://server/voice/incoming",
            voice_method="POST",
            sms_url="https://server/sms/incoming",
            sms_method="POST",
            status_callback="https://server/voice/call-status",
        )
        self.numbers[result.phone_number] = result
        return result

    async def update_routing(self, number: str, webhook_url: str):
        """Update the webhook URL for a phone number."""
        await self.provider.update_phone_number(
            number,
            voice_url=webhook_url,
            voice_method="POST"
        )

    def add_routing_rule(self, rule: RoutingRule):
        """Add a routing rule for incoming calls."""
        self.routing_rules.append(rule)

    async def route_incoming_call(self, from_number: str,
                                   to_number: str) -> str:
        """Determine agent or flow for incoming call."""
        for rule in sorted(self.routing_rules, key=lambda r: -r.priority):
            if rule.matches(from_number, to_number):
                return rule.target_agent
        return "default_agent"

    async def release_number(self, number: str):
        """Release a phone number back to the provider."""
        if number in self.numbers:
            await self.provider.release_phone_number(number)
            del self.numbers[number]
```

### 7.2 Routing Rule Configuration

```yaml
routing_rules:
  - priority: 100
    name: "VIP Customer Routing"
    condition:
      from_number_in: ["+14155551234", "+14155555678"]
      to_number: "+18005551234"
    action:
      target_agent: "vip_support_agent"
      priority_queue: true
    
  - priority: 50
    name: "Hours-Based Routing"
    condition:
      time_of_day:
        start: "09:00"
        end: "17:00"
        timezone: "America/New_York"
      to_number: "+18005551234"
    action:
      target_agent: "business_hours_agent"
    
  - priority: 10
    name: "Fallback Routing"
    condition:
      to_number: "+18005551234"
    action:
      target_agent: "after_hours_agent"
      voicemail_allowed: true
```

---

## 8. SMS and MMS Integration

### 8.1 SMS for AI Agents

Integrating SMS with voice agents enables:
- Two-factor authentication and verification codes
- Appointment reminders and confirmations
- Post-call follow-up and feedback collection
- Document and link sharing during calls
- Customer support via text as alternative to voice

### 8.2 SMS Handler Implementation

```python
@app.post("/sms/incoming")
async def handle_incoming_sms(request: Request):
    """Handle incoming SMS messages."""
    data = await request.form()
    message_sid = data.get("MessageSid")
    from_number = data.get("From")
    to_number = data.get("To")
    body = data.get("Body", "")
    media_urls = data.getlist("MediaUrl0")  # MMS attachments

    logger.info(f"SMS from {from_number}: {body[:100]}")

    # Check for active call session
    active_call = call_manager.get_call_for_number(from_number)

    if active_call:
        # Contextual SMS during active call
        response = await handle_in_call_sms(active_call, body)
    else:
        # Standalone SMS conversation
        response = await agent.process_sms(from_number, body)

    # Send response
    twilio_client.messages.create(
        from_=to_number,
        to=from_number,
        body=response
    )

    return {"status": "ok"}

@app.post("/sms/outbound")
async def send_sms(to: str, body: str,
                   media_url: str = None):
    """Send an outbound SMS message."""
    message = twilio_client.messages.create(
        from_=our_number,
        to=to,
        body=body,
        media_url=[media_url] if media_url else None
    )
    logger.info(f"SMS sent to {to}: SID={message.sid}")
    return {"message_sid": message.sid}
```

### 8.3 SMS Templates for AI Agents

```yaml
sms_templates:
  appointment_reminder: |
    Hi {name}, this is a reminder about your appointment with {company}
    on {date} at {time}. Reply CONFIRM to confirm or RESCHEDULE to
    change your appointment.
  
  verification_code: |
    Your verification code is: {code}. This code expires in 5 minutes.
    Please do not share this code with anyone.
  
  post_call_feedback: |
    Thanks for calling {company}! How was your experience?
    Reply with a number 1-5 (5 = excellent).
  
  document_link: |
    As discussed, here's the document you requested:
    {link}
    Let us know if you have any questions.
  
  outbound_greeting: |
    Hi {name}, this is {agent_name} from {company}. I'm reaching out
    about {topic}. Reply STOP to opt out of messages.
```

---

## 9. Outbound Calling Campaigns

### 9.1 Campaign Architecture

```
Campaign Config → Contact List → Rate Limiter → Dialer → Call Handler → Outcome Tracking
                      ↑                                              ↓
                  Queue Manager                               Retry Scheduler
```

### 9.2 Outbound Campaign Manager

```python
class OutboundCampaign:
    def __init__(self, campaign_id: str, config: dict):
        self.campaign_id = campaign_id
        self.config = config
        self.contacts = []
        self.dialer = Dialer(
            max_concurrent=config.get("max_concurrent", 10),
            calls_per_second=config.get("calls_per_second", 1),
            calling_hours=config.get("calling_hours", {
                "start": "09:00",
                "end": "20:00",
                "timezone": "America/New_York"
            })
        )

    async def execute(self):
        """Execute the outbound calling campaign."""
        self.contacts = await self._load_contacts()

        for contact in self.contacts:
            # Check calling hours
            if not self.dialer.is_within_calling_hours():
                await self._queue_for_later(contact)
                continue

            # Rate limit check
            if not self.dialer.can_call():
                await self._queue_for_later(contact)
                continue

            # Initiate call
            call_result = await self.dialer.call(
                to=contact.phone,
                from_=self.config["caller_id"],
                script=self._personalize_script(contact)
            )

            # Track outcome
            await self._track_outcome(contact, call_result)

            # Handle retries
            if call_result.status in ["busy", "no-answer", "failed"]:
                if contact.retry_count < self.config.get("max_retries", 3):
                    await self._schedule_retry(contact,
                        delay_minutes=self.config.get("retry_delay", 30)
                    )

    def _personalize_script(self, contact: dict) -> str:
        """Personalize the agent script for this contact."""
        script = self.config["script_template"]
        return script.format(
            name=contact.get("name", ""),
            company=contact.get("company", ""),
            topic=contact.get("topic", ""),
            agent_name=self.config.get("agent_name", "Assistant")
        )


class Dialer:
    def __init__(self, max_concurrent=10, calls_per_second=1,
                 calling_hours=None):
        self.max_concurrent = max_concurrent
        self.calls_per_second = calls_per_second
        self.calling_hours = calling_hours
        self.active_calls = 0
        self.last_call_time = 0

    async def call(self, to: str, from_: str,
                   script: str) -> CallResult:
        """Initiate an outbound call."""
        while self.active_calls >= self.max_concurrent:
            await asyncio.sleep(0.1)

        # Rate limiting
        elapsed = time.time() - self.last_call_time
        min_interval = 1.0 / self.calls_per_second
        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)

        self.active_calls += 1
        self.last_call_time = time.time()

        try:
            # Use telephony API to place call
            call = await telephony_client.calls.create(
                url=f"https://server/voice/outbound?script={quote(script)}",
                to=to,
                from_=from_,
                status_callback="https://server/voice/call-status",
                status_callback_event=["completed", "busy", "no-answer",
                                       "failed", "answered"]
            )
            return CallResult(status="initiated", call_sid=call.sid)
        except Exception as e:
            return CallResult(status="failed", error=str(e))
        finally:
            self.active_calls -= 1
```

### 9.3 Compliance for Outbound Calls

Outbound calling is heavily regulated. Key requirements:

```yaml
outbound_compliance:
  # TCPA (Telephone Consumer Protection Act)
  tcpa:
    - "Prior express written consent required for autodialed calls"
    - "Must honor DNC (Do Not Call) registry"
    - "Calling hours: 8am-9pm in called party's timezone"
    - "Must provide opt-out mechanism during call"
    - "Maintain internal DNC list"
  
  # 10DLC (10-Digit Long Code) Requirements
  10dlc:
    - "Campaign registration required for A2P messaging"
    - "Brand registration with The Campaign Registry"
    - "Message volume tier assignment"
    - "Compliance with CTIA guidelines"
  
  # STIR/SHAKEN
  stir_shaken:
    - "Attestation level depends on caller ID verification"
    - "A-level: Full attestation (verified caller)"
    - "B-level: Partial attestation (verified with provider)"
    - "C-level: Gateway attestation (transit call)"
  
  # Call Recording
  recording:
    - "One-party consent states: 38 states + DC"
    - "Two-party consent states: CA, CT, FL, IL, MD, MA, MT, NH, NV, PA, WA"
    - "Must announce recording at start of call in two-party states"
  
  # Industry-Specific
  healthcare:
    - "HIPAA compliance for PHI in voice and SMS"
    - "Encrypted media transport required"
  
  finance:
    - "FINRA recordkeeping requirements"
    - "Must record all calls related to securities transactions"
```

---

## 10. Compliance and Regulations

### 10.1 TCPA Compliance Implementation

```python
class TCPACompliance:
    def __init__(self):
        self.national_dnc = DNCRegistry()
        self.internal_dnc = set()
        self.consent_records = {}
        self.calling_hours = CallingHoursChecker()

    async def check_before_call(self, number: str) -> bool:
        """Check all TCPA requirements before placing call."""
        # Check national DNC
        if await self.national_dnc.is_registered(number):
            logger.warning(f"Number {number} on National DNC")
            return False

        # Check internal DNC
        if number in self.internal_dnc:
            logger.warning(f"Number {number} on Internal DNC")
            return False

        # Check calling hours
        if not self.calling_hours.is_allowed(number):
            logger.warning(f"Calling outside hours for {number}")
            return False

        # Check consent (for marketing calls)
        if not self._has_consent(number):
            logger.warning(f"No consent for {number}")
            return False

        return True

    def record_consent(self, number: str, consent_type: str,
                       timestamp: float = None):
        """Record consent for a phone number."""
        self.consent_records[number] = {
            "type": consent_type,  # "written", "verbal", "electronic"
            "timestamp": timestamp or time.time(),
            "source": "call_recorded"
        }

    def add_to_dnc(self, number: str):
        """Add number to internal do-not-call list."""
        self.internal_dnc.add(number)
        logger.info(f"Added {number} to internal DNC")

    def opt_out_sms(self, number: str) -> str:
        """Handle opt-out SMS reply (e.g., 'STOP')."""
        self.add_to_dnc(number)
        return "You have been unsubscribed. No further messages will be sent."

    def _has_consent(self, number: str) -> bool:
        """Check if we have consent to call this number."""
        record = self.consent_records.get(number)
        if not record:
            return False
        # Consent expires after 18 months
        if time.time() - record["timestamp"] > 18 * 30 * 86400:
            return False
        return True


class CallingHoursChecker:
    def __init__(self):
        # Default TCPA calling hours: 8am-9pm local time
        self.start_hour = 8
        self.end_hour = 21

    def is_allowed(self, number: str) -> bool:
        """Check if calling is allowed right now for this number."""
        # Get timezone for area code
        tz = self._get_timezone_for_number(number)
        now = datetime.now(tz)
        return self.start_hour <= now.hour < self.end_hour

    def _get_timezone_for_number(self, number: str) -> timezone:
        """Map phone number to timezone using area code."""
        area_code = number[-10:-7]  # Extract NPA
        area_codes = {
            "212": "America/New_York",
            "310": "America/Los_Angeles",
            "312": "America/Chicago",
            # ... mapping for all NPAs
        }
        tz_name = area_codes.get(area_code, "America/New_York")
        return timezone(tz_name)
```

### 10.2 Regulatory Reporting

```yaml
regulatory_reports:
  monthly:
    - "Total outbound calls placed"
    - "Call answer rate"
    - "DNC list matches"
    - "Opt-out requests"
    - "Complaints received"
    
  quarterly:
    - "Consent audit"
    - "Recording compliance review"
    - "Data retention verification"
    - "Vendor compliance check"
    
  annual:
    - "Full TCPA compliance audit"
    - "STIR/SHAKEN implementation review"
    - "Privacy policy update"
    - "Third-party risk assessment"
```

---

## 11. STIR/SHAKEN and Call Authentication

### 11.1 Overview

STIR/SHAKEN (Secure Telephone Identity Revisited / Signature-based Handling of Asserted information using toKENs) is a framework to combat caller ID spoofing. It is mandatory for U.S. voice service providers.

### 11.2 Attestation Levels

| Level | Meaning | Typical Use |
|-------|---------|-------------|
| **A** (Full) | Provider verified caller identity and right to use number | Calls from provider's own customers with verified info |
| **B** (Partial) | Provider verified caller but not right to use number | Roaming customers, OTT providers |
| **C** (Gateway) | Provider received call from outside their network | International gateway calls |

### 11.3 STIR/SHAKEN Implementation

```python
class STIRSHAKENHandler:
    def __init__(self, private_key_path: str,
                 certificate_path: str,
                 service_provider_code: str):
        self.private_key = self._load_key(private_key_path)
        self.certificate = self._load_cert(certificate_path)
        self.spc = service_provider_code

    def sign_call(self, from_number: str,
                  to_number: str, call_id: str) -> str:
        """Sign an outbound call with STIR identity header."""
        identity_header = self._create_identity(
            from_number, to_number, call_id
        )
        # Attach to SIP INVITE
        return f"Identity: {identity_header}"

    def verify_call(self, identity_header: str,
                    from_number: str, call_id: str) -> dict:
        """Verify an incoming call's STIR identity."""
        try:
            result = self._verify_identity(
                identity_header, from_number, call_id
            )
            return {
                "verified": result["verified"],
                "attestation": result.get("attestation", "C"),
                "level": result.get("level", "C")
            }
        except Exception as e:
            return {"verified": False, "error": str(e)}

    def _create_identity(self, from_number: str,
                         to_number: str, call_id: str) -> str:
        """Create a STIR identity header (PASSporT)."""
        # PASSporT (Personal Assertion Token) JWT structure
        header = {
            "typ": "passport",
            "ppt": "shaken",
            "alg": "ES256"
        }
        payload = {
            "attest": self._get_attestation_level(from_number),
            "orig": {"tn": [from_number]},
            "dest": {"tn": [to_number]},
            "iat": int(time.time()),
            "origid": call_id
        }
        token = jwt.encode(payload, self.private_key,
                           algorithm="ES256",
                           headers=header)
        return f"{token}"

    def _get_attestation_level(self, number: str) -> str:
        """Determine attestation level for a number."""
        # A-level: If we provisioned this number
        # B-level: If number is verified but not ours
        # C-level: Default for gateway calls
        if self._is_our_number(number):
            return "A"
        if self._is_verified_partner(number):
            return "B"
        return "C"

    def _is_our_number(self, number: str) -> bool:
        """Check if this number belongs to our service."""
        return number in managed_numbers
```

---

## 12. Scalable Call Handling

### 12.1 Architecture for Scale

For handling thousands of concurrent calls, a horizontally scalable architecture is required.

```
                     ┌─────────────┐
                     │  Load       │
                     │  Balancer   │
                     └──────┬──────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
      ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
      │ Call       │  │ Call       │  │ Call       │
      │ Handler 1  │  │ Handler 2  │  │ Handler N  │
      └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
            │               │               │
            └───────────────┼───────────────┘
                            │
                    ┌───────▼───────┐
                    │  Message      │
                    │  Queue        │
                    │  (Redis/Rabbit)│
                    └───────┬───────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
      ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
      │ ASR        │  │ LLM        │  │ TTS        │
      │ Worker 1   │  │ Worker 1   │  │ Worker 1   │
      └───────────┘  └───────────┘  └───────────┘
```

### 12.2 Load Shedding and Backpressure

```python
class CallLoadShedder:
    def __init__(self, max_concurrent_calls=100,
                 cpu_threshold=80, memory_threshold=80):
        self.max_concurrent = max_concurrent_calls
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.active_calls = 0

    async def can_accept_call(self) -> bool:
        """Check if we can accept a new call."""
        if self.active_calls >= self.max_concurrent:
            logger.warning("Max concurrent calls reached, rejecting")
            return False

        cpu_usage = self._get_cpu_usage()
        if cpu_usage > self.cpu_threshold:
            logger.warning(f"CPU at {cpu_usage}%, rejecting call")
            return False

        memory_usage = self._get_memory_usage()
        if memory_usage > self.memory_threshold:
            logger.warning(f"Memory at {memory_usage}%, rejecting call")
            return False

        return True

    async def handle_incoming_call(self, call_data: dict) -> Response:
        """Handle incoming call with load shedding."""
        if not await self.can_accept_call():
            # Return busy signal or voicemail
            return TwiMLResponse(
                say="All agents are currently busy. Please try again later.",
                hangup=True
            )

        self.active_calls += 1
        try:
            return await self._process_call(call_data)
        finally:
            self.active_calls -= 1

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return psutil.cpu_percent(interval=0.1)

    def _get_memory_usage(self) -> float:
        """Get current memory usage percentage."""
        return psutil.virtual_memory().percent
```

### 12.3 Call Queue Management

```python
class CallQueue:
    def __init__(self, max_queue_size=50,
                 max_wait_time_seconds=300):
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self.max_wait_time = max_wait_time_seconds
        self.wait_times = {}

    async def enqueue(self, call_data: dict) -> bool:
        """Add a call to the queue."""
        try:
            call_id = call_data["CallSid"]
            self.wait_times[call_id] = time.time()

            await asyncio.wait_for(
                self.queue.put(call_data),
                timeout=5
            )
            return True
        except asyncio.TimeoutError:
            logger.error("Queue full, rejecting call")
            return False

    async def dequeue(self) -> Optional[dict]:
        """Get next call from queue."""
        try:
            call_data = await asyncio.wait_for(
                self.queue.get(),
                timeout=1
            )
            call_id = call_data["CallSid"]
            wait_time = time.time() - self.wait_times.pop(call_id, time.time())

            # Check if call has been waiting too long
            if wait_time > self.max_wait_time:
                logger.warning(f"Call {call_id} waited {wait_time}s, expiring")
                return None

            return call_data
        except asyncio.TimeoutError:
            return None
```

---

## 13. Call Analytics and Monitoring

### 13.1 Key Telephony Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Answer Rate | % of calls answered by human | > 60% |
| Abandon Rate | % of calls hung up before answer | < 5% |
| Average Handle Time | Average call duration | < 4 min |
| First Call Resolution | % resolved in one call | > 70% |
| Call Quality (MOS) | Mean Opinion Score | > 4.0 |
| Post-Call Survey Score | Customer satisfaction | > 4.2/5 |
| Drop Call Rate | % of calls dropped prematurely | < 0.5% |
| ASR Accuracy on Telephony | Word error rate on phone audio | < 15% |
| Concurrent Call Capacity | Max simultaneous calls | Varies |
| PDD (Post Dial Delay) | Time to ring after dialing | < 5 sec |

### 13.2 Call Detail Record (CDR)

```yaml
call_detail_record:
  call_sid: "CA1234567890abcdef"
  account_sid: "ACabcdef123456"
  from_number: "+14155551234"
  to_number: "+18005551234"
  direction: "inbound"  # inbound, outbound-api, outbound-dialer
  start_time: "2026-06-07T14:30:00Z"
  answer_time: "2026-06-07T14:30:12Z"
  end_time: "2026-06-07T14:33:45Z"
  duration_seconds: 225
  billable_minutes: 4
  status: "completed"
  
  media:
    codec: "PCMU"
    sample_rate: 8000
    recording_url: "https://api.twilio.com/2010-04-01/.../Recordings/RE..."
    recording_duration: 225
  
  quality:
    avg_mos: 4.2
    avg_jitter_ms: 12
    avg_packet_loss_pct: 0.3
    avg_rtt_ms: 85
  
  ai_agent:
    agent_version: "voice-support-v2.4"
    total_llm_tokens: 1245
    total_asr_audio_seconds: 95
    total_tts_audio_seconds: 130
    transcript_s3_path: "s3://calls/2026/06/07/CA1234.json"
    intent_detected: "check_balance"
    escalation_to_human: false
    user_satisfaction_score: 4
  
  routing:
    trunk_group: "primary-sip"
    caller_id_verified: true
    stir_shaken_attestation: "A"
```

### 13.3 Real-Time Monitoring Dashboard

```yaml
telephony_dashboard:
  title: "Telephony AI Agent — Live Dashboard"
  refresh_interval: 5s
  
  panels:
    - title: "Active Calls"
      metric: "active_calls"
      type: "gauge"
      thresholds:
        warning: 75
        critical: 90
    
    - title: "Call Rate"
      metric: "calls_per_minute"
      type: "time_series"
    
    - title: "Answer Rate"
      metric: "answer_rate_pct"
      type: "single_stat"
    
    - title: "Average Handle Time"
      metric: "avg_handle_time_seconds"
      type: "single_stat"
    
    - title: "Call Quality (MOS)"
      metric: "avg_mos"
      type: "time_series"
      thresholds:
        warning: 3.5
        critical: 3.0
    
    - title: "Error Rate"
      metrics: ["asr_errors", "tts_errors", "sip_errors"]
      type: "time_series"
    
    - title: "Agent Performance"
      metrics: ["avg_llm_latency_ms", "avg_asr_latency_ms", "avg_tts_latency_ms"]
      type: "time_series"
    
    - title: "Recent Calls"
      type: "table"
      columns: ["time", "from", "duration", "status", "intent", "quality"]
```

---

## 14. Telephony AI Agent Frameworks

### 14.1 Comprehensive Frameworks

**a) Vocode**
- Open-source voice agent framework with telephony support
- Built-in Twilio, Vonage, and SIP integration
- Modular ASR, LLM, TTS pipeline
- Conversation management and state machines
- Python-based, extensible

**b) Bot Framework (Microsoft)**
- Supports telephony channels via Direct Line
- Integration with Azure Communication Services
- Enterprise-grade compliance and security
- Multi-channel (voice + SMS + chat)

**c) Rasa + Telephony**
- Open-source NLU framework
- Telephony channel via Twilio or Rasa Channel
- Custom action server for call control
- Conversation-driven development

**d) Cognigy.AI**
- Enterprise contact center AI platform
- Native telephony integration
- Pre-built voice agent templates
- Agent assist and co-pilot modes

**e) Kore.ai**
- Contact center automation platform
- Voice bot development with telephony connectors
- IVR replacement and call deflection
- Analytics and optimization tools

### 14.2 Building a Telephony AI Agent from Scratch

```python
# Minimal telephony AI agent using Twilio
from fastapi import FastAPI, Request
from twilio.twiml.voice_response import VoiceResponse
import aiohttp

app = FastAPI()

class TelephonyAgent:
    def __init__(self, asr_endpoint, llm_endpoint, tts_endpoint):
        self.asr = asr_endpoint
        self.llm = llm_endpoint
        self.tts = tts_endpoint
        self.conversations = {}

    async def handle_incoming(self, call_sid: str,
                              from_number: str) -> VoiceResponse:
        """Handle an incoming call."""
        response = VoiceResponse()
        response.say("Hello! I'm your AI assistant. How can I help you today?",
                     voice="Polly.Joanna-Neural")
        response.gather(
            input="speech",
            action=f"/process-speech/{call_sid}",
            speech_timeout="auto",
            timeout=5
        )
        return response

    async def process_speech(self, call_sid: str,
                             speech_result: str) -> VoiceResponse:
        """Process speech and generate response."""
        response = VoiceResponse()

        # Get or create conversation
        if call_sid not in self.conversations:
            self.conversations[call_sid] = []

        self.conversations[call_sid].append({"role": "user",
                                               "content": speech_result})

        # Call LLM for response
        llm_response = await self._call_llm(
            self.conversations[call_sid]
        )

        self.conversations[call_sid].append({"role": "assistant",
                                               "content": llm_response})

        # Speak response and continue gathering
        response.say(llm_response, voice="Polly.Joanna-Neural")
        response.gather(
            input="speech",
            action=f"/process-speech/{call_sid}",
            speech_timeout="auto",
            timeout=5
        )

        return response

    async def _call_llm(self, conversation: list) -> str:
        """Call LLM for response generation."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.llm,
                json={"messages": conversation}
            ) as resp:
                data = await resp.json()
                return data["response"]

# Routes
@app.post("/voice/incoming")
async def incoming_call(request: Request):
    data = await request.form()
    call_sid = data.get("CallSid")
    from_number = data.get("From")
    twiml = await agent.handle_incoming(call_sid, from_number)
    return Response(content=str(twiml), media_type="application/xml")

@app.post("/voice/process-speech/{call_sid}")
async def process_speech(call_sid: str, request: Request):
    data = await request.form()
    speech_result = data.get("SpeechResult", "")
    twiml = await agent.process_speech(call_sid, speech_result)
    return Response(content=str(twiml), media_type="application/xml")
```

### 14.3 Framework Selection Guide

| Requirement | Recommended Framework |
|-------------|---------------------|
| Open-source, customizable | Vocode, Rasa + Telephony |
| Enterprise security | Microsoft Bot Framework |
| Contact center integration | Cognigy, Kore.ai |
| Low-code / no-code | Twilio Studio, Voiceflow |
| Real-time media streaming | SignalWire, custom (FastAPI + WebSocket) |
| High scale (10k+ concurrent) | Custom (FreeSWITCH + RTPEngine) |
| Multi-channel (voice+SMS+chat) | Twilio Flex, Vonage |

---

## 15. References and Further Reading

- Twilio Voice API Documentation — https://www.twilio.com/docs/voice
- Twilio Media Streams Guide — https://www.twilio.com/docs/media-streams
- Vonage Voice API — https://developer.vonage.com/voice/voice-api/overview
- Plivo Voice API — https://www.plivo.com/docs/voice/
- Telnyx SIP Trunking — https://telnyx.com/products/sip-trunking
- SignalWire — https://signalwire.com/
- FreeSWITCH Documentation — https://freeswitch.com/confluence/
- Asterisk Documentation — https://docs.asterisk.org/
- SIP Protocol (RFC 3261) — https://datatracker.ietf.org/doc/html/rfc3261
- RTP Protocol (RFC 3550) — https://datatracker.ietf.org/doc/html/rfc3550
- STIR/SHAKEN — RFC 8224, RFC 8225, RFC 8226
- TCPA Guidelines — https://www.fcc.gov/consumers/guides/telemarketing-calls
- 10DLC / A2P Registration — https://www.campaignregistry.com/
- Vocode — https://github.com/vocodedev/vocode
- "Building Telephony Applications with Twilio" — Twilio.org
- "SIP: Understanding the Session Initiation Protocol" — Alan B. Johnston
- "Voice over IP Fundamentals" — Davidson et al.
- "Real-Time Communication with WebRTC" — Salvatore Loreto
- "Designing Voice User Interfaces for Telephony" — Cathy Pearl
- WebRTC to SIP Interoperability Guide — https://webrtchacks.com/
