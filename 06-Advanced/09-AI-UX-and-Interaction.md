# AI User Experience and Interaction Design
## Table of Contents
1. [Introduction](#1-introduction)
2. [AI Interaction Patterns](#2-patterns)
3. [Conversation Design](#3-conversation)
4. [AI-Assisted Workflows](#4-workflows)
5. [Trust and Transparency](#5-trust)
6. [Error Handling and Feedback](#6-errors)
7. [UI Patterns for AI Products](#7-ui-patterns)
8. [Personalization and User Modeling](#8-personalization)
9. [Accessibility in AI UX](#9-accessibility)
10. [Evaluation of AI UX](#10-evaluation)
11. [AI API Design Patterns](#11-ai-api-design-patterns)
12. [AI UX Design Systems](#12-ai-ux-design-systems)
13. [Cross-References](#13-cross-references)
---

## 1. Introduction
AI UX design focuses on creating intuitive, effective interactions between humans and AI systems. Unlike traditional UI, AI interactions are probabilistic, generative, and unpredictable — requiring fundamentally different design patterns. The key challenge is managing expectation vs. reality: users bring assumptions from human-to-human interaction, but AI systems have different failure modes and limitations. Every design decision should build appropriate trust — not maximal trust.

---

## 2. AI Interaction Patterns

### 2.1 Command vs Conversation
| Aspect | Command | Conversation |
|--------|---------|:------------:|
| **Input** | Structured (click, type) | Natural language |
| **State** | Stateless | Stateful |
| **Feedback** | Immediate | Ongoing |
| **User expertise** | Learn commands | None needed |
| **Predictability** | High | Low |
| **Best for** | Repeated tasks | Complex/open-ended |

### 2.2 Proactive vs Reactive
- **Reactive:** AI responds to user input (most common — ChatGPT, Claude)
- **Proactive:** AI takes initiative (recommendations, suggestions, notifications)
- **Mixed:** AI reacts but proactively suggests next actions

### 2.3 Multimodal Interaction
- Text + voice + visual (GPT-4o, Gemini) · Screen sharing (Claude Computer Use)
- Code + natural language (Copilot, Cursor) · Image + prompt (DALL-E, Midjourney)

### 2.4 Human-in-the-Loop
```
AI Suggests → Human Reviews → Approves/Rejects → AI Learns → Loop
```
Critical for high-stakes decisions (medical, legal, financial) and safety.

### 2.5 Code Examples

#### Streaming Response Handler
Users should see output begin within 200–500ms. Streaming is critical for perceived responsiveness.

```python
import asyncio
from typing import AsyncGenerator, Callable

class StreamingHandler:
    def __init__(self, on_token: Callable[[str], None],
                 on_done: Callable[[str], None],
                 on_error: Callable[[Exception], None]):
        self.on_token, self.on_done, self.on_error = on_token, on_done, on_error
        self.accumulated = ""

    async def stream(self, generator: AsyncGenerator[str, None]) -> None:
        start = asyncio.get_event_loop().time()
        try:
            async for token in generator:
                self.accumulated += token
                self.on_token(token)
            self.on_done(self.accumulated)
        except Exception as e:
            self.on_error(e)
```

#### Multi-Turn Conversation Loop
```python
class ConversationState:
    def __init__(self, system_prompt="", max_tokens=128_000):
        self.messages = []
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        # Trim oldest messages when approaching context limit
        while sum(len(m["content"].split())*1.3 for m in self.messages) > self.max_tokens * 0.9:
            if len(self.messages) <= 1: break
            self.messages.pop(0)

    def to_api(self):
        msgs = self.messages.copy()
        if self.system_prompt:
            msgs.insert(0, {"role": "system", "content": self.system_prompt})
        return msgs

class ConversationLoop:
    """Multi-turn conversation with interruption support."""
    def __init__(self, model_fn, state=None):
        self.model_fn, self.state = model_fn, state or ConversationState()
        self.active = True

    async def turn(self, user_input: str) -> str:
        self.state.add("user", user_input)
        full = ""
        async for chunk in self.model_fn(self.state.to_api()):
            if not self.active: return "[Interrupted]"
            full += chunk
        self.state.add("assistant", full)
        return full

    def interrupt(self): self.active = False
    def reset(self): self.state = ConversationState()
```

---

## 3. Conversation Design

### 3.1 Key Principles
| Principle | Description | Example |
|-----------|-------------|---------|
| **Set expectations** | Tell users what the AI can do | "I help with: writing, coding, analysis" |
| **Clarify uncertainty** | Express confidence levels | "I'm not sure, let me check..." |
| **Show progress** | Indicate when the AI is working | Streamed responses, loading indicators |
| **Enable correction** | Easy to redirect | "Actually, I meant..." |
| **Remember context** | Maintain state across turns | Refer back to earlier conversation |

### 3.2 System Prompts as Interaction Design
Define the AI's interaction model: personality and tone, knowledge boundaries, response structure, guardrails.

### 3.3 Turn Design
- **Single-turn:** User asks, AI answers (search-like)
- **Multi-turn:** Sequential refinement (assistant-like)
- **Structured:** User completes fields (form-like)
- **Collaborative:** AI and user work together (partner-like)

### 3.4 Conversation Openers and Handoffs
- **Cold start:** New user → onboarding suggestions
- **Resumption:** Returning user → summarize previous state
- **Handoff:** AI → human agent → preserve context
- **Exit:** Graceful termination, offer to save or share

---

## 4. AI-Assisted Workflows

| Pattern | Description | Example |
|---------|-------------|---------|
| **Copilot** | AI works alongside, suggests | GitHub Copilot, Figma AI |
| **Agent** | AI works autonomously, user supervises | AutoGPT, Devin |
| **Review** | AI generates draft, user reviews | Cursor, Claude Code |
| **Assistant** | AI answers + executes commands | Siri, Alexa, ChatGPT |
| **Augment** | AI enhances human capability | Grammarly, Notion AI |
| **Tutor** | AI teaches and guides | Khanmigo |

### 4.1 Workflow State Machine
Design with clear state transitions and visible indicators:
```
Input → Understanding → Processing → Review → Output → Feedback
```

---

## 5. Trust and Transparency

### 5.1 Building Trust
- **Competence:** Demonstrate capability early
- **Reliability:** Consistent performance across sessions
- **Honesty:** Admit uncertainty and limitations
- **Controllability:** User can direct and override
- **Privacy:** Clear data handling policies

### 5.2 Transparency Requirements
| Information | Why It Matters |
|-------------|----------------|
| **What model am I talking to?** | Sets capability expectations |
| **Is this AI-generated?** | Regulatory (EU AI Act) |
| **What data is used?** | Privacy concerns |
| **Why did the AI say that?** | Trust and debugging |
| **How confident is the AI?** | Appropriate reliance |

### 5.3 Communicating Uncertainty
| Pattern | Implementation | Example UI |
|---------|---------------|------------|
| **Confidence bars** | Visual indicator per claim | `[█████░░░] 60%` |
| **Per-segment scoring** | Each segment scored | Green/orange/red highlights |
| **Citation badges** | Numbered source references | `[1][2]` inline, expandable |
| **Source footnotes** | Reference list with quotes | Collapsible "View Sources" |
| **Alternative views** | Multiple options | Tabbed: "Option A / B" |
| **Provenance icons** | Model vs. retrieved vs. user | 🧠/📄/👤 |

#### Example: Confidence-Annotated Response
```
Q3 revenue was $2.1M [███████░░░ 70%] with 12% growth [████████░░ 80%,
sourced from [1] Q3 Financial Report].
```

### 5.4 Calibrating Trust
- **Too much hedging:** Warnings ignored → reduce frequency
- **Too little hedging:** Over-reliance on wrong answers → increase for low-confidence domains
- **Domain-sensitive:** More uncertainty for recent events, less for established facts

---

## 6. Error Handling and Feedback

### 6.1 Error Types
- **Hallucination:** AI confidently says something wrong
- **Misunderstanding:** AI interprets user incorrectly
- **Refusal:** AI refuses a legitimate request
- **Incomplete:** Answer doesn't fully address the question
- **Off-topic:** AI goes in unintended direction

### 6.2 Error Recovery Patterns
- **Clarify:** "Did you mean X or Y?"
- **Correct:** "I was wrong. The correct answer is..."
- **Escalate:** "This is beyond my capabilities."
- **Fallback:** "Here's a rough approximation..."
- **Teach:** "I understand now. Next time I'll know to..."

### 6.3 Feedback Collection
- **Implicit:** Engagement time, follow-up rate, task completion
- **Explicit:** Thumbs up/down, ratings, comments
- **Comparative:** "Which response is better?"

### 6.4 Graceful Degradation
1. **Inform** the user about the issue
2. **Offer alternatives:** cached responses, simpler model
3. **Preserve context:** save state for retry
4. **Notify on recovery**

---

## 7. UI Patterns for AI Products

### 7.1 In-Chat Tool Calls
Show tool invocations inline (collapsed by default, expandable):
```
User: "Weather in Tokyo?"
AI:   Let me check... 🔍 [Tool: weather_api()] → 22°C, partly cloudy
```
Design: loading states per tool, anonymize sensitive params, "Retry" on failure.

### 7.2 Inline Editing
Let users directly edit AI output:
- **Diff view:** Side-by-side or tracked-changes overlay
- **Version history:** Undo/redo and revision comparison
- **Selection editing:** Highlight a phrase, request specific changes
- **Accept/reject:** IDE-style diff markers for code

### 7.3 Split-View and Comparison
```
┌───────────┬──────────────────┐
│ Original  │ AI Generated      │
│ Write a   │ Draft an email... │
│ formal... │ [Accept] [Edit]   │
└───────────┴──────────────────┘
```
Variations: side-by-side (translation), stacked (code), slider overlay (visuals), multi-version (2–3 alternatives).

### 7.4 Progressive Disclosure
Reveal complexity gradually:
- Novices see 3–5 controls; experts access 15+
- Defaults work for 80% of use cases
- Advanced settings one click away, not zero
- Tooltips and inline help for each control

```
[Basic] [Advanced]
Prompt: [_______________] [Generate]
[Show Options ▾] Temperature: [──●──] 0.7
```

### 7.5 Stepped Wizards
For complex multi-step tasks:
```
Step 1/4: Goal → Step 2/4: Audience → Step 3/4: Tone → Step 4/4: Review & Generate
```
Users navigate back without losing state; each step previews next requirements.

---

## 7a. Agent Interaction UX Patterns

### 7a.1 Thinking and Reflection States

AI agents increasingly display their reasoning process. The UX challenge is showing enough to build trust without overwhelming users:

| State | What to Show | UX Pattern |
|-------|-------------|------------|
| **Thinking** | "Analyzing your request...", animated dots | Minimal: compact pill or inline badge |
| **Planning** | Structured plan with steps | Expandable: collapsed by default, shows step count |
| **Tool calling** | Which tool, what arguments, result | Collapsible accordion: `🔧 Searching web...` |
| **Self-correction** | "I made an error, retrying..." | Highlighted with diff icon: `🔄 Retrying with different approach` |
| **Reflection** | Summarizing what was done | Checkmark + brief summary: `✅ Completed 3 tasks` |

```python
# Agent state machine for UX
from enum import Enum
from dataclasses import dataclass, field

class AgentState(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    PLANNING = "planning"
    EXECUTING = "executing"
    TOOL_CALLING = "tool_calling"
    SELF_CORRECTING = "self_correcting"
    REFLECTING = "reflecting"
    DONE = "done"
    ERROR = "error"

@dataclass
class AgentUXEvent:
    state: AgentState
    detail: str = ""
    tool_name: str = ""
    duration_ms: float = 0.0
    progress: float = 0.0  # 0.0 to 1.0
    steps_completed: int = 0
    steps_total: int = 0

    def to_ui_payload(self) -> dict:
        icons = {AgentState.THINKING: "💭", AgentState.PLANNING: "📋",
                 AgentState.TOOL_CALLING: "🔧", AgentState.SELF_CORRECTING: "🔄",
                 AgentState.REFLECTING: "✅", AgentState.ERROR: "❌"}
        return {"icon": icons.get(self.state, "🤖"),
                "state": self.state.value,
                "detail": self.detail,
                "progress": self.progress,
                "steps": f"{self.steps_completed}/{self.steps_total}"}
```

**Design guidelines:**
- Show thinking state within 200ms of user input submission
- Collapse intermediate tool calls by default (avoid information overload)
- Animate state transitions smoothly (don't flash between states)
- Keep a running timeline: users should be able to see "what the agent did"

### 7a.2 Tool Call Visualization

When an agent calls tools, users need to understand what's happening without being distracted:

```
User: "What's the weather in Tokyo and should I bring an umbrella?"

Agent:
💭 Analyzing request...
📋 Plan: (1) Get Tokyo weather → (2) Analyze precipitation → (3) Recommend
🔧 [tool: get_weather] Tokyo → 22°C, 60% chance of rain [200ms]
🔧 [tool: search_web] "Tokyo March weather patterns" → Retrieved 3 sources [1.2s]
✅ Results synthesized:
   Tokyo is 22°C with 60% rain probability. Yes, bring an umbrella!
   ☔ Recommended: Compact travel umbrella
```

**Key UX parameters for tool calls:**
| Parameter | Recommendation | Rationale |
|-----------|---------------|-----------|
| Auto-expand | First 2 tool calls only | Establishes the pattern; user knows what to expect |
| Duration display | Always show | Builds trust; users can estimate remaining time |
| Error display | Inline, with retry button | Don't lose context; one-click recovery |
| Sensitive params | Masked (e.g., `password=****`) | Security without hiding the call happened |
| Collapse after N | ≥3 concurrent calls hidden | Prevents scroll-bombing |

### 7a.3 Self-Correction UX

When an agent detects a mistake mid-task, the UX should communicate clearly:

```python
class SelfCorrectionDisplay:
    """Models how to show agent self-correction to users."""
    
    @staticmethod
    def format_correction(original: str, fixed: str, reason: str) -> str:
        return (
            f"🔄 **Self-correction**\n"
            f"- **Issue:** {reason}\n"
            f"- **Before:** `{original}`\n"
            f"- **After:** `{fixed}`\n"
        )
    
    @staticmethod
    def format_retry_attempt(attempt: int, max_retries: int, error: str) -> str:
        return (
            f"🔄 **Retry {attempt}/{max_retries}**\n"
            f"- **Error:** {error}\n" +
            ("- **Last attempt** 😔" if attempt == max_retries else "- **Retrying...**")
        )
```

**Design rules:**
- Always show the correction, not just the fix (builds trust via transparency)
- Limit retries to 3 before escalating to user
- Log corrections as part of session analytics
- Offer an "undo" option if the correction changes the output direction

### 7a.4 Multi-Agent Coordination UX

When multiple agents collaborate, show a **team overview**:

```
┌──────────────────────────────────────────────┐
│ 👥 Agent Squad: Building expense tracker app │
├──────────────────────────────────────────────┤
│ 🟢 Coder       │ Writing API routes...      │
│ 🟡 Reviewer    │ Awaiting code...           │
│ 🔵 Researcher  │ ✅ Found best auth library │
│ 🟣 Tester      │ Writing integration tests  │
└──────────────────────────────────────────────┘
```

- Color-coded status per agent (idle=gray, active=green, blocked=yellow, error=red)
- Click any agent to see its individual state and output
- Show dependency arrows: "Coder blocked on Researcher"
- Overall progress bar: "Step 4/8: Implementing backend"

---

## 8. Personalization and User Modeling

### 8.1 Adapting to User Style
| Dimension | Adaptable | Signal |
|-----------|-----------|--------|
| **Verbosity** | Concise ↔ detailed | "Can you shorten that?" |
| **Technical depth** | Simple ↔ expert | Follow-up questions |
| **Formality** | Casual ↔ professional | User's own language |
| **Format** | Lists ↔ prose ↔ tables | Copy behavior |
| **Language** | Match user's language | Input detection |

### 8.2 Memory Types
| Type | Scope | Duration | Example |
|------|-------|----------|---------|
| **Working** | Current conversation | Session | What was just discussed |
| **Episodic** | Past conversations | Cross-session | "Last time we talked about X" |
| **Semantic** | User facts | Persistent | "User prefers Python over Java" |
| **Procedural** | Interaction patterns | Persistent | "User always checks math" |

Memory UX: viewable/editable "Memory" section, explicit commands ("Remember that...", "Forget that"), surface memories when relevant.

### 8.3 Preference Learning (Snippet)
```python
class PreferenceModel:
    def __init__(self):
        self.prefs = {"verbosity": 0.5, "formality": 0.5, "technical_depth": 0.5}

    def record(self, signal: str, value: float, confidence: float = 0.3):
        key = {"request_shorter": "verbosity", "request_longer": "verbosity",
               "asked_simplification": "technical_depth"}.get(signal)
        if key:
            lr = confidence * 0.1
            self.prefs[key] = self.prefs[key] * (1 - lr) + value * lr

    def apply(self, prompt: str) -> str:
        dirs = []
        if self.prefs["verbosity"] < 0.3: dirs.append("Be extremely concise.")
        elif self.prefs["verbosity"] > 0.7: dirs.append("Be thorough.")
        if self.prefs["technical_depth"] > 0.7: dirs.append("Use technical terminology.")
        return prompt + "\n" + "; ".join(dirs) if dirs else prompt
```

### 8.4 Privacy
- On-device storage when possible · Opt-in memory · Transparency dashboard
- Easy deletion ("Clear my memory") · Data minimization

---

## 9. Accessibility in AI UX

### 9.1 Voice Input/Output
- **VAD:** Auto-detect when user stops speaking
- **States:** "Listening..." / "Processing..." / "Speaking..."
- **Push-to-talk + always-listening** options with mic indicator
- **Barge-in:** Allow interrupting the AI mid-speech
- **Multimodal fallback:** Voice → text for quiet environments

### 9.2 Screen Reader Compatibility
| Challenge | Solution |
|-----------|----------|
| Streaming text | `aria-live="polite"` for incremental updates |
| Tool calls | Announce: "Searching the web..." |
| Code blocks | `<pre><code>` with language labels |
| Tables | Semantic markup, not ASCII art |
| AI images | Always include alt text |
| Notifications | `role="alert"` for urgent messages |

```html
<div id="response" aria-live="polite" aria-atomic="false" role="log"></div>
```

### 9.3 Simplified Interfaces
For cognitive disabilities / low literacy:
- **Guided prompts:** Suggestion chips instead of free text
- **Reduced options:** Max 2–3 choices per decision point
- **Confirmation steps:** "You asked me to send an email. Proceed?"
- **Plain language:** Avoid AI jargon ("model", "token", "hallucination")
- **Large targets:** 44×44px minimum touch targets

### 9.4 Cognitive Load Management
- Scaffold tasks ("Would you like help with writing, research, or analysis?")
- Progress indicators ("Step 2 of 4: Providing examples...")
- Save drafts · Pre-fill common patterns · Undo everywhere

### 9.5 Internationalization
- RTL language support (Arabic, Hebrew) in chat layout
- CJK character width handling in streaming display
- Localized date/number/currency formatting
- Language detection + auto UI switching

---

## 10. Evaluation of AI UX

| Metric | What It Measures | How |
|--------|-----------------|-----|
| **Task Success Rate** | Did user accomplish goal? | Track completion |
| **Time on Task** | How fast? | Log timestamps |
| **Error Rate** | How often does AI fail? | Failure tracking |
| **User Satisfaction** | How did it feel? | Surveys (CSAT, NPS) |
| **Conversation Length** | Efficiency vs depth | Turns per session |
| **Trust Score** | Does user trust the AI? | Survey questions |
| **Learning Curve** | Time to first success | New user onboarding speed |

### 10.1 Metrics Instrumentation
```python
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

@dataclass
class UXEvent:
    event_type: str  # "message", "stream_end", "tool_call", "thumbs_up", "error"
    user_id: str
    session_id: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)
    duration_ms: float = None

    def to_dict(self):
        return {**asdict(self), "iso_time": datetime.fromtimestamp(
            self.timestamp, tz=timezone.utc).isoformat()}

class AnalyticsPipeline:
    """Buffered analytics collection with session-level aggregation."""
    def __init__(self, buffer_size=100):
        self.buffer, self.buffer_size = [], buffer_size
        self.sessions = {}

    def track(self, event: UXEvent):
        self.buffer.append(event)
        if len(self.buffer) >= self.buffer_size: self.flush()

    def flush(self):
        if not self.buffer: return
        # In production: http_post("/analytics/batch", json=batch)
        print(f"[Analytics] Flushed {len(self.buffer)} events")
        self.buffer.clear()

    def session_start(self, uid, sid, meta=None):
        self.track(UXEvent("session_start", uid, sid, metadata=meta or {}))
        self.sessions[sid] = {"start": time.time(), "turns": 0}

    def track_turn(self, uid, sid):
        s = self.sessions.setdefault(sid, {"start": time.time(), "turns": 0})
        s["turns"] += 1
        self.track(UXEvent("message", uid, sid, metadata={"turn": s["turns"]}))

    def summary(self, sid):
        s = self.sessions.get(sid)
        if not s: return None
        events = [e for e in self.buffer if e.session_id == sid]
        return {"session": sid, "duration": time.time()-s["start"],
                "turns": s["turns"],
                "errors": sum(1 for e in events if e.event_type=="error"),
                "pos": sum(1 for e in events if e.event_type=="thumbs_up"),
                "neg": sum(1 for e in events if e.event_type=="thumbs_down")}

# Usage:
# pipe = AnalyticsPipeline()
# sid = pipe.session_start("user_42", "sess_1")
# pipe.track_turn("user_42", sid)
# pipe.track(UXEvent("thumbs_down", "user_42", sid,
#                    metadata={"reason": "hallucination"}))
# print(pipe.summary(sid))
```

### 10.2 A/B Testing
| Metric | Min Sample | Notes |
|--------|-----------|-------|
| Task success rate | 500/variant | Non-deterministic outputs |
| Avg turns | 300/variant | Efficiency change |
| CSAT | 200/variant | Survey-based |
| Error rate | 1000/variant | Rare events need more data |

AI responses are non-deterministic — use chi-square for categorical metrics, t-test for continuous. Run ≥1 business cycle.

### 10.3 Qualitative Methods
- **Cognitive walkthroughs:** Experts simulate tasks, identify friction
- **Heuristic evaluation:** Score against AI-specific heuristics
- **Diary studies:** Users log experiences over 1–2 weeks
- **Error taxonomy:** Categorize user errors to find systematic issues

## 11. AI API Design Patterns

Designing APIs for AI products requires patterns that account for non-determinism, streaming, latency, and cost management.

### 11.1 Streaming Response API

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
from typing import AsyncGenerator

app = FastAPI(title="AI API")

class ChatRequest(BaseModel):
    messages: list[dict]
    stream: bool = True
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 2048

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest):
    if not request.stream:
        # Non-streaming: return complete response
        result = await complete(request.messages, request.model)
        return {"choices": [{"message": {"content": result}}]}
    
    # Streaming: return SSE response
    return StreamingResponse(
        stream_response(request.messages, request.model),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

async def stream_response(messages: list[dict],
                          model: str) -> AsyncGenerator[str, None]:
    """Server-Sent Events streaming with progress metadata."""
    yield 'data: {"choices":[{"delta":{"role":"assistant"},"index":0}]}\n\n'
    async for chunk in call_llm_stream(messages, model):
        yield f'data: {chunk}\n\n'
    yield 'data: [DONE]\n\n'

async def call_llm_stream(messages: list[dict],
                          model: str) -> AsyncGenerator[str, None]:
    """Simulated LLM streaming — replace with real API call."""
    responses = {
        "gpt-4o": "I'll help you with that. First, let me analyze your request...",
        "claude-4": "Let me think about this step by step..."
    }
    text = responses.get(model, "Processing your request...")
    for word in text.split():
        yield f'{{"choices":[{{"delta":{{"content":"{word} "}},"index":0}}]}}'
        await asyncio.sleep(0.05)

async def complete(messages: list[dict], model: str) -> str:
    """Non-streaming completion."""
    responses = {"gpt-4o": "Here is the complete response.", "claude-4": "Step-by-step analysis..."}
    return responses.get(model, "Response.")
```

### 11.2 Cost-Managed Proxy

```python
from fastapi import FastAPI, Header, Depends, HTTPException
from pydantic import BaseModel
import time

app = FastAPI()

class BudgetTracker:
    def __init__(self, monthly_limit: float = 1000.0):
        self.monthly = monthly_limit
        self.spent = 0.0
    
    def check(self, user_id: str, estimated_cost: float):
        """Check budget before allowing the API call."""
        if self.spent + estimated_cost > self.monthly:
            raise HTTPException(429, detail=f"Budget exhausted (${self.monthly:.2f}/month)")
        if estimated_cost > self.monthly * 0.2:
            raise HTTPException(402, detail=f"Single call ${estimated_cost:.4f} exceeds 20% of monthly budget")

tracker = BudgetTracker()

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 500

@app.post("/v1/generate-budgeted")
async def generate_budgeted(req: PromptRequest,
                            x_user_id: str = Header(default="anonymous")):
    cost = estimate_cost(req.prompt, req.max_tokens)
    tracker.check(x_user_id, cost)
    start = time.time()
    result = await call_llm(req.prompt)
    elapsed = time.time() - start
    tracker.spent += cost
    return {"result": result, "cost_usd": cost, "latency_ms": elapsed*1000}

def estimate_cost(prompt: str, max_tokens: int) -> float:
    # GPT-4o pricing: $2.50/1M input, $10/1M output
    input_cost = (len(prompt) / 1000000) * 2.50
    output_cost = (max_tokens / 1000000) * 10.0
    return input_cost + output_cost

async def call_llm(prompt: str) -> str:
    return f"Generated response for: {prompt[:50]}..."
```

### 11.3 API Design Comparison

| Pattern | When to Use | Benefits | Trade-offs |
|---------|-------------|----------|------------|
| **Synchronous** | Short responses (<5s), simple Q&A | Simple client, easy caching | Blocked until complete, bad for UX |
| **Server-Sent Events** | Streaming output, chat | Native browser support, auto-reconnect | One-way only, text-only |
| **WebSocket** | Bidirectional, real-time | Full duplex, low latency | Complex scaling, no HTTP caching |
| **Chunked Transfer** | Large file generation | Simple HTTP, progressive loading | No metadata per chunk |
| **Async Request-Reply** | Long-running (>30s) | Reliable, survives restarts | Needs polling/callbacks |

### 11.4 Error Codes for AI APIs

| Status | Error | When | Recovery |
|:------:|-------|------|----------|
| 400 | **Content filter** | Output blocked by safety | Rephrase prompt with different wording |
| 402 | **Payment required** | Budget exceeded or paywall | Add funds or reduce request size |
| 429 | **Rate limited** | Too many requests | Exponential backoff, retry-after header |
| 500 | **Model unavailable** | Server overload or failure | Retry with fallback model |
| 503 | **Context overflow** | Prompt exceeds context window | Truncate or summarize context |

### 11.5 Rate Limiting with Token Budgets

```python
import time
from collections import defaultdict

class TokenBucketRateLimiter:
    """Token bucket rate limiter for AI API costs."""
    
    def __init__(self, tokens_per_minute: int = 100_000):
        self.capacity = tokens_per_minute
        self.tokens = defaultdict(lambda: tokens_per_minute)
        self.last_refill = defaultdict(time.time)
    
    def consume(self, user_id: str, tokens: int) -> bool:
        now = time.time()
        elapsed = now - self.last_refill[user_id]
        self.tokens[user_id] = min(
            self.capacity,
            self.tokens[user_id] + elapsed * (self.capacity / 60)
        )
        self.last_refill[user_id] = now
        
        if self.tokens[user_id] >= tokens:
            self.tokens[user_id] -= tokens
            return True
        return False

# Usage
limiter = TokenBucketRateLimiter(tokens_per_minute=100_000)
if limiter.consume("user_42", 500):  # 500 tokens for this request
    print("✅ Request allowed")
else:
    print("❌ Rate limited — try again in 300ms")
```

---

## 12. AI UX Design Systems and Component Libraries

As AI products mature, teams need standardized design systems that account for AI-specific UI patterns — streaming text, tool calls, confidence indicators, and agent state displays. A well-crafted design system ensures consistency across products while accelerating development.

### 12.1 Design Tokens for AI Interactions

Design tokens capture low-level design decisions (colors, spacing, timing) as named values. AI UX introduces new token categories:

| Token Category | Token Name | Example Value | Usage |
|---------------|-----------|:-------------:|-------|
| **Typography** | `--font-chat-user` | `system-ui, sans-serif` | User message text |
| | `--font-chat-assistant` | `system-ui, sans-serif` | AI response text |
| | `--font-code-block` | `'Fira Code', monospace` | Code snippet display |
| | `--font-size-streaming` | `1rem` | Base text during streaming |
| **Spacing** | `--space-chat-gap` | `16px` | Between conversation turns |
| | `--space-tool-indent` | `24px` | Nested tool call details |
| | `--space-avatar-gap` | `8px` | Between avatar and message |
| **Timing** | `--dur-think-display` | `200ms` | Delay before showing thinking indicator |
| | `--dur-stream-min` | `50ms` | Minimum chunk display interval |
| | `--dur-type-indicator` | `300ms` | Typing indicator animation cycle |
| | `--dur-transition-state` | `400ms` | Agent state transition animation |
| **Color** | `--clr-thinking` | `#6B7280` (gray-500) | Thinking state indicator |
| | `--clr-planning` | `#3B82F6` (blue-500) | Planning phase accent |
| | `--clr-tool-call` | `#8B5CF6` (violet-500) | Tool invocation accent |
| | `--clr-self-correct` | `#F59E0B` (amber-500) | Self-correction highlight |
| | `--clr-error` | `#EF4444` (red-500) | Error state |
| | `--clr-user-msg` | `var(--surface-2)` | User message background |
| | `--clr-assistant-msg` | `var(--surface-1)` | Assistant message background |
| | `--clr-conf-high` | `#22C55E` (green-500) | High confidence indicator |
| | `--clr-conf-medium` | `#EAB308` (yellow-500) | Medium confidence indicator |
| | `--clr-conf-low` | `#EF4444` (red-500) | Low confidence indicator |

```css
/* Example: CSS custom properties for an AI UX design system */
:root {
  /* Token categories above, applied below */
  --ai-msg-padding: 12px 16px;
  --ai-msg-border-radius: 12px;
  --ai-thinking-min-height: 32px;
}

.ai-message {
  padding: var(--ai-msg-padding);
  border-radius: var(--ai-msg-border-radius);
  animation: fadeIn var(--dur-stream-min) ease-in;
}

.ai-tool-call {
  border-left: 3px solid var(--clr-tool-call);
  padding-left: var(--space-tool-indent);
  margin: 8px 0;
}
```

### 12.2 Component Library for AI Products

| Component | States | Props | Accessibility |
|-----------|--------|-------|---------------|
| **ChatMessage** | user, assistant, system, streaming | `role`, `content`, `sources: []`, `confidence`, `timestamp` | `role="log"`, `aria-label="AI response"` |
| **StreamingText** | idle, streaming, complete, error | `text`, `speed`, `onComplete`, `cursorStyle` | `aria-live="polite"`, `aria-atomic="false"` |
| **AgentStateIndicator** | thinking, planning, executing, tool-calling, self-correcting, done, error | `state`, `detail`, `progress`, `steps` | `role="status"`, `aria-label="Agent is {state}"` |
| **ToolCall** | pending, running, success, error | `toolName`, `arguments`, `result`, `durationMs`, `collapsible` | `role="region"`, `aria-label="Tool: {toolName}"` |
| **ConfidenceIndicator** | high, medium, low, unknown | `value`, `label`, `variant` (bar/dot/badge) | `aria-label="{value}% confidence"` |
| **SourceCitation** | inline, footnote, expandable | `number`, `title`, `url`, `snippet` | `role="doc-footnote"` if expanded |
| **InlineEdit** | view, edit, saving, saved, error | `text`, `onSave`, `validator`, `diffMode` | `aria-label="Editable AI output"` |
| **ComparisonView** | side-by-side, stacked, slider | `original`, `variant`, `mode` | `role="region"`, `aria-roledescription="diff view"` |
| **FeedbackWidget** | thumbs, rating, comment, done | `onFeedback`, `options`, `position` | `aria-label="Rate this response"` |
| **MultiAgentView** | grid, list, timeline | `agents: []`, `layout`, `onSelect` | `role="group"`, `aria-label="Agent team overview"` |

### 12.3 Pattern Library Organization

Organize an AI UX pattern library into tiers:

**Tier 1 — Core Patterns (every AI product needs):**
- Chat message display (user/assistant/system)
- Streaming text rendering
- Typing/thinking indicators
- Input with multi-modal support (text, voice, image)
- Error recovery: retry, clarify, fallback

**Tier 2 — Advanced Patterns (agent-based products):**
- Agent state visualization (thinking → planning → executing)
- Tool call display with results
- Self-correction transparency
- Multi-agent coordination view
- Confidence and uncertainty communication

**Tier 3 — Product-Specific Patterns:**
- Code diff view (coding assistants)
- Image comparison slider (generative AI)
- Conversation branching (creative tools)
- Collaborative editing (document AI)
- Simulation preview (data analysis)

### 12.4 Accessibility-First Design System Checklist

| Requirement | Implementation | WCAG Criterion |
|-------------|---------------|:--------------:|
| **Streaming text** announced | `aria-live="polite"` on output region | 4.1.3 |
| **Tool calls** communicated | Live region updates for start/end | 4.1.3 |
| **Agent state** changes | `role="status"` on state indicator | 4.1.2 |
| **Confidence indicators** | Text label + visual bar (not color alone) | 1.4.1 |
| **Keyboard navigation** | All tool calls focusable, Enter to expand | 2.1.1 |
| **Focus management** | Focus moves to new content after streaming | 2.4.3 |
| **Color contrast** | Token values meet 4.5:1 (AA) minimum | 1.4.3 |
| **Motion sensitivity** | `prefers-reduced-motion` for animations | 2.3.3 |
| **Touch targets** | Minimum 44×44px for interactive elements | 2.5.8 |

### 12.5 Component Architecture Example

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"

@dataclass
class Source:
    title: str
    url: str
    snippet: str
    relevance_score: float = 0.0

@dataclass
class ToolCallInfo:
    name: str
    arguments: dict
    result: Optional[str] = None
    duration_ms: float = 0.0
    status: str = "pending"  # pending, running, success, error

@dataclass
class ChatMessage:
    role: MessageRole
    content: str
    sources: list[Source] = field(default_factory=list)
    tool_calls: list[ToolCallInfo] = field(default_factory=list)
    confidence: Optional[float] = None
    is_streaming: bool = False
    timestamp: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_ui_state(self) -> dict:
        """Serialize to frontend state."""
        return {
            "role": self.role.value,
            "content": self.content,
            "sources": [{"title": s.title, "url": s.url,
                         "snippet": s.snippet} for s in self.sources],
            "toolCalls": [{"name": t.name, "status": t.status,
                           "duration": t.duration_ms} for t in self.tool_calls],
            "confidence": self.confidence,
            "isStreaming": self.is_streaming,
        }

class DesignSystemTheme:
    """Programmatic access to design tokens for theming."""
    def __init__(self, mode: str = "light"):
        self.mode = mode
        self.tokens = self._load_tokens()

    def _load_tokens(self) -> dict:
        return {
            "colors": {
                "thinking": "#6B7280" if self.mode == "light" else "#9CA3AF",
                "planning": "#3B82F6" if self.mode == "light" else "#60A5FA",
                "toolCall": "#8B5CF6" if self.mode == "light" else "#A78BFA",
                "error": "#EF4444" if self.mode == "light" else "#F87171",
                "confHigh": "#22C55E",
                "confMedium": "#EAB308",
                "confLow": "#EF4444",
            },
            "spacing": {"chatGap": 16, "toolIndent": 24, "avatarGap": 8},
            "timing": {
                "thinkDisplay": 200, "streamMin": 50,
                "typeIndicator": 300, "transitionState": 400,
            },
        }

    def css_vars(self) -> str:
        """Generate CSS custom property string."""
        lines = [":root {"]
        for category, values in self.tokens.items():
            for key, val in values.items():
                lines.append(f"  --{category}-{key}: {val};")
        lines.append("}")
        return "\n".join(lines)

# Usage:
theme = DesignSystemTheme("dark")
print(theme.css_vars())
```

---

## 13. Cross-References
| Reference | Description |
|-----------|-------------|
| [08-Reference/03-Agent-Configs-SOUL-SKILL.md] | Agent personality configuration for UX consistency |
| [03-Agents/01-Agent-Architectures.md] | Agent interaction patterns and state machines |
| [06-Advanced/04-Prompt-Engineering.md] | Prompt design fundamentals for UX |
| [06-Advanced/03-Evaluation-Benchmarks.md] | Evaluation methodology for UX metrics |
| [05-Enterprise/01-Enterprise-AI-Deployment.md] | UX at production scale |
| [01-Foundations/03-Deep-Learning.md] | Model architectures behind AI UX |
| [06-Advanced/09-AI-UX-and-Interaction.md] | This document — AI UX design patterns |
|---
*Document version: 2.5 — June 2026 | Expanded: added §12 AI UX Design Systems (design tokens, component library, pattern organization, accessibility checklist, component architecture code example), updated Cross-References*
