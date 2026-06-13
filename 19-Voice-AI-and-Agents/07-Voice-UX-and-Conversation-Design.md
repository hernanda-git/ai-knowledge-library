# 07 — Voice UX and Conversation Design for AI Agents

## Overview

Voice user experience (VUX) and conversation design are critical disciplines for creating AI agents that people actually enjoy interacting with. Unlike graphical interfaces, voice interactions are linear, ephemeral, and demand cognitive effort from users. Poorly designed voice agents frustrate users, erode trust, and increase abandonment rates. This document provides a comprehensive framework for designing natural, effective, and delightful voice interactions for AI agents, covering conversation flow, persona design, turn-taking, error recovery, accessibility, and testing methodologies.

## Table of Contents

1. [Foundations of Voice UX Design](#1-foundations-of-voice-ux-design)
2. [Conversation Architecture](#2-conversation-architecture)
3. [Agent Persona and Voice Branding](#3-agent-persona-and-voice-branding)
4. [Turn-Taking and Flow Management](#4-turn-taking-and-flow-management)
5. [Prompt Design for Voice](#5-prompt-design-for-voice)
6. [Error Recovery Strategies](#6-error-recovery-strategies)
7. [Context and State Management](#7-context-and-state-management)
8. [Multi-Turn Conversation Patterns](#8-multi-turn-conversation-patterns)
9. [Accessibility and Inclusive Design](#9-accessibility-and-inclusive-design)
10. [Emotional Intelligence and Empathy](#10-emotional-intelligence-and-empathy)
11. [Voice UX Testing and Evaluation](#11-voice-ux-testing-and-evaluation)
12. [Conversational Analytics](#12-conversational-analytics)
13. [Voice UX Pattern Library](#13-voice-ux-pattern-library)
14. [Regulatory and Ethical Considerations](#14-regulatory-and-ethical-considerations)
15. [References and Further Reading](#15-references-and-further-reading)

---

## 1. Foundations of Voice UX Design

### 1.1 Voice vs Visual Interfaces

| Dimension | Voice Interface | Visual/GUI Interface |
|-----------|----------------|---------------------|
| Input | Sequential, one-at-a-time | Parallel, visual scan |
| Output | Ephemeral, heard once | Persistent, re-readable |
| Bandwidth | Limited (~150 wpm speech) | High (~300 wpm reading) |
| Memory Load | High (must remember options) | Low (visible choices) |
| Privacy | Audible to bystanders | Private on screen |
| Speed | Slow for data entry | Fast for structured input |
| Accessibility | Good for visual/motor impaired | Good for hearing impaired |
| Hands-Free | Yes | No |
| Emotional | Rich prosodic expression | Limited text-based expression |

### 1.2 Core Voice UX Principles

**1 — Design for the Ear, Not the Eye**
- Use shorter sentences (15–20 words max)
- Avoid lists longer than 3–5 items
- Use simpler vocabulary (avoid jargon)
- Repeat key information when needed

**2 — Minimize Cognitive Load**
- Don't present too many options at once
- Use progressive disclosure (ask step by step)
- Allow users to say "I don't know" and "help"
- Remember past context to avoid re-asking

**3 — Provide Clear Feedback**
- Acknowledge every user input
- Indicate listening state (earcon, visual indicator)
- Confirm understanding before acting
- Signal processing time (avoid dead air)

**4 — Design for Errors**
- Assume ASR will make mistakes
- Assume users will say unexpected things
- Provide graceful recovery paths
- Never leave users stuck

**5 — Be Honest About AI Nature**
- Disclose that user is talking to AI
- Set appropriate expectations
- Escalate to human when needed
- Don't try to pass as human

### 1.3 The Voice UX Design Process

```
Research → Persona → Flows → Scripts → Prototype → Test → Iterate → Launch → Monitor
```

**Phase 1 — Research**
- User needs analysis
- Existing interaction patterns
- Competitor analysis
- Technical constraints assessment

**Phase 2 — Persona Design**
- Define agent personality
- Set tone and language style
- Create voice guidelines
- Design communication patterns

**Phase 3 — Conversation Flows**
- Map user journeys
- Design dialog states
- Plan error handling
- Define escalation paths

**Phase 4 — Script Writing**
- Write dialog for each state
- Plan variations for different contexts
- Design prompting strategy
- Include edge case handling

**Phase 5 — Prototyping**
- Create wizard-of-oz prototype
- Build functional prototype
- Test with real users
- Iterate based on feedback

**Phase 6 — Launch and Monitor**
- Deploy with monitoring
- Track conversational metrics
- Continuously improve
- A/B test variations

---

## 2. Conversation Architecture

### 2.1 Dialog State Machine

A voice conversation can be modeled as a state machine where each state defines what the agent can say and what user inputs it accepts.

```
                           ┌─────────────────────────┐
                           │     WELCOME/GREETING     │
                           └───────────┬─────────────┘
                                       │
                                       ▼
                           ┌─────────────────────────┐
                ┌──────────│     LISTENING/INPUT      │◀──────────┐
                │          └───────────┬─────────────┘           │
                │                      │                         │
                │                      ▼                         │
                │          ┌─────────────────────────┐           │
                │          │   UNDERSTAND/PROCESS     │           │
                │          └───────────┬─────────────┘           │
                │                      │                         │
                │                      ▼                         │
                │          ┌─────────────────────────┐           │
                │◀─────────│       RESPOND/SPEAK      │───────────┘
                │          └───────────┬─────────────┘
                │                      │
                │                      ▼
                │          ┌─────────────────────────┐
                │          │     CONFIRMATION          │
                │          └───────────┬─────────────┘
                │                      │
                └──────────────────────┘
                                       │
                                       ▼
                           ┌─────────────────────────┐
                           │       CLOSING/END        │
                           └─────────────────────────┘
```

### 2.2 Dialog State Definition

```yaml
dialog_state:
  name: "get_account_info"
  description: "Retrieve account information for the user"
  
  entry_prompts:
    - "What account information would you like to know?"
    - "I can check your balance, recent transactions, or account details."
    - "What would you like to look up about your account?"
  
  expected_inputs:
    - type: "intent"
      values: ["balance", "transactions", "account_details", "help"]
  
  slot_filling:
    - slot: "account_type"
      required: true
      prompt: "Which account type? Checking, savings, or credit card?"
      validation: "must_be_valid_account_type"
    
    - slot: "date_range"
      required: false
      prompt: "For which time period?"
  
  confirmation:
    required: true
    template: "Let me look up the {account_type} {intent} for you."
  
  error_handling:
    no_input: "I didn't hear anything. Please say the information you'd like."
    no_match: "I'm sorry, I didn't understand that. You can ask about your balance, recent transactions, or account details."
    timeout: "I'll disconnect now since I didn't hear back. Call again anytime."
  
  transitions:
    on_success: "present_information"
    on_failure: "apologize_and_offer_help"
    on_escalate: "transfer_to_human"
```

### 2.3 Slot Filling Patterns

Slot filling is the process of collecting required information from the user to fulfill a request.

**a) Explicit Slot Filling**
```
Agent: What's your date of birth?
User: January 15th, 1985.
Agent: And what's your ZIP code?
User: 94105.
```

**b) Implicit Slot Filling (One-Shot)**
```
Agent: I can help you book a flight. Where are you flying to?
User: I need to fly from San Francisco to New York tomorrow morning.
```
(Agent extracts all slots from single utterance)

**c) Mixed Initiative**
```
User: I need to check my balance.
Agent: Sure. For security, please tell me your date of birth.
User: January 15th, 1985.
Agent: Thank you. Your checking account balance is $2,450. Would you like to know anything else?
User: Yes, what were my recent transactions?
Agent: (continues in context)
```

### 2.4 Conversation Flow Templates

**Task-Oriented Flow:**
```
Opening → Identify Need → Gather Info → Confirm → Execute → Result → Closing
```

**Exploratory Flow:**
```
Opening → Open Question → Listen → Clarify → Discuss → Summarize → Closing
```

**Troubleshooting Flow:**
```
Opening → State Problem → Diagnose → Offer Solution → Confirm Resolution → Closing
```

**Transactional Flow:**
```
Opening → Authenticate → Select Action → Execute → Confirm → Closing
```

---

## 3. Agent Persona and Voice Branding

### 3.1 Persona Dimensions

**a) Warmth vs Authority**
- Warm: Friendly, casual, empathetic, uses contractions
- Authoritative: Professional, confident, direct, formal
- Most agents should target high warmth with moderate authority

**b) Formality Level**
- Casual: Slang, informal greetings, humor allowed
- Neutral: Polite, standard professional language
- Formal: Honorifics, structured language, minimal humor

**c) Verbosity**
- Concise: Short responses, minimal confirmation
- Balanced: Clear responses with context
- Detailed: Thorough explanations, confirmations, alternatives

**d) Personality Traits**
- Helpful vs Efficient
- Playful vs Serious
- Proactive vs Reactive
- Emotional vs Neutral

### 3.2 Persona Questionnaire

```yaml
persona_definition:
  name: "Alex"
  role: "Customer Service Agent"
  
  traits:
    - helpful: 9/10
    - patient: 8/10
    - knowledgeable: 9/10
    - friendly: 7/10
    - efficient: 6/10
    - humorous: 3/10
  
  language_style:
    greeting: "Hi there! Thanks for calling [Company]. I'm Alex."
    confirmation: "Got it. Let me look into that."
    error: "Sorry about that — I'm having trouble with that request."
    closing: "Happy to help! Is there anything else I can do for you?"
  
  constraints:
    - must_disclose_ai: true
    - no_pretending_to_be_human: true
    - no_sarcasm_or_snark: true
    - must_escalate_to_human_if_inappropriate: true
  
  voice_properties:
    gender: "neutral"  # or male/female
    accent: "american"
    speed: "medium"  # slightly slower for clarity
    pitch: "medium"
    variation: "natural_prosody"
```

### 3.3 Voice and Tone Guidelines

**Do Say:**
- "I understand how frustrating that must be."
- "Let me check that for you right now."
- "Here's what I found..."
- "Is there anything else I can help with?"

**Don't Say:**
- "You're wrong." (Instead: "I have a different record. Let me check again.")
- "I already told you that." (Instead: "As I mentioned earlier...")
- "That's not my problem." (Instead: "Let me transfer you to someone who can help.")
- "I don't know." (Instead: "Let me look that up for you.")

### 3.4 Creating an Agent Bio

```yaml
agent_bio:
  name: "Aria"
  company: "Acme Financial Services"
  role: "Digital Banking Assistant"
  specialization: "Personal banking, account management, bill pay"
  background: "AI assistant developed by Acme Financial to help customers manage their money 24/7."
  limitations:
    - "Cannot process transactions over $10,000"
    - "Cannot modify account ownership"
    - "Cannot provide investment advice"
  escalation_triggers:
    - "Fraud reports"
    - "Account closure requests"
    - "Complex disputes"
  human_transfer_script: "Let me connect you with a specialist who can help with that. One moment please."
```

---

## 4. Turn-Taking and Flow Management

### 4.1 Turn-Taking Signals

In voice conversations, turn-taking must be explicitly managed since there are no visual cues.

**a) End-of-Turn Signals**
- Rising intonation (question)
- Filled pause ("umm", "let's see")
- Explicit handoff ("What do you think?")
- Silence threshold exceeded

**b) Keeping the Turn**
- Floor-holding phrases ("Let me think about that...")
- Discourse markers ("First...", "There are three things...")
- Trail-off intonation indicating more to come

**c) Interruption Handling (Barge-In)**
- Always allow user interruption
- Stop speaking immediately when interrupted
- Process interruption as new input
- Resume from where you left off if appropriate

### 4.2 Turn-Taking Configuration

```python
class TurnManager:
    def __init__(self, silence_timeout_ms=1500,
                 max_turn_duration_ms=30000,
                 barge_in_enabled=True):
        self.silence_timeout = silence_timeout_ms
        self.max_turn_duration = max_turn_duration_ms
        self.barge_in_enabled = barge_in_enabled
        self.current_speaker = "user"  # "user" or "agent"
        self.turn_start_time = 0
        self.last_utterance_time = 0

    def user_starts_speaking(self):
        if self.current_speaker == "agent" and self.barge_in_enabled:
            # Barge-in detected, agent should stop
            return "interrupt_agent"
        self.current_speaker = "user"
        self.turn_start_time = time.time()
        return "continue"

    def user_stops_speaking(self):
        self.last_utterance_time = time.time()
        return "agent_can_speak"

    def agent_starts_speaking(self):
        self.current_speaker = "agent"
        self.turn_start_time = time.time()

    def agent_stops_speaking(self):
        self.current_speaker = "user"
        self.last_utterance_time = time.time()

    def check_timeout(self) -> str:
        """Check if current turn has exceeded limits."""
        elapsed = time.time() - self.turn_start_time
        silence = time.time() - self.last_utterance_time

        if self.current_speaker == "agent" and elapsed > self.max_turn_duration:
            return "agent_timeout"
        if self.current_speaker == "user" and silence > self.silence_timeout:
            return "user_timeout"
        return "ok"
```

### 4.3 Backchanneling and Acknowledgment

Backchanneling signals that the agent is listening and processing without taking the full turn.

**a) Listening Sounds**
- Brief "mm-hmm", "I see", "okay"
- Short acknowledgment before processing
- Must be used sparingly (not after every word)

**b) Processing Indicators**
- "Let me look into that..."
- "One moment please..."
- "That's a great question..."
- Short earcon or chime for continuity

**c) Confirmation Before Action**
- "So you'd like to transfer $500 to savings. Is that correct?"
- "I'll schedule that appointment for Tuesday at 2 PM. Shall I proceed?"
- "I found your account. You're asking about the late fee on your statement, right?"

### 4.4 Handling Silence and Timeouts

| Silence Duration | Action |
|-----------------|--------|
| 0.5–1.5s | Natural pause, wait for more input |
| 1.5–3s | Prompt: "I didn't catch that. Could you repeat?" |
| 3–5s | Re-prompt with specific guidance |
| 5–10s | Offer help or alternatives |
| 10s+ | Closing: "I'll disconnect now. Call back anytime." |

---

## 5. Prompt Design for Voice

### 5.1 Voice Prompt Principles

**1 — One Thought Per Sentence**
- Bad: "Please say your account number, which is on your statement, and then your date of birth."
- Good: "Please say your account number. Then tell me your date of birth."

**2 — Front-Load Key Information**
- Bad: "If you'd like to hear your balance or recent transactions or make a payment, say what you want."
- Good: "You can check your balance, review transactions, or make a payment. What would you like?"

**3 — Use Specific, Actionable Language**
- Bad: "What would you like to do?"
- Good: "Would you like to check your balance, pay a bill, or hear recent transactions?"

**4 — Offer Examples for Open Prompts**
- Bad: "Tell me what you need."
- Good: "Tell me what you need. For example, you can say 'What's my balance?' or 'Pay my credit card bill.'"

**5 — Keep Confirmations Brief**
- Bad: "I just want to confirm that you would like to transfer $500.00 from your checking account ending in 1234 to your savings account ending in 5678. Is that correct?"
- Good: "Transfer $500 from checking to savings. Is that right?"

### 5.2 Prompt Templates

**Greeting Prompt:**
```
"Hi, this is [agent name] from [company]. I'm your AI assistant.
I can [capability 1], [capability 2], or [capability 3].
What can I help you with today?"
```

**Confirmation Prompt:**
```
"Just to confirm, you'd like to [action]. Is that correct?"
```

**Clarification Prompt:**
```
"I want to make sure I understand. Did you mean [option A] or [option B]?"
```

**Error Recovery Prompt:**
```
"I didn't quite get that. Could you try rephrasing?
For example, you can say '[example 1]' or '[example 2]'."
```

**Help Prompt:**
```
"I can help you with several things:
— Check your account balance
— Make a payment
— Find recent transactions
— Update your profile
What would you like to do?"
```

**Closing Prompt:**
```
"Thanks for calling [company]. I hope I was able to help!
If you need anything else, don't hesitate to call back.
Have a great [time of day]!"
```

### 5.3 System Prompt for Voice Agent LLM

```yaml
voice_agent_system_prompt:
  role: "You are a friendly and helpful voice assistant for [Company]."
  
  constraints:
    - "Always respond in natural conversational English, as if speaking aloud."
    - "Keep responses under 3 sentences unless the user asks for details."
    - "Never use markdown, lists, or formatting in your responses."
    - "Use contractions (I'm, you'll, that's) for natural flow."
    - "Pause naturally between key pieces of information."
    - "If you need to give multiple options, present no more than 3 at a time."
    - "Always confirm before taking irreversible actions."
    - "If you don't know something, say so honestly and offer alternatives."
    - "Never pretend to be a human. If asked, clarify you're an AI assistant."
    - "Keep your tone warm, patient, and professional."
  
  format_instructions:
    - "Use '...' to indicate a thoughtful pause."
    - "Use SSML <break time='500ms'/> for longer pauses between sections."
    - "Use <emphasis> for important information."
    - "Speak numbers digit by digit for account numbers: 'one two three four'."
    - "Speak money amounts naturally: 'four hundred and fifty dollars'."
  
  persona:
    name: "Aria"
    company: "[Company]"
    personality: "Helpful, patient, knowledgeable, slightly warm but professional"
    
  examples:
    - input: "What's my balance?"
      output: "Sure, let me check that for you. (pause) Your checking account balance is one thousand two hundred and thirty dollars and forty-five cents. Is there anything else?"
    
    - input: "I'm really frustrated with this late fee"
      output: "I completely understand your frustration. Let me look into what happened with your account and see what we can do to resolve this."
    
    - input: "Are you a real person?"
      output: "I'm an AI assistant designed to help you with your banking needs. I can handle most requests, but I can also connect you with a human representative if you'd prefer."
```

### 5.4 Prompt Length Guidelines

| Context | Optimal Length | Max Length |
|---------|---------------|-----------|
| Greeting | 15–25 words | 30 words |
| Confirmation | 8–15 words | 20 words |
| Error recovery | 15–25 words | 35 words |
| Instructions | 20–40 words | 50 words |
| Options list | 30–50 words (3–5 items) | 70 words (7 items) |
| Help prompt | 40–60 words | 80 words |
| Closing | 15–25 words | 30 words |

---

## 6. Error Recovery Strategies

### 6.1 Types of Voice Errors

**a) ASR Errors (Misrecognition)**
- User said something correctly but ASR transcribed incorrectly
- User said something unexpected
- Background noise caused misrecognition

**b) NLU Errors (Misunderstanding)**
- ASR was correct but intent/entity extraction failed
- User expressed intent in unexpected way
- Ambiguous user request

**c) Fulfillment Errors**
- Action could not be completed
- Missing required information
- Technical failure during execution

### 6.2 Error Recovery Layers

```
Layer 1: Reprompt (ask user to repeat)
Layer 2: Rephrase (suggest different phrasing)
Layer 3: Offer choices (narrow down options)
Layer 4: Transfer to human (when all else fails)
```

### 6.3 Error Recovery Patterns

**Pattern 1: Simple Reprompt**
```
User: [unintelligible]
Agent: "I didn't catch that. Could you please say it again?"
```

**Pattern 2: Guided Reprompt**
```
User: [unintelligible]
Agent: "I'm sorry, I didn't quite get that. You can say things like 'What's my balance?' or 'Pay my bill.'"
```

**Pattern 3: Confirmation with Correction**
```
User: "What's my balance?"
Agent: "Your savings account balance is $2,500."
User: "I meant my checking account."
Agent: "Sorry for the confusion! Your checking account balance is $1,200."
```

**Pattern 4: Progressive Disambiguation**
```
Agent: "Would you like to check your balance or make a payment?"
User: [unintelligible]
Agent: "I still didn't catch that. Let me ask differently. Please say 'balance' or 'payment'."
User: "Balance."
Agent: "Great! Your checking account balance is $2,450."
```

**Pattern 5: Explicit Correction**
```
User: "Transfer $500 to savings."
Agent: "Transfer $500 from checking to savings. Confirm?"
User: "No, transfer $200."
Agent: "Correction. Transfer $200 from checking to savings. Confirm?"
```

### 6.4 Error Recovery Implementation

```python
class VoiceErrorHandler:
    def __init__(self, max_retries=3):
        self.max_retries = max_retries
        self.retry_count = 0
        self.consecutive_failures = 0

    def handle_error(self, error_type: str, context: dict) -> str:
        """Generate appropriate recovery prompt."""
        if error_type == "no_input":
            return self._handle_no_input()
        elif error_type == "no_match":
            return self._handle_no_match(context)
        elif error_type == "low_confidence":
            return self._handle_low_confidence(context)
        elif error_type == "fulfillment_error":
            return self._handle_fulfillment_error(context)
        else:
            return self._handle_generic_error()

    def _handle_no_input(self) -> str:
        """Handle silence or no detectable input."""
        if self.retry_count == 0:
            return "I didn't hear anything. Could you please say that again?"
        elif self.retry_count == 1:
            return "I'm still not hearing anything. Please speak clearly into your microphone."
        else:
            self.consecutive_failures += 1
            return "Let me transfer you to a human agent who can assist further."

    def _handle_no_match(self, context: dict) -> str:
        """Handle input that wasn't understood."""
        self.consecutive_failures += 1
        if self.retry_count == 0:
            examples = context.get("examples", [])
            if examples:
                return (f"I didn't understand that. You can say things like "
                        f"'{examples[0]}' or '{examples[1]}'.")
            return "I didn't understand that. Could you please rephrase?"
        else:
            return "I'm having trouble understanding. Let me connect you with a human agent."

    def _handle_low_confidence(self, context: dict) -> str:
        """Handle ASR results with low confidence."""
        hypothesis = context.get("hypothesis", "")
        return (f"Did you say '{hypothesis}'? "
                f"I want to make sure I understood correctly.")

    def _handle_fulfillment_error(self, context: dict) -> str:
        """Handle failures during action execution."""
        return ("I'm sorry, I'm having trouble processing that request. "
                "Let me try again... or I can transfer you to a human agent.")

    def _handle_generic_error(self) -> str:
        """Handle unexpected errors."""
        return ("I apologize, but something went wrong on my end. "
                "Let me try that again.")

    def should_escalate(self) -> bool:
        """Determine if we should escalate to human."""
        return self.consecutive_failures >= self.max_retries

    def reset(self):
        self.retry_count = 0
        self.consecutive_failures = 0
```

---

## 7. Context and State Management

### 7.1 Conversation Context Model

```python
@dataclass
class VoiceConversationContext:
    session_id: str
    user_id: str
    turn_count: int
    current_state: str
    slots: Dict[str, Any]
    history: List[Dict[str, Any]]
    user_profile: Dict[str, Any]
    system_context: Dict[str, Any]
    metadata: Dict[str, Any]
```

### 7.2 Context Persistence Strategies

**a) In-Session Context (Short-term)**
```
Context: Current conversation turns
Persistence: In-memory, per session
Lifetime: Duration of voice call
Contents: Recent dialog history, active slots, current state
```

**b) Cross-Turn Context (Medium-term)**
```
Context: User preferences for current session
Persistence: Session storage
Lifetime: Entire session
Contents: User choices, dispreferences, pace preferences
```

**c) Cross-Session Context (Long-term)**
```
Context: User history across sessions
Persistence: Database
Lifetime: Days to months (with privacy controls)
Contents: Past interactions, saved preferences, frequently used features
```

### 7.3 Context Window Management for Voice

Voice conversations can be long (15–30+ min), requiring careful context management.

```python
class VoiceContextManager:
    def __init__(self, max_history_turns=20,
                 max_context_tokens=4000):
        self.max_turns = max_history_turns
        self.max_tokens = max_context_tokens
        self.history = []
        self.important_facts = []
        self.summaries = []

    def add_turn(self, role: str, content: str, metadata: dict = None):
        """Add a conversation turn to context."""
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {}
        })

        # Trim history if too long
        if len(self.history) > self.max_turns:
            old_turns = self.history[:-self.max_turns]
            self._summarize_old_turns(old_turns)
            self.history = self.history[-self.max_turns:]

    def _summarize_old_turns(self, turns: list):
        """Create a summary of old conversation turns."""
        summary = self._generate_summary(turns)
        self.summaries.append(summary)

    def _generate_summary(self, turns: list) -> str:
        """Generate a condensed summary of turns."""
        key_points = []
        for turn in turns:
            if turn["metadata"].get("important"):
                key_points.append(turn["content"][:100])
        return " | ".join(key_points) if key_points else ""

    def get_context_for_llm(self) -> str:
        """Get formatted context for LLM processing."""
        parts = []

        # Add summaries
        if self.summaries:
            parts.append("[Previous conversation summary: " +
                        " ".join(self.summaries[-3:]) + "]")

        # Add recent history
        for turn in self.history[-10:]:
            prefix = "User" if turn["role"] == "user" else "Assistant"
            parts.append(f"{prefix}: {turn['content']}")

        return "\n".join(parts)

    def store_important_fact(self, fact: str):
        """Store an important fact that should be remembered."""
        self.important_facts.append(fact)
```

---

## 8. Multi-Turn Conversation Patterns

### 8.1 Common Conversation Patterns

**Pattern 1: Greeting → Task → Detail → Confirm → Execute → Close**

```
Agent: "Welcome to Acme Bank! I'm Aria. You can check your balance,
        make a payment, or review transactions. What can I help with?"
User:  "What's my balance?"
Agent: "I'll need to verify your identity first. What's your date of birth?"
User:  "January 15th, 1985."
Agent: "Thank you. Your checking account balance is $2,450. Anything else?"
User:  "No, that's all."
Agent: "Thanks for calling Acme Bank. Have a great day!"
```

**Pattern 2: Chaining Tasks (Sequential)**

```
Agent: "Your payment has been scheduled. Anything else?"
User:  "Yes, I need to update my address."
Agent: "Sure! What's your new street address?"
User:  "123 Oak Street, Apt 4B, Portland, Oregon 97201."
Agent: "Got it. Your address has been updated to 123 Oak Street, Apt 4B,
        Portland, Oregon 97201. Is there anything else?"
```

**Pattern 3: Task with Interruption**

```
Agent: "Your transaction history for the past month shows...
User:  "Actually, I want the past 3 months."
Agent: "Sure, let me pull that up. Here's your transaction history for
        the past 3 months..."
```

**Pattern 4: Clarification Loop**

```
User:  "I want to transfer money."
Agent: "Which account would you like to transfer from?"
User:  "My savings."
Agent: "How much would you like to transfer?"
User:  "$500."
Agent: "And which account should it go to?"
User:  "My checking account."
Agent: "Transfer $500 from savings to checking. Confirm?"
User:  "Yes."
Agent: "Done! $500 has been transferred to your checking account."
```

**Pattern 5: Multi-Intent Handling**

```
User:  "What's my balance and did my last payment go through?"
Agent: "Your checking balance is $2,450. And yes, your last payment of
        $150 to Electric Company posted on June 5th. What else?"
```

### 8.2 Context Carry-Over

```python
class ContextCarryOver:
    def __init__(self):
        self.context_stack = []
        self.current_context = {}

    def set_context(self, key: str, value: Any):
        """Set a context value for the current turn."""
        self.current_context[key] = value

    def get_context(self, key: str, default=None):
        """Get a context value, checking current and previous turns."""
        if key in self.current_context:
            return self.current_context[key]
        for ctx in reversed(self.context_stack):
            if key in ctx:
                return ctx[key]
        return default

    def next_turn(self):
        """Advance to next turn, saving current context."""
        if self.current_context:
            self.context_stack.append(self.current_context)
            self.current_context = {}
        # Keep only last 10 contexts
        if len(self.context_stack) > 10:
            self.context_stack = self.context_stack[-10:]

    def resolve_pronouns(self, text: str) -> str:
        """Resolve pronouns based on context."""
        # "it" → last mentioned entity
        # "that" → last action or suggestion
        # "there" → last mentioned location
        last_entity = self.get_context("last_entity")
        if last_entity:
            text = text.replace("it", last_entity)
            text = text.replace("that", last_entity)
        return text
```

---

## 9. Accessibility and Inclusive Design

### 9.1 Accessibility Considerations for Voice

**For Users with Speech Impairments:**
- Accept shorter responses (yes/no)
- Allow touch-tone DTMF as alternative input
- Speak more slowly and enunciate clearly
- Don't interrupt or rush the user

**For Users with Hearing Impairments:**
- Provide visual transcript of agent speech
- Offer text-based alternatives (SMS, chat)
- Use clear enunciation (TTS quality matters)
- Avoid rapid speech

**For Users with Cognitive Disabilities:**
- Use very simple sentence structures
- Repeat key information
- Avoid long lists or complex choices
- Allow extra time for responses
- Provide clear, concrete instructions

**For Non-Native Speakers:**
- Use standard, neutral accent TTS
- Avoid idioms, slang, and cultural references
- Speak slightly slower than normal
- Confirm understanding more frequently
- Offer language switching when applicable

### 9.2 Inclusive Language Guidelines

- Use gender-neutral terms ("they" instead of "he/she")
- Avoid ableist language ("see" for "understand")
- Use person-first language ("person with disability")
- Avoid cultural assumptions and stereotypes
- Offer multiple ways to accomplish the same task
- Respect regional language variations

### 9.3 Accessibility Configuration

```yaml
accessibility_config:
  speech_speed:
    default: 1.0  # normal
    slow: 0.75    # for hearing/cognitive impaired
    fast: 1.25    # for power users
    
  input_methods:
    - "voice"  # primary
    - "dtmf"   # touch-tone fallback
    - "sms"    # text fallback
    - "chat"   # web chat fallback
    
  visual_aids:
    show_transcript: true
    show_options_on_screen: true
    show_agent_avatar: false  # avoid uncanny valley
    
  language_support:
    primary: "en-US"
    secondary:
      - "es-US"
      - "zh-CN"
      - "vi-VN"
    
  timeout_multiplier:
    standard: 1.0
    extended: 2.0  # for users who need more time
```

---

## 10. Emotional Intelligence and Empathy

### 10.1 Detecting User Emotion

Voice agents can infer emotion from:
- **Prosody**: Pitch, volume, speech rate, tone
- **Lexical choices**: Word selection indicating emotion
- **Content**: What the user is saying
- **Context**: Known frustrations or difficult situations

### 10.2 Emotion-Responsive Strategies

| User Emotion | Detection Cues | Agent Response Strategy |
|-------------|---------------|----------------------|
| Frustration | Raised voice, sighs, repeated requests | Apologize, offer solutions, escalate if needed |
| Anger | Loud, clipped speech, swearing | Stay calm, validate feelings, offer human transfer |
| Confusion | Hesitation, questions, silence | Simplify, rephrase, offer step-by-step guidance |
| Urgency | Fast speech, brief utterances | Be efficient, prioritize, skip pleasantries |
| Satisfaction | Normal tone, positive words | Maintain warmth, ask for feedback |
| Sadness | Slow speech, low volume | Be empathetic, patient, offer support |

### 10.3 Empathy Script Templates

**When User is Frustrated:**
```
"I can hear this is frustrating. Let me do my best to help resolve this quickly.
Here's what I can do: [specific action]. Would you like me to proceed?"
```

**When User Makes an Error:**
```
"No problem at all — that's easy to fix. Let's try again.
[Rephrase the request more simply]"
```

**When User is in a Hurry:**
```
"I'll make this quick. Here's what you need: [direct answer].
Is there anything else you need right now?"
```

**When Apologizing:**
```
"I sincerely apologize for the confusion. You're right to expect better.
Let me make this right by [specific corrective action]."
```

**When Escalating:**
```
"I want to make sure you get the best help possible. This seems like something
I should have a human specialist handle. Let me transfer you."
```

### 10.4 Emotional State Tracking

```python
class EmotionalStateTracker:
    def __init__(self):
        self.emotion_history = []
        self.current_emotion = "neutral"
        self.frustration_level = 0

    def detect_emotion(self, text: str, prosody: dict = None) -> str:
        """Detect user emotion from text and optional prosody features."""
        # Lexical markers
        frustration_markers = ["ugh", "really", "seriously", "again",
                               "fix", "wrong", "error", "not working"]
        anger_markers = ["angry", "mad", "terrible", "awful", "stupid"]
        urgency_markers = ["hurry", "quick", "asap", "emergency", "now"]

        text_lower = text.lower()

        if any(m in text_lower for m in anger_markers):
            return "angry"
        if any(m in text_lower for m in frustration_markers):
            return "frustrated"
        if any(m in text_lower for m in urgency_markers):
            return "urgent"

        return "neutral"

    def update(self, text: str, prosody: dict = None):
        """Update emotional state based on new input."""
        detected = self.detect_emotion(text, prosody)
        self.emotion_history.append(detected)

        if detected in ("frustrated", "angry"):
            self.frustration_level += 1
        else:
            self.frustration_level = max(0, self.frustration_level - 1)

        self.current_emotion = detected

    def should_escalate(self) -> bool:
        """Determine if user should be transferred to human."""
        return self.frustration_level >= 3
```

---

## 11. Voice UX Testing and Evaluation

### 11.1 Testing Methodologies

**a) Wizard of Oz Testing**
- Human operator simulates the agent
- Tests conversation design before AI implementation
- Captures natural user reactions
- Low cost, high insight

**b) Lab Usability Testing**
- Controlled environment
- Video/audio recording of user sessions
- Think-aloud protocol
- Detailed behavioral analysis

**c) Remote Moderated Testing**
- Users in their natural environment
- Moderator observes via screen/audio share
- More realistic than lab testing
- Can test with diverse populations

**d) Unmoderated Testing**
- Users interact independently
- Automated data collection
- Large sample sizes
- Good for quantitative metrics

**e) A/B Testing in Production**
- Compare two versions of conversation design
- Random assignment of users
- Measure key metrics (completion rate, satisfaction)
- Continuous optimization

### 11.2 Voice UX Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Task Completion Rate | % of tasks completed successfully | > 85% |
| Average Handling Time | Time from start to resolution | < 5 min |
| First Call Resolution | % resolved in single session | > 70% |
| ASR Error Recovery Rate | % of errors successfully recovered | > 90% |
| User Frustration Rate | % of sessions with frustration markers | < 10% |
| Abandonment Rate | % of users who hang up prematurely | < 5% |
| NPS (Net Promoter Score) | Would recommend to others | > 30 |
| CSAT (Customer Satisfaction) | Satisfaction rating | > 4.0/5.0 |
| Transfer Rate | % of sessions transferred to human | < 20% |
| Repeat Call Rate | % who call back about same issue | < 10% |

### 11.3 Voice UX Evaluation Checklist

```yaml
voice_ux_evaluation:
  conversation_design:
    - "Does the greeting set clear expectations?"
    - "Are prompts short and easy to understand?"
    - "Are options limited to 3-5 at a time?"
    - "Is error recovery graceful and helpful?"
    - "Is the closing smooth and natural?"
    
  persona_and_tone:
    - "Is the persona appropriate for the brand?"
    - "Is the tone consistent throughout?"
    - "Is the language inclusive and accessible?"
    - "Does the agent show appropriate empathy?"
    - "Is the AI nature disclosed clearly?"
    
  technical_performance:
    - "Is end-to-end latency under 500ms?"
    - "Does ASR accuracy meet thresholds?"
    - "Does barge-in work reliably?"
    - "Are timeouts set appropriately?"
    - "Does the agent handle silence gracefully?"
    
  user_experience:
    - "Can users complete common tasks easily?"
    - "Do users understand what to say?"
    - "Are errors handled without frustration?"
    - "Is the conversation natural and flowing?"
    - "Do users trust the agent?"
    
  accessibility:
    - "Does it work with speech impairments?"
    - "Are alternatives available for deaf users?"
    - "Is it usable by non-native speakers?"
    - "Are timeouts adjustable for cognitive needs?"
```

### 11.4 Conversation Analysis Framework

```python
class ConversationAnalytics:
    def __init__(self):
        self.sessions = []

    def analyze_session(self, transcript: list) -> dict:
        """Analyze a single conversation session."""
        turns = len(transcript)
        user_turns = [t for t in transcript if t["role"] == "user"]
        agent_turns = [t for t in transcript if t["role"] == "agent"]

        avg_user_utterance = self._avg_length(user_turns)
        avg_agent_utterance = self._avg_length(agent_turns)

        error_recoveries = self._count_error_recoveries(transcript)
        escalations = self._count_escalations(transcript)
        interruptions = self._count_interruptions(transcript)

        return {
            "session_id": transcript[0]["session_id"],
            "total_turns": turns,
            "user_turns": len(user_turns),
            "agent_turns": len(agent_turns),
            "avg_user_words_per_turn": avg_user_utterance,
            "avg_agent_words_per_turn": avg_agent_utterance,
            "error_recoveries": error_recoveries,
            "escalations": escalations,
            "interruptions": interruptions,
            "completed_task": self._task_completed(transcript),
            "user_sentiment": self._analyze_sentiment(transcript),
        }

    def _avg_length(self, turns: list) -> float:
        if not turns:
            return 0
        return sum(len(t["text"].split()) for t in turns) / len(turns)

    def _count_error_recoveries(self, transcript: list) -> int:
        # Count turns where agent handles recognition errors
        count = 0
        for turn in transcript:
            if turn["role"] == "agent":
                if any(p in turn["text"].lower()
                       for p in ["sorry", "didn't catch", "didn't understand",
                                 "rephrase", "say again"]):
                    count += 1
        return count

    def _count_escalations(self, transcript: list) -> int:
        return sum(1 for t in transcript
                   if "transfer" in t.get("text", "").lower()
                   and t["role"] == "agent")

    def _count_interruptions(self, transcript: list) -> int:
        # Simplified: count overlapping speech events
        return 0  # Requires timing data

    def _task_completed(self, transcript: list) -> bool:
        # Check if session has a closing that indicates completion
        last_turn = transcript[-1] if transcript else {}
        completion_markers = ["thank you", "goodbye", "bye",
                              "that's all", "nothing else"]
        text = last_turn.get("text", "").lower()
        return any(m in text for m in completion_markers)
```

---

## 12. Conversational Analytics

### 12.1 Key Analytics Dimensions

**a) Conversation Health Metrics**
- Completion rate by task type
- Average turns to completion
- Drop-off points in flow
- Common failure patterns

**b) User Behavior Metrics**
- Most common intents
- Peak usage times
- Session duration distribution
- Repeat usage patterns

**c) Quality Metrics**
- ASR confidence distribution
- NLU intent confidence
- TTS naturalness scores
- Barge-in frequency

**d) Business Metrics**
- Cost per conversation
- Human agent hours saved
- Conversion/outcome rates
- ROI calculation

### 12.2 Analytics Dashboard Template

```yaml
voice_analytics_dashboard:
  date_range: "2026-06-01 to 2026-06-07"
  
  summary:
    total_sessions: 15234
    completed_sessions: 12892
    completion_rate: 84.6%
    avg_handling_time: 3m 42s
    csat_score: 4.2/5.0
  
  trends:
    daily_sessions:
      mon: 2180
      tue: 2240
      wed: 2300
      thu: 2190
      fri: 2340
      sat: 1950
      sun: 2034
    
  top_intents:
    - intent: "check_balance"
      count: 4560
      completion_rate: 94%
    - intent: "make_payment"
      count: 3210
      completion_rate: 88%
    - intent: "account_help"
      count: 1890
      completion_rate: 76%
    - intent: "transaction_history"
      count: 1450
      completion_rate: 91%
  
  error_analysis:
    asr_errors: 342 (2.2%)
    nlu_errors: 156 (1.0%)
    fulfillment_errors: 89 (0.6%)
    escalation_rate: 12.4%
  
  user_feedback:
    positive: 4523
    neutral: 823
    negative: 345
    nps_score: 42
```

---

## 13. Voice UX Pattern Library

### 13.1 Core Interaction Patterns

**Confirmation Pattern:**
```
[State intent] → [Seek confirmation] → [Confirm or correct] → [Proceed]
```

**Error Recovery Pattern:**
```
[Error detected] → [Apologize] → [Rephrase] → [Try again] → [Escalate if needed]
```

**Multi-Step Pattern:**
```
[Step 1] → [Confirm] → [Step 2] → [Confirm] → ... → [Summary] → [Execute]
```

**Help Pattern:**
```
[Call for help] → [List capabilities] → [User selects] → [Proceed]
```

**Disambiguation Pattern:**
```
[Ambiguous request] → [Present options] → [User selects] → [Proceed]
```

**Handoff Pattern:**
```
[Agent identifies need for human] → [Explain why] → [Transfer] → [Human takes over]
```

### 13.2 Prompt Templates Library

```yaml
prompt_library:
  opening_greeting:
    new_user: "Hi, welcome to {company}! I'm {agent_name}. Before I assist you, could you please tell me your account number?"
    returning_user: "Welcome back! I'm {agent_name}. I see you last called about {last_topic}. How can I help today?"
  
  clarification:
    open: "Could you tell me more about what you need?"
    specific: "Did you mean {option_a} or {option_b}?"
    example: "I didn't understand that. You can say things like '{example_1}'."
  
  confirmation:
    simple: "So {action}. Is that correct?"
    detailed: "Let me confirm: {action_1}, {action_2}, and {action_3}. Is that right?"
  
  error:
    no_input_level1: "I didn't hear anything. Could you please say that again?"
    no_input_level2: "I'm still not hearing you. Please check your microphone and try again."
    no_match_level1: "I didn't understand that. Could you try saying it differently?"
    no_match_level2: "Let me offer some choices. You can say: {option_a} or {option_b}."
    timeout: "I'll disconnect now due to inactivity. Call back anytime you need help!"
  
  escalation:
    offer: "I'd like to connect you with a human specialist who can help with that. Is that okay?"
    transfer: "Please hold while I transfer you to a specialist."
    failed: "I'm having trouble connecting you. Please call back and ask for a specialist."
  
  closing:
    positive: "Glad I could help! Thanks for calling {company}. Have a great day!"
    unresolved: "I'm sorry I couldn't fully resolve this. A specialist will follow up. Thanks for your patience."
    quick: "Anything else? ... Thanks, bye!"
```

### 13.3 Earcon and Audio Cue Library

| Cue | Sound | Purpose |
|-----|-------|---------|
| Listening | Short chime (ascending) | Agent is ready for input |
| Processing | Subtle tone (neutral) | Agent is processing |
| Error | Descending tone | Something went wrong |
| Success | Ascending happy chord | Task completed |
| Transfer | Gentle hold music | Being transferred |
| Timeout | Double beep | Connection ending |
| Barge-in | Cutoff sound | Agent interrupted |

---

## 14. Regulatory and Ethical Considerations

### 14.1 Disclosure Requirements

- **AI Disclosure**: Must clearly state the user is interacting with AI
- **Recording Notice**: Inform if call is being recorded
- **Data Usage**: Explain how voice data is used and stored
- **Human Escalation**: Provide clear path to human agent
- **Opt-Out**: Allow users to opt out of AI interaction

### 14.2 Privacy and Data Protection

- Voice recordings contain biometric data (voiceprint)
- May be classified as sensitive personal data under GDPR
- Requires explicit consent for recording and processing
- Data minimization: only keep what's needed
- Right to erasure: users can request deletion
- Encryption at rest and in transit

### 14.3 Regulatory Compliance Checklist

```yaml
regulatory_compliance:
  gdpr:
    - "Explicit consent for voice recording"
    - "Data processing disclosure"
    - "Right to access recordings"
    - "Right to deletion"
    - "Data retention limits"
    - "DPIA completed"
    
  ccpa:
    - "Right to know what data is collected"
    - "Right to opt out of sale"
    - "Right to deletion"
    - "Notice at collection"
    
  ada:
    - "Accessible to users with disabilities"
    - "Reasonable accommodations"
    - "Alternative communication methods"
    
  ftc:
    - "Clear AI disclosure"
    - "No deceptive practices"
    - "Truthful claims about capabilities"
```

### 14.4 Ethical Design Principles

1. **Transparency**: Users always know they're talking to AI
2. **Safety**: Agent refuses harmful or unethical requests
3. **Fairness**: No bias in treatment of different users
4. **Privacy**: Minimize data collection, maximize security
5. **Accountability**: Clear who is responsible for agent actions
6. **Inclusivity**: Design for all users, not just typical ones
7. **User Control**: User can stop, correct, or leave at any time

---

## 15. References and Further Reading

- "Designing Voice User Interfaces" — Cathy Pearl (O'Reilly)
- "Voice User Interface Design" — Michael H. Cohen et al.
- "Conversational Design" — Erika Hall (A Book Apart)
- "The Conversational Interface" — Michael McTear et al.
- W3C Voice Interaction Standards — https://www.w3.org/standards/webofdevices/voice
- Google Conversation Design — https://designguidelines.withgoogle.com/conversation/
- Amazon Alexa Design Guide — https://developer.amazon.com/en-US/docs/alexa/alexa-design/design-guidelines.html
- Apple Siri Design Guidelines — https://developer.apple.com/siri/
- Microsoft Bot Framework Design Guidelines — https://learn.microsoft.com/en-us/azure/bot-service/bot-service-design-principles
- "Voice User Experience: Designing for the Ear" — Webinar by UXPA
- "Emotional Design for Voice Assistants" — ResearchGate
- "Accessibility in Voice Interfaces" — W3C WAI
- "The Ethics of Conversational AI" — Partnership on AI
- ISO 9241-11: Usability Framework
- ISO/IEC 25010: Software Quality Model for Voice Systems
- NN/g Voice UX Research Articles — https://www.nngroup.com/topic/voice-user-interfaces/
