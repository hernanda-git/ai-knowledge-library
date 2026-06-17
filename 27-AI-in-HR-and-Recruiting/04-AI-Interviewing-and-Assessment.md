# 04 — AI Interviewing and Assessment

## Overview

AI has moved from screening and sourcing into the interview and assessment stage, arguably the most sensitive part of the hiring process. AI-powered interviewing spans automated screening interviews (chatbots and voice bots), technical skill assessments (coding challenges, work sample tests), and asynchronous video interviews with AI analysis of facial expressions, tone, and word choice.

This document covers the full spectrum of AI in interviewing and assessment, including the technology stack, tools, implementation patterns, accuracy benchmarks, and — critically — the substantial ethical concerns and regulatory requirements that apply.

---

## 1. The AI Interviewing Spectrum

| Stage | AI Role | Human Involvement | Risk Level |
|---|---|---|---|
| Pre-screening chatbot | Answer FAQs, schedule interviews, collect basic info | Low — automated | Low |
| Automated phone screen | Voice bot asks structured questions, transcribes answers | Low — automated | Medium |
| Asynchronous video interview | Candidate records responses; AI analyses speech, tone, facial expression | AI scores, human reviews | High |
| Live AI-assisted interview | AI provides real-time suggestions/questions to human interviewer | High — human-led | Low |
| Technical assessment | AI generates problems, evaluates code quality, detects cheating | Low — automated | Medium |
| Skills simulation | AI creates realistic work scenario, evaluates performance | AI scores, human calibrates | Medium |
| Automated reference check | AI bot calls references, extracts structured feedback | Low — automated | Low |

---

## 2. Automated Screening Interviews

### 2.1 Chatbot Pre-Screening

AI chatbots conduct initial screening conversations with candidates via text (web, SMS, WhatsApp, Slack).

**Common screening questions:**
- Current employment status and notice period
- Salary expectations (range)
- Work authorisation / visa status
- Availability for onsite/remote/hybrid
- Willingness to travel
- Years of experience in required skill area
- Highest education level attained

**Architecture:**
```
[Candidate Message]
       │
       ▼
┌─────────────────────┐
│ NLU / Intent Module │
│ (Rasa, Dialogflow,  │
│  LangChain + LLM)   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Dialogue Manager    │
│ (State machine +    │
│  slot filling)      │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Response Generator  │
│ (Template / LLM)   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Integration Layer   │
│ ──→ ATS (candidate  │
│     created/updated)│
│ ──→ Calendar (screen│
│     scheduled)      │
└─────────────────────┘
```

**Tools:** Rasa, Google Dialogflow CX, AWS Lex, LangChain + LLM, Intercom, Drift.

### 2.2 Voice Screening Bots

AI voice bots conduct phone screens, asking questions and parsing natural language responses.

**Capabilities:**
- Text-to-speech (ElevenLabs, Play.ht, Amazon Polly)
- Speech-to-text (Whisper, Deepgram, Google Speech-to-Text)
- NLP for response understanding and scoring
- Dynamic follow-up questions based on answers
- Multi-language support

**Vendors:** Mya (Mya Systems), XOR, Ideal, Paradox (Olive), AllyO (now a trade name).

**Limitations:**
- Accent recognition disparity (lower accuracy for non-native speakers)
- Speech impediment / stutter — misclassified as hesitation or uncertainty
- Background noise sensitivity
- Cannot handle complex or unexpected questions well

---

## 3. Technical Skill Assessment

### 3.1 AI-Generated Coding Challenges

AI generates custom coding problems tailored to the specific role requirements.

**Generation approaches:**
| Approach | How It Works | Quality |
|---|---|---|
| Template-based | Parameterised problem templates (difficulty, topic, constraints) | Good for standardised roles |
| LLM-generated | GPT-4 / Claude prompted with skill requirements | High variety, needs validation |
| Hybrid | LLM generates → human/automated validator checks | Best quality |

**Distinctive features:**
- Role-specific — "Build a REST API for a payment system" for backend roles
- Plagiarism detection — MOSS (Stanford), Codequiry, GPTZero for AI-written code detection
- Real-time IDE — CoderPad, CodeSignal, HackerRank provide browser-based coding environments
- Proctoring — Screen recording, webcam monitoring, keystroke analysis for cheating detection

### 3.2 Assessment Platforms

| Platform | Question Types | AI Features | Languages |
|---|---|---|---|
| **HackerRank** | Coding, SQL, regex, multiple choice | Plagiarism detection, AI-powered problem generation | 40+ languages |
| **CodeSignal** | Coding, tasks, frameworks | Custom assessment builder, cheating detection, real-time IDE | 30+ languages |
| **Codility** | Coding, algorithmic, SQL | Plagiarism detection, fairness analysis | 20+ languages |
| **CoderPad** | Live coding, take-home | AI interview assistant, code analysis | 30+ languages |
| **TestGorilla** | Skill tests, personality, cognitive | AI scoring, anti-cheating | Multi-skill |
| **HireRight / Checkr** | Background-integrated, skills | AI verification | N/A |
| **Codewars** | Katas (graded challenges) | Skill ranking via ELO system | 55+ languages |
| **Qualified** | Real-world scenarios (framework-specific) | Behavioural analysis, code review simulation | 10+ |

### 3.3 Code Evaluation Metrics

AI evaluation of code submissions goes beyond "does it pass tests":

| Metric | Description | Weight |
|---|---|---|
| **Correctness** | Passes all test cases, handles edge cases | 40% |
| **Time complexity** | Big-O analysis of algorithm | 15% |
| **Space complexity** | Memory usage analysis | 10% |
| **Code quality** | Readability, naming conventions, structure | 10% |
| **Testing** | Whether candidate wrote their own tests | 5% |
| **Problem-solving approach** | Efficiency of algorithm choice | 10% |
| **Communication** | Comments, explanation in code | 5% |
| **Language proficiency** | Idiomatic use of language features | 5% |

### 3.4 AI-Assisted Code Review

For take-home assessments, AI can conduct a preliminary code review:
- Static analysis (linting, type checking)
- Security vulnerability scanning
- Performance optimisation suggestions
- Architecture pattern identification

---

## 4. Video Interview Analysis

### 4.1 The Video Interview Pipeline

Asynchronous video interviews involve the candidate recording responses to pre-set questions. AI analyses three modality streams:

```
┌──────────────────────────────────────────────────┐
│                 Video Recording                   │
└────────────┬──────────────┬──────────────┬───────┘
             │              │              │
             ▼              ▼              ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │ Visual       │ │ Audio        │ │ Speech       │
    │ (Facial expr,│ │ (Tone, pitch,│ │ (Transcript) │
    │  gaze,       │ │  pace,       │ │ (content,    │
    │  posture)    │ │  pauses)     │ │  sentiment)  │
    └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
           │                │                │
           └────────────────┼────────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │ Multimodal       │
                   │ Fusion Model     │
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │ Scores:          │
                   │ Communication   │
                   │ Enthusiasm      │
                   │ Competence      │
                   │ Culture fit     │
                   │ Honesty/risk    │
                   └─────────────────┘
```

### 4.2 Facial Expression Analysis

**What it measures:**
- Microexpressions (happiness, surprise, anger, fear, disgust, sadness, contempt)
- Engagement level (looking at camera vs looking away)
- Eye contact duration and frequency
- Smiling frequency and intensity
- Head nodding/shaking
- Blink rate (can indicate stress)

**Technical approach:**
| Component | Tool | Accuracy |
|---|---|---|
| Face detection | MediaPipe, OpenCV, MTCNN | > 99% |
| Landmark detection | MediaPipe Face Mesh (468 points), Dlib | ~ 98% |
| Expression classification | FER+, AffectNet-trained CNN | ~ 90–93% |
| Gaze tracking | Eye movement vectors from landmarks | ~ 85–90% |
| Action Units (AU) | OpenFace 2.0 — 18 AUs | ~ 95% |

### 4.3 Voice and Tone Analysis

**Paralinguistic features extracted:**
- **Pitch (F0)** — Fundamental frequency variation, indicates excitement vs monotony
- **Speaking rate** — Words per minute, pace variation
- **Pauses** — Filled (um, uh, like) vs silent pauses; duration
- **Volume** — Loudness variation, confidence inference
- **Jitter/shimmer** — Vocal fold stability, stress indicator
- **Energy** — Spectral energy distribution, engagement level

**Tools:** openSMILE (feature extraction), Praat, pyAudioAnalysis, Coqui STT for custom models.

### 4.4 Speech Content Analysis

**NLP analysis of transcribed responses:**
| Feature | Measure |
|---|---|
| Sentiment | Positive/negative/neutral tone of answers |
| Topic relevance | How well answer addresses the question |
| Specificity | Concrete examples vs vague generalisations |
| Self-promotion vs humility | Ratio of "I" vs "we" statements |
| Structure | Clear STAR format (Situation, Task, Action, Result) |
| Key phrase detection | Leadership indicators, conflict resolution language |
| Jargon/terminology | Appropriate domain vocabulary use |
| Cognitive load | Filler words ("like", "you know"), hedging ("sort of", "kind of") |

### 4.5 Multimodal Fusion

Individual modalities (face, voice, text) are combined using:

| Fusion Strategy | Description |
|---|---|
| **Early fusion** | Concatenate features before prediction — requires aligned data |
| **Late fusion** | Average/vote individual predictions — simple, robust |
| **Attention fusion** | Learn which modality to weight per response — best accuracy |
| **Hierarchical fusion** | Text first (content), then modulate by tone (delivery), then adjust by expression (engagement) |

### 4.6 Commercial Video Interview Platforms

| Platform | AI Features | Analysis Type |
|---|---|---|
| **HireVue** | Facial expression, tone, word choice, job fit score | Visual + audio + text |
| **SparkHire** | Engagement prediction, structured scoring | Text + basic audio |
| **MyInterview** | Personality traits, communication skills, values | Text + audio + visual |
| **vidCruiter** | Competency scoring, sentiment, keyword analysis | Text + audio |
| **Willo** | Asynchronous video, AI fairness scoring | Text + basic analysis |
| **Cangrade** | Cognitive ability, personality, job fit | Contextual (not video) |
| **Pymetrics** | Neuroscience games, not video | Behavioural |

---

## 5. Structured Question Generation

### 5.1 AI-Generated Interview Questions

AI can generate structured, competency-based interview questions tailored to:
- The specific job description
- The candidate's resume
- The organisation's competency framework
- Prior interview performance data

**By Interview Stage:**
| Stage | Example AI-Generated Questions |
|---|---|
| Screening | "Tell me about a time you had to debug a production issue under time pressure." |
| Technical | "Design a rate-limiting system that handles 10K requests/second." |
| Behavioural | "Describe a situation where you disagreed with your manager's technical approach." |
| Situational | "If you joined our team and found the CI/CD pipeline was taking 45 minutes, what would you do?" |
| Values-based | "How do you ensure code quality when deadlines are tight?" |

### 5.2 Competency-Based Question Banks

AI maintains a question bank tagged by:
- **Competency** (problem-solving, teamwork, leadership, communication, technical depth)
- **Seniority level** (junior, mid, senior, staff, principal)
- **Role type** (engineer, PM, designer, data scientist)
- **Difficulty** (1–5 scale)
- **Diversity & inclusion** (ensuring questions don't accidentally bias)

### 5.3 Dynamic Question Selection

Based on candidate responses, AI selects follow-up questions adaptively:
```
Candidate: "I resolved the outage by rolling back the deployment."
AI follow-up: "What did your post-mortem analysis reveal about the root cause?"
Candidate: "We didn't do a post-mortem."
AI follow-up: "If you could go back, what process would you put in place?"
```

This adaptive approach reveals deeper layers of a candidate's experience and reasoning.

---

## 6. Scoring and Decision Support

### 6.1 Automated Scoring Models

AI assessment platforms typically output scores on multiple dimensions:

| Dimension | Example Metric | Weight |
|---|---|---|
| Technical competence | Correctness, efficiency, code quality | 35% |
| Problem-solving | Approach, structure, handling ambiguity | 20% |
| Communication | Clarity, conciseness, listening | 15% |
| Culture alignment | Values matching, collaboration signals | 10% |
| Motivation | Interest in role, company, mission | 10% |
| Growth potential | Learning ability, adaptability | 10% |

### 6.2 Scoring Calibration

Raw AI scores need calibration against human judgment:

**Calibration techniques:**
1. **Human-AI correlation**: Compare AI scores with interviewer scores on the same candidates
2. **Calibration sets**: 50–100 candidates with known outcomes (hire quality, performance ratings at 6 months)
3. **Score normalisation**: Adjust for interview difficulty (easier platform = lower average scores)
4. **Rater consistency**: AI scores are inherently more consistent than human raters (inter-rater reliability κ = 0.95 for AI vs κ = 0.45 for humans)
5. **Fairness adjustment**: Across demographic groups, ensure scores are not systematically different

### 6.3 Decision Framework

```
AI Score: 85/100
Human Interview Score: 78/100
Combined Score: (85 × 0.4) + (78 × 0.6) = 80.8/100

Decision bands:
  < 60: Reject
  60–74: Borderline (second look or different role)
  75–89: Proceed to next round
  ≥ 90: Fast-track to offer
```

---

## 7. Cheating Detection and Integrity

### 7.1 Assessment Integrity Features

| Method | What It Detects |
|---|---|
| **Copy-paste detection** | Code copied from external sources |
| **Keystroke dynamics** | Typing pattern changes (different person typing) |
| **Screen recording** | Multiple monitors, external devices |
| **Webcam monitoring** | Gaze away from screen, other people present |
| **Browser tab focus** | Switching to other tabs/windows |
| **Audio environment** | Voices in background, reading aloud |
| **AI-generated code detection** | GPTZero-style classifiers for code |
| **IP geolocation** | Candidate location vs claimed location |
| **Timing analysis** | Unrealistically fast or slow |
| **Plagiarism corpus matching** | Search engine / code repository lookup |

### 7.2 False Positive Risks

Cheating detection systems can produce false positives that disproportionately affect:
- Candidates with disabilities (motor impairments → keystroke anomalies)
- Non-native speakers (longer pauses, looking away to think)
- Neurodivergent candidates (atypical eye contact, speech patterns)
- Candidates in shared workspaces (co-working spaces → multiple voices)
- Candidates with slow internet (video/audio desync → analysis errors)

**Mitigation:** Always provide a clear appeal process. Never make automated rejection decisions based solely on cheating detection flags.

---

## 8. Bias and Fairness Concerns

### 8.1 Sources of Bias in AI Interviewing

| Source | Example | Impact |
|---|---|---|
| Training data | Models trained on successful incumbents (historically majority group) | Perpetuates homogeneity |
| Facial analysis | Models less accurate for darker skin tones (FERV39k dataset bias) | Higher uncertainty scores for BIPOC candidates |
| Accent bias | Lower NLP accuracy for non-native English speakers | Lower content scores unfairly |
| Gender voice bias | Deeper voices perceived as more authoritative (tone models) | Lower scores for higher-pitched voices |
| Neurodiversity | Lack of eye contact = "disengaged" (autistic candidates) | Lower engagement scores |
| Cultural norms | Smiling not universal expression of enthusiasm | Cultural false negatives |
| Speech pace | Fast talkers = "confident", slow = "uncertain" | Unfair to deliberate speakers |
| Physical appearance | Attractiveness bias — AI mirrors human bias from training | Unrelated to job performance |

### 8.2 Documented Cases of Bias

**HireVue (2021):**
- Investigation by the UK Equality and Human Rights Commission
- Face and voice analysis raised concerns about indirect discrimination
- HireVue subsequently dropped facial analysis for most customers in 2022
- Now focuses on speech content analysis (less risky)

**Amazon's AI Recruiting Tool (2018):**
- Penalised resumes mentioning "women's" (college, clubs)
- Trained on 10 years of Amazon applications (mostly male)
- Project disbanded after bias could not be eliminated

**University studies (2022–2024):**
- F1 score for speech-to-text: 0.85 for native English, 0.62 for Indian English, 0.58 for Nigerian English
- Expression recognition: 93% accuracy for white males, 78% for Black females
- "Confidence scoring" correlated with height (taller = higher score)

### 8.3 Bias Mitigation Strategies

**Technical mitigations:**
1. **Only use text-based analysis** — Drop video/audio analysis entirely (less predictive of performance, higher bias risk)
2. **Domain adaptation** — Fine-tune models on diverse datasets with balanced demographic representation
3. **Adversarial debiasing** — Train facial models to be invariant to skin tone
4. **Equalised odds post-processing** — Adjust thresholds so false positive rates are equal across groups
5. **Counterfactual evaluation** — Test whether swapping name/voice/accent changes the score
6. **Blind scoring** — Audio-only, video-off for content scoring

**Process mitigations:**
1. **Human review of AI-ranked candidates** — AI provides decision support, not decisions
2. **Bias audit every 6 months** — Demographic analysis of score distributions
3. **Candidate appeals process** — Right to human re-evaluation
4. **Transparency** — Tell candidates which AI tools are used and what data is collected
5. **Opt-out** — Provide a non-AI alternative assessment path

### 8.4 Regulatory Compliance for Interview AI

| Jurisdiction | Key Requirement | Impact |
|---|---|---|
| NYC Local Law 144 | Mandatory bias audit of automated employment decision tools | Applies to video interview AI used for NYC candidates |
| Illinois AI Hiring Law | Notice + consent for AI video interviews | Must inform candidates about AI analysis |
| EU AI Act | High-risk classification → conformity assessment, transparency, human oversight | Full documentation and audit trail required |
| GDPR (Article 22) | Right to not be subject to solely automated decisions | Cannot auto-reject based on AI interview score alone |
| California CCPA | Right to know what data was collected and how it was used | Data inventory for interview recordings |

---

## 9. Implementation Best Practices

### 9.1 Validation Framework

Before deploying AI interviewing at scale:

1. **Criterion validity study**: Does AI score correlate with on-the-job performance (manager ratings at 6 months)?
2. **Fairness audit**: Disparate impact analysis across gender, race, age, and intersectional groups
3. **Reliability study**: Test-retest consistency (same candidate, different AI session)
4. **Human-AI agreement**: How often does AI agree with trained interviewer panel?
5. **Candidate experience survey**: Do candidates feel the process was fair?

### 9.2 Candidate Communication Template

```
[When scheduling the AI interview:]

"This assessment uses AI to help evaluate your responses. Here's what that means:

1. Your responses will be recorded and analysed for content and communication patterns.
2. No facial recognition or emotion analysis is used in scoring.
3. A human recruiter will review every assessment result before any decisions are made.
4. You have the right to request a human-only interview process instead.
5. Your data will be deleted [after 60 days / within 6 months].

If you have any concerns, please contact [recruiter/HR]."
```

### 9.3 Human-in-the-Loop Requirements

| Decision | AI Alone | AI + Human Review | Human Only |
|---|---|---|---|
| Schedule interview | ✓ (with explicit opt-out) | ✓ (recommended) | — |
| Reject candidate | ✗ | ✓ | ✗ |
| Score communication skills | — | ✓ | — |
| Score technical ability | ✓ (verified) | ✓ (recommended) | — |
| Recommend hire/no-hire | ✗ | ✗ | ✓ |
| Make final offer decision | ✗ | ✗ | ✓ |

---

## 10. Future Directions

### 10.1 Live AI Interview Co-Pilot

Real-time AI assistance for human interviewers:
- Suggested follow-up questions based on candidate's answer
- Reminder to cover all competency areas
- Non-verbal cues that the candidate might be confused or disengaged
- Note-taking and structured summary generation
- Bias alert: "You've spent 80% of the interview on technical questions and 20% on behavioural — consider balancing"

### 10.2 Skill Simulation with Generative AI

- Create dynamic, branching role-play scenarios
- "You're a product manager: the engineering team just told you a feature will be delayed by 2 weeks. What do you do?"
- AI plays the role of stakeholders, engineers, customers
- Evaluates candidate's reasoning, communication, and decision-making in real-time

### 10.3 Voice-Only Interviews with Emotion-Neutral Scoring

- Drop visual analysis entirely
- Use only speech-to-text for content scoring
- Normalise for speech pace, accent, and hesitation
- Focus on what is said, not how it is delivered

### 10.4 Standardised AI Interview Benchmarks

Industry-wide benchmarks for AI interview tools:
- **Interview Quality Metric (IQM)** — How well does AI predict on-the-job performance?
- **Fairness Score** — Equalised odds across demographic groups
- **Candidacy Acceptance Rate** — % of candidates willing to use AI interviews
- **Predictive Validity** — Correlation with performance at 6 and 12 months

---

## References

- M. Raghavan et al., "Mitigating Bias in Algorithmic Hiring," ACM FAccT, 2020.
- A. V. Borden et al., "Automatic Analysis of Asynchronous Video Interviews: A Systematic Literature Review," ACM Computing Surveys, 2023.
- UK Equality and Human Rights Commission, "Investigation into HireVue's AI Assessment Tools," 2021.
- S. U. Malik et al., "Accent Bias in Automated Speech Recognition Systems," Interspeech, 2022.
- J. Buolamwini and T. Gebru, "Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification," ACM FAccT, 2018.
- HireVue, "AI Ethics Statement and Technical Report on Algorithmic Fairness," 2022.
- NYC Department of Consumer and Worker Protection, "Local Law 144: Automated Employment Decision Tools," Final Rule, 2023.
- European Commission, "EU AI Act: High-Risk AI Systems Requirements," 2023.
- P. Liang et al., "Holistic Evaluation of Language Models (HELM)," arXiv:2211.09110, 2022.
- J. Otterbacher et al., "The Media Inequality in AI: How Computer Vision Models Reflect Social Biases," PNAS, 2022.
