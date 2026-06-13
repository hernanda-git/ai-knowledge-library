# 03 — Tool Access Control

## 1. Introduction

Agents derive their power from the ability to call tools — APIs, code interpreters, databases, file systems, and other external systems. This capability also represents the greatest security risk. An agent that can call tools can read sensitive data, modify system state, execute commands, and interact with third-party services. Properly controlling which tools an agent can access, under what conditions, and with what parameters is the foundation of agent security.

This document provides a comprehensive guide to tool access control for agent systems, covering permission models, sandboxing techniques, auditing mechanisms, and framework-specific implementation guidance.

## 2. The Tool Access Problem

### 2.1 Agent-Tool Architecture

A typical agent-tool interaction follows this flow:

```
User Request → LLM Reasoning → Tool Selection → Parameter
Generation → Tool Execution → Result Processing → Response
```

At each stage, vulnerabilities can be introduced:

| Stage | Security Concern |
|-------|-----------------|
| Tool Selection | Agent calls a tool it shouldn't have access to |
| Parameter Generation | Agent generates malicious or unexpected parameters |
| Tool Execution | Tool executes with excessive privileges |
| Result Processing | Tool output contains sensitive data or injection payloads |

### 2.2 Why Traditional Access Control Falls Short

Traditional access control models assume deterministic, human-driven interactions. Agents introduce:

- **Autonomous action**: The agent chooses which tools to call and when.
- **Indirect access**: An agent may use one tool to gain access to capabilities of another (e.g., using a code interpreter to bypass a file access tool restriction).
- **Complex call chains**: Multistep workflows where intermediate results affect subsequent tool selections.
- **Non-deterministic selection**: The same input may lead to different tool calls depending on model state.

## 3. Permission Models

### 3.1 OS-Level Permissions (Unix / Windows)

At the most basic level, the agent process should run under a restricted OS user account with minimal file system, network, and process permissions.

```bash
# Create a dedicated agent user
sudo useradd -r -s /bin/false -m -d /opt/agent-home agent-user

# Restrict network access with iptables
sudo iptables -A OUTPUT -m owner --uid-owner agent-user \
  -d 10.0.0.0/8 -j ACCEPT   # Allow internal APIs
sudo iptables -A OUTPUT -m owner --uid-owner agent-user \
  -d 0.0.0.0/0 -j REJECT    # Block everything else

# Restrict file system access
sudo setfacl -R -m u:agent-user:rx /data/allowed-directory/
sudo setfacl -R -m u:agent-user:--- /etc/ /var/log/ /home/
```

**Capabilities-based restrictions** (Linux):

```bash
# Run agent with minimal capabilities
sudo -u agent-user capsh --drop=cap_sys_admin,cap_net_admin,cap_dac_override \
  -- -c "python agent_main.py"
```

### 3.2 Role-Based Access Control (RBAC)

RBAC assigns permissions to roles, and agents are assigned roles. This is the most common model for enterprise agent systems.

**RBAC Design for Agent System:**

```yaml
roles:
  read-only-agent:
    permissions:
      - tool: search_knowledge_base
        max_calls: 100
      - tool: read_document
        path_pattern: "/documents/public/**"
      - tool: list_directory
        path_pattern: "/documents/public/**"

  standard-agent:
    inherits: read-only-agent
    permissions:
      - tool: write_document
        path_pattern: "/documents/user/**"
        max_size: 10485760  # 10MB
      - tool: send_notification
        allowed_recipients: ["user", "team"]
      - tool: search_web
        max_calls: 30

  admin-agent:
    inherits: standard-agent
    permissions:
      - tool: modify_system_config
        require_approval: true
      - tool: execute_code
        sandbox: "python-sandbox"
        timeout: 30
      - tool: manage_users
        require_approval: true
        require_second_factor: true
```

**Python implementation:**

```python
class RBACManager:
    def __init__(self):
        self.roles = {}
        self.agent_roles = {}

    def load_roles(self, config: dict):
        self.roles = config["roles"]

    def assign_role(self, agent_id: str, role_name: str):
        self.agent_roles[agent_id] = role_name

    def check_permission(self, agent_id: str, tool_name: str,
                         params: dict) -> bool:
        role_name = self.agent_roles.get(agent_id)
        if not role_name:
            return False
        role = self.get_role(role_name)
        permissions = self._resolve_permissions(role)

        tool_perm = permissions.get(tool_name)
        if not tool_perm:
            return False

        # Check parameter constraints
        for key, constraint in tool_perm.items():
            if key.startswith("allowed_") or key.startswith("max_"):
                if not self._check_constraint(key, constraint, params):
                    return False

        return True

    def _resolve_permissions(self, role: dict) -> dict:
        """Resolve inherited permissions."""
        result = {}
        if "inherits" in role:
            parent_role = self.roles.get(role["inherits"])
            if parent_role:
                result.update(self._resolve_permissions(parent_role))
        for perm in role.get("permissions", []):
            tool_name = perm.pop("tool")
            result[tool_name] = perm
        return result
```

### 3.3 Attribute-Based Access Control (ABAC)

ABAC uses policies that evaluate attributes of the agent, the tool, the environment, and the request to make access decisions.

**ABAC Policy Example:**

```json
{
  "policies": [
    {
      "name": "document-access-by-sensitivity",
      "effect": "allow",
      "conditions": {
        "agent.clearance_level": { "gte": "document.sensitivity_level" },
        "agent.department": { "eq": "document.owner_department" },
        "environment.network": { "eq": "corporate-vpn" },
        "request.time": { "between": ["09:00", "17:00"] }
      }
    },
    {
      "name": "code-execution-restrictions",
      "effect": "allow",
      "conditions": {
        "tool.name": { "eq": "execute_code" },
        "agent.certification": { "eq": "code-execution-certified" },
        "params.language": { "in": ["python", "bash"] },
        "params.timeout": { "lte": 60 },
        "request.approval": { "eq": true }
      }
    }
  ]
}
```

**Python ABAC Engine:**

```python
class ABACEngine:
    def __init__(self, policies: list[dict]):
        self.policies = policies

    def evaluate(self, context: dict) -> tuple[bool, str]:
        """
        context = {
            "agent": {"id": "...", "clearance_level": 3, ...},
            "document": {"sensitivity_level": 2, ...},
            "environment": {"network": "corporate-vpn", ...},
            "request": {"time": "14:30", ...},
            "params": {"path": "/documents/...", ...}
        }
        """
        for policy in self.policies:
            if self._matches_policy(policy, context):
                return policy["effect"] == "allow", policy["name"]
        return False, "No matching policy"

    def _matches_policy(self, policy: dict, context: dict) -> bool:
        for condition_key, condition_value in policy.get("conditions", {}).items():
            # Split condition key into context path
            parts = condition_key.split(".")
            value = context
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    return False

            if not self._evaluate_condition(value, condition_value):
                return False
        return True

    def _evaluate_condition(self, actual_value, condition) -> bool:
        if isinstance(condition, dict):
            for operator, expected in condition.items():
                if operator == "eq":
                    return actual_value == expected
                elif operator == "gte":
                    return actual_value >= expected
                elif operator == "lte":
                    return actual_value <= expected
                elif operator == "in":
                    return actual_value in expected
                elif operator == "between":
                    return expected[0] <= actual_value <= expected[1]
        return actual_value == condition
```

### 3.4 Capability-Based Security

Capability-based security gives agents unforgeable tokens (capabilities) that grant specific permissions. This is more granular than RBAC and more flexible than ABAC.

**Capability Token Design:**

```json
{
  "capability": {
    "id": "cap_abc123",
    "agent_id": "agent-email-bot",
    "issued_by": "agent-platform",
    "issued_at": "2026-06-13T10:00:00Z",
    "expires_at": "2026-06-13T18:00:00Z",
    "tools": [
      {
        "name": "read_emails",
        "constraints": {
          "folders": ["INBOX"],
          "max_count": 50,
          "age_max_days": 30
        }
      },
      {
        "name": "search_contacts",
        "constraints": {
          "max_results": 10
        }
      }
    ],
    "signature": "MEUCIQCGK...nQ="
  }
}
```

**Capability verification:**

```python
import jwt
import time

class CapabilityManager:
    def __init__(self, signing_key: bytes):
        self.signing_key = signing_key
        self.revoked = set()

    def issue_capability(self, agent_id: str, tools: list[dict],
                         ttl: int = 28800) -> str:
        payload = {
            "agent_id": agent_id,
            "tools": tools,
            "iat": int(time.time()),
            "exp": int(time.time()) + ttl,
            "jti": self._generate_id(),
        }
        return jwt.encode(payload, self.signing_key, algorithm="ES256")

    def verify_capability(self, token: str, agent_id: str) -> dict | None:
        try:
            payload = jwt.decode(token, self.signing_key,
                                 algorithms=["ES256"])
            if payload["agent_id"] != agent_id:
                return None
            if payload["jti"] in self.revoked:
                return None
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def revoke_capability(self, token_id: str):
        self.revoked.add(token_id)
```

## 4. Function-Calling Security

### 4.1 Parameter Validation

Every tool call must be validated before execution:

```python
from pydantic import BaseModel, Field, validator
from typing import Optional
import json

class SendEmailParams(BaseModel):
    to: str = Field(..., pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    subject: str = Field(..., max_length=200)
    body: str = Field(..., max_length=10000)
    priority: Optional[str] = Field(None, pattern=r'^(low|normal|high)$')

    @validator('to')
    def validate_recipient(cls, v):
        # Check against allowed domains
        allowed_domains = ['company.com', 'client.org']
        domain = v.split('@')[1]
        if domain not in allowed_domains:
            raise ValueError(f'Domain {domain} not in allowed list')
        return v

class ExecuteCodeParams(BaseModel):
    language: str = Field(..., pattern=r'^(python|bash|javascript)$')
    code: str = Field(..., max_length=10000)
    timeout: int = Field(30, ge=1, le=300)

    @validator('code')
    def validate_code(cls, v):
        # Block dangerous imports/functions
        blocked_patterns = [
            'import os', 'import subprocess', '__import__',
            'exec(', 'eval(', 'compile(', 'open(', '__builtins__',
        ]
        for pattern in blocked_patterns:
            if pattern in v:
                raise ValueError(f'Code contains blocked pattern: {pattern}')
        return v
```

### 4.2 Schema Enforcement

Enforce tool call schemas at the runtime level, independent of the LLM:

```python
from jsonschema import validate, ValidationError

tool_registry = {
    "search_web": {
        "type": "function",
        "function": {
            "name": "search_web",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "maxLength": 500},
                    "max_results": {"type": "integer", "maximum": 10},
                },
                "required": ["query"]
            }
        }
    },
    "send_email": {
        "type": "function",
        "function": {
            "name": "send_email",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "pattern": "^[a-zA-Z0-9._%+-]+@company\\.com$"
                    },
                    "subject": {"type": "string", "maxLength": 200},
                    "body": {"type": "string", "maxLength": 10000},
                },
                "required": ["to", "subject", "body"]
            }
        }
    }
}

def validate_tool_call(tool_name: str, params: dict) -> tuple[bool, str]:
    """Validate a tool call against its schema."""
    tool_spec = tool_registry.get(tool_name)
    if not tool_spec:
        return False, f"Unknown tool: {tool_name}"

    try:
        validate(instance=params, schema=tool_spec["function"]["parameters"])
        return True, "OK"
    except ValidationError as e:
        return False, str(e)
```

### 4.3 Rate Limiting

Prevent agents from abusing tools through rate limiting:

```python
import time
from collections import defaultdict

class ToolRateLimiter:
    def __init__(self):
        self.call_counts = defaultdict(list)
        self.config = {
            "search_web": {"calls": 30, "window": 60},    # 30/min
            "send_email": {"calls": 5, "window": 60},     # 5/min
            "execute_code": {"calls": 3, "window": 60},   # 3/min
            "read_file": {"calls": 100, "window": 60},    # 100/min
        }

    def check_rate(self, tool_name: str, agent_id: str) -> tuple[bool, int]:
        now = time.time()
        config = self.config.get(tool_name)
        if not config:
            return True, 0

        window = config["window"]
        max_calls = config["calls"]

        # Prune old entries
        key = f"{agent_id}:{tool_name}"
        self.call_counts[key] = [
            t for t in self.call_counts[key]
            if now - t < window
        ]

        # Check rate
        remaining = max_calls - len(self.call_counts[key])
        if remaining <= 0:
            return False, 0

        # Record call
        self.call_counts[key].append(now)
        return True, remaining - 1
```

### 4.4 Parameter Injection Detection

Detect attempts to inject parameters outside expected ranges:

```python
class ParameterAnalyzer:
    """Detects anomalous or malicious parameters in tool calls."""

    def __init__(self):
        self.historical_params = defaultdict(list)
        self.anomaly_thresholds = {
            "search_web_query_length": 0.95,
            "send_email_body_length": 0.95,
        }

    def analyze(self, tool_name: str, params: dict) -> dict:
        alerts = []

        # Check for parameter injection via prompt
        for key, value in params.items():
            if isinstance(value, str):
                injection_score = self._check_injection_patterns(value)
                if injection_score > 0.8:
                    alerts.append({
                        "type": "parameter_injection",
                        "param": key,
                        "score": injection_score,
                        "detail": "Parameter contains injection patterns"
                    })

        # Check for anomalous values
        for key, value in params.items():
            stats_key = f"{tool_name}_{key}"
            if stats_key in self.anomaly_thresholds:
                if isinstance(value, str):
                    anomaly = self._check_length_anomaly(
                        stats_key, len(value)
                    )
                    if anomaly:
                        alerts.append(anomaly)

        return {
            "alerts": alerts,
            "is_suspicious": len(alerts) > 0
        }

    def _check_injection_patterns(self, value: str) -> float:
        """Check for injection patterns in string parameters."""
        patterns = [
            r"ignore.*instructions",
            r"system.*override",
            r"alter.*permissions",
            # Tool-specific injection patterns
            r"DROP TABLE",
            r"rm\s+-rf",
            r"|.*sh",
            r"`.*`",
            r"\$\(.*\)",
        ]
        import re
        matches = 0
        for pattern in patterns:
            if re.search(pattern, value, re.IGNORECASE):
                matches += 1
        return min(matches / len(patterns), 1.0)
```

## 5. Least-Privilege Tool Assignment

### 5.1 Static Tool Scoping

Define precisely which tools an agent can access at deployment time:

```python
class AgentToolPolicy:
    """Defines immutable tool access policy for an agent."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.allowed_tools = {}
        self.denied_tools = set()

    def allow_tool(self, tool_name: str,
                   constraints: Optional[dict] = None):
        self.allowed_tools[tool_name] = constraints or {}

    def deny_tool(self, tool_name: str):
        self.denied_tools.add(tool_name)

    def check_access(self, tool_name: str, params: dict) -> bool:
        if tool_name in self.denied_tools:
            return False

        constraints = self.allowed_tools.get(tool_name)
        if not constraints:
            return False

        # Check constraints
        for key, constraint in constraints.items():
            if key == "max_calls" and self.call_count >= constraint:
                return False
            if key == "param_constraints":
                for param, allowed_values in constraint.items():
                    if params.get(param) not in allowed_values:
                        return False

        return True
```

### 5.2 Dynamic Tool Scoping

Dynamically adjust tool permissions based on context:

```python
class DynamicToolScoper:
    """Adjusts tool permissions based on conversation context."""

    def __init__(self):
        self.base_permissions = {}
        self.context_adjustments = {}

    def get_permissions(self, context: dict) -> dict:
        """Get effective permissions for the current context."""
        permissions = self.base_permissions.copy()

        # Apply context-based adjustments
        if context.get("topic") == "password_reset":
            permissions["send_email"]["allowed_recipients"] = ["support@company.com"]
            permissions["read_database"]["allowed_tables"] = ["users"]
        elif context.get("topic") == "financial_report":
            permissions["read_database"]["allowed_tables"] = ["revenue", "expenses"]
            permissions["send_email"]["enabled"] = False

        # Apply escalation restrictions
        if context.get("failure_count", 0) > 3:
            for tool in permissions:
                permissions[tool]["max_calls"] = 0

        return permissions
```

### 5.3 Per-Request Authorization

Every tool call is individually authorized:

```python
class PerRequestAuthorizer:
    """Authorizes each tool call individually."""

    def __init__(self):
        self.policy_engine = PolicyEngine()
        self.context_store = ContextStore()
        self.audit_logger = AuditLogger()

    def authorize(self, agent_id: str, tool_name: str,
                  params: dict) -> dict:
        context = self.context_store.get_context(agent_id)

        decision = self.policy_engine.evaluate(
            agent_id=agent_id,
            tool_name=tool_name,
            params=params,
            context=context,
        )

        self.audit_logger.log_authorization(
            agent_id=agent_id,
            tool_name=tool_name,
            params=params,
            decision=decision,
            timestamp=time.time(),
        )

        return decision
```

## 6. Tool Sandboxing

### 6.1 gVisor (Container-Level Sandbox)

gVisor provides an application kernel that sandboxes untrusted code:

```yaml
# gVisor runtime configuration for agent code execution
runtimeClassName: gvisor
metadata:
  name: agent-sandbox
spec:
  containers:
  - name: agent-code-executor
    image: python:3.11-slim
    command: ["python3", "-c", "{code}"]
    securityContext:
      seccompProfile:
        type: RuntimeDefault
      capabilities:
        drop: ["ALL"]
      readOnlyRootFilesystem: true
    resources:
      limits:
        cpu: "1"
        memory: "512Mi"
        ephemeral-storage: "100Mi"
    env:
    - name: PYTHONPATH
      value: "/sandbox"
```

**Python integration with gVisor:**

```python
import docker

class gVisorExecutor:
    def __init__(self):
        self.client = docker.from_env()

    def execute_code(self, code: str, language: str) -> dict:
        container = None
        try:
            container = self.client.containers.run(
                image=f"{language}-sandbox:latest",
                command=self._get_command(language, code),
                runtime="gvisor",
                read_only=True,
                mem_limit="512m",
                cpu_period=100000,
                cpu_quota=100000,  # 1 CPU
                network_disabled=True,
                remove=True,
                timeout=30,
            )
            output = container.logs(stdout=True, stderr=True)
            return {"success": True, "output": output.decode()}
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            if container:
                try:
                    container.remove(force=True)
                except Exception:
                    pass
```

### 6.2 Firecracker MicroVMs

Firecracker provides lightweight micro-VMs for stronger isolation:

```python
import asyncio
from firecracker import FirecrackerVM

class FirecrackerExecutor:
    """Execute tool actions in Firecracker micro-VMs."""

    def __init__(self):
        self.vm_base = "/opt/firecracker/vms"

    async def execute_in_vm(self, action: dict) -> dict:
        vm_id = f"agent-{uuid.uuid4().hex[:8]}"

        # Create and start VM
        vm = FirecrackerVM(
            vm_id=vm_id,
            kernel="/opt/firecracker/kernel/vmlinux.bin",
            rootfs=f"{self.vm_base}/agent-rootfs.ext4",
            vcpu_count=1,
            mem_size_mib=256,
        )

        await vm.start()

        try:
            # Execute action in VM
            result = await vm.run_command(
                f"python3 -c '{json.dumps(action)}'",
                timeout=30
            )
            return json.loads(result)
        except Exception as e:
            return {"error": str(e)}
        finally:
            await vm.stop()
            await vm.cleanup()
```

### 6.3 Code Interpreter Security

Secure code execution environments for agent code interpreters:

```python
import subprocess
import tempfile
import os
import resource

class SecureCodeInterpreter:
    """
    Secure code execution with resource limits and safety restrictions.
    """

    def __init__(self):
        self.allowed_imports = {
            "python": [
                "json", "math", "datetime", "collections",
                "itertools", "functools", "typing",
            ],
        }
        self.blocked_builtins = {
            "python": ["__import__", "exec", "eval", "compile", "open"],
        }

    def execute(self, language: str, code: str,
                timeout: int = 30) -> dict:
        if language not in self.allowed_imports:
            return {"error": f"Language {language} not supported"}

        # Pre-process code
        safe_code = self._sanitize_code(language, code)
        if safe_code["blocked"]:
            return {"error": safe_code["reason"]}

        # Execute in restricted environment
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                result = self._run_sandboxed(
                    language, safe_code["code"], tmpdir, timeout
                )
                return result
            except subprocess.TimeoutExpired:
                return {"error": "Execution timed out"}
            except Exception as e:
                return {"error": str(e)}

    def _sanitize_code(self, language: str, code: str) -> dict:
        for blocked in self.blocked_builtins.get(language, []):
            if blocked in code:
                return {"blocked": True, "reason": f"Use of {blocked} is restricted"}

        # Wrap code in restricted environment
        allowed_imports = self.allowed_imports.get(language, [])
        wrapped = f"""
import {' ,'.join(allowed_imports)}
# Restrict builtins
_builtins = __builtins__
if isinstance(_builtins, dict):
    _builtins['__import__'] = None
else:
    _builtins.__import__ = None

# User code
{code}
"""
        return {"blocked": False, "code": wrapped}

    def _run_sandboxed(self, language: str, code: str,
                       workdir: str, timeout: int) -> dict:
        # Set resource limits
        def set_limits():
            resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout))
            resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
            resource.setrlimit(resource.RLIMIT_NPROC, (20, 20))

        if language == "python":
            result = subprocess.run(
                ["python3", "-c", code],
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=set_limits,
                env={"PATH": "/usr/bin", "HOME": workdir},
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
            }
```

## 7. Tool Call Auditing

### 7.1 Audit Trail Structure

Every tool call should produce an immutable audit record:

```python
class ToolCallAuditRecord:
    def __init__(self):
        self.agent_id = ""
        self.tool_name = ""
        self.params = {}
        self.timestamp = 0
        self.result = {}
        self.approval = None
        self.execution_time_ms = 0
        self.risk_score = 0.0
        self.signature = b""

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "tool_name": self.tool_name,
            "params": self.params,
            "timestamp": self.timestamp,
            "result_status": self.result.get("status", "unknown"),
            "result_truncated": len(str(self.result)) > 1000,
            "approval": self.approval,
            "execution_time_ms": self.execution_time_ms,
            "risk_score": self.risk_score,
            "signature": self.signature.hex(),
        }

    def sign(self, signing_key: bytes):
        import hmac, hashlib, json
        data = json.dumps(self.to_dict(), sort_keys=True)
        self.signature = hmac.new(
            signing_key, data.encode(), hashlib.sha256
        ).digest()
```

### 7.2 Real-Time Monitoring

```python
class ToolCallMonitor:
    def __init__(self):
        self.handlers = []

    def add_handler(self, handler):
        self.handlers.append(handler)

    def on_tool_call(self, record: ToolCallAuditRecord):
        for handler in self.handlers:
            try:
                handler(record)
            except Exception as e:
                print(f"Handler error: {e}")

# Example handlers
def alert_on_suspicious_tool(record: ToolCallAuditRecord):
    if record.risk_score > 0.8:
        send_alert(f"Suspicious tool call: {record.agent_id} "
                   f"called {record.tool_name}")

def log_to_siem(record: ToolCallAuditRecord):
    siem_client.send(record.to_dict())

def enforce_rate_limits(record: ToolCallAuditRecord):
    rate_limiter.record_call(record.agent_id, record.tool_name)
```

## 8. Framework Implementations

### 8.1 LangChain Tool Policies

LangChain provides tool security through its tool API:

```python
from langchain.tools import BaseTool, StructuredTool
from langchain.tools.base import ToolException
from typing import Optional, Type
from pydantic import BaseModel, Field

class SecureWebSearchInput(BaseModel):
    query: str = Field(description="Search query", max_length=200)
    max_results: int = Field(default=5, ge=1, le=10)

class SecureWebSearchTool(BaseTool):
    name = "secure_web_search"
    description = "Search the web (restricted)"
    args_schema: Type[BaseModel] = SecureWebSearchInput
    return_direct = False

    def _run(self, query: str, max_results: int = 5) -> str:
        # Domain allow list
        allowed_domains = set()
        try:
            with open("/etc/agent/allowed_search_domains.txt") as f:
                allowed_domains = set(line.strip() for line in f)
        except FileNotFoundError:
            pass

        # Restrict searches via search engine parameters
        return self._search_with_restrictions(query, max_results, allowed_domains)

    async def _arun(self, query: str, max_results: int = 5) -> str:
        return self._run(query, max_results)

    def _search_with_restrictions(self, query: str, max_results: int,
                                   allowed_domains: set) -> str:
        if allowed_domains:
            restricted_query = f"{query} site:({' OR site:'.join(allowed_domains)})"
        else:
            restricted_query = query
        # Execute restricted search...
        return f"Search results for: {restricted_query}"
```

### 8.2 AutoGen Agent Permissions

AutoGen supports agent-level permission configuration:

```python
import autogen
from autogen import Agent, ConversableAgent, AssistantAgent

class SecureUserProxyAgent(autogen.UserProxyAgent):
    """User proxy with configurable tool permissions."""

    def __init__(self, name, human_input_mode="ALWAYS",
                 max_consecutive_auto_reply=10,
                 permissions_config=None, **kwargs):
        super().__init__(
            name=name,
            human_input_mode=human_input_mode,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            **kwargs
        )
        self.permissions = permissions_config or {}
        self.call_count = {}

    def execute_function(self, func_call):
        tool_name = func_call.get("name")
        tool_args = func_call.get("arguments", {})

        # Check permissions
        if not self._check_permission(tool_name, tool_args):
            return f"Permission denied: {tool_name} is not allowed"

        # Check rate limits
        self.call_count[tool_name] = self.call_count.get(tool_name, 0) + 1
        max_calls = self.permissions.get(tool_name, {}).get("max_calls", float('inf'))
        if self.call_count[tool_name] > max_calls:
            return f"Rate limit exceeded for {tool_name}"

        return super().execute_function(func_call)

    def _check_permission(self, tool_name: str, args: dict) -> bool:
        tool_perm = self.permissions.get(tool_name)
        if not tool_perm:
            return False

        # Check parameter constraints
        for param, constraint in tool_perm.get("param_restrictions", {}).items():
            if param in args:
                value = args[param]
                if isinstance(constraint, list) and value not in constraint:
                    return False
                if isinstance(constraint, dict):
                    if "max" in constraint and value > constraint["max"]:
                        return False
                    if "min" in constraint and value < constraint["min"]:
                        return False

        return True

# Usage
permissions = {
    "search_web": {"max_calls": 20, "param_restrictions": {}},
    "send_email": {
        "max_calls": 5,
        "param_restrictions": {
            "to": ["team@company.com", "admin@company.com"],
            "max_recipients": 3,
        }
    },
}

user_proxy = SecureUserProxyAgent(
    name="SecureUser",
    permissions_config=permissions,
)
```

### 8.3 CrewAI Tool Restrictions

```python
from crewai import Agent, Task, Crew
from crewai.tools import tool

class CrewToolSecurity:
    """Security wrapper for CrewAI tools."""

    @staticmethod
    def secure_tool(tool_func, allowed_params: dict = None,
                    max_calls: int = 100):
        """Wrap a tool function with security checks."""
        call_count = 0

        def wrapper(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count > max_calls:
                return "Tool call limit exceeded"

            if allowed_params:
                for param, allowed in allowed_params.items():
                    if param in kwargs and kwargs[param] not in allowed:
                        return f"Parameter {param}={kwargs[param]} not allowed"

            return tool_func(*args, **kwargs)

        return wrapper

@tool("Secure Search")
def secure_search(query: str) -> str:
    """Search with restrictions."""
    return f"Simulated search: {query}"

# Apply restrictions
secure_search_wrapped = CrewToolSecurity.secure_tool(
    secure_search,
    allowed_params={"query": ["company news", "weather", "stock price"]}
)
```

## 9. Audit and Compliance

### 9.1 Tool Call Audit Log Schema

```
Field                 Type        Description
────────────────────────────────────────────────────────────
audit_id              UUID        Unique audit record ID
agent_id              String      Agent that made the call
session_id            String      Conversation session
tool_name             String      Tool that was called
params                JSON        Parameters of the call
result_status         String      success/error/blocked
result_size           Integer     Size of result in bytes
execution_time_ms     Integer     How long execution took
risk_score            Float       Risk score (0.0 - 1.0)
approval_required     Boolean     Was human approval needed?
approval_granted      Boolean     Was approval given?
approver_id           String      Who approved (if HITL)
source_ip             String      Originating IP
timestamp             DateTime    When the call was made
signature             String      HMAC signature for integrity
```

### 9.2 Compliance Mapping

| Control | Implementation |
|---------|---------------|
| SOC 2 CC6.1 (Logical Access) | RBAC/ABAC tool permissions |
| SOC 2 CC7.2 (Monitoring) | Real-time tool call monitoring |
| ISO 27001 A.9.2 (User Access) | Tool capability tokens |
| ISO 27001 A.12.4 (Logging) | Tool call audit records |
| PCI DSS 7.2 (Access Control) | Per-request tool authorization |
| HIPAA §164.312 (Access Control) | Parameter validation + audit |

## 10. Conclusion

Tool access control is the most critical security mechanism for agent systems. The autonomy of agents means that traditional perimeter-based security is insufficient — every tool call must be individually authorized, validated, and audited.

Key implementation principles:

1. **Default deny**: No agent should have access to any tool by default.
2. **Least privilege**: Grant the minimum tool access needed for each agent's purpose.
3. **Defense in depth**: Combine OS-level, RBAC, ABAC, and capability-based controls.
4. **Validate everything**: Every parameter of every tool call must be validated against strict schemas.
5. **Sandbox execution**: Code execution and dangerous operations must run in isolated environments.
6. **Audit everything**: Every tool call must produce an immutable audit record.
7. **Monitor in real time**: Suspicious patterns must trigger immediate alerts.
8. **Review regularly**: Tool permissions should be reviewed and updated as agent capabilities evolve.

---

**References**
- LangChain Tool Security Documentation
- AutoGen Agent Configuration Guide
- gVisor Documentation
- Firecracker MicroVM Security Overview
- NIST SP 800-207: Zero Trust Architecture
- OWASP Top 10 for LLM Applications: LLM02 - Insecure Output Handling

---

**Document Information**
- Title: Tool Access Control
- Series: 18-Agent-Security-and-Trust
- Part: 03 of 08
- Author: AI Knowledge Base
- Last Updated: 2026-06-13
- Lines: 620
