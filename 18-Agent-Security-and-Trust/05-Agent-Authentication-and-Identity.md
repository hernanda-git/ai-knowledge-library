# 05 — Agent Authentication and Identity

## 1. Introduction

Authentication and identity management for AI agents presents unique challenges compared to traditional service-to-service authentication. Agents operate autonomously, often on behalf of human users, interacting with multiple services and other agents. Establishing and maintaining trusted identities for agents is critical for access control, auditability, and accountability.

This document provides a comprehensive technical reference for implementing authentication and identity management in agent systems, covering agent identity models, API key management, OAuth2 flows, service-to-service auth, credential management, and decentralized identity frameworks.

## 2. The Agent Identity Problem

### 2.1 Identity Requirements

Agent systems have distinct identity requirements:

- **Delegated identity**: Agents act on behalf of users but need their own identity.
- **Multi-party trust**: An agent may need to authenticate to multiple services simultaneously.
- **Short-lived sessions**: Agent interactions may be brief, requiring fast authentication.
- **Cross-domain operation**: Agents often span organizational boundaries.
- **Auditability**: Every action must be traceable to a specific agent identity.

### 2.2 Agent Identity Model

```
                  ┌────────────────────┐
                  │   Human User       │
                  │   (Alice@corp)     │
                  └────────┬───────────┘
                           │ Delegates authority
                           ▼
┌────────────────────────────────────────────────────┐
│                 Agent Instance                      │
│  ┌──────────────────────────────────────────────┐  │
│  │ Agent Identity: agent-alice-email-bot-v3     │  │
│  │ Agent Role: email-assistant                  │  │
│  │ Issuer: platform.nousresearch.com            │  │
│  │ Credentials: [JWT / mTLS cert / API key]     │  │
│  │ Capabilities: [read_email, send_email]       │  │
│  └──────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
   ┌─────────┐      ┌────────────┐      ┌─────────┐
   │ Email   │      │ Calendar   │      │ Slack   │
   │ Service │      │ Service    │      │ API     │
   └─────────┘      └────────────┘      └─────────┘
```

### 2.3 Identity Lifecycle

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ Provision│───►│  Activate│───►│  Operate │───►│  Revoke  │
│ Identity │    │          │    │          │    │          │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
     │               │               │               │
     │ Create        │ Issue         │ Rotate        │ Revoke all
     │ identity      │ credential    │ credentials   │ credentials
     │ assign role   │ enable agent  │ monitor usage │ disable agent
     └───────────────┴───────────────┴───────────────┴──────────────┘
```

## 3. Agent Identity Models

### 3.1 Service Account Model

The most straightforward model: agents get service accounts similar to traditional service identities.

```python
class AgentServiceAccount:
    """Service account for an agent."""

    def __init__(self, agent_id: str, project: str):
        self.agent_id = agent_id
        self.project = project
        self.credentials = None
        self.roles = []
        self.created_at = datetime.now()
        self.last_rotation = None

    def generate_credentials(self) -> dict:
        """Generate initial credentials for the agent."""
        self.credentials = {
            "client_id": f"{self.project}:{self.agent_id}",
            "client_secret": secrets.token_urlsafe(32),
            "key_id": hashlib.sha256(f"{self.agent_id}{time.time()}".encode()).hexdigest()[:16],
            "issued_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(days=90)).isoformat(),
        }
        return self.credentials

    def rotate_credentials(self) -> dict:
        """Rotate credentials."""
        old_key_id = self.credentials["key_id"]
        self.credentials = self.generate_credentials()
        self.credentials["previous_key_id"] = old_key_id
        self.last_rotation = datetime.now()
        return self.credentials
```

### 3.2 Delegated Identity Model

Agents authenticate as a user through OAuth delegation:

```python
class DelegatedAgentIdentity:
    """
    Agent that acts on behalf of a user through delegated authentication.
    """

    def __init__(self, user_id: str, agent_id: str):
        self.user_id = user_id
        self.agent_id = agent_id
        self.delegation_token = None
        self.scopes = []
        self.token_expiry = None

    def request_delegation(self, auth_server: str,
                            requested_scopes: list[str],
                            user_consent: dict) -> dict:
        """Request delegated access on behalf of the user."""
        # Exchange user consent for a delegation token
        token_request = {
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "subject_token": user_consent["user_token"],
            "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
            "actor_token": self._get_agent_token(),
            "actor_token_type": "urn:ietf:params:oauth:token-type:id_token",
            "scope": " ".join(requested_scopes),
        }

        response = requests.post(
            f"{auth_server}/token",
            data=token_request,
        )

        if response.status_code == 200:
            token_data = response.json()
            self.delegation_token = token_data["access_token"]
            self.scopes = token_data.get("scope", "").split()
            self.token_expiry = datetime.now() + timedelta(
                seconds=token_data.get("expires_in", 3600)
            )
            return token_data

        raise Exception(f"Delegation failed: {response.text}")

    def get_effective_identity(self) -> dict:
        """Return the effective identity (user + agent)."""
        return {
            "acting_user": self.user_id,
            "agent": self.agent_id,
            "delegation_expiry": self.token_expiry.isoformat(),
            "scopes": self.scopes,
        }
```

### 3.3 Workload Identity Model

For agents running in cloud/container environments:

```yaml
# Kubernetes service account for agent workload
apiVersion: v1
kind: ServiceAccount
metadata:
  name: agent-email-bot
  namespace: agent-ns
  annotations:
    # GCP Workload Identity
    iam.gke.io/gcp-service-account: agent-email-bot@project.iam.gserviceaccount.com
    # AWS IRSA
    eks.amazonaws.com/role-arn: arn:aws:iam::123456789012:role/agent-email-bot
    # Azure Workload Identity
    azure.workload.identity/client-id: 00000000-0000-0000-0000-000000000000
---
apiVersion: v1
kind: Pod
metadata:
  name: agent-pod
  namespace: agent-ns
  labels:
    azure.workload.identity/use: "true"
spec:
  serviceAccountName: agent-email-bot
  containers:
  - name: agent
    image: agent:latest
```

### 3.4 Decentralized Identity (DID) Model

For cross-organizational agent interactions:

```python
import didkit  # DIDKit library for W3C DIDs

class DecentralizedAgentIdentity:
    """Agent identity using W3C Decentralized Identifiers."""

    def __init__(self):
        self.did = None
        self.did_document = None
        self.key_pair = None

    def create_did(self, method: str = "key") -> dict:
        """Create a new DID for the agent."""
        # Generate key pair
        self.key_pair = {
            "private_key": didkit.generate_ed25519_key(),
            "public_key": None,
        }

        # Create DID
        did_str = didkit.key_to_did(method, self.key_pair["private_key"])
        self.did = did_str

        # Resolve DID document
        self.did_document = didkit.resolve_did(did_str)

        return {
            "did": self.did,
            "did_document": self.did_document,
            "key_type": "Ed25519",
        }

    def sign_verifiable_presentation(self, claims: dict) -> str:
        """Sign a Verifiable Presentation with the agent's DID."""
        vp = {
            "@context": ["https://www.w3.org/2018/credentials/v1"],
            "type": ["VerifiablePresentation"],
            "verifiableCredential": [
                self._create_verifiable_credential(claims)
            ],
        }
        signed_vp = didkit.issue_presentation(
            json.dumps(vp),
            json.dumps({
                "verificationMethod": f"{self.did}#key-1",
                "proofPurpose": "assertionMethod",
            }),
            self.key_pair["private_key"],
        )
        return signed_vp

    def verify_peer_did(self, peer_did: str) -> bool:
        """Verify another agent's DID."""
        try:
            doc = didkit.resolve_did(peer_did)
            return doc is not None
        except Exception:
            return False
```

## 4. API Key Management

### 4.1 API Key Generation

```python
import secrets
import hashlib
import base64
from datetime import datetime, timedelta

class AgentAPIKeyManager:
    """Manages API keys for agent authentication."""

    def __init__(self):
        self.active_keys = {}  # key_hash -> key_metadata
        self.revoked_keys = set()

    def create_key(self, agent_id: str, permissions: list[str],
                   ttl_days: int = 90) -> dict:
        """Generate a new API key for an agent."""
        # Generate key
        key_bytes = secrets.token_bytes(32)
        api_key = f"ag_{base64.urlsafe_b64encode(key_bytes).decode().rstrip('=')}"

        # Hash for storage
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        # Store metadata
        metadata = {
            "agent_id": agent_id,
            "permissions": permissions,
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(days=ttl_days)).isoformat(),
            "last_used": None,
            "key_prefix": api_key[:12],  # For identification
        }

        self.active_keys[key_hash] = metadata

        # Return the full key (only time it's visible)
        return {
            "api_key": api_key,
            "key_prefix": metadata["key_prefix"],
            "expires_at": metadata["expires_at"],
        }

    def validate_key(self, api_key: str) -> dict:
        """Validate an API key and return agent info."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        metadata = self.active_keys.get(key_hash)
        if not metadata:
            return {"valid": False, "reason": "Key not found"}

        if key_hash in self.revoked_keys:
            return {"valid": False, "reason": "Key revoked"}

        if datetime.now() > datetime.fromisoformat(metadata["expires_at"]):
            return {"valid": False, "reason": "Key expired"}

        # Update last used
        metadata["last_used"] = datetime.now().isoformat()

        return {
            "valid": True,
            "agent_id": metadata["agent_id"],
            "permissions": metadata["permissions"],
        }

    def revoke_key(self, api_key: str):
        """Revoke an API key."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        if key_hash in self.active_keys:
            self.revoked_keys.add(key_hash)
            self.active_keys[key_hash]["revoked_at"] = datetime.now().isoformat()
```

### 4.2 API Key Rotation

```python
class KeyRotationManager:
    """Automated API key rotation for agents."""

    def __init__(self, key_manager: AgentAPIKeyManager):
        self.key_manager = key_manager
        self.rotation_schedule = {}  # agent_id -> rotation config

    def schedule_rotation(self, agent_id: str,
                          interval_days: int = 30,
                          overlap_hours: int = 24):
        """Schedule automatic key rotation."""
        self.rotation_schedule[agent_id] = {
            "interval_days": interval_days,
            "overlap_hours": overlap_hours,
            "last_rotation": None,
            "next_rotation": datetime.now(),
        }

    def rotate_if_needed(self, agent_id: str) -> Optional[dict]:
        """Rotate key if scheduled."""
        schedule = self.rotation_schedule.get(agent_id)
        if not schedule:
            return None

        if datetime.now() < schedule["next_rotation"]:
            return None

        # Create new key
        old_key = self._get_current_key(agent_id)
        new_key = self.key_manager.create_key(
            agent_id=agent_id,
            permissions=self._get_agent_permissions(agent_id),
        )

        # Schedule old key deletion after overlap period
        schedule["last_rotation"] = datetime.now()
        schedule["next_rotation"] = datetime.now() + timedelta(
            days=schedule["interval_days"]
        )

        if old_key:
            # Schedule cleanup
            cleanup_time = datetime.now() + timedelta(hours=schedule["overlap_hours"])
            self._schedule_key_cleanup(old_key, cleanup_time)

        return new_key

    def _schedule_key_cleanup(self, key: str, cleanup_time: datetime):
        """Schedule old key revocation after overlap period."""
        # In production, use a background job queue
        pass
```

### 4.3 Key Security Best Practices

```python
class SecureKeyStorage:
    """Secure storage for agent credentials."""

    def __init__(self, master_key: bytes):
        self.master_key = master_key
        from cryptography.fernet import Fernet
        import base64
        # Use HKDF to derive encryption key
        derived = hashlib.pbkdf2_hmac(
            'sha256', master_key, b'agent-key-storage', 100000
        )
        self.cipher = Fernet(base64.urlsafe_b64encode(derived))

    def store_credential(self, agent_id: str, credential: str):
        """Encrypt and store a credential."""
        encrypted = self.cipher.encrypt(credential.encode())
        # Store in secure backend (Vault, KMS, etc.)
        self._write_to_backend(f"credentials/{agent_id}", encrypted)

    def retrieve_credential(self, agent_id: str) -> str:
        """Retrieve and decrypt a credential."""
        encrypted = self._read_from_backend(f"credentials/{agent_id}")
        if not encrypted:
            raise ValueError(f"No credential found for {agent_id}")
        decrypted = self.cipher.decrypt(encrypted)
        return decrypted.decode()

    def delete_credential(self, agent_id: str):
        """Securely delete a credential."""
        self._write_to_backend(f"credentials/{agent_id}", None)
```

## 5. OAuth2 for Agents

### 5.1 OAuth2 Client Credentials Flow

The most common OAuth2 flow for non-interactive agents:

```python
class AgentOAuth2Client:
    """OAuth2 client for agent authentication."""

    def __init__(self, client_id: str, client_secret: str,
                 token_url: str, scopes: list[str]):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url
        self.scopes = scopes
        self.access_token = None
        self.token_expiry = None
        self.refresh_token = None

    def authenticate(self) -> str:
        """Authenticate and get an access token."""
        if self._is_token_valid():
            return self.access_token

        return self._request_new_token()

    def _request_new_token(self) -> str:
        """Request a new token using client credentials."""
        response = requests.post(
            self.token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "scope": " ".join(self.scopes),
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        if response.status_code != 200:
            raise Exception(f"OAuth2 auth failed: {response.text}")

        token_data = response.json()
        self.access_token = token_data["access_token"]
        self.token_expiry = datetime.now() + timedelta(
            seconds=token_data.get("expires_in", 3600)
        )
        self.refresh_token = token_data.get("refresh_token")

        return self.access_token

    def _is_token_valid(self) -> bool:
        """Check if the current token is still valid."""
        if not self.access_token or not self.token_expiry:
            return False
        # Refresh 5 minutes before expiry
        return datetime.now() < self.token_expiry - timedelta(minutes=5)
```

### 5.2 OAuth2 Token Exchange (Acting on Behalf)

Use token exchange for delegated agent scenarios:

```python
class OAuth2TokenExchange:
    """
    OAuth 2.0 Token Exchange for agent delegation.
    RFC 8693: OAuth 2.0 Token Exchange
    """

    def __init__(self, auth_server: str, client_config: dict):
        self.auth_server = auth_server
        self.client_id = client_config["client_id"]
        self.client_secret = client_config["client_secret"]

    def exchange_for_agent(self, user_token: str,
                            agent_token: str,
                            requested_scopes: list[str]) -> dict:
        """
        Exchange user + agent tokens for a delegated token.
        The resulting token represents "agent acting on behalf of user."
        """
        response = requests.post(
            f"{self.auth_server}/token",
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
                "subject_token": user_token,
                "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
                "actor_token": agent_token,
                "actor_token_type": "urn:ietf:params:oauth:token-type:id_token",
                "scope": " ".join(requested_scopes),
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        if response.status_code != 200:
            raise Exception(f"Token exchange failed: {response.text}")

        return response.json()

    def validate_delegated_token(self, token: str) -> dict:
        """Validate a delegated token and extract identity info."""
        response = requests.post(
            f"{self.auth_server}/introspect",
            data={
                "token": token,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
        )

        if response.status_code == 200:
            token_info = response.json()
            return {
                "active": token_info.get("active", False),
                "acting_user": token_info.get("sub"),
                "agent": token_info.get("act", {}).get("sub"),
                "scopes": token_info.get("scope", "").split(),
                "expires_at": token_info.get("exp"),
            }

        return {"active": False}
```

### 5.3 OAuth2 with JWT Assertion

For agents that need to authenticate without a client secret:

```python
import jwt
from cryptography.hazmat.primitives import serialization

class AgentJWTAssertion:
    """
    OAuth2 client authentication using JWT assertion (private_key_jwt).
    Allows agents to authenticate using a key pair instead of a shared secret.
    """

    def __init__(self, client_id: str, private_key_path: str,
                 token_url: str, key_id: str = None):
        self.client_id = client_id
        self.token_url = token_url
        self.key_id = key_id

        with open(private_key_path, "rb") as f:
            self.private_key = serialization.load_ssh_private_key(
                f.read(), password=None
            )

    def create_assertion(self, audience: str = None) -> str:
        """Create a JWT assertion for client authentication."""
        now = datetime.now()
        assertion = jwt.encode(
            {
                "iss": self.client_id,
                "sub": self.client_id,
                "aud": audience or self.token_url,
                "jti": secrets.token_hex(16),
                "iat": now,
                "exp": now + timedelta(minutes=5),
            },
            self.private_key,
            algorithm="RS256",
            headers={"kid": self.key_id} if self.key_id else {},
        )
        return assertion

    def authenticate(self) -> str:
        """Authenticate using private_key_jwt client auth."""
        assertion = self.create_assertion()

        response = requests.post(
            self.token_url,
            data={
                "grant_type": "client_credentials",
                "client_assertion_type": (
                    "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
                ),
                "client_assertion": assertion,
                "scope": " ".join(self.scopes),
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        if response.status_code == 200:
            token_data = response.json()
            return token_data["access_token"]

        raise Exception(f"JWT assertion auth failed: {response.text}")
```

## 6. Service-to-Service Authentication

### 6.1 Mutual TLS (mTLS)

mTLS provides strong identity verification at the transport layer:

```python
import ssl
import requests
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager

class mTLSAgentClient:
    """Agent client using mutual TLS for service-to-service auth."""

    def __init__(self, cert_path: str, key_path: str,
                 ca_cert_path: str):
        self.cert_path = cert_path
        self.key_path = key_path
        self.ca_cert_path = ca_cert_path
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        session = requests.Session()

        # Create custom adapter with mTLS
        class mTLSAdapter(HTTPAdapter):
            def init_poolmanager(self, *args, **kwargs):
                ctx = ssl.create_default_context(
                    purpose=ssl.Purpose.SERVER_AUTH,
                    cafile=self.ca_cert_path,
                )
                ctx.load_cert_chain(
                    certfile=self.cert_path,
                    keyfile=self.key_path,
                )
                ctx.verify_mode = ssl.CERT_REQUIRED
                kwargs['ssl_context'] = ctx
                return PoolManager(*args, **kwargs)

        session.mount('https://', mTLSAdapter())
        return session

    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Make an mTLS-authenticated request."""
        return self.session.request(method, url, **kwargs)

class mTLSAgentServer:
    """Server-side mTLS verification for agent authentication."""

    def __init__(self, ca_cert_path: str):
        self.ca_cert_path = ca_cert_path
        self.allowed_agents = {}  # CN -> agent metadata

    def verify_client(self, client_cert: dict) -> dict:
        """Verify an mTLS client certificate."""
        cn = client_cert.get("subject", {}).get("CN")

        if cn not in self.allowed_agents:
            return {"authenticated": False, "reason": "Unknown agent"}

        agent_info = self.allowed_agents[cn]
        return {
            "authenticated": True,
            "agent_id": agent_info["agent_id"],
            "role": agent_info["role"],
        }

    def register_agent(self, agent_id: str, common_name: str,
                        role: str):
        """Register an agent's expected certificate identity."""
        self.allowed_agents[common_name] = {
            "agent_id": agent_id,
            "role": role,
        }
```

### 6.2 SPIFFE/SPIRE for Agent Workloads

SPIFFE (Secure Production Identity Framework for Everyone) provides a standard for workload identity:

```python
import spiffe  # SPIRE client library

class SPIFFEAgentIdentity:
    """Agent identity using SPIFFE standard via SPIRE agent."""

    def __init__(self, socket_path: str = "/tmp/spire-agent/public/api.sock"):
        self.socket_path = socket_path
        self.client = None
        self.svid = None

    def acquire_identity(self) -> dict:
        """Acquire a SPIFFE Verifiable Identity Document (SVID)."""
        # Connect to SPIRE agent
        self.client = spiffe.WorkloadAPIClient(
            socket_path=self.socket_path
        )

        # Fetch X.509 SVID
        self.svid = self.client.fetch_x509_svid()

        return {
            "spiffe_id": str(self.svid.spiffe_id),
            "certificate": self.svid.certificate,
            "private_key": self.svid.private_key,
            "bundle": self.svid.bundle,
        }

    def get_spiffe_id(self) -> Optional[str]:
        """Get the SPIFFE ID of this agent."""
        if self.svid:
            return str(self.svid.spiffe_id)
        return None

    def authenticate_request(self, request: requests.Request):
        """Add SPIFFE authentication to an outgoing request."""
        if not self.svid:
            self.acquire_identity()

        # Add mTLS certificate to request
        request.cert = (self.svid.certificate, self.svid.private_key)
        return request

class SPIFFEAuthMiddleware:
    """Middleware to verify SPIFFE identities for incoming requests."""

    def __init__(self, trust_domain: str):
        self.trust_domain = trust_domain
        self.allowed_spiffe_ids = set()

    def verify(self, request) -> dict:
        """Verify the SPIFFE identity in an incoming request."""
        # Extract client cert from request
        client_cert = request.get_client_cert()
        if not client_cert:
            return {"authenticated": False, "reason": "No client certificate"}

        # Verify cert is from our trust domain
        # SPIFFE IDs look like: spiffe://trust-domain/path
        spiffe_id = self._extract_spiffe_id(client_cert)
        if not spiffe_id:
            return {"authenticated": False, "reason": "Not a valid SPIFFE cert"}

        if str(spiffe_id) not in self.allowed_spiffe_ids:
            return {"authenticated": False, "reason": "SPIFFE ID not authorized"}

        return {
            "authenticated": True,
            "spiffe_id": str(spiffe_id),
            "agent_id": self._spiffe_id_to_agent_id(spiffe_id),
        }

    def add_allowed_agent(self, spiffe_id: str):
        self.allowed_spiffe_ids.add(spiffe_id)
```

### 6.3 API Gateway Authentication

```python
class AgentAPIGatewayAuth:
    """API Gateway authentication for agent services."""

    def __init__(self):
        self.auth_providers = {
            "api_key": APIKeyAuthProvider(),
            "jwt": JWTAuthProvider(),
            "mtls": mTLSAuthProvider(),
            "oauth2": OAuth2AuthProvider(),
        }
        self.agent_registry = AgentRegistry()

    def authenticate(self, request: dict) -> dict:
        """Authenticate an incoming agent request through the gateway."""
        headers = request.get("headers", {})

        # Try each auth method
        for auth_type, provider in self.auth_providers.items():
            auth_header = headers.get(f"X-Auth-{auth_type.upper()}")
            if auth_header:
                result = provider.validate(auth_header)
                if result["authenticated"]:
                    # Map to agent identity
                    agent = self.agent_registry.lookup(result["agent_id"])
                    if agent:
                        return {
                            "authenticated": True,
                            "agent": agent,
                            "auth_method": auth_type,
                        }
                else:
                    return result

        return {"authenticated": False, "reason": "No valid authentication"}

    def generate_auth_token(self, agent_id: str,
                             method: str = "jwt") -> str:
        """Generate an authentication token for an agent."""
        provider = self.auth_providers.get(method)
        if provider:
            return provider.generate(agent_id)
        raise ValueError(f"Unknown auth method: {method}")
```

## 7. Agent Credential Management

### 7.1 HashiCorp Vault Integration

```python
import hvac

class VaultAgentCredentials:
    """Manage agent credentials using HashiCorp Vault."""

    def __init__(self, vault_addr: str, vault_token: str):
        self.client = hvac.Client(url=vault_addr, token=vault_token)

    def create_agent_identity(self, agent_id: str,
                               policies: list[str]) -> dict:
        """Create a Vault identity for an agent."""
        # Create entity
        entity = self.client.secrets.identity.create_entity(
            name=agent_id,
            metadata={"type": "ai_agent"},
        )

        # Attach policies
        for policy in policies:
            self.client.secrets.identity.add_entity_policies(
                entity_id=entity["data"]["id"],
                policies=[policy],
            )

        # Generate periodic token
        token = self.client.auth.token.create(
            policies=policies,
            display_name=agent_id,
            renewable=True,
            ttl="24h",
        )

        return {
            "entity_id": entity["data"]["id"],
            "token": token["auth"]["client_token"],
            "token_ttl": "24h",
        }

    def get_dynamic_credentials(self, agent_id: str,
                                 db_role: str) -> dict:
        """Get dynamic database credentials for an agent."""
        creds = self.client.secrets.database.generate_credentials(
            name=db_role,
        )
        return {
            "username": creds["data"]["username"],
            "password": creds["data"]["password"],
            "lease_duration": creds["lease_duration"],
        }

    def rotate_secret(self, path: str) -> dict:
        """Rotate a secret in Vault."""
        response = self.client.secrets.kv.v2.rotate(
            path=path,
        )
        return response
```

### 7.2 Short-Lived Credentials

```python
class ShortLivedCredentialManager:
    """Manages short-lived credentials for agent security."""

    def __init__(self):
        self.credential_issuers = {}

    def register_issuer(self, service: str, issuer):
        self.credential_issuers[service] = issuer

    def get_sts_credentials(self, service: str,
                             agent_identity: dict,
                             duration: int = 3600) -> dict:
        """Get short-lived credentials for a service."""
        issuer = self.credential_issuers.get(service)
        if not issuer:
            raise ValueError(f"No issuer for {service}")

        return issuer.issue_credentials(
            identity=agent_identity,
            ttl_seconds=duration,
        )

    def refresh_if_needed(self, credential: dict) -> dict:
        """Refresh credentials if they're about to expire."""
        expiry = credential.get("expires_at")
        if not expiry:
            return credential

        if datetime.now() > datetime.fromisoformat(expiry) - timedelta(minutes=5):
            return self.get_sts_credentials(
                service=credential["service"],
                agent_identity=credential["agent_identity"],
            )
        return credential
```

### 7.3 AWS IAM Roles Anywhere for Agents

```python
import boto3
from cryptography.hazmat.primitives import serialization

class IAMRolesAnywhere:
    """Use IAM Roles Anywhere for agent credentials outside AWS."""

    def __init__(self, role_arn: str, profile_arn: str,
                 trust_anchor_arn: str, cert_path: str, key_path: str):
        self.role_arn = role_arn
        self.profile_arn = profile_arn
        self.trust_anchor_arn = trust_anchor_arn
        self.cert_path = cert_path
        self.key_path = key_path

    def get_credentials(self) -> dict:
        """Get temporary AWS credentials using IAM Roles Anywhere."""
        import requests

        # Create a signed request to Roles Anywhere
        with open(self.key_path, "rb") as f:
            private_key = serialization.load_pem_private_key(
                f.read(), password=None
            )

        # In production, use the AWS SDK's Roles Anywhere credential provider
        # This is a simplified example

        signer = self._create_signer(private_key)
        response = requests.post(
            "https://rolesanywhere.us-east-1.amazonaws.com/sessions",
            json={
                "durationSeconds": 3600,
                "profileArn": self.profile_arn,
                "roleArn": self.role_arn,
                "trustAnchorArn": self.trust_anchor_arn,
                "certificate": self._load_cert(),
            },
            auth=signer,
        )

        if response.status_code == 200:
            creds = response.json()["credentialSet"][0]["credentials"]
            return {
                "access_key": creds["accessKeyId"],
                "secret_key": creds["secretAccessKey"],
                "session_token": creds["sessionToken"],
                "expiration": creds["expiration"],
            }

        raise Exception(f"Roles Anywhere failed: {response.text}")
```

## 8. Secret Rotation

### 8.1 Automatic Rotation Framework

```python
class SecretRotationScheduler:
    """Schedules and executes automatic secret rotation for agents."""

    def __init__(self):
        self.rotation_handlers = {}
        self.rotation_log = []

    def register_secret(self, secret_id: str,
                         rotation_handler: callable,
                         interval_days: int = 30,
                         overlap_days: int = 1):
        """Register a secret for automatic rotation."""
        self.rotation_handlers[secret_id] = {
            "handler": rotation_handler,
            "interval_days": interval_days,
            "overlap_days": overlap_days,
            "last_rotation": None,
            "next_rotation": datetime.now(),
        }

    def rotate_due_secrets(self) -> list[dict]:
        """Rotate all secrets that are due for rotation."""
        results = []
        now = datetime.now()

        for secret_id, config in self.rotation_handlers.items():
            if now >= config["next_rotation"]:
                try:
                    result = config["handler"]()

                    config["last_rotation"] = now
                    config["next_rotation"] = now + timedelta(
                        days=config["interval_days"]
                    )

                    self.rotation_log.append({
                        "secret_id": secret_id,
                        "rotated_at": now.isoformat(),
                        "success": True,
                    })

                    results.append({
                        "secret_id": secret_id,
                        "success": True,
                        "overlap_end": now + timedelta(
                            days=config["overlap_days"]
                        ),
                    })

                except Exception as e:
                    self.rotation_log.append({
                        "secret_id": secret_id,
                        "rotated_at": now.isoformat(),
                        "success": False,
                        "error": str(e),
                    })
                    results.append({
                        "secret_id": secret_id,
                        "success": False,
                        "error": str(e),
                    })

        return results

    def get_rotation_status(self) -> dict:
        """Get the status of all registered secrets."""
        return {
            handler["next_rotation"].isoformat(): secret_id
            for secret_id, handler in self.rotation_handlers.items()
        }
```

### 8.2 Credential Rotation Example

```python
class DatabaseCredentialRotator:
    """Rotates database credentials for agents."""

    def __init__(self, vault_client, db_name: str):
        self.vault = vault_client
        self.db_name = db_name

    def rotate(self) -> dict:
        """Rotate database credentials."""
        # Generate new password
        new_password = secrets.token_urlsafe(32)

        # Update database
        self._update_db_password(self.db_name, new_password)

        # Update Vault
        self.vault.secrets.kv.v2.put(
            path=f"database/{self.db_name}/creds",
            data={"password": new_password},
        )

        # Notify connected agents
        self._notify_agents(self.db_name)

        return {
            "db_name": self.db_name,
            "rotated": True,
            "password": new_password,
        }

    def _notify_agents(self, db_name: str):
        """Notify agents that credentials have been rotated."""
        # In production, use a message queue or webhook
        pass
```

## 9. Multi-Agent Authentication

### 9.1 Agent-to-Agent Authentication

```python
class MultiAgentAuth:
    """Authentication between agents in a multi-agent system."""

    def __init__(self, trust_domain: str):
        self.trust_domain = trust_domain
        self.agent_registry = {}
        self.agent_sessions = {}

    def register_agent(self, agent_id: str, public_key: str,
                        capabilities: list[str]):
        """Register an agent with its public key."""
        self.agent_registry[agent_id] = {
            "public_key": public_key,
            "capabilities": capabilities,
            "registered_at": datetime.now(),
        }

    def create_session(self, from_agent: str, to_agent: str,
                        purpose: str) -> dict:
        """Create an authenticated session between two agents."""
        if from_agent not in self.agent_registry:
            raise ValueError(f"Unknown agent: {from_agent}")
        if to_agent not in self.agent_registry:
            raise ValueError(f"Unknown agent: {to_agent}")

        session_id = secrets.token_urlsafe(16)
        session_key = secrets.token_bytes(32)

        self.agent_sessions[session_id] = {
            "from_agent": from_agent,
            "to_agent": to_agent,
            "purpose": purpose,
            "session_key": session_key,
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(hours=1),
        }

        return {
            "session_id": session_id,
            "session_key": base64.b64encode(session_key).decode(),
            "expires_at": self.agent_sessions[session_id]["expires_at"].isoformat(),
        }

    def verify_agent_message(self, session_id: str,
                              message: str, signature: str) -> bool:
        """Verify a signed message between agents."""
        session = self.agent_sessions.get(session_id)
        if not session:
            return False

        if datetime.now() > session["expires_at"]:
            return False

        # Verify HMAC signature using session key
        expected = hmac.new(
            session["session_key"],
            message.encode(),
            hashlib.sha256,
        ).digest()

        return hmac.compare_digest(expected, base64.b64decode(signature))
```

### 9.2 Agent Identity Federation

```python
class AgentIdentityFederation:
    """Cross-organization agent identity federation."""

    def __init__(self, org_id: str):
        self.org_id = org_id
        self.federation_trusts = {}  # org_id -> trust_config
        self.agent_identities = {}   # agent_id -> identity

    def establish_trust(self, remote_org_id: str,
                         trust_config: dict):
        """Establish trust with another organization for agent auth."""
        self.federation_trusts[remote_org_id] = trust_config

    def create_federated_identity(self, agent_id: str,
                                   home_org: str) -> dict:
        """Create a federated identity for cross-org agent auth."""
        federated_id = f"{agent_id}@{home_org}"
        assertion = {
            "sub": federated_id,
            "iss": f"https://auth.{home_org}.com",
            "aud": f"https://auth.{self.org_id}.com",
            "iat": datetime.now(),
            "exp": datetime.now() + timedelta(hours=1),
            "jti": secrets.token_hex(16),
            "agent_id": agent_id,
            "org": home_org,
        }
        # Sign with org's key
        token = jwt.encode(assertion, self._get_signing_key(), algorithm="RS256")
        return token

    def verify_federated_identity(self, token: str) -> dict:
        """Verify a federated agent identity token."""
        try:
            header = jwt.get_unverified_header(token)
            # Resolve signing key from issuer
            payload = jwt.decode(
                token,
                self._get_verification_key(payload["iss"]),
                algorithms=["RS256"],
                audience=f"https://auth.{self.org_id}.com",
            )
            return {
                "valid": True,
                "agent_id": payload["agent_id"],
                "org": payload["org"],
                "federated_id": payload["sub"],
            }
        except jwt.PyJWTError as e:
            return {"valid": False, "error": str(e)}
```

## 10. Audit and Compliance

### 10.1 Identity Audit Trail

```python
class IdentityAuditor:
    """Audit trail for agent identity events."""

    def __init__(self):
        self.events = []

    def log_identity_event(self, event_type: str, agent_id: str,
                            details: dict):
        """Log an identity-related event."""
        entry = {
            "event_type": event_type,  # created, authenticated, rotated, revoked
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat(),
            "details": details,
            "event_hash": self._compute_hash(event_type, agent_id, details),
        }
        self.events.append(entry)
        return entry

    def get_agent_history(self, agent_id: str) -> list[dict]:
        """Get all identity events for an agent."""
        return [
            e for e in self.events
            if e["agent_id"] == agent_id
        ]

    def get_credential_rotation_report(self, days: int = 90) -> dict:
        """Generate a credential rotation compliance report."""
        cutoff = datetime.now() - timedelta(days=days)
        rotations = [
            e for e in self.events
            if e["event_type"] == "rotated"
            and datetime.fromisoformat(e["timestamp"]) > cutoff
        ]

        agents_rotated = set(e["agent_id"] for e in rotations)
        total_agents = self._get_total_agents()

        return {
            "period_days": days,
            "total_rotations": len(rotations),
            "agents_rotated": len(agents_rotated),
            "total_agents": total_agents,
            "compliance_rate": len(agents_rotated) / total_agents if total_agents else 0,
        }
```

### 10.2 Compliance Mapping

| Requirement | Implementation |
|-------------|---------------|
| **SOC 2 CC6.1** | Agent service accounts with unique credentials |
| **SOC 2 CC6.3** | Automatic credential rotation every 90 days |
| **ISO 27001 A.9.2.1** | Agent identity registration and de-registration |
| **ISO 27001 A.9.2.4** | Management of agent credentials |
| **PCI DSS 8.3.1** | Multi-factor authentication for admin agents |
| **HIPAA §164.312(d)** | Unique agent identifiers for access tracking |
| **GDPR Art. 5(2)** | Accountability - agent identity logged for all actions |

## 11. Conclusion

Agent authentication and identity management requires a multi-layered approach that combines traditional IAM practices with agent-specific considerations:

1. **Every agent needs an identity** — even if it acts on behalf of a user, it must have its own verifiable identity.
2. **Use workload identity** — SPIFFE/SPIRE or cloud workload identity for containerized agents.
3. **Prefer short-lived credentials** — minimize the window of vulnerability for compromised credentials.
4. **Implement credential rotation** — automatic, scheduled rotation with overlap periods.
5. **Use delegation carefully** — OAuth2 token exchange for user-delegated agent actions.
6. **Federate for cross-org** — use federation protocols for agents that span organizational boundaries.
7. **Audit everything** — every identity event must be logged and immutable.
8. **Assume compromise** — design for credential compromise with revocation and rotation paths.

---

**References**
- OAuth 2.0 Framework (RFC 6749)
- OAuth 2.0 Token Exchange (RFC 8693)
- SPIFFE/SPIRE Standard
- W3C Decentralized Identifiers (DIDs)
- HashiCorp Vault Documentation
- IAM Roles Anywhere Documentation
- NIST SP 800-63: Digital Identity Guidelines

---

**Document Information**
- Title: Agent Authentication and Identity
- Series: 18-Agent-Security-and-Trust
- Part: 05 of 08
- Author: AI Knowledge Base
- Last Updated: 2026-06-13
- Lines: 598

---
**See also:**
- [07 — AI Coding Agent Sustainability: Managing Hidden Costs, Code Quality Decay, and Developer Burnout](33-AI-Native-Software-Development/07-AI-Coding-Agent-Sustainability.md)
- [48 — MCP Cloud Infrastructure & Agent-as-a-Service: The Production Layer of Agentic AI](48-MCP-Cloud-Infrastructure-Agent-as-a-Service/01-Overview.md)
- [03 — Agent-Native Orchestration: LangGraph, Conductor, Mistral Workflows, Mcp-Agent](31-AI-Workflow-Orchestration-and-Durable-Execution/03-Agent-Native-Orchestration.md)
