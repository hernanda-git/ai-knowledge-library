# 02 — Core Topics: MCP Server Lifecycle, Protocol Details, and Service Mesh Patterns

> **Category:** 48-MCP-Cloud-Infrastructure-Agent-as-a-Service
> **Last Updated:** July 2026
> **Difficulty:** Intermediate to Advanced
> **Cross-References:** [48-01-Overview.md](./01-Overview.md), [03-Agents/04-Protocols-MCP-ACP.md](../03-Agents/04-Protocols-MCP-ACP.md)

---

## Table of Contents

1. [MCP Server Lifecycle in the Cloud](#1-mcp-server-lifecycle-in-the-cloud)
2. [Protocol Deep-Dive: MCP 1.3 for Cloud Deployment](#2-protocol-deep-dive-mcp-13-for-cloud-deployment)
3. [Transport Layer: HTTP, SSE, and WebSocket](#3-transport-layer-http-sse-and-websocket)
4. [Authentication & Authorization Patterns](#4-authentication--authorization-patterns)
5. [Service Discovery & Registry](#5-service-discovery--registry)
6. [MCP Service Mesh Architecture](#6-mcp-service-mesh-architecture)
7. [Tool & Resource Management at Scale](#7-tool--resource-management-at-scale)
8. [Multi-Tenancy Patterns](#8-multi-tenancy-patterns)
9. [Versioning & Backward Compatibility](#9-versioning--backward-compatibility)
10. [Cross-References to Existing Library Docs](#10-cross-references-to-existing-library-docs)

---

## 1. MCP Server Lifecycle in the Cloud

### 1.1 The Five Phases

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  Build   │──▶│  Deploy  │──▶│  Operate │──▶│  Scale   │──▶│ Deprecate│
│          │   │          │   │          │   │          │   │          │
│ • Code   │   │ • Package│   │ • Monitor│   │ • Auto   │   │ • Sunset │
│ • Test   │   │ • Config │   │ • Debug  │   │ • Shard  │   │ • Migrate│
│ • Review │   │ • Auth   │   │ • Patch  │   │ • Cache  │   │ • Archive│
│ • Scan   │   │ • Route  │   │ • Audit  │   │ • Route  │   │          │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
```

### 1.2 Build Phase

```python
# mcp_server.py — Define a cloud-ready MCP server
from fastmcp import FastMCP
from pydantic import BaseModel, Field
from typing import Optional
import hashlib

mcp = FastMCP(
    name="order-service",
    version="2.1.0",
    description="Cloud-native order management MCP server",
)

class OrderQuery(BaseModel):
    """Query parameters for order lookup."""
    customer_id: str = Field(description="Customer identifier")
    status: Optional[str] = Field(default=None, description="Filter by status")
    limit: int = Field(default=50, ge=1, le=500, description="Max results")

class OrderResult(BaseModel):
    """Single order result."""
    order_id: str
    customer_id: str
    status: str
    total: float
    created_at: str

@mcp.tool()
def query_orders(params: OrderQuery) -> list[OrderResult]:
    """Query orders with filtering. Read-only, tenant-scoped."""
    # Tool implementation
    pass

@mcp.resource("orders://stats/{tenant_id}")
def get_order_stats(tenant_id: str) -> dict:
    """Aggregate order statistics for a tenant."""
    pass

@mcp.prompt()
def order_review_prompt(order_id: str) -> str:
    """Generate a prompt for reviewing an order."""
    return f"Please review order {order_id} and provide a quality assessment."
```

### 1.3 Deploy Phase

```python
# deploy.py — Deploy to cloud MCP platform
from manufact_sdk import DeployConfig, MCPServer

config = DeployConfig(
    # Compute
    replicas=3,
    cpu="0.5",
    memory="512Mi",
    gpu=None,  # CPU-only for most tool servers
    
    # Networking
    transport="streamable-http",
    port=8000,
    health_check="/health",
    readiness_check="/ready",
    
    # Auth
    auth={
        "method": "oauth2.1",
        "issuer": "https://auth.company.com",
        "scopes": ["mcp:orders:read"],
    },
    
    # Rate limiting
    rate_limit={
        "global": "5000/min",
        "per_user": "100/min",
        "per_tool": {
            "query_orders": "50/min",
            "get_order_stats": "20/min",
        }
    },
    
    # Environment
    env={
        "DATABASE_URL": "managed://order-db-pool",
        "CACHE_URL": "managed://order-cache",
        "LOG_LEVEL": "INFO",
    },
    
    # Compliance
    data_residency="us-east-1",
    encryption_at_rest=True,
    audit_logging=True,
)

server = MCPServer.from_file("mcp_server.py")
deployment = server.deploy(config)
print(f"MCP Server deployed: {deployment.url}")
# Output: https://mcp.manufact.dev/servers/order-service/v2.1.0
```

### 1.4 Operate Phase

```python
# monitoring.py — Observe MCP server health
from observability import MCPMonitor

monitor = MCPMonitor(
    server="order-service",
    alert_rules=[
        {
            "name": "high_latency",
            "metric": "mcp_tool_latency_p99",
            "threshold": "500ms",
            "window": "5m",
            "action": "page-oncall",
        },
        {
            "name": "error_rate",
            "metric": "mcp_tool_error_rate",
            "threshold": "5%",
            "window": "1m",
            "action": "auto-scale-up",
        },
        {
            "name": "cost_spike",
            "metric": "mcp_tool_cost_per_hour",
            "threshold": "$50",
            "window": "1h",
            "action": "alert-finance",
        },
    ],
    dashboards={
        "overview": "https://grafana.company.com/d/mcp-order",
        "security": "https://grafana.company.com/d/mcp-order-security",
    },
)
```

---

## 2. Protocol Deep-Dive: MCP 1.3 for Cloud Deployment

### 2.1 Message Format

MCP uses JSON-RPC 2.0 as its message format:

```json
// Tool call request
{
  "jsonrpc": "2.0",
  "id": "req-abc-123",
  "method": "tools/call",
  "params": {
    "name": "query_orders",
    "arguments": {
      "params": {
        "customer_id": "cust-456",
        "status": "shipped",
        "limit": 10
      }
    }
  }
}

// Tool call response
{
  "jsonrpc": "2.0",
  "id": "req-abc-123",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Found 7 shipped orders for customer cust-456"
      }
    ],
    "isError": false,
    "metadata": {
      "tool": "query_orders",
      "latency_ms": 45,
      "tenant_id": "tenant-789",
      "cost_micros": 12
    }
  }
}
```

### 2.2 Cloud-Specific Capabilities

MCP 1.3 introduced capabilities specifically for cloud deployment:

```json
// Server capability announcement
{
  "capabilities": {
    "tools": { "listChanged": true },
    "resources": { "subscribe": true },
    "prompts": { "listChanged": true },
    "logging": {},
    // Cloud-specific capabilities
    "cloud": {
      "multiTenant": true,
      "authMethods": ["oauth2.1", "api_key", "mtls"],
      "maxConcurrent": 100,
      "rateLimits": true,
      "auditLogging": true,
      "dataResidency": ["us-east-1", "eu-west-1", "ap-southeast-1"],
      "versioning": "semver",
      "healthCheck": "/health",
      "capabilityNegotiation": true
    }
  }
}
```

### 2.3 Capability Negotiation

Agents can negotiate capabilities with servers at connection time:

```python
# Agent-side capability negotiation
client = MCPClient(
    server_url="https://mcp.company.com/order-service",
    requested_capabilities={
        "tools": True,
        "resources": True,
        "auth": "oauth2.1",
        "max_latency_ms": 500,
        "require_audit": True,
    }
)

# Server responds with actual capabilities
server_caps = client.connect()
print(server_caps)
# {
#   "tools": {"count": 3, "listChanged": true},
#   "resources": {"count": 5, "subscribe": true},
#   "auth": "oauth2.1",
#   "actual_latency_ms": 32,
#   "audit": "enabled",
#   "data_residency": "us-east-1",
# }
```

---

## 3. Transport Layer: HTTP, SSE, and WebSocket

### 3.1 Transport Comparison

| Transport | Use Case | Latency | Complexity | Cloud Support |
|-----------|----------|---------|------------|---------------|
| **stdio** | Local only | Lowest | Simplest | ❌ Not applicable |
| **HTTP + SSE** | Request/response + streaming | Low | Moderate | ✅ Standard |
| **WebSocket** | Bidirectional streaming | Lowest | Higher | ✅ Growing |
| **gRPC** | High-performance | Very low | High | ✅ Enterprise |

### 3.2 HTTP + SSE (Recommended for Cloud)

```python
# SSE transport for streaming tool results
from fastmcp import FastMCP
from sse_starlette import EventSourceResponse

mcp = FastMCP("streaming-analytics")

@mcp.tool()
async def stream_large_dataset(query: str) -> AsyncGenerator[str, None]:
    """Stream large result sets via SSE."""
    async with db.execute(query) as cursor:
        async for row in cursor:
            yield json.dumps(row)

# SSE endpoint for streaming
@app.post("/mcp/stream")
async def mcp_stream(request: Request):
    # ... authentication ...
    async def event_generator():
        async for chunk in mcp.handle_stream(request):
            yield {
                "event": "mcp/message",
                "data": json.dumps(chunk),
            }
    return EventSourceResponse(event_generator())
```

### 3.3 WebSocket for Bidirectional

```python
# WebSocket transport for real-time agent-server communication
from fastmcp import FastMCP
import websockets

mcp = FastMCP("real-time-collab")

@mcp.tool()
async def subscribe_to_updates(channel: str) -> dict:
    """Subscribe to real-time updates via WebSocket."""
    return {"status": "subscribed", "channel": channel}

# WebSocket server
async def websocket_handler(websocket):
    async for message in websocket:
        response = await mcp.handle_message(message)
        await websocket.send(json.dumps(response))

async def main():
    async with websockets.serve(websocket_handler, "0.0.0.0", 8765):
        await asyncio.Future()  # Run forever
```

---

## 4. Authentication & Authorization Patterns

### 4.1 OAuth 2.1 for MCP

The recommended auth pattern for cloud MCP:

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│   Agent  │────▶│  AuthZ   │────▶│ MCP Srv  │
│          │     │  Server  │     │          │
│ 1. Get   │     │          │     │          │
│    token │     │ 2. Issue │     │          │
│          │◀────│    token │     │          │
│ 3. Call  │     │          │     │ 4. Verify│
│    MCP   │────▶│          │────▶│    token │
│          │     │          │     │          │
│ 5. Get   │     │          │     │          │
│    result│◀────│          │◀────│          │
└──────────┘     └──────────┘     └──────────┘
```

### 4.2 Token Scopes for MCP

```python
# Scope definitions for MCP operations
MCP_SCOPES = {
    # Tool scopes
    "mcp:tools:invoke": "Invoke MCP tools",
    "mcp:tools:list": "List available tools",
    
    # Resource scopes
    "mcp:resources:read": "Read MCP resources",
    "mcp:resources:write": "Write MCP resources",
    "mcp:resources:subscribe": "Subscribe to resource changes",
    
    # Prompt scopes
    "mcp:prompts:read": "Read MCP prompts",
    
    # Admin scopes
    "mcp:servers:manage": "Manage MCP server deployments",
    "mcp:registry:admin": "Manage server registry",
    
    # Data scopes
    "mcp:data:export": "Export data from MCP servers",
    "mcp:data:delete": "Delete data via MCP servers",
}
```

### 4.3 Delegated Authentication

When an agent acts on behalf of a user:

```python
# Delegated auth: user token → agent token → MCP server token
class DelegatedAuth:
    async def get_mcp_token(self, user_token: str, server_id: str) -> str:
        # 1. Validate user token
        user_claims = await self.validate_token(user_token)
        
        # 2. Create agent token (scoped to user's permissions)
        agent_token = await self.create_agent_token(
            subject=user_claims["sub"],
            scopes=self.get_allowed_scopes(user_claims, server_id),
            audience=f"mcp://{server_id}",
            ttl=300,  # 5 minutes
        )
        
        # 3. Exchange for MCP server token
        mcp_token = await self.exchange_token(
            agent_token,
            target_server=server_id,
        )
        
        return mcp_token
```

### 4.4 API Key Authentication (Simpler Alternative)

```python
# API key auth for simpler use cases
@mcp.middleware
async def api_key_auth(request, call_next):
    api_key = request.headers.get("X-MCP-API-Key")
    if not api_key:
        return JSONResponse({"error": "Missing API key"}, status_code=401)
    
    # Validate and associate with tenant
    tenant = await validate_api_key(api_key)
    request.state.tenant = tenant
    request.state.scopes = get_scopes_for_key(api_key)
    
    return await call_next(request)
```

---

## 5. Service Discovery & Registry

### 5.1 Registry Schema

```json
{
  "registry": {
    "servers": [
      {
        "id": "order-service-v2.1.0",
        "name": "Order Service",
        "description": "Cloud-native order management",
        "version": "2.1.0",
        "url": "https://mcp.company.com/orders",
        "transport": "streamable-http",
        "capabilities": ["tools", "resources"],
        "tools": [
          {
            "name": "query_orders",
            "description": "Query orders with filtering",
            "inputSchema": { "..." },
            "cost_per_call": 0.001,
            "avg_latency_ms": 45
          }
        ],
        "auth": {
          "method": "oauth2.1",
          "scopes_required": ["mcp:orders:read"]
        },
        "metadata": {
          "owner": "platform-team",
          "tier": "production",
          "sla_uptime": 99.9,
          "data_residency": ["us-east-1", "eu-west-1"],
          "compliance": ["soc2", "hipaa"],
          "rating": 4.7,
          "usage_count": 1250000,
          "last_updated": "2026-07-01T10:30:00Z"
        }
      }
    ]
  }
}
```

### 5.2 Discovery Workflow

```python
# Agent discovers and connects to MCP servers
class MCPDiscovery:
    def __init__(self, registry_url: str, auth_token: str):
        self.registry = MCPRegistryClient(registry_url, auth_token)
    
    async def find_server(
        self,
        capability: str,
        tags: list[str] = None,
        min_uptime: float = 0.99,
        max_latency_ms: int = 100,
    ) -> MCPServerConnection:
        # Query registry
        candidates = await self.registry.search(
            capability=capability,
            tags=tags or [],
            filters={
                "metadata.sla_uptime": {"$gte": min_uptime},
                "metadata.avg_latency_ms": {"$lte": max_latency_ms},
            }
        )
        
        # Rank by composite score
        ranked = sorted(
            candidates,
            key=lambda s: self._score(s),
            reverse=True
        )
        
        # Connect to best candidate
        return await self.connect(ranked[0])
    
    def _score(self, server) -> float:
        """Composite score: uptime * rating / latency * cost"""
        return (
            server.metadata.sla_uptime *
            server.metadata.rating /
            (server.metadata.avg_latency_ms / 100) *
            (1 / max(server.tools[0].cost_per_call, 0.0001))
        )
```

### 5.3 Health Checking

```python
# MCP server health check
class MCPHealthChecker:
    async def check(self, server_url: str) -> HealthStatus:
        # 1. HTTP health endpoint
        health = await self.http.get(f"{server_url}/health")
        
        # 2. MCP capability check
        caps = await self.mcp_client.call(
            server_url,
            method="initialize",
            params={"clientInfo": {"name": "health-checker"}}
        )
        
        # 3. Tool invocation test
        test_result = await self.mcp_client.call(
            server_url,
            method="tools/call",
            params={"name": "health_check_tool", "arguments": {}}
        )
        
        return HealthStatus(
            healthy=health.ok and caps.ok and test_result.ok,
            latency_ms=health.latency,
            version=caps.result.get("serverInfo", {}).get("version"),
            capabilities=caps.result.get("capabilities", {}),
        )
```

---

## 6. MCP Service Mesh Architecture

### 6.1 What is MCP Mesh?

An MCP service mesh extends the concept of a service mesh (like Istio/Linkerd) to MCP servers:

```
┌─────────────────────────────────────────────────────┐
│                    MCP Service Mesh                   │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ MCP Srv  │  │ MCP Srv  │  │ MCP Srv  │          │
│  │ (Orders) │  │ (Users)  │  │ (Payments)│          │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘          │
│       │              │              │                 │
│  ┌────▼──────────────▼──────────────▼────┐          │
│  │           Sidecar Proxy                │          │
│  │  • mTLS between servers               │          │
│  │  • Load balancing                     │          │
│  │  • Circuit breaking                   │          │
│  │  • Distributed tracing                │          │
│  │  • Rate limiting                      │          │
│  └───────────────────────────────────────┘          │
└─────────────────────────────────────────────────────┘
```

### 6.2 Sidecar Proxy for MCP

```python
# MCP sidecar proxy
class MCPSidecar:
    def __init__(self, upstream_url: str, config: MeshConfig):
        self.upstream = upstream_url
        self.config = config
        self.tracer = OpenTelemetryTracer()
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30,
        )
        self.rate_limiter = TokenBucket(
            rate=config.rate_limit,
            burst=config.rate_burst,
        )
    
    async def handle(self, request: MCPRequest) -> MCPResponse:
        # 1. Rate limiting
        if not self.rate_limiter.acquire():
            return MCPResponse.error(429, "Rate limit exceeded")
        
        # 2. Circuit breaker
        if self.circuit_breaker.is_open():
            return MCPResponse.error(503, "Circuit breaker open")
        
        # 3. Add tracing headers
        with self.tracer.span(f"mcp-proxy/{request.method}") as span:
            request.headers["X-Request-ID"] = str(uuid4())
            request.headers["X-Trace-ID"] = span.trace_id
            
            # 4. Forward to upstream
            try:
                response = await self.forward(request)
                self.circuit_breaker.record_success()
                return response
            except Exception as e:
                self.circuit_breaker.record_failure()
                return MCPResponse.error(502, f"Upstream error: {e}")
```

### 6.3 Cross-Server Tool Composition

```python
# Compose tools from multiple MCP servers
class MCPComposer:
    """Compose tools from multiple MCP servers into higher-level operations."""
    
    def __init__(self):
        self.servers = {}
    
    async def add_server(self, name: str, url: str, auth_token: str):
        self.servers[name] = await MCPClient.connect(url, auth_token)
    
    @tool("complete_order")
    async def complete_order(self, order_id: str, payment_method: str):
        """High-level tool that orchestrates across multiple MCP servers."""
        # Step 1: Get order details (from orders server)
        order = await self.servers["orders"].call("get_order", {"order_id": order_id})
        
        # Step 2: Process payment (from payments server)
        payment = await self.servers["payments"].call("process_payment", {
            "amount": order.total,
            "method": payment_method,
            "reference": order_id,
        })
        
        # Step 3: Update order status (from orders server)
        await self.servers["orders"].call("update_order_status", {
            "order_id": order_id,
            "status": "paid",
            "payment_id": payment.id,
        })
        
        # Step 4: Send confirmation (from notifications server)
        await self.servers["notifications"].call("send_confirmation", {
            "customer_id": order.customer_id,
            "order_id": order_id,
            "payment_id": payment.id,
        })
        
        return {"status": "completed", "order_id": order_id, "payment_id": payment.id}
```

---

## 7. Tool & Resource Management at Scale

### 7.1 Tool Catalog Management

```python
# Manage tool catalog across multiple MCP servers
class ToolCatalog:
    def __init__(self):
        self.tools = {}  # name -> ToolMetadata
        self.servers = {}  # server_id -> MCPServerConnection
    
    async def register_server(self, server_id: str, url: str):
        """Register an MCP server and discover its tools."""
        client = await MCPClient.connect(url)
        self.servers[server_id] = client
        
        # Discover tools
        tools = await client.list_tools()
        for tool in tools:
            self.tools[tool.name] = ToolMetadata(
                name=tool.name,
                description=tool.description,
                server=server_id,
                input_schema=tool.inputSchema,
                cost_per_call=tool.metadata.get("cost_per_call", 0),
                avg_latency_ms=tool.metadata.get("avg_latency_ms", 50),
            )
    
    def get_tool(self, name: str) -> ToolMetadata:
        return self.tools.get(name)
    
    def search_tools(self, query: str, tags: list[str] = None) -> list[ToolMetadata]:
        """Search tools by description and tags."""
        results = []
        for tool in self.tools.values():
            if query.lower() in tool.description.lower():
                if not tags or any(t in tool.tags for t in tags):
                    results.append(tool)
        return results
```

### 7.2 Resource Caching

```python
# Cache MCP resources to reduce server load
class MCPResourceCache:
    def __init__(self, redis_url: str, default_ttl: int = 300):
        self.redis = Redis(redis_url)
        self.default_ttl = default_ttl
    
    async def get_or_fetch(
        self,
        server_url: str,
        resource_uri: str,
        ttl: int = None,
    ) -> dict:
        cache_key = f"mcp:resource:{server_url}:{resource_uri}"
        
        # Check cache
        cached = await self.redis.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # Fetch from server
        client = await MCPClient.connect(server_url)
        result = await client.read_resource(resource_uri)
        
        # Cache result
        ttl = ttl or self.default_ttl
        await self.redis.setex(cache_key, ttl, json.dumps(result))
        
        return result
```

---

## 8. Multi-Tenancy Patterns

### 8.1 Tenant Isolation Strategies

| Strategy | Isolation Level | Cost | Complexity |
|----------|----------------|------|------------|
| **Shared server, scoped queries** | Medium | Low | Low |
| **Shared server, row-level security** | High | Low | Medium |
| **Dedicated server per tenant** | Highest | High | Medium |
| **Namespace isolation** | High | Medium | Medium |

### 8.2 Row-Level Security Pattern

```python
# Multi-tenant MCP server with row-level security
from fastmcp import FastMCP
from tenant import TenantContext

mcp = FastMCP("multi-tenant-crm")

@mcp.tool()
async def search_contacts(query: str) -> list[dict]:
    """Search contacts. Automatically scoped to authenticated tenant."""
    tenant_id = TenantContext.current().tenant_id
    
    # Row-level security enforced at query level
    results = await db.query(
        """
        SELECT id, name, email, company 
        FROM contacts 
        WHERE tenant_id = $1 
        AND (name ILIKE $2 OR email ILIKE $2 OR company ILIKE $2)
        LIMIT 50
        """,
        tenant_id,
        f"%{query}%",
    )
    
    return results

@mcp.resource("crm://contacts/{contact_id}")
async def get_contact(contact_id: str) -> dict:
    """Get contact by ID. Tenant-scoped."""
    tenant_id = TenantContext.current().tenant_id
    
    contact = await db.query_one(
        "SELECT * FROM contacts WHERE id = $1 AND tenant_id = $2",
        contact_id,
        tenant_id,
    )
    
    if not contact:
        raise NotFoundError(f"Contact {contact_id} not found")
    
    return contact
```

### 8.3 Tenant Context Propagation

```python
# Propagate tenant context through MCP calls
class TenantContextMiddleware:
    async def __call__(self, request: Request, call_next):
        # Extract tenant from auth token
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        claims = await validate_token(token)
        
        # Set tenant context
        TenantContext.set(TenantContext(
            tenant_id=claims["tenant_id"],
            user_id=claims["sub"],
            scopes=claims["scopes"],
        ))
        
        # Add tenant header for downstream MCP calls
        request.state.tenant_id = claims["tenant_id"]
        
        response = await call_next(request)
        return response
```

---

## 9. Versioning & Backward Compatibility

### 9.1 Semantic Versioning for MCP Servers

```
order-service v2.1.0
  ├── v2.1.0  (current, stable)
  ├── v2.1.1  (patch: bug fixes)
  ├── v2.2.0  (minor: new tool added)
  ├── v3.0.0  (major: breaking schema change)
  └── v1.x.x  (deprecated, sunset in 90 days)
```

### 9.2 Version Negotiation

```python
# Client requests specific version
client = MCPClient(
    server_url="https://mcp.company.com/orders",
    preferred_version="2.1.x",  # Accept patches
    min_version="2.0.0",        # Don't go below 2.0
)

# Server responds with negotiated version
# If client prefers 2.1.x and server has 2.1.3, uses 2.1.3
# If client prefers 2.1.x but server only has 2.2.0, uses 2.2.0 (backward compatible)
# If server only has 3.0.0 and client min is 2.0.0, error
```

### 9.3 Backward Compatibility Rules

```python
# MCP server backward compatibility checklist
BACKWARD_COMPAT_RULES = {
    # Non-breaking changes (minor/patch)
    "add_tool": "New tool added, existing tools unchanged",
    "add_resource": "New resource added",
    "add_optional_param": "New optional parameter added to existing tool",
    "increase_rate_limit": "Rate limit increased",
    
    # Breaking changes (major)
    "remove_tool": "Existing tool removed",
    "rename_tool": "Existing tool renamed",
    "change_tool_schema": "Existing tool input schema changed",
    "remove_resource": "Existing resource removed",
    "change_auth_method": "Authentication method changed",
    "change_data_residency": "Data residency changed",
}
```

---

## 10. Cross-References to Existing Library Docs

| Topic | Library Doc | Relevance |
|-------|-------------|-----------|
| MCP protocol basics | [03-Agents/04-Protocols-MCP-ACP.md](../03-Agents/04-Protocols-MCP-ACP.md) | Foundation for this document |
| Tool implementations | [03-Agents/05-Tool-Implementations.md](../03-Agents/05-Tool-Implementations.md) | Tool design patterns |
| Agent security | [18-Agent-Security-and-Trust/](../18-Agent-Security-and-Trust/) | Security patterns extend to MCP |
| Agent memory | [32-Agent-Memory-Systems/](../32-Agent-Memory-Systems/) | Memory integration via MCP |
| Enterprise deployment | [05-Enterprise/](../05-Enterprise/) | Enterprise context |
| Cost optimization | [41-AI-Cost-Optimization-and-Enterprise-ROI/](../41-AI-Cost-Optimization-and-Enterprise-ROI/) | MCP cost management |
| Multi-cloud | [25-Multi-Cloud-AI-Strategy/](../25-Multi-Cloud-AI-Strategy/) | Cross-cloud MCP |
| Evaluation benchmarks | [06-Advanced/03-Evaluation-Benchmarks.md](../06-Advanced/03-Evaluation-Benchmarks.md) | Testing MCP servers |

---

*This document is part of the AI Knowledge Library auto-enrichment system.*

---
**See also:**
- [07 — AI Coding Agent Sustainability: Managing Hidden Costs, Code Quality Decay, and Developer Burnout](33-AI-Native-Software-Development/07-AI-Coding-Agent-Sustainability.md)
- [03 — Agent-Native Orchestration: LangGraph, Conductor, Mistral Workflows, Mcp-Agent](31-AI-Workflow-Orchestration-and-Durable-Execution/03-Agent-Native-Orchestration.md)
- [AI Agent Commerce & Agent-to-Agent (A2A) Payments: The 2026 Frontier](28-AI-Agent-Commerce-and-A2A-Payments/01-Overview.md)
