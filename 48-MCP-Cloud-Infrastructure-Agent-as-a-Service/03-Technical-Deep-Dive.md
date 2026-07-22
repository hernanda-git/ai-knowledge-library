# 03 — Technical Deep-Dive: Implementing Production MCP Cloud Infrastructure

> **Category:** 48-MCP-Cloud-Infrastructure-Agent-as-a-Service
> **Last Updated:** July 2026
> **Difficulty:** Advanced
> **Cross-References:** [48-01-Overview.md](./01-Overview.md), [48-02-Core-Topics.md](./02-Core-Topics.md)

---

## Table of Contents

1. [Building a Production MCP Gateway from Scratch](#1-building-a-production-mcp-gateway-from-scratch)
2. [MCP Server Containerization & Orchestration](#2-mcp-server-containerization--orchestration)
3. [Distributed Tracing for MCP](#3-distributed-tracing-for-mcp)
4. [MCP Server Testing & Benchmarking](#4-mcp-server-testing--benchmarking)
5. [Performance Optimization Techniques](#5-performance-optimization-techniques)
6. [Disaster Recovery & High Availability](#6-disaster-recovery--high-availability)
7. [Migration: Local MCP to Cloud MCP](#7-migration-local-mcp-to-cloud-mcp)
8. [Cross-References to Existing Library Docs](#8-cross-references-to-existing-library-docs)

---

## 1. Building a Production MCP Gateway from Scratch

### 1.1 Architecture Overview

```
                        ┌─────────────────┐
                        │   Load Balancer  │
                        │   (L7 / ALB)     │
                        └────────┬────────┘
                                 │
                        ┌────────▼────────┐
                        │   MCP Gateway   │
                        │   (FastAPI)      │
                        ├─────────────────┤
                        │ • Auth middleware│
                        │ • Rate limiter  │
                        │ • Request router│
                        │ • Audit logger  │
                        │ • Circuit breaker│
                        └────────┬────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
     ┌────────▼────────┐ ┌──────▼───────┐ ┌───────▼────────┐
     │  MCP Server A   │ │ MCP Server B │ │  MCP Server C  │
     │  (orders)       │ │ (users)      │ │  (payments)    │
     └─────────────────┘ └──────────────┘ └────────────────┘
```

### 1.2 Full Gateway Implementation

```python
# mcp_gateway.py — Production MCP Gateway
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import httpx
import time
import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from collections import defaultdict
import asyncio
import logging

# ============================================================
# Configuration
# ============================================================

@dataclass
class MCPServerConfig:
    """Configuration for an upstream MCP server."""
    name: str
    url: str
    transport: str = "streamable-http"
    auth_method: str = "oauth2.1"
    rate_limit: int = 1000  # requests per minute
    timeout: float = 30.0
    max_retries: int = 3
    health_check_url: str = "/health"
    tools: List[str] = field(default_factory=list)

@dataclass
class GatewayConfig:
    """Gateway configuration."""
    servers: Dict[str, MCPServerConfig]
    default_timeout: float = 30.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_recovery: float = 30.0
    audit_enabled: bool = True
    cache_ttl: int = 300

# ============================================================
# Rate Limiter
# ============================================================

class TokenBucketRateLimiter:
    """Token bucket rate limiter for MCP requests."""
    
    def __init__(self):
        self.buckets: Dict[str, dict] = {}
    
    async def check(self, key: str, rate_limit: int, window: int = 60) -> bool:
        now = time.time()
        
        if key not in self.buckets:
            self.buckets[key] = {
                "tokens": rate_limit,
                "last_refill": now,
                "rate_limit": rate_limit,
            }
        
        bucket = self.buckets[key]
        
        # Refill tokens
        elapsed = now - bucket["last_refill"]
        refill = elapsed * (bucket["rate_limit"] / window)
        bucket["tokens"] = min(bucket["rate_limit"], bucket["tokens"] + refill)
        bucket["last_refill"] = now
        
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True
        
        return False

# ============================================================
# Circuit Breaker
# ============================================================

class CircuitBreaker:
    """Circuit breaker for upstream MCP servers."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures: Dict[str, int] = defaultdict(int)
        self.open_until: Dict[str, float] = {}
    
    def is_open(self, server: str) -> bool:
        if server in self.open_until:
            if time.time() < self.open_until[server]:
                return True
            else:
                # Recovery window — allow one probe request
                del self.open_until[server]
                self.failures[server] = 0
        return False
    
    def record_success(self, server: str):
        self.failures[server] = 0
    
    def record_failure(self, server: str):
        self.failures[server] += 1
        if self.failures[server] >= self.failure_threshold:
            self.open_until[server] = time.time() + self.recovery_timeout
            logging.warning(
                f"Circuit breaker OPEN for {server}. "
                f"Will retry after {self.recovery_timeout}s"
            )

# ============================================================
# Audit Logger
# ============================================================

class AuditLogger:
    """Structured audit logging for MCP operations."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
    
    async def log(
        self,
        event_type: str,
        request: Request,
        server: str,
        tool: Optional[str] = None,
        latency_ms: Optional[float] = None,
        status: str = "success",
        error: Optional[str] = None,
    ):
        if not self.enabled:
            return
        
        entry = {
            "timestamp": time.time(),
            "event_type": event_type,
            "server": server,
            "tool": tool,
            "method": request.method,
            "path": str(request.url.path),
            "status": status,
            "latency_ms": latency_ms,
            "error": error,
            "request_id": request.headers.get("X-Request-ID", "unknown"),
            "tenant_id": getattr(request.state, "tenant_id", "unknown"),
            "user_agent": request.headers.get("User-Agent", "unknown"),
            "ip": request.client.host if request.client else "unknown",
        }
        
        # In production, send to SIEM, S3, or log aggregation
        logging.info(json.dumps(entry))

# ============================================================
# MCP Gateway
# ============================================================

class MCPGateway:
    """Production MCP Gateway with auth, rate limiting, routing."""
    
    def __init__(self, config: GatewayConfig):
        self.config = config
        self.rate_limiter = TokenBucketRateLimiter()
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.circuit_breaker_threshold,
            recovery_timeout=config.circuit_breaker_recovery,
        )
        self.audit = AuditLogger(config.audit_enabled)
        self.tool_to_server: Dict[str, str] = {}
        
        # Build tool-to-server mapping
        for name, server in config.servers.items():
            for tool in server.tools:
                self.tool_to_server[tool] = name
    
    async def route_request(
        self,
        request: Request,
        mcp_method: str,
        params: dict,
    ) -> dict:
        """Route an MCP request to the appropriate server."""
        start_time = time.time()
        
        # 1. Determine target server
        server_name = self._resolve_server(mcp_method, params)
        if not server_name:
            raise HTTPException(404, f"No server found for method: {mcp_method}")
        
        server_config = self.config.servers[server_name]
        
        # 2. Check circuit breaker
        if self.circuit_breaker.is_open(server_name):
            await self.audit.log(
                "circuit_breaker_open", request, server_name
            )
            raise HTTPException(503, f"Server {server_name} is temporarily unavailable")
        
        # 3. Rate limiting
        rate_key = f"{server_name}:{getattr(request.state, 'tenant_id', 'default')}"
        if not await self.rate_limiter.check(rate_key, server_config.rate_limit):
            await self.audit.log("rate_limited", request, server_name)
            raise HTTPException(429, "Rate limit exceeded")
        
        # 4. Forward request
        try:
            response = await self._forward_request(
                server_config, mcp_method, params, request
            )
            self.circuit_breaker.record_success(server_name)
            
            latency_ms = (time.time() - start_time) * 1000
            tool_name = params.get("name") if mcp_method == "tools/call" else None
            
            await self.audit.log(
                "tool_call", request, server_name,
                tool=tool_name, latency_ms=latency_ms,
            )
            
            return response
            
        except Exception as e:
            self.circuit_breaker.record_failure(server_name)
            await self.audit.log(
                "error", request, server_name,
                status="error", error=str(e),
            )
            raise HTTPException(502, f"Upstream error: {e}")
    
    def _resolve_server(self, method: str, params: dict) -> Optional[str]:
        """Resolve which server handles this request."""
        # For tool calls, look up by tool name
        if method == "tools/call":
            tool_name = params.get("name", "")
            return self.tool_to_server.get(tool_name)
        
        # For resource operations, look up by URI pattern
        if method in ("resources/read", "resources/list"):
            uri = params.get("uri", "")
            for name, server in self.config.servers.items():
                if uri.startswith(f"{name}://"):
                    return name
        
        # Default to first server
        return next(iter(self.config.servers), None)
    
    async def _forward_request(
        self,
        server_config: MCPServerConfig,
        method: str,
        params: dict,
        original_request: Request,
    ) -> dict:
        """Forward MCP request to upstream server."""
        mcp_request = {
            "jsonrpc": "2.0",
            "id": original_request.headers.get("X-Request-ID", str(time.time())),
            "method": method,
            "params": params,
        }
        
        async with httpx.AsyncClient(timeout=server_config.timeout) as client:
            # Add auth headers
            headers = {
                "Content-Type": "application/json",
                "X-Forwarded-For": original_request.client.host,
                "X-Tenant-ID": getattr(original_request.state, "tenant_id", ""),
            }
            
            # Add authorization if available
            auth_header = original_request.headers.get("Authorization")
            if auth_header:
                headers["Authorization"] = auth_header
            
            response = await client.post(
                server_config.url,
                json=mcp_request,
                headers=headers,
            )
            
            return response.json()

# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(title="MCP Gateway", version="1.0.0")

# Initialize gateway
config = GatewayConfig(
    servers={
        "orders": MCPServerConfig(
            name="orders",
            url="https://mcp.internal/orders",
            tools=["query_orders", "get_order", "update_order"],
            rate_limit=1000,
        ),
        "users": MCPServerConfig(
            name="users",
            url="https://mcp.internal/users",
            tools=["get_user", "search_users", "update_user"],
            rate_limit=500,
        ),
        "payments": MCPServerConfig(
            name="payments",
            url="https://mcp.internal/payments",
            tools=["process_payment", "refund_payment", "get_transactions"],
            rate_limit=200,
        ),
    }
)

gateway = MCPGateway(config)

@app.post("/mcp")
async def mcp_endpoint(request: Request):
    """Main MCP gateway endpoint."""
    body = await request.json()
    
    # Validate JSON-RPC format
    if body.get("jsonrpc") != "2.0":
        raise HTTPException(400, "Invalid JSON-RPC format")
    
    method = body.get("method")
    params = body.get("params", {})
    
    # Route to appropriate server
    response = await gateway.route_request(request, method, params)
    
    return response

@app.get("/health")
async def health():
    """Gateway health check."""
    return {"status": "healthy", "servers": len(config.servers)}

@app.get("/servers")
async def list_servers():
    """List available MCP servers and their tools."""
    servers = {}
    for name, server in config.servers.items():
        servers[name] = {
            "url": server.url,
            "tools": server.tools,
            "rate_limit": server.rate_limit,
            "circuit_breaker_open": gateway.circuit_breaker.is_open(name),
        }
    return {"servers": servers}
```

### 1.3 Running the Gateway

```bash
# Start the MCP gateway
uvicorn mcp_gateway:app --host 0.0.0.0 --port 9000 --workers 4

# Health check
curl http://localhost:9000/health

# List servers
curl http://localhost:9000/servers

# Forward MCP request
curl -X POST http://localhost:9000/mcp \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "jsonrpc": "2.0",
    "id": "1",
    "method": "tools/call",
    "params": {
      "name": "query_orders",
      "arguments": {
        "params": {"customer_id": "cust-123", "limit": 5}
      }
    }
  }'
```

---

## 2. MCP Server Containerization & Orchestration

### 2.1 Dockerfile for MCP Server

```dockerfile
# Dockerfile for MCP server
FROM python:3.12-slim AS base

# Security: non-root user
RUN groupadd -r mcp && useradd -r -g mcp mcp

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY --chown=mcp:mcp . .

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run as non-root
USER mcp

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2.2 Kubernetes Deployment

```yaml
# k8s-mcp-server.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-order-service
  namespace: mcp-system
  labels:
    app: mcp-order-service
    tier: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mcp-order-service
  template:
    metadata:
      labels:
        app: mcp-order-service
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
    spec:
      containers:
        - name: mcp-server
          image: registry.company.com/mcp/order-service:2.1.0
          ports:
            - containerPort: 8000
              name: http
            - containerPort: 9090
              name: metrics
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: mcp-secrets
                  key: order-db-url
            - name: LOG_LEVEL
              value: "INFO"
          resources:
            requests:
              cpu: "250m"
              memory: "256Mi"
            limits:
              cpu: "500m"
              memory: "512Mi"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: mcp-order-service
  namespace: mcp-system
spec:
  selector:
    app: mcp-order-service
  ports:
    - port: 80
      targetPort: 8000
      name: http
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: mcp-order-service-hpa
  namespace: mcp-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: mcp-order-service
  minReplicas: 3
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Pods
      pods:
        metric:
          name: mcp_requests_per_second
        target:
          type: AverageValue
          averageValue: "100"
```

### 2.3 Helm Chart Structure

```
mcp-server-chart/
├── Chart.yaml
├── values.yaml
├── templates/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── hpa.yaml
│   ├── ingress.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   ├── serviceaccount.yaml
│   ├── networkpolicy.yaml
│   └── servicemonitor.yaml
└── README.md
```

---

## 3. Distributed Tracing for MCP

### 3.1 OpenTelemetry Integration

```python
# tracing.py — OpenTelemetry for MCP
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

# Initialize tracing
provider = TracerProvider()
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="otel-collector:4317"))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("mcp-gateway")

# Instrument FastAPI
FastAPIInstrumentor.instrument_app(app)

# Instrument HTTP client
HTTPXClientInstrumentor().instrument()

# Custom MCP spans
class MCPSpan:
    """Create meaningful spans for MCP operations."""
    
    @staticmethod
    @tracer.start_as_current_span("mcp.tool_call")
    async def tool_call(server: str, tool: str, args: dict) -> dict:
        span = trace.get_current_span()
        span.set_attribute("mcp.server", server)
        span.set_attribute("mcp.tool", tool)
        span.set_attribute("mcp.args_size", len(json.dumps(args)))
        
        try:
            result = await gateway.route_request(
                request, "tools/call",
                {"name": tool, "arguments": args}
            )
            span.set_attribute("mcp.success", True)
            span.set_attribute("mcp.result_size", len(json.dumps(result)))
            return result
        except Exception as e:
            span.set_attribute("mcp.success", False)
            span.set_attribute("mcp.error", str(e))
            raise
```

### 3.2 Trace Visualization

```
Trace: mcp-gateway-request (span_id: abc123)
├── mcp-gateway.auth (45ms)
│   └── oauth2.validate_token (32ms)
├── mcp-gateway.route (3ms)
│   └── mcp-gateway.resolve_server (1ms)
├── mcp-gateway.forward (127ms)
│   └── httpx.post (125ms)
│       └── upstream.mcp-server.tool_call (120ms)
│           └── db.query (85ms)
└── mcp-gateway.audit (2ms)
```

---

## 4. MCP Server Testing & Benchmarking

### 4.1 Unit Testing MCP Tools

```python
# test_order_tools.py
import pytest
from mcp_server import mcp
from testfixtures import LogCapture

@pytest.fixture
def mock_db():
    """Mock database for testing."""
    with patch("mcp_server.db") as db:
        db.query.return_value = [
            {"order_id": "ord-1", "status": "shipped", "total": 99.99}
        ]
        yield db

@pytest.mark.asyncio
async def test_query_orders(mock_db):
    """Test order query tool."""
    result = await mcp.call_tool("query_orders", {
        "params": {"customer_id": "cust-123", "limit": 10}
    })
    
    assert result is not None
    assert len(result["content"]) > 0
    mock_db.query.assert_called_once()

@pytest.mark.asyncio
async def test_query_orders_empty():
    """Test order query with no results."""
    with patch("mcp_server.db") as db:
        db.query.return_value = []
        
        result = await mcp.call_tool("query_orders", {
            "params": {"customer_id": "cust-nonexistent"}
        })
        
        assert "No orders found" in result["content"][0]["text"]

@pytest.mark.asyncio
async def test_query_orders_rate_limit():
    """Test rate limiting."""
    from mcp_gateway import TokenBucketRateLimiter
    
    limiter = TokenBucketRateLimiter()
    
    # Should allow first 1000 requests
    for i in range(1000):
        assert await limiter.check("test:tenant-1", 1000)
    
    # Should block after rate limit
    assert not await limiter.check("test:tenant-1", 1000)
```

### 4.2 Integration Testing

```python
# test_integration.py — Integration tests against running MCP server
import pytest
from mcp_client import MCPClient

@pytest.fixture
async def client():
    """Connect to running MCP server."""
    client = await MCPClient.connect(
        "http://localhost:8000",
        auth_token="test-token"
    )
    yield client
    await client.disconnect()

@pytest.mark.integration
async def test_full_tool_flow(client):
    """Test complete tool invocation flow."""
    # 1. List available tools
    tools = await client.list_tools()
    assert len(tools) > 0
    
    # 2. Call a tool
    result = await client.call_tool("health_check", {})
    assert result.get("content") is not None
    
    # 3. Read a resource
    resources = await client.list_resources()
    assert len(resources) > 0

@pytest.mark.integration
async def test_auth_failure():
    """Test unauthorized access is rejected."""
    client = await MCPClient.connect(
        "http://localhost:8000",
        auth_token="invalid-token"
    )
    
    with pytest.raises(AuthenticationError):
        await client.call_tool("query_orders", {"params": {"customer_id": "test"}})
```

### 4.3 Load Testing

```python
# load_test.py — MCP server load testing
import asyncio
import time
import httpx
import statistics
from dataclasses import dataclass

@dataclass
class LoadTestResult:
    total_requests: int
    successful: int
    failed: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    requests_per_second: float

async def run_load_test(
    url: str,
    concurrent_users: int = 50,
    duration_seconds: int = 60,
    auth_token: str = "test-token",
) -> LoadTestResult:
    """Run load test against MCP server."""
    
    results = []
    start_time = time.time()
    end_time = start_time + duration_seconds
    
    async def send_request(client, request_id):
        request_start = time.time()
        try:
            response = await client.post(
                url,
                json={
                    "jsonrpc": "2.0",
                    "id": str(request_id),
                    "method": "tools/call",
                    "params": {
                        "name": "health_check",
                        "arguments": {}
                    }
                },
                headers={
                    "Authorization": f"Bearer {auth_token}",
                    "Content-Type": "application/json",
                }
            )
            latency = (time.time() - request_start) * 1000
            return {"status": response.status_code, "latency_ms": latency}
        except Exception as e:
            latency = (time.time() - request_start) * 1000
            return {"status": "error", "latency_ms": latency, "error": str(e)}
    
    async with httpx.AsyncClient(timeout=30) as client:
        tasks = []
        request_id = 0
        
        while time.time() < end_time:
            # Launch batch of concurrent requests
            batch_tasks = []
            for _ in range(min(concurrent_users, 100)):
                request_id += 1
                batch_tasks.append(send_request(client, request_id))
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            results.extend([r for r in batch_results if isinstance(r, dict)])
            
            await asyncio.sleep(0.1)  # Small delay between batches
    
    # Calculate statistics
    latencies = [r["latency_ms"] for r in results if r.get("status") != "error"]
    successful = sum(1 for r in results if r.get("status") == 200)
    
    return LoadTestResult(
        total_requests=len(results),
        successful=successful,
        failed=len(results) - successful,
        avg_latency_ms=statistics.mean(latencies) if latencies else 0,
        p50_latency_ms=statistics.median(latencies) if latencies else 0,
        p95_latency_ms=sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
        p99_latency_ms=sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0,
        requests_per_second=len(results) / duration_seconds,
    )

# Run the test
async def main():
    result = await run_load_test(
        url="http://localhost:8000/mcp",
        concurrent_users=50,
        duration_seconds=60,
    )
    
    print(f"Total Requests: {result.total_requests}")
    print(f"Successful: {result.successful}")
    print(f"Failed: {result.failed}")
    print(f"Avg Latency: {result.avg_latency_ms:.1f}ms")
    print(f"P50 Latency: {result.p50_latency_ms:.1f}ms")
    print(f"P95 Latency: {result.p95_latency_ms:.1f}ms")
    print(f"P99 Latency: {result.p99_latency_ms:.1f}ms")
    print(f"RPS: {result.requests_per_second:.1f}")

asyncio.run(main())
```

---

## 5. Performance Optimization Techniques

### 5.1 Connection Pooling

```python
# Reuse HTTP connections to MCP servers
import httpx
from contextlib import asynccontextmanager

class MCPConnectionPool:
    def __init__(self, max_connections: int = 20):
        self.max_connections = max_connections
        self.pools: dict[str, httpx.AsyncClient] = {}
    
    @asynccontextmanager
    async def get_client(self, server_url: str):
        if server_url not in self.pools:
            self.pools[server_url] = httpx.AsyncClient(
                base_url=server_url,
                limits=httpx.Limits(
                    max_connections=self.max_connections,
                    max_keepalive_connections=10,
                ),
                timeout=httpx.Timeout(30.0),
            )
        
        yield self.pools[server_url]
    
    async def close(self):
        for client in self.pools.values():
            await client.aclose()
```

### 5.2 Response Caching

```python
# Cache frequent MCP responses
import hashlib
from functools import lru_cache

class MCPResponseCache:
    def __init__(self, max_size: int = 1000, ttl: int = 300):
        self.cache = {}
        self.ttl = ttl
        self.max_size = max_size
    
    def _make_key(self, method: str, params: dict) -> str:
        content = f"{method}:{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def get_or_fetch(self, method: str, params: dict, fetch_fn):
        key = self._make_key(method, params)
        
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                return entry["result"]
        
        result = await fetch_fn()
        
        # Evict oldest if full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache, key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]
        
        self.cache[key] = {"result": result, "timestamp": time.time()}
        return result
```

### 5.3 Batch Tool Calls

```python
# Batch multiple tool calls into single MCP request
async def batch_tool_calls(
    client: MCPClient,
    calls: list[dict],
) -> list[dict]:
    """Execute multiple tool calls in parallel."""
    tasks = [
        client.call_tool(call["name"], call["arguments"])
        for call in calls
    ]
    return await asyncio.gather(*tasks)
```

---

## 6. Disaster Recovery & High Availability

### 6.1 High Availability Architecture

```
┌─────────────────────────────────────────────────────┐
│                Region A (Primary)                     │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ MCP GW   │  │ MCP GW   │  │ MCP GW   │          │
│  │ Replica 1│  │ Replica 2│  │ Replica 3│          │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘          │
│       └──────────────┼──────────────┘                │
│                      │                                │
│  ┌───────────────────▼───────────────────┐          │
│  │        MCP Server Fleet (Primary)      │          │
│  └───────────────────┬───────────────────┘          │
│                      │                                │
│  ┌───────────────────▼───────────────────┐          │
│  │     Database (Primary + Read Replica)  │          │
│  └───────────────────────────────────────┘          │
└─────────────────────────────────────────────────────┘
                      │
                      │ Cross-Region Replication
                      │
┌─────────────────────────────────────────────────────┐
│                Region B (DR / Failover)              │
│                                                      │
│  ┌──────────┐  ┌──────────┐                          │
│  │ MCP GW   │  │ MCP GW   │                          │
│  │ Replica 1│  │ Replica 2│                          │
│  └────┬─────┘  └────┬─────┘                          │
│       └──────────────┘                                │
│                      │                                │
│  ┌───────────────────▼───────────────────┐          │
│  │        MCP Server Fleet (Standby)      │          │
│  └───────────────────┬───────────────────┘          │
│                      │                                │
│  ┌───────────────────▼───────────────────┐          │
│  │     Database (Read Replica → Promote)  │          │
│  └───────────────────────────────────────┘          │
└─────────────────────────────────────────────────────┘
```

### 6.2 Failover Script

```python
# failover.py — Automated failover for MCP infrastructure
import asyncio
import httpx

class MCPFailoverManager:
    def __init__(self, primary_url: str, dr_url: str):
        self.primary_url = primary_url
        self.dr_url = dr_url
        self.current_active = "primary"
    
    async def health_check(self, url: str) -> bool:
        """Check if MCP gateway is healthy."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{url}/health")
                return resp.status_code == 200
        except Exception:
            return False
    
    async def monitor_and_failover(self):
        """Continuously monitor and failover if needed."""
        while True:
            primary_healthy = await self.health_check(self.primary_url)
            
            if not primary_healthy and self.current_active == "primary":
                # Attempt failover
                dr_healthy = await self.health_check(self.dr_url)
                
                if dr_healthy:
                    await self._execute_failover()
                    self.current_active = "dr"
                else:
                    # Both unhealthy — alert!
                    await self._alert("CRITICAL: Both primary and DR MCP gateways down")
            
            elif primary_healthy and self.current_active == "dr":
                # Primary recovered — failback
                await self._execute_failback()
                self.current_active = "primary"
            
            await asyncio.sleep(10)  # Check every 10 seconds
    
    async def _execute_failover(self):
        """Execute failover to DR region."""
        # Update DNS, load balancer, or service mesh
        print(f"FAILOVER: Switching to DR at {self.dr_url}")
        # In production: update Route53, Consul, Istio, etc.
    
    async def _execute_failback(self):
        """Execute failback to primary region."""
        print(f"FAILBACK: Switching to primary at {self.primary_url}")
    
    async def _alert(self, message: str):
        """Send alert."""
        print(f"ALERT: {message}")
```

---

## 7. Migration: Local MCP to Cloud MCP

### 7.1 Migration Checklist

```
□ Audit current local MCP servers
□ Inventory all tools, resources, and prompts
□ Assess data sensitivity levels
□ Choose cloud MCP platform
□ Set up auth (OAuth 2.1 / API keys)
□ Deploy first MCP server (non-critical)
□ Run parallel (local + cloud) for 1 week
□ Migrate agent configurations
□ Monitor for 2 weeks
□ Decommission local servers
□ Update documentation
```

### 7.2 Migration Script

```python
# migrate_local_to_cloud.py
import json
import asyncio
from mcp_client import MCPClient
from manufact_sdk import MCPServer, DeployConfig

async def migrate_server(
    local_config: dict,
    cloud_platform: str,
    deploy_config: DeployConfig,
):
    """Migrate a local MCP server to cloud."""
    
    # 1. Connect to local server
    local_client = await MCPClient.connect_stdio(
        command=local_config["command"],
        args=local_config["args"],
    )
    
    # 2. Discover capabilities
    tools = await local_client.list_tools()
    resources = await local_client.list_resources()
    prompts = await local_client.list_prompts()
    
    print(f"Discovered: {len(tools)} tools, {len(resources)} resources, {len(prompts)} prompts")
    
    # 3. Test against cloud
    cloud_server = MCPServer.from_stdio_config(local_config)
    cloud_url = cloud_server.deploy(deploy_config)
    
    # 4. Validate parity
    cloud_client = await MCPClient.connect(cloud_url, auth_token=deploy_config.auth_token)
    cloud_tools = await cloud_client.list_tools()
    
    assert len(tools) == len(cloud_tools), "Tool count mismatch!"
    
    # 5. Run smoke tests
    for tool in tools:
        test_result = await cloud_client.call_tool(tool.name, tool.test_args)
        assert test_result.get("isError") == False, f"Smoke test failed for {tool.name}"
        print(f"  ✓ {tool.name} — smoke test passed")
    
    print(f"Migration complete: {cloud_url}")
    return cloud_url

# Usage
async def main():
    local_config = {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/data"],
    }
    
    deploy_config = DeployConfig(
        replicas=3,
        auth="oauth2.1",
        rate_limit="500/min",
        data_residency="us-east-1",
    )
    
    cloud_url = await migrate_server(local_config, "manufact", deploy_config)

asyncio.run(main())
```

### 7.3 Configuration Diff Tool

```python
# diff_configs.py — Compare local vs cloud MCP configurations
def diff_mcp_configs(local: dict, cloud: dict) -> list[str]:
    """Compare local and cloud MCP configs, report differences."""
    differences = []
    
    # Compare tools
    local_tools = {t["name"] for t in local.get("tools", [])}
    cloud_tools = {t["name"] for t in cloud.get("tools", [])}
    
    missing_in_cloud = local_tools - cloud_tools
    extra_in_cloud = cloud_tools - local_tools
    
    if missing_in_cloud:
        differences.append(f"Missing in cloud: {missing_in_cloud}")
    if extra_in_cloud:
        differences.append(f"Extra in cloud: {extra_in_cloud}")
    
    # Compare auth
    if local.get("auth") != cloud.get("auth"):
        differences.append(f"Auth config differs: {local.get('auth')} vs {cloud.get('auth')}")
    
    # Compare rate limits
    if local.get("rate_limit") != cloud.get("rate_limit"):
        differences.append(f"Rate limits differ: {local.get('rate_limit')} vs {cloud.get('rate_limit')}")
    
    return differences
```

---

## 8. Cross-References to Existing Library Docs

| Topic | Library Doc | Relevance |
|-------|-------------|-----------|
| MCP protocol | [03-Agents/04-Protocols-MCP-ACP.md](../03-Agents/04-Protocols-MCP-ACP.md) | Protocol foundation |
| Agent frameworks | [03-Agents/03-Agentic-Frameworks.md](../03-Agents/03-Agentic-Frameworks.md) | Frameworks use MCP |
| Tool implementations | [03-Agents/05-Tool-Implementations.md](../03-Agents/05-Tool-Implementations.md) | Tool design patterns |
| Agent security | [18-Agent-Security-and-Trust/](../18-Agent-Security-and-Trust/) | Security patterns |
| Agent memory | [32-Agent-Memory-Systems/](../32-Agent-Memory-Systems/) | Memory via MCP |
| Workflow orchestration | [31-AI-Workflow-Orchestration-and-Durable-Execution/](../31-AI-Workflow-Orchestration-and-Durable-Execution/) | Orchestration patterns |
| Enterprise deployment | [05-Enterprise/](../05-Enterprise/) | Enterprise context |
| Cost optimization | [41-AI-Cost-Optimization-and-Enterprise-ROI/](../41-AI-Cost-Optimization-and-Enterprise-ROI/) | Cost management |
| Browser automation | [46-Agentic-Browser-Automation-Computer-Use/](../46-Agentic-Browser-Automation-Computer-Use/) | Browser MCP tools |
| Local AI inference | [23-Local-AI-Inference-Self-Hosting/](../23-Local-AI-Inference-Self-Hosting/) | Local vs cloud comparison |

---

*This document is part of the AI Knowledge Library auto-enrichment system.*

---
**See also:**
- [07 — AI Coding Agent Sustainability: Managing Hidden Costs, Code Quality Decay, and Developer Burnout](33-AI-Native-Software-Development/07-AI-Coding-Agent-Sustainability.md)
- [03 — Agent-Native Orchestration: LangGraph, Conductor, Mistral Workflows, Mcp-Agent](31-AI-Workflow-Orchestration-and-Durable-Execution/03-Agent-Native-Orchestration.md)
- [AI Agent Commerce & Agent-to-Agent (A2A) Payments: The 2026 Frontier](28-AI-Agent-Commerce-and-A2A-Payments/01-Overview.md)
