# Tools and Frameworks for AI Agent Financial Governance

> A comprehensive guide to the tools, frameworks, platforms, and libraries available for implementing agent cost control. This document covers open-source solutions, commercial platforms, cloud-native tools, and custom-built components for managing AI agent costs.

---

## Table of Contents

1. [Open-Source Cost Control Libraries](#1-open-source-cost-control-libraries)
2. [Cloud-Native Cost Management Tools](#2-cloud-native-cost-management-tools)
3. [Commercial Agent Cost Platforms](#3-commercial-agent-cost-platforms)
4. [Observability and Monitoring Tools](#4-observability-and-monitoring-tools)
5. [Budget Management Frameworks](#5-budget-management-frameworks)
6. [Model Routing and Optimization](#6-model-routing-and-optimization)
7. [Custom Tool Development](#7-custom-tool-development)
8. [Tool Selection Guide](#8-tool-selection-guide)
9. [Integration Patterns](#9-integration-patterns)
10. [Cost Optimization Tools](#10-cost-optimization-tools)

---

## 1. Open-Source Cost Control Libraries

### 1.1 LiteLLM (Unified LLM Proxy)

**GitHub:** github.com/BerriAI/litellm
**License:** MIT
**Stars:** 15,000+

**Overview:** LiteLLM provides a unified interface to 100+ LLM providers with built-in cost tracking, budget management, and rate limiting.

**Key Features:**
- Unified API for 100+ LLM providers
- Real-time cost tracking per request
- Budget management with per-user/team limits
- Rate limiting and retry logic
- Model fallback and routing
- Cost analytics dashboard

**Installation:**

```bash
pip install litellm
```

**Usage Example:**

```python
from litellm import completion
from litellm import BudgetManager

# Initialize budget manager
budget_manager = BudgetManager(
    project_name="agent-platform",
    max_budget=1000.00,  # $1000 monthly
    max_budget_per_user=50.00  # $50 per user
)

# Track costs
response = completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
    metadata={
        "user_id": "user-123",
        "team": "engineering",
        "project": "code-assistant"
    }
)

# Get cost breakdown
cost = response._hidden_params["cost"]
print(f"Request cost: ${cost:.6f}")

# Check budget
budget_status = budget_manager.get_budget("user-123")
print(f"Remaining budget: ${budget_status['remaining']:.2f}")
```

**Pros:**
- Easy to integrate
- Wide provider support
- Built-in budget management
- Active community

**Cons:**
- Self-hosted for advanced features
- Limited enterprise features
- Requires Redis for distributed deployments

### 1.2 LangSmith (LangChain Observability)

**Website:** smith.langchain.com
**License:** Proprietary (free tier available)

**Overview:** LangSmith provides observability, testing, and monitoring for LLM applications with cost tracking capabilities.

**Key Features:**
- Cost tracking per trace
- Token usage monitoring
- Budget alerts
- Performance analytics
- A/B testing for cost optimization

**Usage Example:**

```python
from langsmith import Client
from langsmith.run_trees import RunTree

# Initialize client
client = Client()

# Create traced run with cost tracking
rt = RunTree(
    name="agent-inference",
    run_type="chain",
    tags=["cost-tracking"]
)

# Log cost metadata
rt.add_metadata({
    "model": "gpt-4o",
    "estimated_cost": 0.045,
    "budget_category": "inference"
})

# Get cost analytics
runs = client.list_runs(
    project_name="agent-platform",
    filter="metadata['model'] = 'gpt-4o'"
)

total_cost = sum(r.extra.get("cost", 0) for r in runs)
print(f"Total cost: ${total_cost:.2f}")
```

### 1.3 PromptLayer (Cost Analytics)

**Website:** promptlayer.com
**License:** Proprietary (free tier available)

**Overview:** PromptLayer provides prompt management and cost analytics for LLM applications.

**Key Features:**
- Prompt versioning
- Cost tracking per prompt
- Usage analytics
- A/B testing
- Budget alerts

**Usage Example:**

```python
import promptlayer

# Track API calls with cost
promptlayer.api_key = "your-api-key"

# Log request with cost metadata
promptlayer.track_cost(
    model="gpt-4o",
    input_tokens=500,
    output_tokens=200,
    metadata={
        "user_id": "user-123",
        "prompt_template": "code-review"
    }
)

# Get cost analytics
analytics = promptlayer.get_cost_analytics(
    date_range="last_30_days",
    group_by="model"
)
```

### 1.4 AgentOps (Agent Observability)

**Website:** agentops.ai
**License:** Apache 2.0

**Overview:** AgentOps provides observability specifically for AI agents with cost tracking and session monitoring.

**Key Features:**
- Agent session tracking
- Cost per session
- Tool call monitoring
- Error tracking
- Performance analytics

**Usage Example:**

```python
import agentops

# Initialize
agentops.init(api_key="your-api-key")

# Start session
session = agentops.start_session()

# Track agent actions
agentops.track_action(
    action_type="inference",
    model="gpt-4o",
    tokens={"input": 1000, "output": 500},
    cost=0.0075
)

agentops.track_action(
    action_type="tool_call",
    tool_name="web_search",
    cost=0.02
)

# End session and get summary
summary = session.end()
print(f"Session cost: ${summary['total_cost']:.4f}")
```

---

## 2. Cloud-Native Cost Management Tools

### 2.1 AWS Cost Management for Bedrock

**Overview:** AWS provides native cost management tools for Amazon Bedrock.

**Key Features:**
- Cost Explorer integration
- Budgets and alerts
- Cost allocation tags
- Usage reports

**Setup:**

```python
import boto3

# Create budget for Bedrock usage
budgets_client = boto3.client('budgets')

budgets_client.create_budget(
    AccountId='123456789012',
    Budget={
        'BudgetName': 'bedrock-agent-costs',
        'BudgetLimit': {
            'Amount': '1000',
            'Unit': 'USD'
        },
        'TimeUnit': 'MONTHLY',
        'BudgetType': 'COST',
        'CostFilters': {
            'Service': ['Amazon Bedrock']
        }
    },
    NotificationsWithSubscribers=[
        {
            'Notification': {
                'NotificationType': 'ACTUAL',
                'ComparisonOperator': 'GREATER_THAN',
                'Threshold': 80
            },
            'Subscribers': [
                {
                    'SubscriptionType': 'EMAIL',
                    'Address': 'finance@company.com'
                }
            ]
        }
    ]
)
```

### 2.2 Azure Cost Management

**Overview:** Azure provides Cost Management + Billing for monitoring and optimizing Azure OpenAI costs.

**Key Features:**
- Cost analysis
- Budget management
- Alert rules
- Cost allocation

**Setup:**

```python
from azure.mgmt.costmanagement import CostManagementClient
from azure.identity import DefaultAzureCredential

# Initialize client
credential = DefaultAzureCredential()
client = CostManagementClient(credential, "subscription-id")

# Query costs
query_result = client.query.run(
    scope="/subscriptions/subscription-id",
    parameters={
        "timeframe": "MonthToDate",
        "dataset": {
            "granularity": "Daily",
            "filter": {
                "and": [
                    {
                        "dimensions": {
                            "name": "ServiceName",
                            "operator": "In",
                            "values": ["Azure OpenAI Service"]
                        }
                    }
                ]
            }
        }
    }
)

# Process results
for row in query_result.rows:
    print(f"Date: {row[1]}, Cost: ${row[0]:.2f}")
```

### 2.3 Google Cloud Cost Management

**Overview:** Google Cloud provides Cost Management tools for Vertex AI.

**Key Features:**
- Cost reports
- Budget alerts
- Recommendations
- Exports

**Setup:**

```python
from google.cloud import billing_v1

# Initialize client
client = billing_v1.CloudBillingClient()

# Create budget
budget = billing_v1.Budget(
    display_name="vertex-ai-budget",
    budget_amount=billing_v1.BudgetAmount(
        specified_amount=billing_v1.CurrencyAmount(
            units=1000  # $1000
        )
    ),
    threshold_rules=[
        billing_v1.ThresholdRule(
            spend_basis=billing_v1.ThresholdRule.SpendBasis.CURRENT_SPEND,
            threshold_percent=0.8
        )
    ],
    all_updates_rule=billing_v1.AllUpdatesRule(
        pubsub_rule=billing_v1.PubsubRule(
            topic="projects/project-id/topics/budget-alerts"
        )
    )
)

# Create budget
parent = "billingAccounts/billing-account-id"
client.create_budget(parent=parent, budget=budget)
```

---

## 3. Commercial Agent Cost Platforms

### 3.1 Portkey.ai

**Website:** portkey.ai
**Pricing:** Free tier available, paid plans from $49/month

**Overview:** Portkey provides AI gateway with cost management, model routing, and observability.

**Key Features:**
- Unified API gateway
- Cost tracking and analytics
- Model routing
- Fallback and retry
- Budget management

**Usage Example:**

```python
from portkey import PORTKEY_GATEWAY_URL, createHeaders

# Configure gateway
headers = createHeaders(
    apiKey="your-api-key",
    provider="openai"
)

# Make request through gateway
import requests

response = requests.post(
    "https://api.portkey.ai/v1/chat/completions",
    headers=headers,
    json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello"}],
        "metadata": {
            "user_id": "user-123",
            "team": "engineering"
        }
    }
)

# Cost tracked automatically
```

### 3.2 Helicone

**Website:** helicone.ai
**Pricing:** Free tier available, paid plans from $20/month

**Overview:** Helicone provides LLM observability with cost tracking and analytics.

**Key Features:**
- Request logging
- Cost analytics
- User tracking
- Custom properties
- Rate limiting

**Usage Example:**

```python
import openai

# Configure with Helicone
openai.api_base = "https://oai.helicone.ai/v1"
openai.api_key = "your-openai-key"

# Add Helicone headers
response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
    headers={
        "Helicone-Auth": "your-helicone-key",
        "Helicone-User-Id": "user-123",
        "Helicone-Property-Team": "engineering"
    }
)

# Cost tracked automatically in Helicone dashboard
```

### 3.3 Langfuse

**Website:** langfuse.com
**Pricing:** Open-source (self-hosted) or cloud from $59/month

**Overview:** Langfuse provides LLM observability with cost tracking, prompt management, and analytics.

**Key Features:**
- Trace logging
- Cost tracking
- Prompt management
- Evaluation
- Self-hosted option

**Usage Example:**

```python
from langfuse import Langfuse
from langfuse.decorators import observe

# Initialize
langfuse = Langfuse(
    public_key="your-public-key",
    secret_key="your-secret-key",
    host="https://cloud.langfuse.com"
)

# Decorate functions for tracing
@observe()
def agent_inference(messages, model="gpt-4o"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages
    )
    return response

# Usage with cost tracking
with langfuse.trace(name="agent-session", user_id="user-123") as trace:
    response = agent_inference(
        messages=[{"role": "user", "content": "Hello"}]
    )
    
    # Cost tracked automatically
    trace.score(name="cost", value=trace.get_cost())
```

---

## 4. Observability and Monitoring Tools

### 4.1 Datadog LLM Observability

**Overview:** Datadog provides LLM Observability as part of its APM platform.

**Key Features:**
- LLM trace monitoring
- Cost tracking
- Token usage
- Performance metrics
- Alerting

**Usage Example:**

```python
from ddtrace import tracer

# Configure tracer
tracer.configure(
    hostname="datadog-agent",
    port=8126
)

# Trace LLM calls
with tracer.trace("llm.inference", service="agent-platform") as span:
    span.set_tag("llm.model", "gpt-4o")
    span.set_tag("llm.input_tokens", 1000)
    span.set_tag("llm.output_tokens", 500)
    span.set_tag("llm.cost", 0.0075)
    
    # Make LLM call
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}]
    )
```

### 4.2 New Relic AI Monitoring

**Overview:** New Relic provides AI monitoring with cost tracking capabilities.

**Key Features:**
- LLM monitoring
- Cost analytics
- Performance tracking
- Error monitoring

### 4.3 Custom Prometheus + Grafana

**Overview:** Build custom monitoring with Prometheus metrics and Grafana dashboards.

**Usage Example:**

```python
from prometheus_client import Counter, Histogram, Gauge
from prometheus_client import start_http_server

# Define metrics
llm_cost = Counter(
    'llm_cost_total',
    'Total LLM cost',
    ['model', 'team', 'project']
)

llm_tokens = Counter(
    'llm_tokens_total',
    'Total LLM tokens',
    ['model', 'type']  # type: input/output
)

agent_session_cost = Gauge(
    'agent_session_cost',
    'Current session cost',
    ['agent_id', 'session_id']
)

# Track costs
def track_inference_cost(model, input_tokens, output_tokens, cost, team, project):
    llm_cost.labels(model=model, team=team, project=project).inc(cost)
    llm_tokens.labels(model=model, type='input').inc(input_tokens)
    llm_tokens.labels(model=model, type='output').inc(output_tokens)

# Start metrics server
start_http_server(8000)
```

**Grafana Dashboard Query Examples:**

```sql
-- Total cost by model
sum(rate(llm_cost_total[5m])) by (model)

-- Cost by team
sum(rate(llm_cost_total[5m])) by (team)

-- Token usage trend
rate(llm_tokens_total[5m])

-- Cost per request
rate(llm_cost_total[5m]) / rate(http_requests_total[5m])
```

---

## 5. Budget Management Frameworks

### 5.1 Custom Budget Framework

```python
from dataclasses import dataclass
from typing import Dict, Optional
from datetime import datetime, timedelta
from enum import Enum

class BudgetPeriod(Enum):
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

@dataclass
class BudgetConfig:
    period: BudgetPeriod
    limit: float
    alert_thresholds: list[float]  # e.g., [0.5, 0.75, 0.9]
    action_on_exceed: str  # "alert", "degrade", "stop"

class BudgetFramework:
    """Custom budget management framework."""
    
    def __init__(self):
        self.budgets: Dict[str, Dict[str, BudgetConfig]] = {}
        self.usage: Dict[str, Dict[str, float]] = {}
    
    def create_budget(
        self,
        entity_type: str,
        entity_id: str,
        config: BudgetConfig
    ):
        """Create a budget for an entity."""
        key = f"{entity_type}:{entity_id}"
        self.budgets[key] = config
        self.usage[key] = {period.value: 0.0 for period in BudgetPeriod}
    
    def record_usage(
        self,
        entity_type: str,
        entity_id: str,
        amount: float
    ):
        """Record usage against a budget."""
        key = f"{entity_type}:{entity_id}"
        
        if key not in self.usage:
            self.usage[key] = {period.value: 0.0 for period in BudgetPeriod}
        
        for period in BudgetPeriod:
            self.usage[key][period.value] += amount
        
        # Check thresholds
        self._check_thresholds(key)
    
    def get_remaining(
        self,
        entity_type: str,
        entity_id: str,
        period: BudgetPeriod
    ) -> float:
        """Get remaining budget."""
        key = f"{entity_type}:{entity_id}"
        
        if key not in self.budgets:
            return 0.0
        
        budget = self.budgets[key]
        used = self.usage.get(key, {}).get(period.value, 0.0)
        
        return max(0.0, budget.limit - used)
    
    def _check_thresholds(self, key: str):
        """Check if any alert thresholds are crossed."""
        budget = self.budgets.get(key)
        if not budget:
            return
        
        used = self.usage.get(key, {}).get(budget.period.value, 0.0)
        utilization = used / budget.limit if budget.limit > 0 else 0
        
        for threshold in budget.alert_thresholds:
            if utilization >= threshold:
                self._send_alert(key, threshold, utilization)
    
    def _send_alert(self, key: str, threshold: float, utilization: float):
        """Send alert when threshold crossed."""
        print(f"ALERT: {key} at {utilization:.1%} (threshold: {threshold:.1%})")
        # Implement actual alerting logic
```

### 5.2 Redis-Based Distributed Budgets

```python
import redis
import json
from typing import Optional

class RedisBudgetManager:
    """Distributed budget manager using Redis."""
    
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
        self.key_prefix = "budget:"
    
    def set_budget(
        self,
        entity_key: str,
        limit: float,
        period: str = "monthly"
    ):
        """Set budget limit."""
        key = f"{self.key_prefix}{entity_key}:{period}"
        self.redis.set(key, limit)
    
    def record_usage(
        self,
        entity_key: str,
        amount: float,
        period: str = "monthly"
    ) -> dict:
        """Record usage and check budget."""
        key = f"{self.key_prefix}{entity_key}:{period}"
        usage_key = f"{key}:usage"
        
        # Atomic increment
        new_usage = self.redis.incrbyfloat(usage_key, amount)
        
        # Get limit
        limit = float(self.redis.get(key) or 0)
        
        # Check thresholds
        utilization = new_usage / limit if limit > 0 else 0
        
        return {
            "usage": new_usage,
            "limit": limit,
            "remaining": max(0, limit - new_usage),
            "utilization": utilization,
            "exceeded": new_usage > limit
        }
    
    def get_usage(
        self,
        entity_key: str,
        period: str = "monthly"
    ) -> dict:
        """Get current usage."""
        key = f"{self.key_prefix}{entity_key}:{period}"
        usage_key = f"{key}:usage"
        
        usage = float(self.redis.get(usage_key) or 0)
        limit = float(self.redis.get(key) or 0)
        
        return {
            "usage": usage,
            "limit": limit,
            "remaining": max(0, limit - usage),
            "utilization": usage / limit if limit > 0 else 0
        }
```

---

## 6. Model Routing and Optimization

### 6.1 Martian Model Router

**Website:** withmartian.com
**Overview:** Martian provides intelligent model routing for cost optimization.

**Key Features:**
- Automatic model selection
- Cost optimization
- Quality-based routing
- Fallback handling

### 6.2 RouteLLM

**Website:** github.com/lm-sys/RouteLLM
**License:** MIT

**Overview:** RouteLLM provides cost-effective routing between LLMs.

**Key Features:**
- Router training
- Cost optimization
- Quality preservation
- Simple integration

**Usage Example:**

```python
from routellm import Controller

# Initialize controller
controller = Controller(
    routers=["mf"],  # matrix factorization router
    api_key="your-api-key"
)

# Route request
response = controller.chat.completions.create(
    model="router-mf-0.5",  # 0.5 = cost threshold
    messages=[{"role": "user", "content": "Hello"}]
)

# Automatically routes to cheaper model when possible
```

---

## 7. Custom Tool Development

### 7.1 Cost Control Middleware Template

```python
from typing import Callable, Any
from functools import wraps

def cost_controlled(
    max_cost: float = 1.0,
    budget_key: str = "default",
    on_exceed: str = "raise"
):
    """Decorator for cost-controlled function calls."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Estimate cost
            estimated_cost = estimate_function_cost(func, args, kwargs)
            
            # Check budget
            budget_manager = get_budget_manager()
            if not budget_manager.can_spend(budget_key, estimated_cost):
                if on_exceed == "raise":
                    raise BudgetExceededException(
                        f"Budget exceeded for {budget_key}"
                    )
                elif on_exceed == "degrade":
                    return degraded_response(func, args, kwargs)
                elif on_exceed == "return_none":
                    return None
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Record actual cost
            actual_cost = calculate_actual_cost(result)
            budget_manager.record_spend(budget_key, actual_cost)
            
            return result
        
        return wrapper
    return decorator

# Usage
@cost_controlled(max_cost=0.50, budget_key="code-review")
def review_code(code: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"Review: {code}"}]
    )
    return response.choices[0].message.content
```

### 7.2 Cost Tracking Wrapper

```python
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional
import time

@dataclass
class CostSession:
    session_id: str
    start_time: float
    total_cost: float = 0.0
    cost_by_model: dict = None
    cost_by_action: dict = None
    
    def __post_init__(self):
        if self.cost_by_model is None:
            self.cost_by_model = {}
        if self.cost_by_action is None:
            self.cost_by_action = {}

@contextmanager
def track_costs(session_id: str):
    """Context manager for tracking costs."""
    session = CostSession(
        session_id=session_id,
        start_time=time.time()
    )
    
    try:
        yield session
    finally:
        # Calculate total time
        duration = time.time() - session.start_time
        
        # Log session summary
        print(f"Session {session_id}:")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Total cost: ${session.total_cost:.4f}")
        print(f"  Cost by model: {session.cost_by_model}")
        print(f"  Cost by action: {session.cost_by_action}")
        
        # Send to monitoring
        send_to_monitoring(session)

def record_cost(
    session: CostSession,
    model: str,
    action: str,
    cost: float
):
    """Record cost in a session."""
    session.total_cost += cost
    session.cost_by_model[model] = session.cost_by_model.get(model, 0) + cost
    session.cost_by_action[action] = session.cost_by_action.get(action, 0) + cost

# Usage
with track_costs("session-123") as session:
    # Make LLM calls
    response1 = call_llm("gpt-4o", "Hello")
    record_cost(session, "gpt-4o", "inference", 0.0075)
    
    # Call tools
    result = call_tool("web_search", "AI trends")
    record_cost(session, "tool", "web_search", 0.02)
```

---

## 8. Tool Selection Guide

### 8.1 Decision Matrix

| Use Case | Recommended Tool | Why |
|---|---|---|
| **Simple cost tracking** | LiteLLM | Easy setup, open-source |
| **Enterprise observability** | Datadog / New Relic | Full APM integration |
| **Budget management** | Custom + Redis | Flexible, scalable |
| **Model routing** | RouteLLM / Martian | Cost optimization |
| **Agent-specific monitoring** | AgentOps | Purpose-built for agents |
| **Self-hosted** | Langfuse | Open-source, customizable |
| **Cloud-native (AWS)** | AWS Cost Explorer + Budgets | Native integration |
| **Cloud-native (Azure)** | Azure Cost Management | Native integration |
| **Cloud-native (GCP)** | GCP Cost Management | Native integration |

### 8.2 Feature Comparison

| Feature | LiteLLM | Helicone | Langfuse | AgentOps | Portkey |
|---|---|---|---|---|---|
| Cost Tracking | ✅ | ✅ | ✅ | ✅ | ✅ |
| Budget Management | ✅ | ❌ | ❌ | ❌ | ✅ |
| Model Routing | ✅ | ❌ | ❌ | ❌ | ✅ |
| Self-Hosted | ✅ | ❌ | ✅ | ❌ | ❌ |
| Agent-Specific | ❌ | ❌ | ❌ | ✅ | ❌ |
| Open Source | ✅ | ❌ | ✅ | ✅ | ❌ |
| Free Tier | ✅ | ✅ | ✅ | ✅ | ✅ |

### 8.3 Cost Comparison

| Tool | Free Tier | Paid Plans | Enterprise |
|---|---|---|---|
| LiteLLM | Unlimited (self-hosted) | N/A | Support plans |
| Helicone | 100k requests/month | $20-100/month | Custom |
| Langfuse | Unlimited (self-hosted) | $59-499/month | Custom |
| AgentOps | 1k sessions/month | $49-199/month | Custom |
| Portkey | 10k requests/month | $49-299/month | Custom |

---

## 9. Integration Patterns

### 9.1 Gateway Pattern

```
┌──────────────┐
│   Agent      │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Gateway    │ ← Cost tracking, routing, budgets
└──────┬───────┘
       │
       ├──────────────────┬──────────────────┐
       ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Model A     │  │  Model B     │  │  Model C     │
└──────────────┘  └──────────────┘  └──────────────┘
```

**Implementation:**

```python
class CostGateway:
    """Gateway pattern for cost control."""
    
    def __init__(self):
        self.interceptor = CostInterceptorMiddleware(...)
        self.router = CostAwareModelRouter(...)
    
    async def forward_request(self, request: dict) -> dict:
        """Forward request through gateway."""
        # 1. Check budget
        budget_check = await self.interceptor.check_budget(
            request["agent_id"],
            request["estimated_cost"]
        )
        
        if not budget_check.allowed:
            return self._handle_budget_exceeded(budget_check)
        
        # 2. Route to model
        model_selection = await self.router.route(
            agent_id=request["agent_id"],
            messages=request["messages"],
            budget_remaining=budget_check.remaining_budget
        )
        
        # 3. Execute request
        response = await self._execute_with_model(
            model_selection["model"],
            request["messages"]
        )
        
        # 4. Record actual cost
        await self.interceptor.record_cost(
            request["agent_id"],
            response["actual_cost"]
        )
        
        return response
```

### 9.2 Middleware Pattern

```python
class CostMiddleware:
    """Middleware pattern for cost control."""
    
    def __init__(self, app):
        self.app = app
        self.interceptor = CostInterceptorMiddleware(...)
    
    async def __call__(self, scope, receive, send):
        """Process request through middleware."""
        if scope["type"] == "http":
            # Extract cost metadata from request
            cost_metadata = self._extract_cost_metadata(scope)
            
            # Check budget
            budget_check = await self.interceptor.check_budget(
                cost_metadata["agent_id"],
                cost_metadata["estimated_cost"]
            )
            
            if not budget_check.allowed:
                # Return budget exceeded response
                response = {
                    "status": 429,
                    "body": {"error": "Budget exceeded"}
                }
                await send({"type": "http.response.start", "status": 429})
                await send({"type": "http.response.body", "body": json.dumps(response).encode()})
                return
            
            # Process request
            await self.app(scope, receive, send)
            
            # Record cost
            await self.interceptor.record_cost(
                cost_metadata["agent_id"],
                cost_metadata["estimated_cost"]
            )
```

### 9.3 Sidecar Pattern

```python
class CostSidecar:
    """Sidecar pattern for cost control."""
    
    def __init__(self, agent_port: int, sidecar_port: int):
        self.agent_port = agent_port
        self.sidecar_port = sidecar_port
        self.interceptor = CostInterceptorMiddleware(...)
    
    async def start(self):
        """Start sidecar alongside agent."""
        # Start sidecar server
        server = await self._start_server()
        
        # Start agent with sidecar proxy
        agent = await self._start_agent(
            proxy=f"http://localhost:{self.sidecar_port}"
        )
        
        return server, agent
    
    async def _start_server(self):
        """Start sidecar HTTP server."""
        from aiohttp import web
        
        app = web.Application()
        app.router.add_post("/proxy", self.handle_proxy)
        
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "localhost", self.sidecar_port)
        await site.start()
    
    async def handle_proxy(self, request):
        """Handle proxied request."""
        body = await request.json()
        
        # Check budget
        budget_check = await self.interceptor.check_budget(
            body["agent_id"],
            body["estimated_cost"]
        )
        
        if not budget_check.allowed:
            return web.json_response(
                {"error": "Budget exceeded"},
                status=429
            )
        
        # Forward to actual API
        async with aiohttp.ClientSession() as session:
            async with session.post(
                body["target_url"],
                json=body["payload"]
            ) as resp:
                response = await resp.json()
        
        # Record cost
        await self.interceptor.record_cost(
            body["agent_id"],
            response.get("cost", 0)
        )
        
        return web.json_response(response)
```

---

## 10. Cost Optimization Tools

### 10.1 Prompt Optimization

```python
class PromptOptimizer:
    """Optimizes prompts to reduce token usage."""
    
    def optimize(self, prompt: str, target_tokens: int) -> str:
        """Optimize prompt to fit within token target."""
        current_tokens = self._count_tokens(prompt)
        
        if current_tokens <= target_tokens:
            return prompt
        
        # Apply optimization techniques
        optimized = prompt
        
        # 1. Remove redundant phrases
        optimized = self._remove_redundancy(optimized)
        
        # 2. Compress examples
        optimized = self._compress_examples(optimized)
        
        # 3. Use shorter alternatives
        optimized = self._simplify_language(optimized)
        
        # 4. Remove filler words
        optimized = self._remove_filler(optimized)
        
        return optimized
    
    def _remove_redundancy(self, text: str) -> str:
        """Remove redundant phrases."""
        redundant_patterns = [
            (r"please\s+", ""),
            (r"kindly\s+", ""),
            (r"if you don't mind,\s+", ""),
            (r"could you possibly\s+", "please "),
        ]
        
        for pattern, replacement in redundant_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
```

### 10.2 Response Caching

```python
from hashlib import sha256
import redis

class ResponseCache:
    """Cache LLM responses to reduce costs."""
    
    def __init__(self, redis_url: str, ttl: int = 3600):
        self.redis = redis.from_url(redis_url)
        self.ttl = ttl
    
    def get_cache_key(self, messages: list, model: str) -> str:
        """Generate cache key from messages and model."""
        content = json.dumps({"messages": messages, "model": model})
        return f"llm:cache:{sha256(content.encode()).hexdigest()}"
    
    def get(self, messages: list, model: str) -> Optional[dict]:
        """Get cached response."""
        key = self.get_cache_key(messages, model)
        cached = self.redis.get(key)
        
        if cached:
            return json.loads(cached)
        
        return None
    
    def set(self, messages: list, model: str, response: dict):
        """Cache response."""
        key = self.get_cache_key(messages, model)
        self.redis.setex(key, self.ttl, json.dumps(response))
    
    def get_or_execute(self, messages: list, model: str, executor) -> dict:
        """Get from cache or execute and cache."""
        cached = self.get(messages, model)
        
        if cached:
            cached["cached"] = True
            cached["cost"] = 0.0  # Cached responses are free
            return cached
        
        response = executor(messages, model)
        self.set(messages, model, response)
        
        response["cached"] = False
        return response
```

### 10.3 Batch Processing Optimizer

```python
class BatchOptimizer:
    """Optimizes LLM calls through batching."""
    
    def __init__(self, max_batch_size: int = 10, max_wait_ms: int = 100):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.pending_requests = []
    
    async def add_request(self, request: dict) -> asyncio.Future:
        """Add request to batch."""
        future = asyncio.Future()
        
        self.pending_requests.append({
            "request": request,
            "future": future
        })
        
        if len(self.pending_requests) >= self.max_batch_size:
            await self._process_batch()
        
        return future
    
    async def _process_batch(self):
        """Process batched requests."""
        if not self.pending_requests:
            return
        
        batch = self.pending_requests[:self.max_batch_size]
        self.pending_requests = self.pending_requests[self.max_batch_size:]
        
        # Combine requests into batch
        batched_request = self._combine_requests([r["request"] for r in batch])
        
        # Execute batch
        batch_response = await self._execute_batch(batched_request)
        
        # Distribute responses
        individual_responses = self._split_response(batch_response, len(batch))
        
        for item, response in zip(batch, individual_responses):
            item["future"].set_result(response)
```

---

## Summary

### Recommended Tool Stack

**For Startups:**
- LiteLLM (cost tracking + routing)
- AgentOps (agent observability)
- PostgreSQL (cost data storage)

**For Enterprise:**
- Datadog/New Relic (full observability)
- Custom budget framework + Redis
- Cloud-native cost management (AWS/Azure/GCP)

**For Self-Hosted:**
- Langfuse (observability)
- LiteLLM (routing)
- PostgreSQL + Redis (storage)

### Key Selection Criteria

1. **Scale:** How many agents/requests per day?
2. **Budget:** What's the cost control budget?
3. **Self-hosted vs SaaS:** Data sovereignty requirements?
4. **Integration:** Existing infrastructure stack?
5. **Features:** What specific capabilities are needed?

---

*Last updated: July 2026*
*Part of the AI Knowledge Library — Category 59: AI Agent Financial Governance and Cost Control*
