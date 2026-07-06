# Core Topics in AI Agent Financial Governance and Cost Control

> Deep dive into the core concepts, mechanisms, and patterns that form the foundation of agent cost governance. This document covers budget systems, cost allocation, spending controls, and the operational practices that keep agent costs predictable and manageable.

---

## Table of Contents

1. [Budget Design Principles](#1-budget-design-principles)
2. [Cost Allocation Models](#2-cost-allocation-models)
3. [Spending Control Mechanisms](#3-spending-control-mechanisms)
4. [Token Economics for Agents](#4-token-economics-for-agents)
5. [Tool Cost Management](#5-tool-cost-management)
6. [Model Cost Optimization](#6-model-cost-optimization)
7. [Multi-Tenant Cost Management](#7-multi-tenant-cost-management)
8. [Cost Forecasting and Planning](#8-cost-forecasting-and-planning)
9. [Chargeback and Showback Models](#9-chargeback-and-showback-models)
10. [Governance Policies and Procedures](#10-governance-policies-and-procedures)

---

## 1. Budget Design Principles

### 1.1 The Budget Hierarchy

Agent budgets should follow a hierarchical structure that mirrors organizational accountability:

```
┌─────────────────────────────────────────────────────────────────────┐
│                      ENTERPRISE BUDGET                               │
│                    $500,000 / quarter                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────┐    ┌─────────────────────┐                │
│  │  BUSINESS UNIT A    │    │  BUSINESS UNIT B    │                │
│  │  $200,000 / quarter │    │  $300,000 / quarter │                │
│  └─────────┬───────────┘    └─────────┬───────────┘                │
│            │                           │                            │
│  ┌─────────┴───────────┐    ┌─────────┴───────────┐                │
│  │  TEAM: Engineering  │    │  TEAM: Support      │                │
│  │  $120,000 / quarter │    │  $180,000 / quarter │                │
│  └─────────┬───────────┘    └─────────┬───────────┘                │
│            │                           │                            │
│  ┌─────────┴───────────┐    ┌─────────┴───────────┐                │
│  │  PROJECT: Code      │    │  PROJECT: Customer  │                │
│  │  Assistant          │    │  Support Bot        │                │
│  │  $80,000 / quarter  │    │  $120,000 / quarter │                │
│  └─────────┬───────────┘    └─────────┬───────────┘                │
│            │                           │                            │
│  ┌─────────┴───────────┐    ┌─────────┴───────────┐                │
│  │  AGENT TYPE: Review │    │  AGENT TYPE: Tier 1 │                │
│  │  $30,000 / quarter  │    │  $60,000 / quarter  │                │
│  │  $1.00 / session    │    │  $0.50 / session    │                │
│  └─────────────────────┘    └─────────────────────┘                │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Budget Granularity Trade-offs

| Granularity Level | Pros | Cons | Best For |
|---|---|---|---|
| Organization-wide | Simple to manage | No visibility into usage | Small teams |
| Team-based | Aligned with accountability | May not reflect actual usage | Mid-size orgs |
| Project-based | Tied to business value | Complex allocation | Enterprise |
| Agent-type-based | Predictable per-session costs | Requires accurate classification | Multi-agent systems |
| Individual agent | Maximum control | High management overhead | Critical systems |

### 1.3 Budget Time Horizons

```python
BUDGET_TIME_FRAMES = {
    "realtime": {
        "window": "per-turn",
        "purpose": "Prevent immediate overspend",
        "enforcement": "Hard block",
        "typical_limit": "$0.01-0.10 per turn"
    },
    "session": {
        "window": "per-session",
        "purpose": "Control individual interaction cost",
        "enforcement": "Graceful degradation",
        "typical_limit": "$0.50-10.00 per session"
    },
    "daily": {
        "window": "per-day",
        "purpose": "Prevent daily overspend",
        "enforcement": "Model downgrade + alerts",
        "typical_limit": "$50-500 per day"
    },
    "weekly": {
        "window": "per-week",
        "purpose": "Weekly budget management",
        "enforcement": "Alerts + reporting",
        "typical_limit": "$200-2,000 per week"
    },
    "monthly": {
        "window": "per-month",
        "purpose": "Monthly budget adherence",
        "enforcement": "Governance reviews",
        "typical_limit": "$1,000-20,000 per month"
    }
}
```

### 1.4 Budget Alert Thresholds

```python
ALERT_THRESHOLDS = {
    "warning": {
        "percentage": 0.50,
        "action": "log_only",
        "notification": "none",
        "description": "Approaching 50% of budget"
    },
    "caution": {
        "percentage": 0.70,
        "action": "reduce_model_tier",
        "notification": "email",
        "description": "70% of budget consumed"
    },
    "elevated": {
        "percentage": 0.85,
        "action": "restrict_tool_usage",
        "notification": "slack + email",
        "description": "85% of budget consumed"
    },
    "critical": {
        "percentage": 0.95,
        "action": "minimal_mode",
        "notification": "slack + email + page",
        "description": "95% of budget consumed"
    },
    "exhausted": {
        "percentage": 1.00,
        "action": "terminate_session",
        "notification": "all_channels",
        "description": "Budget exhausted"
    }
}
```

---

## 2. Cost Allocation Models

### 2.1 Direct Attribution

Costs are directly attributed to the entity that incurred them:

```python
class DirectAttributionModel:
    """Directly attributes costs to the source."""
    
    def attribute(self, cost_event):
        return {
            "team": cost_event.agent_metadata.team,
            "project": cost_event.agent_metadata.project,
            "user": cost_event.user_id,
            "agent": cost_event.agent_id,
            "session": cost_event.session_id,
            "cost": cost_event.cost,
            "timestamp": cost_event.timestamp
        }
```

**When to use:** Simple deployments with single-tenant agents.

### 2.2 Weighted Attribution

Costs are distributed based on resource consumption weights:

```python
class WeightedAttributionModel:
    """Attributes costs based on resource consumption weights."""
    
    def attribute(self, cost_event, resource_weights):
        total_weight = sum(resource_weights.values())
        
        attributions = {}
        for entity, weight in resource_weights.items():
            attribution_pct = weight / total_weight
            attributions[entity] = {
                "cost": cost_event.cost * attribution_pct,
                "weight": weight,
                "attribution_pct": attribution_pct
            }
        
        return attributions
    
    def calculate_weights(self, agent_data):
        """Calculate weights based on resource consumption."""
        return {
            "inference": agent_data.token_usage / agent_data.total_tokens,
            "tools": agent_data.tool_calls / agent_data.total_tool_calls,
            "memory": agent_data.memory_usage / agent_data.total_memory,
            "compute": agent_data.compute_time / agent_data.total_compute
        }
```

**When to use:** Shared agent pools, multi-tenant platforms.

### 2.3 Value-Based Attribution

Costs are allocated based on the business value delivered:

```python
class ValueBasedAttributionModel:
    """Attributes costs based on business value."""
    
    def attribute(self, cost_event, value_metrics):
        # Calculate value score
        value_score = self._calculate_value_score(value_metrics)
        
        # Allocate cost proportionally to value
        return {
            "cost": cost_event.cost,
            "value_score": value_score,
            "cost_per_value": cost_event.cost / max(value_score, 0.01),
            "efficiency_rating": self._rate_efficiency(
                cost_event.cost, value_score
            )
        }
    
    def _calculate_value_score(self, metrics):
        """Calculate a composite value score."""
        weights = {
            "tasks_completed": 0.3,
            "accuracy": 0.25,
            "user_satisfaction": 0.25,
            "time_saved": 0.2
        }
        
        score = 0
        for metric, weight in weights.items():
            if metric in metrics:
                score += metrics[metric] * weight
        
        return score
```

**When to use:** ROI-focused organizations, executive reporting.

### 2.4 Hybrid Attribution

Combines multiple models based on context:

```python
class HybridAttributionModel:
    """Combines attribution models based on cost type."""
    
    ATTRIBUTION_STRATEGIES = {
        "inference": "direct",      # Direct to agent/user
        "tools": "weighted",       # Shared across consumers
        "infrastructure": "value", # Based on business value
        "overhead": "direct"       # Direct to team
    }
    
    def attribute(self, cost_event):
        attributions = {}
        
        for cost_type, strategy in self.ATTRIBUTION_STRATEGIES.items():
            if cost_type in cost_event.breakdown:
                if strategy == "direct":
                    attributions[cost_type] = self._direct_attribute(
                        cost_event.breakdown[cost_type]
                    )
                elif strategy == "weighted":
                    attributions[cost_type] = self._weighted_attribute(
                        cost_event.breakdown[cost_type]
                    )
                elif strategy == "value":
                    attributions[cost_type] = self._value_attribute(
                        cost_event.breakdown[cost_type]
                    )
        
        return attributions
```

---

## 3. Spending Control Mechanisms

### 3.1 Pre-Action Budget Checks

Check budget before executing expensive operations:

```python
class PreActionBudgetCheck:
    """Checks budget before allowing expensive operations."""
    
    def __init__(self, budget_manager):
        self.budget_manager = budget_manager
    
    def check_before_inference(self, agent_id, model, estimated_tokens):
        """Check budget before running inference."""
        estimated_cost = self._estimate_inference_cost(model, estimated_tokens)
        
        budget_status = self.budget_manager.check_budget(
            agent_id=agent_id,
            amount=estimated_cost,
            category="inference"
        )
        
        if not budget_status.approved:
            return Decision(
                allowed=False,
                reason=budget_status.reason,
                alternative=self._suggest_alternative(
                    agent_id, estimated_tokens
                )
            )
        
        return Decision(allowed=True)
    
    def check_before_tool_call(self, agent_id, tool_name):
        """Check budget before executing tool call."""
        tool_cost = self._get_tool_cost(tool_name)
        
        budget_status = self.budget_manager.check_budget(
            agent_id=agent_id,
            amount=tool_cost,
            category="tools"
        )
        
        if not budget_status.approved:
            return Decision(
                allowed=False,
                reason=budget_status.reason,
                alternative=self._suggest_tool_alternative(tool_name)
            )
        
        return Decision(allowed=True)
    
    def check_before_sub_agent(self, parent_id, child_type, estimated_cost):
        """Check budget before spawning sub-agent."""
        budget_status = self.budget_manager.check_budget(
            agent_id=parent_id,
            amount=estimated_cost,
            category="orchestration"
        )
        
        if not budget_status.approved:
            return Decision(
                allowed=False,
                reason="sub_agent_budget_exceeded",
                alternative="handle_in_parent_agent"
            )
        
        return Decision(allowed=True)
```

### 3.2 Real-Time Spending Trackers

Track spending in real-time as operations execute:

```python
class RealTimeSpendingTracker:
    """Tracks spending in real-time during agent execution."""
    
    def __init__(self, agent_id, session_id):
        self.agent_id = agent_id
        self.session_id = session_id
        self.spending = {
            "inference": 0.0,
            "tools": 0.0,
            "infrastructure": 0.0,
            "total": 0.0
        }
        self.turn_costs = []
    
    def record_inference_cost(self, model, input_tokens, output_tokens):
        """Record inference cost."""
        cost = self._calculate_inference_cost(model, input_tokens, output_tokens)
        self.spending["inference"] += cost
        self.spending["total"] += cost
        
        self.turn_costs.append({
            "type": "inference",
            "model": model,
            "tokens": {"input": input_tokens, "output": output_tokens},
            "cost": cost,
            "cumulative": self.spending["total"]
        })
        
        return cost
    
    def record_tool_cost(self, tool_name, actual_cost):
        """Record tool execution cost."""
        self.spending["tools"] += actual_cost
        self.spending["total"] += actual_cost
        
        self.turn_costs.append({
            "type": "tool",
            "tool": tool_name,
            "cost": actual_cost,
            "cumulative": self.spending["total"]
        })
        
        return actual_cost
    
    def get_spending_summary(self):
        """Get current spending summary."""
        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "total_spent": self.spending["total"],
            "breakdown": {
                "inference": self.spending["inference"],
                "tools": self.spending["tools"],
                "infrastructure": self.spending["infrastructure"]
            },
            "turn_count": len(self.turn_costs),
            "avg_cost_per_turn": (
                self.spending["total"] / max(len(self.turn_costs), 1)
            ),
            "cost_trend": self._calculate_trend()
        }
    
    def _calculate_trend(self):
        """Calculate spending trend."""
        if len(self.turn_costs) < 3:
            return "insufficient_data"
        
        recent = [t["cost"] for t in self.turn_costs[-3:]]
        earlier = [t["cost"] for t in self.turn_costs[:3]]
        
        recent_avg = sum(recent) / len(recent)
        earlier_avg = sum(earlier) / len(earlier)
        
        if recent_avg > earlier_avg * 1.5:
            return "accelerating"
        elif recent_avg < earlier_avg * 0.5:
            return "decelerating"
        else:
            return "stable"
```

### 3.3 Adaptive Budget Policies

Dynamically adjust budgets based on conditions:

```python
class AdaptiveBudgetPolicy:
    """Adapts budget policies based on context."""
    
    def __init__(self, base_policies):
        self.base_policies = base_policies
        self.adjustment_factors = {}
    
    def get_policy(self, agent_id, context):
        """Get adapted policy for current context."""
        base_policy = self.base_policies.get(agent_type)
        
        adjustments = {
            "time_of_day": self._time_adjustment(context.time),
            "user_type": self._user_adjustment(context.user_tier),
            "task_complexity": self._complexity_adjustment(context.task),
            "system_load": self._load_adjustment(context.system_load),
            "budget_remaining": self._budget_adjustment(
                context.budget_remaining_pct
            )
        }
        
        adapted_policy = self._apply_adjustments(base_policy, adjustments)
        
        return adapted_policy
    
    def _time_adjustment(self, current_time):
        """Adjust based on time of day."""
        hour = current_time.hour
        
        if 9 <= hour <= 17:  # Business hours
            return 1.0
        elif 17 < hour <= 22:  # Evening
            return 0.8
        else:  # Night/weekend
            return 0.5
    
    def _user_adjustment(self, user_tier):
        """Adjust based on user tier."""
        tier_multipliers = {
            "enterprise": 1.5,
            "business": 1.0,
            "starter": 0.7,
            "free": 0.3
        }
        return tier_multipliers.get(user_tier, 0.5)
    
    def _budget_adjustment(self, remaining_pct):
        """Adjust based on budget remaining."""
        if remaining_pct > 0.7:
            return 1.0
        elif remaining_pct > 0.5:
            return 0.8
        elif remaining_pct > 0.3:
            return 0.6
        else:
            return 0.4
```

### 3.4 Cost Circuit Breakers

Automatic intervention when costs exceed thresholds:

```python
class CostCircuitBreaker:
    """Automatically intervenes when costs exceed thresholds."""
    
    STATES = {
        "CLOSED": "normal_operation",
        "HALF_OPEN": "testing_recovery",
        "OPEN": "intervention_active"
    }
    
    def __init__(self, config):
        self.state = "CLOSED"
        self.config = config
        self.failure_count = 0
        self.last_failure_time = None
        self.recovery_timeout = config.get("recovery_timeout", 300)
    
    def check(self, current_cost, budget_limit):
        """Check if circuit breaker should trip."""
        if self.state == "OPEN":
            # Check if recovery timeout has elapsed
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                return CircuitBreakerResult(
                    state="HALF_OPEN",
                    action="test_with_reduced_capacity"
                )
            else:
                return CircuitBreakerResult(
                    state="OPEN",
                    action="block_all_operations"
                )
        
        # Check for trip condition
        if current_cost > budget_limit * self.config["trip_threshold"]:
            self.state = "OPEN"
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            return CircuitBreakerResult(
                state="OPEN",
                action=self._determine_action(),
                reason="budget_threshold_exceeded"
            )
        
        return CircuitBreakerResult(
            state="CLOSED",
            action="allow_operation"
        )
    
    def _determine_action(self):
        """Determine intervention action based on failure count."""
        if self.failure_count == 1:
            return "reduce_to_economy_model"
        elif self.failure_count == 2:
            return "limit_tool_usage"
        elif self.failure_count == 3:
            return "terminate_session"
        else:
            return "escalate_to_human"
```

---

## 4. Token Economics for Agents

### 4.1 Understanding Token Costs

Token costs are the primary driver of agent inference expenses:

```python
TOKEN_PRICING_2026 = {
    "openai": {
        "gpt-4o": {
            "input": 0.0025,   # $2.50 per 1M input tokens
            "output": 0.01,    # $10.00 per 1M output tokens
            "context_window": 128000
        },
        "gpt-4o-mini": {
            "input": 0.00015,  # $0.15 per 1M input tokens
            "output": 0.0006,  # $0.60 per 1M output tokens
            "context_window": 128000
        },
        "o1-preview": {
            "input": 0.015,    # $15.00 per 1M input tokens
            "output": 0.06,    # $60.00 per 1M output tokens
            "context_window": 128000
        }
    },
    "anthropic": {
        "claude-3.5-sonnet": {
            "input": 0.003,    # $3.00 per 1M input tokens
            "output": 0.015,   # $15.00 per 1M output tokens
            "context_window": 200000
        },
        "claude-3-haiku": {
            "input": 0.00025,  # $0.25 per 1M input tokens
            "output": 0.00125, # $1.25 per 1M output tokens
            "context_window": 200000
        }
    },
    "google": {
        "gemini-1.5-pro": {
            "input": 0.00125,  # $1.25 per 1M input tokens
            "output": 0.005,   # $5.00 per 1M output tokens
            "context_window": 2000000
        },
        "gemini-1.5-flash": {
            "input": 0.000075, # $0.075 per 1M input tokens
            "output": 0.0003,  # $0.30 per 1M output tokens
            "context_window": 1000000
        }
    }
}
```

### 4.2 Token Cost Optimization Strategies

**Strategy 1: Context Window Management**

```python
class ContextWindowOptimizer:
    """Optimizes token usage through context management."""
    
    def optimize_context(self, messages, model, budget_remaining):
        """Optimize context to fit within budget."""
        pricing = TOKEN_PRICING_2026[model.provider][model.name]
        
        # Calculate current token count
        current_tokens = self._count_tokens(messages)
        
        # Calculate budget in tokens
        max_tokens = budget_remaining / pricing["input"] * 1000000
        
        if current_tokens <= max_tokens:
            return messages  # No optimization needed
        
        # Apply optimization strategies
        optimized = messages.copy()
        
        # Strategy 1: Truncate old messages
        optimized = self._truncate_old_messages(optimized, max_tokens * 0.7)
        
        # Strategy 2: Summarize long messages
        optimized = self._summarize_long_messages(optimized, max_tokens * 0.8)
        
        # Strategy 3: Remove low-value messages
        optimized = self._remove_low_value_messages(optimized, max_tokens * 0.9)
        
        return optimized
    
    def _count_tokens(self, messages):
        """Count tokens in messages."""
        total = 0
        for msg in messages:
            # Rough estimate: 1 token per 4 characters
            total += len(msg["content"]) // 4
        return total
```

**Strategy 2: Model Selection Optimization**

```python
class ModelSelectionOptimizer:
    """Selects optimal model based on task requirements."""
    
    TASK_COMPLEXITY_MAP = {
        "simple_qa": ["gpt-4o-mini", "claude-3-haiku"],
        "summarization": ["gpt-4o-mini", "gemini-1.5-flash"],
        "analysis": ["gpt-4o", "claude-3.5-sonnet"],
        "reasoning": ["gpt-4o", "o1-preview"],
        "creative_writing": ["gpt-4o", "claude-3.5-sonnet"],
        "code_generation": ["gpt-4o", "o1-preview"],
        "code_review": ["gpt-4o", "claude-3.5-sonnet"]
    }
    
    def select_model(self, task_type, budget_state):
        """Select model based on task and budget."""
        candidate_models = self.TASK_COMPLEXITY_MAP.get(
            task_type, ["gpt-4o-mini"]
        )
        
        # Filter by budget
        affordable_models = []
        for model in candidate_models:
            if self._is_affordable(model, budget_state):
                affordable_models.append(model)
        
        if not affordable_models:
            return candidate_models[0]  # Fallback to cheapest
        
        # Return cheapest affordable model
        return min(affordable_models, key=lambda m: self._get_cost(m))
```

**Strategy 3: Prompt Optimization**

```python
class PromptOptimizer:
    """Optimizes prompts to reduce token usage."""
    
    def optimize_prompt(self, prompt, target_tokens):
        """Optimize prompt to fit within token target."""
        current_tokens = self._count_tokens(prompt)
        
        if current_tokens <= target_tokens:
            return prompt
        
        # Apply optimization techniques
        optimized = prompt
        
        # Technique 1: Remove redundant instructions
        optimized = self._remove_redundancy(optimized)
        
        # Technique 2: Compress examples
        optimized = self._compress_examples(optimized)
        
        # Technique 3: Simplify language
        optimized = self._simplify_language(optimized)
        
        # Technique 4: Use abbreviations
        optimized = self._apply_abbreviations(optimized)
        
        return optimized
```

### 4.3 Token Budget Enforcement

```python
class TokenBudgetEnforcer:
    """Enforces token budgets at multiple levels."""
    
    def __init__(self, budget_config):
        self.session_budget = budget_config["session_limit"]
        self.turn_budget = budget_config["turn_limit"]
        self.cumulative_used = 0
        self.turn_used = 0
    
    def check_before_call(self, estimated_input_tokens):
        """Check if a model call is allowed."""
        # Check session budget
        estimated_cost = self._estimate_cost(estimated_input_tokens)
        if self.cumulative_used + estimated_cost > self.session_budget:
            return BudgetCheck(
                allowed=False,
                reason="session_budget_exceeded",
                remaining=self.session_budget - self.cumulative_used
            )
        
        # Check turn budget
        if estimated_input_tokens > self.turn_budget:
            return BudgetCheck(
                allowed=False,
                reason="turn_budget_exceeded",
                remaining=self.turn_budget
            )
        
        return BudgetCheck(
            allowed=True,
            remaining=self.session_budget - self.cumulative_used
        )
    
    def record_usage(self, input_tokens, output_tokens, model):
        """Record token usage."""
        cost = self._calculate_cost(model, input_tokens, output_tokens)
        self.cumulative_used += cost
        self.turn_used = input_tokens + output_tokens
    
    def get_status(self):
        """Get current budget status."""
        return {
            "session": {
                "budget": self.session_budget,
                "used": self.cumulative_used,
                "remaining": self.session_budget - self.cumulative_used,
                "utilization_pct": (self.cumulative_used / self.session_budget) * 100
            },
            "turn": {
                "budget": self.turn_budget,
                "used": self.turn_used,
                "remaining": self.turn_budget - self.turn_used,
                "utilization_pct": (self.turn_used / self.turn_budget) * 100
            }
        }
```

---

## 5. Tool Cost Management

### 5.1 Tool Cost Registry

Maintain a registry of tool costs:

```python
TOOL_COST_REGISTRY = {
    # Search Tools
    "web_search": {
        "cost_per_call": 0.02,
        "cost_per_result": 0.005,
        "rate_limit": 10,  # per minute
        "session_limit": 50,
        "category": "search"
    },
    "academic_search": {
        "cost_per_call": 0.05,
        "cost_per_result": 0.01,
        "rate_limit": 5,
        "session_limit": 20,
        "category": "search"
    },
    
    # Data Tools
    "database_query": {
        "cost_per_call": 0.01,
        "cost_per_row": 0.001,
        "rate_limit": 20,
        "session_limit": 100,
        "category": "data"
    },
    "api_call": {
        "cost_per_call": 0.005,
        "cost_per_kb": 0.001,
        "rate_limit": 30,
        "session_limit": 200,
        "category": "data"
    },
    
    # Computation Tools
    "execute_code": {
        "cost_per_call": 0.05,
        "cost_per_second": 0.01,
        "rate_limit": 5,
        "session_limit": 30,
        "category": "computation"
    },
    "run_tests": {
        "cost_per_call": 0.10,
        "cost_per_test": 0.02,
        "rate_limit": 3,
        "session_limit": 15,
        "category": "computation"
    },
    
    # File Tools
    "read_file": {
        "cost_per_call": 0.01,
        "cost_per_kb": 0.002,
        "rate_limit": 20,
        "session_limit": 100,
        "category": "file"
    },
    "write_file": {
        "cost_per_call": 0.02,
        "cost_per_kb": 0.003,
        "rate_limit": 10,
        "session_limit": 50,
        "category": "file"
    }
}
```

### 5.2 Tool Call Optimization

```python
class ToolCallOptimizer:
    """Optimizes tool calls to reduce costs."""
    
    def optimize_calls(self, planned_calls):
        """Optimize a sequence of tool calls."""
        optimized = []
        
        # Strategy 1: Deduplicate calls
        deduplicated = self._deduplicate_calls(planned_calls)
        
        # Strategy 2: Batch similar calls
        batched = self._batch_calls(deduplicated)
        
        # Strategy 3: Cache results
        cached = self._apply_caching(batched)
        
        # Strategy 4: Use cheaper alternatives
        optimized = self._optimize_costs(cached)
        
        return optimized
    
    def _deduplicate_calls(self, calls):
        """Remove duplicate tool calls."""
        seen = set()
        unique_calls = []
        
        for call in calls:
            call_key = (call.tool_name, call.args_hash)
            if call_key not in seen:
                seen.add(call_key)
                unique_calls.append(call)
        
        return unique_calls
    
    def _batch_calls(self, calls):
        """Batch similar tool calls together."""
        batches = defaultdict(list)
        
        for call in calls:
            batches[call.tool_name].append(call)
        
        batched_calls = []
        for tool_name, tool_calls in batches.items():
            if len(tool_calls) > 1 and self._can_batch(tool_name):
                batched_calls.append(self._create_batch_call(tool_calls))
            else:
                batched_calls.extend(tool_calls)
        
        return batched_calls
    
    def _apply_caching(self, calls):
        """Apply result caching to tool calls."""
        cached_calls = []
        
        for call in calls:
            cache_key = self._get_cache_key(call)
            cached_result = self.cache.get(cache_key)
            
            if cached_result:
                call.cached = True
                call.cost = 0  # Cached results are free
            else:
                call.cache_key = cache_key
            
            cached_calls.append(call)
        
        return cached_calls
```

### 5.3 Tool Call Budgets

```python
class ToolCallBudget:
    """Manages budgets for tool calls."""
    
    def __init__(self, budget_config):
        self.session_budget = budget_config["session_limit"]
        self.per_minute_limit = budget_config["rate_limit"]
        self.used = 0
        self.minute_usage = defaultdict(int)
        self.last_minute_reset = time.time()
    
    def check_tool_call(self, tool_name, estimated_cost):
        """Check if a tool call is allowed."""
        self._reset_minute_counts()
        
        # Check session budget
        if self.used + estimated_cost > self.session_budget:
            return ToolCheck(
                allowed=False,
                reason="session_budget_exceeded",
                remaining=self.session_budget - self.used
            )
        
        # Check rate limit
        if self.minute_usage[tool_name] >= self._get_rate_limit(tool_name):
            return ToolCheck(
                allowed=False,
                reason="rate_limit_exceeded",
                retry_after=self._seconds_until_minute_reset()
            )
        
        return ToolCheck(allowed=True)
    
    def record_tool_call(self, tool_name, actual_cost):
        """Record a tool call."""
        self.used += actual_cost
        self.minute_usage[tool_name] += 1
    
    def get_tool_costs(self):
        """Get breakdown of tool costs."""
        return {
            "total": self.used,
            "by_tool": dict(self.tool_costs),
            "by_category": self._aggregate_by_category()
        }
```

---

## 6. Model Cost Optimization

### 6.1 Model Selection Matrix

Choose models based on cost and capability requirements:

```python
MODEL_SELECTION_MATRIX = {
    "simple_tasks": {
        "examples": [
            "Classification",
            "Sentiment analysis",
            "Simple Q&A",
            "Data extraction"
        ],
        "recommended_models": [
            {"model": "gpt-4o-mini", "cost_per_1k": 0.00015},
            {"model": "claude-3-haiku", "cost_per_1k": 0.00025},
            {"model": "gemini-1.5-flash", "cost_per_1k": 0.000075}
        ],
        "max_tokens_per_task": 1000
    },
    "moderate_tasks": {
        "examples": [
            "Summarization",
            "Translation",
            "Code completion",
            "Report generation"
        ],
        "recommended_models": [
            {"model": "gpt-4o", "cost_per_1k": 0.0025},
            {"model": "claude-3.5-sonnet", "cost_per_1k": 0.003},
            {"model": "gemini-1.5-pro", "cost_per_1k": 0.00125}
        ],
        "max_tokens_per_task": 4000
    },
    "complex_tasks": {
        "examples": [
            "Multi-step reasoning",
            "Code generation",
            "Research synthesis",
            "Strategic analysis"
        ],
        "recommended_models": [
            {"model": "gpt-4o", "cost_per_1k": 0.0025},
            {"model": "o1-preview", "cost_per_1k": 0.015},
            {"model": "claude-3.5-opus", "cost_per_1k": 0.015}
        ],
        "max_tokens_per_task": 8000
    }
}
```

### 6.2 Model Routing Optimization

```python
class ModelRouter:
    """Routes requests to optimal models based on cost and requirements."""
    
    def __init__(self, cost_tracker, quality_tracker):
        self.cost_tracker = cost_tracker
        self.quality_tracker = quality_tracker
    
    def route_request(self, request, budget_state):
        """Route a request to the optimal model."""
        # Analyze request complexity
        complexity = self._analyze_complexity(request)
        
        # Get candidate models
        candidates = self._get_candidates(complexity)
        
        # Filter by budget
        affordable = self._filter_by_budget(candidates, budget_state)
        
        # Rank by cost-effectiveness
        ranked = self._rank_by_effectiveness(affordable)
        
        # Select best model
        selected = ranked[0] if ranked else candidates[0]
        
        return ModelSelection(
            model=selected.model,
            reason=selected.reason,
            estimated_cost=selected.estimated_cost,
            quality_score=selected.quality_score
        )
    
    def _analyze_complexity(self, request):
        """Analyze request complexity."""
        factors = {
            "token_count": len(request.tokens),
            "requires_reasoning": request.has_reasoning_keywords,
            "requires_creativity": request.has_creative_keywords,
            "technical_depth": request.technical_complexity,
            "multi_step": request.requires_multiple_steps
        }
        
        # Simple heuristic scoring
        score = sum(factors.values()) / len(factors)
        
        if score < 0.3:
            return "simple"
        elif score < 0.7:
            return "moderate"
        else:
            return "complex"
    
    def _rank_by_effectiveness(self, models):
        """Rank models by cost-effectiveness."""
        for model in models:
            # Calculate effectiveness score
            quality = self.quality_tracker.get_quality_score(model.model)
            cost_efficiency = 1.0 / max(model.estimated_cost, 0.001)
            
            model.effectiveness_score = quality * cost_efficiency
        
        return sorted(models, key=lambda m: m.effectiveness_score, reverse=True)
```

### 6.3 Model Cost Controls

```python
class ModelCostControls:
    """Controls model usage to manage costs."""
    
    def __init__(self, policies):
        self.policies = policies
    
    def check_model_usage(self, agent_id, model, estimated_tokens):
        """Check if model usage is allowed."""
        # Check model whitelist
        allowed_models = self.policies.get_allowed_models(agent_id)
        if model not in allowed_models:
            return ModelCheck(
                allowed=False,
                reason="model_not_allowed",
                suggested_model=self._suggest_alternative(model, allowed_models)
            )
        
        # Check cost per call limit
        estimated_cost = self._estimate_cost(model, estimated_tokens)
        max_per_call = self.policies.get_max_per_call(agent_id)
        
        if estimated_cost > max_per_call:
            return ModelCheck(
                allowed=False,
                reason="cost_per_call_exceeded",
                estimated_cost=estimated_cost,
                limit=max_per_call
            )
        
        # Check daily model budget
        daily_usage = self._get_daily_usage(agent_id, model)
        daily_limit = self.policies.get_daily_model_limit(agent_id, model)
        
        if daily_usage + estimated_cost > daily_limit:
            return ModelCheck(
                allowed=False,
                reason="daily_model_budget_exceeded",
                current_usage=daily_usage,
                limit=daily_limit
            )
        
        return ModelCheck(allowed=True)
    
    def get_model_recommendation(self, agent_id, task_type, budget_state):
        """Get model recommendation based on constraints."""
        allowed_models = self.policies.get_allowed_models(agent_id)
        
        recommendations = []
        for model in allowed_models:
            quality = self._get_quality_score(model, task_type)
            cost = self._estimate_cost(model, 1000)  # Per 1k tokens
            budget_fits = cost <= budget_state.remaining / 100
            
            recommendations.append({
                "model": model,
                "quality": quality,
                "cost_per_1k": cost,
                "budget_fits": budget_fits,
                "recommendation_score": quality / max(cost, 0.001)
            })
        
        return sorted(recommendations, key=lambda r: r["recommendation_score"], reverse=True)
```

---

## 7. Multi-Tenant Cost Management

### 7.1 Tenant Isolation

Each tenant gets isolated budgets and controls:

```python
class TenantCostIsolation:
    """Isolates costs between tenants."""
    
    def __init__(self):
        self.tenant_budgets = {}
        self.tenant_usage = defaultdict(lambda: defaultdict(float))
    
    def create_tenant_budget(self, tenant_id, config):
        """Create budget for a tenant."""
        self.tenant_budgets[tenant_id] = {
            "monthly_limit": config["monthly_limit"],
            "daily_limit": config["daily_limit"],
            "per_session_limit": config["per_session_limit"],
            "allowed_models": config.get("allowed_models", ["gpt-4o-mini"]),
            "rate_limits": config.get("rate_limits", {})
        }
    
    def check_tenant_budget(self, tenant_id, cost_estimate):
        """Check if a tenant has budget remaining."""
        budget = self.tenant_budgets.get(tenant_id)
        if not budget:
            return TenantCheck(allowed=False, reason="tenant_not_found")
        
        # Check monthly limit
        monthly_usage = self.tenant_usage[tenant_id]["monthly"]
        if monthly_usage + cost_estimate > budget["monthly_limit"]:
            return TenantCheck(
                allowed=False,
                reason="monthly_limit_exceeded",
                current=monthly_usage,
                limit=budget["monthly_limit"]
            )
        
        # Check daily limit
        daily_usage = self.tenant_usage[tenant_id]["daily"]
        if daily_usage + cost_estimate > budget["daily_limit"]:
            return TenantCheck(
                allowed=False,
                reason="daily_limit_exceeded",
                current=daily_usage,
                limit=budget["daily_limit"]
            )
        
        return TenantCheck(allowed=True)
    
    def record_tenant_usage(self, tenant_id, cost, category):
        """Record usage for a tenant."""
        self.tenant_usage[tenant_id]["monthly"] += cost
        self.tenant_usage[tenant_id]["daily"] += cost
        self.tenant_usage[tenant_id][f"category_{category}"] += cost
    
    def get_tenant_report(self, tenant_id):
        """Get cost report for a tenant."""
        usage = self.tenant_usage[tenant_id]
        budget = self.tenant_budgets[tenant_id]
        
        return {
            "tenant_id": tenant_id,
            "budget": budget,
            "usage": usage,
            "utilization": {
                "monthly": usage["monthly"] / budget["monthly_limit"],
                "daily": usage["daily"] / budget["daily_limit"]
            },
            "projections": {
                "end_of_month": usage["monthly"] * 30 / max(usage.get("days_active", 1), 1),
                "remaining_days": self._calculate_remaining_budget(budget, usage)
            }
        }
```

### 7.2 Cross-Tenant Cost Sharing

Handle costs shared across tenants:

```python
class CrossTenantCostManager:
    """Manages costs shared across tenants."""
    
    def allocate_shared_cost(self, shared_cost, tenants, allocation_rules):
        """Allocate shared cost across tenants."""
        allocations = {}
        
        for tenant_id, rule in allocation_rules.items():
            if rule["type"] == "percentage":
                allocations[tenant_id] = shared_cost * rule["value"]
            elif rule["type"] == "fixed":
                allocations[tenant_id] = min(rule["value"], shared_cost)
            elif rule["type"] == "usage_based":
                usage_pct = self._get_usage_percentage(tenant_id, tenants)
                allocations[tenant_id] = shared_cost * usage_pct
            elif rule["type"] == "equal_split":
                allocations[tenant_id] = shared_cost / len(tenants)
        
        return allocations
    
    def _get_usage_percentage(self, tenant_id, all_tenants):
        """Get usage percentage for a tenant."""
        tenant_usage = self._get_tenant_usage(tenant_id)
        total_usage = sum(self._get_tenant_usage(t) for t in all_tenants)
        
        return tenant_usage / max(total_usage, 1)
```

---

## 8. Cost Forecasting and Planning

### 8.1 Historical Cost Analysis

```python
class CostForecaster:
    """Forecasts future costs based on historical data."""
    
    def forecast(self, historical_data, forecast_period_days):
        """Forecast costs for a future period."""
        # Calculate trends
        daily_costs = self._aggregate_daily(historical_data)
        trend = self._calculate_trend(daily_costs)
        seasonality = self._calculate_seasonality(daily_costs)
        variance = self._calculate_variance(daily_costs)
        
        # Generate forecast
        forecast = []
        for day in range(forecast_period_days):
            base_cost = daily_costs[-1] * (1 + trend) ** day
            seasonal_adj = seasonality.get(day % 7, 1.0)
            predicted = base_cost * seasonal_adj
            
            forecast.append({
                "day": day + 1,
                "predicted_cost": predicted,
                "confidence_interval": {
                    "low": predicted * (1 - variance),
                    "high": predicted * (1 + variance)
                }
            })
        
        return Forecast(
            daily_forecast=forecast,
            total_predicted=sum(f["predicted_cost"] for f in forecast),
            total_confidence={
                "low": sum(f["confidence_interval"]["low"] for f in forecast),
                "high": sum(f["confidence_interval"]["high"] for f in forecast)
            }
        )
    
    def _calculate_trend(self, daily_costs):
        """Calculate cost trend."""
        if len(daily_costs) < 7:
            return 0
        
        # Simple moving average trend
        recent = daily_costs[-7:]
        earlier = daily_costs[-14:-7] if len(daily_costs) >= 14 else daily_costs[:7]
        
        recent_avg = sum(recent) / len(recent)
        earlier_avg = sum(earlier) / len(earlier)
        
        return (recent_avg - earlier_avg) / max(earlier_avg, 1)
```

### 8.2 Budget Planning Tools

```python
class BudgetPlanner:
    """Helps plan and set agent budgets."""
    
    def plan_budget(self, requirements):
        """Plan budget based on requirements."""
        # Estimate costs based on usage patterns
        estimated_costs = {
            "inference": self._estimate_inference_costs(
                requirements["expected_queries"],
                requirements["avg_tokens_per_query"],
                requirements["model_mix"]
            ),
            "tools": self._estimate_tool_costs(
                requirements["expected_tool_calls"],
                requirements["tool_mix"]
            ),
            "infrastructure": self._estimate_infrastructure_costs(
                requirements["concurrent_agents"],
                requirements["storage_requirements"]
            )
        }
        
        total_estimated = sum(estimated_costs.values())
        
        # Add buffer
        recommended_budget = total_estimated * 1.2  # 20% buffer
        
        return BudgetPlan(
            estimated_costs=estimated_costs,
            total_estimated=total_estimated,
            recommended_budget=recommended_budget,
            breakdown_by_period=self._breakdown_by_period(recommended_budget),
            optimization_opportunities=self._identify_optimizations(estimated_costs)
        )
```

---

## 9. Chargeback and Showback Models

### 9.1 Showback Model

Report costs without enforcing allocation:

```python
class ShowbackModel:
    """Reports costs to stakeholders without enforcement."""
    
    def generate_showback_report(self, period, filters=None):
        """Generate showback report."""
        costs = self._get_costs(period, filters)
        
        report = {
            "period": period,
            "total_cost": sum(c["cost"] for c in costs),
            "breakdown": {
                "by_team": self._aggregate_by_team(costs),
                "by_project": self._aggregate_by_project(costs),
                "by_agent_type": self._aggregate_by_agent_type(costs),
                "by_model": self._aggregate_by_model(costs),
                "by_user": self._aggregate_by_user(costs)
            },
            "trends": self._calculate_trends(costs),
            "efficiency_metrics": self._calculate_efficiency(costs),
            "recommendations": self._generate_recommendations(costs)
        }
        
        return report
```

### 9.2 Chargeback Model

Enforce cost allocation for billing:

```python
class ChargebackModel:
    """Enforces cost allocation for internal billing."""
    
    def __init__(self, billing_rates):
        self.billing_rates = billing_rates
    
    def calculate_chargeback(self, usage_data, period):
        """Calculate chargeback amounts."""
        chargebacks = {}
        
        for entity_id, usage in usage_data.items():
            chargeback = {
                "inference": usage["inference_tokens"] * self.billing_rates["inference_per_token"],
                "tools": usage["tool_calls"] * self.billing_rates["per_tool_call"],
                "infrastructure": usage["compute_seconds"] * self.billing_rates["per_compute_second"],
                "storage": usage["storage_gb"] * self.billing_rates["per_storage_gb"]
            }
            
            chargeback["total"] = sum(chargeback.values())
            chargebacks[entity_id] = chargeback
        
        return chargebacks
```

### 9.3 ROI Calculation

```python
class AgentROICalculator:
    """Calculates ROI for agent deployments."""
    
    def calculate_roi(self, costs, value_metrics):
        """Calculate ROI for agent usage."""
        # Calculate total costs
        total_costs = sum(costs.values())
        
        # Calculate total value
        total_value = self._calculate_value(value_metrics)
        
        # Calculate ROI
        roi = (total_value - total_costs) / max(total_costs, 1) * 100
        
        return ROIReport(
            total_costs=total_costs,
            total_value=total_value,
            roi_percentage=roi,
            cost_per_outcome=total_costs / max(value_metrics["outcomes"], 1),
            value_per_dollar=total_value / max(total_costs, 1),
            payback_period=self._calculate_payback_period(costs, value_metrics)
        )
    
    def _calculate_value(self, metrics):
        """Calculate business value from metrics."""
        value = 0
        
        # Time saved
        if "hours_saved" in metrics:
            value += metrics["hours_saved"] * self._get_hourly_rate()
        
        # Revenue generated
        if "revenue_attributed" in metrics:
            value += metrics["revenue_attributed"]
        
        # Cost savings
        if "costs_avoided" in metrics:
            value += metrics["costs_avoided"]
        
        # Quality improvements
        if "quality_improvement" in metrics:
            value += metrics["quality_improvement"] * self._get_quality_value()
        
        return value
```

---

## 10. Governance Policies and Procedures

### 10.1 Policy Framework

```yaml
# agent-cost-governance-policies.yaml
policies:
  budget_management:
    - name: "Budget Approval Process"
      description: "All agent budgets must be approved by finance"
      owner: "CFO"
      enforcement: "mandatory"
      
    - name: "Quarterly Budget Review"
      description: "Review and adjust budgets quarterly"
      owner: "FinOps Team"
      enforcement: "mandatory"
      
    - name: "Emergency Budget Requests"
      description: "Process for urgent budget increases"
      owner: "Engineering Leadership"
      enforcement: "optional"
  
  cost_controls:
    - name: "Circuit Breaker Implementation"
      description: "All agents must have circuit breakers"
      owner: "Platform Team"
      enforcement: "mandatory"
      
    - name: "Cost Attribution"
      description: "All costs must be attributed to teams"
      owner: "FinOps Team"
      enforcement: "mandatory"
      
    - name: "Anomaly Detection"
      description: "Implement anomaly detection for cost spikes"
      owner: "SRE Team"
      enforcement: "mandatory"
  
  reporting:
    - name: "Daily Cost Reports"
      description: "Generate daily cost summary reports"
      owner: "FinOps Team"
      enforcement: "mandatory"
      
    - name: "Monthly Executive Reports"
      description: "Monthly cost reports for leadership"
      owner: "Finance"
      enforcement: "mandatory"
      
    - name: "Quarterly ROI Reviews"
      description: "Quarterly ROI analysis for all agent deployments"
      owner: "Business Intelligence"
      enforcement: "mandatory"
```

### 10.2 Approval Workflows

```python
class BudgetApprovalWorkflow:
    """Manages budget approval workflows."""
    
    def __init__(self):
        self.approvers = {
            "small": ["team_lead"],
            "medium": ["team_lead", "engineering_manager"],
            "large": ["team_lead", "engineering_manager", "director"],
            "enterprise": ["team_lead", "engineering_manager", "director", "cfo"]
        }
    
    def submit_budget_request(self, request):
        """Submit a budget request for approval."""
        # Determine approval level
        level = self._determine_level(request.amount)
        approvers = self.approvers[level]
        
        # Create approval request
        approval_request = ApprovalRequest(
            requester=request.requester,
            amount=request.amount,
            purpose=request.purpose,
            approvers=approvers,
            status="pending",
            created_at=datetime.now()
        )
        
        # Start approval workflow
        self._start_approval_workflow(approval_request)
        
        return approval_request
    
    def approve_request(self, request_id, approver_id, comments=""):
        """Approve a budget request."""
        request = self._get_request(request_id)
        
        # Check if approver is authorized
        if approver_id not in request.approvers:
            return ApprovalResult(
                success=False,
                reason="unauthorized_approver"
            )
        
        # Record approval
        request.approvals.append({
            "approver": approver_id,
            "timestamp": datetime.now(),
            "comments": comments
        })
        
        # Check if all approvals received
        if len(request.approvals) >= len(request.approvers):
            request.status = "approved"
            self._allocate_budget(request)
        
        return ApprovalResult(success=True)
```

### 10.3 Compliance and Audit

```python
class CostComplianceAuditor:
    """Audits cost compliance with policies."""
    
    def audit(self, period):
        """Run compliance audit for a period."""
        violations = []
        
        # Check budget compliance
        budget_violations = self._check_budget_compliance(period)
        violations.extend(budget_violations)
        
        # Check attribution compliance
        attribution_violations = self._check_attribution_compliance(period)
        violations.extend(attribution_violations)
        
        # Check policy compliance
        policy_violations = self._check_policy_compliance(period)
        violations.extend(policy_violations)
        
        return AuditReport(
            period=period,
            total_violations=len(violations),
            violations=violations,
            compliance_score=self._calculate_compliance_score(violations),
            recommendations=self._generate_recommendations(violations)
        )
    
    def _check_budget_compliance(self, period):
        """Check if all agents stayed within budgets."""
        violations = []
        
        agents = self._get_all_agents()
        for agent in agents:
            usage = self._get_agent_usage(agent.id, period)
            budget = self._get_agent_budget(agent.id)
            
            if usage > budget:
                violations.append(Violation(
                    type="budget_exceeded",
                    severity="high",
                    agent_id=agent.id,
                    usage=usage,
                    budget=budget,
                    overage=usage - budget
                ))
        
        return violations
    
    def _check_attribution_compliance(self, period):
        """Check if all costs are properly attributed."""
        violations = []
        
        costs = self._get_all_costs(period)
        for cost in costs:
            if not cost.has_attribution():
                violations.append(Violation(
                    type="missing_attribution",
                    severity="medium",
                    cost_id=cost.id,
                    amount=cost.amount,
                    recommendation="Add team and project tags"
                ))
        
        return violations
```

---

## Summary

### Key Concepts

1. **Budget Hierarchy:** Enterprise → Team → Project → Agent Type → Individual Session
2. **Cost Allocation:** Direct, weighted, value-based, or hybrid models
3. **Spending Controls:** Pre-action checks, real-time tracking, adaptive policies, circuit breakers
4. **Token Economics:** Understanding pricing, optimizing usage, enforcing budgets
5. **Tool Cost Management:** Registry, optimization, budgets per tool
6. **Model Cost Optimization:** Selection matrix, routing, cost controls
7. **Multi-Tenant Management:** Tenant isolation, cross-tenant cost sharing
8. **Forecasting:** Historical analysis, trend prediction, budget planning
9. **Chargeback/Showback:** Reporting models, ROI calculation
10. **Governance:** Policies, approval workflows, compliance auditing

### Best Practices

1. **Start with simple budgets** and add complexity as needed
2. **Implement circuit breakers** to prevent catastrophic cost events
3. **Attribute costs from day one** — don't wait until you have a problem
4. **Use model tiering** — not every task needs the most expensive model
5. **Monitor tool costs** — they're often overlooked but significant
6. **Forecast and plan** — don't just react to costs
7. **Automate governance** — manual reviews don't scale
8. **Measure efficiency** — cost per successful outcome matters most

### Next Steps

→ See [03-Technical-Deep-Dive.md](03-Technical-Deep-Dive.md) for implementation architectures
→ See [04-Tools-and-Frameworks.md](04-Tools-and-Frameworks.md) for tooling recommendations
→ See [05-Future-Outlook.md](05-Future-Outlook.md) for emerging trends

---

*Last updated: July 2026*
*Part of the AI Knowledge Library — Category 59: AI Agent Financial Governance and Cost Control*
