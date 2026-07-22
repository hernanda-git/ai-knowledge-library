# Agent Registry & Versioning

## Table of Contents

1. [Overview](#overview)
2. [Agent Versioning Fundamentals](#agent-versioning-fundamentals)
3. [Semantic Versioning for Agents](#semantic-versioning-for-agents)
4. [Model Version Pinning](#model-version-pinning)
5. [Tool Version Compatibility](#tool-version-compatibility)
6. [Prompt Versioning](#prompt-versioning)
7. [Agent Registry Architecture](#agent-registry-architecture)
8. [Deployment Strategies](#deployment-strategies)
9. [Rollback Procedures](#rollback-procedures)
10. [A/B Testing for Agents](#ab-testing-for-agents)
11. [Registry Data Model](#registry-data-model)
12. [API Design](#api-design)
13. [CI/CD Pipeline Integration](#cicd-pipeline-integration)
14. [Governance & Compliance](#governance--compliance)
15. [Cross-References](#cross-references)

---

## Overview

As organizations scale from one agent to dozens or hundreds, managing agent versions, configurations, and deployments becomes a critical infrastructure need. Agent registries provide the centralized system for versioning, discovery, and lifecycle management of AI agents.

### Why Agent Versioning Matters

| Challenge | Without Registry | With Registry |
|-----------|----------------|---------------|
| Model updates | Manual tracking across teams | Pinned version per agent, controlled upgrades |
| Debugging | "Which model was running when this error occurred?" | Full version audit trail |
| Rollbacks | Rebuild from git history | One-click rollback to previous version |
| Compliance | No provenance | Complete lineage: prompt→model→tools→deployment |
| Collaboration | Shared configs overwritten | Immutable versions, safe experimentation |

### Agent Registry vs Traditional Service Registry

| Aspect | Service Registry (Consul, Eureka) | Agent Registry |
|--------|-----------------------------------|----------------|
| Registers | IP:port of services | Agent configuration + metadata |
| Versioning | Usually implicit (deployment version) | Explicit semantic versioning |
| Model info | N/A | Pinned model + prompt version |
| Tool definitions | N/A | Declared tools with signatures |
| Health checks | HTTP/TCP | Agent behavior evaluation |
| Rollback | Container version rollback | Rollback config + prompts + tools |

---

## Agent Versioning Fundamentals

### What Constitutes an Agent Version

An agent version is the combination of:

```
Agent Version = {
    agent_definition: version,     # YAML/JSON agent spec
    model_config: {                 # model:version pairs
        primary_llm: "gpt-4o:2026-04-15",
        evaluator_llm: "claude-4:2026-05-01"
    },
    prompt_set: version,           # system prompt + few-shot examples
    tools: [{                      # tool signatures
        name: "search_web",
        version: "2.1.0"
    }],
    dependencies: {                # framework/library versions
        langchain: "0.3.15",
        langgraph: "0.2.8"
    },
    config: {                      # runtime configuration
        max_iterations: 20,
        temperature: 0.7,
        timeout: 30000
    }
}
```

A change to any component produces a new agent version.

### Version Components

```
vMAJOR.MINOR.PATCH-agentBUILD
```

| Component | When to Bump | Example |
|-----------|-------------|---------|
| **MAJOR** | Breaking changes (tool signatures changed, model swapped, behavior-breaking prompt changes) | 1.0.0 → 2.0.0 |
| **MINOR** | New capabilities (added tool, new prompt section, additional model) | 1.0.0 → 1.1.0 |
| **PATCH** | Bug fixes, performance improvements, prompt tweaks | 1.0.0 → 1.0.1 |
| **BUILD** | Auto-increment CI build number | 1.0.0-42 |

---

## Semantic Versioning for Agents

### Version Definition Schema

```yaml
# agent-version.yaml
agent:
  name: customer-support-agent
  version: 2.3.1
  build: 891
  created: 2026-06-10T14:30:00Z
  author: "ml-platform-team"
  
semver:
  major: 2    # Breaking change: switched from GPT-4 to Claude 4
  minor: 3    # Added: web_search tool integration
  patch: 1    # Fixed: timeout handling for long responses
  
changelog:
  major:
    - "Migrated primary LLM from gpt-4-turbo to claude-4-opus"
    - "Restructured system prompt for Claude's XML format"
    - "Removed deprecated tool: legacy_db_query"
  minor:
    - "Added web_search tool with source filtering"
    - "Added fallback model: gpt-4o-mini for simple queries"
    - "Expanded supported languages to 12"
  patch:
    - "Fixed infinite loop on empty search results"
    - "Reduced timeout from 60s to 30s for faster failure"
    - "Improved error messages for API failures"
```

### Breaking Change Detection

Automated detection should flag breaking changes:

```python
def detect_breaking_changes(old_version, new_version):
    breaking = []
    
    # 1. Tool signature changes
    old_tools = {t['name']: t for t in old_version.tools}
    new_tools = {t['name']: t for t in new_version.tools}
    
    for name, new_tool in new_tools.items():
        if name not in old_tools:
            breaking.append(f"New tool added: {name} (non-breaking, MINOR)")
        else:
            old_tool = old_tools[name]
            if new_tool['parameters'] != old_tool['parameters']:
                breaking.append(f"Tool {name} parameters changed (BREAKING)")
            if new_tool['output_schema'] != old_tool['output_schema']:
                breaking.append(f"Tool {name} output changed (BREAKING)")
    
    # 2. Model change
    if new_version.model.primary != old_version.model.primary:
        breaking.append(f"Primary model changed: {old_version.model.primary} → {new_version.model.primary}")
    
    # 3. System prompt structural changes
    if new_version.prompts.system.structure != old_version.prompts.system.structure:
        breaking.append("System prompt structure changed")
    
    return breaking
```

---

## Model Version Pinning

### Why Pin Model Versions

Models change over time — even the same model name can produce different outputs after provider updates:

| Risk | Example | Impact |
|------|---------|--------|
| Silent model update | "gpt-4o" changes behavior | Agent produces different responses |
| Deprecation | "gpt-3.5-turbo" retired | Agent fails at runtime |
| Fine-tune drift | Base model fine-tune updated | Classification accuracy changes |
| Quantization change | Provider switches quantization | Quality degradation |

### Pinning Strategies

**1. Exact Date Pinning (Preferred)**

```yaml
model:
  provider: openai
  name: gpt-4o
  version: "2026-04-15"    # Pinned to specific snapshot
  fallback: gpt-4o-mini    # Low-cost fallback for simple tasks
```

**2. Semantic Version Pinning**

```yaml
model:
  provider: anthropic
  name: claude-4-opus
  version: ">=4.0.0, <5.0.0"  # Accept minor/patch within major
```

**3. Hash Pinning (Maximum Safety)**

```yaml
model:
  provider: anthropic
  name: claude-4-opus
  version: "2026-04-15"
  model_hash: "sha256:abc123..."  # Verify model weights
```

### Model Registry Integration

```
Agent Config → Model Registry → Model Endpoint
                    ↓
          ┌─────────────────┐
          │ Model Metadata   │
          │ - version        │
          │ - hash           │
          │ - endpoint URL   │
          │ - cost/1M tokens │
          │ - latency p50    │
          │ - knowledge cut  │
          └─────────────────┘
```

---

## Tool Version Compatibility

### Tool Version Matrix

```yaml
tools:
  - name: web_search
    version: 2.1.0
    min_agent_version: "2.0.0"
    max_agent_version: "3.0.0"
    description: "Web search with content extraction"
    parameters:
      query: string
      max_results: integer (default: 5)
    output_schema:
      results: array[SearchResult]
    changelog:
      "2.1.0": "Added source_filter parameter"
      "2.0.0": "Initial tool version"
  
  - name: db_query
    version: 1.2.0
    min_agent_version: "1.0.0"
    max_agent_version: "2.5.0"
    description: "Execute read-only SQL queries"
    parameters:
      query: string
      max_rows: integer
    output_schema:
      columns: array[string]
      rows: array[array]
```

### Compatibility Enforcement

```python
def validate_tool_compatibility(agent_version, tool_registry):
    for tool_def in agent_version.tools:
        tool_meta = tool_registry.get(tool_def.name)
        
        if not tool_meta:
            raise ConfigError(f"Tool {tool_def.name} not in registry")
        
        if tool_def.version != tool_meta.current_version:
            logger.warning(f"Tool {tool_def.name}: using v{tool_def.version}, current v{tool_meta.current_version}")
        
        # Check agent compatibility range
        if not (tool_meta.min_agent_version <= agent_version <= tool_meta.max_agent_version):
            raise ConfigError(
                f"Tool {tool_def.name} v{tool_def.version} incompatible with agent v{agent_version}"
            )
```

---

## Prompt Versioning

### Prompt Management

Prompts should be versioned independently of agent code:

```yaml
prompts:
  system:
    version: 3.2.1
    hash: "sha256:def456..."
    content: |
      You are a helpful customer support agent for Acme Corp.
      Follow these rules:
      1. Always be polite and professional
      2. If you don't know, say so — never hallucinate
      3. For refund requests, escalate to human agent
      ...
  
  few_shot:
    version: 2.0.0
    examples:
      - input: "My order hasn't arrived"
        output: "I understand your concern. Let me check the tracking..."
      - input: "I want a refund"
        output: "I'll transfer you to our refund specialist..."
```

### Prompt Diffing

```python
def prompt_diff(old_version, new_version):
    """Show changes between prompt versions for review."""
    from difflib import unified_diff
    
    old_lines = old_version.prompts.system.content.splitlines()
    new_lines = new_version.prompts.system.content.splitlines()
    
    diff = list(unified_diff(old_lines, new_lines, lineterm=''))
    
    # Categorize changes
    additions = [l for l in diff if l.startswith('+') and not l.startswith('+++')]
    removals = [l for l in diff if l.startswith('-') and not l.startswith('---')]
    
    return {
        'lines_added': len(additions),
        'lines_removed': len(removals),
        'diff': '\n'.join(diff),
        'breaking': len(removals) > 10  # Large removals likely breaking
    }
```

---

## Agent Registry Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Agent Registry                       │
├─────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌──────────┐  ┌───────────┐  ┌────────┐ │
│  │ Agent   │  │ Version  │  │ Tool      │  │ Model  │ │
│  │ Catalog │  │ Manager  │  │ Registry  │  │ Catalog│ │
│  └────┬────┘  └────┬─────┘  └─────┬─────┘  └───┬────┘ │
│       │            │              │             │      │
│  ┌────┴────────────┴──────────────┴─────────────┴────┐ │
│  │               Storage Layer (PostgreSQL)          │ │
│  └────────────────────────┬───────────────────────────┘ │
└───────────────────────────┼─────────────────────────────┘
                            │
┌───────────────────────────┼─────────────────────────────┐
│                    API Gateway                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │  REST API (register, list, get, rollback, diff) │   │
│  └─────────────────────────────────────────────────┘   │
└───────────────────────────┼─────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   CI/CD        │   │   Agent       │   │   Monitoring  │
│   Pipeline     │   │   Runtime     │   │   Dashboard   │
└───────────────┘   └───────────────┘   └───────────────┘
```

### Storage Schema (PostgreSQL)

```sql
-- Agent registry tables

CREATE TABLE agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    owner_team VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE agent_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID REFERENCES agents(id),
    version VARCHAR(50) NOT NULL,  -- semver
    build_number INTEGER NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',  -- pending, active, deprecated, rolled_back
    config JSONB NOT NULL,  -- Full agent configuration
    prompt_hash VARCHAR(64),
    model_config JSONB,
    tools JSONB,
    changelog TEXT,
    created_by VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(agent_id, version)
);

CREATE TABLE tool_registry (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    description TEXT,
    parameters JSONB NOT NULL,
    output_schema JSONB,
    min_agent_version VARCHAR(50),
    max_agent_version VARCHAR(50),
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(name, version)
);

CREATE TABLE deployment_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_version_id UUID REFERENCES agent_versions(id),
    environment VARCHAR(50) NOT NULL,  -- dev, staging, prod
    strategy VARCHAR(50),  -- blue_green, canary, rolling
    status VARCHAR(20),  -- in_progress, success, failed, rolled_back
    metadata JSONB,
    deployed_at TIMESTAMPTZ DEFAULT NOW(),
    rolled_back_at TIMESTAMPTZ
);
```

---

## Deployment Strategies

### Blue/Green Deployment

```
User Traffic → Load Balancer
                    │
            ┌───────┴───────┐
            ▼               ▼
      ┌──────────┐   ┌──────────┐
      │ Blue     │   │ Green    │
      │ v2.3.1   │   │ v2.3.2   │
      │ (current)│   │ (candidate)│
      └──────────┘   └──────────┘
                      │
            ┌─────────┴──────────┐
            ▼                    ▼
     Evaluation (metrics)    Smoke Tests
            │
            ├── Pass → Switch traffic to Green
            └── Fail → Keep Blue, rollback Green
```

### Canary Deployment for Agents

```yaml
canary:
  stages:
    - name: "1% traffic"
      duration: 30m
      evaluation: 
        - error_rate < 0.1%
        - avg_latency < baseline * 1.2
        - task_completion > baseline * 0.95
    - name: "10% traffic"
      duration: 1h
      evaluation:
        - error_rate < 0.05%
        - task_completion > baseline
        - user_satisfaction > baseline
    - name: "50% traffic"
      duration: 2h
      evaluation:
        - all_metrics > baseline
    - name: "100% traffic"
      promotion: auto
```

### Rollback Automation

```python
def auto_rollback(agent_name, new_version_id, deployment_id):
    """Automatic rollback if error rate spikes."""
    
    metrics = get_metrics_stream(deployment_id)
    
    # Monitor for 5 minutes
    time.sleep(300)
    
    error_rate = metrics.error_rate()
    baseline = metrics.baseline_error_rate(agent_name)
    
    if error_rate > baseline * 3:  # 3x baseline
        logger.warning(f"Error rate {error_rate} exceeds threshold {baseline*3}")
        rollback(agent_name, new_version_id)
        
        notify_slack(
            channel="#agent-alerts",
            message=f"🚨 Auto-rolled back {agent_name} v{version}\n"
                    f"Error rate: {error_rate:.2%} (threshold: {baseline*3:.2%})"
        )
        return False
    
    return True
```

---

## A/B Testing for Agents

### A/B Test Architecture

```
Users → Traffic Splitter → Variant A (control) → Metrics
         (50/50 split)     Variant B (treatment) → Metrics
                                ↓
                          Statistical Analysis
                                ↓
                          Decision: Promote or Reject
```

### A/B Test Configuration

```yaml
ab_test:
  name: "prompt-optimization-v2"
  agent: customer-support-agent
  start: 2026-06-10T00:00:00Z
  end: 2026-06-17T00:00:00Z
  duration: 7d
  
  variants:
    - name: control
      agent_version: 2.3.1
      traffic_percent: 50
    
    - name: treatment
      agent_version: 2.4.0-rc1
      traffic_percent: 50
      note: "New system prompt with structured output"
  
  metrics:
    primary:
      - name: task_completion_rate
        type: proportion
        improve: increase
      - name: avg_resolution_time
        type: duration
        improve: decrease
    
    secondary:
      - name: user_satisfaction_score
        type: score
      - name: escalation_rate
        type: proportion
      - name: hallucination_rate
        type: proportion
  
  analysis:
    method: "bayesian"  # or "frequentist"
    significance_level: 0.05
    minimum_detectable_effect: 0.02
```

### A/B Test Results Dashboard

```python
def render_ab_results(test_config, results):
    """Render A/B test results for decision-making."""
    
    control = results['control']
    treatment = results['treatment']
    
    print(f"A/B Test: {test_config['name']}")
    print(f"Duration: {test_config['duration']}")
    print()
    print(f"{'Metric':<30} {'Control':<15} {'Treatment':<15} {'Delta':<10} {'Winner':<10}")
    print("-" * 80)
    
    for metric in test_config['metrics']['primary']:
        c_val = control['metrics'][metric['name']]
        t_val = treatment['metrics'][metric['name']]
        delta = t_val - c_val
        better = delta > 0 if metric['improve'] == 'increase' else delta < 0
        winner = "Treatment ✓" if better else "Control"
        print(f"{metric['name']:<30} {c_val:<15.4f} {t_val:<15.4f} {delta:<+10.4f} {winner}")
    
    print()
    significance = results['statistical_significance']
    if significance < 0.05:
        print(f"✅ Result statistically significant (p={significance:.4f})")
        winner = "Treatment" if results['treatment_wins'] else "Control"
        print(f"Recommendation: Promote {winner} to production")
    else:
        print(f"⚠️ Result not statistically significant (p={significance:.4f})")
        print("Recommendation: Extend test or increase sample size")
```

---

## Registry API Design

### REST API Endpoints

```yaml
openapi: 3.0.0
info:
  title: Agent Registry API
  version: 1.0.0

paths:
  /agents:
    get:
      summary: List all registered agents
      parameters:
        - name: status
          in: query
          schema:
            type: string
            enum: [active, deprecated, archived]
    post:
      summary: Register new agent
  
  /agents/{name}/versions:
    get:
      summary: List all versions of an agent
    post:
      summary: Register new version
  
  /agents/{name}/versions/{version}:
    get:
      summary: Get agent version configuration
    delete:
      summary: Deprecate a version
  
  /agents/{name}/deploy:
    post:
      summary: Deploy agent version to environment
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                version:
                  type: string
                environment:
                  type: string
                  enum: [dev, staging, prod]
                strategy:
                  type: string
                  enum: [blue_green, canary, rolling]
  
  /agents/{name}/rollback:
    post:
      summary: Rollback to previous version
  
  /agents/{name}/ab-tests:
    get:
      summary: List A/B tests for agent
    post:
      summary: Create A/B test
```

### Client SDK Example

```python
class AgentRegistryClient:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    def register_version(self, agent_name, config, changelog=""):
        """Register a new agent version."""
        version = self._generate_semver(agent_name, config)
        response = requests.post(
            f"{self.base_url}/agents/{agent_name}/versions",
            json={
                "version": version,
                "config": config,
                "changelog": changelog
            },
            headers=self.headers
        )
        return response.json()
    
    def deploy(self, agent_name, version, environment="prod", strategy="blue_green"):
        """Deploy agent version to environment."""
        response = requests.post(
            f"{self.base_url}/agents/{agent_name}/deploy",
            json={
                "version": version,
                "environment": environment,
                "strategy": strategy
            },
            headers=self.headers
        )
        return response.json()
    
    def rollback(self, agent_name, environment="prod"):
        """Rollback to previous version."""
        response = requests.post(
            f"{self.base_url}/agents/{agent_name}/rollback",
            json={"environment": environment},
            headers=self.headers
        )
        return response.json()
    
    def get_active_version(self, agent_name, environment="prod"):
        """Get currently active version for an agent."""
        response = requests.get(
            f"{self.base_url}/agents/{agent_name}/versions/active",
            params={"environment": environment},
            headers=self.headers
        )
        return response.json()
```

---

## CI/CD Pipeline Integration

### GitHub Actions Workflow

```yaml
name: Agent CI/CD

on:
  push:
    branches: [main]
    paths:
      - 'agents/**'
      - 'prompts/**'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Validate agent config
        run: |
          python scripts/validate_agent_config.py agents/*.yaml
      
      - name: Detect breaking changes
        run: |
          python scripts/detect_breaking.py \
            --current agents/current.yaml \
            --previous agents/previous.yaml
      
      - name: Run agent eval suite
        run: |
          python scripts/evaluate_agent.py \
            --config agents/new.yaml \
            --dataset tests/eval_suite.json

  register:
    needs: validate
    runs-on: ubuntu-latest
    steps:
      - name: Register in Agent Registry
        run: |
          python scripts/register_agent.py \
            --config agents/new.yaml \
            --changelog "$(cat CHANGELOG.md)"
  
  deploy-staging:
    needs: register
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - name: Deploy to staging
        run: |
          python scripts/deploy_agent.py \
            --agent customer-support \
            --version $(cat .agent-version) \
            --environment staging
  
  deploy-prod:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment: prod
    steps:
      - name: Canary deploy (10%)
        run: |
          python scripts/canary_deploy.py \
            --agent customer-support \
            --version $(cat .agent-version) \
            --percentage 10
      
      - name: Monitor canary
        run: |
          python scripts/monitor_canary.py \
            --agent customer-support \
            --duration 30m \
            --threshold-error 0.001
      
      - name: Full rollout
        if: success()
        run: |
          python scripts/full_rollout.py \
            --agent customer-support
```

---

## Governance & Compliance

### Agent Version Audit Trail

```sql
-- Full audit trail for compliance
CREATE VIEW agent_version_audit AS
SELECT 
    a.name AS agent_name,
    av.version,
    av.created_at AS version_created,
    av.created_by,
    av.changelog,
    dh.environment,
    dh.deployed_at,
    dh.deployed_by,
    dh.strategy,
    dh.rolled_back_at
FROM agents a
JOIN agent_versions av ON a.id = av.agent_id
LEFT JOIN deployment_history dh ON av.id = dh.agent_version_id
ORDER BY av.created_at DESC;
```

### Compliance Requirements

| Requirement | Implementation | Audit Evidence |
|-------------|---------------|----------------|
| **Change tracking** | Every version change logged with author, timestamp, reason | `agent_version_audit` view |
| **Approval gates** | Major version bumps require team lead approval | Approval status in registry |
| **Rollback plan** | Previous version retained indefinitely | Immutable version history |
| **Model provenance** | Model name, version, hash recorded | Model catalog entries |
| **Prompt review** | Prompt changes reviewed before deployment | PR approvals linked to version |
| **Testing evidence** | Eval results attached to version | Version metadata contains eval_summary |
| **Data retention** | All versions retained for 3+ years | Storage with archival policy |

---

## Cross-References

- **Agent Tracing & Observability** → [02-Agent-Tracing-and-Observability.md](./02-Agent-Tracing-and-Observability.md) — Trace version changes
- **Agent Evaluation** → [04-Agent-Evaluation-and-Testing.md](./04-Agent-Evaluation-and-Testing.md) — Eval suite for version validation
- **Agent Reliability** → [07-Agent-Reliability-and-Resilience.md](./07-Agent-Reliability-and-Resilience.md) — Deployment reliability patterns
- **CI/CD for AI** → [05-Enterprise/01-Enterprise-AI-Deployment.md](../05-Enterprise/01-Enterprise-AI-Deployment.md) — Deployment infrastructure
- **Agent Development** → [13-Top-Demand/02-AI-Agent-Development.md](../13-Top-Demand/02-AI-Agent-Development.md) — Agent building fundamentals
- **MCP/ACP Protocols** → [13-Top-Demand/03-MCP-ACP-Protocols.md](../13-Top-Demand/03-MCP-ACP-Protocols.md) — Tool definition standards
- **Contribution Templates** → [15-Community-Resources-Templates/08-Contribution-Templates.md](../15-Community-Resources-Templates/08-Contribution-Templates.md) — Version contribution guidelines
- **SOUL/SKILL** → [15-Community-Resources-Templates/02-SOUL-SKILL-Templates.md](../15-Community-Resources-Templates/02-SOUL-SKILL-Templates.md) — Agent configuration standards

---

*Last updated: June 2026 | 400+ lines covering agent registry, versioning, deployment strategies, and A/B testing*

---
**See also:**
- [07 — AI Coding Agent Sustainability: Managing Hidden Costs, Code Quality Decay, and Developer Burnout](33-AI-Native-Software-Development/07-AI-Coding-Agent-Sustainability.md)
- [48 — MCP Cloud Infrastructure & Agent-as-a-Service: The Production Layer of Agentic AI](48-MCP-Cloud-Infrastructure-Agent-as-a-Service/01-Overview.md)
- [03 — Agent-Native Orchestration: LangGraph, Conductor, Mistral Workflows, Mcp-Agent](31-AI-Workflow-Orchestration-and-Durable-Execution/03-Agent-Native-Orchestration.md)
