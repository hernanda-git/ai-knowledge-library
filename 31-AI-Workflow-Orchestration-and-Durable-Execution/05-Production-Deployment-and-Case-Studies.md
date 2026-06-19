# 05 — Production Deployment & Case Studies

> Last updated: June 19, 2026

This document covers the **operational reality** of running AI workflows in production: deployment topologies, observability integration, migration patterns, common pitfalls, and real case studies from teams that ship at scale. If you're about to put a workflow engine in front of paying customers, read this first.

---

## 1. Deployment Topologies

### 1.1 Self-host Temporal (Kubernetes)

```
┌─────────────────────────────────────────────────────────────┐
│                       Kubernetes Cluster                      │
│                                                                │
│   ┌──────────────────────────────────────────────────┐       │
│   │  Temporal Namespace: "production"                │       │
│   │                                                    │       │
│   │   ┌────────────┐  ┌────────────┐  ┌────────────┐ │       │
│   │   │ Frontend   │  │ History    │  │ Matching   │ │       │
│   │   │ (3 pods)   │  │ (3 pods)   │  │ (3 pods)   │ │       │
│   │   └────────────┘  └────────────┘  └────────────┘ │       │
│   │           ↓              ↓              ↓         │       │
│   │   ┌──────────────────────────────────────┐       │       │
│   │   │  Cassandra (3-node, RF=3)            │       │       │
│   │   │  OR Postgres (HA via Patroni)        │       │       │
│   │   └──────────────────────────────────────┘       │       │
│   └──────────────────────────────────────────────────┘       │
│                                                                │
│   ┌──────────────────────────────────────────────────┐       │
│   │  Your Workers (Deployment)                         │       │
│   │   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐  │       │
│   │   │Worker 1│ │Worker 2│ │Worker 3│ │Worker N│  │       │
│   │   │Python  │ │Python  │ │TypeScr │ │Go      │  │       │
│   │   └────────┘ └────────┘ └────────┘ └────────┘  │       │
│   └──────────────────────────────────────────────────┘       │
│                                                                │
│   ┌──────────────────────────────────────────────────┐       │
│   │  Observability                                     │       │
│   │   ┌──────────┐ ┌──────────┐ ┌──────────┐         │       │
│   │   │ OTel     │ │Prometheus│ │ Grafana  │         │       │
│   │   │Collector │ │          │ │          │         │       │
│   │   └──────────┘ └──────────┘ └──────────┘         │       │
│   └──────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

**Resource requirements (typical mid-scale):**
- 3 Frontend pods, 0.5 CPU / 1 GB RAM each
- 3 History pods, 2 CPU / 4 GB RAM each
- 3 Matching pods, 1 CPU / 2 GB RAM each
- 3-node Cassandra cluster, 4 CPU / 16 GB RAM each
- 3+ Worker pods, depends on workload
- Storage: ~100 GB / month for 10M events

**Pros:** Full control, no vendor lock-in, on-prem compliance
**Cons:** Significant operational overhead, Cassandra expertise needed

### 1.2 Temporal Cloud (managed)

- 3-region HA by default
- Automatic upgrades
- SOC 2 Type II, HIPAA-eligible, GDPR-compliant
- Pricing: $25/mo per namespace + $0.025 per 1K workflow transitions + storage

**Pros:** Zero ops, fastest path to production
**Cons:** Expensive at scale, vendor lock-in

### 1.3 Inngest Cloud (serverless)

```typescript
// No infrastructure to manage. Inngest runs on its own infra,
// your functions run on Vercel/Cloudflare/Lambda
import { Inngest } from "inngest";

export const inngest = new Inngest({ id: "my-app" });
```

**Pros:** Free tier generous, zero ops, serverless-native
**Cons:** Vendor dependency, less control over data residency

### 1.4 The hybrid: Temporal Cloud + self-hosted workers

```
Temporal Cloud (managed)
    ↓ gRPC
Your Workers (self-hosted, in your VPC)
    ↓ HTTP
Your Services (databases, APIs, LLM providers)
```

**Pros:** No Temporal server ops, but your code and data stay in your VPC
**Cons:** Still need to run workers

---

## 2. Observability Stack

### 2.1 The three pillars for workflows

| Pillar | Tools | What it shows |
|--------|-------|---------------|
| **Logs** | Loki, ELK, CloudWatch | Per-activity text logs |
| **Metrics** | Prometheus, Datadog | Workflow counts, latencies, error rates |
| **Traces** | Jaeger, Tempo, Honeycomb | Per-workflow call graph |

### 2.2 Key dashboards

A production workflow needs at minimum these dashboards:

**Workflow health:**
- Workflows started/min (by namespace, by workflow type)
- Workflows completed/min (by status: success/failed/cancelled/timeout)
- P50/P95/P99 workflow duration (by type)
- Workflows in-flight (current count)

**Activity health:**
- Activities executed/min (by type, by status)
- Activity P50/P95/P99 duration
- Activity retry rate
- Activity error rate (by error type)

**Worker health:**
- Worker count (active)
- Worker CPU/memory utilization
- Task queue depth
- Worker poll-to-handle latency

**Cost:**
- LLM tokens/min (by model)
- LLM cost/min (USD)
- LLM cost per workflow (P50/P95)
- LLM cost by user/tenant

### 2.3 Alerting rules

| Alert | Condition | Severity |
|-------|-----------|----------|
| Workflow failure spike | Failed > 5% of last 100 | Page |
| Workflow backlog | Queue depth > 1000 for 5 min | Page |
| Worker crash loop | Worker restarts > 3 in 5 min | Page |
| LLM cost spike | Cost > 2× baseline for 15 min | Slack |
| Stuck workflow | Workflow running > 24h | Slack |
| LLM API errors | Error rate > 10% for 5 min | Page |

### 2.4 The Temporal UI / Inngest dashboard

Built-in UI shows:
- Workflow search (by ID, type, status, time range)
- Workflow timeline (every step, input, output, duration)
- Workflow history (state transitions)
- Stack trace on failure

This is the single most useful debugging tool. **Always have it open during incidents.**

---

## 3. Migration Patterns

### 3.1 Naive Python script → Temporal

**Before:**
```python
# jobs/send_welcome_emails.py — runs every hour via cron
async def send_pending_welcome_emails():
    pending = await db.get_pending_welcomes()
    for record in pending:
        try:
            await send_email(record.email)
            await db.mark_sent(record.id)
        except Exception as e:
            logger.error(f"Failed for {record.id}: {e}")
```

**After:**
```python
@workflow.defn
class SendWelcomeEmailWorkflow:
    @workflow.run
    async def run(self, record_id: str) -> None:
        record = await workflow.execute_activity(
            load_record, record_id,
            start_to_close_timeout=timedelta(seconds=10),
        )
        await workflow.execute_activity(
            send_email, record,
            start_to_close_timeout=timedelta(seconds=30),
        )
        await workflow.execute_activity(
            mark_sent, record_id,
            start_to_close_timeout=timedelta(seconds=10),
        )

# Triggered by a Temporal Schedule (replaces cron)
```

**Migration steps:**
1. Wrap each step in an activity
2. Replace try/except with Temporal's retry policy
3. Replace `mark_sent` with an idempotent activity
4. Replace cron with Temporal Schedule
5. Add observability dashboards

### 3.2 LangChain Agent → LangGraph + Temporal

**Before:**
```python
# naive agent.py
def run_agent(question: str) -> str:
    llm = ChatOpenAI()
    tools = [search, calculator]
    agent = create_openai_functions_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools).invoke({"input": question})
```

**After:**
```python
# langgraph_agent.py
class State(TypedDict):
    messages: Annotated[list, add_messages]
graph = StateGraph(State)
graph.add_node("agent", agent_fn)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
graph.add_edge("tools", "agent")
app = graph.compile(checkpointer=PostgresSaver(...))

# temporal_langgraph.py
@workflow.defn
class DurableAgentWorkflow:
    @workflow.run
    async def run(self, request: dict) -> dict:
        config = {"configurable": {"thread_id": request["user_id"]}}
        return await workflow.execute_activity(
            run_langgraph, request, config,
            start_to_close_timeout=timedelta(minutes=30),
        )
```

**Migration steps:**
1. Convert to LangGraph state machine
2. Add Postgres checkpointer
3. Wrap LangGraph call in a Temporal activity
4. Add approval gates for high-cost actions
5. Add cost tracking
6. Add observability

### 3.3 Celery → Temporal

**Common migration challenges:**

| Celery feature | Temporal equivalent |
|----------------|---------------------|
| `@shared_task` | `@activity.defn` |
| `task.delay(x)` | `client.start_workflow(Workflow.run, x, ...)` |
| Celery Beat (cron) | Temporal Schedule |
| `apply_async(eta=...)` | `client.start_workflow(..., start_delay=...)` |
| `chain(task1, task2)` | Sequential `workflow.execute_activity` calls |
| `chord([t1, t2, t3], callback)` | `asyncio.gather` of activities, then callback activity |
| `canvas.group(...)` | `asyncio.gather` of activities |
| `task_failure_callback` | Workflow try/except |
| Celery result backend | Temporal query API |
| `revoke(task_id)` | `handle.cancel()` |

**Key insight:** Celery is a task queue, Temporal is a workflow engine. The mental model shift is from "dispatch a task" to "start a workflow that orchestrates activities".

---

## 4. Case Studies

### 4.1 Case Study 1: AI-powered code migration at a Fortune 500

**Problem:** Migrate 10,000 Python 2 services to Python 3. Each service takes 5-20 hours of agent work (read codebase, plan migration, generate diff, run tests, iterate).

**Architecture:**
```
Developer triggers "migrate service X"
    ↓
Temporal Workflow: ServiceMigrationWorkflow
    ↓
  Step 1: LLM analyzes codebase (1-2 hours, $5-20 in LLM cost)
    ↓
  Step 2: LLM generates migration plan (30 min, $2-5)
    ↓
  Step 3: [Human approval] (1-7 days)
    ↓
  Step 4: LLM applies migration (3-10 hours, $20-100)
    ↓
  Step 5: Run tests in CI (1-2 hours)
    ↓
  Step 6: [Human approval] (1-3 days)
    ↓
  Step 7: Create PR (1 min)
    ↓
  Done
```

**Results:**
- 50 services migrated in first 3 months
- Average human time per service: 2 hours (vs 40 hours manual)
- $0 success rate: 92% (12% needed manual fix-up)
- Total cost: $200/service average

**Key patterns used:**
- Saga with compensation (rollback if tests fail)
- Human-in-loop approval at high-cost steps
- Cost budget per workflow ($200 limit)
- Heartbeating during long LLM calls
- Parallel test execution with `asyncio.gather`

### 4.2 Case Study 2: Customer support agent at a SaaS company

**Problem:** 70% of support tickets can be answered by an LLM, but 30% require human handoff. The 30% handoff was a black box — agents couldn't see what the AI had done, leading to repeated questions.

**Architecture:**
```
Customer submits ticket
    ↓
Temporal Workflow: SupportTicketWorkflow
    ↓
  Step 1: LLM classifies (1-3 sec, $0.01)
    ↓
  Step 2a: If simple → LLM drafts response
  Step 2b: If complex → [human agent picks up]
    ↓
  Step 3: [Customer approval of draft OR human agent reply]
    ↓
  Step 4: Send response, close ticket
    ↓
  Step 5: [Optional] Follow-up survey after 24h
```

**Key patterns used:**
- Workflow versioning (3 versions in 6 months as LLM models improved)
- Per-tenant rate limiting (Enterprise tier: 100 concurrent, Free: 1)
- Idempotency keys on all "send email" activities
- 24h sleep with `step.sleep` (no worker occupied)
- Timeline view in customer support UI (so human agents can see what AI did)

**Results:**
- 65% of tickets fully automated
- 30% partial automation (AI drafted, human sent)
- 5% fully human
- Customer satisfaction unchanged (was 4.2/5, still 4.2/5)
- Cost per ticket: $0.04 (down from $3.50 for fully-human)

### 4.3 Case Study 3: A2A payment processing at an agent marketplace

**Problem:** Agents on the marketplace need to pay other agents for services. Each transaction is a multi-step workflow: verify identity → check balance → charge → confirm → record.

**Architecture:** (See [28-AI-Agent-Commerce-and-A2A-Payments](../28-AI-Agent-Commerce-and-A2A-Payments/) for full A2A context)
```
Agent A wants to call Agent B's API
    ↓
Temporal Workflow: A2APaymentWorkflow
    ↓
  Step 1: Verify Agent A identity (8004 lookup)
  Step 2: Check Agent A balance (USDC wallet)
  Step 3: [X402 payment authorization]
  Step 4: Make API call to Agent B
  Step 5: Charge Agent A (X402 settlement)
  Step 6: Record transaction (on-chain + internal)
  Step 7: [Dispute window] — 7 days, signals only
    ↓
  Done (or disputed → separate workflow)
```

**Key patterns used:**
- Saga with compensation (refund on API failure)
- Heartbeating during the 7-day dispute window
- X402 payment authorization as an activity (idempotent)
- Per-agent rate limiting (to prevent spam)
- Idempotency keys tied to `(workflow_run_id, agent_a, agent_b, request_hash)`

---

## 5. Common Pitfalls

### 5.1 "I'll just put the LLM call in the workflow function"

**Symptom:** Workflows are slow, unrecoverable, and un-observable.

**Fix:** Always wrap LLM calls in activities. Always.

### 5.2 "I'll use a single global task queue"

**Symptom:** One slow customer blocks all other customers.

**Fix:** Use per-tenant task queues, or per-tenant rate limits, or both.

### 5.3 "I'll skip versioning, my workflows are short"

**Symptom:** Code change breaks 1000 in-flight workflows.

**Fix:** Always use `workflow.get_version` (or equivalent) for any change to the workflow's call sequence. Even "short" workflows can be in-flight during a deploy.

### 5.4 "I'll set retries to 100 to be safe"

**Symptom:** Failed workflow retries forever, accumulating LLM cost.

**Fix:** Set explicit `maximum_attempts`. Set explicit `maximum_interval`. Have a dead-letter queue.

### 5.5 "I don't need observability, I have logs"

**Symptom:** "Workflow #abc-123 failed" → 6 hours of grepping logs.

**Fix:** Use the built-in UI (Temporal UI, Inngest dashboard) + OpenTelemetry traces. Logs are the lowest-value observability signal for workflows.

### 5.6 "I'll test in production"

**Symptom:** First workflow corrupts 10,000 customer records.

**Fix:** Use the framework's test environment. Temporal has `temporalio.testing.WorkflowEnvironment` with time-skipping. Inngest has `inngest/test`. LangGraph has `MemorySaver`. **Always** test in a deterministic, replayable environment.

### 5.7 "I'll deploy workers with my application code"

**Symptom:** Workers restart every time you deploy the app, losing in-flight workflows.

**Fix:** Run workers as a separate deployment with independent lifecycle. Workers should be long-lived.

### 5.8 "I'll skip the human approval for speed"

**Symptom:** First bad autonomous action costs $50K in refunds.

**Fix:** Default to approval for any high-cost or irreversible action. Make approval fast (Slack button, one-click).

---

## 6. Future Outlook (2026-2028)

### 6.1 The trend toward managed services

- **Mistral Workflows** (Apr 2026) signals: frontier model vendors will ship workflow engines
- **OpenAI Workflows**, **Anthropic Workflows** likely by 2027
- The future is: "my model provider also runs my workflow engine"

### 6.2 The trend toward AI-native primitives

- Built-in LLM rate-limit handling (already in Inngest)
- Built-in token cost tracking (already in Mistral Workflows)
- Built-in hallucination detection (coming)
- Built-in approval UI (already in Mistral Workflows)

### 6.3 The trend toward edge/serverless

- Restate, Inngest: durable execution at the edge
- Cloudflare Workflows (Durable Objects-based) emerging
- The "serverless" + "durable" combo is the future

### 6.4 The unsolved problems

1. **Cross-framework portability** — no standard "workflow definition language"
2. **Multi-region active-active** — most frameworks are single-region
3. **Workflow search** — finding a specific workflow instance is still painful
4. **Cost attribution** — knowing which team/feature drove LLM cost
5. **Semantic observability** — "what was the agent thinking?" is still answered by reading traces, not by a query

---

## 7. The Builder's Checklist

Before shipping a workflow to production:

### Technical
- [ ] All LLM calls wrapped in activities (not in workflow functions)
- [ ] Timeouts set on every activity (start_to_close, schedule_to_close)
- [ ] Retry policies defined per activity (with non_retryable_error_types)
- [ ] Idempotency keys on all mutating activities
- [ ] Heartbeats on long activities
- [ ] Workflow versioning in place
- [ ] Tests in framework's test environment (deterministic, replayable)
- [ ] OpenTelemetry traces exported

### Operational
- [ ] Dashboards for workflow / activity / worker health
- [ ] Alerts for failure rate, queue depth, worker health, cost spike
- [ ] Runbooks for the top 5 failure modes
- [ ] Workers deployed as separate long-lived service
- [ ] Per-tenant rate limits configured
- [ ] Cost budgets per workflow type

### Business
- [ ] Human approval for high-cost or irreversible actions
- [ ] Audit log of all approvals
- [ ] SLA defined (e.g., 95% of workflows complete in < 5 min)
- [ ] Cost per workflow tracked
- [ ] Failure modes documented for customer support

---

## 8. 12-Month Reading List

- *Designing Data-Intensive Applications* (Kleppmann) — Ch. 8 on stream processing, Ch. 11 on serverless
- *Site Reliability Engineering* (Google) — Ch. 5 on eliminating toil
- *Cloud Design Patterns* (Microsoft) — Saga, CQRS, Event Sourcing
- *Workflow Patterns* (van der Aalst) — the academic reference
- Temporal blog: saga pattern, versioning, testing
- Inngest blog: flow control, AI workflows
- Mistral Workflows documentation (when published)

---

## 9. Cross-References

- `01-Overview-and-Durable-Execution-Primitives.md` — the foundations
- `02-Frameworks-Temporal-Inngest-Restate-Prefect.md` — framework deep-dive
- `03-Agent-Native-Orchestration.md` — agent-specific tools
- `04-Patterns-Sagas-Retries-HITL-Compensation.md` — patterns
- `20-Agent-Infrastructure-and-Observability/03-Agent-Tracing-and-Observability.md` — observability
- `20-Agent-Infrastructure-and-Observability/07-Agent-Reliability-and-Resilience.md` — reliability
- `28-AI-Agent-Commerce-and-A2A-Payments/` — A2A is a use case for durable execution

---

## 10. Resources

- [Temporal Production Checklist](https://docs.temporal.io/production-deployment)
- [Inngest Production Guide](https://www.inngest.com/docs/learn/serving-in-production)
- [Temporal: Migrating from Celery](https://temporal.io/blog/migrating-from-celery-to-temporal)
- [OpenTelemetry for Workflows](https://opentelemetry.io/docs/concepts/instrumentation/)
- [Case studies: Snap, Netflix, Datadog on Temporal](https://temporal.io/case-studies)
- [Case studies: Gitbook, Soundcloud on Inngest](https://www.inngest.com/customers)

---

*This document is part of the AI Knowledge Library — 31-AI-Workflow-Orchestration-and-Durable-Execution directory. Generated by Auto-Enricher cycle 2026-06-19.*
