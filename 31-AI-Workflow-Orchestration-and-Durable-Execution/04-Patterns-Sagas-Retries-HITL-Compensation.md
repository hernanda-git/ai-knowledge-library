# 04 — Patterns: Sagas, Retries, Human-in-Loop, Compensation, Idempotency

> Last updated: June 19, 2026

The difference between a workflow that works in a demo and a workflow that works in production is **patterns**. This document covers the 10 critical patterns every production AI workflow must implement correctly. Each pattern includes the problem, the solution, and code examples in Temporal (Python), Inngest (TypeScript), and LangGraph (Python) — the three most common stacks in 2026.

---

## 1. The Pattern Inventory

| # | Pattern | Solves | Used in |
|---|---------|--------|---------|
| 1 | **Saga with compensation** | Multi-step transactions that may fail mid-way | Payments, deployments, account creation |
| 2 | **Retry with backoff and jitter** | Transient failures (network, 429, 503) | Every LLM call, every API call |
| 3 | **Idempotency keys** | Duplicate side effects on retry | Every activity that mutates state |
| 4 | **Human-in-the-loop approval** | High-cost or irreversible actions | Send email, make payment, deploy code |
| 5 | **Timeouts at every layer** | Hangs that never resolve | Every activity, every workflow |
| 6 | **Deterministic replay** | Surviving crashes via state persistence | The core durable execution pattern |
| 7 | **Workflow versioning** | Updating in-flight workflows safely | Every long-running workflow in production |
| 8 | **Rate limiting per tenant** | Fair resource usage | Multi-tenant SaaS |
| 9 | **Cost budgets and circuit breakers** | Runaway LLM costs | Long-running agent workflows |
| 10 | **Distributed tracing** | Debugging multi-step workflows | Every production workflow |

---

## 2. Pattern 1 — Saga with Compensation

### 2.1 The problem

You have a 5-step transaction: reserve inventory → charge payment → update database → send email → create shipment. Step 3 fails. Now you have a half-completed transaction — inventory is reserved, payment is charged, but the order isn't in the database. You need to **roll back** steps 1 and 2.

### 2.2 The pattern

Every step has a **forward action** and a **compensating action**. If any step fails, the saga runs the compensations in reverse order.

### 2.3 Implementation in Temporal

```python
from datetime import timedelta
from temporalio import workflow, activity
from temporalio.common import RetryPolicy

# === Forward actions ===
@activity.defn
async def reserve_inventory(order_id: str) -> str:
    return await inventory.reserve(order_id)

@activity.defn
async def charge_payment(order_id: str, amount: float) -> str:
    return await payments.charge(order_id, amount)

@activity.defn
async def create_shipment(order_id: str) -> str:
    return await shipping.create(order_id)

# === Compensating actions ===
@activity.defn
async def release_inventory(reservation_id: str) -> None:
    await inventory.release(reservation_id)

@activity.defn
async def refund_payment(charge_id: str) -> None:
    await payments.refund(charge_id)

@activity.defn
async def cancel_shipment(shipment_id: str) -> None:
    await shipping.cancel(shipment_id)

# === Saga workflow ===
@workflow.defn
class OrderSaga:
    def __init__(self):
        self._compensations = []  # Stack of (function, input) to run in reverse on failure

    def _compensate(self, function, input):
        """Register a compensation to run if a later step fails."""
        self._compensations.append((function, input))

    @workflow.run
    async def run(self, order: dict) -> dict:
        try:
            # Step 1
            reservation_id = await workflow.execute_activity(
                reserve_inventory, order["id"],
                start_to_close_timeout=timedelta(seconds=30),
            )
            self._compensate(release_inventory, reservation_id)

            # Step 2
            charge_id = await workflow.execute_activity(
                charge_payment, order["id"], order["amount"],
                start_to_close_timeout=timedelta(seconds=30),
            )
            self._compensate(refund_payment, charge_id)

            # Step 3
            shipment_id = await workflow.execute_activity(
                create_shipment, order["id"],
                start_to_close_timeout=timedelta(seconds=30),
            )
            self._compensate(cancel_shipment, shipment_id)

            return {"status": "completed", "shipment_id": shipment_id}

        except Exception as e:
            # Run compensations in reverse order
            for compensation_fn, input in reversed(self._compensations):
                try:
                    await workflow.execute_activity(
                        compensation_fn, input,
                        start_to_close_timeout=timedelta(minutes=5),
                        # Compensations are best-effort — they may also fail
                        retry_policy=RetryPolicy(maximum_attempts=3),
                    )
                except Exception as comp_error:
                    # Log and continue — best-effort compensation
                    workflow.logger.error(
                        f"Compensation {compensation_fn.__name__} failed: {comp_error}"
                    )
            raise
```

### 2.4 Implementation in Inngest

```typescript
// inngest/functions/order-saga.ts
import { inngest } from "./client";

export const orderSaga = inngest.createFunction(
  { id: "order-saga" },
  { event: "order/created" },
  async ({ event, step }) => {
    // Inngest does NOT have built-in saga. Use try/catch with step.run
    let reservationId: string;
    let chargeId: string;
    let shipmentId: string;

    try {
      reservationId = await step.run("reserve-inventory", () =>
        inventory.reserve(event.data.orderId)
      );
    } catch (e) {
      throw new Error("Reservation failed — no compensation needed");
    }

    try {
      chargeId = await step.run("charge-payment", () =>
        payments.charge(event.data.orderId, event.data.amount)
      );
    } catch (e) {
      // Compensate step 1
      await step.run("release-inventory", () => inventory.release(reservationId));
      throw e;
    }

    try {
      shipmentId = await step.run("create-shipment", () =>
        shipping.create(event.data.orderId)
      );
    } catch (e) {
      // Compensate steps 2 and 1
      await step.run("refund-payment", () => payments.refund(chargeId));
      await step.run("release-inventory", () => inventory.release(reservationId));
      throw e;
    }

    return { status: "completed", shipmentId };
  }
);
```

### 2.5 Real-world saga gotchas

1. **Compensating actions may also fail.** Always retry, then log to a dead-letter queue.
2. **Compensating actions may be irreversible.** If you delete a record in step 1 and step 2 fails, you can't "undelete" — use soft delete + flag instead.
3. **Compensation order matters.** If step 2 took a payment refund, but step 1 was "send to fraud detection", the fraud detection flag should be cleared *first*.
4. **Idempotency is critical.** If the saga is retried after partial completion, compensations must be safe to run multiple times.

---

## 3. Pattern 2 — Retry with Backoff and Jitter

### 3.1 The problem

LLM APIs return 429 (rate limit) and 503 (overloaded) frequently. Naive retries can cause a **thundering herd** that makes things worse.

### 3.2 The pattern

Exponential backoff + jitter:

```
Attempt 1: immediate
Attempt 2: 1s + random(0-1000ms)
Attempt 3: 2s + random(0-1000ms)
Attempt 4: 4s + random(0-1000ms)
Attempt 5: 8s + random(0-1000ms)
```

### 3.3 Implementation

```python
# temporal — built-in
retry_policy = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    maximum_interval=timedelta(seconds=60),
    maximum_attempts=10,
    backoff_coefficient=2.0,
    non_retryable_error_types=["ValidationError", "AuthenticationError"],
)
```

```typescript
// inngest — built-in
inngest.createFunction(
  {
    id: "my-function",
    retries: 10,
  },
  { event: "..." },
  async ({ step }) => { ... }
);
```

### 3.4 Retry policy decisions

| Error type | Retry? | Why |
|------------|--------|-----|
| `429` (rate limit) | ✅ Yes | Transient |
| `503` (overloaded) | ✅ Yes | Transient |
| `500` (internal error) | ✅ Yes | Usually transient |
| `400` (bad request) | ❌ No | Won't get better on retry |
| `401` (auth failed) | ❌ No | Won't get better on retry |
| `ValidationError` | ❌ No | Your code is wrong |
| `ContextLengthError` | ❌ No | Trim your prompt first |
| `TimeoutError` | ✅ Yes | Maybe network issue |

---

## 4. Pattern 3 — Idempotency Keys

### 4.1 The problem

Activity runs, charges the customer's card, but the response is lost. The framework retries the activity. Customer is charged twice.

### 4.2 The pattern

Every activity that has a side effect must take an **idempotency key** — a unique ID derived from the workflow + step + input. The downstream system stores the result of each idempotency key and returns the same result on subsequent calls.

```python
@activity.defn
async def charge_payment(order_id: str, amount: float) -> str:
    # Generate idempotency key
    info = activity.info()
    idempotency_key = f"{info.workflow_run_id}:{info.activity_id}"

    # Pass to payment provider
    return await payments.charge(
        order_id=order_id,
        amount=amount,
        idempotency_key=idempotency_key,
    )
```

### 4.3 Idempotency key sources

| Source | Pros | Cons |
|--------|------|------|
| `workflow_run_id` | Globally unique | Same workflow + retry = same key ✅ |
| `workflow_run_id + activity_id` | Per-step uniqueness | Each retry = same key ✅ |
| `workflow_run_id + step_name + input_hash` | Same input = same key (deduplicates) | More complex |
| UUID generated per activity | Simple | Retries create new keys ❌ |

**Best practice:** Use `workflow_run_id + activity_id`. The activity framework guarantees the same activity_id on retry, so the key is stable.

### 4.4 Idempotency for the LLM itself

LLM calls are typically safe to retry (the API is idempotent in the sense that calling with the same prompt returns the same result most of the time), but token cost doubles. Two strategies:

1. **Cache LLM responses by prompt hash** — `step.run("llm-call", input=prompt)` with the framework's built-in memoization.
2. **Pre-compute the prompt hash as the idempotency key** — store `(prompt_hash → response)` in a cache.

---

## 5. Pattern 4 — Human-in-the-Loop Approval

### 5.1 The problem

The workflow wants to send an email charging a customer $500. This is a high-cost, irreversible action. The human must approve.

### 5.2 The pattern

```
Workflow reaches a "high-cost action" step
  ↓
Workflow sends a notification (email/Slack/dashboard) to the human
  ↓
Workflow SUSPENDS — does not occupy a worker
  ↓
Human reviews and clicks "approve" or "reject"
  ↓
Framework delivers a SIGNAL to the workflow
  ↓
Workflow resumes based on signal
```

### 5.3 Implementation in Temporal

```python
@workflow.defn
class PaymentWorkflow:
    def __init__(self):
        self._approved: bool = False
        self._approver: Optional[str] = None

    @workflow.run
    async def run(self, payment: dict) -> dict:
        # ... do some work ...

        # === Approval gate ===
        await workflow.execute_activity(
            request_approval,
            payment,
            start_to_close_timeout=timedelta(minutes=1),
        )
        # Suspends here — no worker occupied
        await workflow.wait_condition(
            lambda: self._approved is not None,  # True or False, not None
            timeout=timedelta(days=7),
        )

        if not self._approved:
            return {"status": "rejected", "approver": self._approver}

        # Proceed with the irreversible action
        return await workflow.execute_activity(
            execute_payment, payment,
            start_to_close_timeout=timedelta(minutes=5),
        )

    @workflow.signal
    async def approve(self, approver: str) -> None:
        self._approved = True
        self._approver = approver

    @workflow.signal
    async def reject(self, approver: str, reason: str) -> None:
        self._approved = False
        self._approver = approver
        self._rejection_reason = reason
```

### 5.4 Approval UX considerations

1. **Default timeout**: 7 days for most approvals. Shorter (1 day) for time-sensitive actions.
2. **Escalation**: If no response in 24h, notify the manager.
3. **Multi-approver**: Some actions require 2-of-2 approvers (use `wait_condition` with a counter).
4. **Delegation**: If approver is OOO, the system should re-route to a delegate.
5. **Audit log**: Every approval/rejection must be logged with timestamp, approver, reason.

### 5.5 When to require human approval

| Action | Approval required? |
|--------|-------------------|
| Read data | ❌ No |
| Send a Slack message | ❌ No |
| Search the web | ❌ No |
| Send a marketing email | ⚠️ Sometimes (configurable) |
| Create a draft document | ❌ No |
| Send a personal email | ✅ Yes |
| Charge a credit card | ✅ Yes |
| Refund > $100 | ✅ Yes |
| Refund > $10,000 | ✅ Yes (multi-approver) |
| Deploy to production | ✅ Yes |
| Delete a database record | ✅ Yes |
| Execute a stock trade | ✅ Yes (regulatory) |
| Modify someone's account | ✅ Yes |

---

## 6. Pattern 5 — Timeouts at Every Layer

### 6.1 The rule

> Every workflow has a timeout. Every activity has a timeout. Every external call within an activity has a timeout.

### 6.2 The three timeout types

| Timeout | Meaning | Use case |
|---------|---------|----------|
| `schedule_to_close_timeout` | Total time from scheduling to completion (including retries and waits) | Most common, upper bound |
| `start_to_close_timeout` | Time per attempt (per retry) | Single attempt limit |
| `schedule_to_start_timeout` | Time from scheduling to first execution | Detect dead workers |

```python
# Temporal — set all three
await workflow.execute_activity(
    my_activity, arg,
    schedule_to_close_timeout=timedelta(minutes=30),  # Total budget
    start_to_close_timeout=timedelta(minutes=5),       # Per attempt
    schedule_to_start_timeout=timedelta(seconds=30),   # Detect dead workers
)
```

### 6.3 Timeout calibration

- **LLM call**: `start_to_close=60s` for simple calls, `300s` for complex reasoning
- **Web search**: `start_to_close=10s`
- **Database write**: `start_to_close=5s`
- **Email send**: `start_to_close=30s`
- **Human approval**: `schedule_to_close=7 days`

### 6.4 The cost of missing timeouts

A missing timeout means a hung worker — the workflow is "stuck" indefinitely, occupying a worker slot, and the user has no idea. **Always set timeouts.**

---

## 7. Pattern 6 — Workflow Versioning

### 7.1 The problem

You have a workflow running 1000 instances. You want to deploy a code change that changes the workflow logic. But the 1000 in-flight workflows were started with the OLD logic. Replaying them with the NEW logic causes different activity calls → state corruption.

### 7.2 The pattern

Use `workflow.get_version` (Temporal) to branch on the version:

```python
@workflow.defn
class OrderWorkflow:
    @workflow.run
    async def run(self, order: dict) -> dict:
        version = workflow.get_version(
            "add-shipping-step",  # Change ID
            version.default,      # What in-flight workflows see
            version.min(2),       # What new workflows see
        )

        if version >= 2:
            # New code path
            await workflow.execute_activity(calculate_shipping, order, ...)
        else:
            # Old code path (for in-flight workflows started before the change)
            await workflow.execute_activity(simple_shipping, order, ...)
```

### 7.3 Versioning strategies

| Strategy | Use case |
|----------|----------|
| **`get_version` branches** | Conditional logic changes |
| **New workflow type, deprecate old** | Major rewrites |
| **Side-by-side with task-queue routing** | Multi-version deployment |

### 7.4 The "no new activities" rule for in-flight workflows

The cardinal rule: **Never change the sequence or type of activity calls in an existing workflow path**. If you do, replay will diverge from the event log and the workflow will fail.

If you must change the sequence, use `get_version` to branch.

---

## 8. Pattern 7 — Rate Limiting per Tenant

### 8.1 The problem

A multi-tenant SaaS has 100 customers. One customer (Acme) is running 10,000 LLM workflows simultaneously. They are starving the other 99 customers of capacity.

### 8.2 The pattern

Use **per-key concurrency limits** to cap the number of concurrent workflows per tenant:

```python
# Temporal
@workflow.defn
class AcmeWorkflow:
    pass

# In worker setup
worker = Worker(
    client,
    task_queue="default",
    workflows=[AcmeWorkflow],
    activities=[...],
)

# Per-tenant rate limit
rate_limiter = await client.create_schedule(
    Schedule(
        action=StartWorkflowAction(
            AcmeWorkflow,
            ...
        ),
        policies=[
            SchedulePolicy(
                overlap_policy=ScheduleOverlapPolicy.BUFFER_ONE,
            )
        ],
    ),
)
```

```typescript
// Inngest
inngest.createFunction(
  {
    id: "research-agent",
    concurrency: {
      limit: 10,             // Max 10 concurrent across all tenants
      key: "event.data.tenantId",  // Per-tenant (each tenant gets up to 10)
    },
  },
  { event: "research/requested" },
  async ({ event, step }) => { ... }
);
```

### 8.3 Rate limit dimensions

| Dimension | Example |
|-----------|---------|
| **Per-tenant** | Acme can run 10 concurrent |
| **Per-tier** | Free tier = 1 concurrent, Pro = 10, Enterprise = 100 |
| **Per-user** | Each user can run 3 concurrent |
| **Global** | Total across all tenants = 1000 |
| **Per-LLM-provider** | OpenAI: 500 concurrent, Anthropic: 200 |

---

## 9. Pattern 8 — Cost Budgets and Circuit Breakers

### 9.1 The problem

A long-running agent workflow keeps calling the LLM, getting suboptimal outputs, retrying with more prompts, and the cost spirals from $0.50 to $50. The workflow has no concept of "I've spent too much, stop".

### 9.2 The pattern

Track per-workflow cost in workflow state. Set a budget. Open the circuit breaker when budget exceeded.

```python
@workflow.defn
class BudgetedAgentWorkflow:
    def __init__(self):
        self._cost_usd = 0.0
        self._budget_usd = 5.0
        self._circuit_open = False

    @workflow.run
    async def run(self, task: str) -> dict:
        result = await self._do_step(task)
        return result

    async def _do_step(self, task: str) -> dict:
        # Check budget BEFORE calling LLM
        if self._cost_usd >= self._budget_usd:
            raise workflow.ApplicationError(
                f"Budget exceeded: ${self._cost_usd:.2f} / ${self._budget_usd:.2f}"
            )

        # Call LLM
        response = await workflow.execute_activity(
            call_llm, task,
            start_to_close_timeout=timedelta(minutes=2),
        )

        # Track cost (in production, parse from response.usage)
        self._cost_usd += response["usage"]["total_tokens"] * 0.00003

        return response
```

### 9.3 Circuit breaker vs budget

| Mechanism | Scope | Trigger | Action |
|-----------|-------|---------|--------|
| **Budget** | Per-workflow | Cost > limit | Stop workflow |
| **Circuit breaker** | Per-dependency | Failure rate > threshold | Stop calling the dependency |
| **Rate limit** | Per-tenant | Concurrent count > limit | Queue new workflows |
| **Quota** | Per-tenant-per-period | API calls > limit | Reject new requests |

---

## 10. Pattern 9 — Distributed Tracing

### 10.1 The problem

A workflow has 50 activities. The user reports "workflow #abc-123 failed at 3am". You need to find the failing activity and the input that caused it.

### 10.2 The pattern

Every workflow execution emits **distributed traces** that link:
- The workflow start event
- Every activity call
- Every LLM API request
- Every external service call

The trace is **queryable** by workflow ID and **linked to logs and metrics**.

### 10.3 Implementation

All major frameworks (Temporal, Inngest, Restate) emit OpenTelemetry-compatible traces out of the box:

```python
# Temporal — OpenTelemetry
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="http://otel-collector:4317"))
)

# Every workflow execution is now a trace
# Every activity is a span within the trace
```

```typescript
// Inngest — built-in trace export
new Inngest({
  id: "my-app",
  observability: {
    traces: {
      exporter: { type: "otel", endpoint: "http://otel-collector:4317" },
    },
  },
});
```

### 10.4 The 4 golden signals for workflows

| Signal | Metric |
|--------|--------|
| **Latency** | P50/P95/P99 of workflow duration, activity duration |
| **Traffic** | Workflows started/min, activities executed/min |
| **Errors** | Workflows failed/min by error type, activities retried/min |
| **Saturation** | Worker utilization, queue depth, event log size |

---

## 11. Pattern 10 — Heartbeating for Long Activities

### 11.1 The problem

You have a long activity — "scrape 10,000 web pages" — that takes 30 minutes. If the worker dies at minute 25, the framework has no idea. It waits the full 30 minutes, then retries, repeating 25 minutes of work.

### 11.2 The pattern

Activities **heartbeat** back to the framework every 30 seconds. The framework tracks heartbeats. If heartbeats stop, the framework knows the worker is dead and can fail-fast the activity, then retry it on a new worker — with a checkpoint indicating how far the activity got.

```python
@activity.defn
async def scrape_pages(urls: list[str]) -> dict:
    results = {}
    info = activity.info()
    heartbeat_details = {"completed": 0}

    for i, url in enumerate(urls):
        # If we have a heartbeat detail, skip already-completed URLs
        if "completed" in info.heartbeat_details:
            i = info.heartbeat_details["completed"]
            results = {url: ...for url in urls[:i]}  # Replay path

        # Do the work
        results[url] = await scrape_one(url)

        # Heartbeat every URL
        heartbeat_details["completed"] = i + 1
        activity.heartbeat(heartbeat_details)

    return results
```

### 11.3 Heartbeat vs checkpoint

- **Heartbeat**: "I'm still alive, here's my progress" — frequent, lightweight
- **Checkpoint**: "Here's my full state, persist it" — heavier, less frequent

Use heartbeats for progress, checkpoints for state.

---

## 12. The Anti-Patterns (What NOT to do)

### 12.1 "I'll just use a try/except and a database"

```python
# ❌ ANTI-PATTERN
async def run(self, workflow_id):
    state = await db.load(workflow_id)
    state.step += 1
    await db.save(workflow_id, state)  # ← Race condition
    result = await call_api(...)
    state.result = result
    await db.save(workflow_id, state)
```

You will re-invent durable execution badly. Use a framework.

### 12.2 "I'll put the LLM call inside the workflow function"

```python
# ❌ ANTI-PATTERN
@workflow.defn
class MyWorkflow:
    @workflow.run
    async def run(self, x):
        # This is in the workflow function, NOT an activity
        # - Not retried
        # - Not observable
        # - Slow LLM call holds up the workflow
        result = await openai_client.chat(...)  # ← BAD
        return result
```

The workflow function must be deterministic. LLM calls must be in activities.

### 12.3 "I'll skip the human approval — it's slow"

```python
# ❌ ANTI-PATTERN — for high-cost actions
@workflow.run
async def run(self, payment):
    # No approval! Just do it!
    return await workflow.execute_activity(charge_card, payment, ...)
```

The cost of a single bad autonomous action > the cost of every approval UX improvement you'll make.

### 12.4 "I'll set timeouts to 1 hour to be safe"

```python
# ❌ ANTI-PATTERN
start_to_close_timeout=timedelta(hours=1)  # Way too long
```

When something goes wrong, you want to know FAST. Set tight timeouts. If the activity genuinely needs 1 hour, break it into smaller chunks with heartbeats.

### 12.5 "I'll use asyncio.gather and Promise.allSettled for everything"

```python
# ❌ ANTI-PATTERN for anything that may fail or run for > 30s
results = await asyncio.gather(*tasks)
```

`gather` is a parallelism primitive. It's not durable, not observable, not retriable, not signalable. Use a workflow framework.

---

## 13. Cross-References

- `01-Overview-and-Durable-Execution-Primitives.md` — the 5 primitives these patterns use
- `02-Frameworks-Temporal-Inngest-Restate-Prefect.md` — framework-specific implementations
- `03-Agent-Native-Orchestration.md` — agent-specific patterns
- `13-Top-Demand/13-Human-in-the-Loop-Systems.md` — HITL deep-dive
- `20-Agent-Infrastructure-and-Observability/03-Agent-Tracing-and-Observability.md` — observability patterns
- `20-Agent-Infrastructure-and-Observability/07-Agent-Reliability-and-Resilience.md` — reliability patterns
- `28-AI-Agent-Commerce-and-A2A-Payments/` — A2A calls are activities that need these patterns

---

## 14. Resources

- [Temporal: Saga pattern](https://temporal.io/blog/temporal-saga-pattern)
- [Temporal: Versioning workflows](https://docs.temporal.io/workflows#versioning)
- [Microsoft: Distributed Workflow Patterns](https://learn.microsoft.com/en-us/azure/azure-functions/durable/durable-functions-concepts)
- [AWS: Saga pattern (Microservices)](https://docs.aws.amazon.com/prescriptive-guidance/latest/cloud-design-patterns/saga-choreography.html)
- [Google SRE Book: Eliminating Toil](https://sre.google/sre-book/eliminating-toil/)
- [Inngest: Flow control](https://www.inngest.com/docs/guides/flow-control)

---

*This document is part of the AI Knowledge Library — 31-AI-Workflow-Orchestration-and-Durable-Execution directory. Generated by Auto-Enricher cycle 2026-06-19.*
