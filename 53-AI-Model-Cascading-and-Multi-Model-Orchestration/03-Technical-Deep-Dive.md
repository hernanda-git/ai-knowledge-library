# AI Model Cascading and Multi-Model Orchestration: Technical Deep Dive

> **Description:** Advanced technical implementations, algorithms, and production patterns for building robust multi-model orchestration systems. This covers sophisticated routing algorithms, distributed execution, monitoring infrastructure, and advanced optimization techniques.

---

## Table of Contents

1. [Advanced Routing Algorithms](#1-advanced-routing-algorithms)
2. [Distributed Multi-Model Systems](#2-distributed-multi-model-systems)
3. [Production Monitoring Infrastructure](#3-production-monitoring-infrastructure)
4. [Advanced Optimization Techniques](#4-advanced-optimization-techniques)
5. [Error Handling and Recovery](#5-error-handling-and-recovery)
6. [Performance Profiling](#6-performance-profiling)
7. [Advanced Cascading Patterns](#7-advanced-cascading-patterns)
8. [Model Composition Patterns](#8-model-composition-patterns)
9. [Production Deployment Patterns](#9-production-deployment-patterns)
10. [Cross-References](#10-cross-references)

---

## 1. Advanced Routing Algorithms

### 1.1 Contextual Bandit Routing

Use contextual bandits for adaptive model selection:

```python
class ContextualBanditRouter:
    def __init__(self, models, context_dim, alpha=0.1):
        self.models = models
        self.context_dim = context_dim
        self.alpha = alpha
        
        # Linear model for each arm (model)
        self.weights = {m.name: np.zeros(context_dim) for m in models}
        self.biases = {m.name: 0.0 for m in models}
        
        # Thompson sampling parameters
        self.A = {m.name: np.eye(context_dim) for m in models}
        self.b = {m.name: np.zeros(context_dim) for m in models}
    
    def select_model(self, context):
        """Select model using Thompson sampling."""
        sampled_rewards = {}
        
        for model_name in self.models:
            # Sample from posterior
            A_inv = np.linalg.inv(self.A[model_name])
            mu = A_inv @ self.b[model_name]
            
            # Add exploration noise
            theta = np.random.multivariate_normal(mu, self.alpha**2 * A_inv)
            
            # Estimate reward
            sampled_rewards[model_name] = context @ theta
        
        # Select model with highest sampled reward
        return max(sampled_rewards, key=sampled_rewards.get)
    
    def update(self, model_name, context, reward):
        """Update model statistics after observation."""
        # Update linear model
        self.A[model_name] += np.outer(context, context)
        self.b[model_name] += reward * context
    
    def extract_context(self, input_data):
        """Extract context features from input."""
        return np.array([
            len(input_data),  # Input length
            self.count_words(input_data),  # Word count
            self.has_code(input_data),  # Code presence
            self.complexity_score(input_data),  # Complexity
            self.domain_score(input_data),  # Domain relevance
            self.sentiment_score(input_data),  # Sentiment
            self.noun_phrase_count(input_data),  # Noun phrases
            self.named_entity_count(input_data),  # Named entities
        ])
```

### 1.2 Bayesian Optimization for Hyperparameter Tuning

Optimize routing hyperparameters:

```python
class BayesianHyperparameterOptimizer:
    def __init__(self, router, param_space, n_iterations=50):
        self.router = router
        self.param_space = param_space
        self.n_iterations = n_iterations
        self.X_observed = []
        self.y_observed = []
    
    def optimize(self, validation_data):
        """Optimize router hyperparameters using Bayesian optimization."""
        from skopt import gp_minimize
        from skopt.space import Real, Integer, Categorical
        
        def objective(params):
            # Set router parameters
            self.router.set_params(params)
            
            # Evaluate on validation data
            total_reward = 0
            for input_data, expected in validation_data:
                model = self.router.route(input_data)
                output = model.predict(input_data)
                reward = self.compute_reward(output, expected)
                total_reward += reward
            
            return -total_reward  # Minimize negative reward
        
        # Define search space
        space = self.define_search_space()
        
        # Run optimization
        result = gp_minimize(
            objective, 
            space, 
            n_calls=self.n_iterations,
            random_state=42
        )
        
        return result.x, -result.fun
    
    def define_search_space(self):
        """Define Bayesian optimization search space."""
        from skopt.space import Real, Integer, Categorical
        
        return [
            Real(0.1, 0.9, name='exploration_rate'),
            Real(0.5, 0.99, name='confidence_threshold'),
            Integer(1, 10, name='cascade_depth'),
            Real(0.01, 0.1, name='learning_rate'),
            Categorical(['ucb', 'thompson', 'epsilon_greedy'], name='exploration_strategy')
        ]
```

### 1.3 Reinforcement Learning for Dynamic Routing

Learn routing policy through interaction:

```python
class DeepRLRouter:
    def __init__(self, models, state_dim, action_dim):
        self.models = models
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # DQN networks
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Experience replay
        self.memory = ReplayBuffer(capacity=100000)
        self.batch_size = 64
        
        # Exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
    
    def select_action(self, state):
        """Select action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            q_values = self.policy_net(state)
            return q_values.argmax().item()
    
    def train(self):
        """Train DQN on experience replay."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = batch
        
        # Compute Q-values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_net(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Update policy network
        loss = F.smooth_l1_loss(current_q, target_q.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def compute_reward(self, output, expected):
        """Compute reward based on output quality."""
        quality_score = self.compute_quality(output, expected)
        cost_score = 1 - (output.cost / 0.1)
        latency_score = 1 - (output.latency / 1000)
        
        return 0.5 * quality_score + 0.3 * cost_score + 0.2 * latency_score
```

### 1.4 Graph Neural Network for Model Selection

Use GNN to model model relationships:

```python
class GNNModelSelector:
    def __init__(self, models, model_embeddings):
        self.models = models
        self.embeddings = model_embeddings
        
        # Build model similarity graph
        self.graph = self.build_model_graph()
        
        # GNN for model selection
        self.gnn = GraphAttentionNetwork(
            in_channels=128,
            out_channels=64,
            heads=4
        )
    
    def build_model_graph(self):
        """Build graph of model similarities."""
        import networkx as nx
        
        G = nx.Graph()
        
        # Add nodes (models)
        for model in self.models:
            G.add_node(model.name, embedding=self.embeddings[model.name])
        
        # Add edges (similarities)
        for i, m1 in enumerate(self.models):
            for j, m2 in enumerate(self.models):
                if i != j:
                    similarity = self.compute_similarity(m1, m2)
                    if similarity > 0.7:
                        G.add_edge(m1.name, m2.name, weight=similarity)
        
        return G
    
    def select_model(self, task_embedding):
        """Select model using GNN-based scoring."""
        # Get model embeddings
        model_embeddings = torch.tensor([
            self.embeddings[m.name] for m in self.models
        ])
        
        # Add task as temporary node
        task_node = torch.tensor(task_embedding).unsqueeze(0)
        
        # Run GNN
        node_embeddings = self.gnn(model_embeddings, self.graph)
        
        # Score each model
        scores = []
        for i, model in enumerate(self.models):
            score = F.cosine_similarity(
                task_embedding, 
                node_embeddings[i], 
                dim=0
            )
            scores.append(score)
        
        # Select model with highest score
        return self.models[scores.index(max(scores))]
```

---

## 2. Distributed Multi-Model Systems

### 2.1 Distributed Model Serving

Deploy models across multiple servers:

```python
class DistributedModelServer:
    def __init__(self, model_configs, load_balancer):
        self.models = {}
        self.load_balancer = load_balancer
        self.health_checker = HealthChecker()
        
        # Initialize model servers
        for config in model_configs:
            self.deploy_model(config)
    
    def deploy_model(self, config):
        """Deploy model to available server."""
        server = self.allocate_server(config)
        
        if server:
            model = ModelServer(config, server)
            self.models[config.name] = model
            
            # Register with load balancer
            self.load_balancer.register(model)
            
            # Start health checking
            self.health_checker.register(model)
    
    def predict(self, model_name, input_data):
        """Route prediction request to appropriate server."""
        # Get available instances
        instances = self.load_balancer.get_instances(model_name)
        
        if not instances:
            raise ModelUnavailableError(f"No instances available for {model_name}")
        
        # Select instance using load balancing
        instance = self.load_balancer.select(instances)
        
        # Execute prediction
        try:
            return instance.predict(input_data)
        except Exception as e:
            # Mark instance as unhealthy
            self.health_checker.mark_unhealthy(instance)
            
            # Retry with different instance
            return self.retry_with_fallback(model_name, input_data)
    
    def retry_with_fallback(self, model_name, input_data, max_retries=3):
        """Retry with fallback instances."""
        for attempt in range(max_retries):
            instances = self.load_balancer.get_instances(model_name)
            instance = self.load_balancer.select(instances)
            
            try:
                return instance.predict(input_data)
            except Exception:
                continue
        
        raise MaxRetriesExceededError(f"Failed after {max_retries} retries")
```

### 2.2 Load Balancing Strategies

```python
class LoadBalancer:
    def __init__(self, strategy="round_robin"):
        self.strategy = strategy
        self.instances = defaultdict(list)
        self.current_index = {}
        self.health_status = {}
    
    def select(self, instances):
        """Select instance based on load balancing strategy."""
        if self.strategy == "round_robin":
            return self.round_robin(instances)
        elif self.strategy == "least_connections":
            return self.least_connections(instances)
        elif self.strategy == "weighted_round_robin":
            return self.weighted_round_robin(instances)
        elif self.strategy == "resource_based":
            return self.resource_based(instances)
        else:
            return random.choice(instances)
    
    def round_robin(self, instances):
        """Simple round-robin selection."""
        model_name = instances[0].model_name
        current = self.current_index.get(model_name, 0)
        
        selected = instances[current % len(instances)]
        self.current_index[model_name] = current + 1
        
        return selected
    
    def least_connections(self, instances):
        """Select instance with fewest active connections."""
        return min(instances, key=lambda i: i.active_connections)
    
    def weighted_round_robin(self, instances):
        """Weighted round-robin based on instance capacity."""
        total_weight = sum(i.weight for i in instances)
        random_weight = random.uniform(0, total_weight)
        
        cumulative = 0
        for instance in instances:
            cumulative += instance.weight
            if cumulative >= random_weight:
                return instance
        
        return instances[-1]
    
    def resource_based(self, instances):
        """Select based on available resources."""
        def score(instance):
            cpu_score = 1 - (instance.cpu_usage / 100)
            memory_score = 1 - (instance.memory_usage / 100)
            latency_score = 1 - (instance.avg_latency / 1000)
            return cpu_score * 0.4 + memory_score * 0.3 + latency_score * 0.3
        
        return max(instances, key=score)
```

### 2.3 Model Sharding

Distribute large models across multiple servers:

```python
class ModelShardManager:
    def __init__(self, model, shard_count):
        self.model = model
        self.shard_count = shard_count
        self.shards = self.partition_model()
    
    def partition_model(self):
        """Partition model into shards."""
        shards = []
        
        # Partition layers across shards
        layers = list(self.model.layers)
        shard_size = len(layers) // self.shard_count
        
        for i in range(self.shard_count):
            start_idx = i * shard_size
            end_idx = min((i + 1) * shard_size, len(layers))
            
            shard_layers = layers[start_idx:end_idx]
            shard = ModelShard(shard_layers, shard_id=i)
            shards.append(shard)
        
        return shards
    
    def predict(self, input_data):
        """Execute prediction across shards."""
        # Pipeline execution through shards
        current_output = input_data
        
        for shard in self.shards:
            current_output = shard.execute(current_output)
        
        return current_output
    
    def predict_parallel(self, input_data):
        """Execute prediction with parallel shards (for independent layers)."""
        # Identify independent layers
        independent_groups = self.find_independent_groups()
        
        # Execute independent groups in parallel
        outputs = []
        for group in independent_groups:
            group_output = self.execute_group(group, input_data)
            outputs.append(group_output)
        
        # Merge outputs
        return self.merge_outputs(outputs)
```

### 2.4 Distributed Inference Pipeline

```python
class DistributedInferencePipeline:
    def __init__(self, stages, coordinator):
        self.stages = stages
        self.coordinator = coordinator
    
    def execute(self, input_data):
        """Execute pipeline across distributed stages."""
        # Initialize pipeline
        pipeline_id = str(uuid.uuid4())
        context = {
            "pipeline_id": pipeline_id,
            "input": input_data,
            "stage_results": {},
            "metadata": {}
        }
        
        # Execute stages
        for stage in self.stages:
            stage_result = self.execute_stage(stage, context)
            context["stage_results"][stage.name] = stage_result
            
            # Check for early termination
            if stage_result.get("terminate"):
                break
        
        # Return final result
        return context["stage_results"].get("final_output")
    
    def execute_stage(self, stage, context):
        """Execute a single stage."""
        # Get available workers for this stage
        workers = self.coordinator.get_workers(stage.name)
        
        if not workers:
            raise StageExecutionError(f"No workers available for stage {stage.name}")
        
        # Select worker
        worker = self.select_worker(workers, stage)
        
        # Execute on worker
        try:
            result = worker.execute_stage(stage, context)
            return result
        except Exception as e:
            # Handle worker failure
            self.coordinator.handle_worker_failure(worker, e)
            
            # Retry with different worker
            return self.retry_stage(stage, context)
    
    def select_worker(self, workers, stage):
        """Select worker based on stage requirements."""
        if stage.resource_requirements:
            # Filter workers meeting resource requirements
            suitable_workers = [
                w for w in workers
                if w.has_resources(stage.resource_requirements)
            ]
            
            if suitable_workers:
                workers = suitable_workers
        
        # Select least loaded worker
        return min(workers, key=lambda w: w.load)
```

---

## 3. Production Monitoring Infrastructure

### 3.1 Comprehensive Metrics Collection

```python
class MetricsCollector:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
    
    def record_latency(self, model_name, latency_ms):
        """Record model inference latency."""
        self.histograms[f"{model_name}_latency"].append(latency_ms)
        self.gauges[f"{model_name}_p95_latency"] = np.percentile(
            self.histograms[f"{model_name}_latency"], 95
        )
    
    def record_cost(self, model_name, cost):
        """Record model inference cost."""
        self.counters[f"{model_name}_total_cost"] += cost
        self.counters[f"{model_name}_request_count"] += 1
        
        avg_cost = (self.counters[f"{model_name}_total_cost"] / 
                    self.counters[f"{model_name}_request_count"])
        self.gauges[f"{model_name}_avg_cost"] = avg_cost
    
    def record_quality(self, model_name, quality_score):
        """Record output quality score."""
        self.metrics[f"{model_name}_quality"].append(quality_score)
        
        recent = self.metrics[f"{model_name}_quality"][-100:]
        self.gauges[f"{model_name}_avg_quality"] = np.mean(recent)
    
    def record_error(self, model_name, error_type):
        """Record model errors."""
        self.counters[f"{model_name}_error_{error_type}"] += 1
        self.counters[f"{model_name}_total_requests"] += 1
        
        error_rate = (self.counters[f"{model_name}_error_{error_type}"] / 
                     self.counters[f"{model_name}_total_requests"])
        self.gauges[f"{model_name}_error_rate"] = error_rate
    
    def get_dashboard_data(self):
        """Generate data for monitoring dashboard."""
        return {
            "latency": {
                model: self.gauges[f"{model}_p95_latency"]
                for model in self.get_all_models()
            },
            "cost": {
                model: self.gauges[f"{model}_avg_cost"]
                for model in self.get_all_models()
            },
            "quality": {
                model: self.gauges[f"{model}_avg_quality"]
                for model in self.get_all_models()
            },
            "error_rate": {
                model: self.gauges[f"{model}_error_rate"]
                for model in self.get_all_models()
            }
        }
```

### 3.2 Real-Time Alerting

```python
class AlertingSystem:
    def __init__(self, alert_rules, notification_channels):
        self.rules = alert_rules
        self.channels = notification_channels
        self.alert_history = []
        self.suppression_window = 300  # 5 minutes
    
    def evaluate_rules(self, metrics):
        """Evaluate alert rules against current metrics."""
        alerts = []
        
        for rule in self.rules:
            if rule.evaluate(metrics):
                # Check suppression
                if not self.is_suppressed(rule.name):
                    alert = self.create_alert(rule, metrics)
                    alerts.append(alert)
                    self.record_alert(alert)
        
        return alerts
    
    def create_alert(self, rule, metrics):
        """Create alert notification."""
        return {
            "rule": rule.name,
            "severity": rule.severity,
            "message": rule.format_message(metrics),
            "timestamp": time.time(),
            "metrics": metrics,
            "suggested_actions": rule.suggested_actions
        }
    
    def send_alert(self, alert):
        """Send alert through configured channels."""
        for channel in self.channels:
            try:
                channel.send(alert)
            except Exception as e:
                logger.error(f"Failed to send alert through {channel.name}: {e}")
    
    def is_suppressed(self, rule_name):
        """Check if alert is suppressed due to recent firing."""
        recent_alerts = [
            a for a in self.alert_history
            if a["rule"] == rule_name and 
            time.time() - a["timestamp"] < self.suppression_window
        ]
        return len(recent_alerts) > 0

# Alert rules
ALERT_RULES = [
    {
        "name": "high_latency",
        "condition": lambda m: m["latency_p95"] > 2000,
        "severity": "warning",
        "message": "P95 latency exceeds 2000ms",
        "suggested_actions": ["Check model health", "Review routing decisions"]
    },
    {
        "name": "high_error_rate",
        "condition": lambda m: m["error_rate"] > 0.05,
        "severity": "critical",
        "message": "Error rate exceeds 5%",
        "suggested_actions": ["Trigger circuit breaker", "Switch to fallback model"]
    },
    {
        "name": "quality_degradation",
        "condition": lambda m: m["quality_score"] < 0.7,
        "severity": "warning",
        "message": "Quality score below threshold",
        "suggested_actions": ["Review model performance", "Consider model upgrade"]
    }
]
```

### 3.3 Distributed Tracing

```python
class DistributedTracer:
    def __init__(self, service_name):
        self.service_name = service_name
        self.traces = []
        self.spans = defaultdict(list)
    
    def start_trace(self, trace_id):
        """Start a new trace."""
        trace = {
            "trace_id": trace_id,
            "service": self.service_name,
            "start_time": time.time(),
            "spans": []
        }
        self.traces.append(trace)
        return trace
    
    def start_span(self, trace_id, span_name, parent_span_id=None):
        """Start a new span within a trace."""
        span = {
            "span_id": str(uuid.uuid4()),
            "parent_id": parent_span_id,
            "name": span_name,
            "start_time": time.time(),
            "tags": {},
            "logs": []
        }
        self.spans[trace_id].append(span)
        return span
    
    def finish_span(self, trace_id, span_id, success=True):
        """Finish a span."""
        for span in self.spans[trace_id]:
            if span["span_id"] == span_id:
                span["end_time"] = time.time()
                span["duration_ms"] = (span["end_time"] - span["start_time"]) * 1000
                span["success"] = success
                break
    
    def add_span_tag(self, trace_id, span_id, key, value):
        """Add tag to span."""
        for span in self.spans[trace_id]:
            if span["span_id"] == span_id:
                span["tags"][key] = value
                break
    
    def add_span_log(self, trace_id, span_id, message, level="info"):
        """Add log to span."""
        for span in self.spans[trace_id]:
            if span["span_id"] == span_id:
                span["logs"].append({
                    "timestamp": time.time(),
                    "level": level,
                    "message": message
                })
                break
    
    def get_trace(self, trace_id):
        """Get complete trace with all spans."""
        trace = next(t for t in self.traces if t["trace_id"] == trace_id)
        trace["spans"] = self.spans[trace_id]
        return trace
```

### 3.4 Cost Tracking Dashboard

```python
class CostDashboard:
    def __init__(self, metrics_collector, time_window=3600):
        self.metrics = metrics_collector
        self.time_window = time_window
    
    def generate_report(self):
        """Generate comprehensive cost report."""
        current_time = time.time()
        report = {
            "summary": self.generate_summary(),
            "model_breakdown": self.get_model_breakdown(),
            "hourly_trend": self.get_hourly_trend(),
            "cost_optimization_opportunities": self.find_optimization_opportunities()
        }
        return report
    
    def generate_summary(self):
        """Generate cost summary."""
        total_cost = sum(self.metrics.counters.get(f"{m}_total_cost", 0) 
                        for m in self.metrics.get_all_models())
        total_requests = sum(self.metrics.counters.get(f"{m}_request_count", 0) 
                           for m in self.metrics.get_all_models())
        
        return {
            "total_cost": total_cost,
            "total_requests": total_requests,
            "avg_cost_per_request": total_cost / total_requests if total_requests > 0 else 0,
            "cost_per_1k_tokens": self.calculate_cost_per_1k_tokens()
        }
    
    def get_model_breakdown(self):
        """Get cost breakdown by model."""
        breakdown = {}
        
        for model in self.metrics.get_all_models():
            total_cost = self.metrics.counters.get(f"{model}_total_cost", 0)
            request_count = self.metrics.counters.get(f"{model}_request_count", 0)
            
            breakdown[model] = {
                "total_cost": total_cost,
                "request_count": request_count,
                "avg_cost": total_cost / request_count if request_count > 0 else 0,
                "percentage": 0  # Will be calculated
            }
        
        # Calculate percentages
        total = sum(b["total_cost"] for b in breakdown.values())
        for model in breakdown:
            breakdown[model]["percentage"] = (
                breakdown[model]["total_cost"] / total * 100 if total > 0 else 0
            )
        
        return breakdown
    
    def find_optimization_opportunities(self):
        """Identify cost optimization opportunities."""
        opportunities = []
        
        # Check for expensive models used for simple tasks
        for model in self.metrics.get_all_models():
            if self.is_expensive_model(model):
                simple_task_ratio = self.get_simple_task_ratio(model)
                if simple_task_ratio > 0.5:
                    opportunities.append({
                        "type": "model_downgrade",
                        "model": model,
                        "current_usage": f"{simple_task_ratio*100:.1f}% simple tasks",
                        "potential_savings": self.estimate_savings(model, "downgrade")
                    })
        
        # Check for inefficient routing
        routing_efficiency = self.calculate_routing_efficiency()
        if routing_efficiency < 0.7:
            opportunities.append({
                "type": "routing_optimization",
                "current_efficiency": f"{routing_efficiency*100:.1f}%",
                "potential_savings": self.estimate_savings("routing", "optimize")
            })
        
        return opportunities
```

---

## 4. Advanced Optimization Techniques

### 4.1 Multi-Objective Optimization

Optimize multiple objectives simultaneously:

```python
class MultiObjectiveOptimizer:
    def __init__(self, objectives, constraints):
        self.objectives = objectives  # List of objective functions
        self.constraints = constraints  # List of constraint functions
    
    def optimize(self, models, input_data):
        """Find Pareto-optimal solutions."""
        # Generate candidate solutions
        candidates = self.generate_candidates(models, input_data)
        
        # Evaluate objectives for each candidate
        evaluated = []
        for candidate in candidates:
            objective_values = [obj(candidate) for obj in self.objectives]
            constraint_violations = [con(candidate) for con in self.constraints]
            
            evaluated.append({
                "candidate": candidate,
                "objectives": objective_values,
                "violations": constraint_violations
            })
        
        # Filter feasible solutions
        feasible = [e for e in evaluated if all(v == 0 for v in e["violations"])]
        
        # Find Pareto front
        pareto_front = self.find_pareto_front(feasible)
        
        return pareto_front
    
    def find_pareto_front(self, solutions):
        """Find Pareto-optimal solutions."""
        pareto = []
        
        for i, sol_i in enumerate(solutions):
            dominated = False
            
            for j, sol_j in enumerate(solutions):
                if i != j:
                    if self.dominates(sol_j["objectives"], sol_i["objectives"]):
                        dominated = True
                        break
            
            if not dominated:
                pareto.append(sol_i)
        
        return pareto
    
    def dominates(self, obj_a, obj_b):
        """Check if solution A dominates solution B."""
        # Assuming minimization for all objectives
        return all(a <= b for a, b in zip(obj_a, obj_b)) and any(a < b for a, b in zip(obj_a, obj_b))
    
    def generate_candidates(self, models, input_data):
        """Generate candidate model configurations."""
        candidates = []
        
        # Single model selections
        for model in models:
            candidates.append([model])
        
        # Model cascades
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i != j:
                    candidates.append([model1, model2])
        
        # Ensemble configurations
        for size in range(2, min(4, len(models) + 1)):
            for combination in itertools.combinations(models, size):
                candidates.append(list(combination))
        
        return candidates
```

### 4.2 Gradient-Based Optimization

Use gradients to optimize routing parameters:

```python
class GradientBasedOptimizer:
    def __init__(self, router, learning_rate=0.01):
        self.router = router
        self.learning_rate = learning_rate
        self.parameters = self.router.get_parameters()
    
    def optimize(self, training_data, epochs=100):
        """Optimize routing parameters using gradient descent."""
        for epoch in range(epochs):
            total_loss = 0
            
            for input_data, expected_output in training_data:
                # Forward pass
                routing_decision = self.router.route(input_data)
                output = self.predict_with_routing(input_data, routing_decision)
                
                # Compute loss
                loss = self.compute_loss(output, expected_output)
                total_loss += loss
                
                # Compute gradients
                gradients = self.compute_gradients(loss, routing_decision)
                
                # Update parameters
                self.update_parameters(gradients)
            
            # Log progress
            avg_loss = total_loss / len(training_data)
            logger.info(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
    
    def compute_gradients(self, loss, routing_decision):
        """Compute gradients of loss with respect to parameters."""
        gradients = {}
        
        for param_name, param_value in self.parameters.items():
            # Finite difference approximation
            epsilon = 1e-5
            
            # Positive perturbation
            self.parameters[param_name] = param_value + epsilon
            loss_plus = self.evaluate_loss()
            
            # Negative perturbation
            self.parameters[param_name] = param_value - epsilon
            loss_minus = self.evaluate_loss()
            
            # Compute gradient
            gradients[param_name] = (loss_plus - loss_minus) / (2 * epsilon)
            
            # Restore original value
            self.parameters[param_name] = param_value
        
        return gradients
    
    def update_parameters(self, gradients):
        """Update parameters using computed gradients."""
        for param_name, gradient in gradients.items():
            self.parameters[param_name] -= self.learning_rate * gradient
```

### 4.3 Evolutionary Optimization

Use evolutionary algorithms for model selection:

```python
class EvolutionaryOptimizer:
    def __init__(self, models, population_size=50, generations=100):
        self.models = models
        self.population_size = population_size
        self.generations = generations
    
    def optimize(self, fitness_function):
        """Optimize using genetic algorithm."""
        # Initialize population
        population = self.initialize_population()
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = [fitness_function(ind) for ind in population]
            
            # Selection
            selected = self.select(population, fitness_scores)
            
            # Crossover
            offspring = self.crossover(selected)
            
            # Mutation
            mutated = self.mutate(offspring)
            
            # Replace population
            population = self.survive(population, mutated, fitness_scores)
        
        # Return best solution
        best_idx = np.argmax([fitness_function(ind) for ind in population])
        return population[best_idx]
    
    def initialize_population(self):
        """Initialize random population of model configurations."""
        population = []
        
        for _ in range(self.population_size):
            # Random model selection
            config = {
                "primary_model": random.choice(self.models),
                "fallback_model": random.choice(self.models),
                "cascade_threshold": random.uniform(0.5, 0.9),
                "max_retries": random.randint(1, 5)
            }
            population.append(config)
        
        return population
    
    def select(self, population, fitness_scores):
        """Select individuals for reproduction."""
        # Tournament selection
        selected = []
        
        for _ in range(len(population)):
            tournament_size = 3
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx])
        
        return selected
    
    def crossover(self, parents):
        """Create offspring through crossover."""
        offspring = []
        
        for i in range(0, len(parents) - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]
            
            # Uniform crossover
            child = {}
            for key in parent1.keys():
                if random.random() < 0.5:
                    child[key] = parent1[key]
                else:
                    child[key] = parent2[key]
            
            offspring.append(child)
        
        return offspring
    
    def mutate(self, individuals):
        """Apply mutation to individuals."""
        mutated = []
        
        for ind in individuals:
            if random.random() < 0.1:  # 10% mutation rate
                mutated_ind = ind.copy()
                
                # Random mutation
                mutation_type = random.choice(["model", "threshold", "retries"])
                
                if mutation_type == "model":
                    mutated_ind["primary_model"] = random.choice(self.models)
                elif mutation_type == "threshold":
                    mutated_ind["cascade_threshold"] = random.uniform(0.5, 0.9)
                elif mutation_type == "retries":
                    mutated_ind["max_retries"] = random.randint(1, 5)
                
                mutated.append(mutated_ind)
            else:
                mutated.append(ind)
        
        return mutated
```

### 4.4 Meta-Heuristic Optimization

```python
class SimulatedAnnealingOptimizer:
    def __init__(self, models, initial_temp=100, cooling_rate=0.95):
        self.models = models
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
    
    def optimize(self, objective_function):
        """Optimize using simulated annealing."""
        # Initialize solution
        current_solution = self.generate_random_solution()
        current_fitness = objective_function(current_solution)
        
        best_solution = current_solution
        best_fitness = current_fitness
        
        temperature = self.initial_temp
        
        for iteration in range(1000):
            # Generate neighbor
            neighbor = self.generate_neighbor(current_solution)
            neighbor_fitness = objective_function(neighbor)
            
            # Calculate acceptance probability
            delta = neighbor_fitness - current_fitness
            if delta > 0:
                acceptance_prob = 1.0
            else:
                acceptance_prob = math.exp(delta / temperature)
            
            # Accept or reject
            if random.random() < acceptance_prob:
                current_solution = neighbor
                current_fitness = neighbor_fitness
                
                # Update best
                if current_fitness > best_fitness:
                    best_solution = current_solution
                    best_fitness = current_fitness
            
            # Cool down
            temperature *= self.cooling_rate
        
        return best_solution, best_fitness
    
    def generate_random_solution(self):
        """Generate random model configuration."""
        return {
            "primary": random.choice(self.models),
            "fallback": random.choice(self.models),
            "threshold": random.uniform(0.5, 0.9)
        }
    
    def generate_neighbor(self, solution):
        """Generate neighboring solution."""
        neighbor = solution.copy()
        
        # Random perturbation
        perturbation_type = random.choice(["model", "threshold"])
        
        if perturbation_type == "model":
            if random.random() < 0.5:
                neighbor["primary"] = random.choice(self.models)
            else:
                neighbor["fallback"] = random.choice(self.models)
        elif perturbation_type == "threshold":
            neighbor["threshold"] = random.uniform(0.5, 0.9)
        
        return neighbor
```

---

## 5. Error Handling and Recovery

### 5.1 Comprehensive Error Handling

```python
class ErrorHandler:
    def __init__(self, models, fallback_strategy="cascade"):
        self.models = models
        self.strategy = fallback_strategy
        self.error_history = []
        self.recovery_strategies = self.define_recovery_strategies()
    
    def define_recovery_strategies(self):
        """Define error recovery strategies."""
        return {
            "rate_limit": self.handle_rate_limit,
            "timeout": self.handle_timeout,
            "service_unavailable": self.handle_service_unavailable,
            "invalid_response": self.handle_invalid_response,
            "quality_degradation": self.handle_quality_degradation
        }
    
    def handle_error(self, error, context):
        """Handle error with appropriate recovery strategy."""
        error_type = self.classify_error(error)
        
        # Log error
        self.log_error(error, context, error_type)
        
        # Apply recovery strategy
        if error_type in self.recovery_strategies:
            recovery_result = self.recovery_strategies[error_type](error, context)
            return recovery_result
        else:
            # Default: fallback to next model
            return self.fallback_to_next_model(context)
    
    def handle_rate_limit(self, error, context):
        """Handle rate limiting errors."""
        # Wait and retry
        wait_time = self.calculate_backoff(error.retry_after)
        time.sleep(wait_time)
        
        # Try again with same model
        return self.retry_current_model(context)
    
    def handle_timeout(self, error, context):
        """Handle timeout errors."""
        # Try with smaller input
        truncated_input = self.truncate_input(context.input_data)
        
        # Retry with truncated input
        return self.retry_with_input(truncated_input, context)
    
    def handle_service_unavailable(self, error, context):
        """Handle service unavailability."""
        # Switch to fallback model
        return self.switch_to_fallback(context)
    
    def handle_invalid_response(self, error, context):
        """Handle invalid response format."""
        # Try to parse partial response
        parsed = self.parse_partial_response(error.response)
        
        if parsed:
            return {"output": parsed, "partial": True}
        else:
            return self.switch_to_fallback(context)
    
    def handle_quality_degradation(self, error, context):
        """Handle quality degradation."""
        # Use ensemble of remaining models
        return self.use_ensemble(context)
```

### 5.2 Circuit Breaker Implementation

```python
class AdvancedCircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60, half_open_max=3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max = half_open_max
        
        self.state = "closed"
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.half_open_attempts = 0
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
                self.half_open_attempts = 0
            else:
                raise CircuitBreakerOpenError("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            
            if self.state == "half-open":
                self.success_count += 1
                if self.success_count >= self.half_open_max:
                    self.state = "closed"
                    self.failure_count = 0
                    self.success_count = 0
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == "half-open":
                self.half_open_attempts += 1
                if self.half_open_attempts >= self.half_open_max:
                    self.state = "open"
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            
            raise
    
    def get_state_info(self):
        """Get circuit breaker state information."""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure": self.last_failure_time,
            "time_until_recovery": self.time_until_recovery()
        }
    
    def time_until_recovery(self):
        """Calculate time until circuit breaker recovers."""
        if self.state == "closed":
            return 0
        
        if self.last_failure_time:
            elapsed = time.time() - self.last_failure_time
            remaining = self.recovery_timeout - elapsed
            return max(0, remaining)
        
        return 0
```

### 5.3 Graceful Degradation

```python
class GracefulDegradation:
    def __init__(self, models, degradation_levels):
        self.models = models
        self.levels = degradation_levels
        self.current_level = 0
    
    def execute(self, input_data):
        """Execute with graceful degradation."""
        for level in range(self.current_level, len(self.levels)):
            try:
                result = self.execute_at_level(level, input_data)
                return result
            except Exception as e:
                logger.warning(f"Level {level} failed: {e}")
                self.current_level = level + 1
                continue
        
        # All levels failed
        return self.get_fallback_output(input_data)
    
    def execute_at_level(self, level, input_data):
        """Execute at specific degradation level."""
        config = self.levels[level]
        
        if config["type"] == "full":
            return self.execute_full(input_data)
        elif config["type"] == "simplified":
            return self.execute_simplified(input_data, config["simplifications"])
        elif config["type"] == "cached":
            return self.execute_from_cache(input_data)
        elif config["type"] == "template":
            return self.execute_from_template(input_data)
    
    def execute_simplified(self, input_data, simplifications):
        """Execute with simplified processing."""
        # Apply simplifications
        simplified_input = input_data
        for simplification in simplifications:
            simplified_input = simplification(simplified_input)
        
        # Execute with simpler model
        return self.models["simple"].predict(simplified_input)
```

---

## 6. Performance Profiling

### 6.1 Performance Profiler

```python
class PerformanceProfiler:
    def __init__(self):
        self.profiles = defaultdict(list)
        self.current_profile = None
    
    def start_profile(self, operation_name):
        """Start profiling an operation."""
        self.current_profile = {
            "operation": operation_name,
            "start_time": time.time(),
            "memory_start": self.get_memory_usage(),
            "cpu_start": self.get_cpu_usage(),
            "sub_profiles": []
        }
    
    def end_profile(self):
        """End current profiling."""
        if self.current_profile:
            self.current_profile["end_time"] = time.time()
            self.current_profile["duration"] = (
                self.current_profile["end_time"] - self.current_profile["start_time"]
            )
            self.current_profile["memory_end"] = self.get_memory_usage()
            self.current_profile["memory_delta"] = (
                self.current_profile["memory_end"] - self.current_profile["memory_start"]
            )
            
            self.profiles[self.current_profile["operation"]].append(self.current_profile)
            self.current_profile = None
    
    def profile_sub_operation(self, operation_name):
        """Profile a sub-operation within current profile."""
        if self.current_profile:
            sub_profile = {
                "operation": operation_name,
                "start_time": time.time()
            }
            self.current_profile["sub_profiles"].append(sub_profile)
            return sub_profile
        return None
    
    def get_memory_usage(self):
        """Get current memory usage in MB."""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def get_cpu_usage(self):
        """Get current CPU usage percentage."""
        import psutil
        return psutil.cpu_percent()
    
    def generate_report(self):
        """Generate performance report."""
        report = {}
        
        for operation, profiles in self.profiles.items():
            durations = [p["duration"] for p in profiles]
            memory_deltas = [p["memory_delta"] for p in profiles]
            
            report[operation] = {
                "count": len(profiles),
                "avg_duration": np.mean(durations),
                "p95_duration": np.percentile(durations, 95),
                "max_duration": max(durations),
                "avg_memory_delta": np.mean(memory_deltas),
                "total_time": sum(durations)
            }
        
        return report
```

### 6.2 Latency Analysis

```python
class LatencyAnalyzer:
    def __init__(self):
        self.latency_records = []
    
    def record_latency(self, model_name, operation, latency_ms, context=None):
        """Record latency measurement."""
        record = {
            "model": model_name,
            "operation": operation,
            "latency_ms": latency_ms,
            "timestamp": time.time(),
            "context": context or {}
        }
        self.latency_records.append(record)
    
    def analyze_latency(self, model_name=None, time_window=3600):
        """Analyze latency patterns."""
        # Filter records
        current_time = time.time()
        filtered = [
            r for r in self.latency_records
            if current_time - r["timestamp"] <= time_window
        ]
        
        if model_name:
            filtered = [r for r in filtered if r["model"] == model_name]
        
        if not filtered:
            return None
        
        # Calculate statistics
        latencies = [r["latency_ms"] for r in filtered]
        
        analysis = {
            "count": len(latencies),
            "mean": np.mean(latencies),
            "median": np.median(latencies),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99),
            "min": min(latencies),
            "max": max(latencies),
            "std": np.std(latencies)
        }
        
        # Identify latency spikes
        spikes = self.identify_spikes(latencies)
        analysis["spikes"] = spikes
        
        return analysis
    
    def identify_spikes(self, latencies, threshold=2.0):
        """Identify latency spikes."""
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        
        spikes = []
        for i, latency in enumerate(latencies):
            if latency > mean_latency + threshold * std_latency:
                spikes.append({
                    "index": i,
                    "latency": latency,
                    "deviation": (latency - mean_latency) / std_latency
                })
        
        return spikes
    
    def correlate_with_factors(self, latency_records):
        """Correlate latency with various factors."""
        correlations = {}
        
        # Input length correlation
        input_lengths = [r["context"].get("input_length", 0) for r in latency_records]
        latencies = [r["latency_ms"] for r in latency_records]
        
        if input_lengths and latencies:
            correlations["input_length"] = np.corrcoef(input_lengths, latencies)[0, 1]
        
        # Time of day correlation
        hours = [r["timestamp"] % 86400 / 3600 for r in latency_records]
        if hours and latencies:
            correlations["time_of_day"] = np.corrcoef(hours, latencies)[0, 1]
        
        return correlations
```

---

## 7. Advanced Cascading Patterns

### 7.1 Adaptive Threshold Cascade

```python
class AdaptiveThresholdCascade:
    def __init__(self, models, initial_thresholds):
        self.models = models
        self.thresholds = initial_thresholds
        self.performance_history = defaultdict(list)
        self.adaptation_rate = 0.1
    
    def predict(self, input_data):
        """Adaptive cascade with dynamic thresholds."""
        for i, (model, threshold) in enumerate(zip(self.models, self.thresholds)):
            result = model.predict(input_data)
            
            # Record performance
            self.record_performance(model.name, result.confidence, result.cost)
            
            # Adjust threshold based on performance
            self.adjust_threshold(i, result)
            
            if result.confidence >= threshold:
                return result
        
        return result
    
    def adjust_threshold(self, model_index, result):
        """Adjust threshold based on recent performance."""
        model_name = self.models[model_index].name
        
        if len(self.performance_history[model_name]) < 10:
            return
        
        recent = self.performance_history[model_name][-100:]
        avg_confidence = np.mean([p["confidence"] for p in recent])
        avg_cost = np.mean([p["cost"] for p in recent])
        
        # Lower threshold if model is performing well and cheap
        if avg_confidence > 0.9 and avg_cost < 0.01:
            self.thresholds[model_index] *= (1 - self.adaptation_rate)
        
        # Raise threshold if model is expensive or low confidence
        elif avg_cost > 0.05 or avg_confidence < 0.7:
            self.thresholds[model_index] *= (1 + self.adaptation_rate)
        
        # Keep threshold within bounds
        self.thresholds[model_index] = max(0.3, min(0.95, self.thresholds[model_index]))
```

### 7.2 Context-Aware Cascade

```python
class ContextAwareCascade:
    def __init__(self, models, context_extractor):
        self.models = models
        self.extractor = context_extractor
        self.context_rules = self.define_context_rules()
    
    def define_context_rules(self):
        """Define rules for context-aware routing."""
        return {
            "code_generation": {
                "preferred_models": ["claude-fable-5", "gpt-5.5-high"],
                "threshold_adjustment": 0.1
            },
            "creative_writing": {
                "preferred_models": ["claude-opus", "gpt-5.5-high"],
                "threshold_adjustment": 0.05
            },
            "factual_qa": {
                "preferred_models": ["gemini-3-pro", "gpt-5.5-medium"],
                "threshold_adjustment": 0.15
            }
        }
    
    def predict(self, input_data):
        """Context-aware cascade prediction."""
        # Extract context
        context = self.extractor.extract(input_data)
        task_type = context.get("task_type", "general")
        
        # Get context-specific rules
        rules = self.context_rules.get(task_type, {})
        preferred_models = rules.get("preferred_models", [])
        threshold_adjustment = rules.get("threshold_adjustment", 0)
        
        # Reorder models based on context
        ordered_models = self.reorder_models(self.models, preferred_models)
        
        # Run cascade with adjusted thresholds
        for model in ordered_models:
            adjusted_threshold = self.get_threshold(model.name) + threshold_adjustment
            result = model.predict(input_data)
            
            if result.confidence >= adjusted_threshold:
                return result
        
        return result
    
    def reorder_models(self, models, preferred_models):
        """Reorder models based on preferences."""
        # Put preferred models first
        reordered = []
        for model in models:
            if model.name in preferred_models:
                reordered.insert(0, model)
            else:
                reordered.append(model)
        
        return reordered
```

### 7.3 Cost-Quality Pareto Cascade

```python
class ParetoCascade:
    def __init__(self, models, quality_threshold, cost_budget):
        self.models = models
        self.quality_threshold = quality_threshold
        self.cost_budget = cost_budget
        self.pareto_front = self.compute_pareto_front()
    
    def compute_pareto_front(self):
        """Compute Pareto front of models."""
        pareto = []
        
        for model in self.models:
            quality = self.estimate_quality(model)
            cost = self.estimate_cost(model)
            
            # Check if model is on Pareto front
            dominated = False
            for other in self.models:
                other_quality = self.estimate_quality(other)
                other_cost = self.estimate_cost(other)
                
                if (other_quality >= quality and other_cost <= cost and 
                    (other_quality > quality or other_cost < cost)):
                    dominated = True
                    break
            
            if not dominated:
                pareto.append({
                    "model": model,
                    "quality": quality,
                    "cost": cost
                })
        
        return pareto
    
    def select_model(self, input_data):
        """Select model from Pareto front based on constraints."""
        candidates = []
        
        for point in self.pareto_front:
            model = point["model"]
            quality = self.estimate_quality(model, input_data)
            cost = self.estimate_cost(model, input_data)
            
            if quality >= self.quality_threshold and cost <= self.cost_budget:
                efficiency = quality / cost
                candidates.append({
                    "model": model,
                    "quality": quality,
                    "cost": cost,
                    "efficiency": efficiency
                })
        
        if candidates:
            # Select most efficient
            return max(candidates, key=lambda x: x["efficiency"])
        
        # No model meets constraints - relax constraints
        return self.relax_constraints()
```

---

## 8. Model Composition Patterns

### 8.1 Pipeline Composition

```python
class PipelineComposer:
    def __init__(self):
        self.stages = []
        self.connections = []
    
    def add_stage(self, stage_name, model, input_schema, output_schema):
        """Add a stage to the pipeline."""
        stage = {
            "name": stage_name,
            "model": model,
            "input_schema": input_schema,
            "output_schema": output_schema
        }
        self.stages.append(stage)
    
    def connect(self, source_stage, target_stage, mapping):
        """Connect two stages with data mapping."""
        connection = {
            "source": source_stage,
            "target": target_stage,
            "mapping": mapping
        }
        self.connections.append(connection)
    
    def execute(self, input_data):
        """Execute the composed pipeline."""
        context = {"input": input_data}
        
        # Topological sort of stages
        execution_order = self.topological_sort()
        
        for stage_name in execution_order:
            stage = self.get_stage(stage_name)
            
            # Map input from context
            stage_input = self.map_input(context, stage["input_schema"])
            
            # Execute stage
            output = stage["model"].predict(stage_input)
            
            # Store output in context
            context[stage_name] = output
        
        return context
    
    def topological_sort(self):
        """Sort stages in execution order."""
        # Build adjacency list
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        
        for stage in self.stages:
            in_degree[stage["name"]] = 0
        
        for conn in self.connections:
            graph[conn["source"]].append(conn["target"])
            in_degree[conn["target"]] += 1
        
        # Kahn's algorithm
        queue = [s for s in in_degree if in_degree[s] == 0]
        order = []
        
        while queue:
            node = queue.pop(0)
            order.append(node)
            
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return order
```

### 8.2 Graph-Based Composition

```python
class GraphComposer:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.models = {}
    
    def add_model(self, model_name, model, capabilities):
        """Add a model to the composition graph."""
        self.models[model_name] = {
            "model": model,
            "capabilities": capabilities
        }
        self.graph.add_node(model_name, **capabilities)
    
    def add_connection(self, source, target, compatibility_score):
        """Add connection between models."""
        self.graph.add_edge(source, target, weight=compatibility_score)
    
    def compose_for_task(self, task_requirements):
        """Compose models for specific task requirements."""
        # Find models that match task requirements
        matching_models = self.find_matching_models(task_requirements)
        
        if not matching_models:
            raise NoMatchingModelsError("No models match task requirements")
        
        # Find optimal composition path
        composition = self.find_optimal_path(matching_models, task_requirements)
        
        return composition
    
    def find_matching_models(self, requirements):
        """Find models matching task requirements."""
        matching = []
        
        for model_name, model_info in self.models.items():
            if self.matches_requirements(model_info["capabilities"], requirements):
                matching.append(model_name)
        
        return matching
    
    def find_optimal_path(self, start_models, requirements):
        """Find optimal path through model graph."""
        # Use Dijkstra's algorithm
        best_path = None
        best_cost = float('inf')
        
        for start in start_models:
            try:
                # Find shortest path to any end model
                for end in self.get_end_models():
                    path = nx.dijkstra_path(
                        self.graph, start, end, weight='weight'
                    )
                    
                    # Calculate total cost
                    cost = self.calculate_path_cost(path, requirements)
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_path = path
            
            except nx.NetworkXNoPath:
                continue
        
        return best_path
```

### 8.3 Dynamic Composition

```python
class DynamicComposer:
    def __init__(self, models, composition_rules):
        self.models = models
        self.rules = composition_rules
        self.composition_cache = {}
    
    def compose(self, input_data, constraints):
        """Dynamically compose models based on input and constraints."""
        # Analyze input
        input_analysis = self.analyze_input(input_data)
        
        # Generate composition candidates
        candidates = self.generate_compositions(input_analysis, constraints)
        
        # Evaluate candidates
        evaluated = self.evaluate_compositions(candidates, input_data)
        
        # Select best composition
        best = max(evaluated, key=lambda x: x["score"])
        
        return best["composition"]
    
    def analyze_input(self, input_data):
        """Analyze input characteristics."""
        return {
            "length": len(input_data),
            "complexity": self.estimate_complexity(input_data),
            "domain": self.detect_domain(input_data),
            "quality_requirements": self.infer_quality_requirements(input_data),
            "latency_requirements": self.infer_latency_requirements(input_data)
        }
    
    def generate_compositions(self, analysis, constraints):
        """Generate possible model compositions."""
        compositions = []
        
        # Single model compositions
        for model in self.models:
            if self.is_suitable(model, analysis, constraints):
                compositions.append([model])
        
        # Multi-model compositions
        for size in range(2, 4):
            for combo in itertools.combinations(self.models, size):
                if self.is_composition_suitable(combo, analysis, constraints):
                    compositions.append(list(combo))
        
        return compositions
    
    def evaluate_compositions(self, compositions, input_data):
        """Evaluate composition candidates."""
        evaluated = []
        
        for composition in compositions:
            # Estimate performance
            estimated_quality = self.estimate_quality(composition, input_data)
            estimated_cost = self.estimate_cost(composition, input_data)
            estimated_latency = self.estimate_latency(composition, input_data)
            
            # Calculate score
            score = self.calculate_score(
                estimated_quality, estimated_cost, estimated_latency
            )
            
            evaluated.append({
                "composition": composition,
                "score": score,
                "estimated_quality": estimated_quality,
                "estimated_cost": estimated_cost,
                "estimated_latency": estimated_latency
            })
        
        return evaluated
```

---

## 9. Production Deployment Patterns

### 9.1 Blue-Green Deployment

```python
class BlueGreenDeployment:
    def __init__(self, blue_env, green_env):
        self.blue = blue_env
        self.green = green_env
        self.active = "blue"
    
    def deploy_new_version(self, new_models):
        """Deploy new model version to inactive environment."""
        if self.active == "blue":
            target = self.green
        else:
            target = self.blue
        
        # Deploy to target environment
        target.deploy(new_models)
        
        # Run health checks
        if target.health_check():
            # Switch traffic
            self.switch_traffic()
            return True
        else:
            # Rollback
            target.rollback()
            return False
    
    def switch_traffic(self):
        """Switch traffic to other environment."""
        if self.active == "blue":
            self.active = "green"
        else:
            self.active = "blue"
        
        # Update load balancer
        self.update_load_balancer()
    
    def rollback(self):
        """Rollback to previous environment."""
        self.switch_traffic()
```

### 9.2 Canary Deployment

```python
class CanaryDeployment:
    def __init__(self, stable_env, canary_env, canary_percentage=5):
        self.stable = stable_env
        self.canary = canary_env
        self.canary_percentage = canary_percentage
    
    def deploy_canary(self, new_models):
        """Deploy canary version with limited traffic."""
        # Deploy to canary environment
        self.canary.deploy(new_models)
        
        # Route small percentage of traffic to canary
        self.update_routing(percentage=self.canary_percentage)
        
        # Monitor canary performance
        canary_metrics = self.monitor_canary()
        
        if canary_metrics["error_rate"] < 0.01 and canary_metrics["latency_p95"] < 1000:
            # Canary looks good, gradually increase traffic
            self.increase_canary_traffic()
        else:
            # Rollback canary
            self.rollback_canary()
    
    def increase_canary_traffic(self):
        """Gradually increase canary traffic."""
        for percentage in [10, 25, 50, 75, 100]:
            self.update_routing(percentage=percentage)
            
            # Monitor at each level
            metrics = self.monitor_canary()
            
            if metrics["error_rate"] > 0.01 or metrics["latency_p95"] > 1000:
                self.rollback_canary()
                return
            
            time.sleep(300)  # Wait 5 minutes at each level
        
        # Full deployment
        self.promote_canary_to_stable()
```

### 9.3 A/B Testing Framework

```python
class ABTestingFramework:
    def __init__(self, control_model, treatment_model, traffic_split=50):
        self.control = control_model
        self.treatment = treatment_model
        self.split = traffic_split
        
        self.results = {"control": [], "treatment": []}
    
    def route_request(self, user_id):
        """Route request to control or treatment."""
        # Deterministic routing based on user ID
        hash_value = hash(user_id) % 100
        
        if hash_value < self.split:
            return "control", self.control
        else:
            return "treatment", self.treatment
    
    def record_result(self, group, input_data, output, quality_score):
        """Record experiment result."""
        self.results[group].append({
            "input": input_data,
            "output": output,
            "quality": quality_score,
            "timestamp": time.time()
        })
    
    def analyze_results(self):
        """Analyze A/B test results."""
        control_results = self.results["control"]
        treatment_results = self.results["treatment"]
        
        if not control_results or not treatment_results:
            return None
        
        # Calculate statistics
        control_quality = [r["quality"] for r in control_results]
        treatment_quality = [r["quality"] for r in treatment_results]
        
        # Perform t-test
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(control_quality, treatment_quality)
        
        # Calculate improvement
        control_mean = np.mean(control_quality)
        treatment_mean = np.mean(treatment_quality)
        improvement = (treatment_mean - control_mean) / control_mean * 100
        
        return {
            "control_mean": control_mean,
            "treatment_mean": treatment_mean,
            "improvement": improvement,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "control_count": len(control_results),
            "treatment_count": len(treatment_results)
        }
```

---

## 10. Cross-References

### Related Documents in This Library

| Document | Relevance |
|----------|-----------|
| 02-LLMs/10-AI-Model-Routing | Core routing concepts and implementations |
| 03-Agents/02-Multi-Agent-Systems | Agent-based orchestration patterns |
| 06-Advanced/03-Evaluation-Benchmarks | Quality assessment methodologies |
| 31-AI-Workflow-Orchestration | Workflow management patterns |
| 52-AI-Hallucination-Detection | Quality assurance and validation |

### External Resources

- **Research Papers:**
  - "Scaling Laws for Neural Language Models" (Kaplan et al.)
  - "Mixture of Experts Survey" (2024)
  - "Cost-Efficient LLM Inference" (2026)

- **Frameworks and Tools:**
  - Ray Serve (distributed model serving)
  - Kubernetes (container orchestration)
  - Istio (service mesh)
  - Prometheus (monitoring)

- **Industry Reports:**
  - Gartner: "Multi-Model AI Strategies" (2026)
  - McKinsey: "Optimizing AI Infrastructure Costs" (2026)

### Key Takeaways

1. **Advanced routing algorithms** significantly improve cost-efficiency
2. **Distributed systems** enable scaling to production workloads
3. **Comprehensive monitoring** is essential for production reliability
4. **Advanced optimization** requires sophisticated algorithms
5. **Graceful degradation** ensures system availability
6. **Performance profiling** identifies optimization opportunities

---

*Last Updated: July 2026*
*Next Review: October 2026*
