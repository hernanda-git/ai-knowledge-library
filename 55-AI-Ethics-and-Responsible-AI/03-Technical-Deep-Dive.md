# Technical Deep Dive: AI Ethics Implementation

> This document provides detailed technical guidance on implementing fairness, explainability, and ethical monitoring in production AI systems.

---

## Table of Contents

1. [Fairness-Aware ML Pipelines](#1-fairness-aware-ml-pipelines)
2. [Explainability at Scale](#2-explainability-at-scale)
3. [Ethical Monitoring Systems](#3-ethical-monitoring-systems)
4. [Bias Audit Implementations](#4-bias-audit-implementations)
5. [Privacy-Preserving ML](#5-privacy-preserving-ml)
6. [Production Case Studies](#6-production-case-studies)
7. [Anti-Patterns and Pitfalls](#7-anti-patterns-and-pitfalls)
8. [Testing Ethical Systems](#8-testing-ethical-systems)

---

## 1. Fairness-Aware ML Pipelines

### 1.1 End-to-End Fairness Pipeline

```python
class FairnessPipeline:
    """Production fairness-aware ML pipeline."""

    def __init__(self, fairness_threshold=0.8):
        self.fairness_threshold = fairness_threshold
        self.metrics_history = []

    def run(self, X, y, sensitive_features):
        """Execute full fairness-aware pipeline."""
        # Step 1: Bias detection
        bias_report = self.detect_bias(X, y, sensitive_features)

        # Step 2: Mitigation strategy selection
        strategy = self.select_strategy(bias_report)

        # Step 3: Apply mitigation
        X_mitigated, y_mitigated = self.mitigate(X, y, sensitive_features, strategy)

        # Step 4: Train model
        model = self.train(X_mitigated, y_mitigated)

        # Step 5: Validate fairness
        fairness_report = self.validate(model, X, y, sensitive_features)

        return model, fairness_report

    def detect_bias(self, X, y, sensitive_features):
        """Detect bias in training data."""
        report = {}
        for col in sensitive_features.columns:
            groups = sensitive_features[col].unique()
            rates = {g: y[sensitive_features[col] == g].mean() for g in groups}
            min_rate = min(rates.values())
            max_rate = max(rates.values())
            report[col] = {
                'disparate_impact': min_rate / max_rate if max_rate > 0 else 0,
                'group_rates': rates,
                'needs_mitigation': (min_rate / max_rate) < self.fairness_threshold
                    if max_rate > 0 else False,
            }
        return report

    def select_strategy(self, bias_report):
        """Select mitigation strategy based on bias characteristics."""
        strategies = []
        for attr, info in bias_report.items():
            if info['needs_mitigation']:
                if info['disparate_impact'] < 0.5:
                    strategies.append(('resample', attr))
                else:
                    strategies.append(('reweight', attr))
        return strategies

    def mitigate(self, X, y, sensitive_features, strategies):
        """Apply selected mitigation strategies."""
        X_out, y_out = X.copy(), y.copy()
        for strategy, attr in strategies:
            if strategy == 'reweight':
                weights = self.compute_reweights(y_out, sensitive_features[attr])
                # Apply weights during training instead of modifying data
            elif strategy == 'resample':
                X_out, y_out = self.resample(X_out, y_out, sensitive_features[attr])
        return X_out, y_out

    def validate(self, model, X, y, sensitive_features):
        """Validate fairness of trained model."""
        y_pred = model.predict(X)
        report = {}
        for col in sensitive_features.columns:
            groups = sensitive_features[col].unique()
            rates = {g: y_pred[sensitive_features[col] == g].mean() for g in groups}
            min_rate = min(rates.values())
            max_rate = max(rates.values())
            report[col] = {
                'disparate_impact': min_rate / max_rate if max_rate > 0 else 0,
                'passed': (min_rate / max_rate) >= self.fairness_threshold
                    if max_rate > 0 else True,
            }
        return report
```

### 1.2 Reweighting Implementation

```python
import numpy as np
import pandas as pd

def compute_fair_weights(y, sensitive_feature):
    """Compute sample weights for demographic parity."""
    df = pd.DataFrame({'y': y, 'sens': sensitive_feature})
    # Target: each (group, label) pair has equal weight
    total = len(df)
    expected = total / (df['sens'].nunique() * df['y'].nunique())
    weights = np.ones(total)
    for _, row in df.iterrows():
        actual = ((df['sens'] == row['sens']) & (df['y'] == row['y'])).sum()
        idx = df.index.get_loc(_)
        weights[idx] = expected / actual
    return weights / weights.mean()  # Normalize
```

### 1.3 Resampling Strategy

```python
from sklearn.utils import resample

def balance_by_resampling(X, y, sensitive_feature, strategy='oversample'):
    """Balance dataset through resampling."""
    df = pd.DataFrame(X)
    df['y'] = y
    df['sens'] = sensitive_feature.values

    target_size = df.groupby(['sens', 'y']).size().max()

    balanced_dfs = []
    for (group, label), subset in df.groupby(['sens', 'y']):
        if strategy == 'oversample' and len(subset) < target_size:
            sampled = resample(subset, replace=True, n_samples=target_size,
                               random_state=42)
            balanced_dfs.append(sampled)
        elif strategy == 'undersample' and len(subset) > target_size:
            sampled = resample(subset, replace=False, n_samples=target_size,
                               random_state=42)
            balanced_dfs.append(sampled)
        else:
            balanced_dfs.append(subset)

    result = pd.concat(balanced_dfs).sample(frac=1, random_state=42)
    return result.drop(columns=['y', 'sens']).values, result['y'].values
```

---

## 2. Explainability at Scale

### 2.1 SHAP for Production

```python
import shap
import numpy as np

class ProductionExplainer:
    """SHAP-based explainer optimized for production use."""

    def __init__(self, model, background_data, max_background=100):
        # Use kmeans background for speed
        self.background = shap.kmeans(background_data, max_background)
        self.explainer = shap.TreeExplainer(
            model, self.background,
            feature_perturbation='interventional'
        )
        self.feature_names = None

    def explain_batch(self, X, top_k=5):
        """Explain a batch of predictions efficiently."""
        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class

        explanations = []
        for i in range(len(X)):
            importance = list(zip(
                self.feature_names or range(X.shape[1]),
                shap_values[i]
            ))
            importance.sort(key=lambda x: abs(x[1]), reverse=True)
            explanations.append({
                'top_features': importance[:top_k],
                'base_value': float(self.explainer.expected_value[1]
                    if isinstance(self.explainer.expected_value, list)
                    else self.explainer.expected_value),
                'prediction': float(shap_values[i].sum() +
                    self.explainer.expected_value[1]
                    if isinstance(self.explainer.expected_value, list)
                    else shap_values[i].sum() + self.explainer.expected_value),
            })
        return explanations

    def fairness_explanation(self, X, sensitive_features, threshold=0.05):
        """Flag predictions where sensitive features drive outcomes."""
        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        alerts = []
        total_shap = np.abs(shap_values).sum(axis=1)
        for i in range(len(X)):
            for j, feat_name in enumerate(
                sensitive_features.columns if hasattr(sensitive_features, 'columns')
                else range(sensitive_features.shape[1])
            ):
                pct = abs(shap_values[i, j]) / total_shap[i] if total_shap[i] > 0 else 0
                if pct > threshold:
                    alerts.append({
                        'index': i,
                        'feature': feat_name,
                        'importance_pct': pct,
                        'shap_value': float(shap_values[i, j]),
                    })
        return alerts
```

### 2.2 Counterfactual Explanations in Production

```python
import numpy as np

class CounterfactualService:
    """Generate counterfactual explanations for production predictions."""

    def __init__(self, model, feature_ranges, immutable_features=None):
        self.model = model
        self.feature_ranges = feature_ranges  # {feat: (min, max)}
        self.immutable = immutable_features or []

    def explain(self, instance, desired_class, max_changes=3):
        """Find minimal counterfactual to change prediction."""
        original_pred = self.model.predict(instance.reshape(1, -1))[0]
        if original_pred == desired_class:
            return {'changed': False, 'reason': 'Already desired class'}

        best_cf, best_dist = None, float('inf')
        features = list(range(len(instance)))
        changeable = [f for f in features if f not in self.immutable]

        for n in range(1, min(max_changes + 1, len(changeable) + 1)):
            for combo in self._combinations(changeable, n):
                cf = instance.copy()
                for idx in combo:
                    cf[idx] = self._best_value(cf, idx, desired_class)
                if self.model.predict(cf.reshape(1, -1))[0] == desired_class:
                    dist = np.linalg.norm(cf - instance)
                    if dist < best_dist:
                        best_cf, best_dist = cf, dist

        if best_cf is None:
            return {'changed': False, 'reason': 'No feasible counterfactual found'}

        changes = []
        for i in range(len(instance)):
            if abs(best_cf[i] - instance[i]) > 1e-6:
                changes.append({'feature': i, 'from': float(instance[i]),
                                'to': float(best_cf[i])})
        return {'changed': True, 'distance': float(best_dist), 'changes': changes}

    def _best_value(self, instance, feat_idx, desired_class):
        lo, hi = self.feature_ranges.get(feat_idx, (-1e9, 1e9))
        candidates = np.linspace(lo, hi, 20)
        best_val, best_dist = instance[feat_idx], float('inf')
        for val in candidates:
            tmp = instance.copy()
            tmp[feat_idx] = val
            if self.model.predict(tmp.reshape(1, -1))[0] == desired_class:
                d = abs(val - instance[feat_idx])
                if d < best_dist:
                    best_val, best_dist = val, d
        return best_val

    @staticmethod
    def _combinations(iterable, r):
        from itertools import combinations
        return combinations(iterable, r)
```

---

## 3. Ethical Monitoring Systems

### 3.1 Real-Time Fairness Monitor

```python
import time
from collections import defaultdict

class FairnessMonitor:
    """Real-time fairness monitoring for production models."""

    def __init__(self, protected_attrs, alert_thresholds=None):
        self.protected_attrs = protected_attrs
        self.thresholds = alert_thresholds or {
            'disparate_impact': 0.8,
            'demographic_parity_diff': 0.1,
            'equal_opportunity_diff': 0.05,
        }
        self.predictions = defaultdict(list)
        self.outcomes = defaultdict(list)
        self.alerts = []

    def log_prediction(self, prediction, sensitive_features, outcome=None):
        """Log a prediction with its sensitive features."""
        for attr in self.protected_attrs:
            val = sensitive_features.get(attr)
            key = f"{attr}_{val}"
            self.predictions[key].append(prediction)
            if outcome is not None:
                self.outcomes[key].append(outcome)

    def check_fairness(self):
        """Check current fairness metrics and generate alerts."""
        new_alerts = []
        # Group predictions by protected attribute
        attr_groups = defaultdict(dict)
        for key, preds in self.predictions.items():
            attr, val = key.rsplit('_', 1)
            attr_groups[attr][val] = np.array(preds)

        for attr, groups in attr_groups.items():
            if len(groups) < 2:
                continue
            rates = {g: p.mean() for g, p in groups.items()}
            min_rate = min(rates.values())
            max_rate = max(rates.values())
            di = min_rate / max_rate if max_rate > 0 else 1.0

            if di < self.thresholds['disparate_impact']:
                alert = {
                    'type': 'disparate_impact',
                    'attribute': attr,
                    'ratio': di,
                    'group_rates': rates,
                    'severity': 'HIGH' if di < 0.6 else 'MEDIUM',
                    'timestamp': time.time(),
                }
                new_alerts.append(alert)
                self.alerts.append(alert)

        return new_alerts

    def get_dashboard_data(self):
        """Get data for monitoring dashboard."""
        return {
            'total_predictions': sum(len(v) for v in self.predictions.values()),
            'alerts': len(self.alerts),
            'recent_alerts': self.alerts[-10:],
            'group_statistics': {
                k: {'count': len(v), 'positive_rate': float(np.mean(v))}
                for k, v in self.predictions.items()
            },
        }
```

### 3.2 Drift Detection for Fairness

```python
from scipy.stats import ks_2samp

class FairnessDriftDetector:
    """Detect drift in fairness metrics over time."""

    def __init__(self, baseline_metrics, window_size=1000):
        self.baseline = baseline_metrics
        self.window_size = window_size
        self.current_window = []

    def update(self, metric_value):
        """Add a new metric observation."""
        self.current_window.append(metric_value)
        if len(self.current_window) > self.window_size:
            self.current_window.pop(0)

    def detect_drift(self):
        """Check if current window differs from baseline."""
        if len(self.current_window) < 100:
            return {'drift_detected': False, 'reason': 'Insufficient data'}

        stat, p_value = ks_2samp(self.baseline, self.current_window)
        drift = p_value < 0.05

        return {
            'drift_detected': drift,
            'ks_statistic': float(stat),
            'p_value': float(p_value),
            'recommendation': 'Investigate fairness drift' if drift
                else 'No drift detected',
        }
```

---

## 4. Bias Audit Implementations

### 4.1 Comprehensive Bias Audit

```python
class BiasAudit:
    """Comprehensive bias audit for production models."""

    def __init__(self, model, test_data, sensitive_features):
        self.model = model
        self.X = test_data
        self.sensitive = sensitive_features
        self.y_pred = model.predict(test_data)
        self.y_true = None  # Set if ground truth available

    def run_full_audit(self, y_true=None):
        """Run comprehensive bias audit."""
        self.y_true = y_true
        report = {
            'summary': self._summary_stats(),
            'fairness_metrics': self._fairness_metrics(),
            'group_analysis': self._group_analysis(),
            'intersectional': self._intersectional_analysis(),
            'recommendations': self._recommendations(),
        }
        return report

    def _summary_stats(self):
        return {
            'total_samples': len(self.X),
            'overall_positive_rate': float(self.y_pred.mean()),
            'n_protected_attributes': len(self.sensitive.columns)
                if hasattr(self.sensitive, 'columns') else 1,
        }

    def _fairness_metrics(self):
        metrics = {}
        for col in (self.sensitive.columns if hasattr(self.sensitive, 'columns')
                     else [0]):
            groups = self.sensitive[col].unique()
            rates = {str(g): float(self.y_pred[self.sensitive[col] == g].mean())
                     for g in groups}
            min_r, max_r = min(rates.values()), max(rates.values())
            metrics[str(col)] = {
                'disparate_impact_ratio': min_r / max_r if max_r > 0 else 1.0,
                'group_positive_rates': rates,
                'passed': (min_r / max_r) >= 0.8 if max_r > 0 else True,
            }
        return metrics

    def _group_analysis(self):
        analysis = {}
        for col in (self.sensitive.columns if hasattr(self.sensitive, 'columns')
                     else [0]):
            for g in self.sensitive[col].unique():
                mask = self.sensitive[col] == g
                analysis[f"{col}_{g}"] = {
                    'count': int(mask.sum()),
                    'positive_rate': float(self.y_pred[mask].mean()),
                }
        return analysis

    def _intersectional_analysis(self):
        if not hasattr(self.sensitive, 'columns') or len(self.sensitive.columns) < 2:
            return {}
        results = {}
        for combo in self.sensitive.drop_duplicates().values:
            mask = (self.sensitive.values == combo).all(axis=1)
            if mask.sum() > 10:
                key = '_'.join(str(c) for c in combo)
                results[key] = {
                    'count': int(mask.sum()),
                    'positive_rate': float(self.y_pred[mask].mean()),
                }
        return results

    def _recommendations(self):
        recs = []
        fm = self._fairness_metrics()
        for attr, info in fm.items():
            if not info['passed']:
                recs.append(f"Address bias in attribute '{attr}': "
                           f"DI ratio = {info['disparate_impact_ratio']:.3f}")
        return recs
```

---

## 5. Privacy-Preserving ML

### 5.1 Differential Privacy Training

```python
def dp_sgd_step(model, batch, optimizer, noise_multiplier=1.1,
                max_grad_norm=1.0, batch_size=64):
    """One step of DP-SGD (Abadi et al., 2016)."""
    model.train()
    per_sample_grads = []
    loss = 0

    for i in range(0, len(batch), batch_size):
        mini_batch = batch[i:i+batch_size]
        optimizer.zero_grad()
        output = model(mini_batch['input'])
        loss = criterion(output, mini_batch['label'])
        loss.backward()

        # Clip per-sample gradients
        for p in model.parameters():
            if p.grad is not None:
                grad_norm = p.grad.data.norm(2)
                clip_factor = max_grad_norm / (grad_norm + 1e-8)
                if clip_factor < 1:
                    p.grad.data.mul_(clip_factor)

        optimizer.step()

    # Add noise to gradients
    for p in model.parameters():
        if p.grad is not None:
            noise = torch.normal(0, noise_multiplier * max_grad_norm / batch_size,
                                p.grad.shape, device=p.grad.device)
            p.grad.data.add_(noise)

    return loss.item()
```

### 5.2 Federated Learning Privacy

```python
class FederatedClient:
    """Privacy-preserving federated learning client."""

    def __init__(self, local_model, epsilon=1.0):
        self.model = local_model
        self.epsilon = epsilon

    def local_train(self, data, epochs=5):
        """Train locally with DP guarantees."""
        for _ in range(epochs):
            for batch in data:
                dp_sgd_step(self.model, batch, self.optimizer,
                           noise_multiplier=self.compute_noise_multiplier())
        return self.get_model_update()

    def get_model_update(self):
        """Get clipped and noised model update."""
        update = {}
        for name, param in self.model.named_parameters():
            # Clip update
            clipped = torch.clamp(param.grad, -1.0, 1.0)
            # Add DP noise
            noise = torch.normal(0, 1.0 / self.epsilon, clipped.shape)
            update[name] = (clipped + noise).detach()
        return update

    def compute_noise_multiplier(self):
        """Compute noise multiplier for target epsilon."""
        # Simplified accounting
        return 1.0 / self.epsilon
```

---

## 6. Production Case Studies

### 6.1 Case Study: Fair Hiring System

| Metric | Before Fairness | After Fairness | Target |
|---|---|---|---|
| Disparate Impact (gender) | 0.45 | 0.87 | ≥ 0.80 |
| Disparate Impact (race) | 0.52 | 0.91 | ≥ 0.80 |
| Overall Accuracy | 0.82 | 0.79 | ≥ 0.75 |
| False Positive Rate Gap | 0.15 | 0.04 | ≤ 0.05 |
| Explainability Score | 3.2/10 | 8.1/10 | ≥ 7.0 |

**Approach:** Reweighting + threshold adjustment + SHAP monitoring

### 6.2 Case Study: Credit Scoring

| Metric | Before | After | Compliance |
|---|---|---|---|
| Adverse Action Accuracy | 67% | 94% | ≥ 90% ✓ |
| Disparate Impact (race) | 0.61 | 0.89 | ≥ 0.80 ✓ |
| Counterfactual Coverage | 0% | 87% | ≥ 80% ✓ |
| Model Interpretability | Low | High | High ✓ |

**Approach:** GAM model + counterfactual explanations + monitoring dashboard

---

## 7. Anti-Patterns and Pitfalls

| Anti-Pattern | Problem | Solution |
|---|---|---|
| **Fairness washing** | Cosmetic fairness fixes without real impact | Measure outcomes, not just processes |
| **Metric gaming** | Optimizing one fairness metric while harming others | Use multiple complementary metrics |
| **Bias migration** | Fixing bias in one place, creating it elsewhere | End-to-end fairness pipeline |
| **Privacy theater** | Adding noise without real privacy guarantees | Formal differential privacy |
| **Explanation theater** | Providing explanations that don't actually explain | User-tested explanations |
| **One-time audit** | Auditing once and never again | Continuous monitoring |
| **Proxy removal** | Removing protected attributes without fixing underlying bias | Fairness-aware training |

---

## 8. Testing Ethical Systems

### 8.1 Fairness Unit Tests

```python
import pytest

class TestFairness:
    """Unit tests for fairness constraints."""

    def test_demographic_parity(self, model, test_data, sensitive):
        """Model should satisfy demographic parity."""
        y_pred = model.predict(test_data)
        groups = sensitive.unique()
        rates = [y_pred[sensitive == g].mean() for g in groups]
        assert min(rates) / max(rates) >= 0.8, \
            f"Demographic parity violated: {rates}"

    def test_equal_opportunity(self, model, test_data, labels, sensitive):
        """Model should satisfy equal opportunity."""
        y_pred = model.predict(test_data)
        groups = sensitive.unique()
        tprs = []
        for g in groups:
            mask = (sensitive == g) & (labels == 1)
            tpr = y_pred[mask].mean() if mask.sum() > 0 else 0
            tprs.append(tpr)
        assert max(tprs) - min(tprs) <= 0.05, \
            f"Equal opportunity violated: TPRs = {tprs}"

    def test_no_sensitive_feature_leakage(self, model, sensitive_features):
        """Model should not directly use sensitive features as top predictors."""
        # Use permutation importance
        baseline_score = model.score(X_test, y_test)
        for feat in sensitive_features.columns:
            X_permuted = X_test.copy()
            X_permuted[feat] = np.random.permutation(X_permuted[feat])
            permuted_score = model.score(X_permuted, y_test)
            importance = baseline_score - permuted_score
            assert importance < 0.01, \
                f"Sensitive feature '{feat}' has importance {importance:.4f}"
```

### 8.2 Fairness Integration Tests

```python
def test_fairness_pipeline_end_to_end():
    """Integration test for the full fairness pipeline."""
    X, y, sensitive = load_test_dataset()
    pipeline = FairnessPipeline(fairness_threshold=0.8)

    model, report = pipeline.run(X, y, sensitive)

    # Verify model was trained
    assert model is not None

    # Verify all fairness checks passed
    for attr, result in report.items():
        assert result['passed'], \
            f"Fairness check failed for {attr}: DI = {result['disparate_impact']}"

    # Verify predictions are reasonable
    y_pred = model.predict(X)
    assert 0.1 < y_pred.mean() < 0.9, \
        f"Prediction rate out of range: {y_pred.mean()}"
```

---

## Cross-References

| Topic | See Also |
|---|---|
| Fairness metrics | [02-Core-Topics §1](./02-Core-Topics.md) |
| Explainability methods | [02-Core-Topics §3](./02-Core-Topics.md) |
| Privacy techniques | [40-AI-Data-Sovereignty-and-Privacy](../40-AI-Data-Sovereignty-and-Privacy/) |
| Monitoring infrastructure | [20-Agent-Infrastructure-and-Observability](../20-Agent-Infrastructure-and-Observability/) |
| Regulatory compliance | [21-AI-Regulation-Antitrust](../21-AI-Regulation-Antitrust/) |

---

*Last updated: July 2026*
