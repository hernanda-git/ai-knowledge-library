# Core Topics in AI Ethics and Responsible AI

> This document provides detailed exploration of the core ethical topics, frameworks, and technical approaches that underpin responsible AI development and deployment.

---

## Table of Contents

1. [Fairness Metrics and Measurement](#1-fairness-metrics-and-measurement)
2. [Bias Detection and Mitigation](#2-bias-detection-and-mitigation)
3. [Explainability and Interpretability](#3-explainability-and-interpretability)
4. [Transparency and Documentation](#4-transparency-and-documentation)
5. [Privacy-Preserving AI](#5-privacy-preserving-ai)
6. [Human-AI Interaction Ethics](#6-human-ai-interaction-ethics)
7. [Environmental Ethics](#7-environmental-ethics)
8. [Ethics of Autonomous Agents](#8-ethics-of-autonomous-agents)
9. [Global and Cultural Perspectives](#9-global-and-cultural-perspectives)
10. [Ethics in Specific Domains](#10-ethics-in-specific-domains)

---

## 1. Fairness Metrics and Measurement

### 1.1 Individual Fairness

Individual fairness requires that similar individuals receive similar outcomes.

**Distance metric design:**

```python
# Individual fairness framework
class IndividualFairness:
    def __init__(self, distance_metric):
        """
        distance_metric: function d(x1, x2) that measures
        similarity between individuals
        """
        self.distance_metric = distance_metric
    
    def compute_lipschitz_constant(self, model, X):
        """
        Compute the Lipschitz constant L such that:
        |f(x1) - f(x2)| <= L * d(x1, x2)
        
        Lower L means more individually fair.
        """
        max_ratio = 0
        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                d = self.distance_metric(X[i], X[j])
                if d > 0:
                    f_diff = abs(model.predict_proba(X[i])[1] - 
                               model.predict_proba(X[j])[1])
                    ratio = f_diff / d
                    max_ratio = max(max_ratio, ratio)
        
        return max_ratio
    
    def verify_fairness(self, model, X, threshold=1.0):
        """Verify individual fairness against threshold."""
        L = self.compute_lipschitz_constant(model, X)
        violations = []
        
        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                d = self.distance_metric(X[i], X[j])
                if d > 0:
                    f_diff = abs(model.predict_proba(X[i])[1] - 
                               model.predict_proba(X[j])[1])
                    if f_diff > threshold * d:
                        violations.append({
                            'i': i, 'j': j,
                            'distance': d,
                            'outcome_diff': f_diff,
                            'ratio': f_diff / d
                        })
        
        return {
            'lipschitz_constant': L,
            'threshold': threshold,
            'is_fair': L <= threshold,
            'num_violations': len(violations),
            'violations': violations
        }
```

### 1.2 Group Fairness Metrics

**Comprehensive fairness metrics computation:**

```python
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score
)

class FairnessMetrics:
    def __init__(self, y_true, y_pred, sensitive_features):
        self.y_true = y_true
        self.y_pred = y_pred
        self.sensitive_features = sensitive_features
    
    def compute_all_metrics(self):
        """Compute comprehensive fairness and accuracy metrics."""
        metrics = {}
        
        # Overall metrics
        metrics['overall'] = {
            'accuracy': accuracy_score(self.y_true, self.y_pred),
            'precision': precision_score(self.y_true, self.y_pred, 
                                       zero_division=0),
            'recall': recall_score(self.y_true, self.y_pred, 
                                  zero_division=0),
            'f1': f1_score(self.y_true, self.y_pred, zero_division=0),
        }
        
        # Fairness metrics
        metrics['fairness'] = {
            'demographic_parity_diff': demographic_parity_difference(
                self.y_true, self.y_pred, 
                sensitive_features=self.sensitive_features
            ),
            'equalized_odds_diff': equalized_odds_difference(
                self.y_true, self.y_pred, 
                sensitive_features=self.sensitive_features
            ),
        }
        
        # Group-specific metrics
        metric_frame = MetricFrame(
            metrics={
                'accuracy': accuracy_score,
                'positive_rate': lambda y, p: p.mean(),
                'true_positive_rate': recall_score,
                'false_positive_rate': lambda y, p: (
                    ((p == 1) & (y == 0)).sum() / max((y == 0).sum(), 1)
                ),
            },
            y_true=self.y_true,
            y_pred=self.y_pred,
            sensitive_features=self.sensitive_features,
        )
        
        metrics['by_group'] = metric_frame.by_group.to_dict()
        metrics['disparate_impact_ratio'] = (
            metric_frame.by_group['positive_rate'].min() /
            metric_frame.by_group['positive_rate'].max()
        )
        
        return metrics
    
    def generate_report(self):
        """Generate a human-readable fairness report."""
        metrics = self.compute_all_metrics()
        
        report = []
        report.append("=" * 60)
        report.append("FAIRNESS METRICS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Overall performance
        report.append("OVERALL PERFORMANCE:")
        for k, v in metrics['overall'].items():
            report.append(f"  {k}: {v:.4f}")
        report.append("")
        
        # Fairness metrics
        report.append("FAIRNESS METRICS:")
        for k, v in metrics['fairness'].items():
            report.append(f"  {k}: {v:.4f}")
        report.append(f"  disparate_impact_ratio: "
                     f"{metrics['disparate_impact_ratio']:.4f}")
        report.append("")
        
        # Group breakdown
        report.append("BY GROUP:")
        for group, group_metrics in metrics['by_group'].items():
            report.append(f"  Group {group}:")
            for k, v in group_metrics.items():
                report.append(f"    {k}: {v:.4f}")
        
        return "\n".join(report)
```

### 1.3 Intersectional Fairness

Intersectional fairness examines outcomes for combinations of protected attributes, not just individual attributes.

```python
import pandas as pd
from itertools import product

class IntersectionalFairness:
    def __init__(self, y_true, y_pred, sensitive_df):
        """
        sensitive_df: DataFrame with multiple protected attribute columns
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.sensitive_df = sensitive_df
    
    def compute_intersectional_metrics(self, metric_fn):
        """Compute a metric for each intersection of protected attributes."""
        # Create intersection groups
        intersection_groups = self.sensitive_df.apply(
            lambda row: tuple(row), axis=1
        )
        
        results = {}
        for group_name in intersection_groups.unique():
            mask = intersection_groups == group_name
            if mask.sum() > 0:
                results[group_name] = metric_fn(
                    self.y_true[mask], self.y_pred[mask]
                )
        
        return results
    
    def find_worst_group(self, metric_fn, minimize=True):
        """Find the intersection group with the worst metric value."""
        metrics = self.compute_intersectional_metrics(metric_fn)
        
        if not metrics:
            return None
        
        worst_group = min(metrics.items(), key=lambda x: x[1]) if minimize \
            else max(metrics.items(), key=lambda x: x[1])
        
        return {
            'group': worst_group[0],
            'metric_value': worst_group[1],
            'all_metrics': metrics,
            'disparity': max(metrics.values()) - min(metrics.values()),
        }
    
    def intersectional_report(self):
        """Generate a report of intersectional fairness metrics."""
        from sklearn.metrics import recall_score, precision_score
        
        report = {}
        
        for metric_name, metric_fn in [
            ('recall', recall_score),
            ('precision', precision_score),
            ('positive_rate', lambda y, p: p.mean()),
        ]:
            try:
                report[metric_name] = self.find_worst_group(
                    metric_fn, minimize=(metric_name != 'positive_rate')
                )
            except ValueError:
                report[metric_name] = None
        
        return report
```

### 1.4 Fairness Constraints in Optimization

```python
# Fairness-constrained optimization (conceptual)
def fairness_constrained_loss(y_true, y_pred, sensitive_features, 
                               fairness_weight=0.1):
    """
    Combine prediction loss with fairness penalty.
    
    Total loss = prediction_loss + fairness_weight * fairness_penalty
    """
    # Prediction loss (e.g., cross-entropy)
    pred_loss = cross_entropy(y_true, y_pred)
    
    # Fairness penalty (e.g., demographic parity violation)
    groups = sensitive_features.unique()
    group_rates = [
        y_pred[sensitive_features == g].mean() for g in groups
    ]
    overall_rate = y_pred.mean()
    fairness_penalty = sum((rate - overall_rate) ** 2 for rate in group_rates)
    
    # Combined loss
    total_loss = pred_loss + fairness_weight * fairness_penalty
    
    return total_loss
```

---

## 2. Bias Detection and Mitigation

### 2.1 Types of Bias in AI

| Bias Type | Source | Example | Detection Method |
|---|---|---|---|
| **Historical bias** | Reflects past societal inequities | Hiring model trained on biased historical data | Demographic analysis of training data |
| **Representation bias** | Underrepresentation of certain groups | Face recognition with limited diversity | Dataset composition analysis |
| **Measurement bias** | Inconsistent or biased measurements | Different diagnostic criteria across demographics | Outcome distribution analysis |
| **Aggregation bias** | One-size-fits-all model for diverse groups | Medical model that ignores age/gender differences | Subgroup performance analysis |
| **Evaluation bias** | Benchmarks that don't represent real-world use | Standard benchmarks missing certain populations | Benchmark vs. deployment comparison |
| **Deployment bias** | Model used in context different from design | Clinical model used for self-diagnosis | Use case vs. design comparison |
| **Selection bias** | Non-random sample selection | Online surveys excluding non-internet users | Sample vs. population comparison |
| **Label bias** | Biased human labeling | Subjective labels reflecting annotator bias | Inter-annotator agreement analysis |

### 2.2 Pre-Processing Mitigation

```python
# Pre-processing bias mitigation techniques
class PreprocessingMitigation:
    @staticmethod
    def reweight(X, y, sensitive_features):
        """
        Assign weights to training examples to achieve fairness constraints.
        Weight each (group, label) combination equally.
        """
        weights = np.ones(len(y))
        
        for group in sensitive_features.unique():
            for label in y.unique():
                mask = (sensitive_features == group) & (y == label)
                # Target: equal representation
                target_rate = 1.0 / (
                    len(sensitive_features.unique()) * len(np.unique(y))
                )
                actual_rate = mask.sum() / len(y)
                weights[mask] = target_rate / actual_rate
        
        return weights
    
    @staticmethod
    def resample(X, y, sensitive_features, strategy='oversample'):
        """
        Resample to achieve balanced representation.
        """
        if strategy == 'oversample':
            # Oversample minority groups
            target_count = max(
                pd.Series(y).value_counts().values
            )
            indices = []
            for group in sensitive_features.unique():
                for label in y.unique():
                    mask = (sensitive_features == group) & (y == label)
                    group_indices = np.where(mask)[0]
                    if len(group_indices) < target_count:
                        # Oversample with replacement
                        sampled = np.random.choice(
                            group_indices, 
                            size=target_count - len(group_indices),
                            replace=True
                        )
                        indices.extend(group_indices)
                        indices.extend(sampled)
                    else:
                        indices.extend(group_indices[:target_count])
        
        return X[indices], y[indices]
    
    @staticmethod
    def transform_features(X, sensitive_features):
        """
        Learn a fair representation that removes protected information.
        """
        # Conceptual: learn transformation that makes predictions
        # independent of sensitive features
        from sklearn.decomposition import PCA
        
        # Remove correlation with sensitive features
        # (simplified — real implementation would use adversarial learning)
        pca = PCA(n_components=X.shape[1] - len(sensitive_features.columns))
        X_fair = pca.fit_transform(X)
        
        return X_fair
```

### 2.3 In-Processing Mitigation

```python
# In-processing fairness constraints
class FairnessConstrainedModel:
    def __init__(self, base_model, fairness_constraint='dp', 
                 fairness_weight=0.1):
        self.base_model = base_model
        self.fairness_constraint = fairness_constraint
        self.fairness_weight = fairness_weight
    
    def fit(self, X, y, sensitive_features):
        """
        Train with fairness constraints via adversarial debiasing.
        """
        # Adversarial debiasing approach:
        # 1. Train predictor to maximize accuracy
        # 2. Train adversary to predict sensitive features from predictions
        # 3. Train predictor to minimize accuracy loss + maximize adversary loss
        
        for epoch in range(self.n_epochs):
            # Train predictor
            y_pred = self.base_model.predict_proba(X)[:, 1]
            
            # Compute prediction loss
            pred_loss = log_loss(y, y_pred)
            
            # Compute fairness penalty
            fairness_loss = self.compute_fairness_penalty(
                y_pred, sensitive_features
            )
            
            # Combined loss
            total_loss = pred_loss + self.fairness_weight * fairness_loss
            
            # Update model
            self.base_model.fit(X, y, sample_weight=self.compute_weights(
                y_pred, sensitive_features
            ))
        
        return self
    
    def compute_fairness_penalty(self, y_pred, sensitive_features):
        """Compute penalty based on fairness constraint."""
        if self.fairness_constraint == 'dp':
            # Demographic parity penalty
            groups = sensitive_features.unique()
            rates = [y_pred[sensitive_features == g].mean() for g in groups]
            return np.var(rates)
        
        elif self.fairness_constraint == 'eo':
            # Equalized odds penalty
            # (simplified — real implementation would handle TPR and FPR)
            groups = sensitive_features.unique()
            rates = [y_pred[sensitive_features == g].mean() for g in groups]
            return np.var(rates)
        
        return 0
```

### 2.4 Post-Processing Mitigation

```python
# Post-processing fairness adjustments
class PostprocessingMitigation:
    def __init__(self, model, threshold_method='equalized_odds'):
        self.model = model
        self.threshold_method = threshold_method
    
    def find_optimal_thresholds(self, X_val, y_val, sensitive_features):
        """
        Find group-specific thresholds that achieve fairness constraints.
        """
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        
        if self.threshold_method == 'equalized_odds':
            return self._equalized_odds_thresholds(
                y_val, y_pred_proba, sensitive_features
            )
        elif self.threshold_method == 'demographic_parity':
            return self._demographic_parity_thresholds(
                y_val, y_pred_proba, sensitive_features
            )
    
    def _equalized_odds_thresholds(self, y_true, y_pred_proba, 
                                    sensitive_features):
        """
        Find thresholds that equalize true positive rates
        and false positive rates across groups.
        """
        from sklearn.metrics import roc_curve
        
        thresholds = {}
        for group in sensitive_features.unique():
            mask = sensitive_features == group
            fpr, tpr, thresh = roc_curve(y_true[mask], y_pred_proba[mask])
            
            # Find threshold that achieves target TPR
            # (e.g., same as overall TPR)
            # This is a simplified version
            target_tpr = tpr[np.argmin(np.abs(thresh - 0.5))]
            optimal_idx = np.argmin(np.abs(tpr - target_tpr))
            thresholds[group] = thresh[optimal_idx]
        
        return thresholds
    
    def predict_with_fair_thresholds(self, X, sensitive_features):
        """Apply group-specific thresholds."""
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        thresholds = self.find_optimal_thresholds(
            X, self.model.predict(X), sensitive_features
        )
        
        y_pred = np.zeros(len(y_pred_proba))
        for group in sensitive_features.unique():
            mask = sensitive_features == group
            threshold = thresholds.get(group, 0.5)
            y_pred[mask] = (y_pred_proba[mask] >= threshold).astype(int)
        
        return y_pred
```

---

## 3. Explainability and Interpretability

### 3.1 SHAP (SHapley Additive exPlanations)

```python
import shap

class ShapExplainer:
    def __init__(self, model, X_background):
        self.explainer = shap.TreeExplainer(model)  # For tree models
        # self.explainer = shap.KernelExplainer(model.predict_proba, 
        #                                       X_background)  # For any model
        self.X_background = X_background
    
    def explain_prediction(self, x_instance):
        """Generate SHAP explanation for a single prediction."""
        shap_values = self.explainer.shap_values(x_instance)
        
        # For binary classification, shap_values is [class_0, class_1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Explain positive class
        
        # Create explanation summary
        feature_names = getattr(x_instance, 'columns', 
                               [f'feature_{i}' for i in range(len(x_instance))])
        
        explanation = {
            'base_value': self.explainer.expected_value[1] 
                if isinstance(self.explainer.expected_value, list) 
                else self.explainer.expected_value,
            'feature_importance': dict(zip(feature_names, shap_values)),
            'prediction': self.explainer.model.predict_proba(x_instance)[0][1],
        }
        
        return explanation
    
    def global_explanation(self, X, max_display=20):
        """Generate global feature importance explanation."""
        shap_values = self.explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Mean absolute SHAP values
        mean_shap = np.abs(shap_values).mean(axis=0)
        feature_names = getattr(X, 'columns', 
                               [f'feature_{i}' for i in range(X.shape[1])])
        
        global_importance = sorted(
            zip(feature_names, mean_shap),
            key=lambda x: x[1],
            reverse=True
        )[:max_display]
        
        return global_importance
    
    def fairness_aware_explanation(self, x_instance, sensitive_features):
        """
        Generate explanation that accounts for fairness concerns.
        """
        base_explanation = self.explanation(x_instance)
        
        # Check if sensitive features drive the prediction
        fairness_notes = []
        for attr in sensitive_features:
            if attr in base_explanation['feature_importance']:
                importance = abs(base_explanation['feature_importance'][attr])
                total_importance = sum(
                    abs(v) for v in 
                    base_explanation['feature_importance'].values()
                )
                pct = importance / total_importance * 100
                
                if pct > 5:  # Threshold for concern
                    fairness_notes.append({
                        'feature': attr,
                        'importance_pct': pct,
                        'concern': 'Sensitive feature has significant '
                                  'influence on prediction',
                        'recommendation': 'Investigate if this represents '
                                         'legitimate vs. biased relationship'
                    })
        
        base_explanation['fairness_notes'] = fairness_notes
        return base_explanation
```

### 3.2 LIME (Local Interpretable Model-agnostic Explanations)

```python
from lime.lime_tabular import LimeTabularExplainer

class LimeExplainer:
    def __init__(self, X_train, feature_names, class_names):
        self.explainer = LimeTabularExplainer(
            X_train.values,
            feature_names=feature_names,
            class_names=class_names,
            mode='classification'
        )
    
    def explain_instance(self, x_instance, num_features=10):
        """Generate LIME explanation for a single prediction."""
        explanation = self.explainer.explain_instance(
            x_instance.values,
            self.predict_fn,
            num_features=num_features
        )
        
        return {
            'feature_weights': dict(explanation.as_list()),
            'intercept': explanation.intercept[1],
            'local_pred': explanation.local_pred[1],
            'score': explanation.score,
        }
    
    def predict_fn(self, X):
        """Wrapper for model prediction (LIME expects numpy arrays)."""
        return self.model.predict_proba(X)
```

### 3.3 Counterfactual Explanations

```python
# Counterfactual explanation generation
class CounterfactualExplainer:
    def __init__(self, model, X_train, feature_names):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
    
    def generate_counterfactual(self, x_instance, desired_outcome, 
                                 max_changes=3):
        """
        Find minimal changes to flip the prediction.
        
        desired_outcome: 0 or 1 (the desired prediction)
        """
        original_prediction = self.model.predict(x_instance.reshape(1, -1))[0]
        
        if original_prediction == desired_outcome:
            return {'message': 'Already achieves desired outcome'}
        
        # Greedy search for minimal changes
        best_cf = None
        best_distance = float('inf')
        
        for n_changes in range(1, max_changes + 1):
            for combo in combinations(range(len(x_instance)), n_changes):
                cf = x_instance.copy()
                for idx in combo:
                    # Try different values for each feature
                    cf[idx] = self.find_best_value(
                        cf, idx, desired_outcome
                    )
                
                # Check if this achieves desired outcome
                if self.model.predict(cf.reshape(1, -1))[0] == desired_outcome:
                    distance = np.sum(np.abs(cf - x_instance))
                    if distance < best_distance:
                        best_distance = distance
                        best_cf = cf
        
        if best_cf is not None:
            changes = []
            for i, (orig, new) in enumerate(zip(x_instance, best_cf)):
                if orig != new:
                    changes.append({
                        'feature': self.feature_names[i],
                        'original': orig,
                        'counterfactual': new,
                        'change': new - orig,
                    })
            
            return {
                'counterfactual': best_cf,
                'distance': best_distance,
                'changes': changes,
                'n_changes': len(changes),
            }
        
        return {'message': 'No counterfactual found within constraints'}
    
    def find_best_value(self, x_instance, feature_idx, desired_outcome):
        """Find the best value for a feature to achieve desired outcome."""
        # Try values from training data distribution
        feature_values = np.percentile(
            self.X_train[:, feature_idx], 
            [10, 25, 50, 75, 90]
        )
        
        best_value = x_instance[feature_idx]
        best_distance = float('inf')
        
        for value in feature_values:
            cf = x_instance.copy()
            cf[feature_idx] = value
            
            if self.model.predict(cf.reshape(1, -1))[0] == desired_outcome:
                distance = abs(value - x_instance[feature_idx])
                if distance < best_distance:
                    best_distance = distance
                    best_value = value
        
        return best_value
```

### 3.4 Attention-Based Explanations (for Transformers)

```python
# Transformer attention visualization
class AttentionExplainer:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
    
    def get_attention_weights(self, text):
        """Extract attention weights from transformer model."""
        inputs = self.tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        # outputs.attentions is a tuple of (n_layers, batch, n_heads, seq, seq)
        attentions = outputs.attentions
        
        return {
            'input_tokens': self.tokenizer.convert_ids_to_tokens(
                inputs['input_ids'][0]
            ),
            'attention_weights': [layer[0].numpy() for layer in attentions],
        }
    
    def explain_with_attention(self, text, layer=-1, head=0):
        """
        Generate explanation using attention patterns.
        """
        attn_data = self.get_attention_weights(text)
        
        # Average attention across heads for the specified layer
        avg_attention = attn_data['attention_weights'][layer].mean(axis=0)
        
        # Get tokens with highest attention
        token_importance = avg_attention[head].sum(axis=0)
        
        # Sort by importance
        important_tokens = sorted(
            zip(attn_data['input_tokens'], token_importance),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'tokens': attn_data['input_tokens'],
            'attention_matrix': avg_attention,
            'important_tokens': important_tokens[:10],
        }
    
    def compare_attention(self, text1, text2):
        """Compare attention patterns between two texts."""
        attn1 = self.get_attention_weights(text1)
        attn2 = self.get_attention_weights(text2)
        
        # Compute attention difference
        diff = np.array(attn1['attention_weights'][-1]) - \
               np.array(attn2['attention_weights'][-1])
        
        return {
            'text1_tokens': attn1['input_tokens'],
            'text2_tokens': attn2['input_tokens'],
            'attention_difference': diff.mean(axis=(0, 1)),
        }
```

---

## 4. Transparency and Documentation

### 4.1 Model Cards

```python
class ModelCardGenerator:
    def __init__(self, model, training_data, metadata):
        self.model = model
        self.training_data = training_data
        self.metadata = metadata
    
    def generate_card(self):
        """Generate a comprehensive model card."""
        card = {
            'model_details': {
                'name': self.metadata.get('name', 'Unknown'),
                'version': self.metadata.get('version', '1.0'),
                'type': self.metadata.get('type', 'Unknown'),
                'training_date': self.metadata.get('training_date', 'Unknown'),
                'developers': self.metadata.get('developers', 'Unknown'),
                'license': self.metadata.get('license', 'Unknown'),
            },
            'intended_use': {
                'primary_use': self.metadata.get('primary_use', 'Unknown'),
                'out_of_scope_uses': self.metadata.get('out_of_scope', []),
                'users': self.metadata.get('intended_users', 'Unknown'),
                'deployment_context': self.metadata.get('context', 'Unknown'),
            },
            'training_data': {
                'source': self.training_data.get('source', 'Unknown'),
                'size': self.training_data.get('size', 'Unknown'),
                'demographics': self.training_data.get('demographics', {}),
                'preprocessing': self.training_data.get('preprocessing', 'None'),
            },
            'performance': self._compute_performance(),
            'ethical_considerations': self._ethical_analysis(),
            'limitations': self.metadata.get('limitations', []),
            'caveats': self.metadata.get('caveats', []),
        }
        
        return card
    
    def _compute_performance(self):
        """Compute model performance metrics."""
        # This would use actual test data in practice
        return {
            'overall': {},
            'by_subgroup': {},
            'fairness_metrics': {},
        }
    
    def _ethical_analysis(self):
        """Generate ethical analysis."""
        return {
            'bias_risks': self.metadata.get('bias_risks', []),
            'fairness_measures': self.metadata.get('fairness_measures', []),
            'transparency_features': self.metadata.get('transparency', []),
            'human_oversight': self.metadata.get('oversight', 'Required'),
        }
```

### 4.2 Datasheets for Datasets

```python
class DatasheetGenerator:
    def __init__(self, dataset_info):
        self.info = dataset_info
    
    def generate_datasheet(self):
        """Generate a datasheet following Gebru et al. framework."""
        return {
            'motivation': {
                'purpose': self.info.get('purpose', 'Unknown'),
                'creator': self.info.get('creator', 'Unknown'),
                'funder': self.info.get('funder', 'Unknown'),
                'version': self.info.get('version', 'Unknown'),
            },
            'composition': {
                'instances': self.info.get('n_instances', 'Unknown'),
                'features': self.info.get('features', []),
                'labeling': self.info.get('labeling_process', 'Unknown'),
                'sensitive_attributes': self.info.get('sensitive', []),
            },
            'collection': {
                'process': self.info.get('collection_process', 'Unknown'),
                'timeframe': self.info.get('timeframe', 'Unknown'),
                'ethical_review': self.info.get('ethical_review', 'Unknown'),
            },
            'preprocessing': {
                'cleaning': self.info.get('cleaning', 'Unknown'),
                'missing_data': self.info.get('missing_data', 'Unknown'),
                'transformations': self.info.get('transformations', []),
            },
            'uses': {
                'recommended': self.info.get('recommended_uses', []),
                'prohibited': self.info.get('prohibited_uses', []),
                'external_impact': self.info.get('external_impact', 'Unknown'),
            },
            'distribution': {
                'license': self.info.get('license', 'Unknown'),
                'accessibility': self.info.get('accessibility', 'Unknown'),
            },
        }
```

---

## 5. Privacy-Preserving AI

### 5.1 Differential Privacy

```python
# Differential privacy implementation
class DifferentialPrivacy:
    def __init__(self, epsilon=1.0, delta=1e-5):
        self.epsilon = epsilon  # Privacy budget
        self.delta = delta      # Failure probability
    
    def add_laplace_noise(self, value, sensitivity):
        """Add Laplace noise for (epsilon, 0)-differential privacy."""
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        return value + noise
    
    def add_gaussian_noise(self, value, sensitivity, delta):
        """Add Gaussian noise for (epsilon, delta)-differential privacy."""
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / self.epsilon
        noise = np.random.normal(0, sigma)
        return value + noise
    
    def private_mean(self, data, lower_bound, upper_bound):
        """Compute differentially private mean."""
        sensitivity = (upper_bound - lower_bound) / len(data)
        noisy_sum = sum(data) + self.add_laplace_noise(0, sensitivity * len(data))
        return noisy_sum / len(data)
    
    def private_count(self, data, predicate):
        """Compute differentially private count."""
        count = sum(1 for x in data if predicate(x))
        return count + self.add_laplace_noise(0, 1)
    
    def private_histogram(self, data, bins):
        """Compute differentially private histogram."""
        hist, _ = np.histogram(data, bins=bins)
        noisy_hist = hist + np.random.laplace(0, 1 / self.epsilon, len(hist))
        return noisy_hist
```

### 5.2 Federated Learning Ethics

```python
# Federated learning with fairness constraints
class FederatedFairness:
    def __init__(self, global_model, n_clients, fairness_constraint):
        self.global_model = global_model
        self.n_clients = n_clients
        self.fairness_constraint = fairness_constraint
    
    def federated_training_round(self, client_data, client_models):
        """
        Aggregate client models with fairness constraints.
        """
        # Standard federated averaging
        global_weights = self.federated_averaging(client_models)
        
        # Apply fairness constraint
        if self.fairness_constraint:
            global_weights = self.enforce_fairness(
                global_weights, client_data
            )
        
        return global_weights
    
    def federated_averaging(self, client_models):
        """Simple federated averaging."""
        # Weight by client data size
        total_samples = sum(m.n_samples for m in client_models)
        
        global_weights = {}
        for param in client_models[0].parameters():
            weighted_sum = sum(
                m.parameters()[param] * m.n_samples 
                for m in client_models
            ) / total_samples
            global_weights[param] = weighted_sum
        
        return global_weights
    
    def enforce_fairness(self, weights, client_data):
        """
        Adjust global model to be fair across client populations.
        """
        # Evaluate fairness across client groups
        fairness_scores = {}
        for client_id, data in client_data.items():
            # Compute group-specific metrics
            # This would be model-specific
            pass
        
        # Adjust weights to improve fairness
        # (simplified — real implementation would use optimization)
        return weights
```

---

## 6. Human-AI Interaction Ethics

### 6.1 Meaningful Human Control

**Principles for human-AI interaction:**

| Principle | Description | Implementation |
|---|---|---|
| **Transparency** | Humans understand AI's role | Clear labeling of AI-assisted decisions |
| **Controllability** | Humans can override AI | Kill switches, manual mode options |
| **Accountability** | Clear human responsibility | Designated human decision-maker |
| **Informed consent** | Humans know they're interacting with AI | Disclosure requirements |
| **Appropriate trust** | Neither over-trust nor under-trust | Calibrated confidence displays |

### 6.2 Avoiding Automation Bias

```python
# Automation bias detection and mitigation
class AutomationBiasMitigator:
    def __init__(self, ai_model, human_experts):
        self.ai_model = ai_model
        self.human_experts = human_experts
        self.bias_history = []
    
    def measure_automation_bias(self, decisions):
        """
        Measure the degree to which humans defer to AI recommendations.
        """
        human_overrides = sum(
            1 for d in decisions 
            if d['human_decision'] != d['ai_recommendation']
        )
        
        # Compare to expected override rate
        expected_rate = self.estimate_expected_override_rate(decisions)
        actual_rate = 1 - (human_overrides / len(decisions))
        
        bias_score = actual_rate - expected_rate
        
        self.bias_history.append({
            'timestamp': datetime.now(),
            'bias_score': bias_score,
            'human_overrides': human_overrides,
            'total_decisions': len(decisions),
        })
        
        return {
            'bias_score': bias_score,
            'interpretation': self.interpret_bias_score(bias_score),
            'recommendation': self.generate_recommendation(bias_score),
        }
    
    def interpret_bias_score(self, score):
        if score > 0.3:
            return "HIGH automation bias: Humans excessively defer to AI"
        elif score > 0.1:
            return "MODERATE automation bias: Some excessive AI deference"
        elif score > -0.1:
            return "LOW automation bias: Appropriate human-AI balance"
        else:
            return "UNDER-TRUST: Humans excessively override AI"
    
    def generate_recommendation(self, score):
        if score > 0.3:
            return [
                "Increase training on AI limitations",
                "Require human justification for agreeing with AI",
                "Implement periodic blind testing",
            ]
        elif score > 0.1:
            return [
                "Review AI recommendation confidence thresholds",
                "Provide calibration training",
            ]
        return ["Maintain current practices"]
```

---

## 7. Environmental Ethics

### 7.1 Carbon Footprint of AI

| Model | Training CO₂ (estimated) | Energy (MWh) | Equivalent flights NYC→LA |
|---|---|---|---|
| GPT-3 (175B) | ~552 tonnes | ~1,287 | ~60 |
| GPT-4 (~1.8T) | ~5,000+ tonnes | ~11,600+ | ~540 |
| Typical fine-tune | ~0.5–5 tonnes | ~1.2–12 | ~0.05–0.5 |

### 7.2 Sustainable AI Practices

```python
# Carbon-aware model training
class CarbonAwareTrainer:
    def __init__(self, model, carbon_intensity_api):
        self.model = model
        self.carbon_api = carbon_intensity_api
    
    def train_with_carbon_budget(self, training_data, carbon_budget_kg):
        """
        Train model while staying within carbon budget.
        """
        carbon_used = 0
        epoch = 0
        
        while carbon_used < carbon_budget_kg:
            # Check current carbon intensity
            intensity = self.carbon_api.get_current_intensity()
            
            if intensity > self.carbon_api.threshold:
                # Wait for cleaner energy
                print(f"High carbon intensity ({intensity}g/kWh), waiting...")
                time.sleep(self.carbon_api.wait_interval)
                continue
            
            # Train one epoch
            epoch_loss = self.train_epoch(training_data)
            epoch_carbon = self.estimate_epoch_carbon(intensity)
            carbon_used += epoch_carbon
            
            epoch += 1
            print(f"Epoch {epoch}: loss={epoch_loss:.4f}, "
                  f"carbon={carbon_used:.2f}kg CO₂")
        
        return {
            'epochs': epoch,
            'total_carbon_kg': carbon_used,
            'budget_remaining': carbon_budget_kg - carbon_used,
        }
    
    def estimate_epoch_carbon(self, carbon_intensity):
        """Estimate carbon for one epoch based on GPU hours and intensity."""
        # Estimate GPU hours for one epoch
        gpu_hours = self.estimate_gpu_hours()
        
        # Convert to carbon
        carbon_kg = gpu_hours * self.power_consumption_kw * carbon_intensity / 1000
        
        return carbon_kg
```

---

## 8. Ethics of Autonomous Agents

### 8.1 Agent Autonomy Spectrum

```
Agent Autonomy Spectrum
├── Level 0: Advisory (AI recommends, human decides)
├── Level 1: Supervised (AI acts, human monitors)
├── Level 2: Conditional (AI acts within bounds, human overrides)
├── Level 3: High autonomy (AI acts, human intervenes on exceptions)
├── Level 4: Full autonomy (AI acts independently)
└── Level 5: Collective autonomy (Multiple agents coordinate)
```

### 8.2 Ethical Framework for Agent Autonomy

| Autonomy Level | Ethical Requirements | Human Oversight |
|---|---|---|
| Level 0–1 | Standard fairness, transparency | Full human control |
| Level 2 | + Bounded action space, audit logging | Exception-based oversight |
| Level 3 | + Real-time monitoring, automatic kill switch | Periodic review + alerts |
| Level 4 | + Ethical constraints embedded in agent, comprehensive logging | Board-level oversight |
| Level 5 | + Inter-agent ethics, collective governance | System-level monitoring |

### 8.3 Agent-Specific Ethical Challenges

1. **Multi-agent coordination ethics:** When multiple agents interact, emergent behaviors may have ethical implications
2. **Long-horizon consequences:** Agents making sequences of decisions may have cumulative ethical impacts
3. **Accountability gaps:** When an agent acts autonomously, who is responsible for outcomes?
4. **Manipulation potential:** Agents interacting with humans may inadvertently or deliberately manipulate behavior
5. **Value drift:** Agents operating over long periods may drift from their original ethical constraints

---

## 9. Global and Cultural Perspectives

### 9.1 Cultural Variation in AI Ethics

| Dimension | Western Perspective | Eastern Perspective | African Perspective |
|---|---|---|---|
| **Individualism vs. Collectivism** | Individual privacy, autonomy | Community benefit, harmony | Ubuntu philosophy |
| **Fairness emphasis** | Individual fairness | Group harmony | Communal fairness |
| **Transparency** | Right to explanation | Trust in institutions | Community validation |
| **Accountability** | Individual liability | Collective responsibility | Elder/council oversight |

### 9.2 Global Ethics Framework Comparison

| Framework | Origin | Key Principles | Strengths | Limitations |
|---|---|---|---|---|
| OECD AI Principles | International | Inclusive growth, human values, transparency | Broad adoption | Non-binding |
| UNESCO Recommendation | Global | Proportionality, safety, sustainability, privacy | Universal scope | Non-binding |
| IEEE EAD | Technical | Human well-being, accountability, transparency | Technical depth | Adoption limited |
| EU AI Act | Europe | Risk-based, fundamental rights | Legally binding | Regional scope |
| China AI Ethics | China | Social harmony, national security | Clear enforcement | State-centric |

---

## 10. Ethics in Specific Domains

### 10.1 Healthcare AI Ethics

- **Informed consent:** Patients must know when AI is used in their care
- **Diagnostic accuracy:** AI must meet clinical-grade performance standards
- **Equitable access:** AI should reduce, not increase, healthcare disparities
- **Clinician oversight:** AI augments, not replaces, clinical judgment
- **Data privacy:** HIPAA + AI Act compliance for medical data

### 10.2 Financial AI Ethics

- **Fair lending:** No discrimination in credit decisions (ECOA, Fair Housing Act)
- **Explainability:** Adverse action explanations required by law
- **Model validation:** SR 11-7 compliance for model risk management
- **Consumer protection:** Transparent pricing and fee disclosures
- **Fraud prevention:** Balancing security with customer experience

### 10.3 Criminal Justice AI Ethics

- **Racial bias:** Risk assessment tools must be audited for racial fairness
- **Due process:** Defendants have the right to challenge AI-influenced decisions
- **Transparency:** Tool validation studies must be public
- **Human review:** AI should inform, not determine, justice outcomes
- **Proportionality:** AI monitoring must be proportional to the offense

### 10.4 Hiring AI Ethics

- **Adverse impact:** Four-fifths rule compliance for protected groups
- **Job-relatedness:** AI criteria must be validated as job-relevant
- **Disability accommodation:** AI must not disadvantage disabled candidates
- **Adverse action notices:** Required when AI contributes to rejection
- **Regular audits:** Ongoing monitoring for emerging biases

### 10.5 Education AI Ethics

- **Student privacy:** FERPA + GDPR compliance for student data
- **Equitable access:** AI should not widen the digital divide
- **Pedagogical integrity:** AI should enhance, not replace, learning
- **Age-appropriate design:** Child-specific protections for younger users
- **Transparency:** Students and parents should know when AI is used

---

## Cross-References

| Related Category | Connection |
|---|---|
| [01-Foundations](../01-Foundations/) | ML fundamentals underlying ethical techniques |
| [07-Emerging/02-AI-Safety](../07-Emerging/02-AI-Safety.md) | Complementary: alignment, catastrophic risks |
| [07-Emerging/03-AI-Governance](../07-Emerging/03-AI-Governance.md) | Complementary: regulation, policy compliance |
| [18-Agent-Security-and-Trust](../18-Agent-Security-and-Trust/) | Agent-specific trust mechanisms |
| [21-AI-Regulation-Antitrust](../21-AI-Regulation-Antitrust/) | Detailed regulatory requirements |
| [40-AI-Data-Sovereignty-and-Privacy](../40-AI-Data-Sovereignty-and-Privacy/) | Privacy techniques and data sovereignty |
| [43-AI-Data-Provenance-and-Content-Authenticity](../43-AI-Data-Provenance-and-Content-Authenticity/) | Data provenance for ethical accountability |
| [52-AI-Hallucination-Detection-and-Mitigation](../52-AI-Hallucination-Detection-and-Mitigation/) | Factual accuracy as ethical requirement |

---

*Last updated: July 2026*
*See also: [03-Technical-Deep-Dive](./03-Technical-Deep-Dive.md) for implementation details*
