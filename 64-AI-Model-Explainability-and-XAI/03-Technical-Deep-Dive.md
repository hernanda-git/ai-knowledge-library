# AI Model Explainability and XAI at Scale: Technical Deep Dive

> **Description:** Implementation-focused guide covering production XAI architectures, code examples, performance optimization, and enterprise deployment patterns. Includes hands-on implementations for SHAP, LIME, attention visualization, and agent-level explainability.

---

## Table of Contents

1. [Production XAI Architecture](#1-production-xai-architecture)
2. [SHAP Implementation Deep Dive](#2-shap-implementation-deep-dive)
3. [LIME Production Patterns](#3-lime-production-patterns)
4. [Attention Visualization Implementation](#4-attention-visualization-implementation)
5. [Counterfactual Explanation Pipeline](#5-counterfactual-explanation-pipeline)
6. [Agent-Level Explainability](#6-agent-level-explainability)
7. [RAG Explainability](#7-rag-explainability)
8. [LLM Explainability at Scale](#8-llm-explainability-at-scale)
9. [Performance Optimization](#9-performance-optimization)
10. [Testing and Validation](#10-testing-and-validation)
11. [Infrastructure and Deployment](#11-infrastructure-and-deployment)

---

## 1. Production XAI Architecture

### 1.1 XAI Microservice Pattern

```
┌─────────────────────────────────────────────────────────────────┐
│                      XAI Microservice                           │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ Explanation│  │ Feature  │  │ Counter- │  │ Attention    │  │
│  │ Engine    │  │ Attribution│  │ factual  │  │ Visualizer   │  │
│  │ (SHAP)    │  │ Module   │  │ Generator│  │ (for LLMs)   │  │
│  └─────┬────┘  └─────┬────┘  └────┬─────┘  └──────┬───────┘  │
│        │              │            │                │           │
│  ┌─────┴──────────────┴────────────┴────────────────┴───────┐  │
│  │              Unified Explanation Interface                │  │
│  └─────────────────────────┬────────────────────────────────┘  │
│                            │                                   │
│  ┌─────────────────────────┴────────────────────────────────┐  │
│  │                 Explanation Store (Redis/PostgreSQL)       │  │
│  │               Cache · Version · Audit · Feedback          │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Explanation Data Model

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class ExplanationType(Enum):
    SHAP = "shap"
    LIME = "lime"
    COUNTERFACTUAL = "counterfactual"
    ATTENTION = "attention"
    ANCHOR = "anchor"
    GRADCAM = "gradcam"
    INTEGRATED_GRADIENTS = "integrated_gradients"

class AudienceType(Enum):
    TECHNICAL = "technical"         # Data scientists
    DOMAIN_EXPERT = "domain_expert" # Doctors, loan officers
    END_USER = "end_user"           # Consumers, patients
    REGULATOR = "regulator"         # Compliance officers

@dataclass
class FeatureContribution:
    feature_name: str
    value: Any
    contribution: float             # SHAP value, LIME weight, etc.
    rank: int                       # Importance rank (1 = most important)
    direction: str                  # "positive" or "negative"
    confidence: float               # Confidence in this attribution

@dataclass
class CounterfactualSuggestion:
    feature_name: str
    original_value: Any
    suggested_value: Any
    required_change: str            # Human-readable change description
    feasibility_score: float        # 0-1, how realistic this change is

@dataclass
class Explanation:
    explanation_id: str
    model_id: str
    prediction_id: str
    explanation_type: ExplanationType
    audience: AudienceType
    
    # Core explanation data
    prediction: Any
    base_value: float               # Population average prediction
    feature_contributions: List[FeatureContribution]
    counterfactuals: List[CounterfactualSuggestion] = field(default_factory=list)
    
    # Metadata
    confidence: float               # Model confidence in prediction
    explanation_confidence: float   # Confidence in explanation accuracy
    computation_time_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # For LLMs
    attention_data: Optional[Dict] = None
    reasoning_chain: Optional[List[str]] = None
    tool_calls: Optional[List[Dict]] = None
    
    # Audit trail
    model_version: str = ""
    data_hash: str = ""             # Hash of input data for reproducibility
    
    def to_regulatory_report(self) -> Dict:
        """Generate EU AI Act Article 13 compliant report."""
        return {
            "system_identification": self.model_id,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "explanation_type": self.explanation_type.value,
            "feature_contributions": [
                {
                    "feature": fc.feature_name,
                    "value": str(fc.value),
                    "impact": fc.contribution,
                    "direction": fc.direction
                }
                for fc in self.feature_contributions
            ],
            "counterfactuals": [
                {
                    "feature": cf.feature_name,
                    "current": str(cf.original_value),
                    "suggested": str(cf.suggested_value),
                    "feasibility": cf.feasibility_score
                }
                for cf in self.counterfactuals
            ],
            "compliance_metadata": {
                "explanation_generated": self.timestamp.isoformat(),
                "model_version": self.model_version,
                "data_hash": self.data_hash,
                "computation_time_ms": self.computation_time_ms
            }
        }
```

### 1.3 Explanation API Design

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="XAI Service", version="1.0.0")

class ExplanationRequest(BaseModel):
    model_id: str
    input_data: Dict[str, Any]
    explanation_type: str = "shap"
    audience: str = "technical"
    n_counterfactuals: int = 3
    include_attention: bool = False

class ExplanationResponse(BaseModel):
    explanation_id: str
    prediction: Any
    confidence: float
    feature_contributions: List[Dict]
    counterfactuals: List[Dict]
    reasoning_chain: Optional[List[str]] = None
    attention_data: Optional[Dict] = None
    computation_time_ms: float
    compliance_report: Optional[Dict] = None

@app.post("/explain", response_model=ExplanationResponse)
async def explain_prediction(request: ExplanationRequest):
    """
    Generate explanation for a model prediction.
    
    Supports multiple explanation types:
    - shap: SHAP values (feature attribution)
    - lime: LIME (local approximation)
    - counterfactual: What-if analysis
    - attention: Transformer attention visualization
    - anchor: Rule-based explanations
    """
    try:
        # Load model
        model = await load_model(request.model_id)
        
        # Generate prediction
        prediction = model.predict(request.input_data)
        
        # Generate explanation based on type
        explainer = get_explainer(request.explanation_type)
        explanation = await explainer.explain(
            model=model,
            instance=request.input_data,
            prediction=prediction,
            audience=request.audience
        )
        
        # Generate counterfactuals if requested
        if request.n_counterfactuals > 0:
            cf_generator = CounterfactualGenerator()
            counterfactuals = await cf_generator.generate(
                model=model,
                instance=request.input_data,
                prediction=prediction,
                n_suggestions=request.n_counterfactuals
            )
            explanation.counterfactuals = counterfactuals
        
        # Generate compliance report for regulated industries
        compliance_report = None
        if request.audience == "regulator":
            compliance_report = explanation.to_regulatory_report()
        
        return ExplanationResponse(
            explanation_id=explanation.explanation_id,
            prediction=explanation.prediction,
            confidence=explanation.confidence,
            feature_contributions=[
                {
                    "feature": fc.feature_name,
                    "value": fc.value,
                    "contribution": fc.contribution,
                    "rank": fc.rank,
                    "direction": fc.direction
                }
                for fc in explanation.feature_contributions
            ],
            counterfactuals=[
                {
                    "feature": cf.feature_name,
                    "original": cf.original_value,
                    "suggested": cf.suggested_value,
                    "change": cf.required_change,
                    "feasibility": cf.feasibility_score
                }
                for cf in explanation.counterfactuals
            ],
            reasoning_chain=explanation.reasoning_chain,
            attention_data=explanation.attention_data,
            computation_time_ms=explanation.computation_time_ms,
            compliance_report=compliance_report
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 2. SHAP Implementation Deep Dive

### 2.1 TreeSHAP for XGBoost (Production-Ready)

```python
import xgboost as xgb
import shap
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

class ProductionTreeExplainer:
    """
    Production-grade TreeSHAP explainer with caching,
    batching, and monitoring.
    """
    
    def __init__(self, model: xgb.XGBClassifier, 
                 feature_names: List[str],
                 background_data: Optional[pd.DataFrame] = None):
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(
            model,
            data=background_data,
            feature_perturbation="interventional"
        )
        self._cache = {}
        
    def explain_instance(self, instance: pd.DataFrame,
                         cache_key: Optional[str] = None) -> Dict:
        """
        Generate SHAP explanation for a single instance.
        
        Returns structured explanation with feature contributions,
        confidence intervals, and visualizations.
        """
        # Check cache
        if cache_key and cache_key in self._cache:
            return self._cache[cache_key]
        
        import time
        start_time = time.time()
        
        # Compute SHAP values
        shap_values = self.explainer.shap_values(instance)
        
        # For binary classification, shap_values is array of shape
        # (n_samples, n_features) for class 1
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Class 1 (positive class)
        
        # Base value (expected prediction)
        base_value = self.explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[1]
        
        # Prediction
        prediction = self.model.predict_proba(instance)[0]
        
        # Build feature contributions
        contributions = []
        for i, feature_name in enumerate(self.feature_names):
            contributions.append({
                "feature": feature_name,
                "value": float(instance.iloc[0, i]),
                "shap_value": float(shap_values[0, i]),
                "abs_impact": float(abs(shap_values[0, i])),
                "direction": "positive" if shap_values[0, i] > 0 else "negative"
            })
        
        # Sort by absolute impact
        contributions.sort(key=lambda x: x["abs_impact"], reverse=True)
        
        # Add rank
        for i, c in enumerate(contributions):
            c["rank"] = i + 1
        
        computation_time = (time.time() - start_time) * 1000
        
        result = {
            "prediction": float(prediction[1]),
            "prediction_class": int(np.argmax(prediction)),
            "base_value": float(base_value),
            "contributions": contributions,
            "computation_time_ms": round(computation_time, 2),
            "top_features": [c["feature"] for c in contributions[:5]],
            "summary": self._generate_summary(contributions, prediction)
        }
        
        # Cache result
        if cache_key:
            self._cache[cache_key] = result
        
        return result
    
    def explain_batch(self, instances: pd.DataFrame,
                      batch_size: int = 100) -> List[Dict]:
        """
        Explain multiple instances in batches for efficiency.
        """
        results = []
        for i in range(0, len(instances), batch_size):
            batch = instances.iloc[i:i+batch_size]
            shap_values = self.explainer.shap_values(batch)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            for j in range(len(batch)):
                instance_shap = shap_values[j:j+1]
                contributions = []
                for k, feature_name in enumerate(self.feature_names):
                    contributions.append({
                        "feature": feature_name,
                        "value": float(batch.iloc[j, k]),
                        "shap_value": float(instance_shap[0, k]),
                        "abs_impact": float(abs(instance_shap[0, k])),
                        "direction": "positive" if instance_shap[0, k] > 0 else "negative"
                    })
                contributions.sort(key=lambda x: x["abs_impact"], reverse=True)
                for idx, c in enumerate(contributions):
                    c["rank"] = idx + 1
                
                results.append({
                    "instance_index": i + j,
                    "contributions": contributions[:10]  # Top 10 only
                })
        
        return results
    
    def global_explanation(self, data: pd.DataFrame) -> Dict:
        """
        Generate global model explanation.
        """
        shap_values = self.explainer.shap_values(data)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        global_importance = []
        for i, feature_name in enumerate(self.feature_names):
            global_importance.append({
                "feature": feature_name,
                "mean_abs_shap": float(mean_abs_shap[i]),
                "mean_shap": float(np.mean(shap_values[:, i])),
                "std_shap": float(np.std(shap_values[:, i]))
            })
        
        global_importance.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
        
        return {
            "global_importance": global_importance,
            "n_samples": len(data),
            "top_5_features": [g["feature"] for g in global_importance[:5]],
            "feature_interaction_matrix": self._compute_interactions(data, shap_values)
        }
    
    def _generate_summary(self, contributions: List[Dict], 
                          prediction: np.ndarray) -> str:
        """Generate human-readable explanation summary."""
        top_3 = contributions[:3]
        direction = "approved" if prediction[1] > 0.5 else "denied"
        
        summary_parts = [
            f"Prediction: {direction} (confidence: {prediction[1]:.1%})",
            "Top contributing factors:",
        ]
        
        for c in top_3:
            sign = "+" if c["direction"] == "positive" else "-"
            summary_parts.append(
                f"  {c['rank']}. {c['feature']}: {sign}{abs(c['shap_value']):.3f} "
                f"(value: {c['value']:.2f})"
            )
        
        return "\n".join(summary_parts)
    
    def _compute_interactions(self, data: pd.DataFrame,
                               shap_values: np.ndarray) -> Dict:
        """Compute top feature interactions."""
        n_features = len(self.feature_names)
        interaction_matrix = np.zeros((n_features, n_features))
        
        # Simple approximation using correlation of SHAP values
        for i in range(n_features):
            for j in range(i+1, n_features):
                corr = np.corrcoef(shap_values[:, i], shap_values[:, j])[0, 1]
                interaction_matrix[i, j] = corr
                interaction_matrix[j, i] = corr
        
        # Find top interactions
        interactions = []
        for i in range(n_features):
            for j in range(i+1, n_features):
                if abs(interaction_matrix[i, j]) > 0.3:
                    interactions.append({
                        "feature_1": self.feature_names[i],
                        "feature_2": self.feature_names[j],
                        "interaction_strength": float(interaction_matrix[i, j])
                    })
        
        interactions.sort(key=lambda x: abs(x["interaction_strength"]), reverse=True)
        return interactions[:10]


# Usage example
if __name__ == "__main__":
    # Train model
    X_train, X_test, y_train, y_test = load_loan_data()
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='binary:logistic'
    )
    model.fit(X_train, y_train)
    
    # Initialize explainer
    explainer = ProductionTreeExplainer(
        model=model,
        feature_names=list(X_train.columns),
        background_data=X_train.sample(100)
    )
    
    # Explain single instance
    explanation = explainer.explain_instance(
        X_test.iloc[[0]],
        cache_key=f"instance_{0}"
    )
    print(explanation["summary"])
    
    # Global explanation
    global_expl = explainer.global_explanation(X_test)
    print(f"Top 5 features: {global_expl['top_5_features']}")
```

### 2.2 KernelSHAP for Any Model

```python
class KernelSHAPExplainer:
    """
    Model-agnostic SHAP explainer using KernelSHAP.
    For models where TreeSHAP doesn't apply.
    """
    
    def __init__(self, model_predict_fn, background_data: np.ndarray,
                 n_samples: int = 100):
        """
        Args:
            model_predict_fn: Function that takes array and returns predictions
            background_data: Reference dataset for baseline
            n_samples: Number of coalition samples for approximation
        """
        self.predict_fn = model_predict_fn
        self.background = background_data
        self.n_samples = n_samples
        
        # Compute background mean (base value)
        self.base_value = float(np.mean(self.predict_fn(background_data)))
    
    def explain(self, instance: np.ndarray) -> Dict:
        """
        Compute KernelSHAP values for an instance.
        
        Uses sampling-based approximation for efficiency.
        """
        n_features = len(instance)
        
        # Generate coalition mask samples
        masks = np.random.randint(0, 2, size=(self.n_samples, n_features))
        masks[0] = np.ones(n_features)  # Include all features
        masks[1] = np.zeros(n_features)  # Include no features
        
        # Compute predictions for each coalition
        predictions = np.zeros(self.n_samples)
        for i in range(self.n_samples):
            masked_instance = np.where(masks[i], instance, self.background.mean(axis=0))
            predictions[i] = self.predict_fn(masked_instance.reshape(1, -1))[0]
        
        # Weight by kernel function
        kernel_weights = np.zeros(self.n_samples)
        for i in range(self.n_samples):
            n_present = masks[i].sum()
            if n_present == 0 or n_present == n_features:
                kernel_weights[i] = 1e6  # High weight for empty/full coalitions
            else:
                kernel_weights[i] = (n_features - 1) / (
                    comb(n_features, n_present) * n_present * (n_features - n_present)
                )
        
        # Weighted linear regression to estimate SHAP values
        from sklearn.linear_model import LinearRegression
        
        reg = LinearRegression()
        reg.fit(masks, predictions, sample_weight=kernel_weights)
        
        shap_values = reg.coef_
        
        # Build result
        contributions = []
        for i in range(n_features):
            contributions.append({
                "feature_index": i,
                "shap_value": float(shap_values[i]),
                "abs_impact": float(abs(shap_values[i]))
            })
        
        contributions.sort(key=lambda x: x["abs_impact"], reverse=True)
        
        prediction = float(self.predict_fn(instance.reshape(1, -1))[0])
        
        return {
            "prediction": prediction,
            "base_value": self.base_value,
            "shap_values": shap_values.tolist(),
            "contributions": contributions,
            "r_squared": reg.score(masks, predictions, sample_weight=kernel_weights)
        }
```

---

## 3. LIME Production Patterns

### 3.1 Stable LIME Implementation

```python
import lime
import lime.lime_tabular
import numpy as np
from typing import List, Dict

class StableLIMEExplainer:
    """
    LIME implementation with stability improvements:
    - Multiple runs aggregated
    - Confidence intervals
    - Instability detection
    """
    
    def __init__(self, model_predict_fn, training_data: np.ndarray,
                 feature_names: List[str], n_features: int = 10):
        self.predict_fn = model_predict_fn
        self.feature_names = feature_names
        self.n_features = n_features
        
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data,
            feature_names=feature_names,
            mode='classification',
            discretize_continuous=True,
            random_state=42
        )
    
    def explain_stable(self, instance: np.ndarray, 
                       n_runs: int = 10,
                       confidence_level: float = 0.95) -> Dict:
        """
        Generate stable LIME explanation by aggregating multiple runs.
        
        Returns explanation with confidence intervals and stability metrics.
        """
        all_explanations = []
        
        for run in range(n_runs):
            # Each run uses different random perturbations
            self.explainer.random_state = run
            exp = self.explainer.explain_instance(
                instance,
                self.predict_fn,
                num_features=self.n_features,
                num_samples=5000
            )
            
            # Extract feature weights
            weights = dict(exp.as_list())
            all_explanations.append(weights)
        
        # Aggregate results
        aggregated = {}
        for feature in self.feature_names:
            values = [exp.get(feature, 0.0) for exp in all_explanations]
            aggregated[feature] = {
                "mean_weight": np.mean(values),
                "std_weight": np.std(values),
                "ci_lower": np.percentile(values, (1-confidence_level)/2 * 100),
                "ci_upper": np.percentile(values, (1+confidence_level)/2 * 100),
                "stability_score": 1.0 - min(np.std(values) / (abs(np.mean(values)) + 1e-10), 1.0),
                "all_values": values
            }
        
        # Rank by mean absolute weight
        ranked = sorted(aggregated.items(), 
                       key=lambda x: abs(x[1]["mean_weight"]), 
                       reverse=True)
        
        # Stability assessment
        avg_stability = np.mean([v["stability_score"] for v in aggregated.values()])
        
        return {
            "aggregated_explanation": {
                feature: {
                    "weight": data["mean_weight"],
                    "confidence_interval": [data["ci_lower"], data["ci_upper"]],
                    "stability": data["stability_score"]
                }
                for feature, data in ranked
            },
            "stability_metrics": {
                "average_stability": avg_stability,
                "stable_features": sum(1 for _, v in aggregated.items() if v["stability_score"] > 0.8),
                "unstable_features": sum(1 for _, v in aggregated.items() if v["stability_score"] < 0.5),
                "n_runs": n_runs
            },
            "recommendation": self._get_recommendation(avg_stability)
        }
    
    def _get_recommendation(self, avg_stability: float) -> str:
        if avg_stability > 0.8:
            return "LIME explanation is stable. Safe to present to users."
        elif avg_stability > 0.5:
            return "LIME explanation has moderate stability. Consider using SHAP instead."
        else:
            return "LIME explanation is unstable. Use SHAP or anchors for this instance."


# Usage
explainer = StableLIMEExplainer(
    model_predict_fn=model.predict_proba,
    training_data=X_train.values,
    feature_names=list(X_train.columns)
)

result = explainer.explain_stable(
    instance=X_test.iloc[0].values,
    n_runs=10
)

if result["stability_metrics"]["average_stability"] > 0.5:
    print("Stable explanation:")
    for feature, data in list(result["aggregated_explanation"].items())[:5]:
        print(f"  {feature}: {data['weight']:.3f} ± {data['confidence_interval']}")
else:
    print("Unstable explanation, falling back to SHAP")
```

---

## 4. Attention Visualization Implementation

### 4.1 Transformer Attention Extraction

```python
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModel, AutoTokenizer

class AttentionVisualizer:
    """
    Extract and visualize attention patterns from transformer models.
    """
    
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name, 
            output_attentions=True
        )
        self.model.eval()
    
    def get_attention(self, text: str) -> Dict:
        """
        Extract attention weights for all layers and heads.
        
        Returns structured attention data for visualization and analysis.
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # attentions: tuple of (n_layers) tensors, each (1, n_heads, seq_len, seq_len)
        attentions = outputs.attentions
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        # Convert to numpy
        attn_np = [a.squeeze(0).numpy() for a in attentions]  # Remove batch dim
        
        return {
            "tokens": tokens,
            "attentions": attn_np,  # List of (n_heads, seq_len, seq_len) arrays
            "n_layers": len(attn_np),
            "n_heads": attn_np[0].shape[0],
            "seq_len": len(tokens)
        }
    
    def attention_rollout(self, attention_data: Dict) -> np.ndarray:
        """
        Compute attention rollout — aggregate attention across layers.
        
        This gives a better approximation of feature importance than
        raw attention weights.
        """
        attentions = attention_data["attentions"]
        seq_len = attention_data["seq_len"]
        
        # Start with identity matrix (residual connections)
        rollout = np.eye(seq_len)
        
        for layer_attn in attentions:
            # Average across heads
            avg_attn = layer_attn.mean(axis=0)  # (seq_len, seq_len)
            
            # Add residual connection (50/50 split)
            augmented = 0.5 * avg_attn + 0.5 * np.eye(seq_len)
            
            # Normalize rows
            augmented = augmented / augmented.sum(axis=-1, keepdim=True)
            
            # Multiply with accumulated rollout
            rollout = augmented @ rollout
        
        return rollout
    
    def visualize_attention(self, attention_data: Dict, 
                           layer: int = -1, head: int = 0,
                           save_path: str = None):
        """
        Create attention heatmap visualization.
        """
        tokens = attention_data["tokens"]
        attn = attention_data["attentions"][layer][head]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            attn,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='Blues',
            ax=ax
        )
        ax.set_title(f"Layer {layer}, Head {head}")
        ax.set_xlabel("Key")
        ax.set_ylabel("Query")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()
    
    def visualize_rollout(self, attention_data: Dict, save_path: str = None):
        """
        Visualize attention rollout — aggregated importance across layers.
        """
        tokens = attention_data["tokens"]
        rollout = self.attention_rollout(attention_data)
        
        # Get importance of each token (from CLS token perspective)
        importance = rollout[0, :]  # CLS token's attention to all tokens
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(tokens, importance, color='steelblue')
        ax.set_title("Token Importance (Attention Rollout from CLS)")
        ax.set_ylabel("Importance Score")
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        
        # Highlight top 3
        top_3_idx = importance.argsort()[-3:]
        for idx in top_3_idx:
            bars[idx].set_color('darkred')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()
    
    def identify_attention_patterns(self, attention_data: Dict) -> Dict:
        """
        Identify common attention patterns in the model.
        """
        tokens = attention_data["tokens"]
        attentions = attention_data["attentions"]
        
        patterns = {
            "diagonal_attention": 0,    # Token attending to itself
            "previous_token": 0,        # Attending to previous token
            "next_token": 0,            # Attending to next token
            "delimiter_attention": 0,   # Attending to [SEP], [CLS], etc.
        }
        
        for layer_attn in attentions:
            for head_attn in layer_attn:
                # Diagonal (self-attention)
                diag = np.diag(head_attn).mean()
                patterns["diagonal_attention"] += diag
                
                # Previous token
                prev = np.diag(head_attn, k=-1).mean()
                patterns["previous_token"] += prev
                
                # Next token
                nxt = np.diag(head_attn, k=1).mean()
                patterns["next_token"] += nxt
        
        # Normalize
        n_layers = len(attentions)
        n_heads = attentions[0].shape[0]
        total = n_layers * n_heads
        
        for key in patterns:
            patterns[key] /= total
        
        return patterns


# Usage for LLM explainability
def explain_llm_prediction(text: str, model_name: str = "bert-base-uncased"):
    """
    Generate explanation for LLM prediction using attention analysis.
    """
    viz = AttentionVisualizer(model_name)
    attention_data = viz.get_attention(text)
    
    # Get rollout importance
    rollout = viz.attention_rollout(attention_data)
    importance = rollout[0, :]
    
    tokens = attention_data["tokens"]
    token_importance = list(zip(tokens, importance))
    token_importance.sort(key=lambda x: x[1], reverse=True)
    
    # Identify patterns
    patterns = viz.identify_attention_patterns(attention_data)
    
    return {
        "text": text,
        "top_tokens": token_importance[:5],
        "attention_patterns": patterns,
        "n_layers": attention_data["n_layers"],
        "n_heads": attention_data["n_heads"]
    }
```

---

## 5. Counterfactual Explanation Pipeline

### 5.1 DiCE Implementation

```python
import dice_ml
import pandas as pd
import numpy as np
from typing import List, Dict

class CounterfactualPipeline:
    """
    Production counterfactual explanation pipeline using DiCE.
    """
    
    def __init__(self, model, training_data: pd.DataFrame,
                 target_column: str, 
                 continuous_features: List[str] = None,
                 immutable_features: List[str] = None):
        """
        Args:
            model: Trained sklearn/xgboost model
            training_data: Training DataFrame
            target_column: Name of target column
            continuous_features: List of continuous feature names
            immutable_features: Features that cannot be changed (e.g., age, race)
        """
        self.model = model
        self.target_column = target_column
        self.immutable_features = immutable_features or []
        
        # Prepare data for DiCE
        data = dice_ml.Data(
            dataframe=training_data,
            continuous_features=continuous_features or [],
            outcome_name=target_column
        )
        
        ml_model = dice_ml.Model(model=model, backend="sklearn")
        
        self.explainer = dice_ml.Dice(
            data, 
            ml_model,
            method="random"  # or "gradient" for deep learning
        )
    
    def generate_counterfactuals(self, instance: pd.DataFrame,
                                  n_counterfactuals: int = 3,
                                  desired_class: str = "opposite",
                                  features_to_vary: List[str] = None,
                                  permitted_range: Dict = None) -> Dict:
        """
        Generate counterfactual explanations for an instance.
        
        Returns structured counterfactuals with feasibility analysis.
        """
        # Generate counterfactuals
        counterfactuals = self.explainer.generate_counterfactuals(
            instance,
            total_CFs=n_counterfactuals,
            desired_class=desired_class,
            features_to_vary=features_to_vary,
            permitted_range=permitted_range
        )
        
        # Parse results
        cf_df = counterfactuals.cf_examples_list[0].final_cfs_df
        
        original_prediction = self.model.predict_proba(instance)[0][1]
        
        results = {
            "original": {
                "instance": instance.iloc[0].to_dict(),
                "prediction": float(original_prediction),
                "prediction_class": int(np.argmax([1-original_prediction, original_prediction]))
            },
            "counterfactuals": [],
            "feasibility_analysis": {}
        }
        
        for idx, row in cf_df.iterrows():
            cf_instance = row.drop(self.target_column)
            
            # Find changed features
            changes = {}
            for feature in instance.columns:
                if feature == self.target_column:
                    continue
                orig_val = instance.iloc[0][feature]
                cf_val = cf_instance[feature]
                
                if not np.isclose(orig_val, cf_val, rtol=1e-3):
                    changes[feature] = {
                        "original": orig_val,
                        "counterfactual": cf_val,
                        "change": cf_val - orig_val,
                        "feasible": feature not in self.immutable_features,
                        "magnitude": abs(cf_val - orig_val) / (abs(orig_val) + 1e-10)
                    }
            
            cf_prediction = self.model.predict_proba(
                cf_instance.values.reshape(1, -1)
            )[0][1]
            
            results["counterfactuals"].append({
                "features": cf_instance.to_dict(),
                "prediction": float(cf_prediction),
                "prediction_class": int(np.argmax([1-cf_prediction, cf_prediction])),
                "changes": changes,
                "n_changes": len(changes),
                "feasibility_score": self._compute_feasibility(changes)
            })
        
        # Overall feasibility
        results["feasibility_analysis"] = {
            "all_feasible": all(
                cf["feasibility_score"] > 0.8 
                for cf in results["counterfactuals"]
            ),
            "best_counterfactual": max(
                results["counterfactuals"], 
                key=lambda x: x["feasibility_score"]
            )["features"],
            "immutable_features": self.immutable_features,
            "n_immutable_changes": sum(
                1 for cf in results["counterfactuals"]
                for change in cf["changes"].values()
                if not change["feasible"]
            )
        }
        
        return results
    
    def _compute_feasibility(self, changes: Dict) -> float:
        """
        Compute feasibility score for a set of changes.
        
        Score based on:
        - Immutability constraints
        - Change magnitude (smaller = more feasible)
        - Realistic ranges
        """
        if not changes:
            return 1.0
        
        scores = []
        for feature, change in changes.items():
            if not change["feasible"]:
                scores.append(0.0)  # Impossible change
            else:
                # Penalize large changes
                magnitude_score = max(0, 1.0 - change["magnitude"])
                scores.append(magnitude_score)
        
        return np.mean(scores)
    
    def generate_actionable_explanation(self, instance: pd.DataFrame,
                                        n_counterfactuals: int = 3) -> str:
        """
        Generate human-readable actionable explanation.
        """
        results = self.generate_counterfactuals(instance, n_counterfactuals)
        
        lines = [
            f"Current prediction: {results['original']['prediction']:.1%} probability",
            "",
            "To change this prediction, consider these changes:",
            ""
        ]
        
        for i, cf in enumerate(results["counterfactuals"][:n_counterfactuals], 1):
            lines.append(f"Option {i} (feasibility: {cf['feasibility_score']:.0%}):")
            for feature, change in cf["changes"].items():
                feasible = "✓" if change["feasible"] else "✗"
                lines.append(
                    f"  {feasible} Change {feature}: "
                    f"{change['original']:.2f} → {change['counterfactual']:.2f}"
                )
            lines.append(f"  Resulting prediction: {cf['prediction']:.1%}")
            lines.append("")
        
        return "\n".join(lines)
```

---

## 6. Agent-Level Explainability

### 6.1 Agent Trace Structure

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

@dataclass
class AgentStep:
    step_id: int
    action_type: str              # "think", "tool_call", "observe", "decide"
    input_data: Any
    output_data: Any
    reasoning: str                # LLM's reasoning for this step
    tool_used: Optional[str] = None
    tool_input: Optional[Dict] = None
    tool_output: Optional[str] = None
    confidence: float = 0.0
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class AgentTrace:
    trace_id: str
    agent_id: str
    user_query: str
    steps: List[AgentStep] = field(default_factory=list)
    final_answer: str = ""
    total_duration_ms: float = 0.0
    
    def to_explanation(self, audience: str = "technical") -> Dict:
        """
        Convert agent trace to structured explanation.
        """
        if audience == "technical":
            return self._technical_explanation()
        elif audience == "domain_expert":
            return self._expert_explanation()
        else:
            return self._user_explanation()
    
    def _technical_explanation(self) -> Dict:
        return {
            "trace_id": self.trace_id,
            "query": self.user_query,
            "final_answer": self.final_answer,
            "execution_summary": {
                "total_steps": len(self.steps),
                "tool_calls": sum(1 for s in self.steps if s.action_type == "tool_call"),
                "think_steps": sum(1 for s in self.steps if s.action_type == "think"),
                "total_duration_ms": self.total_duration_ms
            },
            "reasoning_chain": [
                {
                    "step": s.step_id,
                    "type": s.action_type,
                    "reasoning": s.reasoning,
                    "tool": s.tool_used,
                    "confidence": s.confidence
                }
                for s in self.steps
            ],
            "dependency_graph": self._build_dependency_graph(),
            "bottleneck_analysis": self._identify_bottlenecks()
        }
    
    def _user_explanation(self) -> Dict:
        """
        Generate user-friendly explanation.
        """
        # Filter to key decision points
        key_steps = [s for s in self.steps if s.action_type in ("think", "decide")]
        
        return {
            "answer": self.final_answer,
            "how_i_got_here": [
                {
                    "what_i_did": s.reasoning,
                    "what_i_found": s.output_data if s.output_data else "N/A"
                }
                for s in key_steps[:5]  # Top 5 steps only
            ],
            "confidence": np.mean([s.confidence for s in self.steps]) if self.steps else 0,
            "sources_used": [
                s.tool_output[:100] for s in self.steps 
                if s.action_type == "tool_call" and s.tool_output
            ][:3]  # Top 3 sources
        }
    
    def _build_dependency_graph(self) -> List[Dict]:
        """Build step dependency graph."""
        edges = []
        for i, step in enumerate(self.steps):
            if i > 0:
                edges.append({
                    "from": self.steps[i-1].step_id,
                    "to": step.step_id,
                    "type": "sequential"
                })
            if step.tool_used:
                edges.append({
                    "from": f"tool_{step.tool_used}",
                    "to": step.step_id,
                    "type": "dependency"
                })
        return edges
    
    def _identify_bottlenecks(self) -> List[Dict]:
        """Identify performance bottlenecks in agent execution."""
        bottlenecks = []
        
        for step in self.steps:
            if step.duration_ms > 1000:  # > 1 second
                bottlenecks.append({
                    "step": step.step_id,
                    "type": step.action_type,
                    "duration_ms": step.duration_ms,
                    "issue": "Slow execution"
                })
            if step.confidence < 0.5:
                bottlenecks.append({
                    "step": step.step_id,
                    "type": step.action_type,
                    "confidence": step.confidence,
                    "issue": "Low confidence decision"
                })
        
        return bottlenecks


class AgentTracer:
    """
    Middleware that wraps agent execution to capture traces.
    """
    
    def __init__(self, agent):
        self.agent = agent
        self.traces: List[AgentTrace] = []
    
    def execute(self, query: str) -> AgentTrace:
        """Execute query with full tracing."""
        trace = AgentTrace(
            trace_id=f"trace_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            agent_id=self.agent.agent_id,
            user_query=query
        )
        
        start_time = datetime.utcnow()
        
        # Intercept agent execution
        # (Implementation depends on agent framework)
        
        trace.total_duration_ms = (
            datetime.utcnow() - start_time
        ).total_seconds() * 1000
        
        self.traces.append(trace)
        return trace
    
    def get_explanation(self, trace_id: str, audience: str = "technical") -> Dict:
        """Get explanation for a completed trace."""
        trace = next(
            (t for t in self.traces if t.trace_id == trace_id), 
            None
        )
        if trace is None:
            raise ValueError(f"Trace {trace_id} not found")
        
        return trace.to_explanation(audience)
```

### 6.2 LangSmith Integration Pattern

```python
from langsmith import Client
from langsmith.run_helpers import trace

class LangSmithExplainer:
    """
    Integration with LangSmith for production agent tracing.
    """
    
    def __init__(self, project_name: str):
        self.client = Client()
        self.project_name = project_name
    
    @trace(name="agent_execution", project_name="xai-agent")
    def explain_agent_run(self, agent, query: str) -> Dict:
        """
        Run agent with LangSmith tracing enabled.
        Automatically captures all LLM calls, tool uses, and chain steps.
        """
        result = agent.run(query)
        
        # Get trace from LangSmith
        run_id = result.get("run_id")
        
        if run_id:
            run = self.client.read_run(run_id)
            
            return {
                "run_id": str(run_id),
                "trace_url": f"https://smith.langchain.com/runs/{run_id}",
                "total_tokens": run.total_tokens,
                "total_cost": run.total_cost,
                "steps": self._extract_steps(run),
                "feedback": self._get_feedback(run_id)
            }
        
        return {"result": result}
    
    def _extract_steps(self, run) -> List[Dict]:
        """Extract execution steps from LangSmith run."""
        steps = []
        for child_run in run.child_runs:
            steps.append({
                "name": child_run.name,
                "type": child_run.run_type,
                "duration_ms": (child_run.end_time - child_run.start_time).total_seconds() * 1000,
                "tokens": child_run.total_tokens,
                "cost": child_run.total_cost
            })
        return steps
    
    def _get_feedback(self, run_id) -> Dict:
        """Get human feedback for a run."""
        feedback = self.client.list_feedback(run_id=run_id)
        
        return {
            "positive": sum(1 for f in feedback if f.value == "positive"),
            "negative": sum(1 for f in feedback if f.value == "negative"),
            "comments": [f.comment for f in feedback if f.comment]
        }
```

---

## 7. RAG Explainability

### 7.1 RAG Attribution System

```python
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class RAGAttribution:
    """
    Attribution for RAG-based predictions.
    Tracks which retrieved documents contributed to the answer.
    """
    query: str
    answer: str
    retrieved_documents: List[Dict]
    document_attributions: List[Dict]
    citation_map: Dict[str, List[str]]  # answer_span → document_ids
    
    def to_explanation(self) -> Dict:
        """
        Generate explanation showing which documents influenced the answer.
        """
        # Sort documents by attribution score
        sorted_docs = sorted(
            self.document_attributions,
            key=lambda x: x["attribution_score"],
            reverse=True
        )
        
        return {
            "answer": self.answer,
            "confidence": self._compute_confidence(),
            "document_sources": [
                {
                    "document_id": doc["doc_id"],
                    "title": doc.get("title", "Unknown"),
                    "attribution_score": doc["attribution_score"],
                    "relevant_excerpt": doc.get("excerpt", ""),
                    "used_in_answer": doc["attribution_score"] > 0.1
                }
                for doc in sorted_docs[:5]
            ],
            "citation_map": self.citation_map,
            "attribution_method": "attention_weighted_retrieval"
        }
    
    def _compute_confidence(self) -> float:
        """Compute confidence based on attribution concentration."""
        scores = [d["attribution_score"] for d in self.document_attributions]
        if not scores:
            return 0.0
        
        # Higher concentration = higher confidence
        max_score = max(scores)
        concentration = max_score / (sum(scores) + 1e-10)
        
        return min(concentration, 1.0)


class RAGExplainer:
    """
    Explain RAG predictions by attributing answer spans to source documents.
    """
    
    def __init__(self, rag_pipeline):
        self.rag = rag_pipeline
    
    def explain(self, query: str) -> RAGAttribution:
        """
        Run RAG and generate attribution explanation.
        """
        # Retrieve documents
        documents = self.rag.retrieve(query)
        
        # Generate answer with attribution
        answer_result = self.rag.generate(query, documents)
        
        # Compute document attributions
        attributions = self._compute_attributions(
            query, documents, answer_result["answer"]
        )
        
        # Build citation map
        citation_map = self._build_citation_map(
            answer_result["answer"], documents, attributions
        )
        
        return RAGAttribution(
            query=query,
            answer=answer_result["answer"],
            retrieved_documents=[
                {"doc_id": i, "content": doc["content"], "metadata": doc.get("metadata", {})}
                for i, doc in enumerate(documents)
            ],
            document_attributions=attributions,
            citation_map=citation_map
        )
    
    def _compute_attributions(self, query, documents, answer) -> List[Dict]:
        """
        Compute attribution scores for each document.
        
        Uses multiple signals:
        1. Retrieval score (how relevant was the document to the query)
        2. Attention score (how much the generator attended to each document)
        3. Overlap score (lexical overlap between document and answer)
        """
        attributions = []
        
        for i, doc in enumerate(documents):
            # Retrieval relevance
            retrieval_score = doc.get("score", 0.0)
            
            # Attention-based attribution (if available)
            attention_score = self._compute_attention_attribution(doc, answer)
            
            # Lexical overlap
            overlap_score = self._compute_overlap(doc["content"], answer)
            
            # Combined attribution
            combined_score = (
                0.4 * retrieval_score +
                0.4 * attention_score +
                0.2 * overlap_score
            )
            
            attributions.append({
                "doc_id": i,
                "attribution_score": combined_score,
                "retrieval_score": retrieval_score,
                "attention_score": attention_score,
                "overlap_score": overlap_score,
                "excerpt": doc["content"][:200]
            })
        
        return attributions
    
    def _compute_attention_attribution(self, document, answer) -> float:
        """Compute how much the generator attended to this document."""
        # Simplified: use token overlap as proxy for attention
        doc_tokens = set(document["content"].lower().split())
        answer_tokens = set(answer.lower().split())
        
        overlap = len(doc_tokens & answer_tokens)
        return overlap / (len(answer_tokens) + 1e-10)
    
    def _compute_overlap(self, text1, text2) -> float:
        """Compute lexical overlap between two texts."""
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        
        return len(intersection) / (len(union) + 1e-10)
    
    def _build_citation_map(self, answer, documents, attributions) -> Dict:
        """
        Map answer spans to source documents.
        """
        citation_map = {}
        
        # Split answer into sentences
        sentences = answer.split('. ')
        
        for sentence in sentences:
            # Find best matching document for each sentence
            best_doc = None
            best_score = 0
            
            for attr in attributions:
                doc_content = documents[attr["doc_id"]]["content"]
                overlap = self._compute_overlap(sentence, doc_content)
                
                if overlap > best_score:
                    best_score = overlap
                    best_doc = attr["doc_id"]
            
            if best_doc is not None and best_score > 0.1:
                sentence_key = sentence.strip()[:50] + "..."
                if sentence_key not in citation_map:
                    citation_map[sentence_key] = []
                citation_map[sentence_key].append(f"doc_{best_doc}")
        
        return citation_map
```

---

## 8. LLM Explainability at Scale

### 8.1 Chain-of-Thought Attribution

```python
class CoTAttributor:
    """
    Attribute LLM chain-of-thought reasoning to specific inputs,
    retrieved documents, or tool outputs.
    """
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def attribute_reasoning(self, query: str, 
                           context: Dict,
                           chain_of_thought: str) -> Dict:
        """
        Analyze chain-of-thought to identify reasoning steps and their sources.
        """
        # Parse chain-of-thought into steps
        steps = self._parse_cot_steps(chain_of_thought)
        
        attributed_steps = []
        for step in steps:
            # For each reasoning step, identify its source
            attribution = self._attribute_step(step, context)
            attributed_steps.append({
                "step": step,
                "source": attribution["source"],
                "source_content": attribution["content"],
                "confidence": attribution["confidence"],
                "type": attribution["type"]  # "retrieval", "tool", "inference", "hallucination?"
            })
        
        # Identify potential hallucinations
        hallucination_risk = self._assess_hallucination_risk(attributed_steps)
        
        return {
            "reasoning_steps": attributed_steps,
            "hallucination_risk": hallucination_risk,
            "trace_completeness": self._assess_trace_completeness(attributed_steps),
            "attribution_coverage": self._compute_attribution_coverage(attributed_steps)
        }
    
    def _parse_cot_steps(self, chain_of_thought: str) -> List[str]:
        """Parse chain-of-thought into individual reasoning steps."""
        # Split on common reasoning markers
        import re
        steps = re.split(r'\n\n|\nStep \d+:|\n\d+\.', chain_of_thought)
        return [s.strip() for s in steps if s.strip()]
    
    def _attribute_step(self, step: str, context: Dict) -> Dict:
        """Attribute a single reasoning step to its source."""
        # Check retrieval context
        for i, doc in enumerate(context.get("retrieved_docs", [])):
            if self._step_references_doc(step, doc):
                return {
                    "source": f"retrieved_doc_{i}",
                    "content": doc["content"][:100],
                    "confidence": 0.8,
                    "type": "retrieval"
                }
        
        # Check tool outputs
        for tool_name, tool_output in context.get("tool_outputs", {}).items():
            if self._step_references_tool_output(step, tool_output):
                return {
                    "source": f"tool_{tool_name}",
                    "content": str(tool_output)[:100],
                    "confidence": 0.9,
                    "type": "tool"
                }
        
        # Check if it's pure inference (no external source)
        return {
            "source": "model_inference",
            "content": "",
            "confidence": 0.5,
            "type": "inference"
        }
    
    def _step_references_doc(self, step: str, doc: Dict) -> bool:
        """Check if a reasoning step references a specific document."""
        doc_tokens = set(doc["content"].lower().split())
        step_tokens = set(step.lower().split())
        
        overlap = len(doc_tokens & step_tokens)
        return overlap > 3  # At least 3 shared tokens
    
    def _step_references_tool_output(self, step: str, tool_output) -> bool:
        """Check if a reasoning step references tool output."""
        output_str = str(tool_output).lower()
        return any(token in output_str for token in step.lower().split() if len(token) > 3)
    
    def _assess_hallucination_risk(self, attributed_steps: List[Dict]) -> Dict:
        """Assess risk of hallucination in the reasoning chain."""
        inference_steps = [s for s in attributed_steps if s["type"] == "inference"]
        total_steps = len(attributed_steps)
        
        inference_ratio = len(inference_steps) / (total_steps + 1e-10)
        
        risk_level = "low"
        if inference_ratio > 0.5:
            risk_level = "high"
        elif inference_ratio > 0.3:
            risk_level = "medium"
        
        return {
            "risk_level": risk_level,
            "inference_steps_ratio": inference_ratio,
            "unattributed_steps": len(inference_steps),
            "recommendation": "Verify claims made in unattributed steps"
        }
    
    def _assess_trace_completeness(self, attributed_steps: List[Dict]) -> float:
        """Assess how complete the reasoning trace is."""
        attributed = sum(1 for s in attributed_steps if s["type"] != "inference")
        return attributed / (len(attributed_steps) + 1e-10)
    
    def _compute_attribution_coverage(self, attributed_steps: List[Dict]) -> float:
        """Compute what percentage of reasoning is attributed to sources."""
        high_confidence = sum(1 for s in attributed_steps if s["confidence"] > 0.7)
        return high_confidence / (len(attributed_steps) + 1e-10)
```

---

## 9. Performance Optimization

### 9.1 XAI Computation Benchmarks

| Method | Single Instance | Batch (1000) | Optimization Available |
|--------|----------------|-------------|----------------------|
| TreeSHAP (exact) | ~5ms | ~2s | Vectorization, GPU |
| KernelSHAP | ~500ms | ~5min | Parallel, sampling |
| LIME | ~200ms | ~2min | Parallel |
| Integrated Gradients | ~50ms | ~10s | GPU, batch processing |
| Attention extraction | ~10ms | ~1s | Batching, caching |
| Counterfactuals (DiCE) | ~500ms | ~5min | Parallel, pre-computation |

### 9.2 Caching Strategy

```python
class ExplanationCache:
    """
    Multi-level caching for XAI explanations.
    """
    
    def __init__(self, redis_client=None, local_cache_size=1000):
        self.redis = redis_client
        self.local_cache = {}  # LRU cache
        self.local_cache_size = local_cache_size
    
    def get(self, cache_key: str) -> Optional[Dict]:
        """Get cached explanation with fallback."""
        # Level 1: Local memory
        if cache_key in self.local_cache:
            return self.local_cache[cache_key]
        
        # Level 2: Redis
        if self.redis:
            cached = self.redis.get(f"explanation:{cache_key}")
            if cached:
                import json
                result = json.loads(cached)
                # Populate local cache
                self.local_cache[cache_key] = result
                return result
        
        return None
    
    def set(self, cache_key: str, explanation: Dict, ttl: int = 3600):
        """Cache explanation at both levels."""
        # Level 1: Local memory
        if len(self.local_cache) >= self.local_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.local_cache))
            del self.local_cache[oldest_key]
        self.local_cache[cache_key] = explanation
        
        # Level 2: Redis
        if self.redis:
            import json
            self.redis.setex(
                f"explanation:{cache_key}",
                ttl,
                json.dumps(explanation)
            )
    
    @staticmethod
    def compute_cache_key(model_id: str, instance_hash: str, 
                          explanation_type: str) -> str:
        """Generate deterministic cache key."""
        return f"{model_id}:{instance_hash}:{explanation_type}"
```

### 9.3 Batch Explanation Pipeline

```python
class BatchExplanationPipeline:
    """
    High-throughput batch explanation pipeline.
    """
    
    def __init__(self, explainer, cache: ExplanationCache = None):
        self.explainer = explainer
        self.cache = cache
    
    async def explain_batch(self, instances: List[Dict], 
                           model_id: str,
                           explanation_type: str = "shap") -> List[Dict]:
        """
        Explain multiple instances in parallel.
        """
        import asyncio
        import hashlib
        
        results = []
        to_compute = []
        
        # Check cache first
        for instance in instances:
            instance_hash = hashlib.md5(
                str(sorted(instance.items())).encode()
            ).hexdigest()
            cache_key = ExplanationCache.compute_cache_key(
                model_id, instance_hash, explanation_type
            )
            
            if self.cache:
                cached = self.cache.get(cache_key)
                if cached:
                    results.append(cached)
                    continue
            
            to_compute.append((instance, cache_key))
        
        # Compute missing explanations in parallel
        if to_compute:
            tasks = [
                self._compute_explanation(instance, model_id, explanation_type, cache_key)
                for instance, cache_key in to_compute
            ]
            
            computed = await asyncio.gather(*tasks)
            
            for (instance, cache_key), explanation in zip(to_compute, computed):
                if self.cache:
                    self.cache.set(cache_key, explanation)
                results.append(explanation)
        
        return results
    
    async def _compute_explanation(self, instance, model_id, 
                                   explanation_type, cache_key) -> Dict:
        """Compute explanation for a single instance."""
        # Implementation depends on explainer type
        return await asyncio.to_thread(
            self.explainer.explain, instance, model_id, explanation_type
        )
```

---

## 10. Testing and Validation

### 10.1 XAI Test Suite

```python
import pytest
import numpy as np

class TestXAIExplanation:
    """
    Test suite for XAI explanation quality.
    """
    
    def test_shap_completeness(self, shap_explainer, test_instance):
        """SHAP values should sum to prediction - base value."""
        explanation = shap_explainer.explain(test_instance)
        
        total_shap = sum(c["shap_value"] for c in explanation["contributions"])
        expected = explanation["prediction"] - explanation["base_value"]
        
        assert abs(total_shap - expected) < 0.01, \
            f"SHAP sum {total_shap} != expected {expected}"
    
    def test_shap_symmetry(self, shap_explainer, test_data):
        """Similar instances should get similar explanations."""
        # Create two similar instances
        instance_a = test_data[0:1]
        instance_b = test_data[0:1].copy()
        instance_b[0, 0] += 0.01  # Tiny perturbation
        
        expl_a = shap_explainer.explain(instance_a)
        expl_b = shap_explainer.explain(instance_b)
        
        # Check top features are the same
        top_a = [c["feature"] for c in expl_a["contributions"][:3]]
        top_b = [c["feature"] for c in expl_b["contributions"][:3]]
        
        assert set(top_a) == set(top_b), \
            f"Top features differ: {top_a} vs {top_b}"
    
    def test_shap_null_player(self, shap_explainer):
        """Irrelevant features should get zero SHAP value."""
        # Add a random noise feature
        instance_with_noise = add_noise_feature(test_instance)
        
        explanation = shap_explainer.explain(instance_with_noise)
        
        # Find noise feature contribution
        noise_contribution = next(
            c for c in explanation["contributions"] 
            if c["feature"] == "noise_feature"
        )
        
        assert abs(noise_contribution["shap_value"]) < 0.01, \
            f"Noise feature has non-zero SHAP: {noise_contribution['shap_value']}"
    
    def test_counterfactual_validity(self, cf_pipeline, test_instance):
        """Counterfactuals should actually change the prediction."""
        result = cf_pipeline.generate_counterfactuals(test_instance)
        
        original_pred = result["original"]["prediction"]
        
        for cf in result["counterfactuals"]:
            cf_pred = cf["prediction"]
            
            # Counterfactual should change prediction
            assert abs(cf_pred - original_pred) > 0.1, \
                f"Counterfactual didn't change prediction: {cf_pred} ≈ {original_pred}"
    
    def test_counterfactual_proximity(self, cf_pipeline, test_instance):
        """Counterfactuals should be close to original instance."""
        result = cf_pipeline.generate_counterfactuals(test_instance)
        
        for cf in result["counterfactuals"]:
            n_changes = cf["n_changes"]
            
            # Should be sparse (few changes)
            assert n_changes <= 3, \
                f"Counterfactual changes too many features: {n_changes}"
    
    def test_explanation_stability(self, lime_explainer, test_instance):
        """LIME explanations should be stable across runs."""
        explanations = []
        for _ in range(5):
            exp = lime_explainer.explain(test_instance)
            explanations.append(exp)
        
        # Check that top feature is consistent
        top_features = [e["contributions"][0]["feature"] for e in explanations]
        
        assert len(set(top_features)) == 1, \
            f"LIME top feature unstable: {set(top_features)}"
    
    def test_attention_rollout_completeness(self, attention_viz, test_text):
        """Attention rollout should cover all tokens."""
        attention_data = attention_viz.get_attention(test_text)
        rollout = attention_viz.attention_rollout(attention_data)
        
        # Each row should sum to ~1 (probability distribution)
        for i in range(rollout.shape[0]):
            row_sum = rollout[i].sum()
            assert abs(row_sum - 1.0) < 0.01, \
                f"Rollout row {i} doesn't sum to 1: {row_sum}"
```

---

## 11. Infrastructure and Deployment

### 11.1 Docker Configuration

```dockerfile
# Dockerfile for XAI Service
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 11.2 Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: xai-service
  labels:
    app: xai-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: xai-service
  template:
    metadata:
      labels:
        app: xai-service
    spec:
      containers:
      - name: xai-service
        image: xai-service:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        env:
        - name: MODEL_CACHE_SIZE
          value: "100"
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: xai-secrets
              key: redis-url
---
apiVersion: v1
kind: Service
metadata:
  name: xai-service
spec:
  selector:
    app: xai-service
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

*Last updated: July 7, 2026*
*Part of the AI Base Knowledge Library — Category 64*
