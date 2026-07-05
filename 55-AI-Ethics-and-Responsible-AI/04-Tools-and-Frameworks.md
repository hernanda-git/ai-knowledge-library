# Tools and Frameworks for AI Ethics and Responsible AI

> A comprehensive guide to the tools, libraries, platforms, and frameworks available for implementing responsible AI in practice.

---

## Table of Contents

1. [Fairness Libraries](#1-fairness-libraries)
2. [Explainability Tools](#2-explainability-tools)
3. [Privacy-Preserving Frameworks](#3-privacy-preserving-frameworks)
4. [Ethics Audit Platforms](#4-ethics-audit-platforms)
5. [Monitoring and Observability](#5-monitoring-and-observability)
6. [Documentation Tools](#6-documentation-tools)
7. [Comparison Matrix](#7-comparison-matrix)
8. [Selection Guide](#8-selection-guide)

---

## 1. Fairness Libraries

### 1.1 Fairlearn (Microsoft)

**Overview:** Open-source toolkit for fairness assessment and mitigation in ML.

```python
from fairlearn.metrics import MetricFrame, demographic_parity_difference
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.linear_model import LogisticRegression

# Fairness assessment
metric_frame = MetricFrame(
    metrics={'accuracy': accuracy_score, 'positive_rate': lambda y, p: p.mean()},
    y_true=y_test, y_pred=y_pred,
    sensitive_features=X_test['gender']
)
print(metric_frame.by_group)
print(metric_frame.difference())  # Worst-case difference across groups

# Fairness-constrained training
constraint = DemographicParity()
reducer = ExponentiatedGradient(
    estimator=LogisticRegression(), constraint=constraint
)
reducer.fit(X_train, y_train, sensitive_features=X_train['gender'])
y_pred_fair = reducer.predict(X_test)
```

| Feature | Details |
|---|---|
| **Strengths** | sklearn-compatible, well-documented, active maintenance |
| **Fairness methods** | Demographic parity, equalized odds, exponentiated gradient |
| **Best for** | Python/sklearn users, binary classification fairness |
| **Limitations** | Limited support for deep learning, no real-time monitoring |

### 1.2 AI Fairness 360 (IBM)

**Overview:** Comprehensive toolkit with 70+ fairness metrics and 10+ mitigation algorithms.

```python
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import AdversarialDebiasing

# Create dataset
dataset = BinaryLabelDataset(
    df=data, label_names=['approved'],
    protected_attribute_names=['race']
)

# Pre-processing mitigation
rw = Reweighing(
    privileged_groups=[{'race': 1}],
    unprivileged_groups=[{'race': 0}]
)
dataset_transformed = rw.fit_transform(dataset)

# In-processing mitigation
debiased = AdversarialDebiasing(
    privileged_groups=[{'race': 1}],
    unprivileged_groups=[{'race': 0}],
    scope_name='debiaser',
    debias=True
)
debiased.fit(train_dataset)
pred = debiased.predict(test_dataset)

# Fairness metrics
metric = ClassificationMetric(test_dataset, pred,
    unprivileged_groups=[{'race': 0}],
    privileged_groups=[{'race': 1}]
)
print(f"Disparate Impact: {metric.disparate_impact():.3f}")
print(f"Equal Opportunity Diff: {metric.equal_opportunity_difference():.3f}")
```

| Feature | Details |
|---|---|
| **Strengths** | Most comprehensive fairness toolkit, research-grade metrics |
| **Fairness methods** | Pre-processing, in-processing, post-processing (10+ algorithms) |
| **Best for** | Research, thorough audits, enterprise compliance |
| **Limitations** | Steeper learning curve, heavier dependencies |

### 1.3 What-If Tool (Google)

**Overview:** Visual tool for exploring ML model fairness and performance.

```python
# What-If Tool as a library
from witwidget.notebook.visualization import WitConfig, WitWidget

# Configure
config = WitConfig().set_model(model).set_dataset(X_test, y_test)
widget = WitWidget(config)
# Provides interactive visualization in Jupyter notebooks
```

| Feature | Details |
|---|---|
| **Strengths** | Interactive visualization, Jupyter integration |
| **Best for** | Exploratory analysis, model comparison, stakeholder demos |
| **Limitations** | Primarily visual, limited programmatic API |

### 1.4 Fairness Comparison

| Capability | Fairlearn | AIF360 | What-If Tool |Themis-ML |
|---|---|---|---|---|
| Fairness assessment | ★★★★★ | ★★★★★ | ★★★★☆ | ★★★☆☆ |
| Pre-processing mitigation | ★★★☆☆ | ★★★★★ | ★★☆☆☆ | ★★★★☆ |
| In-processing mitigation | ★★★★☆ | ★★★★★ | ★★☆☆☆ | ★★★☆☆ |
| Post-processing mitigation | ★★★☆☆ | ★★★★☆ | ★★★☆☆ | ★★★★☆ |
| Visualization | ★★★☆☆ | ★★★☆☆ | ★★★★★ | ★★☆☆☆ |
| Deep learning support | ★★☆☆☆ | ★★★☆☆ | ★★★☆☆ | ★★☆☆☆ |
| Production readiness | ★★★★★ | ★★★☆☆ | ★★☆☆☆ | ★★☆☆☆ |
| Documentation | ★★★★★ | ★★★★☆ | ★★★★☆ | ★★☆☆☆ |

---

## 2. Explainability Tools

### 2.1 SHAP

**Overview:** Unified approach to explaining ML model outputs using Shapley values.

```python
import shap

# Tree models (fast, exact)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Deep learning (DeepSHAP)
explainer = shap.DeepExplainer(model, X_background)
shap_values = explainer.shap_values(X)

# Any model (KernelSHAP — model-agnostic, slower)
explainer = shap.KernelExplainer(model.predict_proba, X_background)
shap_values = explainer.shap_values(X)

# Visualization
shap.summary_plot(shap_values, X)
shap.force_plot(explainer.expected_value, shap_values[0], X.iloc[0])
shap.dependence_plot("feature_name", shap_values, X)
```

| Feature | Details |
|---|---|
| **Strengths** | Theoretically grounded, model-agnostic, great visualizations |
| **Best for** | Feature importance, individual explanations, debugging |
| **Limitations** | Slow for large datasets (KernelSHAP), requires background data |

### 2.2 LIME

**Overview:** Local Interpretable Model-agnostic Explanations.

```python
from lime.lime_tabular import LimeTabularExplainer
from lime.lime_text import LimeTextExplainer

# Tabular data
explainer = LimeTabularExplainer(
    X_train.values, feature_names=feature_names,
    class_names=class_names, mode='classification'
)
exp = explainer.explain_instance(X_test.iloc[0].values, model.predict_proba)
exp.show_in_notebook()

# Text data
text_explainer = LimeTextExplainer(class_names=class_names)
exp = text_explainer.explain_instance(text, model.predict_proba, num_features=10)
```

| Feature | Details |
|---|---|
| **Strengths** | Intuitive explanations, works with any model, text support |
| **Best for** | Text classification, quick individual explanations |
| **Limitations** | Explanations can be unstable, no global explanations |

### 2.3 Captum (Facebook)

**Overview:** Model interpretability library for PyTorch.

```python
from captum.attr import (
    IntegratedGradients, GradientShap, DeepLift,
    Saliency, InputXGradient, Occlusion, FeatureAblation
)

# Integrated Gradients
ig = IntegratedGradients(model)
attributions = ig.attribute(inputs, target=target_class, n_steps=200)

# GradientShap
gs = GradientShap(model)
attributions = gs.attribute(inputs, baselines=background, target=target_class)

# Saliency maps
saliency = Saliency(model)
attributions = saliency.attribute(inputs, target=target_class)
```

| Feature | Details |
|---|---|
| **Strengths** | Deep learning focused, GPU-optimized, comprehensive methods |
| **Best for** | PyTorch models, computer vision, NLP interpretability |
| **Limitations** | PyTorch only, less suitable for tabular data |

### 2.4 InterpretML (Google)

**Overview:** Open-source toolkit for interpretable machine learning.

```python
from interpret.glassbox import ExplainableBoostingClassifier, LinearModel
from interpret import show

# Explainable Boosting Machine (EBM)
ebm = ExplainableBoostingClassifier()
ebm.fit(X_train, y_train)

# Global explanation
ebm_global = ebm.explain_global()
show(ebm_global)

# Local explanation
ebm_local = ebm.explain_local(X_test[:5], y_test[:5])
show(ebm_local)
```

| Feature | Details |
|---|---|
| **Strengths** | Inherently interpretable models, visualization dashboard |
| **Best for** | When you need both accuracy AND interpretability |
| **Limitations** | Limited to tabular data, EBM is less flexible than deep learning |

---

## 3. Privacy-Preserving Frameworks

### 3.1 Opacus (Facebook)

**Overview:** PyTorch library for training models with differential privacy.

```python
from opacus import PrivacyEngine

model = MyModel()
optimizer = SGD(model.parameters(), lr=0.01)
privacy_engine = PrivacyEngine()

model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    epochs=10,
    target_epsilon=3.0,
    target_delta=1e-5,
    max_grad_norm=1.0,
)

# Training loop as usual
for epoch in range(10):
    for batch in train_loader:
        loss = train_step(model, batch, optimizer)
    
    epsilon = privacy_engine.get_epsilon(delta=1e-5)
    print(f"Epoch {epoch}: ε = {epsilon:.2f}")
```

### 3.2 PySyft (OpenMined)

**Overview:** Framework for privacy-preserving machine learning and federated learning.

```python
import syft as sy

# Virtual worker setup
hook = sy.TorchHook(torch)
alice = sy.VirtualWorker(hook, id="alice")
bob = sy.VirtualWorker(hook, id="bob")

# Federated learning
data_alice = dataset_a.send(alice)
data_bob = dataset_b.send(bob)

# Train on distributed data without centralizing
model_ptr = model.send(alice)
# ... federated training loop
```

### 3.3 TF Privacy (TensorFlow)

```python
import tensorflow as tf
from tensorflow_privacy.privacy.optimizers import DPAdamGaussianOptimizer

optimizer = DPAdamGaussianOptimizer(
    l2_norm_clip=1.0,
    noise_multiplier=1.1,
    num_microbatches=1,
    learning_rate=0.01
)

# Use as regular optimizer
model.compile(optimizer=optimizer, loss=loss_fn)
model.fit(train_data, train_labels, epochs=10)
```

---

## 4. Ethics Audit Platforms

### 4.1 Azure Responsible AI Dashboard

**Overview:** Microsoft's integrated responsible AI toolkit in Azure ML.

```
Features:
├── Data Explorer — Understand training data distributions
├── Error Analysis — Identify where model fails
├── Model Interpretability — SHAP-based explanations
├── Fairness Assessment — Group fairness metrics
├── Counterfactual Analysis — What-if scenarios
└── Responsible AI Scorecard — Compliance reporting
```

| Feature | Details |
|---|---|
| **Strengths** | Integrated workflow, enterprise support, compliance reporting |
| **Best for** | Azure/Microsoft shops, enterprise compliance |
| **Limitations** | Azure-specific, less flexible than open-source tools |

### 4.2 Google Model Cards

```python
# Generate model card programmatically
card = {
    "model_details": {
        "name": "My Model",
        "version": "1.0",
        "type": "Binary classifier",
    },
    "intended_use": {
        "primary_use": "Credit scoring",
        "out_of_scope": ["Medical diagnosis"],
    },
    "performance": {
        "overall": {"accuracy": 0.85},
        "by_group": {
            "gender_0": {"accuracy": 0.83},
            "gender_1": {"accuracy": 0.84},
        },
    },
    "ethical_considerations": {
        "bias_risks": ["Historical lending bias"],
        "fairness_measures": ["Demographic parity", "Equal opportunity"],
    },
}
```

### 4.3 Holistic AI

**Overview:** Enterprise platform for AI assurance and compliance.

| Feature | Details |
|---|---|
| **Strengths** | Full audit lifecycle, regulatory alignment, continuous monitoring |
| **Best for** | Enterprise compliance, regulated industries |
| **Limitations** | Commercial, expensive |

### 4.4 Credo AI

**Overview:** Platform for AI governance and policy management.

| Feature | Details |
|---|---|
| **Strengths** | Policy-driven approach, governance workflows |
| **Best for** | Organizations establishing AI governance programs |
| **Limitations** | Commercial, requires process maturity |

---

## 5. Monitoring and Observability

### 5.1 Evidently AI

**Overview:** Open-source ML monitoring for data drift, model performance, and fairness.

```python
from evidently.metrics import *
from evidently.report import Report

# Data drift report
report = Report(metrics=[
    DataDriftTable(),
    DatasetSummaryMetric(),
])
report.run(reference_data=train_data, current_data=production_data)
report.save_html("drift_report.html")

# Fairness monitoring
from evidently.metric_preset import DataDriftPreset

fairness_report = Report(metrics=[
    ClassificationQualityMetric(),
    ClassificationQualityByClass(),
    ClassificationQualityByFeature(),
])
fairness_report.run(
    reference_data=train_data,
    current_data=production_data,
    column_mapping=ColumnMapping(
        target='label', prediction='prediction',
        numerical_features=['age', 'income'],
        categorical_features=['gender', 'race']
    )
)
```

### 5.2 WhyLabs

**Overview:** Platform for ML monitoring in production.

| Feature | Details |
|---|---|
| **Strengths** | Real-time monitoring, alerts, integration with MLflow |
| **Best for** | Production monitoring at scale |
| **Limitations** | Commercial, requires SDK integration |

### 5.3 Custom Monitoring Stack

```python
# Prometheus + Grafana for ethical metrics
from prometheus_client import Counter, Gauge, Histogram

# Metrics
PREDICTION_COUNT = Counter('predictions_total', 'Total predictions', ['model', 'group'])
FAIRNESS_RATIO = Gauge('fairness_disparate_impact', 'Disparate impact ratio', ['model', 'attribute'])
LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency', ['model'])

# Usage
PREDICTION_COUNT.labels(model='credit_scorer', group='gender_0').inc()
FAIRNESS_RATIO.labels(model='credit_scorer', attribute='gender').set(0.87)
```

---

## 6. Documentation Tools

### 6.1 Model Cards Toolkit

```python
# Automated model card generation
from model_cards import ModelCard, ModelCardData

data = ModelCardData(
    language="en",
    license="mit",
    model_details={
        "name": "Fair Credit Scorer",
        "version": "2.1",
        "type": "Logistic Regression",
    }
)
card = ModelCard(data)
card.save("model_card.md")
```

### 6.2 Datasheets for Datasets

Standardized documentation template following Gebru et al. (2021):

```markdown
# Datasheet: [Dataset Name]

## Motivation
- Purpose: [Why was this dataset created?]
- Creator: [Who created it?]
- Funding: [Who funded it?]

## Composition
- Instances: [Number of instances]
- Features: [Feature descriptions]
- Labeling: [How was data labeled?]
- Sensitive attributes: [Protected attributes included]

## Collection
- Process: [How was data collected?]
- Timeframe: [When was data collected?]
- Ethical review: [Was ethical review conducted?]

## Uses
- Recommended: [Approved use cases]
- Prohibited: [Prohibited use cases]
```

---

## 7. Comparison Matrix

| Tool | Fairness | Explainability | Privacy | Monitoring | Docs | License |
|---|---|---|---|---|---|---|
| Fairlearn | ★★★★★ | ★★☆☆☆ | ☆☆☆☆☆ | ★★☆☆☆ | ★★★★★ | MIT |
| AIF360 | ★★★★★ | ★★★☆☆ | ☆☆☆☆☆ | ★★☆☆☆ | ★★★★☆ | Apache 2.0 |
| SHAP | ★★☆☆☆ | ★★★★★ | ☆☆☆☆☆ | ★☆☆☆☆ | ★★★★★ | MIT |
| LIME | ★☆☆☆☆ | ★★★★☆ | ☆☆☆☆☆ | ☆☆☆☆☆ | ★★★★☆ | BSD-2 |
| Captum | ★☆☆☆☆ | ★★★★★ | ☆☆☆☆☆ | ★☆☆☆☆ | ★★★★☆ | MIT |
| InterpretML | ★★☆☆☆ | ★★★★★ | ☆☆☆☆☆ | ☆☆☆☆☆ | ★★★★☆ | MIT |
| Opacus | ☆☆☆☆☆ | ☆☆☆☆☆ | ★★★★★ | ☆☆☆☆☆ | ★★★★☆ | Apache 2.0 |
| Evidently | ★★★★☆ | ★★☆☆☆ | ☆☆☆☆☆ | ★★★★★ | ★★★★☆ | Apache 2.0 |
| Azure RAI | ★★★★★ | ★★★★☆ | ★★★☆☆ | ★★★★☆ | ★★★★★ | Commercial |
| Holistic AI | ★★★★★ | ★★★★☆ | ★★★★☆ | ★★★★★ | ★★★★☆ | Commercial |

---

## 8. Selection Guide

### 8.1 By Use Case

| If you need to... | Recommended Tools |
|---|---|
| Audit fairness of existing model | AIF360 + Fairlearn |
| Add fairness constraints to training | Fairlearn + AIF360 |
| Explain individual predictions | SHAP + LIME |
| Build inherently interpretable model | InterpretML (EBM) |
| Train with privacy guarantees | Opacus (PyTorch) or TF Privacy |
| Monitor fairness in production | Evidently + custom Prometheus |
| Comply with EU AI Act | Azure RAI or Holistic AI |
| Document models for compliance | Model Cards Toolkit |
| Explain deep learning models | Captum + SHAP |

### 8.2 By Organization Size

| Organization | Recommended Stack |
|---|---|
| **Startup / Small Team** | Fairlearn + SHAP + Evidently (all open-source) |
| **Mid-size Company** | AIF360 + SHAP + Evidently + Model Cards |
| **Enterprise** | Azure RAI or Holistic AI + custom monitoring |
| **Regulated Industry** | Holistic AI + AIF360 + full audit pipeline |
| **Research** | AIF360 + Captum + custom evaluation suites |

### 8.3 Quick Start Recommendation

```
Starting from scratch? Use this stack:

1. Fairness:    pip install fairlearn
2. Explainability: pip install shap
3. Monitoring:  pip install evidently
4. Privacy:     pip install opacus  (if needed)
5. Docs:        Use model card template

This covers 80% of responsible AI needs with zero cost.
```

---

## Cross-References

| Topic | See Also |
|---|---|
| Ethical principles | [01-Overview](./01-Overview.md) |
| Implementation details | [03-Technical-Deep-Dive](./03-Technical-Deep-Dive.md) |
| Agent monitoring | [20-Agent-Infrastructure-and-Observability](../20-Agent-Infrastructure-and-Observability/) |
| Privacy regulation | [40-AI-Data-Sovereignty-and-Privacy](../40-AI-Data-Sovereignty-and-Privacy/) |
| Enterprise deployment | [05-Enterprise](../05-Enterprise/) |

---

*Last updated: July 2026*
