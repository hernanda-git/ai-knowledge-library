# AI Model Explainability and XAI at Scale: Tools and Frameworks

> **Description:** Comprehensive guide to XAI tools, frameworks, and platforms. Covers open-source libraries, commercial platforms, LLM-specific tools, and enterprise deployment options. Includes selection criteria, comparison tables, and integration patterns.

---

## Table of Contents

1. [XAI Tool Landscape Overview](#1-xai-tool-landscape-overview)
2. [Open-Source XAI Libraries](#2-open-source-xai-libraries)
3. [Commercial XAI Platforms](#3-commercial-xai-platforms)
4. [LLM and Agent-Specific XAI Tools](#4-llm-and-agent-specific-xai-tools)
5. [Model-Specific Tools](#5-model-specific-tools)
6. [Visualization Tools](#6-visualization-tools)
7. [Enterprise XAI Platforms](#7-enterprise-xai-platforms)
8. [Tool Selection Guide](#8-tool-selection-guide)
9. [Integration Patterns](#9-integration-patterns)
10. [Evaluation and Benchmarking Tools](#10-evaluation-and-benchmarking-tools)

---

## 1. XAI Tool Landscape Overview

### 1.1 Tool Categories

```
XAI Tool Ecosystem (2026):
├── Feature Attribution
│   ├── SHAP (gold standard)
│   ├── LIME (local approximation)
│   ├── Captum (PyTorch)
│   ├── Alibi (counterfactuals + anchors)
│   └── ELI5 (simple explanations)
├── Mechanistic Interpretability
│   ├── TransformerLens (circuit analysis)
│   ├── SAELens (sparse autoencoders)
│   ├── NeuralNetworkLibrary (activation patching)
│   └── BertViz (attention visualization)
├── LLM/Agent Observability
│   ├── LangSmith (LangChain tracing)
│   ├── Langfuse (open-source LLM tracing)
│   ├── Arize Phoenix (LLM monitoring)
│   ├── Helicone (LLM analytics)
│   └── Braintrust (evaluation + tracing)
├── Enterprise Platforms
│   ├── Arthur AI (model monitoring + XAI)
│   ├── Fiddler AI (explainability platform)
│   ├── C3 AI (enterprise AI suite)
│   ├── H2O.ai (autoML + explainability)
│   └── DataRobot (ML platform + XAI)
├── Visualization
│   ├── Google What-If Tool
│   ├── TensorBoard Projector
│   ├── Attention Visualizer
│   └── Netron (model visualization)
└── Evaluation
    ├── Promptfoo (LLM testing)
    ├── LangSmith (evaluation)
    ├── DeepEval (LLM metrics)
    └── RAGAS (RAG evaluation)
```

### 1.2 Maturity Matrix

| Tool | Maturity | Community | Documentation | Enterprise Ready |
|------|----------|-----------|--------------|-----------------|
| **SHAP** | ★★★★★ | Large | Excellent | Partial (self-hosted) |
| **LIME** | ★★★★☆ | Medium | Good | No |
| **Captum** | ★★★★☆ | Medium | Good | Partial |
| **InterpretML** | ★★★☆☆ | Small | Good | No |
| **Alibi** | ★★★☆☆ | Small | Good | No |
| **LangSmith** | ★★★★☆ | Growing | Excellent | Yes |
| **Langfuse** | ★★★☆☆ | Growing | Good | Partial (self-hosted) |
| **Arthur AI** | ★★★★☆ | Enterprise | Excellent | Yes |
| **Fiddler AI** | ★★★★☆ | Enterprise | Excellent | Yes |
| **TransformerLens** | ★★★☆☆ | Research | Moderate | No |
| **SAELens** | ★★☆☆☆ | Research | Basic | No |

---

## 2. Open-Source XAI Libraries

### 2.1 SHAP (SHapley Additive exPlanations)

**Repository:** https://github.com/shap/shap
**Install:** `pip install shap`

```python
import shap
import xgboost as xgb
from sklearn.datasets import make_classification

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
feature_names = [f"feature_{i}" for i in range(10)]

# Train model
model = xgb.XGBClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X, y)

# ─── Method 1: TreeExplainer (fastest for tree models) ───
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X[:5])

# Summary plot
shap.summary_plot(shap_values, X[:5], feature_names=feature_names)

# ─── Method 2: KernelExplainer (model-agnostic) ───
background = shap.sample(X, 100)
explainer_kernel = shap.KernelExplainer(model.predict_proba, background)
shap_values_kernel = explainer_kernel.shap_values(X[:5])

# ─── Method 3: DeepExplainer (for deep learning) ───
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    
    def forward(self, x):
        return self.layers(x)

net = SimpleNet()
explainer_deep = shap.DeepExplainer(net, torch.tensor(X[:100], dtype=torch.float32))
shap_values_deep = explainer_deep.shap_values(torch.tensor(X[:5], dtype=torch.float32))

# ─── Method 4: GradientExplainer (approximate DeepSHAP) ───
explainer_grad = shap.GradientExplainer(net, torch.tensor(X[:100], dtype=torch.float32))
shap_values_grad = explainer_grad.shap_values(torch.tensor(X[:5], dtype=torch.float32))
```

**Key Features:**
- Multiple explainer types for different model families
- Theoretically grounded (Shapley values)
- Visualization suite (summary, force, dependence, waterfall plots)
- GPU acceleration available
- Integration with scikit-learn, XGBoost, LightGBM, PyTorch, TensorFlow

**Limitations:**
- KernelSHAP is slow for high-dimensional data
- Background dataset choice affects results
- Memory intensive for large models

### 2.2 LIME (Local Interpretable Model-agnostic Explanations)

**Repository:** https://github.com/marcotcr/lime
**Install:** `pip install lime`

```python
import lime
import lime.lime_tabular
import lime.lime_text
import lime.lime_image

# ─── Tabular LIME ───
explainer_tabular = lime.lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=feature_names,
    mode='classification',
    discretize_continuous=True,
    class_names=['class_0', 'class_1']
)

# Explain single instance
exp = explainer_tabular.explain_instance(
    X_test[0],
    model.predict_proba,
    num_features=5
)

# Visualize
exp.show_in_notebook()
exp.save_to_file('explanation.html')

# ─── Text LIME ───
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Example: sentiment analysis
texts = ["This movie is great!", "Terrible waste of time."]
labels = [1, 0]

pipeline = make_pipeline(
    TfidfVectorizer(max_features=5000),
    LogisticRegression()
)
pipeline.fit(texts, labels)

explainer_text = lime.lime_text.LimeTextExplainer(class_names=['negative', 'positive'])

text_exp = explainer_text.explain_instance(
    "This movie is absolutely fantastic and wonderful!",
    pipeline.predict_proba,
    num_features=5
)

# ─── Image LIME ───
from lime import lime_image
from skimage.segmentation import slic

explainer_image = lime_image.LimeImageExplainer()

# Explain image classification
image_exp = explainer_image.explain_instance(
    image,
    model.predict,
    top_labels=3,
    hide_color=0,
    num_samples=1000,
    segmentation_fn=lambda x: slic(x, n_segments=50, compactness=30)
)
```

**Key Features:**
- Works with any model (truly model-agnostic)
- Supports tabular, text, and image data
- Intuitive local explanations
- Fast computation

**Limitations:**
- Unstable across runs (random perturbations)
- No global explanation capability
- Limited theoretical guarantees
- No built-in confidence intervals

### 2.3 Captum (PyTorch Model Interpretability)

**Repository:** https://github.com/pytorch/captum
**Install:** `pip install captum`

```python
import captum
from captum.attr import (
    IntegratedGradients,
    GradientShap,
    DeepLift,
    DeepLiftShap,
    Saliency,
    InputXGradient,
    Occlusion,
    FeatureAblation,
    LayerConductance,
    NeuronConductance,
    LayerIntegratedGradients
)

# Initialize model and data
model = SimpleNet()
input_tensor = torch.tensor(X_test[:5], dtype=torch.float32)
target = torch.tensor(y_test[:5], dtype=torch.long)

# ─── Integrated Gradients ───
ig = IntegratedGradients(model)
attributions_ig = ig.attribute(
    input_tensor,
    target=1,
    n_steps=200,
    return_convergence_delta=True
)

# ─── GradientSHAP ───
gs = GradientShap(model)
attributions_gs = gs.attribute(
    input_tensor,
    target=1,
    baselines=torch.randn_like(input_tensor) * 0.1
)

# ─── DeepLift ───
dl = DeepLift(model)
attributions_dl = dl.attribute(
    input_tensor,
    target=1,
    baselines=torch.zeros_like(input_tensor)
)

# ─── Saliency Maps ───
saliency = Saliency(model)
attributions_sal = saliency.attribute(
    input_tensor,
    target=1,
    abs=True
)

# ─── Occlusion (model-agnostic perturbation) ───
occlusion = Occlusion(model)
attributions_occ = occlusion.attribute(
    input_tensor,
    target=1,
    sliding_window_shapes=(1,),  # Feature-by-feature
    baselines=0
)

# ─── Feature Ablation ───
ablation = FeatureAblation(model)
attributions_abl = ablation.attribute(
    input_tensor,
    target=1
)

# ─── Layer-level attribution ───
layer_conductance = LayerConductance(model, model.layers[2])
attributions_lc = layer_conductance.attribute(
    input_tensor,
    target=1,
    n_steps=50
)
```

**Key Features:**
- Deep integration with PyTorch
- Wide range of attribution methods
- Layer-level and neuron-level attribution
- Convergence delta for quality assessment
- GPU support

**Limitations:**
- PyTorch only (no TensorFlow)
- More complex API than SHAP
- Visualization requires additional work

### 2.4 InterpretML

**Repository:** https://github.com/interpretml/interpret
**Install:** `pip install interpret`

```python
from interpret.glassbox import (
    ExplainableBoostingClassifier,
    LogisticRegression,
    ClassificationTree
)
from interpret import show

# ─── Explainable Boosting Machine (EBM) ───
# Glass-box model with XGBoost-level performance
ebm = ExplainableBoostingClassifier(
    max_bins=256,
    interactions=10,
    outer_bags=8,
    inner_bags=0
)
ebm.fit(X_train, y_train)

# Global explanation
global_explanation = ebm.explain_global()
show(global_explanation)

# Local explanation
local_explanation = ebm.explain_local(X_test[:5], y_test[:5])
show(local_explanation)

# ─── Compare with black-box model ───
from interpret.glassbox import BoostedClassifier

blackbox = BoostedClassifier()
blackbox.fit(X_train, y_train)

# Use black-box with LIME-like explanation
blackbox_explanation = blackbox.explain_global()
show(blackbox_explanation)

# ─── Metrics ───
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show

ebm = ExplainableBoostingClassifier()
ebm.fit(X_train, y_train)

perf = ebm.explain_perf(X_test, y_test)
show(perf)
```

**Key Features:**
- Explainable Boosting Machines (EBMs) — glass-box with top performance
- Built-in global and local explanations
- Interactive visualization
- SHAP-compatible explanations
- Production-ready with scikit-learn API

**Limitations:**
- Primarily for tabular data
- Slower than XGBoost for large datasets
- Limited to classification and regression

### 2.5 Alibi

**Repository:** https://github.com/SeldonIO/alibi
**Install:** `pip install alibi`

```python
from alibi.explainers import (
    AnchorTabular,
    AnchorText,
    CounterfactualProto,
    CounterfactualRL,
    KernelShap,
    TreeShap
)
from alibi.datasets import fetch_adult

# ─── Anchor Explanation ───
# If-then rules that explain predictions

# Load data
data = fetch_adult()
X_train, X_test = data.data[:1000], data.data[1000:1100]
feature_names = data.feature_names
categorical_names = data.category_map

# Initialize anchor explainer
anchor_explainer = AnchorTabular(
    predictor=model.predict,
    feature_names=feature_names,
    categorical_names=categorical_names,
    seed=42
)

# Fit on training data
anchor_explainer.fit(X_train)

# Generate explanation
anchor_exp = anchor_explainer.explain(
    X_test[0],
    threshold=0.95,  # 95% precision
    coverage_samples=1000
)

print("Anchor:", anchor_exp.anchor)
print("Precision:", anchor_exp.precision)
print("Coverage:", anchor_exp.coverage)

# ─── Counterfactual Explanation ───
cf_explainer = CounterfactualProto(
    predictor=model.predict,
    X_train=X_train,
    categorical_names=categorical_names,
    theta=0.5,
    max_features=10
)

cf_exp = cf_explainer.explain(X_test[0])

# Visualize
cf_exp.plot()
cf_exp.feature_importance()  # Which features changed

# ─── SHAP wrapper ───
shap_explainer = KernelShap(
    predictor=model.predict,
    background_dataset=X_train[:100]
)

shap_exp = shap_explainer.explain(X_test[:5])
```

### 2.6 ELI5

**Repository:** https://github.com/TeamHG-Memex/eli5
**Install:** `pip install eli5`

```python
import eli5
from eli5.sklearn import explain_weights
from eli5.formatters import format_html

# ─── Explain model weights ───
# For linear models
eli5.show_weights(model, feature_names=feature_names)

# For tree models
eli5.show_weights(model, feature_names=feature_names, top=10)

# ─── Explain single prediction ───
eli5.show_prediction(model, X_test[0], feature_names=feature_names)

# ─── With LIME backend ───
eli5.explain_prediction(
    model,
    X_test[0],
    feature_names=feature_names,
    top=5
)

# ─── Permutation importance ───
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(model, random_state=42).fit(X_test, y_test)
eli5.show_weights(perm, feature_names=feature_names)
```

---

## 3. Commercial XAI Platforms

### 3.1 Arthur AI

**Website:** https://www.arthur.ai
**Type:** Enterprise model monitoring + explainability

**Key Features:**
- Real-time model monitoring and alerting
- Built-in explainability for tabular, NLP, and computer vision
- Fairness and bias detection
- Data drift detection
- LLM-specific monitoring (hallucination detection, prompt injection)
- SOC 2 compliant

**Integration Pattern:**
```python
from arthurai import ArthurAI

# Connect to Arthur
client = ArthurAI(
    api_key="your-api-key",
    url="https://api.arthur.ai"
)

# Register model
model = client.register_model(
    name="loan-default-predictor",
    model_type="binary_classification"
)

# Log predictions with explanations
model.log_predictions(
    features=X_test[:100],
    predictions=y_pred[:100],
    explanations=shap_values[:100]
)

# Get explainability report
report = model.explainability_report(
    num_instances=100,
    methods=["shap", "lime"]
)
```

### 3.2 Fiddler AI

**Website:** https://www.fiddler.ai
**Type:** Explainability and model performance management

**Key Features:**
- Model-agnostic explainability
- What-if analysis
- Prediction dashboards
- Fairness monitoring
- Alerting on model drift
- Custom metric tracking

**Integration Pattern:**
```python
import fiddler as fdl

# Initialize client
client = fdl.Client(
    url='https://app.fiddler.ai',
    auth='your-api-key'
)

# Upload model
project = client.get_project('my-project')
model = project.upload_model(
    model_path='model.pkl',
    dataset=my_dataset,
    metadata={
        'target_col': 'default',
        'features': feature_names,
        'model_type': 'binary'
    }
)

# Explain prediction
explanation = model.explain(
    instance=test_instance,
    method='shap',
    num_features=10
)
```

### 3.3 H2O.ai

**Website:** https://www.h2o.ai
**Type:** AutoML + built-in explainability

**Key Features:**
- Automatic model explanations
- SHAP and LIME integration
- Partial Dependence Plots
- Variable Importance
- Model Fairness (with 7+ fairness metrics)
- Leaderboard with explanations

```python
import h2o
from h2o.automl import H2OAutoML

# Initialize H2O
h2o.init()

# Load data
train = h2o.import_file('train.csv')
test = h2o.import_file('test.csv')

# Run AutoML with explainability
aml = H2OAutoML(
    max_models=10,
    seed=42,
    explainability=True  # Enable automatic explanations
)
aml.train(x=feature_names, y='target', training_frame=train)

# Get leader model
leader = aml.leader

# Generate explanations
leader.explain(test)
leader.explain_row(test, row_index=0)

# Fairness analysis
leader.explain_fairness(test, protected_attributes=['gender', 'race'])
```

---

## 4. LLM and Agent-Specific XAI Tools

### 4.1 LangSmith

**Website:** https://smith.langchain.com
**Type:** LLM observability, evaluation, and debugging

**Key Features:**
- End-to-end tracing of LLM chains and agents
- Automatic capture of LLM calls, tool uses, and chain steps
- Dataset management for evaluation
- Human feedback collection
- Online evaluation with custom metrics
- Prompt playground

```python
from langsmith import Client
from langsmith.run_helpers import trace

client = Client()

# ─── Automatic tracing ───
@trace(name="my_agent", project_id="my-project")
def run_agent(query: str):
    # All LLM calls, tool uses automatically traced
    result = agent.run(query)
    return result

# Run with tracing
result = run_agent("What is the capital of France?")

# ─── Custom tracing ───
from langsmith import traceable

@traceable(run_type="chain", name="rag_chain")
def rag_chain(query: str):
    # Retrieve documents
    docs = retrieve(query)
    
    # Generate answer
    answer = generate(query, docs)
    
    return answer

# ─── Evaluation ───
from langsmith import Client

client = Client()

# Create evaluation dataset
dataset = client.create_dataset("my-evaluation-dataset")

# Add examples
client.create_examples(
    dataset_id=dataset.id,
    examples=[
        {
            "inputs": {"question": "What is XAI?"},
            "outputs": {"answer": "Explainable AI..."}
        }
    ]
)

# Run evaluation
experiment_results = client.run_on_dataset(
    dataset_name="my-evaluation-dataset",
    llm_or_chain_factory=lambda: rag_chain,
    evaluation={
        "qa_correctness": qa_evaluator,
        "relevance": relevance_evaluator
    }
)
```

### 4.2 Langfuse

**Website:** https://langfuse.com
**Type:** Open-source LLM observability

**Key Features:**
- Self-hostable (Docker, Kubernetes)
- OpenAI, LangChain, LlamaIndex integrations
- Prompt management
- Scoring and evaluation
- Cost tracking
- Multi-tenancy support

```python
# ─── Python SDK ───
from langfuse import Langfuse

langfuse = Langfuse(
    public_key="pk-lf-...",
    secret_key="sk-lf-...",
    host="https://cloud.langfuse.com"
)

# Create trace
trace = langfuse.trace(name="my-trace")

# Create generation (LLM call)
generation = trace.generation(
    name="chat-completion",
    model="gpt-4",
    input={"messages": [{"role": "user", "content": "Hello"}]},
    output={"content": "Hi there!"},
    usage={"total": 50}
)

# Create span (tool call)
span = trace.span(
    name="tool-call",
    input={"tool": "search", "query": "XAI"},
    output={"results": ["..."]},
    metadata={"duration_ms": 150}
)

# Score the trace
trace.score(name="quality", value=0.9)
trace.score(name="relevance", value=0.85)

# Flush events
langfuse.flush()

# ─── Decorator-based tracing ───
from langfuse.decorators import observe

@observe(as_type="generation")
def call_llm(messages, model="gpt-4"):
    # Automatically traces this LLM call
    return openai_client.chat.completions.create(
        model=model,
        messages=messages
    )

@observe()  # Creates a trace
def rag_pipeline(query: str):
    docs = retrieve(query)
    answer = call_llm([
        {"role": "system", "content": f"Context: {docs}"},
        {"role": "user", "content": query}
    ])
    return answer
```

### 4.3 Arize Phoenix

**Website:** https://phoenix.arize.com
**Type:** LLM observability and evaluation

**Key Features:**
- Embedding drift detection
- LLM evaluation with custom metrics
- Trace visualization
- RAG analysis (retrieval quality, answer quality)
- Open-source (self-hostable)
- OpenTelemetry compatible

```python
import phoenix as px
from phoenix.trace import TraceDataset
from phoenix.trace.dsl import SpanQuery

# Launch Phoenix UI
px.launch_app()

# Log traces from LangChain
from phoenix.trace.langchain import LangChainInstrumentor

LangChainInstrumentor().instrument()

# Query traces
trace_df = px.Client().get_trace_dataframe()

# Analyze specific spans
retriever_spans = px.Client().query_spans(
    SpanQuery().where("span_kind == 'RETRIEVER'")
)

# Evaluate with custom evaluator
from phoenix.evals import (
    QAEvaluator,
    RelevanceEvaluator,
    HallucinationEvaluator
)

qa_eval = QAEvaluator(model="gpt-4")
relevance_eval = RelevanceEvaluator(model="gpt-4")
hallucination_eval = HallucinationEvaluator(model="gpt-4")

eval_results = qa_eval.run(
    dataframe=test_df,
    qa_reference=output_col,
    model="gpt-4"
)
```

### 4.4 Promptfoo

**Website:** https://promptfoo.dev
**Type:** LLM testing and evaluation

**Key Features:**
- Prompt testing with multiple providers
- Red-teaming and adversarial testing
- Regression testing
- Side-by-side comparison
- Custom scoring functions
- CI/CD integration

```yaml
# promptfooconfig.yaml
description: "LLM Explainability Evaluation"

prompts:
  - "Explain why the model made this prediction: {{input}}"
  - "What factors contributed to this decision for {{input}}?"

providers:
  - openai:gpt-4
  - openai:gpt-3.5-turbo

tests:
  - vars:
      input: "Loan application with high debt ratio"
    assert:
      - type: contains
        value: "debt"
      - type: llm-rubric
        value: "Explanation mentions specific features and their impact"
  
  - vars:
      input: "Medical image showing possible pneumonia"
    assert:
      - type: contains
        value: "imaging"
      - type: cost
        threshold: 0.05

redteam:
  enabled: true
  plugins:
    - prompt-injection
    - jailbreak
```

```bash
# Run evaluation
promptfoo eval

# Open web UI
promptfoo view

# Run in CI/CD
promptfoo eval --output results.json
```

### 4.5 TransformerLens

**Repository:** https://github.com/neelnanda-io/TransformerLens
**Type:** Mechanistic interpretability for transformers

```python
from transformer_lens import HookedTransformer, ActivationCache
import transformer_lens.utils as utils

# Load model
model = HookedTransformer.from_pretrained("gpt2-small")

# Run with cache
logits, cache = model.run_with_cache("The capital of France is")

# Get all activations
for layer in range(model.cfg.n_layers):
    attn_pattern = cache["attn_pattern", layer]  # (n_heads, seq, seq)
    residual = cache["resid_post", layer]  # (seq, d_model)
    
    print(f"Layer {layer}: attn shape {attn_pattern.shape}")

# ─── Activation Patching ───
def patching_experiment(model, clean_text, corrupted_text, 
                       layer, head, position):
    """
    Patch activation from clean run into corrupted run.
    If this restores the clean prediction, this component is causally important.
    """
    # Clean run
    clean_logits, clean_cache = model.run_with_cache(clean_text)
    clean_pred = clean_logits[0, -1, :].argmax()
    
    # Corrupted run
    corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_text)
    corrupted_pred = corrupted_logits[0, -1, :].argmax()
    
    # Patch: Use clean activation at specific point
    def hook_fn(activation, hook):
        activation[0, position, :] = clean_cache[hook.name][0, position, :]
        return activation
    
    patched_logits = model.run_with_hooks(
        corrupted_text,
        fwd_hooks=[(utils.get_act_name("attn", layer, head), hook_fn)]
    )
    
    patched_pred = patched_logits[0, -1, :].argmax()
    
    return {
        "clean_pred": clean_pred.item(),
        "corrupted_pred": corrupted_pred.item(),
        "patched_pred": patched_pred.item(),
        "recovered": patched_pred.item() == clean_pred.item()
    }

# ─── Circuit Analysis ───
def find_induction_heads(model, text):
    """
    Find induction heads — heads that implement in-context learning.
    Pattern: If AB appears earlier, and current token is A,
    induction heads predict B.
    """
    logits, cache = model.run_with_cache(text)
    
    induction_heads = []
    
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attn = cache["pattern", layer][0, head]  # (seq, seq)
            
            # Check for induction pattern
            # Look for heads that attend to positions after matching earlier tokens
            diag_sum = attn.diag().sum().item()
            
            if diag_sum > 0.3:  # Threshold for induction
                induction_heads.append({
                    "layer": layer,
                    "head": head,
                    "strength": diag_sum
                })
    
    return sorted(induction_heads, key=lambda x: x["strength"], reverse=True)
```

### 4.6 SAELens

**Repository:** https://github.com/apartresearch/sae_lens
**Type:** Training and analyzing sparse autoencoders for LLM interpretability

```python
from sae_lens import SAE, SAEViewer, LanguageModelSAERunnerConfig
import torch

# ─── Load a pre-trained SAE ───
sae = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.8.hook_resid_pre",
    device="cuda"
)

# Get SAE features for a token
with torch.no_grad():
    model_output = sae.model(tokens)
    feature_acts = sae.encode(model_output)
    
    # feature_acts: (batch, pos, n_features)
    # Most features are 0 (sparsity)
    active_features = (feature_acts > 0).float().mean()
    print(f"Sparsity: {1 - active_features:.1%} zeros")

# ─── Feature visualization ───
viewer = SAEViewer(sae)
viewer.show()

# ─── Find features for specific concepts ───
def find_features_for_concept(sae, concept_tokens, model):
    """
    Find SAE features that fire for a specific concept.
    """
    with torch.no_grad():
        # Get model activations
        activations = model.get_activations(concept_tokens)
        
        # Encode to SAE features
        sae_features = sae.encode(activations)
        
        # Find most active features
        feature_importance = sae_features.abs().mean(dim=0)
        
        top_features = feature_importance.topk(10)
        
        return [
            {
                "feature_idx": idx.item(),
                "importance": imp.item(),
                "name": f"feature_{idx.item()}"
            }
            for idx, imp in zip(top_features.indices, top_features.values)
        ]

# ─── Feature steering ───
def steer_generation(model, sae, feature_idx, strength=1.0):
    """
    Amplify or suppress a specific SAE feature during generation.
    """
    def steering_hook(activation, hook):
        # Encode current activation
        features = sae.encode(activation)
        
        # Amplify target feature
        features[:, :, feature_idx] += strength
        
        # Decode back
        steered_activation = sae.decode(features)
        
        return steered_activation
    
    return steering_hook
```

---

## 5. Model-Specific Tools

### 5.1 PyTorch Ecosystem

| Tool | Purpose | Key Feature |
|------|---------|-------------|
| **Captum** | Comprehensive attribution | 15+ attribution methods |
| **TorchRx** | Model rewrites | Model-level debugging |
| **TorchScript** | Model optimization | Production deployment |
| **ONNX Runtime** | Cross-platform inference | Hardware acceleration |

### 5.2 TensorFlow Ecosystem

| Tool | Purpose | Key Feature |
|------|---------|-------------|
| **TF-Explain** | GradCAM, Occlusion | Keras integration |
| **Lucid** | Feature visualization | Neural network dreaming |
| **What-If Tool** | Interactive analysis | Colab integration |
| **TensorBoard** | Training visualization | Embedding projector |

### 5.3 Hugging Face Ecosystem

```python
from transformers import AutoModel, AutoTokenizer
from transformers.interpret import (
    Saliency,
    IntegratedGradients,
    LIMETextExplainer,
    AnchorTextExplainer
)

# Load model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# ─── Saliency ───
saliency = Saliency(model)
saliency_attributions = saliency.attribute(
    "This movie is fantastic!",
    target=1,  # POSITIVE class
    embedding_layer=model.get_input_embeddings()
)

# ─── Integrated Gradients ───
ig = IntegratedGradients(model)
ig_attributions = ig.attribute(
    "This movie is fantastic!",
    target=1,
    n_steps=200
)

# ─── LIME ───
lime_explainer = LIMETextExplainer(model, tokenizer)
lime_attributions = lime_explainer.attribute(
    "This movie is fantastic!",
    target=1
)

# ─── Anchors ───
anchor_explainer = AnchorTextExplainer(model, tokenizer)
anchor = anchor_explainer.attribute(
    "This movie is fantastic!",
    target=1
)
print(f"Anchor: {anchor.anchor}")
print(f"Precision: {anchor.precision:.1%}")
```

---

## 6. Visualization Tools

### 6.1 Attention Visualization

```python
# ─── BertViz ───
from bertviz import model_view, head_view

# Get attention
outputs = model(input_ids, output_attentions=True)
attentions = outputs.attentions  # Tuple of (n_layers) tensors

# Head view (interactive)
head_view(attentions, tokens)

# Model view (overview)
model_view(attentions, tokens)

# ─── Attention Heatmap ───
import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention_heatmap(attention, tokens, layer=0, head=0, save_path=None):
    """Plot attention heatmap for a specific layer and head."""
    attn = attention[layer][0][head].detach().numpy()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        attn,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='viridis',
        ax=ax
    )
    ax.set_title(f'Layer {layer}, Head {head}')
    ax.set_xlabel('Key')
    ax.set_ylabel('Query')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# ─── Attention Rollout ───
def compute_attention_rollout(attentions, add_residual=True):
    """
    Compute attention rollout across all layers.
    """
    # Start with identity
    result = torch.eye(attentions[0].shape[-1])
    
    for attention in attentions:
        # Average across heads
        attention_heads_avg = attention.mean(dim=1).squeeze()
        
        if add_residual:
            # Add residual connection
            augmented_attention = 0.5 * attention_heads_avg + 0.5 * torch.eye(attention_heads_avg.shape[-1])
            # Normalize
            augmented_attention = augmented_attention / augmented_attention.sum(dim=-1, keepdim=True)
        else:
            augmented_attention = attention_heads_avg
        
        result = torch.matmul(augmented_attention, result)
    
    return result
```

### 6.2 SHAP Visualization

```python
import shap
import matplotlib.pyplot as plt

# ─── Summary Plot ───
shap.summary_plot(shap_values, X, feature_names=feature_names)

# ─── Bar Plot (Feature Importance) ───
shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar")

# ─── Force Plot (Individual) ───
shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X[0],
    feature_names=feature_names,
    matplotlib=True
)

# ─── Dependence Plot ───
shap.dependence_plot(
    "feature_0",
    shap_values,
    X,
    feature_names=feature_names
)

# ─── Waterfall Plot ───
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X[0],
        feature_names=feature_names
    )
)

# ─── Beeswarm Plot ───
shap.plots.beeswarm(
    shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=X,
        feature_names=feature_names
    )
)
```

---

## 7. Enterprise XAI Platforms

### 7.1 Platform Comparison

| Platform | Focus Area | Pricing Model | Key Strength |
|----------|-----------|---------------|-------------|
| **Arthur AI** | Model monitoring + XAI | Enterprise license | Real-time monitoring |
| **Fiddler AI** | Explainability platform | Enterprise license | What-if analysis |
| **C3 AI** | Enterprise AI suite | Enterprise license | Full AI lifecycle |
| **H2O.ai** | AutoML + XAI | Enterprise license | Automated explanations |
| **DataRobot** | ML platform | Enterprise license | End-to-end ML |
| **Domino Data Lab** | Data science platform | Enterprise license | Collaborative XAI |
| **Weights & Biases** | Experiment tracking | Freemium | Experiment comparison |
| **Neptune.ai** | MLOps | Freemium | Metadata management |

### 7.2 Enterprise Deployment Patterns

```
Pattern 1: XAI Microservice
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Model      │────▶│   XAI        │────▶│  Frontend    │
│   Service    │     │   Service    │     │  Dashboard   │
└──────────────┘     └──────────────┘     └──────────────┘
       │                    │
       │              ┌─────┴─────┐
       │              │  Cache    │
       │              │  (Redis)  │
       │              └───────────┘
       │
  ┌────┴────┐
  │ Model   │
  │ Registry│
  └─────────┘

Pattern 2: Embedded XAI
┌──────────────────────────────┐
│      Application Server      │
│  ┌─────────┐  ┌─────────┐  │
│  │  Model  │  │  XAI    │  │
│  │  Inference │ │  Engine │  │
│  └────┬────┘  └────┬────┘  │
│       │            │       │
│  ┌────┴────────────┴────┐  │
│  │    Explanation Cache  │  │
│  └──────────────────────┘  │
└──────────────────────────────┘

Pattern 3: Async XAI Pipeline
┌──────────┐     ┌──────────┐     ┌──────────┐
│  Model   │────▶│  Event   │────▶│  XAI     │
│  Service │     │  Queue   │     │  Worker  │
└──────────┘     └──────────┘     └────┬─────┘
                                       │
                                  ┌────┴─────┐
                                  │  Store   │
                                  │  (S3)    │
                                  └──────────┘
```

---

## 8. Tool Selection Guide

### 8.1 Decision Matrix

| Scenario | Primary Tool | Secondary Tool | Avoid |
|----------|-------------|----------------|-------|
| Tabular + tree model | TreeSHAP | InterpretML | LIME (less stable) |
| Tabular + any model | KernelSHAP | Alibi Anchors | Raw feature importance |
| Image classification | GradCAM (Captum) | Integrated Gradients | Attention visualization |
| Text classification | HuggingFace XAI | SHAP + tokenizer | LIME for text |
| LLM individual output | Custom SHAP tokenizer | Counterfactual | Attention alone |
| LLM agent chain | LangSmith/Langfuse | Chain-of-thought viz | None — use multi-method |
| LLM debugging | Langfuse (self-hosted) | Arize Phoenix | Closed-source only |
| Mechanistic interpretability | TransformerLens | SAELens | Feature importance |
| Regulated decision | Global + Local SHAP + Counterfactual | Model card | Any single method |
| Real-time system | Pre-computed SHAP + cache | Cached explanations | KernelSHAP (too slow) |
| RAG explainability | LangSmith + custom attribution | Langfuse | None |
| Enterprise production | Arthur AI / Fiddler | Custom + SHAP | Self-built only |

### 8.2 Cost vs. Capability

| Tool | Cost | Capability | Best For |
|------|------|-----------|----------|
| **SHAP** | Free | ★★★★☆ | Feature attribution |
| **LIME** | Free | ★★★☆☆ | Quick local explanations |
| **Captum** | Free | ★★★★☆ | PyTorch models |
| **InterpretML** | Free | ★★★★☆ | Glass-box models |
| **Langfuse** | Free (self-hosted) | ★★★☆☆ | LLM tracing |
| **LangSmith** | $0.50/1K runs | ★★★★★ | LLM observability |
| **Arize Phoenix** | Free (self-hosted) | ★★★★☆ | LLM monitoring |
| **Arthur AI** | Enterprise | ★★★★★ | Enterprise monitoring |
| **Fiddler AI** | Enterprise | ★★★★★ | Enterprise explainability |

---

## 9. Integration Patterns

### 9.1 Flask/FastAPI Integration

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import shap
import numpy as np

app = FastAPI()

class ExplanationRequest(BaseModel):
    instance: list
    features: list
    n_features: int = 5

class ExplanationResponse(BaseModel):
    prediction: float
    shap_values: dict
    summary: str
    top_features: list

# Global explainer (loaded once)
explainer = None

@app.on_event("startup")
async def load_model():
    global explainer
    # Load model and create explainer
    model = load_model_from_registry()
    background = load_background_data()
    explainer = shap.TreeExplainer(model, data=background)

@app.post("/explain", response_model=ExplanationResponse)
async def explain_prediction(request: ExplanationRequest):
    try:
        instance = np.array(request.instance).reshape(1, -1)
        
        # Get prediction
        prediction = model.predict_proba(instance)[0][1]
        
        # Get SHAP values
        shap_values = explainer.shap_values(instance)
        
        # Build response
        contributions = {}
        for i, feature in enumerate(request.features):
            contributions[feature] = float(shap_values[0][i])
        
        # Sort by importance
        sorted_features = sorted(
            contributions.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )[:request.n_features]
        
        summary = f"Prediction: {prediction:.1%}. Top factors: "
        for feat, val in sorted_features:
            direction = "+" if val > 0 else "-"
            summary += f"{feat}({direction}{abs(val):.3f}), "
        
        return ExplanationResponse(
            prediction=float(prediction),
            shap_values=contributions,
            summary=summary.rstrip(", "),
            top_features=[f[0] for f in sorted_features]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 9.2 Airflow Integration

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def generate_explanations(**context):
    """Batch explanation generation for monitoring."""
    import shap
    import pandas as pd
    
    # Load recent predictions
    predictions = load_recent_predictions(hours=24)
    
    # Generate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(predictions[features])
    
    # Store explanations
    store_explanations(predictions, shap_values)
    
    # Generate alerts for unusual explanations
    alerts = detect_explanation_anomalies(shap_values)
    if alerts:
        send_alert(alerts)

dag = DAG(
    'xai_batch_explanations',
    schedule_interval='0 */6 * * *',  # Every 6 hours
    start_date=datetime(2024, 1, 1),
    catchup=False
)

task = PythonOperator(
    task_id='generate_explanations',
    python_callable=generate_explanations,
    dag=dag
)
```

### 9.3 Docker Compose Setup

```yaml
version: '3.8'

services:
  xai-service:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
      - MODEL_REGISTRY_URL=http://model-registry:8080
    depends_on:
      - redis
      - model-registry
    volumes:
      - ./models:/app/models
      - ./cache:/app/cache

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

  model-registry:
    image: mlflow/mlflow:latest
    ports:
      - "8080:8080"
    volumes:
      - mlflow-data:/mlflow
    command: mlflow server --backend-store-uri sqlite:///mlflow/mlflow.db

  monitoring:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana
    depends_on:
      - xai-service

volumes:
  redis-data:
  mlflow-data:
  grafana-data:
```

---

## 10. Evaluation and Benchmarking Tools

### 10.1 Explanation Quality Evaluation

```python
# ─── Faithfulness Test ───
def test_faithfulness(model, explainer, X_test, n_samples=100):
    """
    Test if explanations are faithful to model behavior.
    Higher correlation = more faithful explanations.
    """
    import numpy as np
    from scipy.stats import spearmanr
    
    faithfulness_scores = []
    
    for i in range(min(n_samples, len(X_test))):
        instance = X_test[i:i+1]
        
        # Get explanation
        explanation = explainer.explain(instance)
        feature_importance = np.array([c["shap_value"] for c in explanation["contributions"]])
        
        # Permutation test: remove each feature and measure accuracy drop
        baseline_acc = model.score(X_test, y_test)
        accuracy_drops = []
        
        for j in range(len(feature_importance)):
            # Remove feature j
            X_modified = X_test.copy()
            X_modified[:, j] = 0  # Or use mean value
            
            modified_acc = model.score(X_modified, y_test)
            accuracy_drops.append(baseline_acc - modified_acc)
        
        accuracy_drops = np.array(accuracy_drops)
        
        # Correlation between attribution and accuracy drop
        correlation, _ = spearmanr(feature_importance, accuracy_drops)
        faithfulness_scores.append(correlation)
    
    return {
        "mean_faithfulness": np.mean(faithfulness_scores),
        "std_faithfulness": np.std(faithfulness_scores),
        "n_samples": n_samples
    }

# ─── Stability Test ───
def test_stability(explainer, X_test, n_runs=10):
    """
    Test if explanations are stable across runs.
    """
    import numpy as np
    
    stability_scores = []
    
    for i in range(len(X_test)):
        instance = X_test[i:i+1]
        
        explanations = []
        for _ in range(n_runs):
            explanation = explainer.explain(instance)
            explanations.append(explanation)
        
        # Check if top features are consistent
        top_features = [
            [c["feature"] for c in exp["contributions"][:3]]
            for exp in explanations
        ]
        
        # Stability = fraction of runs where top 3 features match
        most_common = max(set(map(tuple, top_features)), key=top_features.count)
        stability = top_features.count(most_common) / n_runs
        
        stability_scores.append(stability)
    
    return {
        "mean_stability": np.mean(stability_scores),
        "min_stability": np.min(stability_scores),
        "n_samples": len(X_test)
    }
```

### 10.2 Benchmarking Suite

```python
class XAIBenchmark:
    """
    Comprehensive XAI benchmarking suite.
    """
    
    def __init__(self, model, X_test, y_test, feature_names):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
    
    def run_full_benchmark(self, explainers: dict) -> dict:
        """
        Run complete benchmark across multiple explainers.
        """
        results = {}
        
        for name, explainer in explainers.items():
            print(f"Benchmarking {name}...")
            
            results[name] = {
                "computation_time": self._measure_time(explainer),
                "faithfulness": self._measure_faithfulness(explainer),
                "stability": self._measure_stability(explainer),
                "memory_usage": self._measure_memory(explainer),
            }
        
        return results
    
    def _measure_time(self, explainer) -> dict:
        """Measure computation time."""
        import time
        
        times = []
        for i in range(min(100, len(self.X_test))):
            start = time.time()
            explainer.explain(self.X_test[i:i+1])
            times.append(time.time() - start)
        
        return {
            "mean_ms": np.mean(times) * 1000,
            "median_ms": np.median(times) * 1000,
            "p95_ms": np.percentile(times, 95) * 1000
        }
    
    def _measure_faithfulness(self, explainer) -> float:
        """Measure explanation faithfulness."""
        # Implementation from faithfulness test above
        pass
    
    def _measure_stability(self, explainer) -> float:
        """Measure explanation stability."""
        # Implementation from stability test above
        pass
    
    def _measure_memory(self, explainer) -> dict:
        """Measure memory usage."""
        import tracemalloc
        
        tracemalloc.start()
        
        for i in range(min(10, len(self.X_test))):
            explainer.explain(self.X_test[i:i+1])
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return {
            "current_mb": current / 1024 / 1024,
            "peak_mb": peak / 1024 / 1024
        }
    
    def generate_report(self, results: dict) -> str:
        """Generate human-readable benchmark report."""
        report = ["# XAI Benchmark Report\n"]
        
        for name, metrics in results.items():
            report.append(f"## {name}")
            report.append(f"- Computation: {metrics['computation_time']['mean_ms']:.1f}ms mean")
            report.append(f"- Faithfulness: {metrics['faithfulness']:.3f}")
            report.append(f"- Stability: {metrics['stability']:.3f}")
            report.append(f"- Memory: {metrics['memory_usage']['peak_mb']:.1f}MB peak")
            report.append("")
        
        return "\n".join(report)
```

---

## Summary: Tool Ecosystem Quick Reference

```
Choose Your XAI Stack:

For Feature Attribution:
  → SHAP (TreeSHAP for speed, KernelSHAP for flexibility)

For LLM Observability:
  → LangSmith (managed) or Langfuse (self-hosted)

For Model Interpretability:
  → TransformerLens + SAELens (research/understanding)

For Enterprise Production:
  → Arthur AI or Fiddler (commercial) + SHAP (open-source backbone)

For Visualization:
  → SHAP plots + Attention heatmap + Counterfactual UI

For Evaluation:
  → Promptfoo (LLM testing) + Custom faithfulness tests

For Regulated Industries:
  → SHAP + Counterfactuals + Audit logging + Compliance reports
```

---

*Last updated: July 7, 2026*
*Part of the AI Base Knowledge Library — Category 64*
