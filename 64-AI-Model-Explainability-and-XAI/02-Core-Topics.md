# AI Model Explainability and XAI at Scale: Core Topics

> **Description:** Deep dive into the core XAI techniques — SHAP, LIME, attention visualization, counterfactual explanations, and feature attribution methods. Includes mathematical foundations, practical implementation guidance, and comparison of approaches for different model types.

---

## Table of Contents

1. [SHAP (SHapley Additive exPlanations)](#1-shap-shapley-additive-explanations)
2. [LIME (Local Interpretable Model-agnostic Explanations)](#2-lime-local-interpretable-model-agnostic-explanations)
3. [Attention Visualization for Transformers](#3-attention-visualization-for-transformers)
4. [Counterfactual Explanations](#4-counterfactual-explanations)
5. [Anchors and Rule-Based Explanations](#5-anchors-and-rule-based-explanations)
6. [Gradient-Based Methods (GradCAM, Integrated Gradients)](#6-gradient-based-methods)
7. [Mechanistic Interpretability](#7-mechanistic-interpretability)
8. [Feature Importance Methods](#8-feature-importance-methods)
9. [Global vs. Local Explanations](#9-global-vs-local-explanations)
10. [Explanation Evaluation and Quality](#10-explanation-evaluation-and-quality)

---

## 1. SHAP (SHapley Additive exPlanations)

### 1.1 Mathematical Foundation

SHAP is grounded in cooperative game theory. The Shapley value assigns each "player" (feature) a fair share of the total "payout" (prediction).

**Formal Definition:**

For a model `f` with features `{1, 2, ..., n}`, the SHAP value for feature `i` is:

```
φ_i = Σ_{S ⊆ N \ {i}} [|S|!(n-|S|-1)!/n!] × [f(S ∪ {i}) - f(S)]
```

Where:
- `N` = set of all features
- `S` = subset of features not including `i`
- `f(S)` = model prediction using only features in `S`
- The coefficient `|S|!(n-|S|-1)!/n!` is the Shapley weight

**Key Properties (AXIOMS):**

| Axiom | Meaning | Why It Matters |
|-------|---------|---------------|
| **Efficiency** | Σ φ_i = f(x) - E[f(X)] | Explanation sums to the prediction deviation from average |
| **Symmetry** | If features i,j contribute equally in all coalitions, φ_i = φ_j | Equal features get equal attribution |
| **Null Player** | If feature i doesn't affect prediction in any coalition, φ_i = 0 | Irrelevant features get zero attribution |
| **Additivity** | φ_i(f+g) = φ_i(f) + φ_i(g) | Works with ensemble models |

### 1.2 SHAP Computation Methods

| Method | Complexity | Exactness | Use Case |
|--------|-----------|-----------|----------|
| **ExactSHAP** | O(2^n) | Exact | Small feature sets (< 15) |
| **KernelSHAP** | O(N × 2^n) | Approximate | Any model, medium features |
| **TreeSHAP** | O(TLD²) | Exact | Tree-based models (XGBoost, LightGBM) |
| **LinearSHAP** | O(n²) | Exact | Linear models |
| **DeepSHAP** | O(n × L) | Approximate | Deep neural networks |
| **PartitionSHAP** | O(N × n) | Approximate | High-dimensional models |

**Where:**
- N = number of background samples
- T = number of trees
- L = number of leaves
- D = maximum depth

### 1.3 SHAP Value Interpretation

```python
# Example: SHAP values for a loan decision
# Model prediction: 0.35 (35% probability of default)
# Base value (population average): 0.25 (25% average default rate)

shap_values = {
    'credit_score': -0.15,   # Pushed prediction DOWN (good credit score)
    'income': -0.08,          # Pushed prediction DOWN (higher income)
    'debt_ratio': +0.22,      # Pushed prediction UP (high debt)
    'employment_years': -0.04, # Pushed prediction DOWN (stable employment)
    'age': +0.02              # Pushed prediction UP (slightly higher risk)
}

# Verification: base + sum of values = prediction
# 0.25 + (-0.15) + (-0.08) + 0.22 + (-0.04) + 0.02 = 0.22
# (approximate, exact values depend on SHAP implementation)
```

### 1.4 SHAP Visualizations

| Visualization | What It Shows | When to Use |
|--------------|---------------|-------------|
| **Summary Plot** | Feature importance + direction for all predictions | Global understanding |
| **Force Plot** | Individual prediction explanation | Explaining single decisions |
| **Dependence Plot** | Feature value vs. SHAP value relationship | Understanding feature effects |
| **Waterfall Plot** | Step-by-step contribution to prediction | Detailed individual explanation |
| **Heatmap** | SHAP values across many predictions | Pattern identification |
| **Beeswarm** | Distribution of SHAP values per feature | Global feature impact |

### 1.5 SHAP Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| **Exponential complexity** (exact) | Infeasible for > 30 features | Use TreeSHAP or KernelSHAP approximation |
| **Background dataset sensitivity** | Results vary with reference data choice | Use representative background dataset |
| **Correlated features** | SHAP values can be unstable | Use PartitionSHAP or grouped features |
| **No temporal ordering** | Ignores time-dependent feature interactions | Use sequential SHAP variants |
| **Computation cost** | Slow for large datasets | Sample-based SHAP, GPU acceleration |

---

## 2. LIME (Local Interpretable Model-agnostic Explanations)

### 2.1 Core Idea

LIME explains individual predictions by fitting a simple interpretable model (e.g., linear regression) in the neighborhood of the instance being explained.

**Algorithm:**
```
LIME Algorithm:
1. Permute data to generate perturbed samples around instance x
2. Get model predictions for perturbed samples
3. Weight samples by proximity to x (kernel function)
4. Fit interpretable model (e.g., linear regression) on weighted samples
5. Return interpretable model as explanation
```

### 2.2 LIME Variants

| Variant | Data Type | Key Innovation |
|---------|----------|---------------|
| **TabularLIME** | Structional data | Perturbs features independently |
| **ImageLIME** | Images | Superpixel-based perturbation |
| **TextLIME** | Text | Word removal/addition perturbation |
| **AnchorLIME** | Any | Combines LIME with anchor rules |

### 2.3 LIME vs. SHAP Comparison

| Aspect | LIME | SHAP |
|--------|------|------|
| **Scope** | Local only | Both local and global |
| **Foundation** | Local linear approximation | Game theory |
| **Stability** | Low (varies across runs) | High (consistent values) |
| **Speed** | Fast | Variable (method-dependent) |
| **Theoretical guarantees** | None | Axiomatic (efficiency, symmetry, etc.) |
| **Implementation complexity** | Simple | Moderate |
| **Best for** | Quick local explanations | Rigorous attribution |

### 2.4 LIME Instability Problem

```
Problem: Running LIME twice on the same instance can yield different explanations.

Cause: Random perturbation sampling + weight kernel sensitivity

Mitigation Strategies:
├── Aggregate multiple LIME runs (≥ 10)
├── Use KernelSHAP instead (more stable)
├── Fix random seed (reproducibility, but not generalizability)
└── Increase perturbation sample size
```

---

## 3. Attention Visualization for Transformers

### 3.1 How Attention Works

Self-attention computes weighted relationships between all tokens in a sequence:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V

Where:
- Q = Query matrix (what am I looking for?)
- K = Key matrix (what do I contain?)
- V = Value matrix (what do I provide?)
- d_k = Key dimension (scaling factor)
```

### 3.2 Attention Visualization Methods

| Method | What It Shows | Limitations |
|--------|--------------|-------------|
| **Raw attention weights** | Token-to-token attention for one head | May not reflect true reasoning |
| **Attention rollout** | Aggregated attention across layers | Assumes attention = importance |
| **Attention flow** | Maximum attention paths through layers | Computationally expensive |
| **Probing classifiers** | What information is encoded at each layer | Requires training probes |
| **Gradient × attention** | Attention weighted by gradient magnitude | Better reflects true importance |

### 3.3 Why Attention ≠ Explanation (Caveat)

Research has shown that attention weights do NOT always correspond to feature importance:

```
Evidence Against Attention as Explanation:
├── Jain & Wallace (2019): Attention weights poorly correlate with gradient-based importance
├── Wiegreffe & Pinter (2019): Attention can be manipulated without changing predictions
├── Bastings & Filippova (2020): Attention is noisy and inconsistent
└── Prime & Mullen (2023): Attention explains surface form, not reasoning
```

**Best Practice:** Use attention visualization as ONE signal among many, not as a standalone explanation.

### 3.4 Practical Attention Analysis

```python
# Pseudo-code for attention analysis
def analyze_attention(model, input_text):
    tokens = tokenize(input_text)
    outputs = model(tokens, output_attentions=True)
    
    # Get attention for all layers and heads
    attentions = outputs.attentions  # (n_layers, n_heads, seq_len, seq_len)
    
    # 1. Average attention across heads (layer-level view)
    avg_attention = attentions.mean(dim=1)  # (n_layers, seq_len, seq_len)
    
    # 2. Attention rollout (aggregate across layers)
    rollout = torch.eye(seq_len)
    for layer_attn in avg_attention:
        # Add residual connection
        augmented = 0.5 * layer_attn + 0.5 * torch.eye(seq_len)
        # Normalize
        augmented = augmented / augmented.sum(dim=-1, keepdim=True)
        rollout = augmented @ rollout
    
    # 3. Identify most attended tokens for each position
    for i, token in enumerate(tokens):
        top_k = rollout[i].topk(5)
        print(f"'{token}' attends most to: {[tokens[j] for j in top_k.indices]}")
```

### 3.5 Mechanistic Interpretability for LLMs

The frontier of transformer interpretability in 2026:

| Technique | What It Reveals | Maturity |
|-----------|----------------|----------|
| **Induction heads** | How in-context learning works | Well-understood |
| **Superposition** | Features compressed into fewer dimensions | Active research |
| **Sparse Autoencoders** | Decompressing superposed features | Rapidly maturing |
| **Circuit analysis** | Input→output computation paths | Research-stage |
| **Logit attribution** | Which internal components predict next token | Production-ready |

---

## 4. Counterfactual Explanations

### 4.1 Definition and Intuition

A counterfactual explanation answers: **"What is the smallest change to the input that would change the model's decision?"**

```
Original: Loan application with income=$55K, credit_score=680
Model: DENIED

Counterfactual: "If your income were $68,000 (instead of $55,000),
your loan would be APPROVED."
Changes needed: income $55K → $68K (minimum viable change)
```

### 4.2 Counterfactual Generation Methods

| Method | Approach | Key Feature |
|--------|----------|-------------|
| **DiCE (Diverse Counterfactual Explanations)** | Optimization-based | Generates multiple diverse counterfactuals |
| **Growing Spheres** | Tree-based search | Finds closest counterfactual efficiently |
| **Actionable Recourse** | Constraint-aware | Only suggests changes user can actually make |
| **Wachter et al.** | Gradient-based | First formal counterfactual method |
| **FACE** | Graph-based | Respects causal constraints |

### 4.3 Counterfactual Quality Criteria

| Criterion | Definition | Why It Matters |
|-----------|-----------|---------------|
| **Validity** | Counterfactual actually changes prediction | Must be actionable |
| **Proximity** | Minimal change from original | Small changes are more plausible |
| **Sparsity** | Few features changed | Easier to act on |
| **Actionability** | Changes are feasible for the user | Cannot suggest impossible changes |
| **Diversity** | Multiple different counterfactuals | User needs options |
| **Causal consistency** | Changes respect causal relationships | "Change income" shouldn't imply "change eye color" |

### 4.4 Counterfactual Constraints

```python
# Example: DiCE with constraints
import dice_ml

# Define constraints
constraints = {
    'income': {
        'actionable': True,      # User can change this
        'range': [30000, 200000]  # Realistic range
    },
    'credit_score': {
        'actionable': True,
        'range': [300, 850]
    },
    'age': {
        'actionable': False,     # Cannot change (immutable)
    },
    'employment_years': {
        'actionable': True,
        'range': [0, 40]
    }
}

# Generate counterfactuals
counterfactuals = dice_exp.generate_counterfactuals(
    query_instance,
    total_CFs=3,
    desired_class="opposite",
    features_to_vary=['income', 'credit_score', 'employment_years'],
    permitted_range={'income': [60000, 200000]}
)
```

---

## 5. Anchors and Rule-Based Explanations

### 5.1 Anchor Concept

An anchor is a **sufficient condition** — a set of feature constraints that, when satisfied, the prediction is almost always the same.

```
Example:
Prediction: "This email is spam"
Anchor: "IF email contains 'free money' AND 'click here' 
         THEN spam (precision: 97%, coverage: 23%)"

Meaning: When both phrases appear, the model predicts spam 97% of the time,
and this covers 23% of all spam predictions.
```

### 5.2 Anchor Algorithm

```
ANCHOR Algorithm:
1. Start with empty rule R = {}
2. Candidate generation: Sample perturbations of instance x
3. For each candidate:
   a. Check if R → prediction(x) holds
   b. Estimate precision and coverage
4. Greedy search: Add features that maximize coverage while maintaining precision
5. Return anchor when precision > threshold (e.g., 0.95)
```

### 5.3 Anchors vs. Other Methods

| Aspect | Anchors | SHAP | LIME | Counterfactual |
|--------|---------|------|------|---------------|
| **Output type** | If-then rule | Numeric values | Linear model | Changed instance |
| **Interpretability** | Very high | Moderate | Moderate | High |
| **Precision guarantee** | Yes (statistical) | No | No | No |
| **Global insight** | No | Yes | No | No |
| **Best for** | Decision boundaries | Feature attribution | Quick local explanation | Actionable advice |

---

## 6. Gradient-Based Methods

### 6.1 Saliency Maps

The simplest gradient-based explanation: compute ∂output/∂input.

```python
# Saliency map for image classification
def saliency_map(model, image, target_class):
    image.requires_grad = True
    output = model(image)
    output[0, target_class].backward()
    
    # Gradient magnitude = importance
    saliency = image.grad.data.abs()
    return saliency
```

**Limitations:** Noisy, saturating gradients, visual artifacts.

### 6.2 Integrated Gradients

Addresses saliency map limitations by integrating gradients along a path from a baseline to the input:

```
IntegratedGradients_i(x) = (x_i - baseline_i) × ∫₀¹ ∂F(baseline + α(x - baseline)) / ∂x_i dα
```

**Key Properties:**
- **Sensitivity:** If a feature matters, it gets non-zero attribution
- **Implementation Invariance:** Same function = same explanation regardless of architecture
- **Completeness:** Sum of attributions = output - baseline output

### 6.3 GradCAM

For CNNs, uses gradient information flowing into the final convolutional layer:

```
GradCAM Steps:
1. Forward pass: Get activations from last conv layer (A^k)
2. Backward pass: Get gradients of output w.r.t. activations (∂y/∂A^k)
3. Global average pool gradients: α^k = mean(∂y/∂A^k)
4. Weighted combination: L_GradCAM = ReLU(Σ_k α^k × A^k)
5. Result: Heatmap showing discriminative regions
```

### 6.4 Comparison of Gradient Methods

| Method | Input Type | Path Required | Baseline Required | Visual Quality |
|--------|-----------|--------------|-------------------|---------------|
| Saliency Map | Any differentiable | No | No | Low (noisy) |
| Guided Backprop | Any differentiable | No | No | Medium |
| Integrated Gradients | Any differentiable | Yes (from baseline) | Yes | High |
| DeepLIFT | Any differentiable | No | Yes (reference) | Medium-High |
| GradCAM | CNNs only | No | No | High (visual) |
| SmoothGrad | Any differentiable | No | No | Medium (denoised) |

---

## 7. Mechanistic Interpretability

### 7.1 What Is Mechanistic Interpretability?

Unlike feature attribution (which tells you WHAT features matter), mechanistic interpretability aims to understand HOW the model computes its outputs — reverse-engineering the actual circuits and algorithms inside neural networks.

### 7.2 Key Techniques (2026)

#### Sparse Autoencoders (SAEs)

SAEs decompose the superposed activations of LLMs into interpretable features:

```python
# Conceptual SAE architecture
class SparseAutoencoder(nn.Module):
    def __init__(self, d_model, n_features):
        super().__init__()
        # Encoder: d_model → n_features (expansion)
        self.encoder = nn.Linear(d_model, n_features)
        # Decoder: n_features → d_model (reconstruction)
        self.decoder = nn.Linear(n_features, d_model)
    
    def forward(self, x):
        # Encode with sparsity constraint
        features = F.relu(self.encoder(x))
        features = topk(features, k=sparsity_k)  # Enforce sparsity
        # Decode
        reconstructed = self.decoder(features)
        return reconstructed, features
    
    def loss(self, x, reconstructed, features):
        # Reconstruction loss + sparsity penalty
        recon_loss = F.mse_loss(reconstructed, x)
        sparse_loss = features.abs().sum(dim=-1).mean()
        return recon_loss + lambda_sparse * sparse_loss
```

**What SAEs reveal:**
- Individual features corresponding to concepts ("the 'Python' feature", "the 'medical' feature")
- Features that fire for specific patterns across different contexts
- How concepts are represented in superposition

#### Circuit Analysis

Identifies minimal computational subgraphs (circuits) that implement specific behaviors:

```
Circuit Analysis Workflow:
1. Identify behavior (e.g., "indirect object identification")
2. Ablation: Remove components, measure impact
3. Patching: Swap activations between clean and corrupted runs
4. Visualization: Map the circuit diagram

Example: Indirect Object Identification Circuit
├── Duplicate Token Heads (early layers)
├── S-Inhibition Heads (middle layers)
└── Name Mover Heads (late layers)
```

#### Activation Patching

The workhorse of mechanistic interpretability:

```python
# Conceptual activation patching
def activation_patch(model, clean_input, corrupted_input, 
                     layer_to_patch, position):
    # Run clean forward pass
    clean_cache = model.run_with_cache(clean_input)
    
    # Run corrupted forward pass
    corrupted_cache = model.run_with_cache(corrupted_input)
    
    # Patch: Use clean activations at specific point
    patched_output = model.run_with_hooks(
        corrupted_input,
        fwd_hooks=[(layer_to_patch, 
                    lambda x, y: y[:,:,position,:] = 
                    clean_cache[layer_to_patch][:,:,position,:])]
    )
    
    # If patched output matches clean → this component is causally important
    return patched_output
```

### 7.3 Mechanistic Interpretability Findings (2026)

| Finding | Implication |
|---------|------------|
| **Induction heads** implement in-context learning | We understand how few-shot learning works |
| **Superposition** compresses features beyond neuron count | Models represent more concepts than they have neurons |
| **Sparse features** are interpretable at scale | SAEs can decompose superposition into meaningful features |
| **Circuits are modular** | Different behaviors use distinct computational paths |
| **Some behaviors are monosemantic** | Certain features have clear, single meanings |
| **Many features are polysemantic** | Most neurons participate in multiple concepts |

---

## 8. Feature Importance Methods

### 8.1 Traditional Feature Importance

| Method | Approach | Pros | Cons |
|--------|----------|------|------|
| **Permutation Importance** | Shuffle feature, measure accuracy drop | Model-agnostic, intuitive | Correlated features misleading |
| **Mean Decrease Impurity** (trees) | Feature contribution to node impurity reduction | Fast for tree models | Biased toward high-cardinality features |
| **Drop-Column Importance** | Retrain model without feature | Most accurate | Extremely expensive |
| **Boruta** | Statistical test against random features | Robust, handles correlations | Slower than permutation |

### 8.2 Permutation Importance Implementation

```python
from sklearn.inspection import permutation_importance

# Compute permutation importance
result = permutation_importance(
    model, X_test, y_test,
    n_repeats=10,
    random_state=42,
    scoring='accuracy'
)

# Results
for i in result.importances_mean.argsort()[::-1]:
    print(f"{feature_names[i]}: "
          f"{result.importances_mean[i]:.3f} "
          f"+/- {result.importances_std[i]:.3f}")
```

### 8.3 Feature Importance for Different Model Types

| Model Type | Recommended Method | Why |
|-----------|-------------------|-----|
| Linear models | Coefficient magnitude | Directly interpretable |
| Tree-based | TreeSHAP or permutation importance | Handles non-linearity |
| Neural networks | Integrated Gradients or DeepLIFT | Handles deep architecture |
| Any model | KernelSHAP | Universal, but slower |
| High-dimensional | Boruta + SHAP | Handles feature selection |

---

## 9. Global vs. Local Explanations

### 9.1 Scope Comparison

| Aspect | Global Explanation | Local Explanation |
|--------|-------------------|------------------|
| **Question answered** | "How does the model work overall?" | "Why THIS prediction?" |
| **Scope** | Entire model behavior | Single prediction |
| **Method** | Feature importance, partial dependence, global SHAP | LIME, local SHAP, attention, counterfactual |
| **Use case** | Model validation, debugging, compliance | Individual decision explanation, recourse |
| **Audience** | Data scientists, regulators | End users, affected individuals |

### 9.2 Global Explanation Methods

| Method | What It Shows | Complexity |
|--------|--------------|-----------|
| **Global SHAP** | Average absolute SHAP values per feature | O(N × n) |
| **Partial Dependence Plot** | Marginal effect of a feature on prediction | O(N × n) |
| **Accumulated Local Effects (ALE)** | Feature effect accounting for correlations | O(N × n) |
| **Global surrogate** | Simple model approximating full model | Varies |
| **Feature interactions** | SHAP interaction values | O(N × n²) |

### 9.3 Explanation Hierarchy

```
Enterprise Explainability Architecture:
├── Level 1: Global Model Documentation
│   ├── Model card (architecture, training data, performance)
│   ├── Feature importance rankings
│   └── Known limitations and failure modes
├── Level 2: Segment-Level Explanations
│   ├── How model behaves for different populations
│   ├── Partial dependence for key features
│   └── Interaction effects
├── Level 3: Individual Predictions
│   ├── SHAP/LIME values for each prediction
│   ├── Counterfactual explanations
│   └── Confidence calibration
└── Level 4: Real-Time Debugging
    ├── Attention visualization (for LLMs)
    ├── Reasoning chain tracing (for agents)
    └── Tool call attribution
```

---

## 10. Explanation Evaluation and Quality

### 10.1 Explanation Quality Metrics

| Metric | Measures | How to Compute |
|--------|---------|---------------|
| **Faithfulness** | Does explanation reflect actual model behavior? | Correlation with ablation performance |
| **Stability** | Do similar inputs get similar explanations? | Explanation variance for similar inputs |
| **Comprehensibility** | Can humans understand the explanation? | User study accuracy/efficiency |
| **Actionability** | Can users act on the explanation? | User study action completion |
| **Completeness** | Does explanation cover all important factors? | Coverage of ablated features |

### 10.2 Evaluation Framework

```
Explanation Evaluation Pipeline:
1. Automated Metrics
   ├── Faithfulness: Perturbation-based (remove top features, measure accuracy drop)
   ├── Stability: Compute explanation variance across similar instances
   └── Efficiency: Computation time per explanation

2. Human Evaluation
   ├── Comprehension Test: "What does the model focus on?"
   ├── Prediction Task: "Can you predict the model's output from the explanation?"
   ├── Trust Calibration: "Do you trust the model more after seeing the explanation?"
   └── Actionability: "What would you change based on this explanation?"

3. Regulatory Compliance
   ├── EU AI Act Article 13 checklist
   ├── Sector-specific requirements
   └── Audit trail completeness
```

### 10.3 Explanation Debugging Checklist

| Check | What to Look For | Red Flags |
|-------|-----------------|-----------|
| **Feature alignment** | Top SHAP features match domain knowledge | Domain-critical features ranked low |
| **Consistency** | Same input → same explanation | High variance across runs |
| **Sensitivity** | Small input changes → small explanation changes | Large swings from minor perturbations |
| **Coverage** | Explanation accounts for prediction magnitude | Sum of values far from prediction |
| **Counterfactual validity** | Suggested changes actually flip prediction | Invalid counterfactuals |
| **Bias check** | Explanations don't reveal protected attributes | Race/gender correlated with explanations |

---

## Summary: Choosing the Right XAI Method

| Scenario | Primary Method | Secondary Method | Avoid |
|----------|---------------|-----------------|-------|
| Tabular data, tree model | TreeSHAP | Permutation importance | LIME (less stable) |
| Tabular data, any model | KernelSHAP | Anchors | Raw feature importance |
| Image classification | GradCAM | Integrated Gradients | Attention visualization |
| Text classification | SHAP + tokenizer | Attention rollout | LIME for text |
| LLM individual output | SHAP + custom tokenizer | Counterfactual | Attention alone |
| LLM agent chain | LangSmith tracing | Chain-of-thought visualization | None — use multi-method |
| Regulated decision | Global SHAP + Local SHAP + Counterfactual | Model card | Any single method alone |
| Real-time system | Pre-computed explanations | Cached SHAP values | KernelSHAP (too slow) |

---

*Last updated: July 7, 2026*
*Part of the AI Base Knowledge Library — Category 64*
