# Causal Inference and Machine Learning

## Table of Contents
1. [Introduction](#1-introduction)
2. [Causal vs Statistical](#2-causal-vs-statistical)
3. [Causal Graphs (DAGs)](#3-causal-graphs-directed-acyclic-graphs)
4. [Do-Calculus and Identification](#4-do-calculus-and-identification)
5. [Potential Outcomes Framework](#5-potential-outcomes-framework-rubin-causal-model)
6. [Treatment Effect Estimation](#6-treatment-effect-estimation)
7. [Causal Discovery](#7-causal-discovery)
8. [Causal ML in Practice](#8-causal-ml-in-practice)
9. [Code Examples](#9-code-examples)
10. [Practical Guidance](#10-practical-guidance)
10a. [Deep Learning and Causal Inference](#10a-deep-learning-and-causal-inference)
11. [Cross-References](#11-cross-references)

---

## 1. Introduction
Causal inference studies cause-and-effect relationships: what happens to Y if we change X? This is fundamentally different from prediction, which asks: given X, what is Y? Causal reasoning powers: A/B testing, drug effectiveness, policy evaluation, and decision-making under intervention.

**Key Problem:** Correlation ≠ causation. Ice cream sales and drowning deaths are correlated (both increase in summer), but banning ice cream won't reduce drownings.

**Why causal inference matters for ML:**
- ML models excel at prediction but fail under distribution shift or intervention
- Causal structure enables robust generalization, counterfactual reasoning, and explainability
- Decisions (treat a patient, launch a marketing campaign, set a price) require causal estimates, not just predictions

**Three levels of causation (Pearl's Causal Hierarchy):**
| Level | Name | Question | Example |
|-------|------|----------|---------|
| 1 | Association | P(Y \| X) — What is Y if I see X? | "Does aspirin lower fever?" |
| 2 | Intervention | P(Y \| do(X)) — What is Y if I set X? | "If I give aspirin, does fever drop?" |
| 3 | Counterfactual | P(Y_x \| X', Y') — What would Y have been if X were different? | "Would fever have dropped if I hadn't given aspirin?" |

---

## 2. Causal vs Statistical

| Aspect | Statistical | Causal |
|--------|:-----------:|:------:|
| **Question** | P(Y \| X)? | P(Y \| do(X))? |
| **Data** | Observational only | Observational + intervention |
| **Answer** | Association | Effect |
| **Assumptions** | Standard stats | Causal assumptions (graph, ignorability) |
| **Identifiability** | Always identifiable from data | Requires identifying assumptions |
| **Interpretation** | "If we observe X, Y tends to be..." | "If we set X, Y would be..." |
| **Confounding** | Ignores or adjusts mechanically | Explicitly models confounding structure |

### 2.1 Simpson's Paradox
A classic example where statistical and causal reasoning diverge:

| Group | Treated Recovery | Control Recovery | Effect |
|-------|:-:|:-:|:-:|
| Women | 80/100 (80%) | 40/50 (80%) | **0%** |
| Men | 30/50 (60%) | 80/100 (80%) | **-20%** |
| **Combined** | **110/150 (73%)** | **120/150 (80%)** | **-7%** |

Each subgroup shows null or negative treatment effect, but combined data shows a positive effect (Simpson's Paradox). Resolving this requires causal reasoning: if gender affects both treatment assignment and recovery, we must condition on it (back-door adjustment).

---

## 3. Causal Graphs (Directed Acyclic Graphs)
Nodes = variables, Edges = causal relationships.

### 3.1 Three Fundamental Structures
- **Chain:** X → Z → Y (X causes Z causes Y). Conditioning on Z blocks the path.
- **Fork (confounder):** X ← Z → Y (Z causes both X and Y). Conditioning on Z removes spurious correlation.
- **Collider:** X → Z ← Y (X and Y both cause Z). Conditioning on Z induces spurious correlation.

### 3.2 d-Separation
A path is blocked if it contains a non-collider that's conditioned on, or a collider that's not conditioned on. If all paths between two sets are blocked, they're d-separated (conditionally independent).

### 3.3 Graphical Criteria for Confounding

| Graph Type | Path Between X and Y | Adjustment Needed |
|------------|---------------------|-------------------|
| Simple confounder | X ← Z → Y | Condition on Z |
| Mediator | X → M → Y | Do NOT condition on M (blocks causal path) |
| M-bias | X ← Z₁ → Z₃ ← Z₂ → Y | Do NOT condition on Z₃ (opens collider path) |
| Confounder + mediator | Both paths present | Only condition on confounders, not mediators |

### 3.4 Building a Causal Graph — Practical Steps
1. **Define variables:** Treatment X, outcome Y, and all potential confounders, mediators, colliders
2. **Specify direction:** Based on domain knowledge (temporal order helps: causes precede effects)
3. **Check for cycles:** Ensure graph is acyclic (no feedback loops)
4. **Identify adjustment set:** Use d-separation and back-door criterion to find which variables to condition on

---

## 4. Do-Calculus and Identification

### 4.1 The `do()` Operator
The `do(X = x)` operator represents an **intervention** that sets X to value x, removing all external influences on X. In a DAG, this is equivalent to:
- Removing all edges pointing **into** X (X becomes independent of its causes)
- Setting X to the fixed value x

### 4.2 Back-Door Criterion
A set of variables Z satisfies the back-door criterion for estimating P(Y | do(X)) if:
1. **No element of Z is a descendant of X** (no mediators or colliders downstream of X)
2. **Z blocks every back-door path** between X and Y (paths with arrows into X)

**Adjustment formula:**
```
P(Y | do(X = x)) = ∑_z P(Y | X = x, Z = z) P(Z = z)
```

**Example (confounder):** In a DAG with X ← Z → Y, Z satisfies the back-door criterion.
```python
# Back-door adjustment: stratify by confounder Z
ate = 0
for z in unique_values_of_Z:
    ate += (E[Y | X=1, Z=z] - E[Y | X=0, Z=z]) * P(Z=z)
```

### 4.3 Front-Door Criterion
When a confounder is **unobserved** (hidden), the front-door criterion offers an alternative. Z satisfies the front-door criterion if:
1. **Z mediates the effect of X on Y** (X → Z → Y)
2. **No unblocked back-door paths from X to Z**
3. **All back-door paths from Z to Y are blocked by X**

**Adjustment formula:**
```
P(Y | do(X = x)) = ∑_z P(Z = z | X = x) ∑_x' P(Y | X = x', Z = z) P(X = x')
```

**When to use front-door:**
- There is an unmeasured confounder U affecting both X and Y
- A mediator Z exists that fully captures the X→Y effect
- Example: Smoking (X) → Tar deposits (Z) → Lung cancer (Y), with unobserved genetic confounder (U)

### 4.4 Do-Calculus Rules (Pearl, 1995)
Three rules that together allow (or prove impossible) any causal effect identification:

| Rule | Name | Transformation | Condition |
|------|------|---------------|-----------|
| 1 | Insert/delete observations | P(Y \| do(X), Z, W) = P(Y \| do(X), W) | Z and Y are d-separated by X, W in mutilated graph |
| 2 | Action/observation exchange | P(Y \| do(X), do(Z), W) = P(Y \| do(X), Z, W) | Z acts as an instrumental variable-like condition |
| 3 | Insert/delete actions | P(Y \| do(X), do(Z), W) = P(Y \| do(X), W) | No causal path from Z to Y avoiding X |

Do-calculus is **complete** — if a causal effect cannot be identified by repeated application of these three rules, it is not identifiable from observational data alone.

### 4.5 Identification Summary

| Scenario | Criterion | Condition | What to Adjust For |
|----------|-----------|-----------|-------------------|
| All confounders observed | Back-door | Z blocks all back-door paths | Z (confounders) |
| Unobserved confounder + mediator | Front-door | Z mediates X→Y, no X→Z confound | Z (mediator) with special formula |
| Instrument available | IV | Z affects X, Z ⊥ U, Z → Y only via X | None (use 2SLS) |
| No identification possible | — | Neither criterion satisfied | Cannot estimate from obs data alone |

---

## 5. Potential Outcomes Framework (Rubin Causal Model)

For each unit i:
- Y_i(1): outcome if treated
- Y_i(0): outcome if control
- Individual treatment effect: τ_i = Y_i(1) - Y_i(0)
- **Problem:** we only observe one of Y_i(1), Y_i(0) — the **fundamental problem of causal inference**

### 5.1 Average Treatment Effect (ATE)
ATE = E[Y(1) - Y(0)] — average effect across the entire population

### 5.2 Conditional Average Treatment Effect (CATE)
CATE(x) = E[Y(1) - Y(0) | X = x] — effect for a subgroup with features x
Also called: heterogeneous treatment effect, uplift, individual treatment effect (ITE)

### 5.3 Other Key Estimands

| Estimand | Definition | Use Case |
|----------|------------|----------|
| **ATE** | E[Y(1) - Y(0)] | Overall policy decision |
| **ATT** | E[Y(1) - Y(0) \| T = 1] | Effect on the treated (program evaluation) |
| **ATU** | E[Y(1) - Y(0) \| T = 0] | Effect if we treated the untreated |
| **CATE** | E[Y(1) - Y(0) \| X = x] | Personalized treatment decisions |
| **QTE** | Quantile treatment effects | Effect on distribution tails |
| **LATE** | E[Y(1) - Y(0) \| compliers] | Effect for those who comply with instrument |

### 5.4 Key Assumptions

| Assumption | Meaning | How to Check / Satisfy |
|------------|---------|------------------------|
| **Ignorability (unconfoundedness)** | No unmeasured confounders: Y(1), Y(0) ⟂ T \| X | Domain knowledge; sensitivity analysis (E-value) |
| **Positivity (overlap)** | 0 < P(T = 1 \| X = x) < 1 for all x | Check propensity score range; trim non-overlap |
| **Consistency (SUTVA)** | Y_i = Y_i(T_i); well-defined treatment | Stable unit treatment value assumption; no hidden variations |
| **No interference** | One unit's treatment doesn't affect another's outcome | Check for spillover effects; use cluster randomization |

**E-value sensitivity analysis:** The minimum strength of association an unmeasured confounder would need (on the risk ratio scale) to explain away the observed effect. An E-value of 3 means the confounder would need at least a 3-fold association with both T and Y to fully negate the result.

---

## 6. Treatment Effect Estimation

### 6.1 Experimental (RCT)
Gold standard: Randomize treatment assignment → no confounding
Analysis: simple difference in means

```
ATE = mean(Y | T=1) - mean(Y | T=0)
```

**Challenges:**
- Costly, sometimes unethical, often limited sample size
- External validity (generalizability) may be limited
- Non-compliance, attrition, Hawthorne effects

### 6.2 Observational Methods

| Method | Idea | When to Use |
|--------|------|-------------|
| **Matching** | Match treated/control on covariates | High-dimensional covariates |
| **Propensity Score Matching** | Match on P(treatment \| X) | Many covariates |
| **IPW (Inverse Probability Weighting)** | Weight by 1/propensity score | Few treated units |
| **Doubly Robust (AIPW)** | Combine outcome model + propensity model | Best of both — consistent if either model correct |
| **IV (Instrumental Variables)** | Use instrument Z that affects treatment but not outcome | Unmeasured confounding |
| **DiD (Difference-in-Differences)** | Compare treated/control pre/post | Longitudinal data with parallel trends |
| **RDD (Regression Discontinuity)** | Compare units just above/below threshold | Policy changes at a known cutoff |
| **Synthetic Control** | Weighted combination of control units | Single treated unit (e.g., one state adopts a policy) |

#### 6.2.1 Propensity Score Methods — Detail

**Propensity score:** e(x) = P(T = 1 | X = x)

**Propensity Score Matching (PSM):**
1. Estimate e(x) with logistic regression or ML classifier
2. For each treated unit, find control unit with nearest e(x) (caliper matching)
3. Estimate ATE as mean difference in matched pairs
4. Check **balance** (standardized mean differences < 0.1 after matching)

**Inverse Probability Weighting (IPW):**
```
ATE = E[ (T * Y) / e(X) - ((1 - T) * Y) / (1 - e(X)) ]
```
- Units with low probability of their observed treatment get higher weight
- Issue: extremely small propensities cause unstable weights → trim or use stabilized weights

**Doubly Robust (AIPW):**
Combines an outcome model μ(X) = E[Y | X, T] with the propensity score e(X):
```
ATE = 1/n ∑ [ T_i(Y_i - μ_1(X_i))/e(X_i) + μ_1(X_i) - (1-T_i)(Y_i - μ_0(X_i))/(1-e(X_i)) - μ_0(X_i) ]
```
- **Double robustness:** estimator is consistent if *either* the outcome model *or* the propensity model is correctly specified
- Generally preferred over pure matching or IPW in practice

#### 6.2.2 Instrumental Variables (IV)

**Setup:** Z is a valid instrument if:
1. **Relevance:** Z → X (Z affects treatment)
2. **Exclusion:** Z → Y only through X (no direct path)
3. **Independence:** Z ⟂ all confounders U

**Two-Stage Least Squares (2SLS):**
```
Stage 1: X = α + β Z + γ W + ε
Stage 2: Y = τ * X̂ + γ' W + δ
```

**Weak instruments problem:** If Z has weak correlation with X, the IV estimate is biased and has high variance. Rule of thumb: F-statistic in Stage 1 should be > 10.

**LATE interpretation:** IV estimates the treatment effect for **compliers** — units who take treatment iff assigned by the instrument (not always-takers or never-takers).

#### 6.2.3 Difference-in-Differences (DiD)

**Key assumption: Parallel trends** — in the absence of treatment, the treated group would have followed the same trend as the control group.

```
DiD = E[Y_post - Y_pre | T=1] - E[Y_post - Y_pre | T=0]
```

**Extensions:**
- **Event study:** DiD with multiple time periods, including leads and lags
- **Staggered DiD:** Treatment rolls out at different times (Callaway-Sant'Anna estimator)
- **Synthetic DiD:** Combines synthetic control with DiD

### 6.3 Causal ML (Meta-Learners)

| Learner | Approach | Best For |
|---------|----------|----------|
| **S-Learner** | Single model with treatment as feature | Simple, when T confounds X lightly |
| **T-Learner** | Two models (control, treated) | Strong treatment interactions |
| **X-Learner** | Cross-predictions + propensity weighting | Unbalanced treatment/control |
| **Uplift Tree** | Tree split criterion maximizes treatment effect divergence | Marketing, targeting |
| **Causal Forest** | Random forest for CATE (honest splitting) | SOTA non-linear effects, high-dim |
| **DR-Learner** | Doubly robust scores + regression | Best asymptotic properties |

#### 6.3.1 Meta-Learner Details

**S-Learner:**
```python
model = RandomForestRegressor()
model.fit(X, T, Y)  # T is a feature
# CATE(x) = model.predict(x, 1) - model.predict(x, 0)
```
- Pro: simple, uses all data
- Con: may ignore T if X is high-dimensional

**T-Learner:**
```python
model0 = RandomForestRegressor().fit(X[T==0], Y[T==0])
model1 = RandomForestRegressor().fit(X[T==1], Y[T==1])
# CATE(x) = model1.predict(x) - model0.predict(x)
```
- Pro: flexible, separate models for each group
- Con: each model trained on subset of data

**X-Learner:**
1. Train T-learner models μ₀, μ₁
2. Compute imputed treatment effects:
   - τ̃₁ = Y_i - μ₀(X_i) for treated units
   - τ̃₀ = μ₁(X_i) - Y_i for control units
3. Train models on τ̃₁ and τ̃₀ using propensity-weighted combination

**Causal Forest (Athey & Imbens, 2016):**
- Builds on honest random forests (separate data for splitting vs. estimation)
- Each leaf estimates CATE by within-leaf treatment/control difference
- Provides asymptotic normality and confidence intervals
- Handles high-dimensional X and non-linear effects naturally

---

## 7. Causal Discovery
Learn causal structure from observational data.

| Algorithm | Type | Output | Assumptions |
|-----------|------|--------|-------------|
| **PC algorithm** | Constraint-based (conditional independence tests) | CPDAG (equivalence class) | Faithfulness, sufficiency |
| **GES (Greedy Equivalence Search)** | Score-based (search for best graph) | CPDAG | Score decomposability |
| **LiNGAM** | Linear non-Gaussian ICA-based | Full DAG | Linear, non-Gaussian noise |
| **NOTEARS** | Continuous optimization for DAG | Full DAG | Linear, continuous optimization |
| **DAG-GNN** | Non-linear, deep generative | Full DAG | Non-linear, large data |
| **CAM (Causal Additive Model)** | Non-parametric additive | Full DAG | Additive noise structure |

### 7.1 Practical Guidance for Causal Discovery
- **Sample size matters:** PC and GES need hundreds to thousands of samples (n > p often required)
- **Domain knowledge is critical:** Discovery algorithms output equivalence classes (CPDAGs) — we can't distinguish directions in certain structures (e.g., X → Y vs. Y → X when no confounders)
- **Validate with interventions:** If possible, perform a few interventional experiments to orient edges
- **Use with small graphs:** Discovery degrades rapidly when p > 50 variables; use prior knowledge to limit the variable set

---

## 8. Causal ML in Practice

| Application | Method | Impact |
|-------------|--------|--------|
| Drug effectiveness | RCT (gold), Causal Forest (observational) | Treatment effect estimation |
| Marketing campaign | Uplift modeling: who responds to ads? | 20-40% better targeting ROI |
| Pricing elasticity | CATE estimation per customer segment | Optimized pricing |
| Policy evaluation | DiD, RDD, synthetic control | Policy effectiveness |
| Fairness analysis | Causal fairness criteria (counterfactual fairness) | Non-discriminatory AI |
| Recommendation systems | Causal bandits, debiased recommendations | Improved user satisfaction |
| Clinical decision support | CATE-based treatment recommendation | Personalized medicine |

### 8.1 Common Pitfalls

| Pitfall | Description | Mitigation |
|---------|-------------|------------|
| Adjusting for mediators | Blocking causal path of interest | Only adjust for pre-treatment variables |
| Condition on collider | Inducing spurious associations | Use DAG to avoid conditioning on colliders |
| Positivity violations | No overlap in propensity scores | Trim non-overlapping regions, restrict estimand |
| Unmeasured confounding | Critical confounder missing | Sensitivity analysis, IV, or front-door |
| Data dredging | Finding spurious patterns | Pre-register analysis, out-of-sample validation |
| p-hacking with CATE | Fishing for significant subgroups | Pre-specify subgroups, adjust for multiple testing |

---

## 9. Code Examples

### 9.1 DoWhy: Causal Effect Estimation with Graphs

```python
# pip install dowhy
import dowhy
import pandas as pd
import numpy as np

# Generate synthetic data
n = 1000
Z = np.random.normal(size=n)          # confounder
X = 0.5 * Z + np.random.normal(size=n) # treatment
Y = 2.0 * X + 0.8 * Z + np.random.normal(size=n)  # outcome
data = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})

# Step 1: Define causal model with a DAG
model = dowhy.CausalModel(
    data=data,
    treatment='X',
    outcome='Y',
    graph="digraph { Z -> X; Z -> Y; X -> Y }"
)

# Step 2: Identify the causal effect
#   DoWhy automatically finds a valid adjustment set (back-door)
identified = model.identify_effect(proceed_when_unidentifiable=True)
print(f"Identified estimand: {identified.estimand}")

# Step 3: Estimate the effect using back-door adjustment
estimate = model.estimate_effect(identified,
                                 method_name="backdoor.linear_regression")
print(f"ATE estimate: {estimate.value:.3f}")  # Expect ~2.0

# Step 4: Refute the estimate (sensitivity/robustness checks)
refutation = model.refute_estimate(
    identified, estimate,
    method_name="random_common_cause"  # Add random confounder
)
print(f"Refutation p-value: {refutation.new_effect:.3f}")
```

### 9.2 EconML: CATE Estimation with Meta-Learners

```python
# pip install econml
import numpy as np
import pandas as pd
from econml.dr import DRLearner
from econml.metalearners import TLearner, XLearner, SLearner, CausalForest
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Generate heterogeneous treatment effect data
np.random.seed(42)
n = 2000
X = np.random.uniform(0, 1, size=(n, 3))        # features
T = np.random.binomial(1, 0.5, size=n)            # treatment
# True CATE varies with X[:, 0]
true_effect = 1.0 + 2.0 * X[:, 0]
Y = 0.5 + T * true_effect + 0.3 * X[:, 1] + np.random.normal(size=n)

# ---- T-Learner ----
t_learner = TLearner(
    model_regression=RandomForestRegressor(max_depth=5, n_estimators=200)
)
t_learner.fit(Y, T, X=X)
cate_t = t_learner.effect(X)
print(f"T-Learner ATE: {cate_t.mean():.3f} (true: {true_effect.mean():.3f})")

# ---- X-Learner ----
x_learner = XLearner(
    model_regression=RandomForestRegressor(max_depth=5, n_estimators=200),
    model_propensity=RandomForestRegressor(max_depth=3)
)
x_learner.fit(Y, T, X=X)
cate_x = x_learner.effect(X)

# ---- Causal Forest ----
cf = CausalForest(
    n_estimators=500,
    max_depth=10,
    min_samples_leaf=10,
    random_state=42
)
cf.fit(Y, T, X=X)
cate_cf = cf.effect(X)
print(f"Causal Forest ATE: {cate_cf.mean():.3f} (true: {true_effect.mean():.3f})")

# ---- DR-Learner (Doubly Robust, best asymptotic properties) ----
dr_learner = DRLearner(
    model_regression=GradientBoostingRegressor(n_estimators=100, max_depth=4),
    model_propensity=GradientBoostingRegressor(n_estimators=100, max_depth=3)
)
dr_learner.fit(Y, T, X=X)
cate_dr = dr_learner.effect(X)

# Compare CATE accuracy (RMSE of true vs estimated)
print("\nCATE RMSE:")
for name, cate in [("T-Learner", cate_t), ("X-Learner", cate_x),
                    ("Causal Forest", cate_cf), ("DR-Learner", cate_dr)]:
    rmse = np.sqrt(np.mean((cate - true_effect)**2))
    print(f"  {name:15s}: {rmse:.3f}")
```

### 9.3 Instrumental Variables with EconML

```python
from econml.iv.dml import LinearDMLCateEstimator
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor

# Generate IV data with instrument Z
n = 2000
Z = np.random.normal(size=n)                              # instrument
U = np.random.normal(size=n)                              # unmeasured confounder
X = 0.8 * Z + 0.6 * U + np.random.normal(size=n)         # treatment (confounded)
Y = 1.5 * X + 0.7 * U + np.random.normal(size=n)         # outcome
features = np.random.uniform(0, 1, size=(n, 2))          # other features

# Linear DML IV estimator
iv_model = LinearDMLCateEstimator(
    model_y=GradientBoostingRegressor(n_estimators=100),
    model_t=GradientBoostingRegressor(n_estimators=100),
    discrete_treatment=False,
    cv=5
)
iv_model.fit(Y, T, X=features, Z=Z)  # Note: Z=instrumental variables
cate_iv = iv_model.const_marginal_effect()
ate_iv = iv_model.const_marginal_ate()
print(f"IV ATE estimate (constant treatment effect): {ate_iv[0]:.3f} (true: 1.5)")
```

### 9.4 DoWhy with Front-Door Criterion

```python
import dowhy
import numpy as np
import pandas as pd

# Data with unmeasured confounder U
n = 5000
U = np.random.normal(size=n)                  # unobserved confounder
X = U + np.random.normal(size=n)              # treatment
Z = 0.6 * X + np.random.normal(size=n)        # mediator
Y = 0.8 * Z + U + np.random.normal(size=n)    # outcome
data = pd.DataFrame({'X': X, 'Z': Z, 'Y': Y})

# Specify graph with unmeasured confounder (U is latent, not in data)
graph = """
    digraph {
        X -> Z; Z -> Y; X -> Y;  # direct and indirect paths
        U -> X; U -> Y;          # unmeasured confounder (invisible)
    }
"""

model = dowhy.CausalModel(
    data=data,
    treatment='X',
    outcome='Y',
    graph=graph
)

# Identify using front-door (because back-door is blocked by U)
identified = model.identify_effect(method_name="frontdoor")
print(f"Front-door identified estimand: {identified.estimand}")

# Estimate: two-stage regression
estimate = model.estimate_effect(
    identified,
    method_name="frontdoor.two_stage_regression",
    method_params={"first_stage_model": None, "second_stage_model": None}
)
print(f"Front-door ATE estimate: {estimate.value:.3f} (true effect: ~0.48)")
```

### 9.5 Propensity Score Matching (Manual)

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

# Generate data with confounder Z
np.random.seed(42)
n = 1000
Z = np.random.normal(size=n)
# Propensity depends on Z
p = 1 / (1 + np.exp(-(-1 + 0.8 * Z)))
T = np.random.binomial(1, p)
Y = 2.0 * T + 0.5 * Z + np.random.normal(size=n)
data = pd.DataFrame({'T': T, 'Y': Y, 'Z': Z})

# Step 1: Estimate propensity scores
ps_model = LogisticRegression(C=1e6)
ps_model.fit(data[['Z']], data['T'])
data['propensity'] = ps_model.predict_proba(data[['Z']])[:, 1]

# Step 2: Nearest-neighbor matching on logit of propensity
treated = data[data['T'] == 1].copy()
control = data[data['T'] == 0].copy()

nn = NearestNeighbors(n_neighbors=1)
nn.fit(control[['propensity']])
distances, indices = nn.kneighbors(treated[['propensity']])

# Match: for each treated unit, find nearest control (with caliper = 0.05)
caliper = 0.05
matched_indices = []
used_indices = set()
for i, (dist, idx) in enumerate(zip(distances, indices)):
    if dist[0] <= caliper and idx[0] not in used_indices:
        matched_indices.append((i, idx[0]))
        used_indices.add(idx[0])

# Calculate ATT (Average Treatment Effect on the Treated)
matched_treated = treated.iloc[[m[0] for m in matched_indices]]
matched_control = control.iloc[[m[1] for m in matched_indices]]
att = matched_treated['Y'].mean() - matched_control['Y'].mean()
print(f"PSM ATT (caliper={caliper}): {att:.3f} (matched {len(matched_indices)} pairs)")

# Step 3: Balance check — standardized mean difference < 0.1 is good
for col in ['Z']:
    std_diff = (matched_treated[col].mean() - matched_control[col].mean()) / \
               np.sqrt((matched_treated[col].var() + matched_control[col].var())/2)
    print(f"  Std. diff for {col}: {abs(std_diff):.3f} (goal < 0.1)")
```

### 9.6 Difference-in-Differences (DiD)

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Generate DiD data
np.random.seed(42)
n_per_group = 100
time_periods = 2  # pre=0, post=1

data = pd.DataFrame({
    'unit': list(range(n_per_group * 2)),
    'treated': [0]*n_per_group + [1]*n_per_group,
})

# Generate pre/post for each unit
rows = []
for _, row in data.iterrows():
    for t in [0, 1]:
        base = 10 + 2 * row['treated']  # treated have higher baseline
        trend = 3.0 if row['treated'] else 2.0  # parallel-ish trends
        effect = 5.0 if (row['treated'] and t == 1) else 0  # treatment effect
        y = base + trend * t + effect + np.random.normal(0, 1)
        rows.append({
            'unit': row['unit'], 'treated': row['treated'],
            'post': t, 'y': y
        })

did_data = pd.DataFrame(rows)

# DiD via OLS: Y ~ treated + post + treated:post
model = ols('y ~ treated * post', data=did_data).fit()
print(model.summary())

# The coefficient of interest is treated:post (interaction)
did_estimate = model.params['treated:post']
print(f"DiD estimate: {did_estimate:.3f} (true effect: 5.0)")

# Manual calculation
avg_control_pre = did_data[(did_data['treated']==0) & (did_data['post']==0)]['y'].mean()
avg_control_post = did_data[(did_data['treated']==0) & (did_data['post']==1)]['y'].mean()
avg_treat_pre = did_data[(did_data['treated']==1) & (did_data['post']==0)]['y'].mean()
avg_treat_post = did_data[(did_data['treated']==1) & (did_data['post']==1)]['y'].mean()

did_manual = (avg_treat_post - avg_treat_pre) - (avg_control_post - avg_control_pre)
print(f"DiD (manual): {did_manual:.3f}")
```

---

## 10. Practical Guidance

### 10.1 Step-by-Step Causal Analysis Workflow

1. **Define the causal question**
   - What is the intervention (X)? What is the outcome (Y)?
   - What is the target estimand (ATE, ATT, CATE)?
   - For whom are we estimating the effect (population of interest)?

2. **Draw the DAG**
   - Enumerate all relevant variables (treatment, outcome, confounders, mediators, colliders)
   - Use domain expertise + literature review
   - Tools: dagitty.net, DaVE (VS Code), manual drawing

3. **Identify the effect**
   - Apply back-door criterion (if confounders observed)
   - Apply front-door criterion (if unobserved confounder + mediator)
   - Use IV if instrument is available
   - If unidentifiable, consider what additional data or assumptions would help

4. **Choose the estimation method**
   - If RCT data: simple difference in means or regression
   - If observational with few confounders: matching or PSM
   - If many confounders/high-dim: Causal Forest, DR-Learner
   - If longitudinal: DiD, event study
   - If cutoff-based: RDD

5. **Check assumptions**
   - **Positivity:** Plot propensity score distributions; trim if needed
   - **Balance:** After matching/weighting, check covariate balance (std. mean diff < 0.1)
   - **Parallel trends (DiD):** Plot pre-treatment trends for treated vs. control
   - **IV relevance:** Check F-statistic > 10

6. **Sensitivity analysis**
   - Add random common cause (dowhy refutation)
   - Placebo treatment test
   - Subset validation (e.g., leave out one group)
   - E-value calculation

7. **Interpret and communicate**
   - ATE: "The treatment caused an average increase of X in the outcome"
   - CATE: "The effect varies across subgroups; strongest for population A"
   - Always report uncertainty (confidence intervals)
   - Discuss limitations (unmeasured confounding, generalizability)

### 10.2 Method Selection Guide

| Data Type | Sample Size | Confounders | Recommended Method |
|-----------|:-----------:|:-----------:|--------------------|
| RCT | Any | None (randomized) | Difference in means |
| Observational | < 500 | Few | PSM, IPW |
| Observational | 500-5000 | Many | Causal Forest, DR-Learner |
| Observational | > 5000 | High-dim | Causal Forest, X-Learner |
| Longitudinal | > 100 per period | Time-invariant | DiD, Event Study |
| Threshold | Large at cutoff | Local continuity | RDD |
| Confounded + IV | > 1000 | Unmeasured | 2SLS, IV methods |
| Single treated unit | Long pre-period | Regional | Synthetic Control |

### 10.3 Software and Libraries

| Library | Language | Focus | Strengths |
|---------|----------|-------|-----------|
| **DoWhy** | Python | Full pipeline (identification → estimation → refutation) | Graph-based, principled |
| **EconML** | Python | CATE with ML (meta-learners, IV, DML) | Production-ready, scikit-learn API |
| **CausalNex** | Python | Bayesian DAGs + inference | Interpretability |
| **causal-learn** | Python | Causal discovery (PC, FCI, GES, NOTEARS) | Comprehensive discovery |
| **CausalML** | Python | Uplift modeling, causal trees | Marketing focus |
| **YLearn** | Python | Full causal pipeline | Intuitive API |
| **DAGitty** | Web | DAG drawing + identification | Great for learning |
| **Zelig** | R | General causal inference | Legacy, comprehensive |
| **CausalImpact** | R | Bayesian structural time-series | Synthetic control for time-series |

### 10.4 Recommended Learning Path

| Level | Resource | Focus |
|-------|----------|-------|
| **Beginner** | "The Book of Why" (Pearl) | Intuitive causal reasoning |
| **Beginner** | "Causal Inference: The Mixtape" (Cunningham) | Practical, code examples |
| **Intermediate** | "Causal Inference in Statistics" (Pearl) | DAGs, do-calculus formalized |
| **Intermediate** | "Elements of Causal Inference" (Peters) | Foundations + ML connections |
| **Advanced** | "Mostly Harmless Econometrics" (Angrist & Pischke) | Econometric perspective |
| **Advanced** | "Causal Inference: What If" (Hernán & Robins) | Potential outcomes, epidemiology |
| **Practice** | dowhy/examples, EconML notebooks | Hands-on code |

---

## 10a. Deep Learning and Causal Inference

The intersection of deep learning and causal inference has produced powerful methods that combine the representation-learning capabilities of neural networks with rigorous causal reasoning.

### 10a.1 Causal Representation Learning

Causal representation learning aims to recover latent causal variables from high-dimensional observations (images, text, sensor data).

| Approach | Model Type | What It Learns | Training Signal | Scalability |
|:---------|:----------|:---------------|:---------------|:-----------:|
| **iVAE (identifiable VAE)** | Variational autoencoder | Latent factors with causal structure, identifiable up to permutation | Observational data + auxiliary labels (e.g., time, domain index) | Medium (image-scale) |
| **CausalGAN** | Generative adversarial network | Causal generative model: generate counterfactuals by intervening on latent codes | Observational data + known causal graph | Medium |
| **DEAR** | VAE + causal constraints | Disentangled attributes corresponding to causal mechanisms | Paired observations with known interventions | Low (small-scale) |
| **Causal VAE** | VAE with causal prior | Latent representations respecting causal factorization | Known causal graph + observational data | Medium |
| **TCL (Time-Contrastive Learning)** | Contrastive learning | Causal factors from temporal sequences | Time-series data (non-stationary) | High (video-scale) |
| **SlowVAE** | VAE with temporal smoothness | Slow-changing causal factors from video | Video frames (temporal continuity) | High |

**Practical takeaway:** Causal representation learning is still early-stage for large-scale applications. For most practitioners, standard VAEs + post-hoc causal analysis on learned representations is more practical than end-to-end causal representation learning.

### 10a.2 Deep IV (Instrumental Variables)

Deep IV (Hartford et al., 2017) replaces the linear two-stage least squares with neural networks, enabling non-linear IV estimation:

```python
import torch
import torch.nn as nn
import torch.optim as optim

class DeepIV(nn.Module):
    """Deep Instrumental Variables: two-stage neural network."""
    def __init__(self, dim_Z, dim_X, dim_hidden=128):
        super().__init__()
        # Stage 1: Z -> distribution over T
        self.stage1 = nn.Sequential(
            nn.Linear(dim_Z, dim_hidden), nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden), nn.ReLU(),
            nn.Linear(dim_hidden, 1)
        )
        # Stage 2: T + X -> Y
        self.stage2 = nn.Sequential(
            nn.Linear(1 + dim_X, dim_hidden), nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden), nn.ReLU(),
            nn.Linear(dim_hidden, 1)
        )

    def stage1_predict(self, Z):
        return self.stage1(Z)

    def forward(self, Z, X):
        T_hat = self.stage1(Z)
        combined = torch.cat([T_hat, X], dim=-1)
        return self.stage2(combined)

def train_deep_iv(model, Z, T, X, Y, lr=1e-3, epochs=100):
    opt = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        T_pred = model.stage1(Z)
        loss_stage1 = nn.MSELoss()(T_pred, T)
        Y_pred = model(Z, X)
        loss_stage2 = nn.MSELoss()(Y_pred, Y)
        loss = loss_stage1 + loss_stage2
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: S1={loss_stage1.item():.4f} S2={loss_stage2.item():.4f}")
    return model
```

**When to use Deep IV:**
- Traditional linear IV is insufficient (non-linear treatment-response relationship)
- You have a valid instrument Z and sufficient data (n > 1000)
- The treatment T is continuous (not binary)
- You need heterogeneous treatment effects (CATE)

### 10a.3 Causal Effect VAE (CEVAE)

CEVAE (Louizos et al., 2017) uses a variational autoencoder to infer latent confounders from proxies, enabling causal effect estimation when confounders are not directly observed.

**Key limitation:** CEVAE assumes the latent confounder Z fully explains the relationship between X, T, and Y. In practice, this assumption is hard to verify.

### 10a.4 Causal GANs for Counterfactual Generation

Causal GANs learn a generative model of the data that respects causal structure, enabling counterfactual image generation:

| Application | Method | What It Generates | Verification |
|:-----------|:-------|:-----------------|:------------|
| Medical imaging | CausalGAN with known disease DAG | "What would this MRI look like if treated earlier?" | Expert review |
| Facial attribute editing | CausalStyleGAN | "What would this face look like with different lighting?" | Visual inspection |
| Autonomous driving | Causal video GAN | "How would scene change if pedestrian stepped out sooner?" | Simulation fidelity |
| Economics/marketing | Tabular causal GAN | Counterfactual customer behavior under different pricing | Statistical tests |

### 10a.5 Causal Discovery with Deep Learning

Deep learning has improved causal discovery, particularly for non-linear relationships and high-dimensional data:

| Method | Architecture | Handles Non-Linear? | Sample Efficiency | Output |
|:-------|:-----------|:------------------:|:----------------:|:-------|
| **NOTEARS-MLP** | MLP-based continuous DAG optimization | Yes (MLP) | Medium (n ~ 1000) | Full DAG |
| **DAG-GNN** | Graph neural network + VAE | Yes (GNN) | Low (n ~ 5000+) | Full DAG |
| **CORL** | RL over graph space | Yes (via RL policy) | Low (n ~ 10000+) | DAG + order |
| **GraN-DAG** | Gradient-based neural DAG | Yes (neural nets) | Medium (n ~ 1000) | Full DAG |
| **PC with GNN tests** | CI tests with neural test stats | Yes (GNN non-param.) | Medium | CPDAG |
| **AVAE** | VAE over graph posterior | Yes | High (n ~ 500) | Graph posterior |

**Practical recommendation for deep causal discovery:**
- Start with NOTEARS-MLP for continuous data with up to ~100 variables
- Use DAG-GNN if you have >5000 samples and expect complex non-linear relationships
- For very high-dimensional data (images, text), use representation learning first, then apply causal discovery to the learned representations
- Always validate discovered graphs with domain expert knowledge or interventional experiments

### 10a.6 Differentiable Causal Discovery with NOTEARS

NOTEARS reformulates the combinatorial DAG search as a continuous optimization problem:

```python
import torch
import torch.nn as nn
import numpy as np

def notears_mlp(X, lambda1=0.01, rho_max=1e16, h_tol=1e-8, epochs=100):
    d = X.shape[1]
    model = nn.Sequential(
        nn.Linear(d, 10), nn.Sigmoid(),
        nn.Linear(10, 1)
    )

    def dagness(Ws):
        W = Ws[0]
        M = torch.matrix_exp(W * W)
        return torch.trace(M) - d

    def loss_fn(X_pred, X, Ws, rho, alpha):
        recon_loss = nn.MSELoss()(X_pred, X)
        h = dagness(Ws)
        l1_penalty = sum(w.abs().sum() for w in Ws)
        return recon_loss + alpha * h + 0.5 * rho * h**2 + lambda1 * l1_penalty

    rho, alpha, h = 1.0, 0.0, np.inf
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        optimizer.zero_grad()
        X_pred = model(X)
        Ws = [layer.weight for layer in model if isinstance(layer, nn.Linear)]
        loss = loss_fn(X_pred, X, Ws, rho, alpha)
        loss.backward()
        optimizer.step()

        h = dagness(Ws).item()
        if h > 0.25 * h_tol:
            alpha += rho * h
            rho *= 2 if rho < rho_max else rho_max

        if h <= h_tol:
            print(f"DAG found at epoch {epoch}")
            break

    return model, dagness(Ws).item()
```

### 10a.7 Deep Learning for Causal Inference: When and When Not

| Use Case | Deep Learning Approach | Recommended? | Alternative |
|:---------|:----------------------|:-----------:|:------------|
| **High-dimensional confounders** (images, text, genomics) | Causal representation learning + meta-learner | Yes | Manual feature engineering + traditional causal methods |
| **Non-linear IV with valid instrument** | Deep IV (2-stage neural network) | Yes | Linear 2SLS (if linearity holds) |
| **Counterfactual image generation** | Causal GAN / Causal VAE | Yes (if causal graph known) | Non-causal GAN (no guarantees) |
| **Standard ATE/CATE with tabular data** | Deep meta-learners (NN-based T-Learner etc.) | No - Random Forest usually better | Causal Forest, XGBoost learners |
| **Small sample causal inference (n < 500)** | Any deep learning method | No - overfitting risk | Matching, PSM, Bayesian methods |
| **Interpretable causal effects** | Neural network with regularization | No - hard to interpret | Linear DML, Causal Forest with SHAP |

**Key insight:** Deep learning for causal inference is most valuable when dealing with **high-dimensional, unstructured data** (images, text, videos) where traditional causal methods cannot directly handle the input space. For standard tabular causal inference, tree-based methods (Causal Forest, XGBoost) remain state-of-the-art due to better sample efficiency and interpretability.

### 10a.8 Causal Inference in Large Language Models

Large language models present novel opportunities for causal reasoning:

| Capability | Method | Example | Maturity |
|:-----------|:-------|:--------|:--------:|
| **Causal reasoning from text** | LLM prompted to reason about causal relationships | "Does smoking cause cancer? Consider confounders..." | Research - promising but unreliable |
| **Counterfactual generation** | LLM generates alternative scenarios | "What would have happened if treatment had not been administered?" | Research - quality varies |
| **Causal graph extraction from text** | LLM extracts causal relations from unstructured text | "X increases Y, Z decreases Y" to structured graph | Early - inconsistency across prompts |
| **Text-based confounding adjustment** | Extract confounders from text using LLM, then adjust | Extract patient history from clinical notes | Active research |

**Practical recommendation:** Use LLMs as an **aid** (not replacement) for causal inference - they can help with DAG construction, literature review, and counterfactual narrative generation, but should not be relied upon for quantitative causal effect estimation without rigorous validation.

---

## 10b. Causal Inference in Practice: Case Studies and Advanced Topics

### 10b.1 Causal Inference Case Studies

Real-world causal inference applications reveal common patterns and pitfalls that textbook examples miss.

| Domain | Causal Question | Method Used | Key Challenge | Lesson Learned |
|:-------|:---------------|:-----------|:-------------|:---------------|
| **E-commerce** | "Does showing a discount banner increase revenue, or cannibalize full-price sales?" | DiD + Causal Forest (CATE by customer segment) | Seasonality confounded with promotion timing | Use synthetic control for control group; simple DiD understates effect by 40% |
| **Healthcare** | "Does this new drug reduce 30-day readmission rates?" | RCT (gold standard) + IPTW for subgroup analysis | Treatment effect heterogeneity | Pre-register subgroup analyses; use causal forest to discover non-pre-registered subgroups |
| **Marketing** | "Which customers respond to email campaigns vs. should not be contacted?" | Uplift modeling (Causal Forest, X-Learner) | Selection bias — marketing team already targets high-value customers | Build a randomized holdout (5% of list) to debias training data |
| **Policy** | "Did the minimum wage increase reduce employment?" | Synthetic Control + DiD (Callaway-Sant'Anna) | No single comparable control unit | Synthetic control with donor pool of 100+ similar states; placebo tests |
| **Platform** | "Does the new recommendation algorithm increase user engagement?" | Switchback (time-based randomization) + RDD | Network effects — users influence each other, violating SUTVA | Switchback at the time level (alternate algorithm every hour), not user level |
| **Pricing** | "What is the price elasticity of demand for each product category?" | IV (instrument: cost shocks) + CATE estimation | Price is endogenous (correlated with unobserved demand shocks) | Use cost-side instruments (supplier price changes, tariff changes, shipping cost fluctuations) |

### 10b.2 Causal Discovery from Time Series Data

Time series data has unique properties for causal discovery — temporal order provides directional information that cross-sectional data lacks.

| Method | Model Type | Handles Non-Linear? | Handles Latent Confounders? | Assumptions | Best For |
|:-------|:----------|:------------------:|:--------------------------:|:------------|:---------|
| **Granger Causality** | VAR, linear regression | No | No | Stationarity, no instantaneous effects | Simple bivariate time series (economics, neuroscience) |
| **PCMCI** (Runge et al., 2019) | Conditional independence + momentary CI | Yes (via non-parametric tests) | Yes (via momentary CI) | Causal sufficiency within time-lag window; stationarity | Climate science, neuroscience (high-dim time series) |
| **TsLiNGAM** | Linear non-Gaussian | No | No | Linear, non-Gaussian noise, no cycles | Economic time series (stock returns, macro indicators) |
| **DYNOTEARS** | Continuous optimization for time series DAG | No (linear) | No | Linear dynamics, stationary | Medium-dimensional (p<100) financial/economic data |
| **Neural-GC** | Neural networks (RNN, Transformer) | Yes | Partial (via latent variables) | Sufficiently long time series | High-dimensional sensor data (IoT, manufacturing, healthcare) |
| **TCDF** (Temporal Causal Discovery Framework) | Attention-based CNNs | Yes | No | Sufficient data (n > 5000) | Interpretable time-lagged causal discovery |

**Practical workflow for time series causal discovery:**

1. **Domain preparation:** Define lag window (tau_max) based on domain knowledge — too short misses delayed effects, too long introduces noise
2. **Stationarity check:** Apply differencing or detrending if the series is non-stationary (unit root tests: ADF, KPSS)
3. **Run discovery:** Use PCMCI (Python package tigramite) for most applications
4. **Validate:** Check discovered graph against domain knowledge; test stability by bootstrapping over time windows
5. **Interpret:** Directed time-lagged edges give both causal direction and effect timing

### 10b.3 Causal Inference for A/B Testing at Scale

A/B testing (randomized experiments) is the gold standard for causal inference, but real-world deployments face challenges that require causal ML:

| Challenge | Traditional Approach | Causal ML Solution | Impact |
|:----------|:--------------------|:-------------------|:-------|
| **Limited sample for rare events** | Run longer experiment; risk of false negatives | CATE estimation to pool information across segments | 3-5x smaller required sample size for rare outcomes |
| **Network interference** | Cluster randomization (coarse-grained, low power) | Causal inference with network exposure models | 2x power improvement over cluster randomization |
| **Multiple testing** | Bonferroni correction (conservative) | Empirical Bayes shrinkage of segment-level effects | More true discoveries while controlling FDR |
| **Delayed treatment effects** | Decision at fixed horizon | Sequential testing + CATE over time | Earlier detection of long-term effects |
| **Non-compliance** | Intent-to-treat (ITT) — underestimates effect | IV estimation using random assignment as instrument | Recovers treatment-on-treated (TOT) effect |
| **Novelty effects** | Run experiment longer (weeks vs days) | Time-varying CATE estimation; model novelty decay | Earlier reliable read on long-term effect |

```python
# CUPED: Variance reduction for A/B tests using pre-experiment data.
# CUPED typically achieves 30-60% variance reduction, equivalent to 2-5x sample size savings.

import numpy as np
import pandas as pd
from scipy import stats

def cuped_estimate(treatment_outcomes, control_outcomes, pre_treatment_covariates):
    n_treat = len(treatment_outcomes)
    n_control = len(control_outcomes)
    
    # Use a single covariate: pre-experiment value of the same metric
    Y_pre = pre_treatment_covariates.iloc[:, 0].values
    Y_post = np.concatenate([treatment_outcomes, control_outcomes])
    
    # theta = Cov(Y_post, Y_pre) / Var(Y_pre)
    theta = np.cov(Y_post, Y_pre)[0, 1] / np.var(Y_pre)
    
    # CUPED-adjusted outcomes: Y_cv = Y_post - theta * (Y_pre - mean(Y_pre))
    Y_adjusted = Y_post - theta * (Y_pre - np.mean(Y_pre))
    
    adj_treat = Y_adjusted[:n_treat]
    adj_control = Y_adjusted[n_treat:]
    
    ate = np.mean(adj_treat) - np.mean(adj_control)
    var_ate = np.var(adj_treat) / n_treat + np.var(adj_control) / n_control
    var_raw = np.var(treatment_outcomes)/n_treat + np.var(control_outcomes)/n_control
    reduction = var_raw / var_ate
    
    return {
        "ate": ate,
        "std_err": np.sqrt(var_ate),
        "p_value": 2 * (1 - stats.t.cdf(abs(ate / np.sqrt(var_ate)),
                                         df=n_treat + n_control - 2)),
        "variance_reduction_pct": (1 - var_ate / var_raw) * 100,
        "sample_size_reduction_x": reduction
    }
```

### 10b.4 Causal Inference in Recommendation Systems

| Causal Problem | Traditional Approach | Causal Approach | Improvement |
|:---------------|:--------------------|:----------------|:-----------|
| **Popularity bias** | Recommend popular items (rich get richer) | Debiased recommendation via inverse propensity scoring | 30-50% improvement in long-tail item discovery |
| **Exposure bias** | Treat unobserved interactions as negative | Treat unobserved as missing-not-at-random; use propensity-based estimators | 10-20% better generalization |
| **Position bias** | Items in top positions get more clicks regardless of relevance | Position-based propensity model; IPS correction | 15-25% improvement in relevance metrics |
| **Selection bias** | Users self-select which items to rate | Use causal models that account for selection mechanism | More robust offline evaluation |
| **Confounding in interactions** | Collaborative filtering treats interactions as causal | Causal graphs for user-item-context relationships | Better debiasing in production |

**Key algorithm: IPS (Inverse Propensity Scoring) for Debiased Recommendation:**

```python
# IPS estimator for debiasing implicit feedback (e.g., clicks)
# If training data has position bias, items at position 1 get 10x more
# clicks than position 10. IPS corrects by weighting each observation by 1/propensity.

import numpy as np

def debiased_mse(predictions, observations, propensities):
    ips_weight = 1.0 / propensities
    mse = np.mean(ips_weight * (predictions - observations) ** 2)
    return mse

# Propensity estimation typically uses:
# 1. Position-based: P(observed | position) = empirical CTR at each position
# 2. Item popularity-based: P(observed | item) = empirical observation rate
# 3. Model-based: Train a classifier to predict observability from features
```

### 10b.5 Causal Inference Resources and Tooling (2026 Update)

| Resource | Type | Focus | Audience |
|:---------|:----:|:------|:---------|
| **DoWhy** (Python) | Library | Full causal pipeline (graph, identification, estimation, refutation) | Practitioners |
| **EconML** (Python) | Library | CATE estimation with ML (meta-learners, DML, IV) | Data scientists |
| **Tigramite** (Python) | Library | Time series causal discovery (PCMCI, PCMCI+) | Climate/neuroscience researchers |
| **CausalNex** (Python) | Library | Bayesian network learning + inference | Analysts |
| **DAGitty** (Web tool) | Visualization | DAG drawing + automatic adjustment set identification | All causal learners |
| **WhyNot** (Python) | Simulator | Benchmark environments for causal inference methods | Researchers |
| **CausalBench** (Python) | Benchmark | Standardized evaluation of causal discovery methods | Researchers |

---

## 11. Cross-References

| Reference | Description |
|-----------|-------------|
| [01-Foundations/02-Machine-Learning.md] | ML foundations - prediction vs. causal distinction |
| [01-Foundations/05-Training-Methodologies.md] | Causal representation learning, invariant risk minimization |
| [01-Foundations/04-Probability-and-Statistics.md] | Probability foundations, graphical models |
| [01-Foundations/03-Deep-Learning.md] | Deep learning foundations for causal methods |
| [06-Advanced/05-Interpretability.md] | Model interpretability and causal explanations |
| [08-Reference/01-Glossary.md] | Key terms (ATE, CATE, DAG, back-door, etc.) |
| [09-Papers/01-Foundational-Papers.md] | Foundational causal inference papers |

---

*Document version: 3.0 — June 2026 | Major: Added §10b Causal Inference in Practice — real-world case studies (6 domains), time series causal discovery (PCMCI, TCDF, workflow), A/B testing at scale (CUPED implementation), recommendation system debiasing (IPS estimator), updated tooling comparison. Updated Cross-References.]* — June 2026 | Tier 2-3: Gap Fill — Expanded with code examples, do-calculus, back-door/front-door criteria, practical guidance, and §10a Deep Learning and Causal Inference*
