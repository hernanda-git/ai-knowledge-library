# 02 - Machine Learning

> A comprehensive, deeply technical exploration of machine learning — from classical supervised and unsupervised methods to modern self-supervised learning, reinforcement learning, and the statistical foundations that underpin them all.

---

## Table of Contents

1. [Foundations of Machine Learning](#1-foundations-of-machine-learning)
2. [Supervised Learning](#2-supervised-learning)
3. [Unsupervised Learning](#3-unsupervised-learning)
4. [Semi-Supervised Learning](#4-semi-supervised-learning)
5. [Self-Supervised Learning](#5-self-supervised-learning)
6. [Reinforcement Learning](#6-reinforcement-learning)
7. [Statistical Learning Theory](#7-statistical-learning-theory)
8. [Loss Functions](#8-loss-functions)
9. [Optimization Algorithms](#9-optimization-algorithms)
10. [Regularization Techniques](#10-regularization-techniques)
11. [Cross-Validation and Model Selection](#11-cross-validation-and-model-selection)
12. [Ensemble Methods](#12-ensemble-methods)
13. [Practical Considerations](#13-practical-considerations)
14. [Further Reading](#14-further-reading)

---

## 1. Foundations of Machine Learning

### 1.1 What is Machine Learning?

**Machine learning (ML)** is a subset of artificial intelligence focused on algorithms that improve their performance on a task through experience (data). Formally, a computer program is said to learn from experience E with respect to some class of tasks T and performance measure P if its performance on T, as measured by P, improves with experience E (Mitchell, 1997).

The core paradigm is to learn a function `f: X → Y` that maps inputs `x ∈ X` to outputs `y ∈ Y`, parameterized by a set of learnable parameters `θ`:

```
y = f(x; θ)
```

The learning process finds the optimal parameters `θ*` that minimize a loss function `L` over the training data:

```
θ* = argmin_θ Σ_i L(y_i, f(x_i; θ))
```

### 1.2 Categories of Machine Learning

| Category | Data | Labels | Goal |
|---|---|---|---|
| Supervised Learning | x_i, y_i pairs | Required | Predict y from x |
| Unsupervised Learning | x_i only | None | Discover structure in x |
| Semi-Supervised Learning | Few labeled, many unlabeled | Partial | Leverage unlabeled data |
| Self-Supervised Learning | x_i only (with pretext tasks) | Auto-generated | Learn representations |
| Reinforcement Learning | States, actions, rewards | Reward signal | Learn optimal policy |

### 1.3 The Machine Learning Pipeline

A standard ML pipeline consists of:

1. **Problem definition**: Formulate the task, define inputs/outputs
2. **Data collection**: Gather representative samples
3. **Data preprocessing**: Clean, normalize, augment
4. **Feature engineering/selection**: Extract/select informative features
5. **Model selection**: Choose algorithm family
6. **Training**: Optimize parameters on training data
7. **Evaluation**: Assess on held-out data
8. **Hyperparameter tuning**: Optimize meta-parameters
9. **Deployment**: Serve model in production
10. **Monitoring**: Track performance drift

### 1.4 Inductive Bias

Every ML algorithm has an **inductive bias** — a set of assumptions that allows it to generalize beyond training data. Understanding inductive bias is critical for model selection:

- **Linear models**: Assume linear separability
- **Decision trees**: Assume axis-aligned decision boundaries
- **k-NN**: Assume similarity in input space implies similar outputs
- **Neural networks**: Assume compositional hierarchies (via depth)
- **SVM with RBF kernel**: Assume smoothness in transformed space

---

## 2. Supervised Learning

Supervised learning is the most mature and widely deployed category of ML. Given a dataset of input-output pairs `{(x_i, y_i)}_i=1^N`, the goal is to learn a mapping from inputs to outputs.

### 2.1 Regression

**Regression** predicts a continuous output `y ∈ ℝ`.

#### 2.1.1 Linear Regression

The simplest regression model assumes a linear relationship:

```
y = w^T x + b = Σ_j w_j x_j + b
```

where `w` are weights and `b` is bias.

**Ordinary Least Squares (OLS)** minimizes the mean squared error (MSE):

```
L(w, b) = (1/N) Σ_i (y_i - (w^T x_i + b))²
```

The closed-form solution (for the weights) is:

```
w = (X^T X)^{-1} X^T y
```

where `X` is the `N × d` design matrix. This requires `X^T X` to be invertible (full column rank).

**Computational complexity**: O(Nd² + d³) for the closed form, dominated by matrix inversion.

**Limitations**:
- Assumes linear relationship
- Sensitive to outliers (since MSE squares errors)
- Assumes homoscedasticity (constant variance of errors)
- Features should be uncorrelated (multicolinearity inflates variance)

**Diagnostics**:
- R² (coefficient of determination): `1 - SS_res / SS_tot`
- Adjusted R²: penalizes adding irrelevant features
- F-test for overall significance
- t-tests for individual coefficients

#### 2.1.2 Ridge Regression (L2 Regularization)

Adds an L2 penalty to prevent overfitting and handle multicolinearity:

```
L(w) = (1/N) Σ_i (y_i - w^T x_i)² + λ ||w||_2²
```

Closed form: `w = (X^T X + λI)^{-1} X^T y`

The `λI` term ensures `X^T X + λI` is always invertible, even when `X^T X` is singular. Ridge shrinks coefficients toward zero but never exactly to zero.

#### 2.1.3 Lasso Regression (L1 Regularization)

Uses L1 penalty for automatic feature selection:

```
L(w) = (1/N) Σ_i (y_i - w^T x_i)² + λ ||w||_1
```

No closed form — solved via coordinate descent or subgradient methods. Lasso drives coefficients exactly to zero (sparsity). The `λ` parameter controls sparsity: larger `λ` → more zeros.

**Elastic Net** combines L1 and L2:

```
L(w) = MSE + λ₁||w||₁ + λ₂||w||₂²
```

#### 2.1.4 Polynomial Regression

Extends linear regression by adding polynomial features:

```
y = w_0 + w_1 x + w_2 x^2 + ... + w_p x^p
```

Still linear in the parameters (w), but nonlinear in the input (x). Higher degree → more flexibility but risk of overfitting.

#### 2.1.5 Support Vector Regression (SVR)

SVR finds a function `f(x) = w^T x + b` that deviates from the actual targets by at most ε, while being as flat as possible:

```
minimize (1/2)||w||² + C Σ_i (ξ_i + ξ_i*)
```

subject to:
```
y_i - (w^T x_i + b) ≤ ε + ξ_i
(w^T x_i + b) - y_i ≤ ε + ξ_i*
ξ_i, ξ_i* ≥ 0
```

Points within ε of the prediction contribute no loss (ε-insensitive tube). The kernel trick allows nonlinear SVR.

### 2.2 Classification

**Classification** predicts a discrete label `y ∈ {1, ..., K}`.

#### 2.2.1 Logistic Regression

Despite the name, logistic regression is a classification algorithm. It models the probability of a binary outcome:

```
P(y=1|x) = σ(w^T x + b) = 1 / (1 + e^{-(w^T x + b)})
```

where `σ` is the sigmoid/logistic function.

**Loss**: Binary cross-entropy (log loss):

```
L(w) = -(1/N) Σ_i [y_i log(ŷ_i) + (1-y_i) log(1-ŷ_i)]
```

**Multi-class extension (Softmax Regression)**:

```
P(y=k|x) = exp(w_k^T x + b_k) / Σ_j exp(w_j^T x + b_j)
```

Loss: categorical cross-entropy: `L = -Σ_k y_k log(ŷ_k)`

**Properties**:
- Well-calibrated probabilities (if well-specified)
- Linear decision boundary
- Can be regularized (L1, L2, Elastic Net)
- Fast to train (convex optimization via Newton-CG, L-BFGS, SGD)

#### 2.2.2 k-Nearest Neighbors (k-NN)

A non-parametric, instance-based learner:

- **Training**: Store all training examples
- **Inference**: For a new point x, find k closest training points (by Euclidean distance or other metric), vote on label (classification) or average values (regression)

**Key hyperparameters**:
- `k`: Number of neighbors (smaller → more complex, larger → smoother)
- Distance metric: Euclidean, Manhattan, Minkowski, cosine
- Weighting: Uniform or distance-weighted

**Properties**:
- No training phase (lazy learning)
- Decision boundary is piecewise linear
- Sensitive to irrelevant features (curse of dimensionality)
- O(Nd) inference cost (can be reduced with KD-trees, ball trees, ANN)

**Curse of dimensionality**: In high dimensions, distances become nearly uniform, making k-NN ineffective. The volume of space grows exponentially with dimension.

#### 2.2.3 Naive Bayes

A probabilistic classifier based on Bayes' theorem with the **naive** assumption of conditional independence:

```
P(y|x) = P(y) * Π_j P(x_j|y) / P(x)
```

**Variants**:
- **Gaussian NB**: `P(x_j|y) ~ N(μ_{j,y}, σ_{j,y}²)`
- **Multinomial NB**: For discrete counts (text classification)
- **Bernoulli NB**: For binary features

**Properties**:
- Fast to train and predict
- Works well with high-dimensional data (text)
- Strong independence assumption rarely holds, but often works anyway
- Well-calibrated for binary classification

#### 2.2.4 Support Vector Machines (SVM)

SVM finds the **maximum-margin hyperplane** that separates classes:

```
maximize 2/||w||  (margin width)
subject to y_i(w^T x_i + b) ≥ 1, ∀i
```

**Primal problem**:

```
minimize (1/2)||w||²
subject to y_i(w^T x_i + b) ≥ 1
```

**Soft-margin SVM** (allows misclassifications via slack variables):

```
minimize (1/2)||w||² + C Σ_i ξ_i
subject to y_i(w^T x_i + b) ≥ 1 - ξ_i, ξ_i ≥ 0
```

`C` controls the trade-off between margin width and training error.

**Dual formulation** (using Lagrange multipliers α_i):

```
maximize Σ_i α_i - (1/2) Σ_i Σ_j α_i α_j y_i y_j (x_i·x_j)
subject to 0 ≤ α_i ≤ C, Σ_i α_i y_i = 0
```

Decision function: `f(x) = sign(Σ_i α_i y_i (x_i·x) + b)`

**The Kernel Trick**: Replace dot products `(x_i·x_j)` with a kernel function `K(x_i, x_j)` that computes dot products in an implicit high-dimensional feature space:

- **Linear**: `K(x, z) = x^T z`
- **Polynomial**: `K(x, z) = (γ x^T z + r)^d`
- **RBF (Gaussian)**: `K(x, z) = exp(-γ ||x - z||²)`
- **Sigmoid**: `K(x, z) = tanh(γ x^T z + r)`

The RBF kernel corresponds to an infinite-dimensional feature space. `γ` controls the influence radius of each support vector.

**Properties**:
- Effective in high-dimensional spaces
- Memory efficient (uses support vectors only)
- Versatile via kernel trick
- Requires careful hyperparameter tuning (C, γ)
- No direct probability estimates (Platt scaling needed)
- O(N²) to O(N³) training time with kernels

**SVM vs Logistic Regression**:

| Aspect | SVM | Logistic Regression |
|---|---|---|
| Objective | Maximum margin | Maximum likelihood |
| Output | Distance to hyperplane | Calibrated probabilities |
| Kernel trick | Natural | Requires explicit feature expansion |
| Outliers | Less sensitive (hinge loss) | More sensitive |
| Calibration | Needs Platt scaling | Naturally calibrated |

#### 2.2.5 Decision Trees

Decision trees partition the feature space into axis-aligned regions, each assigned a prediction (label or value). They are built recursively:

1. Select the best feature and split point to maximize information gain
2. Split the data
3. Repeat recursively on each child
4. Stop when a criterion is met (max depth, min samples per leaf, purity threshold)

**Split criteria for classification**:

- **Gini impurity**: `G = 1 - Σ_k p_k²` where `p_k` is proportion of class k in a node
- **Entropy**: `H = -Σ_k p_k log(p_k)`
- **Misclassification error**: `E = 1 - max_k p_k`

**Information gain**: `IG = H(parent) - Σ_c (N_c/N) H(child_c)`

**Split criteria for regression**:

- **MSE**: `(1/N) Σ_i (y_i - ȳ)²`
- **MAE**: `(1/N) Σ_i |y_i - ȳ|`

**Tree hyperparameters**:
- `max_depth`: Maximum depth (controls complexity)
- `min_samples_split`: Minimum samples to split
- `min_samples_leaf`: Minimum samples per leaf
- `max_features`: Features considered per split
- `ccp_alpha`: Cost-complexity pruning parameter

**Pruning**: Reduce overfitting by cutting back branches. Cost-complexity pruning finds a subtree that minimizes `R(T) + α|T|` where `R(T)` is training error, `|T|` is number of leaves, and `α` controls the trade-off.

**Properties**:
- Interpretable (can be visualized as rules)
- No feature scaling needed
- Handles mixed data types
- Non-linear, non-parametric
- High variance (small changes in data → different tree)
- Prone to overfitting without pruning

#### 2.2.6 Random Forests

Random forests are an **ensemble** of decision trees, each trained on a bootstrap sample of the data with random feature subsets:

```
for t = 1 to T:
    sample N points with replacement (bootstrap)
    build a decision tree, at each split consider m < d random features
    grow tree to full depth (no pruning)
final prediction = majority vote (classification) or average (regression)
```

**Key insight**: Random forests reduce variance by averaging many high-variance, low-bias trees. The error of a random forest is bounded by:

```
E_rf ≤ ρ * σ² / T
```

where `ρ` is average correlation between trees, `σ²` is individual tree variance, and `T` is number of trees.

**Out-of-bag (OOB) error**: Each tree uses ~63.2% of data (bootstrap). The remaining ~36.8% (OOB) can be used as a validation set without held-out data.

**Feature importance**: Two types:
- **Impurity-based**: Total reduction in split criterion (Gini/entropy) by each feature, averaged across trees
- **Permutation-based**: Drop in accuracy when a feature's values are shuffled

**Properties**:
- Excellent out-of-the-box performance
- Robust to outliers and noise
- Handles high-dimensional data
- Provides feature importance
- Cannot extrapolate beyond training range
- Large model size (many trees)
- Slower inference than single tree

#### 2.2.7 Gradient Boosting Machines (GBM)

Gradient boosting builds an ensemble **sequentially**, where each new tree corrects errors of the previous ensemble:

```
F_0(x) = argmin_γ Σ_i L(y_i, γ)
for m = 1 to M:
    r_im = -[∂L(y_i, F(x_i)) / ∂F(x_i)]_F = F_{m-1}  (pseudo-residuals)
    fit a tree h_m to pseudo-residuals
    find optimal step size: ρ_m = argmin_ρ Σ_i L(y_i, F_{m-1}(x_i) + ρ h_m(x_i))
    F_m(x) = F_{m-1}(x) + ν ρ_m h_m(x)
```

where `ν` is the learning rate (shrinkage).

**Key hyperparameters**:
- `n_estimators` (M): Number of boosting rounds
- `learning_rate` (ν): Shrinkage factor (smaller → more robust, needs more trees)
- `max_depth`: Typically 3-8 (shallower than random forest trees)
- `subsample`: Row sampling for stochastic gradient boosting
- `min_samples_leaf`: Minimum samples per leaf
- `max_features`: Column sampling

**Loss functions**:
- **Regression**: L2 (MSE), L1 (MAE), Huber, Quantile
- **Classification**: Deviance (log loss), Exponential, Binomial

#### 2.2.8 XGBoost

**XGBoost** (Extreme Gradient Boosting) is an optimized implementation of gradient boosting with several innovations:

**1. Regularized objective**:

```
L = Σ_i l(y_i, ŷ_i) + Σ_k [γT_k + (1/2)λ||w_k||² + α||w_k||₁]
```

where `T_k` is number of leaves in tree k, `w_k` are leaf weights.

**2. Second-order approximation** (Newton boosting):

Uses both gradient and Hessian (second derivative) for better convergence:

```
L ≈ Σ_i [g_i f(x_i) + (1/2) h_i f(x_i)²] + Ω(f)
```

where `g_i = ∂L(y_i, ŷ)/∂ŷ` and `h_i = ∂²L(y_i, ŷ)/∂ŷ²`.

**3. Weighted quantile sketch**: Efficient histogram-based split finding for large data.

**4. Column block structure**: Data is pre-sorted into blocks for parallel computation.

**5. Column sub-sampling**: Reduces overfitting and speeds computation.

**6. Sparsity-aware split finding**: Handles missing values natively (learns default direction).

**7. Built-in cross-validation** and early stopping.

**8. System optimizations**: Cache-aware access, out-of-core computation, distributed training.

**Hyperparameters**:
- `max_depth`: Default 6
- `learning_rate` (eta): Default 0.3
- `subsample`: Default 1.0
- `colsample_bytree`, `colsample_bylevel`, `colsample_bynode`
- `min_child_weight`: Minimum sum of instance weight (Hessian) in a child
- `gamma`: Minimum loss reduction required to split
- `lambda` (L2), `alpha` (L1): Regularization on leaf weights
- `tree_method`: 'auto', 'exact', 'approx', 'hist', 'gpu_hist'

**XGBoost vs Standard GBM**:

| Feature | Standard GBM | XGBoost |
|---|---|---|
| Regularization | None | L1 + L2 on leaf weights |
| Tree building | Greedy, level-wise | Level-wise (default) or histogram |
| Missing values | Requires imputation | Learned default direction |
| Parallelism | Limited | Feature parallelism via column block |
| Second-order | No | Yes (Newton) |
| Cross-validation | Manual | Built-in |

#### 2.2.9 LightGBM

**LightGBM** (Light Gradient Boosting Machine) focuses on efficiency and scalability:

**Key innovations**:

**1. Gradient-based One-Side Sampling (GOSS)**: Retains instances with large gradients (under-trained) and randomly samples instances with small gradients (well-trained). This focuses the algorithm on the hardest examples.

**2. Exclusive Feature Bundling (EFB)**: Bundles mutually exclusive features (features that rarely take non-zero values simultaneously) to reduce dimensionality.

**3. Leaf-wise tree growth**: Unlike level-wise growth (XGBoost default), LightGBM grows the leaf with the largest loss reduction, leading to deeper, more asymmetric trees with faster convergence.

```
Level-wise:        Leaf-wise:
    |                   |
   / \                 / \
  /   \               /   \
 / \ / \             /     \
                      \     /
                       \   /
                        \ /
```

**4. Histogram-based splitting**: Continuous features are discretized into bins (typically 256), enabling fast histogram construction and O(bins) split evaluation.

**Properties**:
- Faster training than XGBoost (up to 10x)
- Lower memory usage
- Better accuracy (leaf-wise growth)
- Naturally handles categorical features
- Prone to overfitting with small data (leaf-wise growth on small data is dangerous)
- `num_leaves` should be constrained (< 2^max_depth)

#### 2.2.10 CatBoost

**CatBoost** (Categorical Boosting) handles categorical features exceptionally well:

**Key innovations**:

**1. Ordered Target Encoding**: Standard target encoding causes target leakage. CatBoost uses **ordered** encoding — for each sample, the target statistic is computed only from previous samples in a random permutation.

```
For a categorical feature with categories {A, B, C}:
Standard: cat_mean = (count_positive + prior) / (count_total + 1)
Ordered:  For sample i, use only samples 1...i-1 (in a random permutation)
```

**2. Ordered Boosting**: Similar principle applied to gradient estimation. Traditional boosting uses the same data for computing gradients and building trees (leading to prediction shift). CatBoost uses permutation-based training.

**3. Symmetric trees** (oblivious trees): All nodes at the same depth use the same split condition. This reduces overfitting and speeds inference.

**4. Native support for categorical features**: No preprocessing needed (no one-hot encoding).

**Properties**:
- Best-in-class for datasets with many categorical features
- Less hyperparameter tuning needed (robust defaults)
- Slower than LightGBM on pure numerical data
- GPU-optimized by default

### 2.3 Comparison of Tree-Based Methods

| Algorithm | Variance | Bias | Training Speed | Inference Speed | Interpretability |
|---|---|---|---|---|---|
| Decision Tree | High | Low | Fast | Fast | Excellent |
| Random Forest | Low-Med | Low | Slow (parallel) | Medium | Medium |
| GBM | Low-Med | Low | Medium | Fast | Low |
| XGBoost | Low | Low | Medium | Fast | Low |
| LightGBM | Low | Low | Fast | Fast | Low |
| CatBoost | Low | Low | Medium | Fast | Low |

---

## 3. Unsupervised Learning

Unsupervised learning finds hidden patterns, structure, or representations in unlabeled data.

### 3.1 Clustering

Clustering partitions data into groups (clusters) where intra-cluster similarity is high and inter-cluster similarity is low.

#### 3.1.1 K-Means Clustering

**Algorithm**:

```
1. Initialize K centroids μ_1, ..., μ_K (randomly or via k-means++)
2. Repeat until convergence:
   a. Assignment: assign each point to nearest centroid
      c_i = argmin_k ||x_i - μ_k||²
   b. Update: recompute centroids as mean of assigned points
      μ_k = (1/|C_k|) Σ_{i∈C_k} x_i
```

**K-Means++ Initialization**:
1. Choose first centroid uniformly at random from data points
2. For each subsequent centroid, choose x_i with probability proportional to D(x_i)² (distance to nearest existing centroid)
3. Repeat until K centroids are chosen

This reduces the chance of poor initial centroids compared to random initialization.

**Objective**: Minimize within-cluster sum of squares (WCSS):

```
J = Σ_{k=1}^K Σ_{i∈C_k} ||x_i - μ_k||²
```

**Properties**:
- O(N·K·d·I) complexity (I = iterations)
- Converges to local minimum
- Assumes spherical clusters of similar size
- Sensitive to outliers
- Requires K as input

**Choosing K**:
- **Elbow method**: Plot WCSS vs K, look for "elbow"
- **Silhouette score**: `s = (b - a) / max(a, b)` where a = intra-cluster distance, b = nearest-cluster distance
- **Gap statistic**: Compare WCSS to null reference distribution
- **Davies-Bouldin index**: Ratio of within-cluster to between-cluster distances

**K-Medoids** (PAM): Uses actual data points as centroids (medoids), more robust to outliers.

#### 3.1.2 DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

**Parameters**:
- `ε` (eps): Radius for neighborhood
- `minPts`: Minimum points to form a dense region

**Concepts**:
- **Core point**: Has at least `minPts` points within `ε` (including itself)
- **Border point**: Within `ε` of a core point but has fewer than `minPts` neighbors
- **Noise point**: Neither core nor border

**Algorithm**:

```
1. For each unvisited point p:
   a. Mark p as visited
   b. Find all points within ε of p (neighbors)
   c. If neighbors < minPts, mark p as noise
   d. Else, create new cluster C, expand cluster:
      i.   Add p to C
      ii.  For each q in neighbors:
           - If q is unvisited, mark visited, find its ε-neighbors
           - If q's ε-neighbors >= minPts, add them to neighbors
           - If q is not in any cluster, add q to C
```

**Properties**:
- Does not require K
- Finds arbitrarily shaped clusters
- Robust to outliers
- Deterministic (order-independent for core points)
- Struggles with varying density clusters
- Poor performance in high dimensions (curse of dimensionality)
- `ε` is sensitive: too small → many noise points; too large → one cluster

**HDBSCAN**: Hierarchical extension that builds a cluster hierarchy and extracts flat clustering based on cluster stability. More robust to varying densities.

#### 3.1.3 Hierarchical Clustering

Two approaches:

**Agglomerative (bottom-up)**:

```
1. Start with each point as its own cluster
2. Repeatedly merge the two most similar clusters
3. Stop when K clusters remain or similarity threshold reached
```

**Divisive (top-down)**:

```
1. Start with one cluster containing all points
2. Repeatedly split the most heterogeneous cluster
3. Stop when K clusters formed or all clusters pure enough
```

**Linkage criteria** (agglomerative):

| Linkage | Similarity Definition | Tendency |
|---|---|---|
| Single-link | Min distance between clusters | Produces long, chain-like clusters |
| Complete-link | Max distance between clusters | Produces compact, spherical clusters |
| Average-link | Average distance between clusters | Balance between single and complete |
| Ward's | Variance increase when merging | Minimizes within-cluster variance (similar to K-means objective) |

**Dendrogram**: A tree diagram showing merge/split history. Cluster assignment determined by cutting at a chosen height.

**Complexity**: O(N³) naive, O(N² log N) with priority queues.

#### 3.1.4 Gaussian Mixture Models (GMM)

Models data as a mixture of K Gaussian distributions:

```
P(x) = Σ_{k=1}^K π_k · N(x | μ_k, Σ_k)
```

where `π_k` are mixing coefficients (Σ π_k = 1, π_k ≥ 0).

**Parameters**: `π_k, μ_k, Σ_k` for each component `k`.

**EM Algorithm**:

```
E-step: Compute responsibilities γ_{ik} = P(z_i = k | x_i)
          γ_{ik} = π_k · N(x_i | μ_k, Σ_k) / Σ_j π_j · N(x_i | μ_j, Σ_j)

M-step: Update parameters using weighted MLE
          μ_k = Σ_i γ_{ik} x_i / Σ_i γ_{ik}
          Σ_k = Σ_i γ_{ik} (x_i - μ_k)(x_i - μ_k)^T / Σ_i γ_{ik}
          π_k = (1/N) Σ_i γ_{ik}
```

**Properties**:
- Soft clustering (probabilistic membership)
- Captures elliptical clusters (full covariance matrix)
- Model selection via BIC/AIC
- Sensitive to initialization
- Can degenerate (singular covariance) with insufficient data

### 3.2 Dimensionality Reduction

#### 3.2.1 Principal Component Analysis (PCA)

PCA finds orthogonal directions of maximum variance in the data.

**Computation**:

```
1. Center the data: X_centered = X - μ
2. Compute covariance matrix: C = (1/N) X_centered^T X_centered
3. Eigendecomposition: C = V Λ V^T
4. Select top k eigenvectors (principal components) by largest eigenvalues
5. Project: X_reduced = X_centered · V_k
```

**Variance explained**: `λ_i / Σ_j λ_j` for component i. Cumulative variance guides k selection.

**Properties**:
- Linear, orthogonal transformation
- Unsupervised (ignores labels)
- Decorrelates features
- Maximizes variance, which may not align with class separability

**SVD Approach**: For high-dimensional data (N < d), use SVD instead of eigendecomposition:

```
X = U Σ V^T
Principal components = V_k
Projected data = U_k Σ_k
```

**Kernel PCA**: Uses kernel trick for nonlinear dimensionality reduction:

```
K_centered = K - 1_N K - K 1_N + 1_N K 1_N
Solve: K_centered α = λ α
Projection: z_i = Σ_j α_j K(x_i, x_j)
```

**Incremental PCA**: Processes data in mini-batches for out-of-core computation.

#### 3.2.2 t-SNE (t-Distributed Stochastic Neighbor Embedding)

t-SNE is a nonlinear dimensionality reduction technique optimized for visualization (typically 2D or 3D).

**How it works**:

1. **Compute pairwise similarities in high-dimensional space** using Gaussian kernels:

```
p_{j|i} = exp(-||x_i - x_j||² / 2σ_i²) / Σ_{k≠i} exp(-||x_i - x_k||² / 2σ_i²)
p_{ij} = (p_{j|i} + p_{i|j}) / 2N  (symmetrized)
```

Perplexity controls σ_i: `Perp(P_i) = 2^{H(P_i)}` where H is entropy. Higher perplexity → larger σ.

2. **Initialize low-dimensional points** y_i (random or PCA).

3. **Compute low-dimensional similarities** using Student's t-distribution (heavy tails prevent crowding):

```
q_{ij} = (1 + ||y_i - y_j||²)^{-1} / Σ_{k≠l} (1 + ||y_k - y_l||²)^{-1}
```

4. **Minimize Kullback-Leibler divergence** between P and Q via gradient descent:

```
KL(P||Q) = Σ_{i≠j} p_{ij} log(p_{ij} / q_{ij})
```

**Gradient**:

```
∂KL/∂y_i = 4 Σ_j (p_{ij} - q_{ij})(y_i - y_j)(1 + ||y_i - y_j||²)^{-1}
```

**Properties**:
- Excellent for visualization (preserves local structure)
- Non-convex, sensitive to initialization
- Stochastic (multiple runs give different results)
- O(N²) time and memory (can be approximated with Barnes-Hut or FIt-SNE)
- Global structure is not preserved (distances between clusters are meaningless)
- Perplexity (5-50 typical) controls balance of local vs global structure

**Common pitfalls**:
- Different perplexities give different visualizations
- Cluster sizes in t-SNE are not meaningful
- Distances between clusters are arbitrary
- Random seed can change appearance significantly

#### 3.2.3 UMAP (Uniform Manifold Approximation and Projection)

UMAP is a nonlinear dimensionality reduction technique that preserves more global structure than t-SNE while being faster.

**Key concepts**:

1. **Construct a fuzzy simplicial set** (weighted graph) in high dimensions:
   - For each point, find k nearest neighbors
   - Compute local connectivity based on distance to nearest neighbor
   - Make the graph symmetric via probabilistic union

2. **Optimize low-dimensional representation** to minimize cross-entropy between the high-dimensional and low-dimensional graph:

```
CE(P, Q) = Σ_{ij} [p_{ij} log(p_{ij}/q_{ij}) + (1-p_{ij}) log((1-p_{ij})/(1-q_{ij}))]
```

**Key hyperparameters**:
- `n_neighbors` (k): Larger → more global structure preserved (15 default)
- `min_dist`: Minimum distance between points in low-D (controls cluster tightness)
- `n_components`: Output dimensions

**UMAP vs t-SNE**:

| Aspect | t-SNE | UMAP |
|---|---|---|
| Objective | KL divergence | Cross-entropy (attractive + repulsive) |
| High-D graph | Gaussian kernels | Adaptive local neighborhoods |
| Low-D graph | t-distribution | Modified t-distribution |
| Global structure | Poor | Better |
| Speed | O(N²) (Barnes-Hut: O(N log N)) | O(N log N) |
| Scalability | ~10K points | ~1M+ points |
| Determinism | No (random init) | Partial (can seed) |

---

## 4. Semi-Supervised Learning

Semi-supervised learning leverages a small set of labeled data `(X_l, y_l)` and a large set of unlabeled data `X_u` to improve model performance.

### 4.1 Assumptions

- **Smoothness assumption**: Points close in input space should have similar labels
- **Cluster assumption**: Points in the same cluster likely share a label
- **Manifold assumption**: High-dimensional data lies on a low-dimensional manifold

### 4.2 Self-Training (Pseudo-Labeling)

```
1. Train model on labeled data
2. Use model to predict labels for unlabeled data (pseudo-labels)
3. Add high-confidence predictions to labeled set
4. Retrain and repeat
```

**Risk**: Error reinforcement — incorrect pseudo-labels compound.

**Mitigation**: Use confidence thresholds, uncertainty estimation, or agreement across models.

### 4.3 Co-Training

```
1. Split features into two independent views (sets)
2. Train separate models on each view
3. Each model labels unlabeled data for the other
4. Repeat
```

**Requirement**: Natural feature split exists (e.g., words from two sources describing same web page).

### 4.4 Label Propagation

Construct a graph where nodes are labeled + unlabeled points. Propagate labels via graph diffusion:

```
F = (I - αS)^{-1} Y
```

where `S` is the normalized adjacency matrix, `Y` contains label indicators, and `α` controls propagation strength.

### 4.5 Consistency Regularization

Enforce that the model's predictions are consistent under small perturbations of the input:

```
L = L_supervised(x_l, y_l) + λ * ||f(x_u) - f(x_u + ε)||²
```

where `ε` is small noise (Gaussian, dropout, data augmentation). This is the foundation of **FixMatch**, **MixMatch**, and other modern SSL methods.

**FixMatch**:
- For each unlabeled example, compute pseudo-label from weakly-augmented version
- Enforce prediction consistency on strongly-augmented version
- Only keep pseudo-labels above a confidence threshold

---

## 5. Self-Supervised Learning

Self-supervised learning (SSL) creates supervision signals from unlabeled data by designing **pretext tasks** that require understanding of data structure.

### 5.1 Contrastive Learning

**Core idea**: Pull representations of similar (positive) pairs together, push dissimilar (negative) pairs apart.

**SimCLR** (Chen et al., 2020):

1. Apply random augmentations to each image to create two views
2. Encode both views with a neural network
3. Project to a lower-dimensional space
4. Maximize agreement between positive pairs (same image, different augmentations) using **NT-Xent loss** (normalized temperature-scaled cross-entropy):

```
L(i, j) = -log(exp(sim(z_i, z_j)/τ) / Σ_{k≠i} exp(sim(z_i, z_k)/τ))
```

where `sim()` is cosine similarity, `τ` is temperature.

**Key components**:
- Strong data augmentation
- Large batch size (or memory bank) for many negatives
- Projection head (improves representation quality)
- Temperature scaling

**SimCLR vs Other Methods**:

| Method | Negatives | Architecture | Key Innovation |
|---|---|---|---|
| SimCLR | Batch-level | Standard | NT-Xent, projection head |
| MoCo | Queue | Momentum encoder | Large negative queue |
| BYOL | None | Online + target | Bootstrap without negatives |
| SwAV | None | Online + prototypes | Cluster-based |
| CLIP | Batch-level | Dual encoders | Image-text pairs |

**MoCo** (Momentum Contrast): Maintains a queue of negative samples and a slowly-updated momentum encoder for consistent representations.

**BYOL** (Bootstrap Your Own Latent): Uses an online network and a target network (EMA of online) — no negatives needed. The target network provides regression targets that prevent collapse.

**SimSiam**: Simplified BYOL without momentum encoder — collapse prevented by stop-gradient and predictor.

### 5.2 Masked Modeling

**Masked Autoencoders** (MAE, He et al., 2022):

1. Mask a large portion (75%) of image patches
2. Encode visible patches only
3. Decode complete image in pixel space
4. Loss: MSE on masked patches

This is the vision analog of masked language modeling (BERT).

**Spectral SSL**: Predict the Fourier coefficients or frequency components of the signal, forcing the model to learn global structure.

### 5.3 Pretext Tasks

Beyond contrastive and masked modeling, various pretext tasks have been proposed:

- **Jigsaw puzzles**: Solve rearranged image patches (predict permutation)
- **Rotation prediction**: Predict rotation angle (0°, 90°, 180°, 270°)
- **Relative position**: Predict spatial relationship between patches
- **Colorization**: Predict color channels from grayscale
- **Context prediction**: Predict surrounding context from a patch
- **Video frame order**: Predict temporal ordering of frames

---

## 6. Reinforcement Learning

Reinforcement learning (RL) studies how agents learn optimal behavior through interaction with an environment, receiving rewards or penalties.

### 6.1 Markov Decision Processes (MDP)

An MDP is defined by the tuple `(S, A, P, R, γ)`:

- `S`: Set of states
- `A`: Set of actions
- `P(s'|s, a)`: Transition probability from s to s' when taking action a
- `R(s, a, s')`: Reward function
- `γ ∈ [0, 1]`: Discount factor

**Policy**: `π(a|s)`: Probability of taking action a in state s.

**Return**: `G_t = Σ_{k=0}^∞ γ^k R_{t+k+1}`

**State-value function**: `V_π(s) = E_π[G_t | S_t = s]`

**Action-value function** (Q-function): `Q_π(s, a) = E_π[G_t | S_t = s, A_t = a]`

**Bellman equations**:

```
V_π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s, a)[R(s, a, s') + γ V_π(s')]
Q_π(s, a) = Σ_{s'} P(s'|s, a)[R(s, a, s') + γ Σ_{a'} π(a'|s') Q_π(s', a')]
```

**Optimal value functions**:

```
V*(s) = max_a Q*(s, a)
Q*(s, a) = Σ_{s'} P(s'|s, a)[R(s, a, s') + γ max_{a'} Q*(s', a')]  (Bellman optimality)
```

### 6.2 Value-Based Methods

#### 6.2.1 Q-Learning

An off-policy, model-free RL algorithm that learns the optimal Q-function directly:

```
Q(s, a) ← Q(s, a) + α[R + γ max_{a'} Q(s', a') - Q(s, a)]
```

- **Off-policy**: Learns about optimal policy while following a behavior policy (ε-greedy)
- **Model-free**: Doesn't learn transition dynamics
- **Convergence**: Guaranteed in tabular case with sufficient exploration

**ε-greedy exploration**: With probability ε, choose random action (explore); otherwise choose greedy action `argmax_a Q(s, a)` (exploit). ε typically decays over time.

#### 6.2.2 Deep Q-Networks (DQN)

DQN (Mnih et al., 2013) uses a neural network to approximate the Q-function:

- **Input**: State (e.g., game screen pixels)
- **Output**: Q-values for each action
- **Loss**: `L = E[(r + γ max_{a'} Q(s', a'; θ') - Q(s, a; θ))²]`

**Two key innovations**:

1. **Experience Replay**: Store transitions `(s, a, r, s')` in a buffer, sample random mini-batches during training. This breaks temporal correlations and improves data efficiency.

2. **Target Network**: A separate network `θ'` (periodically updated from `θ`) is used to compute target Q-values. This stabilizes training by reducing moving target problem.

**DQN improvements**:
- **Double DQN**: Use online network for action selection, target network for value estimation (reduces overestimation bias)
- **Dueling DQN**: Split Q-value into state-value V(s) and action advantage A(s, a): `Q(s, a) = V(s) + A(s, a)`
- **Prioritized Experience Replay**: Sample transitions with higher TD error more frequently
- **Noisy Nets**: Learnable noise in network parameters for exploration

### 6.3 Policy Gradient Methods

Policy gradient methods directly optimize the policy `π_θ(a|s)` via gradient ascent on expected return.

**REINFORCE** (Monte Carlo policy gradient):

```
∇J(θ) = E[ G_t · ∇_θ log π_θ(a_t|s_t) ]
```

The gradient points in the direction that increases the probability of actions that led to higher returns.

**Reduce variance by subtracting baseline**:

```
∇J(θ) = E[ (G_t - b(s_t)) · ∇_θ log π_θ(a_t|s_t) ]
```

A common baseline is the state-value function `V(s)`.

**Actor-Critic**: Combines policy gradient (actor) with value function (critic). The actor updates the policy, the critic evaluates the current policy:

```
∇J(θ) = E[ (Q_w(s, a) - V_v(s)) · ∇_θ log π_θ(a|s) ]
```

The term `A(s, a) = Q(s, a) - V(s)` is the **advantage function** — how much better is action a compared to the average.

### 6.4 Proximal Policy Optimization (PPO)

PPO (Schulman et al., 2017) is the most widely used deep RL algorithm. It addresses the problem of choosing the right step size in policy gradients.

**Clipped Surrogate Objective**:

```
L_CLIP(θ) = E_t[min(r_t(θ) Â_t, clip(r_t(θ), 1-ε, 1+ε) Â_t)]
```

where `r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)` is the probability ratio, and `Â_t` is the estimated advantage.

The clipping prevents the policy from changing too much in a single update. If the ratio moves outside [1-ε, 1+ε], the gradient is clipped, providing a stable trust region.

**PPO-Clip vs PPO-Penalty**:
- **PPO-Clip**: Clips the objective directly (most common)
- **PPO-Penalty**: Adds a KL divergence penalty to constrain policy change

**PPO Algorithm**:

```
for iteration = 1 to N:
    for actor = 1 to M:
        Run policy π_θ_old for T timesteps, collect {s, a, r}
        Compute advantages Â_t using GAE
    Optimize L_CLIP with SGD for K epochs, mini-batch size B
    Update θ_old ← θ
```

**Generalized Advantage Estimation (GAE)**:

```
Â_t = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}
```

where `δ_t = r_t + γV(s_{t+1}) - V(s_t)` is the TD error, and `λ` controls bias-variance trade-off.

### 6.5 Soft Actor-Critic (SAC)

SAC (Haarnoja et al., 2018) is an off-policy, maximum-entropy RL algorithm:

**Objective**: Maximize both expected return and policy entropy:

```
J(π) = Σ_t E[(r(s_t, a_t) + α H(π(·|s_t)))]
```

where `α` is the temperature parameter controlling exploration (higher α → more stochasticity).

**Key components**:
- **Soft policy evaluation**: `Q(s, a) = r + γ E[V(s')]` where `V(s) = E[Q(s, a)] - α log π(a|s)`
- **Soft policy improvement**: `π_new = argmin D_KL(π_new(·|s) || exp(Q_π_old(s,·)/α))`
- **Automated temperature tuning**: Adjust α to match target entropy

SAC typically uses:
- Two Q-networks (to reduce overestimation bias)
- A policy network
- A value network (optional, can be computed from Q and policy)

**SAC vs PPO**:

| Aspect | PPO | SAC |
|---|---|---|
| Type | On-policy | Off-policy |
| Sample efficiency | Lower | Higher |
| Stability | Very stable | Stable |
| Exploration | Entropy bonus | Max-entropy framework |
| Hyperparameters | Fewer | More (α, target entropy) |
| Continuous control | Good | Excellent |

### 6.6 Model-Based RL

Instead of learning policy/value from experience alone, model-based RL learns a model of the environment dynamics:

```
learn: ŝ_{t+1} = f(s_t, a_t)
       r̂_t = g(s_t, a_t)
```

The learned model can be used for planning (e.g., via Model Predictive Control) or to generate synthetic experience for model-free RL.

**MuZero** (Schrittwieser et al., 2020): Combines model-based and model-free by learning a model that predicts only reward, value, and policy (not full environment state). This enables planning without access to the true dynamics.

### 6.7 Multi-Agent RL

Multiple agents learn and interact simultaneously, leading to non-stationary environments. Key approaches:
- **Independent Q-learning**: Each agent learns independently, ignoring others
- **Centralized training, decentralized execution (CTDE)**: Critics have global information, actors only local observations (MADDPG, QMIX, VDN)
- **Multi-agent PPO (MAPPO)**: Extends PPO to multi-agent setting

---

## 7. Statistical Learning Theory

### 7.1 Bias-Variance Tradeoff

The generalization error of a model can be decomposed into three components:

```
E[(y - f̂(x))²] = Bias[f̂(x)]² + Var[f̂(x)] + σ²
```

where:
- **Bias**: Error due to model's assumptions (simplifications that miss true patterns)
- **Variance**: Error due to model's sensitivity to training data fluctuations
- **σ²**: Irreducible noise (Bayes error)

```
Bias² = (E[f̂(x)] - f*(x))²    where f* is the true function
Var = E[(f̂(x) - E[f̂(x)])²]
```

**The tradeoff**:
- **High bias, low variance**: Underfitting (e.g., linear model for non-linear data)
- **Low bias, high variance**: Overfitting (e.g., deep tree on noisy data)

```
As model complexity ↑:
  Bias ↓ (model can fit more patterns)
  Variance ↑ (model memorizes noise)
  Total error ↓ then ↑ (U-shaped curve)
```

**Visualization**:

```
Error
  ^
  |  Total Error
  |   /\
  |  /  \
  | /    \_______ Irreducible Error
  |/ Bias²
  |----------------------> Model Complexity
           Var
```

### 7.2 Learning Curves

Learning curves show model performance as a function of training set size:

- **High bias**: Training and validation errors converge to a high value (not enough model capacity)
- **High variance**: Training error is low, validation error is high (too much capacity for data size)

### 7.3 Vapnik-Chervonenkis (VC) Dimension

The VC dimension is a measure of model capacity — the maximum number of points that can be shattered (perfectly classified in all possible labelings) by the model class.

- **Linear classifier in ℝ²**: VC dimension = 3
- **Linear classifier in ℝᵈ**: VC dimension = d + 1
- **SVM with RBF kernel**: Infinite VC dimension

For binary classification, with probability `1 - η`:

```
R(f) ≤ R_emp(f) + sqrt((h(log(2N/h) + 1) - log(η/4)) / N)
```

where `R(f)` is true risk, `R_emp(f)` is empirical risk, `h` is VC dimension, `N` is sample size. This bounds generalization error in terms of model complexity.

### 7.4 Probably Approximately Correct (PAC) Learning

PAC learning bounds the number of samples needed to achieve a given error with high probability:

```
P(R(f) ≤ ε) ≥ 1 - δ
```

Sample complexity `N(ε, δ)` scales with `1/ε` and `log(1/δ)`.

**PAC-Bayes**: Extends PAC to randomized predictors, providing tighter bounds for neural networks.

### 7.5 Regularization

Regularization constrains model complexity to improve generalization. See [Section 10](#10-regularization-techniques) for a detailed treatment.

### 7.6 Cross-Validation

See [Section 11](#11-cross-validation-and-model-selection) for cross-validation methods.

---

## 8. Loss Functions

### 8.1 Regression Losses

#### 8.1.1 Mean Squared Error (MSE / L2 Loss)

```
MSE = (1/N) Σ_i (y_i - ŷ_i)²
```

- **Gradient**: `∂L/∂ŷ = (2/N)(ŷ - y)` — linear in residual
- **Properties**: Penalizes large errors heavily (quadratic), optimal for Gaussian noise
- **Sensitivity**: Very sensitive to outliers

#### 8.1.2 Mean Absolute Error (MAE / L1 Loss)

```
MAE = (1/N) Σ_i |y_i - ŷ_i|
```

- **Gradient**: `∂L/∂ŷ = (1/N) sign(ŷ - y)` — constant magnitude
- **Properties**: Robust to outliers, optimal for Laplacian noise
- **Issue**: Non-differentiable at zero (subgradient used)

**MSE vs MAE**:

| Aspect | MSE | MAE |
|---|---|---|
| Outlier sensitivity | High | Low |
| Gradient magnitude | Proportional to error | Constant |
| Optimal for | Gaussian noise | Laplacian noise |
| Convergence | Faster initially | Slower near optimum |

#### 8.1.3 Huber Loss

Combines MSE and MAE:

```
L_δ(e) = { 0.5 e²,            if |e| ≤ δ
          { δ|e| - 0.5 δ²,    if |e| > δ
```

where `e = y - ŷ`.

- **Properties**: Quadratic for small errors (smooth, differentiable), linear for large errors (robust to outliers)
- **δ**: Controls transition point (tunable hyperparameter)

#### 8.1.4 Quantile Loss

Used for quantile regression (predicting conditional quantiles):

```
L_τ(e) = { τ · e,          if e ≥ 0
          { (τ-1) · e,     if e < 0
```

where `τ ∈ (0, 1)` is the target quantile.

- **τ=0.5**: Median regression (L1 loss)
- **τ=0.9**: Predicts the 90th percentile
- **Asymmetric**: Over-predictions and under-predictions are penalized differently

#### 8.1.5 Log-Cosh Loss

```
L = Σ_i log(cosh(ŷ_i - y_i))
```

Similar to Huber but twice differentiable everywhere.

### 8.2 Classification Losses

#### 8.2.1 Cross-Entropy (Log Loss)

**Binary**:

```
L = -(1/N) Σ_i [y_i log(ŷ_i) + (1-y_i) log(1-ŷ_i)]
```

**Categorical**:

```
L = -(1/N) Σ_i Σ_k y_{ik} log(ŷ_{ik})
```

- **Properties**: Penalizes confident wrong predictions heavily, convex in logits
- **Calibration**: Well-calibrated probabilities with proper training

#### 8.2.2 Hinge Loss (SVM)

```
L = (1/N) Σ_i max(0, 1 - y_i · ŷ_i)
```

where `y_i ∈ {-1, 1}`, `ŷ_i` is the raw score (not probability).

- **Properties**: Creates margin, only penalizes points on wrong side of margin
- **Subgradient**: Zero for well-classified points outside margin

#### 8.2.3 Focal Loss

Designed for class imbalance (object detection):

```
L = -(1/N) Σ_i α (1 - ŷ_i)^γ y_i log(ŷ_i)
```

where `γ ≥ 0` reduces the relative loss for well-classified examples, focusing on hard, misclassified examples.

- **γ=0**: Standard cross-entropy
- **γ=2**: Common default for object detection (e.g., RetinaNet)
- **α**: Class balancing weight

#### 8.2.4 Contrastive Loss

Used in representation learning to pull positive pairs together and push negative pairs apart:

```
L = (1/N) Σ_i [y · D² + (1-y) · max(0, m - D)²]
```

where `D = ||z_1 - z_2||` is distance between embeddings, `y=1` for positive pairs, `y=0` for negative pairs, `m` is margin.

#### 8.2.5 Triplet Loss

Operates on triplets (anchor, positive, negative):

```
L = max(0, D(a, p)² - D(a, n)² + α)
```

where `D(a, p)` is distance between anchor and positive, `D(a, n)` is distance between anchor and negative, `α` is margin.

**Hard negative mining**: Only train on triplets where `D(a, p) - D(a, n) + α > 0` (semi-hard or hard triplets).

#### 8.2.6 InfoNCE (Noise Contrastive Estimation)

Used in contrastive learning:

```
L = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ))
```

Maximizes mutual information between positive pairs. Equivalent to (K+1)-way softmax classification where one class is positive and K are negatives.

---

## 9. Optimization Algorithms

### 9.1 Gradient Descent Variants

#### 9.1.1 Batch Gradient Descent (BGD)

```
θ_{t+1} = θ_t - η · ∇_θ J(θ_t)
```

- Uses the entire dataset per update
- Deterministic (no noise)
- Slow for large datasets

#### 9.1.2 Stochastic Gradient Descent (SGD)

```
θ_{t+1} = θ_t - η · ∇_θ J(θ_t; x^{(i)}, y^{(i)})
```

- One sample per update
- High variance (noisy gradients)
- Can escape shallow local minima

#### 9.1.3 Mini-Batch Gradient Descent

```
θ_{t+1} = θ_t - η · (1/B) Σ_{i=1}^B ∇_θ J(θ_t; x^{(i)}, y^{(i)})
```

- B: batch size (typically 16-1024)
- Reduces variance vs SGD
- Enables vectorized computation (GPU)
- Balances convergence speed and stability

### 9.2 Momentum Methods

#### 9.2.1 SGD with Momentum

Accumulates a velocity vector to accelerate convergence:

```
v_{t+1} = γ · v_t + η · ∇_θ J(θ_t)
θ_{t+1} = θ_t - v_{t+1}
```

- `γ` (momentum coefficient, typically 0.9): Controls contribution of past gradients
- **Effect**: Accelerates in consistent gradient directions, dampens oscillations
- **Analogy**: Ball rolling downhill gaining speed

#### 9.2.2 Nesterov Accelerated Gradient (NAG)

Look-ahead momentum:

```
v_{t+1} = γ · v_t + η · ∇_θ J(θ_t - γ · v_t)
θ_{t+1} = θ_t - v_{t+1}
```

- Computes gradient at "look-ahead" position
- Provides more stable convergence than standard momentum
- Theoretical convergence rate: O(1/t²) for convex functions

### 9.3 Adaptive Learning Rate Methods

#### 9.3.1 AdaGrad

Adapts learning rate per parameter based on historical gradients:

```
g_t = ∇_θ J(θ_t)
G_t = G_{t-1} + g_t ⊙ g_t  (accumulated squared gradients)
θ_{t+1} = θ_t - η / (√(G_t) + ε) ⊙ g_t
```

- **Effect**: Larger updates for infrequent features, smaller updates for frequent features
- **Issue**: Accumulated gradient sum grows monotonically, learning rate → 0

#### 9.3.2 RMSProp

Modifies AdaGrad with exponential moving average of squared gradients:

```
E[g²]_t = β · E[g²]_{t-1} + (1-β) · g_t ⊙ g_t
θ_{t+1} = θ_t - η / (√(E[g²]_t) + ε) ⊙ g_t
```

- `β` (typically 0.9): Decay rate for moving average
- **Effect**: Non-monotonic learning rate decay, works well in non-convex settings

#### 9.3.3 Adam (Adaptive Moment Estimation)

Combines momentum (first moment) with RMSProp (second moment):

```
m_t = β₁ · m_{t-1} + (1-β₁) · g_t            (biased first moment)
v_t = β₂ · v_{t-1} + (1-β₂) · g_t ⊙ g_t      (biased second moment)
m̂_t = m_t / (1 - β₁^t)                        (bias correction)
v̂_t = v_t / (1 - β₂^t)                        (bias correction)
θ_{t+1} = θ_t - η · m̂_t / (√(v̂_t) + ε)
```

**Default hyperparameters** (from Kingma & Ba, 2014):
- `η = 0.001` (learning rate)
- `β₁ = 0.9` (momentum decay)
- `β₂ = 0.999` (RMS decay)
- `ε = 1e-8` (numerical stability)

**Properties**:
- Per-parameter learning rates
- Scale-invariant (resilient to gradient scaling)
- Well-suited for sparse gradients and noisy data
- Often requires less tuning than SGD

**Limitations**:
- Can fail to generalize as well as SGD with proper tuning
- Some variants found to not converge in certain cases

#### 9.3.4 AdamW (Adam with Decoupled Weight Decay)

Fixes a bug in the original Adam where weight decay interacts poorly with adaptive learning rates:

```
θ_{t+1} = θ_t - η · (m̂_t / (√(v̂_t) + ε) + λ · θ_t)
```

Adam's L2 regularization: `θ = θ - η·m̂/√v̂ - η·λ·θ`
AdamW weight decay: `θ = θ - η·m̂/√v̂ - λ_decay·θ`

- **Key insight**: Weight decay should be separated from gradient-based updates
- **Effect**: Better generalization, especially in large models and transfer learning
- **Default λ**: 0.01-0.1 typical for vision, 0.1-1.0 for language models

#### 9.3.5 Other Optimizers

| Optimizer | Key Idea | When to Use |
|---|---|---|
| AdaBelief | Uses variance of gradients instead of squared gradients | When Adam is unstable |
| RAdam | Rectified Adam, warmup-based variance correction | Training from scratch |
| Lion | Symbolic discovery, sign operations only | Very large models |
| Sophia | Uses Hessian diagonal for preconditioning | Large-scale training |
| Shampoo | Full-matrix preconditioning (Kronecker approx) | Very deep networks |
| LAMB | Layer-wise Adaptive Moments, large batch training | Large batch (≥16K) |

### 9.4 Learning Rate Scheduling

#### 9.4.1 Step Decay

```
η_t = η_0 · γ^{⌊t / step_size⌋}
```

Drop learning rate by factor γ every step_size iterations.

#### 9.4.2 Exponential Decay

```
η_t = η_0 · γ^t
```

Smooth, continuous decay.

#### 9.4.3 Cosine Annealing

```
η_t = η_min + (η_max - η_min) · (1 + cos(π · t / T)) / 2
```

- Smooth decay from η_max to η_min
- Often combined with warm restarts (SGDR): periodically reset to η_max
- **Cosine with warmup**: Standard in modern LLM training

#### 9.4.4 Polynomial Decay

```
η_t = η_0 · (1 - t/T)^p
```

- `p=1`: Linear decay
- `p=2`: Quadratic decay
- `p=0.5`: Square root decay

#### 9.4.5 Warmup

Linearly increase learning rate from 0 to target over `W` steps:

```
η_t = η_target · min(1, t/W)
```

**Why warmup?** In early training, model parameters are random; large gradients combined with high learning rate can cause divergence. Warmup allows the optimizer to accumulate stable statistics (momentum, Adam state) before full updates.

### 9.5 Gradient Clipping

Prevent exploding gradients by scaling down large gradients:

```
If ||g|| > threshold:
    g = g · threshold / ||g||
```

- **L2 norm clipping**: Scale whole gradient vector
- **Value clipping**: Clip each element individually
- Typical threshold: 0.5-10 for RNNs, 1.0 for Transformers

---

## 10. Regularization Techniques

### 10.1 L1 (Lasso) Regularization

```
L_reg = L_data + λ ||w||_1 = L_data + λ Σ_j |w_j|
```

- Drives weights exactly to zero (sparsity)
- Feature selection property
- Non-differentiable at zero (subgradient methods)

### 10.2 L2 (Ridge) Regularization

```
L_reg = L_data + λ ||w||_2² = L_data + λ Σ_j w_j²
```

- Shrinks weights toward zero (no exact zeros)
- Equivalent to Gaussian prior on weights
- Differentiable everywhere

### 10.3 Elastic Net

Combines L1 and L2:

```
L_reg = L_data + λ₁||w||_1 + λ₂||w||_2²
```

- Groups correlated features (unlike Lasso which picks one)
- Maintains sparsity

### 10.4 Early Stopping

Stop training when validation loss stops improving:

```
1. Track validation loss at each epoch
2. If no improvement for P epochs (patience), stop
3. Restore model from best validation epoch
```

Acts as implicit regularization by limiting the effective model complexity (similar to L2 regularization).

### 10.5 Dropout

Randomly "drop" (zero out) a fraction `p` of neurons during training:

```
For each training step (forward pass):
    r_j ~ Bernoulli(1 - p)  (binary mask)
    h_j' = r_j · h_j / (1 - p)    (inverted dropout, scaling at training time)
```

- **During inference**: All neurons active, no scaling needed (in inverted dropout)
- **Effect**: Forces network to learn redundant representations (prevents co-adaptation)
- **Interpretation**: Approximate Bayesian inference over network weights
- **Dropout rate p**: 0.5 for hidden layers, 0.2-0.3 for input layers

### 10.6 DropConnect

Generalization of dropout that drops individual weights (connections) rather than whole neurons:

```
For each weight w_ij:
    r_ij ~ Bernoulli(1 - p)
    w_ij' = r_ij · w_ij
```

### 10.7 Label Smoothing

Replaces hard targets (0/1) with soft targets:

```
y'_i = y_i · (1 - α) + α / K
```

where `K` is number of classes, `α` is smoothing parameter (typically 0.1).

**Effect**: Prevents overconfidence, improves calibration, acts as regularization.

### 10.8 Data Augmentation

Generate modified versions of training data:

- **Images**: Rotation, crop, flip, color jitter, cutout, mixup, CutMix
- **Text**: Back-translation, synonym replacement, token masking, mixup
- **Audio**: SpecAugment (time/freq masking), noise addition, pitch shift

### 10.9 Stochastic Depth

Used in deep networks (ResNets): randomly drop entire layers during training:

```
For each layer l in forward pass:
    if random() < p_l:  # drop layer
        output = input (skip via residual connection)
    else:
        output = layer(input)
```

Training is an ensemble of networks of varying depths.

---

## 11. Cross-Validation and Model Selection

### 11.1 Hold-Out Validation

Split data into training (70-80%) and validation (20-30%) sets.

**Pros**: Simple, fast
**Cons**: High variance estimate, wastes data

### 11.2 k-Fold Cross-Validation

```
1. Split data into k equal folds (k=5 or 10 typical)
2. For each fold i:
   a. Train on all folds except i
   b. Evaluate on fold i
3. Report average performance across all k trials
```

**Stratified k-fold**: Preserves class proportions in each fold (important for imbalanced data).

**Repeated k-fold**: Repeat k-fold multiple times with different random splits.

### 11.3 Leave-One-Out Cross-Validation (LOOCV)

k = N (one sample per validation set).

**Extreme case**: Low bias, high variance, very expensive (O(N) training runs). Used for very small datasets.

### 11.4 Bootstrapping

Sample N points with replacement B times:

```
for b = 1 to B:
    sample N points with replacement (bootstrap sample)
    train model on bootstrap sample
    evaluate on out-of-bag (OOB) points
```

**.632 Bootstrap**: Weighted average of training error and OOB error to correct optimistic bias.

### 11.5 Hyperparameter Tuning

#### Grid Search

Exhaustive search over a predefined parameter grid:

```
params = {'learning_rate': [0.001, 0.01, 0.1],
          'max_depth': [3, 5, 7],
          'n_estimators': [100, 200, 300]}

for combination in all_combinations:
    score = cross_val_score(model, params=combination)
    best = combination with highest score
```

Cost: O(P^D) where P = parameters per dimension, D = dimensions.

#### Random Search

Sample hyperparameters randomly from distributions:

```
for i = 1 to N:
    learning_rate ~ LogUniform(0.0001, 1.0)
    max_depth ~ Uniform(3, 10)
    n_estimators ~ UniformInt(50, 500)
    ...
```

More efficient than grid search when some parameters are more important (Bergstra & Bengio, 2012).

#### Bayesian Optimization

Build a probabilistic model (Gaussian Process or Tree-structured Parzen Estimator) mapping hyperparameters to performance, then select next hyperparameters to evaluate via acquisition function (Expected Improvement, Upper Confidence Bound).

**Key libraries**: Hyperopt, Optuna, SMAC, GPyOpt

#### Evolutionary Optimization

Use genetic algorithms or CMA-ES to evolve hyperparameter populations.

---

## 12. Ensemble Methods

Ensemble methods combine multiple models to improve prediction accuracy and robustness.

### 12.1 Voting Ensembles

- **Hard voting**: Majority vote across models (classification)
- **Soft voting**: Average predicted probabilities (classification)
- **Averaging**: Mean prediction (regression)

### 12.2 Bagging (Bootstrap Aggregating)

```
for b = 1 to B:
    sample N points with replacement
    train model M_b on bootstrap sample
final = average(M_1(x), ..., M_B(x))
```

Reduces variance without increasing bias. Most effective for high-variance models (decision trees).

### 12.3 Boosting

Sequential training where each model focuses on mistakes of the previous ensemble. See [Section 2.2.7-2.2.10](#227-gradient-boosting-machines-gbm).

### 12.4 Stacking (Stacked Generalization)

Train a meta-learner to combine base model predictions:

```
Level 0: Train diverse base models on original data
         Get predictions P_1(x), P_2(x), ..., P_M(x)
Level 1: Train meta-model on predictions from level 0
         Use predictions as features for meta-model
```

Best practice: Level 0 predictions should come from cross-validation (not training data) to prevent overfitting.

### 12.5 Bayesian Model Averaging

Weight models by their posterior probability:

```
P(y|x) = Σ_m P(y|x, M_m) · P(M_m|data)
```

### 12.6 Weighted Ensembles

Assign learnable weights to ensemble members:

```
final(x) = Σ_m w_m · f_m(x)
```

Weights can be optimized on a validation set.

---

## 13. Practical Considerations

### 13.1 Data Preprocessing

**Feature scaling** (required for gradient-based methods, SVM, k-NN, PCA):

- **Standardization**: `z = (x - μ) / σ` (zero mean, unit variance)
- **Min-Max scaling**: `z = (x - x_min) / (x_max - x_min)` (range [0, 1])
- **Robust scaling**: `z = (x - median) / IQR` (robust to outliers)

**Handling missing values**:
- Drop rows/columns with missing data
- Imputation: mean, median, mode, KNN, model-based
- Indicator column for missingness

**Categorical encoding**:
- One-hot encoding, label encoding, target encoding
- Ordinal encoding (for ordered categories)
- Count/frequency encoding, hash encoding

### 13.2 Imbalanced Datasets

**Resampling**:
- Random undersampling (majority class)
- Random oversampling (minority class)
- SMOTE (Synthetic Minority Over-sampling TEchnique): Create synthetic samples by interpolating between minority class samples and their k-nearest neighbors
- ADASYN: Adaptive synthetic sampling focusing on harder examples

**Cost-sensitive learning**:
- Class weights in loss function
- Higher penalty for misclassifying minority class

**Algorithmic approaches**:
- Decision trees: Adjust class weight, use balanced splits
- SVM: Class-weighted C parameter
- Ensemble: EasyEnsemble, BalanceCascade

**Evaluation metrics**:
- Accuracy is misleading on imbalanced data
- Use: Precision, Recall, F1-score, AUC-ROC, Precision-Recall curve, Matthews Correlation Coefficient

### 13.3 Feature Engineering

**Feature construction**: Create new features from domain knowledge (ratios, aggregates, time-based features).

**Feature selection**:
- **Filter methods**: Correlation, mutual information, chi-square, ANOVA
- **Wrapper methods**: Forward/backward selection, recursive feature elimination (RFE)
- **Embedded methods**: Lasso, tree importance, regularization

**Dimensionality reduction**: PCA, t-SNE, UMAP, autoencoders (see [Section 3.2](#32-dimensionality-reduction)).

### 13.4 Model Interpretation

| Method | Scope | Description |
|---|---|---|
| Feature importance | Global | Impurity-based or permutation-based |
| Partial dependence plots | Global | Show marginal effect of feature on prediction |
| SHAP (SHapley Additive exPlanations) | Local + Global | Game-theoretic feature attribution |
| LIME (Local Interpretable Model-agnostic Explanations) | Local | Locally approximate model with interpretable surrogate |
| Permutation importance | Global | Drop in performance when feature is shuffled |
| ICE plots (Individual Conditional Expectation) | Local | PDP for individual instances |

### 13.5 Production Considerations

- **Latency vs accuracy tradeoff**: Simpler models (linear, trees) vs complex models (ensembles, neural networks)
- **Model pruning**: Reduce model size for deployment
- **Quantization**: Convert to lower precision (FP16, INT8)
- **A/B testing**: Compare model versions on live traffic
- **Monitoring**: Track input distribution drift (data drift) and prediction distribution shift (model drift)
- **Retraining schedule**: Periodic retraining vs triggered retraining

### 13.6 Ethics and Fairness

- **Bias sources**: Training data, labels, feature selection, model architecture
- **Fairness metrics**: Demographic parity, equal opportunity, equalized odds
- **Mitigations**: Reweighting, adversarial debiasing, fairness constraints
- **Transparency**: Model cards, datasheets for datasets

---

## 14. Further Reading

### Foundational Papers

- "The Elements of Statistical Learning" (Hastie, Tibshirani, Friedman, 2009)
- "Pattern Recognition and Machine Learning" (Bishop, 2006)
- "Statistical Learning Theory" (Vapnik, 1998)

### Algorithm-Specific

- **Random Forests**: Breiman (2001) - "Random Forests"
- **XGBoost**: Chen & Guestrin (2016) - "XGBoost: A Scalable Tree Boosting System"
- **LightGBM**: Ke et al. (2017) - "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"
- **CatBoost**: Prokhorenkova et al. (2018) - "CatBoost: unbiased boosting with categorical features"
- **SVM**: Cortes & Vapnik (1995) - "Support-Vector Networks"
- **t-SNE**: van der Maaten & Hinton (2008) - "Visualizing Data using t-SNE"
- **UMAP**: McInnes et al. (2018) - "UMAP: Uniform Manifold Approximation and Projection"

### Reinforcement Learning

- "Reinforcement Learning: An Introduction" (Sutton & Barto, 2018)
- **DQN**: Mnih et al. (2015) - "Human-level control through deep reinforcement learning"
- **PPO**: Schulman et al. (2017) - "Proximal Policy Optimization Algorithms"
- **SAC**: Haarnoja et al. (2018) - "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL"
- **MuZero**: Schrittwieser et al. (2020) - "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model"

### Self-Supervised Learning

- **SimCLR**: Chen et al. (2020) - "A Simple Framework for Contrastive Learning of Visual Representations"
- **MoCo**: He et al. (2020) - "Momentum Contrast for Unsupervised Visual Representation Learning"
- **BYOL**: Grill et al. (2020) - "Bootstrap Your Own Latent"
- **MAE**: He et al. (2022) - "Masked Autoencoders Are Scalable Vision Learners"

### Optimization

- **Adam**: Kingma & Ba (2014) - "Adam: A Method for Stochastic Optimization"
- **AdamW**: Loshchilov & Hutter (2019) - "Decoupled Weight Decay Regularization"
- **Cosine Annealing**: Loshchilov & Hutter (2017) - "SGDR: Stochastic Gradient Descent with Warm Restarts"

---

*This document is part of the AI Knowledge Base — Foundations category. See also: [01-LLM-and-AI-Models.md](01-LLM-and-AI-Models.md), [03-Deep-Learning.md](03-Deep-Learning.md), [04-Neural-Networks.md](04-Neural-Networks.md), [05-Training-Methodologies.md](05-Training-Methodologies.md).*
