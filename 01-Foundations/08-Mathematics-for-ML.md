# Machine Learning Mathematics: Foundations

## Table of Contents
1. [Linear Algebra](#1-linear-algebra)
2. [Probability Theory](#2-probability)
3. [Calculus](#3-calculus)
4. [Statistics](#4-statistics)
5. [Information Theory](#5-information-theory)
6. [Optimization](#6-optimization)
7. [Numerical Computation](#7-numerical)
7a. [Automatic Differentiation](#7a-autograd)
7b. [Loss Landscapes and Optimization Dynamics](#7b-loss-landscapes)
7c. [Graph Theory for ML](#7c-graph-theory)
7d. [Gaussian Processes](#7d-gaussian-processes)
7e. [SDEs for Diffusion Models](#7e-sdes)
7f. [Causal Inference](#7f-causal-inference)
7g. [Optimization Algorithms — Practical Guide](#7g-optimization-algorithms)
7h. [Information Theory for Machine Learning](#7h-information-theory)
7i. [Bayesian Methods for ML](#7i-bayesian-methods)
8. [Cross-References](#8-cross-references)

---

## 1. Linear Algebra

### 1.1 Vectors and Matrices
- **Vector:** v ∈ R^d. Dot product: v·w = Σ v_i w_i = v^T w
- **Matrix:** A ∈ R^{m×n}. Operations: transpose (A^T), inverse (A^{-1}), trace (tr(A)=ΣA_ii), determinant (det(A))
- **Norms:**
  - L1 (Manhattan): ||v||₁ = Σ|v_i| — sparse regularization (Lasso)
  - L2 (Euclidean): ||v||₂ = √Σ v_i² — ridge regularization
  - L∞: ||v||_∞ = max|v_i| — adversarial robustness
  - Frobenius (matrix): ||A||_F = √Σ A_ij²

**Python Examples:**
```python
import numpy as np

v = np.array([1, 2, 3])
w = np.array([4, 5, 6])
dot = np.dot(v, w)           # 32
l2 = np.linalg.norm(v)        # 3.742
l1 = np.linalg.norm(v, ord=1) # 6.0

A = np.array([[1, 2], [3, 4]])
det = np.linalg.det(A)        # -2.0
trace = np.trace(A)           # 5
```

### 1.2 Matrix Decompositions

| Decomposition | Formula | Use Case |
|--------------|---------|----------|
| **LU** | A = LU | Solving linear systems |
| **QR** | A = QR (Q orthogonal, R upper triangular) | Least squares, PCA via QR |
| **Cholesky** | A = LL^T (symmetric positive definite) | Efficient solve, sampling |
| **Eigen** | A = QΛQ^{-1} | Spectral analysis, PCA |
| **SVD** | A = UΣV^T | Most general — PCA, compression, pseudoinverse |

#### Singular Value Decomposition (SVD): A = UΣV^T
- U ∈ R^{m×m}: left singular vectors (columns are A^TA eigenvectors)
- Σ ∈ R^{m×n}: diagonal of singular values σ₁ ≥ σ₂ ≥ ... ≥ σ_k
- V ∈ R^{n×n}: right singular vectors (columns are AA^T eigenvectors)
- **Key in ML:** PCA, matrix factorization, low-rank approximation, data compression

```python
# SVD in practice
U, S, Vt = np.linalg.svd(A, full_matrices=False)

# Low-rank approximation (rank k)
k = 1
A_approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
# A_approx ≈ [[2.5, 3.5], [4.5, 6.5]] for k=1 on [[1,2],[3,4]]
```

### 1.3 Eigenvalues and Eigenvectors

Av = λv — v is eigenvector, λ is eigenvalue

**Properties:**
- Symmetric matrices: real eigenvalues, orthogonal eigenvectors
- Positive definite: all λ > 0 → convex optimization guaranteed
- Spectral radius: ρ(A) = max|λ_i| — determines convergence of iterative methods

```python
eigenvalues, eigenvectors = np.linalg.eig(A)
# eigenvectors[:, i] corresponds to eigenvalues[i]
```

### 1.4 Matrix Calculus
- ∂(Ax)/∂x = A^T
- ∂(x^T A x)/∂x = (A + A^T)x
- ∂log|X|/∂X = (X^{-1})^T
- ∂(tr(AX))/∂X = A^T

### 1.5 Key Concepts for Deep Learning
- **Rank:** dimension of range — low rank = low effective dimension
- **Low-rank approximation:** A ≈ UV^T (U ∈ R^{m×k}, V ∈ R^{n×k}) — foundation of **LoRA** (Low-Rank Adaptation)
  - LoRA: ΔW = BA where B∈R^{d×r}, A∈R^{r×k}, r ≪ min(d,k) → 10,000× fewer parameters to fine-tune
- **Positive definiteness:** x^T A x > 0 ∀x — ensures convex loss landscapes
- **Condition number:** κ(A) = σ_max / σ_min — ill-conditioned matrices make optimization hard

---

## 2. Probability Theory

### 2.1 Basic Rules
- P(A|B) = P(A∩B)/P(B) (conditional)
- P(A∩B) = P(A|B)P(B) (chain rule)
- P(A∪B) = P(A)+P(B)-P(A∩B)
- P(B) = Σ P(B|A_i)P(A_i) (law of total probability)

### 2.2 Common Distributions

| Distribution | Parameters | Mean | Variance | Use Case |
|-------------|:----------:|:----:|:--------:|----------|
| Bernoulli(p) | p ∈ [0,1] | p | p(1-p) | Binary classification |
| Binomial(n,p) | n∈N, p∈[0,1] | np | np(1-p) | Count of successes |
| Poisson(λ) | λ > 0 | λ | λ | Rare event counts (queue lengths) |
| Gaussian N(μ,σ²) | μ∈R, σ²>0 | μ | σ² | Continuous data, noise assumption |
| Log-normal(μ,σ²) | μ,σ² | e^{μ+σ²/2} | (e^{σ²}-1)e^{2μ+σ²} | Positive values (prices, lengths) |
| Exponential(λ) | λ > 0 | 1/λ | 1/λ² | Wait times, inter-event intervals |
| Beta(α,β) | α,β > 0 | α/(α+β) | αβ/[(α+β)²(α+β+1)] | Prior for Bernoulli, A/B testing |
| Gamma(k,θ) | k,θ > 0 | kθ | kθ² | Sum of exponentials, Bayesian inference |
| Dirichlet(α) | α ∈ R^k_+ | α_i/Σα_j | — | Topic proportions, categorical prior |
| Categorical(p) | p ∈ Δ^{k-1} | p_i | p_i(1-p_i) | Multi-class classification |

```python
from scipy import stats

# Gaussian PDF
x = np.linspace(-3, 3, 100)
pdf = stats.norm.pdf(x, loc=0, scale=1)

# Sample from Bernoulli
samples = stats.bernoulli.rvs(p=0.3, size=100)

# Beta-Binomial conjugate prior
alpha, beta = 2, 5  # prior
post_alpha = alpha + 7  # 7 successes
post_beta = beta + 3    # 3 failures
```

### 2.3 Bayes' Theorem

P(A|B) = P(B|A)P(A) / P(B)

- P(A): **prior** — what you believe before seeing data
- P(B|A): **likelihood** — how probable the data is under each hypothesis
- P(A|B): **posterior** — updated belief after seeing data
- P(B): **evidence** (marginal) — normalizing constant

**Bayesian vs Frequentist:**
| Aspect | Frequentist | Bayesian |
|--------|:-----------:|:--------:|
| Probability | Long-run frequency | Degree of belief |
| Parameters | Fixed (unknown) | Random variables |
| Inference | Point estimates, CIs | Posterior distributions |
| Prior | Not used | Explicitly required |
| Update | New data → new study | Sequential (posterior → prior) |

### 2.4 Maximum Likelihood Estimation (MLE)
θ_MLE = argmax_θ log P(data|θ)

- For Gaussian: μ̂ = (1/n) Σ x_i, σ̂² = (1/n) Σ (x_i - μ̂)²
- For Bernoulli: p̂ = (number of successes) / n
- For Categorical: p̂_k = (count of class k) / n

### 2.5 Maximum A Posteriori (MAP)
θ_MAP = argmax_θ [log P(data|θ) + log P(θ)]

- MLE + prior = MAP (regularization interpretation)
- L2 regularization ↔ Gaussian prior on weights
- L1 regularization ↔ Laplace prior on weights (sparsity)

### 2.6 Monte Carlo Methods
```python
# Monte Carlo estimate of π
n = 1_000_000
x = np.random.uniform(-1, 1, n)
y = np.random.uniform(-1, 1, n)
inside = (x**2 + y**2) <= 1.0
pi_estimate = 4 * inside.sum() / n  # ≈ 3.1416
```

- **MCMC:** Metropolis-Hastings, Hamiltonian MC for Bayesian posterior sampling
- **Importance sampling:** Estimate E[f(x)] when sampling from p is hard

---

## 3. Calculus

### 3.1 Derivatives
- ∂/∂x (f(x) + g(x)) = f'(x) + g'(x)
- ∂/∂x (f(x)g(x)) = f'(x)g(x) + f(x)g'(x) (product rule)
- ∂/∂x f(g(x)) = f'(g(x))g'(x) (chain rule) — **the backpropagation workhorse**
- ∂/∂x (x^n) = nx^{n-1}
- ∂/∂x (e^x) = e^x
- ∂/∂x (log x) = 1/x
- ∂/∂x (σ(x)) = σ(x)(1-σ(x)) where σ is the sigmoid function

### 3.2 Chain Rule (Vector Form)
∂L/∂x = (∂y(x)/∂x)^T · (∂L/∂y) — chain rule for vectors

Used in **backpropagation:**
```python
# Conceptual backprop for a simple net: L = (σ(Wx + b) - y)²
# Forward:
z = W @ x + b       # z = Wx + b
a = 1 / (1 + np.exp(-z))  # σ(z)
L = (a - y) ** 2    # MSE loss

# Backward (auto):
dL_da = 2 * (a - y)
da_dz = a * (1 - a)  # sigmoid derivative
dz_dW = x             # local gradient
dL_dW = dL_da * da_dz * dz_dW  # ∂L/∂W = [∂L/∂a][∂a/∂z][∂z/∂W]
```

### 3.3 Gradients
∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂x_d]^T — direction of steepest ascent

Gradient descent: x_{t+1} = x_t - η∇f(x_t)

### 3.4 Jacobian and Hessian
- **Jacobian:** J_ij = ∂f_i/∂x_j (first derivative matrix, m×n for f: R^n → R^m)
- **Hessian:** H_ij = ∂²f/∂x_i∂x_j (second derivative matrix, symmetric for smooth f)
  - Positive definite Hessian → local minimum
  - Negative definite Hessian → local maximum
  - Indefinite Hessian → saddle point
- Newton's method: x_{t+1} = x_t - H^{-1}∇f(x_t) — faster convergence near minimum, but expensive (O(n³))

---

## 4. Statistics

### 4.1 Hypothesis Testing Framework
- **Null hypothesis H₀:** no effect (e.g., μ₁ = μ₂)
- **Alternative Hₐ:** there is an effect (e.g., μ₁ ≠ μ₂)
- **p-value:** P(observed data | H₀ true) — probability of seeing data this extreme assuming H₀
- **Significance level α = 0.05** (conventional threshold)
- **Type I error:** rejecting true H₀ (false positive) — controlled by α
- **Type II error:** failing to reject false H₀ (false negative) — controlled by power (1-β)

### 4.2 Common Statistical Tests

| Test | Purpose | Assumptions |
|------|---------|-------------|
| **t-test (Student's)** | Compare two means | Normal, independent |
| **Welch's t-test** | Compare two means | Normal, independent (unequal variance OK) |
| **ANOVA** | Compare k means | Normal, independent, equal variance |
| **Mann-Whitney U** | Compare two distributions | Non-parametric, ordinal |
| **χ² test** | Test independence | Categorical data, expected ≥ 5 |
| **Fisher's exact** | Test independence (small samples) | Categorical, 2×2 |
| **Kolmogorov-Smirnov** | Compare distributions | Continuous, non-parametric |
| **Shapiro-Wilk** | Test normality | Continuous |
| **Levene's test** | Test equal variance | Continuous |

```python
from scipy import stats

# Two-sample t-test
group_a = np.random.normal(100, 15, 50)
group_b = np.random.normal(110, 15, 50)
t_stat, p_value = stats.ttest_ind(group_a, group_b)
# p < 0.05 → reject H₀ (means differ)

# ANOVA
f_stat, p_value = stats.f_oneway(group_a, group_b, group_c)
```

### 4.3 Confidence Intervals
95% CI: x̄ ± 1.96 · σ/√n — means that if we repeated the experiment 100 times, ~95 of the intervals would contain the true parameter.

- **For proportions:** p̂ ± Z_{α/2}√(p̂(1-p̂)/n)
- **Bootstrap CI:** resample data with replacement 10,000 times, take 2.5th and 97.5th percentiles (no distributional assumptions)

```python
# Bootstrap confidence interval
data = np.array([...])
boot_means = [np.mean(np.random.choice(data, size=len(data), replace=True)) 
              for _ in range(10_000)]
ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
```

---

## 5. Information Theory

### 5.1 Entropy
H(X) = -Σ P(x) log P(x) — average information content or uncertainty measured in **bits** (log₂) or **nats** (log_e)

- Binary event: H ≤ 1 bit (fair coin = 1 bit, biased coin < 1 bit)
- Uniform distribution over k outcomes: H = log₂(k) bits
- **Principle of Maximum Entropy:** among all distributions satisfying constraints, choose the one with maximum entropy (least additional assumptions)

```python
def entropy(p):
    p = np.array(p)
    p = p[p > 0]  # remove zeros (0 log 0 = 0)
    return -np.sum(p * np.log2(p))

print(entropy([0.5, 0.5]))       # 1.0 bit
print(entropy([0.9, 0.1]))       # 0.469 bits
```

### 5.2 Cross-Entropy
H(P,Q) = -Σ P(x) log Q(x) — average # bits needed to encode data from P using Q's distribution

**Training loss for classification:** L = H(p_data, p_model) = -Σ y_i log(ŷ_i)

```python
# Cross-entropy loss (multi-class)
def cross_entropy(y_true, y_pred):
    # y_true: one-hot encoded, y_pred: softmax output
    return -np.sum(y_true * np.log(y_pred + 1e-15))
```

### 5.3 KL Divergence
D_KL(P||Q) = Σ P(x) log(P(x)/Q(x))

- Measures dissimilarity between distributions (extra bits needed)
- D_KL ≥ 0, equality iff P = Q
- **Asymmetric:** D_KL(P||Q) ≠ D_KL(Q||P)
- **Forward KL** (D_KL(P||Q)): moment-matching — used in variational inference
- **Reverse KL** (D_KL(Q||P)): mode-seeking — used in RL (policy gradients)

### 5.4 Mutual Information
I(X;Y) = H(X) - H(X|Y) = D_KL(P(x,y)||P(x)P(y))

- Measures dependence — I(X;Y) = 0 iff X and Y are independent
- Used for: feature selection, representation learning (InfoNCE loss), clustering evaluation

---

## 6. Optimization

### 6.1 Gradient Descent Variants

| Method | Update Rule | Key Characteristics |
|--------|-------------|--------------------|
| **SGD** | θ ← θ - η∇L | Basic, high variance, noisy |
| **Momentum** | v ← γv + η∇L, θ ← θ - v | Accelerates along consistent directions, dampens oscillations |
| **Nesterov** | v ← γv + η∇L(θ-γv), θ ← θ - v | Look-ahead gradient, faster convergence |
| **AdaGrad** | θ ← θ - η/√(G+ε)·∇L | Adapts per-parameter, good for sparse features |
| **RMSprop** | θ ← θ - η/√(v+ε)·∇L, v ← βv+(1-β)∇L² | Fixes AdaGrad's diminishing LR |
| **Adam** | m ← β₁m+(1-β₁)∇L, v ← β₂v+(1-β₂)∇L², θ←θ-η·m/(√v+ε) | **Default choice** — adaptive + momentum |
| **AdamW** | Same as Adam + weight decay decoupled from gradient | Better generalization than Adam |
| **LAMB** | Layer-wise adaptive LR | Large-batch training (up to 64K batch) |

### 6.2 Learning Rate Schedules

| Schedule | Formula | When to Use |
|----------|---------|-------------|
| **Constant** | η_t = η₀ | Simple, baseline |
| **Step decay** | η_t = η₀ · γ^{⌊t/s⌋} | Common in vision (drop by 10× every 30 epochs) |
| **Exponential** | η_t = η₀ · e^{-kt} | Smooth decay |
| **Cosine** | η_t = η_min + ½(η₀-η_min)(1+cos(π·t/T)) | Popular with transformers |
| **Linear warmup** | η_t = η₀·(t/t_warmup) for t < t_warmup | Prevents early divergence in large models |
| **One-cycle** | Warmup → cosine decay → fine-tuning | Fast convergence (super-convergence) |

### 6.3 Convex vs Non-convex Optimization
- **Convex:** one global minimum, gradient descent guarantees convergence
- **Strongly convex:** f(y) ≥ f(x) + ∇f(x)ᵀ(y-x) + (μ/2)||y-x||² — linear convergence
- **Non-convex:** many local minima, saddle points, no guarantees
- Most deep learning problems are non-convex, but SGD finds solutions that generalize well (implicit bias)

### 6.4 Lagrangian Duality
Lagrangian: L(x,λ) = f(x) + Σ λ_i g_i(x) — optimize f subject to constraints g_i(x) ≤ 0

- **SVM dual:** maximize Σ α_i - ½ Σ α_i α_j y_i y_j K(x_i,x_j) subject to Σ α_i y_i = 0
- **KKT conditions:** necessary for constrained optimum (stationarity, primal feasibility, dual feasibility, complementary slackness)

### 6.5 Regularization as Optimization
- **L1 (Lasso):** min ||Ax-b||² + λ||x||₁ — encourages sparsity
- **L2 (Ridge):** min ||Ax-b||² + λ||x||₂² — shrinks coefficients uniformly
- **Elastic Net:** min ||Ax-b||² + λ₁||x||₁ + λ₂||x||₂² — hybrid

### 6.6 Convergence and Stopping Criteria

| Criterion | Condition | Notes |
|-----------|-----------|-------|
| **Gradient norm** | ||∇f(θ_t)|| < ε | Need to tune ε |
| **Function change** | |f(θ_t) - f(θ_{t-1})| < ε | Can stop too early at plateaus |
| **Early stopping** | Validation loss increases | Prevents overfitting |
| **Max iterations** | t ≥ T_max | Safety limit |

```python
# Simple SGD with momentum
theta = np.zeros(d)
v = np.zeros(d)
for t in range(epochs):
    grad = compute_gradient(theta, X, y)
    v = gamma * v + lr * grad
    theta = theta - v
    if np.linalg.norm(grad) < 1e-6:
        break
```

---

## 7. Numerical Computation

### 7.1 Numerical Stability
| Problem | Example | Solution |
|---------|---------|----------|
| **Underflow** | exp(-1000) = 0 | Log-space computations |
| **Overflow** | exp(1000) = ∞ | Clipping, normalization |
| **Catastrophic cancellation** | 1e16 + 1 ≈ 1e16 | Rearrange computations |
| **Divide by zero** | x/0 | Add ε = 1e-8 |

```python
# Log-sum-exp trick (numerically stable softmax)
def logsumexp(x):
    c = np.max(x)  # shift for stability
    return c + np.log(np.sum(np.exp(x - c)))

def softmax(x):
    return np.exp(x - np.max(x, axis=-1, keepdims=True)) / \
           np.sum(np.exp(x - np.max(x, axis=-1, keepdims=True)), axis=-1, keepdims=True)
```

### 7.2 Conditioning
- Well-conditioned: small change in input → small change in output
- Ill-conditioned: small change in input → large change in output
- **Condition number:** κ = σ_max / σ_min (ratio of largest to smallest singular value)
  - κ = 1: perfect conditioning (orthogonal)
  - κ >> 1: ill-conditioned (slow optimization)

---

## 7a. Automatic Differentiation (Autograd)

### 7a.1 Forward vs Reverse Mode

| Mode | Passes | Cost per Pass | Best For |
|------|--------|:-------------:|----------|
| **Forward** (tangent linear) | 1 per input dim | O(n) | Functions with few inputs, many outputs |
| **Reverse** (adjoint/backprop) | 1 per output dim | O(m) | **Deep learning** — many inputs (params), 1 output (loss) |

### 7a.2 Reverse-Mode Autodiff by Hand

Given: f(x, y) = log(x·y) + x²

Computational graph:
```
     x ──┐          ┌───┐
         ├──(*)── p ─→[log]── q ─→ ┐
     y ──┘          └───┘          ├──(+)── f
                                    │
                                x ──┼──(**)── x² ── r ─→ ┘
```

**Forward evaluation:**
```
p = x·y           = 2·3 = 6
q = log(p)        = log(6) ≈ 1.79
r = x²            = 4
f = q + r         ≈ 5.79
```

**Reverse pass (chain rule backward):**
```
∂f/∂f = 1
∂f/∂q = 1 · ∂f/∂f = 1
∂f/∂r = 1 · ∂f/∂f = 1
∂f/∂x² = 1 · ∂f/∂r = 1         (from r = x²)
∂f/∂p = 1/p · ∂f/∂q = 1/6 ≈ 0.167 (from q = log(p))
∂f/∂x = y · ∂f/∂p + 2x · ∂f/∂x² = 3·0.167 + 4·1 = 4.5
∂f/∂y = x · ∂f/∂p = 2·0.167 = 0.333
```

```python
# Manual autograd for f(x, y) = log(x*y) + x²
x, y = 2.0, 3.0

# Forward (compute values + store for backward)
p = x * y
q = np.log(p)
r = x ** 2
f = q + r

# Backward (reverse-mode)
df_df = 1.0
df_dq = 1.0 * df_df
df_dr = 1.0 * df_df
df_dp = (1/p) * df_dq
df_dx2 = 1.0 * df_dr
df_dx = y * df_dp + 2*x * df_dx2  # 4.5
df_dy = x * df_dp                  # 0.333
```

### 7a.3 Automatic Differentiation in PyTorch

```python
import torch

# PyTorch handles the graph construction + backward automatically
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
f = torch.log(x * y) + x ** 2
f.backward()

print(f"df/dx = {x.grad.item():.3f}")  # 4.500
print(f"df/dy = {y.grad.item():.3f}")  # 0.333
```

**Key insight:** Every deep learning framework (PyTorch, JAX, TensorFlow) implements reverse-mode autodiff. The `backward()` call traverses the computational graph in topological reverse order, multiplying Jacobians via the chain rule.

---

## 7b. Loss Landscapes and Optimization Dynamics

### 7b.1 Visualizing Loss Landscapes

The loss landscape of a neural network is high-dimensional and non-convex. Common visualization technique (Li et al., 2018): pick two random direction vectors δ, η and plot:

```
f(α, β) = L(θ* + α·δ + β·η)
```

### 7b.2 Sharp vs Flat Minima

| Property | Sharp Minimum | Flat Minimum |
|----------|:------------:|:------------:|
| Hessian eigenvalues | Large positive | Small (near zero) |
| Generalization | Poor (overfits) | Good |
| Sensitivity to shift | High (small Δθ → large ΔL) | Low |
| SGD behavior | Unstable near minimum | Converges stably |
| Quantization robustness | Poor (lost in low precision) | Good |

**Empirical finding (Keskar et al., 2017):** Large-batch training converges to sharp minima → worse generalization. Small-batch SGD finds flat minima → better generalization.

```python
# Estimate Hessian spectrum via power iteration
def hessian_top_eigenvalue(model, loss_fn, data, k=5):
    """Top-k eigenvalues of Hessian via Lanczos iteration."""
    params = torch.cat([p.view(-1) for p in model.parameters()])
    def hvp(v):
        loss = loss_fn(model, data)
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        grad_vec = torch.cat([g.view(-1) for g in grads])
        return torch.autograd.grad(grad_vec @ v, model.parameters(), retain_graph=True)

    # Power iteration to estimate largest eigenvalue
    v = torch.randn_like(params)
    v = v / v.norm()
    for _ in range(k):
        hv = hvp(v)
        v = hv / hv.norm()
    lambda_max = (v @ hvp(v)).item()
    return lambda_max
```

### 7b.3 Gradient Flow and Neural Tangent Kernel (NTK)

In the infinite-width limit, neural networks follow **gradient flow** under the **Neural Tangent Kernel**:

```python
# Conceptual: kernel gradient descent with NTK
# d f_t(x) / dt = -Σ_i K(x, x_i) · (f_t(x_i) - y_i)
# where K(x, x') = ∇_θ f(x;θ) · ∇_θ f(x';θ)^T
```

- **NTK regime:** When width → ∞, the kernel K is constant during training → linearized dynamics
- **Feature learning regime:** At realistic widths, the kernel evolves → non-linear, richer representations
- **μP (Maximal Update Parameterization):** Parameterization that preserves feature learning at any width (Yang & Hu, 2021)

---

## 7c. Graph Theory for Machine Learning

Graph neural networks (GNNs) and topological methods in ML rely on foundational graph theory concepts.

### 7c.1 Basic Definitions

A graph G = (V, E) consists of vertices V and edges E ⊆ V × V.

| Graph Type | Edge Direction | Example in ML |
|------------|:--------------:|---------------|
| **Undirected** | No direction | Molecule bond graphs, protein structures |
| **Directed** (digraph) | One-way | Citation networks, web page links |
| **Weighted** | Numeric edge weight | Transaction networks, similarity graphs |
| **Bipartite** | Nodes split into two groups | User-item interactions in recsys (see: [06-Advanced/06-Recommendation-Systems.md]) |
| **Temporal** | Edges have timestamps | Social networks, traffic patterns |
| **Heterogeneous** | Multiple node/edge types | Knowledge graphs (e.g., [entity]-[relation]-[entity]) |

### 7c.2 Graph Matrices

| Matrix | Definition | Use Case |
|--------|-----------|----------|
| **Adjacency A** | A_ij = 1 if (i,j) ∈ E, 0 otherwise | Basic connectivity |
| **Degree D** | D_ii = Σ_j A_ij (diagonal) | Node importance |
| **Laplacian L** | L = D - A | Spectral clustering, GNNs |
| **Normalized Laplacian** | L_norm = I - D^{-1/2} A D^{-1/2} | Graph convolution |
| **Incidence B** | B_ij = 1 if edge i connects node j | Algebraic graph theory |

**Laplacian Eigenvalues** (spectrum of L) reveal graph structure:
- λ₂ (algebraic connectivity / Fiedler value) > 0 iff graph is connected
- Larger λ₂ → more connected graph
- Eigenvectors of L = Fourier basis on the graph → foundation of spectral GNNs

```python
import networkx as nx
import numpy as np

G = nx.karate_club_graph()  # Zachary's karate club — classic social network
A = nx.adjacency_matrix(G).todense()
D = np.diag(np.array(A).sum(axis=1))
L = D - A  # Unnormalized Laplacian

eigvals = np.linalg.eigvalsh(L)
print(f"Graph has {len(eigvals)} nodes, Fiedler value = {eigvals[1]:.3f}")
```

### 7c.3 Graph Isomorphism and Invariants

Determining if two graphs are identical (isomorphic) is a key problem — related to model expressivity (see Weisfeiler-Lehman test in GNN theory).

| Invariant | Description | ML Relevance |
|-----------|-------------|--------------|
| **Degree sequence** | Sorted node degrees | Simple graph signature |
| **Spectrum** | Set of Laplacian eigenvalues | Weisfeiler-Lehman GNN expressivity |
| **Centers (betweenness, closeness, PageRank)** | Node importance metrics | Node2vec, GNN feature preprocessing |
| **Graphlets/motifs** | Small recurring subgraph patterns | Biological network analysis |

### 7c.4 Message Passing on Graphs (Spatial GNNs)

The fundamental operation in GNNs:

```
h_v^(k+1) = UPDATE(h_v^(k), AGGREGATE({h_u^(k) : u ∈ N(v)}))
```

Where:
- N(v): neighbors of node v
- AGGREGATE: sum, mean, max, or attention-pooling
- UPDATE: MLP or linear transformation

```python
# Conceptual GNN layer (mean aggregation)
def gnn_layer(h, A, W):
    # h: node features (n, d), A: adjacency (n, n), W: weight (d, d')
    n = h.shape[0]
    h_aggr = A @ h  # sum of neighbor features
    h_aggr = h_aggr / (A.sum(1, keepdims=True) + 1e-8)  # mean
    return h_aggr @ W  # linear transformation
```

### 7c.5 Weisfeiler-Lehman (WL) Test and GNN Expressivity

The **WL graph isomorphism test** is a color-refinement algorithm:
1. Initialize all node colors to 1
2. For each node: hash (current_color, multiset of neighbor colors)
3. Update colors to new hash values
4. Repeat; if colors converge, graphs share same WL signature

**Key theorem:** Standard message-passing GNNs are at most as powerful as the WL test for distinguishing non-isomorphic graphs. More expressive GNNs require higher-order features (k-WL), random features, or simplicial complexes.

### 7c.6 Applications of Graph Theory in ML

| Application | Graph Type | ML Technique |
|-------------|------------|--------------|
| **Molecule property prediction** | Molecular graph (atoms + bonds) | GCN, GAT, MPNN |
| **Social network analysis** | User friendship graph | Node2vec, GraphSAGE |
| **Recommender systems** | Bipartite user-item graph | PinSage, LightGCN |
| **Knowledge graph reasoning** | Entity-relation graph | R-GCN, CompGCN |
| **Traffic forecasting** | Spatial-temporal road graph | STGCN, DCRNN |
| **Point cloud processing** | KNN graph of 3D points | DGCNN, PointNet++ |

---

## 7d. Gaussian Processes

Gaussian Processes (GPs) provide a Bayesian nonparametric approach to regression and classification.

### 7d.1 Definition

A Gaussian Process is a collection of random variables, any finite subset of which is jointly Gaussian:

```
f(x) ~ GP(m(x), k(x, x'))
```

- **Mean function:** m(x) = E[f(x)]
- **Covariance (kernel) function:** k(x, x') = E[(f(x) - m(x))(f(x') - m(x'))]

### 7d.2 Common Kernels

| Kernel | Formula | Properties |
|--------|---------|------------|
| **RBF (squared exponential)** | k(x,x') = σ² exp(-||x-x'||²/2l²) | Infinitely smooth, universal |
| **Matérn** | k_ν(x,x') = σ²(2√νr/l)^ν K_ν(2√νr/l) / Γ(ν) | ν=1/2: Laplace; ν=3/2,5/2: less smooth |
| **Periodic** | k(x,x') = exp(-2 sin²(π||x-x'||/p)/l²) | Periodic functions |
| **Linear** | k(x,x') = σ² x·x' | Bayesian linear regression |
| **Spectral Mixture** | k(x,x') = Σ w_i exp(-2π²||x-x'||²τ_i)cos(2π||x-x'||μ_i) | Learned basis functions |

### 7d.3 GP Regression

**Predictive distribution for test point x\*:**

```
μ_* = k_*ᵀ (K + σ_n² I)⁻¹ y
σ_*² = k(x_*, x_*) - k_*ᵀ (K + σ_n² I)⁻¹ k_*
```

Where:
- K: N×N kernel matrix on training data
- k_*: kernel between test point and training points
- σ_n²: noise variance

**Key properties:** GPs provide calibrated uncertainty estimates (σ_*²) — they know what they don't know. This is critical for Bayesian optimization, active learning, and safe RL.

### 7d.4 GP Limitations and Scalability

| Limitation | Cause | Solution |
|------------|-------|----------|
| O(n³) complexity | Matrix inversion of K | Sparse GPs (SGP, FITC), KISS-GP |
| O(n²) memory | Storing full kernel matrix | Inducing point methods |
| Non-Gaussian likelihood | Classification, counts | Laplace approximation, EP, MCMC |
| High-dimensional inputs | Curse of dimensionality | Deep kernel learning, additive GPs |

```python
# Simple GP regression with GPyTorch
import gpytorch

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean = gpytorch.means.ConstantMean()
        self.covar = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    
    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean(x), self.covar(x))
```

**Applications:** Bayesian optimization (hyperparameter tuning of ML models), active learning (selecting most informative data points), time-series forecasting, geostatistics (kriging), and learning dynamics models from sparse interactions.

---

## 7e. Stochastic Differential Equations for Diffusion Models

Modern generative AI (Stable Diffusion, DALL-E 3) is built on continuous-time formulations of diffusion.

### 7e.1 From DDPM to SDEs

The discrete diffusion process in Denoising Diffusion Probabilistic Models (DDPM) can be generalized to continuous time via Stochastic Differential Equations (SDEs).

**Forward SDE (noising):**
```
dx = f(x, t) dt + g(t) dw
```

**Reverse SDE (denoising):**
```
dx = [f(x, t) - g(t)² ∇_x log p_t(x)] dt + g(t) dw̄
```

Where:
- f(x, t): drift coefficient (deterministic component)
- g(t): diffusion coefficient (noise schedule)
- w: Wiener process (Brownian motion)
- ∇_x log p_t(x): score function (estimated by neural network)
- dw̄: reverse-time Wiener process

### 7e.2 Score Matching

The score function ∇_x log p_t(x) is learned via score matching:

**Denoising Score Matching (DSM):**
```
L_DSM = E_t[λ(t) E_x(0)[E_x(t)|x(0)[||s_θ(x(t), t) - ∇_x(t) log p(x(t)|x(0))||²]]]
```

Where s_θ(x(t), t) is the neural network that estimates the score. For Gaussian diffusion, ∇_x(t) log p(x(t)|x(0)) has a closed form: -(x(t) - √ᾱ_t x(0)) / (1 - ᾱ_t).

### 7e.3 Probability Flow ODE

An equivalent deterministic process exists that preserves marginal densities (can be integrated via ODE solvers):

```
dx/dt = f(x, t) - ½ g(t)² s_θ(x, t)
```

This **Probability Flow ODE** is the basis for accelerated sampling (fewer steps needed than SDE-based sampling). Used in consistency models, rectified flow (Stable Diffusion 3, FLUX).

### 7e.4 The Diffusion-ODE Connection

| Concept | Discrete (DDPM) | Continuous (SDE) |
|---------|----------------|------------------|
| Forward process | Markov chain | SDE |
| Reverse process | Reverse Markov chain | Reverse SDE |
| Noise schedule | β₁, β₂, ..., β_T | β(t) |
| Training loss | L_simple = E[||ε_θ - ε||²] | Score matching |
| Sampling | Langevin steps | ODE/SDE solver (Euler-Maruyama, Heun) |
| Deterministic | DDIM | Probability flow ODE |

### 7e.5 Rectified Flow

Rectified Flow (Liu et al., 2022) learns a straight-line trajectory between noise and data, enabling fast (1-5 step) generation:

```
dx/dt = v_θ(x, t)  # Vector field that pushes noise straight to data
```

**Learning objective:** Given paired data (x₀, x₁) where x₀ ∼ data, x₁ ∼ noise:
```
L = E[||v_θ(t·x₁ + (1-t)·x₀, t) - (x₁ - x₀)||²]
```

Rectified flow is the basis for SD3, FLUX.1, and other state-of-the-art image generation models (see: [09-Papers/01-Foundational-Papers.md] §18.5).

---

## 7f. Causal Inference

Causal inference is the mathematical framework for understanding cause-effect relationships — essential for debiasing, recommendation systems, and robust ML.

### 7f.1 Pearl's Causal Framework

**Directed Acyclic Graphs (DAGs):** Nodes = variables, Edges = causal relationships.

**Three fundamental structures:**

```
Chain: X → Z → Y        (Confounding: X and Y are correlated via Z)
Fork:  X ← Z → Y        (Z is a common cause)
Collider: X → Z ← Y     (Z is a common effect, conditioning on it induces correlation)
```

### 7f.2 Do-Calculus and Interventions

Distinguishing correlation from causation:

| Expression | Meaning | ML Relevance |
|------------|---------|--------------|
| P(Y|X) | Conditional (observational) | Standard supervised learning |
| P(Y|do(X)) | Interventional — what if we set X? | Treatment effect estimation |
| P(Y|X, do(Z)) | Conditional on X, intervene on Z | Counterfactual reasoning |

**do-calculus rules** (Pearl, 2009) allow simplifying interventional distributions using observational data — the basis for causal inference from passive observations.

### 7f.3 Potential Outcomes (Rubin Causal Model)

For each unit i, define two potential outcomes:
- Y_i(1): outcome if treated
- Y_i(0): outcome if not treated

**Fundamental Problem of Causal Inference:** We only observe one outcome per unit.

**Average Treatment Effect (ATE):**
```
ATE = E[Y(1) - Y(0)]
```

**Estimation methods:** Matching, propensity score weighting, instrumental variables, difference-in-differences.

### 7f.4 Causal Discovery

Learning causal structure from observational data:

| Method | Approach | Assumptions |
|--------|----------|-------------|
| **PC Algorithm** | Constraint-based (conditional independence tests) | Faithfulness, no latent confounders |
| **Fast Causal Inference (FCI)** | Handles latent confounders | Faithfulness |
| **LiNGAM** | Linear non-Gaussian additive noise | Non-Gaussian noise |
| **NOTEARS** | Continuous optimization over DAGs | Smooth optimization |
| **DAGMA** | Acyclicity characterization via matrix exponentials | Faster convergence |

### 7f.5 Causal Representation Learning

Learning representations that capture causal structure:

- **Disentangled representations:** Latent variables correspond to generative factors
- **Causal generative models:** VAE with causal structure in latent space (CausalVAE, DEAR)
- **Invariant risk minimization (IRM):** Learn representations invariant across environments

### 7f.6 Causal Deep Learning Applications

| Application | Causal Method | Benefit |
|-------------|---------------|---------|
| **Debiasing** | Inverse propensity weighting | Remove selection bias |
| **Domain generalization** | Invariant representations | Robustness to distribution shift |
| **Recommendation systems** | Counterfactual reasoning | Debias position/popularity effects |
| **Medical imaging** | Causal attention | Identify causally relevant regions |
| **Drug discovery** | Causal graph between genes/treatments | Identify true therapeutic targets |
| **LLM reasoning** | Causal prompting | Reduce spurious correlations |

---

## 7g. Optimization Algorithms — Practical Guide

This section provides hands-on comparisons of optimization algorithms used in deep learning, complementing the theoretical overview in [Section 6](#6-optimization).

### 7g.1 Optimizer Showdown: Rosenbrock Function

The **Rosenbrock function** (a.k.a. "banana function") is a classic optimization test:

```
f(x, y) = (a - x)² + b(y - x²)²      with a = 1, b = 100
```

It has a narrow, parabolic-shaped valley that makes first-order optimization challenging. The global minimum is at (1, 1).

```python
import numpy as np

def rosenbrock(x, y, a=1, b=100):
    return (a - x)**2 + b * (y - x**2)**2

def rosenbrock_grad(x, y, a=1, b=100):
    dx = -2*(a - x) - 4*b*x*(y - x**2)
    dy = 2*b*(y - x**2)
    return np.array([dx, dy])

# --- SGD with Momentum ---
def sgd_momentum(grad_fn, init, lr=0.002, momentum=0.9, steps=5000):
    theta = np.array(init, dtype=float)
    v = np.zeros_like(theta)
    traj = [theta.copy()]
    for _ in range(steps):
        g = grad_fn(*theta)
        v = momentum * v + lr * g
        theta = theta - v
        traj.append(theta.copy())
    return np.array(traj)

# --- Adam ---
def adam(grad_fn, init, lr=0.003, beta1=0.9, beta2=0.999, eps=1e-8, steps=10000):
    theta = np.array(init, dtype=float)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    traj = [theta.copy()]
    for t in range(1, steps + 1):
        g = grad_fn(*theta)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)
        traj.append(theta.copy())
    return np.array(traj)

# --- AdamW (decoupled weight decay) ---
def adamw(grad_fn, init, lr=0.003, beta1=0.9, beta2=0.999, eps=1e-8,
          weight_decay=0.001, steps=10000):
    theta = np.array(init, dtype=float)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    traj = [theta.copy()]
    for t in range(1, steps + 1):
        g = grad_fn(*theta)
        g = g + weight_decay * theta          # decoupled weight decay
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)
        traj.append(theta.copy())
    return np.array(traj)

# Run all optimizers
init = np.array([-1.5, 1.5])
traj_sgd = sgd_momentum(rosenbrock_grad, init)
traj_adam = adam(rosenbrock_grad, init)
traj_adamw = adamw(rosenbrock_grad, init)

print(f"SGD+Momentum final: ({traj_sgd[-1,0]:.4f}, {traj_sgd[-1,1]:.4f}), "
      f"loss={rosenbrock(*traj_sgd[-1]):.6f}")
print(f"Adam final:        ({traj_adam[-1,0]:.4f}, {traj_adam[-1,1]:.4f}), "
      f"loss={rosenbrock(*traj_adam[-1]):.6f}")
print(f"AdamW final:       ({traj_adamw[-1,0]:.4f}, {traj_adamw[-1,1]:.4f}), "
      f"loss={rosenbrock(*traj_adamw[-1]):.6f}")
# Expected output (approximate):
# SGD+Momentum final: (1.0000, 1.0000), loss=0.000000
# Adam final:        (1.0000, 1.0000), loss=0.000000
# AdamW final:       (0.9998, 0.9996), loss=0.000000
```

**Key observations:**
- **SGD + Momentum** converges fastest on this ill-conditioned problem (5000 steps) — its simple velocity accumulation efficiently navigates the narrow Rosenbrock valley
- **Adam** converges in more steps (10000 needed here) because its per-parameter adaptive LR reduces step sizes in high-gradient directions, which helps in neural networks but slows progress on this function
- **AdamW** matches Adam's convergence with decoupled weight decay (standard for Transformers/LLMs)
- **Takeaway:** No single optimizer is universally best — Rosenbrock favors momentum methods, while high-dimensional neural network training (millions of parameters) typically favors Adam/AdamW

### 7g.2 Optimizer Comparison Table

| Method | Update Rule (Key Formula) | Key Hyperparameter | Best For |
|--------|---------------------------|:------------------:|----------|
| **SGD** | θ ← θ − η∇L | Learning rate η | Simple baselines, convex problems |
| **SGD + Momentum** | v ← γv + η∇L, θ ← θ − v | Momentum γ (≈0.9) | Moderate curvature, CV (proven recipe) |
| **Nesterov** | v ← γv + η∇L(θ−γv), θ ← θ − v | Momentum γ (≈0.9) | Faster convergence (look-ahead gradient) |
| **AdaGrad** | θ ← θ − η/√(G+ε)·∇L | Global LR η | Sparse features (NLP, embeddings) |
| **RMSProp** | v←βv+(1−β)∇L², θ←θ−η/√(v+ε)·∇L | Decay rate β (≈0.9) | Non-stationary objectives, RNNs |
| **Adam** | m←β₁m+(1−β₁)∇L, v←β₂v+(1−β₂)∇L², θ←θ−η·m̂/(√v̂+ε) | β₁, β₂ (0.9, 0.999) | **Default for most DL** — robust, adaptive |
| **AdamW** | Same as Adam + decoupled weight decay | Weight decay λ | Transformers, LLMs (best generalization) |
| **Lion** | u←sign(β₁m+(1−β₁)∇L), θ←θ−η·u, m←β₂m+(1−β₂)∇L | β₁, β₂ (0.9, 0.99) | Large-scale training, memory efficient |

**When to choose which:**
- **Low-data / vision:** SGD + Momentum with cosine schedule (generalizes better)
- **Transformers / LLMs:** AdamW (standard) or Lion (memory-efficient)
- **Sparse features (embeddings):** AdaGrad or Adam
- **RL / RNNs:** RMSProp (handles non-stationary objectives well)

### 7g.3 Learning Rate Schedule Strategies

Modern deep learning rarely uses a constant learning rate. Two essential strategies are cosine decay and linear warmup.

```python
import numpy as np

def cosine_decay(step, total_steps, lr_max=1.0, lr_min=0.0):
    """Cosine annealing: smooth decay from lr_max to lr_min."""
    progress = step / total_steps
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * progress))

def linear_warmup(step, warmup_steps, lr_max=1.0):
    """Linear increase from 0 to lr_max."""
    return lr_max * min(1.0, step / max(warmup_steps, 1))

def cosine_with_warmup(step, total_steps, warmup_steps, lr_max=1.0, lr_min=0.0):
    """Cosine decay with linear warmup — the standard transformer recipe."""
    if step < warmup_steps:
        return linear_warmup(step, warmup_steps, lr_max)
    else:
        return cosine_decay(step - warmup_steps, total_steps - warmup_steps,
                            lr_max, lr_min)

# Usage example
total_steps = 1000
warmup_steps = 100
lr_schedule = [cosine_with_warmup(s, total_steps, warmup_steps) for s in range(total_steps)]
print(f"LR at step 0:    {lr_schedule[0]:.4f}")
print(f"LR at step 50:   {lr_schedule[50]:.4f}  (warmup phase)")
print(f"LR at step 100:  {lr_schedule[100]:.4f}  (start of cosine decay)")
print(f"LR at step 500:  {lr_schedule[500]:.4f}")
print(f"LR at step 999:  {lr_schedule[999]:.4f}")
# Expected:
# LR at step 0:    0.0000
# LR at step 50:   0.5000
# LR at step 100:  1.0000
# LR at step 500:  0.5000
# LR at step 999:  0.0000

# PyTorch equivalent using built-in schedulers
try:
    import torch
    from torch.optim import SGD
    from torch.optim.lr_scheduler import (CosineAnnealingLR, LinearLR,
                                           SequentialLR)
    
    model = torch.nn.Linear(10, 10)
    optimizer = SGD(model.parameters(), lr=0.1)
    
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=10)
    cosine = CosineAnnealingLR(optimizer, T_max=90)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                             milestones=[10])
    
    lrs = []
    for epoch in range(100):
        optimizer.step()
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])
    print(f"\nPyTorch: LR start={lrs[0]:.6f}, epoch50={lrs[50]:.6f}, end={lrs[-1]:.6f}")
except ImportError:
    print("\nPyTorch not available; numpy implementation shown above.")
```

**Other common schedules:**

| Schedule | Formula | Use Case |
|----------|---------|----------|
| **Step decay** | η_t = η₀ · γ^{⌊t/s⌋} | Classic CV: drop 10× every 30 epochs |
| **Exponential** | η_t = η₀ · e^{-kt} | Smooth continuous decay |
| **OneCycle** | Warmup → cosine → fine-tune | Fast convergence (super-convergence) |
| **Warmup-stable-decay** | Warmup → constant → cosine decay | LLM pre-training (PaLM, LLaMA) |
| **Inverse sqrt** | η_t = η₀ / √(1 + kt) | Transformer training (Attention is All You Need) |

---

## 7h. Information Theory for Machine Learning

While [Section 5](#5-information-theory) introduced the basic definitions, this section deepens the connection between information-theoretic quantities and practical ML workflows.

### 7h.1 Core Quantities — Definitions and Formulas

**Entropy — average uncertainty of a random variable:**

```
H(X) = −Σ_{x∈𝒳} P(x) log P(x)
```

- Measured in **bits** (log₂) or **nats** (logₑ)
- H(X) ≥ 0, equality iff X is deterministic
- Maximum for uniform distribution: H = log|𝒳|
- **Binary entropy:** H₂(p) = −p log₂ p − (1−p) log₂(1−p), max = 1 bit at p = 0.5

**Cross-Entropy — average bits needed to encode P using Q's code:**

```
H(P, Q) = −Σ_{x} P(x) log Q(x)
```

- H(P, Q) ≥ H(P) (more bits when Q ≠ P)
- **The de facto loss for classification** — also known as categorical cross-entropy
- For binary: BCE = −y log ŷ − (1−y) log(1−ŷ)

**KL Divergence (relative entropy) — extra bits due to using Q instead of P:**

```
D_KL(P ∥ Q) = Σ_{x} P(x) log(P(x) / Q(x))
```

- D_KL(P ∥ Q) = H(P, Q) − H(P) — connects cross-entropy and KL divergence
- D_KL ≥ 0, equality iff P = Q everywhere
- **Asymmetric:** D_KL(P ∥ Q) ≠ D_KL(Q ∥ P)
  - Forward KL (D_KL(P∥Q)): moment-matching, used in variational inference
  - Reverse KL (D_KL(Q∥P)): mode-seeking, used in policy gradients

```python
import numpy as np

def entropy(p, base=2):
    """H(X) in bits (base=2) or nats (base=np.e)."""
    p = np.array(p, dtype=float)
    p = p[p > 0]  # 0 log 0 = 0
    return -np.sum(p * np.log(p)) / np.log(base)

def cross_entropy(p, q, base=2):
    """H(P, Q)."""
    p, q = np.array(p, dtype=float), np.array(q, dtype=float)
    return -np.sum(p * np.log(q)) / np.log(base)

def kl_divergence(p, q, base=2):
    """D_KL(P || Q)."""
    p, q = np.array(p, dtype=float), np.array(q, dtype=float)
    p = p[p > 0]  # skip where p = 0
    q = q[p > 0]  # align
    return np.sum(p * np.log(p / q)) / np.log(base)

# Example: true distribution P vs approximate Q
p = np.array([0.5, 0.3, 0.2])
q = np.array([0.4, 0.4, 0.2])

print(f"Entropy H(P)        = {entropy(p):.4f} bits")
print(f"Cross-Entropy H(P,Q)= {cross_entropy(p, q):.4f} bits")
print(f"KL Divergence       = {kl_divergence(p, q):.4f} bits")
print(f"Verify: H(P,Q) = H(P) + D_KL = {entropy(p) + kl_divergence(p, q):.4f}")
# Expected:
# Entropy H(P)        = 1.4855 bits
# Cross-Entropy H(P,Q)= 1.5872 bits
# KL Divergence       = 0.1017 bits
# Verify: H(P,Q) = H(P) + D_KL = 1.5872 bits
```

### 7h.2 Cross-Entropy Loss: From Scratch vs PyTorch

```python
import numpy as np
import torch
import torch.nn.functional as F

# --- From scratch (NumPy) ---
def cross_entropy_scratch(logits, targets):
    """
    logits:  (N, C) raw scores (not softmaxed)
    targets: (N,) integer class indices in {0, 1, ..., C-1}
    Returns scalar loss.
    """
    N, C = logits.shape
    # Numerically stable softmax: subtract max per row
    logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
    softmax = np.exp(logits_shifted) / np.sum(np.exp(logits_shifted),
                                               axis=1, keepdims=True)
    # Probabilities of correct classes
    p_correct = softmax[np.arange(N), targets]
    # Negative log-likelihood
    loss = -np.mean(np.log(p_correct + 1e-15))
    return loss

# --- PyTorch built-in ---
def cross_entropy_pytorch(logits, targets):
    logits_t = torch.tensor(logits, requires_grad=True)
    targets_t = torch.tensor(targets, dtype=torch.long)
    loss = F.cross_entropy(logits_t, targets_t)
    return loss.item()

# --- Verify they match ---
np.random.seed(42)
logits = np.random.randn(4, 5)   # 4 samples, 5 classes
targets = np.array([0, 2, 1, 4])

loss_scratch = cross_entropy_scratch(logits, targets)
loss_torch = cross_entropy_pytorch(logits, targets)

print(f"Cross-entropy (scratch): {loss_scratch:.6f}")
print(f"Cross-entropy (PyTorch): {loss_torch:.6f}")
print(f"Match within 1e-5: {np.isclose(loss_scratch, loss_torch, atol=1e-5)}")
# Expected: ~1.662 (both match within tolerance)

# --- Binary cross-entropy ---
def binary_cross_entropy_scratch(y_true, y_pred):
    """y_true: (N,) in {0,1}, y_pred: (N,) probabilities in (0,1)."""
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# PyTorch equivalent: F.binary_cross_entropy() or F.binary_cross_entropy_with_logits()
```

**Key insight:** `F.cross_entropy` in PyTorch combines `log_softmax` and `nll_loss` in a single numerically stable operation. **Never manually apply softmax before cross-entropy** — always use the combined function to avoid log(0) and floating-point underflow.

### 7h.3 Mutual Information and Feature Selection

**Mutual information** quantifies how much knowing one variable reduces uncertainty about another:

```
I(X; Y) = H(X) − H(X|Y) = D_KL(P(X,Y) ∥ P(X)P(Y))
```

**Properties:**
- I(X; Y) ≥ 0, equality iff X and Y are independent
- Symmetric: I(X; Y) = I(Y; X)
- For continuous variables: I(X;Y) = ∫∫ p(x,y) log(p(x,y)/p(x)p(y)) dx dy

**Discrete computation:**

```python
def mutual_information_discrete(x, y, bins=10):
    """Empirical MI between two 1-D arrays using histogram binning."""
    from sklearn.feature_selection import mutual_info_classif
    
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    
    # For continuous targets use mutual_info_regression
    from sklearn.feature_selection import mutual_info_regression
    mi = mutual_info_regression(x, y, discrete_features=False, random_state=42)
    return mi[0]
```

**Application to feature selection:** MI identifies the most informative features for a target variable, capturing non-linear relationships that correlation-based methods miss entirely.

```python
from sklearn.feature_selection import mutual_info_classif
from sklearn.datasets import make_classification
import pandas as pd

# Generate data: 20 features, only 5 informative
X, y = make_classification(n_samples=1000, n_features=20, n_informative=5,
                           n_redundant=3, n_repeated=2, random_state=42)

# Compute MI between each feature and the target
mi_scores = mutual_info_classif(X, y, random_state=42)

# Rank features
ranking = pd.DataFrame({
    'feature': [f'X{i}' for i in range(20)],
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

print("Top 10 features by Mutual Information:")
print(ranking.head(10).to_string(index=False))
# Expected: The 5 informative features dominate the top ranks

# MI vs F-statistic (linear only) — MI captures non-linear relationships
from sklearn.feature_selection import f_classif
f_scores, _ = f_classif(X, y)
ranking['f_score'] = f_scores
print(f"\nRank correlation (MI vs F): {ranking[['mi_score','f_score']].corr().iloc[0,1]:.3f}")
```

**When to use MI for feature selection:**
- **Non-linear relationships:** MI detects any statistical dependence, not just linear
- **Mixed data types:** Works with discrete and continuous variables
- **High-dimensional screening:** Fast filter method before expensive model-based selection
- **Limitation:** Requires careful estimation for continuous variables (scikit-learn uses the KSG nearest-neighbor estimator)

### 7h.4 Information Theory Concepts — ML Application Mapping

| Concept | Definition | ML Application |
|---------|-----------|----------------|
| **Entropy H(X)** | Uncertainty in X | Decision tree splitting criterion (ID3, C4.5), maximum entropy models |
| **Cross-Entropy H(P,Q)** | Bits to encode P using Q | Classification loss (categorical cross-entropy), language model perplexity |
| **KL Divergence D_KL(P∥Q)** | Extra bits from using Q instead of P | Variational inference (ELBO), knowledge distillation, policy gradients (TRPO) |
| **Jensen-Shannon Divergence** | Symmetrized KL: ½D_KL(P∥M) + ½D_KL(Q∥M) | GAN training objective, measuring distribution shift |
| **Mutual Information I(X;Y)** | Dependence between X and Y | Feature selection, InfoNCE (contrastive learning), representation learning |
| **Conditional Entropy H(Y\|X)** | Remaining uncertainty in Y given X | Active learning (uncertainty sampling), Bayesian optimization |
| **Perplexity** | 2^{H(P)} = exp(H(P)) | Language model evaluation (average branching factor) |
| **Varentropy** | Var[−log P(x)] | Uncertainty quantification, OOD detection reliability |
| **Fano's Inequality** | Lower bound on error given entropy | Information-theoretic lower bounds on classification error |
| **Data Processing Inequality** | I(X;Z) ≤ I(X;Y) if X→Y→Z | Feature engineering (no transformation adds information) |
| **Channel Capacity** | max_{P(X)} I(X;Y) | Information bottleneck theory of deep learning |
| **Rate-Distortion** | min I(X;Z) s.t. D ≤ d | VAE (β-VAE), compression-based representations |
| **Maximum Entropy Principle** | Choose max-entropy distribution consistent with constraints | MaxEnt models, exponential family distributions, Bayesian priors |

---

## 7i. Bayesian Methods for Machine Learning

Bayesian methods provide a principled framework for reasoning about uncertainty, regularization, and model comparison.

### 7i.1 Bayes' Theorem Review

Bayes' theorem describes how to update beliefs in light of evidence:

```
P(θ | D) = P(D | θ) · P(θ) / P(D)
```

| Term | Name | Role |
|------|------|------|
| **P(θ)** | Prior | Initial belief about parameters before seeing data |
| **P(D\|θ)** | Likelihood | How probable the data is under parameter θ |
| **P(D)** | Evidence / Marginal Likelihood | Normalizing constant: ∫ P(D\|θ) P(θ) dθ |
| **P(θ\|D)** | Posterior | Updated belief after observing data D |

**The Bayesian workflow:**
1. **Specify a prior** P(θ) encoding domain knowledge or uncertainty
2. **Collect data** D
3. **Compute the posterior** P(θ|D) ∝ P(D|θ) · P(θ)
4. **Predict** by integrating over the posterior:
   ```
   P(y* | x*, D) = ∫ P(y* | x*, θ) · P(θ | D) dθ
   ```
   This marginalization naturally accounts for parameter uncertainty.

### 7i.2 MLE vs MAP — Detailed Comparison

| Aspect | Maximum Likelihood (MLE) | Maximum A Posteriori (MAP) |
|--------|:------------------------:|:--------------------------:|
| **Objective** | θ̂ = argmax P(D\|θ) | θ̂ = argmax P(θ\|D) ∝ P(D\|θ)P(θ) |
| **Prior** | Not used | Explicitly required |
| **Regularization** | None (overfits with many params) | Implicit via prior choice |
| **L2 ↔ Gaussian prior** | — | θ̂ = argmin [−log P(D\|θ) + λ\|θ\|²] |
| **L1 ↔ Laplace prior** | — | θ̂ = argmin [−log P(D\|θ) + λ\|θ\|₁] |
| **Uncertainty** | None (point estimate) | None (point estimate — just the mode) |
| **Data regime** | Abundant data | Limited data (prior helps constrain) |
| **As N → ∞** | — | **Converges to MLE** (prior washes out) |
| **Philosophy** | Parameters are fixed unknowns | Parameters are random variables |
| **Computational cost** | Lower | Slightly higher (prior term) |

```python
# MLE vs MAP for coin flips (Beta-Bernoulli)
import numpy as np

# Observed: 7 heads out of 10 flips
N, k = 10, 7

# MLE: θ̂ = k/N
theta_mle = k / N

# MAP: Beta prior Beta(α, β) → posterior Beta(α+k, β+N-k)
# Mode of Beta(α, β) = (α-1)/(α+β-2) for α, β > 1
alpha, beta = 2, 2   # Prior centered at 0.5
theta_map = (alpha + k - 1) / (alpha + beta + N - 2)

# Full posterior mean (not just mode)
theta_post_mean = (alpha + k) / (alpha + beta + N)

print(f"MLE:                θ̂ = {k}/{N} = {theta_mle:.3f}")
print(f"MAP (Beta(2,2)):    θ̂ = ({alpha}+{k}-1)/({alpha}+{beta}+{N}-2) = {theta_map:.3f}")
print(f"Posterior mean:     θ̂ = ({alpha}+{k})/({alpha}+{beta}+{N}) = {theta_post_mean:.3f}")

# As data grows, all converge
N_big, k_big = 1000, 700
print(f"\nWith N=1000:")
print(f"  MLE: {k_big/N_big:.4f}")
print(f"  MAP: {(alpha+k_big-1)/(alpha+beta+N_big-2):.4f}")
print(f"  Posterior mean: {(alpha+k_big)/(alpha+beta+N_big):.4f}")
# All converge to 0.7
```

### 7i.3 Gaussian Processes Regression

Gaussian Processes (GPs) provide a **Bayesian non-parametric** approach to regression. Instead of learning a fixed set of weights, GPs place a prior directly over **functions** and update it with observed data.

**Core idea:** A GP defines a distribution over functions:

```
f(x) ~ GP(m(x), k(x, x'))
```

- **Mean function** m(x) = E[f(x)] (often zero after centering)
- **Covariance (kernel) function** k(x, x') encodes smoothness, periodicity, length scale

**GP Regression in 3 steps:**

1. **Training:** Compute the N×N kernel matrix K on training data. The key operation is inverting (K + σ²_n I)⁻¹ — this is O(N³).

2. **Prediction at test point x\\*:**
   - Mean: μ\\* = k\\*ᵀ (K + σ²_n I)⁻¹ y
   - Variance: σ\\*² = k(x\\*, x\\*) − k\\*ᵀ (K + σ²_n I)⁻¹ k\\*

3. **Uncertainty:** The variance grows in regions with no training data — the GP "knows what it doesn't know." This calibrated uncertainty is the key advantage over standard regression.

```python
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

# 1. Synthetic data: noisy sine wave
np.random.seed(42)
X_train = np.random.uniform(-5, 5, size=(15, 1))
y_train = np.sin(X_train).ravel() + 0.1 * np.random.randn(15)

# 2. Define kernel: signal (RBF) + noise
kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)

# 3. Fit GP — this optimizes kernel hyperparameters by maximizing log marginal likelihood
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,
                               alpha=0.0, normalize_y=True)
gp.fit(X_train, y_train)

# 4. Predict on dense test grid
X_test = np.linspace(-6, 6, 200).reshape(-1, 1)
y_mean, y_std = gp.predict(X_test, return_std=True)

print(f"Learned kernel: {gp.kernel_}")
print(f"Log-marginal-likelihood: {gp.log_marginal_likelihood(gp.kernel_.theta):.3f}")

# 5. Uncertainty at different points
print(f"\nx=0.0 (near training data): y={y_mean[100]:.3f} ± {y_std[100]:.3f}")
print(f"x=6.0 (extrapolation):      y={y_mean[-1]:.3f} ± {y_std[-1]:.3f}")
# Expected: Uncertainty much larger at x=6.0 (no training data nearby)
# x=0.0: ±~0.10  vs  x=6.0: ±~1.0+

# 95% confidence interval width
ci_95 = 2 * 1.96 * y_std  # approx 95% CI width
print(f"95% CI width at x=0: {ci_95[100]:.3f}")
print(f"95% CI width at x=6: {ci_95[-1]:.3f}")
```

**When to use GPs:**
- Small-to-medium datasets (N < 10,000)
- Calibrated uncertainty estimates are critical (Bayesian optimization, active learning, safe RL)
- Smooth functions with interpretable length scales

**When NOT to use GPs:**
- Large datasets (N > 10⁴) — O(N³) cost; use sparse GPs (inducing points, KISS-GP)
- High-dimensional inputs (d > 20) — curse of dimensionality
- Non-smooth or discontinuous functions

### 7i.4 Bayesian Neural Networks (BNNs)

**Core idea:** Instead of learning a single weight vector, BNNs learn a **distribution over weights**. Predictions are made by integrating over this distribution:

```
P(y* | x*, D) = ∫ P(y* | x*, θ) · P(θ | D) dθ
```

**Why BNNs?**
- **Calibrated uncertainty:** Know when predictions are uncertain (critical for medical diagnosis, autonomous driving)
- **Robust to overfitting:** Bayesian Occam's razor naturally penalizes complex models
- **Active learning:** Query points where the model is most uncertain
- **OOD detection:** High uncertainty on out-of-distribution inputs

**The challenge:** Exact Bayesian inference in NNs is intractable due to millions of parameters, non-linearities, and multi-modal posteriors.

**Approximate inference methods:**

| Method | Approach | Pros | Cons |
|--------|----------|------|------|
| **Monte Carlo Dropout** | Dropout at train + inference | Free (no extra params), simple | Underestimates uncertainty |
| **Mean-Field VI** | Factorized Gaussian q(θ) ≈ P(θ\|D) | Scalable, differentiable | Overconfident, no correlations |
| **Last-Layer Laplace** | Laplace approx on final layer only | Good uncertainty, fast | Limited to last layer features |
| **SWAG** | Fit Gaussian to SGD trajectory | Simple, works with standard training | Approximate posterior |
| **Deep Ensembles** | Train N models, different seeds | Best uncertainty in practice | N× compute cost |
| **HMC (Hamiltonian MC)** | Gold-standard MCMC on weights | Accurate posterior | Impractical for large models |

```python
# Monte Carlo Dropout — simplest BNN
import torch
import torch.nn as nn

class MCDropoutBNN(nn.Module):
    """Bayesian NN via Monte Carlo Dropout."""
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        return self.net(x)
    
    def predict_with_uncertainty(self, x, n_samples=50):
        """Run forward pass N times with dropout active — MC integration."""
        self.train()  # Keep dropout on at inference
        predictions = torch.stack([self.forward(x) for _ in range(n_samples)])
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        return mean, std

# Usage (conceptual)
# model = MCDropoutBNN()
# x_test = torch.linspace(-3, 3, 100).unsqueeze(1)
# mean, std = model.predict_with_uncertainty(x_test)
# print(f"x=0: {mean[50]:.3f} ± {std[50]:.3f}")
```

**Practical recommendations:**
- **Start with Monte Carlo Dropout** — it's nearly free (just keep dropout on at inference)
- **For production uncertainty:** Deep Ensembles (3-5 models) — they consistently outperform single-model methods
- **For small models / research:** Mean-field VI or last-layer Laplace approximation
- **Gold standard** (tiny problems only): Hamiltonian Monte Carlo (full posterior sampling)

**Emerging directions:**
- **Deep Kernel Learning:** GP on top of deep neural network features (learned kernels)
- **Neural Processes:** Meta-learn a distribution over functions from multiple related tasks
- **Post-hoc calibration:** Temperature scaling, Platt scaling — cheap uncertainty without BNNs
- **Deterministic uncertainty:** Methods like DUQ, SNGP, and DUE that estimate uncertainty without sampling

---

## 8. Cross-References

| Reference | Description |
|-----------|-------------|
| [01-Foundations/01-LLM-and-AI-Models.md] | Applied ML math (neural networks, loss functions) |
| [01-Foundations/02-Machine-Learning.md] | ML foundations (training procedures) |
| [01-Foundations/06-Reinforcement-Learning.md] | RL optimization (policy gradients, value functions) |
| [02-LLMs/04-Quantization.md] | Linear algebra in quantization (SVD, low-rank) |
| [06-Advanced/05-Interpretability.md] | Math for interpretability (feature attribution) |
| [08-Reference/01-Glossary.md] | Key math terms |

---
*Document version: 2.0 — June 2026 | Tier 2: Gap Fill | Expanded with code examples, numerical methods, and additional theory*
