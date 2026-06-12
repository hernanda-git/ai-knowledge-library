# 03 - Deep Learning

> A comprehensive, deeply technical exploration of deep learning — from the perceptron through modern diffusion models. Covers fundamental architectures, mathematical derivations, training procedures, and practical implementation guidance.

---

## Table of Contents

1. [History of Deep Learning](#1-history-of-deep-learning)
2. [Multi-Layer Perceptrons (MLPs)](#2-multi-layer-perceptrons-mlps)
3. [Activation Functions](#3-activation-functions)
4. [Convolutional Neural Networks (CNNs)](#4-convolutional-neural-networks-cnns)
5. [CNN Architectures](#5-cnn-architectures)
6. [Recurrent Neural Networks (RNNs)](#6-recurrent-neural-networks-rnns)
7. [Gated RNN Variants: LSTM and GRU](#7-gated-rnn-variants-lstm-and-gru)
8. [Sequence-to-Sequence Models and Attention](#8-sequence-to-sequence-models-and-attention)
9. [Autoencoders](#9-autoencoders)
10. [Generative Adversarial Networks (GANs)](#10-generative-adversarial-networks-gans)
11. [Variational Autoencoders (VAEs)](#11-variational-autoencoders-vaes)
12. [Normalizing Flows](#12-normalizing-flows)
13. [Diffusion Models](#13-diffusion-models)
14. [Further Reading and References](#14-further-reading-and-references)

---

## 1. History of Deep Learning

### 1.1 The Perceptron (1958)

The modern deep learning revolution traces its roots to the **Mark I Perceptron**, developed by Frank Rosenblatt in 1958 at the Cornell Aeronautical Laboratory. The perceptron was the first algorithmically described neural network — a single-layer binary classifier designed for supervised learning.

**Mathematical formulation:**

Given an input vector `x ∈ ℝ^n`, weights `w ∈ ℝ^n`, bias `b ∈ ℝ`, the perceptron computes:

```
z = w^T x + b
ŷ = φ(z) = 1 if z ≥ 0, else 0
```

The perceptron learning algorithm updates weights when a misclassification occurs:

```
w ← w + η(y - ŷ)x
b ← b + η(y - ŷ)
```

Where `η` is the learning rate. The perceptron convergence theorem (Novikoff, 1962) guarantees that if the data is linearly separable, the algorithm will converge to a solution in a finite number of steps. However, Minsky and Papert's 1969 book "Perceptrons" demonstrated fundamental limitations — most famously, the perceptron **cannot learn XOR** because XOR is not linearly separable. This led to the first AI winter.

### 1.2 The Multi-Layer Perceptron and Backpropagation (1986)

The solution to XOR and other non-linear problems was the **multi-layer perceptron (MLP)**, which introduced one or more hidden layers between input and output. The critical enabling technology was **backpropagation**, popularized by Rumelhart, Hinton, and Williams in 1986 (though its roots trace to Werbos's 1974 PhD thesis).

The MLP with one hidden layer computes:

```
h = σ(W_1 x + b_1)
ŷ = σ(W_2 h + b_2)
```

Where `σ` is a differentiable non-linear activation function (typically sigmoid or tanh). The chain rule of calculus enables gradient propagation through the layers.

### 1.3 Convolutional Neural Networks (1989-2012)

The **convolutional neural network (CNN)** was pioneered by Kunihiko Fukushima's Neocognitron (1980) and formalized by Yann LeCun et al. (1989) for handwritten digit recognition (LeNet). CNNs exploit spatial locality through weight sharing and local receptive fields. The 2012 ImageNet victory of AlexNet (Krizhevsky, Sutskever, Hinton) marked the watershed moment that ignited the modern deep learning era.

### 1.4 Recurrent Neural Networks and Long Short-Term Memory (1997)

**Recurrent neural networks (RNNs)**, introduced by Rumelhart et al. (1986) and Jordan (1986), process sequences by maintaining a hidden state. The **Long Short-Term Memory (LSTM)** by Hochreiter and Schmidhuber (1997) solved the vanishing gradient problem through gated cell states, enabling long-range dependency learning.

### 1.5 Generative Adversarial Networks (2014)

Ian Goodfellow's **GAN** (2014) introduced adversarial training as a generative framework — a generator learns to produce realistic data while a discriminator learns to distinguish real from fake. This spawned a proliferation of variants (DCGAN, StyleGAN, CycleGAN, BigGAN).

### 1.6 The Transformer Revolution (2017)

The **Transformer** (Vaswani et al., 2017) replaced recurrence with self-attention, enabling parallel processing of sequences and scaling to unprecedented sizes. This architecture underlies all modern large language models (GPT, BERT, T5, Llama, Claude, Gemini).

### 1.7 Diffusion Models (2020-present)

**Diffusion models** (Sohl-Dickstein 2015; Ho, Jain, Abbeel 2020 - DDPM) reverse a gradual noising process to generate high-quality data. Latent diffusion (Rombach et al., 2022 - Stable Diffusion) reduced computational cost by operating in a compressed latent space, making high-resolution image generation practical. Flow matching and rectified flow represent the latest evolution.

---

## 2. Multi-Layer Perceptrons (MLPs)

### 2.1 Architecture

An MLP is a feedforward neural network composed of:

- An **input layer** of `d` neurons (one per feature)
- One or more **hidden layers** of `n_h` neurons each
- An **output layer** of `c` neurons (one per class for classification, or `1` for regression)

Each layer performs an affine transformation followed by a non-linear activation:

```
a^(1) = W^(1) x + b^(1)
h^(1) = f(a^(1))
a^(2) = W^(2) h^(1) + b^(2)
h^(2) = f(a^(2))
...
ŷ = g(a^(L))
```

Where:
- `W^(ℓ) ∈ ℝ^(d_ℓ × d_{ℓ-1})` is the weight matrix at layer ℓ
- `b^(ℓ) ∈ ℝ^(d_ℓ)` is the bias vector
- `f` is a non-linear activation function (applied element-wise)
- `g` is the output activation (softmax for classification, identity for regression)

### 2.2 Universal Approximation Theorem

The **universal approximation theorem** (Cybenko 1989; Hornik 1991) states that a feedforward network with a single hidden layer containing a finite number of neurons can approximate any continuous function on a compact subset of ℝ^n to arbitrary accuracy, provided the activation function is non-constant, bounded, and continuous (e.g., sigmoid).

**Key insight:** While a single hidden layer suffices in theory, deep networks (many layers) are far more **parameter-efficient** — they can represent complex functions exponentially more compactly than shallow networks (Telgarsky, 2016).

### 2.3 Backpropagation: Full Derivation

Backpropagation computes the gradient of the loss function with respect to all network parameters using the chain rule. We derive it for a general MLP.

**Forward pass** (for a single training example `(x, y)`):

```
z^(1) = W^(1) x + b^(1)
a^(1) = σ(z^(1))
z^(2) = W^(2) a^(1) + b^(2)
a^(2) = σ(z^(2))
...
z^(L) = W^(L) a^(L-1) + b^(L)
ŷ = a^(L) = σ(z^(L))  (for output layer)
```

Loss: `L = ℓ(ŷ, y)` where ℓ is, e.g., mean squared error or cross-entropy.

**Backward pass:**

We define the **error signal** (delta) at each neuron as the derivative of the loss with respect to the pre-activation:

```
δ_j^(ℓ) = ∂L / ∂z_j^(ℓ)
```

For the **output layer** (layer L):

```
δ_j^(L) = ∂L / ∂z_j^(L)
        = ∑_k (∂L / ∂a_k^(L)) · (∂a_k^(L) / ∂z_j^(L))
        = (∂L / ∂a_j^(L)) · σ'(z_j^(L))
```

Because `a_j^(L) = σ(z_j^(L))` and `a_k^(L)` for k≠j does not depend on `z_j^(L)`.

In vector form:
```
δ^(L) = ∇_a L ⊙ σ'(z^(L))
```

Where `⊙` is the Hadamard (element-wise) product.

For cross-entropy loss with softmax output:
```
L = -∑_k y_k log(ŷ_k)
ŷ_k = exp(z_k^(L)) / ∑_j exp(z_j^(L))
```

Then:
```
δ_j^(L) = ŷ_j - y_j
```

A famously elegant gradient — the difference between prediction and ground truth.

For a **hidden layer** ℓ:

```
δ_j^(ℓ) = ∂L / ∂z_j^(ℓ)
        = ∑_k (∂L / ∂z_k^(ℓ+1)) · (∂z_k^(ℓ+1) / ∂a_j^(ℓ)) · (∂a_j^(ℓ) / ∂z_j^(ℓ))
        = ∑_k δ_k^(ℓ+1) · W_kj^(ℓ+1) · σ'(z_j^(ℓ))
```

Where `z_k^(ℓ+1) = ∑_j W_kj^(ℓ+1) a_j^(ℓ) + b_k^(ℓ+1)` and `a_j^(ℓ) = σ(z_j^(ℓ))`.

In vector form (the **backpropagation formula**):
```
δ^(ℓ) = ((W^(ℓ+1))^T δ^(ℓ+1)) ⊙ σ'(z^(ℓ))
```

**Gradients with respect to parameters:**

```
∂L / ∂W_ij^(ℓ) = δ_i^(ℓ) · a_j^(ℓ-1)
∂L / ∂b_i^(ℓ) = δ_i^(ℓ)
```

In matrix form:
```
∂L / ∂W^(ℓ) = δ^(ℓ) · (a^(ℓ-1))^T
∂L / ∂b^(ℓ) = δ^(ℓ)
```

**Parameter updates** (stochastic gradient descent):

```
W^(ℓ) ← W^(ℓ) - η · ∂L / ∂W^(ℓ)
b^(ℓ) ← b^(ℓ) - η · ∂L / ∂b^(ℓ)
```

### 2.4 Backpropagation Algorithm Pseudocode

```
ALGORITHM: Backpropagation
─────────────────────────
Input: Network with L layers, training example (x, y)
Output: Gradients ∂L/∂W^(ℓ), ∂L/∂b^(ℓ) for ℓ = 1, ..., L

// Forward pass
a^(0) = x
for ℓ = 1 to L:
    z^(ℓ) = W^(ℓ) a^(ℓ-1) + b^(ℓ)
    a^(ℓ) = σ(z^(ℓ))
ŷ = a^(L)
L = loss(ŷ, y)

// Backward pass (output layer)
δ^(L) = ∇_a L ⊙ σ'(z^(L))

// Backward pass (hidden layers)
for ℓ = L-1 down to 1:
    δ^(ℓ) = ((W^(ℓ+1))^T δ^(ℓ+1)) ⊙ σ'(z^(ℓ))

// Compute gradients
for ℓ = 1 to L:
    ∂L/∂W^(ℓ) = δ^(ℓ) · (a^(ℓ-1))^T
    ∂L/∂b^(ℓ) = δ^(ℓ)
```

### 2.5 Practical Considerations for MLPs

**Weight initialization:**
- **Xavier/Glorot initialization**: `W ~ U(-√(6/(n_in + n_out)), √(6/(n_in + n_out)))` for tanh/sigmoid
- **He initialization**: `W ~ N(0, √(2/n_in))` for ReLU
- Proper initialization prevents vanishing/exploding gradients in deep networks

**Normalization:**
- **Batch normalization** (Ioffe & Szegedy, 2015): Normalize each mini-batch to zero mean and unit variance, then learn scale γ and shift β
- **Layer normalization** (Ba et al., 2016): Normalize across the feature dimension; preferred in Transformers
- Both accelerate training, enable higher learning rates, and provide some regularization

**Regularization:**
- **L1/L2 weight decay**: Adds penalty `λ||W||_2^2` or `λ||W||_1` to the loss
- **Dropout** (Srivastava et al., 2014): Randomly sets a fraction `p` of neurons to zero during training — prevents co-adaptation
- **Early stopping**: Monitor validation loss and stop when it begins to increase
- **Data augmentation**: Artificially expand training set with label-preserving transformations

**Optimization:**
- **SGD with momentum**: `v_t = γv_{t-1} + η∇L(θ)`; `θ ← θ - v_t`
- **Adam** (Kingma & Ba, 2015): Adaptive moment estimation with bias-corrected first and second moments
- **AdamW** (Loshchilov & Hutter, 2019): Decoupled weight decay — Adam with proper L2 regularization

---

## 3. Activation Functions

### 3.1 Sigmoid (Logistic)

```
σ(x) = 1 / (1 + e^{-x})
σ'(x) = σ(x) · (1 - σ(x))
```

**Range:** (0, 1)
**Properties:**
- Smooth, differentiable, monotonic
- Outputs can be interpreted as probabilities
- **Saturation problem**: Gradient vanishes for |x| > ~5 (σ'(x) → 0)
- Not zero-centered — outputs are always positive, causing zigzag gradient updates
- `σ'(0) = 0.25` — acceptable but not ideal

**When to use:** Output layer for binary classification. Rarely used in hidden layers since the 2010s due to vanishing gradients in deep networks.

### 3.2 Tanh (Hyperbolic Tangent)

```
tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x}) = 2σ(2x) - 1
tanh'(x) = 1 - tanh^2(x)
```

**Range:** (-1, 1)
**Properties:**
- Zero-centered — mitigates the zigzag problem of sigmoid
- Still saturates for large |x|, though gradients are steeper near x=0 (tanh'(0) = 1.0)
- Often preferred over sigmoid in hidden layers of shallow networks or RNNs

**When to use:** Hidden layers in smaller networks or when zero-centered output is beneficial. Largely supplanted by ReLU variants in deep feedforward networks.

### 3.3 ReLU (Rectified Linear Unit)

```
ReLU(x) = max(0, x)
ReLU'(x) = 1 if x > 0, else 0
```

**Range:** [0, ∞)
**Properties:**
- Computationally cheap — just max(0, x)
- Non-saturating for x > 0 — mitigates vanishing gradient
- Sparse activation — approximately 50% of neurons are "dead" (output 0)
- **Dying ReLU problem**: If a neuron's weights push all inputs to negative, the gradient is 0 and the neuron never recovers
- Not differentiable at x=0 (practical implementations use subgradient 0 or 1)

**Why ReLU works:** The non-saturating nature allows gradients to flow freely through active neurons, enabling training of very deep networks (much deeper than sigmoid/tanh networks).

**When to use:** Default choice for hidden layers in CNNs and MLPs. The most widely used activation since 2012.

### 3.4 Leaky ReLU

```
LeakyReLU(x) = max(αx, x)  where α ∈ (0,1), typically 0.01 or 0.2
LeakyReLU'(x) = 1 if x > 0, else α
```

**Range:** (-∞, ∞)
**Properties:**
- Addresses dying ReLU by allowing a small gradient when x < 0
- Still non-saturating for x > 0
- The hyperparameter α is typically small (0.01)—too large risks linearity

**Parametric ReLU (PReLU)** learns α via backprop, potentially improving performance at the cost of additional parameters.

### 3.5 ELU (Exponential Linear Unit)

```
ELU(x) = x if x ≥ 0, else α(e^x - 1)
ELU'(x) = 1 if x > 0, else ELU(x) + α
```

**Range:** (-α, ∞)
**Properties:**
- Smooth (differentiable at x=0 when α=1)
- Negative values push mean activation toward zero, speeding up learning
- Saturates for very negative inputs — provides noise-robustness
- More computationally expensive than ReLU due to exp()
- α controls the value at which negative saturation occurs (typically 1.0)

**When to use:** When training deeper networks where mean-shift toward zero helps convergence. Often outperforms ReLU on some benchmarks.

### 3.6 SELU (Scaled Exponential Linear Unit)

```
SELU(x) = λ · (x if x > 0, else α(e^x - 1))
where λ ≈ 1.0507, α ≈ 1.6733
```

**Properties:**
- Designed for **self-normalizing networks** (Klambauer et al., 2017)
- Under specific conditions (normalized weights, standardized inputs), the output of each layer automatically converges to zero mean and unit variance
- Enables training of very deep networks without batch normalization
- Requires careful initialization (LeCun Normal) and specific dropout variant (AlphaDropout)

**Theoretical guarantee:** For networks with SELU activations, there exists a fixed point where mean and variance are preserved across layers, enabling deep networks to be self-normalizing.

### 3.7 GELU (Gaussian Error Linear Unit)

```
GELU(x) = x · Φ(x)  where Φ(x) = P(X ≤ x) for X ~ N(0,1)
        ≈ 0.5x · (1 + tanh(√(2/π) · (x + 0.044715x^3)))
        ≈ x · σ(1.702x)  (sigmoid approximation)
```

**Range:** (-0.17..., ∞) — has a slight negative dip near x=0

**GELU'(x) = Φ(x) + x · φ(x)** where φ(x) is the standard normal PDF.

**Properties:**
- Smooth everywhere (unlike ReLU's kink at 0)
- Non-convex, non-monotonic — has a small negative region for x ∈ (-1, 0)
- Provides a probabilistic interpretation: gates input by its probability of being "greater than random"
- Used in BERT, GPT-3, ViT, and virtually all modern Transformer models
- Outperforms ReLU empirically in deep Transformer architectures

**Approximation quality:** The tanh approximation is accurate to ~0.001 over the range [-3, 3] and is the form used in most implementations.

### 3.8 Swish (SiLU — Sigmoid Linear Unit)

```
Swish(x) = x · σ(βx)  where σ is the sigmoid function
Swish'(x) = β · Swish(x) + σ(βx) · (1 - β · Swish(x))
         = σ(βx) + βx · σ(βx) · (1 - σ(βx))
```

**Range:** (-0.5/β..., ∞) for β > 0

**Properties:**
- Self-gated: `x · σ(x)` — the input gates itself
- Smooth, non-monotonic (dips slightly below 0 for x negative)
- The **β parameter** controls the "closeness" to ReLU:
  - As β → ∞: Swish(x) → ReLU(x) (hard gating)
  - As β → 0: Swish(x) → x/2 (linear scaling)
  - β=1 is the standard SiLU (Ramachandran et al., 2017)
- **No dead neurons** — the gradient is always positive for x < 0 (unlike ReLU)
- Used in EfficientNet (Swish), Llama (SiLU), and many modern architectures

**When to use:** Generally outperforms ReLU in deep networks at the cost of more computation. Preferred in attention-based architectures.

### 3.9 SwiGLU (Swish-Gated Linear Unit)

```
SwiGLU(x) = Swish(W_1 x + b_1) ⊙ (W_2 x + b_2)
```

**Structure:** A gated linear unit where one path uses Swish activation and the other is a linear projection. The component-wise product of the gated and linear paths gives the output.

**Properties:**
- From Shazeer (2020): "GLU Variants Improve Transformer"
- The Swish gating mechanism provides a "soft" version of ReLU gating
- Typically uses an **intermediate expansion factor of 8/3** (≈2.67) times the hidden dimension
- Used in PaLM, Llama 2/3, Mistral, GPT-4 — the standard FFN activation in modern LLMs
- More expressive than plain Swish but uses more parameters (3 weight matrices instead of 2)

**Comparison to ReLU FFN:**
```
ReLU FFN: FFN(x) = W_2 · ReLU(W_1 x + b_1) + b_2
  Params: 2 · d_model · d_ff (where d_ff = 4 · d_model typically)

SwiGLU FFN: FFN(x) = (Swish(W_1 x + b_1) ⊙ (W_2 x + b_2)) · W_3 + b_3
  Params: 3 · d_model · d_ff
  With d_ff adjusted to (8/3)·d_model, parameter count matches ReLU FFN
```

### 3.10 Mish

```
Mish(x) = x · tanh(ln(1 + e^x)) = x · tanh(softplus(x))
Mish'(x) = sech^2(softplus(x)) · x · σ(x) + Mish(x)/x
```

**Range:** Approximately (-0.31..., ∞)

**Properties:**
- Smooth everywhere (like Swish, GELU)
- **Self-gated** but with the softplus inside tanh, creating a different curvature than Swish
- Has a slight negative dip at x ≈ -0.5 (minimum ≈ -0.308)
- Empirically shown to outperform Swish+BN in some vision tasks (Misra, 2019)
- More computationally expensive than Swish due to nested softplus and tanh
- Not widely adopted in production compared to GELU/Swish/SwiGLU

### 3.11 Activation Function Comparison Summary

| Function | Range | Smooth | Zero-Centered | Dead Neurons | Compute Cost | Saturation |
|---|---|---|---|---|---|---|
| Sigmoid | (0, 1) | Yes | No | No | Medium (exp) | Yes |
| Tanh | (-1, 1) | Yes | Yes | No | Medium (exp) | Yes |
| ReLU | [0, ∞) | No | No | Yes | Cheap | No (pos) |
| Leaky ReLU | (-∞, ∞) | No | No | No | Cheap | No |
| ELU | (-α, ∞) | Yes | Near | No | Medium (exp) | Yes (neg) |
| SELU | (λ·(-α), ∞) | Yes | Self-norm | No | Medium (exp) | Yes (neg) |
| GELU | (-0.17..., ∞) | Yes | No | No | Medium (Φ) | No |
| Swish/SiLU | (-0.28..., ∞) | Yes | No | No | Medium (σ) | No |
| Mish | (-0.31..., ∞) | Yes | No | No | High (exp+tanh) | No |

### 3.12 Practical Guidance for Activation Selection

1. **MLP hidden layers**: ReLU is the safe default. Use Swish/GELU if you have compute budget.
2. **CNN hidden layers**: ReLU is standard. Swish used in EfficientNet. Mish can give marginal gains.
3. **Transformer FFN**: SwiGLU is the dominant modern choice (Llama, Mistral, GPT-4). GELU (BERT, GPT-2).
4. **Output layer**: Softmax (multiclass), Sigmoid (binary/multi-label), Linear (regression).
5. **Very deep networks (>50 layers)**: Consider SELU for self-normalizing properties, or use Swish/GELU with layer normalization.
6. **Mobile/edge devices**: ReLU is fastest due to zero compute cost and no exponential operations.

---

## 4. Convolutional Neural Networks (CNNs)

### 4.1 The Convolution Operation

A convolution is a mathematical operation that combines two functions to produce a third. In CNN context, we convolve an input tensor (image) with a kernel (filter).

**Discrete 2D convolution:**

```
S(i, j) = (I * K)(i, j) = Σ_m Σ_n I(m, n) · K(i - m, j - n)
```

In practice, CNNs use **cross-correlation** (which is equivalent up to kernel rotation):

```
S(i, j) = (I ⋆ K)(i, j) = Σ_m Σ_n I(i + m, j + n) · K(m, n)
```

For a multi-channel input with `C_in` channels and `C_out` filters of size `K_h × K_w`:

```
S_k(i, j) = Σ_{c=1}^{C_in} Σ_{m=1}^{K_h} Σ_{n=1}^{K_w} I_c(i+m, j+n) · K_k_c(m, n) + b_k
```

**Output spatial dimensions:**

Given input size `H × W`, kernel size `K_h × K_w`, padding `P`, stride `S`:

```
H_out = floor((H - K_h + 2P) / S) + 1
W_out = floor((W - K_w + 2P) / S) + 1
```

**Parameter count per layer:** `C_out × (C_in × K_h × K_w + 1)` — the +1 is for bias.

### 4.2 Properties of Convolution

**Weight sharing:** The same kernel is applied across all spatial locations. This dramatically reduces parameters compared to fully-connected layers and provides translation equivariance.

**Local connectivity:** Each output neuron connects to only a local region of the input (the receptive field). This matches the local structure of natural images.

**Sparse interactions:** A single output neuron depends on `K_h × K_w × C_in` input values, not the entire input.

**Equivariance to translation:** Shifting the input by (dx, dy) and convolving produces approximately the same result as convolving then shifting the output (exact down to boundary effects).

**Receptive field:** After L layers of 3×3 convolutions with stride 1, the receptive field is (2L+1) × (2L+1). For dilated convolutions with dilation d: (1 + d·(K-1)) per layer.

### 4.3 Pooling Operations

**Max Pooling:** Takes the maximum value in each pooling window. Provides local translation invariance. Most common in vision CNNs.

```
MaxPool(I)(i, j) = max_{m,n ∈ window} I(i + m, j + n)
```

**Average Pooling:** Takes the mean value in each pooling window. Less aggressive than max pooling.

```
AvgPool(I)(i, j) = (1/|window|) · Σ_{m,n ∈ window} I(i + m, j + n)
```

**Global Average Pooling (GAP):** Pools the entire feature map to a single value per channel. Used instead of fully-connected layers before the output — has zero parameters, prevents overfitting, and is structure-agnostic.

**Strided Convolution vs Pooling:** Modern architectures (ResNet, ConvNeXt) often use strided convolutions (stride=2) instead of explicit pooling layers, as they are learnable and more expressive.

**Other pooling variants:**
- **Lp Pooling**: `(Σ |x|^p)^(1/p)` — generalizes average (p=1) and max (p=∞)
- **Stochastic Pooling** (Zeiler & Fergus, 2013): Samples from a multinomial distribution over spatial regions
- **Spatial Pyramid Pooling** (He et al., 2015): Pools at multiple scales to handle variable-sized inputs
- **Soft Pooling** (Stergiou et al., 2021): Differentiable softmax-weighted pooling

### 4.4 Convolution Mathematics: Detailed Derivations

**Gradient of convolution (for backpropagation):**

During backprop, we need to compute:
1. `∂L / ∂K` — gradient w.r.t. kernel weights
2. `∂L / ∂I` — gradient w.r.t. input (to propagate to earlier layers)

Given `S = I * K` (cross-correlation form), and `δ = ∂L/∂S` from above:

```
∂L / ∂K_{m,n} = Σ_i Σ_j δ_{i,j} · I_{i+m, j+n}
```

This is itself a convolution of `δ` with `I` (technically cross-correlation).

```
∂L / ∂I_{i,j} = Σ_m Σ_n δ_{i-m, j-n} · K_{m,n}
              = (δ full-convolved with rotated K)
```

This is a **full convolution** with the kernel rotated 180°.

**Dilated Convolution:**

```
S(i, j) = Σ_m Σ_n I(i + d·m, j + d·n) · K(m, n)
```

Where `d` is the dilation rate. Dilated convolutions expand the receptive field without increasing parameters.

**Depthwise Separable Convolution:**

Factorizes a standard convolution into:
1. **Depthwise convolution**: Apply one filter per input channel (no cross-channel mixing)
2. **Pointwise convolution**: 1×1 convolution to mix channels

```
Parameters standard: C_in × C_out × K_h × K_w
Parameters depthwise: C_in × K_h × K_w + C_in × C_out × 1 × 1
```

The ratio is approximately `1/C_out + 1/K^2`, offering dramatic parameter reduction at K=3.

### 4.5 Transposed Convolution (Deconvolution)

Used for upsampling in encoder-decoder architectures (segmentation, super-resolution, GAN generators).

**Forward pass:** Given input feature map `x` of size `H × W`, kernel `K`, stride `S`, padding `P`, output size is:

```
H_out = S · (H - 1) + K - 2P
```

Transposed convolution can be thought of as:
1. Insert `S-1` zeros between every element of input
2. Pad the resulting array
3. Apply standard convolution with unit stride

**Checkerboard artifacts:** Transposed convolutions with kernel sizes not divisible by stride can create artifacts. Better alternatives: resize + convolution, or sub-pixel convolution (Shi et al., 2016).

---

## 5. CNN Architectures

### 5.1 LeNet-5 (LeCun et al., 1998)

Designed for handwritten digit recognition (MNIST). The pioneering CNN that demonstrated the power of learned hierarchical features.

**Architecture:**
```
Input: 32×32 grayscale image
Layer 1: Conv(5×5, 6 filters) → AvgPool(2×2, stride 2) → Tanh
Layer 2: Conv(5×5, 16 filters) → AvgPool(2×2, stride 2) → Tanh
Layer 3: FC(120) → Tanh
Layer 4: FC(84) → Tanh
Layer 5: FC(10) → Softmax
Total params: ~60,000
```

**Key innovations:** Use of local receptive fields, weight sharing, and spatial subsampling. The gradients are computed via backpropagation (LeNet already used proper BP, though it wasn't yet widely popularized).

### 5.2 AlexNet (Krizhevsky, Sutskever, Hinton, 2012)

The landmark architecture that won ImageNet 2012 with a dramatic margin (top-5 error 15.3% vs 26.2% for second place). Ignited the deep learning revolution.

**Architecture:**
```
Input: 227×227×3 RGB image
Conv1: 96 filters of 11×11, stride 4, padding 0 → ReLU → LRN → MaxPool(3×3, stride 2)
Conv2: 256 filters of 5×5, stride 1, padding 2 → ReLU → LRN → MaxPool(3×3, stride 2)
Conv3: 384 filters of 3×3, stride 1, padding 1 → ReLU
Conv4: 384 filters of 3×3, stride 1, padding 1 → ReLU
Conv5: 256 filters of 3×3, stride 1, padding 1 → ReLU → MaxPool(3×3, stride 2)
FC6: 4096 → ReLU → Dropout(0.5)
FC7: 4096 → ReLU → Dropout(0.5)
FC8: 1000 → Softmax
Total params: ~60 million
```

**Key innovations:**
- **ReLU activation** — enabled much deeper training than tanh/sigmoid
- **Dropout (0.5)** — reduced overfitting of the 4096-unit FC layers
- **Data augmentation** — random crops, horizontal flips, PCA color jitter
- **Local Response Normalization (LRN)** — lateral inhibition, now largely deprecated
- **GPU training (2x GTX 580)** — demonstrated practical parallelism

**Training details:** SGD with momentum 0.9, batch size 128, learning rate 0.01 reduced manually by factor 10 when validation accuracy plateaued. Trained for ~90 epochs over 5-6 days.

### 5.3 VGGNet (Simonyan & Zisserman, 2014)

Demonstrated that depth is critical: VGG-16 (16 weight layers) and VGG-19 (19 weight layers).

**Architecture design principle:** Use only **3×3 convolutions** with stride 1, padding 1 (preserves spatial size). Spatial downsampling via max pooling (2×2, stride 2) after some conv blocks.

```
Input: 224×224×3
Block 1: [Conv(64, 3×3) × 2] → MaxPool(2,2)
Block 2: [Conv(128, 3×3) × 2] → MaxPool(2,2)
Block 3: [Conv(256, 3×3) × 3] → MaxPool(2,2)
Block 4: [Conv(512, 3×3) × 3] → MaxPool(2,2)
Block 5: [Conv(512, 3×3) × 3] → MaxPool(2,2)
FC layers: 4096 → 4096 → 1000
```

**Why 3×3?** Two stacked 3×3 convolutions have a 5×5 receptive field (3 layers = 7×7) but with fewer parameters: 2 × 9 × C² = 18C² vs 25C² for a single 5×5 layer. The ReLU between two 3×3 layers also adds non-linearity.

**Total params:** VGG-16 has 138M parameters, mostly from the three FC layers (4096+4096+1000=~119M). Modern architectures use GAP to avoid this parameter explosion.

**Practical note:** VGG is slow to train and has large memory footprint, but its simplicity and uniform architecture make it excellent for feature extraction and transfer learning. It remains a common baseline.

### 5.4 ResNet (He et al., 2015 — Best Paper CVPR 2016)

Introduced **residual learning**, enabling training of networks with 152+ layers. Before ResNet, deeper networks suffered from **degradation** — higher training error, not just generalization gap, caused by vanishing gradients.

**Residual Block:**

```
x → [Conv(3×3) → BN → ReLU → Conv(3×3) → BN] → + → ReLU
    └────────────────────────────────── shortcut ───┘

y = F(x, {W_i}) + x
```

Where `F(x) = W_2 · σ(W_1 x + b_1) + b_2` and `σ` is ReLU. When input/output dimensions don't match, the shortcut uses either (a) zero-padded identity or (b) 1×1 convolution with stride to match dimensions.

**Why residual connections work:**
- **Gradient highway**: The identity skip connection allows gradients to flow directly through the network without vanishing — `∂L/∂x = ∂L/∂y · (1 + ∂F/∂x)`
- **Ensemble interpretation**: ResNets behave like ensembles of shallow networks (Veit et al., 2016) — during forward pass, information can skip any subset of residual blocks
- **Easier optimization**: It's easier to learn the residual `F(x)` near zero than to learn the full mapping `H(x) = F(x) + x` from scratch

**Common ResNet variants:**
```
ResNet-18: 4 blocks of [2, 2, 2, 2] basic blocks (64→128→256→512)
ResNet-34: 4 blocks of [3, 4, 6, 3] basic blocks
ResNet-50: 4 blocks of [3, 4, 6, 3] bottleneck blocks
ResNet-101: 4 blocks of [3, 4, 23, 3] bottleneck blocks
ResNet-152: 4 blocks of [3, 8, 36, 3] bottleneck blocks
```

**Bottleneck Block** (for ResNet-50+):
```
1×1 Conv (reduce channels by 4×)
3×3 Conv (spatial processing)
1×1 Conv (expand channels back)
```
This reduces computational cost: a 3×3 on 256 channels vs on 64 channels after reduction.

**Bottleneck parameter comparison:**
```
Basic block: 2 × 3×3 × 256 × 256 = 1,179,648 params
Bottleneck: 1×1(256→64) + 3×3(64→64) + 1×1(64→256) = 64*256 + 9*64*64 + 256*64 = 69,632 params
```
The bottleneck is ~17× more parameter-efficient.

**Pre-activation ResNet (He et al., 2016):**
```
BN → ReLU → Conv → BN → ReLU → Conv
```
Putting BN and ReLU before the convolution (instead of after addition) improves gradient flow and regularization.

**ResNeXt (Xie et al., 2017):**
Applies **grouped convolutions** — splits filters into `cardinality` groups, each learning independent features. The key insight was that **increasing cardinality** (number of groups) is more effective than increasing depth or width.

```
ResNeXt basic block with cardinality = 32:
Input (256 channels)
→ 1×1 Conv (reduce to 4 per group × 32 groups = 128 channels)
→ 32 parallel 3×3 Conv groups (each: 4→4 channels)
→ Concatenate (128 channels)
→ 1×1 Conv (expand to 256 channels)
```

**ResNeSt (Zhang et al., 2022):**
Adds **Split-Attention** to ResNeXt — within each group (cardinal group), further split into radix groups and apply channel-wise attention (SE-Net style). Provides the "ResNet + Attention" combination that achieves state-of-the-art efficiency.

### 5.5 Inception (GoogLeNet) Family

**Inception v1 (Szegedy et al., 2014):**
The core insight is that objects appear at different scales — so why choose one kernel size? Use **multiple kernel sizes in parallel** (1×1, 3×3, 5×5, and 3×3 max pooling) and concatenate outputs.

**Naive Inception module:** Concatenate outputs of 1×1, 3×3, 5×5 convs, and 3×3 max pool. Problem: 5×5 conv on high-dim input is expensive.

**Inception module with dimensionality reduction:** Add 1×1 conv before expensive 3×3 and 5×5 convolutions to reduce channel count by 4×.

```
Inception module:
Input (e.g., 192 channels)
├── 1×1 Conv (64)
├── 1×1 Conv (96) → 3×3 Conv (128)
├── 1×1 Conv (16) → 5×5 Conv (32)
└── 3×3 MaxPool → 1×1 Conv (32)
Output: Concatenate [64, 128, 32, 32] = 256 channels
```

**Auxiliary classifiers:** Two side branches with softmax at intermediate layers to inject additional gradient signal and regularize the network.

**Inception v2/v3 (Szegedy et al., 2015):**
- Factorized 7×7 convolutions into 3×3 layers
- Factorized 5×5 into two 3×3 layers
- Factorized asymmetric convolutions: n×n → 1×n + n×1 (e.g., 3×3 → 1×3 + 3×1), which saves 33% parameters
- **Batch normalization** incorporated
- **Label smoothing** — prevents overconfidence by replacing hard targets (0/1) with smoothed targets (ε/(K-1) for wrong classes, 1-ε for correct)

**Inception v4 / Inception-ResNet (Szegedy et al., 2016):**
Combined Inception modules with residual connections. Inception-ResNet v1 and v2 use residual connections with Inception-style filters, plus **scale of residuals** (≈0.1-0.3) to stabilize training.

**Xception (Chollet, 2017):**
Replaces Inception modules with **depthwise separable convolutions** applied in a specific order: pointwise (1×1) → depthwise (3×3), with ReLU only after the pointwise convolution. Xception hypothesizes that the cross-channel and spatial correlation maps are sufficiently decoupled to be processed independently.

### 5.6 DenseNet (Huang et al., 2017)

Introduces **dense connectivity** — each layer receives feature maps from all preceding layers as input and passes its own feature maps to all subsequent layers.

```
x_ℓ = H_ℓ([x_0, x_1, ..., x_{ℓ-1}])
```

Where `[·]` is channel-wise concatenation.

**Key properties:**
- **Growth rate k**: Each layer produces `k` new feature maps (typically 12-32). Total feature maps grow linearly, not quadratically.
- **Parameter efficiency**: With `k=32`, a DenseNet with 264 layers has only ~30M parameters vs 62M for ResNet-101
- **Gradient flow**: Direct gradient path from early to late layers and vice versa — strong regularizing effect
- **Feature reuse**: Earlier features are directly accessible to later layers

**DenseNet architecture by blocks:**
```
Input → Conv(7×7, stride 2) → MaxPool(3×3, stride 2)
→ Dense Block 1 (6 layers, growth rate 32)
→ Transition Layer (1×1 conv + 2×2 avg pool)
→ Dense Block 2 (12 layers)
→ Transition Layer
→ Dense Block 3 (24 layers)
→ Transition Layer
→ Dense Block 4 (16 layers)
→ Global Avg Pool → FC(1000)
```

**Transition layer:** 1×1 conv (reduces channels by compression factor θ=0.5) followed by 2×2 avg pool.

**Memory concern:** Concatenation of all previous feature maps creates large intermediate tensors. **Memory-efficient DenseNet** implementations compute gradients without storing all intermediate activations, reducing memory from O(L^k) to O(L).

### 5.7 EfficientNet (Tan & Le, 2019)

Systematically study of **compound scaling**: jointly scaling network depth, width, and resolution yields better performance than scaling any single dimension.

**Problem:** Previous architectures scaled one dimension (e.g., ResNet depth from 18→152, resolution from 224→320). EfficientNet asks: what is the optimal ratio between depth (d), width (w), and resolution (r) given a compute budget?

**Scaling method:**
```
depth: d = α^φ
width: w = β^φ
resolution: r = γ^φ
subject to: α · β^2 · γ^2 ≈ 2  (FLOPS constraint)
```

Where α, β, γ are constants determined by a small grid search on the baseline model (EfficientNet-B0), and φ is the scaling coefficient.

The FLOPS constraint comes from: width scales quadratically (W²), resolution scales quadratically (R²), and depth scales linearly (D). So FLOPS ∝ d · w² · r² = α·β²·γ².

**Baseline B0 architecture** (MBConv blocks — Mobile Inverted BottleNeck):
```
Stem: Conv3×3 (32 channels)
Stage 1: MBConv1 k3×3, 16 ch
Stage 2: MBConv6 k3×3, 24 ch × 2
Stage 3: MBConv6 k5×5, 40 ch × 2
Stage 4: MBConv6 k3×3, 80 ch × 3
Stage 5: MBConv6 k5×5, 112 ch × 3
Stage 6: MBConv6 k5×5, 192 ch × 4
Stage 7: MBConv6 k3×3, 320 ch × 1
Head: Conv1×1 (1280) → GAP → FC(N classes)
```

**MBConv (Mobile Inverted Bottleneck):** Uses depthwise separable convolution with an inverted residual structure (expand → depthwise → project), Squeeze-and-Excitation (SE) attention, and Swish activation.

**Performance:**
```
Model     | Top-1 Acc | Params  | FLOPS
B0        | 77.1%     | 5.3M    | 0.4B
B1        | 79.1%     | 7.8M    | 0.7B
B2        | 80.1%     | 9.2M    | 1.0B
B3        | 81.6%     | 12M     | 1.8B
B4        | 82.9%     | 19M     | 4.2B
B5        | 83.7%     | 30M     | 9.9B
B6        | 84.0%     | 43M     | 19B
B7        | 84.3%     | 66M     | 37B
```

B0-B7 vary resolution (224→600), depth coefficient, and width coefficient.

**EfficientNetV2 (Tan & Le, 2021):**
Addresses the training speed issues of V1 (MBConv blocks are memory-bandwidth limited, not compute bound).

**Key changes:**
- **Fused-MBConv**: Replace the 3×3 depthwise + 1×1 pointwise expansion with a single 3×3 regular convolution in early stages — faster on modern hardware (TPUs/GPUs)
- **Progressive learning**: Gradually increase image size, regularization strength, and dropout during training
- **Smaller kernel sizes**: Use 3×3 instead of 5×5 where possible
- **Better training recipe**: RandAugment, MixUp, DropPath

V2 achieves 2-4× faster training than V1 while maintaining accuracy.

### 5.8 MobileNet Family

**MobileNetV1 (Howard et al., 2017):**
Uses **depthwise separable convolutions** as the building block. Designed for mobile and embedded vision applications.

**Architecture:**
```
Depthwise Conv3×3 → BN → ReLU6 → Pointwise Conv1×1 → BN → ReLU6
```

ReLU6 (ReLU with max activation of 6) is used for quantization-friendly characteristics (tighter range).

**Parameters:** MobileNet-224 uses ~4.2M params (vs 138M for VGG-16) with only a modest accuracy drop.

**Width multiplier α ∈ (0,1):** Thins the network uniformly at each layer. At α=0.5, params drop to 1.3M.

**Resolution multiplier ρ ∈ (0,1):** Reduces input resolution and thus internal activations.

**MobileNetV2 (Sandler et al., 2018):**
Introduces **inverted residuals with linear bottlenecks**:

```
Expansion: 1×1 Conv (expand low-dim to high-dim, factor t=6)
Depthwise: 3×3 Depthwise Conv (on expanded dim)
Projection: 1×1 Conv (linear — no ReLU — back to low-dim)
Shortcut connection between bottlenecks (when spatial size matches)
```

**Why linear bottleneck?** ReLU destroys information in low-dimensional spaces but preserves it in high-dimensional spaces. The bottleneck has low dimension, so removing ReLU after the projection preserves information.

**MobileNetV3 (Howard et al., 2019):**
Uses **NAS (Neural Architecture Search)** to find the optimal layer configurations. Incorporates Squeeze-and-Excitation (SE) modules and **hard-swish** activation:

```
h-swish(x) = x · ReLU6(x + 3) / 6
```

Hard-swish is a piecewise linear approximation of Swish that is faster to compute and quantization-friendly.

### 5.9 ShuffleNet (Zhang et al., 2018)

Designed for extremely constrained compute budgets (e.g., mobile GPUs). Key innovations:

**Pointwise Group Convolution + Channel Shuffle:**

Standard 1×1 convolution is expensive (1 × 1 × C_in × C_out). Using grouped 1×1 convolutions reduces cost, but without cross-group communication, features are isolated. **Channel shuffle** permutes the channels so each group receives inputs from all previous groups.

```
Channel Shuffle Operation:
Input: g groups, each with n channels (= g·n total)
Reshape: (g, n) → (n, g) transposed
Flatten: back to g·n channels, now interleaved
```

**ShuffleNet unit:** Pointwise group conv → channel shuffle → depthwise conv → pointwise group conv.

### 5.10 ConvNeXt (Liu et al., 2022)

A modernized pure CNN that matches Transformer performance. The authors systematically "modernized" a ResNet-50 to incorporate design principles from Swin Transformers.

**Modernization steps:**
1. **Training recipe**: AdamW, data augmentation (MixUp, CutMix, RandAugment), regularization (Stochastic Depth, Label Smoothing)
2. **Macro design**: Stage compute ratio from ResNet (3:4:6:3) → Swin-T (1:1:3:1)
3. **Patchify stem**: Replaced 7×7 conv + maxpool with 4×4 conv (stride 4) — like ViT patch embedding
4. **Depthwise conv**: Replace 3×3 regular conv with 3×3 depthwise conv
5. **Inverted bottleneck**: Follows MobileNetV2 — Dim: 96 → 384 → 96 (expansion ratio 4)
6. **Large kernel**: Increase kernel size from 3 to 7 (depthwise)
7. **LayerNorm → BatchNorm**: Replace BN with LN (following Transformers)
8. **Fewer activation functions**: Remove GELUs between depthwise conv and pointwise projection; keep only one GELU in the expansion

ConvNeXt achieved 87.8% ImageNet top-1 on ConvNeXt-XL (comparable to Swin-L), proving that CNNs can match Transformers with proper design.

### 5.11 ResNet-RS (Bello et al., 2021)

Demonstrated that proper training recipes (not architectural changes alone) could dramatically improve ResNet performance. ResNet-RS bumps ResNet-50 from 76% to ~80% on ImageNet with improved regularization.

**Improvements:**
- Increased training epochs (350 from 90)
- Cosine learning rate schedule instead of step decay
- Label smoothing (ε=0.1)
- Stochastic Depth (drop path probability 0.05)
- Increased image size (128→224→160 progressive scaling)
- RandAugment and MixUp augmentation
- EMA (Exponential Moving Average) of weights

**Key insight:** The gap between "old" and "new" architectures was largely attributable to better training techniques, not just architecture design.

### 5.12 Practical Guide for CNN Architecture Selection

| Use Case | Recommended Architecture | Rationale |
|---|---|---|
| ImageNet-scale classification | ConvNeXt / EfficientNetV2 | Best accuracy-efficiency trade-off |
| Mobile/edge deployment | MobileNetV3 / ShuffleNetV2 | Minimizes latency and memory |
| Feature extraction (transfer learning) | ResNet-50/101, ConvNeXt | Well-studied, abundant pretrained weights |
| Object detection backbone | ResNet-50/101 with FPN, ConvNeXt | Good multi-scale features |
| Semantic segmentation | ResNet with dilated convolutions | Maintains spatial resolution |
| Limited compute + need accuracy | EfficientNet-B0 to B5 | Compound scaling gives good Pareto frontier |
| Very deep (>100 layers) | ResNet-152 / DenseNet-201 | Skip connections / dense connections prevent degradation |
| Highway / industrial deployment | ResNet family | Most hardware-optimized, widely supported |

---

## 6. Recurrent Neural Networks (RNNs)

### 6.1 Vanilla RNN

A recurrent neural network processes sequential data by maintaining a **hidden state** that is updated at each time step.

```
h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b_h)
ŷ_t = W_hy · h_t + b_y
```

Where:
- `x_t ∈ ℝ^d` is the input at time step t
- `h_t ∈ ℝ^h` is the hidden state
- `ŷ_t ∈ ℝ^o` is the output (prediction)
- `W_hh ∈ ℝ^(h×h)` is the recurrent weight matrix
- `W_xh ∈ ℝ^(h×d)` is the input weight matrix
- `W_hy ∈ ℝ^(o×h)` is the output weight matrix

**Unrolling through time:** The RNN can be "unrolled" into a deep feedforward network with T layers, where each layer shares the same weights. This is purely for analysis — the computation graph has T× the depth.

### 6.2 Backpropagation Through Time (BPTT)

BPTT applies standard backpropagation to the unrolled computation graph. The critical challenge is that the loss at time t depends on hidden states from all earlier time steps.

**Gradient of loss at time t w.r.t. hidden state at time t:**

```
∂L_t / ∂h_t = (∂L_t / ∂ŷ_t) · W_hy^T
```

**Gradient propagated backward through time:**

```
∂L_t / ∂h_k = ∂L_t / ∂h_{k+1} · ∂h_{k+1} / ∂h_k
            = ∂L_t / ∂h_{k+1} · W_hh^T · diag(tanh'(h_{k+1}))
```

**Total gradient w.r.t. W_hh:**

```
∂L / ∂W_hh = Σ_t Σ_{k=1}^{t} (∂L_t / ∂h_k · h_{k-1}^T)
```

### 6.3 The Vanishing and Exploding Gradient Problem

The Jacobian of the hidden state transition involves repeated multiplication by W_hh:

```
∂h_t / ∂h_1 = Π_{k=2}^{t} W_hh^T · diag(tanh'(h_k))
```

If the eigenvalues of W_hh are small (< 1), the gradient vanishes exponentially with sequence length. If they are large (> 1), it explodes.

**Impact:**
- **Vanishing gradients**: Network cannot learn long-range dependencies (>10-20 time steps)
- **Exploding gradients**: Training becomes unstable; weights may go to NaN

**Mitigations:**
- **Gradient clipping**: Scale gradients if their norm exceeds a threshold (typically 1.0 or 5.0)
- **Proper initialization**: Orthogonal initialization of W_hh (suggested by Saxe et al., 2014)
- **Gated architectures** (LSTM, GRU) — the primary practical solution
- **ReLU activation**: Less saturation than tanh, but can cause exploding activations

### 6.4 Gradient Clipping Pseudocode

```
ALGORITHM: Gradient Clipping
────────────────────────────
θ = model parameters
g = ∇L(θ)  (gradients)

// Clip by global norm (most common)
threshold = 1.0
global_norm = sqrt(Σ ||g_i||² for g_i in g)

if global_norm > threshold:
    scale = threshold / global_norm
    g_i = g_i * scale for all g_i

θ ← θ - η · g
```

### 6.5 RNN Training: Practical Considerations

**Sequence batching:** Pad sequences to equal length within a batch. Use masking to ignore padding contributions to the loss.

**Bucketing:** Group sequences of similar length into the same batch to minimize padding waste.

**Truncated BPTT (TBPTT):** Instead of backpropagating through the entire sequence, unroll for a fixed window of T steps. This limits memory usage and speeds training, though it technically biases the gradient computation.

**Teacher forcing:** During training, feed the ground-truth previous output as input to the next time step (instead of the model's own prediction). This stabilizes training but creates **exposure bias** — the model never sees its own errors during training.

**Scheduled sampling** (Bengio et al., 2015): Gradually anneal from teacher forcing to model-generated outputs during training to mitigate exposure bias.

---

## 7. Gated RNN Variants: LSTM and GRU

### 7.1 Long Short-Term Memory (LSTM)

Hochreiter & Schmidhuber (1997) introduced the LSTM to address the vanishing gradient problem. The key innovation is the **cell state** `c_t` — a separate memory line with linear self-loops that allows gradients to flow unchanged over many time steps.

**LSTM Gate Math:**

```
// Forget gate: decides what to discard from cell state
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
// Alternative formulation with separate weight matrices:
f_t = σ(W_hf · h_{t-1} + W_xf · x_t + b_f)

// Input gate: decides what new information to store
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)

// Candidate cell state: new candidate values
\tilde{c}_t = tanh(W_c · [h_{t-1}, x_t] + b_c)

// Cell state update: forget old + add new
c_t = f_t ⊙ c_{t-1} + i_t ⊙ \tilde{c}_t

// Output gate: decides what to output from cell state
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)

// Hidden state (output)
h_t = o_t ⊙ tanh(c_t)
```

**Dimensions:** If `x_t ∈ ℝ^d` and hidden size = h, then `W_f, W_i, W_c, W_o ∈ ℝ^(h × (h+d))`. Total LSTM params = `4 × (h × (h + d) + h)`.

**Why LSTMs solve vanishing gradients:**

The cell state recurrence `c_t = f_t ⊙ c_{t-1} + i_t ⊙ \tilde{c}_t` has additive (not multiplicative) dynamics. The gradient flows through the forget gate:

```
∂c_t / ∂c_{t-1} = diag(f_t) + ∂(other terms)/∂c_{t-1}
```

When `f_t ≈ 1` (forget gate open), the gradient is ≈ 1 and can propagate unchanged. The LSTM learns when to open/close the forget gate, giving it the ability to capture long-range dependencies when needed.

**Peephole connections** (Gers & Schmidhuber, 2000): Add connections from `c_{t-1}` to the gate computations, allowing the gates to "see" the cell state directly.

**LSTM Variants:**
- **Bidirectional LSTM (BiLSTM)**: One LSTM processes forward, another backward, concatenating hidden states at each step
- **Stacked LSTM**: Multiple LSTM layers on top of each other
- **Layer-Normalized LSTM**: Apply LayerNorm to the hidden-candidate computation

### 7.2 Gated Recurrent Unit (GRU)

Cho et al. (2014) introduced the GRU as a simplified alternative to LSTM with fewer parameters (no separate cell state, output gate removed).

**GRU Gate Math:**

```
// Reset gate: controls influence of previous hidden state
r_t = σ(W_r · [h_{t-1}, x_t] + b_r)

// Update gate: combines forget and input gates
z_t = σ(W_z · [h_{t-1}, x_t] + b_z)

// Candidate hidden state (with reset gate applied)
\tilde{h}_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t] + b_h)

// Hidden state update (interpolation of old and new)
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ \tilde{h}_t
```

**Dimensions:** GRU has 3 weight matrices per layer vs LSTM's 4. GRU total params = `3 × (h × (h + d) + h)`.

**GRU vs LSTM tradeoffs:**
- GRU has fewer parameters, trains faster, and needs less data
- LSTM has an explicit cell state that may better capture very long-range dependencies (>100 steps)
- In practice, GRU often matches LSTM performance on many tasks (especially with sufficient data)
- LSTM remains more common in academic baselines; GRU is preferred for compute-constrained settings

### 7.3 Bidirectional RNNs

A bidirectional RNN processes sequences in both forward and backward directions, then concatenates the representations.

```
Forward: h_t^→ = RNN_forward(x_t, h_{t-1}^→)
Backward: h_t^← = RNN_backward(x_t, h_{t+1}^←)
h_t = [h_t^→; h_t^←]
```

Each position's representation incorporates context from both past and future — essential for tasks like named entity recognition, POS tagging, and text classification where full sequence context helps.

**Note:** Bidirectional RNNs cannot be used for autoregressive generation (language modeling) because they peek at future tokens. They are used for **sequence encoding** tasks.

### 7.4 Depth and Stacking

Stacked RNNs (also called deep RNNs) have multiple recurrent layers:

```
h_t^(1) = RNN(x_t, h_{t-1}^(1))
h_t^(2) = RNN(h_t^(1), h_{t-1}^(2))
...
ŷ_t = W_hy · h_t^(L) + b_y
```

Each layer operates at a different timescale (higher layers tend to model slower-changing representations). Residual connections between layers help training for deep (>4 layer) stacks.

---

## 8. Sequence-to-Sequence Models and Attention

### 8.1 Encoder-Decoder Architecture

The seq2seq architecture (Sutskever et al., 2014; Cho et al., 2014) uses two RNNs:

- **Encoder**: Reads the source sequence `(x_1, ..., x_T)` and produces a sequence of hidden states `(h_1, ..., h_T)` and a final context vector `c = h_T` (or a function of all hidden states)
- **Decoder**: Generates the target sequence `(y_1, ..., y_T')` auto-regressively, conditioned on `c`

**Encoder:** 
```
h_t = RNN_enc(x_t, h_{t-1})
```

**Decoder:**
```
s_0 = tanh(W_c · c)  // initialize decoder state from context
s_t = RNN_dec(y_{t-1}, s_{t-1})
p(y_t | y_<t, c) = softmax(W_o · s_t)
```

### 8.2 Bahdanau Attention (Additive Attention)

The fixed context vector `c` becomes a bottleneck. Attention (Bahdanau et al., 2015) computes a **dynamic context vector** at each decoding step.

**Attention mechanism:**

```
// Alignment scores: how well encoder position j matches decoder position i
e_{ij} = v_a^T · tanh(W_a · s_{i-1} + U_a · h_j)

// Attention weights (normalized)
α_{ij} = exp(e_{ij}) / Σ_k exp(e_{ik})

// Context vector (weighted sum of encoder states)
c_i = Σ_j α_{ij} · h_j

// Decoder input (concatenate context with decoder input)
s_i = RNN_dec([y_{i-1}; c_i], s_{i-1})
```

**Properties:**
- The alignment model `v_a^T · tanh(W·s + U·h)` is a small MLP that is jointly trained
- Attention provides an **alignment matrix** that can be visualized to understand what the model focuses on
- The decoder directly accesses the full source sequence at each step, eliminating the bottleneck

### 8.3 Luong Attention (Multiplicative/Dot-Product Attention)

Luong et al. (2015) proposed simpler, more efficient attention variants:

**Score functions:**
```
dot:      score(s_i, h_j) = s_i^T · h_j
general:  score(s_i, h_j) = s_i^T · W_a · h_j
concat:   score(s_i, h_j) = v_a^T · tanh(W_a · [s_i; h_j])
```

Dot-product is the simplest and most efficient but requires matching dimensions. The "general" variant adds a learned transformation.

**Global vs Local Attention:**
- **Global**: Attends to all encoder positions (like Bahdanau)
- **Local**: Predicts a position `p_t` and attends to a window around it — computationally cheaper and differentiable

### 8.4 Teacher Forcing and Scheduled Sampling

**Teacher Forcing:** During training, feed the ground-truth previous token as decoder input at each step:

```
Training:  s_t = RNN_dec(y_{t-1}^*, s_{t-1})  // y* is ground truth
Inference: s_t = RNN_dec(ŷ_{t-1}, s_{t-1})    // ŷ is model-generated
```

**Problem:** Exposure bias — during inference, the model sees its own errors, which compound over time.

**Scheduled Sampling:** Gradually anneal from teacher forcing to autoregressive generation:

```
ε = max(ε_0 · exp(-k · epoch), ε_min)
Use y* with probability ε, ŷ with probability 1-ε
```

**More robust alternatives to scheduled sampling:**
- **Professor forcing** (Lamb et al., 2016): Adversarial training to make the hidden state distributions of teacher-forced and free-running modes indistinguishable
- **Sequence-level objectives**: Train with REINFORCE or minimum risk training to optimize discrete sequence metrics (BLEU, ROUGE)
- **Dataset aggregation (DAgger)**: Collect rollouts and add them to the training set

### 8.5 Bucketing and Padding

**Problem:** Sequences in a batch have variable lengths. Padding to the max length in the dataset wastes computation.

**Bucketing:** Group sequences by length into buckets:
```
Buckets: [1-10], [11-20], [21-30], [31-50], [51-100]
```

Within each bucket, pad to the bucket max length. This reduces per-batch padding by 50-80%.

**Implementation:**
1. Pre-sort all training examples by length
2. Create buckets (manually or via quantile boundaries)
3. During training, shuffle within each bucket and sample batches from buckets
4. Optionally, weight bucket sampling probability by size

---

## 9. Autoencoders

### 9.1 Vanilla Autoencoder

An autoencoder learns to reconstruct its input through a **bottleneck** (lower-dimensional representation). It consists of an encoder `f_φ` and a decoder `g_θ`.

```
z = f_φ(x) = σ(W_e · x + b_e)         // encode
ŷ = g_θ(z) = σ'(W_d · z + b_d)        // decode
L = ||x - ŷ||_2^2                       // reconstruction loss
```

If both encoder and decoder are linear (`σ` = identity), the autoencoder learns the same subspace as PCA. Non-linear activations enable learning of non-linear manifolds.

**Undercomplete vs Overcomplete:**
- **Undercomplete** (bottleneck dim < input dim): Forces the model to learn efficient representations
- **Overcomplete** (bottleneck dim ≥ input dim): Risk of learning identity mapping; requires regularization

### 9.2 Denoising Autoencoder (DAE)

Vincent et al. (2008) proposed corrupting the input and training the autoencoder to reconstruct the clean original:

```
\tilde{x} = x + noise (e.g., Gaussian noise or masking)
z = f_φ(\tilde{x})
ŷ = g_θ(z)
L = ||x - g_θ(f_φ(x + noise))||_2^2
```

The model learns to **denoise**, which implicitly forces it to capture the data manifold (denoising requires understanding of the underlying structure).

**Types of corruption:**
- **Gaussian noise**: `\tilde{x} = x + ε, ε ~ N(0, σ²)`
- **Masking noise**: Randomly set a fraction of input dimensions to 0
- **Salt-and-pepper noise**: Randomly set dimensions to min/max values

**Connection to score matching:** Vincent (2011) showed that denoising autoencoders implicitly estimate the score function `∇_x log p(x)`.

### 9.3 Sparse Autoencoder

Adds a sparsity constraint on the hidden representation. This forces the model to use only a few active neurons per input, encouraging **feature disentanglement**.

**Sparsity methods:**

1. **L1 penalty:** `L_sparse = L_recon + λ · ||z||_1`
2. **KL divergence sparsity:** Enforce that average activation `ρ̂_j` of neuron j equals a target ρ (small, e.g., 0.05):
```
L_sparse = L_recon + β · Σ_j KL(ρ || ρ̂_j)
KL(ρ || ρ̂_j) = ρ · log(ρ/ρ̂_j) + (1-ρ) · log((1-ρ)/(1-ρ̂_j))
```
3. **Competitive (winner-take-all) sparsity:** Only the top-k activations survive; all others are zeroed during forward pass

**Practical note:** Sparse autoencoders have seen a major resurgence in 2023-2024 as a tool for **mechanistic interpretability** of LLMs — training sparse autoencoders on the residual stream activations to discover interpretable features.

### 9.4 Contractive Autoencoder (CAE)

Rifai et al. (2011) add a penalty on the **Jacobian** of the encoder — the sensitivity of the representation to small input changes:

```
L = L_recon + λ · ||J_f(x)||_F^2
```

Where `J_f(x)_ij = ∂z_i / ∂x_j` is the Jacobian of the encoder at x, and `||·||_F` is the Frobenius norm.

For a sigmoid encoder `z = σ(Wx + b)`:
```
||J_f(x)||_F^2 = Σ_i (σ'(W_i·x + b_i))² · ||W_i||_2^2
```

**Properties:**
- Encourages the representation to be locally invariant to input perturbations
- Explicitly penalizes the representation's variance along tangent directions of the data manifold
- Typically yields better features than DAEs for classification tasks
- Computationally expensive (requires JVP for each training example)

### 9.5 Variational Autoencoder (VAE)

The VAE (Kingma & Welling, 2014) is a generative model that merges autoencoders with variational inference. See Section 11 for complete treatment.

---

## 10. Generative Adversarial Networks (GANs)

### 10.1 The GAN Framework

Introduced by Goodfellow et al. (2014), GANs consist of two neural networks playing a minimax game:

- **Generator G(z; θ_g)**: Maps noise `z ~ p(z)` (e.g., N(0, I)) to data space, producing fake samples `G(z)`
- **Discriminator D(x; θ_d)**: Outputs probability that `x` is a real training sample (vs. generated/fake)

**Objective (minimax game):**

```
min_G max_D V(D, G) = E_{x ~ p_data}[log D(x)] + E_{z ~ p_z}[log(1 - D(G(z)))]
```

**Vanilla GAN training loop:**
```
for n iterations:
    // Train discriminator
    for k steps:
        Sample batch {z_i} from noise prior p(z)
        Sample batch {x_i} from real data
        ∇_D = ∇_θ_d [ (1/m) Σ log D(x_i) + (1/m) Σ log(1 - D(G(z_i))) ]
        θ_d ← θ_d + η · ∇_D

    // Train generator
    Sample batch {z_i} from noise prior p(z)
    ∇_G = ∇_θ_g [ (1/m) Σ log(1 - D(G(z_i))) ]
    θ_g ← θ_g - η · ∇_G
```

**The non-saturating loss** (recommended over the original): Instead of minimizing `log(1 - D(G(z)))`, maximize `log(D(G(z)))`. This provides stronger gradients early in training when D easily rejects G's samples.

### 10.2 Theoretical Properties

**Global optimum:** The optimal discriminator for a fixed generator is:
```
D*(x) = p_data(x) / (p_data(x) + p_g(x))
```

Substituting into the value function:
```
V(G, D*) = 2 · JSD(p_data || p_g) - 2·log(2)
```

Where JSD is the **Jensen-Shannon divergence**. Training G is equivalent to minimizing JSD — a proper divergence with minimum 0 when `p_g = p_data`.

**Convergence and mode collapse:**

In theory, GAN training has a unique Nash equilibrium at `p_g = p_data`, D(x) = 1/2. In practice:
- **Mode collapse**: Generator produces limited diversity (only a few modes of the data distribution)
- **Non-convergence**: The minimax optimization may oscillate rather than converge
- **Vanishing gradients**: When D is too strong, G receives near-zero gradient

### 10.3 DCGAN (Radford et al., 2015)

The first widely successful GAN architecture, establishing design guidelines:

**Generator architecture:**
```
z (100-dim noise) → FC → BN → ReLU
→ Transposed Conv 4×4 (512 channels) → BN → ReLU
→ Transposed Conv 4×4 (256 channels) → BN → ReLU
→ Transposed Conv 4×4 (128 channels) → BN → ReLU
→ Transposed Conv 4×4 (3 channels) → Tanh → 64×64 image
```

**Discriminator architecture:** Mirror of generator, using strided convolutions instead of transposed conv, Leaky ReLU (α=0.2), no batch norm in input layer.

**Key guidelines:**
1. Replace any pooling layers with strided convolutions (D) or fractional-strided convolutions (G)
2. Use batch norm in both networks (except G output and D input)
3. Remove fully-connected hidden layers (FC only before first conv in G)
4. Use ReLU in G (all layers except output Tanh)
5. Use LeakyReLU in D (all layers)

### 10.4 StyleGAN (Karras et al., 2019-2022)

**StyleGAN v1 (2019):**

Key innovation: **Style-based generator** that separates high-level attributes from stochastic variation.

**Architecture changes from DCGAN:**
1. **Mapping network f**: 8-layer MLP that maps latent `z` to intermediate latent `w` — the mapping network disentangles the latent space
2. **AdaIN (Adaptive Instance Normalization)** at each layer:
```
AdaIN(x_i, y) = y_{s,i} · (x_i - μ(x_i)) / σ(x_i) + y_{b,i}
```
Where `y_s` (scale) and `y_b` (bias) are derived from w. AdaIN injects style information.
3. **Synthesis network G**: Starts from a learned constant (4×4×512), not from a noise vector
4. **Stochastic variation**: Noise (Gaussian) injected per pixel per layer for fine-grained detail

**Style mixing:** Two different w vectors at different layers -> the coarse layers (4-8²) control pose/identity while fine layers (32-1024²) control colors/textures, demonstrating **disentanglement**.

**StyleGAN v2 (2020):**

Fixes **droplets** and **blob artifacts** in v1.

**Architectural improvements:**
1. **Weight demodulation**: Replace AdaIN with normalization based on the convolution weights:
```
W'_{ijk} = W_{ijk} / sqrt(Σ_k W_{ijk}^2 + ε)
```
This standardizes feature maps without explicit mean/variance computation.
2. **Lazy regularization**: Compute path length regularization every N steps (not every step) to speed training
3. **No progressive growing** — the network is trained at full resolution from scratch using a **residual block** structure
4. **Better conditioning**: Removes unnecessary noise inputs and biases

**StyleGAN v3 (2021):**

Addresses **texture sticking** — the tendency of generated features to be locked to specific pixel coordinates.

**Key innovation:** **Aliasing-free operation** — all layers are designed to respect the continuous signal processing pipeline:
- All up/downsampling uses proper low-pass filters (not nearest-neighbor or bilinear)
- Activation functions are designed to avoid creating aliased high frequencies
- The generator produces truly continuous representations that can be queried at arbitrary coordinates

StyleGAN v3 achieves unprecedented quality at any resolution and eliminates the "positional preference" artifacts of v2.

### 10.5 Conditional GANs (cGANs)

Both generator and discriminator receive a conditioning variable `c` (e.g., class label, text description, image):

```
min_G max_D V(D, G) = E_{x ~ p_data}[log D(x|c)] + E_{z ~ p_z}[log(1 - D(G(z|c)|c))]
```

**Implementation:** Simply concatenate the condition to the input of D and G (or use projection-based conditioning in advanced variants).

### 10.6 Pix2Pix (Isola et al., 2017)

Conditional GAN for **image-to-image translation** with paired data:

- **Generator U-Net**: Skip connections between encoder and decoder preserve low-level details
- **Discriminator PatchGAN**: Classifies N×N patches as real/fake rather than the whole image — enforces local texture/structure realism
- **Combined loss**:
```
L = L_cGAN(G, D) + λ · L_L1(G)
```
Where `L_L1(G) = E_{x,y}[||y - G(x)||_1]` encourages the generator to stay close to the ground truth (L1 encourages less blur than L2).

**U-Net Generator:** Encoder-downsample → bottleneck → decoder-upsample with skip connections from each encoder layer to the corresponding decoder layer.

**PatchGAN discriminator:** Using a small receptive field (e.g., 70×70 patches) forces the model to focus on high-frequency local structure.

### 10.7 CycleGAN (Zhu et al., 2017)

Enables **unpaired image-to-image translation** without paired examples. Uses two generators (G: X→Y, F: Y→X) and two discriminators (D_X, D_Y).

**Cycle consistency loss:**
```
L_cyc(G, F) = E_{x ~ p_data}[||F(G(x)) - x||_1] + E_{y ~ p_data}[||G(F(y)) - y||_1]
```

**Full objective:**
```
L(G, F, D_X, D_Y) = L_GAN(G, D_Y, X, Y) + L_GAN(F, D_X, Y, X) + λ · L_cyc(G, F)
```

**Theoretical guarantee:** Cycle consistency prevents the degenerate solution where G maps all X to the same Y image — the cycle loss forces G and F to be near-inverses.

**Identity loss** (optional): `L_id(G, F) = E_y[||G(y) - y||_1] + E_x[||F(x) - x||_1]` — encourages the generator to preserve color composition when fed target-domain images.

### 10.8 BigGAN (Brock, Donahue, Simonyan, 2019)

Demonstrated high-fidelity class-conditional generation at large scale (up to 256×256 ImageNet).

**Key techniques:**
1. **Spectral normalization** in both G and D: Constrains the Lipschitz constant of each layer for training stability
2. **Shared class embeddings**: Class embeddings shared across generator layers
3. **Skip connections in generator**: For better gradient flow
4. **Orthogonal regularization**: Encourages weight matrices to be orthogonal (optional, not in final model)
5. **Large batch size**: 2048 samples per batch on 512 TPUs
6. **Truncation trick**: At inference, sample z from a truncated normal (sample values beyond 2σ are re-sampled). This trades diversity for fidelity.

**Scaling insights:** BigGAN showed that GANs benefit from scaling — larger models, larger batches, and more compute consistently improve FID.

### 10.9 Diffusion vs GAN Comparison

| Property | GANs | Diffusion Models |
|---|---|---|
| Training | Adversarial (unstable, sensitive to hyperparameters) | Likelihood-based (stable) |
| Sample quality (FID) | Excellent (leading pre-2022) | State-of-the-art (post-2022) |
| Sample diversity | Can suffer mode collapse | Excellent (full distribution coverage) |
| Sample speed | Single forward pass (fast) | ~50-1000 sequential denoising steps (slow) |
| Likelihood estimation | Not available (no tractable likelihood) | Available (continuous-time diffusion) |
| Latent space structure | Can be excellent (StyleGAN) | Linear in diffusion latent space |
| Training compute | Moderate | Higher (longer to converge) |
| Parameter sensitivity | Very high (requires careful tuning) | Moderate |
| Theoretical foundation | Nash equilibrium, JSD minimization | ELBO, score matching, SDE theory |

**When to use GANs:** Latent space manipulation, real-time generation (single forward pass), domain-specific applications where FID is paramount.

**When to use Diffusion:** Maximum fidelity and diversity required, likelihood estimation needed, avoiding training instability.

---

## 11. Variational Autoencoders (VAEs)

### 11.1 Probabilistic Formulation

The VAE (Kingma & Welling, 2014) frames autoencoding as variational inference over a latent variable model.

**Generative model** (decoder): `p_θ(x|z) · p(z)`
- Prior: `p(z) = N(0, I)` (standard Gaussian)
- Likelihood: `p_θ(x|z)` — typically Gaussian (for real-valued x) or Bernoulli (for binary x)

**Inference model** (encoder/recognition): `q_φ(z|x) ≈ p_θ(z|x)`

### 11.2 The Evidence Lower Bound (ELBO)

We maximize the marginal log-likelihood `log p_θ(x)`, which is intractable due to the integral over z. We instead maximize the ELBO:

```
log p_θ(x) = KL(q_φ(z|x) || p_θ(z|x)) + ELBO(θ, φ; x)
```

Since KL ≥ 0:
```
log p_θ(x) ≥ ELBO(θ, φ; x)
```

**ELBO derivation:**

```
ELBO(θ, φ; x) = E_{z ~ q_φ(z|x)}[log p_θ(x|z)] - KL(q_φ(z|x) || p(z))
```

- **Reconstruction term**: `E_{z~q}[log p_θ(x|z)]` — how well does the decoder reconstruct x from z
- **KL regularization term**: `KL(q_φ(z|x) || p(z))` — how close is the approximate posterior to the prior

**Full ELBO (minibatch form):**

```
L(θ, φ; x) = E_{z ~ q_φ(z|x)}[log p_θ(x|z)] - β · KL(q_φ(z|x) || N(0, I))
```

Where β = 1 for standard VAE. (See β-VAE below.)

### 11.3 The Reparameterization Trick

The sampling operation `z ~ q_φ(z|x)` is non-differentiable. The reparameterization trick moves the randomness outside the network:

Instead of: `z ~ N(μ_φ(x), σ_φ²(x))`, we write:
```
ε ~ N(0, I)
z = μ_φ(x) + σ_φ(x) ⊙ ε
```

Now `z` is a deterministic function of `φ` and `ε`, and gradients can flow from the decoder loss back through μ_φ and σ_φ.

**Without reparameterization:** `∇_φ E_z[q]` requires score function estimator (REINFORCE) — high variance.

**With reparameterization:** `∇_φ E_ε[p(x|μ+σ⊙ε)]` is a standard Monte Carlo gradient — low variance.

### 11.4 VAE Loss Computation (Per Example)

```
// Encoder
μ_z, log_σ_z² = encoder_φ(x)
σ_z = exp(0.5 · log_σ_z²)

// Sample z via reparameterization
ε ~ N(0, I)
z = μ_z + σ_z ⊙ ε

// Decoder
ŷ = decoder_θ(z)

// Reconstruction loss (Gaussian decoder, σ²=1)
L_recon = ||x - ŷ||_2² / (2 · σ²) + 0.5 · log(2π · σ²)
        = ||x - ŷ||_2² / 2 + const (for σ²=1)

// For Bernoulli decoder (binary x):
L_recon = -Σ [x_i · log(ŷ_i) + (1-x_i) · log(1-ŷ_i)]

// KL divergence: KL(N(μ_z, σ_z²) || N(0, I))
L_KL = 0.5 · Σ (μ_z_i² + σ_z_i² - 1 - log(σ_z_i²))

// Total loss
L = L_recon + L_KL   // (for β=1)
```

### 11.5 KL Divergence Derivation

For two multivariate Gaussians with diagonal covariance:

```
KL(q || p) = ∫ q(z) · log(q(z)/p(z)) dz

q(z) = N(z; μ, σ²I)
p(z) = N(z; 0, I)

KL = 0.5 · [tr(σ²I) + μ^T μ - k - log|σ²I|]
   = 0.5 · Σ_j (σ_j² + μ_j² - 1 - log σ_j²)
```

Where `k` is the dimensionality of z.

### 11.6 β-VAE (Higgins et al., 2017)

Adds a weight β to the KL term:

```
L = E[log p_θ(x|z)] - β · KL(q_φ(z|x) || p(z))
```

**β > 1** (>1): Stronger constraint on the latent bottleneck. Encourages **disentanglement** — each dimension of z captures an independent factor of variation (e.g., rotation, scale, color in images).

**Tradeoff:** Higher β → more disentangled but worse reconstruction (the "information bottleneck" tradeoff).

**β = 4-8** typically gives good disentanglement on simple datasets (dSprites, 3D Shapes). For complex data, annealing β from 0 to the target value helps.

**Annealed VAE** (Bowman et al., 2016): Gradually increase β from 0 during training (KL annealing). Prevents the model from ignoring the latent variable (a common failure where the decoder learns to be autoregressive and ignores z — "posterior collapse").

### 11.7 VQ-VAE (van den Oord et al., 2017)

Discrete latent variable model that avoids the "posterior collapse" issue of continuous VAEs.

**Architecture:**
1. **Encoder** produces continuous representation `z_e(x)`
2. **Vector quantization** (VQ) maps `z_e` to nearest embedding `e_k` in a learned codebook `{e_1, ..., e_K}`:
```
z_q(x) = e_k where k = argmin_j ||z_e(x) - e_j||_2
```
3. **Decoder** reconstructs x from `z_q`

**Loss function:**
```
L = ||x - decoder(z_q)||_2²       // reconstruction
  + ||sg[z_e(x)] - e||_2²          // codebook loss (move embeddings toward encoder)
  + β · ||z_e(x) - sg[e]||_2²      // commitment loss (move encoder toward embeddings)
```

Where `sg[·]` is the stop-gradient operator. β is typically 0.25.

**Straight-through gradient estimator:** The gradient of the VQ operation (argmin is non-differentiable) is approximated by copying gradients from `z_q` directly to `z_e`. This works because `z_e` and `z_q` are close.

**VQ-VAE-2** (Razavi et al., 2019): Hierarchical VQ-VAE with multiple latent scales (e.g., top-level captures global structure, bottom-level captures local texture). Produces high-fidelity 1024×1024 images.

**Practical use:** VQ-VAE is the backbone of many modern generative models:
- DALL-E (discrete VAE tokenizer for text-to-image)
- Image GPT (VQ-VAE tokens → autoregressive Transformer)
- Parti (ViT-VQGAN)

### 11.8 VAE vs GAN vs Diffusion

| Property | VAE | GAN | Diffusion |
|---|---|---|---|
| Likelihood | ELBO (lower bound) | Not available | Exact (continuous) |
| Sample quality | Often blurry | Sharp (can be excellent) | State-of-the-art |
| Mode coverage | Excellent (full distribution) | Poor (mode collapse) | Excellent |
| Training stability | Stable | Unstable | Stable |
| Latent space | Well-structured | Variable | Well-structured |
| Computational cost | Low (single pass) | Low (single pass) | High (many steps) |

---

## 12. Normalizing Flows

### 12.1 The Change of Variables Formula

Normalizing flows model a complex distribution `p_X(x)` by applying an invertible transformation `f` to a simple base distribution `p_Z(z)` (typically N(0, I)):

```
x = f(z) where f is invertible, z = f^{-1}(x)
```

The density is given by the change of variables:

```
p_X(x) = p_Z(f^{-1}(x)) · |det(J_f⁻¹(x))|
       = p_Z(z) · |det(J_f(z))|^{-1}
```

Where `J_f(z)` is the Jacobian of f at z.

**The key constraint:** The Jacobian determinant must be tractable. This has driven flow architecture design.

### 12.2 RealNVP (Dinh et al., 2017)

RealNVP uses **affine coupling layers** — a clever choice that makes the Jacobian triangular.

**Affine coupling layer:**

Given input `u ∈ ℝ^D`, split into two parts: `u_1 = u_{1:d}`, `u_2 = u_{d+1:D}`.

```
v_1 = u_1
s, t = NN(u_1)    // scale and translation from neural network
v_2 = u_2 ⊙ exp(s) + t

// Output: v = [v_1; v_2]
```

**Inverse:**
```
u_1 = v_1
s, t = NN(v_1)
u_2 = (v_2 - t) ⊙ exp(-s)
```

**Jacobian determinant:**

The Jacobian of the coupling layer is lower triangular:
```
J = [ I         0   ]
    [ ∂v_2/∂u_1  diag(exp(s)) ]

det(J) = Π_i exp(s_i) = exp(Σ_i s_i)
```

This is trivially computable without evaluating the full Jacobian.

**Masking:** Alternate between splitting patterns (checkerboard vs channel-wise) across layers to ensure all dimensions interact.

**Multi-scale architecture:** Use squeeze operations to trade spatial resolution for channels, then apply coupling layers at each scale.

### 12.3 Glow (Kingma & Dhariwal, 2018)

Extends RealNVP with:

1. **Actnorm** (Activation Normalization): A learnable per-channel scale and shift, initialized such that the first minibatch has zero mean and unit variance. Faster than batch norm and works with invertibility.

2. **Invertible 1×1 convolution**: A learned permutation of channels. The determinant is `det(W)` where W is the D×D weight matrix. This is computed via LU decomposition of W for O(D) cost instead of O(D³).

3. **Affine coupling layers** (same as RealNVP)

Glow architecture:
```
x → ActNorm → Invertible 1×1 Conv → AffineCoupling → Squeeze → [repeat]
```

**Log-likelihood computation for a batch:**
```
log p(x) = log p(z) + Σ_log_det(J_l)
         = -0.5 · Σ z_i² - 0.5 · D · log(2π) + Σ_l log|det(J_l)|
```

### 12.4 Flow-Based Generative Model Training

```
ALGORITHM: Flow Training (Maximum Likelihood)
─────────────────────────────────────────────
for each batch {x_i}:
    // Forward: x → z
    z, log_det_sum = f^{-1}(x)  // invertible transformation
    // z_i should be ~ N(0, I) if model captures data

    // Log-likelihood
    log_p_x = log_p_z(z) + log_det_sum
    // log_p_z(z) = -0.5 * Σ z_i² - 0.5 * D * log(2π)

    Loss = -mean(log_p_x)  // negative log-likelihood

    θ ← θ - η · ∇_θ Loss  // standard gradient descent
```

**Sampling:**
```
z ~ N(0, I)
x = f(z)  // generate new data
```

### 12.5 Flow Limitations

- **Volume preservation**: Each transformation must be invertible with tractable determinant, constraining expressivity
- **Computational cost**: Forward and inverse passes are both needed, each roughly 2× the cost of a standard forward pass
- **Dimensionality constraint**: Flow must transform to same-dimensional latent space (no dimensionality reduction/fixed compression)
- **Less widely adopted** than VAEs or diffusion for large-scale generation, though flows remain important in:
  - **Variational inference**: As flexible posterior approximations for VAEs
  - **Density estimation**: For low-dimensional data with known structure
  - **Audio generation**: WaveGlow, WaveFlow

### 12.6 Residual Flows

An alternative to coupling-based flows using invertible residual networks (i-ResNet, Behrmann et al., 2019):

```
z → z + g(z) → z'
```

With Lipschitz constraint on g (via spectral normalization), the inverse can be approximated via fixed-point iteration. The log-determinant is estimated via a **stochastic** trace estimator (Hutchinson's trace) using the **Neumann series** or **Russian roulette** estimator.

---

## 13. Diffusion Models

### 13.1 Denoising Diffusion Probabilistic Models (DDPM)

Ho, Jain, and Abbeel (2020) formalized diffusion models as a parameterized Markov chain that gradually adds noise to data (forward process) then learns to reverse it (reverse process).

**Forward process (fixed, no learnable parameters):**

For each step t = 1, ..., T (typically T = 1000):

```
q(x_t | x_{t-1}) = N(x_t; sqrt(1 - β_t) · x_{t-1}, β_t · I)
```

Where `β_t ∈ (0,1)` is the noise schedule (typically linear: `β_1=1e-4` to `β_T=0.02`).

**Reparameterization:** We can sample `x_t` directly from `x_0` in closed form:

Let `α_t = 1 - β_t`, `\bar{α}_t = Π_{s=1}^{t} α_s`.

```
q(x_t | x_0) = N(x_t; sqrt(\bar{α}_t) · x_0, (1 - \bar{α}_t) · I)

x_t = sqrt(\bar{α}_t) · x_0 + sqrt(1 - \bar{α}_t) · ε,  where ε ~ N(0, I)
```

As t → T, `\bar{α}_T ≈ 0`, so `x_T ≈ N(0, I)` — the data is fully diffused to Gaussian noise.

**Reverse process (learnable):**

The reverse transition `q(x_{t-1} | x_t)` is intractable (depends on the unknown data distribution). We approximate it with a neural network:

```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t² · I)
```

Where `σ_t² = β_t` (or `\tilde{β}_t = (1 - \bar{α}_{t-1}) / (1 - \bar{α}_t) · β_t`) is fixed.

**Training objective (simplified):**

The model predicts the noise `ε` that was added:

```
L_simple(θ) = E_{t, x_0, ε} [||ε - ε_θ(sqrt(\bar{α}_t) · x_0 + sqrt(1 - \bar{α}_t) · ε, t)||²]
```

Where `ε_θ(x_t, t)` is the denoising U-Net that predicts the noise given a noisy image and timestep.

**Training pseudocode:**
```
ALGORITHM: DDPM Training
────────────────────────
repeat until convergence:
    x_0 ~ p_data(x)                // sample data
    t ~ Uniform({1, ..., T})       // random timestep
    ε ~ N(0, I)                    // random noise
    
    x_t = sqrt(\bar{α}_t) · x_0 + sqrt(1 - \bar{α}_t) · ε
    // Predict noise and compute loss
    L = ||ε - ε_θ(x_t, t)||²
    
    θ ← θ - η · ∇_θ L
```

**Sampling pseudocode:**
```
ALGORITHM: DDPM Sampling
────────────────────────
x_T ~ N(0, I)                      // start from pure noise

for t = T down to 1:
    z ~ N(0, I) if t > 1 else 0   // random noise (not for last step)
    
    // Predict noise
    ε_pred = ε_θ(x_t, t)
    
    // Compute x_{t-1}
    x_{t-1} = (1 / sqrt(α_t)) · (x_t - (1-α_t) / sqrt(1-\bar{α}_t) · ε_pred)
            + σ_t · z
```

### 13.2 Connection to Score Matching

DDPM can be interpreted through the lens of **score-based generative modeling** (Song & Ermon, 2019). The score function is the gradient of the log-density:

```
s_θ(x, σ) = ∇_x log p_θ(x; σ)
```

For DDPM, the optimal noise prediction satisfies:

```
ε_θ(x_t, t) = -√(1 - \bar{α}_t) · ∇_{x_t} log p(x_t)
```

The score function `∇_{x_t} log p(x_t)` tells us which direction to move to increase the data density — the reverse of adding noise.

### 13.3 Noise Scheduling

**Linear schedule (DDPM):** `β_t` increases linearly from `β_1 = 1e-4` to `β_T = 0.02`. Simple but wastes capacity on fine details early and adds too much noise late.

**Cosine schedule** (Improved DDPM, Nichol & Dhariwal, 2021):

```
\bar{α}_t = cos²(π/2 · (t/T + s) / (1 + s)) / cos²(π/2 · s / (1 + s))
```

Where `s = 0.008`. The cosine schedule avoids sudden drops in noise level, providing a smoother signal-to-noise ratio.

**Sigmoid schedule:** Uses a sigmoid curve — more flexible than linear.

**Optimal schedule:** Varies by data type and resolution. For images, cosine generally outperforms linear. For video/audio, other schedules may be better.

### 13.4 DDIM: Denoising Diffusion Implicit Models

Song, Meng, and Ermon (2021) proposed DDIM, which makes the sampling process **deterministic** and enables **fewer sampling steps**.

**Key insight:** The forward process `q(x_t | x_{t-1})` can be made non-Markovian while preserving the marginal `q(x_t | x_0)`. This allows using a **deterministic** reverse process:

```
x_{t-1} = √(\bar{α}_{t-1}) · (x_t - √(1 - \bar{α}_t) · ε_θ(x_t, t)) / √(\bar{α}_t)
        + √(1 - \bar{α}_{t-1}) · ε_θ(x_t, t)
```

**Properties:**
- **Deterministic** mapping from x_T to x_0 — can interpolate in latent space
- **Fewer steps**: Use a subsequence `{τ_1, ..., τ_S}` of timesteps (e.g., S=50 instead of T=1000)
- **Consistency**: Coarse-to-fine generation in latent space
- **DDPM as special case**: Adding noise to the deterministic DDIM trajectory recovers DDPM sampling

**DDIM sampling schedule:** Typically uses stride `(T-1)/(S-1)` to select timesteps. With S=50, quality is comparable to DDPM with T=1000.

### 13.5 Classifier-Free Guidance (CFG)

Ho & Salimans (2022) proposed CFG to trade off diversity for fidelity without an explicit classifier.

**Training:** Jointly train conditional and unconditional models. With probability `p_uncond = 0.1-0.2`, drop the condition `c` (replace with null token).

**Sampling:** Combine conditional and unconditional predictions:

```
ε_θ^guided(x_t, c) = ε_θ(x_t, c) + w · (ε_θ(x_t, c) - ε_θ(x_t, ∅))
```

Where `w ≥ 0` is the guidance scale:
- `w = 0`: Unconditional generation (maximum diversity)
- `w = 1`: Standard conditional generation
- `w = 7.5`: Strong guidance (higher fidelity, lower diversity)

**Why it works:** The difference `(ε_θ(x_t, c) - ε_θ(x_t, ∅))` points in the direction of increased conditional likelihood. Scaling this difference amplifies the effect.

**Practical tip:** CFG is highly sensitive to the guidance scale. For text-to-image, w=7-8 is typical for photorealistic results; w=3-5 for artistic or creative outputs. Too high (w>15) causes saturation and artifacts.

### 13.6 Stable Diffusion Architecture

Rombach et al. (2022) introduced **latent diffusion**, performing the diffusion process in a lower-dimensional latent space rather than pixel space.

**Components:**

1. **VAE (Variational Autoencoder)**: Compresses 256×256×3 images to 64×64×4 latent codes (8× compression). The encoder maps x → z, and the decoder maps z → reconstructed x.

2. **U-Net**: A time-conditional U-Net that processes the latent. Key components:
   - **ResNet blocks** with group normalization
   - **Spatial self-attention** (at resolutions 64, 32, 16, 8)
   - **Cross-attention** to inject text conditioning: Q from U-Net latent, K/V from text encoder output
   - **Time embedding** via sinusoidal positional encoding

3. **Text encoder**: CLIP text encoder (ViT-L/14) — 12-layer Transformer, 768-dim embeddings, 77 token max length.

4. **Sampling schedule**: Typically 50 DDIM steps with CFG (w=7.5) for good quality.

**Training pipeline:**
1. Train VAE (encoder + decoder) on pixel data with KL regularization (β=0.00001)
2. Fix VAE encoder → encode all training images to latents
3. Train U-Net with DDPM loss on latents, conditioned on text embeddings

**SD model sizes:**
```
SD 1.x: 860M U-Net + 34M VAE + 123M CLIP = ~1.0B param
SD 2.x: 865M U-Net (larger) but similar total
SD XL: ~2.6B UNet + VAE + text encoders (two CLIP models)
SD 3 / SD3.5: 8B DiT (Diffusion Transformer) architecture
SD 4: Likely LDM + flow matching variants
```

**Key advantage:** Operating in latent space reduces computational cost by ~4× compared to pixel-space diffusion at equivalent quality.

### 13.7 Flow Matching and Rectified Flow

**Flow Matching** (Lipman et al., 2023) provides a simpler, more general framework for generative modeling through probability paths.

**Core idea:** Learn a time-dependent vector field `v_θ(x_t, t)` that generates a probability path between the noise distribution (t=0) and the data distribution (t=1).

**Objective:**
```
L = E_{t, x_0, x_1} [||v_θ(x_t, t) - (x_1 - x_0)||²]
```

Where `x_t = (1-t) · x_0 + t · x_1` is a linear interpolation between noise `x_0` and data `x_1`.

**Rectified Flow** (Liu et al., 2023) takes this further by **straightening** the probability paths through a **reflow** procedure:

1. **Initial flow**: Train a flow from noise to data using the standard objective
2. **Reflow**: Sample from the initial flow to create paired (noise, sample) data. Train a new flow on these paired data.
3. **Result**: The reflow procedure straightens the trajectories, enabling **one-step generation** with minimal quality loss after enough reflow steps.

**Advantages over DDPM:**
- Simpler objective (no noise schedule, no VLB)
- Can achieve high quality with fewer sampling steps
- Natural support for ODE-based sampling and latent interpolation
- Rectified flow enables 1-2 step generation similar to GAN speed

**Stable Diffusion 3** uses rectified flow matching as its core generative framework, replacing the DDPM noise-prediction objective.

### 13.8 Diffusion Model Practical Guide

**Architecture choices:**
- **Resolution**: Latent diffusion (8× compressed) for >256px; pixel-space for <128px
- **Backbone**: U-Net with attention for ≤1B param; DiT (Diffusion Transformer) for larger models
- **Conditioning**: Cross-attention for text, adaptive layer norm for class labels, concatenation for inpainting

**Training tips:**
- EMA (exponential moving average) of model weights significantly improves sample quality
- Mixed precision training (fp16) works well
- Gradient checkpointing to save memory in attention layers
- Use v_prediction (Salimans & Ho, 2022) for improved high-resolution generation:
  ```
  v_t = α_t · ε - σ_t · x_0  // target = "velocity"
  L = ||v_t - v_θ(x_t, t)||²
  ```
- Loss weighting: L_simple works; some approaches use SNR+1 weighting

**Sampling improvements:**
- **DDIM** with 50 steps vs 1000 for DDPM
- **DPM-Solver** (Lu et al., 2022): 10-20 step sampling via solving the diffusion ODE
- **DPMSolver++**: 10 steps for high-quality results
- **LCM (Latent Consistency Models)** (Luo et al., 2023): Distill diffusion into 1-4 step generators
- **FreeU** (Si et al., 2023): Adjust U-Net skip connection contributions for better quality without retraining

**Common failure modes:**
- **Color shifting**: Due to accumulated error in long sampling chains. Fix: Use DDIM or fewer steps.
- **Oversaturation**: CFG too high. Fix: Reduce w.
- **Mode dropping**: Training data bias. Fix: More diverse training data or data balancing.
- **Incoherent structures**: U-Net capacity insufficient. Fix: Larger model or more attention layers.

---

## 14. Further Reading and References

### Foundational Papers

1. Rosenblatt, F. (1958). "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain."
2. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). "Learning representations by back-propagating errors." Nature.
3. LeCun, Y. et al. (1998). "Gradient-based learning applied to document recognition." Proceedings of the IEEE.
4. Hochreiter, S. & Schmidhuber, J. (1997). "Long Short-Term Memory." Neural Computation.
5. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). "ImageNet Classification with Deep Convolutional Neural Networks." NeurIPS.
6. Goodfellow, I. et al. (2014). "Generative Adversarial Nets." NeurIPS.
7. Kingma, D. P. & Welling, M. (2014). "Auto-Encoding Variational Bayes." ICLR.
8. Vaswani, A. et al. (2017). "Attention Is All You Need." NeurIPS.
9. Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models." NeurIPS.
10. Rombach, R. et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models." CVPR.

### Modern Reference Texts

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- Prince, S. J. D. (2023). *Understanding Deep Learning*. MIT Press (free online).
- Bishop, C. M. & Bishop, H. (2024). *Deep Learning: Foundations and Concepts*. Springer.
- Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press.
- Murphy, K. P. (2023). *Probabilistic Machine Learning: Advanced Topics*. MIT Press.

---

*This document is part of the AI Base Knowledge series. For related topics, see 01-LLM-and-AI-Models.md (Transformers, attention, LLM architecture) and 02-Machine-Learning.md (foundational ML concepts).*
