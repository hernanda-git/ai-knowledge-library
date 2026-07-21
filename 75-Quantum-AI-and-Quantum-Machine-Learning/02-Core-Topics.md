# 02 | Quantum AI — Core Technical Topics

> **From qubits to quantum gradients — the technical foundations every AI practitioner should understand about quantum machine learning.**

---

## 1. Qubits, Superposition, and Entanglement

### Qubit Fundamentals
A qubit is a two-level quantum system. Unlike a classical bit, it can exist in superposition:

```
|ψ⟩ = α|0⟩ + β|1⟩

where |α|² + |β|² = 1
```

This means a single qubit carries two amplitude dimensions. **N qubits carry 2^N amplitude dimensions** — exponentials without the hardware.

### Entanglement — The Secret Sauce
Entanglement is the quantum phenomenon where qubits become correlated such that measuring one instantaneously determines the state of others, regardless of distance. For ML, entanglement enables:
- Representing complex probability distributions that classical models require exponentially more parameters to express
- Quantum kernels that map data into exponentially large feature spaces
- Quantum neural networks with exponentially expressive hidden layers

### Measurement Collapse
When you measure a qubit, superposition collapses to a classical 0 or 1. This means quantum ML algorithms must be **probabilistic** — run many times (shots) and aggregate results. The "readout problem" is one of quantum ML's biggest engineering challenges.

---

## 2. Quantum Machine Learning Paradigms

### Variational Quantum Algorithms (VQAs)
The most practical near-term approach. A parameterized quantum circuit (PQC) — analogous to a neural network — is trained by a classical optimizer:

```
Classical Optimizer ←→ Quantum Circuit
       ↑                       ↓
   Loss Function         Measurement
```

**Key VQA types:**
- **Variational Quantum Eigensolver (VQE)**: Find ground-state energies (quantum chemistry)
- **Quantum Approximate Optimization Algorithm (QAOA)**: Combinatorial optimization
- **Variational Quantum Classifier (VQC)**: Classification with PQC
- **Quantum Circuit Born Machine (QCBM)**: Generative modeling

### Quantum Kernel Methods
Map classical data into a quantum feature space using a quantum circuit, then run a classical SVM on top. The quantum kernel implicitly computes inner products in an exponentially large Hilbert space:

```
K(xᵢ, xⱼ) = |⟨0|U†(xᵢ)U(xⱼ)|0⟩|²
```

**2026 breakthrough**: Exponential quantum advantage proven for kernel-based ML tasks on real quantum hardware (Nature, April 2026). This is the first provable exponential advantage for a practically useful ML task.

### Quantum Neural Networks (QNNs)
Layered parameterized quantum circuits analogous to classical neural networks. Key differences:

| Aspect | Classical NN | QNN |
|:-------|:-------------|:----|
| Neuron | Weighted sum + activation | Unitary gate + measurement |
| Layer | Linear + nonlinear activation | Quantum circuit layer |
| Training | Backpropagation | Parameter shift rule (gradient estimation) |
| Expressivity | Bounded by layer width | Potentially exponential in depth |

### Quantum Natural Language Processing (QNLP)
The pioneering approach uses **DisCoCat** (Distributional Compositional Categorical) models mapped to quantum circuits. Words become qubits, grammatical structure becomes quantum entanglement, and sentence meaning emerges from measurement.

**Key results**: QNLP models on IBM Quantum hardware have demonstrated the ability to classify sentiment and detect semantic similarity using orders of magnitude fewer parameters than classical transformers.

### Quantum Generative Models
- **Quantum GANs (QGANs)**: Generator and discriminator both quantum or hybrid
- **Quantum Boltzmann Machines (QBMs)**: Use quantum thermal distributions for sampling
- **Quantum Circuit Born Machines**: Pure quantum generative models based on measurement probability distributions

---

## 3. Quantum Optimization for AI Training

Training large AI models is fundamentally an optimization problem. Quantum computers can potentially solve optimization landscapes that trap classical optimizers:

- **Quantum Annealing**: D-Wave systems excel at finding global minima in rugged loss landscapes
- **QAOA**: The quantum alternative to gradient descent for combinatorial loss functions
- **Amplitude Amplification**: Quadratic speedup for sampling-based training methods

---

## 4. Hybrid Quantum-Classical Architectures

### The NISQ Paradigm
Current quantum computers are **Noisy Intermediate-Scale Quantum (NISQ)** devices. They have:
- 50–1,000+ physical qubits
- Limited coherence times
- Gate errors (0.1%–1% per gate)
- No full error correction

The practical consequence: **hybrid architectures** where quantum processors accelerate specific subroutines while classical computers handle everything else.

### Hybrid Workflow
```
Classical (CPU/GPU)        Quantum (QPU)
    │                         │
    ├── Data encoding ───────►│
    │                         ├── Quantum circuit execution
    │◄────── Measurement ─────┤
    ├── Loss calculation      │
    ├── Parameter update ────►│ (next iteration)
    │                         │
    └── Orchestration ────────┘
```

### When to Go Hybrid
- **Data loading/encoding** → classical (quantum RAM is not yet practical)
- **Feature mapping** → quantum (exponential feature space)
- **Inference** → quantum (fast for specific tasks)
- **Training loop** → classical (gradient calculation is hybrid)
- **Post-processing** → classical (measurement statistics)

---

## 5. Quantum Error Correction for ML

The biggest barrier to practical quantum ML is noise. 2026 saw a breakthrough:

### Google's RL-Based Quantum Error Correction
Google Quantum AI used reinforcement learning to control real-time quantum error correction (Nature, July 2026). An RL agent learns optimal correction sequences, reducing logical error rates below physical error rates — the threshold for fault-tolerant quantum computing.

### Impact on QML
Fault-tolerant quantum computing will enable:
- Deep quantum circuits without noise accumulation
- Unbounded quantum feature maps
- Reliable quantum gradient estimation
- Practical quantum advantage for ML at scale

**Timeline**: Google projects fault-tolerant quantum ML by 2029–2030. IBM's roadmap targets 2033.

---

## 6. Quantum Data Encoding

Classical data must be encoded into quantum states — this is the "input bottleneck" of QML.

| Encoding | Qubits Needed | Description |
|:---------|:--------------|:------------|
| Basis encoding | N (for N-bit data) | Each bit → one qubit |
| Amplitude encoding | log₂(N) | N classical values → N amplitudes of log₂(N) qubits |
| Angle encoding | N | Each feature → rotation angle of one qubit |
| Hamiltonian encoding | varies | Data encoded in Hamiltonian parameters |
| QRAM encoding | complex | Quantum random access memory (theoretical) |

**Amplitude encoding** is the most efficient (exponential compression) but requires exponentially precise state preparation — a major engineering challenge.

---

## 7. The Barren Plateau Problem

A critical issue in training QNNs: as circuit depth or number of qubits increases, gradients vanish exponentially — the "barren plateau." Solutions being actively researched:

- **Layerwise training**: Train shallow circuits sequentially
- **Problem-informed ansätze**: Use domain knowledge to constrain circuit structure
- **Correlated parameter initialization**: Avoid random initialization
- **Classical pre-training**: Warm-start with classical models
- **Quantum natural gradient**: Use quantum Fisher information for optimization
