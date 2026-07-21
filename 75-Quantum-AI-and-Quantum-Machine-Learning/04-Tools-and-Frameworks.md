# 04 | Quantum AI — Tools and Frameworks

> **The quantum ML software stack is maturing rapidly — from high-level quantum ML libraries to full-stack quantum cloud platforms.**

---

## 1. Quantum ML Libraries

### PennyLane (Xanadu)
**The most comprehensive QML framework** — seamlessly integrates ML with quantum computing.

- **Key feature**: Automatic differentiation of quantum circuits via the parameter-shift rule
- **Supported hardware**: IBM, Google, Amazon Braket, IonQ, Rigetti, Honeywell, Xanadu
- **ML integration**: Native interop with PyTorch, TensorFlow, JAX
- **Unique capability**: Quantum differentiable programming — backpropagate through quantum circuits
- **2026 update**: PennyLane 0.38 adds dynamic circuit compilation, kernel-based learning toolkit, and RL-based quantum control

```python
import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev, diff_method="parameter-shift")
def quantum_circuit(x, weights):
    qml.AngleEmbedding(x, wires=[0, 1])
    qml.BasicEntanglerLayers(weights, wires=[0, 1])
    return qml.expval(qml.PauliZ(0))

# Train with standard ML tools
weights = np.random.random((3, 2))
result = quantum_circuit([0.5, 0.3], weights)
```

### Qiskit (IBM)
**The most mature full-stack quantum computing SDK** with a dedicated ML module.

- **Qiskit Machine Learning**: QNNs, QSVMs, QGANs, QKernels
- **Qiskit Nature**: Quantum chemistry + ML for drug discovery
- **Qiskit Runtime**: Run hybrid classical-quantum workflows on IBM hardware
- **Circuit Knitting Toolbox**: Distribute large quantum circuits across smaller devices
- **2026 update**: IBM Quantum Network expanded to 200+ organizations; Heron processor (1,121 qubits) available via cloud

### TensorFlow Quantum (Google)
Google's quantum ML library within the TensorFlow ecosystem.

- **Design philosophy**: Quantum ML as a TensorFlow layer
- **Batch execution**: Efficient simulation of many quantum circuits for training
- **Keras integration**: Build quantum models with `tf.keras.Sequential`
- **Cirq backend**: Google's quantum circuit framework
- **2026 update**: Integration with Willow processor; real-time quantum error correction for ML workloads

### Cirq (Google)
The circuit-level framework underlying TensorFlow Quantum.

- **NISQ-focused**: Designed for noisy intermediate-scale quantum devices
- **Noise models**: Native support for device-agnostic noise simulation
- **Optimization**: Circuit optimization, compilation, and scheduling for specific hardware

### Braket (Amazon)
AWS's managed quantum computing service.

- **Multi-backend**: Access to IonQ, Rigetti, D-Wave, and simulators from one API
- **Hybrid jobs**: Run classical-quantum hybrid training loops as managed jobs
- **PennyLane integration**: First-class PennyLane support

---

## 2. Quantum-Inspired ML Libraries

### quimb
Tensor network library for quantum-inspired ML — uses matrix product states (MPS) and tensor trains for ML:

```python
import quimb.tensor as qtn

# Represent a dataset as an MPS (efficient for 1D structured data)
mps = qtn.MPS_rand_state(100, bond_dim=16)
# Train as a classifier — exponentially fewer parameters than equivalent NN
```

### TeNPy
Tensor network Python library for many-body physics, now used for ML:
- Matrix product state classifiers
- Density matrix renormalization group (DMRG) for ML optimization
- Entanglement-based feature selection

### torchsimm
Quantum-inspired similarity learning library — uses tensor network contractions for efficient kernel computation.

---

## 3. Quantum Cloud Platforms

| Platform | Hardware | Cost Model | Best For |
|:---------|:---------|:-----------|:---------|
| **IBM Quantum Network** | IBM Heron (1,121q) | Pay-per-circuit minute | Enterprise QML |
| **Google Quantum AI** | Willow (105q) | Research access via grant | Cutting-edge QML research |
| **Amazon Braket** | IonQ, Rigetti, D-Wave | Pay-per-task | Multi-backend development |
| **Azure Quantum** | Quantinuum, IonQ | Pay-per-shot | Integration with Azure ML |
| **Xanadu Cloud** | Photonic (X8, X16) | Pay-per-circuit | Photonic QML |
| **D-Wave Leap** | Annealing 5,000+q | Time-based subscription | Optimization QML |
| **PennyLane Cloud** | Multi-backend | Circuit-based | QML development + training |

---

## 4. Quantum AI Development Workflow

### Typical QML Pipeline
```
1. Data preparation (classical)
   └── Normalize, PCA, feature selection

2. Encoding circuit design
   └── Amplitude / angle / basis encoding

3. Variational circuit design
   └── Hardware-efficient ansatz / problem-specific

4. Training loop
   └── Parameter shift → classical optimizer → repeat

5. Measurement & post-processing
   └── Shot averaging, noise mitigation

6. Deployment
   └── Hybrid cloud (QPU + classical + ML pipeline)
```

### Noise Mitigation Techniques
- **Zero-noise extrapolation**: Run at multiple noise levels, extrapolate to zero-noise
- **Probabilistic error cancellation**: Learn noise model, invert it statistically
- **Readout error mitigation**: Calibrate measurement errors, correct in post-processing
- **Dynamical decoupling**: Pulse sequences that suppress decoherence
- **Error suppression by unitary folding**: Replace gates with identity-equivalent longer sequences

---

## 5. Quantum Datasets and Benchmarks

| Dataset | Task | Qubits | Source |
|:--------|:-----|:-------|:-------|
| **QSVM Benchmark** | Classification on quantum feature maps | 4–12 | IBM Qiskit |
| **Quantum Circuit Learning** | Regression on simulated quantum data | 2–8 | PennyLane |
| **Tensor Network MNIST** | Quantum-inspired classification | 28×28 MPS | Google Research |
| **QNLP (DisCoCat)** | Sentiment, semantic similarity | 4–8 | Cambridge Quantum |
| **Quantum Chemistry (QM9)** | Molecular property prediction | 4–16 | IBM + Qiskit Nature |
| **Quantum Annealing Bench** | Combinatorial optimization | up to 5,000 | D-Wave |

---

## 6. Key 2026 Tooling Developments

- **Real-time quantum error correction** integrated into Google's software stack (Cirq + RL agent)
- **IBM's Qiskit Patterns** — reusable quantum ML workflows with classical pre/post-processing
- **NVIDIA cuQuantum + PennyLane integration** — GPU-accelerated quantum circuit simulation for QML training
- **Classiq + UC Chile** Quantum AI research collaboration with dedicated software stack
- **D-Wave's hybrid solver service (HSS)** — automatic quantum-classical partitioning for ML optimization workloads
- **AWS Braket Hybrid Jobs** — fully managed hybrid QML training with auto-scaling
