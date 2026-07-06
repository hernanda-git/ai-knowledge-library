# Tools and Frameworks for Edge AI and On-Device Inference

> This document provides a comprehensive overview of the tools, frameworks, and platforms available for developing, optimizing, and deploying AI models on edge devices.

## Table of Contents

- [Inference Frameworks](#inference-frameworks)
- [Model Optimization Tools](#model-optimization-tools)
- [Hardware-Specific SDKs](#hardware-specific-sdks)
- [Edge AI Platforms](#edge-ai-platforms)
- [Development Environments](#development-environments)
- [Benchmarking Tools](#benchmarking-tools)
- [Deployment and Management](#deployment-and-management)
- [Open Source Projects](#open-source-projects)

---

## Inference Frameworks

### TensorFlow Lite

**Overview**: Google's framework for on-device ML inference

| Feature | Details |
|---------|---------|
| **Platforms** | Android, iOS, Linux, MCU, Edge TPU |
| **Model Formats** | .tflite (FlatBuffers) |
| **Quantization** | INT8, FP16, Dynamic |
| **Hardware Support** | CPU, GPU, Edge TPU, NNAPI |
| **Language** | C++, Python, Java, Swift |

```bash
# Installation
pip install tflite-runtime

# Convert model
tflite_convert --saved_model_dir=model --output_file=model.tflite

# Run inference
python -c "
import tflite_runtime.interpreter as tflite
import numpy as np

interpreter = tflite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_data = np.random.random(input_details[0]['shape']).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
print('Output shape:', output.shape)
"
```

### ONNX Runtime

**Overview**: Cross-platform inference engine from Microsoft

| Feature | Details |
|---------|---------|
| **Platforms** | Windows, Linux, macOS, Android, iOS, Edge |
| **Model Formats** | .onnx |
| **Quantization** | INT8, INT4, Mixed Precision |
| **Hardware Support** | CPU, GPU, TensorRT, OpenVINO, CoreML |
| **Language** | C++, Python, C#, Java |

```bash
# Installation
pip install onnxruntime onnxruntime-mobile

# Optimize model
python -m onnxruntime.transformers.optimizer \
    --input model.onnx \
    --output model_optimized.onnx \
    --float16 \
    --disable_gelu \
    --disable_layer_norm \
    --disable_attention

# Run inference
python -c "
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('model.onnx')
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
result = session.run([output_name], {input_name: input_data})
print('Output shape:', result[0].shape)
"
```

### PyTorch Mobile

**Overview**: PyTorch's framework for mobile deployment

| Feature | Details |
|---------|---------|
| **Platforms** | Android, iOS, Linux, Windows |
| **Model Formats** | .ptl (Lite Interpreter) |
| **Quantization** | Dynamic, Static, QAT |
| **Hardware Support** | CPU, GPU (Metal/Vulkan) |
| **Language** | Python, C++, Java, Swift |

```python
import torch

# Load and optimize for mobile
model = torch.load("model.pth")
model.eval()

# Trace for mobile
example = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example)

# Optimize for mobile
optimized = torch.utils.mobile_optimizer.optimize_for_mobile(traced_model)

# Export for mobile
optimized._save_for_lite_interpreter("model_mobile.ptl")

# Load on mobile
# interpreter = torch.jit.load("model_mobile.ptl")
```

### Apache TVM

**Overview**: Deep learning compiler for diverse hardware targets

| Feature | Details |
|---------|---------|
| **Platforms** | Cross-platform (compiles to target) |
| **Model Formats** | ONNX, TensorFlow, PyTorch, DarkNet |
| **Quantization** | Auto-tuning, mixed precision |
| **Hardware Support** | CPU, GPU, FPGA, ASIC, MCU |
| **Language** | Python, C++ |

```python
import tvm
from tvm import relay

# Load model
model = load_model("model.onnx")

# Compile for specific target
target = tvm.target.Target("llvm -mcpu=cortex-a78")

# Auto-tune
from tvm.autotvm import autotvm
from tvm.autotvm.tuner import XGBTuner

tuner = XGBTuner(tasks)
tuner.tune(
    n_trial=200,
    measure_option=autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner()
    )
)

# Build
lib = relay.build(model, target=target, params=params)

# Run
ctx = tvm.cpu()
module = tvm.contrib.graph_executor.GraphModule(lib["default"](ctx))
module.set_input("input", input_tvm)
module.run()
```

### NCNN

**Overview**: High-performance neural network inference framework for mobile

| Feature | Details |
|---------|---------|
| **Platforms** | Android, iOS, Linux, Windows, macOS |
| **Model Formats** | .param + .bin |
| **Quantization** | INT8 |
| **Hardware Support** | CPU (ARM NEON, x86 SSE/AVX), Vulkan GPU |
| **Language** | C++ |

```cpp
#include <ncnn/net.h>

// Load model
ncnn::Net net;
net.opt.use_vulkan_compute = true;  // GPU acceleration
net.load_param("model.param");
net.load_model("model.bin");

// Create extractor
ncnn::Extractor ex = net.create_extractor();

// Set input
ncnn::Mat in = ncnn::Mat::from_pixels(
    image_data, ncnn::Mat::PIXEL_BGR2RGB, width, height
);
ex.extract("input", in);

// Run inference
ex.run();

// Get output
ncnn::Mat out;
ex.extract("output", out);
```

### MNN (Mobile Neural Network)

**Overview**: Alibaba's lightweight inference framework

| Feature | Details |
|---------|---------|
| **Platforms** | Android, iOS, Linux, Windows, MCU |
| **Model Formats** | .mnn |
| **Quantization** | INT8, INT16, BF16 |
| **Hardware Support** | CPU (ARM NEON), GPU (OpenCL/Metal), NPU |
| **Language** | C++, Java, Python |

```python
import MNN
import MNN.nn as nn
import MNN.expr as F

# Load model
config = {
    "backend": "CPU",
    "precision": "LOW",
    "numThread": 4,
    "power": "LOW",
}
session = nn.create_session(config)
net = nn.load_module("model.mnn", ["input"], ["output"], session)

# Run inference
input = F.shape([1, 3, 224, 224], dtype=F.halide_type_of())
output = net.forward(input)
```

### TFLite Micro

**Overview**: TensorFlow Lite for microcontrollers

| Feature | Details |
|---------|---------|
| **Platforms** | Arduino, STM32, ESP32, Nordic, NXP |
| **Model Formats** | .tflite |
| **Quantization** | INT8, INT16 |
| **Hardware Support** | Cortex-M, Xtensa, RISC-V |
| **Language** | C, C++ |

```cpp
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Load model
const tflite::Model* model = tflite::GetModel(g_model_data);

// Create interpreter
tflite::AllOpsResolver resolver;
tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);
interpreter.AllocateTensors();

// Get input/output tensors
TfLiteTensor* input = interpreter.input(0);
TfLiteTensor* output = interpreter.output(0);

// Set input
input->data.int8[0] = input_value;

// Run inference
interpreter.Invoke();

// Get output
int8_t result = output->data.int8[0];
```

---

## Model Optimization Tools

### NVIDIA TensorRT

**Overview**: High-performance deep learning inference optimizer and runtime

| Feature | Details |
|---------|---------|
| **Target Hardware** | NVIDIA GPUs, Jetson |
| **Quantization** | INT8, FP16, INT4 |
| **Optimizations** | Layer fusion, kernel auto-tuning, dynamic batching |
| **Model Formats** | .trt (serialized engine) |

```bash
# Convert ONNX to TensorRT
trtexec --onnx=model.onnx --saveEngine=model.trt --fp16

# INT8 quantization
trtexec --onnx=model.onnx --saveEngine=model_int8.trt \
    --int8 --calib=calibration_cache.bin

# Benchmark
trtexec --loadEngine=model.trt --batch=1 --iterations=1000
```

```python
import tensorrt as trt

# Build engine
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

# Parse ONNX model
with open("model.onnx", "rb") as f:
    parser.parse(f.read())

# Configure builder
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
config.set_flag(trt.BuilderFlag.FP16)

# Build engine
engine = builder.build_serialized_network(network, config)

# Run inference
context = engine.create_execution_context()
stream = cuda.Stream()

# Allocate buffers
inputs, outputs, bindings = allocate_buffers(engine)

# Run
cuda.memcpy_htod_async(inputs[0].device, host_input, stream)
context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
cuda.memcpy_dtoh_async(host_output, outputs[0].device, stream)
```

### OpenVINO

**Overview**: Intel's toolkit for optimized inference

| Feature | Details |
|---------|---------|
| **Target Hardware** | Intel CPUs, GPUs, VPUs, FPGAs |
| **Quantization** | INT8, INT4 |
| **Optimizations** | Model optimization, quantization, pruning |
| **Model Formats** | .xml + .bin |

```bash
# Install
pip install openvino

# Convert model
mo --input_model model.onnx --output_dir model_ir --data_type FP16

# Benchmark
benchmark_app -m model_ir/model.xml -d CPU
```

```python
from openvino.runtime import Core

# Load model
core = Core()
model = core.read_model("model.xml")

# Compile for specific device
compiled_model = core.compile_model(model, "CPU")  # or "GPU", "MYRIAD"

# Run inference
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

result = compiled_model({input_layer: input_data})[output_layer]
```

### Core ML Tools

**Overview**: Apple's framework for on-device ML

| Feature | Details |
|---------|---------|
| **Target Hardware** | Apple Neural Engine, GPU, CPU |
| **Quantization** | FP16 |
| **Optimizations** | Neural Engine scheduling, operator fusion |
| **Model Formats** | .mlpackage, .mlmodel |

```python
import coremltools as ct

# Convert from PyTorch
import torch
model = torch.load("model.pth")
model.eval()

example = torch.randn(1, 3, 224, 224)
traced = torch.jit.trace(model, example)

# Convert to Core ML
mlmodel = ct.convert(
    traced,
    inputs=[ct.ImageType(shape=(1, 3, 224, 224))],
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.iOS17
)

# Add metadata
mlmodel.author = "Team"
mlmodel.short_description = "Image classifier"
mlmodel.save("Model.mlpackage")

# Predict
# prediction = mlmodel.predict({"image": input_image})
```

### Hailo Dataflow Compiler

**Overview**: Compiler for Hailo AI processors

| Feature | Details |
|---------|---------|
| **Target Hardware** | Hailo-8, Hailo-8L, Hailo-15 |
| **Quantization** | INT8, INT16 |
| **Optimizations** | Dataflow graph optimization, compiler scheduling |
| **Model Formats** | .hef (Hailo Executable Format) |

```bash
# Compile model for Hailo
hailo compiler model.onnx --output model.hef --hw-arch hailo8

# Optimize
hailo optimize model.hef --optimization-level aggressive

# Benchmark
hailo benchmark model.hef --input-shape "1,3,224,224"
```

### Edge Impulse

**Overview**: End-to-end platform for edge ML development

| Feature | Details |
|---------|---------|
| **Platforms** | Arduino, STM32, ESP32, Nordic, Raspberry Pi |
| **Model Formats** | .tflite, .onnx |
| **Quantization** | INT8 |
| **Hardware Support** | MCU, Edge processors |
| **Language** | Python, C++, JavaScript |

```bash
# Install CLI
npm install -g edge-impulse-cli

# Upload data
edge-impulse-data-uploader --label normal --camera /dev/video0

# Train model
edge-impulse-linux-runner --model model.eim

# Deploy to device
edge-impulse-deployment --device /dev/ttyACM0
```

---

## Hardware-Specific SDKs

### NVIDIA JetPack SDK

| Component | Version | Description |
|-----------|---------|-------------|
| CUDA | 12.x | GPU computing |
| cuDNN | 9.x | Deep learning primitives |
| TensorRT | 10.x | Inference optimization |
| DeepStream | 7.x | Video analytics |
| Isaac | 2026.x | Robotics |
| TAO | 5.x | Transfer learning |

```bash
# Install JetPack
sudo apt-get install nvidia-jetpack

# Run sample
cd /usr/local/cuda/samples
make
./1_Utilities/deviceQuery/deviceQuery
```

### Qualcomm AI Engine Direct

| Component | Description |
|-----------|-------------|
| **QNN SDK** | Qualcomm Neural Network SDK |
| **SNPE** | Snapdragon Neural Processing Engine |
| **AI Hub** | Model zoo and optimization |
| **AI Engine** | Heterogeneous compute manager |

```bash
# Convert model
snpe-onnx-to-dlc --input_network model.onnx --output_path model.dlc

# Quantize
snpe-dlc-quantize --input_dlc model.dlc --input_list calibration_list.txt

# Run inference
snpe-net-run --container model.dlc --input_list input_list.txt
```

### Google Coral SDK

| Component | Description |
|-----------|-------------|
| **Edge TPU Compiler** | Compile models for Edge TPU |
| **PyCoral** | Python API |
| **libcoral** | C++ API |
| **Edgetpu_runtime** | Runtime library |

```bash
# Compile model
edgetpu_compile model.tflite

# Run inference
python3 classify_image.py \
    --model model_edgetpu.tflite \
    --labels labels.txt \
    --input image.jpg
```

### Apple Xcode + Core ML

| Component | Description |
|-----------|-------------|
| **Core ML** | On-device ML framework |
| **Create ML** | No-code model training |
| **Core ML Tools** | Model conversion |
| **Metal** | GPU compute |
| **Neural Engine** | Apple's AI accelerator |

```swift
// Swift code for Core ML inference
import CoreML

let model = try! MyModel(configuration: MLModelConfiguration())

let input = MyModelInput(image: inputImage)
let output = try! model.prediction(input: input)

print("Class: \(output.classLabel)")
print("Confidence: \(output.classLabelProbs)")
```

---

## Edge AI Platforms

### NVIDIA Jetson Platform

| Platform | Description | Use Case |
|----------|-------------|----------|
| **JetPack SDK** | Full development environment | All Jetson devices |
| **DeepStream** | Video analytics pipeline | Smart cities, retail |
| **Isaac** | Robotics development | Manufacturing, logistics |
| **Metropolis** | Video analytics platform | City infrastructure |
| **DRIVE** | Autonomous driving | Automotive |

### Qualcomm AI Cloud

| Service | Description |
|---------|-------------|
| **AI Hub** | Model zoo and optimization |
| **AI Engine Direct** | Low-level API |
| **QNN SDK** | Neural network SDK |
| **Cloud AI** | Cloud inference |

### MediaTek NeuroPilot

| Component | Description |
|-----------|-------------|
| **NeuroPilot SDK** | Development toolkit |
| **APU** | AI Processing Unit |
| **Model Zoo** | Pre-optimized models |
| **Quantization Tools** | INT8/INT4 conversion |

### Intel Edge AI

| Component | Description |
|-----------|-------------|
| **OpenVINO** | Inference optimization |
| **Edge Insights** | Platform management |
| **Dev Cloud** | Cloud development |
| **Distribution Center** | Model deployment |

---

## Development Environments

### Edge Impulse Studio

```yaml
# Edge Impulse project configuration
name: "Defect Detection"
version: 1.0
device: "Arduino Nicla Vision"
target: "Cortex-M7"

pipeline:
  - data_collection:
      - camera: "OV2640"
      - samples: 1000
      - augmentation: ["flip", "rotate", "brightness"]
  
  - preprocessing:
      - resize: [96, 96]
      - normalize: true
      - color_space: "RGB"
  
  - model:
      - architecture: "MobileNetV2"
      - quantization: "INT8"
      - pruning: 0.3
  
  - deployment:
      - format: "C++ library"
      - optimization: "FOMO"
      - memory: "256KB"
```

### Google Colab Edge AI

```python
# Install edge AI tools in Colab
!pip install tflite-runtime
!pip install onnxruntime
!pip install edge-tpu

# Train model
import tensorflow as tf
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Compile for Edge TPU
!edgetpu_compile model.tflite
```

### VS Code + Remote Development

```json
// .vscode/settings.json
{
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "terminal.integrated.env.linux": {
        "TARGET_DEVICE": "jetson-orin-nano",
        "MODEL_FORMAT": "tensorrt"
    }
}
```

---

## Benchmarking Tools

### MLPerf

| Benchmark | Target | Metrics |
|-----------|--------|---------|
| **MLPerf Inference** | Data center, edge | Latency, throughput |
| **MLPerf Tiny** | Microcontrollers | Latency, energy |
| **MLPerf Mobile** | Mobile devices | Latency, accuracy |

```bash
# Run MLPerf Inference benchmark
./run.sh --benchmarks=ssd-resnet34 --scenario=SingleStream

# Run MLPerf Tiny
python run_benchmark.py --model model.tflite --device arduino_nicla
```

### ONNX Runtime Benchmarking

```python
import onnxruntime as ort
import numpy as np
import time

def benchmark_onnx(model_path, num_runs=1000, warmup=100):
    """Benchmark ONNX model"""
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    # Warmup
    for _ in range(warmup):
        input_data = np.random.randn(*input_shape).astype(np.float32)
        session.run(None, {input_name: input_data})
    
    # Benchmark
    latencies = []
    for _ in range(num_runs):
        input_data = np.random.randn(*input_shape).astype(np.float32)
        
        start = time.perf_counter()
        session.run(None, {input_name: input_data})
        end = time.perf_counter()
        
        latencies.append((end - start) * 1000)
    
    return {
        "mean_ms": np.mean(latencies),
        "median_ms": np.median(latencies),
        "p95_ms": np.percentile(latencies, 95),
        "p99_ms": np.percentile(latencies, 99),
        "throughput_fps": 1000 / np.mean(latencies),
    }
```

### AI Benchmark

```bash
# Install
pip install ai-benchmark

# Run benchmark
python -c "
from ai_benchmark import AIBenchmark

benchmark = AIBenchmark()
results = benchmark.run()

print('Device:', results['device'])
print('Overall Score:', results['overall_score'])
print('Inference Score:', results['inference_score'])
print('Training Score:', results['training_score'])
"
```

### TensorFlow Lite Benchmark

```bash
# Build benchmark binary
bazel build //tensorflow/lite/tools/benchmark:benchmark_model

# Run benchmark
./bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model \
    --graph=model.tflite \
    --num_threads=4 \
    --num_runs=100
```

---

## Deployment and Management

### Edge Impulse

```bash
# Create project
edge-impulse-new-project "My Project"

# Upload data
edge-impulse-data-uploader \
    --label "defect" \
    --camera /dev/video0 \
    --count 100

# Train model
edge-impulse-linux-runner \
    --model model.eim \
    --camera /dev/video0
```

### NVIDIA Fleet Command

```bash
# Install Fleet Command
sudo apt-get install fleet-command

# Create deployment
fleet-command deploy \
    --name "factory-inspection" \
    --model model.trt \
    --devices "factory-line-01,factory-line-02"

# Monitor deployment
fleet-command status --deployment-id dep-12345
```

### AWS IoT Greengrass

```bash
# Install Greengrass
curl -s https://d2uenmpfh84ha6.cloudfront.net/latest/greengrass-v2-installer.zip -o gg.zip
unzip gg.zip
sudo ./greengrass-nucleus-latest/install

# Deploy ML component
aws greengrassv2 create-component-version \
    --inline-recipe fileb://model-component.zip

# Create deployment
aws greengrassv2 create-deployment \
    --target-arn "arn:aws:iot:region:account:thing/device-1" \
    --components '{"com.example.ml-model": {"componentVersion": "1.0"}}'
```

### Azure IoT Edge

```bash
# Install IoT Edge
sudo apt-get install iotedge

# Deploy module
az iot edge set-modules \
    --device-id "device-1" \
    --modules '[{
        "name": "ml-inference",
        "type": "docker",
        "settings": {
            "image": "model:latest"
        }
    }]'
```

---

## Open Source Projects

### llama.cpp

**Description**: LLM inference in C/C++

```bash
# Build
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make -j

# Convert model
python convert_hf_to_gguf.py --model ./model --outfile model.gguf

# Quantize
./llama-quantize model.gguf model-q4_k_m.gguf Q4_K_M

# Run
./llama-cli -m model-q4_k_m.gguf -p "Hello" -n 100
```

### MediaPipe

**Description**: Google's cross-platform ML solutions

```python
import mediapipe as mp
import cv2

# Initialize
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Process frame
image = cv2.imread("image.jpg")
results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Draw landmarks
mp.solutions.drawing_utils.draw_landmarks(
    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
)
```

### NCNN

**Description**: Tencent's neural network inference framework

```cpp
// C++ example
#include <ncnn/net.h>

ncnn::Net net;
net.load_param("model.param");
net.load_model("model.bin");

ncnn::Extractor ex = net.create_extractor();
ex.input("input", input_mat);
ex.extract("output", output_mat);
```

### OpenCV DNN

**Description**: Computer vision with DNN module

```python
import cv2

# Load model
net = cv2.dnn.readNetFromONNX("model.onnx")

# Prepare input
blob = cv2.dnn.blobFromImage(image, 1/255.0, (224, 224))

# Run inference
net.setInput(blob)
output = net.forward()

# Process output
class_id = np.argmax(output)
confidence = output[0][class_id]
```

### TinyEngine

**Description**: Lightweight inference engine for MCU

```c
// TinyEngine example
#include "tinyengine.h"

// Load model
model_t model = tinyengine_load("model.bin");

// Run inference
input_t input = {
    .data = input_buffer,
    .height = 28,
    .width = 28,
    .channels = 1,
};

output_t output = tinyengine_run(model, input);

// Process result
int predicted_class = argmax(output.data, output.size);
```

---

## Comparison Matrix

### Framework Comparison

| Framework | Latency | Memory | Ease of Use | Hardware Support |
|-----------|---------|--------|-------------|------------------|
| **TensorFlow Lite** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **ONNX Runtime** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **PyTorch Mobile** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **NCNN** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **MNN** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **TensorRT** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **OpenVINO** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

### Use Case Recommendations

| Use Case | Recommended Framework | Why |
|----------|----------------------|-----|
| **Mobile App** | TensorFlow Lite / Core ML | Wide platform support |
| **IoT Device** | NCNN / TFLite Micro | Ultra-lightweight |
| **Edge Server** | TensorRT / ONNX Runtime | Maximum performance |
| **Robotics** | TensorRT / Isaac | Real-time requirements |
| **Vision** | OpenCV DNN / MediaPipe | Computer vision focus |
| **LLM** | llama.cpp / ONNX Runtime | Language model optimization |
| **MCU** | TFLite Micro / TinyEngine | Minimal resources |

---

## Cross-References

- **01-Foundations**: ML fundamentals
- **23-Local-AI-Inference-Self-Hosting**: Desktop/server local inference
- **29-Reasoning-and-Inference-Scaling**: Inference optimization
- **30-Small-Language-Models**: Small models for edge
- **38-AI-Supply-Chain-and-Chip-Design**: Hardware ecosystem
- **56-MLOps-and-AI-Platform-Engineering**: Production deployment

---

*Last updated: July 7, 2026*
*Category: 62-Edge-AI-and-On-Device-Inference*
