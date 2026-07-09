# Tools and Frameworks: AI Wearables and Ambient Intelligence

> Comprehensive guide to the development tools, frameworks, SDKs, and platforms available for building AI-powered wearable and ambient intelligence applications in 2026.

---

## 1. AI Model Frameworks for Wearables

### 1.1 On-Device Inference Engines

| Framework | Maintainer | Models Supported | Platforms | License |
|-----------|-----------|-----------------|-----------|---------|
| **ONNX Runtime Mobile** | Microsoft | ONNX format | iOS, Android, Linux | MIT |
| **TensorFlow Lite** | Google | TFLite format | Android, iOS, MCU | Apache 2.0 |
| **PyTorch Mobile** | Meta | TorchScript | iOS, Android | BSD-3 |
| **Core ML** | Apple | CoreML format | iOS, visionOS, watchOS | Proprietary |
| **Qualcomm QNN SDK** | Qualcomm | QNN format | Snapdragon NPUs | Proprietary |
| **MediaTek NeuroPilot** | MediaTek | Various | MediaTek SoCs | Proprietary |
| **Samsung ONE** | Samsung | Various | Exynos NPUs | Proprietary |
| **Apache TVM** | Apache | Multiple | Cross-platform | Apache 2.0 |
| **MNN** | Alibaba | MNN format | Android, iOS | Apache 2.0 |
| **ncnn** | Tencent | ncnn format | Android, iOS | BSD-3 |

### 1.2 Model Zoo for Wearables

Pre-optimized models ready for deployment:

**Vision Models:**

| Model | Size | Latency | Accuracy | Use Case |
|-------|------|---------|----------|----------|
| MobileNetV4 | 6MB | 15ms | 75% Top-1 | General classification |
| EfficientNet-Lite0 | 5MB | 20ms | 73% Top-1 | Efficient vision |
| YOLOv8-nano | 5MB | 30ms | 72% mAP | Object detection |
| MobileCLIP-SigLIP | 15MB | 50ms | 78% Top-1 | Image-text matching |
| MediaPipe Face Mesh | 2MB | 10ms | 95% landmark | Face tracking |
| Depth Anything V2-tiny | 8MB | 25ms | 0.85 RMSE | Monocular depth |

**Audio Models:**

| Model | Size | Latency | Accuracy | Use Case |
|-------|------|---------|----------|----------|
| Whisper-tiny | 75MB | 100ms | 12% WER | Speech recognition |
| Whisper-tiny-streaming | 38MB | 60ms | 13% WER | Streaming ASR |
| VITS2-tiny | 20MB | 40ms | 3.8 MOS | Text-to-speech |
| YAMNet | 3MB | 5ms | 92% | Sound classification |
| Silero VAD | 2MB | 2ms | 98% | Voice activity detection |
| Wav2Vec2-tiny | 25MB | 30ms | 15% WER | Robust ASR |

**Language Models:**

| Model | Size (quantized) | Latency | Capability | Use Case |
|-------|------------------|---------|-----------|----------|
| Phi-4-mini (INT4) | 600MB | 200ms | General reasoning | Primary assistant |
| Gemma-3-2B (INT4) | 800MB | 250ms | General reasoning | Secondary assistant |
| SmolLM2-135M (INT4) | 70MB | 50ms | Simple tasks | Quick responses |
| GPT-2 small (INT8) | 120MB | 30ms | Text completion | Autocomplete |
| T5-small (INT8) | 100MB | 40ms | Translation | Real-time translation |
| DistilBERT (INT8) | 65MB | 15ms | Classification | Intent detection |

### 1.3 Model Optimization Tools

```python
# Example: Full optimization pipeline using multiple tools

# Step 1: Export from PyTorch to ONNX
from optimum.onnxruntime import ORTModelForCausalLM

model = ORTModelForCausalLM.from_pretrained(
    "microsoft/phi-4-mini",
    export=True
)
model.save_pretrained("./onnx_model")

# Step 2: Quantize with ONNX Runtime
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

quantizer = ORTQuantizer.from_pretrained(model)
qconfig = AutoQuantizationConfig.arm64(is_static=True, per_channel=True)

quantizer.quantize(
    quantization_config=qconfig,
    save_dir="./quantized_model"
)

# Step 3: Convert to platform-specific format
# For CoreML (Apple):
import coremltools as ct

coreml_model = ct.converters.onnx.convert(
    "./quantized_model/model.onnx",
    minimum_deployment_target=ct.target.iOS17
)

# For TFLite (Android):
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("./quantized_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)
```

---

## 2. Development SDKs and Platforms

### 2.1 Apple Ecosystem

**visionOS SDK (Vision Pro / future smart glasses):**

| Component | Framework | Capability |
|-----------|----------|------------|
| Eye tracking | ARKit + RealityKit | Gaze-based interaction |
| Hand tracking | ARKit | Gesture recognition |
| Spatial computing | RealityKit | 3D content placement |
| On-device AI | CoreML | Model inference |
| Voice | Speech framework | ASR + voice commands |
| Health | HealthKit | Biometric data access |

**watchOS SDK (Apple Watch):**

| Component | Framework | Capability |
|-----------|----------|------------|
| Health monitoring | HealthKit | Heart rate, ECG, SpO2 |
| Fitness | WorkoutKit | Activity tracking |
| Notifications | UserNotifications | Proactive alerts |
| Complications | ClockKit | Watch face data |
| Siri | SiriKit | Voice assistant |

### 2.2 Google Ecosystem

**Wear OS SDK:**

| Component | Library | Capability |
|-----------|---------|------------|
| Tiles | Wear Tiles API | Watch face complications |
| Health | Health Services API | Sensors, exercises |
| Compose | Wear Compose | UI toolkit |
| Data layer | Data Layer API | Phone↔watch sync |
| Tiles AI | Gemini Tiles | AI-powered watch faces |

**MediaPipe (Cross-platform):**

| Task | Model | Size | Use Case |
|------|-------|------|----------|
| Face detection | BlazeFace | 1MB | Real-time face tracking |
| Face mesh | FaceMesh | 2MB | 468-point face landmarks |
| Hand tracking | Hands | 4MB | 21-point hand landmarks |
| Pose detection | MoveNet | 5MB | Full body pose |
| Object detection | SSD-MobileNet | 6MB | Real-time detection |
| Audio classification | YAMNet | 3MB | Sound classification |

### 2.3 Meta Ecosystem

**Meta Quest SDK (VR/AR):**

| Component | API | Capability |
|-----------|-----|------------|
| Passthrough | XR Passthrough API | Real-world view |
| Hand tracking | Hand Tracking API | Hand gesture input |
| Eye tracking | Eye Tracking API | Gaze-based interaction |
| Voice | Voice SDK | Voice commands |
| Spatial anchors | Shared Spatial Anchor API | Persistent AR content |
| Presence Platform | Mixed Reality API | Scene understanding |

**Meta AI SDK (Llama on wearables):**

```python
# Meta AI SDK for on-device Llama inference
from meta_ai_sdk import WearableLlamaClient

client = WearableLlamaClient(
    model="llama-4-scout-quantized",
    backend="qualcomm_qnn",
    memory_limit_mb=500,
    power_budget_mw=300
)

# Streaming inference for real-time response
async for token in client.stream_inference(
    prompt=user_query,
    context=current_context,
    max_tokens=256,
    temperature=0.7
):
    display_streaming(token)
```

### 2.4 Qualcomm Ecosystem

**Snapdragon Spaces SDK:**

| Component | API | Capability |
|-----------|-----|------------|
| 3D rendering | OpenXR | AR content |
| Spatial anchors | Spatial Anchors API | Persistent placement |
| Scene understanding | Scene API | Environmental mesh |
| Hand tracking | Hand Tracking API | Gesture input |
| Eye tracking | Eye Tracking API | Gaze interaction |
| Audio | Spatial Audio API | 3D sound |
| AI inference | QNN SDK | NPU acceleration |

---

## 3. Sensor and Perception Libraries

### 3.1 Computer Vision Libraries

| Library | Purpose | Performance | Platform |
|---------|---------|-------------|----------|
| **OpenCV Mobile** | General CV | Fast | Cross-platform |
| **libyuv** | Image processing | Very fast | Cross-platform |
| **Skia** | 2D rendering | Fast | Cross-platform |
| **ARCore/ARKit** | AR features | Fast | Android/iOS |
| **MediaPipe** | ML perception | Fast | Cross-platform |
| **DepthAI** | Depth sensing | Fast | Intel/Coral |

### 3.2 Audio Processing Libraries

| Library | Purpose | Latency | Platform |
|---------|---------|---------|----------|
| **WebRTC APM** | Audio processing | 5ms | Cross-platform |
| **RNNoise** | Noise suppression | 2ms | Cross-platform |
| **SpeexDSP** | Voice processing | 3ms | Cross-platform |
| **PulseAudio** | Audio routing | Variable | Linux |
| **CoreAudio** | Audio routing | 1ms | Apple |
| **Oboe** | Low-latency audio | 2ms | Android |

### 3.3 Sensor Fusion Libraries

```python
# Example: Multi-sensor fusion using Kalman filtering

import numpy as np
from filterpy.kalman import ExtendedKalmanFilter

class WearableSensorFusion:
    """
    Fuses data from multiple sensors into a unified
    state estimate for the user's context.
    """
    
    def __init__(self):
        # State: [position, velocity, orientation, activity]
        self.ekf = ExtendedKalmanFilter(dim_x=9, dim_z=6)
        
        # State transition matrix
        self.ekf.F = np.array([
            [1, dt, 0, 0, 0, 0, 0, 0, 0],  # position
            [0, 1, 0, 0, 0, 0, 0, 0, 0],  # velocity
            [0, 0, 1, dt, 0, 0, 0, 0, 0],  # orientation
            [0, 0, 0, 1, 0, 0, 0, 0, 0],  # orientation rate
            [0, 0, 0, 0, 1, dt, 0, 0, 0],  # altitude
            [0, 0, 0, 0, 0, 1, 0, 0, 0],  # altitude rate
            [0, 0, 0, 0, 0, 0, 1, 0, 0],  # activity
            [0, 0, 0, 0, 0, 0, 0, 1, 0],  # heart rate
            [0, 0, 0, 0, 0, 0, 0, 0, 1],  # temperature
        ])
    
    def fuse(self, gps_data, imu_data, ppg_data, baro_data):
        """Fuse all sensor inputs into unified state."""
        
        # Prediction step
        self.ekf.predict()
        
        # Update with GPS (if available)
        if gps_data is not None:
            z_gps = np.array([gps_data.lat, gps_data.lon])
            self.ekf.update(z_gps, HJacobian=self.gps_h, Hx=self.gps_hx)
        
        # Update with IMU
        z_imu = np.array([
            imu_data.accel_x, imu_data.accel_y, imu_data.accel_z,
            imu_data.gyro_x, imu_data.gyro_y, imu_data.gyro_z
        ])
        self.ekf.update(z_imu, HJacobian=self.imu_h, Hx=self.imu_hx)
        
        # Update with PPG (heart rate)
        z_ppg = np.array([ppg_data.heart_rate])
        self.ekf.update(z_ppg, HJacobian=self.ppg_h, Hx=self.ppg_hx)
        
        return FusedState(
            position=self.ekf.x[0:2],
            velocity=self.ekf.x[2:4],
            orientation=self.ekf.x[4:6],
            altitude=self.ekf.x[6],
            activity=int(self.ekf.x[7]),
            heart_rate=self.ekf.x[8],
            confidence=self.calculate_confidence()
        )
```

---

## 4. Cloud Services for Wearable AI

### 4.1 Cloud AI Platforms

| Platform | Strength | Latency | Pricing |
|----------|---------|---------|---------|
| **OpenAI API** | GPT-4o, reasoning | 100-500ms | $2.50-10/1M tokens |
| **Google AI Studio** | Gemini 2.5, multimodal | 80-400ms | $1.25-5/1M tokens |
| **Anthropic API** | Claude 4, safety | 100-600ms | $3-15/1M tokens |
| **AWS Bedrock** | Multi-model, enterprise | 100-500ms | Variable |
| **Azure AI** | Enterprise, compliance | 100-500ms | Variable |
| **Replicate** | Open-source models | 200-1000ms | $0.00065/s |

### 4.2 Edge AI Cloud Services

| Service | Purpose | Latency | Use Case |
|---------|---------|---------|----------|
| **AWS IoT Greengrass** | Edge ML inference | 10-50ms | Industrial IoT |
| **Azure IoT Edge** | Edge AI runtime | 10-50ms | Enterprise wearables |
| **Google Distributed Cloud** | Edge inference | 5-30ms | Retail, healthcare |
| **Qualcomm AI Cloud** | Cloud-offload AI | 20-100ms | Mobile/wearable |
| **NVIDIA Jetson Cloud** | Edge AI management | 5-20ms | Robotics, manufacturing |

### 4.3 Real-Time Communication

| Protocol | Latency | Reliability | Use Case |
|----------|---------|-------------|----------|
| **WebRTC** | 50-200ms | Good | Voice/video streaming |
| **QUIC** | 30-100ms | Excellent | General data |
| **MQTT over TLS** | 50-200ms | Good | IoT telemetry |
| **gRPC** | 20-100ms | Good | Service communication |
| **WebSocket** | 50-200ms | Medium | Real-time updates |

---

## 5. Design Tools and UX Frameworks

### 5.1 Wearable UX Design Systems

| Design System | Platform | Key Features |
|--------------|----------|-------------|
| **Material Design 3 for Wear OS** | Google | Dynamic color, ambient mode |
| **Human Interface Guidelines (watchOS)** | Apple | Digital crown, complications |
| **Meta Quest Design System** | Meta | Spatial UI, hand interactions |
| **Samsung One UI Watch** | Samsung | Tile-based, rotating bezel |
| **Figma for Wearables** | Cross-platform | Prototyping, components |

### 5.2 Voice Design Tools

| Tool | Purpose | Capability |
|------|---------|-----------|
| **Voiceflow** | Voice conversation design | Visual flow builder |
| **Botium** | Voice testing | Automated voice testing |
| **Alexa Skill Kit** | Alexa voice apps | Voice-first development |
| **Dialogflow CX** | Conversational AI | Multi-turn dialog |
| **Rasa** | Open-source voice | Custom voice assistants |

### 5.3 Prototyping Tools

| Tool | Type | Platforms |
|------|------|-----------|
| **Reality Composer** | AR prototyping | Apple |
| **Scene Studio** | AR prototyping | Meta |
| **Blender** | 3D modeling | Cross-platform |
| **Unity** | Game/AR engine | Cross-platform |
| **Unreal Engine** | High-fidelity AR | Cross-platform |
| **Figma** | UI design | Cross-platform |

---

## 6. Testing and Quality Assurance Tools

### 6.1 Device Testing

| Tool | Purpose | Platform |
|------|---------|----------|
| **XCTest** | Unit/integration testing | Apple |
| **JUnit** | Unit testing | Android |
| **Espresso** | UI testing | Android |
| **XCUITest** | UI testing | Apple |
| **Firebase Test Lab** | Cloud device testing | Android |
| **BrowserStack** | Cross-device testing | Web |
| **AWS Device Farm** | Device testing | Cross-platform |

### 6.2 Performance Profiling

| Tool | Purpose | Metric |
|------|---------|--------|
| **Instruments (Xcode)** | Profiling | CPU, memory, energy |
| **Android Profiler** | Profiling | CPU, memory, network |
| **Qualcomm Snapdragon Profiler** | NPU profiling | TOPS, power |
| **Netron** | Model visualization | Model structure |
| **NetSpeed** | Network profiling | Bandwidth, latency |
| **PowerTop** | Power profiling | Energy consumption |

### 6.3 Privacy and Security Testing

| Tool | Purpose | Type |
|------|---------|------|
| **OWASP ZAP** | Security testing | Penetration testing |
| **MobSF** | Mobile security | Static analysis |
| **Frida** | Runtime analysis | Dynamic analysis |
| **Objection** | Mobile exploration | Runtime manipulation |
| **Battery Historian** | Privacy audit | Data access patterns |
| **Privacy Checker** | Compliance | GDPR/CCPA audit |

---

## 7. Open Source Projects

### 7.1 Notable Open Source Projects for Wearable AI

| Project | Description | Stars | Language |
|---------|-------------|-------|----------|
| **MediaPipe** | Cross-platform ML perception | 28k+ | C++/Python |
| **ncnn** | Neural network inference | 20k+ | C++ |
| **MNN** | Mobile neural network | 8k+ | C++ |
| **TNN** | Neural network inference | 4k+ | C++ |
| **TFLite Model Maker** | Custom model training | 3k+ | Python |
| **OpenCV Mobile** | Computer vision | 5k+ | C++ |
| **WebRTC** | Real-time communication | 13k+ | C++ |
| **RNNoise** | Noise suppression | 3k+ | C |
| **Whisper.cpp** | Speech recognition | 35k+ | C++ |
| **Ollama** | Local LLM inference | 110k+ | Go |

### 7.2 Contributing to Wearable AI Open Source

```bash
# Getting started with MediaPipe for wearable development

# Clone the repository
git clone https://github.com/google/mediapipe.git
cd mediapipe

# Build for Android
bazel build -c opt --config=android_arm64 \
  //mediapipe/examples/android/src/java/com/google/mediapipe/apps/facegpu:facegpu

# Build for iOS
bazel build -c opt --config=ios_arm64 \
  //mediapipe/ios:MediaPipeApp

# Run tests
bazel test //mediapipe/...

# Run on device
adb install -r bazel-bin/mediapipe/examples/android/.../facegpu.apk
```

---

## 8. Comparison Matrix: Choosing the Right Tools

### 8.1 Decision Framework

```
START
  │
  ├── Is it a consumer product?
  │     ├── YES → Choose platform SDK (Apple/Google/Meta)
  │     └── NO → Continue
  │
  ├── Is it enterprise/industrial?
  │     ├── YES → Qualcomm Spaces + Cloud AI
  │     └── NO → Continue
  │
  ├── Is it research/prototype?
  │     ├── YES → MediaPipe + ONNX Runtime + Python
  │     └── NO → Continue
  │
  ├── Is cross-platform required?
  │     ├── YES → ONNX Runtime + MediaPipe + WebXR
  │     └── NO → Native SDK for target platform
  │
  └── What's the primary interaction?
        ├── Voice → Whisper + VITS + Dialogflow
        ├── Vision → MediaPipe + YOLO + CoreML
        ├── Gesture → MediaPipe Hands + custom ML
        └── Health → HealthKit/Health Services + custom ML
```

### 8.2 Cost Comparison

| Approach | Dev Cost | Hardware | Time to MVP | Scalability |
|----------|---------|----------|-------------|-------------|
| Apple Vision Pro SDK | High | $3,000/unit | 6-12 months | Medium |
| Wear OS + Gemini | Medium | $300/unit | 3-6 months | High |
| Meta Quest SDK | Medium | $500/unit | 4-8 months | High |
| Qualcomm Spaces | High | $400/unit | 6-12 months | High |
| Cross-platform (ONNX) | Medium | Varies | 4-8 months | High |
| Custom hardware | Very high | Custom | 12-24 months | Variable |

---

## 9. Cross-References

| Related Category | Connection |
|-----------------|------------|
| [01-Foundations](../01-Foundations/) | AI fundamentals |
| [02-LLMs](../02-LLMs/) | Language models for wearable AI |
| [23-Local-AI-Inference-Self-Hosting](../23-Local-AI-Inference-Self-Hosting/) | On-device deployment |
| [30-Small-Language-Models](../30-Small-Language-Models/) | Models optimized for edge |
| [31-AI-Workflow-Orchestration](../31-AI-Workflow-Orchestration-and-Durable-Execution/) | Orchestration patterns |
| [41-AI-Cost-Optimization](../41-AI-Cost-Optimization-and-Enterprise-ROI/) | Cost management |

---

## 10. Key Takeaways

1. **ONNX Runtime is the universal inference engine** — deploy the same model across all wearable platforms
2. **MediaPipe is the perception Swiss army knife** — face, hand, pose, audio, and object detection
3. **Platform SDKs are essential for native features** — CoreML for Apple, QNN for Qualcomm, TFLite for Google
4. **Model quantization is mandatory** — INT4/INT8 quantization reduces size 3-4x with minimal accuracy loss
5. **Open source is thriving** — Whisper.cpp, ncnn, MediaPipe provide production-ready foundations
6. **Cloud fallback is necessary** — edge devices need reliable cloud connections for complex tasks
7. **Testing on real hardware is non-negotiable** — simulators miss thermal, power, and sensor issues
