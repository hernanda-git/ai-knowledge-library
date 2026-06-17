# JavaScript ML Frameworks: TensorFlow.js, ONNX Runtime Web, and Beyond

## Introduction

The JavaScript ecosystem for machine learning has matured dramatically by 2026. Multiple production-grade frameworks now enable developers to run AI models directly in the browser, each with different strengths, trade-offs, and ideal use cases. This document provides a comprehensive comparison of the major JavaScript ML frameworks, with code examples for each, performance benchmarks, and guidance on when to use which framework.

## Overview of Major Frameworks

| Framework | Primary Focus | Backend Support | Training | Model Formats | Maturity |
|-----------|--------------|----------------|----------|---------------|----------|
| TensorFlow.js | Full ML lifecycle | WebGPU, WebGL, WASM, Node | Yes | TF SavedModel, Keras | Most mature |
| ONNX Runtime Web | Cross-framework inference | WebGPU, WebNN, WASM | No | ONNX (all formats) | Production-ready |
| Transformers.js | HuggingFace models | WASM, WebGPU, WebNN | No | ONNX (HuggingFace) | Rapidly growing |
| MediaPipe | Vision/Media | WebGPU, WebGL | No | TFLite, MediaPipe | Google-backed |
| WebLLM | LLM inference | WebGPU (primary) | No | GGUF, MLC format | Specialized |

## TensorFlow.js

TensorFlow.js is the most comprehensive JavaScript ML framework, supporting both model inference and training directly in the browser or Node.js.

### Architecture

TensorFlow.js uses a layered architecture:

```
┌─────────────────────────────────────────┐
│           JavaScript (API Layer)         │
│  Layers API  │  Core API  │  Data API   │
├─────────────────────────────────────────┤
│          Op Executor (Kernel)            │
├─────────────────────────────────────────┤
│  WebGPU  │  WebGL  │  WASM  │  Node.js  │
│  Backend │ Backend │Backend │  Backend   │
└─────────────────────────────────────────┘
```

### Setting Up TensorFlow.js

```bash
npm install @tensorflow/tfjs @tensorflow/tfjs-backend-webgpu
```

Or via CDN:

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgpu@latest"></script>
```

### Backend Selection

```javascript
import * as tf from '@tensorflow/tfjs';

// Check available backends
console.log('Available backends:', tf.engine().registryFactory_);

// Set backend
async function setOptimalBackend() {
  // Try WebGPU first (fastest for most operations)
  if (tf.findBackend('webgpu')) {
    await tf.setBackend('webgpu');
    console.log('Using WebGPU backend');
  }
  // Fall back to WebGL
  else if (tf.findBackend('webgl')) {
    await tf.setBackend('webgl');
    console.log('Using WebGL backend');
  }
  // Fall back to WASM
  else if (tf.findBackend('wasm')) {
    await tf.setBackend('wasm');
    console.log('Using WASM backend');
  }
  // CPU as last resort
  else {
    await tf.setBackend('cpu');
    console.log('Using CPU backend');
  }
  
  await tf.ready();
  console.log('Backend:', tf.getBackend());
  console.log('Memory:', tf.memory());
}
```

### Model Loading and Inference

```javascript
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';

class TFJSModelRunner {
  constructor() {
    this.model = null;
    this.backend = null;
  }

  async initialize(backendType = 'webgpu') {
    // Set backend
    try {
      await tf.setBackend(backendType);
      await tf.ready();
      this.backend = backendType;
      console.log(`Backend initialized: ${backendType}`);
    } catch (error) {
      console.warn(`Failed to set ${backendType}, falling back`);
      await tf.setBackend('wasm');
      await tf.ready();
      this.backend = 'wasm';
    }
  }

  async loadModel(modelUrl) {
    // Load model (supports TFJS model.json format)
    this.model = await tf.loadGraphModel(modelUrl);
    
    // Warm up the model
    const dummyInput = tf.zeros(this.model.inputs[0].shape);
    const warmupResult = this.model.predict(dummyInput);
    warmupResult.dispose();
    dummyInput.dispose();
    
    console.log('Model loaded and warmed up');
    return this.model;
  }

  async predict(inputData) {
    const startTime = performance.now();
    
    // Convert input to tensor
    const inputTensor = tf.tensor(inputData);
    
    // Run inference
    const outputTensor = this.model.predict(inputTensor);
    
    // Convert to array
    const outputData = await outputTensor.array();
    
    // Clean up tensors
    inputTensor.dispose();
    outputTensor.dispose();
    
    const inferenceTime = performance.now() - startTime;
    
    return {
      data: outputData,
      time: inferenceTime,
    };
  }

  async predictStreaming(inputData, callback) {
    // For models that support streaming (e.g., text generation)
    // This requires custom model implementation
    // Typically involves running the model repeatedly with state
    
    const maxTokens = 100;
    let generated = [];
    
    for (let i = 0; i < maxTokens; i++) {
      const output = await this.predict(inputData);
      const tokenId = this.sampleToken(output.data);
      
      generated.push(tokenId);
      callback(tokenId);
      
      // Update input with generated token
      // (architecture-specific)
      inputData = this.updateInput(inputData, tokenId);
      
      // Check for end token
      if (tokenId === this.endTokenId) break;
    }
    
    return generated;
  }

  sampleToken(logits) {
    // Simple argmax sampling
    let maxIdx = 0;
    let maxVal = -Infinity;
    for (let i = 0; i < logits.length; i++) {
      if (logits[i] > maxVal) {
        maxVal = logits[i];
        maxIdx = i;
      }
    }
    return maxIdx;
  }

  updateInput(inputData, tokenId) {
    // Placeholder - actual implementation depends on model
    return inputData;
  }

  getMemoryInfo() {
    return tf.memory();
  }

  dispose() {
    if (this.model) {
      this.model.dispose();
    }
    tf.disposeVariables();
  }
}

// Usage
async function runImageClassifier() {
  const runner = new TFJSModelRunner();
  await runner.initialize('webgpu');
  
  // Load MobileNet
  await runner.loadModel(
    'https://tfhub.dev/google/tfjs-model/mobilenet_v2/1/default/1'
  );
  
  // Load and preprocess image
  const img = document.getElementById('input-image');
  const tensor = tf.browser.fromPixels(img)
    .resizeBilinear([224, 224])
    .toFloat()
    .div(255)
    .sub([0.485, 0.456, 0.406])
    .div([0.229, 0.224, 0.225])
    .expandDims();
  
  // Run inference
  const result = await runner.predict(tensor);
  
  // Get predictions
  const predictions = result.data[0];
  const top5 = Array.from(predictions)
    .map((p, i) => ({ probability: p, index: i }))
    .sort((a, b) => b.probability - a.probability)
    .slice(0, 5);
  
  console.log('Top 5 predictions:', top5);
  console.log('Inference time:', result.time, 'ms');
  console.log('Memory:', runner.getMemoryInfo());
}
```

### Training in the Browser

TensorFlow.js uniquely supports training directly in the browser:

```javascript
import * as tf from '@tensorflow/tfjs';

class BrowserTrainer {
  constructor() {
    this.model = null;
  }

  createModel(inputShape, numClasses) {
    this.model = tf.sequential();
    
    this.model.add(tf.layers.dense({
      inputShape: [inputShape],
      units: 128,
      activation: 'relu',
    }));
    
    this.model.add(tf.layers.dropout({ rate: 0.2 }));
    
    this.model.add(tf.layers.dense({
      units: 64,
      activation: 'relu',
    }));
    
    this.model.add(tf.layers.dropout({ rate: 0.2 }));
    
    this.model.add(tf.layers.dense({
      units: numClasses,
      activation: 'softmax',
    }));
    
    this.model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });
    
    return this.model;
  }

  async train(features, labels, config = {}) {
    const {
      batchSize = 32,
      epochs = 10,
      validationSplit = 0.2,
      callbacks = [],
    } = config;
    
    // Convert to tensors
    const xs = tf.tensor2d(features);
    const ys = tf.tensor2d(labels);
    
    // Training history
    const history = {
      loss: [],
      acc: [],
      valLoss: [],
      valAcc: [],
    };
    
    // Custom callback for logging
    const logCallback = {
      onEpochEnd: (epoch, logs) => {
        history.loss.push(logs.loss);
        history.acc.push(logs.acc);
        history.valLoss.push(logs.val_loss);
        history.valAcc.push(logs.val_acc);
        
        console.log(`Epoch ${epoch + 1}/${epochs}`);
        console.log(`  loss: ${logs.loss.toFixed(4)}`);
        console.log(`  accuracy: ${logs.acc.toFixed(4)}`);
        console.log(`  val_loss: ${logs.val_loss.toFixed(4)}`);
        console.log(`  val_accuracy: ${logs.val_acc.toFixed(4)}`);
      },
    };
    
    // Train
    const startTime = performance.now();
    
    await this.model.fit(xs, ys, {
      batchSize,
      epochs,
      validationSplit,
      callbacks: [logCallback, ...callbacks],
      shuffle: true,
    });
    
    const trainingTime = (performance.now() - startTime) / 1000;
    
    // Clean up
    xs.dispose();
    ys.dispose();
    
    return {
      history,
      trainingTime,
      model: this.model,
    };
  }

  async evaluate(X_test, y_test) {
    const xs = tf.tensor2d(X_test);
    const ys = tf.tensor2d(y_test);
    
    const result = this.model.evaluate(xs, ys);
    
    xs.dispose();
    ys.dispose();
    
    return {
      loss: result[0].dataSync()[0],
      accuracy: result[1].dataSync()[0],
    };
  }
}

// Usage
async function trainClassifier() {
  const trainer = new BrowserTrainer();
  
  // Generate synthetic data
  const numSamples = 1000;
  const features = Array.from({ length: numSamples }, () => 
    Array.from({ length: 20 }, () => Math.random())
  );
  
  const labels = features.map(f => {
    const sum = f.reduce((a, b) => a + b, 0);
    return sum / 20 > 0.5 ? [0, 1] : [1, 0];
  });
  
  // Create and train model
  trainer.createModel(20, 2);
  
  const result = await trainer.train(features, labels, {
    epochs: 5,
    validationSplit: 0.2,
  });
  
  console.log(`Training completed in ${result.trainingTime.toFixed(2)}s`);
  console.log('Final accuracy:', result.history.acc[result.history.acc.length - 1]);
}
```

### Performance Optimization Tips for TF.js

```javascript
// 1. Use tf.tidy for automatic memory management
function efficientForward(model, input) {
  return tf.tidy(() => {
    const normalized = input.div(255).sub(0.5);  // Temp tensor auto-cleaned
    return model.predict(normalized);
  });
}

// 2. Reuse tensors instead of creating new ones
class TensorPool {
  constructor() {
    this.pool = new Map();
  }
  
  get(shape, dtype = 'float32') {
    const key = JSON.stringify(shape) + dtype;
    if (this.pool.has(key)) {
      return this.pool.get(key);
    }
    const tensor = tf.zeros(shape, dtype);
    this.pool.set(key, tensor);
    return tensor;
  }
}

// 3. Use the most specific backend for your operation
async function optimizeForGPU() {
  // WebGPU is best for large batch operations
  await tf.setBackend('webgpu');
  
  // WASM can be faster for small operations due to lower overhead
  // Consider operation-specific backend switching
}

// 4. Batch operations
async function batchPredictions(model, inputs, batchSize = 4) {
  const results = [];
  for (let i = 0; i < inputs.length; i += batchSize) {
    const batch = tf.stack(inputs.slice(i, i + batchSize));
    const output = model.predict(batch);
    results.push(...output.arraySync());
    batch.dispose();
    output.dispose();
  }
  return results;
}
```

## ONNX Runtime Web

ONNX Runtime Web brings the ONNX (Open Neural Network Exchange) ecosystem to the browser, supporting models from PyTorch, TensorFlow, scikit-learn, and other frameworks.

### Architecture

```
┌─────────────────────────────────────────┐
│          ONNX Runtime Web API            │
│  InferenceSession  │  Tensor  │  ...    │
├─────────────────────────────────────────┤
│         Session Management Layer         │
├─────────────────────────────────────────┤
│  WebGPU  │  WebNN  │  WASM  │  XNNPACK │
│  EP      │  EP     │  EP    │  EP      │
├─────────────────────────────────────────┤
│         ONNX Model (protobuf)           │
└─────────────────────────────────────────┘
```

### Setup and Basic Usage

```bash
npm install onnxruntime-web
```

```javascript
import * as ort from 'onnxruntime-web';

async function initializeONNX() {
  // Configure WASM path
  ort.env.wasm.wasmPaths = './node_modules/onnxruntime-web/dist/';
  
  // Create session with optimal execution providers
  const session = await ort.InferenceSession.create('./model.onnx', {
    executionProviders: [
      {
        name: 'webgpu',
        deviceType: 'gpu',
      },
      'webnn',
      'wasm',
    ],
    enableGraphOptimization: true,
    executionMode: 'parallel',
    enableCpuMemArena: true,
    enableMemPattern: true,
    optimizationLevel: 'all',
  });
  
  console.log('Session created successfully');
  console.log('Input names:', session.inputNames);
  console.log('Output names:', session.outputNames);
  
  return session;
}
```

### Complete Image Classification Pipeline

```javascript
import * as ort from 'onnxruntime-web';

class ONNXImageClassifier {
  constructor() {
    this.session = null;
    this.labels = [];
    this.profiler = {
      preprocess: 0,
      inference: 0,
      postprocess: 0,
    };
  }

  async loadModel(modelPath, labelsPath) {
    // Load model
    this.session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ['webgpu', 'wasm'],
    });
    
    // Load labels
    const response = await fetch(labelsPath);
    this.labels = await response.json();
  }

  async preprocess(imageElement) {
    const start = performance.now();
    
    // Get image dimensions
    const width = imageElement.naturalWidth || imageElement.width;
    const height = imageElement.naturalHeight || imageElement.height;
    
    // Create canvas for preprocessing
    const canvas = document.createElement('canvas');
    const targetSize = 224;
    canvas.width = targetSize;
    canvas.height = targetSize;
    
    const ctx = canvas.getContext('2d');
    
    // Calculate resize with aspect ratio preservation
    const scale = Math.min(targetSize / width, targetSize / height);
    const scaledWidth = width * scale;
    const scaledHeight = height * scale;
    const dx = (targetSize - scaledWidth) / 2;
    const dy = (targetSize - scaledHeight) / 2;
    
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, targetSize, targetSize);
    ctx.drawImage(imageElement, dx, dy, scaledWidth, scaledHeight);
    
    // Get pixel data
    const imageData = ctx.getImageData(0, 0, targetSize, targetSize);
    const pixels = imageData.data;
    
    // Convert to NCHW format with normalization
    const float32Data = new Float32Array(3 * targetSize * targetSize);
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    
    for (let y = 0; y < targetSize; y++) {
      for (let x = 0; x < targetSize; x++) {
        const pixelIdx = (y * targetSize + x) * 4;
        for (let c = 0; c < 3; c++) {
          const normalized = ((pixels[pixelIdx + c] / 255.0) - mean[c]) / std[c];
          float32Data[c * targetSize * targetSize + y * targetSize + x] = normalized;
        }
      }
    }
    
    this.profiler.preprocess = performance.now() - start;
    return float32Data;
  }

  async classify(imageElement) {
    const start = performance.now();
    
    // Preprocess
    const inputData = await this.preprocess(imageElement);
    
    // Create input tensor
    const inputTensor = new ort.Tensor(
      'float32',
      inputData,
      [1, 3, 224, 224]
    );
    
    // Run inference
    const feeds = { [this.session.inputNames[0]]: inputTensor };
    const results = await this.session.run(feeds);
    
    this.profiler.inference = performance.now() - start - this.profiler.preprocess;
    
    // Postprocess
    const output = results[this.session.outputNames[0]];
    const probabilities = this.softmax(output.data);
    
    const top5 = this.getTopK(probabilities, 5);
    const predictions = top5.map(idx => ({
      label: this.labels[idx] || `Class ${idx}`,
      probability: probabilities[idx],
      index: idx,
    }));
    
    this.profiler.postprocess = performance.now() - start;
    
    return {
      predictions,
      profiler: { ...this.profiler },
    };
  }

  softmax(logits) {
    const maxLogit = Math.max(...logits);
    const expLogits = logits.map(l => Math.exp(l - maxLogit));
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    return expLogits.map(e => e / sumExp);
  }

  getTopK(arr, k) {
    const indices = arr.map((val, idx) => idx);
    indices.sort((a, b) => arr[b] - arr[a]);
    return indices.slice(0, k);
  }
}
```

### ONNX Model Conversion Pipeline

Converting models from PyTorch or TensorFlow to ONNX for web deployment:

```python
# PyTorch to ONNX conversion
import torch
import torchvision.models as models

# Load pre-trained model
model = models.resnet50(pretrained=True)
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "resnet50.onnx",
    opset_version=17,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'},
    },
    do_constant_folding=True,
)

# Quantize for web deployment
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    "resnet50.onnx",
    "resnet50_quantized.onnx",
    weight_type=QuantType.QUInt8,
)
```

```python
# TensorFlow/Keras to ONNX
import tf2onnx
import tensorflow as tf

model = tf.keras.applications.MobileNetV2(weights='imagenet')

spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
output_path = "./mobilenetv2.onnx"

model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    output_path=output_path,
    opset=17,
)
```

### Performance Benchmarks

| Model | ONNX (WebGPU) | ONNX (WASM) | TF.js (WebGPU) | TF.js (WASM) |
|-------|--------------|-------------|----------------|--------------|
| ResNet-50 | 12ms | 150ms | 15ms | 180ms |
| YOLOv8-nano | 8ms | 95ms | 10ms | 110ms |
| BERT-base (128) | 6ms | 85ms | 8ms | 100ms |
| Whisper-small | 120ms | 1800ms | 140ms | N/A |
| T5-small | 8ms | 110ms | 10ms | 130ms |

## MediaPipe

MediaPipe (by Google) specializes in vision and media processing pipelines with WebGPU acceleration.

```javascript
// MediaPipe face detection
import { FaceDetector, FilesetResolver } from '@mediapipe/tasks-vision';

async function initializeFaceDetection() {
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm'
  );
  
  const faceDetector = await FaceDetector.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/face_detector.task',
      delegate: 'GPU',
    },
    runningMode: 'VIDEO',
    minDetectionConfidence: 0.5,
  });
  
  return faceDetector;
}

// Real-time face detection
async function runFaceDetection(videoElement, canvasElement) {
  const faceDetector = await initializeFaceDetection();
  const ctx = canvasElement.getContext('2d');
  
  function detectFrame() {
    const detections = faceDetector.detectForVideo(videoElement, Date.now());
    
    // Clear canvas
    ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    // Draw detections
    for (const detection of detections.detections) {
      const bbox = detection.boundingBox;
      ctx.strokeStyle = '#00FF00';
      ctx.lineWidth = 2;
      ctx.strokeRect(bbox.originX, bbox.originY, bbox.width, bbox.height);
      
      // Draw keypoints
      for (const keypoint of detection.keypoints) {
        ctx.fillStyle = '#00FF00';
        ctx.beginPath();
        ctx.arc(keypoint.x, keypoint.y, 3, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
    
    requestAnimationFrame(detectFrame);
  }
  
  detectFrame();
}
```

## Transformers.js (Covered in detail in 07-Transformers-js-and-WebLLM.md)

Transformers.js enables running HuggingFace models directly in the browser. Here's a quick example:

```javascript
import { pipeline } from '@xenova/transformers';

async function runTransformers() {
  // Create a classification pipeline
  const classifier = await pipeline(
    'sentiment-analysis',
    'Xenova/bert-base-uncased-sst2',
    { quantized: true }  // Use int8 quantized version for smaller size
  );
  
  const result = await classifier('I love using AI in the browser!');
  console.log(result);
  // [{ label: 'POSITIVE', score: 0.9998 }]
}
```

## WebLLM

WebLLM specializes in running large language models via WebGPU. See detailed coverage in 07-Transformers-js-and-WebLLM.md.

## Framework Selection Guide

### Choose TensorFlow.js when:
- You need to train models in the browser
- You want to use Keras-style high-level API
- You're working with TensorFlow ecosystem models
- You need comprehensive data loading and preprocessing
- You want model visualization tools

### Choose ONNX Runtime Web when:
- You need cross-framework model support (PyTorch, TF, scikit-learn)
- You want the best inference performance
- You need quantized model support
- You're deploying from existing ONNX pipelines
- You want hardware acceleration via WebGPU/WebNN

### Choose Transformers.js when:
- You're using HuggingFace models
- You need natural language processing
- You want the simplest API for transformers
- You need quantized model support

### Choose MediaPipe when:
- You need real-time vision processing
- You're building video/AR applications
- You need pre-built vision pipelines
- You want Google's optimized implementations

### Choose WebLLM when:
- You need to run LLMs in the browser
- You want WebGPU-optimized inference
- You need chatbot functionality
- You're working with GGUF model format

## Combining Frameworks

In complex applications, you might combine multiple frameworks:

```javascript
// Hybrid approach: use the best tool for each task
class HybridAIApp {
  constructor() {
    this.onnxClassifier = null;
    this.tfjsModel = null;
    this.mediaPipe = null;
  }
  
  async initialize() {
    // ONNX for image classification (best performance)
    this.onnxClassifier = new ONNXImageClassifier();
    
    // TF.js for custom model (needs training support)
    await tf.setBackend('webgpu');
    this.tfjsModel = await tf.loadLayersModel('model.json');
    
    // MediaPipe for face detection
    this.mediaPipe = await initializeFaceDetection();
  }
  
  async processVideoFrame(video) {
    // Face detection via MediaPipe
    const faces = this.mediaPipe.detectForVideo(video, Date.now());
    
    // For each face, classify emotion via ONNX
    for (const face of faces.detections) {
      const faceCrop = extractFace(video, face.boundingBox);
      const emotion = await this.onnxClassifier.classify(faceCrop);
      
      // Track via TF.js custom model
      const features = extractFeatures(face, emotion);
      const prediction = this.tfjsModel.predict(tf.tensor2d([features]));
      
      // Use prediction for downstream tasks
      updateUI(face, emotion, prediction);
    }
  }
}
```

## Performance Optimization Best Practices

### 1. Model Optimization

```javascript
// Model optimization techniques for all frameworks

// a) Quantization
// Reduce model precision: float32 → float16 or int8
// Typical size reduction: 50-75%
// Typical accuracy loss: <1% for int8, ~0% for float16

// b) Model pruning
// Remove near-zero weights
// Can reduce size by 30-50% with minimal accuracy loss

// c) Operator fusion
// Combine consecutive operations
// ONNX Runtime does this automatically

// d) Shape optimization
// Use fixed shapes when possible
// Avoid dynamic shapes (slower)
```

### 2. Memory Management

```javascript
// Memory management across frameworks

// TensorFlow.js
tf.tidy(() => {
  // Tensors created here are automatically cleaned
  const a = tf.tensor([1, 2, 3]);
  const b = tf.tensor([4, 5, 6]);
  return a.mul(b);
});

// ONNX Runtime Web
// Reuse tensors
const inputTensor = new ort.Tensor('float32', new Float32Array(size), [1, 3, 224, 224]);
// Update data in place
inputTensor.data.set(newData);
await session.run({ 'input': inputTensor });

// Manual cleanup
const tensor = new ort.Tensor(...);
session.release(); // Release session resources
```

### 3. Batch Processing

```javascript
// Batch multiple inputs for better throughput
async function batchClassify(classifier, images, batchSize = 4) {
  const results = [];
  
  for (let i = 0; i < images.length; i += batchSize) {
    const batch = images.slice(i, i + batchSize);
    const batchPromises = batch.map(img => classifier.classify(img));
    const batchResults = await Promise.all(batchPromises);
    results.push(...batchResults);
  }
  
  return results;
}
```

### 4. Web Workers for Non-blocking Inference

```javascript
// main.js
const inferenceWorker = new Worker('inference-worker.js');

inferenceWorker.onmessage = (event) => {
  const { result, requestId } = event.data;
  // Handle result
  pendingRequests.get(requestId).resolve(result);
};

function runInference(input) {
  return new Promise((resolve) => {
    const requestId = generateId();
    pendingRequests.set(requestId, { resolve });
    inferenceWorker.postMessage({ input, requestId });
  });
}
```

## Future Trends

### Framework Convergence
- More frameworks standardizing on ONNX as intermediate format
- WebGPU becoming the universal hardware acceleration backend
- Shared tensor operations across frameworks

### Emerging Frameworks
- **Apache TVM Web Runtime**: Brings TVM's compilation optimizations to the browser
- **OpenAI's WebGPT**: Potentially a new web-native framework
- **Apple's WebCoreML**: Apple's ML framework for Safari

### Cross-Framework Interop
- Standard tensor sharing between frameworks
- Common model format for web deployment
- Unified performance benchmarking

## Conclusion

The JavaScript ML framework ecosystem in 2026 offers mature, production-ready options for every use case. TensorFlow.js remains the most comprehensive framework with training support, ONNX Runtime Web provides the best cross-framework inference performance, MediaPipe excels at vision tasks, and Transformers.js/WebLLM specialize in transformer models and LLMs.

The key to success is choosing the right framework for your specific needs — considering factors like model source, performance requirements, target hardware, and development complexity. In many cases, combining multiple frameworks for different parts of your application yields the best results.
