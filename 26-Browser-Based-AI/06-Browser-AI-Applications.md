# Browser AI Applications: Real-World Use Cases and Implementations

## Introduction

Browser-based AI has moved from experimental to production, powering a diverse range of applications across image processing, audio analysis, natural language understanding, and document handling. This document explores real-world browser AI applications with detailed case studies, architecture patterns, and complete code examples.

## Categories of Browser AI Applications

### 1. Client-Side Image Processing
### 2. Real-Time Video Processing
### 3. Audio Processing and Speech Recognition
### 4. Natural Language Processing
### 5. Large Language Models and Chatbots
### 6. Document Processing and OCR
### 7. Accessibility Tools
### 8. Privacy-Preserving Analytics

## 1. Client-Side Image Processing

### Object Detection

Real-time object detection running entirely in the browser, using ONNX Runtime Web with YOLO models:

```javascript
import * as ort from 'onnxruntime-web';

class BrowserObjectDetector {
  constructor() {
    this.session = null;
    this.anchors = this.generateAnchors();
    this.colors = [
      '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF',
      '#00FFFF', '#FF8000', '#80FF00', '#0080FF', '#FF0080',
    ];
  }

  async initialize() {
    this.session = await ort.InferenceSession.create('./models/yolov8n.onnx', {
      executionProviders: ['webgpu', 'wasm'],
    });
  }

  generateAnchors() {
    // YOLO anchor boxes for different scales
    const strides = [8, 16, 32];
    const anchors = [];
    
    for (const stride of strides) {
      const featureSize = Math.ceil(640 / stride);
      for (let y = 0; y < featureSize; y++) {
        for (let x = 0; x < featureSize; x++) {
          anchors.push({
            cx: (x + 0.5) / featureSize,
            cy: (y + 0.5) / featureSize,
            stride: stride,
          });
        }
      }
    }
    
    return anchors;
  }

  async detect(imageElement) {
    // Preprocess
    const inputData = this.preprocess(imageElement);
    
    // Run inference
    const inputTensor = new ort.Tensor('float32', inputData, [1, 3, 640, 640]);
    const results = await this.session.run({ 'images': inputTensor });
    
    // Postprocess
    const detections = this.postprocess(results[0].data);
    
    return detections;
  }

  preprocess(image) {
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 640;
    const ctx = canvas.getContext('2d');
    
    // Letterbox resize (maintain aspect ratio)
    const scale = Math.min(640 / image.width, 640 / image.height);
    const scaledWidth = image.width * scale;
    const scaledHeight = image.height * scale;
    const dx = (640 - scaledWidth) / 2;
    const dy = (640 - scaledHeight) / 2;
    
    ctx.fillStyle = 'gray';
    ctx.fillRect(0, 0, 640, 640);
    ctx.drawImage(image, dx, dy, scaledWidth, scaledHeight);
    
    // Get pixel data and normalize
    const imageData = ctx.getImageData(0, 0, 640, 640);
    const pixels = imageData.data;
    const data = new Float32Array(3 * 640 * 640);
    
    for (let i = 0; i < 640 * 640; i++) {
      const pi = i * 4;
      data[i] = pixels[pi] / 255.0;           // R
      data[640 * 640 + i] = pixels[pi + 1] / 255.0;  // G
      data[2 * 640 * 640 + i] = pixels[pi + 2] / 255.0; // B
    }
    
    return data;
  }

  postprocess(outputData) {
    const numClasses = 80;
    const numPredictions = 8400; // YOLOv8-nano output size
    const boxSize = 4 + numClasses + 1; // xywh + cls_probs + obj_conf
    
    const boxes = [];
    
    for (let i = 0; i < numPredictions; i++) {
      const offset = i * boxSize;
      const confidence = outputData[offset + 4];
      
      if (confidence > 0.5) {
        // Find best class
        let maxClassProb = 0;
        let bestClass = -1;
        for (let c = 0; c < numClasses; c++) {
          const classProb = outputData[offset + 5 + c];
          if (classProb > maxClassProb) {
            maxClassProb = classProb;
            bestClass = c;
          }
        }
        
        const score = confidence * maxClassProb;
        if (score > 0.5) {
          boxes.push({
            x: outputData[offset] - outputData[offset + 2] / 2,
            y: outputData[offset + 1] - outputData[offset + 3] / 2,
            width: outputData[offset + 2],
            height: outputData[offset + 3],
            class: bestClass,
            score: score,
          });
        }
      }
    }
    
    // Non-maximum suppression
    return this.nms(boxes, 0.45);
  }

  nms(boxes, iouThreshold) {
    boxes.sort((a, b) => b.score - a.score);
    const selected = [];
    
    while (boxes.length > 0) {
      const best = boxes.shift();
      selected.push(best);
      
      boxes = boxes.filter(box => {
        const iou = this.computeIOU(best, box);
        return iou < iouThreshold;
      });
    }
    
    return selected;
  }

  computeIOU(box1, box2) {
    const x1 = Math.max(box1.x, box2.x);
    const y1 = Math.max(box1.y, box2.y);
    const x2 = Math.min(box1.x + box1.width, box2.x + box2.width);
    const y2 = Math.min(box1.y + box1.height, box2.y + box2.height);
    
    const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const union = box1.width * box1.height + box2.width * box2.height - intersection;
    
    return intersection / union;
  }

  renderDetections(canvas, detections) {
    const ctx = canvas.getContext('2d');
    const displayWidth = canvas.width;
    const displayHeight = canvas.height;
    
    // Scale factors for letterbox
    const scale = Math.min(640 / displayWidth, 640 / displayHeight);
    
    for (const det of detections) {
      const x = (det.x - (640 - displayWidth * scale) / 2) / scale;
      const y = (det.y - (640 - displayHeight * scale) / 2) / scale;
      const w = det.width / scale;
      const h = det.height / scale;
      
      ctx.strokeStyle = this.colors[det.class % this.colors.length];
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, w, h);
      
      ctx.fillStyle = this.colors[det.class % this.colors.length];
      ctx.fillText(`Class ${det.class}: ${(det.score * 100).toFixed(0)}%`, x, y - 5);
    }
  }
}
```

### Browser-Based OCR

Client-side OCR using Transformers.js with TrOCR:

```javascript
import { pipeline } from '@xenova/transformers';

class BrowserOCR {
  constructor() {
    this.processor = null;
    this.model = null;
  }

  async initialize() {
    // Load TrOCR model for text recognition
    this.processor = await pipeline('image-to-text', 'Xenova/trocr-base-printed', {
      quantized: true,
    });
  }

  async extractText(imageElement) {
    const result = await this.processor(imageElement);
    return result[0].generated_text;
  }

  async processDocument(file) {
    // Load document image
    const imageUrl = URL.createObjectURL(file);
    const img = new Image();
    
    await new Promise((resolve) => {
      img.onload = resolve;
      img.src = imageUrl;
    });
    
    // Extract text
    const text = await this.extractText(img);
    
    // Clean up
    URL.revokeObjectURL(imageUrl);
    
    return text;
  }
}
```

## 2. Real-Time Video Processing

### Pose Estimation with MediaPipe

```javascript
import { PoseLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

class BrowserPoseEstimator {
  constructor() {
    this.poseLandmarker = null;
    this.lastFrameTime = 0;
    this.fpsHistory = [];
  }

  async initialize() {
    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm'
    );
    
    this.poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task',
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
  }

  estimatePose(videoElement) {
    const startTime = performance.now();
    const result = this.poseLandmarker.detectForVideo(videoElement, startTime);
    
    // Track FPS
    const fps = 1000 / (startTime - this.lastFrameTime);
    this.fpsHistory.push(fps);
    if (this.fpsHistory.length > 60) this.fpsHistory.shift();
    this.lastFrameTime = startTime;
    
    if (result.landmarks && result.landmarks.length > 0) {
      return {
        landmarks: result.landmarks[0],
        worldLandmarks: result.worldLandmarks[0],
        fps: this.fpsHistory.reduce((a, b) => a + b, 0) / this.fpsHistory.length,
      };
    }
    
    return null;
  }

  drawPose(canvas, pose) {
    if (!pose) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Draw connections
    const connections = [
      [11, 12], [11, 23], [12, 24], [23, 24], // Torso
      [11, 13], [13, 15], [12, 14], [14, 16], // Arms
      [23, 25], [25, 27], [24, 26], [26, 28], // Legs
      [15, 17], [15, 19], [15, 21], [16, 18], [16, 20], [16, 22], // Hands/Feet
    ];
    
    ctx.strokeStyle = '#00FF00';
    ctx.lineWidth = 3;
    
    for (const [start, end] of connections) {
      const p1 = pose.landmarks[start];
      const p2 = pose.landmarks[end];
      
      if (p1 && p2 && p1.visibility > 0.5 && p2.visibility > 0.5) {
        ctx.beginPath();
        ctx.moveTo(p1.x * width, p1.y * height);
        ctx.lineTo(p2.x * width, p2.y * height);
        ctx.stroke();
      }
    }
    
    // Draw landmarks
    for (const landmark of pose.landmarks) {
      if (landmark.visibility > 0.5) {
        ctx.fillStyle = '#FF0000';
        ctx.beginPath();
        ctx.arc(landmark.x * width, landmark.y * height, 5, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
  }

  getFPS() {
    if (this.fpsHistory.length === 0) return 0;
    return this.fpsHistory.reduce((a, b) => a + b, 0) / this.fpsHistory.length;
  }
}
```

### Background Removal

```javascript
import * as ort from 'onnxruntime-web';

class BackgroundRemover {
  constructor() {
    this.session = null;
  }

  async initialize() {
    this.session = await ort.InferenceSession.create(
      './models/modnet.onnx',
      { executionProviders: ['webgpu', 'wasm'] }
    );
  }

  async removeBackground(videoElement) {
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoElement, 0, 0);
    
    // Preprocess
    const inputData = this.preprocess(canvas);
    
    // Run inference
    const inputTensor = new ort.Tensor('float32', inputData, [1, 3, 256, 256]);
    const results = await this.session.run({ 'input': inputTensor });
    
    // Postprocess matting
    const matte = this.postprocess(results[0].data, canvas.width, canvas.height);
    
    // Apply to original image
    const outputCanvas = document.createElement('canvas');
    outputCanvas.width = canvas.width;
    outputCanvas.height = canvas.height;
    const outputCtx = outputCanvas.getContext('2d');
    
    // Draw original
    outputCtx.drawImage(canvas, 0, 0);
    
    // Apply matte as alpha
    const imageData = outputCtx.getImageData(0, 0, canvas.width, canvas.height);
    const pixels = imageData.data;
    
    for (let i = 0; i < pixels.length / 4; i++) {
      const x = i % canvas.width;
      const y = Math.floor(i / canvas.width);
      const matteX = Math.round(x * matte.width / canvas.width);
      const matteY = Math.round(y * matte.height / canvas.height);
      pixels[i * 4 + 3] = matte.data[matteY * matte.width + matteX] * 255;
    }
    
    outputCtx.putImageData(imageData, 0, 0);
    
    return outputCanvas;
  }

  preprocess(canvas) {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 256;
    tempCanvas.height = 256;
    const ctx = tempCanvas.getContext('2d');
    ctx.drawImage(canvas, 0, 0, 256, 256);
    
    const imageData = ctx.getImageData(0, 0, 256, 256);
    const pixels = imageData.data;
    const data = new Float32Array(3 * 256 * 256);
    
    for (let i = 0; i < 256 * 256; i++) {
      const pi = i * 4;
      // Normalize to [-1, 1]
      data[i] = (pixels[pi] / 255.0) * 2 - 1;
      data[256 * 256 + i] = (pixels[pi + 1] / 255.0) * 2 - 1;
      data[2 * 256 * 256 + i] = (pixels[pi + 2] / 255.0) * 2 - 1;
    }
    
    return data;
  }

  postprocess(outputData, origWidth, origHeight) {
    const matte = new Float32Array(256 * 256);
    
    // Sigmoid
    for (let i = 0; i < 256 * 256; i++) {
      matte[i] = 1 / (1 + Math.exp(-outputData[i]));
    }
    
    return { data: matte, width: 256, height: 256 };
  }
}

// Real-time background removal in video conference
class VideoConferenceEffect {
  constructor(videoElement, canvasElement) {
    this.video = videoElement;
    this.canvas = canvasElement;
    this.remover = new BackgroundRemover();
    this.isRunning = false;
  }

  async start() {
    this.isRunning = true;
    await this.remover.initialize();
    this.processFrame();
  }

  stop() {
    this.isRunning = false;
  }

  async processFrame() {
    if (!this.isRunning) return;
    
    const result = await this.remover.removeBackground(this.video);
    
    const ctx = this.canvas.getContext('2d');
    ctx.drawImage(result, 0, 0, this.canvas.width, this.canvas.height);
    
    requestAnimationFrame(() => this.processFrame());
  }
}
```

## 3. Audio Processing and Speech Recognition

### Browser-Based Whisper Transcription

```javascript
import { pipeline } from '@xenova/transformers';

class BrowserWhisper {
  constructor() {
    this.transcriber = null;
    this.audioContext = null;
    this.mediaRecorder = null;
    this.audioChunks = [];
  }

  async initialize() {
    this.transcriber = await pipeline(
      'automatic-speech-recognition',
      'Xenova/whisper-small',
      { quantized: true }
    );
  }

  async transcribeFile(audioFile) {
    const audioData = await this.loadAudioFile(audioFile);
    const result = await this.transcriber(audioData, {
      language: 'english',
      task: 'transcribe',
      return_timestamps: true,
    });
    
    return result;
  }

  async transcribeMicrophone() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    
    this.audioContext = new AudioContext();
    const source = this.audioContext.createMediaStreamSource(stream);
    
    // Set up recording
    this.mediaRecorder = new MediaRecorder(stream);
    this.audioChunks = [];
    
    return new Promise((resolve) => {
      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          this.audioChunks.push(event.data);
        }
      };
      
      this.mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
        const audioData = await this.loadAudioBlob(audioBlob);
        const result = await this.transcriber(audioData, {
          language: 'english',
          task: 'transcribe',
          chunk_length_s: 30,
        });
        
        resolve(result);
      };
      
      this.mediaRecorder.start();
    });
  }

  startRecording() {
    this.audioChunks = [];
    this.mediaRecorder.start();
  }

  stopRecording() {
    this.mediaRecorder.stop();
  }

  async loadAudioFile(file) {
    const arrayBuffer = await file.arrayBuffer();
    const audioContext = new AudioContext();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    return audioBuffer.getChannelData(0);
  }

  async loadAudioBlob(blob) {
    const arrayBuffer = await blob.arrayBuffer();
    return this.decodeAudioData(arrayBuffer);
  }

  async decodeAudioData(arrayBuffer) {
    this.audioContext = this.audioContext || new AudioContext();
    const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
    
    // Convert to mono if needed
    const channelData = audioBuffer.numberOfChannels > 1
      ? this.convertToMono(audioBuffer)
      : audioBuffer.getChannelData(0);
    
    // Resample to 16kHz (Whisper expects 16kHz)
    return this.resample(channelData, audioBuffer.sampleRate, 16000);
  }

  convertToMono(audioBuffer) {
    const numChannels = audioBuffer.numberOfChannels;
    const length = audioBuffer.length;
    const monoData = new Float32Array(length);
    
    for (let i = 0; i < length; i++) {
      let sum = 0;
      for (let c = 0; c < numChannels; c++) {
        sum += audioBuffer.getChannelData(c)[i];
      }
      monoData[i] = sum / numChannels;
    }
    
    return monoData;
  }

  resample(audioData, originalSampleRate, targetSampleRate) {
    if (originalSampleRate === targetSampleRate) return audioData;
    
    const ratio = targetSampleRate / originalSampleRate;
    const newLength = Math.round(audioData.length * ratio);
    const resampled = new Float32Array(newLength);
    
    for (let i = 0; i < newLength; i++) {
      const position = i / ratio;
      const index = Math.floor(position);
      const fraction = position - index;
      
      if (index + 1 < audioData.length) {
        resampled[i] = audioData[index] * (1 - fraction) + audioData[index + 1] * fraction;
      } else {
        resampled[i] = audioData[index];
      }
    }
    
    return resampled;
  }
}

// Usage: Real-time transcription UI
class TranscriptionUI {
  constructor() {
    this.whisper = new BrowserWhisper();
    this.transcriptElement = document.getElementById('transcript');
    this.statusElement = document.getElementById('status');
  }

  async initialize() {
    this.statusElement.textContent = 'Loading Whisper model...';
    await this.whisper.initialize();
    this.statusElement.textContent = 'Ready. Click Start to record.';
  }

  async startRecording() {
    this.statusElement.textContent = 'Recording...';
    
    try {
      const result = await this.whisper.transcribeMicrophone();
      
      this.transcriptElement.innerHTML = result.chunks.map(chunk => `
        <p><strong>[${this.formatTime(chunk.timestamp[0])} - ${this.formatTime(chunk.timestamp[1])}]</strong> ${chunk.text}</p>
      `).join('');
      
      this.statusElement.textContent = 'Transcription complete';
    } catch (error) {
      this.statusElement.textContent = `Error: ${error.message}`;
    }
  }

  formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }
}
```

### Real-Time Audio Processing

```javascript
class AudioWorkletProcessor {
  constructor() {
    this.audioContext = null;
    this.analyserNode = null;
    this.isProcessing = false;
    this.sampleRate = 16000;
  }

  async initialize() {
    this.audioContext = new AudioContext({ sampleRate: this.sampleRate });
    
    // Get microphone
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const source = this.audioContext.createMediaStreamSource(stream);
    
    // Create analyser for VAD (Voice Activity Detection)
    this.analyserNode = this.audioContext.createAnalyser();
    this.analyserNode.fftSize = 512;
    source.connect(this.analyserNode);
    
    // Set up audio worklet for real-time processing
    await this.audioContext.audioWorklet.addModule('audio-processor.js');
    const workletNode = new AudioWorkletNode(this.audioContext, 'ai-processor');
    source.connect(workletNode);
    workletNode.connect(this.audioContext.destination);
    
    workletNode.port.onmessage = (event) => {
      this.handleProcessedAudio(event.data);
    };
    
    this.isProcessing = true;
  }

  detectVoiceActivity() {
    const dataArray = new Uint8Array(this.analyserNode.frequencyBinCount);
    this.analyserNode.getByteTimeDomainData(dataArray);
    
    // Simple energy-based VAD
    let sum = 0;
    for (let i = 0; i < dataArray.length; i++) {
      sum += Math.abs(dataArray[i] - 128);
    }
    const energy = sum / dataArray.length;
    
    return energy > 15; // Threshold for voice activity
  }

  handleProcessedAudio(data) {
    // Handle processed audio data (e.g., noise suppression, echo cancellation)
    console.log('Processed audio chunk:', data.length, 'samples');
  }
}

// Audio worklet processor (audio-processor.js)
class AIProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(1024);
    this.bufferIndex = 0;
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];
    
    if (input.length > 0 && input[0].length > 0) {
      const inputChannel = input[0];
      const outputChannel = output[0];
      
      // Simple pass-through with noise gate
      for (let i = 0; i < inputChannel.length; i++) {
        const threshold = 0.01;
        outputChannel[i] = Math.abs(inputChannel[i]) > threshold ? inputChannel[i] : 0;
      }
      
      // Send to main thread for AI processing
      this.buffer.set(inputChannel, this.bufferIndex);
      this.bufferIndex += inputChannel.length;
      
      if (this.bufferIndex >= 1024) {
        this.port.postMessage(this.buffer.slice(0, 1024));
        this.bufferIndex = 0;
      }
    }
    
    return true;
  }
}

registerProcessor('ai-processor', AIProcessor);
```

## 4. Natural Language Processing

### Client-Side Content Moderation

```javascript
import { pipeline } from '@xenova/transformers';

class ContentModerator {
  constructor() {
    this.classifier = null;
    this.thresholds = {
      toxicity: 0.8,
      hate: 0.7,
      spam: 0.9,
    };
  }

  async initialize() {
    this.classifier = await pipeline(
      'text-classification',
      'Xenova/toxic-bert',
      { quantized: true }
    );
  }

  async moderate(text) {
    const results = await this.classifier(text, {
      topk: null, // Return all categories
    });
    
    const flags = {};
    for (const result of results) {
      const category = result.label.toLowerCase();
      const score = result.score;
      
      if (score > (this.thresholds[category] || 0.8)) {
        flags[category] = {
          flagged: true,
          score: score,
          threshold: this.thresholds[category] || 0.8,
        };
      }
    }
    
    return {
      text: text,
      isFlagged: Object.keys(flags).length > 0,
      flags: flags,
      allScores: results,
    };
  }
}

// Usage in a chat application
class ChatModerator {
  constructor() {
    this.moderator = new ContentModerator();
  }

  async initialize() {
    await this.moderator.initialize();
  }

  async processMessage(message) {
    const result = await this.moderator.moderate(message.text);
    
    if (result.isFlagged) {
      return {
        allowed: false,
        message: message,
        flags: result.flags,
        suggestion: this.generateSuggestion(result.flags),
      };
    }
    
    return {
      allowed: true,
      message: message,
    };
  }

  generateSuggestion(flags) {
    if (flags.toxicity) {
      return 'Please rephrase your message to be more respectful.';
    }
    if (flags.hate) {
      return 'Hate speech is not allowed. Please review our community guidelines.';
    }
    return 'Your message was flagged. Please revise and try again.';
  }
}
```

### Sentiment Analysis for Customer Feedback

```javascript
import { pipeline } from '@xenova/transformers';

class SentimentAnalyzer {
  constructor() {
    this.analyzer = null;
    this.history = [];
  }

  async initialize() {
    this.analyzer = await pipeline(
      'sentiment-analysis',
      'Xenova/distilbert-base-uncased-sst2',
      { quantized: true }
    );
  }

  async analyze(text) {
    const result = await this.analyzer(text);
    
    const analysis = {
      text,
      label: result[0].label,
      score: result[0].score,
      timestamp: Date.now(),
      sentiment: result[0].label === 'POSITIVE' ? 1 : -1,
    };
    
    this.history.push(analysis);
    
    return analysis;
  }

  async analyzeBatch(texts) {
    const results = await this.analyzer(texts);
    
    return texts.map((text, i) => ({
      text,
      label: results[i].label,
      score: results[i].score,
      timestamp: Date.now(),
    }));
  }

  getStats() {
    if (this.history.length === 0) return null;
    
    const positive = this.history.filter(h => h.label === 'POSITIVE').length;
    const negative = this.history.filter(h => h.label === 'NEGATIVE').length;
    const avgScore = this.history.reduce((sum, h) => sum + h.score, 0) / this.history.length;
    
    return {
      total: this.history.length,
      positive,
      negative,
      positivePercent: (positive / this.history.length * 100).toFixed(1),
      negativePercent: (negative / this.history.length * 100).toFixed(1),
      averageScore: avgScore.toFixed(3),
    };
  }
}
```

## 5. Document Processing

### In-Browser PDF Analysis

```javascript
import { pipeline } from '@xenova/transformers';

class DocumentProcessor {
  constructor() {
    this.summarizer = null;
    this.classifier = null;
  }

  async initialize() {
    this.summarizer = await pipeline(
      'summarization',
      'Xenova/distilbart-cnn-6-6',
      { quantized: true }
    );
    
    this.classifier = await pipeline(
      'zero-shot-classification',
      'Xenova/bart-large-mnli',
      { quantized: true }
    );
  }

  async processDocument(text, categories) {
    // Classify document type
    const classification = await this.classifier(text, categories);
    
    // Summarize
    const maxChunkSize = 1024;
    let summary = '';
    
    if (text.length > maxChunkSize) {
      const chunks = this.chunkText(text, maxChunkSize);
      const summaries = [];
      
      for (const chunk of chunks) {
        const result = await this.summarizer(chunk, {
          max_length: 150,
          min_length: 40,
        });
        summaries.push(result[0].summary_text);
      }
      
      // Summarize the summaries
      if (summaries.length > 1) {
        const finalSummary = await this.summarizer(
          summaries.join(' '),
          { max_length: 250, min_length: 60 }
        );
        summary = finalSummary[0].summary_text;
      } else {
        summary = summaries[0];
      }
    } else {
      const result = await this.summarizer(text, {
        max_length: 150,
        min_length: 40,
      });
      summary = result[0].summary_text;
    }
    
    return {
      summary,
      classification: {
        labels: classification.labels,
        scores: classification.scores,
        topLabel: classification.labels[0],
        topScore: classification.scores[0],
      },
      metadata: {
        length: text.length,
        wordCount: text.split(/\s+/).length,
      },
    };
  }

  chunkText(text, maxSize) {
    const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
    const chunks = [];
    let currentChunk = '';
    
    for (const sentence of sentences) {
      if ((currentChunk + sentence).length > maxSize) {
        chunks.push(currentChunk.trim());
        currentChunk = sentence;
      } else {
        currentChunk += sentence;
      }
    }
    
    if (currentChunk.trim()) {
      chunks.push(currentChunk.trim());
    }
    
    return chunks;
  }
}
```

## 6. Accessibility Tools

### Real-Time Captioning

```javascript
class LiveCaptions {
  constructor() {
    this.whisper = null;
    this.captionsElement = null;
    this.isRunning = false;
    this.currentCaption = '';
    this.fullTranscript = [];
  }

  async initialize() {
    const { pipeline } = await import('@xenova/transformers');
    this.whisper = await pipeline(
      'automatic-speech-recognition',
      'Xenova/whisper-tiny',
      { quantized: true }
    );
  }

  async start(captionsElementId) {
    this.captionsElement = document.getElementById(captionsElementId);
    this.isRunning = true;
    
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream, {
      mimeType: 'audio/webm;codecs=opus',
    });
    
    let audioChunks = [];
    
    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        audioChunks.push(event.data);
      }
    };
    
    mediaRecorder.onstop = async () => {
      if (!this.isRunning) return;
      
      const blob = new Blob(audioChunks, { type: 'audio/webm' });
      audioChunks = [];
      
      try {
        const result = await this.transcribeBlob(blob);
        const text = result.text.toLowerCase();
        
        if (text && text !== '[blank_audio]') {
          this.currentCaption = text;
          const words = text.split(' ');
          this.fullTranscript.push(...words);
          
          // Show last 10 words
          const displayText = this.fullTranscript.slice(-10).join(' ');
          this.captionsElement.textContent = displayText;
          
          // Auto-scroll
          this.captionsElement.scrollTop = this.captionsElement.scrollHeight;
        }
      } catch (error) {
        console.error('Transcription error:', error);
      }
      
      // Continue recording next chunk
      if (this.isRunning) {
        mediaRecorder.start();
        setTimeout(() => {
          if (mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
          }
        }, 3000); // 3-second chunks
      }
    };
    
    // Start first chunk
    mediaRecorder.start();
    setTimeout(() => {
      if (mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
      }
    }, 3000);
  }

  async transcribeBlob(blob) {
    const arrayBuffer = await blob.arrayBuffer();
    const audioContext = new AudioContext();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    const audioData = audioBuffer.getChannelData(0);
    
    return await this.whisper(audioData, {
      language: 'english',
      task: 'transcribe',
      no_speech_threshold: 0.6,
    });
  }

  stop() {
    this.isRunning = false;
  }
}
```

### Screen Reader Enhancement

```javascript
class AIScreenReader {
  constructor() {
    this.imageCaptioner = null;
    this.ocrProcessor = null;
  }

  async initialize() {
    const { pipeline } = await import('@xenova/transformers');
    
    this.imageCaptioner = await pipeline(
      'image-to-text',
      'Xenova/vit-gpt2-coco-en',
      { quantized: true }
    );
  }

  async describeImage(imageElement) {
    const result = await this.imageCaptioner(imageElement);
    return result[0].generated_text;
  }

  async describePage() {
    const descriptions = [];
    
    // Describe all images on the page
    const images = document.querySelectorAll('img:not([aria-hidden])');
    for (const img of images) {
      if (!img.alt) {
        try {
          const description = await this.describeImage(img);
          img.alt = description;
          descriptions.push({ element: img, description });
        } catch (e) {
          console.warn('Failed to describe image:', e);
        }
      }
    }
    
    return descriptions;
  }
}
```

## 7. Privacy-Preserving Analytics

### Client-Side Data Analysis

```javascript
class PrivacyPreservingAnalytics {
  constructor() {
    this.localData = new Map();
    this.analyticsResults = null;
  }

  trackEvent(category, action, metadata = {}) {
    if (!this.localData.has(category)) {
      this.localData.set(category, []);
    }
    
    this.localData.get(category).push({
      action,
      metadata,
      timestamp: Date.now(),
    });
    
    // Run local analysis
    this.analyzeLocally();
  }

  analyzeLocally() {
    this.analyticsResults = {
      events: {},
      sessions: this.countSessions(),
      userEngagement: this.calculateEngagement(),
    };
    
    for (const [category, events] of this.localData) {
      this.analyticsResults.events[category] = {
        total: events.length,
        unique: new Set(events.map(e => e.action)).size,
        frequency: events.length / this.getTimeSpan(events),
      };
    }
  }

  countSessions() {
    // Simple session detection (30-min timeout)
    const sessions = [];
    let currentSession = [];
    
    const allEvents = Array.from(this.localData.values())
      .flat()
      .sort((a, b) => a.timestamp - b.timestamp);
    
    for (const event of allEvents) {
      if (currentSession.length === 0) {
        currentSession.push(event);
      } else {
        const lastEvent = currentSession[currentSession.length - 1];
        const timeDiff = event.timestamp - lastEvent.timestamp;
        
        if (timeDiff < 30 * 60 * 1000) { // 30 minutes
          currentSession.push(event);
        } else {
          sessions.push(currentSession);
          currentSession = [event];
        }
      }
    }
    
    if (currentSession.length > 0) {
      sessions.push(currentSession);
    }
    
    return {
      count: sessions.length,
      avgDuration: sessions.reduce((sum, s) => {
        return sum + (s[s.length - 1].timestamp - s[0].timestamp);
      }, 0) / sessions.length,
    };
  }

  calculateEngagement() {
    // Calculate engagement score based on event frequency and recency
    const allEvents = Array.from(this.localData.values()).flat();
    const now = Date.now();
    const weekAgo = now - 7 * 24 * 60 * 60 * 1000;
    
    const recentEvents = allEvents.filter(e => e.timestamp > weekAgo);
    
    return {
      dailyActive: recentEvents.length / 7,
      totalEvents: allEvents.length,
      recencyScore: recentEvents.length / Math.max(allEvents.length, 1),
    };
  }

  getTimeSpan(events) {
    if (events.length < 2) return 1;
    const times = events.map(e => e.timestamp).sort((a, b) => a - b);
    return Math.max(times[times.length - 1] - times[0], 1);
  }

  generateReport() {
    // All data stays on device - no server upload needed
    return {
      timestamp: Date.now(),
      isLocalOnly: true,
      data: this.analyticsResults,
    };
  }
}
```

## Performance Considerations

### Real-Time Processing Requirements

| Application | Target FPS | Max Latency | Model Size |
|-------------|-----------|-------------|------------|
| Object Detection | 30 | 100ms | 3-10MB |
| Pose Estimation | 30 | 100ms | 5-15MB |
| Background Removal | 30 | 50ms | 2-5MB |
| Speech Recognition | Real-time | 500ms | 150-300MB |
| Image Classification | 60 | 50ms | 5-100MB |
| OCR | N/A | 1000ms | 200-500MB |

### Optimization Strategies

1. **Frame Skipping**: Process every Nth frame for video
2. **Resolution Scaling**: Process at lower resolution
3. **Model Quantization**: Use int8/float16 models
4. **Web Workers**: Offload inference to background threads
5. **Hardware Acceleration**: Use WebGPU/WebNN when available
6. **Caching**: Cache model results for similar inputs
7. **Streaming**: Process audio/video in chunks

## Case Studies

### Case Study 1: Browser-Based Video Conferencing
A video conferencing platform implemented browser-based background removal, pose tracking for AR effects, and real-time captioning. Results:
- 40% reduction in server costs (no server-side video processing)
- 60% improvement in privacy perception
- 95% accuracy for background removal at 30fps on modern GPUs

### Case Study 2: Client-Side Document Processing
A legal tech company deployed browser-based OCR and document classification for sensitive legal documents:
- Zero data leaves the device
- 99% accuracy on structured documents
- Processing time reduced from 5s (server) to 2s (browser with GPU)

### Case Study 3: Privacy-First Analytics Platform
An analytics company built a browser-only analytics system:
- All user behavior data stays on device
- Aggregate reports generated locally and anonymized
- GDPR/CCPA compliance achieved by design
- 100% data sovereignty for enterprise customers

## Conclusion

Browser-based AI applications have reached production maturity across a wide range of use cases. From real-time video processing to speech recognition and document analysis, the browser now serves as a capable AI runtime that offers significant advantages in privacy, latency, and cost.

The key to successful browser AI applications is choosing the right level of abstraction (WebGPU for performance, frameworks for productivity), optimizing for the target hardware, and implementing graceful degradation for devices with limited capabilities.
