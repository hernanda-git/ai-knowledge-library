# Multimodal AI: Tools and Frameworks — Building, Training, and Deploying

> **Comprehensive guide to the tools, frameworks, and platforms powering multimodal AI development.** From open-source libraries to enterprise platforms, this document maps the entire ecosystem.

---

## Table of Contents

1. [Foundation Model APIs](#foundation-model-apis)
2. [Open-Source Frameworks](#open-source-frameworks)
3. [Training Platforms](#training-platforms)
4. [Deployment and Serving](#deployment-and-serving)
5. [Data and Evaluation Tools](#data-and-evaluation-tools)
6. [Specialized Libraries](#specialized-libraries)
7. [Comparison Matrix](#comparison-matrix)
8. [Getting Started Guide](#getting-started-guide)

---

## Foundation Model APIs

### Google Gemini API

```python
# Google Gemini multimodal API
import google.generativeai as genai

# Initialize
genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel("gemini-3-pro")

# Image + Text
from PIL import Image
image = Image.open("photo.jpg")

response = model.generate_content([
    "What is in this image?",
    image
])
print(response.text)

# Video + Text
import video_bytes
video_file = genai.upload_file(video_bytes)

response = model.generate_content([
    "Summarize this video",
    video_file
])
print(response.text)

# Audio + Text
audio_file = genai.upload_file(open("recording.wav", "rb"))

response = model.generate_content([
    "Transcribe and summarize this audio",
    audio_file
])
print(response.text)

# Multimodal with system instruction
response = model.generate_content(
    [
        "Analyze this document and extract key metrics",
        document_image,
        "Focus on financial data in the table"
    ],
    generation_config=genai.GenerationConfig(
        temperature=0.1,
        max_output_tokens=2048
    )
)
```

### OpenAI API

```python
# OpenAI multimodal API
from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")

# GPT-5 with vision
response = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/photo.jpg",
                        "detail": "high"
                    }
                }
            ]
        }
    ],
    max_tokens=1000
)

# GPT-5 with audio (real-time)
# WebSocket connection for real-time audio
import websockets
import json

async def audio_chat(audio_stream):
    async with websockets.connect("wss://api.openai.com/v1/realtime") as ws:
        # Send audio
        await ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": audio_stream
        }))
        
        # Receive response
        response = await ws.recv()
        return json.loads(response)
```

### Anthropic Claude API

```python
# Anthropic Claude multimodal API
import anthropic

client = anthropic.Anthropic(api_key="YOUR_API_KEY")

# Claude Opus 4 with vision
with open("image.jpg", "rb") as f:
    image_data = base64.standard_b64encode(f.read()).decode("utf-8")

response = client.messages.create(
    model="claude-opus-4-20250514",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data
                    }
                },
                {
                    "type": "text",
                    "text": "Describe this image in detail."
                }
            ]
        }
    ]
)

# PDF + Text
with open("document.pdf", "rb") as f:
    pdf_data = base64.standard_b64encode(f.read()).decode("utf-8")

response = client.messages.create(
    model="claude-opus-4-20250514",
    max_tokens=4096,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": pdf_data
                    }
                },
                {
                    "type": "text",
                    "text": "Summarize the key findings in this report."
                }
            ]
        }
    ]
)
```

### Meta Llama API (Open Source)

```python
# Meta Llama 4 multimodal (via vLLM or HuggingFace)
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

# Load Llama 4 multimodal
model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Process image + text
from PIL import Image
image = Image.open("photo.jpg")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "What do you see?"}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages, 
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=500)
print(processor.decode(outputs[0], skip_special_tokens=True))
```

---

## Open-Source Frameworks

### HuggingFace Transformers

The most comprehensive framework for multimodal AI:

```python
# HuggingFace Transformers multimodal examples
from transformers import (
    # Vision-Language models
    Blip2ForConditionalGeneration,
    LlavaForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    
    # Audio models
    WhisperForConditionalGeneration,
    SeamlessM4Tv2ForSpeechToText,
    
    # Video models
    VideoLlavaForConditionalGeneration,
    
    # Processors
    AutoProcessor,
    AutoTokenizer
)

# LLaVA (Large Language and Vision Assistant)
model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

# Qwen2-VL (state-of-the-art open multimodal)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-72B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-72B-Instruct")

# Process and generate
messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "https://example.com/video.mp4"},
            {"type": "text", "text": "Describe the actions in this video."}
        ]
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], videos=["https://example.com/video.mp4"], return_tensors="pt")
inputs = inputs.to(model.device)

output_ids = model.generate(**inputs, max_new_tokens=512)
output_text = processor.batch_decode(output_ids, skip_special_tokens=True)
print(output_text[0])
```

### LLaMA-Factory

```bash
# LLaMA-Factory: Fine-tune multimodal models
pip install llamafactory

# Fine-tune Qwen2-VL on custom multimodal data
llamafactory-cli train \
    --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
    --dataset my_multimodal_data \
    --template qwen2_vl \
    --finetuning_type lora \
    --lora_rank 16 \
    --output_dir ./output/qwen2vl-finetuned \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 3 \
    --bf16 true
```

### LMStudio (Local Multimodal)

```python
# LMStudio: Run multimodal models locally
import requests

# Start LMStudio server locally
# Load a multimodal model in LMStudio UI

def query_local_multimodal(image_path, question):
    """Query local multimodal model via LMStudio."""
    
    import base64
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()
    
    response = requests.post(
        "http://localhost:1234/v1/chat/completions",
        json={
            "model": "local-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            }
                        },
                        {"type": "text", "text": question}
                    ]
                }
            ],
            "max_tokens": 1000
        }
    )
    
    return response.json()["choices"][0]["message"]["content"]
```

---

## Training Platforms

### Google Vertex AI

```python
# Google Vertex AI for multimodal training
from google.cloud import aiplatform

aiplatform.init(project="your-project", location="us-central1")

# Custom training job
job = aiplatform.CustomTrainingJob(
    display_name="multimodal-finetuning",
    script_path="train.py",
    container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1:latest",
    requirements=["transformers>=4.40.0", "peft>=0.10.0"],
)

model = job.run(
    replica_count=4,
    machine_type="a2-highgpu-8g",
    accelerator_type="NVIDIA_TESLA_A100",
    accelerator_count=4,
    args=[
        "--model", "Qwen/Qwen2-VL-7B-Instruct",
        "--dataset", "gs://your-bucket/multimodal-data",
        "--epochs", "3",
        "--batch_size", "4",
        "--learning_rate", "2e-5",
    ]
)
```

### AWS SageMaker

```python
# AWS SageMaker for multimodal training
import sagemaker
from sagemaker.huggingface import HuggingFace

estimator = HuggingFace(
    entry_point="train.py",
    source_dir="./src",
    instance_type="ml.p4d.24xlarge",
    instance_count=2,
    role=sagemaker.get_execution_role(),
    transformers_version="4.40.0",
    pytorch_version="2.2.0",
    hyperparameters={
        "model_name": "Qwen/Qwen2-VL-7B-Instruct",
        "epochs": 3,
        "batch_size": 4,
        "learning_rate": 2e-5,
        "lora_rank": 16,
    }
)

estimator.fit({"training": "s3://your-bucket/multimodal-data"})
```

### Open-Source Training with DeepSpeed

```python
# DeepSpeed ZeRO-3 for multimodal training
# ds_config.json
{
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu"},
        "offload_param": {"device": "nvme"},
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 100,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}

# Launch training
# deepspeed --num_gpus 8 train.py --deepspeed ds_config.json
```

---

## Deployment and Serving

### vLLM (High-Performance Serving)

```python
# vLLM for multimodal model serving
from vllm import LLM, SamplingParams

# Load multimodal model
llm = LLM(
    model="Qwen/Qwen2-VL-72B-Instruct",
    tensor_parallel_size=4,  # 4 GPUs
    max_model_len=32768,
    gpu_memory_utilization=0.9,
    enforce_eager=True,  # For stability
)

# Prepare multimodal inputs
from vllm import MultiModalDataDict

prompts = [
    {
        "prompt": "What is in this image?",
        "multi_modal_data": {
            "image": image_tensor
        }
    }
]

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=512,
    top_p=0.9
)

# Generate
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

### Ollama (Local Multimodal)

```bash
# Ollama: Run multimodal models locally
ollama pull llava:13b
ollama pull llava-phi3:latest
ollama pull moondream:latest

# Query via API
curl http://localhost:11434/api/generate -d '{
  "model": "llava:13b",
  "prompt": "What is in this image?",
  "images": ["<base64-encoded-image>"]
}'

# Or via Python
import ollama

response = ollama.chat(
    model="llava:13b",
    messages=[{
        "role": "user",
        "content": "What do you see?",
        "images": ["photo.jpg"]
    }]
)
print(response["message"]["content"])
```

### Triton Inference Server

```python
# Triton model configuration for multimodal
# model_repository/
# └── multimodal_model/
#     ├── config.pbtxt
#     └── 1/
#         └── model.py

# config.pbtxt
"""
name: "multimodal_model"
platform: "python"
max_batch_size: 8

input [
    {
        name: "TEXT_INPUT"
        data_type: TYPE_STRING
        dims: [1]
    },
    {
        name: "IMAGE_INPUT"
        data_type: TYPE_FP32
        dims: [3, 224, 224]
    }
]

output [
    {
        name: "TEXT_OUTPUT"
        data_type: TYPE_STRING
        dims: [1]
    }
]

instance_group [
    {
        count: 2
        kind: KIND_GPU
        gpus: [0, 1]
    }
]
"""
```

---

## Data and Evaluation Tools

### Data Preparation

```python
# LLaVA-style multimodal dataset preparation
import json
from PIL import Image

def create_multimodal_dataset(samples, output_path):
    """
    Create multimodal training dataset in LLaVA format.
    
    Sample format:
    {
        "id": "sample_001",
        "image": "path/to/image.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nWhat is this?"
            },
            {
                "from": "gpt",
                "value": "This is a cat."
            }
        ]
    }
    """
    dataset = []
    
    for sample in samples:
        entry = {
            "id": sample["id"],
            "image": sample["image_path"],
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\n{sample['question']}"
                },
                {
                    "from": "gpt",
                    "value": sample["answer"]
                }
            ]
        }
        dataset.append(entry)
    
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)
    
    return len(dataset)

# Example usage
samples = [
    {
        "id": "001",
        "image_path": "images/photo1.jpg",
        "question": "Describe this scene.",
        "answer": "A sunny beach with palm trees and blue water."
    }
]

count = create_multimodal_dataset(samples, "train_dataset.json")
print(f"Created {count} samples")
```

### Evaluation Frameworks

```python
# Comprehensive multimodal evaluation
import evaluate
from typing import List, Dict

class MultimodalEvaluator:
    """Evaluate multimodal models across multiple dimensions."""
    
    def __init__(self):
        self.metrics = {
            "accuracy": evaluate.load("accuracy"),
            "bleu": evaluate.load("bleu"),
            "rouge": evaluate.load("rouge"),
        }
    
    def evaluate_vqa(self, predictions: List[str], 
                     ground_truth: List[str]) -> Dict:
        """Evaluate Visual Question Answering."""
        accuracy = self.metrics["accuracy"].compute(
            predictions=[p.lower().strip() for p in predictions],
            references=[g.lower().strip() for g in ground_truth]
        )
        return {"vqa_accuracy": accuracy["accuracy"]}
    
    def evaluate_captioning(self, predictions: List[str],
                           references: List[str]) -> Dict:
        """Evaluate image/video captioning."""
        bleu = self.metrics["bleu"].compute(
            predictions=predictions,
            references=[[r] for r in references]
        )
        rouge = self.metrics["rouge"].compute(
            predictions=predictions,
            references=references
        )
        return {
            "bleu": bleu["bleu"],
            "rouge1": rouge["rouge1"],
            "rougeL": rouge["rougeL"]
        }
    
    def evaluate_multimodal_reasoning(self, 
                                       model_fn,
                                       test_cases: List[Dict]) -> Dict:
        """Evaluate complex multimodal reasoning."""
        correct = 0
        total = len(test_cases)
        
        for case in test_cases:
            prediction = model_fn(case["inputs"])
            
            if self._check_answer(prediction, case["expected"]):
                correct += 1
        
        return {
            "reasoning_accuracy": correct / total,
            "total_cases": total,
            "correct": correct
        }
    
    def _check_answer(self, prediction: str, expected: str) -> bool:
        """Flexible answer checking."""
        pred_lower = prediction.lower().strip()
        exp_lower = expected.lower().strip()
        
        # Exact match
        if pred_lower == exp_lower:
            return True
        
        # Contains expected answer
        if exp_lower in pred_lower:
            return True
        
        # Semantic similarity (simplified)
        return False
```

---

## Specialized Libraries

### OpenCV + Multimodal

```python
import cv2
import numpy as np

class MultimodalComputerVision:
    """Computer vision utilities for multimodal pipelines."""
    
    @staticmethod
    def extract_keyframes(video_path, method="interval", interval=30):
        """Extract keyframes from video for multimodal processing."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if method == "interval" and frame_count % interval == 0:
                frames.append(frame)
            elif method == "scene_change":
                if len(frames) > 0:
                    diff = cv2.absdiff(frames[-1], frame)
                    if np.mean(diff) > 25:  # Threshold
                        frames.append(frame)
                else:
                    frames.append(frame)
            
            frame_count += 1
        
        cap.release()
        return frames
    
    @staticmethod
    def preprocess_for_model(image, target_size=(224, 224)):
        """Standard preprocessing for vision models."""
        # Resize
        resized = cv2.resize(image, target_size)
        
        # Normalize
        normalized = resized / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (normalized - mean) / std
        
        # Transpose to CHW
        return normalized.transpose(2, 0, 1)
```

### Librosa + Multimodal

```python
import librosa
import numpy as np

class MultimodalAudio:
    """Audio utilities for multimodal pipelines."""
    
    @staticmethod
    def extract_features(audio_path, sr=16000):
        """Extract comprehensive audio features."""
        y, sr = librosa.load(audio_path, sr=sr)
        
        features = {
            "mel_spectrogram": librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=80
            ),
            "mfcc": librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13),
            "chroma": librosa.feature.chroma_stft(y=y, sr=sr),
            "tempo": librosa.beat.tempo(y=y, sr=sr),
            "spectral_centroid": librosa.feature.spectral_centroid(y=y, sr=sr),
        }
        
        return features
    
    @staticmethod
    def detect_speech_segments(audio_path, sr=16000):
        """Detect speech segments for selective processing."""
        y, sr = librosa.load(audio_path, sr=sr)
        
        # Energy-based voice activity detection
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)
        
        rms = librosa.feature.rms(
            y=y, frame_length=frame_length, hop_length=hop_length
        )[0]
        
        threshold = np.mean(rms) * 0.5
        speech_frames = rms > threshold
        
        return speech_frames
```

---

## Comparison Matrix

### API Providers

| Provider | Model | Vision | Audio | Video | Price (1M tokens) | Latency |
|----------|-------|--------|-------|-------|-------------------|---------|
| Google | Gemini 3 Pro | ✅ | ✅ | ✅ | $3.50/$10.50 | Fast |
| OpenAI | GPT-5 | ✅ | ✅ | ✅ | $5.00/$15.00 | Fast |
| Anthropic | Claude Opus 4 | ✅ | ❌ | ❌ | $15.00/$75.00 | Medium |
| Meta | Llama 4 | ✅ | ❌ | ✅ | Self-host | Variable |
| Qwen | Qwen2-VL-72B | ✅ | ✅ | ✅ | Self-host | Variable |

### Open-Source Models

| Model | Params | Vision | Audio | Video | License | VRAM Required |
|-------|--------|--------|-------|-------|---------|---------------|
| Qwen2-VL-72B | 72B MoE | ✅ | ✅ | ✅ | Apache 2.0 | 2x A100 80GB |
| Llama-4-Scout | 17B MoE | ✅ | ❌ | ✅ | Llama License | 1x A100 80GB |
| InternVL2.5 | 78B | ✅ | ❌ | ✅ | Apache 2.0 | 2x A100 80GB |
| Phi-4-multimodal | 14B | ✅ | ✅ | ✅ | MIT | 1x RTX 4090 |
| LLaVA-v1.6 | 7B | ✅ | ❌ | ❌ | Apache 2.0 | 1x RTX 3090 |

### Frameworks

| Framework | Purpose | Ease of Use | Performance | Community |
|-----------|---------|-------------|-------------|-----------|
| HuggingFace | All-in-one | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| vLLM | Serving | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Ollama | Local | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| LMStudio | Local | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| LLaMA-Factory | Fine-tuning | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## Getting Started Guide

### For Beginners

```bash
# Step 1: Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Step 2: Pull a multimodal model
ollama pull llava:7b

# Step 3: Try it
ollama run llava:7b

# Step 4: Python API
pip install ollama

python -c "
import ollama
response = ollama.chat(
    model='llava:7b',
    messages=[{
        'role': 'user',
        'content': 'What do you see?',
        'images': ['photo.jpg']
    }]
)
print(response['message']['content'])
"
```

### For Developers

```bash
# Step 1: Set up environment
pip install transformers torch pillow

# Step 2: Use HuggingFace Transformers
python -c "
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
from PIL import Image

model = LlavaForConditionalGeneration.from_pretrained(
    'llava-hf/llava-v1.6-mistral-7b-hf',
    torch_dtype=torch.float16,
    device_map='auto'
)
processor = AutoProcessor.from_pretrained('llava-hf/llava-v1.6-mistral-7b-hf')

image = Image.open('photo.jpg')
inputs = processor(text='What is this?', images=image, return_tensors='pt')
output = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(output[0], skip_special_tokens=True))
"
```

### For Enterprise

```bash
# Step 1: Deploy on Kubernetes
kubectl apply -f k8s-multimodal-deployment.yaml

# Step 2: Scale based on demand
kubectl autoscale deployment multimodal-ai --min=3 --max=20 --cpu-percent=70

# Step 3: Monitor
kubectl top pods -l app=multimodal-ai

# Step 4: Set up monitoring dashboard
# Grafana + Prometheus for GPU metrics, latency, throughput
```

---

## Key Takeaways

1. **Google Gemini** leads in multimodal breadth (text + image + audio + video)
2. **HuggingFace Transformers** is the universal framework for multimodal development
3. **vLLM** is the go-to for high-performance multimodal serving
4. **Ollama/LMStudio** make local multimodal AI accessible
5. **Qwen2-VL** leads open-source multimodal models
6. **Evaluation** remains fragmented — combine multiple benchmarks

---

*See also: [01-Overview.md](./01-Overview.md) for context, [03-Technical-Deep-Dive.md](./03-Technical-Deep-Dive.md) for implementation patterns*

*Last updated: July 4, 2026*
