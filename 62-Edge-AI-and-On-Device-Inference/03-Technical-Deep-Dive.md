# Technical Deep Dive: Edge AI and On-Device Inference

> This document provides detailed technical analysis of advanced topics in Edge AI, including cutting-edge optimization techniques, hardware-software co-design, and production deployment patterns.

## Table of Contents

- [Advanced Model Compression](#advanced-model-compression)
- [Hardware-Software Co-Design](#hardware-software-co-design)
- [Efficient Attention Mechanisms](#efficient-attention-mechanisms)
- [On-Device Training and Adaptation](#on-device-training-and-adaptation)
- [Multi-Model Orchestration at the Edge](#multi-model-orchestration-at-the-edge)
- [Real-Time Vision Pipelines](#real-time-vision-pipelines)
- [Edge LLM Optimization](#edge-llm-optimization)
- [Production Patterns](#production-patterns)

---

## Advanced Model Compression

### Mixed-Precision Quantization

Not all layers need the same precision. Mixed-precision quantization assigns different bit-widths to different layers based on their sensitivity:

```python
import torch
from torch.ao.quantization import (
    get_default_qconfig,
    QConfigMapping,
    prepare,
    convert
)

def mixed_precision_quantization(model, calibration_data):
    """Apply different quantization levels to different layers"""
    
    # Define per-layer quantization config
    qconfig_mapping = QConfigMapping()
    
    # Sensitive layers: keep FP16
    qconfig_mapping.set_module_name("encoder.layers.0", None)  # No quantization
    qconfig_mapping.set_module_name("encoder.layers.1", None)
    
    # Normal layers: INT8
    qconfig_mapping.set_module_name("encoder.layers.2", get_default_qconfig('x86'))
    qconfig_mapping.set_module_name("encoder.layers.3", get_default_qconfig('x86'))
    
    # Less critical layers: INT4
    qconfig_mapping.set_module_name("decoder.layers.0", 
        get_default_qconfig('x86').with_dtype(torch.qint4))
    
    # Apply quantization
    model.eval()
    model_prepared = prepare(model, qconfig_mapping)
    
    # Calibrate
    with torch.no_grad():
        for batch in calibration_data:
            model_prepared(batch)
    
    # Convert
    quantized_model = convert(model_prepared)
    
    return quantized_model
```

### Sensitivity Analysis

Determining which layers can be aggressively quantized:

```python
def sensitivity_analysis(model, calibration_data, bit_widths=[8, 4, 2]):
    """Analyze sensitivity of each layer to quantization"""
    results = {}
    
    # Get baseline accuracy
    baseline_accuracy = evaluate(model, calibration_data)
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            layer_results = {}
            
            for bits in bit_widths:
                # Quantize only this layer
                quantized_model = quantize_single_layer(model, name, bits)
                
                # Evaluate
                accuracy = evaluate(quantized_model, calibration_data)
                accuracy_drop = baseline_accuracy - accuracy
                
                layer_results[bits] = {
                    "accuracy": accuracy,
                    "drop": accuracy_drop,
                    "size_reduction": calculate_size_reduction(bits),
                }
            
            results[name] = layer_results
    
    # Find optimal configuration
    optimal_config = find_optimal_config(results, target_accuracy=baseline_accuracy * 0.99)
    
    return results, optimal_config
```

### Structured Sparsity

NVIDIA Ampere and newer support 2:4 structured sparsity (2 out of every 4 weights are zero):

```python
import torch.nn.utils.prune as prune

def apply_2_4_sparsity(model):
    """Apply 2:4 structured sparsity for NVIDIA Ampere+"""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Apply 2:4 sparsity
            prune.sparse_dim(
                module,
                name='weight',
                sparsity_dim=0,
                sparsity=0.5  # 50% sparsity (2:4 pattern)
            )
            
            # Verify sparsity pattern
            weight = module.weight.data
            for i in range(0, weight.shape[0], 4):
                group = weight[i:i+4]
                zeros = (group == 0).sum().item()
                assert zeros == 2, f"Expected 2 zeros, got {zeros}"

# With 2:4 sparsity + INT8, you get:
# - 2x speedup from INT8
# - Additional 2x from sparse tensor cores
# - Total: ~4x speedup over FP32
```

### Low-Rank Adaptation (LoRA) for Edge

Using LoRA to efficiently adapt models on edge devices:

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """LoRA layer for efficient edge fine-tuning"""
    def __init__(self, original_layer, rank=8, alpha=16):
        super().__init__()
        self.original = original_layer
        self.original.weight.requires_grad = False
        
        d_in = original_layer.in_features
        d_out = original_layer.out_features
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.randn(d_in, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, d_out))
        self.scaling = alpha / rank
    
    def forward(self, x):
        # Original forward pass + LoRA adaptation
        original_output = self.original(x)
        lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling
        return original_output + lora_output

# Apply LoRA to specific layers
def apply_lora(model, target_modules=["q_proj", "v_proj"], rank=8):
    """Apply LoRA to model for edge fine-tuning"""
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                lora_layer = LoRALinear(module, rank=rank)
                # Replace in model
                set_module_by_name(model, name, lora_layer)
    
    # Only train LoRA parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
```

---

## Hardware-Software Co-Design

### Tensor Core Utilization

NVIDIA Tensor Cores provide massive speedups for specific operations:

```python
# Ensure operations are Tensor Core compatible
import torch

# Use tensor core-friendly dtypes
# Good: FP16, BF16, INT8, TF32
# Bad: FP32 (no tensor core acceleration)

# Enable TF32 for Ampere+ (default in PyTorch 1.12+)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast():  # Automatically uses tensor cores
        output = model(batch)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# Tensor Core utilization checklist:
# ✓ Use FP16/BF16 activations
# ✓ Matrix dimensions divisible by 8 (FP16) or 16 (INT8)
# ✓ Batch sizes that maximize occupancy
# ✓ Avoid operations that break tensor core patterns
```

### Memory Layout Optimization

Different hardware prefers different memory layouts:

```python
# NCHW vs NHWC for vision models
# NCHW: PyTorch default, good for CUDA
# NHWC: TensorFlow/Edge TPU default, good for ARM

import torch

# Convert model to NHWC for edge deployment
def convert_to_nhwc(model):
    """Convert model from NCHW to NHWC layout"""
    # For PyTorch (channels_last memory format)
    model = model.to(memory_format=torch.channels_last)
    
    # Verify
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() == 4:
            assert param.is_contiguous(memory_format=torch.channels_last), \
                f"Weight {name} not in NHWC format"
    
    return model

# For deployment, convert weights
def convert_weights_to_nhwc(state_dict):
    """Convert weight tensors from NCHW to NHWC"""
    new_state_dict = {}
    for key, value in state_dict.items():
        if value.dim() == 4:  # Conv weights: [out_ch, in_ch, H, W]
            # Convert to [out_ch, H, W, in_ch]
            new_state_dict[key] = value.permute(0, 2, 3, 1).contiguous()
        else:
            new_state_dict[key] = value
    return new_state_dict
```

### Operator Fusion

Combining multiple operations into a single kernel:

```python
# Common operator fusions for edge deployment

# Fusion patterns:
# 1. Conv + BatchNorm + ReLU → Single fused kernel
# 2. MatMul + Bias + GELU → Single fused kernel
# 3. LayerNorm + Residual → Single fused kernel

# TensorRT handles this automatically, but for custom deployment:

# Using torch.compile (PyTorch 2.0+)
compiled_model = torch.compile(
    model,
    backend="inductor",  # Or "tensorrt" for NVIDIA
    options={
        "max_autotune": True,
        "triton.cudagraphs": True,
    }
)

# Manual operator fusion with torch.jit
@torch.jit.script
def fused_conv_bn_relu(conv_weight, conv_bias, bn_weight, bn_bias, 
                       bn_mean, bn_var, x):
    """Fused Conv + BatchNorm + ReLU"""
    # Conv
    x = torch.nn.functional.conv2d(x, conv_weight, conv_bias)
    # BatchNorm (fused)
    scale = bn_weight / torch.sqrt(bn_var + 1e-5)
    x = scale * (x - bn_mean) + bn_bias
    # ReLU
    x = torch.relu(x)
    return x
```

### Neon-SIMD Optimization (ARM)

ARM NEON instructions for edge inference:

```python
# Using TVM with NEON autotuning
import tvm
from tvm import autotvm

@autotvm.template("conv2d_neon")
def conv2d_neon(cfg, data, kernel, stride, padding):
    """Optimized Conv2D using ARM NEON instructions"""
    # Define search space for NEON optimization
    cfg.define_split("tile_y", 64, num_outputs=3)
    cfg.define_split("tile_x", 64, num_outputs=3)
    cfg.define_knob("unroll", [True, False])
    
    # NEON-optimized computation
    # ... (TVM schedule definition)
    
    return schedule

# Compile with NEON optimization
target = tvm.target.Target("llvm -mcpu=cortex-a78 -mattr=+neon")
with autotvm.apply_history_best("neon_tuning_log.json"):
    lib = tvm.relay.build(model, target=target)
```

---

## Efficient Attention Mechanisms

### Flash Attention for Edge

Optimized attention that reduces memory and improves speed:

```python
# Flash Attention 2 for edge devices
from flash_attn import flash_attn_func

def efficient_attention(Q, K, V, causal=True):
    """Memory-efficient attention using Flash Attention"""
    # Q, K, V shape: [batch, seq_len, heads, head_dim]
    
    # Flash Attention computes in chunks, avoiding O(n²) memory
    output = flash_attn_func(
        Q, K, V,
        causal=causal,
        window_size=(-1, -1),  # Full attention
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
    )
    
    return output

# For very long sequences on edge, use sliding window attention
def sliding_window_attention(Q, K, V, window_size=1024):
    """Sliding window attention for memory-constrained devices"""
    from flash_attn import flash_attn_with_kvcache
    
    # Process in windows
    output = flash_attn_with_kvcache(
        Q, K, V,
        cache_seqlens=None,
        cache_batch_idx=None,
        cache_leftpad=None,
        block_table=None,
        softmax_scale=None,
        causal=True,
        window_size=(window_size, window_size),
    )
    
    return output
```

### Grouped Query Attention (GQA)

Reduces KV cache size for edge LLM deployment:

```python
import torch
import torch.nn as nn
import math

class GroupedQueryAttention(nn.Module):
    """GQA: Fewer KV heads = smaller KV cache = faster edge inference"""
    
    def __init__(self, embed_dim, num_heads, num_kv_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_queries_per_kv = num_heads // num_kv_heads
        
        self.q_proj = nn.Linear(embed_dim, num_heads * head_dim)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * head_dim)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * head_dim)
        self.o_proj = nn.Linear(num_heads * head_dim, embed_dim)
    
    def forward(self, x, kv_cache=None):
        batch, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        
        # Update KV cache
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=1)
            v = torch.cat([kv_cache[1], v], dim=1)
        
        # Expand KV heads to match Q heads
        k = k.repeat_interleave(self.num_queries_per_kv, dim=2)
        v = v.repeat_interleave(self.num_queries_per_kv, dim=2)
        
        # Compute attention
        attn = torch.matmul(q.transpose(1, 2), k.transpose(1, 2))
        attn = attn / math.sqrt(self.head_dim)
        attn = torch.softmax(attn, dim=-1)
        
        output = torch.matmul(attn, v.transpose(1, 2))
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        
        return self.o_proj(output), (k, v)

# KV Cache optimization
# For 7B model with 32 layers, GQA (8 KV heads vs 32 Q heads):
# - Full MHA: 32 layers × 32 KV heads × 2048 tokens × 128 dims × 2 bytes = 1GB
# - GQA:      32 layers × 8 KV heads × 2048 tokens × 128 dims × 2 bytes = 256MB
```

### Speculative Decoding for Edge LLMs

Using a small draft model to speed up generation:

```python
class SpeculativeDecoder:
    """Speed up edge LLM inference with draft model"""
    
    def __init__(self, target_model, draft_model, draft_length=4):
        self.target = target_model  # Large, accurate model
        self.draft = draft_model    # Small, fast model
        self.draft_length = draft_length
    
    def generate(self, prompt_tokens, max_tokens):
        """Generate with speculative decoding"""
        tokens = prompt_tokens
        generated = []
        
        while len(generated) < max_tokens:
            # Draft phase: generate draft_length tokens with small model
            draft_tokens = []
            draft_probs = []
            
            for _ in range(self.draft_length):
                logits = self.draft(tokens + draft_tokens)
                prob = torch.softmax(logits[:, -1], dim=-1)
                token = torch.multinomial(prob, 1)
                draft_tokens.append(token)
                draft_probs.append(prob)
            
            # Target phase: verify all draft tokens at once
            target_logits = self.target(tokens + draft_tokens)
            
            # Accept/reject each draft token
            accepted = 0
            for i, (draft_token, draft_prob) in enumerate(zip(draft_tokens, draft_probs)):
                target_prob = torch.softmax(target_logits[:, i], dim=-1)
                draft_p = draft_prob[0, draft_token]
                target_p = target_prob[0, draft_token]
                
                # Acceptance criterion
                if torch.rand(1) < min(1, target_p / draft_p):
                    accepted += 1
                    generated.append(draft_token)
                else:
                    # Resample from corrected distribution
                    corrected = torch.clamp(target_prob - draft_prob, min=0)
                    corrected = corrected / corrected.sum()
                    new_token = torch.multinomial(corrected, 1)
                    generated.append(new_token)
                    break
            
            tokens = tokens + generated
        
        return generated

# Performance: ~2-3x speedup with minimal quality loss
# Target model: 7B params, 50ms/token
# Draft model: 1B params, 10ms/token
# Without speculative: 50ms/token
# With speculative (4 draft): ~20ms/token (2.5x speedup)
```

---

## On-Device Training and Adaptation

### Federated Learning on Edge

Training across edge devices without centralizing data:

```python
class FederatedEdgeTrainer:
    """Federated learning for edge devices"""
    
    def __init__(self, model, device_id, server_url):
        self.model = model
        self.device_id = device_id
        self.server = server_url
    
    def local_train(self, local_data, epochs=5, lr=0.001):
        """Train on local data only"""
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(epochs):
            for batch in local_data:
                optimizer.zero_grad()
                output = self.model(batch.inputs)
                loss = criterion(output, batch.labels)
                loss.backward()
                optimizer.step()
        
        # Return model update (delta), not full model
        return self.compute_update()
    
    def compute_update(self):
        """Compute difference from global model"""
        # This is what gets sent to the server
        update = {}
        for name, param in self.model.named_parameters():
            update[name] = param.data - self.global_params[name]
        return update
    
    def aggregate_updates(self, updates):
        """FedAvg aggregation"""
        aggregated = {}
        for name in updates[0].keys():
            # Weighted average of updates
            weighted_sum = torch.zeros_like(updates[0][name])
            total_weight = 0
            
            for update, weight in zip(updates, self.client_weights):
                weighted_sum += update[name] * weight
                total_weight += weight
            
            aggregated[name] = weighted_sum / total_weight
        
        return aggregated
```

### On-Device Fine-Tuning

Lightweight fine-tuning on edge devices:

```python
class OnDeviceFineTuner:
    """Fine-tune model on device with limited resources"""
    
    def __init__(self, model, max_memory_mb=512):
        self.model = model
        self.max_memory = max_memory_mb
    
    def adapter_finetuning(self, train_data, num_steps=100):
        """Fine-tune only adapter layers"""
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Add small adapter layers
        adapters = self.add_adapters(rank=8)
        
        # Only train adapters
        optimizer = torch.optim.Adam(
            [p for p in adapters.parameters() if p.requires_grad],
            lr=1e-4
        )
        
        # Gradient checkpointing to save memory
        for step in range(num_steps):
            batch = train_data[step % len(train_data)]
            
            with torch.cuda.amp.autocast():
                output = self.model(batch.inputs)
                loss = self.criterion(output, batch.labels)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Check memory usage
            if self.get_memory_usage() > self.max_memory:
                print(f"Memory limit reached at step {step}")
                break
    
    def add_adapters(self, rank=8):
        """Add LoRA adapters to model"""
        # Implementation similar to LoRA above
        pass
```

### Continual Learning at the Edge

Adapting to new data distributions over time:

```python
class ContinualEdgeLearner:
    """Learn from streaming data on edge devices"""
    
    def __init__(self, model, memory_buffer_size=1000):
        self.model = model
        self.memory_buffer = []  # Replay buffer
        self.buffer_size = memory_buffer_size
        self.task_id = 0
    
    def learn_new_task(self, new_data, epochs=3):
        """Learn new task while preserving old knowledge"""
        # Sample from memory buffer (old data)
        old_samples = self.sample_buffer(batch_size=len(new_data) // 4)
        
        # Combine old and new data
        combined_data = new_data + old_samples
        
        # Train with experience replay
        for epoch in range(epochs):
            for batch in combined_data:
                loss = self.compute_loss(batch)
                
                # Elastic Weight Consolidation (EWC) penalty
                ewc_loss = self.ewc_penalty()
                
                total_loss = loss + 0.1 * ewc_loss
                total_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        # Update memory buffer
        self.update_buffer(new_data)
        self.task_id += 1
    
    def ewc_penalty(self):
        """Prevent catastrophic forgetting"""
        penalty = 0
        for name, param in self.model.named_parameters():
            if name in self.fisher_information:
                fisher = self.fisher_information[name]
                optimal = self.optimal_params[name]
                penalty += (fisher * (param - optimal) ** 2).sum()
        return penalty
    
    def update_buffer(self, new_data):
        """Update replay buffer with reservoir sampling"""
        for sample in new_data:
            if len(self.memory_buffer) < self.buffer_size:
                self.memory_buffer.append(sample)
            else:
                # Reservoir sampling
                idx = random.randint(0, len(self.memory_buffer))
                if idx < self.buffer_size:
                    self.memory_buffer[idx] = sample
```

---

## Multi-Model Orchestration at the Edge

### Model Cascade

Running different models based on input complexity:

```python
class ModelCascade:
    """Run lightweight model first, escalate to heavier model if needed"""
    
    def __init__(self):
        self.models = [
            {"model": load_model("tiny.onnx"), "threshold": 0.5, "name": "tiny"},
            {"model": load_model("small.onnx"), "threshold": 0.7, "name": "small"},
            {"model": load_model("large.onnx"), "threshold": 0.9, "name": "large"},
        ]
    
    def predict(self, input_data):
        """Cascade through models until confidence is sufficient"""
        for model_info in self.models:
            model = model_info["model"]
            threshold = model_info["threshold"]
            
            output = model.predict(input_data)
            confidence = output.max()
            
            if confidence >= threshold:
                return {
                    "prediction": output.argmax(),
                    "confidence": confidence,
                    "model_used": model_info["name"],
                    "latency_ms": model_info["latency"],
                }
        
        # Fallback to largest model
        return self.models[-1]["model"].predict(input_data)

# Performance:
# - 70% of inputs handled by tiny model (1ms)
# - 20% escalated to small model (5ms)
# - 10% escalated to large model (20ms)
# - Average latency: 0.7*1 + 0.2*5 + 0.1*20 = 3.7ms
```

### Model Ensemble at Edge

Running multiple models in parallel:

```python
class EdgeEnsemble:
    """Ensemble of models for robust edge inference"""
    
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
    
    def predict(self, input_data):
        """Weighted ensemble prediction"""
        predictions = []
        
        for model in self.models:
            output = model.predict(input_data)
            predictions.append(output)
        
        # Weighted average
        ensemble_output = torch.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            ensemble_output += pred * weight
        
        return ensemble_output
    
    def predict_with_uncertainty(self, input_data):
        """Prediction with uncertainty estimation"""
        predictions = []
        
        for model in self.models:
            output = model.predict(input_data)
            predictions.append(output)
        
        # Mean prediction
        mean_pred = torch.stack(predictions).mean(dim=0)
        
        # Uncertainty (variance across models)
        uncertainty = torch.stack(predictions).var(dim=0).mean()
        
        return {
            "prediction": mean_pred,
            "uncertainty": uncertainty,
            "model_agreement": self.calculate_agreement(predictions),
        }
```

---

## Real-Time Vision Pipelines

### Object Detection Pipeline

```python
class RealTimeObjectDetection:
    """Optimized vision pipeline for edge devices"""
    
    def __init__(self, model_path, input_size=(640, 640)):
        self.model = load_model(model_path)
        self.input_size = input_size
        self.preprocessor = Preprocessor(input_size)
        self.postprocessor = Postprocessor(conf_threshold=0.5)
        
        # Pipeline stages
        self.capture = VideoCapture(camera_id=0)
        self.display = Display()
    
    def run(self):
        """Main detection loop"""
        while True:
            # Capture frame
            frame = self.capture.read()
            
            # Preprocess
            input_tensor = self.preprocessor.process(frame)
            
            # Inference
            raw_output = self.model.predict(input_tensor)
            
            # Postprocess
            detections = self.postprocessor.process(raw_output, frame.shape)
            
            # Render
            self.display.draw_detections(frame, detections)
            
            # Calculate FPS
            self.display.draw_fps(self.get_fps())
    
    def optimize_pipeline(self):
        """Apply pipeline optimizations"""
        # 1. Use async capture (capture while processing)
        self.capture.set_async(True)
        
        # 2. Use pipeline parallelism
        self.pipeline = Pipeline([
            Stage("capture", self.capture.read, parallel=True),
            Stage("preprocess", self.preprocessor.process),
            Stage("inference", self.model.predict),
            Stage("postprocess", self.postprocessor.process),
            Stage("display", self.display.draw),
        ])
        
        # 3. Use INT8 quantized model
        self.model = quantize_model(self.model, dtype="int8")
        
        # 4. Use TensorRT for NVIDIA devices
        if self.is_nvidia_device():
            self.model = convert_to_tensorrt(self.model)
```

### Semantic Segmentation

```python
class EdgeSemanticSegmentation:
    """Real-time semantic segmentation on edge"""
    
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.color_map = self.load_color_map()
    
    def segment_frame(self, frame):
        """Segment a single frame"""
        # Preprocess
        input_tensor = self.preprocess(frame)
        
        # Inference
        logits = self.model.predict(input_tensor)
        
        # Postprocess
        segmentation = torch.argmax(logits, dim=1)
        
        # Resize to original frame size
        segmentation = torch.nn.functional.interpolate(
            segmentation.float().unsqueeze(1),
            size=frame.shape[:2],
            mode='nearest'
        ).squeeze(1).long()
        
        # Apply color map
        colored = self.color_map[segmentation]
        
        return colored, segmentation
    
    def optimize_for_realtime(self):
        """Optimize for real-time performance"""
        # Use TensorRT
        self.model = convert_to_tensorrt(self.model, precision="fp16")
        
        # Use sliding window for large images
        self.window_size = 512
        self.stride = 384  # 75% overlap
        
        # Use temporal consistency
        self.prev_segmentation = None
    
    def segment_with_temporal_consistency(self, frame):
        """Use previous frame to speed up current segmentation"""
        if self.prev_segmentation is not None:
            # Only re-segment regions that likely changed
            change_mask = self.detect_changes(frame, self.prev_frame)
            
            if change_mask.sum() < 0.1 * change_mask.size:
                # Less than 10% changed, reuse previous result
                return self.prev_segmentation
        
        # Full segmentation
        result = self.segment_frame(frame)
        self.prev_segmentation = result
        self.prev_frame = frame
        
        return result
```

---

## Edge LLM Optimization

### KV Cache Optimization

```python
class OptimizedKVCache:
    """Memory-efficient KV cache for edge LLMs"""
    
    def __init__(self, num_layers, num_heads, head_dim, max_seq_len, dtype=torch.float16):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Allocate cache in pinned memory for fast transfers
        self.k_cache = torch.zeros(
            num_layers, 1, num_heads, max_seq_len, head_dim,
            dtype=dtype, pin_memory=True
        )
        self.v_cache = torch.zeros(
            num_layers, 1, num_heads, max_seq_len, head_dim,
            dtype=dtype, pin_memory=True
        )
        
        self.current_len = 0
    
    def update(self, layer_idx, k_new, v_new):
        """Update KV cache for a layer"""
        new_len = k_new.shape[2]
        
        self.k_cache[layer_idx, :, :, self.current_len:self.current_len+new_len, :] = k_new
        self.v_cache[layer_idx, :, :, self.current_len:self.current_len+new_len, :] = v_new
    
    def get_kv(self, layer_idx):
        """Get cached KV for a layer"""
        return (
            self.k_cache[layer_idx, :, :, :self.current_len+1, :],
            self.v_cache[layer_idx, :, :, :self.current_len+1, :]
        )
    
    def compress(self, strategy="sliding_window", window_size=2048):
        """Compress KV cache to save memory"""
        if strategy == "sliding_window":
            # Keep only recent window
            if self.current_len > window_size:
                self.k_cache = self.k_cache[:, :, :, -window_size:, :]
                self.v_cache = self.v_cache[:, :, :, -window_size:, :]
                self.current_len = window_size
        
        elif strategy == "token_eviction":
            # Evict tokens based on attention scores
            self.evict_low_attention_tokens()
        
        elif strategy == "quantization":
            # Quantize cached KV
            self.k_cache = self.k_cache.to(torch.int8)
            self.v_cache = self.v_cache.to(torch.int8)
```

### Continuous Batching

```python
class ContinuousBatcher:
    """Maximize throughput with continuous batching"""
    
    def __init__(self, model, max_batch_size=8, max_tokens=2048):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_tokens = max_tokens
        self.requests = []
    
    def add_request(self, request):
        """Add new request to batch"""
        self.requests.append({
            "tokens": request.tokens,
            "generated": [],
            "finished": False,
        })
    
    def step(self):
        """Run one step of continuous batching"""
        # Filter finished requests
        active = [r for r in self.requests if not r["finished"]]
        
        if not active:
            return []
        
        # Pad to same length for batching
        max_len = max(len(r["tokens"]) + len(r["generated"]) for r in active)
        
        # Create batch
        batch_input = []
        for req in active:
            tokens = req["tokens"] + req["generated"]
            # Pad
            tokens = tokens + [0] * (max_len - len(tokens))
            batch_input.append(tokens)
        
        batch_tensor = torch.tensor(batch_input)
        
        # Run inference
        outputs = self.model.predict(batch_tensor)
        
        # Process outputs
        results = []
        for i, req in enumerate(active):
            next_token = outputs[i, len(req["tokens"])+len(req["generated"])-1].argmax()
            
            if next_token == EOS_TOKEN:
                req["finished"] = True
                results.append({
                    "tokens": req["generated"],
                    "finished": True,
                })
            else:
                req["generated"].append(next_token.item())
                results.append({
                    "token": next_token.item(),
                    "finished": False,
                })
        
        # Remove finished requests
        self.requests = [r for r in self.requests if not r["finished"]]
        
        return results
```

---

## Production Patterns

### A/B Testing at the Edge

```python
class EdgeABTest:
    """A/B test different model versions at the edge"""
    
    def __init__(self, model_a, model_b, traffic_split=0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.traffic_split = traffic_split
        self.metrics = {"a": [], "b": []}
    
    def predict(self, input_data):
        """Route to model based on traffic split"""
        if random.random() < self.traffic_split:
            # Model A
            output = self.model_a.predict(input_data)
            self.record_metric("a", input_data, output)
        else:
            # Model B
            output = self.model_b.predict(input_data)
            self.record_metric("b", input_data, output)
        
        return output
    
    def record_metric(self, variant, input_data, output):
        """Record metrics for analysis"""
        self.metrics[variant].append({
            "latency_ms": output.latency,
            "confidence": output.confidence,
            "correct": self.verify_correctness(output),
        })
    
    def analyze_results(self):
        """Analyze A/B test results"""
        results = {}
        for variant in ["a", "b"]:
            metrics = self.metrics[variant]
            results[variant] = {
                "count": len(metrics),
                "avg_latency": np.mean([m["latency_ms"] for m in metrics]),
                "avg_confidence": np.mean([m["confidence"] for m in metrics]),
                "accuracy": np.mean([m["correct"] for m in metrics]),
            }
        
        # Statistical significance test
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(
            [m["latency_ms"] for m in self.metrics["a"]],
            [m["latency_ms"] for m in self.metrics["b"]]
        )
        
        results["statistical_significance"] = p_value < 0.05
        
        return results
```

### Model Versioning and Rollback

```python
class ModelVersionManager:
    """Manage model versions at the edge"""
    
    def __init__(self, storage_path):
        self.storage_path = storage_path
        self.versions = self.load_version_index()
    
    def deploy_version(self, model_path, version, metadata):
        """Deploy a new model version"""
        # Store model
        version_path = os.path.join(self.storage_path, f"v{version}")
        os.makedirs(version_path, exist_ok=True)
        
        # Copy model
        shutil.copy(model_path, os.path.join(version_path, "model.bin"))
        
        # Store metadata
        with open(os.path.join(version_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)
        
        # Update index
        self.versions[version] = {
            "path": version_path,
            "deployed_at": datetime.now(),
            "metadata": metadata,
        }
        self.save_version_index()
    
    def rollback(self, target_version=None):
        """Rollback to previous version"""
        if target_version is None:
            # Rollback to previous version
            sorted_versions = sorted(self.versions.keys(), reverse=True)
            if len(sorted_versions) < 2:
                raise ValueError("No previous version to rollback to")
            target_version = sorted_versions[1]
        
        # Load target version
        version_info = self.versions[target_version]
        model = load_model(os.path.join(version_info["path"], "model.bin"))
        
        # Set as active
        self.active_version = target_version
        
        return model
    
    def get_version_diff(self, v1, v2):
        """Compare two versions"""
        meta1 = self.versions[v1]["metadata"]
        meta2 = self.versions[v2]["metadata"]
        
        return {
            "accuracy_change": meta2.get("accuracy", 0) - meta1.get("accuracy", 0),
            "size_change": meta2.get("size_mb", 0) - meta1.get("size_mb", 0),
            "latency_change": meta2.get("latency_ms", 0) - meta1.get("latency_ms", 0),
        }
```

### Health Monitoring

```python
class EdgeHealthMonitor:
    """Monitor health of edge AI deployment"""
    
    def __init__(self, device_id, alert_callback):
        self.device_id = device_id
        self.alert_callback = alert_callback
        self.metrics = deque(maxlen=10000)
        self.baselines = {}
    
    def record_inference(self, input_data, output, latency_ms):
        """Record inference metrics"""
        metric = {
            "timestamp": time.time(),
            "latency_ms": latency_ms,
            "confidence": output.confidence,
            "input_hash": hash(input_data.tobytes()),
            "output_hash": hash(output.data.tobytes()),
        }
        self.metrics.append(metric)
        
        # Check for anomalies
        self.check_anomalies(metric)
    
    def check_anomalies(self, metric):
        """Detect anomalies in metrics"""
        # Latency anomaly
        if len(self.metrics) > 100:
            latencies = [m["latency_ms"] for m in list(self.metrics)[-100:]]
            mean_latency = np.mean(latencies)
            std_latency = np.std(latencies)
            
            if metric["latency_ms"] > mean_latency + 3 * std_latency:
                self.alert("latency_spike", metric)
        
        # Confidence degradation
        if len(self.metrics) > 100:
            confidences = [m["confidence"] for m in list(self.metrics)[-100:]]
            mean_confidence = np.mean(confidences)
            
            if metric["confidence"] < mean_confidence * 0.8:
                self.alert("confidence_drop", metric)
    
    def alert(self, alert_type, metric):
        """Send alert"""
        self.alert_callback({
            "device_id": self.device_id,
            "type": alert_type,
            "metric": metric,
            "timestamp": time.time(),
        })
    
    def get_health_report(self):
        """Generate health report"""
        latencies = [m["latency_ms"] for m in self.metrics]
        confidences = [m["confidence"] for m in self.metrics]
        
        return {
            "device_id": self.device_id,
            "total_inferences": len(self.metrics),
            "avg_latency_ms": np.mean(latencies),
            "p95_latency_ms": np.percentile(latencies, 95),
            "avg_confidence": np.mean(confidences),
            "uptime_hours": (time.time() - self.metrics[0]["timestamp"]) / 3600,
            "error_rate": self.calculate_error_rate(),
        }
```

---

## Cross-References

- **02-LLMs**: Large language models being optimized for edge
- **29-Reasoning-and-Inference-Scaling**: Inference scaling techniques
- **30-Small-Language-Models**: Small models for edge deployment
- **36-Long-Context-AI**: Long context handling on edge
- **38-AI-Supply-Chain-and-Chip-Design**: Hardware ecosystem
- **56-MLOps-and-AI-Platform-Engineering**: Production deployment patterns
- **60-Physical-AI-and-Embodied-Intelligence**: Edge AI for robotics

---

*Last updated: July 7, 2026*
*Category: 62-Edge-AI-and-On-Device-Inference*
